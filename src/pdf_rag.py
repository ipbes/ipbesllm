import asyncio
import os
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from loguru import logger

from cache import CACHE_VERSION, cache, make_key
# infer_country_names is re-exported for app.py.
from geo import infer_country_names  # noqa: F401
from rag_utils import EMBED_MODEL, MAX_CONTEXT_CHARS, NUM_CTX, chat, embed_query
from thesaurus_helper import get_thesaurus
from settings import CHROMA_DIR


LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

COLLECTION_NAME = "pdf_documents"

# Written by pdf_index.py; its modification time marks "the index changed".
MANIFEST_PATH = Path(CHROMA_DIR) / "pdf_index.json"

# Cache TTLs (seconds)
RETRIEVE_TTL = 60 * 60        # 1 hour
ANSWER_TTL = 60 * 60 * 6      # 6 hours

# Thesaurus re-ranking: fetch RERANK_OVERFETCH x k candidates from Chroma,
# re-rank them, keep the best k. Weight of the thesaurus signal:
# 0.0 = pure vector search; 0.25 = gentle nudge. Same variable as ttl_rag.py.
THESAURUS_RERANK_ALPHA = float(os.getenv("THESAURUS_RERANK_ALPHA", "0.25"))
RERANK_OVERFETCH = max(1, int(os.getenv("PDF_RERANK_OVERFETCH", "3")))

# Part of the cache keys: bump when retrieval or prompts change.
RETRIEVAL_VERSION = "2"
PROMPT_VERSION = "2"


# ---------------------------------------------------------------------------
# Index state / collection
# ---------------------------------------------------------------------------
# The collection handle is cached, and dropped whenever pdf_index.py rewrites
# its manifest (start and end of every run), so a long-lived process never
# keeps a handle to a collection that was rebuilt.

_STATE: dict = {"stamp": None, "collection": None}


def _index_stamp() -> int:
    try:
        return MANIFEST_PATH.stat().st_mtime_ns
    except OSError:
        return 0


def _get_collection():
    stamp = _index_stamp()
    if _STATE["stamp"] != stamp:
        _STATE.update(stamp=stamp, collection=None)
        if stamp:
            _check_manifest()

    if _STATE["collection"] is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            # No embedding function: queries are embedded by embed_query and
            # passed as vectors, so the collection never embeds anything.
            _STATE["collection"] = client.get_collection(name=COLLECTION_NAME)
        except NotFoundError as exc:
            raise SystemExit(
                f"Collection '{COLLECTION_NAME}' not found in '{CHROMA_DIR}'. "
                f"Run pdf_index.py first. (Original error: {exc})"
            )
        except ValueError as exc:
            if "does not exist" in str(exc).lower():
                raise SystemExit(
                    f"Collection '{COLLECTION_NAME}' not found in '{CHROMA_DIR}'. "
                    f"Run pdf_index.py first. (Original error: {exc})"
                )
            raise
    return _STATE["collection"]


def _check_manifest() -> None:
    import json
    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except (OSError, ValueError):
        return
    if data.get("complete") is False:
        logger.warning("The PDF index was not completed (pdf_index.py stopped "
                       "early); results may be partial. Re-run pdf_index.py.")
    if data.get("embed_model") not in (None, EMBED_MODEL):
        logger.warning(f"PDF index was built with {data.get('embed_model')!r} but "
                       f"queries use {EMBED_MODEL!r}; retrieval will be wrong.")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _rerank_enabled() -> bool:
    return THESAURUS_RERANK_ALPHA > 0.0 and get_thesaurus() is not None


def retrieve(
    question: str,
    k: int = 5,
    country_names: set[str] | None = None,
    search_text: str | None = None,
):
    """Top-k chunks for `question`.

    `search_text` (default: the question) is what gets embedded; the cached
    wrapper passes the thesaurus-expanded query here. Re-ranking always uses
    the original question.

    `country_names` is accepted for compatibility but ignored: the PDF index
    has no country metadata.
    """
    if country_names:
        logger.warning("PDF index has no country metadata; ignoring country filter.")

    collection = _get_collection()

    rerank = _rerank_enabled()
    fetch_k = k * RERANK_OVERFETCH if rerank else k

    raw = collection.query(
        query_embeddings=[list(embed_query(search_text or question))],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    ids, docs = raw["ids"][0], raw["documents"][0]
    metas, dists = raw["metadatas"][0], raw["distances"][0]

    if rerank and docs:
        order = get_thesaurus().rerank_order(
            question, docs, dists, THESAURUS_RERANK_ALPHA
        )[:k]
    else:
        order = list(range(min(k, len(docs))))

    return {
        "ids": [[ids[i] for i in order]],
        "documents": [[docs[i] for i in order]],
        "metadatas": [[metas[i] for i in order]],
        "distances": [[dists[i] for i in order]],
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_task(question: str, country_names: set[str]) -> str:
    return (
        "Answer the user's question using ONLY the retrieved chunks below.\n"
        "If the answer is not present, say "
        "'I cannot determine that from the supplied documents.'"
    )


def _build_context(results, max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, int]:
    """Return (context text, number of chunks included).

    Chunks are added in order until `max_chars` would be exceeded; the first
    chunk is always included. The caller reports any truncation.
    """
    parts: list[str] = []
    used = 0
    separator = "\n\n"

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        part = (
            f"SOURCE {i + 1}\n"
            f"File: {metadata.get('source_file', '')}\n"
            f"Title: {metadata.get('title', '')}\n"
            f"Page: {metadata.get('page', '')} of {metadata.get('page_count', '')}\n"
            f"Chunk: {metadata.get('chunk_index', '')} of "
            f"{metadata.get('chunk_total', '')}\n\n"
            f"{document}"
        )
        if parts and used + len(separator) + len(part) > max_chars:
            break
        parts.append(part)
        used += len(separator) + len(part)

    return separator.join(parts), len(parts)


def _thesaurus_blocks(question: str, results) -> tuple[str, str]:
    """(glossary, authoritative definitions) for the prompt; ('', '') if the
    thesaurus is unavailable."""
    th = get_thesaurus()
    if th is None:
        return "", ""
    glossary = definitions = ""
    try:
        glossary = th.build_glossary(question, list(results["documents"][0]))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Glossary build failed: {e}")
    try:
        definitions = th.definitions_block(question)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Thesaurus definitions failed: {e}")
    return glossary, definitions


def generate_answer(
    question: str,
    results,
    country_names: set[str] | None = None,
):
    context, used = _build_context(results)
    total = len(results["documents"][0])
    truncation_note = ""
    if used < total:
        logger.warning(f"Context capped at {used} of {total} retrieved chunks "
                       f"(RAG_MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS}).")
        truncation_note = (
            f"\nNOTE: the context limit was reached, so only the first {used} "
            f"of {total} retrieved chunks are shown above.\n"
        )

    glossary, definitions = _thesaurus_blocks(question, results)
    extras = "".join(f"\n{block}\n" for block in (definitions, glossary) if block)

    task = _build_task(question, country_names or set())

    prompt = f"""
You are answering questions about an organization's documents.

{task}

Do not invent policies, dates, numbers, names, or rules.

Question:
{question}

Context:
{context}
{truncation_note}{extras}
Answer the question clearly.

At the end, list the source pages you relied on.
"""

    return chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied document context. "
                    "If a section titled 'Authoritative IPBES definitions' "
                    "is present, treat it as the primary source for any "
                    "concept it defines."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Cached async wrappers
# ---------------------------------------------------------------------------

async def retrieve_cached(
    question: str,
    k: int = 5,
    country_names: set[str] | None = None,
):
    # Thesaurus expansion only changes what gets embedded; the cache key uses
    # the ORIGINAL question so identical questions share an entry.
    th = get_thesaurus()
    expanded = question
    if th is not None:
        try:
            expanded = th.expand_query(question)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Query expansion failed: {e}")

    key = make_key(
        "pdf:retrieve",
        CACHE_VERSION,
        RETRIEVAL_VERSION,
        EMBED_MODEL,
        COLLECTION_NAME,
        _index_stamp(),                 # a re-index invalidates cached retrievals
        THESAURUS_RERANK_ALPHA if th is not None else 0.0,
        RERANK_OVERFETCH,
        question.strip(),
        k,
    )

    hit = await cache.get(key)
    if hit is not None:
        logger.debug(f"pdf retrieve HIT  q={question[:50]!r}")
        return hit

    result = await asyncio.to_thread(
        retrieve,
        question,
        k,
        country_names,
        expanded,
    )
    # Never cache an empty result: it may just mean the index was missing,
    # incomplete or mid-rebuild, and it would stick for the whole TTL.
    if result["documents"][0]:
        await cache.set(key, result, ttl=RETRIEVE_TTL)
    logger.debug(f"pdf retrieve MISS q={question[:50]!r}")
    return result


async def generate_answer_cached(
    question: str,
    results,
    country_names: set[str] | None = None,
) -> dict:
    context, used = _build_context(results)
    total = len(results["documents"][0])
    glossary, definitions = _thesaurus_blocks(question, results)

    context_signature = make_key(
        "pdf:ctx", context, glossary, definitions
    )[:16]

    key = make_key(
        "pdf:answer",
        CACHE_VERSION,
        PROMPT_VERSION,
        LLM_MODEL,
        NUM_CTX,
        question.strip(),
        context_signature,
    )

    hit = await cache.get(key)
    if hit is not None:
        logger.debug(f"pdf answer HIT  q={question[:50]!r}")
        hit["_cached"] = True
        return hit

    answer = await asyncio.to_thread(
        generate_answer,
        question,
        results,
        country_names,
    )

    payload = {
        "answer": answer,
        "context_chunks": used,
        "context_truncated": used < total,
        "retrieved": [
            {
                "source_file": m.get("source_file"),
                "title": m.get("title"),
                "page": m.get("page"),
                "chunk_index": m.get("chunk_index"),
                "distance": results["distances"][0][i],
            }
            for i, m in enumerate(results["metadatas"][0])
        ],
        "_cached": False,
    }
    await cache.set(key, payload, ttl=ANSWER_TTL)
    logger.debug(f"pdf answer MISS q={question[:50]!r}")
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    print(f"LLM: {LLM_MODEL}")
    print(f"Embedding: {EMBED_MODEL}")
    th = get_thesaurus()
    print("Thesaurus: " + ("loaded" if th is not None
                          else "not available (expansion/re-ranking disabled)"))
    print()

    question = input("Question: ").strip()
    if not question:
        return

    results = await retrieve_cached(question, k=5)

    if not results["documents"][0]:
        print("No results found.")
        return

    payload = await generate_answer_cached(question, results)

    if payload.get("_cached"):
        print("(cache hit)")
    if payload.get("context_truncated"):
        print(f"(Only the first {payload['context_chunks']} of "
              f"{len(results['documents'][0])} retrieved chunks fit in the prompt.)")

    print()
    print("ANSWER")
    print("=" * 80)
    print(payload["answer"])

    print()
    print("RETRIEVED SOURCES")
    print("=" * 80)

    for i, metadata in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][i]

        print(
            f"{i + 1}. "
            f"{metadata.get('source_file', '')} "
            f"page {metadata.get('page', '')} "
            f"(distance={distance:.4f})"
        )


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
