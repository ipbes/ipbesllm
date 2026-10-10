# ttl_rag.py
import json
import os
import re
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from loguru import logger

from cache import CACHE_VERSION, PIPELINE_ID, cache, make_key
# infer_country_names / canonical_country are re-exported for app.py.
from geo import canonical_country, country_code_for_label, infer_country_names
from rag_utils import (
    EMBED_MODEL, MAX_CONTEXT_CHARS, NUM_CTX, chat, concat_results, embed_query,
    empty_results, interleave_results, named_assessments, select_results,
    strip_corpus_words,
)
from thesaurus_helper import get_thesaurus
from settings import CHROMA_DIR


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

COLLECTION_NAME = "ttl_documents"

# Manifest written by ttl_index.py listing the assessments it indexed.
# Used by the RAG to fan out one query per assessment so no single
# assessment can dominate the top-k results.
MANIFEST_PATH = Path(CHROMA_DIR) / "assessments.json"

# Weight of the thesaurus-neighborhood signal in post-retrieval re-ranking.
# 0.0 = pure vector search; 0.25 = gentle nudge (recommended).
THESAURUS_RERANK_ALPHA = float(os.getenv("THESAURUS_RERANK_ALPHA", "0.25"))

# A country filter with more codes than this looks like a bug in
# infer_country_names (the largest IPBES region has about 55 countries).
MAX_COUNTRY_FILTER = 100

# Part of the cache keys: bump when retrieval or prompts change so stale
# entries are never served after a code change.
RETRIEVAL_VERSION = "4"
PROMPT_VERSION = "3"


# ---------------------------------------------------------------------------
# Index state: collection + assessment manifest
# ---------------------------------------------------------------------------
# Both are cached, but dropped whenever ttl_index.py rewrites the manifest
# (it does at the start and end of every run), so a long-lived process such
# as Streamlit never keeps a handle to a collection that was rebuilt.

_STATE: dict = {"stamp": None, "collection": None, "assessments": None}


def _index_stamp() -> int:
    try:
        return MANIFEST_PATH.stat().st_mtime_ns
    except OSError:
        return 0


def _sync_index_state() -> None:
    stamp = _index_stamp()
    if _STATE["stamp"] != stamp:
        _STATE.update(stamp=stamp, collection=None, assessments=None)


def _get_collection():
    _sync_index_state()
    if _STATE["collection"] is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            # No embedding function: queries are embedded by embed_query and
            # passed as vectors, so the collection never embeds anything.
            _STATE["collection"] = client.get_collection(name=COLLECTION_NAME)
        except NotFoundError as exc:
            raise SystemExit(
                f"Collection '{COLLECTION_NAME}' not found. "
                f"Run ttl_index.py first. (Original error: {exc})"
            )
        except ValueError as exc:
            if "does not exist" in str(exc).lower():
                raise SystemExit(
                    f"Collection '{COLLECTION_NAME}' not found. "
                    f"Run ttl_index.py first. (Original error: {exc})"
                )
            raise
    return _STATE["collection"]


# ---------------------------------------------------------------------------
# Thesaurus (shared with pdf_rag via thesaurus_helper.get_thesaurus)
# ---------------------------------------------------------------------------

def _get_thesaurus():
    return get_thesaurus()


# ---------------------------------------------------------------------------
# Assessment discovery (from manifest)
# ---------------------------------------------------------------------------

def _get_assessments() -> list[str]:
    """
    Return the assessment IDs recorded by the indexer in the manifest.

    Falls back to an empty list if the manifest is missing (i.e. an index
    produced before this change). Callers must treat an empty list as
    'per-assessment retrieval not available' and fall back to a single
    query.
    """
    _sync_index_state()
    if _STATE["assessments"] is not None:
        return _STATE["assessments"]

    if not MANIFEST_PATH.exists():
        logger.warning(
            f"Assessment manifest {MANIFEST_PATH} not found; "
            f"per-assessment retrieval disabled. Re-run ttl_index.py."
        )
        _STATE["assessments"] = []
        return _STATE["assessments"]

    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(
            f"Could not read assessment manifest {MANIFEST_PATH}: {e}; "
            f"per-assessment retrieval disabled."
        )
        _STATE["assessments"] = []
        return _STATE["assessments"]

    if data.get("complete") is False:
        logger.warning("The index was not completed (ttl_index.py stopped "
                       "early); results may be partial. Re-run ttl_index.py.")
    if data.get("embed_model") not in (None, EMBED_MODEL):
        logger.warning(f"Index was built with {data.get('embed_model')!r} but "
                       f"queries use {EMBED_MODEL!r}; retrieval will be wrong.")

    _STATE["assessments"] = sorted(data.get("assessments", []))
    logger.debug(f"Loaded assessments from manifest: {_STATE['assessments']}")
    return _STATE["assessments"]


# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------

def _country_codes(country_names) -> set[str]:
    """Canonical country names -> ISO alpha-3 codes (unknown names dropped)."""
    codes = set()
    for name in country_names or ():
        code = country_code_for_label(name)
        if code:
            codes.add(code)
    return codes


def _country_text(m: dict) -> str:
    """Display text for a chunk's country, from the indexer's metadata."""
    names = m.get("country_names")
    if isinstance(names, (list, tuple)):
        return ", ".join(names)
    return names or m.get("country_raw") or ""


def _annotate_metadata(results):
    """Add a display-only `country` string to each metadata dict.

    The index stores country_codes / country_names strings; `country` is derived
    so the context builder, app.py and the CLI printout can keep using it.
    It is never used for filtering.
    """
    for m in results["metadatas"][0]:
        if m is not None and "country" not in m:
            text = _country_text(m)
            if text:
                m["country"] = text
    return results


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _build_where(
    chunk_type: str | None,
    country_codes: set[str] | None,
    assessment: str | None = None,
):
    """Build a Chroma `where` clause from the individual filters.

    Chroma 1.4 rejects list metadata, so the indexer stores one boolean flag
    per country (country_KEN=True); several countries are OR-ed together.
    """
    clauses = []
    if chunk_type:
        clauses.append({"chunk_type": chunk_type})
    if country_codes:
        conds = [{f"country_{c}": True} for c in sorted(country_codes)]
        clauses.append(conds[0] if len(conds) == 1 else {"$or": conds})
    if assessment:
        clauses.append({"assessment": assessment})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _query(collection, query_vector, k: int, where):
    return collection.query(
        query_embeddings=[list(query_vector)],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


# ---------------------------------------------------------------------------
# Thesaurus-aware re-ranking
# ---------------------------------------------------------------------------

def _rerank(question: str, results):
    """Re-order one result set by vector similarity blended with the
    thesaurus neighbourhood score (THESAURUS_RERANK_ALPHA)."""
    th = _get_thesaurus()
    if th is None or THESAURUS_RERANK_ALPHA <= 0.0 or not results["documents"][0]:
        return results
    order = th.rerank_order(question, results["documents"][0],
                            results["distances"][0], THESAURUS_RERANK_ALPHA)
    return select_results(results, order)


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    k: int = 5,
    chunk_type: str | None = None,
    country_names: set[str] | None = None,
    per_assessment: bool = False,
    search_text: str | None = None,
):
    """
    Retrieve chunks.

    `search_text` (default: the question) is what gets embedded; the cached
    wrapper passes a cleaned, thesaurus-expanded version. Re-ranking and the
    choice of assessments use the question itself.

    With `per_assessment=True`, run one query per assessment (`k` results
    each) so no single assessment can crowd out the others; when the question
    names assessments (GA1, IAS, LDR ...) only those are searched. Typed
    results (key messages, persons ...) are concatenated, to be sorted by
    identifier later; general ones are re-ranked with the thesaurus within
    each assessment and interleaved by rank.
    """
    collection = _get_collection()

    codes = _country_codes(country_names)
    if country_names and len(codes) > MAX_COUNTRY_FILTER:
        logger.warning(
            f"country filter has {len(codes)} codes; this looks like a bug "
            f"in infer_country_names. Dropping the filter."
        )
        codes = set()
    elif country_names and not codes:
        # Never fall through to an unfiltered search when the caller asked
        # for specific countries we cannot map to codes.
        logger.warning(f"No ISO codes for countries {sorted(country_names)}; "
                       f"returning no results.")
        return empty_results()

    # Embed once; every per-assessment query reuses the same vector.
    query_vector = embed_query(search_text or question)

    assessments = _get_assessments() if per_assessment else []
    if per_assessment and not assessments:
        # No manifest -> behave like the pre-per-assessment code path.
        logger.warning(
            "per_assessment=True but no assessments available; "
            "falling back to a single query."
        )
    assessments = named_assessments(question, assessments) or assessments

    if not assessments:
        r = _query(collection, query_vector, k, _build_where(chunk_type, codes))
        return _annotate_metadata(r if chunk_type else _rerank(question, r))

    per_assessment_results, errors = [], []
    for a in assessments:
        where = _build_where(chunk_type, codes, assessment=a)

        try:
            r = _query(collection, query_vector, k, where)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Per-assessment query failed for {a!r}: {e}")
            errors.append(e)
            continue

        # Tag each returned metadata with the assessment so downstream
        # code (and the LLM context) can show where it came from.
        for m in r["metadatas"][0]:
            m.setdefault("assessment", a)

        if r["documents"][0]:
            per_assessment_results.append(r if chunk_type else _rerank(question, r))

    if not per_assessment_results:
        if errors and len(errors) == len(assessments):
            # Every query failed: surface it instead of returning "no
            # results" (which would also be cached).
            raise errors[-1]
        return empty_results()

    merge = concat_results if chunk_type else interleave_results
    return _annotate_metadata(merge(per_assessment_results))


def plan_query(question: str, k: int = 5) -> dict:
    """How to retrieve for `question`: chunk type, countries, and results per
    assessment. Shared by main(), app.py and pdf_ttl_compare.py.

    Every question is searched per assessment. Typed questions (key messages,
    persons ...) list everything, so they fetch far more than `k`.
    """
    chunk_type = _infer_chunk_type(question)
    country_names: set[str] = set()
    if chunk_type == "person":
        country_names = {
            c for c in (
                canonical_country(x) for x in infer_country_names(question)
            ) if c
        }
    if country_names:
        k = 200
    elif chunk_type:
        k = 100
    return {
        "chunk_type": chunk_type,
        "country_names": country_names,
        "k": k,
        "per_assessment": True,
    }


# ---------------------------------------------------------------------------
# Chunk-type inference
# ---------------------------------------------------------------------------

def _infer_chunk_type(question: str) -> str | None:
    q = question.lower()

    if "key message" in q or "key messages" in q:
        return "key"
    if "knowledge gap" in q or "knowledge gaps" in q:
        return "kg"
    if "sub-message" in q or "submessage" in q or "sub message" in q:
        return "subm"
    if "background message" in q:
        return "bgm"
    if "illustration" in q or "figure" in q:
        return "il"
    if "reference" in q or "citation" in q:
        return "ref"
    if "author" in q or "person" in q or "expert" in q:
        return "person"
    if "subchapter" in q or "sub-chapter" in q:
        return "sch"

    return None


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

_IDENTIFIER_RE = re.compile(r"^([A-Za-z]+)\s*(\d+)")


def _identifier_sort_key(identifier: str):
    m = _IDENTIFIER_RE.match(identifier or "")
    if m:
        return (m.group(1).upper(), int(m.group(2)))
    return (identifier or "", 0)


def _reorder_by_identifier(results):
    """Sort by (assessment, identifier) so assessments are not interleaved."""
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    ids = results["ids"][0]

    order = sorted(
        range(len(docs)),
        key=lambda i: (
            metas[i].get("assessment", ""),
            _identifier_sort_key(metas[i].get("identifier", "")),
        ),
    )

    results["documents"][0] = [docs[i] for i in order]
    results["metadatas"][0] = [metas[i] for i in order]
    results["distances"][0] = [dists[i] for i in order]
    results["ids"][0] = [ids[i] for i in order]
    return results


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_context(results, max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, int]:
    """Return (context text, number of chunks included).

    Chunks are added in order until `max_chars` would be exceeded; the first
    chunk is always included. The caller reports any truncation.
    """
    parts: list[str] = []
    used = 0
    separator = "\n\n---\n\n"

    for i, document in enumerate(results["documents"][0]):
        m = results["metadatas"][0][i]

        # Drop only the document-level preamble lines, keep the rest
        # (Heading / Chapter / Subchapter / Country / Roles ...).
        lines = document.splitlines()
        keep = [
            ln for ln in lines
            if not ln.startswith("Document:")
            and not ln.startswith("Date:")
        ]
        body = "\n".join(keep).strip()

        prefix_parts = []
        if m.get("identifier"):
            prefix_parts.append(f"id={m['identifier']}")
        if m.get("chunk_type"):
            prefix_parts.append(f"type={m['chunk_type']}")
        if m.get("qualifier"):
            prefix_parts.append(f"qualifier={m['qualifier']}")
        if m.get("assessment"):
            prefix_parts.append(f"assessment={m['assessment']}")
        country = _country_text(m)
        if country:
            prefix_parts.append(f"country={country}")
        if m.get("paragraph"):
            prefix_parts.append(f"part={m['paragraph']}")

        prefix = " ".join(prefix_parts)
        part = f"[{i + 1}] {prefix}\n{body}"

        if parts and used + len(separator) + len(part) > max_chars:
            break
        parts.append(part)
        used += len(separator) + len(part)

    return separator.join(parts), len(parts)


def _build_task(
    question: str,
    chunk_type: str | None,
    country_names: set[str],
) -> str:
    if chunk_type == "key":
        return (
            "The user is asking for the KEY MESSAGES of an IPBES assessment.\n"
            "Every retrieved chunk below is a KeyMessage.\n"
            "\n"
            "List ALL retrieved KeyMessages, in the order given, numbered.\n"
            "For each one output:\n"
            "  - identifier (e.g. A1, B3)\n"
            "  - qualifier (well established / established but incomplete / "
            "unresolved) if present\n"
            "  - the key message text\n"
            "\n"
            "CRITICAL:\n"
            "- Do NOT answer any question that appears inside a key message.\n"
            "- Do NOT invent a heading or rephrase into your own question.\n"
            "- Do NOT summarise; reproduce each key message faithfully.\n"
        )

    if chunk_type == "person" and country_names:
        names = ", ".join(sorted(country_names))
        return (
            f"The user is asking about experts from: {names}.\n"
            f"Every retrieved chunk is a Person whose country field is one "
            f"of those names.\n"
            "\n"
            "List ALL retrieved persons by their full name (the heading).\n"
            "For each, also give the country and the role(s) mentioned in "
            "the body, if any.\n"
            "\n"
            "CRITICAL:\n"
            "- Only include persons whose country matches one of the names "
            "above.\n"
            "- If a retrieved chunk's body shows a different country, drop "
            "it.\n"
            "- Do not invent names, countries, or roles.\n"
        )

    if chunk_type == "person":
        return (
            "The user is asking about experts.\n"
            "Each retrieved chunk is a Person. List the persons and the "
            "country and roles mentioned in the body.\n"
        )

    if chunk_type:
        return (
            f"Answer the user's question using ONLY the retrieved "
            f"{chunk_type} chunks below.\n"
            "If the answer is not present, say "
            "'I cannot determine that from the supplied documents.'"
        )

    return (
        "Answer the user's question using ONLY the retrieved chunks below.\n"
        "Say which assessment (the assessment= field) each point comes from, "
        "and note where assessments differ or one says nothing.\n"
        "If the answer is not present, say "
        "'I cannot determine that from the supplied documents.'"
    )


def generate_answer(
    question: str,
    results,
    chunk_type: str | None = None,
    country_names: set[str] | None = None,
    on_token=None,
) -> str:
    # Thesaurus re-ranking happens in retrieve(), within each assessment.
    context, used = _build_context(results)
    total = len(results["documents"][0])
    truncation_note = ""
    if used < total:
        logger.warning(f"Context capped at {used} of {total} retrieved chunks "
                       f"(RAG_MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS}).")
        truncation_note = (
            f"\nNOTE: the context limit was reached, so only the first {used} "
            f"of {total} retrieved chunks are shown above. Do not claim the "
            f"list is complete.\n"
        )
    task = _build_task(question, chunk_type, country_names or set())

    # -------- Thesaurus glossary + primary definitions --------
    glossary = ""
    thesaurus_primaries = ""
    th = _get_thesaurus()
    if th is not None and not chunk_type:
        try:
            chunk_texts = list(results["documents"][0])
            glossary = th.build_glossary(question, chunk_texts)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Glossary build failed: {e}")
            glossary = ""

        # If the question explicitly names thesaurus concepts, surface
        # their full definitions as an authoritative context block. This
        # matters when the assessment corpus doesn't define the concept
        # itself (e.g. LDR assumes 'NCP' is known to the reader).
        try:
            thesaurus_primaries = th.definitions_block(question)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Thesaurus primary block failed: {e}")

    glossary_block = f"\n{glossary}\n" if glossary else ""
    primary_block = f"\n{thesaurus_primaries}\n" if thesaurus_primaries else ""

    order_note = (" (sorted by identifier)" if chunk_type else
                  " (taken from each assessment in turn, best match first)")

    prompt = f"""
You are answering questions about an IPBES assessment report, represented
as RDF/Turtle using the IPBES ontology.

{task}

Preserve any qualifier (well established / established but incomplete /
unresolved) verbatim. Do not invent facts, dates, names, numbers, or
decisions.

User question:
{question}

Retrieved chunks{order_note}:
{context}
{truncation_note}{primary_block}{glossary_block}"""

    return chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied IPBES ontology context. "
                    "Preserve evidence qualifiers. Follow the task "
                    "instructions exactly, including for list questions. "
                    "If a section titled 'Authoritative IPBES definitions' "
                    "is present, treat it as the primary source for any "
                    "concept it defines, and cite that definition rather "
                    "than inferring one from the retrieved chunks. If the "
                    "retrieved chunks do not address the question, say so "
                    "explicitly rather than substituting an unrelated "
                    "topic."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        on_token=on_token,
    )

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

# ---------- retrieval (async) ----------
async def retrieve_cached(
    question: str,
    k: int = 5,
    chunk_type: str | None = None,
    country_names: set[str] | None = None,
    per_assessment: bool = False,
):
    """Cached Chroma retrieval. Only the search text is cleaned of
    corpus-generic words and expanded with the thesaurus."""
    import anyio

    norm_countries = sorted(country_names) if country_names else None

    # --- Search text (only affects what we send to Chroma) -------------
    th = _get_thesaurus()
    search = strip_corpus_words(question)
    if th is not None:
        try:
            search = th.expand_query(search)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Query expansion failed: {e}")

    # --- Cache key uses the ORIGINAL question so caches stay stable ----
    # (otherwise a change to the thesaurus would invalidate every cache
    # entry, and identical user questions would land in different buckets)
    key = make_key(
        CACHE_VERSION,
        PIPELINE_ID,
        "retrieve",
        RETRIEVAL_VERSION,
        EMBED_MODEL,
        COLLECTION_NAME,
        _index_stamp(),            # a re-index invalidates cached retrievals
        question.strip(),
        k,
        chunk_type,
        norm_countries,
        per_assessment,
    )

    hit = await cache.get(key)
    if hit is not None:
        logger.debug(f"ttl retrieve HIT q={question[:50]!r}")
        return hit

    result = await anyio.to_thread.run_sync(
        lambda: retrieve(
            question,
            k=k,
            chunk_type=chunk_type,
            country_names=country_names,
            per_assessment=per_assessment,
            search_text=search,
        )
    )
    # Never cache an empty result: it may just mean the index was missing,
    # incomplete or mid-rebuild, and it would stick for the whole TTL.
    if result["documents"][0]:
        await cache.set(key, result, ttl=3600)
    logger.debug(f"ttl retrieve MISS q={question[:50]!r}")
    return result

async def generate_answer_cached(
    question: str,
    results,
    chunk_type: str | None = None,
    country_names: set[str] | None = None,
    on_token=None,
) -> dict:
    """Cache the LLM answer using the retrieved context as part of the key."""
    import anyio

    context, used = _build_context(results)
    total = len(results["documents"][0])

    # Add the glossary and definitions to the key so thesaurus updates
    # (including src/enrichment.ttl) invalidate the cache.
    th = _get_thesaurus()
    glossary_for_key = ""
    if th is not None and not chunk_type:
        try:
            glossary_for_key = (
                th.build_glossary(question, list(results["documents"][0]))
                + th.definitions_block(question)
            )
        except Exception:  # noqa: BLE001
            glossary_for_key = ""

    ctx_hash = make_key("ctx", context + "\n" + glossary_for_key)
    key = make_key(
        CACHE_VERSION,
        PIPELINE_ID,
        "answer",
        PROMPT_VERSION,
        LLM_MODEL,
        NUM_CTX,
        question.strip(),
        chunk_type,
        sorted(country_names) if country_names else None,
        ctx_hash,
    )

    hit = await cache.get(key)
    if hit is not None:
        logger.debug(f"ttl answer HIT q={question[:50]!r}")
        return {**hit, "_cached": True}

    answer = await anyio.to_thread.run_sync(
        lambda: generate_answer(
            question,
            results,
            chunk_type=chunk_type,
            country_names=country_names,
            on_token=on_token,
        )
    )

    payload = {
        "answer": answer,
        "context_chunks": used,
        "context_truncated": used < total,
        "retrieved": [
            {
                "identifier": m.get("identifier"),
                "chunk_type": m.get("chunk_type"),
                "assessment": m.get("assessment"),
                "country": _country_text(m),
                "eId": m.get("eId"),
                "distance": results["distances"][0][i],
            }
            for i, m in enumerate(results["metadatas"][0])
        ],
        "_cached": False,
    }
    await cache.set(key, payload, ttl=6 * 60 * 60)
    logger.debug(f"ttl answer MISS q={question[:50]!r}")
    return payload

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def async_main():
    print(f"LLM: {LLM_MODEL}")
    print(f"Embedding: {EMBED_MODEL}")
    print()

    # Pre-load the thesaurus so the first user query isn't slowed by it.
    th = _get_thesaurus()
    if th is not None:
        print(f"Thesaurus: loaded ({len(th._lexicon)} labels)")  # noqa: SLF001
    else:
        print("Thesaurus: not available (expansion disabled)")

    question = input("Question: ").strip()
    if not question:
        return

    plan = plan_query(question)
    chunk_type, country_names = plan["chunk_type"], plan["country_names"]
    if chunk_type:
        print(f"(Filtering to chunk_type='{chunk_type}')")
    if country_names:
        print(f"(Filtering to countries: {sorted(country_names)})")
    print("(Retrieving per assessment)")
    print()

    results = await retrieve_cached(
        question,
        k=plan["k"],
        chunk_type=chunk_type,
        country_names=country_names or None,
        per_assessment=plan["per_assessment"],
    )

    if not results["documents"][0]:
        if country_names:
            print(f"No results found for countries {sorted(country_names)}.")
        else:
            print("No results found.")
        return

    if chunk_type:
        results = _reorder_by_identifier(results)

    payload = await generate_answer_cached(
        question,
        results,
        chunk_type=chunk_type,
        country_names=country_names,
    )

    if payload.get("_cached"):
        print("(cache hit)")
    if payload.get("context_truncated"):
        print(f"(Only the first {payload['context_chunks']} of "
              f"{len(results['documents'][0])} retrieved chunks fit in the "
              f"prompt; raise RAG_MAX_CONTEXT_CHARS / OLLAMA_NUM_CTX to include more.)")

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(payload["answer"])

    print()
    print("=" * 80)
    print("RETRIEVED TTL SOURCES")
    print("=" * 80)

    for i, m in enumerate(results["metadatas"][0]):
        d = results["distances"][0][i]
        print(
            f"{i + 1:>2}. "
            f"{m.get('identifier', ''):<5} "
            f"{m.get('chunk_type', '?'):<8} "
            f"assessment={m.get('assessment', '?'):<8} "
            f"country={m.get('country', ''):<25} "
            f"eId={m.get('eId', '?'):<25} "
            f"distance={d:.4f}"
        )


def main():
    import asyncio
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
