import asyncio
import os

import chromadb
from chromadb.errors import NotFoundError
from loguru import logger

from cache import CACHE_VERSION, cache, make_key
from rag_utils import EMBED_MODEL, MAX_CONTEXT_CHARS, NUM_CTX, chat, embed_query
from settings import CHROMA_DIR


LLM_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.1:latest",
)


COLLECTION_NAME = "xml_documents"

# Part of the answer cache key: bump when the prompt changes.
PROMPT_VERSION = "1"
ANSWER_TTL = 60 * 60 * 6      # 6 hours


def retrieve(
    question: str,
    k: int = 10,
):

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # No embedding function: the question is embedded by embed_query
    # (rag_utils: OLLAMA_URL, OLLAMA_EMBED_MODEL, timeout and retries).
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
        )
    except (NotFoundError, ValueError) as exc:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found in '{CHROMA_DIR}'. "
            f"Run xml_index.py first. (Original error: {exc})"
        )

    return collection.query(
        query_embeddings=[list(embed_query(question, EMBED_MODEL))],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )


def _build_context(results, max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, int]:
    """Return (context text, number of chunks included).

    Chunks are added in order until `max_chars` would be exceeded; the first
    chunk is always included. The caller reports any truncation.
    """
    parts: list[str] = []
    used = 0

    for i, document in enumerate(results["documents"][0]):
        m = results["metadatas"][0][i] or {}
        part = f"""
SOURCE {i + 1}

File: {m.get("source_file", "")}
Document type: {m.get("document_type", "")}
Title: {m.get("title", "")}
Date: {m.get("date", "")}

Section: {m.get("division", "")}
Subsection: {m.get("subdivision", "")}
Paragraph: {m.get("paragraph", "")}

Akoma Ntoso eId: {m.get("eId", "")}
Location: {m.get("xpath", "")}

Content:
{document}
"""
        if parts and used + len(part) + 1 > max_chars:
            break
        parts.append(part)
        used += len(part) + 1

    return "\n".join(parts), len(parts)


def generate_answer(
    question: str,
    results,
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

    prompt = f"""
You are answering questions about
Akoma Ntoso documents.

Use ONLY the supplied context.

The XML contains structural information such as:
- document
- section
- subsection
- paragraph
- table row
- Akoma Ntoso eId
- XML location

Use that information when identifying sources.

If the answer cannot be established from
the supplied context, say:

"I cannot determine that from the supplied documents."

Do not invent:
- facts
- dates
- names
- numbers
- decisions
- policies
- procedural rules

Question:
{question}

Context:
{context}
{truncation_note}
Answer the question clearly.

At the end provide:

Sources:
- document
- section/subsection
- paragraph or table row
- Akoma Ntoso location
"""

    return chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied "
                    "Akoma Ntoso document context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )


async def generate_answer_cached(question: str, results) -> dict:
    """Cache the answer, keyed on the question and the retrieved context."""
    context, used = _build_context(results)
    total = len(results["documents"][0])
    key = make_key(
        "xml:answer",
        CACHE_VERSION,
        PROMPT_VERSION,
        LLM_MODEL,
        NUM_CTX,
        question.strip(),
        make_key("xml:ctx", context),
    )

    hit = await cache.get(key)
    if hit is not None:
        logger.debug(f"xml answer HIT  q={question[:50]!r}")
        return {**hit, "_cached": True}

    answer = await asyncio.to_thread(generate_answer, question, results)
    payload = {
        "answer": answer,
        "context_chunks": used,
        "context_truncated": used < total,
        "_cached": False,
    }
    await cache.set(key, payload, ttl=ANSWER_TTL)
    logger.debug(f"xml answer MISS q={question[:50]!r}")
    return payload


def main():

    print(
        f"LLM: {LLM_MODEL}"
    )

    print(
        f"Embedding: {EMBED_MODEL}"
    )

    print()

    question = input(
        "Question: "
    ).strip()

    if not question:
        return

    results = retrieve(
        question,
        k=5,
    )

    answer = generate_answer(
        question,
        results,
    )

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(answer)

    print()
    print("=" * 80)
    print("RETRIEVED XML SOURCES")
    print("=" * 80)

    for i, metadata in enumerate(
        results["metadatas"][0]
    ):

        distance = results["distances"][0][i]

        print(
            f"{i + 1}. "
            f"{metadata.get('source_file', '')} | "
            f"{metadata.get('division', '')} | "
            f"{metadata.get('subdivision', '')} | "
            f"paragraph {metadata.get('paragraph', '')} | "
            f"distance={distance:.4f}"
        )


if __name__ == "__main__":
    main()