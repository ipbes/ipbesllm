import os

import chromadb
import ollama
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from loguru import logger
from cache import cache, make_key
from geo import infer_country_names


LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHROMA_DIR = "chroma"
COLLECTION_NAME = "pdf_documents"

# Cache TTLs (seconds)
RETRIEVE_TTL = 60 * 60        # 1 hour
ANSWER_TTL = 60 * 60 * 6      # 6 hours


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedding_function = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url="http://localhost:11434/api/embeddings",
    )

    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
        )
    except (NotFoundError, ValueError) as e:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found in "
            f"'{CHROMA_DIR}'. Run pdf_index.py first. "
            f"(Original error: {e})"
        )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    question: str,
    k: int = 5,
    country_names: set[str] | None = None,
):
    collection = _get_collection()

    where = None
    if country_names:
        names = sorted(country_names)
        if len(names) == 1:
            where = {"country": names[0]}
        else:
            where = {"country": {"$in": names}}

    return collection.query(
        query_texts=[question],
        n_results=k,
        where=where,
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_task(question: str, country_names: set[str]) -> str:
    if country_names:
        names = ", ".join(sorted(country_names))
        return (
            f"The user is asking about content from: {names}.\n"
            f"Every retrieved chunk already has a 'country' metadata field "
            f"matching one of those names.\n"
            "\n"
            "Answer the user's question using ONLY the retrieved chunks.\n"
            "If a chunk's metadata shows a country outside the list above, "
            "ignore it.\n"
            "If the answer is not present, say "
            "'I cannot determine that from the supplied documents.'\n"
        )

    return (
        "Answer the user's question using ONLY the retrieved chunks below.\n"
        "If the answer is not present, say "
        "'I cannot determine that from the supplied documents.'"
    )


def generate_answer(
    question: str,
    results,
    country_names: set[str] | None = None,
):
    context_parts = []

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]

        meta_lines = []
        if metadata.get("country"):
            meta_lines.append(f"Country: {metadata['country']}")

        meta_block = "\n".join(meta_lines)
        if meta_block:
            meta_block = "\n" + meta_block

        context_parts.append(
            f"""
SOURCE {i + 1}
File: {metadata.get('source_file', '')}
Title: {metadata.get('title', '')}
Page: {metadata.get('page', '')} of {metadata.get('page_count', '')}
Chunk: {metadata.get('chunk_index', '')} of {metadata.get('chunk_total', '')}{meta_block}

{document}
"""
        )

    context = "\n".join(context_parts)

    task = _build_task(question, country_names or set())

    prompt = f"""
You are answering questions about an organization's documents.

{task}

Do not invent policies, dates, numbers, names, or rules.

Question:
{question}

Context:
{context}

Answer the question clearly.

At the end, list the source pages you relied on.
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied "
                    "document context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={"temperature": 0, "seed": 42},
    )

    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Cached async wrappers
# ---------------------------------------------------------------------------

async def retrieve_cached(
    question: str,
    k: int = 5,
    country_names: set[str] | None = None,
):
    import asyncio

    norm_countries = sorted(country_names) if country_names else None

    key = make_key(
        "pdf:retrieve",
        question,
        k,
        norm_countries,
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
    )
    await cache.set(key, result, ttl=RETRIEVE_TTL)
    logger.debug(f"pdf retrieve MISS q={question[:50]!r}")
    return result


async def generate_answer_cached(
    question: str,
    results,
    country_names: set[str] | None = None,
) -> dict:
    import asyncio

    context_signature = make_key(
        "pdf:ctx",
        question,
        results["documents"][0],
        [m.get("eId") for m in results["metadatas"][0]],
        [round(d, 6) for d in results["distances"][0]],
        sorted(country_names) if country_names else None,
    )[:16]

    key = make_key("pdf:answer", question, context_signature)

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
        "retrieved": [
            {
                "source_file": m.get("source_file"),
                "title": m.get("title"),
                "page": m.get("page"),
                "chunk_index": m.get("chunk_index"),
                "country": m.get("country"),
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
    print()

    question = input("Question: ").strip()
    if not question:
        return

    # PDF country metadata isn't indexed; skip the filter.
    country_names = infer_country_names(question)
    if country_names:
        print(f"(Country hint: {sorted(country_names)} — not filtering)")
        country_names = set()

    k = 10 if country_names else 5

    print()

    results = await retrieve_cached(
        question,
        k=k,
        country_names=country_names or None,
    )


    if not results["documents"][0]:
        if country_names:
            print(
                f"No results found for countries {sorted(country_names)}."
            )
        else:
            print("No results found.")
        return

    payload = await generate_answer_cached(
        question,
        results,
        country_names=country_names,
    )

    if payload.get("_cached"):
        print("(cache hit)")

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
            f"country={metadata.get('country', ''):<25} "
            f"(distance={distance:.4f})"
        )


def main():
    import asyncio
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()