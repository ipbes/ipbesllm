import os

import chromadb
import ollama
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import (
    OllamaEmbeddingFunction,
)


LLM_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.1:latest",
)

EMBED_MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text",
)

CHROMA_DIR = "chroma"
COLLECTION_NAME = "pdf_documents"


def _get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

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


def retrieve(question: str, k: int = 5):
    collection = _get_collection()

    return collection.query(
        query_texts=[question],
        n_results=k,
    )


def generate_answer(question: str, results):
    context_parts = []

    for i, document in enumerate(
        results["documents"][0]
    ):
        metadata = results["metadatas"][0][i]

        context_parts.append(
            f"""
SOURCE {i + 1}
File: {metadata.get('source_file', '')}
Title: {metadata.get('title', '')}
Page: {metadata.get('page', '')} of {metadata.get('page_count', '')}
Chunk: {metadata.get('chunk_index', '')} of {metadata.get('chunk_total', '')}

{document}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are answering questions about an organization's documents.

Use ONLY the supplied context.

If the answer cannot be established from the context,
say:

"I cannot determine that from the supplied documents."

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
    )

    return response["message"]["content"]


def main():
    print(f"LLM: {LLM_MODEL}")
    print(f"Embedding: {EMBED_MODEL}")
    print()

    question = input("Question: ").strip()

    if not question:
        return

    results = retrieve(question, k=5)

    answer = generate_answer(
        question,
        results,
    )

    print()
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print()
    print("RETRIEVED SOURCES")
    print("=" * 80)

    for i, metadata in enumerate(
        results["metadatas"][0]
    ):
        distance = results["distances"][0][i]

        print(
            f"{i + 1}. "
            f"{metadata.get('source_file', '')} "
            f"page {metadata.get('page', '')} "
            f"(distance={distance:.4f})"
        )


if __name__ == "__main__":
    main()