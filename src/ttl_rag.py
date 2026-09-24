import os
import sys

import chromadb
import ollama
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


LLM_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHROMA_DIR = "chroma"
COLLECTION_NAME = "ttl_documents"


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
            f"Collection '{COLLECTION_NAME}' not found. "
            f"Run ttl_index.py first. (Original error: {e})"
        )


def retrieve(question: str, k: int = 5, chunk_type: str | None = None):
    collection = _get_collection()
    where = {"chunk_type": chunk_type} if chunk_type else None
    return collection.query(
        query_texts=[question],
        n_results=k,
        where=where,
    )


def _infer_chunk_type(question: str) -> str | None:
    """Infer a chunk_type filter from the question text."""
    q = question.lower()
    if "key message" in q:
        return "key"
    if "knowledge gap" in q:
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
    if "subchapter" in q or "section" in q:
        return "sch"
    return None


def generate_answer(question: str, results):
    context_parts = []

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]

        context_parts.append(
            f"""
SOURCE {i + 1}

File: {metadata['source_file']}
Title: {metadata['title']}
Chunk type: {metadata.get('chunk_type', '(unknown)')}
Chapter: {metadata['division']}
Subchapter: {metadata['subdivision']}
Subject URI: {metadata.get('xpath', '(unknown)')}
eId: {metadata.get('eId', '(unknown)')}

Content:
{document}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are answering questions about an IPBES assessment report, represented
as RDF/Turtle using the IPBES ontology.

Use ONLY the supplied context.

The context contains structured chunks extracted from the ontology:
- bgm (BackgroundMessage)
- subm (SubMessage)
- key (KeyMessage)
- kg (KnowledgeGap)
- sch (SubChapter)
- il (Illustration)
- ref (Reference)
- person (Person)

Each chunk may carry an "Identifier" (a section number or message number)
and a "Qualifier" (well established / established but incomplete /
unresolved). You MUST preserve the qualifier verbatim when you cite it.

If the answer cannot be established from the supplied context, say:

"I cannot determine that from the supplied documents."

Do not invent facts, dates, names, numbers, decisions, policies, or rules.

Question:
{question}

Context:
{context}

Answer the question clearly.

At the end provide:

Sources:
- chunk type
- subject URI
- identifier (if any)
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the supplied IPBES ontology "
                    "context. Preserve evidence qualifiers."
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

    # Infer chunk type filter from question
    chunk_type = _infer_chunk_type(question)
    if chunk_type:
        print(f"(Filtering to chunk_type='{chunk_type}')")
        print()

    results = retrieve(question, k=5, chunk_type=chunk_type)

    if not results["documents"][0]:
        print("No results found.")
        return

    answer = generate_answer(question, results)

    print()
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(answer)

    print()
    print("=" * 80)
    print("RETRIEVED TTL SOURCES")
    print("=" * 80)

    for i, metadata in enumerate(results["metadatas"][0]):
        distance = results["distances"][0][i]
        print(
            f"{i + 1}. {metadata['source_file']} | "
            f"{metadata.get('chunk_type', '?')} | "
            f"{metadata.get('division', '')} | "
            f"{metadata.get('subdivision', '')} | "
            f"eId={metadata.get('eId', '?')} | "
            f"distance={distance:.4f}"
        )


if __name__ == "__main__":
    main()