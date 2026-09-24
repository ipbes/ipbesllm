import os

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


def retrieve(question: str, k: int = 5):
    collection = _get_collection()
    return collection.query(query_texts=[question], n_results=k)


def generate_answer(question: str, results):
    context_parts = []

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]

        context_parts.append(
            f"""
SOURCE {i + 1}

File: {metadata['source_file']}
Title: {metadata['title']}
Chunk type: {metadata['chunk_type']}
Chapter: {metadata['division']}
Subchapter: {metadata['subdivision']}
Subject URI: {metadata['xpath']}

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
- background_message
- sub_message
- key_message
- knowledge_gap
- subchapter
- illustration
- reference
- person

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

    results = retrieve(question, k=5)
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
            f"{metadata['chunk_type']} | "
            f"{metadata['division']} | "
            f"{metadata['subdivision']} | "
            f"eId={metadata['eId']} | "
            f"distance={distance:.4f}"
        )


if __name__ == "__main__":
    main()