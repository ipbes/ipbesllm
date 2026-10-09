import os

import chromadb

from rag_utils import EMBED_MODEL, chat, embed_query
from settings import CHROMA_DIR


LLM_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.1:latest",
)


COLLECTION_NAME = "xml_documents"


def retrieve(
    question: str,
    k: int = 10,
):

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    # No embedding function: the question is embedded by embed_query
    # (rag_utils: OLLAMA_URL, OLLAMA_EMBED_MODEL, timeout and retries).
    collection = client.get_collection(
        name=COLLECTION_NAME,
    )

    return collection.query(
        query_embeddings=[list(embed_query(question, EMBED_MODEL))],
        n_results=k,
    )


def generate_answer(
    question: str,
    results,
):

    context_parts = []

    for i, document in enumerate(
        results["documents"][0]
    ):

        metadata = results["metadatas"][0][i]

        context_parts.append(
            f"""
SOURCE {i + 1}

File: {metadata["source_file"]}
Document type: {metadata["document_type"]}
Title: {metadata["title"]}
Date: {metadata["date"]}

Section: {metadata["division"]}
Subsection: {metadata["subdivision"]}
Paragraph: {metadata["paragraph"]}

Akoma Ntoso eId: {metadata["eId"]}
Location: {metadata["xpath"]}

Content:
{document}
"""
        )

    context = "\n".join(
        context_parts
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
            f"{metadata['source_file']} | "
            f"{metadata['division']} | "
            f"{metadata['subdivision']} | "
            f"paragraph {metadata['paragraph']} | "
            f"distance={distance:.4f}"
        )


if __name__ == "__main__":
    main()