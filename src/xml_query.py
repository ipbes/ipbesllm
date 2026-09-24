import chromadb

from chromadb.utils.embedding_functions import (
    OllamaEmbeddingFunction,
)


CHROMA_DIR = "chroma"

COLLECTION_NAME = "xml_documents"

EMBED_MODEL = "nomic-embed-text"


def main():

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    embedding_function = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url="http://localhost:11434/api/embeddings",
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    print(
        f"Collection contains "
        f"{collection.count()} chunks."
    )

    print()

    question = input(
        "Question: "
    ).strip()

    results = collection.query(
        query_texts=[question],
        n_results=5,
    )

    print()
    print("=" * 80)
    print("XML RETRIEVAL RESULTS")
    print("=" * 80)

    for i, document in enumerate(
        results["documents"][0]
    ):

        metadata = results["metadatas"][0][i]

        distance = results["distances"][0][i]

        print()
        print(f"RESULT #{i + 1}")
        print("-" * 80)

        print(f"Distance: {distance:.4f}")
        print(
            f"File: "
            f"{metadata['source_file']}"
        )

        print(
            f"Type: "
            f"{metadata['chunk_type']}"
        )

        print(
            f"Section: "
            f"{metadata['division']}"
        )

        print(
            f"Subsection: "
            f"{metadata['subdivision']}"
        )

        print(
            f"Paragraph: "
            f"{metadata['paragraph']}"
        )

        print(
            f"eId: "
            f"{metadata['eId']}"
        )

        print(
            f"XPath: "
            f"{metadata['xpath']}"
        )

        print()
        print(document)


if __name__ == "__main__":
    main()