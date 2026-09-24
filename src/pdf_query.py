import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


CHROMA_DIR = "chroma"
COLLECTION_NAME = "pdf_documents"

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

    question = input("Question: ")

    results = collection.query(
        query_texts=[question],
        n_results=5,
    )

    print()
    print("RETRIEVAL RESULTS")
    print("=" * 80)

    for i, document in enumerate(
        results["documents"][0]
    ):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print()
        print(f"Result #{i + 1}")
        print(f"Distance: {distance}")
        print(f"Source: {metadata['source_file']}")
        print(f"Page: {metadata['page']}")
        print()
        print(document)


if __name__ == "__main__":
    main()