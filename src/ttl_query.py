# ttl_query.py
import chromadb

from rag_utils import embed_query
from settings import CHROMA_DIR


COLLECTION_NAME = "ttl_documents"


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # No embedding function: the question is embedded by embed_query
    # (rag_utils: OLLAMA_URL, OLLAMA_EMBED_MODEL, timeout and retries).
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Collection contains {collection.count()} chunks.")
    print()

    question = input("Question: ").strip()
    if not question:
        return

    results = collection.query(
        query_embeddings=[list(embed_query(question))],
        n_results=5,
    )

    print()
    print("=" * 80)
    print("TTL RETRIEVAL RESULTS")
    print("=" * 80)

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        print()
        print(f"RESULT #{i + 1}")
        print("-" * 80)
        print(f"Distance: {distance:.4f}")
        print(f"File: {metadata.get('source_file', '')}")
        print(f"Type: {metadata.get('chunk_type', '')}")
        print(f"Division: {metadata.get('division', '')}")
        print(f"Subdivision: {metadata.get('subdivision', '')}")
        print(f"eId: {metadata.get('eId', '')}")
        print(f"XPath: {metadata.get('xpath', '')}")
        print()
        print(document)


if __name__ == "__main__":
    main()