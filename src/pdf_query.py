import chromadb
from chromadb.errors import NotFoundError

from rag_utils import EMBED_MODEL, embed_query


CHROMA_DIR = "chroma"
COLLECTION_NAME = "pdf_documents"


def main():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    try:
        # No embedding function needed: the query is embedded below.
        collection = client.get_collection(name=COLLECTION_NAME)
    except (NotFoundError, ValueError) as e:
        raise SystemExit(
            f"Collection '{COLLECTION_NAME}' not found in "
            f"'{CHROMA_DIR}'. Run pdf_index.py first. "
            f"(Original error: {e})"
        )

    question = input("Question: ").strip()

    if not question:
        return

    print(f"(Embedding with {EMBED_MODEL}...)")
    results = collection.query(
        query_embeddings=[list(embed_query(question))],
        n_results=5,
        include=["documents", "metadatas", "distances"],
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
        print(f"Distance: {distance:.4f}")
        print(f"Source: {metadata.get('source_file', '')}")
        print(f"Title: {metadata.get('title', '')}")
        print(
            f"Page: {metadata.get('page', '')} "
            f"of {metadata.get('page_count', '')}"
        )
        print(
            f"Chunk: {metadata.get('chunk_index', '')} "
            f"of {metadata.get('chunk_total', '')}"
        )
        print()
        print(document)


if __name__ == "__main__":
    main()
