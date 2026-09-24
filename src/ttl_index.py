from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from ttl_loader import parse_ttl_file


TTL_DIR = Path("data/ttl")

CHROMA_DIR = "chroma"
COLLECTION_NAME = "ttl_documents"

EMBED_MODEL = "nomic-embed-text"

BATCH_SIZE = 25


def _delete_collection_if_exists(client, name: str) -> None:
    try:
        client.delete_collection(name=name)
        print(f"  Existing collection '{name}' deleted.")
    except NotFoundError:
        print(f"  Collection '{name}' did not exist.")
    except ValueError as e:
        if "does not exist" in str(e).lower():
            print(f"  Collection '{name}' did not exist.")
        else:
            raise


def main():
    print("Starting TTL indexing...")
    print(f"Current directory: {Path.cwd()}")
    print(f"TTL directory: {TTL_DIR.resolve()}")
    print()

    ttl_files = list(TTL_DIR.glob("*.ttl"))

    print(f"TTL files found: {len(ttl_files)}")
    for ttl in ttl_files:
        print(f"  - {ttl}")
    print()

    if not ttl_files:
        print("ERROR: No TTL files found.")
        return

    print("Connecting to Chroma...")

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedding_function = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url="http://localhost:11434/api/embeddings",
    )

    print(f"Deleting existing collection: {COLLECTION_NAME}")
    _delete_collection_if_exists(client, COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Created new collection: {COLLECTION_NAME}")
    print("Distance metric: cosine")
    print()

    all_chunks: list[dict] = []

    for ttl_path in ttl_files:
        print(f"Processing: {ttl_path.name}")

        chunks = parse_ttl_file(str(ttl_path))

        print(f"  Chunks extracted: {len(chunks)}")

        all_chunks.extend(chunks)

    print()
    print(f"Total chunks: {len(all_chunks)}")
    print()

    if not all_chunks:
        print("ERROR: No chunks were extracted.")
        return

    print("Creating embeddings in batches...")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    total = len(all_chunks)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = all_chunks[start:end]

        ids = [chunk["chunk_id"] for chunk in batch]
        documents = [chunk["text"] for chunk in batch]

        metadatas = [
            {
                "source_type": "ttl",
                "source_file": chunk["source_file"],
                "chunk_type": chunk["chunk_type"],
                "document_type": chunk["document_type"],
                "title": chunk["title"],
                "date": chunk["date"],
                "language": chunk["language"],
                "country": chunk["country"],
                "subtype": chunk["subtype"],
                "number": chunk["number"],
                "division": chunk["division"],
                "subdivision": chunk["subdivision"],
                "paragraph": chunk["paragraph"],
                "eId": chunk["eId"],
                "xpath": chunk["xpath"],
                "identifier": chunk["identifier"],
                "qualifier": chunk["qualifier"],
            }
            for chunk in batch
        ]

        print(
            f"Storing chunks {start + 1}-{end} of {total}..."
        )

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"  Stored {end}/{total}")

    print()
    print("=" * 60)
    print("TTL indexing complete.")
    print("=" * 60)
    print(f"Chunks indexed: {total}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total in collection: {collection.count()}")

    # After collecting all chunks:
    from collections import Counter
    type_counts = Counter(c["chunk_type"] for c in all_chunks)
    print("Chunk types:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")
    print()


if __name__ == "__main__":
    main()