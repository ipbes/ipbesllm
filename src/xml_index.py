from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from xml_loader import parse_akn_file


XML_DIR = Path("data/xml")

CHROMA_DIR = "chroma"
COLLECTION_NAME = "xml_documents"

EMBED_MODEL = "nomic-embed-text"

BATCH_SIZE = 25


def main():

    print("Starting Akoma Ntoso XML indexing...")
    print(f"Current directory: {Path.cwd()}")
    print(f"XML directory: {XML_DIR.resolve()}")
    print()

    xml_files = list(XML_DIR.glob("*.xml"))

    print(f"XML files found: {len(xml_files)}")

    for xml_file in xml_files:
        print(f"  - {xml_file}")

    print()

    if not xml_files:
        print("ERROR: No XML files found.")
        return

    print("Connecting to Chroma...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    embedding_function = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url="http://localhost:11434/api/embeddings",
    )

    # Delete the existing collection so every indexing run
    # starts with a completely fresh collection.
    print(f"Deleting existing collection: {COLLECTION_NAME}")

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )
        print("  Existing collection deleted.")
    except Exception as e:
        print(f"  Collection did not exist or could not be deleted: {e}")

    # Create a new empty collection.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    print(f"Created new collection: {COLLECTION_NAME}")
    print()

    all_chunks = []

    for xml_file in xml_files:

        print(f"Processing: {xml_file.name}")

        chunks = parse_akn_file(
            str(xml_file)
        )

        print(
            f"  Chunks extracted: {len(chunks)}"
        )

        all_chunks.extend(chunks)

    print()
    print(f"Total chunks: {len(all_chunks)}")
    print()

    if not all_chunks:
        print("ERROR: No chunks were extracted.")
        return

    print("Creating embeddings...")
    print(f"Embedding model: {EMBED_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    total = len(all_chunks)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = all_chunks[start:end]

        ids = [
            chunk["chunk_id"]
            for chunk in batch
        ]

        documents = [
            chunk["text"]
            for chunk in batch
        ]

        metadatas = []

        for chunk in batch:

            metadata = {
                "source_type": "xml",
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
                "table_row": str(chunk.get("table_row", "")),
                "table_title": chunk.get("table_title", ""),
            }

            metadatas.append(metadata)

        print(
            f"Embedding chunks "
            f"{start + 1}-{end} "
            f"of {total}..."
        )

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        print(
            f"  Stored {end}/{total}"
        )

    print()
    print("=" * 60)
    print("Akoma Ntoso XML indexing complete.")
    print("=" * 60)

    print(f"Chunks indexed: {total}")
    print(f"Collection: {COLLECTION_NAME}")
    print(
        f"Total in collection: "
        f"{collection.count()}"
    )


if __name__ == "__main__":
    main()