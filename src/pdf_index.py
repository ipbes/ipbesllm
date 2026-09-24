from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from pdf_loader import extract_pdf_chunks


PDF_DIR = Path("data/pdf")
CHROMA_DIR = "chroma"
COLLECTION_NAME = "pdf_documents"

EMBED_MODEL = "nomic-embed-text"

BATCH_SIZE = 25


def main():
    print("Starting PDF indexing...")
    print(f"Current directory: {Path.cwd()}")
    print(f"PDF directory: {PDF_DIR.resolve()}")
    print()

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    print(f"PDF files found: {len(pdf_files)}")

    for pdf in pdf_files:
        print(f"  - {pdf}")

    print()

    if not pdf_files:
        print("ERROR: No PDF files found.")
        return

    print("Connecting to Chroma...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    embedding_function = OllamaEmbeddingFunction(
        model_name=EMBED_MODEL,
        url="http://localhost:11434/api/embeddings",
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    print(f"Collection: {COLLECTION_NAME}")
    print()

    all_chunks = []

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")

        chunks = extract_pdf_chunks(
            str(pdf_path)
        )

        print(f"  Chunks extracted: {len(chunks)}")

        all_chunks.extend(chunks)

    print()
    print(f"Total chunks: {len(all_chunks)}")
    print()

    if not all_chunks:
        print("ERROR: No text was extracted from the PDFs.")
        print()
        print(
               "The PDF may be scanned/image-based, "
               "or the PDF extraction needs investigation."
            )
        return

    print("Creating embeddings in batches...")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    total = len(all_chunks)

    for start in range(0, total, BATCH_SIZE):

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

        metadatas = [
            {
                "source_type": "pdf",
                "source_file": chunk["source_file"],
                "page": chunk["page"],
            }
            for chunk in batch
        ]

        print(
            f"Creating embeddings and storing in Chroma..."
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
    print("PDF indexing complete.")
    print("=" * 60)
    print(f"Chunks indexed: {total}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total in collection: {collection.count()}")


if __name__ == "__main__":
    main()