import json
import re
import time
from httpx import ReadTimeout  # Import the specific exception
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from ttl_loader import parse_ttl_file
from geo import canonical_country


TTL_DIR = Path("data/ttl")

CHROMA_DIR = "chroma"
COLLECTION_NAME = "ttl_documents"

# Manifest consumed by ttl_rag.py to fan out one query per assessment.
MANIFEST_PATH = Path(CHROMA_DIR) / "assessments.json"

EMBED_MODEL = "nomic-embed-text"

# Matches a trailing version suffix like "_v09" / "_v01".
_VERSION_SUFFIX_RE = re.compile(r"_v\d+$")

# Manage time out issues ...
BATCH_SIZE = 10  # Reduced from 25 to prevent timeouts
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def assessment_id_from_path(path: Path) -> str:
    """
    Derive a stable assessment ID from a TTL filename.

    Examples:
        GA1_v09.ttl   -> "GA1"
        IAS_v04.ttl   -> "IAS"
        LDR_v01.ttl   -> "LDR"
    """
    return _VERSION_SUFFIX_RE.sub("", path.stem)


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

    ttl_files = sorted(TTL_DIR.glob("*.ttl"))

    print(f"TTL files found: {len(ttl_files)}")
    for ttl in ttl_files:
        print(f"  - {ttl}  -> assessment={assessment_id_from_path(ttl)!r}")
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
    indexed_assessments: set[str] = set()

    for ttl_path in ttl_files:
        assessment_id = assessment_id_from_path(ttl_path)
        indexed_assessments.add(assessment_id)

        print(f"Processing: {ttl_path.name}  (assessment={assessment_id})")

        chunks = parse_ttl_file(str(ttl_path))

        # Tag every chunk with its assessment so it can be filtered
        # per-assessment downstream.
        for chunk in chunks:
            chunk["assessment"] = assessment_id

        print(f"  Chunks extracted: {len(chunks)}")

        all_chunks.extend(chunks)

    print()
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Assessments: {sorted(indexed_assessments)}")
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
                "country": canonical_country(chunk["country"]),
                "country_raw": chunk["country"],
                "subtype": chunk["subtype"],
                "number": chunk["number"],
                "division": chunk["division"],
                "subdivision": chunk["subdivision"],
                "paragraph": chunk["paragraph"],
                "eId": chunk["eId"],
                "xpath": chunk["xpath"],
                "identifier": chunk["identifier"],
                "qualifier": chunk["qualifier"],
                "assessment": chunk["assessment"],
            }
            for chunk in batch
        ]

        print(f"Storing chunks {start + 1}-{end} of {total}...")

        for attempt in range(MAX_RETRIES):
            try:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                )
                print(f"  Stored {end}/{total}")
                break
            except ReadTimeout:
                if attempt < MAX_RETRIES - 1:
                    print(
                        f"  Timeout — retrying in {RETRY_DELAY}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES})..."
                    )
                    time.sleep(RETRY_DELAY)
                else:
                    print(
                        f"  FAILED after {MAX_RETRIES} attempts. "
                        f"Skipping chunks {start + 1}-{end}."
                    )
            except Exception as e:
                print(f"  Unexpected error on chunks {start + 1}-{end}: {e}")
                raise

    # ------------------------------------------------------------------
    # Write the assessment manifest consumed by ttl_rag.py.
    # ------------------------------------------------------------------
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "assessments": sorted(indexed_assessments),
                "collection": COLLECTION_NAME,
                "embed_model": EMBED_MODEL,
            },
            indent=2,
        )
    )
    print(f"Wrote assessment manifest: {MANIFEST_PATH}")

    print()
    print("=" * 60)
    print("TTL indexing complete.")
    print("=" * 60)
    print(f"Chunks indexed: {total}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total in collection: {collection.count()}")
    print(f"Assessments: {sorted(indexed_assessments)}")

    type_counts = Counter(c["chunk_type"] for c in all_chunks)
    print("Chunk types:")
    for t, count in sorted(type_counts.items()):
        print(f"  {t}: {count}")

    assessment_counts = Counter(c["assessment"] for c in all_chunks)
    print("Chunks per assessment:")
    for a, count in sorted(assessment_counts.items()):
        print(f"  {a}: {count}")
    print()


if __name__ == "__main__":
    main()