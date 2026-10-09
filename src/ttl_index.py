import json
import re
import time
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from httpx import ReadTimeout

from ttl_loader import parse_ttl_file
from geo import canonical_country, country_code_for_label


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

# Set to True to abort the whole run if any ipbes:country value in the
# source TTLs cannot be resolved to an ISO alpha-3 code. Leave True in
# production: a silent drop means a record that no country filter can
# ever match.
STRICT_COUNTRY_RESOLUTION = True


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


# ---------------------------------------------------------------------------
# Country normalisation
# ---------------------------------------------------------------------------

def _resolve_country(raw) -> tuple[str | None, str | None, str | None]:
    """
    Normalise a raw ipbes:country value.

    Returns (code, name, raw):
      code  — ISO alpha-3, or None if the raw value is unrecognised.
      name  — human-readable canonical name, or None.
      raw   — the original value, always preserved for provenance.

    The index is keyed on `code`. `name` is for display only.
    """
    raw = ("" if raw is None else str(raw)).strip() or None
    if raw is None:
        return None, None, None

    code = country_code_for_label(raw)
    if code is None:
        return None, None, raw

    return code, canonical_country(raw), raw


def _validate_countries(all_chunks: list[dict]) -> dict[str, int]:
    """
    Collect every distinct raw country value, report unresolved ones.

    Returns a Counter of unresolved raw values (empty if all resolve).
    """
    unresolved: Counter[str] = Counter()
    seen: dict[str, str | None] = {}
    for chunk in all_chunks:
        raw = chunk.get("country")
        raw_key = ("" if raw is None else str(raw)).strip()
        if raw_key in seen:
            if seen[raw_key] is None:
                unresolved[raw_key] += 1
            continue
        code = country_code_for_label(raw_key) if raw_key else None
        seen[raw_key] = code
        if raw_key and code is None:
            unresolved[raw_key] += 1
    return unresolved


def _report_unresolved(unresolved: Counter[str], total: int) -> None:
    print()
    print("!" * 60)
    print("UNRESOLVED COUNTRY VALUES")
    print("!" * 60)
    n = sum(unresolved.values())
    print(f"{n} chunk(s) carry a country value that is not a known ISO code:")
    for raw, count in unresolved.most_common():
        print(f"  {count:5d}  {raw!r}")
    print()
    print("These chunks will be indexed with country_code = None and will")
    print("NOT match any country filter. Fix the source TTLs and re-run.")
    print("!" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pre-index country validation. Do this BEFORE writing anything to
    # Chroma, so an unresolved code doesn't produce a half-built index.
    # ------------------------------------------------------------------
    unresolved = _validate_countries(all_chunks)
    if unresolved:
        _report_unresolved(unresolved, total=len(all_chunks))
        if STRICT_COUNTRY_RESOLUTION:
            print("Aborting: STRICT_COUNTRY_RESOLUTION is enabled.")
            print("Fix the source TTLs (see list above) and re-run.")
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

        metadatas = []
        for chunk in batch:
            code, name, raw = _resolve_country(chunk["country"])
            metadatas.append({
                "source_type": "ttl",
                "source_file": chunk["source_file"],
                "chunk_type": chunk["chunk_type"],
                "document_type": chunk["document_type"],
                "title": chunk["title"],
                "date": chunk["date"],
                "language": chunk["language"],

                # --- country fields -----------------------------------
                # country_code is the filter key. None if unrecognised.
                # country_name is for display only.
                # country_raw preserves the source value for provenance.
                "country_code": code,
                "country_name": name,
                "country_raw": raw,
                # Legacy key kept for one reindex cycle so any query-side
                # code still reading metadata["country"] doesn't crash.
                # Remove after you've updated ttl_rag.py to use
                # country_code.
                "country": name,

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
            })

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
                "country_key": "country_code",   # documents the filter key
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

    # Country coverage summary.
    resolved = sum(
        1 for c in all_chunks if country_code_for_label(c.get("country") or "")
    )
    unresolved_count = len(all_chunks) - resolved
    print()
    print(f"Chunks with a resolved country_code: {resolved}/{len(all_chunks)}")
    if unresolved_count:
        print(f"Chunks with country_code = None:    {unresolved_count}")
    print()


if __name__ == "__main__":
    main()