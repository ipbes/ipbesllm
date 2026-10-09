"""Index PDF chunks into Chroma. Resumable, strict, and batch-size safe.

Run from the project root:

    PYTHONPATH=src python src/pdf_index.py                 # resume / incremental
    PYTHONPATH=src python src/pdf_index.py --rebuild       # wipe and start over
    PYTHONPATH=src python src/pdf_index.py --dry-run       # parse + validate only

Same behaviour as ttl_index.py:
  * Everything is parsed and validated BEFORE anything is deleted or written.
  * The existing collection is kept by default. A chunk is skipped when the
    collection already holds its id with the same text and embedding model;
    new or changed chunks are embedded; chunks whose text is unchanged but
    whose metadata changed get their metadata rewritten without re-embedding;
    chunks no longer in the PDFs are removed. A crash or Ctrl-C costs at most
    one batch: re-run to resume.
  * Embedding uses an explicit timeout (OLLAMA_EMBED_TIMEOUT, default 300s) with
    exponential backoff. A batch that cannot be embedded stops the run unless
    --skip-failed is given.
  * Batches are limited by characters as well as item count.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import chromadb

from pdf_loader import extract_pdf_chunks
from rag_utils import (
    EMBED_MODEL, MAX_BATCH_CHARS, MAX_BATCH_ITEMS, GET_PAGE,
    check_scalar_metadata, delete_collection_if_exists, load_existing,
    make_batches, make_embedding_function, plan_changes, retag,
    run_embedding_loop, text_hash, with_fingerprints,
)

PDF_DIR = Path("data/pdf")
CHROMA_DIR = "chroma"
COLLECTION_NAME = "pdf_documents"

# Read by pdf_rag.py: its modification time invalidates cached retrievals.
MANIFEST_PATH = Path(CHROMA_DIR) / "pdf_index.json"
FAILED_PATH = Path(CHROMA_DIR) / "pdf_failed_chunks.json"


def build_metadata(chunk: dict) -> dict:
    """Chroma metadata for a chunk (scalar values only)."""
    md = {
        "source_type": "pdf",
        "source_file": chunk["source_file"],
        "title": chunk.get("title") or Path(chunk["source_file"]).stem,
        "page": chunk["page"],
        "page_count": chunk["page_count"],
        "chunk_index": chunk["chunk_index"],
        "chunk_total": chunk["chunk_total"],
        "total_chunks": chunk["total_chunks"],
    }
    return with_fingerprints(md, chunk)


def load_chunks() -> tuple[list[dict], list[str]]:
    """Extract and fingerprint every chunk. Returns (chunks, files_with_no_text)."""
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"PDF directory: {PDF_DIR.resolve()}")
    print(f"PDF files found: {len(pdf_files)}")
    if not pdf_files:
        raise SystemExit("ERROR: No PDF files found.")

    chunks: list[dict] = []
    unreadable: dict[str, str] = {}
    empty: list[str] = []
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            extracted = extract_pdf_chunks(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            unreadable[pdf_path.name] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {unreadable[pdf_path.name]}")
            continue
        print(f"  Chunks extracted: {len(extracted)}")
        if not extracted:
            empty.append(pdf_path.name)
        for c in extracted:
            c["text_hash"] = text_hash(c["text"], EMBED_MODEL)
        chunks.extend(extracted)

    if unreadable:
        raise SystemExit(
            f"ERROR: {len(unreadable)} PDF(s) could not be read: {unreadable}. "
            "Fix or remove them and re-run. Nothing was changed."
        )
    if not chunks:
        raise SystemExit(
            "ERROR: No text was extracted from the PDFs. They may be "
            "scanned/image-based (OCR needed)."
        )

    dups = [i for i, n in Counter(c["chunk_id"] for c in chunks).items() if n > 1]
    if dups:
        raise SystemExit(f"ERROR: {len(dups)} duplicate chunk_ids, e.g. {dups[:5]}")
    return chunks, empty


def write_manifest(complete: bool, chunk_count: int, files: int) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({
        "collection": COLLECTION_NAME,
        "embed_model": EMBED_MODEL,
        "complete": complete,
        "chunks": chunk_count,
        "files": files,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))


def manifest_complete() -> bool:
    try:
        data = json.loads(MANIFEST_PATH.read_text())
    except (OSError, ValueError):
        return False
    return data.get("complete") is True and data.get("embed_model") == EMBED_MODEL


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the collection and re-embed everything")
    ap.add_argument("--skip-failed", action="store_true",
                    help="log chunks Ollama rejects and carry on (default: stop)")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and validate, then show the plan; write nothing")
    ap.add_argument("--max-items", type=int, default=MAX_BATCH_ITEMS)
    ap.add_argument("--max-chars", type=int, default=MAX_BATCH_CHARS)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print("Starting PDF indexing...")
    print(f"Current directory: {Path.cwd()}")
    chunks, empty_files = load_chunks()
    total_chars = sum(len(c["text"]) for c in chunks)
    files = len({c["source_file"] for c in chunks})
    print(f"\nTotal chunks: {len(chunks)}  ({total_chars:,} chars) from {files} file(s)")
    if empty_files:
        print(f"WARNING: no extractable text in {empty_files} "
              "(scanned/image-based? they are not indexed)")

    # --- Validate BEFORE touching Chroma ---------------------------------
    check_scalar_metadata(chunks, build_metadata)

    if args.dry_run:
        n = sum(1 for _ in make_batches(chunks, args.max_items, args.max_chars))
        print(f"\nDry run: {n} batches of up to {args.max_items} chunks / "
              f"{args.max_chars:,} chars. Nothing written.")
        return

    # --- Prepare the collection ------------------------------------------
    print(f"\nConnecting to Chroma ({CHROMA_DIR})...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if args.rebuild:
        print(f"Rebuilding: deleting collection '{COLLECTION_NAME}'")
        delete_collection_if_exists(client, COLLECTION_NAME)
    try:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=make_embedding_function(EMBED_MODEL),
            metadata={"hnsw:space": "cosine"},
        )
    except ValueError as exc:
        raise SystemExit(
            f"Could not open the existing collection ({exc}).\n"
            "If this is an embedding-function conflict, re-run with --rebuild."
        )

    existing = load_existing(collection)
    current_ids = {c["chunk_id"] for c in chunks}
    stale = [i for i in existing if i not in current_ids]
    todo, to_retag = plan_changes(chunks, existing, build_metadata)
    todo_chars = sum(len(c["text"]) for c in todo)
    print(f"\nAlready indexed and unchanged: {len(chunks) - len(todo) - len(to_retag)}")
    print(f"To embed (new or changed):     {len(todo)}  ({todo_chars:,} chars)")
    print(f"Metadata changed only:         {len(to_retag)}")
    print(f"Stale (no longer in the PDFs): {len(stale)}")

    if not (todo or to_retag or stale) and manifest_complete():
        # Leave the manifest alone: rewriting it would invalidate pdf_rag's cache.
        print("\nIndex is up to date; nothing to do.")
        return

    write_manifest(False, len(chunks), files)   # incomplete until the end

    if stale:
        for i in range(0, len(stale), GET_PAGE):
            collection.delete(ids=stale[i:i + GET_PAGE])
        print(f"  Removed {len(stale)} stale chunks.")
    if to_retag:
        retag(collection, to_retag, existing, build_metadata)
        print(f"  Rewrote metadata of {len(to_retag)} chunks.")

    # --- Embed and store -------------------------------------------------
    failed = run_embedding_loop(
        collection, todo, build_metadata,
        model=EMBED_MODEL, max_items=args.max_items, max_chars=args.max_chars,
        skip_failed=args.skip_failed, existing=existing,
    )

    # --- Wrap up ---------------------------------------------------------
    if failed:
        FAILED_PATH.write_text(json.dumps(
            [{"chunk_id": c["chunk_id"], "chars": len(c["text"])} for c in failed],
            indent=2))
        print(f"\n{len(failed)} chunk(s) could not be embedded; see {FAILED_PATH}.")
        print("Manifest left marked incomplete.")
        sys.exit(1)
    if FAILED_PATH.exists():
        FAILED_PATH.unlink()

    write_manifest(True, len(chunks), files)

    print("\n" + "=" * 60)
    print("PDF indexing complete.")
    print("=" * 60)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total in collection: {collection.count()} (expected {len(chunks)})")
    print("Chunks per file:")
    for name, n in sorted(Counter(c["source_file"] for c in chunks).items()):
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
