"""Index Akoma Ntoso XML chunks into Chroma. Resumable, strict, and batch-size safe.

Run from the project root:

    PYTHONPATH=src python src/xml_index.py                 # resume / incremental
    PYTHONPATH=src python src/xml_index.py --rebuild       # wipe and start over
    PYTHONPATH=src python src/xml_index.py --dry-run       # parse + validate only

Same behaviour as pdf_index.py:
  * Everything is parsed and validated BEFORE anything is deleted or written,
    so a file that fails to parse leaves the existing collection untouched.
  * The existing collection is kept by default. A chunk is skipped when the
    collection already holds its id with the same text and embedding model;
    new or changed chunks are embedded; chunks whose text is unchanged but
    whose metadata changed get their metadata rewritten without re-embedding;
    chunks no longer in the XML files are removed. A crash or Ctrl-C costs at
    most one batch: re-run to resume.
  * Embedding goes through rag_utils (OLLAMA_URL, OLLAMA_EMBED_MODEL, explicit
    timeout OLLAMA_EMBED_TIMEOUT, exponential backoff). A batch that cannot be
    embedded stops the run unless --skip-failed is given.
  * Batches are limited by characters as well as item count.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import chromadb

from rag_utils import (
    EMBED_MODEL, MAX_BATCH_CHARS, MAX_BATCH_ITEMS, GET_PAGE,
    check_scalar_metadata, delete_collection_if_exists, load_existing,
    make_batches, make_embedding_function, plan_changes, retag,
    run_embedding_loop, text_hash, with_fingerprints,
)
from settings import CHROMA_DIR, XML_DIR
from xml_loader import parse_akn_file

COLLECTION_NAME = "xml_documents"

FAILED_PATH = Path(CHROMA_DIR) / "xml_failed_chunks.json"


def build_metadata(chunk: dict) -> dict:
    """Chroma metadata for a chunk (scalar values only)."""
    md = {
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
    return with_fingerprints(md, chunk)


def load_chunks() -> list[dict]:
    """Parse and fingerprint every chunk. Nothing is written here."""
    xml_files = sorted(XML_DIR.glob("*.xml"))
    print(f"XML directory: {XML_DIR.resolve()}")
    print(f"XML files found: {len(xml_files)}")
    if not xml_files:
        raise SystemExit("ERROR: No XML files found.")

    chunks: list[dict] = []
    unreadable: dict[str, str] = {}
    for xml_path in xml_files:
        print(f"Processing: {xml_path.name}")
        try:
            parsed = parse_akn_file(str(xml_path))
        except Exception as exc:  # noqa: BLE001
            unreadable[xml_path.name] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {unreadable[xml_path.name]}")
            continue
        print(f"  Chunks extracted: {len(parsed)}")
        for c in parsed:
            c["text_hash"] = text_hash(c["text"], EMBED_MODEL)
        chunks.extend(parsed)

    if unreadable:
        raise SystemExit(
            f"ERROR: {len(unreadable)} XML file(s) could not be parsed: {unreadable}. "
            "Fix or remove them and re-run. Nothing was changed."
        )
    if not chunks:
        raise SystemExit("ERROR: No chunks were extracted. Nothing was changed.")

    dups = [i for i, n in Counter(c["chunk_id"] for c in chunks).items() if n > 1]
    if dups:
        raise SystemExit(f"ERROR: {len(dups)} duplicate chunk_ids, e.g. {dups[:5]}")
    return chunks


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

    print("Starting Akoma Ntoso XML indexing...")
    print(f"Current directory: {Path.cwd()}")
    chunks = load_chunks()
    total_chars = sum(len(c["text"]) for c in chunks)
    files = len({c["source_file"] for c in chunks})
    print(f"\nTotal chunks: {len(chunks)}  ({total_chars:,} chars) from {files} file(s)")

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
    print(f"Stale (no longer in the XML):  {len(stale)}")

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
        FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)
        FAILED_PATH.write_text(json.dumps(
            [{"chunk_id": c["chunk_id"], "chars": len(c["text"])} for c in failed],
            indent=2))
        print(f"\n{len(failed)} chunk(s) could not be embedded; see {FAILED_PATH}.")
        sys.exit(1)
    if FAILED_PATH.exists():
        FAILED_PATH.unlink()

    print("\n" + "=" * 60)
    print("Akoma Ntoso XML indexing complete.")
    print("=" * 60)
    print(f"Collection: {COLLECTION_NAME} ({EMBED_MODEL})")
    print(f"Total in collection: {collection.count()} (expected {len(chunks)})")
    print("Chunks per file:")
    for name, n in sorted(Counter(c["source_file"] for c in chunks).items()):
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
