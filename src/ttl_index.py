"""Index IPBES TTL chunks into Chroma. Resumable, strict, and batch-size safe.

Run from the project root (same as before):

    PYTHONPATH=src python src/ttl_index.py                    # resume / incremental
    PYTHONPATH=src python src/ttl_index.py --rebuild          # wipe and start over
    PYTHONPATH=src python src/ttl_index.py --skip-types ref   # leave references out
    PYTHONPATH=src python src/ttl_index.py --dry-run          # parse + validate only

Behaviour:
  * Everything is parsed and validated BEFORE anything is deleted or written.
  * By default the existing collection is kept. A chunk is skipped when the
    collection already holds its id with the same text (and embedding model);
    new or changed chunks are embedded; chunks whose text is unchanged but
    whose metadata changed (e.g. after an ipbes-geo.rdf update) get their
    metadata rewritten without re-embedding; chunks no longer in the sources
    are removed. A crash or Ctrl-C therefore costs at most one batch: re-run.
  * Embedding goes through rag_utils (ollama client, explicit timeout,
    exponential backoff) and the vectors are handed to Chroma. A batch that cannot be
    embedded stops the run unless --skip-failed is given.
  * Batches are limited by characters as well as item count, so a few long
    chunks can never produce a huge request.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import chromadb

from geo import canonical_country, country_code_for_label
from rag_utils import (
    EMBED_MODEL, GET_PAGE, MAX_BATCH_CHARS, MAX_BATCH_ITEMS,
    check_scalar_metadata, delete_collection_if_exists, load_existing,
    make_batches, make_embedding_function, plan_changes, retag,
    run_embedding_loop, text_hash, with_fingerprints,
)
from settings import CHROMA_DIR, TTL_DIR
from ttl_loader import parse_ttl_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COLLECTION_NAME = "ttl_documents"

# Manifest consumed by ttl_rag.py to fan out one query per assessment.
MANIFEST_PATH = Path(CHROMA_DIR) / "assessments.json"
FAILED_PATH = Path(CHROMA_DIR) / "failed_chunks.json"

# Abort the run if a non-empty ipbes:country value resolves to NO ISO code.
STRICT_COUNTRY_RESOLUTION = True

_VERSION_SUFFIX_RE = re.compile(r"_v\d+$")
_COUNTRY_SEP_RE = re.compile(r"\s*[;/]\s*")

_META_FIELDS = (
    "source_file", "chunk_type", "document_type", "title", "date", "language",
    "subtype", "number", "division", "subdivision", "paragraph", "eId",
    "xpath", "identifier", "qualifier", "assessment",
)


def assessment_id_from_path(path: Path) -> str:
    """GA1_v09.ttl -> "GA1", IAS_v04.ttl -> "IAS", LDR_v01.ttl -> "LDR"."""
    return _VERSION_SUFFIX_RE.sub("", path.stem)


# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _resolve_countries(raw: str):
    """Split on ';' or '/', resolve each part to an ISO alpha-3 code.

    Returns (codes, names, missed): codes and names are parallel tuples in
    order of appearance; `missed` holds the parts that did not resolve.
    """
    codes, names, missed = [], [], []
    for part in _COUNTRY_SEP_RE.split(raw.strip()):
        part = part.strip()
        if not part:
            continue
        code = country_code_for_label(part)
        if code is None:
            missed.append(part)
        elif code not in codes:
            codes.append(code)
            names.append(canonical_country(part) or code)
    return tuple(codes), tuple(names), tuple(missed)


def country_info(chunk: dict):
    """(codes, names, raw, missed) for a chunk; raw is None when empty."""
    value = chunk.get("country")
    raw = "" if value is None else str(value).strip()
    if not raw:
        return (), (), None, ()
    codes, names, missed = _resolve_countries(raw)
    return codes, names, raw, missed


def validate_countries(chunks: list[dict]):
    """Return (unresolved, partial) Counters of raw country values."""
    unresolved: Counter[str] = Counter()
    partial: Counter[str] = Counter()
    for chunk in chunks:
        codes, _, raw, missed = country_info(chunk)
        if raw is None:
            continue
        if not codes:
            unresolved[raw] += 1
        elif missed:
            partial[f"{raw!r}  (unresolved part: {', '.join(missed)})"] += 1
    return unresolved, partial


def report_countries(unresolved: Counter, partial: Counter) -> None:
    bar = "!" * 60
    if unresolved:
        print(f"\n{bar}\nUNRESOLVED COUNTRY VALUES (no ISO code found)\n{bar}")
        print(f"{sum(unresolved.values())} chunk(s) will have no country flags and "
              "will NOT match any country filter:")
        for raw, n in unresolved.most_common():
            print(f"  {n:5d}  {raw!r}")
        print(bar)
    if partial:
        print(f"\nPARTIALLY RESOLVED country values (the unresolved part is dropped):")
        for desc, n in partial.most_common():
            print(f"  {n:5d}  {desc}")
    if unresolved or partial:
        print("\n(Empty country values are normal and are not reported.)\n")


# ---------------------------------------------------------------------------
# Parsing and metadata
# ---------------------------------------------------------------------------

def load_chunks(skip_types: set[str]) -> tuple[list[dict], set[str]]:
    ttl_files = sorted(TTL_DIR.glob("*.ttl"))
    print(f"TTL directory: {TTL_DIR.resolve()}")
    print(f"TTL files found: {len(ttl_files)}")
    if not ttl_files:
        raise SystemExit("ERROR: No TTL files found.")

    chunks: list[dict] = []
    assessments: set[str] = set()
    for path in ttl_files:
        aid = assessment_id_from_path(path)
        assessments.add(aid)
        print(f"Processing: {path.name}  (assessment={aid})")
        parsed = parse_ttl_file(str(path))
        for c in parsed:
            c["assessment"] = aid
            c["text_hash"] = text_hash(c["text"], EMBED_MODEL)
        print(f"  Chunks extracted: {len(parsed)}")
        chunks.extend(parsed)

    if skip_types:
        before = len(chunks)
        chunks = [c for c in chunks if c["chunk_type"] not in skip_types]
        print(f"\nSkipping chunk types {sorted(skip_types)}: "
              f"{before - len(chunks)} chunks left out")
    if not chunks:
        raise SystemExit("ERROR: No chunks were extracted.")

    dups = [i for i, n in Counter(c["chunk_id"] for c in chunks).items() if n > 1]
    if dups:
        raise SystemExit(f"ERROR: {len(dups)} duplicate chunk_ids, e.g. {dups[:5]}")
    return chunks, assessments


def build_metadata(chunk: dict) -> dict:
    """Chroma metadata for a chunk. Values must be str/int/float/bool: lists
    are rejected. A chunk with several countries therefore gets one boolean
    flag per ISO code (country_KEN=True, country_UGA=True, ...) for filtering,
    plus plain strings for display.
    """
    codes, names, raw, _ = country_info(chunk)
    md = {"source_type": "ttl", "country_raw": raw}
    for field in _META_FIELDS:
        md[field] = chunk.get(field)
    if codes:
        md["country_codes"] = ";".join(codes)
        md["country_names"] = "; ".join(names)
        for code in codes:
            md[f"country_{code}"] = True
    return with_fingerprints(md, chunk)


def open_collection(client, rebuild: bool):
    if rebuild:
        print(f"Rebuilding: deleting collection '{COLLECTION_NAME}'")
        delete_collection_if_exists(client, COLLECTION_NAME)
    try:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=make_embedding_function(EMBED_MODEL),
            metadata={"hnsw:space": "cosine"},
        )
    except ValueError as exc:
        raise SystemExit(
            f"Could not open the existing collection ({exc}).\n"
            "If this is an embedding-function conflict, re-run with --rebuild."
        )


def write_manifest(assessments: set[str], complete: bool, skip_types: set[str],
                   chunk_count: int) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps({
        "assessments": sorted(assessments),
        "collection": COLLECTION_NAME,
        "embed_model": EMBED_MODEL,
        "country_key": "country_<ISO3>",        # boolean flag per country, e.g. country_KEN
        "complete": complete,
        "skipped_chunk_types": sorted(skip_types),
        "chunks": chunk_count,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the collection and re-embed everything")
    ap.add_argument("--skip-types", nargs="*", default=[], metavar="TYPE",
                    help="chunk types to leave out of the index, e.g. ref")
    ap.add_argument("--skip-failed", action="store_true",
                    help="log chunks Ollama rejects and carry on (default: stop)")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse and validate, then show the plan; write nothing")
    ap.add_argument("--max-items", type=int, default=MAX_BATCH_ITEMS)
    ap.add_argument("--max-chars", type=int, default=MAX_BATCH_CHARS)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    skip_types = set(args.skip_types)

    print("Starting TTL indexing...")
    print(f"Current directory: {Path.cwd()}")
    chunks, assessments = load_chunks(skip_types)
    total_chars = sum(len(c["text"]) for c in chunks)
    print(f"\nTotal chunks: {len(chunks)}  ({total_chars:,} chars)")
    print(f"Assessments: {sorted(assessments)}\n")

    # --- Validate BEFORE touching Chroma ---------------------------------
    unresolved, partial = validate_countries(chunks)
    report_countries(unresolved, partial)
    if unresolved and STRICT_COUNTRY_RESOLUTION:
        raise SystemExit("Aborting: STRICT_COUNTRY_RESOLUTION is on. "
                         "Fix the source TTLs (list above) and re-run. "
                         "Nothing was changed.")

    check_scalar_metadata(chunks, build_metadata)

    if args.dry_run:
        n = sum(1 for _ in make_batches(chunks, args.max_items, args.max_chars))
        print(f"Dry run: {n} batches of up to {args.max_items} chunks / "
              f"{args.max_chars:,} chars. Nothing written.")
        return

    # --- Prepare the collection ------------------------------------------
    print(f"Connecting to Chroma ({CHROMA_DIR})...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = open_collection(client, args.rebuild)

    existing = load_existing(collection)
    current_ids = {c["chunk_id"] for c in chunks}
    stale = [i for i in existing if i not in current_ids]
    todo, to_retag = plan_changes(chunks, existing, build_metadata)
    todo_chars = sum(len(c["text"]) for c in todo)
    print(f"\nAlready indexed and unchanged: {len(chunks) - len(todo) - len(to_retag)}")
    print(f"To embed (new or changed):     {len(todo)}  ({todo_chars:,} chars)")
    print(f"Metadata changed only:         {len(to_retag)}")
    print(f"Stale (no longer in sources):  {len(stale)}")

    # Incomplete until the very end; ttl_rag.py only reads "assessments".
    write_manifest(assessments, False, skip_types, len(chunks))

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

    write_manifest(assessments, True, skip_types, len(chunks))
    print(f"\nWrote assessment manifest: {MANIFEST_PATH}")

    print("\n" + "=" * 60)
    print("TTL indexing complete.")
    print("=" * 60)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total in collection: {collection.count()} (expected {len(chunks)})")
    print(f"Assessments: {sorted(assessments)}")
    print("Chunk types:")
    for t, n in sorted(Counter(c["chunk_type"] for c in chunks).items()):
        print(f"  {t}: {n}")
    print("Chunks per assessment:")
    for a, n in sorted(Counter(c["assessment"] for c in chunks).items()):
        print(f"  {a}: {n}")
    with_country = sum(1 for c in chunks if country_info(c)[0])
    print(f"\nChunks with a country: {with_country}/{len(chunks)} "
          "(most chunk types legitimately have none)")
    if skip_types:
        print(f"Left out of the index: chunk types {sorted(skip_types)}")


if __name__ == "__main__":
    main()