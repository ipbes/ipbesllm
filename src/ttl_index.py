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
    new or changed chunks are embedded; chunks no longer in the sources are
    removed. A crash or Ctrl-C therefore costs at most one batch: re-run.
  * Embedding is done here (ollama client, explicit timeout, exponential
    backoff) and the vectors are handed to Chroma. A batch that cannot be
    embedded stops the run unless --skip-failed is given.
  * Batches are limited by characters as well as item count, so a few long
    chunks can never produce a huge request.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import chromadb
import httpx
import ollama
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from geo import canonical_country, country_code_for_label
from ttl_loader import parse_ttl_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TTL_DIR = Path("data/ttl")

CHROMA_DIR = "chroma"
COLLECTION_NAME = "ttl_documents"

# Manifest consumed by ttl_rag.py to fan out one query per assessment.
MANIFEST_PATH = Path(CHROMA_DIR) / "assessments.json"
FAILED_PATH = Path(CHROMA_DIR) / "failed_chunks.json"

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_TIMEOUT = float(os.getenv("OLLAMA_EMBED_TIMEOUT", "300"))  # seconds per request
KEEP_ALIVE = "30m"          # keep the model loaded between batches

# Batching: a batch ends when EITHER limit would be exceeded.
MAX_BATCH_ITEMS = 64
MAX_BATCH_CHARS = 16_000
BATCH_SIZE = MAX_BATCH_ITEMS  # kept for scripts that import the old name

# Retries (transport errors, timeouts, Ollama 5xx/429): 5s, 10s, 20s, 40s, ...
MAX_RETRIES = 5
RETRY_DELAY = 5
MAX_RETRY_DELAY = 120

# Abort the run if a non-empty ipbes:country value resolves to NO ISO code.
STRICT_COUNTRY_RESOLUTION = True

_VERSION_SUFFIX_RE = re.compile(r"_v\d+$")
_COUNTRY_SEP_RE = re.compile(r"\s*[;/]\s*")
_GET_PAGE = 5000

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
# Parsing, batching, metadata
# ---------------------------------------------------------------------------

def text_hash(text: str) -> str:
    """Fingerprint of what gets embedded; includes the model name."""
    return hashlib.sha1(f"{EMBED_MODEL}\n{text}".encode("utf-8")).hexdigest()[:16]


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
            c["text_hash"] = text_hash(c["text"])
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


def make_batches(chunks, max_items: int = MAX_BATCH_ITEMS,
                 max_chars: int = MAX_BATCH_CHARS):
    """Yield lists of chunks limited by item count AND total characters.

    A single chunk larger than max_chars gets a batch of its own.
    """
    batch, size = [], 0
    for c in chunks:
        n = len(c["text"])
        if batch and (len(batch) >= max_items or size + n > max_chars):
            yield batch
            batch, size = [], 0
        batch.append(c)
        size += n
    if batch:
        yield batch


def build_metadata(chunk: dict) -> dict:
    """Chroma metadata for a chunk. Values must be str/int/float/bool: lists
    are rejected. A chunk with several countries therefore gets one boolean
    flag per ISO code (country_KEN=True, country_UGA=True, ...) for filtering,
    plus plain strings for display.
    """
    codes, names, raw, _ = country_info(chunk)
    md = {"source_type": "ttl", "country_raw": raw, "text_hash": chunk["text_hash"]}
    for field in _META_FIELDS:
        md[field] = chunk.get(field)
    if codes:
        md["country_codes"] = ";".join(codes)
        md["country_names"] = "; ".join(names)
        for code in codes:
            md[f"country_{code}"] = True
    return {k: v for k, v in md.items() if v is not None}


def check_metadata_types(chunks: list[dict]) -> None:
    """Fail before anything is written if Chroma would reject any metadata."""
    bad: Counter[str] = Counter()
    for chunk in chunks:
        for key, value in build_metadata(chunk).items():
            if not isinstance(value, (str, int, float, bool)):
                bad[f"{key} ({type(value).__name__})"] += 1
    if bad:
        raise SystemExit(
            "ERROR: metadata values Chroma will reject (only str/int/float/bool "
            f"are allowed): {dict(bad)}. Nothing was changed."
        )


# ---------------------------------------------------------------------------
# Embedding (explicit timeout + exponential backoff)
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return True  # timeouts, connection resets, Ollama restarting
    if isinstance(exc, ollama.ResponseError):
        status = getattr(exc, "status_code", 0) or 0
        return status >= 500 or status == 429
    return False


def embed_texts(client, texts: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            vectors = client.embed(
                model=EMBED_MODEL, input=texts, keep_alive=KEEP_ALIVE
            )["embeddings"]
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Ollama returned {len(vectors)} vectors for {len(texts)} inputs"
                )
            return vectors
        except Exception as exc:  # noqa: BLE001
            if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                raise
            delay = min(RETRY_DELAY * 2 ** attempt, MAX_RETRY_DELAY)
            print(f"  {type(exc).__name__}: retry in {delay}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
    raise AssertionError("unreachable")


def embed_chunks(client, batch: list[dict], skip_failed: bool):
    """Embed a batch. Returns ([(chunk, vector)], [failed chunks]).

    If Ollama rejects the batch itself (a ResponseError that is not a transient
    5xx), retry one chunk at a time to isolate the culprit. Transport errors
    that survive the retries propagate: Ollama is down or too slow.
    """
    try:
        vectors = embed_texts(client, [c["text"] for c in batch])
        return list(zip(batch, vectors)), []
    except ollama.ResponseError as exc:
        if len(batch) > 1:
            print(f"  Ollama rejected a batch of {len(batch)} ({exc}); "
                  "retrying chunk by chunk to isolate it...")
            ok, bad = [], []
            for c in batch:
                o, b = embed_chunks(client, [c], skip_failed)
                ok += o
                bad += b
            return ok, bad
        c = batch[0]
        msg = f"Could not embed {c['chunk_id']} ({len(c['text'])} chars): {exc}"
        if not skip_failed:
            raise RuntimeError(msg) from exc
        print(f"  SKIPPING: {msg}")
        return [], [c]


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------

def make_embedding_function():
    """Embedding function attached to the collection (used for queries).

    Chroma versions differ in the OllamaEmbeddingFunction signature, so only
    pass the arguments this install supports.
    """
    params = inspect.signature(OllamaEmbeddingFunction.__init__).parameters
    kwargs = {"model_name": EMBED_MODEL}
    if "host" in params:
        kwargs["host"] = OLLAMA_URL
    else:
        kwargs["url"] = f"{OLLAMA_URL}/api/embeddings"
    if "timeout" in params:
        kwargs["timeout"] = int(EMBED_TIMEOUT)
    return OllamaEmbeddingFunction(**kwargs)


def _delete_collection_if_exists(client, name: str) -> None:
    try:
        client.delete_collection(name=name)
        print(f"  Existing collection '{name}' deleted.")
    except NotFoundError:
        print(f"  Collection '{name}' did not exist.")
    except ValueError as exc:
        if "does not exist" in str(exc).lower():
            print(f"  Collection '{name}' did not exist.")
        else:
            raise


def open_collection(client, rebuild: bool):
    if rebuild:
        print(f"Rebuilding: deleting collection '{COLLECTION_NAME}'")
        _delete_collection_if_exists(client, COLLECTION_NAME)
    try:
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=make_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    except ValueError as exc:
        raise SystemExit(
            f"Could not open the existing collection ({exc}).\n"
            "If this is an embedding-function conflict, re-run with --rebuild."
        )


def load_existing(collection) -> dict[str, str]:
    """id -> text_hash for everything already stored ('' for legacy entries)."""
    existing: dict[str, str] = {}
    offset = 0
    while True:
        page = collection.get(include=["metadatas"], limit=_GET_PAGE, offset=offset)
        ids = page["ids"]
        if not ids:
            return existing
        for cid, md in zip(ids, page["metadatas"]):
            existing[cid] = (md or {}).get("text_hash", "")
        offset += len(ids)


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

    check_metadata_types(chunks)

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
    todo = [c for c in chunks if existing.get(c["chunk_id"]) != c["text_hash"]]
    todo_chars = sum(len(c["text"]) for c in todo)
    print(f"\nAlready indexed and unchanged: {len(chunks) - len(todo)}")
    print(f"To embed (new or changed):     {len(todo)}  ({todo_chars:,} chars)")
    print(f"Stale (no longer in sources):  {len(stale)}")

    if stale:
        for i in range(0, len(stale), _GET_PAGE):
            collection.delete(ids=stale[i:i + _GET_PAGE])
        print(f"  Removed {len(stale)} stale chunks.")

    # Incomplete until the very end; ttl_rag.py only reads "assessments".
    write_manifest(assessments, False, skip_types, len(chunks))

    # --- Embed and store -------------------------------------------------
    failed: list[dict] = []
    if todo:
        ollama_client = ollama.Client(host=OLLAMA_URL, timeout=EMBED_TIMEOUT)
        print(f"\nWarming up {EMBED_MODEL} (timeout {EMBED_TIMEOUT:.0f}s)...")
        try:
            embed_texts(ollama_client, ["warmup"])
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Cannot reach Ollama at {OLLAMA_URL}: "
                             f"{type(exc).__name__}: {exc}")

        batches = list(make_batches(todo, args.max_items, args.max_chars))
        print(f"Embedding {len(todo)} chunks in {len(batches)} batches...\n")
        t0, done_chars, stored = time.time(), 0, 0
        try:
            for n, batch in enumerate(batches, 1):
                ok, bad = embed_chunks(ollama_client, batch, args.skip_failed)
                failed += bad
                if ok:
                    collection.upsert(
                        ids=[c["chunk_id"] for c, _ in ok],
                        documents=[c["text"] for c, _ in ok],
                        metadatas=[build_metadata(c) for c, _ in ok],
                        embeddings=[v for _, v in ok],
                    )
                    stored += len(ok)
                done_chars += sum(len(c["text"]) for c in batch)
                elapsed = time.time() - t0
                eta = elapsed / done_chars * (todo_chars - done_chars)
                print(f"  batch {n}/{len(batches)}  stored {stored}/{len(todo)}  "
                      f"elapsed {elapsed / 60:.1f}m  ETA {eta / 60:.1f}m")
        except (KeyboardInterrupt, RuntimeError, httpx.TransportError,
                ConnectionError, ollama.ResponseError) as exc:
            print(f"\nSTOPPED after storing {stored} chunks this run "
                  f"({type(exc).__name__}: {exc}).")
            print("Progress is saved. Re-run the same command to resume.")
            sys.exit(1)

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