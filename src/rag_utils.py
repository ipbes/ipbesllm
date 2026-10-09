"""Shared helpers for pdf_index.py, pdf_rag.py and pdf_query.py.

Embedding goes through the ollama client directly, with an explicit timeout
and exponential backoff, and the vectors are handed to Chroma. (Chroma's own
OllamaEmbeddingFunction uses a short default timeout, which is what made
large batches fail with httpx.ReadTimeout.)

ttl_index.py / ttl_rag.py carry their own copies of this logic; they behave
the same way and use the same environment variables.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
from functools import lru_cache

import httpx
import ollama
from chromadb.errors import NotFoundError
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from loguru import logger

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_TIMEOUT = float(os.getenv("OLLAMA_EMBED_TIMEOUT", "300"))  # seconds / request
KEEP_ALIVE = "30m"               # keep the model loaded between batches

# A batch ends when EITHER limit would be exceeded.
MAX_BATCH_ITEMS = 64
MAX_BATCH_CHARS = 16_000

# Retries (transport errors, timeouts, Ollama 5xx/429): 5s, 10s, 20s, 40s ...
MAX_RETRIES = 5
RETRY_DELAY = 5
MAX_RETRY_DELAY = 120

GET_PAGE = 5000


# ---------------------------------------------------------------------------
# Ollama clients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def embed_client():
    return ollama.Client(host=OLLAMA_URL, timeout=EMBED_TIMEOUT)


@lru_cache(maxsize=1)
def chat_client():
    return ollama.Client(host=OLLAMA_URL)


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return True  # timeouts, connection resets, Ollama restarting
    if isinstance(exc, ollama.ResponseError):
        status = getattr(exc, "status_code", 0) or 0
        return status >= 500 or status == 429
    return False


def embed_texts(client, texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            vectors = client.embed(
                model=model, input=texts, keep_alive=KEEP_ALIVE
            )["embeddings"]
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Ollama returned {len(vectors)} vectors for {len(texts)} inputs"
                )
            return vectors
        except Exception as exc:  # noqa: BLE001
            if not is_retryable(exc) or attempt == MAX_RETRIES - 1:
                raise
            delay = min(RETRY_DELAY * 2 ** attempt, MAX_RETRY_DELAY)
            logger.warning(f"Embedding failed ({type(exc).__name__}); retry in "
                           f"{delay}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
    raise AssertionError("unreachable")


@lru_cache(maxsize=256)
def embed_query(text: str, model: str = EMBED_MODEL) -> tuple[float, ...]:
    """Embed a query once (cached), with a short retry on transient errors."""
    for attempt in range(3):
        try:
            return tuple(embed_client().embed(model=model, input=[text])["embeddings"][0])
        except (httpx.TransportError, ConnectionError) as exc:
            if attempt == 2:
                raise
            delay = 2 * 2 ** attempt
            logger.warning(f"Query embedding failed ({type(exc).__name__}); "
                           f"retrying in {delay}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def embed_chunks(client, batch: list[dict], model: str, skip_failed: bool):
    """Embed a batch. Returns ([(chunk, vector)], [failed chunks]).

    If Ollama rejects the batch itself (a ResponseError that is not a transient
    5xx), retry one chunk at a time to isolate the culprit. Transport errors
    that survive the retries propagate: Ollama is down or too slow.
    """
    try:
        vectors = embed_texts(client, [c["text"] for c in batch], model)
        return list(zip(batch, vectors)), []
    except ollama.ResponseError as exc:
        if len(batch) > 1:
            print(f"  Ollama rejected a batch of {len(batch)} ({exc}); "
                  "retrying chunk by chunk to isolate it...")
            ok, bad = [], []
            for c in batch:
                o, b = embed_chunks(client, [c], model, skip_failed)
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
# Batching and fingerprints
# ---------------------------------------------------------------------------

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


def text_hash(text: str, model: str = EMBED_MODEL) -> str:
    """Fingerprint of what gets embedded; includes the model name."""
    return hashlib.sha1(f"{model}\n{text}".encode("utf-8")).hexdigest()[:16]


def metadata_hash(md: dict) -> str:
    """Fingerprint of a chunk's metadata (without the fingerprints themselves).
    A change here means the metadata must be rewritten, not re-embedded."""
    body = {k: v for k, v in md.items() if k not in ("text_hash", "meta_hash")}
    raw = json.dumps(body, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def with_fingerprints(md: dict, chunk: dict) -> dict:
    """Add text_hash and meta_hash to the metadata built for a chunk."""
    md = {k: v for k, v in md.items() if v is not None}
    md["meta_hash"] = metadata_hash(md)
    md["text_hash"] = chunk["text_hash"]
    return md


def replacing(new_md: dict, old_md: dict | None) -> dict:
    """Metadata that REPLACES the stored record. Chroma's update/upsert merge
    keys, so keys no longer produced (e.g. a dropped country_KEN flag) are
    set to None, which deletes them."""
    if not old_md:
        return new_md
    return {**{k: None for k in old_md if k not in new_md}, **new_md}


def plan_changes(chunks: list[dict], existing: dict[str, dict], build_metadata):
    """Split chunks into (to_embed, to_retag).

    to_embed: new chunks, or chunks whose text (or embedding model) changed.
    to_retag: same text, different metadata: rewrite metadata, keep vectors.
    """
    to_embed, to_retag = [], []
    for c in chunks:
        old = existing.get(c["chunk_id"])
        if old is None or old.get("text_hash") != c["text_hash"]:
            to_embed.append(c)
        elif old.get("meta_hash") != build_metadata(c)["meta_hash"]:
            to_retag.append(c)
    return to_embed, to_retag


def retag(collection, chunks: list[dict], existing: dict[str, dict],
          build_metadata) -> None:
    """Rewrite the metadata of already-embedded chunks (no re-embedding)."""
    for i in range(0, len(chunks), GET_PAGE):
        page = chunks[i:i + GET_PAGE]
        collection.update(
            ids=[c["chunk_id"] for c in page],
            metadatas=[replacing(build_metadata(c), existing.get(c["chunk_id"]))
                       for c in page],
        )


def check_scalar_metadata(chunks: list[dict], build_metadata) -> None:
    """Fail before anything is written if Chroma would reject any metadata
    (only str/int/float/bool/None are accepted; lists are not)."""
    from collections import Counter
    bad: Counter[str] = Counter()
    for chunk in chunks:
        for key, value in build_metadata(chunk).items():
            if not (value is None or isinstance(value, (str, int, float, bool))):
                bad[f"{key} ({type(value).__name__})"] += 1
    if bad:
        raise SystemExit(
            "ERROR: metadata values Chroma will reject (only str/int/float/bool "
            f"are allowed): {dict(bad)}. Nothing was changed."
        )


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------

def make_embedding_function(model: str = EMBED_MODEL):
    """Embedding function attached to a collection (so it records which model
    the collection uses). Only passes arguments this Chroma install supports."""
    params = inspect.signature(OllamaEmbeddingFunction.__init__).parameters
    kwargs = {"model_name": model}
    if "host" in params:
        kwargs["host"] = OLLAMA_URL
    else:
        kwargs["url"] = f"{OLLAMA_URL}/api/embeddings"
    if "timeout" in params:
        kwargs["timeout"] = int(EMBED_TIMEOUT)
    return OllamaEmbeddingFunction(**kwargs)


def delete_collection_if_exists(client, name: str) -> None:
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


def load_existing(collection) -> dict[str, dict]:
    """id -> stored metadata for everything in the collection."""
    existing: dict[str, dict] = {}
    offset = 0
    while True:
        page = collection.get(include=["metadatas"], limit=GET_PAGE, offset=offset)
        ids = page["ids"]
        if not ids:
            return existing
        for cid, md in zip(ids, page["metadatas"]):
            existing[cid] = dict(md or {})
        offset += len(ids)


def run_embedding_loop(collection, todo: list[dict], build_metadata, *,
                       model: str = EMBED_MODEL,
                       max_items: int = MAX_BATCH_ITEMS,
                       max_chars: int = MAX_BATCH_CHARS,
                       skip_failed: bool = False,
                       existing: dict[str, dict] | None = None) -> list[dict]:
    """Embed `todo` in batches and upsert each batch as soon as it is ready.

    `existing` (from load_existing) lets the upsert drop metadata keys the
    stored record has but the new metadata no longer produces.

    Returns the chunks that could not be embedded (only with skip_failed).
    On Ctrl-C or a fatal error it prints how to resume and exits with 1;
    everything stored so far is kept.
    """
    if not todo:
        return []

    client = embed_client()
    print(f"\nWarming up {model} (timeout {EMBED_TIMEOUT:.0f}s)...")
    try:
        embed_texts(client, ["warmup"], model)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Cannot reach Ollama at {OLLAMA_URL}: "
                         f"{type(exc).__name__}: {exc}")

    todo_chars = sum(len(c["text"]) for c in todo)
    batches = list(make_batches(todo, max_items, max_chars))
    print(f"Embedding {len(todo)} chunks in {len(batches)} batches...\n")

    failed: list[dict] = []
    t0, done_chars, stored = time.time(), 0, 0
    try:
        for n, batch in enumerate(batches, 1):
            ok, bad = embed_chunks(client, batch, model, skip_failed)
            failed += bad
            if ok:
                collection.upsert(
                    ids=[c["chunk_id"] for c, _ in ok],
                    documents=[c["text"] for c, _ in ok],
                    metadatas=[replacing(build_metadata(c),
                                         (existing or {}).get(c["chunk_id"]))
                               for c, _ in ok],
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
    return failed
