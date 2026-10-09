from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from typing import Any
from urllib.parse import urlparse

from aiocache import Cache
from aiocache.serializers import JsonSerializer
from loguru import logger

REDIS_URL = os.getenv("REDIS_URL")
CACHE_VERSION = os.getenv("RAG_CACHE_VERSION", "v1")
PIPELINE_ID = os.getenv("RAG_PIPELINE", "ttl")


def _build_cache() -> Cache:
    if REDIS_URL:
        p = urlparse(REDIS_URL)
        return Cache(
            Cache.REDIS,
            endpoint=p.hostname or "localhost",
            port=p.port or 6379,
            db=int((p.path or "/0").lstrip("/") or 0),
            password=p.password,
            namespace="ipbes",
            serializer=JsonSerializer(),
            timeout=2,
        )

    logger.warning("REDIS_URL not set; using in-memory cache (dev only).")
    return Cache(Cache.MEMORY, namespace="ipbes", serializer=JsonSerializer())


cache: Cache = _build_cache()

# The Redis connection is bound to the event loop it is first used on, so
# synchronous callers (Streamlit) must run every cached call on one loop:
# asyncio.run() makes a new loop per call and breaks the shared connection.
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def run_sync(coro):
    """Run a coroutine on this module's long-lived event loop and wait."""
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="cache-event-loop",
                             daemon=True).start()
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


def make_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def invalidate_all() -> None:
    # Without a namespace aiocache runs FLUSHDB on Redis (the whole database).
    await cache.clear(namespace="ipbes")
    logger.info("Cache cleared.")


async def close() -> None:
    await cache.close()
