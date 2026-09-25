# src/cache.py
"""
Shared cache for the IPBES RAG pipelines.

Backends (auto-selected):
  - Redis  if REDIS_URL is set and reachable
  - Memory otherwise (process-local, lost on restart)

Public API:
  cache            -> aiocache.Cache instance
  make_key(*parts) -> deterministic sha256 key
  cached_answer    -> decorator: question -> final answer
  cached_retrieve  -> decorator: retrieval parameters -> chroma results
  invalidate_all   -> nuke everything (call after re-indexing)
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import wraps
from typing import Any, Callable

from aiocache import Cache
from aiocache.serializers import JsonSerializer
from loguru import logger


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL")  # e.g. "redis://localhost:6379/0"


def _build_cache() -> Cache:
    if REDIS_URL:
        # aiocache wants host/port separately, not a URL, so parse.
        from urllib.parse import urlparse

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


# ---------------------------------------------------------------------------
# Key builder
# ---------------------------------------------------------------------------

def make_key(*parts: Any) -> str:
    """
    Deterministic sha256 key. Sort keys so dict ordering never changes
    the key. `default=str` handles URIRefs, Paths, sets, etc.
    """
    raw = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def cached_retrieve(ttl: int = 3600):
    """
    Wrap an async retrieval function whose first arg is the question and
    whose kwargs determine the result (k, chunk_type, country_names).
    Chroma's query() returns dicts of lists -> JSON-serialisable.
    """
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(question: str, **kwargs):
            # sort country_names for stable keys
            norm = dict(kwargs)
            if norm.get("country_names"):
                norm["country_names"] = sorted(norm["country_names"])

            key = make_key("retrieve", fn.__module__, question, norm)
            hit = await cache.get(key)
            if hit is not None:
                logger.debug(f"retrieve cache HIT  q={question[:50]!r}")
                hit["_cached"] = True
                return hit

            result = await fn(question, **kwargs)
            await cache.set(key, result, ttl=ttl)
            logger.debug(f"retrieve cache MISS q={question[:50]!r}")
            return result
        return wrapper
    return deco


def cached_answer(ttl: int = 60 * 60 * 6):
    """
    Wrap an async answer function. Cache key includes the question plus
    every kwargs that changes the answer (chunk_type, country_names).
    """
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(question: str, **kwargs):
            norm = dict(kwargs)
            if norm.get("country_names"):
                norm["country_names"] = sorted(norm["country_names"])

            key = make_key("answer", fn.__module__, question, norm)
            hit = await cache.get(key)
            if hit is not None:
                logger.debug(f"answer cache HIT  q={question[:50]!r}")
                hit["_cached"] = True
                return hit

            result = await fn(question, **kwargs)
            if result.get("answer"):  # only cache successful answers
                await cache.set(key, result, ttl=ttl)
            logger.debug(f"answer cache MISS q={question[:50]!r}")
            return result
        return wrapper
    return deco


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------

async def invalidate_all() -> None:
    """Call after re-indexing (ttl_index.py / pdf_index.py / xml_index.py)."""
    await cache.clear()
    logger.info("Cache cleared.")


async def close() -> None:
    await cache.close()