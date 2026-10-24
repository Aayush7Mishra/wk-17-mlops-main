"""Asynchronous prompt/response cache backends.

The Redis dependency is deliberately imported only by :meth:`from_url`.  The
application and its tests can therefore use the in-memory backend without
installing or connecting to Redis.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any


class CacheBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested cache backend cannot be created."""


def make_cache_key(
    prompt: str,
    *,
    model: str,
    system_prompt: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    namespace: str = "assistant",
) -> str:
    """Return a deterministic, opaque key for one model invocation.

    JSON canonicalization makes mapping insertion order irrelevant.  Prompt
    whitespace remains significant, which prevents semantically different
    requests from sharing an entry.
    """

    if not namespace or ":" in namespace:
        raise ValueError("namespace must be non-empty and cannot contain ':'")

    payload = {
        "model": model,
        "parameters": dict(parameters or {}),
        "prompt": prompt,
        "system_prompt": system_prompt,
    }
    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("cache-key inputs must be JSON serializable") from exc

    digest = hashlib.sha256(serialized).hexdigest()
    return f"{namespace}:{digest}"


class PromptResponseCache(ABC):
    """Minimal asynchronous interface shared by all cache backends."""

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """Return a cached response, or ``None`` when the entry is absent."""

    @abstractmethod
    async def set(
        self,
        key: str,
        response: str,
        *,
        ttl_seconds: float | None = None,
    ) -> None:
        """Store a response using the backend default TTL when none is given."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a cached response if it exists."""

    @abstractmethod
    async def increment(self, key: str) -> int:
        """Atomically increment and return a persistent integer value."""

    async def close(self) -> None:
        """Release backend resources; in-memory implementations need no work."""

        return None


class InMemoryTTLCache(PromptResponseCache):
    """Concurrency-safe process-local TTL cache.

    This backend is intentionally simple and has no eviction policy beyond
    expiry.  It is appropriate for development or a single application
    process.  Use :class:`RedisPromptResponseCache` for shared deployments.
    """

    def __init__(
        self,
        default_ttl_seconds: float | None = 300.0,
        *,
        max_entries: int = 2048,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _validate_ttl(default_ttl_seconds, name="default_ttl_seconds")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.default_ttl_seconds = default_ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._entries: dict[str, tuple[str, float | None]] = {}
        self._lock = asyncio.Lock()

    def _prune_expired(self, now: float) -> None:
        expired = [
            key
            for key, (_, expires_at) in self._entries.items()
            if expires_at is not None and expires_at <= now
        ]
        for key in expired:
            self._entries.pop(key, None)

    async def get(self, key: str) -> str | None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            response, expires_at = entry
            if expires_at is not None and expires_at <= self._clock():
                self._entries.pop(key, None)
                return None
            # Dict insertion order provides a lightweight LRU eviction order.
            self._entries.pop(key)
            self._entries[key] = entry
            return response

    async def set(
        self,
        key: str,
        response: str,
        *,
        ttl_seconds: float | None = None,
    ) -> None:
        if not isinstance(response, str):
            raise TypeError("response must be a string")

        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        _validate_ttl(ttl, name="ttl_seconds")
        async with self._lock:
            if ttl is not None and ttl <= 0:
                self._entries.pop(key, None)
                return
            self._prune_expired(self._clock())
            self._entries.pop(key, None)
            while len(self._entries) >= self.max_entries:
                self._entries.pop(next(iter(self._entries)))
            expires_at = None if ttl is None else self._clock() + ttl
            self._entries[key] = (response, expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._entries.pop(key, None)

    async def increment(self, key: str) -> int:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[1] is not None and entry[1] <= self._clock():
                entry = None
            try:
                value = int(entry[0]) + 1 if entry is not None else 1
            except ValueError as exc:
                raise ValueError("cached counter is not an integer") from exc
            self._prune_expired(self._clock())
            self._entries.pop(key, None)
            while len(self._entries) >= self.max_entries:
                self._entries.pop(next(iter(self._entries)))
            self._entries[key] = (str(value), None)
            return value

    async def clear(self) -> None:
        """Remove every entry from this process-local cache."""

        async with self._lock:
            self._entries.clear()

    async def size(self) -> int:
        """Return the number of live entries, pruning expired entries first."""

        async with self._lock:
            self._prune_expired(self._clock())
            return len(self._entries)


class RedisPromptResponseCache(PromptResponseCache):
    """Redis-backed cache that is safe to share across application workers."""

    def __init__(
        self,
        client: Any,
        *,
        default_ttl_seconds: float | None = 300.0,
        key_prefix: str = "assistant-cache",
        owns_client: bool = False,
    ) -> None:
        _validate_ttl(default_ttl_seconds, name="default_ttl_seconds")
        if not key_prefix:
            raise ValueError("key_prefix must not be empty")
        self._client = client
        self.default_ttl_seconds = default_ttl_seconds
        self.key_prefix = key_prefix.rstrip(":")
        self._owns_client = owns_client

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        default_ttl_seconds: float | None = 300.0,
        key_prefix: str = "assistant-cache",
        timeout_seconds: float = 2.0,
    ) -> RedisPromptResponseCache:
        """Create a backend without connecting until its first operation."""

        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise CacheBackendUnavailable(
                "Redis caching requires the optional 'redis' package"
            ) from exc

        client = Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )
        return cls(
            client,
            default_ttl_seconds=default_ttl_seconds,
            key_prefix=key_prefix,
            owns_client=True,
        )

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    async def ping(self) -> None:
        await self._client.ping()

    async def get(self, key: str) -> str | None:
        response = await self._client.get(self._key(key))
        if response is None:
            return None
        if isinstance(response, bytes):
            return response.decode("utf-8")
        return str(response)

    async def set(
        self,
        key: str,
        response: str,
        *,
        ttl_seconds: float | None = None,
    ) -> None:
        if not isinstance(response, str):
            raise TypeError("response must be a string")

        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        _validate_ttl(ttl, name="ttl_seconds")
        redis_key = self._key(key)
        if ttl is not None and ttl <= 0:
            await self._client.delete(redis_key)
            return
        if ttl is None:
            await self._client.set(redis_key, response)
            return

        ttl_milliseconds = max(1, math.ceil(ttl * 1000))
        await self._client.psetex(redis_key, ttl_milliseconds, response)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._key(key))

    async def increment(self, key: str) -> int:
        return int(await self._client.incr(self._key(key)))

    async def close(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "aclose", None)
        if close is None:  # redis-py < 5 compatibility
            close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


async def create_cache(
    *,
    redis_url: str | None = None,
    redis_client: Any | None = None,
    default_ttl_seconds: float | None = 300.0,
    key_prefix: str = "assistant-cache",
    check_connection: bool = True,
    fallback_to_memory: bool = True,
    timeout_seconds: float = 2.0,
) -> PromptResponseCache:
    """Build the requested backend, optionally degrading to process-local memory.

    ``redis_client`` primarily supports dependency injection and takes precedence
    over ``redis_url``.  A connection is only checked when Redis was requested.
    """

    if redis_client is None and redis_url is None:
        return InMemoryTTLCache(default_ttl_seconds)

    try:
        cache = (
            RedisPromptResponseCache(
                redis_client,
                default_ttl_seconds=default_ttl_seconds,
                key_prefix=key_prefix,
            )
            if redis_client is not None
            else RedisPromptResponseCache.from_url(
                redis_url or "",
                default_ttl_seconds=default_ttl_seconds,
                key_prefix=key_prefix,
                timeout_seconds=timeout_seconds,
            )
        )
        if check_connection:
            await cache.ping()
        return cache
    except Exception:
        if not fallback_to_memory:
            raise
        return InMemoryTTLCache(default_ttl_seconds)


def _validate_ttl(ttl_seconds: float | None, *, name: str) -> None:
    if ttl_seconds is None:
        return
    if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    if not math.isfinite(ttl_seconds):
        raise ValueError(f"{name} must be finite")


__all__ = [
    "CacheBackendUnavailable",
    "InMemoryTTLCache",
    "PromptResponseCache",
    "RedisPromptResponseCache",
    "create_cache",
    "make_cache_key",
]
