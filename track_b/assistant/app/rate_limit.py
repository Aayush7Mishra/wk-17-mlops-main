"""Asynchronous sliding-window rate limiters for the assistant API."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import secrets
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """The outcome of one rate-limit check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after: float

    def as_headers(self) -> dict[str, str]:
        """Return HTTP headers suitable for a FastAPI response."""

        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if not self.allowed:
            headers["Retry-After"] = str(max(1, math.ceil(self.retry_after)))
        return headers


class RateLimiter(ABC):
    """Backend-independent asynchronous rate-limiter interface."""

    @abstractmethod
    async def check(self, identifier: str, *, cost: int = 1) -> RateLimitDecision:
        """Consume one request allowance for ``identifier`` when available."""

    async def allow(self, identifier: str, *, cost: int = 1) -> RateLimitDecision:
        """Readable alias for :meth:`check`."""

        return await self.check(identifier, cost=cost)

    async def close(self) -> None:
        """Release backend resources; in-memory implementations need no work."""

        return None


class InMemorySlidingWindowRateLimiter(RateLimiter):
    """Exact sliding-window limiter for a single application process."""

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        max_identifiers: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _validate_configuration(limit, window_seconds)
        if (
            isinstance(max_identifiers, bool)
            or not isinstance(max_identifiers, int)
            or max_identifiers < 1
        ):
            raise ValueError("max_identifiers must be a positive integer")
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_identifiers = max_identifiers
        self._clock = clock
        self._requests: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, identifier: str, *, cost: int = 1) -> RateLimitDecision:
        if not identifier:
            raise ValueError("identifier must not be empty")
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < 1:
            raise ValueError("cost must be a positive integer")

        async with self._lock:
            now = self._clock()
            cutoff = now - self.window_seconds
            if identifier not in self._requests and len(self._requests) >= self.max_identifiers:
                stale = [
                    key
                    for key, timestamps in self._requests.items()
                    if not timestamps or timestamps[-1] <= cutoff
                ]
                for key in stale:
                    self._requests.pop(key, None)
                while len(self._requests) >= self.max_identifiers:
                    oldest = min(
                        self._requests,
                        key=lambda key: self._requests[key][-1],
                    )
                    self._requests.pop(oldest, None)
            requests = self._requests.setdefault(identifier, deque())
            while requests and requests[0] <= cutoff:
                requests.popleft()

            if cost > self.limit:
                return RateLimitDecision(
                    allowed=False,
                    limit=self.limit,
                    remaining=0,
                    retry_after=self.window_seconds,
                )
            if len(requests) + cost > self.limit:
                must_expire = len(requests) + cost - self.limit
                retry_after = max(
                    0.0,
                    requests[must_expire - 1] + self.window_seconds - now,
                )
                return RateLimitDecision(
                    allowed=False,
                    limit=self.limit,
                    remaining=max(0, self.limit - len(requests)),
                    retry_after=retry_after,
                )

            requests.extend([now] * cost)
            return RateLimitDecision(
                allowed=True,
                limit=self.limit,
                remaining=self.limit - len(requests),
                retry_after=0.0,
            )

    async def reset(self, identifier: str | None = None) -> None:
        """Reset one caller or all callers; useful for administrative tooling."""

        async with self._lock:
            if identifier is None:
                self._requests.clear()
            else:
                self._requests.pop(identifier, None)


_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local request_limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local member = ARGV[3]
local request_cost = tonumber(ARGV[4])

local server_time = redis.call('TIME')
local now_ms = (tonumber(server_time[1]) * 1000) + math.floor(tonumber(server_time[2]) / 1000)
local cutoff_ms = now_ms - window_ms

redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff_ms)
local request_count = redis.call('ZCARD', key)

if request_cost <= request_limit and request_count + request_cost <= request_limit then
    for i = 1, request_cost do
        redis.call('ZADD', key, now_ms, member .. ':' .. i)
    end
    redis.call('PEXPIRE', key, window_ms)
    local remaining = request_limit - request_count - request_cost
    return {1, remaining, 0}
end

if request_cost > request_limit then
    return {0, math.max(0, request_limit - request_count), window_ms}
end

local must_expire = request_count + request_cost - request_limit
local oldest = redis.call('ZRANGE', key, must_expire - 1, must_expire - 1, 'WITHSCORES')
local retry_after_ms = 0
if oldest[2] then
    retry_after_ms = math.max(0, math.ceil(tonumber(oldest[2]) + window_ms - now_ms))
end
return {0, math.max(0, request_limit - request_count), retry_after_ms}
"""


class RedisSlidingWindowRateLimiter(RateLimiter):
    """Atomic Redis implementation suitable for multiple FastAPI workers."""

    def __init__(
        self,
        client: Any,
        limit: int,
        window_seconds: float,
        *,
        key_prefix: str = "assistant-rate-limit",
        owns_client: bool = False,
    ) -> None:
        _validate_configuration(limit, window_seconds)
        if not key_prefix:
            raise ValueError("key_prefix must not be empty")
        self._client = client
        self.limit = limit
        self.window_seconds = window_seconds
        self.window_milliseconds = max(1, math.ceil(window_seconds * 1000))
        self.key_prefix = key_prefix.rstrip(":")
        self._owns_client = owns_client

    @classmethod
    def from_url(
        cls,
        url: str,
        limit: int,
        window_seconds: float,
        *,
        key_prefix: str = "assistant-rate-limit",
        timeout_seconds: float = 2.0,
    ) -> RedisSlidingWindowRateLimiter:
        """Create a Redis limiter without importing Redis at module load time."""

        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Redis rate limiting requires the optional 'redis' package") from exc

        client = Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
        )
        return cls(
            client,
            limit,
            window_seconds,
            key_prefix=key_prefix,
            owns_client=True,
        )

    def _key(self, identifier: str) -> str:
        # Hash identifiers to avoid retaining IP addresses/API keys in Redis keys.
        digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
        return f"{self.key_prefix}:{digest}"

    async def ping(self) -> None:
        await self._client.ping()

    async def check(self, identifier: str, *, cost: int = 1) -> RateLimitDecision:
        if not identifier:
            raise ValueError("identifier must not be empty")
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < 1:
            raise ValueError("cost must be a positive integer")

        result = await self._client.eval(
            _SLIDING_WINDOW_SCRIPT,
            1,
            self._key(identifier),
            self.limit,
            self.window_milliseconds,
            secrets.token_hex(16),
            cost,
        )
        if not isinstance(result, (list, tuple)) or len(result) != 3:
            raise RuntimeError("Redis returned an invalid rate-limit result")

        allowed, remaining, retry_after_ms = (int(value) for value in result)
        return RateLimitDecision(
            allowed=bool(allowed),
            limit=self.limit,
            remaining=max(0, remaining),
            retry_after=max(0.0, retry_after_ms / 1000.0),
        )

    async def reset(self, identifier: str) -> None:
        if not identifier:
            raise ValueError("identifier must not be empty")
        await self._client.delete(self._key(identifier))

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


async def create_rate_limiter(
    *,
    limit: int,
    window_seconds: float,
    redis_url: str | None = None,
    redis_client: Any | None = None,
    key_prefix: str = "assistant-rate-limit",
    check_connection: bool = True,
    fallback_to_memory: bool = True,
    timeout_seconds: float = 2.0,
) -> RateLimiter:
    """Create a limiter, gracefully using memory when Redis is unavailable."""

    if redis_client is None and redis_url is None:
        return InMemorySlidingWindowRateLimiter(limit, window_seconds)

    try:
        limiter = (
            RedisSlidingWindowRateLimiter(
                redis_client,
                limit,
                window_seconds,
                key_prefix=key_prefix,
            )
            if redis_client is not None
            else RedisSlidingWindowRateLimiter.from_url(
                redis_url or "",
                limit,
                window_seconds,
                key_prefix=key_prefix,
                timeout_seconds=timeout_seconds,
            )
        )
        if check_connection:
            await limiter.ping()
        return limiter
    except Exception:
        if not fallback_to_memory:
            raise
        return InMemorySlidingWindowRateLimiter(limit, window_seconds)


def _validate_configuration(limit: int, window_seconds: float) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if (
        isinstance(window_seconds, bool)
        or not isinstance(window_seconds, (int, float))
        or not math.isfinite(window_seconds)
        or window_seconds <= 0
    ):
        raise ValueError("window_seconds must be a positive finite number")


__all__ = [
    "InMemorySlidingWindowRateLimiter",
    "RateLimitDecision",
    "RateLimiter",
    "RedisSlidingWindowRateLimiter",
    "create_rate_limiter",
]
