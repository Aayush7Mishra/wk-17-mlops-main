from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.rate_limit import (
    InMemorySlidingWindowRateLimiter,
    RedisSlidingWindowRateLimiter,
    create_rate_limiter,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeRedis:
    def __init__(
        self,
        result: list[int] | None = None,
        *,
        ping_error: Exception | None = None,
    ) -> None:
        self.result = result or [1, 4, 0]
        self.ping_error = ping_error
        self.eval_calls: list[tuple[Any, ...]] = []
        self.deleted: list[str] = []

    async def ping(self) -> bool:
        if self.ping_error is not None:
            raise self.ping_error
        return True

    async def eval(self, *arguments: Any) -> list[int]:
        self.eval_calls.append(arguments)
        return self.result

    async def delete(self, key: str) -> None:
        self.deleted.append(key)


@pytest.mark.asyncio
async def test_sliding_window_reports_allowance_and_retry_time() -> None:
    clock = FakeClock()
    limiter = InMemorySlidingWindowRateLimiter(2, 10, clock=clock)

    first = await limiter.check("caller")
    clock.advance(2)
    second = await limiter.check("caller")
    denied = await limiter.check("caller")

    assert (first.allowed, first.remaining, first.retry_after) == (True, 1, 0)
    assert (second.allowed, second.remaining, second.retry_after) == (True, 0, 0)
    assert denied.allowed is False
    assert denied.limit == 2
    assert denied.remaining == 0
    assert denied.retry_after == pytest.approx(8)
    assert denied.as_headers() == {
        "X-RateLimit-Limit": "2",
        "X-RateLimit-Remaining": "0",
        "Retry-After": "8",
    }


@pytest.mark.asyncio
async def test_old_requests_leave_the_sliding_window() -> None:
    clock = FakeClock()
    limiter = InMemorySlidingWindowRateLimiter(2, 10, clock=clock)

    await limiter.check("caller")
    clock.advance(5)
    await limiter.check("caller")
    clock.advance(5)

    decision = await limiter.check("caller")
    assert decision.allowed is True
    assert decision.remaining == 0


@pytest.mark.asyncio
async def test_identifiers_have_independent_limits() -> None:
    limiter = InMemorySlidingWindowRateLimiter(1, 60)

    assert (await limiter.check("alice")).allowed is True
    assert (await limiter.check("alice")).allowed is False
    assert (await limiter.check("bob")).allowed is True


@pytest.mark.asyncio
async def test_identifier_tracking_is_bounded() -> None:
    clock = FakeClock()
    limiter = InMemorySlidingWindowRateLimiter(
        2,
        10,
        max_identifiers=2,
        clock=clock,
    )
    await limiter.check("oldest")
    clock.advance(1)
    await limiter.check("newer")
    clock.advance(1)
    await limiter.check("newest")

    assert len(limiter._requests) == 2
    assert "oldest" not in limiter._requests


@pytest.mark.asyncio
async def test_concurrent_checks_cannot_exceed_limit() -> None:
    limiter = InMemorySlidingWindowRateLimiter(5, 60)

    decisions = await asyncio.gather(*(limiter.check("same-caller") for _ in range(25)))

    assert sum(decision.allowed for decision in decisions) == 5


@pytest.mark.asyncio
async def test_weighted_check_is_atomic_and_does_not_spend_rejected_quota() -> None:
    limiter = InMemorySlidingWindowRateLimiter(5, 60)

    assert (await limiter.check("caller", cost=3)).remaining == 2
    rejected = await limiter.check("caller", cost=4)
    final = await limiter.check("caller", cost=2)

    assert rejected.allowed is False
    assert rejected.remaining == 2
    assert final.allowed is True
    assert final.remaining == 0


@pytest.mark.asyncio
async def test_redis_limiter_parses_atomic_script_result() -> None:
    client = FakeRedis([0, 2, 1750])
    limiter = RedisSlidingWindowRateLimiter(
        client, limit=5, window_seconds=12.5, key_prefix="test-limit"
    )

    decision = await limiter.check("user@example.com", cost=3)

    assert decision.allowed is False
    assert decision.limit == 5
    assert decision.remaining == 2
    assert decision.retry_after == pytest.approx(1.75)
    call = client.eval_calls[0]
    assert call[1] == 1
    assert call[2].startswith("test-limit:")
    assert "user@example.com" not in call[2]
    assert call[3:5] == (5, 12500)
    assert call[6] == 3


@pytest.mark.asyncio
async def test_rate_limiter_factory_keeps_healthy_injected_redis() -> None:
    limiter = await create_rate_limiter(
        limit=3,
        window_seconds=60,
        redis_client=FakeRedis(),
    )

    assert isinstance(limiter, RedisSlidingWindowRateLimiter)


@pytest.mark.asyncio
async def test_redis_limiter_rejects_malformed_result() -> None:
    client = FakeRedis([1, 2])
    limiter = RedisSlidingWindowRateLimiter(client, 5, 60)

    with pytest.raises(RuntimeError, match="invalid rate-limit result"):
        await limiter.check("caller")


@pytest.mark.asyncio
async def test_rate_limiter_factory_falls_back_when_redis_is_offline() -> None:
    client = FakeRedis(ping_error=ConnectionError("offline"))

    limiter = await create_rate_limiter(
        limit=3,
        window_seconds=60,
        redis_client=client,
        check_connection=True,
    )

    assert isinstance(limiter, InMemorySlidingWindowRateLimiter)
    assert (await limiter.check("caller")).allowed is True


@pytest.mark.asyncio
async def test_reset_restores_allowance() -> None:
    limiter = InMemorySlidingWindowRateLimiter(1, 60)
    await limiter.check("caller")
    assert (await limiter.check("caller")).allowed is False

    await limiter.reset("caller")
    assert (await limiter.check("caller")).allowed is True


@pytest.mark.parametrize(
    ("limit", "window"),
    [(0, 10), (-1, 10), (True, 10), (1, 0), (1, float("inf"))],
)
def test_invalid_configuration_is_rejected(limit: Any, window: Any) -> None:
    with pytest.raises(ValueError):
        InMemorySlidingWindowRateLimiter(limit, window)
