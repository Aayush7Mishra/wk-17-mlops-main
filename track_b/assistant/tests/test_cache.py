from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.cache import (
    InMemoryTTLCache,
    RedisPromptResponseCache,
    create_cache,
    make_cache_key,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeRedis:
    def __init__(self, *, ping_error: Exception | None = None) -> None:
        self.values: dict[str, str] = {}
        self.psetex_calls: list[tuple[str, int, str]] = []
        self.ping_error = ping_error

    async def ping(self) -> bool:
        if self.ping_error is not None:
            raise self.ping_error
        return True

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str) -> None:
        self.values[key] = value

    async def psetex(self, key: str, ttl_ms: int, value: str) -> None:
        self.psetex_calls.append((key, ttl_ms, value))
        self.values[key] = value

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)

    async def incr(self, key: str) -> int:
        value = int(self.values.get(key, "0")) + 1
        self.values[key] = str(value)
        return value


def test_cache_key_is_deterministic_for_equivalent_parameters() -> None:
    first = make_cache_key(
        "Where is the document?",
        model="demo-model",
        system_prompt="Return JSON",
        parameters={"temperature": 0.1, "response_format": {"type": "json"}},
    )
    second = make_cache_key(
        "Where is the document?",
        model="demo-model",
        system_prompt="Return JSON",
        parameters={"response_format": {"type": "json"}, "temperature": 0.1},
    )

    assert first == second
    assert first.startswith("assistant:")
    assert len(first.removeprefix("assistant:")) == 64


def test_cache_key_changes_when_request_changes() -> None:
    baseline = make_cache_key("question", model="model-a", parameters={"top_p": 1})

    assert baseline != make_cache_key("question ", model="model-a", parameters={"top_p": 1})
    assert baseline != make_cache_key("question", model="model-b", parameters={"top_p": 1})


def test_cache_key_rejects_non_json_parameters() -> None:
    with pytest.raises(ValueError, match="JSON serializable"):
        make_cache_key("question", model="model", parameters={"invalid": object()})


@pytest.mark.asyncio
async def test_in_memory_cache_expires_at_ttl_boundary() -> None:
    clock = FakeClock()
    cache = InMemoryTTLCache(default_ttl_seconds=10, clock=clock)

    await cache.set("key", '{"answer":"cached"}')
    clock.advance(9.999)
    assert await cache.get("key") == '{"answer":"cached"}'

    clock.advance(0.001)
    assert await cache.get("key") is None
    assert await cache.size() == 0


@pytest.mark.asyncio
async def test_in_memory_cache_supports_override_delete_and_no_expiry() -> None:
    clock = FakeClock()
    cache = InMemoryTTLCache(default_ttl_seconds=None, clock=clock)

    await cache.set("permanent", "response")
    await cache.set("temporary", "response", ttl_seconds=2)
    await cache.set("never-store", "response", ttl_seconds=0)
    clock.advance(3)

    assert await cache.get("permanent") == "response"
    assert await cache.get("temporary") is None
    assert await cache.get("never-store") is None
    await cache.delete("permanent")
    assert await cache.get("permanent") is None


@pytest.mark.asyncio
async def test_in_memory_cache_is_bounded_and_evicts_least_recently_used() -> None:
    cache = InMemoryTTLCache(default_ttl_seconds=None, max_entries=2)
    await cache.set("old", "one")
    await cache.set("kept", "two")
    assert await cache.get("old") == "one"

    await cache.set("new", "three")

    assert await cache.get("kept") is None
    assert await cache.get("old") == "one"
    assert await cache.get("new") == "three"


@pytest.mark.asyncio
async def test_cache_counter_is_atomic_and_persistent() -> None:
    cache = InMemoryTTLCache(default_ttl_seconds=1)
    values = await asyncio.gather(*(cache.increment("revision") for _ in range(20)))

    assert sorted(values) == list(range(1, 21))
    assert await cache.get("revision") == "20"


@pytest.mark.asyncio
async def test_redis_cache_uses_prefixed_key_and_millisecond_ttl() -> None:
    client = FakeRedis()
    cache = RedisPromptResponseCache(client, default_ttl_seconds=1.25, key_prefix="test-cache")

    await cache.set("digest", "response")

    assert client.psetex_calls == [("test-cache:digest", 1250, "response")]
    assert await cache.get("digest") == "response"
    await cache.delete("digest")
    assert await cache.get("digest") is None


@pytest.mark.asyncio
async def test_cache_factory_falls_back_when_redis_ping_fails() -> None:
    client = FakeRedis(ping_error=ConnectionError("offline"))

    cache = await create_cache(
        redis_client=client,
        default_ttl_seconds=30,
        check_connection=True,
    )

    assert isinstance(cache, InMemoryTTLCache)
    await cache.set("key", "works without a network")
    assert await cache.get("key") == "works without a network"


@pytest.mark.asyncio
async def test_cache_factory_can_fail_fast_when_fallback_is_disabled() -> None:
    client: Any = FakeRedis(ping_error=ConnectionError("offline"))

    with pytest.raises(ConnectionError, match="offline"):
        await create_cache(
            redis_client=client,
            check_connection=True,
            fallback_to_memory=False,
        )
