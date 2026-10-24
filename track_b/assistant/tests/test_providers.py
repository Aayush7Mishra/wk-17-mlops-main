from __future__ import annotations

import asyncio
import json

import pytest

from app.providers import (
    ModelReply,
    OpenAICompatibleProvider,
    ProviderError,
    ResilientProvider,
)


class FailingProvider:
    name = "failing"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, tools=None, require_json=True):
        self.calls += 1
        raise ProviderError("planned failure")


class StaticProvider:
    name = "fallback"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, *, tools=None, require_json=True):
        self.calls += 1
        return ModelReply(
            content=json.dumps(
                {"answer": "fallback answer", "confidence": 0.8, "follow_up_questions": []}
            ),
            provider=self.name,
            model="test-model",
        )


class NonRetryableProvider(FailingProvider):
    async def complete(self, messages, *, tools=None, require_json=True):
        self.calls += 1
        raise ProviderError("bad request", retryable=False)


class SlowProvider(FailingProvider):
    async def complete(self, messages, *, tools=None, require_json=True):
        self.calls += 1
        await asyncio.sleep(1)
        raise ProviderError("slow failure")


@pytest.mark.asyncio
async def test_provider_retries_then_falls_back() -> None:
    primary = FailingProvider()
    fallback = StaticProvider()
    router = ResilientProvider(
        [primary, fallback],
        retry_attempts=2,
        base_delay_seconds=0,
    )

    reply, degraded = await router.complete([{"role": "user", "content": "hello"}])

    assert reply.provider == "fallback"
    assert degraded is True
    assert primary.calls == 2
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_provider_reports_bounded_failure() -> None:
    primary = FailingProvider()
    router = ResilientProvider([primary], retry_attempts=2, base_delay_seconds=0)

    with pytest.raises(ProviderError, match="All configured providers"):
        await router.complete([{"role": "user", "content": "hello"}])
    assert primary.calls == 2


@pytest.mark.asyncio
async def test_non_retryable_failure_moves_to_fallback_immediately() -> None:
    primary = NonRetryableProvider()
    fallback = StaticProvider()
    router = ResilientProvider([primary, fallback], retry_attempts=4, base_delay_seconds=0)

    reply, degraded = await router.complete([{"role": "user", "content": "hello"}])

    assert reply.provider == "fallback"
    assert degraded is True
    assert primary.calls == 1


@pytest.mark.asyncio
async def test_provider_deadline_reserves_time_for_fallback() -> None:
    primary = SlowProvider()
    fallback = StaticProvider()
    router = ResilientProvider(
        [primary, fallback],
        retry_attempts=3,
        base_delay_seconds=0,
        total_timeout_seconds=0.2,
    )

    reply, degraded = await router.complete([{"role": "user", "content": "hello"}])

    assert reply.provider == "fallback"
    assert degraded is True
    assert primary.calls == 1


@pytest.mark.asyncio
async def test_vllm_does_not_mix_guided_json_with_auto_tool_choice(monkeypatch) -> None:
    recorded: list[dict] = []

    class FakeCompletions:
        async def create(self, **kwargs):
            recorded.append(kwargs)
            message = type("Message", (), {"content": "{}", "tool_calls": []})()
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        async def close(self) -> None:
            return None

    monkeypatch.setattr("openai.AsyncOpenAI", FakeClient)
    provider = OpenAICompatibleProvider(
        name="vllm",
        model="local-model",
        base_url="http://vllm:8000/v1",
        api_key=None,
        temperature=0.2,
        top_p=0.9,
        max_tokens=100,
        timeout_seconds=1,
    )

    await provider.complete(
        [{"role": "user", "content": "calculate"}],
        tools=[{"type": "function", "function": {"name": "calculate"}}],
    )
    await provider.complete([{"role": "user", "content": "answer"}], tools=None)

    assert "response_format" not in recorded[0]
    assert recorded[1]["response_format"] == {"type": "json_object"}
