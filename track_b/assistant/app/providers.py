"""OpenAI-compatible model clients with bounded retry and provider fallback."""

from __future__ import annotations

import asyncio
import json
import random
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import Settings


class ProviderError(RuntimeError):
    """A provider failure that is safe to surface without credentials or payloads."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(slots=True)
class ModelToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ModelReply:
    content: str
    provider: str
    model: str
    tool_calls: list[ModelToolCall] = field(default_factory=list)


class ChatProvider(Protocol):
    name: str

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        require_json: bool = True,
    ) -> ModelReply: ...

    async def close(self) -> None: ...


class OpenAICompatibleProvider:
    """Provider for OpenAI and local vLLM's OpenAI-compatible HTTP API."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        api_key: str | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: float,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ProviderError("The openai client package is not installed") from exc
        self.name = name
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._supports_json_with_tools = name.casefold() != "vllm"
        self._client = AsyncOpenAI(
            api_key=api_key or "local-no-key",
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        require_json: bool = True,
    ) -> ModelReply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if require_json and (not tools or self._supports_json_with_tools):
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.chat.completions.create(**kwargs)
            if not response.choices:
                raise ProviderError(f"{self.name} returned no completion choices")
            message = response.choices[0].message
            tool_calls: list[ModelToolCall] = []
            for call in message.tool_calls or []:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise TypeError("tool arguments must be an object")
                except (json.JSONDecodeError, TypeError) as exc:
                    raise ProviderError(
                        f"{self.name} returned invalid tool arguments",
                        retryable=False,
                    ) from exc
                tool_calls.append(
                    ModelToolCall(id=call.id, name=call.function.name, arguments=arguments)
                )
            return ModelReply(
                content=message.content or "",
                provider=self.name,
                model=self.model,
                tool_calls=tool_calls,
            )
        except ProviderError:
            raise
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = not isinstance(status_code, int) or (
                status_code in {408, 409, 429} or status_code >= 500
            )
            raise ProviderError(
                f"{self.name} request failed ({type(exc).__name__})",
                retryable=retryable,
            ) from exc

    async def close(self) -> None:
        await self._client.close()


class DemoProvider:
    """Deterministic offline provider for tests and a zero-credential smoke mode."""

    name = "demo"
    model = "deterministic-demo"

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        require_json: bool = True,
    ) -> ModelReply:
        tool_messages = [message for message in messages if message.get("role") == "tool"]
        if tool_messages:
            content = tool_messages[-1].get("content", "{}")
            answer = f"The tool returned: {content}"
        else:
            user_text = next(
                (
                    str(message.get("content", ""))
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )
            match = re.search(r"(?:calculate|what is)\s+([\d\s+*/().%-]+)", user_text, re.I)
            if tools and match:
                return ModelReply(
                    content="",
                    provider=self.name,
                    model=self.model,
                    tool_calls=[
                        ModelToolCall(
                            id=f"demo-{uuid.uuid4().hex[:8]}",
                            name="calculate",
                            arguments={"expression": match.group(1).strip().rstrip("?")},
                        )
                    ],
                )
            answer = (
                "Demo mode is active. I received your request, but a configured OpenAI or "
                "vLLM provider is required for a generated answer."
            )
        payload = {"answer": answer, "confidence": 0.25, "follow_up_questions": []}
        return ModelReply(
            content=json.dumps(payload) if require_json else answer,
            provider=self.name,
            model=self.model,
        )

    async def close(self) -> None:
        return None


class ResilientProvider:
    """Retry transient failures, then move to the configured fallback provider."""

    def __init__(
        self,
        providers: list[ChatProvider],
        *,
        retry_attempts: int = 3,
        base_delay_seconds: float = 0.25,
        total_timeout_seconds: float = 60.0,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.retry_attempts = retry_attempts
        self.base_delay_seconds = base_delay_seconds
        self.total_timeout_seconds = total_timeout_seconds

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        require_json: bool = True,
    ) -> tuple[ModelReply, bool]:
        try:
            async with asyncio.timeout(self.total_timeout_seconds):
                provider_budget = self.total_timeout_seconds / len(self.providers)
                for provider_index, provider in enumerate(self.providers):
                    try:
                        async with asyncio.timeout(provider_budget):
                            for attempt in range(self.retry_attempts):
                                try:
                                    reply = await provider.complete(
                                        messages,
                                        tools=tools,
                                        require_json=require_json,
                                    )
                                    return reply, provider_index > 0
                                except ProviderError as exc:
                                    if not exc.retryable:
                                        break
                                    if attempt + 1 < self.retry_attempts:
                                        delay = self.base_delay_seconds * (2**attempt)
                                        delay += random.uniform(0, max(delay * 0.2, 0.001))
                                        await asyncio.sleep(delay)
                    except TimeoutError:
                        continue
        except TimeoutError as exc:
            raise ProviderError("The provider retry deadline was exceeded") from exc
        provider_names = ", ".join(provider.name for provider in self.providers)
        raise ProviderError(f"All configured providers are unavailable: {provider_names}")

    async def close(self) -> None:
        close_calls = []
        for provider in self.providers:
            close = getattr(provider, "close", None)
            if close is not None:
                close_calls.append(close())
        if close_calls:
            await asyncio.gather(*close_calls, return_exceptions=True)


def _secret_value(secret: Any) -> str | None:
    return secret.get_secret_value() if secret is not None else None


def _make_provider(
    *,
    name: str,
    model: str,
    base_url: str | None,
    api_key: Any,
    settings: Settings,
) -> ChatProvider:
    if name.casefold() in {"demo", "mock"}:
        return DemoProvider()
    if not base_url:
        raise ValueError(f"A base URL is required for provider {name}")
    return OpenAICompatibleProvider(
        name=name,
        model=model,
        base_url=base_url,
        api_key=_secret_value(api_key),
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_tokens,
        timeout_seconds=settings.provider_timeout_seconds,
    )


def build_provider_router(settings: Settings) -> ResilientProvider:
    providers: list[ChatProvider] = [
        _make_provider(
            name=settings.primary_provider,
            model=settings.primary_model,
            base_url=settings.primary_base_url,
            api_key=settings.primary_api_key,
            settings=settings,
        )
    ]
    if settings.fallback_provider and settings.fallback_model:
        providers.append(
            _make_provider(
                name=settings.fallback_provider,
                model=settings.fallback_model,
                base_url=settings.fallback_base_url,
                api_key=settings.fallback_api_key,
                settings=settings,
            )
        )
    return ResilientProvider(
        providers,
        retry_attempts=settings.retry_attempts,
        base_delay_seconds=settings.retry_base_delay_seconds,
        total_timeout_seconds=settings.provider_total_timeout_seconds,
    )
