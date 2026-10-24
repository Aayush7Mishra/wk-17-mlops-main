from __future__ import annotations

import pytest

from app.assistant import AssistantService
from app.cache import InMemoryTTLCache
from app.config import Settings
from app.providers import build_provider_router
from app.rag import build_rag_service
from app.schemas import ChatRequest
from app.tools import ToolRegistry


def build_demo_assistant() -> tuple[AssistantService, object]:
    settings = Settings(
        environment="test",
        primary_provider="demo",
        primary_model="demo",
        primary_base_url="http://unused",
        embedding_provider="hash",
        vector_store="memory",
        minimum_relevance_score=-1,
        retry_base_delay_seconds=0,
    )
    rag = build_rag_service(settings)
    tools = ToolRegistry(
        retriever=lambda query, limit, namespace: rag.retrieve(
            query,
            limit,
            namespace=namespace,
        )
    )
    assistant = AssistantService(
        settings=settings,
        provider=build_provider_router(settings),
        rag=rag,
        tools=tools,
        cache=InMemoryTTLCache(default_ttl_seconds=300),
    )
    return assistant, rag


@pytest.mark.asyncio
async def test_assistant_retrieves_and_caches_structured_response() -> None:
    assistant, rag = build_demo_assistant()
    await rag.ingest_text(
        filename="policy.md",
        text="Refund requests are accepted for 30 days after purchase.",
        namespace="test",
    )
    request = ChatRequest(message="What is the refund window?", session_id="test")

    first = await assistant.chat(request)
    second = await assistant.chat(request)

    assert first.cached is False
    assert first.sources and first.sources[0].filename == "policy.md"
    assert first.provider == "demo/deterministic-demo"
    assert second.cached is True
    assert second.answer == first.answer
    assert second.request_id != first.request_id


@pytest.mark.asyncio
async def test_assistant_executes_calculator_tool() -> None:
    assistant, _ = build_demo_assistant()

    response = await assistant.chat(
        ChatRequest(message="Calculate 2 + 3 * 4", use_rag=False, session_id="tools")
    )

    assert response.answer.startswith("The tool returned:")
    assert response.tool_calls[0].name == "calculate"
    assert response.tool_calls[0].result["value"] == 14


@pytest.mark.asyncio
async def test_retrieved_source_fingerprint_invalidates_cached_metadata() -> None:
    assistant, rag = build_demo_assistant()
    request = ChatRequest(message="What is the policy?", session_id="cache")
    await rag.ingest_text(filename="old.md", text="The policy is stable.", namespace="cache")

    first = await assistant.chat(request)
    cached = await assistant.chat(request)
    await rag.ingest_text(filename="new.md", text="The policy is stable.", namespace="cache")
    refreshed = await assistant.chat(request)

    assert first.cached is False
    assert cached.cached is True
    assert refreshed.cached is False
    assert refreshed.sources[0].filename == "new.md"


def test_prompt_budget_caps_multibyte_input_and_marks_truncation() -> None:
    assistant, _ = build_demo_assistant()
    assistant.settings.max_prompt_characters = 4000
    assistant.settings.max_prompt_utf8_bytes = 3000
    request = ChatRequest(message="🙂" * 5000, session_id="budget", use_rag=False)

    messages, truncated = assistant._build_messages(request, [])

    assert truncated is True
    assert sum(len(str(item["content"]).encode("utf-8")) for item in messages) <= 3000
