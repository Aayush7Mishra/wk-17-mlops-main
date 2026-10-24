from __future__ import annotations

import pytest

from app.schemas import Source
from app.tools import ToolRegistry, calculate


def test_calculator_handles_precedence_without_eval() -> None:
    assert calculate("2 + 3 * 4")["value"] == 14
    assert calculate("(10 - 4) / 3")["value"] == 2


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "open('/etc/passwd').read()",
        "2 ** 1000",
        "1 / 0",
    ],
)
def test_calculator_rejects_unsafe_or_unbounded_expressions(expression: str) -> None:
    with pytest.raises(ValueError, match="invalid arithmetic expression"):
        calculate(expression)


@pytest.mark.asyncio
async def test_registry_validates_and_executes_retrieval_tool() -> None:
    async def retrieve(query: str, limit: int, namespace: str) -> list[Source]:
        return [
            Source(
                document_id="doc",
                chunk_id="chunk",
                filename="guide.md",
                text=f"result for {query}",
                score=0.9,
                metadata={"limit": limit, "namespace": namespace},
            )
        ]

    registry = ToolRegistry(retriever=retrieve)
    execution = await registry.execute(
        "search_knowledge_base",
        {"query": "refund", "limit": 2},
        namespace="tenant-a",
    )
    missing = await registry.execute("delete_everything", {})

    assert execution.ok is True
    assert execution.result[0]["filename"] == "guide.md"
    assert execution.result[0]["metadata"]["namespace"] == "tenant-a"
    assert missing.ok is False


@pytest.mark.asyncio
async def test_registry_can_disable_retrieval_per_request() -> None:
    async def retrieve(query: str, limit: int, namespace: str) -> list[Source]:
        raise AssertionError("retrieval must not execute")

    registry = ToolRegistry(retriever=retrieve)

    assert all(
        tool["function"]["name"] != "search_knowledge_base"
        for tool in registry.definitions(include_retrieval=False)
    )
    execution = await registry.execute(
        "search_knowledge_base",
        {"query": "private"},
        allow_retrieval=False,
    )
    assert execution.ok is False
