from __future__ import annotations

import asyncio
import copy
import json

import pytest

from agentic import AgentConfig, OfflinePolicy, PlannerReply, SourceStore, VerificationAgent
from agentic.engine import SAFE_ABSTENTION, SAFE_CLARIFICATION, verified_claims
from agentic.evaluation import run_suite, score_case


class ScriptedPlanner:
    mode = "test_script"

    def __init__(self, decisions):
        self.decisions = iter(decisions)

    async def decide(self, context):
        decision = next(self.decisions)
        return PlannerReply(decision, 10, 5, estimated=True, raw_response=json.dumps(decision))


async def run(query="How many annual leave days?", *, planner=None, config=None, failure=None):
    return await VerificationAgent(
        planner=planner or OfflinePolicy(),
        tools=SourceStore.default(failure=failure),
        config=config or AgentConfig(),
    ).run(query)


@pytest.mark.asyncio
async def test_adaptive_paths_and_authority_resolution():
    agreed = await run("What is the annual leave carryover allowance?")
    conflict = await run()
    ambiguous = await run("What is the policy?")
    missing = await run("What is the sabbatical allowance?")
    assert [r.steps for r in (agreed, conflict, ambiguous, missing)] == [3, 3, 1, 4]
    assert conflict.claims[0]["value"] == "22 paid days per calendar year"
    assert len(conflict.citations) == 2
    assert ambiguous.status == "needs_clarification"
    assert missing.status == "insufficient_evidence"
    assert conflict.usage["basis"] == "estimated_simulated"
    assert conflict.usage["total_tokens"] == sum(
        e["usage"]["input_tokens"] + e["usage"]["output_tokens"] for e in conflict.trace
    )
    assert all(e["decision"]["reason"] for e in conflict.trace)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["registry_unavailable", "malformed_registry", "registry_timeout"]
)
async def test_bad_tool_results_never_produce_confident_answer(failure):
    result = await run(failure=failure, config=AgentConfig(tool_timeout_seconds=0.001))
    assert result.status == "insufficient_evidence"
    assert result.citations == []
    assert result.failure_events
    assert not result.trace[-1]["decision"].get("claims")


@pytest.mark.asyncio
async def test_guard_feedback_causes_adaptive_repair_with_alternate_tool_order():
    decisions = [
        {"action": "search_policy_registry", "arguments": {"query": "annual leave entitlement"}},
        {
            "action": "finish",
            "status": "verified",
            "claims": [
                {
                    "key": "annual_leave_days",
                    "value": "22 paid days per calendar year",
                    "citations": ["registry-leave-2026"],
                }
            ],
        },
        {"action": "search_handbook", "arguments": {"query": "annual leave entitlement"}},
        {
            "action": "finish",
            "status": "verified",
            "claims": [
                {
                    "key": "annual_leave_days",
                    "value": "22 paid days per calendar year",
                    "citations": ["registry-leave-2026", "handbook-leave-2024"],
                }
            ],
        },
    ]
    result = await run(planner=ScriptedPlanner(decisions))
    assert result.status == "verified"
    assert result.steps == 4
    assert "two distinct" in result.trace[1]["observation"]["error"]
    assert result.trace[2]["context"]["history"][-1]["ok"] is False


@pytest.mark.asyncio
async def test_repeated_invalid_decisions_are_cascading_and_bounded():
    planner = ScriptedPlanner([{"action": "search_unknown", "arguments": {}}] * 3)
    result = await run(planner=planner, config=AgentConfig(max_steps=3))
    assert result.status == "budget_exhausted"
    assert result.steps == 3
    assert [e["category"] for e in result.failure_events] == [
        "soft_failure",
        "cascading_soft_failure",
        "cascading_soft_failure",
    ]
    row = score_case(
        {
            "id": "invalid",
            "expected_status": "verified",
            "search_terms": ["leave"],
            "max_reasonable_steps": 3,
        },
        result.to_dict(),
    )
    assert row["failure_class"] == "cascading_soft_failure"
    assert row["correct_calls"] == 0 and row["total_calls"] == 3


@pytest.mark.asyncio
async def test_successful_action_resets_consecutive_soft_failures():
    decisions = [
        {"action": "unknown"},
        {"action": "search_handbook", "arguments": {"query": "annual leave"}},
        {"action": "unknown"},
    ]
    result = await run(planner=ScriptedPlanner(decisions), config=AgentConfig(max_steps=3))
    assert result.trace[1]["observation"]["ok"] is True
    assert [event["category"] for event in result.failure_events] == [
        "soft_failure",
        "soft_failure",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [[], {}, None, 42])
async def test_harness_records_nonstring_actions_without_crashing(action):
    result = await run(
        planner=ScriptedPlanner([{"action": action}]), config=AgentConfig(max_steps=1)
    )
    row = score_case(
        {
            "id": "malformed_action",
            "expected_status": "verified",
            "search_terms": ["leave"],
            "max_reasonable_steps": 1,
        },
        result.to_dict(),
    )
    assert row["completed"] is False
    assert row["failure_class"] == "soft_failure"
    assert row["total_calls"] == 1
    assert row["correct_calls"] == 0
    assert row["tool_calls"][0]["valid"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_status", "safe_answer"),
    [
        (
            {
                "action": "finish",
                "status": "insufficient_evidence",
                "answer": "Employees definitely receive 90 paid days. This is current policy.",
            },
            "insufficient_evidence",
            SAFE_ABSTENTION,
        ),
        (
            {
                "action": "clarify",
                "question": "Contractors receive 90 paid days; would you like more details?",
            },
            "needs_clarification",
            SAFE_CLARIFICATION,
        ),
    ],
)
async def test_unverified_terminal_text_is_retained_only_in_trace(
    decision, expected_status, safe_answer
):
    result = await run(planner=ScriptedPlanner([decision]))
    assert result.status == expected_status
    assert result.answer == safe_answer
    assert "90 paid days" not in result.answer
    assert result.claims == [] and result.citations == []
    assert result.trace[0]["decision"] == decision
    assert "90 paid days" in result.trace[0]["raw_response"]


@pytest.mark.asyncio
async def test_provider_timeout_is_hard_failure_with_incomplete_usage():
    class HangingPlanner:
        mode = "live"

        async def decide(self, context):
            await asyncio.sleep(10)

    result = await run(planner=HangingPlanner(), config=AgentConfig(model_timeout_seconds=0.001))
    assert result.status == "error"
    assert result.failure_events[0]["category"] == "hard_failure"
    assert result.usage["complete"] is False
    assert result.usage["total_tokens"] == 0


@pytest.mark.asyncio
async def test_context_is_capped_while_raw_traces_retain_evidence():
    result = await run(config=AgentConfig(top_k=1, max_notes=2, max_excerpt_chars=80))
    assert result.status == "verified"
    for event in result.trace:
        assert len(event["context"]["notes"]) <= 2
        assert all(len(n["excerpt"]) <= 80 for n in event["context"]["notes"])
    raw = result.trace[0]["raw_tool_result"]["documents"][0]
    assert len(raw["text"]) > 80


@pytest.mark.asyncio
async def test_token_budget_accounts_for_call_that_crosses_limit():
    result = await run(config=AgentConfig(max_tokens=100))
    assert result.status == "budget_exhausted"
    assert result.steps == 1
    assert result.usage["total_tokens"] > 100
    assert not any("tool_call" in e for e in result.trace)


@pytest.mark.parametrize("change", ["unknown", "duplicate", "wrong_value", "uncited_authority"])
def test_evidence_guard_rejects_forgery_and_cherry_picking(change):
    docs = {d["id"]: d for d in SourceStore.default().documents}
    claim = {
        "key": "annual_leave_days",
        "value": "22 paid days per calendar year",
        "citations": ["registry-leave-2026", "handbook-leave-2024"],
    }
    if change == "unknown":
        claim["citations"].append("invented")
    elif change == "duplicate":
        claim["citations"] = ["registry-leave-2026"] * 2
    elif change == "wrong_value":
        claim["value"] = "20 paid days per calendar year"
    else:
        another = copy.deepcopy(docs["handbook-leave-2024"])
        another.update(id="another", publisher="another-publisher")
        docs["another"] = another
        claim.update(
            value="20 paid days per calendar year", citations=["another", "handbook-leave-2024"]
        )
    with pytest.raises(ValueError):
        verified_claims({"claims": [claim]}, docs)


@pytest.mark.asyncio
async def test_custom_harness_keeps_failures_and_raw_traces(tmp_path):
    report = await run_suite(output_dir=tmp_path)
    assert report["metrics"]["query_count"] == 11
    assert report["metrics"]["task_completion_rate"] == pytest.approx(10 / 11)
    assert report["metrics"]["tool_call_correctness"] == 1
    assert len((tmp_path / "traces.jsonl").read_text().splitlines()) == 11
    failures = json.loads((tmp_path / "failures.json").read_text())
    assert failures[0]["id"] == "step_budget"
    assert failures[0]["category"] == "soft_failure"


@pytest.mark.asyncio
async def test_tool_arguments_validated_before_execution():
    planner = ScriptedPlanner([{"action": "search_handbook", "arguments": {"query": 42}}])
    result = await run(planner=planner, config=AgentConfig(max_steps=1))
    assert result.status == "budget_exhausted"
    assert "exactly one string" in result.failure_events[0]["message"]


@pytest.mark.asyncio
async def test_verification_http_auth_and_inherited_chat(monkeypatch):
    import importlib

    import httpx

    from app.config import get_settings

    for name, value in {
        "ENVIRONMENT": "test",
        "PRIMARY_PROVIDER": "demo",
        "PRIMARY_MODEL": "demo",
        "PRIMARY_BASE_URL": "http://unused",
        "EMBEDDING_PROVIDER": "hash",
        "VECTOR_STORE": "memory",
        "APP_API_KEY": "test-key",
        "AGENT_MODE": "offline",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("REDIS_URL", raising=False)
    get_settings.cache_clear()
    import app.main as main_module

    main_module = importlib.reload(main_module)
    async with (
        main_module.lifespan(main_module.app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main_module.app), base_url="http://testserver"
        ) as client,
    ):
        assert (await client.post("/api/v1/verify", json={"query": "leave"})).status_code == 401
        result = await client.post(
            "/api/v1/verify",
            json={"query": "annual leave days"},
            headers={"X-API-Key": "test-key"},
        )
        assert result.status_code == 200
        assert result.json()["status"] == "verified"
        assert result.json()["model_mode"] == "offline_simulation"
        blank = await client.post(
            "/api/v1/verify", json={"query": "  "}, headers={"X-API-Key": "test-key"}
        )
        assert blank.status_code == 422
        chat = await client.post(
            "/api/v1/chat",
            json={"message": "calculate 2 + 2", "use_rag": False, "session_id": "test"},
            headers={"X-API-Key": "test-key"},
        )
        assert chat.status_code == 200
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_live_adapter_retains_reported_usage_for_malformed_json():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from agentic import OpenAIPlanner
    from agentic.planners import SYSTEM_PROMPT

    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not valid JSON"))],
            usage=SimpleNamespace(prompt_tokens=97, completion_tokens=13),
        )
    )
    planner = OpenAIPlanner.__new__(OpenAIPlanner)
    planner.model = "test-live-model"
    planner.system_prompt = SYSTEM_PROMPT
    planner.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result = await run(planner=planner, config=AgentConfig(max_steps=1))
    assert result.status == "budget_exhausted"
    assert result.usage["total_tokens"] == 110
    assert result.usage["basis"] == "provider_reported"
    assert result.usage["estimated"] is False
    assert result.trace[0]["raw_response"] == "not valid JSON"
    assert create.await_args.kwargs["response_format"] == {"type": "json_object"}
