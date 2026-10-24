import json
from types import SimpleNamespace

import pytest
from experiment import (
    ROOT,
    completion,
    evidently_suite,
    judge_response,
    promotion_gate,
    tool_scores,
)


def test_verified_completion_requires_the_right_fact_not_just_status():
    case = {"status": "verified", "fact_key": "annual_leave_days", "fact_value": "22"}
    assert not completion(case, {"status": "verified", "claims": [
        {"key": "annual_leave_days", "value": "20"}]})
    assert completion(case, {"status": "verified", "claims": [
        {"key": "annual_leave_days", "value": "22"}]})


def test_gate_requires_completion_both_judges_and_usage():
    good = {"completed": True, "correctness": True, "groundedness": True,
            "usage_complete": True}
    assert promotion_gate([good])["promote"]
    for key in good:
        assert not promotion_gate([{**good, key: False}])["promote"]
    assert not promotion_gate([])["promote"]


def test_irrelevant_and_malformed_tool_calls_are_not_correct():
    result = {"query": "annual leave days", "trace": [
        {"tool_call": {"name": "search_handbook", "arguments": {"query": "annual leave"}}},
        {"tool_call": {"name": "search_handbook", "arguments": {"query": "lunch"}}},
        {"tool_call": {"name": "search_handbook", "arguments": {"query": "leave", "extra": 1}}},
    ]}
    assert tool_scores(result) == (1, 3)


def test_rejected_tool_attempts_count_in_denominator():
    result = {"query": "annual leave", "trace": [
        {"decision": {"action": "search_handbook", "arguments": []}},
        {"decision": {"action": "invented_tool", "arguments": {"query": "leave"}}},
        {"tool_call": {"name": "search_handbook", "arguments": {"query": "leave"}}},
        {"decision": {"action": "clarify", "question": "Which policy?"}},
    ]}
    assert tool_scores(result) == (1, 3)


def test_non_string_actions_do_not_crash_evaluation():
    result = {"query": "leave", "trace": [
        {"decision": {"action": [], "arguments": {"query": "leave"}}},
        {"decision": {"action": {}, "arguments": {"query": "leave"}}},
        {"decision": []},
    ]}
    assert tool_scores(result) == (0, 3)


@pytest.mark.asyncio
async def test_invalid_judge_output_keeps_usage_and_raw_response():
    class FakeCompletions:
        async def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="broken JSON"))],
                                   usage=SimpleNamespace(total_tokens=123))
    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    result = await judge_response(client, "test", {"query": "q", "reference": "r"},
                                  {"answer": "a", "trace": [], "failure_events": []}, "correctness")
    assert result["passed"] is False
    assert result["tokens"] == 123
    assert result["raw_response"] == "broken JSON"
    assert result["usage_complete"] is True


def test_evidently_gate_actually_fails_on_regression(tmp_path):
    result = evidently_suite([{"correctness": False, "groundedness": True,
                              "completed": False, "usage_complete": True}], tmp_path)
    assert result["summary"]["failed_tests"] == 2
    assert result["summary"]["success_tests"] == 2
    assert (tmp_path / "evidently_regression.html").stat().st_size > 1000


def test_golden_fixture_matches_canonical_corpus():
    sources = json.loads((ROOT / "assistant/agentic/data/sources.json").read_text())
    golden = json.loads((ROOT / "golden.json").read_text())
    for case in golden:
        if case["status"] == "verified":
            assert any(s["current"] and s["authoritative"] and
                       s["facts"].get(case["fact_key"]) == case["fact_value"] for s in sources)
