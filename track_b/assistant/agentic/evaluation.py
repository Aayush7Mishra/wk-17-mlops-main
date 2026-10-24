"""Dependency-free behavior evaluation, built from scratch without an eval framework."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic import AgentConfig, OfflinePolicy, OpenAIPlanner, SourceStore, VerificationAgent
from agentic.planners import SYSTEM_PROMPT
from agentic.sources import DATA_PATH, TOOLS, words

DEFAULT_CASES = Path(__file__).resolve().parents[1] / "evaluation" / "cases.json"


def score_case(case: dict, result: dict) -> dict:
    completed = result["status"] == case["expected_status"]
    if case.get("expected_key"):
        completed = completed and any(
            claim["key"] == case["expected_key"] and claim["value"] == case["expected_value"]
            for claim in result["claims"]
        )
    calls = []
    for event in result["trace"]:
        decision = event.get("decision", {})
        if not isinstance(decision, dict):
            continue
        action = decision.get("action", "")
        if isinstance(action, str) and action in {"finish", "clarify"}:
            continue
        arguments = decision.get("arguments", {})
        valid = (
            isinstance(action, str)
            and action in TOOLS
            and isinstance(arguments, dict)
            and set(arguments) == {"query"}
            and isinstance(arguments.get("query"), str)
            and 1 <= len(arguments["query"].strip()) <= 500
        )
        appropriate = valid and bool(words(arguments["query"]) & set(case["search_terms"]))
        calls.append(
            {
                "step": event["step"],
                "name": action,
                "valid": valid,
                "appropriate": appropriate,
                "correct": bool(valid and appropriate),
            }
        )
    categories = {f["category"] for f in result["failure_events"]}
    failure_class = None
    if not completed:
        failure_class = next(
            (
                kind
                for kind in ("cascading_soft_failure", "hard_failure", "soft_failure")
                if kind in categories
            ),
            "soft_failure",
        )
    return {
        "id": case["id"],
        "completed": bool(completed),
        "status": result["status"],
        "expected_status": case["expected_status"],
        "steps": result["steps"],
        "reasonable_trajectory": result["steps"] <= case["max_reasonable_steps"],
        "tool_calls": calls,
        "correct_calls": sum(c["correct"] for c in calls),
        "total_calls": len(calls),
        "total_tokens": result["usage"]["total_tokens"],
        "usage": result["usage"],
        "failure_class": failure_class,
        "failure_events": result["failure_events"],
        "latency_ms": result["latency_ms"],
        "model_mode": result["model_mode"],
        "result": result,
    }


def run_metadata(planner, cases_path: Path) -> dict:
    return {
        "model": getattr(planner, "model", "handwritten-offline-policy"),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "system_prompt_sha256": hashlib.sha256(
            getattr(planner, "system_prompt", SYSTEM_PROMPT).encode()
        ).hexdigest(),
        "corpus_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
    }


def load_cases(path: Path) -> list[dict]:
    """Tiny local evaluation fixtures; no network IO or background threads."""
    return json.loads(path.read_text())


async def run_suite(
    *,
    config: AgentConfig | None = None,
    mode: str = "offline",
    output_dir: str | Path | None = None,
    cases_path: str | Path | None = None,
    planner=None,
) -> dict[str, Any]:
    if mode not in {"offline", "live"}:
        raise ValueError("mode must be offline or live")
    config = config or AgentConfig()
    cases = load_cases(Path(cases_path or DEFAULT_CASES))
    owned_planner = planner is None
    planner = planner or (OfflinePolicy() if mode == "offline" else OpenAIPlanner())
    rows = []
    try:
        for case in cases:
            case_config = replace(config, max_steps=case.get("max_steps", config.max_steps))
            if case.get("failure") == "registry_timeout":
                case_config = replace(case_config, tool_timeout_seconds=0.01)
            agent = VerificationAgent(
                planner=planner,
                tools=SourceStore.default(failure=case.get("failure")),
                config=case_config,
            )
            result = (await agent.run(case["query"])).to_dict()
            rows.append(score_case(case, result))
    finally:
        if owned_planner and hasattr(planner, "close"):
            await planner.close()
    total_calls = sum(row["total_calls"] for row in rows)
    correct_calls = sum(row["correct_calls"] for row in rows)
    failures = Counter(row["failure_class"] for row in rows if row["failure_class"])
    metrics = {
        "task_completion_rate": sum(row["completed"] for row in rows) / len(rows),
        "tool_call_correctness": correct_calls / total_calls if total_calls else None,
        "mean_trajectory_length": statistics.mean(row["steps"] for row in rows),
        "reasonable_trajectory_rate": sum(row["reasonable_trajectory"] for row in rows) / len(rows),
        "total_tokens": sum(row["total_tokens"] for row in rows),
        "failure_counts": dict(failures),
        "query_count": len(rows),
        "tool_call_count": total_calls,
        "correct_tool_call_count": correct_calls,
    }
    report = {
        "mode": planner.mode,
        "metadata": run_metadata(planner, Path(cases_path or DEFAULT_CASES)),
        "config": asdict(config),
        "metrics": metrics,
        "rows": rows,
        "limitations": (
            "Offline results test a handwritten policy, not model quality. Live results use a "
            "small synthetic policy corpus and lexical tool relevance oracle; semantic correctness "
            "is scoped to structured facts. Token estimates are labeled. The final step-budget "
            "case deliberately cannot complete."
        ),
    }
    if output_dir is not None:
        save_report(report, Path(output_dir))
    return report


def save_report(report: dict, destination: Path) -> None:
    rows = report["rows"]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "traces.jsonl").write_text(
        "".join(json.dumps(row["result"]) + "\n" for row in rows)
    )
    failed = [
        {
            "id": row["id"],
            "category": row["failure_class"],
            "status": row["status"],
            "expected": row["expected_status"],
            "events": row["failure_events"],
        }
        for row in rows
        if not row["completed"]
    ]
    (destination / "failures.json").write_text(json.dumps(failed, indent=2) + "\n")
    (destination / "report.md").write_text(render_report(report))


def render_report(report: dict) -> str:
    m = report["metrics"]
    correctness = (
        f"{m['tool_call_correctness']:.1%}" if m["tool_call_correctness"] is not None else "N/A"
    )
    lines = [
        "# Verification evaluation",
        "",
        f"Mode: **{report['mode']}**. " + report["limitations"],
        f"Model/policy: `{report.get('metadata', {}).get('model', 'unspecified')}`.",
        "",
        f"Completion: **{m['task_completion_rate']:.1%}**; tool correctness: **{correctness}**; "
        f"mean steps: **{m['mean_trajectory_length']:.2f}**.",
        "",
        "Tokens include every iteration. Offline counts use ceil(JSON characters / 4) "
        "and are simulated estimates, not model consumption. Live counts use provider usage "
        "when available. Cost is null unless explicit input/output rates are configured.",
        "",
        "| Query | Complete | Outcome | Steps | Correct tools | Tokens | Failure class |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['id']} | {row['completed']} | {row['status']} | {row['steps']} | "
            f"{row['correct_calls']}/{row['total_calls']} | {row['total_tokens']} | "
            f"{row['failure_class'] or '-'} |"
        )
    lines += [
        "",
        "## Failure observations",
        "",
        "Hard failure means an explicit tool/provider outage or timeout; soft failure means a "
        "plausible invalid decision, malformed data, wrong answer, or incomplete task; cascading "
        "soft failure means repeating a rejected decision without using feedback. Recovered "
        "failures remain in traces even when safe abstention satisfies the task.",
        "",
    ]
    for row in report["rows"]:
        for failure in row["failure_events"]:
            lines.append(
                f"- {row['id']}, step {failure['step']}: "
                f"{failure['category']}: {failure['message']}."
            )
    lines += [
        "",
        "The step-budget case intentionally has one iteration: stopping safely does not count as "
        "answering its policy question. The harness records a soft task failure. "
        "Unit tests force repeated invalid actions and outages to check all taxonomy branches.",
        "",
    ]
    return "\n".join(lines)
