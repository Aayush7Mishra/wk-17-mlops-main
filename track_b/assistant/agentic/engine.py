"""Adaptive, bounded verification loop with explicit evidence and failure boundaries."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from agentic.models import AgentConfig, AgentResult
from agentic.planners import Planner
from agentic.sources import TOOLS, SourceStore

LABELS = {
    "annual_leave_days": "Annual leave",
    "carryover_days": "Leave carryover",
    "remote_days": "Remote work",
    "meal_allowance": "Meal reimbursement",
}
SAFE_ABSTENTION = "Unable to verify this policy from the available evidence."
SAFE_CLARIFICATION = "Which employee policy and employment type do you mean?"


def validate_documents(payload: Any) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
        raise ValueError("malformed source response: documents must be a list")
    documents = payload["documents"]
    for doc in documents:
        if not isinstance(doc, dict):
            raise ValueError("malformed source document")
        for key in ("id", "publisher", "topic", "text", "version", "title", "collection"):
            if not isinstance(doc.get(key), str) or not doc[key]:
                raise ValueError(f"malformed source field: {key}")
        if not isinstance(doc.get("authoritative"), bool) or not isinstance(
            doc.get("current"), bool
        ):
            raise ValueError("malformed source authority flags")
        facts = doc.get("facts")
        if (
            not isinstance(facts, dict)
            or not facts
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in facts.items())
        ):
            raise ValueError("malformed source facts")
    return documents


def verified_claims(decision: dict, evidence: dict[str, dict]) -> list[dict]:
    claims = decision.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= 5:
        raise ValueError("verified answer requires 1 to 5 claims")
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"key", "value", "citations"}:
            raise ValueError("invalid claim schema")
        key, value, citations = claim["key"], claim["value"], claim["citations"]
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("invalid fact key/value")
        if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
            raise ValueError("invalid citations")
        if any(c not in evidence for c in citations):
            raise ValueError("unobserved citation")
        cited = [evidence[c] for c in set(citations)]
        if any(key not in doc["facts"] for doc in cited):
            raise ValueError("citation does not address the claimed fact")
        if len({doc["publisher"] for doc in cited}) < 2:
            raise ValueError("two distinct source publishers must be consulted")
        all_relevant = [doc for doc in evidence.values() if key in doc["facts"]]
        authorities = [doc for doc in all_relevant if doc["authoritative"] and doc["current"]]
        if authorities:
            latest = max(doc["version"] for doc in authorities)
            latest_docs = [doc for doc in authorities if doc["version"] == latest]
            if {doc["facts"][key] for doc in latest_docs} != {value}:
                raise ValueError("claim contradicts current authoritative evidence")
            if not any(doc in cited for doc in latest_docs):
                raise ValueError("current authoritative evidence must be cited")
        elif not all(doc["current"] and doc["facts"][key] == value for doc in cited):
            raise ValueError("unresolved conflict or stale corroboration")
        elif any(doc["current"] and doc["facts"][key] != value for doc in all_relevant):
            raise ValueError("unresolved conflicting evidence")
    return claims


class VerificationAgent:
    def __init__(self, *, planner: Planner, tools: SourceStore, config: AgentConfig | None = None):
        self.planner = planner
        self.tools = tools
        self.config = config or AgentConfig()

    async def run(self, query: str) -> AgentResult:
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 2000:
            raise ValueError("query must contain 1 to 2000 characters")
        started = time.perf_counter()
        config = self.config
        evidence: dict[str, dict] = {}
        notes: list[dict] = []
        history: list[dict] = []
        trace: list[dict] = []
        failures: list[dict] = []
        input_tokens = output_tokens = 0
        estimated = self.planner.mode != "live"
        usage_complete = True
        status, answer = "budget_exhausted", "Verification stopped at the configured step limit."
        claims: list[dict] = []
        citations: list[str] = []
        last_soft_error = None

        def failure(kind: str, message: str, step: int) -> None:
            nonlocal last_soft_error
            if kind == "soft_failure" and last_soft_error == message:
                kind = "cascading_soft_failure"
            failures.append({"step": step, "category": kind, "message": message})
            last_soft_error = message if kind != "hard_failure" else None

        for step in range(1, config.max_steps + 1):
            if input_tokens + output_tokens >= config.max_tokens:
                answer = "Verification stopped at the configured token budget."
                break
            context = {
                "query": query,
                "notes": notes,
                "history": history[-config.max_steps :],
                "remaining_steps": config.max_steps - step + 1,
                "max_steps": config.max_steps,
            }
            event: dict[str, Any] = {"step": step, "context": json.loads(json.dumps(context))}
            trace.append(event)
            try:
                async with asyncio.timeout(config.model_timeout_seconds):
                    reply = await self.planner.decide(context)
                input_tokens += reply.input_tokens
                output_tokens += reply.output_tokens
                estimated = estimated or reply.estimated
                event.update(
                    decision=reply.decision,
                    raw_response=reply.raw_response,
                    usage={
                        "input_tokens": reply.input_tokens,
                        "output_tokens": reply.output_tokens,
                        "estimated": reply.estimated,
                    },
                )
            except Exception as exc:
                # Unsuccessful requests can consume provider tokens without reporting usage.
                usage_complete = False
                event["error"] = f"planner failed: {type(exc).__name__}"
                failure("hard_failure", event["error"], step)
                status, answer = (
                    "error",
                    "The planner is unavailable; no policy answer was verified.",
                )
                break
            if input_tokens + output_tokens > config.max_tokens:
                answer = "Verification stopped after the last model call exceeded the token budget."
                event["observation"] = {"ok": False, "error": "token budget exceeded"}
                break
            decision = reply.decision
            try:
                if not isinstance(decision, dict):
                    raise ValueError("decision must be a JSON object")
                action = decision.get("action")
                if action in TOOLS:
                    arguments = decision.get("arguments")
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be an object")
                    event["tool_call"] = {"name": action, "arguments": arguments}
                    try:
                        async with asyncio.timeout(config.tool_timeout_seconds):
                            payload = await self.tools.execute(
                                action, arguments, top_k=config.top_k
                            )
                        event["raw_tool_result"] = payload
                        docs = validate_documents(payload)[: config.top_k]
                    except (ConnectionError, TimeoutError) as exc:
                        failure("hard_failure", f"{action}: {type(exc).__name__}", step)
                        event["observation"] = {"ok": False, "error": type(exc).__name__}
                    else:
                        for doc in docs:
                            evidence[doc["id"]] = doc
                            compact = {key: value for key, value in doc.items() if key != "text"}
                            compact["excerpt"] = doc["text"][: config.max_excerpt_chars]
                            notes = [note for note in notes if note["id"] != doc["id"]] + [compact]
                        notes = notes[-config.max_notes :]
                        event["observation"] = {
                            "ok": True,
                            "source_ids": [doc["id"] for doc in docs],
                        }
                elif action == "clarify":
                    question = decision.get("question")
                    if not isinstance(question, str) or not 1 <= len(question) <= 500:
                        raise ValueError("clarification requires a concise question")
                    # Unverified model text can contain policy assertions even when
                    # labeled as a question. Keep it in the trace, not the answer.
                    status, answer = "needs_clarification", SAFE_CLARIFICATION
                    event["observation"] = {"ok": True, "terminal": status}
                    break
                elif action == "finish":
                    if decision.get("status") == "verified":
                        claims = verified_claims(decision, evidence)
                        citations = list(
                            dict.fromkeys(c for claim in claims for c in claim["citations"])
                        )
                        answer = "\n".join(
                            f"{LABELS.get(c['key'], c['key'])}: {c['value']}. "
                            + " ".join(f"[{source}]" for source in c["citations"])
                            for c in claims
                        )
                        status = "verified"
                    elif decision.get("status") == "insufficient_evidence":
                        explanation = decision.get("answer")
                        if not isinstance(explanation, str) or not 1 <= len(explanation) <= 1500:
                            raise ValueError("abstention requires an explanation")
                        status = "insufficient_evidence"
                        # An abstention label does not validate its free-form text.
                        # Preserve the model explanation in event.decision while
                        # returning only a fixed, assertion-free controller message.
                        answer = SAFE_ABSTENTION
                    else:
                        raise ValueError("invalid finish status")
                    event["observation"] = {"ok": True, "terminal": status}
                    break
                else:
                    raise ValueError("unknown action")
            except (ValueError, TypeError, KeyError) as exc:
                failure("soft_failure", str(exc), step)
                event["observation"] = {"ok": False, "error": str(exc)}
            except Exception as exc:
                failure("hard_failure", f"tool failed: {type(exc).__name__}", step)
                event["observation"] = {"ok": False, "error": type(exc).__name__}
            if event["observation"]["ok"]:
                # Only consecutive repeated rejections count as a cascade. A
                # successful action shows the agent used the prior feedback.
                last_soft_error = None
            history.append(
                {
                    "action": decision.get("action", "invalid")
                    if isinstance(decision, dict)
                    else "invalid",
                    **event["observation"],
                }
            )

        cost = None
        if (
            config.input_price_per_million is not None
            and config.output_price_per_million is not None
        ):
            cost = (
                input_tokens * config.input_price_per_million
                + output_tokens * config.output_price_per_million
            ) / 1_000_000
        usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated": estimated,
            "complete": usage_complete,
            "cost_usd": cost,
            "basis": "estimated_simulated"
            if self.planner.mode != "live"
            else ("estimated" if estimated else "provider_reported"),
        }
        return AgentResult(
            query,
            status,
            answer,
            citations,
            len(trace),
            trace,
            usage,
            failures,
            self.planner.mode,
            (time.perf_counter() - started) * 1000,
            claims=claims if status == "verified" else [],
        )
