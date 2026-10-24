"""Bounded read-only tools over separate, versioned source collections."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "data" / "sources.json"
TOOLS = {"search_handbook": "handbook", "search_policy_registry": "registry"}
STOPWORDS = {
    "what",
    "are",
    "the",
    "is",
    "how",
    "many",
    "do",
    "does",
    "and",
    "for",
    "per",
    "can",
    "a",
    "to",
    "of",
    "policy",
    "employees",
    "employee",
    "receive",
}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


class SourceStore:
    """No network fan-out, stateful delegation, or hidden model calls inside these tools."""

    def __init__(self, documents: list[dict[str, Any]], *, failure: str | None = None):
        self.documents = documents
        self.failure = failure

    @classmethod
    def default(cls, *, failure: str | None = None) -> SourceStore:
        return cls(json.loads(DATA_PATH.read_text()), failure=failure)

    async def execute(self, name: str, arguments: dict[str, Any], *, top_k: int) -> dict:
        if name not in TOOLS:
            raise ValueError("unknown tool")
        if set(arguments) != {"query"} or not isinstance(arguments.get("query"), str):
            raise ValueError("tool requires exactly one string query")
        query = arguments["query"].strip()
        if not 1 <= len(query) <= 500:
            raise ValueError("query must contain 1 to 500 characters")
        if name == "search_policy_registry":
            if self.failure == "registry_unavailable":
                raise ConnectionError("policy registry unavailable (injected)")
            if self.failure == "registry_timeout":
                await asyncio.sleep(60)
            if self.failure == "malformed_registry":
                return {"documents": "invalid injected payload"}
        candidates = []
        for doc in self.documents:
            if doc["collection"] == TOOLS[name]:
                score = len(words(query) & words(doc["title"] + " " + doc["text"]))
                if score:
                    candidates.append((score, doc))
        candidates.sort(key=lambda item: (-item[0], item[1]["id"]))
        return {"documents": [dict(doc, relevance=score) for score, doc in candidates[:top_k]]}
