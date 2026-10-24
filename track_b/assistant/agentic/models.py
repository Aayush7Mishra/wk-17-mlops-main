"""Small serializable interfaces shared by CLI, HTTP API, and experiment harnesses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    top_k: int = 2
    max_notes: int = 8
    max_excerpt_chars: int = 600
    max_tokens: int = 12000
    model_timeout_seconds: float = 30.0
    tool_timeout_seconds: float = 5.0
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.max_steps <= 20:
            raise ValueError("max_steps must be between 1 and 20")
        if not 1 <= self.top_k <= 5:
            raise ValueError("top_k must be between 1 and 5")
        if not 2 <= self.max_notes <= 20:
            raise ValueError("max_notes must be between 2 and 20")
        if not 80 <= self.max_excerpt_chars <= 2000 or self.max_tokens < 100:
            raise ValueError("invalid context/token budget")
        if min(self.model_timeout_seconds, self.tool_timeout_seconds) <= 0:
            raise ValueError("timeouts must be positive")
        for price in (self.input_price_per_million, self.output_price_per_million):
            if price is not None and price < 0:
                raise ValueError("token prices must be non-negative")


@dataclass
class PlannerReply:
    decision: dict[str, Any]
    input_tokens: int
    output_tokens: int
    estimated: bool = False
    raw_response: str = ""


@dataclass
class AgentResult:
    query: str
    status: str
    answer: str
    citations: list[str]
    steps: int
    trace: list[dict[str, Any]]
    usage: dict[str, Any]
    failure_events: list[dict[str, Any]]
    model_mode: str
    latency_ms: float
    claims: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
