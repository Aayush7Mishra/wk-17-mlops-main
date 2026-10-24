"""Validated API and internal data contracts."""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoryMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=12_000)
    session_id: str = Field(min_length=1, max_length=128)
    use_rag: bool = True
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must contain non-whitespace characters")
        return value


class Source(StrictModel):
    document_id: str
    chunk_id: str
    filename: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolExecution(StrictModel):
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: Any = None
    error: str | None = None


class GeneratedAnswer(StrictModel):
    answer: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    follow_up_questions: list[str] = Field(default_factory=list, max_length=3)


class ChatResponse(GeneratedAnswer):
    sources: list[Source] = Field(default_factory=list)
    tool_calls: list[ToolExecution] = Field(default_factory=list)
    provider: str
    cached: bool = False
    degraded: bool = False
    request_id: str
    latency_ms: float = Field(ge=0.0)


class BatchChatRequest(StrictModel):
    requests: list[ChatRequest] = Field(min_length=1, max_length=8)


class BatchChatResponse(StrictModel):
    responses: list[ChatResponse]
    count: int = Field(ge=1, le=8)


class TextDocumentRequest(StrictModel):
    text: str = Field(min_length=1, max_length=1_000_000)
    filename: str = Field(default="pasted-text.txt", min_length=1, max_length=255)
    session_id: str = Field(min_length=1, max_length=128)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        def inspect(item: Any, *, depth: int = 0) -> None:
            if depth > 4:
                raise ValueError("metadata nesting must not exceed four levels")
            if isinstance(item, dict):
                if len(item) > 50:
                    raise ValueError("metadata objects must not exceed 50 keys")
                for key, child in item.items():
                    if len(str(key)) > 100:
                        raise ValueError("metadata keys must not exceed 100 characters")
                    inspect(child, depth=depth + 1)
            elif isinstance(item, list):
                if len(item) > 100:
                    raise ValueError("metadata lists must not exceed 100 items")
                for child in item:
                    inspect(child, depth=depth + 1)
            elif isinstance(item, str) and len(item) > 2000:
                raise ValueError("metadata strings must not exceed 2000 characters")

        inspect(value)
        try:
            encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must contain finite JSON values") from exc
        if len(encoded.encode("utf-8")) > 16_000:
            raise ValueError("metadata must not exceed 16000 encoded bytes")
        return value


class IngestResponse(StrictModel):
    document_id: str
    filename: str
    session_id: str
    characters: int
    chunks: int
    collection: str


class HealthResponse(StrictModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    dependencies: dict[str, str]


class ErrorResponse(StrictModel):
    error: str
    detail: str
    request_id: str | None = None
