from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.schemas import TextDocumentRequest


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(chunk_size=200, chunk_overlap=200)


def test_document_metadata_has_bounded_encoded_size_and_depth() -> None:
    with pytest.raises(ValidationError, match="16000 encoded bytes"):
        TextDocumentRequest(
            text="document",
            session_id="test",
            metadata={f"key-{index}": "x" * 400 for index in range(50)},
        )

    with pytest.raises(ValidationError, match="four levels"):
        TextDocumentRequest(
            text="document",
            session_id="test",
            metadata={"a": {"b": {"c": {"d": {"e": "too deep"}}}}},
        )
