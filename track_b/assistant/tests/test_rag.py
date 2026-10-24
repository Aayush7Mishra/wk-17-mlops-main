from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from app.rag import (
    DocumentChunker,
    HashingEmbedder,
    InMemoryVectorStore,
    RAGError,
    RAGService,
    extract_document_text,
)


def test_chunker_produces_bounded_stable_chunks_with_overlap() -> None:
    text = "First paragraph has useful facts. " * 12 + "\n\n" + "Second section. " * 15
    chunker = DocumentChunker(chunk_size=220, overlap=40)

    first = chunker.split(text, document_id="doc", filename="facts.md")
    second = chunker.split(text, document_id="doc", filename="facts.md")

    assert len(first) > 1
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(0 < len(chunk.text) <= 220 for chunk in first)
    assert all(a.end > b.start for a, b in zip(first, first[1:], strict=False))


@pytest.mark.asyncio
async def test_ingest_and_retrieve_round_trip() -> None:
    service = RAGService(
        embedder=HashingEmbedder(dimension=128),
        store=InMemoryVectorStore(),
        chunker=DocumentChunker(chunk_size=220, overlap=30),
        collection_name="test",
        minimum_score=-1.0,
        default_top_k=2,
    )

    result = await service.ingest_text(
        filename="policy.md",
        text="Refunds are available for thirty days. Include the order number.",
        metadata={"team": "support"},
    )
    sources = await service.retrieve("How long is the refund window?")

    assert result.chunks == 1
    assert service.revision == 1
    assert sources[0].filename == "policy.md"
    assert sources[0].metadata["team"] == "support"
    assert "thirty days" in sources[0].text


@pytest.mark.asyncio
async def test_retrieval_is_isolated_by_namespace() -> None:
    service = RAGService(
        embedder=HashingEmbedder(dimension=64),
        store=InMemoryVectorStore(),
        chunker=DocumentChunker(chunk_size=220, overlap=30),
        collection_name="test",
        minimum_score=-1.0,
    )
    await service.ingest_text(
        filename="private.md",
        text="Tenant A private policy.",
        namespace="tenant-a",
    )

    assert await service.retrieve("private policy", namespace="tenant-b") == []
    own_sources = await service.retrieve("private policy", namespace="tenant-a")
    assert own_sources[0].filename == "private.md"


def test_text_loader_rejects_unsupported_and_non_utf8_files() -> None:
    with pytest.raises(RAGError, match="Supported document types"):
        extract_document_text("image.png", b"not an image")
    with pytest.raises(RAGError, match="UTF-8"):
        extract_document_text("notes.txt", b"\xff\xfe")


def test_pdf_loader_enforces_page_limit_before_extraction() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    output = io.BytesIO()
    writer.write(output)

    with pytest.raises(RAGError, match="page extraction limit"):
        extract_document_text("large.pdf", output.getvalue(), max_pdf_pages=1)


@pytest.mark.asyncio
async def test_ingestion_does_not_commit_when_a_later_embedding_batch_fails() -> None:
    class FailingEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        async def embed(self, texts: list[str]) -> list[list[float]]:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("planned embedding failure")
            return [[1.0, 0.0] for _ in texts]

    store = InMemoryVectorStore()
    service = RAGService(
        embedder=FailingEmbedder(),
        store=store,
        chunker=DocumentChunker(chunk_size=10, overlap=0),
        collection_name="test",
        embedding_batch_size=1,
    )

    with pytest.raises(RuntimeError, match="planned"):
        await service.ingest_text(filename="multi.txt", text="one two three four five six")

    assert store._points == {}
