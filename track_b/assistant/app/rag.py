"""Document ingestion, embedding, vector persistence, and retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import io
import math
import re
import threading
import uuid
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol
from weakref import WeakKeyDictionary

from app.config import Settings
from app.schemas import IngestResponse, Source

_WORKER_LIMITS: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    asyncio.Semaphore,
] = WeakKeyDictionary()
_WORKER_LIMITS_LOCK = threading.Lock()


def _worker_limit_for_loop(loop: asyncio.AbstractEventLoop) -> asyncio.Semaphore:
    with _WORKER_LIMITS_LOCK:
        limit = _WORKER_LIMITS.get(loop)
        if limit is None:
            limit = asyncio.Semaphore(8)
            _WORKER_LIMITS[loop] = limit
        return limit


async def _run_blocking(function: Any, /, *args: Any, **kwargs: Any) -> Any:
    """Run blocking work off-loop with a process-local active-worker bound."""

    loop = asyncio.get_running_loop()
    async with _worker_limit_for_loop(loop):
        # A one-call executor avoids retaining workers across independent event loops. Its
        # non-blocking shutdown also prevents cancellation from freezing the event loop.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-worker")
        try:
            return await loop.run_in_executor(executor, partial(function, *args, **kwargs))
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


class RAGError(RuntimeError):
    """A safe, user-facing retrieval or ingestion failure."""


class DocumentValidationError(RAGError):
    """The supplied document is unsupported or malformed."""


class RAGDependencyError(RAGError):
    """An embedding or vector-store dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class TextChunk:
    id: str
    document_id: str
    filename: str
    text: str
    start: int
    end: int
    index: int
    namespace: str
    metadata: dict[str, Any]


class DocumentChunker:
    """Paragraph-aware character chunker with bounded overlap."""

    def __init__(self, chunk_size: int = 900, overlap: int = 140) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(
        self,
        text: str,
        *,
        document_id: str,
        filename: str,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
        max_chunks: int | None = None,
    ) -> list[TextChunk]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        boundaries = {0, len(normalized)}
        boundaries.update(
            match.end() for match in re.finditer(r"\n\s*\n|(?<=[.!?])\s+", normalized)
        )
        ordered_boundaries = sorted(boundaries)
        chunks: list[TextChunk] = []
        start = 0
        index = 0

        while start < len(normalized):
            hard_end = min(start + self.chunk_size, len(normalized))
            boundary_position = bisect_right(ordered_boundaries, hard_end) - 1
            candidate_end = ordered_boundaries[boundary_position]
            end = candidate_end if candidate_end > start else hard_end
            if end <= start:
                end = hard_end
            chunk_text = normalized[start:end].strip()
            if chunk_text:
                if max_chunks is not None and len(chunks) >= max_chunks:
                    raise DocumentValidationError(f"Document exceeds the {max_chunks}-chunk limit")
                chunk_id = str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{index}:{start}:{end}")
                )
                chunks.append(
                    TextChunk(
                        id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        text=chunk_text,
                        start=start,
                        end=end,
                        index=index,
                        namespace=namespace,
                        metadata=dict(metadata or {}),
                    )
                )
                index += 1
            if end >= len(normalized):
                break
            next_start = max(end - self.overlap, start + 1)
            whitespace = normalized.find(" ", next_start, min(end + 1, len(normalized)))
            start = whitespace + 1 if whitespace != -1 else next_start
        return chunks


def extract_document_text(
    filename: str,
    data: bytes,
    *,
    max_characters: int = 2_000_000,
    max_pdf_pages: int = 500,
) -> str:
    """Extract UTF-8 text or PDF page text while rejecting unsupported formats."""

    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            if len(reader.pages) > max_pdf_pages:
                raise DocumentValidationError(
                    f"PDF exceeds the {max_pdf_pages}-page extraction limit"
                )
            pages: list[str] = []
            extracted_characters = 0
            for page in reader.pages:
                page_text = (page.extract_text() or "").strip()
                extracted_characters += len(page_text)
                if extracted_characters > max_characters:
                    raise DocumentValidationError(
                        f"Extracted document exceeds the {max_characters}-character limit"
                    )
                if page_text:
                    pages.append(page_text)
            text = "\n\n".join(page for page in pages if page)
        except DocumentValidationError:
            raise
        except Exception as exc:  # library-specific parse failures vary
            raise DocumentValidationError("The PDF could not be parsed") from exc
    elif suffix in {".txt", ".md", ".rst", ".csv", ".json"}:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentValidationError("Text documents must use UTF-8 encoding") from exc
    else:
        raise DocumentValidationError(
            "Supported document types: .pdf, .txt, .md, .rst, .csv, and .json"
        )
    if not text.strip():
        raise DocumentValidationError("The document does not contain extractable text")
    return text


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic offline embedding for tests and no-network demonstrations."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[\w'-]+", text.casefold())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class FastEmbedder:
    """Lazy ONNX-backed semantic embeddings through FastEmbed."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._lock = asyncio.Lock()

    async def _get_model(self) -> Any:
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    try:
                        from fastembed import TextEmbedding

                        self._model = await _run_blocking(
                            TextEmbedding,
                            model_name=self.model_name,
                        )
                    except ImportError as exc:
                        raise RAGDependencyError("fastembed is not installed") from exc
                    except Exception as exc:
                        raise RAGDependencyError("Embedding model initialization failed") from exc
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = await self._get_model()

        def encode() -> list[list[float]]:
            return [vector.tolist() for vector in model.embed(texts)]

        try:
            return await _run_blocking(encode)
        except Exception as exc:
            raise RAGDependencyError("Embedding generation failed") from exc


class VectorStore(Protocol):
    async def upsert(self, chunks: list[TextChunk], vectors: list[list[float]]) -> None: ...

    async def search(
        self,
        vector: list[float],
        limit: int,
        namespace: str,
    ) -> list[Source]: ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...


class InMemoryVectorStore:
    """Process-local vector store used for tests and lightweight offline runs."""

    def __init__(self) -> None:
        self._points: dict[str, tuple[list[float], TextChunk]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, chunks: list[TextChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("each chunk must have exactly one embedding")
        async with self._lock:
            self._points.update(
                {chunk.id: (vector, chunk) for chunk, vector in zip(chunks, vectors, strict=True)}
            )

    async def search(self, vector: list[float], limit: int, namespace: str) -> list[Source]:
        async with self._lock:
            points = [point for point in self._points.values() if point[1].namespace == namespace]
        scored: list[tuple[float, TextChunk]] = []
        query_norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        for candidate, chunk in points:
            candidate_norm = math.sqrt(sum(value * value for value in candidate)) or 1.0
            score = sum(a * b for a, b in zip(vector, candidate, strict=False)) / (
                query_norm * candidate_norm
            )
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [_source_from_chunk(chunk, score) for score, chunk in scored[:limit]]

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def _source_from_chunk(chunk: TextChunk, score: float) -> Source:
    metadata = {"chunk_index": chunk.index, "start": chunk.start, "end": chunk.end}
    metadata.update(chunk.metadata)
    return Source(
        document_id=chunk.document_id,
        chunk_id=chunk.id,
        filename=chunk.filename,
        text=chunk.text,
        score=float(score),
        metadata=metadata,
    )


class QdrantVectorStore:
    """Qdrant-backed persistent vector store with lazy network connection."""

    def __init__(self, url: str, collection_name: str) -> None:
        self.url = url
        self.collection_name = collection_name
        self._client: Any = None
        self._collection_lock = asyncio.Lock()
        self._collection_ready = False

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RAGDependencyError("qdrant-client is not installed") from exc
            self._client = QdrantClient(url=self.url, timeout=8)
        return self._client

    async def _ensure_collection(self, dimension: int) -> None:
        if self._collection_ready:
            return
        async with self._collection_lock:
            if self._collection_ready:
                return
            client = self._get_client()

            def ensure() -> None:
                from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

                if not client.collection_exists(self.collection_name):
                    try:
                        client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
                        )
                    except Exception:
                        # Another replica may have won the first-ingestion race.
                        if not client.collection_exists(self.collection_name):
                            raise
                collection = client.get_collection(self.collection_name)
                payload_schema = getattr(collection, "payload_schema", {}) or {}
                if "namespace" not in payload_schema:
                    try:
                        client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name="namespace",
                            field_schema=PayloadSchemaType.KEYWORD,
                            wait=True,
                        )
                    except Exception:
                        refreshed = client.get_collection(self.collection_name)
                        if "namespace" not in (getattr(refreshed, "payload_schema", {}) or {}):
                            raise

            try:
                await _run_blocking(ensure)
                self._collection_ready = True
            except Exception as exc:
                raise RAGDependencyError("Vector database is unavailable") from exc

    async def upsert(self, chunks: list[TextChunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("each chunk must have exactly one embedding")
        await self._ensure_collection(len(vectors[0]))
        client = self._get_client()

        def write() -> None:
            from qdrant_client.models import PointStruct

            points = []
            for chunk, vector in zip(chunks, vectors, strict=True):
                payload = {
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.id,
                    "filename": chunk.filename,
                    "text": chunk.text,
                    "chunk_index": chunk.index,
                    "start": chunk.start,
                    "end": chunk.end,
                    "namespace": chunk.namespace,
                    "metadata": chunk.metadata,
                }
                points.append(PointStruct(id=chunk.id, vector=vector, payload=payload))
            client.upsert(collection_name=self.collection_name, points=points, wait=True)

        try:
            await _run_blocking(write)
        except Exception as exc:
            raise RAGDependencyError("Document vectors could not be stored") from exc

    async def search(self, vector: list[float], limit: int, namespace: str) -> list[Source]:
        client = self._get_client()

        def query() -> list[Any]:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            if not client.collection_exists(self.collection_name):
                return []
            query_filter = Filter(
                must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
            )
            if hasattr(client, "query_points"):
                return list(
                    client.query_points(
                        collection_name=self.collection_name,
                        query=vector,
                        query_filter=query_filter,
                        limit=limit,
                        with_payload=True,
                    ).points
                )
            return list(
                client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
            )

        try:
            points = await _run_blocking(query)
        except Exception as exc:
            raise RAGDependencyError("Knowledge retrieval is temporarily unavailable") from exc
        sources: list[Source] = []
        for point in points:
            payload = dict(point.payload or {})
            metadata = {
                "chunk_index": payload.get("chunk_index", 0),
                "start": payload.get("start", 0),
                "end": payload.get("end", 0),
            }
            metadata.update(payload.get("metadata") or {})
            sources.append(
                Source(
                    document_id=str(payload.get("document_id", "unknown")),
                    chunk_id=str(payload.get("chunk_id", point.id)),
                    filename=str(payload.get("filename", "unknown")),
                    text=str(payload.get("text", "")),
                    score=float(point.score),
                    metadata=metadata,
                )
            )
        return sources

    async def ping(self) -> bool:
        try:
            await _run_blocking(self._get_client().get_collections)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._client is not None:
            await _run_blocking(self._client.close)


class RAGService:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        chunker: DocumentChunker,
        collection_name: str,
        minimum_score: float = 0.2,
        default_top_k: int = 4,
        max_document_characters: int = 2_000_000,
        max_document_chunks: int = 2500,
        embedding_batch_size: int = 64,
        max_pdf_pages: int = 500,
        embedding_concurrency: int = 2,
        retrieval_concurrency: int = 4,
        ingestion_concurrency: int = 2,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self.collection_name = collection_name
        self.minimum_score = minimum_score
        self.default_top_k = default_top_k
        self.max_document_characters = max_document_characters
        self.max_document_chunks = max_document_chunks
        self.embedding_batch_size = embedding_batch_size
        self.max_pdf_pages = max_pdf_pages
        self._embedding_slots = asyncio.Semaphore(embedding_concurrency)
        self._retrieval_slots = asyncio.Semaphore(retrieval_concurrency)
        self._ingestion_slots = asyncio.Semaphore(ingestion_concurrency)
        self.revision = 0

    async def ingest_bytes(
        self,
        *,
        filename: str,
        data: bytes,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        async with self._ingestion_slots:
            text = await _run_blocking(
                extract_document_text,
                filename,
                data,
                max_characters=self.max_document_characters,
                max_pdf_pages=self.max_pdf_pages,
            )
            return await self._ingest_text(
                filename=filename,
                text=text,
                namespace=namespace,
                metadata=metadata,
            )

    async def ingest_text(
        self,
        *,
        filename: str,
        text: str,
        namespace: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        async with self._ingestion_slots:
            return await self._ingest_text(
                filename=filename,
                text=text,
                namespace=namespace,
                metadata=metadata,
            )

    async def _ingest_text(
        self,
        *,
        filename: str,
        text: str,
        namespace: str,
        metadata: dict[str, Any] | None,
    ) -> IngestResponse:
        clean_text = text.strip()
        if not clean_text:
            raise DocumentValidationError("The document does not contain text")
        if len(clean_text) > self.max_document_characters:
            raise DocumentValidationError(
                f"Extracted document exceeds the {self.max_document_characters}-character limit"
            )
        document_fingerprint = f"{namespace}\0{clean_text}".encode()
        document_id = hashlib.sha256(document_fingerprint).hexdigest()[:24]
        chunks = await _run_blocking(
            self.chunker.split,
            clean_text,
            document_id=document_id,
            filename=Path(filename).name,
            namespace=namespace,
            metadata=metadata,
            max_chunks=self.max_document_chunks,
        )
        if not chunks:
            raise DocumentValidationError("The document did not produce any chunks")
        if len(chunks) > self.max_document_chunks:
            raise DocumentValidationError(
                f"Document exceeds the {self.max_document_chunks}-chunk limit"
            )
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self.embedding_batch_size):
            batch = chunks[start : start + self.embedding_batch_size]
            async with self._embedding_slots:
                vectors.extend(await self.embedder.embed([chunk.text for chunk in batch]))
        # Commit only after every embedding batch succeeds; failed generation cannot leave a
        # partially searchable document.
        await self.store.upsert(chunks, vectors)
        self.revision += 1
        return IngestResponse(
            document_id=document_id,
            filename=Path(filename).name,
            session_id=namespace,
            characters=len(clean_text),
            chunks=len(chunks),
            collection=self.collection_name,
        )

    async def retrieve(
        self,
        query: str,
        limit: int | None = None,
        *,
        namespace: str = "default",
    ) -> list[Source]:
        if not query.strip():
            return []
        async with self._retrieval_slots:
            async with self._embedding_slots:
                vector = (await self.embedder.embed([query]))[0]
            results = await self.store.search(vector, limit or self.default_top_k, namespace)
        return [source for source in results if source.score >= self.minimum_score]

    async def close(self) -> None:
        await self.store.close()


def build_rag_service(settings: Settings) -> RAGService:
    embedder: Embedder
    if settings.embedding_provider == "fastembed":
        embedder = FastEmbedder(settings.embedding_model)
    else:
        embedder = HashingEmbedder(settings.embedding_dimension)
    store: VectorStore
    if settings.vector_store == "qdrant":
        store = QdrantVectorStore(settings.qdrant_url, settings.collection_name)
    else:
        store = InMemoryVectorStore()
    return RAGService(
        embedder=embedder,
        store=store,
        chunker=DocumentChunker(settings.chunk_size, settings.chunk_overlap),
        collection_name=settings.collection_name,
        minimum_score=settings.minimum_relevance_score,
        default_top_k=settings.retrieval_top_k,
        max_document_characters=settings.max_document_characters,
        max_document_chunks=settings.max_document_chunks,
        embedding_batch_size=settings.embedding_batch_size,
        max_pdf_pages=settings.max_pdf_pages,
        embedding_concurrency=settings.embedding_concurrency,
        retrieval_concurrency=settings.retrieval_concurrency,
        ingestion_concurrency=settings.ingestion_concurrency,
    )
