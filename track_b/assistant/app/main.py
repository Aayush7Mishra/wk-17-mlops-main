"""FastAPI entry point for the production assistant."""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from agentic import AgentConfig, OfflinePolicy, OpenAIPlanner, SourceStore, VerificationAgent
from app import __version__
from app.assistant import AssistantService
from app.cache import create_cache
from app.config import Settings, get_settings
from app.providers import build_provider_router
from app.rag import (
    DocumentValidationError,
    RAGDependencyError,
    RAGError,
    RAGService,
    build_rag_service,
)
from app.rate_limit import RateLimiter, create_rate_limiter
from app.schemas import (
    BatchChatRequest,
    BatchChatResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    TextDocumentRequest,
)
from app.tools import ToolRegistry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cache = await create_cache(
        redis_url=settings.redis_url,
        default_ttl_seconds=settings.cache_ttl_seconds,
        timeout_seconds=settings.redis_timeout_seconds,
    )
    rate_limiter = await create_rate_limiter(
        limit=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
        redis_url=settings.redis_url,
        timeout_seconds=settings.redis_timeout_seconds,
    )
    rag = build_rag_service(settings)
    provider = build_provider_router(settings)
    tools = ToolRegistry(
        retriever=lambda query, limit, namespace: rag.retrieve(
            query,
            limit,
            namespace=namespace,
        )
    )
    assistant = AssistantService(
        settings=settings,
        provider=provider,
        rag=rag,
        tools=tools,
        cache=cache,
    )
    app.state.settings = settings
    app.state.cache = cache
    app.state.rate_limiter = rate_limiter
    app.state.rag = rag
    app.state.provider = provider
    app.state.assistant = assistant
    planner = (
        OfflinePolicy()
        if settings.agent_mode == "offline"
        else OpenAIPlanner(
            model=settings.agent_model,
            base_url=settings.agent_base_url,
            api_key=settings.agent_api_key.get_secret_value() if settings.agent_api_key else None,
        )
    )
    app.state.verifier = VerificationAgent(
        planner=planner,
        tools=SourceStore.default(),
        config=AgentConfig(max_steps=settings.agent_max_steps),
    )
    app.state.verification_slots = asyncio.Semaphore(settings.batch_concurrency)
    logger.info("assistant backend started in %s mode", settings.environment)
    try:
        yield
    finally:
        await asyncio.gather(
            cache.close(),
            rate_limiter.close(),
            rag.close(),
            provider.close(),
            return_exceptions=True,
        )
        if hasattr(planner, "close"):
            await planner.close()
        logger.info("assistant backend stopped")


settings_at_import = get_settings()
app = FastAPI(
    title=settings_at_import.app_name,
    version=__version__,
    description="A production-oriented RAG assistant with tool calling and provider fallback.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings_at_import.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Client-ID", "X-Request-ID"],
)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or uuid.uuid4().hex


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "")
    if supplied.isascii() and supplied.strip():
        request.state.request_id = supplied[:128]
    else:
        request.state.request_id = uuid.uuid4().hex
    started = time.perf_counter()

    active_settings = getattr(request.app.state, "settings", settings_at_import)
    if request.method == "POST" and request.url.path.startswith("/api/v1/"):
        try:
            _require_authentication(request, active_settings)
        except HTTPException as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error="request_error",
                    detail=str(exc.detail),
                    request_id=request.state.request_id,
                ).model_dump(),
                headers=exc.headers,
            )
            response.headers["X-Request-ID"] = request.state.request_id
            return response

    if request.method == "POST" and request.url.path.startswith("/api/v1/"):
        content_length = request.headers.get("Content-Length")
        body_limit = active_settings.max_request_body_bytes
        if request.url.path == "/api/v1/documents":
            body_limit = active_settings.max_upload_bytes + 1_000_000
        if content_length:
            try:
                body_bytes = int(content_length)
            except ValueError:
                body_bytes = body_limit + 1
            if body_bytes > body_limit:
                response = JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content=ErrorResponse(
                        error="request_error",
                        detail=f"Request body exceeds the {body_limit}-byte limit",
                        request_id=request.state.request_id,
                    ).model_dump(),
                )
                response.headers["X-Request-ID"] = request.state.request_id
                return response

        body_chunks: list[bytes] = []
        received_bytes = 0
        async for chunk in request.stream():
            received_bytes += len(chunk)
            if received_bytes > body_limit:
                response = JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content=ErrorResponse(
                        error="request_error",
                        detail=f"Request body exceeds the {body_limit}-byte limit",
                        request_id=request.state.request_id,
                    ).model_dump(),
                )
                response.headers["X-Request-ID"] = request.state.request_id
                return response
            body_chunks.append(chunk)
        # Starlette replays a cached body to routing/form parsers after this middleware.
        request._body = b"".join(body_chunks)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    error = "rate_limited" if exc.status_code == 429 else "request_error"
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=error,
            detail=detail,
            request_id=_request_id(request),
        ).model_dump(),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {"msg": "Invalid request"}
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="validation_error",
            detail=str(first_error.get("msg", "Invalid request")),
            request_id=_request_id(request),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled request error", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="internal_error",
            detail="The request could not be completed",
            request_id=_request_id(request),
        ).model_dump(),
    )


def _services(request: Request) -> tuple[Settings, RAGService, AssistantService, RateLimiter]:
    return (
        request.app.state.settings,
        request.app.state.rag,
        request.app.state.assistant,
        request.app.state.rate_limiter,
    )


def _require_authentication(request: Request, settings: Settings) -> None:
    if settings.app_api_key is None:
        return
    expected = settings.app_api_key.get_secret_value()
    supplied = request.headers.get("X-API-Key", "")
    if not expected or not secrets.compare_digest(supplied, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def _apply_rate_limit(
    request: Request,
    response: Response,
    limiter: RateLimiter,
    *,
    suffix: str,
    cost: int = 1,
) -> None:
    client_host = request.client.host if request.client else "unknown"
    client_id = request.headers.get("X-Client-ID", "")
    authenticated_proxy = request.app.state.settings.app_api_key is not None
    identities = [f"ip:{client_host}"]
    if authenticated_proxy and client_id.isascii() and 0 < len(client_id) <= 128:
        identities.append(f"client:{client_id}")
    decisions = []
    for identity in identities:
        try:
            decisions.append(await limiter.check(f"{identity}:{suffix}", cost=cost))
        except Exception as exc:
            logger.warning("rate limiter backend unavailable", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate-limit service is temporarily unavailable",
            ) from exc
    decision = min(decisions, key=lambda item: item.remaining)
    for header, value in decision.as_headers().items():
        response.headers[header] = value
    denied = next((item for item in decisions if not item.allowed), None)
    if denied is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded; retry after the indicated delay",
            headers=denied.as_headers(),
        )


@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health(request: Request) -> HealthResponse:
    settings, rag, _, _ = _services(request)
    vector_ok = await rag.store.ping()
    cache_name = type(request.app.state.cache).__name__
    limiter_name = type(request.app.state.rate_limiter).__name__
    provider_names = ",".join(item.name for item in request.app.state.provider.providers)
    return HealthResponse(
        status="ok" if vector_ok else "degraded",
        version=__version__,
        environment=settings.environment,
        dependencies={
            "vector_store": "up" if vector_ok else "down",
            "cache": cache_name,
            "rate_limiter": limiter_name,
            "providers": provider_names,
        },
    )


@app.post("/api/v1/chat", response_model=ChatResponse, tags=["assistant"])
async def chat(
    payload: ChatRequest,
    request: Request,
    response: Response,
) -> ChatResponse:
    settings, _, assistant, limiter = _services(request)
    _require_authentication(request, settings)
    await _apply_rate_limit(request, response, limiter, suffix="chat")
    return await assistant.chat(payload, request_id=_request_id(request))


@app.post("/api/v1/chat/batch", response_model=BatchChatResponse, tags=["assistant"])
async def chat_batch(
    payload: BatchChatRequest,
    request: Request,
    response: Response,
) -> BatchChatResponse:
    settings, _, assistant, limiter = _services(request)
    _require_authentication(request, settings)
    await _apply_rate_limit(
        request,
        response,
        limiter,
        suffix="chat",
        cost=len(payload.requests),
    )

    async def bounded(index: int, item: ChatRequest) -> ChatResponse:
        item_request_id = f"{_request_id(request)}:{index + 1}"
        return await assistant.chat(item, request_id=item_request_id)

    results = await asyncio.gather(
        *(bounded(index, item) for index, item in enumerate(payload.requests))
    )
    return BatchChatResponse(responses=results, count=len(results))


@app.post("/api/v1/documents", response_model=IngestResponse, tags=["knowledge"])
async def ingest_document(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File()],
    session_id: Annotated[str, Form(min_length=1, max_length=128)],
) -> IngestResponse:
    settings, rag, _, limiter = _services(request)
    _require_authentication(request, settings)
    await _apply_rate_limit(request, response, limiter, suffix="ingest")
    filename = Path(file.filename or "document.txt").name
    data = await file.read(settings.max_upload_bytes + 1)
    content_type = file.content_type
    await file.close()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Document exceeds the {settings.max_upload_bytes}-byte upload limit",
        )
    try:
        result = await rag.ingest_bytes(
            filename=filename,
            data=data,
            namespace=session_id,
            metadata={"content_type": content_type},
        )
        return result
    except DocumentValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (RAGDependencyError, RAGError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.post("/api/v1/documents/text", response_model=IngestResponse, tags=["knowledge"])
async def ingest_text(
    payload: TextDocumentRequest,
    request: Request,
    response: Response,
) -> IngestResponse:
    settings, rag, _, limiter = _services(request)
    _require_authentication(request, settings)
    await _apply_rate_limit(request, response, limiter, suffix="ingest-text")
    try:
        result = await rag.ingest_text(
            filename=Path(payload.filename).name,
            text=payload.text,
            namespace=payload.session_id,
            metadata=payload.metadata,
        )
        return result
    except DocumentValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except (RAGDependencyError, RAGError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    return {
        "name": settings_at_import.app_name,
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


class VerificationRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@app.post("/api/v1/verify", tags=["verification"])
async def verify(payload: VerificationRequest, request: Request, response: Response) -> dict:
    settings, _, _, limiter = _services(request)
    _require_authentication(request, settings)
    await _apply_rate_limit(request, response, limiter, suffix="verify")
    if not payload.query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank")
    async with request.app.state.verification_slots:
        result = await request.app.state.verifier.run(payload.query)
    return result.to_dict()
