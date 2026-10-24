from __future__ import annotations

import importlib

import httpx
import pytest

from app.config import get_settings


@pytest.mark.asyncio
async def test_api_health_ingestion_chat_and_batch(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("PRIMARY_PROVIDER", "demo")
    monkeypatch.setenv("PRIMARY_MODEL", "demo")
    monkeypatch.setenv("PRIMARY_BASE_URL", "http://unused")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("MINIMUM_RELEVANCE_SCORE", "-1")
    monkeypatch.setenv("APP_API_KEY", "test-secret")
    monkeypatch.delenv("REDIS_URL", raising=False)
    get_settings.cache_clear()

    import app.main as main_module

    main_module = importlib.reload(main_module)
    transport = httpx.ASGITransport(app=main_module.app)
    async with (
        main_module.lifespan(main_module.app),
        httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client,
    ):
        health = await client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        unauthorized = await client.post(
            "/api/v1/chat",
            json={"message": "Hello", "session_id": "api", "use_rag": False},
        )
        assert unauthorized.status_code == 401

        headers = {"X-API-Key": "test-secret", "X-Client-ID": "api"}

        ingested = await client.post(
            "/api/v1/documents/text",
            json={
                "filename": "policy.md",
                "text": "Refunds are available for 30 days.",
                "session_id": "api",
            },
            headers=headers,
        )
        assert ingested.status_code == 200
        assert ingested.json()["chunks"] == 1

        chat = await client.post(
            "/api/v1/chat",
            json={"message": "What is the refund window?", "session_id": "api"},
            headers=headers,
        )
        assert chat.status_code == 200
        assert chat.json()["sources"][0]["filename"] == "policy.md"
        assert chat.headers["x-request-id"]
        assert chat.json()["request_id"] == chat.headers["x-request-id"]

        isolated = await client.post(
            "/api/v1/chat",
            json={"message": "What is the refund window?", "session_id": "other"},
            headers={"X-API-Key": "test-secret", "X-Client-ID": "other"},
        )
        assert isolated.status_code == 200
        assert isolated.json()["sources"] == []

        batch = await client.post(
            "/api/v1/chat/batch",
            json={
                "requests": [
                    {"message": "Hello", "session_id": "one", "use_rag": False},
                    {
                        "message": "Calculate 1 + 1",
                        "session_id": "two",
                        "use_rag": False,
                    },
                ]
            },
            headers=headers,
        )
        assert batch.status_code == 200
        assert batch.json()["count"] == 2
