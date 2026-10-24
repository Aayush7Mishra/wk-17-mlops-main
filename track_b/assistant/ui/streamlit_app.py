"""Streamlit chat client for the Week 15 applied systems backend.

Run from the project root with::

    streamlit run ui/streamlit_app.py

Set ``BACKEND_URL`` to point the UI at a backend other than the local default.
"""

from __future__ import annotations

import html
import json
import os
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx
import streamlit as st

DEFAULT_BACKEND_URL = "http://localhost:8000"
CHAT_TIMEOUT = httpx.Timeout(connect=5.0, read=90.0, write=30.0, pool=5.0)
UPLOAD_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=120.0, pool=5.0)
HEALTH_TIMEOUT = httpx.Timeout(5.0)


class BackendError(RuntimeError):
    """A user-safe error raised when the backend cannot satisfy a request."""


def backend_url() -> str:
    """Return a normalized backend base URL from the environment."""

    value = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).strip()
    return (value or DEFAULT_BACKEND_URL).rstrip("/")


def backend_headers(*, session_id: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = os.getenv("BACKEND_API_KEY", "").strip()
    if api_key:
        headers["X-API-Key"] = api_key
    if session_id:
        headers["X-Client-ID"] = session_id
    return headers


def _error_detail(response: httpx.Response) -> str:
    """Extract a useful, bounded error message from an HTTP response."""

    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        payload = None

    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message") or payload.get("error")
        if isinstance(detail, list):
            detail = "; ".join(
                str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                for item in detail
            )
        if detail:
            return str(detail)[:500]

    text = response.text.strip()
    return text[:500] if text else response.reason_phrase


def _decode_json(response: httpx.Response, operation: str) -> dict[str, Any]:
    if response.is_error:
        raise BackendError(
            f"{operation} failed ({response.status_code}): {_error_detail(response)}"
        )
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError) as exc:
        raise BackendError(f"{operation} returned an invalid response.") from exc
    if not isinstance(payload, dict):
        raise BackendError(f"{operation} returned an unexpected response format.")
    return payload


def _friendly_network_error(exc: httpx.HTTPError) -> BackendError:
    if isinstance(exc, httpx.TimeoutException):
        return BackendError("The backend took too long to respond. Please retry in a moment.")
    if isinstance(exc, httpx.ConnectError):
        return BackendError(
            "Could not connect to the backend. Confirm that the API server is running."
        )
    return BackendError(f"The backend request failed: {exc}")


@st.cache_data(ttl=8, show_spinner=False)
def fetch_health(base_url: str) -> dict[str, Any]:
    """Fetch a short-lived health snapshot without raising into the UI."""

    started = time.perf_counter()
    try:
        with httpx.Client(timeout=HEALTH_TIMEOUT) as client:
            response = client.get(f"{base_url}/health")
        payload = _decode_json(response, "Health check")
    except (httpx.HTTPError, BackendError) as exc:
        message = (
            str(_friendly_network_error(exc)) if isinstance(exc, httpx.HTTPError) else str(exc)
        )
        return {"online": False, "error": message}

    return {
        "online": True,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "data": payload,
    }


def send_message(
    base_url: str,
    *,
    message: str,
    session_id: str,
    use_rag: bool,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    """Send one chat turn and validate the minimum response contract."""

    try:
        with httpx.Client(timeout=CHAT_TIMEOUT) as client:
            response = client.post(
                f"{base_url}/api/v1/chat",
                json={
                    "message": message,
                    "session_id": session_id,
                    "use_rag": use_rag,
                    "history": history[-20:],
                },
                headers=backend_headers(session_id=session_id),
            )
        payload = _decode_json(response, "Chat request")
    except httpx.HTTPError as exc:
        raise _friendly_network_error(exc) from exc

    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise BackendError("The backend response did not contain an answer.")
    return payload


def ingest_document(
    base_url: str,
    *,
    filename: str,
    content: bytes,
    content_type: str | None,
    session_id: str,
) -> dict[str, Any]:
    """Upload a document to the backend ingestion endpoint."""

    try:
        with httpx.Client(timeout=UPLOAD_TIMEOUT) as client:
            response = client.post(
                f"{base_url}/api/v1/documents",
                files={
                    "file": (
                        filename,
                        content,
                        content_type or "application/octet-stream",
                    )
                },
                data={"session_id": session_id},
                headers=backend_headers(session_id=session_id),
            )
        return _decode_json(response, "Document ingestion")
    except httpx.HTTPError as exc:
        raise _friendly_network_error(exc) from exc


def initialize_state() -> None:
    defaults: dict[str, Any] = {
        "chat_history": [],
        "session_id": str(uuid.uuid4()),
        "use_rag": True,
        "ingested_documents": [],
        "upload_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_new_conversation() -> None:
    st.session_state.chat_history = []


def _safe_link(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.netloc else None


def _source_title(source: Any, index: int) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, dict):
        for key in ("title", "name", "filename", "source", "document"):
            value = source.get(key)
            if value:
                return str(value)
    return f"Source {index}"


def render_sources(sources: Any) -> None:
    if not isinstance(sources, list) or not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for index, source in enumerate(sources, start=1):
            title = _source_title(source, index)
            if isinstance(source, str):
                st.markdown(f"**{index}.** {html.escape(source)}")
                continue

            if not isinstance(source, dict):
                st.write(source)
                continue

            url = _safe_link(source.get("url") or source.get("uri"))
            if url:
                st.markdown(
                    f"**{index}. [{html.escape(title)}]({url})**",
                    unsafe_allow_html=False,
                )
            else:
                st.markdown(f"**{index}. {html.escape(title)}**")

            excerpt = (
                source.get("excerpt")
                or source.get("snippet")
                or source.get("content")
                or source.get("text")
            )
            if excerpt:
                excerpt_text = str(excerpt)
                if len(excerpt_text) > 700:
                    excerpt_text = f"{excerpt_text[:700].rstrip()}…"
                st.caption(excerpt_text)

            metadata = {
                key: value
                for key, value in source.items()
                if key
                not in {
                    "title",
                    "name",
                    "filename",
                    "source",
                    "document",
                    "url",
                    "uri",
                    "excerpt",
                    "snippet",
                    "content",
                    "text",
                }
            }
            if metadata:
                st.json(metadata, expanded=False)


def render_tool_calls(tool_calls: Any) -> None:
    if not isinstance(tool_calls, list) or not tool_calls:
        return

    with st.expander(f"Tool activity ({len(tool_calls)})", expanded=False):
        for index, call in enumerate(tool_calls, start=1):
            if isinstance(call, dict):
                name = call.get("name") or call.get("tool") or f"Tool call {index}"
                st.markdown(f"**{index}. {html.escape(str(name))}**")
                st.json(call, expanded=False)
            else:
                st.write(call)


def render_response_metadata(payload: dict[str, Any]) -> None:
    details: list[str] = []
    if payload.get("provider"):
        details.append(f"Provider: {payload['provider']}")
    if isinstance(payload.get("cached"), bool):
        details.append("Cache hit" if payload["cached"] else "Fresh response")
    if payload.get("degraded") is True:
        details.append("Degraded mode")
    if payload.get("request_id"):
        details.append(f"Request: {payload['request_id']}")
    if details:
        st.caption(" · ".join(str(item) for item in details))


def render_assistant_message(message: dict[str, Any]) -> None:
    st.markdown(str(message.get("content", "")))
    payload = message.get("metadata")
    if not isinstance(payload, dict):
        return

    if payload.get("degraded") is True:
        st.warning(
            "This answer was produced in degraded mode and may have limited context.",
            icon="⚠️",
        )
    render_sources(payload.get("sources"))
    render_tool_calls(payload.get("tool_calls"))
    render_response_metadata(payload)


def render_chat_history() -> None:
    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        with st.chat_message(role):
            if role == "assistant":
                render_assistant_message(message)
            else:
                st.markdown(str(message.get("content", "")))


def _format_stat_label(key: str) -> str:
    return key.replace("_", " ").strip().title()


def render_ingestion_result(result: dict[str, Any]) -> None:
    st.success("Document ingested and ready for retrieval.", icon="✅")
    scalar_stats = [
        (key, value)
        for key, value in result.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ][:4]
    if scalar_stats:
        columns = st.columns(len(scalar_stats))
        for column, (key, value) in zip(columns, scalar_stats, strict=True):
            column.metric(_format_stat_label(key), value)
    else:
        with st.expander("Ingestion details", expanded=False):
            st.json(result)


def render_sidebar(base_url: str) -> None:
    with st.sidebar:
        st.markdown("## Applied Systems")
        st.caption("Grounded chat with observable tool use")

        health = fetch_health(base_url)
        if health.get("online"):
            data = health.get("data", {})
            status = data.get("status", "healthy") if isinstance(data, dict) else "healthy"
            if str(status).lower() in {"ok", "healthy", "ready", "up"}:
                st.success(f"Backend {status}", icon="✅")
            else:
                st.warning(f"Backend {status}", icon="⚠️")
            st.caption(f"{base_url} · {health.get('latency_ms', '—')} ms")
            if isinstance(data, dict):
                provider = data.get("provider") or data.get("llm_provider")
                if provider:
                    st.caption(f"Model provider: {provider}")
        else:
            st.error("Backend unavailable", icon="🔴")
            st.caption(health.get("error", "Health check failed."))
            st.caption(base_url)

        if st.button("Refresh status", use_container_width=True):
            fetch_health.clear()
            st.rerun()

        st.divider()
        st.toggle(
            "Use document retrieval",
            key="use_rag",
            help="Ground answers in documents indexed by the backend.",
        )

        st.markdown("### Add knowledge")
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "md"],
            key=f"document_upload_{st.session_state.upload_nonce}",
            help="Accepted formats: PDF, TXT, and Markdown.",
        )
        if st.button(
            "Ingest document",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        ):
            assert uploaded_file is not None
            try:
                with st.spinner("Reading and indexing document…"):
                    result = ingest_document(
                        base_url,
                        filename=uploaded_file.name,
                        content=uploaded_file.getvalue(),
                        content_type=uploaded_file.type,
                        session_id=st.session_state.session_id,
                    )
                st.session_state.ingested_documents.append(
                    {"name": uploaded_file.name, "result": result}
                )
                render_ingestion_result(result)
                st.session_state.upload_nonce += 1
            except BackendError as exc:
                st.error(str(exc), icon="⚠️")

        if st.session_state.ingested_documents:
            with st.expander(
                f"Uploaded from this browser ({len(st.session_state.ingested_documents)})"
            ):
                for item in reversed(st.session_state.ingested_documents):
                    st.markdown(f"- {html.escape(str(item['name']))}")

        st.divider()
        st.caption(f"Session `{st.session_state.session_id[:8]}`")
        if st.button("New conversation", use_container_width=True):
            start_new_conversation()
            st.rerun()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 880px;
            padding-top: 2.2rem;
            padding-bottom: 6rem;
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.6rem;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
            border-radius: 16px;
            padding: 0.35rem 0.55rem;
            margin-bottom: 0.7rem;
        }
        [data-testid="stChatInput"] {
            border-radius: 14px;
        }
        .app-kicker {
            color: #6c63ff;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }
        .app-subtitle {
            color: color-mix(in srgb, var(--text-color) 67%, transparent);
            margin-top: -0.4rem;
            margin-bottom: 1.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Applied Systems Assistant",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    initialize_state()
    inject_styles()

    base_url = backend_url()
    render_sidebar(base_url)

    st.markdown('<div class="app-kicker">Week 15 · Production AI</div>', unsafe_allow_html=True)
    st.title("Ask. Retrieve. Act.")
    st.markdown(
        '<div class="app-subtitle">A transparent assistant that can ground answers '
        "in your documents and show its work.</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.chat_history:
        st.info(
            "Ask a question to begin, or upload a document in the sidebar for a grounded answer.",
            icon="💡",
        )

    render_chat_history()

    prompt = st.chat_input("Message the assistant…", max_chars=8_000)
    if not prompt:
        return

    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        st.warning("Please enter a message before sending.")
        return

    st.session_state.chat_history.append({"role": "user", "content": cleaned_prompt})
    with st.chat_message("user"):
        st.markdown(cleaned_prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                payload = send_message(
                    base_url,
                    message=cleaned_prompt,
                    session_id=st.session_state.session_id,
                    use_rag=bool(st.session_state.use_rag),
                    history=[
                        {"role": item["role"], "content": str(item["content"])}
                        for item in st.session_state.chat_history[:-1]
                    ],
                )
            assistant_message = {
                "role": "assistant",
                "content": payload["answer"].strip(),
                "metadata": payload,
            }
            st.session_state.chat_history.append(assistant_message)
            render_assistant_message(assistant_message)
        except BackendError as exc:
            st.error(str(exc), icon="⚠️")
            st.caption(
                "Your message is still in the conversation. Retry when the backend is available."
            )


if __name__ == "__main__":
    main()
