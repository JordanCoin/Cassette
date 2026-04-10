"""Request recording — creates and persists trace/event records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from libs.core.contracts import Event, Trace
from libs.core.ports import EventStore

MAX_CONTENT_LEN = 2000


def _truncate(text: str) -> str:
    if len(text) <= MAX_CONTENT_LEN:
        return text
    return text[:MAX_CONTENT_LEN] + "...[truncated]"


def _summarize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep role and truncated content for each message."""
    return [
        {"role": m.get("role", "unknown"), "content": _truncate(str(m.get("content", "")))}
        for m in messages
    ]


def record_request(
    store: EventStore,
    *,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
) -> Trace:
    """Record a gateway request as a trace and event."""
    trace_id = uuid4()
    now = datetime.now(tz=UTC)

    trace = Trace(
        trace_id=trace_id,
        timestamp=now,
        messages=[{"role": "system", "content": f"{method} {path}"}],
        model="gateway",
        response=str(status_code),
        latency_ms=latency_ms,
    )
    store.save_trace(trace)

    event = Event(
        event_id=uuid4(),
        trace_id=trace_id,
        timestamp=now,
        type="request.completed",
        payload={
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": latency_ms,
        },
        provenance="gateway",
    )
    store.save_event(event)

    return trace


def record_chat(
    store: EventStore,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response: str,
    latency_ms: float,
    provider: str = "",
    backend_url: str = "",
) -> Trace:
    """Record a successful chat completion as a trace and event."""
    trace_id = uuid4()
    now = datetime.now(tz=UTC)

    trace = Trace(
        trace_id=trace_id,
        timestamp=now,
        messages=_summarize_messages(messages),
        model=model,
        response=_truncate(response),
        latency_ms=latency_ms,
    )
    store.save_trace(trace)

    event = Event(
        event_id=uuid4(),
        trace_id=trace_id,
        timestamp=now,
        type="chat.completion",
        payload={
            "model": model,
            "provider": provider,
            "backend_url": backend_url,
            "status": "ok",
            "message_count": len(messages),
            "latency_ms": latency_ms,
        },
        provenance="gateway",
    )
    store.save_event(event)

    return trace


def record_chat_error(
    store: EventStore,
    *,
    model: str,
    messages: list[dict[str, Any]],
    latency_ms: float,
    provider: str = "",
    backend_url: str = "",
    error_type: str = "",
    error_message: str = "",
) -> Trace:
    """Record a failed chat completion as a trace and event."""
    trace_id = uuid4()
    now = datetime.now(tz=UTC)

    trace = Trace(
        trace_id=trace_id,
        timestamp=now,
        messages=_summarize_messages(messages),
        model=model,
        response=f"[error] {error_type}: {_truncate(error_message)}",
        latency_ms=latency_ms,
    )
    store.save_trace(trace)

    event = Event(
        event_id=uuid4(),
        trace_id=trace_id,
        timestamp=now,
        type="chat.completion.error",
        payload={
            "model": model,
            "provider": provider,
            "backend_url": backend_url,
            "status": "error",
            "error_type": error_type,
            "error_message": _truncate(error_message),
            "message_count": len(messages),
            "latency_ms": latency_ms,
        },
        provenance="gateway",
    )
    store.save_event(event)

    return trace
