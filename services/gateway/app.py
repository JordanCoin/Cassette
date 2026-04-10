"""Cassette Gateway — minimal FastAPI application."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from libs.adapters.jsonl_store import JsonlStore
from libs.core.contracts import Event, Task, TaskStatus
from libs.core.ports import ModelProvider
from libs.core.provider_registry import resolve_provider
from libs.core.recorder import record_chat, record_chat_error, record_request
from libs.core.settings import get_provider_name
from services.gateway.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CreateTaskRequest,
    UpdateTaskRequest,
)

DATA_DIR = Path("data/gateway")

app = FastAPI(title="Cassette Gateway", version="0.1.0")
store = JsonlStore(DATA_DIR)
provider: ModelProvider = resolve_provider(get_provider_name())


def _backend_url() -> str:
    """Return the backend URL if the provider exposes one, otherwise empty."""
    return getattr(provider, "_base_url", "")


class RecordingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - start) * 1000
        if request.url.path != "/v1/chat/completions":
            record_request(
                store,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )
        return response


app.add_middleware(RecordingMiddleware)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    start = time.monotonic()
    messages_raw = [m.model_dump() for m in request.messages]
    backend = _backend_url()

    try:
        response_text = provider.complete(messages_raw)
    except (ConnectionError, RuntimeError) as exc:
        latency_ms = (time.monotonic() - start) * 1000
        record_chat_error(
            store,
            model=request.model,
            messages=messages_raw,
            latency_ms=latency_ms,
            provider=provider.name,
            backend_url=backend,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return JSONResponse(  # type: ignore[return-value]
            status_code=502,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            },
        )

    latency_ms = (time.monotonic() - start) * 1000

    record_chat(
        store,
        model=request.model,
        messages=messages_raw,
        response=response_text,
        latency_ms=latency_ms,
        provider=provider.name,
        backend_url=backend,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex[:12]}",
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
    )


# -- Debug / read-side endpoints --


@app.get("/debug/traces")
async def debug_traces(
    trace_id: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
) -> list[dict[str, Any]]:
    if trace_id:
        records = store.get_traces_by_id(trace_id)
    else:
        records = store.get_latest_traces(limit)
    return [r.model_dump(mode="json") for r in records]


@app.get("/debug/events")
async def debug_events(
    event_type: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
) -> list[dict[str, Any]]:
    if event_type:
        records = store.get_events_by_type(event_type)
    else:
        records = store.get_latest_events(limit)
    return [r.model_dump(mode="json") for r in records]


# -- Task ledger endpoints --


def _emit_task_event(task: Task, event_type: str) -> None:
    store.save_event(
        Event(
            event_id=uuid4(),
            trace_id=task.trace_id,
            timestamp=task.updated_at,
            type=event_type,
            payload={
                "task_id": str(task.task_id),
                "status": task.status.value,
            },
            provenance="gateway",
        )
    )


@app.post("/tasks", status_code=201)
async def create_task(request: CreateTaskRequest) -> dict[str, Any]:
    from datetime import UTC, datetime

    now = datetime.now(tz=UTC)
    task = Task(
        task_id=uuid4(),
        trace_id=request.trace_id or uuid4(),
        status=TaskStatus.queued,
        parent_task_id=request.parent_task_id,
        created_at=now,
        updated_at=now,
    )
    store.save_task(task)
    _emit_task_event(task, "task.created")
    return task.model_dump(mode="json")


@app.get("/tasks")
async def list_tasks(
    limit: int = Query(default=20, ge=1, le=200),
) -> list[dict[str, Any]]:
    tasks = store.get_latest_tasks(limit)
    return [t.model_dump(mode="json") for t in tasks]


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> dict[str, Any]:
    task = store.get_task(task_id)
    if task is None:
        return JSONResponse(status_code=404, content={"error": "Task not found"})  # type: ignore[return-value]
    return task.model_dump(mode="json")


@app.patch("/tasks/{task_id}")
async def update_task(task_id: str, request: UpdateTaskRequest) -> dict[str, Any]:
    try:
        new_status = TaskStatus(request.status)
    except ValueError:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"error": f"Invalid status: {request.status!r}"},
        )

    task = store.update_task_status(task_id, new_status)
    if task is None:
        return JSONResponse(status_code=404, content={"error": "Task not found"})  # type: ignore[return-value]

    _emit_task_event(task, "task.updated")
    return task.model_dump(mode="json")


# -- Orchestrator trigger --


@app.post("/orchestrator/run")
async def orchestrator_run(
    request: dict[str, Any],
) -> dict[str, Any]:
    from services.orchestrator.runner import run_stage
    from services.orchestrator.stages import EchoStage

    stage_name = request.get("stage", "echo")
    context = request.get("context", {})

    if stage_name != "echo":
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"error": f"Unknown stage: {stage_name!r}. Available: echo"},
        )

    result = run_stage(
        stage=EchoStage(),
        context=context,
        task_store=store,
        event_store=store,
    )
    return result
