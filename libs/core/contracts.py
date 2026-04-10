"""Canonical contract definitions for Cassette.

These Pydantic models are the single source of truth for all system contracts.
JSON Schema can be exported via `Model.model_json_schema()`.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Immutable event envelope. Every meaningful system action produces one."""

    event_id: UUID
    trace_id: UUID
    timestamp: datetime
    type: str
    payload: dict[str, Any]
    provenance: str = Field(default="", description="Origin system or component")


class TaskStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class Task(BaseModel):
    """Task ledger record. Tracks detached work with status transitions."""

    task_id: UUID
    trace_id: UUID
    status: TaskStatus
    parent_task_id: UUID | None = None
    created_at: datetime
    updated_at: datetime
    terminal_reason: str | None = None


class Trace(BaseModel):
    """Prompt/response trace. Captures a single inference interaction."""

    trace_id: UUID
    timestamp: datetime
    messages: list[dict[str, Any]]
    model: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    response: str = ""
    token_count: int | None = None
    latency_ms: float | None = None


class DatasetRecord(BaseModel):
    """One curated training candidate extracted from system traces."""

    record_id: UUID
    trace_id: UUID
    source: str = Field(description="Origin: chat, gather_sources, etc.")
    messages: list[dict[str, Any]]
    response: str
    model: str
    quality: str = Field(default="unreviewed", description="unreviewed, accepted, rejected")
    content_hash: str = Field(description="For deduplication")


class EvalResult(BaseModel):
    """Result of evaluating one DatasetRecord."""

    record_id: UUID
    content_hash: str
    decision: str = Field(description="accepted, rejected, needs_review")
    flags: list[str] = Field(default_factory=list, description="Reasons for the decision")
    score: float = Field(default=1.0, description="0.0–1.0 quality score")


class DatasetSnapshot(BaseModel):
    """Metadata for a versioned dataset snapshot."""

    snapshot_id: str
    created_at: datetime
    content_hash: str
    record_count: int
    source_path: str


class TrainingProposal(BaseModel):
    """Structured plan for a potential training run."""

    snapshot_id: str
    record_count: int
    base_model: str
    method: str = Field(description="sft, lora, qlora, dpo")
    estimated_tokens: int
    notes: str = ""


class TrainingPlan(BaseModel):
    """Concrete, validated training plan ready for execution."""

    snapshot_id: str
    dataset_path: str
    base_model: str
    method: str
    estimated_tokens: int
    output_dir: str
    command: str = Field(description="Runnable training command")
    notes: str = ""


class TrainingReadiness(BaseModel):
    """Result of pre-training validation."""

    snapshot_id: str
    dataset_path: str
    base_model: str
    method: str
    ready: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    notes: str = ""


class TrainingResult(BaseModel):
    """Result of a training execution."""

    snapshot_id: str
    plan_method: str
    exit_code: int
    success: bool
    output_dir: str
    duration_sec: float
    stdout_tail: str = Field(default="", description="Last lines of training output")
    error: str = ""


class JudgeResult(BaseModel):
    """Result of LLM-as-judge evaluation for one record."""

    record_id: UUID
    content_hash: str
    judge_score: int = Field(ge=1, le=5, description="1-5 quality score")
    reasoning: str = ""
    provider: str = ""
