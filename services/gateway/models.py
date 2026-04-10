"""Minimal request/response models for gateway endpoints."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ResponseFormat(BaseModel):
    type: str = "text"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    response_format: ResponseFormat | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatChoice]


# -- Task models --


class CreateTaskRequest(BaseModel):
    trace_id: UUID | None = None
    parent_task_id: UUID | None = None


class UpdateTaskRequest(BaseModel):
    status: str = Field(description="New status: running, completed, failed, cancelled")
    terminal_reason: str | None = None
