"""Stage protocol and example implementations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from libs.core.contracts import DatasetSnapshot, TrainingProposal
from libs.core.ports import WebFetch, WebSearch

# Type alias for the emitter callback injected by the runner.
EmitFn = Callable[[str, dict[str, Any] | None], None]


def _noop_emit(event_type: str, payload: dict[str, Any] | None = None) -> None:
    """Default no-op emitter when no runner context is available."""


class Stage(Protocol):
    """A single unit of orchestrated work."""

    @property
    def name(self) -> str: ...

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the stage. Returns a result dict. Raises on failure."""
        ...


class EchoStage:
    """Trivial stage that echoes its input context as the result."""

    @property
    def name(self) -> str:
        return "echo"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        clean = {k: v for k, v in context.items() if not k.startswith("_")}
        return {"echoed": clean}


class FailStage:
    """Stage that always fails. Useful for testing error paths."""

    @property
    def name(self) -> str:
        return "fail"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Intentional stage failure")


class GatherSourcesStage:
    """Searches the web and fetches top results."""

    def __init__(
        self, search: WebSearch, fetch: WebFetch, max_results: int = 3
    ) -> None:
        self._search = search
        self._fetch = fetch
        self._max_results = max_results

    @property
    def name(self) -> str:
        return "gather_sources"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        query = context.get("query", "")
        if not query:
            raise ValueError("gather_sources requires a 'query' in context")

        emit: EmitFn = context.get("_emit", _noop_emit)

        # Search
        emit("search.started", {"query": query})
        try:
            results = self._search.search(query)
        except (ConnectionError, RuntimeError) as exc:
            emit("search.failed", {"query": query, "error": str(exc)})
            raise
        top = results[: self._max_results]
        emit("search.completed", {"query": query, "result_count": len(top)})

        # Fetch
        sources: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        for result in top:
            url = result.get("url", "")
            emit("fetch.started", {"url": url})
            try:
                fetched = self._fetch.fetch(url)
                sources.append({
                    "title": result.get("title", ""),
                    "url": url,
                    "snippet": result.get("snippet", ""),
                    "content": fetched.get("content", ""),
                })
                emit("fetch.completed", {"url": url})
            except (ConnectionError, RuntimeError) as exc:
                errors.append({"url": url, "error": str(exc)})
                emit("fetch.failed", {"url": url, "error": str(exc)})

        return {
            "query": query,
            "source_count": len(sources),
            "sources": sources,
            "errors": errors,
        }


# Rough token estimate: ~200 tokens per record (messages + response)
_TOKENS_PER_RECORD = 200
_DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
_MIN_RECORDS_FOR_TRAINING = 10


class ProposeTrainingStage:
    """Generates a training proposal from a dataset snapshot."""

    @property
    def name(self) -> str:
        return "propose_training"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        emit: EmitFn = context.get("_emit", _noop_emit)

        snapshot_data = context.get("snapshot")
        if not snapshot_data:
            raise ValueError("propose_training requires 'snapshot' in context")

        snapshot = DatasetSnapshot.model_validate(snapshot_data)

        emit("proposal.started", {"snapshot_id": snapshot.snapshot_id})

        if snapshot.record_count < _MIN_RECORDS_FOR_TRAINING:
            method = "sft"
            notes = (
                f"Only {snapshot.record_count} records — below minimum "
                f"({_MIN_RECORDS_FOR_TRAINING}). Training not recommended yet."
            )
        elif snapshot.record_count < 100:
            method = "lora"
            notes = "Small dataset. LoRA recommended for parameter efficiency."
        else:
            method = "qlora"
            notes = "Sufficient data for QLoRA fine-tuning."

        proposal = TrainingProposal(
            snapshot_id=snapshot.snapshot_id,
            record_count=snapshot.record_count,
            base_model=_DEFAULT_BASE_MODEL,
            method=method,
            estimated_tokens=snapshot.record_count * _TOKENS_PER_RECORD,
            notes=notes,
        )

        emit("proposal.completed", {
            "snapshot_id": snapshot.snapshot_id,
            "method": method,
            "estimated_tokens": proposal.estimated_tokens,
        })

        return proposal.model_dump(mode="json")
