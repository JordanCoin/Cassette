"""Stage protocol and example implementations."""

from __future__ import annotations

from typing import Any, Protocol


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
        return {"echoed": context}


class FailStage:
    """Stage that always fails. Useful for testing error paths."""

    @property
    def name(self) -> str:
        return "fail"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("Intentional stage failure")
