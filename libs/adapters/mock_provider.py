"""Mock model provider — deterministic, no external calls."""

from __future__ import annotations


class MockProvider:
    """Returns a fixed response for any input."""

    @property
    def name(self) -> str:
        return "mock"

    def complete(self, messages: list[dict[str, str]]) -> str:
        return "This is a mock response."
