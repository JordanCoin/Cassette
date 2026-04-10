"""Minimal in-process metrics registry with Prometheus text format export."""

from __future__ import annotations

import threading


class _Registry:
    def __init__(self) -> None:
        self._counters: dict[str, float] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def get(self, name: str, labels: dict[str, str] | None = None) -> float:
        key = self._key(name, labels)
        with self._lock:
            return self._counters.get(key, 0.0)

    def export(self) -> str:
        """Export all metrics in Prometheus text exposition format."""
        with self._lock:
            lines: list[str] = []
            for key, value in sorted(self._counters.items()):
                # Format value: integer if whole number, else float
                if value == int(value):
                    lines.append(f"{key} {int(value)}")
                else:
                    lines.append(f"{key} {value}")
            return "\n".join(lines) + "\n" if lines else ""

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Global registry
registry = _Registry()
