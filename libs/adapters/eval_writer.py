"""JSONL evaluation results writer."""

from __future__ import annotations

from pathlib import Path

from libs.core.contracts import EvalResult


def write_eval_results(results: list[EvalResult], path: Path) -> int:
    """Write evaluation results to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")
    return len(results)
