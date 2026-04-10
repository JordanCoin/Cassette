"""Prompt loader — reads prompt templates from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_prompt(name: str, prompts_dir: Path | None = None) -> dict[str, Any]:
    """Load a prompt template by name.

    Looks for {name}.yaml in the prompts directory.
    Returns the parsed YAML as a dict with at least 'template'.
    """
    directory = prompts_dir or PROMPTS_DIR
    path = directory / f"{name}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template not found: {path}\n"
            f"Available: {', '.join(p.stem for p in directory.glob('*.yaml'))}"
        )

    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if "template" not in data:
        raise ValueError(f"Prompt {name} is missing 'template' field")

    return data


def render_prompt(name: str, **kwargs: str) -> str:
    """Load a prompt template and render it with the given variables."""
    data = load_prompt(name)
    result: str = data["template"].format(**kwargs)
    return result
