"""Prompt loader — reads prompt templates and output schemas from YAML files.

Each prompt YAML defines:
- system: the role/rules (sent as a system message)
- user: the task-specific input (sent as a user message, with {var} placeholders)
- output_schema: optional JSON schema for structured output
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from libs.core.config import get_str

PROMPTS_DIR = Path(get_str("prompts_dir"))


def load_prompt(name: str, prompts_dir: Path | None = None) -> dict[str, Any]:
    """Load a prompt definition by name."""
    directory = prompts_dir or PROMPTS_DIR
    path = directory / f"{name}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt not found: {path}\n"
            f"Available: {', '.join(p.stem for p in directory.glob('*.yaml'))}"
        )

    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if "system" not in data or "user" not in data:
        raise ValueError(
            f"Prompt {name!r} must define both 'system' and 'user' fields"
        )

    return data


def render_messages(name: str, **kwargs: str) -> list[dict[str, str]]:
    """Render a prompt into OpenAI-style chat messages.

    Returns [{"role": "system", "content": ...}, {"role": "user", "content": ...}].
    Variables in kwargs fill {placeholders} in the user template.
    """
    data = load_prompt(name)
    return [
        {"role": "system", "content": data["system"].strip()},
        {"role": "user", "content": data["user"].format(**kwargs).strip()},
    ]


def get_output_schema(name: str) -> dict[str, Any] | None:
    """Get the output JSON schema for a prompt, if defined."""
    data = load_prompt(name)
    schema: dict[str, Any] | None = data.get("output_schema")
    return schema
