"""Unified configuration — cassette.yaml is the single source of truth.

Env vars with CASSETTE_ prefix override yaml values.
No defaults in Python code. If a key is missing from yaml, it errors.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

_CONFIG: dict[str, Any] | None = None
_CONFIG_PATH = Path("cassette.yaml")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Cassette requires a cassette.yaml file. "
            "Copy cassette.yaml from the repo root."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: {path}")
    return data


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Env vars override yaml. CASSETTE_KEY overrides key."""
    for key in list(config.keys()):
        env_key = f"CASSETTE_{key.upper()}"
        val = os.environ.get(env_key)
        if val is None:
            # Legacy env var names
            legacy = {"provider_url": "CASSETTE_LLAMA_CPP_URL"}
            val = os.environ.get(legacy.get(key, ""))
        if val is None:
            continue
        # Cast to match yaml type
        original = config[key]
        if isinstance(original, bool):
            config[key] = val.lower() in ("1", "true", "yes")
        elif isinstance(original, int):
            config[key] = int(val)
        elif isinstance(original, float):
            config[key] = float(val)
        else:
            config[key] = val
    return config


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load config from yaml with env overrides."""
    global _CONFIG
    if _CONFIG is not None and path is None:
        return _CONFIG
    config = _load_yaml(path or _CONFIG_PATH)
    config = _apply_env_overrides(config)
    if path is None:
        _CONFIG = config
    return config


def get(key: str) -> Any:
    """Get a config value. Raises KeyError if missing."""
    config = load_config()
    if key not in config:
        raise KeyError(
            f"Missing config key: '{key}'\n"
            f"Add it to cassette.yaml"
        )
    return config[key]


def get_str(key: str) -> str:
    return str(get(key))


def get_int(key: str) -> int:
    return int(get(key))


def get_float(key: str) -> float:
    return float(get(key))


def reload() -> dict[str, Any]:
    """Force reload from disk."""
    global _CONFIG
    _CONFIG = None
    return load_config()
