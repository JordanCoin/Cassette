"""Minimal settings — reads configuration from environment variables."""

from __future__ import annotations

import os


def get_provider_name() -> str:
    return os.environ.get("CASSETTE_PROVIDER", "mock")


def get_llama_cpp_url() -> str:
    return os.environ.get("CASSETTE_LLAMA_CPP_URL", "http://localhost:8080")


def get_search_url() -> str:
    return os.environ.get("CASSETTE_SEARCH_URL", "http://localhost:8888")
