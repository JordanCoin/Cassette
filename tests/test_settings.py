"""Tests for settings and provider resolution."""

from __future__ import annotations

import os

import pytest

from libs.core.provider_registry import resolve_provider
from libs.core.settings import get_provider_name


class TestGetProviderName:
    def test_defaults_to_mock(self) -> None:
        env = os.environ.copy()
        os.environ.pop("CASSETTE_PROVIDER", None)
        try:
            assert get_provider_name() == "mock"
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_reads_env_var(self) -> None:
        env = os.environ.copy()
        os.environ["CASSETTE_PROVIDER"] = "custom"
        try:
            assert get_provider_name() == "custom"
        finally:
            os.environ.clear()
            os.environ.update(env)


class TestResolveProvider:
    def test_resolves_mock(self) -> None:
        provider = resolve_provider("mock")
        assert provider.name == "mock"

    def test_mock_provider_works(self) -> None:
        provider = resolve_provider("mock")
        result = provider.complete([{"role": "user", "content": "hi"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider: 'nonexistent'"):
            resolve_provider("nonexistent")

    def test_error_lists_available(self) -> None:
        with pytest.raises(ValueError, match="Available:.*mock"):
            resolve_provider("bad")
