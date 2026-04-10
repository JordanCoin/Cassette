"""Tests for the config system."""

from __future__ import annotations

import os

import pytest

from libs.core.config import reload
from libs.core.provider_registry import resolve_provider


class TestConfig:
    def test_reads_yaml_defaults(self) -> None:
        from libs.core.config import get_str

        reload()
        assert get_str("provider") == "mock"

    def test_env_var_overrides_yaml(self) -> None:
        from libs.core.config import get_str

        env = os.environ.copy()
        os.environ["CASSETTE_PROVIDER"] = "custom"
        try:
            reload()
            assert get_str("provider") == "custom"
        finally:
            os.environ.clear()
            os.environ.update(env)
            reload()

    def test_missing_key_raises(self) -> None:
        from libs.core.config import get

        with pytest.raises(KeyError, match="Missing config key"):
            get("nonexistent_key_xyz")


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
