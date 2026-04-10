"""Tests for milestone 21: hardening, health checks, and loop reliability."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

import services.gateway.app as gateway_module
from libs.adapters.jsonl_store import JsonlStore
from libs.adapters.llama_cpp_http_provider import (
    LlamaCppHttpProvider,
    ProviderResponseError,
    ProviderTimeoutError,
)
from libs.adapters.mock_provider import MockProvider
from libs.cli import main as cli_main
from libs.core.metrics import registry
from services.gateway.app import app

_FAKE_REQ = httpx.Request("POST", "http://test/v1/chat/completions")


def _make_client(tmp_path: Path) -> TestClient:
    registry.reset()
    gateway_module.store = JsonlStore(tmp_path)
    gateway_module.provider = MockProvider()
    return TestClient(app)


# -- Provider timeout and error classification --


class TestProviderTimeout:
    @patch(
        "libs.adapters.llama_cpp_http_provider.httpx.post",
        side_effect=httpx.ReadTimeout("timed out"),
    )
    def test_timeout_raises_provider_timeout_error(self, mock_post: Any) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080", timeout=5.0)
        with pytest.raises(ProviderTimeoutError, match="Timeout after 5.0s"):
            provider.complete([{"role": "user", "content": "hi"}])

    @patch(
        "libs.adapters.llama_cpp_http_provider.httpx.post",
        side_effect=httpx.ConnectTimeout("connect timed out"),
    )
    def test_connect_timeout_raises(self, mock_post: Any) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080")
        with pytest.raises(ProviderTimeoutError):
            provider.complete([{"role": "user", "content": "hi"}])

    @patch("libs.adapters.llama_cpp_http_provider.httpx.post")
    def test_invalid_json_raises_response_error(self, mock_post: Any) -> None:
        resp = httpx.Response(200, text="not json", request=_FAKE_REQ)
        mock_post.return_value = resp
        provider = LlamaCppHttpProvider("http://localhost:8080")
        with pytest.raises(ProviderResponseError, match="Invalid JSON"):
            provider.complete([{"role": "user", "content": "hi"}])

    @patch("libs.adapters.llama_cpp_http_provider.httpx.post")
    def test_empty_choices_raises_response_error(self, mock_post: Any) -> None:
        resp = httpx.Response(200, json={"choices": []}, request=_FAKE_REQ)
        mock_post.return_value = resp
        provider = LlamaCppHttpProvider("http://localhost:8080")
        with pytest.raises(ProviderResponseError, match="No choices"):
            provider.complete([{"role": "user", "content": "hi"}])

    def test_custom_timeout_value(self) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080", timeout=30.0)
        assert provider._timeout == 30.0


# -- Provider health check --


class TestProviderHealthCheck:
    @patch("libs.adapters.llama_cpp_http_provider.httpx.get")
    def test_reachable(self, mock_get: Any) -> None:
        mock_get.return_value = httpx.Response(
            200, json={"data": []},
            request=httpx.Request("GET", "http://test/v1/models"),
        )
        provider = LlamaCppHttpProvider("http://localhost:8080")
        result = provider.health_check()
        assert result["reachable"] is True

    @patch(
        "libs.adapters.llama_cpp_http_provider.httpx.get",
        side_effect=httpx.ConnectError("refused"),
    )
    def test_unreachable(self, mock_get: Any) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080")
        result = provider.health_check()
        assert result["reachable"] is False
        assert result["error"] == "connection_refused"

    @patch(
        "libs.adapters.llama_cpp_http_provider.httpx.get",
        side_effect=httpx.ReadTimeout("timed out"),
    )
    def test_timeout(self, mock_get: Any) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080")
        result = provider.health_check()
        assert result["reachable"] is False
        assert result["error"] == "timeout"


class TestProviderHealthEndpoint:
    def test_mock_provider_health(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        resp = client.get("/healthz/provider")
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "mock"
        assert data["reachable"] is True


# -- Loop reliability --


class TestLoopReliability:
    def test_repeated_runs_no_corruption(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        r1 = client.post("/loop/run").json()
        r2 = client.post("/loop/run").json()
        assert r1["status"] == "completed"
        assert r2["status"] == "completed"

    def test_deterministic_output(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        r1 = client.post("/loop/run").json()
        r2 = client.post("/loop/run").json()
        # Same promoted count for same input
        assert r1.get("promoted") == r2.get("promoted")

    def test_multiple_runs_produce_valid_snapshots(self, tmp_path: Path) -> None:
        client = _make_client(tmp_path)
        client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        r1 = client.post("/loop/run").json()
        r2 = client.post("/loop/run").json()
        assert r1["status"] == "completed"
        assert r2["status"] == "completed"
        # Both runs should have snapshots
        assert r1.get("snapshot") is not None
        assert r2.get("snapshot") is not None


# -- Data sanity --


class TestDataSanity:
    def test_near_empty_response_filtered(self) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from libs.core.contracts import Trace
        from libs.core.extractor import extract_records

        trace = Trace(
            trace_id=uuid4(),
            timestamp=datetime.now(tz=UTC),
            messages=[{"role": "user", "content": "hi"}],
            model="test",
            response=" ",
        )
        assert extract_records([trace]) == []

    def test_no_content_messages_filtered(self) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from libs.core.contracts import Trace
        from libs.core.extractor import extract_records

        trace = Trace(
            trace_id=uuid4(),
            timestamp=datetime.now(tz=UTC),
            messages=[{"role": "user", "content": ""}],
            model="test",
            response="valid response",
        )
        assert extract_records([trace]) == []


# -- CLI health command --


class TestCliHealth:
    def test_health_returns_0(self, tmp_path: Path) -> None:
        code = cli_main(["--data-dir", str(tmp_path), "health"])
        assert code == 0

    def test_health_output_has_provider(self, tmp_path: Path) -> None:
        import sys
        from io import StringIO

        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "health"])
        finally:
            sys.stdout = old
        data = json.loads(buf.getvalue())
        assert "provider" in data
        assert data["status"] == "ok"


# Need json import at module level for the CLI test
import json  # noqa: E402
