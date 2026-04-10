"""Integration tests against a real local model backend.

These tests are SKIPPED by default. To run them:

    CASSETTE_INTEGRATION=1 CASSETTE_PROVIDER=llama_cpp_http \
        uv run pytest tests/test_integration_real.py -v

Requires a running OpenAI-compatible server at
CASSETTE_LLAMA_CPP_URL (default: http://localhost:8080).
"""

from __future__ import annotations

import json
import os
import sys
from io import StringIO
from pathlib import Path

import pytest

from libs.adapters.llama_cpp_http_provider import LlamaCppHttpProvider
from libs.cli import main as cli_main
from libs.core.settings import get_llama_cpp_url

_SKIP = not os.environ.get("CASSETTE_INTEGRATION")
_REASON = "Set CASSETTE_INTEGRATION=1 and start a local model server to run"


@pytest.mark.skipif(_SKIP, reason=_REASON)
class TestRealProviderHealth:
    def test_health_check_reachable(self) -> None:
        provider = LlamaCppHttpProvider(get_llama_cpp_url())
        result = provider.health_check()
        assert result["reachable"] is True, f"Provider not reachable: {result}"

    def test_health_check_returns_url(self) -> None:
        provider = LlamaCppHttpProvider(get_llama_cpp_url())
        result = provider.health_check()
        assert "url" in result


@pytest.mark.skipif(_SKIP, reason=_REASON)
class TestRealChatCompletion:
    def test_single_completion(self) -> None:
        provider = LlamaCppHttpProvider(get_llama_cpp_url(), timeout=120.0)
        result = provider.complete([{"role": "user", "content": "Say hello in one word."}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_multi_turn(self) -> None:
        provider = LlamaCppHttpProvider(get_llama_cpp_url(), timeout=120.0)
        result = provider.complete([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer with just the number."},
        ])
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.skipif(_SKIP, reason=_REASON)
class TestRealDoctor:
    def test_doctor_reports_reachable(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = cli_main(["--data-dir", str(tmp_path), "doctor"])
        finally:
            sys.stdout = old
        output = buf.getvalue()
        # With a real backend, provider should be reachable
        assert "[OK]   provider_reachable" in output or "[SKIP]" in output
        assert code == 0


@pytest.mark.skipif(_SKIP, reason=_REASON)
class TestRealLoopRun:
    def test_full_loop_with_query(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = cli_main([
                "--data-dir", str(tmp_path),
                "run-loop", "--query", "What is 1+1?",
            ])
        finally:
            sys.stdout = old
        assert code == 0, f"Loop failed: {buf.getvalue()}"
        result = json.loads(buf.getvalue())
        assert result["status"] == "completed"
        assert result.get("extracted", 0) >= 1

    def test_persisted_files_exist(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            cli_main([
                "--data-dir", str(tmp_path),
                "run-loop", "--query", "Hello world",
            ])
        finally:
            sys.stdout = old
        assert (tmp_path / "traces.jsonl").exists()
        assert (tmp_path / "events.jsonl").exists()
        assert (tmp_path / "dataset_promoted.jsonl").exists()
        assert (tmp_path / "snapshots" / "manifest.jsonl").exists()
