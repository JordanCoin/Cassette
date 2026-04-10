"""Integration tests against a real model backend.

Run with: CASSETTE_INTEGRATION=1 uv run pytest tests/integration/ -v
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from libs.cli import main as cli_main
from libs.core.provider_registry import resolve_provider
from libs.core.settings import get_provider_name

from .conftest import SKIP_INTEGRATION, SKIP_REASON

skip = pytest.mark.skipif(SKIP_INTEGRATION, reason=SKIP_REASON)


@skip
class TestHardwareDetection:
    def test_detect(self, hardware: dict[str, object]) -> None:
        print(f"\n  Hardware: {json.dumps(hardware, indent=2, default=str)}")
        assert "surface" in hardware
        assert "platform" in hardware


@skip
class TestProviderReachable:
    def test_health_check(self) -> None:
        provider = resolve_provider(get_provider_name())
        health_fn = getattr(provider, "health_check", None)
        if health_fn:
            result = health_fn()
            assert result["reachable"] is True, f"Backend unreachable: {result}"
            print(f"\n  Provider: {result}")
        else:
            print(f"\n  Provider {provider.name}: no health check (ok for mock)")


@skip
class TestRealCompletion:
    def test_simple_completion(self) -> None:
        provider = resolve_provider(get_provider_name())
        result = provider.complete([
            {"role": "user", "content": "What is 2+2? Answer with just the number."},
        ])
        assert len(result) > 0
        print(f"\n  Response: {result[:100]}")

    def test_system_prompt(self) -> None:
        provider = resolve_provider(get_provider_name())
        result = provider.complete([
            {"role": "system", "content": "You are a helpful assistant. Be concise."},
            {"role": "user", "content": "What is Python?"},
        ])
        assert len(result) > 0
        print(f"\n  Response: {result[:200]}")


@skip
class TestDoctorWithBackend:
    def test_doctor(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = cli_main(["--data-dir", str(tmp_path), "doctor"])
        finally:
            sys.stdout = old
        print(buf.getvalue())
        assert code == 0


@skip
class TestFullLoopReal:
    def test_run_loop_with_query(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            code = cli_main([
                "--data-dir", str(tmp_path),
                "run-loop", "--json",
                "--query", "What is gradient descent? Be brief.",
            ])
        finally:
            sys.stdout = old
        assert code == 0, f"Loop failed:\n{buf.getvalue()}"
        result = json.loads(buf.getvalue())
        assert result["status"] == "completed"
        assert result.get("extracted", 0) >= 1
        print(f"\n  Loop result: {json.dumps(result, indent=2, default=str)[:500]}")

    def test_artifacts_created(self, tmp_path: Path) -> None:
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            cli_main([
                "--data-dir", str(tmp_path),
                "run-loop", "--json",
                "--query", "Hello world",
            ])
        finally:
            sys.stdout = old
        assert (tmp_path / "traces.jsonl").exists()
        assert (tmp_path / "events.jsonl").exists()
        assert (tmp_path / "dataset_promoted.jsonl").exists()
        assert (tmp_path / "snapshots" / "manifest.jsonl").exists()

    def test_multiple_queries_accumulate(self, tmp_path: Path) -> None:
        queries = [
            "What is LoRA?",
            "Explain fine-tuning",
            "What is a transformer?",
        ]
        for q in queries:
            old = sys.stdout
            sys.stdout = StringIO()
            try:
                cli_main([
                    "--data-dir", str(tmp_path),
                    "run-loop", "--json",
                    "--query", q,
                ])
            finally:
                sys.stdout = old

        # Check accumulated data
        old = sys.stdout
        sys.stdout = buf = StringIO()
        try:
            cli_main(["--data-dir", str(tmp_path), "list-snapshots"])
        finally:
            sys.stdout = old
        snapshots = json.loads(buf.getvalue())
        assert len(snapshots) >= 1
        print(f"\n  Snapshots after 3 queries: {len(snapshots)}")


@skip
class TestJudgeWithRealProvider:
    def test_judge_evaluation(self, tmp_path: Path) -> None:
        # Seed traces
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            cli_main([
                "--data-dir", str(tmp_path),
                "run-loop", "--json",
                "--query", "Explain what a neural network is",
            ])
        finally:
            sys.stdout = old

        # Run evaluation with judge
        sys.stdout = buf = StringIO()
        try:
            code = cli_main([
                "--data-dir", str(tmp_path),
                "evaluate-dataset", "--use-judge",
            ])
        finally:
            sys.stdout = old
        assert code == 0
        data = json.loads(buf.getvalue())
        print(f"\n  Judge evaluation: {json.dumps(data, indent=2)}")
        assert data["records_evaluated"] >= 1


@skip
class TestValidateTrainingReal:
    def test_validate_reports_deps(self, tmp_path: Path, hardware: dict[str, object]) -> None:
        # Seed and snapshot
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            cli_main([
                "--data-dir", str(tmp_path),
                "run-loop", "--json",
                "--query", "test",
            ])
        finally:
            sys.stdout = old

        sys.stdout = buf = StringIO()
        try:
            cli_main([
                "--data-dir", str(tmp_path),
                "validate-training", "--json",
            ])
        finally:
            sys.stdout = old

        data = json.loads(buf.getvalue())
        print(f"\n  Surface: {hardware.get('surface')}")
        print(f"  Ready: {data['ready']}")
        print(f"  Issues: {data['issues']}")
        print(f"  Warnings: {data['warnings']}")


@skip
class TestSurfaceSpecific:
    """Tests that adapt behavior based on detected hardware."""

    def test_surface_appropriate_response(
        self, tmp_path: Path, hardware: dict[str, object]
    ) -> None:
        surface = hardware.get("surface")
        print(f"\n  Detected surface: {surface}")

        # Run a completion
        provider = resolve_provider(get_provider_name())
        result = provider.complete([
            {"role": "user", "content": "Say hello in 5 words or less."},
        ])
        print(f"  Response: {result[:100]}")
        assert len(result) > 0

        # Validate training with surface context
        old = sys.stdout
        sys.stdout = StringIO()
        try:
            cli_main([
                "--data-dir", str(tmp_path),
                "run-loop", "--json",
                "--query", "test query",
            ])
        finally:
            sys.stdout = old

        sys.stdout = buf = StringIO()
        try:
            cli_main([
                "--data-dir", str(tmp_path),
                "validate-training", "--json",
            ])
        finally:
            sys.stdout = old

        data = json.loads(buf.getvalue())

        if surface == "cloud_gpu":
            print("  GPU detected — training should be viable")
        elif surface == "apple_silicon":
            print("  Apple Silicon — MPS may help, LoRA feasible")
        elif surface == "intel_mac":
            print("  Intel Mac — CPU only, SFT on tiny models only")
        else:
            print(f"  Surface: {surface} — check warnings")

        print(f"  Ready: {data['ready']}")
        print(f"  Method would be: {data['method']}")
