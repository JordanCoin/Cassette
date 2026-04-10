"""Pre-training validation — checks whether training can realistically run."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from libs.core.contracts import TrainingPlan, TrainingReadiness


def _check_dataset(plan: TrainingPlan) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    path = Path(plan.dataset_path)
    if not path.exists():
        issues.append(f"Dataset not found: {plan.dataset_path}")
    elif path.stat().st_size == 0:
        issues.append(f"Dataset is empty: {plan.dataset_path}")
    else:
        lines = path.read_text().strip().split("\n")
        if len(lines) < 5:
            warnings.append(
                f"Dataset has only {len(lines)} records — training may not be effective"
            )
    return issues, warnings


def _check_model(plan: TrainingPlan) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []
    model = plan.base_model
    # Check if it looks like a local path
    if "/" not in model and not model.startswith("meta-"):
        local = Path(model)
        if not local.exists():
            issues.append(f"Model not found at local path: {model}")
    # For HF-style IDs, we can't check without network — just note it
    if "/" in model:
        warnings.append(f"Model '{model}' will be downloaded from HuggingFace if not cached")
    return issues, warnings


def _check_dependencies() -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []

    for pkg, label in [("trl", "TRL"), ("peft", "PEFT"), ("torch", "PyTorch")]:
        try:
            importlib.import_module(pkg)
        except ImportError:
            issues.append(f"{label} ({pkg}) is not installed")

    return issues, warnings


def _check_hardware(method: str) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []

    has_gpu = False
    try:
        import torch

        has_gpu = torch.cuda.is_available()
        if not has_gpu:
            has_gpu = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        pass

    if not has_gpu:
        if method == "qlora":
            issues.append("QLoRA requires GPU — no CUDA or MPS device detected")
        elif method == "lora":
            warnings.append("LoRA training on CPU will be very slow")
        elif method == "sft":
            warnings.append("SFT training on CPU is possible but slow")

    return issues, warnings


def _check_python() -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    v = sys.version_info
    if v < (3, 11):
        warnings.append(f"Python {v.major}.{v.minor} detected — 3.11+ recommended")
    return [], warnings


def validate_training(plan: TrainingPlan) -> TrainingReadiness:
    """Run all pre-training checks and return a readiness report."""
    all_issues: list[str] = []
    all_warnings: list[str] = []

    for check in [
        lambda: _check_dataset(plan),
        lambda: _check_model(plan),
        _check_dependencies,
        lambda: _check_hardware(plan.method),
        _check_python,
    ]:
        issues, warnings = check()
        all_issues.extend(issues)
        all_warnings.extend(warnings)

    ready = len(all_issues) == 0

    notes_parts: list[str] = []
    notes_parts.append(f"Platform: {platform.system()} {platform.machine()}")
    notes_parts.append(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
    if ready:
        notes_parts.append("All checks passed")
    else:
        notes_parts.append(f"{len(all_issues)} blocking issue(s)")

    return TrainingReadiness(
        snapshot_id=plan.snapshot_id,
        dataset_path=plan.dataset_path,
        base_model=plan.base_model,
        method=plan.method,
        ready=ready,
        issues=all_issues,
        warnings=all_warnings,
        notes="; ".join(notes_parts),
    )
