"""Training dry-run planner — validates inputs and builds a concrete plan."""

from __future__ import annotations

from pathlib import Path

from libs.core.contracts import DatasetSnapshot, TrainingPlan
from libs.core.training_plan import TOKENS_PER_RECORD, _resolve_base_model, select_method


def _build_command(plan: TrainingPlan) -> str:
    """Generate a representative training command string."""
    parts = [
        "python -m trl sft",
        f"--model_name_or_path {plan.base_model}",
        f"--dataset_path {plan.dataset_path}",
        f"--output_dir {plan.output_dir}",
    ]
    if plan.method in ("lora", "qlora"):
        parts.append("--use_peft")
        parts.append("--lora_r 16")
        parts.append("--lora_alpha 32")
    if plan.method == "qlora":
        parts.append("--load_in_4bit")
    parts.append("--num_train_epochs 3")
    parts.append("--per_device_train_batch_size 4")
    return " \\\n  ".join(parts)


def build_training_plan(
    snapshot: DatasetSnapshot,
    data_dir: Path,
    output_base: Path | None = None,
) -> TrainingPlan:
    """Validate inputs and produce a concrete TrainingPlan.

    Raises ValueError if the snapshot dataset is missing or empty.
    """
    # Find the snapshot dataset file
    snapshots_dir = data_dir / "snapshots"
    dataset_path = snapshots_dir / f"{snapshot.snapshot_id}.jsonl"

    if not dataset_path.exists():
        # Fall back to promoted dataset
        dataset_path = data_dir / "dataset_promoted.jsonl"

    if not dataset_path.exists():
        raise ValueError(f"Dataset not found for snapshot {snapshot.snapshot_id}")

    if dataset_path.stat().st_size == 0:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    record_count = len(dataset_path.read_text().strip().split("\n"))
    if record_count == 0:
        raise ValueError(f"Dataset has no records: {dataset_path}")

    method = select_method(record_count)
    out_dir = output_base or (data_dir / "models")
    output_dir = out_dir / f"{snapshot.snapshot_id}-{method}"

    plan = TrainingPlan(
        snapshot_id=snapshot.snapshot_id,
        dataset_path=str(dataset_path),
        base_model=_resolve_base_model(),
        method=method,
        estimated_tokens=record_count * TOKENS_PER_RECORD,
        output_dir=str(output_dir),
        command="",  # filled below
        notes=f"{record_count} records, {method} method",
    )
    plan = plan.model_copy(update={"command": _build_command(plan)})
    return plan
