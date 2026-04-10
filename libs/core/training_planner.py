"""Training dry-run planner — validates inputs and builds a concrete plan."""

from __future__ import annotations

from pathlib import Path

from libs.core.config import get_int, get_str
from libs.core.contracts import DatasetSnapshot, TrainingPlan
from libs.core.model_registry import to_hf_name
from libs.core.training_plan import select_method


def _build_command(plan: TrainingPlan) -> str:
    """Generate a representative training command string from config."""
    hf_model = to_hf_name(plan.base_model)
    lora_r = get_int("lora_r")
    lora_alpha = get_int("lora_alpha")
    epochs = get_int("train_epochs")
    batch_size = get_int("train_batch_size")

    parts = [
        "python -m trl sft",
        f"--model_name_or_path {hf_model}",
        f"--dataset_path {plan.dataset_path}",
        f"--output_dir {plan.output_dir}",
    ]
    if plan.method in ("lora", "qlora"):
        parts.append("--use_peft")
        parts.append(f"--lora_r {lora_r}")
        parts.append(f"--lora_alpha {lora_alpha}")
    if plan.method == "qlora":
        parts.append("--load_in_4bit")
    parts.append(f"--num_train_epochs {epochs}")
    parts.append(f"--per_device_train_batch_size {batch_size}")
    return " \\\n  ".join(parts)


def build_training_plan(
    snapshot: DatasetSnapshot,
    data_dir: Path,
    output_base: Path | None = None,
) -> TrainingPlan:
    """Validate inputs and produce a concrete TrainingPlan."""
    promoted_file = get_str("dataset_promoted_file")
    tokens_per_record = get_int("tokens_per_record")

    snapshots_dir = data_dir / "snapshots"
    dataset_path = snapshots_dir / f"{snapshot.snapshot_id}.jsonl"

    if not dataset_path.exists():
        dataset_path = data_dir / promoted_file

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

    base_model = get_str("model")

    plan = TrainingPlan(
        snapshot_id=snapshot.snapshot_id,
        dataset_path=str(dataset_path),
        base_model=base_model,
        method=method,
        estimated_tokens=record_count * tokens_per_record,
        output_dir=str(output_dir),
        command="",
        notes=f"{record_count} records, {method} method",
    )
    plan = plan.model_copy(update={"command": _build_command(plan)})
    return plan
