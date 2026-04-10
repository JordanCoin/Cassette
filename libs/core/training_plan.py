"""Shared training proposal heuristics."""

from __future__ import annotations

from libs.core.config import get_int, get_str
from libs.core.contracts import TrainingProposal


def select_method(record_count: int) -> str:
    min_records = get_int("min_records_for_training")
    qlora_threshold = get_int("qlora_threshold")
    if record_count < min_records:
        return "sft"
    if record_count < qlora_threshold:
        return "lora"
    return "qlora"


def training_notes(record_count: int) -> str:
    min_records = get_int("min_records_for_training")
    qlora_threshold = get_int("qlora_threshold")
    if record_count < min_records:
        return (
            f"Only {record_count} records — below minimum "
            f"({min_records}). Training not recommended yet."
        )
    if record_count < qlora_threshold:
        return "Small dataset. LoRA recommended for parameter efficiency."
    return "Sufficient data for QLoRA fine-tuning."


def build_proposal(snapshot_id: str, record_count: int) -> TrainingProposal:
    """Build a training proposal from snapshot metadata."""
    tokens_per_record = get_int("tokens_per_record")
    return TrainingProposal(
        snapshot_id=snapshot_id,
        record_count=record_count,
        base_model=get_str("model"),
        method=select_method(record_count),
        estimated_tokens=record_count * tokens_per_record,
        notes=training_notes(record_count),
    )
