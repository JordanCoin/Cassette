"""Shared training proposal heuristics."""

from __future__ import annotations

from libs.core.contracts import TrainingProposal
from libs.core.settings import get_model_name

TOKENS_PER_RECORD = 200
MIN_RECORDS_FOR_TRAINING = 10


def _resolve_base_model() -> str:
    """Use the configured model from CASSETTE_MODEL."""
    return get_model_name()


def select_method(record_count: int) -> str:
    if record_count < MIN_RECORDS_FOR_TRAINING:
        return "sft"
    if record_count < 100:
        return "lora"
    return "qlora"


def training_notes(record_count: int) -> str:
    if record_count < MIN_RECORDS_FOR_TRAINING:
        return (
            f"Only {record_count} records — below minimum "
            f"({MIN_RECORDS_FOR_TRAINING}). Training not recommended yet."
        )
    if record_count < 100:
        return "Small dataset. LoRA recommended for parameter efficiency."
    return "Sufficient data for QLoRA fine-tuning."


def build_proposal(snapshot_id: str, record_count: int) -> TrainingProposal:
    """Build a training proposal from snapshot metadata."""
    return TrainingProposal(
        snapshot_id=snapshot_id,
        record_count=record_count,
        base_model=_resolve_base_model(),
        method=select_method(record_count),
        estimated_tokens=record_count * TOKENS_PER_RECORD,
        notes=training_notes(record_count),
    )
