"""Dataset promotion — applies eval decisions to produce quality-labeled datasets."""

from __future__ import annotations

from libs.core.contracts import DatasetRecord, EvalResult


def apply_eval_decisions(
    records: list[DatasetRecord],
    results: list[EvalResult],
) -> list[DatasetRecord]:
    """Return new records with quality updated from eval decisions.

    Does not mutate the originals.
    """
    decision_map = {str(r.record_id): r.decision for r in results}
    updated: list[DatasetRecord] = []
    for record in records:
        decision = decision_map.get(str(record.record_id))
        if decision:
            updated.append(record.model_copy(update={"quality": decision}))
        else:
            updated.append(record.model_copy())
    return updated


def select_promoted(records: list[DatasetRecord]) -> list[DatasetRecord]:
    """Return only records with quality == 'accepted'."""
    return [r for r in records if r.quality == "accepted"]
