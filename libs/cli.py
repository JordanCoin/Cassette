"""Cassette CLI — run workflows without the HTTP server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from libs.adapters.dataset_writer import write_dataset
from libs.adapters.jsonl_store import JsonlStore
from libs.core.evaluator import evaluate_records
from libs.core.extractor import extract_records
from libs.core.pipeline import run_full_loop
from libs.core.promoter import apply_eval_decisions, select_promoted
from libs.core.snapshots import create_snapshot, list_snapshots
from libs.core.training_plan import build_proposal

DEFAULT_DATA_DIR = Path("data/gateway")


def _get_store(data_dir: Path) -> JsonlStore:
    return JsonlStore(data_dir)


def cmd_run_loop(args: argparse.Namespace) -> int:
    store = _get_store(Path(args.data_dir))
    result = run_full_loop(store, Path(args.data_dir), trace_limit=args.limit)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") == "completed" else 1


def cmd_extract_dataset(args: argparse.Namespace) -> int:
    store = _get_store(Path(args.data_dir))
    traces = store.get_latest_traces(args.limit)
    records = extract_records(traces)
    output_path = Path(args.data_dir) / "dataset.jsonl"
    count = write_dataset(records, output_path)
    print(json.dumps({
        "traces_scanned": len(traces),
        "records_extracted": count,
        "output_path": str(output_path),
    }, indent=2))
    return 0


def cmd_snapshot_dataset(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    promoted_path = data_dir / "dataset_promoted.jsonl"
    snapshots_dir = data_dir / "snapshots"
    try:
        snapshot = create_snapshot(promoted_path, snapshots_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(snapshot.model_dump(mode="json"), indent=2, default=str))
    return 0


def cmd_list_snapshots(args: argparse.Namespace) -> int:
    snapshots_dir = Path(args.data_dir) / "snapshots"
    snapshots = list_snapshots(snapshots_dir)
    print(json.dumps([s.model_dump(mode="json") for s in snapshots], indent=2, default=str))
    return 0


def cmd_propose_training(args: argparse.Namespace) -> int:
    snapshots_dir = Path(args.data_dir) / "snapshots"
    snapshots = list_snapshots(snapshots_dir)

    if not snapshots:
        print("Error: No snapshots available", file=sys.stderr)
        return 1

    if args.snapshot_id:
        match = [s for s in snapshots if s.snapshot_id == args.snapshot_id]
        if not match:
            print(f"Error: Snapshot not found: {args.snapshot_id}", file=sys.stderr)
            return 1
        snapshot = match[0]
    else:
        snapshot = snapshots[-1]

    proposal = build_proposal(snapshot.snapshot_id, snapshot.record_count)
    print(json.dumps(proposal.model_dump(mode="json"), indent=2, default=str))
    return 0


def cmd_evaluate_dataset(args: argparse.Namespace) -> int:
    store = _get_store(Path(args.data_dir))
    data_dir = Path(args.data_dir)
    traces = store.get_latest_traces(args.limit)
    records = extract_records(traces)

    eval_results = evaluate_records(records)
    labeled = apply_eval_decisions(records, eval_results)
    promoted = select_promoted(labeled)

    write_dataset(labeled, data_dir / "dataset_labeled.jsonl")
    write_dataset(promoted, data_dir / "dataset_promoted.jsonl")

    accepted = sum(1 for r in eval_results if r.decision == "accepted")
    rejected = sum(1 for r in eval_results if r.decision == "rejected")
    needs_review = sum(1 for r in eval_results if r.decision == "needs_review")

    print(json.dumps({
        "records_evaluated": len(eval_results),
        "accepted": accepted,
        "rejected": rejected,
        "needs_review": needs_review,
        "promoted": len(promoted),
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cassette", description="Cassette CLI")
    parser.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR),
        help="Data directory (default: data/gateway)",
    )
    sub = parser.add_subparsers(dest="command")

    run_loop = sub.add_parser("run-loop", help="Run the full observe-to-proposal loop")
    run_loop.add_argument("--limit", type=int, default=200, help="Max traces to scan")

    extract = sub.add_parser("extract-dataset", help="Extract dataset from traces")
    extract.add_argument("--limit", type=int, default=200, help="Max traces to scan")

    evaluate = sub.add_parser("evaluate-dataset", help="Evaluate, promote, and write datasets")
    evaluate.add_argument("--limit", type=int, default=200, help="Max traces to scan")

    sub.add_parser("snapshot-dataset", help="Snapshot the promoted dataset")

    sub.add_parser("list-snapshots", help="List available dataset snapshots")

    propose = sub.add_parser("propose-training", help="Generate a training proposal")
    propose.add_argument("--snapshot-id", default=None, help="Snapshot ID (default: latest)")

    return parser


_COMMANDS = {
    "run-loop": cmd_run_loop,
    "extract-dataset": cmd_extract_dataset,
    "evaluate-dataset": cmd_evaluate_dataset,
    "snapshot-dataset": cmd_snapshot_dataset,
    "list-snapshots": cmd_list_snapshots,
    "propose-training": cmd_propose_training,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
