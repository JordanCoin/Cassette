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
from libs.core.provider_registry import resolve_provider
from libs.core.settings import get_provider_name, get_search_url
from libs.core.snapshots import create_snapshot, list_snapshots
from libs.core.training_plan import build_proposal

DEFAULT_DATA_DIR = Path("data/gateway")


def _get_store(data_dir: Path) -> JsonlStore:
    return JsonlStore(data_dir)


def _print_json(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def _error(msg: str) -> int:
    print(f"Error: {msg}", file=sys.stderr)
    return 1


def _format_loop_result(result: dict[str, object]) -> str:
    """Format a loop result as readable text."""
    lines: list[str] = []
    status = result.get("status", "unknown")
    lines.append(f"  Status: {status}")

    if result.get("note"):
        lines.append(f"  Note:   {result['note']}")
        return "\n".join(lines)

    if result.get("failed_stage"):
        lines.append(f"  Failed: {result['failed_stage']}")
        lines.append(f"  Error:  {result.get('error', '')}")
        return "\n".join(lines)

    extracted = result.get("extracted", 0)
    lines.append(f"  Extracted:  {extracted} records from traces")

    ev = result.get("evaluation")
    if isinstance(ev, dict):
        lines.append(
            f"  Evaluated:  {ev.get('accepted', 0)} accepted, "
            f"{ev.get('rejected', 0)} rejected, "
            f"{ev.get('needs_review', 0)} needs review"
        )

    promoted = result.get("promoted", 0)
    lines.append(f"  Promoted:   {promoted} records")

    snap = result.get("snapshot")
    if isinstance(snap, dict):
        lines.append(f"  Snapshot:   {snap.get('snapshot_id', 'n/a')}")

    prop = result.get("proposal")
    if isinstance(prop, dict):
        lines.append(
            f"  Proposal:   {prop.get('method', '?')} on {prop.get('base_model', '?')} "
            f"(~{prop.get('estimated_tokens', 0)} tokens)"
        )

    return "\n".join(lines)


# -- Commands --


def cmd_run_loop(args: argparse.Namespace) -> int:
    store = _get_store(Path(args.data_dir))

    if args.query:
        from libs.core.recorder import record_chat

        record_chat(
            store,
            model="cli-input",
            messages=[{"role": "user", "content": args.query}],
            response="[recorded] Query: " + args.query,
            latency_ms=0.0,
            provider="cli",
        )
        print("Recorded query as trace.", file=sys.stderr)

    result = run_full_loop(store, Path(args.data_dir), trace_limit=args.limit)

    if args.json:
        _print_json(result)
    else:
        print()
        print(_format_loop_result(result))
        print()
        data_dir = Path(args.data_dir)
        print("  Artifacts:")
        for name in ["traces.jsonl", "events.jsonl", "dataset_promoted.jsonl"]:
            p = data_dir / name
            if p.exists():
                print(f"    {p}")
        snap_dir = data_dir / "snapshots"
        if snap_dir.exists():
            print(f"    {snap_dir}/")
        print()

    return 0 if result.get("status") == "completed" else 1


def cmd_extract_dataset(args: argparse.Namespace) -> int:
    store = _get_store(Path(args.data_dir))
    traces = store.get_latest_traces(args.limit)
    records = extract_records(traces)
    output_path = Path(args.data_dir) / "dataset.jsonl"
    count = write_dataset(records, output_path)
    _print_json({
        "traces_scanned": len(traces),
        "records_extracted": count,
        "output_path": str(output_path),
    })
    return 0


def cmd_snapshot_dataset(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    promoted_path = data_dir / "dataset_promoted.jsonl"
    snapshots_dir = data_dir / "snapshots"
    try:
        snapshot = create_snapshot(promoted_path, snapshots_dir)
    except FileNotFoundError:
        return _error(
            f"No promoted dataset found at {promoted_path}.\n"
            "  Run 'cassette evaluate-dataset' first to create one."
        )
    except FileExistsError:
        return _error("Snapshot already exists for this dataset content.")
    _print_json(snapshot.model_dump(mode="json"))
    return 0


def cmd_list_snapshots(args: argparse.Namespace) -> int:
    snapshots_dir = Path(args.data_dir) / "snapshots"
    snapshots = list_snapshots(snapshots_dir)
    _print_json([s.model_dump(mode="json") for s in snapshots])
    return 0


def cmd_propose_training(args: argparse.Namespace) -> int:
    snapshots_dir = Path(args.data_dir) / "snapshots"
    snapshots = list_snapshots(snapshots_dir)

    if not snapshots:
        return _error(
            "No snapshots available.\n"
            "  Run 'cassette evaluate-dataset' then 'cassette snapshot-dataset' first."
        )

    if args.snapshot_id:
        match = [s for s in snapshots if s.snapshot_id == args.snapshot_id]
        if not match:
            available = ", ".join(s.snapshot_id for s in snapshots[-3:])
            return _error(f"Snapshot not found: {args.snapshot_id}\n  Available: {available}")
        snapshot = match[0]
    else:
        snapshot = snapshots[-1]

    proposal = build_proposal(snapshot.snapshot_id, snapshot.record_count)
    _print_json(proposal.model_dump(mode="json"))
    return 0


def cmd_evaluate_dataset(args: argparse.Namespace) -> int:
    store = _get_store(Path(args.data_dir))
    data_dir = Path(args.data_dir)
    traces = store.get_latest_traces(args.limit)
    records = extract_records(traces)

    if not records:
        _print_json({
            "records_evaluated": 0,
            "note": "No valid records found. Send some chat requests first.",
        })
        return 0

    eval_results = evaluate_records(records)
    labeled = apply_eval_decisions(records, eval_results)
    promoted = select_promoted(labeled)

    write_dataset(labeled, data_dir / "dataset_labeled.jsonl")
    write_dataset(promoted, data_dir / "dataset_promoted.jsonl")

    accepted = sum(1 for r in eval_results if r.decision == "accepted")
    rejected = sum(1 for r in eval_results if r.decision == "rejected")
    needs_review = sum(1 for r in eval_results if r.decision == "needs_review")

    _print_json({
        "records_evaluated": len(eval_results),
        "accepted": accepted,
        "rejected": rejected,
        "needs_review": needs_review,
        "promoted": len(promoted),
    })
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    store = _get_store(data_dir)

    provider_name = get_provider_name()
    try:
        prov = resolve_provider(provider_name)
    except ValueError as exc:
        _print_json({"status": "error", "error": str(exc)})
        return 1

    health_fn = getattr(prov, "health_check", None)
    if health_fn:
        provider_health = health_fn()
    else:
        provider_health = {"reachable": True, "note": "no health check"}

    _print_json({
        "status": "ok",
        "provider": provider_name,
        "provider_health": provider_health,
        "data_dir": str(data_dir),
        "traces_exist": store.traces_path.exists(),
        "events_exist": store.events_path.exists(),
    })
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run system checks and report pass/fail for each component."""
    data_dir = Path(args.data_dir)
    checks: list[dict[str, str]] = []
    all_ok = True

    # 1. Data directory
    store = _get_store(data_dir)
    checks.append({"check": "data_directory", "status": "ok", "path": str(data_dir)})

    # 2. Provider
    provider_name = get_provider_name()
    try:
        prov = resolve_provider(provider_name)
        checks.append({"check": "provider_configured", "status": "ok", "provider": provider_name})
    except ValueError as exc:
        checks.append({
            "check": "provider_configured", "status": "fail",
            "error": str(exc),
            "fix": "Set CASSETTE_PROVIDER to 'mock' or 'llama_cpp_http'",
        })
        all_ok = False
        prov = None

    # 3. Provider connectivity
    if prov is not None:
        health_fn = getattr(prov, "health_check", None)
        if health_fn:
            health = health_fn()
            if health.get("reachable"):
                checks.append({
                    "check": "provider_reachable", "status": "ok",
                    "url": str(health.get("url", "")),
                })
            else:
                checks.append({
                    "check": "provider_reachable", "status": "fail",
                    "error": str(health.get("error", "unreachable")),
                    "fix": f"Start the model server at {health.get('url', 'configured URL')}",
                })
                all_ok = False
        else:
            checks.append({
                "check": "provider_reachable", "status": "skip",
                "note": f"Provider '{provider_name}' has no health check (this is fine for mock)",
            })

    # 4. Search endpoint
    search_url = get_search_url()
    try:
        import httpx

        httpx.get(f"{search_url}/", timeout=3.0)
        checks.append({"check": "search_endpoint", "status": "ok", "url": search_url})
    except Exception:
        checks.append({
            "check": "search_endpoint", "status": "warn",
            "url": search_url,
            "note": "Search endpoint not reachable. Web research stages will fail.",
            "fix": "Start SearXNG: docker run -p 8888:8080 searxng/searxng",
        })

    # 5. Existing data
    if store.traces_path.exists():
        store.get_latest_traces(1)
        checks.append({
            "check": "existing_data", "status": "ok",
            "note": "Trace data found",
        })
    else:
        checks.append({
            "check": "existing_data", "status": "info",
            "note": "No trace data yet. Send requests or run 'cassette run-loop --query \"test\"'",
        })

    # Output
    print()
    for c in checks:
        status = c["status"].upper()
        label = c["check"]
        if status == "OK":
            note = c.get("note", "")
            print(f"  [OK]   {label}" + (f": {note}" if note else ""))
        elif status == "FAIL":
            print(f"  [FAIL] {label}: {c.get('error', '')}")
            if "fix" in c:
                print(f"         Fix: {c['fix']}")
            all_ok = False
        elif status == "WARN":
            print(f"  [WARN] {label}: {c.get('note', '')}")
            if "fix" in c:
                print(f"         Fix: {c['fix']}")
        elif status == "SKIP":
            print(f"  [SKIP] {label}: {c.get('note', '')}")
        else:
            print(f"  [INFO] {label}: {c.get('note', '')}")
    print()

    if all_ok:
        print("  All checks passed. Cassette is ready.")
    else:
        print("  Some checks failed. See above for fixes.")
    print()

    return 0 if all_ok else 1


_DEMO_QUERIES = [
    "What is gradient descent and why does it matter for neural networks?",
    "Explain the difference between LoRA and full fine-tuning.",
    "How does a transformer attention mechanism work?",
]


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a demo workflow that shows the full Cassette pipeline."""
    data_dir = Path(args.data_dir)
    store = _get_store(data_dir)

    print()
    print("  Cassette Demo")
    print("  =============")
    print()
    print("  This demo seeds sample queries, runs the full pipeline,")
    print("  and shows where to find the results.")
    print()

    # Seed queries
    from libs.core.recorder import record_chat

    for query in _DEMO_QUERIES:
        record_chat(
            store,
            model="demo",
            messages=[{"role": "user", "content": query}],
            response=f"[demo] Recorded: {query[:50]}",
            latency_ms=0.0,
            provider="demo",
        )
        print(f"  Seeded: {query[:60]}...")

    print()
    print("  Running pipeline: extract -> evaluate -> promote -> snapshot -> propose")
    print()

    result = run_full_loop(store, data_dir, trace_limit=200)
    print(_format_loop_result(result))
    print()

    # Show artifacts
    print("  Where to find results:")
    print(f"    Traces:     {data_dir / 'traces.jsonl'}")
    print(f"    Events:     {data_dir / 'events.jsonl'}")
    print(f"    Dataset:    {data_dir / 'dataset_promoted.jsonl'}")
    print(f"    Snapshots:  {data_dir / 'snapshots/'}")
    print()
    print("  Next steps:")
    print("    cassette list-snapshots         # see versioned datasets")
    print("    cassette propose-training       # see training plan")
    print("    cassette doctor                 # check system health")
    print()

    return 0 if result.get("status") == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cassette", description="Cassette CLI")
    parser.add_argument(
        "--data-dir", default=str(DEFAULT_DATA_DIR),
        help="Data directory (default: data/gateway)",
    )
    sub = parser.add_subparsers(dest="command")

    run_loop = sub.add_parser("run-loop", help="Run the full observe-to-proposal pipeline")
    run_loop.add_argument("--limit", type=int, default=200, help="Max traces to scan")
    run_loop.add_argument("--query", default=None, help="Seed a query before running")
    run_loop.add_argument("--json", action="store_true", help="Output raw JSON instead of summary")

    extract = sub.add_parser("extract-dataset", help="Extract dataset from traces")
    extract.add_argument("--limit", type=int, default=200, help="Max traces to scan")

    evaluate = sub.add_parser("evaluate-dataset", help="Evaluate, promote, and write datasets")
    evaluate.add_argument("--limit", type=int, default=200, help="Max traces to scan")

    sub.add_parser("snapshot-dataset", help="Snapshot the promoted dataset")

    sub.add_parser("list-snapshots", help="List available dataset snapshots")

    propose = sub.add_parser("propose-training", help="Generate a training proposal")
    propose.add_argument("--snapshot-id", default=None, help="Snapshot ID (default: latest)")

    sub.add_parser("health", help="Quick provider and system check")
    sub.add_parser("doctor", help="Full system diagnostics with pass/fail checks")
    sub.add_parser("demo", help="Run a sample workflow showing the full pipeline")

    return parser


_COMMANDS = {
    "run-loop": cmd_run_loop,
    "extract-dataset": cmd_extract_dataset,
    "evaluate-dataset": cmd_evaluate_dataset,
    "snapshot-dataset": cmd_snapshot_dataset,
    "list-snapshots": cmd_list_snapshots,
    "propose-training": cmd_propose_training,
    "health": cmd_health,
    "doctor": cmd_doctor,
    "demo": cmd_demo,
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

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (ConnectionError, RuntimeError) as exc:
        return _error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
