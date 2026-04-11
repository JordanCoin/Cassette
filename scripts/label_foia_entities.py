"""Label OpenFOIA entities via Cassette gateway → hosted teacher model.

Reads entities from a local copy of OpenFOIA's SQLite DB, sends each one
through Cassette's gateway using the per_entity prompt (with the surrounding
context included), and writes labeled records to dataset_labeled.jsonl.

Every request flows through the gateway, so traces are recorded and can be
extracted into training records via the existing `extract-dataset` command.

Usage:
    # Set these to point Cassette at the hosted teacher
    export CASSETTE_PROVIDER=llama_cpp_http
    export CASSETTE_PROVIDER_URL=https://api.fireworks.ai/inference
    export CASSETTE_MODEL=accounts/fireworks/models/qwen3-8b
    export CASSETTE_PROVIDER_API_KEY=fw_...

    # Start the gateway in another terminal:
    uv run uvicorn services.gateway.app:app --port 8000

    # Then run:
    uv run python scripts/label_foia_entities.py \
        --db data/foia/foia.db \
        --out data/gateway/foia_labeled.jsonl \
        --limit 50   # small smoke test first
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import httpx

from libs.core.prompt_loader import get_output_schema, load_prompt

GATEWAY_URL = "http://localhost:8000/v1/chat/completions"
HEALTH_URL = "http://localhost:8000/healthz/provider"


def iter_entities(db_path: Path, limit: int | None = None) -> list[dict]:
    """Load entities from OpenFOIA's SQLite DB."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    query = (
        "SELECT id, entity_type, raw_text, context "
        "FROM entities WHERE raw_text IS NOT NULL AND raw_text != ''"
    )
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()
    return [dict(r) for r in rows]


def build_messages(entity_type: str, raw_text: str, context: str) -> list[dict[str, str]]:
    """Render per_entity prompt with context included in the user message."""
    data = load_prompt("per_entity")
    system = data["system"].strip()
    ctx_line = f"\nContext: {context.strip()}" if context and context.strip() else ""
    user = (
        f'Type: {entity_type}\n'
        f'Text: "{raw_text}"'
        f"{ctx_line}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user.strip()},
    ]


def label_one(
    client: httpx.Client,
    entity: dict,
    schema: dict,
) -> dict | None:
    """Call the gateway for one entity. Returns parsed label dict or None."""
    messages = build_messages(
        entity["entity_type"],
        entity["raw_text"],
        entity.get("context") or "",
    )
    payload = {
        "model": "ignored-by-gateway",  # gateway uses its configured model
        "messages": messages,
        "response_format": {"type": "json_object", "schema": schema},
        "temperature": 0,
        "max_tokens": 200,
        "think": False,
    }
    try:
        resp = client.post(GATEWAY_URL, json=payload, timeout=60.0)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"  [error] {exc}", file=sys.stderr)
        return None

    content = resp.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        print(f"  [parse-fail] {content[:100]}", file=sys.stderr)
        return None

    return {
        "entity_id": entity["id"],
        "entity_type": entity["entity_type"],
        "raw_text": entity["raw_text"],
        "context": entity.get("context") or "",
        "decision": parsed.get("decision"),
        "reason": parsed.get("reason", ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--progress-every", type=int, default=50, help="Print status every N entities"
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        return 1

    schema = get_output_schema("per_entity")
    if schema is None:
        print("per_entity prompt has no output_schema", file=sys.stderr)
        return 1

    # Health check the gateway and the provider behind it
    with httpx.Client() as client:
        try:
            health = client.get(HEALTH_URL, timeout=10.0).json()
            print(f"Gateway provider health: {health}")
            if not health.get("reachable"):
                print("Provider not reachable — check CASSETTE_PROVIDER_URL and API key")
                return 1
        except httpx.HTTPError as exc:
            print(f"Gateway not reachable at {HEALTH_URL}: {exc}", file=sys.stderr)
            print("Start it with: uv run uvicorn services.gateway.app:app --port 8000")
            return 1

        entities = iter_entities(args.db, limit=args.limit)
        print(f"Loaded {len(entities)} entities from {args.db}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        labeled = 0
        errors = 0
        keep = 0
        remove = 0
        start = time.monotonic()

        with args.out.open("w") as out_file:
            for i, entity in enumerate(entities, 1):
                result = label_one(client, entity, schema)
                if result is None:
                    errors += 1
                    continue
                out_file.write(json.dumps(result) + "\n")
                out_file.flush()
                labeled += 1
                if result["decision"] == "keep":
                    keep += 1
                elif result["decision"] == "remove":
                    remove += 1

                if i % args.progress_every == 0:
                    elapsed = time.monotonic() - start
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(entities) - i) / rate if rate > 0 else 0
                    print(
                        f"  [{i}/{len(entities)}] "
                        f"keep={keep} remove={remove} err={errors} "
                        f"rate={rate:.1f}/s eta={eta / 60:.1f}min"
                    )

        elapsed = time.monotonic() - start
        print()
        print(f"Done in {elapsed / 60:.1f} min")
        print(f"  Labeled: {labeled}")
        print(f"  Keep:    {keep}")
        print(f"  Remove:  {remove}")
        print(f"  Errors:  {errors}")
        print(f"  Output:  {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
