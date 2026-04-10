# Cassette

Cassette records intelligence as it happens—and improves from it.

**Record. Learn. Rewrite.**

Cassette is a self-improving research and training system that continuously records its behavior, evaluates outcomes, and refines its intelligence loop.

It is not a chatbot, wrapper, or one-off agent.

It is a **system that evolves itself**.

---

## What Cassette Does

Cassette turns every interaction into structured data:

* Prompts → recorded
* Tool usage → traced
* Outcomes → evaluated
* Failures → learned from
* Datasets → generated
* Models → improved

All of it feeds a continuous loop:

> **observe → evaluate → curate → train → redeploy → repeat**

---

## Current State

253 tests passing. Foundation complete through milestone 21.

### What's built

* **Gateway** — OpenAI-compatible API (`/v1/chat/completions`), health checks, Prometheus metrics
* **Provider abstraction** — mock + llama.cpp HTTP backends, configurable via env vars
* **Trace/event system** — every request, stage, and pipeline step is recorded as structured JSONL
* **Task ledger** — create, track, and update units of work with status transitions
* **Orchestrator** — stage-based runner with event instrumentation (echo, gather_sources, propose_training)
* **Web tooling** — search + fetch adapters behind ports for research stages
* **Data pipeline** — extract → evaluate → promote → snapshot → propose, runnable as one loop
* **Dataset versioning** — immutable snapshots with content hashing and manifest tracking
* **Training proposals** — structured plans generated from snapshot metadata
* **CLI** — `cassette` command for all core workflows without HTTP
* **Metrics** — Prometheus-compatible `/metrics` endpoint

---

## Architecture (High-Level)

* **Gateway** — OpenAI-compatible interface + trace logging + metrics
* **Orchestrator** — Stage-based runner for research + evaluation pipelines
* **Data Plane** — JSONL persistence, dataset extraction, evaluation, promotion, snapshots
* **Adapters** — Pluggable backends for model providers, web search, web fetch, storage

---

## Getting Started

```bash
git clone https://github.com/yourname/cassette
cd cassette

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run checks
make check

# Start the gateway
make dev

# Run the full loop
cassette run-loop

# Check system health
cassette health
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CASSETTE_PROVIDER` | `mock` | Active model provider (`mock`, `llama_cpp_http`) |
| `CASSETTE_LLAMA_CPP_URL` | `http://localhost:8080` | llama.cpp server URL |
| `CASSETTE_SEARCH_URL` | `http://localhost:8888` | Search API URL (SearXNG) |
| `CASSETTE_PROVIDER_TIMEOUT` | `60` | Provider HTTP timeout (seconds) |

---

## CLI

```
cassette run-loop              # Full observe-to-proposal pipeline
cassette extract-dataset       # Extract dataset from traces
cassette evaluate-dataset      # Evaluate + promote + write datasets
cassette snapshot-dataset      # Snapshot the promoted dataset
cassette list-snapshots        # List available snapshots
cassette propose-training      # Generate a training proposal
cassette health                # Check system + provider health
```

---

## Design Constraints

* Must run on a **low-resource machine** (Intel MacBook, CPU-only)
* Must scale to **multi-GPU / Kubernetes**
* Must support **web research tooling**
* Must maintain **full traceability**

---

## Philosophy

Cassette treats intelligence as a **process, not a model**.

Models are temporary.

The loop is the product.

---

## License

MIT
