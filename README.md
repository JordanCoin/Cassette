# Cassette

Cassette records intelligence as it happens—and improves from it.

**Record. Learn. Rewrite.**

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/yourname/cassette
cd cassette
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Check everything is working
cassette doctor

# 3. Run the full loop with a test query
cassette run-loop --query "What is gradient descent?"

# 4. Inspect results
cassette list-snapshots
```

That's it. Cassette is running in mock mode — no GPU or model server needed.

### Using a real model backend

```bash
# Start a llama.cpp server
llama-server -m your-model.gguf --port 8080

# Point Cassette at it
export CASSETTE_PROVIDER=llama_cpp_http
export CASSETTE_LLAMA_CPP_URL=http://localhost:8080

# Verify the connection
cassette doctor

# Run for real
cassette run-loop --query "Explain backpropagation"
```

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

253+ tests passing. Foundation complete through milestone 22.

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

## CLI Reference

```
cassette doctor                        # Full system diagnostics
cassette health                        # Quick provider + system check
cassette run-loop                      # Full observe-to-proposal pipeline
cassette run-loop --query "question"   # Seed a query, then run the loop
cassette extract-dataset               # Extract dataset from traces
cassette evaluate-dataset              # Evaluate + promote + write datasets
cassette snapshot-dataset              # Snapshot the promoted dataset
cassette list-snapshots                # List available snapshots
cassette propose-training              # Generate a training proposal
```

All commands accept `--data-dir <path>` to override the data directory (default: `data/gateway`).

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CASSETTE_PROVIDER` | `mock` | Active model provider (`mock`, `llama_cpp_http`) |
| `CASSETTE_LLAMA_CPP_URL` | `http://localhost:8080` | llama.cpp server URL |
| `CASSETTE_SEARCH_URL` | `http://localhost:8888` | Search API URL (SearXNG) |
| `CASSETTE_PROVIDER_TIMEOUT` | `60` | Provider HTTP timeout (seconds) |

Copy `.env.example` to `.env` and adjust for your setup.

---

## HTTP API

```bash
# Start the gateway
make dev

# Health
curl http://localhost:8000/healthz
curl http://localhost:8000/healthz/provider

# Chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"hello"}]}'

# Full loop
curl -X POST http://localhost:8000/loop/run

# Metrics
curl http://localhost:8000/metrics
```

---

## Architecture

```
libs/core/        — domain logic, contracts, ports (no IO)
libs/adapters/    — IO implementations (JSONL, HTTP providers, writers)
services/gateway/ — FastAPI application
services/orchestrator/ — stage runner and stages
```

* **Gateway** — OpenAI-compatible interface + trace logging + metrics
* **Orchestrator** — Stage-based runner for research + evaluation pipelines
* **Data Plane** — JSONL persistence, extraction, evaluation, promotion, snapshots
* **Adapters** — Pluggable backends for model providers, web search, web fetch, storage

---

## Design Constraints

* Must run on a **low-resource machine** (Intel MacBook, CPU-only)
* Must scale to **multi-GPU / Kubernetes**
* Must support **web research tooling**
* Must maintain **full traceability**

---

## Philosophy

Cassette treats intelligence as a **process, not a model**.

Models are temporary. The loop is the product.

---

## License

MIT
