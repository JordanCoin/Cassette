# TASK.md

## Current Status

Milestones 1–21 complete. The foundation is built and hardened.

## What Exists

* Python project scaffold with uv, ruff, mypy, pytest (253 tests passing)
* Canonical Pydantic contracts: Event, Task, Trace, DatasetRecord, EvalResult, DatasetSnapshot, TrainingProposal
* FastAPI gateway with:
  * `/healthz`, `/healthz/provider`
  * `POST /v1/chat/completions` (OpenAI-compatible, mock + llama.cpp HTTP providers)
  * `/metrics` (Prometheus text format)
  * Task ledger CRUD (`/tasks`)
  * Debug read endpoints (`/debug/traces`, `/debug/events`)
  * Data pipeline endpoints (`/debug/extract-dataset`, `/debug/evaluate-dataset`, `/debug/promote-dataset`, `/debug/snapshot-dataset`, `/debug/snapshots`)
  * Full loop trigger (`/loop/run`)
  * Orchestrator trigger (`/orchestrator/run`)
* CLI (`cassette`) with: run-loop, extract-dataset, evaluate-dataset, snapshot-dataset, list-snapshots, propose-training, health
* JSONL append-only persistence with read-back
* ModelProvider abstraction with mock and llama.cpp HTTP backends
* Settings via environment variables
* Enriched trace/event recording with backend metadata and failure recording
* Orchestrator with Stage protocol, stage runner, and stages: echo, gather_sources, propose_training
* Web tooling: WebSearch/WebFetch ports with HTTP adapters
* Fine-grained event instrumentation within stages
* Dataset extraction, evaluation, promotion, and versioned snapshots
* Training proposal generation from snapshot metadata
* Prometheus-style metrics (requests, provider calls, tasks, stages, loop runs)
* Provider hardening: timeout handling, error classification, health checks
* Data sanity guards in extraction

## What's Next

Potential next milestones (pick based on priority):

1. **Real model backend integration** — wire up a live llama.cpp or vLLM server and validate end-to-end
2. **Docker Compose stack** — containerized deployment with gateway + model server
3. **Configuration file** — replace env vars with a unified config.yaml supporting profiles (laptop, workstation, cluster)
4. **Dataset versioning with DVC** — integrate DVC for Git-like dataset/model versioning
5. **Evaluation harness expansion** — add LLM-as-judge, golden test suites, regression detection
6. **Training execution** — LoRA/QLoRA training via TRL/Axolotl, gated by evaluation results

## Constraints (unchanged)

* Keep implementation minimal
* Do not add training logic without evaluation gates
* Do not add orchestration complexity without justification
* Preserve modular architecture for later scale
