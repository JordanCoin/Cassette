# TASK.md

## Current Status

**v0.1.0** — Milestones 1–27 complete. Open source ready.

## What Exists

The full observe-to-plan pipeline runs locally:

* **Gateway** — OpenAI-compatible API, health checks, Prometheus metrics
* **Provider abstraction** — mock + llama.cpp HTTP backends with timeout/error handling
* **Trace/event system** — structured JSONL recording of all system activity
* **Task ledger** — status-tracked units of work
* **Orchestrator** — stage-based runner (echo, gather_sources, propose_training, plan_training, validate_training)
* **Web tooling** — search + fetch adapters with event instrumentation
* **Data pipeline** — extract → evaluate → promote → snapshot → propose → plan → validate
* **Dataset versioning** — immutable snapshots with content hashing
* **Training planning** — concrete plans with TRL commands and hardware validation
* **CLI** — 11 commands including demo, doctor, and full pipeline
* **Docker Compose** — containerized deployment with optional backend/search
* **Metrics** — Prometheus-compatible counters for requests, providers, stages, loops
* **298 tests** passing, strict typing, linting

## What's Next

See GitHub issues for planned work:

* **Training execution** — LoRA/QLoRA via TRL with gated promotion
* **LLM-as-judge evaluation** — model-based quality scoring for dataset curation

## Design Principles

* Everything is recorded and traceable
* No training without evaluation gates
* Prefer honesty over capability (fail clearly, don't pretend)
* Local-first, scales to GPU/Kubernetes
* Systems over scripts
