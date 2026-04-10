# PROGRAM.md — Cassette Build Loop

This file defines the execution loop for building Cassette.

Agents must follow this strictly.

---

## Setup

1. Create a branch:

```bash
git checkout -b build/<date>-<task>
```

2. Read:

* AGENTS.md
* README.md
* contracts/* (if present)

3. Run baseline checks:

```bash
uv run ruff check .
uv run mypy .
uv run pytest -q
```

---

## Core Loop

Repeat until task is complete:

### Step 1 — Select One Task

Pick a single, well-defined deliverable:

* endpoint
* schema
* pipeline stage
* adapter

---

### Step 2 — Understand Before Building

If unclear:

* use web.search
* use web.fetch
* document findings

---

### Step 3 — Implement Minimally

* Write the smallest working version
* Avoid over-engineering
* Follow existing patterns

---

### Step 4 — Add Tests

* Unit tests required
* Validate edge cases
* Ensure reproducibility

---

### Step 5 — Run Checks

```bash
uv run ruff check .
uv run mypy .
uv run pytest -q
```

All must pass.

---

### Step 6 — Update Documentation

* Update README if behavior changes
* Update schemas if needed
* Add notes for future agents

---

### Step 7 — Commit

Commit message must include:

* what changed
* why it changed
* rollback plan

---

## Pipeline Priorities (Build Order)

1. Contracts (event schemas)
2. Gateway (traceable inference)
3. Web tooling (search + fetch)
4. Orchestrator (loop engine)
5. Dataset pipeline
6. Trainer
7. Evaluation harness

---

## Mandatory Behaviors

* Log all traces
* Use structured outputs
* Prefer composition over inheritance
* Keep interfaces stable

---

## Failure Handling

If something fails:

* do not ignore
* log failure
* attempt fix (max 3 retries)
* escalate via documentation

---

## Completion Criteria

A feature is complete when:

* It works end-to-end
* It is logged and traceable
* It integrates with the loop
* It can be reused

---

## Final Rule

Do not build features.

Build the system that builds features.
