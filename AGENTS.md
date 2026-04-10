# AGENTS.md — Cassette

This file defines how autonomous agents must operate within this repository.

---

## Primary Objective

Build and evolve Cassette as a **self-improving system**.

Do not optimize for shortcuts, demos, or one-off scripts.

---

## Hard Constraints

* Python 3.11+
* All external systems behind interfaces (no direct coupling)
* Every meaningful operation must be **traceable**
* Web research tooling is **mandatory**
* No silent side effects

---

## System Rules

### 1. Everything is an Event

* All inputs, outputs, and actions must be recorded
* No “invisible” computation

---

### 2. No Training Without Evaluation

* New data must be:

  * validated
  * deduplicated
  * evaluated
* Training requires explicit approval or passing gates

---

### 3. Deterministic First

* Prefer reproducible pipelines over clever heuristics
* Avoid hidden state

---

### 4. One Change Per Iteration

* Do not batch large changes
* Each commit must:

  * pass tests
  * include reasoning
  * be reversible

---

### 5. Web Usage is Required for Uncertainty

* If implementation details are unclear:

  * use web.search
  * use web.fetch
* Store all sources in trace output

---

## Risk Tiers

### High Risk

* Training logic
* Dataset generation
* Model promotion
* Security / execution boundaries

### Medium Risk

* Pipeline stages
* Evaluation harness
* Tool integrations

### Low Risk

* Docs
* Tests
* Refactors

---

## Commands

```bash
uv run ruff check .
uv run mypy .
uv run pytest -q
```

All must pass before commit.

---

## Anti-Patterns

* Writing code without tests
* Adding dependencies without justification
* Skipping trace logging
* Hardcoding environment-specific behavior
* Silent failures

---

## Expected Behavior

Agents should:

* Think in systems, not functions
* Prefer clarity over cleverness
* Leave the repo in a better state after each change

---

## Definition of Done

A task is complete when:

* Code works
* Tests pass
* Behavior is traceable
* Docs are updated
* Future agents can understand the change
