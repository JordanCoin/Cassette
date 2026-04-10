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

## Core Ideas

* **Everything is recorded**
  Every meaningful interaction becomes an immutable event.

* **Learning is gated**
  No automatic training without evaluation and verification.

* **Research is first-class**
  Web search + retrieval are required, not optional.

* **Systems over scripts**
  Cassette is built as a long-running, composable machine—not a collection of tools.

---

## Architecture (High-Level)

* **Gateway**
  OpenAI-compatible interface + trace logging

* **Loop (Orchestrator)**
  Runs research + evaluation pipelines

* **Archive**
  Stores traces, datasets, artifacts

* **Trainer**
  Fine-tunes models using curated data

* **Engine**
  Model inference layer (local → GPU → cluster)

---

## Design Constraints

* Must run on a **low-resource machine** (Intel MacBook, CPU-only)
* Must scale to **multi-GPU / Kubernetes**
* Must support **web research tooling**
* Must maintain **full traceability**

---

## Getting Started

```bash
git clone https://github.com/yourname/cassette
cd cassette

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

make dev
```

---

## Philosophy

Cassette treats intelligence as a **process, not a model**.

Models are temporary.

The loop is the product.

---

## Status

Early-stage. Built for iteration, not stability.

---

## License

MIT
