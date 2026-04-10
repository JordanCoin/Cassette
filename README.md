# Cassette

A self-improving system that records how AI models behave, evaluates their output, and trains them to be better — all locally, all yours.

**Record. Learn. Rewrite.**

---

## What It Does

Point any app's LLM calls at Cassette. It traces every interaction, builds training datasets from real usage, trains LoRA adapters, and proves whether the trained model is actually better — with numbers, not hope.

```
your app → Cassette gateway → model server (ollama, llama.cpp, vLLM)
                |
                └── traces → dataset → training → better model
```

**Validated with [OpenFOIA](https://github.com/JordanCoin/openfoia):** 29 government documents extracted, 28 training records curated, LoRA adapter trained in 21 minutes on M4 Mac, format compliance improved from 0.20 → 0.90 on entity validation task. Zero code changes in OpenFOIA — just changed the model name.

---

## Quickstart

```bash
# Install
git clone https://github.com/JordanCoin/Cassette
cd Cassette
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Check system
cassette doctor

# Run the full pipeline with a test query
cassette run-loop --query "What is gradient descent?"

# See what was produced
cassette list-snapshots
cassette propose-training
```

No GPU needed. No model server needed. Mock mode works out of the box.

```bash
# Or run the guided demo
cassette demo
```

---

## The Full Loop

```bash
# 1. Route your app's LLM calls through Cassette
export CASSETTE_PROVIDER=llama_cpp_http
export CASSETTE_LLAMA_CPP_URL=http://localhost:11434
export CASSETTE_MODEL=llama3.2:3b
make dev

# 2. Use your app normally — every LLM call is traced

# 3. Run the pipeline
cassette run-loop

# 4. Train a LoRA adapter
pip install cassette[training]
cassette train

# 5. Export to ollama
cassette export-model --name my-model-v1

# 6. Compare base vs trained
cassette compare --base llama3.2:3b --adapter my-model-v1

# 7. If better, update your app config:
#    "model": "my-model-v1"

# 8. Keep using your app → more traces → retrain → repeat
```

See [docs/TRAINING.md](docs/TRAINING.md) for the full training guide.

---

## Real-World Example: OpenFOIA

[OpenFOIA](https://github.com/JordanCoin/openfoia) is an open-source FOIA investigation toolkit. It uses LLMs to validate entity extraction from government documents.

**Integration:** One config change — point the LLM base_url at Cassette's gateway. No code changes.

**Result:** Cassette traced 29 document extractions, built a training dataset, trained a LoRA adapter on Qwen 2.5 1.5B in 21 minutes on an M4 Mac, and the trained model returned clean JSON entity validation instead of code-fenced markdown. Measured improvement: 0.20 → 0.90 format compliance across 10 test records, 0 regressions.

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for how to connect any project.

---

## CLI Reference

```
cassette demo                                # Guided demo of the full pipeline
cassette doctor                              # Full system diagnostics
cassette health                              # Quick provider check
cassette run-loop                            # observe → evaluate → promote → snapshot → propose
cassette run-loop --query "question"         # Seed a query, then run
cassette train                               # Plan → validate → execute LoRA training
cassette export-model --name my-model        # Merge adapter → GGUF → register with ollama
cassette compare --base m1 --adapter m2      # Score base vs adapter across 8 dimensions
cassette validate-training                   # Check if training can run on this hardware
cassette plan-training                       # Show the training command without running it
cassette evaluate-dataset                    # Evaluate + promote dataset records
cassette evaluate-dataset --use-judge        # Add LLM-as-judge scoring
cassette extract-dataset                     # Extract records from traces
cassette snapshot-dataset                    # Version the promoted dataset
cassette list-snapshots                      # List versioned datasets
cassette propose-training                    # Generate training proposal
```

All commands accept `--data-dir <path>` to override the data directory.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `CASSETTE_PROVIDER` | `mock` | Model provider (`mock`, `llama_cpp_http`) |
| `CASSETTE_MODEL` | `default` | Model name (ollama tag or HuggingFace ID) |
| `CASSETTE_LLAMA_CPP_URL` | `http://localhost:8080` | Model server URL |
| `CASSETTE_SEARCH_URL` | `http://localhost:8888` | Search API URL (SearXNG) |
| `CASSETTE_PROVIDER_TIMEOUT` | `60` | Provider HTTP timeout (seconds) |

---

## Architecture

```
libs/core/        — contracts, ports, domain logic (no IO)
libs/adapters/    — JSONL store, HTTP providers, writers
services/gateway/ — FastAPI gateway (OpenAI-compatible)
services/orchestrator/ — stage runner and stages
```

**Gateway** — proxies LLM calls, traces everything, serves metrics
**Orchestrator** — staged execution with event instrumentation
**Data pipeline** — extraction, evaluation, promotion, versioned snapshots
**Training** — LoRA via TRL, model comparison, GGUF export to ollama
**Adapters** — pluggable backends for storage, providers, web tooling

---

## Comparison Scoring Matrix

`cassette compare` scores responses across 8 dimensions:

| Category | Metric | What it measures |
|---|---|---|
| Format | `valid_json` | Response is parseable JSON |
| Format | `no_code_fences` | No markdown wrappers |
| Format | `correct_schema` | Has expected keys (keep/remove) |
| Data | `clean_values` | No type prefixes or confidence strings in values |
| Data | `numeric_confidence` | Confidence values are numbers |
| Data | `has_corrections` | Corrected field present |
| Completeness | `entity_coverage` | Output accounts for input entities |
| Completeness | `no_phantoms` | Doesn't invent entities beyond input |

---

## Docker Compose

```bash
docker compose up --build                    # Mock mode
CASSETTE_PROVIDER=llama_cpp_http \
  docker compose --profile with-backend up   # With model server
```

See [compose.yaml](compose.yaml) for full configuration.

---

## Limitations

* **Small training sets produce conservative models** — 28 records taught format, not deep entity decisions. More data = broader confidence.
* **No DVC integration** — dataset versioning is file-based snapshots
* **Single provider at a time** — no multi-model routing
* **No auth** — endpoints are open (intended for local/dev use)
* **GGUF conversion requires git** — llama.cpp converter is auto-downloaded but needs git

---

## Documentation

| Guide | What |
|---|---|
| [docs/TRAINING.md](docs/TRAINING.md) | Full training guide: traces to deployed model |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | How to connect any project to Cassette |
| [examples/WALKTHROUGH.md](examples/WALKTHROUGH.md) | Step-by-step walkthrough with examples |
| [tests/integration/README.md](tests/integration/README.md) | Multi-surface integration testing |
| [AGENTS.md](AGENTS.md) | Development rules |
| [PROGRAM.md](PROGRAM.md) | Build loop process |

---

## Philosophy

Cassette treats intelligence as a **process, not a model**.

Models are temporary. The loop is the product.

The goal is freedom from subscription-based AI: train what you want, on your data, on your hardware, improving from your actual usage.

---

## Contributing

Early-stage, built for iteration. Issues and PRs welcome.

340+ tests. Strict typing. Full linting. Integration tests validated on M4 Mac with real ollama backend.

---

## License

MIT
