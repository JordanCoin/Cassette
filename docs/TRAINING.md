# Training Guide

End-to-end guide: from traces to a deployed model.

## Prerequisites

```bash
# Core install
uv pip install -e ".[dev]"

# Training dependencies (TRL, PEFT, PyTorch)
uv pip install -e ".[training]"

# Verify
cassette validate-training
```

The training deps are ~3GB to download. They include PyTorch, TRL, PEFT, and the GGUF converter.

## Step 1: Collect Traces

Cassette traces every LLM call that passes through its gateway. Route your app's LLM calls through Cassette:

```bash
# Start Cassette gateway
export CASSETTE_PROVIDER=llama_cpp_http
export CASSETTE_LLAMA_CPP_URL=http://localhost:11434
export CASSETTE_MODEL=llama3.2:3b
make dev
```

Point your app at `http://localhost:8000/v1/chat/completions` instead of your model server directly.

Use your app normally. Every LLM call gets traced.

## Step 2: Run the Loop

```bash
cassette run-loop
```

This:
- Extracts dataset records from traces
- Evaluates quality (rule-based + optional LLM judge)
- Promotes accepted records
- Creates a versioned snapshot
- Proposes a training method

## Step 3: Review the Plan

```bash
cassette plan-training
```

Shows: model, method (sft/lora/qlora), dataset path, estimated tokens, and the exact command.

## Step 4: Validate

```bash
cassette validate-training
```

Checks: dataset exists, model available, TRL/PEFT/PyTorch installed, GPU/MPS detected, method compatible with hardware.

## Step 5: Train

```bash
cassette train
```

This:
- Builds the training plan
- Validates readiness (blocks if not ready)
- Downloads the base model from HuggingFace (~3GB for 1.5B models)
- Runs LoRA training via TRL
- Saves the adapter to `data/gateway/models/<snapshot>-lora/`

Typical times:
- M4 Mac (MPS): ~20 minutes for 28 records, 3 epochs
- CPU only: ~1-2 hours
- GPU (A10): ~5 minutes

## Step 6: Export

```bash
cassette export-model --name my-model-v1
```

This:
1. Merges the LoRA adapter into the base model
2. Converts to GGUF format (auto-downloads llama.cpp converter if needed)
3. Registers with ollama as a named model

After this, `ollama list` shows your model and it's ready to use.

## Step 7: Compare

```bash
cassette compare --base qwen2.5:1.5b --adapter my-model-v1
```

Runs the same prompts through both models and scores across 8 dimensions:
- Format: valid JSON, no code fences, correct schema
- Data quality: clean values, numeric confidence, corrections present
- Completeness: entity coverage, no phantom entities

Only deploy if the adapter scores better than the base.

## Step 8: Deploy

Change your app's model name to the trained model:

```json
{"model": "my-model-v1"}
```

No code changes. No new dependencies. Just the model name.

## Step 9: Continue the Loop

Keep using your app. New traces accumulate. Periodically:

```bash
cassette run-loop
cassette train
cassette export-model --name my-model-v2
cassette compare --base my-model-v1 --adapter my-model-v2
```

Each version trains on all accumulated traces. The model improves with usage.

## Environment Variables

| Variable | Default | Used In |
|---|---|---|
| `CASSETTE_MODEL` | `default` | Training base model (ollama tag) |
| `CASSETTE_PROVIDER` | `mock` | Which backend to proxy to |
| `CASSETTE_LLAMA_CPP_URL` | `http://localhost:8080` | Backend URL |
| `CASSETTE_PROVIDER_TIMEOUT` | `60` | HTTP timeout for inference |

## Model Name Mapping

Cassette maps ollama tags to HuggingFace IDs automatically:

| Ollama Tag | HuggingFace ID |
|---|---|
| `llama3.2:3b` | `meta-llama/Llama-3.2-3B-Instruct` |
| `llama3.2:1b` | `meta-llama/Llama-3.2-1B-Instruct` |
| `qwen2.5:1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` |
| `qwen2.5:3b` | `Qwen/Qwen2.5-3B-Instruct` |
| `qwen3:8b` | `Qwen/Qwen3-8B` |
| `mistral:latest` | `mistralai/Mistral-7B-Instruct-v0.3` |

Note: Meta Llama models are gated — you need to accept the license on HuggingFace and run `huggingface-cli login`. Qwen models are Apache 2.0, no login needed.

## Troubleshooting

**"TRL not installed"** — Run `uv pip install -e ".[training]"`

**"Model not found at local path"** — Set `CASSETTE_MODEL` to an ollama tag or HuggingFace ID

**"401 Unauthorized"** — The model is gated. Use `huggingface-cli login` or switch to an ungated model like Qwen

**"GGUF conversion failed"** — The converter auto-downloads but needs git. Install git if missing.

**Training is slow** — On CPU, use a smaller model (1.5B). On M4 Mac, MPS is used automatically. For faster training, rent a GPU.
