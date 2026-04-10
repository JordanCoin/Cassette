# Cassette Walkthrough

This guide walks through a complete Cassette workflow, explaining what happens at each step and where to find the results.

## 1. Run the demo

```bash
cassette demo
```

This seeds three sample queries, runs the full pipeline, and shows the results. No model server needed — mock mode works.

## 2. What the pipeline does

```
extract     Read recorded traces, filter valid chat records
evaluate    Score each record (empty? error-tainted? valid?)
promote     Keep only accepted records
snapshot    Create an immutable versioned copy of the promoted dataset
propose     Generate a training plan based on dataset size
```

## 3. Inspect the artifacts

After a run, your data directory contains:

```
data/gateway/
  traces.jsonl              # Every recorded interaction
  events.jsonl              # Every system event (requests, stages, loops)
  tasks.jsonl               # Task ledger (pipeline runs, stages)
  dataset_promoted.jsonl    # Accepted training candidates
  dataset_labeled.jsonl     # All records with quality labels
  snapshots/
    manifest.jsonl          # Snapshot index
    20260410-*.jsonl        # Immutable dataset copies
```

### Read a trace

```bash
head -1 data/gateway/traces.jsonl | python -m json.tool
```

### Read the promoted dataset

```bash
cat data/gateway/dataset_promoted.jsonl | python -m json.tool
```

### List snapshots

```bash
cassette list-snapshots
```

### See the training proposal

```bash
cassette propose-training
```

## 4. Run with your own queries

```bash
cassette run-loop --query "How does RLHF work?"
cassette run-loop --query "What are LoRA adapters?"
```

Each run adds traces and re-evaluates the full dataset.

## 5. Run with a real model

```bash
export CASSETTE_PROVIDER=llama_cpp_http
export CASSETTE_LLAMA_CPP_URL=http://localhost:8080

# Start your model server, then:
cassette doctor
cassette run-loop --query "Explain attention mechanisms"
```

## 6. Understanding the training proposal

The proposal output tells you:

- **method**: `sft` (small), `lora` (medium), `qlora` (large) — based on dataset size
- **base_model**: which model would be fine-tuned
- **estimated_tokens**: rough training token count
- **notes**: why this method was chosen

Training is not executed yet — the proposal is advisory.

## 7. Sample workflows

### Research assistant evaluation

```bash
cassette run-loop --query "Compare vLLM and TGI for local inference"
cassette run-loop --query "What quantization methods work with llama.cpp?"
cassette run-loop --query "How to evaluate LLM performance on domain tasks?"
cassette list-snapshots
cassette propose-training
```

### Build up a training dataset

```bash
# Seed many diverse queries
for q in "What is backpropagation?" "Explain transformers" "What is RLHF?"; do
  cassette run-loop --query "$q"
done

# Check dataset quality
cassette evaluate-dataset
cassette propose-training
```
