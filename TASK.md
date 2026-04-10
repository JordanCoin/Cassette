# TASK.md

## Status

The loop is closed. Cassette can observe, evaluate, train, export, compare, and deploy.

Validated end-to-end with OpenFOIA on M4 Mac:
- 29 government documents traced through the gateway
- 28 records curated into a training dataset
- LoRA adapter trained in 21 minutes (Qwen 2.5 1.5B, MPS)
- Adapter exported to GGUF, registered with ollama
- Format compliance: 0.20 → 0.90 (10/10 records improved, 0 regressions)
- Trained model running in OpenFOIA with zero code changes

## What Works

- **Tracing**: every LLM call through the gateway is recorded
- **Evaluation**: rule-based checks + LLM-as-judge scoring
- **Dataset pipeline**: extract → evaluate → promote → snapshot
- **Training**: LoRA via TRL, plan → validate → execute
- **Export**: merge adapter → GGUF → ollama registration
- **Comparison**: 8-dimension scoring matrix (format, data quality, completeness)
- **CLI**: 16 commands covering the full lifecycle
- **Docker**: containerized deployment with optional backends

## What's Next

- More training data (100+ documents) for better entity decision generalization
- Test on Intel Mac, Linux PC, and cloud GPU surfaces
- Publish adapter to HuggingFace or OpenFOIA releases
- Automated training pipeline (run loop on schedule)
