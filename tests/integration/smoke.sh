#!/usr/bin/env bash
# Quick smoke test against a real backend
# Usage: ./tests/integration/smoke.sh [data_dir]
#
# Prerequisites:
#   - Cassette installed: uv pip install -e ".[dev]"
#   - Model server running at CASSETTE_LLAMA_CPP_URL (default: localhost:8080)
#   - CASSETTE_PROVIDER=llama_cpp_http

set -euo pipefail

DATA_DIR="${1:-/tmp/cassette-smoke-$$}"
echo "=== Cassette Smoke Test ==="
echo "Data dir: $DATA_DIR"
echo ""

# 1. Doctor
echo "--- doctor ---"
cassette --data-dir "$DATA_DIR" doctor
echo ""

# 2. Run loop with real query
echo "--- run-loop ---"
cassette --data-dir "$DATA_DIR" run-loop --query "What is gradient descent? Be very brief."
echo ""

# 3. Run again to accumulate
echo "--- second run ---"
cassette --data-dir "$DATA_DIR" run-loop --query "What is a neural network? One sentence."
echo ""

# 4. List snapshots
echo "--- snapshots ---"
cassette --data-dir "$DATA_DIR" list-snapshots
echo ""

# 5. Propose training
echo "--- propose ---"
cassette --data-dir "$DATA_DIR" propose-training
echo ""

# 6. Plan training
echo "--- plan ---"
cassette --data-dir "$DATA_DIR" plan-training
echo ""

# 7. Validate training
echo "--- validate ---"
cassette --data-dir "$DATA_DIR" validate-training
echo ""

# 8. Evaluate with judge
echo "--- evaluate with judge ---"
cassette --data-dir "$DATA_DIR" evaluate-dataset --use-judge
echo ""

echo "=== Smoke test complete ==="
echo "Artifacts at: $DATA_DIR"
echo ""
ls -la "$DATA_DIR"
