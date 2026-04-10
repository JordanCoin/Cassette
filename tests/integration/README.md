# Integration Testing

Cassette integration tests validate the full loop on real hardware with real models.

## Test Surfaces

| Surface | Hardware | Model Size | Method | Expected |
|---|---|---|---|---|
| M4 Mac | Apple Silicon, MPS | 1B-3B GGUF | SFT/LoRA | Full loop works |
| Intel Mac | CPU only | 1B Q4 GGUF | SFT (tiny) | Full loop works, slow |
| Linux PC | CPU (or small GPU) | 1B-3B GGUF | SFT/LoRA | Full loop works |
| Cloud GPU | NVIDIA A10/A100 | 7B-8B | LoRA/QLoRA | Full loop + training |

## Quick Start (any surface)

```bash
# 1. Install Cassette
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Start a model server (pick one)

# llama.cpp (any platform)
llama-server -m model.gguf --port 8080

# OR ollama (any platform)
ollama serve  # then: ollama pull qwen2.5:1.5b

# 3. Point Cassette at it
export CASSETTE_PROVIDER=llama_cpp_http
export CASSETTE_LLAMA_CPP_URL=http://localhost:8080
# For ollama: export CASSETTE_LLAMA_CPP_URL=http://localhost:11434

# 4. Run integration tests
CASSETTE_INTEGRATION=1 uv run pytest tests/integration/ -v

# 5. Or run the full loop manually
cassette doctor
cassette run-loop --query "What is gradient descent?"
cassette validate-training
```

## Recommended Models Per Surface

### M4 Mac (8-16GB unified memory)
```bash
# Small and fast
llama-server -m Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --port 8080 -ngl 99

# Medium (if 16GB+)
llama-server -m Llama-3.2-3B-Instruct-Q4_K_M.gguf --port 8080 -ngl 99
```

### Intel Mac (CPU only, 8-16GB RAM)
```bash
# Smallest viable model — CPU inference
llama-server -m Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --port 8080 -ngl 0 -t 4

# Patience required — expect 5-15 tokens/sec
```

### Linux PC (CPU)
```bash
# Same as Intel Mac
llama-server -m Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --port 8080 -ngl 0
```

### Cloud GPU (RunPod / Vast.ai / Lambda)
```bash
# On the rented machine:
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8080

# Or with llama.cpp + GPU offload:
llama-server -m Llama-3.1-8B-Instruct-Q4_K_M.gguf --port 8080 -ngl 99
```

## Cloud GPU Rental (quick guide)

### RunPod (runpod.io)
- Cheapest for short runs
- Pick "Community Cloud" > A10 or A40 (~$0.30-0.50/hr)
- Use their PyTorch template
- SSH in, clone repo, run tests

### Vast.ai (vast.ai)
- Cheapest raw GPU rental
- Search for A10 24GB (~$0.20-0.40/hr)
- Rent, SSH in, run

### Lambda Labs (lambdalabs.com)
- Most reliable
- A10 instances (~$0.60/hr)
- On-demand or reserved

### Setup on any cloud GPU
```bash
# On the rented machine:
git clone https://github.com/JordanCoin/Cassette
cd Cassette
pip install uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,training]"

# Start model server
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8080 &

# Configure
export CASSETTE_PROVIDER=llama_cpp_http
export CASSETTE_LLAMA_CPP_URL=http://localhost:8080
export CASSETTE_INTEGRATION=1

# Run everything
cassette doctor
cassette run-loop --query "Explain backpropagation"
cassette validate-training
CASSETTE_INTEGRATION=1 uv run pytest tests/integration/ -v
```
