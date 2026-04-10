# Integrating Cassette with Any Project

Cassette works with any application that uses LLMs through an OpenAI-compatible API. No SDK, no dependency, no code changes in your app — just point your LLM calls at Cassette's gateway.

## How It Works

```
Your App → Cassette Gateway (localhost:8000) → Your Model Server (ollama, llama.cpp, vLLM)
                    |
                    └── traces every call, builds training data
```

Your app thinks it's talking to a normal OpenAI-compatible endpoint. Cassette proxies the request to your actual model server, traces the prompt and response, and returns the result unchanged.

## Setup (5 minutes)

### 1. Start your model server

```bash
ollama serve  # or llama-server, or vLLM
```

### 2. Start Cassette

```bash
cd /path/to/cassette
source .venv/bin/activate

export CASSETTE_PROVIDER=llama_cpp_http
export CASSETTE_LLAMA_CPP_URL=http://localhost:11434  # your model server
export CASSETTE_MODEL=llama3.2:3b                     # your model name
make dev
```

### 3. Point your app at Cassette

Change your app's LLM endpoint from your model server to Cassette:

```
# Before: app talks directly to model server
http://localhost:11434/v1/chat/completions

# After: app talks to Cassette, Cassette talks to model server
http://localhost:8000/v1/chat/completions
```

That's it. Your app works exactly the same. Cassette traces everything.

### 4. Run the loop

After your app has been running for a while:

```bash
cassette run-loop          # extract, evaluate, promote, snapshot, propose
cassette plan-training     # see the training command
cassette validate-training # check if training can run
```

### 5. Train and compare

```bash
cassette train                                    # run training
cassette compare --base llama3.2:3b --adapter v1  # compare results
```

## What Your App Needs

- **Uses OpenAI-compatible API** (`/v1/chat/completions`)
- **Has a configurable base URL** for the LLM endpoint
- That's it

## What Your App Does NOT Need

- No Cassette dependency
- No Cassette import
- No code changes (beyond the URL)
- No special output format
- No awareness that Cassette exists

## Framework Examples

### Python (openai SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
```

### Python (httpx/requests)

```python
response = httpx.post("http://localhost:8000/v1/chat/completions", json={...})
```

### JavaScript/TypeScript

```typescript
const response = await fetch("http://localhost:8000/v1/chat/completions", {
  method: "POST",
  body: JSON.stringify({ model: "llama3.2:3b", messages: [...] }),
});
```

### Any language

Any HTTP client that can POST JSON to an OpenAI-compatible endpoint works.

## Tips

- **response_format**: Cassette passes through `response_format: {"type": "json_object"}` to the backend, so structured output works correctly
- **temperature/max_tokens**: also passed through
- **Cassette adds no latency** beyond the proxy hop (typically <1ms)
- **If Cassette goes down**, your app gets connection errors — it does not silently fall back. This is intentional: you want to know if tracing stops
- **Multiple apps** can share one Cassette gateway — all traces go to the same dataset
