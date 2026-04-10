"""Model name mapping between runtime (ollama) and training (HuggingFace) identifiers.

The same model has different names in different contexts:
- Ollama uses short tags: "llama3.2:3b"
- HuggingFace uses org/name: "meta-llama/Llama-3.2-3B-Instruct"
- TRL needs the HuggingFace name to load trainable weights

This registry maps between them so Cassette handles the translation.
"""

from __future__ import annotations

# Known mappings: ollama tag → HuggingFace model ID
_OLLAMA_TO_HF: dict[str, str] = {
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1:70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama2:latest": "meta-llama/Llama-2-7b-chat-hf",
    "mistral:latest": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen3:8b": "Qwen/Qwen3-8B",
    "qwen3.5:9b": "Qwen/Qwen3.5-9B",
}

# Reverse mapping
_HF_TO_OLLAMA: dict[str, str] = {v: k for k, v in _OLLAMA_TO_HF.items()}


def to_hf_name(model: str) -> str:
    """Convert a model name to its HuggingFace identifier for training.

    If already a HuggingFace name (contains '/'), returns as-is.
    If an ollama tag, looks up the mapping.
    If unknown, returns as-is (user may have a custom model).
    """
    if "/" in model:
        return model
    return _OLLAMA_TO_HF.get(model, model)


def to_ollama_name(model: str) -> str:
    """Convert a HuggingFace model ID to its ollama tag for inference.

    If already an ollama tag (no '/'), returns as-is.
    If a known HuggingFace name, looks up the mapping.
    """
    if "/" not in model:
        return model
    return _HF_TO_OLLAMA.get(model, model)


def is_known_model(model: str) -> bool:
    """Check if a model name is in the registry (either format)."""
    return model in _OLLAMA_TO_HF or model in _HF_TO_OLLAMA or "/" in model
