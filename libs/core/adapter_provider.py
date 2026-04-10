# mypy: ignore-errors
"""Adapter provider — loads a base model + LoRA adapter for inference."""

from __future__ import annotations

from pathlib import Path


class AdapterProvider:
    """Loads a HuggingFace model with a LoRA adapter for comparison."""

    def __init__(self, base_model: str, adapter_path: str) -> None:
        self._base_model = base_model
        self._adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return f"adapter:{Path(self._adapter_path).name}"

    def _load(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._adapter_path)
        base = AutoModelForCausalLM.from_pretrained(
            self._base_model, device_map="auto", torch_dtype="auto",
        )
        self._model = PeftModel.from_pretrained(base, self._adapter_path)

    def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self._load()
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        import torch

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
