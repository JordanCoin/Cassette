"""Training runner — executes LoRA/SFT training using TRL programmatically."""

from __future__ import annotations

import json
from pathlib import Path


def prepare_dataset(snapshot_path: Path) -> list[dict[str, list[dict[str, str]]]]:
    """Convert Cassette dataset records to TRL chat format."""
    records = []
    for line in snapshot_path.read_text().strip().split("\n"):
        data = json.loads(line)
        messages = []
        for msg in data.get("messages", []):
            messages.append({
                "role": msg.get("role", "user"),
                "content": str(msg.get("content", "")),
            })
        # Add the response as assistant message
        if data.get("response"):
            messages.append({
                "role": "assistant",
                "content": data["response"],
            })
        if messages:
            records.append({"messages": messages})
    return records


def run_training(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    method: str = "lora",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
) -> dict[str, object]:
    """Run LoRA SFT training. Returns result dict."""
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        return {
            "success": False,
            "error": f"Missing training dependency: {exc}. "
            "Install with: pip install cassette[training]",
        }

    import time

    start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare dataset
    chat_data = prepare_dataset(dataset_path)
    if not chat_data:
        return {"success": False, "error": "Dataset is empty after conversion"}

    dataset = Dataset.from_list(chat_data)

    # Load tokenizer and model
    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Training config
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_seq_length,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )

    # Train
    print(f"  Training {len(chat_data)} examples, {epochs} epochs...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    duration = time.monotonic() - start

    return {
        "success": True,
        "output_dir": str(output_dir),
        "duration_sec": round(duration, 1),
        "records": len(chat_data),
        "epochs": epochs,
        "method": method,
    }
