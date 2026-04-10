"""Training runner — executes LoRA/SFT training using TRL."""

from __future__ import annotations

import json
import time
from pathlib import Path

from libs.core.config import get_float, get_int


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
) -> dict[str, object]:
    """Run LoRA SFT training."""
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

    epochs = get_int("train_epochs")
    batch_size = get_int("train_batch_size")
    learning_rate = get_float("train_learning_rate")
    max_seq_length = get_int("train_max_seq_length")
    lora_r = get_int("lora_r")
    lora_alpha = get_int("lora_alpha")
    lora_dropout = get_float("lora_dropout")

    start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_data = prepare_dataset(dataset_path)
    if not chat_data:
        return {"success": False, "error": "Dataset is empty after conversion"}

    dataset = Dataset.from_list(chat_data)

    print(f"  Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto",
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

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
