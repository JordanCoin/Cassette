# mypy: ignore-errors
"""Model exporter — merges LoRA adapter into base model and registers with ollama."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def merge_adapter(base_model: str, adapter_path: Path, output_path: Path) -> Path:
    """Merge LoRA adapter into base model and save as a full model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype="auto",
    )

    print(f"  Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    print("  Merging adapter into base model...")
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"  Saving merged model to: {output_path}")
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    return output_path


def convert_to_gguf(model_path: Path, output_file: Path) -> Path:
    """Convert a HuggingFace model to GGUF format using llama.cpp's convert script."""
    # Try to find the convert script
    convert_script = shutil.which("convert_hf_to_gguf.py")
    if not convert_script:
        # Try common locations
        for candidate in [
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path("/usr/local/bin/convert_hf_to_gguf.py"),
        ]:
            if candidate.exists():
                convert_script = str(candidate)
                break

    if not convert_script:
        # Fall back to python package
        try:
            result = subprocess.run(
                ["python3", "-m", "llama_cpp.convert", "--help"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                convert_script = "llama_cpp.convert"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if not convert_script:
        raise FileNotFoundError(
            "Cannot find GGUF converter. Install llama-cpp-python or "
            "clone llama.cpp and ensure convert_hf_to_gguf.py is on PATH."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Converting to GGUF: {output_file}")

    cmd = ["python3", convert_script, str(model_path), "--outfile", str(output_file)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"GGUF conversion failed: {result.stderr[:500]}")

    return output_file


def register_with_ollama(gguf_path: Path, model_name: str) -> str:
    """Create an ollama model from a GGUF file."""
    modelfile = gguf_path.parent / "Modelfile"
    modelfile.write_text(f"FROM {gguf_path}\n")

    print(f"  Registering with ollama as: {model_name}")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile)],
        capture_output=True, text=True, timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Ollama registration failed: {result.stderr[:500]}")

    return model_name


def export_model(
    base_model: str,
    adapter_path: Path,
    output_dir: Path,
    model_name: str = "cassette-adapter",
    skip_gguf: bool = False,
    skip_ollama: bool = False,
) -> dict[str, str]:
    """Full export pipeline: merge → convert → register."""
    result: dict[str, str] = {}

    # 1. Merge
    merged_path = output_dir / "merged"
    merge_adapter(base_model, adapter_path, merged_path)
    result["merged_path"] = str(merged_path)

    if skip_gguf:
        result["status"] = "merged_only"
        return result

    # 2. Convert to GGUF
    gguf_path = output_dir / f"{model_name}.gguf"
    try:
        convert_to_gguf(merged_path, gguf_path)
        result["gguf_path"] = str(gguf_path)
    except FileNotFoundError as exc:
        print(f"  Warning: {exc}")
        print("  Skipping GGUF conversion. Merged model is available at:", merged_path)
        result["status"] = "merged_only"
        result["note"] = str(exc)
        return result

    if skip_ollama:
        result["status"] = "gguf_ready"
        return result

    # 3. Register with ollama
    try:
        register_with_ollama(gguf_path, model_name)
        result["ollama_model"] = model_name
        result["status"] = "registered"
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"  Warning: Ollama registration failed: {exc}")
        result["status"] = "gguf_ready"
        result["note"] = str(exc)

    return result
