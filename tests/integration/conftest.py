"""Shared fixtures and skip logic for integration tests."""

from __future__ import annotations

import os
import platform

import pytest

SKIP_INTEGRATION = not os.environ.get("CASSETTE_INTEGRATION")
SKIP_REASON = "Set CASSETTE_INTEGRATION=1 with a running model server"


def detect_hardware() -> dict[str, object]:
    """Detect the current hardware surface."""
    info: dict[str, object] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # GPU detection
    try:
        import torch  # type: ignore[import-not-found]

        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 1
            )
        info["mps"] = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        info["cuda"] = False
        info["mps"] = False

    # Classify surface
    if info.get("cuda"):
        info["surface"] = "cloud_gpu"
    elif info.get("mps"):
        info["surface"] = "apple_silicon"
    elif info["machine"] == "x86_64" and info["platform"] == "Darwin":
        info["surface"] = "intel_mac"
    elif info["platform"] == "Linux":
        info["surface"] = "linux_pc"
    else:
        info["surface"] = "unknown"

    return info


@pytest.fixture(scope="session")
def hardware() -> dict[str, object]:
    return detect_hardware()
