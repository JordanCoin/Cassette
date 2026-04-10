"""Provider registry — resolves a provider name to an implementation."""

from __future__ import annotations

from collections.abc import Callable

from libs.adapters.llama_cpp_http_provider import LlamaCppHttpProvider
from libs.adapters.mock_provider import MockProvider
from libs.core.ports import ModelProvider
from libs.core.settings import get_llama_cpp_url, get_model_name, get_provider_timeout

_PROVIDERS: dict[str, Callable[[], ModelProvider]] = {
    "mock": MockProvider,
    "llama_cpp_http": lambda: LlamaCppHttpProvider(
        get_llama_cpp_url(), timeout=get_provider_timeout(), model=get_model_name()
    ),
}


def resolve_provider(name: str) -> ModelProvider:
    """Return a provider instance for the given name, or raise ValueError."""
    factory = _PROVIDERS.get(name)
    if factory is None:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown provider: {name!r}. Available: {available}")
    return factory()
