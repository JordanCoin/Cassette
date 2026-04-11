"""Provider registry — resolves a provider name to an implementation."""

from __future__ import annotations

from collections.abc import Callable

from libs.adapters.llama_cpp_http_provider import LlamaCppHttpProvider
from libs.adapters.mock_provider import MockProvider
from libs.core.config import get_float, get_str
from libs.core.ports import ModelProvider

_PROVIDERS: dict[str, Callable[[], ModelProvider]] = {
    "mock": MockProvider,
    "llama_cpp_http": lambda: LlamaCppHttpProvider(
        get_str("provider_url"),
        timeout=get_float("provider_timeout"),
        model=get_str("model"),
        api_key=get_str("provider_api_key"),
    ),
}


def resolve_provider(name: str) -> ModelProvider:
    """Return a provider instance for the given name, or raise ValueError."""
    factory = _PROVIDERS.get(name)
    if factory is None:
        available = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown provider: {name!r}. Available: {available}")
    return factory()
