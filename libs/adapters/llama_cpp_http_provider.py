"""llama.cpp HTTP provider — calls an OpenAI-compatible local server."""

from __future__ import annotations

import httpx


class ProviderTimeoutError(ConnectionError):
    """Raised when a provider call times out."""


class ProviderResponseError(RuntimeError):
    """Raised when a provider returns an invalid or unexpected response."""


class LlamaCppHttpProvider:
    """Sends chat completions to a llama.cpp server over HTTP."""

    def __init__(self, base_url: str, timeout: float = 60.0, model: str = "default") -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._model = model

    @property
    def name(self) -> str:
        return "llama_cpp_http"

    def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> str:
        """Call the backend and return the assistant message content."""
        url = f"{self._base_url}/v1/chat/completions"
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
        }
        # Pass through optional params
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("max_tokens") is not None:
            payload["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("think") is not None:
            payload["think"] = kwargs["think"]
        try:
            resp = httpx.post(url, json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Timeout after {self._timeout}s calling {self._base_url}: {exc}"
            ) from exc
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderResponseError(
                f"llama.cpp server returned {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            ) from exc

        try:
            data = resp.json()
        except Exception as exc:
            raise ProviderResponseError(
                f"Invalid JSON response from {self._base_url}"
            ) from exc

        return self._extract_content(data)

    @staticmethod
    def _extract_content(data: dict[str, object]) -> str:
        """Extract assistant message content from various response shapes."""
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise ProviderResponseError(
                f"No choices in response. Keys: {list(data.keys())}"
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ProviderResponseError(f"Unexpected choice type: {type(choice)}")

        # Standard OpenAI shape: choices[0].message.content
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content is not None:
                return str(content)

        # Some backends use choices[0].text
        text = choice.get("text")
        if text is not None:
            return str(text)

        raise ProviderResponseError(
            f"Cannot extract content from choice. Keys: {list(choice.keys())}"
        )

    def health_check(self) -> dict[str, object]:
        """Probe the backend for basic reachability."""
        # Try /v1/models first (standard), fall back to /health or root
        for path in ["/v1/models", "/health", "/"]:
            url = f"{self._base_url}{path}"
            try:
                resp = httpx.get(url, timeout=5.0)
                if resp.status_code < 500:
                    return {
                        "reachable": True,
                        "status": resp.status_code,
                        "url": self._base_url,
                        "probe": path,
                    }
            except httpx.TimeoutException:
                return {"reachable": False, "error": "timeout", "url": self._base_url}
            except httpx.ConnectError:
                return {
                    "reachable": False,
                    "error": "connection_refused",
                    "url": self._base_url,
                }
        return {"reachable": False, "error": "all_probes_failed", "url": self._base_url}
