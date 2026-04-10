"""llama.cpp HTTP provider — calls an OpenAI-compatible local server."""

from __future__ import annotations

import httpx


class ProviderTimeoutError(ConnectionError):
    """Raised when a provider call times out."""


class ProviderResponseError(RuntimeError):
    """Raised when a provider returns an invalid or unexpected response."""


class LlamaCppHttpProvider:
    """Sends chat completions to a llama.cpp server over HTTP."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "llama_cpp_http"

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Call the backend and return the assistant message content."""
        url = f"{self._base_url}/v1/chat/completions"
        payload = {"messages": messages}
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

        choices = data.get("choices", [])
        if not choices:
            raise ProviderResponseError("llama.cpp server returned no choices")
        return str(choices[0]["message"]["content"])

    def health_check(self) -> dict[str, object]:
        """Probe the backend for basic reachability."""
        url = f"{self._base_url}/v1/models"
        try:
            resp = httpx.get(url, timeout=5.0)
            resp.raise_for_status()
            return {"reachable": True, "status": resp.status_code, "url": self._base_url}
        except httpx.TimeoutException:
            return {"reachable": False, "error": "timeout", "url": self._base_url}
        except httpx.ConnectError:
            return {"reachable": False, "error": "connection_refused", "url": self._base_url}
        except httpx.HTTPStatusError as exc:
            return {
                "reachable": True,
                "error": f"status_{exc.response.status_code}",
                "url": self._base_url,
            }
