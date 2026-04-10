"""llama.cpp HTTP provider — calls an OpenAI-compatible local server."""

from __future__ import annotations

import httpx


class LlamaCppHttpProvider:
    """Sends chat completions to a llama.cpp server over HTTP."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "llama_cpp_http"

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Call the backend and return the assistant message content."""
        url = f"{self._base_url}/v1/chat/completions"
        payload = {
            "messages": messages,
        }
        try:
            resp = httpx.post(url, json=payload, timeout=60.0)
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot reach llama.cpp server at {self._base_url}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"llama.cpp server returned {exc.response.status_code}: {exc.response.text}"
            ) from exc

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("llama.cpp server returned no choices")
        return str(choices[0]["message"]["content"])
