"""HTTP-based web search adapter.

Calls an OpenSearch-compatible JSON endpoint (e.g. SearXNG).
Designed for local dev with a configurable base URL.
"""

from __future__ import annotations

from typing import Any

import httpx


class HttpSearchAdapter:
    """Calls a search API and normalizes results."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def search(self, query: str) -> list[dict[str, Any]]:
        url = f"{self._base_url}/search"
        try:
            resp = httpx.get(
                url, params={"q": query, "format": "json"}, timeout=15.0
            )
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Search service unreachable: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Search service returned {exc.response.status_code}"
            ) from exc

        data = resp.json()
        raw_results = data.get("results", [])
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in raw_results[:10]
        ]
