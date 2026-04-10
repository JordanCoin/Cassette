"""HTTP-based URL fetch adapter.

Fetches a URL and returns truncated text content.
No browser automation — plain HTTP only.
"""

from __future__ import annotations

import re
from typing import Any

import httpx

MAX_CONTENT_LEN = 5000


def _extract_text(html: str) -> str:
    """Minimal HTML-to-text: strip tags, collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CONTENT_LEN]


class HttpFetchAdapter:
    """Fetches a URL over HTTP and returns normalized content."""

    def fetch(self, url: str) -> dict[str, Any]:
        try:
            resp = httpx.get(url, timeout=15.0, follow_redirects=True)
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise ConnectionError(f"Cannot reach {url}: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Fetch returned {exc.response.status_code} for {url}"
            ) from exc

        return {
            "url": str(resp.url),
            "status": resp.status_code,
            "content": _extract_text(resp.text),
        }
