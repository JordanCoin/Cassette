"""Tests for the llama.cpp HTTP provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import httpx
import pytest

from libs.adapters.llama_cpp_http_provider import LlamaCppHttpProvider

MESSAGES = [{"role": "user", "content": "hello"}]

MOCK_RESPONSE: dict[str, Any] = {
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hi there!"},
            "finish_reason": "stop",
        }
    ]
}


_FAKE_REQUEST = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")


def _mock_success(*args: Any, **kwargs: Any) -> httpx.Response:
    return httpx.Response(200, json=MOCK_RESPONSE, request=_FAKE_REQUEST)


def _mock_connection_error(*args: Any, **kwargs: Any) -> Any:
    raise httpx.ConnectError("Connection refused")


def _mock_server_error(*args: Any, **kwargs: Any) -> httpx.Response:
    return httpx.Response(500, text="Internal Server Error", request=_FAKE_REQUEST)


def _mock_empty_choices(*args: Any, **kwargs: Any) -> httpx.Response:
    return httpx.Response(200, json={"choices": []}, request=_FAKE_REQUEST)


class TestLlamaCppHttpProvider:
    def test_name(self) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080")
        assert provider.name == "llama_cpp_http"

    @patch("libs.adapters.llama_cpp_http_provider.httpx.post", side_effect=_mock_success)
    def test_successful_completion(self, mock_post: Any) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080")
        result = provider.complete(MESSAGES)
        assert result == "Hi there!"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "/v1/chat/completions" in call_kwargs[0][0]

    @patch("libs.adapters.llama_cpp_http_provider.httpx.post", side_effect=_mock_success)
    def test_passes_messages(self, mock_post: Any) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080")
        provider.complete(MESSAGES)
        payload = mock_post.call_args[1]["json"]
        assert payload["messages"] == MESSAGES

    @patch("libs.adapters.llama_cpp_http_provider.httpx.post", side_effect=_mock_connection_error)
    def test_connection_failure_raises(self, mock_post: Any) -> None:
        provider = LlamaCppHttpProvider("http://localhost:9999")
        with pytest.raises(ConnectionError, match="Cannot reach llama.cpp server"):
            provider.complete(MESSAGES)

    @patch("libs.adapters.llama_cpp_http_provider.httpx.post", side_effect=_mock_server_error)
    def test_server_error_raises(self, mock_post: Any) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080")
        with pytest.raises(RuntimeError, match="llama.cpp server returned 500"):
            provider.complete(MESSAGES)

    @patch("libs.adapters.llama_cpp_http_provider.httpx.post", side_effect=_mock_empty_choices)
    def test_empty_choices_raises(self, mock_post: Any) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080")
        with pytest.raises(RuntimeError, match="returned no choices"):
            provider.complete(MESSAGES)

    @patch("libs.adapters.llama_cpp_http_provider.httpx.post", side_effect=_mock_success)
    def test_strips_trailing_slash_from_base_url(self, mock_post: Any) -> None:
        provider = LlamaCppHttpProvider("http://localhost:8080/")
        provider.complete(MESSAGES)
        url = mock_post.call_args[0][0]
        assert "//" not in url.replace("http://", "")
