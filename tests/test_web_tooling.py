"""Tests for web tooling adapters and gather_sources stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from libs.adapters.http_fetch import HttpFetchAdapter, _extract_text
from libs.adapters.http_search import HttpSearchAdapter
from libs.adapters.jsonl_store import JsonlStore
from services.orchestrator.runner import run_stage
from services.orchestrator.stages import GatherSourcesStage

_FAKE_REQ = httpx.Request("GET", "http://test")


# -- Search adapter tests --


SEARCH_RESPONSE = {
    "results": [
        {"title": "Result 1", "url": "http://a.com", "content": "Snippet 1"},
        {"title": "Result 2", "url": "http://b.com", "content": "Snippet 2"},
        {"title": "Result 3", "url": "http://c.com", "content": "Snippet 3"},
    ]
}


def _mock_search_success(*args: Any, **kwargs: Any) -> httpx.Response:
    return httpx.Response(200, json=SEARCH_RESPONSE, request=_FAKE_REQ)


def _mock_search_fail(*args: Any, **kwargs: Any) -> Any:
    raise httpx.ConnectError("refused")


class TestHttpSearchAdapter:
    @patch("libs.adapters.http_search.httpx.get", side_effect=_mock_search_success)
    def test_returns_normalized_results(self, mock_get: Any) -> None:
        adapter = HttpSearchAdapter("http://localhost:8888")
        results = adapter.search("test query")
        assert len(results) == 3
        assert results[0]["title"] == "Result 1"
        assert results[0]["url"] == "http://a.com"
        assert results[0]["snippet"] == "Snippet 1"

    @patch("libs.adapters.http_search.httpx.get", side_effect=_mock_search_success)
    def test_passes_query_param(self, mock_get: Any) -> None:
        adapter = HttpSearchAdapter("http://localhost:8888")
        adapter.search("test query")
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["q"] == "test query"

    @patch("libs.adapters.http_search.httpx.get", side_effect=_mock_search_fail)
    def test_connection_error(self, mock_get: Any) -> None:
        adapter = HttpSearchAdapter("http://localhost:9999")
        with pytest.raises(ConnectionError, match="Search service unreachable"):
            adapter.search("test")


# -- Fetch adapter tests --


def _mock_fetch_success(*args: Any, **kwargs: Any) -> httpx.Response:
    html = "<html><body><p>Hello world</p></body></html>"
    return httpx.Response(200, text=html, request=_FAKE_REQ)


def _mock_fetch_fail(*args: Any, **kwargs: Any) -> Any:
    raise httpx.ConnectError("refused")


class TestHttpFetchAdapter:
    @patch("libs.adapters.http_fetch.httpx.get", side_effect=_mock_fetch_success)
    def test_returns_normalized_content(self, mock_get: Any) -> None:
        adapter = HttpFetchAdapter()
        result = adapter.fetch("http://example.com")
        assert result["status"] == 200
        assert "Hello world" in result["content"]

    @patch("libs.adapters.http_fetch.httpx.get", side_effect=_mock_fetch_fail)
    def test_connection_error(self, mock_get: Any) -> None:
        adapter = HttpFetchAdapter()
        with pytest.raises(ConnectionError, match="Cannot reach"):
            adapter.fetch("http://bad.example.com")

    def test_extract_text_strips_tags(self) -> None:
        assert "hello" in _extract_text("<b>hello</b>")

    def test_extract_text_truncates(self) -> None:
        long_html = "<p>" + "x" * 10000 + "</p>"
        result = _extract_text(long_html)
        assert len(result) <= 5000


# -- Mock adapters for stage tests --


class MockSearch:
    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or [
            {"title": "A", "url": "http://a.com", "snippet": "snip a"},
            {"title": "B", "url": "http://b.com", "snippet": "snip b"},
        ]

    def search(self, query: str) -> list[dict[str, Any]]:
        return self._results


class MockFetch:
    def __init__(self, fail_urls: set[str] | None = None) -> None:
        self._fail_urls = fail_urls or set()

    def fetch(self, url: str) -> dict[str, Any]:
        if url in self._fail_urls:
            raise ConnectionError(f"Cannot reach {url}")
        return {"url": url, "status": 200, "content": f"Content from {url}"}


class FailingSearch:
    def search(self, query: str) -> list[dict[str, Any]]:
        raise ConnectionError("Search failed")


# -- GatherSourcesStage tests --


class TestGatherSourcesStage:
    def test_successful_gather(self) -> None:
        stage = GatherSourcesStage(MockSearch(), MockFetch())
        result = stage.run({"query": "test"})
        assert result["query"] == "test"
        assert result["source_count"] == 2
        assert len(result["sources"]) == 2
        assert result["sources"][0]["title"] == "A"
        assert "Content from" in result["sources"][0]["content"]

    def test_respects_max_results(self) -> None:
        stage = GatherSourcesStage(MockSearch(), MockFetch(), max_results=1)
        result = stage.run({"query": "test"})
        assert result["source_count"] == 1

    def test_missing_query_raises(self) -> None:
        stage = GatherSourcesStage(MockSearch(), MockFetch())
        with pytest.raises(ValueError, match="requires a 'query'"):
            stage.run({})

    def test_fetch_failure_captured_in_errors(self) -> None:
        stage = GatherSourcesStage(
            MockSearch(), MockFetch(fail_urls={"http://a.com"})
        )
        result = stage.run({"query": "test"})
        assert result["source_count"] == 1  # only b.com succeeded
        assert len(result["errors"]) == 1
        assert result["errors"][0]["url"] == "http://a.com"

    def test_search_failure_propagates(self) -> None:
        stage = GatherSourcesStage(FailingSearch(), MockFetch())
        with pytest.raises(ConnectionError):
            stage.run({"query": "test"})


# -- Integration with runner --


class TestGatherSourcesWithRunner:
    def test_stage_completes_task(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(MockSearch(), MockFetch())
        result = run_stage(stage, {"query": "test"}, task_store=s, event_store=s)
        assert result["status"] == "completed"
        assert result["result"]["source_count"] == 2

    def test_stage_failure_fails_task(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(FailingSearch(), MockFetch())
        result = run_stage(stage, {"query": "test"}, task_store=s, event_store=s)
        assert result["status"] == "failed"

    def test_emits_stage_events(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(MockSearch(), MockFetch())
        run_stage(stage, {"query": "test"}, task_store=s, event_store=s)
        events = s.get_latest_events(10)
        types = [e.type for e in events]
        assert "stage.started" in types
        assert "stage.completed" in types
        started = [e for e in events if e.type == "stage.started"]
        assert started[0].payload["stage"] == "gather_sources"


class TestStepLevelEvents:
    def _run_and_get_types(
        self, tmp_path: Path, search: Any = None, fetch: Any = None
    ) -> list[str]:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(search or MockSearch(), fetch or MockFetch())
        run_stage(stage, {"query": "test"}, task_store=s, event_store=s)
        return [e.type for e in s.get_latest_events(50)]

    def test_emits_search_started(self, tmp_path: Path) -> None:
        types = self._run_and_get_types(tmp_path)
        assert "search.started" in types

    def test_emits_search_completed(self, tmp_path: Path) -> None:
        types = self._run_and_get_types(tmp_path)
        assert "search.completed" in types

    def test_search_completed_has_result_count(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(MockSearch(), MockFetch())
        run_stage(stage, {"query": "test"}, task_store=s, event_store=s)
        completed = [e for e in s.get_latest_events(50) if e.type == "search.completed"]
        assert completed[0].payload["result_count"] == 2

    def test_emits_fetch_started_per_url(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(MockSearch(), MockFetch())
        run_stage(stage, {"query": "test"}, task_store=s, event_store=s)
        fetch_started = [e for e in s.get_latest_events(50) if e.type == "fetch.started"]
        assert len(fetch_started) == 2
        urls = {e.payload["url"] for e in fetch_started}
        assert "http://a.com" in urls
        assert "http://b.com" in urls

    def test_emits_fetch_completed_per_url(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(MockSearch(), MockFetch())
        run_stage(stage, {"query": "test"}, task_store=s, event_store=s)
        fetch_done = [e for e in s.get_latest_events(50) if e.type == "fetch.completed"]
        assert len(fetch_done) == 2

    def test_emits_fetch_failed_on_error(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(
            MockSearch(), MockFetch(fail_urls={"http://a.com"})
        )
        run_stage(stage, {"query": "test"}, task_store=s, event_store=s)
        fetch_failed = [e for e in s.get_latest_events(50) if e.type == "fetch.failed"]
        assert len(fetch_failed) == 1
        assert fetch_failed[0].payload["url"] == "http://a.com"
        assert "error" in fetch_failed[0].payload

    def test_emits_search_failed_on_error(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(FailingSearch(), MockFetch())
        run_stage(stage, {"query": "test"}, task_store=s, event_store=s)
        search_failed = [e for e in s.get_latest_events(50) if e.type == "search.failed"]
        assert len(search_failed) == 1
        assert search_failed[0].payload["query"] == "test"

    def test_all_events_share_trace_id(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(MockSearch(), MockFetch())
        run_stage(stage, {"query": "test"}, task_store=s, event_store=s)
        events = s.get_latest_events(50)
        trace_ids = {str(e.trace_id) for e in events}
        assert len(trace_ids) == 1

    def test_search_started_has_query(self, tmp_path: Path) -> None:
        s = JsonlStore(tmp_path)
        stage = GatherSourcesStage(MockSearch(), MockFetch())
        run_stage(stage, {"query": "my query"}, task_store=s, event_store=s)
        started = [e for e in s.get_latest_events(50) if e.type == "search.started"]
        assert started[0].payload["query"] == "my query"
