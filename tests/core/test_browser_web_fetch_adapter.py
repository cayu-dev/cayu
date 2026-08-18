from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from cayu import (
    BROWSER_FETCH_PLAYWRIGHT_VERSION,
    BROWSER_FETCH_PROTOCOL_VERSION,
    BROWSER_FETCH_WORKER_VERSION,
    DEFAULT_BROWSER_FETCH_MAX_DOM_NODES,
    BrowserWebFetchAdapter,
    ExecCommand,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ToolContext,
    WebFetchTool,
)
from cayu.environments.admission import ExecutionAdmissionCandidate
from cayu.runners import ExecResult, RunnerExecutionError, RunnerUnavailableError
from cayu.tools import MAX_BROWSER_FETCH_MAX_DOM_NODES, MAX_BROWSER_FETCH_MAX_REQUESTS
from cayu.tools import _browser_guest as guest


def _candidate(
    *,
    omit: str | None = None,
    stale: str | None = None,
) -> ExecutionAdmissionCandidate:
    now = datetime.now(UTC)
    claims = []
    for capability in (
        "deny_by_default_network",
        "brokered_egress",
        "confirmed_cancellation",
        "confirmed_cleanup",
    ):
        if capability == omit:
            continue
        if capability == stale:
            claims.append(
                ExecutionCapabilityClaim.live_verified(
                    capability,
                    observation=(
                        "denied" if capability == "deny_by_default_network" else "reachable"
                    ),
                    observed_at=now - timedelta(minutes=10),
                    valid_until=now - timedelta(minutes=5),
                )
            )
        else:
            claims.append(ExecutionCapabilityClaim.available(capability))
    return ExecutionAdmissionCandidate(
        candidate="browser-test-runner",
        evidence=ExecutionCapabilityEvidence(
            subject="browser-test-runner",
            claims=tuple(claims),
        ),
    )


def _success_payload(**overrides: Any) -> str:
    payload: dict[str, Any] = {
        "protocol_version": BROWSER_FETCH_PROTOCOL_VERSION,
        "worker_version": BROWSER_FETCH_WORKER_VERSION,
        "playwright_version": BROWSER_FETCH_PLAYWRIGHT_VERSION,
        "kind": "success",
        "requested_url": "https://example.com/",
        "final_url": "https://example.com/guide",
        "title": "Rendered guide",
        "representation": "text",
        "content": "JavaScript-rendered content",
        "redirects": [
            {
                "status_code": 302,
                "from_url": "https://example.com/",
                "to_url": "https://example.com/guide",
            }
        ],
        "truncation_reasons": [],
        "response_bytes": 512,
        "request_count": 3,
    }
    payload.update(overrides)
    return json.dumps(payload)


class _FakeRunner:
    def __init__(
        self,
        result: ExecResult | None = None,
        *,
        candidate: ExecutionAdmissionCandidate | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.result = result or ExecResult(stdout=_success_payload())
        self.candidate = candidate or _candidate()
        self.error = error
        self.calls: list[tuple[ExecCommand, dict[str, Any]]] = []

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None:
        return self.candidate

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        self.calls.append((command, kwargs))
        if self.error is not None:
            raise self.error
        return self.result


class _ExecOnlyRunner:
    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        del command, kwargs
        raise AssertionError("An unverified runner must not be dispatched.")


def _run(
    runner: Any,
    *,
    url: str = "https://example.com/",
    max_response_bytes: int = 2048,
    max_content_bytes: int = 1024,
    timeout_seconds: float = 2.1,
    max_redirects: int = 2,
    max_requests: int = 128,
    max_dom_nodes: int = DEFAULT_BROWSER_FETCH_MAX_DOM_NODES,
):
    return asyncio.run(
        WebFetchTool(
            adapter=BrowserWebFetchAdapter(
                max_requests=max_requests,
                max_dom_nodes=max_dom_nodes,
            ),
            max_response_bytes=max_response_bytes,
            max_content_bytes=max_content_bytes,
            timeout_seconds=timeout_seconds,
            max_redirects=max_redirects,
        ).run(
            ToolContext(session_id="sess_browser", runner=runner),
            {"url": url},
        )
    )


def test_browser_adapter_preserves_web_fetch_contract_and_dispatches_closed_request() -> None:
    runner = _FakeRunner()

    result = _run(runner)

    assert result.is_error is False
    assert result.structured == {
        "requested_url": "https://example.com/",
        "final_url": "https://example.com/guide",
        "title": "Rendered guide",
        "representation": "text",
        "content": "JavaScript-rendered content",
        "redirects": [
            {
                "status_code": 302,
                "from_url": "https://example.com/",
                "to_url": "https://example.com/guide",
            }
        ],
        "truncated": False,
        "truncation_reasons": [],
    }
    assert "<untrusted_web_content>" in result.content
    assert len(runner.calls) == 1
    command, kwargs = runner.calls[0]
    assert command.argv == [
        "/usr/local/bin/python",
        "-I",
        "/opt/cayu-browser/worker.py",
    ]
    request = json.loads(kwargs["stdin"])
    assert request == {
        "expected_playwright_version": BROWSER_FETCH_PLAYWRIGHT_VERSION,
        "limits": {
            "max_content_bytes": 1024,
            "max_dom_nodes": DEFAULT_BROWSER_FETCH_MAX_DOM_NODES,
            "max_redirects": 2,
            "max_requests": 128,
            "max_response_bytes": 2048,
            "timeout_seconds": 2.1,
        },
        "operation": "fetch",
        "protocol_version": BROWSER_FETCH_PROTOCOL_VERSION,
        "url": "https://example.com/",
        "worker_version": BROWSER_FETCH_WORKER_VERSION,
    }
    assert kwargs["timeout_s"] == 3
    assert kwargs["output_limit_bytes"] == (2 * (1024 + 512 + 4 * (2 + 2 * 2) * 8192) + 64 * 1024)


def test_browser_adapter_projects_accessibility_evidence_without_changing_url_evidence() -> None:
    runner = _FakeRunner(
        ExecResult(
            stdout=_success_payload(
                representation="accessibility",
                content='- navigation "Primary"\n  - link "Guide"',
            )
        )
    )

    result = _run(runner)

    assert result.is_error is False
    assert result.structured["representation"] == "accessibility"
    assert result.structured["content"] == '- navigation "Primary"\n  - link "Guide"'
    assert result.structured["requested_url"] == "https://example.com/"
    assert result.structured["final_url"] == "https://example.com/guide"
    assert result.content.startswith(
        "Fetched web content:\nRepresentation: accessibility\nTruncated: false\n\n"
    )


def test_browser_adapter_bounds_non_ascii_request_and_result_framing() -> None:
    runner = _FakeRunner()
    requested_url = "https://example.com/" + "é" * 8_000

    _run(runner, url=requested_url)

    _command, kwargs = runner.calls[0]
    stdin = kwargs["stdin"]
    assert len(stdin.encode("utf-8")) < 64 * 1024
    assert json.loads(stdin)["url"] == requested_url
    assert kwargs["output_limit_bytes"] >= 4 * (2 + 2 * 2) * 8192


@pytest.mark.parametrize(
    ("runner", "error"),
    [
        (None, "incompatible_runner"),
        (_ExecOnlyRunner(), "incompatible_runner"),
        (_FakeRunner(candidate=_candidate(omit="brokered_egress")), "capability_refused"),
        (
            _FakeRunner(candidate=_candidate(stale="deny_by_default_network")),
            "capability_refused",
        ),
    ],
)
def test_browser_adapter_refuses_missing_or_unverified_runner_before_dispatch(
    runner: Any,
    error: str,
) -> None:
    result = _run(runner)

    assert result.is_error is True
    assert result.structured == {"error": error}
    if isinstance(runner, _FakeRunner):
        assert runner.calls == []


@pytest.mark.parametrize(
    ("execution", "error"),
    [
        (ExecResult(exit_code=2), "browser_unavailable"),
        (ExecResult(exit_code=127), "browser_unavailable"),
        (ExecResult(exit_code=3), "browser_crash"),
        (ExecResult(timed_out=True), "timeout"),
        (ExecResult(cancelled=True), "browser_crash"),
        (ExecResult(stdout="{}", stdout_truncated=True), "oversized_response"),
    ],
)
def test_browser_adapter_classifies_bounded_runner_outcomes(
    execution: ExecResult,
    error: str,
) -> None:
    result = _run(_FakeRunner(execution))

    assert result.structured == {"error": error}


@pytest.mark.parametrize(
    ("runner_error", "error"),
    [
        (
            RunnerUnavailableError(
                "private unavailable detail",
                diagnostic={"adapter": "docker", "status": "unavailable"},
            ),
            "browser_unavailable",
        ),
        (
            RunnerExecutionError(diagnostic={"adapter": "docker", "error_type": "RuntimeError"}),
            "browser_crash",
        ),
        (TimeoutError("private timeout detail"), "timeout"),
    ],
)
def test_browser_adapter_sanitizes_known_runner_failures(
    runner_error: BaseException,
    error: str,
) -> None:
    result = _run(_FakeRunner(error=runner_error))

    assert result.structured == {"error": error}
    assert "private" not in result.content


def test_browser_adapter_sanitizes_unexpected_runner_failure() -> None:
    sensitive = "runner-secret-canary"

    result = _run(_FakeRunner(error=RuntimeError(sensitive)))

    assert result.structured == {"error": "browser_crash"}
    assert sensitive not in result.content


@pytest.mark.parametrize(
    ("stdout", "error"),
    [
        ("not-json", "malformed_browser_result"),
        (json.dumps([]), "malformed_browser_result"),
        (_success_payload(protocol_version="future"), "incompatible_browser"),
        (
            _success_payload(
                protocol_version="cayu.browser-fetch.v1",
                worker_version="1",
            ),
            "incompatible_browser",
        ),
        (_success_payload(worker_version="4"), "incompatible_browser"),
        (_success_payload(playwright_version="1.63.0"), "incompatible_browser"),
        (_success_payload(requested_url="https://other.example/"), "malformed_browser_result"),
        (_success_payload(response_bytes=2049), "malformed_browser_result"),
        (_success_payload(request_count=129), "malformed_browser_result"),
        (_success_payload(representation="dom"), "malformed_browser_result"),
        (_success_payload(content="x" * 1025), "malformed_browser_result"),
        (_success_payload(content="unsafe\x00content"), "malformed_browser_result"),
        (_success_payload(content="unsafe\ud800content"), "malformed_browser_result"),
        (_success_payload(title="unsafe\ud800title"), "malformed_browser_result"),
        ("\ud800", "malformed_browser_result"),
        (_success_payload(unexpected=True), "malformed_browser_result"),
    ],
)
def test_browser_adapter_rejects_malformed_or_incompatible_worker_result(
    stdout: str,
    error: str,
) -> None:
    result = _run(_FakeRunner(ExecResult(stdout=stdout)))

    assert result.structured == {"error": error}


def test_browser_adapter_requires_the_worker_to_declare_its_representation() -> None:
    payload = json.loads(_success_payload())
    del payload["representation"]

    result = _run(_FakeRunner(ExecResult(stdout=json.dumps(payload))))

    assert result.structured == {"error": "malformed_browser_result"}


def test_browser_adapter_rejects_result_above_configured_request_limit() -> None:
    result = _run(
        _FakeRunner(ExecResult(stdout=_success_payload(request_count=4))),
        max_requests=3,
    )

    assert result.structured == {"error": "malformed_browser_result"}


@pytest.mark.parametrize(
    ("code", "status_code", "expected"),
    [
        ("destination_denied", None, {"error": "destination_denied"}),
        ("redirect_denied", None, {"error": "redirect_denied"}),
        ("cleanup_failed", None, {"error": "cleanup_failed"}),
        ("http_status", 429, {"error": "http_status", "status_code": 429}),
    ],
)
def test_browser_adapter_projects_versioned_worker_failures(
    code: str,
    status_code: int | None,
    expected: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "protocol_version": BROWSER_FETCH_PROTOCOL_VERSION,
        "worker_version": BROWSER_FETCH_WORKER_VERSION,
        "playwright_version": BROWSER_FETCH_PLAYWRIGHT_VERSION,
        "kind": "error",
        "error": code,
    }
    if status_code is not None:
        payload["status_code"] = status_code

    result = _run(_FakeRunner(ExecResult(stdout=json.dumps(payload))))

    assert result.structured == expected


def test_browser_adapter_configuration_is_bounded() -> None:
    with pytest.raises(ValueError, match="max_requests"):
        BrowserWebFetchAdapter(max_requests=0)
    with pytest.raises(ValueError, match="max_requests"):
        BrowserWebFetchAdapter(max_requests=MAX_BROWSER_FETCH_MAX_REQUESTS + 1)
    with pytest.raises(ValueError, match="max_dom_nodes"):
        BrowserWebFetchAdapter(max_dom_nodes=0)
    with pytest.raises(TypeError, match="max_dom_nodes"):
        BrowserWebFetchAdapter(max_dom_nodes=True)
    with pytest.raises(ValueError, match="max_dom_nodes"):
        BrowserWebFetchAdapter(max_dom_nodes=MAX_BROWSER_FETCH_MAX_DOM_NODES + 1)
    with pytest.raises(TypeError, match="worker_command"):
        BrowserWebFetchAdapter(worker_command="python")
    with pytest.raises(ValueError, match="worker_command"):
        BrowserWebFetchAdapter(worker_command=[])


def test_browser_adapter_propagates_caller_cancellation_without_fallback() -> None:
    started = asyncio.Event()

    class _BlockingRunner(_FakeRunner):
        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            self.calls.append((command, kwargs))
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def run() -> _BlockingRunner:
        runner = _BlockingRunner()
        operation = asyncio.create_task(
            WebFetchTool(adapter=BrowserWebFetchAdapter()).run(
                ToolContext(session_id="sess_cancel", runner=runner),
                {"url": "https://example.com/"},
            )
        )
        await started.wait()
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        return runner

    runner = asyncio.run(run())

    assert len(runner.calls) == 1


def test_guest_request_framing_and_text_bounds_are_closed() -> None:
    request = guest._request_from_json(
        {
            "protocol_version": guest.PROTOCOL_VERSION,
            "worker_version": guest.WORKER_VERSION,
            "expected_playwright_version": guest.PLAYWRIGHT_VERSION,
            "operation": "fetch",
            "url": "https://example.com/",
            "limits": {
                "max_response_bytes": 100,
                "max_content_bytes": 7,
                "timeout_seconds": 1.5,
                "max_redirects": 1,
                "max_requests": 4,
                "max_dom_nodes": 100,
            },
        }
    )
    content, truncated = guest._normalized_text(
        " hello\x00   world ",
        request.limits.max_content_bytes,
        preserve_lines=True,
    )

    assert request.url == "https://example.com/"
    assert content == "hello w"
    assert truncated is True
    lines, lines_truncated = guest._normalized_text(
        "first\n  second\ud800 line",
        64,
        preserve_lines=True,
    )
    assert lines == "first\nsecond\ufffd line"
    assert lines_truncated is False
    accessibility, accessibility_truncated = guest._normalized_accessibility_text(
        "- navigation\n\t- link  Guide\x00\n" + (" " * 80) + "- button Save",
        256,
    )
    assert accessibility == "- navigation\n - link Guide\n" + (" " * 64) + "- button Save"
    assert accessibility_truncated is False
    bounded_accessibility, bounded_accessibility_truncated = guest._normalized_accessibility_text(
        "\u00e9" * 8, 7
    )
    assert bounded_accessibility == "\u00e9\u00e9\u00e9"
    assert len(bounded_accessibility.encode("utf-8")) <= 7
    assert bounded_accessibility_truncated is True
    with pytest.raises(guest._GuestFailure):
        guest._request_from_json(
            {
                "protocol_version": guest.PROTOCOL_VERSION,
                "worker_version": guest.WORKER_VERSION,
                "expected_playwright_version": guest.PLAYWRIGHT_VERSION,
                "operation": "fetch",
                "url": "https://example.com/",
                "limits": {},
                "arbitrary_browser_command": "evaluate",
            }
        )


def _projection_cdp(
    extracted: dict[str, object],
    *,
    final_loader_id: str = "loader-1",
) -> Mock:
    def frame_tree(loader_id: str) -> dict[str, object]:
        return {
            "frameTree": {
                "frame": {
                    "id": "frame-1",
                    "loaderId": loader_id,
                    "url": "https://example.com/",
                    "mimeType": "text/html",
                }
            }
        }

    projected = {"node_count": 1, **extracted}
    cdp = Mock()
    cdp.send = AsyncMock(
        side_effect=[
            frame_tree("loader-1"),
            {"executionContextId": 7},
            {"result": {"type": "object", "value": projected}},
            frame_tree(final_loader_id),
        ]
    )
    return cdp


def test_guest_keeps_readable_text_for_pages_without_semantic_structure() -> None:
    async def exercise() -> None:
        page = Mock()
        page.main_frame = page
        page.url = "https://example.com/"
        page.child_frames = []
        cdp = _projection_cdp(
            {
                "text": "Rendered article body",
                "truncated": False,
                "title": "Article",
                "title_truncated": False,
                "node_limit_exceeded": False,
                "semantic_structure": False,
            }
        )
        page.locator = Mock()
        request = guest._Request(
            url="https://example.com/",
            limits=guest._Limits(
                max_response_bytes=1024,
                max_content_bytes=128,
                timeout_seconds=1.0,
                max_redirects=1,
                max_requests=4,
                max_dom_nodes=100,
            ),
        )

        result = await guest._extract_page_representation(
            page,
            cdp,
            request,
            operation_timeout_ms=500,
        )

        assert result == ("Article", "text", "Rendered article body", ())
        page.locator.assert_not_called()
        assert cdp.send.await_args_list[0] == call("Page.getFrameTree")
        assert cdp.send.await_args_list[1] == call(
            "Page.createIsolatedWorld",
            {
                "frameId": "frame-1",
                "worldName": guest._BROWSER_INSPECTION_WORLD,
                "grantUniveralAccess": False,
            },
        )
        runtime_call = cdp.send.await_args_list[2]
        assert runtime_call.args[0] == "Runtime.evaluate"
        assert runtime_call.args[1]["contextId"] == 7
        assert (
            'const limits = {"content":128,"nodes":100,"title":512};'
            in (runtime_call.args[1]["expression"])
        )
        assert cdp.send.await_args_list[3] == call("Page.getFrameTree")

    asyncio.run(exercise())


def test_guest_escalates_semantic_pages_to_bounded_accessibility_evidence() -> None:
    async def exercise() -> None:
        locator = Mock()
        bounded_snapshot = '- navigation "Primary"\n  - link   "Guide"\n  - button "Save"'
        locator.aria_snapshot = AsyncMock(
            side_effect=[
                bounded_snapshot,
                bounded_snapshot + '\n    - text "deeper evidence"',
            ]
        )
        page = Mock()
        page.main_frame = page
        page.url = "https://example.com/"
        page.child_frames = []
        cdp = _projection_cdp(
            {
                "text": "Guide Save",
                "truncated": False,
                "title": "Controls",
                "title_truncated": False,
                "node_limit_exceeded": False,
                "semantic_structure": True,
            }
        )
        page.locator.return_value = locator
        request = guest._Request(
            url="https://example.com/",
            limits=guest._Limits(
                max_response_bytes=1024,
                max_content_bytes=128,
                timeout_seconds=1.0,
                max_redirects=1,
                max_requests=4,
                max_dom_nodes=100,
            ),
        )

        result = await guest._extract_page_representation(
            page,
            cdp,
            request,
            operation_timeout_ms=500,
        )

        assert result == (
            "Controls",
            "accessibility",
            '- navigation "Primary"\n  - link "Guide"\n  - button "Save"',
            ("content",),
        )
        page.locator.assert_called_once_with("body")
        assert locator.aria_snapshot.await_args_list == [
            call(
                timeout=500,
                depth=guest._ACCESSIBILITY_SNAPSHOT_DEPTH,
                mode="default",
                boxes=False,
            ),
            call(
                timeout=500,
                depth=guest._ACCESSIBILITY_SNAPSHOT_DEPTH + 1,
                mode="default",
                boxes=False,
            ),
        ]

    asyncio.run(exercise())


def test_guest_rejects_page_above_the_dom_node_ceiling_before_snapshot() -> None:
    async def exercise() -> None:
        page = Mock()
        page.main_frame = page
        page.url = "https://example.com/"
        page.child_frames = []
        cdp = _projection_cdp(
            {
                "text": "",
                "truncated": False,
                "title": "Too large",
                "title_truncated": False,
                "node_limit_exceeded": True,
                "semantic_structure": True,
            }
        )
        page.locator = Mock()
        request = guest._Request(
            url="https://example.com/",
            limits=guest._Limits(
                max_response_bytes=1024,
                max_content_bytes=128,
                timeout_seconds=1.0,
                max_redirects=1,
                max_requests=4,
                max_dom_nodes=100,
            ),
        )

        with pytest.raises(guest._GuestFailure) as captured:
            await guest._extract_page_representation(
                page,
                cdp,
                request,
                operation_timeout_ms=500,
            )

        assert captured.value.code == "oversized_response"
        page.locator.assert_not_called()

    asyncio.run(exercise())


def test_guest_rejects_a_document_replaced_during_accessibility_extraction() -> None:
    async def exercise() -> None:
        locator = Mock()
        locator.aria_snapshot = AsyncMock(return_value='- button "Save"')
        page = Mock()
        page.main_frame = page
        page.url = "https://example.com/"
        page.child_frames = []
        page.locator.return_value = locator
        cdp = _projection_cdp(
            {
                "text": "Save",
                "truncated": False,
                "title": "Controls",
                "title_truncated": False,
                "node_limit_exceeded": False,
                "semantic_structure": True,
            },
            final_loader_id="loader-2",
        )
        request = guest._Request(
            url="https://example.com/",
            limits=guest._Limits(
                max_response_bytes=1024,
                max_content_bytes=128,
                timeout_seconds=1.0,
                max_redirects=1,
                max_requests=4,
                max_dom_nodes=100,
            ),
        )

        with pytest.raises(guest._GuestFailure) as captured:
            await guest._extract_page_representation(
                page,
                cdp,
                request,
                operation_timeout_ms=500,
            )

        assert captured.value.code == "fetch_failed"

    asyncio.run(exercise())


def test_guest_aggregates_child_frames_under_one_node_and_content_budget() -> None:
    async def exercise() -> None:
        frame_tree = {
            "frameTree": {
                "frame": {
                    "id": "main-frame",
                    "loaderId": "main-loader",
                    "url": "https://example.com/",
                    "mimeType": "text/html",
                },
                "childFrames": [
                    {
                        "frame": {
                            "id": "child-frame",
                            "parentId": "main-frame",
                            "loaderId": "child-loader",
                            "url": "https://static.example.com/controls",
                            "mimeType": "text/html",
                        }
                    }
                ],
            }
        }
        main_projection = {
            "node_count": 4,
            "node_limit_exceeded": False,
            "semantic_structure": False,
            "text": "Parent overview",
            "title": "Parent",
            "title_truncated": False,
            "truncated": False,
        }
        child_projection = {
            "node_count": 7,
            "node_limit_exceeded": False,
            "semantic_structure": True,
            "text": "Deploy",
            "title": "Controls",
            "title_truncated": False,
            "truncated": False,
        }
        cdp = Mock()
        cdp.send = AsyncMock(
            side_effect=[
                frame_tree,
                {"backendNodeId": 42},
                {
                    "nodes": [
                        {
                            "backendDOMNodeId": 42,
                            "ignored": False,
                        }
                    ]
                },
                {"executionContextId": 7},
                {"result": {"type": "object", "value": main_projection}},
                {"executionContextId": 8},
                {"result": {"type": "object", "value": child_projection}},
                frame_tree,
                {"backendNodeId": 42},
                {
                    "nodes": [
                        {
                            "backendDOMNodeId": 42,
                            "ignored": False,
                        }
                    ]
                },
            ]
        )
        main_locator = Mock()
        main_locator.aria_snapshot = AsyncMock(return_value="- text: Parent overview\n- iframe")
        child_locator = Mock()
        child_locator.aria_snapshot = AsyncMock(
            return_value='- textbox "Target"\n- button "Deploy"'
        )
        child_frame = Mock()
        child_frame.url = "https://static.example.com/controls"
        child_frame.child_frames = []
        child_frame.locator.return_value = child_locator
        main_frame = Mock()
        main_frame.url = "https://example.com/"
        main_frame.child_frames = [child_frame]
        main_frame.locator.return_value = main_locator
        page = Mock()
        page.main_frame = main_frame
        request = guest._Request(
            url="https://example.com/",
            limits=guest._Limits(
                max_response_bytes=1024,
                max_content_bytes=1024,
                timeout_seconds=1.0,
                max_redirects=1,
                max_requests=4,
                max_dom_nodes=11,
            ),
        )

        result = await guest._extract_page_representation(
            page,
            cdp,
            request,
            operation_timeout_ms=500,
        )

        assert result == (
            "Parent",
            "accessibility",
            "[Main frame]\n"
            "URL: https://example.com/\n"
            "Title: Parent\n"
            "- text: Parent overview\n"
            "- iframe\n\n"
            "[Frame 1]\n"
            "URL: https://static.example.com/controls\n"
            "Parent frame: 0\n"
            "Title: Controls\n"
            '- textbox "Target"\n'
            '- button "Deploy"',
            (),
        )
        runtime_calls = [
            observed
            for observed in cdp.send.await_args_list
            if observed.args[0] == "Runtime.evaluate"
        ]
        assert '"nodes":11' in runtime_calls[0].args[1]["expression"]
        assert '"nodes":7' in runtime_calls[1].args[1]["expression"]

    asyncio.run(exercise())


def test_guest_excludes_ignored_frame_subtrees_from_evidence_membership() -> None:
    async def exercise() -> None:
        identities = (
            guest._FrameIdentity(
                frame_id="main",
                loader_id="main-loader",
                url="https://example.com/",
                mime_type="text/html",
                parent_index=None,
            ),
            guest._FrameIdentity(
                frame_id="hidden",
                loader_id="hidden-loader",
                url="https://static.example.com/hidden",
                mime_type="text/html",
                parent_index=0,
            ),
            guest._FrameIdentity(
                frame_id="hidden-descendant",
                loader_id="hidden-descendant-loader",
                url="https://static.example.com/hidden-child",
                mime_type="text/html",
                parent_index=1,
            ),
            guest._FrameIdentity(
                frame_id="visible",
                loader_id="visible-loader",
                url="https://static.example.com/visible",
                mime_type="text/html",
                parent_index=0,
            ),
        )
        cdp = Mock()
        cdp.send = AsyncMock(
            side_effect=[
                {"backendNodeId": 10},
                {"nodes": [{"backendDOMNodeId": 10, "ignored": True}]},
                {"backendNodeId": 11},
                {"nodes": [{"backendDOMNodeId": 11, "ignored": False}]},
                {"backendNodeId": 12},
                {"nodes": [{"backendDOMNodeId": 12, "ignored": False}]},
            ]
        )

        assert await guest._frame_evidence_membership(cdp, identities) == (
            True,
            False,
            False,
            True,
        )

    asyncio.run(exercise())


def test_guest_rejects_more_than_the_hard_frame_document_limit() -> None:
    async def exercise() -> None:
        main_frame = {
            "id": "main-frame",
            "loaderId": "main-loader",
            "url": "https://example.com/",
            "mimeType": "text/html",
        }
        child_frames = [
            {
                "frame": {
                    "id": f"child-{index}",
                    "parentId": "main-frame",
                    "loaderId": f"loader-{index}",
                    "url": "about:blank",
                    "mimeType": "text/html",
                }
            }
            for index in range(guest._MAX_FRAME_DOCUMENTS)
        ]
        cdp = Mock()
        cdp.send = AsyncMock(
            return_value={
                "frameTree": {
                    "frame": main_frame,
                    "childFrames": child_frames,
                }
            }
        )

        with pytest.raises(guest._GuestFailure) as captured:
            await guest._frame_identities(cdp)

        assert captured.value.code == "oversized_response"

    asyncio.run(exercise())


@pytest.mark.parametrize(
    ("total_seconds", "expected_milliseconds", "expected_cleanup_seconds"),
    [
        (30.0, 27_000, 3.0),
        (120.0, 115_000, 5.0),
        (0.1, 1, 0.1),
    ],
)
def test_guest_reserves_cleanup_time_inside_the_application_deadline(
    total_seconds: float,
    expected_milliseconds: int,
    expected_cleanup_seconds: float,
) -> None:
    assert guest._browser_time_budget(total_seconds) == (
        expected_milliseconds,
        expected_cleanup_seconds,
    )


@pytest.mark.parametrize(
    ("cleanup_seconds", "expected_profile_seconds"),
    [
        (5.0, 1.0),
        (3.0, 0.6),
        (0.1, 0.05),
        (0.01, 0.01),
    ],
)
def test_guest_reserves_profile_cleanup_inside_the_total_cleanup_budget(
    cleanup_seconds: float,
    expected_profile_seconds: float,
) -> None:
    assert guest._temporary_profile_cleanup_reserve_seconds(cleanup_seconds) == pytest.approx(
        expected_profile_seconds
    )


@pytest.mark.parametrize("blocked_stage", ["tasks", "context", "browser", "playwright"])
def test_guest_browser_cleanup_finishes_every_owner_before_republishing_cancellation(
    blocked_stage: str,
) -> None:
    async def exercise() -> None:
        blocked_stage_started = asyncio.Event()
        release_blocked_stage = asyncio.Event()
        calls: list[str] = []

        async def maybe_block(stage: str) -> None:
            calls.append(stage)
            if blocked_stage == stage:
                blocked_stage_started.set()
                await release_blocked_stage.wait()

        async def pending_navigation() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await maybe_block("tasks")
                raise

        class Context:
            async def close(self) -> None:
                await maybe_block("context")

            def remove_listener(self, event: str, listener: object) -> None:
                del listener
                calls.append(f"remove:{event}")

        class Browser:
            async def close(self) -> None:
                await maybe_block("browser")

        class Playwright:
            async def stop(self) -> None:
                await maybe_block("playwright")

        class Page:
            def remove_listener(self, event: str, listener: object) -> None:
                del listener
                calls.append(f"remove:{event}")

        navigation_task = asyncio.create_task(pending_navigation())
        cleanup_task = asyncio.create_task(
            guest._cleanup_browser_resources(
                violation_task=None,
                navigation_task=navigation_task,
                context=Context(),
                page=Page(),
                response_observed=object(),
                unexpected_page_observed=object(),
                browser=Browser(),
                playwright=Playwright(),
                timeout_seconds=2.0,
            )
        )
        supervisor = asyncio.create_task(
            guest._await_browser_cleanup_resisting_cancellation(cleanup_task)
        )
        await asyncio.wait_for(blocked_stage_started.wait(), timeout=1.0)
        supervisor.cancel("caller stopped browser fetch")
        release_blocked_stage.set()
        outcome = await asyncio.wait_for(supervisor, timeout=1.0)

        assert outcome.errors == ()
        assert outcome.cancellation is not None
        assert outcome.cancellation.args == ("caller stopped browser fetch",)
        assert navigation_task.done()
        assert calls == [
            "tasks",
            "context",
            "remove:response",
            "remove:page",
            "browser",
            "playwright",
        ]

    asyncio.run(exercise())


def test_guest_browser_cleanup_timeout_still_attempts_later_owners() -> None:
    async def exercise() -> None:
        calls: list[str] = []

        class Context:
            async def close(self) -> None:
                calls.append("context")
                await asyncio.Event().wait()

        class Browser:
            async def close(self) -> None:
                calls.append("browser")

        class Playwright:
            async def stop(self) -> None:
                calls.append("playwright")

        errors = await guest._cleanup_browser_resources(
            violation_task=None,
            navigation_task=None,
            context=Context(),
            page=None,
            response_observed=None,
            unexpected_page_observed=None,
            browser=Browser(),
            playwright=Playwright(),
            timeout_seconds=0.3,
        )

        assert len(errors) == 1
        assert isinstance(errors[0], TimeoutError)
        assert calls == ["context", "browser", "playwright"]

    asyncio.run(exercise())


def test_guest_browser_cleanup_uses_the_deadline_remaining_after_ca_setup() -> None:
    async def exercise() -> None:
        request = guest._Request(
            url="https://example.com/",
            limits=guest._Limits(
                max_response_bytes=100,
                max_content_bytes=100,
                timeout_seconds=0.5,
                max_redirects=1,
                max_requests=4,
                max_dom_nodes=100,
            ),
        )
        cleanup_calls: list[str] = []
        observed_budgets: list[tuple[float, float]] = []
        loop = asyncio.get_running_loop()
        ca_started_at: float | None = None

        class FakePlaywrightError(Exception):
            pass

        class FakePlaywrightTimeoutError(FakePlaywrightError):
            pass

        async def blocked_navigation(*_args: Any, **_kwargs: Any) -> None:
            await asyncio.Event().wait()

        async def blocked_cleanup(owner: str) -> None:
            cleanup_calls.append(owner)
            if owner == "context":
                await asyncio.Event().wait()

        def cleanup_stage(owner: str):
            async def run() -> None:
                await blocked_cleanup(owner)

            return run

        cdp = Mock()
        cdp.send = AsyncMock()
        page = Mock()
        page.url = request.url
        page.main_frame = object()
        page.goto = AsyncMock(side_effect=blocked_navigation)
        context = Mock()
        context.new_page = AsyncMock(return_value=page)
        context.route = AsyncMock()
        context.new_cdp_session = AsyncMock(return_value=cdp)
        context.close = AsyncMock(side_effect=cleanup_stage("context"))
        browser = Mock()
        browser.new_context = AsyncMock(return_value=context)
        browser.close = AsyncMock(side_effect=cleanup_stage("browser"))
        chromium = Mock()
        chromium.launch = AsyncMock(return_value=browser)
        playwright = Mock()
        playwright.chromium = chromium
        playwright.stop = AsyncMock(side_effect=cleanup_stage("playwright"))
        playwright_manager = Mock()
        playwright_manager.start = AsyncMock(return_value=playwright)

        async_api = types.ModuleType("playwright.async_api")
        async_api.Error = FakePlaywrightError
        async_api.TimeoutError = FakePlaywrightTimeoutError
        async_api.async_playwright = Mock(return_value=playwright_manager)
        playwright_module = types.ModuleType("playwright")
        playwright_module.async_api = async_api

        original_cleanup = guest._cleanup_browser_resources

        async def record_cleanup_budget(**kwargs: Any) -> tuple[BaseException, ...]:
            assert ca_started_at is not None
            cleanup_budget = kwargs["timeout_seconds"]
            # This comparison is deliberately more generous than the real
            # deadline, which starts before CA setup. Recomputing cleanup from
            # the original timeout still exceeds it after the injected delay.
            remaining_budget = max(
                0.0,
                ca_started_at + request.limits.timeout_seconds - loop.time(),
            )
            observed_budgets.append((cleanup_budget, remaining_budget))
            return await original_cleanup(**kwargs)

        async def delayed_ca_install(_home: Path, _ca_path: Path) -> None:
            nonlocal ca_started_at
            ca_started_at = loop.time()
            await asyncio.sleep(0.3)

        with (
            patch.dict(
                sys.modules,
                {
                    "playwright": playwright_module,
                    "playwright.async_api": async_api,
                },
            ),
            patch.object(os, "geteuid", return_value=1000),
            patch.object(guest.importlib.metadata, "version", return_value="1.62.0"),
            patch.object(
                guest,
                "_proxy_and_ca",
                return_value=("http://proxy:8080", Path("/tmp/cayu-ca.pem")),
            ),
            patch.object(guest, "_sanitize_environment"),
            patch.object(guest, "_install_browser_ca", side_effect=delayed_ca_install),
            patch.object(guest, "_cleanup_browser_resources", side_effect=record_cleanup_budget),
            patch.object(guest.os, "chdir"),
            pytest.raises(guest._GuestFailure) as captured,
        ):
            await guest._run(request)

        assert captured.value.code == "cleanup_failed"
        assert len(observed_budgets) == 1
        cleanup_budget, remaining_budget = observed_budgets[0]
        assert cleanup_budget <= remaining_budget + 0.015
        assert cleanup_calls == ["context", "browser", "playwright"]

    asyncio.run(exercise())


def test_guest_browser_cleanup_never_expands_beyond_its_configured_reserve() -> None:
    async def exercise() -> None:
        request = guest._Request(
            url="https://example.com/",
            limits=guest._Limits(
                max_response_bytes=100,
                max_content_bytes=100,
                timeout_seconds=120.0,
                max_redirects=1,
                max_requests=4,
                max_dom_nodes=100,
            ),
        )
        observed_budgets: list[float] = []

        class FakePlaywrightError(Exception):
            pass

        class FakePlaywrightTimeoutError(FakePlaywrightError):
            pass

        class PlaywrightManager:
            async def start(self) -> None:
                raise FakePlaywrightError("browser startup failed")

        async_api = types.ModuleType("playwright.async_api")
        async_api.Error = FakePlaywrightError
        async_api.TimeoutError = FakePlaywrightTimeoutError
        async_api.async_playwright = PlaywrightManager
        playwright_module = types.ModuleType("playwright")
        playwright_module.async_api = async_api

        async def record_cleanup_budget(**kwargs: Any) -> tuple[BaseException, ...]:
            observed_budgets.append(kwargs["timeout_seconds"])
            return ()

        operation_timeout_ms, cleanup_timeout_seconds = guest._browser_time_budget(
            request.limits.timeout_seconds
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "playwright": playwright_module,
                    "playwright.async_api": async_api,
                },
            ),
            patch.object(guest, "_cleanup_browser_resources", side_effect=record_cleanup_budget),
            pytest.raises(guest._GuestFailure) as captured,
        ):
            await guest._fetch_with_browser(
                request,
                "http://proxy:8080",
                state=guest._PageState(
                    max_response_bytes=100,
                    max_redirects=1,
                    max_requests=4,
                ),
                operation_timeout_ms=operation_timeout_ms,
                cleanup_timeout_seconds=cleanup_timeout_seconds,
                cleanup_deadline=(
                    asyncio.get_running_loop().time() + request.limits.timeout_seconds
                ),
            )

        assert captured.value.code == "browser_unavailable"
        assert observed_budgets == [guest._MAX_CLEANUP_RESERVE_SECONDS]

    asyncio.run(exercise())


def test_guest_browser_cleanup_preserves_cancellation_delivered_before_cleanup() -> None:
    async def exercise() -> None:
        cleanup_calls: list[str] = []

        async def operation() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:

                async def cleanup() -> tuple[BaseException, ...]:
                    cleanup_calls.append("cleanup")
                    await asyncio.sleep(0)
                    return ()

                outcome = await guest._await_browser_cleanup_resisting_cancellation(
                    asyncio.create_task(cleanup())
                )
                assert outcome == guest._BrowserCleanupOutcome()
                raise

        operation_task = asyncio.create_task(operation())
        await asyncio.sleep(0)
        operation_task.cancel("cancel before cleanup")
        with pytest.raises(asyncio.CancelledError, match="cancel before cleanup"):
            await operation_task

        assert operation_task.cancelled() is True
        assert cleanup_calls == ["cleanup"]

    asyncio.run(exercise())


@pytest.mark.parametrize("primary_code", [None, "destination_denied"])
def test_guest_classifies_temporary_profile_cleanup_failure(
    tmp_path: Path,
    primary_code: str | None,
) -> None:
    request = guest._request_from_json(
        {
            "protocol_version": guest.PROTOCOL_VERSION,
            "worker_version": guest.WORKER_VERSION,
            "expected_playwright_version": guest.PLAYWRIGHT_VERSION,
            "operation": "fetch",
            "url": "https://example.com/",
            "limits": {
                "max_response_bytes": 100,
                "max_content_bytes": 100,
                "timeout_seconds": 1.0,
                "max_redirects": 1,
                "max_requests": 4,
                "max_dom_nodes": 100,
            },
        }
    )
    cleanup_error = OSError("profile cleanup failed")
    primary_error = guest._GuestFailure(primary_code) if primary_code is not None else None
    profile_owner = guest._TemporaryProfileOwner(
        home=tmp_path / "browser-profile",
        process=Mock(pid=123),
        control_fd=456,
    )
    start_profile_owner = AsyncMock(return_value=profile_owner)
    cleanup_profile_owner = AsyncMock(return_value=(cleanup_error,))

    async def fetch(
        _request: guest._Request,
        _proxy: str,
        *,
        state: guest._PageState,
        operation_timeout_ms: int,
        cleanup_timeout_seconds: float,
        cleanup_deadline: float,
    ) -> dict[str, Any]:
        del state, operation_timeout_ms, cleanup_timeout_seconds, cleanup_deadline
        if primary_error is not None:
            raise primary_error
        return {}

    with (
        patch.object(os, "geteuid", return_value=1000),
        patch.object(guest.importlib.metadata, "version", return_value="1.62.0"),
        patch.object(
            guest,
            "_proxy_and_ca",
            return_value=("http://proxy:8080", Path("/tmp/cayu-ca.pem")),
        ),
        patch.object(
            guest,
            "_start_temporary_profile_owner",
            new=start_profile_owner,
        ),
        patch.object(
            guest,
            "_cleanup_temporary_profile_owner",
            new=cleanup_profile_owner,
        ),
        patch.object(guest, "_sanitize_environment"),
        patch.object(guest, "_install_browser_ca"),
        patch.object(guest, "_fetch_with_browser", side_effect=fetch),
        patch.object(guest.os, "chdir"),
        pytest.raises(guest._GuestFailure) as captured,
    ):
        asyncio.run(guest._run(request))

    assert captured.value.code == "cleanup_failed"
    assert start_profile_owner.await_count == 1
    assert cleanup_profile_owner.await_count == 1
    if primary_error is None:
        assert captured.value.__cause__ is cleanup_error
    else:
        assert isinstance(captured.value.__cause__, BaseExceptionGroup)
        assert captured.value.__cause__.exceptions == (primary_error, cleanup_error)


def test_guest_temporary_profile_owner_deletes_the_released_home() -> None:
    async def exercise() -> None:
        owner = await guest._start_temporary_profile_owner(timeout_seconds=1.0)
        evidence = owner.home / "evidence.txt"
        evidence.write_text("bounded", encoding="utf-8")
        try:
            assert os.getsid(owner.pid) == owner.pid
            errors = await guest._cleanup_temporary_profile_owner(
                owner,
                timeout_seconds=1.0,
            )
            assert errors == ()
            assert owner.home.exists() is False
        finally:
            if owner.home.exists():
                shutil.rmtree(owner.home)

    asyncio.run(exercise())


def test_guest_temporary_profile_helper_rejects_paths_outside_its_root(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "cayu-browser-protected"
    protected.mkdir()
    evidence = protected / "evidence.txt"
    evidence.write_text("retain", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(Path(guest.__file__).resolve()),
            guest._PROFILE_CLEANUP_ARGUMENT,
            str(protected),
            "1.0",
        ],
        input=b"",
        capture_output=True,
        check=False,
        timeout=2,
    )

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert completed.stderr == b""
    assert evidence.read_text(encoding="utf-8") == "retain"


def test_guest_temporary_profile_helper_enforces_its_own_deadline() -> None:
    home = Path(
        guest.tempfile.mkdtemp(
            prefix=guest._TEMPORARY_PROFILE_PREFIX,
            dir=str(guest._TEMPORARY_PROFILE_ROOT),
        )
    )
    fake_stdin = Mock()
    fake_stdin.buffer.read.return_value = b""
    fake_stdout = Mock()

    def blocked_delete(_home: Path) -> None:
        time.sleep(1)

    started_at = time.monotonic()
    try:
        with (
            patch.object(guest.sys, "stdin", fake_stdin),
            patch.object(guest.sys, "stdout", fake_stdout),
            patch.object(guest.shutil, "rmtree", side_effect=blocked_delete),
        ):
            returncode = guest._temporary_profile_cleanup_main(str(home), "0.05")
        elapsed = time.monotonic() - started_at

        assert returncode == 1
        assert elapsed < 0.5
        assert home.exists() is True
    finally:
        if home.exists():
            shutil.rmtree(home)


def test_guest_profile_owner_spawn_failure_fails_without_sync_deletion(
    tmp_path: Path,
) -> None:
    profile_home = tmp_path / "cayu-browser-unowned"
    profile_home.mkdir()

    async def exercise() -> None:
        with (
            patch.object(guest.tempfile, "mkdtemp", return_value=str(profile_home)),
            patch.object(
                guest.subprocess,
                "Popen",
                side_effect=OSError("spawn failed"),
            ),
            patch.object(guest.shutil, "rmtree") as sync_delete,
            pytest.raises(guest._GuestFailure) as captured,
        ):
            await guest._start_temporary_profile_owner(timeout_seconds=0.1)

        assert captured.value.code == "cleanup_failed"
        sync_delete.assert_not_called()

    asyncio.run(exercise())
    assert profile_home.exists() is True


def test_guest_profile_owner_start_timeout_kills_and_reaps_blocked_child(
    tmp_path: Path,
) -> None:
    profile_home = tmp_path / "cayu-browser-blocked-before-ready"
    profile_home.mkdir()
    spawned: list[subprocess.Popen[bytes]] = []
    real_popen = guest.subprocess.Popen

    def blocked_cleanup_command(
        _home: Path,
        *,
        timeout_seconds: float,
    ) -> tuple[str, ...]:
        del timeout_seconds
        return (
            sys.executable,
            "-I",
            "-c",
            "import time; time.sleep(10)",
        )

    def record_spawn(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    async def exercise() -> None:
        with (
            patch.object(guest.tempfile, "mkdtemp", return_value=str(profile_home)),
            patch.object(
                guest,
                "_temporary_profile_cleanup_command",
                side_effect=blocked_cleanup_command,
            ),
            patch.object(guest.subprocess, "Popen", side_effect=record_spawn),
            pytest.raises(TimeoutError),
        ):
            async with asyncio.timeout(0.05):
                await guest._start_temporary_profile_owner(timeout_seconds=0.1)

    started_at = time.monotonic()
    try:
        asyncio.run(exercise())
        elapsed = time.monotonic() - started_at

        assert len(spawned) == 1
        process = spawned[0]
        assert process.poll() == -signal.SIGKILL
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        assert elapsed < 0.5
    finally:
        if profile_home.exists():
            shutil.rmtree(profile_home)


def test_guest_profile_owner_start_preserves_caller_cancellation() -> None:
    request = guest._Request(
        url="https://example.com/",
        limits=guest._Limits(
            max_response_bytes=100,
            max_content_bytes=100,
            timeout_seconds=1.0,
            max_redirects=1,
            max_requests=4,
            max_dom_nodes=100,
        ),
    )

    async def blocked_start(*, timeout_seconds: float) -> guest._TemporaryProfileOwner:
        del timeout_seconds
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def exercise() -> None:
        with (
            patch.object(os, "geteuid", return_value=1000),
            patch.object(guest.importlib.metadata, "version", return_value="1.62.0"),
            patch.object(
                guest,
                "_proxy_and_ca",
                return_value=("http://proxy:8080", Path("/tmp/cayu-ca.pem")),
            ),
            patch.object(
                guest,
                "_start_temporary_profile_owner",
                side_effect=blocked_start,
            ),
        ):
            operation = asyncio.create_task(guest._run(request))
            await asyncio.sleep(0)
            operation.cancel("cancel profile startup")
            with pytest.raises(asyncio.CancelledError, match="cancel profile startup"):
                await operation
            assert operation.cancelled() is True

    asyncio.run(exercise())


def test_guest_profile_owner_start_preserves_authoritative_base_failure(
    tmp_path: Path,
) -> None:
    class AuthoritativeStop(BaseException):
        pass

    primary = AuthoritativeStop("stop worker")
    owner = guest._TemporaryProfileOwner(
        home=tmp_path / "browser-profile",
        process=Mock(pid=123),
        control_fd=456,
    )

    async def exercise() -> None:
        with (
            patch.object(
                guest,
                "_cleanup_temporary_profile_owner",
                new=AsyncMock(return_value=()),
            ),
            pytest.raises(AuthoritativeStop) as captured,
        ):
            await guest._raise_temporary_profile_start_failure(
                owner,
                primary=primary,
                timeout_seconds=0.1,
            )

        assert captured.value is primary

    asyncio.run(exercise())


def test_guest_profile_owner_start_timeout_reports_unproven_cleanup() -> None:
    request = guest._Request(
        url="https://example.com/",
        limits=guest._Limits(
            max_response_bytes=100,
            max_content_bytes=100,
            timeout_seconds=0.05,
            max_redirects=1,
            max_requests=4,
            max_dom_nodes=100,
        ),
    )

    async def blocked_start(*, timeout_seconds: float) -> guest._TemporaryProfileOwner:
        del timeout_seconds
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    started_at = time.monotonic()
    with (
        patch.object(os, "geteuid", return_value=1000),
        patch.object(guest.importlib.metadata, "version", return_value="1.62.0"),
        patch.object(
            guest,
            "_proxy_and_ca",
            return_value=("http://proxy:8080", Path("/tmp/cayu-ca.pem")),
        ),
        patch.object(
            guest,
            "_start_temporary_profile_owner",
            side_effect=blocked_start,
        ),
        pytest.raises(guest._GuestFailure) as captured,
    ):
        asyncio.run(guest._run(request))
    elapsed = time.monotonic() - started_at

    assert captured.value.code == "cleanup_failed"
    assert elapsed < 0.5


def test_guest_blocking_profile_cleanup_returns_bounded_failure(tmp_path: Path) -> None:
    request = guest._request_from_json(
        {
            "protocol_version": guest.PROTOCOL_VERSION,
            "worker_version": guest.WORKER_VERSION,
            "expected_playwright_version": guest.PLAYWRIGHT_VERSION,
            "operation": "fetch",
            "url": "https://example.com/",
            "limits": {
                "max_response_bytes": 100,
                "max_content_bytes": 100,
                "timeout_seconds": 0.1,
                "max_redirects": 1,
                "max_requests": 4,
                "max_dom_nodes": 100,
            },
        }
    )
    profile_home = tmp_path / "cayu-browser-blocked"
    profile_home.mkdir()
    owner_ready = tmp_path / "cleanup-owner-ready"

    def blocking_cleanup_command(
        _home: Path,
        *,
        timeout_seconds: float,
    ) -> tuple[str, ...]:
        del timeout_seconds
        return (
            sys.executable,
            "-I",
            "-c",
            (
                "import pathlib,sys,time;"
                "pathlib.Path(sys.argv[1]).write_text('ready');"
                "sys.stdout.buffer.write(b'1');sys.stdout.buffer.flush();"
                "sys.stdin.buffer.read(1);"
                "time.sleep(1)"
            ),
            str(owner_ready),
        )

    async def fetch(
        _request: guest._Request,
        _proxy: str,
        *,
        state: guest._PageState,
        operation_timeout_ms: int,
        cleanup_timeout_seconds: float,
        cleanup_deadline: float,
    ) -> dict[str, Any]:
        del state, operation_timeout_ms, cleanup_timeout_seconds, cleanup_deadline
        return {}

    started_at = time.monotonic()
    with (
        patch.object(os, "geteuid", return_value=1000),
        patch.object(guest.importlib.metadata, "version", return_value="1.62.0"),
        patch.object(
            guest,
            "_proxy_and_ca",
            return_value=("http://proxy:8080", Path("/tmp/cayu-ca.pem")),
        ),
        patch.object(guest.tempfile, "mkdtemp", return_value=str(profile_home)),
        patch.object(
            guest,
            "_temporary_profile_cleanup_command",
            side_effect=blocking_cleanup_command,
        ),
        patch.object(guest, "_sanitize_environment"),
        patch.object(guest, "_install_browser_ca"),
        patch.object(guest, "_fetch_with_browser", side_effect=fetch),
        patch.object(guest.os, "chdir"),
        pytest.raises(guest._GuestFailure) as captured,
    ):
        asyncio.run(guest._run(request))
    elapsed = time.monotonic() - started_at

    assert captured.value.code == "cleanup_failed"
    assert owner_ready.read_text(encoding="utf-8") == "ready"
    assert elapsed < 0.5
    assert profile_home.exists() is True


def test_guest_operation_deadline_preserves_observed_denial() -> None:
    request = guest._request_from_json(
        {
            "protocol_version": guest.PROTOCOL_VERSION,
            "worker_version": guest.WORKER_VERSION,
            "expected_playwright_version": guest.PLAYWRIGHT_VERSION,
            "operation": "fetch",
            "url": "https://example.com/redirect-out",
            "limits": {
                "max_response_bytes": 100,
                "max_content_bytes": 100,
                "timeout_seconds": 0.1,
                "max_redirects": 1,
                "max_requests": 4,
                "max_dom_nodes": 100,
            },
        }
    )

    async def blocked_fetch(
        _request: guest._Request,
        _proxy: str,
        *,
        state: guest._PageState,
        operation_timeout_ms: int,
        cleanup_timeout_seconds: float,
        cleanup_deadline: float,
    ) -> dict[str, Any]:
        del operation_timeout_ms, cleanup_timeout_seconds, cleanup_deadline
        state.redirects.append(
            {
                "status_code": 302,
                "from_url": "https://example.com/redirect-out",
                "to_url": "https://blocked.example/private",
            }
        )
        state.denied_code = "redirect_denied"
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    profile_owner = guest._TemporaryProfileOwner(
        home=Path("/tmp/cayu-browser-deadline-test"),
        process=Mock(pid=123),
        control_fd=456,
    )

    with (
        patch.object(os, "geteuid", return_value=1000),
        patch.object(guest.importlib.metadata, "version", return_value="1.62.0"),
        patch.object(
            guest,
            "_proxy_and_ca",
            return_value=("http://proxy:8080", Path("/tmp/cayu-ca.pem")),
        ),
        patch.object(guest, "_sanitize_environment"),
        patch.object(guest, "_install_browser_ca"),
        patch.object(guest, "_fetch_with_browser", side_effect=blocked_fetch),
        patch.object(guest.os, "chdir"),
        patch.object(
            guest,
            "_start_temporary_profile_owner",
            new=AsyncMock(return_value=profile_owner),
        ),
        patch.object(
            guest,
            "_cleanup_temporary_profile_owner",
            new=AsyncMock(return_value=()),
        ),
        pytest.raises(guest._GuestFailure) as captured,
    ):
        asyncio.run(guest._run(request))

    assert captured.value.code == "redirect_denied"


def test_guest_page_denial_does_not_erase_redirect_evidence() -> None:
    state = guest._PageState(
        max_response_bytes=100,
        max_redirects=1,
        max_requests=4,
    )

    guest._record_page_denial(state, "destination_denied")
    guest._record_page_denial(state, "redirect_denied")
    guest._record_page_denial(state, "destination_denied")

    assert state.denied_code == "redirect_denied"

    unrelated_state = guest._PageState(
        max_response_bytes=100,
        max_redirects=1,
        max_requests=4,
    )
    guest._record_page_denial(unrelated_state, "dns_failure")
    guest._record_page_denial(unrelated_state, "redirect_denied")

    assert unrelated_state.denied_code == "dns_failure"


@pytest.mark.parametrize(
    ("url", "admitted"),
    [
        ("https://example.com/resource", True),
        ("https://example.com:443/resource", True),
        ("data:text/plain,bounded", True),
        ("blob:https://example.com/id", True),
        ("about:blank", True),
        ("about:srcdoc", True),
        ("about:version", False),
        ("http://example.com/resource", False),
        ("https://user:password@example.com/resource", False),
        ("https://example.com:8443/resource", False),
        ("file:///etc/passwd", False),
        ("javascript:alert(1)", False),
    ],
)
def test_guest_revalidates_every_browser_request_url(url: str, admitted: bool) -> None:
    assert guest._browser_request_is_admissible(url) is admitted


def test_guest_environment_retains_only_browser_operational_authority(tmp_path: Path) -> None:
    source = {
        "AWS_SECRET_ACCESS_KEY": "must-disappear",
        "CURL_CA_BUNDLE": "/ambient/ca.pem",
        "HTTP_PROXY": "http://ambient-proxy",
        "HTTPS_PROXY": "http://cayu-proxy:8080",
        "LD_LIBRARY_PATH": "/ambient/libraries",
        "NODE_EXTRA_CA_CERTS": "/ambient/node-ca.pem",
        "PATH": "/ambient/bin",
        "PLAYWRIGHT_BROWSERS_PATH": "/ambient/browsers",
        "REQUESTS_CA_BUNDLE": "/ambient/requests-ca.pem",
        "SSL_CERT_FILE": "/etc/cayu/ca.pem",
        "https_proxy": "http://cayu-proxy:8080",
    }

    with patch.dict(os.environ, source, clear=True):
        guest._sanitize_environment(
            tmp_path,
            proxy="http://cayu-proxy:8080",
            ca_path=Path("/etc/cayu/ca.pem"),
        )

        assert os.environ == {
            "HOME": str(tmp_path),
            "HTTPS_PROXY": "http://cayu-proxy:8080",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PLAYWRIGHT_BROWSERS_PATH": "/ms-playwright",
            "SSL_CERT_FILE": "/etc/cayu/ca.pem",
            "TMPDIR": str(tmp_path / "tmp"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
        }


def test_guest_rejects_conflicting_ambient_proxy_authority(tmp_path: Path) -> None:
    ca_path = tmp_path / "ca.pem"
    ca_path.write_text("bounded ca")
    environment = {
        "HTTPS_PROXY": "http://managed-proxy:8080",
        "https_proxy": "http://ambient-proxy:9090",
        "SSL_CERT_FILE": str(ca_path),
    }

    with (
        patch.dict(os.environ, environment, clear=True),
        pytest.raises(guest._GuestFailure) as captured,
    ):
        guest._proxy_and_ca()

    assert captured.value.code == "capability_refused"


def test_versioned_browser_image_matches_host_and_guest_contract() -> None:
    example_directory = Path(__file__).parents[2] / "examples" / "browser_fetch"
    dockerfile = (example_directory / "Dockerfile").read_text()
    seccomp_profile = json.loads((example_directory / "seccomp_profile.json").read_text())

    assert f"playwright-{BROWSER_FETCH_PLAYWRIGHT_VERSION}" in dockerfile
    assert f"playwright=={BROWSER_FETCH_PLAYWRIGHT_VERSION}" in dockerfile
    assert "install --no-shell chromium" in dockerfile
    assert "install --with-deps chromium-headless-shell" in dockerfile
    assert "-type f -name chrome_sandbox" in dockerfile
    assert "-name headless_shell -o -name chrome-headless-shell" in dockerfile
    assert "*/chrome-linux/" not in dockerfile
    assert "chrome_sandbox" in dockerfile
    assert "--mode=4755" in dockerfile
    assert 'test -u "${sandbox_target}"' in dockerfile
    assert "COPY --chown=root:root" in dockerfile
    assert "COPY --chown=pwuser:pwuser" not in dockerfile
    assert f'dev.cayu.browser-fetch.protocol="{BROWSER_FETCH_PROTOCOL_VERSION}"' in dockerfile
    assert f'dev.cayu.browser-fetch.worker="{BROWSER_FETCH_WORKER_VERSION}"' in dockerfile
    assert "@sha256:" in dockerfile
    assert "USER pwuser" in dockerfile
    assert seccomp_profile["defaultAction"] == "SCMP_ACT_ERRNO"
    allowed_syscalls = {
        name
        for rule in seccomp_profile["syscalls"]
        if rule["action"] == "SCMP_ACT_ALLOW"
        for name in rule["names"]
    }
    assert {"clone", "setns", "unshare"} <= allowed_syscalls
