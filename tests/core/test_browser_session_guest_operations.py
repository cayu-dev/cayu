"""Interactive browser guest request, operation and artifact transfer contracts."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import types
from pathlib import Path
from typing import Any

import pytest
from tests.core.browser_guest_support import (
    _configure_interactive_daemon_for_test,
    _interactive_limits,
    _interactive_raw_request,
    _interactive_request,
)

import cayu.tools.browser_session as browser_session_module
from cayu.tools import _browser_guest


@pytest.mark.parametrize("filename", ["upload.txt", "a" * 251 + ".txt"])
def test_issue_1275_guest_upload_transfers_owned_payload_without_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    class Locator:
        files: list[dict[str, Any]] = []

        async def evaluate(self, expression: str) -> dict[str, bool]:
            return {"is_file_input": True, "multiple": False}

        async def set_input_files(self, files: list[dict[str, Any]]) -> None:
            self.files = files

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.home = tmp_path
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("observe").__dict__,
                "operation": "upload",
                "operation_id": "owned-upload",
                "ref": "upload-ref",
                "upload_files": (
                    _browser_guest._InteractiveUploadFile(
                        artifact_id="art_upload",
                        content_sha256=hashlib.sha256(b"upload-content").hexdigest(),
                        filename=filename,
                        content_type="text/plain",
                        size_bytes=14,
                        content=b"upload-content",
                    ),
                ),
            }
        )

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("Upload must not enter filesystem materialization or cleanup")

        monkeypatch.setattr(_browser_guest.os, "open", forbidden)
        monkeypatch.setattr(_browser_guest.os, "write", forbidden)
        monkeypatch.setattr(_browser_guest.os, "fsync", forbidden)
        monkeypatch.setattr(_browser_guest.shutil, "rmtree", forbidden)
        locator = Locator()
        result = await daemon._upload_files(
            _browser_guest._InteractivePage(page=object(), session_id="bs_test", page_id="bp_test"),
            request,
            locator,
            "internal-upload-ref",
        )
        assert result["selection_state"] == "selected"
        await asyncio.sleep(0)
        assert locator.files == [
            {
                "name": filename,
                "mimeType": "text/plain",
                "buffer": b"upload-content",
            }
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("phase", ["target", "selection"])
def test_issue_1275_guest_upload_preserves_browser_failure(phase: str, tmp_path: Path) -> None:
    error = RuntimeError("browser dispatch failed")

    class Locator:
        async def evaluate(self, expression: str) -> dict[str, bool]:
            if phase == "target":
                raise error
            return {"is_file_input": True, "multiple": False}

        async def set_input_files(self, files: Any) -> None:
            raise error

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.home = tmp_path
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("observe").__dict__,
                "operation": "upload",
                "operation_id": "failed-upload",
                "ref": "upload-ref",
                "upload_files": (
                    _browser_guest._InteractiveUploadFile(
                        artifact_id="art_upload",
                        content_sha256=hashlib.sha256(b"x").hexdigest(),
                        filename="x.txt",
                        content_type="text/plain",
                        size_bytes=1,
                        content=b"x",
                    ),
                ),
            }
        )
        with pytest.raises(RuntimeError) as raised:
            await daemon._upload_files(
                _browser_guest._InteractivePage(
                    page=object(), session_id="bs_test", page_id="bp_test"
                ),
                request,
                Locator(),
                "internal-upload-ref",
            )
        assert raised.value is error
        assert (
            _browser_guest._interactive_runtime_failure(
                "upload",
                error,
                browser_connected=False,
                page_open=False,
            ).code
            == "browser_crash"
        )
        uncertain = _browser_guest._interactive_runtime_failure(
            "upload",
            error,
            browser_connected=True,
            page_open=True,
        )
        assert uncertain.code == "outcome_ambiguous"
        assert uncertain.allocation_disposition != "live"

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["disconnection", "lost-ack", "cancellation"])
def test_issue_1275_upload_dispatch_failure_through_guest_daemon(
    tmp_path: Path,
    outcome: str,
) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        selected: list[dict[str, Any]] = []
        connected = True

        class Target:
            async def evaluate(self, expression: str) -> Any:
                if "isConnected" in expression:
                    return True
                return {"is_file_input": True, "multiple": False}

            async def set_input_files(self, files: list[dict[str, Any]]) -> None:
                selected.extend(files)
                entered.set()
                await release.wait()
                raise RuntimeError("selection acknowledgement lost")

        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("upload").__dict__,
                "expected_revision": "br_upload",
                "expected_control_epoch": 1,
                "ref": "ref_upload",
                "upload_files": (
                    _browser_guest._InteractiveUploadFile(
                        artifact_id="art_upload",
                        content_sha256=hashlib.sha256(b"x").hexdigest(),
                        filename="x.txt",
                        content_type="text/plain",
                        size_bytes=1,
                        content=b"x",
                    ),
                ),
            }
        )
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.home = tmp_path
        daemon.context = types.SimpleNamespace()
        daemon.browser = types.SimpleNamespace(is_connected=lambda: connected)
        state = _browser_guest._InteractivePage(
            page=types.SimpleNamespace(
                url="https://example.test/", is_closed=lambda: not connected
            ),
            session_id="bs_test",
            page_id="bp_test",
            lifecycle="active",
            public_url="https://example.test/",
            revision="br_upload",
            refs={"ref_upload": "e1"},
            ref_targets={"ref_upload": Target()},
        )
        daemon.pages[state.page_id] = state
        await _configure_interactive_daemon_for_test(daemon, request)
        task = asyncio.create_task(daemon.execute(request))
        await asyncio.wait_for(entered.wait(), timeout=10)
        assert selected[0]["buffer"] == b"x"
        assert state.revision is None
        if outcome == "cancellation":
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            assert task.cancelling() == 1
            assert request.operation_id not in daemon.operations
        else:
            connected = outcome != "disconnection"
            release.set()
            result = await task
            assert result["error"] == (
                "browser_crash" if outcome == "disconnection" else "outcome_ambiguous"
            )
            assert result["allocation_disposition"] == "uncertain"
            assert await daemon.execute(request) == result
            assert len(selected) == 1
        assert state.revision is None

    asyncio.run(scenario())


def test_issue_1275_guest_refuses_unsafe_reload_before_browser_dispatch() -> None:
    class _HistoryCdp:
        async def send(
            self,
            method: str,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del params
            if method == "Page.getFrameTree":
                return {"frameTree": {"frame": {"id": "main", "loaderId": "post"}}}
            assert method == "Page.getNavigationHistory"
            return {"currentIndex": 0, "entries": [{"id": 11}]}

    class _Page:
        reload_called = False

        async def reload(self, **kwargs: Any) -> None:
            del kwargs
            self.reload_called = True

    async def scenario() -> None:
        page = _Page()
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
            cdp=_HistoryCdp(),
            current_document_method="POST",
            document_methods={"post": "POST"},
            history_methods={11: "POST"},
        )

        # A subsequent GET response (204, download, or aborted navigation) does
        # not commit its loader. The retained document must still be POST.
        _browser_guest._retain_document_request(
            state,
            "main",
            {
                "type": "Document",
                "frameId": "main",
                "loaderId": "uncommitted-get",
                "request": {"method": "GET"},
            },
        )
        with pytest.raises(_browser_guest._GuestFailure) as raised:
            await _browser_guest._interactive_safe_reload(
                state,
                _browser_guest._InteractiveRequest(
                    **{
                        **_interactive_request("observe").__dict__,
                        "operation": "reload",
                    }
                ),
            )

        assert raised.value.code == "unsafe_reload"
        assert raised.value.allocation_disposition == "live"
        assert page.reload_called is False

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel", "timeout"])
def test_reload_loader_guard_retains_navigation_waiter_ownership(outcome: str) -> None:
    async def scenario() -> None:
        dispatched = asyncio.Event()
        waiter_closed = asyncio.Event()
        failure = RuntimeError("CDP transport lost")

        class Cdp:
            async def send(self, method: str, params: Any = None) -> dict[str, Any]:
                if method == "Page.getFrameTree":
                    return {"frameTree": {"frame": {"id": "main", "loaderId": "safe"}}}
                if method == "Page.getNavigationHistory":
                    return {"currentIndex": 0, "entries": [{"id": 11}]}
                assert method == "Page.reload"
                assert params == {"loaderId": "safe"}
                dispatched.set()
                if outcome == "failure":
                    raise failure
                if outcome in {"cancel", "timeout"}:
                    await asyncio.Event().wait()
                return {}

        class Page:
            @contextlib.asynccontextmanager
            async def expect_navigation(self, **kwargs: Any):
                assert kwargs["wait_until"] == "load"
                try:
                    yield
                finally:
                    waiter_closed.set()

        state = _browser_guest._InteractivePage(
            page=Page(),
            session_id="bs_test",
            page_id="bp_test",
            cdp=Cdp(),
            document_methods={"safe": "GET"},
        )
        task = asyncio.create_task(
            _browser_guest._interactive_safe_reload(
                state, _interactive_request("reload", limits=_interactive_limits(max_wait_ms=1000))
            )
        )
        await dispatched.wait()
        if outcome == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
        elif outcome == "failure":
            with pytest.raises(RuntimeError) as raised:
                await task
            assert raised.value is failure
        elif outcome == "timeout":
            with pytest.raises(TimeoutError):
                await task
            assert not task.cancelled() and task.cancelling() == 0
        else:
            await task
        assert waiter_closed.is_set()

    asyncio.run(scenario())


def test_issue_1275_guest_refuses_missing_history_entry_before_traversal() -> None:
    class _HistoryCdp:
        async def send(
            self,
            method: str,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            del params
            if method == "Page.getFrameTree":
                return {"frameTree": {"frame": {"id": "main", "loaderId": "get"}}}
            assert method == "Page.getNavigationHistory"
            return {"currentIndex": 0, "entries": [{"id": 11}]}

    class _Page:
        go_back_called = False

        async def go_back(self, **kwargs: Any) -> None:
            del kwargs
            self.go_back_called = True

    async def scenario() -> None:
        page = _Page()
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
            cdp=_HistoryCdp(),
            current_document_method="GET",
            document_methods={"get": "GET"},
            history_methods={11: "GET"},
        )

        with pytest.raises(_browser_guest._GuestFailure) as raised:
            await _browser_guest._interactive_history_traversal(
                state,
                _browser_guest._InteractiveRequest(
                    **{
                        **_interactive_request("observe").__dict__,
                        "operation": "back",
                    }
                ),
                offset=-1,
            )

        assert raised.value.code == "history_unavailable"
        assert raised.value.allocation_disposition == "live"
        assert page.go_back_called is False

    asyncio.run(scenario())


def test_interactive_guest_request_parser_rejects_escape_hatches() -> None:
    raw = _interactive_raw_request("navigate")
    parsed = _browser_guest._interactive_request_from_json(raw)
    assert parsed.operation == "navigate"

    for forbidden in ("selector", "javascript", "proxy", "headers", "launch_args"):
        with pytest.raises(RuntimeError, match="incompatible_browser"):
            _browser_guest._interactive_request_from_json({**raw, forbidden: "secret"})

    navigation_popup = json.loads(json.dumps(raw))
    navigation_popup["page_policy"] = {
        "multi_page": True,
        "popup": {
            "mode": "same_origin",
            "allowed_operations": ["navigate"],
            "allowed_opener_origins": [],
            "allowed_destination_origins": [],
        },
    }
    navigation_popup["limits"].update(
        {
            "max_pages": 2,
            "max_provisional_pages": 1,
            "max_page_creations_per_operation": 1,
            "max_total_page_creations": 2,
        }
    )
    with pytest.raises(RuntimeError, match="incompatible_browser"):
        _browser_guest._interactive_request_from_json(navigation_popup)


def test_interactive_guest_rejects_stale_authority_before_mutating_page_counters() -> None:
    class _Daemon(_browser_guest._InteractiveDaemon):
        async def _execute_page(self, state: Any, request: Any) -> dict[str, Any]:
            del state, request
            raise AssertionError("stale page authority must fail before dispatch")

    async def scenario() -> None:
        daemon = _Daemon("bs_test")
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click").__dict__,
                "page_id": "bp_root",
                "expected_revision": "br_stale",
                "expected_control_epoch": 1,
                "ref": "ref_root",
            }
        )
        daemon.context = types.SimpleNamespace()
        await _configure_interactive_daemon_for_test(daemon, request)
        page = _browser_guest._InteractivePage(
            page=object(),
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            revision="br_current",
            refs={"ref_root": "internal"},
        )
        daemon.pages[page.page_id] = page
        daemon.active_page_id = page.page_id
        daemon.total_page_creations = 1

        result = await daemon.execute(request)

        assert result["error"] == "incompatible_browser"
        assert page.control_epoch == 1
        assert page.operation_count == 0
        assert page.last_operation_id_sha256 is None
        assert daemon.total_operations == 0
        assert page.revision == "br_current"
        assert page.refs == {"ref_root": "internal"}

    asyncio.run(scenario())


def test_interactive_guest_operation_ledger_deduplicates_without_replay() -> None:
    class _LedgerDaemon(_browser_guest._InteractiveDaemon):
        calls = 0

        async def _execute_locked(self, request):
            self.calls += 1
            return {
                "protocol_version": "cayu.browser-session.v4",
                "worker_version": "19",
                "playwright_version": "1.62.0",
                "kind": "success",
                "observation": {"call": self.calls, "operation": request.operation},
                "artifacts": [],
            }

    async def scenario() -> None:
        daemon = _LedgerDaemon("bs_test")
        daemon.context = types.SimpleNamespace()
        request = _interactive_request("observe")
        await _configure_interactive_daemon_for_test(daemon, request)

        first = await daemon.execute(request)
        replay = await daemon.execute(request)
        conflict = await daemon.execute(
            _browser_guest._InteractiveRequest(
                **{
                    **request.__dict__,
                    "operation": "wait",
                    "wait_ms": 1,
                }
            )
        )

        assert first == replay
        assert daemon.calls == 1
        assert conflict["kind"] == "error"
        assert conflict["error"] == "operation_conflict"

    asyncio.run(scenario())


def test_interactive_guest_bounds_full_page_geometry_before_capture() -> None:
    class _Page:
        screenshot_called = False

        async def screenshot(self, **kwargs: Any) -> bytes:
            del kwargs
            self.screenshot_called = True
            raise AssertionError("oversized full pages must not be captured")

    class _Cdp:
        async def send(self, method: str) -> dict[str, Any]:
            assert method == "Page.getLayoutMetrics"
            return {"cssContentSize": {"width": 1280, "height": 10_000}}

    async def scenario() -> None:
        page = _Page()
        request = _interactive_request(
            "screenshot",
            full_page=True,
            limits=_interactive_limits(max_page_height=720),
        )

        with pytest.raises(RuntimeError, match="oversized_artifact"):
            await _browser_guest._interactive_screenshot(page, _Cdp(), request)
        assert page.screenshot_called is False

    asyncio.run(scenario())


def test_interactive_guest_cancels_download_when_response_limit_is_reached() -> None:
    class _Download:
        cancelled = False
        released = asyncio.Event()

        async def path(self) -> str:
            await self.released.wait()
            return "/tmp/not-read"

        async def cancel(self) -> None:
            self.cancelled = True
            self.released.set()

    async def scenario() -> None:
        download = _Download()
        state = _browser_guest._InteractivePage(
            page=object(),
            session_id="bs_test",
            page_id="bp_test",
            limit_exceeded=True,
        )

        with pytest.raises(RuntimeError, match="oversized_response"):
            await _browser_guest._interactive_download_path(
                download,
                state,
                _interactive_limits(),
            )
        assert download.cancelled is True

    asyncio.run(scenario())


def test_interactive_guest_cancels_unrequested_download_through_action_boundary() -> None:
    class _Download:
        def __init__(self) -> None:
            self.cancelled = False

        async def cancel(self) -> None:
            self.cancelled = True

    class _Locator:
        def __init__(
            self,
            daemon: _browser_guest._InteractiveDaemon,
            state: _browser_guest._InteractivePage,
            page: _Page,
        ) -> None:
            self.daemon = daemon
            self.state = state
            self.page = page

        async def element_handle(self) -> _Locator:
            return self

        async def click(self) -> None:
            self.page.handlers["download"](self.page.download)
            task = self.state.unexpected_download_task
            assert task is not None
            assert await task is True

    class _Page:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}
            self.download = _Download()
            self.closed = False
            self.locator_owner: _Locator | None = None

        def on(self, event: str, callback: Any) -> None:
            self.handlers[event] = callback

        def locator(self, selector: str) -> _Locator:
            assert selector == "aria-ref=internal-ref"
            assert self.locator_owner is not None
            return self.locator_owner

        async def close(self) -> None:
            self.closed = True

    class _Context:
        async def route(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            assert callable(callback)

        async def add_init_script(self, script: str) -> None:
            assert script

        async def close(self) -> None:
            return None

    async def scenario() -> None:
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click").__dict__,
                "expected_revision": "br_current",
                "expected_control_epoch": 1,
                "ref": "public-ref",
            }
        )
        page = _Page()
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = _Context()
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
            lifecycle="active",
            configured=True,
            revision="br_current",
            last_observation_revision="br_current",
            refs={"public-ref": "internal-ref"},
            observation_count=1,
            ref_count=1,
        )
        page.locator_owner = _Locator(daemon, state, page)
        daemon.pages[state.page_id] = state
        daemon.active_page_id = state.page_id
        daemon.total_page_creations = 1
        daemon.total_observations = 1
        daemon.total_refs = 1
        await _configure_interactive_daemon_for_test(daemon, request)
        page.on("download", lambda value: daemon._handle_page_download(state, value))

        result = await daemon.execute(request)

        assert result["error"] == "policy_denied"
        assert page.download.cancelled is True
        assert page.closed is True
        assert state.lifecycle == "closed"
        assert result["page_delta"]["closed_page_ids"] == [state.page_id]

    asyncio.run(scenario())


def test_interactive_guest_accepts_playwright_download_path(tmp_path: Path) -> None:
    class _Download:
        async def path(self) -> Path:
            return tmp_path / "download.txt"

    async def scenario() -> None:
        state = _browser_guest._InteractivePage(
            page=object(),
            session_id="bs_test",
            page_id="bp_test",
        )

        assert await _browser_guest._interactive_download_path(
            _Download(),
            state,
            _interactive_limits(),
        ) == str(tmp_path / "download.txt")

    asyncio.run(scenario())


def test_interactive_guest_allows_only_the_exact_authorized_download() -> None:
    class _Download:
        async def cancel(self) -> None:
            raise AssertionError("the exact authorized download must not be cancelled")

    request = _browser_guest._interactive_request_from_json(
        {
            **_interactive_raw_request("download"),
            "ref": "public-ref",
            "expected_revision": "br_current",
            "expected_control_epoch": 1,
        }
    )
    daemon = _browser_guest._InteractiveDaemon("bs_test")
    state = _browser_guest._InteractivePage(
        page=object(),
        session_id="bs_test",
        page_id="bp_test",
        lifecycle="active",
        authorized_download_operation_id_sha256=hashlib.sha256(
            request.operation_id.encode("utf-8")
        ).hexdigest(),
    )
    daemon.active_request = request

    daemon._handle_page_download(state, _Download())

    assert state.unexpected_download_task is None
    assert state.denied_code is None


def test_interactive_transport_envelope_covers_every_supported_maximum() -> None:
    expected = browser_session_module._browser_session_response_envelope_limit(
        max_artifact_bytes=browser_session_module.MAX_BROWSER_SESSION_MAX_ARTIFACT_BYTES,
        max_snapshot_bytes=browser_session_module.MAX_BROWSER_SESSION_MAX_SNAPSHOT_BYTES,
        max_refs=browser_session_module.MAX_BROWSER_SESSION_MAX_REFS,
        max_page_records=(browser_session_module.MAX_BROWSER_SESSION_MAX_TOTAL_PAGE_CREATIONS),
    )

    assert (
        expected + 7 * browser_session_module.BROWSER_PROFILE_MAX_PLAINTEXT_BYTES
        == _browser_guest._INTERACTIVE_MAX_MESSAGE_BYTES
    )
    assert 4 * ((browser_session_module.MAX_BROWSER_SESSION_MAX_ARTIFACT_BYTES + 2) // 3) > (
        40 * 1024 * 1024
    )


def test_interactive_guest_classifies_browser_close_during_download_as_crash() -> None:
    target_closed_type = type(
        "TargetClosedError",
        (RuntimeError,),
        {"__module__": "playwright._impl._errors"},
    )

    class _DownloadInfo:
        async def __aenter__(self) -> _DownloadInfo:
            return self

        async def __aexit__(self, *args: Any) -> None:
            del args

    class _Page:
        def expect_download(self, **kwargs: Any) -> _DownloadInfo:
            del kwargs
            return _DownloadInfo()

    class _Locator:
        async def click(self) -> None:
            raise target_closed_type()

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        state = _browser_guest._InteractivePage(
            page=_Page(),
            session_id="bs_test",
            page_id="bp_test",
        )

        with pytest.raises(RuntimeError, match="browser_crash"):
            await daemon._download_and_observe(
                state,
                _interactive_request("download"),
                _Locator(),
            )

    asyncio.run(scenario())
