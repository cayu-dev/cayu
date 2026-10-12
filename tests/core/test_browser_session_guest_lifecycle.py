"""Interactive browser guest allocation limits, retirement and cleanup contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import types
from pathlib import Path
from typing import Any

import pytest
from tests.core.browser_guest_support import (
    _BoundedSnapshotCdp,
    _configure_interactive_daemon_for_test,
    _interactive_limits,
    _interactive_raw_request,
    _interactive_request,
)
from tests.core.browser_session_support import _context, _WireRunner

import cayu.tools.browser_session as browser_session_module
from cayu.runners.base import ExecCommand, ExecResult
from cayu.runners.workloads import PINNED_BROWSER_SESSION_WORKLOAD
from cayu.tools import _browser_guest
from cayu.tools.browser_session import BrowserSessionTool


def test_interactive_guest_reports_background_page_crash_on_next_operation() -> None:
    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        request = _interactive_request("list_pages", limits=limits)
        daemon.context = types.SimpleNamespace()
        await _configure_interactive_daemon_for_test(daemon, request)
        root = _browser_guest._InteractivePage(
            page=object(),
            session_id="bs_test",
            page_id="bp_root",
            creation_epoch=1,
            lifecycle="background",
            revision="br_root",
            public_url="https://example.test/root",
        )
        active = _browser_guest._InteractivePage(
            page=object(),
            session_id="bs_test",
            page_id="bp_active",
            creation_epoch=2,
            lifecycle="active",
            opener_page_id="bp_root",
            creating_operation_id_sha256=hashlib.sha256(b"open-popup").hexdigest(),
            revision="br_active",
            public_url="https://example.test/popup",
        )
        daemon.pages = {root.page_id: root, active.page_id: active}
        daemon.active_page_id = active.page_id
        daemon.total_page_creations = 2
        daemon._mark_page_crashed(root)

        result = await daemon.execute(request)

        assert result["kind"] == "success"
        assert result["page_delta"]["crashed_page_ids"] == [root.page_id]
        crashed = next(
            page for page in result["page_set"]["pages"] if page["page_id"] == root.page_id
        )
        assert crashed["lifecycle"] == "crashed"
        assert crashed["terminal_reason"] == "browser_crash"
        assert result["page_set"]["active_page_id"] == active.page_id

    asyncio.run(scenario())


def test_interactive_guest_page_close_timeout_retains_one_cleanup_owner() -> None:
    class _Page:
        def __init__(self) -> None:
            self.close_calls = 0
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def close(self) -> None:
            self.close_calls += 1
            self.close_started.set()
            await self.release_close.wait()

    async def scenario() -> None:
        page = _Page()
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
        )

        first = await daemon._await_page_close(state, timeout_seconds=0.001)
        assert first is False
        assert state.cleanup_task is not None
        second = asyncio.create_task(daemon._await_page_close(state, timeout_seconds=1))
        await page.close_started.wait()
        await asyncio.sleep(0)
        assert page.close_calls == 1
        page.release_close.set()
        assert await second is True
        assert page.close_calls == 1

    asyncio.run(scenario())


def test_interactive_guest_page_close_capacity_rejection_preserves_page_authority() -> None:
    class _Page:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        page = _Page()
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        limits = _interactive_limits(max_page_cleanup_operations=1)
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("close_page", limits=limits).__dict__,
                "page_id": "bp_test",
                "operation_id": "close-without-capacity",
            }
        )
        daemon.context = types.SimpleNamespace()
        await _configure_interactive_daemon_for_test(daemon, request)
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
            lifecycle="active",
            revision="br_current",
            last_observation_revision="br_current",
            observation_count=1,
            ref_count=1,
            refs={"ref_current": "internal"},
        )
        daemon.pages[state.page_id] = state
        daemon.active_page_id = state.page_id
        daemon.total_page_creations = 1
        daemon.total_observations = 1
        daemon.total_refs = 1
        daemon.cleanup_operation_count = 1

        result = await daemon.execute(request)

        assert result["kind"] == "error"
        assert result["error"] == "resource_exhausted"
        assert state.lifecycle == "active"
        assert state.control_epoch == 1
        assert state.revision == "br_current"
        assert state.refs == {"ref_current": "internal"}
        assert page.close_calls == 0

    asyncio.run(scenario())


def test_interactive_guest_whole_close_preserves_cancellation_after_all_cleanup() -> None:
    class _Page:
        def __init__(self) -> None:
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            self.close_started.set()
            await self.release_close.wait()

    class _Owner:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        page = _Page()
        context = _Owner()
        browser = _Owner()
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = context
        daemon.browser = browser
        daemon.pages["bp_test"] = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
            lifecycle="active",
            revision="br_test",
        )
        daemon.active_page_id = "bp_test"
        owner = asyncio.create_task(daemon.close(timeout_seconds=1))
        await page.close_started.wait()
        owner.cancel()
        page.release_close.set()

        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled() is True
        assert owner.cancelling() == 1
        assert page.close_calls == 1
        assert context.closed is True
        assert browser.closed is True
        assert daemon.pages == {}

    asyncio.run(scenario())


def test_interactive_guest_operation_ledger_reserves_cleanup_capacity() -> None:
    class _LedgerDaemon(_browser_guest._InteractiveDaemon):
        async def _execute_locked(self, request):
            return {
                "protocol_version": "cayu.browser-session.v4",
                "worker_version": "19",
                "playwright_version": "1.62.0",
                "kind": "success",
                "observation": {"operation": request.operation},
                "artifacts": [],
            }

    async def scenario() -> None:
        daemon = _LedgerDaemon("bs_test")
        daemon.context = types.SimpleNamespace()
        limits = _interactive_limits(max_operations=1)
        await _configure_interactive_daemon_for_test(
            daemon,
            _interactive_request("observe", limits=limits),
        )

        first = await daemon.execute(_interactive_request("observe", limits=limits))
        exhausted = await daemon.execute(_interactive_request("wait", limits=limits))
        cleanup = await daemon.execute(_interactive_request("close", limits=limits))

        assert first["kind"] == "success"
        assert exhausted["error"] == "resource_exhausted"
        assert cleanup["kind"] == "success"

    asyncio.run(scenario())


def test_interactive_guest_close_reports_cleanup_failure() -> None:
    class _FailedContext:
        async def close(self) -> None:
            raise OSError("private cleanup failure")

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = _FailedContext()
        request = _browser_guest._interactive_request_from_json(_interactive_raw_request("close"))
        await _configure_interactive_daemon_for_test(daemon, request)

        result = await daemon.execute(request)

        assert result["kind"] == "error"
        assert result["error"] == "cleanup_failed"
        assert "private cleanup failure" not in json.dumps(result)

    asyncio.run(scenario())


def test_interactive_guest_background_expiry_does_not_hide_cleanup_failure() -> None:
    class _Page:
        async def close(self) -> None:
            raise OSError("private background cleanup failure")

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = types.SimpleNamespace()
        limits = _interactive_limits(max_background_lifetime_seconds=1)
        request = _interactive_request("list_pages", limits=limits)
        await _configure_interactive_daemon_for_test(daemon, request)
        page = _browser_guest._InteractivePage(
            page=_Page(),
            session_id="bs_test",
            page_id="bp_test",
            creation_epoch=1,
            lifecycle="background",
            background_since=0.0,
            revision="br_test",
            public_url="https://example.test",
        )
        daemon.pages[page.page_id] = page
        daemon.active_page_id = page.page_id
        daemon.total_page_creations = 1

        result = await daemon.execute(request)

        assert result["kind"] == "error"
        assert result["error"] == "cleanup_failed"
        assert result["allocation_disposition"] == "uncertain"
        assert result["page_set"]["pages"][0]["lifecycle"] == "uncertain"
        assert daemon.total_operations == 0
        assert page.cleanup_task is not None
        assert page.cleanup_task.done() is True
        assert "private background cleanup failure" not in json.dumps(result)

    asyncio.run(scenario())


def test_interactive_guest_failed_initial_navigation_retires_allocation() -> None:
    class _Page:
        closed = False

        async def close(self) -> None:
            self.closed = True

    class _Context:
        def __init__(self, page: _Page) -> None:
            self.page = page

        async def new_page(self) -> _Page:
            return self.page

        async def close(self) -> None:
            await self.page.close()

    class _FailedNavigateDaemon(_browser_guest._InteractiveDaemon):
        async def _configure_page(self, state, limits) -> None:
            del state, limits

        async def _execute_page(self, state, request):
            del state, request
            raise _browser_guest._GuestFailure("destination_denied")

    async def scenario() -> None:
        page = _Page()
        daemon = _FailedNavigateDaemon("bs_test")
        daemon.context = _Context(page)
        request = _browser_guest._interactive_request_from_json(
            _interactive_raw_request("navigate")
        )
        await _configure_interactive_daemon_for_test(daemon, request)

        result = await daemon.execute(request)
        assert result["kind"] == "error"

        assert result["error"] == "destination_denied"
        assert result["allocation_disposition"] == "retired"
        assert daemon.pages == {}
        assert daemon.close_after_response is True
        assert daemon.closing is True
        assert page.closed is True

    asyncio.run(scenario())


@pytest.mark.parametrize("navigation_limit_order", [None, "before", "after"])
def test_interactive_guest_closes_background_page_at_response_byte_limit(
    navigation_limit_order: str | None,
) -> None:
    class _Cdp:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        async def send(
            self,
            method: str,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            if method == "Network.enable":
                assert params is None
                return {}
            if method == "Page.getFrameTree":
                assert params is None
                return {"frameTree": {"frame": {"id": "frame-main"}}}
            if method == "Fetch.enable":
                assert params is not None
                return {}
            raise AssertionError(f"unexpected CDP method: {method}")

        def on(self, event: str, callback: Any) -> None:
            self.handlers[event] = callback

    class _Page:
        def __init__(self) -> None:
            self.closed = asyncio.Event()
            self.handlers: dict[str, Any] = {}

        def set_default_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def set_default_navigation_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def on(self, event: str, callback: Any) -> None:
            self.handlers[event] = callback

        async def close(self) -> None:
            self.closed.set()

    class _Context:
        def __init__(self, cdp: _Cdp) -> None:
            self.cdp = cdp
            self.closed = False

        async def route(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            assert callable(callback)

        async def new_cdp_session(self, page: _Page) -> _Cdp:
            del page
            return self.cdp

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        limits = _interactive_limits(max_response_bytes=4)
        cdp = _Cdp()
        page = _Page()
        context = _Context(cdp)
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = context
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
        )
        daemon.pages[state.page_id] = state
        request = _interactive_request("observe", limits=limits)
        await _configure_interactive_daemon_for_test(daemon, request)
        await daemon._configure_page(state, limits)

        def exceed_navigation_limit() -> None:
            page.main_frame = object()
            state.navigation_epoch = limits.max_requests
            daemon._mark_page_navigated(state, page.main_frame)

        if navigation_limit_order == "before":
            exceed_navigation_limit()
        cdp.handlers["Network.dataReceived"]({"encodedDataLength": 5})
        if navigation_limit_order == "after":
            exceed_navigation_limit()
        await page.closed.wait()

        assert state.limit_exceeded is True
        result = await daemon.execute(request)
        expected_code = (
            "oversized_response" if navigation_limit_order is None else "resource_exhausted"
        )
        assert result["error"] == expected_code
        assert result["allocation_disposition"] == "retired"
        if navigation_limit_order is None:
            assert result["limit"] == {
                "identifier": "response_bytes",
                "bound": 4,
                "observed": 5,
                "measurement": "lower_bound",
                "units": "bytes",
            }
        else:
            assert "limit" not in result
        parsed = browser_session_module._parse_runner_response(
            json.dumps(result),
            max_artifact_bytes=1024,
            max_page_records=10,
            max_page_creations_per_operation=1,
        )
        assert parsed.failure is not None
        assert parsed.failure.code == expected_code
        assert parsed.failure.limit == result.get("limit")
        assert parsed.allocation_disposition == "retired"
        assert daemon.closing is True
        assert daemon.close_after_response is True
        assert context.closed is True
        assert daemon.pages == {}

    asyncio.run(scenario())


def test_interactive_guest_rechecks_response_limit_after_final_observation() -> None:
    class _Cdp(_BoundedSnapshotCdp):
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        async def send(
            self,
            method: str,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            if method == "Network.enable":
                return {}
            return await super().send(method, params)

        def on(self, event: str, callback: Any) -> None:
            self.handlers[event] = callback

    class _Locator:
        async def aria_snapshot(self, **kwargs: Any) -> str:
            del kwargs
            return '- document "Example"'

    class _Page:
        url = "https://example.test"

        def __init__(self, cdp: _Cdp) -> None:
            self.cdp = cdp
            self.closed = False
            self.handlers: dict[str, Any] = {}

        def set_default_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def set_default_navigation_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def on(self, event: str, callback: Any) -> None:
            self.handlers[event] = callback

        def locator(self, selector: str) -> _Locator:
            assert selector == ":root"
            return _Locator()

        async def title(self) -> str:
            self.cdp.handlers["Network.dataReceived"]({"encodedDataLength": 5})
            return "Example"

        async def close(self) -> None:
            self.closed = True

    class _Context:
        def __init__(self, cdp: _Cdp) -> None:
            self.cdp = cdp
            self.closed = False

        async def route(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            assert callable(callback)

        async def new_cdp_session(self, page: _Page) -> _Cdp:
            del page
            return self.cdp

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        limits = _interactive_limits(max_response_bytes=4)
        cdp = _Cdp()
        page = _Page(cdp)
        context = _Context(cdp)
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = context
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
        )
        daemon.pages[state.page_id] = state
        request = _interactive_request("observe", limits=limits)
        await _configure_interactive_daemon_for_test(daemon, request)
        await daemon._configure_page(state, limits)

        result = await daemon.execute(request)

        assert result["error"] == "oversized_response"
        assert result["allocation_disposition"] == "retired"
        assert result["limit"] == {
            "identifier": "response_bytes",
            "bound": 4,
            "observed": 5,
            "measurement": "lower_bound",
            "units": "bytes",
        }
        assert state.limit_exceeded is True
        assert page.closed is True
        assert context.closed is True
        assert daemon.pages == {}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("max_refs_per_page", "max_total_refs"),
    ((1, 8), (8, 1)),
    ids=("per-page", "aggregate"),
)
def test_interactive_guest_ref_limits_independently_retire_allocation(
    monkeypatch: pytest.MonkeyPatch,
    max_refs_per_page: int,
    max_total_refs: int,
) -> None:
    class _Page:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

        def is_closed(self) -> bool:
            return False

    class _Context:
        def __init__(self) -> None:
            self.close_calls = 0

        async def add_init_script(self, script: str) -> None:
            del script

        async def route(self, pattern: str, callback: Any) -> None:
            del pattern, callback

        async def close(self) -> None:
            self.close_calls += 1

    async def observation(
        state: _browser_guest._InteractivePage,
        limits: Any,
        *,
        browser_version: str,
        visual_policy: Any = None,
        visual_artifacts: Any = None,
    ) -> dict[str, Any]:
        del limits, browser_version, visual_policy, visual_artifacts
        state.revision = "br_rejected_second_observation"
        state.last_observation_revision = state.revision
        state.refs = {"opaque_ref": "aria-ref=private"}
        return {
            "session_id": state.session_id,
            "page_id": state.page_id,
            "revision": state.revision,
            "creation_epoch": state.creation_epoch,
            "control_epoch": state.control_epoch,
            "url": "https://example.test/",
            "title": "Example",
            "snapshot": '- button "Example" [ref=opaque_ref]',
            "refs": [{"ref": "opaque_ref", "role": "button", "name": "Example"}],
            "load_state": "loaded",
            "access_state": "available",
            "access": None,
            "idle_timeout_seconds": 60,
            "truncation_reasons": [],
            "backend_identity": {
                "backend": "playwright",
                "backend_version": "1.62.0",
                "browser": "chromium",
                "browser_version": "test-chromium",
                "worker_protocol": "cayu.browser-session.v4",
                "worker_version": "19",
            },
        }

    monkeypatch.setattr(_browser_guest, "_interactive_observation", observation)

    async def scenario() -> None:
        page = _Page()
        context = _Context()
        limits = _interactive_limits(
            max_refs=1,
            max_refs_per_page=max_refs_per_page,
            max_total_refs=max_total_refs,
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("observe", limits=limits).__dict__,
                "page_id": "bp_root",
            }
        )
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = context
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            revision="br_first_observation",
            last_observation_revision="br_first_observation",
            last_operation_id_sha256=hashlib.sha256(b"navigate-1").hexdigest(),
            operation_count=1,
            observation_count=1,
            ref_count=1,
            refs={"old_ref": "aria-ref=old"},
            public_url="https://example.test/",
        )
        daemon.pages[state.page_id] = state
        daemon.active_page_id = state.page_id
        daemon.total_page_creations = 1
        daemon.total_operations = 1
        daemon.total_observations = 1
        daemon.total_refs = 1

        result = await daemon.execute(request)

        assert result["error"] == "resource_exhausted"
        assert result["allocation_disposition"] == "retired"
        assert page.close_calls == 1
        assert context.close_calls == 1
        assert daemon.pages == {}

    asyncio.run(scenario())


def test_interactive_guest_child_close_cancellation_falls_back_to_context() -> None:
    class _Page:
        async def close(self) -> None:
            raise asyncio.CancelledError("child close cancelled")

    class _Context:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        owner = asyncio.current_task()
        assert owner is not None
        cancellation_requests = owner.cancelling()
        context = _Context()
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = context
        state = _browser_guest._InteractivePage(
            page=_Page(),
            session_id="bs_test",
            page_id="bp_test",
            cdp=_BoundedSnapshotCdp(),
        )

        daemon._schedule_response_limit_abort(state)
        abort_task = state.limit_abort_task
        assert abort_task is not None
        assert await abort_task is True
        assert context.closed is True
        assert daemon.context is None
        assert owner.cancelling() == cancellation_requests

    asyncio.run(scenario())


def test_interactive_guest_shutdown_settles_background_limit_abort_first() -> None:
    class _Page:
        def __init__(self) -> None:
            self.close_started = asyncio.Event()
            self.allow_close = asyncio.Event()

        async def close(self) -> None:
            self.close_started.set()
            await self.allow_close.wait()

    class _Context:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        page = _Page()
        context = _Context()
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = context
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
        )
        daemon.pages[state.page_id] = state
        daemon._schedule_response_limit_abort(state)
        await page.close_started.wait()

        close_task = asyncio.create_task(daemon.close(timeout_seconds=1))
        await asyncio.sleep(0)
        assert context.closed is False

        page.allow_close.set()
        assert await close_task is True
        assert context.closed is True
        assert daemon.pages == {}

    asyncio.run(scenario())


def test_interactive_guest_idle_expiry_waits_for_active_operation() -> None:
    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.idle_timeout_seconds = 0.01
        daemon.last_activity = asyncio.get_running_loop().time()
        await daemon.lock.acquire()
        waiter = asyncio.create_task(_browser_guest._wait_for_interactive_shutdown(daemon))
        await asyncio.sleep(0.03)
        assert waiter.done() is False

        daemon.last_activity = asyncio.get_running_loop().time()
        daemon.lock.release()
        await asyncio.sleep(0)
        assert waiter.done() is False
        daemon.close_requested.set()
        await waiter

    asyncio.run(scenario())


def test_interactive_guest_idle_expiry_rejects_queued_operation() -> None:
    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.idle_timeout_seconds = 1
        daemon.last_activity = 0.0

        await _browser_guest._wait_for_interactive_shutdown(daemon)

        assert daemon.closing is True
        assert daemon.idle_expired is True
        with pytest.raises(RuntimeError, match="session_closed"):
            await daemon.execute(_interactive_request("observe"))

    asyncio.run(scenario())


def test_interactive_guest_idle_retirement_marker_releases_parent_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def no_response(socket_path: Path, raw: Any) -> None:
        del socket_path, raw
        return None

    monkeypatch.setattr(_browser_guest.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        _browser_guest.importlib.metadata,
        "version",
        lambda package: _browser_guest.PLAYWRIGHT_VERSION,
    )
    ca_fixture = tmp_path / "ca.pem"
    ca_fixture.write_bytes(b"fixture public CA")
    monkeypatch.setattr(
        _browser_guest,
        "_proxy_and_ca",
        lambda: ("http://proxy.test:8080", ca_fixture),
    )
    monkeypatch.setattr(_browser_guest, "_interactive_send", no_response)
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_ROOT", tmp_path)
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_CONNECT_SECONDS", 0.01)

    retired_path = _browser_guest._interactive_retired_path("bs_test")
    assert _browser_guest._record_interactive_retirement("bs_test") is True
    retired = asyncio.run(
        _browser_guest._run_interactive_request(_interactive_raw_request("observe"))
    )
    retired_path.unlink()
    uncertain = asyncio.run(
        _browser_guest._run_interactive_request(_interactive_raw_request("observe"))
    )

    assert retired["error"] == "allocation_lost"
    assert retired["allocation_disposition"] == "retired"
    assert uncertain["error"] == "session_closed"
    assert uncertain["allocation_disposition"] == "uncertain"


def test_interactive_guest_requires_durable_marker_before_reporting_retirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    response = _browser_guest._interactive_error_payload(
        _browser_guest._GuestFailure(
            "allocation_lost",
            allocation_disposition="retired",
        )
    )

    async def retired_response(socket_path: Path, raw: Any) -> dict[str, Any]:
        del socket_path, raw
        return response

    monkeypatch.setattr(_browser_guest.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        _browser_guest.importlib.metadata,
        "version",
        lambda package: _browser_guest.PLAYWRIGHT_VERSION,
    )
    ca_fixture = tmp_path / "ca.pem"
    ca_fixture.write_bytes(b"fixture public CA")
    monkeypatch.setattr(
        _browser_guest,
        "_proxy_and_ca",
        lambda: ("http://proxy.test:8080", ca_fixture),
    )
    monkeypatch.setattr(_browser_guest, "_interactive_send", retired_response)
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_ROOT", tmp_path)
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_CONNECT_SECONDS", 0.0)

    missing = asyncio.run(
        _browser_guest._run_interactive_request(_interactive_raw_request("observe"))
    )
    assert missing["error"] == "cleanup_failed"
    assert missing["allocation_disposition"] == "uncertain"

    assert _browser_guest._record_interactive_retirement("bs_test") is True
    settled = asyncio.run(
        _browser_guest._run_interactive_request(_interactive_raw_request("observe"))
    )
    assert settled == response


def test_interactive_guest_retirement_markers_do_not_alias_digest_prefixes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    colliding: tuple[str, str] | None = None
    by_prefix: dict[str, str] = {}
    for index in range(1_000):
        session_id = f"bs_retirement_collision_{index}"
        prefix = _browser_guest._interactive_retirement_token(session_id)[:3]
        previous = by_prefix.setdefault(prefix, session_id)
        if previous != session_id:
            colliding = (previous, session_id)
            break
    assert colliding is not None
    first, second = colliding
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_ROOT", tmp_path)

    assert _browser_guest._interactive_retired_path(
        first
    ) != _browser_guest._interactive_retired_path(second)
    assert _browser_guest._record_interactive_retirement(first) is True
    assert _browser_guest._record_interactive_retirement(second) is True
    assert _browser_guest._interactive_retirement_is_recorded(first) is True
    assert _browser_guest._interactive_retirement_is_recorded(second) is True


@pytest.mark.parametrize("cleanup_ok", [True, False])
def test_interactive_guest_startup_exit_records_only_settled_retirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cleanup_ok: bool,
) -> None:
    close_calls = 0

    class _StartupFailureDaemon:
        def __init__(self, session_id: str) -> None:
            assert session_id == "bs_test"

        async def start(self) -> None:
            raise RuntimeError("private startup failure")

        async def close(self) -> bool:
            nonlocal close_calls
            close_calls += 1
            return cleanup_ok

    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_ROOT", tmp_path)
    monkeypatch.setattr(_browser_guest, "_InteractiveDaemon", _StartupFailureDaemon)

    with pytest.raises(RuntimeError, match="private startup failure"):
        asyncio.run(_browser_guest._interactive_daemon_main("bs_test"))

    assert close_calls == 1
    assert _browser_guest._interactive_retirement_is_recorded("bs_test") is cleanup_ok


def test_interactive_guest_startup_request_consumes_settled_retirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def no_response(socket_path: Path, raw: Any) -> None:
        del socket_path, raw
        return None

    async def retire_during_start(session_id: str, socket_path: Path) -> None:
        del socket_path
        assert _browser_guest._record_interactive_retirement(session_id) is True

    monkeypatch.setattr(_browser_guest.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        _browser_guest.importlib.metadata,
        "version",
        lambda package: _browser_guest.PLAYWRIGHT_VERSION,
    )
    ca_fixture = tmp_path / "ca.pem"
    ca_fixture.write_bytes(b"fixture public CA")
    monkeypatch.setattr(
        _browser_guest,
        "_proxy_and_ca",
        lambda: ("http://proxy.test:8080", ca_fixture),
    )
    monkeypatch.setattr(_browser_guest, "_interactive_send", no_response)
    monkeypatch.setattr(_browser_guest, "_start_interactive_daemon", retire_during_start)
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_ROOT", tmp_path)
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_CONNECT_SECONDS", 0.0)

    with pytest.raises(_browser_guest._GuestFailure, match="browser_unavailable") as raised:
        asyncio.run(_browser_guest._run_interactive_request(_interactive_raw_request("navigate")))

    assert raised.value.allocation_disposition == "retired"


def test_interactive_guest_settled_retirement_prevents_daemon_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    start_calls = 0

    async def no_response(socket_path: Path, raw: Any) -> None:
        del socket_path, raw
        return None

    async def unexpected_start(session_id: str, socket_path: Path) -> None:
        nonlocal start_calls
        del session_id, socket_path
        start_calls += 1

    monkeypatch.setattr(_browser_guest.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        _browser_guest.importlib.metadata,
        "version",
        lambda package: _browser_guest.PLAYWRIGHT_VERSION,
    )
    ca_fixture = tmp_path / "ca.pem"
    ca_fixture.write_bytes(b"fixture public CA")
    monkeypatch.setattr(
        _browser_guest,
        "_proxy_and_ca",
        lambda: ("http://proxy.test:8080", ca_fixture),
    )
    monkeypatch.setattr(_browser_guest, "_interactive_send", no_response)
    monkeypatch.setattr(_browser_guest, "_start_interactive_daemon", unexpected_start)
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_ROOT", tmp_path)
    assert _browser_guest._record_interactive_retirement("bs_test") is True

    with pytest.raises(_browser_guest._GuestFailure, match="browser_unavailable") as raised:
        asyncio.run(_browser_guest._run_interactive_request(_interactive_raw_request("navigate")))

    assert raised.value.allocation_disposition == "retired"
    assert start_calls == 0


def test_interactive_guest_late_startup_retirement_releases_parent_capacity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    delayed_retirement_tasks: list[asyncio.Task[None]] = []

    async def no_response(socket_path: Path, raw: Any) -> None:
        del socket_path, raw
        return None

    async def retire_after_connect_deadline(session_id: str, socket_path: Path) -> None:
        del socket_path

        async def publish() -> None:
            await asyncio.sleep(0.03)
            assert _browser_guest._record_interactive_retirement(session_id) is True

        delayed_retirement_tasks.append(asyncio.create_task(publish()))

    class _LateStartupRetirementRunner(_WireRunner):
        def __init__(self) -> None:
            self.exec_calls = 0

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            self.exec_calls += 1
            if self.exec_calls != 1:
                return await super().exec(command, **kwargs)
            assert command.argv == list(PINNED_BROWSER_SESSION_WORKLOAD.command)
            assert kwargs["timeout_s"] == 45
            raw = json.loads(kwargs["stdin"])
            try:
                response = await _browser_guest._run_interactive_request(raw)
            except _browser_guest._GuestFailure as exc:
                response = _browser_guest._interactive_error_payload(exc)
            return ExecResult(stdout=json.dumps(response))

    async def scenario() -> None:
        runner = _LateStartupRetirementRunner()
        ctx = _context(tmp_path).model_copy(update={"runner": runner})
        tool = BrowserSessionTool(
            expected_runner_candidate="wire-browser",
            max_sessions=1,
        )

        first = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/late-startup-failure",
                "operation_id": "navigate-late-startup-failure",
            },
        )
        replacement = await tool.run(
            ctx,
            {
                "operation": "navigate",
                "url": "https://example.test/replacement",
                "operation_id": "navigate-after-late-retirement",
            },
        )
        await asyncio.gather(*delayed_retirement_tasks)

        assert first.structured["error"] == "browser_unavailable"
        assert first.structured["execution"]["dispatch"] == "completed"
        assert replacement.is_error is False
        assert runner.exec_calls == 2

    monkeypatch.setattr(_browser_guest.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(
        _browser_guest.importlib.metadata,
        "version",
        lambda package: _browser_guest.PLAYWRIGHT_VERSION,
    )
    ca_fixture = tmp_path / "ca.pem"
    ca_fixture.write_bytes(b"fixture public CA")
    monkeypatch.setattr(
        _browser_guest,
        "_proxy_and_ca",
        lambda: ("http://proxy.test:8080", ca_fixture),
    )
    monkeypatch.setattr(_browser_guest, "_interactive_send", no_response)
    monkeypatch.setattr(
        _browser_guest,
        "_start_interactive_daemon",
        retire_after_connect_deadline,
    )
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_ROOT", tmp_path / "browser-sessions")
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_CONNECT_SECONDS", 0.01)
    monkeypatch.setattr(_browser_guest, "_INTERACTIVE_STARTUP_SETTLEMENT_SECONDS", 0.5)

    asyncio.run(scenario())


def test_interactive_guest_classifies_timeout_and_crash_separately() -> None:
    target_closed_type = type(
        "TargetClosedError",
        (RuntimeError,),
        {"__module__": "playwright._impl._errors"},
    )
    assert (
        _browser_guest._interactive_playwright_error("navigate", TimeoutError()).code
        == "navigation_timeout"
    )
    assert (
        _browser_guest._interactive_playwright_error("navigate", RuntimeError()).code
        == "browser_crash"
    )
    for operation in ("click", "download"):
        assert (
            _browser_guest._interactive_playwright_error(
                operation,
                target_closed_type(),
            ).code
            == "browser_crash"
        )
