"""Interactive browser guest popup admission, navigation guards and cleanup contracts."""

from __future__ import annotations

import asyncio
import hashlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from tests.core.browser_guest_support import (
    _BoundedSnapshotCdp,
    _configure_interactive_daemon_for_test,
    _interactive_limits,
    _interactive_request,
)

from cayu.tools import _browser_guest


def _arm_popup_effect_for_test(
    daemon: _browser_guest._InteractiveDaemon,
    opener: _browser_guest._InteractivePage,
) -> None:
    daemon.popup_effect_opener_page_id = opener.page_id
    daemon.popup_effect_opener_origin = daemon._current_page_origin(opener)


def test_interactive_guest_admits_switches_closes_and_tracks_popup_lineage() -> None:
    class _Page:
        def __init__(self, url: str) -> None:
            self.url = url
            self.closed = False
            self.handlers: dict[str, Any] = {}

        def on(self, event: str, callback: Any) -> None:
            self.handlers[event] = callback

        async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            assert state == "domcontentloaded"
            assert 0 < timeout <= 1000

        async def close(self) -> None:
            self.closed = True
            callback = self.handlers.get("close")
            if callback is not None:
                callback()

    class _Context:
        def __init__(self, root: _Page) -> None:
            self.root = root
            self.route_callback: Any = None
            self.closed = False
            self.init_scripts: list[str] = []

        async def add_init_script(self, script: str) -> None:
            self.init_scripts.append(script)

        async def route(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            self.route_callback = callback

        async def new_page(self) -> _Page:
            return self.root

        async def close(self) -> None:
            self.closed = True
            await self.root.close()

    class _Daemon(_browser_guest._InteractiveDaemon):
        def __init__(self, session_id: str, popup: _Page) -> None:
            super().__init__(session_id)
            self.popup = popup

        async def _configure_page(self, state, limits) -> None:
            del limits
            state.configured = True
            state.page.on("close", lambda: self._mark_page_closed(state, "closed_by_page"))
            state.page.on("crash", lambda: self._mark_page_crashed(state))

        async def _observe_page(self, state, limits):
            if (
                state.observation_count >= limits.max_observations_per_page
                or self.total_observations >= limits.max_total_observations
            ):
                raise _browser_guest._GuestFailure("resource_exhausted")
            state.observation_count += 1
            state.ref_count += 1
            self.total_observations += 1
            self.total_refs += 1
            state.revision = f"br_{state.page_id}_{state.observation_count}"
            state.last_observation_revision = state.revision
            state.refs = {f"ref_{state.page_id}": "internal"}
            state.title = "Root" if state.opener_page_id is None else "Popup"
            return {
                "session_id": state.session_id,
                "page_id": state.page_id,
                "revision": state.revision,
                "creation_epoch": state.creation_epoch,
                "control_epoch": state.control_epoch,
                "url": state.page.url,
                "title": state.title,
                "snapshot": f'- button "{state.title}" [ref={next(iter(state.refs))}]',
                "refs": [
                    {
                        "ref": next(iter(state.refs)),
                        "role": "button",
                        "name": state.title,
                    }
                ],
                "load_state": "loaded",
                "access_state": "available",
                "access": None,
                "idle_timeout_seconds": limits.idle_timeout_seconds,
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

        async def _execute_page(self, state, request):
            if request.operation == "click":
                _arm_popup_effect_for_test(self, state)
                self._register_popup_candidate(state, self.popup)
            observation = await self._observe_page(state, request.limits)
            return _browser_guest._interactive_success_payload(observation)

    async def scenario() -> None:
        root = _Page("https://example.test/root")
        popup = _Page("https://example.test/popup")
        daemon = _Daemon("bs_test", popup)
        daemon.context = _Context(root)
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        policy = _browser_guest._InteractivePopupPolicy(
            mode="same_origin",
            allowed_operations=("click",),
            allowed_opener_origins=(),
            allowed_destination_origins=(),
        )

        def request(operation: str, **updates: Any) -> Any:
            return _browser_guest._InteractiveRequest(
                **{
                    **_interactive_request(operation, limits=limits).__dict__,
                    "multi_page": True,
                    "popup_policy": policy,
                    **updates,
                }
            )

        opened = await daemon.execute(request("navigate"))
        assert len(daemon.context.init_scripts) == 1
        assert "__cayuSetPopupAdmission" in daemon.context.init_scripts[0]
        assert "__CAYU_POPUP_CONFIGURATION__" not in daemon.context.init_scripts[0]
        root_id = opened["observation"]["page_id"]
        root_state = daemon.pages[root_id]
        clicked = await daemon.execute(
            request(
                "click",
                expected_revision=root_state.revision,
                expected_control_epoch=root_state.control_epoch,
                ref=next(iter(root_state.refs)),
            )
        )
        assert clicked["kind"] == "success"
        assert len(clicked["page_delta"]["created_page_ids"]) == 1
        popup_id = clicked["page_delta"]["created_page_ids"][0]
        assert clicked["page_delta"]["admitted_page_ids"] == [popup_id]
        popup_summary = next(
            item for item in clicked["page_set"]["pages"] if item["page_id"] == popup_id
        )
        assert popup_summary["opener_page_id"] == root_id
        assert (
            popup_summary["creating_operation_id_sha256"] == hashlib.sha256(b"click-1").hexdigest()
        )
        assert popup_summary["lifecycle"] == "background"
        assert popup_summary["page_id"] != str(id(popup))

        switched = await daemon.execute(
            request(
                "switch_page",
                page_id=popup_id,
                operation_id="switch-popup",
            )
        )
        assert switched["observation"]["page_id"] == popup_id
        assert switched["page_set"]["active_page_id"] == popup_id
        assert daemon.pages[root_id].lifecycle == "background"
        assert not daemon.pages[root_id].refs

        popup.handlers["crash"]()
        listed = await daemon.execute(
            request(
                "list_pages",
                page_id=None,
                operation_id="list-after-crash",
            )
        )
        assert listed["page_set"]["active_page_id"] == root_id
        crashed = next(item for item in listed["page_set"]["pages"] if item["page_id"] == popup_id)
        assert crashed["lifecycle"] == "crashed"
        assert crashed["terminal_reason"] == "browser_crash"

        closed = await daemon.execute(
            request(
                "close_page",
                page_id=root_id,
                operation_id="close-root",
            )
        )
        assert closed["kind"] == "success"
        assert closed["page_set"]["active_page_id"] is None
        assert closed["page_delta"]["closed_page_ids"] == [root_id]
        assert root.closed is True

    asyncio.run(scenario())


def test_interactive_guest_refuses_cross_origin_popup_and_bounds_cleanup() -> None:
    class _Page:
        def __init__(self, url: str) -> None:
            self.url = url
            self.close_calls = 0

        async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            del state, timeout

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = types.SimpleNamespace()
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        policy = _browser_guest._InteractivePopupPolicy(
            mode="same_origin",
            allowed_operations=("click",),
            allowed_opener_origins=(),
            allowed_destination_origins=(),
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "multi_page": True,
                "popup_policy": policy,
            }
        )
        await _configure_interactive_daemon_for_test(daemon, request)
        opener_page = _Page("https://example.test/root")
        opener = _browser_guest._InteractivePage(
            page=opener_page,
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            public_url=opener_page.url,
            revision="br_root",
        )
        daemon.pages[opener.page_id] = opener
        daemon.active_page_id = opener.page_id
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()
        _arm_popup_effect_for_test(daemon, opener)
        popup = _Page("https://blocked.example/popup")
        candidate = daemon._register_popup_candidate(opener, popup)
        assert candidate is not None
        candidate.configured = True

        await daemon._settle_operation_popups(request, daemon.active_delta)

        assert popup.close_calls == 1
        assert candidate.lifecycle == "closed"
        assert candidate.terminal_reason == "destination_denied"
        assert daemon.active_delta.refused == [
            {
                "page_id": candidate.page_id,
                "opener_page_id": opener.page_id,
                "reason": "destination_denied",
            }
        ]
        assert daemon.cleanup_operation_count == 1

    asyncio.run(scenario())


def test_interactive_guest_reports_popup_runtime_failure_after_bounded_cleanup() -> None:
    class _Page:
        url = "https://example.test/popup"

        def __init__(self) -> None:
            self.close_calls = 0

        async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            del state, timeout
            raise RuntimeError("private popup runtime failure")

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = types.SimpleNamespace()
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        policy = _browser_guest._InteractivePopupPolicy(
            mode="same_origin",
            allowed_operations=("click",),
            allowed_opener_origins=(),
            allowed_destination_origins=(),
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "multi_page": True,
                "popup_policy": policy,
            }
        )
        await _configure_interactive_daemon_for_test(daemon, request)
        opener = _browser_guest._InteractivePage(
            page=_Page(),
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            public_url="https://example.test/root",
            revision="br_root",
        )
        daemon.pages[opener.page_id] = opener
        daemon.active_page_id = opener.page_id
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()
        _arm_popup_effect_for_test(daemon, opener)
        popup = _Page()
        candidate = daemon._register_popup_candidate(opener, popup)
        assert candidate is not None
        candidate.configured = True

        with pytest.raises(_browser_guest._GuestFailure, match="browser_crash"):
            await daemon._settle_operation_popups(request, daemon.active_delta)

        assert popup.close_calls == 1
        assert candidate.lifecycle == "closed"
        assert candidate.terminal_reason == "browser_crash"
        assert candidate.page_id in daemon.active_delta.closed_page_ids
        assert daemon.active_delta.refused == []

    asyncio.run(scenario())


def test_interactive_guest_refuses_popup_with_denied_subresource() -> None:
    class _Page:
        url = "https://example.test/popup"

        def __init__(self) -> None:
            self.close_calls = 0

        async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            del state, timeout

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = types.SimpleNamespace()
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        policy = _browser_guest._InteractivePopupPolicy(
            mode="same_origin",
            allowed_operations=("click",),
            allowed_opener_origins=(),
            allowed_destination_origins=(),
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "multi_page": True,
                "popup_policy": policy,
            }
        )
        await _configure_interactive_daemon_for_test(daemon, request)
        opener = _browser_guest._InteractivePage(
            page=_Page(),
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            public_url="https://example.test/root",
            revision="br_root",
        )
        daemon.pages[opener.page_id] = opener
        daemon.active_page_id = opener.page_id
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()
        _arm_popup_effect_for_test(daemon, opener)
        popup = _Page()
        candidate = daemon._register_popup_candidate(opener, popup)
        assert candidate is not None
        candidate.configured = True
        # The response-routing owner records this sticky evidence when any
        # popup subresource is denied, even though the top-level URL is valid.
        candidate.denied_code = "destination_denied"

        await daemon._settle_operation_popups(request, daemon.active_delta)

        assert popup.close_calls == 1
        assert candidate.lifecycle == "closed"
        assert candidate.terminal_reason == "destination_denied"
        assert daemon.active_delta.admitted_page_ids == set()
        assert daemon.active_delta.refused == [
            {
                "page_id": candidate.page_id,
                "opener_page_id": opener.page_id,
                "reason": "destination_denied",
            }
        ]

    asyncio.run(scenario())


def test_interactive_guest_classifies_popup_resource_failure_as_capacity_refusal() -> None:
    class _Page:
        url = "https://example.test/popup"

        def __init__(self) -> None:
            self.close_calls = 0

        async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            del state, timeout

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = types.SimpleNamespace()
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        policy = _browser_guest._InteractivePopupPolicy(
            mode="same_origin",
            allowed_operations=("click",),
            allowed_opener_origins=(),
            allowed_destination_origins=(),
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "multi_page": True,
                "popup_policy": policy,
            }
        )
        await _configure_interactive_daemon_for_test(daemon, request)
        opener = _browser_guest._InteractivePage(
            page=_Page(),
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            public_url="https://example.test/root",
            revision="br_root",
        )
        daemon.pages[opener.page_id] = opener
        daemon.active_page_id = opener.page_id
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()
        _arm_popup_effect_for_test(daemon, opener)
        popup = _Page()
        candidate = daemon._register_popup_candidate(opener, popup)
        assert candidate is not None
        candidate.configured = True
        candidate.limit_exceeded = True
        candidate.limit_error_code = "resource_exhausted"

        with pytest.raises(_browser_guest._GuestFailure, match="resource_exhausted"):
            await daemon._settle_operation_popups(request, daemon.active_delta)

        assert popup.close_calls == 1
        assert candidate.lifecycle == "closed"
        assert candidate.terminal_reason == "capacity_refused"
        assert daemon.active_delta.admitted_page_ids == set()
        assert daemon.active_delta.refused == [
            {
                "page_id": candidate.page_id,
                "opener_page_id": opener.page_id,
                "reason": "capacity_refused",
            }
        ]

    asyncio.run(scenario())


def test_interactive_guest_settles_popup_after_action_failure() -> None:
    class _Page:
        def __init__(self, url: str) -> None:
            self.url = url
            self.close_calls = 0

        async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            del state, timeout

        async def close(self) -> None:
            self.close_calls += 1

        def is_closed(self) -> bool:
            return False

    class _Daemon(_browser_guest._InteractiveDaemon):
        popup: _Page | None = None

        async def _execute_page(
            self,
            state: _browser_guest._InteractivePage,
            request: _browser_guest._InteractiveRequest,
        ) -> dict[str, object]:
            state.revision = None
            state.refs.clear()
            popup = _Page("https://example.test/popup")
            self.popup = popup
            _arm_popup_effect_for_test(self, state)
            candidate = self._register_popup_candidate(state, popup)
            assert candidate is not None
            candidate.configured = True
            raise RuntimeError("action failed after popup creation")

    async def scenario() -> None:
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        policy = _browser_guest._InteractivePopupPolicy(
            mode="same_origin",
            allowed_operations=("click",),
            allowed_opener_origins=(),
            allowed_destination_origins=(),
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "expected_revision": "br_root",
                "expected_control_epoch": 1,
                "ref": "ref_button",
                "multi_page": True,
                "popup_policy": policy,
            }
        )
        daemon = _Daemon("bs_test")
        daemon.context = types.SimpleNamespace()
        daemon.browser = types.SimpleNamespace(is_connected=lambda: True)
        await _configure_interactive_daemon_for_test(daemon, request)
        root_page = _Page("https://example.test/root")
        root = _browser_guest._InteractivePage(
            page=root_page,
            session_id="bs_test",
            page_id=request.page_id or "bp_root",
            lifecycle="active",
            public_url=root_page.url,
            revision="br_root",
            refs={"ref_button": "internal"},
        )
        daemon.pages[root.page_id] = root
        daemon.active_page_id = root.page_id
        daemon.total_page_creations = 1

        response = await daemon._execute_locked(request)

        assert response["kind"] == "error"
        assert response["error"] == "actionability_failed"
        assert len(response["page_delta"]["created_page_ids"]) == 1
        popup_id = response["page_delta"]["created_page_ids"][0]
        assert response["page_delta"]["admitted_page_ids"] == [popup_id]
        assert daemon.pages[popup_id].lifecycle == "background"
        assert all(page["lifecycle"] != "provisional" for page in response["page_set"]["pages"])

    asyncio.run(scenario())


def test_interactive_guest_popup_burst_uses_one_owned_retirement_task() -> None:
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

    class _Page:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}

        def set_default_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def set_default_navigation_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def on(self, event: str, callback: Any) -> None:
            self.handlers[event] = callback

        async def close(self) -> None:
            return None

    class _Context:
        def __init__(self, cdp: _Cdp) -> None:
            self.cdp = cdp
            self.close_calls = 0

        async def route(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            assert callable(callback)

        async def new_cdp_session(self, page: _Page) -> _Cdp:
            del page
            return self.cdp

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
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
        request = _interactive_request("observe")
        await _configure_interactive_daemon_for_test(daemon, request)
        await daemon._configure_page(state, request.limits)

        tasks: list[asyncio.Task[bool]] = []
        for _ in range(100):
            page.handlers["popup"](object())
            assert state.limit_abort_task is not None
            tasks.append(state.limit_abort_task)
        assert len({id(task) for task in tasks}) == 1

        result = await daemon.execute(request)

        assert result["error"] == "resource_exhausted"
        assert result["allocation_disposition"] == "retired"
        assert context.close_calls == 1
        assert daemon.pages == {}

    asyncio.run(scenario())


def test_interactive_guest_multipage_popup_burst_retires_shared_context() -> None:
    class _Page:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    class _Context:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    async def scenario() -> None:
        context = _Context()
        opener_page = _Page()
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = context
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "multi_page": True,
                "popup_policy": _browser_guest._InteractivePopupPolicy(
                    mode="same_origin",
                    allowed_operations=("click",),
                    allowed_opener_origins=(),
                    allowed_destination_origins=(),
                ),
            }
        )
        opener = _browser_guest._InteractivePage(
            page=opener_page,
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            revision="br_root",
            public_url="https://example.test/root",
        )
        daemon.pages[opener.page_id] = opener
        daemon.active_page_id = opener.page_id
        daemon.total_page_creations = 1
        daemon.configuration_limits = limits
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()
        _arm_popup_effect_for_test(daemon, opener)

        popups = [_Page() for _ in range(100)]
        for popup in popups:
            daemon._register_popup_candidate(opener, popup)

        cleanup = daemon.popup_cleanup_task
        assert cleanup is not None
        assert await cleanup is True
        assert daemon.closing is True
        assert daemon.close_after_response is True
        assert context.close_calls == 1
        assert daemon.total_page_creations == 2
        assert len(daemon.pages) == 2
        assert len(daemon.active_delta.refused) <= limits.max_page_creations_per_operation

    asyncio.run(scenario())


def test_interactive_guest_blocks_popup_creation_before_page_scripts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Context:
        def __init__(self) -> None:
            self.init_scripts: list[str] = []
            self.route_callback: Any = None

        async def add_init_script(self, script: str) -> None:
            self.init_scripts.append(script)

        async def route(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            self.route_callback = callback

        async def close(self) -> None:
            return None

    class _Browser:
        version = "test-chromium"

        def __init__(self, context: _Context) -> None:
            self.context = context

        async def new_context(self, **kwargs: Any) -> _Context:
            assert kwargs["service_workers"] == "block"
            return self.context

        def on(self, event: str, callback: Any) -> None:
            assert event == "disconnected"
            assert callable(callback)

        def is_connected(self) -> bool:
            return True

        async def close(self) -> None:
            return None

    class _Chromium:
        def __init__(self, browser: _Browser) -> None:
            self.browser = browser
            self.launch_kwargs: dict[str, Any] | None = None

        async def launch(self, **kwargs: Any) -> _Browser:
            self.launch_kwargs = kwargs
            return self.browser

    class _Playwright:
        def __init__(self, chromium: _Chromium) -> None:
            self.chromium = chromium

        async def stop(self) -> None:
            return None

    class _Starter:
        def __init__(self, playwright: _Playwright) -> None:
            self.playwright = playwright

        async def start(self) -> _Playwright:
            return self.playwright

    async def install_ca(home: Path, ca_path: str) -> None:
        del home, ca_path

    profile_home = tmp_path / "cayu-browser-profile-test"
    profile_home.mkdir()
    profile_owner = types.SimpleNamespace(
        home=profile_home,
        process=types.SimpleNamespace(poll=lambda: 0),
    )
    profile_cleanup_calls: list[tuple[object, float]] = []

    async def start_profile_owner(
        *,
        timeout_seconds: float,
        startup_timeout_seconds: float | None = None,
        existing_home: Path | None = None,
    ) -> object:
        assert timeout_seconds == _browser_guest._MAX_PROFILE_CLEANUP_RESERVE_SECONDS
        assert startup_timeout_seconds is None
        assert existing_home is None
        return profile_owner

    async def cleanup_profile_owner(
        owner: object,
        *,
        timeout_seconds: float,
    ) -> tuple[BaseException, ...]:
        profile_cleanup_calls.append((owner, timeout_seconds))
        return ()

    context = _Context()
    browser = _Browser(context)
    chromium = _Chromium(browser)
    playwright = _Playwright(chromium)
    playwright_api = types.ModuleType("playwright.async_api")
    playwright_api.async_playwright = lambda: _Starter(playwright)  # type: ignore[attr-defined]
    playwright_module = types.ModuleType("playwright")
    playwright_module.async_api = playwright_api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "playwright", playwright_module)
    monkeypatch.setitem(sys.modules, "playwright.async_api", playwright_api)
    ca_fixture = tmp_path / "ca.pem"
    ca_fixture.write_bytes(b"fixture public CA")
    monkeypatch.setattr(
        _browser_guest,
        "_proxy_and_ca",
        lambda: ("http://proxy.test:8080", ca_fixture),
    )
    monkeypatch.setattr(_browser_guest, "_sanitize_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(_browser_guest, "_install_browser_ca", install_ca)
    monkeypatch.setattr(
        _browser_guest,
        "_start_temporary_profile_owner",
        start_profile_owner,
    )
    monkeypatch.setattr(
        _browser_guest,
        "_cleanup_temporary_profile_owner",
        cleanup_profile_owner,
    )

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        await daemon.start()
        assert context.init_scripts == []
        await daemon._ensure_configuration(_interactive_request("navigate"))

        assert chromium.launch_kwargs is not None
        assert chromium.launch_kwargs["ignore_default_args"] == ["--disable-popup-blocking"]
        assert context.init_scripts == [
            _browser_guest._interactive_popup_guard(daemon.popup_guard_token)
        ]
        assert "__CAYU_POPUP_CONFIGURATION__" not in context.init_scripts[0]
        assert daemon.popup_guard_token in context.init_scripts[0]
        assert "__cayuSetPopupAdmission" in context.init_scripts[0]
        assert 'Object.defineProperty(window, "open"' in context.init_scripts[0]
        assert 'Object.defineProperty(Window.prototype, "open"' in context.init_scripts[0]
        assert 'element.ownerDocument, ["base[target]"]' in context.init_scripts[0]
        assert 'attribute(submitter, "formtarget")' in context.init_scripts[0]
        assert 'window.addEventListener("click"' in context.init_scripts[0]
        assert 'parsed.protocol !== "https:"' in context.init_scripts[0]
        assert '=== "_blank"' in context.init_scripts[0]
        assert "recordBlocked(2)" in context.init_scripts[0]
        assert callable(context.route_callback)

        assert await daemon.close() is True
        assert len(profile_cleanup_calls) == 1
        assert profile_cleanup_calls[0][0] is profile_owner
        assert profile_cleanup_calls[0][1] > 0

    asyncio.run(scenario())


def test_interactive_guest_popup_guard_bounds_one_effect_before_target_admission() -> None:
    class _Popup:
        url = "https://example.test/popup"

        async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            assert state == "domcontentloaded"
            assert 0 < timeout <= 1000

        async def close(self) -> None:
            return None

    class _Page:
        def __init__(self) -> None:
            self.allowance = 0
            self.admitted_urls: list[str] = []
            self.blocked = 0
            self.evaluations: list[tuple[str, int]] = []
            self.daemon: _Daemon | None = None
            self.state: _browser_guest._InteractivePage | None = None

        async def evaluate(self, expression: str, values: list[object]) -> dict[str, object]:
            assert "__cayuSetPopupAdmission" in expression
            token, allowance = values
            assert type(token) is str
            assert type(allowance) is int
            previous_blocked = self.blocked
            previous_urls = self.admitted_urls
            self.allowance = allowance
            self.admitted_urls = []
            self.blocked = 0
            self.evaluations.append((token, allowance))
            return {"blocked": previous_blocked, "urls": previous_urls}

        def locator(self, selector: str) -> _Locator:
            assert selector == "aria-ref=internal"
            return _Locator(self)

    class _Locator:
        def __init__(self, page: _Page) -> None:
            self.page = page

        async def element_handle(self) -> _Locator:
            return self

        async def click(self) -> None:
            assert self.page.daemon is not None
            assert self.page.state is not None
            assert self.page.allowance == 1
            self.page.allowance -= 1
            self.page.admitted_urls.append("https://example.test/popup")
            asyncio.get_running_loop().call_later(
                0.01,
                self.page.daemon._observe_popup_candidate,
                self.page.state,
                _Popup(),
            )
            # A second synchronous creation attempt is stopped inside the
            # pre-document guard rather than materialized as another target.
            if self.page.allowance == 0:
                self.page.blocked = 1

    class _Daemon(_browser_guest._InteractiveDaemon):
        async def _configure_page(self, state: Any, limits: Any) -> None:
            del limits
            state.configured = True

        async def _observe_page(self, state: Any, limits: Any) -> dict[str, Any]:
            del limits
            state.revision = "br_after_click"
            state.last_observation_revision = state.revision
            return {
                "session_id": state.session_id,
                "page_id": state.page_id,
                "revision": state.revision,
                "creation_epoch": state.creation_epoch,
                "control_epoch": state.control_epoch,
                "url": "https://example.test/root",
                "title": "Root",
                "snapshot": "- document",
                "refs": [],
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

    async def scenario() -> None:
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        policy = _browser_guest._InteractivePopupPolicy(
            mode="same_origin",
            allowed_operations=("click",),
            allowed_opener_origins=(),
            allowed_destination_origins=(),
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "expected_revision": "br_before_click",
                "expected_control_epoch": 1,
                "ref": "ref_button",
                "multi_page": True,
                "popup_policy": policy,
            }
        )
        page = _Page()
        daemon = _Daemon("bs_test")
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_root",
            creation_epoch=1,
            control_epoch=2,
            lifecycle="active",
            revision="br_before_click",
            refs={"ref_button": "internal"},
            public_url="https://example.test/root",
        )
        page.daemon = daemon
        page.state = state
        daemon.pages[state.page_id] = state
        daemon.active_page_id = state.page_id
        daemon.total_page_creations = 1
        daemon.configuration_limits = limits
        daemon.configuration_multi_page = True
        daemon.configuration_popup_policy = policy
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()

        response = await daemon._execute_page(state, request)
        await daemon._settle_operation_popups(request, daemon.active_delta)

        assert response["kind"] == "success"
        assert [allowance for _, allowance in page.evaluations] == [1, 0]
        assert all(token == daemon.popup_guard_token for token, _ in page.evaluations)
        assert len(daemon.active_delta.created_page_ids) == 1
        assert daemon.active_delta.admitted_page_ids == daemon.active_delta.created_page_ids
        assert len(daemon.active_delta.refused) == 1
        assert daemon.active_delta.refused[0]["opener_page_id"] == state.page_id
        assert daemon.active_delta.refused[0]["reason"] == "capacity_refused"
        assert daemon.total_page_creations == 2

    asyncio.run(scenario())


def test_interactive_guest_popup_guard_classifies_unsafe_target_as_policy_denial() -> None:
    class _Page:
        async def evaluate(self, expression: str, values: list[object]) -> dict[str, object]:
            assert "__cayuSetPopupAdmission" in expression
            assert values[1] == 0
            return {"blocked": 2, "urls": []}

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click").__dict__,
                "multi_page": True,
                "popup_policy": _browser_guest._InteractivePopupPolicy(
                    mode="same_origin",
                    allowed_operations=("click",),
                    allowed_opener_origins=(),
                    allowed_destination_origins=(),
                ),
            }
        )
        state = _browser_guest._InteractivePage(
            page=_Page(),
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
        )
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()

        await daemon._end_popup_effect(state, None)

        assert len(daemon.active_delta.refused) == 1
        assert daemon.active_delta.refused[0]["opener_page_id"] == "bp_root"
        assert daemon.active_delta.refused[0]["reason"] == "policy_denied"

    asyncio.run(scenario())


def test_interactive_guest_access_block_preserves_authority_over_popup_guard_cleanup() -> None:
    access = {
        "schema_version": 1,
        "outcome": "bot_challenge",
        "source": "browser_response",
        "signal": "challenge_header",
        "destination_fingerprint": "a" * 64,
        "status_code": 200,
        "retry_after_seconds": None,
        "retry_after_unrepresentable": False,
    }

    class _Page:
        def __init__(self, state: _browser_guest._InteractivePage) -> None:
            self.state = state
            self.evaluations = 0

        async def evaluate(self, expression: str, values: list[object]) -> dict[str, object]:
            assert "__cayuSetPopupAdmission" in expression
            assert values[1] == 1
            self.evaluations += 1
            return {"blocked": 0, "urls": []}

        async def goto(self, *_args: object, **_kwargs: object) -> None:
            self.state.access_evidence = access
            raise RuntimeError("page was intentionally blocked")

    async def scenario() -> None:
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("navigate", limits=limits).__dict__,
                "multi_page": True,
                "popup_policy": _browser_guest._InteractivePopupPolicy(
                    mode="same_origin",
                    allowed_operations=("navigate",),
                    allowed_opener_origins=(),
                    allowed_destination_origins=(),
                ),
            }
        )
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        state = _browser_guest._InteractivePage(
            page=None,
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            public_url="https://example.test/root",
        )
        page = _Page(state)
        state.page = page
        daemon.pages[state.page_id] = state
        daemon.active_page_id = state.page_id
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()

        with pytest.raises(_browser_guest._GuestFailure) as exc_info:
            await daemon._execute_page(state, request)

        assert exc_info.value.code == "access_blocked"
        assert exc_info.value.access == access
        assert page.evaluations == 1

    asyncio.run(scenario())


def test_interactive_guest_popup_guard_failure_fences_the_page_set() -> None:
    class _Page:
        async def evaluate(self, expression: str, values: list[object]) -> int:
            del expression, values
            raise RuntimeError("hostile popup-guard failure")

    async def scenario() -> None:
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        policy = _browser_guest._InteractivePopupPolicy(
            mode="same_origin",
            allowed_operations=("click",),
            allowed_opener_origins=(),
            allowed_destination_origins=(),
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "multi_page": True,
                "popup_policy": policy,
            }
        )
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        state = _browser_guest._InteractivePage(
            page=_Page(),
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            revision="br_before_click",
            refs={"ref_button": "internal"},
            public_url="https://example.test/root",
        )
        daemon.pages[state.page_id] = state
        daemon.active_page_id = state.page_id
        daemon.total_page_creations = 1
        daemon.configuration_limits = limits
        daemon.configuration_multi_page = True
        daemon.configuration_popup_policy = policy
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()

        with pytest.raises(_browser_guest._GuestFailure, match="browser_crash"):
            await daemon._begin_popup_effect(state, request)

        assert state.lifecycle == "uncertain"
        assert state.revision is None
        assert state.refs == {}
        assert daemon.active_page_id is None
        assert daemon.closing is True
        assert daemon.close_after_response is True
        assert state.terminal_reason == "popup_guard_failed"

    asyncio.run(scenario())


def test_interactive_guest_denies_popup_redirect_before_destination_dispatch() -> None:
    class _Cdp:
        def __init__(self) -> None:
            self.handlers: dict[str, Any] = {}
            self.failed_requests: list[dict[str, str]] = []
            self.closed_targets: list[str] = []

        async def send(
            self,
            method: str,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            if method == "Network.enable":
                return {}
            if method == "Page.getFrameTree":
                return {"frameTree": {"frame": {"id": "frame-main"}}}
            if method == "Fetch.enable":
                return {}
            if method == "Fetch.failRequest":
                assert params is not None
                self.failed_requests.append(params)
                return {}
            if method == "Target.getTargetInfo":
                return {"targetInfo": {"targetId": "target-popup"}}
            if method == "Target.closeTarget":
                assert params is not None
                self.closed_targets.append(params["targetId"])
                return {"success": True}
            raise AssertionError(f"unexpected CDP method: {method}")

        def on(self, event: str, callback: Any) -> None:
            self.handlers[event] = callback

    class _Page:
        def __init__(self) -> None:
            self.main_frame = object()
            self.handlers: dict[str, Any] = {}

        def set_default_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def set_default_navigation_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def on(self, event: str, callback: Any) -> None:
            self.handlers[event] = callback

    class _Context:
        def __init__(self, cdp: _Cdp) -> None:
            self.cdp = cdp

        async def new_cdp_session(self, page: _Page) -> _Cdp:
            del page
            return self.cdp

    async def scenario() -> None:
        page = _Page()
        cdp = _Cdp()
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = _Context(cdp)
        daemon.configuration_popup_policy = _browser_guest._InteractivePopupPolicy(
            mode="same_origin",
            allowed_operations=("click",),
            allowed_opener_origins=(),
            allowed_destination_origins=(),
        )
        daemon.active_delta = _browser_guest._InteractivePageDelta()
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_popup",
            lifecycle="provisional",
            opener_page_id="bp_opener",
            opener_origin="https://docs.browser.test",
        )

        await daemon._configure_page(state, _interactive_limits())
        await cdp.handlers["Fetch.requestPaused"](
            {
                "requestId": "request-1",
                "frameId": "frame-main",
                "resourceType": "Document",
                "responseStatusCode": 302,
                "responseHeaders": [
                    {
                        "name": "Location",
                        "value": "https://blocked.browser.test/private",
                    }
                ],
                "request": {"url": "https://docs.browser.test/redirect-denied"},
            }
        )

        assert state.denied_code == "destination_denied"
        assert cdp.failed_requests == [{"requestId": "request-1", "errorReason": "BlockedByClient"}]
        assert daemon.active_delta.refused == [
            {
                "page_id": "bp_popup",
                "opener_page_id": "bp_opener",
                "reason": "destination_denied",
            }
        ]
        assert await daemon._await_page_close(
            state,
            timeout_seconds=1,
            close_provisional_target=True,
        )
        assert cdp.closed_targets == ["target-popup"]

    asyncio.run(scenario())


def test_interactive_guest_stages_early_popup_navigation_until_guards_exist() -> None:
    class _Page:
        def __init__(self, url: str, *, opener: _Page | None = None) -> None:
            self.url = url
            self.main_frame = types.SimpleNamespace(page=self)
            self._opener = opener
            self.goto_calls: list[str] = []

        async def opener(self) -> _Page | None:
            return self._opener

        async def goto(self, url: str, **kwargs: Any) -> None:
            assert kwargs["wait_until"] == "domcontentloaded"
            assert 0 < kwargs["timeout"] <= 1000
            self.goto_calls.append(url)
            self.url = url

        async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
            assert state == "domcontentloaded"
            assert 0 < timeout <= 1000

        async def close(self) -> None:
            return None

    class _BrowserRequest:
        def __init__(self, page: _Page, url: str) -> None:
            self.url = url
            self.method = "GET"
            self.frame = page.main_frame
            self.redirected_from = None

        def is_navigation_request(self) -> bool:
            return True

    class _Route:
        def __init__(self) -> None:
            self.aborted = False

        async def abort(self, reason: str) -> None:
            assert reason == "blockedbyclient"
            self.aborted = True

        async def continue_(self) -> None:
            raise AssertionError("an unguarded popup document must not execute")

    class _Daemon(_browser_guest._InteractiveDaemon):
        async def _configure_page(self, state: Any, limits: Any) -> None:
            del limits
            state.configured = True

    async def scenario() -> None:
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "multi_page": True,
                "popup_policy": _browser_guest._InteractivePopupPolicy(
                    mode="same_origin",
                    allowed_operations=("click",),
                    allowed_opener_origins=(),
                    allowed_destination_origins=(),
                ),
            }
        )
        opener_page = _Page("https://example.test/root")
        popup_page = _Page("about:blank", opener=opener_page)
        daemon = _Daemon("bs_test")
        daemon.context = types.SimpleNamespace()
        opener = _browser_guest._InteractivePage(
            page=opener_page,
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            revision="br_root",
            public_url=opener_page.url,
        )
        daemon.pages[opener.page_id] = opener
        daemon.active_page_id = opener.page_id
        daemon.total_page_creations = 1
        daemon.configuration_limits = limits
        daemon.configuration_multi_page = True
        daemon.configuration_popup_policy = request.popup_policy
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()
        _arm_popup_effect_for_test(daemon, opener)
        target = "https://example.test/popup"
        route = _Route()

        await daemon._route_interactive_request(
            route,
            _BrowserRequest(popup_page, target),
        )

        state = daemon._state_for_page(popup_page)
        assert route.aborted is True
        assert state is not None
        assert state.opener_page_id == opener.page_id
        assert state.lifecycle == "provisional"
        assert state.staged_initial_url == target
        assert popup_page.goto_calls == []

        await daemon._settle_operation_popups(request, daemon.active_delta)

        assert popup_page.goto_calls == [target]
        assert state.staged_initial_url is None
        assert state.lifecycle == "background"
        assert daemon.active_delta.admitted_page_ids == {state.page_id}

    asyncio.run(scenario())


def test_interactive_guest_popup_policy_uses_creation_time_opener_origin() -> None:
    class _Page:
        def __init__(self, url: str) -> None:
            self.url = url

        async def evaluate(self, expression: str, values: list[object]) -> dict[str, object]:
            del expression, values
            return {"blocked": 0, "urls": []}

        async def close(self) -> None:
            return None

    async def scenario() -> None:
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "multi_page": True,
                "popup_policy": _browser_guest._InteractivePopupPolicy(
                    mode="same_origin",
                    allowed_operations=("click",),
                    allowed_opener_origins=(),
                    allowed_destination_origins=(),
                ),
            }
        )
        opener_page = _Page("https://origin.example/start")
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = types.SimpleNamespace()
        opener = _browser_guest._InteractivePage(
            page=opener_page,
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            revision="br_root",
            public_url=opener_page.url,
        )
        daemon.pages[opener.page_id] = opener
        daemon.active_page_id = opener.page_id
        daemon.total_page_creations = 1
        daemon.configuration_limits = limits
        daemon.configuration_multi_page = True
        daemon.configuration_popup_policy = request.popup_policy
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()

        await daemon._begin_popup_effect(opener, request)
        opener.public_url = "https://later.example/changed"
        opener_page.url = opener.public_url

        candidate = daemon._register_popup_candidate(
            opener,
            _Page("https://later.example/popup"),
        )
        assert candidate is not None
        assert candidate.opener_origin == "https://origin.example/"

        assert (
            daemon._popup_destination_allowed(
                candidate,
                "https://origin.example/allowed",
            )
            is True
        )
        assert (
            daemon._popup_destination_allowed(
                candidate,
                "https://later.example/not-the-creation-origin",
            )
            is False
        )

    asyncio.run(scenario())


def test_interactive_guest_popup_policy_uses_current_opener_origin_at_creation() -> None:
    class _Page:
        def __init__(self, url: str) -> None:
            self.url = url

        async def close(self) -> None:
            return None

    async def scenario() -> None:
        limits = _interactive_limits(
            max_pages=2,
            max_provisional_pages=1,
            max_page_creations_per_operation=1,
            max_total_page_creations=2,
        )
        request = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "multi_page": True,
                "popup_policy": _browser_guest._InteractivePopupPolicy(
                    mode="destination_policy",
                    allowed_operations=("click",),
                    allowed_opener_origins=("https://current.example/",),
                    allowed_destination_origins=("https://popup.example/",),
                ),
            }
        )
        page = _Page("https://current.example/after-navigation")
        opener = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_root",
            lifecycle="active",
            public_url="https://stale.example/before-navigation",
        )
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.pages[opener.page_id] = opener
        daemon.active_page_id = opener.page_id
        daemon.total_page_creations = 1
        daemon.configuration_multi_page = True
        daemon.configuration_limits = limits
        daemon.configuration_popup_policy = request.popup_policy
        daemon.active_request = request
        daemon.active_delta = _browser_guest._InteractivePageDelta()
        _arm_popup_effect_for_test(daemon, opener)

        candidate = daemon._register_popup_candidate(
            opener,
            _Page("https://popup.example/child"),
        )

        assert candidate is not None
        assert candidate.opener_origin == "https://current.example/"

    asyncio.run(scenario())
