"""Interactive browser guest profile import, secrecy and destination contracts."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import select
import shutil
import signal
import time
from pathlib import Path
from typing import Any

import pytest
from tests.core.browser_guest_support import (
    _configure_interactive_daemon_for_test,
    _interactive_limits,
    _interactive_raw_request,
    _interactive_request,
)

from cayu.tools import _browser_guest


def test_interactive_guest_omits_new_profile_values_before_response_publication() -> None:
    class _ProfileContext:
        async def storage_state(self, *, indexed_db: bool) -> dict[str, Any]:
            assert indexed_db is False
            return {
                "cookies": [
                    {
                        "name": "new-private-cookie-name",
                        "value": "new-private-cookie",
                        "domain": "example.test",
                        "path": "/",
                        "expires": -1.0,
                        "httpOnly": True,
                        "secure": True,
                        "sameSite": "Lax",
                    }
                ],
                "origins": [
                    {
                        "origin": "https://example.test",
                        "localStorage": [
                            {
                                "name": "new-private-storage-name",
                                "value": "new-private-storage",
                            }
                        ],
                    }
                ],
            }

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = _ProfileContext()
        daemon.profile_output_values = ("restored-private-value",)
        daemon.profile_allowed_origins = ("https://example.test",)
        daemon.profile_plaintext_limit = 4_096
        daemon.profile_timeout_seconds = 1.0
        observation = {
            "url": "https://example.test/account",
            "title": "new-private-cookie-name",
            "snapshot": (
                "restored-private-value and new-private-storage and new-private-storage-name"
            ),
            "refs": [
                {
                    "ref": "ref_private",
                    "role": "button",
                    "name": "new-private-cookie",
                }
            ],
        }

        protected, proof = await daemon._protect_profile_observation(observation)

        assert proof is True
        assert protected["title"] is None
        assert protected["refs"] == []
        rendered = json.dumps(protected, sort_keys=True)
        assert "restored-private-value" not in rendered
        assert "new-private-cookie" not in rendered
        assert "new-private-storage" not in rendered
        assert "new-private-cookie-name" not in rendered
        assert "new-private-storage-name" not in rendered

        with pytest.raises(_browser_guest._GuestFailure) as exc_info:
            await daemon._protect_profile_observation(
                {
                    **observation,
                    "url": "https://example.test/?token=new-private-cookie",
                }
            )
        assert exc_info.value.code == "policy_denied"

    asyncio.run(scenario())


def test_interactive_guest_profile_ceilings_match_the_host_ceilings() -> None:
    import cayu.browser_profiles as profiles

    assert (
        _browser_guest._INTERACTIVE_MAX_PROFILE_ORIGINS,
        _browser_guest._INTERACTIVE_MAX_PROFILE_COOKIES,
        _browser_guest._INTERACTIVE_MAX_PROFILE_STORAGE_ENTRIES,
        _browser_guest._INTERACTIVE_MAX_PROFILE_NAME_BYTES,
        _browser_guest._INTERACTIVE_MAX_PROFILE_VALUE_BYTES,
        _browser_guest._INTERACTIVE_MAX_PROFILE_PLAINTEXT_BYTES,
    ) == (
        profiles.BROWSER_PROFILE_MAX_ORIGINS,
        profiles.BROWSER_PROFILE_MAX_COOKIES,
        profiles.BROWSER_PROFILE_MAX_STORAGE_ENTRIES,
        profiles.BROWSER_PROFILE_MAX_NAME_BYTES,
        profiles.BROWSER_PROFILE_MAX_VALUE_BYTES,
        profiles.BROWSER_PROFILE_MAX_PLAINTEXT_BYTES,
    )


def test_interactive_guest_restores_a_profile_with_many_cookies() -> None:
    # 300 cookies fit the default profile limits (1,024) but not the old guest cap (256).
    cookies = [
        {
            "name": f"c{index:04d}",
            "value": "v" * 40,
            "domain": "app.example.test",
            "path": "/",
            "expires": -1.0,
            "httpOnly": True,
            "secure": True,
            "sameSite": "Lax",
        }
        for index in range(300)
    ]
    state = {
        "cookies": cookies,
        "origins": [
            {
                "origin": "https://app.example.test",
                "localStorage": [{"name": f"k{index:04d}", "value": "x"} for index in range(600)],
            }
        ],
    }

    restored = _browser_guest._validate_interactive_profile_state(
        state,
        maximum_bytes=_browser_guest._INTERACTIVE_MAX_PROFILE_PLAINTEXT_BYTES,
        allowed_origins=("https://app.example.test",),
        require_canonical_order=True,
    )
    assert len(restored["cookies"]) == 300
    live_values = _browser_guest._bounded_live_browser_private_values(
        state, maximum_bytes=_browser_guest._INTERACTIVE_MAX_PROFILE_PLAINTEXT_BYTES
    )
    assert "c0299" in live_values


def test_interactive_guest_bounds_cumulative_profile_anti_reflection_state() -> None:
    class _ProfileContext:
        async def storage_state(self, *, indexed_db: bool) -> dict[str, Any]:
            assert indexed_db is False
            return {
                "cookies": [
                    {
                        "name": "new-cookie-name",
                        "value": "new-cookie-value",
                        "domain": "example.test",
                        "path": "/",
                        "expires": -1.0,
                        "httpOnly": True,
                        "secure": True,
                        "sameSite": "Lax",
                    }
                ],
                "origins": [],
            }

    async def scenario() -> None:
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = _ProfileContext()
        prior_values = tuple(
            f"prior-private-value-{index}"
            for index in range(_browser_guest._INTERACTIVE_MAX_PROFILE_PRIVATE_VALUES)
        )
        daemon.profile_output_values = prior_values
        daemon.profile_allowed_origins = ("https://example.test",)
        daemon.profile_plaintext_limit = _browser_guest._INTERACTIVE_MAX_PROFILE_PLAINTEXT_BYTES
        daemon.profile_timeout_seconds = 1.0

        with pytest.raises(_browser_guest._GuestFailure) as count_failure:
            await daemon._protect_profile_observation(
                {
                    "url": "https://example.test/account",
                    "title": "safe",
                    "snapshot": "safe",
                    "refs": [],
                }
            )
        assert count_failure.value.code == "resource_exhausted"
        assert daemon.profile_output_values == prior_values

        with pytest.raises(_browser_guest._GuestFailure) as byte_failure:
            _browser_guest._bounded_interactive_profile_private_values(
                ("a" * 200,),
                ("b" * 200,),
                maximum_bytes=256,
            )
        assert byte_failure.value.code == "resource_exhausted"

    asyncio.run(scenario())


def test_interactive_guest_profile_ledger_reserves_restore_plus_every_checkpoint() -> None:
    class _Context:
        async def add_init_script(self, script: str) -> None:
            assert script

        async def route(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            assert callable(callback)

        async def route_web_socket(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            assert callable(callback)

        async def storage_state(self, *, indexed_db: bool) -> dict[str, Any]:
            assert indexed_db is False
            return {"cookies": [], "origins": []}

        async def close(self) -> None:
            return None

    class _Browser:
        async def new_context(self, **kwargs: Any) -> _Context:
            assert kwargs["storage_state"] == {"cookies": [], "origins": []}
            return _Context()

        def on(self, event: str, callback: Any) -> None:
            assert event == "disconnected"
            assert callable(callback)

        def is_connected(self) -> bool:
            return True

    async def scenario() -> None:
        limits = _interactive_limits(max_operations=1)
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.browser = _Browser()
        restore = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("profile_restore", limits=limits).__dict__,
                "page_id": None,
                "operation_id": "profile-restore-1",
                "profile_restore_state": {"cookies": [], "origins": []},
                "profile_allowed_origins": ("https://example.test",),
                "profile_plaintext_limit": 1_024,
                "profile_timeout_seconds": 1.0,
            }
        )
        checkpoint = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("profile_checkpoint", limits=limits).__dict__,
                "page_id": None,
                "operation_id": "profile-checkpoint-1",
                "profile_allowed_origins": ("https://example.test",),
                "profile_capture_limit": 1_024,
                "profile_timeout_seconds": 1.0,
            }
        )

        restored = await daemon.execute(restore)
        captured = await daemon.execute(checkpoint)

        assert restored["kind"] == "profile_restore"
        assert captured["kind"] == "profile_checkpoint"
        assert tuple(daemon.profile_operations) == (
            "profile-restore-1",
            "profile-checkpoint-1",
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "profile_state",
    [
        {
            "cookies": [
                {
                    "name": "session",
                    "value": "value",
                    "domain": "example.test",
                    "path": "/",
                    "expires": -1.0,
                    "httpOnly": True,
                    "secure": False,
                    "sameSite": "Lax",
                }
            ],
            "origins": [],
        },
        {
            "cookies": [
                {
                    "name": "session",
                    "value": "value",
                    "domain": "outside.test",
                    "path": "/",
                    "expires": -1.0,
                    "httpOnly": True,
                    "secure": True,
                    "sameSite": "Lax",
                }
            ],
            "origins": [],
        },
        {
            "cookies": [],
            "origins": [
                {
                    "origin": "https://outside.test",
                    "localStorage": [{"name": "session", "value": "value"}],
                }
            ],
        },
    ],
    ids=["insecure-cookie", "cookie-origin-widening", "storage-origin-widening"],
)
def test_interactive_guest_revalidates_profile_authority_before_import(
    profile_state: dict[str, Any],
) -> None:
    raw = _interactive_raw_request("profile_restore")
    raw.pop("page_id")
    raw["browser_profile"] = {
        "allowed_origins": ["https://example.test"],
        "schema_version": 1,
        "capture": False,
        "max_plaintext_bytes": 4096,
        "restore_state": profile_state,
        "timeout_seconds": 1.0,
    }

    with pytest.raises(_browser_guest._GuestFailure, match="incompatible_browser"):
        _browser_guest._interactive_request_from_json(raw)


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "fork"),
    reason="hard process-loss cleanup requires POSIX fork and signals",
)
def test_interactive_profile_guardian_deletes_home_after_worker_process_loss() -> None:
    read_fd, write_fd = os.pipe()
    child_pid = os.fork()
    profile_home: Path | None = None
    child_reaped = False
    if child_pid == 0:  # pragma: no cover - assertions execute in the parent
        os.close(read_fd)

        async def child() -> None:
            daemon = _browser_guest._InteractiveDaemon("bs_process_loss")
            await daemon._start_profile_home()
            if daemon.home is None:
                raise AssertionError("interactive profile home was not created")
            (daemon.home / "credential-canary.txt").write_text(
                "private-browser-state",
                encoding="utf-8",
            )
            os.write(write_fd, str(daemon.home).encode("utf-8") + b"\n")
            await asyncio.Event().wait()

        try:
            asyncio.run(child())
        except BaseException:
            with contextlib.suppress(OSError):
                os.write(write_fd, b"ERROR\n")
        finally:
            os.close(write_fd)
            os._exit(0)

    os.close(write_fd)
    try:
        readable, _, _ = select.select((read_fd,), (), (), 5.0)
        assert readable
        payload = os.read(read_fd, 4_096).decode("utf-8").strip()
        assert payload and payload != "ERROR"
        profile_home = Path(payload)
        assert profile_home.exists()
        assert (profile_home / "credential-canary.txt").exists()

        os.kill(child_pid, signal.SIGKILL)
        waited_pid, status = os.waitpid(child_pid, 0)
        child_reaped = True
        assert waited_pid == child_pid
        assert os.WIFSIGNALED(status)
        assert os.WTERMSIG(status) == signal.SIGKILL

        deadline = time.monotonic() + 5.0
        while profile_home.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert profile_home.exists() is False
    finally:
        os.close(read_fd)
        if not child_reaped:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
            with contextlib.suppress(ChildProcessError):
                os.waitpid(child_pid, 0)
        if profile_home is not None and profile_home.exists():
            shutil.rmtree(profile_home)


def test_interactive_profile_rejects_https_ip_before_next_action() -> None:
    class _Cdp:
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
            assert event in {
                "Fetch.requestPaused",
                "Network.dataReceived",
                "Network.requestWillBeSent",
            }
            assert callable(callback)

    class _Page:
        def __init__(self) -> None:
            self.main_frame = object()
            self.handlers: dict[str, Any] = {}
            self.locator_called = False

        def set_default_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def set_default_navigation_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def on(self, event: str, callback: Any) -> None:
            self.handlers[event] = callback

        def locator(self, selector: str) -> Any:
            del selector
            self.locator_called = True
            raise AssertionError("a denied page must reject before resolving an action target")

    class _Context:
        def __init__(self) -> None:
            self.route_callback: Any = None
            self.cdp = _Cdp()

        async def route(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            self.route_callback = callback

        async def route_web_socket(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            assert callable(callback)

        async def new_cdp_session(self, page: _Page) -> _Cdp:
            del page
            return self.cdp

    class _Frame:
        def __init__(self, page: _Page) -> None:
            self.page = page

    class _BrowserRequest:
        def __init__(self, page: _Page) -> None:
            self.url = "https://127.0.0.1/private"
            self.frame = _Frame(page)
            self.redirected_from = None

        def is_navigation_request(self) -> bool:
            return False

    class _Route:
        def __init__(self) -> None:
            self.aborted = False

        async def abort(self, reason: str) -> None:
            assert reason == "blockedbyclient"
            self.aborted = True

        async def continue_(self) -> None:
            raise AssertionError("the disallowed background request must not continue")

    async def scenario() -> None:
        limits = _interactive_limits()
        page = _Page()
        context = _Context()
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = context
        daemon.profile_allowed_origins = ("https://allowed.example",)
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
            revision="br_revision_1",
            refs={"ref_save": "internal_save"},
        )
        daemon.pages[state.page_id] = state
        action = _browser_guest._InteractiveRequest(
            **{
                **_interactive_request("click", limits=limits).__dict__,
                "expected_revision": "br_revision_1",
                "expected_control_epoch": 1,
                "ref": "ref_save",
            }
        )
        await _configure_interactive_daemon_for_test(daemon, action)
        await daemon._configure_page(state, limits)

        unknown_page = _Page()
        unknown_route = _Route()
        await context.route_callback(unknown_route, _BrowserRequest(unknown_page))
        assert unknown_route.aborted is True
        assert daemon._state_for_page(unknown_page) is None

        route = _Route()
        await context.route_callback(route, _BrowserRequest(page))
        assert route.aborted is True
        assert state.denied_code == "destination_denied"

        result = await daemon.execute(action)
        assert result["kind"] == "error"
        assert result["error"] == "destination_denied"
        assert page.locator_called is False
        assert state.revision == "br_revision_1"

    asyncio.run(scenario())


def test_interactive_profile_policy_fences_websocket_destinations() -> None:
    class _Cdp:
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
            assert event in {"Fetch.requestPaused", "Network.dataReceived"}
            assert callable(callback)

    class _Page:
        def set_default_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def set_default_navigation_timeout(self, timeout: int) -> None:
            assert timeout == 1000

        def on(self, event: str, callback: Any) -> None:
            assert event in {"response", "popup"}
            assert callable(callback)

    class _Context:
        def __init__(self) -> None:
            self.cdp = _Cdp()
            self.web_socket_callback: Any = None

        async def route(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            assert callable(callback)

        async def add_init_script(self, script: str) -> None:
            assert "__cayuSetPopupAdmission" in script

        async def route_web_socket(self, pattern: str, callback: Any) -> None:
            assert pattern == "**/*"
            self.web_socket_callback = callback

        async def new_cdp_session(self, page: _Page) -> _Cdp:
            del page
            return self.cdp

    class _WebSocketRoute:
        def __init__(self, url: str) -> None:
            self.url = url
            self.closed: tuple[int | None, str | None] | None = None
            self.connected = False

        def close(self, *, code: int | None = None, reason: str | None = None) -> None:
            self.closed = (code, reason)

        def connect_to_server(self) -> None:
            self.connected = True

    async def scenario() -> None:
        context = _Context()
        page = _Page()
        daemon = _browser_guest._InteractiveDaemon("bs_test")
        daemon.context = context
        daemon.profile_allowed_origins = ("https://allowed.example",)
        request = _interactive_request("navigate")
        state = _browser_guest._InteractivePage(
            page=page,
            session_id="bs_test",
            page_id="bp_test",
            lifecycle="active",
        )
        daemon.pages[state.page_id] = state
        daemon.active_page_id = state.page_id
        daemon.total_page_creations = 1
        await daemon._ensure_configuration(request)
        assert callable(context.web_socket_callback)

        allowed = _WebSocketRoute("wss://allowed.example/socket")
        context.web_socket_callback(allowed)
        assert allowed.connected is True
        assert allowed.closed is None

        denied = _WebSocketRoute("wss://broader-egress.example/socket")
        context.web_socket_callback(denied)
        assert denied.connected is False
        assert denied.closed == (1008, "Policy denied.")
        assert state.denied_code == "destination_denied"

    asyncio.run(scenario())
