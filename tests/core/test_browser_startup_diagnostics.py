"""Startup diagnostics cross the daemon boundary without private launch material."""

import asyncio

import pytest
from tests.core.test_browser_session import _interactive_raw_request

from cayu.tools import _browser_guest as guest
from cayu.tools.browser_session import _ERROR_MESSAGES


@pytest.mark.parametrize(
    ("detail", "code"),
    [
        ("Failed to move to new namespace: Operation not permitted", "browser_sandbox_unavailable"),
        ("No usable sandbox!", "browser_sandbox_unavailable"),
        ("Executable doesn't exist at /private/browser", "browser_dependencies_unavailable"),
        ("error while loading shared libraries: lib.so", "browser_dependencies_unavailable"),
        ("unknown failure", "browser_startup_failed"),
    ],
)
def test_launch_classification_never_retains_raw_detail(detail, code):
    secret = "secret-proxy-password"
    assert guest._browser_startup_failure(RuntimeError(detail + secret)) == code
    payload = guest._interactive_error_payload(guest._GuestFailure(code))
    assert code in str(payload)
    assert secret not in str(payload)
    assert code in _ERROR_MESSAGES


@pytest.mark.parametrize("cleanup_ok", [True, False])
def test_failed_daemon_startup_retains_safe_reason_and_cleanup(monkeypatch, tmp_path, cleanup_ok):
    monkeypatch.setattr(guest, "_INTERACTIVE_ROOT", tmp_path)
    closes = []

    class Daemon(guest._InteractiveDaemon):
        async def start(self):
            raise guest._GuestFailure("browser_sandbox_unavailable")

        async def close(self):
            closes.append(True)
            return cleanup_ok

    monkeypatch.setattr(guest, "_InteractiveDaemon", Daemon)
    session_id = "bs_startup"
    assert asyncio.run(guest._interactive_daemon_main(session_id)) == 1
    assert closes
    assert guest._retired_startup_failure(session_id) == "browser_sandbox_unavailable"
    assert guest._interactive_retirement_is_recorded(session_id) is cleanup_ok
    assert (
        next(iter(tmp_path.glob("*.startup-failure"))).read_bytes()
        == b"browser_sandbox_unavailable"
    )


def test_parent_reports_startup_reason_only_with_retirement(monkeypatch, tmp_path):
    monkeypatch.setattr(guest, "_INTERACTIVE_ROOT", tmp_path)
    monkeypatch.setattr(guest.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(guest.importlib.metadata, "version", lambda _: guest.PLAYWRIGHT_VERSION)
    monkeypatch.setattr(guest, "_proxy_and_ca", lambda: ("http://proxy", "/ca.pem"))
    raw = _interactive_raw_request("navigate")
    session_id = raw["session_id"]

    async def no_response(*args):
        return None

    monkeypatch.setattr(guest, "_interactive_send", no_response)
    guest._record_startup_failure(session_id, "browser_sandbox_unavailable")
    assert guest._record_interactive_retirement(session_id)
    with pytest.raises(guest._GuestFailure) as failure:
        asyncio.run(guest._run_interactive_request(raw))
    assert failure.value.code == "browser_sandbox_unavailable"
    assert failure.value.allocation_disposition == "retired"


@pytest.mark.parametrize("payload", [b"private-secret", b"x" * 65, b"\xff"])
def test_startup_marker_rejects_unknown_or_unbounded_content(monkeypatch, tmp_path, payload):
    monkeypatch.setattr(guest, "_INTERACTIVE_ROOT", tmp_path)
    path = guest._interactive_retired_path("bs_test").with_suffix(".startup-failure")
    path.write_bytes(payload)
    assert guest._retired_startup_failure("bs_test") == "browser_unavailable"
