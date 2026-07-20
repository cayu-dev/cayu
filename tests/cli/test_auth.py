from __future__ import annotations

import http.client
import threading
from pathlib import Path

import cayu.cli.auth as auth_cli
from cayu.cli import main
from cayu.providers.openai_subscription import OpenAISubscriptionCredentials


def _credentials() -> OpenAISubscriptionCredentials:
    return OpenAISubscriptionCredentials(
        access_token="secret-access-token",
        refresh_token="secret-refresh-token",
        expires_at=2_000_000_000,
        account_id="acct-cli",
    )


def test_auth_openai_status_and_logout_never_print_tokens(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("CAYU_HOME", str(tmp_path / "cayu-home"))

    assert main(["auth", "openai", "status"]) == 1
    assert "not signed in" in capsys.readouterr().out.lower()

    auth_cli.OpenAISubscriptionAuthStore().save(_credentials())
    assert main(["auth", "openai", "status"]) == 0
    status = capsys.readouterr().out
    assert "acct-cli" in status
    assert "secret-access-token" not in status
    assert "secret-refresh-token" not in status

    assert main(["auth", "openai", "logout"]) == 0
    assert "signed out" in capsys.readouterr().out.lower()
    assert main(["auth", "openai", "status"]) == 1


def test_auth_openai_login_saves_browser_credentials(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("CAYU_HOME", str(tmp_path / "cayu-home"))
    monkeypatch.setattr(
        auth_cli,
        "_browser_login",
        lambda *, transport, open_browser: _credentials(),
    )

    assert main(["auth", "openai", "login", "--no-browser"]) == 0

    output = capsys.readouterr().out
    assert "experimental" in output.lower()
    assert "originator: cayu" in output
    assert auth_cli.OpenAISubscriptionAuthStore().load() == _credentials()


def test_oauth_callback_rejects_error_from_wrong_state() -> None:
    query = {
        "state": ["attacker-state"],
        "error": ["access_denied"],
        "error_description": ["attacker-controlled message"],
    }

    try:
        auth_cli._authorization_code_from_callback(query, expected_state="expected-state")
    except auth_cli.OpenAISubscriptionAuthError as exc:
        assert str(exc) == "OAuth callback state did not match."
        assert "attacker-controlled" not in str(exc)
    else:
        raise AssertionError("callback with the wrong OAuth state must fail")


def test_oauth_callback_keeps_waiting_after_wrong_state() -> None:
    server = auth_cli._CallbackServer(("127.0.0.1", 0), auth_cli._CallbackHandler)
    server.expected_state = "expected-state"
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=2)
    try:
        connection.request(
            "GET",
            "/auth/callback?state=attacker-state&error=access_denied",
        )
        response = connection.getresponse()
        response.read()

        assert response.status == 400
        assert server.authorization_code is None
        assert server.callback_error is None

        connection.request(
            "GET",
            "/auth/callback?state=expected-state",
        )
        response = connection.getresponse()
        response.read()

        assert response.status == 400
        assert server.authorization_code is None
        assert server.callback_error is None

        connection.request(
            "GET",
            "/auth/callback?state=expected-state&code=legitimate-code",
        )
        response = connection.getresponse()
        response.read()

        assert response.status == 200
        assert server.authorization_code == "legitimate-code"
        assert server.callback_error is None
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_device_login_rejects_non_finite_polling_interval() -> None:
    class DeviceTransport:
        issuer = "https://auth.openai.com"

        def __init__(self, interval) -> None:
            self.interval = interval

        def request_device_authorization(self):
            return {
                "device_auth_id": "device-auth-id",
                "user_code": "user-code",
                "interval": self.interval,
            }

        def poll_device_authorization(self, **kwargs):
            raise AssertionError(f"invalid interval must fail before polling: {kwargs}")

    for interval in (float("inf"), float("-inf"), float("nan"), 10**400):
        try:
            auth_cli._device_login(transport=DeviceTransport(interval))
        except auth_cli.OpenAISubscriptionAuthError as exc:
            assert str(exc) == "OpenAI device authorization contained invalid interval."
        else:
            raise AssertionError(f"device polling interval {interval!r} must fail")


def test_device_login_caps_polling_sleep_at_deadline(monkeypatch) -> None:
    class DeviceTransport:
        issuer = "https://auth.openai.com"

        def __init__(self) -> None:
            self.polls = 0

        def request_device_authorization(self):
            return {
                "device_auth_id": "device-auth-id",
                "user_code": "user-code",
                "interval": 2_000,
            }

        def poll_device_authorization(self, **kwargs):
            del kwargs
            self.polls += 1
            return None

    clock = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(auth_cli.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(auth_cli.time, "sleep", sleep)
    transport = DeviceTransport()

    try:
        auth_cli._device_login(transport=transport)
    except auth_cli.OpenAISubscriptionAuthError as exc:
        assert str(exc) == "OpenAI device login timed out."
    else:
        raise AssertionError("device login must stop at its deadline")

    assert transport.polls == 1
    assert sleeps == [900.0]
