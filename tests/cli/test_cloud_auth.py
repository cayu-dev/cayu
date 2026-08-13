from __future__ import annotations

import base64
import json
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import pytest

from cayu.cli import _cloud_auth as cloud_auth
from cayu.cli import cloud as cloud_cli
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiClient
from cayu.cli._cloud_auth import (
    CloudAuthCredentials,
    CloudAuthError,
    CloudAuthStore,
    DeviceAuthorization,
    WorkOSDeviceAuthClient,
)


def _jwt(*, expires_at: float, marker: str) -> str:
    def encode(value: dict[str, object]) -> str:
        payload = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(payload).rstrip(b"=").decode()

    return ".".join(
        (
            encode({"alg": "RS256", "typ": "JWT"}),
            encode(
                {
                    "exp": int(expires_at),
                    "org_id": "org_test",
                    "sub": "user_test",
                    "marker": marker,
                }
            ),
            "signature",
        )
    )


def _start_authentication_server(
    responses: list[tuple[int, dict[str, object]]],
) -> tuple[ThreadingHTTPServer, threading.Thread, list[dict[str, list[str]]]]:
    forms: list[dict[str, list[str]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            forms.append(parse_qs(self.rfile.read(length).decode()))
            status_code, payload = responses.pop(0)
            encoded = json.dumps(payload).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, forms


def _device_authorization(*, expires_in: float = 300) -> DeviceAuthorization:
    return DeviceAuthorization(
        device_code="private-device-code",
        user_code="CAYU-TEST",
        verification_uri="https://auth.example.test/device",
        verification_uri_complete="https://auth.example.test/device?user_code=CAYU-TEST",
        expires_in=expires_in,
        interval=2,
    )


def test_cloud_help_exposes_workos_session_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "--help"])

    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "login" in help_text
    assert "logout" in help_text
    assert "whoami" in help_text


def test_cloud_login_uses_production_despite_environment_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(tmp_path / "missing-auth.json"))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing-config.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_URL", "https://staging-cloud.example.test")
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    selected_api_urls: list[str] = []

    def portal_config(client: WorkOSDeviceAuthClient) -> tuple[str, str]:
        selected_api_urls.append(client.api_url)
        raise CloudAuthError("login_unavailable", "Cayu Cloud login is unavailable.")

    monkeypatch.setattr(WorkOSDeviceAuthClient, "portal_config", portal_config)

    assert main(["cloud", "login", "--no-browser"]) == 2

    streams = capsys.readouterr()
    assert streams.err == ""
    assert selected_api_urls == ["https://cloud.cayu.dev"]
    assert json.loads(streams.out) == {
        "error": {
            "category": "login_unavailable",
            "message": "Cayu Cloud login is unavailable.",
        },
        "ok": False,
    }


def test_cloud_login_without_endpoint_configuration_ignores_saved_nonproduction_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    auth_path = tmp_path / "cloud-auth.json"
    CloudAuthStore(auth_path).save(
        CloudAuthCredentials(
            api_url="https://staging-cloud.example.test",
            workos_api_hostname="auth.example.test",
            workos_client_id="client_test",
            access_token=_jwt(expires_at=time.time() + 3600, marker="staging"),
            refresh_token="private-refresh-token",
            expires_at=time.time() + 3600,
            organization_id="org_test",
            user_id="user_test",
        )
    )
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(auth_path))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing-config.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_URL", "https://staging-cloud.example.test")
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    selected_api_urls: list[str] = []

    def portal_config(client: WorkOSDeviceAuthClient) -> tuple[str, str]:
        selected_api_urls.append(client.api_url)
        raise CloudAuthError("login_unavailable", "Cayu Cloud login is unavailable.")

    monkeypatch.setattr(WorkOSDeviceAuthClient, "portal_config", portal_config)

    assert main(["cloud", "login", "--no-browser"]) == 2

    assert selected_api_urls == ["https://cloud.cayu.dev"]
    assert json.loads(capsys.readouterr().out)["error"]["category"] == "login_unavailable"


def test_cloud_deploy_does_not_follow_saved_nonproduction_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    auth_path = tmp_path / "cloud-auth.json"
    expiring_credentials = CloudAuthCredentials(
        api_url="https://staging-cloud.example.test",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=time.time() + 30, marker="staging"),
        refresh_token="private-refresh-token",
        expires_at=time.time() + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    CloudAuthStore(auth_path).save(expiring_credentials)
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(auth_path))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing-config.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_URL", "https://staging-cloud.example.test")
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    deployments: list[str] = []
    refreshes: list[str] = []

    def deploy(
        _arguments: object,
        *,
        client: CloudApiClient,
        recorder: object,
        project: object,
    ) -> dict[str, object]:
        del recorder, project
        deployments.append(client.api_url)
        return {"operation": "deploy", "result": {}}

    monkeypatch.setattr(cloud_cli, "resolve_project", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cloud_cli, "_deploy", deploy)
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "refresh",
        lambda _client, credentials: refreshes.append(credentials.api_url) or credentials,
    )

    assert (
        main(
            [
                "cloud",
                "--evidence-dir",
                str(tmp_path / "evidence"),
                "deploy",
                "--no-wait",
            ]
        )
        == 2
    )

    assert deployments == []
    assert refreshes == []
    assert json.loads(capsys.readouterr().out) == {
        "error": {
            "category": "login_api_mismatch",
            "message": (
                "The saved login belongs to another Cayu Cloud; run "
                "`cayu cloud login` to sign in to production."
            ),
        },
        "ok": False,
    }


def test_cloud_deploy_uses_production_despite_environment_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    auth_path = tmp_path / "cloud-auth.json"
    CloudAuthStore(auth_path).save(
        CloudAuthCredentials(
            api_url="https://cloud.cayu.dev",
            workos_api_hostname="auth.example.test",
            workos_client_id="client_test",
            access_token=_jwt(expires_at=time.time() + 3600, marker="production"),
            refresh_token="private-refresh-token",
            expires_at=time.time() + 3600,
            organization_id="org_test",
            user_id="user_test",
        )
    )
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(auth_path))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "missing-config.json"))
    monkeypatch.setenv("CAYU_CLOUD_API_URL", "https://staging-cloud.example.test")
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    deployments: list[str] = []

    def deploy(
        _arguments: object,
        *,
        client: CloudApiClient,
        recorder: object,
        project: object,
    ) -> dict[str, object]:
        del recorder, project
        deployments.append(client.api_url)
        return {"operation": "deploy", "result": {}}

    monkeypatch.setattr(cloud_cli, "resolve_project", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(cloud_cli, "_deploy", deploy)

    assert (
        main(
            [
                "cloud",
                "--evidence-dir",
                str(tmp_path / "evidence"),
                "deploy",
                "--no-wait",
            ]
        )
        == 0
    )

    assert deployments == ["https://cloud.cayu.dev"]
    assert json.loads(capsys.readouterr().out)["operation"] == "deploy"


def test_workos_login_refresh_and_logout_power_normal_cloud_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    access_one = _jwt(expires_at=time.time() + 30, marker="first")
    access_two = _jwt(expires_at=time.time() + 3600, marker="second")
    refresh_one = "refresh-token-one"
    refresh_two = "refresh-token-two"
    device_code = "private-device-code"
    requests: list[tuple[str, str, str | None]] = []
    authentication_forms: list[dict[str, list[str]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            authorization = self.headers.get("Authorization")
            requests.append(("GET", self.path, authorization))
            if self.path == "/v1/portal-config":
                payload = {
                    "authentication": "workos",
                    "workos_api_hostname": f"127.0.0.1:{self.server.server_port}",
                    "workos_client_id": "client_test",
                }
            elif self.path == "/v1/me":
                payload = {
                    "authentication": "workos",
                    "organization_id": "org_test",
                    "permissions": [],
                    "role": "member",
                    "user_id": "user_test",
                }
            elif self.path == "/v1/applications":
                payload = {"items": []}
            else:
                self.send_error(404)
                return
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            form = parse_qs(self.rfile.read(length).decode())
            requests.append(("POST", self.path, self.headers.get("Authorization")))
            if self.path == "/user_management/authorize/device":
                payload = {
                    "device_code": device_code,
                    "expires_in": 300,
                    "interval": 0.01,
                    "user_code": "CAYU-TEST",
                    "verification_uri": "https://auth.example.test/device",
                    "verification_uri_complete": (
                        "https://auth.example.test/device?user_code=CAYU-TEST"
                    ),
                }
            elif self.path == "/user_management/authenticate":
                authentication_forms.append(form)
                if form["grant_type"] == ["urn:ietf:params:oauth:grant-type:device_code"]:
                    payload = {
                        "access_token": access_one,
                        "organization_id": "org_test",
                        "refresh_token": refresh_one,
                        "user": {"id": "user_test"},
                    }
                else:
                    payload = {
                        "access_token": access_two,
                        "organization_id": "org_test",
                        "refresh_token": refresh_two,
                        "user": {"id": "user_test"},
                    }
            else:
                self.send_error(404)
                return
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api_url = f"http://127.0.0.1:{server.server_port}"
    auth_path = tmp_path / "cloud-auth.json"
    config_path = tmp_path / "cloud-config.json"
    config_path.write_text(
        json.dumps(
            {
                "active_context": str(tmp_path / "obsolete-context.json"),
                "schema_version": 1,
            }
        )
    )
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(auth_path))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(config_path))
    monkeypatch.delenv("CAYU_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_KEY_FILE", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_URL", raising=False)
    monkeypatch.setattr(cloud_cli, "_PRODUCTION_API_URL", api_url)
    try:
        assert main(["cloud", "login", "--no-browser"]) == 0
        login_streams = capsys.readouterr()
        login = json.loads(login_streams.out)
        assert login["operation"] == "login"
        assert login["result"] == {
            "api_url": api_url,
            "authentication": "workos",
            "organization_id": "org_test",
            "status": "signed_in",
            "user_id": "user_test",
        }
        assert "CAYU-TEST" in login_streams.err
        assert "https://auth.example.test/device" in login_streams.err
        assert device_code not in login_streams.err
        assert access_one not in login_streams.out + login_streams.err
        assert refresh_one not in login_streams.out + login_streams.err
        assert not config_path.exists()

        persisted = json.loads(auth_path.read_text())
        assert persisted["access_token"] == access_one
        assert persisted["refresh_token"] == refresh_one
        assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600

        assert main(["cloud", "applications", "list"]) == 0
        applications = json.loads(capsys.readouterr().out)
        assert applications["operation"] == "applications.list"
        assert json.loads(auth_path.read_text())["refresh_token"] == refresh_two

        assert main(["cloud", "whoami"]) == 0
        identity = json.loads(capsys.readouterr().out)
        assert identity["result"]["organization_id"] == "org_test"
        assert identity["result"]["user_id"] == "user_test"

        context_key = tmp_path / "context-api-key"
        context_key.write_text("context-credential\n")
        context_path = tmp_path / "cloud-context.json"
        context_path.write_text(
            json.dumps(
                {
                    "api_key_file": str(context_key),
                    "api_url": api_url,
                    "deployment_id": "context-test",
                    "region": "us-west-2",
                    "schema_version": 1,
                    "status": "ready",
                }
            )
        )
        assert main(["cloud", "context", "use", str(context_path)]) == 0
        capsys.readouterr()
        assert main(["cloud", "applications", "list"]) == 0
        capsys.readouterr()
        assert requests[-1] == (
            "GET",
            "/v1/applications",
            "Bearer context-credential",
        )

        assert main(["cloud", "context", "clear"]) == 0
        capsys.readouterr()
        assert main(["cloud", "applications", "list"]) == 0
        capsys.readouterr()
        assert requests[-1] == (
            "GET",
            "/v1/applications",
            f"Bearer {access_two}",
        )

        assert main(["cloud", "logout"]) == 0
        logout = json.loads(capsys.readouterr().out)
        assert logout["result"] == {"signed_out": True, "status": "signed_out"}
        assert not auth_path.exists()
    finally:
        server.shutdown()
        thread.join()

    assert authentication_forms == [
        {
            "client_id": ["client_test"],
            "device_code": [device_code],
            "grant_type": ["urn:ietf:params:oauth:grant-type:device_code"],
        },
        {
            "client_id": ["client_test"],
            "grant_type": ["refresh_token"],
            "refresh_token": [refresh_one],
        },
    ]
    assert ("GET", "/v1/me", f"Bearer {access_one}") in requests
    assert ("GET", "/v1/applications", f"Bearer {access_two}") in requests
    assert ("GET", "/v1/me", f"Bearer {access_two}") in requests


def test_cloud_login_uses_explicit_context_and_rejects_a_portal_without_workos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            payload = json.dumps({"authentication": "api_key"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(tmp_path / "cloud-auth.json"))
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_API_URL", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    api_key_file = tmp_path / "api-key"
    api_key_file.write_text("customer-secret-material\n")
    context_path = tmp_path / "cloud-context.json"
    context_path.write_text(
        json.dumps(
            {
                "api_key_file": str(api_key_file),
                "api_url": f"http://127.0.0.1:{server.server_port}",
                "deployment_id": "cloud-test",
                "region": "us-west-2",
                "schema_version": 1,
                "status": "ready",
            }
        )
    )
    try:
        result = main(["cloud", "--context", str(context_path), "login", "--no-browser"])
    finally:
        server.shutdown()
        thread.join()

    assert result == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error == {
        "category": "workos_unavailable",
        "message": "This Cayu Cloud does not offer WorkOS CLI authentication.",
    }


def test_workos_device_polling_respects_pending_and_slow_down(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    responses = [
        (400, {"error": "authorization_pending"}),
        (400, {"error": "slow_down"}),
        (
            200,
            {
                "access_token": _jwt(expires_at=time.time() + 3600, marker="poll"),
                "organization_id": "org_test",
                "refresh_token": "private-refresh-token",
                "user": {"id": "user_test"},
            },
        ),
    ]
    server, thread, forms = _start_authentication_server(responses)
    sleeps: list[float] = []
    monkeypatch.setattr(cloud_auth.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(cloud_auth.time, "sleep", sleeps.append)
    try:
        credentials = WorkOSDeviceAuthClient(
            api_url=f"http://127.0.0.1:{server.server_port}"
        ).poll_for_credentials(
            _device_authorization(),
            client_id="client_test",
            api_hostname=f"127.0.0.1:{server.server_port}",
        )
    finally:
        server.shutdown()
        thread.join()

    assert credentials.organization_id == "org_test"
    assert sleeps == [2, 3]
    assert [form["grant_type"] for form in forms] == [
        ["urn:ietf:params:oauth:grant-type:device_code"],
        ["urn:ietf:params:oauth:grant-type:device_code"],
        ["urn:ietf:params:oauth:grant-type:device_code"],
    ]
    streams = capsys.readouterr()
    assert streams.out == ""
    assert streams.err == ""


@pytest.mark.parametrize(
    ("workos_error", "category"),
    [("access_denied", "login_denied"), ("expired_token", "login_expired")],
)
def test_workos_device_polling_stops_on_terminal_errors(
    workos_error: str,
    category: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server, thread, _ = _start_authentication_server(
        [(400, {"error": workos_error, "device_code": "must-not-leak"})]
    )
    try:
        with pytest.raises(CloudAuthError) as raised:
            WorkOSDeviceAuthClient(
                api_url=f"http://127.0.0.1:{server.server_port}"
            ).poll_for_credentials(
                _device_authorization(),
                client_id="client_test",
                api_hostname=f"127.0.0.1:{server.server_port}",
            )
    finally:
        server.shutdown()
        thread.join()

    assert raised.value.category == category
    assert "must-not-leak" not in str(raised.value)
    streams = capsys.readouterr()
    assert "must-not-leak" not in streams.out + streams.err


def test_workos_device_polling_stops_at_authorization_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, thread, _ = _start_authentication_server([(400, {"error": "authorization_pending"})])
    monotonic_values = iter((0.0, 0.0, 301.0))
    monkeypatch.setattr(cloud_auth.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(cloud_auth.time, "sleep", lambda _seconds: None)
    try:
        with pytest.raises(CloudAuthError) as raised:
            WorkOSDeviceAuthClient(
                api_url=f"http://127.0.0.1:{server.server_port}"
            ).poll_for_credentials(
                _device_authorization(),
                client_id="client_test",
                api_hostname=f"127.0.0.1:{server.server_port}",
            )
    finally:
        server.shutdown()
        thread.join()

    assert raised.value.category == "login_expired"
