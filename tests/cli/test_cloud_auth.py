from __future__ import annotations

import asyncio
import base64
import json
import os
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest

from cayu.cli import _cloud_auth as cloud_auth
from cayu.cli import _cloud_private_state as cloud_private_state
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


def _private_json_staging_paths(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.iterdir()
        if path.name.startswith(
            (
                cloud_private_state._PRIVATE_JSON_STAGING_PREFIX,
                cloud_auth._AUTH_REFRESH_STAGING_PREFIX,
            )
        )
    ]


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


def _start_refresh_and_deployment_server(
    refresh_responses: list[tuple[int, dict[str, object]]],
) -> tuple[
    ThreadingHTTPServer,
    threading.Thread,
    list[dict[str, list[str]]],
    list[str | None],
]:
    forms: list[dict[str, list[str]]] = []
    deployment_authorizations: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            deployment_authorizations.append(self.headers.get("Authorization"))
            payload = json.dumps({"id": "dep_running", "status": "image_built"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            forms.append(parse_qs(self.rfile.read(length).decode()))
            status_code, payload_value = refresh_responses.pop(0)
            payload = json.dumps(payload_value).encode()
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, forms, deployment_authorizations


def _configure_saved_workos_login(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    api_url: str,
    expires_at: float,
) -> tuple[CloudAuthStore, CloudAuthCredentials]:
    credentials = CloudAuthCredentials(
        api_url=api_url,
        workos_api_hostname=api_url.removeprefix("http://"),
        workos_client_id="client_test",
        access_token=_jwt(expires_at=expires_at, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=expires_at,
        organization_id="org_test",
        user_id="user_test",
    )
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    store.save(credentials)
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(store.path))
    monkeypatch.delenv("CAYU_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_KEY_FILE", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_URL", raising=False)
    monkeypatch.setattr(cloud_cli, "_PRODUCTION_API_URL", api_url)
    return store, credentials


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


def test_workos_cloud_client_refreshes_credentials_during_one_long_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    access_one = _jwt(expires_at=time.time() + 3600, marker="first")
    access_two = _jwt(expires_at=time.time() + 7200, marker="second")
    credentials_one = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=access_one,
        refresh_token="refresh-token-one",
        expires_at=time.time() + 3600,
        organization_id="org_test",
        user_id="user_test",
    )
    credentials_two = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=access_two,
        refresh_token="refresh-token-two",
        expires_at=time.time() + 7200,
        organization_id="org_test",
        user_id="user_test",
    )
    authorization_headers: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            authorization_headers.append(self.headers.get("Authorization"))
            payload = b"{}"
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
    api_url = f"http://127.0.0.1:{server.server_port}"
    credentials_one = replace(credentials_one, api_url=api_url)
    credentials_two = replace(credentials_two, api_url=api_url)
    auth_path = tmp_path / "cloud-auth.json"
    CloudAuthStore(auth_path).save(credentials_one)
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(auth_path))
    monkeypatch.delenv("CAYU_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_KEY_FILE", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_URL", raising=False)
    monkeypatch.setattr(
        cloud_cli,
        "_PRODUCTION_API_URL",
        api_url,
    )
    refreshed = iter((credentials_one, credentials_one, credentials_two))
    monkeypatch.setattr(
        cloud_cli,
        "fresh_cloud_credentials",
        lambda *_args, **_kwargs: next(refreshed),
    )
    try:
        client = cloud_cli._cloud_client(
            SimpleNamespace(api_key_file=None, context=None, timeout_seconds=30.0),
            context={},
            context_path=None,
        )
        client.request("GET", "/first")
        client.request("GET", "/second")
    finally:
        server.shutdown()
        thread.join()

    assert authorization_headers == [
        f"Bearer {access_one}",
        f"Bearer {access_two}",
    ]


@pytest.mark.parametrize("transient_status", [None, 429, 503], ids=["network", "429", "503"])
def test_workos_transient_refresh_failure_preserves_running_deployment_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transient_status: int | None,
) -> None:
    refresh_responses = (
        []
        if transient_status is None
        else [(transient_status, {"error": "temporary"}) for _ in range(3)]
    )
    server, thread, forms, deployment_authorizations = _start_refresh_and_deployment_server(
        refresh_responses
    )
    api_url = f"http://127.0.0.1:{server.server_port}"
    now = time.time()
    store, credentials = _configure_saved_workos_login(
        monkeypatch,
        tmp_path,
        api_url=api_url,
        expires_at=now + 3600,
    )
    client = cloud_cli._cloud_client(
        SimpleNamespace(api_key_file=None, context=None, timeout_seconds=1.0),
        context={},
        context_path=None,
    )
    network_attempts: list[str] = []
    if transient_status is None:

        def unavailable_request(
            _client: WorkOSDeviceAuthClient,
            _method: str,
            url: str,
            *,
            form: dict[str, str] | None = None,
            timeout_seconds: float | None = None,
            total_timeout_seconds: float | None = None,
        ) -> object:
            del form, timeout_seconds, total_timeout_seconds
            network_attempts.append(url)
            raise CloudAuthError(
                "login_unavailable",
                "Cayu Cloud authentication service is unavailable.",
            )

        monkeypatch.setattr(WorkOSDeviceAuthClient, "_request", unavailable_request)
    sleeps: list[float] = []
    monkeypatch.setattr(cloud_auth.time, "sleep", sleeps.append)
    monkeypatch.setattr(cloud_auth.time, "time", lambda: now + 3570)
    times = iter((0.0, 10.0))
    try:
        with pytest.raises(cloud_cli._CloudDeploymentStillRunningError) as raised:
            cloud_cli._wait_for_deployment(
                client,
                application_id="outbound-agent",
                deployment_id="dep_running",
                poll_seconds=0.01,
                wait_seconds=1.0,
                sleep=lambda _seconds: None,
                monotonic=lambda: next(times),
            )
    finally:
        server.shutdown()
        thread.join()

    assert raised.value.category == "deployment_still_running"
    assert raised.value.public_details()["deployment_id"] == "dep_running"
    assert sleeps == list(cloud_auth._REFRESH_RETRY_DELAYS_SECONDS)
    assert (len(network_attempts) if transient_status is None else len(forms)) == 3
    assert deployment_authorizations == [f"Bearer {credentials.access_token}"]
    assert store.load() == credentials


def test_workos_transient_refresh_retries_and_persists_rotated_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    refreshed_access_token = _jwt(expires_at=now + 7200, marker="refreshed")
    server, thread, forms, deployment_authorizations = _start_refresh_and_deployment_server(
        [
            (503, {"error": "temporarily_unavailable"}),
            (
                200,
                {
                    "access_token": refreshed_access_token,
                    "organization_id": "org_test",
                    "refresh_token": "refresh-token-refreshed",
                    "user": {"id": "user_test"},
                },
            ),
        ]
    )
    api_url = f"http://127.0.0.1:{server.server_port}"
    store, _ = _configure_saved_workos_login(
        monkeypatch,
        tmp_path,
        api_url=api_url,
        expires_at=now + 3600,
    )
    client = cloud_cli._cloud_client(
        SimpleNamespace(api_key_file=None, context=None, timeout_seconds=1.0),
        context={},
        context_path=None,
    )
    sleeps: list[float] = []
    monkeypatch.setattr(cloud_auth.time, "sleep", sleeps.append)
    monkeypatch.setattr(cloud_auth.time, "time", lambda: now + 3570)
    times = iter((0.0, 10.0))
    try:
        with pytest.raises(cloud_cli._CloudDeploymentStillRunningError):
            cloud_cli._wait_for_deployment(
                client,
                application_id="outbound-agent",
                deployment_id="dep_running",
                poll_seconds=0.01,
                wait_seconds=1.0,
                sleep=lambda _seconds: None,
                monotonic=lambda: next(times),
            )
    finally:
        server.shutdown()
        thread.join()

    assert len(forms) == 2
    assert sleeps == [cloud_auth._REFRESH_RETRY_DELAYS_SECONDS[0]]
    assert deployment_authorizations == [f"Bearer {refreshed_access_token}"]
    persisted = store.load()
    assert persisted is not None
    assert persisted.access_token == refreshed_access_token
    assert persisted.refresh_token == "refresh-token-refreshed"


@pytest.mark.parametrize(
    "failed_reservation_call",
    [1, 2],
    ids=["staging", "cow-headroom"],
)
def test_workos_refresh_storage_preflight_failure_precedes_token_rotation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failed_reservation_call: int,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    credentials = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    store.save(credentials)
    refresh_calls: list[str] = []

    def refresh(
        _client: WorkOSDeviceAuthClient,
        current: CloudAuthCredentials,
    ) -> CloudAuthCredentials:
        refresh_calls.append(current.refresh_token)
        return current

    real_overwrite = cloud_private_state._overwrite_payload
    reservation_calls = 0

    def fail_storage_reservation(descriptor: int, payload: bytes) -> None:
        nonlocal reservation_calls
        reservation_calls += 1
        if reservation_calls == failed_reservation_call:
            raise OSError("injected capacity failure")
        real_overwrite(descriptor, payload)

    monkeypatch.setattr(WorkOSDeviceAuthClient, "refresh", refresh)
    monkeypatch.setattr(
        cloud_private_state,
        "_overwrite_payload",
        fail_storage_reservation,
    )

    with pytest.raises(CloudAuthError) as raised:
        cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert raised.value.category == "cloud_auth_unavailable"
    assert refresh_calls == []
    assert store.load() == credentials
    assert _private_json_staging_paths(tmp_path) == []


@pytest.mark.skipif(not hasattr(os, "pathconf"), reason="requires POSIX NAME_MAX")
def test_workos_refresh_supports_auth_filename_near_name_max(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    auth_path = tmp_path / ("a" * (name_max - 15))
    store = CloudAuthStore(auth_path)
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        original,
        access_token=_jwt(expires_at=now + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=now + 3600,
    )
    store.save(original)
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "refresh",
        lambda _client, current: replacement if current == original else current,
    )

    selected = cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert selected == replacement
    assert store.load() == replacement
    assert set(tmp_path.iterdir()) == {auth_path}


def test_workos_refresh_lock_teardown_failure_releases_staging_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    credentials = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    store.save(credentials)
    staged_descriptors: list[int] = []
    real_prepare = cloud_auth.prepare_private_json

    @contextmanager
    def recording_prepare(
        path: Path,
        payload: dict[str, object],
        *,
        staging_prefix: str,
    ):
        with real_prepare(path, payload, staging_prefix=staging_prefix) as prepared:
            assert prepared.headroom is not None
            staged_descriptors.extend([prepared.descriptor, prepared.headroom.fileno()])
            yield prepared

    @contextmanager
    def fail_lock_teardown():
        yield
        raise CloudAuthError(
            "cloud_auth_unavailable",
            "Could not update the local Cayu Cloud login.",
        )

    monkeypatch.setattr(cloud_auth, "prepare_private_json", recording_prepare)
    monkeypatch.setattr(store, "_exclusive_lock", fail_lock_teardown)

    with pytest.raises(CloudAuthError), store.prepare_refresh(credentials):
        pytest.fail("lock teardown should fail before refresh starts")

    assert len(staged_descriptors) == 2
    for descriptor in staged_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert _private_json_staging_paths(tmp_path) == []


def test_private_json_windows_replacement_requests_write_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, Path, int]] = []
    source = Path("private-staging.json")
    destination = Path("private-auth.json")

    monkeypatch.setattr(cloud_private_state.os, "name", "nt")
    monkeypatch.setattr(
        cloud_private_state,
        "_move_file_ex_windows",
        lambda source, destination, *, flags: calls.append((source, destination, flags)),
    )

    cloud_private_state._replace_private_json(source, destination)

    assert calls == [
        (
            source,
            destination,
            cloud_private_state._WINDOWS_MOVEFILE_REPLACE_EXISTING
            | cloud_private_state._WINDOWS_MOVEFILE_WRITE_THROUGH,
        )
    ]


def test_workos_refresh_releases_staging_descriptor_before_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        original,
        access_token=_jwt(expires_at=now + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=now + 3600,
    )
    store.save(original)
    staging_descriptors: list[int] = []
    real_prepare = cloud_auth.prepare_private_json
    real_replace = cloud_private_state._replace_private_json

    @contextmanager
    def recording_prepare(
        path: Path,
        payload: dict[str, object],
        *,
        staging_prefix: str,
    ):
        with real_prepare(path, payload, staging_prefix=staging_prefix) as prepared:
            assert prepared.headroom is not None
            staging_descriptors.extend([prepared.descriptor, prepared.headroom.fileno()])
            yield prepared

    def replace_after_asserting_descriptor_closed(source: Path, destination: Path) -> None:
        assert len(staging_descriptors) == 2
        for descriptor in staging_descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
        real_replace(source, destination)

    monkeypatch.setattr(cloud_auth, "prepare_private_json", recording_prepare)
    monkeypatch.setattr(
        cloud_private_state,
        "_replace_private_json",
        replace_after_asserting_descriptor_closed,
    )
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "refresh",
        lambda _client, current: replacement if current == original else current,
    )

    selected = cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert selected == replacement
    assert store.load() == replacement
    assert _private_json_staging_paths(tmp_path) == []


def test_workos_refresh_releases_anonymous_cow_headroom_before_rotated_token_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        original,
        access_token=_jwt(expires_at=now + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=now + 3600,
    )
    store.save(original)
    prepared_writes: list[cloud_private_state.PreparedPrivateJsonWrite] = []
    headroom_descriptors: list[int] = []
    cow_headroom_bytes: list[int] = []
    real_prepare = cloud_auth.prepare_private_json
    real_overwrite_reserved = cloud_private_state._overwrite_reserved_payload

    @contextmanager
    def recording_prepare(
        path: Path,
        payload: dict[str, object],
        *,
        staging_prefix: str,
    ):
        with real_prepare(path, payload, staging_prefix=staging_prefix) as prepared:
            prepared_writes.append(prepared)
            yield prepared

    def cow_sensitive_overwrite(
        descriptor: int,
        payload: bytes,
        *,
        reserved_bytes: int,
    ) -> None:
        assert len(prepared_writes) == 1
        prepared = prepared_writes[0]
        assert prepared.headroom_released
        assert prepared.headroom is None
        assert len(headroom_descriptors) == 1
        with pytest.raises(OSError):
            os.fstat(headroom_descriptors[0])
        assert cow_headroom_bytes == [
            reserved_bytes + cloud_private_state._PRIVATE_JSON_COW_METADATA_HEADROOM_BYTES
        ]
        real_overwrite_reserved(
            descriptor,
            payload,
            reserved_bytes=reserved_bytes,
        )

    def refresh(
        _client: WorkOSDeviceAuthClient,
        current: CloudAuthCredentials,
    ) -> CloudAuthCredentials:
        assert current == original
        assert len(prepared_writes) == 1
        prepared = prepared_writes[0]
        assert prepared.headroom is not None
        headroom_descriptor = prepared.headroom.fileno()
        headroom_descriptors.append(headroom_descriptor)
        cow_headroom_bytes.append(os.fstat(headroom_descriptor).st_size)
        return replacement

    monkeypatch.setattr(cloud_auth, "prepare_private_json", recording_prepare)
    monkeypatch.setattr(
        cloud_private_state,
        "_overwrite_reserved_payload",
        cow_sensitive_overwrite,
    )
    monkeypatch.setattr(WorkOSDeviceAuthClient, "refresh", refresh)

    selected = cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert selected == replacement
    assert store.load() == replacement
    assert len(prepared_writes) == 1
    assert prepared_writes[0].published
    assert _private_json_staging_paths(tmp_path) == []


@pytest.mark.parametrize("failure_point", ["write", "fsync"])
@pytest.mark.parametrize("persistent", [False, True])
def test_workos_refresh_retries_rotated_payload_preparation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
    persistent: bool,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        original,
        access_token=_jwt(expires_at=now + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=now + 3600,
    )
    store.save(original)
    prepared_descriptor: list[int] = []
    preparation_failures: list[OSError] = []
    real_prepare = cloud_auth.prepare_private_json
    real_overwrite = cloud_private_state._overwrite_reserved_payload
    real_fsync = cloud_private_state.os.fsync

    @contextmanager
    def recording_prepare(
        path: Path,
        payload: dict[str, object],
        *,
        staging_prefix: str,
    ):
        with real_prepare(path, payload, staging_prefix=staging_prefix) as prepared:
            prepared_descriptor.append(prepared.descriptor)
            yield prepared

    def should_fail(descriptor: int) -> bool:
        return prepared_descriptor == [descriptor] and (persistent or not preparation_failures)

    def fail_payload_write(
        descriptor: int,
        payload: bytes,
        *,
        reserved_bytes: int,
    ) -> None:
        if failure_point == "write" and should_fail(descriptor):
            failure = OSError(
                f"injected rotated payload write failure {len(preparation_failures) + 1}"
            )
            preparation_failures.append(failure)
            raise failure
        real_overwrite(descriptor, payload, reserved_bytes=reserved_bytes)

    def fail_payload_fsync(descriptor: int) -> None:
        if failure_point == "fsync" and should_fail(descriptor):
            failure = OSError(
                f"injected rotated payload fsync failure {len(preparation_failures) + 1}"
            )
            preparation_failures.append(failure)
            raise failure
        real_fsync(descriptor)

    monkeypatch.setattr(cloud_auth, "prepare_private_json", recording_prepare)
    monkeypatch.setattr(
        cloud_private_state,
        "_overwrite_reserved_payload",
        fail_payload_write,
    )
    monkeypatch.setattr(cloud_private_state.os, "fsync", fail_payload_fsync)
    monkeypatch.setattr(cloud_private_state.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "refresh",
        lambda _client, current: replacement if current == original else current,
    )

    if persistent:
        with pytest.raises(CloudAuthError) as raised:
            cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)
        assert raised.value.category == "cloud_auth_unavailable"
        cause = raised.value.__cause__
        assert isinstance(cause, ExceptionGroup)
        assert list(cause.exceptions) == preparation_failures
        assert store.load() == original
    else:
        selected = cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)
        assert selected == replacement
        assert store.load() == replacement

    assert len(preparation_failures) == (2 if persistent else 1)
    assert _private_json_staging_paths(tmp_path) == []


def test_workos_refresh_retries_post_replacement_sync_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        original,
        access_token=_jwt(expires_at=now + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=now + 3600,
    )
    store.save(original)
    sync_calls: list[Path] = []

    real_sync = cloud_private_state._sync_private_json_directory

    def fail_once_post_replacement_directory_sync(path: Path) -> None:
        sync_calls.append(path)
        assert CloudAuthStore(path).load() == replacement
        if len(sync_calls) == 1:
            raise OSError("injected post-replacement sync failure")
        real_sync(path)

    monkeypatch.setattr(
        cloud_private_state,
        "_sync_private_json_directory",
        fail_once_post_replacement_directory_sync,
    )
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "refresh",
        lambda _client, current: replacement if current == original else current,
    )

    selected = cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert selected == replacement
    assert sync_calls == [store.path, store.path]
    assert store.load() == replacement
    assert _private_json_staging_paths(tmp_path) == []


def test_workos_refresh_reconciles_replacement_commit_then_raise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        original,
        access_token=_jwt(expires_at=now + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=now + 3600,
    )
    store.save(original)
    real_replace = cloud_private_state._replace_private_json
    replace_calls: list[tuple[Path, Path]] = []

    def replace_then_raise(source: Path, destination: Path) -> None:
        replace_calls.append((source, destination))
        real_replace(source, destination)
        raise OSError("injected replacement acknowledgement loss")

    monkeypatch.setattr(
        cloud_private_state,
        "_replace_private_json",
        replace_then_raise,
    )
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "refresh",
        lambda _client, current: replacement if current == original else current,
    )

    selected = cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert selected == replacement
    assert len(replace_calls) == 1
    assert store.load() == replacement
    assert _private_json_staging_paths(tmp_path) == []


def test_workos_refresh_retries_replacement_failure_before_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        original,
        access_token=_jwt(expires_at=now + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=now + 3600,
    )
    store.save(original)
    real_replace = cloud_private_state._replace_private_json
    replace_calls: list[tuple[Path, Path]] = []
    retry_delays: list[float] = []

    def fail_once_before_replacement(source: Path, destination: Path) -> None:
        replace_calls.append((source, destination))
        if len(replace_calls) == 1:
            raise OSError("injected transient replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(
        cloud_private_state,
        "_replace_private_json",
        fail_once_before_replacement,
    )
    monkeypatch.setattr(cloud_private_state.time, "sleep", retry_delays.append)
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "refresh",
        lambda _client, current: replacement if current == original else current,
    )

    selected = cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert selected == replacement
    assert len(replace_calls) == 2
    assert retry_delays == [cloud_private_state._PRIVATE_JSON_PUBLICATION_RETRY_DELAY_SECONDS]
    assert store.load() == replacement
    assert _private_json_staging_paths(tmp_path) == []


def test_workos_refresh_preserves_both_persistent_replacement_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        original,
        access_token=_jwt(expires_at=now + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=now + 3600,
    )
    store.save(original)
    replace_calls: list[OSError] = []

    def fail_replacement(_source: Path, _destination: Path) -> None:
        failure = OSError(f"injected replacement failure {len(replace_calls) + 1}")
        replace_calls.append(failure)
        raise failure

    monkeypatch.setattr(
        cloud_private_state,
        "_replace_private_json",
        fail_replacement,
    )
    monkeypatch.setattr(cloud_private_state.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "refresh",
        lambda _client, current: replacement if current == original else current,
    )

    with pytest.raises(CloudAuthError) as raised:
        cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert raised.value.category == "cloud_auth_unavailable"
    cause = raised.value.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert cause.exceptions == tuple(replace_calls)
    assert replace_calls[1].__context__ is None
    assert store.load() == original
    assert _private_json_staging_paths(tmp_path) == []


def test_workos_logout_reclaims_rotated_credentials_after_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        original,
        access_token=_jwt(expires_at=now + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=now + 3600,
    )
    store.save(original)
    replacement_failures: list[OSError] = []
    cleanup_failures: list[OSError] = []
    fail_cleanup = True
    real_unlink = Path.unlink
    staging_prefix = cloud_auth._auth_refresh_staging_prefix(store.path)

    def fail_replacement(_source: Path, _destination: Path) -> None:
        failure = OSError(f"injected replacement failure {len(replacement_failures) + 1}")
        replacement_failures.append(failure)
        raise failure

    def fail_staging_cleanup(path: Path, *, missing_ok: bool = False) -> None:
        if fail_cleanup and path.name.startswith(staging_prefix):
            failure = OSError("injected credential staging cleanup failure")
            cleanup_failures.append(failure)
            raise failure
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(
        cloud_private_state,
        "_replace_private_json",
        fail_replacement,
    )
    monkeypatch.setattr(cloud_private_state.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(Path, "unlink", fail_staging_cleanup)
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "refresh",
        lambda _client, current: replacement if current == original else current,
    )

    with pytest.raises(CloudAuthError) as raised:
        cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert raised.value.category == "cloud_auth_unavailable"
    cause = raised.value.__cause__
    assert isinstance(cause, BaseExceptionGroup)
    assert list(cause.exceptions[0].exceptions) == replacement_failures
    assert cause.exceptions[1] is cleanup_failures[0]
    staging_paths = _private_json_staging_paths(tmp_path)
    assert len(staging_paths) == 1
    assert replacement.refresh_token in staging_paths[0].read_text()
    assert stat.S_IMODE(staging_paths[0].stat().st_mode) == 0o600

    fail_cleanup = False
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(store.path))
    assert main(["cloud", "logout"]) == 0

    logout = json.loads(capsys.readouterr().out)
    assert logout["result"] == {"signed_out": True, "status": "signed_out"}
    assert not store.path.exists()
    assert _private_json_staging_paths(tmp_path) == []


@pytest.mark.skipif(os.name == "nt", reason="symlink creation is privilege-dependent on Windows")
def test_workos_logout_does_not_follow_auth_staging_symlinks(
    tmp_path: Path,
) -> None:
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    target = tmp_path / "credential-canary"
    target.write_text("must-remain")
    staging = tmp_path / f"{cloud_auth._auth_refresh_staging_prefix(store.path)}canary"
    staging.symlink_to(target)

    with pytest.raises(CloudAuthError) as raised:
        store.delete()

    assert raised.value.category == "cloud_auth_unavailable"
    assert target.read_text() == "must-remain"
    assert staging.is_symlink()


@pytest.mark.parametrize("disappearance_phase", ["inspection", "unlink"])
def test_workos_logout_accepts_concurrently_completed_staging_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    disappearance_phase: str,
) -> None:
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    staging = tmp_path / f"{cloud_auth._auth_refresh_staging_prefix(store.path)}canary"
    staging.write_text("private rotated credentials")
    staging.chmod(0o600)
    real_scandir = cloud_private_state.os.scandir
    cleanup_requested = threading.Event()
    cleanup_complete = threading.Event()

    def cleanup() -> None:
        assert cleanup_requested.wait(timeout=5)
        staging.unlink()
        cleanup_complete.set()

    cleanup_thread = threading.Thread(target=cleanup)
    cleanup_thread.start()

    class ConcurrentCleanupEntry:
        def __init__(self, entry: os.DirEntry[str]) -> None:
            self._entry = entry
            self.name = entry.name
            self.path = entry.path

        def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
            if disappearance_phase == "inspection":
                cleanup_requested.set()
                assert cleanup_complete.wait(timeout=5)
            evidence = self._entry.stat(follow_symlinks=follow_symlinks)
            if disappearance_phase == "unlink":
                cleanup_requested.set()
                assert cleanup_complete.wait(timeout=5)
            return evidence

    class ConcurrentCleanupScan:
        def __enter__(self) -> Iterator[ConcurrentCleanupEntry]:
            self._scan = real_scandir(tmp_path)
            return iter(ConcurrentCleanupEntry(entry) for entry in self._scan)

        def __exit__(self, *args: object) -> None:
            self._scan.close()

    monkeypatch.setattr(
        cloud_private_state.os,
        "scandir",
        lambda _directory: ConcurrentCleanupScan(),
    )

    try:
        assert store.delete()
    finally:
        cleanup_requested.set()
        cleanup_thread.join(timeout=5)

    assert not cleanup_thread.is_alive()
    assert not staging.exists()


def test_workos_refresh_reports_persistent_post_replacement_sync_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        original,
        access_token=_jwt(expires_at=now + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=now + 3600,
    )
    store.save(original)
    sync_calls: list[Path] = []

    def fail_post_replacement_directory_sync(path: Path) -> None:
        sync_calls.append(path)
        assert CloudAuthStore(path).load() == replacement
        raise OSError("injected persistent post-replacement sync failure")

    monkeypatch.setattr(
        cloud_private_state,
        "_sync_private_json_directory",
        fail_post_replacement_directory_sync,
    )
    monkeypatch.setattr(
        WorkOSDeviceAuthClient,
        "refresh",
        lambda _client, current: replacement if current == original else current,
    )

    with pytest.raises(CloudAuthError) as raised:
        cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert raised.value.category == "cloud_auth_unavailable"
    assert sync_calls == [store.path, store.path]
    assert store.load() == replacement
    assert _private_json_staging_paths(tmp_path) == []


def test_workos_invalid_grant_stops_deployment_polling_with_relogin_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server, thread, forms, deployment_authorizations = _start_refresh_and_deployment_server(
        [(400, {"error": "invalid_grant", "detail": "must-not-leak"})]
    )
    api_url = f"http://127.0.0.1:{server.server_port}"
    now = time.time()
    _, credentials = _configure_saved_workos_login(
        monkeypatch,
        tmp_path,
        api_url=api_url,
        expires_at=now + 3600,
    )
    client = cloud_cli._cloud_client(
        SimpleNamespace(api_key_file=None, context=None, timeout_seconds=1.0),
        context={},
        context_path=None,
    )
    monkeypatch.setattr(cloud_auth.time, "time", lambda: now + 3570)
    try:
        with pytest.raises(CloudAuthError) as raised:
            cloud_cli._wait_for_deployment(
                client,
                application_id="outbound-agent",
                deployment_id="dep_running",
                poll_seconds=0.01,
                wait_seconds=1.0,
                sleep=lambda _seconds: None,
                monotonic=lambda: 0.0,
            )
    finally:
        server.shutdown()
        thread.join()

    assert raised.value.category == "login_refresh_failed"
    assert "run `cayu cloud login` again" in str(raised.value)
    assert "must-not-leak" not in str(raised.value)
    assert credentials.refresh_token not in str(raised.value)
    assert len(forms) == 1
    assert deployment_authorizations == []


def test_workos_transient_refresh_does_not_reuse_an_expired_access_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.time()
    credentials = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now - 1, marker="expired"),
        refresh_token="refresh-token-current",
        expires_at=now - 1,
        organization_id="org_test",
        user_id="user_test",
    )
    attempts: list[int] = []

    def unavailable_request(*_args: object, **_kwargs: object) -> object:
        attempts.append(1)
        raise CloudAuthError(
            "login_unavailable",
            "Cayu Cloud authentication service is unavailable.",
        )

    monkeypatch.setattr(WorkOSDeviceAuthClient, "_request", unavailable_request)
    monkeypatch.setattr(cloud_auth.time, "sleep", lambda _seconds: None)

    with pytest.raises(CloudAuthError) as raised:
        WorkOSDeviceAuthClient(api_url=credentials.api_url).refresh(credentials)

    assert raised.value.category == "login_refresh_unavailable"
    assert "retry this command" in str(raised.value)
    assert "cloud login" not in str(raised.value)
    assert len(attempts) == 3


def test_workos_refresh_replays_a_lost_rotation_with_the_default_cloud_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    original = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    refreshed_access_token = _jwt(expires_at=now + 3600, marker="refreshed")
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    store.save(original)
    attempts: list[str] = []
    sleeps: list[float] = []
    request_timeouts: list[float | None] = []
    total_request_timeouts: list[float | None] = []
    monotonic_now = [0.0]
    wall_now = [now]

    def commit_then_timeout(
        _client: WorkOSDeviceAuthClient,
        _method: str,
        _url: str,
        *,
        form: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        total_timeout_seconds: float | None = None,
    ) -> httpx.Response:
        del form
        request_timeouts.append(timeout_seconds)
        total_request_timeouts.append(total_timeout_seconds)
        if not attempts:
            attempts.append("externally_rotated_acknowledgement_lost")
            monotonic_now[0] += float(total_timeout_seconds or 0.0)
            wall_now[0] += float(total_timeout_seconds or 0.0)
            raise CloudAuthError(
                "login_unavailable",
                "Cayu Cloud authentication service is unavailable.",
            )
        attempts.append("replayed")
        return httpx.Response(
            200,
            json={
                "access_token": refreshed_access_token,
                "organization_id": original.organization_id,
                "refresh_token": "refresh-token-refreshed",
                "user": {"id": original.user_id},
            },
        )

    monkeypatch.setattr(WorkOSDeviceAuthClient, "_request", commit_then_timeout)
    monkeypatch.setattr(cloud_auth.time, "monotonic", lambda: monotonic_now[0])
    monkeypatch.setattr(cloud_auth.time, "time", lambda: wall_now[0])

    def advance_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        monotonic_now[0] += seconds
        wall_now[0] += seconds

    monkeypatch.setattr(cloud_auth.time, "sleep", advance_sleep)

    selected = cloud_auth.fresh_cloud_credentials(store, timeout_seconds=30.0)

    assert selected is not None
    assert selected.access_token == refreshed_access_token
    assert selected.refresh_token == "refresh-token-refreshed"
    assert store.load() == selected
    assert attempts == ["externally_rotated_acknowledgement_lost", "replayed"]
    assert request_timeouts == [
        cloud_auth._REFRESH_HTTP_ATTEMPT_TIMEOUT_SECONDS,
        cloud_auth._REFRESH_HTTP_ATTEMPT_TIMEOUT_SECONDS,
    ]
    assert total_request_timeouts == [
        cloud_auth._REFRESH_HTTP_ATTEMPT_TIMEOUT_SECONDS,
        cloud_auth._REFRESH_HTTP_ATTEMPT_TIMEOUT_SECONDS,
    ]
    assert sleeps == [cloud_auth._REFRESH_RETRY_DELAYS_SECONDS[0]]
    assert monotonic_now[0] < cloud_auth._REFRESH_RETRY_WINDOW_SECONDS


def test_workos_refresh_request_has_an_absolute_attempt_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_started = threading.Event()
    request_cancelled = threading.Event()
    client_exited = threading.Event()
    client_options: list[tuple[bool, float]] = []
    requests: list[tuple[str, str, dict[str, str] | None]] = []

    class BlockingAsyncClient:
        def __init__(self, *, follow_redirects: bool, timeout: float) -> None:
            client_options.append((follow_redirects, timeout))

        async def __aenter__(self) -> BlockingAsyncClient:
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc_value: object,
            traceback: object,
        ) -> None:
            client_exited.set()

        async def request(
            self,
            method: str,
            url: str,
            *,
            data: dict[str, str] | None,
        ) -> httpx.Response:
            requests.append((method, url, data))
            request_started.set()
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                request_cancelled.set()
                raise
            return httpx.Response(200)

    monkeypatch.setattr(cloud_auth.httpx, "AsyncClient", BlockingAsyncClient)

    with pytest.raises(CloudAuthError) as raised:
        WorkOSDeviceAuthClient(api_url="https://cloud.cayu.dev")._request(
            "POST",
            "https://auth.example.test/user_management/authenticate",
            form={"refresh_token": "refresh-token-current"},
            timeout_seconds=1.0,
            total_timeout_seconds=0.05,
        )

    assert raised.value.category == "login_unavailable"
    assert request_started.is_set()
    assert request_cancelled.is_set()
    assert client_exited.is_set()
    assert client_options == [(False, 1.0)]
    assert requests == [
        (
            "POST",
            "https://auth.example.test/user_management/authenticate",
            {"refresh_token": "refresh-token-current"},
        )
    ]


def test_workos_refresh_rechecks_the_replay_window_after_system_suspension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic_now = [0.0]
    wall_now = [1_000.0]
    credentials = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=wall_now[0] + 3600, marker="current"),
        refresh_token="refresh-token-current",
        expires_at=wall_now[0] + 3600,
        organization_id="org_test",
        user_id="user_test",
    )
    attempts: list[int] = []

    def unavailable_request(*_args: object, **_kwargs: object) -> object:
        attempts.append(1)
        raise CloudAuthError(
            "login_unavailable",
            "Cayu Cloud authentication service is unavailable.",
        )

    def suspended_sleep(delay: float) -> None:
        monotonic_now[0] += delay
        wall_now[0] += cloud_auth._REFRESH_RETRY_WINDOW_SECONDS + 1.0

    monkeypatch.setattr(WorkOSDeviceAuthClient, "_request", unavailable_request)
    monkeypatch.setattr(cloud_auth.time, "monotonic", lambda: monotonic_now[0])
    monkeypatch.setattr(cloud_auth.time, "time", lambda: wall_now[0])
    monkeypatch.setattr(cloud_auth.time, "sleep", suspended_sleep)

    selected = WorkOSDeviceAuthClient(api_url=credentials.api_url).refresh(credentials)

    assert selected is credentials
    assert attempts == [1]


def test_workos_transient_refresh_fallback_preserves_a_concurrent_login(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    now = time.time()
    initial = CloudAuthCredentials(
        api_url="https://cloud.cayu.dev",
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="initial"),
        refresh_token="refresh-token-initial",
        expires_at=now + 30,
        organization_id="org_initial",
        user_id="user_initial",
    )
    replacement = replace(
        initial,
        access_token=_jwt(expires_at=now + 3600, marker="replacement"),
        refresh_token="refresh-token-replacement",
        expires_at=now + 3600,
        organization_id="org_replacement",
        user_id="user_replacement",
    )
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    store.save(initial)

    def refresh(
        _client: WorkOSDeviceAuthClient,
        credentials: CloudAuthCredentials,
    ) -> CloudAuthCredentials:
        assert credentials == initial
        store.save(replacement)
        return credentials

    monkeypatch.setattr(WorkOSDeviceAuthClient, "refresh", refresh)

    selected = cloud_auth.fresh_cloud_credentials(store, timeout_seconds=1.0)

    assert selected == replacement
    assert store.load() == replacement


@pytest.mark.parametrize(
    "category",
    ["login_refresh_unavailable", "login_refresh_failed"],
)
def test_workos_refresh_error_uses_a_concurrent_same_authority_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    category: str,
) -> None:
    now = time.time()
    api_url = "https://cloud.example.test"
    initial = CloudAuthCredentials(
        api_url=api_url,
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="initial"),
        refresh_token="refresh-token-initial",
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        initial,
        access_token=_jwt(expires_at=now + 3600, marker="replacement"),
        refresh_token="refresh-token-replacement",
        expires_at=now + 3600,
    )
    auth_path = tmp_path / "cloud-auth.json"
    store = CloudAuthStore(auth_path)
    store.save(initial)
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(auth_path))
    monkeypatch.delenv("CAYU_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_KEY_FILE", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_URL", raising=False)
    monkeypatch.setattr(cloud_cli, "_PRODUCTION_API_URL", api_url)

    def refresh(
        _client: WorkOSDeviceAuthClient,
        credentials: CloudAuthCredentials,
    ) -> CloudAuthCredentials:
        assert credentials == initial
        store.save(replacement)
        raise CloudAuthError(category, "Refresh did not return credentials.")

    monkeypatch.setattr(WorkOSDeviceAuthClient, "refresh", refresh)

    client = cloud_cli._cloud_client(
        SimpleNamespace(api_key_file=None, context=None, timeout_seconds=1.0),
        context={},
        context_path=None,
    )

    assert client.api_key == replacement.access_token
    assert store.load() == replacement


def test_workos_refresh_reconciliation_handles_non_ascii_stored_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = time.time()
    api_url = "https://cloud.example.test"
    refresh_token = "réfresh-token-canary"
    credentials = CloudAuthCredentials(
        api_url=api_url,
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=now + 30, marker="current"),
        refresh_token=refresh_token,
        expires_at=now + 30,
        organization_id="org_test",
        user_id="user_test",
    )
    auth_path = tmp_path / "cloud-auth.json"
    CloudAuthStore(auth_path).save(credentials)
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(auth_path))
    monkeypatch.delenv("CAYU_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_KEY_FILE", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_URL", raising=False)
    monkeypatch.setattr(cloud_cli, "_PRODUCTION_API_URL", api_url)

    def fail_refresh(
        _client: WorkOSDeviceAuthClient,
        _credentials: CloudAuthCredentials,
    ) -> CloudAuthCredentials:
        raise CloudAuthError(
            "login_refresh_unavailable",
            "The Cayu Cloud login could not refresh temporarily; retry this command.",
        )

    monkeypatch.setattr(WorkOSDeviceAuthClient, "refresh", fail_refresh)

    assert main(["cloud", "whoami"]) == 2

    streams = capsys.readouterr()
    assert streams.err == ""
    assert json.loads(streams.out) == {
        "error": {
            "category": "login_refresh_unavailable",
            "message": ("The Cayu Cloud login could not refresh temporarily; retry this command."),
        },
        "ok": False,
    }
    assert refresh_token not in streams.out + streams.err


@pytest.mark.parametrize("authority_change", ["cloud", "principal"])
def test_workos_cloud_client_fences_changed_refresh_authority_before_transmission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    authority_change: str,
) -> None:
    access_one = _jwt(expires_at=time.time() + 3600, marker="first")
    access_replacement = _jwt(expires_at=time.time() + 7200, marker="replacement")
    authorization_headers: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            authorization_headers.append(self.headers.get("Authorization"))
            payload = b"{}"
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
    api_url = f"http://127.0.0.1:{server.server_port}"
    initial = CloudAuthCredentials(
        api_url=api_url,
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=access_one,
        refresh_token="refresh-token-one",
        expires_at=time.time() + 3600,
        organization_id="org_test",
        user_id="user_test",
    )
    replacement = replace(
        initial,
        access_token=access_replacement,
        refresh_token="refresh-token-replacement",
        expires_at=time.time() + 7200,
        **(
            {"api_url": "https://other-cloud.example.test"}
            if authority_change == "cloud"
            else {"organization_id": "org_replacement", "user_id": "user_replacement"}
        ),
    )
    auth_path = tmp_path / "cloud-auth.json"
    CloudAuthStore(auth_path).save(initial)
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(auth_path))
    monkeypatch.delenv("CAYU_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_KEY_FILE", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_URL", raising=False)
    monkeypatch.setattr(cloud_cli, "_PRODUCTION_API_URL", api_url)
    refreshed = iter((initial, replacement))
    monkeypatch.setattr(
        cloud_cli,
        "fresh_cloud_credentials",
        lambda *_args, **_kwargs: next(refreshed),
    )
    try:
        client = cloud_cli._cloud_client(
            SimpleNamespace(api_key_file=None, context=None, timeout_seconds=30.0),
            context={},
            context_path=None,
        )
        with pytest.raises(CloudAuthError, match="login authority changed") as exc_info:
            client.request("GET", "/must-not-run")
    finally:
        server.shutdown()
        thread.join()

    assert exc_info.value.category == "login_authority_changed"
    assert authorization_headers == []
    assert access_replacement not in str(exc_info.value)


def test_workos_cloud_client_preserves_concurrent_login_during_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api_url = "https://cloud.example.test"
    initial = CloudAuthCredentials(
        api_url=api_url,
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=time.time() + 30, marker="initial"),
        refresh_token="refresh-token-initial",
        expires_at=time.time() + 30,
        organization_id="org_initial",
        user_id="user_initial",
    )
    refreshed = replace(
        initial,
        access_token=_jwt(expires_at=time.time() + 3600, marker="refreshed"),
        refresh_token="refresh-token-refreshed",
        expires_at=time.time() + 3600,
    )
    replacement = replace(
        initial,
        access_token=_jwt(expires_at=time.time() + 7200, marker="replacement"),
        refresh_token="refresh-token-replacement",
        expires_at=time.time() + 7200,
        organization_id="org_replacement",
        user_id="user_replacement",
    )
    auth_path = tmp_path / "cloud-auth.json"
    store = CloudAuthStore(auth_path)
    store.save(initial)
    monkeypatch.setenv("CAYU_CLOUD_AUTH", str(auth_path))
    monkeypatch.delenv("CAYU_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_KEY_FILE", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_URL", raising=False)
    monkeypatch.setattr(cloud_cli, "_PRODUCTION_API_URL", api_url)

    def refresh(
        _client: WorkOSDeviceAuthClient,
        credentials: CloudAuthCredentials,
    ) -> CloudAuthCredentials:
        assert credentials == initial
        CloudAuthStore(auth_path).save(replacement)
        return refreshed

    monkeypatch.setattr(WorkOSDeviceAuthClient, "refresh", refresh)

    with pytest.raises(CloudAuthError, match="login authority changed") as exc_info:
        cloud_cli._cloud_client(
            SimpleNamespace(api_key_file=None, context=None, timeout_seconds=30.0),
            context={},
            context_path=None,
        )

    assert exc_info.value.category == "login_authority_changed"
    assert store.load() == replacement
    assert refreshed.access_token not in str(exc_info.value)
    assert replacement.access_token not in str(exc_info.value)


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
