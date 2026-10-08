"""A Cloud request rejected with HTTP 401 refreshes the WorkOS login once and retries.

The CLI refreshes the login before each request when the token's own expiry is near,
but that check uses the local clock. When the clock runs behind Cayu Cloud's, or the
token is rejected early, Cloud answers 401 mid-command, for example during a long
`deploy --wait` (cayu#2256).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.cli.test_cloud_auth import _jwt

from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_auth import (
    CloudAuthCredentials,
    CloudAuthStore,
    WorkOSDeviceAuthClient,
    fresh_cloud_credentials,
)


@contextmanager
def _cloud(accepted: Callable[[str | None], bool]) -> Iterator[tuple[str, list[str | None]]]:
    """Serve GETs, answering 401 unless ``accepted`` admits the Authorization header."""

    headers: list[str | None] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            authorization = self.headers.get("Authorization")
            headers.append(authorization)
            status = 200 if accepted(authorization) else 401
            payload = json.dumps({"ok": status == 200}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", headers
    finally:
        server.shutdown()
        thread.join()


def _credentials(api_url: str, *, marker: str, expires_in: float) -> CloudAuthCredentials:
    expires_at = time.time() + expires_in
    return CloudAuthCredentials(
        api_url=api_url,
        workos_api_hostname="auth.example.test",
        workos_client_id="client_test",
        access_token=_jwt(expires_at=expires_at, marker=marker),
        refresh_token=f"refresh-token-{marker}",
        expires_at=expires_at,
        organization_id="org_test",
        user_id="user_test",
    )


def test_rejected_token_forces_a_refresh_before_its_expiry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    current = _credentials("https://cloud.cayu.dev", marker="current", expires_in=3600)
    refreshed = _credentials("https://cloud.cayu.dev", marker="refreshed", expires_in=7200)
    store.save(current)
    calls: list[str] = []

    def refresh(_client: WorkOSDeviceAuthClient, credentials: CloudAuthCredentials):
        calls.append(credentials.access_token)
        return refreshed

    monkeypatch.setattr(WorkOSDeviceAuthClient, "refresh", refresh)

    # Not near expiry by the local clock: no refresh.
    assert fresh_cloud_credentials(store, timeout_seconds=5.0) == current
    assert calls == []

    result = fresh_cloud_credentials(
        store, timeout_seconds=5.0, rejected_access_token=current.access_token
    )
    assert result == refreshed
    assert calls == [current.access_token]
    assert store.load() == refreshed


def test_rejected_token_already_replaced_is_not_refreshed_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = CloudAuthStore(tmp_path / "cloud-auth.json")
    newer = _credentials("https://cloud.cayu.dev", marker="newer", expires_in=3600)
    store.save(newer)

    def refresh(*_args: object) -> CloudAuthCredentials:
        raise AssertionError("another command already refreshed the login")

    monkeypatch.setattr(WorkOSDeviceAuthClient, "refresh", refresh)

    stale = _jwt(expires_at=time.time() + 60, marker="stale")
    assert fresh_cloud_credentials(store, timeout_seconds=5.0, rejected_access_token=stale) == (
        newer
    )


def test_client_retries_once_with_the_replacement_key() -> None:
    with _cloud(lambda header: header == "Bearer second") as (api_url, headers):
        replacements: list[str] = []

        def replacement(rejected: str) -> str:
            replacements.append(rejected)
            return "second"

        client = CloudApiClient(
            api_url=api_url, api_key="first", rejected_api_key_provider=replacement
        )
        assert client.request("GET", "/v1/applications") == {"ok": True}

    assert headers == ["Bearer first", "Bearer second"]
    assert replacements == ["first"]


@pytest.mark.parametrize("replacement", ["first", "second"], ids=["unchanged", "also-rejected"])
def test_client_reports_the_rejection_without_retrying_again(replacement: str) -> None:
    with _cloud(lambda _header: False) as (api_url, headers):
        client = CloudApiClient(
            api_url=api_url,
            api_key="first",
            rejected_api_key_provider=lambda _rejected: replacement,
        )
        with pytest.raises(CloudApiError) as raised:
            client.request("GET", "/v1/applications")

    assert raised.value.status_code == 401
    assert headers == (
        ["Bearer first"] if replacement == "first" else ["Bearer first", "Bearer second"]
    )


def test_client_without_a_replacement_provider_does_not_retry() -> None:
    with _cloud(lambda _header: False) as (api_url, headers), pytest.raises(CloudApiError):
        CloudApiClient(api_url=api_url, api_key="first").request("GET", "/v1/applications")

    assert headers == ["Bearer first"]


def test_workos_login_client_refreshes_a_token_cloud_rejects_early(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    refreshed_marker: dict[str, str] = {}

    with _cloud(lambda header: header == refreshed_marker.get("header")) as (api_url, headers):
        # Valid for an hour by the local clock, but Cloud already rejects it.
        current = _credentials(api_url, marker="current", expires_in=3600)
        refreshed = _credentials(api_url, marker="refreshed", expires_in=7200)
        refreshed_marker["header"] = f"Bearer {refreshed.access_token}"
        auth_path = tmp_path / "cloud-auth.json"
        CloudAuthStore(auth_path).save(current)
        monkeypatch.setenv("CAYU_CLOUD_AUTH", str(auth_path))
        for name in (
            "CAYU_CLOUD_API_KEY",
            "CAYU_CLOUD_API_KEY_FILE",
            "CAYU_CLOUD_CONTEXT",
            "CAYU_CLOUD_API_URL",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(cloud_cli, "_PRODUCTION_API_URL", api_url)
        monkeypatch.setattr(
            WorkOSDeviceAuthClient, "refresh", lambda _client, _credentials: refreshed
        )

        client = cloud_cli._cloud_client(
            SimpleNamespace(api_key_file=None, context=None, timeout_seconds=5.0),
            context={},
            context_path=None,
        )
        assert client.request("GET", "/v1/applications/smoke-agent/service") == {"ok": True}
        # Later polls use the refreshed login directly.
        assert client.request("GET", "/v1/applications/smoke-agent/service") == {"ok": True}

    assert headers == [
        f"Bearer {current.access_token}",
        f"Bearer {refreshed.access_token}",
        f"Bearer {refreshed.access_token}",
    ]
    assert CloudAuthStore(auth_path).load() == refreshed
