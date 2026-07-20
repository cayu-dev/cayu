from __future__ import annotations

import asyncio
import json
import stat
import threading
import time
from base64 import urlsafe_b64encode
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from cayu.providers.openai_subscription import (
    OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID,
    HttpxOpenAISubscriptionOAuthTransport,
    OpenAISubscriptionAuth,
    OpenAISubscriptionAuthError,
    OpenAISubscriptionAuthStore,
    OpenAISubscriptionCredentials,
    build_openai_subscription_authorize_url,
    openai_subscription_credentials_from_token_response,
)


def test_subscription_credentials_round_trip_through_private_auth_store(tmp_path: Path) -> None:
    auth_path = tmp_path / "cayu" / "auth.json"
    store = OpenAISubscriptionAuthStore(auth_path)
    credentials = OpenAISubscriptionCredentials(
        access_token="access-token",
        refresh_token="refresh-token",
        expires_at=2_000_000_000.0,
        account_id="acct-cayu",
    )

    store.save(credentials)

    assert store.load() == credentials
    assert stat.S_IMODE(auth_path.stat().st_mode) == 0o600
    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert persisted["version"] == 1
    assert persisted["providers"]["openai_subscription"]["account_id"] == "acct-cayu"


def test_auth_store_preserves_permissions_of_existing_parent(tmp_path: Path) -> None:
    auth_home = tmp_path / "shared-cayu-home"
    auth_home.mkdir(mode=0o755)
    auth_home.chmod(0o755)

    assert OpenAISubscriptionAuthStore(auth_home / "auth.json").load() is None

    assert stat.S_IMODE(auth_home.stat().st_mode) == 0o755
    assert list(auth_home.iterdir()) == []


def test_auth_store_rejects_symlinked_existing_parent_without_side_effects(
    tmp_path: Path,
) -> None:
    target = tmp_path / "auth-target"
    target.mkdir()
    auth_home = tmp_path / "cayu-home"
    auth_home.symlink_to(target, target_is_directory=True)

    try:
        OpenAISubscriptionAuthStore(auth_home / "auth.json").load()
    except ValueError as exc:
        assert str(exc) == f"Refusing to use symlinked auth store directory: {auth_home}"
    else:
        raise AssertionError("symlinked auth store directory must fail validation")

    assert list(target.iterdir()) == []


def test_subscription_credentials_repr_redacts_tokens() -> None:
    credentials = OpenAISubscriptionCredentials(
        access_token="secret-access-token",
        refresh_token="secret-refresh-token",
        expires_at=2_000_000_000.0,
        account_id="acct-cayu",
    )

    rendered = repr(credentials)

    assert "secret-access-token" not in rendered
    assert "secret-refresh-token" not in rendered
    assert "acct-cayu" in rendered


def test_subscription_credentials_reject_non_finite_expiry() -> None:
    for expires_at in (float("inf"), float("-inf"), float("nan"), 10**400):
        try:
            OpenAISubscriptionCredentials(
                access_token="access-token",
                refresh_token="refresh-token",
                expires_at=expires_at,
            )
        except ValueError as exc:
            assert str(exc) == "expires_at must be finite and greater than zero."
        else:
            raise AssertionError(f"expires_at={expires_at!r} must fail")


def test_oauth_token_response_rejects_non_finite_expiry_duration() -> None:
    for expires_in in (float("inf"), float("-inf"), float("nan"), 10**400):
        try:
            openai_subscription_credentials_from_token_response(
                {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "expires_in": expires_in,
                },
                now=1_000.0,
            )
        except OpenAISubscriptionAuthError as exc:
            assert str(exc) == "OpenAI OAuth response contained invalid expires_in."
        else:
            raise AssertionError(f"expires_in={expires_in!r} must fail")


def test_subscription_authorize_url_uses_pkce_and_honest_cayu_originator() -> None:
    url = build_openai_subscription_authorize_url(
        redirect_uri="http://localhost:1455/auth/callback",
        code_challenge="challenge-value",
        state="csrf-state",
    )

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == (
        "https://auth.openai.com/oauth/authorize"
    )
    assert query["client_id"] == [OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == ["challenge-value"]
    assert query["state"] == ["csrf-state"]
    assert query["originator"] == ["cayu"]


def test_http_oauth_refresh_uses_codex_public_client_without_printing_tokens(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    response = HttpxOpenAISubscriptionOAuthTransport().refresh("old-refresh")

    assert response["access_token"] == "new-access"
    assert calls[0]["url"] == "https://auth.openai.com/oauth/token"
    assert calls[0]["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh",
        "client_id": OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID,
    }
    assert calls[0]["headers"]["user-agent"].startswith("cayu/")


class RecordingOAuthTransport:
    def __init__(self, response: dict[str, Any], *, delay_seconds: float = 0) -> None:
        self.response = response
        self.delay_seconds = delay_seconds
        self.refresh_tokens: list[str] = []

    def refresh(self, refresh_token: str) -> dict[str, Any]:
        self.refresh_tokens.append(refresh_token)
        time.sleep(self.delay_seconds)
        return self.response


def _jwt(claims: dict[str, Any]) -> str:
    def encode(value: dict[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{encode({'alg': 'none'})}.{encode(claims)}.signature"


def test_subscription_auth_refreshes_expiring_token_and_persists_rotation(
    tmp_path: Path,
) -> None:
    store = OpenAISubscriptionAuthStore(tmp_path / "auth.json")
    store.save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=time.time() - 1,
            account_id="old-account",
        )
    )
    transport = RecordingOAuthTransport(
        {
            "access_token": _jwt(
                {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-refreshed"}}
            ),
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        }
    )
    auth = OpenAISubscriptionAuth(store=store, oauth_transport=transport)

    credentials = asyncio.run(auth.credentials())

    assert transport.refresh_tokens == ["old-refresh"]
    assert credentials.refresh_token == "new-refresh"
    assert credentials.account_id == "acct-refreshed"
    assert credentials.expires_at > time.time() + 3500
    assert store.load() == credentials


def test_subscription_auth_serializes_refresh_across_auth_instances(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    OpenAISubscriptionAuthStore(auth_path).save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=time.time() - 1,
        )
    )
    transport = RecordingOAuthTransport(
        {
            "access_token": "new-access",
            "refresh_token": "new-refresh",
            "expires_in": 3600,
        },
        delay_seconds=0.05,
    )
    first = OpenAISubscriptionAuth(
        store=OpenAISubscriptionAuthStore(auth_path),
        oauth_transport=transport,
    )
    second = OpenAISubscriptionAuth(
        store=OpenAISubscriptionAuthStore(auth_path),
        oauth_transport=transport,
    )

    async def load_both() -> tuple[OpenAISubscriptionCredentials, ...]:
        return tuple(await asyncio.gather(first.credentials(), second.credentials()))

    credentials = asyncio.run(load_both())

    assert transport.refresh_tokens == ["old-refresh"]
    assert credentials[0] == credentials[1]
    assert credentials[0].refresh_token == "new-refresh"


def test_subscription_auth_does_not_block_event_loop_behind_refresh_lock(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    OpenAISubscriptionAuthStore(auth_path).save(
        OpenAISubscriptionCredentials(
            access_token="expired-access",
            refresh_token="old-refresh",
            expires_at=time.time() - 1,
        )
    )

    class BlockingOAuthTransport(RecordingOAuthTransport):
        def __init__(self) -> None:
            super().__init__(
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "expires_in": 3600,
                }
            )
            self.started = threading.Event()
            self.release = threading.Event()

        def refresh(self, refresh_token: str) -> dict[str, Any]:
            self.started.set()
            if not self.release.wait(timeout=2):
                raise AssertionError("test did not release the OAuth refresh")
            return super().refresh(refresh_token)

    transport = BlockingOAuthTransport()
    first = OpenAISubscriptionAuth(
        store=OpenAISubscriptionAuthStore(auth_path),
        oauth_transport=transport,
    )
    second = OpenAISubscriptionAuth(
        store=OpenAISubscriptionAuthStore(auth_path),
        oauth_transport=transport,
    )

    async def exercise() -> None:
        first_task = asyncio.create_task(first.credentials())
        assert await asyncio.to_thread(transport.started.wait, 1)
        release_timer = threading.Timer(0.25, transport.release.set)
        release_timer.start()
        try:
            second_task = asyncio.create_task(second.credentials())
            started_at = asyncio.get_running_loop().time()
            await asyncio.sleep(0.03)
            assert asyncio.get_running_loop().time() - started_at < 0.15
            assert not transport.release.is_set()
            transport.release.set()
            await asyncio.gather(first_task, second_task)
        finally:
            transport.release.set()
            release_timer.cancel()
            release_timer.join()

    asyncio.run(exercise())
