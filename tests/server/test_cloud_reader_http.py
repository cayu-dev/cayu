"""Cayu Cloud's reader: its own bearer token, the contract route and the activity feed."""

from __future__ import annotations

import asyncio
import hashlib

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest
from tests.core.test_cloud_reporting import ARGUMENT, _app, _start

from cayu import __version__
from cayu.cloud_reporting import CloudActivityPage, read_cloud_activity
from cayu.server import (
    AuthenticatedAccess,
    DashboardConfig,
    ServerApiConfig,
    ServerConfig,
    create_server,
    mount_cayu,
)
from cayu.server.auth import AuthConfigurationError, AuthContext
from cayu.server.cloud_reporting import (
    CLOUD_ACTIVITY_CURSOR_STALE,
    CLOUD_READER_MAX_TOKENS,
    CLOUD_READER_TOKEN_SHA256_VARIABLE,
    CloudReaderBearerAuth,
)
from cayu.server.fork_controller import ForkControllerBearerAuth

TOKEN = "cloud-reader-" + "a" * 40
ROTATED = "cloud-reader-" + "b" * 40
OPERATOR = "operator-" + "c" * 40
NO_STORE = "private, no-store"


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _operator(request: Request) -> AuthContext:
    if request.headers.get("authorization") != f"Bearer {OPERATOR}":
        raise HTTPException(status_code=401)
    return AuthContext(subject="operator")


@pytest.fixture
def reader(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(CLOUD_READER_TOKEN_SHA256_VARIABLE, raising=False)

    def enroll(value: str = f"{_digest(TOKEN)},{_digest(ROTATED)}") -> None:
        monkeypatch.setenv(CLOUD_READER_TOKEN_SHA256_VARIABLE, value)

    return enroll


def _server(**config) -> FastAPI:
    return create_server(
        _app(),
        config=ServerConfig.protected(
            _operator, dashboard=DashboardConfig(enabled=False), **config
        ),
    )


def test_without_digests_no_reader_route_is_served(reader) -> None:
    host = FastAPI()
    mount_cayu(host, _app(), access=AuthenticatedAccess(dependency=_operator), dashboard=False)

    assert (
        TestClient(host).get("/cayu/api/cloud/contract", headers=_bearer(TOKEN)).status_code == 404
    )
    assert (
        TestClient(_server()).get("/api/cloud/contract", headers=_bearer(TOKEN)).status_code == 404
    )


def test_the_reader_token_reads_the_contract_and_nothing_else(reader) -> None:
    reader()
    host = FastAPI()
    app = _app()
    mount_cayu(host, app, access=AuthenticatedAccess(dependency=_operator), dashboard=False)
    client = TestClient(host)
    path = "/cayu/api/cloud/contract"
    store_id = asyncio.run(app.session_store.read_store_identity())

    for token in (TOKEN, ROTATED):
        response = client.get(path, headers=_bearer(token))
        assert response.status_code == 200
        assert response.headers["cache-control"] == NO_STORE
        assert response.json() == {
            "contract": "cayu.cloud-agent",
            "versions": [1],
            "runtime_version": __version__,
            "store_id": store_id,
        }
    for headers in (
        {},
        _bearer(OPERATOR),
        _bearer("x" * 48),
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": f"Bearer {TOKEN} extra"},
    ):
        for route in (path, "/cayu/api/cloud/v1/activity"):
            response = client.get(route, headers=headers)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == "Bearer"
            assert response.headers["cache-control"] == NO_STORE
    repeated = client.get(path, headers=[("authorization", f"Bearer {TOKEN}")] * 2)
    assert repeated.status_code == 401
    # The reader's token is not a Developer dashboard login.
    assert client.get("/cayu/api/sessions", headers=_bearer(TOKEN)).status_code == 401
    assert client.get("/cayu/api/pending-actions", headers=_bearer(TOKEN)).status_code == 401


def test_activity_pages_over_http_carry_no_content(reader) -> None:
    reader(_digest(TOKEN))
    app = _app()
    server = create_server(
        app, config=ServerConfig.protected(_operator, dashboard=DashboardConfig(enabled=False))
    )
    with TestClient(server) as client:
        portal = client.portal
        assert portal is not None
        portal.call(_start, app, "approve", "approver")
        path = "/api/cloud/v1/activity"

        response = client.get(path, headers=_bearer(TOKEN))
        assert response.status_code == 200
        assert response.headers["cache-control"] == NO_STORE
        assert ARGUMENT not in response.text
        body = response.json()
        assert body["contract"] == "cayu.cloud-agent/v1"
        assert isinstance(body["runs"][0]["cost"]["amount"], str)
        page = CloudActivityPage.model_validate(body)
        assert [(run.session_id, run.status.value) for run in page.runs] == [
            ("approve", "waiting_approval")
        ]

        caught_up = client.get(
            path, params={"cursor": page.next_cursor, "limit": 1}, headers=_bearer(TOKEN)
        )
        assert caught_up.status_code == 200
        assert caught_up.json()["runs"] == []
        assert caught_up.json()["next_cursor"] == page.next_cursor

        for params in ({"cursor": "seven"}, {"cursor": "01"}):
            invalid = client.get(path, params=params, headers=_bearer(TOKEN))
            assert invalid.status_code == 422
            assert invalid.json() == {"detail": "Invalid activity cursor."}
            assert invalid.headers["cache-control"] == NO_STORE
        for params in ({"cursor": "1" * 83}, {"limit": 0}, {"limit": 201}):
            assert client.get(path, params=params, headers=_bearer(TOKEN)).status_code == 422


def test_a_cursor_the_store_cannot_continue_answers_410(reader) -> None:
    reader(_digest(TOKEN))
    replaced, app = _app(), _app()
    server = create_server(
        app, config=ServerConfig.protected(_operator, dashboard=DashboardConfig(enabled=False))
    )
    with TestClient(server) as client:
        portal = client.portal
        assert portal is not None
        portal.call(_start, replaced, "before", "approver")
        portal.call(_start, app, "after", "approver")
        stored = portal.call(read_cloud_activity, replaced)
        path = "/api/cloud/v1/activity"

        # A cursor of the store this one replaced, and one issued before cursors
        # named their store.
        for cursor in (stored.next_cursor, "7"):
            stale = client.get(path, params={"cursor": cursor}, headers=_bearer(TOKEN))
            assert stale.status_code == 410
            assert stale.json()["detail"]["code"] == CLOUD_ACTIVITY_CURSOR_STALE
            assert stale.headers["cache-control"] == NO_STORE
        restarted = client.get(path, headers=_bearer(TOKEN))
        assert [run["session_id"] for run in restarted.json()["runs"]] == ["after"]


def test_a_disabled_api_serves_no_reader_routes(reader) -> None:
    reader()
    client = TestClient(_server(api=ServerApiConfig(enabled=False)))

    assert client.get("/api/cloud/contract", headers=_bearer(TOKEN)).status_code == 404


@pytest.mark.parametrize(
    "value",
    [
        "",
        " , ",
        _digest(TOKEN).upper(),
        _digest(TOKEN)[:-1],
        TOKEN,
        f"{_digest(TOKEN)},{_digest(TOKEN)}",
        f"{_digest(TOKEN)},",
        ",".join(_digest(f"{TOKEN}-{index}") for index in range(CLOUD_READER_MAX_TOKENS + 1)),
    ],
)
def test_malformed_digests_stop_the_server_without_echoing_them(reader, value) -> None:
    reader(value)

    with pytest.raises(AuthConfigurationError, match=CLOUD_READER_TOKEN_SHA256_VARIABLE) as raised:
        mount_cayu(
            FastAPI(), _app(), access=AuthenticatedAccess(dependency=_operator), dashboard=False
        )
    assert TOKEN not in str(raised.value)
    assert _digest(TOKEN) not in str(raised.value).lower()


def test_the_authenticator_keeps_only_digests() -> None:
    auth = CloudReaderBearerAuth.from_environment(
        {CLOUD_READER_TOKEN_SHA256_VARIABLE: f" {_digest(TOKEN)} , {_digest(ROTATED)}"}
    )

    assert auth is not None
    assert repr(auth) == "CloudReaderBearerAuth(tokens=2)"
    assert CloudReaderBearerAuth.from_environment({}) is None
    with pytest.raises(TypeError):
        CloudReaderBearerAuth(_digest(TOKEN))


@pytest.mark.parametrize(
    "authorization",
    [
        (f"Bearer {TOKEN}",),
        (f"bearer {TOKEN}",),
        (f"Bearer {ROTATED}",),
        (),
        (f"Bearer {TOKEN}", f"Bearer {TOKEN}"),
        (f"Bearer  {TOKEN}",),
        (f"Bearer {TOKEN} extra",),
        (f"Bearer {TOKEN}\u00e9",),
        (f"Basic {TOKEN}",),
        ("Bearer",),
    ],
)
def test_the_reader_accepts_exactly_the_headers_a_fork_controller_accepts(authorization) -> None:
    request = StarletteRequest(
        {
            "type": "http",
            "headers": [(b"authorization", value.encode("utf-8")) for value in authorization],
        }
    )
    reader = CloudReaderBearerAuth([_digest(TOKEN)])
    controller = ForkControllerBearerAuth.from_token_digests({_digest(TOKEN): "svc:compass"})

    accepted = asyncio.run(reader(request))
    assert accepted is (asyncio.run(controller(request)) == "svc:compass")
    assert accepted is (authorization in ((f"Bearer {TOKEN}",), (f"bearer {TOKEN}",)))
