from __future__ import annotations

# ruff: noqa: E402
import asyncio
import importlib
import sys
import time
import uuid
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
jwt = pytest.importorskip("jwt")

import httpx
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import Depends, FastAPI, WebSocket
from fastapi.testclient import TestClient
from jwt.algorithms import ECAlgorithm

from cayu._operator_credentials import OPERATOR_PASSWORD_VARIABLE, OPERATOR_USERNAME_VARIABLE
from cayu.applications import CayuApp
from cayu.server import (
    AuthConfigurationError,
    AuthContext,
    BasicAuth,
    ControlPlaneRoles,
    ProductPrincipal,
    ProxyIdentityAuth,
    ServerConfig,
    create_server,
)
from cayu.server.oidc import OidcTokenError
from cayu.server.proxy_identity import (
    PROXY_IDENTITY_AUDIENCE_VARIABLE,
    PROXY_IDENTITY_HEADER_VARIABLE,
    PROXY_IDENTITY_ISSUER_VARIABLE,
    PROXY_IDENTITY_JWKS_URL_VARIABLE,
    PROXY_IDENTITY_TOKEN_TYPE,
    proxy_identity_configured,
)

ISSUER = "https://sign-in.example.com"
JWKS_URL = "https://sign-in.example.com/jwks"
AUDIENCE = "agent:app-1"
HEADER = "X-Cayu-Identity"
OPERATOR = ("operator", "secret-password")


@pytest.fixture(scope="module")
def signing_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


@pytest.fixture(scope="module")
def other_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


class FakeJwks:
    """The sign-in proxy's published keys, served through httpx.MockTransport."""

    def __init__(self, key: ec.EllipticCurvePrivateKey, kid: str = "key-1") -> None:
        jwk = ECAlgorithm.to_jwk(key.public_key(), as_dict=True)
        self.keys = [{**jwk, "kid": kid, "use": "sig", "alg": "ES256"}]
        self.unreachable = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if str(request.url) == JWKS_URL:
            return httpx.Response(200, json={"keys": self.keys})
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def _token(
    key: ec.EllipticCurvePrivateKey,
    *,
    role: str = "admin",
    kid: str = "key-1",
    typ: str = PROXY_IDENTITY_TOKEN_TYPE,
    lifetime: int = 60,
    remove: tuple[str, ...] = (),
    **claims: Any,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "person-1",
        "org": "org-1",
        "role": role,
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "iat": now,
        "nbf": now,
        "exp": now + lifetime,
        "jti": uuid.uuid4().hex,
        **claims,
    }
    for name in remove:
        payload.pop(name)
    return jwt.encode(payload, key, algorithm="ES256", headers={"kid": kid, "typ": typ})


def _auth(jwks: FakeJwks, **options: Any) -> ProxyIdentityAuth:
    return ProxyIdentityAuth(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url=JWKS_URL,
        http_client=jwks.client(),
        **options,
    )


def _server(auth: ProxyIdentityAuth) -> TestClient:
    return TestClient(create_server(CayuApp(), config=ServerConfig.protected(auth)))


@pytest.fixture
def jwks(signing_key) -> FakeJwks:
    return FakeJwks(signing_key)


@pytest.fixture
def client(jwks) -> TestClient:
    return _server(_auth(jwks, fallback=BasicAuth(username=OPERATOR[0], password=OPERATOR[1])))


def _as(token: str, **headers: str) -> dict[str, str]:
    return {HEADER: token, **headers}


def test_a_valid_identity_reaches_the_control_plane_as_that_person(signing_key, jwks) -> None:
    auth = _auth(jwks)
    seen: list[AuthContext] = []
    app = FastAPI()

    @app.get("/whoami")
    async def whoami(context: AuthContext = Depends(auth.authenticate)) -> dict[str, Any]:  # noqa: B008
        seen.append(context)
        return context.claims

    response = TestClient(app).get("/whoami", headers=_as(_token(signing_key, role="member")))

    assert response.status_code == 200
    context = seen[0]
    assert context.subject == "person-1"
    assert context.tenant == "org-1"
    assert context.claims["auth"] == "proxy-identity"
    assert context.claims["role"] == "member"
    assert context.claims["name"] == "Ada Lovelace"
    assert context.claims["email"] == "ada@example.com"
    assert context.claims["iss"] == ISSUER
    assert context.claims["aud"] == AUDIENCE
    assert "jti" in context.claims
    assert (
        _server(_auth(jwks)).get("/api/sessions", headers=_as(_token(signing_key))).status_code
        == 200
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda key, other: _token(key, aud="agent:other"), id="wrong-audience"),
        pytest.param(lambda key, other: _token(key, iss="https://evil.example"), id="wrong-issuer"),
        pytest.param(lambda key, other: _token(key, typ="JWT"), id="wrong-type"),
        pytest.param(lambda key, other: _token(key, kid="key-9"), id="unknown-key"),
        pytest.param(lambda key, other: _token(other), id="wrong-signature"),
        pytest.param(
            lambda key, other: _token(key, iat=int(time.time()) - 600, exp=int(time.time()) - 300),
            id="expired",
        ),
        pytest.param(lambda key, other: _token(key, lifetime=121), id="lifetime-too-long"),
        pytest.param(lambda key, other: _token(key, remove=("jti",)), id="missing-jti"),
        pytest.param(lambda key, other: _token(key, remove=("iat",)), id="missing-iat"),
        pytest.param(lambda key, other: _token(key, remove=("org",)), id="missing-org"),
        pytest.param(lambda key, other: _token(key, role="owner"), id="unknown-role"),
        pytest.param(lambda key, other: _token(key, email=["a@example.com"]), id="bad-email"),
        pytest.param(
            lambda key, other: _token(key, nbf=int(time.time()) + 300), id="not-yet-valid"
        ),
        pytest.param(lambda key, other: "not-a-jwt", id="malformed"),
        pytest.param(lambda key, other: "", id="empty"),
    ],
)
def test_an_invalid_identity_is_refused_and_never_falls_back(
    client, signing_key, other_key, build
) -> None:
    response = client.get(
        "/api/sessions", headers=_as(build(signing_key, other_key)), auth=OPERATOR
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid sign-in."}
    assert "Basic" not in response.headers["www-authenticate"]


def test_a_token_signed_with_another_algorithm_is_refused(client) -> None:
    token = jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": "p", "org": "o", "role": "admin"},
        "shared-secret-that-is-long-enough-for-hmac",
        algorithm="HS256",
        headers={"kid": "key-1", "typ": PROXY_IDENTITY_TOKEN_TYPE},
    )

    assert client.get("/api/sessions", headers=_as(token)).status_code == 401


def test_two_identity_headers_are_refused(client, signing_key) -> None:
    headers = [(HEADER, _token(signing_key)), (HEADER, _token(signing_key))]

    assert client.get("/api/sessions", headers=headers).status_code == 401


def test_without_the_header_the_fallback_decides(client) -> None:
    assert client.get("/api/sessions", auth=OPERATOR).status_code == 200
    refused = client.get("/api/sessions")
    assert refused.status_code == 401
    assert refused.headers["www-authenticate"].startswith("Basic")
    assert client.get("/api/sessions", auth=("operator", "wrong")).status_code == 401


def test_without_a_fallback_a_request_without_the_header_is_refused(jwks) -> None:
    assert _server(_auth(jwks)).get("/api/sessions", auth=OPERATOR).status_code == 401


def test_the_operator_login_acts_as_admin(client) -> None:
    # Deleting a session is a full-access route; the session doesn't exist.
    assert client.delete("/api/sessions/missing", auth=OPERATOR).status_code == 204


def test_unreachable_keys_answer_503(jwks, signing_key) -> None:
    jwks.unreachable = True

    response = _server(_auth(jwks)).get("/api/sessions", headers=_as(_token(signing_key)))

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/sessions"),
        ("GET", "/api/sessions/missing"),
        ("GET", "/api/pending-actions"),
        ("GET", "/api/inbox"),
        ("GET", "/api/inbox/item-1"),
        ("POST", "/api/inbox/item-1/settle"),
        ("POST", "/api/tool-approvals/resolve"),
        ("POST", "/api/user-input/resolve"),
        ("GET", "/api/contract"),
        ("GET", "/client.js"),
        ("GET", "/cayu/"),
    ],
)
def test_a_member_may_read_and_decide(client, signing_key, method, path) -> None:
    response = client.request(method, path, headers=_as(_token(signing_key, role="member")))

    assert response.status_code not in {401, 403}, response.text


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/run"),
        ("POST", "/api/resume"),
        ("POST", "/api/sessions/missing/interrupt"),
        ("POST", "/api/sessions/missing/messages"),
        ("DELETE", "/api/sessions/missing"),
        ("PATCH", "/api/sessions/missing/labels"),
        ("POST", "/api/tasks/task-1/pause"),
        ("POST", "/api/tool-approvals/recover"),
        ("GET", "/api/system/diagnostics"),
        ("GET", "/api/recovery/startup"),
    ],
)
def test_a_member_is_refused_everything_else(client, signing_key, method, path) -> None:
    response = client.request(method, path, headers=_as(_token(signing_key, role="member")))

    assert response.status_code == 403
    detail = response.json()["detail"]
    # Some routes answer a private error shape of their own.
    assert type(detail) is str or detail["code"] == "role_forbidden"


@pytest.mark.parametrize("role", ["admin", "developer"])
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/run"),
        ("DELETE", "/api/sessions/missing"),
        ("GET", "/api/system/diagnostics"),
        ("POST", "/api/sessions/missing/messages"),
        ("POST", "/api/inbox/item-1/settle"),
    ],
)
def test_admins_and_developers_use_every_route(client, signing_key, role, method, path) -> None:
    response = client.request(method, path, headers=_as(_token(signing_key, role=role)))

    assert response.status_code not in {401, 403}, response.text


def test_the_member_routes_follow_the_api_path_of_a_mounted_control_plane(
    jwks, signing_key
) -> None:
    from cayu.server import AuthenticatedAccess, mount_cayu

    auth = _auth(jwks)
    host = FastAPI()
    mount_cayu(
        host, CayuApp(), path="/cayu", access=AuthenticatedAccess(dependency=auth), dashboard=False
    )
    client = TestClient(host)
    member = _as(_token(signing_key, role="member"))

    assert auth.control_plane_roles.api_paths == {"/cayu/api"}
    assert client.get("/cayu/api/sessions", headers=member).status_code == 200
    assert client.delete("/cayu/api/sessions/missing", headers=member).status_code == 403


def test_a_role_rule_that_knows_no_api_path_refuses_members(jwks, signing_key) -> None:
    auth = _auth(jwks)
    app = FastAPI()

    @app.get("/sessions", dependencies=[Depends(auth)])
    async def sessions() -> list[str]:
        return []

    client = TestClient(app)

    assert (
        client.get("/sessions", headers=_as(_token(signing_key, role="member"))).status_code == 403
    )
    assert (
        client.get("/sessions", headers=_as(_token(signing_key, role="admin"))).status_code == 200
    )


def test_a_cross_origin_browser_write_is_refused(client, signing_key) -> None:
    token = _token(signing_key)

    cross = client.post(
        "/api/inbox/item-1/settle", headers=_as(token, **{"Sec-Fetch-Site": "same-site"})
    )
    same = client.post(
        "/api/inbox/item-1/settle", headers=_as(token, **{"Sec-Fetch-Site": "same-origin"})
    )
    read = client.get("/api/sessions", headers=_as(token, **{"Sec-Fetch-Site": "cross-site"}))

    assert cross.status_code == 403
    assert same.status_code not in {401, 403}
    assert read.status_code == 200


def test_product_dependency_takes_the_person_from_the_identity_and_checks_roles(
    jwks, signing_key
) -> None:
    auth = _auth(jwks, fallback=BasicAuth(username=OPERATOR[0], password=OPERATOR[1]))
    app = FastAPI()

    admins_only = auth.product_dependency(roles=("admin",))

    @app.get("/reports")
    async def reports(principal: ProductPrincipal = Depends(admins_only)) -> dict[str, str]:  # noqa: B008
        return principal.model_dump()

    client = TestClient(app)

    allowed = client.get("/reports", headers=_as(_token(signing_key, role="admin")))
    assert allowed.status_code == 200
    assert allowed.json() == {"tenant_id": "org-1", "subject_id": "person-1"}
    assert (
        client.get("/reports", headers=_as(_token(signing_key, role="member"))).status_code == 403
    )
    # The operator login is for the control plane, not for product people.
    assert client.get("/reports", auth=OPERATOR).status_code == 401
    with pytest.raises(ValueError, match="roles"):
        auth.product_dependency(roles=("owner",))


def test_verify_token_names_the_reason_without_the_token(jwks, signing_key) -> None:
    auth = _auth(jwks)
    token = _token(signing_key, lifetime=500)

    with pytest.raises(OidcTokenError) as raised:
        asyncio.run(auth.verify_token(token))

    assert raised.value.reason == "token lifetime is too long"
    assert token not in str(raised.value)


def test_configuration_is_validated() -> None:
    jwks = FakeJwks(ec.generate_private_key(ec.SECP256R1()))
    with pytest.raises(ValueError, match="header"):
        _auth(jwks, header="Authorization")
    with pytest.raises(ValueError, match="header"):
        _auth(jwks, header="bad header")
    with pytest.raises(ValueError, match="jwks_url"):
        ProxyIdentityAuth(issuer=ISSUER, audience=AUDIENCE)
    with pytest.raises(ValueError, match="https"):
        ProxyIdentityAuth(issuer=ISSUER, audience=AUDIENCE, jwks_url="http://keys.example/jwks")
    with pytest.raises(ValueError, match="share a role"):
        ControlPlaneRoles(full_access=("admin",), read_and_decide=("admin",))


def _environment(**values: str) -> dict[str, str]:
    return {
        OPERATOR_USERNAME_VARIABLE: OPERATOR[0],
        OPERATOR_PASSWORD_VARIABLE: OPERATOR[1],
        **values,
    }


_PROXY_ENVIRONMENT = {
    PROXY_IDENTITY_ISSUER_VARIABLE: ISSUER,
    PROXY_IDENTITY_AUDIENCE_VARIABLE: AUDIENCE,
    PROXY_IDENTITY_JWKS_URL_VARIABLE: JWKS_URL,
}


def _import_environment_auth(monkeypatch, environment: dict[str, str]):
    for name in (
        OPERATOR_USERNAME_VARIABLE,
        OPERATOR_PASSWORD_VARIABLE,
        *_PROXY_ENVIRONMENT,
        PROXY_IDENTITY_HEADER_VARIABLE,
    ):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    sys.modules.pop("cayu.server.environment_auth", None)
    try:
        return importlib.import_module("cayu.server.environment_auth")
    finally:
        sys.modules.pop("cayu.server.environment_auth", None)


def test_environment_auth_with_every_variable_trusts_the_proxy_and_keeps_the_operator(
    monkeypatch,
) -> None:
    module = _import_environment_auth(monkeypatch, _environment(**_PROXY_ENVIRONMENT))
    auth = module.OPERATOR_BASIC_AUTH

    assert type(auth) is ProxyIdentityAuth
    assert auth.issuer == ISSUER
    assert auth.audience == (AUDIENCE,)
    assert auth.signing_keys.jwks_url == JWKS_URL
    assert auth.header == HEADER
    assert type(auth.fallback) is BasicAuth
    assert auth.fallback.username == OPERATOR[0]
    assert (
        TestClient(create_server(CayuApp(), config=ServerConfig.protected(auth)))
        .get("/api/sessions", auth=OPERATOR)
        .status_code
        == 200
    )


def test_environment_auth_reads_a_custom_header(monkeypatch) -> None:
    environment = _environment(**_PROXY_ENVIRONMENT, **{PROXY_IDENTITY_HEADER_VARIABLE: "X-Person"})

    assert (
        _import_environment_auth(monkeypatch, environment).OPERATOR_BASIC_AUTH.header == "X-Person"
    )


def test_environment_auth_without_proxy_variables_is_the_operator_login(monkeypatch) -> None:
    assert (
        type(_import_environment_auth(monkeypatch, _environment()).OPERATOR_BASIC_AUTH) is BasicAuth
    )


@pytest.mark.parametrize(
    "missing",
    [
        (PROXY_IDENTITY_ISSUER_VARIABLE,),
        (PROXY_IDENTITY_AUDIENCE_VARIABLE, PROXY_IDENTITY_JWKS_URL_VARIABLE),
    ],
)
def test_environment_auth_refuses_a_partial_set(monkeypatch, missing) -> None:
    partial = {name: value for name, value in _PROXY_ENVIRONMENT.items() if name not in missing}

    with pytest.raises(AuthConfigurationError, match="partly configured") as raised:
        _import_environment_auth(monkeypatch, _environment(**partial))

    assert all(name in str(raised.value) for name in missing)
    assert ISSUER not in str(raised.value)


def test_environment_auth_refuses_a_header_name_alone(monkeypatch) -> None:
    with pytest.raises(AuthConfigurationError, match=PROXY_IDENTITY_HEADER_VARIABLE):
        _import_environment_auth(
            monkeypatch, _environment(**{PROXY_IDENTITY_HEADER_VARIABLE: "X-Person"})
        )


def test_environment_auth_names_an_unusable_variable_without_its_value(monkeypatch) -> None:
    environment = _environment(
        **{**_PROXY_ENVIRONMENT, PROXY_IDENTITY_JWKS_URL_VARIABLE: "http://keys.internal/jwks"}
    )

    with pytest.raises(AuthConfigurationError, match=PROXY_IDENTITY_JWKS_URL_VARIABLE) as raised:
        _import_environment_auth(monkeypatch, environment)

    assert "keys.internal" not in str(raised.value)


def test_proxy_identity_configured_reads_only_the_required_set() -> None:
    assert proxy_identity_configured({}) is False
    assert proxy_identity_configured(_PROXY_ENVIRONMENT) is True
    assert proxy_identity_configured({name: " " for name in _PROXY_ENVIRONMENT}) is False


@pytest.mark.parametrize(
    ("origin", "refused"),
    [
        ("https://close.agents.example", False),
        ("https://CLOSE.agents.example:443", False),
        ("http://testserver", True),
        ("https://other.agents.example", True),
        ("http://close.agents.example", True),
    ],
)
def test_a_signed_origin_claim_is_the_origin_browsers_must_match(
    client, signing_key, origin, refused
) -> None:
    # Behind the proxy the Agent sees http://testserver; the browser sees the proxy's origin.
    token = _token(signing_key, origin="https://close.agents.example")

    response = client.post("/api/sessions/unknown/interrupt", headers=_as(token, Origin=origin))

    assert (response.status_code == 403) is refused
    if not refused:
        assert response.status_code != 401


@pytest.mark.parametrize(
    "origin", ["close.agents.example", "https://close.agents.example/path", "ftp://x", 7, ""]
)
def test_an_invalid_origin_claim_refuses_the_token(client, signing_key, origin) -> None:
    response = client.get("/api/sessions", headers=_as(_token(signing_key, origin=origin)))

    assert response.status_code == 401


def test_the_operator_login_alone_still_requires_its_credentials(monkeypatch) -> None:
    with pytest.raises(AuthConfigurationError, match=OPERATOR_PASSWORD_VARIABLE):
        _import_environment_auth(
            monkeypatch,
            {OPERATOR_USERNAME_VARIABLE: OPERATOR[0], **_PROXY_ENVIRONMENT},
        )


@pytest.mark.parametrize(
    ("headers", "refused"),
    [
        ({"Origin": "http://evil.testserver"}, True),
        ({"Origin": "https://testserver"}, True),
        ({"Origin": "http://testserver:81"}, True),
        ({"Origin": "null"}, True),
        ({"Origin": ""}, True),
        ({"Origin": "not-a-url"}, True),
        ({"Origin": "http://testserver:bad"}, True),
        ({"Origin": "http://testserver/path"}, True),
        ({"Origin": "http://user@testserver"}, True),
        ({"Origin": "http://testserver", "Sec-Fetch-Site": "cross-site"}, True),
        ({"Origin": "http://evil.testserver", "Sec-Fetch-Site": "same-origin"}, True),
        ({"Referer": "http://evil.testserver/page"}, True),
        ({"Referer": "null"}, True),
        ({"Referer": ""}, True),
        ({"Origin": "null", "Referer": "http://testserver/page"}, True),
        ({"Origin": "http://testserver"}, False),
        ({"Origin": "http://TESTSERVER:80"}, False),
        ({"Referer": "http://testserver/page?query=1"}, False),
        ({"Origin": "http://testserver", "Referer": "http://elsewhere/page"}, False),
        ({}, False),
    ],
)
def test_bodyless_write_checks_browser_origin_without_fetch_metadata(
    client, signing_key, headers, refused
) -> None:
    response = client.post(
        "/api/sessions/unknown/interrupt", headers=_as(_token(signing_key), **headers)
    )
    assert (response.status_code == 403) is refused
    if refused:
        assert response.json() == {"detail": "Cross-origin request refused."}
    else:
        assert response.status_code not in {401, 403}


@pytest.mark.parametrize("header", ["Origin", "Referer", "Sec-Fetch-Site"])
def test_duplicate_browser_origin_headers_are_refused(client, signing_key, header) -> None:
    value = "same-origin" if header == "Sec-Fetch-Site" else "http://testserver"
    response = client.post(
        "/api/sessions/unknown/interrupt",
        headers=[(HEADER, _token(signing_key)), (header, value), (header, value)],
    )
    assert response.status_code == 403


def test_shared_role_rule_matches_the_longest_api_prefix() -> None:
    from starlette.requests import Request
    from starlette.routing import Route

    roles = ControlPlaneRoles()
    roles.add_api_path("/api")
    roles.add_api_path("/api/control")
    context = AuthContext(subject="person", claims={"role": "member"})
    for path in ("/api/sessions", "/api/control/sessions"):
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": path,
                "headers": [],
                "route": Route(path, lambda _: None),
            }
        )
        assert roles.permits(request, context)


@pytest.mark.parametrize(
    ("scheme", "origin", "refused"),
    [
        ("https", "https://agent.example:443", False),
        ("https", "http://agent.example", True),
        ("https", "https://agent.example:444", True),
        ("wss", "https://agent.example", False),
        ("wss", "https://evil.example", True),
        ("ws", "http://agent.example:80", False),
    ],
)
def test_origin_comparison_uses_effective_ports_and_websocket_scheme(scheme, origin, refused):
    from starlette.requests import HTTPConnection

    from cayu.server.proxy_identity import _cross_origin_write

    request = HTTPConnection(
        {
            "type": "websocket" if scheme in {"ws", "wss"} else "http",
            "method": "POST",
            "scheme": scheme,
            "path": "/write",
            "query_string": b"",
            "headers": [(b"host", b"agent.example"), (b"origin", origin.encode())],
        }
    )
    assert _cross_origin_write(request) is refused


@pytest.mark.parametrize("path", ["with%20space", "with%5Cslash", "with%09tab"])
@pytest.mark.parametrize("header", ["Origin", "Referer"])
def test_same_origin_write_accepts_encoded_path_characters(jwks, signing_key, path, header):
    auth = _auth(jwks)
    app = FastAPI()

    @app.post("/write/{path:path}")
    async def write(context: AuthContext = Depends(auth.authenticate)):  # noqa: B008
        return {"subject": context.subject}

    response = TestClient(app).post(
        f"/write/{path}", headers=_as(_token(signing_key), **{header: "http://testserver"})
    )
    assert response.status_code == 200
    assert response.json() == {"subject": "person-1"}


@pytest.mark.parametrize("prefix", ["/api/agent", "/agent"])
@pytest.mark.parametrize("method", ["get", "head"])
def test_member_dashboard_under_outer_mount_uses_server_local_path(
    tmp_path, jwks, signing_key, prefix, method
):
    from cayu.server import DashboardConfig

    (tmp_path / "index.html").write_text("<!doctype html><title>dashboard</title>")
    (tmp_path / "asset.js").write_text("window.loaded = true;")
    auth = _auth(jwks)
    server = create_server(
        CayuApp(),
        config=ServerConfig.protected(auth, dashboard=DashboardConfig(directory=tmp_path)),
    )
    parent = FastAPI()
    parent.mount(prefix, server)
    client = TestClient(parent)
    member = _as(_token(signing_key, role="member"))
    for path in ("/cayu/", "/cayu/asset.js"):
        assert getattr(client, method)(prefix + path, headers=member).status_code == 200
    assert client.get(prefix + "/api/sessions", headers=member).status_code == 200
    assert client.delete(prefix + "/api/sessions/missing", headers=member).status_code == 403


def test_dashboard_mounted_inside_api_still_requires_member_route_allowlist(tmp_path):
    from starlette.requests import Request

    from cayu.server.static import DashboardStaticFiles

    roles = ControlPlaneRoles()
    roles.add_api_path("/api")
    context = AuthContext(subject="person", claims={"role": "member"})
    endpoint = DashboardStaticFiles(directory=str(tmp_path), base_path="/api/private")
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/agent/api/private/",
            "root_path": "/api/agent/api/private",
            "endpoint": endpoint,
        }
    )
    assert not roles.permits(request, context)


@pytest.mark.parametrize("kind", ["approval", "user_input"])
def test_refreshed_proxy_tokens_keep_decision_identity(jwks, kind):
    from cayu.approvals.tools import ToolApprovalRequest
    from cayu.approvals.user_input import UserInputResponse, user_input_resolution_request_digest
    from cayu.runtime._approval_support import approval_resolution_request_digest
    from cayu.server.authorization_grants import _actor
    from cayu.server.routes import _request_resolution_actor

    auth = _auth(jwks)
    claims = {
        "sub": "person",
        "org": "tenant",
        "role": "member",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": 100,
        "exp": 220,
        "jti": "first",
    }
    original = auth.auth_context(claims)
    refreshed = auth.auth_context({**claims, "iat": 160, "exp": 280, "jti": "second"})
    assert original.claims["jti"] != refreshed.claims["jti"]

    def digest(context):
        actor = _request_resolution_actor(context, None)
        assert actor == _actor(context)
        if kind == "approval":
            return approval_resolution_request_digest(
                ToolApprovalRequest(
                    session_id="session",
                    approval_id="approval",
                    tool_round_id="round",
                    tool_call_id="call",
                    decision="approve",
                    resolved_by=actor,
                )
            )
        return user_input_resolution_request_digest(
            UserInputResponse(
                session_id="session",
                input_id="input",
                answer="yes",
                resolved_by=actor,
            )
        )

    assert digest(original) == digest(refreshed)
    for name, value in (("sub", "other"), ("org", "other"), ("role", "admin")):
        assert digest(original) != digest(auth.auth_context({**claims, name: value}))


@pytest.mark.parametrize("entrypoint", ["authenticate", "control_plane", "product"])
def test_proxy_dependencies_authenticate_websockets(jwks, signing_key, entrypoint):
    auth = _auth(jwks)
    dependency = {
        "authenticate": auth.authenticate,
        "control_plane": auth,
        "product": auth.product_dependency(),
    }[entrypoint]
    server = FastAPI()

    @server.websocket("/ws")
    async def socket(websocket: WebSocket, context=Depends(dependency)):  # noqa: B008
        await websocket.accept()
        subject = context.subject_id if isinstance(context, ProductPrincipal) else context.subject
        await websocket.send_text(subject)
        await websocket.close()

    with TestClient(server) as client:
        token = _token(signing_key, role="developer", origin="https://public.example")
        with client.websocket_connect(
            "/ws", headers=_as(token, Origin="https://public.example")
        ) as socket:
            assert socket.receive_text() == "person-1"


@pytest.mark.parametrize("mounted", [False, True])
def test_member_cannot_read_unrelated_host_routes(jwks, signing_key, mounted):
    from cayu.server import AuthenticatedAccess, mount_cayu

    auth = _auth(jwks)
    host = FastAPI()
    protected = FastAPI() if mounted else host

    @protected.get("/operator/secrets", dependencies=[Depends(auth)])
    def secrets():
        return {"secret": "operator only"}

    if mounted:
        host.mount("/internal", protected)
    mount_cayu(
        host, CayuApp(), path="/agent", access=AuthenticatedAccess(dependency=auth), dashboard=False
    )
    path = ("/internal" if mounted else "") + "/operator/secrets"
    response = TestClient(host).get(path, headers=_as(_token(signing_key, role="member")))
    assert response.status_code == 403


@pytest.mark.parametrize("role", ["member", "admin", "developer"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arguments", {}),
        ("max_steps", 5),
        ("limits", {}),
        ("budget_limits", []),
        ("retry_policy", {}),
        ("structured_output", {}),
        ("thinking", {}),
    ],
)
def test_member_decisions_cannot_override_execution(jwks, signing_key, role, field, value):
    client = _server(_auth(jwks))
    headers = _as(_token(signing_key, role=role))
    for path in ("/api/tool-approvals/resolve", "/api/user-input/resolve"):
        payload = {"session_id": "missing"}
        if "tool-approvals" in path:
            payload.update(
                approval_id="missing",
                tool_round_id="round",
                tool_call_id="call",
                decision="approve",
            )
        else:
            payload.update(input_id="missing", answer="yes")
        response = client.post(path, json={**payload, field: value}, headers=headers)
        assert (response.status_code == 403) is (role == "member"), response.text
        plain = client.post(path, json={**payload, field: None}, headers=headers)
        assert plain.status_code not in {401, 403, 422}, plain.text


def test_member_denies_unknown_mounted_surface():
    from starlette.requests import Request

    policy = ControlPlaneRoles()
    policy.add_api_path("/api")
    request = Request({"type": "http", "method": "GET", "path": "/operator/secrets", "headers": []})
    assert not policy.permits(request, AuthContext(subject="member", claims={"role": "member"}))


@pytest.mark.parametrize("role", ["member", "admin", "developer"])
@pytest.mark.parametrize("field", ["arguments", "approval_revision"])
def test_member_inbox_decisions_cannot_replace_proposal(jwks, signing_key, role, field):
    client = _server(_auth(jwks))
    headers = _as(_token(signing_key, role=role))
    from cayu.server._inbox_routes import _session_item_id, _SessionLocator

    item_id = _session_item_id(_SessionLocator("approval", "missing", 1, "attention"))
    path = f"/api/inbox/{item_id}/settle"
    response = client.post(path, json={"decision": "approve", field: {}}, headers=headers)
    assert (response.status_code == 403) is (role == "member"), response.text
    plain = client.post(path, json={"decision": "approve"}, headers=headers)
    assert plain.status_code not in {401, 403, 422}, plain.text


def test_member_cannot_read_shadowing_browser_client(jwks, signing_key):
    from cayu.server import AuthenticatedAccess, mount_cayu

    auth = _auth(jwks)
    host = FastAPI()

    @host.get("/agent/client.js", dependencies=[Depends(auth)])
    def secrets():
        return {"secret": "operator only"}

    mount_cayu(
        host, CayuApp(), path="/agent", access=AuthenticatedAccess(dependency=auth), dashboard=False
    )
    response = TestClient(host).get(
        "/agent/client.js", headers=_as(_token(signing_key, role="member"))
    )
    assert response.status_code == 403


@pytest.mark.parametrize("role", ["member", "admin", "developer"])
def test_member_cannot_declare_approval_revision_in_metadata(jwks, signing_key, role):
    client = _server(_auth(jwks))
    headers = _as(_token(signing_key, role=role))
    payload = {
        "session_id": "missing",
        "approval_id": "missing",
        "tool_round_id": "round",
        "tool_call_id": "call",
        "decision": "approve",
        "metadata": {
            "cayu:approval_revision": {"proposed_digest": "original", "approved_digest": "edited"}
        },
    }
    response = client.post("/api/tool-approvals/resolve", json=payload, headers=headers)
    assert response.status_code == (403 if role == "member" else 404), response.text
    payload["metadata"] = {"comment": "approved"}
    assert (
        client.post("/api/tool-approvals/resolve", json=payload, headers=headers).status_code == 404
    )
