"""The development identity proxy: Cayu Cloud's signed identity, minted in this process."""

from __future__ import annotations

# ruff: noqa: E402
import asyncio
from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi")
jwt = pytest.importorskip("jwt")

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from cayu._development_people import (
    DEVELOPMENT_ROLES,
    DevelopmentPerson,
    parse_development_person,
    parse_identity_proxy_table,
)
from cayu.server import AuthConfigurationError, ProxyIdentityAuth
from cayu.server.development_identity import (
    DEVELOPMENT_IDENTITY_ISSUER,
    DEVELOPMENT_PERSON_COOKIE,
    DEVELOPMENT_PERSON_HEADER,
    DevelopmentIdentityProxy,
    _safe_next,
    registered_development_identity,
)
from cayu.server.proxy_identity import PROXY_IDENTITY_ROLES, PROXY_IDENTITY_TOKEN_TYPE

LOCAL = "http://127.0.0.1:8000"


@pytest.fixture
def proxy() -> Iterator[DevelopmentIdentityProxy]:
    proxy = DevelopmentIdentityProxy()
    yield proxy
    proxy.unregister()


def test_roles_match_proxy_identity() -> None:
    assert set(DEVELOPMENT_ROLES) == set(PROXY_IDENTITY_ROLES)


def test_a_token_carries_the_claims_cloud_sends(proxy: DevelopmentIdentityProxy) -> None:
    token = proxy.token(
        "alice", org="acme", role="admin", name="Alice", email="a@example.com", origin=LOCAL
    )

    header = jwt.get_unverified_header(token)
    claims = jwt.decode(token, options={"verify_signature": False})
    assert header["typ"] == PROXY_IDENTITY_TOKEN_TYPE and header["alg"] == "ES256"
    assert header["kid"] == proxy.jwks()["keys"][0]["kid"]
    assert {key: claims[key] for key in ("iss", "aud", "sub", "org", "role", "origin")} == {
        "iss": DEVELOPMENT_IDENTITY_ISSUER,
        "aud": "cayu-agent:local",
        "sub": "alice",
        "org": "acme",
        "role": "admin",
        "origin": LOCAL,
    }
    assert (claims["name"], claims["email"]) == ("Alice", "a@example.com")
    assert claims["exp"] - claims["iat"] == 60 and claims["nbf"] == claims["iat"]
    assert claims["jti"]
    # Without a name or email, none is sent.
    bare = jwt.decode(proxy.token("bob"), options={"verify_signature": False})
    assert "name" not in bare and "email" not in bare and "origin" not in bare
    assert (bare["org"], bare["role"]) == ("local", "member")


def test_from_environment_trusts_only_a_registered_proxy(proxy: DevelopmentIdentityProxy) -> None:
    environment = proxy.environment()
    # Copied to a deployment, the variables alone fail at startup.
    with pytest.raises(AuthConfigurationError, match="only `cayu serve --identity-proxy` runs"):
        ProxyIdentityAuth.from_environment(environ=environment)

    proxy.register()
    assert registered_development_identity() is proxy
    auth = ProxyIdentityAuth.from_environment(environ=environment)
    claims = asyncio.run(auth.verify_token(proxy.headers("alice")["X-Cayu-Identity"]))
    assert claims["sub"] == "alice"

    # Another proxy's tokens don't verify, and its audience doesn't match this one.
    other = DevelopmentIdentityProxy(audience="cayu-agent:other")
    with pytest.raises(Exception, match="signing key"):
        asyncio.run(auth.verify_token(other.token("mallory")))
    with pytest.raises(AuthConfigurationError, match="don't match"):
        ProxyIdentityAuth.from_environment(environ=other.environment())

    proxy.unregister()
    assert registered_development_identity() is None
    with pytest.raises(AuthConfigurationError):
        ProxyIdentityAuth.from_environment(environ=environment)


def test_the_environment_names_a_custom_header() -> None:
    proxy = DevelopmentIdentityProxy(header="X-Local-Identity")
    proxy.register()
    try:
        environment = proxy.environment()
        assert environment["CAYU_PROXY_IDENTITY_HEADER"] == "X-Local-Identity"
        auth = ProxyIdentityAuth.from_environment(environ=environment)
        token = proxy.headers("alice")["X-Local-Identity"]
        assert asyncio.run(auth.verify_token(token))["sub"] == "alice"
    finally:
        proxy.unregister()


def test_signing_keys_work_without_registering(proxy: DevelopmentIdentityProxy) -> None:
    auth = ProxyIdentityAuth(
        issuer=proxy.issuer, audience=proxy.audience, signing_keys=proxy.signing_keys()
    )
    server = FastAPI()

    @server.get("/me")
    async def me(context=Depends(auth.authenticate)):  # noqa: B008
        return {"subject": context.subject, "role": context.claims["role"]}

    client = TestClient(server)
    assert client.get("/me", headers=proxy.headers("alice", role="developer")).json() == {
        "subject": "alice",
        "role": "developer",
    }
    assert client.get("/me").status_code == 401


def _app(proxy: DevelopmentIdentityProxy, **options) -> TestClient:
    auth = ProxyIdentityAuth(
        issuer=proxy.issuer, audience=proxy.audience, signing_keys=proxy.signing_keys()
    )
    product = auth.product_dependency()
    server = FastAPI()

    @server.get("/app/me")
    async def me(request: Request, principal=Depends(product)):  # noqa: B008
        return {
            "subject": principal.subject_id,
            "tenant": principal.tenant_id,
            "cookie": request.headers.get("cookie"),
            "person_header": request.headers.get(DEVELOPMENT_PERSON_HEADER),
        }

    @server.post("/app/write")
    async def write(principal=Depends(product)):  # noqa: B008
        return {"subject": principal.subject_id}

    @server.get("/app/raw")
    async def raw(request: Request):
        return {"identity": request.headers.get("x-cayu-identity")}

    return TestClient(proxy.middleware(server, **options), base_url=LOCAL)


def test_the_middleware_signs_in_the_default_person_and_strips_its_own_inputs(
    proxy: DevelopmentIdentityProxy,
) -> None:
    alice = DevelopmentPerson("alice", org="acme", name="Alice")
    client = _app(proxy, people=[alice], default="alice")

    client.cookies.set("theme", "dark")
    body = client.get("/app/me").json()
    assert body == {
        "subject": "alice",
        "tenant": "acme",
        "cookie": "theme=dark",
        "person_header": None,
    }

    # The header names someone else for one request, and never reaches the app.
    body = client.get("/app/me", headers={DEVELOPMENT_PERSON_HEADER: "sub=bob&org=globex"}).json()
    assert (body["subject"], body["tenant"], body["person_header"]) == ("bob", "globex", None)


@pytest.mark.parametrize(
    "cookie", ["sub%3Da%26sub%3Db", "sub%3Dalice%26role%3Downer", "who%3Dalice"]
)
def test_a_malformed_cookie_falls_through_to_the_default(
    proxy: DevelopmentIdentityProxy, cookie: str
) -> None:
    client = _app(proxy, default=DevelopmentPerson("alice"))
    client.cookies.set(DEVELOPMENT_PERSON_COOKIE, cookie)

    assert client.get("/app/me").json()["subject"] == "alice"


def test_a_client_sent_identity_is_dropped(proxy: DevelopmentIdentityProxy) -> None:
    client = _app(proxy)
    forged = proxy.token("mallory", role="admin")

    # Nobody is signed in: the forged header is removed rather than trusted.
    assert client.get("/app/raw", headers={"X-Cayu-Identity": forged}).json() == {"identity": None}
    assert client.get("/app/me", headers={"X-Cayu-Identity": forged}).status_code == 401


def test_a_non_loopback_host_gets_no_identity(proxy: DevelopmentIdentityProxy) -> None:
    client = _app(proxy, people=[DevelopmentPerson("alice")], default="alice")
    client.base_url = "http://rebound.example.com"

    assert client.get("/app/raw").json() == {"identity": None}
    assert client.get("/app/me", headers={DEVELOPMENT_PERSON_HEADER: "alice"}).status_code == 401
    # The pages aren't served there either.
    assert client.get("/.cayu/sign-in").status_code == 404


@pytest.mark.parametrize("host", ["localhost:8000", "127.0.0.1", "[::1]:9000", "127.1.2.3"])
def test_loopback_hosts_get_an_identity(proxy: DevelopmentIdentityProxy, host: str) -> None:
    client = _app(proxy, default=DevelopmentPerson("alice"))

    assert client.get("/app/me", headers={"host": host}).json()["subject"] == "alice"


def test_writes_must_come_from_the_requests_own_origin(proxy: DevelopmentIdentityProxy) -> None:
    client = _app(proxy, default=DevelopmentPerson("alice"))

    assert client.post("/app/write", headers={"origin": LOCAL}).status_code == 200
    refused = client.post("/app/write", headers={"origin": "http://evil.example"})
    assert refused.status_code == 403


@pytest.mark.parametrize(
    "value", ["sub=", "who=alice", "sub=a&sub=b", "sub=alice&role=owner", "   "]
)
def test_an_unusable_person_header_is_refused(proxy: DevelopmentIdentityProxy, value: str) -> None:
    response = _app(proxy).get("/app/me", headers={DEVELOPMENT_PERSON_HEADER: value})

    assert response.status_code == 400
    assert DEVELOPMENT_PERSON_HEADER in response.text


def test_the_picker_sets_the_cookie_and_sign_out_clears_it(
    proxy: DevelopmentIdentityProxy,
) -> None:
    people = [
        DevelopmentPerson("alice", org="acme", name="Alice <A>"),
        DevelopmentPerson("bob", org="globex", role="admin"),
    ]
    client = _app(proxy, people=people, default="alice")

    page = client.get("/.cayu/sign-in")
    assert page.status_code == 200
    assert "Alice &lt;A&gt;" in page.text and 'value="bob"' in page.text

    picked = client.post(
        "/.cayu/sign-in",
        data={"person": "bob", "next": "/app/me"},
        headers={"origin": LOCAL},
        follow_redirects=False,
    )
    assert picked.status_code == 303 and picked.headers["location"] == "/app/me"
    assert DEVELOPMENT_PERSON_COOKIE in picked.headers["set-cookie"]
    assert "HttpOnly" in picked.headers["set-cookie"]
    me = client.get("/app/me").json()
    assert (me["subject"], me["tenant"], me["cookie"]) == ("bob", "globex", None)

    # The header overrides the cookie.
    assert (
        client.get("/app/me", headers={DEVELOPMENT_PERSON_HEADER: "alice"}).json()["subject"]
        == "alice"
    )

    # Someone not configured.
    client.post(
        "/.cayu/sign-in",
        data={"subject": "carol", "org": "initech", "role": "developer", "name": "Carol"},
        headers={"origin": LOCAL},
    )
    assert client.get("/app/me").json()["tenant"] == "initech"

    # Signing out signs out, even from the default person, until someone is picked.
    out = client.get("/.cayu/sign-out", follow_redirects=False)
    assert out.status_code == 303 and out.headers["location"] == "/.cayu/sign-in"
    assert client.get("/app/me").status_code == 401
    assert "Nobody is signed in" in client.get("/.cayu/sign-in").text


def test_the_picker_refuses_another_sites_form(proxy: DevelopmentIdentityProxy) -> None:
    client = _app(proxy, people=[DevelopmentPerson("bob", role="admin")])

    refused = client.post(
        "/.cayu/sign-in",
        data={"person": "bob"},
        headers={"origin": "http://evil.example", "sec-fetch-site": "cross-site"},
        follow_redirects=False,
    )
    assert refused.status_code == 403
    assert "set-cookie" not in refused.headers


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/app", "/app"),
        ("//evil.example", "/"),
        ("https://evil.example", "/"),
        ("/\\evil.example", "/"),
        (None, "/"),
    ],
)
def test_sign_in_only_returns_to_this_site(value: str | None, expected: str) -> None:
    assert _safe_next(value) == expected


def test_people_parse_from_headers_and_cookies() -> None:
    alice = DevelopmentPerson("alice", org="acme", name="Alice")
    people = {"alice": alice}

    assert parse_development_person("alice", people) is alice
    assert parse_development_person("dave", people) == DevelopmentPerson("dave")
    assert parse_development_person("sub=alice&role=admin", people) == DevelopmentPerson(
        "alice", org="acme", role="admin", name="Alice"
    )
    assert parse_development_person(alice.encode()) == alice
    spaced = DevelopmentPerson("a b", name="Ann & Co=1")
    assert parse_development_person(spaced.encode()) == spaced


@pytest.mark.parametrize(
    ("table", "message"),
    [
        ({"people": [{"subject": " "}]}, "subject"),
        ({"people": [{"subject": "a", "role": "owner"}]}, "role"),
        ({"people": [{"subject": "a", "team": "x"}]}, "unsupported keys: team"),
        ({"people": [{"subject": "a"}, {"subject": "a"}]}, "listed twice"),
        ({"default": "nobody"}, "not one of its people"),
        ({"people": "alice"}, "array of tables"),
        ({"admins": []}, "unsupported keys: admins"),
    ],
)
def test_the_people_table_is_validated(table: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_identity_proxy_table(table)


def test_the_people_table_reads_people_and_a_default() -> None:
    people, default = parse_identity_proxy_table(
        {
            "default": "alice",
            "people": [
                {
                    "subject": "alice",
                    "org": "acme",
                    "role": "member",
                    "name": "Alice Example",
                    "email": "alice@example.com",
                }
            ],
        }
    )
    assert default == "alice"
    assert people == (
        DevelopmentPerson("alice", "acme", "member", "Alice Example", "alice@example.com"),
    )
