from __future__ import annotations

# ruff: noqa: E402
import asyncio
import base64
import hashlib
import hmac
import json
import logging
import sys
import time
import tracemalloc
import zlib
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
jwt = pytest.importorskip("jwt")

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from cayu.applications import CayuApp
from cayu.runtime.checks import check_public_service_deployment
from cayu.server import (
    AuthConfigurationError,
    AuthContext,
    AuthenticatedAccess,
    AuthenticatedProductAccess,
    OidcBearerAuth,
    OidcSigningKeys,
    OidcTokenError,
    ProductPrincipal,
    ServerConfig,
    create_server,
)
from cayu.server.auth import resolve_auth_context

ISSUER = "https://idp.example.com/"
DISCOVERY_URL = "https://idp.example.com/.well-known/openid-configuration"
JWKS_URL = "https://idp.example.com/keys/jwks.json"
AUDIENCE = "api://cayu"


@pytest.fixture(scope="module")
def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def other_rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def ec_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


class FakeIssuer:
    """An OIDC issuer served through httpx.MockTransport."""

    def __init__(self, issuer: str = ISSUER, *, jwks_url: str = JWKS_URL) -> None:
        self.issuer = issuer
        self.jwks_url = jwks_url
        self.discovery_issuer = issuer
        self.discovery_jwks_uri = jwks_url
        self.keys: list[dict[str, Any]] = []
        self.cache_control: str | None = "max-age=300"
        self.failing = False
        self.unreachable = False
        self.jwks_body: bytes | None = None
        self.requests: list[str] = []
        self.accept_encodings: list[str | None] = []

    def add_rsa(self, kid: str, key: rsa.RSAPrivateKey, **extra: Any) -> None:
        jwk = RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
        self.keys.append({**jwk, "kid": kid, "use": "sig", "alg": "RS256", **extra})

    def add_ec(self, kid: str, key: ec.EllipticCurvePrivateKey) -> None:
        jwk = ECAlgorithm.to_jwk(key.public_key(), as_dict=True)
        self.keys.append({**jwk, "kid": kid, "use": "sig"})

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        self.accept_encodings.append(request.headers.get("accept-encoding"))
        if self.unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if self.failing:
            return httpx.Response(500)
        if url == self.issuer.rstrip("/") + "/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={"issuer": self.discovery_issuer, "jwks_uri": self.discovery_jwks_uri},
            )
        if url == self.jwks_url:
            headers = {} if self.cache_control is None else {"Cache-Control": self.cache_control}
            if self.jwks_body is not None:
                return httpx.Response(200, content=self.jwks_body, headers=headers)
            return httpx.Response(200, json={"keys": list(self.keys)}, headers=headers)
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def jwks_requests(self) -> int:
        return sum(url == self.jwks_url for url in self.requests)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _token(
    key: Any,
    claims: dict[str, Any] | None = None,
    *,
    kid: str = "key-1",
    algorithm: str = "RS256",
    headers: dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    payload: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-1",
        "iat": now,
        "exp": now + 300,
        **(claims or {}),
    }
    payload = {name: value for name, value in payload.items() if value is not None}
    return jwt.encode(payload, key, algorithm=algorithm, headers={"kid": kid, **(headers or {})})


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unsigned_token(header: dict[str, Any], claims: dict[str, Any], signature: bytes) -> str:
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(claims).encode())}"
    return f"{signing_input}.{_b64(signature)}"


def _auth(issuer: FakeIssuer, clock: FakeClock | None = None, **options: Any) -> OidcBearerAuth:
    auth = OidcBearerAuth(
        issuer=issuer.issuer, audience=AUDIENCE, http_client=issuer.client(), **options
    )
    if clock is not None:
        auth.signing_keys._monotonic = clock
    return auth


def _client(auth: Any) -> TestClient:
    app = FastAPI()

    @app.get("/whoami")
    async def whoami(context: AuthContext = Depends(auth)) -> dict[str, Any]:  # noqa: B008
        return context.model_dump()

    return TestClient(app)


def _verify(auth: OidcBearerAuth, token: str) -> dict[str, Any]:
    return asyncio.run(auth.verify_token(token))


def _rejection(auth: OidcBearerAuth, token: str) -> OidcTokenError:
    with pytest.raises(OidcTokenError) as raised:
        _verify(auth, token)
    return raised.value


def test_valid_rs256_token_returns_auth_context(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    auth = _auth(issuer, tenant_claim="org_id")
    token = _token(rsa_key, {"org_id": "org-7", "scope": "read write", "email": "a@b.example"})

    response = _client(auth).get("/whoami", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 200
    body = response.json()
    assert body["subject"] == "user-1"
    assert body["tenant"] == "org-7"
    assert body["claims"]["iss"] == ISSUER
    assert body["claims"]["aud"] == AUDIENCE
    assert body["claims"]["scope"] == "read write"
    # Only the standard claims are copied by default.
    assert "email" not in body["claims"]
    assert "org_id" not in body["claims"]
    assert issuer.requests == [DISCOVERY_URL, JWKS_URL]


def test_valid_es256_token_and_audience_list(ec_key: ec.EllipticCurvePrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_ec("ec-1", ec_key)
    auth = OidcBearerAuth(
        issuer=ISSUER,
        audience=["other", AUDIENCE],
        jwks_url=JWKS_URL,
        http_client=issuer.client(),
    )

    claims = _verify(
        auth,
        _token(ec_key, {"aud": [AUDIENCE, "unrelated"]}, kid="ec-1", algorithm="ES256"),
    )

    assert claims["sub"] == "user-1"
    # An explicit jwks_url skips discovery.
    assert issuer.requests == [JWKS_URL]


@pytest.mark.parametrize(
    ("claims", "reason"),
    [
        ({"iss": "https://evil.example.com/"}, "issuer"),
        ({"iss": ISSUER.rstrip("/")}, "issuer"),
        ({"iss": None}, "required claim"),
        ({"aud": "api://other"}, "audience"),
        ({"aud": None}, "required claim"),
        ({"exp": -600}, "expired"),
        ({"exp": None}, "required claim"),
        ({"exp": "9999999999"}, "not a number"),
        ({"nbf": 600}, "not yet valid"),
        ({"iat": 600}, "not yet valid"),
        ({"sub": None}, "subject"),
        ({"sub": ""}, "subject"),
        ({"sub": 42}, "invalid"),
    ],
)
def test_claim_validation_rejects_bad_tokens(
    rsa_key: rsa.RSAPrivateKey, claims: dict[str, Any], reason: str
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)

    # Relative claim times must be resolved at execution, after a long shard's
    # collection-to-test delay, so future claims cannot age into validity.
    now = int(time.time())
    claims = {
        name: now + value if name in {"exp", "nbf", "iat"} and type(value) is int else value
        for name, value in claims.items()
    }
    error = _rejection(_auth(issuer), _token(rsa_key, claims))

    assert error.status_code == 401
    assert error.error == "invalid_token"
    assert reason in error.reason


def test_leeway_admits_small_clock_skew(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    now = int(time.time())

    claims = _verify(
        _auth(issuer, leeway_seconds=120),
        _token(rsa_key, {"exp": now - 30, "nbf": now + 30, "iat": now + 30}),
    )

    assert claims["sub"] == "user-1"
    assert (
        "expired"
        in _rejection(_auth(issuer, leeway_seconds=0), _token(rsa_key, {"exp": now - 30})).reason
    )


def test_bad_signature_is_rejected(
    rsa_key: rsa.RSAPrivateKey, other_rsa_key: rsa.RSAPrivateKey
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    forged = _token(other_rsa_key)
    header, payload, _ = _token(rsa_key, {"sub": "attacker"}).split(".")
    original_signature = _token(rsa_key).split(".")[2]

    assert "signature" in _rejection(_auth(issuer), forged).reason
    assert (
        "signature" in _rejection(_auth(issuer), f"{header}.{payload}.{original_signature}").reason
    )


def test_alg_none_is_rejected_before_any_key_fetch(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    auth = _auth(issuer)
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": "attacker", "exp": now + 300}

    with_signature = _unsigned_token({"alg": "none", "kid": "key-1"}, claims, b"x")
    without_signature = with_signature.rsplit(".", 1)[0] + "."

    assert "algorithm" in _rejection(auth, with_signature).reason
    assert "malformed" in _rejection(auth, without_signature).reason
    assert issuer.requests == []


def test_hs256_key_confusion_is_rejected(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    public_pem = rsa_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    now = int(time.time())
    header = {"alg": "HS256", "kid": "key-1", "typ": "JWT"}
    claims = {"iss": ISSUER, "aud": AUDIENCE, "sub": "attacker", "exp": now + 300}
    signing_input = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(claims).encode())}"
    signature = hmac.new(public_pem, signing_input.encode(), hashlib.sha256).digest()

    error = _rejection(_auth(issuer), f"{signing_input}.{_b64(signature)}")

    assert "algorithm" in error.reason
    for algorithms in (["HS256"], ["RS256", "none"], ["none"], []):
        with pytest.raises(ValueError, match="algorithm"):
            OidcBearerAuth(issuer=ISSUER, audience=AUDIENCE, algorithms=algorithms)


def test_key_must_match_algorithm_family(
    rsa_key: rsa.RSAPrivateKey, ec_key: ec.EllipticCurvePrivateKey
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("shared", rsa_key)
    auth = _auth(issuer, algorithms=["RS256", "ES256"])

    # An ES256 token naming an RSA key id is not verified with that key.
    error = _rejection(auth, _token(ec_key, kid="shared", algorithm="ES256"))

    assert "does not match" in error.reason
    assert (
        "algorithm"
        in _rejection(_auth(issuer, algorithms=["ES256"]), _token(rsa_key, kid="shared")).reason
    )


def test_unknown_kid_triggers_one_rate_limited_refetch(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    clock = FakeClock()
    auth = _auth(issuer, clock)
    _verify(auth, _token(rsa_key))
    assert issuer.jwks_requests() == 1

    for attempt in range(5):
        clock.now += 1
        error = _rejection(auth, _token(rsa_key, kid=f"unknown-{attempt}"))
        assert error.reason == "unknown signing key"
    # Only the first unknown kid refetched; the rest fell inside the interval.
    assert issuer.jwks_requests() == 2

    clock.now += auth.signing_keys.refetch_interval_seconds
    _rejection(auth, _token(rsa_key, kid="unknown-later"))
    assert issuer.jwks_requests() == 3
    # Discovery is fetched once and the jwks_uri reused.
    assert issuer.requests.count(DISCOVERY_URL) == 1


def test_key_rotation_is_picked_up_on_unknown_kid(
    rsa_key: rsa.RSAPrivateKey, other_rsa_key: rsa.RSAPrivateKey
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    clock = FakeClock()
    auth = _auth(issuer, clock)
    _verify(auth, _token(rsa_key))

    issuer.keys = []
    issuer.add_rsa("key-2", other_rsa_key)
    clock.now += 1

    assert _verify(auth, _token(other_rsa_key, kid="key-2"))["sub"] == "user-1"
    assert issuer.jwks_requests() == 2
    # The retired key is gone from the refreshed set.
    assert _rejection(auth, _token(rsa_key)).reason == "unknown signing key"


def test_concurrent_requests_share_one_fetch(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    auth = _auth(issuer)
    token = _token(rsa_key)

    async def verify_many() -> list[dict[str, Any]]:
        return await asyncio.gather(*(auth.verify_token(token) for _ in range(20)))

    results = asyncio.run(verify_many())

    assert len(results) == 20
    assert issuer.requests == [DISCOVERY_URL, JWKS_URL]


def test_cache_control_bounds_the_key_cache(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    issuer.cache_control = "public, max-age=120"
    clock = FakeClock()
    auth = _auth(issuer, clock)
    token = _token(rsa_key)
    _verify(auth, token)

    clock.now += 100
    _verify(auth, token)
    assert issuer.jwks_requests() == 1
    clock.now += 30
    _verify(auth, token)
    assert issuer.jwks_requests() == 2

    # A tiny max-age is raised to the 60-second floor.
    issuer.cache_control = "max-age=1"
    clock.now += 200
    _verify(auth, token)
    assert issuer.jwks_requests() == 3
    clock.now += 30
    _verify(auth, token)
    assert issuer.jwks_requests() == 3


def test_issuer_outage_keeps_last_keys_and_backs_off(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    clock = FakeClock()
    auth = _auth(issuer, clock)
    token = _token(rsa_key)
    _verify(auth, token)

    issuer.failing = True
    clock.now += 400
    for _ in range(10):
        assert _verify(auth, token)["sub"] == "user-1"
    assert issuer.jwks_requests() == 2

    clock.now += 6
    _verify(auth, token)
    assert issuer.jwks_requests() == 3


def test_empty_jwks_withdraws_cached_keys(
    rsa_key: rsa.RSAPrivateKey, caplog: pytest.LogCaptureFixture
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    clock = FakeClock()
    auth = _auth(issuer, clock)
    token = _token(rsa_key)
    _verify(auth, token)

    issuer.keys = []
    clock.now += 301
    with caplog.at_level(logging.WARNING, logger="cayu.server.oidc"):
        error = _rejection(auth, token)

    # The issuer answered with an empty key set, so the cached key is gone: 401, not
    # stale-key acceptance and not 503.
    assert issuer.jwks_requests() == 2
    assert error.status_code == 401
    assert error.error == "invalid_token"
    assert error.reason == "unknown signing key"
    assert "no usable signing keys" in caplog.text
    response = _client(auth).get("/whoami", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert 'error="invalid_token"' in response.headers["www-authenticate"]


def test_expiry_refresh_drops_a_removed_key(
    rsa_key: rsa.RSAPrivateKey, other_rsa_key: rsa.RSAPrivateKey
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    issuer.add_rsa("key-2", other_rsa_key)
    clock = FakeClock()
    auth = _auth(issuer, clock)
    _verify(auth, _token(rsa_key))

    issuer.keys = [key for key in issuer.keys if key["kid"] != "key-1"]
    clock.now += 301

    assert _rejection(auth, _token(rsa_key)).reason == "unknown signing key"
    assert _verify(auth, _token(other_rsa_key, kid="key-2"))["sub"] == "user-1"
    assert issuer.jwks_requests() == 2


@pytest.mark.parametrize("after_rotation", ["empty", "filtered"])
def test_unknown_kid_refetch_replaces_keys_even_when_none_are_usable(
    rsa_key: rsa.RSAPrivateKey, after_rotation: str
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    clock = FakeClock()
    auth = _auth(issuer, clock)
    _verify(auth, _token(rsa_key))

    issuer.keys = []
    if after_rotation == "filtered":
        issuer.add_rsa("enc", rsa_key, use="enc")
        issuer.keys.append({"kty": "oct", "kid": "hmac", "k": _b64(b"secret")})
    clock.now += 1
    assert _rejection(auth, _token(rsa_key, kid="key-2")).reason == "unknown signing key"
    assert issuer.jwks_requests() == 2

    # The refetch was authoritative, so the previously cached key-1 is gone too.
    error = _rejection(auth, _token(rsa_key))
    assert (error.status_code, error.reason) == (401, "unknown signing key")


def test_jwks_with_only_unusable_keys_rejects_with_401(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key, use="enc")
    issuer.keys.append({"kty": "oct", "kid": "hmac", "k": _b64(b"secret")})

    error = _rejection(_auth(issuer), _token(rsa_key))

    assert (error.status_code, error.error) == (401, "invalid_token")


@pytest.mark.parametrize(
    "failure", ["unreachable", "http-500", "not-json", "no-keys-array", "too-many-keys"]
)
def test_failed_refresh_keeps_stale_keys_within_the_window(
    rsa_key: rsa.RSAPrivateKey, failure: str
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    clock = FakeClock()
    auth = _auth(issuer, clock)
    token = _token(rsa_key)
    _verify(auth, token)
    expires_at = clock.now + 300

    if failure == "unreachable":
        issuer.unreachable = True
    elif failure == "http-500":
        issuer.failing = True
    elif failure == "not-json":
        issuer.jwks_body = b"<html>maintenance</html>"
    elif failure == "no-keys-array":
        issuer.jwks_body = b'{"keys": "none"}'
    else:
        issuer.keys = issuer.keys * 65

    clock.now = expires_at + 1
    assert _verify(auth, token)["sub"] == "user-1"
    clock.now = expires_at + auth.signing_keys.max_stale_seconds - 1
    assert _verify(auth, token)["sub"] == "user-1"
    clock.now = expires_at + auth.signing_keys.max_stale_seconds
    assert _rejection(auth, token).status_code == 503


def test_max_stale_seconds_is_configurable(rsa_key: rsa.RSAPrivateKey) -> None:
    assert OidcSigningKeys(ISSUER).max_stale_seconds == 3600
    for value in (-1, 24 * 60 * 60 + 1, float("nan"), "60"):
        with pytest.raises(ValueError, match="max_stale_seconds"):
            OidcSigningKeys(ISSUER, max_stale_seconds=value)  # ty: ignore[invalid-argument-type]

    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    clock = FakeClock()
    keys = OidcSigningKeys(ISSUER, http_client=issuer.client(), max_stale_seconds=0)
    keys._monotonic = clock
    auth = OidcBearerAuth(issuer=ISSUER, audience=AUDIENCE, signing_keys=keys)
    token = _token(rsa_key)
    _verify(auth, token)

    issuer.unreachable = True
    clock.now += 299
    assert _verify(auth, token)["sub"] == "user-1"
    # With no stale window, keys stop verifying as soon as they expire.
    clock.now += 1
    assert _rejection(auth, token).status_code == 503


def test_unreachable_issuer_without_keys_returns_503(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.failing = True
    auth = _auth(issuer)
    token = _token(rsa_key)

    response = _client(auth).get("/whoami", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Bearer token verification is temporarily unavailable."}
    assert "www-authenticate" not in response.headers


def test_jwks_with_too_many_keys_is_refused(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    for index in range(70):
        issuer.add_rsa(f"key-{index}", rsa_key)

    error = _rejection(_auth(issuer), _token(rsa_key, kid="key-1"))

    assert error.status_code == 503


def test_jwks_ignores_private_encryption_and_symmetric_keys(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("enc", rsa_key, use="enc")
    issuer.keys.append({"kty": "oct", "kid": "hmac", "k": _b64(b"secret")})
    private = RSAAlgorithm.to_jwk(rsa_key, as_dict=True)
    issuer.keys.append({**private, "kid": "private", "use": "sig"})
    issuer.add_rsa("key-1", rsa_key)
    auth = _auth(issuer)

    assert _verify(auth, _token(rsa_key))["sub"] == "user-1"
    for kid in ("enc", "hmac", "private"):
        assert _rejection(auth, _token(rsa_key, kid=kid)).reason == "unknown signing key"


class _StreamedBody(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, repeat: int = 1) -> None:
        self.chunks = chunks
        self.repeat = repeat
        self.sent = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(self.repeat):
            for chunk in self.chunks:
                self.sent += len(chunk)
                yield chunk


def _streaming_keys(body: _StreamedBody, headers: dict[str, str]) -> OidcSigningKeys:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, stream=body)

    return OidcSigningKeys(
        ISSUER,
        jwks_url=JWKS_URL,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _peak_allocation_of_failed_fetch(keys: OidcSigningKeys) -> int:
    async def fetch() -> None:
        with pytest.raises(OidcTokenError) as raised:
            await keys.signing_key("key-1", "RS256")
        assert raised.value.status_code == 503

    tracemalloc.start()
    try:
        asyncio.run(fetch())
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_key_fetches_request_identity_encoding(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)

    _verify(_auth(issuer), _token(rsa_key))

    assert issuer.accept_encodings == ["identity", "identity"]


def test_compressed_jwks_bomb_is_refused_without_large_allocation() -> None:
    compressor = zlib.compressobj(9, zlib.DEFLATED, 31)
    bomb = compressor.compress(b"[" + b" " * (16 * 1024 * 1024)) + compressor.flush()
    assert len(bomb) < 32 * 1024
    body = _StreamedBody([bomb[index : index + 4096] for index in range(0, len(bomb), 4096)])
    keys = _streaming_keys(body, {"Content-Encoding": "gzip"})

    peak = _peak_allocation_of_failed_fetch(keys)

    assert peak < 2 * 1024 * 1024
    # The compressed body is refused from its headers, before any of it is read.
    assert body.sent == 0


@pytest.mark.parametrize("encoding", ["gzip", "br", "deflate", "gzip, identity"])
def test_any_compressed_document_is_refused(
    rsa_key: rsa.RSAPrivateKey, encoding: str, caplog: pytest.LogCaptureFixture
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    document = json.dumps({"keys": issuer.keys}).encode()
    keys = _streaming_keys(_StreamedBody([document]), {"Content-Encoding": encoding})
    auth = OidcBearerAuth(issuer=ISSUER, audience=AUDIENCE, signing_keys=keys)

    with caplog.at_level(logging.WARNING, logger="cayu.server.oidc"):
        assert _rejection(auth, _token(rsa_key)).status_code == 503

    assert "is compressed" in caplog.text


@pytest.mark.parametrize(
    ("chunk_bytes", "repeat", "chunks_read"),
    [(64 * 1024, 256, 5), (8 * 1024 * 1024, 2, 1)],
)
def test_oversized_identity_stream_is_refused_at_the_cap(
    chunk_bytes: int, repeat: int, chunks_read: int
) -> None:
    chunk = b" " * chunk_bytes
    body = _StreamedBody([chunk], repeat=repeat)
    keys = _streaming_keys(body, {})

    peak = _peak_allocation_of_failed_fetch(keys)

    # A chunk that would cross the 256 KiB cap is refused before it is copied.
    assert peak < 2 * 1024 * 1024
    assert body.sent == chunks_read * chunk_bytes


def test_discovery_issuer_mismatch_is_refused(
    rsa_key: rsa.RSAPrivateKey, caplog: pytest.LogCaptureFixture
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    issuer.discovery_issuer = "https://evil.example.com/"

    with caplog.at_level(logging.WARNING, logger="cayu.server.oidc"):
        error = _rejection(_auth(issuer), _token(rsa_key))

    assert error.status_code == 503
    assert JWKS_URL not in issuer.requests
    assert "issuer does not match" in caplog.text


def test_https_is_required_except_explicit_loopback(rsa_key: rsa.RSAPrivateKey) -> None:
    for issuer_url in ("http://idp.example.com/", "ftp://idp.example.com", "idp.example.com"):
        with pytest.raises(ValueError, match="https://"):
            OidcBearerAuth(issuer=issuer_url, audience=AUDIENCE)
    with pytest.raises(ValueError, match="https://"):
        OidcBearerAuth(issuer=ISSUER, audience=AUDIENCE, jwks_url="http://idp.example.com/keys")
    with pytest.raises(ValueError, match="https://"):
        OidcBearerAuth(
            issuer="http://idp.example.com/", audience=AUDIENCE, allow_insecure_loopback=True
        )
    with pytest.raises(ValueError, match="credentials"):
        OidcBearerAuth(issuer="https://user:pass@idp.example.com/", audience=AUDIENCE)

    local = FakeIssuer("http://127.0.0.1:8080", jwks_url="http://127.0.0.1:8080/jwks")
    local.add_rsa("key-1", rsa_key)
    auth = _auth(local, allow_insecure_loopback=True)
    assert _verify(auth, _token(rsa_key, {"iss": "http://127.0.0.1:8080"}))["sub"] == "user-1"

    # A discovered plain-HTTP jwks_uri is refused even when the issuer is HTTPS.
    remote = FakeIssuer()
    remote.add_rsa("key-1", rsa_key)
    remote.discovery_jwks_uri = "http://idp.example.com/keys/jwks.json"
    assert _rejection(_auth(remote), _token(rsa_key)).status_code == 503


@pytest.mark.parametrize(
    ("authorization", "status", "error"),
    [
        (None, 401, None),
        ("", 401, None),
        ("Basic dXNlcjpwYXNz", 401, None),
        ("Bearer", 401, None),
        ("Bearer ", 401, None),
        ("Bearer not-a-jwt", 401, "invalid_token"),
        ("Bearer a.b", 401, "invalid_token"),
        ("Bearer a.b.c.d.e", 401, "invalid_token"),
        ("Bearer  a.b.c", 401, "invalid_token"),
        ("Bearer bm90LWpzb24.e30.c2ln", 401, "invalid_token"),
        ("Bearer " + "a" * 20000 + ".b.c", 401, "invalid_token"),
    ],
)
def test_malformed_and_missing_credentials_get_bearer_challenges(
    authorization: str | None, status: int, error: str | None
) -> None:
    issuer = FakeIssuer()
    headers = {} if authorization is None else {"Authorization": authorization}

    response = _client(_auth(issuer, realm="Support")).get("/whoami", headers=headers)

    assert response.status_code == status
    challenge = response.headers["www-authenticate"]
    assert challenge.startswith('Bearer realm="Support"')
    if error is None:
        assert "error=" not in challenge
    else:
        assert f'error="{error}"' in challenge
    assert issuer.requests == []


def test_required_scopes_accept_scope_and_scp(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    auth = _auth(issuer, required_scopes=["cayu:operate", "read"])

    assert _verify(auth, _token(rsa_key, {"scope": "read cayu:operate extra"}))
    assert _verify(auth, _token(rsa_key, {"scp": ["cayu:operate", "read"]}))
    assert _verify(auth, _token(rsa_key, {"scp": "read cayu:operate"}))

    response = _client(auth).get(
        "/whoami", headers={"Authorization": f"Bearer {_token(rsa_key, {'scope': 'read'})}"}
    )
    assert response.status_code == 403
    assert response.headers["www-authenticate"] == (
        'Bearer realm="Cayu", error="insufficient_scope", scope="cayu:operate read"'
    )
    with pytest.raises(ValueError, match="scope"):
        OidcBearerAuth(issuer=ISSUER, audience=AUDIENCE, required_scopes=['bad"scope'])


def test_cognito_style_access_token_with_client_id_and_token_use(
    rsa_key: rsa.RSAPrivateKey,
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    auth = _auth(issuer, audience_claim="client_id", required_claims={"token_use": "access"})

    access = _token(rsa_key, {"aud": None, "client_id": AUDIENCE, "token_use": "access"})
    id_token = _token(rsa_key, {"aud": None, "client_id": AUDIENCE, "token_use": "id"})
    other_client = _token(rsa_key, {"aud": None, "client_id": "other", "token_use": "access"})

    assert _verify(auth, access)["client_id"] == AUDIENCE
    assert "required claim" in _rejection(auth, id_token).reason
    assert "audience" in _rejection(auth, other_client).reason
    presence = _auth(issuer, required_claims=["email"])
    assert "required claim" in _rejection(presence, _token(rsa_key)).reason
    assert _verify(presence, _token(rsa_key, {"email": "a@b.example"}))


def test_product_dependency_maps_tenant_claim_and_fails_closed(
    rsa_key: rsa.RSAPrivateKey,
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    auth = _auth(issuer)
    dependency = auth.product_dependency(tenant_claim=("org", "id"))
    app = FastAPI()

    @app.get("/me")
    async def me(principal: ProductPrincipal = Depends(dependency)) -> dict[str, str]:  # noqa: B008
        return principal.model_dump()

    client = TestClient(app)

    def call(token: str) -> httpx.Response:
        return client.get("/me", headers={"Authorization": f"Bearer {token}"})

    ok = call(_token(rsa_key, {"org": {"id": "tenant-a"}}))
    assert ok.status_code == 200
    assert ok.json() == {"tenant_id": "tenant-a", "subject_id": "user-1"}
    for claims in ({}, {"org": {"id": ""}}, {"org": {"id": 7}}, {"org": "tenant-a"}):
        rejected = call(_token(rsa_key, claims))
        assert rejected.status_code == 401
        assert 'error="invalid_token"' in rejected.headers["www-authenticate"]
    assert call("").status_code == 401

    with pytest.raises(ValueError, match="tenant claim"):
        auth.product_dependency()
    namespaced = _auth(issuer, tenant_claim="https://example.com/tenant").product_dependency()
    principal = asyncio.run(
        resolve_product(namespaced, _token(rsa_key, {"https://example.com/tenant": "t-9"}))
    )
    assert principal == ProductPrincipal(tenant_id="t-9", subject_id="user-1")


async def resolve_product(dependency: Any, token: str) -> ProductPrincipal:
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"authorization", f"Bearer {token}".encode())],
    }
    return await dependency(Request(scope))


def test_operator_tenant_claim_is_required_when_configured(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)

    error = _rejection(_auth(issuer, tenant_claim="tid"), _token(rsa_key))

    assert "tenant" in error.reason


def test_claims_mapper_customizes_bounded_context_claims(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    auth = _auth(
        issuer,
        claims_mapper=lambda claims: {"email": claims["email"], "groups": claims["groups"]},
    )
    token = _token(rsa_key, {"email": "a@b.example", "groups": ["ops"]})

    response = _client(auth).get("/whoami", headers={"Authorization": f"Bearer {token}"})

    assert response.json()["claims"] == {"email": "a@b.example", "groups": ["ops"]}

    oversized = _auth(issuer, claims_mapper=lambda claims: {"blob": "x" * 20000})
    with pytest.raises(RuntimeError, match="claims_mapper"):
        oversized.auth_context(_verify(oversized, token))


def test_no_token_text_in_error_bodies_or_logs(
    rsa_key: rsa.RSAPrivateKey,
    other_rsa_key: rsa.RSAPrivateKey,
    caplog: pytest.LogCaptureFixture,
) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    auth = _auth(issuer, required_scopes=["admin"])
    secret_subject = "subject-sentinel-8f1c"
    tokens = [
        _token(rsa_key, {"sub": secret_subject, "exp": int(time.time()) - 900}),
        _token(other_rsa_key, {"sub": secret_subject}),
        _token(rsa_key, {"sub": secret_subject, "aud": "wrong"}),
        _token(rsa_key, {"sub": secret_subject, "scope": "read"}),
        _token(rsa_key, {"sub": secret_subject}, kid="missing-kid"),
    ]
    client = _client(auth)

    with caplog.at_level(logging.DEBUG):
        responses = [
            client.get("/whoami", headers={"Authorization": f"Bearer {token}"}) for token in tokens
        ]

    assert [response.status_code for response in responses] == [401, 401, 401, 403, 401]
    for token, response in zip(tokens, responses, strict=True):
        visible = response.text + json.dumps(dict(response.headers))
        for fragment in token.split("."):
            assert fragment not in visible
            assert fragment not in caplog.text
        assert secret_subject not in visible
    assert secret_subject not in caplog.text
    assert "Rejected an OIDC bearer token" in caplog.text


def test_from_environment_reads_issuer_and_audiences(rsa_key: rsa.RSAPrivateKey) -> None:
    auth = OidcBearerAuth.from_environment(
        environ={"CAYU_OIDC_ISSUER": f" {ISSUER} ", "CAYU_OIDC_AUDIENCE": "api://cayu, other"},
        tenant_claim="org_id",
    )
    assert auth.issuer == ISSUER
    assert auth.audience == ("api://cayu", "other")
    assert auth.tenant_claim == ("org_id",)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, "CAYU_OIDC_ISSUER and CAYU_OIDC_AUDIENCE are unset or empty"),
        ({"CAYU_OIDC_AUDIENCE": "api://secret-audience"}, "CAYU_OIDC_ISSUER is unset or empty"),
        (
            {"CAYU_OIDC_ISSUER": "https://secret-issuer.example/", "CAYU_OIDC_AUDIENCE": "  "},
            "CAYU_OIDC_AUDIENCE is unset or empty",
        ),
        (
            {"CAYU_OIDC_ISSUER": "http://secret-issuer.example", "CAYU_OIDC_AUDIENCE": "a"},
            "CAYU_OIDC_ISSUER cannot be used as the OIDC issuer: `issuer` must be an https:// URL",
        ),
        (
            {
                "CAYU_OIDC_ISSUER": "https://secret-issuer.example/?tenant=x",
                "CAYU_OIDC_AUDIENCE": "a",
            },
            "CAYU_OIDC_ISSUER cannot be used as the OIDC issuer: `issuer` must not contain a query",
        ),
        (
            {"CAYU_OIDC_ISSUER": "https://secret-issuer.example/", "CAYU_OIDC_AUDIENCE": " , ,"},
            "CAYU_OIDC_AUDIENCE does not name an audience",
        ),
    ],
)
def test_from_environment_reports_configuration_errors_without_values(
    environ: dict[str, str], expected: str
) -> None:
    with pytest.raises(AuthConfigurationError) as raised:
        OidcBearerAuth.from_environment(environ=environ)

    message = str(raised.value)
    assert expected in message
    assert "secret-issuer" not in message
    assert "secret-audience" not in message
    assert "tenant=x" not in message
    assert isinstance(raised.value, ValueError)
    if "unset or empty" in message:
        assert "does not fall back to open access" in message
        assert "`cayu serve --dev`" in message


def test_from_environment_keeps_value_error_for_programming_errors() -> None:
    environ = {"ISS": ISSUER, "AUD": AUDIENCE}
    with pytest.raises(ValueError, match="must differ") as same:
        OidcBearerAuth.from_environment("ISS", "ISS", environ=environ)
    with pytest.raises(ValueError, match="environment variable name") as blank:
        OidcBearerAuth.from_environment(" ", "AUD", environ=environ)
    with pytest.raises(ValueError, match="max_token_bytes") as option:
        OidcBearerAuth.from_environment("ISS", "AUD", environ=environ, max_token_bytes=0)
    for raised in (same, blank, option):
        assert not isinstance(raised.value, AuthConfigurationError)


@pytest.mark.parametrize("issuer", [ISSUER, "http://127.0.0.1:8123"])
@pytest.mark.parametrize("allow_insecure_loopback", [1, "true", None])
def test_from_environment_rejects_non_boolean_loopback_option_as_programming_error(
    issuer: str, allow_insecure_loopback: object
) -> None:
    environ = {"CAYU_OIDC_ISSUER": issuer, "CAYU_OIDC_AUDIENCE": AUDIENCE}

    with pytest.raises(ValueError, match="`allow_insecure_loopback` must be a boolean") as raised:
        OidcBearerAuth.from_environment(
            environ=environ, allow_insecure_loopback=allow_insecure_loopback
        )

    assert type(raised.value) is ValueError


def test_from_environment_issuer_check_follows_the_loopback_option() -> None:
    environ = {"CAYU_OIDC_ISSUER": "http://127.0.0.1:8123", "CAYU_OIDC_AUDIENCE": AUDIENCE}
    with pytest.raises(AuthConfigurationError, match="CAYU_OIDC_ISSUER cannot be used"):
        OidcBearerAuth.from_environment(environ=environ)

    auth = OidcBearerAuth.from_environment(environ=environ, allow_insecure_loopback=True)
    assert auth.issuer == "http://127.0.0.1:8123"

    keys = OidcSigningKeys("http://127.0.0.1:8123", allow_insecure_loopback=True)
    shared = OidcBearerAuth.from_environment(environ=environ, signing_keys=keys)
    assert shared.signing_keys is keys


def test_shared_signing_keys_serve_separate_policies(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    keys = OidcSigningKeys(ISSUER, http_client=issuer.client())
    operator = OidcBearerAuth(
        issuer=ISSUER, audience=AUDIENCE, signing_keys=keys, required_scopes=["cayu:operate"]
    )
    customer = OidcBearerAuth(issuer=ISSUER, audience=AUDIENCE, signing_keys=keys)
    token = _token(rsa_key, {"scope": "chat"})

    assert _verify(customer, token)["scope"] == "chat"
    assert _rejection(operator, token).error == "insufficient_scope"
    assert issuer.jwks_requests() == 1
    with pytest.raises(ValueError, match="different issuer"):
        OidcBearerAuth(issuer="https://other.example/", audience=AUDIENCE, signing_keys=keys)
    with pytest.raises(ValueError, match="shared OidcSigningKeys"):
        OidcBearerAuth(issuer=ISSUER, audience=AUDIENCE, signing_keys=keys, jwks_url=JWKS_URL)


def test_missing_pyjwt_reports_the_oidc_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "jwt", None)

    with pytest.raises(RuntimeError, match=r"cayu\[oidc\]"):
        OidcBearerAuth(issuer=ISSUER, audience=AUDIENCE)


def test_protected_server_accepts_only_verified_bearer_tokens(rsa_key: rsa.RSAPrivateKey) -> None:
    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    auth = _auth(issuer)
    client = TestClient(create_server(CayuApp(), config=ServerConfig.protected(auth)))

    assert client.get("/api/health").json() == {"ok": True}
    denied = client.get("/api/sessions")
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"] == 'Bearer realm="Cayu"'
    expired = _token(rsa_key, {"exp": int(time.time()) - 900})
    rejected = client.get("/api/sessions", headers={"Authorization": f"Bearer {expired}"})
    assert rejected.status_code == 401
    assert 'error="invalid_token"' in rejected.headers["www-authenticate"]
    accepted = client.get("/api/sessions", headers={"Authorization": f"Bearer {_token(rsa_key)}"})
    assert accepted.status_code == 200
    assert accepted.json()["sessions"] == []

    context = asyncio.run(resolve_auth_context(auth, _request(_token(rsa_key))))
    assert context.subject == "user-1"


def _request(token: str) -> Any:
    from starlette.requests import Request

    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"authorization", f"Bearer {token}".encode())],
        }
    )


def test_public_service_uses_oidc_for_product_and_operator_access(
    rsa_key: rsa.RSAPrivateKey,
) -> None:
    from tests.server.test_public_service import _build_service

    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    keys = OidcSigningKeys(ISSUER, http_client=issuer.client())
    customers = OidcBearerAuth(
        issuer=ISSUER, audience=AUDIENCE, signing_keys=keys, tenant_claim="org_id"
    )
    operators = OidcBearerAuth(
        issuer=ISSUER, audience=AUDIENCE, signing_keys=keys, required_scopes=["cayu:operate"]
    )
    service, store, _provider = _build_service(
        product_access=AuthenticatedProductAccess(dependency=customers.product_dependency()),
        operator_access=AuthenticatedAccess(dependency=operators),
    )

    assert service.manifest.product_access == "authenticated"
    assert service.manifest.operator_access == "authenticated"
    codes = {finding.code for finding in check_public_service_deployment(service.manifest)}
    assert "PUBLIC_SERVICE_PRODUCT_ACCESS_UNSAFE" not in codes
    assert "PUBLIC_SERVICE_OPERATOR_ACCESS_UNSAFE" not in codes

    client = TestClient(service.asgi_app)
    customer_token = _token(rsa_key, {"org_id": "tenant-a", "sub": "alice"})
    created = client.post(
        "/api/operations",
        headers={"Authorization": f"Bearer {customer_token}", "Idempotency-Key": "oidc-1"},
        json={"request": "summarize"},
    )
    assert created.status_code == 201
    operation = store.by_public_id[created.json()["id"]]
    assert (operation.tenant_id, operation.subject_id) == ("tenant-a", "alice")

    no_tenant = client.post(
        "/api/operations",
        headers={"Authorization": f"Bearer {_token(rsa_key)}", "Idempotency-Key": "oidc-2"},
        json={"request": "summarize"},
    )
    assert no_tenant.status_code == 401
    # A customer token without the operator scope cannot reach the control plane.
    operator_denied = client.get(
        "/cayu/api/sessions", headers={"Authorization": f"Bearer {customer_token}"}
    )
    assert operator_denied.status_code == 403
    operator_token = _token(rsa_key, {"scope": "cayu:operate"})
    assert (
        client.get(
            "/cayu/api/sessions", headers={"Authorization": f"Bearer {operator_token}"}
        ).status_code
        == 200
    )


_SERVE_PROJECT = """import json
import os

import httpx

from cayu import CayuApp
from cayu.server import OidcBearerAuth

_JWKS = json.loads(os.environ["TEST_OIDC_JWKS"])
AUTH = OidcBearerAuth.from_environment(
    jwks_url="https://idp.example.com/keys/jwks.json",
    http_client=httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=_JWKS))
    ),
)


def build_app():
    return CayuApp(enable_logging=False)
"""


def _write_serve_project(tmp_path: Path, module: str) -> None:
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.cayu]\nfactory = "{module}:build_app"\n\n'
        f'[tool.cayu.serve]\nauth = "{module}:AUTH"\n',
        encoding="utf-8",
    )
    (tmp_path / f"{module}.py").write_text(_SERVE_PROJECT, encoding="utf-8")


def test_cayu_serve_uses_oidc_auth_target_from_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rsa_key: rsa.RSAPrivateKey,
) -> None:
    from cayu.cli import main

    issuer = FakeIssuer()
    issuer.add_rsa("key-1", rsa_key)
    _write_serve_project(tmp_path, "oidc_serve_project")
    monkeypatch.setenv("TEST_OIDC_JWKS", json.dumps({"keys": issuer.keys}))
    monkeypatch.setenv("CAYU_OIDC_ISSUER", ISSUER)
    monkeypatch.setenv("CAYU_OIDC_AUDIENCE", AUDIENCE)
    launched: dict[str, Any] = {}
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda server, *, host, port: launched.update(server=server)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "oidc_serve_project", raising=False)

    assert main(["serve", "--host", "0.0.0.0"]) == 0

    with TestClient(launched["server"]) as client:
        assert client.get("/api/sessions").status_code == 401
        response = client.get(
            "/api/sessions", headers={"Authorization": f"Bearer {_token(rsa_key)}"}
        )
        assert response.status_code == 200


def test_cayu_serve_refuses_to_start_without_oidc_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cayu.cli import main

    _write_serve_project(tmp_path, "oidc_unset_project")
    monkeypatch.setenv("TEST_OIDC_JWKS", json.dumps({"keys": []}))
    monkeypatch.delenv("CAYU_OIDC_ISSUER", raising=False)
    monkeypatch.setenv("CAYU_OIDC_AUDIENCE", AUDIENCE)
    uvicorn = ModuleType("uvicorn")
    uvicorn.run = lambda server, *, host, port: pytest.fail("server must not start")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "oidc_unset_project", raising=False)

    assert main(["serve", "--host", "0.0.0.0"]) == 1

    error = capsys.readouterr().err
    assert "CAYU_OIDC_ISSUER is unset or empty" in error
    assert "does not fall back to open access" in error
