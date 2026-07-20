from __future__ import annotations

import asyncio
import base64
import gzip
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx
import pytest
from pydantic import SecretStr

from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    EgressDecision,
    EgressPolicy,
    HttpEgressPolicy,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.vaults import ResolvedSecret, SecretRef, StaticVault

REAL_SECRET = "sk_test_51RealDeadBeefSecretValue"
GITHUB_SECRET = "github_pat_11RealDeadBeefSecretValue"


def _destination_resolver(*addresses: str):  # type: ignore[no-untyped-def]
    async def resolve(_host: str, _port: int) -> tuple[str, ...]:
        return addresses

    return resolve


class _RecordingResolver:
    """Counts resolutions so tests can prove deny-before-resolve."""

    def __init__(self, vault: StaticVault) -> None:
        self._vault = vault
        self.resolve_count = 0

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        self.resolve_count += 1
        return await self._vault.resolve(ref, scope=scope)


class _FakeUpstream:
    def __init__(self, response: CapturedResponse) -> None:
        self._response = response
        self.sent: CapturedRequest | None = None

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        self.sent = request
        return self._response


class _FailingUpstream:
    async def send(self, request: CapturedRequest) -> CapturedResponse:
        raise RuntimeError(f"boom with {REAL_SECRET}")  # secret in exception must not leak out


class _Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def _build(
    *,
    upstream: Any,
    clock: _Clock | None = None,
    policies: Mapping[str, EgressPolicy] | None = None,
) -> tuple[
    TransparentEgressBroker, VirtualCredentialRegistry, _RecordingResolver, list[EgressDecision]
]:
    registry = VirtualCredentialRegistry(clock=clock or _Clock(datetime(2026, 7, 6, tzinfo=UTC)))
    resolver = _RecordingResolver(
        StaticVault(
            {
                "github_token": GITHUB_SECRET,
                "stripe_test_key": REAL_SECRET,
            }
        )
    )
    decisions: list[EgressDecision] = []
    if policies is None:
        policies = {"stripe-example": _stripe_example_policy()}
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=resolver,
        policies=policies,
        upstream=upstream,
        audit=decisions.append,
    )
    return broker, registry, resolver, decisions


def _mint(registry: VirtualCredentialRegistry, **overrides: Any):
    params: dict[str, Any] = {
        "session_id": "sess_1",
        "env_name": "STRIPE_SECRET_KEY",
        "secret": SecretRef(name="stripe_test_key"),
        "destination": "api.stripe.com",
        "credential_kind": "stripe_bearer",
        "policy_name": "stripe-example",
    }
    params.update(overrides)
    return registry.mint(**params)


def _stripe_example_policy() -> HttpEgressPolicy:
    return HttpEgressPolicy(
        name="stripe-example",
        allowed_hosts=["api.stripe.com"],
        allowed_endpoints=[("POST", "/v1/customers")],
        denied_prefixes=["/v1/payouts"],
    )


def _request(grant_value: str, path: str, form: dict[str, str] | None = None) -> CapturedRequest:
    body = urlencode(form).encode() if form else b""
    return CapturedRequest(
        method="POST",
        host="api.stripe.com",
        path=path,
        headers={
            "Authorization": f"Bearer {grant_value}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        body=body,
    )


def _no_real_secret(decisions: list[EgressDecision]) -> None:
    for decision in decisions:
        assert REAL_SECRET not in str(asdict(decision))


def test_allowed_request_injects_real_secret_upstream_only() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b'{"id":"cus_1"}'))
    broker, registry, resolver, decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 200
    # Upstream received the REAL secret...
    assert upstream.sent is not None
    assert upstream.sent.headers["Authorization"] == f"Bearer {REAL_SECRET}"
    # ...and the virtual value never went upstream.
    assert grant.presented_value not in str(upstream.sent.headers)
    # Response returned to the sandbox contains no real secret.
    assert REAL_SECRET not in response.body.decode()
    assert resolver.resolve_count == 1
    assert decisions[-1].allowed is True
    _no_real_secret(decisions)


def test_stripe_basic_request_injects_real_secret_as_bearer_upstream() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b'{"id":"cus_1"}'))
    broker, registry, resolver, decisions = _build(upstream=upstream)
    grant = _mint(registry)
    basic_value = base64.b64encode(f"{grant.presented_value}:".encode()).decode()
    request = CapturedRequest(
        method="POST",
        host="api.stripe.com",
        path="/v1/customers",
        headers={
            "Authorization": f"Basic {basic_value}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )

    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 200
    assert upstream.sent is not None
    assert upstream.sent.headers["Authorization"] == f"Bearer {REAL_SECRET}"
    assert grant.presented_value not in str(upstream.sent.headers)
    assert resolver.resolve_count == 1
    assert decisions[-1].allowed is True
    _no_real_secret(decisions)


def test_opaque_token_request_injects_real_secret_upstream_only() -> None:
    upstream = _FakeUpstream(
        CapturedResponse(
            status_code=200,
            body=f'{{"debug":"{GITHUB_SECRET}"}}'.encode(),
        )
    )
    github_policy = HttpEgressPolicy(
        name="github-read",
        allowed_hosts=["api.github.com"],
        allowed_endpoints=[("GET", "/user")],
    )
    broker, registry, resolver, decisions = _build(
        upstream=upstream,
        policies={"github-read": github_policy},
    )
    grant = _mint(
        registry,
        env_name="GH_TOKEN",
        secret=SecretRef(name="github_token"),
        destination="api.github.com",
        credential_kind="opaque_token",
        policy_name="github-read",
    )
    request = CapturedRequest(
        method="GET",
        host="api.github.com",
        path="/user",
        headers={"Authorization": f"token {grant.presented_value}"},
    )

    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 200
    assert upstream.sent is not None
    assert upstream.sent.headers["Authorization"] == f"token {GITHUB_SECRET}"
    assert grant.presented_value not in str(upstream.sent.headers)
    assert GITHUB_SECRET not in response.body.decode()
    assert b"[REDACTED_SECRET]" in response.body
    assert resolver.resolve_count == 1
    assert decisions[-1].allowed is True
    assert GITHUB_SECRET not in str(asdict(decisions[-1]))
    _no_real_secret(decisions)


@pytest.mark.parametrize(
    ("credential_kind", "authorization_template"),
    [
        ("opaque_bearer", "token {credential}"),
        ("opaque_token", "Bearer {credential}"),
        ("opaque_token", "{credential}"),
    ],
)
def test_opaque_credential_rejects_mismatched_or_missing_authorization_scheme(
    credential_kind: str,
    authorization_template: str,
) -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    github_policy = HttpEgressPolicy(
        name="github-read",
        allowed_hosts=["api.github.com"],
        allowed_endpoints=[("GET", "/user")],
    )
    broker, registry, resolver, decisions = _build(
        upstream=upstream,
        policies={"github-read": github_policy},
    )
    grant = _mint(
        registry,
        env_name="GH_TOKEN",
        secret=SecretRef(name="github_token"),
        destination="api.github.com",
        credential_kind=credential_kind,
        policy_name="github-read",
    )
    request = CapturedRequest(
        method="GET",
        host="api.github.com",
        path="/user",
        headers={"Authorization": authorization_template.format(credential=grant.presented_value)},
    )

    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 403
    assert b"authentication scheme does not match" in response.body
    assert resolver.resolve_count == 0
    assert upstream.sent is None
    assert decisions[-1].allowed is False


def test_allowed_response_redacts_echoed_real_secret() -> None:
    upstream = _FakeUpstream(
        CapturedResponse(
            status_code=200,
            headers={"X-Echo-Secret": f"provider echoed {REAL_SECRET}"},
            body=f'{{"debug":"{REAL_SECRET}"}}'.encode(),
        )
    )
    broker, registry, _resolver, decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 200
    assert REAL_SECRET not in str(response.headers)
    assert REAL_SECRET not in response.body.decode()
    assert "[REDACTED_SECRET]" in response.headers["X-Echo-Secret"]
    assert b"[REDACTED_SECRET]" in response.body
    _no_real_secret(decisions)


def test_denied_endpoint_never_resolves_secret() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, decisions = _build(upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/payouts", {"amount": "100"}))
    )

    assert response.status_code == 403
    assert resolver.resolve_count == 0  # deny-before-resolve
    assert upstream.sent is None  # never forwarded
    assert decisions[-1].allowed is False
    _no_real_secret(decisions)


def test_unknown_credential_denied() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, decisions = _build(upstream=upstream)
    _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request("sk_test_cayu_vc_bogus", "/v1/customers"))
    )

    assert response.status_code == 403
    assert resolver.resolve_count == 0
    _no_real_secret(decisions)


def test_missing_credential_denied() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, _registry, resolver, _decisions = _build(upstream=upstream)

    request = CapturedRequest(method="GET", host="api.stripe.com", path="/v1/customers")
    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 401
    assert resolver.resolve_count == 0


def test_expired_credential_denied() -> None:
    clock = _Clock(datetime(2026, 7, 6, tzinfo=UTC))
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, _ = _build(upstream=upstream, clock=clock)
    grant = _mint(registry, ttl_seconds=60)
    clock.advance(61)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 403
    assert resolver.resolve_count == 0


def test_destination_mismatch_denied() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, _ = _build(upstream=upstream)
    grant = _mint(registry, destination="api.stripe.com")

    # A request whose host differs from the grant binding.
    request = CapturedRequest(
        method="POST",
        host="uploads.stripe.com",
        path="/v1/customers",
        headers={"Authorization": f"Bearer {grant.presented_value}"},
    )
    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 403
    assert resolver.resolve_count == 0


def test_missing_policy_denied() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, _ = _build(upstream=upstream, policies={})
    grant = _mint(registry)

    response = asyncio.run(broker.handle_request(_request(grant.presented_value, "/v1/customers")))

    assert response.status_code == 403
    assert resolver.resolve_count == 0


def test_unsupported_credential_kind_rejected_at_mint() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, resolver, _ = _build(upstream=upstream)

    with pytest.raises(ValueError, match="Unsupported credential kind"):
        _mint(registry, credential_kind="mystery_kind")
    assert resolver.resolve_count == 0
    assert broker.registry is registry


def test_upstream_failure_is_sanitized() -> None:
    broker, registry, _resolver, decisions = _build(upstream=_FailingUpstream())
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 502
    assert REAL_SECRET not in response.body.decode()
    _no_real_secret(decisions)


def test_httpx_upstream_strips_stale_compression_headers_after_decoding() -> None:
    decoded = b'{"ok":true}'
    encoded = gzip.compress(decoded)
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            headers={
                "Content-Encoding": "gzip",
                "Content-Length": str(len(encoded)),
                "Content-Type": "application/json",
            },
            content=encoded,
            request=request,
        )

    async def run() -> CapturedResponse:
        upstream = HttpxUpstream(
            transport=httpx.MockTransport(handler),
            destination_resolver=_destination_resolver("93.184.216.34"),
        )
        return await upstream.send(
            CapturedRequest(method="GET", host="api.stripe.com", path="/v1/customers")
        )

    response = asyncio.run(run())

    assert response.body == decoded
    assert response.headers["content-type"] == "application/json"
    assert "content-encoding" not in {key.lower() for key in response.headers}
    assert "content-length" not in {key.lower() for key in response.headers}
    assert str(captured[0].url) == "https://93.184.216.34/v1/customers"
    assert captured[0].headers["host"] == "api.stripe.com"
    assert captured[0].extensions["sni_hostname"] == "api.stripe.com"


def test_httpx_upstream_routes_logical_host_to_private_service() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(202, json={"accepted": True}, request=request)

    async def run() -> CapturedResponse:
        upstream = HttpxUpstream(
            routes={"receiver.internal": "http://receiver.service.local:8080"},
            transport=httpx.MockTransport(handler),
            destination_resolver=_destination_resolver("10.0.0.10"),
        )
        return await upstream.send(
            CapturedRequest(
                method="POST",
                host="receiver.internal",
                path="/v1/actions",
                query="mode=safe",
                headers={"Authorization": "Bearer real-secret"},
                body=b"{}",
            )
        )

    response = asyncio.run(run())

    assert response.status_code == 202
    assert str(captured[0].url) == "http://10.0.0.10:8080/v1/actions?mode=safe"
    assert captured[0].headers["host"] == "receiver.service.local:8080"
    assert captured[0].headers["authorization"] == "Bearer real-secret"


@pytest.mark.parametrize(
    "addresses",
    [
        ("169.254.169.254",),
        ("127.0.0.1",),
        ("10.0.0.1",),
        ("93.184.216.34", "169.254.169.254"),
    ],
)
def test_httpx_upstream_rejects_dns_rebinding_before_transport(
    addresses: tuple[str, ...],
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request)

    async def run() -> None:
        upstream = HttpxUpstream(
            transport=httpx.MockTransport(handler),
            destination_resolver=_destination_resolver(*addresses),
        )
        with pytest.raises(ValueError, match="prohibited address"):
            await upstream.send(
                CapturedRequest(
                    method="GET",
                    host="docs.example.com",
                    path="/sdk/index.json",
                )
            )

    asyncio.run(run())

    assert requests == []


@pytest.mark.parametrize(
    "route",
    [
        "receiver.service.local:8080",
        "ftp://receiver.service.local",
        "http://user:password@receiver.service.local",
        "http://receiver.service.local/base?unsafe=1",
    ],
)
def test_httpx_upstream_rejects_unsafe_private_service_route(route: str) -> None:
    with pytest.raises(ValueError, match="route"):
        HttpxUpstream(routes={"receiver.internal": route})


def test_deny_body_is_valid_json() -> None:
    import json

    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry, _resolver, _decisions = _build(upstream=upstream)
    grant = _mint(registry)

    # A denied endpoint (policy denial with a reason that contains characters
    # that would break a naive f-string body, e.g. an apostrophe).
    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/payouts", {"amount": "100"}))
    )

    assert response.status_code == 403
    decoded = json.loads(response.body)  # must be valid JSON
    assert isinstance(decoded["error"]["message"], str)


def test_audit_failure_does_not_drop_response() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b'{"id":"cus_1"}'))
    registry = VirtualCredentialRegistry()

    def _boom(_decision: EgressDecision) -> None:
        raise RuntimeError("audit sink is down")

    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe_test_key": REAL_SECRET}),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
        audit=_boom,
    )
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    # The provider response survives a failing audit sink.
    assert response.status_code == 200
    assert upstream.sent is not None


class _FailingResolver:
    async def resolve(self, ref, *, scope=None):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"vault down for {REAL_SECRET}")  # secret in error must not leak


class _RevokingResolver:
    def __init__(self, registry: VirtualCredentialRegistry, presented_value: str) -> None:
        self._registry = registry
        self._presented_value = presented_value
        self.resolve_count = 0

    async def resolve(self, ref, *, scope=None):  # type: ignore[no-untyped-def]
        self.resolve_count += 1
        self._registry.revoke(self._presented_value)
        return ResolvedSecret(name=ref.name, value=SecretStr(REAL_SECRET))


def test_resolver_failure_is_labeled_distinctly() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    registry = VirtualCredentialRegistry()
    decisions: list[EgressDecision] = []
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=_FailingResolver(),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
        audit=decisions.append,
    )
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 502
    assert b"Credential resolution failed" in response.body
    assert upstream.sent is None  # never reached the upstream
    assert REAL_SECRET not in response.body.decode()
    _no_real_secret(decisions)


def test_revoked_after_resolution_is_not_forwarded_upstream() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    registry = VirtualCredentialRegistry()
    grant = _mint(registry)
    resolver = _RevokingResolver(registry, grant.presented_value)
    decisions: list[EgressDecision] = []
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=resolver,
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
        audit=decisions.append,
    )

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 403
    assert resolver.resolve_count == 1
    assert upstream.sent is None
    assert REAL_SECRET not in response.body.decode()
    assert decisions[-1].allowed is False
    _no_real_secret(decisions)


LIVE_SECRET = "sk_live_51ProductionKeyBoundByMistake"


def _broker_with_secret(secret_value: str, *, upstream: Any, require_test_mode: bool = True):
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=StaticVault({"stripe_test_key": secret_value}),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
        require_test_mode_credentials=require_test_mode,
    )
    return broker, registry


def test_live_key_is_refused_by_default() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry = _broker_with_secret(LIVE_SECRET, upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 403
    assert b"test-mode key" in response.body
    assert upstream.sent is None  # never forwarded a live key upstream
    assert LIVE_SECRET not in response.body.decode()


def test_test_mode_key_passes_the_guard() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b'{"id":"cus_1"}'))
    broker, registry = _broker_with_secret("sk_test_51fine", upstream=upstream)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 200
    assert upstream.sent is not None
    assert upstream.sent.headers["Authorization"] == "Bearer sk_test_51fine"


def test_live_key_allowed_when_opted_out() -> None:
    upstream = _FakeUpstream(CapturedResponse(status_code=200, body=b"{}"))
    broker, registry = _broker_with_secret(LIVE_SECRET, upstream=upstream, require_test_mode=False)
    grant = _mint(registry)

    response = asyncio.run(
        broker.handle_request(_request(grant.presented_value, "/v1/customers", {"email": "a@b.co"}))
    )

    assert response.status_code == 200
    assert upstream.sent is not None
    assert upstream.sent.headers["Authorization"] == f"Bearer {LIVE_SECRET}"


class _RotatingResolver:
    """Returns a different resolved value on each call (simulates rotation)."""

    def __init__(self, values: list[str]) -> None:
        self._values = values
        self._index = 0

    async def resolve(self, ref: SecretRef, *, scope: Any = None) -> ResolvedSecret:
        value = self._values[min(self._index, len(self._values) - 1)]
        self._index += 1
        return ResolvedSecret(name=ref.name, value=SecretStr(value))


class _MultiCaptureUpstream:
    def __init__(self) -> None:
        self.authorizations: list[str | None] = []

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        self.authorizations.append(request.headers.get("Authorization"))
        return CapturedResponse(status_code=200, body=b"{}")


def test_broker_uses_rotated_secret_per_request() -> None:
    # The broker resolves the SecretRef fresh on every request, so a rotated
    # vault value takes effect immediately with no change inside the sandbox.
    upstream = _MultiCaptureUpstream()
    registry = VirtualCredentialRegistry()
    broker = TransparentEgressBroker(
        registry=registry,
        resolver=_RotatingResolver(["sk_test_rotated_one", "sk_test_rotated_two"]),
        policies={"stripe-example": _stripe_example_policy()},
        upstream=upstream,
    )
    grant = _mint(registry)

    for _ in range(2):
        asyncio.run(
            broker.handle_request(
                _request(grant.presented_value, "/v1/customers", {"email": "a@b.co"})
            )
        )

    assert upstream.authorizations == [
        "Bearer sk_test_rotated_one",
        "Bearer sk_test_rotated_two",
    ]
