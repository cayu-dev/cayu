from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from cayu import (
    ApprovedEgressDestination,
    Event,
    HttpEgressPolicy,
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
)
from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    EgressBinding,
    SandboxEgressAdapter,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.environments import EnvironmentFactoryRequest
from cayu.runners import ExecCommand, ExecResult, Runner
from cayu.vaults import ResolvedSecret, SecretRef, StaticVault


class _RecordingResolver:
    def __init__(self) -> None:
        self.calls = 0

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        self.calls += 1
        raise AssertionError("credentialless egress must not resolve a secret")


class _RecordingUpstream:
    def __init__(self, response: CapturedResponse | None = None) -> None:
        self.requests: list[CapturedRequest] = []
        self.response = response or CapturedResponse(status_code=200, body=b"ok")

    async def send(self, request: CapturedRequest) -> CapturedResponse:
        self.requests.append(request)
        return self.response


def _public_docs_policy() -> HttpEgressPolicy:
    return HttpEgressPolicy(
        name="public-docs",
        allowed_hosts=["docs.example.com"],
        allowed_endpoints=[("GET", "/sdk/index.json")],
    )


def _approved_docs() -> ApprovedEgressDestination:
    return ApprovedEgressDestination(
        destination="docs.example.com",
        policy_name="public-docs",
        protocol="https",
        port=443,
    )


def _credentialless_broker(
    *,
    upstream: _RecordingUpstream,
    resolver: _RecordingResolver | None = None,
) -> tuple[TransparentEgressBroker, _RecordingResolver, list[Any]]:
    actual_resolver = resolver or _RecordingResolver()
    decisions: list[Any] = []
    return (
        TransparentEgressBroker(
            registry=VirtualCredentialRegistry(),
            resolver=actual_resolver,
            policies={"public-docs": _public_docs_policy()},
            approved_destinations=[_approved_docs()],
            upstream=upstream,
            audit=decisions.append,
        ),
        actual_resolver,
        decisions,
    )


def test_credentialless_request_is_forwarded_unchanged_without_secret_resolution() -> None:
    upstream = _RecordingUpstream()
    broker, resolver, decisions = _credentialless_broker(upstream=upstream)
    request = CapturedRequest(
        method="GET",
        host="docs.example.com",
        path="/sdk/index.json",
        headers={"Accept": "application/json", "Authorization": "Basic caller-owned"},
    )

    response = asyncio.run(broker.handle_request(request))

    assert response.status_code == 200
    assert resolver.calls == 0
    assert upstream.requests == [request]
    assert upstream.requests[0].headers["Authorization"] == "Basic caller-owned"
    assert decisions[-1].authorization_kind == "credentialless"
    assert decisions[-1].grant_id is None
    assert decisions[-1].policy_name == "public-docs"


@pytest.mark.parametrize(
    "captured_request",
    [
        CapturedRequest(method="GET", host="evil.example.com", path="/sdk/index.json"),
        CapturedRequest(
            method="GET",
            host="docs.example.com",
            path="/sdk/index.json",
            port=8443,
        ),
        CapturedRequest(method="GET", host="169.254.169.254", path="/latest/meta-data/"),
        CapturedRequest(method="GET", host="203.0.113.10", path="/sdk/index.json"),
    ],
)
def test_credentialless_route_denies_unapproved_destination_port_and_direct_ip(
    captured_request: CapturedRequest,
) -> None:
    upstream = _RecordingUpstream()
    broker, resolver, decisions = _credentialless_broker(upstream=upstream)

    response = asyncio.run(broker.handle_request(captured_request))

    assert response.status_code == 403
    assert resolver.calls == 0
    assert upstream.requests == []
    assert decisions[-1].allowed is False
    assert decisions[-1].authorization_kind == "credentialless"


def test_redirect_target_is_reauthorized_instead_of_inherited() -> None:
    upstream = _RecordingUpstream(
        CapturedResponse(
            status_code=302,
            headers={"Location": "https://evil.example.com/payload"},
        )
    )
    broker, resolver, _decisions = _credentialless_broker(upstream=upstream)

    first = asyncio.run(
        broker.handle_request(
            CapturedRequest(method="GET", host="docs.example.com", path="/sdk/index.json")
        )
    )
    redirected = asyncio.run(
        broker.handle_request(
            CapturedRequest(method="GET", host="evil.example.com", path="/payload")
        )
    )

    assert first.status_code == 302
    assert redirected.status_code == 403
    assert len(upstream.requests) == 1
    assert resolver.calls == 0


def test_explicitly_approved_redirect_target_uses_its_own_policy() -> None:
    upstream = _RecordingUpstream()
    broker = TransparentEgressBroker(
        registry=VirtualCredentialRegistry(),
        policies={
            "public-docs": _public_docs_policy(),
            "public-cdn": HttpEgressPolicy(
                name="public-cdn",
                allowed_hosts=["cdn.example.com"],
                allowed_endpoints=[("GET", "/sdk/index.json")],
            ),
        },
        approved_destinations=[
            _approved_docs(),
            ApprovedEgressDestination(
                destination="cdn.example.com",
                policy_name="public-cdn",
            ),
        ],
        upstream=upstream,
    )

    response = asyncio.run(
        broker.handle_request(
            CapturedRequest(method="GET", host="cdn.example.com", path="/sdk/index.json")
        )
    )

    assert response.status_code == 200
    assert upstream.requests[0].host == "cdn.example.com"


def test_invalid_virtual_credential_cannot_fall_back_to_credentialless_route() -> None:
    upstream = _RecordingUpstream()
    broker, resolver, decisions = _credentialless_broker(upstream=upstream)

    response = asyncio.run(
        broker.handle_request(
            CapturedRequest(
                method="GET",
                host="docs.example.com",
                path="/sdk/index.json",
                headers={"Authorization": "Bearer cayu_vc_invalid"},
            )
        )
    )

    assert response.status_code == 403
    assert upstream.requests == []
    assert resolver.calls == 0
    assert decisions[-1].authorization_kind == "virtual_credential"


class _FakeRunner(Runner):
    isolation = "fake"
    default_cwd = "/workspace"

    async def exec(
        self,
        command: ExecCommand,
        **kwargs: Any,
    ) -> ExecResult:  # pragma: no cover - not exercised by the factory seam
        raise NotImplementedError

    async def close(self) -> None:
        self._closed = True


class _RecordingAdapter(SandboxEgressAdapter):
    runner_kind = "fake"

    def __init__(self) -> None:
        self.prepare_calls: list[dict[str, Any]] = []
        self.runner_requests: list[Any] = []

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[Any],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        self.prepare_calls.append(
            {"session_id": session_id, "grants": tuple(grants), "broker": broker}
        )
        return EgressBinding(
            env={"HTTPS_PROXY": "http://proxy.internal:8080"},
            ca_cert_pem=b"test-ca",
            runner_kind=self.runner_kind,
            guest_ca_path="/etc/cayu/ca.pem",
        )

    async def create_runner(self, request: Any) -> Runner:
        self.runner_requests.append(request)
        return _FakeRunner()


def _factory_request() -> EnvironmentFactoryRequest:
    return EnvironmentFactoryRequest(
        session_id="session-a",
        agent_name="builder",
        environment_name="sandbox",
    )


def test_factory_supports_credentialless_only_without_a_resolver_or_fake_environment_value() -> (
    None
):
    adapter = _RecordingAdapter()
    factory = VirtualEgressEnvironmentFactory(
        policies={"public-docs": _public_docs_policy()},
        approved_destinations=[_approved_docs()],
        credentials=[],
        adapter=adapter,
    )

    result = asyncio.run(factory.create(_factory_request()))

    assert adapter.prepare_calls[0]["grants"] == ()
    runner_request = adapter.runner_requests[0]
    assert runner_request.egress_destinations == ("docs.example.com",)
    assert dict(runner_request.env_overlay) == {"HTTPS_PROXY": "http://proxy.internal:8080"}
    assert result.environment.spec.metadata["credential_mode"] == "virtual_egress"
    asyncio.run(result.environment.runner.close())  # type: ignore[union-attr]


def test_factory_supports_mixed_credentialed_and_credentialless_destinations() -> None:
    adapter = _RecordingAdapter()
    factory = VirtualEgressEnvironmentFactory(
        resolver=StaticVault({"api_key": "test-secret"}),
        policies={
            "public-docs": _public_docs_policy(),
            "private-api": HttpEgressPolicy(
                name="private-api",
                allowed_hosts=["api.example.com"],
                allowed_endpoints=[("POST", "/v1/jobs")],
            ),
        },
        approved_destinations=[_approved_docs()],
        credentials=[
            VirtualCredentialSpec(
                env_name="API_KEY",
                secret=SecretRef(name="api_key"),
                destination="api.example.com",
                policy_name="private-api",
                credential_kind="opaque_bearer",
            )
        ],
        adapter=adapter,
        require_test_mode_credentials=False,
    )

    result = asyncio.run(factory.create(_factory_request()))

    assert len(adapter.prepare_calls[0]["grants"]) == 1
    runner_request = adapter.runner_requests[0]
    assert runner_request.egress_destinations == (
        "api.example.com",
        "docs.example.com",
    )
    assert dict(runner_request.env_overlay)["API_KEY"].startswith("cayu_vc_")
    asyncio.run(result.environment.runner.close())  # type: ignore[union-attr]


def test_factory_audit_event_distinguishes_credentialless_authorization() -> None:
    async def run() -> tuple[list[Event], _RecordingResolver]:
        adapter = _RecordingAdapter()
        resolver = _RecordingResolver()
        upstream = _RecordingUpstream()
        events: list[Event] = []

        async def emit(event: Event) -> Event:
            events.append(event)
            return event

        factory = VirtualEgressEnvironmentFactory(
            policies={"public-docs": _public_docs_policy()},
            approved_destinations=[_approved_docs()],
            credentials=[],
            resolver=resolver,
            adapter=adapter,
            upstream=upstream,
            event_emitter=emit,
        )
        result = await factory.create(_factory_request())
        broker = adapter.prepare_calls[0]["broker"]
        response = await broker.handle_request(
            CapturedRequest(
                method="GET",
                host="docs.example.com",
                path="/sdk/index.json",
            )
        )
        assert response.status_code == 200
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return events, resolver

    events, resolver = asyncio.run(run())

    authorized = [event for event in events if event.type.value == "egress.request.authorized"]
    assert len(authorized) == 1
    assert authorized[0].payload["authorization_kind"] == "credentialless"
    assert authorized[0].payload["grant_id"] is None
    assert resolver.calls == 0


def test_factory_finalization_revokes_credentialless_authority_before_teardown() -> None:
    async def run() -> tuple[int, int]:
        adapter = _RecordingAdapter()
        upstream = _RecordingUpstream()
        factory = VirtualEgressEnvironmentFactory(
            policies={"public-docs": _public_docs_policy()},
            approved_destinations=[_approved_docs()],
            adapter=adapter,
            upstream=upstream,
        )
        result = await factory.create(_factory_request())
        broker = adapter.prepare_calls[0]["broker"]
        allowed = await broker.handle_request(
            CapturedRequest(
                method="GET",
                host="docs.example.com",
                path="/sdk/index.json",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        denied = await broker.handle_request(
            CapturedRequest(
                method="GET",
                host="docs.example.com",
                path="/sdk/index.json",
            )
        )
        return allowed.status_code, denied.status_code

    allowed_status, denied_status = asyncio.run(run())

    assert allowed_status == 200
    assert denied_status == 403


def test_factory_finalization_revokes_then_drains_inflight_credentialless_request() -> None:
    class _BlockingUpstream:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0

        async def send(self, request: CapturedRequest) -> CapturedResponse:
            self.calls += 1
            self.started.set()
            await self.release.wait()
            return CapturedResponse(status_code=200, body=b"too late")

    async def run() -> tuple[int, int, bool]:
        adapter = _RecordingAdapter()
        upstream = _BlockingUpstream()
        factory = VirtualEgressEnvironmentFactory(
            policies={"public-docs": _public_docs_policy()},
            approved_destinations=[_approved_docs()],
            adapter=adapter,
            upstream=upstream,
        )
        result = await factory.create(_factory_request())
        broker = adapter.prepare_calls[0]["broker"]
        captured = CapturedRequest(
            method="GET",
            host="docs.example.com",
            path="/sdk/index.json",
        )
        inflight = asyncio.create_task(broker.handle_request(captured))
        await upstream.started.wait()

        runner = result.environment.runner
        assert runner is not None
        close_task = asyncio.create_task(runner.close())
        for _ in range(20):
            if not broker._credentialless_authority_active:
                break
            await asyncio.sleep(0)
        else:
            raise AssertionError("Finalization did not revoke credentialless authority.")
        close_waited = not close_task.done()
        denied = await broker.handle_request(captured)
        assert upstream.calls == 1

        upstream.release.set()
        inflight_response = await inflight
        await close_task
        return inflight_response.status_code, denied.status_code, close_waited

    inflight_status, new_request_status, close_waited = asyncio.run(run())

    assert close_waited is True
    assert inflight_status == 403
    assert new_request_status == 403


def test_factory_broker_reauthorizes_redirect_target_and_denies_escape() -> None:
    async def run() -> tuple[int, int, int]:
        adapter = _RecordingAdapter()
        upstream = _RecordingUpstream(
            CapturedResponse(
                status_code=302,
                headers={"Location": "https://evil.example.com/payload"},
            )
        )
        factory = VirtualEgressEnvironmentFactory(
            policies={"public-docs": _public_docs_policy()},
            approved_destinations=[_approved_docs()],
            adapter=adapter,
            upstream=upstream,
        )
        result = await factory.create(_factory_request())
        broker = adapter.prepare_calls[0]["broker"]
        first = await broker.handle_request(
            CapturedRequest(
                method="GET",
                host="docs.example.com",
                path="/sdk/index.json",
            )
        )
        redirected = await broker.handle_request(
            CapturedRequest(method="GET", host="evil.example.com", path="/payload")
        )
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return first.status_code, redirected.status_code, len(upstream.requests)

    first_status, redirected_status, upstream_count = asyncio.run(run())

    assert first_status == 302
    assert redirected_status == 403
    assert upstream_count == 1


def test_factory_rejects_empty_egress_configuration_before_adapter_allocation() -> None:
    adapter = _RecordingAdapter()

    with pytest.raises(ValueError, match="credential or approved destination"):
        VirtualEgressEnvironmentFactory(
            policies={},
            credentials=[],
            approved_destinations=[],
            adapter=adapter,
        )

    assert adapter.prepare_calls == []


@pytest.mark.parametrize(
    "values",
    [
        {"destination": "https://docs.example.com", "protocol": "https", "port": 443},
        {"destination": "203.0.113.10", "protocol": "https", "port": 443},
        {"destination": "docs.example.com", "protocol": "http", "port": 80},
        {"destination": "docs.example.com", "protocol": "https", "port": 8443},
    ],
)
def test_approved_destination_rejects_urls_ips_and_unsupported_transport(
    values: Mapping[str, Any],
) -> None:
    with pytest.raises(ValueError):
        ApprovedEgressDestination(policy_name="public-docs", **values)
