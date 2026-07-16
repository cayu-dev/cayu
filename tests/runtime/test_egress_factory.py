from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from cayu.artifacts import LocalArtifactStore
from cayu.core.events import Event, EventType
from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    CredentialMode,
    EgressAdapterRegistry,
    EgressBinding,
    EgressCapabilityClaim,
    EgressCapabilityEvidence,
    HttpEgressPolicy,
    SandboxEgressAdapter,
    TransparentEgressBroker,
    VirtualCredentialError,
)
from cayu.environments import EFSAccessPointBinding, EnvironmentFactoryRequest
from cayu.environments.bindings import BoundWorkspace, WorkspaceBinding
from cayu.runners.base import ExecCommand, ExecResult, Runner
from cayu.vaults import SecretRef, StaticVault

pytest.importorskip("cryptography")

from cayu.egress.docker_adapter import GUEST_CA_PATH
from cayu.runtime.egress import (
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
)

REAL_SECRET = "sk_test_51FactoryRealSecret"
POLICY_NAME = "provider-example"


class _FakeDocker:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        self.calls.append(list(argv))
        return 0, ""


class _FakeDockerRunner(Runner):
    isolation = "docker"
    last_kwargs: dict[str, Any] = {}
    last_instance: _FakeDockerRunner | None = None

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.closed = False

    @classmethod
    async def create(cls, name: str, **kwargs: Any) -> _FakeDockerRunner:
        _FakeDockerRunner.last_kwargs = kwargs
        instance = cls(name, **kwargs)
        _FakeDockerRunner.last_instance = instance
        return instance

    async def exec(self, command: Any, **kwargs: Any) -> ExecResult:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:
        self.closed = True


def _credential_spec() -> VirtualCredentialSpec:
    return VirtualCredentialSpec(
        env_name="STRIPE_SECRET_KEY",
        secret=SecretRef(name="stripe_test_key"),
        destination="api.stripe.com",
        policy_name=POLICY_NAME,
    )


def _virtual_factory(**kwargs: Any) -> VirtualEgressEnvironmentFactory:
    defaults: dict[str, Any] = {
        "resolver": StaticVault({"stripe_test_key": REAL_SECRET}),
        "policies": {POLICY_NAME: _provider_example_policy()},
        "credentials": [_credential_spec()],
    }
    defaults.update(kwargs)
    return VirtualEgressEnvironmentFactory(**defaults)


def _egress_binding(
    runner_kind: str,
    *,
    teardown: Any = None,
    env: dict[str, str] | None = None,
) -> EgressBinding:
    return EgressBinding(
        env=env or {"HTTPS_PROXY": "http://cayu-egress:8080"},
        ca_cert_pem=b"-----BEGIN CERTIFICATE-----\n",
        runner_kind=runner_kind,
        network="net" if runner_kind == "docker" else None,
        sidecar="car" if runner_kind == "docker" else None,
        guest_ca_path=GUEST_CA_PATH,
        teardown=teardown,
    )


class _RecordingAdapter(SandboxEgressAdapter):
    def __init__(
        self,
        runner_kind: str = "docker",
        *,
        order: list[str] | None = None,
        env: dict[str, str] | None = None,
        runner_factory: Any = None,
    ) -> None:
        self.runner_kind = runner_kind
        self.order = order
        self.env = env
        self.runner_factory = runner_factory
        self.prepare_calls: list[dict[str, Any]] = []
        self.captured: dict[str, Any] = {}
        self.torn_down = 0

    async def prepare(self, *, session_id, grants, broker):  # type: ignore[no-untyped-def]
        self.prepare_calls.append(
            {
                "session_id": session_id,
                "grant_count": len(grants),
                "broker": broker,
            }
        )
        self.captured["broker"] = broker
        if grants:
            self.captured["grant"] = grants[0]

        async def teardown() -> None:
            self.torn_down += 1
            if self.order is not None:
                self.order.append("binding_teardown")

        binding = _egress_binding(self.runner_kind, teardown=teardown, env=self.env)
        self.captured["binding"] = binding
        return binding

    async def create_runner(self, request):  # type: ignore[no-untyped-def]
        self.captured["runner_request"] = request
        if self.runner_factory is not None:
            runner = await self.runner_factory(request)
        else:
            runner = await _FakeDockerRunner.create(request.name)
        self.captured["inner_runner"] = runner
        return runner


class _LifecycleRecordingAdapter(_RecordingAdapter):
    def __init__(self) -> None:
        super().__init__("lambda-microvm")
        self.finalize_calls: list[str | None] = []

    def reconnect_metadata(self, runner: Runner) -> dict[str, Any]:
        return {"microvm_id": "mvm-123", "endpoint": "mvm.internal"}

    async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
        self.finalize_calls.append(outcome)
        await runner.close()


class _CapabilityRecordingAdapter(_RecordingAdapter):
    def capability_evidence(self, runner: Runner) -> EgressCapabilityEvidence:
        return EgressCapabilityEvidence(
            adapter="lambda-microvm",
            claims=(
                EgressCapabilityClaim(
                    capability="proxy_reachability",
                    state="verified",
                    proof_source="agent_preflight",
                    observation="reachable",
                ),
                EgressCapabilityClaim(
                    capability="direct_public_egress",
                    state="verified",
                    proof_source="agent_preflight",
                    observation="denied",
                ),
                EgressCapabilityClaim(
                    capability="metadata_isolation",
                    state="unverified",
                    proof_source="operator_opt_out",
                    observation="not_probed",
                    reason_code="guest_process_boundary_unverified",
                    remediation_code="supply_enforceable_guest_boundary",
                ),
            ),
        )

    def configuration_metadata(self) -> dict[str, Any]:
        return {"metadata_isolation_mode": "unverified"}


class _RetryingLifecycleAdapter(_RecordingAdapter):
    def __init__(self) -> None:
        super().__init__("lambda-microvm")
        self.finalize_calls = 0

    async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
        self.finalize_calls += 1
        if self.finalize_calls == 1:
            raise RuntimeError("suspend failed")
        await runner.close()


def _factory(emitter: Any) -> VirtualEgressEnvironmentFactory:
    from cayu.egress.docker_adapter import DockerEgressAdapter

    adapter = DockerEgressAdapter(docker_exec=_FakeDocker(), proxy_host="127.0.0.1")
    return _virtual_factory(
        adapter=adapter,
        event_emitter=emitter,
    )


def _provider_example_policy() -> HttpEgressPolicy:
    return HttpEgressPolicy(
        name=POLICY_NAME,
        allowed_hosts=["api.stripe.com"],
        allowed_endpoints=[("POST", "/v1/customers")],
    )


def _capturing_event_factory(
    events: list[Event],
) -> tuple[VirtualEgressEnvironmentFactory, dict[str, Any]]:
    adapter = _RecordingAdapter("fake")

    async def emitter(event: Event) -> Event:
        events.append(event)
        return event

    class _AllowedUpstream:
        async def send(self, request: CapturedRequest) -> CapturedResponse:
            return CapturedResponse(status_code=200, body=b"{}")

    return (
        _virtual_factory(
            adapter=adapter,
            event_emitter=emitter,
            upstream=_AllowedUpstream(),
        ),
        adapter.captured,
    )


def _broker_request(presented_value: str, path: str) -> CapturedRequest:
    return CapturedRequest(
        method="POST",
        host="api.stripe.com",
        path=path,
        headers={"Authorization": f"Bearer {presented_value}"},
    )


def test_factory_wires_runner_grants_and_events(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cayu.egress.docker_adapter.DockerRunner", _FakeDockerRunner)
    events: list[Event] = []

    async def emitter(event: Event) -> Event:
        events.append(event)
        return event

    async def run() -> tuple[Any, list[Event]]:
        factory = _factory(emitter)
        request = EnvironmentFactoryRequest(
            session_id="sess_1", agent_name="agent", environment_name="egress-env"
        )
        result = await factory.create(request)
        # Drive the session-end teardown hook.
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None
        assert runner is not None
        bound = await binding.bind(None, runner, session_id="sess_1")
        await binding.finalize(bound, outcome="completed")
        return result, events

    result, events = asyncio.run(run())

    runner = result.environment.runner
    assert runner is not None
    inner_runner = _FakeDockerRunner.last_instance
    assert inner_runner is not None
    assert result.environment.vault is None  # real vault is broker-side only
    # Runner is created in virtual_egress mode, on the enforced network, with the
    # virtual credential + proxy overlay and the CA mounted.
    assert inner_runner.kwargs["credential_mode"] is CredentialMode.VIRTUAL_EGRESS
    assert inner_runner.kwargs["network"].startswith("cayu-egress-net-")
    overlay = inner_runner.kwargs["env_overlay"]
    assert overlay["STRIPE_SECRET_KEY"].startswith("sk_test_cayu_vc_")
    assert overlay["HTTPS_PROXY"].startswith("http://cayu-egress-")
    assert REAL_SECRET not in str(overlay)
    assert inner_runner.kwargs["ca_mount"][1] == "/etc/cayu/ca.pem"
    assert runner.closed is True  # finalize closed the sandbox

    types = [e.type for e in events]
    assert EventType.CREDENTIAL_MODE_SELECTED in types
    assert EventType.EGRESS_GRANT_MINTED in types
    assert EventType.EGRESS_GRANT_REVOKED in types
    # No real secret in any emitted payload.
    for event in events:
        assert event.agent_name == "agent"
        assert REAL_SECRET not in str(event.payload)


def test_factory_requires_a_credential() -> None:
    with pytest.raises(ValueError, match="at least one credential"):
        VirtualEgressEnvironmentFactory(
            resolver=StaticVault({}),
            policies={},
            credentials=[],
        )


def test_factory_rejects_duplicate_credential_env_names() -> None:
    with pytest.raises(ValueError, match="env_name values must be unique"):
        VirtualEgressEnvironmentFactory(
            resolver=StaticVault({"stripe_test_key": REAL_SECRET}),
            policies={POLICY_NAME: _provider_example_policy()},
            credentials=[
                VirtualCredentialSpec(
                    env_name="STRIPE_SECRET_KEY",
                    secret=SecretRef(name="stripe_test_key"),
                    destination="api.stripe.com",
                    policy_name=POLICY_NAME,
                ),
                VirtualCredentialSpec(
                    env_name="STRIPE_SECRET_KEY",
                    secret=SecretRef(name="stripe_test_key"),
                    destination="api.stripe.com",
                    policy_name=POLICY_NAME,
                ),
            ],
        )


def test_virtual_credential_spec_rejects_unsupported_credential_kind() -> None:
    credential_kind: Any = "mystery_kind"

    with pytest.raises(ValueError, match="Unsupported credential kind"):
        VirtualCredentialSpec(
            env_name="API_KEY",
            secret=SecretRef(name="api_key"),
            destination="api.example.com",
            policy_name=POLICY_NAME,
            credential_kind=credential_kind,
        )


def test_factory_resolves_adapter_from_registry_and_uses_adapter_runner() -> None:
    class _CreatingAdapter(_RecordingAdapter):
        def __init__(self) -> None:
            super().__init__("fake", env={"HTTPS_PROXY": "http://fake-egress:8080"})
            self.runner_requests: list[Any] = []

        async def create_runner(self, runner_request):  # type: ignore[no-untyped-def]
            self.runner_requests.append(runner_request)
            return _FakeDockerRunner(
                runner_request.name,
                credential_mode=CredentialMode.VIRTUAL_EGRESS,
                env_overlay=dict(runner_request.env_overlay),
            )

    async def run() -> tuple[Any, _CreatingAdapter, Any]:
        adapter = _CreatingAdapter()
        registry = EgressAdapterRegistry()
        registry.register(adapter)

        factory = _virtual_factory(
            adapter_registry=registry,
            runner_kind="fake",
        )
        request = EnvironmentFactoryRequest(
            session_id="sess_registry",
            agent_name="agent",
            environment_name="egress-env",
        )
        result = await factory.create(request)
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return result, adapter, adapter.runner_requests[0]

    result, adapter, runner_request = asyncio.run(run())

    assert result.environment.spec.metadata["kind"] == "fake"
    assert len(adapter.prepare_calls) == 1
    assert adapter.prepare_calls[0]["session_id"] == "sess_registry"
    assert adapter.prepare_calls[0]["grant_count"] == 1
    assert adapter.torn_down == 1
    assert runner_request.runner_kind == "fake"
    assert runner_request.env_overlay["HTTPS_PROXY"] == "http://fake-egress:8080"
    assert runner_request.env_overlay["STRIPE_SECRET_KEY"].startswith("sk_test_cayu_vc_")


def test_factory_passes_and_returns_adapter_reconnect_metadata() -> None:
    adapter = _LifecycleRecordingAdapter()

    async def run() -> tuple[Any, Any]:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_resume",
                agent_name="agent",
                environment_name="egress-env",
                reconnect_metadata={"microvm_id": "mvm-old", "endpoint": "old.internal"},
            )
        )
        request = adapter.captured["runner_request"]
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return result, request

    result, runner_request = asyncio.run(run())

    assert runner_request.session_id == "sess_resume"
    assert runner_request.parent_session_id is None
    assert runner_request.reconnect_metadata == {
        "microvm_id": "mvm-old",
        "endpoint": "old.internal",
    }
    assert result.reconnect_metadata == {
        "microvm_id": "mvm-123",
        "endpoint": "mvm.internal",
    }
    assert adapter.finalize_calls == [None]


def test_factory_exposes_typed_capability_evidence_separately_from_configuration() -> None:
    adapter = _CapabilityRecordingAdapter("lambda-microvm")

    async def run() -> Any:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_capabilities",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return result

    result = asyncio.run(run())

    expected_evidence = {
        "schema": "cayu.egress_capabilities.v1",
        "adapter": "lambda-microvm",
        "claims": [
            {
                "capability": "direct_public_egress",
                "state": "verified",
                "proof_source": "agent_preflight",
                "observation": "denied",
            },
            {
                "capability": "metadata_isolation",
                "state": "unverified",
                "proof_source": "operator_opt_out",
                "observation": "not_probed",
                "reason_code": "guest_process_boundary_unverified",
                "remediation_code": "supply_enforceable_guest_boundary",
            },
            {
                "capability": "proxy_reachability",
                "state": "verified",
                "proof_source": "agent_preflight",
                "observation": "reachable",
            },
        ],
    }
    expected_configuration = {"metadata_isolation_mode": "unverified"}
    assert result.environment.spec.metadata["egress_capabilities"] == expected_evidence
    assert result.metadata["egress_capabilities"] == expected_evidence
    assert result.environment.spec.metadata["egress_configuration"] == expected_configuration
    assert result.metadata["egress_configuration"] == expected_configuration


def test_factory_exposes_explicit_unclaimed_evidence_for_adapter_without_claims() -> None:
    async def run() -> Any:
        result = await _virtual_factory(adapter=_RecordingAdapter("docker")).create(
            EnvironmentFactoryRequest(
                session_id="sess_unclaimed_capabilities",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return result

    result = asyncio.run(run())

    assert result.metadata["egress_capabilities"] == {
        "schema": "cayu.egress_capabilities.v1",
        "adapter": "docker",
        "claims": [],
        "unclaimed_reason_code": "adapter_capabilities_unclaimed",
    }


def test_factory_rejects_untyped_capability_evidence_and_cleans_up() -> None:
    class _MalformedEvidenceAdapter(_RecordingAdapter):
        def capability_evidence(self, runner: Runner) -> Any:
            return {"metadata_isolation": "verified"}

    adapter = _MalformedEvidenceAdapter("lambda-microvm")

    async def run() -> None:
        with pytest.raises(TypeError, match="EgressCapabilityEvidence"):
            await _virtual_factory(adapter=adapter).create(
                EnvironmentFactoryRequest(
                    session_id="sess_malformed_capabilities",
                    agent_name="agent",
                    environment_name="egress-env",
                )
            )

    asyncio.run(run())

    inner: Runner = adapter.captured["inner_runner"]
    assert inner.closed is True
    assert adapter.torn_down == 1


def test_factory_attaches_durable_artifact_store(tmp_path) -> None:
    adapter = _RecordingAdapter("fake")
    artifact_store = LocalArtifactStore(tmp_path / "artifacts")

    async def run() -> Any:
        result = await _virtual_factory(
            adapter=adapter,
            artifact_store=artifact_store,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_artifacts",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return result

    result = asyncio.run(run())

    assert result.environment.artifact_store is artifact_store
    assert result.environment.vault is None


def test_factory_finalizes_adapter_runner_with_session_outcome() -> None:
    adapter = _LifecycleRecordingAdapter()

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_interrupt",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None
        assert runner is not None
        bound = await binding.bind(None, runner, session_id="sess_interrupt")
        await binding.finalize(bound, outcome="interrupted")

    asyncio.run(run())

    assert adapter.finalize_calls == ["interrupted"]


def test_create_tears_down_egress_when_runner_start_fails() -> None:
    # If DockerRunner.create fails after adapter.prepare succeeded, the prepared
    # egress binding (proxy + network + sidecar) must be torn down, not leaked.
    async def _boom_create(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("image pull failed")

    adapter = _RecordingAdapter(runner_factory=_boom_create)

    async def run() -> None:
        factory = _virtual_factory(adapter=adapter)
        request = EnvironmentFactoryRequest(
            session_id="sess_fail", agent_name="agent", environment_name="egress-env"
        )
        with pytest.raises(RuntimeError, match="image pull failed"):
            await factory.create(request)

    asyncio.run(run())
    assert adapter.torn_down == 1  # the prepared binding was torn down


def test_create_propagates_adapter_prepare_failure_without_binding_cleanup_error() -> None:
    class _FailingPrepareAdapter(SandboxEgressAdapter):
        runner_kind = "docker"

        async def prepare(self, *, session_id, grants, broker):  # type: ignore[no-untyped-def]
            raise RuntimeError("prepare failed")

        async def create_runner(self, request):  # type: ignore[no-untyped-def]
            raise AssertionError("runner creation should not run")

    async def run() -> None:
        factory = _virtual_factory(adapter=_FailingPrepareAdapter())
        request = EnvironmentFactoryRequest(
            session_id="sess_prepare_fail",
            agent_name="agent",
            environment_name="egress-env",
        )
        with pytest.raises(RuntimeError, match="prepare failed"):
            await factory.create(request)

    asyncio.run(run())


def test_bind_failure_cleans_up_egress_resources() -> None:
    adapter = _RecordingAdapter()

    class _FailingBindBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("bind failed")

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    async def run() -> Any:
        factory = _virtual_factory(
            adapter=adapter,
            inner_binding=_FailingBindBinding(),
        )
        request = EnvironmentFactoryRequest(
            session_id="sess_bind_fail",
            agent_name="agent",
            environment_name="egress-env",
        )
        result = await factory.create(request)
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None
        assert runner is not None
        with pytest.raises(RuntimeError, match="bind failed"):
            await binding.bind(None, runner, session_id="sess_bind_fail")
        return runner

    runner = asyncio.run(run())

    assert runner.closed is True
    assert adapter.torn_down == 1


def test_runner_close_before_bind_cleans_up_egress_resources() -> None:
    adapter = _RecordingAdapter()

    async def run() -> Any:
        factory = _virtual_factory(adapter=adapter)
        request = EnvironmentFactoryRequest(
            session_id="sess_abandoned",
            agent_name="agent",
            environment_name="egress-env",
        )
        result = await factory.create(request)
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return runner

    runner = asyncio.run(run())

    assert runner.closed is True
    assert adapter.torn_down == 1


def test_runner_close_reports_binding_teardown_failure_and_retries() -> None:
    adapter = _RecordingAdapter()

    async def run() -> tuple[Runner, int]:
        factory = _virtual_factory(adapter=adapter)
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="sess_retry_cleanup",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        binding: EgressBinding = adapter.captured["binding"]
        original_teardown = binding.teardown
        assert original_teardown is not None
        calls = 0

        async def flaky_teardown() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("egress resource still stopping")
            await original_teardown()

        binding.teardown = flaky_teardown
        with pytest.raises(RuntimeError, match="binding: RuntimeError"):
            await runner.close()
        assert runner._closed is False
        await runner.close()
        return runner, calls

    runner, calls = asyncio.run(run())
    assert runner._closed is True
    assert calls == 2


def test_runner_close_retries_when_inner_runner_close_is_cancelled() -> None:
    async def run() -> tuple[bool, int, int]:
        class _SelfCancellingCloseRunner(_FakeDockerRunner):
            close_calls = 0

            async def close(self) -> None:
                self.close_calls += 1
                if self.close_calls == 1:
                    raise asyncio.CancelledError()
                await super().close()

        inner = _SelfCancellingCloseRunner("runner")
        adapter = _RecordingAdapter(runner_factory=lambda _request: asyncio.sleep(0, result=inner))
        factory = _virtual_factory(adapter=adapter)
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="sess_cancelled_runner_cleanup",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None

        with pytest.raises(asyncio.CancelledError):
            await managed.close()
        assert managed._closed is False
        assert inner.closed is False
        assert adapter.torn_down == 0

        await managed.close()
        return inner.closed, inner.close_calls, adapter.torn_down

    inner_closed, close_calls, teardown_calls = asyncio.run(run())

    assert inner_closed is True
    assert close_calls == 2
    assert teardown_calls == 1


def test_runner_close_bounds_hanging_runner_phase_and_resumes_same_cleanup_task() -> None:
    async def run() -> tuple[bool, int]:
        started = asyncio.Event()
        finish = asyncio.Event()

        class _HangingCloseRunner(_FakeDockerRunner):
            async def close(self) -> None:
                started.set()
                await finish.wait()
                await super().close()

        adapter = _RecordingAdapter(
            runner_factory=lambda _request: asyncio.sleep(0, result=_HangingCloseRunner("runner"))
        )
        factory = _virtual_factory(adapter=adapter)
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="sess_hanging_runner_cleanup",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None
        managed._teardown_timeout_s = 0.01
        managed._authority_revoker.teardown_timeout_s = 0.01
        inner: _HangingCloseRunner = adapter.captured["inner_runner"]

        with pytest.raises(TimeoutError, match="runner cleanup did not complete"):
            await managed.close()
        assert started.is_set()
        assert managed._closed is False
        assert adapter.torn_down == 0

        finish.set()
        await managed.close()
        return inner.closed, adapter.torn_down

    inner_closed, teardown_calls = asyncio.run(run())

    assert inner_closed is True
    assert teardown_calls == 1


def test_runner_close_revokes_grants_before_closing_inner_runner() -> None:
    order: list[str] = []
    adapter = _RecordingAdapter("fake", order=order)

    class _InspectingRunner(Runner):
        isolation = "fake"
        default_cwd = "/"

        async def exec(self, command: Any, **kwargs: Any) -> ExecResult:  # pragma: no cover
            raise NotImplementedError

        async def close(self) -> None:
            broker = adapter.captured["broker"]
            grant = adapter.captured["grant"]
            with pytest.raises(VirtualCredentialError):
                broker.registry.lookup(grant.presented_value)
            order.append("inner_runner_close")

    async def runner_factory(_request: Any) -> Runner:
        return _InspectingRunner()

    adapter.runner_factory = runner_factory

    async def run() -> None:
        factory = _virtual_factory(adapter=adapter)
        request = EnvironmentFactoryRequest(
            session_id="sess_revoke_first",
            agent_name="agent",
            environment_name="egress-env",
        )
        result = await factory.create(request)
        runner = result.environment.runner
        assert runner is not None
        await runner.close()

    asyncio.run(run())

    assert order == ["inner_runner_close", "binding_teardown"]


def test_runner_close_defers_cancellation_until_grant_drain() -> None:
    async def run() -> tuple[_FakeDockerRunner, dict[str, int]]:
        adapter = _RecordingAdapter("fake")

        async def runner_factory(_request: Any) -> Runner:
            runner = _FakeDockerRunner("runner")
            adapter.captured["inner_runner"] = runner
            return runner

        adapter.runner_factory = runner_factory
        factory = _virtual_factory(adapter=adapter)
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="sess_1",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None
        broker: TransparentEgressBroker = adapter.captured["broker"]
        grant = adapter.captured["grant"]
        inner_runner: _FakeDockerRunner = adapter.captured["inner_runner"]
        lease = broker.registry.acquire(grant.presented_value)

        close_task = asyncio.create_task(managed.close())
        await asyncio.sleep(0)
        assert close_task.done() is False

        close_task.cancel()
        await asyncio.sleep(0)
        assert close_task.done() is False
        assert inner_runner.closed is False

        lease.close()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        return inner_runner, {"count": adapter.torn_down}

    runner, teardown_calls = asyncio.run(run())

    assert runner.closed is True
    assert teardown_calls["count"] == 1


def test_runner_close_bounds_grant_drain_and_retries_without_releasing_resources() -> None:
    async def run() -> tuple[_FakeDockerRunner, int]:
        adapter = _RecordingAdapter("fake")
        factory = _virtual_factory(adapter=adapter)
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="sess_bounded_revoke",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None
        managed._teardown_timeout_s = 0.01
        managed._authority_revoker.teardown_timeout_s = 0.01
        broker: TransparentEgressBroker = adapter.captured["broker"]
        grant = adapter.captured["grant"]
        inner_runner: _FakeDockerRunner = adapter.captured["inner_runner"]
        lease = broker.registry.acquire(grant.presented_value)

        with pytest.raises(TimeoutError, match="grant revocation did not complete"):
            await managed.close()
        assert managed._closed is False
        assert inner_runner.closed is False
        assert adapter.torn_down == 0

        lease.close()
        await managed.close()
        return inner_runner, adapter.torn_down

    runner, teardown_calls = asyncio.run(run())

    assert runner.closed is True
    assert teardown_calls == 1


def test_create_cleans_up_when_grant_event_emit_is_cancelled() -> None:
    adapter = _RecordingAdapter()

    async def emitter(event: Event) -> Event:
        if event.type == EventType.EGRESS_GRANT_MINTED:
            raise asyncio.CancelledError()
        return event

    async def run() -> None:
        factory = _virtual_factory(
            adapter=adapter,
            event_emitter=emitter,
        )
        request = EnvironmentFactoryRequest(
            session_id="sess_emit_cancel",
            agent_name="agent",
            environment_name="egress-env",
        )
        with pytest.raises(asyncio.CancelledError):
            await factory.create(request)

    asyncio.run(run())

    assert _FakeDockerRunner.last_instance is not None
    assert _FakeDockerRunner.last_instance.closed is True
    assert adapter.torn_down == 1


def test_finalize_revokes_grants_before_workspace_sync_then_finalizes_runner() -> None:
    order: list[str] = []

    class _OrderingAdapter(_RecordingAdapter):
        async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
            order.append("runner_finalize")
            await runner.close()

    adapter = _OrderingAdapter()

    class _InspectingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            broker = adapter.captured["broker"]
            grant = adapter.captured["grant"]
            with pytest.raises(VirtualCredentialError):
                broker.registry.lookup(grant.presented_value)
            order.append("inner_finalize")
            return None

    async def run() -> Any:
        factory = _virtual_factory(
            adapter=adapter,
            inner_binding=_InspectingBinding(),
        )
        request = EnvironmentFactoryRequest(
            session_id="sess_finalize_revoke_first",
            agent_name="agent",
            environment_name="egress-env",
        )
        result = await factory.create(request)
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None
        assert runner is not None
        bound = await binding.bind(None, runner, session_id="sess_finalize_revoke_first")
        await binding.finalize(bound, outcome="completed")
        return runner

    runner = asyncio.run(run())

    assert order == ["inner_finalize", "runner_finalize"]
    assert runner.closed is True
    assert adapter.torn_down == 1


def test_factory_preserves_trusted_execution_for_aws_workspace_lifecycle() -> None:
    class _ProfileRecordingRunner(Runner):
        isolation = "lambda-microvm"
        default_cwd = "/workspace"

        def __init__(self) -> None:
            self.calls: list[tuple[str, list[str]]] = []
            self.mountpoint_checks = 0

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            return self._record("agent", command)

        async def exec_system(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            return self._record("trusted", command)

        def _record(self, profile: str, command: ExecCommand) -> ExecResult:
            argv = list(command.argv or [])
            self.calls.append((profile, argv))
            if argv[:2] == ["mountpoint", "-q"]:
                self.mountpoint_checks += 1
                return ExecResult(exit_code=1 if self.mountpoint_checks == 1 else 0)
            return ExecResult()

    inner = _ProfileRecordingRunner()
    adapter = _RecordingAdapter(
        "lambda-microvm",
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )
    workspace_binding = EFSAccessPointBinding(
        file_system_id="fs-1",
        access_point_id="fsap-1",
        mount_target_ip="10.0.0.10",
    )

    async def run() -> None:
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=workspace_binding,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_trusted_workspace",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None
        assert runner is not None
        bound = await binding.bind(None, runner, session_id="sess_trusted_workspace")
        await binding.finalize(bound, outcome="completed")

    asyncio.run(run())

    assert inner.calls
    assert {profile for profile, _command in inner.calls} == {"trusted"}
    assert [command[0] for _profile, command in inner.calls] == [
        "mkdir",
        "mountpoint",
        "mount",
        "mountpoint",
        "sync",
        "mountpoint",
        "umount",
    ]


def test_finalize_surfaces_lifecycle_failure_and_runner_close_retries() -> None:
    adapter = _RetryingLifecycleAdapter()

    async def run() -> Runner:
        factory = _virtual_factory(adapter=adapter)
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_retry",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None
        assert runner is not None
        bound = await binding.bind(None, runner, session_id="sess_finalize_retry")
        with pytest.raises(RuntimeError, match="suspend failed"):
            await binding.finalize(bound, outcome="interrupted")
        assert runner.closed is False
        await runner.close()
        return runner

    runner = asyncio.run(run())

    assert runner.closed is True
    assert adapter.finalize_calls == 2


def test_finalize_cleans_up_egress_when_inner_finalize_fails() -> None:
    adapter = _RecordingAdapter()

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise RuntimeError("sync-back failed")

    async def run() -> Any:
        factory = _virtual_factory(
            adapter=adapter,
            inner_binding=_FailingBinding(),
        )
        request = EnvironmentFactoryRequest(
            session_id="sess_finalize_fail",
            agent_name="agent",
            environment_name="egress-env",
        )
        result = await factory.create(request)
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None
        assert runner is not None
        bound = await binding.bind(None, runner, session_id="sess_finalize_fail")
        with pytest.raises(RuntimeError, match="sync-back failed"):
            await binding.finalize(bound, outcome="failed")
        return runner

    runner = asyncio.run(run())

    assert runner.closed is True
    assert adapter.torn_down == 1


def test_factory_emits_authorized_and_denied_request_events() -> None:
    async def run() -> list[Event]:
        events: list[Event] = []
        factory, captured = _capturing_event_factory(events)
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="sess_1",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None
        assert runner is not None
        bound = await binding.bind(None, runner, session_id="sess_1")
        broker: TransparentEgressBroker = captured["broker"]
        grant = captured["grant"]

        await broker.handle_request(_broker_request(grant.presented_value, "/v1/customers"))
        await broker.handle_request(_broker_request(grant.presented_value, "/v1/payouts"))
        await binding.finalize(bound, outcome="completed")
        return events

    events = asyncio.run(run())
    types = {e.type for e in events}
    assert EventType.EGRESS_REQUEST_AUTHORIZED in types
    assert EventType.EGRESS_REQUEST_DENIED in types
    assert {e.agent_name for e in events} == {"agent"}


def test_factory_drains_request_audit_before_revoked_events() -> None:
    async def run() -> list[Event]:
        events: list[Event] = []
        factory, captured = _capturing_event_factory(events)
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="sess_1",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None
        assert runner is not None
        bound = await binding.bind(None, runner, session_id="sess_1")
        broker: TransparentEgressBroker = captured["broker"]
        grant = captured["grant"]

        await broker.handle_request(_broker_request(grant.presented_value, "/v1/customers"))
        await binding.finalize(bound, outcome="completed")
        return events

    events = asyncio.run(run())
    types = [event.type for event in events]
    assert types.index(EventType.EGRESS_REQUEST_AUTHORIZED) < types.index(
        EventType.EGRESS_GRANT_REVOKED
    )
