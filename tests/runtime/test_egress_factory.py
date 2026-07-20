from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from tests.runners.lambda_microvm_harness import (
    ConformanceLambdaClient,
    SupervisorTransport,
)

from cayu.artifacts import LocalArtifactStore
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    CredentialMode,
    EgressAdapterRegistry,
    EgressBinding,
    EgressCapabilityClaim,
    EgressCapabilityEvidence,
    HttpEgressPolicy,
    InvalidEgressReconnectMetadataError,
    SandboxEgressAdapter,
    TransparentEgressBroker,
    UnsupportedEgressAdapter,
    UnsupportedEgressError,
    UnsupportedEgressReconnectError,
    VirtualCredentialError,
)
from cayu.environments import (
    EFSAccessPointBinding,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentSpec,
    ExecutionAdmissionError,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionEvidenceOverride,
    ExecutionRequirements,
)
from cayu.environments.bindings import BoundWorkspace, WorkspaceBinding
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners import LambdaMicroVMRunner
from cayu.runners.base import ExecCommand, ExecResult, Runner
from cayu.runtime import CayuApp, InMemorySessionStore, RunRequest
from cayu.runtime.app import (
    _persist_binding_finalize_failure_event,
    _reconcile_binding_finalize_failure_event,
)
from cayu.runtime.event_sinks import EventSink
from cayu.vaults import REDACTED_SECRET, SecretRedactor, SecretRef, StaticVault

pytest.importorskip("cryptography")

from cayu.egress.adapter import (
    _await_bounded_cleanup_task,
    _raise_primary_with_cleanup_cancellation,
)
from cayu.egress.docker_adapter import GUEST_CA_PATH
from cayu.runtime._binding_cleanup import (
    BINDING_FINALIZE_ERROR_TEXT_MAX_BYTES,
    BindingFinalizeFailure,
    append_binding_finalize_cancellation,
    binding_finalize_failure_payload,
    binding_finalize_fatal_signal,
    record_binding_finalize_failures,
)
from cayu.runtime.egress import (
    VirtualCredentialSpec,
    VirtualEgressEnvironmentFactory,
    _await_cleanup_task,
)

REAL_SECRET = "sk_test_51FactoryRealSecret"
POLICY_NAME = "provider-example"


@pytest.mark.parametrize("bounded", [False, True])
def test_cleanup_wait_preserves_caller_cancellation_and_child_failure(bounded: bool) -> None:
    cleanup_started = asyncio.Event()
    allow_failure = asyncio.Event()
    cleanup_error = RuntimeError("cleanup failed")

    async def cleanup() -> None:
        cleanup_started.set()
        await allow_failure.wait()
        raise cleanup_error

    async def run() -> BaseExceptionGroup:
        child = asyncio.create_task(cleanup())

        async def wait() -> bool:
            if bounded:
                return await _await_bounded_cleanup_task(
                    child,
                    timeout_s=1,
                    timeout_message="cleanup timed out",
                )
            return await _await_cleanup_task(child)

        waiter = asyncio.create_task(wait())
        await cleanup_started.wait()
        waiter.cancel()
        allow_failure.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await waiter
        return exc_info.value

    failure = asyncio.run(run())

    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert failure.exceptions[1] is cleanup_error


def test_bounded_cleanup_preserves_caller_cancellation_when_it_times_out() -> None:
    cleanup_started = asyncio.Event()

    async def cleanup() -> None:
        cleanup_started.set()
        await asyncio.Event().wait()

    async def run() -> BaseExceptionGroup:
        child = asyncio.create_task(cleanup())
        waiter = asyncio.create_task(
            _await_bounded_cleanup_task(
                child,
                timeout_s=0.01,
                timeout_message="cleanup timed out",
            )
        )
        await cleanup_started.wait()
        waiter.cancel()
        try:
            with pytest.raises(BaseExceptionGroup) as exc_info:
                await waiter
            return exc_info.value
        finally:
            child.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await child

    failure = asyncio.run(run())

    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert isinstance(failure.exceptions[1], TimeoutError)


@pytest.mark.parametrize("bounded", [False, True])
def test_cleanup_wait_retains_cancellation_pending_before_entry(bounded: bool) -> None:
    async def run() -> bool:
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        child = asyncio.create_task(asyncio.sleep(0))
        try:
            if bounded:
                return await _await_bounded_cleanup_task(
                    child,
                    timeout_s=1,
                    timeout_message="cleanup timed out",
                )
            return await _await_cleanup_task(child)
        finally:
            current.uncancel()

    assert asyncio.run(run()) is True


def test_managed_cleanup_does_not_replace_grouped_timeout_cancellation() -> None:
    cancellation = asyncio.CancelledError("caller cancelled")
    timeout_error = TimeoutError("runner cleanup timed out")
    timeout_group = BaseExceptionGroup(
        "runner cleanup timed out after cancellation",
        [cancellation, timeout_error],
    )

    async def run() -> BaseExceptionGroup:
        result = await _virtual_factory(adapter=_RecordingAdapter()).create(
            EnvironmentFactoryRequest(
                session_id="sess_grouped_cleanup_timeout",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        async def fail_runner_close(*, outcome: str | None, deadline: float) -> bool:
            raise timeout_group

        runner._await_runner_close = fail_runner_close  # type: ignore[attr-defined,method-assign]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await runner.close()
        return exc_info.value

    failure = asyncio.run(run())

    assert failure is timeout_group
    assert failure.exceptions == (cancellation, timeout_error)


def test_nested_runner_timeout_skips_unattempted_audit_fallback() -> None:
    cancellation = asyncio.CancelledError("caller cancelled")
    timeout_error = TimeoutError("runner cleanup timed out")
    nested_timeout = BaseExceptionGroup(
        "runner cleanup timed out after cancellation",
        [BaseExceptionGroup("nested cancellation", [cancellation]), timeout_error],
    )

    class _CountingAudit:
        calls = 0

        async def drain(self) -> None:
            self.calls += 1

    async def run() -> tuple[BaseExceptionGroup, _CountingAudit]:
        result = await _virtual_factory(adapter=_RecordingAdapter()).create(
            EnvironmentFactoryRequest(
                session_id="sess_nested_runner_timeout",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        audit = _CountingAudit()
        runner._audit = audit  # type: ignore[attr-defined]

        async def fail_runner_close(*, outcome: str | None, deadline: float) -> bool:
            raise nested_timeout

        runner._await_runner_close = fail_runner_close  # type: ignore[attr-defined,method-assign]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await runner.close()
        return exc_info.value, audit

    failure, audit = asyncio.run(run())
    assert audit.calls == 0
    assert sum(error is timeout_error for error in failure.exceptions) == 1


def test_managed_cleanup_preserves_prior_cancellation_when_later_phase_times_out() -> None:
    timeout_error = TimeoutError("runner cleanup timed out")

    async def run() -> BaseExceptionGroup:
        result = await _virtual_factory(adapter=_RecordingAdapter()).create(
            EnvironmentFactoryRequest(
                session_id="sess_prior_cancel_then_timeout",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        async def cancelled_revocation(*, timeout_s: float | None = None) -> bool:
            return True

        async def timed_out_runner_close(*, outcome: str | None, deadline: float) -> bool:
            raise timeout_error

        runner._authority_revoker.revoke = cancelled_revocation  # type: ignore[attr-defined,method-assign]
        runner._await_runner_close = timed_out_runner_close  # type: ignore[attr-defined,method-assign]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await runner.close()
        return exc_info.value

    failure = asyncio.run(run())

    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert failure.exceptions[1] is timeout_error


def test_managed_cleanup_preserves_runner_failure_before_audit_timeout() -> None:
    runner_error = RuntimeError("runner cleanup failed")
    audit_timeout = TimeoutError("audit deadline expired")

    class _FailingAudit:
        async def drain(self) -> None:
            raise audit_timeout

    async def run() -> RuntimeError:
        result = await _virtual_factory(adapter=_RecordingAdapter()).create(
            EnvironmentFactoryRequest(
                session_id="sess_runner_failure_then_audit_timeout",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        runner._audit = _FailingAudit()  # type: ignore[attr-defined]

        async def fail_runner_close(*, outcome: str | None, deadline: float) -> bool:
            raise runner_error

        runner._await_runner_close = fail_runner_close  # type: ignore[attr-defined,method-assign]
        with pytest.raises(RuntimeError) as exc_info:
            await runner.close()
        return exc_info.value

    failure = asyncio.run(run())

    assert "runner: RuntimeError: runner cleanup failed" in str(failure)
    assert "audit: TimeoutError: audit deadline expired" in str(failure)


def test_prepare_rollback_preserves_primary_failure_with_cleanup_cancellation() -> None:
    primary_error = RuntimeError("prepare failed")
    cleanup_error = RuntimeError("rollback failed")
    cancellation = asyncio.CancelledError("caller cancelled")
    cleanup_group = BaseExceptionGroup(
        "rollback cancelled and failed",
        [cancellation, cleanup_error],
    )

    with pytest.raises(BaseExceptionGroup) as exc_info:
        _raise_primary_with_cleanup_cancellation(
            primary_error,
            cleanup_group,
            message="prepare rollback failed after cancellation",
        )

    assert exc_info.value.exceptions == (primary_error, cleanup_group)
    assert exc_info.value.__cause__ is cancellation


def test_prepare_rollback_preserves_cleanup_failure_after_primary_cancellation() -> None:
    cancellation = asyncio.CancelledError("prepare cancelled")
    cleanup_error = RuntimeError("rollback failed")

    with pytest.raises(BaseExceptionGroup) as exc_info:
        _raise_primary_with_cleanup_cancellation(
            cancellation,
            cleanup_error,
            message="prepare rollback failed after cancellation",
        )

    assert exc_info.value.exceptions == (cancellation, cleanup_error)
    assert exc_info.value.__cause__ is cancellation


def test_finalize_evidence_reconciliation_preserves_cancellation_and_failure() -> None:
    reconciliation_started = asyncio.Event()
    allow_failure = asyncio.Event()
    persistence_error = RuntimeError("publication acknowledgement lost")
    reconciliation_error = RuntimeError("reconciliation failed")

    class _Writer:
        async def is_persisted(self, event: Event) -> bool:
            reconciliation_started.set()
            await allow_failure.wait()
            raise reconciliation_error

    async def run() -> BaseExceptionGroup:
        task = asyncio.create_task(
            _reconcile_binding_finalize_failure_event(
                _Writer(),  # type: ignore[arg-type]
                Event(type="custom.test.finalize", session_id="sess_reconcile_cancel"),
                persistence_error=persistence_error,
                cancellation=None,
            )
        )
        await reconciliation_started.wait()
        task.cancel()
        allow_failure.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0] is persistence_error
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)
    assert failure.__cause__ is reconciliation_error


def test_finalize_evidence_persistence_retains_cancellation_pending_before_entry() -> None:
    event = Event(type="custom.test.finalize", session_id="sess_pre_cancelled_persist")

    class _Writer:
        async def persist(self, persisted_event: Event) -> Event:
            assert persisted_event is event
            return persisted_event

    async def run() -> tuple[Event, asyncio.CancelledError | None]:
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        try:
            return await _persist_binding_finalize_failure_event(  # type: ignore[arg-type]
                _Writer(),
                event,
            )
        finally:
            current.uncancel()

    persisted, cancellation = asyncio.run(run())

    assert persisted is event
    assert isinstance(cancellation, asyncio.CancelledError)


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
    supports_reconnect = True

    def __init__(self) -> None:
        super().__init__("lambda-microvm")
        self.finalize_calls: list[str | None] = []

    def reconnect_metadata(self, runner: Runner) -> dict[str, Any]:
        return {"microvm_id": "mvm-123", "endpoint": "mvm.internal"}

    def validate_reconnect_metadata(
        self,
        reconnect_metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if set(reconnect_metadata) - {"microvm_id", "endpoint"}:
            raise InvalidEgressReconnectMetadataError(
                "Test adapter reconnect identity contains unsupported fields."
            )
        microvm_id = reconnect_metadata.get("microvm_id")
        if not isinstance(microvm_id, str) or not microvm_id:
            raise InvalidEgressReconnectMetadataError(
                "Test adapter reconnect identity requires microvm_id."
            )
        endpoint = reconnect_metadata.get("endpoint")
        if endpoint is not None and (not isinstance(endpoint, str) or not endpoint):
            raise InvalidEgressReconnectMetadataError(
                "Test adapter reconnect endpoint must be nonblank when set."
            )
        result = {"microvm_id": microvm_id}
        if endpoint is not None:
            result["endpoint"] = endpoint
        return result

    async def prepare_reconnect(
        self,
        *,
        session_id: str,
        environment_name: str,
        grants: Sequence[Any],
        broker: Any,
        reconnect_metadata: Mapping[str, Any],
    ) -> EgressBinding:
        self.captured["reconnect_identity"] = reconnect_metadata
        self.captured["reconnect_environment_name"] = environment_name
        return await self.prepare(session_id=session_id, grants=grants, broker=broker)

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


def _available_untrusted_execution_evidence(
    subject: str,
    *,
    network_state: str = "available",
    live_ttl: timedelta = timedelta(minutes=5),
) -> ExecutionCapabilityEvidence:
    claims: list[ExecutionCapabilityClaim] = []
    for capability in ExecutionRequirements.untrusted().required_capabilities():
        if capability == "deny_by_default_network" and network_state == "live_verified":
            observed_at = datetime.now(UTC)
            claims.append(
                ExecutionCapabilityClaim.live_verified(
                    capability,
                    observation="denied",
                    observed_at=observed_at,
                    valid_until=observed_at + live_ttl,
                )
            )
        elif capability == "deny_by_default_network" and network_state == "stale":
            observed_at = datetime.now(UTC) - timedelta(minutes=6)
            claims.append(
                ExecutionCapabilityClaim.live_verified(
                    capability,
                    observation="denied",
                    observed_at=observed_at,
                    valid_until=observed_at + timedelta(minutes=5),
                )
            )
        elif capability == "deny_by_default_network" and network_state == "unverified":
            claims.append(
                ExecutionCapabilityClaim(
                    capability=capability,
                    state="unverified",
                    proof_source="operator_opt_out",
                    observation="not_probed",
                    reason_code="network_boundary_unverified",
                    remediation_code="enable_network_preflight",
                )
            )
        else:
            claims.append(
                ExecutionCapabilityClaim(
                    capability=capability,
                    state="available",
                    proof_source="integration_validation",
                    observation="available",
                )
            )
    return ExecutionCapabilityEvidence(subject=subject, claims=tuple(claims))


class _MixedAssuranceAdapter(_RecordingAdapter):
    def __init__(self, runtime_network_state: str) -> None:
        super().__init__("hosted-runner")
        self.runtime_network_state = runtime_network_state

    def execution_capability_evidence(
        self,
        runner: Runner | None = None,
    ) -> ExecutionCapabilityEvidence:
        return _available_untrusted_execution_evidence(
            self.runner_kind,
            network_state=self.runtime_network_state if runner is not None else "available",
        )


def _live_network_execution_requirements() -> ExecutionRequirements:
    return ExecutionRequirements.untrusted(
        evidence_overrides=(
            ExecutionEvidenceOverride(
                capability="deny_by_default_network",
                minimum_evidence="live_verified",
            ),
        )
    )


class _RetryingLifecycleAdapter(_RecordingAdapter):
    def __init__(self, *, first_error: RuntimeError | None = None) -> None:
        super().__init__("lambda-microvm")
        self.finalize_calls = 0
        self.first_error = first_error or RuntimeError("suspend failed")

    async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
        self.finalize_calls += 1
        if self.finalize_calls == 1:
            raise self.first_error
        await runner.close()


class _RetryingReconnectAdapter(_LifecycleRecordingAdapter):
    async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
        self.finalize_calls.append(outcome)
        if len(self.finalize_calls) == 1:
            raise RuntimeError("suspend failed")
        await runner.close()


class _VirtualCredentialEchoingAdapter(_RecordingAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_error: RuntimeError | None = None

    async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
        request = self.captured["runner_request"]
        presented_value = request.env_overlay["STRIPE_SECRET_KEY"]
        self.cleanup_error = RuntimeError(
            f"adapter cleanup echoed virtual credential: {presented_value}"
        )
        raise self.cleanup_error


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


def test_factory_requires_explicit_runner_selection() -> None:
    with pytest.raises(ValueError, match="explicit adapter or runner_kind"):
        _virtual_factory()


def test_factory_does_not_fallback_to_docker_for_an_unavailable_microvm() -> None:
    docker_adapter = _RecordingAdapter("docker")
    registry = EgressAdapterRegistry()
    registry.register(docker_adapter)

    with pytest.raises(UnsupportedEgressError, match="microsandbox"):
        _virtual_factory(
            adapter_registry=registry,
            runner_kind="microsandbox",
        )

    assert docker_adapter.prepare_calls == []


def test_factory_refuses_untrusted_execution_before_adapter_resources() -> None:
    adapter = _RecordingAdapter("custom-runner")

    async def run() -> None:
        factory = _virtual_factory(adapter=adapter)
        with pytest.raises(ExecutionAdmissionError) as raised:
            await factory.create(
                EnvironmentFactoryRequest(
                    session_id="sess_admission",
                    agent_name="agent",
                    environment_name="egress-env",
                    execution_requirements=ExecutionRequirements.untrusted(),
                )
            )
        assert {refusal.capability for refusal in raised.value.decision.refusals} == set(
            ExecutionRequirements.untrusted().required_capabilities()
        )

    asyncio.run(run())

    assert adapter.prepare_calls == []


def test_builtin_docker_is_explicitly_unsupported_for_untrusted_execution() -> None:
    async def run() -> None:
        factory = _virtual_factory(runner_kind="docker")
        with pytest.raises(ExecutionAdmissionError) as raised:
            await factory.create(
                EnvironmentFactoryRequest(
                    session_id="sess_untrusted_docker",
                    agent_name="agent",
                    environment_name="egress-env",
                    execution_requirements=ExecutionRequirements.untrusted(),
                )
            )
        refusal = next(
            item
            for item in raised.value.decision.refusals
            if item.capability == "untrusted_code_isolation"
        )
        assert refusal.code == "unsupported_capability"
        assert refusal.reason_code == "container_isolation_unsupported"

    asyncio.run(run())


def test_factory_does_not_accept_caller_assertions_in_place_of_adapter_evidence() -> None:
    evidence = _available_untrusted_execution_evidence("docker")

    with pytest.raises(TypeError, match="execution_evidence"):
        _virtual_factory(
            runner_kind="docker",
            execution_evidence=evidence,
        )


def test_factory_refuses_weakened_runtime_evidence_and_cleans_up_before_exposure() -> None:
    class _RuntimeEvidenceAdapter(_RecordingAdapter):
        def execution_capability_evidence(
            self,
            runner: Runner | None = None,
        ) -> ExecutionCapabilityEvidence:
            return _available_untrusted_execution_evidence(
                self.runner_kind,
                network_state="unverified" if runner is not None else "available",
            )

    adapter = _RuntimeEvidenceAdapter("hosted-runner")

    async def run() -> None:
        factory = _virtual_factory(adapter=adapter)
        with pytest.raises(ExecutionAdmissionError) as raised:
            await factory.create(
                EnvironmentFactoryRequest(
                    session_id="sess_runtime_admission",
                    agent_name="agent",
                    environment_name="egress-env",
                    execution_requirements=ExecutionRequirements.untrusted(),
                )
            )
        assert raised.value.decision.stage == "pre_exposure"
        assert raised.value.decision.refusals[0].capability == "deny_by_default_network"

    asyncio.run(run())

    assert len(adapter.prepare_calls) == 1
    assert adapter.torn_down == 1
    runner = adapter.captured["inner_runner"]
    assert runner.closed is True


def test_factory_admits_live_network_with_available_isolation_and_lifecycle_evidence() -> None:
    adapter = _MixedAssuranceAdapter("live_verified")

    async def run() -> Any:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_mixed_assurance",
                agent_name="agent",
                environment_name="egress-env",
                execution_requirements=_live_network_execution_requirements(),
            )
        )
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return result

    result = asyncio.run(run())

    claims = {
        claim["capability"]: claim["state"]
        for claim in result.metadata["execution_capabilities"]["claims"]
    }
    assert claims["deny_by_default_network"] == "live_verified"
    assert claims["untrusted_code_isolation"] == "available"


def test_factory_rechecks_live_evidence_after_async_setup_before_return() -> None:
    class _ExpiringEvidenceAdapter(_RecordingAdapter):
        def __init__(self) -> None:
            super().__init__("hosted-runner")
            self.runtime_evidence: ExecutionCapabilityEvidence | None = None

        def execution_capability_evidence(
            self,
            runner: Runner | None = None,
        ) -> ExecutionCapabilityEvidence:
            if runner is None:
                return _available_untrusted_execution_evidence(self.runner_kind)
            if self.runtime_evidence is None:
                self.runtime_evidence = _available_untrusted_execution_evidence(
                    self.runner_kind,
                    network_state="live_verified",
                    live_ttl=timedelta(milliseconds=50),
                )
            return self.runtime_evidence

    adapter = _ExpiringEvidenceAdapter()

    async def emitter(event: Event) -> Event:
        await asyncio.sleep(0.06)
        return event

    async def run() -> None:
        with pytest.raises(ExecutionAdmissionError) as raised:
            await _virtual_factory(adapter=adapter, event_emitter=emitter).create(
                EnvironmentFactoryRequest(
                    session_id="sess_expired_before_return",
                    agent_name="agent",
                    environment_name="egress-env",
                    execution_requirements=_live_network_execution_requirements(),
                )
            )
        assert [(item.capability, item.code) for item in raised.value.decision.refusals] == [
            ("deny_by_default_network", "stale_evidence")
        ]

    asyncio.run(run())

    assert adapter.torn_down == 1
    assert adapter.captured["inner_runner"].closed is True


@pytest.mark.parametrize(
    ("network_state", "expected_code"),
    [
        ("available", "insufficient_evidence"),
        ("stale", "stale_evidence"),
    ],
)
def test_factory_refuses_weakened_or_stale_capability_override(
    network_state: str,
    expected_code: str,
) -> None:
    adapter = _MixedAssuranceAdapter(network_state)

    async def run() -> None:
        with pytest.raises(ExecutionAdmissionError) as raised:
            await _virtual_factory(adapter=adapter).create(
                EnvironmentFactoryRequest(
                    session_id=f"sess_mixed_{network_state}",
                    agent_name="agent",
                    environment_name="egress-env",
                    execution_requirements=_live_network_execution_requirements(),
                )
            )
        assert [(item.capability, item.code) for item in raised.value.decision.refusals] == [
            ("deny_by_default_network", expected_code)
        ]

    asyncio.run(run())

    assert adapter.torn_down == 1
    runner = adapter.captured["inner_runner"]
    assert runner.closed is True


def test_factory_publishes_admitted_requirements_and_execution_evidence() -> None:
    class _AdmissibleAdapter(_RecordingAdapter):
        def execution_capability_evidence(
            self,
            runner: Runner | None = None,
        ) -> ExecutionCapabilityEvidence:
            return _available_untrusted_execution_evidence(self.runner_kind)

    adapter = _AdmissibleAdapter("hosted-runner")

    async def run() -> Any:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_admitted_evidence",
                agent_name="agent",
                environment_name="egress-env",
                execution_requirements=ExecutionRequirements.untrusted(),
            )
        )
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return result

    result = asyncio.run(run())

    assert result.metadata["execution_requirements"]["code_trust"] == "untrusted"
    assert result.metadata["execution_capabilities"]["schema"] == ("cayu.execution_capabilities.v1")
    assert result.metadata["execution_capabilities"]["subject"] == "hosted-runner"
    assert (
        result.environment.spec.metadata["execution_capabilities"]
        == (result.metadata["execution_capabilities"])
    )


def test_factory_rejects_an_unsupported_explicit_runner_without_a_registry() -> None:
    with pytest.raises(UnsupportedEgressError, match="microsandbox"):
        _virtual_factory(runner_kind="microsandbox")


def test_factory_rejects_conflicting_adapter_and_runner_selection() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _virtual_factory(
            adapter=_RecordingAdapter("docker"),
            runner_kind="microsandbox",
        )


def test_factory_rejects_an_explicit_unsupported_adapter() -> None:
    with pytest.raises(UnsupportedEgressError, match="microsandbox"):
        _virtual_factory(adapter=UnsupportedEgressAdapter("microsandbox"))


def test_factory_rejects_duplicate_credential_env_names() -> None:
    with pytest.raises(ValueError, match="env_name values must be unique"):
        VirtualEgressEnvironmentFactory(
            resolver=StaticVault({"stripe_test_key": REAL_SECRET}),
            policies={POLICY_NAME: _provider_example_policy()},
            runner_kind="docker",
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
                operation=EnvironmentFactoryOperation.RECONNECT,
                reconnect_metadata={
                    "version": 1,
                    "runner_kind": "lambda-microvm",
                    "session_id": "sess_resume",
                    "environment_name": "egress-env",
                    "capability": "supported",
                    "identity": {"microvm_id": "mvm-old", "endpoint": "old.internal"},
                },
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
    assert adapter.captured["reconnect_identity"] == runner_request.reconnect_metadata
    assert result.reconnect_metadata == {
        "version": 1,
        "runner_kind": "lambda-microvm",
        "session_id": "sess_resume",
        "environment_name": "egress-env",
        "capability": "supported",
        "identity": {
            "microvm_id": "mvm-123",
            "endpoint": "mvm.internal",
        },
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


def test_factory_reconnect_operation_refuses_missing_durable_metadata() -> None:
    adapter = _LifecycleRecordingAdapter()

    async def run() -> None:
        with pytest.raises(InvalidEgressReconnectMetadataError, match="requires durable"):
            await _virtual_factory(adapter=adapter).create(
                EnvironmentFactoryRequest(
                    session_id="sess_missing_reconnect",
                    agent_name="agent",
                    environment_name="egress-env",
                    operation=EnvironmentFactoryOperation.RECONNECT,
                )
            )

    asyncio.run(run())
    assert adapter.prepare_calls == []


def test_factory_create_operation_refuses_same_session_reconnect_metadata() -> None:
    adapter = _LifecycleRecordingAdapter()

    async def run() -> None:
        with pytest.raises(InvalidEgressReconnectMetadataError, match="explicit reconnect"):
            await _virtual_factory(adapter=adapter).create(
                EnvironmentFactoryRequest(
                    session_id="sess_accidental_attach",
                    agent_name="agent",
                    environment_name="egress-env",
                    reconnect_metadata={
                        "version": 1,
                        "runner_kind": "lambda-microvm",
                        "session_id": "sess_accidental_attach",
                        "environment_name": "egress-env",
                        "capability": "supported",
                        "identity": {"microvm_id": "mvm-old"},
                    },
                )
            )

    asyncio.run(run())
    assert adapter.prepare_calls == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2, "version"),
        ("runner_kind", "microsandbox", "runner kind"),
        ("session_id", "other-session", "different session"),
        ("environment_name", "other-environment", "different environment"),
        ("capability", "mystery", "capability"),
        ("identity", {}, "non-empty object"),
    ],
)
def test_factory_rejects_invalid_reconnect_scope_before_adapter_prepare(
    field: str,
    value: Any,
    message: str,
) -> None:
    adapter = _LifecycleRecordingAdapter()
    metadata = {
        "version": 1,
        "runner_kind": "lambda-microvm",
        "session_id": "sess_resume",
        "environment_name": "egress-env",
        "capability": "supported",
        "identity": {"microvm_id": "mvm-old"},
    }
    metadata[field] = value

    async def run() -> None:
        with pytest.raises(InvalidEgressReconnectMetadataError, match=message):
            await _virtual_factory(adapter=adapter).create(
                EnvironmentFactoryRequest(
                    session_id="sess_resume",
                    agent_name="agent",
                    environment_name="egress-env",
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata=metadata,
                )
            )

    asyncio.run(run())

    assert adapter.prepare_calls == []
    assert "runner_request" not in adapter.captured


@pytest.mark.parametrize(
    "authority_field",
    [
        "token",
        "authToken",
        "authorization",
        "client_secret_value",
        "cookie",
        "apiKey",
        "xApiKeyValue",
        "caPrivateKeyPem",
        "proxy-authorization",
    ],
)
def test_factory_rejects_replayable_authority_in_reconnect_metadata(
    authority_field: str,
) -> None:
    adapter = _LifecycleRecordingAdapter()

    async def run() -> None:
        with pytest.raises(InvalidEgressReconnectMetadataError, match="replayable authority"):
            await _virtual_factory(adapter=adapter).create(
                EnvironmentFactoryRequest(
                    session_id="sess_resume",
                    agent_name="agent",
                    environment_name="egress-env",
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata={
                        "version": 1,
                        "runner_kind": "lambda-microvm",
                        "session_id": "sess_resume",
                        "environment_name": "egress-env",
                        "capability": "supported",
                        "identity": {"microvm_id": "mvm-old", authority_field: "replay-me"},
                    },
                )
            )

    asyncio.run(run())

    assert adapter.prepare_calls == []


def test_factory_rejects_adapter_reconnect_authority_and_rolls_back() -> None:
    class _UnsafeMetadataAdapter(_LifecycleRecordingAdapter):
        def reconnect_metadata(self, runner: Runner) -> dict[str, Any]:
            del runner
            return {"microvm_id": "mvm-1", "token": "replay-me"}

    adapter = _UnsafeMetadataAdapter()

    async def run() -> None:
        with pytest.raises(InvalidEgressReconnectMetadataError, match="unsupported fields"):
            await _virtual_factory(adapter=adapter).create(
                EnvironmentFactoryRequest(
                    session_id="sess_create",
                    agent_name="agent",
                    environment_name="egress-env",
                )
            )

    asyncio.run(run())

    assert adapter.captured["inner_runner"].closed is True
    assert adapter.torn_down == 1


def test_factory_rejects_malformed_reconnect_schema() -> None:
    adapter = _LifecycleRecordingAdapter()

    async def run() -> None:
        with pytest.raises(InvalidEgressReconnectMetadataError, match="invalid schema"):
            await _virtual_factory(adapter=adapter).create(
                EnvironmentFactoryRequest(
                    session_id="sess_resume",
                    agent_name="agent",
                    environment_name="egress-env",
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata={
                        "version": 1,
                        "runner_kind": "lambda-microvm",
                        "session_id": "sess_resume",
                        "capability": "supported",
                        "identity": {"microvm_id": "mvm-old"},
                        "unexpected": True,
                    },
                )
            )

    asyncio.run(run())

    assert adapter.prepare_calls == []


def test_factory_fails_closed_when_adapter_cannot_reconnect() -> None:
    adapter = _RecordingAdapter("docker")

    async def run() -> dict[str, Any]:
        created = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_resume",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = created.environment.runner
        assert runner is not None
        await runner.close()
        metadata = created.reconnect_metadata
        adapter.prepare_calls = []
        adapter.captured = {}
        with pytest.raises(UnsupportedEgressReconnectError, match="explicitly rebuild"):
            await _virtual_factory(adapter=adapter).create(
                EnvironmentFactoryRequest(
                    session_id="sess_resume",
                    agent_name="agent",
                    environment_name="egress-env",
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata=metadata,
                )
            )
        return metadata

    metadata = asyncio.run(run())

    assert metadata["capability"] == "unsupported"
    assert metadata["runner_kind"] == "docker"
    assert "identity" not in metadata
    assert adapter.prepare_calls == []
    assert "runner_request" not in adapter.captured


def test_factory_fork_ignores_valid_parent_reconnect_identity() -> None:
    adapter = _LifecycleRecordingAdapter()

    async def run() -> Any:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="child-session",
                parent_session_id="parent-session",
                agent_name="agent",
                environment_name="egress-env",
                reconnect_metadata={
                    "version": 1,
                    "runner_kind": "lambda-microvm",
                    "session_id": "parent-session",
                    "environment_name": "egress-env",
                    "capability": "supported",
                    "identity": {"microvm_id": "parent-mvm"},
                },
            )
        )
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return result

    result = asyncio.run(run())

    assert adapter.prepare_calls[0]["session_id"] == "child-session"
    assert "reconnect_identity" not in adapter.captured
    assert adapter.captured["runner_request"].reconnect_metadata == {}
    assert result.reconnect_metadata["session_id"] == "child-session"


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
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)
        return runner

    runner = asyncio.run(run())

    assert runner.closed is True
    assert adapter.torn_down == 1


def test_bind_failure_detaches_a_reconnected_environment() -> None:
    adapter = _LifecycleRecordingAdapter()

    class _FailingBindBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("bind failed")

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    async def run() -> None:
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=_FailingBindBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_bind_reconnect",
                agent_name="agent",
                environment_name="egress-env",
                operation=EnvironmentFactoryOperation.RECONNECT,
                reconnect_metadata={
                    "version": 1,
                    "runner_kind": "lambda-microvm",
                    "session_id": "sess_bind_reconnect",
                    "environment_name": "egress-env",
                    "capability": "supported",
                    "identity": {"microvm_id": "mvm-old"},
                },
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        with pytest.raises(RuntimeError, match="bind failed"):
            await binding.bind(None, runner, session_id="sess_bind_reconnect")
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.PRESERVE)

    asyncio.run(run())
    assert adapter.finalize_calls == ["interrupted"]
    assert adapter.torn_down == 1


def test_factory_release_retries_incomplete_unadopted_cleanup() -> None:
    adapter = _RetryingLifecycleAdapter()
    events: list[Event] = []

    class _FailingBindBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("bind failed")

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    async def emit(event: Event) -> Event:
        events.append(event)
        return event

    async def run() -> None:
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=_FailingBindBinding(),
            event_emitter=emit,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_bind_cleanup_failure",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        with pytest.raises(RuntimeError, match="bind failed"):
            await binding.bind(None, runner, session_id="sess_bind_cleanup_failure")
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)
        assert adapter.finalize_calls == 2
        assert runner.closed is True
        assert adapter.torn_down == 1
        assert sum(event.type is EventType.EGRESS_GRANT_REVOKED for event in events) == 1

        # A later close converges without repeating provider or egress cleanup.
        await runner.close()
        assert adapter.finalize_calls == 2
        assert adapter.torn_down == 1

    asyncio.run(run())


def test_app_retries_factory_release_after_bind_failure() -> None:
    adapter = _RetryingReconnectAdapter()
    egress_events: list[Event] = []

    class _FailingBindBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("bind failed")

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    class _UnreachedProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def emit(event: Event) -> Event:
        egress_events.append(event)
        return event

    async def run() -> tuple[list[Event], _UnreachedProvider]:
        store = InMemorySessionStore()
        provider = _UnreachedProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="egress-env"),
            _virtual_factory(
                adapter=adapter,
                inner_binding=_FailingBindBinding(),
                event_emitter=emit,
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_bind_cleanup_app_retry",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        return await store.load_events("sess_bind_cleanup_app_retry"), provider

    events, provider = asyncio.run(run())

    binding_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FAILED
    )
    session_failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
    expected_release = {
        "action": "preserve",
        "callback_provided": True,
        "completed": True,
    }
    assert binding_failed.payload["environment_factory_release"] == expected_release
    assert session_failed.payload["environment_factory_release"] == expected_release
    assert sum(event.type is EventType.EGRESS_GRANT_REVOKED for event in egress_events) == 1
    assert adapter.finalize_calls == ["interrupted", "interrupted"]
    assert adapter.torn_down == 1
    assert provider.requests == []


def test_app_retries_factory_release_during_bind_cancellation() -> None:
    adapter = _RetryingReconnectAdapter()
    bind_started = asyncio.Event()

    class _CancelledBindBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            bind_started.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled bind unexpectedly resumed")

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

    class _UnreachedProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> _UnreachedProvider:
        store = InMemorySessionStore()
        provider = _UnreachedProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="egress-env"),
            _virtual_factory(
                adapter=adapter,
                inner_binding=_CancelledBindBinding(),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def run_app() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_cancelled_bind_cleanup_retry",
                        messages=[Message.text("user", "run")],
                    )
                )
            ]

        run_task = asyncio.create_task(run_app())
        await asyncio.wait_for(bind_started.wait(), timeout=1)
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
        return provider

    provider = asyncio.run(run())

    assert adapter.finalize_calls == ["interrupted", "interrupted"]
    assert adapter.torn_down == 1
    assert provider.requests == []


def test_factory_rollback_preserves_cleanup_failure_after_creation_cancellation() -> None:
    cancellation = asyncio.CancelledError("creation cancelled")
    cleanup_error = RuntimeError("runner rollback failed")

    class _RollbackFailingAdapter(_RecordingAdapter):
        async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
            raise cleanup_error

    async def run() -> BaseExceptionGroup:
        factory = _virtual_factory(adapter=_RollbackFailingAdapter())

        async def cancel_grant_events(*args: Any, **kwargs: Any) -> None:
            raise cancellation

        factory._emit_grant_events = cancel_grant_events  # type: ignore[method-assign]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await factory.create(
                EnvironmentFactoryRequest(
                    session_id="sess_cancelled_factory_rollback_failure",
                    agent_name="assistant",
                    environment_name="egress-env",
                )
            )
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0] is cancellation
    assert isinstance(failure.exceptions[1], RuntimeError)
    assert "runner rollback failed" in str(failure.exceptions[1])


def test_factory_rollback_preserves_cancellation_after_ordinary_creation_failure() -> None:
    rollback_started = asyncio.Event()
    allow_rollback = asyncio.Event()
    creation_error = RuntimeError("grant publication failed")

    class _BlockingRollbackAdapter(_RecordingAdapter):
        async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
            rollback_started.set()
            await allow_rollback.wait()
            await runner.close()

    async def run() -> BaseExceptionGroup:
        factory = _virtual_factory(adapter=_BlockingRollbackAdapter())

        async def fail_grant_events(*args: Any, **kwargs: Any) -> None:
            raise creation_error

        factory._emit_grant_events = fail_grant_events  # type: ignore[method-assign]
        create_task = asyncio.create_task(
            factory.create(
                EnvironmentFactoryRequest(
                    session_id="sess_factory_rollback_cancelled",
                    agent_name="assistant",
                    environment_name="egress-env",
                )
            )
        )
        await rollback_started.wait()
        create_task.cancel()
        allow_rollback.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await create_task
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0] is creation_error
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)


@pytest.mark.parametrize(
    ("action", "expected_outcome"),
    [
        (EnvironmentFactoryReleaseAction.PRESERVE, "interrupted"),
        (EnvironmentFactoryReleaseAction.DISCARD, None),
    ],
)
def test_factory_release_is_idempotent_under_concurrent_calls(
    action: EnvironmentFactoryReleaseAction,
    expected_outcome: str | None,
) -> None:
    adapter = _LifecycleRecordingAdapter()
    events: list[Event] = []

    async def emit(event: Event) -> Event:
        events.append(event)
        return event

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter, event_emitter=emit).create(
            EnvironmentFactoryRequest(
                session_id="sess_factory_release_preserve",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        assert result.release is not None
        await asyncio.gather(result.release(action), result.release(action))
        await result.release(action)

    asyncio.run(run())

    assert adapter.finalize_calls == [expected_outcome]
    assert adapter.torn_down == 1
    assert sum(event.type is EventType.EGRESS_GRANT_REVOKED for event in events) == 1


def test_concurrent_factory_release_escalates_preserve_to_discard_once() -> None:
    adapter = _LifecycleRecordingAdapter()
    events: list[Event] = []

    async def emit(event: Event) -> Event:
        events.append(event)
        return event

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter, event_emitter=emit).create(
            EnvironmentFactoryRequest(
                session_id="sess_factory_release_escalation",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        assert result.release is not None
        await asyncio.gather(
            result.release(EnvironmentFactoryReleaseAction.PRESERVE),
            result.release(EnvironmentFactoryReleaseAction.DISCARD),
        )
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(run())

    assert adapter.finalize_calls[-1] is None
    assert len(adapter.finalize_calls) <= 2
    assert adapter.torn_down == 1
    assert sum(event.type is EventType.EGRESS_GRANT_REVOKED for event in events) == 1


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


def test_factory_preserves_trusted_execution_for_aws_workspace_lifecycle(
    tmp_path: Path,
) -> None:
    mountpoint_checks = 0

    def scripted_exit_code(payload: dict[str, Any]) -> int:
        nonlocal mountpoint_checks
        argv = payload.get("argv", [])
        if argv[:2] == ["mountpoint", "-q"]:
            mountpoint_checks += 1
            return 1 if mountpoint_checks == 1 else 0
        return 0

    transport = SupervisorTransport(tmp_path, scripted_exit_code=scripted_exit_code)
    inner = LambdaMicroVMRunner(
        ConformanceLambdaClient(),
        microvm_id="mvm-factory-composition",
        endpoint="factory.lambda-microvm.invalid",
        image_identifier="arn:aws:lambda:us-east-1:123:microvm-image:factory",
        region_name="us-east-1",
        default_cwd="/workspace",
        close_action="none",
        endpoint_transport=transport,
        poll_interval_s=0,
    )
    assert isinstance(inner, LambdaMicroVMRunner)
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
        assert runner.system_execution_mode == "separate"
        agent_result = await runner.exec(
            ExecCommand.process("agent-command"),
            cwd="/workspace/agent",
            env={"LANE": "agent"},
            timeout_s=17,
            stdin="agent-input",
            output_limit_bytes=321,
        )
        assert agent_result.exit_code == 0
        trusted_result = await runner.exec_system(
            ExecCommand.process("system-command"),
            cwd="/workspace/system",
            env={"LANE": "trusted"},
            timeout_s=23,
            stdin="trusted-input",
            output_limit_bytes=654,
        )
        assert trusted_result.exit_code == 0
        bound = await binding.bind(None, runner, session_id="sess_trusted_workspace")
        await binding.finalize(bound, outcome="completed")
        with pytest.raises(RuntimeError, match="closed"):
            await runner.exec_system(ExecCommand.process("true"))

    asyncio.run(run())

    assert transport.payloads[:2] == [
        {
            "execution_profile": "agent",
            "kind": "process",
            "cwd": "/workspace/agent",
            "env": {"LANE": "agent"},
            "stdin_base64": "YWdlbnQtaW5wdXQ=",
            "timeout_s": 17,
            "output_limit_bytes": 321,
            "argv": ["agent-command"],
        },
        {
            "execution_profile": "trusted",
            "kind": "process",
            "cwd": "/workspace/system",
            "env": {"LANE": "trusted"},
            "stdin_base64": "dHJ1c3RlZC1pbnB1dA==",
            "timeout_s": 23,
            "output_limit_bytes": 654,
            "argv": ["system-command"],
        },
    ]
    assert [payload["execution_profile"] for payload in transport.payloads] == ["agent"] + [
        "trusted"
    ] * 8
    assert [payload["argv"][0] for payload in transport.payloads] == [
        "agent-command",
        "system-command",
        "mkdir",
        "mountpoint",
        "mount",
        "mountpoint",
        "sync",
        "mountpoint",
        "env",
    ]
    assert transport.payloads[-1]["argv"] == [
        "env",
        "--chdir=/",
        "umount",
        "--",
        "/workspace",
    ]


def test_managed_wrapper_preserves_trusted_cancellation_and_inner_exec_latch() -> None:
    class _CancellableSeparateLaneRunner(Runner):
        isolation = "lambda-microvm"
        system_execution_mode = "separate"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False
            self.block_system = True

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            del command, kwargs
            self._ensure_exec_open()
            return ExecResult()

        async def exec_system(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            del command, kwargs
            self._ensure_exec_open()
            if not self.block_system:
                return ExecResult()
            self.started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        def latch_exec(self) -> None:
            self._close_exec("fixture command state is unknown")

    inner = _CancellableSeparateLaneRunner()
    adapter = _RecordingAdapter(
        "lambda-microvm",
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_trusted_state",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        task = asyncio.create_task(runner.exec_system(ExecCommand.process("wait")))
        await inner.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert inner.cancelled is True

        inner.latch_exec()
        with pytest.raises(RuntimeError, match="unknown"):
            await runner.exec(ExecCommand.process("agent"))
        with pytest.raises(RuntimeError, match="unknown"):
            await runner.exec_system(ExecCommand.process("trusted"))

        runner.reopen_exec()
        inner.block_system = False
        assert (await runner.exec_system(ExecCommand.process("trusted"))).exit_code == 0
        await runner.close()

    asyncio.run(run())


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


def test_runner_failure_keeps_binding_ownership_claim_for_retry() -> None:
    adapter = _RetryingLifecycleAdapter()

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_claim_retry",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        with pytest.raises(RuntimeError, match="runner: RuntimeError"):
            await runner.close()
        assert adapter.torn_down == 0
        await runner.close()

    asyncio.run(run())
    assert adapter.torn_down == 1


def test_terminal_retry_escalates_a_completed_interrupted_detach() -> None:
    adapter = _LifecycleRecordingAdapter()

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_escalate_cleanup",
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
                raise RuntimeError("claim release failed")
            await original_teardown()

        binding.teardown = flaky_teardown
        with pytest.raises(RuntimeError, match="binding: RuntimeError"):
            await runner.finalize(outcome="interrupted")
        await runner.close()

    asyncio.run(run())
    assert adapter.finalize_calls == ["interrupted", None]
    assert adapter.torn_down == 1


def test_concurrent_terminal_escalation_keeps_claim_until_remove_completes() -> None:
    detach_started = asyncio.Event()
    allow_detach = asyncio.Event()
    remove_started = asyncio.Event()
    allow_remove = asyncio.Event()

    class _CoordinatedAdapter(_LifecycleRecordingAdapter):
        async def finalize_runner(self, runner: Runner, *, outcome: str | None) -> None:
            self.finalize_calls.append(outcome)
            if outcome == "interrupted":
                detach_started.set()
                await allow_detach.wait()
                return
            remove_started.set()
            await allow_remove.wait()
            await runner.close()

    adapter = _CoordinatedAdapter()

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_concurrent_escalation",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        interrupted = asyncio.create_task(runner.finalize(outcome="interrupted"))
        await detach_started.wait()
        terminal = asyncio.create_task(runner.close())
        await asyncio.sleep(0)
        allow_detach.set()
        await remove_started.wait()
        assert adapter.torn_down == 0
        allow_remove.set()
        await asyncio.gather(interrupted, terminal)

    asyncio.run(run())
    assert adapter.finalize_calls == ["interrupted", None]
    assert adapter.torn_down == 1


def test_revocation_failure_stops_before_workspace_finalize_and_claim_release() -> None:
    adapter = _RecordingAdapter()
    inner_finalized = False

    class _TrackingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            nonlocal inner_finalized
            inner_finalized = True
            return None

    async def run() -> BaseException:
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=_TrackingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_revoke_failure",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(None, runner, session_id="sess_revoke_failure")

        async def fail_revoke() -> bool:
            raise RuntimeError("revocation failed")

        runner.revoke_authority = fail_revoke  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="revocation failed") as exc_info:
            await binding.finalize(bound, outcome="failed")
        return exc_info.value

    failure = asyncio.run(run())
    assert inner_finalized is False
    assert adapter.torn_down == 0
    assert binding_finalize_failure_payload(failure, redactor=SecretRedactor()) == [
        {
            "phase": "managed_resource_cleanup",
            "error": "revocation failed",
            "error_type": "RuntimeError",
        }
    ]


def test_app_persists_revocation_failure_as_managed_cleanup() -> None:
    adapter = _RecordingAdapter()
    inner_finalized = False

    class _TrackingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            nonlocal inner_finalized
            inner_finalized = True
            return None

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> list[Event]:
        store = InMemorySessionStore()
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=_TrackingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_revoke_failure_durable",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        async def fail_revoke() -> bool:
            raise RuntimeError("revocation failed")

        runner.revoke_authority = fail_revoke  # type: ignore[method-assign]
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_revoke_failure_durable",
                    messages=[Message.text("user", "finish")],
                )
            )
        ]
        return await store.load_events("sess_revoke_failure_durable")

    events = asyncio.run(run())
    finalize_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    )

    assert finalize_failed.payload["failures"] == [
        {
            "phase": "managed_resource_cleanup",
            "error": "revocation failed",
            "error_type": "RuntimeError",
        }
    ]
    assert inner_finalized is False
    assert adapter.torn_down == 0


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


def test_finalize_preserves_workspace_and_cleanup_failures_in_order() -> None:
    long_tail = "界" * 300
    workspace_message = f"workspace finalization failed: {REAL_SECRET}: {long_tail}"
    cleanup_message = f"runner cleanup failed: {REAL_SECRET}: {long_tail}"
    workspace_error = RuntimeError(workspace_message)
    cleanup_error = RuntimeError(cleanup_message)
    cleanup_error.credentials = {"token": REAL_SECRET}  # type: ignore[attr-defined]
    cleanup_calls = 0

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    async def run() -> BaseExceptionGroup:
        nonlocal cleanup_calls
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_dual_failure",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(None, runner, session_id="sess_finalize_dual_failure")

        async def fail_cleanup(*, outcome: str | None) -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        with pytest.raises(ExceptionGroup) as exc_info:
            await binding.finalize(bound, outcome="failed")
        return exc_info.value

    failure = asyncio.run(run())

    assert cleanup_calls == 1
    assert failure.exceptions == (workspace_error, cleanup_error)
    assert str(failure.exceptions[0]) == workspace_message
    assert str(failure.exceptions[1]) == cleanup_message
    payload = binding_finalize_failure_payload(
        failure,
        redactor=SecretRedactor(REAL_SECRET),
    )
    assert payload is not None
    assert [item["phase"] for item in payload] == [
        "workspace_finalize",
        "managed_resource_cleanup",
    ]
    assert all(item["error_type"] == "RuntimeError" for item in payload)
    assert all(REDACTED_SECRET in item["error"] for item in payload)
    assert all(REAL_SECRET not in item["error"] for item in payload)
    assert all(
        len(item["error"].encode("utf-8")) <= BINDING_FINALIZE_ERROR_TEXT_MAX_BYTES
        for item in payload
    )
    assert all(item["error"].endswith("... [truncated]") for item in payload)


@pytest.mark.parametrize("failed_phase", ["workspace", "cleanup"])
def test_finalize_preserves_single_failure_identity(failed_phase: str) -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    cleanup_error = RuntimeError("runner cleanup failed")

    class _Binding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            if failed_phase == "workspace":
                raise workspace_error
            return None

    async def run() -> BaseException:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_Binding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id=f"sess_finalize_single_{failed_phase}",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            None,
            runner,
            session_id=f"sess_finalize_single_{failed_phase}",
        )
        if failed_phase == "cleanup":

            async def fail_cleanup(*, outcome: str | None) -> None:
                raise cleanup_error

            runner.finalize = fail_cleanup  # type: ignore[method-assign]
        with pytest.raises(RuntimeError) as exc_info:
            await binding.finalize(bound, outcome="failed")
        return exc_info.value

    failure = asyncio.run(run())
    expected = workspace_error if failed_phase == "workspace" else cleanup_error
    expected_phase = (
        "workspace_finalize" if failed_phase == "workspace" else "managed_resource_cleanup"
    )

    assert failure is expected
    assert binding_finalize_failure_payload(failure, redactor=SecretRedactor()) == [
        {
            "phase": expected_phase,
            "error": str(expected),
            "error_type": "RuntimeError",
        }
    ]


def test_finalize_preserves_external_workspace_cancellation_with_cleanup_failure() -> None:
    workspace_started = asyncio.Event()
    cleanup_error = RuntimeError("runner cleanup failed")

    class _CancellingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            workspace_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def run() -> BaseExceptionGroup:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_CancellingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_workspace_cancelled",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(None, runner, session_id="sess_finalize_workspace_cancelled")

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        task = asyncio.create_task(binding.finalize(bound, outcome="interrupted"))
        await workspace_started.wait()
        task.cancel()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value

    failure = asyncio.run(run())

    assert type(failure) is BaseExceptionGroup
    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert failure.exceptions[1] is cleanup_error
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "managed_resource_cleanup",
    ]


def test_finalize_preserves_workspace_failure_with_external_cleanup_cancellation() -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    async def run() -> BaseExceptionGroup:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_cleanup_cancelled",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(None, runner, session_id="sess_finalize_cleanup_cancelled")

        async def finish_cleanup(*, outcome: str | None) -> None:
            cleanup_started.set()
            await allow_cleanup.wait()

        runner.finalize = finish_cleanup  # type: ignore[method-assign]
        task = asyncio.create_task(binding.finalize(bound, outcome="interrupted"))
        await cleanup_started.wait()
        task.cancel()
        allow_cleanup.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value

    failure = asyncio.run(run())

    assert type(failure) is BaseExceptionGroup
    assert failure.exceptions[0] is workspace_error
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "managed_resource_cleanup",
    ]


def test_finalize_preserves_deferred_revocation_cancellation_with_cleanup_failure() -> None:
    cleanup_error = RuntimeError("runner cleanup failed")

    async def run() -> BaseExceptionGroup:
        result = await _virtual_factory(adapter=_RecordingAdapter()).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_revocation_cancelled",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(None, runner, session_id="sess_finalize_revocation_cancelled")

        async def cancelled_revoke() -> bool:
            return True

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.revoke_authority = cancelled_revoke  # type: ignore[method-assign]
        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await binding.finalize(bound, outcome="interrupted")
        return exc_info.value

    failure = asyncio.run(run())

    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert failure.exceptions[1] is cleanup_error
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "cancellation",
        "managed_resource_cleanup",
    ]


def test_app_persists_dual_finalize_failures_and_retry_converges_once() -> None:
    long_tail = "界" * 300
    workspace_error = RuntimeError(f"workspace finalization failed: {REAL_SECRET}: {long_tail}")
    cleanup_error = RuntimeError(f"suspend failed: {REAL_SECRET}: {long_tail}")
    adapter = _RetryingLifecycleAdapter(first_error=cleanup_error)
    egress_events: list[Event] = []

    class _FailingBinding(WorkspaceBinding):
        def __init__(self) -> None:
            self.bound: BoundWorkspace | None = None
            self.finalize_calls = 0

        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            self.bound = BoundWorkspace(runner=runner)
            return self.bound

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            self.finalize_calls += 1
            raise workspace_error

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    inner_binding = _FailingBinding()

    async def emit(event: Event) -> Event:
        egress_events.append(event)
        return event

    async def run() -> tuple[list[Event], Runner, WorkspaceBinding, BoundWorkspace]:
        store = InMemorySessionStore()
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=inner_binding,
            event_emitter=emit,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_dual_durable",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(REAL_SECRET),
            enable_logging=False,
        )
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_finalize_dual_durable",
                    messages=[Message.text("user", "finish")],
                )
            )
        ]
        assert inner_binding.bound is not None
        return (
            await store.load_events("sess_finalize_dual_durable"),
            runner,
            binding,
            inner_binding.bound,
        )

    events, runner, binding, bound = asyncio.run(run())
    finalize_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    )
    terminal = next(event for event in events if event.type is EventType.SESSION_COMPLETED)
    assert finalize_failed.payload["error_type"] == "ExceptionGroup"
    durable_failures = finalize_failed.payload["failures"]
    assert [item["phase"] for item in durable_failures] == [
        "workspace_finalize",
        "managed_resource_cleanup",
    ]
    assert all(item["error_type"] == "RuntimeError" for item in durable_failures)
    assert all(REDACTED_SECRET in item["error"] for item in durable_failures)
    assert all(REAL_SECRET not in item["error"] for item in durable_failures)
    assert all(
        len(item["error"].encode("utf-8")) <= BINDING_FINALIZE_ERROR_TEXT_MAX_BYTES
        for item in durable_failures
    )
    assert all(item["error"].endswith("... [truncated]") for item in durable_failures)
    assert terminal.payload["binding_finalize_error"] == {
        "error": finalize_failed.payload["error"],
        "error_type": "ExceptionGroup",
        "outcome": "completed",
        "failures": durable_failures,
    }
    assert REAL_SECRET not in str(finalize_failed.payload)
    assert REAL_SECRET not in str(terminal.payload["binding_finalize_error"])
    assert runner.closed is False
    assert adapter.torn_down == 0

    async def retry() -> BaseException:
        with pytest.raises(RuntimeError) as exc_info:
            await binding.finalize(bound, outcome="completed")
        return exc_info.value

    retry_error = asyncio.run(retry())

    assert retry_error is workspace_error
    assert REAL_SECRET in str(retry_error)
    assert str(cleanup_error).endswith(long_tail)
    assert runner.closed is True
    assert adapter.finalize_calls == 2
    assert adapter.torn_down == 1
    assert inner_binding.finalize_calls == 2
    assert sum(event.type is EventType.EGRESS_GRANT_REVOKED for event in egress_events) == 1


def test_app_redacts_factory_owned_virtual_credential_from_finalize_diagnostics() -> None:
    adapter = _VirtualCredentialEchoingAdapter()

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> tuple[list[Event], str, RuntimeError]:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_virtual_credential_finalize_redaction",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner_request = adapter.captured["runner_request"]
        presented_value = runner_request.env_overlay["STRIPE_SECRET_KEY"]
        app = CayuApp(enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_virtual_credential_finalize_redaction",
                    messages=[Message.text("user", "finish")],
                )
            )
        ]
        assert adapter.cleanup_error is not None
        return events, presented_value, adapter.cleanup_error

    events, presented_value, cleanup_error = asyncio.run(run())

    finalize_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    )
    terminal = next(event for event in events if event.type is EventType.SESSION_COMPLETED)
    terminal_diagnostic = terminal.payload["binding_finalize_error"]

    assert presented_value.startswith("sk_test_cayu_vc_")
    assert presented_value in str(cleanup_error)
    assert presented_value not in str(finalize_failed.payload)
    assert presented_value not in str(terminal_diagnostic)
    assert REDACTED_SECRET in finalize_failed.payload["error"]
    assert REDACTED_SECRET in finalize_failed.payload["failures"][0]["error"]
    assert terminal_diagnostic["error"] == finalize_failed.payload["error"]
    assert terminal_diagnostic["failures"] == finalize_failed.payload["failures"]


def test_supplemental_finalize_redactor_survives_cancellation_aggregation() -> None:
    presented_value = "sk_test_cayu_vc_exact_presented_value"
    configured_secret = "cayu"
    failure = RuntimeError(f"cleanup echoed {presented_value} and {configured_secret}")
    record_binding_finalize_failures(
        failure,
        (
            BindingFinalizeFailure(
                phase="managed_resource_cleanup",
                error=failure,
            ),
        ),
        supplemental_redactor=SecretRedactor(presented_value),
    )

    aggregate = append_binding_finalize_cancellation(
        failure,
        asyncio.CancelledError("caller cancelled"),
    )
    payload = binding_finalize_failure_payload(
        aggregate,
        redactor=SecretRedactor(configured_secret),
    )

    assert payload is not None
    assert [item["phase"] for item in payload] == [
        "managed_resource_cleanup",
        "cancellation",
    ]
    assert presented_value not in str(payload)
    assert "sk_test_" not in str(payload)
    assert configured_secret not in str(payload)
    assert REDACTED_SECRET in payload[0]["error"]
    assert presented_value in str(failure)
    assert configured_secret in str(failure)


def test_app_persists_phase_evidence_before_propagating_finalize_cancellation() -> None:
    workspace_started = asyncio.Event()
    cleanup_error = RuntimeError("runner cleanup failed")

    class _CancellingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            workspace_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> tuple[BaseExceptionGroup, list[Event]]:
        store = InMemorySessionStore()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_CancellingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_cancelled_durable",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def collect_events() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_finalize_cancelled_durable",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]

        task = asyncio.create_task(collect_events())
        await workspace_started.wait()
        task.cancel()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return (
            exc_info.value,
            await store.load_events("sess_finalize_cancelled_durable"),
        )

    failure, events = asyncio.run(run())
    finalize_failed = next(
        event for event in events if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    )

    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert failure.exceptions[1] is cleanup_error
    assert finalize_failed.payload["failures"] == [
        {
            "phase": "workspace_finalize",
            "error": "",
            "error_type": "CancelledError",
        },
        {
            "phase": "managed_resource_cleanup",
            "error": "runner cleanup failed",
            "error_type": "RuntimeError",
        },
    ]
    assert all(event.type is not EventType.SESSION_COMPLETED for event in events)


def test_finalize_preserves_phase_failures_when_cancelled_during_revocation_event() -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    cleanup_error = RuntimeError("runner cleanup failed")
    emit_started = asyncio.Event()
    allow_emit = asyncio.Event()
    emitted: list[Event] = []

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    async def emit(event: Event) -> Event:
        if event.type == EventType.EGRESS_GRANT_REVOKED:
            emit_started.set()
            await allow_emit.wait()
        emitted.append(event)
        return event

    async def run() -> BaseExceptionGroup:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
            event_emitter=emit,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_revocation_emit_cancelled",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            None,
            runner,
            session_id="sess_finalize_revocation_emit_cancelled",
        )

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        task = asyncio.create_task(binding.finalize(bound, outcome="interrupted"))
        await emit_started.wait()
        task.cancel()
        allow_emit.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0] is workspace_error
    assert failure.exceptions[1] is cleanup_error
    assert isinstance(failure.exceptions[2], asyncio.CancelledError)
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "managed_resource_cleanup",
        "cancellation",
    ]
    assert sum(event.type == EventType.EGRESS_GRANT_REVOKED for event in emitted) == 1


def test_finalize_preserves_phase_failures_when_revocation_emitter_cancels() -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    cleanup_error = RuntimeError("runner cleanup failed")

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    async def emit(event: Event) -> Event:
        if event.type == EventType.EGRESS_GRANT_REVOKED:
            raise asyncio.CancelledError()
        return event

    async def run() -> BaseExceptionGroup:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
            event_emitter=emit,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_revocation_emitter_cancelled",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            None,
            runner,
            session_id="sess_finalize_revocation_emitter_cancelled",
        )

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await binding.finalize(bound, outcome="interrupted")
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0] is workspace_error
    assert failure.exceptions[1] is cleanup_error
    assert isinstance(failure.exceptions[2], asyncio.CancelledError)
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "managed_resource_cleanup",
        "cancellation",
    ]


def test_finalize_preserves_phase_failures_when_revocation_emitter_groups_cancellation() -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    cleanup_error = RuntimeError("runner cleanup failed")
    emitter_error = RuntimeError("revocation diagnostic failed")

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    async def emit(event: Event) -> Event:
        if event.type == EventType.EGRESS_GRANT_REVOKED:
            raise BaseExceptionGroup(
                "revocation diagnostics cancelled",
                [asyncio.CancelledError(), emitter_error],
            )
        return event

    async def run() -> BaseExceptionGroup:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
            event_emitter=emit,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_revocation_emitter_grouped_cancel",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            None,
            runner,
            session_id="sess_finalize_revocation_emitter_grouped_cancel",
        )

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await binding.finalize(bound, outcome="interrupted")
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0] is workspace_error
    assert failure.exceptions[1] is cleanup_error
    assert isinstance(failure.exceptions[2], asyncio.CancelledError)
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "managed_resource_cleanup",
        "cancellation",
    ]


@pytest.mark.parametrize("failure_boundary", ["audit", "revocation"])
def test_finalize_propagates_fatal_member_from_diagnostic_group(
    failure_boundary: str,
) -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    cleanup_error = RuntimeError("runner cleanup failed")
    diagnostic_error = RuntimeError("revocation diagnostic failed")
    fatal_signal = KeyboardInterrupt("shutdown requested")

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    async def emit(event: Event) -> Event:
        if failure_boundary == "revocation" and event.type == EventType.EGRESS_GRANT_REVOKED:
            raise BaseExceptionGroup(
                "revocation diagnostics interrupted",
                [asyncio.CancelledError(), diagnostic_error, fatal_signal],
            )
        return event

    class _FailingAudit:
        async def drain(self) -> None:
            raise BaseExceptionGroup(
                "audit diagnostics interrupted",
                [asyncio.CancelledError(), diagnostic_error, fatal_signal],
            )

    async def run() -> BaseExceptionGroup:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
            event_emitter=emit,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_revocation_emitter_fatal_group",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        if failure_boundary == "audit":
            binding._audit = _FailingAudit()  # type: ignore[attr-defined]
        bound = await binding.bind(
            None,
            runner,
            session_id="sess_finalize_revocation_emitter_fatal_group",
        )

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await binding.finalize(bound, outcome="interrupted")
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0] is workspace_error
    assert failure.exceptions[1] is cleanup_error
    assert binding_finalize_fatal_signal(failure) is not None
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "managed_resource_cleanup",
    ]


def test_revocation_emission_retry_does_not_duplicate_partially_emitted_grants() -> None:
    revoked_events: list[Event] = []
    cancel_second = True
    secret_names = ("stripe_test_key", "stripe_test_key_2", "stripe_test_key_3")
    credentials = [
        VirtualCredentialSpec(
            env_name=f"STRIPE_SECRET_KEY_{index}",
            secret=SecretRef(name=secret_name),
            destination="api.stripe.com",
            policy_name=POLICY_NAME,
        )
        for index, secret_name in enumerate(secret_names, start=1)
    ]

    async def emit(event: Event) -> Event:
        nonlocal cancel_second
        if event.type == EventType.EGRESS_GRANT_REVOKED:
            revoked_events.append(event)
            if len(revoked_events) == 2 and cancel_second:
                cancel_second = False
                # Model a committed event whose acknowledgement is lost.
                raise asyncio.CancelledError()
        return event

    async def run() -> tuple[Runner, int]:
        adapter = _RecordingAdapter()
        result = await _virtual_factory(
            adapter=adapter,
            credentials=credentials,
            resolver=StaticVault({name: REAL_SECRET for name in secret_names}),
            event_emitter=emit,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_partial_revocation_emission",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            None,
            runner,
            session_id="sess_partial_revocation_emission",
        )

        with pytest.raises(asyncio.CancelledError):
            await binding.finalize(bound, outcome="completed")
        await binding.finalize(bound, outcome="completed")
        return runner, adapter.torn_down

    runner, teardown_calls = asyncio.run(run())

    revoked_grant_ids = [event.payload["grant_id"] for event in revoked_events]
    assert len(revoked_grant_ids) == len(credentials)
    assert len(set(revoked_grant_ids)) == len(credentials)
    assert runner.closed is True
    assert teardown_calls == 1


@pytest.mark.parametrize("blocked_boundary", ["store", "sink"])
def test_app_defers_cancellation_until_finalize_failure_is_durable(
    blocked_boundary: str,
) -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    cleanup_error = RuntimeError("runner cleanup failed")

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class _BlockingFinalizeFailureStore(InMemorySessionStore):
        def __init__(self, *, block: bool) -> None:
            super().__init__()
            self.block = block
            self.append_started = asyncio.Event()
            self.allow_append = asyncio.Event()

        async def append_event(self, session_id: str, event: Event) -> None:
            if self.block and event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                self.append_started.set()
                await self.allow_append.wait()
            await super().append_event(session_id, event)

    class _BlockingFinalizeFailureSink(EventSink):
        def __init__(self) -> None:
            self.emit_started = asyncio.Event()

        async def emit(self, event: Event) -> None:
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                self.emit_started.set()
                await asyncio.Event().wait()

    async def run() -> tuple[BaseExceptionGroup, list[Event]]:
        store = _BlockingFinalizeFailureStore(block=blocked_boundary == "store")
        sink = _BlockingFinalizeFailureSink()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_failure_emit_cancelled",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        app = CayuApp(
            session_store=store,
            event_sinks=[sink] if blocked_boundary == "sink" else (),
            enable_logging=False,
        )
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def collect_events() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_finalize_failure_emit_cancelled",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]

        task = asyncio.create_task(collect_events())
        if blocked_boundary == "store":
            await store.append_started.wait()
        else:
            await sink.emit_started.wait()
        task.cancel()
        if blocked_boundary == "store":
            store.allow_append.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await asyncio.wait_for(task, timeout=1)
        return (
            exc_info.value,
            await store.load_events("sess_finalize_failure_emit_cancelled"),
        )

    failure, events = asyncio.run(run())

    assert failure.exceptions[0] is workspace_error
    assert failure.exceptions[1] is cleanup_error
    assert isinstance(failure.exceptions[2], asyncio.CancelledError)
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "managed_resource_cleanup",
        "cancellation",
    ]
    finalize_failed = next(
        event for event in events if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    )
    assert [item["phase"] for item in finalize_failed.payload["failures"]] == [
        "workspace_finalize",
        "managed_resource_cleanup",
    ]
    assert all(event.type != EventType.SESSION_COMPLETED for event in events)


@pytest.mark.parametrize("failure_point", ["before_commit", "after_commit"])
def test_app_preserves_finalize_failures_when_durable_evidence_write_fails(
    failure_point: str,
) -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    cleanup_error = RuntimeError("runner cleanup failed")
    persistence_error = RuntimeError("finalize failure event unavailable")

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class _FailingFinalizeFailureStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type != EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                await super().append_event(session_id, event)
                return
            if failure_point == "after_commit":
                await super().append_event(session_id, event)
            raise persistence_error

    async def run() -> tuple[BaseExceptionGroup | None, list[Event]]:
        store = _FailingFinalizeFailureStore()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id=f"sess_finalize_evidence_{failure_point}",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        failure: BaseExceptionGroup | None = None
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=f"sess_finalize_evidence_{failure_point}",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]
        except BaseExceptionGroup as exc:
            failure = exc
        return (
            failure,
            await store.load_events(f"sess_finalize_evidence_{failure_point}"),
        )

    failure, events = asyncio.run(run())

    if failure_point == "before_commit":
        assert failure is not None
        assert failure.exceptions == (workspace_error, cleanup_error)
        assert failure.__cause__ is persistence_error
        assert any("durable failure publication also failed" in note for note in failure.__notes__)
    else:
        assert failure is None
    expected_finalize_failed = 1 if failure_point == "after_commit" else 0
    assert (
        sum(event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED for event in events)
        == expected_finalize_failed
    )
    terminal_events = [
        event
        for event in events
        if event.type in {EventType.SESSION_COMPLETED, EventType.SESSION_FAILED}
    ]
    if failure_point == "before_commit":
        assert terminal_events == []
    else:
        assert len(terminal_events) == 1
        assert terminal_events[0].type == EventType.SESSION_COMPLETED
        assert [
            item["phase"]
            for item in terminal_events[0].payload["binding_finalize_error"]["failures"]
        ] == ["workspace_finalize", "managed_resource_cleanup"]


@pytest.mark.parametrize("failure_point", ["before_commit", "after_commit"])
def test_app_reconciles_finalize_evidence_child_task_cancellation(
    failure_point: str,
) -> None:
    workspace_error = RuntimeError("workspace finalization failed")

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class _CancellingStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type != EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                await super().append_event(session_id, event)
                return
            if failure_point == "after_commit":
                await super().append_event(session_id, event)
            raise asyncio.CancelledError("persistence child cancelled")

    async def run() -> tuple[BaseExceptionGroup | None, list[Event]]:
        store = _CancellingStore()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id=f"sess_finalize_child_cancel_{failure_point}",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        failure: BaseExceptionGroup | None = None
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=f"sess_finalize_child_cancel_{failure_point}",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]
        except BaseExceptionGroup as exc:
            failure = exc
        return (
            failure,
            await store.load_events(f"sess_finalize_child_cancel_{failure_point}"),
        )

    failure, events = asyncio.run(run())

    if failure_point == "before_commit":
        assert failure is not None
        assert failure.exceptions[0] is workspace_error
        assert isinstance(failure.exceptions[1], asyncio.CancelledError)
        assert all(event.type != EventType.SESSION_COMPLETED for event in events)
    else:
        assert failure is None
        assert (
            sum(event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED for event in events)
            == 1
        )
        assert sum(event.type == EventType.SESSION_COMPLETED for event in events) == 1


def test_app_preserves_caller_cancellation_after_child_persistence_commits() -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    committed = asyncio.Event()
    allow_child_cancellation = asyncio.Event()

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class _CommitThenCancelStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            await super().append_event(session_id, event)
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                committed.set()
                await allow_child_cancellation.wait()
                raise asyncio.CancelledError("persistence child cancelled after commit")

    async def run() -> tuple[BaseExceptionGroup, list[Event]]:
        store = _CommitThenCancelStore()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_child_and_caller_cancel",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def collect() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_finalize_child_and_caller_cancel",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await committed.wait()
        task.cancel()
        allow_child_cancellation.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return (
            exc_info.value,
            await store.load_events("sess_finalize_child_and_caller_cancel"),
        )

    failure, events = asyncio.run(run())

    assert failure.exceptions[0] is workspace_error
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)
    assert sum(event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED for event in events) == 1
    assert all(event.type != EventType.SESSION_COMPLETED for event in events)


def test_app_ignores_stale_causal_cancellation_after_finalize_evidence_commits() -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    old_cancellation = asyncio.CancelledError("already handled")

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class _CommitThenFailStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            await super().append_event(session_id, event)
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                raise RuntimeError("acknowledgement lost") from old_cancellation

    async def run() -> list[Event]:
        store = _CommitThenFailStore()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_stale_causal_cancel",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        streamed = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_finalize_stale_causal_cancel",
                    messages=[Message.text("user", "finish")],
                )
            )
        ]
        assert streamed[-1].type == EventType.SESSION_COMPLETED
        return await store.load_events("sess_finalize_stale_causal_cancel")

    events = asyncio.run(run())

    assert sum(event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED for event in events) == 1
    assert sum(event.type == EventType.SESSION_COMPLETED for event in events) == 1


def test_app_preserves_child_cancellation_when_reconciliation_fails() -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    reconciliation_error = RuntimeError("reconciliation unavailable")

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class _BrokenReconciliationStore(InMemorySessionStore):
        reconcile = False

        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                self.reconcile = True
                raise asyncio.CancelledError("persistence child cancelled")
            await super().append_event(session_id, event)

        async def query_events(self, query=None):  # type: ignore[no-untyped-def]
            if self.reconcile:
                raise reconciliation_error
            return await super().query_events(query)

    async def run() -> BaseExceptionGroup:
        store = _BrokenReconciliationStore()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_child_cancel_reconcile_failure",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(BaseExceptionGroup) as exc_info:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_finalize_child_cancel_reconcile_failure",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0] is workspace_error
    assert isinstance(failure.exceptions[1], asyncio.CancelledError)
    assert failure.__cause__ is not None
    assert failure.__cause__.__cause__ is reconciliation_error


def test_append_finalize_cancellation_ignores_old_causal_cancellation() -> None:
    old_cancellation = asyncio.CancelledError("old cancellation converted by binding")
    new_cancellation = asyncio.CancelledError("new caller cancellation")
    workspace_error: RuntimeError | None = None

    try:
        try:
            raise old_cancellation
        except asyncio.CancelledError:
            raise RuntimeError("workspace failure") from old_cancellation
    except RuntimeError as error:
        workspace_error = error
        aggregate = append_binding_finalize_cancellation(error, new_cancellation)

    assert workspace_error is not None
    assert isinstance(aggregate, BaseExceptionGroup)
    assert aggregate.exceptions == (workspace_error, new_cancellation)
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(aggregate, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "cancellation",
    ]


def test_app_preserves_cancellation_when_finalize_evidence_write_fails() -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    cleanup_error = RuntimeError("runner cleanup failed")
    persistence_error = RuntimeError("finalize failure event unavailable")
    persist_started = asyncio.Event()
    allow_persist_failure = asyncio.Event()

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class _BlockingFailingStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type != EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                await super().append_event(session_id, event)
                return
            persist_started.set()
            await allow_persist_failure.wait()
            raise persistence_error

    async def run() -> BaseExceptionGroup:
        store = _BlockingFailingStore()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_evidence_cancelled",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def collect_events() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_finalize_evidence_cancelled",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]

        task = asyncio.create_task(collect_events())
        await persist_started.wait()
        task.cancel()
        allow_persist_failure.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0] is workspace_error
    assert failure.exceptions[1] is cleanup_error
    assert isinstance(failure.exceptions[2], asyncio.CancelledError)
    assert isinstance(failure.__cause__, BaseExceptionGroup)
    assert failure.__cause__.exceptions[0] is persistence_error
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "managed_resource_cleanup",
        "cancellation",
    ]
    assert any("durable failure publication also failed" in note for note in failure.__notes__)


def test_app_does_not_duplicate_finalize_cancellation_when_evidence_write_fails() -> None:
    cleanup_error = RuntimeError("runner cleanup failed")
    persistence_error = RuntimeError("finalize failure event unavailable")
    finalize_started = asyncio.Event()

    class _CancellingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            finalize_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class _FailingStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                raise persistence_error
            await super().append_event(session_id, event)

    async def run() -> BaseExceptionGroup:
        store = _FailingStore()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_CancellingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_cancelled_evidence_failure",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def collect_events() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_finalize_cancelled_evidence_failure",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]

        task = asyncio.create_task(collect_events())
        await finalize_started.wait()
        task.cancel()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await task
        return exc_info.value

    failure = asyncio.run(run())

    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert failure.exceptions[1] is cleanup_error
    assert failure.__cause__ is persistence_error
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "managed_resource_cleanup",
    ]
    assert sum(isinstance(exc, asyncio.CancelledError) for exc in failure.exceptions) == 1
    assert any("durable failure publication also failed" in note for note in failure.__notes__)


def test_app_preserves_grouped_cancellation_when_finalize_evidence_write_fails() -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    cleanup_error = RuntimeError("runner cleanup failed")
    persistence_error = RuntimeError("finalize failure event unavailable")

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class _FailingStore(InMemorySessionStore):
        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                raise BaseExceptionGroup(
                    "publication diagnostics cancelled",
                    [asyncio.CancelledError(), persistence_error],
                )
            await super().append_event(session_id, event)

    async def run() -> BaseExceptionGroup:
        store = _FailingStore()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_grouped_cancel_evidence_failure",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None

        async def fail_cleanup(*, outcome: str | None) -> None:
            raise cleanup_error

        runner.finalize = fail_cleanup  # type: ignore[method-assign]
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(BaseExceptionGroup) as exc_info:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_finalize_grouped_cancel_evidence_failure",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions[0] is workspace_error
    assert failure.exceptions[1] is cleanup_error
    assert isinstance(failure.exceptions[2], asyncio.CancelledError)
    assert [
        item["phase"]
        for item in binding_finalize_failure_payload(failure, redactor=SecretRedactor()) or []
    ] == [
        "workspace_finalize",
        "managed_resource_cleanup",
        "cancellation",
    ]
    assert any("durable failure publication also failed" in note for note in failure.__notes__)


@pytest.mark.parametrize("failure_boundary", ["store", "reconciliation", "sink"])
def test_app_propagates_fatal_member_from_finalize_evidence_diagnostic_group(
    failure_boundary: str,
) -> None:
    workspace_error = RuntimeError("workspace finalization failed")
    persistence_error = RuntimeError("finalize failure event unavailable")
    fatal_signal = KeyboardInterrupt("shutdown requested")

    class _FailingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise workspace_error

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    class _FailingStore(InMemorySessionStore):
        reconcile_finalize_failure = False

        async def append_event(self, session_id: str, event: Event) -> None:
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
                if failure_boundary == "store":
                    raise BaseExceptionGroup(
                        "publication diagnostics interrupted",
                        [asyncio.CancelledError(), persistence_error, fatal_signal],
                    )
                if failure_boundary == "reconciliation":
                    self.reconcile_finalize_failure = True
                    raise persistence_error
            await super().append_event(session_id, event)

        async def query_events(self, query=None):  # type: ignore[no-untyped-def]
            if self.reconcile_finalize_failure:
                raise BaseExceptionGroup(
                    "reconciliation diagnostics interrupted",
                    [asyncio.CancelledError(), persistence_error, fatal_signal],
                )
            return await super().query_events(query)

    class _FailingSink(EventSink):
        async def emit(self, event: Event) -> None:
            if (
                failure_boundary == "sink"
                and event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
            ):
                raise BaseExceptionGroup(
                    "fan-out diagnostics interrupted",
                    [asyncio.CancelledError(), persistence_error, fatal_signal],
                )

    async def run() -> BaseExceptionGroup:
        store = _FailingStore()
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=_FailingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_finalize_fatal_evidence_failure",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        app = CayuApp(
            session_store=store,
            event_sinks=[_FailingSink()] if failure_boundary == "sink" else (),
            enable_logging=False,
        )
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(BaseExceptionGroup) as exc_info:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_finalize_fatal_evidence_failure",
                        messages=[Message.text("user", "finish")],
                    )
                )
            ]
        return exc_info.value

    failure = asyncio.run(run())

    assert failure.exceptions == (fatal_signal,)


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
