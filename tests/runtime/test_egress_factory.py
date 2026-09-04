from __future__ import annotations

import asyncio
import contextlib
import warnings
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from tests.environments.sync_ownership_assertions import assert_sync_resources_owned
from tests.provider_traceback_assertions import assert_cayu_traceback_does_not_retain
from tests.runners.lambda_microvm_harness import (
    ConformanceLambdaClient,
    SupervisorTransport,
)

from cayu import CayuConfig, OperationsConfig
from cayu._exception_groups import iter_exception_tree
from cayu.artifacts import LocalArtifactStore
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.egress import (
    ApprovedEgressDestination,
    CapturedRequest,
    CapturedResponse,
    CredentialMode,
    EgressAdapterRegistry,
    EgressAuthorityCutoverStrategy,
    EgressBinding,
    EgressCapabilityClaim,
    EgressCapabilityEvidence,
    EgressUpstreamLimits,
    EgressUpstreamOperation,
    HttpEgressPolicy,
    InvalidEgressReconnectMetadataError,
    RunnerFinalizationResult,
    SandboxEgressAdapter,
    TransparentEgressBroker,
    UnsupportedEgressAdapter,
    UnsupportedEgressError,
    UnsupportedEgressReconnectError,
    VirtualCredentialError,
)
from cayu.environments import (
    EFSAccessPointBinding,
    Environment,
    EnvironmentAllocationContext,
    EnvironmentAllocationIntent,
    EnvironmentAllocationState,
    EnvironmentAllocationUnsupportedError,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentSpec,
    ExecutionAdmissionError,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionEvidenceOverride,
    ExecutionRequirements,
    SyncBinding,
    SyncTargetWorkspacePlan,
)
from cayu.environments.bindings import BoundWorkspace, WorkspaceBinding
from cayu.environments.factory import (
    attach_environment_factory_cleanup_settlement_task,
    environment_factory_cleanup_settlement_task,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners import (
    DockerRunner,
    E2BRunner,
    LambdaMicroVMRunner,
    LocalRunner,
    MicrosandboxRunner,
    RunnerExecutionError,
)
from cayu.runners.base import ExecCommand, ExecResult, Runner
from cayu.runtime import CayuApp, InMemorySessionStore, RunRequest
from cayu.runtime._environment_lifecycle import (
    _persist_binding_finalize_failure_event,
    _reconcile_binding_finalize_failure_event,
)
from cayu.runtime.event_sinks import EventSink
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import InvocationRunnerHandle
from cayu.vaults import REDACTED_SECRET, SecretRedactor, SecretRef, StaticVault
from cayu.workspaces import (
    LocalWorkspace,
    RunnerBoundWorkspace,
    WorkspaceListResult,
    WorkspaceMutationResult,
    WorkspaceReadResult,
)

pytest.importorskip("cryptography")

from cayu.egress.adapter import (
    VirtualEgressRunnerRequest,
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
    _EgressManagedRunner,
    _EgressTeardownBinding,
    _workspace_dispatch_settlement_kind,
)
from cayu.testing import verify_provider_credential_isolation

REAL_SECRET = "sk_test_51FactoryRealSecret"
POLICY_NAME = "provider-example"


def test_egress_wrapper_propagates_sync_completion_recovery_state(
    tmp_path: Path,
) -> None:
    class RecordingBindingFence:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool | None, str | None]] = []

        def begin_workspace_binding(self) -> None:
            self.calls.append(("begin", None, None))

        def finish_workspace_binding(
            self,
            *,
            require_mutation_quiescence: bool,
            workspace_owner_key: str | None = None,
        ) -> None:
            self.calls.append(
                (
                    "finish",
                    require_mutation_quiescence,
                    workspace_owner_key,
                )
            )

    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    inner = SyncBinding(target_workspace=target)
    bound = asyncio.run(inner.bind(source, None, session_id="wrapped-sync"))
    wrapper = object.__new__(_EgressTeardownBinding)
    wrapper._inner = inner

    assert wrapper._completion_requires_successful_finalization(bound) is True
    recovery_state = wrapper._completion_finalization_recovery_state(bound)
    assert recovery_state is not None
    assert recovery_state["generation"] == bound.state_key

    inner.abandon(bound)
    recovered_inner = SyncBinding(target_workspace=target)
    recovered_wrapper = object.__new__(_EgressTeardownBinding)
    recovered_wrapper._inner = recovered_inner
    fence = RecordingBindingFence()
    recovered_wrapper._runner = fence
    recovered = asyncio.run(
        recovered_wrapper._recover_completion_finalization(
            source,
            None,
            session_id="wrapped-sync",
            agent_name="agent",
            environment_name="environment",
            recovery_state=recovery_state,
        )
    )

    assert fence.calls == [
        ("begin", None, None),
        ("finish", True, recovered.state_key),
    ]
    recovered_inner.abandon(recovered)


class _ClosedRejectingRunnerWorkspace(RunnerBoundWorkspace):
    def __init__(
        self,
        runner: Runner,
        delegate: LocalWorkspace,
        *,
        workspace_id: str,
    ) -> None:
        self.id = workspace_id
        self._runner = runner
        self._delegate = delegate
        self.operations_after_close = 0
        self.next_list_error: BaseException | None = None

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("test-runner-workspace", self.id)

    @property
    def bound_runner_resource_key(self) -> tuple[object, ...]:
        return ("test-runner", id(self._runner))

    @property
    def runner_cwd(self) -> str:
        return self._runner.default_cwd

    def is_bound_to_runner(self, runner: Runner) -> bool:
        return self._runner is runner

    def _control_plane_runner(self) -> Runner:
        return self._runner

    def bounded_read_limit(self, max_bytes: int) -> int:
        return self._delegate.bounded_read_limit(max_bytes)

    def _require_open(self) -> None:
        if self._runner.is_closed:
            self.operations_after_close += 1
            raise RuntimeError("runner-backed target is closed")

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        self._require_open()
        return await self._delegate.read_bytes(
            path,
            offset=offset,
            max_bytes=max_bytes,
        )

    async def write_bytes(self, path: str, content: bytes) -> None:
        self._require_open()
        await self._delegate.write_bytes(path, content)

    async def delete(self, path: str) -> None:
        self._require_open()
        await self._delegate.delete(path)

    async def create_bytes(self, path: str, content: bytes) -> WorkspaceMutationResult:
        self._require_open()
        return await self._delegate.create_bytes(path, content)

    async def replace_bytes(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        self._require_open()
        return await self._delegate.replace_bytes(
            path,
            content,
            expected_revision=expected_revision,
        )

    async def delete_if_revision(
        self,
        path: str,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        self._require_open()
        return await self._delegate.delete_if_revision(
            path,
            expected_revision=expected_revision,
        )

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        self._require_open()
        if self.next_list_error is not None:
            error = self.next_list_error
            self.next_list_error = None
            raise error
        return await self._delegate.list(pattern, limit=limit)


class _FailingLocalWorkspace(LocalWorkspace):
    def __init__(self, root: Path, *, workspace_id: str) -> None:
        super().__init__(root, workspace_id=workspace_id)
        self.fail_write_call: int | None = None
        self.fail_delete_call: int | None = None
        self.write_calls = 0
        self.delete_calls = 0

    async def write_bytes(self, path: str, content: bytes) -> None:
        self.write_calls += 1
        if self.write_calls == self.fail_write_call:
            raise OSError(f"injected durable write failure for {path}")
        await super().write_bytes(path, content)

    async def delete(self, path: str) -> None:
        self.delete_calls += 1
        if self.delete_calls == self.fail_delete_call:
            raise OSError(f"injected durable delete failure for {path}")
        await super().delete(path)


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


@pytest.mark.parametrize("bounded", [False, True])
def test_cleanup_wait_preserves_explicit_unwinding_cancellation(
    bounded: bool,
) -> None:
    cleanup_error = RuntimeError("cleanup failed")
    failures: list[BaseExceptionGroup] = []

    async def cleanup() -> None:
        raise cleanup_error

    async def run() -> None:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("unwinding cancellation")
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as cancellation:
            child = asyncio.create_task(cleanup())
            try:
                if bounded:
                    await _await_bounded_cleanup_task(
                        child,
                        timeout_s=1,
                        timeout_message="cleanup timed out",
                        cancellation=cancellation,
                    )
                else:
                    await _await_cleanup_task(child, cancellation=cancellation)
            except BaseExceptionGroup as failure:
                failures.append(failure)
            raise

    with pytest.raises(asyncio.CancelledError, match="unwinding cancellation"):
        asyncio.run(run())

    assert len(failures) == 1
    failure = failures[0]
    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert failure.exceptions[0].args == ("unwinding cancellation",)
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
def test_cleanup_wait_ignores_handled_historical_cancellation(bounded: bool) -> None:
    async def run() -> bool:
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(0)
        child = asyncio.create_task(asyncio.sleep(0))
        try:
            if bounded:
                result = await _await_bounded_cleanup_task(
                    child,
                    timeout_s=1,
                    timeout_message="cleanup timed out",
                )
            else:
                result = await _await_cleanup_task(child)
            assert current.cancelling() == 1
            return result
        finally:
            current.uncancel()

    assert asyncio.run(run()) is False


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


def test_finalize_evidence_persistence_ignores_handled_historical_cancellation() -> None:
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
            result = await _persist_binding_finalize_failure_event(  # type: ignore[arg-type]
                _Writer(),
                event,
            )
            assert current.cancelling() == 1
            return result
        finally:
            current.uncancel()

    persisted, cancellation = asyncio.run(run())

    assert persisted is event
    assert cancellation is None


class _FakeDocker:
    last_instance: _FakeDocker | None = None

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.container_id = "a" * 64
        type(self).last_instance = self

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        self.calls.append(list(argv))
        if list(argv[:4]) == ["inspect", "--type", "container", "--format"]:
            return 0, self.container_id
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


class _ExecutingVirtualRunner(Runner):
    """Hermetic subprocess runner that applies the factory's virtual env overlay."""

    isolation = "lambda-microvm"

    def __init__(self, root: Path, env_overlay: Mapping[str, str]) -> None:
        self._local = LocalRunner(root, credential_mode=CredentialMode.VIRTUAL_EGRESS)
        self._env_overlay = dict(env_overlay)
        self.default_cwd = str(root)

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        merged = {"HOME": str(self._local.root), **self._env_overlay}
        for name, value in (env or {}).items():
            if name in merged and merged[name] != value:
                raise ValueError(f"Virtual runner env collision: {name}")
            merged[name] = value
        return await self._local.exec(
            command,
            cwd=cwd,
            env=merged,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )

    async def close(self) -> None:
        await self._local.close()


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
    process_external_allocation = False

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

    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        del outcome
        await runner.close()
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)


class _LifecycleRecordingAdapter(_RecordingAdapter):
    supports_reconnect = True

    def __init__(self, *, runner_factory: Any = None) -> None:
        super().__init__("lambda-microvm", runner_factory=runner_factory)
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

    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        self.finalize_calls.append(outcome)
        await runner.close()
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)


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

    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        self.finalize_calls += 1
        if self.finalize_calls == 1:
            raise self.first_error
        await runner.close()
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)


class _RetryingReconnectAdapter(_LifecycleRecordingAdapter):
    async def finalize_runner(
        self,
        runner: Runner,
        *,
        outcome: str | None,
    ) -> RunnerFinalizationResult:
        self.finalize_calls.append(outcome)
        if len(self.finalize_calls) == 1:
            raise RuntimeError("suspend failed")
        await runner.close()
        return RunnerFinalizationResult(workspace_mutations_quiescent=True)


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

    docker = _FakeDocker()
    adapter = DockerEgressAdapter(
        docker_exec=docker,
        docker_run=docker,
        proxy_host="127.0.0.1",
    )
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
        def prepare(
            self,
            request: CapturedRequest,
            *,
            limits: EgressUpstreamLimits,
        ) -> EgressUpstreamOperation:
            assert isinstance(request, CapturedRequest)
            assert limits.max_response_bytes > 0

            async def send() -> CapturedResponse:
                return CapturedResponse(status_code=200, body=b"{}")

            return EgressUpstreamOperation(send)

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
            session_id="sess_1",
            agent_name="agent",
            environment_name="egress-env",
            execution_profile_fingerprint="a" * 64,
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
    fake_docker = _FakeDocker.last_instance
    assert fake_docker is not None
    assert [
        "inspect",
        "--type",
        "container",
        "--format",
        "{{.Id}}",
        inner_runner.name,
    ] in fake_docker.calls
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
        assert event.payload["execution_profile_fingerprint"] == "a" * 64
        assert REAL_SECRET not in str(event.payload)


@pytest.mark.parametrize(
    ("credentials", "expected_present"),
    (((), False), ((_credential_spec(),), True)),
)
def test_factory_propagates_overlay_secret_authority_through_managed_handle(
    credentials: tuple[VirtualCredentialSpec, ...],
    expected_present: bool,
) -> None:
    class _AuthorityRunner(_FakeDockerRunner):
        def output_secret_values_present(self) -> bool | None:
            return self.kwargs.get("env_overlay_secret_values_present")

    async def runner_factory(request: VirtualEgressRunnerRequest) -> Runner:
        return _AuthorityRunner(
            request.name,
            env_overlay_secret_values_present=request.env_overlay_secret_values_present,
        )

    adapter = _RecordingAdapter(runner_factory=runner_factory)

    async def run() -> tuple[bool | None, bool | None, bool | None]:
        result = await _virtual_factory(
            adapter=adapter,
            credentials=credentials,
            approved_destinations=(
                ApprovedEgressDestination(
                    destination="api.stripe.com",
                    policy_name=POLICY_NAME,
                ),
            ),
        ).create(
            EnvironmentFactoryRequest(
                session_id=f"sess_output_secret_authority_{expected_present}",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert isinstance(runner, _EgressManagedRunner)
        handle = InvocationRunnerHandle(
            runner,
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=SecretRedactor(),
            ),
        )
        try:
            request = adapter.captured["runner_request"]
            return (
                request.env_overlay_secret_values_present,
                runner.output_secret_values_present(),
                handle.output_secret_values_present(),
            )
        finally:
            await runner.close()

    assert asyncio.run(run()) == (
        expected_present,
        expected_present,
        expected_present,
    )


def test_egress_wrapped_runner_failure_preserves_backend_identity() -> None:
    async def run() -> dict[str, Any]:
        result = await _virtual_factory(adapter=_RecordingAdapter("docker")).create(
            EnvironmentFactoryRequest(
                session_id="sess_failure_adapter",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        handle = InvocationRunnerHandle(
            runner,
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=SecretRedactor(),
            ),
        )
        try:
            with pytest.raises(RunnerExecutionError) as exc_info:
                await handle.exec(ExecCommand.process("fail"))
            return exc_info.value.diagnostic
        finally:
            await runner.close()

    diagnostic = asyncio.run(run())

    assert diagnostic == {
        "type": "cayu.runner_execution_error.v1",
        "adapter": "docker",
        "status": "failed",
        "error_type": "NotImplementedError",
        "timed_out": False,
        "cancelled": False,
    }


def test_factory_rejects_unsupported_remote_allocation_before_adapter_mutation() -> None:
    async def run() -> None:
        adapter = _RecordingAdapter("remote-provider")
        adapter.process_external_allocation = True
        factory = _virtual_factory(adapter=adapter)
        request = EnvironmentFactoryRequest(
            session_id="remote-allocation",
            agent_name="agent",
            environment_name="egress-env",
        )

        with pytest.raises(
            EnvironmentAllocationUnsupportedError,
            match="does not support durable create-or-lookup recovery",
        ):
            factory.allocation_scope(request)
        assert adapter.prepare_calls == []
        assert "runner_request" not in adapter.captured

        with pytest.raises(
            EnvironmentAllocationUnsupportedError,
            match="does not support durable create-or-lookup recovery",
        ):
            await factory.create(request)
        assert adapter.prepare_calls == []
        assert "runner_request" not in adapter.captured

    asyncio.run(run())


def test_factory_uses_durable_allocation_context_for_external_runner_creation() -> None:
    class RecoverableAdapter(_RecordingAdapter):
        process_external_allocation = True
        allocation_provider = "fake-remote"
        allocation_adapter_generation = "v1"

        async def create_or_recover_runner(
            self,
            request: VirtualEgressRunnerRequest,
            *,
            allow_create: bool,
        ) -> Runner:
            self.captured["allow_create"] = allow_create
            return await self.create_runner(request)

    class AllocationContext(EnvironmentAllocationContext):
        def __init__(self) -> None:
            self._state = EnvironmentAllocationState.UNPREPARED
            self._intent = EnvironmentAllocationIntent(
                allocation_id=f"ealloc_{'a' * 32}",
                provider="fake-remote",
                adapter_generation="v1",
                session_id="recoverable-allocation",
                environment_name="egress-env",
                requested_operation=EnvironmentFactoryOperation.CREATE,
            )
            self._acknowledgement: dict[str, Any] | None = None

        @property
        def intent(self) -> EnvironmentAllocationIntent:
            return self._intent

        @property
        def state(self) -> EnvironmentAllocationState:
            return self._state

        @property
        def acknowledged_reconnect_metadata(self) -> dict[str, Any] | None:
            return self._acknowledgement

        async def prepare(self, provider_metadata: Mapping[str, Any]):
            self._intent = self._intent.with_provider_metadata(provider_metadata)
            self._state = EnvironmentAllocationState.PREPARED
            return self._intent

        async def mark_dispatched(self) -> None:
            self._state = EnvironmentAllocationState.DISPATCHED

        async def acknowledge(self, reconnect_metadata: Mapping[str, Any]) -> None:
            self._acknowledgement = dict(reconnect_metadata)
            self._state = EnvironmentAllocationState.ACKNOWLEDGED

        async def mark_reaping(self) -> bool:
            self._state = EnvironmentAllocationState.REAPING
            return True

        async def mark_reaped(self) -> None:
            self._state = EnvironmentAllocationState.REAPED

    async def run() -> None:
        adapter = RecoverableAdapter("fake-remote")
        factory = _virtual_factory(adapter=adapter)
        request = EnvironmentFactoryRequest(
            session_id="recoverable-allocation",
            agent_name="agent",
            environment_name="egress-env",
        )
        scope = factory.allocation_scope(request)
        assert scope is not None
        assert scope.provider == "fake-remote"
        assert scope.adapter_generation == "v1"
        allocation = AllocationContext()

        result = await factory.create_recoverable(request, allocation)

        assert allocation.state is EnvironmentAllocationState.ACKNOWLEDGED
        assert adapter.captured["allow_create"] is True
        runner_request = adapter.captured["runner_request"]
        assert runner_request.allocation_id == f"ealloc_{'a' * 32}"
        assert allocation.acknowledged_reconnect_metadata == result.reconnect_metadata
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(run())


def test_factory_rejects_unclassified_custom_adapter_before_mutation() -> None:
    class _UnclassifiedAdapter(_RecordingAdapter):
        process_external_allocation = None

    async def run() -> None:
        adapter = _UnclassifiedAdapter("custom-provider")
        factory = _virtual_factory(adapter=adapter)
        request = EnvironmentFactoryRequest(
            session_id="unclassified-allocation",
            agent_name="agent",
            environment_name="egress-env",
        )

        with pytest.raises(
            EnvironmentAllocationUnsupportedError,
            match="must explicitly classify",
        ):
            factory.allocation_scope(request)
        with pytest.raises(
            EnvironmentAllocationUnsupportedError,
            match="must explicitly classify",
        ):
            await factory.create(request)
        assert adapter.prepare_calls == []
        assert "runner_request" not in adapter.captured

    asyncio.run(run())


def test_factory_requires_a_credential() -> None:
    with pytest.raises(ValueError, match="at least one credential"):
        VirtualEgressEnvironmentFactory(
            resolver=StaticVault({}),
            policies={},
            credentials=[],
        )


def test_factory_accepts_a_configured_browser_response_limit() -> None:
    factory = _virtual_factory(
        runner_kind="docker",
        browser_max_response_bytes=16 * 1024 * 1024,
    )

    assert factory._browser_max_response_bytes == 16 * 1024 * 1024


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, TypeError),
        (0, ValueError),
        (64 * 1024 * 1024 + 1, ValueError),
    ],
)
def test_factory_rejects_invalid_browser_response_limits(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="max_response_bytes"):
        _virtual_factory(
            runner_kind="docker",
            browser_max_response_bytes=value,
        )


def test_factory_copies_an_application_execution_profile_identity() -> None:
    identity = ExecutionProfileBehaviorIdentity(
        name="tests:virtual-egress",
        behavior_version="1",
        implementation_version="2026.08.18",
    )

    factory = _virtual_factory(
        runner_kind="docker",
        execution_profile_identity=identity,
    )

    assert factory.execution_profile_identity == identity
    assert factory.execution_profile_identity is not identity


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


def test_virtual_egress_factory_delegates_only_the_explicit_workload_capability(
    provider_credential_canaries,
) -> None:
    adapter = _LifecycleRecordingAdapter()

    async def run() -> tuple[Any, Any]:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_provider_boundary",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        await runner.close()
        return result, adapter.captured["runner_request"]

    result, runner_request = asyncio.run(run())

    virtual_value = runner_request.env_overlay["STRIPE_SECRET_KEY"]
    assert virtual_value.startswith("sk_test_cayu_vc_")
    assert virtual_value != REAL_SECRET
    projected = repr(
        {
            "runner_request": runner_request,
            "result_metadata": result.metadata,
            "reconnect_metadata": result.reconnect_metadata,
            "runner_kwargs": adapter.captured["inner_runner"].kwargs,
        }
    )
    assert REAL_SECRET not in projected
    assert all(value not in projected for value in provider_credential_canaries.values.values())


def test_virtual_egress_executes_provider_isolation_probe_on_create_and_reconnect(
    provider_credential_canaries,
    tmp_path: Path,
) -> None:
    guest_root = tmp_path / "virtual-egress-guest"
    guest_root.mkdir()

    async def runner_factory(request: Any) -> Runner:
        return _ExecutingVirtualRunner(guest_root, request.env_overlay)

    adapter = _LifecycleRecordingAdapter(runner_factory=runner_factory)

    async def verify_result(result: Any) -> tuple[Any, str]:
        runner = result.environment.runner
        assert runner is not None
        virtual_value = adapter.captured["runner_request"].env_overlay["STRIPE_SECRET_KEY"]
        visible = await runner.exec(
            ExecCommand.process(
                "python3",
                "-c",
                "import os; print(os.environ['STRIPE_SECRET_KEY'])",
            )
        )
        assert visible.stdout.strip() == virtual_value
        evidence = await verify_provider_credential_isolation(
            runner,
            adapter="virtual_egress_lambda_microvm",
            scope="isolated_guest",
            provider_canaries=provider_credential_canaries.values,
            operational_env={
                "CAYU_PROBE_VISIBLE": provider_credential_canaries.positive_env[
                    "CAYU_PROBE_VISIBLE"
                ]
            },
            workload_env={"STRIPE_SECRET_KEY": virtual_value},
            guest_cwd=str(guest_root),
        )
        await runner.close()
        return evidence, virtual_value

    async def run() -> tuple[Any, Any, str, str]:
        created = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_virtual_create",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        created_evidence, created_virtual = await verify_result(created)
        reconnected = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_virtual_create",
                agent_name="agent",
                environment_name="egress-env",
                operation=EnvironmentFactoryOperation.RECONNECT,
                reconnect_metadata=created.reconnect_metadata,
            )
        )
        reconnect_evidence, reconnect_virtual = await verify_result(reconnected)
        return (
            created_evidence,
            reconnect_evidence,
            created_virtual,
            reconnect_virtual,
        )

    created_evidence, reconnect_evidence, created_virtual, reconnect_virtual = asyncio.run(run())

    assert created_evidence.status == "verified"
    assert reconnect_evidence.status == "verified"
    assert created_evidence.positive_controls == (
        "CAYU_PROBE_VISIBLE",
        "STRIPE_SECRET_KEY",
    )
    assert reconnect_evidence.positive_controls == created_evidence.positive_controls
    assert created_virtual.startswith("sk_test_cayu_vc_")
    assert reconnect_virtual.startswith("sk_test_cayu_vc_")
    assert created_virtual != REAL_SECRET
    assert reconnect_virtual != REAL_SECRET


def test_virtual_egress_redacted_exec_includes_generated_workload_credentials(
    tmp_path: Path,
) -> None:
    guest_root = tmp_path / "virtual-egress-redaction"
    guest_root.mkdir()

    async def runner_factory(request: Any) -> Runner:
        return _ExecutingVirtualRunner(guest_root, request.env_overlay)

    adapter = _LifecycleRecordingAdapter(runner_factory=runner_factory)

    async def run() -> tuple[ExecResult, str]:
        created = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_virtual_redaction",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = created.environment.runner
        assert runner is not None
        virtual_value = adapter.captured["runner_request"].env_overlay["STRIPE_SECRET_KEY"]
        try:
            result = await runner.exec_redacted(
                ExecCommand.process(
                    "python3",
                    "-c",
                    "import os; print(os.environ['STRIPE_SECRET_KEY'])",
                ),
                redactor=SecretRedactor(),
            )
        finally:
            await runner.close()
        return result, virtual_value

    result, virtual_value = asyncio.run(run())

    assert result.stdout.strip() == REDACTED_SECRET
    assert virtual_value not in result.stdout
    assert result.stdout_bytes == len(f"{virtual_value}\n".encode())


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
    "allocation_fingerprint",
    (True, "a" * 63, "A" * 64),
)
def test_factory_rejects_invalid_reconnect_allocation_fingerprint(
    allocation_fingerprint: Any,
) -> None:
    adapter = _LifecycleRecordingAdapter()

    async def run() -> None:
        with pytest.raises(
            InvalidEgressReconnectMetadataError,
            match="allocation fingerprint",
        ):
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
                        "identity": {"microvm_id": "mvm-old"},
                        "allocation_fingerprint": allocation_fingerprint,
                    },
                )
            )

    asyncio.run(run())

    assert adapter.prepare_calls == []
    assert "runner_request" not in adapter.captured


def test_fresh_path_factory_rejects_missing_reconnect_allocation_fingerprint() -> None:
    class FreshPathLifecycleAdapter(_LifecycleRecordingAdapter):
        egress_authority_cutover_strategy = EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH

    adapter = FreshPathLifecycleAdapter()

    async def run() -> None:
        with pytest.raises(
            InvalidEgressReconnectMetadataError,
            match="requires an allocation fingerprint",
        ):
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
                        "identity": {"microvm_id": "mvm-old"},
                    },
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
        process_external_allocation = False

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


def test_egress_teardown_binding_delegates_abandon_to_inner_binding() -> None:
    adapter = _RecordingAdapter()
    abandoned: list[BoundWorkspace] = []

    class _TrackingBinding(WorkspaceBinding):
        async def bind(self, workspace, runner, **kwargs):  # type: ignore[no-untyped-def]
            return BoundWorkspace(runner=runner)

        async def finalize(self, bound, *, outcome=None, metadata=None):  # type: ignore[no-untyped-def]
            raise AssertionError("finalize should not run")

        def abandon(self, bound: BoundWorkspace) -> bool:
            abandoned.append(bound)
            return True

    async def run() -> BoundWorkspace:
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=_TrackingBinding(),
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_abandon",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(None, runner, session_id="sess_abandon")
        # An open managed runner may still mutate the workspace, so an abort
        # cannot release its inner owner merely because a lifecycle returned.
        assert binding.abandon(bound) is False
        assert abandoned == []
        await runner.close()
        assert binding.abandon(bound) is True
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)
        return bound

    bound = asyncio.run(run())
    assert abandoned == [bound]
    assert adapter.torn_down == 1


def test_egress_teardown_retains_sync_owner_until_runner_is_quiescent(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "state.txt").write_text("initial", encoding="utf-8")
    (source_root / "recreated.txt").write_text("original", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="fixed-target")
    inner = SyncBinding(target_workspace=target)
    adapter = _RetryingLifecycleAdapter(first_error=RuntimeError("runner still live"))

    async def workspace_factory(_runner):  # type: ignore[no-untyped-def]
        return source

    async def run() -> None:
        result = await _virtual_factory(
            adapter=adapter,
            workspace_factory=workspace_factory,
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_sync_teardown_fence",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(source, runner, session_id="sess_sync_teardown_fence")
        await target.write_bytes("state.txt", b"first-sync")
        await target.delete("recreated.txt")
        await target.write_bytes("transient.txt", b"first-sync")

        with pytest.raises(RuntimeError, match="runner still live"):
            await binding.finalize(bound, outcome="completed")
        assert not (source_root / "recreated.txt").exists()
        assert (source_root / "transient.txt").read_bytes() == b"first-sync"
        assert_sync_resources_owned(bound, expected=True)
        with pytest.raises(ValueError, match="already bound by an active session"):
            await inner.bind(source, None, session_id="competing-session")

        # The still-live old guest can change its target after the first sync.
        # A teardown retry must sync that later state before releasing ownership.
        await target.write_bytes("state.txt", b"retry-sync")
        await target.delete("transient.txt")
        await target.write_bytes("recreated.txt", b"retry-sync")
        await binding.finalize(bound, outcome="completed")
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}
        assert (source_root / "state.txt").read_bytes() == b"retry-sync"
        assert not (source_root / "transient.txt").exists()
        assert (source_root / "recreated.txt").read_bytes() == b"retry-sync"

    asyncio.run(run())


def test_egress_teardown_does_not_read_runner_bound_target_after_quiescence(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "state.txt").write_text("initial", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    local_target = LocalWorkspace(target_root, workspace_id="shared-target-storage")
    quiesce_started = asyncio.Event()
    allow_quiescence = asyncio.Event()

    target_holder: list[_ClosedRejectingRunnerWorkspace] = []

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        target = _ClosedRejectingRunnerWorkspace(
            context.runner,
            local_target,
            workspace_id="shared-target",
        )
        target_holder.append(target)
        return SyncTargetWorkspacePlan(workspace=target)

    inner = SyncBinding(target_workspace_plan_factory=target_factory)

    class _QuiescingAdapter(_RecordingAdapter):
        def __init__(self) -> None:
            super().__init__("lambda-microvm")
            self.finalize_calls: list[str | None] = []

        async def finalize_runner(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            self.finalize_calls.append(outcome)
            raise AssertionError(f"ordinary finalization was not expected: {outcome}")

        async def finalize_runner_for_binding(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            self.finalize_calls.append(outcome)
            quiesce_started.set()
            await allow_quiescence.wait()
            await runner.close()
            return RunnerFinalizationResult(
                workspace_mutations_quiescent=True,
                allocation_preserved=True,
            )

    adapter = _QuiescingAdapter()

    async def run() -> None:
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_detach_sync_fence",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(source, runner, session_id="sess_detach_sync_fence")
        target = target_holder[0]
        await target.write_bytes("state.txt", b"final-snapshot")

        finalize_task = asyncio.create_task(binding.finalize(bound, outcome="interrupted"))
        await quiesce_started.wait()
        assert (source_root / "state.txt").read_bytes() == b"final-snapshot"
        with pytest.raises(ValueError, match="already bound by an active session"):
            await SyncBinding(target_workspace=target).bind(
                source,
                None,
                session_id="competing-session",
            )

        allow_quiescence.set()
        await finalize_task
        assert runner.is_closed
        assert target.operations_after_close == 0
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}

    asyncio.run(run())
    assert adapter.finalize_calls == ["interrupted"]


@pytest.mark.parametrize("redacted", [False, True], ids=["plain", "redacted"])
def test_egress_teardown_drains_dispatched_write_before_authoritative_sync(
    tmp_path: Path,
    redacted: bool,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "state.txt").write_text("initial", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    local_target = LocalWorkspace(target_root, workspace_id="late-write-storage")
    mutation_started = asyncio.Event()
    allow_mutation = asyncio.Event()

    class _DelayedMutationRunner(_FakeDockerRunner):
        async def exec(self, command: Any, **kwargs: Any) -> ExecResult:
            del command, kwargs
            mutation_started.set()
            await allow_mutation.wait()
            await local_target.write_bytes("state.txt", b"late-dispatched-write")
            return ExecResult()

    async def runner_factory(request: Any) -> Runner:
        return _DelayedMutationRunner(request.name)

    target_holder: list[_ClosedRejectingRunnerWorkspace] = []

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        target = _ClosedRejectingRunnerWorkspace(
            context.runner,
            local_target,
            workspace_id="late-write-target",
        )
        target_holder.append(target)
        return SyncTargetWorkspacePlan(workspace=target)

    inner = SyncBinding(target_workspace_plan_factory=target_factory)

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(runner_factory=runner_factory),
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_late_dispatched_write",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            source,
            runner,
            session_id="sess_late_dispatched_write",
        )
        target = target_holder[0]

        async def dispatch(command: ExecCommand) -> ExecResult:
            if redacted:
                return await runner.exec_redacted(
                    command,
                    redactor=SecretRedactor(),
                )
            return await runner.exec(command)

        mutation_task = asyncio.create_task(dispatch(ExecCommand.process("write-late-state")))
        await mutation_started.wait()
        finalize_task = asyncio.create_task(binding.finalize(bound, outcome="completed"))
        while runner._workspace_dispatch_gate_owner is None:  # type: ignore[attr-defined]
            await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="dispatch is closed"):
            await dispatch(ExecCommand.process("new-work-after-finalization"))
        assert not finalize_task.done()
        assert (source_root / "state.txt").read_bytes() == b"initial"

        allow_mutation.set()
        await mutation_task
        snapshot = await finalize_task
        assert snapshot is not None
        assert runner.is_closed
        assert target.operations_after_close == 0
        assert (source_root / "state.txt").read_bytes() == b"late-dispatched-write"
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}

    asyncio.run(run())


@pytest.mark.parametrize("redacted", [False, True], ids=["plain", "redacted"])
def test_egress_teardown_waits_for_deferred_command_settlement_before_sync(
    tmp_path: Path,
    redacted: bool,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")
    local_target = LocalWorkspace(target_root, workspace_id="deferred-command-storage")
    command_started = asyncio.Event()
    allow_settlement = asyncio.Event()

    class _DeferredMutationRunner(_FakeDockerRunner):
        pending_command_settlement_cancellation_safe = True

        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.settlement_task: asyncio.Task[None] | None = None
            self.settlement_calls = 0

        async def exec(self, command: Any, **kwargs: Any) -> ExecResult:
            del command, kwargs
            command_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:

                async def settle() -> None:
                    await allow_settlement.wait()
                    await local_target.write_bytes(
                        "late.txt",
                        b"written-by-deferred-command",
                    )

                self.settlement_task = asyncio.create_task(settle())
                cancellation.artifacts = [  # type: ignore[attr-defined]
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "test",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ]
                raise

        async def await_pending_command_settlement(self) -> bool:
            self.settlement_calls += 1
            assert self.settlement_task is not None
            await asyncio.shield(self.settlement_task)
            return True

    created_runners: list[_DeferredMutationRunner] = []

    async def runner_factory(request: Any) -> Runner:
        runner = _DeferredMutationRunner(request.name)
        created_runners.append(runner)
        return runner

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        return SyncTargetWorkspacePlan(
            workspace=_ClosedRejectingRunnerWorkspace(
                context.runner,
                local_target,
                workspace_id="deferred-command-target",
            )
        )

    inner = SyncBinding(target_workspace_plan_factory=target_factory)

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(runner_factory=runner_factory),
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_deferred_command_settlement",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            source,
            runner,
            session_id="sess_deferred_command_settlement",
        )

        if redacted:
            command_operation = runner.exec_redacted(
                ExecCommand.process("late-command"),
                redactor=SecretRedactor(),
            )
        else:
            command_operation = runner.exec(ExecCommand.process("late-command"))
        command_task = asyncio.create_task(command_operation)
        await command_started.wait()
        command_task.cancel("interrupt deferred command")
        with pytest.raises(asyncio.CancelledError, match="interrupt deferred command"):
            await command_task

        invocation_settlement = asyncio.create_task(runner.await_pending_command_settlement())
        finalize_task = asyncio.create_task(binding.finalize(bound, outcome="interrupted"))
        while runner._workspace_dispatch_gate_owner is None:  # type: ignore[attr-defined]
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not invocation_settlement.done()
        assert not finalize_task.done()
        assert not (source_root / "late.txt").exists()
        assert created_runners[0].settlement_calls == 1

        allow_settlement.set()
        assert await invocation_settlement is True
        snapshot = await finalize_task
        assert snapshot is not None
        assert (source_root / "late.txt").read_bytes() == b"written-by-deferred-command"
        assert runner.is_closed
        assert created_runners[0].settlement_calls == 1
        assert inner._states == {}

    asyncio.run(run())


def test_egress_teardown_retries_only_after_deferred_sync_command_settles(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")
    local_target = LocalWorkspace(target_root, workspace_id="sync-command-storage")
    command_started = asyncio.Event()
    allow_settlement = asyncio.Event()

    class _DeferredSyncCommandRunner(_FakeDockerRunner):
        pending_command_settlement_cancellation_safe = True

        settlement_task: asyncio.Task[None] | None = None

        async def exec(self, command: Any, **kwargs: Any) -> ExecResult:
            del command, kwargs
            command_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:

                async def settle() -> None:
                    await allow_settlement.wait()

                self.settlement_task = asyncio.create_task(settle())
                cancellation.artifacts = [  # type: ignore[attr-defined]
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "test",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ]
                raise

        async def await_pending_command_settlement(self) -> bool:
            assert self.settlement_task is not None
            await asyncio.shield(self.settlement_task)
            return True

    class _InterruptingSyncWorkspace(_ClosedRejectingRunnerWorkspace):
        interrupt_next_list = False

        async def list(
            self,
            pattern: str = "**/*",
            *,
            limit: int | None = None,
        ) -> WorkspaceListResult:
            if self.interrupt_next_list:
                self.interrupt_next_list = False
                command_task = asyncio.create_task(
                    self._runner.exec(ExecCommand.process("sync-command"))
                )
                await command_started.wait()
                command_task.cancel("interrupt sync command")
                with pytest.raises(asyncio.CancelledError, match="interrupt sync command"):
                    await command_task
                raise RuntimeError("sync command was interrupted")
            return await super().list(pattern, limit=limit)

    async def runner_factory(request: Any) -> Runner:
        return _DeferredSyncCommandRunner(request.name)

    target_holder: list[_InterruptingSyncWorkspace] = []

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        target = _InterruptingSyncWorkspace(
            context.runner,
            local_target,
            workspace_id="sync-command-target",
        )
        target_holder.append(target)
        return SyncTargetWorkspacePlan(workspace=target)

    inner = SyncBinding(target_workspace_plan_factory=target_factory)

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(runner_factory=runner_factory),
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_deferred_sync_command",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            source,
            runner,
            session_id="sess_deferred_sync_command",
        )
        target_holder[0].interrupt_next_list = True

        with pytest.raises(RuntimeError, match="sync command was interrupted"):
            await binding.finalize(bound, outcome="interrupted")
        assert not runner.is_closed
        assert_sync_resources_owned(bound, expected=True)

        retry = asyncio.create_task(binding.finalize(bound, outcome="interrupted"))
        await asyncio.sleep(0)
        assert not retry.done()
        allow_settlement.set()
        assert await retry is not None
        assert runner.is_closed
        assert inner._states == {}

    asyncio.run(run())


def test_egress_teardown_retires_target_killed_by_command_cleanup(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")
    local_target = LocalWorkspace(target_root, workspace_id="killed-command-storage")
    sync_error = RuntimeError("sandbox cleanup removed the sync target")

    class _SandboxCleanupRunner(_FakeDockerRunner):
        async def exec(self, command: Any, **kwargs: Any) -> ExecResult:
            del command, kwargs
            self._closed = True
            return ExecResult(
                timed_out=True,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "test",
                        "action": "kill_sandbox",
                        "status": "completed",
                    }
                ],
            )

        async def close(self) -> None:
            self.closed = True
            self._closed = True

    async def runner_factory(request: Any) -> Runner:
        return _SandboxCleanupRunner(request.name)

    target_holder: list[_ClosedRejectingRunnerWorkspace] = []

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        target = _ClosedRejectingRunnerWorkspace(
            context.runner,
            local_target,
            workspace_id="killed-command-target",
        )
        target_holder.append(target)
        return SyncTargetWorkspacePlan(workspace=target)

    inner = SyncBinding(target_workspace_plan_factory=target_factory)
    adapter = _RecordingAdapter(runner_factory=runner_factory)

    async def run() -> None:
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_killed_command_target",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            source,
            runner,
            session_id="sess_killed_command_target",
        )
        command_result = await runner.exec(ExecCommand.process("times-out"))
        assert command_result.timed_out
        target_holder[0].next_list_error = sync_error

        with pytest.raises(RuntimeError) as first_failure:
            await binding.finalize(bound, outcome="interrupted")
        assert first_failure.value is sync_error
        assert runner.is_closed
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}
        assert adapter.torn_down == 1

        with pytest.raises(RuntimeError) as retry_failure:
            await binding.finalize(bound, outcome="interrupted")
        assert retry_failure.value is sync_error
        assert adapter.torn_down == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    ("settlement_mode", "error_pattern"),
    [
        ("default", "did not prove workspace mutation quiescence"),
        ("child_cancel", "was cancelled without caller cancellation"),
    ],
)
@pytest.mark.parametrize("redacted", [False, True], ids=["plain", "redacted"])
def test_egress_teardown_retains_owner_when_command_settlement_is_uncertain(
    tmp_path: Path,
    settlement_mode: str,
    error_pattern: str,
    redacted: bool,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")
    local_target = LocalWorkspace(target_root, workspace_id="uncertain-command-storage")

    class _UncertainMutationRunner(_FakeDockerRunner):
        pending_command_settlement_cancellation_safe = True

        async def exec(self, command: Any, **kwargs: Any) -> ExecResult:
            del command, kwargs
            return ExecResult(
                timed_out=True,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "test",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ],
            )

        async def await_pending_command_settlement(self) -> bool:
            if settlement_mode == "child_cancel":
                raise asyncio.CancelledError("provider settlement cancelled itself")
            return await super().await_pending_command_settlement()

    async def runner_factory(request: Any) -> Runner:
        return _UncertainMutationRunner(request.name)

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        return SyncTargetWorkspacePlan(
            workspace=_ClosedRejectingRunnerWorkspace(
                context.runner,
                local_target,
                workspace_id="uncertain-command-target",
            )
        )

    inner = SyncBinding(target_workspace_plan_factory=target_factory)

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(runner_factory=runner_factory),
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_uncertain_command_settlement",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            source,
            runner,
            session_id="sess_uncertain_command_settlement",
        )
        if redacted:
            command_result = await runner.exec_redacted(
                ExecCommand.process("uncertain-command"),
                redactor=SecretRedactor(),
            )
        else:
            command_result = await runner.exec(ExecCommand.process("uncertain-command"))
        assert command_result.timed_out

        current_task = asyncio.current_task()
        assert current_task is not None
        with pytest.raises(RuntimeError, match=error_pattern):
            await binding.finalize(bound, outcome="interrupted")
        assert current_task.cancelling() == 0
        assert not runner.is_closed
        assert_sync_resources_owned(bound, expected=True)

        # This explicit operator assertion is the only recovery path when a
        # runner reports deferred cleanup but supplies no settlement contract.
        runner.reopen_exec()
        snapshot = await binding.finalize(bound, outcome="interrupted")
        assert snapshot is not None
        assert runner.is_closed
        assert inner._states == {}

    asyncio.run(run())


@pytest.mark.parametrize(
    "invalid_artifact",
    [
        {
            "type": "cayu.runner_cleanup.v1",
            "action": "kill_command",
            "status": True,
        },
        {
            "type": "cayu.runner_cleanup.v1",
            "action": "future_cleanup_action",
            "status": "completed",
        },
        {
            "type": "cayu.runner_cleanup.v1",
            "action": ["kill_command"],
            "status": "completed",
        },
    ],
)
def test_workspace_dispatch_settlement_rejects_ambiguous_cleanup_evidence(
    invalid_artifact: dict[str, Any],
) -> None:
    result = ExecResult(
        timed_out=True,
        artifacts=[
            {
                "type": "cayu.runner_cleanup.v1",
                "action": "kill_command",
                "status": "completed",
            },
            invalid_artifact,
        ],
    )

    assert _workspace_dispatch_settlement_kind(result=result, error=None) == "uncertain"


def test_workspace_dispatch_settlement_rejects_success_conflicting_with_cleanup() -> None:
    result = ExecResult(
        artifacts=[
            {
                "type": "cayu.runner_cleanup.v1",
                "action": "kill_command",
                "status": "deferred",
            }
        ],
    )

    assert _workspace_dispatch_settlement_kind(result=result, error=None) == "deferred"


def test_cancelled_egress_teardown_retains_gate_until_dispatched_write_syncs(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "state.txt").write_text("initial", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    local_target = LocalWorkspace(target_root, workspace_id="cancelled-drain-storage")
    mutation_started = asyncio.Event()
    allow_mutation = asyncio.Event()

    class _DelayedMutationRunner(_FakeDockerRunner):
        async def exec(self, command: Any, **kwargs: Any) -> ExecResult:
            del command, kwargs
            mutation_started.set()
            await allow_mutation.wait()
            await local_target.write_bytes("state.txt", b"settled-after-cancellation")
            return ExecResult()

    async def runner_factory(request: Any) -> Runner:
        return _DelayedMutationRunner(request.name)

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        return SyncTargetWorkspacePlan(
            workspace=_ClosedRejectingRunnerWorkspace(
                context.runner,
                local_target,
                workspace_id="cancelled-drain-target",
            )
        )

    inner = SyncBinding(target_workspace_plan_factory=target_factory)

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(runner_factory=runner_factory),
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_cancelled_dispatch_drain",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            source,
            runner,
            session_id="sess_cancelled_dispatch_drain",
        )
        # Put finalization past authority revocation so cancellation is
        # delivered specifically while the pre-fence command drain is waiting.
        assert await runner.revoke_authority() is False  # type: ignore[attr-defined]
        mutation_task = asyncio.create_task(runner.exec(ExecCommand.process("write-after-cancel")))
        await mutation_started.wait()
        finalize_task = asyncio.create_task(binding.finalize(bound, outcome="interrupted"))
        while runner._workspace_dispatch_gate_owner is None:  # type: ignore[attr-defined]
            await asyncio.sleep(0)
        finalize_task.cancel("caller abandoned finalization")
        await asyncio.sleep(0)
        # Owned cleanup temporarily consumes the request while the already
        # dispatched mutation drains, then republishes cancellation only after
        # the authoritative synchronization and provider close.
        assert finalize_task.cancelling() == 0
        assert not finalize_task.done()
        with pytest.raises(RuntimeError, match="dispatch is closed"):
            await runner.exec(ExecCommand.process("competing-work"))

        allow_mutation.set()
        await mutation_task
        with pytest.raises(
            asyncio.CancelledError,
            match="caller abandoned finalization",
        ):
            await finalize_task
        assert finalize_task.cancelled()
        assert runner.is_closed
        assert (source_root / "state.txt").read_bytes() == b"settled-after-cancellation"
        assert_sync_resources_owned(bound, expected=True)

        snapshot = await binding.finalize(bound, outcome="interrupted")
        assert snapshot is not None
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}

    asyncio.run(run())


def test_egress_teardown_retains_sync_owner_until_post_cleanup_diagnostics_finish(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "state.txt").write_text("initial", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    local_target = LocalWorkspace(target_root, workspace_id="fixed-target-storage")
    target_holder: list[_ClosedRejectingRunnerWorkspace] = []

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        target = _ClosedRejectingRunnerWorkspace(
            context.runner,
            local_target,
            workspace_id="fixed-target",
        )
        target_holder.append(target)
        return SyncTargetWorkspacePlan(workspace=target)

    inner = SyncBinding(target_workspace_plan_factory=target_factory)
    cancel_revocation_once = True

    async def workspace_factory(_runner):  # type: ignore[no-untyped-def]
        return source

    async def emit(event: Event) -> Event:
        nonlocal cancel_revocation_once
        if event.type == EventType.EGRESS_GRANT_REVOKED and cancel_revocation_once:
            cancel_revocation_once = False
            raise asyncio.CancelledError("revocation diagnostic interrupted")
        return event

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            workspace_factory=workspace_factory,
            inner_binding=inner,
            event_emitter=emit,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_sync_diagnostic_fence",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(source, runner, session_id="sess_sync_diagnostic_fence")
        target = target_holder[0]

        with pytest.raises(asyncio.CancelledError, match="revocation diagnostic interrupted"):
            await binding.finalize(bound, outcome="completed")
        assert runner.is_closed
        assert target.operations_after_close == 0
        assert_sync_resources_owned(bound, expected=True)
        assert bound.state_key in inner._states

        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None
        assert target.operations_after_close == 0
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}

    asyncio.run(run())


def test_egress_teardown_retries_partial_runner_bound_copy_before_cleanup(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("old-a", encoding="utf-8")
    (source_root / "b.txt").write_text("old-b", encoding="utf-8")
    source = _FailingLocalWorkspace(source_root, workspace_id="durable-source")
    local_target = LocalWorkspace(target_root, workspace_id="partial-copy-storage")
    target_holder: list[_ClosedRejectingRunnerWorkspace] = []

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        target = _ClosedRejectingRunnerWorkspace(
            context.runner,
            local_target,
            workspace_id="partial-copy-target",
        )
        target_holder.append(target)
        return SyncTargetWorkspacePlan(workspace=target)

    inner = SyncBinding(target_workspace_plan_factory=target_factory)

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_partial_copy_retry",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(source, runner, session_id="sess_partial_copy_retry")
        target = target_holder[0]
        await target.write_bytes("a.txt", b"new-a")
        await target.write_bytes("b.txt", b"new-b")
        source.fail_write_call = 2

        with pytest.raises(OSError, match="injected durable write failure"):
            await binding.finalize(bound, outcome="completed")
        assert not runner.is_closed
        assert (source_root / "a.txt").read_bytes() == b"new-a"
        assert (source_root / "b.txt").read_bytes() == b"old-b"
        assert_sync_resources_owned(bound, expected=True)
        assert bound.state_key in inner._states
        with pytest.raises(ValueError, match="already bound by an active session"):
            await SyncBinding(target_workspace=target).bind(
                source,
                None,
                session_id="partial-copy-contender",
            )

        source.fail_write_call = None
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None
        assert runner.is_closed
        assert target.operations_after_close == 0
        assert (source_root / "a.txt").read_bytes() == b"new-a"
        assert (source_root / "b.txt").read_bytes() == b"new-b"
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}

    asyncio.run(run())


def test_egress_teardown_retries_partial_runner_bound_delete_before_cleanup(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("delete-a", encoding="utf-8")
    (source_root / "b.txt").write_text("delete-b", encoding="utf-8")
    source = _FailingLocalWorkspace(source_root, workspace_id="durable-source")
    local_target = LocalWorkspace(target_root, workspace_id="partial-delete-storage")
    target_holder: list[_ClosedRejectingRunnerWorkspace] = []

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        target = _ClosedRejectingRunnerWorkspace(
            context.runner,
            local_target,
            workspace_id="partial-delete-target",
        )
        target_holder.append(target)
        return SyncTargetWorkspacePlan(workspace=target)

    inner = SyncBinding(target_workspace_plan_factory=target_factory)

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_partial_delete_retry",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(source, runner, session_id="sess_partial_delete_retry")
        target = target_holder[0]
        await target.delete("a.txt")
        await target.delete("b.txt")
        source.fail_delete_call = 2

        with pytest.raises(OSError, match="injected durable delete failure"):
            await binding.finalize(bound, outcome="completed")
        assert not runner.is_closed
        assert not (source_root / "a.txt").exists()
        assert (source_root / "b.txt").exists()
        assert_sync_resources_owned(bound, expected=True)
        assert bound.state_key in inner._states

        source.fail_delete_call = None
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None
        assert runner.is_closed
        assert target.operations_after_close == 0
        assert not (source_root / "a.txt").exists()
        assert not (source_root / "b.txt").exists()
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}

    asyncio.run(run())


def test_egress_teardown_retains_readable_target_after_sync_failure(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "state.txt").write_text("initial", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    local_target = LocalWorkspace(target_root, workspace_id="failed-target-storage")
    target_holder: list[_ClosedRejectingRunnerWorkspace] = []

    def target_factory(context):  # type: ignore[no-untyped-def]
        assert context.runner is not None
        target = _ClosedRejectingRunnerWorkspace(
            context.runner,
            local_target,
            workspace_id="failed-target",
        )
        target_holder.append(target)
        return SyncTargetWorkspacePlan(workspace=target)

    inner = SyncBinding(target_workspace_plan_factory=target_factory)
    sync_error = RuntimeError("target listing failed before teardown")

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_failed_sync_unreadable_target",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            source,
            runner,
            session_id="sess_failed_sync_unreadable_target",
        )
        target = target_holder[0]
        target.next_list_error = sync_error

        with pytest.raises(RuntimeError) as exc_info:
            await binding.finalize(bound, outcome="completed")
        assert exc_info.value is sync_error
        assert not runner.is_closed
        assert target.operations_after_close == 0
        assert_sync_resources_owned(bound, expected=True)
        assert bound.state_key in inner._states

        assert await binding.finalize(bound, outcome="completed") is not None
        assert runner.is_closed
        assert target.operations_after_close == 0
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}

    asyncio.run(run())


def test_egress_teardown_retains_sync_owner_after_revocation_cancellation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "state.txt").write_text("initial", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="fixed-target")
    inner = SyncBinding(target_workspace=target)
    revocation_started = asyncio.Event()
    allow_revocation = asyncio.Event()

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_sync_revocation_cancel",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(source, runner, session_id="sess_sync_revocation_cancel")

        revoker = runner._authority_revoker  # type: ignore[attr-defined]
        original_revoke = revoker._broker.revoke_authority_and_wait

        async def block_revocation(presented_values: Sequence[str]) -> int:
            revocation_started.set()
            await allow_revocation.wait()
            return await original_revoke(presented_values)

        revoker._broker.revoke_authority_and_wait = block_revocation
        finalize_task = asyncio.create_task(binding.finalize(bound, outcome="completed"))
        await revocation_started.wait()
        finalize_task.cancel("cancel during authority revocation")
        allow_revocation.set()
        with pytest.raises(asyncio.CancelledError):
            await finalize_task
        assert finalize_task.cancelling() == 0
        assert finalize_task.cancelled()
        assert_sync_resources_owned(bound, expected=True)
        assert bound.state_key in inner._states
        with pytest.raises(ValueError, match="already bound by an active session"):
            await inner.bind(source, None, session_id="competing-session")

        await binding.finalize(bound, outcome="completed")
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}

    asyncio.run(run())


def test_nested_egress_teardown_retains_sync_owner_until_outer_runner_is_quiescent(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "state.txt").write_text("initial", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="fixed-target")
    sync_binding = SyncBinding(target_workspace=target)
    outer_cleanup_started = asyncio.Event()
    allow_outer_cleanup = asyncio.Event()

    async def run() -> None:
        inner_result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=sync_binding,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_nested_inner",
                agent_name="agent",
                environment_name="inner-egress",
            )
        )
        inner_binding = inner_result.environment.binding
        inner_runner = inner_result.environment.runner
        assert inner_binding is not None and inner_runner is not None

        outer_result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=inner_binding,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_nested_outer",
                agent_name="agent",
                environment_name="outer-egress",
            )
        )
        outer_binding = outer_result.environment.binding
        outer_runner = outer_result.environment.runner
        assert outer_binding is not None and outer_runner is not None
        bound = await outer_binding.bind(source, outer_runner, session_id="sess_nested_outer")

        original_outer_finalize = outer_runner.finalize_for_binding

        async def block_outer_finalize(
            *,
            outcome: str | None,
            require_workspace_mutations_quiescent: bool,
            workspace_owner_key: str | None = None,
        ) -> None:
            outer_cleanup_started.set()
            await allow_outer_cleanup.wait()
            await original_outer_finalize(
                outcome=outcome,
                require_workspace_mutations_quiescent=(require_workspace_mutations_quiescent),
                workspace_owner_key=workspace_owner_key,
            )

        outer_runner.finalize_for_binding = block_outer_finalize  # type: ignore[method-assign]
        finalize_task = asyncio.create_task(outer_binding.finalize(bound, outcome="completed"))
        await outer_cleanup_started.wait()
        assert inner_runner.is_closed
        assert not outer_runner.is_closed
        assert_sync_resources_owned(bound, expected=True)
        with pytest.raises(ValueError, match="already bound by an active session"):
            await sync_binding.bind(source, None, session_id="competing-session")

        allow_outer_cleanup.set()
        await finalize_task
        assert outer_runner.is_closed
        assert_sync_resources_owned(bound, expected=False)
        assert sync_binding._states == {}

    asyncio.run(run())


@pytest.mark.parametrize(
    ("sync_back", "outcome"),
    [
        ("never", "completed"),
        ("on_success", "failed"),
    ],
)
def test_egress_teardown_defers_no_sync_back_release_until_runner_is_quiescent(
    tmp_path: Path,
    sync_back: str,
    outcome: str,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "state.txt").write_text("initial", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="fixed-target")
    inner = SyncBinding(target_workspace=target, sync_back=sync_back)  # type: ignore[arg-type]
    adapter = _RetryingLifecycleAdapter(first_error=RuntimeError("runner still live"))

    async def workspace_factory(_runner):  # type: ignore[no-untyped-def]
        return source

    async def run() -> None:
        result = await _virtual_factory(
            adapter=adapter,
            workspace_factory=workspace_factory,
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_no_sync_teardown_fence",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(source, runner, session_id="sess_no_sync_teardown_fence")

        with pytest.raises(RuntimeError, match="runner still live"):
            await binding.finalize(bound, outcome=outcome)
        assert_sync_resources_owned(bound, expected=True)
        assert inner._states[bound.state_key].defer_finalize_release
        with pytest.raises(ValueError, match="already bound by an active session"):
            await inner.bind(source, None, session_id="competing-session")

        await binding.finalize(bound, outcome=outcome)
        assert_sync_resources_owned(bound, expected=False)
        assert inner._states == {}

    asyncio.run(run())


def test_app_lazily_retries_retained_egress_cleanup_before_new_environment_work(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "state.txt").write_text("initial", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="fixed-target")
    inner = SyncBinding(target_workspace=target)
    adapter = _RetryingLifecycleAdapter(first_error=RuntimeError("runner still live"))

    async def workspace_factory(_runner):  # type: ignore[no-untyped-def]
        return source

    class _CompletingProvider(ModelProvider):
        name = "fake"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        result = await _virtual_factory(
            adapter=adapter,
            workspace_factory=workspace_factory,
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="sess_retained_egress_cleanup",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(_CompletingProvider(), default=True)
        app.register_environment(result.environment, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="cleanup-trigger")),
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_retained_egress_cleanup",
                    messages=[Message.text("user", "finish")],
                )
            )
        ]
        generation = next(iter(inner._states))
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=True,
        )
        assert (
            "sess_retained_egress_cleanup" in app._environment_lifecycle._active_environment_setups
        )

        # A normal later run drives the bounded cleanup sweep. It retries the
        # retained egress finalizer before admitting unrelated environment work.
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_cleanup_trigger",
                    environment_name="cleanup-trigger",
                    messages=[Message.text("user", "continue")],
                )
            )
        ]
        # Admission gives retained cleanup a deliberately small polling budget.
        # Under a loaded event loop the successful retry may still be awaiting
        # its final harvest when the unrelated run returns.
        assert adapter.finalize_calls == 2
        assert await app.drain_environment_cleanups(timeout_s=0.2) is True
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=False,
        )
        assert inner._states == {}
        assert (
            "sess_retained_egress_cleanup"
            not in app._environment_lifecycle._active_environment_setups
        )

    asyncio.run(run())


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


def test_app_retains_failed_unadopted_cleanup_until_later_retry_succeeds() -> None:
    class _FailingTwiceReconnectAdapter(_LifecycleRecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.third_finalize_started = asyncio.Event()
            self.allow_third_finalize = asyncio.Event()

        async def finalize_runner(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            self.finalize_calls.append(outcome)
            if len(self.finalize_calls) <= 2:
                raise RuntimeError(f"suspend attempt {len(self.finalize_calls)} failed")
            self.third_finalize_started.set()
            await self.allow_third_finalize.wait()
            await runner.close()
            return RunnerFinalizationResult(workspace_mutations_quiescent=True)

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

    async def run() -> tuple[
        list[Event],
        list[Event],
        _FailingTwiceReconnectAdapter,
        _UnreachedProvider,
        CayuApp,
        list[Event],
    ]:
        adapter = _FailingTwiceReconnectAdapter()
        egress_events: list[Event] = []
        store = InMemorySessionStore()
        provider = _UnreachedProvider()

        async def emit(event: Event) -> Event:
            egress_events.append(event)
            return event

        app = CayuApp(
            session_store=store,
            enable_logging=False,
            config=CayuConfig(operations=OperationsConfig(max_environment_lifecycle_owners=1)),
        )
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

        first = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_retained_unadopted_cleanup",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        async with asyncio.timeout(0.2):
            await adapter.third_finalize_started.wait()
        assert "sess_retained_unadopted_cleanup" in (
            app._environment_lifecycle._deferred_factory_cleanup_tasks
        )
        assert "sess_retained_unadopted_cleanup" in (
            app._environment_lifecycle._pending_environment_owner_admissions
        )

        contender = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_retained_unadopted_contender",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        assert len(adapter.prepare_calls) == 1

        adapter.allow_third_finalize.set()
        assert await app.drain_environment_cleanups(timeout_s=0.2) is True
        return first, contender, adapter, provider, app, egress_events

    first, contender, adapter, provider, app, egress_events = asyncio.run(run())

    assert first[-1].type is EventType.SESSION_FAILED
    assert first[-1].payload["error"] == "bind failed"
    assert first[-1].payload["environment_factory_release"]["completed"] is False
    assert contender[-1].type is EventType.SESSION_FAILED
    assert "EnvironmentCapacityError" in str(contender[-1].payload)
    assert adapter.finalize_calls == ["interrupted", "interrupted", "interrupted"]
    assert adapter.torn_down == 1
    assert provider.requests == []
    assert sum(event.type is EventType.EGRESS_GRANT_REVOKED for event in egress_events) == 1
    assert app._environment_lifecycle._deferred_factory_cleanup_tasks == {}
    assert app._environment_lifecycle._pending_environment_owner_admissions == set()


def test_unbound_release_waits_for_adapter_cleanup_owner_before_retry() -> None:
    class _SettlementOwningAdapter(_LifecycleRecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.allow_owned_cleanup = asyncio.Event()
            self.owned_cleanup_started = asyncio.Event()
            self.retry_started = asyncio.Event()
            self.owned_cleanup_task: asyncio.Task[None] | None = None

        async def finalize_runner(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            self.finalize_calls.append(outcome)
            if len(self.finalize_calls) == 1:
                raise RuntimeError("first cleanup attempt failed")
            if len(self.finalize_calls) == 2:

                async def settle_owned_cleanup() -> None:
                    self.owned_cleanup_started.set()
                    await self.allow_owned_cleanup.wait()

                self.owned_cleanup_task = asyncio.create_task(
                    settle_owned_cleanup(),
                    name="adapter-owned-cleanup",
                )
                error = RuntimeError("second cleanup attempt handed off")
                attach_environment_factory_cleanup_settlement_task(
                    error,
                    self.owned_cleanup_task,
                )
                raise error
            self.retry_started.set()
            await runner.close()
            return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    async def run() -> tuple[
        _SettlementOwningAdapter,
        asyncio.Task[None],
    ]:
        adapter = _SettlementOwningAdapter()
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_adapter_cleanup_owner",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        release = result.release
        managed = result.environment.runner
        assert release is not None
        assert managed is not None
        managed._teardown_timeout_s = 0.01
        with pytest.raises(BaseExceptionGroup) as release_error:
            await release(EnvironmentFactoryReleaseAction.DISCARD)
        settlement_task = environment_factory_cleanup_settlement_task(release_error.value)
        assert settlement_task is not None
        await adapter.owned_cleanup_started.wait()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(adapter.retry_started.wait(), timeout=0.02)

        adapter.allow_owned_cleanup.set()
        async with asyncio.timeout(0.2):
            await settlement_task
        return adapter, settlement_task

    adapter, settlement_task = asyncio.run(run())

    assert adapter.owned_cleanup_task is not None
    assert adapter.owned_cleanup_task.done()
    assert settlement_task.done()
    assert adapter.finalize_calls == [None, None, None]
    assert adapter.torn_down == 1


def test_unbound_release_retains_first_cleanup_handoff_before_retry() -> None:
    class _FirstAttemptSettlementAdapter(_LifecycleRecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.allow_owned_cleanup = asyncio.Event()
            self.owned_cleanup_started = asyncio.Event()
            self.retry_started = asyncio.Event()
            self.owned_cleanup_task: asyncio.Task[None] | None = None

        async def finalize_runner(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            self.finalize_calls.append(outcome)
            if len(self.finalize_calls) == 1:

                async def settle_owned_cleanup() -> None:
                    self.owned_cleanup_started.set()
                    await self.allow_owned_cleanup.wait()

                self.owned_cleanup_task = asyncio.create_task(
                    settle_owned_cleanup(),
                    name="first-adapter-owned-cleanup",
                )
                error = RuntimeError("first cleanup attempt handed off")
                attach_environment_factory_cleanup_settlement_task(
                    error,
                    self.owned_cleanup_task,
                )
                raise error
            self.retry_started.set()
            await runner.close()
            return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    async def run() -> tuple[
        _FirstAttemptSettlementAdapter,
        asyncio.Task[None],
    ]:
        adapter = _FirstAttemptSettlementAdapter()
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_first_adapter_cleanup_owner",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        release = result.release
        managed = result.environment.runner
        assert release is not None
        assert managed is not None
        managed._teardown_timeout_s = 0.01
        with pytest.raises(BaseExceptionGroup) as release_error:
            await release(EnvironmentFactoryReleaseAction.DISCARD)
        settlement_task = environment_factory_cleanup_settlement_task(release_error.value)
        assert settlement_task is not None
        await adapter.owned_cleanup_started.wait()
        repeated_release = asyncio.create_task(
            release(EnvironmentFactoryReleaseAction.DISCARD),
            name="repeated-unbound-release",
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(adapter.retry_started.wait(), timeout=0.02)
        async with asyncio.timeout(0.1):
            with pytest.raises(BaseExceptionGroup) as repeated_error:
                await repeated_release
        assert environment_factory_cleanup_settlement_task(repeated_error.value) is settlement_task

        adapter.allow_owned_cleanup.set()
        async with asyncio.timeout(0.2):
            await settlement_task
        return adapter, settlement_task

    adapter, settlement_task = asyncio.run(run())

    assert adapter.owned_cleanup_task is not None
    assert adapter.owned_cleanup_task.done()
    assert settlement_task.done()
    assert adapter.finalize_calls == [None, None]
    assert adapter.torn_down == 1


def test_unbound_release_reused_error_does_not_cycle_settlement_owner() -> None:
    class _ReusedErrorAdapter(_LifecycleRecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.allow_owned_cleanup = asyncio.Event()
            self.owned_cleanup_started = asyncio.Event()
            self.owned_cleanup_task: asyncio.Task[None] | None = None
            self.reused_error = RuntimeError("provider reused cleanup failure")

        async def finalize_runner(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            self.finalize_calls.append(outcome)
            if len(self.finalize_calls) == 1:

                async def settle_owned_cleanup() -> None:
                    self.owned_cleanup_started.set()
                    await self.allow_owned_cleanup.wait()

                self.owned_cleanup_task = asyncio.create_task(
                    settle_owned_cleanup(),
                    name="reused-error-owned-cleanup",
                )
                attach_environment_factory_cleanup_settlement_task(
                    self.reused_error,
                    self.owned_cleanup_task,
                )
                raise self.reused_error
            if len(self.finalize_calls) == 2:
                raise self.reused_error
            await runner.close()
            return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    async def run() -> tuple[_ReusedErrorAdapter, asyncio.Task[None]]:
        adapter = _ReusedErrorAdapter()
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_reused_cleanup_error",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        release = result.release
        assert release is not None
        with pytest.raises(BaseExceptionGroup) as release_error:
            await release(EnvironmentFactoryReleaseAction.DISCARD)
        settlement_task = environment_factory_cleanup_settlement_task(release_error.value)
        assert settlement_task is not None
        assert (
            environment_factory_cleanup_settlement_task(adapter.reused_error)
            is adapter.owned_cleanup_task
        )
        await adapter.owned_cleanup_started.wait()
        adapter.allow_owned_cleanup.set()
        async with asyncio.timeout(0.3):
            await settlement_task
        return adapter, settlement_task

    adapter, settlement_task = asyncio.run(run())

    assert adapter.owned_cleanup_task is not None
    assert adapter.owned_cleanup_task.done()
    assert settlement_task.done()
    assert adapter.finalize_calls == [None, None, None]
    assert adapter.torn_down == 1


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


def test_factory_rollback_retains_timed_out_managed_cleanup_owner() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    creation_error = RuntimeError("workspace creation failed")

    class _BlockingRollbackAdapter(_RecordingAdapter):
        async def prepare(self, **kwargs: Any) -> EgressBinding:
            binding = await super().prepare(**kwargs)
            binding.teardown_timeout_s = 0.01
            return binding

        async def finalize_runner(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            del outcome
            close_started.set()
            await allow_close.wait()
            await runner.close()
            return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    async def fail_workspace_creation(_runner: Runner) -> LocalWorkspace:
        raise creation_error

    async def run() -> tuple[BaseException, asyncio.Task[None]]:
        adapter = _BlockingRollbackAdapter()
        factory = _virtual_factory(adapter=adapter)
        factory._workspace_factory = fail_workspace_creation
        with pytest.raises(RuntimeError, match="workspace creation failed") as exc_info:
            await factory.create(
                EnvironmentFactoryRequest(
                    session_id="sess_timed_out_managed_rollback",
                    agent_name="assistant",
                    environment_name="egress-env",
                )
            )
        settlement_task = environment_factory_cleanup_settlement_task(exc_info.value)
        assert settlement_task is not None
        assert close_started.is_set()
        assert not settlement_task.done()
        allow_close.set()
        async with asyncio.timeout(0.2):
            await asyncio.shield(settlement_task)
        return exc_info.value, settlement_task

    failure, settlement_task = asyncio.run(run())

    assert failure is creation_error
    assert settlement_task.done()
    assert not settlement_task.cancelled()


@pytest.mark.parametrize("error_type", [TimeoutError, RuntimeError])
def test_managed_factory_settlement_backs_off_immediate_failures(
    error_type: type[Exception],
) -> None:
    async def run() -> int:
        result = await _virtual_factory(adapter=_RecordingAdapter()).create(
            EnvironmentFactoryRequest(
                session_id="sess_immediate_cleanup_timeout",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None
        managed._teardown_timeout_s = 0.01
        attempts = 0

        async def immediate_failure(*, outcome: str | None) -> None:
            nonlocal attempts
            del outcome
            attempts += 1
            raise error_type("provider cleanup immediately failed")

        managed.finalize = immediate_failure  # type: ignore[method-assign]
        settlement_task = managed.defer_finalization_settlement(outcome=None)
        await asyncio.sleep(0.025)
        settlement_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await settlement_task
        return attempts

    attempts = asyncio.run(run())

    assert 1 <= attempts <= 3


def test_managed_factory_settlement_preserves_cancellation_during_attempt() -> None:
    async def run() -> tuple[asyncio.Task[None], asyncio.Task[None]]:
        result = await _virtual_factory(adapter=_RecordingAdapter()).create(
            EnvironmentFactoryRequest(
                session_id="sess_cleanup_settlement_cancelled",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None
        attempt_started = asyncio.Event()
        allow_failure = asyncio.Event()
        allow_owned_cleanup = asyncio.Event()
        retry_started = asyncio.Event()
        attempts = 0

        async def fail_after_cancellation(
            *,
            deadline: float,
            retained_cutover_failure: BaseException | None = None,
        ) -> None:
            nonlocal attempts
            del deadline
            assert retained_cutover_failure is None
            attempts += 1
            if attempts == 1:
                attempt_started.set()
                await allow_failure.wait()

                async def settle_owned_cleanup() -> None:
                    await allow_owned_cleanup.wait()

                owned_cleanup = asyncio.create_task(
                    settle_owned_cleanup(),
                    name="cancelled-attempt-owned-cleanup",
                )
                error = RuntimeError("cleanup failed after settlement cancellation")
                attach_environment_factory_cleanup_settlement_task(
                    error,
                    owned_cleanup,
                )
                raise ExceptionGroup("cleanup handed off during cancellation", [error])
            retry_started.set()
            managed._closed = True
            managed._completed_runner_action = managed._requested_runner_action

        managed._finalize_serialized = fail_after_cancellation  # type: ignore[method-assign]
        settlement_task = managed.defer_finalization_settlement(outcome=None)
        await attempt_started.wait()
        settlement_task.cancel("stop retained cleanup")
        allow_failure.set()
        with pytest.raises(asyncio.CancelledError, match="stop retained cleanup"):
            await settlement_task
        retry_task = managed.defer_finalization_settlement(outcome=None)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(retry_started.wait(), timeout=0.02)
        allow_owned_cleanup.set()
        async with asyncio.timeout(0.2):
            await retry_task
        return settlement_task, retry_task

    settlement_task, retry_task = asyncio.run(run())

    assert settlement_task.cancelled()
    assert retry_task.done()


def test_managed_factory_settlement_preserves_concurrent_fatal_signal() -> None:
    async def run() -> tuple[BaseExceptionGroup, asyncio.Task[None], GeneratorExit]:
        result = await _virtual_factory(adapter=_RecordingAdapter()).create(
            EnvironmentFactoryRequest(
                session_id="sess_cleanup_settlement_cancelled_with_fatal",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None
        attempt_started = asyncio.Event()
        allow_failure = asyncio.Event()
        fatal_signal = GeneratorExit("provider cleanup interrupted")

        async def fail_after_cancellation(
            *,
            deadline: float,
            retained_cutover_failure: BaseException | None = None,
        ) -> None:
            del deadline
            assert retained_cutover_failure is None
            attempt_started.set()
            await allow_failure.wait()
            raise fatal_signal

        managed._finalize_serialized = fail_after_cancellation  # type: ignore[method-assign]
        settlement_task = managed.defer_finalization_settlement(outcome=None)
        await attempt_started.wait()
        settlement_task.cancel("stop retained cleanup")
        assert settlement_task.cancelling() == 1
        await asyncio.sleep(0)
        assert not settlement_task.done()
        allow_failure.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await settlement_task
        return exc_info.value, settlement_task, fatal_signal

    failure, settlement_task, fatal_signal = asyncio.run(run())

    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert failure.exceptions[1] is fatal_signal
    propagated_fatal = binding_finalize_fatal_signal(failure)
    assert isinstance(propagated_fatal, BaseExceptionGroup)
    assert propagated_fatal.exceptions == (fatal_signal,)
    assert settlement_task.cancelled() is False


def test_managed_factory_prerequisite_preserves_concurrent_fatal_signal() -> None:
    async def run() -> tuple[
        BaseExceptionGroup,
        asyncio.Task[None],
        BaseExceptionGroup,
        GeneratorExit,
    ]:
        result = await _virtual_factory(adapter=_RecordingAdapter()).create(
            EnvironmentFactoryRequest(
                session_id="sess_cleanup_prerequisite_cancelled_with_fatal",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None
        prerequisite_started = asyncio.Event()
        allow_failure = asyncio.Event()
        fatal_signal = GeneratorExit("predecessor cleanup interrupted")
        fatal_failure = BaseExceptionGroup(
            "predecessor cleanup carried a fatal signal",
            [fatal_signal],
        )

        async def fail_prerequisite() -> None:
            prerequisite_started.set()
            await allow_failure.wait()
            raise fatal_failure

        prerequisite_task = asyncio.create_task(
            fail_prerequisite(),
            name="fatal-cleanup-prerequisite",
        )
        managed._factory_cleanup_prerequisite_tasks.add(prerequisite_task)
        drain_task = asyncio.create_task(
            managed._drain_factory_cleanup_prerequisites(
                deadline=asyncio.get_running_loop().time() + 1,
            ),
            name="cancelled-cleanup-prerequisite-drain",
        )
        await prerequisite_started.wait()
        await asyncio.sleep(0)
        drain_task.cancel("stop predecessor cleanup wait")
        assert drain_task.cancelling() == 1
        await asyncio.sleep(0)
        assert not drain_task.done()
        allow_failure.set()
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await drain_task
        return exc_info.value, drain_task, fatal_failure, fatal_signal

    failure, drain_task, fatal_failure, fatal_signal = asyncio.run(run())

    assert isinstance(failure.exceptions[0], asyncio.CancelledError)
    assert failure.exceptions[1] is fatal_failure
    propagated_fatal = binding_finalize_fatal_signal(failure)
    assert isinstance(propagated_fatal, BaseExceptionGroup)
    assert propagated_fatal.exceptions == (fatal_signal,)
    assert drain_task.cancelled() is False


def test_managed_factory_settlement_follows_grouped_prerequisite_handoff() -> None:
    async def run() -> tuple[asyncio.Task[None], asyncio.Task[None]]:
        result = await _virtual_factory(adapter=_RecordingAdapter()).create(
            EnvironmentFactoryRequest(
                session_id="sess_grouped_cleanup_prerequisite",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None
        allow_successor = asyncio.Event()
        retry_started = asyncio.Event()

        async def successor() -> None:
            await allow_successor.wait()

        successor_task = asyncio.create_task(
            successor(),
            name="grouped-cleanup-successor",
        )

        async def prerequisite() -> None:
            error = RuntimeError("prerequisite handed off cleanup")
            attach_environment_factory_cleanup_settlement_task(
                error,
                successor_task,
            )
            raise ExceptionGroup("grouped prerequisite failure", [error])

        prerequisite_task = asyncio.create_task(
            prerequisite(),
            name="grouped-cleanup-prerequisite",
        )

        async def complete_retry(
            *,
            deadline: float,
            retained_cutover_failure: BaseException | None = None,
        ) -> None:
            del deadline
            assert retained_cutover_failure is None
            retry_started.set()
            managed._closed = True
            managed._completed_runner_action = managed._requested_runner_action

        managed._finalize_serialized = complete_retry  # type: ignore[method-assign]
        settlement_task = managed.defer_finalization_settlement(
            outcome=None,
            prerequisite_task=prerequisite_task,
        )
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(retry_started.wait(), timeout=0.02)
        allow_successor.set()
        async with asyncio.timeout(0.2):
            await settlement_task
        return successor_task, settlement_task

    successor_task, settlement_task = asyncio.run(run())

    assert successor_task.done()
    assert settlement_task.done()


def test_managed_factory_settlement_retries_child_only_cancellation() -> None:
    class _ChildCancellingAdapter(_RecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        async def finalize_runner(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            del outcome
            self.attempts += 1
            if self.attempts <= 2:
                raise asyncio.CancelledError("provider cleanup self-cancelled")
            await runner.close()
            return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    async def run() -> int:
        adapter = _ChildCancellingAdapter()
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_cleanup_settlement_child_cancelled",
                agent_name="assistant",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None
        settlement_task = managed.defer_finalization_settlement(outcome=None)
        async with asyncio.timeout(0.3):
            await settlement_task
        return adapter.attempts

    assert asyncio.run(run()) == 3


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


def test_later_release_escalation_refreshes_completed_settlement_owner() -> None:
    class _TwiceFailingEscalationAdapter(_LifecycleRecordingAdapter):
        async def finalize_runner(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            self.finalize_calls.append(outcome)
            if len(self.finalize_calls) in {1, 2, 4, 5}:
                raise RuntimeError(f"cleanup attempt {len(self.finalize_calls)} failed")
            await runner.close()
            return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    async def run() -> tuple[
        asyncio.Task[None],
        asyncio.Task[None],
        _TwiceFailingEscalationAdapter,
        Runner,
    ]:
        adapter = _TwiceFailingEscalationAdapter()
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_release_escalation_settlement",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        release = result.release
        runner = result.environment.runner
        assert release is not None and runner is not None

        with pytest.raises(BaseExceptionGroup) as preserve_error:
            await release(EnvironmentFactoryReleaseAction.PRESERVE)
        preserve_settlement = environment_factory_cleanup_settlement_task(preserve_error.value)
        assert preserve_settlement is not None
        async with asyncio.timeout(0.2):
            await preserve_settlement

        with pytest.raises(BaseExceptionGroup) as discard_error:
            await release(EnvironmentFactoryReleaseAction.DISCARD)
        discard_settlement = environment_factory_cleanup_settlement_task(discard_error.value)
        assert discard_settlement is not None
        assert discard_settlement is not preserve_settlement
        assert not discard_settlement.done()
        async with asyncio.timeout(0.2):
            await discard_settlement
        return preserve_settlement, discard_settlement, adapter, runner

    preserve_settlement, discard_settlement, adapter, runner = asyncio.run(run())

    assert preserve_settlement.done()
    assert discard_settlement.done()
    assert adapter.finalize_calls == [
        "interrupted",
        "interrupted",
        "interrupted",
        None,
        None,
        None,
    ]
    assert adapter.torn_down == 1
    assert runner.closed is True


def test_pending_preserve_settlement_absorbs_later_discard_escalation() -> None:
    class _BlockedPreserveAdapter(_LifecycleRecordingAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.binding_close_started = asyncio.Event()
            self.allow_binding_close = asyncio.Event()

        async def prepare(self, **kwargs: Any) -> EgressBinding:
            binding = await super().prepare(**kwargs)
            binding.teardown_timeout_s = 0.01
            original_teardown = binding.teardown
            assert original_teardown is not None

            async def blocked_teardown() -> None:
                self.binding_close_started.set()
                await self.allow_binding_close.wait()
                await original_teardown()

            binding.teardown = blocked_teardown
            return binding

        async def finalize_runner(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            self.finalize_calls.append(outcome)
            if len(self.finalize_calls) <= 2:
                raise RuntimeError(f"detach attempt {len(self.finalize_calls)} failed")
            if outcome is not None:
                return RunnerFinalizationResult(
                    workspace_mutations_quiescent=True,
                    allocation_preserved=True,
                )
            await runner.close()
            return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    async def run() -> tuple[_BlockedPreserveAdapter, Runner, asyncio.Task[None]]:
        adapter = _BlockedPreserveAdapter()
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_pending_preserve_escalation",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        release = result.release
        runner = result.environment.runner
        assert release is not None and runner is not None

        with pytest.raises(BaseExceptionGroup) as preserve_error:
            await release(EnvironmentFactoryReleaseAction.PRESERVE)
        preserve_settlement = environment_factory_cleanup_settlement_task(preserve_error.value)
        assert preserve_settlement is not None
        async with asyncio.timeout(0.2):
            await adapter.binding_close_started.wait()
        assert not preserve_settlement.done()

        with pytest.raises(BaseExceptionGroup) as discard_error:
            await release(EnvironmentFactoryReleaseAction.DISCARD)
        discard_settlement = environment_factory_cleanup_settlement_task(discard_error.value)
        assert discard_settlement is preserve_settlement

        adapter.allow_binding_close.set()
        async with asyncio.timeout(0.2):
            await discard_settlement
        return adapter, runner, discard_settlement

    adapter, runner, discard_settlement = asyncio.run(run())

    assert discard_settlement.done()
    assert adapter.finalize_calls == [
        "interrupted",
        "interrupted",
        "interrupted",
        None,
    ]
    assert adapter.torn_down == 1
    assert runner.closed is True


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


def test_runner_close_settles_upstream_before_virtual_grant_drain() -> None:
    class _StallingUpstream:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.settled = asyncio.Event()

        def prepare(
            self,
            request: CapturedRequest,
            *,
            limits: EgressUpstreamLimits,
        ) -> EgressUpstreamOperation:
            assert isinstance(request, CapturedRequest)
            assert limits.max_response_bytes > 0

            async def send() -> CapturedResponse:
                self.started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.settled.set()
                raise AssertionError("Unreachable stalled upstream completed.")

            async def cancel_and_wait(task: asyncio.Task[CapturedResponse]) -> None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            return EgressUpstreamOperation(send, cancel_and_wait=cancel_and_wait)

    async def run() -> tuple[int, int, bool, dict[str, int]]:
        adapter = _RecordingAdapter("fake")
        upstream = _StallingUpstream()
        factory = _virtual_factory(adapter=adapter, upstream=upstream)
        result = await factory.create(
            EnvironmentFactoryRequest(
                session_id="sess_settle_before_grant_drain",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        managed = result.environment.runner
        assert managed is not None
        broker: TransparentEgressBroker = adapter.captured["broker"]
        grant = adapter.captured["grant"]
        captured = _broker_request(grant.presented_value, "/v1/customers")
        inflight = asyncio.create_task(broker.handle_request(captured))
        await upstream.started.wait()

        await asyncio.wait_for(managed.close(), timeout=1.0)
        inflight_response = await asyncio.wait_for(inflight, timeout=1.0)
        denied = await broker.handle_request(captured)
        return (
            inflight_response.status_code,
            denied.status_code,
            upstream.settled.is_set(),
            dict(broker.registry._active_counts),
        )

    inflight_status, new_request_status, settled, active_counts = asyncio.run(run())

    assert inflight_status == 502
    assert new_request_status == 403
    assert settled is True
    assert active_counts == {}


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
            "omit_truncated_output": False,
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
            "omit_truncated_output": False,
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


@pytest.mark.parametrize("method", ("exec", "exec_redacted", "exec_system"))
@pytest.mark.parametrize(
    ("invalid_kwargs", "error_type"),
    (
        ({"env": {"INVALID\x00NAME": "value"}}, ValueError),
        ({"env": {"VALID": "invalid\x00value"}}, ValueError),
        ({"env": {"VALID": "invalid\ud800value"}}, ValueError),
        ({"env_remove": ("INVALID\x00NAME",)}, ValueError),
        ({"env_remove": None}, TypeError),
        ({"cwd": "/outside-runner-root"}, ValueError),
        ({"timeout_s": 0}, ValueError),
        ({"stdin": b"invalid"}, TypeError),
        ({"output_limit_bytes": 0}, ValueError),
    ),
)
def test_managed_runner_rejects_invalid_environment_before_dispatch_admission(
    method: str,
    invalid_kwargs: dict[str, Any],
    error_type: type[Exception],
) -> None:
    class _RecordingRunner(Runner):
        isolation = "docker"
        default_cwd = "/workspace"

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            del command
            self.calls.append(dict(kwargs))
            return ExecResult()

    inner = _RecordingRunner()
    adapter = _RecordingAdapter(
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id=f"sess_invalid_env_{method}",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert isinstance(runner, _EgressManagedRunner)
        managed = cast("_EgressManagedRunner", runner)
        invoke = getattr(managed, method)
        kwargs = dict(invalid_kwargs)
        if method == "exec_redacted":
            kwargs["redactor"] = SecretRedactor()

        with pytest.raises(error_type):
            await invoke(ExecCommand.process("invalid"), **kwargs)

        assert inner.calls == []
        assert managed._active_workspace_dispatches == set()
        assert managed._uncertain_workspace_dispatches == {}

        valid_kwargs: dict[str, Any] = {"env_remove": ("REMOVE_ME",)}
        if method == "exec_redacted":
            valid_kwargs["redactor"] = SecretRedactor()
        assert (await invoke(ExecCommand.process("valid"), **valid_kwargs)).exit_code == 0
        assert len(inner.calls) == 1
        await managed.close()

    asyncio.run(run())

    assert inner.is_closed
    assert adapter.torn_down == 1


def test_managed_docker_preflight_rejects_transport_key_before_secret_state_or_admission() -> None:
    class _CountingVault(StaticVault):
        def __init__(self) -> None:
            super().__init__({"runner-token": "secret-value"})
            self.resolve_calls = 0

        async def resolve(self, ref, *, scope=None):  # type: ignore[no-untyped-def]
            self.resolve_calls += 1
            return await super().resolve(ref, scope=scope)

    vault = _CountingVault()
    inner = DockerRunner(
        "managed-preflight",
        docker_path="/usr/bin/docker",
        secret_env={"TOKEN": SecretRef(name="runner-token")},
        secret_resolver=vault,
    )
    adapter = _RecordingAdapter(
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )
    registry_calls = 0

    def redactor_snapshot() -> InvocationRedactorSnapshot:
        nonlocal registry_calls
        registry_calls += 1
        return InvocationRedactorSnapshot(revision=0, redactor=SecretRedactor())

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_managed_docker_preflight",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert isinstance(runner, _EgressManagedRunner)
        managed = cast("_EgressManagedRunner", runner)
        handle = InvocationRunnerHandle(
            managed,
            redactor_snapshot_provider=redactor_snapshot,
        )

        with pytest.raises(RunnerExecutionError):
            await handle.exec(
                ExecCommand.process("true"),
                env={"#COMMENT": "value"},
            )

        assert vault.resolve_calls == 0
        assert registry_calls == 0
        assert managed._active_workspace_dispatches == set()
        assert managed._uncertain_workspace_dispatches == {}
        owner_key = "managed-docker-binding-owner"
        managed.begin_workspace_binding()
        managed.finish_workspace_binding(
            require_mutation_quiescence=True,
            workspace_owner_key=owner_key,
        )
        assert await managed.prepare_workspace_sync(owner_key) is None
        await managed.finalize_for_binding(
            outcome=None,
            require_workspace_mutations_quiescent=True,
            workspace_owner_key=owner_key,
        )

    asyncio.run(run())

    assert inner.is_closed
    assert adapter.torn_down == 1


@pytest.mark.parametrize("runner_kind", ("lambda-microvm", "e2b", "microsandbox"))
def test_managed_remote_preflight_rejects_invalid_overlay_before_secret_state_or_admission(
    runner_kind: str,
) -> None:
    def create_runner() -> Runner:
        env_overlay = {"INVALID\x00NAME": "value"}
        if runner_kind == "lambda-microvm":
            return LambdaMicroVMRunner(
                object(),
                microvm_id="mvm-preflight",
                endpoint="mvm.internal",
                endpoint_transport=object(),  # type: ignore[arg-type]
                env_overlay=env_overlay,
            )
        if runner_kind == "e2b":
            return E2BRunner(
                object(),
                sandbox_id="e2b-preflight",
                env_overlay=env_overlay,
            )
        return MicrosandboxRunner(
            object(),
            name="microsandbox-preflight",
            env_overlay=env_overlay,
        )

    inner = create_runner()
    adapter = _RecordingAdapter(
        runner_kind=runner_kind,
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )
    registry_calls = 0

    def redactor_snapshot() -> InvocationRedactorSnapshot:
        nonlocal registry_calls
        registry_calls += 1
        return InvocationRedactorSnapshot(revision=0, redactor=SecretRedactor())

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id=f"sess_managed_{runner_kind}_preflight",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert isinstance(runner, _EgressManagedRunner)
        managed = cast("_EgressManagedRunner", runner)
        handle = InvocationRunnerHandle(
            managed,
            redactor_snapshot_provider=redactor_snapshot,
        )

        with pytest.raises(RunnerExecutionError):
            await handle.exec(ExecCommand.process("true"))

        assert registry_calls == 0
        assert managed._active_workspace_dispatches == set()
        assert managed._uncertain_workspace_dispatches == {}
        owner_key = f"managed-{runner_kind}-binding-owner"
        managed.begin_workspace_binding()
        managed.finish_workspace_binding(
            require_mutation_quiescence=True,
            workspace_owner_key=owner_key,
        )
        assert await managed.prepare_workspace_sync(owner_key) is None
        await managed.finalize_for_binding(
            outcome=None,
            require_workspace_mutations_quiescent=True,
            workspace_owner_key=owner_key,
        )

    asyncio.run(run())

    assert inner.is_closed
    assert adapter.torn_down == 1


def test_managed_runner_retains_genuine_post_admission_failure_as_uncertain() -> None:
    class _FailingRunner(Runner):
        isolation = "docker"
        default_cwd = "/workspace"

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            del command, kwargs
            raise RuntimeError("failure after managed admission")

    inner = _FailingRunner()
    adapter = _RecordingAdapter(
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_managed_post_admission_failure",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert isinstance(runner, _EgressManagedRunner)
        managed = cast("_EgressManagedRunner", runner)

        with pytest.raises(RuntimeError, match="failure after managed admission"):
            await managed.exec(ExecCommand.process("true"))

        assert managed._active_workspace_dispatches == set()
        assert len(managed._uncertain_workspace_dispatches) == 1
        managed.reopen_exec()
        await managed.close()

    asyncio.run(run())

    assert adapter.torn_down == 1


def test_managed_runner_detaches_deferred_settlement_process_signal() -> None:
    canary = "PRIVATE_EGRESS_SETTLEMENT_SIGNAL_CANARY"
    raw_signal = SystemExit(canary)

    class _SignalRunner(Runner):
        pending_command_settlement_cancellation_safe = True
        isolation = "docker"
        default_cwd = "/workspace"

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            del command, kwargs
            return ExecResult(
                timed_out=True,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "test",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ],
            )

        async def await_pending_command_settlement(self) -> bool:
            raise raw_signal

    inner = _SignalRunner()
    adapter = _RecordingAdapter(
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )

    async def run() -> tuple[BaseException, bool]:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_managed_settlement_signal",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert isinstance(runner, _EgressManagedRunner)
        managed = cast("_EgressManagedRunner", runner)
        owner_key = "managed-settlement-signal-owner"
        managed.begin_workspace_binding()
        managed.finish_workspace_binding(
            require_mutation_quiescence=True,
            workspace_owner_key=owner_key,
        )

        command_result = await managed.exec(ExecCommand.process("mutate"))
        assert command_result.timed_out is True
        await managed._active_workspace_dispatches_drained.wait()
        with pytest.raises(SystemExit) as raised:
            await managed.prepare_workspace_sync(owner_key)
        replayed = await managed.await_pending_command_settlement()
        return raised.value, replayed

    failure, replayed = asyncio.run(run())

    assert isinstance(failure, SystemExit)
    assert failure.code == 1
    assert canary not in repr(failure)
    assert replayed is False
    assert_cayu_traceback_does_not_retain(failure, raw_signal)


def test_managed_runner_invocation_settlement_delivers_process_signal_once() -> None:
    canary = "PRIVATE_EGRESS_INVOCATION_SETTLEMENT_SIGNAL_CANARY"
    raw_signal = SystemExit(canary)

    class _SignalRunner(Runner):
        pending_command_settlement_cancellation_safe = True
        isolation = "docker"
        default_cwd = "/workspace"

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            del command, kwargs
            return ExecResult(
                timed_out=True,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "test",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ],
            )

        async def await_pending_command_settlement(self) -> bool:
            raise raw_signal

    inner = _SignalRunner()
    adapter = _RecordingAdapter(
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )

    async def run() -> tuple[BaseException, bool]:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_managed_invocation_settlement_signal",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert isinstance(runner, _EgressManagedRunner)
        managed = cast("_EgressManagedRunner", runner)

        command_result = await managed.exec(ExecCommand.process("mutate"))
        assert command_result.timed_out is True
        await managed._active_workspace_dispatches_drained.wait()
        with pytest.raises(SystemExit) as raised:
            await managed.await_pending_command_settlement()
        replayed = await managed.await_pending_command_settlement()
        return raised.value, replayed

    failure, replayed = asyncio.run(run())

    assert isinstance(failure, SystemExit)
    assert failure.code == 1
    assert canary not in repr(failure)
    assert replayed is False
    assert_cayu_traceback_does_not_retain(failure, raw_signal)


def test_runtime_propagates_managed_runner_settlement_signal_once(
    tmp_path: Path,
    caplog,
    capsys,
) -> None:
    canary = "PRIVATE_EGRESS_RUNTIME_SETTLEMENT_SIGNAL_CANARY"
    raw_signal = SystemExit(canary)

    class _SignalRunner(Runner):
        pending_command_settlement_cancellation_safe = True
        isolation = "docker"
        default_cwd = "/workspace"

        @property
        def resource_key(self) -> tuple[object, ...]:
            return ("test-signal-runner", id(self))

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            del command, kwargs
            return ExecResult(
                timed_out=True,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "test",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ],
            )

        async def await_pending_command_settlement(self) -> bool:
            raise raw_signal

    class _MutationTool(Tool):
        spec = ToolSpec(
            name="managed_runner_mutation",
            parallel_safe=False,
            workspace_mutation=True,
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del args
            assert ctx.runner is not None
            await ctx.runner.exec(ExecCommand.process("mutate"))
            raise AssertionError("Settlement process signal did not propagate.")

    class _ToolProvider(ModelProvider):
        name = "managed-signal"

        def __init__(self) -> None:
            self.requests = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.requests += 1
            yield ModelStreamEvent.tool_call(
                id="call-managed-runner-mutation",
                name="managed_runner_mutation",
                arguments={},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

    inner = _SignalRunner()
    adapter = _RecordingAdapter(
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_delegate = LocalWorkspace(
        workspace_root,
        workspace_id="managed-signal-workspace-storage",
    )

    class _ManagedRunnerWorkspace(_ClosedRejectingRunnerWorkspace):
        @property
        def bound_runner_resource_key(self) -> tuple[object, ...] | None:
            return self._runner.resource_key

    def workspace_factory(runner: Runner) -> RunnerBoundWorkspace:
        return _ManagedRunnerWorkspace(
            runner,
            workspace_delegate,
            workspace_id="managed-signal-workspace",
        )

    async def run() -> tuple[BaseException, list[Event], _ToolProvider]:
        store = InMemorySessionStore()
        provider = _ToolProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="egress-env"),
            _virtual_factory(
                adapter=adapter,
                workspace_factory=workspace_factory,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="managed-signal-model"),
            tools=[_MutationTool()],
        )

        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sess_managed_runtime_settlement_signal",
                        messages=[Message.text("user", "mutate")],
                    )
                )
            ]
        except BaseException as failure:
            return (
                failure,
                await store.load_events("sess_managed_runtime_settlement_signal"),
                provider,
            )
        raise AssertionError("Managed settlement process signal did not propagate.")

    with warnings.catch_warnings(record=True) as captured_warnings:
        failure, events, provider = asyncio.run(run())
    captured = capsys.readouterr()

    signals = [
        candidate for candidate in iter_exception_tree(failure) if isinstance(candidate, SystemExit)
    ]
    assert len(signals) == 1
    assert signals[0].code == 1
    assert provider.requests == 1
    assert not any(event.type is EventType.WORKSPACE_MUTATION_RECORDED for event in events)
    assert_cayu_traceback_does_not_retain(failure, raw_signal)
    combined = repr(
        (
            failure,
            [event.model_dump(mode="json") for event in events],
            captured_warnings,
            [record.getMessage() for record in caplog.records],
            captured.out,
            captured.err,
        )
    )
    assert canary not in combined


def test_managed_runner_does_not_start_undeclared_inner_settlement_waiter() -> None:
    class _UnsafeSettlementRunner(Runner):
        isolation = "docker"
        default_cwd = "/workspace"

        def __init__(self) -> None:
            self.settlement_calls = 0

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            del command, kwargs
            return ExecResult(
                timed_out=True,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "test",
                        "action": "kill_command",
                        "status": "deferred",
                    }
                ],
            )

        async def await_pending_command_settlement(self) -> bool:
            self.settlement_calls += 1
            await asyncio.Event().wait()
            return True

    inner = _UnsafeSettlementRunner()
    adapter = _RecordingAdapter(
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )

    async def run() -> BaseException:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_managed_unsafe_settlement",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert isinstance(runner, _EgressManagedRunner)
        managed = cast("_EgressManagedRunner", runner)
        owner_key = "managed-unsafe-settlement-owner"
        managed.begin_workspace_binding()
        managed.finish_workspace_binding(
            require_mutation_quiescence=True,
            workspace_owner_key=owner_key,
        )

        assert (await managed.exec(ExecCommand.process("mutate"))).timed_out is True
        await managed._active_workspace_dispatches_drained.wait()
        with pytest.raises(RuntimeError) as raised:
            await managed.prepare_workspace_sync(owner_key)
        return raised.value

    failure = asyncio.run(run())

    assert str(failure) == "Managed runner deferred command settlement is not cancellation-safe."
    assert inner.settlement_calls == 0


@pytest.mark.parametrize("method", ("exec", "exec_redacted", "exec_system"))
@pytest.mark.parametrize("command_kind", ("process", "shell"))
@pytest.mark.parametrize("invalid_text", ("invalid\x00command", "invalid\ud800command"))
def test_managed_runner_rejects_mutated_command_before_dispatch_admission(
    method: str,
    command_kind: str,
    invalid_text: str,
) -> None:
    class _RecordingRunner(Runner):
        isolation = "docker"
        default_cwd = "/workspace"

        def __init__(self) -> None:
            self.calls = 0

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            del command, kwargs
            self.calls += 1
            return ExecResult()

    inner = _RecordingRunner()
    adapter = _RecordingAdapter(
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id=f"sess_invalid_command_{method}_{command_kind}",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert isinstance(runner, _EgressManagedRunner)
        managed = cast("_EgressManagedRunner", runner)
        command = (
            ExecCommand.process("valid") if command_kind == "process" else ExecCommand.bash("true")
        )
        if command_kind == "process":
            assert command.argv is not None
            command.argv[0] = invalid_text
        else:
            command.shell = invalid_text
        kwargs: dict[str, Any] = {}
        if method == "exec_redacted":
            kwargs["redactor"] = SecretRedactor()

        with pytest.raises(ValueError):
            await getattr(managed, method)(command, **kwargs)

        assert inner.calls == 0
        assert managed._active_workspace_dispatches == set()
        assert managed._uncertain_workspace_dispatches == {}
        await managed.close()

    asyncio.run(run())

    assert inner.is_closed
    assert adapter.torn_down == 1


@pytest.mark.parametrize("method", ("exec", "exec_redacted", "exec_system"))
def test_managed_closed_runner_rejects_before_inner_validation(method: str) -> None:
    class _ResolvingRunner(Runner):
        isolation = "docker"
        default_cwd = "/workspace"

        def __init__(self) -> None:
            self.resolve_calls = 0

        def resolve_cwd(self, cwd: str | None = None) -> str:
            self.resolve_calls += 1
            return super().resolve_cwd(cwd)

        async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
            del command, kwargs
            return ExecResult()

    inner = _ResolvingRunner()
    adapter = _RecordingAdapter(
        runner_factory=lambda _request: asyncio.sleep(0, result=inner),
    )

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id=f"sess_closed_preflight_{method}",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert isinstance(runner, _EgressManagedRunner)
        managed = cast("_EgressManagedRunner", runner)
        await managed.close()
        invoke = getattr(managed, method)
        kwargs: dict[str, Any] = {"cwd": "/outside-runner-root"}
        if method == "exec_redacted":
            kwargs["redactor"] = SecretRedactor()

        with pytest.raises(RuntimeError, match="closed"):
            await invoke(ExecCommand.process("true"), **kwargs)

        assert inner.resolve_calls == 0

    asyncio.run(run())

    assert adapter.torn_down == 1


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


def test_concurrent_terminal_escalation_follows_preserving_quiescence() -> None:
    quiesce_started = asyncio.Event()
    allow_quiesce = asyncio.Event()
    remove_started = asyncio.Event()
    allow_remove = asyncio.Event()

    class _CoordinatedAdapter(_LifecycleRecordingAdapter):
        async def finalize_runner_for_binding(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            assert outcome == "interrupted"
            self.finalize_calls.append("quiesce")
            quiesce_started.set()
            await allow_quiesce.wait()
            return RunnerFinalizationResult(
                workspace_mutations_quiescent=True,
                allocation_preserved=True,
            )

        async def finalize_runner(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            self.finalize_calls.append(outcome)
            remove_started.set()
            await allow_remove.wait()
            await runner.close()
            return RunnerFinalizationResult(workspace_mutations_quiescent=True)

    adapter = _CoordinatedAdapter()

    async def run() -> None:
        result = await _virtual_factory(adapter=adapter).create(
            EnvironmentFactoryRequest(
                session_id="sess_concurrent_quiescence_escalation",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        runner = result.environment.runner
        assert runner is not None
        runner.arm_workspace_mutation_quiescence()

        interrupted = asyncio.create_task(runner.finalize(outcome="interrupted"))
        await quiesce_started.wait()
        terminal = asyncio.create_task(runner.close())
        await asyncio.sleep(0)
        allow_quiesce.set()
        await remove_started.wait()
        assert adapter.torn_down == 0
        allow_remove.set()
        await asyncio.gather(interrupted, terminal)

    asyncio.run(run())

    assert adapter.finalize_calls == ["quiesce", None]
    assert adapter.torn_down == 1


def test_binding_quiescence_is_armed_before_late_finalizer_claim_release(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="late-finalizer-target")
    inner = SyncBinding(target_workspace=target)
    claim_release_started = asyncio.Event()
    allow_claim_release = asyncio.Event()

    class _QuiescenceRecordingAdapter(_RecordingAdapter):
        def __init__(self) -> None:
            super().__init__("lambda-microvm")
            self.binding_finalize_calls = 0

        async def finalize_runner_for_binding(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            assert outcome == "interrupted"
            self.binding_finalize_calls += 1
            await runner.close()
            return RunnerFinalizationResult(
                workspace_mutations_quiescent=True,
                allocation_preserved=True,
            )

    adapter = _QuiescenceRecordingAdapter()

    async def run() -> None:
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="late-binding-finalizer",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            source,
            runner,
            session_id="late-binding-finalizer",
        )
        egress_binding: EgressBinding = adapter.captured["binding"]
        original_teardown = egress_binding.teardown
        assert original_teardown is not None

        async def blocked_teardown() -> None:
            claim_release_started.set()
            await allow_claim_release.wait()
            await original_teardown()

        egress_binding.teardown = blocked_teardown
        ordinary_finalize = asyncio.create_task(runner.finalize(outcome="interrupted"))
        while runner._workspace_dispatch_gate_owner is None:  # type: ignore[attr-defined]
            await asyncio.sleep(0)
        assert not claim_release_started.is_set()
        binding_finalize = asyncio.create_task(binding.finalize(bound, outcome="interrupted"))
        await claim_release_started.wait()

        with pytest.raises(ValueError, match="already bound by an active session"):
            await SyncBinding(target_workspace=target).bind(
                source,
                None,
                session_id="competing-session",
            )

        allow_claim_release.set()
        await asyncio.gather(ordinary_finalize, binding_finalize)
        replacement = SyncBinding(target_workspace=target)
        replacement_bound = await replacement.bind(
            source,
            None,
            session_id="replacement-session",
        )
        replacement.abandon(replacement_bound)

    asyncio.run(run())

    assert adapter.binding_finalize_calls == 1
    assert adapter.torn_down == 1


def test_managed_runner_rejects_second_stateful_binding_before_inner_mutation(
    tmp_path: Path,
) -> None:
    first_source_root = tmp_path / "first-source"
    second_source_root = tmp_path / "second-source"
    target_root = tmp_path / "target"
    first_source_root.mkdir()
    second_source_root.mkdir()
    target_root.mkdir()
    first_source = LocalWorkspace(first_source_root, workspace_id="first-source")
    second_source = LocalWorkspace(second_source_root, workspace_id="second-source")
    target = LocalWorkspace(target_root, workspace_id="single-owner-target")
    target_factory_calls = 0

    def target_factory(_context):  # type: ignore[no-untyped-def]
        nonlocal target_factory_calls
        target_factory_calls += 1
        return SyncTargetWorkspacePlan(workspace=target)

    inner = SyncBinding(target_workspace_plan_factory=target_factory)

    async def run() -> None:
        result = await _virtual_factory(
            adapter=_RecordingAdapter(),
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="single-managed-binding",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None
        bound = await binding.bind(
            first_source,
            runner,
            session_id="first-managed-binding",
        )

        with pytest.raises(
            RuntimeError,
            match="already has an active stateful workspace binding",
        ):
            await binding.bind(
                second_source,
                runner,
                session_id="second-managed-binding",
            )
        assert target_factory_calls == 1
        assert_sync_resources_owned(bound, expected=True)

        await binding.finalize(bound, outcome="completed")
        assert_sync_resources_owned(bound, expected=False)

    asyncio.run(run())


def test_inflight_binding_admission_blocks_runner_release_until_quiescence_is_armed(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="inflight-bind-target")
    inner_bound = asyncio.Event()
    allow_bind_return = asyncio.Event()

    class _BlockingSyncBinding(SyncBinding):
        async def bind(self, *args: Any, **kwargs: Any) -> BoundWorkspace:
            bound = await super().bind(*args, **kwargs)
            inner_bound.set()
            await allow_bind_return.wait()
            return bound

    inner = _BlockingSyncBinding(target_workspace=target)

    class _QuiescenceRecordingAdapter(_RecordingAdapter):
        def __init__(self) -> None:
            super().__init__("lambda-microvm")
            self.binding_finalize_calls = 0

        async def prepare(self, **kwargs: Any) -> EgressBinding:
            binding = await super().prepare(**kwargs)
            binding.teardown_timeout_s = 0.01
            return binding

        async def finalize_runner_for_binding(
            self,
            runner: Runner,
            *,
            outcome: str | None,
        ) -> RunnerFinalizationResult:
            assert outcome == "interrupted"
            self.binding_finalize_calls += 1
            await runner.close()
            return RunnerFinalizationResult(
                workspace_mutations_quiescent=True,
                allocation_preserved=True,
            )

    adapter = _QuiescenceRecordingAdapter()

    async def run() -> None:
        result = await _virtual_factory(
            adapter=adapter,
            inner_binding=inner,
        ).create(
            EnvironmentFactoryRequest(
                session_id="inflight-binding-finalizer",
                agent_name="agent",
                environment_name="egress-env",
            )
        )
        binding = result.environment.binding
        runner = result.environment.runner
        assert binding is not None and runner is not None

        bind_task = asyncio.create_task(
            binding.bind(
                source,
                runner,
                session_id="inflight-binding-finalizer",
            )
        )
        await inner_bound.wait()
        with pytest.raises(
            RuntimeError,
            match="workspace binding admission in progress",
        ):
            await binding.bind(
                source,
                runner,
                session_id="competing-inflight-binding",
            )
        finalizer = asyncio.create_task(runner.finalize(outcome="interrupted"))
        await asyncio.sleep(0)

        assert not finalizer.done()
        assert adapter.binding_finalize_calls == 0
        assert adapter.torn_down == 0
        with pytest.raises(TimeoutError, match="binding lifecycle boundary"):
            async with asyncio.timeout(0.2):
                await finalizer
        generation = next(iter(inner._states))
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=True,
        )
        assert adapter.binding_finalize_calls == 0

        allow_bind_return.set()
        bound = await bind_task
        await binding.finalize(bound, outcome="interrupted")
        assert adapter.binding_finalize_calls == 1
        assert_sync_resources_owned(bound, expected=False)

    asyncio.run(run())

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
        invocation_lifecycle_command_version = 1

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
        invocation_lifecycle_command_version = 1

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
        invocation_lifecycle_command_version = 1

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
        invocation_lifecycle_command_version = 1

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
        invocation_lifecycle_command_version = 1

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
        invocation_lifecycle_command_version = 1
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
        invocation_lifecycle_command_version = 1

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
        invocation_lifecycle_command_version = 1

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
        invocation_lifecycle_command_version = 1

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
        invocation_lifecycle_command_version = 1
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
                execution_profile_fingerprint="b" * 64,
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
    assert {e.payload.get("execution_profile_fingerprint") for e in events} == {"b" * 64}


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
