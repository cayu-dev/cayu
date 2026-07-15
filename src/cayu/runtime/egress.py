"""Runtime wiring for virtual egress credentials.

Turns the egress library into a first-class, session-lifecycle-managed mode: a
``VirtualEgressEnvironmentFactory`` mints per-session grants, stands up the
broker + an adapter-enforced sandbox, and emits audit events; teardown (revoke +
remove runtime network resources + stop proxy) runs from the workspace binding's
``finalize`` hook that the runtime already calls at session end.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import os
import secrets
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cayu._validation import copy_json_value
from cayu.artifacts import ArtifactStore
from cayu.core.events import Event, EventType
from cayu.egress import (
    ApprovedEgressDestination,
    CredentialKind,
    CredentialMode,
    EgressAdapterRegistry,
    EgressBinding,
    EgressDecision,
    EgressPolicy,
    EgressUpstream,
    SandboxEgressAdapter,
    TransparentEgressBroker,
    UnsupportedEgressError,
    VirtualCredentialGrant,
    VirtualCredentialRegistry,
    VirtualEgressRunnerRequest,
)
from cayu.egress.adapter import (
    DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS,
    _await_bounded_cleanup_task,
)
from cayu.egress.credential_kinds import validate_credential_kind
from cayu.egress.destinations import validate_approved_destinations
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import (
    BoundWorkspace,
    NativeBinding,
    NoWorkspaceBinding,
    WorkspaceBinding,
    WorkspaceSnapshot,
)
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerWorkspaceCapabilityT,
)
from cayu.vaults import SecretRef, SecretResolver
from cayu.workspaces import RunnerBoundWorkspace, Workspace

EventEmitter = Callable[[Event], Awaitable[Event]]
VirtualEgressWorkspaceFactory = Callable[[Runner], Workspace | Awaitable[Workspace]]

DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"
VIRTUAL_EGRESS_EVENT_TYPES = (
    EventType.CREDENTIAL_MODE_SELECTED,
    EventType.EGRESS_GRANT_MINTED,
    EventType.EGRESS_REQUEST_AUTHORIZED,
    EventType.EGRESS_REQUEST_DENIED,
    EventType.EGRESS_GRANT_REVOKED,
)


@dataclass(frozen=True)
class VirtualCredentialSpec:
    """Declares one virtual credential the sandbox should present."""

    env_name: str
    secret: SecretRef
    destination: str
    policy_name: str
    credential_kind: CredentialKind = "stripe_bearer"
    ttl_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "credential_kind", validate_credential_kind(self.credential_kind))


class VirtualEgressEnvironmentFactory(EnvironmentFactory):
    """Per-session environment factory that enforces virtual egress.

    ``create`` mints grants, builds a broker (wired to emit audit events),
    prepares the selected egress adapter, and returns an ``Environment`` whose
    runner is on the enforced network and whose binding tears everything down
    at session end. Unsupported runners fail closed inside the adapter registry.

    Scope: virtual egress governs the **sandbox process** credential — the value
    the sandboxed app can read. It does not govern MCP servers: ``McpServerSpec``
    ``secret_env``/``secret_headers`` are resolved *host-side* (into the MCP
    server subprocess or the host HTTP client), never injected into the sandbox,
    so they sit at the ``trusted_tool`` boundary and are outside this factory.
    """

    def __init__(
        self,
        *,
        policies: Mapping[str, EgressPolicy],
        credentials: Sequence[VirtualCredentialSpec] = (),
        approved_destinations: Sequence[ApprovedEgressDestination] = (),
        resolver: SecretResolver | None = None,
        image: str = DEFAULT_SANDBOX_IMAGE,
        setup_commands: Sequence[str] = (),
        adapter: SandboxEgressAdapter | None = None,
        adapter_registry: EgressAdapterRegistry | None = None,
        runner_kind: str = "docker",
        inner_binding: WorkspaceBinding | None = None,
        workspace_factory: VirtualEgressWorkspaceFactory | None = None,
        artifact_store: ArtifactStore | None = None,
        event_emitter: EventEmitter | None = None,
        upstream: EgressUpstream | None = None,
        require_test_mode_credentials: bool = True,
    ) -> None:
        if not credentials and not approved_destinations:
            raise ValueError(
                "VirtualEgressEnvironmentFactory requires at least one credential or "
                "approved destination."
            )
        if credentials and resolver is None:
            raise ValueError("Virtual-egress credentials require a secret resolver.")
        if adapter is not None and adapter_registry is not None:
            raise ValueError("Pass either adapter or adapter_registry, not both.")
        duplicate_env_names = _duplicate_env_names(credentials)
        if duplicate_env_names:
            raise ValueError(
                "Virtual credential env_name values must be unique: "
                + ", ".join(duplicate_env_names)
            )
        self._resolver = resolver
        self._policies = dict(policies)
        self._credentials = tuple(credentials)
        self._approved_destinations = validate_approved_destinations(
            approved_destinations,
        )
        self._image = image
        self._setup_commands = tuple(setup_commands)
        self._adapter = adapter
        self._adapter_registry = adapter_registry
        self._runner_kind = adapter.runner_kind if adapter is not None else runner_kind
        if workspace_factory is not None and not callable(workspace_factory):
            raise TypeError("workspace_factory must be callable or None.")
        self._workspace_factory = workspace_factory
        self._inner_binding = inner_binding or (
            NativeBinding() if workspace_factory is not None else NoWorkspaceBinding()
        )
        if artifact_store is not None and not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must be an ArtifactStore.")
        self._artifact_store = artifact_store
        self._emitter = event_emitter
        self._upstream = upstream
        self._require_test_mode = require_test_mode_credentials

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        loop = asyncio.get_running_loop()
        adapter = self._adapter or self._resolve_adapter(loop)
        runner_kind = adapter.runner_kind
        registry = VirtualCredentialRegistry()
        grants = [
            registry.mint(
                session_id=request.session_id,
                env_name=spec.env_name,
                secret=spec.secret,
                destination=spec.destination,
                credential_kind=spec.credential_kind,
                policy_name=spec.policy_name,
                ttl_seconds=spec.ttl_seconds,
            )
            for spec in self._credentials
        ]

        audit = _EgressAuditBridge(
            loop=loop,
            emitter=self._emitter,
            session_id=request.session_id,
            agent_name=request.agent_name,
            environment_name=request.environment_name,
        )
        broker = TransparentEgressBroker(
            registry=registry,
            resolver=self._resolver,
            policies=self._policies,
            approved_destinations=self._approved_destinations,
            upstream=self._upstream,
            audit=audit,
            require_test_mode_credentials=self._require_test_mode,
        )
        authority_revoker = _EgressAuthorityRevoker(
            grants=grants,
            broker=broker,
        )

        # From here on, adapter.prepare may return resources (proxy thread +
        # docker network + sidecar) before workspace binding/finalization is
        # guaranteed. Guard the whole handoff so the factory owns cleanup until
        # the returned Environment owns it.
        binding: EgressBinding | None = None
        ca_dir: str | None = None
        runner: Runner | None = None
        managed_runner: _EgressManagedRunner | None = None
        workspace: Workspace | None = None
        capability_metadata: dict[str, Any] = {}
        try:
            binding = await adapter.prepare(
                session_id=request.session_id,
                grants=grants,
                broker=broker,
            )
            authority_revoker.teardown_timeout_s = binding.teardown_timeout_s
            ca_dir = tempfile.mkdtemp(prefix="cayu-egress-ca-")
            ca_host = os.path.join(ca_dir, "ca.pem")
            with open(ca_host, "wb") as handle:
                handle.write(binding.ca_cert_pem or b"")

            env_overlay = {**binding.env, **{g.env_name: g.presented_value for g in grants}}
            guest_ca_path = _required_binding_field(binding, "guest_ca_path")
            runner_request = VirtualEgressRunnerRequest(
                name=f"cayu-egress-sandbox-{secrets.token_hex(4)}",
                runner_kind=runner_kind,
                image=self._image,
                binding=binding,
                env_overlay=env_overlay,
                ca_cert_host_path=ca_host,
                guest_ca_path=guest_ca_path,
                setup_commands=self._setup_commands,
                egress_destinations=_ordered_destinations(
                    grants,
                    self._approved_destinations,
                ),
                session_id=request.session_id,
                parent_session_id=request.parent_session_id,
                reconnect_metadata=request.reconnect_metadata,
            )
            runner = await adapter.create_runner(runner_request)
            reconnect_metadata = adapter.reconnect_metadata(runner)
            raw_capabilities = adapter.capability_metadata(runner)
            if type(raw_capabilities) is not dict:
                raise TypeError("Egress adapter capability_metadata must return a dict.")
            capability_metadata = copy_json_value(
                raw_capabilities,
                "egress_capabilities",
            )

            managed_runner = _EgressManagedRunner(
                runner=runner,
                adapter=adapter,
                egress_binding=binding,
                ca_dir=ca_dir,
                authority_revoker=authority_revoker,
                audit=audit,
            )
            runner = None
            binding = None
            ca_dir = None
            workspace = await self._create_workspace(managed_runner)
            teardown_binding = _EgressTeardownBinding(
                inner=self._inner_binding,
                runner=managed_runner,
                grants=grants,
                emitter=self._emitter,
                audit=audit,
                session_id=request.session_id,
                agent_name=request.agent_name,
                environment_name=request.environment_name,
            )

            await self._emit_grant_events(request, grants, runner_kind=runner_kind)
        except BaseException as original:
            cleanup_errors: list[tuple[str, BaseException]] = []
            deadline = asyncio.get_running_loop().time() + (
                binding.teardown_timeout_s
                if binding is not None
                else DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS
            )
            if managed_runner is not None:
                try:
                    await managed_runner.close()
                except asyncio.CancelledError:
                    pass
                except BaseException as cleanup_error:
                    cleanup_errors.append(("managed runner", cleanup_error))
            else:
                revocation_complete = False
                try:
                    await authority_revoker.revoke(
                        timeout_s=_remaining_before_deadline(
                            deadline,
                            "Virtual-egress rollback timed out before grant revocation.",
                        )
                    )
                    revocation_complete = True
                except asyncio.CancelledError:
                    revocation_complete = True
                except BaseException as cleanup_error:
                    cleanup_errors.append(("grant revocation", cleanup_error))
                if runner is not None and revocation_complete:
                    try:
                        await _await_rollback_phase(
                            runner.close,
                            deadline=deadline,
                            phase="runner",
                        )
                    except asyncio.CancelledError:
                        pass
                    except BaseException as cleanup_error:
                        cleanup_errors.append(("runner", cleanup_error))
                if binding is not None and revocation_complete:
                    try:
                        await _await_rollback_phase(
                            binding.close,
                            deadline=deadline,
                            phase="binding",
                        )
                    except asyncio.CancelledError:
                        pass
                    except BaseException as cleanup_error:
                        cleanup_errors.append(("binding", cleanup_error))
            if ca_dir is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    shutil.rmtree(ca_dir, ignore_errors=True)
            if cleanup_errors:
                details = "; ".join(
                    f"{phase}: {type(error).__name__}" for phase, error in cleanup_errors
                )
                original.add_note(f"Virtual-egress rollback incomplete: {details}.")
            raise
        environment_metadata: dict[str, Any] = {
            "kind": runner_kind,
            "credential_mode": CredentialMode.VIRTUAL_EGRESS.value,
        }
        result_metadata: dict[str, Any] = {}
        if capability_metadata:
            environment_metadata["egress_capabilities"] = capability_metadata
            result_metadata["egress_capabilities"] = capability_metadata
        spec = EnvironmentSpec(name=request.environment_name, metadata=environment_metadata)
        environment = Environment(
            spec,
            workspace=workspace,
            artifact_store=self._artifact_store,
            runner=managed_runner,
            binding=teardown_binding,
        )
        return EnvironmentFactoryResult(
            environment=environment,
            metadata=result_metadata,
            reconnect_metadata=reconnect_metadata,
        )

    async def _create_workspace(self, runner: Runner) -> Workspace | None:
        if self._workspace_factory is None:
            return None
        created = self._workspace_factory(runner)
        if inspect.isawaitable(created):
            created = await created
        if not isinstance(created, Workspace):
            raise TypeError("workspace_factory must return a Workspace.")
        if isinstance(self._inner_binding, NativeBinding):
            if not isinstance(created, RunnerBoundWorkspace):
                raise TypeError(
                    "A NativeBinding virtual-egress workspace must implement "
                    "RunnerBoundWorkspace. Use an explicit non-native inner_binding for an "
                    "external workspace."
                )
            if created.bound_runner is not runner:
                raise ValueError(
                    "A NativeBinding virtual-egress workspace must be bound to the managed "
                    "runner passed to workspace_factory."
                )
            runner_key = runner.resource_key
            workspace_runner_key = created.bound_runner_resource_key
            if runner_key is None or workspace_runner_key is None:
                raise ValueError(
                    "A NativeBinding virtual-egress runner and workspace must expose stable "
                    "resource identity."
                )
            if workspace_runner_key != runner_key:
                raise ValueError(
                    "A NativeBinding virtual-egress workspace targets a different runner "
                    "resource than the managed runner."
                )
        return created

    def _resolve_adapter(self, loop: asyncio.AbstractEventLoop):
        if self._adapter_registry is not None:
            return self._adapter_registry.resolve(self._runner_kind)

        # Lazy import so `import cayu` never requires the [egress] extra.
        from cayu.egress.docker_adapter import DockerEgressAdapter

        registry = EgressAdapterRegistry()
        registry.register(DockerEgressAdapter(loop=loop))
        return registry.resolve(self._runner_kind)

    async def _emit_grant_events(
        self,
        request: EnvironmentFactoryRequest,
        grants: Sequence[VirtualCredentialGrant],
        *,
        runner_kind: str,
    ) -> None:
        if self._emitter is None:
            return
        with contextlib.suppress(Exception):
            await self._emitter(
                Event(
                    type=EventType.CREDENTIAL_MODE_SELECTED,
                    session_id=request.session_id,
                    agent_name=request.agent_name,
                    environment_name=request.environment_name,
                    payload={
                        "credential_mode": CredentialMode.VIRTUAL_EGRESS.value,
                        "runner_kind": runner_kind,
                        "grant_count": len(grants),
                        "approved_destination_count": len(self._approved_destinations),
                    },
                )
            )
            for grant in grants:
                await self._emitter(
                    Event(
                        type=EventType.EGRESS_GRANT_MINTED,
                        session_id=request.session_id,
                        agent_name=request.agent_name,
                        environment_name=request.environment_name,
                        payload=_grant_payload(grant),
                    )
                )


class _EgressAuthorityRevoker:
    """Disables a session's credentialed and credentialless egress authority."""

    def __init__(
        self,
        *,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> None:
        self._presented_values = tuple(grant.presented_value for grant in grants)
        self._broker = broker
        self._revoked = False
        self._drained = False
        self._task: asyncio.Task[None] | None = None
        self.teardown_timeout_s = DEFAULT_EGRESS_TEARDOWN_TIMEOUT_SECONDS

    async def revoke(self, *, timeout_s: float | None = None) -> bool:
        if self._drained:
            return False
        if self._task is None:
            self._task = asyncio.create_task(self._revoke_and_wait())
        task = self._task
        effective_timeout_s = self.teardown_timeout_s if timeout_s is None else timeout_s
        try:
            cancelled = await _await_cleanup_task(
                task,
                timeout_s=effective_timeout_s,
                timeout_message=(
                    "Virtual-egress grant revocation did not complete within "
                    f"{effective_timeout_s:g} seconds."
                ),
            )
        except BaseException:
            if task.done() and self._task is task:
                self._task = None
            raise
        self._drained = True
        return cancelled

    async def _revoke_and_wait(self) -> None:
        if not self._revoked:
            await self._broker.revoke_authority_and_wait(self._presented_values)
            self._revoked = True


async def _await_cleanup_task(
    task: asyncio.Task[None],
    *,
    timeout_s: float | None = None,
    timeout_message: str | None = None,
) -> bool:
    """Wait for cleanup to finish even if the awaiting task is cancelled."""
    if timeout_s is not None:
        return await _await_bounded_cleanup_task(
            task,
            timeout_s=timeout_s,
            timeout_message=timeout_message or "Cleanup timed out.",
        )
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
            return cancelled
        except asyncio.CancelledError:
            cancelled = True
            if task.done():
                await task
                return cancelled


async def _await_cleanup(awaitable: Awaitable[None]) -> bool:
    async def _run() -> None:
        await awaitable

    return await _await_cleanup_task(asyncio.create_task(_run()))


def _remaining_before_deadline(deadline: float, timeout_message: str) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise TimeoutError(timeout_message)
    return remaining


async def _await_rollback_phase(
    close: Callable[[], Awaitable[None]],
    *,
    deadline: float,
    phase: str,
) -> bool:
    timeout_message = f"Virtual-egress {phase} rollback timed out."
    remaining = _remaining_before_deadline(deadline, timeout_message)

    async def run_close() -> None:
        await close()

    task = asyncio.create_task(run_close())
    return await _await_bounded_cleanup_task(
        task,
        timeout_s=remaining,
        timeout_message=timeout_message,
    )


class _EgressManagedRunner(Runner):
    """Runner wrapper that also owns pre-bind egress resources.

    Workspace binding finalization remains the normal session-end cleanup path,
    but a caller that closes the runner before binding/finalization still tears
    down the egress proxy/network/grants and CA material.
    """

    def __init__(
        self,
        *,
        runner: Runner,
        adapter: SandboxEgressAdapter,
        egress_binding: EgressBinding,
        ca_dir: str,
        authority_revoker: _EgressAuthorityRevoker,
        audit: _EgressAuditBridge | None = None,
    ) -> None:
        self._runner = runner
        self._adapter = adapter
        self._egress_binding = egress_binding
        self._ca_dir = ca_dir
        self._authority_revoker = authority_revoker
        self._audit = audit
        self._teardown_timeout_s = egress_binding.teardown_timeout_s
        self._runner_close_task: asyncio.Task[None] | None = None
        self._binding_close_task: asyncio.Task[None] | None = None
        self._audit_drain_task: asyncio.Task[None] | None = None
        self.isolation = runner.isolation
        self.default_cwd = runner.default_cwd

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        return self._runner.resource_key

    @property
    def closed(self) -> bool:
        """Report whether managed finalization completed."""

        return self._closed

    def workspace_capability(
        self,
        capability_type: type[RunnerWorkspaceCapabilityT],
    ) -> RunnerWorkspaceCapabilityT | None:
        """Delegate only the explicit, lifecycle-free workspace capability."""

        return self._runner.workspace_capability(capability_type)

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        self._ensure_exec_open()
        return await self._runner.exec(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )

    async def exec_system(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        self._ensure_exec_open()
        return await self._runner.exec_system(
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )

    async def close(self) -> None:
        await self.finalize(outcome=None)

    async def revoke_authority(self) -> bool:
        return await self._authority_revoker.revoke()

    async def finalize(self, *, outcome: str | None) -> None:
        if self._closed:
            return
        deadline = asyncio.get_running_loop().time() + self._teardown_timeout_s
        cancelled = await self._authority_revoker.revoke(
            timeout_s=self._remaining_teardown_time(deadline)
        )
        # Do not release enforcement resources unless revocation completed.
        # A revocation error leaves this runner open for a truthful retry.
        errors: list[tuple[str, Exception]] = []
        try:
            cancelled = (
                await self._await_close_phase(
                    "_runner_close_task",
                    lambda: self._adapter.finalize_runner(self._runner, outcome=outcome),
                    deadline=deadline,
                    phase="runner",
                )
                or cancelled
            )
        except TimeoutError:
            raise
        except Exception as exc:
            errors.append(("runner", exc))
        try:
            cancelled = (
                await self._await_close_phase(
                    "_binding_close_task",
                    self._egress_binding.close,
                    deadline=deadline,
                    phase="binding",
                )
                or cancelled
            )
        except TimeoutError:
            raise
        except Exception as exc:
            errors.append(("binding", exc))
        if self._audit is not None:
            try:
                cancelled = (
                    await self._await_close_phase(
                        "_audit_drain_task",
                        self._audit.drain,
                        deadline=deadline,
                        phase="audit",
                    )
                    or cancelled
                )
            except TimeoutError:
                raise
            except Exception as exc:
                errors.append(("audit", exc))
        if errors:
            details = "; ".join(f"{phase}: {type(exc).__name__}: {exc}" for phase, exc in errors)
            raise RuntimeError(f"Virtual-egress resource cleanup incomplete: {details}")
        with contextlib.suppress(Exception):
            shutil.rmtree(self._ca_dir, ignore_errors=True)
        self._closed = True
        if cancelled:
            raise asyncio.CancelledError()

    def _remaining_teardown_time(self, deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(
                "Virtual-egress teardown did not complete within "
                f"{self._teardown_timeout_s:g} seconds."
            )
        return remaining

    async def _await_close_phase(
        self,
        task_field: str,
        close: Callable[[], Awaitable[None]],
        *,
        deadline: float,
        phase: str,
    ) -> bool:
        task = getattr(self, task_field)
        if task is None:

            async def run_close() -> None:
                await close()

            task = asyncio.create_task(run_close())
            setattr(self, task_field, task)
        try:
            return await _await_bounded_cleanup_task(
                task,
                timeout_s=self._remaining_teardown_time(deadline),
                timeout_message=(
                    f"Virtual-egress {phase} cleanup did not complete within "
                    f"{self._teardown_timeout_s:g} seconds."
                ),
            )
        except BaseException:
            if task.done() and getattr(self, task_field) is task:
                setattr(self, task_field, None)
            raise

    def reopen_exec(self) -> None:
        self._runner.reopen_exec()

    def resolve_cwd(self, cwd: str | None = None) -> str:
        return self._runner.resolve_cwd(cwd)


class _EgressAuditBridge:
    """Turns secret-free ``EgressDecision`` records into runtime events.

    Called synchronously from inside ``broker.handle_request`` (which runs on the
    app loop), so it schedules the async emit onto that loop without blocking.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        emitter: EventEmitter | None,
        session_id: str,
        agent_name: str,
        environment_name: str,
    ) -> None:
        self._loop = loop
        self._emitter = emitter
        self._session_id = session_id
        self._agent_name = agent_name
        self._environment_name = environment_name
        self._pending: set[concurrent.futures.Future[Event]] = set()

    def __call__(self, decision: EgressDecision) -> None:
        if self._emitter is None:
            return
        event = Event(
            type=EventType.EGRESS_REQUEST_AUTHORIZED
            if decision.allowed
            else EventType.EGRESS_REQUEST_DENIED,
            session_id=self._session_id,
            agent_name=self._agent_name,
            environment_name=self._environment_name,
            payload={
                "allowed": decision.allowed,
                "status_code": decision.status_code,
                "destination": decision.destination,
                "method": decision.method,
                "path": decision.path,
                "grant_id": decision.grant_id,
                "policy_name": decision.policy_name,
                "reason": decision.reason,
                "authorization_kind": decision.authorization_kind,
            },
        )
        emitter = self._emitter

        async def _emit() -> Event:
            return await emitter(event)

        coro = _emit()
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except RuntimeError:
            # Loop stopped (e.g. a late request during teardown): close the
            # coroutine so it isn't left un-awaited.
            coro.close()
            return
        self._pending.add(future)
        future.add_done_callback(self._pending.discard)

    async def drain(self) -> None:
        while self._pending:
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in tuple(self._pending)),
                return_exceptions=True,
            )


class _EgressTeardownBinding(WorkspaceBinding):
    """Wraps an inner binding and runs egress teardown at session end."""

    def __init__(
        self,
        *,
        inner: WorkspaceBinding,
        runner: _EgressManagedRunner,
        grants: Sequence[VirtualCredentialGrant],
        emitter: EventEmitter | None,
        audit: _EgressAuditBridge | None,
        session_id: str,
        agent_name: str,
        environment_name: str,
    ) -> None:
        self._inner = inner
        self._runner = runner
        self._grants = tuple(grants)
        self._emitter = emitter
        self._audit = audit
        self._session_id = session_id
        self._agent_name = agent_name
        self._environment_name = environment_name

    async def bind(
        self,
        workspace: Any,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        try:
            return await self._inner.bind(
                workspace,
                runner,
                session_id=session_id,
                agent_name=agent_name,
                environment_name=environment_name,
                metadata=metadata,
            )
        except BaseException:
            # Runtime does not call finalize() after bind failure, so the
            # virtual-egress resources must be released here.
            cancelled = False
            try:
                await self._close_resources()
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                # Preserve the binding failure; there is no successfully
                # returned runner/binding handle for the caller to retry.
                pass
            await self._drain_audit()
            await self._emit_revoked()
            if cancelled:
                raise asyncio.CancelledError() from None
            raise

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        snapshot: WorkspaceSnapshot | None = None
        inner_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        cancelled = False
        try:
            # Disable guest-side authority before workspace commands run. The
            # runner stays alive until the workspace has flushed and unmounted.
            cancelled = await self._runner.revoke_authority()
        except BaseException as exc:
            cleanup_error = exc
        try:
            snapshot = await self._inner.finalize(bound, outcome=outcome, metadata=metadata)
        except BaseException as exc:
            inner_error = exc
        try:
            # Workspace sync/unmount must finish before an interrupted MicroVM
            # is suspended or a terminal MicroVM is terminated.
            await self._close_resources(outcome=outcome)
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        await self._drain_audit()
        await self._emit_revoked()
        if inner_error is not None:
            raise inner_error
        if cleanup_error is not None:
            raise cleanup_error
        if cancelled:
            raise asyncio.CancelledError()
        return snapshot

    async def _close_resources(
        self,
        *,
        outcome: str | None = None,
    ) -> None:
        cancelled = await _await_cleanup(self._runner.finalize(outcome=outcome))
        if cancelled:
            raise asyncio.CancelledError()

    async def _drain_audit(self) -> None:
        if self._audit is None:
            return
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await self._audit.drain()

    async def _emit_revoked(self) -> None:
        if self._emitter is None:
            return
        with contextlib.suppress(Exception):
            for grant in self._grants:
                await self._emitter(
                    Event(
                        type=EventType.EGRESS_GRANT_REVOKED,
                        session_id=self._session_id,
                        agent_name=self._agent_name,
                        environment_name=self._environment_name,
                        payload=_grant_payload(grant),
                    )
                )


def _grant_payload(grant: VirtualCredentialGrant) -> dict[str, Any]:
    return {
        "grant_id": grant.grant_id,
        "destination": grant.destination,
        "credential_kind": grant.credential_kind,
        "policy_name": grant.policy_name,
        "env_name": grant.env_name,
    }


def _required_binding_field(binding: EgressBinding, field_name: str) -> str:
    value = getattr(binding, field_name)
    if not isinstance(value, str) or not value:
        raise UnsupportedEgressError(
            f"Egress adapter did not return {field_name}; refusing to start "
            "a virtual-egress sandbox with an incomplete adapter binding."
        )
    return value


def _duplicate_env_names(credentials: Sequence[VirtualCredentialSpec]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for credential in credentials:
        if credential.env_name in seen:
            duplicates.add(credential.env_name)
        seen.add(credential.env_name)
    return tuple(sorted(duplicates))


def _ordered_destinations(
    grants: Sequence[VirtualCredentialGrant],
    approved_destinations: Sequence[ApprovedEgressDestination],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                *(grant.destination for grant in grants),
                *(destination.destination for destination in approved_destinations),
            ]
        )
    )
