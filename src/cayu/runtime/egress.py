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
import os
import secrets
import shutil
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cayu.core.events import Event, EventType
from cayu.egress import (
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
from cayu.egress.credential_kinds import validate_credential_kind
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import (
    BoundWorkspace,
    NoWorkspaceBinding,
    WorkspaceBinding,
    WorkspaceSnapshot,
)
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.runners.base import DEFAULT_EXEC_OUTPUT_LIMIT_BYTES, ExecCommand, ExecResult, Runner
from cayu.vaults import SecretRef, SecretResolver

EventEmitter = Callable[[Event], Awaitable[Event]]

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
        resolver: SecretResolver,
        policies: Mapping[str, EgressPolicy],
        credentials: Sequence[VirtualCredentialSpec],
        image: str = DEFAULT_SANDBOX_IMAGE,
        setup_commands: Sequence[str] = (),
        adapter: SandboxEgressAdapter | None = None,
        adapter_registry: EgressAdapterRegistry | None = None,
        runner_kind: str = "docker",
        inner_binding: WorkspaceBinding | None = None,
        event_emitter: EventEmitter | None = None,
        upstream: EgressUpstream | None = None,
        require_test_mode_credentials: bool = True,
    ) -> None:
        if not credentials:
            raise ValueError("VirtualEgressEnvironmentFactory requires at least one credential.")
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
        self._image = image
        self._setup_commands = tuple(setup_commands)
        self._adapter = adapter
        self._adapter_registry = adapter_registry
        self._runner_kind = adapter.runner_kind if adapter is not None else runner_kind
        self._inner_binding = inner_binding or NoWorkspaceBinding()
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
            upstream=self._upstream,
            audit=audit,
            require_test_mode_credentials=self._require_test_mode,
        )
        grant_revoker = _EgressGrantRevoker(registry=registry, grants=grants)

        # From here on, adapter.prepare may return resources (proxy thread +
        # docker network + sidecar) before workspace binding/finalization is
        # guaranteed. Guard the whole handoff so the factory owns cleanup until
        # the returned Environment owns it.
        binding: EgressBinding | None = None
        ca_dir: str | None = None
        runner: Runner | None = None
        managed_runner: _EgressManagedRunner | None = None
        try:
            binding = await adapter.prepare(
                session_id=request.session_id,
                grants=grants,
                broker=broker,
            )
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
                egress_destinations=tuple(grant.destination for grant in grants),
            )
            runner = await adapter.create_runner(runner_request)

            managed_runner = _EgressManagedRunner(
                runner=runner,
                egress_binding=binding,
                ca_dir=ca_dir,
                grant_revoker=grant_revoker,
                audit=audit,
            )
            runner = None
            binding = None
            ca_dir = None
            teardown_binding = _EgressTeardownBinding(
                inner=self._inner_binding,
                runner=managed_runner,
                grants=grants,
                grant_revoker=grant_revoker,
                emitter=self._emitter,
                audit=audit,
                session_id=request.session_id,
                agent_name=request.agent_name,
                environment_name=request.environment_name,
            )

            await self._emit_grant_events(request, grants, runner_kind=runner_kind)
        except BaseException:
            if managed_runner is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await managed_runner.close()
            else:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await grant_revoker.revoke()
                if runner is not None:
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await runner.close()
            if binding is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await binding.close()
            if ca_dir is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    shutil.rmtree(ca_dir, ignore_errors=True)
            raise
        spec = EnvironmentSpec(
            name=request.environment_name,
            metadata={"kind": runner_kind, "credential_mode": CredentialMode.VIRTUAL_EGRESS.value},
        )
        environment = Environment(
            spec,
            runner=managed_runner,
            binding=teardown_binding,
        )
        return EnvironmentFactoryResult(environment=environment)

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


class _EgressGrantRevoker:
    """Disables a session's virtual credentials before slow resource cleanup."""

    def __init__(
        self,
        *,
        registry: VirtualCredentialRegistry,
        grants: Sequence[VirtualCredentialGrant],
    ) -> None:
        self._registry = registry
        self._presented_values = tuple(grant.presented_value for grant in grants)
        self._grant_ids = tuple(grant.grant_id for grant in grants)
        self._revoked = False
        self._drained = False
        self._task: asyncio.Task[None] | None = None

    async def revoke(self) -> bool:
        if self._drained:
            return False
        if self._task is None:
            self._task = asyncio.create_task(self._revoke_and_wait())
        cancelled = await _await_cleanup_task(self._task)
        self._drained = True
        return cancelled

    async def _revoke_and_wait(self) -> None:
        if not self._revoked:
            for value in self._presented_values:
                self._registry.revoke(value)
            self._revoked = True
        await self._registry.wait_for_inactive_grants(self._grant_ids)


async def _await_cleanup_task(task: asyncio.Task[None]) -> bool:
    """Wait for cleanup to finish even if the awaiting task is cancelled."""
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
        egress_binding: EgressBinding,
        ca_dir: str,
        grant_revoker: _EgressGrantRevoker,
        audit: _EgressAuditBridge | None = None,
    ) -> None:
        self._runner = runner
        self._egress_binding = egress_binding
        self._ca_dir = ca_dir
        self._grant_revoker = grant_revoker
        self._audit = audit
        self.isolation = runner.isolation
        self.default_cwd = runner.default_cwd

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)

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

    async def close(self) -> None:
        if self._closed:
            return
        cancelled = False
        revocation_error: BaseException | None = None
        grants_drained = False
        try:
            cancelled = await self._grant_revoker.revoke()
            grants_drained = True
        except asyncio.CancelledError as exc:
            cancelled = True
            revocation_error = exc
        except Exception as exc:
            revocation_error = exc
        with contextlib.suppress(Exception, asyncio.CancelledError):
            cancelled = await _await_cleanup(self._runner.close()) or cancelled
        with contextlib.suppress(Exception, asyncio.CancelledError):
            cancelled = await _await_cleanup(self._egress_binding.close()) or cancelled
        if self._audit is not None:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                cancelled = await _await_cleanup(self._audit.drain()) or cancelled
        with contextlib.suppress(Exception, asyncio.CancelledError):
            shutil.rmtree(self._ca_dir, ignore_errors=True)
        if revocation_error is None and grants_drained:
            self._closed = True
        if revocation_error is not None:
            raise revocation_error
        if cancelled:
            raise asyncio.CancelledError()

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
        runner: Runner,
        grants: Sequence[VirtualCredentialGrant],
        grant_revoker: _EgressGrantRevoker | None = None,
        emitter: EventEmitter | None,
        audit: _EgressAuditBridge | None,
        session_id: str,
        agent_name: str,
        environment_name: str,
    ) -> None:
        self._inner = inner
        self._runner = runner
        self._grants = tuple(grants)
        self._grant_revoker = grant_revoker
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
        cancelled = False
        try:
            cancelled = await self._revoke_grants() or cancelled
        except asyncio.CancelledError:
            cancelled = True
        try:
            return await self._inner.finalize(bound, outcome=outcome, metadata=metadata)
        finally:
            # Grants are already revoked before sync-back; this closes the slower
            # runner/proxy/CA resources even if the inner binding failed.
            try:
                await self._close_resources(revoke=False)
            except asyncio.CancelledError:
                cancelled = True
            await self._drain_audit()
            await self._emit_revoked()
            if cancelled:
                raise asyncio.CancelledError()

    async def _revoke_grants(self) -> bool:
        if self._grant_revoker is None:
            return False
        try:
            return await self._grant_revoker.revoke()
        except asyncio.CancelledError:
            return True
        except Exception:
            return False

    async def _close_resources(self, *, revoke: bool = True) -> None:
        cancelled = False
        if revoke:
            cancelled = await self._revoke_grants() or cancelled
        with contextlib.suppress(Exception, asyncio.CancelledError):
            cancelled = await _await_cleanup(self._runner.close()) or cancelled
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
