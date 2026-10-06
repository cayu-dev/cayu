"""Deferred environment materialization: create a runner on first use.

A factory that opts in returns an environment whose ``runner`` is a
:class:`DeferredRunner` and whose ``binding`` is a :class:`DeferredWorkspaceBinding`
sharing one :class:`DeferredMaterialization`. Session start binds nothing. The
first command, workspace operation or finalization that needs the runner
materializes it once (single-flight), runs the factory's real binding against
it, and only then performs the original call. A run that never uses the runner
never creates it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, BinaryIO, Literal

from cayu._validation import require_clean_nonblank
from cayu.environments.bindings import BoundWorkspace, WorkspaceBinding
from cayu.runners import DEFAULT_EXEC_OUTPUT_LIMIT_BYTES, ExecCommand, Runner
from cayu.runners.base import (
    RunnerBinaryStreamCapability,
    RunnerExecutionAdmissionObserver,
    RunnerUnavailableError,
)
from cayu.workspaces.revisions import WorkspaceIdentity, WorkspaceRevisionObservation

if TYPE_CHECKING:
    from cayu.environments.admission import (
        ExecutionAdmissionCandidate,
        ExecutionEnvironmentAuthority,
        ExecutionRequirements,
    )
    from cayu.environments.bindings import WorkspaceSnapshot
    from cayu.runners import ExecResult
    from cayu.workspaces import Workspace
    from cayu.workspaces.revisions import WorkspaceWriterIsolationEvidence

MaterializationMode = Literal["use", "recover"]
"""``use`` creates a fresh resource; ``recover`` must adopt the existing one.

A clean run end finalizes the binding and disposes the resource, so a resource
found under the reserved identity in ``use`` mode is a crash leftover: remove
it and create a fresh one instead of adopting it.
"""

MaterializationEventKind = Literal[
    "started", "completed", "failed", "binding_started", "binding_completed"
]
MaterializationObserver = Callable[
    [MaterializationEventKind, dict[str, Any], "BoundWorkspace | None"], Awaitable[None]
]
MaterializationAdmission = Callable[[Runner], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class MaterializationTrigger:
    """The tool call whose first runner use materialized the environment.

    ``event_sink`` receives the persisted materialization events so the tool
    round can yield them in the live run stream next to its runner events.
    """

    tool_call_id: str | None
    tool_name: str | None
    event_sink: Callable[[Any], None] | None = None


_TRIGGER: ContextVar[MaterializationTrigger | None] = ContextVar(
    "cayu_deferred_materialization_trigger", default=None
)


@contextmanager
def materialization_trigger(
    *,
    tool_call_id: str | None,
    tool_name: str | None,
    event_sink: Callable[[Any], None] | None = None,
) -> Iterator[None]:
    """Attribute any materialization inside this scope to one tool call."""

    token = _TRIGGER.set(
        MaterializationTrigger(
            tool_call_id=tool_call_id, tool_name=tool_name, event_sink=event_sink
        )
    )
    try:
        yield
    finally:
        _TRIGGER.reset(token)


def current_materialization_trigger() -> MaterializationTrigger | None:
    """Return the tool call attributed to materializations in this context."""

    return _TRIGGER.get()


class EnvironmentMaterializationError(RunnerUnavailableError):
    """A deferred environment could not be created, admitted or bound.

    The triggering tool call fails with this error; the session stays usable and
    a later runner use attempts materialization again.
    """

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(
            message,
            diagnostic={"kind": "environment_materialization_failed", "reason": reason},
        )
        self.reason = reason


@dataclass(slots=True)
class _Materialized:
    runner: Runner
    binding: WorkspaceBinding | None
    bound: BoundWorkspace | None


@dataclass(slots=True)
class _BindRequest:
    workspace: Workspace | None
    session_id: str
    agent_name: str | None
    environment_name: str | None
    metadata: dict[str, Any] | None


MaterializeCallable = Callable[
    [MaterializationMode], Awaitable[tuple[Runner, WorkspaceBinding | None]]
]


class DeferredMaterialization:
    """Shared single-flight state behind one deferred runner and binding.

    ``materialize(mode)`` is factory-owned and returns the exact runner plus the
    real binding to run against it. It must reuse the factory's ordinary
    creation path so admission and isolation evidence match an eager start.
    ``dispose_unmaterialized`` removes a resource a crashed process created but
    never bound; it runs at finalization when this process never materialized.
    """

    def __init__(
        self,
        materialize: MaterializeCallable,
        *,
        default_cwd: str,
        placeholder_workspace: Callable[[Runner], Workspace | None] | None = None,
        configured_candidate: Callable[[ExecutionRequirements], ExecutionAdmissionCandidate | None]
        | None = None,
        environment_authority: ExecutionEnvironmentAuthority | None = None,
        dispose_unmaterialized: Callable[[], Awaitable[None]] | None = None,
        isolation: str = "unknown",
    ) -> None:
        if not callable(materialize):
            raise TypeError("materialize must be callable.")
        self._materialize = materialize
        self._lock = asyncio.Lock()
        self._state: _Materialized | None = None
        # Created for an admission collection, not yet admitted or bound.
        self._created: _Materialized | None = None
        self._created_started = 0.0
        # Runners that must never be used again but whose close has not yet
        # succeeded; finalization and the wrapper's close retry them.
        self._pending_close: list[Runner] = []
        self._bind_request: _BindRequest | None = None
        self._observer: MaterializationObserver | None = None
        self._admission: MaterializationAdmission | None = None
        self._dispose_unmaterialized = dispose_unmaterialized
        self._finalized = False
        self.runner = DeferredRunner(
            self,
            default_cwd=default_cwd,
            configured_candidate=configured_candidate,
            environment_authority=environment_authority,
            isolation=isolation,
        )
        self.binding = DeferredWorkspaceBinding(self)
        self._placeholder_workspace = (
            None if placeholder_workspace is None else placeholder_workspace(self.runner)
        )

    @property
    def is_materialized(self) -> bool:
        return self._state is not None

    @property
    def materialized_runner(self) -> Runner | None:
        """The live runner once created, even before its first-use bind."""

        if self._state is not None:
            return self._state.runner
        return None if self._created is None else self._created.runner

    def attach_observer(self, observer: MaterializationObserver | None) -> None:
        """Receive started/completed/failed notifications (runtime-owned)."""

        self._observer = observer

    def attach_admission(self, admission: MaterializationAdmission | None) -> None:
        """Run the runtime's live admission on the new runner before it is used.

        A refusal raises; the runner is closed and the triggering call fails.
        """

        self._admission = admission

    async def _notify(
        self,
        kind: MaterializationEventKind,
        payload: dict[str, Any],
        bound: BoundWorkspace | None = None,
    ) -> None:
        observer = self._observer
        if observer is None:
            return
        trigger = _TRIGGER.get()
        await observer(
            kind,
            {
                **payload,
                "trigger_tool_call_id": None if trigger is None else trigger.tool_call_id,
                "trigger_tool_name": None if trigger is None else trigger.tool_name,
            },
            bound,
        )

    async def materialize(self, mode: MaterializationMode = "use") -> Runner:
        """Return the materialized runner, creating and binding it once."""

        return await self._materialize_once(mode, run_admission=True)

    async def _materialize_once(
        self, mode: MaterializationMode, *, run_admission: bool, bind: bool = True
    ) -> Runner:
        """Create the runner (once), admit it, and bind it (once).

        ``run_admission=False, bind=False`` is only for an admission observer's
        collection: it creates the runner without copying anything in, and its
        caller collects and evaluates the live evidence. The bind then happens
        on the first use, after that admission passed, and without admitting
        the same runner twice. Nothing is ever bound before admission.
        """

        if self._state is not None:
            return self._state.runner
        if not bind and self._created is not None:
            return self._created.runner
        async with self._lock:
            if self._state is not None:
                return self._state.runner
            if not bind and self._created is not None:
                return self._created.runner
            if self._finalized:
                raise EnvironmentMaterializationError(
                    "The deferred environment was already finalized.", reason="finalized"
                )
            created = self._created
            if created is None:
                started = time.monotonic()
                await self._notify("started", {"mode": mode})
            else:
                # One materialization per runner: the create already recorded
                # "started"; this bind completes (or fails) that same record.
                started = self._created_started
            runner: Runner | None = None if created is None else created.runner
            try:
                if created is None:
                    runner, binding = await self._materialize(mode)
                    if not isinstance(runner, Runner) or isinstance(runner, DeferredRunner):
                        raise TypeError("Deferred materialization must return a concrete Runner.")
                    if mode == "use" and run_admission and self._admission is not None:
                        # Live admission before copy-in, so a refused runner never
                        # receives workspace content or runs the triggering call.
                        try:
                            await self._admission(runner)
                        except EnvironmentMaterializationError:
                            raise
                        except Exception as refusal:
                            raise EnvironmentMaterializationError(
                                "The deferred environment was refused by execution admission.",
                                reason="admission_refused",
                            ) from refusal
                    created = _Materialized(runner=runner, binding=binding, bound=None)
                    self._created = created
                    self._created_started = started
                else:
                    # Created for an admission collection whose caller admitted it.
                    binding = created.binding
                    runner = created.runner
                if bind:
                    bound = None
                    request = self._bind_request
                    if binding is not None and request is not None and mode == "use":
                        await self._notify("binding_started", {})
                        bound = await binding.bind(
                            request.workspace,
                            runner,
                            session_id=request.session_id,
                            agent_name=request.agent_name,
                            environment_name=request.environment_name,
                            metadata=request.metadata,
                        )
                        await self._notify("binding_completed", {}, bound)
                    self._state = _Materialized(runner=runner, binding=binding, bound=bound)
                    self._created = None
                else:
                    # Created for an admission collection: the materialization
                    # completes when the first use after admission binds it.
                    return runner
            except BaseException as error:
                if runner is not None and self._state is None:
                    # Never reuse it, but keep it until its close succeeds.
                    self._created = None
                    self._pending_close.append(runner)
                    with suppress(Exception):
                        await self._close_pending()
                reason = (
                    error.reason
                    if isinstance(error, EnvironmentMaterializationError)
                    else type(error).__name__
                )
                if isinstance(error, asyncio.CancelledError):
                    # Close the started record; the cancellation still propagates.
                    # Shielded so a repeated cancellation cannot leave it half-written.
                    with suppress(BaseException):
                        await asyncio.shield(
                            self._notify(
                                "failed",
                                {
                                    "mode": mode,
                                    "reason": "cancelled",
                                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                                },
                            )
                        )
                elif isinstance(error, Exception):
                    await self._notify(
                        "failed",
                        {
                            "mode": mode,
                            "reason": reason,
                            "elapsed_ms": int((time.monotonic() - started) * 1000),
                        },
                    )
                    if isinstance(error, EnvironmentMaterializationError):
                        raise
                    raise EnvironmentMaterializationError(
                        f"Deferred environment materialization failed: {type(error).__name__}.",
                        reason=reason,
                    ) from error
                raise
            await self._notify(
                "completed",
                {"mode": mode, "elapsed_ms": int((time.monotonic() - started) * 1000)},
                self._state.bound,
            )
            return runner

    async def _close_pending(self) -> None:
        """Close every retired runner; one whose close fails stays pending."""

        failure: BaseException | None = None
        for runner in tuple(self._pending_close):
            try:
                await runner.close()
            except BaseException as error:
                failure = failure or error
                continue
            if runner in self._pending_close:
                self._pending_close.remove(runner)
        if failure is not None:
            raise failure

    def _install_recovered(self, runner: Runner, binding: WorkspaceBinding, bound: BoundWorkspace):
        self._state = _Materialized(runner=runner, binding=binding, bound=bound)


class _DeferredAdmissionObserver(RunnerExecutionAdmissionObserver):
    """Report configured evidence until live evidence is genuinely required."""

    async def collect(self) -> ExecutionAdmissionCandidate | None:
        runner = self.runner
        assert isinstance(runner, DeferredRunner)
        return await runner._collect_admission(self.requirements)


class _DeferredBinaryStreamCapability(RunnerBinaryStreamCapability):
    def __init__(self, runner: DeferredRunner) -> None:
        self._runner = runner

    async def exec_stream(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        return await self._runner.exec_stream(command, **kwargs)


class DeferredRunner(Runner):
    """A runner that materializes on first command execution.

    Command, workspace and admission calls that need a live target materialize
    once and then delegate to the exact runner the factory created. Admission
    for requirements that only need declared evidence is answered from the
    factory's configured candidate, so session start dispatches nothing.
    Requirements that need live evidence (declared executables) materialize
    eagerly at session start, exactly like an eager environment.
    """

    def __init__(
        self,
        state: DeferredMaterialization,
        *,
        default_cwd: str,
        configured_candidate: Callable[[ExecutionRequirements], ExecutionAdmissionCandidate | None]
        | None,
        environment_authority: ExecutionEnvironmentAuthority | None,
        isolation: str,
    ) -> None:
        self._deferred = state
        self.default_cwd = default_cwd
        self.isolation = isolation
        self._configured_candidate = configured_candidate
        self._authority = environment_authority

    @property
    def materialization(self) -> DeferredMaterialization:
        return self._deferred

    def _live(self) -> Runner | None:
        return self._deferred.materialized_runner

    async def _target(self) -> Runner:
        self._ensure_exec_open()
        return await self._deferred.materialize("use")

    def resolve_cwd(self, cwd: str | None = None) -> str:
        live = self._live()
        return live.resolve_cwd(cwd) if live is not None else super().resolve_cwd(cwd)

    def preflight_exec(self, command: ExecCommand, **kwargs: Any) -> None:
        live = self._live()
        if live is not None:
            live.preflight_exec(command, **kwargs)
            return
        super().preflight_exec(command, **kwargs)

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        target = await self._target()
        return await target.exec(
            command,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )

    async def exec_redacted(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        target = await self._target()
        return await target.exec_redacted(command, **kwargs)

    async def exec_system(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        target = await self._target()
        return await target.exec_system(command, **kwargs)

    def binary_stream_capability(self) -> RunnerBinaryStreamCapability | None:
        live = self._live()
        if live is None or live.binary_stream_capability() is None:
            return None
        return _DeferredBinaryStreamCapability(self)

    async def exec_stream(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
        stdout_limit_bytes: int | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        target = await self._target()
        capability = target.binary_stream_capability()
        if capability is None:
            raise RuntimeError("The materialized runner does not support binary streams.")
        return await capability.exec_stream(
            command,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            stdout=stdout,
            stdout_limit_bytes=stdout_limit_bytes,
            output_limit_bytes=output_limit_bytes,
        )

    async def close(self) -> None:
        await self._deferred._close_pending()
        live = self._live()
        if live is not None:
            await live.close()
        self._closed = True

    async def await_pending_command_settlement(self) -> bool:
        live = self._live()
        return True if live is None else await live.await_pending_command_settlement()

    async def refresh_execution_admission(self) -> None:
        live = self._live()
        if live is not None:
            await live.refresh_execution_admission()

    def execution_environment_authority(self) -> ExecutionEnvironmentAuthority:
        if self._authority is not None:
            return self._authority
        return super().execution_environment_authority()

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None:
        live = self._live()
        if live is not None:
            return live.execution_admission_candidate()
        if self._configured_candidate is None:
            return None
        from cayu.environments.admission import ExecutionRequirements

        return self._configured_candidate(ExecutionRequirements.trusted())

    def execution_admission_candidate_for(
        self, requirements: ExecutionRequirements
    ) -> ExecutionAdmissionCandidate | None:
        live = self._live()
        if live is not None:
            return live.execution_admission_candidate_for(requirements)
        return (
            None if self._configured_candidate is None else self._configured_candidate(requirements)
        )

    def execution_admission_observer(
        self, requirements: ExecutionRequirements
    ) -> RunnerExecutionAdmissionObserver:
        return _DeferredAdmissionObserver(self, requirements)

    async def _collect_admission(
        self, requirements: ExecutionRequirements
    ) -> ExecutionAdmissionCandidate | None:
        live = self._live()
        if live is None and (requirements.executable_names() or self._configured_candidate is None):
            # Live executable evidence cannot be answered from configuration
            # (pre_exposure admission requires live_verified executables).
            # Create the runner but bind nothing: the caller of this collection
            # evaluates admission on the returned evidence, and copy-in waits
            # for the first use after that admission passed.
            live = await self._deferred._materialize_once("use", run_admission=False, bind=False)
        if live is None:
            return self.execution_admission_candidate_for(requirements)
        return await live.execution_admission_observer(requirements).collect()

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        live = self._live()
        return None if live is None else live.resource_key

    def workspace_capability(self, capability_type):
        live = self._live()
        if live is None:
            return super().workspace_capability(capability_type)
        return live.workspace_capability(capability_type)

    def output_secret_values_present(self) -> bool | None:
        live = self._live()
        return False if live is None else live.output_secret_values_present()


class DeferredWorkspaceBinding(WorkspaceBinding):
    """Bind nothing at session start; run the real binding at materialization.

    Finalization, recovery and abandonment delegate to the real binding once
    the environment materialized, and are no-ops (apart from disposing an
    unbound resource left by a crashed process) when it never did.
    """

    def __init__(self, state: DeferredMaterialization) -> None:
        self._deferred = state

    @property
    def materialization(self) -> DeferredMaterialization:
        return self._deferred

    def _materialized(self) -> _Materialized | None:
        return self._deferred._state

    async def bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        if runner is not self._deferred.runner:
            raise ValueError("Deferred binding requires its own deferred runner.")
        self._deferred._bind_request = _BindRequest(
            workspace=workspace,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=None if metadata is None else dict(metadata),
        )
        return BoundWorkspace(
            workspace=self._deferred._placeholder_workspace,
            source_workspace=workspace,
            runner=runner,
            path=self._deferred.runner.default_cwd,
            metadata={"deferred": True},
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        state = self._materialized()
        self._deferred._finalized = True
        # A runner whose earlier close failed is retried before anything else;
        # if it still fails, finalization fails and a later attempt retries it.
        await self._deferred._close_pending()
        if state is None:
            created = self._deferred._created
            if created is not None:
                # Created for admission but never bound (for example refused):
                # nothing to publish, so only its resource is removed, and its
                # open materialization record ends as failed. It moves to the
                # pending list first, so a failed close keeps it owned.
                self._deferred._created = None
                self._deferred._pending_close.append(created.runner)
                try:
                    await self._deferred._notify(
                        "failed",
                        {
                            "mode": "use",
                            "reason": "not_bound",
                            "elapsed_ms": int(
                                (time.monotonic() - self._deferred._created_started) * 1000
                            ),
                        },
                    )
                finally:
                    await self._deferred._close_pending()
                return None
            if self._deferred._dispose_unmaterialized is not None:
                await self._deferred._dispose_unmaterialized()
            return None
        if state.binding is None or state.bound is None:
            await state.runner.close()
            return None
        return await state.binding.finalize(state.bound, outcome=outcome, metadata=metadata)

    def abandon(self, bound: BoundWorkspace) -> bool:
        state = self._materialized()
        if state is None or state.binding is None or state.bound is None:
            return True
        return state.binding.abandon(state.bound)

    def _completion_requires_successful_finalization(self, bound: BoundWorkspace) -> bool:
        state = self._materialized()
        if state is None or state.binding is None or state.bound is None:
            return False
        return state.binding._completion_requires_successful_finalization(state.bound)

    def _completion_finalization_recovery_state(
        self, bound: BoundWorkspace
    ) -> dict[str, Any] | None:
        state = self._materialized()
        if state is None or state.binding is None or state.bound is None:
            return None
        inner = state.binding._completion_finalization_recovery_state(state.bound)
        if inner is None:
            return None
        return {"version": 1, "kind": "deferred_binding", "inner": inner}

    async def _recover_completion_finalization(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None,
        environment_name: str | None,
        recovery_state: dict[str, Any],
    ) -> BoundWorkspace:
        if recovery_state.get("kind") != "deferred_binding" or not isinstance(
            recovery_state.get("inner"), dict
        ):
            raise ValueError("Deferred binding recovery state has an unsupported format.")
        # Recovery must reach the exact resource that holds unpublished output.
        live, binding = await self._deferred._materialize("recover")
        if binding is None:
            raise RuntimeError("Deferred recovery requires the factory's real binding.")
        inner_bound = await binding._recover_completion_finalization(
            workspace,
            live,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            recovery_state=recovery_state["inner"],
        )
        self._deferred._install_recovered(live, binding, inner_bound)
        return BoundWorkspace(
            workspace=self._deferred._placeholder_workspace,
            source_workspace=workspace,
            runner=runner,
            path=self._deferred.runner.default_cwd,
            metadata={"deferred": True},
        )

    async def observe_revision(self, bound: BoundWorkspace) -> WorkspaceRevisionObservation:
        """Observe the materialized workspace under this binding's identity.

        The runtime registered this binding, with the session-start placeholder
        workspace, so it expects that identity. The real binding's observation
        describes the same workspace once materialized and is re-stamped with
        it. Before materialization nothing exists to observe: unsupported.
        """

        state = self._materialized()
        if state is None or state.binding is None or state.bound is None:
            return await super().observe_revision(bound)
        observed = await state.binding.observe_revision(state.bound)
        if type(observed) is not WorkspaceRevisionObservation:
            return observed  # The runtime's own validation rejects it.
        workspace = bound.workspace or bound.source_workspace
        identity = WorkspaceIdentity(
            workspace_id=(
                "workspace-unavailable"
                if workspace is None
                else require_clean_nonblank(workspace.id, "workspace.id")
            ),
            observer=type(self).__name__,
        )
        return observed.model_copy(update={"identity": identity})

    def observe_writer_isolation(self, bound: BoundWorkspace) -> WorkspaceWriterIsolationEvidence:
        state = self._materialized()
        if state is None or state.binding is None or state.bound is None:
            return super().observe_writer_isolation(bound)
        return state.binding.observe_writer_isolation(state.bound)


__all__ = [
    "DeferredMaterialization",
    "DeferredRunner",
    "DeferredWorkspaceBinding",
    "EnvironmentMaterializationError",
    "MaterializationMode",
    "MaterializationTrigger",
    "current_materialization_trigger",
    "materialization_trigger",
]
