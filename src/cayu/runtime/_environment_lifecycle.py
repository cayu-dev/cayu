"""Session-scoped environment provisioning, binding, and finalization.

The lifecycle owns concrete environment resources and their durable reconnect
state. Session orchestration, task ownership, hook execution, and terminal
status decisions remain with :class:`CayuApp`.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from cayu._task_wait import await_shielded_task_outcome
from cayu._validation import copy_json_value, copy_label_map
from cayu.core.events import Event, EventType
from cayu.environments import (
    BoundWorkspace,
    Environment,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    ExecutionAdmissionCandidate,
    ExecutionAdmissionError,
    ExecutionRequirements,
    WorkspaceInstructions,
    WorkspaceSnapshot,
    copy_environment,
    copy_workspace_snapshot,
    evaluate_execution_admission,
    load_workspace_instructions,
)
from cayu.runners import Runner
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._binding_cleanup import (
    BINDING_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE,
    append_binding_finalize_cancellation,
    attach_binding_cleanup_status,
    binding_cleanup_payload,
    binding_cleanup_status,
    binding_finalize_cancellation,
    binding_finalize_error_details,
    binding_finalize_explicit_cancellation,
    binding_finalize_failure_payload,
    binding_finalize_fatal_signal,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime.sessions import CheckpointTransform, Session, SessionStore
from cayu.vaults import SecretRedactor

ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY = "environment_factory_reconnect"
ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY = "environment_factory_allocation_owner"
_ENVIRONMENT_FACTORY_CHECKPOINT_MAY_BE_COMMITTED_ATTRIBUTE = (
    "_cayu_environment_factory_checkpoint_may_be_committed"
)
_ENVIRONMENT_FACTORY_RELEASE_ERROR_ATTRIBUTE = "_cayu_environment_factory_release"

CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]


@dataclass(frozen=True)
class EnvironmentBindingResult:
    registered_environment: runtime_records.RegisteredEnvironment | None
    events: list[Event]
    error: Exception | None = None


@dataclass(frozen=True)
class EnvironmentFactoryResolutionResult:
    registered_environment: runtime_records.RegisteredEnvironment | None
    events: list[Event]
    error: Exception | None = None


@dataclass(frozen=True)
class EnvironmentBindingFinalizeResult:
    event: Event
    events: list[Event]


@dataclass
class _ActiveEnvironmentSetup:
    registered_environment: runtime_records.RegisteredEnvironment
    cleanup_started: bool = False
    cleanup_error: BaseException | None = None


class EnvironmentLifecycle:
    """Own environment factory, workspace binding, and reconnect state."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        checkpoint_transform: CheckpointTransformFactory,
        secret_redactor: SecretRedactor | None = None,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._checkpoint_transform = checkpoint_transform
        self._secret_redactor = secret_redactor or SecretRedactor()
        # Factory results and bound workspaces contain process-local handles
        # that cannot be reconstructed from durable session state. Retain the
        # authoritative owner across async-generator yield boundaries until the
        # setup is adopted or finalized. The run fence permits one owner per
        # session.
        self._active_environment_setups: dict[str, _ActiveEnvironmentSetup] = {}

    async def load_workspace_instructions(
        self,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> WorkspaceInstructions | None:
        if registered_environment is None:
            return None
        return await load_workspace_instructions(registered_environment.environment)

    async def emit_factory_started(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> Event | None:
        """Persist the factory acceptance boundary before provisioning begins."""
        if registered_environment is None or registered_environment.factory is None:
            return None
        environment_name = registered_environment.spec.name
        return await self._event_writer.emit(
            Event(
                type=EventType.ENVIRONMENT_FACTORY_STARTED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=_environment_factory_base_payload(
                    session=session,
                    registered_environment=registered_environment,
                ),
            )
        )

    async def resolve_factory(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        started_event: Event | None,
        operation: EnvironmentFactoryOperation,
    ) -> EnvironmentFactoryResolutionResult:
        if registered_environment is None or registered_environment.factory is None:
            if started_event is not None:
                raise AssertionError("Factory start event exists without a registered factory.")
            return EnvironmentFactoryResolutionResult(
                registered_environment=registered_environment,
                events=[],
            )
        if started_event is None:
            raise AssertionError("Registered environment factory was not started.")

        factory = registered_environment.factory
        environment_name = registered_environment.spec.name
        base_payload = _environment_factory_base_payload(
            session=session,
            registered_environment=registered_environment,
        )
        events: list[Event] = []
        result: EnvironmentFactoryResult | None = None
        environment: Environment | None = None
        allocation_checkpointed = False
        allocation_checkpoint_may_be_committed = False
        effective_operation = operation
        try:
            reconnect_metadata, allocation_owner = await self._load_factory_reconnect_state(
                session_id=session.id,
                environment_name=environment_name,
            )
            if (
                operation is EnvironmentFactoryOperation.RECONNECT
                and allocation_owner != session.id
            ):
                # A failed setup has no session-owned allocation to reconnect,
                # and a fork inherits its parent's checkpoint only as context.
                # Allocation provenance is authoritative; Cayu does not scan
                # historical events to infer ownership.
                effective_operation = EnvironmentFactoryOperation.CREATE
            request = EnvironmentFactoryRequest(
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                operation=effective_operation,
                parent_session_id=session.parent_session_id,
                causal_budget_id=session.causal_budget_id,
                labels=session.labels,
                metadata=session.metadata,
                reconnect_metadata=reconnect_metadata,
                execution_requirements=registered_agent.execution_requirements,
            )
            admission_candidate = factory.execution_admission_candidate(request)
            if admission_candidate is not None and not isinstance(
                admission_candidate,
                ExecutionAdmissionCandidate,
            ):
                raise TypeError(
                    "EnvironmentFactory.execution_admission_candidate must return "
                    "ExecutionAdmissionCandidate or None."
                )
            evaluate_execution_admission(
                candidate=(
                    environment_name
                    if admission_candidate is None
                    else admission_candidate.candidate
                ),
                requirements=request.execution_requirements,
                evidence=None if admission_candidate is None else admission_candidate.evidence,
                stage="pre_create",
            ).require_admitted()
            result = await factory.create(request)
            if type(result) is not EnvironmentFactoryResult:
                raise TypeError("EnvironmentFactory.create must return EnvironmentFactoryResult.")
            environment = copy_environment(result.environment)
            if environment.spec.name != environment_name:
                raise ValueError(
                    "Environment factory returned a different environment name: "
                    f"{environment.spec.name!r} != {environment_name!r}"
                )
            if environment.runner is not None or environment.binding is None:
                self._require_runner_admitted(
                    execution_candidate=(
                        None if admission_candidate is None else admission_candidate.candidate
                    ),
                    fallback_candidate=environment_name,
                    requirements=request.execution_requirements,
                    runner=environment.runner,
                )
            reconnect_metadata = copy_json_value(
                result.reconnect_metadata,
                "reconnect_metadata",
            )
            try:
                await self._checkpoint_factory_reconnect_metadata(
                    session_id=session.id,
                    environment_name=environment_name,
                    reconnect_metadata=reconnect_metadata,
                )
            except BaseException as exc:
                # The checkpoint helper reconciles any failure after the
                # transactional write begins. Only a durable read proving the
                # expected owner and metadata absent permits the allocation to
                # be discarded.
                allocation_checkpoint_may_be_committed = bool(
                    getattr(
                        exc,
                        _ENVIRONMENT_FACTORY_CHECKPOINT_MAY_BE_COMMITTED_ATTRIBUTE,
                        False,
                    )
                )
                raise
            allocation_checkpointed = True
            events.append(
                await self._event_writer.emit(
                    Event(
                        type=EventType.ENVIRONMENT_FACTORY_COMPLETED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            **base_payload,
                            "environment_name": environment.spec.name,
                            "result_metadata": copy_json_value(
                                result.metadata,
                                "result_metadata",
                            ),
                            "reconnect_metadata": reconnect_metadata,
                        },
                    )
                )
            )
            if session.id in self._active_environment_setups:
                raise RuntimeError(
                    f"Session {session.id!r} already owns an active environment setup."
                )
            if result is None:
                raise RuntimeError("Environment factory did not return an owned result.")
            if environment is None:
                raise RuntimeError("Environment factory did not produce an environment.")
            resolved_environment = runtime_records.RegisteredEnvironment(
                spec=registered_environment.spec,
                environment=environment,
                execution_candidate=(
                    None if admission_candidate is None else admission_candidate.candidate
                ),
                unclaimed_factory_result=result,
            )
            self._active_environment_setups[session.id] = _ActiveEnvironmentSetup(
                registered_environment=resolved_environment
            )
        except BaseException as exc:
            if result is not None:
                release_payload = await _release_unclaimed_factory_result(
                    result,
                    action=(
                        EnvironmentFactoryReleaseAction.PRESERVE
                        if allocation_checkpointed
                        or allocation_checkpoint_may_be_committed
                        or effective_operation is EnvironmentFactoryOperation.RECONNECT
                        else EnvironmentFactoryReleaseAction.DISCARD
                    ),
                    original_error=exc,
                )
                setattr(exc, _ENVIRONMENT_FACTORY_RELEASE_ERROR_ATTRIBUTE, release_payload)
            ordinary_failure = isinstance(exc, Exception) or (
                isinstance(exc, BaseExceptionGroup) and exc.subgroup(Exception) is not None
            )
            fatal_signal = binding_finalize_fatal_signal(exc)
            if fatal_signal is not None and not ordinary_failure:
                raise
            if ordinary_failure:
                try:
                    events.append(
                        await self._event_writer.emit(
                            Event(
                                type=EventType.ENVIRONMENT_FACTORY_FAILED,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                payload={
                                    **base_payload,
                                    **exception_failure_payload(exc),
                                },
                            )
                        )
                    )
                except BaseException as publication_error:
                    raise BaseExceptionGroup(
                        "Environment factory failure publication also failed.",
                        [exc, publication_error],
                    ) from publication_error
            if fatal_signal is not None or not isinstance(exc, Exception):
                raise
            return EnvironmentFactoryResolutionResult(
                registered_environment=registered_environment,
                events=events,
                error=exc,
            )

        return EnvironmentFactoryResolutionResult(
            registered_environment=resolved_environment,
            events=events,
        )

    async def checkpoint_preserving_runtime_state(
        self,
        session_id: str,
        checkpoint: dict[str, Any],
    ) -> None:
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        runtime_keys = (
            ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY,
            ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY,
        )
        if any(key not in copied_checkpoint for key in runtime_keys):
            current_checkpoint = await self._session_store.load_checkpoint(session_id)
            if current_checkpoint is not None:
                for key in runtime_keys:
                    if key in copied_checkpoint:
                        continue
                    state = current_checkpoint.get(key)
                    if state is not None:
                        if type(state) is not dict:
                            raise ValueError(f"{key} checkpoint state must be an object.")
                        copied_checkpoint[key] = copy_json_value(state, key)
        await self._session_store.transform_checkpoint(
            session_id,
            self._checkpoint_transform(copied_checkpoint),
        )

    async def emit_binding_started(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> Event | None:
        """Persist the binding acceptance boundary before workspace setup begins."""
        if (
            registered_environment is None
            or registered_environment.bound_workspace is not None
            or registered_environment.environment.binding is None
        ):
            return None
        environment_name = _environment_name(registered_environment)
        return await self._event_writer.emit(
            Event(
                type=EventType.ENVIRONMENT_BINDING_STARTED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=_binding_base_payload(registered_environment),
            )
        )

    def _require_runner_admitted(
        self,
        *,
        execution_candidate: str | None,
        fallback_candidate: str,
        requirements: ExecutionRequirements,
        runner: Runner | None,
    ) -> None:
        if execution_candidate is None and not requirements.required_capabilities():
            return
        admission_candidate = None if runner is None else runner.execution_admission_candidate()
        if admission_candidate is not None and not isinstance(
            admission_candidate,
            ExecutionAdmissionCandidate,
        ):
            raise TypeError(
                "Runner.execution_admission_candidate must return "
                "ExecutionAdmissionCandidate or None."
            )
        if execution_candidate is not None and admission_candidate is None:
            missing_evidence = evaluate_execution_admission(
                candidate=execution_candidate,
                requirements=requirements,
                evidence=None,
                stage="pre_exposure",
            )
            if missing_evidence.status == "refused":
                missing_evidence.require_admitted()
            raise RuntimeError(
                f"Execution candidate {execution_candidate!r} supplied pre-create evidence, "
                "but the final runner supplied no execution admission evidence."
            )
        candidate = execution_candidate
        if candidate is None:
            candidate = (
                fallback_candidate if admission_candidate is None else admission_candidate.candidate
            )
        evaluate_execution_admission(
            candidate=candidate,
            requirements=requirements,
            evidence=None if admission_candidate is None else admission_candidate.evidence,
            stage="pre_exposure",
        ).require_admitted()

    def _require_registered_environment_admitted(
        self,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment,
    ) -> None:
        self._require_runner_admitted(
            execution_candidate=registered_environment.execution_candidate,
            fallback_candidate=registered_environment.spec.name,
            requirements=registered_agent.execution_requirements,
            runner=registered_environment.environment.runner,
        )

    async def _release_unexposed_factory_environment(
        self,
        registered_environment: runtime_records.RegisteredEnvironment,
        *,
        error: BaseException,
    ) -> tuple[runtime_records.RegisteredEnvironment, dict[str, Any] | None]:
        result = registered_environment.unclaimed_factory_result
        if result is None:
            return registered_environment, None
        # Resolution checkpoints every factory result before returning it to
        # binding. Once committed, release may detach live handles but must not
        # destroy the durable allocation that a later resume will reconnect.
        release_payload = await _release_unclaimed_factory_result(
            result,
            action=EnvironmentFactoryReleaseAction.PRESERVE,
            original_error=error,
        )
        setattr(error, _ENVIRONMENT_FACTORY_RELEASE_ERROR_ATTRIBUTE, release_payload)
        return (
            replace(
                registered_environment,
                unclaimed_factory_result=None,
            ),
            release_payload,
        )

    async def bind(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        started_event: Event | None,
    ) -> EnvironmentBindingResult:
        if registered_environment is None:
            if started_event is not None:
                raise AssertionError("Binding start event exists without an environment.")
            return EnvironmentBindingResult(registered_environment=None, events=[])
        if registered_environment.bound_workspace is not None:
            if started_event is not None:
                raise AssertionError("Binding start event exists for an already-bound workspace.")
            try:
                self._require_registered_environment_admitted(
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
            except Exception as exc:
                return EnvironmentBindingResult(
                    registered_environment=registered_environment,
                    events=[],
                    error=exc,
                )
            adopted_environment = replace(
                registered_environment,
                unclaimed_factory_result=None,
            )
            setup_owner = self._active_environment_setups.get(session.id)
            if setup_owner is not None:
                setup_owner.registered_environment = adopted_environment
            return EnvironmentBindingResult(
                registered_environment=adopted_environment,
                events=[],
            )
        binding = registered_environment.environment.binding
        if binding is None:
            if started_event is not None:
                raise AssertionError("Binding start event exists without a workspace binding.")
            try:
                self._require_registered_environment_admitted(
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
            except Exception as exc:
                (
                    registered_environment,
                    _release_payload,
                ) = await self._release_unexposed_factory_environment(
                    registered_environment,
                    error=exc,
                )
                self._active_environment_setups.pop(session.id, None)
                return EnvironmentBindingResult(
                    registered_environment=registered_environment,
                    events=[],
                    error=exc,
                )
            adopted_environment = replace(
                registered_environment,
                unclaimed_factory_result=None,
            )
            self._active_environment_setups.pop(session.id, None)
            return EnvironmentBindingResult(
                registered_environment=adopted_environment,
                events=[],
            )
        if started_event is None:
            raise AssertionError("Registered workspace binding was not started.")

        environment_name = _environment_name(registered_environment)
        events: list[Event] = []
        base_payload = _binding_base_payload(registered_environment)
        try:
            bound = await binding.bind(
                registered_environment.environment.workspace,
                registered_environment.environment.runner,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
            )
        except BaseException as exc:
            cleanup_status = binding_cleanup_status(exc)
            retry_error: BaseException | None = None
            if cleanup_status is not None:
                cleanup_status.retry_attempted = True
                try:
                    await cleanup_status.retry()
                except asyncio.CancelledError as cleanup_exc:
                    cleanup_status.retry_error = cleanup_exc
                    retry_error = cleanup_exc
                except BaseException as cleanup_exc:
                    cleanup_status.retry_error = cleanup_exc
                    retry_error = cleanup_exc
            ordinary_failure = isinstance(exc, Exception) or (
                isinstance(exc, BaseExceptionGroup) and exc.subgroup(Exception) is not None
            )
            fatal_signal = binding_finalize_fatal_signal(exc)
            if fatal_signal is not None and not ordinary_failure:
                raise
            propagated_error: BaseException = exc
            if retry_error is not None:
                propagated_error = BaseExceptionGroup(
                    "Binding and binding-owned cleanup both failed.",
                    [exc, retry_error],
                )
                if cleanup_status is not None:
                    attach_binding_cleanup_status(propagated_error, cleanup_status)
            (
                registered_environment,
                _release_payload,
            ) = await self._release_unexposed_factory_environment(
                registered_environment,
                error=exc,
            )
            self._active_environment_setups.pop(session.id, None)
            if ordinary_failure:
                failure_payload = {**base_payload, **exception_failure_payload(exc)}
                try:
                    events.append(
                        await self._event_writer.emit(
                            Event(
                                type=EventType.ENVIRONMENT_BINDING_FAILED,
                                session_id=session.id,
                                agent_name=registered_agent.spec.name,
                                environment_name=environment_name,
                                payload=failure_payload,
                            )
                        )
                    )
                except BaseException as publication_error:
                    publication_failure = BaseExceptionGroup(
                        "Binding failure publication also failed.",
                        [propagated_error, publication_error],
                    )
                    if cleanup_status is not None:
                        attach_binding_cleanup_status(publication_failure, cleanup_status)
                    raise publication_failure from (
                        fatal_signal or binding_finalize_cancellation(exc) or publication_error
                    )
            if fatal_signal is not None or not isinstance(exc, Exception):
                if retry_error is not None:
                    raise propagated_error from (fatal_signal or binding_finalize_cancellation(exc))
                raise
            if (
                retry_error is not None
                and binding_finalize_explicit_cancellation(retry_error) is not None
            ):
                raise propagated_error from retry_error
            return EnvironmentBindingResult(
                registered_environment=registered_environment,
                events=events,
                error=exc,
            )

        bound_environment = copy_environment(registered_environment.environment)
        bound_environment.workspace = bound.workspace
        bound_environment.runner = bound.runner
        bound_registered_environment = runtime_records.RegisteredEnvironment(
            spec=registered_environment.spec,
            environment=bound_environment,
            bound_workspace=bound,
            binding_payload=copy_json_value(base_payload, "binding_payload"),
            execution_candidate=registered_environment.execution_candidate,
            preserve_factory_allocation=(
                registered_environment.unclaimed_factory_result is not None
            ),
        )
        # Binding owns the live handles from this point. Record that transfer
        # before publishing it so cancellation or an event-store failure cannot
        # leave cleanup using the stale pre-bound value.
        setup_owner = self._active_environment_setups.get(session.id)
        if setup_owner is None:
            setup_owner = _ActiveEnvironmentSetup(
                registered_environment=bound_registered_environment
            )
            self._active_environment_setups[session.id] = setup_owner
        else:
            setup_owner.registered_environment = bound_registered_environment
        events.append(
            await self._event_writer.emit(
                Event(
                    type=EventType.ENVIRONMENT_BINDING_COMPLETED,
                    session_id=session.id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        **base_payload,
                        **_bound_workspace_payload(bound),
                    },
                )
            )
        )
        try:
            self._require_registered_environment_admitted(
                registered_agent=registered_agent,
                registered_environment=bound_registered_environment,
            )
        except Exception as exc:
            return EnvironmentBindingResult(
                registered_environment=bound_registered_environment,
                events=events,
                error=exc,
            )
        adopted_environment = replace(
            bound_registered_environment,
            preserve_factory_allocation=False,
        )
        setup_owner.registered_environment = adopted_environment
        return EnvironmentBindingResult(
            registered_environment=adopted_environment,
            events=events,
        )

    async def finalize_terminal_event(
        self,
        *,
        event: Event,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> EnvironmentBindingFinalizeResult:
        try:
            return await self._finalize_terminal_event_once(
                event=event,
                session=session,
                registered_environment=registered_environment,
            )
        except BaseException as exc:
            setup_owner = self._active_environment_setups.get(session.id)
            if setup_owner is not None and setup_owner.cleanup_started:
                setup_owner.cleanup_error = exc
            raise

    async def _finalize_terminal_event_once(
        self,
        *,
        event: Event,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> EnvironmentBindingFinalizeResult:
        setup_owner = self._active_environment_setups.get(session.id)
        if setup_owner is not None:
            if setup_owner.cleanup_started:
                if setup_owner.cleanup_error is not None:
                    raise setup_owner.cleanup_error
                return EnvironmentBindingFinalizeResult(event=event, events=[])
            setup_owner.cleanup_started = True
            registered_environment = setup_owner.registered_environment
        if (
            registered_environment is not None
            and registered_environment.unclaimed_factory_result is not None
        ):
            setup_error = RuntimeError(
                "Environment setup ended before the factory result was adopted."
            )
            (
                registered_environment,
                release_payload,
            ) = await self._release_unexposed_factory_environment(
                registered_environment,
                error=setup_error,
            )
            if release_payload is not None:
                terminal_payload = copy_json_value(event.payload, "payload")
                terminal_payload["environment_factory_release"] = release_payload
                event = _copy_event_with_payload(event, terminal_payload)
        if registered_environment is None or registered_environment.bound_workspace is None:
            return EnvironmentBindingFinalizeResult(event=event, events=[])
        binding = registered_environment.environment.binding
        if binding is None:
            return EnvironmentBindingFinalizeResult(event=event, events=[])

        terminal_outcome = _binding_outcome_for_terminal_event(event.type)
        preserve_factory_allocation = registered_environment.preserve_factory_allocation
        outcome = "interrupted" if preserve_factory_allocation else terminal_outcome
        environment_name = _environment_name(registered_environment)
        base_payload = {
            **_binding_base_payload(registered_environment),
            **_bound_workspace_payload(registered_environment.bound_workspace),
            "outcome": outcome,
        }
        if preserve_factory_allocation:
            base_payload["terminal_outcome"] = terminal_outcome
            base_payload["factory_allocation_action"] = "preserve"
        events: list[Event] = []
        start_publication_error: BaseException | None = None
        try:
            events.append(
                await self._event_writer.emit(
                    Event(
                        type=EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
                        session_id=session.id,
                        agent_name=event.agent_name,
                        environment_name=environment_name,
                        payload=base_payload,
                    )
                )
            )
        except BaseException as exc:
            start_publication_error = exc

        try:
            final_snapshot = await binding.finalize(
                registered_environment.bound_workspace,
                outcome=outcome,
                metadata={
                    "event_type": str(event.type),
                    "session_id": session.id,
                },
            )
            final_snapshot = copy_workspace_snapshot(final_snapshot)
        except (BaseExceptionGroup, Exception, asyncio.CancelledError) as exc:
            if start_publication_error is not None:
                exc = BaseExceptionGroup(
                    "Binding finalization and start-event publication failed.",
                    [start_publication_error, exc],
                )
            finalize_error_payload = _binding_finalize_error_payload(
                exc,
                outcome=outcome,
                redactor=self._secret_redactor,
            )
            error_payload = {
                **base_payload,
                **finalize_error_payload,
            }
            try:
                failure_event, persist_cancellation = await _persist_binding_finalize_failure_event(
                    self._event_writer,
                    Event(
                        type=EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
                        session_id=session.id,
                        agent_name=event.agent_name,
                        environment_name=environment_name,
                        payload=error_payload,
                    ),
                )
            except BaseException as diagnostic_error:
                setattr(exc, BINDING_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE, finalize_error_payload)
                exc.add_note(
                    "Binding finalization durable failure publication also failed: "
                    f"{type(diagnostic_error).__name__}: {diagnostic_error}."
                )
                fatal_signal = binding_finalize_fatal_signal(diagnostic_error)
                if fatal_signal is not None:
                    raise fatal_signal from diagnostic_error
                cancellation = (
                    diagnostic_error
                    if isinstance(diagnostic_error, asyncio.CancelledError)
                    else binding_finalize_explicit_cancellation(diagnostic_error)
                )
                if cancellation is not None:
                    aggregate = append_binding_finalize_cancellation(exc, cancellation)
                    aggregate.add_note(
                        "Binding finalization durable failure publication also failed."
                    )
                    raise aggregate from diagnostic_error
                raise exc from diagnostic_error
            if persist_cancellation is not None:
                raise append_binding_finalize_cancellation(
                    exc, persist_cancellation
                ) from persist_cancellation
            try:
                fanout_task = asyncio.create_task(
                    self._event_writer.fan_out_persisted([failure_event])
                )
                failure_event = (await asyncio.shield(fanout_task))[0]
            except asyncio.CancelledError as cancellation:
                setattr(exc, BINDING_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE, finalize_error_payload)
                raise append_binding_finalize_cancellation(exc, cancellation) from cancellation
            except BaseException as diagnostic_error:
                setattr(exc, BINDING_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE, finalize_error_payload)
                exc.add_note(
                    "Binding finalization diagnostic fan-out failed: "
                    f"{type(diagnostic_error).__name__}: {diagnostic_error}."
                )
                fatal_signal = binding_finalize_fatal_signal(diagnostic_error)
                if fatal_signal is not None:
                    raise fatal_signal from diagnostic_error
                cancellation = binding_finalize_explicit_cancellation(diagnostic_error)
                if cancellation is not None:
                    aggregate = append_binding_finalize_cancellation(exc, cancellation)
                    aggregate.add_note(
                        "Binding finalization durable failure publication also failed."
                    )
                    raise aggregate from diagnostic_error
                raise exc from diagnostic_error
            events.append(failure_event)
            if not isinstance(exc, Exception):
                raise
            terminal_payload = copy_json_value(event.payload, "payload")
            terminal_payload["binding_finalize_error"] = finalize_error_payload
            return EnvironmentBindingFinalizeResult(
                event=Event(
                    type=event.type,
                    session_id=event.session_id,
                    id=event.id,
                    timestamp=event.timestamp,
                    agent_name=event.agent_name,
                    environment_name=event.environment_name,
                    workflow_name=event.workflow_name,
                    tool_name=event.tool_name,
                    payload=terminal_payload,
                ),
                events=events,
            )

        completion_publication_error: BaseException | None = None
        try:
            events.append(
                await self._event_writer.emit(
                    Event(
                        type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                        session_id=session.id,
                        agent_name=event.agent_name,
                        environment_name=environment_name,
                        payload={
                            **base_payload,
                            "final_snapshot": _workspace_snapshot_payload(final_snapshot),
                        },
                    )
                )
            )
        except BaseException as exc:
            completion_publication_error = exc
        publication_failures = tuple(
            (phase, error)
            for phase, error in (
                ("finalize_started_event", start_publication_error),
                ("finalize_completed_event", completion_publication_error),
            )
            if error is not None
        )
        if publication_failures:
            terminal_payload = copy_json_value(event.payload, "payload")
            terminal_payload["binding_finalize_publication_error"] = (
                _binding_finalize_publication_failure_payload(
                    publication_failures,
                    outcome=outcome,
                    redactor=self._secret_redactor,
                )
            )
            event = _copy_event_with_payload(event, terminal_payload)
        return EnvironmentBindingFinalizeResult(event=event, events=events)

    async def abort_environment_setup(
        self,
        *,
        session_id: str,
        original_error: BaseException | None,
    ) -> None:
        """Release a live setup when no terminal event can own its cleanup."""

        setup_owner = self._active_environment_setups.pop(session_id, None)
        if setup_owner is None or setup_owner.cleanup_started:
            return
        setup_owner.cleanup_started = True
        registered_environment = setup_owner.registered_environment
        if original_error is None:
            original_error = RuntimeError("Environment setup ended without terminal cleanup.")
        if registered_environment.unclaimed_factory_result is not None:
            try:
                await self._release_unexposed_factory_environment(
                    registered_environment,
                    error=original_error,
                )
            except BaseException as cleanup_error:
                if cleanup_error is original_error:
                    raise
                raise BaseExceptionGroup(
                    "Environment factory cleanup failed while aborting setup.",
                    [original_error, cleanup_error],
                ) from cleanup_error
            return
        binding = registered_environment.environment.binding
        if binding is None or registered_environment.bound_workspace is None:
            return
        try:
            await binding.finalize(
                registered_environment.bound_workspace,
                outcome="interrupted",
                metadata={
                    "event_type": "environment_setup_aborted",
                    "session_id": session_id,
                },
            )
        except BaseException as cleanup_error:
            original_error.add_note(
                "Environment binding cleanup failed while aborting setup: "
                f"{type(cleanup_error).__name__}: {cleanup_error}."
            )
            if cleanup_error is original_error:
                raise
            raise BaseExceptionGroup(
                "Environment binding cleanup failed while aborting setup.",
                [original_error, cleanup_error],
            ) from cleanup_error

    async def _load_factory_reconnect_state(
        self,
        *,
        session_id: str,
        environment_name: str,
    ) -> tuple[dict[str, Any], str | None]:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        if checkpoint is None:
            return {}, None
        state = checkpoint.get(ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY)
        if state is None:
            metadata: dict[str, Any] = {}
        else:
            if type(state) is not dict:
                raise ValueError("Environment factory reconnect checkpoint must be an object.")
            candidate_metadata = state.get(environment_name)
            if candidate_metadata is None:
                metadata = {}
            elif type(candidate_metadata) is not dict:
                raise ValueError("Environment factory reconnect metadata must be an object.")
            else:
                metadata = copy_json_value(candidate_metadata, "reconnect_metadata")
        owners = checkpoint.get(ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY)
        if owners is None:
            return metadata, None
        if type(owners) is not dict:
            raise ValueError("Environment factory allocation owners must be an object.")
        owner = owners.get(environment_name)
        if owner is None:
            return metadata, None
        if not isinstance(owner, str) or not owner:
            raise ValueError("Environment factory allocation owner must be a nonblank string.")
        return metadata, owner

    async def _checkpoint_factory_reconnect_metadata(
        self,
        *,
        session_id: str,
        environment_name: str,
        reconnect_metadata: dict[str, Any],
    ) -> None:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        copied_checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        state = copied_checkpoint.get(ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY)
        if state is None:
            state = {}
        elif type(state) is not dict:
            raise ValueError("Environment factory reconnect checkpoint must be an object.")
        else:
            state = copy_json_value(state, "environment_factory_reconnect")
        state[environment_name] = copy_json_value(reconnect_metadata, "reconnect_metadata")
        copied_checkpoint[ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY] = state
        owners = copied_checkpoint.get(ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY)
        if owners is None:
            owners = {}
        elif type(owners) is not dict:
            raise ValueError("Environment factory allocation owners must be an object.")
        else:
            owners = copy_json_value(owners, "environment_factory_allocation_owner")
        owners[environment_name] = session_id
        copied_checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY] = owners
        try:
            await self._session_store.transform_checkpoint(
                session_id,
                self._checkpoint_transform(copied_checkpoint),
            )
        except BaseException as exc:
            current_task = asyncio.current_task()
            caller_cancellation = (
                exc
                if isinstance(exc, asyncio.CancelledError)
                and current_task is not None
                and current_task.cancelling() > 0
                else None
            )
            outcome = await await_shielded_task_outcome(
                asyncio.create_task(
                    self._factory_checkpoint_matches(
                        session_id=session_id,
                        environment_name=environment_name,
                        reconnect_metadata=reconnect_metadata,
                    )
                ),
                cancellation=caller_cancellation,
            )
            checkpoint_may_be_committed = outcome.error is not None or bool(outcome.result)
            if outcome.error is not None:
                fatal_signal = binding_finalize_fatal_signal(outcome.error)
                if fatal_signal is not None:
                    setattr(
                        fatal_signal,
                        _ENVIRONMENT_FACTORY_CHECKPOINT_MAY_BE_COMMITTED_ATTRIBUTE,
                        True,
                    )
                    fatal_signal.add_note(
                        "The environment factory checkpoint write also failed; "
                        "its commit outcome could not be reconciled."
                    )
                    raise fatal_signal from exc
                exc.add_note(
                    "Could not reconcile whether the environment factory checkpoint "
                    "committed; the allocation will be preserved."
                )
            propagated_error: BaseException = exc
            if outcome.cancellation is not None and outcome.cancellation is not exc:
                propagated_error = BaseExceptionGroup(
                    "Environment factory checkpoint write failed during caller cancellation.",
                    [exc, outcome.cancellation],
                )
            if checkpoint_may_be_committed:
                setattr(
                    propagated_error,
                    _ENVIRONMENT_FACTORY_CHECKPOINT_MAY_BE_COMMITTED_ATTRIBUTE,
                    True,
                )
            if outcome.error is not None:
                raise propagated_error from outcome.error
            if propagated_error is exc:
                raise
            raise propagated_error from exc

    async def _factory_checkpoint_matches(
        self,
        *,
        session_id: str,
        environment_name: str,
        reconnect_metadata: dict[str, Any],
    ) -> bool:
        persisted_metadata, allocation_owner = await self._load_factory_reconnect_state(
            session_id=session_id,
            environment_name=environment_name,
        )
        return allocation_owner == session_id and persisted_metadata == reconnect_metadata


def render_initial_system_prompt(
    *,
    agent_system_prompt: str | None,
    workspace_instructions: WorkspaceInstructions | None,
) -> str | None:
    agent_prompt = agent_system_prompt.strip() if agent_system_prompt else ""
    if workspace_instructions is None:
        return agent_prompt or None

    workspace_content = workspace_instructions.content.strip()
    source_list = ", ".join(workspace_instructions.sources)
    workspace_section = (
        "[Workspace instructions]\n"
        f"Source: {source_list}\n"
        "These instructions apply only to the active workspace. If they conflict "
        "with agent, tool, approval, sandbox, or secret policy, follow the "
        "higher-priority runtime policy.\n\n"
        f"{workspace_content}"
    )
    if not agent_prompt:
        return workspace_section
    return f"[Agent instructions]\n{agent_prompt}\n\n{workspace_section}"


def exception_failure_payload(error: BaseException) -> dict[str, Any]:
    safe_payload = getattr(error, BINDING_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE, None)
    if not isinstance(safe_payload, dict) and isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            child_payload = exception_failure_payload(child)
            if "failures" in child_payload and "outcome" in child_payload:
                safe_payload = child_payload
                break
    if isinstance(safe_payload, dict):
        return copy_json_value(safe_payload, "binding_finalize_safe_payload")
    payload: dict[str, Any] = {
        "error": str(error),
        "error_type": type(error).__name__,
    }
    if isinstance(error, ExecutionAdmissionError):
        payload["execution_admission"] = error.decision.model_dump(mode="json")
    cleanup_payload = binding_cleanup_payload(error)
    if cleanup_payload is not None:
        payload["binding_cleanup"] = cleanup_payload
    factory_release = getattr(error, _ENVIRONMENT_FACTORY_RELEASE_ERROR_ATTRIBUTE, None)
    if isinstance(factory_release, dict):
        payload["environment_factory_release"] = copy_json_value(
            factory_release,
            "environment_factory_release",
        )
    return payload


def _copy_event_with_payload(event: Event, payload: dict[str, Any]) -> Event:
    return Event(
        type=event.type,
        session_id=event.session_id,
        id=event.id,
        timestamp=event.timestamp,
        agent_name=event.agent_name,
        environment_name=event.environment_name,
        workflow_name=event.workflow_name,
        tool_name=event.tool_name,
        payload=payload,
    )


async def _reconcile_binding_finalize_failure_event(
    writer: RuntimeEventWriter,
    event: Event,
    *,
    persistence_error: BaseException,
    cancellation: asyncio.CancelledError | None,
) -> tuple[bool, asyncio.CancelledError | None]:
    outcome = await await_shielded_task_outcome(
        asyncio.create_task(writer.is_persisted(event)),
        cancellation=cancellation,
    )
    cancellation = outcome.cancellation
    if outcome.error is None:
        return bool(outcome.result), cancellation
    fatal_signal = binding_finalize_fatal_signal(outcome.error)
    if fatal_signal is not None:
        raise fatal_signal
    if cancellation is not None:
        persistence_error.add_note(
            "Could not reconcile whether the binding finalization failure event committed."
        )
        raise BaseExceptionGroup(
            "Binding finalization failure reconciliation failed after caller cancellation.",
            [persistence_error, cancellation],
        ) from outcome.error
    raise persistence_error from outcome.error


async def _persist_binding_finalize_failure_event(
    writer: RuntimeEventWriter,
    event: Event,
) -> tuple[Event, asyncio.CancelledError | None]:
    outcome = await await_shielded_task_outcome(asyncio.create_task(writer.persist(event)))
    persistence_error = outcome.error
    cancellation = outcome.cancellation
    if persistence_error is None:
        return event, cancellation
    fatal_signal = binding_finalize_fatal_signal(persistence_error)
    if fatal_signal is not None:
        raise fatal_signal
    persisted, cancellation = await _reconcile_binding_finalize_failure_event(
        writer,
        event,
        persistence_error=persistence_error,
        cancellation=cancellation,
    )
    if persisted:
        return event, cancellation
    if cancellation is not None:
        raise BaseExceptionGroup(
            "Binding finalization failure publication failed after caller cancellation.",
            [persistence_error, cancellation],
        ) from persistence_error
    raise persistence_error


def _binding_finalize_error_payload(
    error: BaseException,
    *,
    outcome: str,
    redactor: Any,
) -> dict[str, Any]:
    details = binding_finalize_error_details(error, redactor=redactor)
    failures = binding_finalize_failure_payload(error, redactor=redactor)
    if failures is None:
        failures = [{"phase": "workspace_finalize", **details}]
    return {**details, "outcome": outcome, "failures": failures}


def _binding_finalize_publication_failure_payload(
    failures: tuple[tuple[str, BaseException], ...],
    *,
    outcome: str,
    redactor: Any,
) -> dict[str, Any]:
    errors = [error for _phase, error in failures]
    combined_error: BaseException = errors[0]
    if len(errors) > 1:
        combined_error = BaseExceptionGroup(
            "Binding finalization lifecycle publication failed.",
            errors,
        )
    fatal_signal = binding_finalize_fatal_signal(combined_error)
    if fatal_signal is not None:
        raise fatal_signal
    cancellation = binding_finalize_explicit_cancellation(combined_error)
    if cancellation is not None:
        if combined_error is cancellation:
            raise cancellation
        raise combined_error from cancellation
    return {
        "outcome": outcome,
        "failures": [
            {
                "phase": phase,
                **binding_finalize_error_details(error, redactor=redactor),
            }
            for phase, error in failures
        ],
    }


def _environment_factory_base_payload(
    *,
    session: Session,
    registered_environment: runtime_records.RegisteredEnvironment,
) -> dict[str, Any]:
    factory = registered_environment.factory
    if factory is None:
        raise AssertionError("Environment factory payload requires a registered factory.")
    environment_name = registered_environment.spec.name
    return {
        "factory_type": type(factory).__name__,
        "requested_environment_name": environment_name,
        "parent_session_id": session.parent_session_id,
        "causal_budget_id": session.causal_budget_id,
        "labels": copy_label_map(session.labels, "labels"),
    }


def _binding_base_payload(
    registered_environment: runtime_records.RegisteredEnvironment,
) -> dict[str, Any]:
    if registered_environment.binding_payload is not None:
        return copy_json_value(registered_environment.binding_payload, "binding_payload")
    binding = registered_environment.environment.binding
    return {
        "binding_type": type(binding).__name__ if binding is not None else None,
        "configured_workspace_id": _workspace_object_id(
            registered_environment.environment.workspace
        ),
        "has_configured_runner": registered_environment.environment.runner is not None,
    }


def _bound_workspace_payload(bound: BoundWorkspace) -> dict[str, Any]:
    return {
        "source_workspace_id": _workspace_object_id(bound.source_workspace),
        "bound_workspace_id": _workspace_object_id(bound.workspace),
        "bound_path": bound.path,
        "bound_metadata": copy_json_value(bound.metadata, "bound_metadata"),
        "bound_snapshot": _workspace_snapshot_payload(bound.snapshot),
        "has_bound_runner": bound.runner is not None,
    }


def _workspace_snapshot_payload(snapshot: WorkspaceSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "snapshot_id": snapshot.snapshot_id,
        "workspace_id": snapshot.workspace_id,
        "version": snapshot.version,
        "source": snapshot.source,
        "metadata": copy_json_value(snapshot.metadata, "metadata"),
    }


def _workspace_object_id(workspace: Any) -> str | None:
    if workspace is None:
        return None
    workspace_id = getattr(workspace, "id", None)
    return workspace_id if isinstance(workspace_id, str) else None


def _binding_outcome_for_terminal_event(event_type: EventType | str) -> str:
    if event_type == EventType.SESSION_COMPLETED:
        return "completed"
    if event_type == EventType.SESSION_FAILED:
        return "failed"
    if event_type == EventType.SESSION_INTERRUPTED:
        return "interrupted"
    return str(event_type)


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None:
        return None
    return registered_environment.spec.name


async def _release_unclaimed_factory_result(
    result: EnvironmentFactoryResult,
    *,
    action: EnvironmentFactoryReleaseAction,
    original_error: BaseException,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "action": action.value,
        "callback_provided": result.release is not None,
    }
    if result.release is not None:
        release = result.release

        async def run_release() -> None:
            await release(action)

        release_task = asyncio.create_task(run_release())
        try:
            cancelled = await _await_bounded_environment_factory_release(
                release_task,
                timeout_s=result.release_timeout_s,
            )
        except BaseException as cleanup_error:
            payload.update(
                {
                    "completed": False,
                    "error": str(cleanup_error),
                    "error_type": type(cleanup_error).__name__,
                    "timeout_s": result.release_timeout_s,
                }
            )
            original_error.add_note(
                "Environment factory result release failed after "
                f"{action.value}: {type(cleanup_error).__name__}: {cleanup_error}."
            )
            fatal_signal = binding_finalize_fatal_signal(cleanup_error)
            if fatal_signal is not None:
                if fatal_signal is cleanup_error:
                    raise
                raise fatal_signal from cleanup_error
            if binding_finalize_explicit_cancellation(cleanup_error) is not None:
                raise cleanup_error
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                cancellation = asyncio.CancelledError()
                cancellation.add_note(
                    "Environment factory result release failed while cancellation was pending: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}."
                )
                raise BaseExceptionGroup(
                    "Environment factory result release failed after caller cancellation.",
                    [cancellation, cleanup_error],
                ) from cleanup_error
        else:
            payload["completed"] = True
            if cancelled:
                raise asyncio.CancelledError()
        return payload
    if action is EnvironmentFactoryReleaseAction.PRESERVE:
        payload.update(
            {
                "completed": False,
                "error": "Durable factory result has no release callback.",
                "error_type": "MissingEnvironmentFactoryRelease",
            }
        )
        original_error.add_note(
            "Environment factory result has durable reconnect state but no release callback; "
            "the runtime left the live allocation untouched rather than closing it terminally."
        )
        return payload

    cleanup_errors: list[tuple[str, Exception]] = []

    async def run_fallback_release() -> None:
        runner = result.environment.runner
        if runner is not None:
            try:
                await runner.close()
            except Exception as cleanup_error:
                cleanup_errors.append(("runner", cleanup_error))

        binding = result.environment.binding
        close = getattr(binding, "close", None)
        if callable(close):
            try:
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result
            except Exception as cleanup_error:
                cleanup_errors.append(("binding", cleanup_error))

    fallback_task = asyncio.create_task(run_fallback_release())
    try:
        cancelled = await _await_bounded_environment_factory_release(
            fallback_task,
            timeout_s=result.release_timeout_s,
        )
    except BaseException as cleanup_error:
        payload.update(
            {
                "completed": False,
                "error": str(cleanup_error),
                "error_type": type(cleanup_error).__name__,
                "timeout_s": result.release_timeout_s,
            }
        )
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            cancellation = asyncio.CancelledError()
            cancellation.add_note(
                "Environment factory fallback release failed while cancellation was pending: "
                f"{type(cleanup_error).__name__}: {cleanup_error}."
            )
            raise cancellation from cleanup_error
        return payload
    if cancelled:
        raise asyncio.CancelledError()
    payload["completed"] = not cleanup_errors
    if cleanup_errors:
        details = "; ".join(
            f"{phase}: {type(error).__name__}: {error}" for phase, error in cleanup_errors
        )
        payload["error"] = details
        payload["error_type"] = type(cleanup_errors[0][1]).__name__
        original_error.add_note(
            f"Environment factory fallback release incomplete after {action.value}: {details}."
        )
    return payload


async def _await_bounded_environment_factory_release(
    task: asyncio.Task[None],
    *,
    timeout_s: float,
) -> bool:
    """Finish a factory release despite cancellation, within its declared bound."""

    cancelled = False
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not task.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            task.cancel()
            task.add_done_callback(_consume_background_task_result)
            raise TimeoutError(
                f"Environment factory result release did not complete within {timeout_s:g} seconds."
            )
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.CancelledError:
            if task.done():
                task.result()
            cancelled = True
        except TimeoutError as exc:
            if task.done():
                task.result()
                break
            task.cancel()
            task.add_done_callback(_consume_background_task_result)
            raise TimeoutError(
                f"Environment factory result release did not complete within {timeout_s:g} seconds."
            ) from exc
    task.result()
    return cancelled


def _consume_background_task_result(task: asyncio.Task[Any]) -> None:
    with contextlib.suppress(BaseException):
        task.result()
