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
from dataclasses import dataclass
from typing import Any

from cayu._validation import copy_json_value, copy_label_map
from cayu.core.events import Event, EventType
from cayu.environments import (
    BoundWorkspace,
    Environment,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    WorkspaceInstructions,
    WorkspaceSnapshot,
    copy_environment,
    copy_workspace_snapshot,
    load_workspace_instructions,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._binding_cleanup import binding_cleanup_payload, binding_cleanup_status
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime.sessions import CheckpointTransform, Session, SessionStore

ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY = "environment_factory_reconnect"
ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY = "environment_factory_allocation_owner"
_ENVIRONMENT_FACTORY_CHECKPOINT_COMMIT_UNCERTAIN_ATTRIBUTE = (
    "_cayu_environment_factory_checkpoint_commit_uncertain"
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


class EnvironmentLifecycle:
    """Own environment factory, workspace binding, and reconnect state."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        checkpoint_transform: CheckpointTransformFactory,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._checkpoint_transform = checkpoint_transform

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
        allocation_checkpoint_commit_uncertain = False
        effective_operation = operation
        try:
            reconnect_metadata, allocation_owner = await self._load_factory_reconnect_state(
                session_id=session.id,
                environment_name=environment_name,
            )
            if (
                operation is EnvironmentFactoryOperation.RECONNECT
                and session.parent_session_id is not None
                and allocation_owner != session.id
            ):
                completed_for_child = False
                if allocation_owner is None:
                    # Compatibility for checkpoints written before allocation
                    # provenance was persisted atomically with reconnect state.
                    prior_events = await self._session_store.load_events(session.id)
                    completed_for_child = any(
                        event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED
                        and event.environment_name == environment_name
                        for event in prior_events
                    )
                if not completed_for_child:
                    # A fork inherits its parent's checkpoint as context, but
                    # its first factory allocation belongs to the child. Only
                    # later child resumes reconnect that child allocation.
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
            )
            result = await factory.create(request)
            if type(result) is not EnvironmentFactoryResult:
                raise TypeError("EnvironmentFactory.create must return EnvironmentFactoryResult.")
            environment = copy_environment(result.environment)
            if environment.spec.name != environment_name:
                raise ValueError(
                    "Environment factory returned a different environment name: "
                    f"{environment.spec.name!r} != {environment_name!r}"
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
            except asyncio.CancelledError as exc:
                # The checkpoint helper marks only cancellation during the
                # transactional write as uncertain; a preliminary read cancellation
                # still proves that no reconnect identity was committed.
                allocation_checkpoint_commit_uncertain = bool(
                    getattr(
                        exc,
                        _ENVIRONMENT_FACTORY_CHECKPOINT_COMMIT_UNCERTAIN_ATTRIBUTE,
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
        except BaseException as exc:
            if result is not None:
                release_payload = await _release_unclaimed_factory_result(
                    result,
                    action=(
                        EnvironmentFactoryReleaseAction.PRESERVE
                        if allocation_checkpointed
                        or allocation_checkpoint_commit_uncertain
                        or effective_operation is EnvironmentFactoryOperation.RECONNECT
                        else EnvironmentFactoryReleaseAction.DISCARD
                    ),
                    original_error=exc,
                )
                setattr(exc, _ENVIRONMENT_FACTORY_RELEASE_ERROR_ATTRIBUTE, release_payload)
            if not isinstance(exc, Exception):
                raise
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
            return EnvironmentFactoryResolutionResult(
                registered_environment=registered_environment,
                events=events,
                error=exc,
            )

        if environment is None:
            raise RuntimeError("Environment factory did not produce an environment.")
        return EnvironmentFactoryResolutionResult(
            registered_environment=runtime_records.RegisteredEnvironment(
                spec=registered_environment.spec,
                environment=environment,
            ),
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
            return EnvironmentBindingResult(
                registered_environment=registered_environment,
                events=[],
            )
        binding = registered_environment.environment.binding
        if binding is None:
            if started_event is not None:
                raise AssertionError("Binding start event exists without a workspace binding.")
            return EnvironmentBindingResult(
                registered_environment=registered_environment,
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
            if cleanup_status is not None:
                cleanup_status.retry_attempted = True
                try:
                    await cleanup_status.retry()
                except asyncio.CancelledError:
                    raise
                except Exception as cleanup_exc:
                    cleanup_status.retry_error = cleanup_exc
            if not isinstance(exc, Exception):
                raise
            failure_payload = {**base_payload, **exception_failure_payload(exc)}
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
            return EnvironmentBindingResult(
                registered_environment=registered_environment,
                events=events,
                error=exc,
            )

        bound_environment = copy_environment(registered_environment.environment)
        bound_environment.workspace = bound.workspace
        bound_environment.runner = bound.runner
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
        return EnvironmentBindingResult(
            registered_environment=runtime_records.RegisteredEnvironment(
                spec=registered_environment.spec,
                environment=bound_environment,
                bound_workspace=bound,
                binding_payload=copy_json_value(base_payload, "binding_payload"),
            ),
            events=events,
        )

    async def finalize_terminal_event(
        self,
        *,
        event: Event,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> EnvironmentBindingFinalizeResult:
        if registered_environment is None or registered_environment.bound_workspace is None:
            return EnvironmentBindingFinalizeResult(event=event, events=[])
        binding = registered_environment.environment.binding
        if binding is None:
            return EnvironmentBindingFinalizeResult(event=event, events=[])

        outcome = _binding_outcome_for_terminal_event(event.type)
        environment_name = _environment_name(registered_environment)
        base_payload = {
            **_binding_base_payload(registered_environment),
            **_bound_workspace_payload(registered_environment.bound_workspace),
            "outcome": outcome,
        }
        events: list[Event] = [
            await self._event_writer.emit(
                Event(
                    type=EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
                    session_id=session.id,
                    agent_name=event.agent_name,
                    environment_name=environment_name,
                    payload=base_payload,
                )
            )
        ]
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
        except Exception as exc:
            error_payload = {
                **base_payload,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            events.append(
                await self._event_writer.emit(
                    Event(
                        type=EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
                        session_id=session.id,
                        agent_name=event.agent_name,
                        environment_name=environment_name,
                        payload=error_payload,
                    )
                )
            )
            terminal_payload = copy_json_value(event.payload, "payload")
            terminal_payload["binding_finalize_error"] = {
                "error": str(exc),
                "error_type": type(exc).__name__,
                "outcome": outcome,
            }
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
        return EnvironmentBindingFinalizeResult(event=event, events=events)

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
        except asyncio.CancelledError as exc:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling() > 0:
                setattr(
                    exc,
                    _ENVIRONMENT_FACTORY_CHECKPOINT_COMMIT_UNCERTAIN_ATTRIBUTE,
                    True,
                )
            raise


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
    payload: dict[str, Any] = {
        "error": str(error),
        "error_type": type(error).__name__,
    }
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
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                cancellation = asyncio.CancelledError()
                cancellation.add_note(
                    "Environment factory result release failed while cancellation was pending: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}."
                )
                raise cancellation from cleanup_error
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
