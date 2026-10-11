"""Durable interruption cleanup and credential-safe cancellation propagation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from dataclasses import dataclass

from cayu._exception_groups import (
    _failure_without_existing_exception_identities,
    exception_group_children,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._exception_state import exception_state, set_exception_state
from cayu._task_wait import (
    capture_awaitable_outcome,
    consume_pending_task_cancellation,
    unexpected_child_cancellation_error,
)
from cayu.events import Event, EventType
from cayu.execution_profiles import ExecutionProfileIdentity
from cayu.providers._credential_boundary import detach_credential_safe_provider_cancellation
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._foreground_subagent_recovery import ForegroundSubagentRecoveryRequired
from cayu.runtime._interruption_coordinator import _INTERRUPTIBLE_SESSION_STATUSES
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._model_errors import detach_billing_identity_cancellation
from cayu.runtime._pending_tool_round_recovery import PendingToolRoundRecovery
from cayu.runtime._recovery_ownership import RecoveryOwnership
from cayu.runtime._run_limits import SessionUsageTracker
from cayu.runtime._session_control import ActiveSessionRun
from cayu.runtime._session_finalization import SessionFinalization
from cayu.runtime._tool_round_executor import InterruptedToolRoundRequest
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._invocation_scope import _mark_session_invocation_terminal_event
from cayu.sessions.base import SessionStore
from cayu.sessions.records import Session, SessionStatus
from cayu.workspaces.observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
    workspace_observation_pending_cancellation_requests,
)

_PROVIDER_CANCELLATION_FINALIZATION_FAILURE_STATE = (
    "_cayu_provider_cancellation_finalization_failure"
)


_PROVIDER_CANCELLATION_FINALIZATION_FAILURE_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class _ProviderCancellationFinalizationFailure:
    """Authenticated proof that runtime terminal publication did not finish."""

    authority: object


def _mark_provider_cancellation_finalization_failure(
    cancellation: asyncio.CancelledError,
) -> None:
    if not set_exception_state(
        cancellation,
        _PROVIDER_CANCELLATION_FINALIZATION_FAILURE_STATE,
        _ProviderCancellationFinalizationFailure(
            authority=_PROVIDER_CANCELLATION_FINALIZATION_FAILURE_AUTHORITY,
        ),
    ):
        raise RuntimeError("Could not retain provider cancellation finalization failure.")


def _has_provider_cancellation_finalization_failure(
    cancellation: asyncio.CancelledError,
) -> bool:
    handoff = exception_state(
        cancellation,
        _PROVIDER_CANCELLATION_FINALIZATION_FAILURE_STATE,
    )
    return (
        type(handoff) is _ProviderCancellationFinalizationFailure
        and handoff.authority is _PROVIDER_CANCELLATION_FINALIZATION_FAILURE_AUTHORITY
    )


def _copy_authenticated_cancellation_diagnostics_for_public(
    source: asyncio.CancelledError,
    detached: asyncio.CancelledError,
    *,
    finalization_failed: bool,
) -> None:
    """Retain runtime-owned diagnostics after credential-safe detachment."""

    notes = getattr(source, "__notes__", ())
    if type(notes) is list:
        for note in notes:
            if type(note) is str:
                detached.add_note(note)
    if finalization_failed and not set_exception_cause(
        detached,
        RuntimeError("Session interruption finalization failed after provider cancellation."),
    ):
        raise RuntimeError("Could not retain detached provider cancellation diagnostics.")


def _detach_provider_cancellation_for_public(
    cancellation: asyncio.CancelledError,
) -> asyncio.CancelledError | None:
    detached = detach_credential_safe_provider_cancellation(cancellation)
    finalization_failed = _has_provider_cancellation_finalization_failure(cancellation)
    if detached is None:
        if not finalization_failed:
            return None
        detached = asyncio.CancelledError("Provider operation cancelled")
    pending_cancellation_requests = workspace_observation_pending_cancellation_requests(
        cancellation
    )
    if pending_cancellation_requests:
        retain_workspace_observation_pending_cancellation_requests(
            detached,
            pending_cancellation_requests,
        )
    if finalization_failed:
        _mark_provider_cancellation_finalization_failure(detached)
    _copy_authenticated_cancellation_diagnostics_for_public(
        cancellation,
        detached,
        finalization_failed=finalization_failed,
    )
    return detached


def _detach_billing_cancellation_for_public(
    cancellation: asyncio.CancelledError,
) -> asyncio.CancelledError | None:
    detached = detach_billing_identity_cancellation(cancellation)
    if detached is None:
        return None
    pending_cancellation_requests = workspace_observation_pending_cancellation_requests(
        cancellation
    )
    if pending_cancellation_requests:
        retain_workspace_observation_pending_cancellation_requests(
            detached,
            pending_cancellation_requests,
        )
    finalization_failed = _has_provider_cancellation_finalization_failure(cancellation)
    if finalization_failed:
        _mark_provider_cancellation_finalization_failure(detached)
    _copy_authenticated_cancellation_diagnostics_for_public(
        cancellation,
        detached,
        finalization_failed=finalization_failed,
    )
    return detached


def _session_interruption_failure_with_additional_control(
    authoritative_failure: BaseException,
    additional_failure: BaseException,
    *,
    group_message: str,
    historical_cancellation_requests: int,
) -> BaseException:
    """Retain one later failure without losing interruption cancellation state."""

    if type(historical_cancellation_requests) is not int or historical_cancellation_requests < 0:
        raise ValueError("Historical cancellation requests must be a non-negative int.")
    current_task = asyncio.current_task()
    requests_before = 0 if current_task is None else current_task.cancelling()
    delivered_cancellation: asyncio.CancelledError | None = None
    if isinstance(additional_failure, asyncio.CancelledError):
        delivered_cancellation = consume_pending_task_cancellation(
            additional_failure,
            preserve_requests=historical_cancellation_requests,
        )
    requests_after = 0 if current_task is None else current_task.cancelling()
    requests_consumed = max(requests_before - requests_after, 0)

    failures: list[BaseException] = [additional_failure]
    if delivered_cancellation is not None and all(
        delivered_cancellation is not failure for failure in failures
    ):
        failures.append(delivered_cancellation)

    propagated_failure = authoritative_failure
    for failure in failures:
        existing_ids = {id(candidate) for candidate in iter_exception_tree(propagated_failure)}
        retained_failure = _failure_without_existing_exception_identities(
            failure,
            existing_ids,
        )
        if retained_failure is None:
            continue
        propagated_failure = BaseExceptionGroup(
            group_message,
            [propagated_failure, retained_failure],
        )

    pending_request_count = (
        workspace_observation_pending_cancellation_requests(authoritative_failure)
        + requests_consumed
    )
    if pending_request_count:
        retain_workspace_observation_pending_cancellation_requests(
            propagated_failure,
            pending_request_count,
        )
    return propagated_failure


def _session_interruption_cleanup_child_error(
    error: BaseException | None,
    *,
    operation: str,
    preserved_failure: BaseException | None = None,
) -> BaseException | None:
    """Classify cleanup-owned cancellation as an operational failure."""

    if error is None:
        return None
    if error is preserved_failure:
        return error
    if isinstance(error, asyncio.CancelledError):
        return unexpected_child_cancellation_error(
            error,
            operation=operation,
        )
    if not isinstance(error, BaseExceptionGroup):
        return error

    pending: list[tuple[BaseException, bool]] = [(error, False)]
    children_by_group: dict[int, tuple[BaseException, ...]] = {}
    classified_by_identity: dict[int, tuple[BaseException, bool]] = {}
    while pending:
        candidate, expanded = pending.pop()
        candidate_id = id(candidate)
        if candidate_id in classified_by_identity:
            continue
        if candidate is preserved_failure:
            classified_by_identity[candidate_id] = (candidate, True)
            continue
        if isinstance(candidate, asyncio.CancelledError):
            classified_by_identity[candidate_id] = (
                unexpected_child_cancellation_error(
                    candidate,
                    operation=operation,
                ),
                False,
            )
            continue
        if not isinstance(candidate, BaseExceptionGroup):
            classified_by_identity[candidate_id] = (candidate, False)
            continue
        if expanded:
            children = children_by_group.pop(candidate_id, ())
            classified_children: list[BaseException] = []
            contains_preserved_failure = False
            changed = False
            for child in children:
                classified_child, child_contains_preserved = classified_by_identity[id(child)]
                classified_children.append(classified_child)
                contains_preserved_failure = contains_preserved_failure or child_contains_preserved
                changed = changed or classified_child is not child
            if contains_preserved_failure or changed:
                classified_by_identity[candidate_id] = (
                    BaseExceptionGroup(
                        f"{operation} child failures.",
                        classified_children,
                    ),
                    contains_preserved_failure,
                )
            else:
                classified_by_identity[candidate_id] = (candidate, False)
            continue
        children = exception_group_children(candidate)
        if children is None:
            classified_by_identity[candidate_id] = (
                RuntimeError(f"{operation} returned an unreadable exception group."),
                False,
            )
            continue
        children_by_group[candidate_id] = children
        pending.append((candidate, True))
        pending.extend((child, False) for child in reversed(children))

    classified, _contains_preserved_failure = classified_by_identity[id(error)]
    return classified


class SessionInterruptionCleanup:
    """Settle interruption cleanup while retaining caller control and failures."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        session_finalization: SessionFinalization,
        recovery_ownership: RecoveryOwnership,
        pending_tool_round_recovery: PendingToolRoundRecovery,
    ) -> None:
        self.session_store = session_store
        self._session_finalization = session_finalization
        self._recovery_ownership = recovery_ownership
        self._pending_tool_round_recovery = pending_tool_round_recovery

    async def handle_session_interrupted_preserving_failure(
        self,
        *,
        authoritative_failure: BaseException,
        foreground_subagent_pending: ForegroundSubagentRecoveryRequired | None = None,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        prepare_interruption: Callable[[], Awaitable[Iterable[Event] | None]] | None = None,
        include_interrupted_session: bool = False,
        run_started_at: float | None = None,
        turn_usage_tracker: SessionUsageTracker | None = None,
        active_run: ActiveSessionRun[SessionUsageTracker] | None = None,
    ) -> tuple[list[Event], BaseException]:
        """Settle interruption durably without losing later caller control."""

        current_task = asyncio.current_task()
        retained_request_count = workspace_observation_pending_cancellation_requests(
            authoritative_failure
        )
        current_request_count = 0 if current_task is None else current_task.cancelling()
        historical_request_count = max(current_request_count - retained_request_count, 0)
        entry_request_count = max(retained_request_count, current_request_count)
        concurrent_controls: list[BaseException] = []
        if current_task is not None and current_request_count > historical_request_count:
            # Requests already represented by the authoritative failure were
            # restored before it crossed this boundary. Consume only those
            # owned requests; an older, previously handled cancellation count
            # is task history, not a newly delivered control signal.
            consume_pending_task_cancellation(
                preserve_requests=historical_request_count,
            )

        events: list[Event] = []
        owned_cleanup_failures: list[BaseException] = []

        async def collect_events() -> None:
            persisted_session = await self.session_store.load(session.id)
            allowed_statuses = _INTERRUPTIBLE_SESSION_STATUSES | {SessionStatus.INTERRUPTING}
            if include_interrupted_session:
                allowed_statuses = allowed_statuses | {SessionStatus.INTERRUPTED}
            if persisted_session is None or persisted_session.status not in allowed_statuses:
                return
            if prepare_interruption is not None:
                try:
                    prepared_events = await prepare_interruption()
                except BaseException as preparation_failure:
                    # Preparatory tool-round or deferred-input cleanup must not
                    # prevent the authoritative session transition. Retain the
                    # exact failure and continue to terminalize the session.
                    owned_cleanup_failures.append(preparation_failure)
                else:
                    if prepared_events is not None:
                        events.extend(prepared_events)
            async for event in self._session_finalization.handle_session_interrupted(
                session=persisted_session,
                foreground_subagent_pending=foreground_subagent_pending,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                run_started_at=run_started_at,
                turn_usage_tracker=turn_usage_tracker,
                active_run=active_run,
                terminal_finalization_handoff_source_task=current_task,
            ):
                events.append(event)

        cleanup_task = asyncio.create_task(
            capture_awaitable_outcome(collect_events),
            name="cayu-session-interruption-cleanup",
        )
        cancellation_requests_consumed = 0
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as cancellation:
                requests_before = 0 if current_task is None else current_task.cancelling()
                retained_cancellation = consume_pending_task_cancellation(
                    cancellation,
                    preserve_requests=historical_request_count,
                )
                requests_after = 0 if current_task is None else current_task.cancelling()
                cancellation_requests_consumed += max(
                    requests_before - requests_after,
                    0,
                )
                concurrent_controls.append(retained_cancellation or cancellation)
            except BaseException as control:
                # A supervisory exit cannot orphan the durable interruption
                # mutation. Preserve it as data until the exact cleanup task
                # reaches a terminal outcome.
                concurrent_controls.append(control)

        captured = cleanup_task.result()
        for event in events:
            if event.type in {
                EventType.SESSION_COMPLETED,
                EventType.SESSION_FAILED,
                EventType.SESSION_INTERRUPTED,
            }:
                _mark_session_invocation_terminal_event(event)
        classified_cleanup_failures: list[BaseException] = []
        for cleanup_failure in (*owned_cleanup_failures, captured.error):
            cleanup_error = _session_interruption_cleanup_child_error(
                cleanup_failure,
                operation="Session interruption cleanup",
            )
            if cleanup_error is not None and all(
                cleanup_error is not retained for retained in classified_cleanup_failures
            ):
                classified_cleanup_failures.append(cleanup_error)

        failures: list[BaseException] = [authoritative_failure]
        for additional_failure in (*concurrent_controls, *classified_cleanup_failures):
            existing_ids = {
                id(candidate)
                for retained_failure in failures
                for candidate in iter_exception_tree(retained_failure)
            }
            retained_failure = _failure_without_existing_exception_identities(
                additional_failure,
                existing_ids,
            )
            if retained_failure is not None:
                failures.append(retained_failure)
        propagated_failure = (
            authoritative_failure
            if len(failures) == 1
            else BaseExceptionGroup(
                "Session interruption cleanup received additional failures.",
                failures,
            )
        )
        pending_request_count = entry_request_count + cancellation_requests_consumed
        if pending_request_count:
            retain_workspace_observation_pending_cancellation_requests(
                propagated_failure,
                pending_request_count,
            )
        return events, propagated_failure

    async def settle_session_finalizer_preserving_interruption_failure(
        self,
        *,
        authoritative_failure: BaseException,
        steps: tuple[tuple[str, Callable[[], Awaitable[None]]], ...],
    ) -> BaseException:
        """Settle finalizer awaits without replacing the authoritative interruption."""

        if isinstance(authoritative_failure, asyncio.CancelledError) and not (
            workspace_observation_pending_cancellation_requests(authoritative_failure)
        ):
            # Ordinary caller cancellation retains its public exception shape.
            # A failed classifier can leave no terminal settlement; fence-release
            # failure is then secondary, not a replacement for cancellation.
            await self._recovery_ownership.run_cleanup_steps(
                authoritative_failure=authoritative_failure, steps=steps
            )
            return authoritative_failure

        propagated_failure = authoritative_failure
        current_task = asyncio.current_task()
        retained_request_count = workspace_observation_pending_cancellation_requests(
            authoritative_failure
        )
        current_request_count = 0 if current_task is None else current_task.cancelling()
        historical_request_count = max(current_request_count - retained_request_count, 0)
        if current_task is not None and current_request_count > historical_request_count:
            # Python 3.11 and 3.12 do not rescind an undelivered cancellation
            # when ``uncancel()`` reaches zero. Observe the pending injection
            # first, then normalize only requests already represented by the
            # authoritative failure. Preserve and aggregate a genuinely new
            # request that arrives during this checkpoint.
            pending_control: asyncio.CancelledError | None = None
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as control:
                pending_control = control
            if pending_control is not None:
                request_count_after_checkpoint = current_task.cancelling()
                new_request_count = max(
                    request_count_after_checkpoint - current_request_count,
                    0,
                )
                requests_to_preserve = historical_request_count + new_request_count
                consume_pending_task_cancellation(
                    pending_control,
                    preserve_requests=requests_to_preserve,
                )
                if new_request_count:
                    propagated_failure = _session_interruption_failure_with_additional_control(
                        propagated_failure,
                        pending_control,
                        group_message=(
                            "Session interruption finalization received additional control signals."
                        ),
                        historical_cancellation_requests=requests_to_preserve,
                    )
                    retain_workspace_observation_pending_cancellation_requests(
                        propagated_failure,
                        retained_request_count + new_request_count,
                    )

        for operation, step in steps:
            current_task = asyncio.current_task()
            historical_cancellation_requests = (
                0 if current_task is None else current_task.cancelling()
            )
            cleanup_task = asyncio.create_task(
                capture_awaitable_outcome(step),
                name="cayu-session-interruption-finalizer",
            )
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except BaseException as control:
                    propagated_failure = _session_interruption_failure_with_additional_control(
                        propagated_failure,
                        control,
                        group_message=(
                            "Session interruption finalization received additional control signals."
                        ),
                        historical_cancellation_requests=(historical_cancellation_requests),
                    )

            captured = cleanup_task.result()
            cleanup_error = _session_interruption_cleanup_child_error(
                captured.error,
                operation=operation,
                preserved_failure=authoritative_failure,
            )
            if cleanup_error is not None:
                propagated_failure = _session_interruption_failure_with_additional_control(
                    propagated_failure,
                    cleanup_error,
                    group_message="Session interruption finalization failed.",
                    historical_cancellation_requests=(historical_cancellation_requests),
                )
        return propagated_failure

    async def close_durable_pending_tool_round_after_interrupt(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Close a round when interruption precedes creation of its live runner."""
        _checkpoint, pending_round = await pending_round_reader.load_pending_tool_round(
            self.session_store,
            session.id,
        )
        if pending_round is None:
            return
        messages = await self.session_store.load_transcript(session.id)
        request = InterruptedToolRoundRequest(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            messages=messages,
            tool_calls=tool_round_recovery.pending_round_tool_calls(pending_round),
            tool_outcomes=[],
            tool_round_identity=pending_rounds.pending_tool_round_identity(pending_round),
            cancellation_artifacts=None,
            cancellation_artifacts_by_id=None,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
        )
        async for event in self._pending_tool_round_recovery.close_interrupted_tool_round(request):
            yield event
