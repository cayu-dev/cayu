"""Own recovery claims, worker settlement and supervised release."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any, TypeVar
from uuid import uuid4

from cayu._exception_groups import (
    _attach_exception_cause_preserving_graph,
    exception_cause,
    failure_control_cause,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    copy_durable_record,
    copy_json_value,
    require_clean_nonblank,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
)
from cayu.runtime._interruption_coordinator import (
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
    _release_invocation_command_with_cleanup_authority,
    prepare_rebind_invocation_command,
)
from cayu.runtime._recovery_claims import (
    _IncompleteRecoveryClaim,
    _IncompleteRecoveryClaimAuthority,
    _IncompleteRecoveryClaimLost,
    _RecoveryWorkerSettlement,
    _require_live_incomplete_recovery_claim_acknowledgement,
)
from cayu.runtime._run_limits import (
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    SessionControl,
)
from cayu.runtime._terminal_evidence_reader import (
    _TERMINAL_EVENT_TYPE_BY_STATUS,
    TerminalEvidenceReader,
)
from cayu.runtime._terminal_finalization_lifetime import _terminal_finalization_process_control
from cayu.sessions._checkpoint_preservation import (
    _invocation_lifecycle_authority_read_scope,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.sessions._invocation_lifecycle import (
    InvocationLifecycleCommandConflict,
    InvocationMutationResult,
    ReleaseInvocationCommand,
    invocation_lifecycle_receipt_history_present,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalOutcome,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_active_profile,
    invocation_terminal_decision_matches_recovery_profile,
)
from cayu.sessions._provider_operation_cancellation_claim import (
    active_provider_operation_cancellation_claim_from_checkpoint,
)
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
    _SESSION_RUN_OPERATION_CHECKPOINT_KEY,
    _session_run_operation_from_checkpoint,
    interruption_request_id_from_payload,
)
from cayu.sessions.base import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    CheckpointTransform,
    SessionRunFenced,
    SessionStatusConflict,
    SessionStore,
    StoreTimeCheckpointTransform,
    _activate_owned_session_run_fence,
    _activate_session_run_fence,
    _deactivate_session_run_fence,
    _incomplete_recovery_claim_from_checkpoint,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.cleanup import (
    RecoveryCleanupStepInput,
    RecoveryCleanupSupervisor,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.interactions import (
    INTERACTION_LIFECYCLE_EVENT_TYPES,
    INTERACTION_TERMINAL_EVENT_TYPES,
)
from cayu.sessions.records import (
    Session,
    SessionStatus,
)

_INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(minutes=5)

_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS = 30.0

_INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_RETRY_SECONDS = 5.0

_MANUAL_RECOVERY_INTERRUPT_POLL_INTERVAL_SECONDS = 0.25

_RecoveryResultT = TypeVar("_RecoveryResultT")


def _recovery_abandonment_signal(
    error: BaseException | None,
    *,
    cancellation_baseline: int = 0,
) -> GeneratorExit | asyncio.CancelledError | None:
    """Find explicit abandonment, preferring cancellation for cleanup shielding."""
    if isinstance(error, GeneratorExit | asyncio.CancelledError):
        return error
    if isinstance(error, BaseExceptionGroup):
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        cancellation_delivered = task is None or task.cancelling() > cancellation_baseline
        generator_exit: GeneratorExit | None = None
        for candidate in iter_exception_tree(error):
            if isinstance(candidate, asyncio.CancelledError) and cancellation_delivered:
                return candidate
            if isinstance(candidate, GeneratorExit) and generator_exit is None:
                generator_exit = candidate
        return generator_exit
    return None


def _prepend_exception_cause(error: BaseException, cause: BaseException) -> None:
    """Preserve a new structured cause without discarding an existing chain."""
    set_exception_cause(cause, exception_cause(error))
    set_exception_cause(error, cause)


def _recovery_failure_contains_process_control(failure: BaseException | None) -> bool:
    if failure is None:
        return False
    return any(
        isinstance(candidate, (GeneratorExit, KeyboardInterrupt, SystemExit))
        for candidate in iter_exception_tree(failure)
        if not isinstance(candidate, BaseExceptionGroup)
    )


def _authoritative_recovery_ownership_failure(
    operation_failure: BaseException | None,
    ownership_failure: BaseException,
) -> BaseException:
    """Select recovery-operation authority without dropping ownership evidence."""

    if _recovery_failure_contains_process_control(operation_failure) or isinstance(
        operation_failure,
        asyncio.CancelledError,
    ):
        assert operation_failure is not None
        _attach_exception_cause_preserving_graph(operation_failure, ownership_failure)
        return operation_failure
    if operation_failure is not None:
        _attach_exception_cause_preserving_graph(ownership_failure, operation_failure)
    return ownership_failure


async def _run_recovery_cleanup_steps(
    *,
    authoritative_failure: BaseException | None,
    steps: tuple[RecoveryCleanupStepInput, ...],
    cancellation_baseline: int = 0,
    supervisor: RecoveryCleanupSupervisor | None = None,
) -> tuple[tuple[str, BaseException], ...]:
    """Run every handoff cleanup without obscuring its triggering failure.

    Once task cancellation starts a continuation handoff, a later ``cancel()``
    must not interrupt finalization or fence release. Run that cleanup in a
    shielded child task which inherits the current run-fence context, and wait
    through repeated cancellation requests until the shared finite deadline
    transfers outcome-unknown ownership. ``GeneratorExit`` is different: an
    explicit ``aclose()`` consumes it, so a cleanup failure must remain visible to
    the caller instead of being reduced to an exception note.
    """

    abandonment = _recovery_abandonment_signal(
        authoritative_failure,
        cancellation_baseline=cancellation_baseline,
    )
    cleanup_supervisor = supervisor or RecoveryCleanupSupervisor()
    cleanup_outcome = await cleanup_supervisor.run_steps_with_control(
        steps=steps,
        shield_caller_cancellation=isinstance(abandonment, asyncio.CancelledError),
    )
    cleanup_failures = cleanup_outcome.failures

    control: BaseException | None = cleanup_outcome.caller_cancellation
    for _operation, failure in cleanup_failures:
        fatal = _terminal_finalization_process_control(failure)
        if fatal is not None:
            control = fatal
            break
    if (
        control is not None
        and authoritative_failure is not None
        and abandonment is None
        and not _recovery_failure_contains_process_control(authoritative_failure)
    ):
        # Fatal signals come from explicit cleanup trees; caller cancellation
        # comes only from the supervising await, never a child or old cause.
        cause = failure_control_cause(
            [authoritative_failure, *(failure for _operation, failure in cleanup_failures)],
            control,
        )
        if cause is not None:
            _attach_exception_cause_preserving_graph(control, cause)
        raise control from exception_cause(control)

    if not cleanup_failures:
        return ()
    if isinstance(abandonment, asyncio.CancelledError) and authoritative_failure is not None:
        fatal_cleanup_failures = [
            failure
            for _operation, failure in cleanup_failures
            if any(
                not isinstance(candidate, BaseExceptionGroup)
                and not isinstance(candidate, (Exception, asyncio.CancelledError))
                for candidate in iter_exception_tree(failure)
            )
        ]
        if fatal_cleanup_failures:
            fatal_failure: BaseException
            if len(cleanup_failures) == 1:
                fatal_failure = cleanup_failures[0][1]
            else:
                fatal_failure = BaseExceptionGroup(
                    "Continuation recovery cleanup and process-control failures",
                    [failure for _operation, failure in cleanup_failures],
                )
            if not _attach_exception_cause_preserving_graph(
                fatal_failure,
                authoritative_failure,
            ):
                fatal_failure = BaseExceptionGroup(
                    "Continuation recovery cancellation and fatal cleanup failure",
                    [fatal_failure, authoritative_failure],
                )
            raise fatal_failure
    if authoritative_failure is not None and not isinstance(authoritative_failure, GeneratorExit):
        failures = tuple(cleanup_failures)
        for operation, cleanup_failure in cleanup_failures:
            authoritative_failure.add_note(
                "Continuation recovery cleanup failed during "
                f"{operation}: {type(cleanup_failure).__name__}. "
                "The original failure remains authoritative."
            )
        cleanup_group = BaseExceptionGroup(
            "Continuation recovery cleanup failures",
            [failure for _operation, failure in failures],
        )
        _prepend_exception_cause(authoritative_failure, cleanup_group)
        return failures

    operation, first_failure = cleanup_failures[0]
    for later_operation, later_failure in cleanup_failures[1:]:
        first_failure.add_note(
            "Additional continuation recovery cleanup failure during "
            f"{later_operation}: {later_failure!r}."
        )
    first_failure.add_note(f"Continuation recovery cleanup failed during {operation}.")
    if len(cleanup_failures) > 1:
        _prepend_exception_cause(
            first_failure,
            BaseExceptionGroup(
                "Additional continuation recovery cleanup failures",
                [failure for _operation, failure in cleanup_failures[1:]],
            ),
        )
    raise first_failure


def _consume_incomplete_recovery_store_task(task: asyncio.Task[Any]) -> None:
    """Observe a store mutation retained past the local ownership deadline."""

    with contextlib.suppress(BaseException):
        task.result()


def _incomplete_recovery_session_reservation_authority(
    session: Session,
) -> tuple[object, ...]:
    """Return the session fields that authorize stalled-run takeover.

    Labels, user metadata, and ``updated_at`` are deliberately excluded:
    annotation writes do not refresh ``last_activity_at`` and therefore must
    not let an expired owner evade recovery. Immutable identity, invocation,
    status, epoch, and activity evidence remain exact.
    """

    return (
        session.id,
        session.instance_id,
        session.agent_name,
        session.provider_name,
        session.model,
        session.parent_session_id,
        session.causal_budget_id,
        session.runtime_name,
        session.runtime_version,
        session.runtime_build_provenance,
        session.environment_name,
        session.status,
        session.created_at,
        session.last_activity_at,
        session.run_epoch,
        session.invocation,
    )


def _checkpoint_with_rebased_session_run_operation(
    checkpoint: dict[str, Any],
    *,
    previous_run_epoch: int,
    run_epoch: int,
) -> dict[str, Any]:
    """Transfer an unfinished run publication to a newly fenced recovery epoch."""
    if run_epoch != previous_run_epoch + 1:
        raise ValueError(
            "A session run operation can be rebased only to the next fenced run epoch."
        )
    operation = _session_run_operation_from_checkpoint(checkpoint)
    if operation is None:
        return checkpoint
    if operation.run_epoch > previous_run_epoch:
        raise RuntimeError(
            "Session run operation belongs to a future run epoch and cannot be recovered."
        )
    updated = copy_durable_record(checkpoint, "checkpoint")
    marker: dict[str, Any] = {
        "version": 1,
        "operation_id": operation.operation_id,
        "run_epoch": run_epoch,
    }
    if operation.terminal_event_id is not None:
        marker["terminal_event_id"] = operation.terminal_event_id
    if operation.queue_task_id is not None:
        marker["queue_task_id"] = operation.queue_task_id
    updated[_SESSION_RUN_OPERATION_CHECKPOINT_KEY] = marker
    return updated


def _require_aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value


class RecoveryOwnership:
    """Own recovery claims, worker settlement and supervised release."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        environment_lifecycle: EnvironmentLifecycle,
        session_control: SessionControl[SessionUsageTracker],
        recovery_cleanup_supervisor: RecoveryCleanupSupervisor,
        terminal_evidence: TerminalEvidenceReader,
    ) -> None:
        self._session_store = session_store
        self._environment_lifecycle = environment_lifecycle
        self._session_control = session_control
        self._recovery_cleanup_supervisor = recovery_cleanup_supervisor
        self._terminal_evidence = terminal_evidence
        self._recovery_claim_workers: dict[tuple[str, str], _RecoveryWorkerSettlement] = {}
        # Retained writes may still be using stores after their lease expired.
        self._detached_recovery_writes: set[asyncio.Task[Any]] = set()

    def detached_recovery_work(self) -> set[asyncio.Future[Any]]:
        """Return store writes that must settle before application shutdown."""
        return set(self._detached_recovery_writes)

    @property
    def claim_lease_duration(self) -> timedelta:
        """Share the current store-time lease policy with live finalization."""
        return _INCOMPLETE_RECOVERY_CLAIM_LEASE

    async def run_cleanup_steps(
        self,
        *,
        authoritative_failure: BaseException | None,
        steps: tuple[RecoveryCleanupStepInput, ...],
        cancellation_baseline: int = 0,
    ) -> tuple[tuple[str, BaseException], ...]:
        return await _run_recovery_cleanup_steps(
            authoritative_failure=authoritative_failure,
            steps=steps,
            cancellation_baseline=cancellation_baseline,
            supervisor=self._recovery_cleanup_supervisor,
        )

    async def fence_or_rebind_active_invocation(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
        target_status: SessionStatus | None = None,
    ) -> Session:
        """Advance one recovery epoch through the typed invocation seam.

        Sessions with positive active-invocation authority use the command
        protocol.  Its durable CAS rejects a stale preflight.  A session with
        no such authority remains an intentional pre-invocation recovery case.
        """

        session = await self._session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        checkpoint = await self._session_store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active_profile is None:
            if target_status is None:
                return await self._session_store.fence_run_and_transform_checkpoint(
                    session_id,
                    statuses=statuses,
                    checkpoint_transform=checkpoint_transform,
                )
            return await self._session_store.transition_status_and_checkpoint(
                session_id,
                from_statuses=statuses,
                to_status=target_status,
                checkpoint_transform=checkpoint_transform,
            )
        command = prepare_rebind_invocation_command(
            session,
            checkpoint,
            expected_statuses=statuses,
            checkpoint_transform=checkpoint_transform,
            target_status=target_status,
        )
        result = await self._session_store.apply_invocation_lifecycle_command(command)
        if type(result) is not InvocationMutationResult:
            raise RuntimeError("Invocation rebind returned an incompatible result.")
        return result.session

    async def reserve_and_fence_incomplete_recovery(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_for_seconds: int | None,
        checkpoint_transform: StoreTimeCheckpointTransform,
        target_status: SessionStatus | None = None,
    ) -> Session:
        """Reserve by store time, then cross the typed invocation rebind seam."""

        desired_checkpoint: dict[str, Any] | None = None
        reserved_checkpoint: dict[str, Any] | None = None
        reserved_session: Session | None = None

        def reserve(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> dict[str, Any] | None:
            nonlocal desired_checkpoint, reserved_checkpoint, reserved_session
            from cayu.runtime._abandoned_session_recovery import require_abandoned_execution_matches

            require_abandoned_execution_matches(current_session)
            desired = checkpoint_transform(current_session, checkpoint, store_now)
            if desired is None:
                return None
            marker = desired.get(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY)
            if type(marker) is not dict:
                raise RuntimeError("Recovery reservation did not produce its claim marker.")
            reserved = {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
            reserved[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
            reserved[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = copy_json_value(
                marker,
                "incomplete_session_recovery_claim",
            )
            desired_checkpoint = copy_durable_record(desired, "checkpoint")
            reserved_checkpoint = copy_durable_record(reserved, "checkpoint")
            reserved_session = current_session.model_copy(deep=True)
            return reserved

        async def release_tentative_reservation() -> None:
            if (
                reserved_checkpoint is None
                or reserved_session is None
                or desired_checkpoint is None
            ):
                return
            reserved_marker = reserved_checkpoint.get(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY)

            def release(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
                _store_now: datetime,
            ) -> dict[str, Any] | None:
                if checkpoint is None or (
                    checkpoint.get(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY) != reserved_marker
                ):
                    return None
                if (
                    current_session.run_epoch == reserved_session.run_epoch + 1
                    and checkpoint == desired_checkpoint
                ):
                    # The fencing transition committed and only its
                    # acknowledgement was lost. Leave reconciliation authority
                    # intact for the caller.
                    return None
                updated = copy_durable_record(checkpoint, "checkpoint")
                updated.pop(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY, None)
                return updated

            await self._session_store.reserve_stalled_run_recovery(
                session_id,
                statuses=set(SessionStatus),
                inactive_for_seconds=None,
                checkpoint_transform=release,
            )

        try:
            with _invocation_lifecycle_authority_read_scope():
                reserved = await self._session_store.reserve_stalled_run_recovery(
                    session_id,
                    statuses=statuses,
                    inactive_for_seconds=inactive_for_seconds,
                    checkpoint_transform=reserve,
                )
        except BaseException as failure:
            await _run_recovery_cleanup_steps(
                authoritative_failure=failure,
                steps=(
                    (
                        "tentative recovery reservation release",
                        release_tentative_reservation,
                    ),
                ),
            )
            raise
        if (
            reserved is None
            or reserved_session is None
            or reserved_checkpoint is None
            or desired_checkpoint is None
        ):
            raise _IncompleteRecoveryClaimLost(
                "Incomplete-session recovery is not eligible at store time."
            )
        if reserved != reserved_session:
            raise RuntimeError("Recovery reservation returned conflicting session authority.")

        def fence_reserved(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            if (
                _incomplete_recovery_session_reservation_authority(current_session)
                != _incomplete_recovery_session_reservation_authority(reserved_session)
                or checkpoint != reserved_checkpoint
            ):
                raise _IncompleteRecoveryClaimLost(
                    "Incomplete-session recovery reservation changed before fencing."
                )
            return copy_durable_record(desired_checkpoint, "checkpoint")

        try:
            return await self.fence_or_rebind_active_invocation(
                session_id,
                statuses=statuses,
                checkpoint_transform=fence_reserved,
                target_status=target_status,
            )
        except BaseException as failure:
            await _run_recovery_cleanup_steps(
                authoritative_failure=failure,
                steps=(("tentative recovery reservation release", release_tentative_reservation),),
            )
            raise

    async def claim_provider_operation_interruption(
        self,
        session: Session,
        execution_profile_snapshot: ActiveInvocationExecutionProfile,
        *,
        interruption_request_id: str,
        invocation_context: InvocationContext,
    ) -> Session:
        """Fence a dead provider worker and bind interruption to one recovery epoch."""

        if type(execution_profile_snapshot) is not ActiveInvocationExecutionProfile:
            raise TypeError(
                "Provider-operation interruption requires active invocation profile authority."
            )
        if type(invocation_context) is not InvocationContext:
            raise TypeError(
                "Provider-operation interruption requires authenticated invocation context."
            )
        if (
            invocation_context.binding.session_id != session.id
            or invocation_context.binding.session_instance_id != session.instance_id
            or invocation_context.binding.run_epoch != session.run_epoch
            or invocation_context.active_profile is not execution_profile_snapshot
        ):
            raise RuntimeError(
                "Provider-operation interruption substituted frozen invocation authority."
            )
        interruption_request_id = require_clean_nonblank(
            interruption_request_id,
            "interruption_request_id",
        )

        def claim_interruption(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            current_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            terminal_decision = invocation_terminal_decision_from_checkpoint(checkpoint)
            pending_interrupt = (
                None
                if checkpoint is None
                else checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            )
            if (
                checkpoint is None
                or type(pending_interrupt) is not dict
                or interruption_request_id_from_payload(pending_interrupt)
                != interruption_request_id
                or _incomplete_recovery_claim_from_checkpoint(checkpoint) is not None
            ):
                raise SessionRunFenced(
                    "Provider-operation interruption request ownership changed before claim."
                )
            if (
                type(current_profile) is not ActiveInvocationExecutionProfile
                or current_profile.session_id != execution_profile_snapshot.session_id
                or current_profile.interaction_id != execution_profile_snapshot.interaction_id
                or current_profile.profile != execution_profile_snapshot.profile
                or execution_profile_snapshot.run_epoch != current_session.run_epoch
                or current_profile.run_epoch
                not in {
                    current_session.run_epoch,
                    current_session.run_epoch - 1,
                }
            ):
                raise RuntimeError(
                    "Provider-operation interruption profile changed before recovery claimed it."
                )
            if terminal_decision is not None and (
                terminal_decision.outcome is not InvocationTerminalOutcome.INTERRUPTED
                or terminal_decision.interruption_request_id != interruption_request_id
                or not invocation_terminal_decision_matches_active_profile(
                    terminal_decision,
                    session_id=current_session.id,
                    session_instance_id=current_session.instance_id,
                    run_epoch=current_session.run_epoch,
                    interaction_id=current_profile.interaction_id,
                    execution_profile_fingerprint=current_profile.profile.fingerprint,
                )
            ):
                raise SessionRunFenced(
                    "Provider-operation interruption lost its terminal decision authority."
                )
            if current_profile.session_id != current_session.id:
                raise SessionRunFenced(
                    "Provider-operation interruption no longer owns the active invocation epoch."
                )
            return checkpoint_with_active_invocation_execution_profile(
                _checkpoint_with_rebased_session_run_operation(
                    checkpoint,
                    previous_run_epoch=current_session.run_epoch,
                    run_epoch=current_session.run_epoch + 1,
                ),
                session_id=current_session.id,
                interaction_id=current_profile.interaction_id,
                run_epoch=current_session.run_epoch + 1,
                profile=current_profile.profile,
                expected=current_profile,
            )

        claim_task = asyncio.create_task(
            self.fence_or_rebind_active_invocation(
                session.id,
                statuses={SessionStatus.INTERRUPTING},
                checkpoint_transform=claim_interruption,
            )
        )
        outcome = await await_shielded_task_outcome(claim_task)
        error = outcome.error
        if isinstance(error, asyncio.CancelledError) and outcome.cancellation is None:
            error = unexpected_child_cancellation_error(
                error,
                operation="Provider-operation interruption claim",
            )
        cancellation = outcome.cancellation
        claimed = outcome.result
        if error is not None:
            reconciliation = await await_shielded_task_outcome(
                asyncio.create_task(
                    self.load_claimed_provider_operation_interruption(
                        session_id=session.id,
                        interaction_id=execution_profile_snapshot.interaction_id,
                        run_epoch=session.run_epoch + 1,
                        execution_profile=execution_profile_snapshot.profile,
                        interruption_request_id=interruption_request_id,
                    )
                ),
                cancellation=cancellation,
            )
            cancellation = reconciliation.cancellation or cancellation
            reconciliation_error = reconciliation.error
            if (
                isinstance(reconciliation_error, asyncio.CancelledError)
                and reconciliation.cancellation is None
            ):
                reconciliation_error = unexpected_child_cancellation_error(
                    reconciliation_error,
                    operation="Provider-operation interruption claim reconciliation",
                )
            if reconciliation_error is not None:
                if cancellation is not None:
                    cancellation.add_note(
                        "Provider-operation interruption claim reconciliation also failed: "
                        f"{type(reconciliation_error).__name__}."
                    )
                    _prepend_exception_cause(
                        cancellation,
                        BaseExceptionGroup(
                            "Provider-operation interruption claim failures",
                            [error, reconciliation_error],
                        ),
                    )
                    raise cancellation
                if not isinstance(reconciliation_error, Exception):
                    raise reconciliation_error from error
                error.add_note(
                    "Provider-operation interruption claim reconciliation failed: "
                    f"{type(reconciliation_error).__name__}."
                )
                raise error from reconciliation_error
            claimed = reconciliation.result
            if claimed is None:
                if cancellation is not None:
                    cancellation.add_note(
                        f"Provider-operation interruption claim also failed: {type(error).__name__}."
                    )
                    raise cancellation from error
                raise error
            claimed_invocation_context = invocation_context.with_rebound_session(
                claimed,
                active_profile=execution_profile_snapshot.model_copy(
                    update={"run_epoch": claimed.run_epoch}
                ),
            )
            _activate_session_run_fence(claimed)
            authoritative_failure = cancellation or error
            try:
                await self.run_cleanup_steps(
                    authoritative_failure=authoritative_failure,
                    steps=(
                        (
                            "failed provider-operation interruption claim release",
                            lambda: (
                                self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                    session_id=claimed.id,
                                    execution_profile=execution_profile_snapshot.profile,
                                    invocation_context=claimed_invocation_context,
                                )
                            ),
                        ),
                    ),
                )
            finally:
                _deactivate_session_run_fence(claimed.id)
            if cancellation is not None:
                cancellation.add_note(
                    f"Provider-operation interruption claim also failed: {type(error).__name__}."
                )
                raise cancellation from error
            raise error
        if claimed is None:
            missing = RuntimeError(
                "Provider-operation interruption claim returned no session authority."
            )
            if cancellation is not None:
                cancellation.add_note(str(missing))
                raise cancellation from missing
            raise missing
        claimed_invocation_context = invocation_context.with_rebound_session(
            claimed,
            active_profile=execution_profile_snapshot.model_copy(
                update={"run_epoch": claimed.run_epoch}
            ),
        )
        _activate_session_run_fence(claimed)
        if cancellation is not None:
            try:
                await self.run_cleanup_steps(
                    authoritative_failure=cancellation,
                    steps=(
                        (
                            "cancelled provider-operation interruption run-fence release",
                            lambda: (
                                self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                                    session_id=claimed.id,
                                    execution_profile=execution_profile_snapshot.profile,
                                    invocation_context=claimed_invocation_context,
                                )
                            ),
                        ),
                    ),
                )
            finally:
                _deactivate_session_run_fence(claimed.id)
            raise cancellation
        return claimed

    async def load_claimed_provider_operation_interruption(
        self,
        *,
        session_id: str,
        interaction_id: str,
        run_epoch: int,
        execution_profile: ExecutionProfileIdentity,
        interruption_request_id: str,
    ) -> Session | None:
        """Reconcile acknowledgement loss only against the complete claimed identity."""

        current = await self._session_store.load(session_id)
        if (
            current is None
            or current.status is not SessionStatus.INTERRUPTING
            or current.run_epoch != run_epoch
        ):
            return None
        checkpoint = await self._session_store.load_checkpoint(session_id)
        current_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        terminal_decision = invocation_terminal_decision_from_checkpoint(checkpoint)
        pending_interrupt = (
            None
            if checkpoint is None
            else checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
        )
        if (
            current_profile is None
            or current_profile.session_id != session_id
            or current_profile.interaction_id != interaction_id
            or current_profile.run_epoch != run_epoch
            or current_profile.profile != execution_profile
            or type(pending_interrupt) is not dict
            or interruption_request_id_from_payload(pending_interrupt) != interruption_request_id
            or _incomplete_recovery_claim_from_checkpoint(checkpoint) is not None
            or (
                terminal_decision is not None
                and (
                    terminal_decision.outcome is not InvocationTerminalOutcome.INTERRUPTED
                    or terminal_decision.interruption_request_id != interruption_request_id
                    or not invocation_terminal_decision_matches_recovery_profile(
                        terminal_decision,
                        session_id=current.id,
                        session_instance_id=current.instance_id,
                        current_run_epoch=current.run_epoch,
                        interaction_id=current_profile.interaction_id,
                        execution_profile_fingerprint=current_profile.profile.fingerprint,
                    )
                )
            )
        ):
            return None
        return current

    async def fence_expired_incomplete_recovery_claim(
        self,
        *,
        session: Session,
        claim_id: str,
    ) -> bool:
        """Fence and clear one observed expired recovery owner.

        The claim id makes the takeover conditional: a concurrent heartbeat or
        claimant that changes ownership causes this operation to leave the
        session untouched.
        """
        claim: _IncompleteRecoveryClaim | None = None
        authoritative_failure: BaseException | None = None
        try:
            claim = await self.claim(
                session=session,
                inactive_for_seconds=None,
                required_expired_claim_id=claim_id,
            )
            return claim is not None
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            if claim is not None:
                await self.cleanup_claim(
                    authority=claim.require_authority(),
                    authoritative_failure=authoritative_failure,
                )

    async def cleanup_claim(
        self,
        *,
        authority: _IncompleteRecoveryClaimAuthority,
        authoritative_failure: BaseException | None,
        release_environment_cleanup: bool = False,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        retain_open_interaction_invocation: bool = False,
        retain_invocation_context: Callable[[InvocationContext], None] | None = None,
        claim_has_not_dispatched_work: bool = False,
        recovery_work_quiescent: bool = False,
    ) -> None:
        if not await authority.begin_finalization():
            return
        session_id = authority.session_id
        claim_id = authority.claim_id
        recovery_run_epoch = authority.run_epoch

        async def release_owned_recovery_run_fence() -> None:
            checkpoint = await self._session_store.load_checkpoint(session_id)
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            session = await self.require_session(session_id)
            persisted_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            owns_durable_claim = (
                persisted_claim is not None
                and persisted_claim[0] == claim_id
                and session.run_epoch == authority.run_epoch
            )
            if not owns_durable_claim:
                if (
                    active_profile is not None
                    and invocation_context is not None
                    and session.run_epoch > invocation_context.binding.run_epoch
                ):
                    # A successor owns the session, but the environment cleanup
                    # owner may still need to retire this claim's exact epoch.
                    await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session_id,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    )
                authority.retire()
                return
            if active_profile is not None:
                if invocation_context is not None:
                    if invocation_context.binding.session_instance_id != session.instance_id:
                        raise RuntimeError(
                            "Recovery cleanup lost exact session-incarnation authority."
                        )
                    if session.run_epoch < invocation_context.binding.run_epoch:
                        raise RuntimeError(
                            "Recovery cleanup context is ahead of durable authority."
                        )
                    if session.run_epoch > invocation_context.binding.run_epoch:
                        raise RuntimeError("Recovery cleanup context is behind durable authority.")
                    if invocation_context.active_profile != active_profile:
                        raise RuntimeError(
                            "Recovery cleanup lost exact active invocation authority."
                        )
                if retain_open_interaction_invocation:
                    latest_interaction = await self._session_store.query_events(
                        EventQuery(
                            session_id=session_id,
                            event_types=INTERACTION_LIFECYCLE_EVENT_TYPES,
                            order_by=EventOrder.SEQUENCE_DESC,
                            limit=1,
                        )
                    )
                    if latest_interaction and (
                        latest_interaction[0].event.type not in INTERACTION_TERMINAL_EVENT_TYPES
                    ):
                        # A prior paused settlement may remain a valid receipt
                        # for an older epoch, but it cannot authorize release
                        # while this public recovery still owes the current
                        # interaction's terminal publication.
                        if retain_invocation_context is not None:
                            if invocation_context is None:
                                raise RuntimeError(
                                    "Open recovery invocation lost its authenticated context."
                                )
                            retain_invocation_context(invocation_context.without_recovery_claim())
                        return
                settlement = await self._session_store.load_invocation_settlement_transition(
                    session_id,
                    expected_session_instance_id=session.instance_id,
                    expected_active_invocation_profile=active_profile,
                )
                if settlement is not None and not (
                    settlement.to_status is session.status
                    or (
                        settlement.only_if_no_queued_messages
                        and session.status in settlement.from_statuses
                    )
                ):
                    # Rebind preserves the interaction/profile lineage, so the
                    # store may return a valid receipt from an older epoch. It
                    # cannot settle a different terminal outcome produced by
                    # this exact recovery claim.
                    settlement = None
                if (
                    settlement is not None
                    and settlement.only_if_no_queued_messages
                    and settlement.to_status is SessionStatus.COMPLETED
                    and session.status is SessionStatus.COMPLETED
                ):
                    # The transition spec does not prove its conditional status
                    # change committed. Rejected-only draining may have completed
                    # the session separately, leaving an immutable predecessor
                    # receipt with status_changed=False. Let this recovery owner
                    # release under its exact claim only after the existing
                    # quiescence and terminal-evidence checks below; do not reuse
                    # that predecessor as proof of session completion.
                    completed_receipt = (
                        await self._session_store._load_interaction_transition_receipt_by_event_id(
                            session.id,
                            event_id=settlement.event.id,
                            expected_session_instance_id=session.instance_id,
                            expected_active_invocation_profile=active_profile,
                        )
                    )
                    if (
                        completed_receipt is None
                        or not completed_receipt.status_changed
                        or completed_receipt.transition != settlement
                        or completed_receipt.session.run_epoch != active_profile.run_epoch
                    ):
                        settlement = None
                if settlement is None:
                    if authoritative_failure is not None and not (
                        claim_has_not_dispatched_work or recovery_work_quiescent
                    ):
                        # A failed recovery did not prove its work quiescent.
                        return
                    if session.status not in _TERMINAL_EVENT_TYPE_BY_STATUS:
                        # An exact background retrieval can succeed while its
                        # operation remains pending. Finishing this local claim
                        # does not settle that invocation or authorize release.
                        # Keep the fence for the next exact recovery owner.
                        return
                    inspection = await self._terminal_evidence.inspect(
                        session=session,
                        checkpoint=checkpoint,
                    )
                    if (
                        (inspection.event is None and inspection.terminal_event_required)
                        or inspection.pending_interrupt_payload is not None
                        or inspection.run_operation is not None
                    ):
                        # The claim has not finished its terminal repair. Keep
                        # the invocation fenced for exact retry.
                        return
                    command = _release_invocation_command_with_cleanup_authority(
                        ReleaseInvocationCommand(
                            session_id=session.id,
                            expected_session_instance_id=session.instance_id,
                            expected_run_epoch=active_profile.run_epoch,
                            expected_active_profile=active_profile,
                            recovery_claim_id=claim_id,
                        )
                    )
                    await self._session_store.apply_invocation_lifecycle_command(command)
                else:
                    await self._environment_lifecycle.release_run_fence_after_environment_cleanup(
                        session_id=session_id,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    )
            else:
                if invocation_lifecycle_receipt_history_present(checkpoint):
                    raise RuntimeError(
                        "Incomplete-session recovery cleanup lost durable invocation "
                        "profile authority."
                    )
                await self._session_store.release_run_fence(session_id)
            if recovery_run_epoch is not None:
                self._environment_lifecycle.retire_repaired_run_fence_releases(
                    session_id=session_id,
                    repaired_run_epoch=recovery_run_epoch,
                )

        async def release_recovery_run_fence() -> None:
            # The finalizer can run in a supervisor with an empty or unrelated
            # ContextVar state. Install only this exact owner for store and
            # environment adapters that still consult task-local authority.
            if authority.run_fence.retired:
                return
            with authority.run_fence.activate():
                await release_owned_recovery_run_fence()

        claim_release_completed = False

        async def exact_recovery_claim_is_still_persisted() -> bool:
            checkpoint = await self._session_store.load_checkpoint(session_id)
            persisted_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            return persisted_claim is not None and persisted_claim[0] == claim_id

        async def release_recovery_claim() -> None:
            nonlocal claim_release_completed
            try:
                await self.release_claim(session_id, claim_id)
            except BaseException as release_failure:
                release_cancellation = (
                    release_failure if isinstance(release_failure, asyncio.CancelledError) else None
                )
                reconciliation = await await_shielded_task_outcome(
                    asyncio.create_task(exact_recovery_claim_is_still_persisted()),
                    cancellation=release_cancellation,
                )
                cancellation = reconciliation.cancellation
                reconciliation_failure = reconciliation.error
                if isinstance(reconciliation_failure, asyncio.CancelledError) and (
                    cancellation is None
                ):
                    reconciliation_failure = unexpected_child_cancellation_error(
                        reconciliation_failure,
                        operation="Incomplete recovery claim release reconciliation",
                    )
                if reconciliation_failure is not None:
                    if cancellation is not None:
                        cancellation.add_note(
                            "Incomplete recovery claim release reconciliation also failed: "
                            f"{type(reconciliation_failure).__name__}."
                        )
                        if cancellation is release_failure:
                            _prepend_exception_cause(cancellation, reconciliation_failure)
                        else:
                            _prepend_exception_cause(
                                cancellation,
                                BaseExceptionGroup(
                                    "Incomplete recovery claim release failures",
                                    [release_failure, reconciliation_failure],
                                ),
                            )
                        restore_task_cancellation_requests(
                            reconciliation.cancellation_requests_consumed,
                            cancellation=cancellation,
                        )
                        raise cancellation from exception_cause(cancellation)
                    if not isinstance(reconciliation_failure, Exception):
                        raise reconciliation_failure from release_failure
                    release_failure.add_note(
                        "Incomplete recovery claim release reconciliation failed: "
                        f"{type(reconciliation_failure).__name__}."
                    )
                    raise release_failure from reconciliation_failure

                claim_release_completed = reconciliation.result is False
                if cancellation is not None:
                    if cancellation is not release_failure:
                        cancellation.add_note(
                            "Incomplete recovery claim release also failed: "
                            f"{type(release_failure).__name__}."
                        )
                        _prepend_exception_cause(cancellation, release_failure)
                    restore_task_cancellation_requests(
                        reconciliation.cancellation_requests_consumed,
                        cancellation=cancellation,
                    )
                    raise cancellation from exception_cause(cancellation)
                raise
            else:
                claim_release_completed = True

        async def release_recovery_presence() -> None:
            if claim_release_completed:
                # Worker quiescence and exact claim removal precede liveness
                # release. Do not let a retired heartbeat fence an immediate
                # retry or outlive the store that public recovery returns to.
                await self._session_control.execution_presence.stop_and_wait(
                    session_id, run_epoch=recovery_run_epoch
                )

        try:
            await self.run_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(
                    ("recovery worker settlement", authority.await_worker_settlement),
                    (
                        "run fence release",
                        release_recovery_run_fence,
                    ),
                    (
                        "incomplete recovery claim release",
                        release_recovery_claim,
                    ),
                    ("recovery execution presence release", release_recovery_presence),
                ),
            )
        finally:
            if claim_release_completed:
                # A successful exact marker removal confirms either release or
                # transfer to ordinary incomplete-session recovery. It remains
                # safe to retire this owner even if fence release itself failed.
                authority.finish_finalization()
            else:
                # The marker may still name this owner (including after a lost
                # acknowledgement). Keep finalization electable so this or a
                # waiting finalizer can reconcile and complete exact cleanup;
                # a durable successor may already have retired the old fence.
                authority.abort_finalization()

    async def load_owned_incomplete_recovery_claim(
        self,
        session_id: str,
        claim_id: str,
        *,
        expected_run_epoch: int | None,
    ) -> Session | None:
        """Return the session only while the exact claim and its epoch are owned."""
        owned = await self.load_owned_incomplete_recovery_claim_snapshot(
            session_id,
            claim_id,
            expected_run_epoch=expected_run_epoch,
            require_unexpired=False,
        )
        return None if owned is None else owned[0]

    async def load_owned_incomplete_recovery_claim_snapshot(
        self,
        session_id: str,
        claim_id: str,
        *,
        expected_run_epoch: int | None,
        require_unexpired: bool,
    ) -> tuple[Session, datetime] | None:
        """Return the exact durable owner and its latest lease expiry."""

        if expected_run_epoch is None:
            return None
        owned: tuple[Session, datetime] | None = None

        def inspect(
            session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> None:
            nonlocal owned
            persisted_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if persisted_claim is None or persisted_claim[0] != claim_id:
                return None
            if session.run_epoch != expected_run_epoch or (
                require_unexpired and persisted_claim[1] <= store_now
            ):
                return None
            owned = (session.model_copy(deep=True), persisted_claim[1])
            return None

        await self._session_store.transform_checkpoint_with_store_time(session_id, inspect)
        return owned

    async def claim(
        self,
        *,
        session: Session,
        inactive_for_seconds: int | None,
        required_expired_claim_id: str | None = None,
        execution_profile_snapshot: ActiveInvocationExecutionProfile | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
        target_status: SessionStatus | None = None,
    ) -> _IncompleteRecoveryClaim | None:
        if required_expired_claim_id is not None:
            required_expired_claim_id = require_clean_nonblank(
                required_expired_claim_id,
                "required_expired_claim_id",
            )
        replacing_expired_owner = required_expired_claim_id is not None
        operation_label = (
            "expired recovery takeover"
            if replacing_expired_owner
            else "incomplete-session recovery claim"
        )
        operation_label_title = (
            "Expired recovery takeover"
            if replacing_expired_owner
            else "Incomplete-session recovery claim"
        )
        rejection_note = (
            "Expired recovery takeover was rejected while cancellation was pending."
            if replacing_expired_owner
            else ("Incomplete-session recovery claim was rejected while cancellation was pending.")
        )
        claim_id = str(uuid4())
        claim_expires_at: datetime | None = None
        claim_run_epoch: int | None = None
        session_before_fence: Session | None = None
        authority: _IncompleteRecoveryClaimAuthority | None = None

        def claim_checkpoint(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            claimed_at: datetime,
        ) -> dict[str, Any]:
            nonlocal claim_expires_at, claim_run_epoch, session_before_fence
            _require_aware_datetime(claimed_at, "recovery claim clock")
            if (current_session.id, current_session.instance_id) != (
                session.id,
                session.instance_id,
            ):
                raise _IncompleteRecoveryClaimLost(
                    "Incomplete-session recovery target incarnation changed."
                )
            if (
                active_provider_operation_cancellation_claim_from_checkpoint(
                    checkpoint,
                    now=claimed_at,
                )
                is not None
            ):
                raise _IncompleteRecoveryClaimLost(
                    "Provider-operation cancellation still owns the session epoch."
                )

            existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if replacing_expired_owner:
                if (
                    existing is None
                    or existing[0] != required_expired_claim_id
                    or existing[1] > claimed_at
                ):
                    raise _IncompleteRecoveryClaimLost(
                        "Expired incomplete-session recovery ownership changed."
                    )
            elif current_session.status != session.status or (
                existing is not None and existing[1] > claimed_at
            ):
                raise _IncompleteRecoveryClaimLost(
                    "Incomplete-session recovery ownership changed before it was claimed."
                )

            claim_expires_at = claimed_at + _INCOMPLETE_RECOVERY_CLAIM_LEASE
            next_run_epoch = current_session.run_epoch + 1
            claim_run_epoch = next_run_epoch
            session_before_fence = current_session.model_copy(deep=True)
            updated = {} if checkpoint is None else copy_durable_record(checkpoint, "checkpoint")
            current_profile = active_invocation_execution_profile_from_checkpoint(updated)
            if execution_profile_snapshot is not None and (
                current_profile != execution_profile_snapshot
            ):
                raise _IncompleteRecoveryClaimLost(
                    "Active invocation profile changed before recovery takeover."
                    if replacing_expired_owner
                    else "Active invocation profile changed before recovery was claimed."
                )
            if current_profile is not None:
                updated = checkpoint_with_active_invocation_execution_profile(
                    updated,
                    session_id=current_session.id,
                    interaction_id=current_profile.interaction_id,
                    run_epoch=next_run_epoch,
                    profile=current_profile.profile,
                    expected=current_profile,
                )
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = {
                "version": 1,
                "claim_id": claim_id,
                "claimed_at": claimed_at.isoformat(),
                "claim_expires_at": claim_expires_at.isoformat(),
            }
            if not replacing_expired_owner and checkpoint_transform is not None:
                transformed = checkpoint_transform(current_session, updated)
                if transformed is None:
                    raise _IncompleteRecoveryClaimLost(
                        "Incomplete-session recovery checkpoint transfer deleted authority."
                    )
                updated = transformed
            return _checkpoint_with_rebased_session_run_operation(
                updated,
                previous_run_epoch=current_session.run_epoch,
                run_epoch=next_run_epoch,
            )

        try:
            fence_task = asyncio.create_task(
                self.reserve_and_fence_incomplete_recovery(
                    session.id,
                    statuses={session.status},
                    inactive_for_seconds=inactive_for_seconds,
                    checkpoint_transform=claim_checkpoint,
                    target_status=target_status,
                )
            )
            outcome = await await_shielded_task_outcome(fence_task)
            if isinstance(
                outcome.error,
                _IncompleteRecoveryClaimLost
                | InvocationLifecycleCommandConflict
                | SessionRunFenced
                | SessionStatusConflict,
            ):
                if outcome.cancellation is not None:
                    outcome.cancellation.add_note(rejection_note)
                    raise outcome.cancellation from outcome.error
                return None

            authoritative_failure = outcome.cancellation or outcome.error
            fenced = outcome.result
            if outcome.error is not None:
                reconciliation_outcome = await await_shielded_task_outcome(
                    asyncio.create_task(
                        self.load_owned_incomplete_recovery_claim(
                            session.id,
                            claim_id,
                            expected_run_epoch=claim_run_epoch,
                        )
                    ),
                    cancellation=outcome.cancellation,
                )
                reconciliation_cancellation = reconciliation_outcome.cancellation
                reconciliation_failure = reconciliation_outcome.error
                if reconciliation_cancellation is None and isinstance(
                    reconciliation_failure,
                    asyncio.CancelledError,
                ):
                    reconciliation_cancellation = reconciliation_failure
                authoritative_failure = (
                    reconciliation_cancellation or outcome.cancellation or outcome.error
                )
                if reconciliation_failure is not None:
                    if not isinstance(
                        reconciliation_failure,
                        Exception | asyncio.CancelledError,
                    ):
                        raise reconciliation_failure from outcome.error
                    authoritative_failure.add_note(
                        f"Could not reconcile whether the {operation_label} committed: "
                        f"{type(reconciliation_failure).__name__}."
                    )
                    if reconciliation_cancellation is not None:
                        reconciliation_cancellation.add_note(
                            f"{operation_label_title} also failed: {type(outcome.error).__name__}."
                        )
                        raise reconciliation_cancellation from outcome.error
                    raise outcome.error
                fenced = reconciliation_outcome.result
                if fenced is None:
                    if reconciliation_cancellation is not None:
                        reconciliation_cancellation.add_note(
                            f"{operation_label_title} also failed: {type(outcome.error).__name__}."
                        )
                        raise reconciliation_cancellation from outcome.error
                    raise outcome.error

            if fenced is None:
                raise RuntimeError(f"{operation_label_title} returned no session.")
            run_fence = _activate_owned_session_run_fence(fenced)
            authority = _IncompleteRecoveryClaimAuthority(
                session_id=fenced.id,
                claim_id=claim_id,
                run_fence=run_fence,
            )
            if (
                claim_expires_at is None
                or claim_run_epoch is None
                or session_before_fence is None
                or fenced.run_epoch != claim_run_epoch
            ):
                raise RuntimeError(
                    "Expired recovery takeover did not persist its claim."
                    if replacing_expired_owner
                    else ("Incomplete-session recovery claim was not persisted atomically.")
                )
            if authoritative_failure is not None:
                if outcome.cancellation is not None:
                    if outcome.error is not None:
                        outcome.cancellation.add_note(
                            f"{operation_label_title} also failed: {type(outcome.error).__name__}."
                        )
                    raise outcome.cancellation from outcome.error
                raise outcome.error

            try:
                renewal_started = time.monotonic()
                renewed_until = await self.renew_claim(
                    session.id,
                    claim_id,
                )
            except SessionRunFenced:
                await self.cleanup_claim(
                    authority=authority,
                    authoritative_failure=None,
                )
                return None
            if renewed_until is None:
                await self.cleanup_claim(
                    authority=authority,
                    authoritative_failure=None,
                )
                return None
            claim = _IncompleteRecoveryClaim(
                claim_id=claim_id,
                claim_expires_at=renewed_until,
                local_lease_deadline=(
                    renewal_started + _INCOMPLETE_RECOVERY_CLAIM_LEASE.total_seconds()
                ),
                session_before_fence=session_before_fence,
                session=fenced,
                authority=authority,
            )
            _require_live_incomplete_recovery_claim_acknowledgement(
                session_id=session.id,
                local_lease_deadline=claim.local_lease_deadline,
            )
            await self._session_control.execution_presence.ensure(fenced)
            return claim
        except BaseException as exc:
            if authority is not None:
                await self.cleanup_claim(
                    authority=authority,
                    authoritative_failure=exc,
                    execution_profile=(
                        None
                        if execution_profile_snapshot is None
                        else execution_profile_snapshot.profile
                    ),
                    claim_has_not_dispatched_work=True,
                )
            elif not replacing_expired_owner:
                await self.run_cleanup_steps(
                    authoritative_failure=exc,
                    steps=(
                        (
                            "incomplete recovery claim release",
                            lambda: self.release_claim(
                                session.id,
                                claim_id,
                            ),
                        ),
                    ),
                )
            raise

    async def run_with_heartbeat(
        self,
        *,
        claim: _IncompleteRecoveryClaim,
        recovery: Callable[[], Awaitable[_RecoveryResultT]],
    ) -> _RecoveryResultT:
        stop_heartbeat = asyncio.Event()
        recovery_outcome_observed = False
        shutdown_recovery_outcome_observed = False
        shutdown_recovery_failure: BaseException | None = None

        _require_live_incomplete_recovery_claim_acknowledgement(
            session_id=claim.session.id,
            local_lease_deadline=claim.local_lease_deadline,
        )

        async def run_live_recovery() -> _RecoveryResultT:
            # Recheck in the child at the exact dispatch boundary. Creating a
            # task can yield long enough to consume the remaining local lease.
            _require_live_incomplete_recovery_claim_acknowledgement(
                session_id=claim.session.id,
                local_lease_deadline=claim.local_lease_deadline,
            )
            return await recovery()

        async def run_recovery() -> CapturedAwaitableOutcome[_RecoveryResultT]:
            return await capture_awaitable_outcome(run_live_recovery)

        def recovery_outcome() -> _RecoveryResultT:
            nonlocal recovery_outcome_observed
            captured = recovery_task.result()
            recovery_outcome_observed = True
            if captured.error is not None:
                raise captured.error
            if captured.result is None:
                raise RuntimeError("Owned terminal operation returned no result.")
            return captured.result

        # A preclaimed terminal finalizer borrows its caller's invocation and
        # owns only its claim/work lifetime, not another run fence.
        workers = claim.authority or _RecoveryWorkerSettlement()
        worker_key = (claim.session.id, claim.claim_id)
        prior_workers = self._recovery_claim_workers.get(worker_key)
        if prior_workers is not None and not prior_workers.settled:
            raise RuntimeError("Recovery claim already has active workers.")
        recovery_task = asyncio.create_task(run_recovery())
        heartbeat_task = asyncio.create_task(
            self.heartbeat(
                session_id=claim.session.id,
                claim_id=claim.claim_id,
                local_lease_deadline=claim.local_lease_deadline,
                stop=stop_heartbeat,
            )
        )
        recovery_task.add_done_callback(lambda _completed: stop_heartbeat.set())
        workers.own_workers(recovery_task, heartbeat_task)
        self._recovery_claim_workers[worker_key] = workers
        authoritative_failure: BaseException | None = None

        async def stop_workers() -> None:
            nonlocal recovery_outcome_observed
            nonlocal shutdown_recovery_failure, shutdown_recovery_outcome_observed
            if (
                isinstance(
                    authoritative_failure,
                    asyncio.CancelledError | _IncompleteRecoveryClaimLost,
                )
                and not recovery_task.done()
            ):
                # Cancellation-opaque work must reach natural settlement before
                # this process reports quiescence. On ownership loss the
                # heartbeat is already terminal, but cancelling an asyncio
                # wrapper would not stop an underlying thread or process.
                await await_shielded_task_outcome(recovery_task)
            stop_heartbeat.set()
            if not recovery_task.done():
                recovery_task.cancel()
            if not heartbeat_task.done() and not isinstance(
                authoritative_failure,
                asyncio.CancelledError,
            ):
                heartbeat_task.cancel()
            await asyncio.gather(recovery_task, heartbeat_task, return_exceptions=True)
            if recovery_outcome_observed or recovery_task.cancelled():
                return
            captured = recovery_task.result()
            if isinstance(authoritative_failure, _IncompleteRecoveryClaimLost):
                recovery_outcome_observed = True
                shutdown_recovery_outcome_observed = True
                shutdown_recovery_failure = captured.error
                return
            if captured.error is None:
                return
            if isinstance(captured.error, asyncio.CancelledError):
                secondary = exception_cause(captured.error)
                if secondary is not None:
                    raise secondary
                return
            raise captured.error

        try:
            done, _pending = await asyncio.wait(
                {recovery_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                heartbeat_failure = heartbeat_task.exception()
                if heartbeat_failure is not None:
                    raise heartbeat_failure
                if recovery_task not in done:
                    raise RuntimeError(
                        "Incomplete-session recovery claim heartbeat stopped unexpectedly."
                    )
            if recovery_task in done:
                try:
                    result = recovery_outcome()
                except BaseException as recovery_failure:
                    if heartbeat_task.done() and not heartbeat_task.cancelled():
                        heartbeat_failure = heartbeat_task.exception()
                        if (
                            heartbeat_failure is not None
                            and heartbeat_failure is not recovery_failure
                            and not _attach_exception_cause_preserving_graph(
                                recovery_failure,
                                heartbeat_failure,
                            )
                        ):
                            raise BaseExceptionGroup(
                                "Incomplete recovery and claim heartbeat failed concurrently.",
                                [recovery_failure, heartbeat_failure],
                            ) from None
                    raise
                stop_heartbeat.set()
                if not heartbeat_task.done():
                    await heartbeat_task
                return result
            raise RuntimeError("Incomplete-session recovery owner produced no outcome.")
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self.run_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(("incomplete recovery worker shutdown", stop_workers),),
            )
            if (
                isinstance(authoritative_failure, _IncompleteRecoveryClaimLost)
                and shutdown_recovery_outcome_observed
            ):
                selected_failure = _authoritative_recovery_ownership_failure(
                    shutdown_recovery_failure,
                    authoritative_failure,
                )
                if selected_failure is not authoritative_failure:
                    raise selected_failure

    def retain_detached_recovery_write(self, task: asyncio.Task[Any]) -> None:
        self._detached_recovery_writes.add(task)

        def settled(completed: asyncio.Task[Any]) -> None:
            self._detached_recovery_writes.discard(completed)
            _consume_incomplete_recovery_store_task(completed)

        task.add_done_callback(settled)

    async def heartbeat(
        self,
        *,
        session_id: str,
        claim_id: str,
        local_lease_deadline: float,
        stop: asyncio.Event,
    ) -> None:
        sleep_seconds = _INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS
        last_renewal_failure: BaseException | None = None
        while not stop.is_set():
            remaining = local_lease_deadline - time.monotonic()
            if remaining <= 0:
                failure = _IncompleteRecoveryClaimLost(
                    "Incomplete-session recovery stopped before its store lease could "
                    f"be renewed for session {session_id}."
                )
                if last_renewal_failure is None:
                    raise failure
                raise failure from last_renewal_failure
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=min(sleep_seconds, remaining),
                )
            except TimeoutError:
                pass
            else:
                return
            renewal_started = time.monotonic()
            renewal_task = asyncio.create_task(
                self.renew_claim(
                    session_id,
                    claim_id,
                )
            )
            try:
                outcome = await await_shielded_task_outcome(
                    renewal_task,
                    timeout_s=max(
                        0.0,
                        local_lease_deadline - time.monotonic(),
                    ),
                )
            except asyncio.CancelledError:
                self.retain_detached_recovery_write(renewal_task)
                raise
            if outcome.cancellation is not None:
                self.retain_detached_recovery_write(renewal_task)
                restore_task_cancellation_requests(
                    outcome.cancellation_requests_consumed,
                    cancellation=outcome.cancellation,
                )
                raise outcome.cancellation
            if outcome.timed_out:
                self.retain_detached_recovery_write(renewal_task)
                raise _IncompleteRecoveryClaimLost(
                    "Incomplete-session recovery could not confirm store lease renewal "
                    f"before its local deadline for session {session_id}."
                ) from last_renewal_failure
            if outcome.error is not None:
                if isinstance(outcome.error, asyncio.CancelledError):
                    raise _IncompleteRecoveryClaimLost(
                        "Incomplete-session recovery store lease renewal was cancelled "
                        f"without owner cancellation for session {session_id}."
                    ) from unexpected_child_cancellation_error(
                        outcome.error,
                        operation="Incomplete-session recovery store lease renewal",
                    )
                if not isinstance(outcome.error, Exception):
                    raise outcome.error
                last_renewal_failure = outcome.error
                # Elapsed monotonic time is only a conservative local stop
                # boundary. It cannot authorize takeover, but it prevents
                # this worker from continuing after the store-owned lease
                # could have expired while renewal was unavailable.
                remaining = local_lease_deadline - time.monotonic()
                if remaining <= 0:
                    raise _IncompleteRecoveryClaimLost(
                        "Incomplete-session recovery could not renew its store lease "
                        f"for session {session_id}."
                    ) from outcome.error
                sleep_seconds = min(
                    _INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_RETRY_SECONDS,
                    remaining,
                )
                continue
            renewed_until = outcome.result
            if renewed_until is None:
                raise _IncompleteRecoveryClaimLost(
                    f"Incomplete-session recovery claim lost for session {session_id}."
                ) from None
            last_renewal_failure = None
            local_lease_deadline = (
                renewal_started + _INCOMPLETE_RECOVERY_CLAIM_LEASE.total_seconds()
            )
            _require_live_incomplete_recovery_claim_acknowledgement(
                session_id=session_id,
                local_lease_deadline=local_lease_deadline,
            )
            sleep_seconds = _INCOMPLETE_RECOVERY_CLAIM_HEARTBEAT_INTERVAL_SECONDS

    async def watch_manual_recovery_interruption(
        self,
        *,
        session_id: str,
        interrupted_baseline_id: str | None,
        stop: asyncio.Event,
    ) -> bool:
        """Observe another worker's durable stop request while delivery is paused."""
        while not stop.is_set():
            session = await self.require_session(session_id)
            local_run_handles_interrupt = self._session_control.is_emitting_interrupted(
                session_id
            ) or (
                bool(self._session_control.active_runs(session_id))
                and (
                    self._session_control.is_interruption_request_active(session_id)
                    or self._session_control.interrupt_signalled(session_id)
                )
            )
            # Local operator dispatch owns cancellation of the active run. The
            # durable watcher must not race it with a second cancellation while
            # that run is publishing its terminal evidence.
            # Publication unregisters the execution task before its first await;
            # the local emission owner remains authoritative until publication
            # and its cleanup finish.
            if session.status == SessionStatus.INTERRUPTING and not local_run_handles_interrupt:
                return True
            if session.status == SessionStatus.INTERRUPTED and not local_run_handles_interrupt:
                latest_interrupted = await self._session_control.latest_interrupted_event(
                    session_id
                )
                if (
                    latest_interrupted is not None
                    and latest_interrupted.id != interrupted_baseline_id
                    and latest_interrupted.payload.get("interruption_type")
                    == _INTERRUPTION_TYPE_OPERATOR_REQUESTED
                ):
                    return True
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=_MANUAL_RECOVERY_INTERRUPT_POLL_INTERVAL_SECONDS,
                )
            except TimeoutError:
                continue
        return False

    async def renew_claim(
        self,
        session_id: str,
        claim_id: str,
    ) -> datetime | None:
        renewed_until: datetime | None = None

        def renew_claim(
            _session: Session,
            checkpoint: dict[str, Any] | None,
            now: datetime,
        ) -> dict[str, Any] | None:
            nonlocal renewed_until
            existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            _require_aware_datetime(now, "recovery claim clock")
            if existing is None or existing[0] != claim_id or existing[1] <= now:
                return None
            updated = copy_durable_record(checkpoint, "checkpoint")
            marker = copy_json_value(
                updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY],
                "incomplete_session_recovery_claim",
            )
            renewed_until = now + _INCOMPLETE_RECOVERY_CLAIM_LEASE
            marker["claim_expires_at"] = renewed_until.isoformat()
            marker["renewed_at"] = now.isoformat()
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = marker
            return updated

        await self._session_store.transform_checkpoint_with_store_time(
            session_id,
            renew_claim,
        )
        return renewed_until

    def owns_current_worker(self, session_id: str, claim_id: str) -> bool:
        """Positive in-process ownership, not authority inferred from a durable ID."""
        workers = self._recovery_claim_workers.get((session_id, claim_id))
        return workers is not None and workers.owns_current_worker()

    async def release_claim(
        self,
        session_id: str,
        claim_id: str,
    ) -> None:
        worker_key = (session_id, claim_id)
        workers = self._recovery_claim_workers.get(worker_key)
        if workers is not None:
            # Covers both the inner finalizer and outer stream/interrupt cleanup
            # callbacks. A bounded observer may leave before its work finishes;
            # none of those callbacks may remove the live worker's durable claim.
            await workers.await_worker_settlement()
        no_owned_claim = _IncompleteRecoveryClaimLost("Recovery claim is no longer retained.")

        def release_claim(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            if checkpoint is None:
                raise no_owned_claim
            existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if existing is None or existing[0] != claim_id:
                raise no_owned_claim
            updated = copy_durable_record(checkpoint, "checkpoint")
            updated.pop(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY, None)
            return updated

        try:
            await self._session_store.transform_checkpoint(session_id, release_claim)
        except _IncompleteRecoveryClaimLost as failure:
            if failure is not no_owned_claim:
                raise
            # Returning an unchanged checkpoint can still update last_activity_at.
            # A stale owner must abort the transaction, not touch its successor.
        if self._recovery_claim_workers.get(worker_key) is workers:
            self._recovery_claim_workers.pop(worker_key, None)

    async def require_session(self, session_id: str) -> Session:
        loaded = await self._session_store.load(session_id)
        if loaded is None:
            raise KeyError(f"Session not found: {session_id}") from None
        return loaded


def _checkpoint_without_active_incomplete_recovery_claim(
    checkpoint: dict[str, Any] | None,
    *,
    now: datetime,
) -> dict[str, Any] | None:
    """Reject live recovery ownership and remove an expired internal marker."""
    _require_aware_datetime(now, "now")
    if checkpoint is None:
        return None
    updated = copy_durable_record(checkpoint, "checkpoint")
    existing = _incomplete_recovery_claim_from_checkpoint(updated)
    if existing is None:
        return updated
    if existing[1] > now:
        raise RuntimeError("Session has an active incomplete-session recovery operation.")
    updated.pop(_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY, None)
    return updated
