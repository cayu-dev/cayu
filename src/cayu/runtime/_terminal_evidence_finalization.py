"""Terminal-evidence inspection, crash repair and live finalization."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol, TypeVar, cast
from uuid import UUID, uuid4, uuid5

from cayu._exception_groups import (
    _attach_exception_cause_preserving_graph,
    add_exception_note_safely,
    exception_cause,
    exception_group_children,
    iter_exception_tree,
)
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import copy_durable_record, copy_json_value
from cayu.approvals.user_input import (
    AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY,
    USER_INPUT_SUPERSESSION_INTENT_KEY,
    AmbiguousUserInputSupersessionIntent,
    UserInputPauseState,
    UserInputSupersessionIntent,
    event_with_ambiguous_user_input_supersession_authority,
    event_with_user_input_supersession_authority,
    pending_user_input_interruption_payload,
    user_input_lifecycle_authority_from_checkpoint,
)
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.providers._credential_boundary import copy_provider_cancellation_failures
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime._approval_support import _pending_approval_and_round_for_atomic_claim
from cayu.runtime._event_writer import RuntimeEventWriter, _reconcile_exact_persisted_event
from cayu.runtime._interruption_coordinator import _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY
from cayu.runtime._recovery_claims import (
    _IncompleteRecoveryClaim,
    _IncompleteRecoveryClaimLost,
    _require_live_incomplete_recovery_claim_acknowledgement,
)
from cayu.runtime._run_limits import SessionUsageTracker
from cayu.runtime._session_control import SessionControl
from cayu.runtime._terminal_evidence import (
    TERMINAL_EVIDENCE_EVENT_TYPES,
    TERMINAL_EVIDENCE_QUERY_LIMIT,
    classify_current_terminal_evidence,
    interruption_request_id_from_payload,
    require_interruption_event_matches_pending_marker,
)
from cayu.runtime._tool_completion import recorded_terminal_tool_completion_payload
from cayu.runtime.execution_profiles import event_with_execution_profile_fingerprint_authority
from cayu.sessions.base import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    EventOrder,
    EventQuery,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryResult,
    Session,
    SessionRuntimePublicationConflict,
    SessionStatus,
    SessionStore,
    _checkpoint_after_session_run_operation_cleanup,
    _event_with_session_run_operation,
    _incomplete_recovery_claim_from_checkpoint,
    _session_run_operation_from_checkpoint,
    _SessionRunOperation,
)
from cayu.sessions.cleanup import RecoveryCleanupStepInput
from cayu.vaults.redaction import SecretRedactor

_RecoveryResultT = TypeVar("_RecoveryResultT")
_TERMINAL_FINALIZATION_PROCESS_CONTROL_SIGNALS = (GeneratorExit, KeyboardInterrupt, SystemExit)
_INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED = "tool_approval_required"
_INTERRUPTION_TYPE_USER_INPUT_REQUIRED = "user_input_required"
_INTERRUPTION_TYPE_RUNTIME_INTERRUPTED = "runtime_interrupted"
_INTERRUPTION_TYPE_OPERATOR_REQUESTED = "operator_requested"
_TERMINAL_EVIDENCE_REPAIR_NAMESPACE = UUID("bd021bef-ec8f-4e1e-950d-734e2c9ac513")
_TERMINAL_EVENT_TYPE_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}


def _terminal_finalization_process_control(
    error: BaseException | None,
) -> BaseException | None:
    """Return the first scalar process-control signal in transfer evidence."""

    if error is None:
        return None
    return next(
        (
            candidate
            for candidate in iter_exception_tree(error)
            if not isinstance(candidate, BaseExceptionGroup)
            and isinstance(candidate, _TERMINAL_FINALIZATION_PROCESS_CONTROL_SIGNALS)
        ),
        None,
    )


def _terminal_finalization_failure_without_identity(
    error: BaseException,
    excluded: BaseException,
    *,
    remaining_nodes: list[int] | None = None,
    visited: set[int] | None = None,
) -> BaseException | None:
    """Retain ordered transfer evidence without duplicating its public signal."""

    if error is excluded:
        return None
    if remaining_nodes is None:
        remaining_nodes = [128]
    if visited is None:
        visited = set()
    if remaining_nodes[0] < 1:
        return RuntimeError("Additional terminal finalization failures were omitted.")
    remaining_nodes[0] -= 1
    if not isinstance(error, BaseExceptionGroup):
        return error
    error_id = id(error)
    if error_id in visited:
        return RuntimeError("Cyclic terminal finalization failure evidence was omitted.")
    visited.add(error_id)
    children = exception_group_children(error)
    if children is None:
        return RuntimeError("Invalid terminal finalization failure evidence was omitted.")
    retained = [
        child_without_signal
        for child in children
        if (
            child_without_signal := _terminal_finalization_failure_without_identity(
                child,
                excluded,
                remaining_nodes=remaining_nodes,
                visited=visited,
            )
        )
        is not None
    ]
    if not retained:
        return None
    return BaseExceptionGroup(
        "Terminal finalization claim transfer retained additional failures.",
        retained,
    )


@dataclass(frozen=True)
class _TerminalFinalizationClaimAcquisition:
    claim_id: str
    claim_expires_at: datetime
    cancellation: asyncio.CancelledError | None = None
    transfer_failure: BaseException | None = None
    process_control: BaseException | None = None


def _provider_cancellation_interrupt_payload(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return one exact reconstructed provider-cancellation interrupt marker."""

    if checkpoint is None:
        return None
    marker = checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
    if marker is None:
        return None
    if type(marker) is not dict:
        raise ValueError("Pending session interrupt checkpoint must be an object.")
    payload = copy_json_value(marker, "pending_session_interrupt")
    failures = payload.get("provider_cancellation_failures")
    if failures is None:
        return None
    copied_failures = copy_provider_cancellation_failures(failures)
    if not copied_failures:
        raise ValueError("Provider cancellation interruption diagnostics cannot be empty.")
    interruption_type = payload.get("interruption_type")
    if type(interruption_type) is not str or interruption_type not in (
        _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
        _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
    ):
        raise ValueError("Provider cancellation interruption type is invalid.")
    if interruption_request_id_from_payload(payload) is None:
        raise ValueError("Provider cancellation interruption request identity is invalid.")
    payload["provider_cancellation_failures"] = [dict(item) for item in copied_failures]
    return payload


@dataclass(frozen=True)
class _TerminalEvidenceInspection:
    event: Event | None
    pending_interrupt_payload: dict[str, Any] | None
    pending_action_interrupt_payload: dict[str, Any] | None
    run_operation: _SessionRunOperation | None
    terminal_event_required: bool


class _FinalizationRecoveryOperations(Protocol):
    """Reuse recovery supervision, session loading and user-input authority."""

    async def _require_session(self, session_id: str) -> Session: ...

    async def _validated_user_input_supersession_interrupt_payload(
        self,
        *,
        session: Session,
        pending_interrupt_payload: dict[str, Any],
    ) -> dict[str, Any] | None: ...

    async def _classify_user_input_pause(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        input_id: str,
    ) -> UserInputPauseState: ...

    @property
    def claim_lease_duration(self) -> timedelta: ...

    async def _heartbeat_incomplete_recovery_claim(
        self,
        *,
        session_id: str,
        claim_id: str,
        local_lease_deadline: float,
        stop: asyncio.Event,
    ) -> None: ...

    async def _recover_incomplete_session_with_heartbeat(
        self,
        *,
        claim: _IncompleteRecoveryClaim,
        recovery: Callable[[], Awaitable[_RecoveryResultT]],
    ) -> _RecoveryResultT: ...

    async def _release_incomplete_recovery_claim(
        self,
        session_id: str,
        claim_id: str,
    ) -> None: ...

    async def _run_cleanup_steps(
        self,
        *,
        authoritative_failure: BaseException | None,
        steps: tuple[RecoveryCleanupStepInput, ...],
        cancellation_baseline: int = 0,
    ) -> tuple[tuple[str, BaseException], ...]: ...


class TerminalEvidenceFinalization:
    """Inspect, repair and settle terminal publication under shared recovery claims.

    Recovery retains the claim supervisor, cleanup supervisor and worker registry.
    This owner never acquires an invocation fence or creates a second registry.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        session_control: SessionControl[SessionUsageTracker],
        recovery: _FinalizationRecoveryOperations,
        event_writer: RuntimeEventWriter,
        secret_redactor: SecretRedactor,
        logger: logging.Logger,
    ) -> None:
        self._session_store = session_store
        self._session_control = session_control
        self._recovery = recovery
        self._event_writer = event_writer
        self._secret_redactor = secret_redactor
        self._logger = logger

    def new_claim_id(
        self,
    ) -> str:
        """Create an identity whose lease is installed only by authoritative store time."""

        return str(uuid4())

    async def claim_pending(
        self,
        *,
        session: Session,
        expected_payload: dict[str, Any],
    ) -> _TerminalFinalizationClaimAcquisition | Event | None:
        """Claim pending publication, or return its exact already-settled event."""

        expected_payload = copy_json_value(
            expected_payload,
            "expected_pending_session_interrupt",
        )
        claim_id = self.new_claim_id()
        claim_expires_at: datetime | None = None
        claim_installed = False

        def require_exact_pending_authority(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> None:
            if (
                current_session.id != session.id
                or current_session.instance_id != session.instance_id
                or current_session.status is not session.status
                or current_session.run_epoch != session.run_epoch
            ):
                raise SessionRuntimePublicationConflict(
                    "Terminal finalization session authority changed before ownership transfer."
                )
            current_payload = (
                None
                if checkpoint is None
                else checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            )
            if current_payload != expected_payload:
                raise SessionRuntimePublicationConflict(
                    "Terminal finalization interrupt identity changed before ownership transfer."
                )

        def claim_pending_finalization(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> dict[str, Any] | None:
            nonlocal claim_expires_at, claim_installed
            require_exact_pending_authority(current_session, checkpoint)
            existing_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if existing_claim is not None and existing_claim[1] > store_now:
                return None
            assert checkpoint is not None
            updated = copy_durable_record(checkpoint, "checkpoint")
            claim_expires_at = store_now + self._recovery.claim_lease_duration
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = copy_json_value(
                {
                    "version": 1,
                    "claim_id": claim_id,
                    "claimed_at": store_now.isoformat(),
                    "claim_expires_at": claim_expires_at.isoformat(),
                },
                "terminal_finalization_claim",
            )
            claim_installed = True
            return updated

        claim_task = asyncio.create_task(
            capture_awaitable_outcome(
                lambda: self._session_store.transform_checkpoint_with_store_time(
                    session.id,
                    claim_pending_finalization,
                )
            )
        )
        outcome = await await_shielded_task_outcome(claim_task)
        error = outcome.error
        if error is None:
            captured = outcome.result
            if type(captured) is not CapturedAwaitableOutcome:
                error = RuntimeError(
                    "Terminal evidence finalization claim transfer returned an invalid outcome."
                )
            else:
                error = captured.error
        if isinstance(error, asyncio.CancelledError) and outcome.cancellation is None:
            error = unexpected_child_cancellation_error(
                error,
                operation="Terminal evidence finalization claim transfer",
            )
        cancellation = outcome.cancellation
        if isinstance(error, SessionRuntimePublicationConflict):
            if cancellation is not None:
                raise cancellation from error
            # The live owner can finish between the caller's pending-state
            # read and this atomic claim. Never relax the mutation comparison:
            # only exact, already-published evidence permits joining instead.
            # Finalization may advance the epoch; this result grants no claim
            # and must be returned without attempting another repair mutation.
            current = await self._session_store.load(session.id)
            request_id = interruption_request_id_from_payload(expected_payload)
            if (
                current is not None
                and current.instance_id == session.instance_id
                and current.run_epoch >= session.run_epoch
                and current.status is SessionStatus.INTERRUPTED
                and request_id is not None
            ):
                event = await self._session_control.latest_interrupted_event(
                    session.id, interruption_request_id=request_id
                )
                if event is not None:
                    require_interruption_event_matches_pending_marker(event, expected_payload)
                    return event
            raise error

        async def reconcile_claim() -> bool:
            claim_matches = False

            def inspect_claim(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
                store_now: datetime,
            ) -> None:
                nonlocal claim_expires_at, claim_matches
                require_exact_pending_authority(current_session, checkpoint)
                current_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
                claim_matches = (
                    current_claim is not None
                    and current_claim[0] == claim_id
                    and current_claim[1] > store_now
                )
                if claim_matches:
                    assert current_claim is not None
                    claim_expires_at = current_claim[1]
                return None

            await self._session_store.transform_checkpoint_with_store_time(
                session.id, inspect_claim
            )
            return claim_matches

        if error is not None or cancellation is not None:
            reconciliation = await await_shielded_task_outcome(
                asyncio.create_task(reconcile_claim()),
                cancellation=cancellation,
            )
            cancellation = reconciliation.cancellation or cancellation
            reconciliation_error = reconciliation.error
            if isinstance(reconciliation_error, asyncio.CancelledError) and (
                reconciliation.cancellation is None
            ):
                reconciliation_error = unexpected_child_cancellation_error(
                    reconciliation_error,
                    operation="Terminal evidence finalization claim reconciliation",
                )
            if reconciliation_error is not None:
                if cancellation is not None:
                    cancellation.add_note(
                        "Terminal finalization claim reconciliation also failed: "
                        f"{type(reconciliation_error).__name__}."
                    )
                    if error is not None:
                        raise cancellation from BaseExceptionGroup(
                            "Terminal finalization claim transfer failures",
                            [error, reconciliation_error],
                        )
                    raise cancellation from reconciliation_error
                if error is not None:
                    raise BaseExceptionGroup(
                        "Terminal finalization claim transfer and reconciliation failed.",
                        [error, reconciliation_error],
                    ) from None
                raise reconciliation_error from error
            claim_installed = reconciliation.result is True

        if cancellation is not None:
            if claim_installed:
                assert claim_expires_at is not None
                process_control = _terminal_finalization_process_control(error)
                return _TerminalFinalizationClaimAcquisition(
                    claim_id=claim_id,
                    claim_expires_at=claim_expires_at,
                    cancellation=cancellation,
                    transfer_failure=(
                        error
                        if process_control is None or error is None
                        else _terminal_finalization_failure_without_identity(
                            error,
                            process_control,
                        )
                    ),
                    process_control=process_control,
                )
            if error is not None:
                raise cancellation from error
            raise cancellation
        if error is not None and not claim_installed:
            raise error
        if not claim_installed:
            return None
        assert claim_expires_at is not None
        process_control = _terminal_finalization_process_control(error)
        return _TerminalFinalizationClaimAcquisition(
            claim_id=claim_id,
            claim_expires_at=claim_expires_at,
            transfer_failure=(
                None
                if process_control is None or error is None
                else _terminal_finalization_failure_without_identity(
                    error,
                    process_control,
                )
            ),
            process_control=process_control,
        )

    def start_heartbeat(
        self,
        *,
        session_id: str,
        claim_id: str,
        local_lease_deadline: float,
    ) -> tuple[asyncio.Event, asyncio.Task[None]]:
        """Retain a live claim from atomic interrupt until its run handler owns it."""

        stop = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._recovery._heartbeat_incomplete_recovery_claim(
                session_id=session_id,
                claim_id=claim_id,
                local_lease_deadline=local_lease_deadline,
                stop=stop,
            ),
            name=f"cayu-terminal-finalization-heartbeat:{session_id}",
        )
        return stop, heartbeat

    async def await_operation(
        self,
        *,
        heartbeat_task: asyncio.Task[None],
        operation: Callable[[], Awaitable[_RecoveryResultT]],
        operation_name: str,
    ) -> _RecoveryResultT:
        """Run one pre-finalization operation only while its keeper is live."""

        operation_task = asyncio.create_task(capture_awaitable_outcome(operation))
        try:
            done, _pending = await asyncio.wait(
                {operation_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException as caller_control:
            if not operation_task.done():
                operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            operation_failure: BaseException | None = None
            if not operation_task.cancelled():
                captured = operation_task.result()
                if not isinstance(captured.error, asyncio.CancelledError):
                    operation_failure = captured.error
            if (
                operation_failure is not None
                and operation_failure is not caller_control
                and not _attach_exception_cause_preserving_graph(
                    caller_control,
                    operation_failure,
                )
            ):
                raise BaseExceptionGroup(
                    f"{operation_name} and caller control failed concurrently.",
                    [caller_control, operation_failure],
                ) from None
            raise
        if heartbeat_task in done:
            try:
                heartbeat_failure = heartbeat_task.exception()
            except asyncio.CancelledError as cancellation:
                heartbeat_failure = unexpected_child_cancellation_error(
                    cancellation,
                    operation="Terminal finalization claim heartbeat",
                )
            if heartbeat_failure is None:
                heartbeat_failure = RuntimeError(
                    "Terminal finalization claim heartbeat stopped unexpectedly."
                )
            if not operation_task.done():
                operation_task.cancel()
            await asyncio.gather(operation_task, return_exceptions=True)
            operation_failure: BaseException | None = None
            if not operation_task.cancelled():
                captured = operation_task.result()
                if not isinstance(captured.error, asyncio.CancelledError):
                    operation_failure = captured.error
            if operation_failure is not None and operation_failure is not heartbeat_failure:
                raise heartbeat_failure from operation_failure
            raise heartbeat_failure
        captured = operation_task.result()
        if captured.error is not None:
            raise captured.error
        return cast("_RecoveryResultT", captured.result)

    async def renew_claim(
        self,
        *,
        session: Session,
        claim_id: str,
        expected_payload: dict[str, Any],
    ) -> tuple[Session, datetime, float] | None:
        """Atomically re-prove and renew the complete terminal owner tuple."""

        expected_payload = copy_json_value(
            expected_payload,
            "expected_pending_session_interrupt",
        )
        renewed: tuple[Session, datetime] | None = None

        def renew_exact_claim(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> dict[str, Any] | None:
            nonlocal renewed
            if (
                current_session.id != session.id
                or current_session.instance_id != session.instance_id
                or current_session.status is not session.status
                or current_session.run_epoch != session.run_epoch
            ):
                raise SessionRuntimePublicationConflict(
                    "Terminal finalization session authority changed before lease renewal."
                )
            current_payload = (
                None
                if checkpoint is None
                else checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            )
            if current_payload != expected_payload:
                raise SessionRuntimePublicationConflict(
                    "Terminal finalization interrupt identity changed before lease renewal."
                )
            existing = _incomplete_recovery_claim_from_checkpoint(checkpoint)
            if existing is None or existing[0] != claim_id or existing[1] <= store_now:
                return None
            assert checkpoint is not None
            updated = copy_durable_record(checkpoint, "checkpoint")
            marker = copy_json_value(
                updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY],
                "terminal_finalization_claim",
            )
            renewed_until = store_now + self._recovery.claim_lease_duration
            marker["claim_expires_at"] = renewed_until.isoformat()
            marker["renewed_at"] = store_now.isoformat()
            updated[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY] = marker
            renewed = (current_session.model_copy(deep=True), renewed_until)
            return updated

        renewal_started = time.monotonic()
        await self._session_store.transform_checkpoint_with_store_time(
            session.id, renew_exact_claim
        )
        if renewed is None:
            return None
        local_lease_deadline = renewal_started + self._recovery.claim_lease_duration.total_seconds()
        _require_live_incomplete_recovery_claim_acknowledgement(
            session_id=session.id,
            local_lease_deadline=local_lease_deadline,
        )
        return renewed[0], renewed[1], local_lease_deadline

    async def _run_claimed(
        self,
        *,
        session: Session,
        claim_id: str,
        expected_payload: dict[str, Any],
        finalization: Callable[[], Awaitable[_RecoveryResultT]],
    ) -> _RecoveryResultT:
        """Run the live finalizer under the same lease used by crash recovery."""

        owned_claim = await self.renew_claim(
            session=session,
            claim_id=claim_id,
            expected_payload=expected_payload,
        )
        if owned_claim is None:
            raise _IncompleteRecoveryClaimLost(
                "Terminal evidence finalization ownership changed before execution."
            )
        owned_session, current_claim_expires_at, local_lease_deadline = owned_claim
        claim = _IncompleteRecoveryClaim(
            claim_id=claim_id,
            claim_expires_at=current_claim_expires_at,
            local_lease_deadline=local_lease_deadline,
            session_before_fence=owned_session,
            session=owned_session,
        )
        authoritative_failure: BaseException | None = None
        try:
            return await self._recovery._recover_incomplete_session_with_heartbeat(
                claim=claim,
                recovery=finalization,
            )
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            await self._recovery._run_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(
                    (
                        "terminal evidence finalization claim release",
                        lambda: self._recovery._release_incomplete_recovery_claim(
                            session.id,
                            claim_id,
                        ),
                    ),
                ),
            )

    async def stream(
        self,
        *,
        session: Session,
        claim_id: str,
        expected_payload: dict[str, Any],
        finalization: AsyncIterator[_RecoveryResultT],
    ) -> AsyncGenerator[_RecoveryResultT, None]:
        """Stream a live finalizer while retaining its durable lease."""

        events: asyncio.Queue[_RecoveryResultT] = asyncio.Queue(maxsize=1)
        observer_closed = False

        async def collect_finalization() -> bool:
            try:
                async for item in finalization:
                    if not observer_closed:
                        await events.put(item)
                return True
            finally:
                close = getattr(finalization, "aclose", None)
                if close is not None:
                    await close()

        async def run_owned_finalization() -> CapturedAwaitableOutcome[bool]:
            return await capture_awaitable_outcome(
                lambda: self._run_claimed(
                    session=session,
                    claim_id=claim_id,
                    expected_payload=expected_payload,
                    finalization=collect_finalization,
                )
            )

        owner = asyncio.create_task(run_owned_finalization())
        owner_outcome_observed = False
        pending_get: asyncio.Task[_RecoveryResultT] | None = None

        def require_owner_outcome() -> None:
            nonlocal owner_outcome_observed
            captured = owner.result()
            owner_outcome_observed = True
            if captured.error is not None:
                raise captured.error
            if captured.result is not True:
                raise RuntimeError("Owned terminal stream returned no completion result.")

        async def stop_owner() -> None:
            if pending_get is not None and not pending_get.done():
                pending_get.cancel()
            if not owner.done():
                owner.cancel()
            await asyncio.gather(
                *(task for task in (pending_get, owner) if task is not None),
                return_exceptions=True,
            )
            if owner_outcome_observed or owner.cancelled():
                return
            captured = owner.result()
            if captured.error is None:
                return
            if isinstance(captured.error, asyncio.CancelledError):
                secondary = exception_cause(captured.error)
                if secondary is not None:
                    raise secondary
                return
            raise captured.error

        authoritative_failure: BaseException | None = None
        try:
            while True:
                if not events.empty():
                    yield events.get_nowait()
                    continue
                if owner.done():
                    require_owner_outcome()
                    return
                pending_get = asyncio.create_task(events.get())
                done, _pending = await asyncio.wait(
                    {pending_get, owner},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if pending_get in done:
                    item = pending_get.result()
                    pending_get = None
                    yield item
                    continue
                pending_get.cancel()
                await asyncio.gather(pending_get, return_exceptions=True)
                pending_get = None
                require_owner_outcome()
        except BaseException as exc:
            authoritative_failure = exc
            raise
        finally:
            # Delivery backpressure belongs to the observer, not the durable
            # finalizer. Once abandoned, drain the one bounded slot to unblock
            # an already-waiting put; later items must not queue for a consumer
            # that no longer exists. Native publication/cleanup still runs under
            # the retained worker and claim until positive settlement.
            observer_closed = True
            while not events.empty():
                events.get_nowait()
            await self._recovery._run_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(("terminal evidence stream owner shutdown", stop_owner),),
            )

    async def repair_required(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> bool:
        inspection = await self.inspect(
            session=session,
            checkpoint=checkpoint,
        )
        return (
            (inspection.event is None and inspection.terminal_event_required)
            or inspection.pending_interrupt_payload is not None
            or inspection.run_operation is not None
            or (checkpoint is not None and _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY in checkpoint)
        )

    async def inspect(
        self,
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> _TerminalEvidenceInspection:
        expected_event_type = _TERMINAL_EVENT_TYPE_BY_STATUS.get(session.status)
        if expected_event_type is None:
            raise ValueError(f"Session is not terminal: {session.status}.")
        run_operation = _session_run_operation_from_checkpoint(checkpoint)

        pending_interrupt_payload: dict[str, Any] | None = None
        pending_interrupt_request_id: str | None = None
        if checkpoint is not None and _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY in checkpoint:
            marker = checkpoint[_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY]
            if type(marker) is not dict:
                raise ValueError("Pending session interrupt checkpoint must be an object.")
            pending_interrupt_payload = copy_json_value(marker, "pending_session_interrupt")
            provider_interrupt_payload = _provider_cancellation_interrupt_payload(checkpoint)
            if provider_interrupt_payload is not None:
                pending_interrupt_payload = provider_interrupt_payload
            if session.status != SessionStatus.INTERRUPTED:
                raise RuntimeError(
                    "Terminal evidence is contradictory: a non-interrupted session retains "
                    "a pending interruption marker."
                )
            pending_interrupt_request_id = interruption_request_id_from_payload(
                pending_interrupt_payload
            )
            if pending_interrupt_request_id is None:
                raise RuntimeError(
                    "Terminal evidence is not repairable: the pending interruption marker "
                    "has no stable request identity."
                )
            await self._recovery._validated_user_input_supersession_interrupt_payload(
                session=session,
                pending_interrupt_payload=pending_interrupt_payload,
            )

        pending_approval = approval_support.pending_approval_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
        )
        pending_user_input, _resolution_intent = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            current_run_epoch=session.run_epoch,
            runtime_session=session,
        )
        pending_tool_round = tool_round_recovery.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        approval_owns_tool_round = False
        if pending_approval is not None and pending_tool_round is not None:
            _pending_approval_and_round_for_atomic_claim(
                checkpoint,
                approval_id=pending_approval.approval_id,
                tool_round_id=pending_approval.tool_round_id,
                gating_tool_call_id=pending_approval.tool_call_id,
                redactor=self._secret_redactor,
                runtime_session=session,
            )
            approval_owns_tool_round = True
        pending_actions = tuple(
            action
            for action in (
                pending_approval,
                pending_user_input,
                None if approval_owns_tool_round else pending_tool_round,
            )
            if action is not None
        )
        if len(pending_actions) > 1:
            raise RuntimeError(
                "Terminal evidence is not repairable: the checkpoint contains "
                "conflicting pending actions."
            )
        if pending_user_input is not None:
            pause_state = await self._recovery._classify_user_input_pause(
                session=session,
                checkpoint=checkpoint,
                input_id=pending_user_input.input_id,
            )
            if pause_state not in {
                UserInputPauseState.ACTIVE,
                UserInputPauseState.ANSWERING,
            }:
                raise SessionRuntimePublicationConflict(
                    "Terminal user-input evidence has ambiguous pause authority."
                )
        pending_action_interrupt_payload: dict[str, Any] | None = None
        if pending_approval is not None and session.status == SessionStatus.INTERRUPTED:
            pending_action_interrupt_payload = {
                "interruption_type": _INTERRUPTION_TYPE_TOOL_APPROVAL_REQUIRED,
                "model_step_id": pending_approval.model_step_id,
                "model_attempt_id": pending_approval.model_attempt_id,
                "tool_round_id": pending_approval.tool_round_id,
                **approval_support.bounded_pending_approval_event_payload(
                    pending_approval,
                    redactor=self._secret_redactor,
                ),
            }
        elif pending_user_input is not None and session.status == SessionStatus.INTERRUPTED:
            pending_action_interrupt_payload = {
                "interruption_type": _INTERRUPTION_TYPE_USER_INPUT_REQUIRED,
                "model_step_id": pending_user_input.model_step_id,
                "model_attempt_id": pending_user_input.model_attempt_id,
                "tool_round_id": pending_user_input.tool_round_id,
                **pending_user_input_interruption_payload(pending_user_input),
            }
        elif pending_tool_round is not None and session.status == SessionStatus.INTERRUPTED:
            pending_action_interrupt_payload = {
                **tool_round_recovery.pending_tool_round_identity(pending_tool_round).payload(),
                "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                "reason": "terminal_event_evidence_repaired",
                "recovered": True,
            }

        evidence_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=TERMINAL_EVIDENCE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=TERMINAL_EVIDENCE_QUERY_LIMIT,
            )
        )
        classification = classify_current_terminal_evidence(
            evidence_events=tuple(record.event for record in evidence_records),
            expected_event_type=expected_event_type,
            run_operation_id=(None if run_operation is None else run_operation.operation_id),
            interruption_request_id=pending_interrupt_request_id,
        )
        terminal_events = classification.events
        if classification.run_operation_conflict:
            raise RuntimeError(
                "Terminal evidence is contradictory: the interruption event and "
                "pending run operation have different identities."
            )
        if any(event.type != expected_event_type for event in terminal_events):
            raise RuntimeError(
                "Terminal evidence is contradictory: the durable event type does not "
                f"match session status {session.status.value}."
            )
        if len(terminal_events) > 1:
            raise RuntimeError(
                "Terminal evidence is contradictory: more than one terminal event exists "
                "for the current run."
            )

        existing_event = None if not terminal_events else terminal_events[0].model_copy(deep=True)
        exact_interrupt_marker_retained = pending_interrupt_payload is not None and (
            "provider_cancellation_failures" in pending_interrupt_payload
            or pending_interrupt_payload.get("terminal_publication_repair") is True
            or USER_INPUT_SUPERSESSION_INTENT_KEY in pending_interrupt_payload
            or AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY in pending_interrupt_payload
        )
        if existing_event is not None and exact_interrupt_marker_retained:
            require_interruption_event_matches_pending_marker(
                existing_event,
                pending_interrupt_payload,
            )
        return _TerminalEvidenceInspection(
            event=existing_event,
            pending_interrupt_payload=pending_interrupt_payload,
            pending_action_interrupt_payload=pending_action_interrupt_payload,
            run_operation=run_operation,
            terminal_event_required=(
                run_operation is not None
                or pending_interrupt_payload is not None
                or pending_action_interrupt_payload is not None
                or classification.latest_lifecycle_event_type != EventType.SESSION_FORKED
            ),
        )

    async def repair(
        self,
        *,
        session: Session,
        terminal_run_epoch: int,
        terminal_timestamp: datetime,
        previous_status: SessionStatus,
        claim_id: str,
    ) -> IncompleteSessionRecoveryResult:
        checkpoint = await self._session_store.load_checkpoint(session.id)
        inspection = await self.inspect(
            session=session,
            checkpoint=checkpoint,
        )
        terminal_event = inspection.event
        if terminal_event is None and inspection.terminal_event_required:
            repair_event = self._repair_event(
                session=session,
                terminal_run_epoch=terminal_run_epoch,
                terminal_timestamp=terminal_timestamp,
                pending_interrupt_payload=inspection.pending_interrupt_payload,
                pending_action_interrupt_payload=(inspection.pending_action_interrupt_payload),
                run_operation=inspection.run_operation,
            )
            if session.status is SessionStatus.COMPLETED:
                repair_event.payload.update(
                    await recorded_terminal_tool_completion_payload(self._session_store, session)
                )
            terminal_event = await self._persist_repair_event(repair_event)
        if inspection.pending_interrupt_payload is not None:
            await self.clear_pending_interrupt(
                session_id=session.id,
                claim_id=claim_id,
                expected_payload=inspection.pending_interrupt_payload,
            )
        if inspection.run_operation is not None:
            await self.clear_run_operation(
                session_id=session.id,
                operation=inspection.run_operation,
                required_claim_id=claim_id,
                terminal_evidence_durable=True,
            )

        if terminal_event is not None:
            try:
                await self._event_writer.fan_out_persisted([terminal_event])
            except Exception as exc:
                self._logger.warning(
                    "Terminal evidence was repaired but durable side-effect delivery remains "
                    "pending: session_id=%s event_id=%s error_type=%s",
                    session.id,
                    terminal_event.id,
                    type(exc).__name__,
                )

        current = await self._recovery._require_session(session.id)
        if current.status != session.status or current.run_epoch != session.run_epoch:
            raise RuntimeError("Terminal session changed while its evidence was repaired.")
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=current.status,
            actions=(IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_EVIDENCE,),
            events=(() if terminal_event is None else (terminal_event,)),
            message=(
                "Reconciled durable terminal evidence."
                if terminal_event is None
                else "Repaired durable terminal event evidence."
            ),
        )

    def _repair_event(
        self,
        *,
        session: Session,
        terminal_run_epoch: int,
        terminal_timestamp: datetime,
        pending_interrupt_payload: dict[str, Any] | None,
        pending_action_interrupt_payload: dict[str, Any] | None,
        run_operation: _SessionRunOperation | None,
    ) -> Event:
        event_type = _TERMINAL_EVENT_TYPE_BY_STATUS[session.status]
        pending_interrupt_request_id = (
            None
            if pending_interrupt_payload is None
            else interruption_request_id_from_payload(pending_interrupt_payload)
        )
        if pending_interrupt_request_id is not None:
            operation_identity = f"interrupt_request:{pending_interrupt_request_id}"
        elif run_operation is not None:
            operation_identity = run_operation.operation_id
        else:
            operation_identity = f"run_epoch:{terminal_run_epoch}"
        event_id = (
            run_operation.terminal_event_id
            if run_operation is not None and run_operation.terminal_event_id is not None
            else str(
                uuid5(
                    _TERMINAL_EVIDENCE_REPAIR_NAMESPACE,
                    f"{session.id}\0{operation_identity}\0{session.status.value}",
                )
            )
        )
        if session.status == SessionStatus.COMPLETED:
            payload: dict[str, Any] = {
                "recovered": True,
                "terminal_evidence_repaired": True,
            }
        elif session.status == SessionStatus.FAILED:
            payload = {
                "error": "Original terminal failure details were not durably recorded.",
                "error_type": "TerminalFailureEvidenceUnavailable",
                "recovered": True,
                "terminal_evidence_repaired": True,
            }
        elif pending_interrupt_payload is not None:
            payload = copy_json_value(
                pending_interrupt_payload,
                "pending_session_interrupt",
            )
        elif pending_action_interrupt_payload is not None:
            payload = copy_json_value(
                pending_action_interrupt_payload,
                "pending_action_interrupt",
            )
        else:
            payload = {
                "interruption_type": _INTERRUPTION_TYPE_RUNTIME_INTERRUPTED,
                "reason": "terminal_event_evidence_repaired",
                "recovered": True,
                "terminal_evidence_repaired": True,
            }
        event = event_with_runtime_envelope_authority(
            event_with_runtime_generated_id(
                Event(
                    id=event_id,
                    type=event_type,
                    session_id=session.id,
                    timestamp=terminal_timestamp,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                    payload=payload,
                )
            ),
            "session_id",
        )
        if type(payload.get("interruption_request_id")) is str:
            event = event_with_runtime_payload_authority(
                event,
                "interruption_request_id",
            )
        supersession_payload = payload.get(USER_INPUT_SUPERSESSION_INTENT_KEY)
        if supersession_payload is not None:
            try:
                supersession_intent = UserInputSupersessionIntent.model_validate(
                    supersession_payload
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError("User-input supersession evidence is malformed.") from exc
            event = event_with_user_input_supersession_authority(
                event,
                supersession_intent,
            )
        ambiguous_supersession_payload = payload.get(AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY)
        if ambiguous_supersession_payload is not None:
            try:
                ambiguous_supersession_intent = AmbiguousUserInputSupersessionIntent.model_validate(
                    ambiguous_supersession_payload
                )
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Ambiguous user-input supersession evidence is malformed."
                ) from exc
            event = event_with_ambiguous_user_input_supersession_authority(
                event,
                ambiguous_supersession_intent,
            )
        raw_profile_fingerprint = payload.get("execution_profile_fingerprint")
        if raw_profile_fingerprint is None:
            profile_fingerprint = None
        elif type(raw_profile_fingerprint) is str:
            profile_fingerprint = raw_profile_fingerprint
        else:
            raise TypeError("execution_profile_fingerprint must be a string or None.")
        event = event_with_execution_profile_fingerprint_authority(event, profile_fingerprint)
        return (
            event
            if run_operation is None
            else _event_with_session_run_operation(event, run_operation)
        )

    async def _persist_repair_event(self, event: Event) -> Event:
        # Freeze the publication-safe shape before the append attempt. The
        # writer may redact workload secrets, so acknowledgement-loss
        # reconciliation must compare durable evidence with this prepared
        # snapshot rather than with the raw checkpoint-derived payload.
        event = self._event_writer.prepare(event)
        try:
            return await self._event_writer.persist(event)
        except Exception as append_failure:
            try:
                records = await self._session_store.query_events(
                    EventQuery(
                        session_id=event.session_id,
                        event_id=event.id,
                        limit=1,
                    )
                )
            except Exception as reconciliation_failure:
                add_exception_note_safely(
                    append_failure,
                    "Terminal evidence append reconciliation failed: "
                    f"{type(reconciliation_failure).__name__}.",
                )
                raise ExceptionGroup(
                    "Terminal evidence append and reconciliation both failed.",
                    [append_failure, reconciliation_failure],
                ) from None
            try:
                persisted = _reconcile_exact_persisted_event(
                    event,
                    records,
                    conflict_message=(
                        "Terminal evidence repair event identity is already used by "
                        "different durable evidence."
                    ),
                )
            except RuntimeError as conflict:
                raise conflict from append_failure
            if persisted is None:
                raise
            return persisted

    async def clear_pending_interrupt(
        self,
        *,
        session_id: str,
        claim_id: str,
        expected_payload: dict[str, Any],
    ) -> None:
        def clear_marker(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> dict[str, Any]:
            if current_session.status != SessionStatus.INTERRUPTED:
                raise RuntimeError("Session status changed during terminal interruption repair.")
            if checkpoint is None:
                raise _IncompleteRecoveryClaimLost(
                    "Terminal evidence recovery checkpoint disappeared."
                )
            updated = copy_durable_record(checkpoint, "checkpoint")
            claim = _incomplete_recovery_claim_from_checkpoint(updated)
            if claim is None or claim[0] != claim_id or claim[1] <= store_now:
                raise _IncompleteRecoveryClaimLost(
                    "Terminal evidence recovery ownership changed before marker cleanup."
                )
            current_payload = updated.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            if current_payload != expected_payload:
                raise RuntimeError(
                    "Pending interruption identity changed during terminal evidence repair."
                )
            updated.pop(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            return updated

        await self._session_store.transform_checkpoint_with_store_time(session_id, clear_marker)

    async def clear_run_operation(
        self,
        *,
        session_id: str,
        operation: _SessionRunOperation,
        required_claim_id: str | None = None,
        terminal_evidence_durable: bool = False,
    ) -> None:
        def clear_operation(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            current_operation = _session_run_operation_from_checkpoint(checkpoint)
            if current_operation is None:
                return checkpoint
            if current_operation != operation:
                raise RuntimeError(
                    "Session run operation changed before terminal evidence cleanup."
                )
            if required_claim_id is not None:
                claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
                if claim is None or claim[0] != required_claim_id:
                    raise _IncompleteRecoveryClaimLost(
                        "Terminal evidence recovery ownership changed before run cleanup."
                    )
            return _checkpoint_after_session_run_operation_cleanup(
                checkpoint,
                operation=operation,
                retain_terminal_receipt=terminal_evidence_durable,
            )

        await self._session_store.transform_checkpoint(session_id, clear_operation)
