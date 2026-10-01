"""Live terminal-evidence claim transfer and streamed finalization."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol, TypeVar, cast
from uuid import uuid4

from cayu._exception_groups import (
    _attach_exception_cause_preserving_graph,
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
from cayu.events import Event
from cayu.runtime._interruption_coordinator import _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY
from cayu.runtime._recovery_claims import (
    _IncompleteRecoveryClaim,
    _IncompleteRecoveryClaimLost,
    _require_live_incomplete_recovery_claim_acknowledgement,
)
from cayu.runtime._run_limits import SessionUsageTracker
from cayu.runtime._session_control import SessionControl
from cayu.runtime._terminal_evidence import (
    interruption_request_id_from_payload,
    require_interruption_event_matches_pending_marker,
)
from cayu.sessions.base import (
    _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY,
    Session,
    SessionRuntimePublicationConflict,
    SessionStatus,
    SessionStore,
    _incomplete_recovery_claim_from_checkpoint,
)
from cayu.sessions.cleanup import RecoveryCleanupStepInput

_RecoveryResultT = TypeVar("_RecoveryResultT")
_TERMINAL_FINALIZATION_PROCESS_CONTROL_SIGNALS = (GeneratorExit, KeyboardInterrupt, SystemExit)


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


class _RecoveryClaimOperations(Protocol):
    """Use the existing recovery supervisor and registry for borrowed claims."""

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
    """Transfer and settle live terminal publication under shared recovery claims.

    Recovery retains the claim supervisor, cleanup supervisor and worker registry.
    This owner never acquires an invocation fence or creates a second registry.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        session_control: SessionControl[SessionUsageTracker],
        recovery: _RecoveryClaimOperations,
    ) -> None:
        self._session_store = session_store
        self._session_control = session_control
        self._recovery = recovery

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
