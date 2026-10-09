"""Scoped terminal-finalization lifetimes for request and interrupted-run owners."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from cayu._exception_groups import (
    _failure_without_existing_exception_identities,
    exception_cause,
    iter_exception_tree,
    set_exception_cause,
)
from cayu._task_wait import await_shielded_task_outcome
from cayu._validation import copy_json_value
from cayu.approvals.user_input import (
    AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY,
    USER_INPUT_SUPERSESSION_INTENT_KEY,
)
from cayu.events import Event
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime._interruption_coordinator import _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY
from cayu.runtime._recovery_claims import _IncompleteRecoveryClaimLost
from cayu.runtime._session_control import TerminalFinalizationClaimHandoff
from cayu.sessions._invocation_terminal_decision import InvocationTerminalDecision
from cayu.sessions._terminal_evidence import interruption_request_id_from_payload
from cayu.sessions.base import (
    SessionRuntimePublicationConflict,
    _incomplete_recovery_claim_from_checkpoint,
)
from cayu.sessions.records import Session

if TYPE_CHECKING:
    from cayu.runtime._terminal_evidence_finalization import TerminalEvidenceFinalization

_OperationResultT = TypeVar("_OperationResultT")


async def _joined_terminal_event(event: Event) -> AsyncGenerator[Event, None]:
    yield event


async def _await_stopped_heartbeat(
    heartbeat: asyncio.Task[None], *, superseded_by_exact_renewal: bool
) -> None:
    """Retain process control even when exact renewal supersedes a keeper failure."""
    if not superseded_by_exact_renewal:
        await heartbeat
        return
    result = await asyncio.gather(heartbeat, return_exceptions=True)
    failure = result[0]
    if isinstance(
        failure, (KeyboardInterrupt, SystemExit, GeneratorExit, BaseExceptionGroup)
    ) and not isinstance(failure, Exception):
        raise failure


def _raise_terminal_finalization_process_control(
    signal: BaseException,
    secondary_failures: Iterable[BaseException],
) -> NoReturn:
    """Redeliver committed claim-transfer control after exact settlement."""

    signal_graph_ids = {id(candidate) for candidate in iter_exception_tree(signal)}
    retained: list[BaseException] = []
    existing_cause = exception_cause(signal)
    if existing_cause is not None:
        retained.append(existing_cause)
    existing_ids = {
        id(candidate) for failure in retained for candidate in iter_exception_tree(failure)
    } | signal_graph_ids
    for failure in secondary_failures:
        retained_failure = _failure_without_existing_exception_identities(
            failure,
            existing_ids,
        )
        if retained_failure is None:
            continue
        retained.append(retained_failure)
        existing_ids.update(id(candidate) for candidate in iter_exception_tree(retained_failure))
    cause: BaseException | None
    if not retained:
        cause = None
    elif len(retained) == 1:
        cause = retained[0]
    else:
        cause = BaseExceptionGroup(
            "Terminal finalization process control retained settlement failures.",
            retained,
        )
    if not set_exception_cause(signal, cause):
        raise BaseExceptionGroup(
            "Terminal finalization process control and settlement failures.",
            [signal, *retained],
        ) from None
    raise signal from cause


class InterruptionFinalization:
    """Own preparation, handoff and settlement for one interruption request.

    The claim ID is also used by the engine's atomic interruption transition.
    A transferred handoff belongs to its receiving task; an unaccepted one retains
    durable authority for recovery after its preparation heartbeat stops.
    """

    def __init__(
        self,
        finalization: TerminalEvidenceFinalization,
        *,
        session_id: str,
        adopting_pending: bool,
    ) -> None:
        self._finalization = finalization
        self._session_store = finalization._session_store
        self._control = finalization._session_control
        self._recovery = finalization._recovery
        self._session_id = session_id
        self._adopting_pending = adopting_pending
        self._claim_id = None if adopting_pending else finalization.new_claim_id()
        self._claim_expires_at: datetime | None = None
        self._transfer_cancellation: asyncio.CancelledError | None = None
        self._transfer_failure: BaseException | None = None
        self._transfer_process_control: BaseException | None = None
        self._heartbeat_stop: asyncio.Event | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._retained_for_recovery = False
        self._handed_to_active_run = False
        self._preparation_cleaned = False
        self._request_marker_active = False

    @property
    def claim_id(self) -> str | None:
        return self._claim_id

    @property
    def transition_claim_id(self) -> str | None:
        return self._claim_id if self._heartbeat_task is not None else None

    @property
    def handed_to_active_run(self) -> bool:
        return self._handed_to_active_run

    def begin_request(self) -> None:
        self._control.begin_interruption_request(self._session_id)
        self._request_marker_active = True

    def end_request(self) -> None:
        if self._request_marker_active:
            self._request_marker_active = False
            self._control.end_interruption_request(self._session_id)

    async def claim_or_join(
        self, *, session: Session, expected_payload: dict[str, Any]
    ) -> Event | None:
        """Retain an exact claim, or join an already-published terminal event."""
        if self._claim_id is not None and self._claim_expires_at is not None:
            return None
        acquired = await self._finalization.claim_pending(
            session=session, expected_payload=expected_payload
        )
        if isinstance(acquired, Event):
            return acquired
        if acquired is None:
            return await self._finalization.repair_and_join(
                session=session, expected_payload=expected_payload
            )
        self._claim_id = acquired.claim_id
        self._claim_expires_at = acquired.claim_expires_at
        self._transfer_cancellation = acquired.cancellation
        self._transfer_failure = acquired.transfer_failure
        self._transfer_process_control = acquired.process_control
        return None

    async def prepare_dispatch(self, *, session: Session, interruption_request_id: str) -> Session:
        """Keep exact supersession ownership alive through dispatch and handoff."""
        if self._claim_id is not None:
            interrupt_checkpoint = await self._session_store.load_checkpoint(session.id)
            persisted_claim = _incomplete_recovery_claim_from_checkpoint(interrupt_checkpoint)
            if persisted_claim is not None and persisted_claim[0] == self._claim_id:
                self._claim_expires_at = persisted_claim[1]
            persisted_payload = (
                None
                if interrupt_checkpoint is None
                else interrupt_checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
            )
            claim_owns_user_input_supersession = (
                persisted_claim is not None
                and persisted_claim[0] == self._claim_id
                and type(persisted_payload) is dict
                and (
                    USER_INPUT_SUPERSESSION_INTENT_KEY in persisted_payload
                    or AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY in persisted_payload
                )
                and interruption_request_id_from_payload(persisted_payload)
                == interruption_request_id
            )
            if claim_owns_user_input_supersession:
                assert type(persisted_payload) is dict
                renewed_claim = await self._finalization.renew_claim(
                    session=session,
                    claim_id=self._claim_id,
                    expected_payload=persisted_payload,
                )
                if renewed_claim is None:
                    self._retained_for_recovery = True
                    raise _IncompleteRecoveryClaimLost(
                        "Terminal finalization ownership expired before live dispatch."
                    )
                (
                    session,
                    self._claim_expires_at,
                    terminal_finalization_local_lease_deadline,
                ) = renewed_claim
                (
                    self._heartbeat_stop,
                    self._heartbeat_task,
                ) = self._finalization.start_heartbeat(
                    session_id=session.id,
                    claim_id=self._claim_id,
                    local_lease_deadline=terminal_finalization_local_lease_deadline,
                )
                if not self._adopting_pending:
                    try:
                        self._handed_to_active_run = (
                            self._control.register_terminal_finalization_claim_handoff(
                                session.id,
                                session_instance_id=session.instance_id,
                                run_epoch=session.run_epoch,
                                interruption_request_id=interruption_request_id,
                                expected_interrupt_payload=persisted_payload,
                                claim_id=self._claim_id,
                                heartbeat_stop=self._heartbeat_stop,
                                heartbeat_task=self._heartbeat_task,
                            )
                        )
                    except BaseException:
                        self._heartbeat_stop.set()
                        self._heartbeat_task.cancel()
                        await asyncio.gather(
                            self._heartbeat_task,
                            return_exceptions=True,
                        )
                        self._heartbeat_stop = None
                        self._heartbeat_task = None
                        raise
                    if self._handed_to_active_run:
                        self._heartbeat_stop = None
                        self._heartbeat_task = None
        return session

    async def _stop_heartbeat(
        self,
        *,
        superseded_by_exact_renewal: bool = False,
    ) -> None:
        heartbeat_stop = self._heartbeat_stop
        heartbeat_task = self._heartbeat_task
        self._heartbeat_stop = None
        self._heartbeat_task = None
        if heartbeat_stop is None or heartbeat_task is None:
            return
        heartbeat_stop.set()
        await _await_stopped_heartbeat(
            heartbeat_task, superseded_by_exact_renewal=superseded_by_exact_renewal
        )

    async def await_operation(
        self,
        operation: Callable[[], Awaitable[_OperationResultT]],
        *,
        operation_name: str,
    ) -> _OperationResultT:
        heartbeat_task = self._heartbeat_task
        if heartbeat_task is None:
            return await operation()
        return await self._finalization.await_operation(
            heartbeat_task=heartbeat_task,
            operation=operation,
            operation_name=operation_name,
        )

    async def _cleanup_preparation(
        self,
        authoritative_failure: BaseException | None,
    ) -> None:
        """Settle every owner created before offline finalization starts."""
        if self._preparation_cleaned:
            return
        try:
            reclaimed_handoff_heartbeat: asyncio.Task[None] | None = None
            if self._claim_id is not None and self._handed_to_active_run:
                reclaimed_handoff_heartbeat = (
                    self._control.reclaim_unaccepted_terminal_finalization_claim_handoff(
                        self._session_id,
                        claim_id=self._claim_id,
                    )
                )
            if reclaimed_handoff_heartbeat is not None:
                self._retained_for_recovery = True
                await self._recovery._run_cleanup_steps(
                    authoritative_failure=authoritative_failure,
                    steps=(
                        (
                            "unaccepted terminal finalization heartbeat shutdown",
                            lambda: reclaimed_handoff_heartbeat,
                        ),
                    ),
                )
            if (
                self._claim_id is not None
                and not self._handed_to_active_run
                and not self._retained_for_recovery
            ):
                claim_id = self._claim_id
                await self._recovery._run_cleanup_steps(
                    authoritative_failure=authoritative_failure,
                    steps=(
                        (
                            "unstarted terminal finalization heartbeat shutdown",
                            self._stop_heartbeat,
                        ),
                        (
                            "unstarted terminal finalization claim release",
                            lambda: self._recovery._release_incomplete_recovery_claim(
                                self._session_id,
                                claim_id,
                            ),
                        ),
                    ),
                )
        finally:
            if self._request_marker_active:
                self._request_marker_active = False
                self._control.end_interruption_request(self._session_id)
            self._preparation_cleaned = True

    async def finish_preparation(
        self,
        authoritative_failure: BaseException | None,
    ) -> None:
        """Clean local ownership and redeliver a carried transfer signal."""
        control_signal: BaseException | None = (
            self._transfer_cancellation or self._transfer_process_control
        )
        cleanup_failure: BaseException | None = None
        try:
            await self._cleanup_preparation(
                control_signal or authoritative_failure,
            )
        except BaseException as failure:
            cleanup_failure = failure
        if control_signal is None:
            if cleanup_failure is not None:
                raise cleanup_failure
            return
        if (
            isinstance(control_signal, asyncio.CancelledError)
            and cleanup_failure is not None
            and not isinstance(cleanup_failure, (Exception, asyncio.CancelledError))
        ):
            raise cleanup_failure
        secondary_failures = [
            failure
            for failure in (
                self._transfer_process_control
                if control_signal is self._transfer_cancellation
                else None,
                self._transfer_failure,
                authoritative_failure,
                cleanup_failure,
            )
            if failure is not None and failure is not control_signal
        ]
        self._transfer_cancellation = None
        self._transfer_failure = None
        self._transfer_process_control = None
        _raise_terminal_finalization_process_control(
            control_signal,
            secondary_failures,
        )

    async def release(
        self,
        authoritative_failure: BaseException | None,
    ) -> None:
        claim_id = self._claim_id
        if claim_id is None:
            return
        steps: list[tuple[str, Callable[[], Awaitable[None]]]] = [
            (
                "terminal evidence finalization heartbeat shutdown",
                self._stop_heartbeat,
            )
        ]
        if not self._retained_for_recovery:
            steps.append(
                (
                    "terminal evidence finalization claim release",
                    lambda: self._recovery._release_incomplete_recovery_claim(
                        self._session_id,
                        claim_id,
                    ),
                )
            )
        await self._recovery._run_cleanup_steps(
            authoritative_failure=authoritative_failure,
            steps=tuple(steps),
        )

    @asynccontextmanager
    async def finalize(
        self,
        *,
        session: Session,
        expected_payload: dict[str, Any],
        finalization: Callable[[Session], AsyncGenerator[Event, None]],
        terminal_decision: InvocationTerminalDecision | None,
    ) -> AsyncIterator[AsyncIterator[Event]]:
        """Renew and settle under the exact owner, then redeliver transfer control."""
        joined = await self.claim_or_join(session=session, expected_payload=expected_payload)
        if joined is not None:
            async with _close_delegated_event_stream(_joined_terminal_event(joined)) as owned:
                yield owned
            return
        assert self._claim_id is not None
        renewed = await self._finalization.renew_claim(
            session=session, claim_id=self._claim_id, expected_payload=expected_payload
        )
        if renewed is None:
            self._retained_for_recovery = True
            raise _IncompleteRecoveryClaimLost(
                "Terminal finalization ownership changed before offline settlement."
            )
        session, self._claim_expires_at, _local_deadline = renewed
        await self._stop_heartbeat(superseded_by_exact_renewal=True)
        if terminal_decision is not None:
            # The terminal-decision handler owns its renewal, monitoring and
            # release. A second monitor would mistake successful release for loss.
            owned_finalization = finalization(session)
        else:
            owned_finalization = self._finalization.stream(
                session=session,
                claim_id=self._claim_id,
                expected_payload=expected_payload,
                finalization=finalization(session),
            )
        if self._transfer_cancellation is not None:

            async def settle_cancelled_finalization() -> bool:
                async with _close_delegated_event_stream(owned_finalization) as owned:
                    async for _event in owned:
                        pass
                return True

            settlement = await await_shielded_task_outcome(
                asyncio.create_task(settle_cancelled_finalization()),
                cancellation=self._transfer_cancellation,
            )
            authoritative_cancellation = settlement.cancellation or self._transfer_cancellation
            secondary_failures = [
                failure
                for failure in (
                    self._transfer_process_control,
                    self._transfer_failure,
                    settlement.error,
                    settlement.subsequent_cancellation,
                )
                if failure is not None
            ]
            if not secondary_failures:
                raise authoritative_cancellation
            if len(secondary_failures) == 1:
                raise authoritative_cancellation from secondary_failures[0]
            raise authoritative_cancellation from BaseExceptionGroup(
                "Terminal finalization cancellation evidence",
                secondary_failures,
            )
        if self._transfer_process_control is not None:

            async def settle_process_control_finalization() -> bool:
                async with _close_delegated_event_stream(owned_finalization) as owned:
                    async for _event in owned:
                        pass
                return True

            settlement = await await_shielded_task_outcome(
                asyncio.create_task(settle_process_control_finalization()),
            )
            secondary_failures = [
                failure
                for failure in (
                    self._transfer_failure,
                    settlement.error,
                    settlement.cancellation,
                    settlement.subsequent_cancellation,
                )
                if failure is not None
            ]
            _raise_terminal_finalization_process_control(
                self._transfer_process_control,
                secondary_failures,
            )
        async with _close_delegated_event_stream(owned_finalization) as owned:
            yield owned


class InterruptedRunFinalization:
    """Accept a live handoff while preserving an existing recovery owner's claim."""

    def __init__(
        self,
        finalization: TerminalEvidenceFinalization,
        *,
        session: Session,
        task: asyncio.Task[Any] | None,
        transferred_from: asyncio.Task[Any] | None,
    ) -> None:
        self._finalization = finalization
        self._session_store = finalization._session_store
        self._control = finalization._session_control
        self._recovery = finalization._recovery
        self._session_id = session.id
        self._task = task
        self._claim_id: str | None = None
        self._handoff: TerminalFinalizationClaimHandoff | None = None
        self._borrowed = False
        if task is not None:
            self._handoff = self._control.take_terminal_finalization_claim_handoff(
                session.id,
                task=task,
                session_instance_id=session.instance_id,
                run_epoch=session.run_epoch,
                transferred_from=transferred_from,
            )
            if self._handoff is not None:
                self._claim_id = self._handoff.claim_id

    @property
    def claim_id(self) -> str | None:
        return self._claim_id

    async def await_operation(
        self,
        operation: Callable[[], Awaitable[_OperationResultT]],
        *,
        operation_name: str,
    ) -> _OperationResultT:
        if self._handoff is None:
            return await operation()
        return await self._finalization.await_operation(
            heartbeat_task=self._handoff.heartbeat_task,
            operation=operation,
            operation_name=operation_name,
        )

    async def _stop_handoff(
        self,
        *,
        superseded_by_exact_renewal: bool = False,
    ) -> None:
        if self._claim_id is None:
            return
        heartbeat_task = self._control.end_terminal_finalization_claim_handoff(
            self._session_id,
            claim_id=self._claim_id,
            task=self._task,
        )
        if (
            heartbeat_task is None
            and self._handoff is not None
            and self._handoff.claimed_by is self._task
        ):
            self._handoff.heartbeat_stop.set()
            heartbeat_task = self._handoff.heartbeat_task
        if heartbeat_task is None:
            return
        await _await_stopped_heartbeat(
            heartbeat_task, superseded_by_exact_renewal=superseded_by_exact_renewal
        )

    async def prepare(
        self,
        *,
        session: Session,
        payload: dict[str, Any],
        interruption_request_id: str,
    ) -> Session | Event:
        """Authenticate the handoff or join exact repair before live publication."""
        loaded_interrupted = session
        user_input_supersession_retained = (
            USER_INPUT_SUPERSESSION_INTENT_KEY in payload
            or AMBIGUOUS_USER_INPUT_SUPERSESSION_INTENT_KEY in payload
        )
        if user_input_supersession_retained:
            if self._task is None or self._handoff is None:
                authenticated_payload = await self._finalization._user_input_evidence.validated_user_input_supersession_interrupt_payload(
                    session=loaded_interrupted,
                    pending_interrupt_payload=payload,
                )
                if authenticated_payload is None:
                    raise SessionRuntimePublicationConflict(
                        "User-input supersession has no authenticated durable authority."
                    )
                checkpoint = await self._session_store.load_checkpoint(session.id)
                shared_claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
                joined_claim = None
                if shared_claim is not None:
                    joined_claim = await self._finalization.renew_claim(
                        session=loaded_interrupted,
                        claim_id=shared_claim[0],
                        expected_payload=authenticated_payload,
                    )
                if joined_claim is not None:
                    assert self._task is not None
                    assert shared_claim is not None
                    (
                        loaded_interrupted,
                        _joined_claim_expires_at,
                        joined_local_lease_deadline,
                    ) = joined_claim
                    heartbeat_stop, heartbeat_task = self._finalization.start_heartbeat(
                        session_id=session.id,
                        claim_id=shared_claim[0],
                        local_lease_deadline=joined_local_lease_deadline,
                    )
                    self._claim_id = shared_claim[0]
                    self._borrowed = self._recovery._owns_current_recovery_worker(
                        session.id, shared_claim[0]
                    )
                    self._handoff = TerminalFinalizationClaimHandoff(
                        session_instance_id=loaded_interrupted.instance_id,
                        run_epoch=loaded_interrupted.run_epoch,
                        interruption_request_id=interruption_request_id,
                        expected_interrupt_payload=copy_json_value(
                            authenticated_payload,
                            "expected_interrupt_payload",
                        ),
                        claim_id=shared_claim[0],
                        eligible_tasks=frozenset({self._task}),
                        heartbeat_stop=heartbeat_stop,
                        heartbeat_task=heartbeat_task,
                        claimed_by=self._task,
                    )
                else:
                    joined_event = await self._finalization.wait_for_repair(
                        session=loaded_interrupted,
                        expected_payload=authenticated_payload,
                    )
                    if joined_event is None:
                        joined_event = await self._finalization.repair_and_join(
                            session=loaded_interrupted,
                            expected_payload=authenticated_payload,
                        )
                    return joined_event
            if (
                loaded_interrupted.instance_id != self._handoff.session_instance_id
                or loaded_interrupted.run_epoch != self._handoff.run_epoch
                or payload != self._handoff.expected_interrupt_payload
                or interruption_request_id != self._handoff.interruption_request_id
            ):
                raise SessionRuntimePublicationConflict(
                    "User-input supersession changed after its live finalization handoff."
                )
            renewed_claim = await self._finalization.renew_claim(
                session=loaded_interrupted,
                claim_id=self._handoff.claim_id,
                expected_payload=self._handoff.expected_interrupt_payload,
            )
            if renewed_claim is None:
                raise _IncompleteRecoveryClaimLost(
                    "Terminal finalization ownership changed before live settlement."
                )
            loaded_interrupted, _renewed_until, _local_deadline = renewed_claim
        elif self._handoff is not None:
            raise SessionRuntimePublicationConflict(
                "Terminal finalization handoff lost its user-input supersession authority."
            )
        return loaded_interrupted

    @asynccontextmanager
    async def finalize(
        self,
        *,
        session: Session,
        expected_payload: dict[str, Any],
        finalization: Callable[[Session], AsyncGenerator[Event, None]],
    ) -> AsyncIterator[AsyncIterator[Event]]:
        """Settle borrowed work inline; only an owner starts a supervised stream."""
        if self._claim_id is None:
            raise RuntimeError("User-input supersession finalization lost its durable owner.")
        renewed = await self._finalization.renew_claim(
            session=session, claim_id=self._claim_id, expected_payload=expected_payload
        )
        if renewed is None:
            raise _IncompleteRecoveryClaimLost(
                "Terminal finalization ownership changed before terminal publication."
            )
        session, _renewed_until, _local_deadline = renewed
        await self._stop_handoff(superseded_by_exact_renewal=True)
        # A second supervisor would wait for the outer worker awaiting this stream.
        owned_finalization = (
            finalization(session)
            if self._borrowed
            else self._finalization.stream(
                session=session,
                claim_id=self._claim_id,
                expected_payload=expected_payload,
                finalization=finalization(session),
            )
        )
        async with _close_delegated_event_stream(owned_finalization) as owned:
            yield owned

    async def release(self, authoritative_failure: BaseException | None) -> None:
        """Stop the handoff and release only locally owned, settled work."""
        claim_id = self._claim_id
        if claim_id is not None:
            await self._recovery._run_cleanup_steps(
                authoritative_failure=authoritative_failure,
                steps=(
                    (
                        "terminal evidence finalization handoff shutdown",
                        self._stop_handoff,
                    ),
                    # A borrower cannot release its supervisor's claim.
                    *(
                        ()
                        if self._borrowed
                        else (
                            (
                                "terminal evidence finalization claim release",
                                lambda: self._recovery._release_incomplete_recovery_claim(
                                    self._session_id,
                                    claim_id,
                                ),
                            ),
                        )
                    ),
                ),
            )
