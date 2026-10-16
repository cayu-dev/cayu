"""Continuation admission for an execution whose durable owner has expired."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from cayu.events import Event
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions._invocation_lifecycle import require_released_invocation_command_authority
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.records import Session, SessionStatus, SessionStatusConflict
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    SessionExecutionInProgress,
)
from cayu.sessions.store import SessionStore


@dataclass(frozen=True)
class AbandonedExecution:
    session_id: str
    session_instance_id: str
    run_epoch: int
    status: SessionStatus
    # Whether the continuation that follows the takeover replays elected calls.
    replays_tool_calls: bool = False
    retain_recovery_run_epoch: Callable[[int], None] | None = None

    def require_matches(self, session: Session) -> None:
        if (session.id, session.instance_id, session.run_epoch) != (
            self.session_id,
            self.session_instance_id,
            self.run_epoch,
        ):
            raise SessionExecutionInProgress(session.id, session.status)


_ABANDONED_EXECUTION: ContextVar[AbandonedExecution | None] = ContextVar(
    "cayu_abandoned_execution", default=None
)


@contextmanager
def _abandoned_execution(
    session: Session,
    *,
    replays_tool_calls: bool = False,
    retain_recovery_run_epoch: Callable[[int], None] | None = None,
) -> Iterator[None]:
    token = _ABANDONED_EXECUTION.set(
        AbandonedExecution(
            session.id,
            session.instance_id,
            session.run_epoch,
            session.status,
            replays_tool_calls,
            retain_recovery_run_epoch,
        )
    )
    try:
        yield
    finally:
        _ABANDONED_EXECUTION.reset(token)


def abandoned_execution_takeover(session: Session) -> bool:
    """Whether recovering ``session`` is part of a continuation's takeover of it."""

    current = _ABANDONED_EXECUTION.get()
    return current is not None and (session.id, session.instance_id) == (
        current.session_id,
        current.session_instance_id,
    )


def abandoned_running_execution(session: Session) -> bool:
    """Whether recovering ``session`` takes over its dead running execution to replay.

    Only the session incarnation that was taken over qualifies; its epoch was
    checked when the recovery reserved it, and that recovery has since fenced it.
    An execution that was already being interrupted when its owner died is not one,
    and neither is a takeover by a continuation that can't replay tool calls (an
    answer, approval or provider-operation resolution): it closes the round instead.
    """

    current = _ABANDONED_EXECUTION.get()
    return (
        current is not None
        and (session.id, session.instance_id) == (current.session_id, current.session_instance_id)
        and current.status is SessionStatus.RUNNING
        and current.replays_tool_calls
    )


def require_abandoned_execution_matches(session: Session) -> None:
    """Check the inspected incarnation/epoch inside the store's recovery reservation.

    A different process cannot replace an execution owner within an epoch. The native reservation
    also rejects live owners under the same lock/transaction, so a changed epoch
    or a successor awaiting its first heartbeat cannot be mistaken for this owner.
    """
    expected = _ABANDONED_EXECUTION.get()
    if expected is not None:
        expected.require_matches(session)


def retain_abandoned_execution_recovery(session: Session) -> None:
    """Retain the native claim's epoch before its recovery work is dispatched."""

    expected = _ABANDONED_EXECUTION.get()
    if expected is None or session.id != expected.session_id:
        return
    if (
        session.instance_id != expected.session_instance_id
        or session.run_epoch != expected.run_epoch + 1
    ):
        raise SessionStatusConflict("Abandoned recovery claimed a different session run.")
    if expected.retain_recovery_run_epoch is not None:
        expected.retain_recovery_run_epoch(session.run_epoch)


async def recover_abandoned_execution(
    *,
    store: SessionStore,
    session: Session,
    locally_active: bool,
    recover: Callable[
        [IncompleteSessionRecoveryRequest], Awaitable[IncompleteSessionRecoveryResult]
    ],
    settle_model_dispatch: Callable[[Session], Awaitable[tuple[Event, ...]]] | None = None,
    replays_tool_calls: bool = False,
    retain_recovery_run_epoch: Callable[[int], None] | None = None,
) -> tuple[Event, ...]:
    if session.status not in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
        return ()
    if locally_active or not store.supports_session_execution:
        raise SessionExecutionInProgress(session.id, session.status)
    execution = await store._inspect_session_execution_owner(session.id)
    if execution.state != "owner_lost" or execution.run_epoch != session.run_epoch:
        raise SessionExecutionInProgress(session.id, session.status)
    recovery_run_epoch: int | None = None

    def retain_claim(run_epoch: int) -> None:
        nonlocal recovery_run_epoch
        recovery_run_epoch = run_epoch

    async def require_recovered_run() -> None:
        # A fresh read alone cannot identify our recovery: another continuation
        # may already have completed, or the public id may have been reused.
        current = await store.load(session.id)
        if current is None or current.instance_id != session.instance_id:
            raise SessionStatusConflict("The session was replaced before this turn could resume.")
        if recovery_run_epoch is None or current.run_epoch not in {
            recovery_run_epoch,
            recovery_run_epoch + 1,
        }:
            raise SessionStatusConflict("The session run changed before this turn could resume.")
        if current.run_epoch != recovery_run_epoch:
            # Cleanup advances the epoch once more. Accept that change only
            # with the exact durable release receipt for our recovery claim.
            checkpoint = await store.load_checkpoint(session.id)
            profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            if profile is None or profile.run_epoch != recovery_run_epoch:
                raise SessionStatusConflict(
                    "The session run changed before this turn could resume."
                )
            try:
                require_released_invocation_command_authority(
                    current,
                    checkpoint,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    active_profile=profile,
                )
            except SessionRunFenced as exc:
                raise SessionStatusConflict(
                    "The session run changed before this turn could resume."
                ) from exc
        if current.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
            raise SessionExecutionInProgress(session.id, current.status)
        if retain_recovery_run_epoch is not None:
            retain_recovery_run_epoch(current.run_epoch)

    with _abandoned_execution(
        session,
        replays_tool_calls=replays_tool_calls,
        retain_recovery_run_epoch=retain_claim,
    ):
        # The lost owner may have died with a provider call in flight. Its outcome is
        # unknown, so settle it as interrupted (charging its budget reservations in full)
        # before ordinary recovery fences the run.
        settled = () if settle_model_dispatch is None else await settle_model_dispatch(session)
        if recovery_run_epoch is not None:
            # The settlement ran its own fenced recovery (a lost call from another
            # Cayu build is terminalized from its record), which already ended the run.
            await require_recovered_run()
            return settled
        result = await recover(
            IncompleteSessionRecoveryRequest(
                session_id=session.id,
                inactive_for_seconds=0,
                reason="continuation_recovered_abandoned_execution",
                metadata={"previous_run_epoch": session.run_epoch},
            )
        )
    if result.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
        raise SessionExecutionInProgress(session.id, result.status)
    await require_recovered_run()
    return (*settled, *result.events)
