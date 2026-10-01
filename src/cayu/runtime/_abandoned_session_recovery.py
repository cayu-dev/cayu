"""Continuation admission for an execution whose durable owner has expired."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from cayu.events import Event
from cayu.sessions.base import (
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    Session,
    SessionExecutionInProgress,
    SessionStatus,
    SessionStore,
)


@dataclass(frozen=True)
class AbandonedExecution:
    session_id: str
    session_instance_id: str
    run_epoch: int

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
def _abandoned_execution(session: Session) -> Iterator[None]:
    token = _ABANDONED_EXECUTION.set(
        AbandonedExecution(session.id, session.instance_id, session.run_epoch)
    )
    try:
        yield
    finally:
        _ABANDONED_EXECUTION.reset(token)


def require_abandoned_execution_matches(session: Session) -> None:
    """Check the inspected incarnation/epoch inside the store's recovery reservation.

    A different process cannot replace an execution owner within an epoch. The native reservation
    also rejects live owners under the same lock/transaction, so a changed epoch
    or a successor awaiting its first heartbeat cannot be mistaken for this owner.
    """
    expected = _ABANDONED_EXECUTION.get()
    if expected is not None:
        expected.require_matches(session)


async def recover_abandoned_execution(
    *,
    store: SessionStore,
    session: Session,
    locally_active: bool,
    recover: Callable[
        [IncompleteSessionRecoveryRequest], Awaitable[IncompleteSessionRecoveryResult]
    ],
) -> tuple[Event, ...]:
    if session.status not in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
        return ()
    if locally_active or not store.supports_session_execution:
        raise SessionExecutionInProgress(session.id, session.status)
    execution = await store._inspect_session_execution_owner(session.id)
    if execution.state != "owner_lost" or execution.run_epoch != session.run_epoch:
        raise SessionExecutionInProgress(session.id, session.status)
    with _abandoned_execution(session):
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
    return result.events
