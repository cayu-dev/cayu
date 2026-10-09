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
    SessionExecutionInProgress,
    SessionStore,
)
from cayu.sessions.records import Session, SessionStatus


@dataclass(frozen=True)
class AbandonedExecution:
    session_id: str
    session_instance_id: str
    run_epoch: int
    status: SessionStatus
    # Whether the continuation that follows the takeover replays elected calls.
    replays_tool_calls: bool = False

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
def _abandoned_execution(session: Session, *, replays_tool_calls: bool = False) -> Iterator[None]:
    token = _ABANDONED_EXECUTION.set(
        AbandonedExecution(
            session.id,
            session.instance_id,
            session.run_epoch,
            session.status,
            replays_tool_calls,
        )
    )
    try:
        yield
    finally:
        _ABANDONED_EXECUTION.reset(token)


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
) -> tuple[Event, ...]:
    if session.status not in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
        return ()
    if locally_active or not store.supports_session_execution:
        raise SessionExecutionInProgress(session.id, session.status)
    execution = await store._inspect_session_execution_owner(session.id)
    if execution.state != "owner_lost" or execution.run_epoch != session.run_epoch:
        raise SessionExecutionInProgress(session.id, session.status)
    with _abandoned_execution(session, replays_tool_calls=replays_tool_calls):
        # The lost owner may have died with a provider call in flight. Its outcome is
        # unknown, so settle it as interrupted (charging its budget reservations in full)
        # before ordinary recovery fences the run.
        settled = () if settle_model_dispatch is None else await settle_model_dispatch(session)
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
    return (*settled, *result.events)
