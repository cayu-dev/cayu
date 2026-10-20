"""Normalize owned deadline stops only after durable terminalization and cleanup."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import cast

from cayu.deadlines import (
    ExecutionDeadline,
    ExecutionDeadlineExceeded,
    deadline_stream,
    resumed_execution_deadline,
)
from cayu.events import Event, EventType
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import Session
from cayu.sessions.store import SessionStore


@dataclass
class DeadlineOutcomeAdmission:
    """Exact admission observed by this stream before setup emits a lifecycle event."""

    session_id: str | None = None
    session_instance_id: str | None = None
    run_epoch: int | None = None

    def observe(self, session: Session) -> None:
        self.session_id = session.id
        self.session_instance_id = session.instance_id
        self.run_epoch = session.run_epoch


def continuation_deadline_outcome(
    stream: AsyncGenerator[Event, None],
    store: SessionStore,
    session: Session | None,
    *,
    admission: DeadlineOutcomeAdmission | None = None,
) -> AsyncGenerator[Event, None]:
    if session is None:
        return stream
    return deadline_outcome_stream(
        stream,
        resumed_execution_deadline(session.execution_deadline),
        session_id=session.id,
        store=store,
        admission=admission,
    )


def admitted_continuation_deadline_outcome(
    stream: AsyncGenerator[Event, None], store: SessionStore, session: Session
) -> AsyncGenerator[Event, None]:
    """Time execution after receipt settlement using its exact admitted epoch."""
    admission = DeadlineOutcomeAdmission()
    admission.observe(session)
    return continuation_deadline_outcome(stream, store, session, admission=admission)


@dataclass(frozen=True)
class ContinuationDeadlineStart:
    """Private stream boundary: receipt settlement is over; execution starts next."""

    session: Session


async def settled_continuation_deadline_outcome(
    stream: AsyncGenerator[Event | ContinuationDeadlineStart, None],
    store: SessionStore,
) -> AsyncGenerator[Event, None]:
    """Allow receipt settlement, then time every remaining continuation operation."""
    # The fenced adapter only advances/closes its producer; this private phase
    # marker is consumed here before the public event stream sees it.
    async with _close_delegated_event_stream(
        cast("AsyncGenerator[Event, None]", stream)
    ) as owned_stream:
        async for item in owned_stream:
            if isinstance(item, ContinuationDeadlineStart):
                timed = admitted_continuation_deadline_outcome(
                    cast("AsyncGenerator[Event, None]", owned_stream), store, item.session
                )
                async with _close_delegated_event_stream(timed) as execution:
                    async for event in execution:
                        yield event
                return
            yield item


async def deadline_outcome_stream(
    stream: AsyncGenerator[Event, None],
    boundary: ExecutionDeadline,
    *,
    session_id: str,
    store: SessionStore,
    admission: DeadlineOutcomeAdmission | None = None,
) -> AsyncGenerator[Event, None]:
    lifecycle_types = (
        EventType.SESSION_STARTED,
        EventType.SESSION_RESUMED,
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    )
    previous_lifecycle_id: str | None = None
    observed_epoch: int | None = None
    emitted_terminal: str | None = None
    if boundary.expires_at is not None:
        previous = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=lifecycle_types,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if previous:
            previous_lifecycle_id = previous[0].event.id
    stream = deadline_stream(stream, boundary)
    try:
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for item in owned_stream:
                if item.session_id == session_id and item.type in {
                    EventType.SESSION_STARTED,
                    EventType.SESSION_RESUMED,
                }:
                    value = item.payload.get("run_epoch")
                    if type(value) is int:
                        observed_epoch = value
                if item.session_id == session_id and item.type is EventType.SESSION_INTERRUPTED:
                    emitted_terminal = item.id
                yield item
    except ExecutionDeadlineExceeded:
        records = await store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=lifecycle_types,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=2,
            )
        )
        terminal = None if not records else records[0].event
        evidence = None if terminal is None else terminal.payload.get("failure_evidence")
        if observed_epoch is None and len(records) == 2:
            origin = records[1].event
            if origin.id != previous_lifecycle_id and origin.type in {
                EventType.SESSION_STARTED,
                EventType.SESSION_RESUMED,
            }:
                value = origin.payload.get("run_epoch")
                if type(value) is int:
                    observed_epoch = value
        if observed_epoch is None and admission is not None and admission.session_id == session_id:
            current_session = await store.load(session_id)
            if (
                current_session is not None
                and current_session.instance_id == admission.session_instance_id
            ):
                observed_epoch = admission.run_epoch
        terminal_deadline = None if terminal is None else terminal.payload.get("execution_deadline")

        if (
            terminal is None
            or terminal.id == previous_lifecycle_id
            or terminal.type is not EventType.SESSION_INTERRUPTED
            or terminal.payload.get("reason") != "limit_reached"
            or terminal.payload.get("limit") != "execution_deadline"
            or type(evidence) is not dict
            or evidence.get("classification") != "deadline"
            or observed_epoch is None
            or evidence.get("run_epoch") != observed_epoch
            or evidence.get("session_id") != session_id
            or type(terminal_deadline) is not dict
            or terminal_deadline.get("expires_at") != boundary.model_dump(mode="json")["expires_at"]
            or not boundary.expired
        ):
            raise
        if terminal.id != emitted_terminal:
            yield terminal
