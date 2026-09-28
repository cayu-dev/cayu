"""Typed source-owned discovery for the collaboration host.

These reads are neither claims nor dispatch grants. Cursors are local scan hints;
restart begins a fresh sweep against the same durable source indexes. Missing
role coverage must remain explicit when the host composes this adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from cayu.collaboration._clarification_recovery_types import (
    ClarificationDueQuestionPage,
    ClarificationPendingDeliveryPage,
    ClarificationPendingServicePage,
)
from cayu.collaboration._contracts import ContractValue
from cayu.collaboration._planning_inspection import list_pending_plans
from cayu.collaboration._planning_records import RequestPlanningCursor, RequestPlanningPage
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._producer_recovery import ProducerPendingPage, pending_producer_outputs
from cayu.collaboration._wait_discovery import WaitDiscoveryCursor, WaitDiscoveryPage
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.clarifications import ClarificationDueCursor
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import ParticipantRef
from cayu.collaboration.requests import RequestDueCursor, RequestDuePage
from cayu.sessions._participant_discovery import (
    ParticipantSessionCursor,
    discover_participant_sessions,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp

Source = Literal[
    "requests", "plans", "producers", "questions", "deliveries", "services", "waits", "sessions"
]
Cursor = (
    RequestDueCursor
    | RequestPlanningCursor
    | ClarificationDueCursor
    | WaitDiscoveryCursor
    | ParticipantSessionCursor
    | int
    | None
)


@dataclass(frozen=True, slots=True)
class HostDiscoveryPage:
    """Private owner records; public host inspection must project content-free status."""

    source: Source
    items: tuple[ContractValue, ...]
    next_cursor: Cursor
    retained_bytes: int
    observed_at_ms: int | None = None

    @property
    def reached_scan_end(self) -> bool:
        """Only this source sweep ended; no global quiescence is implied."""
        return self.next_cursor is None


async def observe_host_source(reads, key, app, source, **query):
    """Retain one exact source query without monopolizing the service pass."""
    redactor = app._secret_redactor
    cursor = query.get("cursor")
    participant = query.get("participant")
    material = b"\0".join(
        (
            source.encode("ascii"),
            contract_bytes(query["context"], redactor=redactor),
            b"null" if participant is None else contract_bytes(participant, redactor=redactor),
            str(cursor).encode("ascii")
            if cursor is None or type(cursor) is int
            else contract_bytes(cursor, redactor=redactor),
            str(query["limit"]).encode("ascii"),
            str(query["max_bytes"]).encode("ascii"),
        )
    )
    return await reads.observe(
        key,
        expectation=material,
        reserved_bytes=query["max_bytes"] + len(material),
        read=lambda: discover_host_source(app, source, **query),
    )


async def discover_host_source(
    app: CayuApp,
    source: Source,
    *,
    context: MandateAccessContext | CollaborationAccessContext,
    participant: ParticipantRef | None = None,
    cursor: Cursor = None,
    limit: int,
    max_bytes: int,
) -> HostDiscoveryPage:
    """Read one bounded page using a freshly supplied existing access context.

    No context is inferred from a principal, receipt, or another role's grant.
    The owner authenticates it normally. A failed read does not advance a cursor.
    """
    if type(limit) is not int or not 1 <= limit <= 32:
        raise ValueError("Host discovery limit must be between 1 and 32.")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 64 * 1024:
        raise ValueError("Host discovery byte bound must be between 1 and 65536.")
    redactor = app._secret_redactor
    observed_at_ms = None
    if source in ("requests", "plans"):
        if type(context) is not MandateAccessContext:
            raise TypeError("Request discovery requires current mandate access.")
        checked = prepare_contract(MandateAccessContext, context, redactor=redactor)
        if participant is not None:
            raise ValueError("Scope-wide discovery does not accept a participant filter.")
        if source == "requests":
            if cursor is not None and type(cursor) is not RequestDueCursor:
                raise TypeError("Request discovery cursor belongs to another source.")
            page = prepare_contract(
                RequestDuePage,
                await app._request_coordinator.due(
                    context=checked, cursor=cursor, limit=limit, wait_for_settlement=True
                ),
                redactor=redactor,
            )
            items, next_cursor = page.items, page.next_cursor
            observed_at_ms = page.observed_at_ms
        else:
            if cursor is not None and type(cursor) is not RequestPlanningCursor:
                raise TypeError("Planning discovery cursor belongs to another source.")
            plans = prepare_contract(
                RequestPlanningPage,
                await list_pending_plans(
                    app._request_coordinator,
                    context=checked,
                    after=cursor,
                    limit=limit,
                    wait_for_settlement=True,
                ),
                redactor=redactor,
            )
            items, next_cursor = plans.records, plans.next_cursor
    else:
        if type(context) is not CollaborationAccessContext:
            raise TypeError("Maintenance discovery requires current collaboration access.")
        access = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
        if source == "sessions":
            if cursor is not None and type(cursor) is not ParticipantSessionCursor:
                raise TypeError("Session discovery cursor belongs to another source.")
            items, next_cursor = await discover_participant_sessions(
                app, participant, context=access, cursor=cursor, limit=limit
            )
        elif source == "waits":
            if participant is not None:
                raise ValueError("Scope-wide discovery does not accept a participant filter.")
            if cursor is not None and type(cursor) is not WaitDiscoveryCursor:
                raise TypeError("Wait discovery cursor belongs to another source.")
            waits = prepare_contract(
                WaitDiscoveryPage,
                await app.list_collaboration_waits(context=access, cursor=cursor, limit=limit),
                redactor=redactor,
            )
            items, next_cursor = waits.items, waits.next_cursor
        elif source == "producers":
            if participant is None:
                raise ValueError("Producer discovery requires an exact participant.")
            participant = prepare_contract(ParticipantRef, participant, redactor=redactor)
            if cursor is not None and (type(cursor) is not int or not 0 <= cursor <= 2**53 - 1):
                raise TypeError("Producer discovery cursor belongs to another source.")
            producers = prepare_contract(
                ProducerPendingPage,
                await pending_producer_outputs(
                    app,
                    participant,
                    context=access,
                    after=0 if cursor is None else cursor,
                    limit=limit,
                    wait_for_settlement=True,
                ),
                redactor=redactor,
            )
            items, next_cursor = producers.items, producers.next_cursor
        else:
            if participant is not None:
                raise ValueError("Scope-wide discovery does not accept a participant filter.")
            if cursor is not None and type(cursor) is not ClarificationDueCursor:
                raise TypeError("Clarification discovery cursor belongs to another source.")
            if source == "questions":
                questions = prepare_contract(
                    ClarificationDueQuestionPage,
                    await app._clarification_coordinator.due_questions(
                        context=access, cursor=cursor, limit=limit, wait_for_settlement=True
                    ),
                    redactor=redactor,
                )
                items, next_cursor = questions.items, questions.next_cursor
            elif source == "deliveries":
                deliveries = prepare_contract(
                    ClarificationPendingDeliveryPage,
                    await app._clarification_coordinator.pending_deliveries(
                        context=access, cursor=cursor, limit=limit, wait_for_settlement=True
                    ),
                    redactor=redactor,
                )
                items, next_cursor = deliveries.items, deliveries.next_cursor
            elif source == "services":
                services = prepare_contract(
                    ClarificationPendingServicePage,
                    await app._clarification_coordinator.pending_services(
                        context=access, cursor=cursor, limit=limit, wait_for_settlement=True
                    ),
                    redactor=redactor,
                )
                items, next_cursor = services.items, services.next_cursor
            else:
                raise ValueError("Unknown host discovery source.")
    if len(items) > limit:
        raise ValueError("Host discovery owner exceeded its requested item limit.")
    retained_bytes = sum(len(contract_bytes(item, redactor=redactor)) for item in items)
    if next_cursor is not None:
        retained_bytes += (
            len(str(next_cursor))
            if isinstance(next_cursor, int)
            else len(contract_bytes(next_cursor, redactor=redactor))
        )
    if retained_bytes > max_bytes:
        raise ValueError("Host discovery page exceeds its retained-byte limit.")
    return HostDiscoveryPage(source, tuple(items), next_cursor, retained_bytes, observed_at_ms)
