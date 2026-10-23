"""Run summaries and human touches for Cayu Cloud, through one versioned contract.

Cayu Cloud shows each agent's runs and the decisions people made on them
without reading conversations. The agent serves a content-free change feed:
each page holds the summaries of the runs that changed after a cursor and the
touches that settled in the same range. Summaries carry identities, names,
times, counts and an estimated cost; touches carry who decided, the reason they
gave, the tool and the attention identity. Neither carries messages, tool
arguments, questions, answers or results. Cloud reads a conversation live from
the agent's Developer dashboard when someone opens it.

The cursor names the session store and a position in its durable event
sequence, so a reader that keeps it continues after its own or the agent's
restart without a gap. A cursor from another store, or from history the store
no longer holds because it was restored from a backup, is refused as stale, and
the reader reads again from the start. A run's summary is its current state, so
it can arrive again on a later page; readers keep the latest summary per
``run_id`` and each touch once per ``touch_id``.

Within a contract version fields are only added. Readers ignore fields and
values they do not know.
"""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal
from functools import cache
from hashlib import sha256
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, ConfigDict

from cayu._validation import MAX_DURABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu._version import __version__
from cayu.approvals.actors import ResolutionActorSource
from cayu.budgets.pricing import PriceBook, SessionCostTotals, default_price_book
from cayu.events import EventType
from cayu.runtime._cost_accounting import COST_EVENT_TYPES
from cayu.sessions._run_identity import (
    RUN_LINEAGE_METADATA_KEY,
    is_eval_trial,
    session_run_id,
    session_run_lineage,
)
from cayu.sessions._supervision_projection import (
    SUPERVISION_ANCHOR_EVENT_TYPES,
    classify_supervision_anchors,
)
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.overview import SessionOverviewFacts, SessionOverviewReader
from cayu.sessions.overview import SessionOverviewStatus as CloudRunStatus
from cayu.sessions.pending_action_contracts import (
    PendingActionRecord,
)
from cayu.sessions.records import Session
from cayu.sessions.supervision_events import (
    SupervisionEvent,
    SupervisionEventKind,
    SupervisionMessageOrigin,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.sessions.records import EventRecord

CLOUD_AGENT_CONTRACT = "cayu.cloud-agent"
#: The contract versions this Runtime serves, oldest first.
CLOUD_AGENT_CONTRACT_VERSIONS: tuple[int, ...] = (1,)
CLOUD_ACTIVITY_DEFAULT_LIMIT = 100
CLOUD_ACTIVITY_MAX_LIMIT = 200
_ACTIVITY_CONTRACT = f"{CLOUD_AGENT_CONTRACT}/v1"
# Events read per store query, and per page, while collecting changes.
_SCAN_BATCH = 200
_SCAN_BUDGET = 2_000
# A cursor is ``{store_id}.0`` before any event, else ``{store_id}.{sequence}.{mark}``,
# the mark naming the exact event at that sequence.
_STORE_IDENTITY = re.compile(r"[0-9a-f]{32}")
_CURSOR = re.compile(r"([0-9a-f]{32})\.(?:0|([1-9][0-9]{0,15})\.([0-9a-f]{32}))")
# Cursors issued before they named their store: a bare sequence.
_STORELESS_CURSOR = re.compile(r"0|[1-9][0-9]{0,15}")
#: The longest cursor the contract issues.
CLOUD_ACTIVITY_CURSOR_MAX_CHARS = 32 + 1 + 16 + 1 + 32
# A session is reported from the event that starts or forks it. After these
# events a run's summary can read differently (its status, waiting items,
# Release or cost), and the anchors settle touches.
_ACTIVITY_EVENT_TYPES = tuple(
    dict.fromkeys(
        (
            EventType.SESSION_STARTED,
            EventType.SESSION_FORKED,
            EventType.SESSION_RESUMED,
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
            EventType.SESSION_INTERRUPTED,
            EventType.SESSION_AWAITING_USER_INPUT,
            EventType.SESSION_DELEGATED_ACTION_UPDATED,
            EventType.TOOL_CALL_APPROVAL_REQUESTED,
            *COST_EVENT_TYPES,
            *SUPERVISION_ANCHOR_EVENT_TYPES,
        )
    )
)
_ANCHOR_EVENT_TYPES = frozenset(SUPERVISION_ANCHOR_EVENT_TYPES)


class CloudActivityCursorInvalid(ValueError):
    """An activity cursor is not one this contract issues; the message never echoes it."""


class CloudActivityCursorStale(ValueError):
    """A cursor this store cannot continue: read again from the start.

    It names another store, such as one that replaced the agent's database or
    an in-memory store before a restart, or an event the store no longer holds
    at its position, as after a restore from a backup taken before the cursor.
    """


class _ContractModel(BaseModel):
    # Fields are only added within a version, so readers ignore the ones they don't know.
    model_config = ConfigDict(extra="ignore", frozen=True)


class CloudContract(_ContractModel):
    """What the agent serves: the contract, the versions of it and the store it reads.

    ``store_id`` is the session store's identity, which every activity cursor
    names; it is ``None`` when the store has none, and then activity is not
    served.
    """

    contract: str = CLOUD_AGENT_CONTRACT
    versions: tuple[int, ...] = CLOUD_AGENT_CONTRACT_VERSIONS
    runtime_version: str = __version__
    features: tuple[str, ...] = ("activity", "tool_knowledge_registration", "tool_knowledge_usage")
    store_id: str | None = None


class CloudPrincipal(_ContractModel):
    """Who started a run, as its invocation origin recorded it.

    ``trust`` is ``server_verified`` when an authenticated server request
    started it and ``host_asserted`` when the application named the subject.
    """

    subject: str
    trust: str


class CloudActor(_ContractModel):
    """Who settled a touch, as the resolution recorded it.

    ``source`` is ``http_auth`` for an authenticated server request,
    ``request`` for an identity the caller asserted and ``system`` for the
    Runtime itself.
    """

    subject: str
    source: str | None


class CloudRunCost(_ContractModel):
    """Estimated cost of a session's own model use, from Cayu's bundled price book.

    ``complete`` is false when some model use could not be priced, so
    ``amount`` counts less than was spent.
    """

    amount: Decimal
    currency: str
    complete: bool


class CloudWaitingItem(_ContractModel):
    """One pending action a run waits on, without its question or arguments.

    ``kind`` is ``tool_approval``, ``user_input``, ``manual_recovery`` or
    ``delegated_action``. A delegated action points at the child session whose
    own item a person settles; it has no ``attention_id``. ``requested_at`` is
    when the run began waiting on it.
    """

    kind: str
    attention_id: str | None
    tool_name: str | None
    requested_at: datetime
    child_session_id: str | None
    child_run_id: str | None


class CloudRun(_ContractModel):
    """The current summary of one session incarnation, keyed by ``run_id``.

    Every session is summarized from when it starts or is forked, including
    subagents: ``root_run_id`` links to the original root incarnation when
    known; a root positively identifies itself with ``root_run_id == run_id``.
    ``parent_run_id`` links to the original immediate parent. Session IDs are
    reusable lookup aliases, never persistence or attribution keys, and
    ``fork_controller`` names the outside controller, such as Compass, whose
    copy this is. A fork that has not run carries its source's status, no
    ``ended_at`` and only its own counts and cost. ``eval`` says whether the
    session is an eval trial or was created under one, as a child or a fork.
    ``reason`` is the Runtime's code for the current state, such as
    ``completed``, ``tool_approval_required`` or ``operator_requested``.
    ``ended_at`` is set once the run completed, failed or stopped and is
    cleared when it continues. ``release_id`` is the Release of its most recent
    work. ``model_steps`` and ``tool_calls`` count the session's own steps.
    """

    run_id: str
    root_run_id: str | None
    parent_run_id: str | None
    session_id: str
    root_session_id: str
    parent_session_id: str | None
    source: str
    forked: bool
    fork_controller: str | None
    eval: bool
    agent_name: str
    status: CloudRunStatus
    reason: str
    started_by: CloudPrincipal | None
    started_at: datetime
    ended_at: datetime | None
    release_id: str | None
    model_steps: int
    tool_calls: int
    cost: CloudRunCost
    waiting_for: tuple[CloudWaitingItem, ...]


class CloudTouch(_ContractModel):
    """One settled human touch on a session, without its content.

    ``kind`` is ``approve``, ``edit``, ``deny``, ``expire``, ``answer``,
    ``recover``, ``interrupt`` or ``steer``. ``stepped_in`` says whether a
    person did it: false for an expiry, for a system actor and for a steering
    message another session or peer sent. ``attention_id`` names the waiting
    item it settled.
    """

    touch_id: str
    run_id: str
    root_run_id: str | None
    parent_run_id: str | None
    session_id: str
    agent_name: str | None
    kind: str
    outcome: str
    stepped_in: bool
    by: CloudActor | None
    reason: str | None
    tool_name: str | None
    attention_id: str | None
    requested_at: datetime | None
    settled_at: datetime
    release_id: str | None


class CloudActivityPage(_ContractModel):
    """Runs that changed and touches that settled after a cursor, in sequence order.

    Continue from ``next_cursor``. ``has_more`` says whether more activity was
    already available; when it is false, read again later from the same cursor.
    """

    contract: str = _ACTIVITY_CONTRACT
    runs: tuple[CloudRun, ...]
    touches: tuple[CloudTouch, ...]
    next_cursor: str
    has_more: bool


async def read_cloud_contract(app: CayuApp) -> CloudContract:
    """The contract the agent serves, naming its session store."""

    try:
        store_id = await _store_identity(app)
    except NotImplementedError:
        store_id = None
    return CloudContract(store_id=store_id)


async def _store_identity(app: CayuApp) -> str:
    identity = await app.session_store.read_store_identity()
    if type(identity) is not str or _STORE_IDENTITY.fullmatch(identity) is None:
        raise ValueError("The session store returned an invalid store identity.")
    return identity


def _event_mark(record: EventRecord) -> str:
    """Name the exact event at a sequence, so a store that reissued it is told apart."""

    event = record.event
    identity = [record.sequence, event.session_id, event.id, event.timestamp.isoformat()]
    return sha256(canonical_durable_json_bytes(identity, "activity cursor")).hexdigest()[:32]


def _cursor(store_id: str, position: int, mark: str | None) -> str:
    return f"{store_id}.0" if mark is None else f"{store_id}.{position}.{mark}"


async def _cursor_position(
    app: CayuApp, store_id: str, cursor: str | None
) -> tuple[int, str | None]:
    """The sequence and event mark a cursor continues from, once it proves this store's.

    A store whose event at the cursor's sequence is another event has reissued
    that sequence: the cursor is stale. A position without an event is
    accepted, because deleting a session deletes its events.
    """

    if cursor is None:
        return 0, None
    if type(cursor) is not str:
        raise CloudActivityCursorInvalid("Invalid activity cursor.")
    if _STORELESS_CURSOR.fullmatch(cursor) is not None:
        raise CloudActivityCursorStale("The activity cursor names no store.")
    match = _CURSOR.fullmatch(cursor)
    if match is None:
        raise CloudActivityCursorInvalid("Invalid activity cursor.")
    if match[1] != store_id:
        raise CloudActivityCursorStale("The activity cursor names another store.")
    if match[2] is None:
        return 0, None
    position, mark = int(match[2]), match[3]
    if position > MAX_DURABLE_JSON_INTEGER:
        raise CloudActivityCursorInvalid("Invalid activity cursor.")
    held = await app.session_store.query_events(
        EventQuery(after_sequence=position - 1, limit=1, order_by=EventOrder.SEQUENCE_ASC)
    )
    if held and held[0].sequence == position and _event_mark(held[0]) != mark:
        raise CloudActivityCursorStale("The store no longer holds the activity cursor's event.")
    return position, mark


def _scan_query(after_sequence: int, limit: int) -> EventQuery:
    return EventQuery(
        event_types=_ACTIVITY_EVENT_TYPES,
        after_sequence=after_sequence,
        limit=limit,
        order_by=EventOrder.SEQUENCE_ASC,
    )


async def read_cloud_activity(
    app: CayuApp,
    *,
    cursor: str | None = None,
    limit: int = CLOUD_ACTIVITY_DEFAULT_LIMIT,
) -> CloudActivityPage:
    """Read one page of the agent's activity after ``cursor``, or from the start.

    A page holds at most ``limit`` touches, and at most ``limit`` runs unless
    its first change alone affects more: a change reports its session and every
    session above it whose status can follow it. It reads the store's
    cross-session event order, which never moves past an event a concurrent
    writer has yet to commit. Raises ``CloudActivityCursorStale`` for a cursor
    this store cannot continue.
    """

    if type(limit) is not int or not 1 <= limit <= CLOUD_ACTIVITY_MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {CLOUD_ACTIVITY_MAX_LIMIT}.")
    store_id = await _store_identity(app)
    position, mark = await _cursor_position(app, store_id, cursor)
    last: EventRecord | None = None
    store = app.session_store
    reader = SessionOverviewReader(store, pricing=_bundled_price_book())
    changed: dict[str, None] = {}
    anchors: list[EventRecord] = []
    scanned = 0
    has_more = False
    while True:
        batch_limit = min(_SCAN_BATCH, _SCAN_BUDGET - scanned)
        if batch_limit == 0:
            has_more = await store.event_exists(_scan_query(position, 1))
            break
        batch = await store.query_events(_scan_query(position, batch_limit))
        scanned += len(batch)
        for record in batch:
            anchor = record.event.type in _ANCHOR_EVENT_TYPES
            affected = [
                session_id
                for session_id in await reader.affected_by(record.event.session_id)
                if session_id not in changed
            ]
            if (changed and len(changed) + len(affected) > limit) or (
                anchor and len(anchors) == limit
            ):
                # The page is full; this record starts the next one. A page always
                # takes its first change whole, with every summary it affects.
                has_more = True
                break
            changed.update(dict.fromkeys(affected))
            if anchor:
                anchors.append(record)
            position, last = record.sequence, record
        if has_more or len(batch) < batch_limit:
            break
    touches = await classify_supervision_anchors(store, anchors)
    return CloudActivityPage(
        runs=tuple(_project_run(app, facts) for facts in await reader.read(tuple(changed))),
        touches=tuple(_touch(app, session, event) for session, event in touches),
        next_cursor=_cursor(store_id, position, mark if last is None else _event_mark(last)),
        has_more=has_more,
    )


@cache
def _bundled_price_book() -> PriceBook:
    return default_price_book()


def _project_run(app: CayuApp, facts: SessionOverviewFacts) -> CloudRun:
    session, outcome, status = facts.session, facts.outcome, facts.status
    usage, cost, release_id = facts.usage, facts.cost, facts.release_id
    origin = session.invocation.origin
    root_run_id, parent_run_id = session_run_lineage(session)
    marker = session.metadata.get(RUN_LINEAGE_METADATA_KEY)
    parent_session_id = (
        marker["parent_session_id"] if isinstance(marker, dict) else session.parent_session_id
    )
    return CloudRun(
        run_id=session_run_id(session),
        root_run_id=root_run_id,
        parent_run_id=parent_run_id,
        session_id=app.project_session_id_for_exposure(session.id),
        root_session_id=app.project_session_id_for_exposure(session.invocation.root_session_id),
        parent_session_id=(
            None
            if parent_session_id is None
            else app.project_session_id_for_exposure(parent_session_id)
        ),
        source=session.invocation.source.value,
        forked=facts.forked,
        fork_controller=_text(app, facts.fork_controller),
        eval=is_eval_trial(session),
        agent_name=_redacted(app, session.agent_name),
        status=status,
        reason=_redacted(app, outcome.reason),
        started_by=(
            None
            if origin.subject is None
            else CloudPrincipal(subject=_redacted(app, origin.subject), trust=origin.trust.value)
        ),
        started_at=session.created_at,
        ended_at=facts.ended_at,
        release_id=_text(app, release_id),
        model_steps=0 if usage is None else usage.model_steps,
        tool_calls=0 if usage is None else usage.tool_calls,
        cost=_cost(cost),
        waiting_for=tuple(_waiting_item(app, action, child) for action, child in facts.waiting),
    )


def _cost(totals: SessionCostTotals | None) -> CloudRunCost:
    if totals is None:
        return CloudRunCost(amount=Decimal(0), currency="USD", complete=True)
    return CloudRunCost(
        amount=totals.total_cost,
        currency=totals.currency,
        complete=totals.unpriced_model_steps == 0 and totals.unpriced_auxiliary_attempts == 0,
    )


def _waiting_item(
    app: CayuApp, action: PendingActionRecord, child: Session | None
) -> CloudWaitingItem:
    reference = action.delegated_action
    return CloudWaitingItem(
        kind=action.kind.value,
        attention_id=action.attention_id,
        tool_name=_text(app, action.tool_name),
        requested_at=action.event.event.timestamp,
        child_run_id=None if child is None else session_run_id(child),
        child_session_id=(
            None
            if reference is None
            else app.project_session_id_for_exposure(reference.child_session_id)
        ),
    )


# Steering another session or a peer sent; neither is a person stepping in.
_MACHINE_MESSAGE_ORIGINS = frozenset(
    {SupervisionMessageOrigin.SESSION, SupervisionMessageOrigin.PEER}
)


def _stepped_in(event: SupervisionEvent) -> bool:
    if event.kind is SupervisionEventKind.EXPIRE:
        return False
    if event.actor is not None and event.actor.source is ResolutionActorSource.SYSTEM:
        return False
    if event.kind is SupervisionEventKind.STEER:
        # A message without recorded provenance counts, like an unattributed answer.
        return event.message_origin not in _MACHINE_MESSAGE_ORIGINS
    return True


def _touch(app: CayuApp, session: Session, event: SupervisionEvent) -> CloudTouch:
    actor = event.actor
    root_run_id, parent_run_id = session_run_lineage(session)
    return CloudTouch(
        touch_id=event.supervision_id,
        run_id=session_run_id(session),
        root_run_id=root_run_id,
        parent_run_id=parent_run_id,
        session_id=app.project_session_id_for_exposure(event.session_id),
        agent_name=_text(app, event.agent_name),
        kind=event.kind.value,
        outcome=event.outcome.value,
        stepped_in=_stepped_in(event),
        by=(
            None
            if actor is None
            else CloudActor(
                subject=_redacted(app, actor.subject),
                source=None if actor.source is None else actor.source.value,
            )
        ),
        reason=_text(app, event.reason),
        tool_name=_text(app, event.tool_name),
        attention_id=event.attention_id,
        requested_at=event.requested_at,
        settled_at=event.resolved_at,
        release_id=_text(app, event.release_id),
    )


def _redacted(app: CayuApp, value: str) -> str:
    """Text as the control plane shows it: every configured secret redacted."""

    return cast("str", app.redact_json(value))


def _text(app: CayuApp, value: str | None) -> str | None:
    return None if value is None else _redacted(app, value)


__all__ = [
    "CLOUD_ACTIVITY_CURSOR_MAX_CHARS",
    "CLOUD_ACTIVITY_DEFAULT_LIMIT",
    "CLOUD_ACTIVITY_MAX_LIMIT",
    "CLOUD_AGENT_CONTRACT",
    "CLOUD_AGENT_CONTRACT_VERSIONS",
    "CloudActivityCursorInvalid",
    "CloudActivityCursorStale",
    "CloudActivityPage",
    "CloudActor",
    "CloudContract",
    "CloudPrincipal",
    "CloudRun",
    "CloudRunCost",
    "CloudRunStatus",
    "CloudTouch",
    "CloudWaitingItem",
    "read_cloud_activity",
    "read_cloud_contract",
]
