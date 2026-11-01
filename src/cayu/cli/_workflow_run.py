"""One workflow run as a whole: its steps, their child sessions, usage and custom events.

`cayu session workflow SESSION_ID` reads a workflow run through a `SessionReader`, so a local
store and a Cloud Agent give the same answer. It composes reads the other `cayu session`
commands already make:

- the run's own events (`query_events`), for the `workflow.*` journal that names each step's
  child session and for the `custom.*` events the workflow emitted;
- the session lineage (`list_sessions` filtered by `parent_session_id`, the filter the
  topology view and `GET /api/sessions` use), walked breadth-first from the run so children
  of children count;
- each session's model-call aggregate (`model_calls`, what `cayu session usage` reports).

A step's numbers cover its child session and everything under it. The run's totals cover
the workflow session and every session reached under it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from cayu.budgets.pricing import PriceBook
from cayu.budgets.usage import (
    AggregateUsageMetrics,
    add_aggregate_usage,
    build_aggregate_usage_metrics,
)
from cayu.cli._session_reader import SessionReader
from cayu.events import EventType
from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY
from cayu.sessions._call_inspection import InspectionWindow, currency_totals_text
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import EventRecord

_EVENT_PAGE = 200
_CHILD_PAGE = 200
_CUSTOM_PREFIX = "custom."
_RESERVED_CUSTOM_PREFIX = "custom.cayu."
_TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})
# A command step's journal kind (`cayu.workflows.commands.COMMAND_STEP_KIND`).
_COMMAND_KIND = "command"
_LIVE_STATUSES = frozenset({"pending", "running", "interrupting"})


@dataclass
class SessionUsage:
    """Model-call counters of one session, or summed over several."""

    sessions: int = 0
    model_calls: int = 0
    model_calls_with_usage: int = 0
    tool_calls: int = 0
    usage: AggregateUsageMetrics = field(default_factory=build_aggregate_usage_metrics)
    cost: dict[str, Decimal] = field(default_factory=dict)
    # None once any summed session couldn't be priced at all (an Agent that doesn't price).
    unpriced_calls: int | None = 0
    statuses: dict[str, int] = field(default_factory=dict)

    def add(self, other: SessionUsage) -> None:
        self.sessions += other.sessions
        self.model_calls += other.model_calls
        self.model_calls_with_usage += other.model_calls_with_usage
        self.tool_calls += other.tool_calls
        self.usage = add_aggregate_usage(self.usage, other.usage)
        for currency, amount in other.cost.items():
            self.cost[currency] = self.cost.get(currency, Decimal(0)) + amount
        if self.unpriced_calls is None or other.unpriced_calls is None:
            self.unpriced_calls = None
        else:
            self.unpriced_calls += other.unpriced_calls
        for status, count in other.statuses.items():
            self.statuses[status] = self.statuses.get(status, 0) + count

    def estimated_cost(self, *, priced: bool) -> dict[str, str] | None:
        if not priced or self.unpriced_calls is None:
            return None
        return currency_totals_text(self.cost)


@dataclass
class WorkflowStep:
    """One journaled step (or gated-loop item) and the child session it ran in."""

    step_id: str | None
    kind: str
    agent: str | None
    child_session_id: str | None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    outcome: str | None = None
    has_output: bool | None = None
    # A command step's exit code; its outcome is passed, failed, timed_out or cancelled.
    exit_code: int | None = None
    # A gated-loop item's loop and item key; its step_id is derived from them.
    loop_name: str | None = None
    item_key: str | None = None


@dataclass
class _Journal:
    name: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    steps: list[WorkflowStep] = field(default_factory=list)
    ids_withheld: bool = False


@dataclass
class SessionNode:
    id: str
    agent: str | None
    status: str | None
    created_at: str | None
    last_activity_at: str | None
    children: list[str] = field(default_factory=list)


@dataclass
class WorkflowRun:
    session: dict[str, Any]
    name: str | None
    started_at: datetime | None
    completed_at: datetime | None
    steps: list[WorkflowStep]
    # The reader's event view withheld journal step and child ids (an Agent's public API).
    step_ids_withheld: bool
    custom_events: list[EventRecord]
    nodes: dict[str, SessionNode]
    # The subtree under (and including) each reached session.
    subtree_usage: dict[str, SessionUsage]
    totals: SessionUsage
    # Direct children of the run that no journaled step names.
    unlinked_children: list[str]
    sessions_complete: bool
    events_complete: bool


async def read_workflow_run(
    reader: SessionReader,
    session_id: str,
    *,
    pricing: PriceBook | None,
    max_sessions: int,
    max_events: int,
) -> WorkflowRun:
    """Read a workflow run's journal, session tree and usage through ``reader``."""

    overview = await reader.overview(session_id)
    session = overview["session"]
    journal, custom_events, events_complete = await _read_run_events(
        reader, session_id, max_events=max_events
    )
    parsed = _steps_from_journal(journal)
    steps = parsed.steps

    nodes = {session_id: _node_from_session(session_id, session)}
    roots = [session_id]
    sessions_complete = await _walk(reader, nodes, session_id, max_sessions=max_sessions)
    # A custom journal may name a child session without recording lineage; read it too.
    for step in steps:
        child_id = step.child_session_id
        if child_id is None or child_id in nodes or not sessions_complete:
            continue
        if len(nodes) >= max_sessions:
            sessions_complete = False
            break
        try:
            child = (await reader.overview(child_id))["session"]
        except ValueError:
            continue
        nodes[child_id] = _node_from_session(child_id, child)
        roots.append(child_id)
        sessions_complete = await _walk(reader, nodes, child_id, max_sessions=max_sessions)

    own_usage = {
        node_id: await _session_usage(reader, node, pricing=pricing)
        for node_id, node in nodes.items()
    }
    # Roots' subtrees are disjoint: the walks never revisit a session.
    subtree_usage: dict[str, SessionUsage] = {}
    totals = SessionUsage()
    for root in roots:
        totals.add(_sum_subtree(root, nodes, own_usage, subtree_usage))

    named = {step.child_session_id for step in steps}
    unlinked = [child for child in nodes[session_id].children if child not in named]
    return WorkflowRun(
        session=session,
        name=parsed.name,
        started_at=parsed.started_at,
        completed_at=parsed.completed_at,
        steps=steps,
        step_ids_withheld=parsed.ids_withheld,
        custom_events=custom_events,
        nodes=nodes,
        subtree_usage=subtree_usage,
        totals=totals,
        unlinked_children=unlinked,
        sessions_complete=sessions_complete,
        events_complete=events_complete,
    )


def _node_from_session(session_id: str, session: dict[str, Any]) -> SessionNode:
    """A tree node from a `SessionReader.overview` session section."""

    return SessionNode(
        id=session_id,
        agent=session.get("agent"),
        status=session.get("status"),
        created_at=session.get("created_at"),
        last_activity_at=session.get("last_activity_at"),
    )


def run_status(run: WorkflowRun) -> str | None:
    """The run's own status, from its journal rather than its anchor session.

    The shipped journal marks its anchor session completed before the run starts, so the
    anchor's status says nothing about the run. ``completed`` means the journal recorded
    ``workflow.completed`` for the latest attempt (a rerun under the same session id starts
    a new one); ``running`` that it didn't yet and a session in the run is still live;
    ``stopped`` that it didn't and no session in the run is live: a step raised, the run
    was cut off, or its process is between steps or running a ``command()`` step, which
    has no session (its row's journal stays ``started`` until it ends; the journal can't
    tell a command still running from a process that died running it). A session without a
    journal reports its own status.
    """

    if run.completed_at is not None:
        return "completed"
    if run.started_at is None and not run.steps:
        return run.session.get("status")
    if any(node.status in _LIVE_STATUSES for node in run.nodes.values()):
        return "running"
    return "stopped"


async def _read_run_events(
    reader: SessionReader, session_id: str, *, max_events: int
) -> tuple[list[EventRecord], list[EventRecord], bool]:
    """The run's `workflow.*` journal and `custom.*` events, in sequence order."""

    journal_types = {
        str(EventType.WORKFLOW_STARTED),
        str(EventType.WORKFLOW_STEP_STARTED),
        str(EventType.WORKFLOW_STEP_COMPLETED),
        str(EventType.WORKFLOW_COMPLETED),
    }
    journal: list[EventRecord] = []
    custom: list[EventRecord] = []
    after: int | None = None
    read = 0
    while read < max_events:
        limit = min(_EVENT_PAGE, max_events - read)
        records = await reader.query_events(
            EventQuery(
                session_id=session_id,
                after_sequence=after,
                limit=limit,
                order_by=EventOrder.SEQUENCE_ASC,
            )
        )
        for record in records:
            event_type = str(record.event.type)
            if event_type in journal_types:
                journal.append(record)
            elif event_type.startswith(_CUSTOM_PREFIX) and not event_type.startswith(
                _RESERVED_CUSTOM_PREFIX
            ):
                # custom.cayu.* are Cayu's own markers, such as workflow attempt fences.
                custom.append(record)
        read += len(records)
        if len(records) < limit:
            return journal, custom, True
        after = records[-1].sequence
    # Stopped at the cap; the run is complete only if nothing follows.
    more = await reader.query_events(
        EventQuery(session_id=session_id, after_sequence=after, limit=1)
    )
    return journal, custom, not more


def _steps_from_journal(journal: list[EventRecord]) -> _Journal:
    parsed = _Journal()
    steps: dict[tuple[str, str | None], WorkflowStep] = {}
    # A gated-loop item learns its child session only when it completes.
    open_loop_items: dict[object, WorkflowStep] = {}
    # A command step runs in no child session; its completion pairs with its start by step id,
    # or, where the event view withholds step ids, by its argv and environment.
    open_commands: dict[object, list[WorkflowStep]] = {}
    for record in journal:
        event = record.event
        payload = event.payload
        if event.type in {EventType.WORKFLOW_STARTED, EventType.WORKFLOW_COMPLETED}:
            parsed.name = parsed.name or _text(payload.get("workflow")) or event.workflow_name
            if event.type == EventType.WORKFLOW_STARTED:
                if parsed.started_at is None or parsed.completed_at is not None:
                    # The first start, or a rerun of a completed run: only the new attempt's
                    # end counts. A resume of an unfinished run keeps its first start.
                    parsed.started_at = event.timestamp
                    parsed.completed_at = None
            else:
                parsed.completed_at = event.timestamp
            continue
        step_id = _text(payload.get("step_id"))
        child_id = _text(payload.get("child_session_id"))
        if _withheld(payload.get("step_id")) or _withheld(payload.get("child_session_id")):
            parsed.ids_withheld = True
        agent = _text(payload.get("agent")) or event.agent_name
        if payload.get("kind") == "gated_loop":
            loop_name = _text(payload.get("loop_name"))
            item_key = _text(payload.get("item_key"))
            # Withheld step ids still pair by the loop's name and item key.
            item = step_id or (loop_name, item_key)
            step = open_loop_items.get(item)
            if step is None:
                step = WorkflowStep(
                    step_id=step_id,
                    kind="gated_loop",
                    agent=agent,
                    child_session_id=None,
                    loop_name=loop_name,
                    item_key=item_key,
                )
                steps[(f"gated_loop:{record.sequence}", None)] = step
            if event.type == EventType.WORKFLOW_STEP_STARTED:
                step.started_at = event.timestamp
                open_loop_items[item] = step
            else:
                open_loop_items.pop(item, None)
                step.child_session_id = child_id
                step.completed_at = event.timestamp
                step.outcome = _text(payload.get("outcome"))
            continue
        if payload.get("kind") == _COMMAND_KIND:
            identity = _command_identity(step_id, payload)
            if event.type == EventType.WORKFLOW_STEP_STARTED:
                # Each run of a command (a rerun too) is its own row.
                step = WorkflowStep(
                    step_id=step_id,
                    kind=_COMMAND_KIND,
                    agent=agent,
                    child_session_id=None,
                    started_at=event.timestamp,
                )
                steps[(f"command:{record.sequence}", None)] = step
                if identity is not None:
                    open_commands.setdefault(identity, []).append(step)
                continue
            waiting = [] if identity is None else open_commands.get(identity, [])
            if len(waiting) == 1 or (waiting and step_id is not None):
                # A step id runs once at a time, so its latest start is this run's; a start
                # an earlier attempt never completed stays open.
                step = waiting.pop()
            else:
                # No start, or several that withheld ids can't tell apart: the completion is
                # a row of its own rather than another run's end.
                step = WorkflowStep(
                    step_id=step_id, kind=_COMMAND_KIND, agent=agent, child_session_id=None
                )
                steps[(f"command:{record.sequence}", None)] = step
            step.completed_at = event.timestamp
            step.outcome = _command_outcome(payload)
            exit_code = payload.get("exit_code")
            step.exit_code = exit_code if type(exit_code) is int else None
            continue
        if step_id is None or child_id is None:
            # Without both ids a step can't be told from a rerun; its child session is
            # still listed among the run's children.
            continue
        key = (step_id, child_id)
        step = steps.get(key)
        if step is None:
            step = steps[key] = WorkflowStep(
                step_id=step_id,
                kind="step",
                agent=agent,
                child_session_id=child_id,
            )
        if event.type == EventType.WORKFLOW_STEP_STARTED:
            step.started_at = step.started_at or event.timestamp
        else:
            step.completed_at = event.timestamp
            has_output = payload.get("has_output")
            step.has_output = has_output if isinstance(has_output, bool) else None
    # The journal is read in sequence order, so steps are in order of first appearance.
    parsed.steps = list(steps.values())
    return parsed


async def _walk(
    reader: SessionReader,
    nodes: dict[str, SessionNode],
    root: str,
    *,
    max_sessions: int,
) -> bool:
    """Add every session under ``root`` to ``nodes``; False if ``max_sessions`` stopped it."""

    pending = deque([root])
    while pending:
        parent_id = pending.popleft()
        cursor: str | None = None
        while True:
            page = await reader.list_sessions(
                SessionQuery(
                    parent_session_id=parent_id,
                    order_by=SessionOrder.CREATED_AT_ASC,
                    limit=_CHILD_PAGE,
                    cursor=cursor,
                )
            )
            for row in page.rows:
                child_id = row.get("id")
                if not isinstance(child_id, str) or child_id in nodes:
                    continue
                if len(nodes) >= max_sessions:
                    return False
                nodes[child_id] = SessionNode(
                    id=child_id,
                    agent=row.get("agent"),
                    status=row.get("status"),
                    created_at=row.get("created_at"),
                    # An Agent's session list may leave last activity out.
                    last_activity_at=row.get("last_activity_at") or row.get("updated_at"),
                )
                nodes[parent_id].children.append(child_id)
                pending.append(child_id)
            cursor = page.next_cursor
            if cursor is None or not page.rows:
                break
    return True


async def _session_usage(
    reader: SessionReader, node: SessionNode, *, pricing: PriceBook | None
) -> SessionUsage:
    result = await reader.model_calls(
        node.id, InspectionWindow(), offset=0, limit=1, pricing=pricing
    )
    aggregate = result["aggregate"]
    cost = aggregate.get("estimated_cost")
    unpriced = aggregate.get("unpriced_calls")
    return SessionUsage(
        sessions=1,
        model_calls=int(aggregate.get("model_calls") or 0),
        model_calls_with_usage=int(aggregate.get("model_calls_with_usage") or 0),
        tool_calls=int(aggregate.get("tool_calls") or 0),
        usage=aggregate["usage"],
        cost={str(k): Decimal(str(v)) for k, v in cost.items()} if isinstance(cost, dict) else {},
        unpriced_calls=unpriced if type(unpriced) is int and isinstance(cost, dict) else None,
        statuses={node.status: 1} if node.status else {},
    )


def _sum_subtree(
    node_id: str,
    nodes: dict[str, SessionNode],
    own: dict[str, SessionUsage],
    out: dict[str, SessionUsage],
) -> SessionUsage:
    # Iterative post-order, so a deep lineage can't exhaust the recursion limit.
    stack: list[tuple[str, bool]] = [(node_id, False)]
    while stack:
        current, expanded = stack.pop()
        if current in out:
            continue
        if not expanded:
            stack.append((current, True))
            stack.extend((child, False) for child in nodes[current].children)
            continue
        total = SessionUsage()
        total.add(own[current])
        for child in nodes[current].children:
            total.add(out[child])
        out[current] = total
    return out[node_id]


def step_times(step: WorkflowStep, node: SessionNode | None) -> tuple[str | None, str | None]:
    """When a step started and ended: its journal times, else its child session's."""

    started = step.started_at.isoformat() if step.started_at is not None else None
    if started is None and node is not None:
        started = node.created_at
    ended = step.completed_at.isoformat() if step.completed_at is not None else None
    if ended is None and node is not None:
        ended = session_ended_at(node)
    return started, ended


def session_ended_at(node: SessionNode) -> str | None:
    """A finished session's last activity; None while it may still run."""

    return node.last_activity_at if node.status in _TERMINAL_STATUSES else None


def _command_identity(step_id: str | None, payload: dict[str, Any]) -> object | None:
    """What pairs a command step's start with its completion: its step id (one run of a step
    id at a time), else, with the id withheld, its argv and environment; None without either."""

    if step_id is not None:
        return step_id
    argv = payload.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return None
    environment = payload.get("environment_name")
    return ("withheld", tuple(argv), environment if isinstance(environment, str) else None)


def _command_outcome(payload: dict[str, Any]) -> str | None:
    """A command step's outcome from its completion: passed, timed_out, cancelled or failed."""

    if payload.get("timed_out") is True:
        return "timed_out"
    if payload.get("cancelled") is True:
        return "cancelled"
    passed = payload.get("passed")
    if isinstance(passed, bool):
        return "passed" if passed else "failed"
    exit_code = payload.get("exit_code")
    if type(exit_code) is int:
        return "passed" if exit_code == 0 else "failed"
    return None


def _text(value: object) -> str | None:
    if not isinstance(value, str) or not value or _withheld(value):
        return None
    return value


def _withheld(value: object) -> bool:
    return value == PRIVATE_EVENT_AUTHORITY
