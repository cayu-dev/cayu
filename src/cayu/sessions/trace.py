"""Read-only, sequence-paged trace topology from private durable event identity.

Only durable sequence anchors leave this module as span identifiers. Content is
provided by the application's existing event and transcript projectors.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.events import Event, EventType
from cayu.runtime._event_projection import public_event_id, public_event_sequence
from cayu.sessions._correlation import _schema_identity
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.records import EventRecord, Session
from cayu.sessions.store import SessionStore
from cayu.sessions.trace_events import TraceEvent, trace_event_of
from cayu.tools.outcomes import _TOOL_CALL_TERMINAL_DISPOSITIONS

_PAGE_SIZE = 5000
Key = tuple[str, ...]
_ROOT: Key = ("session",)
_SESSION_EVENTS = (
    EventType.SESSION_STARTED,
    EventType.SESSION_RESUMED,
    EventType.SESSION_COMPLETED,
    EventType.SESSION_FAILED,
    EventType.SESSION_INTERRUPTED,
)
_WAIT_EVENTS = (
    EventType.SESSION_AWAITING_USER_INPUT,
    EventType.SESSION_RESUMED,
    EventType.TOOL_CALL_APPROVAL_REQUESTED,
    EventType.TOOL_CALL_APPROVED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.TOOL_CALL_APPROVAL_EXPIRED,
)


def _wait_key(event: Event) -> Key | None:
    # These exact durable event families own supervision identity; a similarly
    # named field on arbitrary workload events is not correlation evidence.
    if event.type not in _WAIT_EVENTS:
        return None
    field = (
        "input_id"
        if event.type in {EventType.SESSION_AWAITING_USER_INPUT, EventType.SESSION_RESUMED}
        else "approval_id"
    )
    if (
        event.type == EventType.SESSION_RESUMED
        and event.payload.get("interruption_type") != "user_input_required"
    ):
        return None
    value = event.payload.get(field)
    return ("wait", field, value) if type(value) is str and value.strip() else None


_INTERACTION_EVENTS = tuple(event for event in EventType if event.value.startswith("interaction."))
_STARTS: dict[str, tuple[EventType, ...]] = {
    "session": (EventType.SESSION_STARTED, EventType.SESSION_RESUMED),
    "interaction": (EventType.INTERACTION_STARTED,),
    "model": (EventType.MODEL_STARTED,),
    "round": (EventType.MODEL_COMPLETED,),
    "tool": (EventType.TOOL_CALL_STARTED,),
    "exec": (EventType.RUNNER_EXEC_STARTED,),
    "knowledge": (EventType.KNOWLEDGE_LOOKUP,),
    "wait": (EventType.SESSION_AWAITING_USER_INPUT, EventType.TOOL_CALL_APPROVAL_REQUESTED),
}


class TraceReadConflict(ValueError):
    """The session incarnation changed while its trace was being read."""


class TraceSpanNotFound(LookupError):
    """No span with this ID in the session's retained trace."""


def _model_step(event: Event) -> str | None:
    identity = _schema_identity(event, "model_step_id")
    if identity is not None:
        return identity
    # User-input observations carry model identity as a documented observation,
    # not as public mutation authority. Read it only on the durable request
    # event, and retain the independently owned round/call tuple below.
    if event.type == EventType.SESSION_AWAITING_USER_INPUT:
        value = event.payload.get("model_step_id")
        if type(value) is str and value.strip():
            return value
    return None


def _path(event: Event) -> list[Key]:
    path = [_ROOT]
    if event.interaction_id:
        path.append(("interaction", event.interaction_id))
    step = _model_step(event)
    round_id = _schema_identity(event, "tool_round_id")
    call_id = _schema_identity(event, "tool_call_id")
    execution = _schema_identity(event, "execution_id")
    if step:
        path.append(("model", step))
    if round_id and step:
        path.append(("round", round_id))
    if call_id and round_id and step:
        path.append(("tool", round_id, call_id))
    if execution and call_id and round_id and step:
        path.append(("exec", round_id, call_id, execution))
    if event.type == EventType.KNOWLEDGE_LOOKUP:
        path.append(("knowledge", event.id))
    if wait := _wait_key(event):
        path.append(wait)
    return path


def _terminal(kind: str, event: Event) -> str | None:
    try:
        event_type = EventType(event.type)
    except ValueError:
        return None
    if kind == "knowledge" and event_type == EventType.KNOWLEDGE_LOOKUP:
        return "completed"
    if kind == "wait":
        return {
            EventType.SESSION_RESUMED: "completed",
            EventType.TOOL_CALL_APPROVED: "completed",
            EventType.TOOL_CALL_APPROVAL_DENIED: "blocked",
            EventType.TOOL_CALL_APPROVAL_EXPIRED: "blocked",
        }.get(event_type)
    if kind == "session":
        return {
            EventType.SESSION_COMPLETED: "completed",
            EventType.SESSION_FAILED: "failed",
            EventType.SESSION_INTERRUPTED: "interrupted",
        }.get(event_type)
    if kind == "interaction":
        return {
            EventType.INTERACTION_COMPLETED: "completed",
            EventType.INTERACTION_FAILED: "failed",
            EventType.INTERACTION_INTERRUPTED: "interrupted",
        }.get(event_type)
    if kind == "model":
        if event_type == EventType.MODEL_COMPLETED:
            return "completed"
        if event_type == EventType.MODEL_ERROR:
            return "failed"
    if kind == "tool":
        if (
            event_type in _TOOL_CALL_TERMINAL_DISPOSITIONS
            and event.payload.get("interrupted") is True
        ):
            return "interrupted"
        return _TOOL_CALL_TERMINAL_DISPOSITIONS.get(event_type)
    if kind == "exec" and event_type == EventType.RUNNER_EXEC_COMPLETED:
        payload = event.payload
        if payload.get("cancelled") is True:
            return "interrupted"
        if (
            payload.get("timed_out") is True
            or (type(payload.get("error_type")) is str and bool(payload["error_type"].strip()))
            or (type(payload.get("exit_code")) is int and payload["exit_code"] != 0)
        ):
            return "failed"
        if type(payload.get("exit_code")) is int:
            return "completed"
    return None


@dataclass
class _Node:
    key: Key
    parent: Key | None
    anchor: int
    started_at: datetime
    name: str
    last_sequence: int
    name_sequence: int | None = None
    first_start_sequence: int | None = None
    ended_at: datetime | None = None
    status: str = "unknown"
    transcript_links: list[dict[str, Any]] = field(default_factory=list)
    detail: str | None = None

    @property
    def id(self) -> str:
        return "session" if self.key == _ROOT else f"{self.key[0]}:{self.anchor}"


#: Characters of a tool call's argument preview.
DETAIL_MAX_CHARS = 200


def _argument_preview(arguments: Any) -> str | None:
    """The first non-empty string argument, else the arguments as compact JSON, cut short."""

    if type(arguments) is not dict:
        return None
    first = next((value for value in arguments.values() if type(value) is str and value), None)
    text = (
        first
        if first is not None
        else json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    )
    return None if text == "{}" else text[:DETAIL_MAX_CHARS]


class SessionTraceReader:
    def __init__(
        self,
        store: SessionStore,
        *,
        project_event: Callable[[EventRecord], dict[str, Any]],
        project_part: Callable[[Any], dict[str, Any]],
        project_session: Callable[[str], str],
        project_tool_name: Callable[[str], str],
    ) -> None:
        """``project_tool_name`` must give the ``tool_name`` that ``project_event`` gives."""

        self.store = store
        self.project_event = project_event
        self.project_part = project_part
        self.project_session = project_session
        self.project_tool_name = project_tool_name

    async def _history(self, query: EventQuery) -> list[TraceEvent]:
        """Every trace event ``query`` matches, read in bounded pages."""

        rows: list[TraceEvent] = []
        after = query.after_sequence
        while True:
            page = await self.store.query_trace_events(
                query.model_copy(update={"after_sequence": after, "limit": _PAGE_SIZE})
            )
            rows.extend(page)
            if len(page) < _PAGE_SIZE:
                return rows
            next_after = page[-1].sequence
            if after is not None and next_after <= after:
                raise TraceReadConflict("Trace event traversal did not advance.")
            after = next_after

    async def read(
        self,
        session_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: int | None = None,
        session_instance_id: str | None = None,
        limit: int = 100,
        include_events: bool = True,
    ) -> dict[str, Any] | None:
        """Read one page of a session's trace.

        Without ``include_events`` the response carries the page's spans and
        their event IDs but no events, and no event payload is read.
        """

        session = await self.store.load(session_id)
        if session is None:
            return None
        if session_instance_id is not None and session_instance_id != session.instance_id:
            raise TraceReadConflict("Session incarnation no longer matches the trace cursor.")
        try:
            return await self._read_snapshot(
                session,
                after_sequence=after_sequence,
                through_sequence=through_sequence,
                limit=limit,
                include_events=include_events,
            )
        except KeyError:
            # Native transcript readers signal disappearance before the final
            # publication check. Reconcile against positive incarnation
            # evidence; unrelated store errors retain their original meaning.
            await self._require_current_instance(session)
            raise

    async def read_span_events(
        self,
        session_id: str,
        span_id: str,
        *,
        through_sequence: int | None = None,
        session_instance_id: str | None = None,
        include_children: bool = False,
    ) -> dict[str, Any] | None:
        """One span's events, whole, in sequence order, with its direct children's on request.

        The span is one a trace read of the whole retained history through
        ``through_sequence`` gives. Returns ``None`` when the session does not
        exist and raises :class:`TraceSpanNotFound` when the span does not.
        Only the returned events' payloads are read.
        """

        session = await self.store.load(session_id)
        if session is None:
            return None
        if session_instance_id is not None and session_instance_id != session.instance_id:
            raise TraceReadConflict("Session incarnation no longer matches the trace cursor.")
        try:
            whole = await self._read_snapshot(
                session,
                after_sequence=0,
                through_sequence=through_sequence,
                limit=MAX_DURABLE_JSON_INTEGER,
                include_events=False,
            )
        except KeyError:
            await self._require_current_instance(session)
            raise
        spans = whole["spans"]
        span = next((item for item in spans if item["id"] == span_id), None)
        if span is None:
            raise TraceSpanNotFound(span_id)
        event_ids = list(span["event_ids"])
        if include_children:
            event_ids.extend(
                event_id
                for item in spans
                if item["parent_id"] == span_id
                for event_id in item["event_ids"]
            )
        sequences = sorted(
            {sequence for event_id in event_ids if (sequence := public_event_sequence(event_id))}
        )
        records: list[EventRecord] = []
        # Contiguous runs, such as a model step's text deltas, read as one range.
        start = 0
        while start < len(sequences):
            end = start
            while (
                end + 1 < len(sequences)
                and sequences[end + 1] == sequences[end] + 1
                and end + 1 - start < _PAGE_SIZE
            ):
                end += 1
            first, last = sequences[start], sequences[end]
            records.extend(
                await self.store.query_events(
                    EventQuery(
                        session_id=session_id,
                        after_sequence=first - 1,
                        before_sequence=last + 1 if last < MAX_DURABLE_JSON_INTEGER else None,
                        limit=last - first + 1,
                    )
                )
            )
            start = end + 1
        await self._require_current_instance(session)
        return {
            "session_id": self.project_session(session_id),
            "session_instance_id": session.instance_id,
            "span_id": span_id,
            "through_sequence": whole["through_sequence"],
            "events": [self.project_event(record) for record in records],
        }

    async def _require_current_instance(self, session: Session) -> None:
        current = await self.store.load(session.id)
        if current is None or current.instance_id != session.instance_id:
            raise TraceReadConflict("Session changed while reading its trace.") from None

    async def _read_snapshot(
        self,
        session: Session,
        *,
        after_sequence: int,
        through_sequence: int | None,
        limit: int,
        include_events: bool,
    ) -> dict[str, Any]:
        session_id = session.id
        latest = await self.store.query_event_heads(
            EventQuery(session_id=session_id, order_by=EventOrder.SEQUENCE_DESC, limit=1)
        )
        newest = latest[0].sequence if latest else 0
        through = newest if through_sequence is None else min(through_sequence, newest)
        query = EventQuery(
            session_id=session_id,
            before_sequence=through + 1 if through < MAX_DURABLE_JSON_INTEGER else None,
        )
        # The trace reads each event's columns and the few payload members that
        # place it, never whole payloads. Every span selection below matches
        # what an indexed store query over the whole events would return.
        history = await self._history(query)
        # A cursor can outlive retained history, or deliberately lie beyond a
        # selected snapshot. It is an empty window, not an invalid native query.
        records: list[EventRecord] = []
        if after_sequence >= through:
            rows: list[TraceEvent] = []
        elif include_events:
            records = await self.store.query_events(
                query.model_copy(update={"after_sequence": after_sequence, "limit": limit + 1})
            )
            by_sequence = {row.sequence: row for row in history}
            rows = [
                by_sequence.get(record.sequence) or trace_event_of(record) for record in records
            ]
        else:
            rows = [row for row in history if row.sequence > after_sequence][: limit + 1]
        page, has_more = rows[:limit], len(rows) > limit
        nodes: dict[Key, _Node] = {}
        selected: set[Key] = {_ROOT}
        event_ids: dict[Key, list[str]] = {}
        public_events = (
            [self.project_event(record) for record in records[:limit]] if include_events else []
        )
        for row in page:
            path = _path(row.event)
            selected.update(path)
            event_ids.setdefault(path[-1], []).append(public_event_id(row.sequence))
        nodes[_ROOT] = _Node(_ROOT, None, 0, session.created_at, "Session", 0)
        tool_names: dict[str, str] = {}

        def public_tool_name(event: Event) -> str | None:
            name = event.tool_name
            if name is None:
                return None
            if name not in tool_names:
                tool_names[name] = self.project_tool_name(name)
            return tool_names[name]

        def collect(row: TraceEvent, *, path: list[Key] | None = None) -> None:
            event = row.event
            sequence = row.sequence
            tool_name = public_tool_name(event)
            parent = None
            for key in _path(event) if path is None else path:
                kind = key[0]
                name = {
                    "session": "Session",
                    "interaction": "Turn",
                    "model": "Model step",
                    "round": "Tool round",
                    "exec": "Command",
                    "knowledge": "Knowledge lookup",
                    "wait": "Human input"
                    if kind == "wait" and key[1] == "input_id"
                    else "Approval",
                }.get(kind, tool_name or "Tool call")
                node = nodes.get(key)
                if node is None:
                    node = nodes[key] = _Node(key, parent, sequence, event.timestamp, name, 0)
                elif sequence < node.anchor:
                    node.anchor, node.parent = sequence, parent
                    if node.first_start_sequence is None:
                        node.started_at = event.timestamp
                if (
                    kind == "tool"
                    and tool_name
                    and (node.name_sequence is None or sequence < node.name_sequence)
                ):
                    node.name = tool_name
                    node.name_sequence = sequence
                is_start = event.type in _STARTS.get(kind, ())
                if is_start and (
                    node.first_start_sequence is None or sequence < node.first_start_sequence
                ):
                    node.first_start_sequence = sequence
                    node.started_at = event.timestamp
                terminal = _terminal(kind, event)
                # Content observations must not supersede lifecycle evidence.
                # Context families are selected separately and can arrive here
                # out of sequence; only lifecycle records advance this cursor.
                if sequence >= node.last_sequence and (terminal is not None or is_start):
                    node.status = (
                        terminal
                        if terminal is not None
                        else ("waiting" if kind == "wait" else "running")
                    )
                    node.ended_at = event.timestamp if terminal is not None else None
                    node.last_sequence = sequence
                parent = key

        by_step: dict[str, list[TraceEvent]] = {}
        by_interaction: dict[str, list[TraceEvent]] = {}
        for row in history:
            # The same match as the stores' model_step_id selector: a text member.
            step = row.event.payload.get("model_step_id")
            if type(step) is str:
                by_step.setdefault(step, []).append(row)
            if row.event.interaction_id is not None:
                by_interaction.setdefault(row.event.interaction_id, []).append(row)

        if any(key[0] == "wait" for key in selected):
            for row in history:
                if row.event.type in _WAIT_EVENTS and _wait_key(row.event) in selected:
                    collect(row)

        # Resolve one model's private topology together.
        steps = {value for row in page if (value := _model_step(row.event))}
        steps.update(key[1] for key in nodes if key[0] == "model")
        for step in sorted(steps):
            for row in by_step.get(step, ()):
                collect(row)
        for row in page:
            collect(row)
        interactions = {key[1] for key in nodes if key[0] == "interaction"}
        for interaction in sorted(interactions):
            # A retained non-lifecycle observation may precede every retained
            # start/completion. Anchor the turn independently of this page,
            # without importing unrelated models or children into its context.
            turn = by_interaction.get(interaction, [])
            if turn:
                collect(turn[0], path=[_ROOT, ("interaction", interaction)])
            for row in turn:
                if row.event.type in _INTERACTION_EVENTS:
                    collect(row)
        for row in history:
            # Session observations can carry model identity; only the root's
            # lifecycle belongs to this independent read.
            if row.event.type in _SESSION_EVENTS:
                collect(row, path=[_ROOT])

        expected_round_calls: dict[Key, set[Key]] = {}
        # A transcript call's round binds reused provider call IDs exactly.
        # Read the tool calls of the interactions owned by the selected contexts.
        calls = await self.store.query_transcript_tool_calls(session_id, sorted(interactions))
        for call in sorted(
            calls, key=lambda item: (item.interaction_id or "", item.message_index, item.part_index)
        ):
            part = call.part
            if part.tool_round_id is None:
                continue
            round_node = nodes.get(("round", part.tool_round_id))
            if round_node is None or round_node.parent != ("model", part.model_step_id):
                continue
            call_key = ("tool", part.tool_round_id, part.tool_call_id)
            expected_round_calls.setdefault(round_node.key, set()).add(call_key)
            node = nodes.get(call_key)
            if node is None or node.parent != round_node.key:
                continue
            projected = self.project_part(part)
            if node.detail is None:
                node.detail = _argument_preview(projected.get("arguments"))
            node.transcript_links.append(
                {
                    "message_index": call.message_index,
                    "part_index": call.part_index,
                    "tool_call_id": projected["tool_call_id"],
                }
            )

        for round_key, calls in expected_round_calls.items():
            round_node = nodes[round_key]
            model = nodes.get(round_node.parent) if round_node.parent else None
            if model is None or model.status != "completed":
                continue
            # The completed model and its durable transcript enumerate the
            # round, rather than inferring completion from whatever calls a
            # single event page happened to contain.
            if any(key not in nodes for key in calls):
                round_node.status = "running"
                continue
            children = [nodes[key] for key in calls]
            if all(child.ended_at is not None for child in children):
                round_node.ended_at = max(child.ended_at for child in children if child.ended_at)
                round_node.status = next(
                    (
                        status
                        for status in ("failed", "interrupted", "blocked")
                        if any(child.status == status for child in children)
                    ),
                    "completed",
                )
            else:
                round_node.status = "running"

        for key in tuple(selected):
            parent = nodes.get(key)
            while parent is not None and parent.parent is not None:
                selected.add(parent.parent)
                parent = nodes.get(parent.parent)
        await self._require_current_instance(session)
        return {
            "session_id": self.project_session(session_id),
            "root_span_id": "session",
            "session_instance_id": session.instance_id,
            "through_sequence": through,
            "next_sequence": page[-1].sequence if page else after_sequence,
            "has_more": has_more,
            "events": public_events,
            "spans": [
                {
                    "id": node.id,
                    "parent_id": nodes[node.parent].id if node.parent else None,
                    "kind": node.key[0],
                    "name": node.name,
                    "started_at": (
                        node.started_at.isoformat()
                        if node.first_start_sequence is not None or node.key == _ROOT
                        else None
                    ),
                    "ended_at": node.ended_at.isoformat() if node.ended_at else None,
                    "status": node.status,
                    "event_ids": event_ids.get(node.key, []),
                    "transcript_links": node.transcript_links,
                    "detail": node.detail,
                }
                for node in sorted(
                    (nodes[key] for key in selected if key in nodes),
                    key=lambda item: (item.anchor, item.key[0]),
                )
            ],
        }
