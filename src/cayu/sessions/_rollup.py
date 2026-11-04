"""Usage, time and tool failures across many sessions, grouped, read a page at a time.

`cayu session rollup` answers questions about many sessions at once: how many tokens each
agent used per day, what that cost (priced per call from a price book), where time went (model calls, tool calls, environment setup, and the
runtime gaps between them), and which tools failed and why. It reads one page of sessions
at a time, and each session's events in bounded pages of only the types it needs, keeping
per-session facts and per-group sums, never whole histories. Groups merge by adding, so a
caller can read page after page (`next_cursor`) and combine them, and an Agent can serve one
page per request at `POST /api/sessions/rollup`.

Time per session:

- ``active_ms``: the active time of its finished turns (`interaction.*` terminal events),
  which leaves out time spent waiting for a person; ``wall_ms`` includes it.
- ``model_ms``, ``tool_ms``, ``setup_ms``: the union of model-call, tool-call and
  environment setup intervals (factory, binding and materialization), so parallel tool
  calls count once, with waits for a person (approval or input requests until answered)
  cut out, as turn time leaves them out.
- ``teardown_ms``: the union of environment teardown intervals (binding finalization:
  copy-back and release). Teardown runs after a turn's answer, so it is not part of
  ``active_ms``.
- ``gap_ms``: active time not covered by any of those intervals, the runtime's own work
  between steps. It is approximate: an interval of an unfinished turn still counts as busy.

`session_timing` reads the same events for one session and splits its whole elapsed time
into parts that add up to it (`cayu session show`).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu.budgets.aggregates import (
    aggregate_hosted_tool_usage_metrics_from_event_payload,
    aggregate_usage_metrics_from_event_payload,
    summary_usage_metrics_from_event_payload,
)
from cayu.budgets.pricing import PriceBook
from cayu.budgets.usage import (
    AggregateUsageMetrics,
    UsageMetrics,
    add_aggregate_usage,
    build_aggregate_usage_metrics,
)
from cayu.events import Event, EventType
from cayu.releases import recorded_release_identity
from cayu.sessions._call_inspection import (
    CallCost,
    CallPricer,
    add_cost,
    bounded_price_book,
    currency_totals_text,
)
from cayu.sessions.event_queries import EventOrder, EventQuery, EventQueryResultTooLarge
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import EventRecord, Session, SessionStatus
from cayu.sessions.store import SessionStore

ROLLUP_MAX_SESSIONS_PER_PAGE = 500
# Distinct (tool, kind) failure counts one group keeps; the rest are counted as "other".
# Failure kinds a rollup shows per group (`top_failure_kinds`); groups keep exact counts
# for every kind until then, so merging pages in any order gives the same answer.
ROLLUP_MAX_FAILURE_KINDS = 200
_EVENT_PAGE = 200
_EVENT_PAGE_BYTES = 8 * 1024 * 1024
_NO_VALUE = "(none)"

_SETUP_STARTS: dict[EventType | str, tuple[str, str]] = {
    EventType.ENVIRONMENT_FACTORY_STARTED: ("factory", "allocation_id"),
    EventType.ENVIRONMENT_BINDING_STARTED: ("binding", "binding_generation_id"),
    EventType.ENVIRONMENT_MATERIALIZATION_STARTED: ("materialization", "binding_generation_id"),
}
_SETUP_ENDS: dict[EventType | str, tuple[str, str]] = {
    EventType.ENVIRONMENT_FACTORY_COMPLETED: ("factory", "allocation_id"),
    EventType.ENVIRONMENT_FACTORY_FAILED: ("factory", "allocation_id"),
    EventType.ENVIRONMENT_BINDING_COMPLETED: ("binding", "binding_generation_id"),
    EventType.ENVIRONMENT_BINDING_FAILED: ("binding", "binding_generation_id"),
    EventType.ENVIRONMENT_MATERIALIZATION_COMPLETED: ("materialization", "binding_generation_id"),
    EventType.ENVIRONMENT_MATERIALIZATION_FAILED: ("materialization", "binding_generation_id"),
}
_TEARDOWN_STARTS: dict[EventType | str, tuple[str, str]] = {
    EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED: ("finalize", "binding_generation_id"),
}
_TEARDOWN_ENDS: dict[EventType | str, tuple[str, str]] = {
    EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED: ("finalize", "binding_generation_id"),
    EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED: ("finalize", "binding_generation_id"),
}
_FAILED_ENVIRONMENT_ENDS = frozenset(
    {
        EventType.ENVIRONMENT_FACTORY_FAILED,
        EventType.ENVIRONMENT_BINDING_FAILED,
        EventType.ENVIRONMENT_MATERIALIZATION_FAILED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
    }
)
_TURN_STARTS = (EventType.INTERACTION_STARTED, EventType.INTERACTION_RESUMED)
_TURN_ENDS = (
    EventType.INTERACTION_COMPLETED,
    EventType.INTERACTION_FAILED,
    EventType.INTERACTION_INTERRUPTED,
)
ROLLUP_EVENT_TYPES = (
    EventType.SESSION_STARTED,
    EventType.MODEL_STARTED,
    EventType.MODEL_COMPLETED,
    EventType.MODEL_ERROR,
    EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
    EventType.MODEL_HOSTED_TOOL_CALL,
    EventType.TOOL_CALL_STARTED,
    EventType.TOOL_CALL_COMPLETED,
    EventType.TOOL_CALL_FAILED,
    EventType.TOOL_CALL_BLOCKED,
    *_SETUP_STARTS,
    *_SETUP_ENDS,
    *_TEARDOWN_STARTS,
    *_TEARDOWN_ENDS,
    *_TURN_STARTS,
    EventType.INTERACTION_PAUSED,
    *_TURN_ENDS,
    # Waits for a person, which turn time leaves out.
    EventType.TOOL_CALL_APPROVAL_REQUESTED,
    EventType.TOOL_CALL_APPROVED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.TOOL_CALL_APPROVAL_EXPIRED,
    EventType.SESSION_AWAITING_USER_INPUT,
    EventType.SESSION_RESUMED,
)
_WAIT_ENDS = (
    EventType.TOOL_CALL_APPROVED,
    EventType.TOOL_CALL_APPROVAL_DENIED,
    EventType.TOOL_CALL_APPROVAL_EXPIRED,
)

RollupDimension = Literal["agent", "day", "release", "label"]


class SessionRollupQuery(BaseModel):
    """Which sessions a rollup reads and how it groups them.

    ``since``/``until`` select sessions by creation time, half-open. ``group_by`` is
    ``agent``, ``day`` (UTC creation date), ``release`` (the Release the session started on)
    or ``label:KEY``. ``causal_budget_id`` keeps the sessions charged to one causal budget (a
    run and the child sessions it spawned), ``root_session_id`` the sessions one root
    invocation started, root included, and ``task_id`` the sessions run under one task
    (``SessionQuery.task_id``). ``cursor`` continues after the previous page's last session.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    since: datetime | None = None
    until: datetime | None = None
    group_by: str = "agent"
    agent_name: str | None = None
    environment_name: str | None = None
    status: SessionStatus | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    causal_budget_id: str | None = None
    root_session_id: str | None = None
    task_id: str | None = None
    cursor: str | None = None
    max_sessions: StrictInt = Field(default=100, ge=1, le=ROLLUP_MAX_SESSIONS_PER_PAGE)
    # Prices every model call on its own date; without it groups report no cost.
    pricing: PriceBook | None = None

    @field_validator("pricing")
    @classmethod
    def validate_pricing(cls, value: PriceBook | None) -> PriceBook | None:
        return bounded_price_book(value)

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, value: str) -> str:
        rollup_grouping(value)
        return value

    @model_validator(mode="after")
    def validate_window(self) -> SessionRollupQuery:
        for name in ("since", "until"):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must be timezone-aware.")
        if self.since is not None and self.until is not None and self.since >= self.until:
            raise ValueError("since must be before until.")
        return self

    def session_query(self) -> SessionQuery:
        return SessionQuery(
            agent_name=self.agent_name,
            environment_name=self.environment_name,
            status=self.status,
            labels=self.labels,
            causal_budget_id=self.causal_budget_id,
            root_session_id=self.root_session_id,
            task_id=self.task_id,
            created_since=self.since,
            created_before=self.until,
            cursor=self.cursor,
            limit=self.max_sessions,
            order_by=SessionOrder.CREATED_AT_ASC,
        )


def rollup_grouping(value: str) -> tuple[RollupDimension, str | None]:
    """``agent``, ``day``, ``release`` or ``label:KEY``, as (dimension, label key)."""

    if value == "agent":
        return "agent", None
    if value == "day":
        return "day", None
    if value == "release":
        return "release", None
    dimension, separator, key = value.partition(":")
    if dimension == "label" and separator and key.strip() == key and key:
        return "label", key
    raise ValueError("group_by must be agent, day, release or label:KEY.")


# Rollup results cross HTTP from an Agent that may run a newer Cayu than the CLI; a field
# a later Cayu adds (such as cost) is ignored rather than failing the read.
_RESPONSE_CONFIG = ConfigDict(extra="ignore")


class RollupFailureKind(BaseModel):
    model_config = _RESPONSE_CONFIG

    tool: str
    kind: str
    count: StrictInt = Field(ge=0)


class RollupDurations(BaseModel):
    """Milliseconds summed over a group's sessions; see the module docstring."""

    model_config = _RESPONSE_CONFIG

    active_ms: StrictInt = Field(default=0, ge=0)
    wall_ms: StrictInt = Field(default=0, ge=0)
    model_ms: StrictInt = Field(default=0, ge=0)
    tool_ms: StrictInt = Field(default=0, ge=0)
    setup_ms: StrictInt = Field(default=0, ge=0)
    # An Agent whose Cayu predates teardown timing reports 0.
    teardown_ms: StrictInt = Field(default=0, ge=0)
    gap_ms: StrictInt = Field(default=0, ge=0)


class RollupGroup(BaseModel):
    """One group's sums. Two groups with the same key merge by adding."""

    model_config = _RESPONSE_CONFIG

    key: str
    sessions: StrictInt = Field(default=0, ge=0)
    statuses: dict[str, StrictInt] = Field(default_factory=dict)
    model_calls: StrictInt = Field(default=0, ge=0)
    model_calls_with_usage: StrictInt = Field(default=0, ge=0)
    model_errors: StrictInt = Field(default=0, ge=0)
    tool_calls: StrictInt = Field(default=0, ge=0)
    tool_failures: StrictInt = Field(default=0, ge=0)
    usage: AggregateUsageMetrics = Field(default_factory=build_aggregate_usage_metrics)
    # Priced model calls' estimated cost per currency (decimal strings), and the calls the
    # price book had no price for. Both empty when the query carried no price book.
    cost: dict[str, str] = Field(default_factory=dict)
    unpriced_model_calls: StrictInt = Field(default=0, ge=0)
    durations: RollupDurations = Field(default_factory=RollupDurations)
    failure_kinds: list[RollupFailureKind] = Field(default_factory=list)

    def merge(self, other: RollupGroup) -> RollupGroup:
        if other.key != self.key:
            raise ValueError("Only groups with the same key merge.")
        statuses = Counter(self.statuses)
        statuses.update(other.statuses)
        kinds: Counter[tuple[str, str]] = Counter()
        for item in (*self.failure_kinds, *other.failure_kinds):
            kinds[(item.tool, item.kind)] += item.count
        return RollupGroup(
            key=self.key,
            sessions=self.sessions + other.sessions,
            statuses=dict(sorted(statuses.items())),
            model_calls=self.model_calls + other.model_calls,
            model_calls_with_usage=self.model_calls_with_usage + other.model_calls_with_usage,
            model_errors=self.model_errors + other.model_errors,
            tool_calls=self.tool_calls + other.tool_calls,
            tool_failures=self.tool_failures + other.tool_failures,
            usage=add_aggregate_usage(self.usage, other.usage),
            cost=_add_costs(self.cost, other.cost),
            unpriced_model_calls=self.unpriced_model_calls + other.unpriced_model_calls,
            durations=RollupDurations(
                **{
                    name: getattr(self.durations, name) + getattr(other.durations, name)
                    for name in RollupDurations.model_fields
                }
            ),
            failure_kinds=_kind_rows(kinds),
        )


class RollupPage(BaseModel):
    """One page of a rollup: groups for `sessions_read` sessions and where to continue."""

    model_config = _RESPONSE_CONFIG

    groups: list[RollupGroup]
    sessions_read: StrictInt = Field(ge=0)
    next_cursor: str | None = None


def merge_groups(groups: Iterable[RollupGroup]) -> list[RollupGroup]:
    """Merge groups by key, ordered by key."""

    merged: dict[str, RollupGroup] = {}
    for group in groups:
        current = merged.get(group.key)
        merged[group.key] = group if current is None else current.merge(group)
    return [merged[key] for key in sorted(merged)]


@dataclass
class SessionFacts:
    """What a rollup keeps about one session after reading its events."""

    release: str | None = None
    model_calls: int = 0
    model_calls_with_usage: int = 0
    model_errors: int = 0
    tool_calls: int = 0
    tool_failures: Counter[tuple[str, str]] = field(default_factory=Counter)
    usage: AggregateUsageMetrics = field(default_factory=build_aggregate_usage_metrics)
    cost: dict[str, Decimal] = field(default_factory=dict)
    unpriced_model_calls: int = 0
    active_ms: int = 0
    wall_ms: int = 0
    model_ms: int = 0
    tool_ms: int = 0
    setup_ms: int = 0
    teardown_ms: int = 0
    gap_ms: int = 0


async def rollup_page(store: SessionStore, query: SessionRollupQuery) -> RollupPage:
    """Read one page of sessions and return their groups."""

    dimension, label_key = rollup_grouping(query.group_by)
    result = await store.list_sessions(query.session_query())
    groups: list[RollupGroup] = []
    for session in result.sessions:
        facts = await session_facts(store, session.id, pricing=query.pricing)
        key = _group_key(session, facts, dimension, label_key)
        groups.append(_session_group(key, session, facts))
    return RollupPage(
        groups=merge_groups(groups),
        sessions_read=len(result.sessions),
        next_cursor=result.next_cursor,
    )


async def session_facts(
    store: SessionStore, session_id: str, *, pricing: PriceBook | None = None
) -> SessionFacts:
    """Read one session's rollup facts from its events, a bounded page at a time."""

    return (await _scan_session(store, session_id, pricing=pricing)).facts


Span = tuple[datetime, datetime]
_NEVER = datetime.max.replace(tzinfo=UTC)


@dataclass
class _EnvironmentOperation:
    kind: str
    environment: str | None
    started_at: datetime | None
    ended_at: datetime | None = None
    outcome: Literal["completed", "failed", "running"] = "running"
    phase_timings: object = None


@dataclass
class _SessionScan:
    facts: SessionFacts
    model_spans: list[Span]
    tool_spans: list[Span]
    setup_spans: list[Span]
    teardown_spans: list[Span]
    wait_spans: list[Span]
    turn_spans: list[Span]
    # Model calls, tool calls, environment operations and a turn still running when the
    # read ended, from their start; the rollup leaves them out, the timing view doesn't.
    unfinished: dict[str, list[Span]]
    operations: list[_EnvironmentOperation]
    operations_truncated: bool
    first_at: datetime | None
    last_at: datetime | None
    first_model_at: datetime | None
    # Each finished model call's span, and each finished tool call's with its tool's name.
    model_call_spans: list[Span]
    tool_call_spans: list[tuple[str, Span]]
    # The read stopped at ``max_events``; everything above covers only the events read.
    truncated: bool


async def _scan_session(
    store: SessionStore,
    session_id: str,
    *,
    pricing: PriceBook | None,
    max_events: int | None = None,
) -> _SessionScan:
    facts = SessionFacts()
    model_open: dict[str, datetime] = {}
    tool_open: dict[tuple[str, str], datetime] = {}
    # Setup and teardown keys never collide: teardown's family is always "finalize".
    setup_open: dict[tuple[str, str], datetime] = {}
    turn_open: datetime | None = None
    model_spans: list[Span] = []
    tool_spans: list[Span] = []
    setup_spans: list[Span] = []
    teardown_spans: list[Span] = []
    turn_spans: list[Span] = []
    operations: list[_EnvironmentOperation] = []
    open_operations: dict[tuple[str, str], _EnvironmentOperation] = {}
    operations_truncated = False
    first_at: datetime | None = None
    last_at: datetime | None = None
    first_model_at: datetime | None = None
    pricer = None if pricing is None else CallPricer(pricing)
    wait_open: dict[tuple[str, str], datetime] = {}
    wait_spans: list[Span] = []
    tool_call_spans: list[tuple[str, Span]] = []
    truncated = False
    read = 0

    def start_operation(key: tuple[str, str] | None, event: Event) -> None:
        nonlocal operations_truncated
        if key is None or key in open_operations:
            return
        if len(operations) >= SESSION_TIMING_MAX_OPERATIONS:
            operations_truncated = True
            return
        operation = _EnvironmentOperation(
            kind=key[0], environment=event.environment_name, started_at=event.timestamp
        )
        operations.append(operation)
        open_operations[key] = operation

    def end_operation(key: tuple[str, str] | None, event: Event, started: datetime | None) -> None:
        nonlocal operations_truncated
        operation = None if key is None else open_operations.pop(key, None)
        if operation is None:
            if len(operations) >= SESSION_TIMING_MAX_OPERATIONS:
                operations_truncated = True
                return
            operation = _EnvironmentOperation(
                kind=key[0] if key is not None else "environment",
                environment=event.environment_name,
                started_at=started,
            )
            operations.append(operation)
        operation.ended_at = event.timestamp
        operation.outcome = "failed" if event.type in _FAILED_ENVIRONMENT_ENDS else "completed"
        operation.phase_timings = event.payload.get("phase_timings")

    async for record in _events(store, session_id):
        if max_events is not None and read >= max_events:
            truncated = True
            break
        read += 1
        event = record.event
        payload = event.payload
        at = event.timestamp
        first_at = at if first_at is None else min(first_at, at)
        last_at = at if last_at is None else max(last_at, at)
        if event.type == EventType.SESSION_STARTED:
            release = recorded_release_identity(payload)
            if facts.release is None and release is not None:
                facts.release = release.id
        elif event.type == EventType.MODEL_STARTED:
            # The model running again proves every earlier wait ended, whatever path
            # resolved it.
            wait_spans.extend((started, at) for started in wait_open.values())
            wait_open.clear()
            first_model_at = at if first_model_at is None else min(first_model_at, at)
            attempt = _text(payload.get("model_attempt_id"))
            if attempt is not None:
                model_open.setdefault(attempt, at)
        elif event.type in {EventType.MODEL_COMPLETED, EventType.MODEL_ERROR}:
            if event.type == EventType.MODEL_COMPLETED:
                facts.model_calls += 1
                metrics = _usage(payload)
                if metrics is not None:
                    facts.model_calls_with_usage += 1
                    facts.usage = add_aggregate_usage(facts.usage, metrics)
                _count_cost(
                    facts, None if pricer is None else pricer.observe(event), model_call=True
                )
            else:
                facts.model_errors += 1
            started = model_open.pop(_text(payload.get("model_attempt_id")) or "", None)
            if started is not None:
                model_spans.append((started, at))
        elif event.type == EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED:
            # Auxiliary inference and provider-billed hosted tools add usage, not model
            # calls, as the store's usage totals count them.
            metrics = _usage_only(aggregate_usage_metrics_from_event_payload, payload)
            if metrics is not None:
                facts.usage = add_aggregate_usage(facts.usage, metrics)
            _count_cost(facts, None if pricer is None else pricer.observe(event), model_call=True)
        elif event.type == EventType.MODEL_HOSTED_TOOL_CALL:
            metrics = _usage_only(aggregate_hosted_tool_usage_metrics_from_event_payload, payload)
            if metrics is not None:
                facts.usage = add_aggregate_usage(facts.usage, metrics)
            # Folded into the cost of the model attempt it belongs to.
            if pricer is not None:
                pricer.observe(event)
        elif event.type == EventType.TOOL_CALL_STARTED:
            facts.tool_calls += 1
            call = _tool_call(payload)
            if call is not None:
                tool_open.setdefault(call, at)
        elif event.type in {
            EventType.TOOL_CALL_COMPLETED,
            EventType.TOOL_CALL_FAILED,
            EventType.TOOL_CALL_BLOCKED,
        }:
            call = _tool_call(payload)
            started = None if call is None else tool_open.pop(call, None)
            if started is not None:
                tool_spans.append((started, at))
                tool_call_spans.append((event.tool_name or _NO_VALUE, (started, at)))
            # An approval can also resolve as a block (ambiguous policy evidence).
            waited = wait_open.pop(("approval", _text(payload.get("approval_id")) or ""), None)
            if waited is not None:
                wait_spans.append((waited, at))
            if event.type != EventType.TOOL_CALL_COMPLETED:
                facts.tool_failures[(event.tool_name or _NO_VALUE, tool_failure_kind(event))] += 1
        elif (
            start := _SETUP_STARTS.get(event.type) or _TEARDOWN_STARTS.get(event.type)
        ) is not None:
            setup_key = _setup_key(event, start)
            if setup_key is not None:
                setup_open.setdefault(setup_key, at)
            start_operation(setup_key, event)
        elif (end := _SETUP_ENDS.get(event.type) or _TEARDOWN_ENDS.get(event.type)) is not None:
            teardown = event.type in _TEARDOWN_ENDS
            setup_key = _setup_key(event, end)
            started = None if setup_key is None else setup_open.pop(setup_key, None)
            elapsed = payload.get("elapsed_ms")
            if started is None and type(elapsed) is int and elapsed >= 0:
                started = at - timedelta(milliseconds=elapsed)
            if started is not None:
                (teardown_spans if teardown else setup_spans).append((started, at))
            end_operation(setup_key, event, started)
        elif event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED:
            approval = _text(payload.get("approval_id"))
            if approval is not None:
                wait_open.setdefault(("approval", approval), at)
        elif event.type in _WAIT_ENDS:
            started = wait_open.pop(("approval", _text(payload.get("approval_id")) or ""), None)
            if started is not None:
                wait_spans.append((started, at))
        elif event.type == EventType.SESSION_AWAITING_USER_INPUT:
            question = _text(payload.get("input_id"))
            if question is not None:
                wait_open.setdefault(("input", question), at)
        elif event.type == EventType.SESSION_RESUMED:
            # A resume ends the input or approval wait it names.
            for kind, field_name in (("input", "input_id"), ("approval", "approval_id")):
                started = wait_open.pop((kind, _text(payload.get(field_name)) or ""), None)
                if started is not None:
                    wait_spans.append((started, at))
        elif event.type in _TURN_STARTS:
            turn_open = at if turn_open is None else turn_open
        elif event.type == EventType.INTERACTION_PAUSED:
            # A paused turn waits for a person; its terminal event covers it if it ends.
            turn_open = None
        elif event.type in _TURN_ENDS:
            turn_open = None
            facts.active_ms += _milliseconds(payload.get("active_duration_ms"))
            wall_ms = _milliseconds(payload.get("wall_duration_ms"))
            facts.wall_ms += wall_ms
            if wall_ms:
                turn_spans.append((at - timedelta(milliseconds=wall_ms), at))
    for cost in [] if pricer is None else pricer.unclaimed():
        # A hosted call no model call claimed still leaves the total incomplete when unpriced.
        _count_cost(facts, cost, model_call=True)
    # A tool call that waited for approval spans the wait; turn time leaves waits out, so
    # every phase does too. A wait still open runs to the end of what was read.
    wait_spans.extend((started, _NEVER) for started in wait_open.values())
    model_call_spans = model_spans
    model_spans = _without(model_spans, wait_spans)
    tool_spans = _without(tool_spans, wait_spans)
    setup_spans = _without(setup_spans, wait_spans)
    teardown_spans = _without(teardown_spans, wait_spans)
    facts.model_ms = _union_ms(model_spans)
    facts.tool_ms = _union_ms(tool_spans)
    facts.setup_ms = _union_ms(setup_spans)
    facts.teardown_ms = _union_ms(teardown_spans)
    facts.gap_ms = max(facts.active_ms - _union_ms([*model_spans, *tool_spans, *setup_spans]), 0)
    unfinished: dict[str, list[Span]] = {
        "model": [(started, _NEVER) for started in model_open.values()],
        "tool": [(started, _NEVER) for started in tool_open.values()],
        "setup": [(started, _NEVER) for key, started in setup_open.items() if key[0] != "finalize"],
        "teardown": [
            (started, _NEVER) for key, started in setup_open.items() if key[0] == "finalize"
        ],
        "turn": [] if turn_open is None else [(turn_open, _NEVER)],
    }
    return _SessionScan(
        facts=facts,
        model_spans=model_spans,
        tool_spans=tool_spans,
        setup_spans=setup_spans,
        teardown_spans=teardown_spans,
        wait_spans=wait_spans,
        turn_spans=turn_spans,
        unfinished=unfinished,
        operations=operations,
        operations_truncated=operations_truncated,
        first_at=first_at,
        last_at=last_at,
        first_model_at=first_model_at,
        model_call_spans=model_call_spans,
        tool_call_spans=tool_call_spans,
        truncated=truncated,
    )


SESSION_TIMING_MAX_OPERATIONS = 100
# Distinct tools `per_tool` lists; the rest are left out (`per_tool_truncated`).
SESSION_TIMING_MAX_TOOLS = 100
# Timed events (model, tool, environment, turn and wait events) one breakdown reads.
SESSION_TIMING_MAX_EVENTS = 100_000


class SessionTimingPhase(BaseModel):
    """One step of an environment operation, from its ``phase_timings`` (see
    ``cayu.environments._phase_timing``). ``depth`` is how many steps it runs inside."""

    model_config = _RESPONSE_CONFIG

    phase: str
    started_at: datetime
    duration_ms: StrictInt = Field(ge=0)
    outcome: str
    depth: StrictInt = Field(default=0, ge=0)
    items: StrictInt | None = Field(default=None, ge=0)
    bytes: StrictInt | None = Field(default=None, ge=0)


class SessionTimingOperation(BaseModel):
    """One environment operation: ``factory`` (allocate or reconnect), ``binding`` (copy-in
    and setup), ``materialization`` (a deferred environment starting) or ``finalize``
    (copy-back and release). ``phases`` is empty for a session recorded before Cayu timed
    the steps, or for a binding that reports none."""

    model_config = _RESPONSE_CONFIG

    kind: str
    environment: str | None = None
    started_at: datetime | None = None
    duration_ms: StrictInt | None = Field(default=None, ge=0)
    outcome: Literal["completed", "failed", "running"]
    phases: list[SessionTimingPhase] = Field(default_factory=list)


class SessionTimingCalls(BaseModel):
    """How long individual calls took: ``calls`` that finished, the ``total_ms`` of their
    durations (parallel calls each count in full), and their ``average_ms`` and ``max_ms``.
    A tool call's duration leaves out time it waited for an approval."""

    model_config = _RESPONSE_CONFIG

    calls: StrictInt = Field(default=0, ge=0)
    total_ms: StrictInt = Field(default=0, ge=0)
    average_ms: StrictInt | None = Field(default=None, ge=0)
    max_ms: StrictInt | None = Field(default=None, ge=0)


class SessionTimingTool(SessionTimingCalls):
    """One tool's calls; see `SessionTimingCalls`."""

    tool: str


class SessionTiming(BaseModel):
    """Where one session's time went, in milliseconds.

    ``elapsed_ms`` runs from the session's first recorded event to its last. It splits into
    parts that add up to it, each moment counted once, in the first part it belongs to:
    ``waiting_ms`` (approval or input requests until answered), ``setup_ms`` (environment
    factory, binding and materialization), ``teardown_ms`` (binding finalization), then
    ``model_ms`` and ``tool_ms`` (calls running, parallel calls counted once).
    ``other_ms`` is the rest of the time inside turns, the runtime's own work between
    steps; ``idle_ms`` is time outside any turn and its teardown, such as between turns.
    A call, environment operation or turn still running when the events were read counts
    until the last event. ``first_model_call_ms`` is how long after the start the first
    model call began.

    ``gap_ms`` is the time after the first model call began (the whole session without one)
    not counted in ``model_ms`` or ``tool_ms``: the runtime's work between steps, setup,
    teardown, waits (also those inside a call) and idle time. So ``first_model_call_ms``,
    ``model_ms``, ``tool_ms`` and ``gap_ms`` add up to the elapsed time, unless a tool call
    ran before the first model call (that time counts in both). ``per_model_call`` and
    ``per_tool`` say how long single calls took. ``truncated`` says the session had more timed events than
    one breakdown reads; it then covers the time up to the last event read.
    """

    model_config = _RESPONSE_CONFIG

    started_at: datetime | None = None
    ended_at: datetime | None = None
    elapsed_ms: StrictInt = Field(default=0, ge=0)
    setup_ms: StrictInt = Field(default=0, ge=0)
    model_ms: StrictInt = Field(default=0, ge=0)
    tool_ms: StrictInt = Field(default=0, ge=0)
    teardown_ms: StrictInt = Field(default=0, ge=0)
    waiting_ms: StrictInt = Field(default=0, ge=0)
    other_ms: StrictInt = Field(default=0, ge=0)
    idle_ms: StrictInt = Field(default=0, ge=0)
    model_calls: StrictInt = Field(default=0, ge=0)
    tool_calls: StrictInt = Field(default=0, ge=0)
    first_model_call_ms: StrictInt | None = Field(default=None, ge=0)
    environment: list[SessionTimingOperation] = Field(default_factory=list)
    environment_truncated: bool = False
    # Agents whose Cayu predates these fields report their defaults.
    gap_ms: StrictInt = Field(default=0, ge=0)
    per_model_call: SessionTimingCalls = Field(default_factory=SessionTimingCalls)
    per_tool: list[SessionTimingTool] = Field(default_factory=list)
    per_tool_truncated: bool = False
    truncated: bool = False


async def session_timing(store: SessionStore, session_id: str) -> SessionTiming:
    """Split one session's elapsed time into setup, model, tools, teardown and the rest."""

    scan = await _scan_session(
        store, session_id, pricing=None, max_events=SESSION_TIMING_MAX_EVENTS
    )
    # The scan reads only the event types it accounts for; the session's own first and
    # last events (streamed text, progress, ...) bound its elapsed time.
    bounds = [
        record.event.timestamp
        for order in (EventOrder.SEQUENCE_ASC, EventOrder.SEQUENCE_DESC)
        for record in await store.query_events(
            EventQuery(session_id=session_id, limit=1, order_by=order)
        )
    ]
    seen = [*bounds, *(at for at in (scan.first_at, scan.last_at) if at is not None)]
    if not seen:
        return SessionTiming()
    start = min([*seen, *(span[0] for span in scan.turn_spans)])
    # A truncated read says nothing about the time after its last event.
    end = scan.last_at if scan.truncated and scan.last_at is not None else max(seen)
    unfinished = scan.unfinished
    # Each part is what its spans add to everything claimed before it, so rounding to
    # milliseconds keeps the parts adding up to the elapsed time.
    claimed: list[Span] = []
    claimed_ms = 0
    parts: dict[str, int] = {}
    for name, spans in (
        ("waiting", scan.wait_spans),
        ("setup", [*scan.setup_spans, *unfinished["setup"]]),
        ("teardown", [*scan.teardown_spans, *unfinished["teardown"]]),
        ("model", [*scan.model_spans, *unfinished["model"]]),
        ("tool", [*scan.tool_spans, *unfinished["tool"]]),
        ("other", [*scan.turn_spans, *unfinished["turn"]]),
    ):
        claimed = _merged([*claimed, *_clip(spans, start, end)])
        total = _union_ms(claimed)
        parts[name] = total - claimed_ms
        claimed_ms = total
    elapsed = _union_ms([(start, end)])
    # Calls at work as ``model_ms`` and ``tool_ms`` count them: the waits, setup and
    # teardown inside a call (unfinished ones too) are cut out and fall in the gaps.
    calls = _without(
        [*scan.model_spans, *unfinished["model"], *scan.tool_spans, *unfinished["tool"]],
        [
            *scan.wait_spans,
            *scan.setup_spans,
            *unfinished["setup"],
            *scan.teardown_spans,
            *unfinished["teardown"],
        ],
    )
    first_model_at = scan.first_model_at if scan.first_model_at is not None else start
    busy = _union_ms(_clip([(start, first_model_at), *calls], start, end))
    per_tool, per_tool_truncated = _per_tool(scan.tool_call_spans, scan.wait_spans)
    return SessionTiming(
        started_at=start,
        ended_at=end,
        elapsed_ms=elapsed,
        setup_ms=parts["setup"],
        model_ms=parts["model"],
        tool_ms=parts["tool"],
        teardown_ms=parts["teardown"],
        waiting_ms=parts["waiting"],
        other_ms=parts["other"],
        idle_ms=max(elapsed - claimed_ms, 0),
        model_calls=scan.facts.model_calls,
        tool_calls=scan.facts.tool_calls,
        first_model_call_ms=(
            None
            if scan.first_model_at is None
            else max(round((scan.first_model_at - start).total_seconds() * 1000), 0)
        ),
        environment=[_timing_operation(item) for item in scan.operations],
        environment_truncated=scan.operations_truncated,
        gap_ms=max(elapsed - busy, 0),
        per_model_call=_call_durations([_span_ms(span) for span in scan.model_call_spans]),
        per_tool=per_tool,
        per_tool_truncated=per_tool_truncated,
        truncated=scan.truncated,
    )


def _span_ms(span: Span) -> int:
    return max(round((span[1] - span[0]).total_seconds() * 1000), 0)


def _call_durations(durations: list[int]) -> SessionTimingCalls:
    return _calls(len(durations), sum(durations), max(durations, default=None))


def _calls(calls: int, total_ms: int, max_ms: int | None) -> SessionTimingCalls:
    return SessionTimingCalls(
        calls=calls,
        total_ms=total_ms,
        average_ms=round(total_ms / calls) if calls else None,
        max_ms=max_ms,
    )


def _per_tool(
    calls: list[tuple[str, Span]], waits: list[Span]
) -> tuple[list[SessionTimingTool], bool]:
    """Each tool's call durations, waits cut out, longest total first."""

    durations: dict[str, list[int]] = {}
    for tool, span in calls:
        durations.setdefault(tool, []).append(_union_ms(_without([span], waits)))
    return _tool_rows({tool: _call_durations(values) for tool, values in durations.items()})


def _tool_rows(tools: dict[str, SessionTimingCalls]) -> tuple[list[SessionTimingTool], bool]:
    rows = sorted(tools.items(), key=lambda item: (-item[1].total_ms, item[0]))
    return [
        SessionTimingTool(tool=tool, **calls.model_dump())
        for tool, calls in rows[:SESSION_TIMING_MAX_TOOLS]
    ], len(rows) > SESSION_TIMING_MAX_TOOLS


class SessionTimingTotals(BaseModel):
    """`SessionTiming` added up over several sessions, such as those a task ran.

    ``sessions`` counts the sessions added. Every duration and count is the sum of theirs;
    sessions can run at the same time, so ``elapsed_ms`` can exceed the wall time they
    spanned. ``first_model_call_ms`` adds up each session's time before its first model
    call. ``per_model_call`` and ``per_tool`` merge their calls; ``truncated`` says some
    session's breakdown was partial.
    """

    model_config = _RESPONSE_CONFIG

    sessions: StrictInt = Field(default=0, ge=0)
    elapsed_ms: StrictInt = Field(default=0, ge=0)
    first_model_call_ms: StrictInt = Field(default=0, ge=0)
    setup_ms: StrictInt = Field(default=0, ge=0)
    model_ms: StrictInt = Field(default=0, ge=0)
    tool_ms: StrictInt = Field(default=0, ge=0)
    teardown_ms: StrictInt = Field(default=0, ge=0)
    waiting_ms: StrictInt = Field(default=0, ge=0)
    other_ms: StrictInt = Field(default=0, ge=0)
    idle_ms: StrictInt = Field(default=0, ge=0)
    gap_ms: StrictInt = Field(default=0, ge=0)
    model_calls: StrictInt = Field(default=0, ge=0)
    tool_calls: StrictInt = Field(default=0, ge=0)
    per_model_call: SessionTimingCalls = Field(default_factory=SessionTimingCalls)
    per_tool: list[SessionTimingTool] = Field(default_factory=list)
    per_tool_truncated: bool = False
    truncated: bool = False


_SUMMED_TIMING_FIELDS = (
    "elapsed_ms",
    "setup_ms",
    "model_ms",
    "tool_ms",
    "teardown_ms",
    "waiting_ms",
    "other_ms",
    "idle_ms",
    "gap_ms",
    "model_calls",
    "tool_calls",
)


def add_session_timings(timings: Iterable[SessionTiming]) -> SessionTimingTotals:
    """Add several sessions' breakdowns up; see `SessionTimingTotals`."""

    sums = dict.fromkeys(_SUMMED_TIMING_FIELDS, 0)
    sessions = first_model = 0
    tools: dict[str, SessionTimingCalls] = {}
    model = SessionTimingCalls()
    truncated = per_tool_truncated = False
    for timing in timings:
        sessions += 1
        for name in _SUMMED_TIMING_FIELDS:
            sums[name] += getattr(timing, name)
        first_model += timing.first_model_call_ms or 0
        model = _merge_calls(model, timing.per_model_call)
        for tool in timing.per_tool:
            tools[tool.tool] = _merge_calls(tools.get(tool.tool, SessionTimingCalls()), tool)
        truncated = truncated or timing.truncated
        per_tool_truncated = per_tool_truncated or timing.per_tool_truncated
    per_tool, more_tools = _tool_rows(tools)
    return SessionTimingTotals(
        sessions=sessions,
        first_model_call_ms=first_model,
        **sums,
        per_model_call=model,
        per_tool=per_tool,
        per_tool_truncated=per_tool_truncated or more_tools,
        truncated=truncated,
    )


def _merge_calls(left: SessionTimingCalls, right: SessionTimingCalls) -> SessionTimingCalls:
    worst = [value for value in (left.max_ms, right.max_ms) if value is not None]
    return _calls(
        left.calls + right.calls, left.total_ms + right.total_ms, max(worst, default=None)
    )


def _timing_operation(operation: _EnvironmentOperation) -> SessionTimingOperation:
    duration = (
        None
        if operation.started_at is None or operation.ended_at is None
        else max(round((operation.ended_at - operation.started_at).total_seconds() * 1000), 0)
    )
    phases: list[SessionTimingPhase] = []
    if isinstance(operation.phase_timings, list):
        for item in operation.phase_timings:
            try:
                phases.append(SessionTimingPhase.model_validate(item))
            except ValueError:
                # Recorded by a binding this reader can't read; the operation still shows.
                continue
    return SessionTimingOperation(
        kind=operation.kind,
        environment=operation.environment,
        started_at=operation.started_at,
        duration_ms=duration,
        outcome=operation.outcome,
        phases=phases,
    )


def tool_failure_kind(event: Event) -> str:
    """A failed or blocked tool call's error kind, from the structured evidence only.

    ``blocked:<by>`` for a blocked call; otherwise the first exception type the failure
    evidence names, then its classification (``timeout``, ``deadline``, ...), an isolated
    tool's failure code, ``tool_error`` when the tool itself returned an error, else
    ``unknown``. Free-text reasons are never used, so kinds stay short and safe to show.
    """

    payload = event.payload
    if event.type == EventType.TOOL_CALL_BLOCKED:
        return f"blocked:{_identifier(payload.get('blocked_by')) or 'policy'}"
    evidence = payload.get("failure_evidence")
    if isinstance(evidence, dict):
        types = evidence.get("exception_types")
        if isinstance(types, list) and types and _identifier(types[0]):
            return str(types[0])
        classification = _identifier(evidence.get("classification"))
        if classification and classification != "unknown":
            return classification
    code = _identifier(payload.get("isolated_tool_failure_code"))
    if code:
        return code
    result = payload.get("result")
    if isinstance(result, dict) and result.get("is_error") is True:
        return "tool_error"
    return "unknown"


async def _events(store: SessionStore, session_id: str) -> AsyncIterator[EventRecord]:
    after = 0
    page = _EVENT_PAGE
    bounded = True
    while True:
        query = EventQuery(
            session_id=session_id,
            event_types=ROLLUP_EVENT_TYPES,
            after_sequence=after,
            limit=page,
            order_by=EventOrder.SEQUENCE_ASC,
        )
        try:
            if bounded:
                records = await store.query_events_bounded(query, max_bytes=_EVENT_PAGE_BYTES)
            else:
                records = await store.query_events(query)
        except NotImplementedError:
            # A store without byte-bounded reads gets small pages instead.
            bounded, page = False, 25
            continue
        except EventQueryResultTooLarge:
            if page == 1:
                raise
            page = max(page // 2, 1)
            continue
        for record in records:
            yield record
        if len(records) < page:
            return
        after = records[-1].sequence
        page = _EVENT_PAGE if bounded else page


def _session_group(key: str, session: Session, facts: SessionFacts) -> RollupGroup:
    return RollupGroup(
        key=key,
        sessions=1,
        statuses={session.status.value: 1},
        model_calls=facts.model_calls,
        model_calls_with_usage=facts.model_calls_with_usage,
        model_errors=facts.model_errors,
        tool_calls=facts.tool_calls,
        tool_failures=sum(facts.tool_failures.values()),
        usage=facts.usage,
        cost=currency_totals_text(facts.cost),
        unpriced_model_calls=facts.unpriced_model_calls,
        durations=RollupDurations(
            active_ms=facts.active_ms,
            wall_ms=facts.wall_ms,
            model_ms=facts.model_ms,
            tool_ms=facts.tool_ms,
            setup_ms=facts.setup_ms,
            teardown_ms=facts.teardown_ms,
            gap_ms=facts.gap_ms,
        ),
        failure_kinds=_kind_rows(facts.tool_failures),
    )


def _group_key(
    session: Session, facts: SessionFacts, dimension: RollupDimension, label_key: str | None
) -> str:
    if dimension == "agent":
        return session.agent_name
    if dimension == "day":
        return session.created_at.date().isoformat()
    if dimension == "release":
        return facts.release or _NO_VALUE
    return session.labels.get(label_key or "", _NO_VALUE)


def _count_cost(facts: SessionFacts, cost: CallCost | None, *, model_call: bool) -> None:
    if cost is None:
        return
    add_cost(facts.cost, cost)
    if not cost.priced and model_call:
        facts.unpriced_model_calls += 1


def _add_costs(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    totals = {currency: Decimal(amount) for currency, amount in left.items()}
    for currency, amount in right.items():
        totals[currency] = totals.get(currency, Decimal(0)) + Decimal(amount)
    return currency_totals_text(totals)


def _kind_rows(kinds: Counter[tuple[str, str]]) -> list[RollupFailureKind]:
    return [
        RollupFailureKind(tool=tool, kind=kind, count=count)
        for (tool, kind), count in sorted(kinds.items(), key=lambda item: (-item[1], item[0]))
    ]


def top_failure_kinds(
    group: RollupGroup, limit: int = ROLLUP_MAX_FAILURE_KINDS
) -> list[RollupFailureKind]:
    """A group's most common failure kinds, the rest counted as one ``*``/``other`` row."""

    kept = group.failure_kinds[:limit]
    rest = sum(kind.count for kind in group.failure_kinds[limit:])
    return [*kept, *([RollupFailureKind(tool="*", kind="other", count=rest)] if rest else [])]


def _usage(payload: dict[str, Any]):
    try:
        return summary_usage_metrics_from_event_payload(payload)
    except (TypeError, ValueError):
        return None


def _usage_only(read: Callable[[dict[str, Any]], UsageMetrics | None], payload: dict[str, Any]):
    try:
        return read(payload)
    except (TypeError, ValueError):
        return None


def _tool_call(payload: dict[str, Any]) -> tuple[str, str] | None:
    round_id, call_id = _text(payload.get("tool_round_id")), _text(payload.get("tool_call_id"))
    return None if round_id is None or call_id is None else (round_id, call_id)


def _setup_key(event: Event, spec: tuple[str, str]) -> tuple[str, str] | None:
    family, field_name = spec
    value = _text(event.payload.get(field_name)) or event.environment_name
    return None if value is None else (family, value)


def _merged(spans: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(span for span in spans if span[1] > span[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _without(
    spans: list[tuple[datetime, datetime]], waits: list[tuple[datetime, datetime]]
) -> list[tuple[datetime, datetime]]:
    """``spans`` with every wait cut out of them."""

    pauses = _merged(waits)
    if not pauses:
        return spans
    kept: list[tuple[datetime, datetime]] = []
    for start, end in spans:
        for pause_start, pause_end in pauses:
            if pause_end <= start or pause_start >= end:
                continue
            if pause_start > start:
                kept.append((start, pause_start))
            start = max(start, pause_end)
            if start >= end:
                break
        if start < end:
            kept.append((start, end))
    return kept


def _clip(spans: list[Span], start: datetime, end: datetime) -> list[Span]:
    return [(max(a, start), min(b, end)) for a, b in spans if min(b, end) > max(a, start)]


def _union_ms(spans: list[tuple[datetime, datetime]]) -> int:
    total = 0.0
    current: tuple[datetime, datetime] | None = None
    for start, end in sorted(span for span in spans if span[1] > span[0]):
        if current is None or start > current[1]:
            if current is not None:
                total += (current[1] - current[0]).total_seconds()
            current = (start, end)
        elif end > current[1]:
            current = (current[0], end)
    if current is not None:
        total += (current[1] - current[0]).total_seconds()
    return round(total * 1000)


def _milliseconds(value: object) -> int:
    return value if type(value) is int and value >= 0 else 0


def _text(value: object) -> str | None:
    return value if type(value) is str and value.strip() else None


def _identifier(value: object) -> str | None:
    if type(value) is not str or not value or len(value) > 128:
        return None
    return value if all(character.isalnum() or character in "_.-" for character in value) else None
