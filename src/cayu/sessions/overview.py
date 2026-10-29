"""Provider-neutral session state and batched current-incarnation facts.

The reader owns store joins. Audience-specific redaction and wire models belong to
its callers; neither the application facade nor a server is a dependency here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from cayu._validation import revalidate_model_input
from cayu.budgets.pricing import PriceBook, SessionCostTotals, add_cost_amounts
from cayu.budgets.usage import SessionUsageSummary
from cayu.releases import recorded_release_identity
from cayu.sessions._foreground_child_checkpoint import foreground_child_state_from_checkpoint
from cayu.sessions._run_identity import (
    RUN_LINEAGE_METADATA_KEY,
    is_eval_trial,
    session_run_id,
    session_run_lineage,
)
from cayu.sessions._supervision_projection import RELEASE_MARKER_EVENT_TYPES
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.sessions.fork_overlays import resolve_fork_lineage_async
from cayu.sessions.invocation import SessionExecutionSource
from cayu.sessions.pending_action_contracts import (
    MAX_PENDING_ACTION_RESULT_BYTES,
    PendingActionQuery,
    PendingActionRecord,
)
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import PendingActionKind, Session, SessionStatus

if TYPE_CHECKING:
    from cayu.sessions.store import SessionStore
    from cayu.sessions.summaries import SessionOutcome

_MAX_DELEGATION_DEPTH = 8
_PENDING_ACTION_STATUSES = frozenset(
    {SessionStatus.INTERRUPTED, SessionStatus.FAILED, SessionStatus.COMPLETED}
)


class SessionOverviewStatus(StrEnum):
    """Where a run is. A new status needs a new contract version."""

    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_INPUT = "waiting_input"
    NEEDS_RECOVERY = "needs_recovery"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class SessionOverviewFacts:
    """Best-available state plus completeness for consumers requiring exact evidence.

    Cloud preserves its existing summary projection when pending rows have
    issues. Control-plane overviews and counts require ``pending_complete``.
    """

    session: Session
    status: SessionOverviewStatus
    outcome: SessionOutcome
    usage: SessionUsageSummary | None
    cost: SessionCostTotals | None
    forked: bool
    fork_controller: str | None
    release_id: str | None
    waiting: tuple[tuple[PendingActionRecord, Session | None], ...]
    pending_complete: bool

    @property
    def ended_at(self) -> datetime | None:
        return _ended_at(self.status, self.outcome)


@dataclass(frozen=True, slots=True)
class _PendingEvidence:
    actions: tuple[PendingActionRecord, ...] = ()
    complete: bool = True


@dataclass(frozen=True, slots=True)
class _WaitingEvidence:
    kinds: frozenset[PendingActionKind]
    complete: bool


@dataclass(frozen=True, slots=True)
class _StateEvidence:
    status: SessionOverviewStatus
    pending: _PendingEvidence
    complete: bool


class SessionOverviewUnavailable(RuntimeError):
    """An incomplete source cannot establish a truthful overview or count."""


@dataclass(frozen=True, slots=True)
class SessionOverviewCounts:
    since: datetime
    until: datetime
    running_now: int
    counts_by_state: dict[SessionOverviewStatus, int]
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    tokens_complete: bool
    cost: Decimal | None
    cost_complete: bool

    @property
    def total_count(self) -> int:
        return sum(self.counts_by_state.values())


class SessionOverviewReader:
    """Summarizes the sessions of one page, sharing session loads among them."""

    def __init__(self, store: SessionStore, *, pricing: PriceBook | None = None) -> None:
        self._store = store
        self._pricing = pricing
        self._loaded: dict[str, Session | None] = {}

    async def _load(self, session_id: str) -> Session | None:
        if session_id not in self._loaded:
            self._loaded[session_id] = await self._store.load(session_id)
        return self._loaded[session_id]

    async def affected_by(self, session_id: str) -> list[str]:
        """The session and the sessions above it whose summaries a change in it can alter.

        A session waiting on a subagent reports what the subagent waits on, as
        deep as ``_waiting_kinds`` follows it, so a change in the subagent can
        change every summary above it. A fork's source never waits on the fork.
        """

        affected = [session_id]
        session = await self._load(session_id)
        for _depth in range(_MAX_DELEGATION_DEPTH):
            if (
                session is None
                or session.parent_session_id is None
                or session.invocation.source is SessionExecutionSource.FORK
            ):
                break
            parent = await self._load(session.parent_session_id)
            if parent is None:
                break
            if RUN_LINEAGE_METADATA_KEY not in session.metadata:
                # Legacy rows lack retained ancestry, but the parent's current
                # foreground wait can still prove this exact dependency. This
                # refreshes a summary; it does not invent historical run links.
                follows = await self._retained_wait_matches(parent, session)
            else:
                follows = session_run_lineage(session)[1] == session_run_id(parent)
            if not follows:
                break
            affected.append(parent.id)
            session = parent
        return affected

    async def read(self, session_ids: tuple[str, ...]) -> tuple[SessionOverviewFacts, ...]:
        """Summarize the sessions from reads made after the scan.

        The scan read session rows only to find their verified parents,
        and kept reading events after it read them. Reading again now
        means every summary reflects at least every event up to the page's
        cursor, and any later change has a later sequence, so a later page
        reports the session again.
        """

        self._loaded.clear()
        sessions = [
            session
            for session_id in session_ids
            if (session := await self._load(session_id)) is not None
        ]
        if not sessions:
            return ()
        query = EventQuery(session_ids=tuple(session.id for session in sessions))
        usage = await self._store.read_usage_accounting(query, by_session=True)
        cost = (
            None
            if self._pricing is None
            else await self._store.read_cost_accounting(query, self._pricing, by_session=True)
        )
        usage_by_session = {summary.session_id: summary for summary in usage.session_summaries}
        cost_by_session = (
            {} if cost is None else {totals.session_id: totals for totals in cost.session_totals}
        )
        runs = []
        for session in sessions:
            run = await self._run(
                session, usage_by_session.get(session.id), cost_by_session.get(session.id)
            )
            # Accounting and outcome reads are separate store snapshots. A
            # replacement must never donate its cost or status to the old run.
            current = await self._store.load(session.id)
            if (
                run is not None
                and current is not None
                and current.instance_id == session.instance_id
            ):
                runs.append(run)
        return tuple(runs)

    async def counts(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        include_eval: bool = False,
        include_forks: bool = False,
    ) -> SessionOverviewCounts:
        """Complete current-read counts, not a cross-store atomic snapshot.

        The window counts sessions created in [since, until); running_now also
        includes older running sessions. Pages are followed through the upper
        creation boundary. Mutable update times never paginate this population.
        """
        until = datetime.now(UTC) if until is None else until
        since = until - timedelta(days=1) if since is None else since
        if any(t.tzinfo is None or t.utcoffset() is None for t in (since, until)) or since >= until:
            raise ValueError("Overview counts require aware since < until.")
        if type(include_eval) is not bool or type(include_forks) is not bool:
            raise TypeError("Overview inclusion flags must be bools.")
        counts = dict.fromkeys(SessionOverviewStatus, 0)
        input_tokens = output_tokens = cached_tokens = 0
        tokens_complete = True
        cost = None if self._pricing is None else Decimal(0)
        cost_complete = True
        running = 0
        cursor = None
        cursors: set[str] = set()
        seen: set[str] = set()
        while True:
            page = await self._store.list_sessions(
                SessionQuery(
                    limit=200,
                    cursor=cursor,
                    order_by=SessionOrder.CREATED_AT_ASC,
                )
            )
            eligible = [session for session in page.sessions if session.created_at < until]
            reached_until = len(eligible) != len(page.sessions)
            expected = {session.id: session.instance_id for session in eligible}
            self._loaded.clear()
            accounting_sessions: dict[str, str | None] = {}
            for session_id, instance_id in expected.items():
                session = await self._load(session_id)
                if session is None or session.instance_id != instance_id:
                    continue
                identity = session_run_id(session)
                if identity in seen:
                    continue
                seen.add(identity)
                if session.created_at >= until:
                    continue
                if not include_eval and is_eval_trial(session):
                    continue
                if not include_forks:
                    lineage = await resolve_fork_lineage_async(session, self._load)
                    if lineage.forked:
                        continue
                state = await self._state(session)
                # Pending and lineage joins may outlive this incarnation. Match
                # the full overview selection fence before including its facts.
                current = await self._store.load(session.id)
                if current is None or current.instance_id != session.instance_id:
                    continue
                accounting_sessions[session.id] = session.instance_id
                if not state.complete:
                    raise SessionOverviewUnavailable("Pending session evidence is unavailable.")
                if state.status is SessionOverviewStatus.RUNNING:
                    running += 1
                if session.created_at >= since:
                    counts[state.status] += 1
            if accounting_sessions:
                accounting_query = EventQuery(
                    session_ids=tuple(accounting_sessions), since=since, until=until
                )
                try:
                    usage = await self._store.read_usage_accounting(accounting_query)
                    summary = revalidate_model_input(usage.summary, SessionUsageSummary)
                    if not isinstance(summary, SessionUsageSummary):
                        raise TypeError("Invalid accounting summary.")
                    priced = (
                        None
                        if self._pricing is None
                        else await self._store.read_cost_accounting(accounting_query, self._pricing)
                    )
                    totals = (
                        None
                        if priced is None
                        else revalidate_model_input(priced.totals, SessionCostTotals)
                    )
                    if totals is not None and not isinstance(totals, SessionCostTotals):
                        raise TypeError("Invalid accounting totals.")
                except (TypeError, ValueError):
                    raise SessionOverviewUnavailable(
                        "Session accounting evidence is invalid."
                    ) from None
                # The selection and accounting are separate snapshots. A replacement
                # must not donate consumption to the selected incarnation.
                for session_id, instance_id in accounting_sessions.items():
                    current = await self._store.load(session_id)
                    if current is None or current.instance_id != instance_id:
                        raise SessionOverviewUnavailable("Session changed during accounting.")
                input_tokens += summary.usage.input_tokens
                output_tokens += summary.usage.output_tokens
                cached_tokens += summary.usage.cache.read_tokens
                tokens_complete = tokens_complete and summary.unmeasured_model_attempts == 0
                if totals is not None:
                    assert cost is not None
                    cost = add_cost_amounts(cost, totals.total_cost)
                    cost_complete = cost_complete and (
                        totals.unpriced_model_steps == 0 and totals.unpriced_auxiliary_attempts == 0
                    )
            if reached_until or page.next_cursor is None:
                break
            if page.next_cursor in cursors:
                raise SessionOverviewUnavailable("Session paging did not advance.")
            cursors.add(page.next_cursor)
            cursor = page.next_cursor
        return SessionOverviewCounts(
            since,
            until,
            running,
            counts,
            input_tokens,
            output_tokens,
            cached_tokens,
            tokens_complete,
            cost,
            cost_complete,
        )

    async def _pending(self, session: Session) -> _PendingEvidence:
        actions: list[PendingActionRecord] = []
        complete = session.instance_id is not None
        cursor = None
        seen: set[str] = set()
        cursors: set[str] = set()
        while True:
            page = await self._store.query_pending_actions(
                PendingActionQuery(
                    session_id=session.id,
                    limit=200,
                    cursor=cursor,
                    max_result_bytes=MAX_PENDING_ACTION_RESULT_BYTES,
                )
            )
            if page.issues:
                complete = False
            for action in page.actions:
                # A pending query is keyed by reusable ID. Never attach another
                # incarnation's actions to the session this reader selected.
                if (
                    session.instance_id is None
                    or action.session.id != session.id
                    or action.session.instance_id != session.instance_id
                ):
                    complete = False
                    continue
                if action.attention_id is None:
                    actions.append(action)
                elif action.attention_id not in seen:
                    actions.append(action)
                    seen.add(action.attention_id)
            if not page.has_more:
                return _PendingEvidence(tuple(actions), complete)
            if page.next_cursor is None or page.next_cursor in cursors:
                raise SessionOverviewUnavailable("Pending session paging did not advance.")
            cursors.add(page.next_cursor)
            cursor = page.next_cursor

    async def _waiting_kinds(
        self, session: Session, pending: _PendingEvidence, depth: int
    ) -> _WaitingEvidence:
        """The kinds a person settles, following delegated actions to their children.

        A delegated action names the kind its child waited on when the parent
        recorded the wait, which can lag the child: a child whose answer finds an
        unfinished call needs manual recovery while its parent still names the
        question. So each child's own pending actions are read now.
        """

        kinds: set[PendingActionKind] = set()
        complete = pending.complete
        for action in pending.actions:
            reference = action.delegated_action
            if reference is None:
                kinds.add(action.kind)
            elif (
                depth < _MAX_DELEGATION_DEPTH
                and (child := await self._delegated_child(session, action)) is not None
            ):
                child_waiting = await self._waiting_kinds(
                    child, await self._pending(child), depth + 1
                )
                # The retained wait was verified before the recursive awaits.
                # Even an empty or correctly bound page can outlive its child;
                # use an uncached fence before contributing any descendant state.
                current = await self._store.load(child.id)
                if current is None or current.instance_id != child.instance_id:
                    complete = False
                    continue
                kinds |= child_waiting.kinds
                complete = complete and child_waiting.complete
        return _WaitingEvidence(frozenset(kinds), complete)

    async def _delegated_child(
        self, parent: Session, action: PendingActionRecord
    ) -> Session | None:
        reference = action.delegated_action
        child = None if reference is None else await self._load(reference.child_session_id)
        if child is None:
            return None
        return child if await self._retained_wait_matches(parent, child) else None

    async def _retained_wait_matches(self, parent: Session, child: Session) -> bool:
        # A reused child ID can have the same parent. Verify the exact wait's
        # retained incarnation as well as the relationship, never just names.
        wait, _ = foreground_child_state_from_checkpoint(
            await self._store.load_checkpoint(parent.id)
        )
        return (
            wait is not None
            and wait.parent_effect.session_id == parent.id
            and wait.parent_effect.session_instance_id == parent.instance_id
            and wait.child_session_id == child.id
            and wait.child_session_instance_id == child.instance_id
        )

    async def _release_id(self, session_id: str) -> str | None:
        markers = await self._store.query_events(
            EventQuery(
                session_id=session_id,
                event_types=RELEASE_MARKER_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        release = None if not markers else recorded_release_identity(markers[0].event.payload)
        return None if release is None else release.id

    async def _state(self, session: Session) -> _StateEvidence:
        """Shared state evidence without outcome or display-only joins."""
        waiting = (
            await self._pending(session)
            if session.status in _PENDING_ACTION_STATUSES
            else _PendingEvidence()
        )
        waiting_kinds = await self._waiting_kinds(session, waiting, 0)
        status = _run_status(
            session.status,
            waiting_kinds.kinds,
            delegated=any(action.delegated_action is not None for action in waiting.actions),
        )
        return _StateEvidence(status, waiting, waiting_kinds.complete)

    async def _run(
        self,
        session: Session,
        usage: SessionUsageSummary | None,
        cost: SessionCostTotals | None,
    ) -> SessionOverviewFacts | None:
        try:
            outcome = await self._store.summarize_outcome(session.id)
        except KeyError:
            # A row can disappear during the preceding accounting reads or
            # inside this read. Only a confirmed missing/replaced incarnation
            # explains the error; an unrelated store failure must propagate.
            current = await self._store.load(session.id)
            if current is None or current.instance_id != session.instance_id:
                return None
            raise
        state = await self._state(session)
        lineage = await resolve_fork_lineage_async(session, self._load)
        return SessionOverviewFacts(
            session=session,
            status=state.status,
            outcome=outcome,
            usage=usage,
            cost=cost,
            forked=lineage.forked,
            fork_controller=lineage.owner() if lineage.settled else None,
            release_id=await self._release_id(session.id),
            waiting=tuple(
                [
                    (action, await self._delegated_child(session, action))
                    for action in state.pending.actions
                ]
            ),
            pending_complete=state.complete,
        )


# A run waiting on several items reports the first of these it waits on.
_WAITING_STATUSES = (
    (PendingActionKind.MANUAL_RECOVERY, SessionOverviewStatus.NEEDS_RECOVERY),
    (PendingActionKind.TOOL_APPROVAL, SessionOverviewStatus.WAITING_APPROVAL),
    (PendingActionKind.USER_INPUT, SessionOverviewStatus.WAITING_INPUT),
)
_ENDED_STATUSES = {
    SessionStatus.COMPLETED: SessionOverviewStatus.COMPLETED,
    SessionStatus.FAILED: SessionOverviewStatus.FAILED,
    SessionStatus.INTERRUPTED: SessionOverviewStatus.INTERRUPTED,
}


def _run_status(
    status: SessionStatus, waiting: frozenset[PendingActionKind], *, delegated: bool
) -> SessionOverviewStatus:
    for kind, run_status in _WAITING_STATUSES:
        if kind in waiting:
            return run_status
    if delegated:
        # Its subagent waits on no person: the work goes on below it, and the
        # session continues when the subagent ends.
        return SessionOverviewStatus.RUNNING
    return _ENDED_STATUSES.get(status, SessionOverviewStatus.RUNNING)


def _ended_at(status: SessionOverviewStatus, outcome: SessionOutcome) -> datetime | None:
    terminal = outcome.terminal_event
    if status not in _ENDED_STATUSES.values() or terminal is None:
        return None
    return terminal.event.timestamp
