"""Provider-neutral session state and batched current-incarnation facts.

The reader owns store joins. Audience-specific redaction and wire models belong to
its callers; neither the application facade nor a server is a dependency here.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import Field, model_validator

from cayu._time_buckets import (
    MAX_TIME_BUCKETS,
    TimeBucket,
    TimeSeriesSpec,
    bucket_index,
    time_buckets,
)
from cayu._validation import canonical_durable_json_bytes, revalidate_model_input
from cayu.budgets.pricing import PriceBook, SessionCostTotals, add_cost_amounts
from cayu.budgets.usage import USAGE_BEARING_EVENT_TYPES, AggregateCount, SessionUsageSummary
from cayu.releases import recorded_release_identity
from cayu.sessions._foreground_child_checkpoint import (
    MAX_FOREGROUND_CHAIN_DEPTH,
    foreground_child_state_from_checkpoint,
)
from cayu.sessions._run_identity import (
    is_eval_trial,
    session_run_id,
)
from cayu.sessions._supervision_projection import RELEASE_MARKER_EVENT_TYPES
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.fork_overlays import resolve_fork_lineages_async
from cayu.sessions.invocation import SessionExecutionSource
from cayu.sessions.pending_action_contracts import (
    MAX_PENDING_ACTION_RESULT_BYTES,
    PendingActionListResult,
    PendingActionQuery,
    PendingActionRecord,
)
from cayu.sessions.queries import (
    SessionAggregateFilter,
    SessionOrder,
    SessionQuery,
    _session_matches,
    session_query_from_aggregate_filter,
)
from cayu.sessions.records import PendingActionKind, Session, SessionLineage, SessionStatus
from cayu.sessions.summaries import SessionStepCounts

if TYPE_CHECKING:
    from cayu.events import EventType
    from cayu.sessions.store import SessionStore
    from cayu.sessions.summaries import SessionOutcome

_PENDING_ACTION_STATUSES = frozenset(
    {SessionStatus.INTERRUPTED, SessionStatus.FAILED, SessionStatus.COMPLETED}
)

# How many sessions of one overview page are read at once. Each session's read is
# a short sequence of store round trips, so a page costs about (sessions / this)
# sequences instead of one per session in turn. It stays below the default
# PostgreSQL pool of 5 so an open dashboard can't hold every connection.
OVERVIEW_READ_CONCURRENCY = 4

# Sessions per counts page, and event heads per read when finding older usage.
_COUNT_PAGE = 200
_USAGE_HEAD_PAGE = 5000
_MIN_USAGE_RANGE = timedelta(milliseconds=1)
_UNFILTERED = session_query_from_aggregate_filter(SessionAggregateFilter())

_T = TypeVar("_T")
_NO_STEPS = SessionStepCounts()


async def gather_in_order(
    calls: Iterable[Callable[[], Awaitable[_T]]], *, limit: int = OVERVIEW_READ_CONCURRENCY
) -> list[_T]:
    """Run the calls at most ``limit`` at a time and return their results in call order.

    The first failure is raised as it is, not wrapped in an ExceptionGroup, after
    the calls still running are cancelled and have finished.
    """

    semaphore = asyncio.Semaphore(limit)

    async def bounded(call: Callable[[], Awaitable[_T]]) -> _T:
        async with semaphore:
            return await call()

    tasks = [asyncio.ensure_future(bounded(call)) for call in calls]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


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
    #: ``None`` when the session recorded no usage, or the reader was asked
    #: for step counts only.
    usage: SessionUsageSummary | None
    cost: SessionCostTotals | None
    forked: bool
    fork_controller: str | None
    release_id: str | None
    waiting: tuple[tuple[PendingActionRecord, Session | None], ...]
    pending_complete: bool
    steps: SessionStepCounts = _NO_STEPS

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


class SessionStartBucket(TimeBucket):
    total_count: AggregateCount = Field(ge=0)
    failed: AggregateCount = Field(ge=0)

    @model_validator(mode="after")
    def failures_are_subset(self) -> SessionStartBucket:
        if self.failed > self.total_count:
            raise ValueError("Failed starts cannot exceed total starts.")
        return self


class SessionStartSeries(TimeSeriesSpec):
    """Current state of sessions grouped by their creation time, not failure time."""

    buckets: tuple[SessionStartBucket, ...] = Field(min_length=1, max_length=MAX_TIME_BUCKETS)


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
    series: SessionStartSeries | None = None

    @property
    def total_count(self) -> int:
        return sum(self.counts_by_state.values())


_Totals = TypeVar("_Totals", SessionCostTotals, SessionUsageSummary)


class _SessionTotalsCache(Generic[_Totals]):
    """Per-session totals keyed by what they were computed from, least recently used first."""

    def __init__(self, *, max_sessions: int = 4096) -> None:
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer.")
        self._max_sessions = max_sessions
        self._totals: OrderedDict[tuple[Any, ...], _Totals | None] = OrderedDict()

    def _get(self, key: tuple[Any, ...]) -> tuple[bool, _Totals | None]:
        if key not in self._totals:
            return False, None
        self._totals.move_to_end(key)
        totals = self._totals[key]
        return True, None if totals is None else totals.model_copy(deep=True)

    def _put(self, key: tuple[Any, ...], totals: _Totals | None) -> None:
        self._totals[key] = None if totals is None else totals.model_copy(deep=True)
        self._totals.move_to_end(key)
        while len(self._totals) > self._max_sessions:
            self._totals.popitem(last=False)


class SessionCostCache(_SessionTotalsCache[SessionCostTotals]):
    """Sessions' cost totals, reused while their cost-bearing events stay the same.

    A total is reused for the same session incarnation, the same count and
    latest sequence of its cost-bearing events, and the same price book. Events
    are only appended, and removing one lowers the count, so equal positions
    mean the same events, priced the same. Holds at most ``max_sessions``
    totals, dropping the least recently used.
    """


class SessionUsageCache(_SessionTotalsCache[SessionUsageSummary]):
    """Sessions' usage summaries, reused while their usage-bearing events stay the same.

    The rule ``SessionCostCache`` follows, for the events usage accounting
    reads: a summary is reused for the same session incarnation and the same
    count and latest sequence of its usage-bearing events.
    """


class SessionOverviewReader:
    """Summarizes the sessions of one page, sharing session loads among them.

    With ``token_usage=False`` a summary carries only its step counts, which the
    store counts without reading events, and ``usage`` is ``None``. With a
    ``cost_cache``, a page prices only the sessions whose cost-bearing events
    changed since the cache last priced them; with a ``usage_cache``, it sums
    usage only for the sessions whose usage-bearing events changed.
    """

    def __init__(
        self,
        store: SessionStore,
        *,
        pricing: PriceBook | None = None,
        token_usage: bool = True,
        cost_cache: SessionCostCache | None = None,
        usage_cache: SessionUsageCache | None = None,
    ) -> None:
        if type(token_usage) is not bool:
            raise TypeError("token_usage must be a bool.")
        if cost_cache is not None and type(cost_cache) is not SessionCostCache:
            raise TypeError("cost_cache must be a SessionCostCache.")
        if usage_cache is not None and type(usage_cache) is not SessionUsageCache:
            raise TypeError("usage_cache must be a SessionUsageCache.")
        self._store = store
        self._pricing = pricing
        self._token_usage = token_usage
        self._cost_cache = cost_cache
        self._usage_cache = usage_cache
        # Whether a parent's retained wait names a child, by both incarnations.
        self._retained_waits: dict[tuple[str, str | None, str, str | None], bool] = {}
        self._loaded: dict[str, Session | None] = {}
        # What a scan follows from each session: its parent, never its whole row.
        self._lineages: dict[str, SessionLineage | None] = {}
        self._loading: dict[str, asyncio.Future[Session | None]] = {}
        self._load_waiters: dict[asyncio.Future[Session | None], int] = {}

    async def _load(self, session_id: str) -> Session | None:
        # A page's sessions are summarized concurrently. Concurrent loads of one
        # session (a parent shared by its children) share one read, so every
        # summary on the page sees the same snapshot of it.
        if session_id in self._loaded:
            return self._loaded[session_id]
        pending = self._loading.get(session_id)
        if pending is None:
            pending = asyncio.ensure_future(self._store.load(session_id))
            # Retrieve the outcome here too, so a read that a cancelled page no
            # longer awaits never logs an unretrieved exception.
            pending.add_done_callback(lambda done: done.cancelled() or done.exception())
            self._loading[session_id] = pending
        self._load_waiters[pending] = self._load_waiters.get(pending, 0) + 1
        try:
            session = await asyncio.shield(pending)
        finally:
            remaining = self._load_waiters[pending] - 1
            if remaining:
                self._load_waiters[pending] = remaining
            else:
                del self._load_waiters[pending]
                if self._loading.get(session_id) is pending:
                    del self._loading[session_id]
                # Shielding protects other readers of this shared load. Once
                # the last reader leaves, it owns cancelling and draining it.
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)
        self._loaded[session_id] = session
        return session

    async def prefetch(self, session_ids: Iterable[str]) -> None:
        """Load the lineages of the sessions and the parents ``affected_by`` follows.

        A level at a time. A scan calls this once per batch of events, so
        ``affected_by`` reads no session of its own for any session the batch
        names. Only lineages are read; ``read`` reads the rows it summarizes.
        """

        level = list(dict.fromkeys(session_ids))
        for _depth in range(MAX_FOREGROUND_CHAIN_DEPTH + 1):
            await self._load_lineages(level)
            level = [
                parent_id
                for session_id in level
                if (lineage := self._lineages.get(session_id)) is not None
                and (parent_id := lineage.parent_session_id) is not None
                and lineage.source is not SessionExecutionSource.FORK
                and parent_id not in self._lineages
            ]
            if not level:
                return

    async def _load_lineages(self, session_ids: Iterable[str]) -> None:
        missing = [
            session_id
            for session_id in dict.fromkeys(session_ids)
            if session_id not in self._lineages
        ]
        if missing:
            loaded = await self._store.load_session_lineages(missing)
            for session_id in missing:
                self._lineages[session_id] = loaded.get(session_id)

    async def _lineage(self, session_id: str) -> SessionLineage | None:
        await self._load_lineages((session_id,))
        return self._lineages[session_id]

    async def affected_by(self, session_id: str) -> list[str]:
        """The session and the sessions above it whose summaries a change in it can alter.

        A session waiting on a subagent reports what the subagent waits on, as
        deep as ``_waiting_page`` follows it, so a change in the subagent can
        change every summary above it. A fork's source never waits on the fork.
        """

        affected = [session_id]
        session = await self._lineage(session_id)
        for _depth in range(MAX_FOREGROUND_CHAIN_DEPTH):
            if (
                session is None
                or session.parent_session_id is None
                or session.source is SessionExecutionSource.FORK
            ):
                break
            parent = await self._lineage(session.parent_session_id)
            if parent is None:
                break
            if not session.has_run_lineage:
                # Legacy rows lack retained ancestry, but the parent's current
                # foreground wait can still prove this exact dependency. This
                # refreshes a summary; it does not invent historical run links.
                follows = await self._retained_wait_matches(parent, session)
            else:
                follows = session.parent_run_id == parent.run_id
            if not follows:
                break
            affected.append(parent.id)
            session = parent
        return affected

    async def read(self, session_ids: tuple[str, ...]) -> tuple[SessionOverviewFacts, ...]:
        """Summarize the sessions from reads made after the scan.

        The scan read only their lineages, and kept reading events after it
        read them. Reading the rows now means every summary reflects at least
        every event up to the page's cursor, and any later change has a later
        sequence, so a later page reports the session again.
        """

        await self.load_rows(session_ids)
        return await self.read_loaded(session_ids)

    async def load_rows(self, session_ids: tuple[str, ...]) -> dict[str, Session]:
        """Read the rows ``read`` summarizes, so a caller can use them first.

        ``read_loaded`` then summarizes these rows without reading them again.
        """

        self._loaded.clear()
        await self._load_many(session_ids)
        return {
            session_id: session
            for session_id in session_ids
            if (session := self._loaded.get(session_id)) is not None
        }

    async def read_loaded(self, session_ids: tuple[str, ...]) -> tuple[SessionOverviewFacts, ...]:
        """``read`` from the rows ``load_rows`` read."""

        return await self._read(
            [session for session_id in session_ids if (session := self._loaded.get(session_id))]
        )

    async def read_listed(self, sessions: Sequence[Session]) -> tuple[SessionOverviewFacts, ...]:
        """Summarize sessions exactly as the caller's listing just read them.

        A listed row is as new as the listing's cursor, and every read below is
        made after it, so the rows are not loaded again. A session replaced since
        it was listed is left out, as ``read`` leaves it out.
        """

        listed = {session.id: session for session in sessions}
        self._loaded.clear()
        self._loaded.update(listed)
        return await self._read(list(listed.values()))

    async def _read(self, sessions: list[Session]) -> tuple[SessionOverviewFacts, ...]:
        if not sessions:
            return ()
        if self._token_usage:
            usage_by_session = await self._usages(sessions)
            steps = {
                session_id: SessionStepCounts(summary.model_steps, summary.tool_calls)
                for session_id, summary in usage_by_session.items()
            }
        else:
            usage_by_session = {}
            steps = await self._store.count_session_steps(tuple(session.id for session in sessions))
        cost_by_session = await self._costs(sessions)
        # Every read below covers the whole page, so a page costs the same
        # number of store reads however many sessions it holds.
        while sessions:
            try:
                outcomes = await self._store.summarize_outcomes(
                    tuple(session.id for session in sessions)
                )
                break
            except KeyError:
                current = await self._current(sessions)
                if len(current) == len(sessions):
                    raise
                sessions = [session for session in sessions if session.id in current]
        if not sessions:
            return ()
        if unsummarized := [session for session in sessions if session.id not in outcomes]:
            # A row can disappear during the preceding accounting reads or
            # inside this read. Only a confirmed missing/replaced incarnation
            # explains the absence; otherwise the store failed to summarize it.
            if still := await self._current(unsummarized):
                raise KeyError(f"Session not found: {sorted(still)[0]}")
            sessions = [session for session in sessions if session.id in outcomes]
            if not sessions:
                return ()
        states = await self._states(sessions)
        lineages = await resolve_fork_lineages_async(sessions, self._parents)
        markers = await self._store.query_latest_events(
            tuple(session.id for session in sessions), RELEASE_MARKER_EVENT_TYPES
        )
        runs = []
        for session, (state, children), lineage in zip(sessions, states, lineages, strict=True):
            marker = markers.get(session.id)
            release = None if marker is None else recorded_release_identity(marker.event.payload)
            runs.append(
                SessionOverviewFacts(
                    session=session,
                    status=state.status,
                    outcome=outcomes[session.id],
                    usage=usage_by_session.get(session.id),
                    cost=cost_by_session.get(session.id),
                    forked=lineage.forked,
                    fork_controller=lineage.owner() if lineage.settled else None,
                    release_id=None if release is None else release.id,
                    waiting=tuple(zip(state.pending.actions, children, strict=True)),
                    pending_complete=state.complete,
                    steps=steps.get(session.id, _NO_STEPS),
                )
            )
        # Accounting and outcome reads are separate store snapshots. A
        # replacement must never donate its cost or status to the old run.
        current = await self._current(sessions)
        return tuple(run for run in runs if run.session.id in current)

    async def _usages(self, sessions: Sequence[Session]) -> dict[str, SessionUsageSummary]:
        """Each session's usage summary; sessions without usage-bearing events are absent."""

        async def read(session_ids: tuple[str, ...]) -> dict[str, SessionUsageSummary]:
            usage = await self._store.read_usage_accounting(
                EventQuery(session_ids=session_ids), by_session=True
            )
            return {summary.session_id: summary for summary in usage.session_summaries}

        return await self._cached_totals(
            sessions, self._usage_cache, USAGE_BEARING_EVENT_TYPES, (), read
        )

    async def costs(self, sessions: Sequence[Session]) -> dict[str, SessionCostTotals]:
        """Each session's cost totals under the reader's price book.

        Empty without a price book; sessions without cost-bearing events are
        absent. With a ``cost_cache``, only sessions whose cost-bearing events
        changed since it last priced them are priced again.
        """

        return await self._costs(sessions)

    async def _costs(self, sessions: Sequence[Session]) -> dict[str, SessionCostTotals]:
        """Each session's cost totals; sessions without cost-bearing events are absent."""

        pricing = self._pricing
        if pricing is None:
            return {}
        from cayu.runtime._cost_accounting import COST_EVENT_TYPES

        async def read(session_ids: tuple[str, ...]) -> dict[str, SessionCostTotals]:
            cost = await self._store.read_cost_accounting(
                EventQuery(session_ids=session_ids), pricing, by_session=True
            )
            return {totals.session_id: totals for totals in cost.session_totals}

        return await self._cached_totals(
            sessions, self._cost_cache, COST_EVENT_TYPES, (_price_book_digest(pricing),), read
        )

    async def _cached_totals(
        self,
        sessions: Sequence[Session],
        cache: _SessionTotalsCache[_Totals] | None,
        event_types: Sequence[EventType],
        basis: tuple[str, ...],
        read: Callable[[tuple[str, ...]], Awaitable[dict[str, _Totals]]],
    ) -> dict[str, _Totals]:
        """Per-session totals, read again only for sessions whose ``event_types`` changed.

        A cached total is keyed by the session's incarnation, the count and latest
        sequence of its events of ``event_types``, and ``basis`` (what else the
        total depends on). It is kept only when those events were the same before
        and after it was read, so a total counts exactly the events its key names.
        """

        session_ids = tuple(session.id for session in sessions)
        totals: dict[str, _Totals] = {}
        keys: dict[str, tuple[Any, ...]] = {}
        unread = session_ids
        if cache is not None:
            positions = await self._store.summarize_event_positions(session_ids, event_types)
            for session in sessions:
                position = positions.get(session.id)
                if position is None:
                    continue
                key = (
                    session.id,
                    session.instance_id,
                    position.count,
                    position.last_sequence,
                    *basis,
                )
                found, cached = cache._get(key)
                if not found:
                    keys[session.id] = key
                elif cached is not None:
                    totals[session.id] = cached
            unread = tuple(
                session_id
                for session_id in session_ids
                if session_id in keys or session_id not in positions
            )
        if unread:
            fresh = await read(unread)
            totals.update(fresh)
            if cache is not None and keys:
                after = await self._store.summarize_event_positions(tuple(keys), event_types)
                for session_id, key in keys.items():
                    position = after.get(session_id)
                    if (
                        position is not None
                        and (position.count, position.last_sequence) == key[2:4]
                    ):
                        cache._put(key, fresh.get(session_id))
        return totals

    async def _load_many(self, session_ids: Iterable[str]) -> None:
        """Cache every session not yet loaded for this page with one store read."""

        missing = [
            session_id
            for session_id in dict.fromkeys(session_ids)
            if session_id not in self._loaded
        ]
        if missing:
            loaded = await self._store.load_sessions(missing)
            for session_id in missing:
                self._loaded[session_id] = loaded.get(session_id)

    async def _parents(self, session_ids: tuple[str, ...]) -> dict[str, Session | None]:
        await self._load_many(session_ids)
        return {session_id: self._loaded.get(session_id) for session_id in session_ids}

    async def _current(
        self, sessions: Sequence[Session], query: SessionQuery | None = None
    ) -> set[str]:
        """The sessions whose selected incarnation is still current, from one uncached read.

        With ``query``, a current row must also still match it.
        """

        if not sessions:
            return set()
        session_ids = tuple(session.id for session in sessions)
        if query is None:
            # Only the incarnation decides; the rows themselves are not needed.
            instances = await self._store.load_session_instance_ids(session_ids)
            return {
                session.id
                for session in sessions
                if instances.get(session.id) == session.instance_id
            }
        current = await self._store.load_sessions(session_ids)
        return {
            session.id
            for session in sessions
            if (row := current.get(session.id)) is not None
            and row.instance_id == session.instance_id
            and _session_matches(row, query)
        }

    async def _states(
        self, sessions: Sequence[Session]
    ) -> list[tuple[_StateEvidence, tuple[Session | None, ...]]]:
        """Each session's state evidence and the verified child of each pending action."""

        if not sessions:
            return []
        pending = await self._pending_page(
            [session for session in sessions if session.status in _PENDING_ACTION_STATUSES]
        )
        evidence = [pending.get(session.id, _PendingEvidence()) for session in sessions]
        waits = await self._waiting_page(list(zip(sessions, evidence, strict=True)), 0)
        return [
            (
                _StateEvidence(
                    _run_status(
                        session.status,
                        waiting.kinds,
                        delegated=any(
                            action.delegated_action is not None for action in actions.actions
                        ),
                    ),
                    actions,
                    waiting.complete,
                ),
                children,
            )
            for session, actions, (waiting, children) in zip(sessions, evidence, waits, strict=True)
        ]

    async def _pending_page(self, sessions: Sequence[Session]) -> dict[str, _PendingEvidence]:
        """``_pending`` for several sessions, reading their pages together."""

        if not sessions:
            return {}
        collected = _PendingCollector(sessions)
        cursor = None
        cursors: set[str] = set()
        while True:
            page = await self._store.query_pending_actions_for_sessions(
                collected.session_ids,
                PendingActionQuery(
                    limit=200, cursor=cursor, max_result_bytes=MAX_PENDING_ACTION_RESULT_BYTES
                ),
            )
            collected.add(page)
            if not page.has_more:
                evidence = collected.evidence()
                if cursors:
                    # Cross-session cursors sort by mutable activity time. A
                    # session updated between pages may move before the cursor;
                    # reconcile those rows through their single-session read.
                    current = await self._store.load_sessions(collected.session_ids)
                    changed = [
                        session
                        for session in sessions
                        if (row := current.get(session.id)) is not None
                        and row.instance_id == session.instance_id
                        and row.updated_at != session.updated_at
                    ]
                    refreshed = await gather_in_order(
                        lambda session=session: self._pending(session) for session in changed
                    )
                    evidence.update(
                        zip((session.id for session in changed), refreshed, strict=True)
                    )
                return evidence
            if page.next_cursor is None or page.next_cursor in cursors:
                raise SessionOverviewUnavailable("Pending session paging did not advance.")
            cursors.add(page.next_cursor)
            cursor = page.next_cursor

    async def _waiting_page(
        self, items: Sequence[tuple[Session, _PendingEvidence]], depth: int
    ) -> list[tuple[_WaitingEvidence, tuple[Session | None, ...]]]:
        """The kinds a person settles, following delegated actions to their children.

        A delegated action names the kind its child waited on when the parent
        recorded the wait, which can lag the child: a child whose answer finds an
        unfinished call needs manual recovery while its parent still names the
        question. So each child's own pending actions are read now, one level of
        delegation at a time for all the sessions together.

        Each item also gets the verified child of each of its pending actions,
        ``None`` for an action that delegates to no current child.
        """

        if not items:
            return []
        # A wait is followed only so deep; the page's own level always resolves
        # its children, since each overview names the child it waits on.
        resolved = (
            await self._delegated_children(
                [
                    (session, action)
                    for session, pending in items
                    for action in pending.actions
                    if action.delegated_action is not None
                ]
            )
            if depth < MAX_FOREGROUND_CHAIN_DEPTH
            else {}
        )
        followed = list(
            {child.id: child for child in resolved.values() if child is not None}.values()
        )
        child_pending = await self._pending_page(followed)
        child_waits = await self._waiting_page(
            [(child, child_pending[child.id]) for child in followed], depth + 1
        )
        waiting_by_child = {
            child.id: waiting for child, (waiting, _) in zip(followed, child_waits, strict=True)
        }
        # The retained waits were verified before the deeper reads. Even an
        # empty or correctly bound page can outlive its child; use an uncached
        # fence before contributing any descendant state.
        current = await self._current(followed)
        results = []
        for session, pending in items:
            kinds: set[PendingActionKind] = set()
            complete = pending.complete
            children = tuple(resolved.get((session.id, id(action))) for action in pending.actions)
            for action, child in zip(pending.actions, children, strict=True):
                if action.delegated_action is None:
                    kinds.add(action.kind)
                elif child is not None:
                    if child.id not in current:
                        complete = False
                        continue
                    kinds |= waiting_by_child[child.id].kinds
                    complete = complete and waiting_by_child[child.id].complete
            results.append((_WaitingEvidence(frozenset(kinds), complete), children))
        return results

    async def _delegated_children(
        self, delegated: Sequence[tuple[Session, PendingActionRecord]]
    ) -> dict[tuple[str, int], Session | None]:
        """The verified child each delegated action waits on, keyed by parent ID and action.

        A child counts only when its parent's retained wait names its exact
        incarnation; otherwise the action maps to ``None``.
        """

        if not delegated:
            return {}
        references = [
            (parent, action, reference.child_session_id)
            for parent, action in delegated
            if (reference := action.delegated_action) is not None
        ]
        await self._load_many(child_id for _, _, child_id in references)
        candidates = [
            (parent, action, child)
            for parent, action, child_id in references
            if (child := self._loaded.get(child_id)) is not None
        ]
        checkpoints = (
            await self._store.load_checkpoints(
                tuple(dict.fromkeys(parent.id for parent, _, _ in candidates))
            )
            if candidates
            else {}
        )
        resolved: dict[tuple[str, int], Session | None] = {
            (parent.id, id(action)): None for parent, action in delegated
        }
        for parent, action, child in candidates:
            if _wait_matches(checkpoints.get(parent.id), parent, child):
                resolved[(parent.id, id(action))] = child
        return resolved

    async def counts(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        include_eval: bool = False,
        include_forks: bool = False,
        filters: SessionAggregateFilter | None = None,
        series: TimeSeriesSpec | None = None,
    ) -> SessionOverviewCounts:
        """Complete current-read counts, not a cross-store atomic snapshot.

        The window counts sessions created in [since, until); running_now also
        includes older running sessions, and usage includes older sessions'
        usage in the window. Only those sessions are read: the window's, by
        creation time, and the older sessions that are running, waiting on a
        subagent, or recorded usage in the window. Mutable update times never
        select this population.
        """
        until = datetime.now(UTC) if until is None else until
        since = until - timedelta(days=1) if since is None else since
        if any(t.tzinfo is None or t.utcoffset() is None for t in (since, until)) or since >= until:
            raise ValueError("Overview counts require aware since < until.")
        if type(include_eval) is not bool or type(include_forks) is not bool:
            raise TypeError("Overview inclusion flags must be bools.")
        filters = (
            SessionAggregateFilter()
            if filters is None
            else SessionAggregateFilter.model_validate(filters.model_dump())
        )
        filter_query = session_query_from_aggregate_filter(filters)
        tally = _CountTally(
            since,
            until,
            () if series is None else time_buckets(since, until, series),
            priced=self._pricing is not None,
        )
        seen: set[str] = set()
        cursor = None
        cursors: set[str] = set()
        window = filter_query.model_copy(
            update={
                "created_since": since,
                "created_before": until,
                "limit": _COUNT_PAGE,
                "order_by": SessionOrder.CREATED_AT_ASC,
            }
        )
        while True:
            page = await self._store.list_sessions(window.model_copy(update={"cursor": cursor}))
            # A listed row is as new as the listing, and every read below is made
            # after it, so the window's rows are counted as listed.
            self._loaded.clear()
            self._loaded.update((session.id, session) for session in page.sessions)
            await self._count_page(
                page.sessions,
                filter_query,
                tally,
                seen,
                include_eval=include_eval,
                include_forks=include_forks,
            )
            # Rows come oldest first, so a row from ``until`` on means the window
            # is done even if a store's cursor goes on; newer rows are never read.
            if (
                page.next_cursor is None
                or not page.sessions
                or page.sessions[-1].created_at >= until
            ):
                break
            if page.next_cursor in cursors:
                raise SessionOverviewUnavailable("Session paging did not advance.")
            cursors.add(page.next_cursor)
            cursor = page.next_cursor
        older = list(await self._older_count_candidates(filter_query, since, until))
        for offset in range(0, len(older), _COUNT_PAGE):
            self._loaded.clear()
            ids = older[offset : offset + _COUNT_PAGE]
            await self._load_many(ids)
            await self._count_page(
                [
                    session
                    for session_id in ids
                    if (session := self._loaded.get(session_id)) is not None
                    # The window's own sessions come from its creation listing.
                    and session.created_at < since
                ],
                filter_query,
                tally,
                seen,
                include_eval=include_eval,
                include_forks=include_forks,
            )
        return tally.result(series)

    async def _older_count_candidates(
        self, filter_query: SessionQuery, since: datetime, until: datetime
    ) -> tuple[str, ...]:
        """Sessions created before the window that counts may still include.

        Older sessions add to counts in only two ways: as running now, and with
        usage in the window. A run is running while its row is pending, running
        or interrupting, or while it is interrupted waiting on a subagent, which
        is a delegated pending action. Usage in the window is an event of a
        usage-bearing type with a timestamp in it. Each read pages by something
        that cannot change during the request, so a row's update cannot move it
        out of a scan, and each is bounded by what it finds, not by history.
        """

        candidates: dict[str, None] = {}
        older = filter_query.model_copy(
            update={
                "created_before": since,
                "limit": _COUNT_PAGE,
                "order_by": SessionOrder.CREATED_AT_ASC,
            }
        )
        await self._list_running(older, candidates)
        async for page in self._list_pages(
            older.model_copy(update={"status": SessionStatus.INTERRUPTED})
        ):
            candidates.update(dict.fromkeys(await self._delegating(page)))
        # A run can move between running and waiting on a subagent while the
        # reads above run; listing the running rows again finds one that left
        # the interrupted scan's reach after the first listing.
        await self._list_running(older, candidates)
        await self._usage_sessions(since, until, candidates)
        return tuple(candidates)

    async def _list_running(self, query: SessionQuery, candidates: dict[str, None]) -> None:
        for status in (SessionStatus.PENDING, SessionStatus.RUNNING, SessionStatus.INTERRUPTING):
            async for page in self._list_pages(query.model_copy(update={"status": status})):
                candidates.update(dict.fromkeys(session.id for session in page))

    async def _list_pages(self, query: SessionQuery) -> AsyncIterator[list[Session]]:
        cursor = None
        cursors: set[str] = set()
        while True:
            page = await self._store.list_sessions(query.model_copy(update={"cursor": cursor}))
            yield page.sessions
            if page.next_cursor is None:
                return
            if page.next_cursor in cursors:
                raise SessionOverviewUnavailable("Session paging did not advance.")
            cursors.add(page.next_cursor)
            cursor = page.next_cursor

    async def _delegating(self, sessions: Sequence[Session]) -> list[str]:
        """The interrupted sessions that may be waiting on a subagent.

        A pending-action read pages by mutable activity time, so when one needs
        more than a page, every session of this listing page is kept and
        ``_count_page`` reads each one's pending actions itself. Sessions the
        read cannot judge are kept too, so their counts fail closed.
        """

        if not sessions:
            return []
        session_ids = tuple(session.id for session in sessions)
        page = await self._store.query_pending_actions_for_sessions(
            session_ids,
            PendingActionQuery(
                statuses=frozenset({SessionStatus.INTERRUPTED}),
                kind=PendingActionKind.DELEGATED_ACTION,
                limit=200,
                max_result_bytes=MAX_PENDING_ACTION_RESULT_BYTES,
            ),
        )
        if page.has_more:
            return list(session_ids)
        return [
            *(action.session.id for action in page.actions),
            *(issue.session_id for issue in page.issues),
        ]

    async def _usage_sessions(
        self, since: datetime, until: datetime, candidates: dict[str, None]
    ) -> None:
        """Add the sessions with a usage-bearing event in [since, until).

        Each read covers a time range in one statement, with no sequence cursor:
        a cross-session sequence cursor can hide committed events behind an open
        writer, but one statement sees every event committed before it. A range
        that fills a read is split in half and read again. A range too short to
        split, where many events share an instant, is read by the store's usage
        accounting instead, which reads every event of the range in one stable
        snapshot and names each session it found.
        """

        ranges = [(since, until)]
        while ranges:
            start, end = ranges.pop()
            heads = await self._store.query_event_heads(
                EventQuery(
                    event_types=USAGE_BEARING_EVENT_TYPES,
                    since=start,
                    until=end,
                    limit=_USAGE_HEAD_PAGE,
                )
            )
            if len(heads) < _USAGE_HEAD_PAGE:
                candidates.update(dict.fromkeys(head.session_id for head in heads))
                continue
            middle = start + (end - start) / 2
            if end - start > _MIN_USAGE_RANGE and start < middle < end:
                ranges.extend(((start, middle), (middle, end)))
                continue
            usage = await self._store.read_usage_accounting(
                EventQuery(since=start, until=end), by_session=True
            )
            candidates.update(
                dict.fromkeys(summary.session_id for summary in usage.session_summaries)
            )

    async def _count_page(
        self,
        sessions: Sequence[Session],
        filter_query: SessionQuery,
        tally: _CountTally,
        seen: set[str],
        *,
        include_eval: bool,
        include_forks: bool,
    ) -> None:
        """Add one page of sessions, already read into ``_loaded``, to the tally.

        Each read below covers the whole page, so counting costs a fixed number
        of reads per page.
        """

        since, until = tally.since, tally.until
        # Without a filter every current row matches, so the fences below need
        # only each session's incarnation, not its whole row.
        fence = None if filter_query == _UNFILTERED else filter_query
        selected: list[Session] = []
        for session in sessions:
            if not _session_matches(session, filter_query) or session.created_at >= until:
                continue
            identity = session_run_id(session)
            if identity in seen:
                continue
            seen.add(identity)
            if not include_eval and is_eval_trial(session):
                continue
            selected.append(session)
        if not include_forks and selected:
            lineages = await resolve_fork_lineages_async(selected, self._parents)
            selected = [
                session
                for session, lineage in zip(selected, lineages, strict=True)
                if not lineage.forked
            ]
        states = await self._states(selected)
        # Pending and lineage joins may outlive this incarnation. Match
        # the full overview selection fence before including its facts.
        current = await self._current(selected, fence)
        counted: list[Session] = []
        for session, (state, _) in zip(selected, states, strict=True):
            if session.id not in current:
                continue
            counted.append(session)
            if not state.complete:
                raise SessionOverviewUnavailable("Pending session evidence is unavailable.")
            tally.add_state(session, state.status)
        if not counted:
            return
        accounting_query = EventQuery(
            session_ids=tuple(session.id for session in counted), since=since, until=until
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
                None if priced is None else revalidate_model_input(priced.totals, SessionCostTotals)
            )
            if totals is not None and not isinstance(totals, SessionCostTotals):
                raise TypeError("Invalid accounting totals.")
        except (TypeError, ValueError):
            raise SessionOverviewUnavailable("Session accounting evidence is invalid.") from None
        # The selection and accounting are separate snapshots. A replacement
        # must not donate consumption to the selected incarnation.
        if len(await self._current(counted, fence)) != len(counted):
            raise SessionOverviewUnavailable("Session changed during accounting.")
        tally.add_usage(summary, totals)

    async def _pending(self, session: Session) -> _PendingEvidence:
        collected = _PendingCollector((session,))
        cursor = None
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
            collected.add(page)
            if not page.has_more:
                return collected.evidence()[session.id]
            if page.next_cursor is None or page.next_cursor in cursors:
                raise SessionOverviewUnavailable("Pending session paging did not advance.")
            cursors.add(page.next_cursor)
            cursor = page.next_cursor

    async def _retained_wait_matches(
        self, parent: Session | SessionLineage, child: Session | SessionLineage
    ) -> bool:
        # A scan asks once per event of the child; the parent's checkpoint is
        # read once per pair of incarnations.
        key = (parent.id, parent.instance_id, child.id, child.instance_id)
        matched = self._retained_waits.get(key)
        if matched is None:
            matched = _wait_matches(await self._store.load_checkpoint(parent.id), parent, child)
            self._retained_waits[key] = matched
        return matched


class _CountTally:
    """What ``counts`` adds up across its pages."""

    def __init__(
        self,
        since: datetime,
        until: datetime,
        buckets: Sequence[TimeBucket],
        *,
        priced: bool,
    ) -> None:
        self.since = since
        self.until = until
        self._buckets = tuple(buckets)
        self._starts = tuple(bucket.start_at for bucket in self._buckets)
        self._start_counts = [0 for _ in self._buckets]
        self._failed_counts = [0 for _ in self._buckets]
        self._states = dict.fromkeys(SessionOverviewStatus, 0)
        self._running = 0
        self._input_tokens = self._output_tokens = self._cached_tokens = 0
        self._tokens_complete = True
        self._cost: Decimal | None = Decimal(0) if priced else None
        self._cost_complete = True

    def add_state(self, session: Session, status: SessionOverviewStatus) -> None:
        if status is SessionOverviewStatus.RUNNING:
            self._running += 1
        if self.since <= session.created_at < self.until:
            self._states[status] += 1
            if self._buckets:
                index = bucket_index(self._starts, session.created_at)
                self._start_counts[index] += 1
                self._failed_counts[index] += int(status is SessionOverviewStatus.FAILED)

    def add_usage(self, summary: SessionUsageSummary, totals: SessionCostTotals | None) -> None:
        self._input_tokens += summary.usage.input_tokens
        self._output_tokens += summary.usage.output_tokens
        self._cached_tokens += summary.usage.cache.read_tokens
        self._tokens_complete = self._tokens_complete and summary.unmeasured_model_attempts == 0
        if totals is not None:
            assert self._cost is not None
            self._cost = add_cost_amounts(self._cost, totals.total_cost)
            self._cost_complete = self._cost_complete and (
                totals.unpriced_model_steps == 0 and totals.unpriced_auxiliary_attempts == 0
            )

    def result(self, series: TimeSeriesSpec | None) -> SessionOverviewCounts:
        return SessionOverviewCounts(
            self.since,
            self.until,
            self._running,
            self._states,
            self._input_tokens,
            self._output_tokens,
            self._cached_tokens,
            self._tokens_complete,
            self._cost,
            self._cost_complete,
            None
            if series is None
            else SessionStartSeries(
                unit=series.unit,
                timezone=series.timezone,
                buckets=tuple(
                    SessionStartBucket(
                        start_at=b.start_at, end_at=b.end_at, total_count=count, failed=failed
                    )
                    for b, count, failed in zip(
                        self._buckets, self._start_counts, self._failed_counts, strict=True
                    )
                ),
            ),
        )


class _PendingCollector:
    """Pending evidence for the sessions of one query, gathered page by page."""

    def __init__(self, sessions: Iterable[Session]) -> None:
        self._sessions = {session.id: session for session in sessions}
        self._actions: dict[str, list[PendingActionRecord]] = {key: [] for key in self._sessions}
        self._seen: dict[str, set[str]] = {key: set() for key in self._sessions}
        self._complete = {
            key: session.instance_id is not None for key, session in self._sessions.items()
        }

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(self._sessions)

    def _incomplete(self, session_id: str) -> None:
        if session_id in self._complete:
            self._complete[session_id] = False
        else:
            # Evidence that names no queried session cannot be attributed.
            self._complete = dict.fromkeys(self._complete, False)

    def add(self, page: PendingActionListResult) -> None:
        for issue in page.issues:
            self._incomplete(issue.session_id)
        for action in page.actions:
            session = self._sessions.get(action.session.id)
            # A pending query is keyed by reusable ID. Never attach another
            # incarnation's actions to the session this reader selected.
            if (
                session is None
                or session.instance_id is None
                or action.session.instance_id != session.instance_id
            ):
                self._incomplete(action.session.id)
                continue
            seen = self._seen[session.id]
            if action.attention_id is None:
                self._actions[session.id].append(action)
            elif action.attention_id not in seen:
                self._actions[session.id].append(action)
                seen.add(action.attention_id)

    def evidence(self) -> dict[str, _PendingEvidence]:
        return {
            session_id: _PendingEvidence(tuple(actions), self._complete[session_id])
            for session_id, actions in self._actions.items()
        }


def _price_book_digest(pricing: PriceBook) -> str:
    return sha256(
        canonical_durable_json_bytes(pricing.model_dump(mode="json"), "cost pricing")
    ).hexdigest()


def _wait_matches(
    checkpoint: dict[str, Any] | None,
    parent: Session | SessionLineage,
    child: Session | SessionLineage,
) -> bool:
    # A reused child ID can have the same parent. Verify the exact wait's
    # retained incarnation as well as the relationship, never just names.
    wait, _ = foreground_child_state_from_checkpoint(checkpoint)
    return (
        wait is not None
        and wait.parent_effect.session_id == parent.id
        and wait.parent_effect.session_instance_id == parent.instance_id
        and wait.child_session_id == child.id
        and wait.child_session_instance_id == child.instance_id
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
