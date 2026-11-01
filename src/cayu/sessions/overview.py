"""Provider-neutral session state and batched current-incarnation facts.

The reader owns store joins. Audience-specific redaction and wire models belong to
its callers; neither the application facade nor a server is a dependency here.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import Field, model_validator

from cayu._time_buckets import (
    MAX_TIME_BUCKETS,
    TimeBucket,
    TimeSeriesSpec,
    bucket_index,
    time_buckets,
)
from cayu._validation import revalidate_model_input
from cayu.budgets.pricing import PriceBook, SessionCostTotals, add_cost_amounts
from cayu.budgets.usage import AggregateCount, SessionUsageSummary
from cayu.releases import recorded_release_identity
from cayu.sessions._foreground_child_checkpoint import (
    MAX_FOREGROUND_CHAIN_DEPTH,
    foreground_child_state_from_checkpoint,
)
from cayu.sessions._run_identity import (
    RUN_LINEAGE_METADATA_KEY,
    is_eval_trial,
    session_run_id,
    session_run_lineage,
)
from cayu.sessions._supervision_projection import RELEASE_MARKER_EVENT_TYPES
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.fork_overlays import resolve_fork_lineage_async, resolve_fork_lineages_async
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
    _session_matches,
    session_query_from_aggregate_filter,
)
from cayu.sessions.records import PendingActionKind, Session, SessionStatus

if TYPE_CHECKING:
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

_T = TypeVar("_T")


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


class SessionOverviewReader:
    """Summarizes the sessions of one page, sharing session loads among them."""

    def __init__(self, store: SessionStore, *, pricing: PriceBook | None = None) -> None:
        self._store = store
        self._pricing = pricing
        self._loaded: dict[str, Session | None] = {}
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

    async def affected_by(self, session_id: str) -> list[str]:
        """The session and the sessions above it whose summaries a change in it can alter.

        A session waiting on a subagent reports what the subagent waits on, as
        deep as ``_waiting_kinds`` follows it, so a change in the subagent can
        change every summary above it. A fork's source never waits on the fork.
        """

        affected = [session_id]
        session = await self._load(session_id)
        for _depth in range(MAX_FOREGROUND_CHAIN_DEPTH):
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
        await self._load_many(session_ids)
        sessions = [
            session for session_id in session_ids if (session := self._loaded.get(session_id))
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
        pending = await self._pending_page(
            [session for session in sessions if session.status in _PENDING_ACTION_STATUSES]
        )
        evidence = [pending.get(session.id, _PendingEvidence()) for session in sessions]
        waits = await self._waiting_page(list(zip(sessions, evidence, strict=True)), 0)
        lineages = await resolve_fork_lineages_async(sessions, self._parents)
        markers = await self._store.query_latest_events(
            tuple(session.id for session in sessions), RELEASE_MARKER_EVENT_TYPES
        )
        runs = []
        for session, actions, (waiting, children), lineage in zip(
            sessions, evidence, waits, lineages, strict=True
        ):
            marker = markers.get(session.id)
            release = None if marker is None else recorded_release_identity(marker.event.payload)
            runs.append(
                SessionOverviewFacts(
                    session=session,
                    status=_run_status(
                        session.status,
                        waiting.kinds,
                        delegated=any(
                            action.delegated_action is not None for action in actions.actions
                        ),
                    ),
                    outcome=outcomes[session.id],
                    usage=usage_by_session.get(session.id),
                    cost=cost_by_session.get(session.id),
                    forked=lineage.forked,
                    fork_controller=lineage.owner() if lineage.settled else None,
                    release_id=None if release is None else release.id,
                    waiting=tuple(zip(actions.actions, children, strict=True)),
                    pending_complete=waiting.complete,
                )
            )
        # Accounting and outcome reads are separate store snapshots. A
        # replacement must never donate its cost or status to the old run.
        current = await self._current(sessions)
        return tuple(run for run in runs if run.session.id in current)

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

    async def _current(self, sessions: Sequence[Session]) -> set[str]:
        """The sessions whose selected incarnation is still current, from one uncached read."""

        if not sessions:
            return set()
        current = await self._store.load_sessions(tuple(session.id for session in sessions))
        return {
            session.id
            for session in sessions
            if (row := current.get(session.id)) is not None
            and row.instance_id == session.instance_id
        }

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
        """``_waiting_kinds`` for several sessions, one level of delegation at a time.

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
        """``_delegated_child`` for several actions, keyed by parent ID and action identity."""

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
        includes older running sessions. Pages are followed through the upper
        creation boundary. Mutable update times never paginate this population.
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
        buckets = () if series is None else time_buckets(since, until, series)
        starts = tuple(bucket.start_at for bucket in buckets)
        start_counts = [0 for _ in buckets]
        failed_counts = [0 for _ in buckets]
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
                filter_query.model_copy(
                    update={"limit": 200, "cursor": cursor, "order_by": SessionOrder.CREATED_AT_ASC}
                )
            )
            eligible = [session for session in page.sessions if session.created_at < until]
            reached_until = len(eligible) != len(page.sessions)
            expected = {session.id: session.instance_id for session in eligible}
            self._loaded.clear()
            accounting_sessions: dict[str, str | None] = {}
            for session_id, instance_id in expected.items():
                session = await self._load(session_id)
                if (
                    session is None
                    or session.instance_id != instance_id
                    or not _session_matches(session, filter_query)
                ):
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
                if (
                    current is None
                    or current.instance_id != session.instance_id
                    or not _session_matches(current, filter_query)
                ):
                    continue
                accounting_sessions[session.id] = session.instance_id
                if not state.complete:
                    raise SessionOverviewUnavailable("Pending session evidence is unavailable.")
                if state.status is SessionOverviewStatus.RUNNING:
                    running += 1
                if session.created_at >= since:
                    counts[state.status] += 1
                    if buckets:
                        index = bucket_index(starts, session.created_at)
                        start_counts[index] += 1
                        failed_counts[index] += int(state.status is SessionOverviewStatus.FAILED)
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
                    if (
                        current is None
                        or current.instance_id != instance_id
                        or not _session_matches(current, filter_query)
                    ):
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
            None
            if series is None
            else SessionStartSeries(
                unit=series.unit,
                timezone=series.timezone,
                buckets=tuple(
                    SessionStartBucket(
                        start_at=b.start_at, end_at=b.end_at, total_count=count, failed=failed
                    )
                    for b, count, failed in zip(buckets, start_counts, failed_counts, strict=True)
                ),
            ),
        )

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
                depth < MAX_FOREGROUND_CHAIN_DEPTH
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
        return _wait_matches(await self._store.load_checkpoint(parent.id), parent, child)

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


def _wait_matches(checkpoint: dict[str, Any] | None, parent: Session, child: Session) -> bool:
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
