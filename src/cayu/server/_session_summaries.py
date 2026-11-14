"""Overview documents stored on ``cayu_session_summaries`` rows.

``SessionSummaries`` serves a page of overviews from stored summary rows. A row's
document is current while the row is settled: its generation is the one the
document was derived from. The page's other rows are derived again together, by
one read of the reference reader, ``SessionOverviewReader``, through a store view
that records the stamp of every session it reads. Each document is stored with a
compare-and-set on its row's generation, together with the stamps of the
sessions it depends on. A stored document is current only while every one of
those still has its stamp, so a write committed meanwhile, to the session or to
anything it read, leaves the row to be derived again. A page costs the same
number of store reads and one settlement however many of its rows are stale.

A session's overview reads only sessions of its own lineage tree (its
ancestors, its descendants and theirs), so each document depends on the
sessions of its tree the read touched, not on the rest of the page.

Usage is extended incrementally: while the session's ``usage_epoch`` (advanced by
changed or deleted events) is the one the document recorded, only events after
the recorded sequence are read and combined. Usage summaries combine exactly,
which is how the reference reader itself folds a session's events page by page.

``check_session_summary`` compares a stored document with a full recompute; the
CLI's ``cayu session overview --recompute`` and the property tests use it.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cayu.budgets.usage import SessionUsageSummary, combine_session_usage_summaries
from cayu.sessions.overview import (
    SessionOverviewFacts,
    SessionOverviewReader,
    SessionOverviewStatus,
)
from cayu.sessions.records import PendingActionKind, Session
from cayu.sessions.store import SessionStore
from cayu.sessions.summary_rows import (
    SessionSummaryRow,
    SessionSummaryStamp,
    SummarySettlement,
)

#: Bump when a document's fields or their derivation change; older documents
#: are then derived again on first read.
SUMMARY_DOCUMENT_VERSION = 1

# Fields that depend on how a document was computed, not on what it says.
_BOOKKEEPING = frozenset({"version", "usage_through", "usage_epoch", "instance_id"})

UsesBrowser = Callable[[SessionStore, list[Session]], Awaitable[dict[str, bool]]]


def _session_ids(name: str) -> Callable[..., Iterable[str]]:
    """Read the session IDs a store method is asked about from its arguments."""

    def extract(*args: Any, **kwargs: Any) -> Iterable[str]:
        value = args[0] if args else kwargs[name]
        if isinstance(value, str):
            return (value,)
        ids: list[str] = []
        for item in value:
            ids.append(item if isinstance(item, str) else item.id)
        return ids

    return extract


def _event_query_ids(*args: Any, **kwargs: Any) -> Iterable[str]:
    query = args[0] if args else kwargs["query"]
    ids = list(query.session_ids or ())
    if query.session_id is not None:
        ids.append(query.session_id)
    if not ids:
        raise ValueError("A summary derivation read events across sessions.")
    return ids


def _pending_query_ids(*args: Any, **kwargs: Any) -> Iterable[str]:
    query = args[0] if args else kwargs.get("query")
    if query is None or query.session_id is None:
        raise ValueError("A summary derivation read pending actions across sessions.")
    return (query.session_id,)


# Every store read a derivation may make, with the sessions it reads. A read not
# listed here fails, so a new dependency of the reference reader cannot go
# unrecorded.
_RECORDED_READS: dict[str, Callable[..., Iterable[str]]] = {
    "load": _session_ids("session_id"),
    "load_sessions": _session_ids("session_ids"),
    "load_session_lineages": _session_ids("session_ids"),
    "load_session_instance_ids": _session_ids("session_ids"),
    "load_checkpoint": _session_ids("session_id"),
    "load_checkpoints": _session_ids("session_ids"),
    "load_checkpoint_fields": _session_ids("session_ids"),
    "load_session_operations": _session_ids("session_ids"),
    "query_pending_actions_for_sessions": _session_ids("session_ids"),
    "query_pending_actions": _pending_query_ids,
    "summarize_outcomes": _session_ids("session_ids"),
    "summarize_event_positions": _session_ids("session_ids"),
    "count_session_steps": _session_ids("session_ids"),
    "query_latest_events": _session_ids("session_ids"),
    "count_child_runs": _session_ids("parents"),
    "read_usage_accounting": _event_query_ids,
}


class _RecordingStore:
    """A store view that records the stamp of every session read through it.

    A session's stamp is read before its first read, so whatever the read
    returns is at least as new as the stamp; a later change shows as a newer one.
    The sessions being derived start with their rows' stamps, read first.
    """

    def __init__(self, store: SessionStore, known: Mapping[str, SessionSummaryStamp]) -> None:
        self._store = store
        self.stamps: dict[str, SessionSummaryStamp] = dict(known)

    async def _note(self, session_ids: Iterable[str]) -> None:
        unseen = [
            session_id for session_id in dict.fromkeys(session_ids) if session_id not in self.stamps
        ]
        if unseen:
            found = await self._store.load_session_summary_stamps(unseen)
            for session_id in unseen:
                self.stamps[session_id] = found.get(session_id)

    def __getattr__(self, name: str) -> Any:
        extract = _RECORDED_READS.get(name)
        if extract is None:
            raise AttributeError(f"Session summary derivation cannot record store.{name}.")
        method = getattr(self._store, name)

        async def recorded(*args: Any, **kwargs: Any) -> Any:
            await self._note(extract(*args, **kwargs))
            return await method(*args, **kwargs)

        return recorded


def _waiting_counts(facts: SessionOverviewFacts) -> dict[str, int]:
    counts = dict.fromkeys((kind.value for kind in PendingActionKind), 0)
    for action, _child in facts.waiting:
        counts[action.kind.value] += 1
    return counts


def _usage_json(usage: SessionUsageSummary | None) -> dict[str, Any] | None:
    return None if usage is None else usage.model_dump(mode="json")


def _usage_from(value: Any) -> SessionUsageSummary | None:
    # A JSON-mode dump writes aggregate counts as strings, which only JSON input parses.
    return (
        None
        if value is None
        else SessionUsageSummary.model_validate_json(json.dumps(value, separators=(",", ":")))
    )


def _document(
    facts: SessionOverviewFacts,
    *,
    usage: SessionUsageSummary | None,
    browser: bool,
    subagents: int,
) -> dict[str, Any]:
    error = facts.outcome.details.get("error")
    ended_at = facts.ended_at
    return {
        "version": SUMMARY_DOCUMENT_VERSION,
        "instance_id": facts.session.instance_id,
        "state": facts.status.value,
        "reason": facts.outcome.reason,
        "error": error if type(error) is str else None,
        "ended_at": None if ended_at is None else ended_at.isoformat(),
        "forked": facts.forked,
        "fork_controller": facts.fork_controller,
        "release_id": facts.release_id,
        "pending_complete": facts.pending_complete,
        "waiting": _waiting_counts(facts),
        "browser": browser,
        "subagents": subagents,
        "usage": _usage_json(usage),
    }


@dataclass(frozen=True, slots=True)
class SummaryDocument:
    """A stored or freshly derived overview document, read as typed values."""

    values: dict[str, Any]

    @property
    def state(self) -> SessionOverviewStatus:
        return SessionOverviewStatus(self.values["state"])

    @property
    def reason(self) -> str:
        return self.values["reason"]

    @property
    def error(self) -> str | None:
        return self.values["error"]

    @property
    def ended_at(self) -> datetime | None:
        value = self.values["ended_at"]
        return None if value is None else datetime.fromisoformat(value)

    @property
    def forked(self) -> bool:
        return self.values["forked"]

    @property
    def pending_complete(self) -> bool:
        return self.values["pending_complete"]

    @property
    def browser(self) -> bool:
        return self.values["browser"]

    @property
    def subagents(self) -> int:
        return self.values["subagents"]

    @property
    def usage(self) -> SessionUsageSummary | None:
        return _usage_from(self.values["usage"])


def comparable(document: dict[str, Any]) -> dict[str, Any]:
    """The document without bookkeeping, as a recompute must reproduce it."""

    return {key: value for key, value in document.items() if key not in _BOOKKEEPING}


def _is_current(row: SessionSummaryRow) -> bool:
    """Whether the row's stored document is current.

    A settled row is current: any change to the session or to a session its
    document read advances its generation in the writing transaction.
    """

    return (
        row.settled
        and row.derived is not None
        and row.derived.get("version") == SUMMARY_DOCUMENT_VERSION
        and row.derived.get("instance_id") == row.instance_id
    )


def _incremental_base(row: SessionSummaryRow) -> tuple[SessionUsageSummary | None, int] | None:
    """The stored usage and the sequence it covers, while the row's events only grew."""

    previous = row.derived
    if (
        previous is None
        or previous.get("version") != SUMMARY_DOCUMENT_VERSION
        or previous.get("instance_id") != row.instance_id
        or previous.get("usage_epoch") != row.usage_epoch
        or type(previous.get("usage_through")) is not int
    ):
        return None
    return _usage_from(previous.get("usage")), previous["usage_through"]


@dataclass(frozen=True, slots=True)
class _Derived:
    documents: dict[str, dict[str, Any]]
    settled: frozenset[str]
    #: The stamps each document's reads were made against, by session.
    dependencies: dict[str, dict[str, SessionSummaryStamp]]


class SessionSummaries:
    """Serves overview documents from summary rows, deriving the ones not current."""

    def __init__(self, store: SessionStore, *, uses_browser: UsesBrowser) -> None:
        self._store = store
        self._uses_browser = uses_browser

    async def documents(self, sessions: Sequence[Session]) -> dict[str, SummaryDocument] | None:
        """Each listed session's current document, ``None`` when the store keeps no rows.

        A session whose row is missing or holds another incarnation than the one
        listed is absent: it was deleted or replaced after the listing.
        """

        session_ids = tuple(session.id for session in sessions)
        try:
            rows = await self._store.load_session_summaries(session_ids)
        except NotImplementedError:
            return None
        documents: dict[str, SummaryDocument] = {}
        stale: list[SessionSummaryRow] = []
        for session in sessions:
            row = rows.get(session.id)
            if row is None or row.instance_id != session.instance_id:
                continue
            if _is_current(row) and row.derived is not None:
                documents[session.id] = SummaryDocument(row.derived)
            else:
                stale.append(row)
        derived = await self._derive(stale)
        for session_id, document in derived.documents.items():
            documents[session_id] = SummaryDocument(document)
        return {
            session_id: documents[session_id]
            for session_id in session_ids
            if session_id in documents
        }

    async def derive(
        self, session_id: str, row: SessionSummaryRow | None = None
    ) -> dict[str, Any] | None:
        """Derive one session's document with the reference reader and try to store it.

        Returns the document, stored or not (a write committed meanwhile leaves
        the row to be derived again), or ``None`` when the session is gone.
        """

        if row is None:
            rows = await self._store.load_session_summaries((session_id,))
            row = rows.get(session_id)
            if row is None:
                return None
        return (await self._derive([row])).documents.get(session_id)

    async def _derive(self, rows: Sequence[SessionSummaryRow]) -> _Derived:
        """Derive the rows' documents in one reference read and settle them together."""

        if not rows:
            return _Derived({}, frozenset(), {})
        by_id = {row.session_id: row for row in rows}
        # Read after the rows, so each session is at least as new as its row's stamp.
        loaded = await self._store.load_sessions(tuple(by_id))
        sessions = [
            session
            for session_id, row in by_id.items()
            if (session := loaded.get(session_id)) is not None
            and session.instance_id == row.instance_id
        ]
        recorder = _RecordingStore(
            self._store, {session.id: by_id[session.id].stamp for session in sessions}
        )
        computed = await self._compute(recorder, sessions)
        if not computed:
            return _Derived({}, frozenset(), {})
        usage = await self._usage([fact.session for fact, _, _ in computed], by_id)
        documents: dict[str, dict[str, Any]] = {}
        for fact, browser, subagents in computed:
            session_id = fact.session.id
            session_usage, through = usage[session_id]
            document = _document(fact, usage=session_usage, browser=browser, subagents=subagents)
            document["usage_through"] = through
            document["usage_epoch"] = by_id[session_id].usage_epoch
            documents[session_id] = document
        trees = await self._trees([fact.session for fact, _, _ in computed], tuple(recorder.stamps))
        settlements = [
            SummarySettlement(
                session_id=session_id,
                instance_id=by_id[session_id].instance_id,
                generation=by_id[session_id].source_generation,
                state=document["state"],
                forked=document["forked"],
                derived=document,
                dependencies={
                    other: stamp
                    for other, stamp in recorder.stamps.items()
                    if other != session_id and trees[other] == trees[session_id]
                },
            )
            for session_id, document in documents.items()
        ]
        settled = await self._store.settle_session_summaries(settlements)
        return _Derived(
            documents,
            settled,
            {item.session_id: dict(item.dependencies) for item in settlements},
        )

    async def _compute(
        self, store: Any, sessions: list[Session]
    ) -> list[tuple[SessionOverviewFacts, bool, int]]:
        """The reference facts, browser use and child runs of each session still present.

        A session deleted or replaced during the read is left out, as the
        reference read leaves it out; any other store failure propagates.
        """

        while sessions:
            try:
                facts = await SessionOverviewReader(store, token_usage=False).read_listed(sessions)
                listed = [fact.session for fact in facts]
                browser = await self._uses_browser(store, listed)
                subagents = await store.count_child_runs(listed)
            except KeyError:
                current = await self._store.load_session_instance_ids(
                    tuple(session.id for session in sessions)
                )
                kept = [
                    session
                    for session in sessions
                    if current.get(session.id) == session.instance_id
                ]
                if len(kept) == len(sessions):
                    raise
                sessions = kept
                continue
            return [
                (fact, browser.get(fact.session.id, False), subagents.get(fact.session.id, 0))
                for fact in facts
            ]
        return []

    async def _usage(
        self, sessions: Sequence[Session], rows: Mapping[str, SessionSummaryRow]
    ) -> dict[str, tuple[SessionUsageSummary | None, int]]:
        """Each session's usage and the sequence it covers, from one read of new events."""

        bases: dict[str, SessionUsageSummary | None] = {}
        afters: dict[str, int] = {}
        for session in sessions:
            row = rows.get(session.id)
            base = None if row is None else _incremental_base(row)
            if base is None:
                afters[session.id] = 0
            else:
                bases[session.id], afters[session.id] = base
        deltas = await self._store.read_session_usage_deltas(afters)
        usage: dict[str, tuple[SessionUsageSummary | None, int]] = {}
        for session_id, after in afters.items():
            base = bases.get(session_id)
            delta = deltas.summaries.get(session_id)
            through = max(after, deltas.through_sequences.get(session_id, after))
            if base is None or delta is None:
                usage[session_id] = (delta if base is None else base, through)
            else:
                usage[session_id] = (
                    combine_session_usage_summaries(session_id, (base, delta)),
                    through,
                )
        return usage

    async def _trees(self, sessions: Sequence[Session], read: Iterable[str]) -> dict[str, str]:
        """The root of each session's lineage tree, by session.

        A parent link changes only with a write that advances the session's own
        generation and its old parent's, and a read reaches another session of
        its tree only through the sessions between them, which it records too. So
        a link that moved after the read either changes a stamp the document
        recorded or leaves the session in the tree the read saw.
        """

        parents: dict[str, str | None] = {
            session.id: session.parent_session_id for session in sessions
        }
        level = [
            session_id
            for session_id in dict.fromkeys(
                [*read, *(parent for parent in parents.values() if parent is not None)]
            )
            if session_id not in parents
        ]
        while level:
            lineages = await self._store.load_session_lineages(level)
            for session_id in level:
                lineage = lineages.get(session_id)
                parents[session_id] = None if lineage is None else lineage.parent_session_id
            level = list(
                dict.fromkeys(
                    parent
                    for session_id in level
                    if (parent := parents[session_id]) is not None and parent not in parents
                )
            )
        roots: dict[str, str] = {}
        for session_id in parents:
            path: list[str] = []
            node = session_id
            while node not in roots and node not in path:
                path.append(node)
                parent = parents.get(node)
                if parent is None:
                    break
                node = parent
            root = roots.get(node, node)
            for visited in path:
                roots[visited] = root
        return roots

    async def recompute(self, session_id: str) -> dict[str, Any] | None:
        """The document a full reference read gives now, without storing it."""

        loaded = await self._store.load_sessions((session_id,))
        session = loaded.get(session_id)
        if session is None:
            return None
        computed = await self._compute(self._store, [session])
        if not computed:
            return None
        ((fact, browser, subagents),) = computed
        deltas = await self._store.read_session_usage_deltas({session_id: 0})
        document = _document(
            fact,
            usage=deltas.summaries.get(session_id),
            browser=browser,
            subagents=subagents,
        )
        document["usage_through"] = deltas.through_sequences.get(session_id, 0)
        document["usage_epoch"] = None
        return document


@dataclass(frozen=True, slots=True)
class SummaryCheck:
    session_id: str
    stored: dict[str, Any] | None
    current: bool
    recomputed: dict[str, Any] | None
    #: Whether the session stayed unchanged from the stored document to the
    #: recompute; a check of a session that kept changing compares nothing.
    stable: bool = True

    @property
    def matches(self) -> bool:
        if not self.stable:
            return True
        if self.recomputed is None:
            return self.stored is None
        return self.stored is not None and comparable(self.stored) == comparable(self.recomputed)


_CHECK_ATTEMPTS = 3


async def check_session_summary(
    summaries: SessionSummaries, session_id: str, *, recompute: bool = True
) -> SummaryCheck:
    """Compare the session's served document with a full recompute.

    The served document is the stored one when it is current, else one derived
    now, exactly as a page read serves it, and stored when the store can. Without
    ``recompute``, ``recomputed`` is ``None``. The check is ``stable`` when the
    session and every session the served document read kept their stamps until
    the recompute finished; a session that keeps changing, as a running one
    does, is checked again and reported as not ``stable``.
    """

    store = summaries._store
    check = SummaryCheck(session_id, None, False, None)
    for _attempt in range(_CHECK_ATTEMPTS):
        rows = await store.load_session_summaries((session_id,))
        row = rows.get(session_id)
        current = row is not None and _is_current(row)
        stored: dict[str, Any] | None = None
        watched: dict[str, SessionSummaryStamp] = {}
        if row is not None:
            watched[session_id] = row.stamp
            if current:
                stored = row.derived
            else:
                derived = await summaries._derive([row])
                stored = derived.documents.get(session_id)
                watched.update(derived.dependencies.get(session_id, {}))
        if not recompute:
            return SummaryCheck(session_id, stored, current, None)
        recomputed = await summaries.recompute(session_id)
        after = await store.load_session_summary_stamps(tuple(watched) or (session_id,))
        stable = all(after.get(key) == stamp for key, stamp in watched.items())
        if row is None:
            stable = after.get(session_id) is None
        elif current:
            # A current row's dependencies are checked by reading it again.
            again = (await store.load_session_summaries((session_id,))).get(session_id)
            stable = stable and again is not None and _is_current(again)
        check = SummaryCheck(session_id, stored, current, recomputed, stable)
        if stable:
            return check
    return check


__all__ = [
    "SUMMARY_DOCUMENT_VERSION",
    "SessionSummaries",
    "SummaryCheck",
    "SummaryDocument",
    "check_session_summary",
    "comparable",
]
