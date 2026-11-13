"""Count inbox actions from the stored checkpoint pending-action flags.

``query_pending_actions`` rebuilds each action from its checkpoint and events,
which costs a checkpoint decode per session. A count only needs to know which
kind of action each waiting session has, and every checkpoint write already
stores that as ``pending_action_flags`` (``pending_action_checkpoint_metrics``),
indexed by ``idx_cayu_checkpoints_pending_control_action``. One grouped
statement over that index counts them, for SQLite and PostgreSQL alike.

The flags classify a checkpoint the way the list does: an approval comes
before a question, and a question before a tool round's recovery. Approvals and
questions wait only in interrupted sessions, and a completed session's leftover
round is an issue, not a recovery item. A delegated wait or a sleep is not an
inbox item, with one exception the flags can't show: a handed-over wake whose
continuation stopped on a tool a person must settle (``session.interrupted``
with ``manual_recovery_required``) is that tool's recovery. Those few sessions
are found by their events and rebuilt the way the list builds them
(``count_woken_recoveries``).

The count doesn't re-check the event ledger otherwise, so a tool round whose
every call finished, or a checkpoint whose pending state doesn't decode, still
counts once until the session moves on, where the list leaves it out or reports
an issue instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from cayu.sessions.pending_action_contracts import PendingActionKind, PendingActionQuery
from cayu.sessions.records import SessionStatus

if TYPE_CHECKING:
    from cayu.sessions.store import SessionStore

#: SQLite keeps ``pending_action_flags`` after the checkpoint document in each
#: row, so reading it from the table walks the document's overflow pages. This
#: partial index covers every column the count reads. PostgreSQL stores large
#: documents out of line and reads the flags from the row itself.
SQLITE_PENDING_ACTION_COUNTS_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_cayu_checkpoints_pending_action_counts
    ON cayu_checkpoints(session_id, pending_action_flags, pending_action_metrics_ready)
    WHERE pending_action_flags <> 0;
"""
SQLITE_PENDING_ACTION_COUNTS_INDEX = "idx_cayu_checkpoints_pending_action_counts"

_APPROVAL, _QUESTION, _ROUND, _DELEGATED, _WAKE = 1, 2, 4, 8, 16
_INTERRUPTED = SessionStatus.INTERRUPTED.value
_COMPLETED = SessionStatus.COMPLETED.value


def pending_action_counts_sql(
    query: PendingActionQuery, *, placeholder: str, index_hint: str = ""
) -> tuple[str, list[Any]]:
    """One statement returning ``approvals, questions, recovery`` for ``query``.

    It reads the waiting checkpoints first (``CROSS JOIN`` keeps that order in
    SQLite) and each one's session by key, so its cost follows waiting sessions,
    not retained ones.

    ``query.kind``, ``cursor``, ``limit`` and ``q`` don't apply to a count; the
    caller rejects ``q``, which only the decoded items can match.
    """

    where_sql, filter_params = _session_filters(
        query, placeholder, f"({_FLAGS} & {_DELEGATED | _WAKE}) = 0"
    )
    flags = _FLAGS
    params: list[Any] = [_INTERRUPTED, _INTERRUPTED, _COMPLETED, *filter_params]
    sql = f"""
        SELECT
            COALESCE(SUM(CASE
                WHEN ({flags} & {_APPROVAL}) <> 0 AND cayu_sessions.status = {placeholder}
                THEN 1 ELSE 0 END), 0) AS approvals,
            COALESCE(SUM(CASE
                WHEN ({flags} & {_APPROVAL}) = 0 AND ({flags} & {_QUESTION}) <> 0
                     AND cayu_sessions.status = {placeholder}
                THEN 1 ELSE 0 END), 0) AS questions,
            COALESCE(SUM(CASE
                WHEN ({flags} & {_APPROVAL | _QUESTION}) = 0 AND ({flags} & {_ROUND}) <> 0
                     AND cayu_sessions.status <> {placeholder}
                THEN 1 ELSE 0 END), 0) AS recovery
        FROM cayu_checkpoints {index_hint}
        CROSS JOIN cayu_sessions
        WHERE cayu_sessions.id = cayu_checkpoints.session_id AND {where_sql}
    """
    return sql, params


def woken_recovery_sessions_sql(
    query: PendingActionQuery, *, placeholder: str, dialect: Literal["sqlite", "postgres"]
) -> tuple[str, list[Any]]:
    """Interrupted, woken sessions whose events name a manual recovery.

    A sleep's flags are only the wake bit, so the count above leaves it out.
    These sessions may instead wait on a tool's recovery, which only their newest
    ``session.interrupted`` event shows (``woken_round_needs_manual_recovery``);
    the caller rebuilds them with ``query_pending_actions_for_sessions``. A plain
    sleeper costs one backward read of its own events, which stops at its pause.
    """

    if SessionStatus.INTERRUPTED not in query.statuses:
        return "", []
    required = (
        "json_type(event.payload_json, '$.manual_recovery_required') = 'true'"
        if dialect == "sqlite"
        else "event.payload -> 'manual_recovery_required' = 'true'::jsonb"
    )
    where_sql, params = _session_filters(
        query.model_copy(update={"statuses": frozenset({SessionStatus.INTERRUPTED})}),
        placeholder,
        f"({_FLAGS} & {_DELEGATED | _WAKE}) = {_WAKE}",
    )
    sql = f"""
        SELECT cayu_checkpoints.session_id
        FROM cayu_checkpoints
        CROSS JOIN cayu_sessions
        WHERE cayu_sessions.id = cayu_checkpoints.session_id AND {where_sql}
          AND (
              SELECT {required} FROM cayu_events AS event
              WHERE event.session_id = cayu_checkpoints.session_id
                AND event.event_type = 'session.interrupted'
              ORDER BY event.sequence DESC
              LIMIT 1
          ) IS TRUE
        ORDER BY cayu_checkpoints.session_id
    """
    return sql, params


async def count_woken_recoveries(
    store: SessionStore,
    counts: dict[PendingActionKind, int],
    session_ids: Sequence[str],
    query: PendingActionQuery,
) -> dict[PendingActionKind, int]:
    """Add the actions the list builds for these woken sessions to ``counts``."""

    page_query = PendingActionQuery(
        statuses=frozenset({SessionStatus.INTERRUPTED}),
        agent_name=query.agent_name,
        environment_name=query.environment_name,
        limit=_REBUILD_PAGE,
    )
    for start in range(0, len(session_ids), _REBUILD_PAGE):
        page = list(session_ids[start : start + _REBUILD_PAGE])
        cursor: str | None = None
        while True:
            listed = await store.query_pending_actions_for_sessions(
                page, page_query.model_copy(update={"cursor": cursor})
            )
            for action in listed.actions:
                if action.kind in counts:
                    counts[action.kind] += 1
            if not listed.has_more or listed.next_cursor is None:
                break
            cursor = listed.next_cursor
    return counts


_FLAGS = "cayu_checkpoints.pending_action_flags"
_REBUILD_PAGE = 200


def _session_filters(
    query: PendingActionQuery, placeholder: str, flag_filter: str
) -> tuple[str, list[Any]]:
    statuses = sorted(status.value for status in query.statuses)
    params: list[Any] = list(statuses)
    filters = [
        f"cayu_sessions.status IN ({', '.join(placeholder for _ in statuses)})",
        "cayu_checkpoints.pending_action_metrics_ready = TRUE",
        f"{_FLAGS} <> 0",
        flag_filter,
    ]
    for column, value in (
        ("id", query.session_id),
        ("agent_name", query.agent_name),
        ("environment_name", query.environment_name),
    ):
        if value is not None:
            filters.append(f"cayu_sessions.{column} = {placeholder}")
            params.append(value)
    return " AND ".join(f"({clause})" for clause in filters), params


def pending_action_counts(row: Any) -> dict[PendingActionKind, int]:
    approvals, questions, recovery = (0, 0, 0) if row is None else row
    return {
        PendingActionKind.TOOL_APPROVAL: int(approvals),
        PendingActionKind.USER_INPUT: int(questions),
        PendingActionKind.MANUAL_RECOVERY: int(recovery),
    }
