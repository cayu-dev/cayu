"""Send the reads that follow a session row lock in the lock's own round trip.

A session write locks the session row and then reads the store's clock, the
session's closure-lineage owners, its transcript cursor, its checkpoint and an
operation record, one round trip each, although none of them depends on
another's result. Pipelined with the lock read, they still run on the server
one after another and only after the lock is granted, so:

- ``clock_timestamp()`` stays after any lock wait, and
- each read takes its own statement snapshot after the lock, as separate round
  trips did.

Every read keeps its place in the operation's statement order. The operation's
checks between the reads only raise, so when one fails (or the session is
missing) the later reads have run for nothing, which costs server time but
changes no state: they are plain reads, and the transaction rolls back.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cayu._validation import copy_durable_json_object
from cayu.sessions.records import Session
from cayu.storage import _postgres_alias_checks as alias_checks
from cayu.storage import _postgres_support as pg_support
from cayu.storage._phase_timing import timed_postgres_connection

Rows = list[tuple[Any, ...]]

SESSION_LABELS_QUERY = """
    SELECT key, value
    FROM cayu_session_labels
    WHERE session_id = %s
    ORDER BY key ASC
"""


@dataclass(frozen=True, slots=True)
class FollowRead:
    """One read sent after the session read, decoded only for a loaded session."""

    query: str
    parameters: tuple[Any, ...]
    decode: Callable[[Rows], Any]


def store_now() -> FollowRead:
    return FollowRead("SELECT clock_timestamp()", (), _decode_store_now)


def closure_owners(targets: Iterable[str]) -> FollowRead:
    targets = list(targets)
    return FollowRead(
        "SELECT progress_json FROM cayu_session_closure_progress AS p "
        "WHERE root_session_id = ANY(%s) OR EXISTS "
        "(SELECT 1 FROM jsonb_array_elements(p.progress_json->'descendants') AS child "
        "WHERE child->>'session_id' = ANY(%s))",
        (targets, targets),
        decode_closure_owners,
    )


def checkpoint(session_id: str) -> FollowRead:
    return FollowRead(
        "SELECT state FROM cayu_checkpoints WHERE session_id = %s",
        (session_id,),
        decode_checkpoint,
    )


def transcript_cursor(session_id: str) -> FollowRead:
    return FollowRead(
        "SELECT transcript_seq FROM cayu_sessions WHERE id = %s",
        (session_id,),
        lambda rows: _decode_transcript_cursor(rows, session_id),
    )


def operation_record(session_id: str, idempotency_key: str) -> FollowRead:
    return FollowRead(
        "SELECT record FROM cayu_session_operations WHERE session_id = %s AND idempotency_key = %s",
        (session_id, idempotency_key),
        lambda rows: None if not rows else pg_support._json_obj(rows[0][0]),
    )


def decode_store_now(row: tuple[Any, ...] | None) -> datetime:
    if row is None or type(row[0]) is not datetime:
        raise RuntimeError("Postgres did not return authoritative session-store time.")
    return pg_support.to_utc(row[0])


def decode_closure_owners(rows: Rows) -> tuple[dict[str, Any], ...]:
    return tuple(dict(row[0]) for row in rows)


def decode_checkpoint(rows: Rows) -> dict[str, Any] | None:
    if not rows:
        return None
    return copy_durable_json_object(pg_support._json_obj(rows[0][0]), "checkpoint")


def _decode_store_now(rows: Rows) -> datetime:
    return decode_store_now(rows[0] if rows else None)


def _decode_transcript_cursor(rows: Rows, session_id: str) -> int:
    if not rows:
        raise KeyError(f"Session not found: {session_id}")
    return int(rows[0][0])


async def load_session_then_read(
    cur: Any,
    session_id: str,
    lock: str,
    reads: Sequence[FollowRead] = (),
) -> tuple[Session | None, tuple[Any, ...]]:
    """The session row, its labels and then ``reads``, in one round trip.

    The row read takes ``lock`` (``""``, ``" FOR UPDATE"`` or
    ``" FOR KEY SHARE"``). The results of ``reads`` are decoded, in order, only
    when the session exists; for a missing session they are all ``None``.
    """

    conn = timed_postgres_connection(cur.connection)
    followers = [conn.cursor() for _ in range(1 + len(reads))]
    try:
        await alias_checks.execute_pipelined(
            (
                (
                    cur,
                    f"SELECT {pg_support.SESSION_COLUMNS} FROM cayu_sessions WHERE id = %s{lock}",
                    (session_id,),
                ),
                (followers[0], SESSION_LABELS_QUERY, (session_id,)),
                *(
                    (follower, read.query, read.parameters)
                    for follower, read in zip(followers[1:], reads, strict=True)
                ),
            )
        )
        row = await cur.fetchone()
        if row is None:
            return None, tuple(None for _ in reads)
        session = pg_support.session_from_row(
            row,
            labels={label[0]: label[1] for label in await followers[0].fetchall()},
        )
        results = []
        for follower, read in zip(followers[1:], reads, strict=True):
            results.append(read.decode(list(await follower.fetchall())))
        return session, tuple(results)
    finally:
        for follower in followers:
            await follower.close()


async def read_pipelined(cur: Any, reads: Sequence[FollowRead]) -> tuple[Any, ...]:
    """Run ``reads`` in one round trip and decode them in order."""

    if not reads:
        return ()
    conn = timed_postgres_connection(cur.connection)
    followers = [cur, *(conn.cursor() for _ in reads[1:])]
    try:
        await alias_checks.execute_pipelined(
            tuple(
                (follower, read.query, read.parameters)
                for follower, read in zip(followers, reads, strict=True)
            )
        )
        return tuple(
            [
                read.decode(list(await follower.fetchall()))
                for follower, read in zip(followers, reads, strict=True)
            ]
        )
    finally:
        for follower in followers[1:]:
            await follower.close()
