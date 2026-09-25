"""Store-native bounded record pages under one authorization snapshot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cayu._validation import require_clean_nonblank

if TYPE_CHECKING:
    from cayu.sessions.access import _SessionAccessBounds


@dataclass(frozen=True, slots=True)
class SessionRecordPage:
    records: tuple[dict[str, Any], ...]
    next_offset: int | None


def validate_page(session_id, kind, offset, limit, max_bytes):
    session_id = require_clean_nonblank(session_id, "session_id")
    if kind not in {"events", "transcript", "checkpoint", "access_audit"}:
        raise ValueError("Unsupported session record kind.")
    for value, minimum, maximum in (
        (offset, 0, 1_000_000),
        (limit, 1, 1000),
        (max_bytes, 1, 4_194_304),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError("Session record page exceeds supported bounds.")
    if kind == "checkpoint" and offset:
        raise ValueError("Checkpoint reads do not support offsets.")
    return session_id


def page(records, *, offset, limit, max_bytes):
    retained = records[:limit]
    if len(json.dumps(retained, ensure_ascii=False, separators=(",", ":")).encode()) > max_bytes:
        raise ValueError("Session record page exceeds its byte bound.")
    return SessionRecordPage(
        tuple(retained), offset + len(retained) if len(records) > limit else None
    )


async def memory_read(store, bounds, session_id, kind, offset, limit, max_bytes):
    session_id = validate_page(session_id, kind, offset, limit, max_bytes)
    async with store._lock:
        session = bounds.require_read(store._sessions.get(session_id))
        if kind == "checkpoint":
            bounds.require_action(session, "inspect_state")
            value = store._checkpoints.get(session_id)
            records = [] if value is None else [json.loads(json.dumps(value))]
        elif kind == "access_audit":
            values = sorted(
                (record["occurred_at"], key, record)
                for key, record in store._session_operation_records.get(session_id, {}).items()
                if key.startswith("cayu.resource_access.relabel:")
            )
            records = [
                json.loads(json.dumps(value[2])) for value in values[offset : offset + limit + 1]
            ]
        else:
            values = (store._events if kind == "events" else store._transcripts).get(session_id, [])
            records = [v.model_dump(mode="json") for v in values[offset : offset + limit + 1]]
        return page(records, offset=offset, limit=limit, max_bytes=max_bytes)


def _sql(kind, *, postgres):
    if kind == "access_audit":
        return (
            "cayu_session_operations",
            "record" if postgres else "record_json",
            "updated_at, idempotency_key",
            "record" if postgres else "record_json",
        )
    if kind == "events":
        if postgres:
            return "cayu_events", "event", "sequence", "event"
        from cayu.storage.sqlite import _EVENT_COLUMN_NAMES

        return "cayu_events", ", ".join(_EVENT_COLUMN_NAMES), "sequence", "payload_json"
    if kind == "transcript":
        return (
            "cayu_transcript_messages",
            "message" if postgres else "message_json",
            "session_order",
            "message" if postgres else "message_json",
        )
    return (
        "cayu_checkpoints",
        "state" if postgres else "state_json",
        "session_id",
        "state" if postgres else "state_json",
    )


async def sqlite_read(
    store, bounds: _SessionAccessBounds, session_id, kind, offset, limit, max_bytes
):
    from cayu.storage.sqlite import _event_from_row, _load_session

    session_id = validate_page(session_id, kind, offset, limit, max_bytes)
    table, columns, order, payload = _sql(kind, postgres=False)
    restriction = _record_restriction(kind)

    def read(connection):
        connection.execute("BEGIN")
        try:
            session = bounds.require_read(_load_session(connection, session_id))
            if kind == "checkpoint":
                bounds.require_action(session, "inspect_state")
            sizes = connection.execute(
                f"SELECT length(CAST({payload} AS BLOB)) FROM {table} WHERE session_id = ?{restriction} ORDER BY {order} LIMIT ? OFFSET ?",
                (session_id, limit + 1, offset),
            ).fetchall()
            if sum(row[0] for row in sizes) > max_bytes:
                raise ValueError("Session record page exceeds its byte bound.")
            rows = connection.execute(
                f"SELECT {columns} FROM {table} WHERE session_id = ?{restriction} ORDER BY {order} LIMIT ? OFFSET ?",
                (session_id, limit + 1, offset),
            ).fetchall()
            records = [
                _event_from_row(row).model_dump(mode="json")
                if kind == "events"
                else json.loads(row[0])
                for row in rows
            ]
            return page(records, offset=offset, limit=limit, max_bytes=max_bytes)
        finally:
            connection.rollback()

    return await store._run_read(read)


async def postgres_read(
    store, bounds: _SessionAccessBounds, session_id, kind, offset, limit, max_bytes
):
    session_id = validate_page(session_id, kind, offset, limit, max_bytes)
    table, columns, order, payload = _sql(kind, postgres=True)
    restriction = _record_restriction(kind).replace("%", "%%")
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        session = bounds.require_read(await store._load(cur, session_id))
        if kind == "checkpoint":
            bounds.require_action(session, "inspect_state")
        await cur.execute(
            f"SELECT octet_length({payload}::text) FROM {table} WHERE session_id = %s{restriction} ORDER BY {order} LIMIT %s OFFSET %s",
            (session_id, limit + 1, offset),
        )
        if sum(row[0] for row in await cur.fetchall()) > max_bytes:
            raise ValueError("Session record page exceeds its byte bound.")
        await cur.execute(
            f"SELECT {columns} FROM {table} WHERE session_id = %s{restriction} ORDER BY {order} LIMIT %s OFFSET %s",
            (session_id, limit + 1, offset),
        )
        records = [row[0] for row in await cur.fetchall()]
        return page(records, offset=offset, limit=limit, max_bytes=max_bytes)


def sqlite_owner_read(connection, bounds, session_id, operation, *, action="read"):
    """Keep native projections and their owner classification in one snapshot."""
    if bounds is None:
        return operation(connection)
    from cayu.storage.sqlite import _load_session

    connection.execute("BEGIN")
    try:
        bounds.require_action(_load_session(connection, session_id), action)
        return operation(connection)
    finally:
        connection.rollback()


def _record_restriction(kind):
    return (
        " AND idempotency_key LIKE 'cayu.resource_access.relabel:%'"
        if kind == "access_audit"
        else ""
    )
