"""Backend transactions for the zero-work interruption proof."""

from __future__ import annotations

import json
from itertools import islice
from typing import TYPE_CHECKING

from cayu.core.events import Event
from cayu.core.messages import Message
from cayu.runtime._zero_work_interruption import (
    MAX_EVIDENCE_ITEMS,
    ZeroWorkInterruptionPublication,
    ZeroWorkInterruptionRequest,
    prepare_zero_work_interruption,
)

if TYPE_CHECKING:
    from cayu.runtime.sessions import InMemorySessionStore
    from cayu.storage.postgres import PostgresSessionStore
    from cayu.storage.sqlite import SQLiteSessionStore


async def memory_terminalize(
    store: InMemorySessionStore, request: ZeroWorkInterruptionRequest
) -> ZeroWorkInterruptionPublication | None:
    sid = request.session.id
    async with store._lock:
        session = store._sessions.get(sid)
        if session is None:
            return None
        publication = prepare_zero_work_interruption(
            request,
            session=session,
            checkpoint=store._checkpoints.get(sid),
            events=list(store._events.get(sid, ())[: MAX_EVIDENCE_ITEMS + 1]),
            messages=list(store._transcripts.get(sid, ())[: MAX_EVIDENCE_ITEMS + 1]),
            operations=dict(
                islice(
                    store._session_operation_records.get(sid, {}).items(), MAX_EVIDENCE_ITEMS + 1
                )
            ),
            blocked=bool(
                store._deferred_interaction_inputs.get(sid)
                or store._queued_session_messages_by_idempotency.get(sid)
                or store._child_session_keys_by_parent.get(sid)
            ),
            now=store._ownership_clock(),
        )
        if publication is None or not request.commit or publication.replayed:
            return publication
        prepared_events = store._prepare_event_append_unlocked(session, publication.events)
        prepared_checkpoint = store._prepare_checkpoint_store_unlocked(sid, publication.checkpoint)
        store._apply_event_append_unlocked(
            session, prepared_events, activity_at=publication.session.updated_at
        )
        store._apply_checkpoint_store_unlocked(sid, prepared_checkpoint)
        store._session_operation_records[sid].update(publication.operations)
        store._sessions[sid] = publication.session.model_copy(deep=True)
        return publication


async def sqlite_terminalize(
    store: SQLiteSessionStore, request: ZeroWorkInterruptionRequest
) -> ZeroWorkInterruptionPublication | None:
    from cayu.storage import _sqlite_support as support
    from cayu.storage.sqlite import _append_events_in_transaction, _event_from_row

    sid = request.session.id

    def statement(conn):
        try:
            conn.execute("BEGIN IMMEDIATE" if request.commit else "BEGIN")
            session = store._load_unlocked(sid)
            if session is None:
                conn.rollback()
                return None
            blocked = any(
                conn.execute(f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1", (sid,)).fetchone()
                is not None
                for table, column in (
                    ("cayu_deferred_interaction_inputs", "session_id"),
                    ("cayu_session_message_queue", "session_id"),
                    ("cayu_sessions", "parent_session_id"),
                )
            )
            publication = prepare_zero_work_interruption(
                request,
                session=session,
                checkpoint=store._load_checkpoint_unlocked(sid),
                events=[
                    _event_from_row(r)
                    for r in conn.execute(
                        "SELECT * FROM cayu_events WHERE session_id = ? ORDER BY sequence LIMIT ?",
                        (sid, MAX_EVIDENCE_ITEMS + 1),
                    )
                ],
                messages=[
                    Message.model_validate(json.loads(r[0]))
                    for r in conn.execute(
                        "SELECT message_json FROM cayu_transcript_messages WHERE session_id = ? ORDER BY sequence LIMIT ?",
                        (sid, MAX_EVIDENCE_ITEMS + 1),
                    )
                ],
                operations={
                    r[0]: json.loads(r[1])
                    for r in conn.execute(
                        "SELECT idempotency_key, record_json FROM cayu_session_operations WHERE session_id = ? LIMIT ?",
                        (sid, MAX_EVIDENCE_ITEMS + 1),
                    )
                },
                blocked=blocked,
                now=store._ownership_clock(),
            )
            if publication is None or not request.commit or publication.replayed:
                conn.rollback()
                return publication
            now = publication.session.updated_at
            _append_events_in_transaction(conn, sid, publication.events, activity_at=now)
            conn.execute(
                "UPDATE cayu_sessions SET status = ?, run_epoch = ?, updated_at = ?, last_activity_at = ? WHERE id = ?",
                (
                    str(publication.session.status),
                    publication.session.run_epoch,
                    support.format_datetime(now),
                    support.format_datetime(now),
                    sid,
                ),
            )
            conn.execute(
                "INSERT INTO cayu_checkpoints (session_id, state_json, updated_at, pending_action_source_bytes, pending_action_tool_call_count, pending_action_flags, pending_action_metrics_ready) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at, pending_action_source_bytes = excluded.pending_action_source_bytes, pending_action_tool_call_count = excluded.pending_action_tool_call_count, pending_action_flags = excluded.pending_action_flags, pending_action_metrics_ready = excluded.pending_action_metrics_ready",
                support.checkpoint_row_values(sid, publication.checkpoint, now),
            )
            conn.executemany(
                "INSERT INTO cayu_session_operations (session_id, idempotency_key, record_json, updated_at) VALUES (?, ?, ?, ?)",
                [
                    (sid, key, support.json_dumps(record), support.format_datetime(now))
                    for key, record in publication.operations.items()
                ],
            )
            conn.commit()
            return publication
        except BaseException:
            conn.rollback()
            raise

    return await store._run_write(statement)


async def postgres_terminalize(
    store: PostgresSessionStore, request: ZeroWorkInterruptionRequest
) -> ZeroWorkInterruptionPublication | None:
    from cayu.storage.postgres import _dumps, _json_obj

    sid = request.session.id
    await store._ensure_ready()
    async with store._connection() as conn:
        try:
            async with conn.cursor() as cur:
                session = (
                    await store._load_for_update(cur, sid)
                    if request.commit
                    else await store._load(cur, sid)
                )
                if session is None:
                    await conn.rollback()
                    return None
                blocked = False
                for table, column in (
                    ("cayu_deferred_interaction_inputs", "session_id"),
                    ("cayu_session_message_queue", "session_id"),
                    ("cayu_sessions", "parent_session_id"),
                ):
                    await cur.execute(f"SELECT 1 FROM {table} WHERE {column} = %s LIMIT 1", (sid,))
                    blocked = blocked or await cur.fetchone() is not None
                checkpoint = await store._load_checkpoint(cur, sid)
                await cur.execute(
                    "SELECT event FROM cayu_events WHERE session_id = %s ORDER BY sequence LIMIT %s",
                    (sid, MAX_EVIDENCE_ITEMS + 1),
                )
                events = [Event.model_validate(_json_obj(r[0])) for r in await cur.fetchall()]
                await cur.execute(
                    "SELECT message FROM cayu_transcript_messages WHERE session_id = %s ORDER BY session_order LIMIT %s",
                    (sid, MAX_EVIDENCE_ITEMS + 1),
                )
                messages = [Message.model_validate(_json_obj(r[0])) for r in await cur.fetchall()]
                await cur.execute(
                    "SELECT idempotency_key, record FROM cayu_session_operations WHERE session_id = %s LIMIT %s",
                    (sid, MAX_EVIDENCE_ITEMS + 1),
                )
                operations = {r[0]: _json_obj(r[1]) for r in await cur.fetchall()}
                publication = prepare_zero_work_interruption(
                    request,
                    session=session,
                    checkpoint=checkpoint,
                    events=events,
                    messages=messages,
                    operations=operations,
                    blocked=blocked,
                    now=await store._session_store_now(cur),
                )
                if publication is None or not request.commit or publication.replayed:
                    await conn.rollback()
                    return publication
                now = publication.session.updated_at
                await store._append_events_with_cursor(
                    cur, sid, publication.events, expected_run_epoch=session.run_epoch
                )
                await cur.execute(
                    "UPDATE cayu_sessions SET status = %s, run_epoch = %s, updated_at = %s, last_activity_at = %s WHERE id = %s",
                    (str(publication.session.status), publication.session.run_epoch, now, now, sid),
                )
                await store._upsert_checkpoint(cur, sid, publication.checkpoint, now)
                for key, record in publication.operations.items():
                    await cur.execute(
                        "INSERT INTO cayu_session_operations (session_id, idempotency_key, record, updated_at) VALUES (%s, %s, %s, %s)",
                        (sid, key, _dumps(record), now),
                    )
                await conn.commit()
                return publication
        except BaseException:
            await conn.rollback()
            raise
