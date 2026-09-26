"""Atomic content-free producer observation on qualified native stores."""

import json

from cayu.runtime._producer_progress import progress_from_snapshot, published_progress
from cayu.runtime._producer_release import release_from_snapshot, release_read_target


def _project(command, kind, session, checkpoint, attachment):
    if kind is None:
        return release_from_snapshot(command, session, checkpoint, attachment)
    return progress_from_snapshot(command, kind, session, checkpoint, attachment)


async def memory_observation(store, command, *, kind=None):
    if kind == "published":
        return await published_progress(store, command)
    command, sid, key = release_read_target(command)
    async with store._lock:
        return _project(
            command,
            kind,
            store._sessions.get(sid),
            store._checkpoints.get(sid),
            store._session_operation_records.get(sid, {}).get(key),
        )


async def sqlite_observation(store, command, *, kind=None):
    if kind == "published":
        return await published_progress(store, command)
    from cayu.storage.sqlite import _load_checkpoint_state, _load_session

    command, sid, key = release_read_target(command)

    def query(connection):
        with connection:
            connection.execute("BEGIN")
            session = _load_session(connection, sid)
            checkpoint = _load_checkpoint_state(connection, sid)
            row = connection.execute(
                "SELECT record_json FROM cayu_session_operations "
                "WHERE session_id = ? AND idempotency_key = ?",
                (sid, key),
            ).fetchone()
            return _project(
                command, kind, session, checkpoint, None if row is None else json.loads(row[0])
            )

    return await store._run_read(query)


async def postgres_observation(store, command, *, kind=None):
    if kind == "published":
        return await published_progress(store, command)
    from cayu.storage.postgres import _json_obj

    command, sid, key = release_read_target(command)
    await store._ensure_ready()
    async with store._connection() as connection, connection.cursor() as cursor:
        await cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        session = await store._load(cursor, sid)
        checkpoint = await store._load_checkpoint(cursor, sid)
        await cursor.execute(
            "SELECT record FROM cayu_session_operations "
            "WHERE session_id = %s AND idempotency_key = %s",
            (sid, key),
        )
        row = await cursor.fetchone()
        return _project(
            command, kind, session, checkpoint, None if row is None else _json_obj(row[0])
        )
