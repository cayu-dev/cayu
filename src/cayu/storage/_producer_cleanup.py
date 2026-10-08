"""Atomic native retention release with a session-independent exact receipt."""

from __future__ import annotations

import json
from bisect import insort

from cayu.runtime._producer_cleanup_receipt import (
    _NativeCleanupHandoff,
    cleanup_target,
    prepare_cleanup_publication,
    read_cleanup_receipt,
)
from cayu.runtime._producer_retirement import require_unretired, retirement_for, retirement_key


def _request(registration, authority, commit):
    registration, sid, key = cleanup_target(registration)
    if commit:
        if type(authority) is not _NativeCleanupHandoff:
            raise PermissionError("Native cleanup requires a source-owner handoff.")
        authority.require(registration)
    return registration, sid, key


async def memory_cleanup(store, registration, *, authority=None, commit=False):
    registration, sid, key = _request(registration, authority, commit)
    retirement = retirement_for(registration.command)
    namespace = retirement_key(retirement)
    async with store._lock:
        require_unretired(retirement, store._producer_cleanup_retirements.get(namespace))
        prior = read_cleanup_receipt(registration, store._producer_cleanup_receipts.get(key))
        if prior is not None or not commit:
            return prior
        receipt, checkpoint, operations = prepare_cleanup_publication(
            registration,
            authority=authority,
            session=store._sessions.get(sid),
            checkpoint=store._checkpoints.get(sid),
            attachment=store._session_operation_records.get(sid, {}).get(key),
            children=bool(store._child_session_keys_by_parent.get(sid)),
        )
        from cayu.sessions._producer_checkpoint import (
            ROOT_KEY,
            NativeProducerIndex,
            _publication_scope,
        )

        with _publication_scope(NativeProducerIndex.model_validate(checkpoint[ROOT_KEY])):
            prepared = store._prepare_checkpoint_store_unlocked(sid, checkpoint)
        # Prepare every value before mutating any of the authoritative maps.
        document = receipt.model_dump(mode="json")
        store._apply_checkpoint_store_unlocked(sid, prepared)
        store._session_operation_records[sid].update(operations)
        store._producer_cleanup_receipts[key] = document
        insort(
            store._producer_cleanup_index.setdefault(namespace, []),
            (retirement.namespace.generation, key),
        )
        return receipt


async def sqlite_cleanup(store, registration, *, authority=None, commit=False):
    from cayu.storage import _sqlite_records as sqlite_records

    registration, sid, key = _request(registration, authority, commit)
    retirement = retirement_for(registration.command)
    namespace = retirement_key(retirement)

    def transaction(conn):
        try:
            conn.execute("BEGIN IMMEDIATE" if commit else "BEGIN")
            fence = conn.execute(
                "SELECT through_generation FROM cayu_producer_cleanup_retirements WHERE namespace_key = ?",
                (namespace,),
            ).fetchone()
            require_unretired(retirement, None if fence is None else fence[0])
            row = conn.execute(
                "SELECT receipt_json FROM cayu_producer_cleanup_receipts WHERE operation_key = ?",
                (key,),
            ).fetchone()
            prior = read_cleanup_receipt(registration, None if row is None else json.loads(row[0]))
            if prior is not None or not commit:
                conn.rollback()
                return prior
            session = store._load_unlocked(sid)
            row = conn.execute(
                "SELECT record_json FROM cayu_session_operations WHERE session_id = ? AND idempotency_key = ?",
                (sid, key),
            ).fetchone()
            children = (
                conn.execute(
                    "SELECT 1 FROM cayu_sessions WHERE parent_session_id = ? LIMIT 1", (sid,)
                ).fetchone()
                is not None
            )
            receipt, checkpoint, operations = prepare_cleanup_publication(
                registration,
                authority=authority,
                session=session,
                checkpoint=store._load_checkpoint_unlocked(sid),
                attachment=None if row is None else json.loads(row[0]),
                children=children,
            )
            now = store._ownership_clock()
            conn.execute(
                "INSERT INTO cayu_checkpoints (session_id, state_json, updated_at, pending_action_source_bytes, pending_action_tool_call_count, pending_action_flags, pending_action_metrics_ready) VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at, pending_action_source_bytes = excluded.pending_action_source_bytes, pending_action_tool_call_count = excluded.pending_action_tool_call_count, pending_action_flags = excluded.pending_action_flags, pending_action_metrics_ready = excluded.pending_action_metrics_ready",
                sqlite_records.checkpoint_row_values(sid, checkpoint, now),
            )
            conn.executemany(
                "INSERT INTO cayu_session_operations (session_id, idempotency_key, record_json, updated_at) VALUES (?, ?, ?, ?)",
                [
                    (
                        sid,
                        item,
                        sqlite_records.json_dumps(value),
                        sqlite_records.format_datetime(now),
                    )
                    for item, value in operations.items()
                ],
            )
            conn.execute(
                "INSERT INTO cayu_producer_cleanup_receipts (operation_key, namespace_key, generation, receipt_json) VALUES (?, ?, ?, ?)",
                (key, namespace, retirement.namespace.generation, receipt.model_dump_json()),
            )
            conn.commit()
            return receipt
        except BaseException as primary:
            try:
                conn.rollback()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Producer cleanup transaction and rollback failed.", [primary, cleanup]
                ) from None
            raise

    return await (store._run_write(transaction) if commit else store._run_read(transaction))


async def postgres_cleanup(store, registration, *, authority=None, commit=False):
    from cayu.storage._postgres_support import _dumps, _json_obj

    registration, sid, key = _request(registration, authority, commit)
    retirement = retirement_for(registration.command)
    namespace = retirement_key(retirement)
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        try:
            if commit:
                # Publication and retirement share the namespace fence, including
                # a delayed cleanup whose individual receipt was already erased.
                await cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (namespace,)
                )
                # Same operation serializes replay even after its session was
                # deleted. A hash collision only over-serializes unrelated work.
                await cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (key,))
            else:
                await cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            await cur.execute(
                "SELECT through_generation FROM cayu_producer_cleanup_retirements WHERE namespace_key = %s",
                (namespace,),
            )
            fence = await cur.fetchone()
            require_unretired(retirement, None if fence is None else fence[0])
            await cur.execute(
                "SELECT receipt_json FROM cayu_producer_cleanup_receipts WHERE operation_key = %s",
                (key,),
            )
            row = await cur.fetchone()
            prior = read_cleanup_receipt(registration, None if row is None else _json_obj(row[0]))
            if prior is not None or not commit:
                await conn.rollback()
                return prior
            session = await store._load_for_update(cur, sid)
            checkpoint = await store._load_checkpoint(cur, sid)
            await cur.execute(
                "SELECT record FROM cayu_session_operations WHERE session_id = %s AND idempotency_key = %s",
                (sid, key),
            )
            row = await cur.fetchone()
            attachment = None if row is None else _json_obj(row[0])
            await cur.execute(
                "SELECT 1 FROM cayu_sessions WHERE parent_session_id = %s LIMIT 1", (sid,)
            )
            children = await cur.fetchone() is not None
            receipt, checkpoint, operations = prepare_cleanup_publication(
                registration,
                authority=authority,
                session=session,
                checkpoint=checkpoint,
                attachment=attachment,
                children=children,
            )
            now = await store._session_store_now(cur)
            await store._upsert_checkpoint(cur, sid, checkpoint, now)
            for item, value in operations.items():
                await cur.execute(
                    "INSERT INTO cayu_session_operations (session_id, idempotency_key, record, updated_at) VALUES (%s, %s, %s, %s)",
                    (sid, item, _dumps(value), now),
                )
            await cur.execute(
                "INSERT INTO cayu_producer_cleanup_receipts (operation_key, namespace_key, generation, receipt_json) VALUES (%s, %s, %s, %s)",
                (
                    key,
                    namespace,
                    retirement.namespace.generation,
                    _dumps(receipt.model_dump(mode="json")),
                ),
            )
            await conn.commit()
            return receipt
        except BaseException as primary:
            try:
                await conn.rollback()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Producer cleanup transaction and rollback failed.", [primary, cleanup]
                ) from None
            raise
