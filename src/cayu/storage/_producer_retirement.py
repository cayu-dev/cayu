"""Bounded native receipt reclamation, atomic with its monotonic namespace fence."""

import json
from bisect import bisect_right

from cayu.runtime._producer_retirement import (
    ProducerCleanupReclamation,
    prepare_retirement,
    validate_retiring_receipt,
)


async def memory_retirement(store, retirement, *, authority, limit):
    retirement, key = prepare_retirement(retirement, authority, limit)
    generation = retirement.namespace.generation
    async with store._lock:
        index = store._producer_cleanup_index.get(key, [])
        end = bisect_right(index, (generation, "\U0010ffff"))
        selected = index[: min(limit, end)]
        for indexed_generation, operation in selected:
            receipt = validate_retiring_receipt(
                retirement, operation, store._producer_cleanup_receipts.get(operation)
            )
            if receipt.registration.generation != indexed_generation:
                raise ValueError("Native cleanup receipt index conflicts.")
        result = ProducerCleanupReclamation(
            retirement=retirement, removed=len(selected), remaining=end > limit
        )
        store._producer_cleanup_retirements[key] = max(
            generation, store._producer_cleanup_retirements.get(key, 0)
        )
        for _, operation in selected:
            del store._producer_cleanup_receipts[operation]
        del index[: len(selected)]
        if not index:
            store._producer_cleanup_index.pop(key, None)
        return result


async def sqlite_retirement(store, retirement, *, authority, limit):
    retirement, key = prepare_retirement(retirement, authority, limit)
    generation = retirement.namespace.generation

    def transaction(conn):
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT operation_key, generation, receipt_json FROM cayu_producer_cleanup_receipts WHERE namespace_key = ? AND generation <= ? ORDER BY generation, operation_key LIMIT ?",
                (key, generation, limit + 1),
            ).fetchall()
            for operation, indexed_generation, raw in rows:
                receipt = validate_retiring_receipt(retirement, operation, json.loads(raw))
                if receipt.registration.generation != indexed_generation:
                    raise ValueError("Native cleanup receipt index conflicts.")
            result = ProducerCleanupReclamation(
                retirement=retirement, removed=min(limit, len(rows)), remaining=len(rows) > limit
            )
            conn.execute(
                "INSERT INTO cayu_producer_cleanup_retirements (namespace_key, through_generation) VALUES (?, ?) ON CONFLICT(namespace_key) DO UPDATE SET through_generation = MAX(through_generation, excluded.through_generation)",
                (key, generation),
            )
            conn.executemany(
                "DELETE FROM cayu_producer_cleanup_receipts WHERE operation_key = ?",
                [(row[0],) for row in rows[:limit]],
            )
            conn.commit()
            return result
        except BaseException as primary:
            try:
                conn.rollback()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Producer reclamation and rollback failed.", [primary, cleanup]
                ) from None
            raise

    return await store._run_write(transaction)


async def postgres_retirement(store, retirement, *, authority, limit):
    from cayu.storage._postgres_support import _json_obj

    retirement, key = prepare_retirement(retirement, authority, limit)
    generation = retirement.namespace.generation
    await store._ensure_ready()
    async with store._connection() as conn, conn.cursor() as cur:
        try:
            await cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (key,))
            await cur.execute(
                "SELECT operation_key, generation, receipt_json FROM cayu_producer_cleanup_receipts WHERE namespace_key = %s AND generation <= %s ORDER BY generation, operation_key LIMIT %s",
                (key, generation, limit + 1),
            )
            rows = await cur.fetchall()
            for operation, indexed_generation, raw in rows:
                receipt = validate_retiring_receipt(retirement, operation, _json_obj(raw))
                if receipt.registration.generation != indexed_generation:
                    raise ValueError("Native cleanup receipt index conflicts.")
            result = ProducerCleanupReclamation(
                retirement=retirement, removed=min(limit, len(rows)), remaining=len(rows) > limit
            )
            await cur.execute(
                "INSERT INTO cayu_producer_cleanup_retirements (namespace_key, through_generation) VALUES (%s, %s) ON CONFLICT(namespace_key) DO UPDATE SET through_generation = GREATEST(cayu_producer_cleanup_retirements.through_generation, excluded.through_generation)",
                (key, generation),
            )
            await cur.executemany(
                "DELETE FROM cayu_producer_cleanup_receipts WHERE operation_key = %s",
                [(row[0],) for row in rows[:limit]],
            )
            await conn.commit()
            return result
        except BaseException as primary:
            try:
                await conn.rollback()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Producer reclamation and rollback failed.", [primary, cleanup]
                ) from None
            raise
