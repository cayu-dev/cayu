"""Native PostgreSQL recall and context-exposure persistence.

Complete evidence operations retain closure admission, replay and writes
in their native transactions. Composition supplies explicit capabilities.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

from cayu.memory.evidence import (
    MAX_RECALL_RECEIPT_ITEMS,
    ContextExposure,
    ContextExposurePage,
    ContextExposureTransitionConflict,
    ContextExposureTransitionRequest,
    RecallEvidenceConflict,
    RecallEvidenceQuery,
    RecallItemExposure,
    RecallReceipt,
    RecallReceiptPage,
    append_context_exposure_transition,
    context_exposure_creation_matches,
    context_exposure_transition_replays,
    copy_context_exposure,
    copy_recall_item_exposure,
    copy_recall_receipt,
    decode_recall_evidence_cursor,
    encode_recall_evidence_cursor,
    memory_evidence_document_bytes,
    recall_item_exposure_matches_receipt_item,
    require_memory_evidence_id,
    require_memory_evidence_session_id,
    validate_context_exposure_carried_receipt_scope,
    validate_context_exposure_receipt_scope,
    validate_new_context_exposure,
)
from cayu.sessions.base import _check_closure_lineage_owner
from cayu.storage import _postgres_support as pg_support
from cayu.storage._postgres_transcript import PostgresConnection


async def create_recall_receipt(
    connect: PostgresConnection,
    receipt: RecallReceipt,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
    lock_closure_lineage: Callable[[Any], Awaitable[None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
) -> RecallReceipt:
    copied = copy_recall_receipt(receipt)
    document = memory_evidence_document_bytes(copied, "recall receipt")
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await lock_closure_lineage(cur)
                await _lock_evidence_id(
                    cur,
                    f"recall-receipt:{copied.receipt_id}",
                )
                await cur.execute(
                    """
                        SELECT receipt_id, session_id, interaction_id, model_step_id,
                               created_at, receipt_json, document_bytes
                        FROM cayu_recall_receipts
                        WHERE receipt_id = %s
                        """,
                    (copied.receipt_id,),
                )
                row = await cur.fetchone()
                if row is not None:
                    current = recall_receipt_from_row(row)
                    if memory_evidence_document_bytes(current, "stored recall receipt") != document:
                        raise RecallEvidenceConflict("Recall receipt", copied.receipt_id)
                    await conn.commit()
                    return current
                for owner in await closure_owners(cur, (copied.session_id,)):
                    _check_closure_lineage_owner(owner, (copied.session_id,))
                await cur.execute(
                    "SELECT 1 FROM cayu_sessions WHERE id = %s FOR KEY SHARE",
                    (copied.session_id,),
                )
                if await cur.fetchone() is None:
                    raise KeyError(f"Session not found: {copied.session_id}")
                await cur.execute(
                    """
                        INSERT INTO cayu_recall_receipts (
                            receipt_id, session_id, interaction_id, model_step_id,
                            created_at, receipt_json, document_bytes
                        ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                        """,
                    (
                        copied.receipt_id,
                        copied.session_id,
                        copied.interaction_id,
                        copied.model_step_id,
                        copied.created_at,
                        document.decode("utf-8"),
                        len(document),
                    ),
                )
            await conn.commit()
            return copied
        except BaseException:
            await conn.rollback()
            raise


async def load_recall_receipt(
    connect: PostgresConnection,
    session_id: str,
    receipt_id: str,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> RecallReceipt | None:
    session_id = require_memory_evidence_session_id(session_id)
    receipt_id = require_memory_evidence_id(receipt_id, "receipt_id")
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            """
                SELECT receipt_id, session_id, interaction_id, model_step_id,
                       created_at, receipt_json, document_bytes
                FROM cayu_recall_receipts
                WHERE session_id = %s AND receipt_id = %s
                """,
            (session_id, receipt_id),
        )
        row = await cur.fetchone()
        return None if row is None else recall_receipt_from_row(row)


async def list_recall_receipts(
    connect: PostgresConnection,
    query: RecallEvidenceQuery,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> RecallReceiptPage:
    copied_query = RecallEvidenceQuery.model_validate(query.model_dump(mode="python"))
    query_fingerprint = copied_query.fingerprint("receipt")
    after = (
        None
        if copied_query.cursor is None
        else decode_recall_evidence_cursor(
            copied_query.cursor,
            record_kind="receipt",
            query_fingerprint=query_fingerprint,
        )
    )
    clauses = ["session_id = %s"]
    parameters: list[object] = [copied_query.session_id]
    if copied_query.interaction_id is not None:
        clauses.append("interaction_id = %s")
        parameters.append(copied_query.interaction_id)
    if copied_query.model_step_id is not None:
        clauses.append("model_step_id = %s")
        parameters.append(copied_query.model_step_id)
    if after is not None:
        clauses.append('(created_at, receipt_id COLLATE "C") > (%s, %s)')
        parameters.extend(after)
    where = " AND ".join(clauses)
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            f"""
                SELECT receipt_id, session_id, interaction_id, model_step_id,
                       created_at, receipt_json, document_bytes
                FROM cayu_recall_receipts
                WHERE {where}
                ORDER BY created_at, receipt_id COLLATE "C"
                LIMIT %s
                """,
            (*parameters, copied_query.limit + 1),
        )
        rows = await cur.fetchall()
    retained: list[RecallReceipt] = []
    retained_bytes = 2
    for row in rows:
        if len(retained) >= copied_query.limit:
            break
        receipt = recall_receipt_from_row(row)
        document_bytes = len(memory_evidence_document_bytes(receipt, "recall receipt page item"))
        separator_bytes = 1 if retained else 0
        if retained_bytes + separator_bytes + document_bytes > copied_query.max_bytes:
            break
        retained.append(receipt)
        retained_bytes += separator_bytes + document_bytes
    truncated = len(retained) < len(rows)
    return RecallReceiptPage(
        items=tuple(retained),
        next_cursor=(
            encode_recall_evidence_cursor(
                record_kind="receipt",
                query_fingerprint=query_fingerprint,
                created_at=retained[-1].created_at,
                record_id=retained[-1].receipt_id,
            )
            if truncated and retained
            else None
        ),
        truncated=truncated,
    )


async def create_context_exposure(
    connect: PostgresConnection,
    exposure: ContextExposure,
    item_exposures: tuple[RecallItemExposure, ...] = (),
    *,
    ensure_ready: Callable[[], Awaitable[None]],
    lock_closure_lineage: Callable[[Any], Awaitable[None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
) -> ContextExposure:
    copied = copy_context_exposure(exposure)
    copied_items = tuple(copy_recall_item_exposure(item) for item in item_exposures)
    validate_new_context_exposure(copied, copied_items)
    document = memory_evidence_document_bytes(copied, "context exposure")
    item_documents = tuple(
        memory_evidence_document_bytes(item, "recall item exposure") for item in copied_items
    )
    lock_ids = sorted(
        {
            f"context-exposure:{copied.exposure_id}",
            f"model-attempt:{copied.session_id}:{copied.model_attempt_id}",
            f"provider-attempt:{copied.session_id}:{copied.provider_attempt_id}",
        }
    )
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await lock_closure_lineage(cur)
                for lock_id in lock_ids:
                    await _lock_evidence_id(cur, lock_id)
                await cur.execute(
                    """
                        SELECT exposure_id, session_id, interaction_id, model_step_id,
                               model_attempt_id, provider_attempt_id, state, state_revision,
                               created_at, updated_at, exposure_json, document_bytes
                        FROM cayu_context_exposures
                        WHERE exposure_id = %s
                        FOR UPDATE
                        """,
                    (copied.exposure_id,),
                )
                row = await cur.fetchone()
                if row is not None:
                    current = context_exposure_from_row(row)
                    await cur.execute(
                        """
                            SELECT exposure_id, ordinal, receipt_id, receipt_item_ordinal,
                                   item_json, document_bytes
                            FROM cayu_recall_item_exposures
                            WHERE exposure_id = %s
                            ORDER BY ordinal
                            LIMIT %s
                            """,
                        (copied.exposure_id, MAX_RECALL_RECEIPT_ITEMS + 1),
                    )
                    current_item_rows = await cur.fetchall()
                    if len(current_item_rows) > MAX_RECALL_RECEIPT_ITEMS:
                        raise ValueError("Stored recall item exposures exceed their count bound.")
                    current_items = recall_item_exposures_from_rows(current_item_rows)
                    if (
                        not context_exposure_creation_matches(current, copied)
                        or tuple(
                            memory_evidence_document_bytes(item, "stored recall item exposure")
                            for item in current_items
                        )
                        != item_documents
                    ):
                        raise RecallEvidenceConflict(
                            "Context exposure",
                            copied.exposure_id,
                        )
                    await conn.commit()
                    return current
                for owner in await closure_owners(cur, (copied.session_id,)):
                    _check_closure_lineage_owner(owner, (copied.session_id,))
                await cur.execute(
                    "SELECT 1 FROM cayu_sessions WHERE id = %s FOR KEY SHARE",
                    (copied.session_id,),
                )
                if await cur.fetchone() is None:
                    raise KeyError(f"Session not found: {copied.session_id}")
                await cur.execute(
                    """
                        SELECT exposure_id
                        FROM cayu_context_exposures
                        WHERE session_id = %s AND model_attempt_id = %s
                        """,
                    (copied.session_id, copied.model_attempt_id),
                )
                model_attempt_collision = await cur.fetchone()
                if model_attempt_collision is not None:
                    raise RecallEvidenceConflict(
                        "Model-attempt exposure",
                        str(model_attempt_collision[0]),
                    )
                await cur.execute(
                    """
                        SELECT exposure_id
                        FROM cayu_context_exposures
                        WHERE session_id = %s AND provider_attempt_id = %s
                        """,
                    (copied.session_id, copied.provider_attempt_id),
                )
                provider_attempt_collision = await cur.fetchone()
                if provider_attempt_collision is not None:
                    raise RecallEvidenceConflict(
                        "Provider-attempt exposure",
                        str(provider_attempt_collision[0]),
                    )
                receipts: dict[str, RecallReceipt] = {}
                for receipt_id in copied.receipt_ids:
                    await cur.execute(
                        """
                            SELECT receipt_id, session_id, interaction_id, model_step_id,
                                   created_at, receipt_json, document_bytes
                            FROM cayu_recall_receipts
                            WHERE receipt_id = %s
                            FOR SHARE
                            """,
                        (receipt_id,),
                    )
                    receipt_row = await cur.fetchone()
                    if receipt_row is None:
                        raise KeyError(f"Recall receipt not found: {receipt_id}")
                    receipt = recall_receipt_from_row(receipt_row)
                    validate_context_exposure_receipt_scope(copied, receipt)
                    receipts[receipt_id] = receipt
                for receipt_id in copied.carried_receipt_ids:
                    await cur.execute(
                        """
                            SELECT receipt_id, session_id, interaction_id, model_step_id,
                                   created_at, receipt_json, document_bytes
                            FROM cayu_recall_receipts
                            WHERE receipt_id = %s
                            FOR SHARE
                            """,
                        (receipt_id,),
                    )
                    receipt_row = await cur.fetchone()
                    if receipt_row is None:
                        raise KeyError(f"Recall receipt not found: {receipt_id}")
                    validate_context_exposure_carried_receipt_scope(
                        copied,
                        recall_receipt_from_row(receipt_row),
                    )
                for item in copied_items:
                    if not recall_item_exposure_matches_receipt_item(
                        item,
                        receipts[item.receipt_id],
                    ):
                        raise ValueError(
                            "Recall item exposure differs from its immutable receipt item."
                        )
                await cur.execute(
                    """
                        INSERT INTO cayu_context_exposures (
                            exposure_id, session_id, interaction_id, model_step_id,
                            model_attempt_id, provider_attempt_id, state, state_revision,
                            created_at, updated_at, exposure_json, document_bytes
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s::jsonb, %s
                        )
                        """,
                    (
                        copied.exposure_id,
                        copied.session_id,
                        copied.interaction_id,
                        copied.model_step_id,
                        copied.model_attempt_id,
                        copied.provider_attempt_id,
                        str(copied.state),
                        copied.state_revision,
                        copied.created_at,
                        copied.updated_at,
                        document.decode("utf-8"),
                        len(document),
                    ),
                )
                await cur.executemany(
                    """
                        INSERT INTO cayu_recall_item_exposures (
                            exposure_id, ordinal, receipt_id, receipt_item_ordinal,
                            item_json, document_bytes
                        ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                        """,
                    (
                        (
                            item.exposure_id,
                            item.ordinal,
                            item.receipt_id,
                            item.receipt_item_ordinal,
                            item_document.decode("utf-8"),
                            len(item_document),
                        )
                        for item, item_document in zip(
                            copied_items,
                            item_documents,
                            strict=True,
                        )
                    ),
                )
            await conn.commit()
            return copied
        except BaseException:
            await conn.rollback()
            raise


async def load_context_exposure(
    connect: PostgresConnection,
    session_id: str,
    exposure_id: str,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> ContextExposure | None:
    session_id = require_memory_evidence_session_id(session_id)
    exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            """
                SELECT exposure_id, session_id, interaction_id, model_step_id,
                       model_attempt_id, provider_attempt_id, state, state_revision,
                       created_at, updated_at, exposure_json, document_bytes
                FROM cayu_context_exposures
                WHERE session_id = %s AND exposure_id = %s
                """,
            (session_id, exposure_id),
        )
        row = await cur.fetchone()
        return None if row is None else context_exposure_from_row(row)


async def load_recall_item_exposures(
    connect: PostgresConnection,
    session_id: str,
    exposure_id: str,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> tuple[RecallItemExposure, ...]:
    session_id = require_memory_evidence_session_id(session_id)
    exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            """
                SELECT item.exposure_id, item.ordinal, item.receipt_id,
                       item.receipt_item_ordinal, item.item_json, item.document_bytes
                FROM cayu_recall_item_exposures AS item
                JOIN cayu_context_exposures AS exposure
                  ON exposure.exposure_id = item.exposure_id
                WHERE exposure.session_id = %s AND exposure.exposure_id = %s
                ORDER BY item.ordinal
                LIMIT %s
                """,
            (session_id, exposure_id, MAX_RECALL_RECEIPT_ITEMS + 1),
        )
        rows = await cur.fetchall()
        if len(rows) > MAX_RECALL_RECEIPT_ITEMS:
            raise ValueError("Stored recall item exposures exceed their count bound.")
        return recall_item_exposures_from_rows(rows)


async def list_context_exposures(
    connect: PostgresConnection,
    query: RecallEvidenceQuery,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
) -> ContextExposurePage:
    copied_query = RecallEvidenceQuery.model_validate(query.model_dump(mode="python"))
    query_fingerprint = copied_query.fingerprint("exposure")
    after = (
        None
        if copied_query.cursor is None
        else decode_recall_evidence_cursor(
            copied_query.cursor,
            record_kind="exposure",
            query_fingerprint=query_fingerprint,
        )
    )
    clauses = ["session_id = %s"]
    parameters: list[object] = [copied_query.session_id]
    if copied_query.interaction_id is not None:
        clauses.append("interaction_id = %s")
        parameters.append(copied_query.interaction_id)
    if copied_query.model_step_id is not None:
        clauses.append("model_step_id = %s")
        parameters.append(copied_query.model_step_id)
    if after is not None:
        clauses.append('(created_at, exposure_id COLLATE "C") > (%s, %s)')
        parameters.extend(after)
    where = " AND ".join(clauses)
    await ensure_ready()
    async with connect() as conn, conn.cursor() as cur:
        await cur.execute(
            f"""
                SELECT exposure_id, session_id, interaction_id, model_step_id,
                       model_attempt_id, provider_attempt_id, state, state_revision,
                       created_at, updated_at, exposure_json, document_bytes
                FROM cayu_context_exposures
                WHERE {where}
                ORDER BY created_at, exposure_id COLLATE "C"
                LIMIT %s
                """,
            (*parameters, copied_query.limit + 1),
        )
        rows = await cur.fetchall()
    retained: list[ContextExposure] = []
    retained_bytes = 2
    for row in rows:
        if len(retained) >= copied_query.limit:
            break
        exposure = context_exposure_from_row(row)
        document_bytes = len(memory_evidence_document_bytes(exposure, "context exposure page item"))
        separator_bytes = 1 if retained else 0
        if retained_bytes + separator_bytes + document_bytes > copied_query.max_bytes:
            break
        retained.append(exposure)
        retained_bytes += separator_bytes + document_bytes
    truncated = len(retained) < len(rows)
    return ContextExposurePage(
        items=tuple(retained),
        next_cursor=(
            encode_recall_evidence_cursor(
                record_kind="exposure",
                query_fingerprint=query_fingerprint,
                created_at=retained[-1].created_at,
                record_id=retained[-1].exposure_id,
            )
            if truncated and retained
            else None
        ),
        truncated=truncated,
    )


async def transition_context_exposure(
    connect: PostgresConnection,
    session_id: str,
    exposure_id: str,
    request: ContextExposureTransitionRequest,
    *,
    ensure_ready: Callable[[], Awaitable[None]],
    lock_closure_lineage: Callable[[Any], Awaitable[None]],
    closure_owners: Callable[[Any, Iterable[str]], Awaitable[tuple[dict[str, Any], ...]]],
) -> ContextExposure:
    session_id = require_memory_evidence_session_id(session_id)
    exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
    copied_request = ContextExposureTransitionRequest.model_validate(
        request.model_dump(mode="python")
    )
    await ensure_ready()
    async with connect() as conn:
        try:
            async with conn.cursor() as cur:
                await lock_closure_lineage(cur)
                await cur.execute(
                    """
                        SELECT exposure_id, session_id, interaction_id, model_step_id,
                               model_attempt_id, provider_attempt_id, state, state_revision,
                               created_at, updated_at, exposure_json, document_bytes
                        FROM cayu_context_exposures
                        WHERE session_id = %s AND exposure_id = %s
                        FOR UPDATE
                        """,
                    (session_id, exposure_id),
                )
                row = await cur.fetchone()
                if row is None:
                    raise KeyError(f"Context exposure not found: {exposure_id}")
                current = context_exposure_from_row(row)
                replay = next(
                    (
                        transition
                        for transition in current.transitions
                        if transition.transition_id == copied_request.transition_id
                    ),
                    None,
                )
                if replay is not None:
                    if not context_exposure_transition_replays(current, copied_request):
                        raise RecallEvidenceConflict(
                            "Context exposure transition",
                            copied_request.transition_id,
                        )
                    await conn.commit()
                    return current
                for owner in await closure_owners(cur, (session_id,)):
                    _check_closure_lineage_owner(owner, (session_id,))
                updated = append_context_exposure_transition(current, copied_request)
                updated_document = memory_evidence_document_bytes(
                    updated,
                    "context exposure",
                )
                await cur.execute(
                    """
                        UPDATE cayu_context_exposures
                        SET state = %s, state_revision = %s, updated_at = %s,
                            exposure_json = %s::jsonb, document_bytes = %s
                        WHERE session_id = %s AND exposure_id = %s
                          AND state = %s AND state_revision = %s
                        """,
                    (
                        str(updated.state),
                        updated.state_revision,
                        updated.updated_at,
                        updated_document.decode("utf-8"),
                        len(updated_document),
                        session_id,
                        exposure_id,
                        str(copied_request.expected_state),
                        copied_request.expected_revision,
                    ),
                )
                if cur.rowcount != 1:
                    raise ContextExposureTransitionConflict(
                        exposure_id,
                        expected_state=copied_request.expected_state,
                        expected_revision=copied_request.expected_revision,
                        actual_state=current.state,
                        actual_revision=current.state_revision,
                    )
            await conn.commit()
            return updated
        except BaseException:
            await conn.rollback()
            raise


async def _lock_evidence_id(cur: Any, lock_id: str) -> None:
    await cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (lock_id,),
    )


def _json_document(value: Any, field_name: str) -> dict[str, Any]:
    if type(value) is str:
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Postgres {field_name} contains invalid JSON.") from exc
    if type(value) is not dict:
        raise RuntimeError(f"Postgres {field_name} must be a JSON object.")
    return value


def recall_receipt_from_row(row: Sequence[Any]) -> RecallReceipt:
    try:
        receipt = RecallReceipt.model_validate(_json_document(row[5], "recall receipt"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Postgres recall receipt contains invalid durable material.") from exc
    document = memory_evidence_document_bytes(receipt, "stored recall receipt")
    if (
        receipt.receipt_id != row[0]
        or receipt.session_id != row[1]
        or receipt.interaction_id != row[2]
        or receipt.model_step_id != row[3]
        or receipt.created_at != pg_support.to_utc(row[4])
        or len(document) != row[6]
    ):
        raise RuntimeError("Postgres recall receipt index columns conflict with its document.")
    return receipt


def context_exposure_from_row(row: Sequence[Any]) -> ContextExposure:
    try:
        exposure = ContextExposure.model_validate(_json_document(row[10], "context exposure"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Postgres context exposure contains invalid durable material.") from exc
    document = memory_evidence_document_bytes(exposure, "stored context exposure")
    if (
        exposure.exposure_id != row[0]
        or exposure.session_id != row[1]
        or exposure.interaction_id != row[2]
        or exposure.model_step_id != row[3]
        or exposure.model_attempt_id != row[4]
        or exposure.provider_attempt_id != row[5]
        or str(exposure.state) != row[6]
        or exposure.state_revision != row[7]
        or exposure.created_at != pg_support.to_utc(row[8])
        or exposure.updated_at != pg_support.to_utc(row[9])
        or len(document) != row[11]
    ):
        raise RuntimeError("Postgres context exposure index columns conflict with its document.")
    return exposure


def recall_item_exposures_from_rows(
    rows: Sequence[Sequence[Any]],
) -> tuple[RecallItemExposure, ...]:
    items: list[RecallItemExposure] = []
    for row in rows:
        try:
            item = RecallItemExposure.model_validate(_json_document(row[4], "recall item exposure"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Postgres recall item exposure contains invalid durable material."
            ) from exc
        document = memory_evidence_document_bytes(item, "stored recall item exposure")
        if (
            item.exposure_id != row[0]
            or item.ordinal != row[1]
            or item.receipt_id != row[2]
            or item.receipt_item_ordinal != row[3]
            or len(document) != row[5]
        ):
            raise RuntimeError(
                "Postgres recall item exposure index columns conflict with its document."
            )
        items.append(item)
    if tuple(item.ordinal for item in items) != tuple(range(len(items))):
        raise RuntimeError("Postgres recall item exposure ordinals are incomplete.")
    return tuple(items)
