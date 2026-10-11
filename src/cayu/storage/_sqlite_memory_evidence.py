"""Native SQLite recall and context-exposure persistence.

Complete evidence operations retain closure admission, replay and writes
in their native transactions. Composition supplies explicit capabilities.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence

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
from cayu.sessions._closure_records import _check_closure_lineage_owner
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage._sqlite_connection import SQLiteOperationRunner
from cayu.storage._sqlite_transcript import ClosureOwners


async def create_recall_receipt(
    run_write: SQLiteOperationRunner, receipt: RecallReceipt, *, closure_owners: ClosureOwners
) -> RecallReceipt:
    copied = copy_recall_receipt(receipt)
    document = memory_evidence_document_bytes(copied, "recall receipt")

    def statement(connection: sqlite3.Connection) -> RecallReceipt:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM cayu_recall_receipts WHERE receipt_id = ?",
                (copied.receipt_id,),
            ).fetchone()
            if row is not None:
                current = recall_receipt_from_row(row)
                if memory_evidence_document_bytes(current, "stored recall receipt") != document:
                    raise RecallEvidenceConflict("Recall receipt", copied.receipt_id)
                connection.commit()
                return current
            for owner in closure_owners((copied.session_id,), connection=connection):
                _check_closure_lineage_owner(owner, (copied.session_id,))
            if not connection.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = ?",
                (copied.session_id,),
            ).fetchone():
                raise KeyError(f"Session not found: {copied.session_id}")
            connection.execute(
                """
                    INSERT INTO cayu_recall_receipts (
                        receipt_id, session_id, interaction_id, model_step_id,
                        created_at, receipt_json, document_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    copied.receipt_id,
                    copied.session_id,
                    copied.interaction_id,
                    copied.model_step_id,
                    sqlite_records.format_datetime(copied.created_at),
                    document.decode("utf-8"),
                    len(document),
                ),
            )
            connection.commit()
            return copied
        except BaseException:
            connection.rollback()
            raise

    return copy_recall_receipt(await run_write(statement))


async def load_recall_receipt(
    run_read: SQLiteOperationRunner, session_id: str, receipt_id: str
) -> RecallReceipt | None:
    session_id = require_memory_evidence_session_id(session_id)
    receipt_id = require_memory_evidence_id(receipt_id, "receipt_id")

    def query(connection: sqlite3.Connection) -> RecallReceipt | None:
        row = connection.execute(
            """
                SELECT * FROM cayu_recall_receipts
                WHERE session_id = ? AND receipt_id = ?
                """,
            (session_id, receipt_id),
        ).fetchone()
        return None if row is None else recall_receipt_from_row(row)

    loaded = await run_read(query)
    return None if loaded is None else copy_recall_receipt(loaded)


async def list_recall_receipts(
    run_read: SQLiteOperationRunner, query: RecallEvidenceQuery
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

    def read(connection: sqlite3.Connection) -> RecallReceiptPage:
        clauses = ["session_id = ?"]
        parameters: list[object] = [copied_query.session_id]
        if copied_query.interaction_id is not None:
            clauses.append("interaction_id = ?")
            parameters.append(copied_query.interaction_id)
        if copied_query.model_step_id is not None:
            clauses.append("model_step_id = ?")
            parameters.append(copied_query.model_step_id)
        if after is not None:
            clauses.append("(created_at, receipt_id) > (?, ?)")
            created_at = sqlite_records.format_datetime(after[0])
            parameters.extend((created_at, after[1]))
        where = " AND ".join(clauses)
        rows = connection.execute(
            f"""
                SELECT * FROM cayu_recall_receipts
                WHERE {where}
                ORDER BY created_at, receipt_id COLLATE BINARY
                LIMIT ?
                """,
            (*parameters, copied_query.limit + 1),
        ).fetchall()
        retained: list[RecallReceipt] = []
        retained_bytes = 2
        for row in rows:
            if len(retained) >= copied_query.limit:
                break
            receipt = recall_receipt_from_row(row)
            document_bytes = len(
                memory_evidence_document_bytes(receipt, "recall receipt page item")
            )
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

    return await run_read(read)


async def create_context_exposure(
    run_write: SQLiteOperationRunner,
    exposure: ContextExposure,
    item_exposures: tuple[RecallItemExposure, ...] = (),
    *,
    closure_owners: ClosureOwners,
) -> ContextExposure:
    copied = copy_context_exposure(exposure)
    copied_items = tuple(copy_recall_item_exposure(item) for item in item_exposures)
    validate_new_context_exposure(copied, copied_items)
    document = memory_evidence_document_bytes(copied, "context exposure")
    item_documents = tuple(
        memory_evidence_document_bytes(item, "recall item exposure") for item in copied_items
    )

    def statement(connection: sqlite3.Connection) -> ContextExposure:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM cayu_context_exposures WHERE exposure_id = ?",
                (copied.exposure_id,),
            ).fetchone()
            if row is not None:
                current = context_exposure_from_row(row)
                current_item_rows = connection.execute(
                    """
                        SELECT item.*
                        FROM cayu_recall_item_exposures AS item
                        WHERE item.exposure_id = ?
                        ORDER BY item.ordinal
                        LIMIT ?
                        """,
                    (copied.exposure_id, MAX_RECALL_RECEIPT_ITEMS + 1),
                ).fetchall()
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
                    raise RecallEvidenceConflict("Context exposure", copied.exposure_id)
                connection.commit()
                return current
            for owner in closure_owners((copied.session_id,), connection=connection):
                _check_closure_lineage_owner(owner, (copied.session_id,))
            if not connection.execute(
                "SELECT 1 FROM cayu_sessions WHERE id = ?",
                (copied.session_id,),
            ).fetchone():
                raise KeyError(f"Session not found: {copied.session_id}")

            model_attempt_collision = connection.execute(
                """
                    SELECT exposure_id
                    FROM cayu_context_exposures
                    WHERE session_id = ? AND model_attempt_id = ?
                    """,
                (copied.session_id, copied.model_attempt_id),
            ).fetchone()
            if model_attempt_collision is not None:
                raise RecallEvidenceConflict(
                    "Model-attempt exposure",
                    model_attempt_collision[0],
                )
            provider_attempt_collision = connection.execute(
                """
                    SELECT exposure_id
                    FROM cayu_context_exposures
                    WHERE session_id = ? AND provider_attempt_id = ?
                    """,
                (copied.session_id, copied.provider_attempt_id),
            ).fetchone()
            if provider_attempt_collision is not None:
                raise RecallEvidenceConflict(
                    "Provider-attempt exposure",
                    provider_attempt_collision[0],
                )

            receipts: dict[str, RecallReceipt] = {}
            for receipt_id in copied.receipt_ids:
                receipt_row = connection.execute(
                    "SELECT * FROM cayu_recall_receipts WHERE receipt_id = ?",
                    (receipt_id,),
                ).fetchone()
                if receipt_row is None:
                    raise KeyError(f"Recall receipt not found: {receipt_id}")
                receipt = recall_receipt_from_row(receipt_row)
                validate_context_exposure_receipt_scope(copied, receipt)
                receipts[receipt_id] = receipt
            for receipt_id in copied.carried_receipt_ids:
                receipt_row = connection.execute(
                    "SELECT * FROM cayu_recall_receipts WHERE receipt_id = ?",
                    (receipt_id,),
                ).fetchone()
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
            connection.execute(
                """
                    INSERT INTO cayu_context_exposures (
                        exposure_id, session_id, interaction_id, model_step_id,
                        model_attempt_id, provider_attempt_id, state, state_revision,
                        created_at, updated_at, exposure_json, document_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    sqlite_records.format_datetime(copied.created_at),
                    sqlite_records.format_datetime(copied.updated_at),
                    document.decode("utf-8"),
                    len(document),
                ),
            )
            connection.executemany(
                """
                    INSERT INTO cayu_recall_item_exposures (
                        exposure_id, ordinal, receipt_id, receipt_item_ordinal,
                        item_json, document_bytes
                    ) VALUES (?, ?, ?, ?, ?, ?)
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
            connection.commit()
            return copied
        except BaseException:
            connection.rollback()
            raise

    return copy_context_exposure(await run_write(statement))


async def load_context_exposure(
    run_read: SQLiteOperationRunner, session_id: str, exposure_id: str
) -> ContextExposure | None:
    session_id = require_memory_evidence_session_id(session_id)
    exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")

    def query(connection: sqlite3.Connection) -> ContextExposure | None:
        row = connection.execute(
            """
                SELECT * FROM cayu_context_exposures
                WHERE session_id = ? AND exposure_id = ?
                """,
            (session_id, exposure_id),
        ).fetchone()
        return None if row is None else context_exposure_from_row(row)

    loaded = await run_read(query)
    return None if loaded is None else copy_context_exposure(loaded)


async def load_recall_item_exposures(
    run_read: SQLiteOperationRunner, session_id: str, exposure_id: str
) -> tuple[RecallItemExposure, ...]:
    session_id = require_memory_evidence_session_id(session_id)
    exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")

    def query(connection: sqlite3.Connection) -> tuple[RecallItemExposure, ...]:
        rows = connection.execute(
            """
                SELECT item.*
                FROM cayu_recall_item_exposures AS item
                JOIN cayu_context_exposures AS exposure
                  ON exposure.exposure_id = item.exposure_id
                WHERE exposure.session_id = ? AND exposure.exposure_id = ?
                ORDER BY item.ordinal
                LIMIT ?
                """,
            (session_id, exposure_id, MAX_RECALL_RECEIPT_ITEMS + 1),
        ).fetchall()
        if len(rows) > MAX_RECALL_RECEIPT_ITEMS:
            raise ValueError("Stored recall item exposures exceed their count bound.")
        return recall_item_exposures_from_rows(rows)

    return tuple(copy_recall_item_exposure(item) for item in await run_read(query))


async def list_context_exposures(
    run_read: SQLiteOperationRunner, query: RecallEvidenceQuery
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

    def read(connection: sqlite3.Connection) -> ContextExposurePage:
        clauses = ["session_id = ?"]
        parameters: list[object] = [copied_query.session_id]
        if copied_query.interaction_id is not None:
            clauses.append("interaction_id = ?")
            parameters.append(copied_query.interaction_id)
        if copied_query.model_step_id is not None:
            clauses.append("model_step_id = ?")
            parameters.append(copied_query.model_step_id)
        if after is not None:
            clauses.append("(created_at, exposure_id) > (?, ?)")
            created_at = sqlite_records.format_datetime(after[0])
            parameters.extend((created_at, after[1]))
        where = " AND ".join(clauses)
        rows = connection.execute(
            f"""
                SELECT * FROM cayu_context_exposures
                WHERE {where}
                ORDER BY created_at, exposure_id COLLATE BINARY
                LIMIT ?
                """,
            (*parameters, copied_query.limit + 1),
        ).fetchall()
        retained: list[ContextExposure] = []
        retained_bytes = 2
        for row in rows:
            if len(retained) >= copied_query.limit:
                break
            exposure = context_exposure_from_row(row)
            document_bytes = len(
                memory_evidence_document_bytes(exposure, "context exposure page item")
            )
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

    return await run_read(read)


async def transition_context_exposure(
    run_write: SQLiteOperationRunner,
    session_id: str,
    exposure_id: str,
    request: ContextExposureTransitionRequest,
    *,
    closure_owners: ClosureOwners,
) -> ContextExposure:
    session_id = require_memory_evidence_session_id(session_id)
    exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
    copied_request = ContextExposureTransitionRequest.model_validate(
        request.model_dump(mode="python")
    )

    def statement(connection: sqlite3.Connection) -> ContextExposure:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """
                    SELECT * FROM cayu_context_exposures
                    WHERE session_id = ? AND exposure_id = ?
                    """,
                (session_id, exposure_id),
            ).fetchone()
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
                connection.commit()
                return current
            for owner in closure_owners((session_id,), connection=connection):
                _check_closure_lineage_owner(owner, (session_id,))
            updated = append_context_exposure_transition(current, copied_request)
            updated_document = memory_evidence_document_bytes(
                updated,
                "context exposure",
            )
            cursor = connection.execute(
                """
                    UPDATE cayu_context_exposures
                    SET state = ?, state_revision = ?, updated_at = ?,
                        exposure_json = ?, document_bytes = ?
                    WHERE session_id = ? AND exposure_id = ?
                      AND state = ? AND state_revision = ?
                    """,
                (
                    str(updated.state),
                    updated.state_revision,
                    sqlite_records.format_datetime(updated.updated_at),
                    updated_document.decode("utf-8"),
                    len(updated_document),
                    session_id,
                    exposure_id,
                    str(copied_request.expected_state),
                    copied_request.expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                latest_row = connection.execute(
                    "SELECT * FROM cayu_context_exposures WHERE exposure_id = ?",
                    (exposure_id,),
                ).fetchone()
                if latest_row is None:
                    raise KeyError(f"Context exposure not found: {exposure_id}")
                latest = context_exposure_from_row(latest_row)
                raise ContextExposureTransitionConflict(
                    exposure_id,
                    expected_state=copied_request.expected_state,
                    expected_revision=copied_request.expected_revision,
                    actual_state=latest.state,
                    actual_revision=latest.state_revision,
                )
            connection.commit()
            return updated
        except BaseException:
            connection.rollback()
            raise

    return copy_context_exposure(await run_write(statement))


def recall_receipt_from_row(row: sqlite3.Row) -> RecallReceipt:
    try:
        receipt = RecallReceipt.model_validate(json.loads(row["receipt_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("SQLite recall receipt contains invalid durable material.") from exc
    document = memory_evidence_document_bytes(receipt, "stored recall receipt")
    if (
        receipt.receipt_id != row["receipt_id"]
        or receipt.session_id != row["session_id"]
        or receipt.interaction_id != row["interaction_id"]
        or receipt.model_step_id != row["model_step_id"]
        or receipt.created_at != sqlite_records.parse_datetime(row["created_at"])
        or len(document) != row["document_bytes"]
    ):
        raise RuntimeError("SQLite recall receipt index columns conflict with its document.")
    return receipt


def context_exposure_from_row(row: sqlite3.Row) -> ContextExposure:
    try:
        exposure = ContextExposure.model_validate(json.loads(row["exposure_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("SQLite context exposure contains invalid durable material.") from exc
    document = memory_evidence_document_bytes(exposure, "stored context exposure")
    if (
        exposure.exposure_id != row["exposure_id"]
        or exposure.session_id != row["session_id"]
        or exposure.interaction_id != row["interaction_id"]
        or exposure.model_step_id != row["model_step_id"]
        or exposure.model_attempt_id != row["model_attempt_id"]
        or exposure.provider_attempt_id != row["provider_attempt_id"]
        or str(exposure.state) != row["state"]
        or exposure.state_revision != row["state_revision"]
        or exposure.created_at != sqlite_records.parse_datetime(row["created_at"])
        or exposure.updated_at != sqlite_records.parse_datetime(row["updated_at"])
        or len(document) != row["document_bytes"]
    ):
        raise RuntimeError("SQLite context exposure index columns conflict with its document.")
    return exposure


def recall_item_exposures_from_rows(
    rows: Sequence[sqlite3.Row],
) -> tuple[RecallItemExposure, ...]:
    items: list[RecallItemExposure] = []
    for row in rows:
        try:
            item = RecallItemExposure.model_validate(json.loads(row["item_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "SQLite recall item exposure contains invalid durable material."
            ) from exc
        document = memory_evidence_document_bytes(item, "stored recall item exposure")
        if (
            item.exposure_id != row["exposure_id"]
            or item.ordinal != row["ordinal"]
            or item.receipt_id != row["receipt_id"]
            or item.receipt_item_ordinal != row["receipt_item_ordinal"]
            or len(document) != row["document_bytes"]
        ):
            raise RuntimeError(
                "SQLite recall item exposure index columns conflict with its document."
            )
        items.append(item)
    if tuple(item.ordinal for item in items) != tuple(range(len(items))):
        raise RuntimeError("SQLite recall item exposure ordinals are incomplete.")
    return tuple(items)
