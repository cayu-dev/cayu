"""In-memory recall receipts and context exposures over explicit locked state."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from itertools import islice
from typing import Any, NamedTuple

from cayu.memory.evidence import (
    RecallEvidenceConflict,
    RecallEvidenceQuery,
    RecallReceipt,
    RecallReceiptPage,
    copy_recall_receipt,
    decode_recall_evidence_cursor,
    encode_recall_evidence_cursor,
    memory_evidence_document_bytes,
    require_memory_evidence_id,
    require_memory_evidence_session_id,
)
from cayu.sessions import _closure_records as closure_records
from cayu.sessions.records import Session


class MemoryRecallState(NamedTuple):
    """Borrowed state resolved after acquiring the store lock for each operation.

    Reads and writes share that lock, including closure checks and index updates.
    The caller retains state ownership; no view may outlive its lock scope.
    """

    recall_receipt_ids_by_session: dict[str, set[str]]
    recall_receipt_page_keys_by_interaction: dict[tuple[str, str], list[tuple[datetime, str]]]
    recall_receipt_page_keys_by_interaction_step: dict[
        tuple[str, str, str], list[tuple[datetime, str]]
    ]
    recall_receipt_page_keys_by_session: dict[str, list[tuple[datetime, str]]]
    recall_receipt_page_keys_by_step: dict[tuple[str, str], list[tuple[datetime, str]]]
    recall_receipts: dict[str, RecallReceipt]
    session_closure_progress: dict[tuple[str, str], dict[str, Any]]
    sessions: dict[str, Session]


LockedRecallState = Callable[[], AbstractAsyncContextManager[MemoryRecallState]]


async def create_recall_receipt(locked: LockedRecallState, receipt: RecallReceipt) -> RecallReceipt:
    copied = copy_recall_receipt(receipt)
    async with locked() as state:
        current = state.recall_receipts.get(copied.receipt_id)
        if current is not None:
            if memory_evidence_document_bytes(
                current, "stored recall receipt"
            ) != memory_evidence_document_bytes(copied, "recall receipt"):
                raise RecallEvidenceConflict("Recall receipt", copied.receipt_id)
            return copy_recall_receipt(current)
        if copied.session_id not in state.sessions:
            raise KeyError(f"Session not found: {copied.session_id}")
        for owner in state.session_closure_progress.values():
            closure_records._check_closure_lineage_owner(owner, (copied.session_id,))
        state.recall_receipts[copied.receipt_id] = copied
        state.recall_receipt_ids_by_session.setdefault(copied.session_id, set()).add(
            copied.receipt_id
        )
        page_key = copied.created_at, copied.receipt_id
        for page_keys in (
            state.recall_receipt_page_keys_by_session.setdefault(copied.session_id, []),
            state.recall_receipt_page_keys_by_interaction.setdefault(
                (copied.session_id, copied.interaction_id), []
            ),
            state.recall_receipt_page_keys_by_step.setdefault(
                (copied.session_id, copied.model_step_id), []
            ),
            state.recall_receipt_page_keys_by_interaction_step.setdefault(
                (copied.session_id, copied.interaction_id, copied.model_step_id), []
            ),
        ):
            page_keys.insert(bisect_right(page_keys, page_key), page_key)
        return copy_recall_receipt(copied)


async def load_recall_receipt(
    locked: LockedRecallState,
    session_id: str,
    receipt_id: str,
) -> RecallReceipt | None:
    session_id = require_memory_evidence_session_id(session_id)
    receipt_id = require_memory_evidence_id(receipt_id, "receipt_id")
    async with locked() as state:
        receipt = state.recall_receipts.get(receipt_id)
        if receipt is None or receipt.session_id != session_id:
            return None
        return copy_recall_receipt(receipt)


async def list_recall_receipts(
    locked: LockedRecallState,
    query: RecallEvidenceQuery,
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
    async with locked() as state:
        if copied_query.interaction_id is not None and copied_query.model_step_id is not None:
            page_keys = state.recall_receipt_page_keys_by_interaction_step.get(
                (
                    copied_query.session_id,
                    copied_query.interaction_id,
                    copied_query.model_step_id,
                ),
                (),
            )
        elif copied_query.interaction_id is not None:
            page_keys = state.recall_receipt_page_keys_by_interaction.get(
                (copied_query.session_id, copied_query.interaction_id),
                (),
            )
        elif copied_query.model_step_id is not None:
            page_keys = state.recall_receipt_page_keys_by_step.get(
                (copied_query.session_id, copied_query.model_step_id),
                (),
            )
        else:
            page_keys = state.recall_receipt_page_keys_by_session.get(
                copied_query.session_id,
                (),
            )
        start = 0 if after is None else bisect_right(page_keys, after)
        candidates = tuple(
            state.recall_receipts[receipt_id]
            for _, receipt_id in islice(
                page_keys,
                start,
                start + copied_query.limit + 1,
            )
        )
        retained: list[RecallReceipt] = []
        retained_bytes = 2
        for receipt in candidates:
            document_bytes = len(
                memory_evidence_document_bytes(receipt, "recall receipt page item")
            )
            separator_bytes = 1 if retained else 0
            if len(retained) >= copied_query.limit or (
                retained_bytes + separator_bytes + document_bytes > copied_query.max_bytes
            ):
                break
            retained.append(receipt)
            retained_bytes += separator_bytes + document_bytes
        truncated = len(retained) < len(candidates)
        next_cursor = (
            encode_recall_evidence_cursor(
                record_kind="receipt",
                query_fingerprint=query_fingerprint,
                created_at=retained[-1].created_at,
                record_id=retained[-1].receipt_id,
            )
            if truncated and retained
            else None
        )
        return RecallReceiptPage(
            items=tuple(retained),
            next_cursor=next_cursor,
            truncated=truncated,
        )
