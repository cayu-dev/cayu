"""In-memory recall receipts and context exposures over explicit locked state."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from itertools import islice
from typing import NamedTuple

from cayu.memory.evidence import (
    MAX_RECALL_RECEIPT_ITEMS,
    ContextExposure,
    ContextExposurePage,
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
    session_closure_progress: closure_records.ClosureProgressIndex
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
        for owner in state.session_closure_progress.owners((copied.session_id,)):
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


class MemoryContextExposureState(NamedTuple):
    """Borrowed state resolved after acquiring the store lock for each operation.

    Reads and writes share that lock, including closure checks and index updates.
    The caller retains state ownership; no view may outlive its lock scope.
    """

    context_exposure_by_model_attempt: dict[tuple[str, str], str]
    context_exposure_by_provider_attempt: dict[tuple[str, str], str]
    context_exposure_ids_by_session: dict[str, set[str]]
    context_exposure_page_keys_by_interaction: dict[tuple[str, str], list[tuple[datetime, str]]]
    context_exposure_page_keys_by_interaction_step: dict[
        tuple[str, str, str], list[tuple[datetime, str]]
    ]
    context_exposure_page_keys_by_session: dict[str, list[tuple[datetime, str]]]
    context_exposure_page_keys_by_step: dict[tuple[str, str], list[tuple[datetime, str]]]
    context_exposures: dict[str, ContextExposure]
    recall_item_exposures: dict[str, tuple[RecallItemExposure, ...]]
    recall_receipts: dict[str, RecallReceipt]
    session_closure_progress: closure_records.ClosureProgressIndex
    sessions: dict[str, Session]


LockedContextExposureState = Callable[[], AbstractAsyncContextManager[MemoryContextExposureState]]


async def create_context_exposure(
    locked: LockedContextExposureState,
    exposure: ContextExposure,
    item_exposures: tuple[RecallItemExposure, ...] = (),
) -> ContextExposure:
    copied = copy_context_exposure(exposure)
    copied_items = tuple(copy_recall_item_exposure(item) for item in item_exposures)
    validate_new_context_exposure(copied, copied_items)

    async with locked() as state:
        current = state.context_exposures.get(copied.exposure_id)
        if current is not None:
            current_items = state.recall_item_exposures[copied.exposure_id]
            if not context_exposure_creation_matches(current, copied) or tuple(
                memory_evidence_document_bytes(item, "stored recall item exposure")
                for item in current_items
            ) != tuple(
                memory_evidence_document_bytes(item, "recall item exposure")
                for item in copied_items
            ):
                raise RecallEvidenceConflict("Context exposure", copied.exposure_id)
            return copy_context_exposure(current)
        if copied.session_id not in state.sessions:
            raise KeyError(f"Session not found: {copied.session_id}")
        for owner in state.session_closure_progress.owners((copied.session_id,)):
            closure_records._check_closure_lineage_owner(owner, (copied.session_id,))

        attempt_key = copied.session_id, copied.model_attempt_id
        conflicting_exposure_id = state.context_exposure_by_model_attempt.get(attempt_key)
        if conflicting_exposure_id is not None:
            raise RecallEvidenceConflict("Model-attempt exposure", conflicting_exposure_id)
        provider_attempt_key = copied.session_id, copied.provider_attempt_id
        conflicting_exposure_id = state.context_exposure_by_provider_attempt.get(
            provider_attempt_key
        )
        if conflicting_exposure_id is not None:
            raise RecallEvidenceConflict(
                "Provider-attempt exposure",
                conflicting_exposure_id,
            )

        receipts: dict[str, RecallReceipt] = {}
        for receipt_id in copied.receipt_ids:
            receipt = state.recall_receipts.get(receipt_id)
            if receipt is None:
                raise KeyError(f"Recall receipt not found: {receipt_id}")
            validate_context_exposure_receipt_scope(copied, receipt)
            receipts[receipt_id] = receipt
        for receipt_id in copied.carried_receipt_ids:
            receipt = state.recall_receipts.get(receipt_id)
            if receipt is None:
                raise KeyError(f"Recall receipt not found: {receipt_id}")
            validate_context_exposure_carried_receipt_scope(copied, receipt)
        for item in copied_items:
            if not recall_item_exposure_matches_receipt_item(
                item,
                receipts[item.receipt_id],
            ):
                raise ValueError("Recall item exposure differs from its immutable receipt item.")

        state.context_exposures[copied.exposure_id] = copied
        state.context_exposure_ids_by_session.setdefault(copied.session_id, set()).add(
            copied.exposure_id
        )
        state.recall_item_exposures[copied.exposure_id] = copied_items
        state.context_exposure_by_model_attempt[attempt_key] = copied.exposure_id
        state.context_exposure_by_provider_attempt[provider_attempt_key] = copied.exposure_id
        page_key = copied.created_at, copied.exposure_id
        for page_keys in (
            state.context_exposure_page_keys_by_session.setdefault(copied.session_id, []),
            state.context_exposure_page_keys_by_interaction.setdefault(
                (copied.session_id, copied.interaction_id), []
            ),
            state.context_exposure_page_keys_by_step.setdefault(
                (copied.session_id, copied.model_step_id), []
            ),
            state.context_exposure_page_keys_by_interaction_step.setdefault(
                (copied.session_id, copied.interaction_id, copied.model_step_id), []
            ),
        ):
            page_keys.insert(bisect_right(page_keys, page_key), page_key)
        return copy_context_exposure(copied)


async def load_context_exposure(
    locked: LockedContextExposureState,
    session_id: str,
    exposure_id: str,
) -> ContextExposure | None:
    session_id = require_memory_evidence_session_id(session_id)
    exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
    async with locked() as state:
        exposure = state.context_exposures.get(exposure_id)
        if exposure is None or exposure.session_id != session_id:
            return None
        return copy_context_exposure(exposure)


async def load_recall_item_exposures(
    locked: LockedContextExposureState,
    session_id: str,
    exposure_id: str,
) -> tuple[RecallItemExposure, ...]:
    session_id = require_memory_evidence_session_id(session_id)
    exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
    async with locked() as state:
        exposure = state.context_exposures.get(exposure_id)
        if exposure is None or exposure.session_id != session_id:
            return ()
        if len(state.recall_item_exposures[exposure_id]) > MAX_RECALL_RECEIPT_ITEMS:
            raise ValueError("Stored recall item exposures exceed their count bound.")
        return tuple(
            copy_recall_item_exposure(item) for item in state.recall_item_exposures[exposure_id]
        )


async def list_context_exposures(
    locked: LockedContextExposureState,
    query: RecallEvidenceQuery,
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
    async with locked() as state:
        if copied_query.interaction_id is not None and copied_query.model_step_id is not None:
            page_keys = state.context_exposure_page_keys_by_interaction_step.get(
                (
                    copied_query.session_id,
                    copied_query.interaction_id,
                    copied_query.model_step_id,
                ),
                (),
            )
        elif copied_query.interaction_id is not None:
            page_keys = state.context_exposure_page_keys_by_interaction.get(
                (copied_query.session_id, copied_query.interaction_id),
                (),
            )
        elif copied_query.model_step_id is not None:
            page_keys = state.context_exposure_page_keys_by_step.get(
                (copied_query.session_id, copied_query.model_step_id),
                (),
            )
        else:
            page_keys = state.context_exposure_page_keys_by_session.get(
                copied_query.session_id,
                (),
            )
        start = 0 if after is None else bisect_right(page_keys, after)
        candidates = tuple(
            state.context_exposures[exposure_id]
            for _, exposure_id in islice(
                page_keys,
                start,
                start + copied_query.limit + 1,
            )
        )
        retained: list[ContextExposure] = []
        retained_bytes = 2
        for exposure in candidates:
            document_bytes = len(
                memory_evidence_document_bytes(exposure, "context exposure page item")
            )
            separator_bytes = 1 if retained else 0
            if len(retained) >= copied_query.limit or (
                retained_bytes + separator_bytes + document_bytes > copied_query.max_bytes
            ):
                break
            retained.append(exposure)
            retained_bytes += separator_bytes + document_bytes
        truncated = len(retained) < len(candidates)
        next_cursor = (
            encode_recall_evidence_cursor(
                record_kind="exposure",
                query_fingerprint=query_fingerprint,
                created_at=retained[-1].created_at,
                record_id=retained[-1].exposure_id,
            )
            if truncated and retained
            else None
        )
        return ContextExposurePage(
            items=tuple(retained),
            next_cursor=next_cursor,
            truncated=truncated,
        )


async def transition_context_exposure(
    locked: LockedContextExposureState,
    session_id: str,
    exposure_id: str,
    request: ContextExposureTransitionRequest,
) -> ContextExposure:
    session_id = require_memory_evidence_session_id(session_id)
    exposure_id = require_memory_evidence_id(exposure_id, "exposure_id")
    copied_request = ContextExposureTransitionRequest.model_validate(
        request.model_dump(mode="python")
    )
    async with locked() as state:
        exposure = state.context_exposures.get(exposure_id)
        if exposure is None or exposure.session_id != session_id:
            raise KeyError(f"Context exposure not found: {exposure_id}")
        replay = next(
            (
                transition
                for transition in exposure.transitions
                if transition.transition_id == copied_request.transition_id
            ),
            None,
        )
        if replay is not None:
            if not context_exposure_transition_replays(exposure, copied_request):
                raise RecallEvidenceConflict(
                    "Context exposure transition",
                    copied_request.transition_id,
                )
            return copy_context_exposure(exposure)
        for owner in state.session_closure_progress.owners((session_id,)):
            closure_records._check_closure_lineage_owner(owner, (session_id,))
        updated = append_context_exposure_transition(exposure, copied_request)
        state.context_exposures[exposure_id] = updated
        return copy_context_exposure(updated)
