"""Read-only bounded projection of private durable memory evidence."""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from cayu._validation import canonical_durable_json_bytes, compact_json_utf8_size
from cayu.memory_attribution import (
    MemoryAttribution,
    MemoryAttributionBounds,
    MemoryAttributionStatus,
    MemoryAttributionUnavailableReason,
    MemoryContextExposureAttribution,
    MemoryEvidenceAlias,
    MemoryExposureItemAttribution,
    MemoryExposureTransitionAttribution,
    MemoryRecallAttribution,
    MemoryRecallItemAttribution,
)
from cayu.memory_evidence import (
    MAX_MEMORY_EVIDENCE_PAGE_BYTES,
    MAX_MEMORY_EVIDENCE_PAGE_LIMIT,
    MAX_RECALL_ITEM_EXPOSURE_BYTES,
    MAX_RECALL_RECEIPT_ITEMS,
    MIN_MEMORY_EVIDENCE_PAGE_BYTES,
    ContextExposure,
    ContextExposurePage,
    RecallEvidenceQuery,
    RecallItemExposure,
    RecallReceipt,
    RecallReceiptPage,
    RecallSourceCoverageState,
    memory_evidence_document_bytes,
    recall_item_exposure_matches_receipt_item,
    validate_context_exposure_receipt_scope,
    validate_new_context_exposure,
)
from cayu.runtime._memory_evidence import MemoryEvidenceKey
from cayu.runtime.sessions import SessionStore

_ALIAS_CONTEXT = b"cayu.memory-attribution.alias.v1"
_AliasKind = Literal["receipt", "exposure", "item", "interaction"]
_MAX_ITEM_EXPOSURE_BUNDLE_BYTES = (
    2 + MAX_RECALL_RECEIPT_ITEMS * MAX_RECALL_ITEM_EXPOSURE_BYTES + MAX_RECALL_RECEIPT_ITEMS - 1
)


@dataclass(slots=True)
class MemoryAttributionCaptureBudget:
    """Mutable global accounting shared across one complete public capture."""

    bounds: MemoryAttributionBounds
    receipts: int = 0
    exposures: int = 0
    items: int = 0
    source_bytes: int = 0
    projection_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.bounds) is not MemoryAttributionBounds:
            raise TypeError("bounds must be a MemoryAttributionBounds.")

    @property
    def remaining_receipts(self) -> int:
        return max(self.bounds.max_receipts - self.receipts, 0)

    @property
    def remaining_exposures(self) -> int:
        return max(self.bounds.max_exposures - self.exposures, 0)

    @property
    def remaining_items(self) -> int:
        return max(self.bounds.max_items - self.items, 0)

    @property
    def remaining_source_bytes(self) -> int:
        return max(self.bounds.max_source_bytes - self.source_bytes, 0)

    def retain_projection(self, value: BaseModel) -> bool:
        if not isinstance(value, BaseModel):
            raise TypeError("projected memory evidence must be a Pydantic model.")
        size = compact_json_utf8_size(value.model_dump(mode="json"))
        if self.projection_bytes + size > self.bounds.max_projection_bytes:
            return False
        self.projection_bytes += size
        return True


@dataclass(frozen=True, slots=True)
class _SourceCapture:
    receipts: tuple[RecallReceipt, ...]
    exposures: tuple[ContextExposure, ...]
    receipts_complete: bool
    exposures_complete: bool
    receipt_more_at_least: int
    exposure_more_at_least: int


async def project_memory_attribution(
    store: SessionStore,
    session_id: str,
    *,
    key: MemoryEvidenceKey | None,
    budget: MemoryAttributionCaptureBudget,
) -> MemoryAttribution:
    """Project one session without running or mutating application behavior."""

    if not isinstance(store, SessionStore):
        raise TypeError("store must be a SessionStore.")
    if type(session_id) is not str or not session_id:
        raise ValueError("session_id must be a non-empty string.")
    if key is not None and type(key) is not MemoryEvidenceKey:
        raise TypeError("key must be a MemoryEvidenceKey or None.")
    if type(budget) is not MemoryAttributionCaptureBudget:
        raise TypeError("budget must be a MemoryAttributionCaptureBudget.")
    if not store.supports_recall_evidence:
        return _empty_attribution(
            status=MemoryAttributionStatus.UNAVAILABLE,
            reason=MemoryAttributionUnavailableReason.STORE_UNSUPPORTED,
        )

    try:
        source = await _capture_source(store, session_id, budget)
    except Exception:
        return _empty_attribution(
            status=MemoryAttributionStatus.UNAVAILABLE,
            reason=MemoryAttributionUnavailableReason.EVIDENCE_READ_FAILED,
        )

    observed_receipts = len(source.receipts)
    observed_exposures = len(source.exposures)
    source_truncated = not source.receipts_complete or not source.exposures_complete
    if key is None:
        if observed_receipts or observed_exposures:
            observed_items = sum(len(receipt.items) for receipt in source.receipts)
            return MemoryAttribution(
                status=MemoryAttributionStatus.REDACTED,
                truncated=source_truncated,
                reason=MemoryAttributionUnavailableReason.ALIAS_KEY_UNAVAILABLE,
                observed_receipt_count=observed_receipts,
                observed_exposure_count=observed_exposures,
                observed_item_count=observed_items,
                omitted_receipt_count_at_least=(observed_receipts + source.receipt_more_at_least),
                omitted_exposure_count_at_least=(
                    observed_exposures + source.exposure_more_at_least
                ),
                omitted_item_count_at_least=observed_items,
            )
        if source_truncated:
            return _empty_attribution(
                status=MemoryAttributionStatus.TRUNCATED,
                truncated=True,
            )
        return _empty_attribution(status=MemoryAttributionStatus.COMPLETE)

    receipt_by_id = {receipt.receipt_id: receipt for receipt in source.receipts}
    if len(receipt_by_id) != len(source.receipts):
        return _contradictory(source)
    if (
        len({exposure.exposure_id for exposure in source.exposures}) != len(source.exposures)
        or len({exposure.model_attempt_id for exposure in source.exposures})
        != len(source.exposures)
        or len({exposure.provider_attempt_id for exposure in source.exposures})
        != len(source.exposures)
    ):
        return _contradictory(source)
    receipt_order = tuple((receipt.created_at, receipt.receipt_id) for receipt in source.receipts)
    exposure_order = tuple(
        (exposure.created_at, exposure.exposure_id) for exposure in source.exposures
    )
    if receipt_order != tuple(sorted(receipt_order)) or exposure_order != tuple(
        sorted(exposure_order)
    ):
        return _contradictory(source)
    try:
        for receipt in source.receipts:
            if receipt.session_id != session_id:
                raise ValueError("Recall receipt escaped its session scope.")
        for exposure in source.exposures:
            if exposure.session_id != session_id:
                raise ValueError("Context exposure escaped its session scope.")
            for receipt_id in exposure.receipt_ids:
                receipt = receipt_by_id.get(receipt_id)
                if receipt is None:
                    if source.receipts_complete:
                        raise ValueError("Context exposure references an absent recall receipt.")
                    continue
                validate_context_exposure_receipt_scope(exposure, receipt)
    except (TypeError, ValueError):
        return _contradictory(source)

    receipts: list[MemoryRecallAttribution] = []
    exposures: list[MemoryContextExposureAttribution] = []
    missing_linked_receipt_count = len(
        {
            receipt_id
            for exposure in source.exposures
            for receipt_id in exposure.receipt_ids
            if receipt_id not in receipt_by_id
        }
    )
    omitted_receipts = max(source.receipt_more_at_least, missing_linked_receipt_count)
    omitted_exposures = source.exposure_more_at_least
    omitted_items = 0
    observed_items = sum(len(receipt.items) for receipt in source.receipts)
    truncated = source_truncated

    for index, receipt in enumerate(source.receipts):
        available_items = min(len(receipt.items), budget.remaining_items)
        projected = _project_receipt(
            receipt,
            key=key,
            retained_item_count=available_items,
            projection_ordinal=index,
        )
        if not budget.retain_projection(projected):
            omitted_receipts += len(source.receipts) - index
            omitted_items += sum(len(candidate.items) for candidate in source.receipts[index:])
            truncated = True
            break
        receipts.append(projected)
        budget.items += available_items
        if available_items < len(receipt.items):
            omitted_items += len(receipt.items) - available_items
            truncated = True

    for index, exposure in enumerate(source.exposures):
        item_exposures: tuple[RecallItemExposure, ...] = ()
        verified_item_exposures: tuple[RecallItemExposure, ...] = ()
        unverified_item_count = 0
        items_complete = True
        if budget.remaining_source_bytes < _MAX_ITEM_EXPOSURE_BUNDLE_BYTES:
            items_complete = False
        else:
            try:
                item_exposures = await store.load_recall_item_exposures(
                    session_id,
                    exposure.exposure_id,
                )
                if type(item_exposures) is not tuple or any(
                    type(item) is not RecallItemExposure for item in item_exposures
                ):
                    raise TypeError("Recall item evidence has an invalid store shape.")
                if len(item_exposures) > MAX_RECALL_RECEIPT_ITEMS:
                    raise ValueError("Recall item evidence exceeded its count bound.")
                item_bytes = _document_sequence_bytes(
                    tuple(
                        memory_evidence_document_bytes(item, "memory attribution item")
                        for item in item_exposures
                    )
                )
                if item_bytes > budget.remaining_source_bytes:
                    raise ValueError("Recall item evidence exceeded its source byte bound.")
                budget.source_bytes += item_bytes
            except Exception:
                return _empty_attribution(
                    status=MemoryAttributionStatus.UNAVAILABLE,
                    reason=MemoryAttributionUnavailableReason.EVIDENCE_READ_FAILED,
                    observed_receipt_count=observed_receipts,
                    observed_exposure_count=observed_exposures,
                    observed_item_count=observed_items,
                    omitted_receipt_count_at_least=observed_receipts,
                    omitted_exposure_count_at_least=observed_exposures,
                    omitted_item_count_at_least=observed_items,
                )
        if items_complete:
            try:
                validate_new_context_exposure(
                    exposure.model_copy(
                        update={
                            "updated_at": exposure.created_at,
                            "state": exposure.transitions[0].state,
                            "state_revision": 0,
                            "transitions": exposure.transitions[:1],
                        }
                    ),
                    item_exposures,
                )
                verified_items: list[RecallItemExposure] = []
                for item in item_exposures:
                    receipt = receipt_by_id.get(item.receipt_id)
                    if receipt is None:
                        if source.receipts_complete:
                            raise ValueError("Recall item references an absent receipt.")
                        unverified_item_count += 1
                        continue
                    if not recall_item_exposure_matches_receipt_item(item, receipt):
                        raise ValueError("Recall item contradicts its immutable receipt item.")
                    verified_items.append(item)
                verified_item_exposures = tuple(verified_items)
            except (TypeError, ValueError):
                return _contradictory(
                    source,
                    observed_item_count=observed_items + len(item_exposures),
                )

        observed_items += len(item_exposures)

        available_items = min(len(verified_item_exposures), budget.remaining_items)
        projected = _project_exposure(
            exposure,
            verified_item_exposures[:available_items],
            key=key,
            projection_ordinal=index,
            items_truncated=(
                not items_complete
                or unverified_item_count > 0
                or available_items < len(verified_item_exposures)
            ),
            omitted_item_count_at_least=(
                unverified_item_count + max(len(verified_item_exposures) - available_items, 0)
            ),
        )
        if not budget.retain_projection(projected):
            omitted_exposures += len(source.exposures) - index
            omitted_items += len(item_exposures)
            truncated = True
            break
        exposures.append(projected)
        budget.items += available_items
        if projected.items_truncated:
            omitted_items += projected.omitted_item_count_at_least
            truncated = True

    status = MemoryAttributionStatus.TRUNCATED if truncated else MemoryAttributionStatus.COMPLETE
    try:
        return MemoryAttribution(
            status=status,
            truncated=truncated,
            observed_receipt_count=observed_receipts,
            observed_exposure_count=observed_exposures,
            observed_item_count=observed_items,
            omitted_receipt_count_at_least=omitted_receipts,
            omitted_exposure_count_at_least=omitted_exposures,
            omitted_item_count_at_least=omitted_items,
            receipts=tuple(receipts),
            exposures=tuple(exposures),
        )
    except (TypeError, ValueError):
        return _contradictory(source, observed_item_count=observed_items)


async def _capture_source(
    store: SessionStore,
    session_id: str,
    budget: MemoryAttributionCaptureBudget,
) -> _SourceCapture:
    receipts, receipts_complete, receipt_more = await _capture_receipts(
        store,
        session_id,
        budget,
    )
    exposures, exposures_complete, exposure_more = await _capture_exposures(
        store,
        session_id,
        budget,
    )
    return _SourceCapture(
        receipts=receipts,
        exposures=exposures,
        receipts_complete=receipts_complete,
        exposures_complete=exposures_complete,
        receipt_more_at_least=receipt_more,
        exposure_more_at_least=exposure_more,
    )


async def _capture_receipts(
    store: SessionStore,
    session_id: str,
    budget: MemoryAttributionCaptureBudget,
) -> tuple[tuple[RecallReceipt, ...], bool, int]:
    retained: list[RecallReceipt] = []
    cursor: str | None = None
    more_at_least = 0
    while (
        budget.remaining_receipts
        and budget.remaining_source_bytes >= MIN_MEMORY_EVIDENCE_PAGE_BYTES
    ):
        query = RecallEvidenceQuery(
            session_id=session_id,
            limit=min(MAX_MEMORY_EVIDENCE_PAGE_LIMIT, budget.remaining_receipts),
            max_bytes=min(MAX_MEMORY_EVIDENCE_PAGE_BYTES, budget.remaining_source_bytes),
            cursor=cursor,
        )
        page = await store.list_recall_receipts(query)
        if type(page) is not RecallReceiptPage:
            raise TypeError("Recall receipt page has an invalid store shape.")
        if len(page.items) > query.limit:
            raise ValueError("Recall receipt page exceeded its requested bounds.")
        page_bytes = _bounded_page_bytes(
            page.items,
            max_bytes=query.max_bytes,
            label="memory attribution receipt",
        )
        budget.source_bytes += page_bytes
        budget.receipts += len(page.items)
        retained.extend(page.items)
        if not page.truncated:
            return tuple(retained), True, 0
        more_at_least = 1
        if page.next_cursor is None or page.next_cursor == cursor:
            raise ValueError("Truncated recall receipt page has no forward cursor.")
        cursor = page.next_cursor
    return tuple(retained), False, more_at_least


async def _capture_exposures(
    store: SessionStore,
    session_id: str,
    budget: MemoryAttributionCaptureBudget,
) -> tuple[tuple[ContextExposure, ...], bool, int]:
    retained: list[ContextExposure] = []
    cursor: str | None = None
    more_at_least = 0
    while (
        budget.remaining_exposures
        and budget.remaining_source_bytes >= MIN_MEMORY_EVIDENCE_PAGE_BYTES
    ):
        query = RecallEvidenceQuery(
            session_id=session_id,
            limit=min(MAX_MEMORY_EVIDENCE_PAGE_LIMIT, budget.remaining_exposures),
            max_bytes=min(MAX_MEMORY_EVIDENCE_PAGE_BYTES, budget.remaining_source_bytes),
            cursor=cursor,
        )
        page = await store.list_context_exposures(query)
        if type(page) is not ContextExposurePage:
            raise TypeError("Context exposure page has an invalid store shape.")
        if len(page.items) > query.limit:
            raise ValueError("Context exposure page exceeded its requested bounds.")
        page_bytes = _bounded_page_bytes(
            page.items,
            max_bytes=query.max_bytes,
            label="memory attribution exposure",
        )
        budget.source_bytes += page_bytes
        budget.exposures += len(page.items)
        retained.extend(page.items)
        if not page.truncated:
            return tuple(retained), True, 0
        more_at_least = 1
        if page.next_cursor is None or page.next_cursor == cursor:
            raise ValueError("Truncated context exposure page has no forward cursor.")
        cursor = page.next_cursor
    return tuple(retained), False, more_at_least


def _project_receipt(
    receipt: RecallReceipt,
    *,
    key: MemoryEvidenceKey,
    retained_item_count: int,
    projection_ordinal: int,
) -> MemoryRecallAttribution:
    source_counts = {
        state: sum(source.state is state for source in receipt.sources)
        for state in RecallSourceCoverageState
    }
    return MemoryRecallAttribution(
        receipt_alias=_alias(receipt.receipt_id, "receipt", receipt.session_id, key),
        interaction_alias=_alias(
            receipt.interaction_id,
            "interaction",
            receipt.session_id,
            key,
        ),
        projection_ordinal=projection_ordinal,
        model_step_id=receipt.model_step_id,
        created_at=receipt.created_at,
        inspected_count=receipt.inspected_count,
        eligible_count=receipt.eligible_count,
        admitted_count=receipt.admitted_count,
        offered_count=receipt.offered_count,
        silent_count=receipt.silent_count,
        omitted_count=receipt.omitted_count,
        complete_source_count=source_counts[RecallSourceCoverageState.COMPLETE],
        partial_source_count=source_counts[RecallSourceCoverageState.PARTIAL],
        unavailable_source_count=source_counts[RecallSourceCoverageState.UNAVAILABLE],
        failed_source_count=source_counts[RecallSourceCoverageState.FAILED],
        truncated=receipt.truncated,
        items=tuple(
            MemoryRecallItemAttribution(
                item_alias=_item_alias(
                    receipt.receipt_id,
                    item.ordinal,
                    item.identity.model_dump(mode="json"),
                    item.representation_id,
                    item.content_sha256,
                    session_id=receipt.session_id,
                    key=key,
                ),
                ordinal=item.ordinal,
                admission=item.admission,
                selection_reason=item.selection_reason,
            )
            for item in receipt.items[:retained_item_count]
        ),
        items_truncated=retained_item_count < len(receipt.items),
        omitted_item_count_at_least=max(len(receipt.items) - retained_item_count, 0),
    )


def _project_exposure(
    exposure: ContextExposure,
    items: tuple[RecallItemExposure, ...],
    *,
    key: MemoryEvidenceKey,
    projection_ordinal: int,
    items_truncated: bool,
    omitted_item_count_at_least: int,
) -> MemoryContextExposureAttribution:
    return MemoryContextExposureAttribution(
        exposure_alias=_alias(exposure.exposure_id, "exposure", exposure.session_id, key),
        interaction_alias=_alias(
            exposure.interaction_id,
            "interaction",
            exposure.session_id,
            key,
        ),
        projection_ordinal=projection_ordinal,
        model_step_id=exposure.model_step_id,
        model_attempt_id=exposure.model_attempt_id,
        provider_attempt_id=exposure.provider_attempt_id,
        created_at=exposure.created_at,
        updated_at=exposure.updated_at,
        state=exposure.state,
        state_revision=exposure.state_revision,
        provider_exposure_proven=exposure.provider_exposure_proven,
        receipt_aliases=tuple(
            _alias(receipt_id, "receipt", exposure.session_id, key)
            for receipt_id in exposure.receipt_ids
        ),
        contributor_count=len(exposure.contributor_ids),
        transitions=tuple(
            MemoryExposureTransitionAttribution(
                revision=transition.revision,
                state=transition.state,
                occurred_at=transition.occurred_at,
                evidence_kind=transition.evidence_kind,
            )
            for transition in exposure.transitions
        ),
        items=tuple(
            MemoryExposureItemAttribution(
                item_alias=_item_alias(
                    item.receipt_id,
                    item.receipt_item_ordinal,
                    item.identity.model_dump(mode="json"),
                    item.representation_id,
                    item.content_sha256,
                    session_id=exposure.session_id,
                    key=key,
                ),
                receipt_alias=_alias(item.receipt_id, "receipt", exposure.session_id, key),
                ordinal=item_projection_ordinal,
                receipt_item_ordinal=item.receipt_item_ordinal,
                admission=item.admission,
                selection_reason=item.selection_reason,
            )
            for item_projection_ordinal, item in enumerate(items)
        ),
        items_truncated=items_truncated,
        omitted_item_count_at_least=omitted_item_count_at_least,
    )


def _item_alias(
    receipt_id: str,
    receipt_item_ordinal: int,
    identity: object,
    representation_id: str,
    content_sha256: str,
    *,
    session_id: str,
    key: MemoryEvidenceKey,
) -> MemoryEvidenceAlias:
    private_value = canonical_durable_json_bytes(
        {
            "receipt_id": receipt_id,
            "receipt_item_ordinal": receipt_item_ordinal,
            "identity": identity,
            "representation_id": representation_id,
            "content_sha256": content_sha256,
        },
        "memory attribution item alias",
    ).decode("utf-8")
    return _alias(private_value, "item", session_id, key)


def _document_sequence_bytes(documents: tuple[bytes, ...]) -> int:
    return 2 + sum(map(len, documents)) + max(len(documents) - 1, 0)


def _bounded_page_bytes(
    items: tuple[RecallReceipt, ...] | tuple[ContextExposure, ...],
    *,
    max_bytes: int,
    label: str,
) -> int:
    """Measure a page while stopping as soon as its requested byte ceiling is crossed."""

    total = 2
    for index, item in enumerate(items):
        total += len(memory_evidence_document_bytes(item, label)) + (1 if index else 0)
        if total > max_bytes:
            raise ValueError("Memory evidence page exceeded its requested byte bound.")
    return total


def _alias(
    private_value: str,
    kind: _AliasKind,
    session_id: str,
    key: MemoryEvidenceKey,
) -> MemoryEvidenceAlias:
    payload = canonical_durable_json_bytes(
        {
            "version": 1,
            "kind": kind,
            "session_id": session_id,
            "private_value": private_value,
        },
        "memory evidence alias",
    )
    return MemoryEvidenceAlias(
        key_id=key.key_id,
        kind=kind,
        digest=hmac.digest(key.key, _ALIAS_CONTEXT + b"\x00" + payload, "sha256").hex(),
    )


def _contradictory(
    source: _SourceCapture,
    *,
    observed_item_count: int | None = None,
) -> MemoryAttribution:
    if observed_item_count is None:
        observed_item_count = sum(len(receipt.items) for receipt in source.receipts)
    return MemoryAttribution(
        status=MemoryAttributionStatus.CONTRADICTORY,
        truncated=not source.receipts_complete or not source.exposures_complete,
        reason=MemoryAttributionUnavailableReason.CONTRADICTORY_EVIDENCE,
        observed_receipt_count=len(source.receipts),
        observed_exposure_count=len(source.exposures),
        observed_item_count=observed_item_count,
        omitted_receipt_count_at_least=(len(source.receipts) + source.receipt_more_at_least),
        omitted_exposure_count_at_least=(len(source.exposures) + source.exposure_more_at_least),
        omitted_item_count_at_least=observed_item_count,
    )


def _empty_attribution(
    *,
    status: MemoryAttributionStatus,
    truncated: bool = False,
    reason: MemoryAttributionUnavailableReason | None = None,
    observed_receipt_count: int = 0,
    observed_exposure_count: int = 0,
    observed_item_count: int = 0,
    omitted_receipt_count_at_least: int = 0,
    omitted_exposure_count_at_least: int = 0,
    omitted_item_count_at_least: int = 0,
) -> MemoryAttribution:
    return MemoryAttribution(
        status=status,
        truncated=truncated,
        reason=reason,
        observed_receipt_count=observed_receipt_count,
        observed_exposure_count=observed_exposure_count,
        observed_item_count=observed_item_count,
        omitted_receipt_count_at_least=omitted_receipt_count_at_least,
        omitted_exposure_count_at_least=omitted_exposure_count_at_least,
        omitted_item_count_at_least=omitted_item_count_at_least,
    )


__all__ = ["MemoryAttributionCaptureBudget", "project_memory_attribution"]
