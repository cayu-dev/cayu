"""Backend-neutral authority rules for runtime-owned product operation stores.

SQLite and PostgreSQL differ only in how they fence one operation row; every
decision about reservation, claim ownership, receipts, and settlement is made
here so both stores keep identical semantics.
"""

from __future__ import annotations

import json
from typing import Any, Literal, cast, get_args

from cayu._validation import require_durable_clean_nonblank
from cayu.server import (
    ProductExecutionClaimLost,
    ProductIdempotencyConflict,
    ProductOperation,
    ProductOperationReservation,
    ProductOperationSettlementConflict,
    ProductRecoveryStatus,
    ProductResultReceipt,
    ProductResultReceiptConflict,
)
from cayu.server.service import MAX_PRODUCT_IDENTITY_CHARS
from cayu.storage._product_operation_schema import PRODUCT_OPERATIONS_REVISION

PRODUCT_OPERATION_MIN_REQUIRED_REVISION = PRODUCT_OPERATIONS_REVISION

_RECOVERY_STATUSES = frozenset(get_args(ProductRecoveryStatus))
_TERMINAL_STATUSES = frozenset(("completed", "failed"))


def identity(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string.")
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > MAX_PRODUCT_IDENTITY_CHARS:
        raise ValueError(f"{field_name} must not exceed {MAX_PRODUCT_IDENTITY_CHARS} characters.")
    return value


def lookup_identity(value: object, field_name: str) -> str | None:
    """Return a lookup key, or ``None`` when no stored identity could match."""

    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string.")
    try:
        return identity(value, field_name)
    except ValueError:
        return None


def lease_seconds(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("lease_seconds must be a positive integer.")
    return value


def recovery_status(value: object) -> ProductRecoveryStatus:
    if type(value) is not str or value not in _RECOVERY_STATUSES:
        raise ValueError("recovery_status must be a supported product recovery status.")
    return cast("ProductRecoveryStatus", value)


def settlement(status: object, result: object) -> tuple[Literal["completed", "failed"], str | None]:
    if type(status) is not str or status not in _TERMINAL_STATUSES:
        raise ValueError("status must be 'completed' or 'failed'.")
    if result is not None and type(result) is not str:
        raise TypeError("result must be a string or None.")
    return cast('Literal["completed", "failed"]', status), result


def result_receipt(value: object) -> ProductResultReceipt:
    if not isinstance(value, ProductResultReceipt):
        raise TypeError("receipt must be a ProductResultReceipt.")
    return ProductResultReceipt.model_validate(value.model_dump(mode="python"))


def encode_receipt(receipt: ProductResultReceipt) -> str:
    return json.dumps(
        receipt.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def pending_operation(
    *,
    tenant_id: str,
    subject_id: str,
    idempotency_key: str,
    request_fingerprint: str,
    public_id: str,
    work_id: str,
    session_id: str,
    task_id: str,
    request_text: str,
) -> ProductOperation:
    return ProductOperation(
        tenant_id=tenant_id,
        subject_id=subject_id,
        public_id=public_id,
        work_id=work_id,
        idempotency_key=idempotency_key,
        request_fingerprint=request_fingerprint,
        session_id=session_id,
        task_id=task_id,
        request_text=request_text,
        status="pending",
        result=None,
    )


def operation_from_row(row: Any) -> ProductOperation:
    """Reconstruct the public-model projection of one stored row.

    Claim ownership columns never leave the store boundary.
    """

    fields = {name: row[name] for name in ProductOperation.model_fields}
    if isinstance(fields["result_receipt"], str):
        fields["result_receipt"] = json.loads(fields["result_receipt"])
    return ProductOperation.model_validate(fields)


def existing_reservation(
    existing: ProductOperation,
    *,
    tenant_id: str,
    request_fingerprint: str,
) -> ProductOperationReservation:
    """Return an idempotent replay or reject a key bound to other trusted work."""

    if existing.tenant_id != tenant_id or existing.request_fingerprint != request_fingerprint:
        raise ProductIdempotencyConflict
    return ProductOperationReservation(operation=existing, created=False)


def claim_available(
    *,
    current_claim_id: str | None,
    claim_id: str,
    lease_expired: bool,
) -> bool:
    """Whether ``claim_id`` may acquire or renew pending work."""

    return current_claim_id is None or current_claim_id == claim_id or lease_expired


def heartbeat_recognizes_settlement(
    *,
    status: str,
    current_claim_id: str | None,
    claim_id: str,
) -> bool:
    return status != "pending" and current_claim_id == claim_id


def release_outcome(
    *,
    status: str,
    current_claim_id: str | None,
    claim_id: str,
) -> bool | None:
    """Return a final answer, or ``None`` when ``claim_id`` must be cleared."""

    if status != "pending" or current_claim_id != claim_id:
        return status == "pending" and current_claim_id is None
    return None


def receipt_write_required(
    operation: ProductOperation,
    *,
    current_claim_id: str | None,
    claim_id: str,
    receipt: ProductResultReceipt,
) -> bool:
    """Decide whether the exact claim may insert or advance ``receipt``.

    ``False`` means the identical receipt is already committed and must be
    returned unchanged after an ambiguous acknowledgement.
    """

    if operation.status != "pending" or current_claim_id != claim_id:
        if (
            operation.status != "pending"
            and current_claim_id == claim_id
            and operation.result_receipt == receipt
        ):
            return False
        raise ProductExecutionClaimLost(
            "Product execution ownership was lost before result publication."
        )
    if (
        receipt.work_id != operation.work_id
        or receipt.public_id != operation.public_id
        or receipt.request_fingerprint != operation.request_fingerprint
        or receipt.session_id != operation.session_id
        or receipt.task_id != operation.task_id
    ):
        raise ProductResultReceiptConflict(
            "Result receipt does not belong to this product operation."
        )
    recorded = operation.result_receipt
    if recorded is not None:
        if recorded == receipt:
            return False
        if receipt.source_event_sequence <= recorded.source_event_sequence:
            raise ProductResultReceiptConflict("Product work already has newer result evidence.")
    return True


def require_recovery_owner(
    operation: ProductOperation,
    *,
    current_claim_id: str | None,
    claim_id: str,
) -> None:
    if operation.status != "pending" or current_claim_id != claim_id:
        raise ProductExecutionClaimLost(
            "Product execution ownership was lost before recovery reporting."
        )


def finish_write_required(
    operation: ProductOperation,
    *,
    current_claim_id: str | None,
    claim_id: str,
    status: Literal["completed", "failed"],
    result: str | None,
) -> bool:
    """Decide conditional settlement, or recognize this claim's committed write."""

    receipt = operation.result_receipt
    if status == "completed" and (
        receipt is None or receipt.publication_status != "completed" or receipt.result != result
    ):
        raise ProductOperationSettlementConflict(
            "Completed product work does not match its result receipt."
        )
    if status == "failed" and result is not None:
        raise ProductOperationSettlementConflict(
            "Failed product work cannot persist a public result."
        )
    if operation.status != "pending":
        if (
            current_claim_id == claim_id
            and operation.status == status
            and operation.result == result
        ):
            return False
        if current_claim_id != claim_id:
            raise ProductExecutionClaimLost(
                "Product execution ownership was lost before completion."
            )
        raise ProductOperationSettlementConflict(
            "Product work already has a different terminal result."
        )
    if current_claim_id != claim_id:
        raise ProductExecutionClaimLost("Product execution ownership was lost before completion.")
    return True
