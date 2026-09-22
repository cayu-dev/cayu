"""Typed metadata for the post-acceptance request receipt families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from cayu.collaboration._contracts import CollaborationContractError, OperationRef
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.requests import (
    RequestAdmissionReceipt,
    RequestCommand,
    RequestObservationReceipt,
    RequestOutcomeReceipt,
    RequestProgressReceipt,
)
from cayu.vaults.redaction import SecretRedactor

RequestReceiptMode = Literal[
    "request_admission", "request_progress", "request_outcome", "request_observation"
]
RequestReceipt = (
    RequestAdmissionReceipt
    | RequestProgressReceipt
    | RequestOutcomeReceipt
    | RequestObservationReceipt
)


@dataclass(frozen=True)
class RequestReceiptMetadata:
    """Validated operation identity and parent request for one new receipt."""

    mode: RequestReceiptMode
    operation: OperationRef
    expected: RequestCommand
    receipt: RequestReceipt


_SCHEMAS: dict[RequestReceiptMode, type[RequestReceipt]] = {
    "request_admission": RequestAdmissionReceipt,
    "request_progress": RequestProgressReceipt,
    "request_outcome": RequestOutcomeReceipt,
    "request_observation": RequestObservationReceipt,
}


def request_receipt_metadata(
    raw: object, *, redactor: SecretRedactor
) -> RequestReceiptMetadata | None:
    """Validate and describe a recognized new request receipt.

    The mode is read only as a schema selector. A record with a recognized mode
    is then fully prepared before it is returned. Therefore malformed records
    cannot be mistaken for a valid different-family operation conflict.
    """

    if not isinstance(raw, dict):
        return None
    document = cast("dict[str, object]", raw)
    mode: object = document.get("mode")
    command = document.get("command")
    if mode is None and isinstance(command, dict):
        mode = cast("dict[str, object]", command).get("mode")
    if not isinstance(mode, str) or mode not in _SCHEMAS:
        return None
    mode = cast("RequestReceiptMode", mode)
    schema = _SCHEMAS[mode]
    receipt = prepare_contract(schema, raw, redactor=redactor)
    if isinstance(receipt, RequestObservationReceipt):
        operation = receipt.operation
        expected = receipt.expected
    else:
        operation = receipt.command.operation
        expected = receipt.command.expected
    return RequestReceiptMetadata(
        mode=mode,
        operation=operation,
        expected=expected,
        receipt=receipt,
    )


def record_operation(raw: object, *, redactor: SecretRedactor) -> OperationRef:
    """Extract a stored operation after validating request-family records."""

    metadata = request_receipt_metadata(raw, redactor=redactor)
    if metadata is not None:
        return metadata.operation
    if not isinstance(raw, dict):
        raise CollaborationContractError("Stored operation record is malformed.")
    document = cast("dict[str, object]", raw)
    if document.get("mode") == "collaboration_wait":
        from cayu.collaboration.waits import WaitSnapshot

        return prepare_contract(
            WaitSnapshot, document, redactor=redactor
        ).registration.wait.operation
    if document.get("record_type") in ("permit_settlement_reserved", "permit_settled"):
        expected = document.get("expected")
        if not isinstance(expected, dict):
            raise CollaborationContractError("Stored permit settlement is malformed.")
        expected = cast("dict[str, object]", expected)
        intent = expected.get("intent")
        if not isinstance(intent, dict):
            raise CollaborationContractError("Stored permit settlement is malformed.")
        request = cast("dict[str, object]", intent).get("request")
        if not isinstance(request, dict):
            raise CollaborationContractError("Stored permit settlement is malformed.")
        operation = cast("dict[str, object]", request).get("settlement_operation")
    elif isinstance(document.get("expected"), dict):
        operation = cast("dict[str, object]", document["expected"]).get("operation")
    elif isinstance(document.get("command"), dict):
        operation = cast("dict[str, object]", document["command"]).get("operation")
    else:
        operation = document.get("operation")
    return prepare_contract(OperationRef, operation, redactor=redactor)
