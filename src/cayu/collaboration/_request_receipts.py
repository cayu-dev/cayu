"""Typed metadata for the post-acceptance request receipt families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from cayu.collaboration._clarification_commands import (
    ClarificationCloseReceipt,
    ClarificationOpenReceipt,
    ClarificationReplyReceipt,
)
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
    "request_admission",
    "request_progress",
    "request_outcome",
    "producer_outcome",
    "request_observation",
    "clarification_open",
    "clarification_reply",
    "clarification_close",
]
RequestReceipt = (
    RequestAdmissionReceipt
    | RequestProgressReceipt
    | RequestOutcomeReceipt
    | RequestObservationReceipt
    | ClarificationOpenReceipt
    | ClarificationReplyReceipt
    | ClarificationCloseReceipt
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
    "producer_outcome": RequestOutcomeReceipt,
    "request_observation": RequestObservationReceipt,
    "clarification_open": ClarificationOpenReceipt,
    "clarification_reply": ClarificationReplyReceipt,
    "clarification_close": ClarificationCloseReceipt,
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
    if document.get("mode") == "request_plan_stage":
        from cayu.collaboration._planning_records import RequestPlanningStageRecord

        return prepare_contract(
            RequestPlanningStageRecord, document, redactor=redactor
        ).intent.operation
    if document.get("mode") in {"producer_destination_exclusion", "producer_export_rejected"}:
        from cayu.collaboration._producer_pruning import producer_record

        record = producer_record(document, redactor=redactor)
        assert record is not None
        return record.operation
    if document.get("mode") in {
        "producer_output_record",
        "producer_request_index",
        "producer_output_registered",
        "producer_launch_decision",
        "producer_cleanup",
        "producer_admitted_cleanup",
        "producer_completion",
        "producer_export",
        "producer_export_published",
        "producer_delivery",
        "producer_delivery_accepted",
        "producer_delivery_index",
    }:
        from cayu.collaboration._producer_contracts import (
            ProducerAdmittedCleanup,
            ProducerCleanupRecord,
            ProducerCompletionRecord,
            ProducerDeliveryAccepted,
            ProducerDeliveryIndex,
            ProducerDeliveryRecord,
            ProducerExportPublished,
            ProducerExportRecord,
            ProducerLaunchDecision,
            ProducerOutputRecord,
            ProducerRegistrationEvent,
            ProducerRequestIndex,
        )

        if document.get("mode") == "producer_delivery_index":
            return prepare_contract(ProducerDeliveryIndex, document, redactor=redactor).operation
        if document.get("mode") == "producer_delivery":
            return prepare_contract(ProducerDeliveryRecord, document, redactor=redactor).operation
        if document.get("mode") == "producer_delivery_accepted":
            return prepare_contract(ProducerDeliveryAccepted, document, redactor=redactor).operation
        if document.get("mode") == "producer_export":
            return prepare_contract(ProducerExportRecord, document, redactor=redactor).operation
        if document.get("mode") == "producer_export_published":
            return prepare_contract(ProducerExportPublished, document, redactor=redactor).operation
        if document.get("mode") == "producer_completion":
            return prepare_contract(ProducerCompletionRecord, document, redactor=redactor).operation
        if document.get("mode") == "producer_cleanup":
            return prepare_contract(ProducerCleanupRecord, document, redactor=redactor).operation
        if document.get("mode") == "producer_admitted_cleanup":
            return prepare_contract(ProducerAdmittedCleanup, document, redactor=redactor).operation
        if document.get("mode") == "producer_launch_decision":
            return prepare_contract(ProducerLaunchDecision, document, redactor=redactor).operation
        if document.get("mode") == "producer_output_record":
            return prepare_contract(
                ProducerOutputRecord, document, redactor=redactor
            ).command.operation
        if document.get("mode") == "producer_request_index":
            return prepare_contract(ProducerRequestIndex, document, redactor=redactor).operation
        return prepare_contract(ProducerRegistrationEvent, document, redactor=redactor).operation
    if document.get("mode") == "producer_cleanup_finalized":
        from cayu.collaboration._producer_cleanup_finalization import ProducerCleanupFinalized

        return prepare_contract(ProducerCleanupFinalized, document, redactor=redactor).operation
    if document.get("mode") == "clarification_delivery":
        from cayu.collaboration._clarification_deliveries import ClarificationDeliveryRecord

        return prepare_contract(
            ClarificationDeliveryRecord, document, redactor=redactor
        ).intent.operation
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
