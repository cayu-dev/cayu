"""Final source acknowledgement after the native owner's durable retention release."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import (
    CollaborationConflict,
    ContractValue,
    Generation,
    OperationRef,
)
from cayu.collaboration._permit_store import settle_permit_in_transaction
from cayu.collaboration._permits import ReceivingSettlementReceipt
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import (
    ProducerAdmittedCleanup,
    ProducerCleanupRecord,
    ProducerOutputRecord,
)
from cayu.collaboration._producer_delivery_store import read_delivery
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._producer_terminal import producer_terminal
from cayu.collaboration._request_store import operation_key, retained_request
from cayu.collaboration.base import _Anchor
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestSnapshot
from cayu.runtime._producer_cleanup_receipt import read_cleanup_receipt
from cayu.sessions._producer_cleanup_contract import NativeProducerCleanupReceipt

_SEAL = object()


@dataclass(frozen=True)
class _FinalizationAuthority:
    command: bytes
    native: bytes
    seal: object

    def require(self, command, native, redactor):
        if (
            self.seal is not _SEAL
            or self.command != contract_bytes(command, redactor=redactor)
            or self.native != contract_bytes(native, redactor=redactor)
        ):
            raise CollaborationConflict("Producer finalization lacks native-owner readback.")


def _received_native_cleanup(command, native, redactor):
    return _FinalizationAuthority(
        contract_bytes(command, redactor=redactor), contract_bytes(native, redactor=redactor), _SEAL
    )


class ProducerCleanupFinalized(ContractValue):
    mode: Literal["producer_cleanup_finalized"] = "producer_cleanup_finalized"
    operation: OperationRef
    registration: OperationRef
    cleanup: OperationRef
    sequence: Generation
    native_receipt: NativeProducerCleanupReceipt
    delivery: Literal["published", "excluded"]


def finalization_operation(command, redactor):
    return command.operation.model_copy(
        update={
            "caller_key": "producer-cleanup-final:"
            + sha256(contract_bytes(command.operation, redactor=redactor)).hexdigest(),
        }
    )


async def read_finalization(tx, record, *, redactor):
    operation = finalization_operation(record.command, redactor)
    raw = await tx.get("operations", operation_key(operation))
    if record.cleanup_ack is None:
        if raw is not None:
            raise CollaborationUnavailable("Producer cleanup acknowledgement is inconsistent.")
        return None
    final = prepare_contract(ProducerCleanupFinalized, raw, redactor=redactor)
    if (
        not isinstance(record.cleanup, (ProducerAdmittedCleanup, ProducerCleanupRecord))
        or final.operation != operation
        or record.cleanup_ack != operation
        or final.registration != record.command.operation
        or final.cleanup != record.cleanup.operation
        or final.sequence <= record.cleanup.sequence
    ):
        raise CollaborationConflict("Producer final cleanup identity conflicts.")
    read_cleanup_receipt(record, final.native_receipt)
    if final.delivery != await delivery_summary(tx, record, redactor=redactor):
        raise CollaborationConflict("Producer final delivery summary conflicts.")
    return final


async def delivery_summary(tx, record, *, redactor):
    """All destinations are terminal; individual receipts preserve each outcome."""
    cleanup = record.cleanup
    if isinstance(cleanup, ProducerCleanupRecord):
        return "excluded"
    assert isinstance(cleanup, ProducerAdmittedCleanup)
    published = False
    for settled in cleanup.evidence.destinations:
        destination = next(
            (item for item in record.command.destinations if item.operation == settled.destination),
            None,
        )
        if destination is None:
            raise CollaborationConflict("Producer cleanup has an unregistered destination.")
        delivery = await read_delivery(tx, record.command, destination, redactor=redactor)
        if settled.kind == "destination_exclusion":
            from cayu.collaboration._producer_destination_exclusion import (
                read_destination_exclusion,
            )

            if (
                delivery is not None
                or await read_destination_exclusion(
                    tx, record.command, destination, redactor=redactor
                )
                is None
            ):
                raise CollaborationUnavailable("Producer exclusion is unavailable.")
            continue
        if settled.kind == "export_retirement":
            from cayu.collaboration._producer_export_store import read_export

            if (
                cleanup.evidence.terminal_kind != "closure"
                or delivery is not None
                or await read_export(tx, record.command, destination, redactor=redactor) is None
            ):
                raise CollaborationConflict("Producer retired-export responsibility conflicts.")
            continue
        if delivery is None or delivery.receipt is None or delivery.acceptance is None:
            raise CollaborationUnavailable("Producer destination settlement is unavailable.")
        published |= delivery.receipt.status == "appended"
    return "published" if published else "excluded"


async def finalize_cleanup(store, tx, initialized, command, native, *, authority, redactor):
    """Private receiving-owner seam; native must come from fixed exact readback."""
    if type(authority) is not _FinalizationAuthority:
        raise CollaborationConflict("Producer finalization requires authenticated native readback.")
    authority.require(command, native, redactor)
    record = await read_output_registration(tx, command, redactor=redactor)
    if record is None or not isinstance(
        record.cleanup, (ProducerAdmittedCleanup, ProducerCleanupRecord)
    ):
        raise CollaborationUnavailable("Producer cleanup acceptance is unavailable.")
    native = read_cleanup_receipt(record, native)
    if native is None:
        raise CollaborationUnavailable("Native producer cleanup remains unresolved.")
    prior = await read_finalization(tx, record, redactor=redactor)
    if prior is not None:
        require_exact_contract(prior.native_receipt, native, redactor=redactor)
        return prior
    expected = command.admission.expected
    request = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if (
        request is None
        or request.producer_operation != command.operation
        or request.producer_settlement is not None
    ):
        raise CollaborationUnavailable("Producer request settlement evidence conflicts.")
    expected_terminal = (
        ("closure", record.cleanup.exclusion.control_operation)
        if isinstance(record.cleanup, ProducerCleanupRecord)
        else (record.cleanup.evidence.terminal_kind, record.cleanup.evidence.terminal)
    )
    if producer_terminal(request, command, redactor=redactor) != expected_terminal:
        raise CollaborationConflict("Producer final cleanup terminal decision changed.")
    receipt_id = (
        "native-producer-cleanup:" + sha256(contract_bytes(native, redactor=redactor)).hexdigest()
    )
    for permit in (request.permit, record.permit):
        await settle_permit_in_transaction(
            store,
            tx,
            initialized,
            permit,
            ReceivingSettlementReceipt(
                expected=permit,
                receiving_owner=command.receiver.owner,
                receipt_id=receipt_id,
                outcome="quiescent",
                admission_excluded=isinstance(record.cleanup, ProducerCleanupRecord),
            ),
            redactor,
        )
    anchor = await store._anchor(tx, initialized, redactor)
    operation = finalization_operation(command, redactor)
    final = ProducerCleanupFinalized(
        operation=operation,
        registration=command.operation,
        cleanup=record.cleanup.operation,
        sequence=anchor.event_sequence + 1,
        native_receipt=native,
        delivery=await delivery_summary(tx, record, redactor=redactor),
    )
    updated = prepare_contract(
        ProducerOutputRecord,
        record.model_copy(
            update={
                "cleanup_ack": operation,
                "reserved_operations": 0,
                "reserved_events": 0,
                "reserved_bytes": 0,
            }
        ),
        redactor=redactor,
    )
    updated_request = prepare_contract(
        RequestSnapshot,
        request.model_copy(update={"producer_settlement": operation, "delivery": final.delivery}),
        redactor=redactor,
    )
    charge = sum(
        len(contract_bytes(item, redactor=redactor))
        for item in (
            final,
            updated,
            updated_request,
        )
    ) - sum(len(contract_bytes(item, redactor=redactor)) for item in (record, request))
    if charge > record.reserved_bytes:
        raise CollaborationUnavailable("Producer final cleanup exceeds reserved capacity.")
    accounting = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": final.sequence,
                "reserved_operations": anchor.reserved_operations - record.reserved_operations,
                "reserved_events": anchor.reserved_events - record.reserved_events,
                "reserved_bytes": anchor.reserved_bytes - record.reserved_bytes,
                "retained_bytes": anchor.retained_bytes + charge,
            }
        ),
        redactor=redactor,
    )
    require_capacity(accounting, ordinary=False)
    await tx.put("operations", operation_key(operation), final, insert=True)
    await tx.put("operations", operation_key(command.operation), updated, insert=False)
    await tx.put("requests", operation_key(expected.operation), updated_request, insert=False)
    await tx.put("anchors", (), accounting, insert=False)
    return final
