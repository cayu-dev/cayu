"""Atomic source acceptance of registered-owner completion/settlement evidence."""

from dataclasses import dataclass
from hashlib import sha256

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES, MAX_ID_BYTES, CollaborationConflict
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._producer_contracts import (
    ProducerAdmittedCleanup,
    ProducerDestinationSettlement,
    ProducerOutputRecord,
    ProducerSettlementEvidence,
)
from cayu.collaboration._producer_delivery_store import read_delivery
from cayu.collaboration._producer_store import read_output_registration
from cayu.collaboration._producer_terminal import cleanup_destinations, producer_terminal
from cayu.collaboration._request_store import operation_key, retained_request
from cayu.collaboration.base import _Anchor
from cayu.collaboration.participants import CollaborationUnavailable

_SEAL = object()


@dataclass(frozen=True)
class _CleanupAuthority:
    command: bytes
    evidence: bytes
    seal: object

    def require(self, command, evidence, redactor):
        if (
            self.seal is not _SEAL
            or self.command != contract_bytes(command, redactor=redactor)
            or self.evidence != contract_bytes(evidence, redactor=redactor)
        ):
            raise CollaborationConflict("Producer cleanup requires authenticated owner evidence.")


def _received_cleanup(command, evidence, redactor):
    return _CleanupAuthority(
        contract_bytes(command, redactor=redactor),
        contract_bytes(evidence, redactor=redactor),
        _SEAL,
    )


def cleanup_operation(command, redactor):
    return command.operation.model_copy(
        update={
            "caller_key": "producer-cleanup:"
            + sha256(contract_bytes(command.operation, redactor=redactor)).hexdigest()
        }
    )


def preflight_cleanup(record, redactor):
    """Size the complete future acceptance envelope before native dispatch."""
    from cayu.collaboration._producer_delivery_store import delivery_operation

    command = record.command
    digest = "sha256:" + "f" * 64
    cleanup = ProducerAdmittedCleanup(
        operation=cleanup_operation(command, redactor),
        registration=command.operation,
        sequence=2**53 - 1,
        evidence=ProducerSettlementEvidence(
            registration=command.operation,
            registration_commitment=digest,
            completion=record.completion,
            completion_commitment=digest,
            # A later control key is caller-selected, unlike our hashed outcome
            # key. Reserve its full escaped representation before dispatch.
            terminal=command.operation.model_copy(update={"caller_key": "\x01" * MAX_ID_BYTES}),
            terminal_kind="closure",
            native_release_commitment=digest,
            native_release_run_epoch=2**53 - 1,
            budget_settlement_commitment=digest,
            destinations=tuple(
                ProducerDestinationSettlement(
                    destination=item.operation,
                    delivery=delivery_operation(item, redactor),
                    receipt_commitment=digest,
                    export_settlement_commitment=digest,
                    export_state="released",
                )
                for item in command.destinations
            ),
        ),
    )
    prepare_contract(
        ProducerOutputRecord, record.model_copy(update={"cleanup": cleanup}), redactor=redactor
    )
    from cayu.collaboration._producer_cleanup_finalization import finalization_operation

    prepare_contract(
        ProducerOutputRecord,
        record.model_copy(
            update={
                "cleanup": cleanup,
                "cleanup_ack": finalization_operation(command, redactor),
                "reserved_operations": 0,
                "reserved_events": 0,
                "reserved_bytes": 0,
            }
        ),
        redactor=redactor,
    )


async def accept_admitted_cleanup(
    store, tx, initialized, command, evidence, *, authority, redactor
):
    """Private seam: caller must authenticate all foreign owner readbacks.

    Serialized evidence is not authority; this function is not a public mutation
    entrance. The registered cleanup coordinator owns the preceding reads.
    This prepares cleanup only: permits and remaining capacity stay pending until
    the native acknowledgement and final responsibility settlement are retained.
    """
    evidence = prepare_contract(ProducerSettlementEvidence, evidence, redactor=redactor)
    if type(authority) is not _CleanupAuthority:
        raise CollaborationConflict("Producer cleanup requires registered-owner readback.")
    authority.require(command, evidence, redactor)
    record = await read_output_registration(tx, command, redactor=redactor)
    if (
        record is None
        or record.state != "launch_claimed"
        or record.completion != evidence.completion
        or evidence.registration != command.operation
        or evidence.registration_commitment
        != "sha256:" + sha256(contract_bytes(command, redactor=redactor)).hexdigest()
    ):
        raise CollaborationConflict("Producer cleanup requires its exact admitted registration.")
    operation = cleanup_operation(command, redactor)
    if record.cleanup is not None:
        if not isinstance(record.cleanup, ProducerAdmittedCleanup):
            raise CollaborationConflict("Producer cleanup disposition conflicts.")
        require_exact_contract(record.cleanup.evidence, evidence, redactor=redactor)
        return record.cleanup
    expected = command.admission.expected
    request = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if (
        request is None
        or request.producer_operation != command.operation
        or request.producer_settlement is not None
    ):
        raise CollaborationUnavailable("Producer cleanup lacks its exact terminal disposition.")
    if producer_terminal(request, command, redactor=redactor) != (
        evidence.terminal_kind,
        evidence.terminal,
    ):
        raise CollaborationConflict("Producer cleanup terminal decision changed.")
    raw = await tx.get("operations", operation_key(record.completion))
    from cayu.collaboration._producer_contracts import ProducerCompletionRecord

    completion = prepare_contract(ProducerCompletionRecord, raw, redactor=redactor)
    if evidence.completion_commitment != (
        "sha256:" + sha256(contract_bytes(completion, redactor=redactor)).hexdigest()
    ):
        raise CollaborationConflict("Producer cleanup completion changed.")
    destinations = cleanup_destinations(record, completion, request, redactor=redactor)
    if destinations:
        if tuple(item.destination for item in evidence.destinations) != tuple(
            item.operation for item in destinations
        ):
            raise CollaborationConflict("Producer cleanup omits a registered destination.")
        for destination, settled in zip(destinations, evidence.destinations, strict=True):
            delivery = await read_delivery(tx, command, destination, redactor=redactor)
            if settled.kind == "destination_exclusion":
                from cayu.collaboration._producer_destination_exclusion import (
                    read_destination_exclusion,
                )

                if (
                    delivery is not None
                    or await read_destination_exclusion(tx, command, destination, redactor=redactor)
                    is None
                ):
                    raise CollaborationUnavailable("Producer destination exclusion is unavailable.")
                continue
            if settled.kind == "export_retirement":
                if evidence.terminal_kind != "closure" or delivery is not None:
                    raise CollaborationUnavailable(
                        "Export retirement cannot settle a receiving attempt."
                    )
                # The sealed cleanup handoff authenticates native terminal export
                # evidence; durable closure and this transaction fence new intent.
                continue
            if (
                delivery is None
                or delivery.receipt is None
                or delivery.acceptance is None
                or delivery.operation != settled.delivery
                or settled.receipt_commitment
                != "sha256:"
                + sha256(contract_bytes(delivery.receipt, redactor=redactor)).hexdigest()
            ):
                raise CollaborationUnavailable("Producer receiving responsibility is unsettled.")
    elif evidence.destinations or record.exports or record.deliveries:
        raise CollaborationConflict("Failed producer cleanup has unexpected deliveries.")
    if await tx.get("operations", operation_key(operation)) is not None:
        raise CollaborationConflict("Producer cleanup operation is occupied.")
    anchor = await store._anchor(tx, initialized, redactor)
    reservation = 2 * MAX_ENVELOPE_BYTES
    cleanup = ProducerAdmittedCleanup(
        operation=operation,
        registration=command.operation,
        sequence=anchor.event_sequence + 1,
        evidence=evidence,
    )
    updated = prepare_contract(
        ProducerOutputRecord,
        record.model_copy(
            update={
                "cleanup": cleanup,
                "reserved_operations": record.reserved_operations - 1,
                "reserved_events": record.reserved_events - 1,
                "reserved_bytes": record.reserved_bytes - reservation,
            }
        ),
        redactor=redactor,
    )
    charge = sum(len(contract_bytes(item, redactor=redactor)) for item in (cleanup, updated)) - len(
        contract_bytes(record, redactor=redactor)
    )
    if charge > reservation:
        raise CollaborationUnavailable("Producer cleanup exceeds reserved capacity.")
    accounting = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": cleanup.sequence,
                "reserved_operations": anchor.reserved_operations - 1,
                "reserved_events": anchor.reserved_events - 1,
                "reserved_bytes": anchor.reserved_bytes - reservation,
                "retained_bytes": anchor.retained_bytes + charge,
            }
        ),
        redactor=redactor,
    )
    require_capacity(accounting, ordinary=False)
    await tx.put("operations", operation_key(operation), cleanup, insert=True)
    await tx.put("operations", operation_key(command.operation), updated, insert=False)
    await tx.put("anchors", (), accounting, insert=False)
    return cleanup
