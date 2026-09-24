"""Lifecycle evidence composed with the existing atomic request admission.

This permit owns only the local admission mutation. It is registered and settled
in the same transaction, never exported as a provider/resource execution grant.
"""

from __future__ import annotations

from hashlib import sha256

from cayu.collaboration._contracts import (
    CollaborationConflict,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ObjectRef,
)
from cayu.collaboration._permit_store import (
    prepare_permit,
    prepare_permit_record,
    register_permit_in_transaction,
    registered_receipt,
    require_event,
    settle_permit_in_transaction,
)
from cayu.collaboration._permits import (
    PermitCommand,
    PermitIntent,
    PermitReceipt,
    PermitRegistration,
    PermitSettlement,
    PermitSnapshot,
    ReceivingSettlementReceipt,
    ReservedPermitSettlement,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration.base import CollaborationStore, _key, _Repository, _stored_mode
from cayu.collaboration.participants import CollaborationInitialization, CollaborationUnavailable
from cayu.collaboration.prepared_admission import require_secret_free_prepared
from cayu.collaboration.requests import RequestAdmissionCommand, RequestAdmissionReceipt
from cayu.vaults.redaction import SecretRedactor


def _receiving_id(command: RequestAdmissionCommand, redactor: SecretRedactor) -> str:
    return (
        "prepared-admission:"
        + sha256(contract_bytes(command.operation, redactor=redactor)).hexdigest()
    )


async def register_prepared_admission(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    command: RequestAdmissionCommand,
    *,
    redactor: SecretRedactor,
) -> PermitReceipt:
    """Called only by request arbitration inside its receiving-authorized transaction."""
    prepared = command.prepared
    if prepared is None:
        raise CollaborationUnavailable("Prepared admission evidence is required.")
    identity = _receiving_id(command, redactor)
    operation = command.operation.model_copy(update={"caller_key": identity + ":permit"})
    settlement = command.operation.model_copy(update={"caller_key": identity + ":settled"})
    reference = command.expected.intent.selection.reference
    registration = PermitRegistration(
        operation=operation,
        participant=prepared.recipient,
        expected_lifecycle_revision=prepared.lifecycle_revision,
        expected_configuration_revision=prepared.configuration_revision,
        admission_generation=prepared.admission_generation,
        admission_commitment=sha256(contract_bytes(command, redactor=redactor)).hexdigest(),
        source_operation=command.operation,
        target=ObjectRef(
            owner=reference.owner,
            kind="collaboration_request",
            object_id=reference.request_id,
            incarnation=reference.incarnation,
        ),
        target_state="existing",
        effect_scope="request_prepared_admission",
        required_settlement="quiescence",
        settlement_operation=settlement,
    )
    expected = prepare_permit(
        initialized,
        PermitCommand(
            operation=operation,
            source=initialized.owner,
            destination=initialized.owner,
            initiator=command.initiator,
            intent=PermitIntent(request=registration, limits=initialized.binding.limits),
        ),
        redactor,
    )
    receipt = await register_permit_in_transaction(store, tx, initialized, expected, redactor)
    # The same transaction either commits the request receipt as well, or rolls
    # back both operations. This is not settlement of a foreign child or producer.
    await settle_permit_in_transaction(
        store,
        tx,
        initialized,
        expected,
        ReceivingSettlementReceipt(
            expected=expected,
            receiving_owner=initialized.owner,
            receipt_id=identity,
            outcome="quiescent",
        ),
        redactor,
    )
    return receipt


async def require_prepared_admission_evidence(
    tx: _Repository, receipt: RequestAdmissionReceipt, *, redactor: SecretRedactor
) -> None:
    """Authenticate durable registration, settlement, snapshot and both events."""
    permit = receipt.admission_permit
    if receipt.command.prepared is None:
        return
    require_secret_free_prepared(receipt.command.prepared, redactor)
    if permit is None:
        raise CollaborationUnavailable("Prepared admission lifecycle evidence is missing.")
    expected = permit.expected
    retained = await registered_receipt(tx, expected, redactor)
    if retained is None:
        raise CollaborationUnavailable("Prepared admission permit is missing.")
    require_exact_contract(permit, retained, redactor=redactor)
    settlement = prepare_contract(
        PermitSettlement,
        await tx.get(
            "operations",
            (
                expected.intent.request.settlement_operation.namespace_incarnation,
                expected.intent.request.settlement_operation.generation,
                expected.intent.request.settlement_operation.caller_key,
            ),
        ),
        redactor=redactor,
    )
    require_exact_contract(expected, settlement.expected, redactor=redactor)
    expected_receiving = ReceivingSettlementReceipt(
        expected=expected,
        receiving_owner=expected.destination,
        receipt_id=_receiving_id(receipt.command, redactor),
        outcome="quiescent",
    )
    require_exact_contract(expected_receiving, settlement.receiving_receipt, redactor=redactor)
    await require_event(tx, settlement.event, redactor)
    snapshot = prepare_contract(
        PermitSnapshot, await tx.get("permits", _key(expected)), redactor=redactor
    )
    require_exact_contract(expected, snapshot.expected, redactor=redactor)
    if (
        snapshot.state != "settled"
        or snapshot.position != permit.position
        or snapshot.settlement != expected_receiving
        or not permit.event.sequence < settlement.event.sequence < receipt.event.sequence
    ):
        raise CollaborationUnavailable("Prepared admission settlement is inconsistent.")


async def lookup_admission_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    expected: RequestAdmissionCommand,
    *,
    redactor: SecretRedactor,
) -> ExactLookup[RequestAdmissionReceipt]:
    """Read-only exact lookup; authorization is held by the request facade."""
    from cayu.collaboration._request_receipts import request_receipt_metadata
    from cayu.collaboration._request_store import (
        operation_key,
        require_request_absence,
        require_request_event,
        retained_request,
    )
    from cayu.collaboration.requests import RequestControlReceipt, RequestReceipt

    prior = await retained_request(
        store,
        tx,
        initialized,
        expected.expected.intent.request,
        expected.expected.initiator,
        redactor,
    )
    if prior is None:
        raise CollaborationUnavailable("Admission request acceptance is unavailable.")
    require_exact_contract(expected.expected, prior.receipt.expected, redactor=redactor)
    raw = await tx.get("operations", operation_key(expected.operation))
    if raw is None:
        await require_request_absence(store, tx, initialized, expected.operation, redactor)
        return ExactNotFound()
    if _stored_mode(raw) != "request_admission":
        # Validate known sibling evidence before classifying a key collision.
        metadata = request_receipt_metadata(raw, redactor=redactor)
        if metadata is not None:
            await require_request_event(tx, metadata.receipt.event, redactor)
        elif _stored_mode(raw) == "request":
            other = prepare_contract(RequestReceipt, raw, redactor=redactor)
            await require_request_event(tx, other.event, redactor)
        elif _stored_mode(raw) == "request_control":
            control = prepare_contract(RequestControlReceipt, raw, redactor=redactor)
            if expected.operation != control.expected.operation:
                raise CollaborationUnavailable("Control key contradicts its retained identity.")
            await require_request_event(tx, control.event, redactor)
        elif _stored_mode(raw) == "permit":
            from cayu.collaboration._request_receipts import record_operation

            other_permit = prepare_permit_record(raw, redactor)
            if isinstance(other_permit, ReservedPermitSettlement):
                raise CollaborationUnavailable(
                    "Admission key is reserved for unresolved responsibility."
                )
            await require_event(tx, other_permit.event, redactor)
            require_exact_contract(
                expected.operation, record_operation(raw, redactor=redactor), redactor=redactor
            )
        else:
            raise CollaborationUnavailable("Admission key contains unavailable evidence.")
        raise CollaborationConflict("Admission key belongs to another operation.")
    receipt = prepare_contract(RequestAdmissionReceipt, raw, redactor=redactor)
    require_exact_contract(expected, receipt.command, redactor=redactor)
    await require_request_event(tx, receipt.event, redactor)
    await require_prepared_admission_evidence(tx, receipt, redactor=redactor)
    return ExactMatch[RequestAdmissionReceipt](receipt=receipt)
