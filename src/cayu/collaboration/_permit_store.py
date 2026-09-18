"""Retained source responsibility and authenticated receiving-owner settlement."""

from __future__ import annotations

from typing import cast
from uuid import uuid4

from cayu.collaboration._capacity import PERMIT_SETTLEMENT_BYTES, require_capacity
from cayu.collaboration._contracts import (
    MAX_ID_BYTES,
    CollaborationConflict,
    CollaborationContractError,
    ContractValue,
    ExactLookup,
    ExactMatch,
)
from cayu.collaboration._namespace_store import load_namespace, require_open_namespace
from cayu.collaboration._participant_state import ParticipantPermitState
from cayu.collaboration._permits import (
    PermitCommand,
    PermitExclusion,
    PermitReceipt,
    PermitSettlement,
    PermitSettlementReader,
    PermitSnapshot,
    ReceivingSettlementReceipt,
    ReservedPermitSettlement,
    RetiredPermitExclusion,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration.base import CollaborationStore, _Anchor, _key, _Repository
from cayu.collaboration.lifecycle import (
    NamespaceRef,
    NamespaceRetirementEvidence,
    NamespaceSnapshot,
)
from cayu.collaboration.participants import (
    CollaborationInitialization,
    CollaborationUnavailable,
    ParticipantEvent,
)
from cayu.vaults.redaction import SecretRedactor


class _ReceivingReadback(ContractValue):
    result: ExactLookup[ReceivingSettlementReceipt]


def prepare_permit(
    initialized: CollaborationInitialization,
    expected: PermitCommand,
    redactor: SecretRedactor,
) -> PermitCommand:
    expected = prepare_contract(PermitCommand, expected, redactor=redactor)
    if (
        expected.source != initialized.owner
        or expected.operation.application_scope != initialized.binding.application_scope
        or expected.operation.namespace_incarnation != initialized.namespace_incarnation
        or expected.intent.limits != initialized.binding.limits
    ):
        raise CollaborationConflict("Permit command conflicts with initialized authority.")
    # Admission must cover individual envelopes as well as aggregate storage.
    # A valid one-byte control character expands to six JSON bytes; plain ASCII
    # is not a worst-case identifier. These probes are not authority or evidence.
    probe_redactor = SecretRedactor()
    maximum = prepare_contract(
        ReceivingSettlementReceipt,
        {
            "expected": expected,
            "receiving_owner": expected.intent.request.target.owner,
            "receipt_id": "\x01" * MAX_ID_BYTES,
            "outcome": "quiescent",
        },
        redactor=probe_redactor,
    )
    prepare_contract(
        PermitSnapshot,
        {
            "expected": expected,
            "position": 2**53 - 1,
            "state": "settled",
            "settlement": maximum,
        },
        redactor=probe_redactor,
    )
    event = prepare_contract(
        ParticipantEvent,
        {
            "id": "f" * 32,  # Runtime-generated UUID hex, not a receiver identifier.
            "sequence": 2**53 - 1,
            "operation": expected.intent.request.settlement_operation,
            "type": "permit_settled",
            "participants": (expected.intent.request.participant,),
        },
        redactor=probe_redactor,
    )
    prepare_contract(
        PermitSettlement,
        {"expected": expected, "receiving_receipt": maximum, "event": event},
        redactor=probe_redactor,
    )
    if expected.intent.request.required_settlement == "exclusion":
        prepare_contract(
            PermitExclusion,
            {
                "expected": expected,
                "receiving_receipt": maximum.model_copy(update={"outcome": "excluded"}),
                "event": event.model_copy(
                    update={
                        "operation": expected.operation,
                        "type": "permit_excluded",
                    }
                ),
            },
            redactor=probe_redactor,
        )
    prepare_contract(
        _ReceivingReadback,
        {"result": {"status": "match", "receipt": maximum}},
        redactor=probe_redactor,
    )
    return expected


def prepare_permit_record(
    raw: object, redactor: SecretRedactor
) -> PermitReceipt | ReservedPermitSettlement | PermitSettlement | PermitExclusion:
    tag = cast("dict[object, object]", raw).get("record_type") if isinstance(raw, dict) else None
    if tag == "permit_registered":
        return prepare_contract(PermitReceipt, raw, redactor=redactor)
    if tag == "permit_settlement_reserved":
        return prepare_contract(ReservedPermitSettlement, raw, redactor=redactor)
    if tag == "permit_settled":
        return prepare_contract(PermitSettlement, raw, redactor=redactor)
    if tag == "permit_excluded":
        return prepare_contract(PermitExclusion, raw, redactor=redactor)
    raise CollaborationContractError("Unknown permit operation evidence.")


async def require_event(tx: _Repository, event: ParticipantEvent, redactor: SecretRedactor) -> None:
    raw = await tx.get("events", (event.sequence,))
    if raw is None:
        raise CollaborationUnavailable("Permit event evidence is unavailable.")
    require_exact_contract(
        event, prepare_contract(ParticipantEvent, raw, redactor=redactor), redactor=redactor
    )


async def registered_receipt(
    tx: _Repository, expected: PermitCommand, redactor: SecretRedactor
) -> PermitReceipt | None:
    raw = await tx.get("operations", _key(expected))
    if raw is None:
        return None
    # A shared key occupied by any other family can never grant new admission.
    from cayu.collaboration.base import _stored_mode

    if _stored_mode(raw) != "permit":
        raise CollaborationConflict("Operation key already has different intent.")
    receipt = prepare_permit_record(raw, redactor)
    if not isinstance(receipt, PermitReceipt):
        raise CollaborationConflict("Operation key is reserved for another permit phase.")
    require_exact_contract(expected, receipt.expected, redactor=redactor)
    await require_event(tx, receipt.event, redactor)
    raw_snapshot = await tx.get("permits", _key(expected))
    if raw_snapshot is None:
        raise CollaborationUnavailable("Retained permit is unavailable.")
    snapshot = prepare_contract(PermitSnapshot, raw_snapshot, redactor=redactor)
    if snapshot.expected != expected or snapshot.position != receipt.position:
        raise CollaborationUnavailable("Retained permit conflicts with registration.")
    operation = expected.intent.request.settlement_operation
    retained = prepare_permit_record(
        await tx.get(
            "operations",
            (
                operation.namespace_incarnation,
                operation.generation,
                operation.caller_key,
            ),
        ),
        redactor,
    )
    require_exact_contract(expected, retained.expected, redactor=redactor)
    if snapshot.state == "pending":
        if not isinstance(retained, ReservedPermitSettlement):
            raise CollaborationUnavailable("Pending permit lacks its reserved settlement slot.")
    elif (
        not isinstance(retained, PermitSettlement)
        or retained.receiving_receipt != snapshot.settlement
    ):
        raise CollaborationUnavailable("Permit settlement representations conflict.")
    else:
        await require_event(tx, retained.event, redactor)
    return receipt


async def register_permit(
    store: CollaborationStore,
    initialized: CollaborationInitialization,
    expected: PermitCommand,
    redactor: SecretRedactor,
) -> PermitReceipt:
    async with store._transaction(initialized.binding.application_scope, write=True) as tx:
        return await register_permit_in_transaction(store, tx, initialized, expected, redactor)


async def register_permit_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    expected: PermitCommand,
    redactor: SecretRedactor,
) -> PermitReceipt:
    """Compose permit admission with the receiving owner's atomic mutation."""
    anchor = await store._anchor(tx, initialized, redactor)
    replay = await registered_receipt(tx, expected, redactor)
    if replay is not None:
        return replay
    namespace = await require_open_namespace(tx, anchor, expected.operation, redactor)
    request = expected.intent.request
    participant = await store._participant(tx, request.participant, initialized.owner, redactor)
    if (
        participant.lifecycle != "active"
        or participant.lifecycle_revision != request.expected_lifecycle_revision
        or participant.admission_generation != request.admission_generation
    ):
        raise CollaborationConflict("Participant no longer admits this permit authority.")
    settlement_key = (
        request.settlement_operation.namespace_incarnation,
        request.settlement_operation.generation,
        request.settlement_operation.caller_key,
    )
    if await tx.get("operations", settlement_key) is not None:
        raise CollaborationConflict("Settlement key already carries different responsibility.")
    current = await store._permit_state(tx, participant.reference, redactor)
    updated_permits = prepare_contract(
        ParticipantPermitState,
        current.model_copy(
            update={
                "issued_frontier": current.issued_frontier + 1,
                "outstanding": current.outstanding + 1,
            }
        ),
        redactor=redactor,
    )
    updated_namespace = prepare_contract(
        NamespaceSnapshot,
        namespace.model_copy(
            update={
                "outstanding_obligations": namespace.outstanding_obligations + 1,
            }
        ),
        redactor=redactor,
    )
    event = ParticipantEvent(
        id=uuid4().hex,
        sequence=anchor.event_sequence + 1,
        operation=expected.operation,
        type="permit_registered",
        participants=(participant.reference,),
    )
    receipt = prepare_contract(
        PermitReceipt,
        PermitReceipt(
            expected=expected,
            position=updated_permits.issued_frontier,
            event=event,
        ),
        redactor=redactor,
    )
    snapshot = prepare_contract(
        PermitSnapshot,
        PermitSnapshot(
            expected=expected,
            position=receipt.position,
            state="pending",
            settlement=None,
        ),
        redactor=redactor,
    )
    reserved = ReservedPermitSettlement(expected=expected)
    charge = sum(
        len(contract_bytes(v, redactor=redactor))
        for v in (
            receipt,
            snapshot,
            reserved,
            event,
            updated_permits,
            updated_namespace,
        )
    ) - sum(len(contract_bytes(v, redactor=redactor)) for v in (current, namespace))
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 2,
                "permit_count": anchor.permit_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "reserved_events": anchor.reserved_events + 1,
                "retained_bytes": anchor.retained_bytes + charge,
                "reserved_bytes": anchor.reserved_bytes + PERMIT_SETTLEMENT_BYTES,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=True)
    await tx.put("operations", _key(expected), receipt, insert=True)
    await tx.put("operations", settlement_key, reserved, insert=True)
    await tx.put("permits", _key(expected), snapshot, insert=True)
    await tx.put(
        "participant_permits",
        (participant.reference.participant_id,),
        updated_permits,
        insert=False,
    )
    await tx.put(
        "namespaces",
        (namespace.reference.namespace_incarnation, namespace.reference.generation),
        updated_namespace,
        insert=False,
    )
    await tx.put("events", (event.sequence,), event, insert=True)
    await tx.put("anchors", (), updated, insert=False)
    return receipt


async def exclude_permit(
    store: CollaborationStore,
    initialized: CollaborationInitialization,
    expected: PermitCommand,
    reader: PermitSettlementReader,
    redactor: SecretRedactor,
) -> PermitExclusion | PermitSettlement | RetiredPermitExclusion:
    """Fence a never-admitted registration, or settle the registration winner.

    The receiving owner's permanent exclusion precedes this operation. Its read
    happens outside our transaction; the competing registration is decided by
    the native transaction, not by an earlier not-found lookup.
    """
    if (
        not isinstance(reader, PermitSettlementReader)
        or reader.owner != expected.intent.request.target.owner
    ):
        raise CollaborationConflict("Exact trusted receiving-owner reader is required.")
    found = prepare_contract(
        _ReceivingReadback, {"result": await reader.lookup(expected)}, redactor=redactor
    ).result
    if not isinstance(found, ExactMatch) or found.receipt.outcome != "excluded":
        raise CollaborationUnavailable("Positive receiving exclusion is unavailable.")
    require_exact_contract(expected, found.receipt.expected, redactor=redactor)
    async with store._transaction(initialized.binding.application_scope, write=True) as tx:
        anchor = await store._anchor(tx, initialized, redactor)
        raw = await tx.get("operations", _key(expected))
        if raw is not None:
            from cayu.collaboration.base import _stored_mode

            if _stored_mode(raw) != "permit":
                raise CollaborationConflict("Operation key already has different intent.")
            retained = prepare_permit_record(raw, redactor)
            require_exact_contract(expected, retained.expected, redactor=redactor)
            if isinstance(retained, PermitExclusion):
                await require_event(tx, retained.event, redactor)
                return retained
            # Fully validate all existing registration/settlement representations.
            await registered_receipt(tx, expected, redactor)
        else:
            if expected.operation.generation <= anchor.retired_through:
                # Retirement is permanent admission rejection, including after
                # exact history pruning. It is not fabricated receipt matching.
                content = "pruned"
                if expected.operation.generation > anchor.pruned_through:
                    content = (
                        await load_namespace(tx, anchor, expected.operation.generation, redactor)
                    ).content
                return prepare_contract(
                    RetiredPermitExclusion,
                    {
                        "expected": expected,
                        "receiving_receipt": found.receipt,
                        "retirement": NamespaceRetirementEvidence(
                            namespace=NamespaceRef(
                                owner=initialized.owner,
                                namespace_incarnation=initialized.namespace_incarnation,
                                generation=expected.operation.generation,
                            ),
                            retired_through=anchor.retired_through,
                            pruned_through=anchor.pruned_through,
                            content=content,
                        ),
                    },
                    redactor=redactor,
                )
            namespace = await load_namespace(tx, anchor, expected.operation.generation, redactor)
            if namespace.state == "retired":
                raise CollaborationUnavailable(
                    "Retired namespace cannot retain new exclusion evidence."
                )
            await store._participant(
                tx, expected.intent.request.participant, initialized.owner, redactor
            )
            event = prepare_contract(
                ParticipantEvent,
                {
                    "id": uuid4().hex,
                    "sequence": anchor.event_sequence + 1,
                    "operation": expected.operation,
                    "type": "permit_excluded",
                    "participants": (expected.intent.request.participant,),
                },
                redactor=redactor,
            )
            excluded = prepare_contract(
                PermitExclusion,
                {"expected": expected, "receiving_receipt": found.receipt, "event": event},
                redactor=redactor,
            )
            updated = prepare_contract(
                _Anchor,
                anchor.model_copy(
                    update={
                        "operation_count": anchor.operation_count + 1,
                        "event_count": anchor.event_count + 1,
                        "event_sequence": event.sequence,
                        "retained_bytes": anchor.retained_bytes
                        + len(contract_bytes(excluded, redactor=redactor))
                        + len(contract_bytes(event, redactor=redactor)),
                    }
                ),
                redactor=redactor,
            )
            # A never-admitted command has no reserved settlement slot. Its new
            # negative record is ordinary admission; preserve maintenance capacity
            # so a full namespace can still retire and prove permanent rejection.
            require_capacity(updated, ordinary=True)
            await tx.put("operations", _key(expected), excluded, insert=True)
            await tx.put("events", (event.sequence,), event, insert=True)
            await tx.put("anchors", (), updated, insert=False)
            return excluded
    # Registration won the transaction. Its pre-reserved settlement capacity and
    # exact receiving readback now discharge that obligation through the usual path.
    return await settle_permit(store, initialized, expected, reader, redactor)


async def settle_permit(
    store: CollaborationStore,
    initialized: CollaborationInitialization,
    expected: PermitCommand,
    reader: PermitSettlementReader,
    redactor: SecretRedactor,
) -> PermitSettlement:
    request = expected.intent.request
    settlement_key = (
        request.settlement_operation.namespace_incarnation,
        request.settlement_operation.generation,
        request.settlement_operation.caller_key,
    )
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        await store._anchor(tx, initialized, redactor)
        if await registered_receipt(tx, expected, redactor) is None:
            raise CollaborationUnavailable("Permit registration is unavailable.")
        reserved = prepare_permit_record(await tx.get("operations", settlement_key), redactor)
        require_exact_contract(expected, reserved.expected, redactor=redactor)
        if isinstance(reserved, PermitSettlement):
            await require_event(tx, reserved.event, redactor)
            return reserved
        if not isinstance(reserved, ReservedPermitSettlement):
            raise CollaborationConflict("Settlement key has another operation kind.")
    if not isinstance(reader, PermitSettlementReader) or reader.owner != request.target.owner:
        raise CollaborationConflict("Exact trusted receiving-owner reader is required.")
    # This is the only external read. No source transaction is held while awaiting it.
    found = prepare_contract(
        _ReceivingReadback,
        {
            "result": await reader.lookup(expected),
        },
        redactor=redactor,
    ).result
    if not isinstance(found, ExactMatch):
        raise CollaborationUnavailable("Positive exact receiving settlement is unavailable.")
    require_exact_contract(expected, found.receipt.expected, redactor=redactor)
    async with store._transaction(initialized.binding.application_scope, write=True) as tx:
        return await settle_permit_in_transaction(
            store, tx, initialized, expected, found.receipt, redactor
        )


async def settle_permit_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    expected: PermitCommand,
    receiving: ReceivingSettlementReceipt,
    redactor: SecretRedactor,
) -> PermitSettlement:
    """Commit already-authenticated receiver evidence with its local terminal mutation.

    Callers must authenticate receiving evidence before entering this private seam;
    constructing ReceivingSettlementReceipt alone is never receiving authority.
    """
    receiving = prepare_contract(ReceivingSettlementReceipt, receiving, redactor=redactor)
    require_exact_contract(expected, receiving.expected, redactor=redactor)
    request = expected.intent.request
    operation = request.settlement_operation
    settlement_key = (operation.namespace_incarnation, operation.generation, operation.caller_key)
    anchor = await store._anchor(tx, initialized, redactor)
    if await registered_receipt(tx, expected, redactor) is None:
        raise CollaborationUnavailable("Permit registration is unavailable.")
    reserved = prepare_permit_record(await tx.get("operations", settlement_key), redactor)
    require_exact_contract(expected, reserved.expected, redactor=redactor)
    if isinstance(reserved, PermitSettlement):
        await require_event(tx, reserved.event, redactor)
        return reserved
    if not isinstance(reserved, ReservedPermitSettlement):
        raise CollaborationConflict("Settlement key has another operation kind.")
    prior = prepare_contract(
        PermitSnapshot, await tx.get("permits", _key(expected)), redactor=redactor
    )
    if prior.state != "pending":
        raise CollaborationUnavailable("Permit settlement representations conflict.")
    current = await store._permit_state(tx, request.participant, redactor)
    namespace = await load_namespace(tx, anchor, expected.operation.generation, redactor)
    if not current.outstanding or not namespace.outstanding_obligations:
        raise CollaborationUnavailable("Permit accounting lacks retained responsibility.")
    event = ParticipantEvent(
        id=uuid4().hex,
        sequence=anchor.event_sequence + 1,
        operation=request.settlement_operation,
        type="permit_settled",
        participants=(request.participant,),
    )
    receipt = prepare_contract(
        PermitSettlement,
        PermitSettlement(
            expected=expected,
            receiving_receipt=receiving,
            event=event,
        ),
        redactor=redactor,
    )
    snapshot = prepare_contract(
        PermitSnapshot,
        prior.model_copy(
            update={
                "state": "settled",
                "settlement": receiving,
            }
        ),
        redactor=redactor,
    )
    updated_permits = prepare_contract(
        ParticipantPermitState,
        current.model_copy(
            update={
                "outstanding": current.outstanding - 1,
            }
        ),
        redactor=redactor,
    )
    updated_namespace = prepare_contract(
        NamespaceSnapshot,
        namespace.model_copy(
            update={
                "outstanding_obligations": namespace.outstanding_obligations - 1,
            }
        ),
        redactor=redactor,
    )
    charge = sum(
        len(contract_bytes(v, redactor=redactor))
        for v in (
            receipt,
            snapshot,
            event,
            updated_permits,
            updated_namespace,
        )
    ) - sum(
        len(contract_bytes(v, redactor=redactor)) for v in (reserved, prior, current, namespace)
    )
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "reserved_events": anchor.reserved_events - 1,
                "retained_bytes": anchor.retained_bytes + charge,
                "reserved_bytes": anchor.reserved_bytes - PERMIT_SETTLEMENT_BYTES,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=False)
    await tx.put("operations", settlement_key, receipt, insert=False)
    await tx.put("permits", _key(expected), snapshot, insert=False)
    await tx.put(
        "participant_permits",
        (request.participant.participant_id,),
        updated_permits,
        insert=False,
    )
    await tx.put(
        "namespaces",
        (namespace.reference.namespace_incarnation, namespace.reference.generation),
        updated_namespace,
        insert=False,
    )
    await tx.put("events", (event.sequence,), event, insert=True)
    await tx.put("anchors", (), updated, insert=False)
    return receipt
