"""Atomic clarification service charging and existing-permit registration.

Only a registered coordinator may call this with authenticated source and runtime
preparation. The transaction never calls SessionStore or a provider.
"""

from __future__ import annotations

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._clarification_commands import ClarificationOpenReceipt
from cayu.collaboration._clarification_records import ClarificationLineageRecord
from cayu.collaboration._clarification_services import ClarificationServiceRecord
from cayu.collaboration._clarification_state import ClarificationQuestionState
from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES, CollaborationConflict
from cayu.collaboration._permit_store import (
    prepare_permit,
    register_permit_in_transaction,
    registered_receipt,
    require_event,
)
from cayu.collaboration._permits import (
    PermitCommand,
    PermitIntent,
    PermitRegistration,
    PermitSettlement,
    ReceivingSettlementReceipt,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import operation_key, retained_request
from cayu.collaboration.base import CollaborationStore, _Anchor, _Repository
from cayu.collaboration.clarifications import ClarificationDueCursor, ClarificationLineageUsage
from cayu.collaboration.participants import CollaborationInitialization, CollaborationUnavailable
from cayu.collaboration.waits import request_object_ref
from cayu.runtime._session_continuation import continuation_digest
from cayu.runtime._temporary_continuation import (
    TemporaryServiceDispatch,
    TemporaryServicePreparation,
)
from cayu.vaults.redaction import SecretRedactor

# The permit owner separately reserves its settlement. This reserve covers the
# service record's receiving receipt and lineage counter update, not extra work.
SERVICE_SETTLEMENT_BYTES = 2 * MAX_ENVELOPE_BYTES


async def register_runtime_service_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    dispatch: TemporaryServiceDispatch,
    *,
    redactor: SecretRedactor,
    authority_expires_at_ms: int | None = None,
) -> ClarificationServiceRecord:
    """Resolve permit revisions under the source owner, not from caller claims.

    The registered coordinator supplies authenticated runtime preparation and
    holds current disclosure authority. This helper is not a public grant API.
    Exact prior registration wins over later lifecycle changes on replay.
    """
    prepared = await prepare_runtime_service_in_transaction(
        store, tx, initialized, dispatch, redactor=redactor
    )
    return await register_service_in_transaction(
        store,
        tx,
        initialized,
        prepared.dispatch,
        prepared.permit,
        redactor=redactor,
        authority_expires_at_ms=authority_expires_at_ms,
    )


async def prepare_runtime_service_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    dispatch: TemporaryServiceDispatch,
    *,
    redactor: SecretRedactor,
) -> TemporaryServicePreparation:
    """Read the expected tuple without registering authority or charging work.

    Registration subsequently compares these exact revisions in its own write
    transaction. A later lifecycle/configuration change must not silently replace
    the prepared authority. Existing registered work retains its original tuple.
    """
    dispatch = prepare_contract(TemporaryServiceDispatch, dispatch, redactor=redactor)
    await store._anchor(tx, initialized, redactor)
    intent = dispatch.intent
    raw = await tx.get("clarification_services", operation_key(intent.operation))
    if raw is not None:
        retained = prepare_contract(ClarificationServiceRecord, raw, redactor=redactor)
        require_exact_contract(dispatch, retained.dispatch, redactor=redactor)
        if await registered_receipt(tx, retained.permit.expected, redactor) != retained.permit:
            raise CollaborationUnavailable("Prepared service lost its permit evidence.")
        return TemporaryServicePreparation(dispatch=dispatch, permit=retained.permit.expected)
    participant = await store._participant(
        tx, intent.question.responder, initialized.owner, redactor
    )
    identity = continuation_digest(intent.operation)
    registration = PermitRegistration(
        operation=intent.operation.model_copy(
            update={"caller_key": "clarification-service-permit:" + identity}
        ),
        participant=participant.reference,
        expected_lifecycle_revision=participant.lifecycle_revision,
        expected_configuration_revision=participant.configuration_revision,
        admission_generation=participant.admission_generation,
        admission_commitment=continuation_digest(dispatch),
        source_operation=intent.operation,
        target=intent.target,
        target_state="existing",
        effect_scope="clarification_service",
        required_settlement="quiescence",
        settlement_operation=intent.operation.model_copy(
            update={"caller_key": "clarification-service-settlement:" + identity}
        ),
    )
    expected = PermitCommand(
        operation=registration.operation,
        source=initialized.owner,
        destination=initialized.owner,
        initiator=intent.initiator,
        intent=PermitIntent(request=registration, limits=initialized.binding.limits),
    )
    return TemporaryServicePreparation(dispatch=dispatch, permit=expected)


async def discover_services_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    *,
    after: ClarificationDueCursor | None,
    limit: int,
    redactor: SecretRedactor,
) -> tuple[ClarificationServiceRecord, ...]:
    """Pending responsibilities survive question expiry and require exact settlement."""
    await store._anchor(tx, initialized, redactor)
    records = []
    for raw in await tx.scan_pending_clarification_services(after=after, limit=limit):
        record = prepare_contract(ClarificationServiceRecord, raw, redactor=redactor)
        receipt = await registered_receipt(tx, record.permit.expected, redactor)
        if receipt != record.permit or record.state != "pending":
            raise CollaborationUnavailable("Pending service lost its permit registration.")
        records.append(record)
    return tuple(records)


async def settle_service_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    dispatch: TemporaryServiceDispatch,
    permit: PermitCommand,
    *,
    redactor: SecretRedactor,
) -> ClarificationServiceRecord:
    """Consume only the existing permit owner's authenticated durable settlement."""
    dispatch = prepare_contract(TemporaryServiceDispatch, dispatch, redactor=redactor)
    permit = prepare_contract(PermitCommand, permit, redactor=redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    key = operation_key(dispatch.intent.operation)
    current = prepare_contract(
        ClarificationServiceRecord, await tx.get("clarification_services", key), redactor=redactor
    )
    require_exact_contract(dispatch, current.dispatch, redactor=redactor)
    require_exact_contract(permit, current.permit.expected, redactor=redactor)
    if await registered_receipt(tx, permit, redactor) != current.permit:
        raise CollaborationUnavailable("Service permit registration is unavailable.")
    settlement = prepare_contract(
        PermitSettlement,
        await tx.get("operations", operation_key(permit.intent.request.settlement_operation)),
        redactor=redactor,
    )
    require_exact_contract(permit, settlement.expected, redactor=redactor)
    await require_event(tx, settlement.event, redactor)
    if current.state == "settled":
        assert current.settlement is not None
        require_exact_contract(current.settlement, settlement.receiving_receipt, redactor=redactor)
        return current
    lineage_key = operation_key(dispatch.intent.question.lineage)
    lineage = prepare_contract(
        ClarificationLineageRecord,
        await tx.get("clarification_lineages", lineage_key),
        redactor=redactor,
    )
    updated_lineage = prepare_contract(
        ClarificationLineageRecord,
        lineage.model_copy(
            update={
                "usage": lineage.usage.model_copy(update={"pending": lineage.usage.pending - 1})
            }
        ),
        redactor=redactor,
    )
    result = prepare_contract(
        ClarificationServiceRecord,
        current.model_copy(update={"state": "settled", "settlement": settlement.receiving_receipt}),
        redactor=redactor,
    )
    delta = sum(
        len(contract_bytes(value, redactor=redactor)) for value in (result, updated_lineage)
    ) - sum(len(contract_bytes(value, redactor=redactor)) for value in (current, lineage))
    updated_anchor = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "retained_bytes": anchor.retained_bytes + delta,
                "reserved_bytes": anchor.reserved_bytes - SERVICE_SETTLEMENT_BYTES,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated_anchor, ordinary=False)
    await tx.put("clarification_services", key, result, insert=False)
    await tx.put("clarification_lineages", lineage_key, updated_lineage, insert=False)
    await tx.put("anchors", (), updated_anchor, insert=False)
    return result


async def register_service_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    dispatch: TemporaryServiceDispatch,
    permit: PermitCommand,
    *,
    redactor: SecretRedactor,
    authority_expires_at_ms: int | None = None,
) -> ClarificationServiceRecord:
    dispatch = prepare_contract(TemporaryServiceDispatch, dispatch, redactor=redactor)
    permit = prepare_permit(initialized, permit, redactor)
    await store._anchor(tx, initialized, redactor)
    key = operation_key(dispatch.intent.operation)
    raw = await tx.get("clarification_services", key)
    registered = await registered_receipt(tx, permit, redactor)
    if raw is not None:
        previous = prepare_contract(ClarificationServiceRecord, raw, redactor=redactor)
        require_exact_contract(dispatch, previous.dispatch, redactor=redactor)
        require_exact_contract(permit, previous.permit.expected, redactor=redactor)
        if registered != previous.permit:
            raise CollaborationUnavailable("Clarification service permit evidence is missing.")
        return previous
    if registered is not None:
        raise CollaborationUnavailable("Clarification service responsibility is missing.")
    if authority_expires_at_ms is not None and (
        type(authority_expires_at_ms) is not int
        or not await tx.now_ms() < authority_expires_at_ms <= 2**53 - 1
    ):
        raise CollaborationConflict("Service source authorization expired before registration.")
    question = dispatch.intent.question
    opening = prepare_contract(
        ClarificationOpenReceipt,
        await tx.get("operations", operation_key(question.operation)),
        redactor=redactor,
    )
    require_exact_contract(question, opening.command.question, redactor=redactor)
    expected = opening.command.expected
    parent = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    decision = prepare_contract(
        ClarificationQuestionState,
        await tx.get("clarification_questions", operation_key(question.operation)),
        redactor=redactor,
    )
    require_exact_contract(question, decision.question, redactor=redactor)
    if (
        parent is None
        or parent.state != "open"
        or decision.state != "open"
        or question.deadline_at_ms <= await tx.now_ms()
        or parent.clarification.input_revision != question.input_revision
    ):
        raise CollaborationConflict("Clarification no longer admits new service work.")
    lineage_key = operation_key(question.lineage)
    lineage = prepare_contract(
        ClarificationLineageRecord,
        await tx.get("clarification_lineages", lineage_key),
        redactor=redactor,
    )
    if (
        lineage.policy != question.policy
        or lineage.budget_binding != question.budget_binding
        or lineage.budget_authority_sha256 != question.budget_authority_sha256
        or request_object_ref(lineage.root_request) not in dispatch.intent.ticket.targets
    ):
        raise CollaborationUnavailable("Clarification service lineage authority conflicts.")
    usage = prepare_contract(
        ClarificationLineageUsage,
        lineage.usage.model_copy(
            update={
                "service_turns": lineage.usage.service_turns + 1,
                "pending": lineage.usage.pending + 1,
            }
        ),
        redactor=redactor,
    )
    updated_lineage = prepare_contract(
        ClarificationLineageRecord, lineage.model_copy(update={"usage": usage}), redactor=redactor
    )
    receipt = await register_permit_in_transaction(store, tx, initialized, permit, redactor)
    record = prepare_contract(
        ClarificationServiceRecord,
        ClarificationServiceRecord(dispatch=dispatch, permit=receipt),
        redactor=redactor,
    )
    # Size-only witness: reserve a representable terminal envelope before service
    # can run. This synthetic value is never published as receiving evidence.
    prepare_contract(
        ClarificationServiceRecord,
        record.model_copy(
            update={
                "state": "settled",
                "settlement": ReceivingSettlementReceipt(
                    expected=permit,
                    receiving_owner=permit.intent.request.target.owner,
                    receipt_id='"' * 512,
                    outcome="quiescent",
                    admission_excluded=True,
                ),
            }
        ),
        redactor=redactor,
    )
    # Re-read the anchor updated by the permit owner; never overwrite its charge.
    anchor = await store._anchor(tx, initialized, redactor)
    charge = (
        len(contract_bytes(record, redactor=redactor))
        + len(contract_bytes(updated_lineage, redactor=redactor))
        - len(contract_bytes(lineage, redactor=redactor))
    )
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "retained_bytes": anchor.retained_bytes + charge,
                "reserved_bytes": anchor.reserved_bytes + SERVICE_SETTLEMENT_BYTES,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=True)
    await tx.put("clarification_services", key, record, insert=True)
    await tx.put("clarification_lineages", lineage_key, updated_lineage, insert=False)
    await tx.put("anchors", (), updated, insert=False)
    return record
