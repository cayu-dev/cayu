"""Native handoff accounting; no export callbacks or receiving I/O in a transaction.

The registered coordinator supplies fresh held export authority on registration.
Settlement consumes exact native receiving readback, not a submitted receipt.
Question decisions, queue append and model exposure remain separate operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._clarification_commands import ClarificationOpenReceipt
from cayu.collaboration._clarification_deliveries import (
    ClarificationDeliveryIntent,
    ClarificationDeliveryRecord,
)
from cayu.collaboration._clarification_records import ClarificationLineageRecord
from cayu.collaboration._clarification_state import ClarificationQuestionState
from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES, CollaborationConflict
from cayu.collaboration._namespace_store import require_open_namespace
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import operation_key, require_request_event, retained_request
from cayu.collaboration.base import CollaborationStore, _Anchor, _Repository, _stored_mode
from cayu.collaboration.clarifications import ClarificationDueCursor
from cayu.collaboration.participants import CollaborationInitialization, CollaborationUnavailable
from cayu.collaboration.peer_content import PeerContentReceipt
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.sessions.base import SessionStore

DELIVERY_SETTLEMENT_BYTES = 2 * MAX_ENVELOPE_BYTES


async def discover_deliveries_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    *,
    after: ClarificationDueCursor | None,
    limit: int,
    redactor: SecretRedactor,
) -> tuple[ClarificationDeliveryRecord, ...]:
    """Discover unresolved debt after restart, including elapsed deadlines.

    The result is private recovery material. A current export-owner guard is
    still required before attempting append or disclosing the retained payload.
    """
    await store._anchor(tx, initialized, redactor)
    records = []
    for raw in await tx.scan_pending_clarification_deliveries(after=after, limit=limit):
        record = prepare_contract(ClarificationDeliveryRecord, raw, redactor=redactor)
        exact = await load_delivery_in_transaction(
            store, tx, initialized, record.intent, redactor=redactor
        )
        if exact != record or record.state != "pending":
            raise CollaborationUnavailable("Delivery discovery contradicts exact owner evidence.")
        records.append(record)
    return tuple(records)


async def load_delivery_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    intent: ClarificationDeliveryIntent,
    *,
    redactor: SecretRedactor,
) -> ClarificationDeliveryRecord | None:
    """Compare the complete expected intent before returning historical evidence."""
    intent = prepare_contract(ClarificationDeliveryIntent, intent, redactor=redactor)
    await store._anchor(tx, initialized, redactor)
    if (
        intent.operation.application_scope != initialized.binding.application_scope
        or intent.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Delivery belongs to another owner namespace.")
    key = operation_key(intent.operation)
    raw = await tx.get("clarification_deliveries", key)
    registration = await tx.get("operations", key)
    if registration is not None and _stored_mode(registration) != "clarification_delivery":
        raise CollaborationConflict("Delivery operation is already bound to another family.")
    if raw is None and registration is None:
        return None
    if raw is None or registration is None:
        raise CollaborationUnavailable("Delivery responsibility lost its registration.")
    original = prepare_contract(ClarificationDeliveryRecord, registration, redactor=redactor)
    current = prepare_contract(ClarificationDeliveryRecord, raw, redactor=redactor)
    require_exact_contract(intent, original.intent, redactor=redactor)
    if original.state != "pending" or current.intent != original.intent:
        raise CollaborationUnavailable("Delivery responsibility contradicts its registration.")
    return current


async def register_delivery_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    intent: ClarificationDeliveryIntent,
    *,
    authority_expires_at_ms: int,
    redactor: SecretRedactor,
) -> ClarificationDeliveryRecord:
    """Reserve before foreign append; historical replay never renews disclosure."""
    intent = prepare_contract(ClarificationDeliveryIntent, intent, redactor=redactor)
    prior = await load_delivery_in_transaction(store, tx, initialized, intent, redactor=redactor)
    if prior is not None:
        return prior
    anchor = await store._anchor(tx, initialized, redactor)
    await require_open_namespace(tx, anchor, intent.operation, redactor)
    now = await tx.now_ms()
    if (
        type(authority_expires_at_ms) is not int
        or not now < authority_expires_at_ms <= 2**53 - 1
        or now >= intent.append.attempt_key.deadline_at_ms
    ):
        raise CollaborationConflict("Delivery authority or deadline has expired.")
    question = intent.question
    opening = prepare_contract(
        ClarificationOpenReceipt,
        await tx.get("operations", operation_key(question.operation)),
        redactor=redactor,
    )
    require_exact_contract(question, opening.command.question, redactor=redactor)
    await require_request_event(tx, opening.event, redactor)
    expected = opening.command.expected
    parent = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    decision = prepare_contract(
        ClarificationQuestionState,
        await tx.get("clarification_questions", operation_key(question.operation)),
        redactor=redactor,
    )
    selected = expected.intent.selection
    sender, recipient = (
        (selected.recipient.reference, selected.sender.reference)
        if intent.reply is None
        else (selected.sender.reference, selected.recipient.reference)
    )
    if (
        decision.question != question
        or parent is None
        or parent.state != "open"
        or parent.receipt.expected != expected
        or (intent.reply is None and decision.state != "open")
        or (
            intent.reply is not None
            and (decision.state != "answered" or decision.reply != intent.reply)
        )
        or (intent.sender, intent.recipient) != (sender, recipient)
    ):
        raise CollaborationConflict("Delivery conflicts with its retained question decision.")
    for participant in (sender, recipient):
        current = await store._participant(tx, participant, initialized.owner, redactor)
        if current.lifecycle != "active":
            raise CollaborationConflict("Participant no longer admits new delivery.")
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
    ):
        raise CollaborationUnavailable("Delivery lineage authority conflicts.")
    updated_lineage = prepare_contract(
        ClarificationLineageRecord,
        lineage.model_copy(
            update={
                "usage": lineage.usage.model_copy(update={"pending": lineage.usage.pending + 1})
            }
        ),
        redactor=redactor,
    )
    record = ClarificationDeliveryRecord(intent=intent)
    # Validate representability before delivery. This size-only witness is never
    # stored or returned as evidence that the receiving queue accepted anything.
    prepare_contract(
        ClarificationDeliveryRecord,
        record.model_copy(
            update={
                "state": "settled",
                "receipt": PeerContentReceipt(
                    operation_key=intent.append.operation_key,
                    append_key=intent.append.append_key,
                    attempt_generation=intent.append.attempt_key.attempt_generation,
                    status="appended",
                    occurrence=intent.append.occurrence,
                    queue_id='"' * 512,
                    transcript_event_id='"' * 512,
                    target_session_id=intent.append.append_key.target_session_id,
                    target_session_instance_id=intent.append.append_key.target_session_instance_id,
                ),
            }
        ),
        redactor=redactor,
    )
    charge = (
        2 * len(contract_bytes(record, redactor=redactor))
        + len(contract_bytes(updated_lineage, redactor=redactor))
        - len(contract_bytes(lineage, redactor=redactor))
    )
    updated_anchor = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "retained_bytes": anchor.retained_bytes + charge,
                "reserved_bytes": anchor.reserved_bytes + DELIVERY_SETTLEMENT_BYTES,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated_anchor, ordinary=True)
    key = operation_key(intent.operation)
    await tx.put("operations", key, record, insert=True)
    await tx.put("clarification_deliveries", key, record, insert=True)
    await tx.put("clarification_lineages", lineage_key, updated_lineage, insert=False)
    await tx.put("anchors", (), updated_anchor, insert=False)
    return record


async def reconcile_delivery(
    store: CollaborationStore,
    initialized: CollaborationInitialization,
    intent: ClarificationDeliveryIntent,
    receiving_store: SessionStore,
    *,
    redactor: SecretRedactor,
) -> ClarificationDeliveryRecord:
    """Settle only exact receiving readback; unknown/cancelled reads stay pending.

    This private coordinator primitive is not a disclosure entrance. It never
    invokes append, creates a replacement attempt, or renews source authority.
    Receiving I/O finishes before the collaboration settlement transaction.
    """
    intent = prepare_contract(ClarificationDeliveryIntent, intent, redactor=redactor)
    scope = initialized.binding.application_scope
    async with store._transaction(scope, write=False) as tx:
        current = await load_delivery_in_transaction(
            store, tx, initialized, intent, redactor=redactor
        )
    if current is None:
        raise CollaborationUnavailable("Delivery was not durably registered.")
    if current.state == "settled":
        return current
    if receiving_store.peer_content_version != 1:
        raise CollaborationUnavailable("Receiving store lacks exact peer readback.")
    receipt = await receiving_store.read_peer_content_attempt(intent.append)
    if receipt is None:
        return current
    receipt = prepare_contract(PeerContentReceipt, receipt, redactor=redactor)
    if receipt.status == "pending":
        if (
            receipt.operation_key != intent.append.operation_key
            or receipt.append_key != intent.append.append_key
            or receipt.attempt_generation != intent.append.attempt_key.attempt_generation
        ):
            raise CollaborationUnavailable("Pending peer readback contradicts the exact attempt.")
        return current
    # replayed describes this read, not a different receiving outcome.
    receipt = receipt.model_copy(update={"replayed": False})
    terminal = prepare_contract(
        ClarificationDeliveryRecord,
        ClarificationDeliveryRecord(intent=intent, state="settled", receipt=receipt),
        redactor=redactor,
    )
    async with store._transaction(scope, write=True) as tx:
        current = await load_delivery_in_transaction(
            store, tx, initialized, intent, redactor=redactor
        )
        if current is None:
            raise CollaborationUnavailable("Delivery responsibility disappeared during readback.")
        if current.state == "settled":
            require_exact_contract(terminal, current, redactor=redactor)
            return current
        anchor = await store._anchor(tx, initialized, redactor)
        lineage_key = operation_key(intent.question.lineage)
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
        delta = (
            len(contract_bytes(terminal, redactor=redactor))
            - len(contract_bytes(current, redactor=redactor))
            + len(contract_bytes(updated_lineage, redactor=redactor))
            - len(contract_bytes(lineage, redactor=redactor))
        )
        if delta > DELIVERY_SETTLEMENT_BYTES:
            raise CollaborationUnavailable("Delivery settlement exceeds its reserved capacity.")
        updated_anchor = prepare_contract(
            _Anchor,
            anchor.model_copy(
                update={
                    "retained_bytes": anchor.retained_bytes + delta,
                    "reserved_bytes": anchor.reserved_bytes - DELIVERY_SETTLEMENT_BYTES,
                }
            ),
            redactor=redactor,
        )
        require_capacity(updated_anchor, ordinary=False)
        await tx.put(
            "clarification_deliveries", operation_key(intent.operation), terminal, insert=False
        )
        await tx.put("clarification_lineages", lineage_key, updated_lineage, insert=False)
        await tx.put("anchors", (), updated_anchor, insert=False)
        return terminal
