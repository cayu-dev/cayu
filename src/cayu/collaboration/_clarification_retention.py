"""Clarification responsibility participating in the existing retirement owner."""

from cayu.collaboration._clarification_commands import ClarificationOpenReceipt
from cayu.collaboration._clarification_deliveries import ClarificationDeliveryRecord
from cayu.collaboration._clarification_services import ClarificationServiceRecord
from cayu.collaboration._clarification_state import ClarificationQuestionState
from cayu.collaboration._contracts import OperationRef
from cayu.collaboration._permit_store import prepare_permit_record
from cayu.collaboration._permits import PermitExclusion
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.base import _stored_mode
from cayu.collaboration.clarifications import ClarificationQuestion
from cayu.collaboration.lifecycle import NamespaceRef
from cayu.collaboration.participants import CollaborationUnavailable


def _pins_namespace(
    question: ClarificationQuestion, namespace: NamespaceRef, operation: OperationRef
) -> bool:
    return any(
        (ref.namespace_incarnation, ref.generation)
        == (namespace.namespace_incarnation, namespace.generation)
        # The opening contract binds the question operation to the parent
        # request's namespace. RequestRef itself is an object identity, not an
        # operation or authorization tuple.
        for ref in (operation, question.operation, question.lineage)
    )


async def operation_retains_clarification(tx, raw, namespace, redactor):
    """Inspect exact current debt using the already bounded namespace scan.

    An initial pending receipt is not current debt; the typed native record is.
    Conversely, settling a permit does not erase the separate source service
    settlement obligation. No receiving I/O or authority callbacks run here.
    """
    mode = _stored_mode(raw)
    if mode == "clarification_open":
        opening = prepare_contract(ClarificationOpenReceipt, raw, redactor=redactor)
        question = opening.command.question
        if not _pins_namespace(question, namespace, question.operation):
            return False
        state = prepare_contract(
            ClarificationQuestionState,
            await tx.get("clarification_questions", operation_key(question.operation)),
            redactor=redactor,
        )
        require_exact_contract(question, state.question, redactor=redactor)
        return state.state == "open"
    if mode == "clarification_delivery":
        registration = prepare_contract(ClarificationDeliveryRecord, raw, redactor=redactor)
        intent = registration.intent
        if not _pins_namespace(intent.question, namespace, intent.operation):
            return False
        current = prepare_contract(
            ClarificationDeliveryRecord,
            await tx.get("clarification_deliveries", operation_key(intent.operation)),
            redactor=redactor,
        )
        require_exact_contract(intent, current.intent, redactor=redactor)
        if registration.state != "pending":
            raise CollaborationUnavailable("Delivery registration contradicts its retained owner.")
        return current.state == "pending"
    if mode == "permit":
        permit = prepare_permit_record(raw, redactor)
        expected = permit.expected
        if expected.intent.request.effect_scope != "clarification_service" or isinstance(
            permit, PermitExclusion
        ):
            return False
        operation = expected.intent.request.source_operation
        service = prepare_contract(
            ClarificationServiceRecord,
            await tx.get("clarification_services", operation_key(operation)),
            redactor=redactor,
        )
        require_exact_contract(expected, service.permit.expected, redactor=redactor)
        if service.dispatch.intent.operation != operation:
            raise CollaborationUnavailable(
                "Service responsibility conflicts with its permit owner."
            )
        return service.state == "pending" and _pins_namespace(
            service.dispatch.intent.question, namespace, operation
        )
    return False


async def prune_delivery_record(tx, raw, redactor) -> int:
    """Remove one settled delivery bundle inside authorized namespace pruning."""
    registration = prepare_contract(ClarificationDeliveryRecord, raw, redactor=redactor)
    key = operation_key(registration.intent.operation)
    current = prepare_contract(
        ClarificationDeliveryRecord,
        await tx.get("clarification_deliveries", key),
        redactor=redactor,
    )
    require_exact_contract(registration.intent, current.intent, redactor=redactor)
    if registration.state != "pending" or current.state != "settled":
        raise CollaborationUnavailable("Delivery pruning requires positive receiving settlement.")
    released = len(contract_bytes(registration, redactor=redactor)) + len(
        contract_bytes(current, redactor=redactor)
    )
    await tx.delete("clarification_deliveries", key)
    await tx.delete("operations", key)
    return released


async def prune_service_record(tx, registered, settled, redactor) -> int:
    """Service accounting may leave only with its exact settled permit bundle."""
    expected = registered.expected
    if expected.intent.request.effect_scope != "clarification_service":
        return 0
    operation = expected.intent.request.source_operation
    key = operation_key(operation)
    current = prepare_contract(
        ClarificationServiceRecord,
        await tx.get("clarification_services", key),
        redactor=redactor,
    )
    require_exact_contract(registered, current.permit, redactor=redactor)
    if (
        current.dispatch.intent.operation != operation
        or current.state != "settled"
        or current.settlement != settled.receiving_receipt
    ):
        raise CollaborationUnavailable(
            "Service pruning requires exact source and permit settlement."
        )
    released = len(contract_bytes(current, redactor=redactor))
    await tx.delete("clarification_services", key)
    return released
