"""Native clarification record projections, not receiving authorization.

Every repository validates the same complete record before accepting its key.
Relational readers additionally compare the projected secondary-index columns.
"""

from __future__ import annotations

from pydantic import model_validator

from cayu.collaboration._clarification_deliveries import ClarificationDeliveryRecord
from cayu.collaboration._clarification_services import ClarificationServiceRecord
from cayu.collaboration._clarification_state import ClarificationQuestionState
from cayu.collaboration._contracts import (
    CollaborationContractError,
    ContractValue,
    ObjectRef,
    OperationRef,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.clarifications import (
    MAX_CLARIFICATION_PENDING,
    MAX_CLARIFICATION_QUESTIONS,
    ClarificationDueCursor,
    ClarificationInputRevision,
    ClarificationLineageUsage,
    ClarificationPolicy,
    Commitment,
)
from cayu.collaboration.requests import RequestRef
from cayu.vaults.redaction import SecretRedactor

CLARIFICATION_RECORD_FAMILIES = frozenset(
    {
        "request_pruning",
        "clarification_questions",
        "clarification_inputs",
        "clarification_lineages",
        "clarification_services",
        "clarification_deliveries",
    }
)


MAX_CLARIFICATION_RETENTION_BATCH = 64


def prepare_handoff_scan(
    family: object, request: object, limit: int, *, scope: str, pending_only: bool = False
):
    if (
        type(family) is not str
        or family not in {"clarification_services", "clarification_deliveries"}
        or type(limit) is not int
        or not 1 <= limit <= MAX_CLARIFICATION_RETENTION_BATCH
        or type(pending_only) is not bool
    ):
        raise CollaborationContractError("Handoff scan requires a bounded native family.")
    request = prepare_contract(RequestRef, request, redactor=SecretRedactor())
    if request.owner.application_scope != scope:
        raise CollaborationContractError("Handoff scan belongs to another scope.")
    schema = (
        ClarificationServiceRecord
        if family == "clarification_services"
        else ClarificationDeliveryRecord
    )
    return request, schema


def prepare_lineage_scan(lineage: object, limit: int, *, scope: str) -> OperationRef:
    if type(limit) is not int or not 1 <= limit <= MAX_CLARIFICATION_QUESTIONS + 1:
        raise CollaborationContractError("Lineage scan requires a bounded limit.")
    lineage = prepare_contract(OperationRef, lineage, redactor=SecretRedactor())
    if lineage.application_scope != scope:
        raise CollaborationContractError("Lineage scan belongs to another scope.")
    return lineage


def handoff_request(record: ClarificationServiceRecord | ClarificationDeliveryRecord) -> RequestRef:
    return (
        record.dispatch.intent.question.request
        if isinstance(record, ClarificationServiceRecord)
        else record.intent.question.request
    )


def prepare_question_scan(request: object, limit: int) -> RequestRef:
    """Bound materialization before reading documents, including a sentinel row."""
    if type(limit) is not int or not 1 <= limit <= MAX_CLARIFICATION_QUESTIONS + 1:
        raise CollaborationContractError("Clarification scan requires a bounded limit.")
    return prepare_contract(RequestRef, request, redactor=SecretRedactor())


def prepare_due_scan(
    *, scope: str, after: object, now_ms: int, limit: int
) -> ClarificationDueCursor | None:
    if (
        type(limit) is not int
        or not 1 <= limit <= MAX_CLARIFICATION_PENDING
        or type(now_ms) is not int
        or not 1 <= now_ms < 2**53
    ):
        raise CollaborationContractError(
            "Clarification due scan requires bounded owner-time inputs."
        )
    if after is None:
        return None
    cursor = prepare_contract(ClarificationDueCursor, after, redactor=SecretRedactor())
    if cursor.operation.application_scope != scope:
        raise CollaborationContractError("Clarification due cursor belongs to another scope.")
    return cursor


def due_cursor_key(cursor: ClarificationDueCursor) -> tuple[int, str, int, str]:
    operation = cursor.operation
    return (
        cursor.deadline_at_ms,
        operation.namespace_incarnation,
        operation.generation,
        operation.caller_key,
    )


class ClarificationLineageRecord(ContractValue):
    """Common-root capacity; children cannot substitute a fresh budget or policy."""

    operation: OperationRef
    root_request: RequestRef
    policy: ClarificationPolicy
    budget_binding: ObjectRef
    budget_authority_sha256: Commitment
    usage: ClarificationLineageUsage

    @model_validator(mode="after")
    def coherent(self) -> ClarificationLineageRecord:
        scope = self.operation.application_scope
        if any(
            owner.application_scope != scope
            for owner in (
                self.root_request.owner,
                self.policy.reference.owner,
                self.budget_binding.owner,
            )
        ):
            raise ValueError("Clarification lineage belongs to another application scope.")
        self.usage.require_within(self.policy)
        return self


def clarification_record_projection(
    family: str, value: object, *, scope: str, key: tuple[str | int, ...]
) -> tuple[ContractValue, tuple[str | int, ...]]:
    """Return detached validated material and its exact native index projection."""
    redactor = SecretRedactor()
    projection: tuple[str | int, ...] = ()
    record: ContractValue
    if family == "request_pruning":
        from cayu.collaboration._request_pruning import RequestPruningProgress

        record = prepare_contract(RequestPruningProgress, value, redactor=redactor)
        operation = record.operation
        expected_key = (operation.namespace_incarnation, operation.generation, operation.caller_key)
        expected_scope = operation.application_scope
    elif family == "clarification_questions":
        question = prepare_contract(ClarificationQuestionState, value, redactor=redactor)
        record = question
        operation = question.question.operation
        expected_key = (operation.namespace_incarnation, operation.generation, operation.caller_key)
        expected_scope = operation.application_scope
        projection = (
            question.question.request.request_id,
            question.question.request.incarnation,
            question.question.responder.participant_id,
            question.state,
            question.question.deadline_at_ms,
            question.question.lineage.namespace_incarnation,
            question.question.lineage.generation,
            question.question.lineage.caller_key,
        )
    elif family == "clarification_inputs":
        revision = prepare_contract(ClarificationInputRevision, value, redactor=redactor)
        record = revision
        expected_key = (
            revision.request.request_id,
            revision.request.incarnation,
            revision.revision,
        )
        expected_scope = revision.request.owner.application_scope
        if any(
            ref.application_scope != expected_scope for ref in (revision.question, revision.reply)
        ):
            raise CollaborationContractError("Clarification input scope conflicts.")
    elif family == "clarification_deliveries":
        delivery = prepare_contract(ClarificationDeliveryRecord, value, redactor=redactor)
        record = delivery
        operation = delivery.intent.operation
        expected_key = (operation.namespace_incarnation, operation.generation, operation.caller_key)
        expected_scope = operation.application_scope
        projection = (
            delivery.intent.recipient.participant_id,
            delivery.state,
            delivery.intent.append.attempt_key.deadline_at_ms,
            delivery.intent.question.request.request_id,
            delivery.intent.question.request.incarnation,
        )
    elif family == "clarification_services":
        service = prepare_contract(ClarificationServiceRecord, value, redactor=redactor)
        record = service
        operation = service.dispatch.intent.operation
        expected_key = (operation.namespace_incarnation, operation.generation, operation.caller_key)
        expected_scope = operation.application_scope
        projection = (
            service.dispatch.intent.question.responder.participant_id,
            service.state,
            service.dispatch.intent.question.deadline_at_ms,
            service.dispatch.intent.question.request.request_id,
            service.dispatch.intent.question.request.incarnation,
        )
    elif family == "clarification_lineages":
        lineage = prepare_contract(ClarificationLineageRecord, value, redactor=redactor)
        record = lineage
        operation = lineage.operation
        expected_key = (operation.namespace_incarnation, operation.generation, operation.caller_key)
        expected_scope = operation.application_scope
    else:
        raise CollaborationContractError("Unknown clarification record family.")
    if (
        expected_scope != scope
        or len(key) != len(expected_key)
        or any(type(a) is not type(b) or a != b for a, b in zip(key, expected_key, strict=True))
    ):
        raise CollaborationContractError("Clarification lookup index contradicts its record.")
    return record, projection
