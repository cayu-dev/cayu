"""Owner-clock expiry using the existing question decision transaction."""

from cayu.collaboration._clarification_commands import (
    ClarificationCloseCommand,
    ClarificationOpenReceipt,
)
from cayu.collaboration._clarification_recovery_types import (
    ClarificationDueQuestion,
    ClarificationDueQuestionPage,
    ClarificationExpiryReceipt,
    ClarificationExpiryRequest,
    ClarificationPendingServiceQuery,
    ClarificationQuestionRecovery,
)
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._clarification_store import (
    close_in_transaction,
    discover_due_in_transaction,
)
from cayu.collaboration._contracts import CollaborationConflict, InitiatorBinding
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.access import CollaborationAccessContext
from cayu.collaboration.base import REQUEST_FAMILY
from cayu.collaboration.participants import CollaborationUnavailable


def _authorize(coordinator, context, *, mutation):
    participants = coordinator.requests._participants
    redactor = coordinator.requests._redactor
    context = prepare_contract(CollaborationAccessContext, context, redactor=redactor)
    store, initialized = participants._ready()
    participants._capability(store, initialized, mutation=mutation, family=REQUEST_FAMILY)
    _, grant = participants._authorize(
        context, "request_control" if mutation else "request_readback"
    )
    participants._require_refs(grant, (), create=True)
    return participants, store, initialized, redactor, context


async def due_questions(coordinator, query, *, context):
    participants, store, initialized, redactor, _ = _authorize(coordinator, context, mutation=False)
    query = prepare_contract(ClarificationPendingServiceQuery, query, redactor=redactor)
    if query.cursor is not None and (
        query.cursor.operation.application_scope != initialized.binding.application_scope
        or query.cursor.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Question recovery cursor belongs to another owner.")
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        openings = await discover_due_in_transaction(
            store, tx, initialized, after=query.cursor, limit=query.limit, redactor=redactor
        )
    items = tuple(
        ClarificationDueQuestion(
            recovery=ClarificationQuestionRecovery(
                operation=opening.command.operation,
                question_sha256=clarification_commitment(opening.command.question, redactor),
                request_sha256=clarification_commitment(opening.command.expected, redactor),
            ),
            deadline_at_ms=opening.command.question.deadline_at_ms,
        )
        for opening in openings
    )
    return participants._page(
        ClarificationDueQuestionPage,
        "items",
        items,
        lambda item: {
            "deadline_at_ms": item.deadline_at_ms,
            "operation": item.recovery.operation.model_dump(mode="json"),
        },
        query.limit,
    )


async def expire_question(coordinator, request, *, context):
    _, store, initialized, redactor, context = _authorize(coordinator, context, mutation=True)
    request = prepare_contract(ClarificationExpiryRequest, request, redactor=redactor)
    recovery = request.recovery
    if (
        recovery.operation.application_scope != initialized.binding.application_scope
        or recovery.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Question recovery belongs to another owner.")
    async with store._transaction(initialized.binding.application_scope, write=True) as tx:
        raw = await tx.get("operations", operation_key(recovery.operation))
        if raw is None:
            raise CollaborationUnavailable("Original question evidence is unavailable.")
        opening = prepare_contract(ClarificationOpenReceipt, raw, redactor=redactor)
        if (
            opening.command.operation != recovery.operation
            or clarification_commitment(opening.command.question, redactor)
            != recovery.question_sha256
            or clarification_commitment(opening.command.expected, redactor)
            != recovery.request_sha256
        ):
            raise CollaborationConflict(
                "Question recovery does not match the complete original intent."
            )
        command = ClarificationCloseCommand(
            operation=request.operation,
            expected=opening.command.expected,
            question=recovery.operation,
            question_sha256=recovery.question_sha256,
            kind="expired",
            initiator=InitiatorBinding(
                issuer=initialized.owner,
                principal=context.principal,
                participant=None,
                mandate=None,
                invocation_id=None,
                interaction_id=None,
            ),
        )
        # The existing owner checks deadline against native time, authenticates
        # the complete question/parent frontier, and elects against replies.
        # Exact replay precedes current deadline checks. This settles no handoff.
        receipt = await close_in_transaction(store, tx, initialized, command, redactor=redactor)
        return prepare_contract(
            ClarificationExpiryReceipt,
            {
                "expected": request,
                "event_sequence": receipt.event.sequence,
                "expired_at_ms": receipt.decision.closed_at_ms,
            },
            redactor=redactor,
        )
