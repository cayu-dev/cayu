"""Exact receipt propagation over real request-acceptance evidence."""

import pytest
from tests.core.test_clarification_contracts import fixture
from tests.core.test_collaboration_request_foundation import accept, setup

from cayu.collaboration._clarification_commands import (
    ClarificationOpenCommand,
    ClarificationOpenReceipt,
)
from cayu.collaboration._clarification_state import (
    ClarificationQuestionState,
    clarification_commitment,
)
from cayu.collaboration._contracts import CollaborationContractError, snapshot_input
from cayu.collaboration._history_references import history_references
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration._request_receipts import record_operation, request_receipt_metadata
from cayu.collaboration.clarifications import ClarificationQuestion
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.requests import RequestEvent
from cayu.vaults.redaction import SecretRedactor


def question_for(expected):
    """Only construct test intent; no producer/export authority is fabricated."""
    state, _ = fixture()
    owner = snapshot_input(expected.source)
    operation = expected.operation

    def replace(value):
        if isinstance(value, dict):
            if set(value) == {"application_scope", "owner_id", "incarnation"}:
                return owner.copy()
            if set(value) == {
                "application_scope",
                "namespace_incarnation",
                "generation",
                "caller_key",
            }:
                return dict(snapshot_input(operation), caller_key=value["caller_key"])
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [replace(item) for item in value]
        return value

    value = replace(snapshot_input(state.question))
    value.update(
        request=snapshot_input(expected.intent.selection.reference),
        request_sha256=clarification_commitment(expected, SecretRedactor()),
        input_sha256=clarification_commitment(expected, SecretRedactor()),
        responder=snapshot_input(expected.intent.selection.sender.reference),
        deadline_at_ms=expected.intent.selection.accepted_at_ms + 50_000,
    )
    return ClarificationQuestion.model_validate(value)


@pytest.mark.anyio
async def test_clarification_open_receipt_joins_existing_operation_and_history_contract():
    store = InMemoryCollaborationStore()
    try:
        values = await setup(store)
        accepted = (await accept(store, values)).receipt
        expected = accepted.expected
        question = question_for(expected)
        command = ClarificationOpenCommand(
            operation=question.operation, expected=expected, expected_revision=2, question=question
        )
        decision = ClarificationQuestionState(
            question=question,
            opened_at_ms=expected.intent.selection.accepted_at_ms + 1,
            state="open",
        )
        event = RequestEvent(
            id="opened",
            sequence=accepted.event.sequence + 1,
            operation=command.operation,
            request=question.request,
            type="clarification_opened",
            participants=accepted.event.participants,
        )
        receipt = ClarificationOpenReceipt(command=command, decision=decision, event=event)
        raw = snapshot_input(receipt)
        redactor = SecretRedactor()
        metadata = request_receipt_metadata(raw, redactor=redactor)
        assert metadata is not None
        assert metadata.receipt == receipt
        assert metadata.expected == expected
        assert record_operation(raw, redactor=redactor) == question.operation
        assert history_references(receipt) == history_references(accepted)
        for name, replacement in (
            ("operation", expected.operation),
            ("expected_revision", True),
            ("question", question.model_copy(update={"request_sha256": "0" * 64})),
            (
                "question",
                question.model_copy(
                    update={"responder": expected.intent.selection.recipient.reference}
                ),
            ),
        ):
            with pytest.raises(CollaborationContractError):
                prepare_contract(
                    ClarificationOpenCommand,
                    command.model_copy(update={name: replacement}),
                    redactor=redactor,
                )
        with pytest.raises(CollaborationContractError):
            prepare_contract(
                ClarificationOpenReceipt,
                receipt.model_copy(
                    update={"event": event.model_copy(update={"type": "request_accepted"})}
                ),
                redactor=redactor,
            )
    finally:
        await store.close()
