"""Terminal clarification material reclaimed by the namespace transaction owner."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING

from cayu.collaboration._clarification_records import ClarificationLineageRecord
from cayu.collaboration._clarification_state import (
    ClarificationQuestionState,
    clarification_commitment,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.clarifications import MAX_CLARIFICATION_QUESTIONS
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestSnapshot
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.collaboration.base import _Repository


async def question_pruning_material(
    tx: _Repository, snapshot: RequestSnapshot, redactor: SecretRedactor
) -> tuple[tuple[ClarificationQuestionState, ...], str]:
    request = snapshot.receipt.expected.intent.selection.reference
    records = tuple(
        sorted(
            (
                prepare_contract(ClarificationQuestionState, raw, redactor=redactor)
                for raw in await tx.scan_clarification_questions(
                    request, limit=MAX_CLARIFICATION_QUESTIONS + 1
                )
            ),
            key=lambda record: record.question.generation,
        )
    )
    if len(records) != snapshot.clarification.generation:
        raise CollaborationUnavailable("Clarification pruning frontier is incomplete.")
    for index, record in enumerate(records, start=1):
        if (
            record.state == "open"
            or record.question.request != request
            or record.question.generation != index
            or record.question.lineage != snapshot.clarification.lineage
        ):
            raise CollaborationUnavailable("Clarification pruning requires exact terminal history.")
        if record.input is not None:
            require_exact_contract(
                record.input,
                prepare_contract(
                    type(record.input),
                    await tx.get(
                        "clarification_inputs",
                        (request.request_id, request.incarnation, record.input.revision),
                    ),
                    redactor=redactor,
                ),
                redactor=redactor,
            )
    # Filter in the native indexed query, not after a page of settled records:
    # an arbitrarily later unresolved handoff must still fence reclamation.
    for family in ("clarification_services", "clarification_deliveries"):
        if await tx.scan_clarification_request_handoffs(
            request, family=family, pending_only=True, limit=1
        ):
            raise CollaborationUnavailable("Clarification handoff still owns request history.")
    # Hash fixed-width per-record commitments to avoid materializing an unbounded
    # aggregate envelope. Input values are embedded in each validated decision.
    digest = sha256(
        b"".join(bytes.fromhex(clarification_commitment(record, redactor)) for record in records)
    ).hexdigest()
    return records, digest


async def prune_question_material(
    tx: _Repository, records: tuple[ClarificationQuestionState, ...], redactor: SecretRedactor
) -> int:
    if not records:
        return 0
    lineage_operation = records[0].question.lineage
    lineage_key = operation_key(lineage_operation)
    lineage = prepare_contract(
        ClarificationLineageRecord,
        await tx.get("clarification_lineages", lineage_key),
        redactor=redactor,
    )
    released = 0
    for record in records:
        if (
            record.state == "open"
            or lineage.operation != record.question.lineage
            or lineage.policy != record.question.policy
            or lineage.budget_binding != record.question.budget_binding
            or lineage.budget_authority_sha256 != record.question.budget_authority_sha256
        ):
            raise CollaborationUnavailable("Clarification pruning lineage authority conflicts.")
        await tx.delete("clarification_questions", operation_key(record.question.operation))
        released += len(contract_bytes(record, redactor=redactor))
        if record.input is not None:
            await tx.delete(
                "clarification_inputs",
                (
                    record.input.request.request_id,
                    record.input.request.incarnation,
                    record.input.revision,
                ),
            )
            released += len(contract_bytes(record.input, redactor=redactor))
    if not await tx.scan_clarification_lineage_questions(lineage_operation, limit=1):
        if lineage.usage.pending:
            raise CollaborationUnavailable(
                "Last clarification still retains pending responsibility."
            )
        await tx.delete("clarification_lineages", lineage_key)
        released += len(contract_bytes(lineage, redactor=redactor))
    # Never decrement lifetime usage while another request shares the lineage.
    return released
