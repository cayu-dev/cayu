"""Durable bounded reclamation within the existing namespace retirement owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import Field, StrictInt, model_validator

from cayu.collaboration._contracts import ContractValue, OperationRef
from cayu.collaboration._history_references import history_references
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._request_receipts import request_receipt_metadata
from cayu.collaboration._request_store import operation_key, retained_request
from cayu.collaboration.clarifications import MAX_CLARIFICATION_QUESTIONS, Commitment
from cayu.collaboration.participants import CollaborationUnavailable, VersionOne
from cayu.collaboration.requests import (
    RequestCommand,
    RequestControlReceipt,
    RequestEvent,
    RequestReceipt,
    RequestRef,
    RequestSnapshot,
)
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.collaboration.base import CollaborationStore, _Anchor, _Repository

# RequestSnapshot's 64-event frontier plus one opening and one terminal
# decision for each bounded clarification question. Batch size remains 32.
MAX_REQUEST_PRUNING_EVENTS = 64 + 2 * MAX_CLARIFICATION_QUESTIONS


class RequestPruningProgress(ContractValue):
    """Receiving-owner cursor, not an executable request or a replay receipt."""

    schema_version: VersionOne = 1
    operation: OperationRef
    request: RequestRef
    snapshot_sha256: Commitment
    clarification_sha256: Commitment
    clarification_events: tuple[StrictInt, ...] = Field(max_length=2 * MAX_CLARIFICATION_QUESTIONS)
    next_event_index: StrictInt = Field(ge=1, lt=MAX_REQUEST_PRUNING_EVENTS)

    @model_validator(mode="after")
    def coherent(self) -> RequestPruningProgress:
        if self.request.owner.application_scope != self.operation.application_scope:
            raise ValueError("Request pruning belongs to another owner.")
        if (
            any(not 1 <= sequence < 2**53 for sequence in self.clarification_events)
            or tuple(sorted(set(self.clarification_events))) != self.clarification_events
        ):
            raise ValueError("Clarification pruning events must be ordered and unique.")
        return self


@dataclass(frozen=True)
class RequestPruningResult:
    released_bytes: int
    removed_operations: int
    removed_events: int


async def prune_request_batch(
    store: CollaborationStore,
    tx: _Repository,
    anchor: _Anchor,
    command: RequestCommand,
    *,
    limit: int,
    redactor: SecretRedactor,
) -> RequestPruningResult:
    from cayu.collaboration._clarification_pruning import (
        prune_question_material,
        question_pruning_material,
    )
    from cayu.collaboration._clarification_state import clarification_commitment
    from cayu.collaboration._namespace_store import load_namespace
    from cayu.collaboration._retention_store import release_unused_history
    from cayu.collaboration.base import _stored_mode

    if type(limit) is not int or not 1 <= limit <= 32:
        raise CollaborationUnavailable("Request pruning requires a bounded batch.")
    namespace = await load_namespace(tx, anchor, command.operation.generation, redactor)
    if namespace.state != "retired":
        raise CollaborationUnavailable("Request pruning requires retired namespace authority.")
    key = operation_key(command.operation)
    raw_progress = await tx.get("request_pruning", key)
    prior = None
    if raw_progress is None:
        snapshot = await retained_request(
            store, tx, anchor.initialization, command.intent.request, command.initiator, redactor
        )
        start = 0
    else:
        prior = prepare_contract(RequestPruningProgress, raw_progress, redactor=redactor)
        snapshot = prepare_contract(
            RequestSnapshot, await tx.get("requests", key), redactor=redactor
        )
        if (
            prior.operation != command.operation
            or prior.request != command.intent.selection.reference
            or prior.snapshot_sha256 != clarification_commitment(snapshot, redactor)
        ):
            raise CollaborationUnavailable("Pruning cursor contradicts its immutable source.")
        start = prior.next_event_index
    if snapshot is None or snapshot.state == "open" or snapshot.receipt.expected != command:
        raise CollaborationUnavailable("Retired request lacks exact terminal evidence.")
    if raw_progress is None:
        from cayu.collaboration._planning_retention import prune_request_plans

        planning = await prune_request_plans(
            store, tx, anchor.initialization, command, limit=limit, redactor=redactor
        )
        if planning is not None:
            return planning
    questions, question_digest = await question_pruning_material(tx, snapshot, redactor)
    if prior is not None and prior.clarification_sha256 != question_digest:
        raise CollaborationUnavailable("Clarification pruning material changed between batches.")

    if prior is None:
        clarification_events = []
        # Clarification decisions have their own frontier. They deliberately do
        # not advance the parent's observation/revision frontier. Capture their
        # exact receipts before deleting any of the combined history.
        for question in questions:
            terminal = (
                question.reply.operation
                if question.reply is not None
                else question.closure_operation
            )
            for operation in (question.question.operation, terminal):
                if operation is None:
                    raise CollaborationUnavailable("Clarification terminal identity is missing.")
                metadata = request_receipt_metadata(
                    await tx.get("operations", operation_key(operation)), redactor=redactor
                )
                if (
                    metadata is None
                    or metadata.operation != operation
                    or metadata.expected != command
                ):
                    raise CollaborationUnavailable(
                        "Clarification pruning lacks exact receipt evidence."
                    )
                clarification_events.append(metadata.receipt.event.sequence)
        question_events = tuple(sorted(clarification_events))
    else:
        question_events = prior.clarification_events
    if len(question_events) != 2 * len(questions) or set(question_events).intersection(
        snapshot.event_sequences
    ):
        raise CollaborationUnavailable(
            "Clarification pruning frontier conflicts with parent history."
        )
    frontier = tuple(sorted((*snapshot.event_sequences, *question_events)))
    if (
        len(frontier) > MAX_REQUEST_PRUNING_EVENTS
        or len(set(frontier)) != len(frontier)
        or not start < len(frontier)
    ):
        raise CollaborationUnavailable("Request pruning frontier is incomplete or unbounded.")
    for sequence in frontier[:start]:
        if await tx.get("request_events", (sequence,)) is not None:
            raise CollaborationUnavailable("Pruning cursor skipped retained event evidence.")

    stop = min(start + limit, len(frontier))
    related = []
    references = []
    for sequence in frontier[start:stop]:
        event = prepare_contract(
            RequestEvent, await tx.get("request_events", (sequence,)), redactor=redactor
        )
        if event.sequence != sequence or event.request != snapshot.receipt.event.request:
            raise CollaborationUnavailable("Request event frontier conflicts with its source.")
        raw = await tx.get("operations", operation_key(event.operation))
        metadata = request_receipt_metadata(raw, redactor=redactor)
        if metadata is not None:
            receipt, parent = metadata.receipt, metadata.expected
        elif _stored_mode(raw) == "request":
            receipt = prepare_contract(RequestReceipt, raw, redactor=redactor)
            parent = receipt.expected
        elif _stored_mode(raw) == "request_control":
            receipt = prepare_contract(RequestControlReceipt, raw, redactor=redactor)
            parent = receipt.expected.intent.expected
        else:
            raise CollaborationUnavailable("Request pruning receipt family is unavailable.")
        if parent != command or getattr(receipt, "event", None) != event:
            raise CollaborationUnavailable("Request pruning receipt contradicts its event.")
        related.append((receipt, event))
        references.extend(history_references(receipt))
    if not related or (start == 0 and related[0][0] != snapshot.receipt):
        raise CollaborationUnavailable("Request pruning lost its ordered frontier.")

    released = 0 if prior is None else len(contract_bytes(prior, redactor=redactor))
    if stop < len(frontier):
        progress = RequestPruningProgress(
            operation=command.operation,
            request=command.intent.selection.reference,
            snapshot_sha256=clarification_commitment(snapshot, redactor),
            clarification_sha256=question_digest,
            clarification_events=question_events,
            next_event_index=stop,
        )
        released -= len(contract_bytes(progress, redactor=redactor))
        await tx.put("request_pruning", key, progress, insert=prior is None)
    else:
        released += await prune_question_material(tx, questions, redactor)
        await tx.delete("request_pruning", key)
        await tx.delete("requests", key)
        released += len(contract_bytes(snapshot, redactor=redactor))
    for receipt, event in related:
        await tx.delete("operations", operation_key(event.operation))
        await tx.delete("request_events", (event.sequence,))
        released += len(contract_bytes(receipt, redactor=redactor)) + len(
            contract_bytes(event, redactor=redactor)
        )
    released += await release_unused_history(tx, tuple(references), redactor)
    # The enclosing owner accounts for this exact signed byte delta and commits
    # its maintenance receipt atomically; failure rolls back cursor and deletions.
    return RequestPruningResult(released, len(related), len(related))
