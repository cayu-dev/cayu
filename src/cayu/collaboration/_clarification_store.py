"""Clarification mutations composed inside the existing request-owner transaction.

These private functions do not authenticate a producer or call foreign stores.
The registered receiving coordinator must supply authenticated commands while
holding its disclosure/admission guard. No public capability is enabled here.
"""

from __future__ import annotations

from uuid import uuid4

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._clarification_commands import (
    ClarificationCloseCommand,
    ClarificationCloseReceipt,
    ClarificationOpenCommand,
    ClarificationOpenReceipt,
    ClarificationReplyCommand,
    ClarificationReplyReceipt,
)
from cayu.collaboration._clarification_records import ClarificationLineageRecord
from cayu.collaboration._clarification_state import (
    ClarificationQuestionState,
    accept_clarification_reply,
    clarification_commitment,
)
from cayu.collaboration._contracts import (
    MAX_ENVELOPE_BYTES,
    CollaborationConflict,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
)
from cayu.collaboration._namespace_store import require_open_namespace
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import (
    operation_key,
    require_request_absence,
    require_request_event,
    retained_request,
)
from cayu.collaboration.base import CollaborationStore, _Anchor, _Repository, _stored_mode
from cayu.collaboration.clarifications import (
    MAX_CLARIFICATION_QUESTIONS,
    ClarificationDueCursor,
    ClarificationLineageUsage,
)
from cayu.collaboration.participants import CollaborationInitialization, CollaborationUnavailable
from cayu.collaboration.requests import RequestEvent, RequestSnapshot
from cayu.vaults.redaction import SecretRedactor

# One terminal decision replaces the question and appends its receipt, input
# revision (answer only), event and lineage growth. Service/delivery handoffs reserve separately
# before their own admission; this is not permission to dispatch service work.
CLARIFICATION_DECISION_BYTES = 6 * MAX_ENVELOPE_BYTES


async def lookup_clarification_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    expected: ClarificationOpenCommand | ClarificationCloseCommand,
    *,
    redactor: SecretRedactor,
    include_state: bool = False,
):
    """Exact historical metadata, not renewal of source/execution authority."""
    expected = prepare_contract(type(expected), expected, redactor=redactor)
    await store._anchor(tx, initialized, redactor)
    if (
        expected.operation.namespace_incarnation != initialized.namespace_incarnation
        or expected.operation.application_scope != initialized.binding.application_scope
    ):
        return ExactConflict()
    key = operation_key(expected.operation)
    raw = await tx.get("operations", key)
    if raw is None:
        state = await tx.get(
            "clarification_questions",
            key
            if isinstance(expected, ClarificationOpenCommand)
            else operation_key(expected.question),
        )
        if state is not None:
            current = prepare_contract(ClarificationQuestionState, state, redactor=redactor)
            if (
                isinstance(expected, ClarificationOpenCommand)
                or current.closure_operation == expected.operation
            ):
                raise CollaborationUnavailable("Clarification is missing its decision receipt.")
        await require_request_absence(store, tx, initialized, expected.operation, redactor)
        return ExactNotFound()
    if _stored_mode(raw) != expected.mode:
        return ExactConflict()
    receipt = prepare_contract(
        ClarificationOpenReceipt
        if isinstance(expected, ClarificationOpenCommand)
        else ClarificationCloseReceipt,
        raw,
        redactor=redactor,
    )
    question_key = (
        key
        if isinstance(receipt, ClarificationOpenReceipt)
        else operation_key(receipt.command.question)
    )
    current = prepare_contract(
        ClarificationQuestionState,
        await tx.get("clarification_questions", question_key),
        redactor=redactor,
    )
    if operation_key(receipt.command.operation) != key or (
        current.question != receipt.command.question
        if isinstance(receipt, ClarificationOpenReceipt)
        else current != receipt.decision
    ):
        raise CollaborationUnavailable("Clarification indexes contradict retained evidence.")
    parent = receipt.command.expected
    retained = await retained_request(
        store, tx, initialized, parent.intent.request, parent.initiator, redactor
    )
    if retained is None or retained.receipt.expected != parent:
        raise CollaborationUnavailable("Clarification parent evidence is unavailable.")
    await require_request_event(tx, receipt.event, redactor)
    if receipt.command != expected:
        return ExactConflict()
    if include_state:
        return ExactMatch[ClarificationQuestionState](receipt=current)
    return ExactMatch[ClarificationOpenReceipt | ClarificationCloseReceipt](receipt=receipt)


async def discover_due_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    *,
    after: ClarificationDueCursor | None,
    limit: int,
    redactor: SecretRedactor,
) -> tuple[ClarificationOpenReceipt, ...]:
    """Owner-clock discovery, not permission to disclose or service the result.

    A later sweep starts with no cursor: new work can become due before an older
    sweep's cursor. Selection neither claims a worker slot nor settles a decision.
    """
    await store._anchor(tx, initialized, redactor)
    now = await tx.now_ms()
    records = await tx.scan_due_clarifications(after=after, now_ms=now, limit=limit)
    result = []
    for raw in records:
        decision = prepare_contract(ClarificationQuestionState, raw, redactor=redactor)
        opening = prepare_contract(
            ClarificationOpenReceipt,
            await tx.get("operations", operation_key(decision.question.operation)),
            redactor=redactor,
        )
        expected = opening.command.expected
        require_exact_contract(decision.question, opening.command.question, redactor=redactor)
        if decision.state != "open" or decision.question.deadline_at_ms > now:
            raise CollaborationUnavailable(
                "Clarification due lookup conflicts with owner evidence."
            )
        parent = await retained_request(
            store, tx, initialized, expected.intent.request, expected.initiator, redactor
        )
        if parent is None:
            raise CollaborationUnavailable("Clarification recovery parent is unavailable.")
        require_exact_contract(expected, parent.receipt.expected, redactor=redactor)
        result.append(opening)
    return tuple(result)


async def _input_head(
    tx: _Repository,
    snapshot: RequestSnapshot,
    records: list[ClarificationQuestionState],
    redactor: SecretRedactor,
) -> tuple[int, str]:
    expected = snapshot.receipt.expected
    request = expected.intent.selection.reference
    frontier = snapshot.clarification
    if (
        len(records) != frontier.generation
        or sorted(record.question.generation for record in records)
        != list(range(1, frontier.generation + 1))
        or any(record.question.lineage != frontier.lineage for record in records)
    ):
        raise CollaborationUnavailable("Clarification question frontier is incomplete.")
    commitment = clarification_commitment(expected, redactor)
    if any(
        record.question.request != request or record.question.request_sha256 != commitment
        for record in records
    ):
        raise CollaborationUnavailable("Clarification input belongs to another request.")
    for record in records:
        opening = prepare_contract(
            ClarificationOpenReceipt,
            await tx.get("operations", operation_key(record.question.operation)),
            redactor=redactor,
        )
        require_exact_contract(expected, opening.command.expected, redactor=redactor)
        require_exact_contract(record.question, opening.command.question, redactor=redactor)
        await require_request_event(tx, opening.event, redactor)
        if record.reply is not None:
            answer = prepare_contract(
                ClarificationReplyReceipt,
                await tx.get("operations", operation_key(record.reply.operation)),
                redactor=redactor,
            )
            require_exact_contract(expected, answer.command.expected, redactor=redactor)
            require_exact_contract(record, answer.decision, redactor=redactor)
            await require_request_event(tx, answer.event, redactor)
        elif record.closure_operation is not None:
            closure = prepare_contract(
                ClarificationCloseReceipt,
                await tx.get("operations", operation_key(record.closure_operation)),
                redactor=redactor,
            )
            require_exact_contract(expected, closure.command.expected, redactor=redactor)
            require_exact_contract(record, closure.decision, redactor=redactor)
            await require_request_event(tx, closure.event, redactor)
    revisions = sorted(
        (record.input for record in records if record.input is not None),
        key=lambda revision: revision.revision,
    )
    input_revision = 0
    for revision in revisions:
        if revision.revision != input_revision + 1 or revision.previous_sha256 != commitment:
            raise CollaborationUnavailable("Clarification input ancestry is incomplete.")
        retained = await tx.get(
            "clarification_inputs", (request.request_id, request.incarnation, revision.revision)
        )
        require_exact_contract(
            revision,
            prepare_contract(type(revision), retained, redactor=redactor),
            redactor=redactor,
        )
        input_revision = revision.revision
        commitment = clarification_commitment(revision, redactor)
    if input_revision != frontier.input_revision or (
        input_revision > 0 and commitment != frontier.input_sha256
    ):
        raise CollaborationUnavailable("Clarification effective-input head is incomplete.")
    return input_revision, commitment


async def validate_request_frontier(
    tx: _Repository, snapshot: RequestSnapshot, redactor: SecretRedactor
) -> None:
    records = [
        prepare_contract(ClarificationQuestionState, value, redactor=redactor)
        for value in await tx.scan_clarification_questions(
            snapshot.receipt.expected.intent.selection.reference,
            limit=MAX_CLARIFICATION_QUESTIONS + 1,
        )
    ]
    await _input_head(tx, snapshot, records, redactor)


async def open_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    command: ClarificationOpenCommand,
    *,
    redactor: SecretRedactor,
    _planned_stage=None,
) -> ClarificationOpenReceipt:
    command = prepare_contract(ClarificationOpenCommand, command, redactor=redactor)
    from cayu.collaboration._planning_stages import finish_stage, require_receiving_stage

    await require_receiving_stage(
        store, tx, initialized, command, _planned_stage, redactor=redactor
    )
    anchor = await store._anchor(tx, initialized, redactor)
    expected = command.expected
    prior = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if prior is None:
        raise CollaborationUnavailable("Clarification parent acceptance is unavailable.")
    require_exact_contract(expected, prior.receipt.expected, redactor=redactor)
    key = operation_key(command.operation)
    raw = await tx.get("operations", key)
    if raw is not None:
        if _stored_mode(raw) != "clarification_open":
            raise CollaborationConflict("Clarification operation belongs to another family.")
        receipt = prepare_contract(ClarificationOpenReceipt, raw, redactor=redactor)
        require_exact_contract(command, receipt.command, redactor=redactor)
        current = prepare_contract(
            ClarificationQuestionState,
            await tx.get("clarification_questions", key),
            redactor=redactor,
        )
        require_exact_contract(command.question, current.question, redactor=redactor)
        await require_request_event(tx, receipt.event, redactor)
        return receipt
    if await tx.get("clarification_questions", key) is not None:
        raise CollaborationUnavailable("Clarification is missing its opening receipt.")
    await require_open_namespace(tx, anchor, command.operation, redactor)
    question = command.question
    now = await tx.now_ms()
    if (
        prior.state != "open"
        or prior.admission != "clarifying"
        or prior.revision != command.expected_revision
        or prior.admission_operation != question.admission
        or now >= question.deadline_at_ms
        or question.deadline_at_ms - now > question.policy.service_timeout_ms
    ):
        raise CollaborationConflict("Clarification request admission or deadline changed.")
    frontier = prior.clarification
    if frontier.generation == 0 and await tx.scan_clarification_questions(
        question.request, limit=1
    ):
        raise CollaborationUnavailable("Clarification records lack their request frontier.")
    if frontier.generation >= question.policy.max_questions:
        raise CollaborationUnavailable("Clarification request question capacity is exhausted.")
    if question.generation != frontier.generation + 1 or (
        frontier.lineage is not None and frontier.lineage != question.lineage
    ):
        raise CollaborationConflict("Clarification generation or lineage conflicts.")
    # retained_request has already authenticated this frontier against receipts,
    # events and input records under this same transaction.
    input_revision = frontier.input_revision
    input_sha256 = frontier.input_sha256 or clarification_commitment(expected, redactor)
    if (question.input_revision, question.input_sha256) != (input_revision, input_sha256):
        raise CollaborationConflict("Clarification does not bind the current effective input.")
    lineage_key = operation_key(question.lineage)
    raw_lineage = await tx.get("clarification_lineages", lineage_key)
    if raw_lineage is None:
        if question.parent_question is not None or frontier.generation:
            raise CollaborationUnavailable("Clarification ancestor lineage is unavailable.")
        # Reclamation must not let a new root resurrect the identity of an
        # already retired lineage with fresh lifetime counters.
        if question.lineage.namespace_incarnation != initialized.namespace_incarnation:
            raise CollaborationConflict("Clarification lineage belongs to another namespace.")
        await require_open_namespace(tx, anchor, question.lineage, redactor)
        lineage = ClarificationLineageRecord(
            operation=question.lineage,
            root_request=question.request,
            policy=question.policy,
            budget_binding=question.budget_binding,
            budget_authority_sha256=question.budget_authority_sha256,
            usage=ClarificationLineageUsage(),
        )
    else:
        lineage = prepare_contract(ClarificationLineageRecord, raw_lineage, redactor=redactor)
        if (
            lineage.operation != question.lineage
            or lineage.policy != question.policy
            or lineage.budget_binding != question.budget_binding
            or lineage.budget_authority_sha256 != question.budget_authority_sha256
            or (question.parent_question is None and lineage.root_request != question.request)
        ):
            raise CollaborationConflict("Clarification cannot replace its lineage authority.")
    if question.parent_question is not None:
        parent = prepare_contract(
            ClarificationQuestionState,
            await tx.get("clarification_questions", operation_key(question.parent_question)),
            redactor=redactor,
        )
        if (
            parent.question.operation != question.parent_question
            or parent.question.lineage != question.lineage
            or parent.question.depth + 1 != question.depth
            or parent.state != "open"
        ):
            raise CollaborationConflict("Clarification parent no longer admits nested work.")
    usage = prepare_contract(
        ClarificationLineageUsage,
        lineage.usage.model_copy(
            update={
                "questions": lineage.usage.questions + 1,
                "content_bytes": lineage.usage.content_bytes
                + question.source.content_bytes
                + question.policy.max_reply_bytes,
                "pending": lineage.usage.pending + 1,
            }
        ),
        redactor=redactor,
    )
    updated_lineage = prepare_contract(
        ClarificationLineageRecord, lineage.model_copy(update={"usage": usage}), redactor=redactor
    )
    decision = ClarificationQuestionState(question=question, opened_at_ms=now, state="open")
    event = RequestEvent(
        id=uuid4().hex,
        sequence=anchor.event_sequence + 1,
        operation=command.operation,
        request=question.request,
        type="clarification_opened",
        participants=prior.receipt.event.participants,
    )
    receipt = prepare_contract(
        ClarificationOpenReceipt,
        ClarificationOpenReceipt(command=command, decision=decision, event=event),
        redactor=redactor,
    )
    updated_request = prepare_contract(
        RequestSnapshot,
        prior.model_copy(
            update={
                "clarification": prior.clarification.model_copy(
                    update={"generation": question.generation, "lineage": question.lineage}
                )
            }
        ),
        redactor=redactor,
    )
    added_bytes = (
        sum(
            len(contract_bytes(value, redactor=redactor))
            for value in (receipt, decision, event, updated_lineage, updated_request)
        )
        - len(contract_bytes(prior, redactor=redactor))
        - (len(contract_bytes(lineage, redactor=redactor)) if raw_lineage is not None else 0)
    )
    updated_anchor = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "retained_bytes": anchor.retained_bytes + added_bytes,
                "reserved_operations": anchor.reserved_operations + 1,
                "reserved_events": anchor.reserved_events + 1,
                "reserved_bytes": anchor.reserved_bytes + CLARIFICATION_DECISION_BYTES,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated_anchor, ordinary=True)
    await tx.put("operations", key, receipt, insert=True)
    await tx.put("clarification_questions", key, decision, insert=True)
    await tx.put("clarification_lineages", lineage_key, updated_lineage, insert=raw_lineage is None)
    await tx.put("request_events", (event.sequence,), event, insert=True)
    await tx.put("requests", operation_key(expected.operation), updated_request, insert=False)
    await tx.put("anchors", (), updated_anchor, insert=False)
    if _planned_stage is not None:
        await finish_stage(
            store,
            tx,
            initialized,
            _planned_stage.plan,
            _planned_stage.intent,
            receipt,
            redactor=redactor,
        )
    return receipt


async def reply_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    command: ClarificationReplyCommand,
    *,
    redactor: SecretRedactor,
) -> ClarificationReplyReceipt:
    command = prepare_contract(ClarificationReplyCommand, command, redactor=redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    expected = command.expected
    prior = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if prior is None:
        raise CollaborationUnavailable("Clarification parent acceptance is unavailable.")
    require_exact_contract(expected, prior.receipt.expected, redactor=redactor)
    key = operation_key(command.operation)
    question_key = operation_key(command.reply.question)
    opening = prepare_contract(
        ClarificationOpenReceipt, await tx.get("operations", question_key), redactor=redactor
    )
    require_exact_contract(expected, opening.command.expected, redactor=redactor)
    await require_request_event(tx, opening.event, redactor)
    current = prepare_contract(
        ClarificationQuestionState,
        await tx.get("clarification_questions", question_key),
        redactor=redactor,
    )
    require_exact_contract(opening.command.question, current.question, redactor=redactor)
    raw = await tx.get("operations", key)
    if raw is not None:
        if _stored_mode(raw) != "clarification_reply":
            raise CollaborationConflict("Reply operation belongs to another family.")
        receipt = prepare_contract(ClarificationReplyReceipt, raw, redactor=redactor)
        require_exact_contract(command, receipt.command, redactor=redactor)
        require_exact_contract(receipt.decision, current, redactor=redactor)
        await require_request_event(tx, receipt.event, redactor)
        assert current.input is not None
        retained = await tx.get(
            "clarification_inputs",
            (
                current.input.request.request_id,
                current.input.request.incarnation,
                current.input.revision,
            ),
        )
        require_exact_contract(
            current.input,
            prepare_contract(type(current.input), retained, redactor=redactor),
            redactor=redactor,
        )
        return receipt
    if current.state != "open":
        raise CollaborationConflict("Clarification already has a terminal decision.")
    if prior.clarification.generation < current.question.generation:
        raise CollaborationUnavailable("Clarification question is outside its request frontier.")
    revision = prior.clarification.input_revision
    commitment = prior.clarification.input_sha256 or clarification_commitment(expected, redactor)
    now = await tx.now_ms()
    decision = accept_clarification_reply(
        current,
        command.reply,
        request_is_open=prior.state == "open",
        current_input_revision=revision,
        current_input_sha256=commitment,
        now_ms=now,
        redactor=redactor,
    )
    lineage_key = operation_key(current.question.lineage)
    lineage = prepare_contract(
        ClarificationLineageRecord,
        await tx.get("clarification_lineages", lineage_key),
        redactor=redactor,
    )
    if (
        lineage.operation != current.question.lineage
        or lineage.policy != current.question.policy
        or lineage.budget_binding != current.question.budget_binding
        or lineage.budget_authority_sha256 != current.question.budget_authority_sha256
        or lineage.usage.pending < 1
        or lineage.usage.content_bytes < current.question.policy.max_reply_bytes
        or anchor.reserved_operations < 1
        or anchor.reserved_events < 1
        or anchor.reserved_bytes < CLARIFICATION_DECISION_BYTES
    ):
        raise CollaborationUnavailable("Clarification decision reservation is unavailable.")
    usage = lineage.usage.model_copy(
        update={
            "pending": lineage.usage.pending - 1,
            "content_bytes": lineage.usage.content_bytes
            - current.question.policy.max_reply_bytes
            + command.reply.source.content_bytes,
        }
    )
    updated_lineage = prepare_contract(
        ClarificationLineageRecord, lineage.model_copy(update={"usage": usage}), redactor=redactor
    )
    event = RequestEvent(
        id=uuid4().hex,
        sequence=anchor.event_sequence + 1,
        operation=command.operation,
        request=current.question.request,
        type="clarification_replied",
        participants=prior.receipt.event.participants,
    )
    receipt = prepare_contract(
        ClarificationReplyReceipt,
        ClarificationReplyReceipt(command=command, decision=decision, event=event),
        redactor=redactor,
    )
    assert decision.input is not None
    updated_request = prepare_contract(
        RequestSnapshot,
        prior.model_copy(
            update={
                "clarification": prior.clarification.model_copy(
                    update={
                        "input_revision": decision.input.revision,
                        "input_sha256": clarification_commitment(decision.input, redactor),
                    }
                )
            }
        ),
        redactor=redactor,
    )
    await _publish_decision(
        tx,
        anchor=anchor,
        prior=prior,
        current=current,
        lineage=lineage,
        receipt=receipt,
        updated_lineage=updated_lineage,
        updated_request=updated_request,
        redactor=redactor,
    )
    return receipt


async def close_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    command: ClarificationCloseCommand,
    *,
    redactor: SecretRedactor,
) -> ClarificationCloseReceipt:
    command = prepare_contract(ClarificationCloseCommand, command, redactor=redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    expected = command.expected
    prior = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if prior is None:
        raise CollaborationUnavailable("Clarification parent acceptance is unavailable.")
    require_exact_contract(expected, prior.receipt.expected, redactor=redactor)
    current = prepare_contract(
        ClarificationQuestionState,
        await tx.get("clarification_questions", operation_key(command.question)),
        redactor=redactor,
    )
    if (
        current.question.operation != command.question
        or current.question.request != expected.intent.selection.reference
        or current.question.request_sha256 != clarification_commitment(expected, redactor)
        or clarification_commitment(current.question, redactor) != command.question_sha256
        or current.question.generation > prior.clarification.generation
    ):
        raise CollaborationConflict("Clarification closure does not match the retained question.")
    raw = await tx.get("operations", operation_key(command.operation))
    if raw is not None:
        if _stored_mode(raw) != "clarification_close":
            raise CollaborationConflict("Closure operation belongs to another family.")
        receipt = prepare_contract(ClarificationCloseReceipt, raw, redactor=redactor)
        require_exact_contract(command, receipt.command, redactor=redactor)
        require_exact_contract(current, receipt.decision, redactor=redactor)
        await require_request_event(tx, receipt.event, redactor)
        return receipt
    if current.state != "open":
        raise CollaborationConflict("Clarification already has a terminal decision.")
    now = await tx.now_ms()
    if (
        (command.kind == "expired" and now < current.question.deadline_at_ms)
        or (command.kind == "request_terminal" and prior.state == "open")
        or (
            command.kind == "superseded"
            and prior.clarification.input_revision <= current.question.input_revision
        )
    ):
        raise CollaborationConflict("Clarification closure lacks authoritative terminal evidence.")
    decision = prepare_contract(
        ClarificationQuestionState,
        current.model_copy(
            update={
                "state": command.kind,
                "closed_at_ms": now,
                "closure_operation": command.operation,
            }
        ),
        redactor=redactor,
    )
    lineage = prepare_contract(
        ClarificationLineageRecord,
        await tx.get("clarification_lineages", operation_key(current.question.lineage)),
        redactor=redactor,
    )
    if (
        lineage.operation != current.question.lineage
        or lineage.policy != current.question.policy
        or lineage.budget_binding != current.question.budget_binding
        or lineage.budget_authority_sha256 != current.question.budget_authority_sha256
        or lineage.usage.pending < 1
        or lineage.usage.content_bytes < current.question.policy.max_reply_bytes
    ):
        raise CollaborationUnavailable("Clarification lineage reservation is unavailable.")
    updated_lineage = prepare_contract(
        ClarificationLineageRecord,
        lineage.model_copy(
            update={
                "usage": lineage.usage.model_copy(
                    update={
                        "pending": lineage.usage.pending - 1,
                        "content_bytes": lineage.usage.content_bytes
                        - current.question.policy.max_reply_bytes,
                    }
                )
            }
        ),
        redactor=redactor,
    )
    event = RequestEvent(
        id=uuid4().hex,
        sequence=anchor.event_sequence + 1,
        operation=command.operation,
        request=current.question.request,
        type="clarification_closed",
        participants=prior.receipt.event.participants,
    )
    receipt = prepare_contract(
        ClarificationCloseReceipt,
        ClarificationCloseReceipt(command=command, decision=decision, event=event),
        redactor=redactor,
    )
    await _publish_decision(
        tx,
        anchor=anchor,
        prior=prior,
        current=current,
        lineage=lineage,
        receipt=receipt,
        updated_lineage=updated_lineage,
        updated_request=prior,
        redactor=redactor,
    )
    return receipt


async def _publish_decision(
    tx: _Repository,
    *,
    anchor: _Anchor,
    prior: RequestSnapshot,
    current: ClarificationQuestionState,
    lineage: ClarificationLineageRecord,
    receipt: ClarificationReplyReceipt | ClarificationCloseReceipt,
    updated_lineage: ClarificationLineageRecord,
    updated_request: RequestSnapshot,
    redactor: SecretRedactor,
) -> None:
    decision, event = receipt.decision, receipt.event
    if (
        current.state != "open"
        or anchor.reserved_operations < 1
        or anchor.reserved_events < 1
        or anchor.reserved_bytes < CLARIFICATION_DECISION_BYTES
    ):
        raise CollaborationUnavailable("Clarification decision reservation is unavailable.")
    material = (receipt, decision, event, updated_lineage, updated_request)
    if decision.input is not None:
        material = (*material, decision.input)
    added_bytes = sum(len(contract_bytes(value, redactor=redactor)) for value in material) - sum(
        len(contract_bytes(value, redactor=redactor)) for value in (current, lineage, prior)
    )
    if added_bytes > CLARIFICATION_DECISION_BYTES:
        raise CollaborationUnavailable("Clarification decision exceeds its reserved envelope.")
    updated_anchor = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "retained_bytes": anchor.retained_bytes + added_bytes,
                "reserved_operations": anchor.reserved_operations - 1,
                "reserved_events": anchor.reserved_events - 1,
                "reserved_bytes": anchor.reserved_bytes - CLARIFICATION_DECISION_BYTES,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated_anchor, ordinary=False)
    await tx.put("operations", operation_key(receipt.command.operation), receipt, insert=True)
    await tx.put(
        "clarification_questions", operation_key(current.question.operation), decision, insert=False
    )
    if decision.input is not None:
        await tx.put(
            "clarification_inputs",
            (
                decision.input.request.request_id,
                decision.input.request.incarnation,
                decision.input.revision,
            ),
            decision.input,
            insert=True,
        )
    await tx.put(
        "clarification_lineages", operation_key(lineage.operation), updated_lineage, insert=False
    )
    await tx.put("request_events", (event.sequence,), event, insert=True)
    await tx.put(
        "requests", operation_key(prior.receipt.expected.operation), updated_request, insert=False
    )
    await tx.put("anchors", (), updated_anchor, insert=False)
