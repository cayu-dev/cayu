"""Trusted request admission, progress, outcome, and observation transactions."""

from __future__ import annotations

from hashlib import sha256
from typing import Literal, cast
from uuid import uuid4

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import (
    MAX_ENVELOPE_BYTES,
    CollaborationConflict,
    ContractValue,
    InitiatorBinding,
)
from cayu.collaboration._permit_store import settle_permit_in_transaction
from cayu.collaboration._permits import ReceivingSettlementReceipt
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_store import (
    observation_operation,
    preflight_control,
    require_request_event,
    retained_request,
)
from cayu.collaboration.base import CollaborationStore, _Anchor, _key, _Repository
from cayu.collaboration.participants import CollaborationInitialization, CollaborationUnavailable
from cayu.collaboration.requests import (
    AdmissionState,
    RequestAdmissionCommand,
    RequestAdmissionReceipt,
    RequestEvent,
    RequestObservation,
    RequestObservationReceipt,
    RequestOutcomeCommand,
    RequestOutcomeReceipt,
    RequestProgressCommand,
    RequestProgressReceipt,
    RequestSnapshot,
    source_export_matches_request,
)
from cayu.vaults.redaction import SecretRedactor


def _operation_of(value):
    command = getattr(value, "command", None)
    return getattr(command, "operation", getattr(value, "operation", None))


def _record_key(value):
    operation = _operation_of(value)
    return _operation_key(operation)


def _operation_key(operation):
    if operation is None:
        raise CollaborationConflict("Request evidence has no operation identity.")
    return operation.namespace_incarnation, operation.generation, operation.caller_key


def _event(
    anchor: _Anchor,
    command,
    request,
    event_type: Literal[
        "request_accepted",
        "request_admission",
        "request_progress",
        "request_answered",
        "request_failed",
        "request_declined",
        "request_cancelled",
        "request_expired",
        "request_observation_registered",
    ],
    participants,
    sequence: int,
    operation=None,
):
    return RequestEvent(
        id=uuid4().hex,
        sequence=sequence,
        operation=command.operation if operation is None else operation,
        request=request,
        type=event_type,
        participants=participants,
    )


async def _write_snapshot(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    anchor: _Anchor,
    prior: RequestSnapshot,
    *,
    operation,
    snapshot: RequestSnapshot,
    receipts: tuple[ContractValue, ...],
    events: tuple[RequestEvent, ...],
    consume_reserved: bool = False,
    redactor: SecretRedactor,
) -> None:
    sequences = (*prior.event_sequences, *(event.sequence for event in events))
    if not consume_reserved and prior.state == "open" and len(sequences) > 62:
        raise CollaborationUnavailable("Optional events cannot consume terminal frontier capacity.")
    snapshot = prepare_contract(
        RequestSnapshot,
        snapshot.model_copy(update={"event_sequences": sequences}),
        redactor=redactor,
    )
    if not consume_reserved and prior.state == "open":
        preflight_control(snapshot, redactor)
    if consume_reserved and (
        anchor.reserved_operations < 1
        or anchor.reserved_events < 1
        or anchor.reserved_bytes < REQUEST_TERMINAL_BYTES
    ):
        raise CollaborationUnavailable("Request lacks reserved terminal capacity.")
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + len(receipts),
                "event_count": anchor.event_count + len(events),
                "event_sequence": events[-1].sequence,
                "retained_bytes": anchor.retained_bytes
                + sum(
                    len(contract_bytes(value, redactor=redactor))
                    for value in (*receipts, snapshot, *events)
                )
                - len(contract_bytes(prior, redactor=redactor)),
                "reserved_operations": anchor.reserved_operations - (1 if consume_reserved else 0),
                "reserved_events": anchor.reserved_events - (1 if consume_reserved else 0),
                "reserved_bytes": anchor.reserved_bytes
                - (REQUEST_TERMINAL_BYTES if consume_reserved else 0),
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=True)
    for receipt in receipts:
        await tx.put("operations", _record_key(receipt), receipt, insert=True)
    await tx.put("requests", _key(prior.receipt.expected), snapshot, insert=False)
    for event in events:
        await tx.put("request_events", (event.sequence,), event, insert=True)
    await tx.put("anchors", (), updated, insert=False)


async def _expected_prior(store, tx, initialized, command, redactor):
    expected = command.expected
    if (
        command.operation.application_scope != expected.operation.application_scope
        or command.operation.namespace_incarnation != expected.operation.namespace_incarnation
        or command.operation.generation != expected.operation.generation
        or command.operation == expected.operation
    ):
        raise CollaborationConflict("Receiving operation conflicts with the request namespace.")
    prior = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if prior is None:
        raise CollaborationUnavailable("Request acceptance is unavailable.")
    require_exact_contract(expected, prior.receipt.expected, redactor=redactor)
    return prior


async def admit_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    command: RequestAdmissionCommand,
    *,
    settlement: ReceivingSettlementReceipt | None = None,
    redactor: SecretRedactor,
) -> RequestAdmissionReceipt:
    command = prepare_contract(RequestAdmissionCommand, command, redactor=redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    prior = await _expected_prior(store, tx, initialized, command, redactor)
    raw = await tx.get("operations", _record_key(command))
    if raw is not None:
        receipt = prepare_contract(RequestAdmissionReceipt, raw, redactor=redactor)
        require_exact_contract(command, receipt.command, redactor=redactor)
        await require_request_event(tx, receipt.event, redactor)
        return receipt
    if prior.state != "open":
        raise CollaborationUnavailable("Request is not available for admission.")
    if prior.revision != command.expected_revision or prior.admission in {"admitted", "closed"}:
        raise CollaborationConflict("Request admission revision or state changed.")
    if command.generation != prior.admission_generation + 1:
        raise CollaborationConflict("Admission generation is not the next generation.")
    now = await tx.now_ms()
    if now >= prior.receipt.expected.intent.selection.expires_at_ms:
        raise CollaborationConflict("Request expired before admission.")
    state = cast(
        "AdmissionState",
        {
            "continue": "admitted" if command.evidence else "preparing",
            "fork": "admitted" if command.evidence else "preparing",
            "fresh": "admitted" if command.evidence else "preparing",
            "defer": "deferred",
            "clarify": "clarifying",
            "decline": "closed",
        }[command.decision],
    )
    event = _event(
        anchor,
        command,
        prior.receipt.expected.intent.selection.reference,
        "request_admission",
        prior.receipt.event.participants,
        anchor.event_sequence + 1,
    )
    receipt = prepare_contract(
        RequestAdmissionReceipt,
        RequestAdmissionReceipt(
            command=command,
            state=state,
            revision=prior.revision + 1,
            decided_at_ms=now,
            event=event,
        ),
        redactor=redactor,
    )
    outcome_receipt = None
    extra_receipts: tuple[ContractValue, ...] = ()
    events: tuple[RequestEvent, ...] = (event,)
    if command.decision == "decline":
        outcome_operation = command.operation.model_copy(
            update={
                "caller_key": "decline:" + sha256(command.operation.caller_key.encode()).hexdigest()
            }
        )
        outcome_command = prepare_contract(
            RequestOutcomeCommand,
            RequestOutcomeCommand(
                operation=outcome_operation,
                expected=prior.receipt.expected,
                expected_revision=receipt.revision,
                outcome="declined",
                commitment=command.proposal_commitment,
                source_receipt=None,
                initiator=command.initiator,
            ),
            redactor=redactor,
        )
        outcome_event = _event(
            anchor,
            outcome_command,
            prior.receipt.expected.intent.selection.reference,
            "request_declined",
            prior.receipt.event.participants,
            anchor.event_sequence + 2,
        )
        outcome_receipt = prepare_contract(
            RequestOutcomeReceipt,
            RequestOutcomeReceipt(
                command=outcome_command,
                revision=receipt.revision + 1,
                elected_at_ms=now,
                event=outcome_event,
            ),
            redactor=redactor,
        )
        extra_receipts = (outcome_receipt,)
        events = (event, outcome_event)
        if settlement is None and prior.admission != "undecided":
            raise CollaborationUnavailable("Prior preparation lacks settlement evidence.")
        if settlement is None:
            settlement = ReceivingSettlementReceipt(
                expected=prior.permit,
                receiving_owner=initialized.owner,
                receipt_id=outcome_event.id,
                outcome="quiescent",
            )
        await settle_permit_in_transaction(
            store, tx, initialized, prior.permit, settlement, redactor
        )
        anchor = await store._anchor(tx, initialized, redactor)
        event = event.model_copy(update={"sequence": anchor.event_sequence + 1})
        receipt = receipt.model_copy(update={"event": event})
        outcome_event = outcome_event.model_copy(update={"sequence": anchor.event_sequence + 2})
        outcome_receipt = outcome_receipt.model_copy(update={"event": outcome_event})
        extra_receipts = (outcome_receipt,)
        events = (event, outcome_event)
    snapshot = prepare_contract(
        RequestSnapshot,
        prior.model_copy(
            update={
                "revision": receipt.revision
                if outcome_receipt is None
                else outcome_receipt.revision,
                "admission": state,
                "admission_generation": command.generation,
                "admission_decision": command.decision,
                "admission_operation": command.operation,
                "state": "declined" if outcome_receipt is not None else prior.state,
                "outcome": outcome_receipt,
                "delivery": "excluded" if outcome_receipt is not None else prior.delivery,
                "next_due_at_ms": 0
                if state in {"admitted", "closed"}
                else prior.receipt.expected.intent.selection.expires_at_ms,
            }
        ),
        redactor=redactor,
    )
    await _write_snapshot(
        store,
        tx,
        initialized,
        anchor,
        prior,
        operation=command.operation,
        snapshot=snapshot,
        receipts=(receipt, *extra_receipts),
        events=events,
        consume_reserved=outcome_receipt is not None,
        redactor=redactor,
    )
    return receipt


async def progress_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    command: RequestProgressCommand,
    *,
    redactor: SecretRedactor,
) -> RequestProgressReceipt:
    command = prepare_contract(RequestProgressCommand, command, redactor=redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    prior = await _expected_prior(store, tx, initialized, command, redactor)
    raw = await tx.get("operations", _record_key(command))
    if raw is not None:
        receipt = prepare_contract(RequestProgressReceipt, raw, redactor=redactor)
        require_exact_contract(command, receipt.command, redactor=redactor)
        await require_request_event(tx, receipt.event, redactor)
        return receipt
    if prior.admission not in {"preparing", "admitted"}:
        raise CollaborationConflict("Progress requires an active admission.")
    if prior.admission_generation != command.admission_generation:
        raise CollaborationConflict("Progress belongs to another admission generation.")
    if prior.admission_operation is None:
        raise CollaborationConflict("Progress lacks its admitted export identity.")
    raw_admission = await tx.get("operations", _operation_key(prior.admission_operation))
    if raw_admission is None:
        raise CollaborationUnavailable("Progress admission evidence is unavailable.")
    admission = prepare_contract(RequestAdmissionReceipt, raw_admission, redactor=redactor)
    require_exact_contract(prior.receipt.expected, admission.command.expected, redactor=redactor)
    await require_request_event(tx, admission.event, redactor)
    if (
        admission.command.source_export is None
        or command.source_receipt is None
        or command.source_receipt.expected.intent.request.ref != admission.command.source_export
        or not source_export_matches_request(
            command.source_receipt,
            admission.command.expected.intent.request,
            admission.command.expected.initiator,
        )
    ):
        raise CollaborationConflict("Progress source does not match the admitted export.")
    if prior.revision != command.expected_revision:
        raise CollaborationConflict("Progress revision changed.")
    if prior.state != "open":
        raise CollaborationConflict("Terminal request cannot receive progress.")
    if command.sequence != len(prior.progress) + 1:
        raise CollaborationConflict("Progress sequence is not the next occurrence.")
    if any(item.command.sequence == command.sequence for item in prior.progress):
        raise CollaborationConflict("Progress sequence is already occupied.")
    event = _event(
        anchor,
        command,
        prior.receipt.expected.intent.selection.reference,
        "request_progress",
        prior.receipt.event.participants,
        anchor.event_sequence + 1,
    )
    receipt = prepare_contract(
        RequestProgressReceipt,
        RequestProgressReceipt(command=command, revision=prior.revision + 1, event=event),
        redactor=redactor,
    )
    snapshot = prepare_contract(
        RequestSnapshot,
        prior.model_copy(
            update={"revision": receipt.revision, "progress": (*prior.progress, receipt)}
        ),
        redactor=redactor,
    )
    await _write_snapshot(
        store,
        tx,
        initialized,
        anchor,
        prior,
        operation=command.operation,
        snapshot=snapshot,
        receipts=(receipt,),
        events=(event,),
        redactor=redactor,
    )
    return receipt


async def outcome_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    command: RequestOutcomeCommand,
    *,
    settlement: ReceivingSettlementReceipt | None = None,
    redactor: SecretRedactor,
) -> RequestOutcomeReceipt:
    command = prepare_contract(RequestOutcomeCommand, command, redactor=redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    prior = await _expected_prior(store, tx, initialized, command, redactor)
    raw = await tx.get("operations", _record_key(command))
    if raw is not None:
        receipt = prepare_contract(RequestOutcomeReceipt, raw, redactor=redactor)
        require_exact_contract(command, receipt.command, redactor=redactor)
        await require_request_event(tx, receipt.event, redactor)
        return receipt
    if prior.state != "open":
        raise CollaborationConflict("Request is already terminal or unavailable.")
    if prior.revision != command.expected_revision:
        raise CollaborationConflict("Request outcome revision changed.")
    if command.outcome == "answered" and prior.admission != "admitted":
        raise CollaborationConflict("An answer requires an admitted request.")
    if command.outcome in {"failed", "declined"} and prior.admission not in {
        "preparing",
        "admitted",
    }:
        raise CollaborationConflict("Failure or decline requires an active admission.")
    if command.outcome == "answered":
        if prior.admission_operation is None:
            raise CollaborationConflict("Answer lacks its admitted export identity.")
        raw_admission = await tx.get("operations", _operation_key(prior.admission_operation))
        if raw_admission is None:
            raise CollaborationUnavailable("Request admission evidence is unavailable.")
        admission = prepare_contract(RequestAdmissionReceipt, raw_admission, redactor=redactor)
        require_exact_contract(
            prior.receipt.expected, admission.command.expected, redactor=redactor
        )
        await require_request_event(tx, admission.event, redactor)
        source_receipt = command.source_receipt
        if (
            admission.command.source_export is None
            or source_receipt is None
            or source_receipt.expected.intent.request.ref != admission.command.source_export
            or not source_export_matches_request(
                source_receipt,
                admission.command.expected.intent.request,
                admission.command.expected.initiator,
            )
        ):
            raise CollaborationConflict("Answer source does not match the admitted export.")
    if settlement is None:
        raise CollaborationUnavailable("Terminal outcome lacks authenticated settlement evidence.")
    await settle_permit_in_transaction(store, tx, initialized, prior.permit, settlement, redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    now = await tx.now_ms()
    if now >= prior.receipt.expected.intent.selection.expires_at_ms:
        raise CollaborationConflict("Request deadline won before answer publication.")
    event = _event(
        anchor,
        command,
        prior.receipt.expected.intent.selection.reference,
        "request_" + command.outcome,
        prior.receipt.event.participants,
        anchor.event_sequence + 1,
    )
    receipt = prepare_contract(
        RequestOutcomeReceipt,
        RequestOutcomeReceipt(
            command=command, revision=prior.revision + 1, elected_at_ms=now, event=event
        ),
        redactor=redactor,
    )
    snapshot = prepare_contract(
        RequestSnapshot,
        prior.model_copy(
            update={
                "revision": receipt.revision,
                "state": command.outcome,
                "outcome": receipt,
                "admission": "closed",
                "delivery": "published" if command.outcome == "answered" else "excluded",
                "next_due_at_ms": 0,
            }
        ),
        redactor=redactor,
    )
    await _write_snapshot(
        store,
        tx,
        initialized,
        anchor,
        prior,
        operation=command.operation,
        snapshot=snapshot,
        receipts=(receipt,),
        events=(event,),
        consume_reserved=True,
        redactor=redactor,
    )
    return receipt


async def register_observation_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    expected,
    observation: RequestObservation,
    *,
    initiator: InitiatorBinding,
    redactor: SecretRedactor,
) -> RequestObservationReceipt:
    anchor = await store._anchor(tx, initialized, redactor)
    prior = await retained_request(
        store, tx, initialized, expected.intent.request, expected.initiator, redactor
    )
    if prior is None:
        raise CollaborationUnavailable("Request is unavailable for observation.")
    require_exact_contract(expected, prior.receipt.expected, redactor=redactor)
    operation = observation_operation(expected, observation.key)
    raw = await tx.get("operations", _operation_key(operation))
    if raw is not None:
        receipt = prepare_contract(RequestObservationReceipt, raw, redactor=redactor)
        require_exact_contract(receipt.expected, expected, redactor=redactor)
        require_exact_contract(receipt.intent, observation, redactor=redactor)
        require_exact_contract(receipt.initiator, initiator, redactor=redactor)
        await require_request_event(tx, receipt.event, redactor)
        return receipt
    if observation.retention_until_ms is not None:
        raise CollaborationUnavailable("Timed observation retention is not yet qualified.")
    if observation.after_sequence > anchor.event_sequence:
        raise CollaborationConflict("Observation cursor is beyond the published frontier.")
    if any(item.key == observation.key for item in prior.observations):
        raise CollaborationUnavailable("Observation registration receipt is missing.")
    elif len(prior.observations) >= 32:
        raise CollaborationConflict("Observation registration capacity is exhausted.")
    else:
        current = observation.model_copy(
            update={
                "coverage_sequence": anchor.event_sequence,
                "revision": prior.observation_revision + 1,
            }
        )
        snapshot = prepare_contract(
            RequestSnapshot,
            prior.model_copy(
                update={
                    "observations": (*prior.observations, current),
                    "observation_revision": prior.observation_revision + 1,
                }
            ),
            redactor=redactor,
        )
        event = _event(
            anchor,
            expected,
            prior.receipt.expected.intent.selection.reference,
            "request_observation_registered",
            prior.receipt.event.participants,
            anchor.event_sequence + 1,
            operation=operation,
        )
        receipt = prepare_contract(
            RequestObservationReceipt,
            RequestObservationReceipt(
                operation=operation,
                expected=expected,
                intent=observation,
                initiator=initiator,
                request=prior.receipt.expected.intent.selection.reference,
                observation=current,
                event=event,
            ),
            redactor=redactor,
        )
        await _write_snapshot(
            store,
            tx,
            initialized,
            anchor,
            prior,
            operation=operation,
            snapshot=snapshot,
            receipts=(receipt,),
            events=(event,),
            redactor=redactor,
        )
        return receipt


REQUEST_TERMINAL_BYTES = 4 * MAX_ENVELOPE_BYTES
