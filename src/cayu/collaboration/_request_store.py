"""Request transactions. Authentication is supplied by the request coordinator.

These private composition primitives perform no policy callbacks or foreign I/O.
Only the authenticated request coordinator exposes them as a public capability.
"""

from __future__ import annotations

from hashlib import sha256
from uuid import uuid4

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import (
    MAX_ENVELOPE_BYTES,
    MAX_ID_BYTES,
    CollaborationConflict,
    CollaborationContractError,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
)
from cayu.collaboration._namespace_store import load_namespace, require_open_namespace
from cayu.collaboration._permit_store import (
    prepare_permit,
    register_permit_in_transaction,
    settle_permit_in_transaction,
)
from cayu.collaboration._permits import (
    PermitCommand,
    PermitIntent,
    PermitRegistration,
    ReceivingSettlementReceipt,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration.base import CollaborationStore, _Anchor, _key, _Repository, _stored_mode
from cayu.collaboration.mandates import MandateResolution
from cayu.collaboration.participants import (
    CollaborationInitialization,
    CollaborationUnavailable,
    ParticipantAlias,
    ParticipantSnapshot,
)
from cayu.collaboration.requests import (
    MAX_CONTROL_INITIATOR_BYTES,
    CollaborationRequest,
    RequestAdmissionReceipt,
    RequestAlias,
    RequestCommand,
    RequestControl,
    RequestControlCommand,
    RequestControlReceipt,
    RequestDueCursor,
    RequestDuePage,
    RequestEvent,
    RequestIntent,
    RequestObservationReceipt,
    RequestOutcomeReceipt,
    RequestProgressReceipt,
    RequestReceipt,
    RequestRef,
    RequestSelection,
    RequestSnapshot,
)
from cayu.vaults.redaction import SecretRedactor

REQUEST_CONTROL_BYTES = 4 * MAX_ENVELOPE_BYTES


def operation_key(operation: OperationRef) -> tuple[str, int, str]:
    return operation.namespace_incarnation, operation.generation, operation.caller_key


async def require_request_absence(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    operation: OperationRef,
    redactor: SecretRedactor,
) -> None:
    """Missing bytes prove absence only while namespace history is complete."""
    anchor = await store._anchor(tx, initialized, redactor)
    # Preserve the request facade's separate retired-namespace outcome.
    await load_namespace(tx, anchor, operation.generation, redactor)
    result = await store._missing_operation(tx, anchor, operation.generation, redactor)
    if isinstance(result, ExactConflict):
        raise CollaborationConflict("Request namespace conflicts with retained authority.")
    if not isinstance(result, ExactNotFound):
        raise CollaborationUnavailable("Request history is no longer complete.")


async def retained_request(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    request: CollaborationRequest,
    initiator: InitiatorBinding,
    redactor: SecretRedactor,
) -> RequestSnapshot | None:
    anchor = await store._anchor(tx, initialized, redactor)
    if (
        request.sender.owner != initialized.owner
        or request.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Request belongs to another owner.")
    await load_namespace(tx, anchor, request.operation.generation, redactor)
    raw = await tx.get("operations", operation_key(request.operation))
    if raw is None:
        if await tx.get("requests", operation_key(request.operation)) is not None:
            raise CollaborationUnavailable("Request is missing its acceptance receipt.")
        await require_request_absence(store, tx, initialized, request.operation, redactor)
        return None
    if _stored_mode(raw) != "request":
        raise CollaborationConflict("Operation key already has a different family.")
    receipt = prepare_contract(RequestReceipt, raw, redactor=redactor)
    require_exact_contract(request, receipt.expected.intent.request, redactor=redactor)
    require_exact_contract(initiator, receipt.expected.initiator, redactor=redactor)
    if (
        receipt.expected.source != initialized.owner
        or receipt.expected.intent.limits != initialized.binding.limits
    ):
        raise CollaborationConflict("Request receipt belongs to another owner contract.")
    snapshot = prepare_contract(
        RequestSnapshot,
        await tx.get("requests", operation_key(request.operation)),
        redactor=redactor,
    )
    require_exact_contract(receipt, snapshot.receipt, redactor=redactor)
    await require_request_event(tx, receipt.event, redactor)
    for sequence in snapshot.event_sequences:
        event = prepare_contract(
            RequestEvent, await tx.get("request_events", (sequence,)), redactor=redactor
        )
        if event.sequence != sequence or event.request != receipt.event.request:
            raise CollaborationUnavailable("Request event frontier has conflicting evidence.")
    if snapshot.terminal is not None:
        terminal = prepare_contract(
            RequestControlReceipt,
            await tx.get("operations", _key(snapshot.terminal.expected)),
            redactor=redactor,
        )
        require_exact_contract(snapshot.terminal, terminal, redactor=redactor)
        await require_request_event(tx, terminal.event, redactor)
    admission = None
    if snapshot.admission_operation is not None:
        admission = prepare_contract(
            RequestAdmissionReceipt,
            await tx.get("operations", operation_key(snapshot.admission_operation)),
            redactor=redactor,
        )
        require_exact_contract(receipt.expected, admission.command.expected, redactor=redactor)
        if (
            admission.command.operation != snapshot.admission_operation
            or admission.command.generation != snapshot.admission_generation
            or admission.command.decision != snapshot.admission_decision
            or admission.revision > snapshot.revision
            or (snapshot.state == "open" and admission.state != snapshot.admission)
        ):
            raise CollaborationUnavailable("Request admission contradicts its receipt.")
        await require_request_event(tx, admission.event, redactor)
    elif snapshot.admission_generation:
        raise CollaborationUnavailable("Request lacks its admission receipt identity.")
    for progress in snapshot.progress:
        retained = prepare_contract(
            RequestProgressReceipt,
            await tx.get("operations", operation_key(progress.command.operation)),
            redactor=redactor,
        )
        require_exact_contract(progress, retained, redactor=redactor)
        require_exact_contract(receipt.expected, retained.command.expected, redactor=redactor)
        await require_request_event(tx, retained.event, redactor)
    if snapshot.outcome is not None:
        outcome = prepare_contract(
            RequestOutcomeReceipt,
            await tx.get("operations", operation_key(snapshot.outcome.command.operation)),
            redactor=redactor,
        )
        require_exact_contract(snapshot.outcome, outcome, redactor=redactor)
        if outcome.command.outcome == "answered" and (
            admission is None
            or admission.command.source_export is None
            or outcome.command.source_receipt is None
            or outcome.command.source_receipt.expected.intent.request.ref
            != admission.command.source_export
        ):
            raise CollaborationUnavailable("Answer conflicts with its admitted export identity.")
        await require_request_event(tx, outcome.event, redactor)
    for observation in snapshot.observations:
        operation = observation_operation(receipt.expected, observation.key)
        registration = prepare_contract(
            RequestObservationReceipt,
            await tx.get("operations", operation_key(operation)),
            redactor=redactor,
        )
        require_exact_contract(receipt.expected, registration.expected, redactor=redactor)
        require_exact_contract(observation, registration.observation, redactor=redactor)
        if registration.operation != operation:
            raise CollaborationUnavailable("Observation operation identity conflicts.")
        await require_request_event(tx, registration.event, redactor)
    return snapshot


def observation_operation(expected: RequestCommand, key: str) -> OperationRef:
    """Reconstruct the bounded, request-scoped exact registration identity."""
    identity = expected.intent.selection.reference.request_id + ":" + key
    return expected.operation.model_copy(
        update={"caller_key": "observation:" + sha256(identity.encode()).hexdigest()}
    )


async def control_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    expected: RequestControlCommand,
    *,
    authority_expires_at_ms: int,
    settlement: ReceivingSettlementReceipt | None = None,
    redactor: SecretRedactor,
) -> RequestControlReceipt:
    expected = prepare_contract(RequestControlCommand, expected, redactor=redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    if (
        expected.destination != initialized.owner
        or expected.operation.namespace_incarnation != initialized.namespace_incarnation
    ):
        raise CollaborationConflict("Request control belongs to another owner.")
    await load_namespace(tx, anchor, expected.operation.generation, redactor)
    raw = await tx.get("operations", _key(expected))
    if raw is not None:
        if _stored_mode(raw) != "request_control":
            raise CollaborationConflict("Control key already has a different operation.")
        replay = prepare_contract(RequestControlReceipt, raw, redactor=redactor)
        require_exact_contract(expected, replay.expected, redactor=redactor)
        await require_request_event(tx, replay.event, redactor)
        return replay
    original = expected.intent.expected
    prior = await retained_request(
        store, tx, initialized, original.intent.request, original.initiator, redactor
    )
    if prior is None:
        raise CollaborationUnavailable("Request acceptance is unavailable.")
    require_exact_contract(original, prior.receipt.expected, redactor=redactor)
    if prior.state != "open" or prior.revision != expected.intent.expected_revision:
        raise CollaborationConflict("Request already closed or revision changed.")
    if expected.intent.source_receipt is not None:
        if prior.admission_operation is None:
            raise CollaborationConflict("Control source lacks an admitted export identity.")
        admission = prepare_contract(
            RequestAdmissionReceipt,
            await tx.get("operations", operation_key(prior.admission_operation)),
            redactor=redactor,
        )
        if (
            admission.command.source_export is None
            or expected.intent.source_receipt.expected.intent.request.ref
            != admission.command.source_export
        ):
            raise CollaborationConflict("Control source does not match the admitted export.")
    now = await tx.now_ms()
    if type(authority_expires_at_ms) is not int or now >= authority_expires_at_ms:
        raise CollaborationConflict("Control authority expired before election.")
    expired = now >= original.intent.selection.expires_at_ms
    if expected.kind == "expire" and not expired:
        raise CollaborationConflict("Request deadline has not passed.")
    state = "expired" if expired else "cancelled"
    event = RequestEvent(
        id=uuid4().hex,
        sequence=anchor.event_sequence + 2,
        operation=expected.operation,
        request=original.intent.selection.reference,
        type="request_expired" if expired else "request_cancelled",
        participants=prior.receipt.event.participants,
    )
    receipt = prepare_contract(
        RequestControlReceipt,
        RequestControlReceipt(
            expected=expected,
            state=state,
            revision=prior.revision + 1,
            elected_at_ms=now,
            event=event,
        ),
        redactor=redactor,
    )
    updated_snapshot = prepare_contract(
        RequestSnapshot,
        prior.model_copy(
            update={
                "revision": receipt.revision,
                "state": state,
                "terminal": receipt,
                "admission": "closed",
                "delivery": "excluded",
                "next_due_at_ms": 0,
                "event_sequences": (*prior.event_sequences, event.sequence),
            }
        ),
        redactor=redactor,
    )
    # Undecided requests have no admitted producer responsibility. Once an
    # admission decision exists, only the registered receiving owner may prove
    # that the outstanding responsibility is quiescent before cancellation or
    # expiry is committed.
    if prior.admission == "undecided" and prior.delivery == "pending":
        settlement = ReceivingSettlementReceipt(
            expected=prior.permit,
            receiving_owner=initialized.owner,
            receipt_id=event.id,
            outcome="quiescent",
        )
    if settlement is None:
        raise CollaborationUnavailable("Request responsibility needs receiving-owner settlement.")
    await settle_permit_in_transaction(store, tx, initialized, prior.permit, settlement, redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    if not anchor.reserved_operations or not anchor.reserved_events:
        raise CollaborationUnavailable("Request lacks reserved terminal responsibility.")
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "reserved_operations": anchor.reserved_operations - 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "reserved_events": anchor.reserved_events - 1,
                "reserved_bytes": anchor.reserved_bytes - REQUEST_CONTROL_BYTES,
                "retained_bytes": anchor.retained_bytes
                + sum(
                    len(contract_bytes(value, redactor=redactor))
                    for value in (receipt, updated_snapshot, event)
                )
                - len(contract_bytes(prior, redactor=redactor)),
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=False)
    await tx.put("operations", _key(expected), receipt, insert=True)
    await tx.put("requests", _key(original), updated_snapshot, insert=False)
    await tx.put("request_events", (event.sequence,), event, insert=True)
    await tx.put("anchors", (), updated, insert=False)
    return receipt


async def require_request_event(
    tx: _Repository, event: RequestEvent, redactor: SecretRedactor
) -> None:
    retained = prepare_contract(
        RequestEvent, await tx.get("request_events", (event.sequence,)), redactor=redactor
    )
    require_exact_contract(event, retained, redactor=redactor)


def preflight_control(snapshot: RequestSnapshot, redactor: SecretRedactor) -> None:
    """Prove mandatory terminal representations fit before retaining responsibility."""
    expected = snapshot.receipt.expected
    prepare_contract(
        ExactMatch[RequestReceipt],
        {"receipt": snapshot.receipt},
        redactor=redactor,
    )
    prepare_contract(
        RequestDuePage,
        RequestDuePage(
            items=(snapshot,),
            next_cursor=RequestDueCursor(after=2**53 - 1),
            observed_at_ms=2**53 - 1,
        ),
        redactor=redactor,
    )
    # Permitted one-byte control characters expand to six JSON bytes. These
    # schema-owned probes are bounds, never persisted authority or user data.
    maximum = "\x01" * MAX_ID_BYTES
    # Retain the maximum structural shape too: byte slack alone cannot prove
    # nested-node and depth limits for optional authority references.
    reference = ObjectRef(
        owner=expected.destination,
        kind="control",
        object_id="control",
        incarnation="control",
        revision=2**53 - 1,
    )
    initiator = InitiatorBinding(
        issuer=expected.destination,
        principal="control",
        participant=reference,
        mandate=reference,
        invocation_id="control",
        interaction_id="control",
    )
    probe_redactor = SecretRedactor()
    initiator_slack = MAX_CONTROL_INITIATOR_BYTES - len(
        contract_bytes(initiator, redactor=probe_redactor)
    )
    operation = expected.operation.model_copy(update={"caller_key": maximum})
    control = RequestControlCommand(
        operation=operation,
        source=expected.source,
        destination=expected.destination,
        initiator=initiator,
        kind="expire",
        intent=RequestControl(
            operation=operation, expected=expected, expected_revision=1, kind="expire"
        ),
    )
    event = RequestEvent(
        id="f" * 32,  # Runtime-generated UUID hex, not caller-controlled.
        sequence=2**53 - 1,
        operation=operation,
        request=expected.intent.selection.reference,
        type="request_expired",
        participants=snapshot.receipt.event.participants,
    )
    terminal = prepare_contract(
        RequestControlReceipt,
        RequestControlReceipt(
            expected=control,
            state="expired",
            revision=2,
            elected_at_ms=2**53 - 1,
            event=event,
        ),
        # These are schema-owned bound probes, not persisted authority or user data.
        redactor=SecretRedactor(),
    )
    cancelled = terminal.model_copy(
        update={
            "expected": control.model_copy(
                update={
                    "kind": "cancel",
                    "intent": control.intent.model_copy(update={"kind": "cancel"}),
                }
            ),
            "state": "cancelled",
            "elected_at_ms": expected.intent.selection.expires_at_ms - 1,
            "event": event.model_copy(update={"type": "request_cancelled"}),
        }
    )
    for candidate in (terminal, cancelled):
        readback = prepare_contract(
            ExactMatch[RequestControlReceipt],
            {"receipt": candidate},
            redactor=SecretRedactor(),
        )
        closed = prepare_contract(
            RequestSnapshot,
            snapshot.model_copy(
                update={
                    "revision": 2,
                    "state": candidate.state,
                    "admission": "closed",
                    "delivery": "excluded",
                    "terminal": candidate,
                    "next_due_at_ms": 0,
                    "event_sequences": (*snapshot.event_sequences, candidate.event.sequence),
                }
            ),
            redactor=SecretRedactor(),
        )
        # Each representation embeds the new control initiator exactly once;
        # original request authority is already frozen and included verbatim.
        if any(
            len(contract_bytes(value, redactor=probe_redactor)) + initiator_slack
            > MAX_ENVELOPE_BYTES
            for value in (candidate, readback, closed)
        ):
            raise CollaborationContractError("Request cannot retain a bounded terminal control.")
    # The actual source values still pass the workload-secret boundary.
    prepare_contract(RequestSnapshot, snapshot, redactor=redactor)


async def accept_in_transaction(
    store: CollaborationStore,
    tx: _Repository,
    initialized: CollaborationInitialization,
    request: CollaborationRequest,
    initiator: InitiatorBinding,
    *,
    sender: ParticipantSnapshot,
    recipient: ParticipantSnapshot,
    authority: MandateResolution,
    authority_expires_at_ms: int,
    redactor: SecretRedactor,
) -> RequestSnapshot:
    """Commit only selections already authorized by the coordinator's held guard."""
    old = await retained_request(store, tx, initialized, request, initiator, redactor)
    if old is not None:
        return old
    anchor = await store._anchor(tx, initialized, redactor)
    await require_open_namespace(tx, anchor, request.operation, redactor)
    current_sender = await store._participant(tx, request.sender, initialized.owner, redactor)
    current_recipient = await store._participant(
        tx, recipient.reference, initialized.owner, redactor
    )
    require_exact_contract(sender, current_sender, redactor=redactor)
    require_exact_contract(recipient, current_recipient, redactor=redactor)
    if sender.lifecycle != "active" or recipient.lifecycle != "active":
        raise CollaborationConflict("Participant no longer admits new requests.")
    if isinstance(request.target, RequestAlias):
        alias = prepare_contract(
            ParticipantAlias, await tx.get("aliases", (request.target.alias,)), redactor=redactor
        )
        if alias.target != recipient.reference:
            raise CollaborationConflict("Alias changed before request acceptance.")
    elif request.target != recipient.reference:
        raise CollaborationConflict("Recipient selection conflicts with the original target.")
    now = await tx.now_ms()
    if (
        type(authority_expires_at_ms) is not int
        or now >= authority_expires_at_ms
        or now + request.ttl_ms > 2**53 - 1
    ):
        raise CollaborationConflict("Request authority expired before acceptance.")
    reference = RequestRef(owner=initialized.owner, request_id=uuid4().hex, incarnation=uuid4().hex)
    expected = prepare_contract(
        RequestCommand,
        RequestCommand(
            operation=request.operation,
            kind=request.kind,
            source=initialized.owner,
            destination=initialized.owner,
            initiator=initiator,
            intent=RequestIntent(
                request=request,
                authority=authority,
                limits=initialized.binding.limits,
                selection=RequestSelection(
                    reference=reference,
                    sender=sender,
                    recipient=recipient,
                    accepted_at_ms=now,
                    expires_at_ms=now + request.ttl_ms,
                ),
            ),
        ),
        redactor=redactor,
    )

    def child_key() -> OperationRef:
        return request.operation.model_copy(update={"caller_key": uuid4().hex})

    permit_request = PermitRegistration(
        operation=child_key(),
        settlement_operation=child_key(),
        participant=recipient.reference,
        expected_lifecycle_revision=recipient.lifecycle_revision,
        admission_generation=recipient.admission_generation,
        source_operation=request.operation,
        target=ObjectRef(
            owner=reference.owner,
            kind="collaboration_request",
            object_id=reference.request_id,
            incarnation=reference.incarnation,
        ),
        target_state="future",
        effect_scope="request_acceptance",
        required_settlement="quiescence",
    )
    permit = prepare_permit(
        initialized,
        PermitCommand(
            operation=permit_request.operation,
            source=initialized.owner,
            destination=initialized.owner,
            initiator=InitiatorBinding(
                issuer=initialized.owner,
                principal=initialized.owner.owner_id,
                participant=None,
                mandate=None,
                invocation_id=None,
                interaction_id=None,
            ),
            intent=PermitIntent(request=permit_request, limits=initialized.binding.limits),
        ),
        redactor,
    )
    # Composition uses the existing authoritative lifecycle transaction.
    await register_permit_in_transaction(store, tx, initialized, permit, redactor)
    anchor = await store._anchor(tx, initialized, redactor)
    event = RequestEvent(
        id=uuid4().hex,
        sequence=anchor.event_sequence + 1,
        operation=request.operation,
        request=reference,
        type="request_accepted",
        participants=tuple(dict.fromkeys((sender.reference, recipient.reference))),
    )
    receipt = prepare_contract(
        RequestReceipt, RequestReceipt(expected=expected, event=event), redactor=redactor
    )
    snapshot = prepare_contract(
        RequestSnapshot,
        RequestSnapshot(
            receipt=receipt,
            permit=permit,
            revision=1,
            state="open",
            admission="undecided",
            delivery="pending",
            terminal=None,
            next_due_at_ms=expected.intent.selection.accepted_at_ms,
            event_sequences=(event.sequence,),
        ),
        redactor=redactor,
    )
    preflight_control(snapshot, redactor)
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "reserved_operations": anchor.reserved_operations + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "reserved_events": anchor.reserved_events + 1,
                "reserved_bytes": anchor.reserved_bytes + REQUEST_CONTROL_BYTES,
                "retained_bytes": anchor.retained_bytes
                + sum(
                    len(contract_bytes(value, redactor=redactor))
                    for value in (receipt, snapshot, event)
                ),
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=True)
    await tx.put("operations", _key(expected), receipt, insert=True)
    await tx.put("requests", _key(expected), snapshot, insert=True)
    await tx.put("request_events", (event.sequence,), event, insert=True)
    await tx.put("anchors", (), updated, insert=False)
    return snapshot
