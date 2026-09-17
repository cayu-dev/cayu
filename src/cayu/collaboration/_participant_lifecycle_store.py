"""Participant lifecycle election under the same lock as permit registration."""

from __future__ import annotations

from uuid import uuid4

from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._namespace_store import require_open_namespace
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.base import CollaborationStore, _Anchor, _key, _Repository
from cayu.collaboration.lifecycle import (
    LifecycleCommand,
    LifecycleReceipt,
    ParticipantLifecycleChange,
)
from cayu.collaboration.participants import (
    ParticipantEvent,
    ParticipantLifecycleEvidence,
    ParticipantSnapshot,
)
from cayu.vaults.redaction import SecretRedactor

_TRANSITIONS = {
    "active": frozenset({"draining", "disabled", "retired"}),
    "draining": frozenset({"active", "disabled", "retired"}),
    "disabled": frozenset({"active", "retired"}),
    "retired": frozenset(),
}


async def elect_participant_lifecycle(
    store: CollaborationStore,
    tx: _Repository,
    anchor: _Anchor,
    expected: LifecycleCommand,
    redactor: SecretRedactor,
) -> LifecycleReceipt:
    request = expected.intent.request
    assert isinstance(request, ParticipantLifecycleChange)
    await require_open_namespace(tx, anchor, expected.operation, redactor)
    current = await store._participant(
        tx, request.participant, anchor.initialization.owner, redactor
    )
    if (
        current.lifecycle_revision != request.expected_lifecycle_revision
        or request.state not in _TRANSITIONS[current.lifecycle]
    ):
        raise CollaborationConflict("Participant lifecycle revision or transition conflicts.")
    permits = await store._permit_state(tx, request.participant, redactor)
    if request.state == "retired" and permits.outstanding:
        raise CollaborationConflict(
            "Participant retirement requires positive obligation settlement."
        )
    participant = prepare_contract(
        ParticipantSnapshot,
        current.model_copy(
            update={
                "lifecycle": request.state,
                "lifecycle_revision": current.lifecycle_revision + 1,
                "admission_generation": current.admission_generation
                + int(request.state == "active"),
                "covered_permit_frontier": permits.issued_frontier,
                "control_policy": request.control_policy,
            }
        ),
        redactor=redactor,
    )
    history = ParticipantLifecycleEvidence.from_snapshot(participant)
    event = ParticipantEvent(
        id=uuid4().hex,
        sequence=anchor.event_sequence + 1,
        operation=expected.operation,
        type="participant_lifecycle_changed",
        participants=(participant.reference,),
    )
    receipt = prepare_contract(
        LifecycleReceipt,
        LifecycleReceipt(
            expected=expected,
            participant=participant,
            event=event,
        ),
        redactor=redactor,
    )
    charge = sum(
        len(contract_bytes(value, redactor=redactor))
        for value in (
            participant,
            history,
            event,
            receipt,
        )
    ) - len(contract_bytes(current, redactor=redactor))
    updated = prepare_contract(
        _Anchor,
        anchor.model_copy(
            update={
                "operation_count": anchor.operation_count + 1,
                "event_count": anchor.event_count + 1,
                "event_sequence": event.sequence,
                "retained_bytes": anchor.retained_bytes + charge,
            }
        ),
        redactor=redactor,
    )
    await tx.put("participants", (participant.reference.participant_id,), participant, insert=False)
    await tx.put(
        "lifecycle_history",
        (
            participant.reference.participant_id,
            participant.lifecycle_revision,
        ),
        history,
        insert=True,
    )
    await tx.put("operations", _key(expected), receipt, insert=True)
    await tx.put("events", (event.sequence,), event, insert=True)
    from cayu.collaboration._retention_store import release_unused_history

    released = await release_unused_history(
        tx,
        (("lifecycle_history", current.reference.participant_id, current.lifecycle_revision),),
        redactor,
    )
    updated = prepare_contract(
        _Anchor,
        updated.model_copy(
            update={
                "retained_bytes": updated.retained_bytes - released,
            }
        ),
        redactor=redactor,
    )
    require_capacity(updated, ordinary=False)
    await tx.put("anchors", (), updated, insert=False)
    return receipt
