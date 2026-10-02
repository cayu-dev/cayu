"""Exact native failure readback for an independently authenticated producer."""

from hashlib import sha256

from cayu._validation import canonical_durable_json_bytes
from cayu.collaboration._producer_contracts import ProducerNativeFailure
from cayu.events import EventType
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions.base import (
    EventOrder,
    EventQuery,
    SessionStatus,
    _invocation_lifecycle_authority_read_scope,
)


async def read_native_failure(store, attachment, index):
    """Read native settlement, never infer failure from a mutable session status.

    The producer attachment pins the session and its immutable operation receipts.
    The event query only locates a receipt; the native owner authenticates its
    complete invocation/profile/incarnation before it becomes production evidence.
    No output, execution authority, effect exclusion or ledger settlement is minted.
    """
    invocation = index.invocation
    if invocation is None:
        return None
    prepared = attachment.command.admission.prepared
    assert prepared is not None
    profile = ExecutionProfileIdentity.model_validate_json(prepared.execution_profile_json)
    if invocation.profile_commitment != "sha256:" + profile.fingerprint:
        raise ValueError("Producer failure profile conflicts.")
    active = ActiveInvocationExecutionProfile(
        session_id=index.session_id,
        interaction_id=invocation.interaction_id,
        run_epoch=invocation.run_epoch,
        profile=profile,
    )
    rows = await store.query_events(
        EventQuery(
            session_id=index.session_id,
            interaction_id=invocation.interaction_id,
            # A failed invocation may retain a pending tool round. Its
            # interaction event then remains paused, while the exact native
            # transition records FAILED. Event classification is not proof.
            event_types=(
                EventType.INTERACTION_FAILED,
                EventType.INTERACTION_PAUSED,
                EventType.INTERACTION_INTERRUPTED,
            ),
            # Locate the latest candidate, then authenticate its exact native
            # receipt and epoch. Prior human pauses are not terminal failures.
            order_by=EventOrder.SEQUENCE_DESC,
            limit=1,
        )
    )
    if not rows:
        return None
    event = rows[0].event
    session = await store.load(index.session_id)
    if session is None or session.instance_id != index.session_instance_id:
        raise ValueError("Producer failure session is unavailable.")
    if session.run_epoch == invocation.run_epoch:
        receipt = await store._load_interaction_transition_receipt_by_event_id(
            index.session_id,
            event_id=event.id,
            expected_session_instance_id=index.session_instance_id,
            expected_active_invocation_profile=active,
        )
    else:
        receipt = await store.load_historical_interaction_settlement(
            index.session_id,
            expected_session_instance_id=index.session_instance_id,
            expected_event=event,
            expected_profile=profile,
        )
    if receipt is None or (
        not receipt.status_changed
        or receipt.transition.event != event
        or event.interaction_id != invocation.interaction_id
    ):
        raise ValueError("Producer failure lacks exact native settlement.")
    from cayu.sessions._invocation_lifecycle import (
        require_invocation_rebind_lineage,
    )

    with _invocation_lifecycle_authority_read_scope():
        checkpoint = await store.load_checkpoint(index.session_id)
    current = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if index.cleanup_receipt is not None:
        assert index.cleanup_receipt.run_epoch is not None
        current = ActiveInvocationExecutionProfile(
            session_id=index.session_id,
            interaction_id=invocation.interaction_id,
            run_epoch=index.cleanup_receipt.run_epoch,
            profile=profile,
        )
    if current is None:
        raise ValueError("Producer failure lacks retained invocation authority.")
    require_invocation_rebind_lineage(
        checkpoint, session_instance_id=index.session_instance_id, original=active, current=current
    )
    require_invocation_rebind_lineage(
        checkpoint,
        session_instance_id=index.session_instance_id,
        original=active,
        current=ActiveInvocationExecutionProfile(
            session_id=index.session_id,
            interaction_id=invocation.interaction_id,
            run_epoch=receipt.session.run_epoch,
            profile=profile,
        ),
    )
    if receipt.session.run_epoch < current.run_epoch:
        return None
    if receipt.session.run_epoch != current.run_epoch:
        raise ValueError("Producer failure belongs to a future invocation.")
    disposition = "failed"
    settlement = receipt.transition.model_dump(mode="json")
    if receipt.transition.to_status is SessionStatus.INTERRUPTED:
        stopped = None
        if index.paused_stop is not None:
            released = await store._read_native_producer_release(attachment.command)
            if (
                released.run_epoch != receipt.session.run_epoch
                or released.run_epoch != index.paused_stop.run_epoch
                or released.release_commitment != index.paused_stop.release_commitment
            ):
                raise ValueError("Paused producer stop release conflicts.")
            stopped = {
                "paused_stop": index.paused_stop.model_dump(mode="json"),
                "release": released.model_dump(mode="json"),
            }
        else:
            stopped = await _stopped_invocation_evidence(store, attachment, index, event)
        if stopped is None:
            if event.type is not EventType.INTERACTION_INTERRUPTED:
                # Human gates and retained provider/effect recovery use PAUSED,
                # even when the containing session has interrupted status.
                return None
            # Stream abandonment and caller cancellation can terminalize without
            # an explicit producer-steering request. The exact transition above
            # authenticates the terminal interaction; independently require the
            # protected lifecycle release (including any recovery rebind chain).
            # This proves native completion, not external-effect settlement.
            released = await store._read_native_producer_release(attachment.command)
            if (
                released.registration != attachment.command
                or released.session_id != index.session_id
                or released.session_instance_id != index.session_instance_id
                or released.interaction_id != invocation.interaction_id
            ):
                raise ValueError("Producer interruption release belongs to another invocation.")
            stopped = {"release": released.model_dump(mode="json")}
        disposition = "stopped"
        settlement = {"transition": settlement, **stopped}
    elif receipt.transition.to_status is not SessionStatus.FAILED:
        # Approval/input and effect-reconciliation pauses remain unresolved,
        # never an invented failure or permission to replay the producer.
        return None
    return ProducerNativeFailure(
        registration=attachment.command,
        interaction_id=invocation.interaction_id,
        run_epoch=receipt.session.run_epoch,
        event_id=event.id,
        settlement_commitment="sha256:"
        + sha256(
            canonical_durable_json_bytes(settlement, "producer_failure_settlement")
        ).hexdigest(),
        disposition=disposition,
    )


async def _stopped_invocation_evidence(store, attachment, index, event):
    """A paused status or a requested stop is not proof that steering completed."""
    from cayu.collaboration._preparation import contract_bytes
    from cayu.runtime._session_steering import steering_operation_key
    from cayu.runtime.session_steering import SessionSteeringReceipt
    from cayu.sessions._invocation_terminal_decision import (
        InvocationTerminalOutcome,
        settled_invocation_terminal_decision_from_checkpoint,
    )
    from cayu.sessions.base import _invocation_lifecycle_authority_read_scope
    from cayu.vaults.redaction import SecretRedactor

    invocation = index.invocation
    assert invocation is not None
    key = steering_operation_key(index.session_instance_id, invocation.interaction_id)
    with _invocation_lifecycle_authority_read_scope():
        checkpoint = await store.load_checkpoint(index.session_id)
    decision = settled_invocation_terminal_decision_from_checkpoint(checkpoint)
    if (
        decision is None
        or decision.outcome is not InvocationTerminalOutcome.INTERRUPTED
        or decision.interruption_request_id != key
    ):
        return None
    raw = await store.load_session_operation(index.session_id, key)
    if raw is None:
        raise ValueError("Producer stop decision lacks native steering evidence.")
    steering = SessionSteeringReceipt.model_validate(raw)
    expected_key = (
        "producer-stop:"
        + sha256(
            contract_bytes(attachment.command.operation, redactor=SecretRedactor())
        ).hexdigest()
    )
    if (
        steering.request.idempotency_key != expected_key
        or steering.request.session_id != index.session_id
        or steering.request.session_instance_id != index.session_instance_id
        or steering.request.interaction_id != invocation.interaction_id
        or decision.session_id != index.session_id
        or decision.session_instance_id != index.session_instance_id
        or steering.request.expected_run_epoch > decision.run_epoch
        or decision.profile_interaction_id != invocation.interaction_id
        or decision.interaction_id != event.interaction_id
        or event.id
        not in (decision.interaction_event_id, decision.predecessor_interaction_event_id)
        or "sha256:" + decision.execution_profile_fingerprint != invocation.profile_commitment
        or steering.execution_profile_fingerprint != decision.execution_profile_fingerprint
    ):
        raise ValueError("Producer stop terminal evidence conflicts with its invocation.")
    from cayu.runtime._producer_lineage import require_producer_epoch

    prepared = attachment.command.admission.prepared
    assert prepared is not None
    profile = ExecutionProfileIdentity.model_validate_json(prepared.execution_profile_json)
    require_producer_epoch(index, checkpoint, profile, steering.request.expected_run_epoch)
    require_producer_epoch(index, checkpoint, profile, decision.run_epoch)
    return {
        "decision": decision.model_dump(mode="json"),
        "steering": steering.model_dump(mode="json"),
    }
