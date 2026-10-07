"""Exact human-pause closure, serialized against native continuation."""

from cayu.approvals.user_input import user_input_lifecycle_authority_from_checkpoint
from cayu.runtime._producer_release import release_from_snapshot
from cayu.runtime._session_steering import require_steering_receipt, steering_operation_key
from cayu.runtime.session_steering import SessionSteeringConflict, SessionSteeringReceipt
from cayu.sessions._checkpoint_preservation import _invocation_lifecycle_authority_read_scope
from cayu.sessions._pending_approval_reader import pending_approval_from_checkpoint
from cayu.sessions._producer_checkpoint import (
    ROOT_KEY,
    NativeProducerIndex,
    NativeProducerPausedStop,
    _publication_scope,
)
from cayu.sessions.base import SessionOperationPublication, SessionStatus


async def accept_paused_stop(store, command, index, attachment, request, closure_commitment):
    session = await store.load(index.session_id)
    if session is None or session.status is not SessionStatus.INTERRUPTED:
        return None
    key = steering_operation_key(index.session_instance_id, request.interaction_id)
    released = await store._read_native_producer_release(command)
    decision = NativeProducerPausedStop(
        run_epoch=released.run_epoch,
        closure_commitment=closure_commitment,
        release_commitment=released.release_commitment,
    )
    desired = index.model_copy(update={"paused_stop": decision})

    class NotHumanPause(Exception):
        pass

    def publish(current_session, checkpoint, existing):
        current = NativeProducerIndex.model_validate((checkpoint or {}).get(ROOT_KEY))
        if current != index or current_session.status is not SessionStatus.INTERRUPTED:
            raise SessionSteeringConflict()
        current_release = release_from_snapshot(command, current_session, checkpoint, attachment)
        if current_release != released or released.run_epoch != request.expected_run_epoch:
            raise SessionSteeringConflict()
        approval = pending_approval_from_checkpoint(checkpoint)
        pending_input, _ = user_input_lifecycle_authority_from_checkpoint(
            checkpoint,
            current_run_epoch=current_session.run_epoch,
            runtime_session=current_session,
        )
        if approval is None and pending_input is None:
            raise NotHumanPause()
        if current.paused_stop is not None and current.paused_stop != decision:
            raise SessionSteeringConflict()
        assert current.invocation is not None
        receipt = SessionSteeringReceipt(
            request=request,
            execution_profile_fingerprint=current.invocation.profile_commitment.removeprefix(
                "sha256:"
            ),
        )
        if existing is not None and require_steering_receipt(existing) != receipt:
            raise SessionSteeringConflict()
        return SessionOperationPublication(
            checkpoint={**checkpoint, ROOT_KEY: desired.model_dump(mode="json")},
            operation_records={key: receipt.model_dump(mode="json")},
        )

    try:
        with _publication_scope(desired), _invocation_lifecycle_authority_read_scope():
            await store.publish_session_operation(
                index.session_id,
                idempotency_key=key,
                operation_transform=publish,
                events=[],
            )
    except NotHumanPause:
        return None
    receipt = SessionSteeringReceipt.model_validate(
        await store.load_session_operation(index.session_id, key)
    )
    if receipt.request != request:
        raise SessionSteeringConflict()
    return receipt
