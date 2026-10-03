"""Exact peer-attempt transition rules shared by native transaction owners."""

from cayu.collaboration.peer_content import (
    PeerContentAppendRequest,
    PeerContentConflict,
    PeerContentReceipt,
)


def parked_delivery_key(checkpoint, *, session_id: str, instance_id: str):
    """Select a native waiting owner, never infer permission from interruption."""
    from cayu.sessions._session_continuation_store import ROOT_KEY, ContinuationRoot
    from cayu.sessions.checkpoints import decode_runtime_checkpoint

    checkpoint = decode_runtime_checkpoint(checkpoint, session_id=session_id)
    raw = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    if raw is None:
        return None
    root = ContinuationRoot.model_validate(raw)
    if (root.namespace.session_id, root.namespace.session_instance_id) != (session_id, instance_id):
        raise PeerContentConflict("Peer target wait belongs to another incarnation.")
    waiting = [entry for entry in root.entries if entry.state == "WAITING"]
    if len(waiting) > 1:
        raise PeerContentConflict("Peer target has conflicting waiting owners.")
    return None if not waiting else waiting[0].ticket_key


def permits_parked_delivery_append(
    request, checkpoint, record, *, session_id: str, instance_id: str, run_epoch: int
) -> bool:
    """Called with the actual record under the peer append transaction/lock.

    Only inert peer delivery is enabled. Clarifications require an unlatched
    clarification wait. Native producer results must match a selected request
    under live owner provenance, including when its final latch arrived first.
    Neither grants execution or bypasses ordinary peer disclosure authority.
    """
    from cayu.collaboration.peer_content import PeerContentUnavailable
    from cayu.sessions._invocation_lifecycle import (
        _invocation_lifecycle_receipt_from_checkpoint,
    )
    from cayu.sessions._session_continuation import (
        ContinuationRecord,
        continuation_digest,
        continuation_operation_key,
        require_record_writer_generation,
    )
    from cayu.sessions._session_continuation_store import ROOT_KEY, ContinuationRoot
    from cayu.sessions.base import PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY
    from cayu.sessions.checkpoints import decode_runtime_checkpoint

    if record is None:
        raise PeerContentUnavailable("Indexed peer target wait is unavailable.")
    checkpoint = decode_runtime_checkpoint(checkpoint, session_id=session_id)
    if checkpoint is None or PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY in checkpoint:
        raise PeerContentUnavailable("Peer target wait has not settled its native writer.")
    record = ContinuationRecord.model_validate(record)
    root = ContinuationRoot.model_validate(checkpoint[ROOT_KEY])
    indexed = next(
        (
            entry
            for entry in root.entries
            if entry.ticket_key == continuation_operation_key(record.ticket)
        ),
        None,
    )
    if indexed is None or indexed.record_sha256 != continuation_digest(record):
        raise PeerContentConflict("Peer target wait lost its exact native index.")
    if (
        record.ticket.state != "WAITING"
        or record.ticket.session_id != session_id
        or record.ticket.session_instance_id != instance_id
    ):
        raise PeerContentUnavailable("Peer target wait is not available for delivery.")
    from cayu.collaboration._producer_peer_scope import permits_producer_wait_delivery

    if request.wake_policy != "none" or not (
        (record.ticket.service_policy == "clarification" and record.latch is None)
        or permits_producer_wait_delivery(request, record)
    ):
        # A durable waiting owner is not a terminal-session exclusion. In
        # particular, the generic pending-peer worker cannot reconstruct a
        # producer's private provenance: leave its exact attempt pending for
        # the producer owner rather than irreversibly excluding that result.
        raise PeerContentUnavailable("Peer target wait requires its registered delivery owner.")
    require_record_writer_generation(record, run_epoch, allow_released_next_generation=True)
    release = _invocation_lifecycle_receipt_from_checkpoint(
        checkpoint, command_identity=f"release:{session_id}:{instance_id}:{run_epoch - 1}"
    )
    released = release is not None and (
        release.kind.value == "release"
        and release.session_id == session_id
        and release.session_instance_id == instance_id
        and release.result_session.run_epoch == run_epoch
    )
    if not released:
        raise PeerContentUnavailable("Peer target wait has no authenticated writer release.")
    return True


def require_capacity(outstanding: int) -> None:
    from cayu.collaboration.peer_content import (
        PEER_CONTENT_MAX_OUTSTANDING_PER_CONSUMER,
        PeerContentUnavailable,
    )

    if outstanding >= PEER_CONTENT_MAX_OUTSTANDING_PER_CONSUMER:
        raise PeerContentUnavailable("Peer consumer outstanding capacity exhausted.")


def qualify(qualify_target, session) -> None:
    from cayu.collaboration.peer_content import PeerContentUnavailable

    if qualify_target is None:
        raise PeerContentUnavailable("Peer admission requires target provider qualification.")
    result = qualify_target(session.model_copy(deep=True))
    if result is not None:
        import inspect

        if inspect.iscoroutine(result):
            result.close()
        raise PeerContentUnavailable("Peer target qualification must be synchronous.")


def validate_receipt(request, receipt):
    if (
        receipt.operation_key != request.operation_key
        or receipt.append_key != request.append_key
        or receipt.attempt_generation != request.attempt_key.attempt_generation
        or receipt.disclosure != "available"
        or (receipt.status == "appended" and receipt.occurrence != request.occurrence)
    ):
        raise PeerContentConflict()


def replay_or_advance(request, previous, receipt, *, exclusion=False):
    """Return a terminal replay, or validate the one permitted successor."""
    validate_receipt(previous, receipt)
    if previous.operation_key == request.operation_key:
        if previous != request:
            raise PeerContentConflict()
        return (
            receipt.model_copy(update={"replayed": True}) if receipt.status != "pending" else None
        )
    if (
        exclusion
        or receipt.status != "excluded"
        or request.replaces_operation_key != previous.operation_key
        or request.attempt_key.attempt_generation <= previous.attempt_key.attempt_generation
        or request.append_key != previous.append_key
        or request.occurrence != previous.occurrence
        or request.wake_policy != previous.wake_policy
    ):
        raise PeerContentConflict()
    return None


def receiving_cursor(request, previous_receipt, pending_transcript_cursor):
    """Separate the receiving owner's refreshed fence from immutable caller intent."""
    if pending_transcript_cursor is None:
        return request.attempt_key.target_transcript_cursor
    if (
        type(pending_transcript_cursor) is not int
        or pending_transcript_cursor < 0
        or previous_receipt is None
        or previous_receipt.status != "pending"
        or previous_receipt.operation_key != request.operation_key
    ):
        raise PeerContentConflict()
    return pending_transcript_cursor


def historical_replay(request, row):
    if row is None:
        return None
    previous = (
        PeerContentAppendRequest.model_validate_json(row[0])
        if isinstance(row[0], str)
        else PeerContentAppendRequest.model_validate(row[0])
    )
    receipt = (
        PeerContentReceipt.model_validate_json(row[1])
        if isinstance(row[1], str)
        else PeerContentReceipt.model_validate(row[1])
    )
    validate_receipt(previous, receipt)
    if receipt.status == "pending":
        replay_or_advance(request, previous, receipt)
        return None
    if previous != request:
        raise PeerContentConflict()
    return receipt.model_copy(update={"replayed": True})


def exact_read(request, row):
    if row is None:
        return None
    previous = (
        PeerContentAppendRequest.model_validate_json(row[0])
        if isinstance(row[0], str)
        else PeerContentAppendRequest.model_validate(row[0])
    )
    if previous != request:
        raise PeerContentConflict()
    receipt = (
        PeerContentReceipt.model_validate_json(row[1])
        if isinstance(row[1], str)
        else PeerContentReceipt.model_validate(row[1])
    )
    validate_receipt(previous, receipt)
    return receipt
