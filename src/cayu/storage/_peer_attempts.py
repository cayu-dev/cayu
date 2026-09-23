"""Exact peer-attempt transition rules shared by native transaction owners."""

from cayu.collaboration.peer_content import (
    PeerContentAppendRequest,
    PeerContentConflict,
    PeerContentReceipt,
)


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
