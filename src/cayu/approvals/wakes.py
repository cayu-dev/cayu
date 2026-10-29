"""Durable checkpoint state for a session asleep in ``sleep_until``.

A wake is its own pause kind: it lives under ``pending_wake``, opens under the
``wake-open:{wake_id}`` receipt and closes once under ``wake-close:{wake_id}``.
Nothing that serves people (pending user input, the inbox, human attention)
reads it. Closing hands the parked tool round, as ``continuation``, to the shared
pause-resolution machinery, which resumes it with the timer result.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from cayu._validation import canonical_durable_json_bytes
from cayu.approvals.user_input import (
    PendingUserInput,
    SessionWakeTimer,
    copy_pending_user_input,
    pending_user_input_digest,
    pending_user_input_identity,
)

PENDING_WAKE_CHECKPOINT_KEY = "pending_wake"
WAKE_SCHEDULED_INTERRUPTION_TYPE = "wake_scheduled"
WakeCloseTransition = Literal["woke"]


class PendingSessionWake(BaseModel):
    """One parked ``sleep_until`` call and the round it resumes when it wakes."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    continuation: PendingUserInput

    @model_validator(mode="after")
    def validate_continuation(self) -> PendingSessionWake:
        if self.continuation.wake is None:
            raise ValueError("A pending wake must carry its timer.")
        return self

    @property
    def wake_id(self) -> str:
        return self.continuation.input_id

    @property
    def timer(self) -> SessionWakeTimer:
        timer = self.continuation.wake
        assert timer is not None
        return timer


def copy_pending_session_wake(wake: PendingSessionWake) -> PendingSessionWake:
    if type(wake) is not PendingSessionWake:
        raise TypeError("pending wake must be a PendingSessionWake.")
    return PendingSessionWake(continuation=copy_pending_user_input(wake.continuation))


def pending_session_wake_digest(wake: PendingSessionWake) -> str:
    """Bind every field of one durable wake pause."""

    return sha256(
        canonical_durable_json_bytes(wake.model_dump(mode="json"), PENDING_WAKE_CHECKPOINT_KEY)
    ).hexdigest()


def pending_session_wake_identity(wake: PendingSessionWake) -> dict[str, Any]:
    """Return the bounded authority tuple one wake publication binds."""

    identity = pending_user_input_identity(wake.continuation)
    identity.pop("input_id")
    return {
        **identity,
        "wake_id": wake.wake_id,
        "pause_digest": pending_session_wake_digest(wake),
        # The digest of the round's shared pause, which an interruption supersedes.
        "continuation_pause_digest": pending_user_input_digest(wake.continuation),
    }


def pending_session_wake_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> PendingSessionWake | None:
    """Read the wake pause, refusing a malformed one instead of ignoring it."""

    raw = None if checkpoint is None else checkpoint.get(PENDING_WAKE_CHECKPOINT_KEY)
    if raw is None:
        return None
    if type(raw) is not dict:
        raise ValueError("Pending wake checkpoint is malformed.")
    return PendingSessionWake.model_validate(raw)


def public_pending_wake_payload(wake: PendingSessionWake) -> dict[str, Any]:
    """Bounded, content-free description: which call sleeps and until when."""

    timer = wake.timer
    return {
        "wake_id": wake.wake_id,
        "tool_call_id": wake.continuation.tool_call_id,
        "tool_name": wake.continuation.tool_name,
        "deadline": timer.deadline.isoformat().replace("+00:00", "Z"),
        "task_id": timer.task_id,
    }


def require_no_pending_wake(checkpoint: dict[str, Any] | None) -> None:
    """Refuse work that would bypass a sleeping tool call."""

    raw = None if checkpoint is None else checkpoint.get(PENDING_WAKE_CHECKPOINT_KEY)
    if raw is not None:
        raise RuntimeError(
            "Session is asleep in sleep_until; it continues when its wake task runs. "
            "Interrupt it to stop waiting."
        )
