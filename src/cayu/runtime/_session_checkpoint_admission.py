"""Admission rules for resumable session checkpoints."""

from __future__ import annotations

from typing import Any

from cayu.approvals.user_input import (
    user_input_lifecycle_authority_from_checkpoint,
)
from cayu.runtime import _prompt_transition, _session_operation_state
from cayu.runtime._interruption_coordinator import (
    _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY,
)
from cayu.runtime.provider_operations import (
    pending_provider_operation_disposition_from_checkpoint,
)
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions.base import (
    _initial_transcript_pending_interaction_id,
)
from cayu.sessions.records import (
    Session,
    SessionStatus,
)
from cayu.vaults.redaction import SecretRedactor


class _ExpiredIncompleteRecoveryClaim(Exception):
    def __init__(self, claim_id: str) -> None:
        self.claim_id = claim_id
        super().__init__("Fence an expired incomplete-session recovery owner.")


_RESUMABLE_SESSION_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}


def _reject_pending_provider_operation_disposition(
    checkpoint: dict[str, Any] | None,
) -> None:
    if pending_provider_operation_disposition_from_checkpoint(checkpoint) is not None:
        raise RuntimeError(
            "Session has an accepted provider-operation resolution pending. "
            "Recover that disposition before starting other session work."
        )


def _reject_unresumable_session_checkpoint(
    session: Session,
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor,
    allow_active_operation: bool = False,
    allow_pending_tool_round: bool = False,
    allowed_initial_transcript_interaction_id: str | None = None,
) -> None:
    _prompt_transition._reject_prepared_prompt_transition_intent(checkpoint)
    _reject_pending_provider_operation_disposition(checkpoint)
    pending_initial_interaction_id = _initial_transcript_pending_interaction_id(checkpoint)
    if pending_initial_interaction_id is not None and (
        allowed_initial_transcript_interaction_id is None
        or pending_initial_interaction_id != allowed_initial_transcript_interaction_id
    ):
        raise RuntimeError(
            "Session setup did not publish its authoritative initial transcript; "
            "resume, fork, and compaction fail closed. Start a new session."
        )
    if (
        pending_approval_reader.pending_approval_from_checkpoint(
            checkpoint,
            redactor=redactor,
            consume_on_rejection=True,
        )
        is not None
    ):
        raise RuntimeError("Session has a pending tool approval.")
    pending_user_input, _ = user_input_lifecycle_authority_from_checkpoint(
        checkpoint,
        redactor=redactor,
        consume_on_rejection=True,
        current_run_epoch=session.run_epoch,
        runtime_session=session,
    )
    if pending_user_input is not None:
        raise RuntimeError("Session is awaiting user input.")
    if not allow_pending_tool_round and (
        pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        is not None
    ):
        raise RuntimeError("Session has a pending tool round.")
    if checkpoint is not None and _PENDING_INTERRUPTION_CASCADE_CHECKPOINT_KEY in checkpoint:
        raise RuntimeError("Session has an incomplete background interruption cascade.")
    if checkpoint is not None and not allow_active_operation:
        operations = _session_operation_state._session_operation_state(checkpoint)
        active_operation_id = operations.get("active_operation_id")
        if active_operation_id is not None:
            raise RuntimeError(f"Session has an active durable operation: {active_operation_id}")
