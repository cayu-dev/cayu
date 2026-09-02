"""Stable identities for caller-asserted session-fork source state."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes, copy_json_value
from cayu.runtime._model_completion_publication import (
    LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
)
from cayu.runtime.checkpoints import (
    ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
    AUTOMATIC_RECALL_CHECKPOINT_KEY,
    COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
)

_SESSION_OPERATIONS_CHECKPOINT_KEY = "session_operations"
_SESSION_RUN_OPERATION_CHECKPOINT_KEY = "session_run_operation"
_QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY = "queued_dispatch_terminal_receipts"
_PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY = "prompt_anatomy_transition_intents"
_EGRESS_AUTHORITY_TRANSITION_CHECKPOINT_KEY = "cayu:egress_authority_transition"
# These records coordinate work owned by one source-session incarnation. A child
# inherits resumable application state, never source run, replay, or publication authority.
_SOURCE_OWNED_CHECKPOINT_KEYS = frozenset(
    {
        _SESSION_OPERATIONS_CHECKPOINT_KEY,
        _SESSION_RUN_OPERATION_CHECKPOINT_KEY,
        _QUEUED_DISPATCH_TERMINAL_RECEIPTS_CHECKPOINT_KEY,
        _PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY,
        COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY,
        ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY,
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        LAST_MODEL_STEP_PUBLICATION_CHECKPOINT_KEY,
        AUTOMATIC_RECALL_CHECKPOINT_KEY,
        _EGRESS_AUTHORITY_TRANSITION_CHECKPOINT_KEY,
    }
)


def fork_source_checkpoint_projection(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    """Project child-relevant source state outside runtime authority records."""

    projected = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
    strip_source_owned_fork_checkpoint_state(projected)
    return projected


def strip_source_owned_fork_checkpoint_state(checkpoint: dict[str, Any]) -> None:
    """Remove source-bound runtime records from a detached fork checkpoint."""

    for key in _SOURCE_OWNED_CHECKPOINT_KEYS:
        checkpoint.pop(key, None)


def fork_source_checkpoint_sha256(checkpoint: dict[str, Any] | None) -> str:
    """Hash child-relevant source state while excluding runtime bookkeeping."""

    projected = fork_source_checkpoint_projection(checkpoint)
    return sha256(canonical_durable_json_bytes(projected, "fork_source.checkpoint")).hexdigest()


__all__ = [
    "fork_source_checkpoint_projection",
    "fork_source_checkpoint_sha256",
    "strip_source_owned_fork_checkpoint_state",
]
