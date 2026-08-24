"""Stable identities for coordinator-frozen fork source state."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from cayu._validation import canonical_durable_json_bytes, copy_json_value
from cayu.runtime.checkpoints import COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY

_SESSION_OPERATIONS_CHECKPOINT_KEY = "session_operations"
_PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY = "prompt_anatomy_transition_intents"


def fork_source_checkpoint_sha256(checkpoint: dict[str, Any] | None) -> str:
    """Hash source checkpoint state while excluding coordinator bookkeeping."""

    projected = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
    projected.pop(_SESSION_OPERATIONS_CHECKPOINT_KEY, None)
    projected.pop(_PROMPT_ANATOMY_TRANSITION_INTENTS_CHECKPOINT_KEY, None)
    projected.pop(COMPLETION_RESULT_EVENT_PUBLICATIONS_CHECKPOINT_KEY, None)
    return sha256(canonical_durable_json_bytes(projected, "fork_source.checkpoint")).hexdigest()


__all__ = ["fork_source_checkpoint_sha256"]
