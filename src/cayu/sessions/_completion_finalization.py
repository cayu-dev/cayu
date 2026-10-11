"""Saved completion-finalization evidence shared by storage and runtime cleanup.

Reading a marker validates and detaches its data; it does not grant authority
for cleanup or publication. Native stores and runtime owners retain those checks.
"""

from __future__ import annotations

from typing import Any

import cayu.sessions as sessions
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_clean_nonblank,
)
from cayu.sessions import _checkpoint_mutations as checkpoint_mutation_rules
from cayu.sessions._checkpoint_mutations import RuntimePublicationMutation
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions.authority import SessionRunFenced

PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY = "pending_completion_finalization"
_MAX_COMPLETION_FINALIZATION_CHECKPOINT_BYTES = 4 * 1024 * 1024


def pending_completion_finalization_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return private finalization retry state for an invocation's terminal outcome."""

    if checkpoint is None:
        return None
    raw = checkpoint.get(PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY)
    if raw is None:
        return None
    if type(raw) is not dict:
        raise ValueError("Pending completion finalization checkpoint must be an object.")
    marker = copy_durable_json_object(raw, "pending completion finalization")
    if (
        type(marker.get("version")) is not int
        or marker["version"] != 1
        or marker.get("outcome") not in ("completed", "failed", "interrupted")
    ):
        raise ValueError("Pending completion finalization checkpoint has an unsupported format.")
    for field_name in (
        "environment_name",
        "binding_generation_id",
        "execution_profile_fingerprint",
    ):
        value = marker.get(field_name)
        if type(value) is not str:
            raise ValueError(f"Pending completion finalization {field_name} must be a string.")
        require_clean_nonblank(value, field_name)
    disposal_state = marker.get("disposal_state")
    if disposal_state is not None and type(disposal_state) is not dict:
        raise ValueError("Completion disposal state must be an object.")
    task_id = marker.get("task_id")
    if task_id is not None:
        if type(task_id) is not str:
            raise ValueError("Pending completion finalization task_id must be a string.")
        require_clean_nonblank(task_id, "task_id")
    if type(marker.get("binding_state")) is not dict:
        raise ValueError("Pending completion finalization binding state must be an object.")
    if (
        len(
            canonical_durable_json_bytes(
                marker,
                "pending completion finalization",
            )
        )
        > _MAX_COMPLETION_FINALIZATION_CHECKPOINT_BYTES
    ):
        raise ValueError("Pending completion finalization checkpoint exceeds its byte limit.")
    return marker


def _validate_completion_finalization_mutation_shape(value: dict[str, Any]) -> None:
    """Completion may only install one previously absent finalization marker."""

    operations = value.get("operations")
    if type(operations) is not list or len(operations) != 1:
        raise ValueError(
            "A completion-finalization transition requires exactly one checkpoint operation."
        )
    operation = operations[0]
    if (
        type(operation) is not dict
        or operation.get("key") != PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY
        or operation.get("expected_value_digest") is not None
        or operation.get("action") != "set"
        or type(operation.get("value")) is not dict
    ):
        raise ValueError("A completion-finalization transition must install a new pending marker.")


def _prepare_queue_completion_checkpoint_mutation(
    value: dict[str, Any] | None,
    to_status: sessions.SessionStatus,
) -> RuntimePublicationMutation | None:
    if value is None:
        return None
    if to_status is not sessions.SessionStatus.RUNNING:
        raise ValueError("Queue completion checkpointing must retain running status.")
    copied = copy_durable_json_object(value, "checkpoint_mutation")
    _validate_completion_finalization_mutation_shape(copied)
    mutation = RuntimePublicationMutation.model_validate(copied)
    pending_completion_finalization_from_checkpoint(
        {PENDING_COMPLETION_FINALIZATION_CHECKPOINT_KEY: mutation.operations[0].value}
    )
    return mutation


def _apply_queue_completion_checkpoint_mutation(
    session: sessions.Session,
    mutation: RuntimePublicationMutation,
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check durable owner authority under the session lock, then apply the marker CAS."""

    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    marker = mutation.operations[0].value
    if (
        session.status is not sessions.SessionStatus.RUNNING
        or active is None
        or active.session_id != session.id
        or active.run_epoch != session.run_epoch
        or marker["execution_profile_fingerprint"] != active.profile.fingerprint
        or marker["environment_name"] != session.environment_name
    ):
        raise SessionRunFenced("Queue completion finalization lost its active owner authority.")
    updated = checkpoint_mutation_rules._apply_runtime_publication_checkpoint_mutation(
        mutation, checkpoint
    )
    assert updated is not None
    return updated
