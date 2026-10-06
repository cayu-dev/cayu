"""Saved completion-finalization evidence shared by storage and runtime cleanup.

Reading a marker validates and detaches its data; it does not grant authority
for cleanup or publication. Native stores and runtime owners retain those checks.
"""

from __future__ import annotations

from typing import Any

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_clean_nonblank,
)

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
