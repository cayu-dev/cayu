"""Durable checkpoint records shared by session-operation entrances."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cayu._validation import (
    copy_json_value,
    require_clean_nonblank,
)

_SESSION_OPERATIONS_CHECKPOINT_KEY = "session_operations"


def _session_operation_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    stored = checkpoint.get(_SESSION_OPERATIONS_CHECKPOINT_KEY)
    if stored is None:
        return {"version": 1, "active_operation_id": None, "records": {}}
    if type(stored) is not dict:
        raise ValueError("Session operation checkpoint must be an object.")
    operations = copy_json_value(stored, _SESSION_OPERATIONS_CHECKPOINT_KEY)
    if operations.get("version") != 1:
        raise ValueError("Unsupported session operation checkpoint version.")
    records = operations.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    active_operation_id = operations.get("active_operation_id")
    if active_operation_id is not None:
        require_clean_nonblank(active_operation_id, "active_operation_id")
    return operations


def _active_session_operation_id(checkpoint: dict[str, Any] | None) -> str | None:
    if checkpoint is None or _SESSION_OPERATIONS_CHECKPOINT_KEY not in checkpoint:
        return None
    active_operation_id = _session_operation_state(checkpoint).get("active_operation_id")
    return active_operation_id if type(active_operation_id) is str else None


def _operation_claim_expiry(record: dict[str, Any]) -> datetime | None:
    value = record.get("claim_expires_at")
    if type(value) is not str:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _abandon_expired_session_operation(
    operations: dict[str, Any],
    *,
    now: datetime,
) -> str | None:
    active_operation_id = operations.get("active_operation_id")
    if type(active_operation_id) is not str:
        return None
    records = operations.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    active_record = next(
        (
            record
            for record in records.values()
            if type(record) is dict and record.get("operation_id") == active_operation_id
        ),
        None,
    )
    if active_record is None:
        raise RuntimeError("Active durable session operation record is missing.")
    expiry = _operation_claim_expiry(active_record)
    if active_record.get("status") != "running" or expiry is None or expiry > now:
        return None
    active_record["status"] = "abandoned"
    active_record["abandoned_at"] = now.isoformat()
    active_record["updated_at"] = now.isoformat()
    active_record.pop("claim_expires_at", None)
    operations["active_operation_id"] = None
    return active_operation_id


def _store_session_operation_state(
    checkpoint: dict[str, Any],
    operations: dict[str, Any],
) -> None:
    records = operations.get("records")
    if type(records) is not dict:
        raise ValueError("Session operation checkpoint records must be an object.")
    if records or operations.get("active_operation_id") is not None:
        checkpoint[_SESSION_OPERATIONS_CHECKPOINT_KEY] = operations
    else:
        checkpoint.pop(_SESSION_OPERATIONS_CHECKPOINT_KEY, None)


def _archive_inactive_session_operation_records(
    checkpoint: dict[str, Any],
    *,
    except_idempotency_key: str,
) -> dict[str, dict[str, Any]]:
    """Move inactive operation records out of the live checkpoint."""

    operations = _session_operation_state(checkpoint)
    records = operations["records"]
    archived: dict[str, dict[str, Any]] = {}
    for key, record in list(records.items()):
        if key == except_idempotency_key:
            continue
        if type(record) is not dict:
            raise ValueError("Session operation checkpoint records must be objects.")
        if record.get("status") == "running":
            raise RuntimeError("Checkpoint contains an untracked running session operation.")
        archived[key] = records.pop(key)
    _store_session_operation_state(checkpoint, operations)
    return archived
