"""Deterministic task-failure authority for recovered continuations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Any, Literal

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.runtime.tasks import (
    Task,
    TaskStatus,
    TaskStore,
    TaskTerminalizationReceipt,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    prepare_task_terminalization,
)

_RUNTIME_TASK_FAILURE_MARKER_KEY = "runtime_task_failure"
_RUNTIME_TASK_FAILURE_SCHEMA = "cayu.runtime-task-failure.v2"


@dataclass(frozen=True)
class RuntimeTaskFailureIdentity:
    """Durable identity of one runtime-caught attached-task failure."""

    task_id: str
    session_id: str
    session_instance_id: str
    run_epoch: int
    interaction_id: str
    execution_profile_fingerprint: str
    observed_at: datetime

    def __post_init__(self) -> None:
        for field_name in (
            "task_id",
            "session_id",
            "session_instance_id",
            "interaction_id",
            "execution_profile_fingerprint",
        ):
            require_durable_clean_nonblank(getattr(self, field_name), field_name)
        if len(self.execution_profile_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.execution_profile_fingerprint
        ):
            raise ValueError("execution_profile_fingerprint must be a lowercase SHA-256 digest.")
        if type(self.run_epoch) is not int or self.run_epoch < 0:
            raise ValueError("run_epoch must be a non-negative integer.")
        object.__setattr__(
            self,
            "observed_at",
            normalize_utc_datetime(self.observed_at, "observed_at"),
        )

    @property
    def failure_id(self) -> str:
        material = canonical_durable_json_bytes(
            {
                "schema": _RUNTIME_TASK_FAILURE_SCHEMA,
                "task_id": self.task_id,
                "session_id": self.session_id,
                "session_instance_id": self.session_instance_id,
                "run_epoch": self.run_epoch,
                "interaction_id": self.interaction_id,
                "execution_profile_fingerprint": self.execution_profile_fingerprint,
            },
            "runtime_task_failure_identity",
        )
        return f"runtime-task-failure:v2:{sha256(material).hexdigest()}"


def runtime_task_failure_payload(
    *,
    identity: RuntimeTaskFailureIdentity,
    diagnostic_payload: dict[str, Any],
    session_failure_payload: dict[str, Any],
    turn_completed_payload: dict[str, Any],
) -> dict[str, Any]:
    """Bind a portable diagnostic to replayable runtime failure authority."""

    if type(identity) is not RuntimeTaskFailureIdentity:
        raise TypeError("identity must be a RuntimeTaskFailureIdentity.")
    payload = dict(diagnostic_payload)
    if payload.get("session_id") != identity.session_id:
        raise ValueError("Runtime task failure diagnostic has a conflicting session_id.")
    payload[_RUNTIME_TASK_FAILURE_MARKER_KEY] = {
        "schema": _RUNTIME_TASK_FAILURE_SCHEMA,
        "failure_id": identity.failure_id,
        "session_instance_id": identity.session_instance_id,
        "run_epoch": identity.run_epoch,
        "interaction_id": identity.interaction_id,
        "execution_profile_fingerprint": identity.execution_profile_fingerprint,
        "observed_at": identity.observed_at.isoformat(),
        "session_failure_payload": copy_durable_json_object(
            session_failure_payload,
            "session_failure_payload",
        ),
        "turn_completed_payload": copy_durable_json_object(
            turn_completed_payload,
            "turn_completed_payload",
        ),
    }
    # Apply the task error's durable-value boundary before this reaches a store.
    return copy_durable_json_object(payload, "task.error")


def runtime_task_failure_identity_from_task(
    task: Task,
    *,
    session_id: str,
    session_instance_id: str,
) -> RuntimeTaskFailureIdentity | None:
    """Parse and authenticate the runtime-owned identity retained by a failed task."""

    if (
        type(task) is not Task
        or task.status is not TaskStatus.FAILED
        or task.session_id != session_id
        or task.session_instance_id != session_instance_id
        or type(task.error) is not dict
        or task.error.get("session_id") != session_id
    ):
        return None
    marker = task.error.get(_RUNTIME_TASK_FAILURE_MARKER_KEY)
    if type(marker) is not dict or set(marker) != {
        "schema",
        "failure_id",
        "session_instance_id",
        "run_epoch",
        "interaction_id",
        "execution_profile_fingerprint",
        "observed_at",
        "session_failure_payload",
        "turn_completed_payload",
    }:
        return None
    if (
        marker.get("schema") != _RUNTIME_TASK_FAILURE_SCHEMA
        or marker.get("session_instance_id") != session_instance_id
        or type(marker.get("run_epoch")) is not int
        or type(marker.get("interaction_id")) is not str
        or type(marker.get("execution_profile_fingerprint")) is not str
        or type(marker.get("observed_at")) is not str
        or type(marker.get("session_failure_payload")) is not dict
        or type(marker.get("turn_completed_payload")) is not dict
    ):
        return None
    try:
        identity = RuntimeTaskFailureIdentity(
            task_id=task.id,
            session_id=session_id,
            session_instance_id=session_instance_id,
            run_epoch=marker["run_epoch"],
            interaction_id=marker["interaction_id"],
            execution_profile_fingerprint=marker["execution_profile_fingerprint"],
            observed_at=datetime.fromisoformat(marker["observed_at"]),
        )
    except (TypeError, ValueError):
        return None
    if marker.get("failure_id") != identity.failure_id:
        return None
    return identity


def runtime_task_failure_session_payload(task: Task) -> dict[str, Any] | None:
    """Return the exact terminal session payload retained with a runtime failure."""

    if type(task.error) is not dict:
        return None
    marker = task.error.get(_RUNTIME_TASK_FAILURE_MARKER_KEY)
    if type(marker) is not dict or type(marker.get("session_failure_payload")) is not dict:
        return None
    try:
        return copy_durable_json_object(
            marker["session_failure_payload"],
            "runtime_task_failure.session_failure_payload",
        )
    except (TypeError, ValueError):
        return None


def runtime_task_failure_turn_payload(task: Task) -> dict[str, Any] | None:
    """Return the exact invocation summary retained with a runtime failure."""

    if type(task.error) is not dict:
        return None
    marker = task.error.get(_RUNTIME_TASK_FAILURE_MARKER_KEY)
    if type(marker) is not dict or type(marker.get("turn_completed_payload")) is not dict:
        return None
    try:
        return copy_durable_json_object(
            marker["turn_completed_payload"],
            "runtime_task_failure.turn_completed_payload",
        )
    except (TypeError, ValueError):
        return None


def runtime_task_failure_event_id(
    identity: RuntimeTaskFailureIdentity,
    outcome: Literal[
        "task_failed",
        "interaction_failed",
        "turn_completed",
        "session_failed",
    ],
) -> str:
    """Return one stable event identity for generic task-failure convergence."""

    if type(identity) is not RuntimeTaskFailureIdentity:
        raise TypeError("identity must be a RuntimeTaskFailureIdentity.")
    return f"{identity.failure_id}:{outcome}"


def runtime_task_failure_receipt_matches(
    *,
    receipt: TaskTerminalizationReceipt | None,
    task: Task | None,
    task_worker_id: str,
    task_handoff_id: str | None,
    session_id: str,
    session_instance_id: str,
) -> bool:
    """Validate exact claimed-worker authority for a generic failure replay."""

    if task is None or task.error is None:
        return False
    identity = runtime_task_failure_identity_from_task(
        task,
        session_id=session_id,
        session_instance_id=session_instance_id,
    )
    if identity is None:
        return False
    request, request_sha256 = prepare_task_terminalization(
        TaskTerminalizationRequest(
            task_id=task.id,
            worker_id=task_worker_id,
            handoff_id=task_handoff_id,
            kind=TaskTerminalKind.FAILED,
            error=task.error,
            idempotency_key=runtime_task_terminalization_idempotency_key(
                task_id=task.id,
                session_id=session_id,
                kind=TaskTerminalKind.FAILED,
            ),
        )
    )
    return bool(
        receipt is not None
        and receipt.task_id == task.id
        and receipt.worker_id == task_worker_id
        and receipt.kind is TaskTerminalKind.FAILED
        and receipt.idempotency_key == request.idempotency_key
        and receipt.request_sha256 == request_sha256
        and receipt.task == task
    )


async def load_direct_runtime_task_failure_replay(
    task_store: TaskStore,
    *,
    task_id: str,
    session_id: str,
    session_instance_id: str,
) -> tuple[Task, RuntimeTaskFailureIdentity] | None:
    """Load exact workerless generic failure evidence from a receipt-capable store."""

    task = await task_store.load_task(task_id)
    if task is None or task.error is None:
        return None
    identity = runtime_task_failure_identity_from_task(
        task,
        session_id=session_id,
        session_instance_id=session_instance_id,
    )
    if identity is None:
        return None
    replayed = await load_direct_task_failure_replay(
        task_store,
        task_id=task_id,
        session_id=session_id,
        session_instance_id=session_instance_id,
        expected_error=task.error,
        claimed_terminalization_idempotency_key=runtime_task_terminalization_idempotency_key(
            task_id=task_id,
            session_id=session_id,
            kind=TaskTerminalKind.FAILED,
        ),
    )
    if replayed is None:
        return None
    return replayed, identity


def runtime_task_terminalization_idempotency_key(
    *,
    task_id: str,
    session_id: str,
    kind: TaskTerminalKind,
) -> str:
    """Return the runtime-owned terminalization key shared by live and replay paths."""

    task_id = require_durable_clean_nonblank(task_id, "task_id")
    session_id = require_durable_clean_nonblank(session_id, "session_id")
    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.runtime-task-terminalization.v1",
            "task_id": task_id,
            "session_id": session_id,
            "kind": kind.value,
        },
        "runtime_task_terminalization",
    )
    return f"runtime-task-terminal:v1:{sha256(material).hexdigest()}"


async def load_direct_task_failure_replay(
    task_store: TaskStore,
    *,
    task_id: str,
    session_id: str,
    session_instance_id: str,
    expected_error: dict[str, Any],
    claimed_terminalization_idempotency_key: str,
) -> Task | None:
    """Load exact ownerless failure evidence without adopting a claimed failure.

    Built-in receipt-capable stores commit a claimed task mutation and its receipt
    atomically. Reading the terminal task before the receipt therefore makes an
    absent receipt authoritative: a matching immutable task was terminalized by
    the workerless direct path, not by a worker whose credential was omitted on
    replay. Stores without that atomic receipt contract cannot prove the origin
    after terminalization and fail closed.
    """

    if not task_store.supports_idempotent_terminalization:
        return None
    task = await task_store.load_task(task_id)
    if (
        task is None
        or task.status is not TaskStatus.FAILED
        or task.session_id != session_id
        or task.session_instance_id != session_instance_id
        or task.worker_id is not None
        or task.lease_expires_at is not None
        or task.interrupted_handoff_id is not None
        or task.error != expected_error
    ):
        return None
    receipt = await task_store.load_task_terminalization_receipt(
        task_id,
        claimed_terminalization_idempotency_key,
    )
    if receipt is not None:
        return None
    return task


def provider_operation_task_failure_payload(*, session_id: str) -> dict[str, str]:
    """Return the stable task failure intent for an explicit provider disposition."""

    session_id = require_durable_clean_nonblank(session_id, "session_id")
    return {
        "message": "Provider operation was explicitly failed after recovery.",
        "type": "provider_operation_unavailable",
        "session_id": session_id,
    }


@dataclass(frozen=True)
class ApprovalTaskFailureIdentity:
    """Immutable approval identity retained by task-failure receipts."""

    approval_id: str
    tool_round_id: str
    tool_call_id: str
    resolution_request_digest: str

    def __post_init__(self) -> None:
        for field_name in (
            "approval_id",
            "tool_round_id",
            "tool_call_id",
            "resolution_request_digest",
        ):
            require_durable_clean_nonblank(getattr(self, field_name), field_name)


def approval_task_failure_payload(
    *,
    session_id: str,
    identity: ApprovalTaskFailureIdentity,
) -> dict[str, str]:
    """Return the stable task failure intent for a closed approval continuation."""

    session_id = require_durable_clean_nonblank(session_id, "session_id")
    return {
        "message": "Tool approval continuation failed after durable closure.",
        "type": "tool_approval_continuation_failed",
        "session_id": session_id,
        "approval_id": identity.approval_id,
        "tool_round_id": identity.tool_round_id,
        "tool_call_id": identity.tool_call_id,
        "resolution_request_digest": identity.resolution_request_digest,
    }


def approval_task_terminalization_idempotency_key(
    *,
    task_id: str,
    session_id: str,
    identity: ApprovalTaskFailureIdentity,
) -> str:
    """Return one content-addressed key for the exact approval failure."""

    task_id = require_durable_clean_nonblank(task_id, "task_id")
    session_id = require_durable_clean_nonblank(session_id, "session_id")
    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.approval-continuation-task-failure.v1",
            "task_id": task_id,
            "session_id": session_id,
            "approval_id": identity.approval_id,
            "tool_round_id": identity.tool_round_id,
            "tool_call_id": identity.tool_call_id,
            "resolution_request_digest": identity.resolution_request_digest,
        },
        "approval_continuation_task_failure",
    )
    return f"approval-task-failure:v1:{sha256(material).hexdigest()}"


def approval_task_terminalization_request(
    *,
    task_id: str,
    task_worker_id: str,
    task_handoff_id: str | None,
    session_id: str,
    identity: ApprovalTaskFailureIdentity,
) -> TaskTerminalizationRequest:
    """Build the exact receipt-backed failure used by first-run and replay."""

    return TaskTerminalizationRequest(
        task_id=task_id,
        worker_id=task_worker_id,
        handoff_id=task_handoff_id,
        kind=TaskTerminalKind.FAILED,
        error=approval_task_failure_payload(session_id=session_id, identity=identity),
        idempotency_key=approval_task_terminalization_idempotency_key(
            task_id=task_id,
            session_id=session_id,
            identity=identity,
        ),
    )


def approval_task_failure_receipt_matches(
    *,
    receipt: TaskTerminalizationReceipt | None,
    task: Task | None,
    task_id: str,
    task_worker_id: str,
    task_handoff_id: str | None,
    session_id: str,
    session_instance_id: str,
    identity: ApprovalTaskFailureIdentity,
) -> bool:
    """Validate exact immutable authority for an approval-failure replay."""

    request, request_sha256 = prepare_task_terminalization(
        approval_task_terminalization_request(
            task_id=task_id,
            task_worker_id=task_worker_id,
            task_handoff_id=task_handoff_id,
            session_id=session_id,
            identity=identity,
        )
    )
    return bool(
        receipt is not None
        and task is not None
        and receipt.task_id == task_id
        and receipt.worker_id == task_worker_id
        and receipt.kind is TaskTerminalKind.FAILED
        and receipt.idempotency_key == request.idempotency_key
        and receipt.request_sha256 == request_sha256
        and receipt.task == task
        and task.status is TaskStatus.FAILED
        and task.session_id == session_id
        and task.session_instance_id == session_instance_id
        and task.error == request.error
    )


def approval_failure_event_id(
    identity: ApprovalTaskFailureIdentity,
    outcome: Literal["task_failed", "session_failed"],
) -> str:
    """Return one stable event identity for the approval failure outcome."""

    material = canonical_durable_json_bytes(
        {
            "schema": "cayu.approval-continuation-failure-event.v1",
            "approval_id": identity.approval_id,
            "tool_round_id": identity.tool_round_id,
            "tool_call_id": identity.tool_call_id,
            "resolution_request_digest": identity.resolution_request_digest,
        },
        "approval_continuation_failure_event",
    )
    return f"approval-failure:v1:{sha256(material).hexdigest()}:{outcome}"
