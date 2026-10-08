"""Interrupted-task handoff records, identities and bounded continuation pages."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    revalidate_model_input,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.sessions.invocation import (
    SessionInvocationBinding,
)
from cayu.tasks.cancellation import _task_cancellation_requested, _task_retry_cancellation_requested
from cayu.tasks.records import (
    Task,
    TaskStatus,
    _validate_positive_int,
    copy_task,
)

_TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE = 100


class TaskInterruptedHandoffConflict(ValueError):
    """An interrupted-task handoff conflicts with current durable authority."""


class TaskInterruptedHandoffRequest(BaseModel):
    """Exact authority for releasing one interrupted task's worker ownership."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    worker_id: str
    lease_expires_at: datetime
    session_id: str
    session_instance_id: str
    session_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    handoff_id: str

    @field_validator("task_id", "worker_id", "session_id", "handoff_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("session_instance_id")
    @classmethod
    def validate_session_instance_id(cls, value: str) -> str:
        return SessionInvocationBinding.validate_session_instance_id(value)

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease_expires_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "lease_expires_at")


def _interrupted_task_handoff_request_sha256(
    request: TaskInterruptedHandoffRequest,
) -> str:
    return sha256(
        canonical_durable_json_bytes(
            {
                "schema": "cayu.task-interrupted-handoff.v1",
                **request.model_dump(mode="json", warnings=False),
            },
            "task_interrupted_handoff",
        )
    ).hexdigest()


class TaskInterruptedHandoffReceipt(BaseModel):
    """Immutable commit evidence for one exact interrupted-task handoff."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    request: TaskInterruptedHandoffRequest
    request_sha256: str
    task: Task
    committed_at: datetime

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> object:
        return revalidate_model_input(value, TaskInterruptedHandoffRequest)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest.")
        return value

    @field_validator("task", mode="before")
    @classmethod
    def copy_released_task(cls, value: Task) -> Task:
        if type(value) is not Task:
            raise TypeError("task must be a Task instance.")
        return copy_task(value)

    @field_validator("committed_at")
    @classmethod
    def normalize_committed_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "committed_at")

    @model_validator(mode="after")
    def validate_receipt_task(self) -> TaskInterruptedHandoffReceipt:
        request = self.request
        task = self.task
        if (
            self.request_sha256 != _interrupted_task_handoff_request_sha256(request)
            or task.id != request.task_id
            or task.status is not TaskStatus.RUNNING
            or task.session_id != request.session_id
            or task.session_instance_id != request.session_instance_id
            or task.interrupted_handoff_id != request.handoff_id
            or task.worker_id is not None
            or task.lease_expires_at is not None
        ):
            raise ValueError("Interrupted-task handoff receipt conflicts with its task.")
        return self


class InterruptedTaskContinuationClaimPage(BaseModel):
    """One bounded continuation scan and its optional atomic task claim."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task: Task | None = None
    next_after: tuple[datetime, str] | None = None
    scanned_candidates: StrictInt = Field(
        ge=0,
        le=_TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE,
    )
    rejected_candidates: StrictInt = Field(
        ge=0,
        le=_TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE,
    )
    filtered_candidates: StrictInt = Field(
        default=0,
        ge=0,
        le=_TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE,
    )
    replayed: StrictBool = False
    exhausted: StrictBool

    @field_validator("task", mode="before")
    @classmethod
    def copy_claimed_task(cls, value: Task | None) -> Task | None:
        if value is None:
            return None
        if type(value) is not Task:
            raise TypeError("task must be an exact Task instance.")
        return copy_task(value)

    @field_validator("next_after", mode="before")
    @classmethod
    def validate_next_after(
        cls,
        value: tuple[datetime, str] | None,
    ) -> tuple[datetime, str] | None:
        copied, _ = prepare_interrupted_task_continuation_claim_page(
            after=value,
            limit=1,
        )
        return copied

    @model_validator(mode="after")
    def validate_page(self) -> InterruptedTaskContinuationClaimPage:
        skipped_candidates = self.rejected_candidates + self.filtered_candidates
        if self.replayed:
            if (
                self.task is None
                or self.scanned_candidates != 0
                or skipped_candidates != 0
                or self.next_after != (self.task.created_at, self.task.id)
                or self.exhausted
            ):
                raise ValueError("A continuation claim replay requires exact live claim evidence.")
            return self
        if skipped_candidates > self.scanned_candidates:
            raise ValueError("Rejected and filtered candidates cannot exceed scanned_candidates.")
        if self.scanned_candidates == 0:
            if self.next_after is not None or not self.exhausted:
                raise ValueError("An empty continuation page must be exhausted without a cursor.")
        elif self.next_after is None:
            raise ValueError("A non-empty continuation page requires its last inspected cursor.")
        if self.task is None:
            if skipped_candidates != self.scanned_candidates:
                raise ValueError(
                    "An unclaimed continuation page must classify every inspected row."
                )
        else:
            expected_cursor = (self.task.created_at, self.task.id)
            if self.next_after != expected_cursor:
                raise ValueError("A continuation claim cursor must identify its claimed task.")
            if skipped_candidates >= self.scanned_candidates:
                raise ValueError("A claimed continuation page must contain one accepted row.")
        if not self.exhausted and self.next_after is None:
            raise ValueError("A non-exhausted continuation page requires a cursor.")
        return self


def prepare_interrupted_task_handoff(
    request: TaskInterruptedHandoffRequest,
) -> tuple[TaskInterruptedHandoffRequest, str]:
    """Detach and digest one exact interrupted-task handoff authority."""

    if type(request) is not TaskInterruptedHandoffRequest:
        raise TypeError(
            "Interrupted-task handoffs require TaskInterruptedHandoffRequest instances."
        )
    copied = TaskInterruptedHandoffRequest(
        task_id=request.task_id,
        worker_id=request.worker_id,
        lease_expires_at=request.lease_expires_at,
        session_id=request.session_id,
        session_instance_id=request.session_instance_id,
        session_run_epoch=request.session_run_epoch,
        handoff_id=request.handoff_id,
    )
    request_sha256 = _interrupted_task_handoff_request_sha256(copied)
    return copied, request_sha256


def interrupted_task_handoff_request(
    task: Task,
    *,
    session_run_epoch: int,
) -> TaskInterruptedHandoffRequest:
    """Bind one running task snapshot to an exact interrupted-session handoff."""

    if type(task) is not Task:
        raise TypeError("Interrupted-task handoff requests require a Task instance.")
    if (
        task.worker_id is None
        or task.lease_expires_at is None
        or task.session_id is None
        or task.session_instance_id is None
    ):
        raise TaskInterruptedHandoffConflict(
            "Task does not retain complete interrupted-handoff authority."
        )
    material = {
        "schema": "cayu.task-interrupted-handoff-id.v2",
        "task_id": task.id,
        "worker_id": task.worker_id,
        "lease_expires_at": task.lease_expires_at.isoformat(),
        "session_id": task.session_id,
        "session_instance_id": task.session_instance_id,
        "session_run_epoch": session_run_epoch,
        "prior_handoff_lineage_id": task.interrupted_handoff_id,
    }
    handoff_id = sha256(
        canonical_durable_json_bytes(material, "task_interrupted_handoff_id")
    ).hexdigest()
    return TaskInterruptedHandoffRequest(
        task_id=task.id,
        worker_id=task.worker_id,
        lease_expires_at=task.lease_expires_at,
        session_id=task.session_id,
        session_instance_id=task.session_instance_id,
        session_run_epoch=session_run_epoch,
        handoff_id=handoff_id,
    )


def prepare_interrupted_task_handoff_receipt_lookup(
    task_id: str,
    handoff_id: str,
) -> tuple[str, str]:
    return (
        require_clean_nonblank(task_id, "task_id"),
        require_clean_nonblank(handoff_id, "handoff_id"),
    )


def prepare_interrupted_task_handoff_candidate_page(
    *,
    after: tuple[datetime, str] | None,
    limit: int,
) -> tuple[tuple[datetime, str] | None, int]:
    """Validate and detach one stable bounded expired-handoff page."""

    limit = _validate_positive_int(limit, "limit")
    if limit > _TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE:
        raise ValueError(f"limit must be <= {_TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE}.")
    copied_after: tuple[datetime, str] | None = None
    if after is not None:
        if type(after) is not tuple or len(after) != 2:
            raise TypeError("Interrupted-task handoff cursor must be a timestamp/task-id tuple.")
        lease_expires_at, task_id = after
        if type(lease_expires_at) is not datetime:
            raise TypeError("Interrupted-task handoff cursor timestamp must be a datetime.")
        copied_after = (
            normalize_utc_datetime(lease_expires_at, "after lease_expires_at"),
            require_clean_nonblank(task_id, "after task_id"),
        )
    return copied_after, limit


def prepare_interrupted_task_continuation_claim_page(
    *,
    after: tuple[datetime, str] | None,
    limit: int,
) -> tuple[tuple[datetime, str] | None, int]:
    """Validate and detach one stable bounded continuation-claim page."""

    limit = _validate_positive_int(limit, "scan_limit")
    if limit > _TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE:
        raise ValueError(
            f"scan_limit must be <= {_TASK_INTERRUPTED_HANDOFF_RECOVERY_MAX_PAGE_SIZE}."
        )
    copied_after: tuple[datetime, str] | None = None
    if after is not None:
        if type(after) is not tuple or len(after) != 2:
            raise TypeError(
                "Interrupted-task continuation cursor must be a timestamp/task-id tuple."
            )
        created_at, task_id = after
        if type(created_at) is not datetime:
            raise TypeError("Interrupted-task continuation cursor timestamp must be a datetime.")
        copied_after = (
            normalize_utc_datetime(created_at, "after created_at"),
            require_clean_nonblank(task_id, "after task_id"),
        )
    return copied_after, limit


def _copy_interrupted_task_handoff_receipt(
    receipt: TaskInterruptedHandoffReceipt,
) -> TaskInterruptedHandoffReceipt:
    if type(receipt) is not TaskInterruptedHandoffReceipt:
        raise TypeError(
            "Interrupted-task handoff receipt loads must return "
            "TaskInterruptedHandoffReceipt instances."
        )
    return TaskInterruptedHandoffReceipt(
        request=receipt.request,
        request_sha256=receipt.request_sha256,
        task=copy_task(receipt.task),
        committed_at=receipt.committed_at,
    )


def _replay_interrupted_task_handoff_receipt(
    *,
    request: TaskInterruptedHandoffRequest,
    request_sha256: str,
    receipt: TaskInterruptedHandoffReceipt,
) -> TaskInterruptedHandoffReceipt:
    if receipt.request != request or receipt.request_sha256 != request_sha256:
        raise TaskInterruptedHandoffConflict(
            "Interrupted-task handoff identity conflicts with another request."
        )
    return _copy_interrupted_task_handoff_receipt(receipt)


def new_interrupted_task_continuation_handoff_id() -> str:
    """Create caller-owned authority for one replayable continuation claim."""

    return str(uuid4())


def _interrupted_task_continuation_handoff_id_sha256(handoff_id: str) -> str:
    """Hash one validated claim generation for permanent one-use registration."""

    handoff_id = require_clean_nonblank(handoff_id, "handoff_id")
    return sha256(handoff_id.encode("utf-8")).hexdigest()


def _require_interrupted_task_handoff_authority(
    task: Task,
    request: TaskInterruptedHandoffRequest,
    *,
    now: datetime,
    recover_expired: bool,
) -> None:
    if (
        task.id != request.task_id
        or task.status is not TaskStatus.RUNNING
        or task.session_id != request.session_id
        or task.session_instance_id != request.session_instance_id
        or task.worker_id != request.worker_id
        or task.lease_expires_at != request.lease_expires_at
    ):
        raise TaskInterruptedHandoffConflict(
            "Interrupted-task handoff authority no longer matches the task."
        )
    if _task_cancellation_requested(task) or _task_retry_cancellation_requested(task):
        raise TaskInterruptedHandoffConflict(
            "Task cancellation is still draining under its current owner."
        )
    expired = request.lease_expires_at <= now
    if recover_expired is not expired:
        boundary = "expired" if recover_expired else "live"
        raise TaskInterruptedHandoffConflict(
            f"Interrupted-task handoff requires an exact {boundary} worker lease."
        )
