"""Typed registered-application recovery planning and execution contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER, require_durable_clean_nonblank
from cayu.environments import EnvironmentAllocationState
from cayu.runtime.sessions import IncompleteSessionRecoveryAction, PendingActionKind, SessionStatus
from cayu.runtime.tasks import TaskStatus

RECOVERY_PLAN_SCHEMA_VERSION = 1
RECOVERY_PLAN_MAX_ITEMS = 1000
RECOVERY_PLAN_MAX_INSPECTIONS = 10_000
RECOVERY_PLAN_MAX_CONCURRENCY = 32
RECOVERY_PLAN_MAX_ID_CHARS = 512
RECOVERY_PLAN_MAX_CURSOR_BYTES = 8192

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    revalidate_instances="always",
    validate_default=True,
)


def _clean_id(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > RECOVERY_PLAN_MAX_ID_CHARS:
        raise ValueError(f"{field_name} cannot exceed {RECOVERY_PLAN_MAX_ID_CHARS} characters.")
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


class RecoveryPlanSelection(BaseModel):
    """Explicit, bounded selection of durable sessions to inspect."""

    model_config = _MODEL_CONFIG

    session_ids: tuple[str, ...] = Field(default=(), max_length=RECOVERY_PLAN_MAX_ITEMS)
    statuses: frozenset[SessionStatus] = frozenset()
    inactive_for_seconds: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    cursor: str | None = None

    @field_validator("session_ids", mode="before")
    @classmethod
    def validate_session_ids(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise TypeError("session_ids must be a list or tuple.")
        items = cast("list[object] | tuple[object, ...]", value)
        if any(type(item) is not str for item in items):
            raise TypeError("session_ids items must be strings.")
        copied = tuple(_clean_id(cast("str", item), "session_ids") for item in items)
        if len(set(copied)) != len(copied):
            raise ValueError("session_ids must not contain duplicates.")
        return copied

    @field_validator("statuses", mode="before")
    @classmethod
    def validate_statuses(cls, value: object) -> frozenset[SessionStatus]:
        if value is None:
            return frozenset()
        if not isinstance(value, (set, frozenset, list, tuple)):
            raise TypeError("statuses must be a collection of SessionStatus values.")
        return frozenset(
            status if isinstance(status, SessionStatus) else SessionStatus(status)
            for status in value
        )

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_durable_clean_nonblank(value, "cursor")
        if len(value.encode("utf-8")) > RECOVERY_PLAN_MAX_CURSOR_BYTES:
            raise ValueError(f"cursor exceeds its {RECOVERY_PLAN_MAX_CURSOR_BYTES}-byte limit.")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        if bool(self.session_ids) == bool(self.statuses):
            raise ValueError("Select exactly one of session_ids or statuses.")
        if self.session_ids and self.cursor is not None:
            raise ValueError("cursor is valid only for status-based recovery planning.")
        return self


class RecoveryPlanBounds(BaseModel):
    """Hard read and result bounds for one recovery plan page."""

    model_config = _MODEL_CONFIG

    item_limit: StrictInt = Field(default=100, ge=1, le=RECOVERY_PLAN_MAX_ITEMS)
    inspection_limit: StrictInt = Field(
        default=1000,
        ge=1,
        le=RECOVERY_PLAN_MAX_INSPECTIONS,
    )


class RecoveryPlanRequest(BaseModel):
    model_config = _MODEL_CONFIG

    selection: RecoveryPlanSelection
    bounds: RecoveryPlanBounds = Field(default_factory=RecoveryPlanBounds)


class RecoveryRegistrationStatus(StrEnum):
    READY = "ready"
    MISSING_AGENT = "missing_agent"
    MISSING_PROVIDER = "missing_provider"
    MISSING_ENVIRONMENT = "missing_environment"
    INCOMPATIBLE = "incompatible"


class RecoveryRegistrationEvidence(BaseModel):
    """Safe result of comparing durable execution authority with this app."""

    model_config = _MODEL_CONFIG

    status: RecoveryRegistrationStatus
    expected_execution_profile_fingerprint: str | None = None
    validated_execution_profile_fingerprint: str | None = None
    reason_code: str | None = None

    @field_validator(
        "expected_execution_profile_fingerprint",
        "validated_execution_profile_fingerprint",
    )
    @classmethod
    def validate_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_durable_clean_nonblank(value, "reason_code")
        if len(value) > 128:
            raise ValueError("reason_code cannot exceed 128 characters.")
        return value


class RecoveryPlanAction(StrEnum):
    AUTOMATIC_REPAIR = "automatic_repair"
    LEAVE_INTACT = "leave_intact"
    MODEL_MARK_FAILED = "model_mark_failed"
    MODEL_MARK_INTERRUPTED = "model_mark_interrupted"
    TOOL_MARK_COMPLETED = "tool_mark_completed"
    TOOL_MARK_FAILED = "tool_mark_failed"


class RecoveryBlockerCode(StrEnum):
    REGISTRATION_UNAVAILABLE = "registration_unavailable"
    REGISTRATION_INCOMPATIBLE = "registration_incompatible"
    ACTIVE_RECOVERY_CLAIM = "active_recovery_claim"
    ACTIVE_TASK_CLAIM = "active_task_claim"
    MODEL_EFFECT_OUTCOME_UNKNOWN = "model_effect_outcome_unknown"
    TOOL_EFFECT_OUTCOME_UNKNOWN = "tool_effect_outcome_unknown"
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"
    USER_INPUT_REQUIRED = "user_input_required"
    INVALID_DURABLE_STATE = "invalid_durable_state"


class RecoveryPlanBlocker(BaseModel):
    model_config = _MODEL_CONFIG

    code: RecoveryBlockerCode
    action_ref: str | None = None

    @field_validator("action_ref")
    @classmethod
    def validate_action_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean_id(value, "action_ref")


class RecoveryClaimEvidence(BaseModel):
    model_config = _MODEL_CONFIG

    claim_ref: str
    lease_expires_at: datetime

    @field_validator("claim_ref")
    @classmethod
    def validate_claim_ref(cls, value: str) -> str:
        return _clean_id(value, "claim_ref")

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease(cls, value: datetime) -> datetime:
        return _utc(value, "lease_expires_at")


class RecoveryTaskClaimEvidence(BaseModel):
    model_config = _MODEL_CONFIG

    task_ref: str
    status: TaskStatus
    ownership_status: Literal["active", "expired", "unowned", "invalid"] = "active"
    worker_ref: str | None = None
    lease_expires_at: datetime | None = None

    @field_validator("task_ref", "worker_ref")
    @classmethod
    def validate_refs(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean_id(value, info.field_name)

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _utc(value, "lease_expires_at")


class RecoveryPendingActionEvidence(BaseModel):
    model_config = _MODEL_CONFIG

    action_ref: str
    kind: PendingActionKind
    tool_name: str | None = None
    round_ref: str | None = None
    tool_call_ref: str | None = None

    @field_validator("action_ref", "round_ref", "tool_call_ref")
    @classmethod
    def validate_refs(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean_id(value, info.field_name)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "tool_name")


class RecoveryModelStageEvidence(BaseModel):
    model_config = _MODEL_CONFIG

    stage_ref: str
    state: Literal["in_flight", "completed"]
    dispatched: StrictBool
    provider_reattachment_supported: StrictBool
    provider_operation_ref: str | None = None
    reservation_count: StrictInt = Field(ge=0, le=32)

    @field_validator("stage_ref", "provider_operation_ref")
    @classmethod
    def validate_refs(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean_id(value, info.field_name)


class RecoveryEnvironmentEvidence(BaseModel):
    model_config = _MODEL_CONFIG

    allocation_states: tuple[EnvironmentAllocationState, ...] = Field(
        default=(),
        max_length=64,
    )
    completion_finalization_pending: StrictBool = False


class RecoveryInterruptionCascadeEvidence(BaseModel):
    model_config = _MODEL_CONFIG

    attempt_ref: str
    generation: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    claim_ref: str | None = None
    lease_expires_at: datetime | None = None
    failure_recorded: StrictBool = False

    @field_validator("attempt_ref", "claim_ref")
    @classmethod
    def validate_refs(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean_id(value, info.field_name)

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _utc(value, "lease_expires_at")


class RecoveryPlanExecutionEvidence(BaseModel):
    model_config = _MODEL_CONFIG

    plan_ref: str
    item_ref: str
    execution_ref: str
    decision_ref: str
    lease_expires_at: datetime

    @field_validator("plan_ref", "item_ref", "execution_ref", "decision_ref")
    @classmethod
    def validate_refs(cls, value: str, info) -> str:
        return _clean_id(value, info.field_name)

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease(cls, value: datetime) -> datetime:
        return _utc(value, "lease_expires_at")


class RecoveryPlanItem(BaseModel):
    """One secret-free snapshot whose exact state gates later mutation."""

    model_config = _MODEL_CONFIG

    item_id: str
    state_fingerprint: str
    authority_fingerprint: str
    session_id: str
    agent_name: str
    provider_name: str
    environment_name: str | None = None
    status: SessionStatus
    run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    execution_profile_fingerprint: str | None = None
    registration: RecoveryRegistrationEvidence
    recovery_claim: RecoveryClaimEvidence | None = None
    plan_execution: RecoveryPlanExecutionEvidence | None = None
    run_fence_ref: str
    run_operation_ref: str | None = None
    task_claims: tuple[RecoveryTaskClaimEvidence, ...] = Field(default=(), max_length=1000)
    pending_actions: tuple[RecoveryPendingActionEvidence, ...] = Field(default=(), max_length=200)
    active_model_stage: RecoveryModelStageEvidence | None = None
    environment_recovery: RecoveryEnvironmentEvidence
    interruption_cascade: RecoveryInterruptionCascadeEvidence | None = None
    blockers: tuple[RecoveryPlanBlocker, ...] = Field(default=(), max_length=256)
    allowed_actions: tuple[RecoveryPlanAction, ...] = Field(min_length=1, max_length=8)

    @field_validator(
        "item_id",
        "session_id",
        "agent_name",
        "provider_name",
        "run_fence_ref",
    )
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _clean_id(value, info.field_name)

    @field_validator("environment_name", "run_operation_ref")
    @classmethod
    def validate_optional_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean_id(value, info.field_name)

    @field_validator(
        "state_fingerprint",
        "authority_fingerprint",
        "execution_profile_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_actions(self) -> Self:
        if len(set(self.allowed_actions)) != len(self.allowed_actions):
            raise ValueError("allowed_actions must not contain duplicates.")
        if RecoveryPlanAction.LEAVE_INTACT not in self.allowed_actions:
            raise ValueError("Every recovery item must allow leave_intact.")
        return self


class RecoveryPlan(BaseModel):
    """One immutable, content-addressed read-only recovery plan page."""

    model_config = _MODEL_CONFIG

    record_type: Literal["cayu.recovery-plan"] = "cayu.recovery-plan"
    schema_version: Literal[1] = RECOVERY_PLAN_SCHEMA_VERSION
    plan_id: str
    created_at: datetime
    request: RecoveryPlanRequest
    items: tuple[RecoveryPlanItem, ...] = Field(default=(), max_length=RECOVERY_PLAN_MAX_ITEMS)
    inspected_session_count: StrictInt = Field(ge=0, le=RECOVERY_PLAN_MAX_INSPECTIONS)
    next_cursor: str | None = None

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, value: str) -> str:
        return _clean_id(value, "plan_id")

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        return _utc(value, "created_at")

    @field_validator("next_cursor")
    @classmethod
    def validate_next_cursor(cls, value: str | None) -> str | None:
        return RecoveryPlanSelection.validate_cursor(value)


class RecoveryDecision(BaseModel):
    """One operator-selected action, bound to an exact plan item."""

    model_config = _MODEL_CONFIG

    item_id: str
    action: RecoveryPlanAction
    message: str | None = Field(default=None, max_length=4096)

    @field_validator("item_id")
    @classmethod
    def validate_item_id(cls, value: str) -> str:
        return _clean_id(value, "item_id")

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "message")

    @model_validator(mode="after")
    def require_tool_evidence_message(self) -> Self:
        if (
            self.action
            in {
                RecoveryPlanAction.TOOL_MARK_COMPLETED,
                RecoveryPlanAction.TOOL_MARK_FAILED,
            }
            and self.message is None
        ):
            raise ValueError("Tool recovery decisions require an operator evidence message.")
        return self


class RecoveryExecutionRequest(BaseModel):
    model_config = _MODEL_CONFIG

    plan: RecoveryPlan
    execution_id: str
    decisions: tuple[RecoveryDecision, ...] = Field(default=(), max_length=RECOVERY_PLAN_MAX_ITEMS)
    max_concurrency: StrictInt = Field(default=1, ge=1, le=RECOVERY_PLAN_MAX_CONCURRENCY)

    @field_validator("execution_id")
    @classmethod
    def validate_execution_id(cls, value: str) -> str:
        return _clean_id(value, "execution_id")

    @model_validator(mode="after")
    def validate_decisions(self) -> Self:
        item_ids = {item.item_id for item in self.plan.items}
        selected = [decision.item_id for decision in self.decisions]
        if len(set(selected)) != len(selected):
            raise ValueError("decisions must not contain duplicate item ids.")
        unknown = set(selected) - item_ids
        if unknown:
            raise ValueError("decisions contain item ids outside the supplied plan.")
        return self


class RecoveryItemExecutionStatus(StrEnum):
    EXECUTED = "executed"
    LEFT_INTACT = "left_intact"
    BLOCKED = "blocked"
    FAILED = "failed"


class RecoveryItemReceipt(BaseModel):
    model_config = _MODEL_CONFIG

    plan_id: str
    item_id: str
    execution_id: str
    session_id: str
    action: RecoveryPlanAction
    status: RecoveryItemExecutionStatus
    final_session_status: SessionStatus
    final_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    recovery_actions: tuple[IncompleteSessionRecoveryAction, ...] = ()
    event_ids: tuple[str, ...] = Field(default=(), max_length=256)
    receipt_event_id: str | None = None
    replayed: StrictBool = False
    error_code: str | None = None

    @field_validator(
        "plan_id",
        "item_id",
        "execution_id",
        "session_id",
        "receipt_event_id",
        "error_code",
    )
    @classmethod
    def validate_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean_id(value, info.field_name)


class RecoveryReceipt(BaseModel):
    """Bounded aggregate of per-session durable recovery receipts."""

    model_config = _MODEL_CONFIG

    record_type: Literal["cayu.recovery-receipt"] = "cayu.recovery-receipt"
    schema_version: Literal[1] = RECOVERY_PLAN_SCHEMA_VERSION
    plan_id: str
    execution_id: str
    items: tuple[RecoveryItemReceipt, ...] = Field(max_length=RECOVERY_PLAN_MAX_ITEMS)

    @field_validator("plan_id", "execution_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _clean_id(value, info.field_name)


class StaleRecoveryPlanError(RuntimeError):
    """The durable state no longer matches the supplied plan item."""


class RecoveryPlanExecutionFenced(RuntimeError):
    """Another live owner holds the per-session plan execution slot."""


__all__ = [
    "RECOVERY_PLAN_MAX_CONCURRENCY",
    "RECOVERY_PLAN_MAX_INSPECTIONS",
    "RECOVERY_PLAN_MAX_ITEMS",
    "RECOVERY_PLAN_SCHEMA_VERSION",
    "RecoveryBlockerCode",
    "RecoveryClaimEvidence",
    "RecoveryDecision",
    "RecoveryEnvironmentEvidence",
    "RecoveryExecutionRequest",
    "RecoveryInterruptionCascadeEvidence",
    "RecoveryItemExecutionStatus",
    "RecoveryItemReceipt",
    "RecoveryModelStageEvidence",
    "RecoveryPendingActionEvidence",
    "RecoveryPlan",
    "RecoveryPlanAction",
    "RecoveryPlanBlocker",
    "RecoveryPlanBounds",
    "RecoveryPlanExecutionEvidence",
    "RecoveryPlanExecutionFenced",
    "RecoveryPlanItem",
    "RecoveryPlanRequest",
    "RecoveryPlanSelection",
    "RecoveryReceipt",
    "RecoveryRegistrationEvidence",
    "RecoveryRegistrationStatus",
    "RecoveryTaskClaimEvidence",
    "StaleRecoveryPlanError",
]
