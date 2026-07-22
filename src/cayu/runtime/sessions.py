from __future__ import annotations

import asyncio
import base64
import heapq
import json
import math
from abc import ABC, abstractmethod
from bisect import bisect_left, bisect_right
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Literal
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
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - Pydantic needs this at runtime.

from cayu._validation import (
    JsonUtf8SizeCounter,
    compact_json_utf8_size,
    copy_json_object,
    copy_json_value,
    copy_label_map,
    json_utf8_size_within_limit,
    require_clean_nonblank,
    require_durable_json_text,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.core.events import Event, EventType, copy_event
from cayu.core.messages import Message, MessageRole, ThinkingPart, copy_message, detach_message
from cayu.core.thinking import ThinkingConfig
from cayu.runtime.aggregates import (
    EXACT_AGGREGATE,
    AggregateAccuracy,
    AggregateAccuracyKind,
    BoundedUsagePricingInputAccumulator,
    UsageAggregateBreakdown,
    UsageAggregateGroup,
    UsageAggregateRemainder,
    UsageAggregateTotals,
    UsageRollupStoreResult,
    add_aggregate_usage,
    aggregate_usage_metrics_from_event_payload,
    normalize_aggregate_event_timestamp,
)
from cayu.runtime.approvals import (
    ResolutionActor,
    copy_resolution_actor,
    resolution_actor_payload,
)
from cayu.runtime.budgets import (
    BudgetLimit,
    SessionBudgetInspection,
    copy_request_budget_limits,
    is_budget_inspection_event,
    project_budget_inspection_event,
    session_budget_inspection,
)
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import StructuredOutputSpec, copy_structured_output_spec
from cayu.runtime.usage import (
    SessionUsageSummary,
    UsageMetrics,
    count_model_steps_with_usage,
    project_usage_inspection_event,
    session_usage_summary,
)


class SessionStatusConflict(ValueError):
    """A session status transition was rejected because the session was not in an
    allowed source status (e.g. resuming a session another worker is already
    running). Subclasses ``ValueError`` so existing ``except ValueError`` handlers
    keep working; callers that need to react specifically (e.g. requeue) catch this.
    """


class SessionRunFenced(RuntimeError):
    """A durable write was rejected because its run no longer owns the session epoch."""


class PersistedEventSideEffectClaimLost(RuntimeError):
    """A side-effect acknowledgement lost ownership to a replacement claim."""


class SessionQueuedMessagesPending(RuntimeError):
    """Terminalization lost a race to durable queued session input."""


_SESSION_RUN_FENCES: ContextVar[dict[str, int] | None] = ContextVar(
    "cayu_session_run_fence",
    default=None,
)


def _current_session_run_epoch(session_id: str) -> int | None:
    fences = _SESSION_RUN_FENCES.get()
    return None if fences is None else fences.get(session_id)


def _activate_session_run_fence(session: Session) -> None:
    fences = dict(_SESSION_RUN_FENCES.get() or {})
    fences[session.id] = session.run_epoch
    _SESSION_RUN_FENCES.set(fences)


def _deactivate_session_run_fence(session_id: str) -> None:
    fences = _SESSION_RUN_FENCES.get()
    if fences is None or session_id not in fences:
        return
    remaining = dict(fences)
    remaining.pop(session_id)
    _SESSION_RUN_FENCES.set(remaining or None)


def _assert_session_run_epoch(session_id: str, session: Session) -> None:
    _assert_session_run_epoch_value(session_id, session.run_epoch)


def _assert_session_run_epoch_value(session_id: str, current_run_epoch: int) -> None:
    expected = _current_session_run_epoch(session_id)
    if expected is not None and current_run_epoch != expected:
        raise SessionRunFenced(
            f"Session run epoch no longer owns {session_id}: expected {expected}, "
            f"current {current_run_epoch}."
        )


class SessionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTING = "interrupting"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


SESSION_MESSAGE_CONTENT_MAX_BYTES = 65_536
SESSION_MESSAGE_DELIVERY_BATCH_LIMIT = 100
SESSION_RUNTIME_METADATA_KEYS = frozenset({"subagent"})
SESSION_RUNTIME_METADATA_PREFIX = "cayu:"


def is_runtime_owned_session_metadata_key(key: str) -> bool:
    """Return whether a session metadata entry is owned by Cayu's runtime."""

    return key in SESSION_RUNTIME_METADATA_KEYS or key.startswith(SESSION_RUNTIME_METADATA_PREFIX)


def copy_session_user_metadata(replacement: dict[str, Any]) -> dict[str, Any]:
    """Validate and detach a complete user-authored metadata replacement."""

    copied_replacement = copy_json_value(replacement, "metadata")
    if type(copied_replacement) is not dict:
        raise TypeError("Session metadata must be an object.")
    require_durable_json_text(copied_replacement, "metadata")
    for key in copied_replacement:
        if is_runtime_owned_session_metadata_key(key):
            raise ValueError(
                f"Session metadata key {key!r} is runtime-owned and cannot be replaced."
            )
    return copied_replacement


def replace_session_user_metadata(
    current: dict[str, Any],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    """Combine validated user metadata with the current runtime-owned entries.

    Callers perform this merge while holding the store's session lock. Runtime
    metadata participates in recovery and policy enforcement, so it must be read
    and retained in the same transaction that writes the replacement.
    """

    if type(current) is not dict:
        raise TypeError("Current session metadata must be an object.")
    if type(replacement) is not dict:
        raise TypeError("Session metadata replacement must be an object.")
    if any(is_runtime_owned_session_metadata_key(key) for key in replacement):
        raise ValueError("Session user metadata replacement contains a runtime-owned key.")
    runtime_metadata = {
        key: copy_json_value(value, f"current_metadata.{key}")
        for key, value in current.items()
        if is_runtime_owned_session_metadata_key(key)
    }
    return {**replacement, **runtime_metadata}


class SessionMessageDeliveryMode(StrEnum):
    NEXT_TURN = "next_turn"
    ON_IDLE = "on_idle"


class SessionMessageQueueStatus(StrEnum):
    QUEUED = "queued"
    DELIVERED = "delivered"


class SessionDebugState(StrEnum):
    NEEDS_ATTENTION = "needs_attention"
    SESSION_FAILURE = "session_failure"
    TOOL_ISSUE = "tool_issue"
    INTERRUPTION = "interruption"


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    agent_name: str
    messages: list[Message]
    # Optional caller-provided id for a new session. It must be unique.
    session_id: str | None = None
    parent_session_id: str | None = None
    # Durable budget/accounting identity shared by related sessions. Defaults to
    # task_id when present, otherwise session_id. Forks inherit the source value.
    causal_budget_id: str | None = None
    task_id: str | None = None
    task_worker_id: str | None = None
    # Per-run provider override. Resolution order for new sessions:
    # request.provider_name -> agent spec provider_name -> model-pattern route ->
    # app default provider.
    provider_name: str | None = None
    # Per-run model override for new sessions. Resume keeps the stored session
    # model unless ResumeRequest.model is set.
    model: str | None = None
    environment_name: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("labels", mode="before")
    @classmethod
    def copy_request_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels", allow_reserved=False)

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")

    @field_validator("agent_name")
    @classmethod
    def validate_nonblank_agent_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "session_id",
        "parent_session_id",
        "causal_budget_id",
        "task_id",
        "task_worker_id",
        "provider_name",
        "model",
        "environment_name",
    )
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_task_worker_handoff(self) -> RunRequest:
        if self.task_worker_id is not None and self.task_id is None:
            raise ValueError("RunRequest.task_worker_id requires task_id.")
        return self


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    session_id: str
    messages: list[Message]
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    max_steps: StrictInt = Field(default=16, ge=1, le=256)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    retry_policy: RetryPolicy | None = None
    structured_output: StructuredOutputSpec | None = None
    thinking: ThinkingConfig | None = None
    loop_policies: SkipJsonSchema[tuple[LoopPolicy, ...]] = Field(
        default_factory=tuple,
        exclude=True,
    )

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        copied_messages = [copy_message(message) for message in value]
        if not copied_messages:
            raise ValueError("ResumeRequest messages cannot be empty.")
        return copied_messages

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("structured_output")
    @classmethod
    def copy_structured_output(
        cls,
        value: StructuredOutputSpec | None,
    ) -> StructuredOutputSpec | None:
        return copy_structured_output_spec(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("loop_policies", mode="before")
    @classmethod
    def copy_loop_policies(cls, value) -> tuple[LoopPolicy, ...]:
        return validate_loop_policies(value, field_name="loop_policies")

    @field_validator("session_id", "model")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class CompactSessionRequest(BaseModel):
    """Request an explicit, application-owned compaction of durable session context."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    idempotency_key: str = Field(max_length=256)
    expected_run_epoch: StrictInt = Field(ge=0)
    expected_transcript_cursor: StrictInt = Field(ge=0)
    reason: Literal["application_requested"] = "application_requested"
    instructions: str | None = Field(default=None, max_length=4096)
    limits: RunLimits = Field(default_factory=RunLimits)
    budget_limits: tuple[BudgetLimit, ...] = Field(default_factory=tuple)
    requested_by: ResolutionActor | None = None

    @field_validator("session_id", "idempotency_key")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        value = require_clean_nonblank(value, info.field_name)
        value = require_unicode_scalar_text(value, info.field_name)
        if "\x00" in value:
            raise ValueError(f"`{info.field_name}` must not contain NUL characters.")
        return value

    @field_validator("instructions")
    @classmethod
    def validate_optional_instructions(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, "instructions")

    @field_validator("limits")
    @classmethod
    def copy_limits(cls, value: RunLimits) -> RunLimits:
        return copy_run_limits(value)

    @field_validator("budget_limits", mode="before")
    @classmethod
    def copy_budget_limits(cls, value) -> tuple[BudgetLimit, ...]:
        return copy_request_budget_limits(value)

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

    @model_validator(mode="after")
    def validate_durable_text(self) -> CompactSessionRequest:
        require_durable_json_text(
            self.model_dump(mode="json", exclude={"requested_by"}),
            "CompactSessionRequest",
        )
        if self.requested_by is not None:
            require_durable_json_text(
                self.requested_by.model_dump(mode="json", exclude={"claims"}),
                "CompactSessionRequest.requested_by",
            )
        return self


class EnqueueSessionMessageRequest(BaseModel):
    """Submit durable user steering for an active session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    idempotency_key: str = Field(max_length=256)
    content: str = Field(max_length=SESSION_MESSAGE_CONTENT_MAX_BYTES)
    delivery_mode: SessionMessageDeliveryMode
    requested_by: ResolutionActor | None = None

    @field_validator("session_id", "idempotency_key")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = require_nonblank(value, "content")
        require_durable_json_text(value, "content")
        if len(value.encode("utf-8")) > SESSION_MESSAGE_CONTENT_MAX_BYTES:
            raise ValueError(
                "content exceeds the maximum encoded size of "
                f"{SESSION_MESSAGE_CONTENT_MAX_BYTES} bytes."
            )
        return value

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

    @model_validator(mode="after")
    def validate_durable_text(self) -> EnqueueSessionMessageRequest:
        require_durable_json_text(
            self.model_dump(mode="json", exclude={"requested_by"}),
            "EnqueueSessionMessageRequest",
        )
        if self.requested_by is not None:
            require_durable_json_text(
                self.requested_by.model_dump(mode="json", exclude={"claims"}),
                "EnqueueSessionMessageRequest.requested_by",
            )
        return self


class SessionQueuedMessage(BaseModel):
    """One durable queued user message and its delivery state."""

    model_config = ConfigDict(extra="forbid")

    queue_id: str
    session_id: str
    idempotency_key: str
    content: str
    delivery_mode: SessionMessageDeliveryMode
    status: SessionMessageQueueStatus
    ordering_key: StrictInt = Field(ge=1)
    accepted_run_epoch: StrictInt = Field(ge=0)
    accepted_transcript_cursor: StrictInt = Field(ge=0)
    accepted_event_id: str
    accepted_at: datetime
    requested_by: ResolutionActor | None = None
    delivered_run_epoch: StrictInt | None = Field(default=None, ge=0)
    delivered_transcript_cursor: StrictInt | None = Field(default=None, ge=0)
    delivered_event_id: str | None = None
    delivered_at: datetime | None = None

    @field_validator(
        "queue_id",
        "session_id",
        "idempotency_key",
        "content",
        "accepted_event_id",
        "delivered_event_id",
    )
    @classmethod
    def validate_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)


class EnqueueSessionMessageResult(BaseModel):
    """Typed enqueue result, including the durable acceptance event."""

    model_config = ConfigDict(extra="forbid")

    message: SessionQueuedMessage
    event: Event
    replayed: StrictBool = False

    @field_validator("message")
    @classmethod
    def copy_message(cls, value: SessionQueuedMessage) -> SessionQueuedMessage:
        if type(value) is not SessionQueuedMessage:
            raise TypeError("message must be a SessionQueuedMessage.")
        return value.model_copy(deep=True)

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)


class SessionMessageDeliveryBatch(BaseModel):
    """One bounded atomic queue-delivery batch at a fixed eligibility cutoff."""

    model_config = ConfigDict(extra="forbid")

    messages: tuple[SessionQueuedMessage, ...] = Field(default_factory=tuple)
    events: tuple[Event, ...] = Field(default_factory=tuple)
    eligible_through: StrictInt = Field(ge=0)
    has_more: StrictBool = False

    @field_validator("messages", mode="before")
    @classmethod
    def copy_messages(cls, value) -> tuple[SessionQueuedMessage, ...]:
        return tuple(message.model_copy(deep=True) for message in value)

    @field_validator("events", mode="before")
    @classmethod
    def copy_events(cls, value) -> tuple[Event, ...]:
        return tuple(copy_event(event) for event in value)


class InterruptSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    requested_by: ResolutionActor | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)


class ForkSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_session_id: str
    session_id: str | None = None
    agent_name: str | None = None
    model: str | None = None
    environment_name: str | None = None
    transcript_cursor: StrictInt | None = Field(default=None, ge=0)
    copy_checkpoint: StrictBool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_session_id",
        "session_id",
        "agent_name",
        "model",
        "environment_name",
    )
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_request_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class SessionIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    model: str
    runtime_name: str = "cayu"
    runtime_version: str | None = None

    @field_validator("provider_name", "model", "runtime_name")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("runtime_version")
    @classmethod
    def validate_optional_runtime_version(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class Session(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # SessionStore implementations may set this from RunRequest.session_id.
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None = None
    causal_budget_id: str
    runtime_name: str = "cayu"
    runtime_version: str | None = None
    environment_name: str | None = None
    status: SessionStatus = SessionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    run_epoch: StrictInt = Field(default=0, ge=0)
    labels: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def default_causal_budget_id(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            session_id = value.get("id")
            if session_id is None:
                session_id = str(uuid4())
                value["id"] = session_id
            if value.get("causal_budget_id") is None and isinstance(session_id, str):
                value["causal_budget_id"] = session_id
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator(
        "id",
        "agent_name",
        "provider_name",
        "model",
        "causal_budget_id",
        "runtime_name",
    )
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("parent_session_id", "environment_name", "runtime_version")
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class SessionStateSnapshot(BaseModel):
    """Bounded session state for status polling and control-plane coordination."""

    model_config = ConfigDict(extra="forbid")

    id: str
    status: SessionStatus
    updated_at: datetime
    last_activity_at: datetime

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "id")

    @field_validator("updated_at", "last_activity_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)


_INTERRUPTION_CASCADE_ATTEMPT_ID_MAX_CHARS = 128
_INTERRUPTION_CASCADE_CLAIM_ID_MAX_CHARS = 128
_INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS = 64
_INTERRUPTION_CASCADE_GENERATION_MAX_CHARS = 32
_INTERRUPTION_CASCADE_MISSING = object()


def _project_interruption_cascade_marker_fields(
    marker_type: str | None,
    field_types: Mapping[str, str | None],
    field_values: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the bounded marker view used by control-plane status polling."""

    if marker_type is None or marker_type == "null":
        return None
    if marker_type != "object":
        return {}

    projected: dict[str, Any] = {}

    def required_string(field: str, max_chars: int) -> None:
        value = field_values.get(field)
        if (
            field_types.get(field) == "string"
            and type(value) is str
            and value.strip()
            and len(value) <= max_chars
        ):
            projected[field] = value
        else:
            projected[field] = None

    def optional_string(field: str, max_chars: int) -> None:
        field_type = field_types.get(field)
        if field_type is None or field_type == "null":
            return
        value = field_values.get(field)
        if type(value) is str and field_type == "string" and len(value) <= max_chars:
            projected[field] = value
        else:
            projected[field] = 0

    required_string("attempt_id", _INTERRUPTION_CASCADE_ATTEMPT_ID_MAX_CHARS)
    projected["interrupt_payload"] = (
        {} if field_types.get("interrupt_payload") == "object" else None
    )

    generation_type = field_types.get("generation")
    if generation_type is not None:
        generation = field_values.get("generation")
        try:
            if (
                generation_type not in {"integer", "number"}
                or type(generation) is not str
                or len(generation) > _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS
                or not generation.lstrip("-").isdigit()
            ):
                raise ValueError
            projected["generation"] = int(generation)
        except ValueError:
            projected["generation"] = None

    failure_type = field_types.get("failure_recorded")
    if failure_type is not None:
        failure = field_values.get("failure_recorded")
        if failure_type == "boolean" and type(failure) is bool:
            projected["failure_recorded"] = failure
        else:
            projected["failure_recorded"] = None

    optional_string("claim_id", _INTERRUPTION_CASCADE_CLAIM_ID_MAX_CHARS)
    optional_string("claim_expires_at", _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS)
    optional_string("created_at", _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS)
    return projected


def _project_interruption_cascade_marker(marker: Any) -> dict[str, Any] | None:
    if marker is _INTERRUPTION_CASCADE_MISSING or marker is None:
        return None
    if type(marker) is not dict:
        return {}

    field_types: dict[str, str | None] = {}
    field_values: dict[str, Any] = {}
    for field in (
        "attempt_id",
        "interrupt_payload",
        "generation",
        "failure_recorded",
        "claim_id",
        "claim_expires_at",
        "created_at",
    ):
        if field not in marker:
            field_types[field] = None
            continue
        value = marker[field]
        if value is None:
            field_types[field] = "null"
        elif type(value) is str:
            field_types[field] = "string"
            max_chars = {
                "attempt_id": _INTERRUPTION_CASCADE_ATTEMPT_ID_MAX_CHARS,
                "claim_id": _INTERRUPTION_CASCADE_CLAIM_ID_MAX_CHARS,
                "claim_expires_at": _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS,
                "created_at": _INTERRUPTION_CASCADE_TIMESTAMP_MAX_CHARS,
            }.get(field, _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS)
            field_values[field] = value[: max_chars + 1]
        elif type(value) is bool:
            field_types[field] = "boolean"
            field_values[field] = value
        elif type(value) is int:
            field_types[field] = "integer"
            field_values[field] = str(value)[: _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS + 1]
        elif type(value) is float:
            field_types[field] = "number"
            field_values[field] = str(value)[: _INTERRUPTION_CASCADE_GENERATION_MAX_CHARS + 1]
        elif type(value) is dict:
            field_types[field] = "object"
        elif type(value) is list:
            field_types[field] = "array"
        else:
            field_types[field] = "other"
    return _project_interruption_cascade_marker_fields("object", field_types, field_values)


CheckpointTransform = Callable[
    [Session, dict[str, Any] | None],
    dict[str, Any] | None,
]


class SessionOperationPublication(BaseModel):
    """One atomic checkpoint/event publication plus terminal operation records."""

    model_config = ConfigDict(extra="forbid")

    checkpoint: dict[str, Any]
    operation_records: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("checkpoint", mode="before")
    @classmethod
    def copy_checkpoint(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_object(value, "checkpoint")

    @field_validator("operation_records", mode="before")
    @classmethod
    def copy_operation_records(
        cls,
        value: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        copied = copy_json_object(value, "operation_records")
        for key, record in copied.items():
            require_clean_nonblank(key, "operation_records key")
            if type(record) is not dict:
                raise ValueError("Session operation records must be objects.")
        return copied


SessionOperationTransform = Callable[
    [Session, dict[str, Any] | None, dict[str, Any] | None],
    SessionOperationPublication,
]


class SessionOrder(StrEnum):
    CREATED_AT_ASC = "created_at_asc"
    CREATED_AT_DESC = "created_at_desc"
    UPDATED_AT_ASC = "updated_at_asc"
    UPDATED_AT_DESC = "updated_at_desc"
    LAST_ACTIVITY_AT_ASC = "last_activity_at_asc"
    LAST_ACTIVITY_AT_DESC = "last_activity_at_desc"


class EventOrder(StrEnum):
    SEQUENCE_ASC = "sequence_asc"
    SEQUENCE_DESC = "sequence_desc"


class PendingActionKind(StrEnum):
    TOOL_APPROVAL = "tool_approval"
    USER_INPUT = "user_input"
    MANUAL_RECOVERY = "manual_recovery"


class PendingActionIssueCode(StrEnum):
    """Why one pending-action candidate could not be projected safely."""

    SOURCE_TOO_LARGE = "source_too_large"
    SOURCE_TOO_COMPLEX = "source_too_complex"
    SOURCE_INVALID = "source_invalid"


DEFAULT_PENDING_ACTION_RESULT_MAX_BYTES = 2 * 1024 * 1024
MAX_PENDING_ACTION_RESULT_BYTES = 16 * 1024 * 1024
# A model/runtime should never produce hundreds of calls in one tool round. This
# cap is also a storage-safety boundary: SQL stores inspect the count before
# expanding checkpoint call identifiers into rows.
MAX_PENDING_ACTION_TOOL_CALLS = 256


class PendingActionResultTooLarge(RuntimeError):
    """A pending-action page exceeded its caller-selected serialized byte ceiling."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(
            f"Pending-action response exceeds the {max_bytes}-byte result limit. "
            "Request a smaller page or inspect the session directly."
        )


PENDING_ACTION_EVENT_TYPE_VALUES = frozenset(
    {
        "tool.call.approval_requested",
        "session.awaiting_user_input",
        "session.interrupted",
        "session.resumed",
        "session.completed",
        "session.failed",
        "tool.call.started",
        "tool.call.completed",
        "tool.call.failed",
        "tool.call.blocked",
        "tool.call.approval_denied",
    }
)
PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES = frozenset(
    {"session.resumed", "session.completed", "session.failed"}
)


class LabelSelectorOperator(StrEnum):
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    IN = "in"
    NOT_IN = "not_in"


class LabelSelectorRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    operator: LabelSelectorOperator
    values: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        return next(iter(copy_label_map({value: "_"}, "label selector").keys()))

    @field_validator("values", mode="before")
    @classmethod
    def copy_values(cls, value) -> tuple[str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError("`values` must be a sequence of strings.")
        values = tuple(value)
        copied: list[str] = []
        for index, item in enumerate(values):
            if type(item) is not str:
                raise ValueError("`values` must contain only strings.")
            copied_value = next(
                iter(copy_label_map({f"value_{index}": item}, "label selector value").values())
            )
            if copied_value in copied:
                raise ValueError("`values` must not contain duplicates.")
            copied.append(copied_value)
        return tuple(copied)

    @model_validator(mode="after")
    def validate_operator_values(self) -> LabelSelectorRequirement:
        if self.operator in {LabelSelectorOperator.EXISTS, LabelSelectorOperator.NOT_EXISTS}:
            if self.values:
                raise ValueError(f"`{self.operator}` label selector must not include values.")
        elif not self.values:
            raise ValueError(f"`{self.operator}` label selector requires at least one value.")
        return self


def copy_label_selector_requirements(
    value: Any,
    field_name: str = "label_selectors",
) -> tuple[LabelSelectorRequirement, ...]:
    if value is None:
        return ()
    if type(value) is LabelSelectorRequirement:
        return (value.model_copy(deep=True),)
    if type(value) in {str, dict}:
        raise ValueError(f"`{field_name}` must be a sequence of label selector requirements.")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"`{field_name}` must be a sequence of label selector requirements."
        ) from exc
    return tuple(
        item.model_copy(deep=True)
        if type(item) is LabelSelectorRequirement
        else LabelSelectorRequirement.model_validate(item)
        for item in values
    )


class SessionQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str | None = None
    status: SessionStatus | None = None
    debug_state: SessionDebugState | None = None
    agent_name: str | None = None
    provider_name: str | None = None
    model: str | None = None
    environment_name: str | None = None
    parent_session_id: str | None = None
    causal_budget_id: str | None = None
    last_activity_before: datetime | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    label_selectors: tuple[LabelSelectorRequirement, ...] = Field(default_factory=tuple)
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    offset: StrictInt = Field(default=0, ge=0)
    cursor: str | None = None
    include_total_count: StrictBool = False
    order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "cursor")

    @model_validator(mode="after")
    def reject_cursor_with_offset(self) -> SessionQuery:
        # A keyset cursor and offset are two different paging schemes; combining them
        # would silently ignore the offset, so reject it explicitly.
        if self.cursor is not None and self.offset:
            raise ValueError("cursor and a non-zero offset cannot be combined.")
        return self

    @field_validator(
        "q",
        "agent_name",
        "provider_name",
        "model",
        "environment_name",
        "parent_session_id",
        "causal_budget_id",
    )
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("last_activity_before")
    @classmethod
    def validate_last_activity_before(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("last_activity_before must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_query_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("label_selectors", mode="before")
    @classmethod
    def copy_query_label_selectors(cls, value) -> tuple[LabelSelectorRequirement, ...]:
        return copy_label_selector_requirements(value)


MAX_AGGREGATE_LABEL_FILTERS = 50
MAX_AGGREGATE_LABEL_SELECTORS = 25
MAX_AGGREGATE_LABEL_SELECTOR_VALUES = 100


class SessionAggregateFilter(BaseModel):
    """Current session attributes that may scope a store-native aggregate."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str | None = None
    provider_name: str | None = None
    model: str | None = None
    environment_name: str | None = None
    parent_session_id: str | None = None
    causal_budget_id: str | None = None
    labels: dict[str, str] = Field(
        default_factory=dict,
        max_length=MAX_AGGREGATE_LABEL_FILTERS,
    )
    label_selectors: tuple[LabelSelectorRequirement, ...] = Field(
        default_factory=tuple,
        max_length=MAX_AGGREGATE_LABEL_SELECTORS,
    )

    @field_validator(
        "agent_name",
        "provider_name",
        "model",
        "environment_name",
        "parent_session_id",
        "causal_budget_id",
    )
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_filter_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("label_selectors", mode="before")
    @classmethod
    def copy_filter_label_selectors(cls, value) -> tuple[LabelSelectorRequirement, ...]:
        return copy_label_selector_requirements(value)

    @model_validator(mode="after")
    def validate_selector_value_count(self) -> SessionAggregateFilter:
        value_count = sum(len(selector.values) for selector in self.label_selectors)
        if value_count > MAX_AGGREGATE_LABEL_SELECTOR_VALUES:
            raise ValueError(
                "Aggregate label selectors cannot contain more than "
                f"{MAX_AGGREGATE_LABEL_SELECTOR_VALUES} total values."
            )
        return self


class SessionStatusCounts(BaseModel):
    """Complete current-session counts for every lifecycle status."""

    model_config = ConfigDict(extra="forbid")

    pending: StrictInt = Field(ge=0)
    running: StrictInt = Field(ge=0)
    interrupting: StrictInt = Field(ge=0)
    completed: StrictInt = Field(ge=0)
    failed: StrictInt = Field(ge=0)
    interrupted: StrictInt = Field(ge=0)


class SessionOperationalSnapshot(BaseModel):
    """Exact current session counts captured by one store-local read snapshot."""

    model_config = ConfigDict(extra="forbid")

    as_of: datetime
    total_count: StrictInt = Field(ge=0)
    counts_by_status: SessionStatusCounts
    accuracy: AggregateAccuracy

    @field_validator("as_of")
    @classmethod
    def normalize_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_total(self) -> SessionOperationalSnapshot:
        if sum(self.counts_by_status.model_dump().values()) != self.total_count:
            raise ValueError("Session status counts must sum to total_count.")
        return self


MAX_USAGE_ROLLUP_WINDOW = timedelta(days=366)


class UsageRollupQuery(BaseModel):
    """Bounded event-time usage query with explicit current-session filtering."""

    model_config = ConfigDict(extra="forbid")

    start_at: datetime
    end_at: datetime
    sessions: SessionAggregateFilter = Field(default_factory=SessionAggregateFilter)
    group_limit: StrictInt = Field(default=20, ge=1, le=100)
    include_pricing_inputs: StrictBool = False
    pricing_input_limit: StrictInt = Field(default=1000, ge=1, le=5000)

    @field_validator("start_at", "end_at")
    @classmethod
    def normalize_window_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_window(self) -> UsageRollupQuery:
        if self.start_at >= self.end_at:
            raise ValueError("Usage rollup start_at must be before end_at.")
        if self.end_at - self.start_at > MAX_USAGE_ROLLUP_WINDOW:
            raise ValueError(
                f"Usage rollup window cannot exceed {MAX_USAGE_ROLLUP_WINDOW.days} days."
            )
        return self


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: StrictInt = Field(ge=1)
    event: Event

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)


class PersistedEventSideEffectStatus(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD_LETTERED = "dead_lettered"


class PersistedEventSideEffectClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1)
    event: Event
    attempt: StrictInt = Field(ge=1)
    claim_id: str = Field(default_factory=lambda: str(uuid4()))
    lease_expires_at: datetime

    @field_validator("session_id", "event_id", "claim_id")
    @classmethod
    def validate_clean_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("event")
    @classmethod
    def copy_claim_event(cls, value: Event) -> Event:
        return copy_event(value)

    @field_validator("lease_expires_at")
    @classmethod
    def normalize_lease_expires_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("lease_expires_at must be timezone-aware.")
        return value.astimezone(UTC)


class PersistedEventSideEffectDelivery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    event_id: str
    event_sequence: StrictInt = Field(ge=1)
    status: PersistedEventSideEffectStatus
    attempts: StrictInt = Field(default=0, ge=0)
    claim_id: str | None = None
    lease_expires_at: datetime | None = None
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("session_id", "event_id", "claim_id", "last_error")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if info.field_name == "last_error":
            if not value.strip():
                raise ValueError("last_error cannot be blank.")
            return value
        return require_clean_nonblank(value, info.field_name)

    @field_validator("lease_expires_at", "next_attempt_at", "updated_at")
    @classmethod
    def normalize_delivery_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)


class PendingActionQuery(BaseModel):
    """Bounded query for durable control-plane actions blocking a session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    kind: PendingActionKind | None = None
    agent_name: str | None = None
    environment_name: str | None = None
    q: str | None = None
    cursor: str | None = None
    limit: StrictInt = Field(default=50, ge=1, le=200)
    max_result_bytes: StrictInt = Field(
        default=DEFAULT_PENDING_ACTION_RESULT_MAX_BYTES,
        ge=1024,
        le=MAX_PENDING_ACTION_RESULT_BYTES,
    )

    @field_validator(
        "session_id",
        "agent_name",
        "environment_name",
        "q",
        "cursor",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class PendingActionSession(BaseModel):
    """Bounded session identity embedded in pending-action query results."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None = None
    causal_budget_id: str
    runtime_name: str
    runtime_version: str | None = None
    environment_name: str | None = None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    labels: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_session(cls, session: Session) -> PendingActionSession:
        return cls(
            id=session.id,
            agent_name=session.agent_name,
            provider_name=session.provider_name,
            model=session.model,
            parent_session_id=session.parent_session_id,
            causal_budget_id=session.causal_budget_id,
            runtime_name=session.runtime_name,
            runtime_version=session.runtime_version,
            environment_name=session.environment_name,
            status=session.status,
            created_at=session.created_at,
            updated_at=session.updated_at,
            labels=session.labels,
        )

    @field_validator(
        "id",
        "agent_name",
        "provider_name",
        "model",
        "causal_budget_id",
        "runtime_name",
    )
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("parent_session_id", "environment_name", "runtime_version")
    @classmethod
    def validate_optional_nonblank_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value: dict[str, str]) -> dict[str, str]:
        return copy_label_map(value, "labels")


class PendingActionRecord(BaseModel):
    """One current action derived from a session checkpoint and its source event."""

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: PendingActionKind
    session: PendingActionSession
    event: EventRecord
    title: str
    detail: str | None = None
    tool_name: str | None = None
    approval_id: str | None = None
    input_id: str | None = None
    round_id: str | None = None
    tool_call_id: str | None = None
    question: str | None = None
    options: list[str] = Field(default_factory=list)
    arguments: dict[str, Any] | None = None

    @field_validator("id", "title")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "detail",
        "tool_name",
        "approval_id",
        "input_id",
        "round_id",
        "tool_call_id",
        "question",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)

    @field_validator("session")
    @classmethod
    def copy_session(cls, value: PendingActionSession) -> PendingActionSession:
        return value.model_copy(deep=True)

    @field_validator("event")
    @classmethod
    def copy_event_record(cls, value: EventRecord) -> EventRecord:
        return value.model_copy(deep=True)

    @field_validator("options", mode="before")
    @classmethod
    def copy_options(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if type(value) is not list:
            raise TypeError("options must be a list of strings.")
        return [require_nonblank(item, "options") for item in value]

    @field_validator("arguments", mode="before")
    @classmethod
    def copy_arguments(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_json_value(value, "arguments")


class PendingActionIssue(BaseModel):
    """Bounded visibility for a candidate that could not be materialized."""

    model_config = ConfigDict(extra="forbid")

    code: PendingActionIssueCode
    session_id: str
    agent_name: str
    status: SessionStatus
    updated_at: datetime
    detail: str

    @field_validator("session_id", "agent_name", "detail")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("updated_at")
    @classmethod
    def normalize_updated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware.")
        return value.astimezone(UTC)

    @classmethod
    def source_too_large(
        cls,
        session: PendingActionSession,
        *,
        max_bytes: int,
    ) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_TOO_LARGE,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending-action source data for this session exceeds the "
                f"{max_bytes}-byte inspection limit. Open the session directly "
                "to inspect or resolve it."
            ),
        )

    @classmethod
    def source_too_complex(
        cls,
        session: PendingActionSession,
        *,
        max_tool_calls: int,
    ) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_TOO_COMPLEX,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending tool-round state for this session exceeds the "
                f"{max_tool_calls}-call inspection limit. Open the session directly "
                "to inspect or resolve it."
            ),
        )

    @classmethod
    def source_invalid(cls, session: PendingActionSession) -> PendingActionIssue:
        return cls(
            code=PendingActionIssueCode.SOURCE_INVALID,
            session_id=session.id,
            agent_name=session.agent_name,
            status=session.status,
            updated_at=session.updated_at,
            detail=(
                "Pending-action state for this session is incomplete or inconsistent. "
                "Inspect the session directly before attempting to resume it."
            ),
        )


class PendingActionListResult(BaseModel):
    """One stable page of pending actions and its candidate continuation cursor."""

    model_config = ConfigDict(extra="forbid")

    actions: list[PendingActionRecord] = Field(default_factory=list)
    issues: list[PendingActionIssue] = Field(default_factory=list)
    next_cursor: str | None = None
    has_more: StrictBool = False
    total_count: StrictInt | None = Field(default=None, ge=0)
    inspected_candidate_count: StrictInt = Field(default=0, ge=0)


def enforce_pending_action_result_size(
    result: PendingActionListResult,
    *,
    max_bytes: int,
) -> PendingActionListResult:
    """Return ``result`` or fail before an oversized API body is serialized."""
    if not json_utf8_size_within_limit(result, max_bytes):
        raise PendingActionResultTooLarge(max_bytes)
    return result


class EventSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    total_events: StrictInt = Field(ge=0)
    counts_by_type: dict[str, StrictInt] = Field(default_factory=dict)
    latest_event: EventRecord | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")


class SerializedRecordSummary(BaseModel):
    """Exact serialized-size totals for one kind of durable session record."""

    model_config = ConfigDict(extra="forbid")

    record_count: StrictInt = Field(ge=0)
    total_bytes: StrictInt = Field(ge=0)
    largest_record_bytes: StrictInt = Field(ge=0)


class SessionInspectionIdentity(BaseModel):
    """Bounded, metadata-free identity used by operator inspection."""

    model_config = ConfigDict(extra="forbid")

    id: str
    agent_name: str
    provider_name: str
    model: str
    parent_session_id: str | None
    causal_budget_id: str
    runtime_name: str
    runtime_version: str | None
    environment_name: str | None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    run_epoch: StrictInt = Field(ge=0)
    labels: dict[str, str] = Field(default_factory=dict)
    label_count: StrictInt = Field(ge=0)
    labels_truncated: StrictBool = False


class SessionInspectionSummary(BaseModel):
    """Bounded backend-neutral diagnostic overview for one durable session."""

    model_config = ConfigDict(extra="forbid")

    session: SessionInspectionIdentity
    transcript: SerializedRecordSummary
    events: SerializedRecordSummary
    usage: SessionUsageSummary
    model_calls: StrictInt = Field(ge=0)
    model_calls_with_usage: StrictInt = Field(ge=0)
    tool_calls: StrictInt = Field(ge=0)
    pending_action_count: StrictInt = Field(ge=0)
    pending_action_kinds: tuple[PendingActionKind, ...] = ()
    pending_action_issue_count: StrictInt = Field(ge=0)
    queued_message_count: StrictInt = Field(ge=0)
    delivered_message_count: StrictInt = Field(ge=0)
    outstanding_message_count: StrictInt = Field(ge=0)
    operation_event_count: StrictInt = Field(ge=0)
    terminal_failure_state: Literal["none", "failed", "interrupted"]
    budget: SessionBudgetInspection


_SESSION_INSPECTION_MAX_RECORDS = 100_000
_SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES = 64 * 1024 * 1024
_SESSION_INSPECTION_PAGE_SIZE = 200
SESSION_INSPECTION_LABEL_LIMIT = 200


def _bounded_session_inspection_labels(
    labels: dict[str, str],
) -> tuple[dict[str, str], int, bool]:
    label_count = len(labels)
    retained_keys = heapq.nsmallest(SESSION_INSPECTION_LABEL_LIMIT, labels)
    return (
        {key: labels[key] for key in retained_keys},
        label_count,
        label_count > len(retained_keys),
    )


def _retain_session_inspection_event(current_bytes: int, event: Event) -> int:
    retained_bytes = current_bytes + compact_json_utf8_size(event.model_dump(mode="json"))
    if retained_bytes > _SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES:
        raise ValueError(
            "Session inspection exceeds the retained-event safety limit of "
            f"{_SESSION_INSPECTION_MAX_RETAINED_EVENT_BYTES} bytes."
        )
    return retained_bytes


class SessionOutcome(BaseModel):
    """Derived reason for the current session state.

    The outcome is computed from durable events. It is intentionally not stored
    as separate state so event replay remains the source of truth.
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    status: SessionStatus
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)
    retry: dict[str, Any] | None = None
    terminal_event: EventRecord | None = None
    latest_retry_event: EventRecord | None = None

    @field_validator("session_id", "reason")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("details", mode="before")
    @classmethod
    def copy_details(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "details")

    @field_validator("retry", mode="before")
    @classmethod
    def copy_retry(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_json_value(value, "retry")


class IncompleteSessionRecoveryAction(StrEnum):
    SKIPPED_ACTIVE = "skipped_active"
    SKIPPED_TERMINAL = "skipped_terminal"
    SKIPPED_UNREGISTERED_AGENT = "skipped_unregistered_agent"
    PENDING_APPROVAL = "pending_approval"
    PENDING_USER_INPUT = "pending_user_input"
    REPAIRED_TOOL_ROUND = "repaired_tool_round"
    INTERRUPTED_ABANDONED = "interrupted_abandoned"
    FINALIZED_INTERRUPT = "finalized_interrupt"
    FAILED = "failed"


class IncompleteSessionRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    inactive_before: datetime | None = None
    reason: str = "worker_recovered_incomplete_session"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "reason")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("inactive_before")
    @classmethod
    def validate_inactive_before(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("inactive_before must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class IncompleteSessionsRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statuses: set[SessionStatus]
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    inactive_before: datetime | None = None
    reason: str = "worker_recovered_incomplete_session"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("statuses", mode="before")
    @classmethod
    def copy_statuses(cls, value) -> set[SessionStatus]:
        if value is None:
            raise ValueError("statuses is required for batch incomplete-session recovery.")
        if not isinstance(value, (set, list, tuple)):
            raise ValueError("statuses must be a set of SessionStatus values.")
        statuses: set[SessionStatus] = set()
        for status in value:
            if not isinstance(status, SessionStatus):
                status = SessionStatus(status)
            statuses.add(status)
        if not statuses:
            raise ValueError("statuses must not be empty.")
        recoverable_statuses = {
            SessionStatus.PENDING,
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
        }
        unsupported_statuses = statuses - recoverable_statuses
        if unsupported_statuses:
            unsupported = ", ".join(sorted(status.value for status in unsupported_statuses))
            supported = ", ".join(sorted(status.value for status in recoverable_statuses))
            raise ValueError(
                f"statuses contains unsupported recovery status values: {unsupported}. "
                f"Supported values are: {supported}."
            )
        return statuses

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return require_clean_nonblank(value, "reason")

    @field_validator("inactive_before")
    @classmethod
    def validate_inactive_before(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("inactive_before must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class IncompleteSessionRecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    previous_status: SessionStatus
    status: SessionStatus
    actions: tuple[IncompleteSessionRecoveryAction, ...]
    events: tuple[Event, ...] = Field(default_factory=tuple)
    pending_approval_id: str | None = None
    pending_user_input_id: str | None = None
    message: str

    @field_validator("session_id", "message")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("events")
    @classmethod
    def copy_events(cls, value) -> tuple[Event, ...]:
        return tuple(copy_event(event) for event in value)


class EventQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    session_ids: tuple[str, ...] = Field(default_factory=tuple)
    event_id: str | None = None
    causal_budget_id: str | None = None
    event_type: EventType | str | None = None
    event_types: tuple[EventType | str, ...] = Field(default_factory=tuple)
    exclude_event_types: tuple[EventType | str, ...] = Field(default_factory=tuple)
    agent_name: str | None = None
    environment_name: str | None = None
    workflow_name: str | None = None
    tool_name: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    after_sequence: StrictInt | None = Field(default=None, ge=0)
    before_sequence: StrictInt | None = Field(default=None, ge=1)
    limit: StrictInt = Field(default=100, ge=1, le=5000)
    order_by: EventOrder = EventOrder.SEQUENCE_ASC

    @field_validator(
        "session_id",
        "event_id",
        "causal_budget_id",
        "agent_name",
        "environment_name",
        "workflow_name",
        "tool_name",
    )
    @classmethod
    def validate_optional_nonblank_fields(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("session_ids", mode="before")
    @classmethod
    def copy_session_ids(cls, value) -> tuple[str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError("`session_ids` must be a sequence of strings.")
        values = tuple(value)
        copied: list[str] = []
        for index, item in enumerate(values):
            clean_item = require_clean_nonblank(item, f"session_ids[{index}]")
            if clean_item in copied:
                raise ValueError("`session_ids` must not contain duplicates.")
            copied.append(clean_item)
        return tuple(copied)

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: EventType | str | None) -> EventType | str | None:
        if value is None:
            return None
        if isinstance(value, EventType):
            return value
        return Event(type=value, session_id="query").type

    @field_validator("event_types", "exclude_event_types", mode="before")
    @classmethod
    def copy_event_types(cls, value, info) -> tuple[EventType | str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError(f"`{info.field_name}` must be a sequence of event types.")
        normalized: list[EventType | str] = []
        for item in tuple(value):
            if not isinstance(item, EventType):
                item = Event(type=item, session_id="query").type
            if item in normalized:
                raise ValueError(f"`{info.field_name}` must not contain duplicates.")
            normalized.append(item)
        return tuple(normalized)

    @field_validator("since", "until")
    @classmethod
    def validate_query_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_time_range(self) -> EventQuery:
        if self.session_id is not None and self.session_ids:
            raise ValueError("Use either `session_id` or `session_ids`, not both.")
        if self.event_id is not None and self.session_id is None:
            raise ValueError("EventQuery event_id requires session_id.")
        if self.event_type is not None and self.event_types:
            raise ValueError("Use either `event_type` or `event_types`, not both.")
        if self.since is not None and self.until is not None and self.since >= self.until:
            raise ValueError("EventQuery since must be before until.")
        if (
            self.before_sequence is not None
            and self.session_id is None
            and len(self.session_ids) != 1
        ):
            raise ValueError("EventQuery before_sequence requires exactly one session.")
        if (
            self.after_sequence is not None
            and self.before_sequence is not None
            and self.after_sequence >= self.before_sequence
        ):
            raise ValueError("EventQuery after_sequence must be before before_sequence.")
        return self


class TranscriptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: StrictInt = Field(ge=0)
    message: Message

    @field_validator("message")
    @classmethod
    def copy_message(cls, value: Message) -> Message:
        return copy_message(value)


class TranscriptPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[TranscriptRecord] = Field(default_factory=list)
    total_records: StrictInt = Field(ge=0)


class SessionListResult(BaseModel):
    """One page of a session listing plus its keyset cursor and (optional) total count."""

    model_config = ConfigDict(extra="forbid")

    sessions: list[Session] = Field(default_factory=list)
    next_cursor: str | None = None
    # None unless the query opted in via include_total_count (COUNT is expensive at scale).
    total_count: StrictInt | None = Field(default=None, ge=0)


class TranscriptQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    role: MessageRole | str | None = None
    offset: StrictInt = Field(default=0, ge=0)
    limit: StrictInt = Field(default=100, ge=1, le=5000)
    # When False, ThinkingPart content is stripped from the returned messages. This is a
    # content view, not a record filter: `total_records` stays the role-matched total, a
    # page may hold fewer than `limit` records when thinking-only turns drop out, and each
    # record keeps its true transcript `index` (so offset pagination is unaffected).
    include_thinking: StrictBool = True

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "session_id")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: MessageRole | str | None) -> MessageRole | None:
        if value is None:
            return None
        return MessageRole(value)


def _without_thinking_parts(message: Message) -> Message | None:
    kept = [part for part in message.content if type(part) is not ThinkingPart]
    if not kept:
        return None
    return Message(role=message.role, content=tuple(kept))


def filter_transcript_records(
    records: list[TranscriptRecord], *, include_thinking: bool
) -> list[TranscriptRecord]:
    """Apply a `TranscriptQuery.include_thinking` filter to a page of records.

    When ``include_thinking`` is False, ThinkingParts are stripped from each message and
    records whose message is left empty (a thinking-only turn) are dropped. Every
    surviving message on that path is freshly rebuilt through full validation, so it
    shares no payload state with the input records; when True, records pass through
    unchanged and the caller is responsible for any isolation.
    """
    if include_thinking:
        return records
    filtered: list[TranscriptRecord] = []
    for record in records:
        message = _without_thinking_parts(record.message)
        if message is not None:
            filtered.append(TranscriptRecord(index=record.index, message=message))
    return filtered


class SessionStore(ABC):
    """Persistent store for sessions and append-only events.

    Implementations own the run-epoch contract. A successful transition to
    ``RUNNING`` claims a new epoch for the current execution context. Runtime
    progress writes made by that context must reject a stale epoch with
    ``SessionRunFenced``. Releasing the claim revokes the durable epoch before
    clearing task-local ownership so inherited child contexts cannot write late.
    """

    @abstractmethod
    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
    ) -> Session:
        """Create a session for a run request."""

    @abstractmethod
    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
    ) -> Session:
        """Create a forked session with copied transcript/checkpoint state."""

    @abstractmethod
    async def load(self, session_id: str) -> Session | None:
        """Load a session by id."""

    @abstractmethod
    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        """Load bounded mutable state without labels or unbounded metadata."""

    @abstractmethod
    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        """Load bounded session identity without materializing session metadata."""

    @abstractmethod
    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        """Update session status and return the updated session."""

    @abstractmethod
    async def update_model(self, session_id: str, model: str) -> Session:
        """Update the active model for a session and return the updated session."""

    @abstractmethod
    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        """Atomically transition status, claiming a new epoch when entering RUNNING."""

    @abstractmethod
    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        """Atomically persist a transition/checkpoint and claim RUNNING when requested."""

    @abstractmethod
    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        """Atomically terminalize only when no durable queued input remains."""

    @abstractmethod
    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_before: datetime,
    ) -> Session | None:
        """Atomically evict a stale run and claim its newly incremented epoch.

        Returns ``None`` when the session is no longer in one of ``statuses`` or
        has activity newer than ``inactive_before``.
        """
        raise NotImplementedError

    @abstractmethod
    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        """Atomically transform a checkpoint and claim a newly incremented epoch.

        The transform must return the replacement checkpoint. An exception or
        ``None`` result aborts both the checkpoint update and the epoch fence.
        This operation is for ownership that must persist a checkpoint lease
        and fence the prior run as one transaction. Recovery that needs only an
        inactivity predicate should continue to use :meth:`fence_stalled_run`.
        """
        raise NotImplementedError

    @abstractmethod
    async def release_run_fence(self, session_id: str) -> None:
        """Revoke this task's epoch after all trailing writes finish."""
        _deactivate_session_run_fence(require_clean_nonblank(session_id, "session_id"))

    async def append_event(self, session_id: str, event: Event) -> None:
        """Append one event to a session."""
        await self.append_events(session_id, [event])

    @abstractmethod
    async def append_events(self, session_id: str, events: list[Event]) -> None:
        """Append events to a session in one durable batch."""

    @abstractmethod
    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        """Claim the oldest eligible persisted event side-effect handoff."""

    @abstractmethod
    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        """Load one persisted event side-effect handoff by event identity."""

    @abstractmethod
    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        """Mark one claimed event's configured side effects delivered."""

    @abstractmethod
    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        """Release one failed claim for retry or dead-letter it."""

    @abstractmethod
    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        """Inspect persisted side-effect delivery state in event order.

        ``claimable_only`` excludes terminal deliveries and live leases.
        """

    @abstractmethod
    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
    ) -> EnqueueSessionMessageResult:
        """Atomically accept queued input with its causal event, or replay it."""

    @abstractmethod
    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    ) -> SessionMessageDeliveryBatch:
        """Atomically append and mark one bounded queue batch delivered."""

    @abstractmethod
    async def publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        """Atomically transform a checkpoint and append its causal event batch."""

    @abstractmethod
    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """Load one terminal durable operation record by caller idempotency key."""

    @abstractmethod
    async def publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        """Atomically publish a checkpoint, events, and terminal operation records."""

    @abstractmethod
    async def load_events(self, session_id: str) -> list[Event]:
        """Load all events for a session."""

    @abstractmethod
    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        """Query stored events with durable sequence cursors."""

    @abstractmethod
    async def summarize_events(self, session_id: str) -> EventSummary:
        """Summarize stored events for one session without loading every event."""

    @abstractmethod
    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        """Derive the current session outcome from durable events."""

    async def inspect_summary(self, session_id: str) -> SessionInspectionSummary:
        """Return one exact, content-free diagnostic summary.

        The default implementation deliberately composes the public paginated
        query contracts so out-of-tree stores gain the same semantics without a
        backend-specific schema dependency. Implementations may replace it with
        an equivalent bounded native aggregate.
        """

        session_id = require_clean_nonblank(session_id, "session_id")
        identity = await self.inspect_identity(session_id)

        event_count = 0
        event_total_bytes = 0
        event_largest_bytes = 0
        usage_events: list[Event] = []
        budget_events: list[Event] = []
        retained_event_bytes = 0
        queued_message_count = 0
        delivered_message_count = 0
        operation_event_count = 0
        after_sequence = 0
        while True:
            records = await self.query_events(
                EventQuery(
                    session_id=session_id,
                    after_sequence=after_sequence,
                    limit=_SESSION_INSPECTION_PAGE_SIZE,
                    order_by=EventOrder.SEQUENCE_ASC,
                )
            )
            if not records:
                break
            event_count += len(records)
            if event_count > _SESSION_INSPECTION_MAX_RECORDS:
                raise ValueError(
                    "Session inspection exceeds the "
                    f"{_SESSION_INSPECTION_MAX_RECORDS}-event safety limit."
                )
            for record in records:
                payload_bytes = compact_json_utf8_size(record.event.payload)
                event_total_bytes += payload_bytes
                event_largest_bytes = max(event_largest_bytes, payload_bytes)
                event = record.event
                if event.type in {EventType.MODEL_COMPLETED, EventType.TOOL_CALL_STARTED}:
                    usage_event = project_usage_inspection_event(event)
                    retained_event_bytes = _retain_session_inspection_event(
                        retained_event_bytes,
                        usage_event,
                    )
                    usage_events.append(usage_event)
                if is_budget_inspection_event(event):
                    budget_event = project_budget_inspection_event(event)
                    retained_event_bytes = _retain_session_inspection_event(
                        retained_event_bytes,
                        budget_event,
                    )
                    budget_events.append(budget_event)
                queued_message_count += event.type == EventType.SESSION_MESSAGE_QUEUED
                delivered_message_count += event.type == EventType.SESSION_MESSAGE_DELIVERED
                operation_event_count += event.type == EventType.SERVER_MUTATION_ACCEPTED
            after_sequence = records[-1].sequence
            if len(records) < _SESSION_INSPECTION_PAGE_SIZE:
                break

        transcript_count = 0
        transcript_total_bytes = 0
        transcript_largest_bytes = 0
        offset = 0
        while True:
            page = await self.query_transcript(
                TranscriptQuery(
                    session_id=session_id,
                    offset=offset,
                    limit=_SESSION_INSPECTION_PAGE_SIZE,
                )
            )
            if offset == 0 and page.total_records > _SESSION_INSPECTION_MAX_RECORDS:
                raise ValueError(
                    "Session inspection exceeds the "
                    f"{_SESSION_INSPECTION_MAX_RECORDS}-message safety limit."
                )
            for record in page.records:
                message_bytes = compact_json_utf8_size(record.message.model_dump(mode="json"))
                transcript_count += 1
                transcript_total_bytes += message_bytes
                transcript_largest_bytes = max(transcript_largest_bytes, message_bytes)
            offset += _SESSION_INSPECTION_PAGE_SIZE
            if offset >= page.total_records:
                break

        pending = await self.query_pending_actions(
            PendingActionQuery(session_id=session_id, limit=200)
        )
        usage = session_usage_summary(session_id, usage_events)
        model_calls_with_usage = count_model_steps_with_usage(usage_events)
        budget = session_budget_inspection(budget_events)
        return SessionInspectionSummary(
            session=identity,
            transcript=SerializedRecordSummary(
                record_count=transcript_count,
                total_bytes=transcript_total_bytes,
                largest_record_bytes=transcript_largest_bytes,
            ),
            events=SerializedRecordSummary(
                record_count=event_count,
                total_bytes=event_total_bytes,
                largest_record_bytes=event_largest_bytes,
            ),
            usage=usage,
            model_calls=usage.model_steps,
            model_calls_with_usage=model_calls_with_usage,
            tool_calls=usage.tool_calls,
            pending_action_count=len(pending.actions),
            pending_action_kinds=tuple(
                sorted({action.kind for action in pending.actions}, key=str)
            ),
            pending_action_issue_count=len(pending.issues),
            queued_message_count=queued_message_count,
            delivered_message_count=delivered_message_count,
            outstanding_message_count=max(
                queued_message_count - delivered_message_count,
                0,
            ),
            operation_event_count=operation_event_count,
            terminal_failure_state=(
                "failed"
                if identity.status is SessionStatus.FAILED
                else "interrupted"
                if identity.status is SessionStatus.INTERRUPTED
                else "none"
            ),
            budget=budget,
        )

    @abstractmethod
    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        """List sessions (filtered/sorted/paginated) with a keyset cursor and total count."""

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        """Count current session states in one store-local read snapshot.

        Default raises ``NotImplementedError`` so adding this control-plane read
        model does not make existing out-of-tree stores uninstantiable.
        """
        raise NotImplementedError(
            "This SessionStore does not support operational aggregate snapshots."
        )

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        """Aggregate bounded event-time activity and usage without loading histories.

        Default raises ``NotImplementedError`` so adding this control-plane read
        model does not make existing out-of-tree stores uninstantiable.
        """
        raise NotImplementedError("This SessionStore does not support usage aggregates.")

    @abstractmethod
    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        """List only sessions carrying a pending interruption cascade marker.

        Implementations should filter at the storage layer. This is the
        restart-discovery path, so scanning all historical sessions and loading
        their checkpoints individually is not an acceptable implementation.
        """

    @abstractmethod
    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
    ) -> PendingActionListResult:
        """List current checkpoint-backed actions without scanning session histories.

        Implementations must bound candidate and event reads, apply filtering at
        the storage layer where possible, and avoid per-session history queries.
        """

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and cascade to its events, transcript, and checkpoint.

        Raises ``ValueError`` if the session is in-flight (``RUNNING`` or
        ``INTERRUPTING`` — interrupt it first). Idempotent: deleting a session
        that does not exist is a no-op.

        Default raises ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support delete_session.")

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        """Replace a session's labels (full replacement, not a merge) and return it.

        Annotation writes update ``updated_at`` but do not refresh the runtime
        recovery signal in ``last_activity_at``.

        Raises ``KeyError`` if the session does not exist. Default raises
        ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support update_labels.")

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        """Replace a session's user metadata and return the updated session.

        This is a full replacement of user-authored metadata, not a merge.
        Runtime-owned entries are retained atomically and cannot be supplied by
        callers. Annotation writes update ``updated_at`` but do not refresh the
        runtime recovery signal in ``last_activity_at``.

        Raises ``KeyError`` if the session does not exist. Default raises
        ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support update_metadata.")

    @abstractmethod
    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
    ) -> None:
        """Append provider-neutral transcript messages to a session.

        Stored state must not alias caller-passed messages: implementations
        must detach nested JSON payloads (serialize, or use
        `cayu.core.messages.detach_message`) so a producer mutating a message
        after append cannot rewrite history.
        """

    @abstractmethod
    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        """Append transcript messages and atomically transform the current checkpoint.

        The transform must be synchronous and thread-safe; a store may execute
        it on a worker thread while holding its transactional write boundary.
        """

    @abstractmethod
    async def load_transcript(self, session_id: str) -> list[Message]:
        """Load provider-neutral transcript messages for a session.

        Returned messages must not alias stored state: callers own the result
        and may mutate nested payloads without corrupting the transcript.
        """

    @abstractmethod
    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        """Query provider-neutral transcript messages with stable message indexes.

        Same isolation contract as `load_transcript`.
        """

    @abstractmethod
    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist a checkpoint for resume/replay."""

    @abstractmethod
    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        """Atomically transform the latest checkpoint for a session.

        The transform runs while the store owns its session/checkpoint write
        boundary and must be synchronous and thread-safe because a store may
        execute it on a worker thread. Returning ``None`` leaves the checkpoint
        unchanged.
        """

    @abstractmethod
    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Load the latest checkpoint for a session."""

    @abstractmethod
    async def load_interruption_cascade_marker(self, session_id: str) -> dict[str, Any] | None:
        """Load a bounded structural projection of the durable cascade marker."""


class InMemorySessionStore(SessionStore):
    """In-process session store for tests, local development, and examples."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sessions: dict[str, Session] = {}
        self._events: dict[str, list[Event]] = {}
        self._event_records: list[EventRecord] = []
        # Secondary indexes over ``_event_records`` (same EventRecord objects, kept in
        # sequence-ascending order) so per-session summaries and type/session-scoped
        # queries — including the per-step budget read path — stop scanning the global
        # event list. Keyed by session id and by ``str(event.type)`` respectively.
        self._session_event_records: dict[str, list[EventRecord]] = {}
        self._pending_action_event_records: dict[
            str,
            dict[str, dict[str, EventRecord]],
        ] = {}
        # Present only after an identifier map has been pruned. The value lists
        # identifiers subsequently rebuilt from the complete event log; absence
        # means the live append index has covered the session since creation.
        self._pending_action_rebuilt_lookup_ids: dict[str, frozenset[str]] = {}
        self._pending_action_latest_barrier_records: dict[str, EventRecord] = {}
        self._event_records_by_id: dict[tuple[str, str], EventRecord] = {}
        self._type_event_records: dict[str, list[EventRecord]] = {}
        self._event_ids: dict[str, set[str]] = {}
        self._persisted_event_side_effect_deliveries: dict[
            tuple[str, str], PersistedEventSideEffectDelivery
        ] = {}
        self._next_event_sequence = 1
        self._transcripts: dict[str, list[Message]] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._session_operation_records: dict[str, dict[str, dict[str, Any]]] = {}
        self._pending_action_session_ids: set[str] = set()
        # Retain every accepted message for idempotent replay, but keep queued
        # delivery state in per-session/mode deques. Hot delivery and terminal
        # fencing therefore depend on the pending set, not historical deliveries.
        self._queued_session_messages_by_idempotency: dict[
            str, dict[str, SessionQueuedMessage]
        ] = {}
        self._pending_session_messages: dict[
            tuple[str, SessionMessageDeliveryMode], deque[SessionQueuedMessage]
        ] = {}
        self._next_session_message_ordering_key = 1

    def _store_checkpoint_unlocked(self, session_id: str, checkpoint: dict[str, Any]) -> None:
        from cayu.runtime.pending_actions import (
            checkpoint_has_pending_action_candidate,
            pending_action_checkpoint_lookup_ids,
            pending_action_event_lookup_id,
            pending_action_lookup_key,
            project_pending_action_event_record,
        )

        self._checkpoints[session_id] = checkpoint
        if checkpoint_has_pending_action_candidate(checkpoint):
            lookup_keys = frozenset(
                pending_action_lookup_key(identifier)
                for identifier in pending_action_checkpoint_lookup_ids(checkpoint)
            )
            rebuilt_lookup_ids = self._pending_action_rebuilt_lookup_ids.get(session_id)
            if rebuilt_lookup_ids is not None and not lookup_keys.issubset(rebuilt_lookup_ids):
                # A public checkpoint transform may legitimately reintroduce a
                # previously cleared durable action. SQL stores retain the source
                # events, so rebuild the bounded identifier/type projection here
                # to preserve identical behavior across backends.
                rebuilt: dict[str, dict[str, EventRecord]] = {}
                for record in reversed(self._session_event_records.get(session_id, [])):
                    event_type = str(record.event.type)
                    if (
                        event_type not in PENDING_ACTION_EVENT_TYPE_VALUES
                        or event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES
                    ):
                        continue
                    lookup_id = pending_action_event_lookup_id(record.event)
                    if lookup_id is None:
                        continue
                    lookup_key = pending_action_lookup_key(lookup_id)
                    if lookup_key not in lookup_keys:
                        continue
                    by_event_type = rebuilt.setdefault(lookup_key, {})
                    by_event_type.setdefault(
                        event_type,
                        project_pending_action_event_record(record),
                    )
                self._pending_action_event_records[session_id] = rebuilt
                self._pending_action_rebuilt_lookup_ids[session_id] = lookup_keys
            self._pending_action_session_ids.add(session_id)
        else:
            self._pending_action_session_ids.discard(session_id)
            # Once the checkpoint no longer names a pending action, identifier-
            # scoped event history cannot contribute to a future action. Keep the
            # latest lifecycle barrier, but release the potentially growing map.
            removed = self._pending_action_event_records.pop(session_id, None)
            if removed or session_id in self._pending_action_rebuilt_lookup_ids:
                self._pending_action_rebuilt_lookup_ids[session_id] = frozenset()

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
    ) -> Session:
        if type(request) is not RunRequest:
            raise TypeError("Session creation requires a RunRequest.")
        request = copy_run_request(request)
        identity = copy_session_identity(identity)
        async with self._lock:
            session_id = request.session_id or str(uuid4())
            if session_id in self._sessions:
                raise ValueError(f"Session already exists: {session_id}")
            if request.parent_session_id == session_id:
                raise ValueError("Session cannot be its own parent.")
            if (
                request.parent_session_id is not None
                and request.parent_session_id not in self._sessions
            ):
                raise ValueError(f"Parent session not found: {request.parent_session_id}")

            now = datetime.now(UTC)
            session = Session(
                id=session_id,
                agent_name=request.agent_name,
                provider_name=identity.provider_name,
                model=identity.model,
                parent_session_id=request.parent_session_id,
                causal_budget_id=request.causal_budget_id or request.task_id or session_id,
                runtime_name=identity.runtime_name,
                runtime_version=identity.runtime_version,
                environment_name=request.environment_name,
                status=SessionStatus.PENDING,
                created_at=now,
                updated_at=now,
                last_activity_at=now,
                labels=request.labels,
                metadata=deepcopy(request.metadata),
            )
            self._sessions[session.id] = session
            self._events[session.id] = []
            self._event_ids[session.id] = set()
            self._session_event_records[session.id] = []
            self._pending_action_event_records[session.id] = {}
            self._session_operation_records[session.id] = {}
            self._transcripts[session.id] = []
            return session.model_copy(deep=True)

    async def create_fork(
        self,
        *,
        source_session_id: str,
        fork: Session,
        source_statuses: set[SessionStatus],
        transcript_cursor: int | None,
        checkpoint_transform: CheckpointTransform | None,
        expected_source_run_epoch: int,
    ) -> Session:
        source_session_id, fork, allowed_statuses, transcript_cursor = (
            _prepare_session_fork_request(
                source_session_id=source_session_id,
                fork=fork,
                source_statuses=source_statuses,
                transcript_cursor=transcript_cursor,
            )
        )
        async with self._lock:
            source_session = _validate_session_fork_source(
                source_session=self._sessions.get(source_session_id),
                source_session_id=source_session_id,
                fork=fork,
                allowed_statuses=allowed_statuses,
                expected_source_run_epoch=expected_source_run_epoch,
            )
            if fork.id in self._sessions:
                raise ValueError(f"Session already exists: {fork.id}")

            source_transcript = self._transcripts.get(source_session_id, [])
            # Fork and source transcripts may share message objects: internal
            # transcript lists never escape except through the detaching
            # load/append/query boundaries, so the cheap share stays invisible.
            if transcript_cursor is None:
                copied_transcript = [copy_message(message) for message in source_transcript]
            else:
                if transcript_cursor > len(source_transcript):
                    raise ValueError("transcript_cursor is greater than source transcript length.")
                copied_transcript = [
                    copy_message(message) for message in source_transcript[:transcript_cursor]
                ]
            copied_checkpoint = None
            if checkpoint_transform is not None:
                source_checkpoint = self._checkpoints.get(source_session_id)
                copied_checkpoint = checkpoint_transform(
                    source_session.model_copy(deep=True),
                    None if source_checkpoint is None else deepcopy(source_checkpoint),
                )
                if copied_checkpoint is not None:
                    copied_checkpoint = copy_json_value(copied_checkpoint, "checkpoint")

            self._sessions[fork.id] = fork.model_copy(deep=True)
            self._events[fork.id] = []
            self._event_ids[fork.id] = set()
            self._session_event_records[fork.id] = []
            self._pending_action_event_records[fork.id] = {}
            self._session_operation_records[fork.id] = {}
            self._transcripts[fork.id] = copied_transcript
            if copied_checkpoint is not None:
                self._store_checkpoint_unlocked(fork.id, copied_checkpoint)
            return fork.model_copy(deep=True)

    async def load(self, session_id: str) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.model_copy(deep=True)

    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return SessionStateSnapshot(
                id=session.id,
                status=session.status,
                updated_at=session.updated_at,
                last_activity_at=session.last_activity_at,
            )

    async def inspect_identity(self, session_id: str) -> SessionInspectionIdentity:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            labels, label_count, labels_truncated = _bounded_session_inspection_labels(
                session.labels
            )
            return SessionInspectionIdentity(
                id=session.id,
                agent_name=session.agent_name,
                provider_name=session.provider_name,
                model=session.model,
                parent_session_id=session.parent_session_id,
                causal_budget_id=session.causal_budget_id,
                runtime_name=session.runtime_name,
                runtime_version=session.runtime_version,
                environment_name=session.environment_name,
                status=session.status,
                created_at=session.created_at,
                updated_at=session.updated_at,
                last_activity_at=session.last_activity_at,
                run_epoch=session.run_epoch,
                labels=labels,
                label_count=label_count,
                labels_truncated=labels_truncated,
            )

    async def update_status(self, session_id: str, status: SessionStatus) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(status, SessionStatus):
            raise ValueError("Session status must be a SessionStatus.")
        # Delegate to the guarded transition machine with every source status
        # allowed: preserves the unconditional (any -> status) setter contract
        # while sharing one write path and its not-found guard.
        return await self.transition_status(
            session_id,
            from_statuses=set(SessionStatus),
            to_status=status,
        )

    async def update_model(self, session_id: str, model: str) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        model = require_clean_nonblank(model, "model")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            now = datetime.now(UTC)
            updated = session.model_copy(
                update={
                    "model": model,
                    "updated_at": now,
                    "last_activity_at": now,
                }
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def delete_session(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return  # idempotent: deleting a missing session is a no-op
            if session.status in DELETE_BLOCKED_SESSION_STATUSES:
                raise ValueError(
                    f"Cannot delete a session while it is {session.status}; "
                    f"interrupt it first: {session_id}"
                )
            checkpoint = self._checkpoints.get(session_id)
            deletion_now = datetime.now(UTC)
            active_recovery_claim_id = _active_unexpired_incomplete_recovery_claim_id(
                checkpoint,
                now=deletion_now,
            )
            if active_recovery_claim_id is not None:
                raise ValueError(
                    "Cannot delete a session while incomplete-session recovery claim "
                    f"{active_recovery_claim_id} is active: {session_id}"
                )
            active_operation_id = _active_unexpired_session_operation_id(
                checkpoint,
                now=deletion_now,
            )
            if active_operation_id is not None:
                raise ValueError(
                    "Cannot delete a session while durable operation "
                    f"{active_operation_id} is active: {session_id}"
                )
            self._sessions.pop(session_id, None)
            self._events.pop(session_id, None)
            self._event_ids.pop(session_id, None)
            self._persisted_event_side_effect_deliveries = {
                key: delivery
                for key, delivery in self._persisted_event_side_effect_deliveries.items()
                if key[0] != session_id
            }
            self._session_event_records.pop(session_id, None)
            self._pending_action_event_records.pop(session_id, None)
            self._pending_action_rebuilt_lookup_ids.pop(session_id, None)
            self._pending_action_latest_barrier_records.pop(session_id, None)
            self._transcripts.pop(session_id, None)
            self._checkpoints.pop(session_id, None)
            self._session_operation_records.pop(session_id, None)
            self._pending_action_session_ids.discard(session_id)
            self._queued_session_messages_by_idempotency.pop(session_id, None)
            for delivery_mode in SessionMessageDeliveryMode:
                self._pending_session_messages.pop((session_id, delivery_mode), None)
            self._event_records = [
                record for record in self._event_records if record.event.session_id != session_id
            ]
            self._event_records_by_id = {
                key: record
                for key, record in self._event_records_by_id.items()
                if key[0] != session_id
            }
            for event_type, records in list(self._type_event_records.items()):
                remaining = [record for record in records if record.event.session_id != session_id]
                if remaining:
                    self._type_event_records[event_type] = remaining
                else:
                    del self._type_event_records[event_type]
            # Mirror the SQL FK's ON DELETE SET NULL: children keep loading, parent ref cleared.
            for child_id, child in list(self._sessions.items()):
                if child.parent_session_id == session_id:
                    self._sessions[child_id] = child.model_copy(update={"parent_session_id": None})

    async def update_labels(self, session_id: str, labels: dict[str, str]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_labels = copy_label_map(labels, "labels", allow_reserved=False)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            now = datetime.now(UTC)
            updated = session.model_copy(update={"labels": new_labels, "updated_at": now})
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        user_metadata = copy_session_user_metadata(metadata)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            new_metadata = replace_session_user_metadata(session.metadata, user_metadata)
            now = datetime.now(UTC)
            updated = session.model_copy(update={"metadata": new_metadata, "updated_at": now})
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def transition_status(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )

            now = datetime.now(UTC)
            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": now,
                    "last_activity_at": now,
                    "run_epoch": session.run_epoch + (to_status == SessionStatus.RUNNING),
                }
            )
            self._sessions[session_id] = updated
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(result)
            return result

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )

            current_checkpoint = self._checkpoints.get(session_id)
            transformed_checkpoint = checkpoint_transform(
                session.model_copy(deep=True),
                None if current_checkpoint is None else deepcopy(current_checkpoint),
            )
            if transformed_checkpoint is not None:
                transformed_checkpoint = copy_json_value(
                    transformed_checkpoint,
                    "checkpoint",
                )

            now = datetime.now(UTC)
            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": now,
                    "last_activity_at": now,
                    "run_epoch": session.run_epoch + (to_status == SessionStatus.RUNNING),
                }
            )
            self._sessions[session_id] = updated
            if transformed_checkpoint is not None:
                self._store_checkpoint_unlocked(session_id, transformed_checkpoint)
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(result)
            return result

    async def transition_status_if_no_queued_messages(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(from_statuses, "from_statuses")
        if not isinstance(to_status, SessionStatus):
            raise ValueError("to_status must be a SessionStatus.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )
            if any(
                (session_id, delivery_mode) in self._pending_session_messages
                for delivery_mode in SessionMessageDeliveryMode
            ):
                raise SessionQueuedMessagesPending(
                    f"Session has durable queued messages: {session_id}"
                )
            now = datetime.now(UTC)
            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": now,
                    "last_activity_at": now,
                    "run_epoch": session.run_epoch + (to_status == SessionStatus.RUNNING),
                }
            )
            self._sessions[session_id] = updated
            result = updated.model_copy(deep=True)
            if to_status == SessionStatus.RUNNING:
                _activate_session_run_fence(result)
            return result

    async def fence_stalled_run(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        inactive_before: datetime,
    ) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        if inactive_before.tzinfo is None or inactive_before.utcoffset() is None:
            raise ValueError("inactive_before must be timezone-aware.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.status not in allowed_statuses or session.last_activity_at > inactive_before:
                return None
            fenced = session.model_copy(
                update={
                    "run_epoch": session.run_epoch + 1,
                    "last_activity_at": datetime.now(UTC),
                }
            )
            self._sessions[session_id] = fenced
            result = fenced.model_copy(deep=True)
            _activate_session_run_fence(result)
            return result

    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        allowed_statuses = _validate_status_set(statuses, "statuses")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(f"Session status cannot be fenced: {session.status}")
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                None if current is None else deepcopy(current),
            )
            if transformed is None:
                raise ValueError("Fenced checkpoint transform must return a checkpoint.")
            transformed = copy_json_value(transformed, "checkpoint")
            fenced = session.model_copy(
                update={
                    "run_epoch": session.run_epoch + 1,
                    "last_activity_at": datetime.now(UTC),
                }
            )
            self._store_checkpoint_unlocked(session_id, transformed)
            self._sessions[session_id] = fenced
            result = fenced.model_copy(deep=True)
            _activate_session_run_fence(result)
            return result

    async def release_run_fence(self, session_id: str) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        expected_run_epoch = _current_session_run_epoch(session_id)
        if expected_run_epoch is None:
            return
        try:
            async with self._lock:
                session = self._sessions.get(session_id)
                if session is not None and session.run_epoch == expected_run_epoch:
                    self._sessions[session_id] = session.model_copy(
                        update={"run_epoch": session.run_epoch + 1}
                    )
        finally:
            _deactivate_session_run_fence(session_id)

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    def _append_events_unlocked(
        self,
        session: Session,
        events: list[Event],
    ) -> Session:
        from cayu.runtime.pending_actions import (
            pending_action_event_lookup_id,
            pending_action_lookup_key,
            project_pending_action_event_record,
        )

        session_id = session.id
        existing_ids = self._event_ids[session_id]
        for event in events:
            if event.id in existing_ids:
                raise ValueError(f"Event already exists for session {session_id}: {event.id}")

        prepared: list[tuple[EventRecord, str, EventRecord | None, str | None]] = []
        next_sequence = self._next_event_sequence
        for event in events:
            stored_event = event.model_copy(deep=True)
            record = EventRecord(sequence=next_sequence, event=stored_event)
            event_type = str(stored_event.type)
            projected_record: EventRecord | None = None
            lookup_key: str | None = None
            if event_type in PENDING_ACTION_EVENT_TYPE_VALUES:
                projected_record = project_pending_action_event_record(record)
                lookup_id = pending_action_event_lookup_id(stored_event)
                if lookup_id is not None:
                    lookup_key = pending_action_lookup_key(lookup_id)
            prepared.append((record, event_type, projected_record, lookup_key))
            next_sequence += 1

        session_records = self._session_event_records.setdefault(session_id, [])
        for record, event_type, projected_record, lookup_key in prepared:
            stored_event = record.event
            self._events[session_id].append(stored_event)
            self._event_records.append(record)
            self._event_records_by_id[(session_id, stored_event.id)] = record
            session_records.append(record)
            if projected_record is not None:
                if event_type in PENDING_ACTION_BARRIER_EVENT_TYPE_VALUES:
                    self._pending_action_latest_barrier_records[session_id] = projected_record
                elif lookup_key is not None:
                    by_lookup_id = self._pending_action_event_records.setdefault(session_id, {})
                    by_event_type = by_lookup_id.setdefault(lookup_key, {})
                    by_event_type[event_type] = projected_record
            self._type_event_records.setdefault(event_type, []).append(record)
            existing_ids.add(stored_event.id)
            if event_type != str(EventType.RUNTIME_SINK_FAILED):
                self._persisted_event_side_effect_deliveries[(session_id, stored_event.id)] = (
                    PersistedEventSideEffectDelivery(
                        session_id=session_id,
                        event_id=stored_event.id,
                        event_sequence=record.sequence,
                        status=PersistedEventSideEffectStatus.PENDING,
                    )
                )
        self._next_event_sequence = next_sequence
        if not events:
            return session
        return session.model_copy(update={"last_activity_at": datetime.now(UTC)})

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id, copied_events = _copy_session_event_batch(session_id, events)

        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            self._sessions[session_id] = self._append_events_unlocked(session, copied_events)

    async def claim_persisted_event_side_effect(
        self,
        *,
        session_id: str | None = None,
        event_id: str | None = None,
        lease_seconds: float = 300.0,
    ) -> PersistedEventSideEffectClaim | None:
        if session_id is not None:
            session_id = require_clean_nonblank(session_id, "session_id")
        if event_id is not None:
            event_id = require_clean_nonblank(event_id, "event_id")
        if (session_id is None) != (event_id is None):
            raise ValueError("session_id and event_id must be supplied together.")
        if type(lease_seconds) not in {int, float} or lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than 0.")
        async with self._lock:
            now = datetime.now(UTC)
            deliveries = sorted(
                self._persisted_event_side_effect_deliveries.values(),
                key=lambda delivery: delivery.event_sequence,
            )
            for delivery in deliveries:
                if session_id is not None and (
                    delivery.session_id != session_id or delivery.event_id != event_id
                ):
                    continue
                if delivery.status in {
                    PersistedEventSideEffectStatus.DELIVERED,
                    PersistedEventSideEffectStatus.DEAD_LETTERED,
                }:
                    continue
                if (
                    delivery.status is PersistedEventSideEffectStatus.LEASED
                    and delivery.lease_expires_at is not None
                    and delivery.lease_expires_at > now
                ):
                    continue
                if (
                    delivery.status is PersistedEventSideEffectStatus.FAILED
                    and delivery.next_attempt_at is not None
                    and delivery.next_attempt_at > now
                ):
                    continue
                record = self._event_records_by_id.get((delivery.session_id, delivery.event_id))
                if record is None:
                    raise RuntimeError("Persisted side-effect delivery lost its source event.")
                claim = PersistedEventSideEffectClaim(
                    session_id=delivery.session_id,
                    event_id=delivery.event_id,
                    event_sequence=delivery.event_sequence,
                    event=record.event,
                    attempt=delivery.attempts + 1,
                    lease_expires_at=now + timedelta(seconds=float(lease_seconds)),
                )
                self._persisted_event_side_effect_deliveries[
                    (delivery.session_id, delivery.event_id)
                ] = delivery.model_copy(
                    update={
                        "status": PersistedEventSideEffectStatus.LEASED,
                        "attempts": claim.attempt,
                        "claim_id": claim.claim_id,
                        "lease_expires_at": claim.lease_expires_at,
                        "next_attempt_at": None,
                        "last_error": None,
                        "updated_at": now,
                    },
                    deep=True,
                )
                return claim
            return None

    async def get_persisted_event_side_effect_delivery(
        self,
        *,
        session_id: str,
        event_id: str,
    ) -> PersistedEventSideEffectDelivery | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        event_id = require_clean_nonblank(event_id, "event_id")
        async with self._lock:
            delivery = self._persisted_event_side_effect_deliveries.get((session_id, event_id))
            return None if delivery is None else delivery.model_copy(deep=True)

    async def mark_persisted_event_side_effect_delivered(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        async with self._lock:
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            updated = delivery.model_copy(
                update={
                    "status": PersistedEventSideEffectStatus.DELIVERED,
                    "claim_id": None,
                    "lease_expires_at": None,
                    "next_attempt_at": None,
                    "last_error": None,
                    "updated_at": datetime.now(UTC),
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                updated
            )
            return updated.model_copy(deep=True)

    async def mark_persisted_event_side_effect_failed(
        self,
        claim: PersistedEventSideEffectClaim,
        *,
        error: str,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> PersistedEventSideEffectDelivery:
        claim = PersistedEventSideEffectClaim.model_validate(claim)
        if type(error) is not str or not error.strip():
            raise ValueError("error must be a non-empty string.")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than or equal to 1.")
        if (
            type(retry_delay_seconds) not in {int, float}
            or not math.isfinite(retry_delay_seconds)
            or retry_delay_seconds < 0
        ):
            raise ValueError("retry_delay_seconds must be a finite non-negative number.")
        dead_lettered = claim.attempt >= max_attempts
        async with self._lock:
            now = datetime.now(UTC)
            delivery = self._matching_persisted_event_side_effect_claim_unlocked(claim)
            updated = delivery.model_copy(
                update={
                    "status": (
                        PersistedEventSideEffectStatus.DEAD_LETTERED
                        if dead_lettered
                        else PersistedEventSideEffectStatus.FAILED
                    ),
                    "claim_id": None,
                    "lease_expires_at": None,
                    "next_attempt_at": (
                        None
                        if dead_lettered
                        else now + timedelta(seconds=float(retry_delay_seconds))
                    ),
                    "last_error": error,
                    "updated_at": now,
                },
                deep=True,
            )
            self._persisted_event_side_effect_deliveries[(claim.session_id, claim.event_id)] = (
                updated
            )
            return updated.model_copy(deep=True)

    async def list_persisted_event_side_effect_deliveries(
        self,
        *,
        statuses: set[PersistedEventSideEffectStatus] | None = None,
        claimable_only: bool = False,
        after_sequence: int | None = None,
        limit: int = 100,
    ) -> list[PersistedEventSideEffectDelivery]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000.")
        if type(claimable_only) is not bool:
            raise TypeError("claimable_only must be a bool.")
        if after_sequence is not None and (type(after_sequence) is not int or after_sequence < 0):
            raise ValueError("after_sequence must be a non-negative integer.")
        selected_statuses = (
            None
            if statuses is None
            else {PersistedEventSideEffectStatus(status) for status in statuses}
        )
        async with self._lock:
            now = datetime.now(UTC)
            deliveries = sorted(
                self._persisted_event_side_effect_deliveries.values(),
                key=lambda delivery: delivery.event_sequence,
            )
            return [
                delivery.model_copy(deep=True)
                for delivery in deliveries
                if after_sequence is None or delivery.event_sequence > after_sequence
                if selected_statuses is None or delivery.status in selected_statuses
                if not claimable_only
                or (
                    delivery.status
                    in {
                        PersistedEventSideEffectStatus.PENDING,
                        PersistedEventSideEffectStatus.FAILED,
                    }
                    and (
                        delivery.status is not PersistedEventSideEffectStatus.FAILED
                        or delivery.next_attempt_at is None
                        or delivery.next_attempt_at <= now
                    )
                )
                or (
                    delivery.status is PersistedEventSideEffectStatus.LEASED
                    and delivery.lease_expires_at is not None
                    and delivery.lease_expires_at <= now
                )
            ][:limit]

    def _matching_persisted_event_side_effect_claim_unlocked(
        self,
        claim: PersistedEventSideEffectClaim,
    ) -> PersistedEventSideEffectDelivery:
        delivery = self._persisted_event_side_effect_deliveries.get(
            (claim.session_id, claim.event_id)
        )
        if delivery is None:
            raise ValueError("Persisted event side-effect delivery was not found.")
        if (
            delivery.status is not PersistedEventSideEffectStatus.LEASED
            or delivery.claim_id != claim.claim_id
            or delivery.attempts != claim.attempt
        ):
            raise PersistedEventSideEffectClaimLost(
                "Persisted event side-effect claim is no longer active."
            )
        return delivery

    async def enqueue_session_message(
        self,
        request: EnqueueSessionMessageRequest,
    ) -> EnqueueSessionMessageResult:
        request = copy_enqueue_session_message_request(request)
        async with self._lock:
            session = self._sessions.get(request.session_id)
            if session is None:
                raise KeyError(f"Session not found: {request.session_id}")
            messages_by_idempotency = self._queued_session_messages_by_idempotency.get(
                request.session_id
            )
            existing = (
                None
                if messages_by_idempotency is None
                else messages_by_idempotency.get(request.idempotency_key)
            )
            if existing is not None:
                _validate_equivalent_queued_session_message(existing, request)
                record = self._event_records_by_id.get(
                    (request.session_id, existing.accepted_event_id)
                )
                if record is None:
                    raise RuntimeError(
                        "Queued session message is missing its durable acceptance event."
                    )
                return EnqueueSessionMessageResult(
                    message=existing,
                    event=record.event,
                    replayed=True,
                )
            if session.status not in {SessionStatus.PENDING, SessionStatus.RUNNING}:
                raise SessionStatusConflict(
                    "Session messages may be enqueued only while a session is pending or running."
                )
            accepted_at = datetime.now(UTC)
            queue_id = str(uuid4())
            ordering_key = self._next_session_message_ordering_key
            accepted_event = Event(
                type=EventType.SESSION_MESSAGE_QUEUED,
                session_id=session.id,
                agent_name=session.agent_name,
                environment_name=session.environment_name,
                timestamp=accepted_at,
                payload=_queued_session_message_event_payload(
                    queue_id=queue_id,
                    delivery_mode=request.delivery_mode,
                    ordering_key=ordering_key,
                    actor=request.requested_by,
                    run_epoch=session.run_epoch,
                    transcript_cursor=len(self._transcripts.get(session.id, [])),
                ),
            )
            queued_message = SessionQueuedMessage(
                queue_id=queue_id,
                session_id=session.id,
                idempotency_key=request.idempotency_key,
                content=request.content,
                delivery_mode=request.delivery_mode,
                status=SessionMessageQueueStatus.QUEUED,
                ordering_key=ordering_key,
                accepted_run_epoch=session.run_epoch,
                accepted_transcript_cursor=len(self._transcripts.get(session.id, [])),
                accepted_event_id=accepted_event.id,
                accepted_at=accepted_at,
                requested_by=_audit_resolution_actor(request.requested_by),
            )
            self._sessions[session.id] = self._append_events_unlocked(
                session,
                [accepted_event],
            )
            self._queued_session_messages_by_idempotency.setdefault(session.id, {})[
                queued_message.idempotency_key
            ] = queued_message
            pending_key = (session.id, queued_message.delivery_mode)
            self._pending_session_messages.setdefault(pending_key, deque()).append(queued_message)
            self._next_session_message_ordering_key += 1
            return EnqueueSessionMessageResult(
                message=queued_message,
                event=accepted_event,
            )

    async def deliver_queued_session_messages(
        self,
        session_id: str,
        *,
        include_on_idle: bool,
        eligible_through: int | None = None,
        limit: int = SESSION_MESSAGE_DELIVERY_BATCH_LIMIT,
    ) -> SessionMessageDeliveryBatch:
        session_id = require_clean_nonblank(session_id, "session_id")
        if type(include_on_idle) is not bool:
            raise TypeError("include_on_idle must be a bool.")
        if eligible_through is not None and eligible_through < 0:
            raise ValueError("eligible_through must be greater than or equal to zero.")
        if type(limit) is not int or not 1 <= limit <= SESSION_MESSAGE_DELIVERY_BATCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {SESSION_MESSAGE_DELIVERY_BATCH_LIMIT}.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if session.status != SessionStatus.RUNNING:
                raise SessionStatusConflict(
                    "Queued session messages may be delivered only while running."
                )
            boundary = (
                self._next_session_message_ordering_key - 1
                if eligible_through is None
                else eligible_through
            )

            selected_key: tuple[str, SessionMessageDeliveryMode] | None = None
            selected: list[SessionQueuedMessage] = []
            delivery_modes: tuple[SessionMessageDeliveryMode, ...] = (
                SessionMessageDeliveryMode.NEXT_TURN,
            )
            if include_on_idle:
                delivery_modes += (SessionMessageDeliveryMode.ON_IDLE,)
            for delivery_mode in delivery_modes:
                pending_key = (session_id, delivery_mode)
                pending = self._pending_session_messages.get(pending_key)
                if pending is None or pending[0].ordering_key > boundary:
                    continue
                selected_key = pending_key
                for message in pending:
                    if len(selected) >= limit or message.ordering_key > boundary:
                        break
                    selected.append(message)
                break
            if not selected:
                return SessionMessageDeliveryBatch(
                    eligible_through=boundary,
                    has_more=False,
                )
            if selected_key is None:
                raise RuntimeError("Queued message selection lost its pending index.")

            transcript_cursor = len(self._transcripts.get(session_id, []))
            delivered_at = datetime.now(UTC)
            updated_messages: list[SessionQueuedMessage] = []
            delivery_events: list[Event] = []
            transcript_messages: list[Message] = []
            for offset, queued_message in enumerate(selected, start=1):
                delivered_cursor = transcript_cursor + offset
                delivery_event = Event(
                    type=EventType.SESSION_MESSAGE_DELIVERED,
                    session_id=session.id,
                    agent_name=session.agent_name,
                    environment_name=session.environment_name,
                    timestamp=delivered_at,
                    payload={
                        **_queued_session_message_event_payload(
                            queue_id=queued_message.queue_id,
                            delivery_mode=queued_message.delivery_mode,
                            ordering_key=queued_message.ordering_key,
                            actor=queued_message.requested_by,
                            run_epoch=session.run_epoch,
                            transcript_cursor=delivered_cursor,
                        ),
                        "accepted_run_epoch": queued_message.accepted_run_epoch,
                        "accepted_transcript_cursor": (queued_message.accepted_transcript_cursor),
                    },
                )
                updated_messages.append(
                    queued_message.model_copy(
                        update={
                            "status": SessionMessageQueueStatus.DELIVERED,
                            "delivered_run_epoch": session.run_epoch,
                            "delivered_transcript_cursor": delivered_cursor,
                            "delivered_event_id": delivery_event.id,
                            "delivered_at": delivered_at,
                        },
                        deep=True,
                    )
                )
                delivery_events.append(delivery_event)
                transcript_messages.append(
                    detach_message(Message.text(MessageRole.USER, queued_message.content))
                )

            updated_session = self._append_events_unlocked(session, delivery_events)
            self._transcripts.setdefault(session_id, []).extend(transcript_messages)
            selected_queue = self._pending_session_messages[selected_key]
            for _ in selected:
                selected_queue.popleft()
            if not selected_queue:
                del self._pending_session_messages[selected_key]
            messages_by_idempotency = self._queued_session_messages_by_idempotency[session_id]
            for updated_message in updated_messages:
                messages_by_idempotency[updated_message.idempotency_key] = updated_message
            self._sessions[session_id] = updated_session

            def has_eligible(delivery_mode: SessionMessageDeliveryMode) -> bool:
                pending = self._pending_session_messages.get((session_id, delivery_mode))
                return pending is not None and pending[0].ordering_key <= boundary

            has_more = has_eligible(SessionMessageDeliveryMode.NEXT_TURN) or (
                include_on_idle and has_eligible(SessionMessageDeliveryMode.ON_IDLE)
            )
            return SessionMessageDeliveryBatch(
                messages=tuple(updated_messages),
                events=tuple(delivery_events),
                eligible_through=boundary,
                has_more=has_more,
            )

    async def publish_checkpoint_and_events(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        session_id, copied_events = _copy_session_event_batch(session_id, events)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        allowed_statuses = (
            None
            if expected_statuses is None
            else _validate_status_set(expected_statuses, "expected_statuses")
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if allowed_statuses is not None and session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status is not eligible for checkpoint publication: {session.status}"
                )
            if expected_run_epoch is not None and session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if (
                expected_transcript_cursor is not None
                and current_cursor != expected_transcript_cursor
            ):
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{expected_transcript_cursor}, current {current_cursor}."
                )
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                None if current is None else deepcopy(current),
            )
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            copied_checkpoint = copy_json_value(transformed, "checkpoint")
            updated = self._append_events_unlocked(session, copied_events)
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def load_session_operation(
        self,
        session_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        idempotency_key = require_clean_nonblank(idempotency_key, "idempotency_key")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            record = self._session_operation_records.get(session_id, {}).get(idempotency_key)
            return None if record is None else copy_json_value(record, "session_operation")

    async def publish_session_operation(
        self,
        session_id: str,
        *,
        idempotency_key: str,
        operation_transform: SessionOperationTransform,
        events: list[Event],
        expected_statuses: set[SessionStatus] | None = None,
        expected_run_epoch: int | None = None,
        expected_transcript_cursor: int | None = None,
    ) -> Session:
        session_id, copied_events = _copy_session_event_batch(session_id, events)
        idempotency_key = require_clean_nonblank(idempotency_key, "idempotency_key")
        if operation_transform is None:
            raise TypeError("operation_transform is required.")
        allowed_statuses = (
            None
            if expected_statuses is None
            else _validate_status_set(expected_statuses, "expected_statuses")
        )
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if allowed_statuses is not None and session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status is not eligible for checkpoint publication: {session.status}"
                )
            if expected_run_epoch is not None and session.run_epoch != expected_run_epoch:
                raise SessionRunFenced(
                    f"Session source run epoch is stale: expected {expected_run_epoch}, "
                    f"current {session.run_epoch}."
                )
            current_cursor = len(self._transcripts.get(session_id, []))
            if (
                expected_transcript_cursor is not None
                and current_cursor != expected_transcript_cursor
            ):
                raise ValueError(
                    "Session source transcript cursor is stale: expected "
                    f"{expected_transcript_cursor}, current {current_cursor}."
                )
            current_checkpoint = self._checkpoints.get(session_id)
            current_record = self._session_operation_records.get(session_id, {}).get(
                idempotency_key
            )
            publication = operation_transform(
                session.model_copy(deep=True),
                None if current_checkpoint is None else deepcopy(current_checkpoint),
                None
                if current_record is None
                else copy_json_value(current_record, "session_operation"),
            )
            if type(publication) is not SessionOperationPublication:
                raise TypeError(
                    "Session operation transform must return a SessionOperationPublication."
                )
            copied_checkpoint = copy_json_value(publication.checkpoint, "checkpoint")
            copied_records = copy_json_value(
                publication.operation_records,
                "operation_records",
            )
            updated = self._append_events_unlocked(session, copied_events)
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            operation_records = self._session_operation_records.setdefault(session_id, {})
            operation_records.update(copied_records)
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def load_events(self, session_id: str) -> list[Event]:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [event.model_copy(deep=True) for event in self._events.get(session_id, [])]

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        query = copy_event_query(query)
        # `event_type` and `event_types` are mutually exclusive, so they
        # collapse into one filter set (empty = no type filter).
        if query.event_type is not None:
            event_types = frozenset((str(query.event_type),))
        else:
            event_types = frozenset(str(event_type) for event_type in query.event_types)
        excluded_event_types = frozenset(
            str(event_type) for event_type in query.exclude_event_types
        )
        async with self._lock:
            candidates = self._query_candidate_records(query, event_types)
            start = (
                bisect_right(candidates, query.after_sequence, key=lambda record: record.sequence)
                if query.after_sequence is not None
                else 0
            )
            stop = (
                bisect_left(candidates, query.before_sequence, key=lambda record: record.sequence)
                if query.before_sequence is not None
                else len(candidates)
            )
            indexes = (
                range(stop - 1, start - 1, -1)
                if query.order_by == EventOrder.SEQUENCE_DESC
                else range(start, stop)
            )
            records: list[EventRecord] = []
            for index in indexes:
                record = candidates[index]
                if not _event_record_matches(
                    record,
                    query,
                    event_types,
                    excluded_event_types,
                ):
                    continue
                if not _event_record_matches_session(record, query, self._sessions):
                    continue
                records.append(
                    EventRecord(
                        sequence=record.sequence,
                        event=record.event,
                    )
                )
                if len(records) == query.limit:
                    break
            return records

    def _query_candidate_records(
        self,
        query: EventQuery,
        event_types: frozenset[str],
    ) -> list[EventRecord]:
        """Pick the narrowest index that still covers a query's candidate rows.

        All returned lists stay sequence-ascending so downstream ordering and
        ``after_sequence`` paging behave exactly as a full scan would.
        """
        if query.event_id is not None and query.session_id is not None:
            record = self._event_records_by_id.get((query.session_id, query.event_id))
            return [] if record is None else [record]
        if query.session_id is not None:
            return self._session_event_records.get(query.session_id, [])
        if query.session_ids:
            merged: list[EventRecord] = []
            for session_id in query.session_ids:
                merged.extend(self._session_event_records.get(session_id, []))
            merged.sort(key=lambda record: record.sequence)
            return merged
        if event_types:
            if len(event_types) == 1:
                return self._type_event_records.get(next(iter(event_types)), [])
            merged = []
            for event_type in event_types:
                merged.extend(self._type_event_records.get(event_type, []))
            merged.sort(key=lambda record: record.sequence)
            return merged
        return self._event_records

    async def summarize_events(self, session_id: str) -> EventSummary:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")

            records = self._session_event_records.get(session_id, [])
            return event_summary_from_records(session_id, records)

    async def summarize_outcome(self, session_id: str) -> SessionOutcome:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")

            records = self._session_event_records.get(session_id, [])
            return session_outcome_from_records(session, records)

    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        return await self._list_sessions(query, pending_interruption_cascade_only=False)

    async def aggregate_operational_snapshot(
        self,
        filters: SessionAggregateFilter | None = None,
    ) -> SessionOperationalSnapshot:
        filters = copy_session_aggregate_filter(filters)
        session_query = session_query_from_aggregate_filter(filters)
        async with self._lock:
            as_of = datetime.now(UTC)
            counts = {status: 0 for status in SessionStatus}
            total_count = 0
            for session in self._sessions.values():
                if _session_matches(session, session_query):
                    counts[session.status] += 1
                    total_count += 1
            return SessionOperationalSnapshot(
                as_of=as_of,
                total_count=total_count,
                counts_by_status=SessionStatusCounts.model_validate(counts),
                accuracy=EXACT_AGGREGATE.model_copy(),
            )

    async def aggregate_usage(self, query: UsageRollupQuery) -> UsageRollupStoreResult:
        query = copy_usage_rollup_query(query)
        session_query = session_query_from_aggregate_filter(query.sessions)
        async with self._lock:
            as_of = datetime.now(UTC)
            matching_session_count = 0
            active_session_count = 0
            for session in self._sessions.values():
                if not _session_matches(session, session_query):
                    continue
                matching_session_count += 1
                active_session_count += session.status in {
                    SessionStatus.PENDING,
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                }

            def matching_session_records() -> Iterable[tuple[str, Iterable[EventRecord]]]:
                for session_id, session in self._sessions.items():
                    if _session_matches(session, session_query):
                        yield session_id, self._session_event_records.get(session_id, ())

            return _usage_rollup_from_session_records(
                session_records=matching_session_records,
                query=query,
                as_of=as_of,
                matching_session_count=matching_session_count,
                active_session_count=active_session_count,
            )

    async def list_sessions_with_pending_interruption_cascade(
        self,
        query: SessionQuery | None = None,
    ) -> SessionListResult:
        return await self._list_sessions(query, pending_interruption_cascade_only=True)

    async def query_pending_actions(
        self,
        query: PendingActionQuery | None = None,
    ) -> PendingActionListResult:
        from cayu.runtime.pending_actions import (
            PENDING_ACTION_CHECKPOINT_KEYS,
            PENDING_ACTION_SESSION_STATUSES,
            pending_action_event_projection_bytes,
            pending_action_from_records,
            pending_action_matches_query,
            pending_action_source_is_invalid,
            project_pending_action_checkpoint,
            select_pending_action_indexed_records,
        )

        if query is None:
            query = PendingActionQuery()
        elif type(query) is not PendingActionQuery:
            raise TypeError("Pending-action queries must be PendingActionQuery instances.")
        else:
            query = query.model_copy(deep=True)

        candidate_limit = min(query.limit * 4, 800) + 1
        async with self._lock:
            if query.session_id is None:
                candidate_ids = self._pending_action_session_ids
            elif query.session_id in self._pending_action_session_ids:
                candidate_ids = {query.session_id}
            else:
                candidate_ids = set()
            candidates = [
                session
                for session_id in candidate_ids
                if (session := self._sessions.get(session_id)) is not None
                and session.status in PENDING_ACTION_SESSION_STATUSES
                and (query.agent_name is None or session.agent_name == query.agent_name)
                and (
                    query.environment_name is None
                    or session.environment_name == query.environment_name
                )
            ]
            candidates = _sort_sessions(candidates, SessionOrder.UPDATED_AT_DESC)
            if query.cursor is not None:
                cursor_dt, cursor_id = decode_session_cursor(query.cursor)
                candidates = [
                    session
                    for session in candidates
                    if _session_after_cursor(
                        session,
                        SessionOrder.UPDATED_AT_DESC,
                        cursor_dt,
                        cursor_id,
                    )
                ]

            candidate_window = candidates[:candidate_limit]
            has_more_candidates = len(candidate_window) == candidate_limit
            inspected_candidates = candidate_window[: candidate_limit - 1]
            actions: list[PendingActionRecord] = []
            issues: list[PendingActionIssue] = []
            materialized_source_bytes = 0
            inspected = 0
            more_matching = False
            last_inspected_session: PendingActionSession | None = None
            for session in inspected_candidates:
                previous_last_inspected_session = last_inspected_session
                projected_session = PendingActionSession.from_session(session)
                checkpoint = self._checkpoints.get(session.id)
                pending_checkpoint_source = (
                    None
                    if checkpoint is None
                    else {
                        key: checkpoint[key]
                        for key in PENDING_ACTION_CHECKPOINT_KEYS
                        if checkpoint.get(key) is not None
                    }
                )
                pending_round_source = (
                    pending_checkpoint_source.get("pending_tool_round")
                    if pending_checkpoint_source is not None
                    else None
                )
                pending_tool_calls = (
                    pending_round_source.get("tool_calls")
                    if type(pending_round_source) is dict
                    else None
                )
                source_too_complex = (
                    type(pending_tool_calls) is list
                    and len(pending_tool_calls) > MAX_PENDING_ACTION_TOOL_CALLS
                )
                candidate_size = JsonUtf8SizeCounter(query.max_result_bytes)
                source_fits = (
                    not source_too_complex
                    and candidate_size.value(projected_session)
                    and candidate_size.value(pending_checkpoint_source)
                )
                projected_checkpoint = None
                records: list[EventRecord] = []
                if source_fits:
                    projected_checkpoint = project_pending_action_checkpoint(checkpoint)
                    records = select_pending_action_indexed_records(
                        projected_checkpoint,
                        self._pending_action_event_records.get(session.id, {}),
                        self._pending_action_latest_barrier_records.get(session.id),
                    )
                    source_fits = all(
                        (projection_bytes := pending_action_event_projection_bytes(record)) is None
                        or projection_bytes <= query.max_result_bytes
                        for record in records
                    ) and all(candidate_size.value(record) for record in records)
                candidate_source_bytes = query.max_result_bytes - candidate_size.remaining
                if source_fits and (
                    materialized_source_bytes + candidate_source_bytes > query.max_result_bytes
                ):
                    more_matching = True
                    break

                inspected += 1
                last_inspected_session = projected_session
                if not source_fits:
                    if len(actions) + len(issues) == query.limit:
                        more_matching = True
                        inspected -= 1
                        last_inspected_session = previous_last_inspected_session
                        break
                    issues.append(
                        PendingActionIssue.source_too_complex(
                            projected_session,
                            max_tool_calls=MAX_PENDING_ACTION_TOOL_CALLS,
                        )
                        if source_too_complex
                        else PendingActionIssue.source_too_large(
                            projected_session,
                            max_bytes=query.max_result_bytes,
                        )
                    )
                    continue

                materialized_source_bytes += candidate_source_bytes
                action = pending_action_from_records(
                    projected_session,
                    records,
                    projected_checkpoint,
                )
                if pending_action_source_is_invalid(
                    projected_session,
                    projected_checkpoint,
                    action,
                    records,
                ):
                    if len(actions) + len(issues) == query.limit:
                        more_matching = True
                        inspected -= 1
                        last_inspected_session = previous_last_inspected_session
                        break
                    issues.append(PendingActionIssue.source_invalid(projected_session))
                    continue
                if action is None:
                    continue
                if query.kind is not None and action.kind != query.kind:
                    continue
                if not pending_action_matches_query(action, query.q):
                    continue
                if len(actions) + len(issues) == query.limit:
                    more_matching = True
                    inspected -= 1
                    last_inspected_session = previous_last_inspected_session
                    break
                actions.append(action)

            has_more = more_matching or has_more_candidates
            cursor_session = last_inspected_session
            next_cursor = (
                encode_session_cursor(cursor_session, SessionOrder.UPDATED_AT_DESC)
                if has_more and cursor_session is not None
                else None
            )
            return enforce_pending_action_result_size(
                PendingActionListResult(
                    actions=actions,
                    issues=issues,
                    next_cursor=next_cursor,
                    has_more=has_more,
                    # A cursor query sees only the suffix after that cursor, so a
                    # final page's local length is not a global total. The first
                    # page can report an exact total only when it exhausted the
                    # candidate set without hitting either bound.
                    total_count=(
                        len(actions) + len(issues)
                        if query.cursor is None and not has_more
                        else None
                    ),
                    inspected_candidate_count=inspected,
                ),
                max_bytes=query.max_result_bytes,
            )

    async def _list_sessions(
        self,
        query: SessionQuery | None,
        *,
        pending_interruption_cascade_only: bool,
    ) -> SessionListResult:
        query = copy_session_query(query)
        base_query = query.model_copy(update={"debug_state": None})
        async with self._lock:
            candidates = (
                (
                    self._sessions[session_id]
                    for session_id, checkpoint in self._checkpoints.items()
                    if "pending_interruption_cascade" in checkpoint and session_id in self._sessions
                )
                if pending_interruption_cascade_only
                else self._sessions.values()
            )
            matching = [
                session
                for session in candidates
                if _session_matches(session, base_query)
                and _session_matches_debug_state(
                    session,
                    self._session_event_records.get(session.id, []),
                    query.debug_state,
                )
            ]
            total = len(matching) if query.include_total_count else None
            ordered = _sort_sessions(matching, query.order_by)
            if query.cursor is not None:
                cursor_dt, cursor_id = decode_session_cursor(query.cursor)
                ordered = [
                    session
                    for session in ordered
                    if _session_after_cursor(session, query.order_by, cursor_dt, cursor_id)
                ]
                window = ordered[: query.limit + 1]
            else:
                window = ordered[query.offset : query.offset + query.limit + 1]
            page = window[: query.limit]
            next_cursor = session_next_cursor(page, len(window) > query.limit, query.order_by)
            return SessionListResult(
                sessions=[session.model_copy(deep=True) for session in page],
                next_cursor=next_cursor,
                total_count=total,
            )

    async def append_transcript_messages(
        self,
        session_id: str,
        messages: list[Message],
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        copied_messages = _detach_transcript_messages(messages)
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            if not copied_messages:
                return
            self._transcripts[session_id].extend(copied_messages)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )

    async def append_transcript_messages_and_transform_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        copied_messages = _detach_transcript_messages(messages)
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                None if current is None else deepcopy(current),
            )
            if transformed is None:
                raise ValueError("Checkpoint transform must return a checkpoint.")
            copied_checkpoint = copy_json_value(transformed, "checkpoint")
            if copied_messages:
                self._transcripts[session_id].extend(copied_messages)
            self._store_checkpoint_unlocked(session_id, copied_checkpoint)
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )

    async def load_transcript(self, session_id: str) -> list[Message]:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return [detach_message(message) for message in self._transcripts.get(session_id, [])]

    async def query_transcript(self, query: TranscriptQuery) -> TranscriptPage:
        query = copy_transcript_query(query)
        async with self._lock:
            if query.session_id not in self._sessions:
                raise KeyError(f"Session not found: {query.session_id}")

            indexed_messages = list(enumerate(self._transcripts.get(query.session_id, [])))
            if query.role is not None:
                indexed_messages = [
                    (index, message)
                    for index, message in indexed_messages
                    if message.role == query.role
                ]

            page = indexed_messages[query.offset : query.offset + query.limit]
            # Per filter_transcript_records' contract, the include_thinking=False
            # path already isolates — detach only pass-through records.
            records = [
                TranscriptRecord(
                    index=index,
                    message=detach_message(message) if query.include_thinking else message,
                )
                for index, message in page
            ]
            return TranscriptPage(
                records=filter_transcript_records(records, include_thinking=query.include_thinking),
                total_records=len(indexed_messages),
            )

    async def checkpoint(self, session_id: str, state: dict[str, Any]) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            self._store_checkpoint_unlocked(
                session_id,
                copy_json_value(state, "checkpoint"),
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        if checkpoint_transform is None:
            raise TypeError("checkpoint_transform is required.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            _assert_session_run_epoch(session_id, session)
            current = self._checkpoints.get(session_id)
            transformed = checkpoint_transform(
                session.model_copy(deep=True),
                None if current is None else deepcopy(current),
            )
            if transformed is None:
                return
            self._store_checkpoint_unlocked(
                session_id,
                copy_json_value(transformed, "checkpoint"),
            )
            self._sessions[session_id] = session.model_copy(
                update={"last_activity_at": datetime.now(UTC)}
            )

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            if checkpoint is None:
                return None
            return deepcopy(checkpoint)

    async def load_interruption_cascade_marker(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            marker = (
                _INTERRUPTION_CASCADE_MISSING
                if checkpoint is None or "pending_interruption_cascade" not in checkpoint
                else checkpoint["pending_interruption_cascade"]
            )
            return _project_interruption_cascade_marker(marker)


def event_summary_from_records(
    session_id: str,
    records: list[EventRecord],
) -> EventSummary:
    session_id = require_clean_nonblank(session_id, "session_id")
    counts_by_type: dict[str, int] = {}
    latest_record: EventRecord | None = None
    for record in records:
        if record.event.session_id != session_id:
            continue
        event_type = str(record.event.type)
        counts_by_type[event_type] = counts_by_type.get(event_type, 0) + 1
        if latest_record is None or record.sequence > latest_record.sequence:
            latest_record = record
    return EventSummary(
        session_id=session_id,
        total_events=sum(counts_by_type.values()),
        counts_by_type=counts_by_type,
        latest_event=_copy_event_record(latest_record),
    )


def session_outcome_from_records(
    session: Session,
    records: list[EventRecord],
) -> SessionOutcome:
    session = copy_session(session)
    session_records = [record for record in records if record.event.session_id == session.id]

    latest_lifecycle_sequence = 0
    for record in reversed(session_records):
        if _is_outcome_lifecycle_event(record.event):
            latest_lifecycle_sequence = record.sequence
            break

    terminal_record: EventRecord | None = None
    for record in reversed(session_records):
        if record.sequence <= latest_lifecycle_sequence:
            break
        if _is_outcome_terminal_event(record.event):
            terminal_record = record
            break

    retry_record: EventRecord | None = None
    for record in reversed(session_records):
        if record.sequence <= latest_lifecycle_sequence:
            break
        if record.event.type == EventType.MODEL_RETRY:
            retry_record = record
            break

    return session_outcome(
        session,
        terminal_event=terminal_record,
        latest_retry_event=retry_record,
    )


def session_outcome(
    session: Session,
    *,
    terminal_event: EventRecord | None,
    latest_retry_event: EventRecord | None,
) -> SessionOutcome:
    session = copy_session(session)
    terminal_event = _copy_event_record(terminal_event)
    latest_retry_event = _copy_event_record(latest_retry_event)
    if not _terminal_event_matches_status(session, terminal_event):
        terminal_event = None
    reason, details = _outcome_reason_and_details(session, terminal_event)
    return SessionOutcome(
        session_id=session.id,
        status=session.status,
        reason=reason,
        details=details,
        retry=_retry_details(latest_retry_event),
        terminal_event=terminal_event,
        latest_retry_event=latest_retry_event,
    )


def _is_outcome_terminal_event(event: Event) -> bool:
    return event.type in {
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    }


def _is_outcome_lifecycle_event(event: Event) -> bool:
    return event.type in {
        EventType.SESSION_STARTED,
        EventType.SESSION_RESUMED,
    }


def _outcome_reason_and_details(
    session: Session,
    terminal_event: EventRecord | None,
) -> tuple[str, dict[str, Any]]:
    if session.status not in _OUTCOME_TERMINAL_STATUSES:
        return session.status.value, {}
    if terminal_event is None:
        return session.status.value, {}

    event = terminal_event.event
    if event.type != _OUTCOME_EVENT_TYPE_BY_STATUS[session.status]:
        return session.status.value, {}

    payload = event.payload
    if event.type == EventType.SESSION_COMPLETED:
        return "completed", {}
    if event.type == EventType.SESSION_FAILED:
        return "failed", _copy_payload_fields(payload, ("error", "error_type"))
    if event.type == EventType.SESSION_INTERRUPTED:
        reason = _optional_payload_string(payload, "interruption_type") or "interrupted"
        details = _copy_payload_fields(
            payload,
            (
                "interruption_type",
                "reason",
                "limit",
                "maximum",
                "actual",
                "message",
                "error",
                "error_type",
                "manual_recovery_required",
                "tool_call_id",
                "tool_name",
            ),
        )
        return reason, details
    return session.status.value, {}


def _terminal_event_matches_status(
    session: Session,
    terminal_event: EventRecord | None,
) -> bool:
    if terminal_event is None:
        return False
    expected_event_type = _OUTCOME_EVENT_TYPE_BY_STATUS.get(session.status)
    if expected_event_type is None:
        return False
    return terminal_event.event.type == expected_event_type


def _retry_details(latest_retry_event: EventRecord | None) -> dict[str, Any] | None:
    if latest_retry_event is None:
        return None
    return _copy_payload_fields(
        latest_retry_event.event.payload,
        (
            "provider",
            "model",
            "step",
            "attempt",
            "next_attempt",
            "max_attempts",
            "delay_seconds",
            "reason",
            "status_code",
        ),
    )


def _copy_event_record(record: EventRecord | None) -> EventRecord | None:
    if record is None:
        return None
    return EventRecord(sequence=record.sequence, event=record.event)


def _copy_payload_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for field in fields:
        if field in payload and payload[field] is not None:
            copied[field] = copy_json_value(payload[field], field)
    return copied


def _optional_payload_string(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if type(value) is str and value.strip():
        return value
    return None


_OUTCOME_TERMINAL_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.INTERRUPTED,
}

_OUTCOME_EVENT_TYPE_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}


def copy_transcript_messages(messages: list[Message]) -> list[Message]:
    if type(messages) is not list:
        raise TypeError("Transcript messages must be a list.")
    return [copy_message(message) for message in messages]


def _detach_transcript_messages(messages: list[Message]) -> list[Message]:
    if type(messages) is not list:
        raise TypeError("Transcript messages must be a list.")
    return [detach_message(message) for message in messages]


def copy_run_request(request: RunRequest) -> RunRequest:
    if type(request) is not RunRequest:
        raise TypeError("Session creation requires a RunRequest.")
    messages = getattr(request, "messages", None)
    if type(messages) is not list:
        raise ValueError("RunRequest messages must be a list.")
    return RunRequest(
        agent_name=request.agent_name,
        messages=[copy_message(message) for message in messages],
        session_id=request.session_id,
        parent_session_id=request.parent_session_id,
        causal_budget_id=request.causal_budget_id,
        task_id=request.task_id,
        task_worker_id=request.task_worker_id,
        provider_name=request.provider_name,
        model=request.model,
        environment_name=request.environment_name,
        labels=copy_label_map(request.labels, "labels"),
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )


def copy_resume_request(request: ResumeRequest) -> ResumeRequest:
    if type(request) is not ResumeRequest:
        raise TypeError("Session resume requires a ResumeRequest.")
    messages = getattr(request, "messages", None)
    if type(messages) is not list:
        raise ValueError("ResumeRequest messages must be a list.")
    return ResumeRequest(
        session_id=request.session_id,
        messages=[copy_message(message) for message in messages],
        model=request.model,
        metadata=copy_json_value(request.metadata, "metadata"),
        max_steps=request.max_steps,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        retry_policy=copy_retry_policy(request.retry_policy) if request.retry_policy else None,
        structured_output=copy_structured_output_spec(request.structured_output),
        thinking=request.thinking,
        loop_policies=validate_loop_policies(request.loop_policies, field_name="loop_policies"),
    )


def copy_compact_session_request(request: CompactSessionRequest) -> CompactSessionRequest:
    if type(request) is not CompactSessionRequest:
        raise TypeError("Session compaction requires a CompactSessionRequest.")
    return CompactSessionRequest(
        session_id=request.session_id,
        idempotency_key=request.idempotency_key,
        expected_run_epoch=request.expected_run_epoch,
        expected_transcript_cursor=request.expected_transcript_cursor,
        reason=request.reason,
        instructions=request.instructions,
        limits=copy_run_limits(request.limits),
        budget_limits=copy_request_budget_limits(request.budget_limits),
        requested_by=copy_resolution_actor(request.requested_by),
    )


def copy_enqueue_session_message_request(
    request: EnqueueSessionMessageRequest,
) -> EnqueueSessionMessageRequest:
    if type(request) is not EnqueueSessionMessageRequest:
        raise TypeError("Queued input requires an EnqueueSessionMessageRequest.")
    return EnqueueSessionMessageRequest(
        session_id=request.session_id,
        idempotency_key=request.idempotency_key,
        content=request.content,
        delivery_mode=request.delivery_mode,
        requested_by=copy_resolution_actor(request.requested_by),
    )


def copy_interrupt_session_request(request: InterruptSessionRequest) -> InterruptSessionRequest:
    if type(request) is not InterruptSessionRequest:
        raise TypeError("Session interruption requires an InterruptSessionRequest.")
    return InterruptSessionRequest(
        session_id=request.session_id,
        reason=request.reason,
        metadata=copy_json_value(request.metadata, "metadata"),
        requested_by=copy_resolution_actor(request.requested_by),
    )


def copy_incomplete_session_recovery_request(
    request: IncompleteSessionRecoveryRequest,
) -> IncompleteSessionRecoveryRequest:
    if type(request) is not IncompleteSessionRecoveryRequest:
        raise TypeError("Incomplete session recovery requires an IncompleteSessionRecoveryRequest.")
    return IncompleteSessionRecoveryRequest(
        session_id=request.session_id,
        inactive_before=request.inactive_before,
        reason=request.reason,
        metadata=copy_json_value(request.metadata, "metadata"),
    )


def copy_incomplete_sessions_recovery_request(
    request: IncompleteSessionsRecoveryRequest,
) -> IncompleteSessionsRecoveryRequest:
    if type(request) is not IncompleteSessionsRecoveryRequest:
        raise TypeError(
            "Incomplete sessions recovery requires an IncompleteSessionsRecoveryRequest."
        )
    return IncompleteSessionsRecoveryRequest(
        statuses=set(request.statuses),
        limit=request.limit,
        inactive_before=request.inactive_before,
        reason=request.reason,
        metadata=copy_json_value(request.metadata, "metadata"),
    )


def copy_fork_session_request(request: ForkSessionRequest) -> ForkSessionRequest:
    if type(request) is not ForkSessionRequest:
        raise TypeError("Session fork requires a ForkSessionRequest.")
    return ForkSessionRequest(
        source_session_id=request.source_session_id,
        session_id=request.session_id,
        agent_name=request.agent_name,
        model=request.model,
        environment_name=request.environment_name,
        transcript_cursor=request.transcript_cursor,
        copy_checkpoint=request.copy_checkpoint,
        metadata=copy_json_value(request.metadata, "metadata"),
    )


def copy_session(session: Session) -> Session:
    if type(session) is not Session:
        raise TypeError("Session copy requires a Session.")
    return Session(
        id=session.id,
        agent_name=session.agent_name,
        provider_name=session.provider_name,
        model=session.model,
        parent_session_id=session.parent_session_id,
        causal_budget_id=session.causal_budget_id,
        runtime_name=session.runtime_name,
        runtime_version=session.runtime_version,
        environment_name=session.environment_name,
        status=session.status,
        created_at=session.created_at,
        updated_at=session.updated_at,
        last_activity_at=session.last_activity_at,
        run_epoch=session.run_epoch,
        labels=copy_label_map(session.labels, "labels"),
        metadata=copy_json_value(session.metadata, "metadata"),
    )


def copy_session_identity(identity: SessionIdentity) -> SessionIdentity:
    if type(identity) is not SessionIdentity:
        raise TypeError("Session creation requires a SessionIdentity.")
    return SessionIdentity(
        provider_name=identity.provider_name,
        model=identity.model,
        runtime_name=identity.runtime_name,
        runtime_version=identity.runtime_version,
    )


def _validate_status_set(
    statuses: set[SessionStatus],
    field_name: str,
) -> set[SessionStatus]:
    if type(statuses) is not set:
        raise TypeError(f"{field_name} must be a set of SessionStatus values.")
    if not statuses:
        raise ValueError(f"{field_name} cannot be empty.")
    for status in statuses:
        if not isinstance(status, SessionStatus):
            raise ValueError(f"{field_name} must contain SessionStatus values.")
    return set(statuses)


def _audit_resolution_actor(actor: ResolutionActor | None) -> ResolutionActor | None:
    if actor is None:
        return None
    return ResolutionActor(
        subject=actor.subject,
        tenant=actor.tenant,
        source=actor.source,
    )


def _queued_session_message_event_payload(
    *,
    queue_id: str,
    delivery_mode: SessionMessageDeliveryMode,
    ordering_key: int,
    actor: ResolutionActor | None,
    run_epoch: int,
    transcript_cursor: int,
) -> dict[str, Any]:
    return {
        "queue_id": queue_id,
        "delivery_mode": str(delivery_mode),
        "ordering_key": ordering_key,
        "actor": resolution_actor_payload(actor),
        "run_epoch": run_epoch,
        "transcript_cursor": transcript_cursor,
    }


def _validate_equivalent_queued_session_message(
    existing: SessionQueuedMessage,
    request: EnqueueSessionMessageRequest,
) -> None:
    if (
        existing.content != request.content
        or existing.delivery_mode != request.delivery_mode
        or resolution_actor_payload(existing.requested_by)
        != resolution_actor_payload(request.requested_by)
    ):
        raise ValueError(
            "Session message idempotency key was already used for a different request."
        )


def _prepare_session_fork_request(
    *,
    source_session_id: str,
    fork: Session,
    source_statuses: set[SessionStatus],
    transcript_cursor: int | None,
) -> tuple[str, Session, set[SessionStatus], int | None]:
    source_session_id = require_clean_nonblank(source_session_id, "source_session_id")
    fork = copy_session(fork)
    allowed_statuses = _validate_status_set(source_statuses, "source_statuses")
    if fork.parent_session_id != source_session_id:
        raise ValueError("Fork parent_session_id must match source_session_id.")
    if transcript_cursor is not None and transcript_cursor < 0:
        raise ValueError("transcript_cursor must be greater than or equal to 0.")
    return source_session_id, fork, allowed_statuses, transcript_cursor


def _validate_session_fork_source(
    *,
    source_session: Session | None,
    source_session_id: str,
    fork: Session,
    allowed_statuses: set[SessionStatus],
    expected_source_run_epoch: int,
) -> Session:
    if source_session is None:
        raise KeyError(f"Session not found: {source_session_id}")
    if source_session.status not in allowed_statuses:
        raise ValueError(f"Source session status is not forkable: {source_session.status}")
    if source_session.run_epoch != expected_source_run_epoch:
        raise ValueError(
            "Source session changed while the fork was being prepared: "
            f"run_epoch {source_session.run_epoch} != {expected_source_run_epoch}"
        )
    if fork.status != source_session.status:
        raise ValueError(
            "Fork status must match source session status: "
            f"{fork.status} != {source_session.status}"
        )
    if fork.provider_name != source_session.provider_name:
        raise ValueError(
            "Fork provider_name must match source session provider_name: "
            f"{fork.provider_name} != {source_session.provider_name}"
        )
    return source_session


def _copy_session_event_batch(session_id: str, events: list[Event]) -> tuple[str, list[Event]]:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(events) is not list:
        raise TypeError("Session events must be a list.")

    copied_events: list[Event] = []
    seen_event_ids: set[str] = set()
    for event in events:
        if type(event) is not Event:
            raise TypeError("Session events must be Event instances.")
        copied_event = copy_event(event)
        if copied_event.session_id != session_id:
            raise ValueError("Event session_id does not match target session.")
        if copied_event.id in seen_event_ids:
            raise ValueError(f"Event already exists for session {session_id}: {copied_event.id}")
        seen_event_ids.add(copied_event.id)
        copied_events.append(copied_event)
    return session_id, copied_events


def copy_session_query(query: SessionQuery | None) -> SessionQuery:
    if query is None:
        return SessionQuery()
    if type(query) is not SessionQuery:
        raise TypeError("Session queries must be SessionQuery instances.")
    return SessionQuery(
        q=query.q,
        status=query.status,
        debug_state=query.debug_state,
        agent_name=query.agent_name,
        provider_name=query.provider_name,
        model=query.model,
        environment_name=query.environment_name,
        parent_session_id=query.parent_session_id,
        causal_budget_id=query.causal_budget_id,
        last_activity_before=query.last_activity_before,
        labels=copy_label_map(query.labels, "labels"),
        label_selectors=copy_label_selector_requirements(query.label_selectors),
        limit=query.limit,
        offset=query.offset,
        cursor=query.cursor,
        include_total_count=query.include_total_count,
        order_by=query.order_by,
    )


def copy_session_aggregate_filter(
    filters: SessionAggregateFilter | None,
) -> SessionAggregateFilter:
    if filters is None:
        return SessionAggregateFilter()
    if type(filters) is not SessionAggregateFilter:
        raise TypeError("Session aggregate filters must be SessionAggregateFilter instances.")
    return SessionAggregateFilter.model_validate(filters.model_dump(mode="python"))


def session_query_from_aggregate_filter(filters: SessionAggregateFilter) -> SessionQuery:
    filters = copy_session_aggregate_filter(filters)
    return SessionQuery(
        agent_name=filters.agent_name,
        provider_name=filters.provider_name,
        model=filters.model,
        environment_name=filters.environment_name,
        parent_session_id=filters.parent_session_id,
        causal_budget_id=filters.causal_budget_id,
        labels=filters.labels,
        label_selectors=filters.label_selectors,
    )


def copy_usage_rollup_query(query: UsageRollupQuery) -> UsageRollupQuery:
    if type(query) is not UsageRollupQuery:
        raise TypeError("Usage aggregate queries must be UsageRollupQuery instances.")
    return UsageRollupQuery.model_validate(query.model_dump(mode="python"))


def copy_event_query(query: EventQuery | None) -> EventQuery:
    if query is None:
        return EventQuery()
    if type(query) is not EventQuery:
        raise TypeError("Event queries must be EventQuery instances.")
    return EventQuery(
        session_id=query.session_id,
        session_ids=query.session_ids,
        event_id=query.event_id,
        causal_budget_id=query.causal_budget_id,
        event_type=query.event_type,
        event_types=query.event_types,
        exclude_event_types=query.exclude_event_types,
        agent_name=query.agent_name,
        environment_name=query.environment_name,
        workflow_name=query.workflow_name,
        tool_name=query.tool_name,
        since=query.since,
        until=query.until,
        after_sequence=query.after_sequence,
        before_sequence=query.before_sequence,
        limit=query.limit,
        order_by=query.order_by,
    )


def copy_transcript_query(query: TranscriptQuery) -> TranscriptQuery:
    if type(query) is not TranscriptQuery:
        raise TypeError("Transcript queries must be TranscriptQuery instances.")
    return TranscriptQuery(
        session_id=query.session_id,
        role=query.role,
        offset=query.offset,
        limit=query.limit,
        include_thinking=query.include_thinking,
    )


def _session_matches(session: Session, query: SessionQuery) -> bool:
    if query.q is not None and not _session_query_text_matches(session, query.q):
        return False
    if query.status is not None and session.status != query.status:
        return False
    if query.debug_state is not None:
        raise ValueError("SessionQuery debug_state requires event-aware store filtering.")
    if query.agent_name is not None and session.agent_name != query.agent_name:
        return False
    if query.provider_name is not None and session.provider_name != query.provider_name:
        return False
    if query.model is not None and session.model != query.model:
        return False
    if query.parent_session_id is not None and session.parent_session_id != query.parent_session_id:
        return False
    if query.causal_budget_id is not None and session.causal_budget_id != query.causal_budget_id:
        return False
    if (
        query.last_activity_before is not None
        and session.last_activity_at > query.last_activity_before
    ):
        return False
    for key, value in query.labels.items():
        if session.labels.get(key) != value:
            return False
    for selector in query.label_selectors:
        if not _label_selector_matches(session.labels, selector):
            return False
    return not (
        query.environment_name is not None and session.environment_name != query.environment_name
    )


@dataclass
class _UsageAccumulator:
    session_count: int = 0
    model_steps: int = 0
    model_steps_with_usage: int = 0
    usage: UsageMetrics = dataclass_field(default_factory=UsageMetrics)

    def add(self, metrics: UsageMetrics | None) -> None:
        self.model_steps += 1
        if metrics is None:
            return
        self.model_steps_with_usage += 1
        self.usage = add_aggregate_usage(self.usage, metrics)

    def totals(self, *, tool_calls: int = 0) -> UsageAggregateTotals:
        return UsageAggregateTotals(
            session_count=self.session_count,
            model_steps=self.model_steps,
            model_steps_with_usage=self.model_steps_with_usage,
            tool_calls=tool_calls,
            usage=self.usage,
        )


_IN_MEMORY_USAGE_GROUP_CANDIDATE_LIMIT = 512
_UsageGroupKey = tuple[str | None, str | None]
_UsageGroupSortKey = tuple[bool, str, bool, str]
_SessionRecordsFactory = Callable[[], Iterable[tuple[str, Iterable[EventRecord]]]]


@dataclass
class _InMemoryUsageGroupCandidates:
    """Bounded heavy-hitter candidates for one in-memory breakdown dimension."""

    limit: int = _IN_MEMORY_USAGE_GROUP_CANDIDATE_LIMIT
    _estimates: dict[_UsageGroupKey, tuple[int, int, int]] = dataclass_field(default_factory=dict)
    _heap: list[tuple[int, int, bool, str, bool, str, int, _UsageGroupKey]] = dataclass_field(
        default_factory=list
    )
    _generation: int = 0
    sampled: bool = False

    def observe(self, key: _UsageGroupKey, metrics: UsageMetrics | None) -> None:
        token_weight = 0 if metrics is None else metrics.total_tokens
        current = self._estimates.get(key)
        if current is not None:
            self._record(key, current[0] + token_weight, current[1] + 1)
            return
        if len(self._estimates) < self.limit:
            self._record(key, token_weight, 1)
            return

        self.sampled = True
        minimum_tokens, minimum_steps, minimum_key = self._pop_minimum()
        del self._estimates[minimum_key]
        self._record(
            key,
            minimum_tokens + token_weight,
            minimum_steps + 1,
        )

    @property
    def keys(self) -> tuple[_UsageGroupKey, ...]:
        return tuple(self._estimates)

    def _record(self, key: _UsageGroupKey, tokens: int, steps: int) -> None:
        self._generation += 1
        generation = self._generation
        self._estimates[key] = tokens, steps, generation
        heapq.heappush(
            self._heap,
            (
                tokens,
                steps,
                *_usage_group_identity_sort_key(key),
                generation,
                key,
            ),
        )
        if len(self._heap) > self.limit * 2:
            self._heap = [
                (
                    item_tokens,
                    item_steps,
                    *_usage_group_identity_sort_key(item_key),
                    item_generation,
                    item_key,
                )
                for item_key, (item_tokens, item_steps, item_generation) in (
                    self._estimates.items()
                )
            ]
            heapq.heapify(self._heap)

    def _pop_minimum(self) -> tuple[int, int, _UsageGroupKey]:
        while self._heap:
            tokens, steps, _, _, _, _, generation, key = heapq.heappop(self._heap)
            if self._estimates.get(key) == (tokens, steps, generation):
                return tokens, steps, key
        raise RuntimeError("In-memory usage candidate heap lost its retained groups.")


def _usage_rollup_from_session_records(
    *,
    session_records: _SessionRecordsFactory,
    query: UsageRollupQuery,
    as_of: datetime,
    matching_session_count: int,
    active_session_count: int,
) -> UsageRollupStoreResult:
    totals = _UsageAccumulator()
    pricing = BoundedUsagePricingInputAccumulator(query.pricing_input_limit)
    provider_candidates = _InMemoryUsageGroupCandidates()
    model_candidates = _InMemoryUsageGroupCandidates()
    activity_session_count = 0
    tool_calls = 0

    for _, records in session_records():
        session_has_activity = False
        for record in records:
            event = record.event
            event_timestamp = normalize_aggregate_event_timestamp(event.timestamp)
            if event_timestamp < query.start_at or event_timestamp >= query.end_at:
                continue
            if event.type == EventType.TOOL_CALL_STARTED:
                session_has_activity = True
                tool_calls += 1
                continue
            if event.type != EventType.MODEL_COMPLETED:
                continue

            session_has_activity = True
            metrics = aggregate_usage_metrics_from_event_payload(event.payload)
            totals.add(metrics)
            provider_candidates.observe(
                _usage_group_key(metrics, dimension="provider"),
                metrics,
            )
            model_candidates.observe(
                _usage_group_key(metrics, dimension="model"),
                metrics,
            )

            if not query.include_pricing_inputs or pricing.truncated:
                continue
            pricing.add_payload(
                effective_on=event_timestamp.date(),
                occurrences=1,
                payload=event.payload,
            )
        if session_has_activity:
            activity_session_count += 1

    pricing_items, pricing_group_count, pricing_accuracy = pricing.result()

    totals.session_count = activity_session_count
    return UsageRollupStoreResult(
        as_of=as_of,
        start_at=query.start_at,
        end_at=query.end_at,
        totals=totals.totals(tool_calls=tool_calls),
        provider_breakdown=_bounded_in_memory_usage_breakdown(
            session_records,
            query=query,
            limit=query.group_limit,
            dimension="provider",
            candidates=provider_candidates,
        ),
        model_breakdown=_bounded_in_memory_usage_breakdown(
            session_records,
            query=query,
            limit=query.group_limit,
            dimension="model",
            candidates=model_candidates,
        ),
        pricing_inputs=pricing_items,
        pricing_inputs_included=query.include_pricing_inputs,
        pricing_input_group_count=pricing_group_count,
        pricing_inputs_accuracy=pricing_accuracy,
        active_session_count=active_session_count,
        matching_session_count=matching_session_count,
    )


def _bounded_in_memory_usage_breakdown(
    session_records: _SessionRecordsFactory,
    *,
    query: UsageRollupQuery,
    limit: int,
    dimension: Literal["provider", "model"],
    candidates: _InMemoryUsageGroupCandidates,
) -> UsageAggregateBreakdown:
    accumulators = _accumulate_usage_group_batch(
        session_records,
        query=query,
        dimension=dimension,
        keys=candidates.keys,
    )
    visible_items = sorted(accumulators.items(), key=_usage_group_rank_key)[:limit]

    visible = tuple(
        UsageAggregateGroup(
            provider_name=key[0],
            model=key[1],
            totals=accumulator.totals(),
        )
        for key, accumulator in visible_items
    )
    if candidates.sampled:
        return UsageAggregateBreakdown(
            groups=visible,
            remainder=None,
            accuracy=AggregateAccuracy(
                kind=AggregateAccuracyKind.SAMPLED,
                reason=(
                    f"Distinct {dimension} groups exceed the bounded in-memory "
                    "heavy-hitter candidate limit."
                ),
                limit=candidates.limit,
            ),
        )

    group_count = len(candidates.keys)
    if group_count <= limit:
        return UsageAggregateBreakdown(
            groups=visible,
            remainder=None,
            accuracy=EXACT_AGGREGATE.model_copy(),
        )

    remainder = _accumulate_usage_remainder(
        session_records,
        query=query,
        dimension=dimension,
        visible_keys={(group.provider_name, group.model) for group in visible},
    )
    return UsageAggregateBreakdown(
        groups=visible,
        remainder=UsageAggregateRemainder(
            group_count=group_count - len(visible),
            totals=remainder.totals(),
        ),
        accuracy=AggregateAccuracy(
            kind=AggregateAccuracyKind.TRUNCATED,
            reason=f"Distinct {dimension} groups exceed group_limit.",
            limit=limit,
        ),
    )


def _accumulate_usage_group_batch(
    session_records: _SessionRecordsFactory,
    *,
    query: UsageRollupQuery,
    dimension: Literal["provider", "model"],
    keys: tuple[_UsageGroupKey, ...],
) -> dict[_UsageGroupKey, _UsageAccumulator]:
    accumulators = {key: _UsageAccumulator() for key in keys}
    for _, records in session_records():
        seen: set[_UsageGroupKey] = set()
        for record in records:
            event = record.event
            if not _aggregate_model_event_is_in_window(event, query):
                continue
            metrics = aggregate_usage_metrics_from_event_payload(event.payload)
            key = _usage_group_key(metrics, dimension=dimension)
            accumulator = accumulators.get(key)
            if accumulator is None:
                continue
            accumulator.add(metrics)
            seen.add(key)
        for key in seen:
            accumulators[key].session_count += 1
    return accumulators


def _accumulate_usage_remainder(
    session_records: _SessionRecordsFactory,
    *,
    query: UsageRollupQuery,
    dimension: Literal["provider", "model"],
    visible_keys: set[_UsageGroupKey],
) -> _UsageAccumulator:
    remainder = _UsageAccumulator()
    for _, records in session_records():
        session_has_remainder = False
        for record in records:
            event = record.event
            if not _aggregate_model_event_is_in_window(event, query):
                continue
            metrics = aggregate_usage_metrics_from_event_payload(event.payload)
            if _usage_group_key(metrics, dimension=dimension) in visible_keys:
                continue
            remainder.add(metrics)
            session_has_remainder = True
        remainder.session_count += session_has_remainder
    return remainder


def _aggregate_model_event_is_in_window(event: Event, query: UsageRollupQuery) -> bool:
    if event.type != EventType.MODEL_COMPLETED:
        return False
    timestamp = normalize_aggregate_event_timestamp(event.timestamp)
    return query.start_at <= timestamp < query.end_at


def _usage_group_key(
    metrics: UsageMetrics | None,
    *,
    dimension: Literal["provider", "model"],
) -> _UsageGroupKey:
    provider_name = None if metrics is None else metrics.provider_name
    model = None if metrics is None or dimension == "provider" else metrics.model
    return provider_name, model


def _usage_group_identity_sort_key(key: _UsageGroupKey) -> _UsageGroupSortKey:
    return key[0] is None, key[0] or "", key[1] is None, key[1] or ""


def _usage_group_rank_key(
    item: tuple[_UsageGroupKey, _UsageAccumulator],
) -> tuple[int, int, bool, str, bool, str]:
    key, accumulator = item
    return (
        -accumulator.usage.total_tokens,
        -accumulator.model_steps,
        *_usage_group_identity_sort_key(key),
    )


def _session_matches_debug_state(
    session: Session,
    records: list[EventRecord],
    debug_state: SessionDebugState | None,
) -> bool:
    if debug_state is None:
        return True
    has_tool_debug_event = any(_is_tool_debug_event(record.event.type) for record in records)
    has_session_failure = session.status == SessionStatus.FAILED
    has_interruption = session.status == SessionStatus.INTERRUPTED
    if debug_state == SessionDebugState.TOOL_ISSUE:
        return has_tool_debug_event
    if debug_state == SessionDebugState.SESSION_FAILURE:
        return has_session_failure
    if debug_state == SessionDebugState.INTERRUPTION:
        return has_interruption
    if debug_state == SessionDebugState.NEEDS_ATTENTION:
        return has_session_failure or has_interruption or has_tool_debug_event
    return False


def _is_tool_debug_event(event_type: EventType | str) -> bool:
    return event_type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_BLOCKED}


def _session_query_text_matches(session: Session, query: str) -> bool:
    needle = query.casefold()
    haystacks = [
        session.id,
        session.agent_name,
        session.provider_name,
        session.model,
        session.environment_name,
        session.parent_session_id,
        session.causal_budget_id,
        *session.labels.keys(),
        *session.labels.values(),
    ]
    return any(needle in value.casefold() for value in haystacks if type(value) is str and value)


def _label_selector_matches(
    labels: dict[str, str],
    selector: LabelSelectorRequirement,
) -> bool:
    value = labels.get(selector.key)
    if selector.operator == LabelSelectorOperator.EXISTS:
        return value is not None
    if selector.operator == LabelSelectorOperator.NOT_EXISTS:
        return value is None
    if selector.operator == LabelSelectorOperator.IN:
        return value in selector.values
    if selector.operator == LabelSelectorOperator.NOT_IN:
        return value is None or value not in selector.values
    raise ValueError(f"Unsupported label selector operator: {selector.operator}")


def _sort_sessions(sessions: list[Session], order_by: SessionOrder) -> list[Session]:
    if order_by == SessionOrder.CREATED_AT_ASC:
        return sorted(sessions, key=lambda session: (session.created_at, session.id))
    if order_by == SessionOrder.CREATED_AT_DESC:
        return sorted(
            sorted(sessions, key=lambda session: session.id),
            key=lambda session: session.created_at,
            reverse=True,
        )
    if order_by == SessionOrder.UPDATED_AT_ASC:
        return sorted(sessions, key=lambda session: (session.updated_at, session.id))
    if order_by == SessionOrder.UPDATED_AT_DESC:
        return sorted(
            sorted(sessions, key=lambda session: session.id),
            key=lambda session: session.updated_at,
            reverse=True,
        )
    if order_by == SessionOrder.LAST_ACTIVITY_AT_ASC:
        return sorted(sessions, key=lambda session: (session.last_activity_at, session.id))
    return sorted(
        sorted(sessions, key=lambda session: session.id),
        key=lambda session: session.last_activity_at,
        reverse=True,
    )


# Sessions that must be interrupted before they can be deleted (in-flight work).
DELETE_BLOCKED_SESSION_STATUSES = frozenset({SessionStatus.RUNNING, SessionStatus.INTERRUPTING})
_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY = "incomplete_session_recovery_claim"


def _incomplete_recovery_claim_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> tuple[str, datetime] | None:
    """Parse the runtime-owned incomplete-session recovery lease."""
    if checkpoint is None or _INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY not in checkpoint:
        return None
    marker = checkpoint[_INCOMPLETE_RECOVERY_CLAIM_CHECKPOINT_KEY]
    if type(marker) is not dict or marker.get("version") != 1:
        raise ValueError("Incomplete-session recovery claim checkpoint is invalid.")
    claim_id = require_clean_nonblank(marker.get("claim_id"), "recovery_claim.claim_id")
    claimed_at_value = require_clean_nonblank(
        marker.get("claimed_at"),
        "recovery_claim.claimed_at",
    )
    expires_at_value = require_clean_nonblank(
        marker.get("claim_expires_at"),
        "recovery_claim.claim_expires_at",
    )
    try:
        claimed_at = datetime.fromisoformat(claimed_at_value)
        expires_at = datetime.fromisoformat(expires_at_value)
    except ValueError as exc:
        raise ValueError("Incomplete-session recovery claim timestamps are invalid.") from exc
    if (
        claimed_at.tzinfo is None
        or claimed_at.utcoffset() is None
        or expires_at.tzinfo is None
        or expires_at.utcoffset() is None
        or expires_at <= claimed_at
    ):
        raise ValueError("Incomplete-session recovery claim timestamps are invalid.")
    return claim_id, expires_at


def _active_unexpired_incomplete_recovery_claim_id(
    checkpoint: dict[str, Any] | None,
    *,
    now: datetime,
) -> str | None:
    """Return the durable recovery owner while its lease remains live."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
    if claim is None:
        return None
    claim_id, expires_at = claim
    return claim_id if expires_at.astimezone(UTC) > now.astimezone(UTC) else None


def _active_unexpired_session_operation_id(
    checkpoint: dict[str, Any] | None,
    *,
    now: datetime,
) -> str | None:
    """Return the active durable operation while its claim lease is valid."""

    if checkpoint is None:
        return None
    stored = checkpoint.get("session_operations")
    if stored is None:
        return None
    if type(stored) is not dict:
        raise ValueError("Session operation checkpoint must be an object.")
    active_operation_id = stored.get("active_operation_id")
    if active_operation_id is None:
        return None
    active_operation_id = require_clean_nonblank(
        active_operation_id,
        "active_operation_id",
    )
    records = stored.get("records")
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
        raise ValueError("Active durable session operation record is missing.")
    claim_expires_at = active_record.get("claim_expires_at")
    if type(claim_expires_at) is not str:
        return active_operation_id
    try:
        expiry = datetime.fromisoformat(claim_expires_at)
    except ValueError:
        return active_operation_id
    if expiry.tzinfo is None or expiry.utcoffset() is None:
        return active_operation_id
    return active_operation_id if expiry.astimezone(UTC) > now.astimezone(UTC) else None


_DESCENDING_SESSION_ORDERS = frozenset(
    {
        SessionOrder.CREATED_AT_DESC,
        SessionOrder.UPDATED_AT_DESC,
        SessionOrder.LAST_ACTIVITY_AT_DESC,
    }
)
_CREATED_AT_ORDERS = frozenset({SessionOrder.CREATED_AT_ASC, SessionOrder.CREATED_AT_DESC})
_LAST_ACTIVITY_AT_ORDERS = frozenset(
    {SessionOrder.LAST_ACTIVITY_AT_ASC, SessionOrder.LAST_ACTIVITY_AT_DESC}
)


def session_order_is_descending(order_by: SessionOrder) -> bool:
    return order_by in _DESCENDING_SESSION_ORDERS


def session_sort_column(order_by: SessionOrder) -> str:
    """The session column an order sorts by — the keyset cursor's primary key."""
    if order_by in _CREATED_AT_ORDERS:
        return "created_at"
    if order_by in _LAST_ACTIVITY_AT_ORDERS:
        return "last_activity_at"
    return "updated_at"


def _session_sort_value(
    session: Session | PendingActionSession,
    order_by: SessionOrder,
) -> datetime:
    if order_by in _CREATED_AT_ORDERS:
        return session.created_at
    if order_by in _LAST_ACTIVITY_AT_ORDERS:
        return session.last_activity_at if isinstance(session, Session) else session.updated_at
    return session.updated_at


def encode_session_cursor(
    session: Session | PendingActionSession,
    order_by: SessionOrder,
) -> str:
    """Opaque keyset cursor for the last row of a page: (sort value, session id)."""
    sort_value = _session_sort_value(session, order_by).astimezone(UTC).isoformat()
    raw = json.dumps([sort_value, session.id], separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def decode_session_cursor(cursor: str) -> tuple[datetime, str]:
    """Decode a cursor to (sort value, session id). Raises ValueError if malformed."""
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        decoded = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid session cursor.") from exc
    if (
        type(decoded) is not list
        or len(decoded) != 2
        or type(decoded[0]) is not str
        or type(decoded[1]) is not str
    ):
        raise ValueError("Invalid session cursor.")
    try:
        sort_value = datetime.fromisoformat(decoded[0])
    except ValueError as exc:
        raise ValueError("Invalid session cursor.") from exc
    # Sort values are always encoded as UTC-aware timestamps; a naive datetime is
    # a malformed/forged cursor that would raise TypeError when later compared
    # against the timezone-aware session timestamps.
    if sort_value.tzinfo is None:
        raise ValueError("Invalid session cursor.")
    return sort_value, decoded[1]


def session_next_cursor(page: list[Session], has_more: bool, order_by: SessionOrder) -> str | None:
    """The keyset cursor for the next page: the last row's cursor, or None if no more."""
    return encode_session_cursor(page[-1], order_by) if has_more and page else None


def _session_after_cursor(
    session: Session,
    order_by: SessionOrder,
    cursor_value: datetime,
    cursor_id: str,
) -> bool:
    """Whether ``session`` falls strictly after the cursor under ``order_by`` (id tiebreak ASC)."""
    value = _session_sort_value(session, order_by)
    if value != cursor_value:
        if session_order_is_descending(order_by):
            return value < cursor_value
        return value > cursor_value
    return session.id > cursor_id


def _event_record_matches(
    record: EventRecord,
    query: EventQuery,
    event_types: frozenset[str],
    excluded_event_types: frozenset[str],
) -> bool:
    event = record.event
    if query.after_sequence is not None and record.sequence <= query.after_sequence:
        return False
    if query.before_sequence is not None and record.sequence >= query.before_sequence:
        return False
    if query.session_id is not None and event.session_id != query.session_id:
        return False
    if query.session_ids and event.session_id not in query.session_ids:
        return False
    if query.event_id is not None and event.id != query.event_id:
        return False
    event_timestamp = event.timestamp.astimezone(UTC)
    if query.since is not None and event_timestamp < query.since:
        return False
    if query.until is not None and event_timestamp >= query.until:
        return False
    if event_types and str(event.type) not in event_types:
        return False
    if str(event.type) in excluded_event_types:
        return False
    if query.agent_name is not None and event.agent_name != query.agent_name:
        return False
    if query.environment_name is not None and event.environment_name != query.environment_name:
        return False
    if query.workflow_name is not None and event.workflow_name != query.workflow_name:
        return False
    return not (query.tool_name is not None and event.tool_name != query.tool_name)


def _event_record_matches_session(
    record: EventRecord,
    query: EventQuery,
    sessions: dict[str, Session],
) -> bool:
    if query.causal_budget_id is None:
        return True
    session = sessions.get(record.event.session_id)
    return session is not None and session.causal_budget_id == query.causal_budget_id
