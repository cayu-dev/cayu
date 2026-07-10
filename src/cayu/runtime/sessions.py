from __future__ import annotations

import asyncio
import base64
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
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

from cayu._validation import copy_json_value, copy_label_map, require_clean_nonblank
from cayu.core.events import Event, EventType, copy_event
from cayu.core.messages import Message, MessageRole, ThinkingPart, copy_message, detach_message
from cayu.core.thinking import ThinkingConfig
from cayu.runtime.budgets import BudgetLimit, copy_request_budget_limits
from cayu.runtime.loop_policies import LoopPolicy, validate_loop_policies
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy
from cayu.runtime.stop_policy import RunLimits, copy_run_limits
from cayu.runtime.structured_output import StructuredOutputSpec, copy_structured_output_spec


class SessionStatusConflict(ValueError):
    """A session status transition was rejected because the session was not in an
    allowed source status (e.g. resuming a session another worker is already
    running). Subclasses ``ValueError`` so existing ``except ValueError`` handlers
    keep working; callers that need to react specifically (e.g. requeue) catch this.
    """


class SessionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTING = "interrupting"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


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


class InterruptSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

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


CheckpointTransform = Callable[
    [Session, dict[str, Any] | None],
    dict[str, Any] | None,
]


class SessionOrder(StrEnum):
    CREATED_AT_ASC = "created_at_asc"
    CREATED_AT_DESC = "created_at_desc"
    UPDATED_AT_ASC = "updated_at_asc"
    UPDATED_AT_DESC = "updated_at_desc"


class EventOrder(StrEnum):
    SEQUENCE_ASC = "sequence_asc"
    SEQUENCE_DESC = "sequence_desc"


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

    @field_validator("labels", mode="before")
    @classmethod
    def copy_query_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")

    @field_validator("label_selectors", mode="before")
    @classmethod
    def copy_query_label_selectors(cls, value) -> tuple[LabelSelectorRequirement, ...]:
        return copy_label_selector_requirements(value)


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: StrictInt = Field(ge=1)
    event: Event

    @field_validator("event")
    @classmethod
    def copy_event(cls, value: Event) -> Event:
        return copy_event(value)


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
    reason: str = "worker_recovered_incomplete_session"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("session_id", "reason")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class IncompleteSessionsRecoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statuses: set[SessionStatus]
    limit: StrictInt = Field(default=100, ge=1, le=1000)
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
    causal_budget_id: str | None = None
    event_type: EventType | str | None = None
    event_types: tuple[EventType | str, ...] = Field(default_factory=tuple)
    agent_name: str | None = None
    environment_name: str | None = None
    workflow_name: str | None = None
    tool_name: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    after_sequence: StrictInt | None = Field(default=None, ge=0)
    limit: StrictInt = Field(default=100, ge=1, le=5000)
    order_by: EventOrder = EventOrder.SEQUENCE_ASC

    @field_validator(
        "session_id",
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

    @field_validator("event_types", mode="before")
    @classmethod
    def copy_event_types(cls, value) -> tuple[EventType | str, ...]:
        if value is None:
            return ()
        if type(value) is str:
            raise ValueError("`event_types` must be a sequence of event types.")
        normalized: list[EventType | str] = []
        for item in tuple(value):
            if not isinstance(item, EventType):
                item = Event(type=item, session_id="query").type
            if item in normalized:
                raise ValueError("`event_types` must not contain duplicates.")
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
        if self.event_type is not None and self.event_types:
            raise ValueError("Use either `event_type` or `event_types`, not both.")
        if self.since is not None and self.until is not None and self.since >= self.until:
            raise ValueError("EventQuery since must be before until.")
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
    """Persistent store for sessions and append-only events."""

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
    ) -> Session:
        """Create a forked session with copied transcript/checkpoint state."""

    @abstractmethod
    async def load(self, session_id: str) -> Session | None:
        """Load a session by id."""

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
        """Atomically transition a session status when its current status is allowed."""

    @abstractmethod
    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        *,
        from_statuses: set[SessionStatus],
        to_status: SessionStatus,
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        """Atomically transition status and persist transformed checkpoint state."""

    async def append_event(self, session_id: str, event: Event) -> None:
        """Append one event to a session."""
        await self.append_events(session_id, [event])

    @abstractmethod
    async def append_events(self, session_id: str, events: list[Event]) -> None:
        """Append events to a session in one durable batch."""

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

    @abstractmethod
    async def list_sessions(self, query: SessionQuery | None = None) -> SessionListResult:
        """List sessions (filtered/sorted/paginated) with a keyset cursor and total count."""

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

        Raises ``KeyError`` if the session does not exist. Default raises
        ``NotImplementedError`` so out-of-tree stores keep working.
        """
        raise NotImplementedError("This SessionStore does not support update_labels.")

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        """Replace a session's metadata (full replacement, not a merge) and return it.

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
    async def append_transcript_messages_and_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint: dict[str, Any],
    ) -> None:
        """Append transcript messages and persist a checkpoint atomically.

        Same isolation contract as `append_transcript_messages`.
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
    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        """Load the latest checkpoint for a session."""


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
        self._type_event_records: dict[str, list[EventRecord]] = {}
        self._event_ids: dict[str, set[str]] = {}
        self._next_event_sequence = 1
        self._transcripts: dict[str, list[Message]] = {}
        self._checkpoints: dict[str, dict[str, Any]] = {}

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
                labels=request.labels,
                metadata=deepcopy(request.metadata),
            )
            self._sessions[session.id] = session
            self._events[session.id] = []
            self._event_ids[session.id] = set()
            self._session_event_records[session.id] = []
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
            self._transcripts[fork.id] = copied_transcript
            if copied_checkpoint is not None:
                self._checkpoints[fork.id] = copied_checkpoint
            return fork.model_copy(deep=True)

    async def load(self, session_id: str) -> Session | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.model_copy(deep=True)

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

            updated = session.model_copy(
                update={
                    "model": model,
                    "updated_at": datetime.now(UTC),
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
            self._sessions.pop(session_id, None)
            self._events.pop(session_id, None)
            self._event_ids.pop(session_id, None)
            self._session_event_records.pop(session_id, None)
            self._transcripts.pop(session_id, None)
            self._checkpoints.pop(session_id, None)
            self._event_records = [
                record for record in self._event_records if record.event.session_id != session_id
            ]
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
            updated = session.model_copy(
                update={"labels": new_labels, "updated_at": datetime.now(UTC)}
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

    async def update_metadata(self, session_id: str, metadata: dict[str, Any]) -> Session:
        session_id = require_clean_nonblank(session_id, "session_id")
        new_metadata = copy_json_value(metadata, "metadata")
        if type(new_metadata) is not dict:
            raise TypeError("Session metadata must be an object.")
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session not found: {session_id}")
            updated = session.model_copy(
                update={"metadata": new_metadata, "updated_at": datetime.now(UTC)}
            )
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
            if session.status not in allowed_statuses:
                raise SessionStatusConflict(
                    f"Session status transition not allowed: {session.status} -> {to_status}"
                )

            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._sessions[session_id] = updated
            return updated.model_copy(deep=True)

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

            updated = session.model_copy(
                update={
                    "status": to_status,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._sessions[session_id] = updated
            if transformed_checkpoint is not None:
                self._checkpoints[session_id] = transformed_checkpoint
            return updated.model_copy(deep=True)

    async def append_event(self, session_id: str, event: Event) -> None:
        await self.append_events(session_id, [event])

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        session_id, copied_events = _copy_session_event_batch(session_id, events)

        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            existing_ids = self._event_ids[session_id]
            for event in copied_events:
                if event.id in existing_ids:
                    raise ValueError(f"Event already exists for session {session_id}: {event.id}")

            session_records = self._session_event_records.setdefault(session_id, [])
            for event in copied_events:
                stored_event = event.model_copy(deep=True)
                self._events[session_id].append(stored_event)
                record = EventRecord(
                    sequence=self._next_event_sequence,
                    event=stored_event,
                )
                self._event_records.append(record)
                # Share the same EventRecord across the secondary indexes; records are
                # immutable in practice and query paths copy before returning.
                session_records.append(record)
                self._type_event_records.setdefault(str(stored_event.type), []).append(record)
                existing_ids.add(stored_event.id)
                self._next_event_sequence += 1

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
        async with self._lock:
            candidates = self._query_candidate_records(query, event_types)
            records = [
                record
                for record in candidates
                if _event_record_matches(record, query, event_types)
                and _event_record_matches_session(record, query, self._sessions)
            ]
            if query.order_by == EventOrder.SEQUENCE_DESC:
                records = list(reversed(records))
            return [
                EventRecord(
                    sequence=record.sequence,
                    event=record.event,
                )
                for record in records[: query.limit]
            ]

    def _query_candidate_records(
        self,
        query: EventQuery,
        event_types: frozenset[str],
    ) -> list[EventRecord]:
        """Pick the narrowest index that still covers a query's candidate rows.

        All returned lists stay sequence-ascending so downstream ordering and
        ``after_sequence`` paging behave exactly as a full scan would.
        """
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
        query = copy_session_query(query)
        base_query = query.model_copy(update={"debug_state": None})
        async with self._lock:
            matching = [
                session
                for session in self._sessions.values()
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
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            if not copied_messages:
                return
            self._transcripts[session_id].extend(copied_messages)

    async def append_transcript_messages_and_checkpoint(
        self,
        session_id: str,
        messages: list[Message],
        checkpoint: dict[str, Any],
    ) -> None:
        session_id = require_clean_nonblank(session_id, "session_id")
        copied_messages = _detach_transcript_messages(messages)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint state must be a dictionary.")
        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            if copied_messages:
                self._transcripts[session_id].extend(copied_messages)
            self._checkpoints[session_id] = copied_checkpoint

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
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            self._checkpoints[session_id] = copy_json_value(state, "checkpoint")

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        session_id = require_clean_nonblank(session_id, "session_id")
        async with self._lock:
            checkpoint = self._checkpoints.get(session_id)
            if checkpoint is None:
                return None
            return deepcopy(checkpoint)


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


def copy_interrupt_session_request(request: InterruptSessionRequest) -> InterruptSessionRequest:
    if type(request) is not InterruptSessionRequest:
        raise TypeError("Session interruption requires an InterruptSessionRequest.")
    return InterruptSessionRequest(
        session_id=request.session_id,
        reason=request.reason,
        metadata=copy_json_value(request.metadata, "metadata"),
    )


def copy_incomplete_session_recovery_request(
    request: IncompleteSessionRecoveryRequest,
) -> IncompleteSessionRecoveryRequest:
    if type(request) is not IncompleteSessionRecoveryRequest:
        raise TypeError("Incomplete session recovery requires an IncompleteSessionRecoveryRequest.")
    return IncompleteSessionRecoveryRequest(
        session_id=request.session_id,
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
) -> Session:
    if source_session is None:
        raise KeyError(f"Session not found: {source_session_id}")
    if source_session.status not in allowed_statuses:
        raise ValueError(f"Source session status is not forkable: {source_session.status}")
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
        labels=copy_label_map(query.labels, "labels"),
        label_selectors=copy_label_selector_requirements(query.label_selectors),
        limit=query.limit,
        offset=query.offset,
        cursor=query.cursor,
        include_total_count=query.include_total_count,
        order_by=query.order_by,
    )


def copy_event_query(query: EventQuery | None) -> EventQuery:
    if query is None:
        return EventQuery()
    if type(query) is not EventQuery:
        raise TypeError("Event queries must be EventQuery instances.")
    return EventQuery(
        session_id=query.session_id,
        session_ids=query.session_ids,
        causal_budget_id=query.causal_budget_id,
        event_type=query.event_type,
        event_types=query.event_types,
        agent_name=query.agent_name,
        environment_name=query.environment_name,
        workflow_name=query.workflow_name,
        tool_name=query.tool_name,
        since=query.since,
        until=query.until,
        after_sequence=query.after_sequence,
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
    for key, value in query.labels.items():
        if session.labels.get(key) != value:
            return False
    for selector in query.label_selectors:
        if not _label_selector_matches(session.labels, selector):
            return False
    return not (
        query.environment_name is not None and session.environment_name != query.environment_name
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
    return sorted(
        sorted(sessions, key=lambda session: session.id),
        key=lambda session: session.updated_at,
        reverse=True,
    )


# Sessions that must be interrupted before they can be deleted (in-flight work).
DELETE_BLOCKED_SESSION_STATUSES = frozenset({SessionStatus.RUNNING, SessionStatus.INTERRUPTING})

_DESCENDING_SESSION_ORDERS = frozenset({SessionOrder.CREATED_AT_DESC, SessionOrder.UPDATED_AT_DESC})
_CREATED_AT_ORDERS = frozenset({SessionOrder.CREATED_AT_ASC, SessionOrder.CREATED_AT_DESC})


def session_order_is_descending(order_by: SessionOrder) -> bool:
    return order_by in _DESCENDING_SESSION_ORDERS


def session_sort_column(order_by: SessionOrder) -> str:
    """The session column an order sorts by — the keyset cursor's primary key."""
    return "created_at" if order_by in _CREATED_AT_ORDERS else "updated_at"


def _session_sort_value(session: Session, order_by: SessionOrder) -> datetime:
    return session.created_at if order_by in _CREATED_AT_ORDERS else session.updated_at


def encode_session_cursor(session: Session, order_by: SessionOrder) -> str:
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
) -> bool:
    event = record.event
    if query.after_sequence is not None and record.sequence <= query.after_sequence:
        return False
    if query.session_id is not None and event.session_id != query.session_id:
        return False
    if query.session_ids and event.session_id not in query.session_ids:
        return False
    event_timestamp = event.timestamp.astimezone(UTC)
    if query.since is not None and event_timestamp < query.since:
        return False
    if query.until is not None and event_timestamp >= query.until:
        return False
    if event_types and str(event.type) not in event_types:
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
