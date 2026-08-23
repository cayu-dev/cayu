from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import require_clean_nonblank
from cayu.core.events import Event, EventType
from cayu.runtime.tool_grants import TARGETED_TOOL_GRANT_MAX_REQUESTS
from cayu.runtime.usage import (
    USAGE_BEARING_EVENT_TYPES,
    AggregateUsageMetrics,
    SessionUsageSummary,
    aggregate_usage_metrics_from_durable_payload,
    build_aggregate_usage_metrics,
    session_usage_summary,
)


class InteractionStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


INTERACTION_LIFECYCLE_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.INTERACTION_STARTED,
    EventType.INTERACTION_RESUMED,
    EventType.INTERACTION_PAUSED,
    EventType.INTERACTION_COMPLETED,
    EventType.INTERACTION_FAILED,
    EventType.INTERACTION_INTERRUPTED,
)
INTERACTION_TERMINAL_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.INTERACTION_COMPLETED,
        EventType.INTERACTION_FAILED,
        EventType.INTERACTION_INTERRUPTED,
    }
)
INTERACTION_OPEN_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.INTERACTION_STARTED,
        EventType.INTERACTION_RESUMED,
        EventType.INTERACTION_PAUSED,
    }
)
INTERACTION_SUMMARY_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.MODEL_STARTED,
    *USAGE_BEARING_EVENT_TYPES,
)


def interaction_usage_summary(
    session_id: str,
    events: list[Event],
) -> SessionUsageSummary:
    """Aggregate response-scoped attempts, tool calls, and successful usage."""

    usage_summary = session_usage_summary(session_id, events)
    provider_names: list[str] = []
    models: list[str] = []
    model_steps = 0
    for event in events:
        if event.type != EventType.MODEL_STARTED:
            continue
        model_steps += 1
        for target, value in (
            (provider_names, event.payload.get("provider")),
            (models, event.payload.get("model")),
        ):
            if type(value) is str and value.strip() and value not in target:
                target.append(value)
    for target, values in (
        (provider_names, usage_summary.provider_names),
        (models, usage_summary.models),
    ):
        for value in values:
            if value not in target:
                target.append(value)
    return usage_summary.model_copy(
        update={
            "model_steps": model_steps,
            "provider_names": provider_names,
            "models": models,
        }
    )


class InteractionSummaryEvidence(BaseModel):
    """Validated, durable projection carried by interaction lifecycle events."""

    model_config = ConfigDict(extra="forbid")

    status: InteractionStatus
    start_event_id: str
    start_event_sequence: StrictInt | None = Field(default=None, ge=1)
    source_transcript_start: StrictInt | None = Field(default=None, ge=0)
    source_transcript_end: StrictInt | None = Field(default=None, ge=0)
    result_transcript_start: StrictInt | None = Field(default=None, ge=0)
    result_transcript_end: StrictInt | None = Field(default=None, ge=0)
    started_at: datetime
    completed_at: datetime | None = None
    active_duration_ms: StrictInt = Field(default=0, ge=0)
    wall_duration_ms: StrictInt | None = Field(default=None, ge=0)
    model_step_count: StrictInt = Field(default=0, ge=0)
    tool_call_count: StrictInt = Field(default=0, ge=0)
    token_usage: AggregateUsageMetrics = Field(default_factory=build_aggregate_usage_metrics)
    provider_names: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    pending_action_kind: str | None = None
    targeted_tool_grant_count: StrictInt | None = Field(
        default=None,
        ge=1,
        le=TARGETED_TOOL_GRANT_MAX_REQUESTS,
        exclude_if=lambda value: value is None,
    )
    targeted_tool_grant_batch_fingerprint: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )

    @field_validator("token_usage", mode="before")
    @classmethod
    def validate_token_usage(cls, value: object) -> AggregateUsageMetrics:
        if type(value) is AggregateUsageMetrics:
            return value
        return aggregate_usage_metrics_from_durable_payload(value)

    @field_validator("start_event_id", "pending_action_kind")
    @classmethod
    def validate_optional_nonblank(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("provider_names", "models")
    @classmethod
    def validate_names(cls, value: list[str], info) -> list[str]:
        names = [
            require_clean_nonblank(item, f"{info.field_name}[{index}]")
            for index, item in enumerate(value)
        ]
        if len(names) != len(set(names)):
            raise ValueError(f"{info.field_name} must not contain duplicates.")
        return names

    @field_validator("started_at", "completed_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_ranges_and_terminal_time(self) -> InteractionSummaryEvidence:
        if (self.targeted_tool_grant_count is None) != (
            self.targeted_tool_grant_batch_fingerprint is None
        ):
            raise ValueError("Targeted grant count and batch fingerprint must be present together.")
        for label, start, end in (
            ("source transcript", self.source_transcript_start, self.source_transcript_end),
            ("result transcript", self.result_transcript_start, self.result_transcript_end),
        ):
            if (start is None) != (end is None):
                raise ValueError(f"{label} range endpoints must both be present or absent.")
            if start is not None and end is not None and start > end:
                raise ValueError(f"{label} range start must not exceed its end.")
        terminal = self.status in {
            InteractionStatus.COMPLETED,
            InteractionStatus.FAILED,
            InteractionStatus.INTERRUPTED,
        }
        if terminal != (self.completed_at is not None):
            raise ValueError("completed_at must be present exactly for terminal interactions.")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at.")
        return self
