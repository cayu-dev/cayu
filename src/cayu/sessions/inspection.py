"""Bounded session inspection contracts and shared record projection rules."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    compact_json_utf8_size,
    copy_durable_json_value,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.budgets.aggregates import AggregateUsageMetrics, build_aggregate_usage_metrics
from cayu.budgets.base import SessionBudgetInspection
from cayu.budgets.usage import UsageMetrics
from cayu.build_provenance import RuntimeBuildProvenance, legacy_runtime_build_provenance
from cayu.events import Event, EventType
from cayu.sessions.records import PendingActionKind, SessionStatus
from cayu.sessions.usage import _UsageAccumulator


class SerializedRecordSummary(BaseModel):
    """Exact serialized-size totals for one kind of durable session record."""

    model_config = ConfigDict(extra="forbid")

    record_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    total_bytes: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    largest_record_bytes: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)


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
    runtime_build_provenance: RuntimeBuildProvenance = Field(
        default_factory=legacy_runtime_build_provenance
    )
    environment_name: str | None
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    last_activity_at: datetime
    run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    labels: dict[str, str] = Field(default_factory=dict)
    label_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    labels_truncated: StrictBool = False


class SessionInspectionUsageSummary(BaseModel):
    """Lossless usage totals for bounded operator inspection."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    model_steps: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    tool_calls: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    provider_names: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    usage: AggregateUsageMetrics = Field(default_factory=build_aggregate_usage_metrics)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("provider_names", "models", mode="before")
    @classmethod
    def copy_string_lists(cls, value: list[str], info) -> list[str]:
        copied = copy_durable_json_value(value, info.field_name)
        if type(copied) is not list:
            raise ValueError(f"{info.field_name} must be a list.")
        result: list[str] = []
        for index, item in enumerate(copied):
            if type(item) is not str:
                raise ValueError(f"{info.field_name}[{index}] must be a string.")
            result.append(require_clean_nonblank(item, f"{info.field_name}[{index}]"))
        return result


class SessionInspectionSummary(BaseModel):
    """Bounded backend-neutral diagnostic overview for one durable session."""

    model_config = ConfigDict(extra="forbid")

    session: SessionInspectionIdentity
    transcript: SerializedRecordSummary
    events: SerializedRecordSummary
    usage: SessionInspectionUsageSummary
    model_calls: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    model_calls_with_usage: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    tool_calls: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    pending_action_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    pending_action_kinds: tuple[PendingActionKind, ...] = ()
    pending_action_issue_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    queued_message_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    delivered_message_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    outstanding_message_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    operation_event_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
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


@dataclass
class _SessionInspectionUsageAccumulator:
    """Single-pass inspection fold with native aggregate usage semantics."""

    totals: _UsageAccumulator = dataclass_field(default_factory=_UsageAccumulator)
    provider_names: list[str] = dataclass_field(default_factory=list)
    models: list[str] = dataclass_field(default_factory=list)
    _provider_names_seen: set[str] = dataclass_field(default_factory=set)
    _models_seen: set[str] = dataclass_field(default_factory=set)
    tool_calls: int = 0

    def add(self, event_type: str, metrics: UsageMetrics | None) -> None:
        if event_type == EventType.TOOL_CALL_STARTED:
            self.tool_calls += 1
            return
        if event_type in {
            EventType.MODEL_HOSTED_TOOL_CALL,
            EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
        }:
            if metrics is None:
                return
            self.totals.add_usage_only(metrics)
            self._record_identity(metrics)
            return
        if event_type != EventType.MODEL_COMPLETED:
            return
        self.totals.add(metrics)
        if metrics is None:
            return
        self._record_identity(metrics)

    def _record_identity(self, metrics: UsageMetrics) -> None:
        if (
            metrics.provider_name is not None
            and metrics.provider_name not in self._provider_names_seen
        ):
            self._provider_names_seen.add(metrics.provider_name)
            self.provider_names.append(metrics.provider_name)
        if metrics.model is not None and metrics.model not in self._models_seen:
            self._models_seen.add(metrics.model)
            self.models.append(metrics.model)

    def result(self, session_id: str) -> tuple[SessionInspectionUsageSummary, int]:
        return (
            SessionInspectionUsageSummary(
                session_id=session_id,
                model_steps=self.totals.model_steps,
                tool_calls=self.tool_calls,
                provider_names=self.provider_names,
                models=self.models,
                usage=self.totals.usage,
            ),
            self.totals.model_steps_with_usage,
        )
