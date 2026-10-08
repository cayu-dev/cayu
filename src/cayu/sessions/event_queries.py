"""Event query contracts, copying and record selection rules."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, EventType
from cayu.sessions.records import EventRecord


class EventOrder(StrEnum):
    SEQUENCE_ASC = "sequence_asc"
    SEQUENCE_DESC = "sequence_desc"


class EventQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str | None = None
    session_ids: tuple[str, ...] = Field(default_factory=tuple)
    event_id: str | None = None
    interaction_id: str | None = None
    causal_budget_id: str | None = None
    budget_limit_id: str | None = None
    model_step_id: str | None = None
    parent_model_step_id: str | None = None
    operation_attempt_id: str | None = None
    event_type: EventType | str | None = None
    event_types: tuple[EventType | str, ...] = Field(default_factory=tuple)
    exclude_event_types: tuple[EventType | str, ...] = Field(default_factory=tuple)
    agent_name: str | None = None
    environment_name: str | None = None
    workflow_name: str | None = None
    workflow_attempt_id: str | None = None
    workflow_step_id: str | None = None
    workflow_attempt_fenced: StrictBool = False
    tool_name: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    after_sequence: StrictInt | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    before_sequence: StrictInt | None = Field(default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    limit: StrictInt = Field(default=100, ge=1, le=5000)
    order_by: EventOrder = EventOrder.SEQUENCE_ASC

    @field_validator(
        "session_id",
        "event_id",
        "interaction_id",
        "causal_budget_id",
        "budget_limit_id",
        "model_step_id",
        "parent_model_step_id",
        "operation_attempt_id",
        "agent_name",
        "environment_name",
        "workflow_name",
        "workflow_attempt_id",
        "workflow_step_id",
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
        if self.workflow_attempt_id is not None and self.workflow_step_id is None:
            raise ValueError("EventQuery workflow_attempt_id requires workflow_step_id.")
        if self.workflow_attempt_fenced and self.workflow_step_id is None:
            raise ValueError("EventQuery workflow_attempt_fenced requires workflow_step_id.")
        if self.workflow_attempt_fenced and self.workflow_attempt_id is not None:
            raise ValueError("Use workflow_attempt_id or workflow_attempt_fenced, not both.")
        if (
            self.workflow_step_id is not None
            and self.workflow_attempt_id is None
            and not self.workflow_attempt_fenced
        ):
            raise ValueError(
                "EventQuery workflow_step_id requires an exact or fenced attempt scope."
            )
        if self.workflow_step_id is not None and (
            self.session_id is None
            or self.workflow_name is None
            or (self.event_type is None and not self.event_types)
        ):
            raise ValueError(
                "Workflow-step event queries require session_id, workflow_name, "
                "and an event-type filter."
            )
        if self.since is not None and self.until is not None and self.since >= self.until:
            raise ValueError("EventQuery since must be before until.")
        if (
            self.after_sequence is not None
            and self.before_sequence is not None
            and self.after_sequence >= self.before_sequence
        ):
            raise ValueError("EventQuery after_sequence must be before before_sequence.")
        return self


class EventQueryResultTooLarge(ValueError):
    """A bounded event query would hydrate more serialized bytes than allowed."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(f"Event query exceeds the {max_bytes}-byte safety limit.")


def copy_event_query(
    query: EventQuery | None,
    *,
    update: Mapping[str, object] | None = None,
) -> EventQuery:
    if query is not None and type(query) is not EventQuery:
        raise TypeError("Event queries must be EventQuery instances.")
    source = EventQuery() if query is None else query
    values = source.model_dump(mode="python")
    if update is not None:
        values.update(update)
    return EventQuery.model_validate(values)


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
    if query.interaction_id is not None and event.interaction_id != query.interaction_id:
        return False
    if (
        query.budget_limit_id is not None
        and event.payload.get("budget_limit_id") != query.budget_limit_id
    ):
        return False
    if (
        query.model_step_id is not None
        and event.payload.get("model_step_id") != query.model_step_id
    ):
        return False
    if (
        query.operation_attempt_id is not None
        and event.payload.get("attempt_id") != query.operation_attempt_id
    ):
        return False
    if (
        query.parent_model_step_id is not None
        and event.payload.get("parent_model_step_id") != query.parent_model_step_id
    ):
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
    if (
        query.workflow_attempt_id is not None
        and event.payload.get("attempt_id") != query.workflow_attempt_id
    ):
        return False
    if (
        query.workflow_step_id is not None
        and event.payload.get("step_id") != query.workflow_step_id
    ):
        return False
    return not (query.tool_name is not None and event.tool_name != query.tool_name)
