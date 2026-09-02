from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
)
from cayu.core.events import EVENT_ID_MAX_CHARS, EventType

CHILD_SESSION_NOTIFICATION_INTENT_KEY = "child_session_notifications"
CHILD_SESSION_NOTIFICATION_STAGE_BINDING_VERSION = (
    "cayu.child-session-notification-stage-binding.v1"
)
CHILD_SESSION_NOTIFICATION_CONSUMPTION_RECORD_TYPE = "cayu.child-session-notification-consumption"
CHILD_SESSION_NOTIFICATION_CONSUMPTION_SCHEMA_VERSION = 1
CHILD_SESSION_NOTIFICATION_OPERATION_KEY_PREFIX = "__cayu_child_session_notification_v1__:"

CHILD_SESSION_NOTIFICATION_MAX_CHILDREN_INSPECTED = 128
CHILD_SESSION_NOTIFICATION_DEFAULT_CHILDREN_INSPECTED = 64
CHILD_SESSION_NOTIFICATION_MAX_ENTRIES = 32
CHILD_SESSION_NOTIFICATION_DEFAULT_ENTRIES = 16
CHILD_SESSION_NOTIFICATION_MAX_PROJECTION_BYTES = 32 * 1024
CHILD_SESSION_NOTIFICATION_DEFAULT_PROJECTION_BYTES = 16 * 1024
CHILD_SESSION_NOTIFICATION_MAX_RETRIEVAL_REFERENCES = 16
CHILD_SESSION_ADMISSION_OCCURRENCE_TYPE = "session.record.created"


class ChildSessionLifecycleState(StrEnum):
    ADMITTED = "admitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    @property
    def terminal(self) -> bool:
        return self in {
            ChildSessionLifecycleState.COMPLETED,
            ChildSessionLifecycleState.FAILED,
            ChildSessionLifecycleState.INTERRUPTED,
        }


class ChildSessionRelationship(StrEnum):
    DIRECT_CHILD = "direct_child"
    SESSION_FORK = "session_fork"


class ChildSessionNotificationFreshness(StrEnum):
    CURRENT = "current"
    FRESH = "fresh"
    CONSUMED = "consumed"


class ChildSessionLifecycleOccurrenceSource(StrEnum):
    SESSION = "session"
    EVENT = "event"


class ChildSessionLifecycleOccurrence(BaseModel):
    """Exact private durable occurrence underlying one lifecycle projection."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    source: ChildSessionLifecycleOccurrenceSource
    source_id: str = Field(max_length=EVENT_ID_MAX_CHARS)
    source_sequence: StrictInt | None = Field(
        default=None,
        ge=1,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    source_type: str
    occurred_at: datetime

    @field_validator("source_id", "source_type")
    @classmethod
    def validate_source_identity(cls, value: str, info: Any) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_source(self) -> ChildSessionLifecycleOccurrence:
        if self.source is ChildSessionLifecycleOccurrenceSource.SESSION:
            if (
                self.source_sequence is not None
                or self.source_type != CHILD_SESSION_ADMISSION_OCCURRENCE_TYPE
            ):
                raise ValueError("A session occurrence must identify exact admission.")
        else:
            if self.source_sequence is None:
                raise ValueError("An event occurrence requires its durable sequence.")
            try:
                EventType(self.source_type)
            except ValueError as exc:
                raise ValueError("An event occurrence requires a known lifecycle type.") from exc
        return self


class ChildSessionLifecycleEntry(BaseModel):
    """Bounded private canonical state for one authorized direct child."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    parent_session_id: str
    parent_session_instance_id: str
    child_session_id: str
    child_session_instance_id: str
    relationship: ChildSessionRelationship
    state: ChildSessionLifecycleState
    occurrence: ChildSessionLifecycleOccurrence
    freshness: ChildSessionNotificationFreshness
    consumed_by_stage_id: str | None = None
    created_at: datetime
    updated_at: datetime

    @field_validator(
        "parent_session_id",
        "parent_session_instance_id",
        "child_session_id",
        "child_session_instance_id",
        "consumed_by_stage_id",
    )
    @classmethod
    def validate_identity(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime, info: Any) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must include a timezone.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_freshness(self) -> ChildSessionLifecycleEntry:
        if self.state.terminal:
            expected_consumed = self.freshness is ChildSessionNotificationFreshness.CONSUMED
            if expected_consumed != (self.consumed_by_stage_id is not None):
                raise ValueError("Terminal notification freshness conflicts with consumption.")
            if self.freshness is ChildSessionNotificationFreshness.CURRENT:
                raise ValueError("Terminal lifecycle entries cannot have current freshness.")
        elif (
            self.freshness is not ChildSessionNotificationFreshness.CURRENT
            or self.consumed_by_stage_id is not None
        ):
            raise ValueError("Nonterminal lifecycle entries must have current freshness.")
        return self


class ChildSessionLifecycleQuery(BaseModel):
    """Bounded direct-child read used by model context composition."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    parent_session_id: str
    max_children_inspected: StrictInt = Field(
        default=CHILD_SESSION_NOTIFICATION_DEFAULT_CHILDREN_INSPECTED,
        ge=1,
        le=CHILD_SESSION_NOTIFICATION_MAX_CHILDREN_INSPECTED,
    )

    @field_validator("parent_session_id")
    @classmethod
    def validate_parent_session_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "parent_session_id")


class ChildSessionLifecyclePage(BaseModel):
    """One deterministic snapshot with explicit bounded-coverage evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    parent_session_id: str
    parent_session_instance_id: str
    entries: tuple[ChildSessionLifecycleEntry, ...] = ()
    inspected_child_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    unavailable_child_count: StrictInt = Field(default=0, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    has_more: StrictBool = False

    @field_validator("parent_session_id", "parent_session_instance_id")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("entries", mode="before")
    @classmethod
    def copy_entries(cls, value: Any) -> tuple[ChildSessionLifecycleEntry, ...]:
        if type(value) not in {list, tuple}:
            raise TypeError("entries must be a list or tuple.")
        return tuple(ChildSessionLifecycleEntry.model_validate(entry) for entry in value)

    @model_validator(mode="after")
    def validate_page(self) -> ChildSessionLifecyclePage:
        if self.inspected_child_count != len(self.entries) + self.unavailable_child_count:
            raise ValueError(
                "inspected_child_count must match retained and unavailable lifecycle entries."
            )
        identities = tuple(entry.child_session_id for entry in self.entries)
        if len(identities) != len(set(identities)):
            raise ValueError("A lifecycle page cannot repeat a child session.")
        if any(
            entry.parent_session_id != self.parent_session_id
            or entry.parent_session_instance_id != self.parent_session_instance_id
            for entry in self.entries
        ):
            raise ValueError("Lifecycle page entries conflict with their parent authority.")
        return self


class ChildSessionNotificationClaim(BaseModel):
    """Exact terminal occurrence selected for one provider dispatch."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    child_session_id: str
    child_session_instance_id: str
    occurrence: ChildSessionLifecycleOccurrence

    @field_validator("child_session_id", "child_session_instance_id")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class ChildSessionNotificationStageBinding(BaseModel):
    """Private immutable claims bound to one exact model-stage request."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.child-session-notification-stage-binding.v1"] = (
        CHILD_SESSION_NOTIFICATION_STAGE_BINDING_VERSION
    )
    parent_session_id: str
    parent_session_instance_id: str
    consumption_scope: Literal["parent_session_instance"] = "parent_session_instance"
    claims: tuple[ChildSessionNotificationClaim, ...]

    @field_validator("parent_session_id", "parent_session_instance_id")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("claims", mode="before")
    @classmethod
    def copy_claims(cls, value: Any) -> tuple[ChildSessionNotificationClaim, ...]:
        if type(value) not in {list, tuple}:
            raise TypeError("claims must be a list or tuple.")
        claims = tuple(ChildSessionNotificationClaim.model_validate(claim) for claim in value)
        if not claims:
            raise ValueError("claims must not be empty.")
        if len(claims) > CHILD_SESSION_NOTIFICATION_MAX_ENTRIES:
            raise ValueError("claims exceed the child-notification entry limit.")
        return claims

    @model_validator(mode="after")
    def validate_binding(self) -> ChildSessionNotificationStageBinding:
        identities = tuple(
            (claim.child_session_id, claim.occurrence.source_id) for claim in self.claims
        )
        if len(identities) != len(set(identities)):
            raise ValueError("A stage binding cannot repeat a terminal occurrence.")
        if any(
            claim.occurrence.source is not ChildSessionLifecycleOccurrenceSource.EVENT
            or EventType(claim.occurrence.source_type)
            not in {
                EventType.SESSION_COMPLETED,
                EventType.SESSION_FAILED,
                EventType.SESSION_INTERRUPTED,
            }
            for claim in self.claims
        ):
            raise ValueError("A stage binding can consume only terminal occurrences.")
        return self


class ChildSessionNotificationConsumption(BaseModel):
    """Minimal durable at-most-once authority; child truth remains canonical elsewhere."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.child-session-notification-consumption"] = (
        CHILD_SESSION_NOTIFICATION_CONSUMPTION_RECORD_TYPE
    )
    schema_version: Literal[1] = CHILD_SESSION_NOTIFICATION_CONSUMPTION_SCHEMA_VERSION
    parent_session_id: str
    parent_session_instance_id: str
    child_session_id: str
    child_session_instance_id: str
    occurrence: ChildSessionLifecycleOccurrence
    stage_id: str
    model_step_id: str
    model_attempt_id: str
    source_run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    consumed_at: datetime

    @field_validator(
        "parent_session_id",
        "parent_session_instance_id",
        "child_session_id",
        "child_session_instance_id",
        "stage_id",
        "model_step_id",
        "model_attempt_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("consumed_at")
    @classmethod
    def normalize_consumed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("consumed_at must include a timezone.")
        return value.astimezone(UTC)


def child_session_notification_occurrence_id(
    *,
    parent_session_instance_id: str,
    child_session_instance_id: str,
    occurrence: ChildSessionLifecycleOccurrence,
) -> str:
    """Return a stable public, non-authorizing commitment to one occurrence."""

    material = canonical_durable_json_bytes(
        {
            "schema_version": 1,
            "parent_session_instance_id": require_durable_clean_nonblank(
                parent_session_instance_id,
                "parent_session_instance_id",
            ),
            "child_session_instance_id": require_durable_clean_nonblank(
                child_session_instance_id,
                "child_session_instance_id",
            ),
            "source": str(occurrence.source),
            "source_id": occurrence.source_id,
            "source_sequence": occurrence.source_sequence,
            "source_type": occurrence.source_type,
        },
        "child_session_notification_occurrence",
    )
    return "cayu_child_occurrence_v1_" + sha256(material).hexdigest()


def child_session_notification_storage_key(
    child_session_instance_id: str,
    occurrence_source_id: str,
) -> str:
    child_session_instance_id = require_durable_clean_nonblank(
        child_session_instance_id,
        "child_session_instance_id",
    )
    occurrence_source_id = require_durable_clean_nonblank(
        occurrence_source_id,
        "occurrence_source_id",
    )
    # The length-prefixed child incarnation and exact event id let durable stores
    # join canonical state to consumption without a second identity. Length
    # prefixing stays unambiguous for every durable Unicode scalar string and is
    # reproducible with SQLite ``length`` and PostgreSQL ``char_length``.
    return (
        CHILD_SESSION_NOTIFICATION_OPERATION_KEY_PREFIX
        + str(len(child_session_instance_id))
        + ":"
        + child_session_instance_id
        + occurrence_source_id
    )


def child_session_notification_stage_binding(
    intent: dict[str, Any],
) -> ChildSessionNotificationStageBinding | None:
    raw = intent.get(CHILD_SESSION_NOTIFICATION_INTENT_KEY)
    if raw is None:
        return None
    try:
        return ChildSessionNotificationStageBinding.model_validate(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Model-stage child-session notification binding is malformed.") from exc


__all__ = [
    "CHILD_SESSION_NOTIFICATION_DEFAULT_CHILDREN_INSPECTED",
    "CHILD_SESSION_NOTIFICATION_DEFAULT_ENTRIES",
    "CHILD_SESSION_NOTIFICATION_DEFAULT_PROJECTION_BYTES",
    "CHILD_SESSION_NOTIFICATION_INTENT_KEY",
    "CHILD_SESSION_NOTIFICATION_MAX_CHILDREN_INSPECTED",
    "CHILD_SESSION_NOTIFICATION_MAX_ENTRIES",
    "CHILD_SESSION_NOTIFICATION_MAX_PROJECTION_BYTES",
    "CHILD_SESSION_NOTIFICATION_MAX_RETRIEVAL_REFERENCES",
    "ChildSessionLifecycleEntry",
    "ChildSessionLifecycleOccurrence",
    "ChildSessionLifecyclePage",
    "ChildSessionLifecycleQuery",
    "ChildSessionLifecycleState",
    "ChildSessionNotificationClaim",
    "ChildSessionNotificationConsumption",
    "ChildSessionNotificationFreshness",
    "ChildSessionNotificationStageBinding",
    "ChildSessionRelationship",
    "child_session_notification_occurrence_id",
    "child_session_notification_stage_binding",
    "child_session_notification_storage_key",
]
