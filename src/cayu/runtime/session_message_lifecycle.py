"""Bounded authority for managing the existing durable session-message queue.

These values do not authorize execution. The application authorizes access;
SessionStore owns admission, freshness evaluation, and terminal compare-and-set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
)
from cayu.runtime.approvals import ResolutionActor, copy_resolution_actor


class _MessageLifecycleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class SessionMessageQueueStatus(StrEnum):
    QUEUED = "queued"
    DELIVERED = "delivered"
    WITHDRAWN = "withdrawn"
    QUARANTINED = "quarantined"
    STALE = "stale"
    EXPIRED = "expired"


class SessionMessageConflict(ValueError):
    def __init__(self) -> None:
        super().__init__("Session-message authority or terminal outcome conflicts.")


class SessionMessageAccessDenied(PermissionError):
    def __init__(self) -> None:
        super().__init__("Session-message access is not authorized.")


class SessionMessageSource(_MessageLifecycleModel):
    """Exact source observation, checked by the store before first acceptance.

    Source observations are historical after admission: identical retries do not
    reinterpret them against a newer source. A copied observation is not access
    authority; the application must independently authorize the source session.
    """

    session_id: StrictStr = Field(min_length=1, max_length=512)
    session_instance_id: StrictStr = Field(min_length=1, max_length=512)
    run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    transcript_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("session_id", "session_instance_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class SessionMessageTarget(_MessageLifecycleModel):
    """Expected target immediately before this message's transcript append.

    The instance prevents deletion/recreation from satisfying an old request.
    Both epoch and permanent transcript cursor must match exactly. Earlier
    delivered messages advance that cursor; terminal rejection does not. This
    definition is independent of the store's delivery batch size.
    """

    session_instance_id: StrictStr = Field(min_length=1, max_length=512)
    run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("session_instance_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "session_instance_id")


class SessionMessageConditions(_MessageLifecycleModel):
    """Optional provenance and delivery constraints; never ordinary metadata."""

    source: SessionMessageSource | None = None
    target: SessionMessageTarget | None = None
    expires_at: datetime | None = None

    @field_validator("source", mode="before")
    @classmethod
    def copy_source(cls, value):
        if isinstance(value, SessionMessageSource):
            if type(value) is not SessionMessageSource:
                raise TypeError("source must be a SessionMessageSource.")
            return {
                "session_id": value.session_id,
                "session_instance_id": value.session_instance_id,
                "run_epoch": value.run_epoch,
                "transcript_cursor": value.transcript_cursor,
                "transcript_sha256": value.transcript_sha256,
                "checkpoint_sha256": value.checkpoint_sha256,
            }
        return value

    @field_validator("target", mode="before")
    @classmethod
    def copy_target(cls, value):
        if isinstance(value, SessionMessageTarget):
            if type(value) is not SessionMessageTarget:
                raise TypeError("target must be a SessionMessageTarget.")
            return {
                "session_instance_id": value.session_instance_id,
                "run_epoch": value.run_epoch,
                "transcript_cursor": value.transcript_cursor,
            }
        return value

    @field_validator("expires_at")
    @classmethod
    def validate_expiry(cls, value: datetime | None) -> datetime | None:
        return None if value is None else normalize_utc_datetime(value, "expires_at")


def copy_session_message_conditions(value: SessionMessageConditions) -> SessionMessageConditions:
    """Revalidate caller-owned instances without invoking their serializers."""

    if type(value) is not SessionMessageConditions:
        raise TypeError("conditions must be SessionMessageConditions.")
    return SessionMessageConditions(
        source=value.source, target=value.target, expires_at=value.expires_at
    )


class SessionMessageCursor(_MessageLifecycleModel):
    """Incarnation-bound position within a fixed admission high-water mark.

    Priority is NEXT_TURN (0), ON_IDLE (1), then unreadable modes (2).
    The cursor is pagination input, never application access authority.
    """

    session_instance_id: StrictStr = Field(min_length=1, max_length=512)
    through_ordering_key: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    after_priority: StrictInt = Field(ge=0, le=2)
    after_ordering_key: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("session_instance_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "session_instance_id")

    @model_validator(mode="after")
    def validate_position(self) -> SessionMessageCursor:
        if self.after_ordering_key > self.through_ordering_key:
            raise ValueError("Session-message cursor exceeds its admission boundary.")
        return self


def copy_session_message_cursor(value: SessionMessageCursor) -> SessionMessageCursor:
    if type(value) is not SessionMessageCursor:
        raise TypeError("cursor must be SessionMessageCursor.")
    return SessionMessageCursor(
        session_instance_id=value.session_instance_id,
        through_ordering_key=value.through_ordering_key,
        after_priority=value.after_priority,
        after_ordering_key=value.after_ordering_key,
    )


class SessionMessageQuery(_MessageLifecycleModel):
    """Bounded delivery-priority inspection, including terminal and unreadable rows."""

    session_id: StrictStr = Field(min_length=1, max_length=512)
    cursor: SessionMessageCursor | None = None
    limit: StrictInt = Field(default=50, ge=1, le=100)

    @field_validator("cursor", mode="before")
    @classmethod
    def copy_cursor(cls, value):
        return (
            copy_session_message_cursor(value) if isinstance(value, SessionMessageCursor) else value
        )

    @field_validator("session_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "session_id")


class SessionMessageActionRequest(_MessageLifecycleModel):
    """Compare-and-set withdrawal/quarantine of an inspected queue record."""

    session_id: StrictStr = Field(min_length=1, max_length=512)
    session_instance_id: StrictStr = Field(min_length=1, max_length=512)
    queue_id: StrictStr = Field(min_length=1, max_length=512)
    idempotency_key: StrictStr = Field(min_length=1, max_length=256)
    expected_revision: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    action: Literal["withdraw", "quarantine"]
    requested_by: ResolutionActor | None = None

    @field_validator("session_id", "session_instance_id", "queue_id", "idempotency_key")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("requested_by", mode="before")
    @classmethod
    def copy_actor(cls, value):
        return copy_resolution_actor(value) if isinstance(value, ResolutionActor) else value


class SessionMessageAccessContext(_MessageLifecycleModel):
    """Verified identity from trusted SDK code or HTTP authentication, not JSON metadata."""

    subject: StrictStr = Field(min_length=1, max_length=512)
    tenant: StrictStr | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("subject", "tenant")
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        return None if value is None else require_durable_clean_nonblank(value, info.field_name)


def copy_session_message_action(value: SessionMessageActionRequest) -> SessionMessageActionRequest:
    if type(value) is not SessionMessageActionRequest:
        raise TypeError("Queue actions require a SessionMessageActionRequest.")
    return SessionMessageActionRequest(
        session_id=value.session_id,
        session_instance_id=value.session_instance_id,
        queue_id=value.queue_id,
        idempotency_key=value.idempotency_key,
        expected_revision=value.expected_revision,
        action=value.action,
        requested_by=copy_resolution_actor(value.requested_by),
    )


class SessionMessageAccessPolicy(ABC):
    """Application-owned, side-effect-free authorization using trusted ownership data.

    Authentication and matching tenant strings are not ownership evidence. The
    implementation must resolve its own subject/tenant/session permission map.
    Source access is requested independently from target admission or inspection.
    """

    @abstractmethod
    def authorize(
        self,
        context: SessionMessageAccessContext,
        *,
        session_id: str,
        session_instance_id: str,
        action: Literal["inspect", "enqueue", "source", "withdraw", "quarantine"],
    ) -> bool: ...


def session_message_rejection(
    conditions: SessionMessageConditions,
    *,
    session_instance_id: str,
    run_epoch: int,
    transcript_cursor: int,
    now: datetime,
) -> SessionMessageQueueStatus | None:
    """Pure store-transaction decision. Expiry (including equality) wins over stale."""

    conditions = copy_session_message_conditions(conditions)
    now = normalize_utc_datetime(now, "now")
    if conditions.expires_at is not None and conditions.expires_at <= now:
        return SessionMessageQueueStatus.EXPIRED
    target = conditions.target
    if target is not None and (
        target.session_instance_id != session_instance_id
        or target.run_epoch != run_epoch
        or target.transcript_cursor != transcript_cursor
    ):
        return SessionMessageQueueStatus.STALE
    return None


def session_message_checkpoint_sha256(checkpoint: object) -> str:
    """Bind the exact stored checkpoint, not the child-fork projection of it."""

    return sha256(
        canonical_durable_json_bytes(checkpoint, "session-message checkpoint")
    ).hexdigest()
