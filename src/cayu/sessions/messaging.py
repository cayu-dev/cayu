"""Bounded authority for managing the existing durable session-message queue.

These values do not authorize execution. The application authorizes access;
SessionStore owns admission, freshness evaluation, and terminal compare-and-set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    compact_json_utf8_size,
    require_durable_clean_nonblank,
    require_durable_json_text,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.approvals.tools import ResolutionActor, copy_resolution_actor, resolution_actor_payload
from cayu.events import Event, copy_event
from cayu.messages import Message, MessageRole, PeerContentPart, detach_message
from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile


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


class SessionQueuedMessagesPending(RuntimeError):
    """Terminalization lost a race to durable queued session input."""


# Steering messages match server prompts: a 1 MiB request minus a JSON envelope.
SESSION_MESSAGE_CONTENT_MAX_BYTES = 1024 * 1024 - 64 * 1024


# Queue reads project a stored column inline only up to this size; larger
# values are reported as an out-of-band digest (an unreadable record). Twice the
# content limit leaves room for the typed message JSON envelope.
SESSION_MESSAGE_QUEUE_STORAGE_VALUE_MAX_BYTES = 2 * SESSION_MESSAGE_CONTENT_MAX_BYTES


SESSION_MESSAGE_DELIVERY_BATCH_LIMIT = 100


class SessionMessageDeliveryMode(StrEnum):
    NEXT_TURN = "next_turn"
    ON_IDLE = "on_idle"


class EnqueueSessionMessageRequest(BaseModel):
    """Submit durable user steering for an active session."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str
    idempotency_key: str = Field(max_length=256)
    # ``content`` remains the bounded human-readable projection used by
    # existing clients and audit surfaces. ``message`` is the authoritative
    # typed input when present, allowing queued multimodal user messages to
    # retain their artifact references across durable delivery.
    content: str = Field(max_length=SESSION_MESSAGE_CONTENT_MAX_BYTES)
    message: Message | None = None
    delivery_mode: SessionMessageDeliveryMode
    requested_by: ResolutionActor | None = None
    conditions: SessionMessageConditions = Field(default_factory=SessionMessageConditions)
    _input_redactions_applied: bool = PrivateAttr(default=False)

    @field_validator("conditions", mode="before")
    @classmethod
    def copy_conditions(cls, value):
        if isinstance(value, SessionMessageConditions):
            return copy_session_message_conditions(value)
        return value

    @field_validator("session_id", "idempotency_key")
    @classmethod
    def validate_required_strings(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = require_nonblank(value, "content")
        if len(value.encode("utf-8")) > SESSION_MESSAGE_CONTENT_MAX_BYTES:
            raise ValueError(
                "content exceeds the maximum encoded size of "
                f"{SESSION_MESSAGE_CONTENT_MAX_BYTES} bytes."
            )
        return value

    @field_validator("message")
    @classmethod
    def copy_typed_message(cls, value: Message | None) -> Message | None:
        if value is None:
            return None
        message = detach_message(value)
        if message.role is not MessageRole.USER:
            raise ValueError("Queued session messages must have the user role.")
        if compact_json_utf8_size(message.model_dump(mode="json")) > (
            SESSION_MESSAGE_CONTENT_MAX_BYTES
        ):
            raise ValueError(
                "message exceeds the maximum encoded size of "
                f"{SESSION_MESSAGE_CONTENT_MAX_BYTES} bytes."
            )
        return message

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

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    queue_id: str
    session_id: str
    idempotency_key: str
    content: str
    message: Message | None = None
    delivery_mode: SessionMessageDeliveryMode
    status: SessionMessageQueueStatus
    ordering_key: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    accepted_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    accepted_transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    accepted_event_id: str
    accepted_at: datetime
    requested_by: ResolutionActor | None = None
    delivered_run_epoch: StrictInt | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)
    delivered_transcript_cursor: StrictInt | None = Field(
        default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER
    )
    delivered_event_id: str | None = None
    delivered_at: datetime | None = None
    conditions: SessionMessageConditions = Field(default_factory=SessionMessageConditions)

    @field_validator("conditions", mode="before")
    @classmethod
    def copy_conditions(cls, value):
        if isinstance(value, SessionMessageConditions):
            return copy_session_message_conditions(value)
        return value

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

    @field_validator("message")
    @classmethod
    def copy_typed_message(cls, value: Message | None) -> Message | None:
        if value is None:
            return None
        message = detach_message(value)
        if message.role is not MessageRole.USER and not (
            message.role is MessageRole.ASSISTANT
            and message.content
            and all(type(part) is PeerContentPart for part in message.content)
        ):
            raise ValueError(
                "Queued session messages must have the user role or peer assistant content."
            )
        return message


class SessionMessageInspectionRecord(BaseModel):
    """Safe row envelope: unreadable content never becomes deliverable input."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    queue_id: StrictStr = Field(min_length=1, max_length=512)
    ordering_key: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    revision: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    status: SessionMessageQueueStatus | None
    validity: Literal["valid", "unreadable"]
    message: SessionQueuedMessage | None = Field(default=None, repr=False)
    terminal_event_id: StrictStr | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_inspection(self) -> SessionMessageInspectionRecord:
        if self.validity == "valid":
            if (
                self.message is None
                or self.status != self.message.status
                or self.queue_id != self.message.queue_id
                or self.ordering_key != self.message.ordering_key
            ):
                raise ValueError("Queue inspection record has inconsistent typed evidence.")
        elif self.message is not None:
            raise ValueError("Unreadable queue inspection cannot contain message content.")
        return self


class SessionMessageInspection(BaseModel):
    """One protected delivery-priority page and its exact session instance."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: StrictStr = Field(min_length=1, max_length=512)
    session_instance_id: StrictStr = Field(min_length=1, max_length=512)
    records: tuple[SessionMessageInspectionRecord, ...] = Field(default=(), max_length=100)
    next_cursor: SessionMessageCursor | None = None

    @model_validator(mode="after")
    def validate_cursor(self) -> SessionMessageInspection:
        if self.next_cursor is not None and (
            self.next_cursor.session_instance_id != self.session_instance_id
            or not self.records
            or self.next_cursor.after_ordering_key != self.records[-1].ordering_key
            or any(
                record.ordering_key > self.next_cursor.through_ordering_key
                for record in self.records
            )
        ):
            raise ValueError("Queue inspection cursor conflicts with its page.")
        return self


class SessionMessageActionResult(BaseModel):
    """Terminal queue mutation and its atomically persisted content-free event."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record: SessionMessageInspectionRecord
    event: Event
    replayed: StrictBool = False


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
    delivery_id: str
    interaction_id: str | None = None
    eligible_through: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    has_more: StrictBool = False
    replayed: StrictBool = False
    active_invocation_profile: ActiveInvocationExecutionProfile | None = None

    @field_validator("messages", mode="before")
    @classmethod
    def copy_messages(cls, value) -> tuple[SessionQueuedMessage, ...]:
        return tuple(message.model_copy(deep=True) for message in value)

    @field_validator("events", mode="before")
    @classmethod
    def copy_events(cls, value) -> tuple[Event, ...]:
        return tuple(copy_event(event) for event in value)

    @field_validator("delivery_id", "interaction_id")
    @classmethod
    def validate_identifiers(cls, value: str | None, info) -> str | None:
        if value is None:
            if info.field_name == "delivery_id":
                raise ValueError("delivery_id is required.")
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("active_invocation_profile", mode="before")
    @classmethod
    def copy_active_invocation_profile(
        cls,
        value: object,
    ) -> ActiveInvocationExecutionProfile | None:
        if value is None:
            return None
        if isinstance(value, ActiveInvocationExecutionProfile):
            value = value.model_dump(mode="json")
        return ActiveInvocationExecutionProfile.model_validate(value)


def copy_enqueue_session_message_request(
    request: EnqueueSessionMessageRequest,
) -> EnqueueSessionMessageRequest:
    if type(request) is not EnqueueSessionMessageRequest:
        raise TypeError("Queued input requires an EnqueueSessionMessageRequest.")
    copied = EnqueueSessionMessageRequest(
        session_id=request.session_id,
        idempotency_key=request.idempotency_key,
        content=request.content,
        message=(None if request.message is None else detach_message(request.message)),
        delivery_mode=request.delivery_mode,
        requested_by=copy_resolution_actor(request.requested_by),
        conditions=copy_session_message_conditions(request.conditions),
    )
    copied._input_redactions_applied = request._input_redactions_applied
    return copied


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
        or existing.message != request.message
        or existing.delivery_mode != request.delivery_mode
        or existing.conditions != request.conditions
        or resolution_actor_payload(existing.requested_by)
        != resolution_actor_payload(request.requested_by)
    ):
        raise ValueError(
            "Session message idempotency key was already used for a different request."
        )


def queued_session_message_input(message: SessionQueuedMessage) -> Message:
    """Return the authoritative detached user input for a queued record."""

    if type(message) is not SessionQueuedMessage:
        raise TypeError("message must be a SessionQueuedMessage.")
    if message.message is not None:
        return detach_message(message.message)
    return Message.text(MessageRole.USER, message.content)


def enqueue_session_message_input(request: EnqueueSessionMessageRequest) -> Message:
    """Return the authoritative detached user input for an enqueue request."""

    if type(request) is not EnqueueSessionMessageRequest:
        raise TypeError("request must be an EnqueueSessionMessageRequest.")
    if request.message is not None:
        return detach_message(request.message)
    return Message.text(MessageRole.USER, request.content)
