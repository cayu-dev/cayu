from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_value,
    require_durable_clean_nonblank,
)

if TYPE_CHECKING:
    from cayu.providers.base import ModelRequest, ModelStreamEvent

PROVIDER_OPERATION_ID_MAX_CHARS = 512
PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS = 128


class ProviderOperationMode(StrEnum):
    """Whether one configured provider uses its optional operation adapter."""

    SYNCHRONOUS = "synchronous"
    BACKGROUND = "background"


class ProviderOperationStatus(StrEnum):
    """Provider-neutral lifecycle states for reconnectable model work."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    UNAVAILABLE = "unavailable"

    @property
    def terminal(self) -> bool:
        return self not in {ProviderOperationStatus.QUEUED, ProviderOperationStatus.IN_PROGRESS}


class ProviderOperationRecoveryMetadata(BaseModel):
    """Small provider-neutral continuation state, never request or response data.

    Runtime core deliberately admits only one opaque cursor. Provider adapters
    translate it to their own API parameter names; arbitrary metadata objects
    cannot smuggle credentials, request bodies, reasoning, or response content
    into durable recovery state.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)

    cursor: (
        Annotated[
            StrictInt,
            Field(ge=0, le=MAX_DURABLE_JSON_INTEGER),
        ]
        | None
    ) = None


class ProviderOperationState(BaseModel):
    """Bounded durable state needed to inspect or reconnect provider-owned work.

    Provider adapters must never put credentials, raw request bodies, or model
    reasoning in ``recovery_metadata``. Cayu enforces portable durable JSON and
    a small byte bound; the metadata remains runtime-private when operation
    evidence is projected to public events.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)

    version: Literal[1] = 1
    operation_id: str = Field(max_length=PROVIDER_OPERATION_ID_MAX_CHARS)
    stream_protocol: str = Field(max_length=PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS)
    recovery_metadata: ProviderOperationRecoveryMetadata = Field(
        default_factory=ProviderOperationRecoveryMetadata
    )

    @field_validator("operation_id", "stream_protocol")
    @classmethod
    def validate_bounded_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("recovery_metadata", mode="before")
    @classmethod
    def validate_recovery_metadata(cls, value: object) -> ProviderOperationRecoveryMetadata:
        if type(value) is ProviderOperationRecoveryMetadata:
            value = value.model_dump(mode="python", exclude_none=True)
        if type(value) is not dict:
            raise ValueError("recovery_metadata must be a JSON object.")
        try:
            metadata = ProviderOperationRecoveryMetadata.model_validate(value)
        except ValueError:
            raise ValueError(
                "recovery_metadata must contain only a bounded provider-operation cursor "
                "and must not contain credentials, requests, reasoning, or responses."
            ) from None
        copied = copy_durable_json_value(
            metadata.model_dump(mode="python", exclude_none=True),
            "recovery_metadata",
        )
        return ProviderOperationRecoveryMetadata.model_validate(copied)


@dataclass(frozen=True)
class ProviderOperationConnection:
    """A newly started or reconnected operation and its normalized event stream."""

    state: ProviderOperationState
    status: ProviderOperationStatus
    events: AsyncIterator[ModelStreamEvent]


@dataclass(frozen=True)
class ProviderOperationStartRequest:
    """One explicitly enabled dispatch with stable provider idempotency authority."""

    request: ModelRequest
    idempotency_key: str


@dataclass(frozen=True)
class ProviderOperationSnapshot:
    """The provider's current interpretation of one durable operation identity."""

    state: ProviderOperationState
    status: ProviderOperationStatus
    events: tuple[ModelStreamEvent, ...] = ()


class ProviderOperationAdapter(ABC):
    """Optional provider-owned background-operation capability.

    Keeping this adapter separate from ``ModelProvider.stream`` makes the
    capability explicit while preserving the synchronous provider contract.
    """

    @abstractmethod
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        """Submit one request and return its reconnectable identity and stream."""

    @abstractmethod
    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        """Inspect an operation without opening a stream."""

    @abstractmethod
    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        """Reconnect to an existing provider-owned stream."""

    @abstractmethod
    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        """Request cancellation and return the resulting provider state."""


def copy_provider_operation_state(state: ProviderOperationState) -> ProviderOperationState:
    """Detach one provider-owned operation identity at the runtime boundary."""

    if type(state) is not ProviderOperationState:
        raise TypeError("Provider operation state must be a ProviderOperationState instance.")
    return ProviderOperationState.model_validate(state.model_dump(mode="python"))


def copy_provider_operation_connection(
    connection: ProviderOperationConnection,
) -> ProviderOperationConnection:
    """Validate a provider-owned connection without consuming its event stream."""

    if type(connection) is not ProviderOperationConnection:
        raise TypeError("Provider operation start must return ProviderOperationConnection.")
    if type(connection.status) is not ProviderOperationStatus:
        raise TypeError("Provider operation status must be a ProviderOperationStatus.")
    if not hasattr(connection.events, "__aiter__"):
        raise TypeError("Provider operation events must be an async iterator.")
    return ProviderOperationConnection(
        state=copy_provider_operation_state(connection.state),
        status=connection.status,
        events=connection.events,
    )


def copy_provider_operation_snapshot(
    snapshot: ProviderOperationSnapshot,
) -> ProviderOperationSnapshot:
    """Detach one provider-owned inspection or cancellation result."""

    from cayu.providers.base import copy_model_stream_event

    if type(snapshot) is not ProviderOperationSnapshot:
        raise TypeError("Provider operation inspection must return ProviderOperationSnapshot.")
    if type(snapshot.status) is not ProviderOperationStatus:
        raise TypeError("Provider operation status must be a ProviderOperationStatus.")
    return ProviderOperationSnapshot(
        state=copy_provider_operation_state(snapshot.state),
        status=snapshot.status,
        events=tuple(copy_model_stream_event(event) for event in snapshot.events),
    )


__all__ = [
    "PROVIDER_OPERATION_ID_MAX_CHARS",
    "PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS",
    "ProviderOperationAdapter",
    "ProviderOperationConnection",
    "ProviderOperationMode",
    "ProviderOperationRecoveryMetadata",
    "ProviderOperationSnapshot",
    "ProviderOperationStartRequest",
    "ProviderOperationState",
    "ProviderOperationStatus",
    "copy_provider_operation_connection",
    "copy_provider_operation_snapshot",
    "copy_provider_operation_state",
]
