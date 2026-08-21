from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_json_value,
    require_durable_clean_nonblank,
)

if TYPE_CHECKING:
    from cayu.providers.base import ModelRequest, ModelStreamEvent

PROVIDER_OPERATION_ID_MAX_CHARS = 512
PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS = 128
PROVIDER_OPERATION_RECOVERY_OPAQUE_MAX_BYTES = 4096


class ProviderOperationMode(StrEnum):
    """Whether one configured provider uses its optional operation adapter."""

    SYNCHRONOUS = "synchronous"
    BACKGROUND = "background"


class ProviderOperationCancellationSupport(StrEnum):
    """Whether an adapter can target an existing durable operation."""

    UNSUPPORTED = "unsupported"
    SUPPORTED = "supported"


class ProviderOperationStartIdempotencySupport(StrEnum):
    """Whether recovering an accepted start from its key alone is proven exact."""

    UNSUPPORTED = "unsupported"
    EXACT = "exact"


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


class ProviderOperationMalformedError(ValueError):
    """A provider operation returned invalid protocol data during recovery."""


class ProviderOperationRecoveryMetadata(BaseModel):
    """Small provider-neutral continuation state, never request or response data.

    ``cursor`` is the runtime-owned monotonic boundary number. ``opaque`` is a
    bounded adapter-owned continuation value: Cayu copies and stores it without
    interpreting provider-specific field names, and gives it back only to the
    same operation adapter. Provider adapters must not put credentials, raw
    request or response bodies, or model reasoning in it.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)

    cursor: (
        Annotated[
            StrictInt,
            Field(ge=0, le=MAX_DURABLE_JSON_INTEGER),
        ]
        | None
    ) = None
    opaque: dict[str, object] = Field(default_factory=dict, exclude_if=lambda value: not value)

    @field_validator("opaque", mode="before")
    @classmethod
    def validate_opaque(cls, value: object) -> dict[str, object]:
        copied = copy_durable_json_object(value, "recovery_metadata.opaque")
        if (
            len(
                canonical_durable_json_bytes(
                    copied,
                    "recovery_metadata.opaque",
                )
            )
            > PROVIDER_OPERATION_RECOVERY_OPAQUE_MAX_BYTES
        ):
            raise ValueError(
                "recovery_metadata.opaque exceeds the durable byte limit of "
                f"{PROVIDER_OPERATION_RECOVERY_OPAQUE_MAX_BYTES}."
            )
        return copied


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
                "recovery_metadata must contain only a monotonic cursor and bounded opaque "
                "adapter state, and must not contain credentials, requests, reasoning, or "
                "responses."
            ) from None
        copied = copy_durable_json_value(
            metadata.model_dump(mode="python", exclude_none=True),
            "recovery_metadata",
        )
        return ProviderOperationRecoveryMetadata.model_validate(copied)


@dataclass(frozen=True)
class ProviderOperationConnection:
    """A started, recovered, or reconnected operation and its normalized event stream."""

    state: ProviderOperationState
    status: ProviderOperationStatus
    events: AsyncIterator[ModelStreamEvent]


@dataclass(frozen=True)
class ProviderOperationStartRequest:
    """One explicitly enabled dispatch with stable provider idempotency authority."""

    request: ModelRequest
    idempotency_key: str


@dataclass(frozen=True)
class ProviderOperationStartRecoveryRequest:
    """Recover one accepted start from provider-owned idempotency authority alone."""

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

    @property
    def start_idempotency_support(self) -> ProviderOperationStartIdempotencySupport:
        """Advertise exact key-only start recovery only when the provider proves it.

        ``EXACT`` means :meth:`recover_start` can recover the accepted operation
        from the idempotency key without receiving or durably retaining the raw
        provider request. Providers that require the request to be submitted again
        must leave this as ``UNSUPPORTED``.
        """

        return ProviderOperationStartIdempotencySupport.UNSUPPORTED

    async def recover_start(
        self,
        request: ProviderOperationStartRecoveryRequest,
    ) -> ProviderOperationConnection:
        """Recover an accepted start by its exact provider idempotency identity.

        Adapters advertising ``EXACT`` must override this method. The default is
        intentionally unavailable so adding the recovery seam remains compatible
        with adapters that do not advertise exact start recovery. Implementations
        must only look up work already accepted under the key; this method must not
        create or resubmit provider work.
        """

        del request
        raise NotImplementedError("Provider operation start recovery is unsupported.")

    @property
    def cancellation_support(self) -> ProviderOperationCancellationSupport:
        """Advertise exact-operation cancellation without implying universal support."""

        return ProviderOperationCancellationSupport.UNSUPPORTED

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        """Request cancellation and return the resulting provider state.

        Non-cancellable adapters inherit a truthful bounded ``UNAVAILABLE``
        snapshot. Adapters advertising ``SUPPORTED`` must override this method.
        """

        return ProviderOperationSnapshot(
            state=copy_provider_operation_state(state),
            status=ProviderOperationStatus.UNAVAILABLE,
        )


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
        raise TypeError("Provider operation boundary must return ProviderOperationConnection.")
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
    "PROVIDER_OPERATION_RECOVERY_OPAQUE_MAX_BYTES",
    "PROVIDER_OPERATION_STREAM_PROTOCOL_MAX_CHARS",
    "ProviderOperationAdapter",
    "ProviderOperationCancellationSupport",
    "ProviderOperationConnection",
    "ProviderOperationMalformedError",
    "ProviderOperationMode",
    "ProviderOperationRecoveryMetadata",
    "ProviderOperationSnapshot",
    "ProviderOperationStartIdempotencySupport",
    "ProviderOperationStartRecoveryRequest",
    "ProviderOperationStartRequest",
    "ProviderOperationState",
    "ProviderOperationStatus",
    "copy_provider_operation_connection",
    "copy_provider_operation_snapshot",
    "copy_provider_operation_state",
]
