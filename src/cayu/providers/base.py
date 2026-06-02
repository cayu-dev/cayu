from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_nonblank
from cayu.core.messages import Message, copy_message


class ModelStreamEventType(StrEnum):
    TEXT_DELTA = "text_delta"
    TOOL_CALL = "tool_call"
    COMPLETED = "completed"
    ERROR = "error"


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("tools", "options", mode="before")
    @classmethod
    def copy_json_request_data(cls, value, info):
        return copy_json_value(value, info.field_name)

    @field_validator("model")
    @classmethod
    def validate_nonblank_model(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)


class ModelStreamEvent(BaseModel):
    """Provider-native stream event.

    Provider adapters may expose this lower-level shape while normalizing SDK
    responses. Runtime code must convert these events into framework `Event`
    records before persisting, dashboarding, or forwarding them.
    """

    model_config = ConfigDict(extra="forbid")

    type: ModelStreamEventType
    delta: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "payload")

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, value: object) -> ModelStreamEventType:
        if isinstance(value, ModelStreamEventType):
            return value
        return ModelStreamEventType(require_nonblank(value, "type"))

    @classmethod
    def text_delta(cls, delta: str) -> "ModelStreamEvent":
        return cls(type=ModelStreamEventType.TEXT_DELTA, delta=delta)

    @classmethod
    def tool_call(
        cls,
        *,
        name: str,
        arguments: dict[str, Any],
        id: str | None = None,
    ) -> "ModelStreamEvent":
        if not isinstance(arguments, dict):
            raise ValueError("`arguments` must be a dictionary.")
        payload: dict[str, Any] = {
            "name": require_nonblank(name, "name"),
            "arguments": copy_json_value(arguments, "arguments"),
        }
        if id is not None:
            payload["id"] = require_nonblank(id, "id")
        return cls(type=ModelStreamEventType.TOOL_CALL, payload=payload)

    @classmethod
    def completed(cls, payload: dict[str, Any] | None = None) -> "ModelStreamEvent":
        return cls(
            type=ModelStreamEventType.COMPLETED,
            payload={} if payload is None else payload,
        )

    @classmethod
    def error(cls, message: str) -> "ModelStreamEvent":
        return cls(
            type=ModelStreamEventType.ERROR,
            payload={"error": require_nonblank(message, "message")},
        )


def copy_model_stream_event(event: ModelStreamEvent) -> ModelStreamEvent:
    if type(event) is not ModelStreamEvent:
        raise TypeError("Model providers must yield ModelStreamEvent instances.")
    event_type = event.type
    if type(event_type) is not ModelStreamEventType:
        raise ValueError("Model provider stream event type must be a ModelStreamEventType.")
    if type(event.delta) is not str:
        raise ValueError("Model provider stream event delta must be a string.")
    if type(event.payload) is not dict:
        raise ValueError("Model provider stream event payload must be an object.")
    return ModelStreamEvent(
        type=event_type,
        delta=event.delta,
        payload=copy_json_value(event.payload, "payload"),
    )


class ModelProvider(ABC):
    """Normalizes provider-specific model streams."""

    name: str

    @abstractmethod
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Stream model events for one request."""
