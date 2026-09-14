"""Detached, provider-neutral result of one completed model request."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import copy_durable_json_object
from cayu.providers.base import (
    ModelCompletion,
    ModelStreamEvent,
    ModelStreamEventType,
    copy_model_completion,
    copy_model_stream_event,
)


class ModelResponse(BaseModel):
    """One response, not an agent loop or a runtime authority handle.

    Events are the single source for text, thinking, ordered tool-call data,
    usage and completion metadata. Construction detaches provider observations.
    The managed inference entrance separately prohibits tool calls and owns
    redaction, byte limits, accounting, and failure publication.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    events: tuple[ModelStreamEvent, ...]

    @field_validator("events", mode="before")
    @classmethod
    def copy_events(cls, value: object) -> tuple[ModelStreamEvent, ...]:
        if type(value) is not list and type(value) is not tuple:
            raise ValueError("Model response events must be a list or tuple.")
        copied = []
        for item in value:
            if type(item) is dict:
                item = ModelStreamEvent.model_validate(item)
            if type(item) is not ModelStreamEvent:
                raise ValueError("Model response requires ModelStreamEvent values.")
            copied.append(copy_model_stream_event(item))
        return tuple(copied)

    @model_validator(mode="after")
    def require_one_completion(self) -> ModelResponse:
        if (
            not self.events
            or self.events[-1].type is not ModelStreamEventType.COMPLETED
            or sum(event.type is ModelStreamEventType.COMPLETED for event in self.events) != 1
            or any(event.type is ModelStreamEventType.ERROR for event in self.events)
        ):
            raise ValueError("Model response requires exactly one final completion and no errors.")
        return self

    @property
    def text(self) -> str:
        return "".join(e.delta for e in self.events if e.type is ModelStreamEventType.TEXT_DELTA)

    @property
    def thinking(self) -> str:
        return "".join(e.delta for e in self.events if e.type is ModelStreamEventType.THINKING)

    @property
    def tool_calls(self) -> tuple[dict[str, Any], ...]:
        """Ordered provider-neutral call payloads; never executes the calls."""

        return tuple(
            copy_durable_json_object(e.payload, "tool_call")
            for e in self.events
            if e.type is ModelStreamEventType.TOOL_CALL
        )

    @property
    def completion(self) -> ModelCompletion:
        completion = copy_model_completion(self.events[-1].completion)
        if completion is None:
            raise ValueError("Model response completion metadata is missing.")
        return completion

    @property
    def payload(self) -> dict[str, Any]:
        return copy_durable_json_object(self.events[-1].payload, "completion_payload")
