from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.core.messages import Message, copy_message


class ModelStreamEventType(StrEnum):
    TEXT_DELTA = "text_delta"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    COMPLETED = "completed"
    ERROR = "error"


class ModelFinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


class InputTokenCountMethod(StrEnum):
    """How a provider counted one model request before submission."""

    OFFICIAL = "official"
    LOCAL_TOKENIZER = "local_tokenizer"
    HEURISTIC = "heuristic"
    UNAVAILABLE = "unavailable"


class InputTokenCountConfidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNAVAILABLE = "unavailable"


class ModelContextOverflowError(RuntimeError):
    """Provider-neutral signal that a model request was too large for context.

    Provider adapters should raise this only for clear context-window or request-size
    overflow responses. Runtime recovery can then shrink model-facing context and
    retry without depending on provider-specific exception classes.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        status_code: int | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(require_nonblank(message, "message"))
        self.provider = require_clean_nonblank(provider, "provider")
        if status_code is not None and (type(status_code) is not int or status_code < 100):
            raise ValueError("status_code must be a valid HTTP status code.")
        self.status_code = status_code
        self.error_type = _optional_clean_error_field(error_type, "error_type")
        self.error_code = _optional_clean_error_field(error_code, "error_code")
        self.request_id = _optional_clean_error_field(request_id, "request_id")
        self.response_body = response_body


def _optional_clean_error_field(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return require_clean_nonblank(value, field_name)


class InputTokenCountResult(BaseModel):
    """Provider-neutral input token count for a model request.

    Official provider counters should use `method="official"` and
    `confidence="high"`. Official remote counters can add latency and consume
    provider rate limits. Their billing behavior is provider-specific and
    should be documented in `metadata` when known. Local tokenizers and
    heuristics are useful for observability, but callers should not treat them
    as hard provider-limit guarantees.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    method: InputTokenCountMethod
    confidence: InputTokenCountConfidence
    components: dict[str, int] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("method", mode="before")
    @classmethod
    def validate_method(cls, value: object) -> InputTokenCountMethod:
        if isinstance(value, InputTokenCountMethod):
            return value
        if not isinstance(value, str):
            raise ValueError("`method` must be a string.")
        return InputTokenCountMethod(require_clean_nonblank(value, "method"))

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, value: object) -> InputTokenCountConfidence:
        if isinstance(value, InputTokenCountConfidence):
            return value
        if not isinstance(value, str):
            raise ValueError("`confidence` must be a string.")
        return InputTokenCountConfidence(require_clean_nonblank(value, "confidence"))

    @field_validator("components", mode="before")
    @classmethod
    def copy_components(cls, value: dict[str, Any]) -> dict[str, int]:
        copied = copy_json_value(value, "components")
        if type(copied) is not dict:
            raise ValueError("`components` must be a dictionary.")
        result: dict[str, int] = {}
        for key, component_value in copied.items():
            if type(key) is not str:
                raise ValueError("Input token count component keys must be strings.")
            clean_key = require_clean_nonblank(key, "component key")
            if type(component_value) is not int:
                raise ValueError("Input token count component values must be integers.")
            if component_value < 0:
                raise ValueError("Input token count component values must be non-negative.")
            result[clean_key] = component_value
        return result

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_json_value(value, "metadata")
        if type(copied) is not dict:
            raise ValueError("`metadata` must be a dictionary.")
        return copied


class ModelCompletion(BaseModel):
    """Provider-neutral completion metadata for a model step."""

    model_config = ConfigDict(extra="forbid")

    finish_reason: ModelFinishReason
    raw_finish_reason: str | None = None
    status: str | None = None

    @field_validator("finish_reason", mode="before")
    @classmethod
    def validate_finish_reason(cls, value: object) -> ModelFinishReason:
        if isinstance(value, ModelFinishReason):
            return value
        if not isinstance(value, str):
            raise ValueError("`finish_reason` must be a string.")
        return ModelFinishReason(require_clean_nonblank(value, "finish_reason"))

    @field_validator("raw_finish_reason", "status")
    @classmethod
    def validate_optional_clean_string(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


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
        return require_clean_nonblank(value, info.field_name)


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
    completion: ModelCompletion | None = None

    @field_validator("payload", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "payload")

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, value: object) -> ModelStreamEventType:
        if isinstance(value, ModelStreamEventType):
            return value
        if not isinstance(value, str):
            raise ValueError("`type` must be a string.")
        return ModelStreamEventType(require_clean_nonblank(value, "type"))

    @model_validator(mode="after")
    def validate_completion(self) -> ModelStreamEvent:
        if self.type == ModelStreamEventType.COMPLETED:
            if self.completion is None:
                self.completion = normalize_model_completion(self.payload)
            return self
        if self.completion is not None:
            raise ValueError("Only completed model stream events can include completion metadata.")
        return self

    @classmethod
    def text_delta(cls, delta: str) -> ModelStreamEvent:
        return cls(type=ModelStreamEventType.TEXT_DELTA, delta=delta)

    @classmethod
    def thinking(
        cls,
        delta: str = "",
        *,
        provider_state: dict[str, Any] | None = None,
    ) -> ModelStreamEvent:
        """A reasoning/thinking event.

        `delta` is the (possibly empty) reasoning text. `provider_state` carries the
        opaque round-trip payload of a *complete* block — the Anthropic ``signature``
        or ``redacted_thinking`` data. When present, the runtime materializes a
        standalone `ThinkingPart`; events without it accumulate as streamed text.
        """
        if type(delta) is not str:
            raise ValueError("`delta` must be a string.")
        payload: dict[str, Any] = {}
        if provider_state is not None:
            if not isinstance(provider_state, dict):
                raise ValueError("`provider_state` must be a dictionary.")
            payload["provider_state"] = copy_json_value(provider_state, "provider_state")
        return cls(type=ModelStreamEventType.THINKING, delta=delta, payload=payload)

    @classmethod
    def tool_call(
        cls,
        *,
        name: str,
        arguments: dict[str, Any],
        id: str | None = None,
    ) -> ModelStreamEvent:
        if not isinstance(arguments, dict):
            raise ValueError("`arguments` must be a dictionary.")
        payload: dict[str, Any] = {
            "name": require_clean_nonblank(name, "name"),
            "arguments": copy_json_value(arguments, "arguments"),
        }
        if id is not None:
            payload["id"] = require_clean_nonblank(id, "id")
        return cls(type=ModelStreamEventType.TOOL_CALL, payload=payload)

    @classmethod
    def completed(cls, payload: dict[str, Any] | None = None) -> ModelStreamEvent:
        payload = {} if payload is None else payload
        return cls(
            type=ModelStreamEventType.COMPLETED,
            payload=payload,
        )

    @classmethod
    def error(cls, message: str) -> ModelStreamEvent:
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
        completion=copy_model_completion(event.completion),
    )


def copy_model_completion(completion: ModelCompletion | None) -> ModelCompletion | None:
    if completion is None:
        return None
    if type(completion) is not ModelCompletion:
        raise TypeError("Model completion must be a ModelCompletion instance.")
    return ModelCompletion(
        finish_reason=completion.finish_reason,
        raw_finish_reason=completion.raw_finish_reason,
        status=completion.status,
    )


def copy_input_token_count_result(
    result: InputTokenCountResult | None,
) -> InputTokenCountResult | None:
    if result is None:
        return None
    if type(result) is not InputTokenCountResult:
        raise TypeError("Input token count result must be an InputTokenCountResult instance.")
    return InputTokenCountResult(
        input_tokens=result.input_tokens,
        method=result.method,
        confidence=result.confidence,
        components=copy_json_value(result.components, "components"),
        metadata=copy_json_value(result.metadata, "metadata"),
    )


def normalize_model_completion(payload: dict[str, Any]) -> ModelCompletion:
    """Normalize known provider completion payloads without discarding raw fields."""

    if type(payload) is not dict:
        raise ValueError("Model completed payload must be a dictionary.")
    status = _optional_payload_string(payload, "status")
    raw_finish_reason = _raw_finish_reason(payload)
    finish_reason = _normalized_finish_reason(
        raw_finish_reason=raw_finish_reason,
        status=status,
        incomplete_details=payload.get("incomplete_details"),
    )
    return ModelCompletion(
        finish_reason=finish_reason,
        raw_finish_reason=raw_finish_reason,
        status=status,
    )


def _raw_finish_reason(payload: dict[str, Any]) -> str | None:
    for key in ("finish_reason", "stop_reason", "reason"):
        value = _optional_payload_string(payload, key)
        if value is not None:
            return value
    incomplete_details = payload.get("incomplete_details")
    if isinstance(incomplete_details, dict):
        return _optional_payload_string(incomplete_details, "reason")
    return None


def _normalized_finish_reason(
    *,
    raw_finish_reason: str | None,
    status: str | None,
    incomplete_details: object,
) -> ModelFinishReason:
    if status == "failed":
        return ModelFinishReason.ERROR
    if status == "incomplete":
        reason = raw_finish_reason
        if reason in {"max_output_tokens", "max_tokens", "length"}:
            return ModelFinishReason.LENGTH
        if reason in {"content_filter", "safety", "refusal"}:
            return ModelFinishReason.CONTENT_FILTER
        return ModelFinishReason.UNKNOWN
    if raw_finish_reason is None:
        return ModelFinishReason.UNKNOWN
    if raw_finish_reason in {"stop", "end_turn"}:
        return ModelFinishReason.STOP
    if raw_finish_reason in {"tool_calls", "tool_use"}:
        return ModelFinishReason.TOOL_CALLS
    if raw_finish_reason in {"length", "max_tokens", "max_output_tokens"}:
        return ModelFinishReason.LENGTH
    if raw_finish_reason in {"content_filter", "safety", "refusal"}:
        return ModelFinishReason.CONTENT_FILTER
    if raw_finish_reason in {"error", "failed"}:
        return ModelFinishReason.ERROR
    if isinstance(incomplete_details, dict):
        reason = _optional_payload_string(cast("dict[str, Any]", incomplete_details), "reason")
        if reason is not None:
            return _normalized_finish_reason(
                raw_finish_reason=reason,
                status="incomplete",
                incomplete_details=None,
            )
    return ModelFinishReason.UNKNOWN


def _optional_payload_string(payload: dict[str, Any], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if type(value) is not str:
        raise ValueError(f"Model completed payload `{key}` must be a string.")
    return require_clean_nonblank(value, key)


class ModelProvider(ABC):
    """Normalizes provider-specific model streams."""

    name: str

    async def count_input_tokens(
        self,
        request: ModelRequest,
    ) -> InputTokenCountResult | None:
        """Optionally count the input tokens for one request before submission.

        Providers that need to call a remote counting endpoint should do so here.
        Remote counters are opt-in observability/calibration hooks, not default
        context-overflow enforcement. The default implementation is intentionally
        unavailable so existing providers remain source-compatible.
        """

        return None

    @abstractmethod
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Stream model events for one request."""
