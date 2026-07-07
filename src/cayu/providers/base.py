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


class UsageDialect(StrEnum):
    """How a provider's raw usage payload encodes token counters.

    The runtime's usage normalizer folds cache tokens differently per dialect
    (Anthropic reports cache read/write tokens in separate fields excluded from
    ``input_tokens``; OpenAI nests cached input in ``*_tokens_details``). A
    provider whose registered ``name`` is not one of the built-in aliases —
    Claude reached through Bedrock, a gateway, or a renamed adapter — must
    declare its dialect here so the normalizer folds cache tokens correctly
    instead of silently undercounting (and under-billing) them. ``AUTO`` (the
    default) lets the normalizer infer the dialect from the payload shape.
    """

    AUTO = "auto"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GENERIC = "generic"


class ModelProviderError(RuntimeError):
    """Provider-neutral structured model provider failure.

    Provider adapters should raise subclasses of this (or wrap SDK/HTTP failures
    into it) instead of flattening failures to bare message strings. The typed
    fields let runtime code classify retries (`status_code`, `retryable`,
    `retry_after_s`) and let observability keep the provider's own error
    identity (`error_type`, `error_code`, `request_id`) without re-parsing
    message text. `retryable` is tri-state: ``None`` means the provider did not
    classify the failure, leaving the decision to runtime retry policy.
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
        retryable: bool | None = None,
        retry_after_s: float | None = None,
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
        if retryable is not None and type(retryable) is not bool:
            raise ValueError("retryable must be a boolean.")
        self.retryable = retryable
        if retry_after_s is not None:
            if type(retry_after_s) not in {int, float} or retry_after_s < 0:
                raise ValueError("retry_after_s must be a non-negative number.")
            retry_after_s = float(retry_after_s)
        self.retry_after_s = retry_after_s
        self.response_body = response_body

    def error_payload_fields(self) -> dict[str, Any]:
        """JSON-safe structured fields for model stream error payloads.

        Key naming mirrors runtime context-overflow event payloads: the
        provider's own error identity uses the ``provider_error_*`` prefix so
        it cannot collide with the Python-exception ``error_type`` key.
        """
        payload: dict[str, Any] = {"provider": self.provider}
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.error_type is not None:
            payload["provider_error_type"] = self.error_type
        if self.error_code is not None:
            payload["provider_error_code"] = self.error_code
        if self.request_id is not None:
            payload["request_id"] = self.request_id
        if self.retryable is not None:
            payload["retryable"] = self.retryable
        if self.retry_after_s is not None:
            payload["retry_after_s"] = self.retry_after_s
        return payload


class ModelContextOverflowError(ModelProviderError):
    """Provider-neutral signal that a model request was too large for context.

    Provider adapters should raise this only for clear context-window or request-size
    overflow responses. Runtime recovery can then shrink model-facing context and
    retry without depending on provider-specific exception classes. Overflow is
    never retryable as-is (`retryable=False`); recovery must shrink context first.
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
        ModelProviderError.__init__(
            self,
            message,
            provider=provider,
            status_code=status_code,
            error_type=error_type,
            error_code=error_code,
            request_id=request_id,
            retryable=False,
            response_body=response_body,
        )


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


class ModelContextPressureProfile(BaseModel):
    """Provider-supplied local context-pressure estimation hints."""

    model_config = ConfigDict(extra="forbid")

    image_min_tokens: int = Field(default=32, ge=0)
    document_min_tokens: int = Field(default=0, ge=0)
    document_bytes_per_token: int = Field(default=3, ge=1)
    tool_schema_chars_per_token: int = Field(default=4, ge=1)


def copy_model_context_pressure_profile(
    profile: ModelContextPressureProfile | None,
) -> ModelContextPressureProfile:
    if profile is None:
        return ModelContextPressureProfile()
    if type(profile) is not ModelContextPressureProfile:
        raise TypeError("Context pressure profile must be a ModelContextPressureProfile.")
    return ModelContextPressureProfile(**profile.model_dump())


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
    def error(cls, message: str, *, cause: Exception | None = None) -> ModelStreamEvent:
        """An error event; `cause` preserves typed classification in the payload.

        When `cause` is a `ModelProviderError`, its structured fields (provider,
        status_code, provider_error_type/provider_error_code, request_id,
        retryable, retry_after_s) join the payload so retry classification and
        observability survive the event boundary instead of collapsing to text.
        A `ModelContextOverflowError` cause additionally sets
        ``context_overflow: True`` so runtime overflow recovery keeps its typed
        signal even when a provider flattens the overflow into an error event
        instead of raising it as `stream()` requires.
        """
        payload: dict[str, Any] = {"error": require_nonblank(message, "message")}
        if cause is not None:
            if not isinstance(cause, Exception):
                raise ValueError("`cause` must be an Exception.")
            payload["error_type"] = type(cause).__name__
            if isinstance(cause, ModelProviderError):
                payload.update(cause.error_payload_fields())
            if isinstance(cause, ModelContextOverflowError):
                payload["context_overflow"] = True
        return cls(
            type=ModelStreamEventType.ERROR,
            payload=payload,
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
    usage_dialect: UsageDialect = UsageDialect.AUTO
    """Usage payload dialect for cache-token normalization; see ``UsageDialect``.

    Defaults to ``AUTO`` (infer from payload shape). Adapters that emit a fixed
    dialect regardless of their registered ``name`` should override this so
    renamed or gateway-routed deployments still fold cache tokens correctly.
    """
    supports_native_structured_output: bool = False
    """Whether the adapter honors ``options.structured_output`` with
    ``strategy: "native"`` by constraining decoding provider-side (e.g. OpenAI
    ``json_schema`` response format). The runtime rejects ``NATIVE`` specs
    before running when the resolved provider does not set this.
    """

    @property
    def context_pressure_profile(self) -> ModelContextPressureProfile:
        return ModelContextPressureProfile()

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
        """Stream model events for one request.

        Error contract: `ModelContextOverflowError` must propagate as an
        exception (never be flattened into an error event) so runtime
        context-overflow recovery can shrink context and retry. Other failures
        should surface as `ModelStreamEvent.error(message, cause=exc)` events
        so typed classification fields survive into the runtime payload.
        """
