from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Mapping
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    copy_durable_json_value,
    copy_json_value,
    require_clean_nonblank,
    require_durable_clean_nonblank,
    require_durable_text,
    require_nonblank,
)
from cayu.artifacts.attachments import file_attachment_from_payload
from cayu.core.billing import BillingIdentity
from cayu.core.messages import (
    FilePart,
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    detach_message,
)
from cayu.providers.cache import CachePolicy, RequestCacheProjection
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationMode,
    ProviderOperationRecoveryMetadata,
)

_REQUEST_FOOTPRINT_SAFE_PROVIDER_OPTION_KEYS = frozenset(
    {
        "frequency_penalty",
        "logprobs",
        "max_completion_tokens",
        "max_output_tokens",
        "max_tokens",
        "n",
        "output_config",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning",
        "reasoning_effort",
        "seed",
        "service_tier",
        "stop",
        "stop_sequences",
        "temperature",
        "thinking",
        "tool_choice",
        "top_k",
        "top_logprobs",
        "top_p",
    }
)

_DEFAULT_FINGERPRINT_RUNTIME_OPTION_KEYS = frozenset(
    {
        "agent_metadata",
        "cache_policy",
        "cayu_file_attachments",
        "environment_metadata",
        "step",
        "structured_output",
        "thinking",
    }
)


def privacy_safe_provider_option_projection(value: object) -> dict[str, Any]:
    """Copy allow-listed provider tuning options without arbitrary metadata."""

    if type(value) is not dict:
        return {}
    return {
        key: copy_json_value(option, f"request footprint provider option {key}")
        for key, option in value.items()
        if type(key) is str
        and key in _REQUEST_FOOTPRINT_SAFE_PROVIDER_OPTION_KEYS
        and option is not None
    }


def _preflight_provider_portable_messages(
    *,
    model: str,
    messages: list[Message],
    tools: list[dict[str, Any]],
    supports_system_messages: bool,
    supports_tool_history: bool,
    supports_tool_definitions: bool,
    supports_file_attachments: bool,
    tool_name_validator: Callable[[str], None] | None = None,
    tool_definition_validator: Callable[[Mapping[str, Any]], object] | None = None,
) -> None:
    """Validate the portable request material explicitly supported by one adapter."""

    require_clean_nonblank(model, "model")
    if type(messages) is not list or any(type(message) is not Message for message in messages):
        raise TypeError("Portable messages must be a list of exact Message instances.")
    if type(tools) is not list or any(type(tool) is not dict for tool in tools):
        raise TypeError("Portable tools must be a list of exact dictionaries.")
    if type(supports_system_messages) is not bool:
        raise TypeError("supports_system_messages must be a bool.")
    if type(supports_tool_history) is not bool:
        raise TypeError("supports_tool_history must be a bool.")
    if type(supports_tool_definitions) is not bool:
        raise TypeError("supports_tool_definitions must be a bool.")
    if type(supports_file_attachments) is not bool:
        raise TypeError("supports_file_attachments must be a bool.")
    if tool_name_validator is not None and not callable(tool_name_validator):
        raise TypeError("tool_name_validator must be callable or None.")
    if tool_definition_validator is not None and not callable(tool_definition_validator):
        raise TypeError("tool_definition_validator must be callable or None.")

    for message in messages:
        if message.role is MessageRole.SYSTEM and not supports_system_messages:
            raise ValueError("Target provider does not declare system-message support.")
        for part in message.content:
            if type(part) is TextPart:
                continue
            if type(part) is ToolCallPart:
                if not supports_tool_history:
                    raise ValueError(
                        "Target provider does not declare portable tool-history support."
                    )
                if tool_name_validator is not None:
                    tool_name_validator(part.tool_name)
                continue
            if type(part) is ToolResultPart:
                if not supports_tool_history:
                    raise ValueError(
                        "Target provider does not declare portable tool-history support."
                    )
                if tool_name_validator is not None:
                    tool_name_validator(part.tool_name)
                for artifact in part.artifacts:
                    try:
                        attachment = file_attachment_from_payload(artifact)
                    except (TypeError, ValueError):
                        raise ValueError(
                            "Portable tool-result artifact claims an invalid file attachment."
                        ) from None
                    if attachment is not None and not supports_file_attachments:
                        raise ValueError(
                            "Target provider does not declare portable file-attachment support."
                        )
                continue
            if type(part) is FilePart:
                if supports_file_attachments:
                    continue
                raise ValueError(
                    "Target provider does not declare portable file-attachment support."
                )
            if type(part) in {ProviderStatePart, ThinkingPart}:
                raise ValueError(
                    "Portable target messages cannot contain provider state or thinking parts."
                )
            raise TypeError("Portable target messages contain an unsupported message part.")

    if tools and not supports_tool_definitions:
        raise ValueError("Target provider does not declare portable tool-definition support.")
    for tool in tools:
        copied_tool = copy_durable_json_value(tool, "portable tools")
        if type(copied_tool) is not dict:
            raise AssertionError("Portable tool copying changed its object shape.")
        name = copied_tool.get("name")
        if type(name) is not str:
            raise ValueError("Portable tool definitions require a string name.")
        require_clean_nonblank(name, "tool name")
        description = copied_tool.get("description", "")
        if type(description) is not str:
            raise ValueError("Portable tool definitions require a string description.")
        input_schema = copied_tool.get("input_schema", {})
        if type(input_schema) is not dict:
            raise ValueError("Portable tool definitions require an object input_schema.")
        if tool_definition_validator is not None:
            tool_definition_validator(copied_tool)


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
    ``input_tokens``; Gemini may exclude hidden thinking tokens from
    ``completion_tokens`` while including them in ``total_tokens``; OpenAI nests
    cached input in ``*_tokens_details``). A
    provider whose registered ``name`` is not one of the built-in aliases —
    Claude reached through Bedrock, a gateway, or a renamed adapter — must
    declare its dialect here so the normalizer folds cache tokens correctly
    instead of silently undercounting (and under-billing) them. ``AUTO`` (the
    default) lets the normalizer infer the dialect from the payload shape.
    """

    AUTO = "auto"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENAI = "openai"
    GENERIC = "generic"


def copy_usage_dialect(value: object, field_name: str = "usage_dialect") -> UsageDialect:
    """Validate and detach a provider's declared usage-accounting dialect.

    Provider attributes are extension-owned and remain mutable after registration.
    Accept exact built-in strings for compatibility, but never invoke methods on a
    provider-owned ``str`` subclass while establishing accounting authority.
    """

    if type(value) is UsageDialect:
        return value
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a UsageDialect or string.")
    try:
        return UsageDialect(value)
    except ValueError:
        raise ValueError(
            f"{field_name} must be one of: auto, anthropic, gemini, openai, generic."
        ) from None


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
        if status_code is not None and (
            type(status_code) is not int or status_code < 100 or status_code > 599
        ):
            raise ValueError("status_code must be a valid HTTP status code.")
        self.status_code = status_code
        self.error_type = _optional_clean_error_field(error_type, "error_type")
        self.error_code = _optional_clean_error_field(error_code, "error_code")
        self.request_id = _optional_clean_error_field(request_id, "request_id")
        if retryable is not None and type(retryable) is not bool:
            raise ValueError("retryable must be a boolean.")
        self.retryable = retryable
        if retry_after_s is not None:
            if type(retry_after_s) not in {int, float}:
                raise ValueError("retry_after_s must be a finite non-negative number.")
            try:
                retry_after_s = float(retry_after_s)
            except OverflowError:
                raise ValueError("retry_after_s must be a finite non-negative number.") from None
            if not math.isfinite(retry_after_s) or retry_after_s < 0:
                raise ValueError("retry_after_s must be a finite non-negative number.")
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


class NativeStructuredOutputSchemaInvalid(ValueError):
    """A ``strategy=NATIVE`` structured-output JSON Schema uses constructs the
    resolved provider's native mode would reject at request time.

    Raised by ``ModelProvider.preflight_native_structured_output_schema``
    before any session is created or transitioned, so the caller can fix the
    schema (the message names the offending JSON path, e.g.
    ``$/properties/address``) or retry with ``strategy="tool"`` (same JSON
    contract, provider-neutral transport). Subclasses ``ValueError`` so
    existing handlers (including the server's 4xx mapping) keep working.
    """


class InputTokenCountResult(BaseModel):
    """Provider-neutral input token count for a model request.

    Official provider counters should use `method="official"` and
    `confidence="high"`. Official remote counters can add latency and consume
    provider rate limits. Their billing behavior is provider-specific and
    should be documented in `metadata` when known. Local tokenizers and
    heuristics are useful for observability, but callers should not treat them
    as hard provider-limit guarantees.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    input_tokens: int | None = Field(default=None, ge=0, le=MAX_DURABLE_JSON_INTEGER)
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
        return InputTokenCountMethod(require_durable_clean_nonblank(value, "method"))

    @field_validator("confidence", mode="before")
    @classmethod
    def validate_confidence(cls, value: object) -> InputTokenCountConfidence:
        if isinstance(value, InputTokenCountConfidence):
            return value
        if not isinstance(value, str):
            raise ValueError("`confidence` must be a string.")
        return InputTokenCountConfidence(require_durable_clean_nonblank(value, "confidence"))

    @field_validator("components", mode="before")
    @classmethod
    def copy_components(cls, value: dict[str, Any]) -> dict[str, int]:
        copied = copy_durable_json_value(value, "components")
        if type(copied) is not dict:
            raise ValueError("`components` must be a dictionary.")
        result: dict[str, int] = {}
        for key, component_value in copied.items():
            if type(key) is not str:
                raise ValueError("Input token count component keys must be strings.")
            clean_key = require_durable_clean_nonblank(key, "component key")
            if type(component_value) is not int:
                raise ValueError("Input token count component values must be integers.")
            if component_value < 0:
                raise ValueError("Input token count component values must be non-negative.")
            result[clean_key] = component_value
        return result

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_json_value(value, "metadata")
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

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    finish_reason: ModelFinishReason
    raw_finish_reason: str | None = None
    status: str | None = None
    end_turn: bool | None = None

    @field_validator("finish_reason", mode="before")
    @classmethod
    def validate_finish_reason(cls, value: object) -> ModelFinishReason:
        if isinstance(value, ModelFinishReason):
            return value
        if not isinstance(value, str):
            raise ValueError("`finish_reason` must be a string.")
        return ModelFinishReason(require_durable_clean_nonblank(value, "finish_reason"))

    @field_validator("raw_finish_reason", "status")
    @classmethod
    def validate_optional_clean_string(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("end_turn", mode="before")
    @classmethod
    def validate_end_turn(cls, value: object) -> bool | None:
        if value is None or type(value) is bool:
            return value
        raise ValueError("`end_turn` must be a boolean or null.")


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    model: str
    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [detach_message(message) for message in value]

    @field_validator("tools", "options", mode="before")
    @classmethod
    def copy_json_request_data(cls, value, info):
        return copy_durable_json_value(value, info.field_name)

    @field_validator("model")
    @classmethod
    def validate_nonblank_model(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


def _copy_provider_operation_recovery_metadata(
    value: ProviderOperationRecoveryMetadata | dict[str, Any] | None,
) -> ProviderOperationRecoveryMetadata | None:
    if value is None:
        return None
    if type(value) is ProviderOperationRecoveryMetadata:
        value = value.model_dump(mode="python")
    return ProviderOperationRecoveryMetadata.model_validate(value)


class ModelStreamEvent(BaseModel):
    """Provider-native stream event.

    Provider adapters may expose this lower-level shape while normalizing SDK
    responses. Runtime code must convert these events into Cayu `Event`
    records before persisting, dashboarding, or forwarding them.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    type: ModelStreamEventType
    delta: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    completion: ModelCompletion | None = None
    recovery_metadata: ProviderOperationRecoveryMetadata | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("payload", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        # Provider stream events are ephemeral, untrusted observations. They
        # must remain representable long enough to retain terminal usage even
        # when unrelated provider metadata is non-portable. Runtime consumers
        # validate strictly before any transcript, event, or checkpoint write.
        return copy_json_value(value, "payload")

    @field_validator("completion")
    @classmethod
    def copy_completion(
        cls,
        value: ModelCompletion | None,
    ) -> ModelCompletion | None:
        return copy_model_completion(value)

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, value: object) -> ModelStreamEventType:
        if isinstance(value, ModelStreamEventType):
            return value
        if not isinstance(value, str):
            raise ValueError("`type` must be a string.")
        return ModelStreamEventType(require_durable_clean_nonblank(value, "type"))

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
    def text_delta(
        cls,
        delta: str,
        *,
        recovery_metadata: ProviderOperationRecoveryMetadata | dict[str, Any] | None = None,
    ) -> ModelStreamEvent:
        return cls(
            type=ModelStreamEventType.TEXT_DELTA,
            delta=delta,
            recovery_metadata=_copy_provider_operation_recovery_metadata(recovery_metadata),
        )

    @classmethod
    def thinking(
        cls,
        delta: str = "",
        *,
        provider_state: dict[str, Any] | None = None,
        recovery_metadata: ProviderOperationRecoveryMetadata | dict[str, Any] | None = None,
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
        return cls(
            type=ModelStreamEventType.THINKING,
            delta=delta,
            payload=payload,
            recovery_metadata=_copy_provider_operation_recovery_metadata(recovery_metadata),
        )

    @classmethod
    def tool_call(
        cls,
        *,
        name: str,
        arguments: dict[str, Any],
        id: str | None = None,
        recovery_metadata: ProviderOperationRecoveryMetadata | dict[str, Any] | None = None,
    ) -> ModelStreamEvent:
        if not isinstance(arguments, dict):
            raise ValueError("`arguments` must be a dictionary.")
        payload: dict[str, Any] = {
            "name": require_clean_nonblank(name, "name"),
            "arguments": copy_json_value(arguments, "arguments"),
        }
        if id is not None:
            payload["id"] = require_clean_nonblank(id, "id")
        return cls(
            type=ModelStreamEventType.TOOL_CALL,
            payload=payload,
            recovery_metadata=_copy_provider_operation_recovery_metadata(recovery_metadata),
        )

    @classmethod
    def completed(
        cls,
        payload: dict[str, Any] | None = None,
        *,
        recovery_metadata: ProviderOperationRecoveryMetadata | dict[str, Any] | None = None,
    ) -> ModelStreamEvent:
        payload = {} if payload is None else payload
        return cls(
            type=ModelStreamEventType.COMPLETED,
            payload=payload,
            recovery_metadata=_copy_provider_operation_recovery_metadata(recovery_metadata),
        )

    @classmethod
    def error(
        cls,
        message: str,
        *,
        cause: Exception | None = None,
        recovery_metadata: ProviderOperationRecoveryMetadata | dict[str, Any] | None = None,
    ) -> ModelStreamEvent:
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
            recovery_metadata=_copy_provider_operation_recovery_metadata(recovery_metadata),
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
        delta=require_durable_text(event.delta, "delta"),
        payload=copy_durable_json_value(event.payload, "payload"),
        completion=copy_model_completion(event.completion),
        recovery_metadata=(
            None
            if event.recovery_metadata is None
            else ProviderOperationRecoveryMetadata.model_validate(
                event.recovery_metadata.model_dump(mode="python")
            )
        ),
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
        end_turn=completion.end_turn,
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
        components=copy_durable_json_value(result.components, "components"),
        metadata=copy_durable_json_value(result.metadata, "metadata"),
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
        end_turn=_optional_payload_boolean(payload, "end_turn"),
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
    if raw_finish_reason in {"stop", "end_turn", "stop_sequence"}:
        return ModelFinishReason.STOP
    if raw_finish_reason in {"function_call", "tool_calls", "tool_use"}:
        return ModelFinishReason.TOOL_CALLS
    if raw_finish_reason in {
        "length",
        "max_tokens",
        "max_output_tokens",
        "model_context_window_exceeded",
    }:
        return ModelFinishReason.LENGTH
    if raw_finish_reason in {
        "content_filter",
        "content_filtered",
        "guardrail_intervened",
        "safety",
        "refusal",
    }:
        return ModelFinishReason.CONTENT_FILTER
    if raw_finish_reason in {"error", "failed", "malformed_model_output"}:
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
    return require_durable_clean_nonblank(value, key)


def _optional_payload_boolean(payload: dict[str, Any], key: str) -> bool | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if type(value) is not bool:
        raise ValueError(f"Model completed payload `{key}` must be a boolean.")
    return value


class ModelProvider(ABC):
    """Normalizes provider-specific model streams."""

    name: str
    billing_provider_name: str | None = None
    """Canonical commercial provider key, when billing differs from ``name``."""
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
    def provider_operation_mode(self) -> ProviderOperationMode:
        """Explicit dispatch mode; capability support alone never enables it."""

        return ProviderOperationMode.SYNCHRONOUS

    @property
    def provider_operations(self) -> ProviderOperationAdapter | None:
        """Optional reconnectable-operation adapter.

        Returning ``None`` preserves the ordinary synchronous ``stream`` path.
        Providers opt in explicitly by returning a composed adapter.
        """

        return None

    @property
    def context_pressure_profile(self) -> ModelContextPressureProfile:
        return ModelContextPressureProfile()

    def request_cache_policy(self, request: ModelRequest) -> CachePolicy | None:
        """Describe the effective provider-visible cache breakpoints, when known.

        Implementations must be deterministic, side-effect free, and perform no I/O.
        """

        return None

    def request_cache_projection(self, request: ModelRequest) -> RequestCacheProjection | None:
        """Project the effective cache policy and provider-selected prefix.

        Providers whose message preparation can filter or normalize content should
        override this hook and return the exact ephemeral prefix through the marker
        they applied. The default preserves compatibility with policy-only provider
        extensions and lets the runtime derive a provider-neutral prefix.
        Implementations must be deterministic, side-effect free, and perform no I/O.
        """

        policy = self.request_cache_policy(request)
        if policy is None:
            return None
        if type(policy) is not CachePolicy:
            raise TypeError("ModelProvider.request_cache_policy() must return CachePolicy or None.")
        return RequestCacheProjection(policy=policy)

    def request_footprint_options(self, request: ModelRequest) -> dict[str, Any]:
        """Project privacy-safe provider-visible options for local measurement.

        Implementations must return copied JSON, omit arbitrary metadata and
        secret-bearing extension fields, and perform no I/O.
        """

        return {}

    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        """Project complete effective provider-visible options for local analysis.

        The runtime keeps this copied value ephemeral: it supplies keyed HMAC input
        when configured and identifies which non-allowlisted option categories are
        active, but the value and its field names are never persisted. Built-in
        adapters override this hook so ignored namespaces, defaults, and normalized
        controls follow their payload-building semantics. The default conservatively
        treats every non-runtime request option as provider-visible.
        Implementations must be deterministic, side-effect free, and perform no I/O.
        """

        return {
            key: copy_json_value(value, f"request fingerprint option {key}")
            for key, value in request.options.items()
            if key not in _DEFAULT_FINGERPRINT_RUNTIME_OPTION_KEYS and value is not None
        }

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        """Reject portable messages or active tools this adapter cannot render.

        The runtime calls this side-effect-free hook before durably adopting a
        different provider/model target. Adapters must explicitly override this
        method to admit system messages, tool history, active tool definitions, or
        file attachments. The conservative default accepts only user/assistant text
        with no active tools, so an existing custom provider cannot silently claim
        support for request material its renderer may reject.
        """

        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=False,
            supports_tool_history=False,
            supports_tool_definitions=False,
            supports_file_attachments=False,
        )

    async def billing_identity_for_request(
        self,
        request: ModelRequest,
    ) -> BillingIdentity | None:
        """Resolve optional request context needed for commercial accounting."""

        return None

    def billing_identity_for_completion(
        self,
        identity: BillingIdentity | None,
        payload: dict[str, Any],
    ) -> BillingIdentity | None:
        """Merge provider-reported completion facts into a request identity."""

        return identity

    def preflight_native_structured_output_schema(self, json_schema: dict[str, Any]) -> None:
        """Optionally reject a NATIVE structured-output schema this adapter's
        provider API would refuse, before any model request is made.

        The runtime calls this at every entrance (after the
        ``supports_native_structured_output`` check, before any session is
        created or transitioned) with a caller-owned copy of the schema;
        implementations must not mutate it. Adapters should raise
        ``NativeStructuredOutputSchemaInvalid`` with a path-specific message.
        The default accepts everything: adapters without provider-specific
        schema rules stay source-compatible.
        """

        return None

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
