from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_json_value,
    require_durable_clean_nonblank,
    require_execution_unit_id,
)
from cayu.artifacts.attachments import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    FileAttachmentKind,
    ResolvedFileAttachment,
    file_attachment_from_payload,
    resolved_file_attachments_from_options,
)
from cayu.core.messages import FilePart, Message, MessageRole, ToolCallPart, ToolResultPart
from cayu.providers.base import (
    ModelContextPressureProfile,
    ModelProvider,
    ModelRequest,
    copy_model_context_pressure_profile,
)
from cayu.providers.cache import CacheBreakpoint, CachePolicy, RequestCacheProjection
from cayu.runtime.context import (
    ContextPressureEstimate,
    ObservedDeltaContextEstimator,
    estimate_model_request_context_pressure,
)
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME

REQUEST_FOOTPRINT_SCHEMA_VERSION = 2
PROMPT_CONTRIBUTION_MANIFEST_SCHEMA_VERSION = 1
REQUEST_FOOTPRINT_CANONICALIZATION_VERSION = 1
_HMAC_CONTEXT = b"cayu.request-footprint"
_KEY_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")
_HMAC_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_ONLY_OPTION_KEYS = frozenset(
    {
        "agent_metadata",
        "environment_metadata",
        "step",
        "structured_output",
        "thinking",
        RESOLVED_FILE_ATTACHMENTS_OPTION,
    }
)
_KNOWN_PROVIDER_OPTION_CATEGORIES = frozenset(
    {
        "cache_policy",
        "max_output_tokens",
        "max_tokens",
        "parallel_tool_calls",
        "reasoning_effort",
        "response_format",
        "stop",
        "structured_output",
        "temperature",
        "tool_choice",
        "top_k",
        "top_p",
    }
)


@dataclass(frozen=True, slots=True)
class _RequestMeasurementProjection:
    visible_options: dict[str, Any]
    measured_options: dict[str, Any]
    native_structured_output: dict[str, Any] | None
    known_categories: tuple[str, ...]
    unknown_count: int


class RequestVariant(StrEnum):
    INITIAL = "initial"
    STRUCTURED_OUTPUT_REPAIR = "structured_output_repair"
    CONTEXT_OVERFLOW_RECOVERY = "context_overflow_recovery"
    CONTEXT_COMPACTION = "context_compaction"


class RequestFingerprintAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class PromptContributionKind(StrEnum):
    AGENT_INSTRUCTIONS = "agent_instructions"
    WORKSPACE_INSTRUCTIONS = "workspace_instructions"
    CAYU_FRAMING = "cayu_framing"


class PromptContributionAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class RequestFootprintConfig(BaseModel):
    """Controls local privacy-safe request-footprint observation.

    Footprints are local and content-free, so they are enabled by default. Keyed
    request identities remain disabled until both a key ID and secret key are
    supplied. This configuration never enables provider-backed token counting.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    enabled: StrictBool = True
    fingerprint_key_id: str | None = None
    fingerprint_key: SecretStr | None = Field(default=None, repr=False)

    @field_validator("fingerprint_key_id")
    @classmethod
    def validate_fingerprint_key_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_durable_clean_nonblank(value, "fingerprint_key_id")
        if _KEY_ID_PATTERN.fullmatch(value) is None:
            raise ValueError(
                "fingerprint_key_id must be 1-64 lowercase ASCII letters, digits, dots, "
                "underscores, or hyphens and must start and end with a letter or digit."
            )
        return value

    @model_validator(mode="after")
    def validate_fingerprint_key_pair(self) -> RequestFootprintConfig:
        if (self.fingerprint_key_id is None) != (self.fingerprint_key is None):
            raise ValueError("fingerprint_key_id and fingerprint_key must be configured together.")
        if self.fingerprint_key is not None:
            key_bytes = self.fingerprint_key.get_secret_value().encode("utf-8")
            if len(key_bytes) < 32:
                raise ValueError("fingerprint_key must contain at least 32 bytes.")
        return self


def copy_request_footprint_config(
    config: RequestFootprintConfig | None,
) -> RequestFootprintConfig:
    if config is None:
        return RequestFootprintConfig()
    if type(config) is not RequestFootprintConfig:
        raise TypeError("Request footprint config must be a RequestFootprintConfig instance.")
    key = config.fingerprint_key
    return RequestFootprintConfig(
        enabled=config.enabled,
        fingerprint_key_id=config.fingerprint_key_id,
        fingerprint_key=(None if key is None else SecretStr(key.get_secret_value())),
    )


class RequestSize(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    characters: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    utf8_bytes: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    canonical_json_bytes: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)


class RequestComponentFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    size: RequestSize


class RequestContentGroupFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: MessageRole
    part_type: str
    count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    size: RequestSize

    @field_validator("part_type")
    @classmethod
    def validate_part_type(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "part_type")


class RequestMessagesFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    system: RequestComponentFootprint
    groups: tuple[RequestContentGroupFootprint, ...] = ()
    size: RequestSize


class RequestAttachmentGroupFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: FileAttachmentKind
    count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_bytes: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)


class RequestAttachmentsFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_bytes: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    groups: tuple[RequestAttachmentGroupFootprint, ...] = ()


class RequestOptionsFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    known_categories: tuple[str, ...] = ()
    unknown_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    size: RequestSize

    @field_validator("known_categories")
    @classmethod
    def validate_known_categories(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("known_categories must be unique and sorted.")
        return value


class RequestComponentTokenEstimates(BaseModel):
    """Labeled local token estimates for the final request's major components.

    Structured-output contribution can overlap the system/tool/options component
    in which the provider-neutral request carries it. These fields are evidence
    about component pressure, not additive billing facts.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    confidence: str
    total_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    system_message_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    non_system_message_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    tool_schema_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    structured_output_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    attachment_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    request_options_input_tokens: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("method", "confidence")
    @classmethod
    def validate_labels(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)


class RequestFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    availability: RequestFingerprintAvailability
    value: str | None = None
    algorithm: Literal["hmac-sha256"] | None = None
    key_id: str | None = None
    canonicalization_version: Literal[1]
    unavailable_reason: str | None = None

    @field_validator("canonicalization_version", mode="before")
    @classmethod
    def validate_canonicalization_version(cls, value: object) -> object:
        if type(value) is not int or value != REQUEST_FOOTPRINT_CANONICALIZATION_VERSION:
            raise ValueError("Request fingerprint canonicalization_version must be the integer 1.")
        return value

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str | None) -> str | None:
        if value is not None and _HMAC_SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("Request fingerprint values must be lowercase HMAC-SHA-256 hex.")
        return value

    @field_validator("key_id")
    @classmethod
    def validate_key_id(cls, value: str | None) -> str | None:
        if value is not None and _KEY_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("Request fingerprint key IDs must use the configured key-ID format.")
        return value

    @field_validator("unavailable_reason")
    @classmethod
    def validate_unavailable_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "unavailable_reason")

    @model_validator(mode="after")
    def validate_availability(self) -> RequestFingerprint:
        if self.availability == RequestFingerprintAvailability.AVAILABLE:
            if self.value is None or self.algorithm is None or self.key_id is None:
                raise ValueError(
                    "Available request fingerprints require value, algorithm, and key_id."
                )
            if self.unavailable_reason is not None:
                raise ValueError(
                    "Available request fingerprints cannot have an unavailable reason."
                )
        else:
            if self.value is not None or self.algorithm is not None or self.key_id is not None:
                raise ValueError("Unavailable request fingerprints cannot carry identity material.")
            if self.unavailable_reason is None:
                raise ValueError("Unavailable request fingerprints require a reason.")
        return self


class RequestFingerprintSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_neutral_request: RequestFingerprint
    provider_wire_request: RequestFingerprint
    system: RequestFingerprint
    tool_manifest: RequestFingerprint
    conversation_prefix: RequestFingerprint


class RequestCacheBreakpointFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: CacheBreakpoint
    ttl: Literal["standard", "extended"]
    fingerprint: RequestFingerprint


class PromptContributionFootprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: PromptContributionKind
    size: RequestSize
    fingerprint: RequestFingerprint


class PromptContributionManifest(BaseModel):
    """Content-free creation-time evidence for the rendered initial system prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    system: RequestComponentFootprint
    system_fingerprint: RequestFingerprint
    contributions: tuple[PromptContributionFootprint, ...]

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != PROMPT_CONTRIBUTION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Prompt contribution schema_version must be the integer 1.")
        return value

    @field_validator("contributions")
    @classmethod
    def validate_contributions(
        cls,
        value: tuple[PromptContributionFootprint, ...],
    ) -> tuple[PromptContributionFootprint, ...]:
        kinds = tuple(item.kind for item in value)
        if tuple(sorted(set(kinds), key=lambda item: item.value)) != kinds:
            raise ValueError("Prompt contributions must have unique kinds in sorted order.")
        return value


class RequestPromptContributionAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    availability: PromptContributionAvailability
    contributions: tuple[PromptContributionFootprint, ...] = ()
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> RequestPromptContributionAttribution:
        if self.availability == PromptContributionAvailability.AVAILABLE:
            if not self.contributions:
                raise ValueError("Available prompt attribution requires contributions.")
            if self.unavailable_reason is not None:
                raise ValueError("Available prompt attribution cannot have an unavailable reason.")
        else:
            if self.contributions:
                raise ValueError("Unavailable prompt attribution cannot expose contributions.")
            if self.unavailable_reason is None:
                raise ValueError("Unavailable prompt attribution requires a reason.")
        return self


class RequestFootprint(BaseModel):
    """Versioned, content-free evidence about one prepared provider request."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1, 2]
    execution_profile_fingerprint: str | None = None
    observation_id: str
    provider_name: str
    model: str
    step: StrictInt | None = Field(default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    attempt: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    max_attempts: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    request_variant: RequestVariant
    model_step_id: str
    model_attempt_id: str
    operation_id: str | None = None
    attempt_id: str | None = None
    total: RequestComponentFootprint
    messages: RequestMessagesFootprint
    tools: RequestComponentFootprint
    attachments: RequestAttachmentsFootprint
    options: RequestOptionsFootprint
    structured_output: RequestComponentFootprint
    context_pressure: ContextPressureEstimate
    component_tokens: RequestComponentTokenEstimates
    prompt_contributions: RequestPromptContributionAttribution
    fingerprints: RequestFingerprintSet
    cache_breakpoints: tuple[RequestCacheBreakpointFootprint, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value not in (1, REQUEST_FOOTPRINT_SCHEMA_VERSION):
            raise ValueError("Request footprint schema_version must be integer 1 or 2.")
        return value

    @field_validator("execution_profile_fingerprint")
    @classmethod
    def validate_execution_profile_fingerprint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("execution_profile_fingerprint must be a lowercase SHA-256 digest.")
        return value

    @field_validator("observation_id", "provider_name", "model")
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("operation_id", "attempt_id")
    @classmethod
    def validate_optional_causal_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("model_step_id", "model_attempt_id")
    @classmethod
    def validate_execution_identity(cls, value: str, info) -> str:
        try:
            return require_execution_unit_id(value, info.field_name)
        except ValueError:
            # Public runtime events replace private execution-unit identities with
            # field-scoped linkage aliases. Keep the exported footprint model usable
            # for both durable private records and the public event stream.
            from cayu.runtime._event_projection import (
                PRIVATE_EVENT_AUTHORITY,
                public_event_linkage_sequence,
            )

            if (
                value == PRIVATE_EVENT_AUTHORITY
                or public_event_linkage_sequence(
                    value,
                    field_name=info.field_name,
                )
                is not None
            ):
                return value
            raise

    @model_validator(mode="after")
    def validate_attempt(self) -> RequestFootprint:
        if self.schema_version == 1 and self.execution_profile_fingerprint is not None:
            raise ValueError("Request footprint schema v1 cannot carry an execution profile.")
        if self.schema_version >= 2 and self.execution_profile_fingerprint is None:
            raise ValueError("Request footprint schema v2 requires an execution profile.")
        if self.attempt > self.max_attempts:
            raise ValueError("attempt cannot exceed max_attempts.")
        if (self.operation_id is None) != (self.attempt_id is None):
            raise ValueError("operation_id and attempt_id must be configured together.")
        if self.step is None and (
            self.request_variant != RequestVariant.CONTEXT_COMPACTION or self.operation_id is None
        ):
            raise ValueError("Only operation-linked context compaction footprints may omit step.")
        breakpoint_kinds = tuple(item.kind for item in self.cache_breakpoints)
        canonical_breakpoint_kinds = tuple(
            sorted(set(breakpoint_kinds), key=lambda item: item.value)
        )
        if breakpoint_kinds != canonical_breakpoint_kinds:
            raise ValueError("cache_breakpoints must be unique and canonically ordered.")
        return self


def _request_cache_projection(
    model_request: ModelRequest,
    *,
    provider: ModelProvider,
) -> RequestCacheProjection | None:
    projection = provider.request_cache_projection(model_request.model_copy(deep=True))
    if projection is not None and type(projection) is not RequestCacheProjection:
        raise TypeError(
            "ModelProvider.request_cache_projection() must return RequestCacheProjection or None."
        )
    return projection


def analyze_request_footprint(
    model_request: ModelRequest,
    *,
    provider: ModelProvider,
    provider_name: str,
    step: int | None,
    attempt: int,
    max_attempts: int,
    request_variant: RequestVariant | str,
    observation_id: str,
    model_step_id: str,
    model_attempt_id: str,
    config: RequestFootprintConfig | None = None,
    prompt_contribution_manifest: PromptContributionManifest | None = None,
    structured_output_instruction: str | None = None,
    operation_id: str | None = None,
    operation_attempt_id: str | None = None,
    execution_profile_fingerprint: str | None = None,
) -> RequestFootprint:
    """Analyze one detached request with the provider's effective cache policy."""

    if not isinstance(provider, ModelProvider):
        raise TypeError("provider must be a ModelProvider.")
    resolved_config = copy_request_footprint_config(config)
    detached_request = model_request.model_copy(deep=True)
    cache_projection = _request_cache_projection(detached_request, provider=provider)
    cache_policy = None if cache_projection is None else cache_projection.policy
    measured_provider_options = provider.request_footprint_options(
        detached_request.model_copy(deep=True)
    )
    if type(measured_provider_options) is not dict:
        raise TypeError("ModelProvider.request_footprint_options() must return a dict.")
    fingerprint_provider_options = provider.request_fingerprint_options(
        detached_request.model_copy(deep=True)
    )
    if type(fingerprint_provider_options) is not dict:
        raise TypeError("ModelProvider.request_fingerprint_options() must return a dict.")
    return build_request_footprint(
        detached_request,
        provider_name=provider_name,
        step=step,
        attempt=attempt,
        max_attempts=max_attempts,
        request_variant=request_variant,
        observation_id=observation_id,
        model_step_id=model_step_id,
        model_attempt_id=model_attempt_id,
        config=resolved_config,
        context_pressure_profile=provider.context_pressure_profile,
        cache_policy=cache_policy,
        provider_conversation_prefix=(
            None if cache_projection is None else cache_projection.conversation_prefix
        ),
        measured_provider_options=measured_provider_options,
        fingerprint_provider_options=fingerprint_provider_options,
        prompt_contribution_manifest=prompt_contribution_manifest,
        structured_output_instruction=structured_output_instruction,
        operation_id=operation_id,
        operation_attempt_id=operation_attempt_id,
        execution_profile_fingerprint=execution_profile_fingerprint,
    )


def analyze_request_context_pressure(
    model_request: ModelRequest,
    *,
    provider: ModelProvider,
) -> ContextPressureEstimate:
    """Measure one prepared request through the canonical provider-aware projection."""

    if type(model_request) is not ModelRequest:
        raise TypeError("model_request must be a ModelRequest.")
    if not isinstance(provider, ModelProvider):
        raise TypeError("provider must be a ModelProvider.")
    detached_request = model_request.model_copy(deep=True)
    cache_projection = _request_cache_projection(detached_request, provider=provider)
    cache_policy = None if cache_projection is None else cache_projection.policy
    measured_provider_options = provider.request_footprint_options(
        detached_request.model_copy(deep=True)
    )
    if type(measured_provider_options) is not dict:
        raise TypeError("ModelProvider.request_footprint_options() must return a dict.")
    projection = _request_measurement_projection(
        detached_request,
        measured_provider_options=measured_provider_options,
    )
    measured_options = _measured_request_options(
        projection.measured_options,
        cache_markers=_provider_neutral_cache_marker_payload(cache_policy),
    )
    return _estimate_projected_context_pressure(
        model_request=detached_request,
        messages=detached_request.messages,
        measured_options=measured_options,
        native_structured_output=projection.native_structured_output,
        profile=copy_model_context_pressure_profile(provider.context_pressure_profile),
        estimator=ObservedDeltaContextEstimator(),
    )


def build_request_footprint(
    model_request: ModelRequest,
    *,
    provider_name: str,
    step: int | None,
    attempt: int,
    max_attempts: int,
    request_variant: RequestVariant | str,
    observation_id: str,
    model_step_id: str,
    model_attempt_id: str,
    config: RequestFootprintConfig | None = None,
    context_pressure_profile: ModelContextPressureProfile | None = None,
    cache_policy: CachePolicy | None = None,
    provider_conversation_prefix: tuple[dict[str, Any], ...] | None = None,
    measured_provider_options: dict[str, Any] | None = None,
    fingerprint_provider_options: dict[str, Any] | None = None,
    prompt_contribution_manifest: PromptContributionManifest | None = None,
    structured_output_instruction: str | None = None,
    operation_id: str | None = None,
    operation_attempt_id: str | None = None,
    execution_profile_fingerprint: str | None = None,
) -> RequestFootprint:
    """Analyze one final provider-neutral request without retaining its content."""

    if type(model_request) is not ModelRequest:
        raise TypeError("model_request must be a ModelRequest.")
    resolved_config = copy_request_footprint_config(config)
    profile = copy_model_context_pressure_profile(context_pressure_profile)
    request_variant = RequestVariant(request_variant)
    provider_name = require_durable_clean_nonblank(provider_name, "provider_name")
    observation_id = require_durable_clean_nonblank(observation_id, "observation_id")
    model_step_id = require_execution_unit_id(model_step_id, "model_step_id")
    model_attempt_id = require_execution_unit_id(model_attempt_id, "model_attempt_id")
    if step is not None:
        _require_positive_int(step, "step")
    _require_positive_int(attempt, "attempt")
    _require_positive_int(max_attempts, "max_attempts")
    if attempt > max_attempts:
        raise ValueError("attempt cannot exceed max_attempts.")
    if cache_policy is not None and type(cache_policy) is not CachePolicy:
        raise TypeError("cache_policy must be a CachePolicy or None.")
    if provider_conversation_prefix is not None:
        if type(provider_conversation_prefix) is not tuple:
            raise TypeError("provider_conversation_prefix must be a tuple or None.")
        if not provider_conversation_prefix or any(
            type(message) is not dict for message in provider_conversation_prefix
        ):
            raise ValueError("provider_conversation_prefix must contain one or more JSON objects.")
        if (
            cache_policy is None
            or CacheBreakpoint.CONVERSATION_PREFIX not in cache_policy.breakpoints
        ):
            raise ValueError(
                "provider_conversation_prefix requires an effective conversation-prefix breakpoint."
            )
    if measured_provider_options is not None and type(measured_provider_options) is not dict:
        raise TypeError("measured_provider_options must be a dict or None.")
    if fingerprint_provider_options is not None and type(fingerprint_provider_options) is not dict:
        raise TypeError("fingerprint_provider_options must be a dict or None.")
    if (
        prompt_contribution_manifest is not None
        and type(prompt_contribution_manifest) is not PromptContributionManifest
    ):
        raise TypeError(
            "prompt_contribution_manifest must be a PromptContributionManifest or None."
        )
    if structured_output_instruction is not None and type(structured_output_instruction) is not str:
        raise TypeError("structured_output_instruction must be a string or None.")
    if execution_profile_fingerprint is not None and (
        type(execution_profile_fingerprint) is not str
        or len(execution_profile_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in execution_profile_fingerprint)
    ):
        raise ValueError("execution_profile_fingerprint must be a lowercase SHA-256 digest.")

    resolved_attachments = resolved_file_attachments_from_options(model_request.options)
    attachment_occurrences = _attachment_occurrences(
        model_request.messages,
        resolved_attachments=resolved_attachments,
    )
    message_payloads = _provider_neutral_message_payloads(
        model_request.messages,
        resolved_attachments=resolved_attachments,
    )
    system_payloads = [
        payload
        for message, payload in zip(model_request.messages, message_payloads, strict=True)
        if message.role == MessageRole.SYSTEM
    ]
    grouped_parts: dict[tuple[MessageRole, str], list[dict[str, Any]]] = defaultdict(list)
    for message, message_payload in zip(
        model_request.messages,
        message_payloads,
        strict=True,
    ):
        if message.role == MessageRole.SYSTEM:
            continue
        content_payloads = message_payload["content"]
        for part, part_payload in zip(message.content, content_payloads, strict=True):
            grouped_parts[(message.role, part.type)].append(part_payload)
    message_groups = tuple(
        RequestContentGroupFootprint(
            role=role,
            part_type=part_type,
            count=len(payloads),
            size=_request_size(payloads),
        )
        for (role, part_type), payloads in sorted(
            grouped_parts.items(),
            key=lambda item: (item[0][0].value, item[0][1]),
        )
    )
    messages = RequestMessagesFootprint(
        count=len(model_request.messages),
        system=RequestComponentFootprint(
            count=len(system_payloads),
            size=_request_size(system_payloads),
        ),
        groups=message_groups,
        size=_request_size(message_payloads),
    )
    tool_manifest, hosted_tool_payloads = _request_tool_manifest(model_request)
    tools = RequestComponentFootprint(
        count=len(model_request.tools) + len(hosted_tool_payloads),
        size=_request_size(tool_manifest),
    )
    attachments = _attachment_footprint(attachment_occurrences)
    measurement_projection = _request_measurement_projection(
        model_request,
        measured_provider_options=measured_provider_options,
        effective_provider_options=fingerprint_provider_options,
    )
    visible_options = measurement_projection.visible_options
    cache_marker_payload = _provider_neutral_cache_marker_payload(cache_policy)
    measured_options = _measured_request_options(
        measurement_projection.measured_options,
        cache_markers=cache_marker_payload,
    )
    known_option_categories = set(measurement_projection.known_categories)
    if cache_marker_payload is not None:
        known_option_categories.add("cache_policy")
    structured_output_option = measurement_projection.native_structured_output
    effective_fingerprint_options = (
        None
        if fingerprint_provider_options is None
        else copy_json_value(fingerprint_provider_options, "fingerprint_provider_options")
    )
    fingerprint_options = _provider_neutral_fingerprint_options(
        visible_options=visible_options,
        fingerprint_provider_options=effective_fingerprint_options,
    )
    options = RequestOptionsFootprint(
        known_categories=tuple(sorted(known_option_categories)),
        unknown_count=measurement_projection.unknown_count,
        size=_request_size(measured_options),
    )
    structured_output_tools = [
        tool for tool in model_request.tools if tool.get("name") == STRUCTURED_OUTPUT_TOOL_NAME
    ]
    structured_output_payload = (
        structured_output_option
        if structured_output_option is not None
        else {
            "tools": structured_output_tools,
            "instruction": structured_output_instruction,
        }
    )
    has_structured_output = structured_output_option is not None or bool(
        structured_output_tools or structured_output_instruction
    )
    structured_output_footprint = RequestComponentFootprint(
        count=1 if has_structured_output else 0,
        size=_request_size(structured_output_payload if has_structured_output else {}),
    )
    conversation_prefix = _conversation_prefix_payload(
        model_request,
        message_payloads=message_payloads,
        cache_policy=cache_policy,
        provider_conversation_prefix=provider_conversation_prefix,
    )
    measured_request_shape = {
        "model": model_request.model,
        "messages": message_payloads,
        "tools": model_request.tools,
        "options": measured_options,
    }
    if hosted_tool_payloads:
        measured_request_shape["hosted_tools"] = hosted_tool_payloads
    fingerprint_request_shape = {
        **measured_request_shape,
        "options": fingerprint_options,
    }
    if structured_output_option is not None:
        measured_request_shape["structured_output"] = structured_output_option
        fingerprint_request_shape["structured_output"] = structured_output_option
    cache_policy_payload = _provider_neutral_cache_policy_payload(
        cache_policy=cache_policy,
        conversation_prefix=conversation_prefix,
    )
    if cache_policy_payload is not None:
        fingerprint_request_shape["cache_policy"] = cache_policy_payload
    estimator = ObservedDeltaContextEstimator()
    context_pressure = _estimate_projected_context_pressure(
        model_request=model_request,
        messages=model_request.messages,
        measured_options=measured_options,
        native_structured_output=structured_output_option,
        profile=profile,
        estimator=estimator,
    )
    structured_output_tokens = _structured_output_tokens(
        estimator=estimator,
        native_structured_output_option=structured_output_option,
        structured_output_tools=structured_output_tools,
        structured_output_instruction=structured_output_instruction,
        tool_schema_chars_per_token=profile.tool_schema_chars_per_token,
        json_chars_per_token=context_pressure.json_chars_per_token,
    )
    request_options_tokens = context_pressure.estimated_request_options_input_tokens
    system_message_tokens = sum(
        estimator.estimate_message_tokens(message)
        for message in model_request.messages
        if message.role == MessageRole.SYSTEM
    )
    component_tokens = RequestComponentTokenEstimates(
        method=context_pressure.method,
        confidence=context_pressure.confidence,
        total_input_tokens=context_pressure.estimated_context_input_tokens,
        system_message_input_tokens=system_message_tokens,
        non_system_message_input_tokens=max(
            0,
            context_pressure.estimated_message_input_tokens - system_message_tokens,
        ),
        tool_schema_input_tokens=context_pressure.estimated_tool_schema_input_tokens,
        structured_output_input_tokens=structured_output_tokens,
        attachment_input_tokens=context_pressure.estimated_attachment_input_tokens,
        request_options_input_tokens=request_options_tokens,
    )
    fingerprints = RequestFingerprintSet(
        provider_neutral_request=_fingerprint(
            fingerprint_request_shape,
            scope="provider-neutral-request",
            config=resolved_config,
        ),
        provider_wire_request=_unavailable_fingerprint("provider_wire_not_observed"),
        system=_fingerprint(
            system_payloads,
            scope="system",
            config=resolved_config,
            unavailable_reason="system_not_present" if not system_payloads else None,
        ),
        tool_manifest=_fingerprint(
            tool_manifest,
            scope="tool-manifest",
            config=resolved_config,
            unavailable_reason=(
                "tools_not_present"
                if not model_request.tools and not hosted_tool_payloads
                else None
            ),
        ),
        conversation_prefix=_fingerprint(
            conversation_prefix,
            scope="conversation-prefix",
            config=resolved_config,
            unavailable_reason=(
                "cache_conversation_prefix_unavailable" if conversation_prefix is None else None
            ),
        ),
    )
    return RequestFootprint(
        schema_version=(
            REQUEST_FOOTPRINT_SCHEMA_VERSION if execution_profile_fingerprint is not None else 1
        ),
        execution_profile_fingerprint=execution_profile_fingerprint,
        observation_id=observation_id,
        provider_name=provider_name,
        model=model_request.model,
        step=step,
        attempt=attempt,
        max_attempts=max_attempts,
        request_variant=request_variant,
        model_step_id=model_step_id,
        model_attempt_id=model_attempt_id,
        operation_id=operation_id,
        attempt_id=operation_attempt_id,
        total=RequestComponentFootprint(count=1, size=_request_size(measured_request_shape)),
        messages=messages,
        tools=tools,
        attachments=attachments,
        options=options,
        structured_output=structured_output_footprint,
        context_pressure=context_pressure,
        component_tokens=component_tokens,
        prompt_contributions=_prompt_contribution_attribution(
            manifest=prompt_contribution_manifest,
            final_system_fingerprint=fingerprints.system,
        ),
        fingerprints=fingerprints,
        cache_breakpoints=_cache_breakpoint_footprints(
            model_request=model_request,
            system_payloads=system_payloads,
            conversation_prefix=conversation_prefix,
            cache_policy=cache_policy,
            config=resolved_config,
        ),
    )


def build_prompt_contribution_manifest(
    *,
    rendered_system_prompt: str | None,
    contributions: dict[PromptContributionKind | str, tuple[str, ...]],
    config: RequestFootprintConfig | None = None,
) -> PromptContributionManifest | None:
    """Build content-free creation evidence from exact rendered prompt fragments."""

    if rendered_system_prompt is None:
        return None
    if type(rendered_system_prompt) is not str or not rendered_system_prompt:
        raise ValueError("rendered_system_prompt must be a non-empty string or None.")
    if type(contributions) is not dict:
        raise TypeError("contributions must be a dict.")
    resolved_config = copy_request_footprint_config(config)
    normalized: dict[PromptContributionKind, tuple[str, ...]] = {}
    for raw_kind, raw_fragments in contributions.items():
        kind = PromptContributionKind(raw_kind)
        if type(raw_fragments) is not tuple or any(type(item) is not str for item in raw_fragments):
            raise TypeError("Prompt contribution fragments must be tuples of strings.")
        if not raw_fragments or not any(raw_fragments):
            continue
        normalized[kind] = raw_fragments
    if not normalized:
        raise ValueError("A rendered system prompt requires at least one contribution.")
    system_payloads = [Message.text("system", rendered_system_prompt).model_dump(mode="json")]
    return PromptContributionManifest(
        schema_version=PROMPT_CONTRIBUTION_MANIFEST_SCHEMA_VERSION,
        system=RequestComponentFootprint(count=1, size=_request_size(system_payloads)),
        system_fingerprint=_fingerprint(
            system_payloads,
            scope="system",
            config=resolved_config,
        ),
        contributions=tuple(
            PromptContributionFootprint(
                kind=kind,
                size=_request_size(fragments),
                fingerprint=_fingerprint(
                    list(fragments),
                    scope=f"prompt-contribution:{kind.value}",
                    config=resolved_config,
                ),
            )
            for kind, fragments in sorted(normalized.items(), key=lambda item: item[0].value)
        ),
    )


def _prompt_contribution_attribution(
    *,
    manifest: PromptContributionManifest | None,
    final_system_fingerprint: RequestFingerprint,
) -> RequestPromptContributionAttribution:
    if manifest is None:
        return RequestPromptContributionAttribution(
            availability=PromptContributionAvailability.UNAVAILABLE,
            unavailable_reason="creation_manifest_unavailable",
        )
    creation_fingerprint = manifest.system_fingerprint
    if (
        creation_fingerprint.availability != RequestFingerprintAvailability.AVAILABLE
        or final_system_fingerprint.availability != RequestFingerprintAvailability.AVAILABLE
    ):
        return RequestPromptContributionAttribution(
            availability=PromptContributionAvailability.UNAVAILABLE,
            unavailable_reason="system_identity_unavailable",
        )
    if (
        creation_fingerprint.algorithm != final_system_fingerprint.algorithm
        or creation_fingerprint.key_id != final_system_fingerprint.key_id
        or creation_fingerprint.canonicalization_version
        != final_system_fingerprint.canonicalization_version
    ):
        return RequestPromptContributionAttribution(
            availability=PromptContributionAvailability.UNAVAILABLE,
            unavailable_reason="system_identity_not_comparable",
        )
    if not hmac.compare_digest(
        creation_fingerprint.value or "",
        final_system_fingerprint.value or "",
    ):
        return RequestPromptContributionAttribution(
            availability=PromptContributionAvailability.UNAVAILABLE,
            unavailable_reason="final_system_changed",
        )
    return RequestPromptContributionAttribution(
        availability=PromptContributionAvailability.AVAILABLE,
        contributions=manifest.contributions,
    )


def _request_size(value: Any) -> RequestSize:
    encoded = _canonical_json(value)
    characters, utf8_bytes = _string_value_sizes(value)
    return RequestSize(
        characters=characters,
        utf8_bytes=utf8_bytes,
        canonical_json_bytes=len(encoded),
    )


def _string_value_sizes(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        return len(value), len(value.encode("utf-8"))
    if isinstance(value, dict):
        characters = 0
        utf8_bytes = 0
        for key, item in value.items():
            key_characters, key_bytes = _string_value_sizes(key)
            item_characters, item_bytes = _string_value_sizes(item)
            characters += key_characters + item_characters
            utf8_bytes += key_bytes + item_bytes
        return characters, utf8_bytes
    if isinstance(value, list | tuple):
        characters = 0
        utf8_bytes = 0
        for item in value:
            item_characters, item_bytes = _string_value_sizes(item)
            characters += item_characters
            utf8_bytes += item_bytes
        return characters, utf8_bytes
    return 0, 0


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _attachment_occurrences(
    messages: list[Message],
    *,
    resolved_attachments: dict[str, dict[str, Any]],
) -> tuple[ResolvedFileAttachment, ...]:
    occurrences: list[ResolvedFileAttachment] = []
    for message in messages:
        for part in message.content:
            if type(part) is FilePart:
                payloads = (part.attachment,)
            elif type(part) is ToolResultPart:
                payloads = tuple(part.artifacts)
            else:
                continue
            for payload in payloads:
                reference = file_attachment_from_payload(payload)
                if reference is None:
                    continue
                resolved = resolved_attachments.get(reference.artifact_id)
                if resolved is None:
                    raise ValueError(f"Missing resolved file attachment: {reference.artifact_id}")
                occurrences.append(ResolvedFileAttachment.model_validate(resolved))
    return tuple(occurrences)


def _attachment_footprint(
    occurrences: tuple[ResolvedFileAttachment, ...],
) -> RequestAttachmentsFootprint:
    grouped: dict[FileAttachmentKind, list[int]] = defaultdict(list)
    for attachment in occurrences:
        try:
            source_bytes = len(base64.b64decode(attachment.data_base64, validate=True))
        except ValueError as exc:  # pragma: no cover - runtime construction validates this path.
            raise ValueError("Resolved file attachment contains invalid base64 data.") from exc
        grouped[attachment.kind].append(source_bytes)
    groups = tuple(
        RequestAttachmentGroupFootprint(
            kind=kind,
            count=len(sizes),
            source_bytes=sum(sizes),
        )
        for kind, sizes in sorted(grouped.items(), key=lambda item: item[0].value)
    )
    return RequestAttachmentsFootprint(
        count=sum(group.count for group in groups),
        source_bytes=sum(group.source_bytes for group in groups),
        groups=groups,
    )


def _provider_neutral_attachment_payload(
    attachment: ResolvedFileAttachment,
) -> dict[str, Any]:
    return {
        "kind": attachment.kind.value,
        "filename": attachment.filename,
        "content_type": attachment.content_type,
        "data_base64": attachment.data_base64,
    }


def _provider_neutral_message_payloads(
    messages: list[Message],
    *,
    resolved_attachments: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in messages:
        message_payload = message.model_dump(mode="json")
        content_payloads = message_payload.get("content")
        if type(content_payloads) is not list or len(content_payloads) != len(message.content):
            raise ValueError("Model request message content could not be canonicalized.")
        for index, part in enumerate(message.content):
            part_payload = content_payloads[index]
            if type(part_payload) is not dict:
                raise ValueError("Model request message part could not be canonicalized.")
            if type(part) in {ToolCallPart, ToolResultPart}:
                for key in ("tool_round_id", "model_step_id", "model_attempt_id"):
                    part_payload.pop(key, None)
            if type(part) is FilePart:
                reference = file_attachment_from_payload(part.attachment)
                if reference is None:  # pragma: no cover - FilePart validation owns this path.
                    raise ValueError("User file part contains an invalid attachment reference.")
                resolved = resolved_attachments.get(reference.artifact_id)
                if resolved is None:
                    raise ValueError(f"Missing resolved file attachment: {reference.artifact_id}")
                part_payload["attachment"] = _provider_neutral_attachment_payload(
                    ResolvedFileAttachment.model_validate(resolved)
                )
            elif type(part) is ToolResultPart:
                part_payload.pop("structured", None)
                artifacts: list[dict[str, Any]] = []
                for artifact in part.artifacts:
                    reference = file_attachment_from_payload(artifact)
                    if reference is None:
                        continue
                    resolved = resolved_attachments.get(reference.artifact_id)
                    if resolved is None:
                        raise ValueError(
                            f"Missing resolved file attachment: {reference.artifact_id}"
                        )
                    artifacts.append(
                        {
                            "type": "file",
                            "attachment": _provider_neutral_attachment_payload(
                                ResolvedFileAttachment.model_validate(resolved)
                            ),
                        }
                    )
                part_payload["artifacts"] = artifacts
        payloads.append(message_payload)
    return payloads


def _native_structured_output_projection(options: dict[str, Any]) -> dict[str, Any] | None:
    raw = options.get("structured_output")
    if raw is None:
        return None
    if type(raw) is not dict:
        raise ValueError("ModelRequest options.structured_output must be an object.")
    if raw.get("strategy", "tool") != "native":
        return None
    schema = raw.get("schema")
    if type(schema) is not dict:
        raise ValueError("Native structured output schema must be an object.")
    raw_name = raw.get("name")
    name = (
        "structured_output"
        if raw_name is None
        else require_durable_clean_nonblank(raw_name, "structured_output.name")
    )
    return {
        "type": "json_schema",
        "name": name,
        "schema": copy_json_value(schema, "structured_output.schema"),
        "strict": True,
    }


def _request_measurement_projection(
    model_request: ModelRequest,
    *,
    measured_provider_options: dict[str, Any] | None,
    effective_provider_options: dict[str, Any] | None = None,
) -> _RequestMeasurementProjection:
    visible_options = {
        key: value
        for key, value in model_request.options.items()
        if key not in _RUNTIME_ONLY_OPTION_KEYS and value is not None
    }
    if measured_provider_options is None:
        measured_options = {
            key: copy_json_value(value, f"measured request option {key}")
            for key, value in visible_options.items()
            if key in _KNOWN_PROVIDER_OPTION_CATEGORIES
        }
        known_categories = set(measured_options)
    else:
        measured_options = copy_json_value(
            measured_provider_options,
            "measured_provider_options",
        )
        known_categories = _option_category_paths(measured_options)
    native_structured_output = _native_structured_output_projection(model_request.options)
    if native_structured_output is not None:
        known_categories.add("structured_output")
    return _RequestMeasurementProjection(
        visible_options=visible_options,
        measured_options=measured_options,
        native_structured_output=native_structured_output,
        known_categories=tuple(sorted(known_categories)),
        unknown_count=_unknown_option_count(
            (visible_options if effective_provider_options is None else effective_provider_options),
            measured_options,
        ),
    )


def _measured_request_options(
    provider_options: dict[str, Any],
    *,
    cache_markers: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    copied_provider_options = copy_json_value(
        provider_options,
        "measured provider request options",
    )
    if cache_markers is None:
        return copied_provider_options
    return {
        "provider_options": copied_provider_options,
        "cache_control_markers": copy_json_value(
            cache_markers,
            "measured cache control markers",
        ),
    }


def _estimate_projected_context_pressure(
    *,
    model_request: ModelRequest,
    messages: list[Message],
    measured_options: dict[str, Any],
    native_structured_output: dict[str, Any] | None,
    profile: ModelContextPressureProfile,
    estimator: ObservedDeltaContextEstimator,
) -> ContextPressureEstimate:
    measured_request = ModelRequest(
        model=model_request.model,
        messages=messages,
        tools=model_request.tools,
        hosted_tools=model_request.hosted_tools,
        options=measured_options,
    )
    pressure = estimate_model_request_context_pressure(
        model_request=measured_request,
        image_min_tokens=profile.image_min_tokens,
        document_min_tokens=profile.document_min_tokens,
        document_bytes_per_token=profile.document_bytes_per_token,
        tool_schema_chars_per_token=profile.tool_schema_chars_per_token,
        estimator=estimator,
    )
    if native_structured_output is None:
        return pressure
    native_tokens = _estimated_json_tokens(
        native_structured_output,
        chars_per_token=pressure.json_chars_per_token,
    )
    return pressure.model_copy(
        update={
            "estimated_structured_output_input_tokens": native_tokens,
            "estimated_request_overhead_input_tokens": (
                pressure.estimated_request_overhead_input_tokens + native_tokens
            ),
            "estimated_request_overhead_delta_tokens": (
                pressure.estimated_request_overhead_delta_tokens + native_tokens
            ),
            "estimated_delta_input_tokens": pressure.estimated_delta_input_tokens + native_tokens,
            "estimated_context_input_tokens": (
                pressure.estimated_context_input_tokens + native_tokens
            ),
            "estimated_context_window_tokens": (
                pressure.estimated_context_window_tokens + native_tokens
            ),
        },
        deep=True,
    )


def _provider_neutral_fingerprint_options(
    *,
    visible_options: dict[str, Any],
    fingerprint_provider_options: dict[str, Any] | None,
) -> dict[str, Any]:
    if fingerprint_provider_options is not None:
        return copy_json_value(
            fingerprint_provider_options,
            "provider-neutral fingerprint options",
        )
    return {
        key: copy_json_value(value, f"provider-neutral fingerprint option {key}")
        for key, value in visible_options.items()
        if key not in _RUNTIME_ONLY_OPTION_KEYS
    }


def _provider_neutral_cache_policy_payload(
    *,
    cache_policy: CachePolicy | None,
    conversation_prefix: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if cache_policy is None or not cache_policy.breakpoints:
        return None
    breakpoints = tuple(sorted(set(cache_policy.breakpoints), key=lambda item: item.value))
    payload: dict[str, Any] = {
        "breakpoints": [breakpoint.value for breakpoint in breakpoints],
        "ttl": "extended" if cache_policy.ttl == "extended" else "standard",
    }
    if CacheBreakpoint.CONVERSATION_PREFIX in breakpoints:
        payload["conversation_prefix"] = conversation_prefix
    return payload


def _provider_neutral_cache_marker_payload(
    cache_policy: CachePolicy | None,
) -> list[dict[str, Any]] | None:
    if cache_policy is None or not cache_policy.breakpoints:
        return None
    marker = cache_policy.marker()
    return [
        {
            "breakpoint": breakpoint.value,
            "cache_control": copy_json_value(marker, "cache control marker"),
        }
        for breakpoint in sorted(
            set(cache_policy.breakpoints),
            key=lambda item: item.value,
        )
    ]


def _fingerprint(
    value: Any,
    *,
    scope: str,
    config: RequestFootprintConfig,
    unavailable_reason: str | None = None,
) -> RequestFingerprint:
    if unavailable_reason is not None:
        return _unavailable_fingerprint(unavailable_reason)
    if config.fingerprint_key is None or config.fingerprint_key_id is None:
        return _unavailable_fingerprint("fingerprint_key_not_configured")
    material = b"\x00".join(
        (
            _HMAC_CONTEXT,
            str(REQUEST_FOOTPRINT_CANONICALIZATION_VERSION).encode("ascii"),
            config.fingerprint_key_id.encode("ascii"),
            scope.encode("ascii"),
            canonical_durable_json_bytes(value, "request fingerprint material"),
        )
    )
    value_digest = hmac.digest(
        config.fingerprint_key.get_secret_value().encode("utf-8"),
        material,
        hashlib.sha256,
    ).hex()
    return RequestFingerprint(
        availability=RequestFingerprintAvailability.AVAILABLE,
        value=value_digest,
        algorithm="hmac-sha256",
        key_id=config.fingerprint_key_id,
        canonicalization_version=REQUEST_FOOTPRINT_CANONICALIZATION_VERSION,
    )


def _unavailable_fingerprint(reason: str) -> RequestFingerprint:
    return RequestFingerprint(
        availability=RequestFingerprintAvailability.UNAVAILABLE,
        canonicalization_version=REQUEST_FOOTPRINT_CANONICALIZATION_VERSION,
        unavailable_reason=require_durable_clean_nonblank(reason, "unavailable_reason"),
    )


def _structured_output_tokens(
    *,
    estimator: ObservedDeltaContextEstimator,
    native_structured_output_option: dict[str, Any] | None,
    structured_output_tools: list[dict[str, Any]],
    structured_output_instruction: str | None,
    tool_schema_chars_per_token: int,
    json_chars_per_token: int,
) -> int:
    if native_structured_output_option is not None:
        return _estimated_json_tokens(
            native_structured_output_option,
            chars_per_token=json_chars_per_token,
        )
    tool_tokens = estimator.estimate_tool_schema_tokens(
        structured_output_tools,
        chars_per_token=tool_schema_chars_per_token,
    )
    instruction_tokens = (
        0
        if structured_output_instruction is None
        else estimator.estimate_message_tokens(
            Message.text(MessageRole.SYSTEM, structured_output_instruction)
        )
    )
    return tool_tokens + instruction_tokens


def _option_category_paths(
    value: dict[str, Any],
) -> set[str]:
    categories: set[str] = set()
    for namespace, options in value.items():
        if isinstance(options, dict) and options:
            categories.update(f"{namespace}.{key}" for key in options)
        else:
            categories.add(namespace)
    return categories


def _unknown_option_count(raw: dict[str, Any], measured: dict[str, Any]) -> int:
    count = 0
    for key, value in raw.items():
        if key not in measured:
            count += 1
            continue
        measured_value = measured[key]
        if isinstance(value, dict) and isinstance(measured_value, dict):
            count += _unknown_option_count(value, measured_value)
    return count


def _estimated_json_tokens(value: Any, *, chars_per_token: int) -> int:
    if value is None or value == {} or value == []:
        return 0
    encoded = _canonical_json(value).decode("utf-8")
    return math.ceil(len(encoded) / chars_per_token)


def _conversation_prefix_payload(
    model_request: ModelRequest,
    *,
    message_payloads: list[dict[str, Any]],
    cache_policy: CachePolicy | None,
    provider_conversation_prefix: tuple[dict[str, Any], ...] | None,
) -> list[dict[str, Any]] | None:
    if (
        cache_policy is None
        or CacheBreakpoint.CONVERSATION_PREFIX not in cache_policy.breakpoints
        or cache_policy.conversation_prefix_strategy == "none"
    ):
        return None
    if provider_conversation_prefix is not None:
        return copy_json_value(
            list(provider_conversation_prefix),
            "provider conversation prefix",
        )
    messages = [
        payload
        for message, payload in zip(
            model_request.messages,
            message_payloads,
            strict=True,
        )
        if message.role != MessageRole.SYSTEM
    ]
    skip = (
        cache_policy.conversation_prefix_n
        if cache_policy.conversation_prefix_strategy == "all_but_last_n"
        else 1
    )
    retained = len(messages) - skip
    if retained <= 0:
        return None
    return messages[:retained]


def _request_tool_manifest(model_request: ModelRequest) -> tuple[Any, list[dict[str, Any]]]:
    hosted_tools = [tool.model_dump(mode="json") for tool in model_request.hosted_tools]
    if not hosted_tools:
        return model_request.tools, hosted_tools
    return {
        "function_tools": model_request.tools,
        "hosted_tools": hosted_tools,
    }, hosted_tools


def _cache_breakpoint_footprints(
    *,
    model_request: ModelRequest,
    system_payloads: list[dict[str, Any]],
    conversation_prefix: list[dict[str, Any]] | None,
    cache_policy: CachePolicy | None,
    config: RequestFootprintConfig,
) -> tuple[RequestCacheBreakpointFootprint, ...]:
    if cache_policy is None:
        return ()
    breakpoints = tuple(sorted(set(cache_policy.breakpoints), key=lambda item: item.value))
    ttl = "extended" if cache_policy.ttl == "extended" else "standard"
    tool_manifest, hosted_tool_payloads = _request_tool_manifest(model_request)
    payloads: dict[CacheBreakpoint, tuple[Any, str | None]] = {
        CacheBreakpoint.SYSTEM_PROMPT: (
            {"system": system_payloads, "ttl": ttl},
            None if system_payloads else "system_not_present",
        ),
        CacheBreakpoint.TOOL_DEFINITIONS: (
            {"system": system_payloads, "tools": tool_manifest, "ttl": ttl},
            (None if model_request.tools or hosted_tool_payloads else "tools_not_present"),
        ),
        CacheBreakpoint.CONVERSATION_PREFIX: (
            {
                "system": system_payloads,
                "tools": tool_manifest,
                "messages": conversation_prefix,
                "ttl": ttl,
            },
            None if conversation_prefix is not None else "cache_conversation_prefix_unavailable",
        ),
    }
    return tuple(
        RequestCacheBreakpointFootprint(
            kind=breakpoint,
            ttl=ttl,
            fingerprint=_fingerprint(
                payloads[breakpoint][0],
                scope=f"cache-breakpoint:{breakpoint.value}",
                config=config,
                unavailable_reason=payloads[breakpoint][1],
            ),
        )
        for breakpoint in breakpoints
    )


def _require_positive_int(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 1:
        raise ValueError(f"{field_name} must be greater than zero.")
    if value > MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"{field_name} must fit in a signed 64-bit integer.")
    return value
