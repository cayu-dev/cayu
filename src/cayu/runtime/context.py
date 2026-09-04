from __future__ import annotations

import asyncio
import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._exception_groups import exception_cause
from cayu._task_wait import consume_pending_task_cancellation
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    DurableValueError,
    copy_durable_json_object,
    copy_durable_json_value,
    copy_json_value,
    require_clean_nonblank,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_durable_text,
    require_nonblank,
    safe_durable_value_error_details,
)
from cayu.artifacts import (
    RESOLVED_FILE_ATTACHMENTS_OPTION,
    FileAttachment,
    FileAttachmentKind,
    file_attachment_from_payload,
)
from cayu.core.agents import AgentSpec, copy_agent_spec
from cayu.core.billing import (
    BillingIdentity,
)
from cayu.core.events import EventType
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.messages import (
    CitationPart,
    FilePart,
    HostedToolCallPart,
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    copy_message,
    copy_message_part,
)
from cayu.core.tools import ToolSpec
from cayu.providers.base import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEvent,
    ModelStreamEventType,
    UsageDialect,
    copy_usage_dialect,
)
from cayu.runtime._checkpoint_redaction import require_secret_free_durable_object
from cayu.runtime._completion_projection import portable_model_completion_projection
from cayu.runtime._model_errors import (
    ProviderExceptionControl,
    copy_provider_exception_control,
    copy_provider_hook_error_control,
    detach_billing_identity_cancellation,
    model_provider_error_from_payload,
    nonportable_model_provider_error,
    resolve_completion_billing_identity,
    resolve_request_billing_identity,
)
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    RUNTIME_AUTHORED_USER_MESSAGE_CHECKPOINT_KEY,
    RUNTIME_AUTHORED_USER_MESSAGE_CHECKPOINT_VERSION,
    SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)
from cayu.runtime.execution_units import (
    ModelAttemptIdentity,
    copy_model_attempt_identity,
    strip_runtime_owned_execution_identity,
)
from cayu.runtime.retry_policy import RetryPolicy, copy_retry_policy, retry_decision
from cayu.runtime.sessions import Session, copy_session
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.runtime.usage import (
    ModelCompletionPurpose,
    UsageMetrics,
    durable_model_completed_payload,
    normalize_usage_metrics_with_overflow_error,
    strip_provider_billing_identity,
    usage_metrics_from_event_payload,
    usage_metrics_payload,
)
from cayu.vaults import SecretRedactor

_COMPACTION_CHECKPOINT_KEY = "context_compaction"
_COMPACTION_CHECKPOINT_VERSION = 2
_DEFAULT_CHECKPOINT_COMPACTION_SUMMARY_PREFIX = "Previous session context summary:"
_COMPACTION_PROGRESS_STATE_KEY = "progress"
_COMPACTION_PROGRESS_EXHAUSTED_KEY = "exhausted"
_COMPACTION_PROGRESS_KEY = "key"
_CONTEXT_CHECKPOINT_EVENT_STRUCTURE_KEYS = frozenset(
    {
        "checkpoint",
        "compacted_transcript_cursor",
        "estimated_context_input_tokens",
        "estimated_context_window_tokens",
        "estimated_delta_input_tokens",
        "last_input_tokens",
        "last_total_tokens",
        "last_transcript_cursor",
        "min_input_tokens",
        "min_total_tokens",
        "newly_compacted_message_count",
        "previous_compacted_transcript_cursor",
        "provider_count_context_window_tokens",
        "provider_count_input_tokens",
        "recent_message_count",
        "reserved_output_tokens",
        "trigger_estimated_context_tokens",
    }
)
_USAGE_TRIGGERED_CHECKPOINT_KEY = "usage_triggered_context"
_USAGE_TRIGGERED_CHECKPOINT_VERSION = 1
_DEFAULT_ESTIMATE_CHARS_PER_TOKEN = 5
_DEFAULT_ESTIMATE_JSON_CHARS_PER_TOKEN = 5
_DEFAULT_ESTIMATE_JSON_TEXT_CHARS_PER_TOKEN = 3
_DEFAULT_ESTIMATE_BINARY_BYTES_PER_TOKEN = 3
_DEFAULT_ESTIMATE_IMAGE_MIN_TOKENS = 32
_DEFAULT_ESTIMATE_DOCUMENT_MIN_TOKENS = 0
_DEFAULT_ESTIMATE_TOOL_SCHEMA_CHARS_PER_TOKEN = 4
_INTERNAL_REQUEST_OPTION_KEYS = frozenset(
    {
        "agent_metadata",
        "environment_metadata",
        "step",
        "structured_output",
        RESOLVED_FILE_ATTACHMENTS_OPTION,
    }
)


class ContextPressureOverhead(BaseModel):
    """Known provider-request overhead included in local context pressure estimates."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    tools: list[dict[str, Any]] = Field(default_factory=list)
    structured_output_instruction: str | None = None
    request_options: dict[str, Any] = Field(default_factory=dict)
    image_min_tokens: StrictInt = Field(default=_DEFAULT_ESTIMATE_IMAGE_MIN_TOKENS, ge=0)
    document_min_tokens: StrictInt = Field(
        default=_DEFAULT_ESTIMATE_DOCUMENT_MIN_TOKENS,
        ge=0,
    )
    document_bytes_per_token: StrictInt = Field(
        default=_DEFAULT_ESTIMATE_BINARY_BYTES_PER_TOKEN,
        ge=1,
    )
    tool_schema_chars_per_token: StrictInt = Field(
        default=_DEFAULT_ESTIMATE_TOOL_SCHEMA_CHARS_PER_TOKEN,
        ge=1,
    )

    @field_validator("tools", "request_options", mode="before")
    @classmethod
    def copy_json_data(cls, value, info):
        return copy_durable_json_value(value, info.field_name)

    @field_validator("structured_output_instruction")
    @classmethod
    def validate_optional_instruction(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, "structured_output_instruction")


def copy_context_pressure_overhead(
    overhead: ContextPressureOverhead | None,
) -> ContextPressureOverhead:
    if overhead is None:
        return ContextPressureOverhead()
    if type(overhead) is not ContextPressureOverhead:
        raise TypeError("Context pressure overhead must be a ContextPressureOverhead instance.")
    return ContextPressureOverhead(**overhead.model_dump())


class ContextPressureEstimate(BaseModel):
    """Estimated current context pressure.

    This is a local pressure signal, not provider-authoritative token counting and
    not billing data. Estimates may use previous actual usage plus a transcript delta
    or a full local estimate of the model-facing request shape.
    """

    model_config = ConfigDict(extra="forbid")

    method: str = "observed_plus_estimated_delta"
    confidence: str = "estimated"
    observed_context_input_tokens: StrictInt = Field(ge=0)
    estimated_delta_input_tokens: StrictInt = Field(ge=0)
    estimated_message_input_tokens: StrictInt = Field(ge=0)
    estimated_tool_schema_input_tokens: StrictInt = Field(ge=0)
    estimated_structured_output_input_tokens: StrictInt = Field(ge=0)
    estimated_request_options_input_tokens: StrictInt = Field(ge=0)
    estimated_request_overhead_input_tokens: StrictInt = Field(default=0, ge=0)
    previous_request_overhead_input_tokens: StrictInt | None = Field(default=None, ge=0)
    estimated_request_overhead_delta_tokens: StrictInt = 0
    estimated_attachment_input_tokens: StrictInt = Field(ge=0)
    estimated_context_input_tokens: StrictInt = Field(ge=0)
    reserved_output_tokens: StrictInt = Field(default=0, ge=0)
    estimated_context_window_tokens: StrictInt = Field(ge=0)
    provider_count_input_tokens: StrictInt | None = Field(default=None, ge=0)
    provider_count_context_window_tokens: StrictInt | None = Field(default=None, ge=0)
    anchor_transcript_cursor: StrictInt = Field(ge=0)
    current_transcript_cursor: StrictInt = Field(ge=0)
    estimated_message_count: StrictInt = Field(ge=0)
    chars_per_token: StrictInt = Field(ge=1)
    json_chars_per_token: StrictInt = Field(ge=1)
    binary_bytes_per_token: StrictInt = Field(ge=1)

    @field_validator("method", "confidence")
    @classmethod
    def validate_nonblank(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


def copy_context_pressure_estimate(
    estimate: ContextPressureEstimate | None,
) -> ContextPressureEstimate | None:
    if estimate is None:
        return None
    if type(estimate) is not ContextPressureEstimate:
        raise TypeError("Context pressure estimate must be a ContextPressureEstimate instance.")
    return ContextPressureEstimate(**estimate.model_dump())


class ContextUsageState(BaseModel):
    """Actual provider usage from the previous completed model request."""

    model_config = ConfigDict(extra="forbid")

    last_input_tokens: StrictInt | None = Field(default=None, ge=0)
    last_output_tokens: StrictInt | None = Field(default=None, ge=0)
    last_total_tokens: StrictInt | None = Field(default=None, ge=0)
    last_transcript_cursor: StrictInt | None = Field(default=None, ge=0)
    last_context_overhead_input_tokens: StrictInt | None = Field(default=None, ge=0)
    last_provider_name: str | None = None
    last_requested_model: str | None = None
    last_model: str | None = None
    input_pressure: ContextPressureEstimate | None = None

    @field_validator("last_provider_name", "last_requested_model", "last_model")
    @classmethod
    def validate_optional_nonblank(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("input_pressure")
    @classmethod
    def copy_input_pressure(
        cls,
        value: ContextPressureEstimate | None,
    ) -> ContextPressureEstimate | None:
        return copy_context_pressure_estimate(value)


def copy_context_usage_state(state: ContextUsageState) -> ContextUsageState:
    if type(state) is not ContextUsageState:
        raise TypeError("Context usage state must be a ContextUsageState instance.")
    return ContextUsageState(**state.model_dump())


class ObservedDeltaContextEstimator:
    """Estimates context pressure from actual usage plus a local tail estimate."""

    def __init__(
        self,
        *,
        chars_per_token: int = _DEFAULT_ESTIMATE_CHARS_PER_TOKEN,
        json_chars_per_token: int = _DEFAULT_ESTIMATE_JSON_CHARS_PER_TOKEN,
        binary_bytes_per_token: int = _DEFAULT_ESTIMATE_BINARY_BYTES_PER_TOKEN,
    ) -> None:
        self.chars_per_token = _validate_positive_int(chars_per_token, "chars_per_token")
        self.json_chars_per_token = _validate_positive_int(
            json_chars_per_token,
            "json_chars_per_token",
        )
        self.binary_bytes_per_token = _validate_positive_int(
            binary_bytes_per_token,
            "binary_bytes_per_token",
        )

    def estimate(
        self,
        *,
        usage: ContextUsageState,
        messages: list[Message],
        image_min_tokens: int = _DEFAULT_ESTIMATE_IMAGE_MIN_TOKENS,
        document_min_tokens: int = _DEFAULT_ESTIMATE_DOCUMENT_MIN_TOKENS,
        document_bytes_per_token: int = _DEFAULT_ESTIMATE_BINARY_BYTES_PER_TOKEN,
    ) -> ContextPressureEstimate | None:
        if type(usage) is not ContextUsageState:
            raise TypeError("usage must be a ContextUsageState.")
        if usage.last_input_tokens is None or usage.last_transcript_cursor is None:
            return None
        current_cursor = len(messages)
        if usage.last_transcript_cursor > current_cursor:
            return None
        tail_messages = messages[usage.last_transcript_cursor :]
        message_tokens = sum(self.estimate_message_tokens(message) for message in tail_messages)
        attachment_tokens = sum(
            self.estimate_message_attachment_tokens(
                message,
                image_min_tokens=image_min_tokens,
                document_min_tokens=document_min_tokens,
                document_bytes_per_token=document_bytes_per_token,
            )
            for message in tail_messages
        )
        delta_tokens = message_tokens + attachment_tokens
        return ContextPressureEstimate(
            observed_context_input_tokens=usage.last_input_tokens,
            estimated_delta_input_tokens=delta_tokens,
            estimated_message_input_tokens=message_tokens,
            estimated_tool_schema_input_tokens=0,
            estimated_structured_output_input_tokens=0,
            estimated_request_options_input_tokens=0,
            estimated_request_overhead_input_tokens=0,
            previous_request_overhead_input_tokens=usage.last_context_overhead_input_tokens,
            estimated_request_overhead_delta_tokens=0,
            estimated_attachment_input_tokens=attachment_tokens,
            estimated_context_input_tokens=usage.last_input_tokens + delta_tokens,
            reserved_output_tokens=0,
            estimated_context_window_tokens=usage.last_input_tokens + delta_tokens,
            anchor_transcript_cursor=usage.last_transcript_cursor,
            current_transcript_cursor=current_cursor,
            estimated_message_count=len(tail_messages),
            chars_per_token=self.chars_per_token,
            json_chars_per_token=self.json_chars_per_token,
            binary_bytes_per_token=self.binary_bytes_per_token,
        )

    def estimate_full_request(
        self,
        *,
        usage: ContextUsageState,
        messages: list[Message],
        overhead: ContextPressureOverhead | None = None,
        reserved_output_tokens: int = 0,
    ) -> ContextPressureEstimate:
        if type(usage) is not ContextUsageState:
            raise TypeError("usage must be a ContextUsageState.")
        overhead = copy_context_pressure_overhead(overhead)
        reserved_output_tokens = _validate_nonnegative_int(
            reserved_output_tokens,
            "reserved_output_tokens",
        )
        message_tokens = sum(self.estimate_message_tokens(message) for message in messages)
        attachment_tokens = sum(
            self.estimate_message_attachment_tokens(
                message,
                image_min_tokens=overhead.image_min_tokens,
                document_min_tokens=overhead.document_min_tokens,
                document_bytes_per_token=overhead.document_bytes_per_token,
            )
            for message in messages
        )
        tool_schema_tokens = self.estimate_tool_schema_tokens(
            overhead.tools,
            chars_per_token=overhead.tool_schema_chars_per_token,
        )
        structured_output_tokens = self._estimate_text(overhead.structured_output_instruction or "")
        request_options_tokens = self._estimate_request_options(overhead.request_options)
        overhead_tokens = tool_schema_tokens + structured_output_tokens + request_options_tokens
        total_tokens = message_tokens + attachment_tokens + overhead_tokens
        return ContextPressureEstimate(
            method="local_full_request_estimate",
            observed_context_input_tokens=usage.last_input_tokens or 0,
            estimated_delta_input_tokens=total_tokens,
            estimated_message_input_tokens=message_tokens,
            estimated_tool_schema_input_tokens=tool_schema_tokens,
            estimated_structured_output_input_tokens=structured_output_tokens,
            estimated_request_options_input_tokens=request_options_tokens,
            estimated_request_overhead_input_tokens=overhead_tokens,
            previous_request_overhead_input_tokens=usage.last_context_overhead_input_tokens,
            estimated_request_overhead_delta_tokens=overhead_tokens,
            estimated_attachment_input_tokens=attachment_tokens,
            estimated_context_input_tokens=total_tokens,
            reserved_output_tokens=reserved_output_tokens,
            estimated_context_window_tokens=total_tokens + reserved_output_tokens,
            anchor_transcript_cursor=usage.last_transcript_cursor or 0,
            current_transcript_cursor=len(messages),
            estimated_message_count=len(messages),
            chars_per_token=self.chars_per_token,
            json_chars_per_token=self.json_chars_per_token,
            binary_bytes_per_token=self.binary_bytes_per_token,
        )

    def estimate_anchored_request(
        self,
        *,
        usage: ContextUsageState,
        messages: list[Message],
        overhead: ContextPressureOverhead | None = None,
        reserved_output_tokens: int = 0,
    ) -> ContextPressureEstimate:
        if type(usage) is not ContextUsageState:
            raise TypeError("usage must be a ContextUsageState.")
        overhead = copy_context_pressure_overhead(overhead)
        reserved_output_tokens = _validate_nonnegative_int(
            reserved_output_tokens,
            "reserved_output_tokens",
        )
        base = self.estimate(
            usage=usage,
            messages=messages,
            image_min_tokens=overhead.image_min_tokens,
            document_min_tokens=overhead.document_min_tokens,
            document_bytes_per_token=overhead.document_bytes_per_token,
        )
        if base is None:
            return self.estimate_full_request(
                usage=usage,
                messages=messages,
                overhead=overhead,
                reserved_output_tokens=reserved_output_tokens,
            )

        tool_schema_tokens = self.estimate_tool_schema_tokens(
            overhead.tools,
            chars_per_token=overhead.tool_schema_chars_per_token,
        )
        structured_output_tokens = self._estimate_text(overhead.structured_output_instruction or "")
        request_options_tokens = self._estimate_request_options(overhead.request_options)
        overhead_tokens = tool_schema_tokens + structured_output_tokens + request_options_tokens
        previous_overhead_tokens = usage.last_context_overhead_input_tokens
        overhead_delta_tokens = (
            0 if previous_overhead_tokens is None else overhead_tokens - previous_overhead_tokens
        )
        estimated_context_input_tokens = max(
            0,
            base.estimated_context_input_tokens + overhead_delta_tokens,
        )
        estimated_delta_input_tokens = max(
            0,
            base.estimated_delta_input_tokens + overhead_delta_tokens,
        )
        return ContextPressureEstimate(
            method="observed_plus_estimated_delta_with_overhead",
            observed_context_input_tokens=base.observed_context_input_tokens,
            estimated_delta_input_tokens=estimated_delta_input_tokens,
            estimated_message_input_tokens=base.estimated_message_input_tokens,
            estimated_tool_schema_input_tokens=tool_schema_tokens,
            estimated_structured_output_input_tokens=structured_output_tokens,
            estimated_request_options_input_tokens=request_options_tokens,
            estimated_request_overhead_input_tokens=overhead_tokens,
            previous_request_overhead_input_tokens=previous_overhead_tokens,
            estimated_request_overhead_delta_tokens=overhead_delta_tokens,
            estimated_attachment_input_tokens=base.estimated_attachment_input_tokens,
            estimated_context_input_tokens=estimated_context_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            estimated_context_window_tokens=(
                estimated_context_input_tokens + reserved_output_tokens
            ),
            anchor_transcript_cursor=base.anchor_transcript_cursor,
            current_transcript_cursor=base.current_transcript_cursor,
            estimated_message_count=base.estimated_message_count,
            chars_per_token=self.chars_per_token,
            json_chars_per_token=self.json_chars_per_token,
            binary_bytes_per_token=self.binary_bytes_per_token,
        )

    def estimate_message_tokens(self, message: Message) -> int:
        if type(message) is not Message:
            raise TypeError("message must be a Message.")
        total = 0
        for part in message.content:
            if type(part) is TextPart:
                total += self._estimate_text(part.text)
            elif type(part) is ToolCallPart:
                total += self._estimate_text(part.tool_name)
                total += self._estimate_json(part.arguments)
            elif type(part) is FilePart:
                # FilePart is a runtime lookup reference, not provider-visible
                # prompt text. Its provider-facing descriptor and payload are
                # accounted for exactly once by the attachment estimator.
                continue
            elif type(part) is ToolResultPart:
                total += self._estimate_text(part.content)
            elif type(part) is ProviderStatePart:
                total += self._estimate_text(part.provider)
                total += self._estimate_json(part.state)
            elif type(part) is ThinkingPart:
                total += self._estimate_text(part.text)
                total += self._estimate_json(part.provider_state)
            else:  # pragma: no cover - Message validation should keep this closed.
                total += self._estimate_json(part.model_dump(mode="json"))
        return total

    def estimate_message_attachment_tokens(
        self,
        message: Message,
        *,
        image_min_tokens: int = _DEFAULT_ESTIMATE_IMAGE_MIN_TOKENS,
        document_min_tokens: int = _DEFAULT_ESTIMATE_DOCUMENT_MIN_TOKENS,
        document_bytes_per_token: int = _DEFAULT_ESTIMATE_BINARY_BYTES_PER_TOKEN,
    ) -> int:
        if type(message) is not Message:
            raise TypeError("message must be a Message.")
        total = 0
        for part in message.content:
            if type(part) is FilePart:
                attachments = (file_attachment_from_payload(part.attachment),)
            elif type(part) is ToolResultPart:
                attachments = tuple(
                    attachment
                    for payload in part.artifacts
                    if (attachment := file_attachment_from_payload(payload)) is not None
                )
            else:
                continue
            for attachment in attachments:
                if attachment is None:
                    continue
                total += self._estimate_file_attachment(
                    attachment,
                    image_min_tokens=image_min_tokens,
                    document_min_tokens=document_min_tokens,
                    document_bytes_per_token=document_bytes_per_token,
                )
        return total

    def estimate_tool_schema_tokens(
        self,
        tools: list[dict[str, Any]] | list[ToolSpec],
        *,
        chars_per_token: int = _DEFAULT_ESTIMATE_TOOL_SCHEMA_CHARS_PER_TOKEN,
    ) -> int:
        chars_per_token = _validate_positive_int(chars_per_token, "chars_per_token")
        total = 0
        for tool in tools:
            if type(tool) is ToolSpec:
                payload = tool.model_dump(mode="json")
            else:
                payload = copy_json_value(tool, "tool")
            total += self._estimate_json(payload, chars_per_token=chars_per_token)
        return total

    def _estimate_file_attachment(
        self,
        attachment: FileAttachment,
        *,
        image_min_tokens: int,
        document_min_tokens: int,
        document_bytes_per_token: int,
    ) -> int:
        descriptor_tokens = self._estimate_text(attachment.filename) + self._estimate_text(
            attachment.content_type
        )
        if attachment.kind == FileAttachmentKind.IMAGE:
            # Providers account for image blocks with modality-specific formulas.
            # The provider adapter supplies the conservative floor; the runtime
            # estimator applies it without branching on provider identity.
            payload_tokens = max(
                image_min_tokens,
                math.ceil(attachment.size_bytes / 16),
            )
            return descriptor_tokens + payload_tokens
        if attachment.kind == FileAttachmentKind.DOCUMENT:
            payload_tokens = max(
                document_min_tokens,
                math.ceil(attachment.size_bytes / document_bytes_per_token),
            )
            return descriptor_tokens + payload_tokens
        payload_tokens = math.ceil(attachment.size_bytes / self.binary_bytes_per_token)
        return descriptor_tokens + payload_tokens

    def _estimate_request_options(self, options: dict[str, Any]) -> int:
        visible_options: dict[str, Any] = {}
        for key, value in options.items():
            if key not in _INTERNAL_REQUEST_OPTION_KEYS and value is not None:
                visible_options[key] = value
        structured_output = options.get("structured_output")
        if isinstance(structured_output, dict) and structured_output.get("strategy") == "native":
            visible_options["structured_output"] = structured_output
        return self._estimate_json(visible_options)

    def _estimate_text(self, value: str) -> int:
        if not value:
            return 0
        if self._looks_like_json_lines(value):
            return math.ceil(len(value) / _DEFAULT_ESTIMATE_JSON_TEXT_CHARS_PER_TOKEN)
        return math.ceil(len(value) / self._text_chars_per_token(value))

    def _text_chars_per_token(self, value: str) -> float:
        length = len(value)
        if length < 200:
            return float(self.chars_per_token)
        whitespace_count = sum(1 for char in value if char.isspace())
        alnum_count = sum(1 for char in value if char.isalnum())
        punctuation_count = length - whitespace_count - alnum_count
        whitespace_ratio = whitespace_count / length
        punctuation_ratio = punctuation_count / length
        quote_ratio = value.count('"') / length
        comma_ratio = value.count(",") / length
        digit_ratio = sum(1 for char in value if char.isdigit()) / length
        if whitespace_ratio <= 0.03 or punctuation_ratio >= 0.30:
            if comma_ratio >= 0.05 or digit_ratio >= 0.12 or punctuation_ratio < 0.10:
                return min(float(self.chars_per_token), 2.5)
            return min(float(self.chars_per_token), 3.0)
        if punctuation_ratio >= 0.25 or quote_ratio >= 0.08:
            return min(float(self.chars_per_token), 2.5)
        if digit_ratio >= 0.15 and punctuation_ratio >= 0.10:
            return min(float(self.chars_per_token), 2.75)
        if punctuation_ratio >= 0.12:
            return min(float(self.chars_per_token), 3.75)
        return float(self.chars_per_token)

    def _estimate_json(
        self,
        value: Any,
        *,
        chars_per_token: int | None = None,
    ) -> int:
        if value is None:
            return 0
        if value == {} or value == []:
            return 0
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if not encoded:
            return 0
        return math.ceil(len(encoded) / (chars_per_token or self.json_chars_per_token))

    def _looks_like_json_lines(self, value: str) -> bool:
        stripped = value.strip()
        if len(stripped) < 200:
            return False
        if stripped[0] not in "[{":
            return False
        quote_ratio = stripped.count('"') / len(stripped)
        colon_ratio = stripped.count(":") / len(stripped)
        if quote_ratio < 0.05 or colon_ratio < 0.02:
            return False
        if "\n" not in stripped:
            return True
        if stripped[0] in "[{":
            return True
        sample_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if not sample_lines:
            return False
        sample = sample_lines[: min(5, len(sample_lines))]
        return all(line[0] in "[{" and ("," in line or ":" in line) for line in sample)


def estimate_context_pressure(
    *,
    usage: ContextUsageState,
    messages: list[Message],
    image_min_tokens: int = _DEFAULT_ESTIMATE_IMAGE_MIN_TOKENS,
    document_min_tokens: int = _DEFAULT_ESTIMATE_DOCUMENT_MIN_TOKENS,
    document_bytes_per_token: int = _DEFAULT_ESTIMATE_BINARY_BYTES_PER_TOKEN,
    estimator: ObservedDeltaContextEstimator | None = None,
) -> ContextUsageState:
    if estimator is None:
        estimator = ObservedDeltaContextEstimator()
    pressure = estimator.estimate(
        usage=usage,
        messages=messages,
        image_min_tokens=image_min_tokens,
        document_min_tokens=document_min_tokens,
        document_bytes_per_token=document_bytes_per_token,
    )
    return usage.model_copy(update={"input_pressure": pressure})


class ContextRequest(BaseModel):
    """Input passed to an agent context policy before each model request."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session: Session
    agent: AgentSpec
    messages: list[Message]
    step: StrictInt = Field(ge=1)
    interaction_id: str | None = Field(default=None, exclude=True)
    model_step_id: str | None = Field(default=None, exclude=True)
    environment_name: str | None = None
    session_store: Any = Field(default=None, exclude=True)
    knowledge_store: Any = Field(default=None, exclude=True)
    knowledge_access_scope: Any = Field(default=None, exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)
    context_usage: ContextUsageState = Field(default_factory=ContextUsageState)
    pressure_overhead: ContextPressureOverhead = Field(default_factory=ContextPressureOverhead)
    count_input_tokens: Callable[[list[Message]], Awaitable[int | None]] | None = Field(
        default=None,
        exclude=True,
    )
    build_cache_prefix_request: Callable[[list[Message]], Awaitable[ModelRequest]] | None = Field(
        default=None,
        exclude=True,
    )
    force_compaction: StrictBool = False
    force_bounded_compaction: StrictBool = False
    compaction_instructions: str | None = Field(default=None, max_length=4096)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("session")
    @classmethod
    def copy_session_contract(cls, value: Session) -> Session:
        return copy_session(value)

    @field_validator("agent")
    @classmethod
    def copy_agent_contract(cls, value: AgentSpec) -> AgentSpec:
        return copy_agent_spec(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")

    @field_validator("context_usage")
    @classmethod
    def copy_context_usage(cls, value: ContextUsageState) -> ContextUsageState:
        return copy_context_usage_state(value)

    @field_validator("pressure_overhead")
    @classmethod
    def copy_pressure_overhead(
        cls,
        value: ContextPressureOverhead,
    ) -> ContextPressureOverhead:
        return copy_context_pressure_overhead(value)

    @field_validator("environment_name")
    @classmethod
    def validate_optional_environment_name(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "environment_name")

    @field_validator("interaction_id", "model_step_id")
    @classmethod
    def validate_optional_runtime_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("compaction_instructions")
    @classmethod
    def validate_optional_compaction_instructions(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, "compaction_instructions")


class ContextPolicy(ABC):
    """Builds the model-facing context for a runtime step.

    Policies may trim, summarize, replace tool results, or inject retrieved
    context. They must not be used as durable transcript storage.
    """

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        """Optional application-versioned identity for opaque policy behavior."""

        return None

    @abstractmethod
    async def build(self, request: ContextRequest) -> list[Message]:
        """Return provider-neutral messages for one model request."""


class ContextCompactionTelemetry(BaseModel):
    """Compaction telemetry that the runtime converts into events.

    ``MODEL_COMPLETED`` is allowed so a provider-backed compactor's
    summarization spend lands in the durable event log and is counted by
    usage, cost, budget, and run-limit accounting like any other model step.
    """

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def validate_context_event_type(cls, value: EventType) -> EventType:
        if value not in {
            EventType.CONTEXT_COMPACTION_STARTED,
            EventType.CONTEXT_COMPACTION_COMPLETED,
            EventType.CONTEXT_COMPACTION_FAILED,
            EventType.MODEL_COMPLETED,
        }:
            raise ValueError("Context compaction telemetry event_type is not supported.")
        return value

    @field_validator("payload", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "payload")


class _AutomaticCompactionLifecyclePhase(StrEnum):
    START_PUBLICATION = "start_publication"
    BUDGET_ADMISSION = "budget_admission"
    BUDGET_RESERVATION = "budget_reservation"
    REQUEST_FOOTPRINT_PUBLICATION = "request_footprint_publication"
    MODEL_START_PUBLICATION = "model_start_publication"
    PROVIDER_DISPATCH = "provider_dispatch"
    COMPLETION_PUBLICATION = "completion_publication"
    CHECKPOINT_INSTALLATION = "checkpoint_installation"


class _AutomaticCompactionFailureReason(StrEnum):
    PUBLICATION_TIMEOUT = "publication_timeout"
    PUBLICATION_FAILED = "publication_failed"
    ADMISSION_REJECTED = "admission_rejected"
    RESERVATION_FAILED = "reservation_failed"
    PROVIDER_FAILED = "provider_failed"
    CHECKPOINT_FAILED = "checkpoint_failed"
    CANCELLED = "cancelled"
    INTERNAL_FAILED = "internal_failed"


class _AutomaticCompactionDispatchDisposition(StrEnum):
    NOT_DISPATCHED = "not_dispatched"
    DISPATCHED = "dispatched"
    UNKNOWN = "unknown"


class _AutomaticCompactionRecoveryAction(StrEnum):
    RETRY_PUBLICATION = "retry_publication"
    RESUME_SESSION = "resume_session"
    STOP_SESSION = "stop_session"
    RECONCILE_COMPLETION = "reconcile_completion"
    FAIL_CLOSED = "fail_closed"


class _AutomaticCompactionFailureDisposition(BaseModel):
    """Bounded runtime-owned classification for automatic compaction failures."""

    model_config = ConfigDict(extra="forbid")

    phase: _AutomaticCompactionLifecyclePhase
    reason: _AutomaticCompactionFailureReason
    elapsed_ms: int = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    retryable: StrictBool
    provider_dispatch_disposition: _AutomaticCompactionDispatchDisposition
    recovery_action: _AutomaticCompactionRecoveryAction


_AUTOMATIC_COMPACTION_FAILURE_DISPOSITION_KEY = "_cayu_automatic_compaction_failure_disposition"


def _attach_automatic_compaction_failure_disposition(
    error: BaseException,
    disposition: _AutomaticCompactionFailureDisposition,
) -> None:
    """Attach only bounded runtime-owned evidence to an arbitrary failure."""

    if not isinstance(error, BaseException):
        raise TypeError("error must be a BaseException.")
    if type(disposition) is not _AutomaticCompactionFailureDisposition:
        raise TypeError("disposition must be an _AutomaticCompactionFailureDisposition.")
    error.__dict__[_AUTOMATIC_COMPACTION_FAILURE_DISPOSITION_KEY] = disposition.model_copy(
        deep=True
    )


def automatic_compaction_failure_disposition_payload(
    error: BaseException,
) -> dict[str, Any] | None:
    """Return detached lifecycle evidence carried by a compaction failure."""

    if not isinstance(error, BaseException):
        raise TypeError("error must be a BaseException.")
    candidate: BaseException | None = error
    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        disposition = candidate.__dict__.get(_AUTOMATIC_COMPACTION_FAILURE_DISPOSITION_KEY)
        if type(disposition) is _AutomaticCompactionFailureDisposition:
            return copy_durable_json_object(
                disposition.model_dump(mode="json"),
                "automatic_compaction_failure_disposition",
            )
        if isinstance(candidate, ContextBuildError):
            candidate = candidate.cause
        else:
            cause = candidate.__cause__
            candidate = cause if isinstance(cause, BaseException) else None
    return None


_COMPACTION_EVENT_TEXT_MAX_BYTES = 512
_COMPACTION_EVENT_INTEGER_MAX = 9_223_372_036_854_775_807
_COMPACTION_COVERAGE_MODES = frozenset(
    {"pending", "full", "partial_prefix", "no_progress", "failed"}
)
_COMPACTION_CHUNK_MODES = frozenset(
    {
        "pending",
        "failed",
        "single_request",
        "message_prefix",
        "hierarchical_atomic_unit",
        "digest_prefix",
        "digest_capacity_exhausted",
        "provider_native_exact",
        "custom",
    }
)


def _compaction_event_text(value: Any) -> str | None:
    if type(value) is not str or not value or value != value.strip():
        return None
    if any(
        0xD800 <= ord(char) <= 0xDFFF or ord(char) < 0x20 or ord(char) == 0x7F for char in value
    ):
        return None
    if len(value.encode("utf-8")) > _COMPACTION_EVENT_TEXT_MAX_BYTES:
        return None
    return value


def _compaction_event_integer(value: Any) -> int | None:
    if type(value) is not int or value < 0 or value > _COMPACTION_EVENT_INTEGER_MAX:
        return None
    return value


def _compaction_event_bool(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _compaction_usage_integer_field(
    value: dict[str, Any],
    key: str,
) -> tuple[int | None, bool]:
    if key not in value:
        return None, False
    bounded = _compaction_event_integer(value[key])
    return bounded, bounded is None


def _compaction_raw_usage(value: Any) -> tuple[dict[str, Any] | None, bool]:
    if value is None:
        return None, False
    if type(value) is not dict:
        return None, True
    raw_usage: dict[str, Any] = {}
    for key in (
        "input_tokens",
        "prompt_tokens",
        "output_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        bounded, invalid = _compaction_usage_integer_field(value, key)
        if invalid:
            return None, True
        if bounded is not None:
            raw_usage[key] = bounded
    for key, allowed_keys in (
        ("input_tokens_details", ("cached_tokens",)),
        ("prompt_tokens_details", ("cached_tokens",)),
        ("output_tokens_details", ("reasoning_tokens", "thinking_tokens")),
        ("completion_tokens_details", ("reasoning_tokens", "thinking_tokens")),
    ):
        details = value.get(key)
        if key in value and details is not None and type(details) is not dict:
            return None, True
        if type(details) is not dict:
            continue
        bounded_details: dict[str, int] = {}
        for detail_key in allowed_keys:
            if detail_key in details and details[detail_key] is None:
                continue
            bounded, invalid = _compaction_usage_integer_field(details, detail_key)
            if invalid:
                return None, True
            if bounded is not None:
                bounded_details[detail_key] = bounded
        if bounded_details:
            raw_usage[key] = bounded_details
    cache_creation = value.get("cache_creation")
    if (
        "cache_creation" in value
        and cache_creation is not None
        and type(cache_creation) is not dict
    ):
        return None, True
    if type(cache_creation) is dict:
        if len(cache_creation) > 16:
            return None, True
        cache_creation_total = 0
        for cache_value in cache_creation.values():
            bounded = _compaction_event_integer(cache_value)
            if bounded is None:
                return None, True
            cache_creation_total += bounded
            if cache_creation_total > _COMPACTION_EVENT_INTEGER_MAX:
                return None, True
        if cache_creation_total:
            raw_usage["cache_creation"] = {"bounded_total": cache_creation_total}
    return raw_usage or None, False


def _compaction_usage_metrics(
    payload: dict[str, Any],
) -> tuple[UsageMetrics | None, bool, BillingIdentity | None]:
    if payload.get("usage_normalization_failed") is True:
        return None, False, None
    supplied_metrics = payload.get("usage_metrics")
    provider_name = _compaction_event_text(
        supplied_metrics.get("provider_name") if type(supplied_metrics) is dict else None
    ) or _compaction_event_text(payload.get("provider_name"))
    requested_model = _compaction_event_text(
        supplied_metrics.get("requested_model") if type(supplied_metrics) is dict else None
    ) or _compaction_event_text(payload.get("requested_model"))
    model = _compaction_event_text(
        supplied_metrics.get("model") if type(supplied_metrics) is dict else None
    ) or _compaction_event_text(payload.get("model"))
    if "usage_metrics" in payload and type(supplied_metrics) is not dict:
        return None, True, None
    if type(supplied_metrics) is dict:
        metrics_identity = _compaction_billing_identity(
            supplied_metrics.get("billing_identity"),
            "usage_metrics.billing_identity",
        )
        sanitized_metrics: dict[str, Any] = {
            key: value
            for key, value in (
                ("provider_name", provider_name),
                ("requested_model", requested_model),
                ("model", model),
            )
            if value is not None
        }
        has_usage_counter = False
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_output_tokens",
        ):
            value, invalid = _compaction_usage_integer_field(supplied_metrics, key)
            if invalid:
                return None, True, metrics_identity
            if value is not None:
                sanitized_metrics[key] = value
                has_usage_counter = True
        supplied_cache = supplied_metrics.get("cache")
        if "cache" in supplied_metrics and type(supplied_cache) is not dict:
            return None, True, metrics_identity
        if type(supplied_cache) is dict:
            sanitized_cache = {}
            for key in (
                "read_tokens",
                "write_tokens",
                "write_5m_tokens",
                "write_1h_tokens",
                "write_unknown_ttl_tokens",
                "cached_input_tokens",
                "uncached_input_tokens",
            ):
                value, invalid = _compaction_usage_integer_field(supplied_cache, key)
                if invalid:
                    return None, True, metrics_identity
                if value is not None:
                    sanitized_cache[key] = value
            if sanitized_cache:
                sanitized_metrics["cache"] = sanitized_cache
                has_usage_counter = True
        if metrics_identity is not None:
            sanitized_metrics["billing_identity"] = metrics_identity
        if not has_usage_counter:
            return None, False, metrics_identity
        return UsageMetrics(**sanitized_metrics), False, metrics_identity
    raw_usage, invalid = _compaction_raw_usage(payload.get("usage"))
    if invalid:
        return None, True, None
    if raw_usage is None:
        return None, False, None
    return (
        usage_metrics_from_event_payload(
            {
                "provider_name": provider_name,
                "requested_model": requested_model,
                "model": model,
                "usage": raw_usage,
            }
        ),
        False,
        None,
    )


def _compaction_billing_identity(
    value: Any,
    field_name: str,
) -> BillingIdentity | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError(f"`{field_name}` must be a billing identity object.")
    try:
        return BillingIdentity.model_validate(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{field_name}` must be a valid billing identity.") from exc


def sanitize_context_compaction_telemetry(
    telemetry: ContextCompactionTelemetry,
) -> ContextCompactionTelemetry:
    """Project compactor-controlled telemetry onto Cayu's bounded public schema."""

    if type(telemetry) is not ContextCompactionTelemetry:
        raise TypeError(
            "Context compaction telemetry must be ContextCompactionTelemetry instances."
        )
    source = telemetry.payload
    payload: dict[str, Any] = {}
    if telemetry.event_type == EventType.MODEL_COMPLETED:
        metrics, invalid_metrics, metrics_identity = _compaction_usage_metrics(source)
        billing_identity = _compaction_billing_identity(
            source.get("billing_identity"),
            "billing_identity",
        )
        if (
            billing_identity is not None
            and metrics_identity is not None
            and billing_identity != metrics_identity
        ):
            raise ValueError("Compaction model.completed billing identities do not match.")
        if billing_identity is None:
            billing_identity = metrics_identity
        payload["purpose"] = ModelCompletionPurpose.CONTEXT_COMPACTION.value
        if billing_identity is not None:
            payload["billing_identity"] = billing_identity.model_dump(mode="json")
        raw_usage, invalid_raw_usage = _compaction_raw_usage(source.get("usage"))
        usage_metrics_rejected = source.get("usage_metrics_rejected")
        rejected_usage_evidence: dict[str, Any] | None = None
        invalid_rejected_usage = False
        if usage_metrics_rejected is True:
            rejected_usage_evidence, invalid_rejected_usage = _compaction_raw_usage(
                source.get("rejected_usage_evidence")
            )
        invalid_usage = (
            invalid_metrics
            or invalid_raw_usage
            or invalid_rejected_usage
            or usage_metrics_rejected is True
            or ("usage_metrics_rejected" in source and type(usage_metrics_rejected) is not bool)
        )
        if invalid_usage:
            metrics = None
            raw_usage = None
        if raw_usage is not None:
            payload["usage"] = raw_usage
        if usage_metrics_rejected is True:
            payload["usage_metrics_rejected"] = True
            if rejected_usage_evidence is not None:
                payload["rejected_usage_evidence"] = rejected_usage_evidence
        if metrics is not None:
            serialized_metrics = metrics.model_dump()
            serialized_metrics.pop("billing_identity", None)
            payload["usage_metrics"] = serialized_metrics
            for key in ("provider_name", "requested_model", "model"):
                value = getattr(metrics, key)
                if value is not None:
                    payload[key] = value
        else:
            supplied_metrics = source.get("usage_metrics")
            for key in ("provider_name", "requested_model", "model"):
                value = _compaction_event_text(
                    supplied_metrics.get(key) if type(supplied_metrics) is dict else None
                ) or _compaction_event_text(source.get(key))
                if value is not None:
                    payload[key] = value
        normalization_failed = source.get("usage_normalization_failed") is True
        if normalization_failed:
            payload["usage_normalization_failed"] = True
        for key in (
            "compactor",
            "compaction_outcome",
            "usage_unavailable_reason",
            "finish_reason",
            "error_type",
        ):
            value = _compaction_event_text(source.get(key))
            if value is not None:
                payload[key] = value
        if normalization_failed or invalid_usage:
            payload["usage_unavailable_reason"] = "invalid compaction usage telemetry"
        context_overflow = _compaction_event_bool(source.get("context_overflow"))
        if context_overflow is not None:
            payload["context_overflow"] = context_overflow
    else:
        payload["checkpoint"] = "context_compaction"
        integer_fields = [
            "previous_compacted_transcript_cursor",
            "newly_compacted_message_count",
            "recent_message_count",
            "requested_source_start",
            "requested_source_end",
            "represented_source_start",
            "represented_source_end",
            "represented_message_count",
            "chunk_count",
        ]
        integer_fields.append("compacted_transcript_cursor")
        if telemetry.event_type == EventType.CONTEXT_COMPACTION_COMPLETED:
            integer_fields.append("summary_chars")
        for key in integer_fields:
            value = _compaction_event_integer(source.get(key))
            if value is not None:
                payload[key] = value
        for key in ("compactor", "error_type"):
            value = _compaction_event_text(source.get(key))
            if value is not None:
                payload[key] = value
        coverage_mode = _compaction_event_text(source.get("coverage_mode"))
        if coverage_mode in _COMPACTION_COVERAGE_MODES:
            payload["coverage_mode"] = coverage_mode
        chunk_mode = _compaction_event_text(source.get("chunk_mode"))
        if chunk_mode is not None:
            payload["chunk_mode"] = (
                chunk_mode if chunk_mode in _COMPACTION_CHUNK_MODES else "custom"
            )
        for key in ("bounded_input", "compaction_failed"):
            value = _compaction_event_bool(source.get(key))
            if value is not None:
                payload[key] = value
        if telemetry.event_type == EventType.CONTEXT_COMPACTION_FAILED:
            phase = _compaction_event_text(source.get("phase"))
            if phase in {item.value for item in _AutomaticCompactionLifecyclePhase}:
                payload["phase"] = phase
            reason = _compaction_event_text(source.get("reason"))
            if reason in {item.value for item in _AutomaticCompactionFailureReason}:
                payload["reason"] = reason
            elapsed_ms = _compaction_event_integer(source.get("elapsed_ms"))
            if elapsed_ms is not None:
                payload["elapsed_ms"] = elapsed_ms
            retryable = _compaction_event_bool(source.get("retryable"))
            if retryable is not None:
                payload["retryable"] = retryable
            dispatch_disposition = _compaction_event_text(
                source.get("provider_dispatch_disposition")
            )
            if dispatch_disposition in {
                item.value for item in _AutomaticCompactionDispatchDisposition
            }:
                payload["provider_dispatch_disposition"] = dispatch_disposition
            recovery_action = _compaction_event_text(source.get("recovery_action"))
            if recovery_action in {item.value for item in _AutomaticCompactionRecoveryAction}:
                payload["recovery_action"] = recovery_action
    return ContextCompactionTelemetry(event_type=telemetry.event_type, payload=payload)


class ContextRecallTelemetry(BaseModel):
    """Automatic recall telemetry that the runtime converts into events."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("event_type")
    @classmethod
    def validate_context_event_type(cls, value: EventType) -> EventType:
        if value not in {
            EventType.AUTOMATIC_RECALL_STARTED,
            EventType.AUTOMATIC_RECALL_COMPLETED,
            EventType.AUTOMATIC_RECALL_FAILED,
            EventType.AUTOMATIC_RECALL_ADMITTED,
        }:
            raise ValueError("Context recall telemetry event_type is not supported.")
        return value

    @field_validator("payload", mode="before")
    @classmethod
    def copy_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "payload")


def copy_context_compaction_telemetry(
    telemetry: ContextCompactionTelemetry,
) -> ContextCompactionTelemetry:
    if type(telemetry) is not ContextCompactionTelemetry:
        raise TypeError(
            "Context compaction telemetry must be ContextCompactionTelemetry instances."
        )
    return ContextCompactionTelemetry(
        event_type=telemetry.event_type,
        payload=copy_json_value(telemetry.payload, "payload"),
    )


def copy_context_recall_telemetry(
    telemetry: ContextRecallTelemetry,
) -> ContextRecallTelemetry:
    if type(telemetry) is not ContextRecallTelemetry:
        raise TypeError("Context recall telemetry must be ContextRecallTelemetry instances.")
    return ContextRecallTelemetry(
        event_type=telemetry.event_type,
        payload=copy_json_value(telemetry.payload, "payload"),
    )


_ContextRecallTelemetryPublisher = Callable[
    [ContextRecallTelemetry],
    Awaitable[None],
]
_CONTEXT_RECALL_TELEMETRY_PUBLISHER: ContextVar[_ContextRecallTelemetryPublisher | None] = (
    ContextVar(
        "cayu_context_recall_telemetry_publisher",
        default=None,
    )
)


@contextmanager
def _context_recall_telemetry_publisher_scope(
    publisher: _ContextRecallTelemetryPublisher | None,
) -> Iterator[None]:
    token = _CONTEXT_RECALL_TELEMETRY_PUBLISHER.set(publisher)
    try:
        yield
    finally:
        _CONTEXT_RECALL_TELEMETRY_PUBLISHER.reset(token)


async def _publish_or_record_recall_telemetry(
    telemetry: ContextRecallTelemetry,
    *,
    recorded: list[ContextRecallTelemetry],
) -> None:
    if telemetry.event_type not in {
        EventType.AUTOMATIC_RECALL_STARTED,
        EventType.AUTOMATIC_RECALL_COMPLETED,
        EventType.AUTOMATIC_RECALL_FAILED,
    }:
        raise TypeError("Only automatic-recall operation telemetry can publish immediately.")
    publisher = _CONTEXT_RECALL_TELEMETRY_PUBLISHER.get()
    if publisher is None:
        recorded.append(copy_context_recall_telemetry(telemetry))
        return
    await publisher(copy_context_recall_telemetry(telemetry))


class ContextBuildResult(BaseModel):
    """Runtime-managed context result that may include checkpoint updates."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message]
    checkpoint: dict[str, Any] | None = None
    checkpoint_event_payload: dict[str, Any] | None = None
    compaction_telemetry: list[ContextCompactionTelemetry] = Field(default_factory=list)
    recall_telemetry: list[ContextRecallTelemetry] = Field(default_factory=list)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("compaction_telemetry")
    @classmethod
    def copy_compaction_telemetry(cls, value):
        return [copy_context_compaction_telemetry(item) for item in value]

    @field_validator("recall_telemetry")
    @classmethod
    def copy_recall_telemetry(cls, value):
        return [copy_context_recall_telemetry(item) for item in value]

    @field_validator("checkpoint", "checkpoint_event_payload", mode="before")
    @classmethod
    def copy_optional_json_data(cls, value, info):
        if value is None:
            return None
        return copy_json_value(value, info.field_name)


def clear_context_build_result_payload(result: ContextBuildResult) -> None:
    """Discard extension-provided data before a validation error crosses the boundary."""

    if not isinstance(result, ContextBuildResult):
        raise TypeError("result must be a ContextBuildResult.")
    result.messages.clear()
    if result.checkpoint is not None:
        result.checkpoint.clear()
    if result.checkpoint_event_payload is not None:
        result.checkpoint_event_payload.clear()
    result.compaction_telemetry.clear()
    result.recall_telemetry.clear()


def _copy_secret_free_context_checkpoint_evidence(
    checkpoint: dict[str, Any] | None,
    checkpoint_event_payload: dict[str, Any] | None,
    *,
    redactor: SecretRedactor,
    checkpoint_field_name: str,
    checkpoint_event_payload_field_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Copy the two checkpoint representations through one validation seam."""

    private_authority_keys = {
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
        SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
    }
    if checkpoint is not None and private_authority_keys.intersection(checkpoint):
        raise ValueError(
            f"{checkpoint_field_name} cannot contain private invocation lifecycle authority."
        )
    safe_checkpoint = (
        None
        if checkpoint is None
        else require_secret_free_durable_object(
            checkpoint,
            redactor=redactor,
            field_name=checkpoint_field_name,
        )
    )
    safe_checkpoint_event_payload = (
        None
        if checkpoint_event_payload is None
        else _require_secret_free_checkpoint_event_payload(
            checkpoint_event_payload,
            redactor=redactor,
            field_name=checkpoint_event_payload_field_name,
        )
    )
    return safe_checkpoint, safe_checkpoint_event_payload


def sanitize_context_build_result_checkpoint(
    result: ContextBuildResult,
    *,
    redactor: SecretRedactor,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Copy safe checkpoint evidence or consume the rejected extension result."""

    if not isinstance(result, ContextBuildResult):
        raise TypeError("result must be a ContextBuildResult.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    checkpoint = result.checkpoint
    checkpoint_event_payload = result.checkpoint_event_payload
    try:
        safe_checkpoint, safe_checkpoint_event_payload = (
            _copy_secret_free_context_checkpoint_evidence(
                checkpoint,
                checkpoint_event_payload,
                redactor=redactor,
                checkpoint_field_name="ContextBuildResult.checkpoint",
                checkpoint_event_payload_field_name=("ContextBuildResult.checkpoint_event_payload"),
            )
        )
    except BaseException:
        clear_context_build_result_payload(result)
        checkpoint = None
        checkpoint_event_payload = None
        del result
        raise
    return safe_checkpoint, safe_checkpoint_event_payload


class ContextBuildError(RuntimeError):
    """Context build failure with compaction telemetry to emit first."""

    def __init__(
        self,
        message: str,
        *,
        compaction_telemetry: list[ContextCompactionTelemetry],
        recall_telemetry: list[ContextRecallTelemetry] | None = None,
        checkpoint: dict[str, Any] | None = None,
        checkpoint_event_payload: dict[str, Any] | None = None,
        cause: Exception,
    ) -> None:
        super().__init__(message)
        self.compaction_telemetry = tuple(
            copy_context_compaction_telemetry(item) for item in compaction_telemetry
        )
        self.recall_telemetry = tuple(
            copy_context_recall_telemetry(item)
            for item in ([] if recall_telemetry is None else recall_telemetry)
        )
        self.checkpoint = None if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        self.checkpoint_event_payload = (
            None
            if checkpoint_event_payload is None
            else copy_json_value(checkpoint_event_payload, "checkpoint_event_payload")
        )
        self.cause = cause


class _ContextCountAuthorityError(RuntimeError):
    """Carry a runtime-owned count authority failure across policy fallback."""

    def __init__(self, cause: Exception) -> None:
        super().__init__("Context token counting was rejected by runtime authority.")
        self.cause = cause


def sanitize_context_build_error_checkpoint(
    error: ContextBuildError,
    *,
    redactor: SecretRedactor,
    field_name: str = "ContextBuildError.checkpoint",
) -> None:
    """Retain safe failure progress while discarding an unsafe checkpoint update."""

    if not isinstance(error, ContextBuildError):
        raise TypeError("error must be a ContextBuildError.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    checkpoint = error.checkpoint
    checkpoint_event_payload = error.checkpoint_event_payload
    try:
        safe_checkpoint, safe_checkpoint_event_payload = (
            _copy_secret_free_context_checkpoint_evidence(
                checkpoint,
                checkpoint_event_payload,
                redactor=redactor,
                checkpoint_field_name=field_name,
                checkpoint_event_payload_field_name=("ContextBuildError.checkpoint_event_payload"),
            )
        )
    except Exception:
        if checkpoint is not None:
            checkpoint.clear()
        if checkpoint_event_payload is not None:
            checkpoint_event_payload.clear()
        error.checkpoint = None
        error.checkpoint_event_payload = None
        error.add_note(
            "The context policy's failure checkpoint update was discarded because "
            "it contained a workload secret."
        )
        return
    error.checkpoint = safe_checkpoint
    error.checkpoint_event_payload = safe_checkpoint_event_payload


def _require_secret_free_checkpoint_event_payload(
    value: dict[str, Any],
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> dict[str, Any]:
    """Copy and validate checkpoint event evidence without retaining rejected keys."""

    copied = copy_json_value(value, field_name)
    if type(copied) is not dict:
        raise AssertionError("Checkpoint event payload copied as a non-object.")
    contains_secret = redactor.redact_json_values(
        copied
    ) != copied or _json_object_keys_contain_secret(
        copied,
        redactor=redactor,
        preserve_keys=_CONTEXT_CHECKPOINT_EVENT_STRUCTURE_KEYS,
    )
    if not contains_secret:
        return copied
    copied.clear()
    value.clear()
    value = {}
    raise ValueError(f"{field_name} contains a workload secret; refusing to publish it.")


def _json_object_keys_contain_secret(
    value: Any,
    *,
    redactor: SecretRedactor,
    preserve_keys: frozenset[str],
    structural_boundary: bool = True,
) -> bool:
    """Return whether a JSON tree has a secret-bearing non-structural key."""

    if value is None or type(value) in {str, bool, int, float}:
        return False
    if type(value) is list:
        return any(
            _json_object_keys_contain_secret(
                item,
                redactor=redactor,
                preserve_keys=preserve_keys,
                structural_boundary=False,
            )
            for item in value
        )
    if type(value) is dict:
        return any(
            (
                (not structural_boundary or key not in preserve_keys)
                and redactor.redact_text(key) != key
            )
            or _json_object_keys_contain_secret(
                item,
                redactor=redactor,
                preserve_keys=preserve_keys,
                structural_boundary=False,
            )
            for key, item in value.items()
        )
    raise AssertionError("Checkpoint event payload contains non-JSON-compatible data.")


class _ContextBuildTerminationDiagnostics:
    """Immutable context evidence carried by an unwrapped fatal signal."""

    def __init__(self, compaction_telemetry: list[ContextCompactionTelemetry]) -> None:
        self.compaction_telemetry = tuple(
            copy_context_compaction_telemetry(item) for item in compaction_telemetry
        )


_CONTEXT_BUILD_TERMINATION_DIAGNOSTICS_KEY = "_cayu_context_build_termination_diagnostics"


def _attach_context_build_termination_diagnostics(
    error: BaseException,
    *,
    compaction_telemetry: list[ContextCompactionTelemetry],
) -> None:
    """Attach evidence without wrapping cancellation, abandonment, or fatal signals."""

    previous = error.__dict__.get(_CONTEXT_BUILD_TERMINATION_DIAGNOSTICS_KEY)
    existing = (
        list(previous.compaction_telemetry)
        if isinstance(previous, _ContextBuildTerminationDiagnostics)
        else []
    )
    error.__dict__[_CONTEXT_BUILD_TERMINATION_DIAGNOSTICS_KEY] = (
        _ContextBuildTerminationDiagnostics([*existing, *compaction_telemetry])
    )


def context_build_termination_compaction_telemetry(
    error: BaseException,
) -> tuple[ContextCompactionTelemetry, ...]:
    """Return detached compaction evidence attached to an authoritative signal."""

    diagnostics = error.__dict__.get(_CONTEXT_BUILD_TERMINATION_DIAGNOSTICS_KEY)
    if not isinstance(diagnostics, _ContextBuildTerminationDiagnostics):
        return ()
    return tuple(
        copy_context_compaction_telemetry(item) for item in diagnostics.compaction_telemetry
    )


class RuntimeManagedContextPolicy(ContextPolicy):
    """Context policy whose checkpoint writes are owned by the runtime."""

    @abstractmethod
    async def build_with_checkpoint(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        """Return model-facing context and optional checkpoint updates."""

    async def build(self, request: ContextRequest) -> list[Message]:
        result = await self.build_with_checkpoint(request, checkpoint=None)
        return result.messages


class DefaultContextPolicy(ContextPolicy):
    """Default policy that sends transcript context with bounded file attachments."""

    def __init__(self, *, max_attachment_results: int = 1) -> None:
        self.max_attachment_results = _validate_max_attachment_results(max_attachment_results)

    async def build(self, request: ContextRequest) -> list[Message]:
        return strip_old_file_attachments(
            request.messages,
            max_attachment_results=self.max_attachment_results,
        )


class MessageWindowContextPolicy(ContextPolicy):
    """Built-in policy that keeps a valid recent message window."""

    def __init__(
        self,
        *,
        max_messages: int,
        preserve_system: bool = True,
        max_attachment_results: int = 1,
    ) -> None:
        if type(max_messages) is not int:
            raise TypeError("max_messages must be an integer.")
        if type(preserve_system) is not bool:
            raise TypeError("preserve_system must be a bool.")
        if max_messages < 1:
            raise ValueError("max_messages must be greater than zero.")
        self.max_messages = max_messages
        self.preserve_system = preserve_system
        self.max_attachment_results = _validate_max_attachment_results(max_attachment_results)

    async def build(self, request: ContextRequest) -> list[Message]:
        trimmed = trim_context_messages(
            request.messages,
            max_messages=self.max_messages,
            preserve_system=self.preserve_system,
        )
        return strip_old_file_attachments(
            trimmed,
            max_attachment_results=self.max_attachment_results,
        )


class RecentTurnsContextPolicy(ContextPolicy):
    """Built-in policy that keeps recent user turns and complete tool rounds."""

    def __init__(
        self,
        *,
        max_user_turns: int,
        preserve_system: bool = True,
        max_attachment_results: int = 1,
    ) -> None:
        if type(max_user_turns) is not int:
            raise TypeError("max_user_turns must be an integer.")
        if type(preserve_system) is not bool:
            raise TypeError("preserve_system must be a bool.")
        if max_user_turns < 1:
            raise ValueError("max_user_turns must be greater than zero.")
        self.max_user_turns = max_user_turns
        self.preserve_system = preserve_system
        self.max_attachment_results = _validate_max_attachment_results(max_attachment_results)

    async def build(self, request: ContextRequest) -> list[Message]:
        trimmed = trim_context_turns(
            request.messages,
            max_user_turns=self.max_user_turns,
            preserve_system=self.preserve_system,
        )
        return strip_old_file_attachments(
            trimmed,
            max_attachment_results=self.max_attachment_results,
        )


class UsageTriggeredContextPolicy(RuntimeManagedContextPolicy):
    """Switch context policy after previous actual provider usage crosses a threshold.

    The runtime populates ``ContextRequest.context_usage`` from the previous completed
    model call in the same session. This wrapper keeps normal context behavior below the
    configured thresholds and delegates to ``triggered_policy`` on the next call once a
    threshold is reached. By default the trigger is sticky and stored in the session
    checkpoint so later low-usage calls continue using ``triggered_policy``.
    """

    def __init__(
        self,
        *,
        triggered_policy: ContextPolicy,
        base_policy: ContextPolicy | None = None,
        min_input_tokens: int | None = None,
        trigger_estimated_context_tokens: int | None = None,
        reserved_output_tokens: int = 0,
        verify_estimate_with_provider_count: bool = False,
        provider_count_threshold_ratio: float = 0.9,
        provider_count_min_delta_tokens: int | None = None,
        min_total_tokens: int | None = None,
        sticky: bool = True,
    ) -> None:
        if base_policy is None:
            self.base_policy = DefaultContextPolicy()
        elif isinstance(base_policy, ContextPolicy):
            self.base_policy = base_policy
        else:
            raise TypeError("base_policy must be a ContextPolicy.")
        if not isinstance(triggered_policy, ContextPolicy):
            raise TypeError("triggered_policy must be a ContextPolicy.")
        self.triggered_policy = triggered_policy
        self.min_input_tokens = _validate_optional_positive_int(
            min_input_tokens,
            "min_input_tokens",
        )
        self.trigger_estimated_context_tokens = _validate_optional_positive_int(
            trigger_estimated_context_tokens,
            "trigger_estimated_context_tokens",
        )
        self.reserved_output_tokens = _validate_nonnegative_int(
            reserved_output_tokens,
            "reserved_output_tokens",
        )
        if type(verify_estimate_with_provider_count) is not bool:
            raise TypeError("verify_estimate_with_provider_count must be a bool.")
        self.verify_estimate_with_provider_count = verify_estimate_with_provider_count
        self.provider_count_threshold_ratio = _validate_ratio(
            provider_count_threshold_ratio,
            "provider_count_threshold_ratio",
        )
        self.provider_count_min_delta_tokens = _validate_optional_positive_int(
            provider_count_min_delta_tokens,
            "provider_count_min_delta_tokens",
        )
        self.min_total_tokens = _validate_optional_positive_int(
            min_total_tokens,
            "min_total_tokens",
        )
        if (
            self.min_input_tokens is None
            and self.trigger_estimated_context_tokens is None
            and self.min_total_tokens is None
        ):
            raise ValueError("At least one usage threshold must be configured.")
        if self.trigger_estimated_context_tokens is not None and isinstance(
            self.base_policy, RuntimeManagedContextPolicy
        ):
            raise ValueError(
                "Estimated context triggers require a side-effect-free base_policy. "
                "Do not use RuntimeManagedContextPolicy as the base policy because "
                "the base policy must be evaluated before deciding whether to switch."
            )
        if type(sticky) is not bool:
            raise TypeError("sticky must be a bool.")
        self.sticky = sticky

    async def build_with_checkpoint(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        checkpoint_state = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        previous = _usage_triggered_checkpoint(checkpoint_state)
        already_triggered = self.sticky and previous is not None
        threshold_triggered = self._actual_usage_is_triggered(request)
        use_triggered = already_triggered or threshold_triggered
        if use_triggered:
            result = await _build_policy_context(
                self.triggered_policy,
                request,
                checkpoint=checkpoint_state,
            )
        elif self.trigger_estimated_context_tokens is not None:
            base_result = await _build_policy_context(
                self.base_policy,
                request,
                checkpoint=checkpoint_state,
            )
            estimate = _estimate_model_facing_context_pressure(
                request=request,
                messages=base_result.messages,
                reserved_output_tokens=self.reserved_output_tokens,
            )
            estimate = await self._maybe_verify_estimate_with_provider_count(
                request=request,
                messages=base_result.messages,
                estimate=estimate,
            )
            request = request.model_copy(
                update={
                    "context_usage": request.context_usage.model_copy(
                        update={"input_pressure": estimate}
                    )
                }
            )
            threshold_triggered = (
                _context_window_tokens_for_decision(estimate)
                >= self.trigger_estimated_context_tokens
            )
            use_triggered = threshold_triggered
            if use_triggered:
                result = await _build_policy_context(
                    self.triggered_policy,
                    request,
                    checkpoint=checkpoint_state,
                )
            else:
                result = base_result
        else:
            result = await _build_policy_context(
                self.base_policy,
                request,
                checkpoint=checkpoint_state,
            )
        if not self.sticky or not use_triggered:
            return result
        if result.checkpoint is None and result.checkpoint_event_payload is not None:
            return result

        marker = (
            previous
            if previous is not None
            else _usage_triggered_checkpoint_marker(
                policy=self,
                request=request,
            )
        )
        if result.checkpoint is None and previous is not None:
            return result

        checkpoint_update = (
            copy_json_value(result.checkpoint, "checkpoint")
            if result.checkpoint is not None
            else copy_json_value(checkpoint_state, "checkpoint")
        )
        checkpoint_update[_USAGE_TRIGGERED_CHECKPOINT_KEY] = marker
        checkpoint_event_payload = result.checkpoint_event_payload
        if checkpoint_event_payload is None and previous is None:
            checkpoint_event_payload = _usage_triggered_checkpoint_event_payload(marker)
        if checkpoint_event_payload is None:
            return result.model_copy(update={"checkpoint": checkpoint_update})
        return result.model_copy(
            update={
                "checkpoint": checkpoint_update,
                "checkpoint_event_payload": checkpoint_event_payload,
            }
        )

    def _actual_usage_is_triggered(self, request: ContextRequest) -> bool:
        usage = request.context_usage
        return (
            self.min_input_tokens is not None
            and usage.last_input_tokens is not None
            and usage.last_input_tokens >= self.min_input_tokens
        ) or (
            self.min_total_tokens is not None
            and usage.last_total_tokens is not None
            and usage.last_total_tokens >= self.min_total_tokens
        )

    async def _maybe_verify_estimate_with_provider_count(
        self,
        *,
        request: ContextRequest,
        messages: list[Message],
        estimate: ContextPressureEstimate,
    ) -> ContextPressureEstimate:
        if (
            not self.verify_estimate_with_provider_count
            or request.count_input_tokens is None
            or self.trigger_estimated_context_tokens is None
        ):
            return estimate
        near_threshold = estimate.estimated_context_window_tokens >= math.ceil(
            self.trigger_estimated_context_tokens * self.provider_count_threshold_ratio
        )
        large_delta = (
            self.provider_count_min_delta_tokens is not None
            and estimate.estimated_delta_input_tokens >= self.provider_count_min_delta_tokens
        )
        if not near_threshold and not large_delta:
            return estimate
        try:
            input_tokens = await request.count_input_tokens(messages)
        except _ContextCountAuthorityError as authority_error:
            # Only the runtime-owned wrapper authenticates this as an
            # authority failure. An identical exception raised by provider
            # code remains an optional counter failure below.
            raise authority_error.cause from None
        except Exception:
            return estimate
        if input_tokens is None:
            return estimate
        input_tokens = _validate_nonnegative_int(input_tokens, "provider input token count")
        window_tokens = input_tokens + self.reserved_output_tokens
        return estimate.model_copy(
            update={
                "confidence": "high",
                "provider_count_input_tokens": input_tokens,
                "provider_count_context_window_tokens": window_tokens,
            }
        )


def _estimate_model_facing_context_pressure(
    *,
    request: ContextRequest,
    messages: list[Message],
    reserved_output_tokens: int = 0,
) -> ContextPressureEstimate:
    estimator = ObservedDeltaContextEstimator()
    if messages == request.messages:
        return estimator.estimate_anchored_request(
            usage=request.context_usage,
            messages=messages,
            overhead=request.pressure_overhead,
            reserved_output_tokens=reserved_output_tokens,
        )
    return estimator.estimate_full_request(
        usage=request.context_usage,
        messages=messages,
        overhead=request.pressure_overhead,
        reserved_output_tokens=reserved_output_tokens,
    )


def estimate_model_request_context_pressure(
    *,
    model_request: ModelRequest,
    image_min_tokens: int = _DEFAULT_ESTIMATE_IMAGE_MIN_TOKENS,
    document_min_tokens: int = _DEFAULT_ESTIMATE_DOCUMENT_MIN_TOKENS,
    document_bytes_per_token: int = _DEFAULT_ESTIMATE_BINARY_BYTES_PER_TOKEN,
    tool_schema_chars_per_token: int = _DEFAULT_ESTIMATE_TOOL_SCHEMA_CHARS_PER_TOKEN,
    reserved_output_tokens: int = 0,
    estimator: ObservedDeltaContextEstimator | None = None,
) -> ContextPressureEstimate:
    if type(model_request) is not ModelRequest:
        raise TypeError("model_request must be a ModelRequest.")
    if estimator is None:
        estimator = ObservedDeltaContextEstimator()
    return estimator.estimate_full_request(
        usage=ContextUsageState(),
        messages=model_request.messages,
        overhead=ContextPressureOverhead(
            tools=[
                *model_request.tools,
                *(
                    ()
                    if model_request.targeted_tool_projection is None
                    else model_request.targeted_tool_projection.tools
                ),
            ],
            request_options=model_request.options,
            image_min_tokens=image_min_tokens,
            document_min_tokens=document_min_tokens,
            document_bytes_per_token=document_bytes_per_token,
            tool_schema_chars_per_token=tool_schema_chars_per_token,
        ),
        reserved_output_tokens=reserved_output_tokens,
    )


def _context_window_tokens_for_decision(estimate: ContextPressureEstimate) -> int:
    if estimate.provider_count_context_window_tokens is not None:
        return estimate.provider_count_context_window_tokens
    return estimate.estimated_context_window_tokens


def _usage_triggered_checkpoint_marker(
    *,
    policy: UsageTriggeredContextPolicy,
    request: ContextRequest,
) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "version": _USAGE_TRIGGERED_CHECKPOINT_VERSION,
        "min_input_tokens": policy.min_input_tokens,
        "min_total_tokens": policy.min_total_tokens,
        "last_input_tokens": request.context_usage.last_input_tokens,
        "last_total_tokens": request.context_usage.last_total_tokens,
    }
    if policy.trigger_estimated_context_tokens is not None:
        marker["trigger_estimated_context_tokens"] = policy.trigger_estimated_context_tokens
        marker["reserved_output_tokens"] = policy.reserved_output_tokens
        marker["last_transcript_cursor"] = request.context_usage.last_transcript_cursor
        if request.context_usage.input_pressure is not None:
            marker["estimated_context_input_tokens"] = (
                request.context_usage.input_pressure.estimated_context_input_tokens
            )
            marker["estimated_context_window_tokens"] = (
                request.context_usage.input_pressure.estimated_context_window_tokens
            )
            marker["estimated_delta_input_tokens"] = (
                request.context_usage.input_pressure.estimated_delta_input_tokens
            )
            if request.context_usage.input_pressure.provider_count_input_tokens is not None:
                marker["provider_count_input_tokens"] = (
                    request.context_usage.input_pressure.provider_count_input_tokens
                )
            if (
                request.context_usage.input_pressure.provider_count_context_window_tokens
                is not None
            ):
                marker["provider_count_context_window_tokens"] = (
                    request.context_usage.input_pressure.provider_count_context_window_tokens
                )
    return marker


def _usage_triggered_checkpoint_event_payload(marker: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "checkpoint": _USAGE_TRIGGERED_CHECKPOINT_KEY,
        "min_input_tokens": marker.get("min_input_tokens"),
        "min_total_tokens": marker.get("min_total_tokens"),
        "last_input_tokens": marker.get("last_input_tokens"),
        "last_total_tokens": marker.get("last_total_tokens"),
    }
    for key in (
        "trigger_estimated_context_tokens",
        "reserved_output_tokens",
        "estimated_context_input_tokens",
        "estimated_context_window_tokens",
        "estimated_delta_input_tokens",
        "provider_count_input_tokens",
        "provider_count_context_window_tokens",
        "last_transcript_cursor",
    ):
        if key in marker:
            payload[key] = marker.get(key)
    return payload


class CompactionRequest(BaseModel):
    """Input passed to a compactor when older context needs summarizing.

    ``messages`` is only the newly compactable transcript delta.
    ``context_messages`` is the current full provider-facing projection for
    compatibility with custom compactors. ``cache_prefix_request`` is the exact
    runtime request shape available to cache-aware compactors.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session: Session
    agent: AgentSpec
    messages: list[Message]
    existing_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    context_messages: list[Message] = Field(default_factory=list)
    cache_prefix_request: ModelRequest | None = None
    force_bounded_compaction: StrictBool = False
    instructions: str | None = Field(default=None, max_length=4096)

    @field_validator("messages")
    @classmethod
    def copy_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("session")
    @classmethod
    def copy_session_contract(cls, value: Session) -> Session:
        return copy_session(value)

    @field_validator("agent")
    @classmethod
    def copy_agent_contract(cls, value: AgentSpec) -> AgentSpec:
        return copy_agent_spec(value)

    @field_validator("context_messages")
    @classmethod
    def copy_context_messages(cls, value):
        return [copy_message(message) for message in value]

    @field_validator("cache_prefix_request")
    @classmethod
    def copy_cache_prefix_request(cls, value: ModelRequest | None) -> ModelRequest | None:
        if value is None:
            return None
        return value.model_copy(deep=True)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")

    @field_validator("existing_summary")
    @classmethod
    def validate_optional_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_nonblank(value, "existing_summary")

    @field_validator("instructions")
    @classmethod
    def validate_optional_instructions(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_nonblank(value, "instructions")


class CompactionPrompt(BaseModel):
    """A custom compaction prompt with explicit source coverage."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    prompt: str
    covered_message_count: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        return require_durable_nonblank(value, "prompt")


CompactionPromptBuilder = Callable[[CompactionRequest], CompactionPrompt]

_COMPACTION_PROMPT_FIELDS = frozenset({"prompt", "covered_message_count"})
_COMPACTION_RESULT_FIELDS = frozenset(
    {
        "summary",
        "covered_message_count",
        "represented_existing_summary_sha256",
        "source_chunk_count",
        "source_chunk_mode",
        "bounded_input",
        "progress_exhausted",
        "progress_key",
        "metadata",
        "model_completed_payloads",
    }
)


def _exact_compaction_model_fields(
    value: object,
    *,
    expected_type: type[BaseModel],
    expected_fields: frozenset[str],
    error_message: str,
) -> dict[str, Any]:
    """Read an exact Pydantic model without hostile mapping-key lookup."""

    if type(value) is not expected_type:
        raise TypeError(error_message)
    try:
        raw_fields = object.__getattribute__(value, "__dict__")
    except BaseException:
        raise TypeError(error_message) from None
    if type(raw_fields) is not dict:
        raise TypeError(error_message)
    fields: dict[str, Any] = {}
    for key, field_value in raw_fields.items():
        if type(key) is not str or key not in expected_fields:
            raise TypeError(error_message)
        fields[key] = field_value
    if fields.keys() != expected_fields:
        raise TypeError(error_message)
    return fields


def _detach_compaction_prompt(value: object) -> CompactionPrompt:
    """Revalidate an extension-owned prompt without retaining its object graph."""

    error_message = (
        "Custom compaction prompt builders must return CompactionPrompt "
        "with explicit source coverage."
    )
    fields = _exact_compaction_model_fields(
        value,
        expected_type=CompactionPrompt,
        expected_fields=_COMPACTION_PROMPT_FIELDS,
        error_message=error_message,
    )
    return CompactionPrompt(
        prompt=fields["prompt"],
        covered_message_count=fields["covered_message_count"],
    )


def _validate_compaction_summary(value: str) -> str:
    """Validate summary text against the complete durable checkpoint boundary."""

    return require_durable_nonblank(value, "summary")


def _compaction_summary_sha256(summary: str) -> str:
    return hashlib.sha256(summary.encode("utf-8")).hexdigest()


class CompactionResult(BaseModel):
    """Compacted representation of older model-facing context.

    ``covered_message_count`` declares the contiguous new-source prefix
    represented by ``summary``. When a request carries ``existing_summary``, a
    positive-coverage result must bind that exact prior representation through
    ``represented_existing_summary_sha256``. A zero-coverage result must return
    the existing summary unchanged. ``source_chunk_count`` and
    ``source_chunk_mode`` describe the bounded source work used to produce the
    result; they never substitute for either explicit coverage claim.

    ``model_completed_payloads`` carries one event-ready ``model.completed``
    payload per provider call the compactor made, so the runtime can account
    for summarization spend in usage, cost, budget, and run-limit tracking.
    Runtime-created payloads may temporarily carry ``compaction_attempt_id``;
    wrapping compactors must preserve it so Cayu can correlate recovered calls.
    Cayu removes the correlation field before emitting public events.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    summary: str
    covered_message_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    represented_existing_summary_sha256: str | None = None
    source_chunk_count: StrictInt = Field(
        default=1,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    source_chunk_mode: str = "single_request"
    bounded_input: StrictBool = False
    progress_exhausted: StrictBool = False
    progress_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    model_completed_payloads: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        return _validate_compaction_summary(value)

    @field_validator("represented_existing_summary_sha256")
    @classmethod
    def validate_represented_existing_summary_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            len(value) != 64
            or value != value.lower()
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(
                "represented_existing_summary_sha256 must be a lowercase SHA-256 digest."
            )
        return value

    @field_validator("source_chunk_mode")
    @classmethod
    def validate_source_chunk_mode(cls, value: str) -> str:
        value = require_clean_nonblank(value, "source_chunk_mode")
        if _compaction_event_text(value) is None:
            raise ValueError(
                "source_chunk_mode must contain valid Unicode without control characters "
                f"and be at most {_COMPACTION_EVENT_TEXT_MAX_BYTES} UTF-8 bytes."
            )
        return value

    @field_validator("progress_key")
    @classmethod
    def validate_progress_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, "progress_key")
        if _compaction_event_text(value) is None:
            raise ValueError(
                "progress_key must contain valid Unicode without control characters "
                f"and be at most {_COMPACTION_EVENT_TEXT_MAX_BYTES} UTF-8 bytes."
            )
        return value

    @model_validator(mode="after")
    def validate_progress_exhaustion(self) -> CompactionResult:
        if self.progress_exhausted:
            if self.covered_message_count != 0:
                raise ValueError("Exhausted compaction progress must report zero coverage.")
            if self.progress_key is None:
                raise ValueError("Exhausted compaction progress requires progress_key.")
        elif self.progress_key is not None:
            raise ValueError("progress_key requires progress_exhausted=true.")
        return self

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")

    @field_validator("model_completed_payloads", mode="before")
    @classmethod
    def copy_model_completed_payloads(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return copy_durable_json_value(value, "model_completed_payloads")


def _snapshot_compaction_result(value: object) -> dict[str, Any]:
    """Snapshot raw fields and copy completion evidence before full validation."""

    error_message = "Context compactors must return CompactionResult."
    fields = _exact_compaction_model_fields(
        value,
        expected_type=CompactionResult,
        expected_fields=_COMPACTION_RESULT_FIELDS,
        error_message=error_message,
    )
    payloads = copy_durable_json_value(
        fields["model_completed_payloads"],
        "model_completed_payloads",
    )
    if type(payloads) is not list or any(type(payload) is not dict for payload in payloads):
        raise TypeError("CompactionResult.model_completed_payloads must be a list of objects.")
    fields["model_completed_payloads"] = payloads
    return fields


def _detach_compaction_result(
    fields: dict[str, Any],
) -> CompactionResult:
    """Revalidate an extension-owned result without retaining its object graph."""

    return CompactionResult(
        summary=fields["summary"],
        covered_message_count=fields["covered_message_count"],
        represented_existing_summary_sha256=fields["represented_existing_summary_sha256"],
        source_chunk_count=fields["source_chunk_count"],
        source_chunk_mode=fields["source_chunk_mode"],
        bounded_input=fields["bounded_input"],
        progress_exhausted=fields["progress_exhausted"],
        progress_key=fields["progress_key"],
        metadata=fields["metadata"],
        model_completed_payloads=fields["model_completed_payloads"],
    )


class ContextCompactor(ABC):
    """Summarizes older context into durable checkpoint data."""

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        """Optional application-versioned identity for opaque compaction behavior."""

        return None

    def provider_budget_identity(self, session: Session) -> tuple[str, str] | None:
        """Declare the provider/model charged by one compaction invocation.

        Return ``None`` only for a compactor that never performs provider work.
        This identity supplies pricing attribution; it does not declare how many
        provider calls an opaque compactor may make. Automatic compaction under
        run or cost limits therefore also requires Cayu's built-in per-dispatch
        instrumentation.
        """

        raise NotImplementedError(
            f"{type(self).__name__} must declare its provider budget identity."
        )

    def _provider_budget_identity_for_request(
        self,
        request: CompactionRequest,
    ) -> tuple[str, str] | None:
        """Declare the provider/model actually selected for this invocation."""

        return self.provider_budget_identity(request.session)

    def _uses_runtime_provider_dispatch_runner_for_request(
        self,
        request: CompactionRequest,
    ) -> bool:
        """Whether every provider dispatch uses Cayu's instrumented model runner."""

        del request
        return False

    def _uses_runtime_provider_dispatch_runner_for_forced_compaction(self) -> bool:
        """Whether forced bounded compaction exposes every provider dispatch."""

        return False

    def _progress_key(self) -> str | None:
        """Identify a configuration whose terminal no-progress result is reusable."""

        return None

    def _progress_key_for_context_request(
        self,
        request: ContextRequest,
        *,
        previous_summary: str | None,
    ) -> str | None:
        """Identify the no-progress configuration selected for this policy build."""

        del request, previous_summary
        return self._progress_key()

    def _bounded_input_for_request(self, request: CompactionRequest) -> bool | None:
        """Declare whether this invocation is bounded before it executes."""

        del request
        return None

    @abstractmethod
    async def compact(self, request: CompactionRequest) -> CompactionResult:
        """Return a compact summary for older transcript messages."""


_AutomaticCompactionRunner = Callable[
    [
        ContextCompactor,
        CompactionRequest,
        ContextCompactionTelemetry,
        Callable[[], Awaitable[CompactionResult]],
        Callable[[], list[dict[str, Any]]],
    ],
    Awaitable[CompactionResult],
]
_AUTOMATIC_COMPACTION_RUNNER: ContextVar[_AutomaticCompactionRunner | None] = ContextVar(
    "automatic_compaction_runner",
    default=None,
)
_CONTEXT_SECRET_REDACTOR: ContextVar[SecretRedactor | None] = ContextVar(
    "cayu_context_secret_redactor",
    default=None,
)

_AutomaticCompactionDispatchRunner = Callable[
    [
        ModelProvider,
        str,
        str,
        str,
        UsageDialect,
        BillingIdentity | None,
        ModelRequest,
        int,
        int,
        Callable[[], Awaitable[tuple[str, dict[str, Any]]]],
    ],
    Awaitable[tuple[str, dict[str, Any]]],
]
_AUTOMATIC_COMPACTION_DISPATCH_RUNNER: ContextVar[_AutomaticCompactionDispatchRunner | None] = (
    ContextVar(
        "automatic_compaction_dispatch_runner",
        default=None,
    )
)
_COMPACTION_MODEL_ATTEMPT_IDENTITY: ContextVar[ModelAttemptIdentity | None] = ContextVar(
    "compaction_model_attempt_identity",
    default=None,
)
_DEFER_BILLING_IDENTITY_CANCELLATION: ContextVar[bool] = ContextVar(
    "defer_billing_identity_cancellation",
    default=False,
)


@contextmanager
def _automatic_compaction_runner_scope(
    runner: _AutomaticCompactionRunner | None,
) -> Iterator[None]:
    token = _AUTOMATIC_COMPACTION_RUNNER.set(runner)
    try:
        yield
    finally:
        _AUTOMATIC_COMPACTION_RUNNER.reset(token)


@contextmanager
def _context_secret_redactor_scope(redactor: SecretRedactor) -> Iterator[None]:
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    token = _CONTEXT_SECRET_REDACTOR.set(redactor)
    try:
        yield
    finally:
        _CONTEXT_SECRET_REDACTOR.reset(token)


def _active_context_secret_redactor() -> SecretRedactor:
    return _CONTEXT_SECRET_REDACTOR.get() or SecretRedactor()


@contextmanager
def _automatic_compaction_dispatch_runner_scope(
    runner: _AutomaticCompactionDispatchRunner | None,
) -> Iterator[None]:
    token = _AUTOMATIC_COMPACTION_DISPATCH_RUNNER.set(runner)
    try:
        yield
    finally:
        _AUTOMATIC_COMPACTION_DISPATCH_RUNNER.reset(token)


@contextmanager
def _compaction_model_attempt_identity_scope(
    identity: ModelAttemptIdentity,
) -> Iterator[None]:
    """Expose one runtime-owned identity only across its provider dispatch."""

    copied = copy_model_attempt_identity(identity)
    token = _COMPACTION_MODEL_ATTEMPT_IDENTITY.set(copied)
    try:
        yield
    finally:
        _COMPACTION_MODEL_ATTEMPT_IDENTITY.reset(token)


@contextmanager
def _defer_billing_identity_cancellation_scope() -> Iterator[None]:
    """Keep private billing markers intact until a provider-free runtime boundary."""

    token = _DEFER_BILLING_IDENTITY_CANCELLATION.set(True)
    try:
        yield
    finally:
        _DEFER_BILLING_IDENTITY_CANCELLATION.reset(token)


class TranscriptDigestCompactor(ContextCompactor):
    """Deterministic fallback compactor that represents an atomic message prefix."""

    def __init__(self, *, max_summary_chars: int = 8000) -> None:
        if type(max_summary_chars) is not int:
            raise TypeError("max_summary_chars must be an integer.")
        if max_summary_chars < 200:
            raise ValueError("max_summary_chars must be at least 200.")
        self.max_summary_chars = max_summary_chars

    def provider_budget_identity(self, session: Session) -> None:
        return None

    def _progress_key(self) -> str:
        implementation = (
            f"{type(self).__module__}:{type(self).__qualname__}:"
            f"{type(self).compact.__module__}:{type(self).compact.__qualname__}"
        )
        implementation_digest = hashlib.sha256(implementation.encode("utf-8")).hexdigest()
        return (
            "transcript-digest:v2:implementation="
            f"{implementation_digest}:max-summary-chars={self.max_summary_chars}"
        )

    def _bounded_input_for_request(self, request: CompactionRequest) -> bool:
        del request
        return True

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        progress_key = self._progress_key()
        if (
            request.existing_summary is not None
            and len(request.existing_summary) > self.max_summary_chars
        ):
            return CompactionResult(
                summary=request.existing_summary,
                covered_message_count=0,
                source_chunk_count=0,
                source_chunk_mode="digest_capacity_exhausted",
                bounded_input=True,
                progress_exhausted=True,
                progress_key=progress_key,
                metadata={
                    "compactor": type(self).__name__,
                    "max_summary_chars": self.max_summary_chars,
                    "progress_reason": "existing_summary_exceeds_limit",
                },
            )
        covered_message_count = 0
        summary = request.existing_summary
        if request.messages:
            digest_lines = [_message_digest(message) for message in request.messages]
            digest_length = 0
            previous_count = 0
            fixed_length = len(_DIGEST_NEW_TRANSCRIPT_HEADER)
            if request.existing_summary is not None:
                fixed_length += (
                    len(_DIGEST_PREVIOUS_SUMMARY_HEADER)
                    + len(request.existing_summary)
                    + len(_DIGEST_SECTION_JOINER)
                )
            for count in _compaction_atomic_prefix_counts(request.messages):
                for index in range(previous_count, count):
                    if index:
                        digest_length += 1
                    digest_length += len(digest_lines[index])
                previous_count = count
                if fixed_length + digest_length > self.max_summary_chars:
                    break
                covered_message_count = count
            if covered_message_count:
                sections: list[str] = []
                if request.existing_summary is not None:
                    sections.append(_DIGEST_PREVIOUS_SUMMARY_HEADER + request.existing_summary)
                sections.append(
                    _DIGEST_NEW_TRANSCRIPT_HEADER + "\n".join(digest_lines[:covered_message_count])
                )
                summary = _DIGEST_SECTION_JOINER.join(sections)
            if covered_message_count == 0 and summary is None:
                # Do not inject a clipped fragment beside the same uncovered
                # source, which would duplicate an arbitrary portion of it in
                # the effective context while acknowledging nothing.
                summary = _DIGEST_ZERO_COVERAGE_SUMMARY
        if summary is None:
            raise ValueError("Compaction requires source messages or an existing summary.")
        progress_exhausted = covered_message_count == 0 and bool(request.messages)
        return CompactionResult(
            summary=summary,
            covered_message_count=covered_message_count,
            represented_existing_summary_sha256=(
                _compaction_summary_sha256(request.existing_summary)
                if request.existing_summary is not None and covered_message_count > 0
                else None
            ),
            source_chunk_count=0 if progress_exhausted else 1,
            source_chunk_mode=(
                "digest_capacity_exhausted" if progress_exhausted else "digest_prefix"
            ),
            bounded_input=covered_message_count < len(request.messages),
            progress_exhausted=progress_exhausted,
            progress_key=progress_key if progress_exhausted else None,
            metadata={
                "compactor": type(self).__name__,
                "max_summary_chars": self.max_summary_chars,
                **({"progress_reason": "no_atomic_prefix_fits"} if progress_exhausted else {}),
            },
        )


_DIGEST_PREVIOUS_SUMMARY_HEADER = "Previous summary:\n"
_DIGEST_NEW_TRANSCRIPT_HEADER = "Newly compacted transcript:\n"
_DIGEST_SECTION_JOINER = "\n\n"
_DIGEST_ZERO_COVERAGE_SUMMARY = "No source history was compacted."


@dataclass(frozen=True, slots=True)
class _CompactionProviderSnapshot:
    provider_name: str
    pricing_provider_name: str
    usage_dialect: UsageDialect


@dataclass(frozen=True, slots=True)
class _ModelCompactorInvocationIdentity:
    owner_id: int
    provider: ModelProvider
    provider_snapshot: _CompactionProviderSnapshot
    model: str
    compactor_name: str
    system_prompt: str
    options: dict[str, Any]
    retry_policy: RetryPolicy
    max_input_chars: int | None
    max_hierarchy_calls: int


_MODEL_COMPACTOR_INVOCATION_IDENTITY: ContextVar[_ModelCompactorInvocationIdentity | None] = (
    ContextVar(
        "model_compactor_invocation_identity",
        default=None,
    )
)


def _compaction_provider_snapshot(
    provider: ModelProvider,
    *,
    usage_dialect: UsageDialect | None = None,
) -> _CompactionProviderSnapshot:
    provider_name = require_durable_clean_nonblank(provider.name, "provider.name")
    billing_provider_name = provider.billing_provider_name
    pricing_provider_name = (
        provider_name
        if billing_provider_name is None
        else require_durable_clean_nonblank(
            billing_provider_name,
            "provider.billing_provider_name",
        )
    )
    return _CompactionProviderSnapshot(
        provider_name=provider_name,
        pricing_provider_name=pricing_provider_name,
        usage_dialect=copy_usage_dialect(
            provider.usage_dialect if usage_dialect is None else usage_dialect,
            "provider.usage_dialect",
        ),
    )


class ModelCompactor(ContextCompactor):
    """Provider-backed compactor that asks a model to summarize older context."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str,
        system_prompt: str = (
            "You summarize prior agent session context for a future model call. "
            "Return only the compact summary. Do not call tools."
        ),
        options: dict[str, Any] | None = None,
        max_input_chars: int | None = 120_000,
        max_hierarchy_calls: int = 64,
        prompt_builder: CompactionPromptBuilder | None = None,
        retry_policy: RetryPolicy | None = None,
        _usage_dialect: UsageDialect | None = None,
        _provider_snapshot: _CompactionProviderSnapshot | None = None,
    ) -> None:
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must be a ModelProvider.")
        if max_input_chars is not None:
            if type(max_input_chars) is not int:
                raise TypeError("max_input_chars must be an integer or None.")
            if max_input_chars < 1000:
                raise ValueError("max_input_chars must be at least 1000.")
        if type(max_hierarchy_calls) is not int:
            raise TypeError("max_hierarchy_calls must be an integer.")
        if max_hierarchy_calls < 2:
            raise ValueError("max_hierarchy_calls must be at least 2.")
        if _provider_snapshot is not None and type(_provider_snapshot) is not (
            _CompactionProviderSnapshot
        ):
            raise TypeError("_provider_snapshot must be a _CompactionProviderSnapshot.")
        self.provider = provider
        self._provider_snapshot = (
            _compaction_provider_snapshot(provider, usage_dialect=_usage_dialect)
            if _provider_snapshot is None
            else _CompactionProviderSnapshot(
                provider_name=require_durable_clean_nonblank(
                    _provider_snapshot.provider_name,
                    "provider.name",
                ),
                pricing_provider_name=require_durable_clean_nonblank(
                    _provider_snapshot.pricing_provider_name,
                    "provider.billing_provider_name",
                ),
                usage_dialect=copy_usage_dialect(
                    _provider_snapshot.usage_dialect,
                    "provider.usage_dialect",
                ),
            )
        )
        self._usage_dialect = copy_usage_dialect(
            self._provider_snapshot.usage_dialect,
            "provider.usage_dialect",
        )
        self._compactor_name = require_durable_clean_nonblank(
            type(self).__name__,
            "compactor",
        )
        self.model = require_durable_clean_nonblank(model, "model")
        self.system_prompt = require_nonblank(system_prompt, "system_prompt")
        self.options = copy_json_value({} if options is None else options, "options")
        self.max_input_chars = max_input_chars
        self.max_hierarchy_calls = max_hierarchy_calls
        if prompt_builder is not None and not callable(prompt_builder):
            raise TypeError("prompt_builder must be callable.")
        self.prompt_builder = prompt_builder
        # `None` selects the shared default retry policy.
        self.retry_policy = copy_retry_policy(retry_policy)

    def provider_budget_identity(self, session: Session) -> tuple[str, str]:
        del session
        return self._provider_snapshot.pricing_provider_name, self.model

    def _uses_runtime_provider_dispatch_runner_for_request(
        self,
        request: CompactionRequest,
    ) -> bool:
        del request
        return self._uses_builtin_provider_dispatch_boundary()

    def _uses_runtime_provider_dispatch_runner_for_forced_compaction(self) -> bool:
        return self._uses_builtin_provider_dispatch_boundary()

    def _bounded_input_for_request(self, request: CompactionRequest) -> bool:
        del request
        return self.max_input_chars is not None

    def _uses_builtin_provider_dispatch_boundary(self) -> bool:
        """Reject subclasses that can route a provider call around admission."""

        implementation = type(self)
        return all(
            (
                implementation.compact is ModelCompactor.compact,
                implementation._capture_invocation_identity
                is ModelCompactor._capture_invocation_identity,
                implementation._compact_with_invocation_identity
                is ModelCompactor._compact_with_invocation_identity,
                implementation._compact_prompt_once is ModelCompactor._compact_prompt_once,
                implementation._compact_oversized_atomic_unit
                is ModelCompactor._compact_oversized_atomic_unit,
            )
        )

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        identity = self._capture_invocation_identity()
        token = _MODEL_COMPACTOR_INVOCATION_IDENTITY.set(identity)
        del identity
        try:
            return await self._compact_with_invocation_identity(request)
        finally:
            _MODEL_COMPACTOR_INVOCATION_IDENTITY.reset(token)

    def _capture_invocation_identity(self) -> _ModelCompactorInvocationIdentity:
        provider_snapshot = self._provider_snapshot
        if type(provider_snapshot) is not _CompactionProviderSnapshot:
            raise TypeError("_provider_snapshot must be a _CompactionProviderSnapshot.")
        detached_snapshot = _CompactionProviderSnapshot(
            provider_name=require_durable_clean_nonblank(
                provider_snapshot.provider_name,
                "provider.name",
            ),
            pricing_provider_name=require_durable_clean_nonblank(
                provider_snapshot.pricing_provider_name,
                "provider.billing_provider_name",
            ),
            usage_dialect=copy_usage_dialect(
                provider_snapshot.usage_dialect,
                "provider.usage_dialect",
            ),
        )
        provider = self.provider
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must be a ModelProvider.")
        max_input_chars = self.max_input_chars
        if max_input_chars is not None and (
            type(max_input_chars) is not int or max_input_chars < 1000
        ):
            raise ValueError("max_input_chars must be None or an integer of at least 1000.")
        max_hierarchy_calls = self.max_hierarchy_calls
        if type(max_hierarchy_calls) is not int or max_hierarchy_calls < 2:
            raise ValueError("max_hierarchy_calls must be an integer of at least 2.")
        return _ModelCompactorInvocationIdentity(
            owner_id=id(self),
            provider=provider,
            provider_snapshot=detached_snapshot,
            model=require_durable_clean_nonblank(self.model, "model"),
            compactor_name=require_durable_clean_nonblank(
                self._compactor_name,
                "compactor",
            ),
            system_prompt=require_durable_nonblank(self.system_prompt, "system_prompt"),
            options=copy_durable_json_object(self.options, "options"),
            retry_policy=copy_retry_policy(self.retry_policy),
            max_input_chars=max_input_chars,
            max_hierarchy_calls=max_hierarchy_calls,
        )

    def _current_invocation_identity(self) -> _ModelCompactorInvocationIdentity:
        identity = _MODEL_COMPACTOR_INVOCATION_IDENTITY.get()
        if identity is None or identity.owner_id != id(self):
            return self._capture_invocation_identity()
        return identity

    async def _compact_with_invocation_identity(
        self,
        request: CompactionRequest,
    ) -> CompactionResult:
        identity = self._current_invocation_identity()
        max_input_chars = identity.max_input_chars
        max_hierarchy_calls = identity.max_hierarchy_calls
        del identity
        prompt_builder = self.prompt_builder
        if prompt_builder is None or prompt_builder is default_compaction_prompt:
            bounded_prompt, input_truncated, covered_message_count = (
                _bounded_default_compaction_prompt(
                    request,
                    max_chars=max_input_chars,
                )
            )
            if bounded_prompt is None:
                with _compaction_dispatch_counter_scope(max_hierarchy_calls) as dispatch_counter:
                    result = await self._compact_oversized_atomic_unit(request)
                metadata = copy_json_value(result.metadata, "metadata")
                metadata["hierarchy_dispatch_count"] = dispatch_counter.count
                return result.model_copy(
                    update={
                        "metadata": metadata,
                        "represented_existing_summary_sha256": (
                            _compaction_summary_sha256(request.existing_summary)
                            if request.existing_summary is not None
                            else None
                        ),
                    },
                    deep=True,
                )
        else:
            custom_prompt = _detach_compaction_prompt(prompt_builder(request))
            user_prompt = custom_prompt.prompt
            if request.existing_summary is not None:
                user_prompt = (
                    "Existing summary that the replacement summary must continue to "
                    "represent:\n"
                    f"{request.existing_summary}\n\n"
                    "Custom compaction prompt:\n"
                    f"{user_prompt}"
                )
            covered_message_count = custom_prompt.covered_message_count
            _validate_compaction_result_coverage(
                messages=request.messages,
                previous_cursor=0,
                compactable_cursor=len(request.messages),
                covered_message_count=covered_message_count,
            )
            bounded_prompt, input_truncated = _bounded_prompt_text(
                user_prompt,
                max_chars=max_input_chars,
            )
            if input_truncated:
                raise ValueError(
                    "Custom compaction prompts must fit max_input_chars without truncation."
                )
        result = await self._compact_prompt_once(
            bounded_prompt,
            covered_message_count=covered_message_count,
            metadata={
                "input_truncated": input_truncated,
                "max_input_chars": max_input_chars,
            },
        )
        return result.model_copy(
            update={
                "represented_existing_summary_sha256": (
                    _compaction_summary_sha256(request.existing_summary)
                    if request.existing_summary is not None
                    else None
                ),
                "source_chunk_count": 1,
                "source_chunk_mode": ("message_prefix" if input_truncated else "single_request"),
                "bounded_input": input_truncated,
            },
            deep=True,
        )

    async def _compact_prompt_once(
        self,
        user_prompt: str,
        *,
        covered_message_count: int,
        metadata: dict[str, Any],
    ) -> CompactionResult:
        identity = self._current_invocation_identity()
        provider = identity.provider
        model = identity.model
        compactor_name = identity.compactor_name
        retry_policy = identity.retry_policy
        provider_name = identity.provider_snapshot.provider_name
        pricing_provider_name = identity.provider_snapshot.pricing_provider_name
        usage_dialect = identity.provider_snapshot.usage_dialect
        system_prompt = identity.system_prompt
        options = identity.options
        del identity
        completion_ledger = _COMPACTION_COMPLETION_LEDGER.get()
        first_completion_index = (
            0 if completion_ledger is None else len(completion_ledger.completed_payloads)
        )
        model_request = ModelRequest(
            model=model,
            messages=[
                Message.text(MessageRole.SYSTEM, system_prompt),
                Message.text(MessageRole.USER, user_prompt),
            ],
            tools=[],
            options=options,
        )
        terminal_observation_error: ModelProviderError | None = None
        terminal_value_error: DurableValueError | None = None
        try:
            try:
                summary, completed_metadata, completion_payloads = await _run_compaction_model(
                    provider=provider,
                    provider_name=provider_name,
                    pricing_provider_name=pricing_provider_name,
                    model_request=model_request,
                    retry_policy=retry_policy,
                    compactor=compactor_name,
                    usage_dialect=usage_dialect,
                    observe_completion=_compaction_completion_observer(
                        provider_name=pricing_provider_name,
                        model=model,
                        compactor=compactor_name,
                        usage_dialect=usage_dialect,
                    ),
                )
            finally:
                del provider
        except _CompactionCompletionObservationError as exc:
            terminal_observation_error = _record_compaction_completion_observation_failure(
                exc,
                provider_name=pricing_provider_name,
                model=model,
                compactor=compactor_name,
                usage_dialect=usage_dialect,
            )
        except _CompactionCompletionValueError as exc:
            terminal_value_error = _record_invalid_compaction_completion(
                exc,
                provider_name=pricing_provider_name,
                model=model,
                compactor=compactor_name,
                usage_dialect=usage_dialect,
            )
        except _CompactionToolCallError as exc:
            # A terminal completion is real provider spend even though the tool-call
            # protocol violation makes the summary unusable. An unfinished stream has
            # no authoritative completion payload and must not fabricate usage.
            if exc.completed_metadata is not None:
                _record_compaction_model_completed_payloads(
                    [
                        _rejected_compaction_tool_call_payload(
                            error=exc,
                            provider_name=pricing_provider_name,
                            model=model,
                            compactor=compactor_name,
                            usage_dialect=usage_dialect,
                        )
                    ]
                )
            await _publish_compaction_ledger_since(
                completion_ledger,
                first_completion_index,
            )
            raise
        if terminal_observation_error is not None:
            await _publish_compaction_ledger_since(
                completion_ledger,
                first_completion_index,
            )
            raise terminal_observation_error from None
        if terminal_value_error is not None:
            await _publish_compaction_ledger_since(
                completion_ledger,
                first_completion_index,
            )
            raise terminal_value_error from None
        try:
            result = _provider_compaction_result(
                summary=summary,
                completed_metadata=completed_metadata,
                provider_name=pricing_provider_name,
                model=model,
                compactor=compactor_name,
                usage_dialect=usage_dialect,
                metadata={
                    **copy_json_value(metadata, "metadata"),
                },
                covered_message_count=covered_message_count,
            )
        except BaseException:
            await _publish_compaction_ledger_since(
                completion_ledger,
                first_completion_index,
            )
            raise
        await _publish_compaction_ledger_since(
            completion_ledger,
            first_completion_index,
        )
        return result.model_copy(
            update={"model_completed_payloads": completion_payloads},
            deep=True,
        )

    async def _compact_oversized_atomic_unit(
        self,
        request: CompactionRequest,
    ) -> CompactionResult:
        identity = self._current_invocation_identity()
        max_input_chars = identity.max_input_chars
        max_hierarchy_calls = identity.max_hierarchy_calls
        compactor_name = identity.compactor_name
        provider_name = identity.provider_snapshot.provider_name
        model = identity.model
        del identity
        if max_input_chars is None:
            raise RuntimeError("Unbounded compaction unexpectedly required hierarchy.")
        atomic_counts = _compaction_atomic_prefix_counts(request.messages)
        if not atomic_counts:
            raise ValueError("Compaction requires at least one source message.")
        covered_message_count = atomic_counts[0]
        source = _messages_digest(request.messages[:covered_message_count])
        source_prompt_prefix = _hierarchy_source_prompt_prefix(request.instructions)
        merge_prompt_prefix = _hierarchy_merge_prompt_prefix(request.instructions)
        source_fragments = _split_hierarchy_text(
            source,
            max_chars=max_input_chars,
            prompt_prefix=source_prompt_prefix,
        )
        merge_required = request.existing_summary is not None or len(source_fragments) > 1
        if merge_required:
            # Validate deterministic assembly capacity before any provider work.
            # Leaf summaries are not useful unless at least one bounded merge item
            # can be represented alongside the instructions.
            _split_hierarchy_items(
                ["x"],
                max_chars=max_input_chars,
                prompt_prefix=merge_prompt_prefix,
            )
        minimum_merge_calls = 0
        if merge_required:
            known_initial_items = ["x"] * len(source_fragments)
            if request.existing_summary is not None:
                known_initial_items.insert(0, request.existing_summary)
            optimistic_items = known_initial_items
            while len(optimistic_items) > 1:
                expanded_items = _split_hierarchy_items(
                    optimistic_items,
                    max_chars=max_input_chars,
                    prompt_prefix=merge_prompt_prefix,
                )
                merge_groups = _pack_hierarchy_items(
                    expanded_items,
                    max_chars=max_input_chars,
                    prompt_prefix=merge_prompt_prefix,
                )
                minimum_merge_calls += len(merge_groups)
                if len(merge_groups) == 1:
                    break
                # One Unicode scalar is the smallest valid summary each merge
                # can return. Simulating every later level with that optimistic
                # output computes a true lower bound for the complete tree.
                optimistic_items = ["x"] * len(merge_groups)
                if len(source_fragments) + minimum_merge_calls > max_hierarchy_calls:
                    break
        minimum_calls = len(source_fragments) + minimum_merge_calls
        if minimum_calls > max_hierarchy_calls:
            raise ValueError(
                "Oversized compaction source exceeds max_hierarchy_calls before dispatch."
            )

        completed_payloads: list[dict[str, Any]] = []
        leaf_summaries: list[str] = []
        dispatch_count = 0
        for index, fragment in enumerate(source_fragments, start=1):
            dispatch_count += 1
            leaf = await self._compact_prompt_once(
                _hierarchy_source_prompt(
                    fragment,
                    index=index,
                    prompt_prefix=source_prompt_prefix,
                ),
                covered_message_count=0,
                metadata={
                    "input_truncated": True,
                    "max_input_chars": max_input_chars,
                    "hierarchy_phase": "source",
                },
            )
            completed_payloads.extend(leaf.model_completed_payloads)
            leaf_summaries.append(leaf.summary)

        items = list(leaf_summaries)
        if request.existing_summary is not None:
            items.insert(0, request.existing_summary)
        allow_initial_expansion = True
        while len(items) > 1:
            current_measure = (len(items), sum(len(item) for item in items))
            expanded_items = _split_hierarchy_items(
                items,
                max_chars=max_input_chars,
                prompt_prefix=merge_prompt_prefix,
            )
            groups = _pack_hierarchy_items(
                expanded_items,
                max_chars=max_input_chars,
                prompt_prefix=merge_prompt_prefix,
            )
            next_items: list[str] = []
            for group in groups:
                if dispatch_count >= max_hierarchy_calls:
                    raise ValueError("Oversized compaction source exceeded max_hierarchy_calls.")
                dispatch_count += 1
                merged = await self._compact_prompt_once(
                    _hierarchy_merge_prompt(
                        group,
                        prompt_prefix=merge_prompt_prefix,
                    ),
                    covered_message_count=0,
                    metadata={
                        "input_truncated": True,
                        "max_input_chars": max_input_chars,
                        "hierarchy_phase": "merge",
                    },
                )
                completed_payloads.extend(merged.model_completed_payloads)
                next_items.append(merged.summary)
            measure = (len(next_items), sum(len(item) for item in next_items))
            if not allow_initial_expansion and measure >= current_measure:
                raise ValueError("Hierarchical compaction did not converge within its bound.")
            allow_initial_expansion = False
            items = next_items

        return CompactionResult(
            summary=items[0],
            covered_message_count=covered_message_count,
            source_chunk_count=len(source_fragments),
            source_chunk_mode="hierarchical_atomic_unit",
            bounded_input=True,
            metadata={
                "compactor": compactor_name,
                "provider": provider_name,
                "model": model,
                "input_truncated": True,
                "max_input_chars": max_input_chars,
                "hierarchy_dispatch_count": dispatch_count,
            },
            model_completed_payloads=completed_payloads,
        )


_DEFAULT_PROMPT_CACHE_COMPACTION_INSTRUCTION = (
    "Summarize the conversation above so a future agent step can continue "
    "with the important context. Preserve concrete user requests, decisions, "
    "files or resources mentioned, tool results, errors, and pending work. "
    "Do not invent facts. Keep the summary concise but specific. "
    "Do not call tools. Return only the summary text."
)


class _CompactionToolCallError(RuntimeError):
    """Compaction protocol failure with any provider-reported completion metadata."""

    def __init__(self, *, completed_metadata: dict[str, Any] | None) -> None:
        super().__init__("Compaction model must not call tools.")
        self.completed_metadata = (
            None
            if completed_metadata is None
            else copy_json_value(completed_metadata, "completed_metadata")
        )


class _CompactionCompletionValueError(RuntimeError):
    """A completed provider call carried non-portable auxiliary metadata."""

    def __init__(
        self,
        *,
        error: DurableValueError,
        completed_metadata: dict[str, Any],
        rejected_usage_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.completed_metadata = copy_durable_json_object(
            completed_metadata,
            "completed_metadata",
        )
        self.rejected_usage_payload = (
            None
            if rejected_usage_payload is None
            else copy_durable_json_object(
                rejected_usage_payload,
                "rejected_usage_payload",
            )
        )


class _CompactionAccountingUsageError(RuntimeError):
    """Normalized compaction counters exceeded the portable value contract."""

    def __init__(self, *, payload: dict[str, Any]) -> None:
        error = DurableValueError("integer_out_of_range", "usage_metrics")
        super().__init__(str(error))
        self.error = error
        self.payload = copy_durable_json_object(payload, "model_completed_payload")


class _CompactionCompletionObservationError(RuntimeError):
    """A terminal completion could not finish billing/ledger observation."""

    def __init__(
        self,
        *,
        error: ModelProviderError,
        completed_metadata: dict[str, Any],
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.completed_metadata = copy_durable_json_object(
            completed_metadata,
            "completed_metadata",
        )


class _ProviderDispatchFailed(RuntimeError):
    """A detached provider failure crossing the instrumented dispatch runner."""

    def __init__(
        self,
        control: ProviderExceptionControl,
        *,
        completed_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(control.message)
        self.control = control
        self.completed_metadata = (
            None
            if completed_metadata is None
            else copy_durable_json_object(completed_metadata, "completed_metadata")
        )


_COMPACTION_ATTEMPT_ID_KEY = "compaction_attempt_id"


class _CompactionDispatchCounter:
    def __init__(self, maximum: int) -> None:
        self.maximum = maximum
        self.count = 0

    def before_dispatch(self) -> None:
        if self.count >= self.maximum:
            raise ValueError("Oversized compaction source exceeded max_hierarchy_calls.")
        self.count += 1


_COMPACTION_DISPATCH_COUNTER: ContextVar[_CompactionDispatchCounter | None] = ContextVar(
    "compaction_dispatch_counter", default=None
)


@contextmanager
def _compaction_dispatch_counter_scope(
    maximum: int,
) -> Iterator[_CompactionDispatchCounter]:
    counter = _CompactionDispatchCounter(maximum)
    token = _COMPACTION_DISPATCH_COUNTER.set(counter)
    try:
        yield counter
    finally:
        _COMPACTION_DISPATCH_COUNTER.reset(token)


_CompactionCompletionPublisher = Callable[[list[dict[str, Any]]], Awaitable[None]]
_COMPACTION_COMPLETION_PUBLISHER: ContextVar[_CompactionCompletionPublisher | None] = ContextVar(
    "compaction_completion_publisher", default=None
)


@contextmanager
def _compaction_completion_publisher_scope(
    publisher: _CompactionCompletionPublisher | None,
) -> Iterator[None]:
    token = _COMPACTION_COMPLETION_PUBLISHER.set(publisher)
    try:
        yield
    finally:
        _COMPACTION_COMPLETION_PUBLISHER.reset(token)


class _CompactionCompletionLedger:
    def __init__(self) -> None:
        self.completed_payloads: list[dict[str, Any]] = []
        self.indices_by_attempt_id: dict[str, int] = {}

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        identified = copy_durable_json_object(payload, "model_completed_payload")
        attempt_id = identified.get(_COMPACTION_ATTEMPT_ID_KEY)
        if type(attempt_id) is not str or attempt_id not in self.indices_by_attempt_id:
            attempt_id = uuid4().hex
            identified[_COMPACTION_ATTEMPT_ID_KEY] = attempt_id
            self.indices_by_attempt_id[attempt_id] = len(self.completed_payloads)
            self.completed_payloads.append(identified)
        else:
            self.completed_payloads[self.indices_by_attempt_id[attempt_id]] = identified
        return copy_durable_json_object(identified, "model_completed_payload")

    def merge_returned_payloads(
        self,
        payloads: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        copied_payloads = copy_durable_json_value(payloads, "model_completed_payloads")
        if type(copied_payloads) is not list or any(
            type(payload) is not dict for payload in copied_payloads
        ):
            raise TypeError("CompactionResult.model_completed_payloads must be a list of objects.")

        candidate_payloads = copy_durable_json_value(
            self.completed_payloads,
            "model_completed_payloads",
        )
        candidate_indices = dict(self.indices_by_attempt_id)
        returned_payloads: list[dict[str, Any]] = []
        returned_ids_seen: set[str] = set()
        for identified in copied_payloads:
            attempt_id = identified.get(_COMPACTION_ATTEMPT_ID_KEY)
            if type(attempt_id) is str and attempt_id in candidate_indices:
                if attempt_id in returned_ids_seen:
                    raise ValueError(
                        "CompactionResult.model_completed_payloads contains a duplicate "
                        "runtime attempt identity."
                    )
                anchored = candidate_payloads[candidate_indices[attempt_id]]
                if identified != anchored:
                    raise ValueError(
                        "CompactionResult cannot rewrite runtime-owned completion evidence."
                    )
                identified = copy_durable_json_object(
                    anchored,
                    "model_completed_payload",
                )
            else:
                attempt_id = uuid4().hex
                while attempt_id in candidate_indices or attempt_id in returned_ids_seen:
                    attempt_id = uuid4().hex
                identified[_COMPACTION_ATTEMPT_ID_KEY] = attempt_id
            returned_ids_seen.add(attempt_id)
            returned_payloads.append(identified)

        # The returned list can supply calls that a wrapping compactor observed before
        # an inner provider-backed compactor registered itself. Bucket those payloads
        # around runtime-anchored calls, then rebuild once. This preserves omitted
        # runtime observations and keeps merging linear in the ledger size.
        before_anchors: dict[int, list[dict[str, Any]]] = {}
        after_anchors: dict[int, list[dict[str, Any]]] = {}
        pending: list[dict[str, Any]] = []
        previous_anchor: int | None = None
        for payload in returned_payloads:
            attempt_id = payload[_COMPACTION_ATTEMPT_ID_KEY]
            anchor = candidate_indices.get(attempt_id)
            if anchor is None:
                pending.append(payload)
                continue
            if previous_anchor is not None and anchor <= previous_anchor:
                raise ValueError(
                    "CompactionResult cannot reorder runtime-owned completion evidence."
                )
            if previous_anchor is None:
                before_anchors[anchor] = pending
            else:
                after_anchors[previous_anchor] = pending
            pending = []
            previous_anchor = anchor
        if previous_anchor is None:
            trailing = pending
        else:
            after_anchors[previous_anchor] = pending
            trailing = []

        merged_payloads: list[dict[str, Any]] = []
        for index, payload in enumerate(candidate_payloads):
            merged_payloads.extend(before_anchors.get(index, ()))
            merged_payloads.append(payload)
            merged_payloads.extend(after_anchors.get(index, ()))
        merged_payloads.extend(trailing)
        candidate_payloads = merged_payloads

        candidate_indices = {
            payload[_COMPACTION_ATTEMPT_ID_KEY]: index
            for index, payload in enumerate(candidate_payloads)
        }
        self.completed_payloads = candidate_payloads
        self.indices_by_attempt_id = candidate_indices
        return copy_durable_json_value(candidate_payloads, "model_completed_payloads")


_COMPACTION_COMPLETION_LEDGER: ContextVar[_CompactionCompletionLedger | None] = ContextVar(
    "compaction_completion_ledger", default=None
)


def _record_compaction_model_completed_payloads(
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Record provider completions immediately in their observed order."""

    active_identity = _COMPACTION_MODEL_ATTEMPT_IDENTITY.get()
    identified_payloads = copy_durable_json_value(payloads, "model_completed_payloads")
    if active_identity is not None:
        identity_payload = copy_model_attempt_identity(active_identity).payload()
        for payload in identified_payloads:
            payload.update(identity_payload)
    ledger = _COMPACTION_COMPLETION_LEDGER.get()
    if ledger is None:
        for payload in identified_payloads:
            payload.pop(_COMPACTION_ATTEMPT_ID_KEY, None)
        return identified_payloads
    return [ledger.upsert(payload) for payload in identified_payloads]


async def _publish_compaction_completion_payloads(
    payloads: list[dict[str, Any]],
) -> None:
    publisher = _COMPACTION_COMPLETION_PUBLISHER.get()
    if publisher is None or not payloads:
        return
    await publisher(copy_durable_json_value(payloads, "model_completed_payloads"))


async def _publish_compaction_ledger_since(
    ledger: _CompactionCompletionLedger | None,
    first_index: int,
) -> None:
    if ledger is None:
        return
    await _publish_compaction_completion_payloads(ledger.completed_payloads[first_index:])


def _public_compaction_model_completed_payload(payload: dict[str, Any]) -> dict[str, Any]:
    public_payload = copy_durable_json_object(payload, "model_completed_payload")
    public_payload.pop(_COMPACTION_ATTEMPT_ID_KEY, None)
    return public_payload


class _PromptCacheCompactionMode(StrEnum):
    EXACT = "exact"
    BOUNDED = "bounded"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class _PromptCacheCompactorInvocationIdentity:
    provider: ModelProvider
    provider_snapshot: _CompactionProviderSnapshot
    model: str | None
    compactor_name: str
    compaction_instruction: str
    options: dict[str, Any]
    retry_policy: RetryPolicy
    fallback: ContextCompactor


def _prompt_cache_compaction_mode(
    *,
    request: ContextRequest,
    compactor: PromptCacheCompactor,
    previous_summary: str | None,
) -> _PromptCacheCompactionMode:
    """Choose the first-checkpoint cache path from one auditable decision."""

    if previous_summary is not None or request.force_bounded_compaction:
        return _PromptCacheCompactionMode.BOUNDED
    if any(
        (
            compactor.model not in {None, request.session.model},
            compactor._provider_snapshot.provider_name != request.session.provider_name,
            request.context_usage.last_provider_name is not None
            and request.context_usage.last_provider_name
            != compactor._provider_snapshot.provider_name,
            request.context_usage.last_requested_model is not None
            and request.context_usage.last_requested_model != request.session.model,
            request.pressure_overhead.structured_output_instruction is not None,
        )
    ):
        return _PromptCacheCompactionMode.BOUNDED
    if (
        request.context_usage.last_provider_name == compactor._provider_snapshot.provider_name
        and request.context_usage.last_requested_model == request.session.model
        and request.build_cache_prefix_request is not None
    ):
        return _PromptCacheCompactionMode.EXACT
    return _PromptCacheCompactionMode.FALLBACK


class PromptCacheCompactor(ContextCompactor):
    """Compactor that reuses the first provider prompt-cache prefix.

    On the first compaction, extends the runtime's exact ``ModelRequest`` with a
    compaction instruction. This preserves model, messages, tool definitions,
    thinking configuration, provider options, and resolved file attachments at
    the cache boundary. Compactor options recursively override the copied
    request options; native structured-output enforcement is disabled because
    the compactor must return summary text and must not call tools.
    A configured model override that differs from the cached request uses bounded
    ``ModelCompactor`` input because provider caches are model-bound. Provider
    identity mismatches and tool-based structured-output requests also use the
    bounded path so the exact transcript, tools, synthetic instruction, and
    resolved attachment bytes cannot cross an incompatible request boundary.
    Cross-provider compaction requires an explicit provider-compatible ``model``;
    Cayu never forwards the session provider's model name to another provider.

    Later compactions use bounded ``ModelCompactor`` input containing only the
    previous checkpoint summary and newly compactable messages. This avoids
    rebuilding an unbounded raw-transcript prefix after the cache checkpoint.

    Falls back to the configured fallback compactor when no completed-request
    cursor plus matching durable provider/requested-model identity is available
    to reconstruct the exact runtime request.
    """

    def __init__(
        self,
        *,
        provider: ModelProvider,
        model: str | None = None,
        compaction_instruction: str | None = None,
        options: dict[str, Any] | None = None,
        fallback_compactor: ContextCompactor | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must be a ModelProvider.")
        if model is not None:
            model = require_durable_clean_nonblank(model, "model")
        self.provider = provider
        self._provider_snapshot = _compaction_provider_snapshot(provider)
        self._usage_dialect = copy_usage_dialect(
            self._provider_snapshot.usage_dialect,
            "provider.usage_dialect",
        )
        self._compactor_name = require_durable_clean_nonblank(
            type(self).__name__,
            "compactor",
        )
        self.model = model
        self.compaction_instruction = (
            compaction_instruction
            if compaction_instruction is not None
            else _DEFAULT_PROMPT_CACHE_COMPACTION_INSTRUCTION
        )
        require_nonblank(self.compaction_instruction, "compaction_instruction")
        self.options = copy_json_value({} if options is None else options, "options")
        self.retry_policy = copy_retry_policy(retry_policy)
        if fallback_compactor is None:
            self._fallback: ContextCompactor = TranscriptDigestCompactor()
        elif isinstance(fallback_compactor, ContextCompactor):
            self._fallback = fallback_compactor
        else:
            raise TypeError("fallback_compactor must be a ContextCompactor.")

    def provider_budget_identity(self, session: Session) -> tuple[str, str]:
        if self._provider_snapshot.provider_name != session.provider_name and self.model is None:
            raise ValueError(
                "model is required when the compactor provider differs from the session provider."
            )
        return (
            self._provider_snapshot.pricing_provider_name,
            self.model if self.model is not None else session.model,
        )

    def _provider_budget_identity_for_request(
        self,
        request: CompactionRequest,
    ) -> tuple[str, str] | None:
        provider_differs = self._provider_snapshot.provider_name != request.session.provider_name
        bounded_model = self.model if self.model is not None else request.session.model
        if request.existing_summary is not None or request.force_bounded_compaction:
            return self._provider_snapshot.pricing_provider_name, bounded_model
        if provider_differs:
            if self.model is None:
                raise ValueError(
                    "model is required when the compactor provider differs from "
                    "the session provider."
                )
            return self._provider_snapshot.pricing_provider_name, bounded_model

        cached_request = request.cache_prefix_request
        if cached_request is None:
            if self.model is not None and self.model != request.session.model:
                return self._provider_snapshot.pricing_provider_name, self.model
            return self._fallback._provider_budget_identity_for_request(request)
        cached_model = cached_request.model
        if cached_model != request.session.model:
            return self._provider_snapshot.pricing_provider_name, bounded_model
        if self.model is not None and self.model != cached_model:
            return self._provider_snapshot.pricing_provider_name, self.model
        return (
            self._provider_snapshot.pricing_provider_name,
            self.model if self.model is not None else cached_model,
        )

    def _uses_runtime_provider_dispatch_runner_for_request(
        self,
        request: CompactionRequest,
    ) -> bool:
        if (
            type(self).compact is not PromptCacheCompactor.compact
            or type(self)._capture_invocation_identity
            is not PromptCacheCompactor._capture_invocation_identity
        ):
            return False
        provider_differs = self._provider_snapshot.provider_name != request.session.provider_name
        if request.existing_summary is not None or request.force_bounded_compaction:
            return type(self)._compact_bounded is PromptCacheCompactor._compact_bounded
        if provider_differs:
            return type(self)._compact_bounded is PromptCacheCompactor._compact_bounded
        cached_request = request.cache_prefix_request
        if cached_request is None:
            if self.model is not None and self.model != request.session.model:
                return type(self)._compact_bounded is PromptCacheCompactor._compact_bounded
            return self._fallback._uses_runtime_provider_dispatch_runner_for_request(request)
        cached_model = cached_request.model
        if (
            cached_model != request.session.model
            or (self.model is not None and self.model != cached_model)
            or _has_structured_output_tool(cached_request.tools)
        ):
            return type(self)._compact_bounded is PromptCacheCompactor._compact_bounded
        return (
            type(self)._compact_bounded_after_exact_failure
            is PromptCacheCompactor._compact_bounded_after_exact_failure
            and type(self)._compact_bounded is PromptCacheCompactor._compact_bounded
        )

    def _uses_runtime_provider_dispatch_runner_for_forced_compaction(self) -> bool:
        return (
            type(self).compact is PromptCacheCompactor.compact
            and type(self)._capture_invocation_identity
            is PromptCacheCompactor._capture_invocation_identity
            and type(self)._compact_bounded is PromptCacheCompactor._compact_bounded
        )

    def _progress_key_for_context_request(
        self,
        request: ContextRequest,
        *,
        previous_summary: str | None,
    ) -> str | None:
        if type(self).compact is not PromptCacheCompactor.compact:
            return self._progress_key()
        mode = _prompt_cache_compaction_mode(
            request=request,
            compactor=self,
            previous_summary=previous_summary,
        )
        if mode == _PromptCacheCompactionMode.FALLBACK or (
            mode == _PromptCacheCompactionMode.EXACT
            and _prompt_cache_previous_input_cursor(request) is None
        ):
            return self._fallback._progress_key_for_context_request(
                request,
                previous_summary=previous_summary,
            )
        return self._progress_key()

    def _bounded_input_for_request(self, request: CompactionRequest) -> bool | None:
        provider_differs = self._provider_snapshot.provider_name != request.session.provider_name
        if request.existing_summary is not None or request.force_bounded_compaction:
            return True
        if provider_differs:
            return True
        cached_request = request.cache_prefix_request
        if cached_request is None:
            if self.model is not None and self.model != request.session.model:
                return True
            return self._fallback._bounded_input_for_request(request)
        if (
            cached_request.model != request.session.model
            or (self.model is not None and self.model != cached_request.model)
            or _has_structured_output_tool(cached_request.tools)
        ):
            return True
        # The exact cache path can switch to bounded fallback only after a
        # provider overflow, so boundedness is not knowable before execution.
        return None

    def _capture_invocation_identity(self) -> _PromptCacheCompactorInvocationIdentity:
        provider = self.provider
        if not isinstance(provider, ModelProvider):
            raise TypeError("provider must be a ModelProvider.")
        provider_snapshot = self._provider_snapshot
        if type(provider_snapshot) is not _CompactionProviderSnapshot:
            raise TypeError("_provider_snapshot must be a _CompactionProviderSnapshot.")
        model = self.model
        if model is not None:
            model = require_durable_clean_nonblank(model, "model")
        fallback = self._fallback
        if not isinstance(fallback, ContextCompactor):
            raise TypeError("fallback_compactor must be a ContextCompactor.")
        return _PromptCacheCompactorInvocationIdentity(
            provider=provider,
            provider_snapshot=_CompactionProviderSnapshot(
                provider_name=require_durable_clean_nonblank(
                    provider_snapshot.provider_name,
                    "provider.name",
                ),
                pricing_provider_name=require_durable_clean_nonblank(
                    provider_snapshot.pricing_provider_name,
                    "provider.billing_provider_name",
                ),
                usage_dialect=copy_usage_dialect(
                    provider_snapshot.usage_dialect,
                    "provider.usage_dialect",
                ),
            ),
            model=model,
            compactor_name=require_durable_clean_nonblank(
                self._compactor_name,
                "compactor",
            ),
            compaction_instruction=require_durable_nonblank(
                self.compaction_instruction,
                "compaction_instruction",
            ),
            options=copy_durable_json_object(self.options, "options"),
            retry_policy=copy_retry_policy(self.retry_policy),
            fallback=fallback,
        )

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        identity = self._capture_invocation_identity()
        provider_snapshot = identity.provider_snapshot
        provider_differs = provider_snapshot.provider_name != request.session.provider_name
        if provider_differs and identity.model is None:
            raise ValueError(
                "model is required when the compactor provider differs from the session provider."
            )
        bounded_model = identity.model if identity.model is not None else request.session.model

        if request.existing_summary is not None:
            return await self._compact_bounded(
                request,
                model=bounded_model,
                identity=identity,
            )

        if request.force_bounded_compaction:
            return await self._compact_bounded(
                request,
                model=bounded_model,
                identity=identity,
            )

        if provider_differs:
            return await self._compact_bounded(
                request,
                model=bounded_model,
                identity=identity,
            )

        cached_request = request.cache_prefix_request
        if cached_request is None:
            if identity.model is not None and identity.model != request.session.model:
                return await self._compact_bounded(
                    request,
                    model=identity.model,
                    identity=identity,
                )
            return await identity.fallback.compact(request)
        cached_model = cached_request.model
        if cached_model != request.session.model:
            return await self._compact_bounded(
                request,
                model=bounded_model,
                identity=identity,
            )
        if identity.model is not None and identity.model != cached_model:
            return await self._compact_bounded(
                request,
                model=identity.model,
                identity=identity,
            )
        model = identity.model if identity.model is not None else cached_model
        model = require_durable_clean_nonblank(model, "model")
        if _has_structured_output_tool(cached_request.tools):
            return await self._compact_bounded(
                request,
                model=model,
                identity=identity,
            )

        compaction_messages = [copy_message(message) for message in cached_request.messages]
        tools = copy_json_value(cached_request.tools, "cache_prefix_request.tools")
        base_options = cached_request.options
        compaction_messages.append(Message.text(MessageRole.USER, identity.compaction_instruction))
        options = _merged_json_options(base_options, identity.options)
        if "structured_output" in options:
            options["structured_output"] = None

        model_request = ModelRequest(
            model=model,
            messages=compaction_messages,
            tools=tools,
            targeted_tool_projection=cached_request.targeted_tool_projection,
            tool_discovery_projection=cached_request.tool_discovery_projection,
            options=options,
        )

        terminal_observation_error: ModelProviderError | None = None
        terminal_value_error: DurableValueError | None = None
        completion_ledger = _COMPACTION_COMPLETION_LEDGER.get()
        first_completion_index = (
            0 if completion_ledger is None else len(completion_ledger.completed_payloads)
        )
        try:
            summary, completed_metadata, completion_payloads = await _run_compaction_model(
                provider=identity.provider,
                provider_name=provider_snapshot.provider_name,
                pricing_provider_name=provider_snapshot.pricing_provider_name,
                model_request=model_request,
                retry_policy=identity.retry_policy,
                compactor=identity.compactor_name,
                usage_dialect=provider_snapshot.usage_dialect,
                observe_completion=_compaction_completion_observer(
                    provider_name=provider_snapshot.pricing_provider_name,
                    model=model,
                    compactor=identity.compactor_name,
                    usage_dialect=provider_snapshot.usage_dialect,
                ),
            )
        except _CompactionCompletionObservationError as exc:
            terminal_observation_error = _record_compaction_completion_observation_failure(
                exc,
                provider_name=provider_snapshot.pricing_provider_name,
                model=model,
                compactor=identity.compactor_name,
                usage_dialect=provider_snapshot.usage_dialect,
            )
        except _CompactionCompletionValueError as exc:
            terminal_value_error = _record_invalid_compaction_completion(
                exc,
                provider_name=provider_snapshot.pricing_provider_name,
                model=model,
                compactor=identity.compactor_name,
                usage_dialect=provider_snapshot.usage_dialect,
            )
        except _CompactionToolCallError as exc:
            if getattr(exc, "_cayu_compaction_budget_settlement_failed", False):
                raise
            completion_ledger = _COMPACTION_COMPLETION_LEDGER.get()
            recorded_tool_call = (
                None
                if completion_ledger is None or not completion_ledger.completed_payloads
                else completion_ledger.completed_payloads[-1]
            )
            if (
                recorded_tool_call is None
                or recorded_tool_call.get("compaction_outcome") != "rejected_tool_call"
            ):
                recorded_tool_call = _rejected_compaction_tool_call_payload(
                    error=exc,
                    provider_name=provider_snapshot.pricing_provider_name,
                    model=model,
                    compactor=identity.compactor_name,
                    usage_dialect=provider_snapshot.usage_dialect,
                )
            return await self._compact_bounded_after_exact_failure(
                request,
                model=model,
                exact_attempt="rejected_tool_call",
                exact_attempt_payload=recorded_tool_call,
                identity=identity,
            )
        except ModelContextOverflowError as exc:
            if getattr(exc, "_cayu_compaction_budget_settlement_failed", False):
                raise
            completion_ledger = _COMPACTION_COMPLETION_LEDGER.get()
            recorded_overflow = (
                None
                if completion_ledger is None or not completion_ledger.completed_payloads
                else completion_ledger.completed_payloads[-1]
            )
            if (
                recorded_overflow is None
                or recorded_overflow.get("compaction_outcome") != "context_overflow"
            ):
                recorded_overflow = _context_overflow_compaction_payload(
                    error=exc,
                    provider_name=provider_snapshot.pricing_provider_name,
                    model=model,
                    compactor=identity.compactor_name,
                    usage_dialect=provider_snapshot.usage_dialect,
                )
            return await self._compact_bounded_after_exact_failure(
                request,
                model=model,
                exact_attempt="context_overflow",
                exact_attempt_payload=recorded_overflow,
                identity=identity,
            )
        if terminal_observation_error is not None:
            await _publish_compaction_ledger_since(
                completion_ledger,
                first_completion_index,
            )
            raise terminal_observation_error from None
        if terminal_value_error is not None:
            await _publish_compaction_ledger_since(
                completion_ledger,
                first_completion_index,
            )
            raise terminal_value_error from None
        try:
            result = _provider_compaction_result(
                summary=summary,
                completed_metadata=completed_metadata,
                provider_name=provider_snapshot.pricing_provider_name,
                model=model,
                compactor=identity.compactor_name,
                usage_dialect=provider_snapshot.usage_dialect,
                metadata={
                    "prompt_cache_compaction": True,
                    "context_message_count": len(request.context_messages),
                    "attachment_results_preserved": len(
                        options.get(RESOLVED_FILE_ATTACHMENTS_OPTION, {})
                    ),
                },
                covered_message_count=len(request.messages),
            )
        except BaseException:
            await _publish_compaction_ledger_since(
                completion_ledger,
                first_completion_index,
            )
            raise
        await _publish_compaction_ledger_since(
            completion_ledger,
            first_completion_index,
        )
        return result.model_copy(
            update={
                "model_completed_payloads": completion_payloads,
                "source_chunk_count": 1,
                "source_chunk_mode": "provider_native_exact",
                "bounded_input": False,
            },
            deep=True,
        )

    async def _compact_bounded_after_exact_failure(
        self,
        request: CompactionRequest,
        *,
        model: str,
        exact_attempt: str,
        exact_attempt_payload: dict[str, Any],
        identity: _PromptCacheCompactorInvocationIdentity,
    ) -> CompactionResult:
        # Record this known-earlier failed attempt before the bounded call so a
        # later bounded failure is emitted in provider-call order.
        exact_attempt_payload = _record_compaction_model_completed_payloads(
            [exact_attempt_payload]
        )[0]
        await _publish_compaction_completion_payloads([exact_attempt_payload])
        bounded_result = await self._compact_bounded(
            request,
            model=model,
            identity=identity,
        )
        bounded_metadata = copy_json_value(bounded_result.metadata, "bounded_metadata")
        bounded_metadata["prompt_cache_exact_attempt"] = exact_attempt
        return CompactionResult(
            summary=bounded_result.summary,
            covered_message_count=bounded_result.covered_message_count,
            represented_existing_summary_sha256=(
                bounded_result.represented_existing_summary_sha256
            ),
            source_chunk_count=bounded_result.source_chunk_count,
            source_chunk_mode=bounded_result.source_chunk_mode,
            bounded_input=bounded_result.bounded_input,
            metadata=bounded_metadata,
            model_completed_payloads=[
                exact_attempt_payload,
                *bounded_result.model_completed_payloads,
            ],
        )

    async def _compact_bounded(
        self,
        request: CompactionRequest,
        *,
        model: str,
        identity: _PromptCacheCompactorInvocationIdentity,
    ) -> CompactionResult:
        incremental_options = _merged_json_options(
            request.agent.provider_options,
            identity.options,
        )
        incremental_options.pop(RESOLVED_FILE_ATTACHMENTS_OPTION, None)
        if "structured_output" in incremental_options:
            incremental_options["structured_output"] = None
        incremental_compactor = ModelCompactor(
            provider=identity.provider,
            model=require_durable_clean_nonblank(model, "model"),
            system_prompt=identity.compaction_instruction,
            options=incremental_options,
            retry_policy=identity.retry_policy,
            _provider_snapshot=identity.provider_snapshot,
        )
        return await incremental_compactor.compact(request)


def _merged_json_options(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy_json_value(base, "base_options")
    for key, value in override.items():
        existing = result.get(key)
        if type(existing) is dict and type(value) is dict:
            result[key] = _merged_json_options(existing, value)
        else:
            result[key] = copy_json_value(value, f"options.{key}")
    return result


def _has_structured_output_tool(tools: list[dict[str, Any]]) -> bool:
    return any(tool.get("name") == STRUCTURED_OUTPUT_TOOL_NAME for tool in tools)


def _compaction_completion_observer(
    *,
    provider_name: str,
    model: str,
    compactor: str,
    usage_dialect: UsageDialect,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    provider_name = require_durable_clean_nonblank(
        provider_name,
        "provider.billing_provider_name",
    )

    def observe(completed_metadata: dict[str, Any]) -> dict[str, Any]:
        observed_metadata = copy_json_value(completed_metadata, "completed_metadata")
        # This correlation key is runtime-owned; provider metadata cannot select or
        # overwrite another compaction attempt's ledger entry.
        observed_metadata.pop(_COMPACTION_ATTEMPT_ID_KEY, None)
        strip_runtime_owned_execution_identity(observed_metadata)
        accounting_usage_error: _CompactionAccountingUsageError | None = None
        try:
            payload = _compaction_model_completed_payload(
                completed_payload=observed_metadata,
                provider_name=provider_name,
                fallback_model=model,
                compactor=compactor,
                usage_dialect=usage_dialect,
            )
        except _CompactionAccountingUsageError as exc:
            accounting_usage_error = exc
            durable_payload = exc.payload
        else:
            durable_payload = _durable_compaction_completion_evidence(
                payload,
                provider_name=provider_name,
                fallback_model=model,
                compactor=compactor,
            )
        registered_payload = _record_compaction_model_completed_payloads([durable_payload])[0]
        attempt_id = registered_payload.get(_COMPACTION_ATTEMPT_ID_KEY)
        if type(attempt_id) is str:
            observed_metadata[_COMPACTION_ATTEMPT_ID_KEY] = attempt_id
        for key in ("model_step_id", "model_attempt_id"):
            value = registered_payload.get(key)
            if type(value) is str:
                observed_metadata[key] = value
        if accounting_usage_error is not None:
            raise _CompactionCompletionValueError(
                error=accounting_usage_error.error,
                completed_metadata=observed_metadata,
                rejected_usage_payload=registered_payload,
            )
        return observed_metadata

    return observe


def _completion_metadata_with_billing_identity(
    completed_metadata: dict[str, Any],
    identity: BillingIdentity | None,
) -> dict[str, Any]:
    """Attach only a runtime-resolved identity to detached completion facts."""

    observed_metadata = copy_durable_json_object(
        completed_metadata,
        "completed_metadata",
    )
    strip_runtime_owned_execution_identity(observed_metadata)
    strip_provider_billing_identity(observed_metadata)
    if identity is not None:
        observed_metadata["billing_identity"] = identity.model_dump(mode="json")
    return copy_durable_json_object(observed_metadata, "completed_metadata")


def _observe_terminal_compaction_evidence(
    observe_completion: Callable[[dict[str, Any]], dict[str, Any]],
    completed_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Best-effort ledger publication after provider completion is terminal."""

    detached = copy_durable_json_object(completed_metadata, "completed_metadata")
    try:
        return observe_completion(detached)
    except _CompactionCompletionValueError:
        # Usage overflow is an authoritative terminal classification, not a
        # best-effort observer failure. Preserve its registered attempt id and
        # let the caller publish exactly that rejected completion.
        raise
    except Exception:
        # The caller propagates a separately detached authoritative failure.
        # Retain the portable facts for outer telemetry publication without
        # attaching this potentially provider-owned exception as context.
        return copy_durable_json_object(completed_metadata, "completed_metadata")


def _safe_compaction_durable_value_error(error: DurableValueError) -> DurableValueError:
    """Rebuild a public/mutable validation error without its caller label."""

    code, path = safe_durable_value_error_details(error)
    return DurableValueError(code, "provider stream value", path=path)


def _compaction_provider_failure_control(
    error: Exception,
    *,
    fallback_provider: str,
) -> ProviderExceptionControl:
    """Detach one provider exception without retaining extension-owned state."""

    if isinstance(error, DurableValueError):
        safe_error = _safe_compaction_durable_value_error(error)
        return ProviderExceptionControl(
            message=str(safe_error),
            error_type=type(safe_error).__name__,
            cause=safe_error,
        )
    try:
        return copy_provider_exception_control(error)
    except DurableValueError as portability_error:
        safe_error, _ = nonportable_model_provider_error(
            portability_error,
            fallback_provider=fallback_provider,
        )
        return ProviderExceptionControl(
            message=str(safe_error),
            error_type=type(safe_error).__name__,
            cause=safe_error,
        )


def _detach_compaction_model_request(request: ModelRequest) -> ModelRequest:
    """Revalidate and detach one request before exposing it to a provider."""

    if type(request) is not ModelRequest:
        raise TypeError("request must be a ModelRequest.")
    return ModelRequest(
        model=request.model,
        messages=request.messages,
        tools=request.tools,
        hosted_tools=request.hosted_tools,
        targeted_tool_projection=request.targeted_tool_projection,
        tool_discovery_projection=request.tool_discovery_projection,
        options=request.options,
    )


def _compaction_metadata_with_model_attempt_identity(
    completed_metadata: dict[str, Any],
    identity: ModelAttemptIdentity | None,
) -> dict[str, Any]:
    detached = copy_durable_json_object(completed_metadata, "completed_metadata")
    if identity is not None:
        detached.update(copy_model_attempt_identity(identity).payload())
    return detached


def _identify_compaction_dispatch_failure(
    error: BaseException,
    identity: ModelAttemptIdentity | None,
) -> None:
    """Retain exact execution identity on runtime-owned terminal metadata."""

    if identity is None:
        return
    identity_payload = copy_model_attempt_identity(identity).payload()
    if type(error) in {
        _CompactionCompletionValueError,
        _CompactionCompletionObservationError,
        _CompactionToolCallError,
        _ProviderDispatchFailed,
    } or isinstance(error, asyncio.CancelledError):
        completed_metadata = error.__dict__.get("completed_metadata")
        if type(completed_metadata) is dict:
            identified_metadata = copy_durable_json_object(
                completed_metadata,
                "completed_metadata",
            )
            identified_metadata.update(identity_payload)
            error.__dict__["completed_metadata"] = identified_metadata
    if type(error) is _CompactionCompletionValueError:
        rejected_usage_payload = error.__dict__.get("rejected_usage_payload")
        if type(rejected_usage_payload) is dict:
            identified_payload = copy_durable_json_object(
                rejected_usage_payload,
                "rejected_usage_payload",
            )
            identified_payload.update(identity_payload)
            error.__dict__["rejected_usage_payload"] = identified_payload


async def _await_owned_compaction_provider_stream(
    operation: Awaitable[tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]]:
    """Settle one governed stream without losing caller cancellation identity.

    The governed provider boundary deliberately replaces provider-visible
    cancellation details with a credential-safe projection. Automatic
    compaction still needs the original caller cancellation as the authority
    carried through durable completion and budget settlement. Keep that signal
    in this runtime-owned task and send only a generic cancellation into the
    opaque provider child, then wait until the governed child has actually
    settled before returning control to accounting.
    """

    owner_task = asyncio.current_task()
    if owner_task is None:  # pragma: no cover - coroutine execution invariant
        raise RuntimeError("Automatic compaction requires an owning task.")
    cancellation_requests = owner_task.cancelling()
    provider_task = asyncio.ensure_future(operation)
    caller_cancellation: asyncio.CancelledError | None = None
    try:
        while True:
            try:
                result = await asyncio.shield(provider_task)
            except asyncio.CancelledError as exc:
                current_requests = owner_task.cancelling()
                if current_requests > cancellation_requests:
                    cancellation_requests = current_requests
                    if caller_cancellation is None:
                        caller_cancellation = exc
                    else:
                        caller_cancellation.add_note(
                            "Automatic compaction provider settlement received an "
                            "additional caller cancellation request."
                        )
                    if not provider_task.done():
                        provider_task.cancel("Automatic compaction provider cancelled")
                    continue
                if caller_cancellation is not None:
                    raise caller_cancellation from None
                if provider_task.done():
                    return provider_task.result()
                raise
            except BaseException as provider_failure:
                if caller_cancellation is None:
                    raise
                if isinstance(provider_failure, (GeneratorExit, KeyboardInterrupt, SystemExit)):
                    raise provider_failure from caller_cancellation
                raise caller_cancellation from provider_failure
            if caller_cancellation is not None:
                raise caller_cancellation
            return result
    finally:
        if not provider_task.done():
            provider_task.cancel("Automatic compaction provider owner terminated")
            provider_task.add_done_callback(_consume_compaction_provider_task_outcome)


def _consume_compaction_provider_task_outcome(
    task: asyncio.Future[tuple[str, dict[str, Any]]],
) -> None:
    """Consume a retained compactor task after process-control abandonment."""

    with suppress(BaseException):
        task.result()


async def _run_compaction_model(
    *,
    provider: ModelProvider,
    provider_name: str,
    pricing_provider_name: str,
    model_request: ModelRequest,
    retry_policy: RetryPolicy,
    compactor: str,
    usage_dialect: UsageDialect,
    observe_completion: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    # Keep this validated template private. Provider hooks and each stream
    # attempt receive independent copies so mutation cannot alter a later
    # dispatch or the logical model used for accounting.
    provider_name = require_durable_clean_nonblank(provider_name, "provider.name")
    pricing_provider_name = require_durable_clean_nonblank(
        pricing_provider_name,
        "provider.billing_provider_name",
    )
    request_template = _detach_compaction_model_request(model_request)
    runtime_dispatch_boundary = _DEFER_BILLING_IDENTITY_CANCELLATION.get()
    billing_identity: BillingIdentity | None = None
    billing_cancellation: asyncio.CancelledError | None = None
    billing_failure: ModelProviderError | None = None
    try:
        billing_identity = await resolve_request_billing_identity(
            provider,
            _detach_compaction_model_request(request_template),
            provider_name=provider_name,
        )
    except asyncio.CancelledError as exc:
        billing_cancellation = detach_billing_identity_cancellation(exc)
        if billing_cancellation is None:
            raise
        if runtime_dispatch_boundary:
            del provider
            raise
    except ModelProviderError as exc:
        billing_failure = exc
    if billing_cancellation is not None:
        del provider
        raise billing_cancellation
    if billing_failure is not None:
        del provider
        raise billing_failure

    def observe_completion_with_billing_identity(
        completed_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        provider_metadata = copy_durable_json_object(
            completed_metadata,
            "completed_metadata",
        )
        strip_runtime_owned_execution_identity(provider_metadata)
        strip_provider_billing_identity(provider_metadata)
        best_known_metadata = _completion_metadata_with_billing_identity(
            provider_metadata,
            billing_identity,
        )
        hook_error: ModelProviderError | None = None
        try:
            completed_identity = resolve_completion_billing_identity(
                provider,
                billing_identity,
                copy_durable_json_object(
                    provider_metadata,
                    "completed_metadata",
                ),
                provider_name=provider_name,
            )
        except ModelProviderError as exc:
            hook_error = exc
        if hook_error is not None:
            # A provider completion is terminal even when identity enrichment
            # fails. Register the already-known request identity and usage
            # before propagating the detached hook failure so reservation
            # settlement sees this dispatch exactly once.
            observed_metadata = _observe_terminal_compaction_evidence(
                observe_completion,
                best_known_metadata,
            )
            raise _CompactionCompletionObservationError(
                error=hook_error,
                completed_metadata=observed_metadata,
            ) from None

        observed_metadata = _completion_metadata_with_billing_identity(
            provider_metadata,
            completed_identity,
        )
        observer_error: ModelProviderError | None = None
        try:
            return observe_completion(observed_metadata)
        except _CompactionCompletionValueError:
            # This is a runtime-owned terminal classification produced after
            # the provider completed, not a failure of the observer hook.
            raise
        except Exception as exc:
            observer_error, _ = copy_provider_hook_error_control(
                exc,
                fallback_provider=provider_name,
                generic_error_code="completion_observation_failed",
            )
        fallback_metadata = observed_metadata
        if "usage" in fallback_metadata:
            fallback_metadata = portable_model_completion_projection(
                fallback_metadata,
                provider_name=provider_name,
                requested_model=request_template.model,
                usage_dialect=usage_dialect,
            )
        fallback_metadata = _observe_terminal_compaction_evidence(
            observe_completion,
            fallback_metadata,
        )
        assert observer_error is not None
        raise _CompactionCompletionObservationError(
            error=observer_error,
            completed_metadata=fallback_metadata,
        ) from None

    dispatch_started = False
    dispatch_cancellation_requests = 0
    dispatch_model_attempt_identity: ModelAttemptIdentity | None = None

    async def dispatch() -> tuple[str, dict[str, Any]]:
        nonlocal dispatch_cancellation_requests
        nonlocal dispatch_model_attempt_identity
        nonlocal dispatch_started
        dispatch_counter = _COMPACTION_DISPATCH_COUNTER.get()
        if dispatch_counter is not None:
            dispatch_counter.before_dispatch()
        active_identity = _COMPACTION_MODEL_ATTEMPT_IDENTITY.get()
        dispatch_model_attempt_identity = (
            None if active_identity is None else copy_model_attempt_identity(active_identity)
        )
        current_task = asyncio.current_task()
        dispatch_cancellation_requests = 0 if current_task is None else current_task.cancelling()
        dispatch_started = True
        provider_failure: ProviderExceptionControl | None = None
        try:
            summary, completed_metadata = await _await_owned_compaction_provider_stream(
                _stream_compaction_model(
                    provider=provider,
                    provider_name=provider_name,
                    model_request=_detach_compaction_model_request(request_template),
                    usage_dialect=usage_dialect,
                    observe_completion=observe_completion_with_billing_identity,
                )
            )
        except (
            _CompactionCompletionValueError,
            _CompactionCompletionObservationError,
            _CompactionToolCallError,
            _ProviderDispatchFailed,
        ) as exc:
            _identify_compaction_dispatch_failure(
                exc,
                dispatch_model_attempt_identity,
            )
            raise
        except asyncio.CancelledError as exc:
            _identify_compaction_dispatch_failure(
                exc,
                dispatch_model_attempt_identity,
            )
            raise
        except Exception as exc:
            provider_failure = _compaction_provider_failure_control(
                exc,
                fallback_provider=provider_name,
            )
        else:
            return (
                summary,
                _compaction_metadata_with_model_attempt_identity(
                    completed_metadata,
                    dispatch_model_attempt_identity,
                ),
            )
        if provider_failure is None:  # pragma: no cover - every Exception is captured above
            raise RuntimeError("Provider dispatch lost its failure state.")
        raise _ProviderDispatchFailed(provider_failure) from None

    existing_ledger = _COMPACTION_COMPLETION_LEDGER.get()
    owns_ledger = existing_ledger is None
    completion_ledger = (
        _CompactionCompletionLedger() if existing_ledger is None else existing_ledger
    )
    first_completion_index = len(completion_ledger.completed_payloads)
    completion_ledger_token = (
        _COMPACTION_COMPLETION_LEDGER.set(completion_ledger) if owns_ledger else None
    )
    try:
        attempt = 1
        terminal_dispatch_error: BaseException | None = None
        terminal_dispatch_cause: BaseException | None = None
        while True:
            dispatch_started = False
            dispatch_model_attempt_identity = None
            attempt_completion_index = len(completion_ledger.completed_payloads)
            billing_dispatch_cancellation: asyncio.CancelledError | None = None
            try:
                run_dispatch = _AUTOMATIC_COMPACTION_DISPATCH_RUNNER.get()
                if run_dispatch is None:
                    summary, completed_metadata = await dispatch()
                else:
                    summary, completed_metadata = await run_dispatch(
                        provider,
                        provider_name,
                        pricing_provider_name,
                        request_template.model,
                        usage_dialect,
                        billing_identity,
                        _detach_compaction_model_request(request_template),
                        attempt,
                        retry_policy.max_attempts,
                        dispatch,
                    )
                completion_payloads = copy_durable_json_value(
                    completion_ledger.completed_payloads[first_completion_index:],
                    "model_completed_payloads",
                )
                if owns_ledger:
                    completion_payloads = [
                        _public_compaction_model_completed_payload(payload)
                        for payload in completion_payloads
                    ]
                return summary, completed_metadata, completion_payloads
            except BaseException as exc:
                if isinstance(
                    exc,
                    (
                        _CompactionCompletionValueError,
                        _CompactionCompletionObservationError,
                    ),
                ):
                    # These terminal classifications already own any completion
                    # evidence in the ledger. Their caller annotates and publishes
                    # that exact record before deciding whether fallback is safe.
                    del provider
                    raise

                provider_failure = exc.control if isinstance(exc, _ProviderDispatchFailed) else None
                failure = provider_failure.cause if provider_failure is not None else exc
                failure_type = (
                    provider_failure.error_type
                    if provider_failure is not None
                    else type(failure).__name__
                )
                attempt_payloads = completion_ledger.completed_payloads[attempt_completion_index:]
                if attempt_payloads:
                    finalized_attempt_payloads: list[dict[str, Any]] = []
                    for payload in attempt_payloads:
                        failed_payload = copy_durable_json_object(
                            payload,
                            "model_completed_payload",
                        )
                        if isinstance(failure, ModelContextOverflowError):
                            failed_payload.update(
                                _context_overflow_compaction_payload(
                                    error=failure,
                                    provider_name=pricing_provider_name,
                                    model=request_template.model,
                                    compactor=compactor,
                                    usage_dialect=usage_dialect,
                                )
                            )
                            if "usage" in failed_payload or "usage_metrics" in failed_payload:
                                failed_payload.pop("usage_unavailable_reason", None)
                        elif isinstance(failure, asyncio.CancelledError):
                            failed_payload["compaction_outcome"] = "cancelled_after_completion"
                            failed_payload["error_type"] = failure_type
                        elif isinstance(failure, _CompactionToolCallError):
                            failed_payload["compaction_outcome"] = "rejected_tool_call"
                            failed_payload["error_type"] = failure_type
                        else:
                            failed_payload["compaction_outcome"] = "provider_error_after_completion"
                            failed_payload["error_type"] = failure_type
                        if dispatch_model_attempt_identity is not None:
                            failed_payload.update(
                                copy_model_attempt_identity(
                                    dispatch_model_attempt_identity
                                ).payload()
                            )
                        finalized_attempt_payloads.extend(
                            _record_compaction_model_completed_payloads([failed_payload])
                        )
                elif dispatch_started:
                    failed_attempt_payload = (
                        _context_overflow_compaction_payload(
                            error=failure,
                            provider_name=pricing_provider_name,
                            model=request_template.model,
                            compactor=compactor,
                            usage_dialect=usage_dialect,
                        )
                        if isinstance(failure, ModelContextOverflowError)
                        else _failed_compaction_provider_attempt_payload(
                            error=failure,
                            error_type=failure_type,
                            provider_name=pricing_provider_name,
                            model=request_template.model,
                            compactor=compactor,
                            usage_dialect=usage_dialect,
                        )
                    )
                    if dispatch_model_attempt_identity is not None:
                        failed_attempt_payload.update(
                            copy_model_attempt_identity(dispatch_model_attempt_identity).payload()
                        )
                    finalized_attempt_payloads = _record_compaction_model_completed_payloads(
                        [failed_attempt_payload]
                    )
                else:
                    finalized_attempt_payloads = []
                if finalized_attempt_payloads:
                    if isinstance(failure, asyncio.CancelledError):
                        current_task = asyncio.current_task()
                        current_requests = 0 if current_task is None else current_task.cancelling()
                        if current_requests > dispatch_cancellation_requests:
                            # A caller cancellation delivered after dispatch is
                            # authoritative. Preserve older handled requests while
                            # normalizing only the new signal before publication.
                            consume_pending_task_cancellation(
                                failure,
                                preserve_requests=dispatch_cancellation_requests,
                            )
                    try:
                        await _publish_compaction_completion_payloads(finalized_attempt_payloads)
                    except asyncio.CancelledError as publication_cancellation:
                        failure.add_note(
                            "Compaction provider failure evidence publication was interrupted "
                            "by cancellation; publication diagnostics are attached to the "
                            "cancellation."
                        )
                        if isinstance(failure, asyncio.CancelledError):
                            # The publisher redelivers the caller cancellation
                            # after durable cleanup. Do not let that duplicate
                            # signal overwrite the original cancellation's
                            # authoritative settlement/provider cause.
                            public_cancellation = detach_billing_identity_cancellation(failure)
                            if public_cancellation is not None:
                                if runtime_dispatch_boundary:
                                    del provider
                                    raise failure from exception_cause(failure)
                                del provider
                                raise public_cancellation from None
                            del provider
                            raise failure from exception_cause(failure)
                        del provider
                        raise publication_cancellation from failure
                    except Exception as publication_error:
                        if isinstance(failure, ModelProviderError):
                            failure.retryable = False
                        publication_disposition = automatic_compaction_failure_disposition_payload(
                            publication_error
                        )
                        if publication_disposition is not None:
                            _attach_automatic_compaction_failure_disposition(
                                failure,
                                _AutomaticCompactionFailureDisposition.model_validate(
                                    publication_disposition
                                ),
                            )
                        failure.add_note(
                            "Compaction provider failure evidence publication also failed: "
                            f"{type(publication_error).__name__}: {publication_error}"
                        )
                        del provider
                        raise failure from publication_error

                if isinstance(failure, asyncio.CancelledError):
                    billing_dispatch_cancellation = detach_billing_identity_cancellation(failure)
                    if billing_dispatch_cancellation is not None and runtime_dispatch_boundary:
                        del provider
                        raise
                if billing_dispatch_cancellation is None:
                    if provider_failure is None:
                        del provider
                        raise
                    if exc.__dict__.get("_cayu_compaction_budget_settlement_failed") is True:
                        failure.__dict__["_cayu_compaction_budget_settlement_failed"] = True
                        if isinstance(failure, ModelProviderError):
                            failure.retryable = False
                        terminal_dispatch_cause = exception_cause(exc)
                        if terminal_dispatch_cause is not None:
                            failure.add_note(
                                "Automatic compaction budget settlement also failed: "
                                f"{type(terminal_dispatch_cause).__name__}: "
                                f"{terminal_dispatch_cause}"
                            )
                        terminal_dispatch_error = failure
                        break
                    provider_error = failure if isinstance(failure, ModelProviderError) else None
                    decision = retry_decision(
                        policy=retry_policy,
                        attempt=attempt,
                        error=provider_failure.message,
                        status_code=(
                            None if provider_error is None else provider_error.status_code
                        ),
                        retryable=(None if provider_error is None else provider_error.retryable),
                        retry_after_s=(
                            None if provider_error is None else provider_error.retry_after_s
                        ),
                        unknown_provider_error=(
                            provider_error is not None
                            and provider_error.status_code is None
                            and provider_error.retryable is None
                        ),
                    )
                    if not decision.retry or decision.next_attempt is None:
                        terminal_dispatch_error = failure
                        break
                    if decision.delay_seconds > 0:
                        await asyncio.sleep(decision.delay_seconds)
                    attempt = decision.next_attempt
            if billing_dispatch_cancellation is not None:
                del provider
                raise billing_dispatch_cancellation
        if terminal_dispatch_error is not None:
            del provider
            raise terminal_dispatch_error from terminal_dispatch_cause
        raise RuntimeError("Compaction dispatch exited without a result.")
    finally:
        if completion_ledger_token is not None:
            _COMPACTION_COMPLETION_LEDGER.reset(completion_ledger_token)


async def _stream_compaction_model(
    *,
    provider: ModelProvider,
    provider_name: str,
    model_request: ModelRequest,
    usage_dialect: UsageDialect,
    observe_completion: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    provider_name = require_durable_clean_nonblank(provider_name, "provider.name")
    text_parts: list[str] = []
    completed_payload: dict[str, Any] | None = None
    tool_call_seen = False
    tool_call_failure: _CompactionToolCallError | None = None
    completed_dispatch_failure: _ProviderDispatchFailed | None = None
    try:
        async for raw_event in provider.runtime_stream(model_request):
            event = _copy_compaction_stream_event_for_accounting(raw_event)
            if completed_payload is not None:
                raise RuntimeError(
                    f"Compaction provider emitted event after completed: {event.type}"
                )
            if event.type == ModelStreamEventType.TEXT_DELTA:
                text_parts.append(event.delta)
            elif event.type == ModelStreamEventType.THINKING:
                continue
            elif event.type == ModelStreamEventType.TOOL_CALL:
                tool_call_seen = True
            elif event.type == ModelStreamEventType.ERROR:
                provider_error = model_provider_error_from_payload(
                    event.payload,
                    fallback_provider=provider_name,
                    fallback_message="Compaction model provider error",
                )
                if provider_error is not None:
                    raise provider_error
                raise RuntimeError(
                    str(event.payload.get("error") or "Compaction model provider error")
                )
            elif event.type == ModelStreamEventType.COMPLETED:
                try:
                    portable_payload = copy_durable_json_object(event.payload, "payload")
                except DurableValueError as exc:
                    safe_payload = portable_model_completion_projection(
                        event.payload,
                        provider_name=provider_name,
                        requested_model=model_request.model,
                        usage_dialect=usage_dialect,
                    )
                    safe_completed_metadata = _provider_completed_metadata(
                        safe_payload,
                        preserve_usage_metrics=True,
                        preserve_usage_failure=True,
                    )
                    try:
                        completed_payload = observe_completion(safe_completed_metadata)
                    except _CompactionCompletionObservationError as observation_failure:
                        # The portability failure is authoritative once a
                        # terminal completion has been observed. In particular,
                        # a failing completion-billing hook must not redispatch
                        # the model and erase this attempt's known spend.
                        completed_payload = observation_failure.completed_metadata
                    raise _CompactionCompletionValueError(
                        error=exc,
                        completed_metadata=completed_payload,
                    ) from None
                completed_payload = observe_completion(
                    _provider_completed_metadata(portable_payload)
                )
            else:
                raise RuntimeError(f"Compaction provider emitted unsupported event: {event.type}")
    except asyncio.CancelledError as exc:
        if completed_payload is not None:
            exc.__dict__["completed_metadata"] = copy_json_value(
                completed_payload,
                "completed_metadata",
            )
        raise
    except Exception as exc:
        del provider
        if isinstance(exc, ModelStreamDeadlineError):
            # A deadline remains authoritative even after the exact cache-prefix
            # request emitted a forbidden tool call. Reclassifying it as ordinary
            # tool-call degradation would authorize another provider dispatch
            # while the timed-out operation's outcome is still unknown.
            raise
        if isinstance(
            exc,
            (_CompactionCompletionValueError, _CompactionCompletionObservationError),
        ):
            # Failures discovered only after a terminal completion are
            # authoritative. A prior tool-call event must not reinterpret them
            # as a protocol fallback eligible for another provider dispatch.
            raise
        if tool_call_seen:
            completed_metadata = None if completed_payload is None else completed_payload
            tool_call_failure = _CompactionToolCallError(
                completed_metadata=completed_metadata,
            )
        elif completed_payload is not None:
            completed_dispatch_failure = _ProviderDispatchFailed(
                _compaction_provider_failure_control(
                    exc,
                    fallback_provider=provider_name,
                ),
                completed_metadata=completed_payload,
            )
        else:
            raise

    if tool_call_failure is not None:
        raise tool_call_failure from None
    if completed_dispatch_failure is not None:
        raise completed_dispatch_failure from None

    if completed_payload is None:
        if tool_call_seen:
            raise _CompactionToolCallError(completed_metadata=None)
        raise RuntimeError("Compaction model stream ended without a completed event.")
    completed_metadata = completed_payload
    if tool_call_seen:
        raise _CompactionToolCallError(completed_metadata=completed_metadata)
    return "".join(text_parts), completed_metadata


def _copy_compaction_stream_event_for_accounting(value: object) -> ModelStreamEvent:
    """Detach an ephemeral provider event without discarding terminal usage.

    Compaction validates its final summary and projects completion metadata to
    a portable accounting record before persistence. Strict per-event copying
    here would abort on an unrelated malformed completion field before that
    safe projection can retain provider-reported spend.
    """

    if type(value) is not ModelStreamEvent:
        raise TypeError("Model providers must yield ModelStreamEvent instances.")
    if type(value.type) is not ModelStreamEventType:
        raise ValueError("Model provider stream event type must be a ModelStreamEventType.")
    if type(value.delta) is not str:
        raise ValueError("Model provider stream event delta must be a string.")
    if type(value.payload) is not dict:
        raise ValueError("Model provider stream event payload must be an object.")
    if value.type != ModelStreamEventType.COMPLETED and value.completion is not None:
        raise ValueError("Only completed model stream events can include completion metadata.")
    if value.type == ModelStreamEventType.COMPLETED:
        # Completion metadata needs the accounting-aware projection below so a
        # malformed auxiliary field cannot erase provider-reported spend.
        delta = value.delta
        payload = dict(value.payload)
    elif value.type == ModelStreamEventType.ERROR:
        # Error payloads are control data: retry classification and externally
        # visible failure events must never be derived from a value that cannot
        # cross the durable boundary. There is no completion usage to retain.
        delta = require_durable_text(value.delta, "delta")
        payload = copy_durable_json_object(value.payload, "payload")
    else:
        delta = value.delta
        payload = copy_json_value(value.payload, "payload")
    return ModelStreamEvent.model_construct(
        type=value.type,
        delta=delta,
        payload=payload,
        completion=None,
    )


def _provider_compaction_result(
    *,
    summary: str,
    completed_metadata: dict[str, Any],
    provider_name: str,
    model: str,
    compactor: str,
    usage_dialect: UsageDialect,
    metadata: dict[str, Any],
    covered_message_count: int,
) -> CompactionResult:
    provider_name = require_durable_clean_nonblank(
        provider_name,
        "provider.billing_provider_name",
    )
    # Build attributable evidence before validating the summary so completed spend
    # survives an unusable-text failure.
    model_completed_payload = _compaction_model_completed_payload(
        completed_payload=completed_metadata,
        provider_name=provider_name,
        fallback_model=model,
        compactor=compactor,
        usage_dialect=usage_dialect,
    )
    ledger_payload = _durable_compaction_completion_evidence(
        model_completed_payload,
        provider_name=provider_name,
        fallback_model=model,
        compactor=compactor,
    )
    registered_payload = _record_compaction_model_completed_payloads([ledger_payload])[0]
    attempt_id = registered_payload.get(_COMPACTION_ATTEMPT_ID_KEY)
    if type(attempt_id) is str:
        model_completed_payload[_COMPACTION_ATTEMPT_ID_KEY] = attempt_id
    try:
        validated_summary = _validate_compaction_summary(summary)
    except ValueError:
        invalid_summary_payload = copy_json_value(
            registered_payload,
            "model_completed_payload",
        )
        invalid_summary_payload["compaction_outcome"] = "invalid_summary"
        _record_compaction_model_completed_payloads([invalid_summary_payload])
        raise
    try:
        public_completed_metadata = copy_durable_json_object(
            completed_metadata,
            "completed_metadata",
        )
    except DurableValueError:
        invalid_metadata_payload = copy_json_value(
            registered_payload,
            "model_completed_payload",
        )
        invalid_metadata_payload["compaction_outcome"] = "invalid_completion_metadata"
        _record_compaction_model_completed_payloads([invalid_metadata_payload])
        raise
    public_completed_metadata.pop(_COMPACTION_ATTEMPT_ID_KEY, None)
    return CompactionResult(
        summary=validated_summary,
        covered_message_count=covered_message_count,
        metadata={
            "compactor": compactor,
            "provider": provider_name,
            "model": model,
            **copy_json_value(metadata, "metadata"),
            "completed": public_completed_metadata,
        },
        model_completed_payloads=[model_completed_payload],
    )


def _record_compaction_completion_observation_failure(
    failure: _CompactionCompletionObservationError,
    *,
    provider_name: str,
    model: str,
    compactor: str,
    usage_dialect: UsageDialect,
) -> ModelProviderError:
    """Publish best-known terminal spend before propagating a safe hook error."""

    provider_name = require_durable_clean_nonblank(
        provider_name,
        "provider.billing_provider_name",
    )
    model_completed_payload = _compaction_model_completed_payload(
        completed_payload=failure.completed_metadata,
        provider_name=provider_name,
        fallback_model=model,
        compactor=compactor,
        usage_dialect=usage_dialect,
    )
    durable_payload = _durable_compaction_completion_evidence(
        model_completed_payload,
        provider_name=provider_name,
        fallback_model=model,
        compactor=compactor,
    )
    durable_payload["compaction_outcome"] = "completion_observation_failed"
    _record_compaction_model_completed_payloads([durable_payload])
    if failure.__dict__.get("_cayu_compaction_budget_settlement_failed") is True:
        failure.error.__dict__["_cayu_compaction_budget_settlement_failed"] = True
        failure.error.retryable = False
    return failure.error


def _record_invalid_compaction_completion(
    failure: _CompactionCompletionValueError,
    *,
    provider_name: str,
    model: str,
    compactor: str,
    usage_dialect: UsageDialect,
) -> DurableValueError:
    """Publish safe terminal spend before propagating a completion value error."""

    provider_name = require_durable_clean_nonblank(
        provider_name,
        "provider.billing_provider_name",
    )
    if failure.rejected_usage_payload is not None:
        durable_payload = copy_durable_json_object(
            failure.rejected_usage_payload,
            "model_completed_payload",
        )
    else:
        model_completed_payload = _compaction_model_completed_payload(
            completed_payload=failure.completed_metadata,
            provider_name=provider_name,
            fallback_model=model,
            compactor=compactor,
            usage_dialect=usage_dialect,
        )
        durable_payload = _durable_compaction_completion_evidence(
            model_completed_payload,
            provider_name=provider_name,
            fallback_model=model,
            compactor=compactor,
            force_projection=True,
        )
    if (
        "usage" not in durable_payload
        and "usage_metrics" not in durable_payload
        and "usage_unavailable_reason" not in durable_payload
    ):
        durable_payload["usage_unavailable_reason"] = "invalid compaction usage telemetry"
    durable_payload["compaction_outcome"] = "invalid_completion_metadata"
    _record_compaction_model_completed_payloads([durable_payload])
    return failure.error


def _rejected_compaction_usage_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build portable terminal evidence when normalized counters overflow."""

    rejected = copy_json_value(payload, "model_completed_payload")
    raw_usage = rejected.pop("usage", None)
    rejected.pop("usage_metrics", None)
    if raw_usage is not None:
        rejected["rejected_usage_evidence"] = copy_durable_json_value(
            raw_usage,
            "rejected_usage_evidence",
        )
    rejected["usage_metrics_rejected"] = True
    return copy_durable_json_object(rejected, "model_completed_payload")


def _durable_compaction_completion_evidence(
    payload: dict[str, Any],
    *,
    provider_name: str,
    fallback_model: str,
    compactor: str,
    force_projection: bool = False,
) -> dict[str, Any]:
    """Retain full durable metadata or a safe normalized accounting projection."""

    copied = copy_json_value(payload, "model_completed_payload")
    try:
        portable = copy_durable_json_object(copied, "model_completed_payload")
    except DurableValueError:
        portable = None
    if portable is not None and not force_projection:
        return portable

    resolved_model = copied.get("model")
    try:
        resolved_model = require_nonblank(
            require_durable_text(resolved_model, "model"),
            "model",
        )
    except ValueError:
        resolved_model = fallback_model
    fallback_fields: dict[str, Any] = {
        "model": resolved_model,
        "provider_name": provider_name,
        "requested_model": fallback_model,
        "purpose": ModelCompletionPurpose.CONTEXT_COMPACTION.value,
        "compactor": compactor,
    }
    attempt_id = copied.get(_COMPACTION_ATTEMPT_ID_KEY)
    if type(attempt_id) is str:
        try:
            attempt_id = require_durable_nonblank(
                attempt_id,
                _COMPACTION_ATTEMPT_ID_KEY,
            )
        except ValueError:
            pass
        else:
            fallback_fields[_COMPACTION_ATTEMPT_ID_KEY] = attempt_id
    if "model_step_id" in copied or "model_attempt_id" in copied:
        try:
            model_attempt_identity = ModelAttemptIdentity.model_validate(
                {
                    "model_step_id": copied.get("model_step_id"),
                    "model_attempt_id": copied.get("model_attempt_id"),
                }
            )
        except (TypeError, ValueError):
            pass
        else:
            fallback_fields.update(model_attempt_identity.payload())
    source = copied
    if force_projection:
        projection_fields = (
            "usage",
            "usage_metrics",
            "billing_identity",
            "usage_normalization_failed",
            "usage_unavailable_reason",
            "usage_metrics_rejected",
            "rejected_usage_evidence",
        )
        source = {key: copied[key] for key in projection_fields if key in copied}
        # Forced projection intentionally drops provider-owned auxiliary
        # metadata, but runtime identity must remain attached even when the
        # retained accounting subset is already portable. In particular, the
        # attempt id makes this terminal rewrite replace the ledger entry
        # recorded by the completion observer instead of appending a duplicate.
        for key, value in fallback_fields.items():
            source.setdefault(key, value)
    return durable_model_completed_payload(
        source,
        fallback_fields=fallback_fields,
        unavailable_reason="invalid compaction usage telemetry",
    )


def _rejected_compaction_tool_call_payload(
    *,
    error: _CompactionToolCallError,
    provider_name: str,
    model: str,
    compactor: str,
    usage_dialect: UsageDialect,
) -> dict[str, Any]:
    provider_name = require_durable_clean_nonblank(
        provider_name,
        "provider.billing_provider_name",
    )
    completed_metadata = {} if error.completed_metadata is None else error.completed_metadata
    payload = _compaction_model_completed_payload(
        completed_payload=completed_metadata,
        provider_name=provider_name,
        fallback_model=model,
        compactor=compactor,
        usage_dialect=usage_dialect,
    )
    payload = _durable_compaction_completion_evidence(
        payload,
        provider_name=provider_name,
        fallback_model=model,
        compactor=compactor,
    )
    payload["compaction_outcome"] = "rejected_tool_call"
    if error.completed_metadata is None:
        payload["usage_unavailable_reason"] = (
            "compaction tool-call attempt ended without provider completion usage"
        )
    return payload


def _failed_compaction_provider_attempt_payload(
    *,
    error: BaseException,
    error_type: str,
    provider_name: str,
    model: str,
    compactor: str,
    usage_dialect: UsageDialect,
) -> dict[str, Any]:
    """Represent a dispatched attempt whose provider usage is unknowable."""

    provider_name = require_durable_clean_nonblank(
        provider_name,
        "provider.billing_provider_name",
    )
    payload = _compaction_model_completed_payload(
        completed_payload={},
        provider_name=provider_name,
        fallback_model=model,
        compactor=compactor,
        usage_dialect=usage_dialect,
    )
    payload = _durable_compaction_completion_evidence(
        payload,
        provider_name=provider_name,
        fallback_model=model,
        compactor=compactor,
    )
    if isinstance(error, asyncio.CancelledError):
        outcome = "cancelled"
        unavailable_reason = "compaction provider dispatch was cancelled without completion usage"
    elif isinstance(error, _CompactionToolCallError):
        outcome = "rejected_tool_call"
        unavailable_reason = "compaction tool-call attempt ended without provider completion usage"
    elif isinstance(error, ModelProviderError):
        outcome = "provider_error"
        unavailable_reason = "compaction provider dispatch failed without completion usage"
    else:
        outcome = "unfinished_stream"
        unavailable_reason = "compaction provider dispatch ended without completion usage"
    payload.update(
        {
            "compaction_outcome": outcome,
            "error_type": require_durable_text(error_type, "error_type"),
            "usage_unavailable_reason": unavailable_reason,
        }
    )
    if isinstance(error, ModelStreamDeadlineError):
        # Keep the exact runtime-owned deadline claim alongside accounting
        # evidence. The completion event sanitizer intentionally retains only
        # accounting fields; the model-step owner uses these fields to publish
        # separate durable recovery evidence before propagating the deadline.
        payload.update(
            {key: value for key, value in error.error_payload_fields().items() if key != "provider"}
        )
    return payload


def _context_overflow_compaction_payload(
    *,
    error: ModelContextOverflowError,
    provider_name: str,
    model: str,
    compactor: str,
    usage_dialect: UsageDialect,
) -> dict[str, Any]:
    provider_name = require_durable_clean_nonblank(
        provider_name,
        "provider.billing_provider_name",
    )
    payload = _compaction_model_completed_payload(
        completed_payload={},
        provider_name=provider_name,
        fallback_model=model,
        compactor=compactor,
        usage_dialect=usage_dialect,
    )
    payload.update(error.error_payload_fields())
    payload.update(
        {
            "compaction_outcome": "context_overflow",
            "context_overflow": True,
            "error_type": type(error).__name__,
            "usage_unavailable_reason": (
                "exact prompt-cache compaction overflowed without provider completion usage"
            ),
        }
    )
    return payload


class CheckpointCompactionContextPolicy(RuntimeManagedContextPolicy):
    """Checkpoint-backed context policy for long-running sessions.

    It keeps the durable transcript intact, stores a compact summary in the
    session checkpoint, and sends system messages + summary + recent turns to
    the model.
    """

    def __init__(
        self,
        *,
        compactor: ContextCompactor | None = None,
        max_user_turns: int = 10,
        compact_after_messages: int = 40,
        compact_after_estimated_context_tokens: int | None = None,
        max_recent_context_tokens: int | None = None,
        reserved_output_tokens: int = 0,
        summary_prefix: str = _DEFAULT_CHECKPOINT_COMPACTION_SUMMARY_PREFIX,
        max_attachment_results: int = 1,
    ) -> None:
        if compactor is None:
            self.compactor = TranscriptDigestCompactor()
        elif isinstance(compactor, ContextCompactor):
            self.compactor = compactor
        else:
            raise TypeError("compactor must be a ContextCompactor.")
        if type(max_user_turns) is not int:
            raise TypeError("max_user_turns must be an integer.")
        if type(compact_after_messages) is not int:
            raise TypeError("compact_after_messages must be an integer.")
        if max_user_turns < 1:
            raise ValueError("max_user_turns must be greater than zero.")
        if compact_after_messages < 1:
            raise ValueError("compact_after_messages must be greater than zero.")
        compact_after_estimated_context_tokens = _validate_optional_positive_int(
            compact_after_estimated_context_tokens,
            "compact_after_estimated_context_tokens",
        )
        max_recent_context_tokens = _validate_optional_positive_int(
            max_recent_context_tokens,
            "max_recent_context_tokens",
        )
        reserved_output_tokens = _validate_nonnegative_int(
            reserved_output_tokens,
            "reserved_output_tokens",
        )
        if (compact_after_estimated_context_tokens is None) != (max_recent_context_tokens is None):
            raise ValueError(
                "compact_after_estimated_context_tokens and max_recent_context_tokens "
                "must be configured together."
            )
        if (
            compact_after_estimated_context_tokens is not None
            and max_recent_context_tokens is not None
            and max_recent_context_tokens + reserved_output_tokens
            >= compact_after_estimated_context_tokens
        ):
            raise ValueError(
                "max_recent_context_tokens plus reserved_output_tokens must be less "
                "than compact_after_estimated_context_tokens."
            )
        self.max_user_turns = max_user_turns
        self.compact_after_messages = compact_after_messages
        self.compact_after_estimated_context_tokens = compact_after_estimated_context_tokens
        self.max_recent_context_tokens = max_recent_context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.summary_prefix = require_nonblank(summary_prefix, "summary_prefix")
        self.max_attachment_results = _validate_max_attachment_results(max_attachment_results)

    async def build_with_checkpoint(
        self,
        request: ContextRequest,
        *,
        checkpoint: dict[str, Any] | None,
    ) -> ContextBuildResult:
        checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        previous = _compaction_checkpoint(checkpoint)
        previous_summary = previous.get("summary") if previous is not None else None
        previous_progress = (
            previous.get(_COMPACTION_PROGRESS_STATE_KEY) if previous is not None else None
        )
        if type(previous_progress) is not dict:
            previous_progress = {}
        previous_cursor = (
            previous.get("compacted_transcript_cursor") if previous is not None else None
        )
        if type(previous_summary) is not str:
            previous_summary = None
            previous_cursor = None

        system_prefix, _ = _split_system_prefix(request.messages, True)
        if self.compact_after_estimated_context_tokens is not None:
            system_prefix = _preserve_original_user_task(request.messages, system_prefix)
        first_compactable_cursor = len(system_prefix)
        size_selection_triggered = False
        size_selection_target_satisfied = True
        if (
            previous_summary is None
            or type(previous_cursor) is not int
            or previous_cursor < first_compactable_cursor
            or previous_cursor > len(request.messages)
            or not _is_compaction_boundary(request.messages, previous_cursor)
        ):
            previous_cursor = first_compactable_cursor
            previous_summary = None
            previous_progress = {}

        if self.compact_after_estimated_context_tokens is None:
            (
                system_prefix,
                compactable_messages,
                recent_messages,
                compactable_cursor,
            ) = _split_recent_turns(
                request.messages,
                max_user_turns=self.max_user_turns,
            )
            if previous_cursor > compactable_cursor:
                previous_cursor = first_compactable_cursor
                previous_summary = None
                previous_progress = {}
        else:
            assert self.max_recent_context_tokens is not None
            (
                compactable_messages,
                recent_messages,
                compactable_cursor,
                size_selection_triggered,
                size_selection_target_satisfied,
            ) = _split_recent_context_by_size(
                request,
                system_prefix=system_prefix,
                previous_summary=previous_summary,
                previous_cursor=previous_cursor,
                compact_after_estimated_context_tokens=(
                    self.compact_after_estimated_context_tokens
                ),
                max_recent_context_tokens=self.max_recent_context_tokens,
                reserved_output_tokens=self.reserved_output_tokens,
                summary_prefix=self.summary_prefix,
                max_attachment_results=self.max_attachment_results,
            )

        newly_compactable = request.messages[previous_cursor:compactable_cursor]
        should_compact = (
            request.force_compaction
            or size_selection_triggered
            or (
                self.compact_after_estimated_context_tokens is None
                and len(compactable_messages) >= self.compact_after_messages
            )
        ) and bool(newly_compactable)
        current_progress_key = self.compactor._progress_key_for_context_request(
            request,
            previous_summary=previous_summary,
        )
        if (
            should_compact
            and not request.force_compaction
            and current_progress_key is not None
            and previous_progress.get(_COMPACTION_PROGRESS_EXHAUSTED_KEY) is True
            and previous_progress.get(_COMPACTION_PROGRESS_KEY) == current_progress_key
        ):
            should_compact = False

        checkpoint_update = None
        checkpoint_event_payload = None
        compaction_telemetry: list[ContextCompactionTelemetry] = []
        completion_ledger = _CompactionCompletionLedger()
        summary = previous_summary
        represented_cursor = previous_cursor
        attempt_bounded_input: bool | None = True if request.force_bounded_compaction else None
        if should_compact:
            compaction_started = _compaction_telemetry(
                event_type=EventType.CONTEXT_COMPACTION_STARTED,
                compactor=self.compactor,
                compacted_cursor=compactable_cursor,
                previous_cursor=previous_cursor,
                newly_compacted_message_count=len(newly_compactable),
                recent_message_count=len(recent_messages),
                payload={
                    "requested_source_start": previous_cursor,
                    "requested_source_end": compactable_cursor,
                    "represented_source_start": previous_cursor,
                    "represented_source_end": previous_cursor,
                    "represented_message_count": 0,
                    "coverage_mode": "pending",
                    "chunk_count": 0,
                    **(
                        {"bounded_input": attempt_bounded_input}
                        if attempt_bounded_input is not None
                        else {}
                    ),
                },
            )
            compaction_telemetry.append(compaction_started)
            try:
                context_messages = strip_old_file_attachments(
                    request.messages,
                    max_attachment_results=self.max_attachment_results,
                )
                cache_prefix_request = None
                force_bounded_compaction = request.force_bounded_compaction
                prompt_cache_mode = None
                if isinstance(self.compactor, PromptCacheCompactor):
                    prompt_cache_mode = _prompt_cache_compaction_mode(
                        request=request,
                        compactor=self.compactor,
                        previous_summary=previous_summary,
                    )
                    force_bounded_compaction = (
                        prompt_cache_mode == _PromptCacheCompactionMode.BOUNDED
                    )
                    attempt_bounded_input = True if force_bounded_compaction else None
                if (
                    prompt_cache_mode == _PromptCacheCompactionMode.EXACT
                    and request.build_cache_prefix_request is not None
                ):
                    extension_messages = _prompt_cache_extension_messages(
                        request,
                        max_attachment_results=self.max_attachment_results,
                    )
                    if extension_messages is not None:
                        cache_prefix_request = await request.build_cache_prefix_request(
                            extension_messages
                        )
                compaction_request = CompactionRequest(
                    session=request.session,
                    agent=request.agent,
                    messages=newly_compactable,
                    existing_summary=previous_summary,
                    metadata=request.metadata,
                    context_messages=context_messages,
                    cache_prefix_request=cache_prefix_request,
                    force_bounded_compaction=force_bounded_compaction,
                    instructions=request.compaction_instructions,
                )
                declared_bounded_input = self.compactor._bounded_input_for_request(
                    compaction_request
                )
                started_payload = copy_json_value(compaction_started.payload, "payload")
                if declared_bounded_input is None:
                    attempt_bounded_input = None
                    started_payload.pop("bounded_input", None)
                else:
                    attempt_bounded_input = declared_bounded_input
                    started_payload["bounded_input"] = declared_bounded_input
                compaction_started = compaction_started.model_copy(
                    update={"payload": started_payload},
                    deep=True,
                )
                compaction_telemetry[-1] = compaction_started
                completion_ledger_token = _COMPACTION_COMPLETION_LEDGER.set(completion_ledger)
                try:

                    async def execute_compaction() -> CompactionResult:
                        extension_result = await self.compactor.compact(compaction_request)
                        result_fields = _snapshot_compaction_result(extension_result)
                        returned_payloads = result_fields["model_completed_payloads"]
                        completion_ledger.merge_returned_payloads(returned_payloads)
                        return _detach_compaction_result(result_fields)

                    def completed_payloads_snapshot() -> list[dict[str, Any]]:
                        return copy_durable_json_value(
                            completion_ledger.completed_payloads,
                            "model_completed_payloads",
                        )

                    run_compaction = _AUTOMATIC_COMPACTION_RUNNER.get()
                    if run_compaction is None:
                        result = await execute_compaction()
                    else:
                        result = await run_compaction(
                            self.compactor,
                            compaction_request,
                            compaction_started,
                            execute_compaction,
                            completed_payloads_snapshot,
                        )
                    result = _detach_compaction_result(_snapshot_compaction_result(result))
                    completed_payloads = completed_payloads_snapshot()
                    covered_message_count = result.covered_message_count
                    _validate_compaction_result_coverage(
                        messages=request.messages,
                        previous_cursor=previous_cursor,
                        compactable_cursor=compactable_cursor,
                        covered_message_count=covered_message_count,
                    )
                    if covered_message_count == 0 and not result.progress_exhausted:
                        raise ValueError(
                            "Compaction results with zero coverage must report "
                            "progress_exhausted=true."
                        )
                    if result.progress_exhausted and result.progress_key != current_progress_key:
                        raise ValueError(
                            "Compactor progress exhaustion key does not match its "
                            "current configuration."
                        )
                    if (
                        covered_message_count == 0
                        and previous_summary is not None
                        and result.summary != previous_summary
                    ):
                        raise ValueError(
                            "A zero-coverage compaction must preserve the existing summary "
                            "unchanged."
                        )
                    expected_existing_summary_sha256 = (
                        _compaction_summary_sha256(previous_summary)
                        if previous_summary is not None and covered_message_count > 0
                        else None
                    )
                    if (
                        result.represented_existing_summary_sha256
                        != expected_existing_summary_sha256
                    ):
                        if expected_existing_summary_sha256 is None:
                            raise ValueError(
                                "Compaction result cannot bind "
                                "represented_existing_summary_sha256 without positive "
                                "coverage of an existing summary."
                            )
                        raise ValueError(
                            "Compaction result must bind "
                            "represented_existing_summary_sha256 to the exact existing "
                            "summary."
                        )
                    summary = result.summary
                    represented_cursor = previous_cursor + covered_message_count
                finally:
                    _COMPACTION_COMPLETION_LEDGER.reset(completion_ledger_token)
            except BaseException as exc:
                failure_disposition = automatic_compaction_failure_disposition_payload(exc)
                failure_telemetry = [
                    ContextCompactionTelemetry(
                        event_type=EventType.MODEL_COMPLETED,
                        payload=copy_json_value(payload, "model_completed_payload"),
                    )
                    for payload in completion_ledger.completed_payloads
                ]
                failure_telemetry.append(
                    _compaction_telemetry(
                        event_type=EventType.CONTEXT_COMPACTION_FAILED,
                        compactor=self.compactor,
                        compacted_cursor=previous_cursor,
                        previous_cursor=previous_cursor,
                        newly_compacted_message_count=0,
                        recent_message_count=len(recent_messages),
                        payload={
                            "error_type": type(exc).__name__,
                            "requested_source_start": previous_cursor,
                            "requested_source_end": compactable_cursor,
                            "represented_source_start": previous_cursor,
                            "represented_source_end": previous_cursor,
                            "represented_message_count": 0,
                            "coverage_mode": "failed",
                            "chunk_count": len(completion_ledger.completed_payloads),
                            "chunk_mode": "failed",
                            **(
                                {"bounded_input": attempt_bounded_input}
                                if attempt_bounded_input is not None
                                else {}
                            ),
                            **(failure_disposition if failure_disposition is not None else {}),
                            "compaction_failed": True,
                        },
                    )
                )
                compaction_telemetry.extend(failure_telemetry)
                if not isinstance(exc, Exception):
                    _attach_context_build_termination_diagnostics(
                        exc,
                        compaction_telemetry=compaction_telemetry,
                    )
                    raise
                raise ContextBuildError(
                    str(exc),
                    compaction_telemetry=compaction_telemetry,
                    cause=exc,
                ) from exc
            # Surface the compactor's provider spend as model.completed telemetry
            # so the runtime logs it into usage/cost/budget/limit accounting.
            compaction_telemetry.extend(
                ContextCompactionTelemetry(
                    event_type=EventType.MODEL_COMPLETED,
                    payload=copy_json_value(payload, "model_completed_payload"),
                )
                for payload in completed_payloads
            )
            checkpoint_update = copy_json_value(checkpoint, "checkpoint")
            compaction_checkpoint = {
                "version": _COMPACTION_CHECKPOINT_VERSION,
                "summary": summary,
                "compacted_transcript_cursor": represented_cursor,
                "metadata": copy_json_value(result.metadata, "metadata"),
            }
            if result.progress_exhausted:
                compaction_checkpoint[_COMPACTION_PROGRESS_STATE_KEY] = {
                    _COMPACTION_PROGRESS_EXHAUSTED_KEY: True,
                    _COMPACTION_PROGRESS_KEY: result.progress_key,
                }
            checkpoint_update[_COMPACTION_CHECKPOINT_KEY] = compaction_checkpoint
            checkpoint_event_payload = {
                "checkpoint": _COMPACTION_CHECKPOINT_KEY,
                "compacted_transcript_cursor": represented_cursor,
                "previous_compacted_transcript_cursor": previous_cursor,
                "newly_compacted_message_count": covered_message_count,
                "recent_message_count": len(recent_messages),
            }
            compaction_telemetry.append(
                _compaction_telemetry(
                    event_type=EventType.CONTEXT_COMPACTION_COMPLETED,
                    compactor=self.compactor,
                    compacted_cursor=represented_cursor,
                    previous_cursor=previous_cursor,
                    newly_compacted_message_count=covered_message_count,
                    recent_message_count=len(recent_messages),
                    payload={
                        "summary_chars": len(summary),
                        "requested_source_start": previous_cursor,
                        "requested_source_end": compactable_cursor,
                        "represented_source_start": previous_cursor,
                        "represented_source_end": represented_cursor,
                        "represented_message_count": covered_message_count,
                        "coverage_mode": (
                            "no_progress"
                            if result.progress_exhausted
                            else (
                                "partial_prefix"
                                if covered_message_count < len(newly_compactable)
                                else "full"
                            )
                        ),
                        "chunk_count": result.source_chunk_count,
                        "chunk_mode": result.source_chunk_mode,
                        "bounded_input": result.bounded_input,
                        "compaction_failed": False,
                    },
                )
            )

        if summary is None:
            messages = [copy_message(message) for message in request.messages]
        else:
            messages = [copy_message(message) for message in system_prefix]
            messages.append(Message.text(MessageRole.USER, f"{self.summary_prefix}\n{summary}"))
            messages.extend(
                copy_message(message)
                for message in request.messages[represented_cursor:compactable_cursor]
            )
            messages.extend(copy_message(message) for message in recent_messages)
        messages = strip_old_file_attachments(
            messages,
            max_attachment_results=self.max_attachment_results,
        )
        if size_selection_triggered:
            assert self.compact_after_estimated_context_tokens is not None
            assert self.max_recent_context_tokens is not None
            effective_pressure = _estimate_model_facing_context_pressure(
                request=request,
                messages=messages,
                reserved_output_tokens=self.reserved_output_tokens,
            )
            projection_exceeds_target = (
                size_selection_target_satisfied
                and effective_pressure.estimated_context_input_tokens
                > self.max_recent_context_tokens
            )
            projection_still_triggered = (
                effective_pressure.estimated_context_window_tokens
                >= self.compact_after_estimated_context_tokens
            )
            if projection_exceeds_target or projection_still_triggered:
                cause = ValueError(
                    "Checkpoint compaction did not produce a model-facing context "
                    "within the configured size bounds."
                )
                raise ContextBuildError(
                    str(cause),
                    compaction_telemetry=compaction_telemetry,
                    checkpoint=checkpoint_update,
                    checkpoint_event_payload=checkpoint_event_payload,
                    cause=cause,
                )
        return ContextBuildResult(
            messages=messages,
            checkpoint=checkpoint_update,
            checkpoint_event_payload=checkpoint_event_payload,
            compaction_telemetry=compaction_telemetry,
        )


def _prompt_cache_extension_messages(
    request: ContextRequest,
    *,
    max_attachment_results: int,
) -> list[Message] | None:
    """Rebuild the last provider projection, then append the transcript delta.

    File projection depends on which user turn was current. Re-projecting the
    entire present transcript would omit the formerly-current attachment and
    change the cached prefix; using the entire durable transcript would instead
    resurrect older attachments that the last provider request had omitted.
    """

    messages = [copy_message(message) for message in request.messages]
    previous_input_cursor = _prompt_cache_previous_input_cursor(request)
    if previous_input_cursor is None:
        return None
    previous_projection = strip_old_file_attachments(
        messages[:previous_input_cursor],
        max_attachment_results=max_attachment_results,
    )
    return previous_projection + [
        copy_message(message) for message in messages[previous_input_cursor:]
    ]


def _prompt_cache_previous_input_cursor(request: ContextRequest) -> int | None:
    """Return the reconstructable prior provider-input boundary, if available."""

    completed_cursor = request.context_usage.last_transcript_cursor
    if completed_cursor is None or completed_cursor < 1 or completed_cursor > len(request.messages):
        return None
    if request.messages[completed_cursor - 1].role == MessageRole.ASSISTANT:
        return completed_cursor - 1
    return completed_cursor


def copy_context_messages(messages: list[Message]) -> list[Message]:
    if type(messages) is not list:
        raise TypeError("ContextPolicy.build() must return a list of Message instances.")
    if not messages:
        raise ValueError("ContextPolicy.build() must return at least one message.")
    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)
    return copied_messages


def trim_context_messages(
    messages: list[Message],
    *,
    max_messages: int,
    preserve_system: bool = True,
) -> list[Message]:
    """Return a recent valid suffix without cutting through a tool round."""

    if type(max_messages) is not int:
        raise TypeError("max_messages must be an integer.")
    if type(preserve_system) is not bool:
        raise TypeError("preserve_system must be a bool.")
    if max_messages < 1:
        raise ValueError("max_messages must be greater than zero.")
    copied_messages = [copy_message(message) for message in messages]
    system_prefix, body = _split_system_prefix(copied_messages, preserve_system)
    candidate = system_prefix + body
    if len(candidate) <= max_messages:
        validate_context_messages(candidate)
        return [copy_message(message) for message in candidate]

    body_limit = max(1, max_messages - len(system_prefix))
    start = max(0, len(body) - body_limit)
    for index in range(start, len(body)):
        candidate = system_prefix + body[index:]
        try:
            validate_context_messages(candidate)
        except ValueError:
            continue
        return [copy_message(message) for message in candidate]
    raise ValueError("Cannot trim context without cutting through a tool round.")


def trim_context_turns(
    messages: list[Message],
    *,
    max_user_turns: int,
    preserve_system: bool = True,
) -> list[Message]:
    """Return the latest user turns with complete assistant/tool follow-up."""

    if type(max_user_turns) is not int:
        raise TypeError("max_user_turns must be an integer.")
    if type(preserve_system) is not bool:
        raise TypeError("preserve_system must be a bool.")
    if max_user_turns < 1:
        raise ValueError("max_user_turns must be greater than zero.")

    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)

    system_prefix, body = _split_system_prefix(copied_messages, preserve_system)
    turn_starts = [index for index, message in enumerate(body) if message.role == MessageRole.USER]
    if not turn_starts:
        candidate = system_prefix + body
        validate_context_messages(candidate)
        return [copy_message(message) for message in candidate]
    if len(turn_starts) <= max_user_turns:
        candidate = system_prefix + body
        validate_context_messages(candidate)
        return [copy_message(message) for message in candidate]

    start = turn_starts[-max_user_turns]
    candidate = system_prefix + body[start:]
    validate_context_messages(candidate)
    return [copy_message(message) for message in candidate]


def strip_old_file_attachments(
    messages: list[Message],
    *,
    max_attachment_results: int = 1,
) -> list[Message]:
    """Remove old native file attachment refs from provider-facing context.

    Durable transcript messages keep their original artifacts. This helper only
    projects model-facing context so providers do not receive the same native
    file bytes on every subsequent model request.
    """

    if type(max_attachment_results) is not int:
        raise TypeError("max_attachment_results must be an integer.")
    if max_attachment_results < 0:
        raise ValueError("max_attachment_results must be non-negative.")

    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)

    # Tool-result attachments: keep only the latest `max_attachment_results` positions.
    attachment_positions: list[tuple[int, int]] = []
    for message_index, message in enumerate(copied_messages):
        if message.role != MessageRole.TOOL:
            continue
        for part_index, part in enumerate(message.content):
            if type(part) is not ToolResultPart:
                continue
            if _file_attachments_in_part(part):
                attachment_positions.append((message_index, part_index))

    tool_stripping_needed = len(attachment_positions) > max_attachment_results

    # Prompt file parts: keep files provider-resolvable only on the current attach turn. A
    # file-bearing user message is projected to a text note once its turn has been answered AND a
    # newer user turn has begun — i.e. an assistant/tool response sits between it and the latest user
    # message. This keeps every file from the same run live (multiple file messages with no response
    # between them) and keeps a file live through its own run's tool loop (no newer user message yet),
    # while stopping the bytes from being re-resolved and re-sent on every later turn. Independent of
    # the tool-result budget above.
    user_file_message_indices = [
        message_index
        for message_index, message in enumerate(copied_messages)
        if message.role == MessageRole.USER
        and any(type(part) is FilePart for part in message.content)
    ]
    last_user_index = max(
        (i for i, message in enumerate(copied_messages) if message.role == MessageRole.USER),
        default=-1,
    )
    strip_user_file_indices = {
        message_index
        for message_index in user_file_message_indices
        if any(
            copied_messages[j].role in (MessageRole.ASSISTANT, MessageRole.TOOL)
            for j in range(message_index + 1, last_user_index)
        )
    }

    if not tool_stripping_needed and not strip_user_file_indices:
        return [copy_message(message) for message in copied_messages]

    if not tool_stripping_needed:
        keep_positions = set(attachment_positions)
    elif max_attachment_results == 0:
        keep_positions = set()
    else:
        keep_positions = set(attachment_positions[-max_attachment_results:])

    projected_messages: list[Message] = []
    for message_index, message in enumerate(copied_messages):
        if message.role == MessageRole.TOOL:
            projected_messages.append(
                _strip_old_tool_result_attachments(message, keep_positions, message_index)
            )
        elif message_index in strip_user_file_indices:
            projected_messages.append(_strip_file_parts_from_user_message(message))
        else:
            projected_messages.append(copy_message(message))

    validate_context_messages(projected_messages)
    return [copy_message(message) for message in projected_messages]


def validate_context_messages(messages: list[Message]) -> None:
    if type(messages) is not list:
        raise TypeError("Context messages must be a list of Message instances.")
    if not messages:
        raise ValueError("Context messages cannot be empty.")

    pending_tool_call_ids: set[str] | None = None
    for message in messages:
        if type(message) is not Message:
            raise TypeError("Context messages must be Message instances.")

        if pending_tool_call_ids is not None:
            if message.role != MessageRole.TOOL:
                raise ValueError(
                    "Context messages contain assistant tool calls that are not "
                    "followed by matching tool results."
                )
            result_parts = [part for part in message.content if type(part) is ToolResultPart]
            result_ids = [part.tool_call_id for part in result_parts]
            if len(result_ids) != len(set(result_ids)):
                raise ValueError("Context messages contain duplicate tool result ids.")
            if set(result_ids) != pending_tool_call_ids:
                raise ValueError(
                    "Context messages contain tool results that do not match the "
                    "preceding assistant tool calls."
                )
            pending_tool_call_ids = None
            continue

        if message.role == MessageRole.TOOL:
            raise ValueError(
                "Context messages contain tool results without preceding assistant tool calls."
            )

        if message.role == MessageRole.ASSISTANT:
            tool_call_ids = [
                part.tool_call_id for part in message.content if type(part) is ToolCallPart
            ]
            if len(tool_call_ids) != len(set(tool_call_ids)):
                raise ValueError("Context messages contain duplicate tool call ids.")
            if tool_call_ids:
                pending_tool_call_ids = set(tool_call_ids)

    if pending_tool_call_ids is not None:
        raise ValueError(
            "Context messages end with assistant tool calls that have no matching tool results."
        )


def _strip_file_attachments_from_tool_result(part: ToolResultPart) -> ToolResultPart:
    kept_artifacts: list[dict[str, Any]] = []
    stripped_attachments: list[FileAttachment] = []
    for payload in part.artifacts:
        attachment = file_attachment_from_payload(payload)
        if attachment is None:
            kept_artifacts.append(copy_json_value(payload, "artifact"))
        else:
            stripped_attachments.append(attachment)

    if not stripped_attachments:
        return ToolResultPart(
            tool_call_id=part.tool_call_id,
            tool_name=part.tool_name,
            content=part.content,
            structured=copy_json_value(part.structured, "structured"),
            artifacts=kept_artifacts,
            is_error=part.is_error,
            tool_round_id=part.tool_round_id,
            model_step_id=part.model_step_id,
            model_attempt_id=part.model_attempt_id,
        )

    content = _content_with_stripped_file_attachment_note(part.content, stripped_attachments)
    structured = copy_json_value(part.structured, "structured")
    if structured is None:
        structured = {}
    structured["cayu_file_attachments_stripped"] = [
        {
            "artifact_id": attachment.artifact_id,
            "filename": attachment.filename,
            "content_type": attachment.content_type,
            "size_bytes": attachment.size_bytes,
            "kind": attachment.kind.value,
        }
        for attachment in stripped_attachments
    ]
    return ToolResultPart(
        tool_call_id=part.tool_call_id,
        tool_name=part.tool_name,
        content=content,
        structured=structured,
        artifacts=kept_artifacts,
        is_error=part.is_error,
        tool_round_id=part.tool_round_id,
        model_step_id=part.model_step_id,
        model_attempt_id=part.model_attempt_id,
    )


def _strip_old_tool_result_attachments(
    message: Message,
    keep_positions: set[tuple[int, int]],
    message_index: int,
) -> Message:
    projected_parts: list[
        TextPart
        | ToolCallPart
        | ToolResultPart
        | ProviderStatePart
        | ThinkingPart
        | FilePart
        | HostedToolCallPart
        | CitationPart
    ] = []
    for part_index, part in enumerate(message.content):
        if type(part) is not ToolResultPart or (message_index, part_index) in keep_positions:
            projected_parts.append(copy_message_part(part))
            continue
        projected_parts.append(_strip_file_attachments_from_tool_result(part))
    return Message(role=message.role, content=tuple(projected_parts))


def _format_stripped_attachment_lines(attachments: list[FileAttachment]) -> str:
    return "\n".join(
        f"- {attachment.filename} ({attachment.content_type}, "
        f"{attachment.size_bytes} bytes, artifact_id={attachment.artifact_id})"
        for attachment in attachments
    )


def _content_with_stripped_file_attachment_note(
    content: str,
    attachments: list[FileAttachment],
) -> str:
    note = "File attachments from this older tool result were omitted from this provider request:\n"
    note += _format_stripped_attachment_lines(attachments)
    if content:
        return f"{content}\n\n{note}"
    return note


def _file_attachments_in_part(part: ToolResultPart) -> tuple[FileAttachment, ...]:
    attachments: list[FileAttachment] = []
    for payload in part.artifacts:
        attachment = file_attachment_from_payload(payload)
        if attachment is not None:
            attachments.append(attachment)
    return tuple(attachments)


def _strip_file_parts_from_user_message(message: Message) -> Message:
    kept_parts: list[
        TextPart
        | ToolCallPart
        | ToolResultPart
        | ProviderStatePart
        | ThinkingPart
        | FilePart
        | HostedToolCallPart
        | CitationPart
    ] = []
    stripped_attachments: list[FileAttachment] = []
    for part in message.content:
        if type(part) is FilePart:
            attachment = file_attachment_from_payload(part.attachment)
            if attachment is not None:
                stripped_attachments.append(attachment)
                continue
        kept_parts.append(copy_message_part(part))

    if stripped_attachments:
        kept_parts.append(TextPart(text=_prompt_file_stripped_note(stripped_attachments)))
    return Message(role=message.role, content=tuple(kept_parts))


def _prompt_file_stripped_note(attachments: list[FileAttachment]) -> str:
    note = "Files attached to this earlier prompt were omitted from this provider request:\n"
    return note + _format_stripped_attachment_lines(attachments)


def noteify_unresolvable_prompt_files(
    messages: list[Message],
    artifact_ids: set[str],
) -> list[Message]:
    """Project user-prompt `FilePart`s whose artifacts can't be resolved down to a text note.

    Model-facing projection only (the durable transcript keeps the original `FilePart`). Lets the
    runtime proceed with a note instead of failing a request forever when a live prompt file is
    unresolvable (wrong session at attach time, or a deleted artifact).
    """
    if not artifact_ids:
        return messages
    projected: list[Message] = []
    for message in messages:
        if message.role != MessageRole.USER:
            projected.append(message)
            continue
        kept_parts: list[
            TextPart
            | ToolCallPart
            | ToolResultPart
            | ProviderStatePart
            | ThinkingPart
            | FilePart
            | HostedToolCallPart
            | CitationPart
        ] = []
        removed_attachments: list[FileAttachment] = []
        for part in message.content:
            if type(part) is FilePart:
                attachment = file_attachment_from_payload(part.attachment)
                if attachment is not None and attachment.artifact_id in artifact_ids:
                    removed_attachments.append(attachment)
                    continue
            kept_parts.append(copy_message_part(part))
        if not removed_attachments:
            projected.append(message)
            continue
        kept_parts.append(TextPart(text=_unresolvable_prompt_file_note(removed_attachments)))
        projected.append(Message(role=message.role, content=tuple(kept_parts)))
    return projected


def _unresolvable_prompt_file_note(attachments: list[FileAttachment]) -> str:
    note = (
        "Files attached to this prompt could not be resolved (check the session_id used at attach "
        "time, or whether the artifact still exists) and were omitted from this provider request:\n"
    )
    return note + _format_stripped_attachment_lines(attachments)


def _validate_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _validate_nonnegative_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return value


def _validate_ratio(value: float, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a number.")
    parsed = float(value)
    if parsed <= 0 or parsed > 1:
        raise ValueError(f"{field_name} must be greater than 0 and at most 1.")
    return parsed


def _user_message_digest(
    message: Message,
    *,
    ignored_manifests: set[str] | None = None,
) -> str:
    if message.role != MessageRole.USER:
        raise ValueError("Knowledge anchors must be user messages.")
    content = [
        copy_message_part(part).model_dump(mode="json")
        for part in message.content
        if not (
            type(part) is TextPart
            and ignored_manifests is not None
            and part.text in ignored_manifests
        )
    ]
    canonical = json.dumps(
        {"content": content, "role": MessageRole.USER.value},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _runtime_authored_user_message_checkpoint_transform(
    *,
    anchor_index: int,
    message: Message,
) -> Callable[[Session, dict[str, Any] | None], dict[str, Any]]:
    anchor_index = _validate_nonnegative_int(anchor_index, "anchor_index")
    anchor_digest = _user_message_digest(message)

    def transform(
        _session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        updated: dict[str, Any] = (
            {CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION}
            if checkpoint is None
            else copy_json_value(checkpoint, "checkpoint")
        )
        updated[RUNTIME_AUTHORED_USER_MESSAGE_CHECKPOINT_KEY] = {
            "anchor_transcript_index": anchor_index,
            "user_message_sha256": anchor_digest,
            "version": RUNTIME_AUTHORED_USER_MESSAGE_CHECKPOINT_VERSION,
        }
        return updated

    return transform


async def _build_policy_context(
    policy: ContextPolicy,
    request: ContextRequest,
    *,
    checkpoint: dict[str, Any] | None,
) -> ContextBuildResult:
    if isinstance(policy, RuntimeManagedContextPolicy):
        return await policy.build_with_checkpoint(request, checkpoint=checkpoint)
    messages = await policy.build(request)
    return ContextBuildResult(messages=messages)


def _validate_optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer or None.")
    if value < 1:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _validate_max_attachment_results(value: int) -> int:
    if type(value) is not int:
        raise TypeError("max_attachment_results must be an integer.")
    if value < 0:
        raise ValueError("max_attachment_results must be non-negative.")
    return value


def _split_system_prefix(
    messages: list[Message],
    preserve_system: bool,
) -> tuple[list[Message], list[Message]]:
    system_prefix: list[Message] = []
    for message in messages:
        if message.role != MessageRole.SYSTEM:
            break
        system_prefix.append(message)
    body = messages[len(system_prefix) :]
    if not preserve_system:
        return [], body
    return system_prefix, body


def _split_recent_turns(
    messages: list[Message],
    *,
    max_user_turns: int,
) -> tuple[list[Message], list[Message], list[Message], int]:
    copied_messages = [copy_message(message) for message in messages]
    validate_context_messages(copied_messages)
    system_prefix, body = _split_system_prefix(copied_messages, True)
    turn_starts = [index for index, message in enumerate(body) if message.role == MessageRole.USER]
    if not turn_starts or len(turn_starts) <= max_user_turns:
        return system_prefix, [], body, len(system_prefix)
    recent_start = turn_starts[-max_user_turns]
    compactable_cursor = len(system_prefix) + recent_start
    return system_prefix, body[:recent_start], body[recent_start:], compactable_cursor


def _split_recent_context_by_size(
    request: ContextRequest,
    *,
    system_prefix: list[Message],
    previous_summary: str | None,
    previous_cursor: int,
    compact_after_estimated_context_tokens: int,
    max_recent_context_tokens: int,
    reserved_output_tokens: int,
    summary_prefix: str,
    max_attachment_results: int,
) -> tuple[list[Message], list[Message], int, bool, bool]:
    """Choose a size-bounded recent suffix without splitting a tool round."""

    messages = [copy_message(message) for message in request.messages]
    validate_context_messages(messages)
    summary_message = (
        []
        if previous_summary is None
        else [
            Message.text(
                MessageRole.USER,
                f"{summary_prefix}\n{previous_summary}",
            )
        ]
    )
    current_projection = strip_old_file_attachments(
        [
            *(copy_message(message) for message in system_prefix),
            *summary_message,
            *(copy_message(message) for message in messages[previous_cursor:]),
        ],
        max_attachment_results=max_attachment_results,
    )
    pressure = _estimate_model_facing_context_pressure(
        request=request,
        messages=current_projection,
        reserved_output_tokens=reserved_output_tokens,
    )
    if (
        not request.force_compaction
        and not request.force_bounded_compaction
        and pressure.estimated_context_window_tokens < compact_after_estimated_context_tokens
    ):
        return [], messages[previous_cursor:], previous_cursor, False, True

    projected_summary = previous_summary or "Compacted prior context."
    best_candidate: tuple[list[Message], list[Message], int] | None = None
    best_candidate_tokens: int | None = None
    # Keep the newest protocol-atomic unit verbatim whenever that can satisfy
    # the configured bounds. If no valid projection retaining that unit can
    # fit, the terminal boundary is the only safe smaller projection: compact
    # the whole unit rather than split an assistant tool call from its results.
    current_projection_exceeds_bounds = (
        pressure.estimated_context_input_tokens > max_recent_context_tokens
        or pressure.estimated_context_window_tokens >= compact_after_estimated_context_tokens
    )
    include_terminal_boundary = messages[-1].role != MessageRole.USER and (
        request.force_bounded_compaction or current_projection_exceeds_bounds
    )
    candidate_stop = len(messages) + (1 if include_terminal_boundary else 0)
    for compactable_cursor in range(previous_cursor + 1, candidate_stop):
        if not _is_compaction_boundary(messages, compactable_cursor):
            continue
        candidate_projection = strip_old_file_attachments(
            [
                *(copy_message(message) for message in system_prefix),
                Message.text(
                    MessageRole.USER,
                    f"{summary_prefix}\n{projected_summary}",
                ),
                *(copy_message(message) for message in messages[compactable_cursor:]),
            ],
            max_attachment_results=max_attachment_results,
        )
        try:
            validate_context_messages(candidate_projection)
        except ValueError:
            continue
        candidate_pressure = _estimate_model_facing_context_pressure(
            request=request,
            messages=candidate_projection,
            reserved_output_tokens=0,
        )
        candidate_tokens = candidate_pressure.estimated_context_input_tokens
        candidate = (
            messages[previous_cursor:compactable_cursor],
            messages[compactable_cursor:],
            compactable_cursor,
        )
        if candidate_tokens <= max_recent_context_tokens:
            return (
                *candidate,
                True,
                True,
            )
        if best_candidate_tokens is None or candidate_tokens <= best_candidate_tokens:
            best_candidate = candidate
            best_candidate_tokens = candidate_tokens

    if best_candidate is not None:
        return *best_candidate, True, False
    return [], messages[previous_cursor:], previous_cursor, True, False


def _preserve_original_user_task(
    messages: list[Message],
    system_prefix: list[Message],
) -> list[Message]:
    """Keep the first user task verbatim outside every size-compacted prefix."""

    cursor = len(system_prefix)
    if cursor < len(messages) and messages[cursor].role == MessageRole.USER:
        return [
            *(copy_message(message) for message in system_prefix),
            copy_message(messages[cursor]),
        ]
    return [copy_message(message) for message in system_prefix]


def _compaction_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    value = checkpoint.get(_COMPACTION_CHECKPOINT_KEY)
    if type(value) is not dict:
        return None
    if value.get("version") != _COMPACTION_CHECKPOINT_VERSION:
        return None
    return copy_json_value(value, _COMPACTION_CHECKPOINT_KEY)


def project_compaction_invocation_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any] | None:
    """Copy a checkpoint with descriptive compaction state safe for extensions."""

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    projected = project_runtime_managed_context_checkpoint(checkpoint)
    if projected is None:
        return None
    compaction = projected.get(_COMPACTION_CHECKPOINT_KEY)
    if type(compaction) is not dict:
        return projected
    summary = compaction.get("summary")
    if type(summary) is str:
        compaction["summary"] = redactor.redact_text(summary)
    return projected


def project_runtime_managed_context_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Project checkpoint state visible to checkpoint-aware context policies.

    Invocation lifecycle receipts and terminal decisions are private
    store/runtime authority. A generic context policy may carry ordinary
    checkpoint state forward, but it must neither observe those records nor
    acquire their provenance by returning identically named objects.
    """

    if checkpoint is None:
        return None
    projected = copy_json_value(checkpoint, "checkpoint")
    projected.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY, None)
    projected.pop(INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY, None)
    projected.pop(SETTLED_INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY, None)
    return projected


def _is_compaction_boundary(messages: list[Message], cursor: int) -> bool:
    """Return whether ``cursor`` leaves no assistant/tool round split."""

    if type(cursor) is not int or cursor < 0 or cursor > len(messages):
        return False
    if cursor == 0:
        return True
    try:
        validate_context_messages(messages[:cursor])
    except (TypeError, ValueError):
        return False
    return True


def _validate_compaction_result_coverage(
    *,
    messages: list[Message],
    previous_cursor: int,
    compactable_cursor: int,
    covered_message_count: int,
) -> None:
    requested_count = compactable_cursor - previous_cursor
    if covered_message_count > requested_count:
        raise ValueError("Compactor reported coverage beyond its requested source range.")
    covered_cursor = previous_cursor + covered_message_count
    if not _is_compaction_boundary(messages, covered_cursor):
        raise ValueError("Compactor reported coverage that splits an assistant/tool round.")


def _usage_triggered_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    value = checkpoint.get(_USAGE_TRIGGERED_CHECKPOINT_KEY)
    if type(value) is not dict:
        return None
    if value.get("version") != _USAGE_TRIGGERED_CHECKPOINT_VERSION:
        return None
    return copy_json_value(value, _USAGE_TRIGGERED_CHECKPOINT_KEY)


def _compaction_telemetry(
    *,
    event_type: EventType,
    compactor: ContextCompactor,
    compacted_cursor: int,
    previous_cursor: int,
    newly_compacted_message_count: int,
    recent_message_count: int,
    payload: dict[str, Any] | None = None,
) -> ContextCompactionTelemetry:
    event_payload = {
        "checkpoint": _COMPACTION_CHECKPOINT_KEY,
        "compactor": type(compactor).__name__,
        "compacted_transcript_cursor": compacted_cursor,
        "previous_compacted_transcript_cursor": previous_cursor,
        "newly_compacted_message_count": newly_compacted_message_count,
        "recent_message_count": recent_message_count,
    }
    if payload is not None:
        event_payload.update(copy_json_value(payload, "payload"))
    return ContextCompactionTelemetry(event_type=event_type, payload=event_payload)


def _messages_digest(messages: list[Message]) -> str:
    return "\n".join(_message_digest(message) for message in messages)


def _message_digest(message: Message) -> str:
    parts = [_message_part_digest(part) for part in message.content]
    return f"{message.role}: " + " ".join(parts)


def _message_part_digest(
    part: TextPart
    | ToolCallPart
    | ToolResultPart
    | ProviderStatePart
    | ThinkingPart
    | FilePart
    | HostedToolCallPart
    | CitationPart,
) -> str:
    if type(part) is TextPart:
        return part.text
    if type(part) is ToolCallPart:
        return (
            f"[tool_call id={part.tool_call_id} name={part.tool_name} "
            f"arguments={copy_json_value(part.arguments, 'arguments')}]"
        )
    if type(part) is ToolResultPart:
        return (
            f"[tool_result id={part.tool_call_id} name={part.tool_name} "
            f"error={part.is_error} content={part.content} "
            f"structured={copy_json_value(part.structured, 'structured')} "
            f"artifacts={copy_json_value(part.artifacts, 'artifacts')}]"
        )
    if type(part) is ProviderStatePart:
        return f"[provider_state provider={part.provider}]"
    if type(part) is ThinkingPart:
        # Marker only: reasoning text is provider-internal and must not leak into the
        # compaction digest shown to the model.
        return "[thinking]"
    if type(part) is FilePart:
        return f"[file attachment={copy_json_value(part.attachment, 'attachment')}]"
    if type(part) is HostedToolCallPart:
        return f"[hosted_tool_call type={part.hosted_tool} status={part.status}]"
    if type(part) is CitationPart:
        return f"[citation url={part.url}]"
    raise TypeError("Unsupported message part.")


def _provider_completed_metadata(
    payload: dict[str, Any],
    *,
    preserve_usage_metrics: bool = False,
    preserve_usage_failure: bool = False,
) -> dict[str, Any]:
    copied = copy_json_value(payload, "completed")
    if type(copied) is not dict:
        raise ValueError("Provider completed payload must be an object.")
    strip_runtime_owned_execution_identity(copied)
    copied.pop("provider_state", None)
    strip_provider_billing_identity(copied)
    normalization_failed = copied.pop("usage_normalization_failed", None)
    copied.pop("usage_unavailable_reason", None)
    if not preserve_usage_metrics and copied.get("usage") is not None:
        copied.pop("usage_metrics", None)
    copied.pop("usage_metrics_rejected", None)
    copied.pop("rejected_usage_evidence", None)
    if preserve_usage_failure and normalization_failed is True:
        copied["usage_normalization_failed"] = True
    return copied


def _compaction_model_completed_payload(
    *,
    completed_payload: dict[str, Any],
    provider_name: str,
    fallback_model: str,
    compactor: str,
    usage_dialect: str | None = None,
) -> dict[str, Any]:
    """Build an event-ready ``model.completed`` payload for a compaction call.

    Mirrors the runtime's model-step payload shape closely enough for the
    usage/cost/budget aggregators: normalized ``usage_metrics`` when the
    provider reported usage, the resolved model name, and a ``purpose`` marker
    so the spend is attributable to context compaction.
    """

    payload = copy_json_value(completed_payload, "completed")
    resolved_model = payload.get("model")
    if type(resolved_model) is not str or not resolved_model.strip():
        resolved_model = fallback_model
        payload["model"] = fallback_model
    payload["provider_name"] = provider_name
    payload["requested_model"] = fallback_model
    payload["purpose"] = ModelCompletionPurpose.CONTEXT_COMPACTION.value
    payload["compactor"] = compactor
    # When raw usage is present, the runtime below owns its normalized
    # projection and durable failure decision. Preserve the established path
    # for providers that expose only normalized usage metrics.
    has_raw_usage = payload.get("usage") is not None
    if has_raw_usage:
        payload.pop("usage_metrics", None)
    usage_normalization_failed = payload.pop("usage_normalization_failed", None) is True
    payload.pop("usage_unavailable_reason", None)
    payload.pop("usage_metrics_rejected", None)
    payload.pop("rejected_usage_evidence", None)
    raw_billing_identity = payload.get("billing_identity")
    billing_identity = (
        BillingIdentity.model_validate(raw_billing_identity)
        if type(raw_billing_identity) is dict
        else None
    )
    try:
        usage_metrics = usage_metrics_payload(
            normalize_usage_metrics_with_overflow_error(
                provider_name=provider_name,
                model=resolved_model,
                requested_model=fallback_model,
                raw_usage=payload.get("usage"),
                usage_dialect=usage_dialect,
                billing_identity=billing_identity,
            )
        )
    except (TypeError, ValueError):
        # Runtime-derived totals and cache aggregates can exceed the portable
        # int64 range even when every provider counter is independently valid.
        raise _CompactionAccountingUsageError(
            payload=_rejected_compaction_usage_payload(payload)
        ) from None
    if usage_metrics is not None:
        # The event-level identity is the only durable authority. Readers attach
        # it to parsed usage after validating the completion payload.
        usage_metrics.pop("billing_identity", None)
        try:
            payload["usage_metrics"] = copy_durable_json_object(
                usage_metrics,
                "usage_metrics",
            )
        except DurableValueError:
            raise _CompactionAccountingUsageError(
                payload=_rejected_compaction_usage_payload(payload)
            ) from None
    elif has_raw_usage or usage_normalization_failed:
        payload["usage_normalization_failed"] = True
    return payload


def default_compaction_prompt(
    request: CompactionRequest,
) -> CompactionPrompt:
    """Build the default user prompt for model-backed context compaction."""

    prefix, transcript_prefix, transcript_digest = _default_compaction_prompt_parts(request)
    return CompactionPrompt(
        prompt=f"{prefix}\n\n{transcript_prefix}{transcript_digest}",
        covered_message_count=len(request.messages),
    )


def _bounded_default_compaction_prompt(
    request: CompactionRequest,
    *,
    max_chars: int | None,
) -> tuple[str | None, bool, int]:
    full_count = len(request.messages)
    prefix, transcript_prefix, transcript_digest = _default_compaction_prompt_parts(request)
    prompt = f"{prefix}\n\n{transcript_prefix}{transcript_digest}"
    if max_chars is None or len(prompt) <= max_chars:
        return prompt, False, full_count

    atomic_counts = _compaction_atomic_prefix_counts(request.messages)
    lower = 0
    upper = len(atomic_counts) - 1
    best_prompt: str | None = None
    best_count = 0
    while lower <= upper:
        midpoint = (lower + upper) // 2
        count = atomic_counts[midpoint]
        bounded_request = request.model_copy(
            update={"messages": request.messages[:count]},
        )
        bounded_prefix, bounded_transcript_prefix, bounded_digest = (
            _default_compaction_prompt_parts(bounded_request)
        )
        bounded_prompt = f"{bounded_prefix}\n\n{bounded_transcript_prefix}{bounded_digest}"
        if len(bounded_prompt) <= max_chars:
            best_prompt = bounded_prompt
            best_count = count
            lower = midpoint + 1
        else:
            upper = midpoint - 1
    return best_prompt, True, best_count


_HIERARCHY_SOURCE_PROMPT_PREFIX = (
    "Summarize this ordered fragment of one protocol-atomic transcript unit. "
    "Preserve concrete requests, decisions, identifiers, errors, files, tool "
    "calls/results, and pending work. Do not invent facts. Return only the "
    "fragment summary.\n\n"
)
_HIERARCHY_MERGE_PROMPT_PREFIX = (
    "Merge these ordered partial context summaries into one compact summary. "
    "Preserve every concrete request, decision, identifier, error, file, tool "
    "call/result, and pending item. Do not invent facts or duplicate repeated "
    "items. Return only the merged summary.\n\n"
)


def _hierarchy_instructions_suffix(instructions: str | None) -> str:
    if instructions is None:
        return ""
    return f"Additional compaction instructions:\n{instructions}\n\n"


def _hierarchy_source_prompt_prefix(instructions: str | None) -> str:
    return _HIERARCHY_SOURCE_PROMPT_PREFIX + _hierarchy_instructions_suffix(instructions)


def _hierarchy_merge_prompt_prefix(instructions: str | None) -> str:
    return _HIERARCHY_MERGE_PROMPT_PREFIX + _hierarchy_instructions_suffix(instructions)


def _hierarchy_source_prompt(
    fragment: str,
    *,
    index: int,
    prompt_prefix: str,
) -> str:
    return f"{prompt_prefix}Fragment {index}:\n{fragment}"


def _hierarchy_merge_prompt(
    items: list[str],
    *,
    prompt_prefix: str,
) -> str:
    rendered = "\n\n".join(f"Part {index}:\n{item}" for index, item in enumerate(items, start=1))
    return prompt_prefix + rendered


def _split_hierarchy_text(
    text: str,
    *,
    max_chars: int,
    prompt_prefix: str,
) -> list[str]:
    # Reserve enough space for a stable numeric label. Python string slicing
    # operates on Unicode code points, so it cannot cut a scalar's UTF-8 bytes.
    available = max_chars - len(prompt_prefix) - len("Fragment 999999:\n")
    if available < 1:
        raise ValueError("max_input_chars is too small for hierarchical compaction.")
    return [text[index : index + available] for index in range(0, len(text), available)]


def _split_hierarchy_items(
    items: list[str],
    *,
    max_chars: int,
    prompt_prefix: str,
) -> list[str]:
    available = max_chars - len(prompt_prefix) - len("Part 999999:\n")
    if available < 1:
        raise ValueError("max_input_chars is too small for hierarchical assembly.")
    expanded: list[str] = []
    for item in items:
        expanded.extend(item[index : index + available] for index in range(0, len(item), available))
    return expanded


def _pack_hierarchy_items(
    items: list[str],
    *,
    max_chars: int,
    prompt_prefix: str,
) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    for item in items:
        candidate = [*current, item]
        if len(_hierarchy_merge_prompt(candidate, prompt_prefix=prompt_prefix)) <= max_chars:
            current = candidate
            continue
        if not current:
            raise ValueError("Hierarchical compaction item does not fit its request bound.")
        groups.append(current)
        current = [item]
    if current:
        groups.append(current)
    return groups


def _compaction_atomic_prefix_counts(messages: list[Message]) -> list[int]:
    """Return prefix lengths that do not split an assistant/tool round."""

    if not messages:
        return []
    validate_context_messages(messages)
    counts: list[int] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        has_tool_calls = message.role == MessageRole.ASSISTANT and any(
            type(part) is ToolCallPart for part in message.content
        )
        index += 2 if has_tool_calls else 1
        counts.append(index)
    return counts


def _default_compaction_prompt_parts(
    request: CompactionRequest,
) -> tuple[str, str, str]:
    sections = [
        "Summarize the transcript below so a future agent step can continue with the important context.",
        "Preserve concrete user requests, decisions, files or resources mentioned, tool results, errors, and pending work.",
        "Do not invent facts. Keep the summary concise but specific.",
        f"Session: {request.session.id}",
        f"Agent: {request.agent.name}",
    ]
    if request.existing_summary is not None:
        sections.append("Existing summary:\n" + request.existing_summary)
    if request.instructions is not None:
        sections.append("Additional compaction instructions:\n" + request.instructions)
    prefix = "\n\n".join(sections)
    transcript_prefix = "Transcript to compact:\n"
    transcript_digest = _messages_digest(request.messages)
    return prefix, transcript_prefix, transcript_digest


def _bounded_prompt_text(
    prompt: str,
    *,
    max_chars: int | None,
) -> tuple[str, bool]:
    if max_chars is None or len(prompt) <= max_chars:
        return prompt, False
    marker = "[compaction input clipped to latest content]\n"
    keep_chars = max_chars - len(marker)
    if keep_chars <= 0:
        raise ValueError("max_chars is too small for compaction prompt marker.")
    return marker + prompt[-keep_chars:], True
