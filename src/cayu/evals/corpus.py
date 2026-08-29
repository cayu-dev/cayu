from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from decimal import Context, Decimal, localcontext
from functools import lru_cache, partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_PORTABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    compact_json_utf8_size,
    copy_durable_json_object,
    durable_json_object_from_pairs,
    json_utf8_size_within_limit,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_durable_text,
)
from cayu.evals._structural_paths import _validate_portable_structural_workspace_path
from cayu.evals.external import OpaqueExternalCaseRefV1
from cayu.evals.json_subset import copy_eval_tool_json_object
from cayu.evals.models import (
    ARTIFACT_PUBLIC_TEXT_MAX_BYTES,
    EvalCaseContractV1,
    EvalRunContractV1,
)
from cayu.runtime.costs import PriceBook

EVAL_CORPUS_SCHEMA_VERSION = 2
EVALUATION_EVIDENCE_POLICY_SCHEMA_VERSION = 1
PRICING_PROFILE_IDENTITY_SCHEMA_VERSION = 1
PRICING_PROFILE_SEMANTICS_VERSION = 1
EVALUATION_SOURCE_IDENTITY_SCHEMA_VERSION = 1

# This execution-only metadata key crosses from the trusted compiler to the
# publisher. It is intentionally absent from portable corpus documents: the
# target, not the corpus author, resolves the concrete judge implementation.
_MODEL_JUDGE_RESOLVED_IMPLEMENTATION_REVISION_METADATA_KEY = (
    "cayu.model_judge.resolved_implementation_revision"
)
_STRUCTURED_MODEL_JUDGE_RESULT_METADATA_KEY = "cayu.structured_model_judge.result"

EVAL_CORPUS_MAX_BYTES = 8 << 20
EVAL_CORPUS_MAX_SUITES = 64
EVAL_CORPUS_MAX_CASES = 1_000
EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE = 64
# A published result repeats every assertion once per trial. This combined
# ceiling keeps the complete result graph safely below its 32 MiB hard limit,
# including worst-case 4-byte tool names and per-trial/case envelopes, while
# still allowing configurations such as 100 cases x 10 assertions x 10 trials.
EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS = 10_000
EVAL_CORPUS_MAX_MESSAGES_PER_CASE = 16
EVAL_CORPUS_MAX_MESSAGE_CHARS = 65_536
EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS = 262_144
EVAL_CORPUS_MAX_FINAL_OUTPUT_ASSERTION_CHARS = 65_536
EVAL_CORPUS_MAX_JUDGE_RUBRIC_CHARS = 16_384
EVAL_CORPUS_MAX_JUDGE_RUBRIC_VERSION_CHARS = 256
EVAL_CORPUS_MAX_JUDGE_CRITERIA = 8
EVAL_CORPUS_MAX_JUDGE_CRITERION_NAME_CHARS = 128
EVAL_CORPUS_MAX_JUDGE_CRITERION_DESCRIPTION_CHARS = 2_048
EVAL_CORPUS_MAX_JUDGE_EXPLANATION_CHARS = 2_048
EVAL_CORPUS_MAX_JUDGE_REFERENCE_ANSWER_CHARS = 65_536
EVAL_CORPUS_MAX_JUDGE_REFERENCE_FACTS = 64
EVAL_CORPUS_MAX_JUDGE_REFERENCE_FACT_CHARS = 2_048
EVAL_CORPUS_MAX_PUBLISHED_JUDGE_EXPLANATION_CHARS = 2 << 20
EVAL_CORPUS_MAX_TOOL_NAMES = 256
EVAL_CORPUS_MAX_PROCESS_EVENTS = 256
EVAL_CORPUS_MAX_WORKSPACE_PATH_CHARS = 1_024
EVAL_CORPUS_MAX_ARTIFACT_TEXT_ASSERTION_CHARS = 32_768
EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS = 4_096
EVAL_ARTIFACT_EVIDENCE_MAX_ARTIFACTS = 256
EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS = 256
EVAL_CORPUS_MAX_TRIALS = 100
EVAL_CORPUS_MAX_TIMEOUT_SECONDS = 3_600
EVAL_CORPUS_MAX_MERGE_INPUTS = 256

EVIDENCE_MAX_FINAL_OUTPUT_CHARS = 65_536
EVIDENCE_MAX_CHILD_SESSIONS = 500
EVIDENCE_MAX_TOOL_CALLS = 4_096
EVIDENCE_MAX_MODEL_STEPS = 4_096
# Eval corpora cross browser and other IEEE-754 JSON boundaries. Keep every
# numeric token counter exactly representable; larger durable usage is reported
# as limit-exceeded evidence instead of being silently rounded in transit.
EVIDENCE_MAX_TOTAL_TOKENS = MAX_PORTABLE_JSON_INTEGER

_PORTABLE_ID_PATTERN = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z", re.ASCII)
_SHA256_REVISION_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)
_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CURRENCY_PATTERN = re.compile(r"[A-Z][A-Z0-9._-]{0,15}\Z", re.ASCII)
_CANONICAL_NONNEGATIVE_DECIMAL_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z",
    re.ASCII,
)
_STRUCTURED_JUDGE_DECIMAL_CONTEXT = Context(prec=64)

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class _PortableModel(BaseModel):
    """Immutable, closed input accepted by every supported JSON boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
        revalidate_instances="always",
    )


class _SchemaV1PortableModel(_PortableModel):
    """Portable schema root whose version never coerces from another JSON type."""

    schema_version: Literal[1] = 1

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value


class _SchemaV2PortableModel(_PortableModel):
    """Portable schema root whose v2 discriminator never coerces JSON types."""

    schema_version: Literal[2] = 2

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 2.")
        return value


def _bounded_durable_text(
    value: str,
    field_name: str,
    *,
    max_chars: int,
    nonblank: bool,
    clean: bool,
) -> str:
    if type(value) is not str:
        raise ValueError(f"`{field_name}` must be a string.")
    # Every scalar character occupies at least one UTF-8 byte. Apply the cheap
    # declared character ceiling before the durable Unicode/whitespace scan so
    # direct Python construction cannot force work proportional to an
    # arbitrarily oversized string.
    if len(value) > max_chars:
        raise ValueError(f"`{field_name}` must be at most {max_chars} characters.")
    if clean:
        value = require_durable_clean_nonblank(value, field_name)
    elif nonblank:
        value = require_durable_nonblank(value, field_name)
    else:
        value = require_durable_text(value, field_name)
    return value


def _portable_id(value: str, field_name: str) -> str:
    value = _bounded_durable_text(
        value,
        field_name,
        max_chars=128,
        nonblank=True,
        clean=True,
    )
    if _PORTABLE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"`{field_name}` must start with a lowercase ASCII letter and contain only "
            "lowercase ASCII letters, digits, '.', '_', or '-'."
        )
    return value


def _sha256_revision(value: str, field_name: str) -> str:
    value = _bounded_durable_text(
        value,
        field_name,
        max_chars=71,
        nonblank=True,
        clean=True,
    )
    if _SHA256_REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"`{field_name}` must be a lowercase sha256 content revision.")
    return value


def _sha256_hex(value: str, field_name: str) -> str:
    value = _bounded_durable_text(
        value,
        field_name,
        max_chars=64,
        nonblank=True,
        clean=True,
    )
    if _SHA256_HEX_PATTERN.fullmatch(value) is None:
        raise ValueError(f"`{field_name}` must be a lowercase SHA-256 hex digest.")
    return value


def _canonical_decimal_text(
    value: str,
    field_name: str,
    *,
    max_chars: int = 64,
) -> str:
    value = _bounded_durable_text(
        value,
        field_name,
        max_chars=max_chars,
        nonblank=True,
        clean=True,
    )
    if _CANONICAL_NONNEGATIVE_DECIMAL_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"`{field_name}` must use canonical decimal notation for a non-negative value."
        )
    return value


def _exact_decimal_sum(values: Iterable[Decimal]) -> Decimal:
    """Add bounded structured-judge decimals independently of ambient context."""

    with localcontext(_STRUCTURED_JUDGE_DECIMAL_CONTEXT):
        return sum(values, Decimal(0))


def _exact_weighted_decimal(
    values: Iterable[tuple[str, Decimal]],
) -> Decimal:
    """Multiply and add bounded structured scores without contextual rounding."""

    with localcontext(_STRUCTURED_JUDGE_DECIMAL_CONTEXT):
        return sum(
            (Decimal(weight) * score for weight, score in values),
            Decimal(0),
        )


def _exact_decimal_difference(left: Decimal, right: Decimal) -> Decimal:
    """Subtract bounded structured decimals without ambient-context rounding."""

    with localcontext(_STRUCTURED_JUDGE_DECIMAL_CONTEXT):
        return left - right


def _ordered_sequence_input(
    value: object,
    field_name: str,
) -> list[Any] | tuple[Any, ...]:
    """Reject unordered Python containers before Pydantic can coerce them."""

    if not isinstance(value, list | tuple):
        raise ValueError(f"`{field_name}` must be an ordered array.")
    return value


def _ordered_sequence_argument(value: object, field_name: str) -> list[Any] | tuple[Any, ...]:
    """Validate an ordered public factory argument without consuming iterators."""

    if not isinstance(value, list | tuple):
        raise TypeError(f"{field_name} must be an ordered sequence (a list or tuple).")
    return value


def _model_python_input(value: BaseModel) -> dict[str, Any]:
    """Dump potentially forged public state without trusting serializer warnings."""

    return value.model_dump(mode="python", round_trip=True, warnings="none")


def _validated_model_document(
    model: _ModelT,
    *,
    model_type: type[_ModelT],
    field_name: str,
) -> tuple[_ModelT, dict[str, Any]]:
    if type(model) is not model_type:
        raise TypeError(f"{field_name} must be a {model_type.__name__}.")
    validated = model_type.model_validate(_model_python_input(model))
    document = copy_durable_json_object(validated.model_dump(mode="json"), field_name)
    return validated, document


def _content_revision(document: Mapping[str, Any], field_name: str) -> str:
    copied = copy_durable_json_object(document, field_name)
    copied.pop("revision", None)
    digest = hashlib.sha256(canonical_durable_json_bytes(copied, field_name)).hexdigest()
    return f"sha256:{digest}"


def _model_content_revision(model: BaseModel, field_name: str) -> str:
    return _content_revision(model.model_dump(mode="json"), field_name)


def _pretty_json_size_within_limit(value: BaseModel, max_bytes: int) -> bool:
    """Bound the deterministic export form after the allocation-free compact guard."""

    document = value.model_dump(mode="json")
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, sort_keys=True)
    total_bytes = 1  # final newline
    for chunk in encoder.iterencode(document):
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > max_bytes:
            return False
    return True


class CorpusUserMessageSpec(_PortableModel):
    """One portable user-role text message; no structured or executable input."""

    role: Literal["user"] = "user"
    text: StrictStr

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_MESSAGE_CHARS,
            nonblank=True,
            clean=False,
        )


class RunInputSpec(_PortableModel):
    """The authority-free user input supplied after a trusted local bootstrap."""

    messages: tuple[CorpusUserMessageSpec, ...] = Field(
        default=(),
        max_length=EVAL_CORPUS_MAX_MESSAGES_PER_CASE,
    )
    opaque_external_case_ref: OpaqueExternalCaseRefV1 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("opaque_external_case_ref", mode="before")
    @classmethod
    def copy_opaque_external_case_ref(cls, value: object) -> object:
        if type(value) is OpaqueExternalCaseRefV1:
            return value.model_dump(mode="json")
        return value

    @model_validator(mode="after")
    def validate_total_text(self) -> RunInputSpec:
        if not self.messages and self.opaque_external_case_ref is None:
            raise ValueError("Run input requires messages or one opaque external case reference.")
        total = sum(len(message.text) for message in self.messages)
        if total > EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS:
            raise ValueError(
                "Run input text must not exceed "
                f"{EVAL_CORPUS_MAX_TOTAL_MESSAGE_CHARS} total characters."
            )
        return self


class TrialRequestSpec(_PortableModel):
    """Sequential, bounded fresh-evaluation execution settings."""

    trials: StrictInt = Field(default=1, ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    timeout_seconds: StrictInt = Field(default=300, ge=1, le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS)


class _AssertionSpecBase(_PortableModel):
    id: StrictStr
    description: StrictStr | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=2_048,
            nonblank=True,
            clean=True,
        )


class RootStatusAssertionSpec(_AssertionSpecBase):
    kind: Literal["root_status"] = "root_status"
    expected: Literal["completed", "failed"]


class ChildStatusAssertionSpec(_AssertionSpecBase):
    kind: Literal["child_status"] = "child_status"
    expected: Literal["completed", "failed", "interrupted"]
    min_count: StrictInt = Field(default=1, ge=0, le=EVIDENCE_MAX_CHILD_SESSIONS)
    max_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_CHILD_SESSIONS)

    @model_validator(mode="after")
    def validate_count_range(self) -> ChildStatusAssertionSpec:
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count.")
        return self


class FinalOutputEqualsAssertionSpec(_AssertionSpecBase):
    kind: Literal["final_output_equals"] = "final_output_equals"
    expected: StrictStr

    @field_validator("expected")
    @classmethod
    def validate_expected(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_FINAL_OUTPUT_ASSERTION_CHARS,
            nonblank=False,
            clean=False,
        )


class FinalOutputContainsAssertionSpec(_AssertionSpecBase):
    kind: Literal["final_output_contains"] = "final_output_contains"
    expected: StrictStr

    @field_validator("expected")
    @classmethod
    def validate_expected(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_FINAL_OUTPUT_ASSERTION_CHARS,
            nonblank=True,
            clean=False,
        )


class ToolCalledAssertionSpec(_AssertionSpecBase):
    kind: Literal["tool_called"] = "tool_called"
    tool_name: StrictStr
    min_count: StrictInt = Field(default=1, ge=0, le=EVIDENCE_MAX_TOOL_CALLS)
    max_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOOL_CALLS)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_count_range(self) -> ToolCalledAssertionSpec:
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count.")
        return self


class _ToolJsonSubsetAssertionSpec(_AssertionSpecBase):
    tool_name: StrictStr
    occurrence: StrictInt = Field(default=1, ge=1, le=EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS)
    expected_subset: dict[str, Any]

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("expected_subset", mode="before")
    @classmethod
    def validate_expected_subset(cls, value: object, info) -> dict[str, Any]:
        return copy_eval_tool_json_object(value, info.field_name)


class ToolArgumentsContainAssertionSpec(_ToolJsonSubsetAssertionSpec):
    """Require one started invocation's public arguments to contain bounded JSON."""

    kind: Literal["tool_arguments_contain"] = "tool_arguments_contain"


class ToolResultContainsAssertionSpec(_ToolJsonSubsetAssertionSpec):
    """Require one explicitly retained public-safe result to contain bounded JSON."""

    kind: Literal["tool_result_contains"] = "tool_result_contains"

    @model_validator(mode="after")
    def validate_result_root(self) -> ToolResultContainsAssertionSpec:
        unsupported = sorted(set(self.expected_subset) - {"content", "structured", "is_error"})
        if unsupported:
            raise ValueError(
                "Tool-result subsets support only content, structured, and is_error; got: "
                + ", ".join(unsupported)
                + "."
            )
        if not self.expected_subset:
            raise ValueError("Tool-result subsets must select at least one public result field.")
        return self


class ToolsCalledInOrderAssertionSpec(_AssertionSpecBase):
    kind: Literal["tools_called_in_order"] = "tools_called_in_order"
    tool_names: tuple[StrictStr, ...] = Field(max_length=EVAL_CORPUS_MAX_TOOL_NAMES)

    @field_validator("tool_names", mode="before")
    @classmethod
    def validate_tool_names(cls, value: object, info) -> tuple[str, ...]:
        ordered = _ordered_sequence_input(value, info.field_name)
        return tuple(
            _bounded_durable_text(
                item,
                f"{info.field_name}[{index}]",
                max_chars=256,
                nonblank=True,
                clean=True,
            )
            for index, item in enumerate(ordered)
        )


EvalProcessEventKind: TypeAlias = Literal[
    "session_started",
    "session_resumed",
    "session_awaiting_user_input",
    "session_completed",
    "session_failed",
    "session_interrupted",
    "session_limit_reached",
    "tool_call_started",
    "tool_call_completed",
    "tool_call_failed",
    "tool_call_blocked",
    "tool_approval_requested",
    "tool_approved",
    "tool_approval_denied",
    "tool_approval_expired",
    "structured_output_validated",
    "structured_output_failed",
    "budget_limit_reached",
]


class ProcessEventAssertionSpec(_AssertionSpecBase):
    """Require or forbid one closed, payload-free runtime process fact."""

    kind: Literal["process_event"] = "process_event"
    event: EvalProcessEventKind
    min_count: StrictInt = Field(default=1, ge=0, le=EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS)
    max_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS,
    )

    @model_validator(mode="after")
    def validate_count_range(self) -> ProcessEventAssertionSpec:
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count.")
        return self


class ProcessEventsInOrderAssertionSpec(_AssertionSpecBase):
    """Require the exact filtered order of selected portable process facts."""

    kind: Literal["process_events_in_order"] = "process_events_in_order"
    events: tuple[EvalProcessEventKind, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_PROCESS_EVENTS,
    )

    @field_validator("events", mode="before")
    @classmethod
    def validate_events_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)


class WorkspaceFileAssertionSpec(_AssertionSpecBase):
    """Require one declared workspace path to have bounded structural properties."""

    kind: Literal["workspace_file"] = "workspace_file"
    path: StrictStr
    present: StrictBool = True
    minimum_bytes: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    maximum_bytes: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    sha256: StrictStr | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str, info) -> str:
        path = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_WORKSPACE_PATH_CHARS,
            nonblank=True,
            clean=True,
        )
        return _validate_portable_structural_workspace_path(path)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> WorkspaceFileAssertionSpec:
        if self.maximum_bytes is not None and (
            self.minimum_bytes is not None and self.maximum_bytes < self.minimum_bytes
        ):
            raise ValueError("maximum_bytes must be greater than or equal to minimum_bytes.")
        if not self.present and any(
            value is not None for value in (self.minimum_bytes, self.maximum_bytes, self.sha256)
        ):
            raise ValueError("Absent workspace assertions cannot require size or digest values.")
        return self


class ArtifactAssertionSpec(_AssertionSpecBase):
    """Require bounded structural or explicitly retained public-text artifact evidence."""

    kind: Literal["artifact"] = "artifact"
    scope: Literal["session", "environment"] = "session"
    filename: StrictStr | None = None
    content_type: StrictStr | None = None
    minimum_bytes: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    maximum_bytes: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    sha256: StrictStr | None = None
    text_contains: StrictStr | None = None
    min_count: StrictInt = Field(default=1, ge=0, le=EVAL_ARTIFACT_EVIDENCE_MAX_ARTIFACTS)
    max_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_ARTIFACT_EVIDENCE_MAX_ARTIFACTS,
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=1_024,
            nonblank=True,
            clean=False,
        )

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        content_type = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=1_024,
            nonblank=True,
            clean=True,
        )
        if any(ord(character) < 0x20 or ord(character) > 0x7E for character in content_type):
            raise ValueError("content_type must contain printable ASCII characters only.")
        return content_type

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("text_contains")
    @classmethod
    def validate_text_contains(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        text = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_ARTIFACT_TEXT_ASSERTION_CHARS,
            nonblank=True,
            clean=False,
        )
        if len(text.encode("utf-8")) > ARTIFACT_PUBLIC_TEXT_MAX_BYTES:
            raise ValueError(
                f"{info.field_name} must be at most {ARTIFACT_PUBLIC_TEXT_MAX_BYTES} UTF-8 bytes."
            )
        return text

    @model_validator(mode="after")
    def validate_contract(self) -> ArtifactAssertionSpec:
        if self.maximum_bytes is not None and (
            self.minimum_bytes is not None and self.maximum_bytes < self.minimum_bytes
        ):
            raise ValueError("maximum_bytes must be greater than or equal to minimum_bytes.")
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count.")
        return self


class MaxToolCallsAssertionSpec(_AssertionSpecBase):
    kind: Literal["max_tool_calls"] = "max_tool_calls"
    maximum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_TOOL_CALLS)


class MaxModelStepsAssertionSpec(_AssertionSpecBase):
    kind: Literal["max_model_steps"] = "max_model_steps"
    maximum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_MODEL_STEPS)


class UsageRecordedAssertionSpec(_AssertionSpecBase):
    kind: Literal["usage_recorded"] = "usage_recorded"
    min_total_tokens: StrictInt = Field(default=1, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)


class MaxTotalTokensAssertionSpec(_AssertionSpecBase):
    kind: Literal["max_total_tokens"] = "max_total_tokens"
    maximum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)


class MaxEstimatedCostAssertionSpec(_AssertionSpecBase):
    kind: Literal["max_estimated_cost"] = "max_estimated_cost"
    maximum: StrictStr
    currency: StrictStr = "USD"

    @field_validator("maximum")
    @classmethod
    def validate_maximum(cls, value: str, info) -> str:
        return _canonical_decimal_text(value, info.field_name)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str, info) -> str:
        value = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=16,
            nonblank=True,
            clean=True,
        )
        if _CURRENCY_PATTERN.fullmatch(value) is None:
            raise ValueError("currency must use uppercase ASCII letters, digits, '.', '_', or '-'.")
        return value


class ModelJudgeAssertionSpec(_AssertionSpecBase):
    """Authority-free graded evaluation resolved by one trusted target."""

    kind: Literal["model_judge"] = "model_judge"
    evaluator_key: StrictStr
    rubric: StrictStr
    rubric_version: StrictStr
    threshold: StrictFloat = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    include_transcript: StrictBool = False

    @field_validator("evaluator_key")
    @classmethod
    def validate_evaluator_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("rubric")
    @classmethod
    def validate_rubric(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_JUDGE_RUBRIC_CHARS,
            nonblank=True,
            clean=False,
        )

    @field_validator("rubric_version")
    @classmethod
    def validate_rubric_version(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_JUDGE_RUBRIC_VERSION_CHARS,
            nonblank=True,
            clean=True,
        )


def _unit_interval_decimal_text(value: str, field_name: str) -> str:
    value = _canonical_decimal_text(value, field_name, max_chars=20)
    if Decimal(value) > 1:
        raise ValueError(f"`{field_name}` must be between 0 and 1.")
    return value


class StructuredRubricCriterionV1(_PortableModel):
    """One stable, weighted dimension that Cayu—not the judge—aggregates."""

    id: StrictStr
    name: StrictStr
    description: StrictStr
    weight: StrictStr

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_JUDGE_CRITERION_NAME_CHARS,
            nonblank=True,
            clean=True,
        )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_JUDGE_CRITERION_DESCRIPTION_CHARS,
            nonblank=True,
            clean=False,
        )

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: str, info) -> str:
        return _unit_interval_decimal_text(value, info.field_name)


class StructuredRubricV1(_SchemaV1PortableModel):
    """Content-addressed rubric with an exact, deterministic weight partition."""

    id: StrictStr
    revision: StrictStr
    criteria: tuple[StructuredRubricCriterionV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("criteria", mode="before")
    @classmethod
    def validate_criteria_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> StructuredRubricV1:
        criterion_ids = tuple(criterion.id for criterion in self.criteria)
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("Structured rubric criterion IDs must be unique.")
        if _exact_decimal_sum(Decimal(criterion.weight) for criterion in self.criteria) != 1:
            raise ValueError("Structured rubric criterion weights must sum exactly to 1.")
        if self.revision != _model_content_revision(self, "structured rubric"):
            raise ValueError("Structured rubric revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        id: str,
        criteria: Sequence[StructuredRubricCriterionV1],
    ) -> StructuredRubricV1:
        ordered = _ordered_sequence_argument(criteria, "criteria")
        validated: list[StructuredRubricCriterionV1] = []
        for criterion in ordered:
            if type(criterion) is not StructuredRubricCriterionV1:
                raise TypeError(
                    "criteria must contain exact StructuredRubricCriterionV1 instances."
                )
            validated.append(
                StructuredRubricCriterionV1.model_validate(_model_python_input(criterion))
            )
        document = {
            "schema_version": 1,
            "id": id,
            "criteria": [criterion.model_dump(mode="json") for criterion in validated],
        }
        return cls(
            revision=_content_revision(document, "structured rubric"),
            id=id,
            criteria=tuple(validated),
        )


class EvalJudgeEvidenceSelectionV1(_SchemaV1PortableModel):
    """Candidate evidence a public judge profile is requested to receive."""

    include_final_output: Literal[True] = True
    include_transcript: StrictBool = False

    @field_validator("include_final_output", mode="before")
    @classmethod
    def validate_final_output_type(cls, value: object) -> object:
        if type(value) is not bool or value is not True:
            raise ValueError("include_final_output must be true.")
        return value


class PublicJudgeReferenceV1(_SchemaV1PortableModel):
    """Evaluator-only, deliberately portable reference truth."""

    kind: Literal["public_reference"] = "public_reference"
    id: StrictStr
    revision: StrictStr
    expected_answer: StrictStr | None = None
    expected_facts: tuple[StrictStr, ...] = Field(
        default=(),
        max_length=EVAL_CORPUS_MAX_JUDGE_REFERENCE_FACTS,
    )

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("expected_answer")
    @classmethod
    def validate_expected_answer(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_JUDGE_REFERENCE_ANSWER_CHARS,
            nonblank=True,
            clean=False,
        )

    @field_validator("expected_facts", mode="before")
    @classmethod
    def validate_expected_facts(cls, value: object, info) -> tuple[str, ...]:
        ordered = _ordered_sequence_input(value, info.field_name)
        return tuple(
            _bounded_durable_text(
                fact,
                f"{info.field_name}[{index}]",
                max_chars=EVAL_CORPUS_MAX_JUDGE_REFERENCE_FACT_CHARS,
                nonblank=True,
                clean=False,
            )
            for index, fact in enumerate(ordered)
        )

    @model_validator(mode="after")
    def validate_contract(self) -> PublicJudgeReferenceV1:
        if self.expected_answer is None and not self.expected_facts:
            raise ValueError("A public judge reference requires an answer or expected facts.")
        if self.revision != _model_content_revision(self, "public judge reference"):
            raise ValueError("Public judge reference revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        id: str,
        expected_answer: str | None = None,
        expected_facts: Sequence[str] = (),
    ) -> PublicJudgeReferenceV1:
        ordered_facts = _ordered_sequence_argument(expected_facts, "expected_facts")
        document = {
            "schema_version": 1,
            "kind": "public_reference",
            "id": id,
            "expected_answer": expected_answer,
            "expected_facts": list(ordered_facts),
        }
        return cls(
            revision=_content_revision(document, "public judge reference"),
            id=id,
            expected_answer=expected_answer,
            expected_facts=tuple(ordered_facts),
        )


class PrivateJudgeReferenceV1(_SchemaV1PortableModel):
    """Authority-free identity for evaluator truth retained only by the server."""

    kind: Literal["private_reference"] = "private_reference"
    key: StrictStr
    revision: StrictStr
    privacy_policy_key: StrictStr
    privacy_policy_revision: StrictStr

    @field_validator("key", "privacy_policy_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision", "privacy_policy_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)


JudgeReferenceV1: TypeAlias = Annotated[
    PublicJudgeReferenceV1 | PrivateJudgeReferenceV1,
    Field(discriminator="kind"),
]


class JudgePrivacyPolicyV1(_SchemaV1PortableModel):
    """Public identity of the server policy controlling evaluator-only data."""

    key: StrictStr
    revision: StrictStr
    allow_transcript: StrictBool = False
    allow_public_reference: StrictBool = True
    allow_private_reference: StrictBool = False

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> JudgePrivacyPolicyV1:
        if self.revision != _model_content_revision(self, "judge privacy policy"):
            raise ValueError("Judge privacy policy revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        key: str,
        allow_transcript: bool = False,
        allow_public_reference: bool = True,
        allow_private_reference: bool = False,
    ) -> JudgePrivacyPolicyV1:
        document = {
            "schema_version": 1,
            "key": key,
            "allow_transcript": allow_transcript,
            "allow_public_reference": allow_public_reference,
            "allow_private_reference": allow_private_reference,
        }
        return cls(revision=_content_revision(document, "judge privacy policy"), **document)

    @classmethod
    def public_only(cls) -> JudgePrivacyPolicyV1:
        return cls.create(key="public-only")


class JudgeProfileIdentityV1(_SchemaV1PortableModel):
    """Safe, immutable public snapshot of one trusted model-judge route."""

    key: StrictStr
    revision: StrictStr
    label: StrictStr
    provider_name: StrictStr
    model: StrictStr
    implementation_revision: StrictStr
    allowed_evidence: tuple[
        Literal["final_output", "transcript", "public_reference", "private_reference"], ...
    ] = Field(min_length=1, max_length=4)
    timeout_seconds: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS)
    max_input_tokens: StrictInt = Field(ge=1, le=EVIDENCE_MAX_TOTAL_TOKENS)
    max_output_tokens: StrictInt = Field(ge=1, le=EVIDENCE_MAX_TOTAL_TOKENS)
    max_total_tokens: StrictInt = Field(ge=1, le=EVIDENCE_MAX_TOTAL_TOKENS)
    max_estimated_cost: StrictStr | None = None
    cost_currency: StrictStr | None = None
    pricing_profile_fingerprint: StrictStr | None = None
    privacy_policy_key: StrictStr
    privacy_policy_revision: StrictStr
    same_model_use: Literal["forbidden", "allowed_and_labeled"] = "forbidden"

    @field_validator("key", "privacy_policy_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision", "implementation_revision", "privacy_policy_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("label", "provider_name", "model")
    @classmethod
    def validate_public_text(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=512,
            nonblank=True,
            clean=True,
        )

    @field_validator("allowed_evidence", mode="before")
    @classmethod
    def validate_allowed_evidence_order(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("max_estimated_cost")
    @classmethod
    def validate_max_estimated_cost(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = _canonical_decimal_text(value, info.field_name)
        if Decimal(value) <= 0:
            raise ValueError("max_estimated_cost must be greater than zero.")
        return value

    @field_validator("cost_currency")
    @classmethod
    def validate_cost_currency(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=16,
            nonblank=True,
            clean=True,
        )
        if _CURRENCY_PATTERN.fullmatch(value) is None:
            raise ValueError("cost_currency must be a portable uppercase identifier.")
        return value

    @field_validator("pricing_profile_fingerprint")
    @classmethod
    def validate_pricing_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> JudgeProfileIdentityV1:
        canonical_evidence = tuple(
            item
            for item in (
                "final_output",
                "transcript",
                "public_reference",
                "private_reference",
            )
            if item in self.allowed_evidence
        )
        if self.allowed_evidence != canonical_evidence:
            raise ValueError("allowed_evidence must be unique and canonically ordered.")
        if self.allowed_evidence[0] != "final_output":
            raise ValueError("Judge profiles must allow final-output evidence.")
        if self.max_total_tokens < max(self.max_input_tokens, self.max_output_tokens):
            raise ValueError("max_total_tokens cannot be below an individual token ceiling.")
        cost_fields = (
            self.max_estimated_cost,
            self.cost_currency,
            self.pricing_profile_fingerprint,
        )
        if any(item is None for item in cost_fields) and any(
            item is not None for item in cost_fields
        ):
            raise ValueError("Judge-profile cost ceilings require complete pricing identity.")
        if self.revision != _model_content_revision(self, "judge profile identity"):
            raise ValueError("Judge profile revision does not match its content.")
        return self


class StructuredModelJudgeAssertionSpec(_AssertionSpecBase):
    """Typed rubric judgment bound to an exact trusted server profile."""

    kind: Literal["structured_model_judge"] = "structured_model_judge"
    judge_profile_key: StrictStr
    judge_profile_revision: StrictStr
    rubric: StructuredRubricV1
    reference: JudgeReferenceV1 | None = None
    threshold: StrictStr = "0.5"
    evidence: EvalJudgeEvidenceSelectionV1 = Field(default_factory=EvalJudgeEvidenceSelectionV1)

    @field_validator("judge_profile_key")
    @classmethod
    def validate_judge_profile_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("judge_profile_revision")
    @classmethod
    def validate_judge_profile_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: str, info) -> str:
        return _unit_interval_decimal_text(value, info.field_name)


AssertionSpec: TypeAlias = Annotated[
    RootStatusAssertionSpec
    | ChildStatusAssertionSpec
    | FinalOutputEqualsAssertionSpec
    | FinalOutputContainsAssertionSpec
    | ToolCalledAssertionSpec
    | ToolArgumentsContainAssertionSpec
    | ToolResultContainsAssertionSpec
    | ToolsCalledInOrderAssertionSpec
    | ProcessEventAssertionSpec
    | ProcessEventsInOrderAssertionSpec
    | WorkspaceFileAssertionSpec
    | ArtifactAssertionSpec
    | MaxToolCallsAssertionSpec
    | MaxModelStepsAssertionSpec
    | UsageRecordedAssertionSpec
    | MaxTotalTokensAssertionSpec
    | MaxEstimatedCostAssertionSpec
    | ModelJudgeAssertionSpec
    | StructuredModelJudgeAssertionSpec,
    Field(discriminator="kind"),
]

_ASSERTION_SPEC_TYPES = (
    RootStatusAssertionSpec,
    ChildStatusAssertionSpec,
    FinalOutputEqualsAssertionSpec,
    FinalOutputContainsAssertionSpec,
    ToolCalledAssertionSpec,
    ToolArgumentsContainAssertionSpec,
    ToolResultContainsAssertionSpec,
    ToolsCalledInOrderAssertionSpec,
    ProcessEventAssertionSpec,
    ProcessEventsInOrderAssertionSpec,
    WorkspaceFileAssertionSpec,
    ArtifactAssertionSpec,
    MaxToolCallsAssertionSpec,
    MaxModelStepsAssertionSpec,
    UsageRecordedAssertionSpec,
    MaxTotalTokensAssertionSpec,
    MaxEstimatedCostAssertionSpec,
    ModelJudgeAssertionSpec,
    StructuredModelJudgeAssertionSpec,
)


def _validated_assertion_spec(spec: AssertionSpec) -> AssertionSpec:
    spec_type = type(spec)
    if spec_type not in _ASSERTION_SPEC_TYPES:
        raise TypeError("Portable assertions must use an exact built-in AssertionSpec type.")
    validated = spec_type.model_validate(_model_python_input(spec))
    return validated


def assertion_spec_revision(spec: AssertionSpec) -> str:
    validated = _validated_assertion_spec(spec)
    return _model_content_revision(validated, "assertion spec")


class EvaluationSourceIdentityV1(_SchemaV1PortableModel):
    """Diagnostic capture or authored-definition provenance without runtime authority."""

    schema_version: Literal[1] = EVALUATION_SOURCE_IDENTITY_SCHEMA_VERSION
    application_release_id: StrictStr
    app_manifest_schema_version: StrictStr
    app_manifest_fingerprint: StrictStr
    evidence_revision: StrictStr

    @field_validator("application_release_id")
    @classmethod
    def validate_release_id(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("app_manifest_schema_version")
    @classmethod
    def validate_manifest_schema_version(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=32,
            nonblank=True,
            clean=True,
        )

    @field_validator("app_manifest_fingerprint")
    @classmethod
    def validate_manifest_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("evidence_revision")
    @classmethod
    def validate_evidence_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)


class PricingProfileIdentityV1(_SchemaV1PortableModel):
    """Content identity for trusted pricing, never the actual PriceBook."""

    schema_version: Literal[1] = PRICING_PROFILE_IDENTITY_SCHEMA_VERSION
    pricing_semantics_version: Literal[1] = PRICING_PROFILE_SEMANTICS_VERSION
    fingerprint: StrictStr
    price_book_version: StrictStr
    generated_at: StrictStr
    currencies: tuple[StrictStr, ...] = Field(min_length=1, max_length=32)

    @field_validator("pricing_semantics_version", mode="before")
    @classmethod
    def validate_pricing_semantics_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("pricing_semantics_version must be the integer 1.")
        return value

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("price_book_version", "generated_at")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("currencies", mode="before")
    @classmethod
    def validate_currencies(cls, value: object, info) -> tuple[str, ...]:
        ordered = _ordered_sequence_input(value, info.field_name)
        cleaned: list[str] = []
        for currency in ordered:
            currency = _bounded_durable_text(
                currency,
                info.field_name,
                max_chars=16,
                nonblank=True,
                clean=True,
            )
            if _CURRENCY_PATTERN.fullmatch(currency) is None:
                raise ValueError("Pricing currencies must use a portable uppercase identifier.")
            cleaned.append(currency)
        if tuple(cleaned) != tuple(sorted(set(cleaned))):
            raise ValueError("Pricing currencies must be unique and sorted.")
        return tuple(cleaned)


def pricing_profile_identity(price_book: PriceBook) -> PricingProfileIdentityV1:
    """Fingerprint one validated PriceBook without carrying its executable pricing data."""

    if type(price_book) is not PriceBook:
        raise TypeError("price_book must be an exact PriceBook.")
    validated = PriceBook.model_validate(_model_python_input(price_book))
    return _pricing_profile_identity_from_validated_price_book(validated)


def _pricing_profile_identity_from_validated_price_book(
    validated: PriceBook,
) -> PricingProfileIdentityV1:
    """Fingerprint one detached, already validated exact PriceBook snapshot."""

    if type(validated) is not PriceBook:
        raise TypeError("validated must be an exact PriceBook.")
    ordered = validated.model_copy(
        update={
            "prices": tuple(
                sorted(
                    validated.prices,
                    key=lambda price: (
                        price.provider_name,
                        price.model,
                        price.match,
                        (
                            ()
                            if price.pricing_context is None
                            else price.pricing_context.storage_key()
                        ),
                    ),
                )
            ),
            "contextual_pricing_requirements": tuple(
                sorted(
                    validated.contextual_pricing_requirements,
                    key=lambda requirement: requirement.provider_name,
                )
            ),
            "resource_mappings": tuple(
                sorted(
                    validated.resource_mappings,
                    key=lambda mapping: (mapping.provider_name, mapping.resource_id),
                )
            ),
        }
    )
    price_book_document = copy_durable_json_object(
        ordered.model_dump(mode="json"),
        "price book",
    )
    fingerprint_document = {
        "pricing_semantics_version": PRICING_PROFILE_SEMANTICS_VERSION,
        "price_book": price_book_document,
    }
    if not json_utf8_size_within_limit(fingerprint_document, EVAL_CORPUS_MAX_BYTES):
        raise ValueError(
            f"Price book identity input exceeds {EVAL_CORPUS_MAX_BYTES} canonical JSON bytes."
        )
    fingerprint = (
        "sha256:"
        + hashlib.sha256(
            canonical_durable_json_bytes(fingerprint_document, "price book identity")
        ).hexdigest()
    )
    currencies = tuple(
        sorted(
            {
                schedule.pricing.currency.upper()
                for price in validated.prices
                for schedule in price.schedules
            }
        )
    )
    return PricingProfileIdentityV1(
        pricing_semantics_version=PRICING_PROFILE_SEMANTICS_VERSION,
        fingerprint=fingerprint,
        price_book_version=validated.price_book_version,
        generated_at=validated.generated_at,
        currencies=currencies,
    )


class EvaluationEvidencePolicySpec(_SchemaV1PortableModel):
    """Complete v1 redaction, tool retention, and cardinality behavior."""

    schema_version: Literal[1] = EVALUATION_EVIDENCE_POLICY_SCHEMA_VERSION
    revision: StrictStr
    event_projection: Literal["runtime_public_alias_free_v1"] = "runtime_public_alias_free_v1"
    include_event_payloads: Literal[False] = False
    include_transcript_text: Literal[False] = False
    include_tool_arguments: StrictBool = True
    include_tool_results: StrictBool = False
    include_artifact_text: StrictBool = False
    max_final_output_chars: Literal[65536] = EVIDENCE_MAX_FINAL_OUTPUT_CHARS
    max_child_sessions: Literal[500] = EVIDENCE_MAX_CHILD_SESSIONS
    max_tool_calls: Literal[4096] = EVIDENCE_MAX_TOOL_CALLS
    max_model_steps: Literal[4096] = EVIDENCE_MAX_MODEL_STEPS
    max_total_tokens: Literal[9007199254740991] = EVIDENCE_MAX_TOTAL_TOKENS

    @field_validator(
        "include_event_payloads",
        "include_transcript_text",
        mode="before",
    )
    @classmethod
    def validate_omission_flag_types(cls, value: object, info) -> object:
        if type(value) is not bool:
            raise ValueError(f"{info.field_name} must be false.")
        return value

    @field_validator(
        "include_tool_arguments",
        "include_tool_results",
        "include_artifact_text",
        mode="before",
    )
    @classmethod
    def validate_tool_evidence_flag_types(cls, value: object, info) -> object:
        if type(value) is not bool:
            raise ValueError(f"{info.field_name} must be a boolean.")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_revision(self) -> EvaluationEvidencePolicySpec:
        expected = _model_content_revision(self, "evaluation evidence policy")
        if self.revision != expected:
            raise ValueError("Evaluation evidence policy revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        include_tool_arguments: bool = True,
        include_tool_results: bool = False,
        include_artifact_text: bool = False,
    ) -> EvaluationEvidencePolicySpec:
        if type(include_tool_arguments) is not bool:
            raise TypeError("include_tool_arguments must be a bool.")
        if type(include_tool_results) is not bool:
            raise TypeError("include_tool_results must be a bool.")
        if type(include_artifact_text) is not bool:
            raise TypeError("include_artifact_text must be a bool.")
        document = {
            "schema_version": EVALUATION_EVIDENCE_POLICY_SCHEMA_VERSION,
            "event_projection": "runtime_public_alias_free_v1",
            "include_event_payloads": False,
            "include_transcript_text": False,
            "include_tool_arguments": include_tool_arguments,
            "include_tool_results": include_tool_results,
            "include_artifact_text": include_artifact_text,
            "max_final_output_chars": EVIDENCE_MAX_FINAL_OUTPUT_CHARS,
            "max_child_sessions": EVIDENCE_MAX_CHILD_SESSIONS,
            "max_tool_calls": EVIDENCE_MAX_TOOL_CALLS,
            "max_model_steps": EVIDENCE_MAX_MODEL_STEPS,
            "max_total_tokens": EVIDENCE_MAX_TOTAL_TOKENS,
        }
        return cls.model_validate(
            {
                "revision": _content_revision(document, "evaluation evidence policy"),
                **document,
            }
        )

    @classmethod
    def standard(cls) -> EvaluationEvidencePolicySpec:
        """Default policy: existing public arguments, but no result retention."""

        return cls.create()

    @classmethod
    @lru_cache(maxsize=1)
    def _supported_policies(cls) -> tuple[EvaluationEvidencePolicySpec, ...]:
        return tuple(
            cls.create(
                include_tool_arguments=include_arguments,
                include_tool_results=include_results,
                include_artifact_text=include_artifact_text,
            )
            for include_arguments in (False, True)
            for include_results in (False, True)
            for include_artifact_text in (False, True)
        )

    @classmethod
    @lru_cache(maxsize=1)
    def supported_revisions(cls) -> frozenset[str]:
        """Every fixed policy combination understood by this wire generation."""

        return frozenset(policy.revision for policy in cls._supported_policies())

    @classmethod
    def policy_for_revision(cls, revision: str) -> EvaluationEvidencePolicySpec:
        """Resolve one supported revision to the exact retention policy it identifies."""

        if type(revision) is not str:
            raise TypeError("revision must be a string.")
        for policy in cls._supported_policies():
            if policy.revision == revision:
                return policy
        raise ValueError("Evaluation evidence policy revision is not supported.")


class EvalSuiteSpec(_PortableModel):
    """Reusable execution settings; case membership remains mergeable by suite_id."""

    id: StrictStr
    revision: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    trial_request: TrialRequestSpec = Field(default_factory=TrialRequestSpec)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=2_048,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_revision(self) -> EvalSuiteSpec:
        expected = _model_content_revision(self, "eval suite spec")
        if self.revision != expected:
            raise ValueError("Eval suite revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        id: str,
        name: str,
        description: str | None = None,
        trial_request: TrialRequestSpec | None = None,
    ) -> EvalSuiteSpec:
        if trial_request is None:
            validated_trial_request = TrialRequestSpec()
        elif type(trial_request) is TrialRequestSpec:
            validated_trial_request = TrialRequestSpec.model_validate(
                _model_python_input(trial_request)
            )
        else:
            raise TypeError("trial_request must be an exact TrialRequestSpec.")
        document: dict[str, Any] = {
            "id": id,
            "name": name,
            "description": description,
            "trial_request": validated_trial_request.model_dump(mode="json"),
        }
        return cls(revision=_content_revision(document, "eval suite spec"), **document)


class EvalCaseSpec(_PortableModel):
    """One deterministic expectation set with optional fresh-run input.

    Captured-session evaluations intentionally set ``input`` to ``None`` when
    the retained evidence cannot be represented as one corpus-v2 invocation.
    Such cases remain portable historical evaluation contracts, but execution
    rejects them until a runnable input or scenario is authored.
    """

    id: StrictStr
    revision: StrictStr
    suite_id: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    source: EvaluationSourceIdentityV1
    # The field is required even when its value is null.  Requiring an explicit
    # null keeps captured-only cases intentional and prevents a malformed
    # runnable corpus that merely omitted ``input`` from being accepted.
    input: RunInputSpec | None
    assertions: tuple[AssertionSpec, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )

    @field_validator("assertions", mode="before")
    @classmethod
    def validate_assertions_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("id", "suite_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=2_048,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_contract(self) -> EvalCaseSpec:
        ids = tuple(assertion.id for assertion in self.assertions)
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
        if duplicates:
            raise ValueError(
                "Eval case assertion IDs must be unique; duplicated: " + ", ".join(duplicates)
            )
        expected = _model_content_revision(self, "eval case spec")
        if self.revision != expected:
            raise ValueError("Eval case revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        id: str,
        suite_id: str,
        name: str,
        source: EvaluationSourceIdentityV1,
        input: RunInputSpec | None,
        assertions: Sequence[AssertionSpec],
        description: str | None = None,
    ) -> EvalCaseSpec:
        if type(source) is not EvaluationSourceIdentityV1:
            raise TypeError("source must be an exact EvaluationSourceIdentityV1.")
        if input is not None and type(input) is not RunInputSpec:
            raise TypeError("input must be an exact RunInputSpec or None.")
        validated_source = EvaluationSourceIdentityV1.model_validate(_model_python_input(source))
        validated_input = (
            None if input is None else RunInputSpec.model_validate(_model_python_input(input))
        )
        ordered_assertions = _ordered_sequence_argument(assertions, "assertions")
        validated_assertions = tuple(_validated_assertion_spec(item) for item in ordered_assertions)
        document: dict[str, Any] = {
            "id": id,
            "suite_id": suite_id,
            "name": name,
            "description": description,
            "source": validated_source.model_dump(mode="json"),
            "input": (None if validated_input is None else validated_input.model_dump(mode="json")),
            "assertions": [assertion.model_dump(mode="json") for assertion in validated_assertions],
        }
        return cls(revision=_content_revision(document, "eval case spec"), **document)


class EvalCorpusDocument(_SchemaV2PortableModel):
    """One canonical, authority-free corpus for exactly one trusted target key."""

    schema_version: Literal[2] = EVAL_CORPUS_SCHEMA_VERSION
    revision: StrictStr
    target_key: StrictStr
    evidence_policy: EvaluationEvidencePolicySpec
    pricing_profile: PricingProfileIdentityV1 | None = None
    suites: tuple[EvalSuiteSpec, ...] = Field(min_length=1, max_length=EVAL_CORPUS_MAX_SUITES)
    cases: tuple[EvalCaseSpec, ...] = Field(min_length=1, max_length=EVAL_CORPUS_MAX_CASES)

    @field_validator("suites", "cases", mode="before")
    @classmethod
    def validate_collections_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> EvalCorpusDocument:
        suite_ids = tuple(suite.id for suite in self.suites)
        case_ids = tuple(case.id for case in self.cases)
        if suite_ids != tuple(sorted(suite_ids)):
            raise ValueError("Eval corpus suites must be sorted by id.")
        if case_ids != tuple(sorted(case_ids)):
            raise ValueError("Eval corpus cases must be sorted by id.")
        if len(suite_ids) != len(set(suite_ids)):
            raise ValueError("Eval corpus suite IDs must be unique.")
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Eval corpus case IDs must be unique.")
        known_suites = set(suite_ids)
        unknown_suites = sorted(
            {case.suite_id for case in self.cases if case.suite_id not in known_suites}
        )
        if unknown_suites:
            raise ValueError(
                "Eval cases reference unknown suites: " + ", ".join(unknown_suites) + "."
            )
        populated_suites = {case.suite_id for case in self.cases}
        empty_suites = sorted(known_suites - populated_suites)
        if empty_suites:
            raise ValueError(
                "Eval suites require at least one case: " + ", ".join(empty_suites) + "."
            )
        trials_by_suite = {suite.id: suite.trial_request.trials for suite in self.suites}
        published_results_by_suite: Counter[str] = Counter()
        published_judge_explanation_slots_by_suite: Counter[str] = Counter()
        for case in self.cases:
            published_results_by_suite[case.suite_id] += len(case.assertions)
            published_judge_explanation_slots_by_suite[case.suite_id] += sum(
                len(assertion.rubric.criteria)
                for assertion in case.assertions
                if type(assertion) is StructuredModelJudgeAssertionSpec
                and type(assertion.reference) is not PrivateJudgeReferenceV1
            )
        for suite_id, assertions_per_trial in published_results_by_suite.items():
            published_results = assertions_per_trial * trials_by_suite[suite_id]
            if published_results > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
                raise ValueError(
                    f"Eval suite {suite_id!r} expands to {published_results} published assertion "
                    "results; the maximum is "
                    f"{EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS}."
                )
            explanation_chars = (
                published_judge_explanation_slots_by_suite[suite_id]
                * trials_by_suite[suite_id]
                * EVAL_CORPUS_MAX_JUDGE_EXPLANATION_CHARS
            )
            if explanation_chars > EVAL_CORPUS_MAX_PUBLISHED_JUDGE_EXPLANATION_CHARS:
                raise ValueError(
                    f"Eval suite {suite_id!r} permits {explanation_chars} published judge "
                    "explanation characters; the maximum is "
                    f"{EVAL_CORPUS_MAX_PUBLISHED_JUDGE_EXPLANATION_CHARS}."
                )
        cost_currencies = {
            assertion.currency
            for case in self.cases
            for assertion in case.assertions
            if isinstance(assertion, MaxEstimatedCostAssertionSpec)
        }
        requires_tool_arguments = any(
            type(assertion) is ToolArgumentsContainAssertionSpec
            for case in self.cases
            for assertion in case.assertions
        )
        requires_tool_results = any(
            type(assertion) is ToolResultContainsAssertionSpec
            for case in self.cases
            for assertion in case.assertions
        )
        requires_artifact_text = any(
            type(assertion) is ArtifactAssertionSpec and assertion.text_contains is not None
            for case in self.cases
            for assertion in case.assertions
        )
        if requires_tool_arguments and not self.evidence_policy.include_tool_arguments:
            raise ValueError(
                "Tool-argument assertions require a target evidence policy that publishes "
                "tool arguments."
            )
        if requires_tool_results and not self.evidence_policy.include_tool_results:
            raise ValueError(
                "Tool-result assertions require a target evidence policy that explicitly "
                "retains public-safe tool results."
            )
        if requires_artifact_text and not self.evidence_policy.include_artifact_text:
            raise ValueError(
                "Artifact-text assertions require a target evidence policy that explicitly "
                "retains public-safe artifact text."
            )
        if cost_currencies and self.pricing_profile is None:
            raise ValueError("Cost assertions require a pricing profile identity.")
        if self.pricing_profile is not None:
            unsupported_currencies = sorted(cost_currencies - set(self.pricing_profile.currencies))
            if unsupported_currencies:
                raise ValueError(
                    "Cost assertion currencies are absent from the pricing profile: "
                    + ", ".join(unsupported_currencies)
                    + "."
                )
        if not json_utf8_size_within_limit(self, EVAL_CORPUS_MAX_BYTES):
            raise ValueError(f"Eval corpus exceeds {EVAL_CORPUS_MAX_BYTES} canonical JSON bytes.")
        if not _pretty_json_size_within_limit(self, EVAL_CORPUS_MAX_BYTES):
            raise ValueError(f"Eval corpus exceeds {EVAL_CORPUS_MAX_BYTES} serialized JSON bytes.")
        expected = _model_content_revision(self, "eval corpus")
        if self.revision != expected:
            raise ValueError("Eval corpus revision does not match its content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        target_key: str,
        evidence_policy: EvaluationEvidencePolicySpec,
        suites: Sequence[EvalSuiteSpec],
        cases: Sequence[EvalCaseSpec],
        pricing_profile: PricingProfileIdentityV1 | None = None,
    ) -> EvalCorpusDocument:
        if type(evidence_policy) is not EvaluationEvidencePolicySpec:
            raise TypeError("evidence_policy must be an exact EvaluationEvidencePolicySpec.")
        validated_policy = EvaluationEvidencePolicySpec.model_validate(
            _model_python_input(evidence_policy)
        )
        ordered_suite_input = _ordered_sequence_argument(suites, "suites")
        validated_suites: list[EvalSuiteSpec] = []
        for suite in ordered_suite_input:
            if type(suite) is not EvalSuiteSpec:
                raise TypeError("suites must contain exact EvalSuiteSpec instances.")
            validated_suites.append(EvalSuiteSpec.model_validate(_model_python_input(suite)))
        ordered_case_input = _ordered_sequence_argument(cases, "cases")
        validated_cases: list[EvalCaseSpec] = []
        for case in ordered_case_input:
            if type(case) is not EvalCaseSpec:
                raise TypeError("cases must contain exact EvalCaseSpec instances.")
            validated_cases.append(EvalCaseSpec.model_validate(_model_python_input(case)))
        if pricing_profile is None:
            validated_pricing = None
        elif type(pricing_profile) is PricingProfileIdentityV1:
            validated_pricing = PricingProfileIdentityV1.model_validate(
                _model_python_input(pricing_profile)
            )
        else:
            raise TypeError("pricing_profile must be an exact PricingProfileIdentityV1.")
        ordered_suites = tuple(sorted(validated_suites, key=lambda suite: suite.id))
        ordered_cases = tuple(sorted(validated_cases, key=lambda case: case.id))
        validated_target_key = _portable_id(target_key, "target_key")
        preflight_document = {
            "schema_version": EVAL_CORPUS_SCHEMA_VERSION,
            "revision": "sha256:" + "0" * 64,
            "target_key": validated_target_key,
            "evidence_policy": validated_policy,
            "pricing_profile": validated_pricing,
            "suites": ordered_suites,
            "cases": ordered_cases,
        }
        if not json_utf8_size_within_limit(preflight_document, EVAL_CORPUS_MAX_BYTES):
            raise ValueError(f"Eval corpus exceeds {EVAL_CORPUS_MAX_BYTES} canonical JSON bytes.")
        document: dict[str, Any] = {
            "schema_version": EVAL_CORPUS_SCHEMA_VERSION,
            "target_key": validated_target_key,
            "evidence_policy": validated_policy.model_dump(mode="json"),
            "pricing_profile": (
                None if validated_pricing is None else validated_pricing.model_dump(mode="json")
            ),
            "suites": [suite.model_dump(mode="json") for suite in ordered_suites],
            "cases": [case.model_dump(mode="json") for case in ordered_cases],
        }
        return cls(revision=_content_revision(document, "eval corpus"), **document)


def eval_run_contract_for_corpus(
    corpus: EvalCorpusDocument,
    suite_id: str,
) -> EvalRunContractV1:
    """Freeze the exact portable contract that must precede suite execution."""

    validated, _ = _validated_model_document(
        corpus,
        model_type=EvalCorpusDocument,
        field_name="eval corpus",
    )
    return _eval_run_contract_for_validated_corpus(validated, suite_id)


def _eval_run_contract_for_validated_corpus(
    corpus: EvalCorpusDocument,
    suite_id: str,
) -> EvalRunContractV1:
    """Build a run contract from an already validated exact corpus snapshot."""

    if type(corpus) is not EvalCorpusDocument:
        raise TypeError("corpus must be an exact EvalCorpusDocument.")
    validated_suite_id = _portable_id(suite_id, "suite_id")
    suite = next((item for item in corpus.suites if item.id == validated_suite_id), None)
    if suite is None:
        raise ValueError(f"Eval corpus does not contain suite {validated_suite_id!r}.")
    cases = tuple(case for case in corpus.cases if case.suite_id == suite.id)
    uses_pricing = any(
        assertion.kind == "max_estimated_cost" for case in cases for assertion in case.assertions
    )
    return EvalRunContractV1(
        corpus_revision=corpus.revision,
        target_key=corpus.target_key,
        suite_id=suite.id,
        suite_revision=suite.revision,
        evidence_policy_revision=corpus.evidence_policy.revision,
        pricing_profile_fingerprint=(
            corpus.pricing_profile.fingerprint
            if uses_pricing and corpus.pricing_profile is not None
            else None
        ),
        trials=suite.trial_request.trials,
        timeout_seconds=suite.trial_request.timeout_seconds,
        cases=tuple(
            EvalCaseContractV1(case_id=case.id, case_revision=case.revision) for case in cases
        ),
    )


def eval_corpus_to_json(corpus: EvalCorpusDocument) -> str:
    """Return deterministic, human-readable corpus v2 JSON."""

    _, document = _validated_model_document(
        corpus,
        model_type=EvalCorpusDocument,
        field_name="eval corpus",
    )
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2, sort_keys=True)
    chunks: list[str] = []
    total_bytes = 1  # final newline
    for chunk in encoder.iterencode(document):
        total_bytes += len(chunk.encode("utf-8"))
        if total_bytes > EVAL_CORPUS_MAX_BYTES:
            raise ValueError(f"Eval corpus JSON exceeds {EVAL_CORPUS_MAX_BYTES} bytes.")
        chunks.append(chunk)
    return "".join(chunks) + "\n"


def eval_corpus_from_json(source: str) -> EvalCorpusDocument:
    """Load one bounded corpus v2 JSON document from text."""

    if type(source) is not str:
        raise TypeError("eval_corpus_from_json requires text.")
    # UTF-8 uses at least one byte for every scalar character. Reject an
    # arbitrarily oversized Python string before allocating its encoded copy;
    # the subsequent byte check handles bounded multibyte input exactly.
    if len(source) > EVAL_CORPUS_MAX_BYTES:
        raise ValueError(f"Eval corpus JSON exceeds {EVAL_CORPUS_MAX_BYTES} bytes.")
    try:
        raw = source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Eval corpus JSON must contain valid Unicode scalar text.") from exc
    if len(raw) > EVAL_CORPUS_MAX_BYTES:
        raise ValueError(f"Eval corpus JSON exceeds {EVAL_CORPUS_MAX_BYTES} bytes.")
    try:
        decoded = json.loads(
            source,
            parse_int=partial(
                parse_durable_json_integer_literal,
                field_name="eval corpus JSON",
            ),
            parse_constant=partial(
                reject_nonportable_json_constant,
                field_name="eval corpus JSON",
            ),
            object_pairs_hook=partial(
                durable_json_object_from_pairs,
                field_name="eval corpus JSON",
            ),
        )
    except RecursionError as exc:
        raise ValueError("Eval corpus JSON nesting exceeds the supported depth.") from exc
    document = copy_durable_json_object(decoded, "eval corpus JSON")
    raw_version = document.get("schema_version")
    if type(raw_version) is not int or raw_version != EVAL_CORPUS_SCHEMA_VERSION:
        raise ValueError(
            f"Eval corpus has unsupported schema_version {raw_version!r}; this Cayu version "
            f"supports only {EVAL_CORPUS_SCHEMA_VERSION}."
        )
    return EvalCorpusDocument.model_validate(document)


def load_eval_corpus(path: str | Path) -> EvalCorpusDocument:
    """Read at most the corpus hard limit before decoding or validating JSON."""

    resolved = Path(path)
    with resolved.open("rb") as handle:
        raw = handle.read(EVAL_CORPUS_MAX_BYTES + 1)
    if len(raw) > EVAL_CORPUS_MAX_BYTES:
        raise ValueError(f"Eval corpus JSON exceeds {EVAL_CORPUS_MAX_BYTES} bytes.")
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Eval corpus JSON must be UTF-8.") from exc
    return eval_corpus_from_json(source)


class EvalCorpusSuiteInspectionV1(_SchemaV1PortableModel):
    """Bounded structural summary for one corpus suite."""

    id: StrictStr
    revision: StrictStr
    name: StrictStr
    case_count: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_CASES)
    assertion_count: StrictInt = Field(ge=1)
    trials: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    timeout_seconds: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_counts(self) -> EvalCorpusSuiteInspectionV1:
        if (
            not self.case_count
            <= self.assertion_count
            <= (self.case_count * EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE)
        ):
            raise ValueError("Corpus suite inspection assertion_count is impossible.")
        if self.assertion_count * self.trials > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
            raise ValueError("Corpus suite inspection expanded result count is impossible.")
        return self


class EvalCorpusInspectionV1(_SchemaV1PortableModel):
    """Stable authority-free summary returned by corpus validation and inspection."""

    revision: StrictStr
    target_key: StrictStr
    evidence_policy_revision: StrictStr
    pricing_profile_fingerprint: StrictStr | None = None
    suite_count: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_SUITES)
    case_count: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_CASES)
    assertion_count: StrictInt = Field(ge=1)
    expanded_assertion_result_count: StrictInt = Field(
        ge=1,
        le=(EVAL_CORPUS_MAX_SUITES * EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS),
    )
    suites: tuple[EvalCorpusSuiteInspectionV1, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_SUITES,
    )

    @field_validator(
        "revision",
        "evidence_policy_revision",
        "pricing_profile_fingerprint",
    )
    @classmethod
    def validate_revision_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("suites", mode="before")
    @classmethod
    def validate_suites_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_summary(self) -> EvalCorpusInspectionV1:
        suite_ids = tuple(item.id for item in self.suites)
        if suite_ids != tuple(sorted(set(suite_ids))):
            raise ValueError("Corpus inspection suites must be unique and sorted by id.")
        if self.suite_count != len(self.suites):
            raise ValueError("Corpus inspection suite_count does not match suites.")
        if self.case_count != sum(item.case_count for item in self.suites):
            raise ValueError("Corpus inspection case_count does not match suites.")
        if self.assertion_count != sum(item.assertion_count for item in self.suites):
            raise ValueError("Corpus inspection assertion_count does not match suites.")
        expected_expanded = sum(item.assertion_count * item.trials for item in self.suites)
        if self.expanded_assertion_result_count != expected_expanded:
            raise ValueError(
                "Corpus inspection expanded_assertion_result_count does not match suites."
            )
        return self


def inspect_eval_corpus(corpus: EvalCorpusDocument) -> EvalCorpusInspectionV1:
    """Return a defensive bounded structural summary of one corpus."""

    validated, _ = _validated_model_document(
        corpus,
        model_type=EvalCorpusDocument,
        field_name="eval corpus",
    )
    cases_by_suite: dict[str, list[EvalCaseSpec]] = {suite.id: [] for suite in validated.suites}
    for case in validated.cases:
        cases_by_suite[case.suite_id].append(case)
    suites = tuple(
        EvalCorpusSuiteInspectionV1(
            id=suite.id,
            revision=suite.revision,
            name=suite.name,
            case_count=len(cases_by_suite[suite.id]),
            assertion_count=sum(len(case.assertions) for case in cases_by_suite[suite.id]),
            trials=suite.trial_request.trials,
            timeout_seconds=suite.trial_request.timeout_seconds,
        )
        for suite in validated.suites
    )
    return EvalCorpusInspectionV1(
        revision=validated.revision,
        target_key=validated.target_key,
        evidence_policy_revision=validated.evidence_policy.revision,
        pricing_profile_fingerprint=(
            None if validated.pricing_profile is None else validated.pricing_profile.fingerprint
        ),
        suite_count=len(suites),
        case_count=len(validated.cases),
        assertion_count=sum(item.assertion_count for item in suites),
        expanded_assertion_result_count=sum(item.assertion_count * item.trials for item in suites),
        suites=suites,
    )


def eval_corpus_inspection_to_json(inspection: EvalCorpusInspectionV1) -> str:
    """Return deterministic JSON for one defensive corpus inspection."""

    if type(inspection) is not EvalCorpusInspectionV1:
        raise TypeError("inspection must be an exact EvalCorpusInspectionV1.")
    validated = EvalCorpusInspectionV1.model_validate(_model_python_input(inspection))
    return (
        json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def merge_eval_corpora(
    corpora: list[EvalCorpusDocument] | tuple[EvalCorpusDocument, ...],
    *,
    replace_conflicts: bool = False,
) -> EvalCorpusDocument:
    """Merge compatible corpora by immutable ID/revision contracts."""

    if type(replace_conflicts) is not bool:
        raise TypeError("replace_conflicts must be a bool.")
    if type(corpora) not in (list, tuple):
        raise TypeError("corpora must be an ordered sequence (a list or tuple).")
    if not corpora:
        raise ValueError("corpora must contain at least one document.")
    if len(corpora) > EVAL_CORPUS_MAX_MERGE_INPUTS:
        raise ValueError(
            f"corpora cannot contain more than {EVAL_CORPUS_MAX_MERGE_INPUTS} documents."
        )
    return _merge_eval_corpus_documents(
        corpora,
        replace_conflicts=replace_conflicts,
    )


_MergeItemT = TypeVar("_MergeItemT", EvalSuiteSpec, EvalCaseSpec)


def _merge_eval_corpus_documents(
    corpora: Iterable[EvalCorpusDocument],
    *,
    replace_conflicts: bool,
) -> EvalCorpusDocument:
    """Merge a bounded caller-validated stream without retaining every source."""

    first: EvalCorpusDocument | None = None
    suites: dict[str, EvalSuiteSpec] = {}
    cases: dict[str, EvalCaseSpec] = {}
    suite_sizes: dict[str, int] = {}
    case_sizes: dict[str, int] = {}
    retained_bytes = 0

    def merge_item(
        kind: str,
        key: str,
        value: _MergeItemT,
        destination: dict[str, _MergeItemT],
        sizes: dict[str, int],
        max_items: int,
    ) -> None:
        nonlocal retained_bytes
        existing = destination.get(key)
        if existing == value:
            return
        if existing is not None and not replace_conflicts:
            raise ValueError(
                f"Eval corpus {kind} {key!r} has conflicting content revisions "
                f"{existing.revision!r} and {value.revision!r}."
            )
        if existing is None and len(destination) >= max_items:
            raise ValueError(f"Merged eval corpus cannot contain more than {max_items} {kind}s.")

        value_size = compact_json_utf8_size(value.model_dump(mode="json"))
        if existing is None:
            # Empty and one-item arrays both contribute their two brackets; only
            # the item, plus a comma after the first item, grows the document.
            size_delta = value_size + (1 if destination else 0)
        else:
            size_delta = value_size - sizes[key]
        if retained_bytes + size_delta > EVAL_CORPUS_MAX_BYTES:
            raise ValueError(
                f"Merged eval corpus exceeds {EVAL_CORPUS_MAX_BYTES} canonical JSON bytes."
            )
        destination[key] = value
        sizes[key] = value_size
        retained_bytes += size_delta

    for corpus in corpora:
        if type(corpus) is not EvalCorpusDocument:
            raise TypeError("corpora must contain exact EvalCorpusDocument instances.")
        item, _ = _validated_model_document(
            corpus,
            model_type=EvalCorpusDocument,
            field_name="eval corpus",
        )
        if first is None:
            first = item
            retained_bytes = compact_json_utf8_size(
                {
                    "schema_version": EVAL_CORPUS_SCHEMA_VERSION,
                    "revision": "sha256:" + "0" * 64,
                    "target_key": first.target_key,
                    "evidence_policy": first.evidence_policy.model_dump(mode="json"),
                    "pricing_profile": (
                        None
                        if first.pricing_profile is None
                        else first.pricing_profile.model_dump(mode="json")
                    ),
                    "suites": [],
                    "cases": [],
                }
            )
        else:
            if item.target_key != first.target_key:
                raise ValueError("Cannot merge eval corpora with different target keys.")
            if item.evidence_policy != first.evidence_policy:
                raise ValueError("Cannot merge eval corpora with different evidence policies.")
            if item.pricing_profile != first.pricing_profile:
                raise ValueError("Cannot merge eval corpora with different pricing profiles.")
        for suite in item.suites:
            merge_item(
                "suite",
                suite.id,
                suite,
                suites,
                suite_sizes,
                EVAL_CORPUS_MAX_SUITES,
            )
        for case in item.cases:
            merge_item(
                "case",
                case.id,
                case,
                cases,
                case_sizes,
                EVAL_CORPUS_MAX_CASES,
            )
    if first is None:  # guarded by the non-empty check above
        raise RuntimeError("Eval corpus merge lost its first input.")
    return EvalCorpusDocument.create(
        target_key=first.target_key,
        evidence_policy=first.evidence_policy,
        pricing_profile=first.pricing_profile,
        suites=tuple(suites.values()),
        cases=tuple(cases.values()),
    )


def merge_eval_corpus_files(
    destination: str | Path,
    inputs: list[str | Path] | tuple[str | Path, ...],
    *,
    replace_conflicts: bool = False,
) -> EvalCorpusDocument:
    """Validate, merge, and atomically replace one corpus destination."""

    if type(replace_conflicts) is not bool:
        raise TypeError("replace_conflicts must be a bool.")
    if type(inputs) not in (list, tuple):
        raise TypeError("inputs must be an ordered sequence of paths.")
    if not inputs:
        raise ValueError("inputs must contain at least one path.")
    if len(inputs) > EVAL_CORPUS_MAX_MERGE_INPUTS:
        raise ValueError(f"inputs cannot contain more than {EVAL_CORPUS_MAX_MERGE_INPUTS} paths.")
    ordered_inputs = tuple(inputs)
    destination_path = Path(destination)
    if destination_path.is_symlink():
        raise ValueError("Eval corpus destination cannot be a symbolic link.")
    if destination_path.exists() and not destination_path.is_file():
        raise ValueError("Eval corpus destination must be a file path.")
    destination_exists = destination_path.exists()
    source_count = len(ordered_inputs) + int(destination_exists)
    if source_count > EVAL_CORPUS_MAX_MERGE_INPUTS:
        raise ValueError(
            "Existing destination and inputs cannot total more than "
            f"{EVAL_CORPUS_MAX_MERGE_INPUTS} corpus documents."
        )
    source_paths = ((destination_path,) if destination_exists else ()) + ordered_inputs
    merged = _merge_eval_corpus_documents(
        (load_eval_corpus(path) for path in source_paths),
        replace_conflicts=replace_conflicts,
    )
    _atomic_write_corpus(destination_path, eval_corpus_to_json(merged))
    return merged


def _atomic_write_corpus(destination: Path, content: str) -> None:
    parent = destination.parent
    if not parent.is_dir():
        raise ValueError("Eval corpus destination parent must be an existing directory.")
    existing_mode = None
    if destination.exists():
        existing_mode = stat.S_IMODE(destination.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        if existing_mode is not None:
            os.fchmod(descriptor, existing_mode)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = None
        try:
            # The file itself is durable before publication. Directory fsync is
            # a best-effort crash-consistency strengthening because some valid
            # filesystems reject it after the atomic replace has already
            # succeeded; surfacing that late error would falsely report a
            # failed merge even though the destination changed.
            with suppress(OSError):
                directory_descriptor = os.open(parent, os.O_RDONLY)
            if directory_descriptor is not None:
                with suppress(OSError):
                    os.fsync(directory_descriptor)
        finally:
            if directory_descriptor is not None:
                with suppress(OSError):
                    os.close(directory_descriptor)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
