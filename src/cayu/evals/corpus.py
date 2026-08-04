from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    durable_json_object_from_pairs,
    json_utf8_size_within_limit,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_durable_text,
)

EVAL_CORPUS_SCHEMA_VERSION = 1
EVALUATION_EVIDENCE_POLICY_SCHEMA_VERSION = 1
PRICING_PROFILE_IDENTITY_SCHEMA_VERSION = 1
EVALUATION_SOURCE_IDENTITY_SCHEMA_VERSION = 1

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
EVAL_CORPUS_MAX_TOOL_NAMES = 256
EVAL_CORPUS_MAX_TRIALS = 100
EVAL_CORPUS_MAX_TIMEOUT_SECONDS = 3_600

EVIDENCE_MAX_FINAL_OUTPUT_CHARS = 65_536
EVIDENCE_MAX_CHILD_SESSIONS = 500
EVIDENCE_MAX_TOOL_CALLS = 4_096
EVIDENCE_MAX_MODEL_STEPS = 4_096
EVIDENCE_MAX_TOTAL_TOKENS = 2**63 - 1

_PORTABLE_ID_PATTERN = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z", re.ASCII)
_SHA256_REVISION_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)
_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CURRENCY_PATTERN = re.compile(r"[A-Z][A-Z0-9._-]{0,15}\Z", re.ASCII)
_CANONICAL_NONNEGATIVE_DECIMAL_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z",
    re.ASCII,
)

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
        min_length=1,
        max_length=EVAL_CORPUS_MAX_MESSAGES_PER_CASE,
    )

    @field_validator("messages", mode="before")
    @classmethod
    def validate_messages_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_total_text(self) -> RunInputSpec:
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
    expected: Literal["completed", "failed"]
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


AssertionSpec: TypeAlias = Annotated[
    RootStatusAssertionSpec
    | ChildStatusAssertionSpec
    | FinalOutputEqualsAssertionSpec
    | FinalOutputContainsAssertionSpec
    | ToolCalledAssertionSpec
    | ToolsCalledInOrderAssertionSpec
    | MaxToolCallsAssertionSpec
    | MaxModelStepsAssertionSpec
    | UsageRecordedAssertionSpec
    | MaxTotalTokensAssertionSpec
    | MaxEstimatedCostAssertionSpec,
    Field(discriminator="kind"),
]

_ASSERTION_SPEC_TYPES = (
    RootStatusAssertionSpec,
    ChildStatusAssertionSpec,
    FinalOutputEqualsAssertionSpec,
    FinalOutputContainsAssertionSpec,
    ToolCalledAssertionSpec,
    ToolsCalledInOrderAssertionSpec,
    MaxToolCallsAssertionSpec,
    MaxModelStepsAssertionSpec,
    UsageRecordedAssertionSpec,
    MaxTotalTokensAssertionSpec,
    MaxEstimatedCostAssertionSpec,
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
    """Diagnostic capture provenance without runtime or session authority."""

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
    fingerprint: StrictStr
    price_book_version: StrictStr
    generated_at: StrictStr
    currencies: tuple[StrictStr, ...] = Field(min_length=1, max_length=32)

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


class EvaluationEvidencePolicySpec(_SchemaV1PortableModel):
    """Complete v1 omission, redaction, and cardinality behavior."""

    schema_version: Literal[1] = EVALUATION_EVIDENCE_POLICY_SCHEMA_VERSION
    revision: StrictStr
    event_projection: Literal["runtime_public_alias_free_v1"] = "runtime_public_alias_free_v1"
    include_event_payloads: Literal[False] = False
    include_transcript_text: Literal[False] = False
    include_tool_arguments: Literal[False] = False
    include_tool_results: Literal[False] = False
    max_final_output_chars: Literal[65536] = EVIDENCE_MAX_FINAL_OUTPUT_CHARS
    max_child_sessions: Literal[500] = EVIDENCE_MAX_CHILD_SESSIONS
    max_tool_calls: Literal[4096] = EVIDENCE_MAX_TOOL_CALLS
    max_model_steps: Literal[4096] = EVIDENCE_MAX_MODEL_STEPS
    max_total_tokens: Literal[9223372036854775807] = 9223372036854775807

    @field_validator(
        "include_event_payloads",
        "include_transcript_text",
        "include_tool_arguments",
        "include_tool_results",
        mode="before",
    )
    @classmethod
    def validate_omission_flag_types(cls, value: object, info) -> object:
        if type(value) is not bool:
            raise ValueError(f"{info.field_name} must be false.")
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
    def standard(cls) -> EvaluationEvidencePolicySpec:
        document = {
            "schema_version": EVALUATION_EVIDENCE_POLICY_SCHEMA_VERSION,
            "event_projection": "runtime_public_alias_free_v1",
            "include_event_payloads": False,
            "include_transcript_text": False,
            "include_tool_arguments": False,
            "include_tool_results": False,
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
    """One portable request and deterministic expectation set."""

    id: StrictStr
    revision: StrictStr
    suite_id: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    source: EvaluationSourceIdentityV1
    input: RunInputSpec
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
        input: RunInputSpec,
        assertions: Sequence[AssertionSpec],
        description: str | None = None,
    ) -> EvalCaseSpec:
        if type(source) is not EvaluationSourceIdentityV1:
            raise TypeError("source must be an exact EvaluationSourceIdentityV1.")
        if type(input) is not RunInputSpec:
            raise TypeError("input must be an exact RunInputSpec.")
        validated_source = EvaluationSourceIdentityV1.model_validate(_model_python_input(source))
        validated_input = RunInputSpec.model_validate(_model_python_input(input))
        ordered_assertions = _ordered_sequence_argument(assertions, "assertions")
        validated_assertions = tuple(_validated_assertion_spec(item) for item in ordered_assertions)
        document: dict[str, Any] = {
            "id": id,
            "suite_id": suite_id,
            "name": name,
            "description": description,
            "source": validated_source.model_dump(mode="json"),
            "input": validated_input.model_dump(mode="json"),
            "assertions": [assertion.model_dump(mode="json") for assertion in validated_assertions],
        }
        return cls(revision=_content_revision(document, "eval case spec"), **document)


class EvalCorpusDocument(_SchemaV1PortableModel):
    """One canonical, authority-free corpus for exactly one trusted target key."""

    schema_version: Literal[1] = EVAL_CORPUS_SCHEMA_VERSION
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
        trials_by_suite = {suite.id: suite.trial_request.trials for suite in self.suites}
        published_results_by_suite: Counter[str] = Counter()
        for case in self.cases:
            published_results_by_suite[case.suite_id] += len(case.assertions)
        for suite_id, assertions_per_trial in published_results_by_suite.items():
            published_results = assertions_per_trial * trials_by_suite[suite_id]
            if published_results > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
                raise ValueError(
                    f"Eval suite {suite_id!r} expands to {published_results} published assertion "
                    "results; the maximum is "
                    f"{EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS}."
                )
        cost_currencies = {
            assertion.currency
            for case in self.cases
            for assertion in case.assertions
            if isinstance(assertion, MaxEstimatedCostAssertionSpec)
        }
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


def eval_corpus_to_json(corpus: EvalCorpusDocument) -> str:
    """Return deterministic, human-readable corpus v1 JSON."""

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
    """Load one bounded corpus v1 JSON document from text."""

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
        raise ValueError("Eval corpus JSON exceeds the supported nesting depth.") from exc
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
