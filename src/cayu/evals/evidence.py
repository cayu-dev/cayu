from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, cast

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from cayu._validation import (
    DurableValueError,
    canonical_durable_json_bytes,
    json_utf8_size_within_limit,
)
from cayu.core.events import Event, EventType
from cayu.core.messages import ToolCallPart
from cayu.core.tools import ToolResult
from cayu.evals._memory_attribution import (
    eval_memory_attribution_evidence_from_trajectory,
)
from cayu.evals.corpus import (
    _CURRENCY_PATTERN,
    EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS,
    EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS,
    EVIDENCE_MAX_CHILD_SESSIONS,
    EVIDENCE_MAX_FINAL_OUTPUT_CHARS,
    EVIDENCE_MAX_MODEL_STEPS,
    EVIDENCE_MAX_TOOL_CALLS,
    EVIDENCE_MAX_TOTAL_TOKENS,
    EvalProcessEventKind,
    EvaluationEvidencePolicySpec,
    PricingProfileIdentityV1,
    _bounded_durable_text,
    _canonical_decimal_text,
    _content_revision,
    _model_content_revision,
    _ordered_sequence_input,
    _PortableModel,
    _pricing_profile_identity_from_validated_price_book,
    _sha256_revision,
)
from cayu.evals.json_subset import copy_eval_tool_json_object
from cayu.evals.memory_attribution import (
    EVAL_MEMORY_ATTRIBUTION_MAX_BYTES,
    EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES,
    EvalMemoryAttributionEvidenceV1,
)
from cayu.evals.models import (
    Trajectory,
    _model_instance_python_input,
    _validate_trajectory_record_contract,
)
from cayu.runtime._memory_evidence import memory_evidence_key
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import PriceBook, _estimate_session_cost
from cayu.runtime.usage import AggregateCount

ASSERTION_EVIDENCE_SCHEMA_VERSION = 4
ASSERTION_EVIDENCE_MAX_BYTES = 10 << 20
ASSERTION_EVIDENCE_MAX_TOOL_NAME_CHARS = 256
ASSERTION_EVIDENCE_MAX_COST_CURRENCIES = 32

EvidenceState = Literal["complete", "unavailable", "limit_exceeded"]
ToolCallEvidenceState = Literal["complete", "unavailable", "limit_exceeded", "incompatible"]
ToolValueEvidenceState = Literal[
    "available",
    "unavailable",
    "unsupported",
    "malformed",
    "truncated",
    "incompatible",
]
TerminalEvidenceStatus = Literal["completed", "failed", "interrupted"]


_PORTABLE_PROCESS_EVENT_KINDS: dict[str, EvalProcessEventKind] = {
    str(EventType.SESSION_STARTED): "session_started",
    str(EventType.SESSION_RESUMED): "session_resumed",
    str(EventType.SESSION_AWAITING_USER_INPUT): "session_awaiting_user_input",
    str(EventType.SESSION_COMPLETED): "session_completed",
    str(EventType.SESSION_FAILED): "session_failed",
    str(EventType.SESSION_INTERRUPTED): "session_interrupted",
    str(EventType.SESSION_LIMIT_REACHED): "session_limit_reached",
    str(EventType.TOOL_CALL_STARTED): "tool_call_started",
    str(EventType.TOOL_CALL_COMPLETED): "tool_call_completed",
    str(EventType.TOOL_CALL_FAILED): "tool_call_failed",
    str(EventType.TOOL_CALL_BLOCKED): "tool_call_blocked",
    str(EventType.TOOL_CALL_APPROVAL_REQUESTED): "tool_approval_requested",
    str(EventType.TOOL_CALL_APPROVED): "tool_approved",
    str(EventType.TOOL_CALL_APPROVAL_DENIED): "tool_approval_denied",
    str(EventType.TOOL_CALL_APPROVAL_EXPIRED): "tool_approval_expired",
    str(EventType.STRUCTURED_OUTPUT_VALIDATED): "structured_output_validated",
    str(EventType.STRUCTURED_OUTPUT_FAILED): "structured_output_failed",
    str(EventType.BUDGET_LIMIT_REACHED): "budget_limit_reached",
}


@dataclass(frozen=True, slots=True)
class _ValidatedPricingSnapshot:
    """One detached pricing snapshot and its canonical behavior identity."""

    price_book: PriceBook
    identity: PricingProfileIdentityV1


def _canonical_decimal(value: Decimal, *, max_chars: int = 128) -> str:
    if not value.is_finite() or value < 0:
        raise ValueError("Evidence cost must be a finite non-negative decimal.")
    _, raw_digits, raw_exponent = value.as_tuple()
    if not isinstance(raw_exponent, int):
        raise ValueError("Evidence cost must be a finite non-negative decimal.")
    trailing_zeros = 0
    for digit in reversed(raw_digits):
        if digit != 0:
            break
        trailing_zeros += 1
    digit_count = len(raw_digits) - trailing_zeros
    if digit_count == 0:
        return "0"
    exponent = raw_exponent + trailing_zeros
    point_position = digit_count + exponent
    if exponent >= 0:
        expanded_chars = point_position
    elif point_position > 0:
        expanded_chars = digit_count + 1
    else:
        expanded_chars = 2 - point_position + digit_count
    if expanded_chars > max_chars:
        raise ValueError(f"Evidence cost must be at most {max_chars} canonical decimal characters.")
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return "0" if result in {"", "-0"} else result


class AssertionCostEvidenceV1(_PortableModel):
    """One currency-local aggregate with no provider, model, or line-item identity."""

    currency: StrictStr
    total_cost: StrictStr
    model_steps: StrictInt = Field(ge=0, le=EVIDENCE_MAX_MODEL_STEPS)
    priced_model_steps: StrictInt = Field(ge=0, le=EVIDENCE_MAX_MODEL_STEPS)
    unpriced_model_steps: StrictInt = Field(ge=0, le=EVIDENCE_MAX_MODEL_STEPS)

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
            raise ValueError("currency must be a portable uppercase identifier.")
        return value

    @field_validator("total_cost")
    @classmethod
    def validate_total_cost(cls, value: str, info) -> str:
        return _canonical_decimal_text(value, info.field_name, max_chars=128)

    @model_validator(mode="after")
    def validate_step_partition(self) -> AssertionCostEvidenceV1:
        if self.priced_model_steps + self.unpriced_model_steps != self.model_steps:
            raise ValueError("Cost evidence step counts must form an exact partition.")
        return self


class ToolCallValueEvidenceV1(_PortableModel):
    """One bounded public value or an exact reason no comparison is possible."""

    state: ToolValueEvidenceState
    value: dict[str, Any] | None = None

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: object, info) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_eval_tool_json_object(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> ToolCallValueEvidenceV1:
        if (self.state == "available") != (self.value is not None):
            raise ValueError("Available tool-call value evidence requires exactly one value.")
        return self


class ToolCallEvidenceV1(_PortableModel):
    """Stable ordered identity plus bounded argument and result observations."""

    invocation_index: StrictInt = Field(ge=1, le=EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS)
    invocation_revision: StrictStr
    tool_name: StrictStr
    occurrence: StrictInt = Field(ge=1, le=EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS)
    arguments: ToolCallValueEvidenceV1
    result: ToolCallValueEvidenceV1

    @field_validator("invocation_revision")
    @classmethod
    def validate_invocation_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=ASSERTION_EVIDENCE_MAX_TOOL_NAME_CHARS,
            nonblank=True,
            clean=True,
        )


class AssertionEvidenceView(_PortableModel):
    """The bounded, content-minimized data consumed by every portable assertion."""

    schema_version: Literal[4] = ASSERTION_EVIDENCE_SCHEMA_VERSION
    revision: StrictStr
    policy_revision: StrictStr
    pricing_profile_fingerprint: StrictStr | None = None
    root_evidence_available: StrictBool
    root_status: TerminalEvidenceStatus | None = None
    child_statuses: tuple[TerminalEvidenceStatus, ...] = Field(
        max_length=EVIDENCE_MAX_CHILD_SESSIONS
    )
    child_evidence_state: EvidenceState
    final_output: StrictStr
    final_output_state: EvidenceState
    requested_tool_names: tuple[StrictStr, ...] = Field(max_length=EVIDENCE_MAX_TOOL_CALLS)
    started_tool_names: tuple[StrictStr, ...] = Field(max_length=EVIDENCE_MAX_TOOL_CALLS)
    tool_calls_started: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVIDENCE_MAX_TOTAL_TOKENS,
    )
    tool_evidence_state: EvidenceState
    tool_calls: tuple[ToolCallEvidenceV1, ...] = Field(max_length=EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS)
    tool_call_evidence_state: ToolCallEvidenceState
    process_events: tuple[EvalProcessEventKind, ...] = Field(
        max_length=EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS
    )
    process_event_evidence_state: EvidenceState
    model_steps: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVIDENCE_MAX_TOTAL_TOKENS,
    )
    model_step_evidence_state: EvidenceState
    total_tokens: AggregateCount | None = Field(default=None, ge=0)
    usage_evidence_state: EvidenceState
    costs: tuple[AssertionCostEvidenceV1, ...] = Field(
        max_length=ASSERTION_EVIDENCE_MAX_COST_CURRENCIES
    )
    memory_attribution: EvalMemoryAttributionEvidenceV1

    @field_validator(
        "child_statuses",
        "requested_tool_names",
        "started_tool_names",
        "tool_calls",
        "process_events",
        "costs",
        mode="before",
    )
    @classmethod
    def validate_collections_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("revision", "policy_revision")
    @classmethod
    def validate_revision_shape(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 4.")
        return value

    @field_validator("pricing_profile_fingerprint")
    @classmethod
    def validate_pricing_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @field_validator("final_output")
    @classmethod
    def validate_final_output(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVIDENCE_MAX_FINAL_OUTPUT_CHARS,
            nonblank=False,
            clean=False,
        )

    @field_validator("requested_tool_names", "started_tool_names")
    @classmethod
    def validate_tool_names(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return tuple(
            _bounded_durable_text(
                item,
                f"{info.field_name}[{index}]",
                max_chars=ASSERTION_EVIDENCE_MAX_TOOL_NAME_CHARS,
                nonblank=True,
                clean=True,
            )
            for index, item in enumerate(value)
        )

    @model_validator(mode="after")
    def validate_contract(self) -> AssertionEvidenceView:
        try:
            evidence_policy = EvaluationEvidencePolicySpec.policy_for_revision(self.policy_revision)
        except ValueError as exc:
            raise ValueError("Assertion evidence policy revision is not supported.") from exc
        if not self.root_evidence_available:
            if self.root_status is not None:
                raise ValueError("Unavailable root evidence cannot carry a root status.")
            derived_states = (
                self.child_evidence_state,
                self.final_output_state,
                self.tool_evidence_state,
                self.tool_call_evidence_state,
                self.process_event_evidence_state,
                self.model_step_evidence_state,
                self.usage_evidence_state,
            )
            if any(state != "unavailable" for state in derived_states) or self.costs:
                raise ValueError(
                    "Unavailable root evidence cannot carry conclusive derived evidence."
                )
        if self.child_evidence_state == "unavailable" and self.child_statuses:
            raise ValueError("Unavailable child evidence cannot carry observations.")
        if (
            self.child_evidence_state == "limit_exceeded"
            and len(self.child_statuses) != EVIDENCE_MAX_CHILD_SESSIONS
        ):
            raise ValueError("Limited child evidence must retain exactly its bounded prefix.")
        if self.final_output_state == "unavailable":
            if self.final_output:
                raise ValueError("Unavailable final-output evidence cannot carry text.")
        elif (
            self.final_output_state == "limit_exceeded"
            and len(self.final_output) != EVIDENCE_MAX_FINAL_OUTPUT_CHARS
        ):
            raise ValueError(
                "Limited final-output evidence must retain exactly its bounded prefix."
            )
        if self.tool_evidence_state == "unavailable":
            if (
                self.requested_tool_names
                or self.started_tool_names
                or self.tool_calls_started is not None
            ):
                raise ValueError("Unavailable tool evidence cannot carry observations.")
        elif self.tool_calls_started is None:
            raise ValueError("Available tool evidence requires the exact started-call count.")
        elif self.tool_calls_started < len(self.started_tool_names):
            raise ValueError("Started tool names cannot exceed the exact started-call count.")
        elif self.tool_evidence_state == "complete" and self.tool_calls_started != len(
            self.started_tool_names
        ):
            raise ValueError("Complete tool evidence requires every started tool name.")
        elif self.tool_evidence_state == "complete" and (
            Counter(self.started_tool_names) - Counter(self.requested_tool_names)
        ):
            raise ValueError(
                "Complete started-tool evidence must originate from requested tool calls."
            )
        elif self.tool_evidence_state == "limit_exceeded" and not (
            len(self.requested_tool_names) == EVIDENCE_MAX_TOOL_CALLS
            or len(self.started_tool_names) == EVIDENCE_MAX_TOOL_CALLS
            or self.tool_calls_started > EVIDENCE_MAX_TOOL_CALLS
        ):
            raise ValueError("Limited tool evidence must reach a declared tool bound.")
        if self.tool_call_evidence_state == "unavailable":
            if self.tool_calls:
                raise ValueError("Unavailable tool-call evidence cannot carry observations.")
        elif self.tool_call_evidence_state == "incompatible":
            if self.tool_calls:
                raise ValueError("Incompatible tool-call evidence cannot carry observations.")
        elif self.tool_evidence_state == "unavailable":
            raise ValueError("Tool-call evidence requires available tool-name evidence.")
        elif self.tool_call_evidence_state == "complete":
            if self.tool_calls_started != len(self.tool_calls):
                raise ValueError("Complete tool-call evidence requires every started call.")
        elif self.tool_call_evidence_state == "limit_exceeded" and (
            len(self.tool_calls) != EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS
            or self.tool_calls_started is None
            or self.tool_calls_started <= EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS
        ):
            raise ValueError("Limited tool-call evidence must retain its complete bounded prefix.")
        expected_indexes = tuple(range(1, len(self.tool_calls) + 1))
        if tuple(item.invocation_index for item in self.tool_calls) != expected_indexes:
            raise ValueError("Tool-call evidence must retain canonical invocation order.")
        occurrences: Counter[str] = Counter()
        for item in self.tool_calls:
            occurrences[item.tool_name] += 1
            if item.occurrence != occurrences[item.tool_name]:
                raise ValueError("Tool-call evidence has a non-canonical tool occurrence.")
            if evidence_policy.include_tool_arguments:
                if item.arguments.state == "unsupported":
                    raise ValueError(
                        "Enabled tool-argument evidence cannot have an unsupported state."
                    )
            elif item.arguments.state != "unsupported":
                raise ValueError("Disabled tool-argument evidence must have an unsupported state.")
            if evidence_policy.include_tool_results:
                if item.result.state == "unsupported":
                    raise ValueError(
                        "Enabled tool-result evidence cannot have an unsupported state."
                    )
            elif item.result.state != "unsupported":
                raise ValueError("Disabled tool-result evidence must have an unsupported state.")
        if self.process_event_evidence_state == "unavailable":
            if self.process_events:
                raise ValueError("Unavailable process-event evidence cannot carry observations.")
        elif self.process_event_evidence_state == "limit_exceeded" and (
            len(self.process_events) != EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS
        ):
            raise ValueError(
                "Limited process-event evidence must retain exactly its bounded prefix."
            )
        if self.model_step_evidence_state == "unavailable":
            if self.model_steps is not None:
                raise ValueError("Unavailable model-step evidence cannot carry a count.")
        elif self.model_steps is None:
            raise ValueError("Available model-step evidence requires an exact count.")
        elif self.model_step_evidence_state == "complete" and (
            self.model_steps > EVIDENCE_MAX_MODEL_STEPS
        ):
            raise ValueError("Complete model-step evidence exceeds its declared bound.")
        elif self.model_step_evidence_state == "limit_exceeded" and (
            self.model_steps <= EVIDENCE_MAX_MODEL_STEPS
        ):
            raise ValueError("Limited model-step evidence must exceed its declared bound.")
        if self.usage_evidence_state == "unavailable":
            if self.total_tokens is not None:
                raise ValueError("Unavailable usage evidence cannot carry a token count.")
        elif self.total_tokens is None:
            raise ValueError("Available usage evidence requires an exact token count.")
        elif (
            self.usage_evidence_state == "complete"
            and self.total_tokens > EVIDENCE_MAX_TOTAL_TOKENS
        ):
            raise ValueError("Complete usage evidence exceeds its declared bound.")
        elif (
            self.usage_evidence_state == "limit_exceeded"
            and self.total_tokens <= EVIDENCE_MAX_TOTAL_TOKENS
        ):
            raise ValueError("Limited usage evidence must exceed its declared bound.")
        if self.root_status is None and self.costs:
            raise ValueError("Cost evidence requires a durable root session.")
        if self.pricing_profile_fingerprint is None and self.costs:
            raise ValueError("Cost evidence requires a pricing profile fingerprint.")
        if self.costs and self.model_step_evidence_state != "complete":
            raise ValueError("Cost evidence requires complete model-step evidence.")
        if self.model_steps is not None and any(
            cost.model_steps != self.model_steps for cost in self.costs
        ):
            raise ValueError("Cost evidence must cover the view's exact model-step count.")
        currencies = tuple(cost.currency for cost in self.costs)
        if currencies != tuple(sorted(set(currencies))):
            raise ValueError("Cost evidence currencies must be unique and sorted.")
        if not json_utf8_size_within_limit(self, ASSERTION_EVIDENCE_MAX_BYTES):
            raise ValueError(
                f"Assertion evidence exceeds {ASSERTION_EVIDENCE_MAX_BYTES} canonical JSON bytes."
            )
        expected = _model_content_revision(self, "assertion evidence")
        if self.revision != expected:
            raise ValueError("Assertion evidence revision does not match its content.")
        return self


def _validated_policy(policy: EvaluationEvidencePolicySpec) -> EvaluationEvidencePolicySpec:
    if type(policy) is not EvaluationEvidencePolicySpec:
        raise TypeError("evidence_policy must be an exact EvaluationEvidencePolicySpec.")
    return EvaluationEvidencePolicySpec.model_validate(
        policy.model_dump(mode="python", round_trip=True, warnings="none")
    )


def _validated_pricing(
    pricing: PriceBook | None,
) -> _ValidatedPricingSnapshot | None:
    if pricing is None:
        return None
    validated = PriceBook.model_validate(
        pricing.model_dump(mode="python", round_trip=True, warnings="none")
    )
    return _ValidatedPricingSnapshot(
        price_book=validated,
        identity=_pricing_profile_identity_from_validated_price_book(validated),
    )


def _validated_currencies(currencies: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(currencies, list | tuple):
        raise TypeError("cost_currencies must be an ordered sequence (a list or tuple).")
    ordered_currencies = cast("Sequence[str]", currencies)
    if len(ordered_currencies) > ASSERTION_EVIDENCE_MAX_COST_CURRENCIES:
        raise ValueError(
            "cost_currencies exceeds the assertion-evidence currency limit of "
            f"{ASSERTION_EVIDENCE_MAX_COST_CURRENCIES}."
        )
    cleaned = tuple(
        _bounded_durable_text(
            currency,
            f"cost_currencies[{index}]",
            max_chars=16,
            nonblank=True,
            clean=True,
        )
        for index, currency in enumerate(ordered_currencies)
    )
    if any(_CURRENCY_PATTERN.fullmatch(currency) is None for currency in cleaned):
        raise ValueError("cost_currencies must use portable uppercase identifiers.")
    if cleaned != tuple(sorted(set(cleaned))):
        raise ValueError("cost_currencies must be unique and sorted.")
    return cleaned


def _redacted_text(app: CayuApp | None, value: str, field_name: str) -> str:
    if app is None:
        return value
    redacted = app.redact_json(value)
    if type(redacted) is not str:
        raise ValueError(f"{field_name} redaction did not return text.")
    return redacted


def _bounded_tool_names(
    names: Iterable[str | None],
    *,
    app: CayuApp | None,
) -> tuple[tuple[str, ...], bool] | None:
    values: list[str] = []
    for name in names:
        if len(values) >= EVIDENCE_MAX_TOOL_CALLS:
            return tuple(values), True
        if type(name) is not str:
            return None
        name = _redacted_text(app, name, "tool name")
        try:
            name = _bounded_durable_text(
                name,
                "tool name",
                max_chars=ASSERTION_EVIDENCE_MAX_TOOL_NAME_CHARS,
                nonblank=True,
                clean=True,
            )
        except ValueError:
            return None
        values.append(name)
    return tuple(values), False


def _requested_tool_names(trajectory: Trajectory) -> Iterable[str]:
    return (
        part.tool_name
        for message in trajectory.transcript
        for part in message.content
        if type(part) is ToolCallPart
    )


def _started_tool_names(trajectory: Trajectory) -> Iterable[str | None]:
    return (
        event.tool_name for event in trajectory.events if event.type == EventType.TOOL_CALL_STARTED
    )


def _tool_round_identity_from_part(
    part: ToolCallPart,
) -> tuple[str, str, str] | None:
    if part.tool_round_id is None:
        return None
    # ToolCallPart validation requires all three values together.
    assert part.model_step_id is not None
    assert part.model_attempt_id is not None
    return part.tool_round_id, part.model_step_id, part.model_attempt_id


def _tool_round_identity_from_event_payload(
    payload: dict[str, Any],
) -> tuple[bool, tuple[str, str, str] | None]:
    tool_round_id = payload.get("tool_round_id")
    model_step_id = payload.get("model_step_id")
    model_attempt_id = payload.get("model_attempt_id")
    if tool_round_id is model_step_id is model_attempt_id is None:
        return True, None
    if (
        type(tool_round_id) is not str
        or not tool_round_id
        or type(model_step_id) is not str
        or not model_step_id
        or type(model_attempt_id) is not str
        or not model_attempt_id
    ):
        return False, None
    return True, (tool_round_id, model_step_id, model_attempt_id)


def _tool_lifecycle_matches_transcript(trajectory: Trajectory) -> bool:
    """Require every started call to originate from one transcript request.

    Correlation happens on unredacted identities. A counter preserves duplicate
    tool names while keeping this check linear at the public 10,000-call bound.
    Requested calls that never started are valid (for example, policy denial
    before execution), but a started call without an exact request is not
    conclusive assertion evidence.
    """

    requested = Counter(
        (
            part.tool_call_id,
            part.tool_name,
            _tool_round_identity_from_part(part),
        )
        for message in trajectory.transcript
        for part in message.content
        if type(part) is ToolCallPart
    )
    started: Counter[tuple[str, str, tuple[str, str, str] | None]] = Counter()
    for event in trajectory.events:
        if event.type != EventType.TOOL_CALL_STARTED:
            continue
        tool_call_id = event.payload.get("tool_call_id")
        valid_identity, event_identity = _tool_round_identity_from_event_payload(event.payload)
        if (
            type(tool_call_id) is not str
            or not tool_call_id
            or type(event.tool_name) is not str
            or not event.tool_name
            or not valid_identity
        ):
            return False
        started[(tool_call_id, event.tool_name, event_identity)] += 1
    return not bool(started - requested)


def _project_tool_evidence(
    trajectory: Trajectory,
    *,
    max_tool_calls: int,
    app: CayuApp | None,
    root_evidence_available: bool | None = None,
    allow_event_count_fallback: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], int | None, EvidenceState]:
    """Project one internally consistent, bounded tool-evidence observation."""

    root_available = (
        trajectory.session is not None
        if root_evidence_available is None
        else root_evidence_available
    )
    if not root_available:
        return (), (), None, "unavailable"
    if not _tool_lifecycle_matches_transcript(trajectory):
        return (), (), None, "unavailable"
    requested_result = _bounded_tool_names(_requested_tool_names(trajectory), app=app)
    started_result = _bounded_tool_names(_started_tool_names(trajectory), app=app)
    usage = trajectory.usage_summary
    tool_count = (
        sum(event.type == EventType.TOOL_CALL_STARTED for event in trajectory.events)
        if usage is None and allow_event_count_fallback
        else (None if usage is None else usage.tool_calls)
    )
    if requested_result is None or started_result is None or tool_count is None:
        return (), (), None, "unavailable"
    requested_names, requested_overflow = requested_result
    started_names, started_overflow = started_result
    if (started_overflow and tool_count <= len(started_names)) or (
        not started_overflow and tool_count != len(started_names)
    ):
        return (), (), None, "unavailable"
    state: EvidenceState = (
        "limit_exceeded"
        if requested_overflow or started_overflow or tool_count > max_tool_calls
        else "complete"
    )
    return requested_names, started_names, tool_count, state


_TOOL_TERMINAL_EVENT_TYPES = frozenset(
    {
        EventType.TOOL_CALL_COMPLETED,
        EventType.TOOL_CALL_FAILED,
        EventType.TOOL_CALL_BLOCKED,
    }
)


def _tool_event_key(event: Event) -> tuple[str, str, tuple[str, str, str] | None] | None:
    tool_call_id = event.payload.get("tool_call_id")
    valid_identity, round_identity = _tool_round_identity_from_event_payload(event.payload)
    if (
        type(tool_call_id) is not str
        or not tool_call_id
        or type(event.tool_name) is not str
        or not event.tool_name
        or not valid_identity
    ):
        return None
    return tool_call_id, event.tool_name, round_identity


def _unavailable_tool_value(
    state: Literal["unavailable", "unsupported", "malformed", "incompatible"] = "unavailable",
) -> ToolCallValueEvidenceV1:
    return ToolCallValueEvidenceV1(state=state)


def _project_tool_json_value(
    value: object,
    *,
    app: CayuApp | None,
) -> ToolCallValueEvidenceV1:
    try:
        projected = value if app is None else app.redact_json(value)
        copied = copy_eval_tool_json_object(projected, "tool call evidence")
    except DurableValueError as exc:
        state: Literal["truncated", "malformed"] = (
            "truncated"
            if exc.code in {"json_value_too_large", "nesting_too_deep", "too_many_json_nodes"}
            else "malformed"
        )
        return ToolCallValueEvidenceV1(state=state)
    except (TypeError, ValueError):
        return ToolCallValueEvidenceV1(state="malformed")
    return ToolCallValueEvidenceV1(state="available", value=copied)


def _project_tool_arguments(
    terminal: Event | None,
    *,
    include: bool,
    incompatible: bool,
    app: CayuApp | None,
) -> ToolCallValueEvidenceV1:
    if not include:
        return _unavailable_tool_value("unsupported")
    if incompatible:
        return _unavailable_tool_value("incompatible")
    if terminal is None:
        return _unavailable_tool_value()
    arguments_state = terminal.payload.get("arguments_state")
    if arguments_state != "finalized":
        return _unavailable_tool_value(
            "unavailable" if arguments_state in {None, "unavailable"} else "malformed"
        )
    arguments = terminal.payload.get("effective_arguments", terminal.payload.get("arguments"))
    if type(arguments) is not dict:
        return _unavailable_tool_value("malformed")
    return _project_tool_json_value(arguments, app=app)


def _project_tool_result(
    terminal: Event | None,
    *,
    include: bool,
    incompatible: bool,
    app: CayuApp | None,
) -> ToolCallValueEvidenceV1:
    if not include:
        return _unavailable_tool_value("unsupported")
    if incompatible:
        return _unavailable_tool_value("incompatible")
    if terminal is None:
        return _unavailable_tool_value()
    projection = terminal.payload.get("tool_result_projection")
    if type(projection) is dict:
        if projection.get("status") == "externalized":
            return ToolCallValueEvidenceV1(state="truncated")
        if projection.get("status") == "failed":
            return _unavailable_tool_value()
    raw_result = terminal.payload.get("result")
    if type(raw_result) is not dict:
        return _unavailable_tool_value("malformed")
    try:
        result = ToolResult.model_validate(raw_result)
    except (TypeError, ValueError):
        return _unavailable_tool_value("malformed")
    structured = result.structured
    if type(structured) is dict and structured.get("portable_result_evidence_incomplete") is True:
        return ToolCallValueEvidenceV1(state="truncated")
    return _project_tool_json_value(
        {
            "content": result.content,
            "structured": result.structured,
            "is_error": result.is_error,
        },
        app=app,
    )


def _project_tool_call_evidence(
    trajectory: Trajectory,
    *,
    evidence_policy: EvaluationEvidencePolicySpec,
    app: CayuApp | None,
    root_evidence_available: bool,
    tool_evidence_state: EvidenceState,
    tool_calls_started: int | None,
    started_tool_names: tuple[str, ...],
) -> tuple[tuple[ToolCallEvidenceV1, ...], ToolCallEvidenceState]:
    if not root_evidence_available:
        return (), "unavailable"
    if tool_evidence_state == "unavailable" or tool_calls_started is None:
        has_tool_lifecycle = any(
            event.type == EventType.TOOL_CALL_STARTED or event.type in _TOOL_TERMINAL_EVENT_TYPES
            for event in trajectory.events
        )
        return (), "incompatible" if has_tool_lifecycle else "unavailable"
    started_records = tuple(
        (position, event)
        for position, event in enumerate(trajectory.events)
        if event.type == EventType.TOOL_CALL_STARTED
    )
    if len(started_records) != tool_calls_started:
        return (), "incompatible"
    terminal_by_key: dict[
        tuple[str, str, tuple[str, str, str] | None],
        tuple[int, Event] | None,
    ] = {}
    ambiguous_keys = {
        key
        for key, count in Counter(_tool_event_key(event) for _, event in started_records).items()
        if key is not None and count > 1
    }
    for position, event in enumerate(trajectory.events):
        if event.type not in _TOOL_TERMINAL_EVENT_TYPES:
            continue
        key = _tool_event_key(event)
        if key is None:
            continue
        if key in terminal_by_key:
            terminal_by_key[key] = None
            ambiguous_keys.add(key)
        else:
            terminal_by_key[key] = (position, event)

    retained_count = min(len(started_records), EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS)
    occurrences: Counter[str] = Counter()
    projected: list[ToolCallEvidenceV1] = []
    for index, (started_position, event) in enumerate(
        started_records[:retained_count],
        start=1,
    ):
        key = _tool_event_key(event)
        if key is None or index > len(started_tool_names):
            return (), "incompatible"
        public_tool_name = started_tool_names[index - 1]
        occurrences[public_tool_name] += 1
        invocation_document = {
            "schema_version": 1,
            "tool_call_id": key[0],
            "tool_name": key[1],
            "tool_round_identity": None if key[2] is None else list(key[2]),
            "invocation_index": index,
        }
        invocation_revision = (
            "sha256:"
            + hashlib.sha256(
                canonical_durable_json_bytes(invocation_document, "tool invocation identity")
            ).hexdigest()
        )
        terminal_record = terminal_by_key.get(key)
        terminal = None if terminal_record is None else terminal_record[1]
        incompatible = key in ambiguous_keys or (
            terminal_record is not None and terminal_record[0] <= started_position
        )
        projected.append(
            ToolCallEvidenceV1(
                invocation_index=index,
                invocation_revision=invocation_revision,
                tool_name=public_tool_name,
                occurrence=occurrences[public_tool_name],
                arguments=_project_tool_arguments(
                    terminal,
                    include=evidence_policy.include_tool_arguments,
                    incompatible=incompatible,
                    app=app,
                ),
                result=_project_tool_result(
                    terminal,
                    include=evidence_policy.include_tool_results,
                    incompatible=incompatible,
                    app=app,
                ),
            )
        )
    state: EvidenceState = (
        "limit_exceeded" if tool_calls_started > EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS else "complete"
    )
    return tuple(projected), state


def _build_assertion_evidence_view(
    trajectory: Trajectory,
    *,
    evidence_policy: EvaluationEvidencePolicySpec,
    pricing_snapshot: _ValidatedPricingSnapshot | None,
    cost_currencies: tuple[str, ...],
    app: CayuApp | None,
    root_evidence_available: bool | None = None,
    allow_event_count_fallback: bool = False,
    expected_pricing_profile_fingerprint: str | None = None,
    bind_pricing_profile: bool = False,
    memory_attribution_evidence: EvalMemoryAttributionEvidenceV1 | None = None,
) -> AssertionEvidenceView:
    session = trajectory.session
    root_status = None if session is None else session.status.value
    root_available = (
        session is not None if root_evidence_available is None else root_evidence_available
    )

    children_available = root_available and not trajectory.children_incomplete
    retained_children = trajectory.children[: evidence_policy.max_child_sessions]
    if not children_available or any(child.session is None for child in retained_children):
        child_state = "unavailable"
        child_statuses = ()
    elif len(trajectory.children) > evidence_policy.max_child_sessions:
        child_statuses = tuple(
            child.session.status.value for child in retained_children if child.session is not None
        )
        child_state = "limit_exceeded"
    else:
        child_statuses = tuple(
            child.session.status.value for child in retained_children if child.session is not None
        )
        child_state = "complete"

    if not root_available:
        final_output = ""
        final_output_state: EvidenceState = "unavailable"
    else:
        redacted_output = _redacted_text(app, trajectory.final_output, "final output")
        if len(redacted_output) > evidence_policy.max_final_output_chars:
            final_output = redacted_output[: evidence_policy.max_final_output_chars]
            final_output_state = "limit_exceeded"
        else:
            final_output = redacted_output
            final_output_state = "complete"

    requested_tool_names, started_tool_names, tool_count, tool_state = _project_tool_evidence(
        trajectory,
        max_tool_calls=evidence_policy.max_tool_calls,
        app=app,
        root_evidence_available=root_available,
        allow_event_count_fallback=allow_event_count_fallback,
    )
    tool_calls, tool_call_state = _project_tool_call_evidence(
        trajectory,
        evidence_policy=evidence_policy,
        app=app,
        root_evidence_available=root_available,
        tool_evidence_state=tool_state,
        tool_calls_started=tool_count,
        started_tool_names=started_tool_names,
    )

    if not root_available:
        process_events: tuple[EvalProcessEventKind, ...] = ()
        process_event_state: EvidenceState = "unavailable"
    else:
        retained_process_events: list[EvalProcessEventKind] = []
        process_events_overflow = False
        for event in trajectory.events:
            process_event = _PORTABLE_PROCESS_EVENT_KINDS.get(event.type)
            if process_event is None:
                continue
            if len(retained_process_events) == EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS:
                process_events_overflow = True
                break
            retained_process_events.append(process_event)
        process_events = tuple(retained_process_events)
        process_event_state = "limit_exceeded" if process_events_overflow else "complete"

    # Public projection derives completeness from the durable root. The compiled
    # EvalAssertion adapter may instead receive an explicitly complete synthetic
    # context, matching the existing direct-assertion contract.
    usage = trajectory.usage_summary if root_available else None
    model_steps = (
        sum(event.type == EventType.MODEL_COMPLETED for event in trajectory.events)
        if usage is None and root_available and allow_event_count_fallback
        else (None if usage is None else usage.model_steps)
    )
    if model_steps is None:
        model_step_state: EvidenceState = "unavailable"
    elif model_steps > evidence_policy.max_model_steps:
        model_step_state = "limit_exceeded"
    else:
        model_step_state = "complete"

    total_tokens = None if usage is None else usage.usage.total_tokens
    if total_tokens is None:
        usage_state: EvidenceState = "unavailable"
    elif total_tokens > evidence_policy.max_total_tokens:
        usage_state = "limit_exceeded"
    else:
        usage_state = "complete"

    pricing_profile_fingerprint = (
        None if pricing_snapshot is None else pricing_snapshot.identity.fingerprint
    )
    if bind_pricing_profile and pricing_profile_fingerprint != expected_pricing_profile_fingerprint:
        raise ValueError("Compiled pricing profile changed after assertion compilation.")
    costs: list[AssertionCostEvidenceV1] = []
    if pricing_snapshot is not None and session is not None and model_step_state == "complete":
        events = list(trajectory.events)
        for currency in cost_currencies:
            summary = _estimate_session_cost(
                session_id=session.id,
                events=events,
                pricing=pricing_snapshot.price_book,
                currency=currency,
            )

            costs.append(
                AssertionCostEvidenceV1(
                    currency=summary.currency,
                    total_cost=_canonical_decimal(summary.total_cost),
                    model_steps=summary.model_steps,
                    priced_model_steps=summary.priced_model_steps,
                    unpriced_model_steps=summary.unpriced_model_steps,
                )
            )

    alias_key = None if app is None else memory_evidence_key(app._request_footprint)
    projected_memory_attribution = eval_memory_attribution_evidence_from_trajectory(
        trajectory,
        effective_bounds=(
            None
            if memory_attribution_evidence is None
            else memory_attribution_evidence.effective_bounds
        ),
        effective_source_limit=(
            EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES
            if memory_attribution_evidence is None
            else memory_attribution_evidence.effective_source_limit
        ),
        effective_max_bytes=(
            EVAL_MEMORY_ATTRIBUTION_MAX_BYTES
            if memory_attribution_evidence is None
            else memory_attribution_evidence.effective_max_bytes
        ),
        source_alias_key_id=None if alias_key is None else alias_key.key_id,
        source_alias_key=None if alias_key is None else alias_key.key,
    )
    if memory_attribution_evidence is None:
        memory_attribution = projected_memory_attribution
    else:
        if type(memory_attribution_evidence) is not EvalMemoryAttributionEvidenceV1:
            raise TypeError(
                "memory_attribution_evidence must be exact eval memory evidence or None."
            )
        memory_attribution = EvalMemoryAttributionEvidenceV1.model_validate(
            memory_attribution_evidence.model_dump(
                mode="python",
                round_trip=True,
                warnings="none",
            )
        )
        if memory_attribution != projected_memory_attribution:
            raise ValueError("Prepared memory attribution does not match the assertion trajectory.")

    document: dict[str, Any] = {
        "schema_version": ASSERTION_EVIDENCE_SCHEMA_VERSION,
        "policy_revision": evidence_policy.revision,
        "pricing_profile_fingerprint": pricing_profile_fingerprint,
        "root_evidence_available": root_available,
        "root_status": root_status,
        "child_statuses": list(child_statuses),
        "child_evidence_state": child_state,
        "final_output": final_output,
        "final_output_state": final_output_state,
        "requested_tool_names": list(requested_tool_names),
        "started_tool_names": list(started_tool_names),
        "tool_calls_started": tool_count,
        "tool_evidence_state": tool_state,
        "tool_calls": [item.model_dump(mode="json") for item in tool_calls],
        "tool_call_evidence_state": tool_call_state,
        "process_events": list(process_events),
        "process_event_evidence_state": process_event_state,
        "model_steps": model_steps,
        "model_step_evidence_state": model_step_state,
        "total_tokens": total_tokens,
        "usage_evidence_state": usage_state,
        "costs": [cost.model_dump(mode="json") for cost in costs],
        "memory_attribution": memory_attribution.model_dump(mode="json"),
    }
    revision_document = dict(document)
    if total_tokens is not None:
        # AggregateCount deliberately serializes as a canonical decimal string
        # so exact sums may exceed the signed-int64 JSON number domain.
        revision_document["total_tokens"] = str(total_tokens)
    return AssertionEvidenceView(
        revision=_content_revision(revision_document, "assertion evidence"),
        **document,
    )


def project_assertion_evidence_view(
    app: CayuApp,
    trajectory: Trajectory,
    *,
    evidence_policy: EvaluationEvidencePolicySpec,
    pricing: PriceBook | None = None,
    cost_currencies: Sequence[str] = (),
    memory_attribution_evidence: EvalMemoryAttributionEvidenceV1 | None = None,
) -> AssertionEvidenceView:
    """Build a detached, redacted assertion view from one validated trajectory."""

    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp.")
    if type(trajectory) is not Trajectory:
        raise TypeError("trajectory must be an exact Trajectory.")
    validated_trajectory = Trajectory.model_validate(_model_instance_python_input(trajectory))
    _validate_trajectory_record_contract(validated_trajectory)
    validated_policy = _validated_policy(evidence_policy)
    currencies = _validated_currencies(cost_currencies)
    pricing_snapshot = _validated_pricing(pricing) if currencies else None
    return _build_assertion_evidence_view(
        validated_trajectory,
        evidence_policy=validated_policy,
        pricing_snapshot=pricing_snapshot,
        cost_currencies=currencies,
        app=app,
        memory_attribution_evidence=memory_attribution_evidence,
    )
