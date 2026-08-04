from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, cast

from pydantic import Field, StrictInt, StrictStr, field_validator, model_validator

from cayu._validation import json_utf8_size_within_limit
from cayu.core.events import EventType
from cayu.core.messages import ToolCallPart
from cayu.evals.corpus import (
    _CURRENCY_PATTERN,
    EVIDENCE_MAX_CHILD_SESSIONS,
    EVIDENCE_MAX_FINAL_OUTPUT_CHARS,
    EVIDENCE_MAX_MODEL_STEPS,
    EVIDENCE_MAX_TOOL_CALLS,
    EVIDENCE_MAX_TOTAL_TOKENS,
    EvaluationEvidencePolicySpec,
    PricingProfileIdentityV1,
    _bounded_durable_text,
    _canonical_decimal_text,
    _content_revision,
    _model_content_revision,
    _ordered_sequence_input,
    _PortableModel,
    _pricing_profile_identity_from_validated_price_book,
    _SchemaV1PortableModel,
    _sha256_revision,
)
from cayu.evals.models import (
    Trajectory,
    _model_instance_python_input,
    _validate_trajectory_record_contract,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.costs import PriceBook, _estimate_session_cost
from cayu.runtime.usage import AggregateCount

ASSERTION_EVIDENCE_SCHEMA_VERSION = 1
ASSERTION_EVIDENCE_MAX_BYTES = 10 << 20
ASSERTION_EVIDENCE_MAX_TOOL_NAME_CHARS = 256
ASSERTION_EVIDENCE_MAX_COST_CURRENCIES = 32

EvidenceState = Literal["complete", "unavailable", "limit_exceeded"]
TerminalEvidenceStatus = Literal["completed", "failed", "interrupted"]


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


class AssertionEvidenceView(_SchemaV1PortableModel):
    """The bounded, alias-free data consumed by every portable assertion."""

    schema_version: Literal[1] = ASSERTION_EVIDENCE_SCHEMA_VERSION
    revision: StrictStr
    policy_revision: StrictStr
    pricing_profile_fingerprint: StrictStr | None = None
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

    @field_validator(
        "child_statuses",
        "requested_tool_names",
        "started_tool_names",
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
        if self.policy_revision != EvaluationEvidencePolicySpec.standard().revision:
            raise ValueError("Assertion evidence policy revision is not supported.")
        if self.root_status is None and self.child_evidence_state == "complete":
            raise ValueError("Child evidence cannot be complete without root evidence.")
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
        if self.root_status is None and any(
            state != "unavailable"
            for state in (
                self.child_evidence_state,
                self.final_output_state,
                self.tool_evidence_state,
                self.model_step_evidence_state,
                self.usage_evidence_state,
            )
        ):
            raise ValueError("A missing root cannot carry available assertion evidence.")
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


def _build_assertion_evidence_view(
    trajectory: Trajectory,
    *,
    evidence_policy: EvaluationEvidencePolicySpec,
    pricing_snapshot: _ValidatedPricingSnapshot | None,
    cost_currencies: tuple[str, ...],
    app: CayuApp | None,
) -> AssertionEvidenceView:
    session = trajectory.session
    root_status = None if session is None else session.status.value

    children_available = session is not None and not trajectory.children_incomplete
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

    if session is None:
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
    )

    # A trajectory without a durable root is missing evidence, even when a
    # synthetic or partially reconstructed value carries stray usage fields.
    # This mirrors the public replay boundary and keeps direct and compiled
    # assertions from assigning different meaning to the same trajectory.
    usage = trajectory.usage_summary if session is not None else None
    model_steps = None if usage is None else usage.model_steps
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

    document: dict[str, Any] = {
        "schema_version": ASSERTION_EVIDENCE_SCHEMA_VERSION,
        "policy_revision": evidence_policy.revision,
        "pricing_profile_fingerprint": pricing_profile_fingerprint,
        "root_status": root_status,
        "child_statuses": list(child_statuses),
        "child_evidence_state": child_state,
        "final_output": final_output,
        "final_output_state": final_output_state,
        "requested_tool_names": list(requested_tool_names),
        "started_tool_names": list(started_tool_names),
        "tool_calls_started": tool_count,
        "tool_evidence_state": tool_state,
        "model_steps": model_steps,
        "model_step_evidence_state": model_step_state,
        "total_tokens": total_tokens,
        "usage_evidence_state": usage_state,
        "costs": [cost.model_dump(mode="json") for cost in costs],
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
    )
