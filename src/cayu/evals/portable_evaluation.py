from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from cayu.evals.corpus import (
    AssertionSpec,
    ChildStatusAssertionSpec,
    FinalOutputContainsAssertionSpec,
    FinalOutputEqualsAssertionSpec,
    MaxEstimatedCostAssertionSpec,
    MaxModelStepsAssertionSpec,
    MaxToolCallsAssertionSpec,
    MaxTotalTokensAssertionSpec,
    ModelJudgeAssertionSpec,
    RootStatusAssertionSpec,
    ToolCalledAssertionSpec,
    ToolsCalledInOrderAssertionSpec,
    UsageRecordedAssertionSpec,
    _validated_assertion_spec,
    assertion_spec_revision,
)
from cayu.evals.evidence import (
    AssertionCostEvidenceV1,
    AssertionEvidenceView,
    EvidenceState,
    _canonical_decimal,
)
from cayu.evals.models import EvalAssertionResult, EvalOutcome


def _result(
    name: str,
    outcome: EvalOutcome,
    message: str,
    *,
    metadata: dict | None = None,
) -> EvalAssertionResult:
    return EvalAssertionResult(
        name=name,
        outcome=outcome,
        score=1.0 if outcome is EvalOutcome.PASSED else 0.0,
        message=message,
        metadata={} if metadata is None else metadata,
    )


def _unavailable(name: str, evidence_area: str, state: EvidenceState) -> EvalAssertionResult:
    message = (
        "Usage evidence is unavailable because no usage summary was recorded."
        if evidence_area == "usage" and state == "unavailable"
        else f"{evidence_area} evidence is {state.replace('_', ' ')}."
    )
    return EvalAssertionResult(
        name=name,
        outcome=EvalOutcome.UNAVAILABLE,
        message=message,
        metadata={"evidence_area": evidence_area, "evidence_state": state},
    )


def _evaluate_root_status(
    *,
    name: str,
    expected: str,
    actual: str | None,
) -> EvalAssertionResult:
    if actual is None:
        return _unavailable(name, "root status", "unavailable")
    outcome = EvalOutcome.PASSED if actual == expected else EvalOutcome.FAILED
    return _result(
        name,
        outcome,
        (
            f"Root status matched {expected}."
            if outcome is EvalOutcome.PASSED
            else f"Expected root status {expected}, got {actual}."
        ),
        metadata={"expected": expected, "actual": actual},
    )


def _evaluate_child_status(
    *,
    name: str,
    expected: str,
    statuses: Sequence[str],
    state: EvidenceState,
    minimum: int,
    maximum: int | None,
) -> EvalAssertionResult:
    if state != "complete":
        return _unavailable(name, "child status", state)
    count = sum(status == expected for status in statuses)
    within_range = count >= minimum and (maximum is None or count <= maximum)
    return _result(
        name,
        EvalOutcome.PASSED if within_range else EvalOutcome.FAILED,
        (
            f"Observed {count} direct child session(s) with status {expected}."
            if within_range
            else f"Child status count {count} is outside the required range."
        ),
        metadata={
            "expected": expected,
            "count": count,
            "minimum": minimum,
            "maximum": maximum,
        },
    )


def _evaluate_final_output(
    *,
    name: str,
    expected: str,
    actual: str,
    state: EvidenceState,
    contains: bool,
) -> EvalAssertionResult:
    if state != "complete":
        return _unavailable(name, "final output", state)
    matched = expected in actual if contains else expected == actual
    comparison = "contained" if contains else "equaled"
    return _result(
        name,
        EvalOutcome.PASSED if matched else EvalOutcome.FAILED,
        (
            f"Final output {comparison} the expected text."
            if matched
            else f"Final output did not {comparison.removesuffix('ed')} the expected text."
        ),
        metadata={"matched": matched},
    )


def _evaluate_tool_called(
    *,
    name: str,
    tool_name: str,
    started_tool_names: Sequence[str],
    state: EvidenceState,
    minimum: int,
    maximum: int | None,
) -> EvalAssertionResult:
    if state != "complete":
        return _unavailable(name, "tool", state)
    count = sum(actual == tool_name for actual in started_tool_names)
    within_range = count >= minimum and (maximum is None or count <= maximum)
    return _result(
        name,
        EvalOutcome.PASSED if within_range else EvalOutcome.FAILED,
        (
            f"Observed tool {tool_name} {count} time(s)."
            if within_range
            else f"Tool {tool_name} count {count} is outside the required range."
        ),
        metadata={
            "tool_name": tool_name,
            "count": count,
            "minimum": minimum,
            "maximum": maximum,
        },
    )


def _evaluate_tools_in_order(
    *,
    name: str,
    expected: Sequence[str],
    actual: Sequence[str],
    state: EvidenceState,
) -> EvalAssertionResult:
    if state != "complete":
        return _unavailable(name, "tool", state)
    matched = tuple(actual) == tuple(expected)
    return _result(
        name,
        EvalOutcome.PASSED if matched else EvalOutcome.FAILED,
        (
            "Tool calls matched the exact expected transcript order."
            if matched
            else "Tool calls did not match the exact expected transcript order."
        ),
        metadata={"expected": list(expected), "actual": list(actual)},
    )


def _evaluate_maximum(
    *,
    name: str,
    evidence_area: str,
    actual: int | None,
    state: EvidenceState,
    maximum: int,
) -> EvalAssertionResult:
    if state != "complete" or actual is None:
        return _unavailable(name, evidence_area, state)
    passed = actual <= maximum
    return _result(
        name,
        EvalOutcome.PASSED if passed else EvalOutcome.FAILED,
        (
            f"{evidence_area.title()} count {actual} is within limit {maximum}."
            if passed
            else f"{evidence_area.title()} count {actual} exceeded limit {maximum}."
        ),
        metadata={"actual": actual, "maximum": maximum},
    )


def _evaluate_usage_recorded(
    *,
    name: str,
    total_tokens: int | None,
    state: EvidenceState,
    minimum: int,
) -> EvalAssertionResult:
    if state != "complete" or total_tokens is None:
        return _unavailable(name, "usage", state)
    passed = total_tokens >= minimum
    return _result(
        name,
        EvalOutcome.PASSED if passed else EvalOutcome.FAILED,
        (
            f"Total tokens {total_tokens} meets minimum {minimum}."
            if passed
            else f"Total tokens {total_tokens} is below minimum {minimum}."
        ),
        metadata={"total_tokens": total_tokens, "minimum": minimum},
    )


def _evaluate_max_cost(
    *,
    name: str,
    maximum: Decimal,
    currency: str,
    cost: AssertionCostEvidenceV1 | None,
) -> EvalAssertionResult:
    if cost is None:
        return _unavailable(name, "cost", "unavailable")
    maximum_text = _canonical_decimal(maximum)
    metadata = {
        "estimated_cost": cost.total_cost,
        "maximum": maximum_text,
        "priced_model_steps": cost.priced_model_steps,
        "unpriced_model_steps": cost.unpriced_model_steps,
        "currency": currency,
    }
    if cost.unpriced_model_steps:
        return EvalAssertionResult(
            name=name,
            outcome=EvalOutcome.UNAVAILABLE,
            message=("Cost evidence is unavailable because one or more model steps are unpriced."),
            metadata=metadata,
        )
    actual = Decimal(cost.total_cost)
    passed = actual <= maximum
    return _result(
        name,
        EvalOutcome.PASSED if passed else EvalOutcome.FAILED,
        (
            f"Estimated cost {actual} {currency} is within limit {maximum_text}."
            if passed
            else f"Estimated cost {actual} {currency} exceeded limit {maximum_text}."
        ),
        metadata=metadata,
    )


def _validated_evidence_view(evidence: AssertionEvidenceView) -> AssertionEvidenceView:
    if type(evidence) is not AssertionEvidenceView:
        raise TypeError("evidence must be an exact AssertionEvidenceView.")
    return AssertionEvidenceView.model_validate(
        evidence.model_dump(mode="python", round_trip=True, warnings="none")
    )


def _evaluate_validated_assertion_outcome(
    spec: AssertionSpec,
    evidence: AssertionEvidenceView,
) -> EvalAssertionResult:
    validated_spec = spec
    if type(validated_spec) is ModelJudgeAssertionSpec:
        raise ValueError("Portable model judges require a trusted CorpusTarget evaluator binding.")
    if type(validated_spec) is RootStatusAssertionSpec:
        return _evaluate_root_status(
            name=validated_spec.id,
            expected=validated_spec.expected,
            actual=evidence.root_status,
        )
    if type(validated_spec) is ChildStatusAssertionSpec:
        return _evaluate_child_status(
            name=validated_spec.id,
            expected=validated_spec.expected,
            statuses=evidence.child_statuses,
            state=evidence.child_evidence_state,
            minimum=validated_spec.min_count,
            maximum=validated_spec.max_count,
        )
    if type(validated_spec) is FinalOutputEqualsAssertionSpec:
        return _evaluate_final_output(
            name=validated_spec.id,
            expected=validated_spec.expected,
            actual=evidence.final_output,
            state=evidence.final_output_state,
            contains=False,
        )
    if type(validated_spec) is FinalOutputContainsAssertionSpec:
        return _evaluate_final_output(
            name=validated_spec.id,
            expected=validated_spec.expected,
            actual=evidence.final_output,
            state=evidence.final_output_state,
            contains=True,
        )
    if type(validated_spec) is ToolCalledAssertionSpec:
        return _evaluate_tool_called(
            name=validated_spec.id,
            tool_name=validated_spec.tool_name,
            started_tool_names=evidence.started_tool_names,
            state=evidence.tool_evidence_state,
            minimum=validated_spec.min_count,
            maximum=validated_spec.max_count,
        )
    if type(validated_spec) is ToolsCalledInOrderAssertionSpec:
        return _evaluate_tools_in_order(
            name=validated_spec.id,
            expected=validated_spec.tool_names,
            actual=evidence.requested_tool_names,
            state=evidence.tool_evidence_state,
        )
    if type(validated_spec) is MaxToolCallsAssertionSpec:
        return _evaluate_maximum(
            name=validated_spec.id,
            evidence_area="tool call",
            actual=evidence.tool_calls_started,
            state=evidence.tool_evidence_state,
            maximum=validated_spec.maximum,
        )
    if type(validated_spec) is MaxModelStepsAssertionSpec:
        return _evaluate_maximum(
            name=validated_spec.id,
            evidence_area="model step",
            actual=evidence.model_steps,
            state=evidence.model_step_evidence_state,
            maximum=validated_spec.maximum,
        )
    if type(validated_spec) is UsageRecordedAssertionSpec:
        return _evaluate_usage_recorded(
            name=validated_spec.id,
            total_tokens=evidence.total_tokens,
            state=evidence.usage_evidence_state,
            minimum=validated_spec.min_total_tokens,
        )
    if type(validated_spec) is MaxTotalTokensAssertionSpec:
        return _evaluate_maximum(
            name=validated_spec.id,
            evidence_area="total token",
            actual=evidence.total_tokens,
            state=evidence.usage_evidence_state,
            maximum=validated_spec.maximum,
        )
    if type(validated_spec) is MaxEstimatedCostAssertionSpec:
        cost = next(
            (item for item in evidence.costs if item.currency == validated_spec.currency),
            None,
        )
        return _evaluate_max_cost(
            name=validated_spec.id,
            maximum=Decimal(validated_spec.maximum),
            currency=validated_spec.currency,
            cost=cost,
        )
    raise AssertionError("Unreachable portable assertion type.")


def _evaluate_validated_assertion_spec(
    spec: AssertionSpec,
    evidence: AssertionEvidenceView,
    *,
    known_revision: str | None = None,
) -> EvalAssertionResult:
    """Evaluate one validated spec and bind the result to that exact definition."""

    result = _evaluate_validated_assertion_outcome(spec, evidence)
    revision = assertion_spec_revision(spec) if known_revision is None else known_revision
    return EvalAssertionResult.model_validate(
        {
            **result.model_dump(mode="python"),
            "assertion_revision": revision,
        }
    )


def evaluate_assertion_spec(
    spec: AssertionSpec,
    evidence: AssertionEvidenceView,
) -> EvalAssertionResult:
    """Evaluate one exact built-in spec against one detached evidence view."""

    return _evaluate_validated_assertion_spec(
        _validated_assertion_spec(spec),
        _validated_evidence_view(evidence),
    )


def evaluate_assertion_specs(
    specs: Sequence[AssertionSpec],
    evidence: AssertionEvidenceView,
) -> tuple[EvalAssertionResult, ...]:
    """Evaluate an ordered assertion-spec sequence without runtime side effects."""

    if not isinstance(specs, list | tuple):
        raise TypeError("specs must be an ordered sequence (a list or tuple).")
    validated_evidence = _validated_evidence_view(evidence)
    return tuple(
        _evaluate_validated_assertion_spec(
            _validated_assertion_spec(spec),
            validated_evidence,
        )
        for spec in specs
    )
