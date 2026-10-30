"""Estimated cost of a direct eval run, priced from its trials' recorded usage.

Each trial's recorded model usage is priced with the run's price book by the
same estimator that Cayu's cost budgets and server-run cost assertions use. A
run's cost adds up its trials' costs, and when it is unavailable it says why.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import require_durable_clean_nonblank
from cayu.budgets.pricing import _estimate_session_cost, add_cost_amounts
from cayu.budgets.usage import count_model_steps_with_usage
from cayu.events import EventType

if TYPE_CHECKING:
    from cayu.evals.evidence import _ValidatedPricingSnapshot
    from cayu.events import Event

# Eval run cost is estimated in US dollars: the currency of Cayu's bundled price
# book, of `BudgetLimit.usd()` and of the eval cost assertions' default.
EVAL_RUN_COST_CURRENCY = "USD"

EvalTrialCostUnavailableReason = Literal["no_price_book", "usage_unavailable", "unpriced_usage"]
# ``not_recorded`` marks a result written before runs recorded their cost, or by
# a path that doesn't price its trials.
EvalRunCostUnavailableReason = Literal[
    "no_price_book",
    "usage_unavailable",
    "unpriced_usage",
    "not_recorded",
]
EvalCostDifferenceUnavailableReason = Literal[
    "baseline_unavailable",
    "current_unavailable",
    "pricing_differs",
]

_COST_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
# The bounds of Cayu's runtime cost evidence, which a real estimate fits well
# within. They stop a loaded amount from expanding into an enormous plain-notation
# string, or an enormous exact sum.
_COST_MAX_DIGITS = 128
_COST_MAX_DECIMAL_PLACES = 64
_COST_MAX_INTEGER_DIGITS = 64


def _bounded_cost_amount(amount: Decimal) -> Decimal:
    exponent = amount.as_tuple().exponent
    if (
        not amount.is_finite()
        or not isinstance(exponent, int)
        or exponent < -_COST_MAX_DECIMAL_PLACES
        or len(amount.as_tuple().digits) > _COST_MAX_DIGITS
        or amount.adjusted() >= _COST_MAX_INTEGER_DIGITS
    ):
        raise ValueError("A cost amount must be a finite decimal within its digit bounds.")
    return amount


def plain_cost_amount(amount: Decimal) -> str:
    """Write an exact amount in plain notation without trailing zeros, such as ``0.00032``."""

    text = format(_bounded_cost_amount(amount), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


# Amounts are exact, bounded decimals, written to JSON in plain notation.
_CostAmount = Annotated[
    Decimal,
    AfterValidator(_bounded_cost_amount),
    PlainSerializer(plain_cost_amount, return_type=str, when_used="json"),
]
# A run is unavailable for its most basic reason first.
_RUN_REASON_ORDER: tuple[EvalTrialCostUnavailableReason, ...] = (
    "usage_unavailable",
    "unpriced_usage",
)


class _PricedCost(BaseModel):
    """The price book and currency an observation was priced with."""

    model_config = _COST_MODEL_CONFIG

    availability: Literal["observed", "unavailable"]
    price_book_version: StrictStr | None = None
    pricing_profile_fingerprint: StrictStr | None = None
    currency: StrictStr | None = None

    @field_validator("price_book_version", "pricing_profile_fingerprint", "currency")
    @classmethod
    def validate_pricing_identity(cls, value: str | None, info) -> str | None:
        return None if value is None else require_durable_clean_nonblank(value, info.field_name)

    def _pricing(self) -> tuple[str | None, str | None, str | None]:
        return (self.price_book_version, self.pricing_profile_fingerprint, self.currency)

    def _require_pricing_identity(self, *, priced: bool) -> None:
        pricing = self._pricing()
        if priced and any(value is None for value in pricing):
            raise ValueError("A priced cost names its price book and currency.")
        if not priced and any(value is not None for value in pricing):
            raise ValueError("A cost without a price book names no price book or currency.")


class EvalTrialCost(_PricedCost):
    """One trial's recorded model usage, priced with its run's price book.

    ``observed`` means every model step the trial recorded was priced.
    Otherwise ``unavailable_reason`` says why: ``no_price_book``,
    ``usage_unavailable`` (the trial's usage wasn't completely recorded) or
    ``unpriced_usage`` (the price book has no price for some of its model calls).
    An unavailable trial keeps whatever could be priced, and that amount is not
    its whole cost.
    """

    unavailable_reason: EvalTrialCostUnavailableReason | None = None
    estimated_cost: _CostAmount | None = Field(default=None, ge=0)
    model_steps: StrictInt | None = Field(default=None, ge=0)
    priced_model_steps: StrictInt | None = Field(default=None, ge=0)
    unpriced_model_steps: StrictInt | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_availability(self) -> EvalTrialCost:
        if (self.availability == "observed") == (self.unavailable_reason is not None):
            raise ValueError("A trial cost has an unavailable reason exactly when unavailable.")
        self._require_pricing_identity(priced=self.unavailable_reason != "no_price_book")
        observations = (
            self.estimated_cost,
            self.model_steps,
            self.priced_model_steps,
            self.unpriced_model_steps,
        )
        if all(value is None for value in observations) and self.unavailable_reason in {
            "no_price_book",
            "usage_unavailable",
        }:
            return self
        if self.unavailable_reason == "no_price_book" or any(
            value is None for value in observations
        ):
            raise ValueError("A priced trial cost records its complete step accounting.")
        assert self.priced_model_steps is not None and self.unpriced_model_steps is not None
        if self.priced_model_steps + self.unpriced_model_steps != self.model_steps:
            raise ValueError("Priced and unpriced steps must sum to model_steps.")
        if self.availability == "observed" and self.unpriced_model_steps:
            raise ValueError("An observed trial cost has no unpriced model steps.")
        return self


class EvalRunCost(_PricedCost):
    """A run's estimated cost: its trials' costs added up, or why it is unavailable.

    ``observed`` means every trial's cost was observed, and ``estimated_cost``
    is the run's cost. When it is unavailable with a price book,
    ``estimated_cost`` covers only the ``observed_trials``.
    """

    unavailable_reason: EvalRunCostUnavailableReason | None = None
    estimated_cost: _CostAmount | None = Field(default=None, ge=0)
    observed_trials: StrictInt = Field(default=0, ge=0)

    @classmethod
    def not_recorded(cls) -> EvalRunCost:
        return cls(availability="unavailable", unavailable_reason="not_recorded")

    @model_validator(mode="after")
    def validate_availability(self) -> EvalRunCost:
        if (self.availability == "observed") == (self.unavailable_reason is not None):
            raise ValueError("A run cost has an unavailable reason exactly when unavailable.")
        priced = self.unavailable_reason not in {"no_price_book", "not_recorded"}
        self._require_pricing_identity(priced=priced)
        if (self.estimated_cost is None) != (self.observed_trials == 0):
            raise ValueError("A run cost's estimate covers exactly its observed trials.")
        if not priced and self.observed_trials:
            raise ValueError("A run without a price book has no observed cost.")
        if self.availability == "observed" and self.estimated_cost is None:
            raise ValueError("An observed run cost has an estimate.")
        return self


class EvalCostDifference(BaseModel):
    """The current run's estimated cost minus the baseline's.

    It is ``observed`` only when both runs' costs are observed with the same
    price book and currency.
    """

    model_config = _COST_MODEL_CONFIG

    availability: Literal["observed", "unavailable"]
    unavailable_reason: EvalCostDifferenceUnavailableReason | None = None
    currency: StrictStr | None = None
    estimated_cost: _CostAmount | None = None

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str | None, info) -> str | None:
        return None if value is None else require_durable_clean_nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_availability(self) -> EvalCostDifference:
        observed = self.availability == "observed"
        if observed == (self.unavailable_reason is not None):
            raise ValueError(
                "A cost difference has an unavailable reason exactly when unavailable."
            )
        if observed != (self.currency is not None and self.estimated_cost is not None):
            raise ValueError("Only an observed cost difference has an amount and currency.")
        return self


def eval_trial_cost(
    pricing: _ValidatedPricingSnapshot | None,
    *,
    session_id: str | None,
    events: Sequence[Event] | None,
    evidence_complete: bool,
) -> EvalTrialCost:
    """Price a trial's recorded model events with its run's price book.

    ``events`` are the events behind the trial's usage summary, or None when
    none were captured. Its usage is complete when ``evidence_complete`` and
    every model call it started recorded usage, the runner's rule for a
    complete ``usage_evidence_state``.
    """

    if pricing is None:
        return EvalTrialCost(availability="unavailable", unavailable_reason="no_price_book")
    version = pricing.identity.price_book_version
    fingerprint = pricing.identity.fingerprint
    if session_id is None or events is None:
        return EvalTrialCost(
            availability="unavailable",
            unavailable_reason="usage_unavailable",
            price_book_version=version,
            pricing_profile_fingerprint=fingerprint,
            currency=EVAL_RUN_COST_CURRENCY,
        )
    summary = _estimate_session_cost(
        session_id=session_id,
        events=list(events),
        pricing=pricing.price_book,
        currency=EVAL_RUN_COST_CURRENCY,
    )
    model_calls = sum(event.type == EventType.MODEL_STARTED for event in events)
    reason: EvalTrialCostUnavailableReason | None = None
    if not evidence_complete or count_model_steps_with_usage(events) < model_calls:
        reason = "usage_unavailable"
    elif any(not item.priced for item in summary.line_items):
        # Includes auxiliary attempts and hosted-tool charges outside model steps.
        reason = "unpriced_usage"
    return EvalTrialCost(
        availability="observed" if reason is None else "unavailable",
        unavailable_reason=reason,
        estimated_cost=summary.total_cost,
        model_steps=summary.model_steps,
        priced_model_steps=summary.priced_model_steps,
        unpriced_model_steps=summary.unpriced_model_steps,
        price_book_version=version,
        pricing_profile_fingerprint=fingerprint,
        currency=EVAL_RUN_COST_CURRENCY,
    )


def eval_run_cost(trial_costs: Iterable[EvalTrialCost | None]) -> EvalRunCost:
    """Add up a run's trial costs.

    A trial without a cost record makes the run's cost ``not_recorded``. Every
    priced trial in one run must share one price book and currency.
    """

    costs = tuple(trial_costs)
    if not costs:
        raise ValueError("A run cost needs at least one trial.")
    recorded = tuple(cost for cost in costs if cost is not None)
    if len(recorded) != len(costs):
        return EvalRunCost.not_recorded()
    pricings = {cost._pricing() for cost in recorded}
    if len(pricings) != 1:
        raise ValueError("Every trial in a run is priced with the same price book and currency.")
    ((price_book_version, fingerprint, currency),) = pricings
    if fingerprint is None:
        return EvalRunCost(availability="unavailable", unavailable_reason="no_price_book")
    reasons = {cost.unavailable_reason for cost in recorded}
    reason = next((item for item in _RUN_REASON_ORDER if item in reasons), None)
    observed = [cost.estimated_cost for cost in recorded if cost.availability == "observed"]
    return EvalRunCost(
        availability="observed" if reason is None else "unavailable",
        unavailable_reason=reason,
        price_book_version=price_book_version,
        pricing_profile_fingerprint=fingerprint,
        currency=currency,
        estimated_cost=(
            add_cost_amounts(*(amount for amount in observed if amount is not None))
            if observed
            else None
        ),
        observed_trials=len(observed),
    )


def eval_cost_difference(baseline: EvalRunCost, current: EvalRunCost) -> EvalCostDifference:
    """Subtract the baseline run's cost from the current run's."""

    reason: EvalCostDifferenceUnavailableReason | None = None
    if baseline.availability != "observed":
        reason = "baseline_unavailable"
    elif current.availability != "observed":
        reason = "current_unavailable"
    elif (baseline.pricing_profile_fingerprint, baseline.currency) != (
        current.pricing_profile_fingerprint,
        current.currency,
    ):
        reason = "pricing_differs"
    if reason is not None:
        return EvalCostDifference(availability="unavailable", unavailable_reason=reason)
    assert baseline.estimated_cost is not None and current.estimated_cost is not None
    return EvalCostDifference(
        availability="observed",
        currency=current.currency,
        # copy_negate changes the sign without the context rounding unary minus applies.
        estimated_cost=add_cost_amounts(
            current.estimated_cost, baseline.estimated_cost.copy_negate()
        ),
    )


__all__ = [
    "EVAL_RUN_COST_CURRENCY",
    "EvalCostDifference",
    "EvalCostDifferenceUnavailableReason",
    "EvalRunCost",
    "EvalRunCostUnavailableReason",
    "EvalTrialCost",
    "EvalTrialCostUnavailableReason",
    "eval_cost_difference",
    "eval_run_cost",
    "eval_trial_cost",
    "plain_cost_amount",
]
