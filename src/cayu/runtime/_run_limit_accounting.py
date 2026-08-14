from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu.core.events import Event
from cayu.runtime.budgets import BudgetLimit, _effective_budget_limit_id
from cayu.runtime.stop_policy import RunLimits, has_run_limits
from cayu.runtime.usage import (
    SessionUsageSummary,
    aggregate_usage_metrics_from_durable_payload,
    session_usage_summary,
)


class RunBudgetAccountingAuthority(BaseModel):
    """Bounded original-run origin for one effective budget limit."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    budget_limit_id: str
    currency: str
    started_at: datetime

    @field_validator("budget_limit_id", "currency")
    @classmethod
    def validate_nonblank_identity(cls, value: str, info) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError(f"{info.field_name} must be a nonblank string.")
        return value.strip() if info.field_name == "currency" else value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.upper()

    @field_validator("started_at")
    @classmethod
    def validate_started_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware.")
        return value.astimezone(UTC)


class RunLimitAccountingContext(BaseModel):
    """Durable run-scoped usage and elapsed-time authority across a pause."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    started_at: datetime
    baseline: SessionUsageSummary
    run_budget_authorities: tuple[RunBudgetAccountingAuthority, ...] = ()

    @field_validator("started_at")
    @classmethod
    def validate_started_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("started_at must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("baseline", mode="before")
    @classmethod
    def copy_relevant_baseline(cls, value: object) -> SessionUsageSummary:
        if type(value) is SessionUsageSummary:
            copied = SessionUsageSummary.model_validate(
                value.model_dump(mode="python", warnings=False)
            )
        elif type(value) is dict and "usage" in value:
            durable_value = cast("dict[str, object]", value)
            copied = SessionUsageSummary.model_validate(
                {
                    **durable_value,
                    "usage": aggregate_usage_metrics_from_durable_payload(durable_value["usage"]),
                }
            )
        else:
            copied = SessionUsageSummary.model_validate(value)
        # Provider/model lists and model-step count are descriptive. Run-limit
        # subtraction uses only aggregate token counters and tool-call count;
        # retaining only those fields keeps this durable authority bounded.
        return SessionUsageSummary(
            session_id=copied.session_id,
            tool_calls=copied.tool_calls,
            usage=copied.usage,
        )

    @field_validator("run_budget_authorities", mode="before")
    @classmethod
    def copy_run_budget_authorities(
        cls,
        value: object,
    ) -> tuple[RunBudgetAccountingAuthority, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("run_budget_authorities must be a list or tuple.")
        copied: list[RunBudgetAccountingAuthority] = []
        for item in cast("list[object] | tuple[object, ...]", value):
            if type(item) is RunBudgetAccountingAuthority:
                item = item.model_dump(mode="python", warnings=False)
            copied.append(RunBudgetAccountingAuthority.model_validate(item))
        return tuple(copied)

    @model_validator(mode="after")
    def validate_distinct_budget_authorities(self) -> RunLimitAccountingContext:
        identities = [authority.budget_limit_id for authority in self.run_budget_authorities]
        if len(identities) != len(set(identities)):
            raise ValueError("Run budget authorities must have distinct limit identities.")
        return self


def has_run_limit_accounting_authority(
    limits: RunLimits | None,
    budget_limits: tuple[BudgetLimit, ...] | None,
) -> bool:
    """Return whether one request carries run-scoped accounting authority."""

    return (limits is not None and limits.scope == "run" and has_run_limits(limits)) or any(
        limit.scope == "run" for limit in budget_limits or ()
    )


def capture_run_limit_accounting_context(
    *,
    session_id: str,
    run_started_at: float,
    run_baseline: SessionUsageSummary | None,
    budget_limits: tuple[BudgetLimit, ...],
    now: datetime,
) -> RunLimitAccountingContext:
    """Snapshot one run's original baseline and cross-process time origin."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    elapsed_seconds = max(0.0, time.monotonic() - run_started_at)
    baseline = run_baseline or SessionUsageSummary(session_id=session_id)
    if baseline.session_id != session_id:
        raise ValueError("Run-limit baseline belongs to a different session.")
    durable_started_at = now.astimezone(UTC) - timedelta(seconds=elapsed_seconds)
    run_budget_authorities: list[RunBudgetAccountingAuthority] = []
    for limit in budget_limits:
        if limit.scope != "run":
            continue
        run_budget_authorities.append(
            RunBudgetAccountingAuthority(
                budget_limit_id=_effective_budget_limit_id(limit),
                currency=limit.currency,
                started_at=durable_started_at,
            )
        )
    return RunLimitAccountingContext(
        started_at=durable_started_at,
        baseline=baseline,
        run_budget_authorities=tuple(run_budget_authorities),
    )


def run_budget_authorities_from_context(
    context: RunLimitAccountingContext,
    *,
    budget_limits: tuple[BudgetLimit, ...],
) -> dict[str, RunBudgetAccountingAuthority]:
    """Authenticate every original run-budget identity and origin."""

    expected = {
        _effective_budget_limit_id(limit): limit for limit in budget_limits if limit.scope == "run"
    }
    actual = {authority.budget_limit_id: authority for authority in context.run_budget_authorities}
    if actual.keys() != expected.keys():
        raise ValueError("Run budget accounting does not match the effective budget limits.")
    for budget_limit_id, limit in expected.items():
        authority = actual[budget_limit_id]
        if authority.currency != limit.currency:
            raise ValueError("Run budget accounting currency does not match its limit.")
    return actual


def rebase_run_limit_accounting_context(
    context: RunLimitAccountingContext,
    *,
    session_id: str,
    limits: RunLimits,
    budget_limits: tuple[BudgetLimit, ...],
    events: list[Event],
    reset_run_limits: bool,
    reset_budgets: bool,
    now: datetime,
) -> RunLimitAccountingContext | None:
    """Apply independent continuation overrides without resetting the other authority."""

    if type(context) is not RunLimitAccountingContext:
        raise TypeError("Run-limit accounting must be a RunLimitAccountingContext.")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    if context.baseline.session_id != session_id:
        raise ValueError("Run-limit accounting belongs to a different session.")
    if not has_run_limit_accounting_authority(limits, budget_limits):
        return None

    if reset_run_limits:
        started_at = now.astimezone(UTC)
        baseline = session_usage_summary(session_id, events)
    else:
        started_at = context.started_at
        baseline = context.baseline

    if reset_budgets:
        run_budget_authorities = tuple(
            RunBudgetAccountingAuthority(
                budget_limit_id=_effective_budget_limit_id(limit),
                currency=limit.currency,
                started_at=now,
            )
            for limit in budget_limits
            if limit.scope == "run"
        )
    else:
        run_budget_authorities_from_context(context, budget_limits=budget_limits)
        run_budget_authorities = context.run_budget_authorities

    return RunLimitAccountingContext(
        started_at=started_at,
        baseline=baseline,
        run_budget_authorities=run_budget_authorities,
    )


def restore_run_limit_accounting_context(
    context: RunLimitAccountingContext,
    *,
    session_id: str,
    budget_limits: tuple[BudgetLimit, ...],
    now: datetime,
) -> tuple[
    float,
    SessionUsageSummary,
    dict[str, RunBudgetAccountingAuthority],
]:
    """Reconstruct the monotonic origin while authenticating the baseline owner."""

    if type(context) is not RunLimitAccountingContext:
        raise TypeError("Run-limit accounting must be a RunLimitAccountingContext.")
    if context.baseline.session_id != session_id:
        raise ValueError("Run-limit accounting belongs to a different session.")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    elapsed_seconds = max(
        0.0,
        (now.astimezone(UTC) - context.started_at).total_seconds(),
    )
    baseline = SessionUsageSummary.model_validate(
        context.baseline.model_dump(mode="python", warnings=False)
    )
    run_budget_authorities = run_budget_authorities_from_context(
        context,
        budget_limits=budget_limits,
    )
    return time.monotonic() - elapsed_seconds, baseline, run_budget_authorities
