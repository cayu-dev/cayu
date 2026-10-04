"""Capture and restore the live clock origin for durable run-limit accounting."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

from cayu.budgets import _run_limit_accounting as run_accounting
from cayu.budgets.base import BudgetLimit, _effective_budget_limit_id
from cayu.budgets.usage import SessionUsageSummary


def capture_run_limit_accounting_context(
    *,
    session_id: str,
    run_started_at: float,
    run_baseline: SessionUsageSummary | None,
    budget_limits: tuple[BudgetLimit, ...],
    now: datetime,
) -> run_accounting.RunLimitAccountingContext:
    """Snapshot one run's original baseline and cross-process time origin."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    elapsed_seconds = max(0.0, time.monotonic() - run_started_at)
    baseline = run_baseline or SessionUsageSummary(session_id=session_id)
    if baseline.session_id != session_id:
        raise ValueError("Run-limit baseline belongs to a different session.")
    durable_started_at = now.astimezone(UTC) - timedelta(seconds=elapsed_seconds)
    run_budget_authorities: list[run_accounting.RunBudgetAccountingAuthority] = []
    for limit in budget_limits:
        if limit.scope != "run":
            continue
        run_budget_authorities.append(
            run_accounting.RunBudgetAccountingAuthority(
                budget_limit_id=_effective_budget_limit_id(limit),
                currency=limit.currency,
                started_at=durable_started_at,
            )
        )
    return run_accounting.RunLimitAccountingContext(
        started_at=durable_started_at,
        baseline=baseline,
        run_budget_authorities=tuple(run_budget_authorities),
    )


def restore_run_limit_accounting_context(
    context: run_accounting.RunLimitAccountingContext,
    *,
    session_id: str,
    budget_limits: tuple[BudgetLimit, ...],
    now: datetime,
) -> tuple[
    float,
    SessionUsageSummary,
    dict[str, run_accounting.RunBudgetAccountingAuthority],
]:
    """Reconstruct the monotonic origin while authenticating the baseline owner."""

    if type(context) is not run_accounting.RunLimitAccountingContext:
        raise TypeError("Run-limit accounting must be a RunLimitAccountingContext.")
    if context.baseline.session_id != session_id:
        raise ValueError("Run-limit accounting belongs to a different session.")
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    if context.pause_started_at is not None:
        raise ValueError("Run-limit accounting must close its human pause before execution.")
    elapsed_seconds = max(
        0.0,
        (now.astimezone(UTC) - context.started_at).total_seconds(),
    )
    baseline = SessionUsageSummary.model_validate(
        context.baseline.model_dump(mode="python", warnings=False)
    )
    run_budget_authorities = run_accounting.run_budget_authorities_from_context(
        context,
        budget_limits=budget_limits,
    )
    return time.monotonic() - elapsed_seconds, baseline, run_budget_authorities
