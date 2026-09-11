"""Declared maintenance budget requirements, not a live billing certificate."""

from decimal import Decimal

from cayu import BudgetLimit, BudgetPolicy


def require_maintenance_budget(value: object) -> BudgetPolicy:
    """Revalidate configured policy using Runtime's owner before accepting work.

    Production also needs shared durable history/ledger, pinned provider pricing
    and proven dispatch bounds. This check neither supplies nor certifies those.
    """
    try:
        if (
            type(value) is not BudgetPolicy
            or type(value.limits) is not tuple
            or len(value.limits) != 1
            or type(value.limits[0]) is not BudgetLimit
        ):
            raise ValueError
        policy = BudgetPolicy(limits=value.limits)
        limit = policy.limits[0]
        if (
            limit.scope != "app"
            or limit.window.kind != "all_time"
            or limit.currency != "USD"
            or limit.max_estimated_cost > Decimal("1")
            or limit.reservation is None
            or limit.allow_unpriced
            or limit.action != "interrupt"
        ):
            raise ValueError
        return policy
    except Exception:
        raise ValueError(
            "Maintenance requires a priced, reserved app budget of at most USD 1."
        ) from None
