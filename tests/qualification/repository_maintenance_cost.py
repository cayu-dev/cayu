"""Read-only recorded causal cost; neither billing nor completion authority."""

import json
from hashlib import sha256

from cayu import CausalBudgetCostSummary, copy_price_book
from tests.qualification.repository_maintenance_budget import require_maintenance_budget
from tests.qualification.repository_maintenance_identity import copy_identity


async def inspect_cost_evidence(app, expected):
    """Caller owns authorization; Runtime owns causal discovery and accounting."""
    identity = copy_identity(expected)
    policy = require_maintenance_budget(app.budget_policy)
    pricing = copy_price_book(policy.limits[0].pricing)
    encoded = json.dumps(
        pricing.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    causal_id = app.project_causal_budget_id_for_exposure(
        identity.workflow_session_id, session_ids=(identity.workflow_session_id,)
    )
    response: dict[str, object] = {
        "id": identity.public_id,
        "evidence": "not_recorded",
        "basis": "recorded_runtime_events",
        "pricing_basis": "current_configured_policy",
        "pricing_fingerprint": "sha256:" + sha256(encoded).hexdigest(),
        "currency": "USD",
        "estimated_total": None,
        "session_count": None,
        "model_steps": None,
        "observed_line_items": None,
        "unpriced_line_items": None,
        "pricing_coverage": "not_available",
        "billing_completeness": "not_established",
    }
    try:
        observed = await app.get_causal_budget_cost(causal_id, pricing, currency="USD")
    except KeyError:
        return response
    if type(observed) is not CausalBudgetCostSummary:
        raise ValueError("Invalid maintenance cost evidence.")
    # Validate attributes before serialization; malformed models must not emit
    # serializer warnings containing private provider or pricing information.
    summary = CausalBudgetCostSummary.model_validate(
        {field: getattr(observed, field) for field in CausalBudgetCostSummary.model_fields}
    )
    if (
        summary.causal_budget_id != causal_id
        or summary.currency != "USD"
        or not summary.total_cost.is_finite()
    ):
        raise ValueError("Invalid maintenance cost evidence.")
    unpriced = sum(not item.priced for item in summary.line_items)
    response.update(
        evidence="recorded",
        estimated_total=str(summary.total_cost),
        session_count=summary.session_count,
        model_steps=summary.model_steps,
        observed_line_items=len(summary.line_items),
        unpriced_line_items=unpriced,
        pricing_coverage=(
            "unpriced_observations"
            if unpriced
            else "all_observed_items_priced"
            if summary.line_items
            else "no_cost_observations"
        ),
    )
    return response
