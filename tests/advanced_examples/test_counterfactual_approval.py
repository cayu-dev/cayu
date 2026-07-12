from __future__ import annotations

import asyncio
from pathlib import Path

from examples.counterfactual_approval.deterministic import run


def test_counterfactual_approval_uses_authority_free_futures_before_one_mutation(
    tmp_path: Path,
) -> None:
    result = asyncio.run(run(tmp_path))

    assert result.status == "verified"
    assert result.assertions == {
        "analyses_are_authority_free": True,
        "approval_recovery_preserved_receipt": True,
        "approval_recovered_after_runtime_restart": True,
        "decision_brief_complete": True,
        "external_state_revalidated": True,
        "exactly_one_mutation": True,
        "losing_future_discarded": True,
        "stale_snapshot_rejected": True,
        "verifier_confirmed_actual_state": True,
    }
    assert result.metrics["mutation_count"] == 1
    assert result.metrics["retained_future"] == "approve"
    assert {session.role for session in result.sessions} == {
        "approval-request",
        "approve-future",
        "deny-future",
        "explainer",
        "verifier",
        "stale-approval-probe",
    }
    primary = next(session for session in result.sessions if session.role == "approval-request")
    assert primary.recovery_state == "manually-reconciled"
    assert result.outputs["deployment_receipt"] == {
        "service": "payments",
        "release": "2026.07.11",
        "version": 8,
        "mutation_count": 1,
    }
