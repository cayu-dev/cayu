from __future__ import annotations

import asyncio
from pathlib import Path

from examples.counterfactual_approval.deterministic import run
from examples.counterfactual_approval.scenario import DeploymentState, DeployServiceTool

from cayu import ToolEffect


def test_counterfactual_deployment_effect_matches_its_stable_receipt_contract() -> None:
    state = DeploymentState()
    first = state.deploy(
        service="payments",
        release="2026.07.11",
        expected_version=7,
        idempotency_key="stable-deployment-key",
    )
    replay = state.deploy(
        service="payments",
        release="2026.07.11",
        expected_version=7,
        idempotency_key="stable-deployment-key",
    )

    assert DeployServiceTool.spec.effect is ToolEffect.IDEMPOTENT
    assert first.is_error is False
    assert replay.structured is not None
    assert replay.structured["receipt_reused"] is True
    assert state.mutation_count == 1


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
