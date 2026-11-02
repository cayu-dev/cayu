"""Replaying a recorded reconciliation observation rechecks current review permission (#3206)."""

import asyncio

import pytest
from tests.core.test_human_review_recovery import (
    RecordedReviewEffectReconciler,
    interrupted_decision,
    receipt_recovery_request,
)

from cayu.approvals.review import HumanReviewDenied
from cayu.approvals.tools import ToolApprovalRecoveryOutcome, ToolApprovalRecoveryRequest
from cayu.approvals.user_input import UserInputRecoveryRequest
from cayu.events import EventType
from cayu.runtime.tool_effects import ToolEffectReconciliationResult


@pytest.mark.parametrize("approval", [False, True])
def test_observation_replay_rechecks_review_permission(monkeypatch, approval):
    async def observe(self, *, context, receipt):
        return ToolEffectReconciliationResult(outcome="not_found", observation="outcome_unknown")

    monkeypatch.setattr(RecordedReviewEffectReconciler, "reconcile", observe)

    async def scenario():
        app, store, policy, tool, original, view = await interrupted_decision(approval=approval)
        common = {
            "session_id": view.session_id,
            "tool_call_id": "call_first",
            "outcome": ToolApprovalRecoveryOutcome.COMPLETED,
            "message": "Completed externally.",
            "review_reference": view.reference,
        }
        if approval:
            recovery = ToolApprovalRecoveryRequest(
                **common,
                approval_id=view.interaction_id,
                tool_round_id=view.tool_round_id,
            )
        else:
            recovery = UserInputRecoveryRequest(
                **common,
                input_id=view.interaction_id,
                answer=original.answer,
                answer_review_reference=original.review_reference,
            )
        request = await receipt_recovery_request(app, recovery)
        events = [event async for event in app.reconcile_tool_effect(request)]
        assert any(event.type is EventType.TOOL_EFFECT_RECONCILIATION_OBSERVED for event in events)
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        before = await store.load_events("review-session")
        policy.can_decide = False
        with pytest.raises(HumanReviewDenied):
            _ = [event async for event in app.reconcile_tool_effect(request)]
        assert await store.load_events("review-session") == before
        assert tool.calls == [{"value": "first"}]

    asyncio.run(scenario())
