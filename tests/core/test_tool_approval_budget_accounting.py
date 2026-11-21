"""Budget adoption preserves unchanged run authorities and their prior spending."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from tests.core.test_tool_approval_profile_adoption import (
    _AdoptExplicitRequests,
    _app,
    _approve,
    _collect,
    _durable_store,
    _FileBug,
    _intent,
    _Reporter,
)
from tests.runtime.runtime_budget_support import fake_budget_limit

import cayu.runtime._recovery_coordinator as recovery_coordinator
from cayu.budgets._run_limit_accounting import run_budget_authorities_from_context
from cayu.budgets.base import request_budget_limits_for_session
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._run_limits import RunLimitController
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions.requests import RunRequest


class _PricedReporter(_Reporter):
    @staticmethod
    def _respond(request):
        return [
            *_Reporter._respond(request)[:-1],
            ModelStreamEvent.completed(
                {"usage": {"input_tokens": 250_000, "output_tokens": 0, "total_tokens": 250_000}}
            ),
        ]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("changed_scope", ["session", "run"])
def test_adoption_preserves_unchanged_run_budget_origins(
    monkeypatch, tmp_path, backend, changed_scope
):
    unchanged_costs = []
    cost_for_budget = RunLimitController._cost_for_budget

    async def capture_cost(self, **kwargs):
        result = await cost_for_budget(self, **kwargs)
        limit = kwargs["limit"]
        if limit.scope == "run" and limit.max_estimated_cost == Decimal("10"):
            unchanged_costs.append(result.total_cost)
        return result

    monkeypatch.setattr(RunLimitController, "_cost_for_budget", capture_cost)
    monkeypatch.setattr("tests.core.test_tool_approval_profile_adoption._Reporter", _PricedReporter)
    rebased = []
    rebase = recovery_coordinator.rebase_run_limit_accounting_context

    def capture_rebase(*args, **kwargs):
        result = rebase(*args, **kwargs)
        rebased.append(result)
        return result

    monkeypatch.setattr(recovery_coordinator, "rebase_run_limit_accounting_context", capture_rebase)

    async def exercise():
        store = _durable_store(backend, tmp_path, None)
        tool = _FileBug()
        app = _app(store, tool=tool, profile_policy=_AdoptExplicitRequests())
        unchanged = fake_budget_limit("10", scope="run", allow_unpriced=True)
        original = (unchanged, fake_budget_limit("1", scope=changed_scope, allow_unpriced=True))
        # Also reorder the limits: identities, rather than positions, own origins.
        adopted = (fake_budget_limit("2", scope=changed_scope, allow_unpriced=True), unchanged)
        try:
            paused = await _collect(
                app.run(
                    RunRequest(
                        agent_name="reporter",
                        session_id="bug-report",
                        messages=[Message.text("user", "Save crashes the editor.")],
                        budget_limits=original,
                    )
                )
            )
            checkpoint = await store.load_checkpoint("bug-report")
            pending = pending_approval_reader.pending_approval_from_checkpoint(checkpoint)
            assert pending is not None and pending.run_limit_accounting is not None
            original_authorities = pending.run_limit_accounting.run_budget_authorities
            requested = next(
                event.payload
                for event in paused
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            resolution = _approve(
                requested, profile_adoption=_intent("adopt-budget"), budget_limits=adopted
            )
            unchanged_costs.clear()
            continued = []
            intent = None
            async for event in app.resolve_tool_approval(resolution):
                continued.append(event)
                if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED:
                    checkpoint = await store.load_checkpoint("bug-report")
                    intent = pending_approval_reader.approval_resolution_intent_from_checkpoint(
                        checkpoint
                    )
            assert intent is not None and intent.pause_resolved_at is not None
            assert continued[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == [{"title": "Crash on save"}]
            assert unchanged_costs[0] == Decimal("0.25")
            assert unchanged_costs[-1] == Decimal("0.50")
            [accounting] = rebased
            assert accounting.baseline == pending.run_limit_accounting.baseline
            effective = request_budget_limits_for_session(
                limits=adopted, agent_name="reporter", causal_budget_id="bug-report"
            )
            authorities = run_budget_authorities_from_context(accounting, budget_limits=effective)
            # The unchanged run budget must still include all pre-approval spending.
            unchanged_authority = original_authorities[0]
            assert authorities[unchanged_authority.budget_limit_id] == unchanged_authority
            if changed_scope == "run":
                assert original_authorities[1].budget_limit_id not in authorities
                [new_authority] = [
                    value
                    for key, value in authorities.items()
                    if key != unchanged_authority.budget_limit_id
                ]
                assert new_authority.started_at == intent.pause_resolved_at
            else:
                assert tuple(authorities.values()) == (unchanged_authority,)
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(exercise())
