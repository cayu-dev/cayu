"""Per-session dollar spend caps built with ``BudgetLimit.usd``.

These tests pin what an application sees at the cap: where the check happens
relative to provider dispatch, the session status and events, how a raised cap
resumes the same session, and that side effects are not replayed.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events, collect_resume_events
from tests.runtime.runtime_tool_support import SideEffectTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals import ResolutionActor, ResolutionActorSource
from cayu.budgets import BudgetLimit, BudgetReservation, default_price_book
from cayu.events import EventType
from cayu.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyRequest,
    ExecutionProfilePolicyResult,
)
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime import ExecutionProfilePolicy
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import ResumeRequest, RunRequest

# claude-sonnet-5 is priced at $2 / $10 per million input / output tokens in the
# bundled book, so 500k input tokens cost exactly $1.00.
_MODEL = "claude-sonnet-5"
_ONE_DOLLAR_OF_INPUT = 500_000


class _AnthropicFake(FakeProvider):
    name = "anthropic"


class _AdoptSpendCapChanges(ExecutionProfilePolicy):
    """The guide's example: adopt a changed cap or price book, nothing else."""

    identity = "tests:adopt-spend-cap-changes:v1"

    async def decide(self, request: ExecutionProfilePolicyRequest) -> ExecutionProfilePolicyResult:
        budget_components = {
            ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
            ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY,
            ExecutionProfileComponentClass.RUNTIME,
        }
        if set(request.changed_component_classes) <= budget_components:
            return ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="The operator changed the spend cap.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
        return ExecutionProfilePolicyResult(
            action=ExecutionProfilePolicyAction.REJECT,
            reason="Only spend-cap changes are adopted.",
        )


def _tool_call_step(call_id: str, input_tokens: int) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=call_id, name="side_effect", arguments={}),
        ModelStreamEvent.completed(
            {
                "finish_reason": "tool_calls",
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "total_tokens": input_tokens,
                },
            }
        ),
    ]


def _final_step() -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta("done"),
        ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 1_000, "output_tokens": 100, "total_tokens": 1_100},
            }
        ),
    ]


def _adoption(key: str) -> ExecutionProfileAdoptionIntent:
    return ExecutionProfileAdoptionIntent(
        idempotency_key=key,
        reason="Raise the spend cap for this session.",
        requested_by=ResolutionActor(subject="operator", source=ResolutionActorSource.REQUEST),
    )


def _application(
    provider: FakeProvider,
    tool: SideEffectTool,
    *,
    policy: ExecutionProfilePolicy | None = None,
) -> tuple[CayuApp, InMemorySessionStore]:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, execution_profile_policy=policy)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=_MODEL), tools=[tool])
    return app, store


def test_session_cap_stops_at_a_clean_boundary_without_running_pending_tools() -> None:
    async def run():
        tool = SideEffectTool()
        provider = _AnthropicFake(
            [
                _tool_call_step("call_1", _ONE_DOLLAR_OF_INPUT),
                _tool_call_step("call_2", _ONE_DOLLAR_OF_INPUT),
            ]
        )
        app, store = _application(provider, tool)
        cap = BudgetLimit.usd("1.50")

        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_spend_cap",
                messages=[Message.text("user", "work")],
                budget_limits=(cap,),
                max_steps=5,
            ),
        )
        session = await store.load("sess_spend_cap")
        # Resuming under the same cap is refused before any provider call.
        resumed = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_spend_cap",
                messages=[Message.text("user", "keep going")],
                budget_limits=(cap,),
            ),
        )
        cost = await app.get_session_cost("sess_spend_cap", default_price_book())
        return events, session, resumed, provider, tool, cost

    events, session, resumed, provider, tool, cost = asyncio.run(run())

    # The cap is checked before each model call: the second call was admitted
    # at $1.00 spent, and the stop happens after it completes, before its tool.
    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TOOL_CALL_FAILED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    reached = events[7].payload
    assert reached["limit"] == "estimated_cost"
    assert Decimal(reached["actual"]) == Decimal("2")
    assert Decimal(reached["maximum"]) == Decimal("1.50")
    assert reached["cost_summary"]["unpriced_model_steps"] == 0
    assert events[8].payload["reason"] == "limit_reached"
    assert events[-1].payload["interruption_type"] == "limit_reached"
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert len(tool.calls) == 1
    assert len(provider.requests) == 2

    assert [event.type for event in resumed] == [
        EventType.SESSION_RESUMED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert len(provider.requests) == 2

    assert cost.model_steps == 2
    assert cost.unpriced_model_steps == 0
    assert cost.total_cost == Decimal("2")


def test_raised_cap_resumes_the_session_without_replaying_side_effects() -> None:
    async def run():
        tool = SideEffectTool()
        provider = _AnthropicFake(
            [
                _tool_call_step("call_1", _ONE_DOLLAR_OF_INPUT),
                _tool_call_step("call_2", _ONE_DOLLAR_OF_INPUT),
                _final_step(),
            ]
        )
        app, store = _application(provider, tool, policy=_AdoptSpendCapChanges())
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_raised_cap",
                messages=[Message.text("user", "work")],
                budget_limits=(BudgetLimit.usd("1.50"),),
                max_steps=5,
            ),
        )
        raised = BudgetLimit.usd(5)
        # The cap is part of the session's execution profile; changing it
        # without explicit adoption is refused before any work.
        with pytest.raises(ExecutionProfileMismatchError, match="invocation_budget_policy"):
            await collect_resume_events(
                app,
                ResumeRequest(
                    session_id="sess_raised_cap",
                    messages=[Message.text("user", "continue")],
                    budget_limits=(raised,),
                ),
            )
        assert len(provider.requests) == 2
        resumed = await collect_resume_events(
            app,
            ResumeRequest(
                session_id="sess_raised_cap",
                messages=[Message.text("user", "continue")],
                budget_limits=(raised,),
                profile_adoption=_adoption("raise-cap-to-5"),
            ),
        )
        session = await store.load("sess_raised_cap")
        cost = await app.get_session_cost("sess_raised_cap", default_price_book())
        return resumed, session, provider, tool, cost

    resumed, session, provider, tool, cost = asyncio.run(run())

    assert [event.type for event in resumed] == [
        EventType.SESSION_EXECUTION_PROFILE_DECIDED,
        EventType.SESSION_RESUMED,
        EventType.MODEL_STARTED,
        EventType.MODEL_TEXT_DELTA,
        EventType.MODEL_COMPLETED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    # call_1 ran once before the cap; call_2 was skipped at the cap and is not
    # re-run on resume.
    assert len(tool.calls) == 1
    assert len(provider.requests) == 3
    assert cost.model_steps == 3
    assert cost.priced_model_steps == 3
    # $1 + $1 + (1,000 x $2 + 100 x $10) / 1M
    assert cost.total_cost == Decimal("2.003")


def test_strict_session_tree_cap_reserves_before_dispatch() -> None:
    """A causal-scope cap with a reservation never dispatches past the cap."""

    async def run():
        tool = SideEffectTool()
        provider = _AnthropicFake(
            [
                _tool_call_step("call_1", _ONE_DOLLAR_OF_INPUT // 2),
                _tool_call_step("call_2", _ONE_DOLLAR_OF_INPUT // 2),
            ]
        )
        app, store = _application(provider, tool)
        # Causal budget ids default to the session id, so this caps the session
        # and any child sessions that share its causal budget.
        cap = BudgetLimit.usd(
            1,
            scope="causal",
            key="sess_strict_cap",
            # 300k input tokens at $2/M: each call reserves $0.60 up front.
            reservation=BudgetReservation(max_input_tokens=300_000, max_output_tokens=0),
        )
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_strict_cap",
                messages=[Message.text("user", "work")],
                budget_limits=(cap,),
                max_steps=5,
            ),
        )
        session = await store.load("sess_strict_cap")
        cost = await app.get_session_cost("sess_strict_cap", default_price_book())
        return events, session, provider, tool, cost

    events, session, provider, tool, cost = asyncio.run(run())

    types = [event.type for event in events]
    assert types == [
        EventType.SESSION_STARTED,
        EventType.BUDGET_RESERVED,
        EventType.MODEL_STARTED,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_RECONCILED,
        EventType.TOOL_CALL_STARTED,
        EventType.TOOL_CALL_COMPLETED,
        EventType.BUDGET_RESERVATION_FAILED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    reserved = events[types.index(EventType.BUDGET_RESERVED)].payload
    assert Decimal(reserved["requested"]) == Decimal("0.6")
    reconciled = events[types.index(EventType.BUDGET_RECONCILED)].payload
    assert Decimal(reconciled["actual_amount"]) == Decimal("0.5")
    failed = events[types.index(EventType.BUDGET_RESERVATION_FAILED)].payload
    assert failed["accepted"] is False
    assert Decimal(failed["requested"]) == Decimal("0.6")
    assert Decimal(failed["actual"]) == Decimal("1.1")
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    # The second call was refused before dispatch, so spend stays under the cap.
    assert len(provider.requests) == 1
    assert len(tool.calls) == 1
    assert cost.total_cost == Decimal("0.5")


def test_one_line_cap_refuses_an_unpriced_model_before_dispatch() -> None:
    async def run():
        provider = _AnthropicFake([_final_step()])
        store = InMemorySessionStore()
        app = CayuApp(session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="claude-unreleased"))
        events = await collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_unpriced_cap",
                messages=[Message.text("user", "work")],
                budget_limits=(BudgetLimit.usd(5),),
            ),
        )
        return events, provider

    events, provider = asyncio.run(run())

    assert [event.type for event in events] == [
        EventType.SESSION_STARTED,
        EventType.SESSION_LIMIT_REACHED,
        EventType.TURN_COMPLETED,
        EventType.SESSION_INTERRUPTED,
    ]
    assert "no matching pricing" in events[1].payload["message"]
    assert provider.requests == []
