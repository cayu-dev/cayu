"""An approval's effective budgets must also govern inference inside the approved tool."""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest
from tests.core.test_tool_approval_profile_adoption import (
    _AdoptExplicitRequests,
    _AskFirst,
    _durable_store,
    _intent,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.approvals.tools import ToolApprovalRequest
from cayu.budgets.base import BudgetLimit, BudgetReservation
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.tools.inference import AuxiliaryInferencePolicy, InferenceLimits

_BOUNDS = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=10)


class _Summarize(Tool):
    def __init__(self) -> None:
        self.results: list[str] = []
        self.refusals: list[str] = []
        self.spec = ToolSpec(
            name="summarize",
            description="Summarize using auxiliary inference.",
            input_schema={"type": "object"},
            max_terminal_payload_bytes=1024,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:adopted-budget-summary",
                behavior_version="1",
                implementation_version="1",
            ),
            auxiliary_inference=AuxiliaryInferencePolicy(
                limits=_BOUNDS, purposes=("tool.summary",)
            ),
        )
        super().__init__()

    async def run(self, ctx, args):
        try:
            response = await ctx.inference.invoke(
                ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                purpose="tool.summary",
                limits=_BOUNDS,
            )
        except (RuntimeError, ValueError) as exc:
            self.refusals.append(str(exc))
            return ToolResult(content="budget refused", is_error=True)
        self.results.append(response.text)
        return ToolResult(content=response.text)


def _budget(max_estimated_cost: str, *, scope: str, action: str = "interrupt") -> BudgetLimit:
    return BudgetLimit(
        scope=scope,
        action=action,
        max_estimated_cost=Decimal(max_estimated_cost),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        ),
        # Only shared scopes carry reservation envelopes.
        reservation=(
            BudgetReservation(max_input_tokens=10, max_output_tokens=10) if scope == "app" else None
        ),
    )


async def _approve_summary(store, *, initial, approved, adopt):
    tool = _Summarize()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(name="summarize", id="summary", arguments={}),
                ModelStreamEvent.completed({"usage": {"input_tokens": 6, "output_tokens": 6}}),
            ],
            [
                ModelStreamEvent.text_delta("summary"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 10, "output_tokens": 10}}),
            ],
            [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
        ],
        name="fake",
    )
    app = CayuApp(
        session_store=store,
        execution_profile_policy=_AdoptExplicitRequests(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="model"), tools=[tool], tool_policy=_AskFirst()
    )
    try:
        paused = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="summary",
                    agent_name="assistant",
                    messages=[Message.text("user", "summarize")],
                    budget_limits=initial,
                )
            )
        ]
        approval = next(
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        resumed = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="summary",
                    approval_id=approval.payload["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision="approve",
                    budget_limits=approved,
                    profile_adoption=_intent("change-budget") if adopt else None,
                )
            )
        ]
    finally:
        await app.aclose()
    assert (
        any(
            event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
            and event.payload["decision"] == "adopted"
            for event in resumed
        )
        is adopt
    )
    return tool, provider, resumed


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_adopted_shared_cap_reserves_before_auxiliary_dispatch(backend, tmp_path):
    async def scenario():
        store = _durable_store(backend, tmp_path, None)
        # One twenty-token auxiliary envelope does not fit the adopted cap, so
        # the tool's inference must be refused before it reaches the provider.
        cap = _budget("0.000015", scope="app")
        try:
            tool, provider, resumed = await _approve_summary(
                store, initial=(), approved=(cap,), adopt=True
            )
        finally:
            if backend == "sqlite":
                await store.close()
        assert tool.results == [], {
            "provider_requests": len(provider.requests),
            "terminal": resumed[-1].type,
        }
        assert tool.refusals == ["Auxiliary budget reservation was refused."]
        assert len(provider.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("adopt", [False, True])
def test_auxiliary_inference_uses_the_continuation_run_budget_identity(adopt, backend, tmp_path):
    async def scenario():
        store = _durable_store(backend, tmp_path, None)
        initial = _budget("0.000050", scope="run", action="notify")
        # A raised cap is a new identity; the continuation rebases its run
        # accounting to it, and the tool must account under the same one.
        approved = _budget("0.000060", scope="run", action="notify") if adopt else initial
        try:
            tool, provider, resumed = await _approve_summary(
                store, initial=(initial,), approved=(approved,), adopt=adopt
            )
        finally:
            if backend == "sqlite":
                await store.close()
        assert tool.results == ["summary"], {"refusals": tool.refusals}
        assert len(provider.requests) == 3
        assert resumed[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action", ["notify", "interrupt"])
@pytest.mark.parametrize("adopt", [False, True])
def test_first_run_budget_adopted_at_approval_governs_auxiliary_inference(
    adopt, action, backend, tmp_path
):
    async def scenario():
        store = _durable_store(backend, tmp_path, None)
        budget = _budget("0.000050", scope="run", action=action)
        # When the pause had no run budget there is no accounting to carry
        # over; the adopted one must behave as if it had been there all along.
        try:
            tool, provider, resumed = await _approve_summary(
                store, initial=() if adopt else (budget,), approved=(budget,), adopt=adopt
            )
        finally:
            if backend == "sqlite":
                await store.close()
        if action == "notify":
            assert tool.results == ["summary"], {"refusals": tool.refusals}
            assert len(provider.requests) == 3
        else:
            # Run scope carries no reservation envelope, so a hard run cap
            # refuses auxiliary inference before dispatch.
            assert tool.results == []
            assert tool.refusals == [
                "Auxiliary hard budgets require a configured reservation envelope."
            ]
            assert len(provider.requests) == 2
        assert resumed[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(scenario())
