"""Native bridge controls keep budgets and caller liveness through approvals."""

import asyncio
from contextlib import suppress

import pytest
from tests.core.test_external_agents import ScriptedAdapter, pricing, request

from cayu import CayuApp, RunRequest
from cayu.approvals import PendingToolApprovalEventView
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.budgets import BudgetLimit, BudgetPolicy, default_price_book
from cayu.deadlines import ExecutionDeadline
from cayu.events import EventType
from cayu.external_agents import ExternalAgentEvent, run_external_agent
from cayu.external_agents.recording import ExternalAgentRecorder
from cayu.external_agents.tools import (
    CayuToolBridge,
    ExternalPermissionTool,
    external_tool_price_book,
)
from cayu.tasks import InMemoryTaskStore, TaskCreate
from cayu.tools.policy import AllowAllToolPolicy, AlwaysRequireApprovalToolPolicy


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("scope", ["app", "causal", "policy"])
def test_native_bridge_reports_known_zero_usage_under_budget(custom, scope):
    async def scenario():
        book = external_tool_price_book(pricing()) if custom else default_price_book()
        store = InMemoryTaskStore()
        task = await store.create_task(TaskCreate(type="external"))
        limit = BudgetLimit.usd(
            "10",
            pricing=book,
            scope="causal" if scope == "causal" else "app",
            key=task.id if scope == "causal" else None,
        )
        app = CayuApp(
            task_store=store,
            enable_logging=False,
            budget_policy=BudgetPolicy(limits=(limit,)) if scope == "policy" else None,
        )
        children = []

        class Permission(ExternalPermissionTool):
            async def run(self, ctx, args):
                children.append(ctx.session_id)
                return await super().run(ctx, args)

        bridge = CayuToolBridge(
            app, name="bridge", tools=[Permission("Read")], policy=AllowAllToolPolicy()
        )

        class Adapter(ScriptedAdapter):
            provider_name = "scripted-external" if custom else "openai"
            model = "test" if custom else "gpt-6-sol"

            async def stream(self, context):
                yield ExternalAgentEvent("usage", {"input_tokens": 1, "output_tokens": 1}, "step")
                for index in range(2):
                    allowed = await bridge.authorize(
                        "Read",
                        {},
                        str(index),
                        parent_session_id=context.session_id,
                        execution_deadline=context.execution_deadline,
                    )
                    assert allowed
                yield ExternalAgentEvent("result", {"output": "done"}, "result")

        result = await run_external_agent(
            app,
            # This checks full-catalog budget accounting, not deadline expiry.
            # The catalog can take over 20 seconds to validate on a loaded runner.
            RunRequest(
                agent_name="vendor-agent",
                messages=[],
                task_id=task.id,
                budget_limits=() if scope == "policy" else (limit,),
                execution_deadline=ExecutionDeadline.after(300),
            ),
            Adapter([]),
            prompt="review",
        )
        assert result.output == "done" and len(children) == 2
        for child in children:
            cost = await app.get_session_cost(child, pricing=book)
            assert cost.model_steps == cost.priced_model_steps == 2
            assert cost.unpriced_model_steps == 0
            assert cost.total_cost == 0
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("scope", ["session", "run"])
@pytest.mark.parametrize("kind", ["rolling", "calendar"])
def test_external_local_budget_uses_current_window(monkeypatch, scope, kind):
    from datetime import UTC, datetime, timedelta
    from decimal import Decimal

    from cayu.budgets import BudgetWindow
    from cayu.events import Event
    from cayu.external_agents.base import ExternalAgentBudgetExceeded

    async def scenario():
        clock = [datetime(2026, 10, 18, 23, 59, 59, tzinfo=UTC)]
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False, clock=lambda: clock[0])
        window = (
            BudgetWindow.rolling(seconds=60)
            if kind == "rolling"
            else BudgetWindow.calendar(period="day")
        )
        checked = []
        admitted = []
        original = ExternalAgentRecorder.check_budget

        async def capture(self):
            await original(self)
            checked.append(self)

        monkeypatch.setattr(ExternalAgentRecorder, "check_budget", capture)

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                recorder = checked[-1]
                for index in range(3):
                    if index == 1:
                        clock[0] += timedelta(seconds=61)
                    await app.emit_event(
                        Event(
                            type=EventType.MODEL_COMPLETED,
                            session_id=context.session_id,
                            agent_name="vendor-agent",
                            timestamp=clock[0] - timedelta(microseconds=1),
                            payload={
                                "provider_name": self.provider_name,
                                "model": self.model,
                                "usage": {"input_tokens": 600_000, "output_tokens": 0},
                            },
                        )
                    )
                    if index < 2:
                        await recorder.check_budget()
                        events = await app.session_store.load_events(context.session_id)
                        check = next(
                            e for e in reversed(events) if e.type is EventType.BUDGET_CHECKED
                        )
                        assert Decimal(check.payload["actual"]) == Decimal("0.6")
                    else:
                        with pytest.raises(ExternalAgentBudgetExceeded):
                            await recorder.check_budget()
                    admitted.append(index)
                yield ExternalAgentEvent("result", {"output": "checked"}, "result")

        try:
            # The final after-step guard also rejects the two current-window calls.
            with pytest.raises(ExternalAgentBudgetExceeded):
                await run_external_agent(
                    app,
                    request(
                        budget_limits=(
                            BudgetLimit.usd("1", scope=scope, pricing=pricing(), window=window),
                        )
                    ),
                    Adapter([]),
                    prompt="check windows",
                )
            assert admitted == [0, 1, 2]
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["caller_exit", "budget", "access"])
def test_approval_cannot_dispatch_after_caller_exits_during_budget_check(monkeypatch, failure):
    async def scenario():
        app = CayuApp(task_store=InMemoryTaskStore(), enable_logging=False)
        calls = []
        checked, release = asyncio.Event(), asyncio.Event()
        resolver = None
        original = ExternalAgentRecorder.check_budget

        async def delayed_check(self):
            await original(self)
            if asyncio.current_task() is resolver:
                checked.set()
                await release.wait()
                if failure == "budget":
                    from cayu.external_agents.base import ExternalAgentBudgetExceeded

                    raise ExternalAgentBudgetExceeded("budget exhausted after approval")
                if failure == "access":
                    from cayu._resource_access_errors import ResourceAccessDenied

                    raise ResourceAccessDenied()

        monkeypatch.setattr(ExternalAgentRecorder, "check_budget", delayed_check)

        class Permission(ExternalPermissionTool):
            async def run(self, ctx, args):
                calls.append(args)
                return await super().run(ctx, args)

        bridge = CayuToolBridge(
            app,
            name="bridge",
            tools=[Permission("Read")],
            policy=AlwaysRequireApprovalToolPolicy(),
        )

        class Adapter(ScriptedAdapter):
            async def stream(self, context):
                await bridge.call(
                    "Read",
                    {},
                    parent_session_id=context.session_id,
                    execution_deadline=context.execution_deadline,
                    call_id="read",
                )
                yield ExternalAgentEvent("result", {"output": "done"}, "result")

        running = asyncio.create_task(
            run_external_agent(app, request(), Adapter([]), prompt="read")
        )
        async with asyncio.timeout(10):
            while True:
                child = next(iter(bridge.active), None)
                if child is not None and await app.session_store.load(child) is not None:
                    events = await app.session_store.load_events(child)
                    pending = next(
                        (e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED),
                        None,
                    )
                    if pending is not None:
                        break
                await asyncio.sleep(0.01)
        approval = PendingToolApprovalEventView.from_event(pending)

        async def resolve():
            return [
                e
                async for e in app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=child,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]

        resolver = asyncio.create_task(resolve())
        try:
            await asyncio.wait_for(checked.wait(), 5)
            if failure == "caller_exit":
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(running, 5)
                assert not bridge.active and not bridge.invocations
            release.set()
            events = await asyncio.wait_for(resolver, 5)
            assert any(e.type is EventType.TOOL_CALL_BLOCKED for e in events)
            if failure != "caller_exit":
                assert (await asyncio.wait_for(running, 5)).output == "done"
            assert not calls and not bridge.invocations
        finally:
            release.set()
            for task in (resolver, running):
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            await app.aclose()

    asyncio.run(scenario())
