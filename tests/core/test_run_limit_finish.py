from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.run_limits import RunLimits
from cayu.configuration import CayuConfig, RunDefaults
from cayu.context.structured_output import STRUCTURED_OUTPUT_TOOL_NAME, StructuredOutputSpec
from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded
from cayu.evals.testing import ScriptedModelProvider, model_facing_text
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec


class ReadTool(Tool):
    spec = ToolSpec(
        name="read", input_schema={"type": "object", "properties": {}}, effect=ToolEffect.NONE
    )

    def __init__(self):
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls += 1
        return ToolResult(content="The answer is 42.")


def spec(strategy):
    return StructuredOutputSpec(
        strategy=strategy,
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )


def answer(strategy, valid=True):
    value = {"answer": 42} if valid else {"wrong": 42}
    if strategy == "native":
        import json

        return [
            ModelStreamEvent.text_delta(json.dumps(value)),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    return [
        ModelStreamEvent.tool_call(
            id="answer", name=STRUCTURED_OUTPUT_TOOL_NAME, arguments={"output": value}
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


@pytest.mark.parametrize("strategy", ["native", "tool"])
@pytest.mark.parametrize("valid", [True, False])
def test_final_allowed_step_withdraws_tools_and_ends_with_limit_outcome(strategy, valid):
    tool = ReadTool()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="read-1", name="read", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            answer(strategy, valid),
        ],
        supports_native_structured_output=True,
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake"), tools=[tool])

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="finish",
                    messages=[Message.text("user", "Find the answer")],
                    max_steps=2,
                    limits=RunLimits(on_limit="finish"),
                    structured_output=spec(strategy),
                )
            )
        ]
        assert tool.calls == 1
        assert len(provider.requests) == 2
        assert [entry["name"] for entry in provider.requests[-1].tools] == (
            [] if strategy == "native" else [STRUCTURED_OUTPUT_TOOL_NAME]
        )
        assert "final model step" in model_facing_text(provider.requests[-1])
        terminal = next(event for event in events if event.type is EventType.SESSION_INTERRUPTED)
        assert terminal.payload["reason"] == "limit_reached"
        assert terminal.payload["limit"] == "model_steps"
        assert terminal.payload["structured_output_pending"] is not valid
        assert terminal.payload["finish_up"] is True
        assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events) is valid
        assert (await app.session_store.load("finish")).status is SessionStatus.INTERRUPTED

    asyncio.run(scenario())


@pytest.mark.parametrize("strategy", ["native", "tool"])
def test_deadline_reserve_produces_a_structured_answer(strategy):
    provider = ScriptedModelProvider(answer(strategy), supports_native_structured_output=True)
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake"), tools=[ReadTool()])

    async def scenario():
        boundary = ExecutionDeadline.after(10)
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "Answer now")],
                    max_steps=None,
                    execution_deadline=boundary,
                    limits=RunLimits(on_limit="finish"),
                    structured_output=spec(strategy),
                )
            )
        ]
        assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
        terminal = next(event for event in events if event.type is EventType.SESSION_INTERRUPTED)
        assert terminal.payload["limit"] == "execution_deadline"
        assert terminal.payload["structured_output_pending"] is False
        assert not boundary.expired

    asyncio.run(scenario())


def test_finish_step_cannot_execute_an_ordinary_tool():
    tool = ReadTool()
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.tool_call(id="unauthorized", name="read", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake"), tools=[tool])

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "Answer")],
                    max_steps=1,
                    limits=RunLimits(on_limit="finish"),
                    structured_output=spec("tool"),
                )
            )
        ]
        assert tool.calls == 0
        assert any(event.type is EventType.SESSION_INTERRUPTED for event in events)

    asyncio.run(scenario())


@pytest.mark.parametrize("task_backed", [False, True])
def test_unlimited_run_reaches_more_than_default_steps(task_backed):
    def respond(request):
        return [
            ModelStreamEvent.text_delta("working"),
            ModelStreamEvent.completed(
                {"finish_reason": "stop", "end_turn": request.options["step"] >= 66}
            ),
        ]

    provider = ScriptedModelProvider(response_factory=respond)
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake"))

    async def scenario():
        fields = {"max_steps": None}
        if task_backed:
            await task_store.create_task(TaskCreate(task_id="task", type="test"))
            fields = {"task_id": "task"}
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "Work")],
                    execution_deadline=ExecutionDeadline.after(30),
                    **fields,
                )
            )
        ]
        assert len(provider.requests) == 66
        assert any(event.type is EventType.SESSION_COMPLETED for event in events)

    asyncio.run(scenario())


def test_explicit_application_step_cap_still_applies_to_tasks():
    app = CayuApp(config=CayuConfig(run=RunDefaults(max_steps=7)), enable_logging=False)
    request = RunRequest(agent_name="assistant", task_id="task", messages=[])
    assert app._with_application_run_defaults(request).max_steps == 7


class StallProvider(ModelProvider):
    name = "stall"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        await asyncio.sleep(60)
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_admitted_deadline_expiry_ends_iterator_normally():
    app = CayuApp(enable_logging=False)
    app.register_provider(StallProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake"))

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="timed",
                    messages=[Message.text("user", "Wait")],
                    execution_deadline=ExecutionDeadline.after(0.5),
                )
            )
        ]
        terminal = next(event for event in events if event.type is EventType.SESSION_INTERRUPTED)
        assert terminal.payload["reason"] == "limit_reached"
        assert terminal.payload["limit"] == "execution_deadline"
        assert terminal.payload["structured_output_pending"] is False
        assert terminal.payload["failure_evidence"]["classification"] == "deadline"
        with pytest.raises(ExecutionDeadlineExceeded):
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        messages=[],
                        execution_deadline=ExecutionDeadline.after(0),
                    )
                )
            ]

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["input_tokens", "tool_calls", "estimated_cost"])
@pytest.mark.parametrize("scope", ["run", "session"])
def test_observed_limits_reserve_answer_and_publish_it_before_interrupting(kind, scope):
    from decimal import Decimal

    from cayu.budgets.base import BudgetLimit
    from cayu.budgets.pricing import ModelPrice, PriceBook

    tool = ReadTool()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="first", name="read", arguments={}),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {"input_tokens": 90, "output_tokens": 0, "total_tokens": 90},
                    }
                ),
            ],
            [
                ModelStreamEvent.text_delta('{"answer":42}'),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {"input_tokens": 20, "output_tokens": 0, "total_tokens": 20},
                    }
                ),
            ],
        ],
        supports_native_structured_output=True,
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake"), tools=[tool])
    limits = RunLimits(
        on_limit="finish",
        scope=scope,
        **(
            {"max_input_tokens": 100}
            if kind == "input_tokens"
            else {"max_tool_calls": 1}
            if kind == "tool_calls"
            else {}
        ),
    )
    budgets = (
        ()
        if kind != "estimated_cost"
        else (
            BudgetLimit(
                scope=scope,
                max_estimated_cost=Decimal("1"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="scripted",
                            model="fake",
                            input_per_million=Decimal("10000"),
                            output_per_million=Decimal("0"),
                        ),
                    )
                ),
            ),
        )
    )

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "Answer")],
                    max_steps=None,
                    limits=limits,
                    budget_limits=budgets,
                    structured_output=spec("native"),
                )
            )
        ]
        assert len(provider.requests) == 2
        assert tool.calls == 1
        validated = next(
            event for event in events if event.type is EventType.STRUCTURED_OUTPUT_VALIDATED
        )
        terminal = next(event for event in events if event.type is EventType.SESSION_INTERRUPTED)
        assert events.index(validated) < events.index(terminal)
        assert terminal.payload["limit"] == kind
        assert terminal.payload["structured_output_pending"] is False

    asyncio.run(scenario())


@pytest.mark.parametrize("discovery", [None, "search_tools"])
def test_finish_withdraws_stable_catalogue_and_discovery(discovery):
    provider = ScriptedModelProvider(answer("tool"))
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        tools=[ReadTool()],
        tool_exposure_mode="stable_catalogue" if discovery is None else "filtered_tools",
        tool_discovery_mode=discovery,
    )

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    messages=[Message.text("user", "Answer")],
                    max_steps=1,
                    limits=RunLimits(on_limit="finish"),
                    structured_output=spec("tool"),
                )
            )
        ]
        assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
        assert [tool["name"] for tool in provider.requests[0].tools] == [
            STRUCTURED_OUTPUT_TOOL_NAME
        ]

    asyncio.run(scenario())


def test_unlimited_approval_continuation_keeps_explicit_authority():
    from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy

    tool = ReadTool()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="read-paused", name="read", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        tools=[tool],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="approval-unlimited",
                    messages=[Message.text("user", "Read")],
                    max_steps=None,
                )
            )
        ]
        approval = next(
            event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        checkpoint = await app.session_store.load_checkpoint("approval-unlimited")
        assert checkpoint["pending_tool_approval"]["max_steps_unbounded"] is True
        resumed = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="approval-unlimited",
                    approval_id=approval.payload["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        assert tool.calls == 1
        assert len(provider.requests) == 2
        assert any(event.type is EventType.SESSION_COMPLETED for event in resumed)

    asyncio.run(scenario())


def test_legacy_null_pending_step_limit_does_not_gain_unlimited_authority():
    from tests.core.test_recovery_coordinator import _pending_approval

    from cayu.approvals.tools import copy_pending_tool_approval
    from cayu.runtime._recovery_coordinator import _effective_approval_max_steps

    legacy = _pending_approval()
    with pytest.raises(ValueError, match="requires recorded invocation max_steps"):
        _effective_approval_max_steps(max_steps=None, pending_approval=legacy)
    unlimited = copy_pending_tool_approval(legacy.model_copy(update={"max_steps_unbounded": True}))
    assert _effective_approval_max_steps(max_steps=None, pending_approval=unlimited) is None


@pytest.mark.parametrize("structured", [False, True])
def test_approval_continuation_deadline_ends_normally_after_admission(structured):
    from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
    from cayu.tools.policy import AlwaysRequireApprovalToolPolicy

    class WaitingTool(ReadTool):
        async def run(self, ctx, args):
            self.calls += 1
            await asyncio.sleep(60)
            return ToolResult(content="late")

    tool = WaitingTool()
    provider = ScriptedModelProvider(
        [
            ModelStreamEvent.tool_call(id="waiting", name="read", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        tools=[tool],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="deadline-approval",
                    messages=[Message.text("user", "Read")],
                    max_steps=None,
                    structured_output=spec("tool") if structured else None,
                    execution_deadline=ExecutionDeadline.after(2),
                )
            )
        ]
        approval = next(
            event for event in events if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        resumed = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="deadline-approval",
                    approval_id=approval.payload["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        terminal = next(event for event in resumed if event.type is EventType.SESSION_INTERRUPTED)
        assert terminal.payload["reason"] == "limit_reached"
        assert terminal.payload["limit"] == "execution_deadline"
        assert terminal.payload["structured_output_pending"] is structured
        assert tool.calls == 1
        assert len(provider.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("valid", [True, False])
def test_finish_round_recovers_lost_publication_acknowledgement_without_another_step(valid):
    from tests.core.test_structured_output_tool_round_recovery import (
        _ToolPublicationAcknowledgementLostStore,
    )

    store = _ToolPublicationAcknowledgementLostStore()
    provider = ScriptedModelProvider(answer("tool", valid))
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake"), tools=[ReadTool()])

    async def scenario():
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="finish-ack-loss",
                    messages=[Message.text("user", "Answer")],
                    max_steps=1,
                    limits=RunLimits(on_limit="finish"),
                    structured_output=spec("tool"),
                )
            )
        ]
        assert store.lost_acknowledgement
        assert store.tool_publication_calls == 2
        assert len(provider.requests) == 1
        assert not any(event.type is EventType.STRUCTURED_OUTPUT_RETRY for event in events)
        terminal = next(event for event in events if event.type is EventType.SESSION_INTERRUPTED)
        assert terminal.payload["structured_output_pending"] is not valid
        stored = await store.load_events("finish-ack-loss")
        assert sum(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in stored) == int(
            valid
        )
        assert sum(event.type is EventType.STRUCTURED_OUTPUT_FAILED for event in stored) == int(
            not valid
        )

    asyncio.run(scenario())
