from __future__ import annotations

import asyncio

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.configuration import MAX_STEPS, RunDefaults
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._model_step_executor import ModelCompletionRecoveryContext
from cayu.runtime.work_attempt_semantics import WorkAttemptRunSemantics
from cayu.sessions.base import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workflows.workflow import StepRunOptions


@pytest.mark.parametrize("max_steps", [257, 10000, MAX_STEPS])
@pytest.mark.parametrize(
    "model", [RunDefaults, StepRunOptions, ModelCompletionRecoveryContext, WorkAttemptRunSemantics]
)
def test_large_step_allowance_json_round_trip(model, max_steps):
    original = model(max_steps=max_steps)
    assert model.model_validate_json(original.model_dump_json()).max_steps == max_steps


class _Tick(Tool):
    spec = ToolSpec(
        name="tick",
        description="Advance one step.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="ok")


@pytest.mark.parametrize("max_steps,completed", [(257, False), (258, True)])
def test_deterministic_run_beyond_256_steps(max_steps, completed, tmp_path):
    async def run():
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="tick", arguments={}, id=f"tick-{index}"),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
                for index in range(257)
            ]
            + [
                [
                    ModelStreamEvent.text_delta("finished"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ],
        )
        store = SQLiteSessionStore(tmp_path / "long-run.sqlite")
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="scripted"), tools=[_Tick()])
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        messages=[Message.text("user", "Count.")],
                        max_steps=max_steps,
                    )
                )
            ]
            assert len(provider.requests) == (258 if completed else 257)
            assert sum(event.type == EventType.MODEL_COMPLETED for event in events) == len(
                provider.requests
            )
            assert events[-1].type == (
                EventType.SESSION_COMPLETED if completed else EventType.SESSION_INTERRUPTED
            )
        finally:
            await store.close()

    asyncio.run(run())
