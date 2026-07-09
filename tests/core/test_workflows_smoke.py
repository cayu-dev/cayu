"""Smoke test for the workflow batteries.

Drives a tiny ``WorkflowBase`` subclass that fans out with ``parallel`` and then
chains a ``pipeline`` typed edge, using ``ScriptedModelProvider`` (no API key).
Asserts the run emits ``workflow.completed``, yields the expected typed results,
and journals per-step completions under the workflow run id (the resume
substrate).
"""

from __future__ import annotations

import asyncio

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    ScriptedModelProvider,
    WorkflowSpec,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.workflows import EventStoreJournal, WorkflowBase, parallel, pipeline, step

COUNT_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}
TOTAL_SCHEMA = {
    "type": "object",
    "properties": {"total": {"type": "integer"}},
    "required": ["total"],
    "additionalProperties": False,
}


def _submit(output: dict) -> list[ModelStreamEvent]:
    """One scripted model step that submits `output` via the structured-output tool."""
    return [
        ModelStreamEvent.tool_call(
            id="call_out",
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            arguments={"output": output},
        ),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


class DemoWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="smoke-demo")

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        fan = await parallel(
            step(
                ctx, agent="auditor", step_id=f"audit-{label}", prompt="audit", schema=COUNT_SCHEMA
            )
            for label in ("a", "b")
        )
        self.fan = fan
        final = await pipeline(
            fan,
            [
                lambda prev: step(
                    ctx, agent="synth", step_id="synth", prompt="synth", schema=TOTAL_SCHEMA
                )
            ],
        )
        self.final = final
        yield await ctx.completed({"total": final.output})


def test_workflow_parallel_pipeline_smoke():
    provider = ScriptedModelProvider(
        [
            _submit({"n": 2}),  # first parallel branch
            _submit({"n": 2}),  # second parallel branch (identical → order-agnostic)
            _submit({"total": 4}),  # pipeline synthesis stage
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="auditor", model="scripted-model"))
    app.register_agent(AgentSpec(name="synth", model="scripted-model"))

    workflow = DemoWorkflow(app)

    async def drive() -> list:
        return [event async for event in workflow.run("wf-smoke")]

    events = asyncio.run(drive())
    event_types = [event.type for event in events]

    # The run brackets itself with the reserved workflow.* events.
    assert event_types[0] == EventType.WORKFLOW_STARTED
    assert EventType.WORKFLOW_COMPLETED in event_types
    completed = events[event_types.index(EventType.WORKFLOW_COMPLETED)]
    assert completed.workflow_name == "smoke-demo"
    assert completed.payload["total"] == {"total": 4}

    # parallel → typed results, nothing dropped.
    assert workflow.fan.ok
    assert sorted(result.output["n"] for result in workflow.fan.successes) == [2, 2]

    # pipeline → the validated Python object flows through as the typed edge.
    assert workflow.final.output == {"total": 4}

    # Resume substrate: step completions are journaled under the run id.
    async def completed_step_ids() -> set[str]:
        journal = EventStoreJournal(app.session_store, "wf-smoke", "smoke-demo")
        attempt_id = await journal.latest_attempt_id()
        assert attempt_id is not None
        return await journal.completed_step_ids(attempt_id=attempt_id)

    journaled = asyncio.run(completed_step_ids())
    assert {"audit-a", "audit-b", "synth"} <= journaled
