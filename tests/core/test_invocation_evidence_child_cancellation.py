"""An independently cancelled evidence child cannot cancel the run's caller."""

import asyncio

import pytest
from tests.core.test_workspace_mutation_receipts import _portable_environment_spec

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.environments import Environment
from cayu.evals import ScriptedModelProvider
from cayu.observability.events import EventSink
from cayu.providers import ModelStreamEvent
from cayu.runners.base import ExecCommand
from cayu.runners.local import LocalRunner
from cayu.sessions import InMemorySessionStore
from cayu.tools.base import ToolEffect


@pytest.mark.parametrize("cancel_evidence_child", [False, True])
def test_evidence_child_cancellation_is_not_caller_cancellation(tmp_path, cancel_evidence_child):
    async def scenario():
        cancelled = []
        tool_returned = []

        class CancelledDelivery(EventSink):
            async def emit(self, event):
                if (
                    cancel_evidence_child
                    and event.type is EventType.RUNNER_EXEC_COMPLETED
                    and not cancelled
                ):
                    cancelled.append(True)

                    # A cancelled sink-owned request, not a cancellation request
                    # on either the evidence publisher or the run's consumer.
                    async def request():
                        asyncio.current_task().cancel("sink request cancelled itself")
                        await asyncio.sleep(0)

                    await asyncio.create_task(request())

        class Inspect(Tool):
            spec = ToolSpec(name="inspect", effect=ToolEffect.NONE)

            async def run(self, ctx, args):
                await ctx.runner.exec(ExecCommand(argv=["true"]))
                tool_returned.append(True)
                return ToolResult(content="inspection complete")

        store = InMemorySessionStore()
        app = CayuApp(session_store=store, event_sinks=[CancelledDelivery()], enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id="inspect-1", name="inspect", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                _portable_environment_spec("local"),
                runner=LocalRunner(tmp_path),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[Inspect()])

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="evidence-child",
                        agent_name="worker",
                        messages=[Message.text("user", "inspect")],
                    )
                )
            ]

        caller = asyncio.create_task(consume())
        try:
            try:
                events = await asyncio.wait_for(caller, 15)
            except asyncio.CancelledError:
                assert tool_returned == [True]
                assert cancelled == [True]
                assert caller.cancelling() == 0
                pytest.fail(
                    "Evidence child cancellation escaped as unrequested caller cancellation"
                )
            assert tool_returned == [True]
            assert caller.cancelling() == 0 and not caller.cancelled()
            if cancel_evidence_child:
                # The operational failure may fail the run, or a best-effort
                # sink boundary may contain it. Neither is caller cancellation.
                assert events[-1].type in {EventType.SESSION_FAILED, EventType.SESSION_COMPLETED}
            else:
                assert events[-1].type is EventType.SESSION_COMPLETED
        finally:
            if not caller.done():
                caller.cancel()
            await asyncio.gather(caller, return_exceptions=True)
            await app.aclose()

    asyncio.run(scenario())
