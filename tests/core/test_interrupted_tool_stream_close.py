"""Public stream abandonment must finish an already accepted tool interruption."""

import asyncio

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventQuery,
    ExecutionProfileBehaviorIdentity,
    InterruptSessionRequest,
    Message,
    RunRequest,
    SessionStatus,
    SQLiteSessionStore,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelProvider, ModelStreamEvent


class Provider(ModelProvider):
    name = "fixture"

    def __init__(self):
        self.requests = 0

    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="fixture:batch", behavior_version="1", implementation_version="1"
        )

    async def stream(self, request):
        self.requests += 1
        assert self.requests == 1, "Unexpected model request after interruption"
        for i in range(3):
            yield ModelStreamEvent.tool_call(id=f"call-{i}", name="record", arguments={"index": i})
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class Record(Tool):
    def __init__(self, seen):
        self.seen = seen
        super().__init__(
            ToolSpec(
                name="record",
                description="Record a fixture item",
                input_schema={
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                    "required": ["index"],
                },
                parallel_safe=False,
                effect=ToolEffect.IDEMPOTENT,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="fixture:record", behavior_version="1", implementation_version="1"
                ),
            )
        )

    async def run(self, ctx, args):
        # Scheduling the interruption may race with dispatch of the next call.
        # Keep later calls unfinished until the accepted request cancels them.
        if args["index"] > 0:
            await asyncio.Event().wait()
        self.seen.append(args["index"])
        return ToolResult(content="Recorded")


@pytest.mark.parametrize("close_after_failures", [None, 1, 2])
def test_close_during_interrupted_tool_outcomes_settles_original_request(
    tmp_path, close_after_failures
):
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        seen = []
        provider = Provider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="fixture"), tools=[Record(seen)])
        interruption = None

        async def stop():
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id="fixture", reason="requested pause")
                )
            ]

        stream = app.run(
            RunRequest(
                agent_name="worker",
                session_id="fixture",
                messages=[Message.text("user", "Record the items")],
                max_steps=2,
            )
        )
        failures = 0
        try:
            async with asyncio.timeout(15):
                try:
                    async for event in stream:
                        if str(event.type) == "tool.call.completed" and interruption is None:
                            interruption = asyncio.create_task(stop())
                        if str(event.type) == "tool.call.failed":
                            failures += 1
                            if failures == close_after_failures:
                                break
                finally:
                    await stream.aclose()
                assert interruption is not None
                receipt = await interruption
                assert await app.drain_background_interruptions(timeout_s=2)
            assert seen == [0]
            assert provider.requests == 1
            session = await store.load("fixture")
            assert session.status is SessionStatus.INTERRUPTED
            records = await store.query_events(EventQuery(session_id="fixture", limit=100))
            terminal = [r.event for r in records if str(r.event.type) == "session.interrupted"]
            assert len(terminal) == 1
            terminal_record = next(r for r in records if r.event.id == terminal[0].id)
            assert app.project_event_record_for_exposure(terminal_record).event.id == receipt[-1].id
            assert terminal[0].payload["reason"] == "requested pause"
            assert terminal[0].payload.get("abandoned") is not True
            tool_terminals = [
                r.event
                for r in records
                if str(r.event.type) in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(tool_terminals) == 3
            assert len({e.payload["tool_call_id"] for e in tool_terminals}) == 3
            assert not any(str(r.event.type) == "session.failed" for r in records)
        finally:
            if interruption is not None:
                await asyncio.gather(interruption, return_exceptions=True)
            await app.drain_background_interruptions(timeout_s=2)
            await store.close()
        # The terminal receipt and all call outcomes survive reopening the store.
        reopened = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        try:
            assert (await reopened.load("fixture")).status is SessionStatus.INTERRUPTED
            records = await reopened.query_events(EventQuery(session_id="fixture", limit=100))
            assert sum(str(r.event.type) == "session.interrupted" for r in records) == 1
            assert sum(str(r.event.type) == "tool.call.failed" for r in records) == 2
        finally:
            await reopened.close()

    asyncio.run(scenario())
