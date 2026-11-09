"""An operator stop remains authoritative when tool-round closure fails."""

from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec


@pytest.mark.parametrize("fail_closure", [False, True], ids=["control", "storage-error"])
def test_remote_interrupt_survives_round_closure_storage_failure(fail_closure):
    asyncio.run(_scenario(fail_closure))


async def _scenario(fail_closure):
    class ClosureFailureStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self):
            super().__init__()
            self.failed_closure = False

        async def append_events(self, session_id, events):
            if (
                fail_closure
                and not self.failed_closure
                and any(
                    event.type is EventType.TOOL_CALL_FAILED
                    and event.payload.get("interrupted") is True
                    for event in events
                )
            ):
                self.failed_closure = True
                raise OSError("interrupted terminal write failed")
            return await super().append_events(session_id, events)

    started = asyncio.Event()
    release = asyncio.Event()

    class WaitTool(Tool):
        spec = ToolSpec(
            name="wait",
            effect=ToolEffect.NONE,
            parallel_safe=False,
            max_terminal_payload_bytes=64 * 1024,
        )

        async def run(self, ctx, args):
            started.set()
            await release.wait()
            return ToolResult(content="completed")

    store = ClosureFailureStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="first", name="wait", arguments={}),
                ModelStreamEvent.tool_call(id="second", name="wait", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    owner = CayuApp(session_store=store, enable_logging=False)
    # This app shares durable storage but cannot cancel the owner's local task.
    # Its request is observed cooperatively as SessionInterruptedByRequest.
    interrupter = CayuApp(session_store=store, enable_logging=False)
    for app in (owner, interrupter):
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[WaitTool()])
    run_events = []
    session_id = "round-closure-storage-failure"

    async def execute():
        async for event in owner.run(
            RunRequest(
                session_id=session_id,
                agent_name="assistant",
                messages=[Message.text("user", "run")],
            )
        ):
            run_events.append(event)

    async def interrupt():
        return [
            event
            async for event in interrupter.interrupt_session(
                InterruptSessionRequest(session_id=session_id)
            )
        ]

    task = asyncio.create_task(execute())
    interrupt_task = None
    try:
        async with asyncio.timeout(15):
            await started.wait()
            interrupt_task = asyncio.create_task(interrupt())
            while (await store.load(session_id)).status is not SessionStatus.INTERRUPTING:
                await asyncio.sleep(0.01)
            release.set()
            await task
            assert store.failed_closure is fail_closure
            assert not any(event.type is EventType.SESSION_FAILED for event in run_events), [
                event.payload for event in run_events if event.type is EventType.SESSION_FAILED
            ]
            await interrupt_task
        assert (await store.load(session_id)).status is SessionStatus.INTERRUPTED
    finally:
        tasks = [pending for pending in (task, interrupt_task) if pending is not None]
        for pending in tasks:
            if not pending.done():
                pending.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
