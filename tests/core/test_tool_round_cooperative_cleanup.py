"""Cooperative remote interruption retains the sibling's cancellation receipts."""

from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.runners.base import ExecCommand, ExecResult, Runner, attach_cancellation_artifacts
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import InterruptSessionRequest
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec

CLEANUP = {
    "type": "cayu.runner_cleanup.v1",
    "adapter": "lambda-microvm",
    "action": "kill_command",
    "status": "completed",
    "timeout_s": 5.0,
}


@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
def test_cooperative_interrupt_keeps_cancelled_sibling_cleanup(remote):
    asyncio.run(_scenario(remote))


async def _scenario(remote):
    running = asyncio.Event()
    sibling_running = asyncio.Event()
    release_sibling = asyncio.Event()
    cleanups = []

    class CleanupRunner(Runner):
        async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
            running.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as cancellation:
                cleanups.append(CLEANUP)
                attach_cancellation_artifacts(cancellation, [CLEANUP])
                raise
            return ExecResult()

    class BlockingTool(Tool):
        spec = ToolSpec(name="block", effect=ToolEffect.NONE, max_terminal_payload_bytes=64 * 1024)

        async def run(self, ctx, args):
            await ctx.runner.exec(ExecCommand.process("blocked"))
            return ToolResult(content="unexpected")

    class SiblingTool(Tool):
        spec = ToolSpec(
            name="sibling", effect=ToolEffect.NONE, max_terminal_payload_bytes=64 * 1024
        )

        async def run(self, ctx, args):
            sibling_running.set()
            await release_sibling.wait()
            return ToolResult(content="sibling completed")

    store = InMemorySessionStore()
    owner = CayuApp(session_store=store, enable_logging=False)
    interrupter = CayuApp(session_store=store, enable_logging=False) if remote else owner
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="block-call", name="block", arguments={}),
                ModelStreamEvent.tool_call(id="sibling-call", name="sibling", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        ]
    )
    for app in (owner, interrupter) if remote else (owner,):
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), runner=CleanupRunner()), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[BlockingTool(), SiblingTool()],
        )
    session_id = "cooperative-sibling-cleanup"

    async def execute():
        return [
            event
            async for event in owner.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "run both")],
                )
            )
        ]

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
            await running.wait()
            await sibling_running.wait()
            interrupt_task = asyncio.create_task(interrupt())
            while (await store.load(session_id)).status is not SessionStatus.INTERRUPTING:
                if interrupt_task.done():
                    await interrupt_task
                    raise AssertionError("Interruption ended before status changed")
                await asyncio.sleep(0)
            if remote:
                release_sibling.set()
            events = await task
            await interrupt_task
        assert events[-1].type is EventType.SESSION_INTERRUPTED
        assert cleanups == [CLEANUP]
        results = {
            part.tool_call_id: part
            for message in await store.load_transcript(session_id)
            for part in message.content
            if isinstance(part, ToolResultPart)
        }
        assert set(results) == {"block-call", "sibling-call"}
        if remote:
            assert results["sibling-call"].content == "sibling completed"
            assert results["sibling-call"].is_error is False
        assert results["sibling-call"].artifacts == []
        assert results["block-call"].artifacts == [CLEANUP]
    finally:
        pending = [t for t in (task, interrupt_task) if t is not None]
        for t in pending:
            if not t.done():
                t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if interrupter is not owner:
            await interrupter.aclose()
        await owner.aclose()
