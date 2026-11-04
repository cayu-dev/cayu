"""A sibling storage failure must not erase the cancelled call's cleanup evidence."""

from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import ToolResultPart
from cayu.providers.base import ModelStreamEvent
from cayu.runners.base import ExecCommand, ExecResult, Runner, attach_cancellation_artifacts
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec

CLEANUP = {
    "type": "cayu.runner_cleanup.v1",
    "adapter": "lambda-microvm",
    "action": "kill_command",
    "status": "failed",
    "timeout_s": 5.0,
}


@pytest.mark.parametrize("closure_fails", [False, True])
def test_parallel_sibling_failure_preserves_cleanup_artifact_attribution(
    monkeypatch, closure_fails
):
    async def scenario():
        started = asyncio.Event()

        class CleanupRunner(Runner):
            async def exec(self, command: ExecCommand, **kwargs) -> ExecResult:
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as cancellation:
                    attach_cancellation_artifacts(cancellation, [CLEANUP])
                    raise BaseExceptionGroup(
                        "runner cleanup failed",
                        [cancellation, RuntimeError("kill failed")],
                    ) from None
                return ExecResult()

        class BlockTool(Tool):
            spec = ToolSpec(
                name="block",
                input_schema={"type": "object", "properties": {}},
                effect=ToolEffect.NONE,
            )

            async def run(self, ctx, args):
                try:
                    await ctx.runner.exec(ExecCommand.process("blocked"))
                except asyncio.CancelledError as cancellation:
                    raise BaseExceptionGroup("tool cleanup", [cancellation]) from None
                return ToolResult(content="unexpected")

        class SiblingTool(Tool):
            spec = ToolSpec(
                name="sibling",
                input_schema={"type": "object", "properties": {}},
                effect=ToolEffect.NONE,
            )

            async def run(self, ctx, args):
                return ToolResult(content="unexpected")

        from cayu._exception_groups import iter_exception_tree
        from cayu.runtime import _invocation_secrets as invocation_secrets
        from cayu.vaults import SecretRedactor

        app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor("synthetic-tool-secret"))
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id="blocked-call", name="block", arguments={}),
                        ModelStreamEvent.tool_call(id="sibling-call", name="sibling", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), runner=CleanupRunner()), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"),
            tools=[BlockTool(), SiblingTool()],
        )
        writer = app._tool_round_executor._event_writer
        original_emit = writer.emit

        async def fail_sibling_start(event):
            if event.type is EventType.TOOL_CALL_STARTED and event.tool_name == "sibling":
                await started.wait()
                raise OSError("temporary event-store failure")
            return await original_emit(event)

        monkeypatch.setattr(writer, "emit", fail_sibling_start)
        if closure_fails:

            async def fail_closure(_request):
                raise OSError("closure unavailable")
                yield  # pragma: no cover

            monkeypatch.setattr(app._tool_round_executor, "_close_interrupted_round", fail_closure)
        escaped = None
        try:
            async with asyncio.timeout(15):
                async for _ in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="sibling-cleanup",
                        messages=[Message.text("user", "run both")],
                    )
                ):
                    pass
        except BaseExceptionGroup as failure:
            escaped = failure
        assert escaped is not None
        cancellations = [
            node
            for node in iter_exception_tree(escaped)
            if isinstance(node, asyncio.CancelledError)
        ]
        assert cancellations
        for cancellation in cancellations:
            assert not invocation_secrets.has_cancellation_evidence(cancellation)
            assert invocation_secrets.cancellation_redactors_by_id(cancellation) is None
        if closure_fails:
            await app.aclose()
            return
        transcript = await app.session_store.load_transcript("sibling-cleanup")
        results = {
            part.tool_call_id: part
            for message in transcript
            for part in message.content
            if isinstance(part, ToolResultPart)
        }
        assert set(results) == {"blocked-call", "sibling-call"}
        assert results["sibling-call"].artifacts == []
        assert results["blocked-call"].artifacts == [CLEANUP]
        await app.aclose()

    asyncio.run(scenario())
