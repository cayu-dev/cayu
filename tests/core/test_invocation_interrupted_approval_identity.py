"""Interrupted workspace calls retain their approval execution identity."""

import asyncio

import pytest
from tests.core.test_workspace_mutation_receipts import (
    DeterministicWorkspaceBinding,
    _portable_environment_spec,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, Tool, ToolResult, ToolSpec
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.environments import Environment
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import _approval_support as approval_support
from cayu.sessions import InMemorySessionStore
from cayu.sessions import _pending_approval_reader as approval_reader
from cayu.sessions import _pending_tool_round_reader as round_reader
from cayu.tools.base import ToolEffect
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("cancel", [False, True])
def test_interrupted_approved_workspace_call_retains_pause_identity(tmp_path, cancel):
    asyncio.run(_scenario(tmp_path, cancel))


async def _scenario(tmp_path, cancel):
    entered = asyncio.Event()
    release = asyncio.Event()

    class Write(Tool):
        spec = ToolSpec(
            name="write",
            effect=ToolEffect.IDEMPOTENT,
            parallel_safe=False,
            workspace_mutation=True,
            input_schema={"type": "object"},
        )

        async def run(self, ctx, args):
            await ctx.workspace.write_bytes("out.txt", b"written")
            entered.set()
            await release.wait()
            return ToolResult(content="written")

    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="write-1", name="write", arguments={}),
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
            workspace=LocalWorkspace(tmp_path, workspace_id="workspace"),
            binding=DeterministicWorkspaceBinding(),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="worker", model="scripted-model"),
        tools=[Write()],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    paused = [
        event
        async for event in app.run(
            RunRequest(
                session_id="approved-workspace",
                agent_name="worker",
                messages=[Message.text("user", "write")],
            )
        )
    ]
    requested = next(e for e in paused if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
    request = ToolApprovalRequest(
        session_id="approved-workspace",
        approval_id=requested.payload["approval_id"],
        tool_round_id=requested.payload["tool_round_id"],
        tool_call_id=requested.payload["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
    )

    async def consume():
        return [event async for event in app.resolve_tool_approval(request)]

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 10)
        if cancel:
            task.cancel("cancel resumed writer")
            with pytest.raises(asyncio.CancelledError):
                await task
            checkpoint = await store.load_checkpoint("approved-workspace")
            approval = approval_reader.pending_approval_from_checkpoint(checkpoint)
            pending_round = round_reader.pending_tool_round_from_checkpoint(checkpoint)
            assert approval is not None and pending_round is not None
            assert len(pending_round.staged_terminals) == 1
            # This is the exact evidence check used by approval retry before
            # publishing the retained stage. A runtime-created interruption
            # must not be classified as contradictory private evidence.
            durable = await store.load_events("approved-workspace")
            approval_support.recorded_tool_outcomes(
                events=durable,
                approval=approval,
                staged_terminals=pending_round.staged_terminals,
            )
            stage = pending_round.staged_terminals[0].event
            start = next(e for e in durable if e.type is EventType.TOOL_CALL_STARTED)
            assert stage.payload["approval_id"] == start.payload["approval_id"]
            assert stage.payload["idempotency_key"] == start.payload["idempotency_key"]
            return
        else:
            release.set()
            await task
        checkpoint = await store.load_checkpoint("approved-workspace")
        durable = await store.load_events("approved-workspace")
        terminals = [
            e
            for e in durable
            if e.type
            in {
                EventType.TOOL_CALL_COMPLETED,
                EventType.TOOL_CALL_FAILED,
            }
        ]
        assert (tmp_path / "out.txt").read_bytes() == b"written"
        assert terminals, (
            f"No terminal was published; events={[e.type.value for e in durable]}; checkpoint keys={list(checkpoint)}"
        )
        starts = [e for e in durable if e.type is EventType.TOOL_CALL_STARTED]
        assert len(starts) == 1
        assert terminals[0].payload.get("approval_id") == starts[0].payload["approval_id"]
        assert terminals[0].payload["idempotency_key"] == starts[0].payload["idempotency_key"]
    finally:
        release.set()
        if not task.done():
            task.cancel()
            try:  # noqa: SIM105
                await task
            except BaseException:
                pass
