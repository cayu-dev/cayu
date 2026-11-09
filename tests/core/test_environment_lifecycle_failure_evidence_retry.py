"""Recovering failure diagnostics must not skip the output-bearing finalize retry."""

import asyncio

import pytest
from tests.faults.test_workspace_sync_failure import (
    FailingFinalizeEvidenceStore,
    FailOnceWriteWorkspace,
)

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment, EnvironmentSpec, SyncBinding
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.sessions._completion_finalization import pending_completion_finalization_from_checkpoint
from cayu.tools.base import Tool, ToolResult, ToolSpec
from cayu.workspaces import LocalWorkspace


class _WriteOutput(Tool):
    spec = ToolSpec(
        name="write_output",
        description="Write the run's output into its bound workspace.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx, args):
        await ctx.workspace.write_bytes("result.txt", b"completed work")
        return ToolResult(content="written")


@pytest.mark.parametrize("diagnostic_unavailable", [False, True], ids=["durable", "retry"])
def test_failure_evidence_retry_preserves_pending_copy_back(tmp_path, diagnostic_unavailable):
    async def scenario():
        source_root = tmp_path / "source"
        target_root = tmp_path / "target"
        source_root.mkdir()
        target_root.mkdir()
        (source_root / "result.txt").write_text("original")
        source = FailOnceWriteWorkspace(
            source_root, workspace_id="retry-source", fail_path="result.txt"
        )
        target = LocalWorkspace(target_root, workspace_id="retry-target")
        binding = SyncBinding(target_workspace=target)
        store = FailingFinalizeEvidenceStore()
        store.fail_finalize_evidence = diagnostic_unavailable
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="write-result", name="write_output", arguments={}
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                ],
                name="output-provider",
            ),
            default=True,
        )
        app.register_environment(
            Environment(EnvironmentSpec(name="sync"), workspace=source, binding=binding),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"), tools=[_WriteOutput()]
        )
        session_id = "copy-back-diagnostic-retry"
        initial_error = None
        events = []
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "Write the result.")],
                    )
                )
            ]
        except OSError as error:
            initial_error = error
        assert (initial_error is not None) is diagnostic_unavailable
        if not diagnostic_unavailable:
            assert events[-1].type is EventType.SESSION_FAILED
        assert source.failed_writes == 1
        assert (source_root / "result.txt").read_text() == "original"
        assert (target_root / "result.txt").read_text() == "completed work"
        assert binding._states, "The failed copy-back must still own its generation"
        assert (
            pending_completion_finalization_from_checkpoint(await store.load_checkpoint(session_id))
            is not None
        )

        # Both the source write and the event store are now available. Drain
        # must finish copy-back before releasing the target for another run.
        store.fail_finalize_evidence = False
        drained = await app.drain_environment_cleanups(timeout_s=2)
        successor = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="next-copy-back-session",
                    agent_name="assistant",
                    messages=[Message.text("user", "Run without changing files.")],
                )
            )
        ]
        assert successor[-1].type is EventType.SESSION_COMPLETED
        # A new bind cleans this fixed target and copies the source into it.
        # Releasing the old generation without syncing loses its only output.
        assert (source_root / "result.txt").read_text() == "completed work", (
            "Diagnostic retry abandoned pending output instead of retrying copy-back",
            drained,
            bool(binding._states),
            session_id in app._environment_lifecycle._active_environment_setups,
            (target_root / "result.txt").read_text(),
        )
        assert (
            pending_completion_finalization_from_checkpoint(await store.load_checkpoint(session_id))
            is None
        )
        assert binding._states == {}

    asyncio.run(scenario())
