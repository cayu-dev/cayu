"""Interrupted dispatch must not hide a detached writer from other apps."""

import asyncio

import pytest
from tests.core.test_workspace_mutation_receipts import (
    _AsyncBlockingWorkspace,
    _ExclusiveWriterBinding,
    _portable_environment_spec,
    _PublicWorkspaceWriteTool,
    _SingleToolProvider,
    _SupervisoryDetachedWorkspaceMutationTool,
    collect_events,
)

import cayu.runtime._tool_invocation.invocation as invocation_module
from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import Environment
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("grouped", [False, True])
def test_interrupted_dispatch_preserves_overlap_for_independent_app(tmp_path, monkeypatch, grouped):
    async def scenario():
        windows = []
        original_begin = invocation_module.begin_workspace_mutation_window

        def begin(*args, **kwargs):
            window = original_begin(*args, **kwargs)
            windows.append(window)
            return window

        monkeypatch.setattr(invocation_module, "begin_workspace_mutation_window", begin)
        workspace = _AsyncBlockingWorkspace(tmp_path, workspace_id="shared")
        failure = GeneratorExit("supervisor abandoned tool execution")
        if grouped:
            failure = BaseExceptionGroup("supervisory failures", [failure, RuntimeError("cleanup")])

        def build_app(workspace, tool, arguments):
            app = CayuApp(enable_logging=False)
            app.register_provider(
                _SingleToolProvider(tool_name=tool.spec.name, arguments=arguments), default=True
            )
            app.register_environment(
                Environment(
                    _portable_environment_spec("local"),
                    workspace=workspace,
                    binding=_ExclusiveWriterBinding(),
                ),
                default=True,
            )
            app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[tool])
            return app

        first = build_app(workspace, _SupervisoryDetachedWorkspaceMutationTool(failure), {})
        followup = build_app(
            LocalWorkspace(tmp_path, workspace_id="shared"),
            _PublicWorkspaceWriteTool(),
            {"path": "next.txt"},
        )

        def request(session_id):
            return RunRequest(
                session_id=session_id,
                agent_name="worker",
                messages=[Message.text("user", "write")],
            )

        try:
            with pytest.raises(BaseExceptionGroup if grouped else GeneratorExit):
                await collect_events(first, request("interrupted-writer"))
            assert workspace.started.is_set()
            assert not (tmp_path / "settled-after-supervisory-exit.txt").exists()

            # A separate app has no access to the first environment's fence.
            events = await collect_events(followup, request("independent-writer"))
            assert events[-1].type is EventType.SESSION_COMPLETED
            receipt = next(
                event for event in events if event.type is EventType.WORKSPACE_MUTATION_RECORDED
            )
            assert receipt.payload["attribution"]["overlap_detected"] is True, receipt.payload
            assert windows[-1].overlap_detected
            assert not windows[0].closed
        finally:
            workspace.release.set()
            await first.drain_environment_cleanups(timeout_s=10)
            for window in windows:
                window.close(discard_history=True)
            await followup.aclose()
            await first.aclose()

    asyncio.run(scenario())
