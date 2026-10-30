"""A settled invocation must not leave a process-local writer window open."""

import asyncio
import contextlib

import pytest
from tests.core.test_workspace_mutation_receipts import (
    _ExclusiveWriterBinding,
    _portable_environment_spec,
    _PublicWorkspaceWriteTool,
    _SingleToolProvider,
    collect_events,
)

import cayu.runtime._tool_invocation.invocation as invocation_module
from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.artifacts import LocalArtifactStore
from cayu.environments import Environment
from cayu.sessions import InMemorySessionStore
from cayu.workspaces import LocalWorkspace
from cayu.workspaces.checkpoints import WorkspaceCheckpointPolicy


@pytest.mark.parametrize("storage_outage", [False, True])
def test_checkpoint_failure_does_not_poison_later_workspace_invocations(
    tmp_path, monkeypatch, storage_outage
):
    async def scenario():
        root = tmp_path / "workspace"
        root.mkdir()
        windows = []
        begin = invocation_module.begin_workspace_mutation_window

        def track_window(*args, **kwargs):
            window = begin(*args, **kwargs)
            windows.append(window)
            return window

        monkeypatch.setattr(invocation_module, "begin_workspace_mutation_window", track_window)

        class RecoverableArtifactStore(LocalArtifactStore):
            unavailable = storage_outage

            async def put_bytes(self, content, **kwargs):
                if self.unavailable and (root / "first.txt").exists():
                    raise OSError("temporary checkpoint storage outage")
                return await super().put_bytes(content, **kwargs)

        artifacts = RecoverableArtifactStore(tmp_path / "artifacts")
        store = InMemorySessionStore()

        async def run_session(session_id, filename):
            # A new session has no unresolved checkpoint from the failed session.
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                _SingleToolProvider(
                    tool_name="public_workspace_write", arguments={"path": filename}
                ),
                default=True,
            )
            spec = _portable_environment_spec("local")
            spec.workspace_checkpoint_policy = WorkspaceCheckpointPolicy()
            app.register_environment(
                Environment(
                    spec,
                    workspace=LocalWorkspace(root, workspace_id="owned"),
                    binding=_ExclusiveWriterBinding(),
                    artifact_store=artifacts,
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="scripted-model"),
                tools=[_PublicWorkspaceWriteTool()],
            )
            return await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create")],
                ),
            )

        try:
            with contextlib.suppress(Exception):
                await run_session("failed-checkpoint", "first.txt")
            assert (root / "first.txt").exists()
            artifacts.unavailable = False
            events = await run_session("healthy-checkpoint", "second.txt")
            assert (root / "second.txt").exists(), "The second tool executed successfully"
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            assert any(event.type is EventType.TOOL_CALL_COMPLETED for event in events)
            assert not windows[-1].overlap_detected, "No other invocation is still running"
        finally:
            # Do not leave the buggy process-local registry polluted for other tests.
            for window in windows:
                window.close(discard_history=True)

    asyncio.run(scenario())


def test_retained_checkpoint_cannot_publish_after_its_writer_window_closes(tmp_path, monkeypatch):
    import cayu.workspaces.checkpoint_lifecycle as checkpoints
    from cayu.workspaces.checkpoints import WorkspaceCheckpointError

    async def scenario():
        root = tmp_path / "workspace"
        root.mkdir()
        store = InMemorySessionStore()
        artifacts = LocalArtifactStore(tmp_path / "artifacts")
        blocked = asyncio.Event()
        release = asyncio.Event()
        retained = []
        windows = []
        capture = checkpoints.capture_workspace_checkpoint
        begin = invocation_module.begin_workspace_mutation_window

        def track_window(*args, **kwargs):
            window = begin(*args, **kwargs)
            windows.append(window)
            return window

        async def block_first_capture(*args, **kwargs):
            if (root / "first.txt").exists() and not blocked.is_set():
                retained.append(asyncio.current_task())
                blocked.set()
                await release.wait()
            return await capture(*args, **kwargs)

        monkeypatch.setattr(invocation_module, "begin_workspace_mutation_window", track_window)
        monkeypatch.setattr(checkpoints, "capture_workspace_checkpoint", block_first_capture)
        apps = []

        async def run_session(session_id, filename):
            app = CayuApp(session_store=store, enable_logging=False)
            apps.append(app)
            app.register_provider(
                _SingleToolProvider(
                    tool_name="public_workspace_write", arguments={"path": filename}
                ),
                default=True,
            )
            spec = _portable_environment_spec("local")
            spec.workspace_checkpoint_policy = WorkspaceCheckpointPolicy(timeout_seconds=1)
            app.register_environment(
                Environment(
                    spec,
                    workspace=LocalWorkspace(root, workspace_id="shared"),
                    binding=_ExclusiveWriterBinding(),
                    artifact_store=artifacts,
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="scripted-model"),
                tools=[_PublicWorkspaceWriteTool()],
            )
            return await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "create")],
                ),
            )

        try:
            first_events = await asyncio.wait_for(run_session("timed-out", "first.txt"), 10)
            assert first_events[-1].type is EventType.SESSION_FAILED
            assert blocked.is_set() and not retained[0].done()
            assert windows[0].closed
            events = await run_session("next-writer", "second.txt")
            assert events[-1].type is EventType.SESSION_COMPLETED
            release.set()
            (result,) = await asyncio.gather(*retained, return_exceptions=True)
            assert isinstance(result, WorkspaceCheckpointError)
            assert "isolation is ambiguous" in str(result)
            checkpoint = await store.load_checkpoint("timed-out")
            receipts = checkpoint[checkpoints.WORKSPACE_CHECKPOINTS_KEY]
            assert all(receipt["phase"] != "durable" for receipt in receipts.values())
        finally:
            release.set()
            await asyncio.gather(*retained, return_exceptions=True)
            for window in windows:
                window.close(discard_history=True)
            for app in apps:
                await app.aclose()

    asyncio.run(scenario())
