"""Real process-death worker; all durable state lives outside the disposable container."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from tests.core.test_workspace_mutation_receipts import (
    _ExclusiveWriterBinding,
    _portable_environment_spec,
    _PublicWorkspaceWriteTool,
    _SingleToolProvider,
    collect_events,
)

from cayu.artifacts import LocalArtifactStore
from cayu.core import AgentSpec, EventType, Message
from cayu.environments import Environment
from cayu.runners import DockerRunner
from cayu.runtime import CayuApp, EventQuery, RunRequest
from cayu.runtime._runtime_records import RegisteredEnvironment
from cayu.runtime.workspace_checkpoints import (
    WORKSPACE_CHECKPOINTS_KEY,
    ensure_workspace_checkpoint,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.workspaces import RunnerWorkspace
from cayu.workspaces.checkpoints import WorkspaceCheckpointError, WorkspaceCheckpointPolicy


def kill():
    os.kill(os.getpid(), signal.SIGKILL)


async def main(root: Path, container: str, phase: str, mode: str):
    class CrashStore(SQLiteSessionStore):
        invocation_lifecycle_command_version = 1

        async def publish_checkpoint_and_events(self, session_id, **kwargs):
            result = await super().publish_checkpoint_and_events(session_id, **kwargs)
            for event in kwargs["events"]:
                if event.type == EventType.WORKSPACE_CHECKPOINT_UPDATED:
                    payload = event.payload
                    if (phase == "intent" and payload["phase"] == "mutating") or (
                        phase == "checkpointing" and payload["phase"] == "checkpointing"
                    ):
                        kill()
                    if (
                        phase == "durable"
                        and payload["phase"] == "durable"
                        and payload["tool_call_id"]
                    ):
                        kill()
            return result

    class CrashArtifacts(LocalArtifactStore):
        async def put_bytes(self, content, **kwargs):
            result = await super().put_bytes(content, **kwargs)
            if phase == "uploaded" and kwargs["filename"] == "workspace-file.bin":
                kill()
            return result

        async def pin(self, artifact_id, *, owner):
            await super().pin(artifact_id, owner=owner)
            if phase == "pinned":
                read = await self.read_bytes(artifact_id, max_bytes=1)
                if read.metadata.filename == "workspace-file.bin":
                    kill()

    class CrashProvider(_SingleToolProvider):
        async def stream(self, request):
            if self.requests == 1 and phase == "model":
                kill()
            async for item in super().stream(request):
                yield item

    store = (
        CrashStore(root / "sessions.db")
        if mode == "produce"
        else SQLiteSessionStore(root / "sessions.db")
    )
    runner = DockerRunner(container, default_cwd="/workspace", close_action="none")
    workspace = RunnerWorkspace(runner, workspace_id="workspace")
    spec = _portable_environment_spec("owned")
    spec.workspace_checkpoint_policy = WorkspaceCheckpointPolicy()
    artifacts = (
        CrashArtifacts(root / "artifacts")
        if mode == "produce"
        else LocalArtifactStore(root / "artifacts")
    )
    environment = Environment(
        spec,
        workspace=workspace,
        runner=runner,
        binding=_ExclusiveWriterBinding(),
        artifact_store=artifacts,
    )
    if mode == "produce":
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            CrashProvider(tool_name="public_workspace_write", arguments={"path": "created.txt"}),
            default=True,
        )
        app.register_environment(environment, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="scripted-model"), tools=[_PublicWorkspaceWriteTool()]
        )
        await collect_events(
            app,
            RunRequest(
                agent_name="assistant", session_id="crash", messages=[Message.text("user", "write")]
            ),
        )
        raise AssertionError("Requested crash boundary was not reached")
    session = await store.load("crash")
    assert session is not None
    registered = RegisteredEnvironment(spec=spec, environment=environment)
    if phase in {"intent", "checkpointing", "uploaded", "pinned"}:
        try:
            await ensure_workspace_checkpoint(store, session, registered)
        except WorkspaceCheckpointError as exc:
            assert "unknown" in str(exc)
        else:
            raise AssertionError("Unknown mutation was accepted")
        assert not any(
            record.event.type == EventType.TOOL_CALL_COMPLETED
            for record in await store.query_events(EventQuery(session_id="crash"))
        )
    else:
        await ensure_workspace_checkpoint(store, session, registered)
        assert (await workspace.read_bytes("created.txt")).content == b"public"
        receipt = (await store.load_checkpoint("crash"))[WORKSPACE_CHECKPOINTS_KEY]["owned"]
        assert receipt["phase"] == "durable"
        # Recovery verifies the persisted revision and pins; no tool is dispatched.
        await ensure_workspace_checkpoint(store, session, registered)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]))
