from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from cayu import (
    AgentSpec,
    BoundWorkspace,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SyncBinding,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    WorkspaceSnapshot,
)
from cayu.runners import Runner
from cayu.runtime import SessionStatus
from cayu.storage import SQLiteSessionStore
from cayu.workspaces import LocalWorkspace, Workspace


class FailOnceWriteWorkspace(LocalWorkspace):
    def __init__(self, root: Path, *, workspace_id: str, fail_path: str) -> None:
        super().__init__(root, workspace_id=workspace_id)
        self.fail_path = fail_path
        self.failed_writes = 0

    async def write_bytes(self, path: str, content: bytes) -> None:
        if path == self.fail_path:
            self.fail_path = ""
            self.failed_writes += 1
            raise OSError(f"forced sync write failure: {path}")
        await super().write_bytes(path, content)


class CapturingSyncBinding(SyncBinding):
    def __init__(self, *, target_workspace: Workspace) -> None:
        super().__init__(target_workspace=target_workspace)
        self.last_bound: BoundWorkspace | None = None

    async def bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        bound = await super().bind(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )
        self.last_bound = bound
        return bound


class MutateBoundWorkspaceTool(Tool):
    spec = ToolSpec(
        name="mutate_bound_workspace",
        description="Make deterministic file changes in the bound workspace.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if ctx.workspace is None:
            raise RuntimeError("bound workspace is unavailable")
        await ctx.workspace.write_bytes("a-updated.txt", b"updated-a")
        await ctx.workspace.write_bytes("b-fail.txt", b"updated-b")
        await ctx.workspace.write_bytes("created.txt", b"created")
        await ctx.workspace.delete("removed.txt")
        return ToolResult(content="workspace mutated")


def test_sync_binding_partial_finalize_failure_is_durable_and_retry_converges(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    # LocalWorkspace.list() sorts paths, so the a-file copies before the injected
    # b-file failure and makes the partial-finalization point deterministic.
    (source_root / "a-updated.txt").write_text("original-a", encoding="utf-8")
    (source_root / "b-fail.txt").write_text("original-b", encoding="utf-8")
    (source_root / "removed.txt").write_text("remove-me", encoding="utf-8")
    (target_root / "stale-target.txt").write_text("clean-me", encoding="utf-8")

    source = FailOnceWriteWorkspace(
        source_root,
        workspace_id="durable-source",
        fail_path="b-fail.txt",
    )
    target = LocalWorkspace(target_root, workspace_id="ephemeral-target")
    binding = CapturingSyncBinding(target_workspace=target)
    store_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(store_path)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_mutate_workspace",
                    name="mutate_bound_workspace",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Workspace changes are ready."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ],
        name="workspace-sync-fault-provider",
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="sync-fault"),
            workspace=source,
            binding=binding,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="workspace-sync-assistant",
            model="scripted-model",
            provider_name="workspace-sync-fault-provider",
        ),
        tools=[MutateBoundWorkspaceTool()],
    )

    async def exercise_contract():
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="workspace-sync-assistant",
                        session_id="workspace-sync-failure",
                        messages=[Message.text("user", "Apply the workspace changes.")],
                    )
                )
            ]
        finally:
            await store.close()

        partial_source = {
            path.name: path.read_text(encoding="utf-8") for path in source_root.iterdir()
        }
        reopened = SQLiteSessionStore(store_path)
        try:
            durable_session = await reopened.load("workspace-sync-failure")
            durable_events = await reopened.load_events("workspace-sync-failure")
        finally:
            await reopened.close()

        assert binding.last_bound is not None
        retry_snapshot = await binding.finalize(
            binding.last_bound,
            outcome="completed",
            metadata={
                "event_type": "session.completed",
                "session_id": "workspace-sync-failure",
            },
        )
        rebound = await binding.bind(source, None, session_id="workspace-sync-rebind")
        binding.abandon(rebound)
        return events, partial_source, durable_session, durable_events, retry_snapshot

    events, partial_source, session, durable_events, retry_snapshot = asyncio.run(
        exercise_contract()
    )

    assert source.failed_writes == 1
    assert partial_source == {
        "a-updated.txt": "updated-a",
        "b-fail.txt": "original-b",
        "removed.txt": "remove-me",
    }

    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert [event.type for event in events[-3:]] == [
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        EventType.SESSION_COMPLETED,
    ]
    assert [event.type for event in durable_events[-3:]] == [
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        EventType.SESSION_COMPLETED,
    ]
    finalize_failure = durable_events[-2]
    assert finalize_failure.payload["error"] == "forced sync write failure: b-fail.txt"
    assert finalize_failure.payload["error_type"] == "OSError"
    assert durable_events[-1].payload["binding_finalize_error"] == {
        "error": "forced sync write failure: b-fail.txt",
        "error_type": "OSError",
        "outcome": "completed",
        "failures": [
            {
                "phase": "workspace_finalize",
                "error": "forced sync write failure: b-fail.txt",
                "error_type": "OSError",
            }
        ],
    }

    assert type(retry_snapshot) is WorkspaceSnapshot
    assert retry_snapshot.metadata["copied_files"] == 3
    assert retry_snapshot.metadata["deleted_files"] == 1
    assert {path.name: path.read_text(encoding="utf-8") for path in source_root.iterdir()} == {
        "a-updated.txt": "updated-a",
        "b-fail.txt": "updated-b",
        "created.txt": "created",
    }
