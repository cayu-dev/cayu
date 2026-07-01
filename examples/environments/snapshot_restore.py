"""Environment factory example: snapshot restore/save via a custom WorkspaceBinding.

Usage:
    uv sync --extra dev
    PYTHONPATH=src .venv/bin/python examples/environments/snapshot_restore.py

API-key-free. Shows the reproducibility / fork pattern: a custom ``WorkspaceBinding``
restores the workspace from the last saved snapshot on ``bind`` (before the run) and writes
a new snapshot on ``finalize`` (after the run). Core does not own a snapshot store — this
binding uses a local directory; a production binding would target durable object storage
(S3, GCS, ...) instead.

Two sessions run against a SHARED snapshot directory but FRESH per-session workspaces:
  * session A writes a file -> finalize saves it to the snapshot;
  * session B starts with an empty workspace -> bind restores the snapshot -> the file is there.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

from cayu import (
    AgentSpec,
    BoundWorkspace,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    ListFilesTool,
    LocalRunner,
    LocalWorkspace,
    Message,
    RunRequest,
    WorkspaceBinding,
    WorkspaceSnapshot,
    WriteFileTool,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runners import Runner
from cayu.workspaces import Workspace


class SnapshotBinding(WorkspaceBinding):
    """Restore the workspace from the last snapshot on bind; save a new snapshot on finalize.

    The snapshot store here is just a local directory; the point is the lifecycle, not the
    backend. A real binding would copy to/from durable object storage.
    """

    def __init__(self, snapshot_dir: Path) -> None:
        self._snapshot_dir = snapshot_dir

    async def bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> BoundWorkspace:
        # Note the two file APIs: workspace files always go through the async Workspace API
        # (write_bytes/read_bytes/list); the snapshot store is a plain local directory we read
        # and write directly with pathlib.
        restored = 0
        if workspace is not None and self._snapshot_dir.exists():
            for file in sorted(self._snapshot_dir.rglob("*")):
                if file.is_file():
                    rel = file.relative_to(self._snapshot_dir).as_posix()
                    await workspace.write_bytes(rel, file.read_bytes())
                    restored += 1
        # The returned WorkspaceSnapshot is descriptive metadata surfaced via binding events
        # (it isn't re-consumed by this binding) — this is the "snapshot" the guide means, not
        # a separate SnapshotBinding class.
        return BoundWorkspace(
            workspace=workspace,
            source_workspace=workspace,
            runner=runner,
            snapshot=WorkspaceSnapshot(
                snapshot_id=f"restore:{session_id}",
                workspace_id=workspace.id if workspace is not None else None,
                source="snapshot",
                metadata={"restored_files": restored},
            ),
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> WorkspaceSnapshot | None:
        workspace = bound.workspace
        saved = 0
        if workspace is not None:
            # A snapshot is point-in-time — replace the stored tree, don't merge into it, so
            # files deleted during the run don't linger in the snapshot forever.
            shutil.rmtree(self._snapshot_dir, ignore_errors=True)
            listing = await workspace.list("**/*")
            for path in listing.paths:
                result = await workspace.read_bytes(path)
                dest = self._snapshot_dir / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(result.content)
                saved += 1
        return WorkspaceSnapshot(
            snapshot_id=f"save:{outcome or 'unknown'}",
            workspace_id=workspace.id if workspace is not None else None,
            source="snapshot",
            metadata={"saved_files": saved, "outcome": outcome},
        )


class SnapshotFactory(EnvironmentFactory):
    """Provision a fresh per-session local environment bound by ``SnapshotBinding``."""

    def __init__(self, workspaces_root: Path, snapshot_dir: Path) -> None:
        self._workspaces_root = workspaces_root
        self._snapshot_dir = snapshot_dir

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        root = self._workspaces_root / request.session_id
        root.mkdir(parents=True, exist_ok=True)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=request.environment_name),
                workspace=LocalWorkspace(root, workspace_id=f"ws-{request.session_id}"),
                runner=LocalRunner(root),
                binding=SnapshotBinding(self._snapshot_dir),
            )
        )


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self._batches = [
            # session A: write a file
            [
                ModelStreamEvent.tool_call(
                    id="a1",
                    name="write_file",
                    arguments={"path": "data/answer.txt", "content": "42"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("wrote data/answer.txt"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            # session B: list files (should see the restored file)
            [
                ModelStreamEvent.tool_call(
                    id="b1", name="list_files", arguments={"pattern": "**/*"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("listed restored files"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self._batches[len(self.requests) - 1]:
            yield event


async def _run(app: CayuApp, session_id: str, prompt: str) -> None:
    async for event in app.run(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", prompt)],
        )
    ):
        if event.tool_name == "list_files" and event.type.endswith("completed"):
            print("  list_files result:", event.payload)


async def main() -> None:
    with TemporaryDirectory(prefix="cayu-snapshot-") as base_dir:
        base = Path(base_dir)
        snapshot_dir = base / "snapshots"
        app = CayuApp()
        app.register_provider(FakeProvider(), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="snapshot"),
            SnapshotFactory(base / "workspaces", snapshot_dir),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[WriteFileTool(), ListFilesTool()],
        )

        print("session A — writes a file, finalize saves a snapshot:")
        await _run(app, "session_a", "write the answer")
        saved = sorted(
            p.relative_to(snapshot_dir).as_posix() for p in snapshot_dir.rglob("*") if p.is_file()
        )
        print("  snapshot now holds:", saved)

        print("session B — fresh workspace, bind restores the snapshot:")
        await _run(app, "session_b", "list the files")


if __name__ == "__main__":
    asyncio.run(main())
