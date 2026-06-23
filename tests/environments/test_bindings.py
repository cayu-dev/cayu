from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from cayu.environments import (
    BoundWorkspace,
    NativeBinding,
    NoWorkspaceBinding,
    WorkspaceBinding,
    copy_bound_workspace,
)
from cayu.runners import ExecCommand, ExecResult, Runner
from cayu.workspaces import Workspace, WorkspaceListResult, WorkspaceReadResult


class StubWorkspace(Workspace):
    id = "stub-workspace"

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        return WorkspaceReadResult(content=b"", total_bytes=0)

    async def write_bytes(self, path: str, content: bytes) -> None:
        pass

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        return WorkspaceListResult(paths=(), total_count=0)


class StubRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        return ExecResult(stdout="ok")


def test_native_binding_passes_configured_workspace_and_runner_through() -> None:
    workspace = StubWorkspace()
    runner = StubRunner()
    metadata = {"mount": {"id": "mnt_1"}}

    bound = asyncio.run(
        NativeBinding(default_path="/workspace").bind(
            workspace,
            runner,
            session_id="sess_1",
            agent_name="agent",
            environment_name="env",
            metadata=metadata,
        )
    )

    assert bound.workspace is workspace
    assert bound.runner is runner
    assert bound.path == "/workspace"
    assert bound.metadata == {"mount": {"id": "mnt_1"}}

    metadata["mount"]["id"] = "mutated"
    assert bound.metadata == {"mount": {"id": "mnt_1"}}


def test_no_workspace_binding_hides_workspace() -> None:
    workspace = StubWorkspace()
    runner = StubRunner()

    bound = asyncio.run(
        NoWorkspaceBinding().bind(
            workspace,
            runner,
            session_id="sess_1",
            metadata={"reason": "api-only"},
        )
    )

    assert bound.workspace is None
    assert bound.runner is runner
    assert bound.path is None
    assert bound.metadata == {"reason": "api-only"}


def test_bind_request_rejects_invalid_values() -> None:
    invalid_workspace: Any = object()
    invalid_runner: Any = object()
    invalid_metadata: Any = []

    with pytest.raises(TypeError, match="workspace"):
        asyncio.run(NativeBinding().bind(invalid_workspace, None, session_id="sess_1"))
    with pytest.raises(TypeError, match="runner"):
        asyncio.run(NativeBinding().bind(None, invalid_runner, session_id="sess_1"))
    with pytest.raises(ValueError, match="session_id"):
        asyncio.run(NativeBinding().bind(None, None, session_id=" "))
    with pytest.raises(ValueError, match="agent_name"):
        asyncio.run(NativeBinding().bind(None, None, session_id="sess_1", agent_name=" "))
    with pytest.raises(ValueError, match="environment_name"):
        asyncio.run(
            NativeBinding().bind(None, None, session_id="sess_1", environment_name=" ")
        )
    with pytest.raises(TypeError, match="metadata"):
        asyncio.run(NativeBinding().bind(None, None, session_id="sess_1", metadata=invalid_metadata))
    with pytest.raises(ValueError, match="metadata"):
        asyncio.run(NativeBinding().bind(None, None, session_id="sess_1", metadata={"bad": object()}))


def test_binding_finalize_methods_are_noops() -> None:
    bound = BoundWorkspace()

    async def run() -> None:
        await NativeBinding().finalize(bound, outcome="completed")
        await NoWorkspaceBinding().finalize(bound, outcome="completed", metadata={"ok": True})

    asyncio.run(run())


def test_binding_finalize_rejects_invalid_values() -> None:
    invalid_bound: Any = object()
    invalid_metadata: Any = []

    async def run() -> None:
        binding = NativeBinding()

        with pytest.raises(TypeError, match="BoundWorkspace"):
            await binding.finalize(invalid_bound)
        with pytest.raises(ValueError, match="outcome"):
            await binding.finalize(BoundWorkspace(), outcome=" ")
        with pytest.raises(TypeError, match="metadata"):
            await binding.finalize(BoundWorkspace(), metadata=invalid_metadata)
        with pytest.raises(ValueError, match="metadata"):
            await binding.finalize(BoundWorkspace(), metadata={"bad": object()})

    asyncio.run(run())


def test_bound_workspace_validates_shape_and_copies_metadata() -> None:
    workspace = StubWorkspace()
    runner = StubRunner()
    metadata = {"nested": {"value": 1}}

    bound = BoundWorkspace(
        workspace=workspace,
        runner=runner,
        path="/workspace",
        metadata=metadata,
    )

    metadata["nested"]["value"] = 2
    assert bound.workspace is workspace
    assert bound.runner is runner
    assert bound.path == "/workspace"
    assert bound.metadata == {"nested": {"value": 1}}

    with pytest.raises(FrozenInstanceError):
        bound.__setattr__("path", "/other")


def test_bound_workspace_rejects_invalid_values() -> None:
    invalid_workspace: Any = object()
    invalid_runner: Any = object()
    invalid_path: Any = 123
    invalid_metadata: Any = []

    with pytest.raises(TypeError, match="workspace"):
        BoundWorkspace(workspace=invalid_workspace)
    with pytest.raises(TypeError, match="runner"):
        BoundWorkspace(runner=invalid_runner)
    with pytest.raises(TypeError, match="path"):
        BoundWorkspace(path=invalid_path)
    with pytest.raises(ValueError, match="path"):
        BoundWorkspace(path=" ")
    with pytest.raises(TypeError, match="metadata"):
        BoundWorkspace(metadata=invalid_metadata)
    with pytest.raises(ValueError, match="metadata"):
        BoundWorkspace(metadata={"bad": object()})


def test_binding_constructors_validate_values() -> None:
    invalid_path: Any = 123

    with pytest.raises(TypeError, match="default_path"):
        NativeBinding(default_path=invalid_path)
    with pytest.raises(ValueError, match="default_path"):
        NativeBinding(default_path=" ")


def test_copy_bound_workspace_defensively_copies_metadata() -> None:
    bound = BoundWorkspace(metadata={"token": {"cursor": "a"}})

    copied = copy_bound_workspace(bound)
    bound.metadata["token"]["cursor"] = "b"

    assert copied is not bound
    assert copied.metadata == {"token": {"cursor": "a"}}


def test_copy_bound_workspace_rejects_invalid_value() -> None:
    invalid_bound: Any = object()

    with pytest.raises(TypeError, match="BoundWorkspace"):
        copy_bound_workspace(invalid_bound)


def test_workspace_binding_is_abstract() -> None:
    abstract_cls: Any = WorkspaceBinding

    with pytest.raises(TypeError):
        abstract_cls()
