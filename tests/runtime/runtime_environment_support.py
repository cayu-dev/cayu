"""Environment support."""

from __future__ import annotations

from typing import Any

from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import BoundWorkspace, WorkspaceBinding, WorkspaceSnapshot
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.runners.base import DEFAULT_EXEC_OUTPUT_LIMIT_BYTES, ExecCommand, ExecResult, Runner
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workspaces.base import Workspace, WorkspaceListResult, WorkspaceReadResult


class MemoryWorkspace(Workspace):
    def __init__(self, workspace_id: str) -> None:
        self.id = workspace_id

    def bounded_read_limit(self, max_bytes: int) -> int:
        return max_bytes

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        return WorkspaceReadResult(content=b"", total_bytes=0)

    async def write_bytes(self, path: str, content: bytes) -> None:
        return None

    async def delete(self, path: str) -> None:
        return None

    async def create_bytes(self, path: str, content: bytes):
        raise NotImplementedError

    async def replace_bytes(self, path: str, content: bytes, *, expected_revision: str):
        raise NotImplementedError

    async def delete_if_revision(self, path: str, *, expected_revision: str):
        raise NotImplementedError

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        return WorkspaceListResult(paths=(), total_count=0)


class WorkspaceIdTool(Tool):
    spec = ToolSpec(
        name="workspace_id",
        description="Return the active workspace id.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(
            content=ctx.workspace_id or "none",
            structured={"workspace_id": ctx.workspace_id},
        )


class RecordingWorkspaceBinding(WorkspaceBinding):
    def __init__(
        self,
        *,
        bound_workspace: Workspace | None = None,
        snapshot: WorkspaceSnapshot | None = None,
        final_snapshot: WorkspaceSnapshot | None = None,
        fail_bind: bool = False,
        fail_finalize: bool = False,
        finalize_error: RuntimeError | None = None,
    ) -> None:
        self.bound_workspace = bound_workspace
        self.snapshot = snapshot
        self.final_snapshot = final_snapshot
        self.fail_bind = fail_bind
        self.fail_finalize = fail_finalize
        self.finalize_error = finalize_error
        self.bind_calls: list[dict] = []
        self.finalize_calls: list[dict] = []

    async def bind(
        self,
        workspace: Workspace | None,
        runner: Any,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        self.bind_calls.append(
            {
                "workspace": workspace,
                "runner": runner,
                "session_id": session_id,
                "agent_name": agent_name,
                "environment_name": environment_name,
                "metadata": metadata,
            }
        )
        if self.fail_bind:
            raise RuntimeError("bind failed")
        return BoundWorkspace(
            workspace=self.bound_workspace if self.bound_workspace is not None else workspace,
            source_workspace=workspace,
            runner=runner,
            path="/bound",
            metadata={"binding": "recording"},
            snapshot=self.snapshot,
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        self.finalize_calls.append(
            {
                "bound": bound,
                "outcome": outcome,
                "metadata": metadata,
            }
        )
        if self.fail_finalize:
            raise self.finalize_error or RuntimeError("finalize failed")
        return self.final_snapshot


class RecordingEnvironmentFactory(EnvironmentFactory):
    def __init__(
        self,
        environment: Environment,
        *,
        fail_create: bool = False,
        metadata: dict[str, Any] | None = None,
        reconnect_metadata: dict[str, Any] | None = None,
        release: Any = None,
        release_timeout_s: float = 15.0,
        create_error: BaseException | None = None,
        allocation_fingerprint: str | None = None,
    ) -> None:
        self.environment = environment
        self.fail_create = fail_create
        self.metadata = metadata or {}
        self.reconnect_metadata = reconnect_metadata or {}
        self.release = release
        self.release_timeout_s = release_timeout_s
        self.create_error = create_error
        self.allocation_fingerprint = allocation_fingerprint
        self.requests: list[EnvironmentFactoryRequest] = []

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        if self.create_error is not None:
            raise self.create_error
        if self.fail_create:
            raise RuntimeError("factory failed")
        return EnvironmentFactoryResult(
            environment=self.environment,
            metadata=self.metadata,
            reconnect_metadata=self.reconnect_metadata,
            release=self.release,
            release_timeout_s=self.release_timeout_s,
            allocation_fingerprint=self.allocation_fingerprint,
        )


class ProvisionedTestRunner(Runner):
    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        return ExecResult(stdout="unused")


class RecordingWorkspaceAllocator:
    def __init__(self) -> None:
        self.requests: list[EnvironmentFactoryRequest] = []
        self.workspaces: list[MemoryWorkspace] = []

    def allocate(self, request: EnvironmentFactoryRequest) -> Workspace:
        self.requests.append(request)
        workspace = MemoryWorkspace(f"workspace-{request.session_id}")
        self.workspaces.append(workspace)
        return workspace


class RecordingRunnerAllocator:
    def __init__(self) -> None:
        self.requests: list[EnvironmentFactoryRequest] = []
        self.runners: list[ProvisionedTestRunner] = []

    def allocate(self, request: EnvironmentFactoryRequest) -> Runner:
        self.requests.append(request)
        runner = ProvisionedTestRunner()
        self.runners.append(runner)
        return runner


class RecordingBindingAllocator:
    def __init__(self) -> None:
        self.requests: list[EnvironmentFactoryRequest] = []
        self.bindings: list[RecordingWorkspaceBinding] = []

    def allocate(self, request: EnvironmentFactoryRequest) -> WorkspaceBinding:
        self.requests.append(request)
        binding = RecordingWorkspaceBinding()
        self.bindings.append(binding)
        return binding


class AllocatingEnvironmentFactory(EnvironmentFactory):
    def __init__(
        self,
        environment_name: str,
        *,
        workspace_allocator: RecordingWorkspaceAllocator,
        runner_allocator: RecordingRunnerAllocator,
        binding_allocator: RecordingBindingAllocator,
    ) -> None:
        self.environment_name = environment_name
        self.workspace_allocator = workspace_allocator
        self.runner_allocator = runner_allocator
        self.binding_allocator = binding_allocator
        self.requests: list[EnvironmentFactoryRequest] = []

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        self.requests.append(request)
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=self.environment_name),
                workspace=self.workspace_allocator.allocate(request),
                runner=self.runner_allocator.allocate(request),
                binding=self.binding_allocator.allocate(request),
            ),
            reconnect_metadata={"sandbox_id": "unexpected"},
        )


class EnvironmentProvisioningRecorder:
    def __init__(self, environment_name: str) -> None:
        self.workspace_allocator = RecordingWorkspaceAllocator()
        self.runner_allocator = RecordingRunnerAllocator()
        self.binding_allocator = RecordingBindingAllocator()
        self.factory = AllocatingEnvironmentFactory(
            environment_name,
            workspace_allocator=self.workspace_allocator,
            runner_allocator=self.runner_allocator,
            binding_allocator=self.binding_allocator,
        )


def assert_environment_was_not_provisioned(recorder: EnvironmentProvisioningRecorder) -> None:
    assert recorder.factory.requests == []
    assert recorder.workspace_allocator.requests == []
    assert recorder.workspace_allocator.workspaces == []
    assert recorder.runner_allocator.requests == []
    assert recorder.runner_allocator.runners == []
    assert recorder.binding_allocator.requests == []
    assert recorder.binding_allocator.bindings == []
