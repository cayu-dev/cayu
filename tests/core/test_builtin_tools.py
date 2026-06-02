from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator

import pytest

from cayu import Environment, EnvironmentSpec
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runners import ExecCommand, ExecResult, LocalRunner, Runner
from cayu.runtime import CayuApp, RunRequest
from cayu.tools import ExecCommandTool
from cayu.tools.commands import (
    DEFAULT_OUTPUT_LIMIT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_LIMIT_BYTES,
    MAX_TIMEOUT_SECONDS,
)
from cayu.tools.files import (
    DEFAULT_LIST_LIMIT,
    DEFAULT_READ_LIMIT_BYTES,
    DEFAULT_WRITE_LIMIT_BYTES,
    MAX_LIST_LIMIT,
    MAX_READ_LIMIT_BYTES,
    MAX_WRITE_LIMIT_BYTES,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from cayu.workspaces import LocalWorkspace


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
    ) -> None:
        if events and isinstance(events[0], list):
            self.event_batches = events  # type: ignore[assignment]
        else:
            self.event_batches = [events]  # type: ignore[list-item]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        for event in self.event_batches[len(self.requests) - 1]:
            yield event


class ContextRecordingTool(Tool):
    spec = ToolSpec(
        name="record_context",
        description="Record runtime tool context.",
        input_schema={"type": "object"},
    )

    def __init__(self) -> None:
        super().__init__()
        self.context: ToolContext | None = None

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.context = ctx
        return ToolResult(content="recorded")


class RecordingRunner(Runner):
    def __init__(self, result: ExecResult | None = None) -> None:
        self.result = result or ExecResult(stdout="ok\n")
        self.timeout_s: int | None = None

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
        self.timeout_s = timeout_s
        return self.result


def test_tool_context_carries_services_without_serializing_them(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    runner = LocalRunner(tmp_path)
    ctx = ToolContext(
        session_id="sess_1",
        agent_name="assistant",
        environment_name="local-dev",
        workspace_id="local",
        workspace=workspace,
        runner=runner,
        mcp_servers=[object()],
    )

    dumped = ctx.model_dump()

    assert ctx.workspace is workspace
    assert ctx.runner is runner
    assert ctx.mcp_servers
    assert dumped == {
        "session_id": "sess_1",
        "agent_name": "assistant",
        "environment_name": "local-dev",
        "workspace_id": "local",
        "metadata": {},
    }


def test_builtin_tool_limits_are_model_context_sized():
    assert DEFAULT_READ_LIMIT_BYTES == 256 * 1024
    assert MAX_READ_LIMIT_BYTES == 4 * 1024 * 1024
    assert DEFAULT_WRITE_LIMIT_BYTES == 256 * 1024
    assert MAX_WRITE_LIMIT_BYTES == 4 * 1024 * 1024
    assert DEFAULT_LIST_LIMIT == 500
    assert MAX_LIST_LIMIT == 10_000
    assert DEFAULT_OUTPUT_LIMIT_BYTES == 50_000
    assert MAX_OUTPUT_LIMIT_BYTES == 200_000
    assert DEFAULT_TIMEOUT_SECONDS == 60
    assert MAX_TIMEOUT_SECONDS == 600

    assert ReadFileTool().schema["properties"]["max_bytes"]["default"] == 256 * 1024
    assert ReadFileTool().schema["properties"]["max_bytes"]["maximum"] == 4 * 1024 * 1024
    assert WriteFileTool().schema["properties"]["max_bytes"]["default"] == 256 * 1024
    assert WriteFileTool().schema["properties"]["max_bytes"]["maximum"] == 4 * 1024 * 1024
    assert ListFilesTool().schema["properties"]["limit"]["default"] == 500
    assert ListFilesTool().schema["properties"]["limit"]["maximum"] == 10_000
    assert ExecCommandTool().schema["properties"]["max_output_bytes"]["default"] == 50_000
    assert ExecCommandTool().schema["properties"]["max_output_bytes"]["maximum"] == 200_000
    assert ExecCommandTool().schema["properties"]["timeout_s"]["default"] == 60
    assert ExecCommandTool().schema["properties"]["timeout_s"]["maximum"] == 600
    assert ExecCommandTool().schema["properties"]["argv"]["minItems"] == 1
    assert ExecCommandTool().schema["properties"]["argv"]["items"] == {
        "type": "string",
        "minLength": 1,
        "pattern": r"\S",
    }
    assert ExecCommandTool().schema["properties"]["shell"]["minLength"] == 1
    assert ExecCommandTool().schema["properties"]["shell"]["pattern"] == r"\S"
    assert "oneOf" not in ExecCommandTool().schema
    assert "anyOf" not in ExecCommandTool().schema
    assert "allOf" not in ExecCommandTool().schema


def test_workspace_tools_read_write_and_list_files(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    write_result = asyncio.run(
        WriteFileTool().run(ctx, {"path": "notes/result.txt", "content": "hello"})
    )
    read_result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes/result.txt"}))
    list_result = asyncio.run(ListFilesTool().run(ctx, {"pattern": "**/*.txt"}))

    assert write_result.is_error is False
    assert write_result.structured == {
        "path": "notes/result.txt",
        "bytes": 5,
        "encoding": "utf-8",
    }
    assert read_result.content == "hello"
    assert read_result.structured == {
        "path": "notes/result.txt",
        "bytes": 5,
        "total_bytes": 5,
        "encoding": "utf-8",
        "truncated": False,
    }
    assert list_result.content == "notes/result.txt"
    assert list_result.structured == {
        "pattern": "**/*.txt",
        "files": ["notes/result.txt"],
        "total_files": 1,
        "truncated": False,
    }


def test_workspace_tools_return_error_without_workspace():
    ctx = ToolContext(session_id="sess_1")

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes/result.txt"}))

    assert result.is_error is True
    assert result.content == "No workspace configured for this tool call."


def test_exec_command_tool_runs_process_and_reports_failures(tmp_path):
    ctx = ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path))

    ok = asyncio.run(
        ExecCommandTool().run(
            ctx,
            {
                "kind": "process",
                "argv": [sys.executable, "-c", "print('ok')"],
            },
        )
    )
    failed = asyncio.run(
        ExecCommandTool().run(
            ctx,
            {
                "kind": "process",
                "argv": [sys.executable, "-c", "import sys; sys.exit(3)"],
            },
        )
    )

    assert ok.is_error is False
    assert ok.content == "ok"
    assert ok.structured["exit_code"] == 0
    assert ok.structured["stdout_truncated"] is False
    assert ok.structured["stderr_truncated"] is False
    assert failed.is_error is True
    assert failed.structured["exit_code"] == 3


def test_builtin_tools_truncate_model_facing_large_outputs(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    file_ctx = ToolContext(session_id="sess_1", workspace=workspace)
    run_ctx = ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path))

    asyncio.run(WriteFileTool().run(file_ctx, {"path": "large.txt", "content": "abcdef"}))
    asyncio.run(WriteFileTool().run(file_ctx, {"path": "other.txt", "content": ""}))
    read_result = asyncio.run(ReadFileTool().run(file_ctx, {"path": "large.txt", "max_bytes": 3}))
    list_result = asyncio.run(ListFilesTool().run(file_ctx, {"pattern": "*.txt", "limit": 1}))
    command_result = asyncio.run(
        ExecCommandTool().run(
            run_ctx,
            {
                "argv": [sys.executable, "-c", "print('abcdef')"],
                "max_output_bytes": 3,
            },
        )
    )

    assert read_result.content == "abc\n\n[file truncated]"
    assert read_result.structured["truncated"] is True
    assert read_result.structured["total_bytes"] == 6
    assert list_result.content.endswith("[file list truncated]")
    assert list_result.structured["total_files"] is None
    assert list_result.structured["truncated"] is True
    assert command_result.structured["stdout"] == "abc"
    assert command_result.structured["stdout_truncated"] is True
    assert "[output truncated]" in command_result.content


def test_write_file_tool_refuses_oversized_content(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    ctx = ToolContext(session_id="sess_1", workspace=workspace)

    result = asyncio.run(
        WriteFileTool().run(
            ctx,
            {
                "path": "large.txt",
                "content": "abcdef",
                "max_bytes": 3,
            },
        )
    )

    assert result.is_error is True
    assert result.content == ("Write refused: content is 6 bytes, which exceeds max_bytes=3.")
    assert result.structured == {
        "path": "large.txt",
        "bytes": 6,
        "max_bytes": 3,
        "encoding": "utf-8",
    }
    assert not (tmp_path / "large.txt").exists()


def test_exec_command_tool_applies_default_and_max_timeout(tmp_path):
    runner = RecordingRunner()
    ctx = ToolContext(session_id="sess_1", runner=runner)

    result = asyncio.run(
        ExecCommandTool().run(
            ctx,
            {
                "argv": [sys.executable, "-c", "print('ok')"],
            },
        )
    )

    assert result.is_error is False
    assert runner.timeout_s == 60
    with pytest.raises(ValueError, match="at most 600"):
        asyncio.run(
            ExecCommandTool().run(
                ctx,
                {
                    "argv": [sys.executable, "-c", "print('ok')"],
                    "timeout_s": 601,
                },
            )
        )


def test_exec_command_tool_reports_timeout_and_cancellation():
    timed_out_runner = RecordingRunner(ExecResult(exit_code=-9, timed_out=True))
    cancelled_runner = RecordingRunner(ExecResult(exit_code=-9, cancelled=True))

    timed_out = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=timed_out_runner),
            {
                "argv": [sys.executable, "-c", "print('ok')"],
                "timeout_s": 3,
            },
        )
    )
    cancelled = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="sess_1", runner=cancelled_runner),
            {
                "argv": [sys.executable, "-c", "print('ok')"],
            },
        )
    )

    assert timed_out.is_error is True
    assert timed_out.content == "Command timed out after 3 seconds."
    assert timed_out.structured["timed_out"] is True
    assert cancelled.is_error is True
    assert cancelled.content == "Command was cancelled."
    assert cancelled.structured["cancelled"] is True


def test_exec_command_tool_returns_error_without_runner():
    ctx = ToolContext(session_id="sess_1")

    result = asyncio.run(
        ExecCommandTool().run(
            ctx,
            {
                "kind": "process",
                "argv": [sys.executable, "-c", "print('ok')"],
            },
        )
    )

    assert result.is_error is True
    assert result.content == "No runner configured for this tool call."


def test_runtime_passes_environment_services_to_tool_context(tmp_path):
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    runner = LocalRunner(tmp_path)
    tool = ContextRecordingTool()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_1",
                    name="record_context",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-dev"),
            workspace=workspace,
            runner=runner,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "record context")],
            ),
        )
    )

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert tool.context is not None
    assert tool.context.environment_name == "local-dev"
    assert tool.context.workspace_id == "local"
    assert tool.context.workspace is workspace
    assert tool.context.runner is runner


async def _collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]
