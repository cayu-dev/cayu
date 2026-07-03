"""Unified structured error contract for built-in tools.

Bad model-supplied arguments must produce structured ``is_error`` results
with ``{"error": "invalid_arguments"}`` (the contract the knowledge tools
established) instead of raising raw ``ValueError`` into the framework
exception path. Host misconfiguration (``TypeError``) must still raise.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from cayu.artifacts import LocalArtifactStore
from cayu.core.tools import ToolContext
from cayu.runners import LocalRunner
from cayu.runtime.sessions import InMemorySessionStore
from cayu.tools import (
    ExecCommandTool,
    ListArtifactsTool,
    ListFilesTool,
    ReadFileTool,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
    WriteFileTool,
)
from cayu.tools._errors import invalid_tool_arguments_result
from cayu.workspaces import LocalWorkspace


class _UnusedRuntime:
    """Argument validation must fail before the runtime is ever touched."""

    def run(self, request):  # pragma: no cover - must not be reached
        raise AssertionError("runtime.run must not be called for invalid arguments")

    def interrupt_session(self, request):  # pragma: no cover - must not be reached
        raise AssertionError("interrupt_session must not be called for invalid arguments")


def _workspace_ctx(tmp_path) -> ToolContext:
    return ToolContext(
        session_id="sess_1",
        workspace=LocalWorkspace(tmp_path, workspace_id="local"),
        artifact_store=LocalArtifactStore(tmp_path / "artifacts", store_id="artifacts"),
    )


def _assert_invalid_arguments(result, *, match: str) -> None:
    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert match in result.content


def test_invalid_tool_arguments_result_shape():
    result = invalid_tool_arguments_result(ValueError("bad input"))
    assert result.is_error is True
    assert result.content == "bad input"
    assert result.structured == {"error": "invalid_arguments"}


def test_read_file_invalid_arguments_return_structured_error(tmp_path):
    ctx = _workspace_ctx(tmp_path)

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "a.txt", "max_bytes": "big"}))
    _assert_invalid_arguments(result, match="must be an integer")

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "   "}))
    _assert_invalid_arguments(result, match="cannot be blank")


def test_write_file_invalid_arguments_return_structured_error(tmp_path):
    ctx = _workspace_ctx(tmp_path)

    result = asyncio.run(WriteFileTool().run(ctx, {"path": 7, "content": "x"}))
    _assert_invalid_arguments(result, match="must be a string")

    result = asyncio.run(
        WriteFileTool().run(ctx, {"path": "a.txt", "content": "x", "max_bytes": 0})
    )
    _assert_invalid_arguments(result, match="greater than zero")


def test_list_files_invalid_pattern_returns_structured_error(tmp_path):
    ctx = _workspace_ctx(tmp_path)

    result = asyncio.run(ListFilesTool().run(ctx, {"pattern": 5}))
    _assert_invalid_arguments(result, match="must be a string")


def test_list_artifacts_invalid_scope_returns_structured_error(tmp_path):
    ctx = _workspace_ctx(tmp_path)

    result = asyncio.run(ListArtifactsTool().run(ctx, {"scope": "galaxy"}))
    _assert_invalid_arguments(result, match="unsupported scope")


def test_exec_command_invalid_arguments_return_structured_error(tmp_path):
    ctx = ToolContext(session_id="sess_1", runner=LocalRunner(tmp_path))
    tool = ExecCommandTool()

    result = asyncio.run(tool.run(ctx, {"kind": "teleport"}))
    _assert_invalid_arguments(result, match="`kind` must be `process` or `shell`")

    result = asyncio.run(tool.run(ctx, {"argv": "not-a-list"}))
    _assert_invalid_arguments(result, match="`argv` must be a list")


def test_subagent_tool_invalid_arguments_return_structured_error():
    tool = SubagentTool(
        _UnusedRuntime(),
        agents={"reviewer": SubagentSpec(agent_name="reviewer")},
    )
    ctx = ToolContext(session_id="sess_parent")

    result = asyncio.run(tool.run(ctx, {"agent": 42, "task": "review"}))
    _assert_invalid_arguments(result, match="must be a string")

    result = asyncio.run(tool.run(ctx, {"agent": "reviewer", "task": ""}))
    _assert_invalid_arguments(result, match="cannot be blank")


def test_subagent_result_tool_invalid_arguments_return_structured_error():
    tool = SubagentResultTool(InMemorySessionStore())
    ctx = ToolContext(session_id="sess_parent")

    result = asyncio.run(tool.run(ctx, {"child_session_id": "child", "timeout_s": -1}))
    _assert_invalid_arguments(result, match="timeout_s")

    result = asyncio.run(tool.run(ctx, {"child_session_id": "child", "max_chars": 0}))
    _assert_invalid_arguments(result, match="max_chars")


def test_host_misconfiguration_rejected_at_context_construction(tmp_path):
    class _NotAWorkspace:
        pass

    # Typed ToolContext handles reject misconfigured hosts up front instead
    # of deferring to a TypeError inside the first tool call.
    with pytest.raises(ValidationError, match="WorkspaceHandle"):
        ToolContext(session_id="sess_1", workspace=_NotAWorkspace())
