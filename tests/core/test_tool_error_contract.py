"""Unified structured error contract for built-in tools.

Bad model-supplied arguments must produce structured ``is_error`` results
with ``{"error": "invalid_arguments"}`` (the contract the knowledge tools
established) instead of raising raw ``ValueError`` into the framework
exception path. Host misconfiguration (``TypeError``) must still raise.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, StrictInt, ValidationError

from cayu.artifacts import LocalArtifactStore
from cayu.core.tools import ToolContext, ToolResult
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
from cayu.tools._errors import (
    invalid_tool_arguments_result,
    structured_invalid_arguments,
    tool_argument_validation,
)
from cayu.workspaces import LocalWorkspace


class _UnusedRuntime:
    """Argument validation must fail before the runtime is ever touched."""

    def run(self, request):  # pragma: no cover - must not be reached
        raise AssertionError("runtime.run must not be called for invalid arguments")

    def interrupt_session(self, request):  # pragma: no cover - must not be reached
        raise AssertionError("interrupt_session must not be called for invalid arguments")


class _OperationalValueErrorWorkspace(LocalWorkspace):
    async def read_bytes(self, path, *, max_bytes=None):
        raise ValueError("workspace read failed")

    async def write_bytes(self, path, content):
        raise ValueError("workspace write failed")

    async def list(self, pattern="**/*", *, limit=None):
        raise ValueError("workspace list failed")


class _OperationalValueErrorArtifactStore(LocalArtifactStore):
    async def list(self, **kwargs):
        raise ValueError("artifact listing failed")


class _OperationalValueErrorRuntime:
    def run(self, request):
        del request
        return self._events()

    async def _events(self):
        raise ValueError("subagent runtime failed")
        yield  # pragma: no cover - keeps this an async iterator

    def interrupt_session(self, request):  # pragma: no cover - not reached
        raise AssertionError("interrupt_session must not be called")


class _OperationalValueErrorSessionStore(InMemorySessionStore):
    async def load(self, session_id):
        del session_id
        raise ValueError("session load failed")


class _StrictArguments(BaseModel):
    count: StrictInt


class _ValidationBoundaryTool:
    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        phase = args["phase"]
        if phase == "value":
            with tool_argument_validation():
                raise ValueError("bad model value")
        if phase == "pydantic":
            with tool_argument_validation():
                _StrictArguments.model_validate({"count": "many"})
        if phase == "operational":
            raise ValueError("backend state changed")
        if phase == "host":
            raise TypeError("host misconfigured")
        return ToolResult(content="ok")


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


@pytest.mark.parametrize("phase", ["value", "pydantic"])
def test_explicit_argument_validation_boundary_returns_structured_error(phase):
    result = asyncio.run(
        _ValidationBoundaryTool().run(ToolContext(session_id="sess_1"), {"phase": phase})
    )

    _assert_invalid_arguments(result, match="bad model value" if phase == "value" else "count")


def test_unmarked_operational_value_error_propagates():
    with pytest.raises(ValueError, match="backend state changed"):
        asyncio.run(
            _ValidationBoundaryTool().run(
                ToolContext(session_id="sess_1"),
                {"phase": "operational"},
            )
        )


def test_type_error_inside_tool_still_propagates():
    with pytest.raises(TypeError, match="host misconfigured"):
        asyncio.run(
            _ValidationBoundaryTool().run(
                ToolContext(session_id="sess_1"),
                {"phase": "host"},
            )
        )


def test_read_file_invalid_arguments_return_structured_error(tmp_path):
    ctx = _workspace_ctx(tmp_path)

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "a.txt", "max_bytes": "big"}))
    _assert_invalid_arguments(result, match="must be an integer")

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "   "}))
    _assert_invalid_arguments(result, match="cannot be blank")

    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "pages": "1"}))
    _assert_invalid_arguments(result, match="only valid for PDF")

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "../outside.txt"}))
    _assert_invalid_arguments(result, match="escapes the workspace root")

    result = asyncio.run(ReadFileTool().run(ctx, {"artifact_id": "not-an-artifact-id"}))
    _assert_invalid_arguments(result, match="Artifact id must match")

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "bad\0path"}))
    _assert_invalid_arguments(result, match="NUL")

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "bad\ud800path"}))
    _assert_invalid_arguments(result, match="surrogate")

    result = asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt", "pages": "1\ud800"}))
    _assert_invalid_arguments(result, match="surrogate")


def test_write_file_invalid_arguments_return_structured_error(tmp_path):
    ctx = _workspace_ctx(tmp_path)

    result = asyncio.run(WriteFileTool().run(ctx, {"path": 7, "content": "x"}))
    _assert_invalid_arguments(result, match="must be a string")

    result = asyncio.run(
        WriteFileTool().run(ctx, {"path": "a.txt", "content": "x", "max_bytes": 0})
    )
    _assert_invalid_arguments(result, match="greater than zero")

    result = asyncio.run(WriteFileTool().run(ctx, {"path": "/outside.txt", "content": "x"}))
    _assert_invalid_arguments(result, match="must be relative")

    for path in (r"C:\outside.txt", r"\outside.txt", r"..\outside.txt"):
        result = asyncio.run(WriteFileTool().run(ctx, {"path": path, "content": "x"}))
        _assert_invalid_arguments(result, match="Workspace path")


def test_list_files_invalid_pattern_returns_structured_error(tmp_path):
    ctx = _workspace_ctx(tmp_path)

    result = asyncio.run(ListFilesTool().run(ctx, {"pattern": 5}))
    _assert_invalid_arguments(result, match="must be a string")

    result = asyncio.run(ListFilesTool().run(ctx, {"pattern": "../*"}))
    _assert_invalid_arguments(result, match="stay inside the workspace")

    result = asyncio.run(ListFilesTool().run(ctx, {"pattern": "bad\0pattern"}))
    _assert_invalid_arguments(result, match="NUL")

    result = asyncio.run(ListFilesTool().run(ctx, {"pattern": "bad\ud800pattern"}))
    _assert_invalid_arguments(result, match="surrogate")


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

    malformed_strings = (
        {"argv": ["echo", "bad\0argument"]},
        {"shell": "echo bad\0script"},
        {"argv": ["echo"], "env": {"BAD\0KEY": "value"}},
        {"argv": ["echo"], "env": {"KEY": "bad\0value"}},
        {"argv": ["echo"], "cwd": "bad\0cwd"},
    )
    for args in malformed_strings:
        result = asyncio.run(tool.run(ctx, args))
        _assert_invalid_arguments(result, match="NUL")

    invalid_portable_strings = (
        ({"argv": ["true"], "env": {"BAD=KEY": "value"}}, "must not contain '='"),
        ({"argv": ["true"], "env": {"KEY": "line\nvalue"}}, "newlines"),
        ({"argv": ["echo", "bad\ud800argument"]}, "surrogate"),
        ({"shell": "echo bad\ud800script"}, "surrogate"),
        ({"argv": ["true"], "stdin": "bad\ud800input"}, "surrogate"),
    )
    for args, message in invalid_portable_strings:
        result = asyncio.run(tool.run(ctx, args))
        _assert_invalid_arguments(result, match=message)


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

    result = asyncio.run(tool.run(ctx, {"agent": "reviewer", "task": "review", "metadata": []}))
    _assert_invalid_arguments(result, match="must be a JSON object")

    result = asyncio.run(tool.run(ctx, {"agent": "reviewer", "task": "bad\ud800task"}))
    _assert_invalid_arguments(result, match="surrogate")

    result = asyncio.run(
        tool.run(
            ctx,
            {
                "agent": "reviewer",
                "task": "review",
                "metadata": {"nested": ["bad\ud800value"]},
            },
        )
    )
    _assert_invalid_arguments(result, match="surrogate")


def test_subagent_result_tool_invalid_arguments_return_structured_error():
    tool = SubagentResultTool(InMemorySessionStore())
    ctx = ToolContext(session_id="sess_parent")

    result = asyncio.run(tool.run(ctx, {"child_session_id": "child", "timeout_s": -1}))
    _assert_invalid_arguments(result, match="timeout_s")

    result = asyncio.run(tool.run(ctx, {"child_session_id": "child", "max_chars": 0}))
    _assert_invalid_arguments(result, match="max_chars")

    result = asyncio.run(tool.run(ctx, {"child_session_id": "   "}))
    _assert_invalid_arguments(result, match="cannot be blank")

    result = asyncio.run(tool.run(ctx, {"child_session_id": "bad\ud800child"}))
    _assert_invalid_arguments(result, match="surrogate")

    result = asyncio.run(tool.run(ctx, {}))
    _assert_invalid_arguments(result, match="requires child_session_id")

    result = asyncio.run(tool.run(ctx, {"child_session_id": 42}))
    _assert_invalid_arguments(result, match="requires child_session_id")

    result = asyncio.run(tool.run(ctx, {"all": True, "child_session_id": "child"}))
    _assert_invalid_arguments(result, match="either child_session_id or all=true")


def test_workspace_operational_value_errors_propagate(tmp_path):
    ctx = ToolContext(
        session_id="sess_1",
        workspace=_OperationalValueErrorWorkspace(tmp_path, workspace_id="local"),
    )

    with pytest.raises(ValueError, match="workspace read failed"):
        asyncio.run(ReadFileTool().run(ctx, {"path": "notes.txt"}))

    with pytest.raises(ValueError, match="workspace write failed"):
        asyncio.run(WriteFileTool().run(ctx, {"path": "notes.txt", "content": "hello"}))

    with pytest.raises(ValueError, match="workspace list failed"):
        asyncio.run(ListFilesTool().run(ctx, {}))


def test_artifact_store_operational_value_error_propagates(tmp_path):
    ctx = ToolContext(
        session_id="sess_1",
        artifact_store=_OperationalValueErrorArtifactStore(
            tmp_path / "artifacts",
            store_id="artifacts",
        ),
    )

    with pytest.raises(ValueError, match="artifact listing failed"):
        asyncio.run(ListArtifactsTool().run(ctx, {}))


def test_subagent_runtime_operational_value_error_propagates():
    tool = SubagentTool(
        _OperationalValueErrorRuntime(),
        agents={"reviewer": SubagentSpec(agent_name="reviewer")},
    )

    with pytest.raises(ValueError, match="subagent runtime failed"):
        asyncio.run(
            tool.run(
                ToolContext(session_id="sess_parent"),
                {"agent": "reviewer", "task": "review"},
            )
        )


def test_subagent_session_store_operational_value_error_propagates():
    tool = SubagentResultTool(_OperationalValueErrorSessionStore())

    with pytest.raises(ValueError, match="session load failed"):
        asyncio.run(
            tool.run(
                ToolContext(session_id="sess_parent"),
                {"child_session_id": "child"},
            )
        )


def test_host_misconfiguration_rejected_at_context_construction(tmp_path):
    class _NotAWorkspace:
        pass

    # Typed ToolContext handles reject misconfigured hosts up front instead
    # of deferring to a TypeError inside the first tool call.
    with pytest.raises(ValidationError, match="WorkspaceHandle"):
        ToolContext(session_id="sess_1", workspace=_NotAWorkspace())
