from __future__ import annotations

import asyncio
import sys

import pytest

from cayu.core.tools import ToolContext
from cayu.runners import LocalRunner
from cayu.tools import ListFilesTool, ReadFileTool, WriteFileTool
from cayu.workspaces import RunnerWorkspace


def _workspace(root) -> RunnerWorkspace:
    return RunnerWorkspace(
        LocalRunner(root, inherit_env=False),
        workspace_id="runner",
        python_executable=sys.executable,
    )


def test_runner_workspace_reads_writes_and_lists_through_runner(tmp_path) -> None:
    workspace = _workspace(tmp_path)

    asyncio.run(workspace.write_bytes("notes/a.txt", b"abcdef"))
    asyncio.run(workspace.write_bytes("root.txt", b"root"))

    read_result = asyncio.run(workspace.read_bytes("notes/a.txt", max_bytes=3))
    list_result = asyncio.run(workspace.list("**/*.txt", limit=10))

    assert read_result.content == b"abc"
    assert read_result.total_bytes == 6
    assert read_result.truncated is True
    assert list_result.paths == ("notes/a.txt", "root.txt")
    assert list_result.total_count == 2
    assert list_result.truncated is False


def test_runner_workspace_uses_default_remote_bounds(tmp_path) -> None:
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        workspace_id="runner",
        python_executable=sys.executable,
        default_read_limit_bytes=4,
        default_list_limit=1,
    )

    asyncio.run(workspace.write_bytes("a.txt", b"abcdef"))
    asyncio.run(workspace.write_bytes("b.txt", b""))

    read_result = asyncio.run(workspace.read_bytes("a.txt"))
    list_result = asyncio.run(workspace.list("*.txt"))

    assert read_result.content == b"abcd"
    assert read_result.total_bytes == 6
    assert read_result.truncated is True
    assert len(list_result.paths) == 1
    assert list_result.total_count == 2
    assert list_result.truncated is True


def test_runner_workspace_rejects_path_and_pattern_escape(tmp_path) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="relative"):
        asyncio.run(workspace.read_bytes(str(tmp_path / "file.txt")))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("../outside.txt", b"no"))

    with pytest.raises(ValueError, match="pattern"):
        asyncio.run(workspace.list("../*"))

    with pytest.raises(ValueError, match="relative"):
        RunnerWorkspace(
            LocalRunner(tmp_path, inherit_env=False),
            cwd="/workspace",
            python_executable=sys.executable,
        )


def test_runner_workspace_rejects_symlink_escape_inside_runner_root(tmp_path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}_outside.txt"
    outside.write_bytes(b"secret")
    (tmp_path / "link.txt").symlink_to(outside)
    workspace = _workspace(tmp_path)

    try:
        with pytest.raises(ValueError, match="escapes"):
            asyncio.run(workspace.read_bytes("link.txt"))
    finally:
        outside.unlink(missing_ok=True)


def test_runner_workspace_reports_runner_failure_when_python_cannot_start(tmp_path) -> None:
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        workspace_id="runner",
        python_executable="missing-python-executable",
    )

    with pytest.raises(RuntimeError, match="Runner workspace operation failed"):
        asyncio.run(workspace.read_bytes("notes/result.txt"))


def test_builtin_file_tools_use_runner_workspace(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    ctx = ToolContext(
        session_id="session",
        workspace_id=workspace.id,
        workspace=workspace,
    )

    write_result = asyncio.run(
        WriteFileTool().run(
            ctx,
            {"path": "notes/result.txt", "content": "runner workspace"},
        )
    )
    read_result = asyncio.run(
        ReadFileTool().run(
            ctx,
            {"path": "notes/result.txt"},
        )
    )
    list_result = asyncio.run(
        ListFilesTool().run(
            ctx,
            {"pattern": "**/*.txt"},
        )
    )

    assert write_result.is_error is False
    assert "Wrote 16 bytes" in write_result.content
    assert read_result.content == "runner workspace"
    assert list_result.structured == {
        "pattern": "**/*.txt",
        "files": ["notes/result.txt"],
        "total_files": 1,
        "truncated": False,
    }
