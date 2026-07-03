from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import pytest
from guard_harness import make_local_guard_exec

from cayu.runners import E2BRunner, ExecResult
from cayu.workspaces import DEFAULT_E2B_WORKSPACE_LIST_DEPTH, E2BWorkspace
from cayu.workspaces._guest_guard import GUEST_PYTHON


class FakeFileType(Enum):
    FILE = "file"
    DIR = "dir"


@dataclass
class FakeEntry:
    path: str
    type: str | FakeFileType
    size: int = 0
    symlink_target: str | None = None


class FakeE2BFs:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs = {"/", "/home", "/home/user", "/home/user/workspace"}
        self.symlinks: dict[str, str] = {}
        self.fail_list_paths: set[str] = set()
        self.fail_info_paths: set[str] = set()
        self.list_calls: list[tuple[str, int | None, dict[str, Any]]] = []

    async def get_info(self, path: str, **kwargs: Any) -> FakeEntry:
        if path in self.fail_info_paths:
            raise RuntimeError("stat failed")
        if path in self.symlinks:
            return FakeEntry(
                path=path,
                type=FakeFileType.FILE,
                size=0,
                symlink_target=self.symlinks[path],
            )
        if path in self.files:
            return FakeEntry(path=path, type=FakeFileType.FILE, size=len(self.files[path]))
        if path in self.dirs:
            return FakeEntry(path=path, type=FakeFileType.DIR, size=0)
        raise FileNotFoundError(path)

    async def list(
        self,
        path: str,
        *,
        depth: int | None,
        **kwargs: Any,
    ) -> Sequence[FakeEntry]:
        self.list_calls.append((path, depth, dict(kwargs)))
        if path in self.fail_list_paths:
            raise RuntimeError("list failed")
        entries: list[FakeEntry] = []
        prefix = path.rstrip("/") + "/"
        for directory in self.dirs:
            if directory != path and directory.startswith(prefix):
                entries.append(FakeEntry(path=directory, type=FakeFileType.DIR))
        for file_path, content in self.files.items():
            if file_path.startswith(prefix):
                entries.append(FakeEntry(path=file_path, type=FakeFileType.FILE, size=len(content)))
        for link_path, target in self.symlinks.items():
            if link_path.startswith(prefix):
                entries.append(
                    FakeEntry(
                        path=link_path,
                        type=FakeFileType.FILE,
                        symlink_target=target,
                    )
                )
        return entries


class FakeSandbox:
    def __init__(self, fs: FakeE2BFs) -> None:
        self.sandbox_id = "sbx_workspace"
        self.files = fs


def _workspace(root: str | None = None) -> tuple[E2BWorkspace, FakeE2BFs]:
    fs = FakeE2BFs()
    runner = E2BRunner(FakeSandbox(fs), e2b_module=object())
    kwargs: dict[str, Any] = {}
    if root is not None:
        kwargs["root"] = root
    return (
        E2BWorkspace(
            runner,
            workspace_id="e2b-workspace",
            user="sandbox-user",
            request_timeout_s=5,
            **kwargs,
        ),
        fs,
    )


def _replace_runner_exec(workspace: E2BWorkspace, func: Any) -> None:
    runner = cast("Any", workspace.runner)
    runner.exec = func


def _guard_workspace(tmp_path: Path) -> tuple[E2BWorkspace, Any]:
    """Workspace rooted at tmp_path whose exec runs the real guard locally."""

    workspace, _ = _workspace(root=str(tmp_path))
    fake_exec = make_local_guard_exec()
    _replace_runner_exec(workspace, fake_exec)
    return workspace, fake_exec


def test_e2b_workspace_reads_and_writes_through_guest_guard(tmp_path: Path) -> None:
    workspace, fake_exec = _guard_workspace(tmp_path)

    asyncio.run(workspace.write_bytes("notes/a.txt", b"abcdef"))
    read_result = asyncio.run(workspace.read_bytes("notes/a.txt", max_bytes=3))

    assert (tmp_path / "notes" / "a.txt").read_bytes() == b"abcdef"
    assert read_result.content == b"abc"
    assert read_result.total_bytes == 6
    assert read_result.truncated is True
    assert all(command.argv[0] == GUEST_PYTHON for command in fake_exec.calls)


def test_e2b_workspace_read_missing_file_raises_not_found(tmp_path: Path) -> None:
    workspace, _ = _guard_workspace(tmp_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        asyncio.run(workspace.read_bytes("missing.txt"))


def test_e2b_workspace_read_rejects_directory(tmp_path: Path) -> None:
    workspace, _ = _guard_workspace(tmp_path)
    (tmp_path / "notes").mkdir()

    with pytest.raises(FileNotFoundError, match="not found"):
        asyncio.run(workspace.read_bytes("notes"))


def test_e2b_workspace_deletes_files_through_guest_guard(tmp_path: Path) -> None:
    workspace, _ = _guard_workspace(tmp_path)
    target = tmp_path / "notes" / "a.txt"
    target.parent.mkdir()
    target.write_bytes(b"abcdef")

    asyncio.run(workspace.delete("notes/a.txt"))
    asyncio.run(workspace.delete("notes/a.txt"))  # missing file is a no-op

    assert not target.exists()


def test_e2b_workspace_uses_default_read_limit(tmp_path: Path) -> None:
    workspace, _ = _guard_workspace(tmp_path)
    workspace.default_read_limit_bytes = 4
    (tmp_path / "a.txt").write_bytes(b"abcdef")

    read_result = asyncio.run(workspace.read_bytes("a.txt"))

    assert read_result.content == b"abcd"
    assert read_result.total_bytes == 6
    assert read_result.truncated is True


def test_e2b_workspace_list_uses_native_fs_and_default_bounds() -> None:
    workspace, fs = _workspace()
    workspace.default_list_limit = 1
    fs.files["/home/user/workspace/a.txt"] = b"abcdef"
    fs.files["/home/user/workspace/b.txt"] = b""

    list_result = asyncio.run(workspace.list("*.txt"))

    assert len(list_result.paths) == 1
    assert list_result.total_count == 2
    assert list_result.truncated is True
    assert fs.list_calls[-1] == (
        "/home/user/workspace",
        DEFAULT_E2B_WORKSPACE_LIST_DEPTH,
        {"user": "sandbox-user", "request_timeout": 5.0},
    )


def test_e2b_workspace_list_pattern_is_anchored() -> None:
    workspace, fs = _workspace()
    fs.files["/home/user/workspace/root.txt"] = b"root"
    fs.files["/home/user/workspace/notes/a.txt"] = b"nested"
    fs.dirs.add("/home/user/workspace/notes")

    top_level = asyncio.run(workspace.list("*.txt"))
    recursive = asyncio.run(workspace.list("**/*.txt"))

    assert top_level.paths == ("root.txt",)
    assert top_level.total_count == 1
    assert recursive.paths == ("notes/a.txt", "root.txt")
    assert recursive.total_count == 2


def test_e2b_workspace_rejects_path_and_pattern_escape() -> None:
    workspace, _ = _workspace()

    with pytest.raises(ValueError, match="relative"):
        asyncio.run(workspace.read_bytes("/home/user/workspace/file.txt"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("../outside.txt", b"no"))

    with pytest.raises(ValueError, match="pattern"):
        asyncio.run(workspace.list("../*"))

    with pytest.raises(ValueError, match="absolute"):
        E2BWorkspace(workspace.runner, root="workspace")


def test_e2b_workspace_rejects_symlink_component_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "passwd").write_bytes(b"secret")
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "link").symlink_to(outside)
    workspace, _ = _guard_workspace(root)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.read_bytes("link/passwd"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("link/passwd", b"overwrite"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.delete("link/passwd"))

    assert (outside / "passwd").read_bytes() == b"secret"


def test_e2b_workspace_rejects_symlink_leaf(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "out").symlink_to(outside)
    workspace, _ = _guard_workspace(root)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.read_bytes("out"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("out", b"overwrite"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.delete("out"))

    assert outside.read_bytes() == b"secret"
    assert (root / "out").is_symlink()


def test_e2b_workspace_rejects_symlink_leaf_inside_workspace(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_bytes(b"keep")
    os.symlink(tmp_path / "target.txt", tmp_path / "link")
    workspace, _ = _guard_workspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("link", b"overwrite"))

    assert (tmp_path / "target.txt").read_bytes() == b"keep"


def test_e2b_workspace_rejects_deleting_directory(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    workspace, _ = _guard_workspace(tmp_path)

    with pytest.raises(IsADirectoryError, match="not a file"):
        asyncio.run(workspace.delete("notes"))

    assert (tmp_path / "notes").is_dir()


def test_e2b_workspace_skips_symlinks_when_listing() -> None:
    workspace, fs = _workspace()
    fs.files["/home/user/workspace/a.txt"] = b"a"
    fs.symlinks["/home/user/workspace/link.txt"] = "/etc/passwd"

    result = asyncio.run(workspace.list("*.txt"))

    assert result.paths == ("a.txt",)


def test_e2b_workspace_surfaces_operational_failures() -> None:
    workspace, fs = _workspace()
    fs.fail_list_paths.add("/home/user/workspace")

    with pytest.raises(RuntimeError, match="Failed to list"):
        asyncio.run(workspace.list("**/*"))

    async def failing_exec(command: Any, **kwargs: Any) -> ExecResult:
        return ExecResult(exit_code=1, stderr="disk exploded")

    _replace_runner_exec(workspace, failing_exec)
    with pytest.raises(RuntimeError, match="Failed to write.*disk exploded"):
        asyncio.run(workspace.write_bytes("a.txt", b"no"))


def test_e2b_workspace_surfaces_missing_guest_python() -> None:
    workspace, _ = _workspace()

    async def missing_python_exec(command: Any, **kwargs: Any) -> ExecResult:
        return ExecResult(exit_code=127, stderr="python3: command not found")

    _replace_runner_exec(workspace, missing_python_exec)
    with pytest.raises(RuntimeError, match="python3 is required inside the guest"):
        asyncio.run(workspace.read_bytes("a.txt"))


def test_e2b_workspace_rejects_truncated_guard_output() -> None:
    workspace, _ = _workspace()

    async def truncated_exec(command: Any, **kwargs: Any) -> ExecResult:
        return ExecResult(exit_code=0, stdout="ok 6\nabc", stdout_truncated=True)

    _replace_runner_exec(workspace, truncated_exec)
    with pytest.raises(RuntimeError, match="truncated"):
        asyncio.run(workspace.read_bytes("a.txt"))


def test_e2b_workspace_rejects_closed_runner() -> None:
    workspace, _ = _workspace()
    workspace.runner._closed = True

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(workspace.list("**/*"))

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(workspace.read_bytes("a.txt"))
