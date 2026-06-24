from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import pytest

from cayu.runners import E2BRunner, ExecResult
from cayu.workspaces import DEFAULT_E2B_WORKSPACE_LIST_DEPTH, E2BWorkspace


class FakeFileType(Enum):
    FILE = "file"
    DIR = "dir"


@dataclass
class FakeEntry:
    path: str
    type: str | FakeFileType
    size: int = 0
    symlink_target: str | None = None


class FakeStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        await asyncio.sleep(0)
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


class FakeE2BFs:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs = {"/", "/home", "/home/user", "/home/user/workspace"}
        self.symlinks: dict[str, str] = {}
        self.fail_list_paths: set[str] = set()
        self.fail_info_paths: set[str] = set()
        self.write_calls: list[tuple[str, bytes, dict[str, Any]]] = []
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

    async def read(self, path: str, *, format: str, **kwargs: Any) -> FakeStream:
        assert format == "stream"
        content = self.files[path]
        return FakeStream([content[:2], content[2:]])

    async def write(self, path: str, content: bytes, **kwargs: Any) -> None:
        self.write_calls.append((path, content, dict(kwargs)))
        parent = posix_parent(path)
        self.dirs.add(parent)
        self.files[path] = content

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


def posix_parent(path: str) -> str:
    parent = path.rsplit("/", 1)[0]
    return parent or "/"


def _workspace() -> tuple[E2BWorkspace, FakeE2BFs]:
    fs = FakeE2BFs()
    runner = E2BRunner(FakeSandbox(fs), e2b_module=object())
    return (
        E2BWorkspace(
            runner,
            workspace_id="e2b-workspace",
            user="sandbox-user",
            request_timeout_s=5,
        ),
        fs,
    )


def _replace_runner_exec(workspace: E2BWorkspace, func: Any) -> None:
    runner = cast("Any", workspace.runner)
    runner.exec = func


def test_e2b_workspace_reads_writes_and_lists_native_fs() -> None:
    workspace, fs = _workspace()

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
    assert fs.write_calls[0] == (
        "/home/user/workspace/notes/a.txt",
        b"abcdef",
        {"user": "sandbox-user", "request_timeout": 5.0},
    )
    assert fs.list_calls[0] == (
        "/home/user/workspace",
        DEFAULT_E2B_WORKSPACE_LIST_DEPTH,
        {"user": "sandbox-user", "request_timeout": 5.0},
    )


def test_e2b_workspace_deletes_files_through_runner_exec() -> None:
    workspace, fs = _workspace()
    fs.files["/home/user/workspace/notes/a.txt"] = b"abcdef"
    calls: list[tuple[list[str], str | None]] = []

    async def fake_exec(command, *, cwd=None, **kwargs):
        calls.append((list(command.argv or []), cwd))
        assert command.argv == ["rm", "-f", "--", "/home/user/workspace/notes/a.txt"]
        assert cwd is None
        fs.files.pop("/home/user/workspace/notes/a.txt", None)
        return ExecResult(exit_code=0)

    _replace_runner_exec(workspace, fake_exec)

    asyncio.run(workspace.delete("notes/a.txt"))

    assert calls == [(["rm", "-f", "--", "/home/user/workspace/notes/a.txt"], None)]
    assert "/home/user/workspace/notes/a.txt" not in fs.files


def test_e2b_workspace_uses_default_bounds() -> None:
    workspace, _ = _workspace()
    workspace.default_read_limit_bytes = 4
    workspace.default_list_limit = 1

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


def test_e2b_workspace_rejects_symlink_escape() -> None:
    workspace, fs = _workspace()
    fs.symlinks["/home/user/workspace/link"] = "/etc"
    fs.files["/etc/passwd"] = b"secret"

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.read_bytes("link/passwd"))


def test_e2b_workspace_rejects_writing_through_symlink_leaf() -> None:
    workspace, fs = _workspace()
    fs.symlinks["/home/user/workspace/out"] = "/etc/passwd"

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("out", b"secret"))

    assert "/etc/passwd" not in fs.files


def test_e2b_workspace_rejects_writing_symlink_leaf_inside_workspace() -> None:
    workspace, fs = _workspace()
    fs.symlinks["/home/user/workspace/link"] = "/home/user/workspace/target.txt"
    fs.files["/home/user/workspace/target.txt"] = b"keep"

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("link", b"overwrite"))

    assert fs.files["/home/user/workspace/target.txt"] == b"keep"


def test_e2b_workspace_rejects_deleting_through_symlink_leaf() -> None:
    workspace, fs = _workspace()
    fs.symlinks["/home/user/workspace/out"] = "/etc/passwd"
    called = False

    async def fake_exec(command, **kwargs):
        nonlocal called
        called = True
        return ExecResult(exit_code=0)

    _replace_runner_exec(workspace, fake_exec)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.delete("out"))

    assert called is False


def test_e2b_workspace_rejects_deleting_directory_before_exec() -> None:
    workspace, fs = _workspace()
    fs.dirs.add("/home/user/workspace/notes")
    called = False

    async def fake_exec(command, **kwargs):
        nonlocal called
        called = True
        return ExecResult(exit_code=0)

    _replace_runner_exec(workspace, fake_exec)

    with pytest.raises(IsADirectoryError, match="not a file"):
        asyncio.run(workspace.delete("notes"))

    assert called is False


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

    fs.fail_info_paths.add("/home/user/workspace/a.txt")
    with pytest.raises(RuntimeError, match="Failed to inspect"):
        asyncio.run(workspace.write_bytes("a.txt", b"no"))


def test_e2b_workspace_rejects_closed_runner() -> None:
    workspace, _ = _workspace()
    workspace.runner._closed = True

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(workspace.list("**/*"))
