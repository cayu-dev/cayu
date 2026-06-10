from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from cayu.runners import MicrosandboxRunner
from cayu.workspaces import MicrosandboxWorkspace


@dataclass
class FakeFsMetadata:
    kind: str
    size: int


@dataclass
class FakeFsEntry:
    path: str
    kind: str
    size: int = 0


class FakeReadStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        await asyncio.sleep(0)
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)


class FakeMicrosandboxFs:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs = {"/workspace"}
        self.symlinks: dict[str, str] = {}
        self.mkdir_calls: list[str] = []
        self.fail_list_paths: set[str] = set()
        self.fail_real_path_paths: set[str] = set()
        self.fail_stat_paths: set[str] = set()
        self.fail_mkdir_paths: set[str] = set()

    async def stat(self, path: str) -> FakeFsMetadata:
        path = self.real_path(path)
        if path in self.fail_stat_paths:
            raise RuntimeError("stat failed")
        if path in self.files:
            return FakeFsMetadata(kind="file", size=len(self.files[path]))
        if path in self.dirs:
            return FakeFsMetadata(kind="dir", size=0)
        raise FileNotFoundError(path)

    async def read_stream(self, path: str) -> FakeReadStream:
        path = self.real_path(path)
        content = self.files[path]
        return FakeReadStream([content[:2], content[2:]])

    async def read(self, path: str) -> bytes:
        path = self.real_path(path)
        return self.files[path]

    async def write(self, path: str, data: bytes) -> None:
        path = self.real_path(path)
        self.files[path] = data

    async def mkdir(self, path: str) -> None:
        self.mkdir_calls.append(path)
        if path in self.fail_mkdir_paths:
            raise RuntimeError("mkdir failed")
        self.dirs.add(path)

    async def list(self, path: str) -> list[FakeFsEntry]:
        path = self.real_path(path)
        if path in self.fail_list_paths:
            raise RuntimeError("list failed")
        entries: list[FakeFsEntry] = []
        prefix = path.rstrip("/") + "/"
        seen_dirs: set[str] = set()
        for directory in self.dirs:
            if directory == path or not directory.startswith(prefix):
                continue
            rel = directory.removeprefix(prefix)
            if "/" not in rel:
                entries.append(FakeFsEntry(path=directory, kind="dir"))
        for file_path, content in self.files.items():
            if not file_path.startswith(prefix):
                continue
            rel = file_path.removeprefix(prefix)
            if "/" in rel:
                child_dir = prefix + rel.split("/", 1)[0]
                if child_dir not in seen_dirs:
                    entries.append(FakeFsEntry(path=child_dir, kind="dir"))
                    seen_dirs.add(child_dir)
            else:
                entries.append(FakeFsEntry(path=file_path, kind="file", size=len(content)))
        return entries

    def real_path(self, path: str) -> str:
        if path in self.fail_real_path_paths:
            raise RuntimeError("realpath failed")
        for link_path, target_path in sorted(
            self.symlinks.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if path == link_path or path.startswith(f"{link_path}/"):
                suffix = path.removeprefix(link_path).lstrip("/")
                return target_path if not suffix else f"{target_path.rstrip('/')}/{suffix}"
        if path in self.files or path in self.dirs:
            return path
        parent = path.rsplit("/", 1)[0] or "/"
        if parent in self.dirs:
            return path
        raise FileNotFoundError(path)


class FakeSftp:
    def __init__(self, fs: FakeMicrosandboxFs) -> None:
        self.fs = fs
        self.closed = False

    async def real_path(self, path: str) -> str:
        return self.fs.real_path(path)

    async def close(self) -> None:
        self.closed = True


class FakeSshClient:
    def __init__(self, fs: FakeMicrosandboxFs) -> None:
        self.fs = fs
        self.closed = False

    async def sftp(self) -> FakeSftp:
        return FakeSftp(self.fs)

    async def close(self) -> None:
        self.closed = True


class FakeSsh:
    def __init__(self, fs: FakeMicrosandboxFs) -> None:
        self.fs = fs

    async def connect(self, *, sftp: bool = True) -> FakeSshClient:
        return FakeSshClient(self.fs)


class FakeSandbox:
    def __init__(self, fs: FakeMicrosandboxFs) -> None:
        self.fs = fs

    def ssh(self) -> FakeSsh:
        return FakeSsh(self.fs)


def _workspace() -> tuple[MicrosandboxWorkspace, FakeMicrosandboxFs]:
    fs = FakeMicrosandboxFs()
    runner = MicrosandboxRunner(
        FakeSandbox(fs),
        name="sandbox",
        sandbox_module=object(),
    )
    return MicrosandboxWorkspace(runner, workspace_id="sandbox-workspace"), fs


def test_microsandbox_workspace_reads_writes_and_lists_native_fs() -> None:
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
    assert "/workspace/notes" in fs.mkdir_calls


def test_microsandbox_workspace_uses_default_bounds() -> None:
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


def test_microsandbox_workspace_rejects_path_and_pattern_escape() -> None:
    workspace, _ = _workspace()

    with pytest.raises(ValueError, match="relative"):
        asyncio.run(workspace.read_bytes("/workspace/file.txt"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("../outside.txt", b"no"))

    with pytest.raises(ValueError, match="pattern"):
        asyncio.run(workspace.list("../*"))

    with pytest.raises(ValueError, match="absolute"):
        MicrosandboxWorkspace(workspace.runner, root="workspace")


def test_microsandbox_workspace_rejects_realpath_escape() -> None:
    workspace, fs = _workspace()
    fs.symlinks["/workspace/link"] = "/etc"
    fs.files["/etc/passwd"] = b"secret"

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.read_bytes("link/passwd"))


def test_microsandbox_workspace_surfaces_list_failure() -> None:
    workspace, fs = _workspace()
    fs.fail_list_paths.add("/workspace")

    with pytest.raises(RuntimeError, match="Failed to list"):
        asyncio.run(workspace.list("**/*"))


def test_microsandbox_workspace_read_surfaces_realpath_operational_failure() -> None:
    workspace, fs = _workspace()
    fs.files["/workspace/a.txt"] = b"x"
    fs.fail_real_path_paths.add("/workspace/a.txt")

    with pytest.raises(RuntimeError, match="Failed to resolve"):
        asyncio.run(workspace.read_bytes("a.txt"))


def test_microsandbox_workspace_write_surfaces_realpath_operational_failure() -> None:
    workspace, fs = _workspace()
    fs.fail_real_path_paths.add("/workspace/a.txt")

    with pytest.raises(RuntimeError, match="Failed to resolve"):
        asyncio.run(workspace.write_bytes("a.txt", b"no"))

    assert "/workspace/a.txt" not in fs.files


def test_microsandbox_workspace_read_surfaces_stat_operational_failure() -> None:
    workspace, fs = _workspace()
    fs.files["/workspace/a.txt"] = b"x"
    fs.fail_stat_paths.add("/workspace/a.txt")

    with pytest.raises(RuntimeError, match="Failed to stat"):
        asyncio.run(workspace.read_bytes("a.txt"))


def test_microsandbox_workspace_write_surfaces_mkdir_operational_failure() -> None:
    workspace, fs = _workspace()
    fs.fail_mkdir_paths.add("/workspace/notes")

    with pytest.raises(RuntimeError, match="Failed to create"):
        asyncio.run(workspace.write_bytes("notes/a.txt", b"x"))

    assert "/workspace/notes/a.txt" not in fs.files


def test_microsandbox_workspace_write_ignores_existing_directory_mkdir_failure() -> None:
    workspace, fs = _workspace()
    fs.dirs.add("/workspace/notes")
    fs.fail_mkdir_paths.add("/workspace/notes")

    asyncio.run(workspace.write_bytes("notes/a.txt", b"x"))

    assert fs.files["/workspace/notes/a.txt"] == b"x"


def test_microsandbox_workspace_read_rejects_closed_runner() -> None:
    workspace, fs = _workspace()
    fs.files["/workspace/a.txt"] = b"x"
    workspace.runner._closed = True

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(workspace.read_bytes("a.txt"))


def test_microsandbox_workspace_rejects_closed_runner() -> None:
    workspace, _ = _workspace()
    workspace.runner._closed = True

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(workspace.list("**/*"))
