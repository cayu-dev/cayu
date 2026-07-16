from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from guard_harness import make_local_guard_exec

from cayu.runners import ExecResult, MicrosandboxRunner
from cayu.workspaces import MicrosandboxWorkspace
from cayu.workspaces.microsandbox import _is_path_not_found_error


@dataclass
class FakeFsEntry:
    path: str
    kind: str
    size: int = 0


class FakeMicrosandboxFs:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.dirs = {"/workspace"}
        self.symlinks: dict[str, str] = {}
        self.fail_list_paths: set[str] = set()
        self.fail_real_path_paths: set[str] = set()
        self.connect_calls = 0
        self.sftp_calls = 0
        self.closed_clients = 0
        self.closed_sftps = 0
        # When set, the next `real_path` on the SFTP channel raises a
        # session-level error once, simulating a dropped connection.
        self.drop_session_once = False

    async def list(self, path: str) -> Sequence[FakeFsEntry]:
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
        if self.fs.drop_session_once:
            self.fs.drop_session_once = False
            raise ConnectionResetError("SFTP session dropped")
        return self.fs.real_path(path)

    async def close(self) -> None:
        self.closed = True
        self.fs.closed_sftps += 1


class FakeSshClient:
    def __init__(self, fs: FakeMicrosandboxFs) -> None:
        self.fs = fs
        self.closed = False

    async def sftp(self) -> FakeSftp:
        self.fs.sftp_calls += 1
        return FakeSftp(self.fs)

    async def close(self) -> None:
        self.closed = True
        self.fs.closed_clients += 1


class FakeSsh:
    def __init__(self, fs: FakeMicrosandboxFs) -> None:
        self.fs = fs

    async def open_client(
        self, *, user: str = "root", term: str | None = None, sftp: bool = True
    ) -> FakeSshClient:
        self.fs.connect_calls += 1
        return FakeSshClient(self.fs)


class FakeSandbox:
    def __init__(self, fs: FakeMicrosandboxFs) -> None:
        self.fs = fs

    def ssh(self) -> FakeSsh:
        return FakeSsh(self.fs)


def _workspace(root: str | None = None) -> tuple[MicrosandboxWorkspace, FakeMicrosandboxFs]:
    fs = FakeMicrosandboxFs()
    runner = MicrosandboxRunner(
        FakeSandbox(fs),
        name="sandbox",
        sandbox_module=object(),
    )
    kwargs: dict[str, Any] = {}
    if root is not None:
        kwargs["root"] = root
    return MicrosandboxWorkspace(runner, workspace_id="sandbox-workspace", **kwargs), fs


def _replace_runner_exec(workspace: MicrosandboxWorkspace, func: Any) -> None:
    runner = cast("Any", workspace._control_plane_runner())
    runner.exec = func


def _guard_workspace(tmp_path: Path) -> tuple[MicrosandboxWorkspace, Any]:
    """Workspace rooted at tmp_path whose exec runs the real guard locally."""

    workspace, _ = _workspace(root=str(tmp_path))
    fake_exec = make_local_guard_exec()
    _replace_runner_exec(workspace, fake_exec)
    return workspace, fake_exec


def test_microsandbox_workspace_reads_and_writes_through_guest_guard(tmp_path: Path) -> None:
    workspace, _ = _guard_workspace(tmp_path)

    asyncio.run(workspace.write_bytes("notes/a.txt", b"abcdef"))
    read_result = asyncio.run(workspace.read_bytes("notes/a.txt", max_bytes=3))

    assert (tmp_path / "notes" / "a.txt").read_bytes() == b"abcdef"
    assert read_result.content == b"abc"
    assert read_result.total_bytes == 6
    assert read_result.truncated is True


def test_microsandbox_workspace_read_missing_file_raises_not_found(tmp_path: Path) -> None:
    workspace, _ = _guard_workspace(tmp_path)

    with pytest.raises(FileNotFoundError, match="not found"):
        asyncio.run(workspace.read_bytes("missing.txt"))


def test_microsandbox_workspace_deletes_files_through_guest_guard(tmp_path: Path) -> None:
    workspace, _ = _guard_workspace(tmp_path)
    target = tmp_path / "notes" / "a.txt"
    target.parent.mkdir()
    target.write_bytes(b"abcdef")

    asyncio.run(workspace.delete("notes/a.txt"))
    asyncio.run(workspace.delete("notes/a.txt"))  # missing file is a no-op

    assert not target.exists()


def test_microsandbox_workspace_uses_default_read_limit(tmp_path: Path) -> None:
    workspace, _ = _guard_workspace(tmp_path)
    workspace.default_read_limit_bytes = 4
    (tmp_path / "a.txt").write_bytes(b"abcdef")

    read_result = asyncio.run(workspace.read_bytes("a.txt"))

    assert workspace.bounded_read_limit(10) == 4
    assert workspace.bounded_read_limit(2) == 2
    assert read_result.content == b"abcd"
    assert read_result.total_bytes == 6
    assert read_result.truncated is True


def test_microsandbox_workspace_list_uses_native_fs_and_default_bounds() -> None:
    workspace, fs = _workspace()
    workspace.default_list_limit = 1
    fs.files["/workspace/a.txt"] = b"abcdef"
    fs.files["/workspace/b.txt"] = b""

    list_result = asyncio.run(workspace.list("*.txt"))

    assert len(list_result.paths) == 1
    assert list_result.total_count == 2
    assert list_result.truncated is True


def test_microsandbox_workspace_list_pattern_is_anchored() -> None:
    workspace, fs = _workspace()
    fs.files["/workspace/root.txt"] = b"root"
    fs.files["/workspace/notes/a.txt"] = b"nested"
    fs.dirs.add("/workspace/notes")

    top_level = asyncio.run(workspace.list("*.txt"))
    recursive = asyncio.run(workspace.list("**/*.txt"))

    assert top_level.paths == ("root.txt",)
    assert top_level.total_count == 1
    assert recursive.paths == ("notes/a.txt", "root.txt")
    assert recursive.total_count == 2


def test_microsandbox_workspace_rejects_path_and_pattern_escape() -> None:
    workspace, _ = _workspace()

    with pytest.raises(ValueError, match="relative"):
        asyncio.run(workspace.read_bytes("/workspace/file.txt"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("../outside.txt", b"no"))

    with pytest.raises(ValueError, match="pattern"):
        asyncio.run(workspace.list("../*"))

    with pytest.raises(ValueError, match="absolute"):
        MicrosandboxWorkspace(workspace._control_plane_runner(), root="workspace")


def test_microsandbox_workspace_rejects_symlink_component_escape(tmp_path: Path) -> None:
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


def test_microsandbox_workspace_rejects_symlink_leaf_inside_workspace(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_bytes(b"keep")
    os.symlink(tmp_path / "target.txt", tmp_path / "link")
    workspace, _ = _guard_workspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("link", b"overwrite"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.delete("link"))

    assert (tmp_path / "target.txt").read_bytes() == b"keep"
    assert (tmp_path / "link").is_symlink()


def test_microsandbox_workspace_rejects_symlink_parent_inside_workspace(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "a.txt").write_bytes(b"keep")
    os.symlink(target, tmp_path / "link")
    workspace, _ = _guard_workspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("link/a.txt", b"overwrite"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.delete("link/a.txt"))

    assert (target / "a.txt").read_bytes() == b"keep"


def test_microsandbox_workspace_rejects_deleting_directory(tmp_path: Path) -> None:
    (tmp_path / "notes").mkdir()
    workspace, _ = _guard_workspace(tmp_path)

    with pytest.raises(IsADirectoryError, match="not a file"):
        asyncio.run(workspace.delete("notes"))

    assert (tmp_path / "notes").is_dir()


def test_microsandbox_workspace_surfaces_list_failure() -> None:
    workspace, fs = _workspace()
    fs.fail_list_paths.add("/workspace")

    with pytest.raises(RuntimeError, match="Failed to list"):
        asyncio.run(workspace.list("**/*"))


def test_microsandbox_workspace_list_surfaces_realpath_operational_failure() -> None:
    workspace, fs = _workspace()
    fs.fail_real_path_paths.add("/workspace")

    with pytest.raises(RuntimeError, match="Failed to resolve"):
        asyncio.run(workspace.list("**/*"))


def test_microsandbox_workspace_surfaces_guard_operational_failure() -> None:
    workspace, _ = _workspace()

    async def failing_exec(command: Any, **kwargs: Any) -> ExecResult:
        return ExecResult(exit_code=1, stderr="disk exploded")

    _replace_runner_exec(workspace, failing_exec)

    with pytest.raises(RuntimeError, match="Failed to read.*disk exploded"):
        asyncio.run(workspace.read_bytes("a.txt"))

    with pytest.raises(RuntimeError, match="Failed to write.*disk exploded"):
        asyncio.run(workspace.write_bytes("a.txt", b"no"))


def test_microsandbox_workspace_surfaces_missing_guest_python() -> None:
    workspace, _ = _workspace()

    async def missing_python_exec(command: Any, **kwargs: Any) -> ExecResult:
        return ExecResult(exit_code=127, stderr="python3: command not found")

    _replace_runner_exec(workspace, missing_python_exec)

    with pytest.raises(RuntimeError, match="python3 is required inside the guest"):
        asyncio.run(workspace.write_bytes("a.txt", b"x"))


def test_microsandbox_workspace_read_rejects_closed_runner() -> None:
    workspace, _ = _workspace()
    workspace._control_plane_runner()._closed = True

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(workspace.read_bytes("a.txt"))


def test_microsandbox_workspace_rejects_closed_runner() -> None:
    workspace, _ = _workspace()
    workspace._control_plane_runner()._closed = True

    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(workspace.list("**/*"))


def test_microsandbox_list_reuses_single_sftp_handshake() -> None:
    workspace, fs = _workspace()
    for index in range(20):
        fs.files[f"/workspace/f{index}.txt"] = b"x"

    list_result = asyncio.run(workspace.list("**/*"))

    assert list_result.total_count == 20
    # The whole listing resolves ~20 paths (plus the root) through one cached
    # SSH client and one SFTP channel instead of a handshake per path.
    assert fs.connect_calls == 1
    assert fs.sftp_calls == 1


def test_microsandbox_real_path_reconnects_after_dropped_session() -> None:
    workspace, fs = _workspace()
    fs.files["/workspace/a.txt"] = b"x"
    # Prime the cached session.
    asyncio.run(workspace._control_plane_runner().real_path("/workspace/a.txt"))
    assert fs.connect_calls == 1

    # A dropped session is retried once against a fresh handshake.
    fs.drop_session_once = True
    resolved = asyncio.run(workspace._control_plane_runner().real_path("/workspace/a.txt"))

    assert resolved == "/workspace/a.txt"
    assert fs.connect_calls == 2
    assert fs.sftp_calls == 2
    assert fs.closed_clients == 1
    assert fs.closed_sftps == 1


def test_microsandbox_close_tears_down_cached_sftp_session() -> None:
    workspace, fs = _workspace()
    fs.files["/workspace/a.txt"] = b"x"
    asyncio.run(workspace._control_plane_runner().real_path("/workspace/a.txt"))
    assert fs.connect_calls == 1

    asyncio.run(workspace._control_plane_runner().close())

    assert fs.closed_clients == 1
    assert fs.closed_sftps == 1
    assert workspace._control_plane_runner()._sftp is None


def test_is_path_not_found_error_recognizes_sftp_enoent_message() -> None:
    # Microsandbox SFTP real_path raises a generic error carrying the ENOENT
    # text rather than a typed not-found error; it must still be treated
    # as "file not found" so list surfaces FileNotFoundError.
    assert _is_path_not_found_error(RuntimeError("SFTP error: No such file: /workspace/x"))
    assert _is_path_not_found_error(FileNotFoundError("/x"))
    assert not _is_path_not_found_error(RuntimeError("permission denied"))
