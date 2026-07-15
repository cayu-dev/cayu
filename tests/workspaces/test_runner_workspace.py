from __future__ import annotations

import asyncio
import io
import os
import sys
import tarfile

import pytest

import cayu.workspaces.runner as runner_workspace_module
from cayu.core.tools import ToolContext
from cayu.runners import LocalRunner
from cayu.tools import ListFilesTool, ReadFileTool, WriteFileTool
from cayu.workspaces import BoundedTarReader, RunnerWorkspace, TarWriter
from cayu.workspaces._tar import tar_archive_size_bound


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


def test_runner_workspace_deletes_files_through_runner(tmp_path) -> None:
    workspace = _workspace(tmp_path)

    asyncio.run(workspace.write_bytes("notes/a.txt", b"abcdef"))
    asyncio.run(workspace.delete("notes/a.txt"))
    asyncio.run(workspace.delete("notes/missing.txt"))

    assert not (tmp_path / "notes" / "a.txt").exists()
    list_result = asyncio.run(workspace.list("**/*.txt", limit=10))
    assert list_result.paths == ()
    assert list_result.total_count == 0


def test_runner_workspace_rejects_delete_symlink_leaf_inside_runner_root(tmp_path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"keep")
    (tmp_path / "link.txt").symlink_to(target)
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.delete("link.txt"))

    assert target.read_bytes() == b"keep"
    assert (tmp_path / "link.txt").is_symlink()


def test_runner_workspace_rejects_write_symlink_leaf_inside_runner_root(tmp_path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"keep")
    (tmp_path / "link.txt").symlink_to(target)
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("link.txt", b"overwrite"))

    assert target.read_bytes() == b"keep"
    assert (tmp_path / "link.txt").is_symlink()


def test_runner_workspace_rejects_delete_through_symlink_parent_inside_runner_root(
    tmp_path,
) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "a.txt"
    target.write_bytes(b"keep")
    (tmp_path / "link").symlink_to(target_dir)
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.delete("link/a.txt"))

    assert target.read_bytes() == b"keep"
    assert (tmp_path / "link").is_symlink()


def test_runner_workspace_rejects_write_through_symlink_parent_inside_runner_root(
    tmp_path,
) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "a.txt"
    target.write_bytes(b"keep")
    (tmp_path / "link").symlink_to(target_dir)
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.write_bytes("link/a.txt", b"overwrite"))

    assert target.read_bytes() == b"keep"
    assert (tmp_path / "link").is_symlink()


def test_runner_workspace_list_skips_symlink_paths_inside_runner_root(tmp_path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"keep")
    (tmp_path / "link.txt").symlink_to(target)
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    nested = target_dir / "a.txt"
    nested.write_bytes(b"nested")
    (tmp_path / "link_dir").symlink_to(target_dir)
    workspace = _workspace(tmp_path)

    result = asyncio.run(workspace.list("**/*.txt"))

    assert result.paths == ("target.txt", "target/a.txt")
    assert result.total_count == 2
    assert result.truncated is False


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

    assert isinstance(workspace, BoundedTarReader)
    assert isinstance(workspace, TarWriter)
    assert workspace.bounded_read_limit(10) == 4
    assert workspace.bounded_read_limit(2) == 2
    assert read_result.content == b"abcd"
    assert read_result.total_bytes == 6
    assert read_result.truncated is True
    assert len(list_result.paths) == 1
    assert list_result.total_count == 2
    assert list_result.truncated is True


def test_runner_workspace_list_limit_returns_sorted_prefix(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    asyncio.run(workspace.write_bytes("c.txt", b""))
    asyncio.run(workspace.write_bytes("a.txt", b""))
    asyncio.run(workspace.write_bytes("b.txt", b""))

    result = asyncio.run(workspace.list("*.txt", limit=2))

    assert result.paths == ("a.txt", "b.txt")
    assert result.total_count == 3
    assert result.truncated is True


@pytest.mark.skipif(os.name == "nt", reason="requires long POSIX filesystem paths")
def test_runner_workspace_lists_long_paths_without_transport_truncation(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    long_directory = tmp_path.joinpath(*("d" * 200 for _ in range(3)))
    long_directory.mkdir(parents=True)
    expected: list[str] = []
    for index in range(500):
        path = long_directory / f"file-{index:03d}.txt"
        path.write_bytes(b"")
        expected.append(path.relative_to(tmp_path).as_posix())

    assert sum(map(len, expected)) / len(expected) > 512
    result = asyncio.run(workspace.list("**/*.txt", limit=500))

    assert result.paths == tuple(sorted(expected))
    assert result.total_count == 500
    assert result.truncated is False


def test_runner_workspace_list_returns_sorted_prefix_at_payload_limit(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner_workspace_module,
        "RUNNER_WORKSPACE_LIST_PAYLOAD_LIMIT_BYTES",
        1024,
    )
    workspace = _workspace(tmp_path)
    directory = tmp_path / ("d" * 100)
    directory.mkdir()
    expected: list[str] = []
    for index in range(20):
        path = directory / f"{index:02d}-{'f' * 80}.txt"
        path.write_bytes(b"")
        expected.append(path.relative_to(tmp_path).as_posix())

    result = asyncio.run(workspace.list("**/*.txt", limit=20))

    assert 0 < len(result.paths) < len(expected)
    assert result.paths == tuple(sorted(expected)[: len(result.paths)])
    assert result.total_count == len(expected)
    assert result.truncated is True


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


def test_runner_workspace_bulk_tar_round_trip(tmp_path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source = _workspace(source_root)
    target = _workspace(target_root)
    asyncio.run(source.write_bytes("a.txt", b"alpha"))
    asyncio.run(source.write_bytes("nested/b.txt", b"beta"))

    data = asyncio.run(source.read_tar_bytes(("a.txt", "nested/b.txt")))

    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as archive:
        names = sorted(member.name for member in archive.getmembers())
    assert names == ["a.txt", "nested/b.txt"]

    asyncio.run(target.write_tar_bytes(data))

    assert (target_root / "a.txt").read_bytes() == b"alpha"
    assert (target_root / "nested" / "b.txt").read_bytes() == b"beta"


def test_runner_workspace_read_tar_rejects_oversized_file(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    asyncio.run(workspace.write_bytes("big.txt", b"abcdef"))

    with pytest.raises(RuntimeError, match="exceeds max_file_bytes=3"):
        asyncio.run(workspace.read_tar_bytes(("big.txt",), max_file_bytes=3))


def test_runner_workspace_read_tar_enforces_total_bytes_before_archiving(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    asyncio.run(workspace.write_bytes("a.txt", b"abc"))
    asyncio.run(workspace.write_bytes("b.txt", b"def"))

    data = asyncio.run(
        workspace.read_tar_bytes(
            ("a.txt", "b.txt"),
            max_total_bytes=6,
        )
    )

    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as archive:
        assert sum(member.size for member in archive.getmembers()) == 6
    with pytest.raises(RuntimeError, match="files exceed max_total_bytes=5"):
        asyncio.run(
            workspace.read_tar_bytes(
                ("a.txt", "b.txt"),
                max_total_bytes=5,
            )
        )


def test_runner_workspace_read_tar_validates_total_bytes_limit(tmp_path) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="max_total_bytes"):
        asyncio.run(workspace.read_tar_bytes(("a.txt",), max_total_bytes=0))
    with pytest.raises(TypeError, match="max_total_bytes"):
        asyncio.run(workspace.read_tar_bytes(("a.txt",), max_total_bytes=True))

    with pytest.raises(ValueError, match="max_archive_bytes"):
        asyncio.run(workspace.read_tar_bytes(("a.txt",), max_archive_bytes=0))
    with pytest.raises(TypeError, match="max_archive_bytes"):
        asyncio.run(workspace.read_tar_bytes(("a.txt",), max_archive_bytes=True))


def test_runner_workspace_read_tar_preflights_raw_archive_size(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    asyncio.run(workspace.write_bytes("a.txt", b"abc"))
    asyncio.run(workspace.write_bytes("b.txt", b"def"))
    paths = ("a.txt", "b.txt")
    archive_bound = tar_archive_size_bound(6, paths)

    data = asyncio.run(workspace.read_tar_bytes(paths, max_archive_bytes=archive_bound))

    assert len(data) <= archive_bound
    with pytest.raises(RuntimeError, match="tar exceeds max_archive_bytes"):
        asyncio.run(
            workspace.read_tar_bytes(
                paths,
                max_archive_bytes=archive_bound - 1,
            )
        )


def test_tar_archive_size_bound_accounts_for_long_pax_paths() -> None:
    paths = tuple(f"{index}/" + ("a" * 3998) for index in range(10))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in paths:
            info = tarfile.TarInfo(name=path)
            info.size = 0
            archive.addfile(info, io.BytesIO())

    bound = tar_archive_size_bound(1, paths)

    assert len(buffer.getvalue()) <= bound


def test_runner_workspace_read_tar_validates_paths(tmp_path) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="at least one path"):
        asyncio.run(workspace.read_tar_bytes(()))

    with pytest.raises(TypeError, match="sequence of strings"):
        asyncio.run(workspace.read_tar_bytes("a.txt"))

    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(workspace.read_tar_bytes(("../outside.txt",)))


def test_runner_workspace_write_tar_rejects_escaping_member(tmp_path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo(name="../evil.txt")
        info.size = 4
        archive.addfile(info, io.BytesIO(b"evil"))
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="inside the workspace"):
        asyncio.run(workspace.write_tar_bytes(buffer.getvalue()))

    assert not (tmp_path.parent / "evil.txt").exists()


def test_runner_workspace_write_tar_rejects_symlink_member(tmp_path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo(name="link.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="regular file"):
        asyncio.run(workspace.write_tar_bytes(buffer.getvalue()))

    assert not (tmp_path / "link.txt").exists()


def test_runner_workspace_write_tar_rejects_non_bytes(tmp_path) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(TypeError, match="bytes"):
        asyncio.run(workspace.write_tar_bytes("not-bytes"))


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
