from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
from guard_harness import (
    instrument_directory_identity_alias,
    instrument_directory_open_barrier,
    instrument_write_target_open_barrier,
    instrument_write_target_preopen_barrier,
    instrument_write_truncate_barrier,
    require_bounded_tar_member_reads,
)

import cayu.workspaces._guest_guard as guest_guard_module
import cayu.workspaces.runner as runner_workspace_module
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.core.tools import ToolContext
from cayu.environments import SyncBinding
from cayu.runners import ExecCommand, ExecResult, LocalRunner, Runner
from cayu.tools import ListFilesTool, ReadFileTool, WriteFileTool
from cayu.workspaces import BoundedTarReader, LocalWorkspace, RunnerWorkspace, TarWriter
from cayu.workspaces._tar import tar_archive_size_bound


def _workspace(root) -> RunnerWorkspace:
    return RunnerWorkspace(
        LocalRunner(root, inherit_env=False),
        workspace_id="runner",
        python_executable=sys.executable,
    )


class _ListResultRunner(Runner):
    default_cwd = "/workspace"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.exec_calls = 0

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("list-result-runner", id(self))

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        self.exec_calls += 1
        del command, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
        return ExecResult(stdout=json.dumps(self.payload))


def _list_runner_payload(
    payload: dict[str, Any],
    *,
    pattern: str = "**/*",
    limit: int | None = 10,
    default_list_limit: int = 500,
):
    workspace = RunnerWorkspace(
        _ListResultRunner(payload),
        workspace_id="custom-runner",
        default_list_limit=default_list_limit,
    )
    return asyncio.run(workspace.list(pattern, limit=limit))


def _tar_bytes(entries: tuple[tuple[str, bytes], ...]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in entries:
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def _install_directory_open_barrier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ready: Path,
    release: Path,
) -> None:
    program = runner_workspace_module._RUNNER_WORKSPACE_PROGRAM
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        instrument_directory_open_barrier(program, ready=ready, release=release),
    )
    guest_program = guest_guard_module.GUEST_GUARD_PROGRAM
    monkeypatch.setattr(
        guest_guard_module,
        "GUEST_GUARD_PROGRAM",
        instrument_directory_open_barrier(guest_program, ready=ready, release=release),
    )


async def _run_through_directory_open_barrier(
    operation: Any,
    *,
    ready: Path,
    release: Path,
    mutate: Any,
) -> Any:
    task = asyncio.create_task(operation())
    try:
        for _ in range(1000):
            if ready.exists():
                break
            if task.done():
                return await task
            await asyncio.sleep(0.001)
        else:
            pytest.fail("Runner workspace guest did not reach the directory-open barrier.")
        mutate()
    finally:
        release.write_text("release")
    return await task


def test_runner_workspace_does_not_publish_control_plane_runner(tmp_path) -> None:
    runner = LocalRunner(tmp_path, inherit_env=False)
    workspace = RunnerWorkspace(runner)

    assert workspace.is_bound_to_runner(runner)
    assert not hasattr(workspace, "runner")
    assert not hasattr(workspace, "bound_runner")
    assert not hasattr(workspace, "exec_system")


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


@pytest.mark.parametrize(
    ("payload", "expected_paths", "expected_total", "expected_truncated"),
    (
        (
            {"ok": True, "paths": ["a.txt", "nested/b.txt"], "total_count": 2},
            ("a.txt", "nested/b.txt"),
            2,
            False,
        ),
        (
            {"ok": True, "paths": ["a.txt"], "total_count": 3},
            ("a.txt",),
            3,
            True,
        ),
        ({"ok": True, "paths": [], "total_count": 0}, (), 0, False),
    ),
)
def test_runner_workspace_validates_custom_runner_list_evidence(
    payload,
    expected_paths,
    expected_total,
    expected_truncated,
) -> None:
    result = _list_runner_payload(payload)

    assert result.paths == expected_paths
    assert result.total_count == expected_total
    assert result.truncated is expected_truncated


@pytest.mark.parametrize(
    "path",
    (
        "../outside.txt",
        "nested/../outside.txt",
        "/absolute.txt",
        "",
        "   ",
        ".",
        "./file.txt",
        "nested/./file.txt",
        "nested//file.txt",
        "nested/file.txt/",
    ),
)
def test_runner_workspace_rejects_invalid_or_non_normalized_runner_list_paths(path) -> None:
    with pytest.raises(ValueError, match="path"):
        _list_runner_payload({"ok": True, "paths": [path], "total_count": 1})


@pytest.mark.parametrize("path", ("safe\x00.txt", "safe\ud800.txt"))
def test_runner_workspace_rejects_nonportable_runner_list_paths(path) -> None:
    with pytest.raises(ValueError, match="invalid path") as captured:
        _list_runner_payload({"ok": True, "paths": [path], "total_count": 1})

    assert path not in str(captured.value)
    assert path not in repr(captured.value)


@pytest.mark.parametrize("path", (None, True, 1, 1.5, {}, []))
def test_runner_workspace_rejects_non_string_runner_list_paths(path) -> None:
    with pytest.raises(TypeError, match="non-string path"):
        _list_runner_payload({"ok": True, "paths": [path], "total_count": 1})


@pytest.mark.parametrize(
    ("paths", "pattern", "message"),
    (
        (["a.txt", "a.txt"], "**/*", "duplicate"),
        (["b.txt", "a.txt"], "**/*", "non-deterministic order"),
        (["a.md"], "*.txt", "requested pattern"),
        (["nested/a.txt"], "*.txt", "requested pattern"),
    ),
)
def test_runner_workspace_rejects_contradictory_runner_list_paths(
    paths,
    pattern,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        _list_runner_payload(
            {"ok": True, "paths": paths, "total_count": len(paths)},
            pattern=pattern,
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"ok": True, "total_count": 0},
        {"ok": True, "paths": None, "total_count": 0},
        {"ok": True, "paths": {}, "total_count": 0},
    ),
)
def test_runner_workspace_rejects_malformed_runner_list_paths(payload) -> None:
    with pytest.raises(TypeError, match="invalid paths"):
        _list_runner_payload(payload)


@pytest.mark.parametrize(
    "payload",
    (
        {"ok": True, "paths": []},
        {"ok": True, "paths": [], "total_count": None},
        {"ok": True, "paths": [], "total_count": True},
        {"ok": True, "paths": [], "total_count": "0"},
    ),
)
def test_runner_workspace_rejects_malformed_runner_list_total_count(payload) -> None:
    with pytest.raises(TypeError, match="invalid total_count"):
        _list_runner_payload(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ({"ok": True, "paths": [], "total_count": -1}, "negative total_count"),
        (
            {"ok": True, "paths": ["a.txt", "b.txt"], "total_count": 1},
            "smaller than its paths",
        ),
    ),
)
def test_runner_workspace_rejects_invalid_runner_list_counts(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        _list_runner_payload(payload)


def test_runner_workspace_rejects_oversized_runner_list_total_count() -> None:
    with pytest.raises(ValueError, match="oversized total_count"):
        _list_runner_payload(
            {
                "ok": True,
                "paths": [],
                "total_count": MAX_DURABLE_JSON_INTEGER + 1,
            }
        )


def test_sync_binding_rejects_nonportable_runner_list_before_target_mutation(tmp_path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source = LocalWorkspace(source_root, workspace_id="source")
    runner = _ListResultRunner(
        {
            "ok": True,
            "paths": ["a.txt", "z\x00.txt"],
            "total_count": 2,
        }
    )
    target = RunnerWorkspace(runner, workspace_id="target")

    async def bind() -> None:
        await SyncBinding(target_workspace=target).bind(
            source,
            None,
            session_id="sess_nonportable_runner_list",
        )

    with pytest.raises(ValueError, match="invalid path"):
        asyncio.run(bind())

    assert runner.exec_calls == 1


@pytest.mark.parametrize(
    ("limit", "default_list_limit"),
    ((1, 500), (None, 1)),
)
def test_runner_workspace_rejects_paths_over_effective_list_limit(
    limit,
    default_list_limit,
) -> None:
    with pytest.raises(ValueError, match="effective limit"):
        _list_runner_payload(
            {"ok": True, "paths": ["a.txt", "b.txt"], "total_count": 2},
            limit=limit,
            default_list_limit=default_list_limit,
        )


def test_runner_workspace_list_validation_does_not_echo_invalid_path() -> None:
    canary = "secret-list-path-canary"

    with pytest.raises(ValueError) as captured:
        _list_runner_payload(
            {"ok": True, "paths": [f"../{canary}"], "total_count": 1},
        )

    assert canary not in str(captured.value)
    assert canary not in repr(captured.value)
    current = captured.value.__traceback__
    while current is not None:
        if "/src/cayu/" in current.tb_frame.f_code.co_filename:
            assert all(canary not in repr(value) for value in current.tb_frame.f_locals.values())
        current = current.tb_next


def test_runner_workspace_deletes_files_through_runner(tmp_path) -> None:
    workspace = _workspace(tmp_path)

    asyncio.run(workspace.write_bytes("notes/a.txt", b"abcdef"))
    asyncio.run(workspace.delete("notes/a.txt"))
    asyncio.run(workspace.delete("notes/missing.txt"))

    assert not (tmp_path / "notes" / "a.txt").exists()
    list_result = asyncio.run(workspace.list("**/*.txt", limit=10))
    assert list_result.paths == ()
    assert list_result.total_count == 0


def test_runner_workspace_delete_missing_path_below_file_is_noop(tmp_path) -> None:
    (tmp_path / "parent").write_bytes(b"file")
    workspace = _workspace(tmp_path)

    asyncio.run(workspace.delete("parent/missing.txt"))

    assert (tmp_path / "parent").read_bytes() == b"file"


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


def test_runner_workspace_list_keeps_distinct_directory_alias_paths(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "first").mkdir()
    (tmp_path / "first" / "one.txt").write_bytes(b"one")
    (tmp_path / "second").mkdir()
    (tmp_path / "second" / "two.txt").write_bytes(b"two")
    program = runner_workspace_module._RUNNER_WORKSPACE_PROGRAM
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        instrument_directory_identity_alias(program, paths=("first", "second")),
    )
    workspace = _workspace(tmp_path)

    result = asyncio.run(workspace.list("**/*"))

    assert result.paths == ("first/one.txt", "second/two.txt")
    assert result.total_count == 2


@pytest.mark.parametrize(
    "operation_name",
    (
        "read",
        "overwrite",
        "create",
        "delete",
        "list",
        "read_tar",
        "write_tar",
    ),
)
def test_runner_workspace_uses_opened_parent_without_following_replacement(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    pivot = root / "pivot"
    pivot.mkdir()
    inside_file = pivot / "inside.txt"
    inside_file.write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "inside.txt"
    outside_file.write_bytes(b"outside")
    workspace = _workspace(root)
    ready = tmp_path / "barrier-ready"
    release = tmp_path / "barrier-release"
    # Relocate the descriptor-authorized inode outside the workspace, then put
    # a symlink at its old name. The active traversal may continue through the
    # inode it already opened, but it must never follow the replacement into
    # the distinct external target. Multi-pass tar operations traverse again
    # from root and therefore reject the replacement symlink.
    held = tmp_path / "relocated-pivot"
    _install_directory_open_barrier(monkeypatch, ready=ready, release=release)

    result: Any = None

    async def operation() -> Any:
        if operation_name == "read":
            return await workspace.read_bytes("pivot/inside.txt")
        if operation_name == "overwrite":
            return await workspace.write_bytes("pivot/inside.txt", b"changed")
        if operation_name == "create":
            return await workspace.write_bytes("pivot/new/created.txt", b"created")
        if operation_name == "delete":
            return await workspace.delete("pivot/inside.txt")
        if operation_name == "list":
            return await workspace.list("**/*.txt")
        if operation_name == "read_tar":
            return await workspace.read_tar_bytes(("pivot/inside.txt",))
        if operation_name == "write_tar":
            return await workspace.write_tar_bytes(_tar_bytes((("pivot/archive.txt", b"archive"),)))
        raise AssertionError(f"unknown operation: {operation_name}")

    def swap_parent() -> None:
        pivot.rename(held)
        pivot.symlink_to(outside, target_is_directory=True)

    async def run() -> None:
        nonlocal result
        result = await _run_through_directory_open_barrier(
            operation,
            ready=ready,
            release=release,
            mutate=swap_parent,
        )

    try:
        if operation_name in {"read_tar", "write_tar"}:
            with pytest.raises(ValueError, match="escapes"):
                asyncio.run(run())
        else:
            asyncio.run(run())

        assert outside_file.read_bytes() == b"outside"
        assert not (outside / "new").exists()
        assert not (outside / "archive.txt").exists()
        if operation_name == "read":
            assert result.content == b"inside"
            assert (held / "inside.txt").read_bytes() == b"inside"
        elif operation_name == "overwrite":
            assert (held / "inside.txt").read_bytes() == b"changed"
        elif operation_name == "create":
            assert (held / "new" / "created.txt").read_bytes() == b"created"
        elif operation_name == "delete":
            assert not (held / "inside.txt").exists()
        elif operation_name == "list":
            assert result.paths == ("pivot/inside.txt",)
        elif operation_name in {"read_tar", "write_tar"}:
            assert (held / "inside.txt").read_bytes() == b"inside"
            assert not (held / "archive.txt").exists()
    finally:
        if pivot.is_symlink():
            pivot.unlink()
        if held.exists():
            held.rename(pivot)


@pytest.mark.parametrize(
    "operation_name",
    ("read", "write", "delete", "list", "read_tar", "write_tar"),
)
def test_runner_workspace_leaf_swap_cannot_reach_external_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    pivot = root / "pivot"
    pivot.mkdir()
    target = pivot / "target.txt"
    target.write_bytes(b"inside")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    workspace = _workspace(root)
    ready = tmp_path / "barrier-ready"
    release = tmp_path / "barrier-release"
    _install_directory_open_barrier(monkeypatch, ready=ready, release=release)

    async def operation() -> Any:
        if operation_name == "read":
            return await workspace.read_bytes("pivot/target.txt")
        if operation_name == "write":
            return await workspace.write_bytes("pivot/target.txt", b"changed")
        if operation_name == "delete":
            return await workspace.delete("pivot/target.txt")
        if operation_name == "list":
            return await workspace.list("**/*.txt")
        if operation_name == "read_tar":
            return await workspace.read_tar_bytes(("pivot/target.txt",))
        if operation_name == "write_tar":
            return await workspace.write_tar_bytes(_tar_bytes((("pivot/target.txt", b"archive"),)))
        raise AssertionError(f"unknown operation: {operation_name}")

    def swap_leaf() -> None:
        target.unlink()
        target.symlink_to(outside)

    async def run() -> Any:
        return await _run_through_directory_open_barrier(
            operation,
            ready=ready,
            release=release,
            mutate=swap_leaf,
        )

    if operation_name == "list":
        result = asyncio.run(run())
        assert result.paths == ()
    else:
        with pytest.raises(ValueError, match="escapes"):
            asyncio.run(run())
    assert outside.read_bytes() == b"outside"


def test_runner_workspace_rejects_hard_link_overwrite(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    os.link(outside, root / "linked.txt")
    workspace = _workspace(root)

    with pytest.raises(ValueError, match="multiple hard links"):
        asyncio.run(workspace.write_bytes("linked.txt", b"changed"))
    with pytest.raises(ValueError, match="multiple hard links"):
        asyncio.run(workspace.write_tar_bytes(_tar_bytes((("linked.txt", b"archive"),))))

    assert outside.read_bytes() == b"outside"
    assert (root / "linked.txt").read_bytes() == b"outside"


def test_runner_workspace_open_descriptor_survives_late_external_hard_link(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.txt"
    target.write_bytes(b"original")
    outside = tmp_path / "outside.txt"
    ready = tmp_path / "write-open-ready"
    release = tmp_path / "write-open-release"
    program = runner_workspace_module._RUNNER_WORKSPACE_PROGRAM
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        instrument_write_target_open_barrier(program, ready=ready, release=release),
    )
    workspace = _workspace(root)

    async def operation() -> None:
        await workspace.write_bytes("target.txt", b"replacement")

    def add_external_hard_link() -> None:
        os.link(target, outside)

    asyncio.run(
        _run_through_directory_open_barrier(
            operation,
            ready=ready,
            release=release,
            mutate=add_external_hard_link,
        )
    )

    assert target.read_bytes() == b"replacement"
    assert outside.read_bytes() == b"replacement"
    assert target.stat().st_ino == outside.stat().st_ino


def test_runner_workspace_overwrite_preserves_permissions(tmp_path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"original")
    target.chmod(0o751)
    workspace = _workspace(tmp_path)

    asyncio.run(workspace.write_bytes("target.txt", b"replacement"))

    assert target.read_bytes() == b"replacement"
    assert target.stat().st_mode & 0o777 == 0o751


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX umask")
@pytest.mark.parametrize(
    ("umask", "expected_mode"),
    (("077", 0o600), ("002", 0o664)),
)
def test_runner_workspace_write_applies_umask_to_new_files(
    tmp_path,
    umask: str,
    expected_mode: int,
) -> None:
    wrapper = tmp_path / "private-umask-python"
    wrapper.write_text(
        f'#!/bin/sh\numask {umask}\nexec "{sys.executable}" "$@"\n',
    )
    wrapper.chmod(0o755)
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        workspace_id="runner",
        python_executable=str(wrapper),
    )

    asyncio.run(workspace.write_bytes("new.txt", b"content"))

    assert (tmp_path / "new.txt").stat().st_mode & 0o777 == expected_mode


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX umask")
def test_runner_workspace_tar_import_applies_umask_to_new_files(tmp_path) -> None:
    wrapper = tmp_path / "group-umask-python"
    wrapper.write_text(
        f'#!/bin/sh\numask 002\nexec "{sys.executable}" "$@"\n',
    )
    wrapper.chmod(0o755)
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        workspace_id="runner",
        python_executable=str(wrapper),
    )

    asyncio.run(workspace.write_tar_bytes(_tar_bytes((("imported.txt", b"content"),))))

    assert (tmp_path / "imported.txt").stat().st_mode & 0o777 == 0o664


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX umask")
@pytest.mark.parametrize("operation", ("write", "write_tar"))
def test_runner_workspace_deleted_target_is_recreated_with_current_umask(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.txt"
    target.write_bytes(b"original")
    target.chmod(0o751)
    wrapper = tmp_path / "group-umask-python"
    wrapper.write_text(
        f'#!/bin/sh\numask 002\nexec "{sys.executable}" "$@"\n',
    )
    wrapper.chmod(0o755)
    ready = tmp_path / "write-preopen-ready"
    release = tmp_path / "write-preopen-release"
    program = runner_workspace_module._RUNNER_WORKSPACE_PROGRAM
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        instrument_write_target_preopen_barrier(program, ready=ready, release=release),
    )
    workspace = RunnerWorkspace(
        LocalRunner(root, inherit_env=False),
        workspace_id="runner",
        python_executable=str(wrapper),
    )

    async def write() -> None:
        if operation == "write":
            await workspace.write_bytes("target.txt", b"replacement")
        else:
            await workspace.write_tar_bytes(_tar_bytes((("target.txt", b"replacement"),)))

    asyncio.run(
        _run_through_directory_open_barrier(
            write,
            ready=ready,
            release=release,
            mutate=target.unlink,
        )
    )

    assert target.read_bytes() == b"replacement"
    assert target.stat().st_mode & 0o777 == 0o664


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX umask")
def test_runner_workspace_nested_write_applies_umask_to_new_directories(tmp_path) -> None:
    wrapper = tmp_path / "group-umask-python"
    wrapper.write_text(
        f'#!/bin/sh\numask 002\nexec "{sys.executable}" "$@"\n',
    )
    wrapper.chmod(0o755)
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        workspace_id="runner",
        python_executable=str(wrapper),
    )

    asyncio.run(workspace.write_bytes("shared/nested/new.txt", b"content"))

    assert (tmp_path / "shared").stat().st_mode & 0o777 == 0o775
    assert (tmp_path / "shared" / "nested").stat().st_mode & 0o777 == 0o775


def test_runner_workspace_write_failure_can_leave_truncated_target(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"original")
    program = runner_workspace_module._RUNNER_WORKSPACE_PROGRAM
    marker = "        # CAYU_TEST_AFTER_WRITE_TRUNCATE"
    assert program.count(marker) == 1
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        program.replace(marker, '        raise OSError("injected write failure")'),
    )
    workspace = _workspace(tmp_path)

    with pytest.raises(RuntimeError, match="injected write failure"):
        asyncio.run(workspace.write_bytes("target.txt", b"replacement"))

    assert target.read_bytes() == b""


def test_runner_workspace_cancellation_after_truncate_preserves_cancellation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.txt"
    target.write_bytes(b"original")
    ready = tmp_path / "write-truncated-ready"
    release = tmp_path / "write-truncated-release"
    program = runner_workspace_module._RUNNER_WORKSPACE_PROGRAM
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        instrument_write_truncate_barrier(program, ready=ready, release=release),
    )
    workspace = _workspace(root)

    async def run() -> None:
        operation = asyncio.create_task(workspace.write_bytes("target.txt", b"replacement"))
        try:
            for _ in range(1000):
                if ready.exists():
                    break
                if operation.done():
                    await operation
                    pytest.fail("Guarded write completed before its truncate barrier.")
                await asyncio.sleep(0.001)
            else:
                pytest.fail("Guarded write did not reach its truncate barrier.")

            operation.cancel()
            assert operation.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await operation
            assert operation.cancelled()
        finally:
            release.write_text("release")

    asyncio.run(run())

    assert target.read_bytes() == b""


def test_runner_workspace_list_during_write_has_no_internal_entries(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target.txt"
    target.write_bytes(b"original")
    ready = tmp_path / "write-open-ready"
    release = tmp_path / "write-open-release"
    program = runner_workspace_module._RUNNER_WORKSPACE_PROGRAM
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        instrument_write_target_open_barrier(program, ready=ready, release=release),
    )
    workspace = _workspace(root)

    async def run() -> None:
        operation = asyncio.create_task(workspace.write_bytes("target.txt", b"replacement"))
        try:
            for _ in range(1000):
                if ready.exists():
                    break
                if operation.done():
                    await operation
                    pytest.fail("Guarded write completed before its open barrier.")
                await asyncio.sleep(0.001)
            else:
                pytest.fail("Guarded write did not reach its open barrier.")
            listed = await workspace.list("**/*")
            assert listed.paths == ("target.txt",)
        finally:
            release.write_text("release")
        await operation

    asyncio.run(run())

    assert target.read_bytes() == b"replacement"


@pytest.mark.parametrize("operation", ("write", "write_tar"))
def test_runner_workspace_treats_former_staging_names_as_ordinary_files(
    tmp_path,
    operation: str,
) -> None:
    foreign = tmp_path / (".cayu-write-99999999-" + ("a" * 32))
    foreign.write_bytes(b"in progress in another guest")
    workspace = _workspace(tmp_path)

    if operation == "write":
        asyncio.run(workspace.write_bytes("target.txt", b"content"))
    else:
        asyncio.run(workspace.write_tar_bytes(_tar_bytes((("target.txt", b"content"),))))
    listed = asyncio.run(workspace.list("**/*"))

    assert foreign.read_bytes() == b"in progress in another guest"
    assert (tmp_path / "target.txt").read_bytes() == b"content"
    assert listed.paths == (foreign.name, "target.txt")


def test_runner_workspace_operations_traverse_search_only_directory(tmp_path) -> None:
    directory = tmp_path / "search-only"
    directory.mkdir()
    target = directory / "target.txt"
    target.write_bytes(b"original")
    directory.chmod(0o311)
    workspace = _workspace(tmp_path)

    try:
        assert asyncio.run(workspace.read_bytes("search-only/target.txt")).content == b"original"
        asyncio.run(workspace.write_bytes("search-only/target.txt", b"replacement"))
        assert target.read_bytes() == b"replacement"
        asyncio.run(workspace.delete("search-only/target.txt"))
        assert not target.exists()
    finally:
        directory.chmod(0o700)


def test_runner_workspace_delete_does_not_require_file_read_permission(tmp_path) -> None:
    target = tmp_path / "target.txt"
    target.write_bytes(b"content")
    target.chmod(0)
    workspace = _workspace(tmp_path)

    try:
        asyncio.run(workspace.delete("target.txt"))
    finally:
        if target.exists():
            target.chmod(0o600)

    assert not target.exists()


def test_runner_workspace_delete_hard_link_only_unlinks_workspace_name(tmp_path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    os.link(outside, root / "linked.txt")
    workspace = _workspace(root)

    asyncio.run(workspace.delete("linked.txt"))

    assert outside.read_bytes() == b"outside"
    assert not (root / "linked.txt").exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX file permissions")
def test_runner_workspace_lists_unreadable_regular_files_without_opening_them(
    tmp_path,
) -> None:
    unreadable = tmp_path / "secret.bin"
    unreadable.write_bytes(b"secret")
    unreadable.chmod(0)
    (tmp_path / "visible.txt").write_bytes(b"visible")
    workspace = _workspace(tmp_path)

    try:
        matching = asyncio.run(workspace.list("secret.bin"))
        non_matching = asyncio.run(workspace.list("*.txt"))
    finally:
        unreadable.chmod(0o600)

    assert matching.paths == ("secret.bin",)
    assert matching.total_count == 1
    assert matching.truncated is False
    assert non_matching.paths == ("visible.txt",)
    assert non_matching.total_count == 1
    assert non_matching.truncated is False


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX special files")
def test_runner_workspace_rejects_special_file_targets_without_blocking(tmp_path) -> None:
    os.mkfifo(tmp_path / "pipe")
    workspace = _workspace(tmp_path)

    with pytest.raises(FileNotFoundError):
        asyncio.run(workspace.read_bytes("pipe"))
    with pytest.raises(IsADirectoryError):
        asyncio.run(workspace.write_bytes("pipe", b"content"))
    with pytest.raises(IsADirectoryError):
        asyncio.run(workspace.delete("pipe"))
    with pytest.raises(FileNotFoundError):
        asyncio.run(workspace.read_tar_bytes(("pipe",)))
    with pytest.raises(IsADirectoryError):
        asyncio.run(workspace.write_tar_bytes(_tar_bytes((("pipe", b"content"),))))

    assert asyncio.run(workspace.list("**/*")).paths == ()


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

    with pytest.raises(RuntimeError, match="Failed to read Runner workspace file"):
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


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX descriptor limits")
def test_runner_workspace_read_tar_does_not_retain_one_descriptor_per_file(tmp_path) -> None:
    wrapper = tmp_path / "limited-python"
    wrapper.write_text(
        f'#!/bin/sh\nulimit -n 32\nexec "{sys.executable}" "$@"\n',
    )
    wrapper.chmod(0o755)
    paths = tuple(f"file-{index:03d}.txt" for index in range(80))
    for path in paths:
        (tmp_path / path).write_bytes(path.encode())
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        workspace_id="runner",
        python_executable=str(wrapper),
    )

    data = asyncio.run(workspace.read_tar_bytes(paths))

    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as archive:
        assert tuple(member.name for member in archive.getmembers()) == paths


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


def test_runner_workspace_write_tar_validates_all_members_before_mutation(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    data = _tar_bytes(
        (
            ("safe.txt", b"would-have-been-written"),
            ("../evil.txt", b"evil"),
        )
    )

    with pytest.raises(ValueError, match="inside the workspace"):
        asyncio.run(workspace.write_tar_bytes(data))

    assert not (tmp_path / "safe.txt").exists()
    assert not (tmp_path.parent / "evil.txt").exists()


def test_runner_workspace_write_tar_partial_failure_is_retryable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second = tmp_path / "second.txt"
    second.write_bytes(b"original-second")
    archive = _tar_bytes(
        (
            ("first.txt", b"replacement-first"),
            ("second.txt", b"replacement-second"),
        )
    )
    program = runner_workspace_module._RUNNER_WORKSPACE_PROGRAM
    marker = "        # CAYU_TEST_AFTER_WRITE_TRUNCATE"
    assert program.count(marker) == 1
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        program.replace(
            marker,
            '        if name == "second.txt":\n'
            '            raise OSError("injected second-member write failure")',
        ),
    )
    workspace = _workspace(tmp_path)

    with pytest.raises(RuntimeError, match="injected second-member write failure"):
        asyncio.run(workspace.write_tar_bytes(archive))

    assert (tmp_path / "first.txt").read_bytes() == b"replacement-first"
    assert second.read_bytes() == b""

    monkeypatch.setattr(runner_workspace_module, "_RUNNER_WORKSPACE_PROGRAM", program)
    asyncio.run(workspace.write_tar_bytes(archive))

    assert (tmp_path / "first.txt").read_bytes() == b"replacement-first"
    assert second.read_bytes() == b"replacement-second"


@pytest.mark.parametrize("hazard", ("symlink_parent", "hard_link", "fifo"))
def test_runner_workspace_write_tar_preflights_destination_hazards_before_mutation(
    tmp_path,
    hazard: str,
) -> None:
    if hazard == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("requires POSIX special files")
    safe = tmp_path / "safe.txt"
    safe.write_bytes(b"original")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "target.txt"
    outside_file.write_bytes(b"outside")
    if hazard == "symlink_parent":
        (tmp_path / "hazard").symlink_to(outside, target_is_directory=True)
        hazardous_path = "hazard/target.txt"
    elif hazard == "hard_link":
        os.link(outside_file, tmp_path / "hazard.txt")
        hazardous_path = "hazard.txt"
    else:
        os.mkfifo(tmp_path / "hazard.txt")
        hazardous_path = "hazard.txt"
    workspace = _workspace(tmp_path)
    data = _tar_bytes(
        (
            ("safe.txt", b"replacement"),
            (hazardous_path, b"must-not-be-written"),
        )
    )

    with pytest.raises((ValueError, IsADirectoryError)):
        asyncio.run(workspace.write_tar_bytes(data))

    assert safe.read_bytes() == b"original"
    assert outside_file.read_bytes() == b"outside"


def test_runner_workspace_write_tar_rejects_conflicting_members_before_mutation(
    tmp_path,
) -> None:
    workspace = _workspace(tmp_path)
    data = _tar_bytes(
        (
            ("node", b"file"),
            ("node/child.txt", b"child"),
        )
    )

    with pytest.raises(ValueError, match="conflicting paths"):
        asyncio.run(workspace.write_tar_bytes(data))

    assert not (tmp_path / "node").exists()


def test_runner_workspace_write_tar_reads_member_content_in_bounded_chunks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = runner_workspace_module._RUNNER_WORKSPACE_PROGRAM
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        require_bounded_tar_member_reads(program),
    )
    workspace = _workspace(tmp_path)
    content = b"content" * 20_000

    asyncio.run(workspace.write_tar_bytes(_tar_bytes((("large.bin", content),))))

    assert (tmp_path / "large.bin").read_bytes() == content


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


def test_runner_workspace_fails_closed_without_descriptor_primitives(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = runner_workspace_module._RUNNER_WORKSPACE_PROGRAM
    marker = "def require_descriptor_guard_support():\n"
    assert program.count(marker) == 1
    monkeypatch.setattr(
        runner_workspace_module,
        "_RUNNER_WORKSPACE_PROGRAM",
        program.replace(
            marker,
            marker + '    raise GuardPathError("unsupported")\n',
        ),
    )
    workspace = _workspace(tmp_path)

    with pytest.raises(RuntimeError, match="POSIX descriptor-relative"):
        asyncio.run(workspace.write_bytes("file.txt", b"content"))

    assert not (tmp_path / "file.txt").exists()


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
            {
                "path": "notes/result.txt",
                "content": "runner workspace",
                "mode": "create",
            },
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
    assert read_result.content.endswith("[/read_file metadata]\nrunner workspace")
    assert '"revision":"sha256:' in read_result.content
    assert list_result.structured == {
        "pattern": "**/*.txt",
        "files": ["notes/result.txt"],
        "total_files": 1,
        "truncated": False,
    }
