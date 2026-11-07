"""read_file with several files reads them in one runner command."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

from cayu.runners import ExecCommand, ExecResult, LocalRunner
from cayu.runtime._invocation_secrets import InvocationSecretTracker
from cayu.sandbox_broker.client import MAX_COMMAND_OUTPUT_BYTES
from cayu.tools import ReadFileTool
from cayu.tools._resources import invocation_workspace_handle
from cayu.tools.base import ToolContext
from cayu.vaults import SecretRedactor
from cayu.workspaces import (
    RunnerWorkspace,
    WorkspaceFileReadRequest,
    WorkspaceLineRead,
    WorkspaceReadOffsetError,
    WorkspaceReadResult,
)


class _CountingRunner(LocalRunner):
    """A local runner that counts commands and, like a broker, may cap their output."""

    def __init__(self, root: Any, *, cap: int | None = None) -> None:
        super().__init__(root, inherit_env=False)
        self.max_exec_output_bytes = cap
        self.commands = 0

    async def exec(
        self,
        command: ExecCommand,
        *,
        output_limit_bytes: int | None = None,
        **kwargs: Any,
    ) -> ExecResult:
        self.commands += 1
        cap = self.max_exec_output_bytes
        if cap is not None:
            output_limit_bytes = cap if output_limit_bytes is None else min(output_limit_bytes, cap)
        result = await super().exec(command, output_limit_bytes=output_limit_bytes, **kwargs)
        assert not result.stdout_truncated, "a guarded read asked for more than the cap"
        return result


class _OneAtATime(RunnerWorkspace):
    """The same workspace without batch reads, so each file is read on its own."""

    def file_batch_reader(self):
        return None


def _workspaces(root: Any, *, cap: int | None = None) -> tuple[Any, Any]:
    batched = RunnerWorkspace(
        _CountingRunner(root, cap=cap),
        workspace_id="batched",
        python_executable=sys.executable,
        excluded_directory_names=("secret",),
    )
    single = _OneAtATime(
        _CountingRunner(root, cap=cap),
        workspace_id="single",
        python_executable=sys.executable,
        excluded_directory_names=("secret",),
    )
    return batched, single


def _read(workspace: Any, args: dict[str, Any], **context: Any) -> Any:
    ctx = ToolContext(session_id="sess", workspace=workspace, **context)
    return asyncio.run(ReadFileTool().run(ctx, args))


def _same_read(root: Any, args: dict[str, Any], *, cap: int | None = None) -> tuple[Any, int, int]:
    """Read in one batch and one file at a time; return the result and both command counts."""

    batched, single = _workspaces(root, cap=cap)
    batch_result = _read(batched, args)
    single_result = _read(single, args)
    assert batch_result.structured == single_result.structured
    assert batch_result.content == single_result.content
    assert batch_result.is_error == single_result.is_error
    return batch_result, batched._runner.commands, single._runner.commands


def _tree(root: Any) -> None:
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text("".join(f"a = {n}\n" for n in range(1, 41)))
    (root / "pkg" / "b.py").write_text("print('b')\n")
    (root / "notes.md").write_text("# notes\n" + "word " * 200 + "\n")
    (root / "image.bin").write_bytes(b"\x00\x01\x02binary\x00" * 10)
    (root / "secret").mkdir()
    (root / "secret" / "key.txt").write_text("hidden\n")


def test_several_files_read_in_one_command(tmp_path) -> None:
    _tree(tmp_path)
    files = [{"path": "pkg/a.py"}, {"path": "pkg/b.py"}, {"path": "notes.md"}]

    result, batched, single = _same_read(tmp_path, {"files": files})

    assert not result.is_error
    assert [item["path"] for item in result.structured["files"]] == [
        "pkg/a.py",
        "pkg/b.py",
        "notes.md",
    ]
    assert all(item["revision"] for item in result.structured["files"])
    assert (batched, single) == (1, 3)


def test_line_ranges_and_byte_pages_read_in_one_command(tmp_path) -> None:
    _tree(tmp_path)
    files = [
        {"path": "pkg/a.py", "start_line": 10, "end_line": 12},
        {"path": "notes.md", "offset": 8},
        {"path": "pkg/b.py", "start_line": 1},
    ]

    result, batched, single = _same_read(tmp_path, {"files": files})

    first, second, third = result.structured["files"]
    assert (first["start_line"], first["end_line"], first["total_lines"]) == (10, 12, 40)
    assert "a = 10\na = 11\na = 12\n" in result.content
    assert second["offset"] == 8
    assert third["revision"] is not None
    assert (batched, single) == (1, 3)


def test_each_file_keeps_its_own_error(tmp_path) -> None:
    _tree(tmp_path)
    files = [
        {"path": "pkg/a.py"},
        {"path": "missing.py"},
        {"path": "image.bin"},
        {"path": "pkg"},
        {"path": "pkg/b.py", "start_line": 5},
        {"path": "notes.md", "offset": 10_000},
    ]

    result, batched, _ = _same_read(tmp_path, {"files": files})

    errors = [item.get("error") for item in result.structured["files"]]
    assert errors[0] is None
    assert errors[1:] == [
        "not_found",
        "binary_file",
        "not_found",
        "invalid_arguments",
        "invalid_arguments",
    ]
    assert result.structured["failed"] == 5
    assert batched == 1


def test_a_path_the_workspace_refuses_fails_like_its_own_read(tmp_path) -> None:
    _tree(tmp_path)

    batched, single = _workspaces(tmp_path)
    args = {"files": [{"path": "pkg/b.py"}, {"path": "secret/key.txt"}]}
    with pytest.raises(ValueError, match="excluded directory") as batch_error:
        _read(batched, args)
    with pytest.raises(ValueError, match="excluded directory") as single_error:
        _read(single, args)
    assert str(batch_error.value) == str(single_error.value)


def test_files_share_the_call_max_bytes(tmp_path) -> None:
    _tree(tmp_path)
    (tmp_path / "big.txt").write_text("x" * 5000)
    files = [{"path": "pkg/b.py"}, {"path": "big.txt"}, {"path": "notes.md"}]

    result, batched, _ = _same_read(tmp_path, {"files": files, "max_bytes": 2000})

    small, big, notes = result.structured["files"]
    assert small["truncated"] is False
    assert big["truncated"] is True and big["next_offset"] == big["bytes"]
    assert result.structured["bytes"] <= 2000
    assert small["bytes"] + big["bytes"] + notes["bytes"] == result.structured["bytes"]
    assert batched == 1


def test_files_share_a_capped_runner_output(tmp_path) -> None:
    # Eight files whose asks outgrow one command's output through a broker's cap.
    # Each gets a share of it, which still covers its part of max_bytes.
    for n in range(8):
        (tmp_path / f"f{n}.txt").write_text(f"{n}\n" + "y" * 300_000 + "\n")
    files = [{"path": f"f{n}.txt"} for n in range(8)]

    result, batched, single = _same_read(tmp_path, {"files": files}, cap=MAX_COMMAND_OUTPUT_BYTES)

    assert all(item["truncated"] for item in result.structured["files"])
    assert result.structured["bytes"] <= 256 * 1024
    assert (batched, single) == (1, 8)


def test_a_file_its_share_doesnt_cover_is_read_again(tmp_path) -> None:
    # max_bytes larger than one command's output: the shares can't hold every
    # page, so the files they miss are read on their own.
    for name in ("one", "two", "three"):
        (tmp_path / f"{name}.txt").write_text(f"{name}\n" + "y" * 900_000 + "\n")
    (tmp_path / "small.txt").write_text("small\n")
    files = [{"path": f"{name}.txt"} for name in ("small", "one", "two", "three")]

    result, batched, single = _same_read(
        tmp_path,
        {"files": files, "max_bytes": 2_000_000},
        cap=MAX_COMMAND_OUTPUT_BYTES,
    )

    assert result.structured["files"][0]["truncated"] is False
    assert result.structured["bytes"] <= 2_000_000
    assert batched <= single


def test_a_small_files_line_range_keeps_its_window_when_output_is_shared(tmp_path) -> None:
    # The shares cover a small file's head and window, so it isn't read again.
    for n in range(7):
        (tmp_path / f"f{n}.txt").write_text(f"{n}\n" + "y" * 300_000 + "\n")
    (tmp_path / "small.py").write_text("".join(f"x{n} = {n}\n" for n in range(1, 50)))
    files = [{"path": f"f{n}.txt"} for n in range(7)]
    files.append({"path": "small.py", "start_line": 5, "end_line": 7})

    result, batched, single = _same_read(tmp_path, {"files": files}, cap=MAX_COMMAND_OUTPUT_BYTES)

    assert result.structured["files"][-1]["start_line"] == 5
    assert "x5 = 5\nx6 = 6\nx7 = 7\n" in result.content
    assert (batched, single) == (1, 8)


def test_a_share_too_short_to_project_reads_the_file_on_its_own(tmp_path) -> None:
    # Through a small output cap, each file's share is shorter than the handle's
    # redaction lookbehind; each file is then read on its own, as it would be.
    for n in range(20):
        (tmp_path / f"f{n}.txt").write_text("z" * 200_000)
    files = [{"path": f"f{n}.txt", "offset": 4096} for n in range(20)]
    batched, single = _workspaces(tmp_path, cap=32 * 1024)
    tracker = InvocationSecretTracker(SecretRedactor("s" * 1024))
    context = {"invocation_secret_snapshot_provider": tracker.snapshot}

    batch_result = _read(_handle(batched, tracker), {"files": files}, **context)
    single_result = _read(_handle(single, tracker), {"files": files}, **context)

    assert not batch_result.is_error, batch_result.content
    assert batch_result.structured == single_result.structured
    assert batch_result.content == single_result.content


def test_batch_reader_returns_each_outcome(tmp_path) -> None:
    _tree(tmp_path)
    batched, _ = _workspaces(tmp_path)
    reader = batched.file_batch_reader()
    assert reader is batched

    outcomes = asyncio.run(
        reader.read_files(
            [
                WorkspaceFileReadRequest(path="pkg/b.py", max_bytes=4),
                WorkspaceFileReadRequest(path="missing.py", max_bytes=10),
                WorkspaceFileReadRequest(path="notes.md", offset=10_000, max_bytes=10),
                WorkspaceFileReadRequest(path="secret/key.txt", max_bytes=10),
                WorkspaceFileReadRequest(
                    path="pkg/a.py", start_line=2, end_line=3, max_bytes=100, head_bytes=8
                ),
            ]
        )
    )

    page, missing, offset, refused, lines = outcomes
    assert type(page) is WorkspaceReadResult
    assert (page.content, page.truncated, page.revision) == (b"prin", True, None)
    assert type(missing) is FileNotFoundError
    assert type(offset) is WorkspaceReadOffsetError and offset.offset == 10_000
    assert type(refused) is ValueError
    assert type(lines) is WorkspaceLineRead
    assert lines.window.content == b"a = 2\na = 3\n"
    assert lines.head is not None and lines.head.content == b"a = 1\na "
    assert batched._runner.commands == 1


def test_a_subclass_that_changes_reads_reads_one_file_at_a_time(tmp_path) -> None:
    class Custom(RunnerWorkspace):
        async def read_bytes(self, path: str, **kwargs: Any) -> WorkspaceReadResult:
            return await super().read_bytes(path, **kwargs)

    workspace = Custom(LocalRunner(tmp_path, inherit_env=False), python_executable=sys.executable)
    assert workspace.file_batch_reader() is None


def _handle(workspace: Any, tracker: InvocationSecretTracker) -> Any:
    return invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
    )


def test_invocation_handle_reads_the_batch_in_one_command(tmp_path) -> None:
    _tree(tmp_path)
    files = [
        {"path": "pkg/a.py", "start_line": 3, "end_line": 4},
        {"path": "missing.py"},
        {"path": "notes.md"},
    ]
    batched, single = _workspaces(tmp_path)
    tracker = InvocationSecretTracker(SecretRedactor())

    batch_result = _read(
        _handle(batched, tracker),
        {"files": files},
        invocation_secret_snapshot_provider=tracker.snapshot,
    )
    single_result = _read(
        _handle(single, tracker),
        {"files": files},
        invocation_secret_snapshot_provider=tracker.snapshot,
    )

    assert batch_result.structured == single_result.structured
    assert batch_result.content == single_result.content
    assert batched._runner.commands == 1


def test_invocation_handle_redacts_each_batched_file(tmp_path) -> None:
    secret = "batch-secret-canary-0123456789ABCDEF"
    (tmp_path / "config.txt").write_text(f"token={secret}\nend\n")
    (tmp_path / "plain.txt").write_text("nothing to hide\n")
    (tmp_path / "other.txt").write_text("nor here\n")
    (tmp_path / "lines.txt").write_text(f"one\n{secret}\nthree\n")
    files = [
        {"path": "plain.txt"},
        {"path": "other.txt"},
        {"path": "config.txt"},
        {"path": "lines.txt", "start_line": 2, "end_line": 3},
    ]
    batched, single = _workspaces(tmp_path)
    tracker = InvocationSecretTracker(SecretRedactor(secret))

    batch_result = _read(
        _handle(batched, tracker),
        {"files": files},
        invocation_secret_snapshot_provider=tracker.snapshot,
    )
    single_result = _read(
        _handle(single, tracker),
        {"files": files},
        invocation_secret_snapshot_provider=tracker.snapshot,
    )

    assert batch_result.structured == single_result.structured
    assert batch_result.content == single_result.content
    assert secret not in batch_result.content
    assert batch_result.structured["files"][0]["revision"] is not None
    # The plain files came from one command; the one holding the secret and the
    # line read (left to its own read while a secret is in scope) were read again.
    assert batched._runner.commands == single._runner.commands - 1
