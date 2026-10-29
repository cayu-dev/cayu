"""Line-range reads through runners that cap command output."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

from cayu.runners import ExecCommand, ExecResult, LocalRunner
from cayu.sandbox_broker.client import MAX_COMMAND_OUTPUT_BYTES
from cayu.sandbox_broker.runner import BrokeredSandboxRunner
from cayu.tools import ReadFileTool
from cayu.tools._file_lines import locate_workspace_lines
from cayu.tools.base import ToolContext
from cayu.vaults import REDACTED_SECRET, SecretRedactor
from cayu.workspaces import LocalWorkspace, RunnerWorkspace
from cayu.workspaces._guest_guard import guard_read_limit


class _CappedRunner(LocalRunner):
    """A local runner whose command output is cut at a fixed size, like a broker's."""

    def __init__(self, root: Any, *, cap: int) -> None:
        super().__init__(root, inherit_env=False)
        self.max_exec_output_bytes = cap
        self.stdout_sizes: list[int] = []

    async def exec(
        self,
        command: ExecCommand,
        *,
        output_limit_bytes: int | None = None,
        **kwargs: Any,
    ) -> ExecResult:
        cap = self.max_exec_output_bytes
        assert cap is not None
        limit = cap if output_limit_bytes is None else min(output_limit_bytes, cap)
        result = await super().exec(command, output_limit_bytes=limit, **kwargs)
        assert not result.stdout_truncated, "a guarded read asked for more than the cap"
        self.stdout_sizes.append(len(result.stdout))
        return result


def _capped_workspace(root: Any, cap: int) -> tuple[RunnerWorkspace, _CappedRunner]:
    runner = _CappedRunner(root, cap=cap)
    workspace = RunnerWorkspace(runner, workspace_id="capped", python_executable=sys.executable)
    return workspace, runner


def _read(workspace: Any, args: dict[str, Any]) -> Any:
    ctx = ToolContext(session_id="sess", workspace=workspace)
    return asyncio.run(ReadFileTool().run(ctx, args))


def _text(result: Any) -> str:
    return result.content.rsplit("[/read_file metadata]\n", 1)[1]


def _numbered_lines(count: int, width: int = 120) -> bytes:
    return b"".join(
        f"line {n:06d} ".encode().ljust(width - 1, b".") + b"\n" for n in range(1, count + 1)
    )


def test_brokered_runner_declares_its_output_cap() -> None:
    assert BrokeredSandboxRunner.max_exec_output_bytes == MAX_COMMAND_OUTPUT_BYTES
    assert LocalRunner.max_exec_output_bytes is None


def test_line_range_deep_in_a_file_larger_than_the_runner_cap(tmp_path) -> None:
    # A 1.6 MiB file read by line range through a 2 MiB output cap.
    content = _numbered_lines(14_000)
    assert len(content) > 1_600_000
    (tmp_path / "contracts.md").write_bytes(content)
    workspace, runner = _capped_workspace(tmp_path, MAX_COMMAND_OUTPUT_BYTES)

    result = _read(workspace, {"path": "contracts.md", "start_line": 13_470, "end_line": 13_530})

    assert result.is_error is False
    lines = content.splitlines(keepends=True)
    assert _text(result) == b"".join(lines[13_469:13_530]).decode()
    assert result.structured["start_line"] == 13_470
    assert result.structured["end_line"] == 13_530
    assert result.structured["total_lines"] == 14_000
    assert result.structured["next_start_line"] is None
    assert result.structured["truncated"] is False
    # Only the 61 requested lines (about 7 KiB) and the file's first 8,000
    # bytes, for the binary check, cross the runner, base64-encoded.
    assert sum(runner.stdout_sizes) < 24 * 1024


def test_line_range_pages_through_a_window_smaller_than_the_range(tmp_path) -> None:
    content = _numbered_lines(3_000)
    (tmp_path / "big.txt").write_bytes(content)
    cap = 16 * 1024
    workspace, runner = _capped_workspace(tmp_path, cap)
    window = guard_read_limit(runner)
    assert window is not None and window < 4_000 * 120

    first = _read(workspace, {"path": "big.txt", "start_line": 2_000, "end_line": 2_999})

    assert first.is_error is False
    lines = content.splitlines(keepends=True)
    returned = first.structured["end_line"] - 2_000 + 1
    assert 0 < returned < 1_000
    assert _text(first) == b"".join(lines[1_999 : 1_999 + returned]).decode()
    assert first.structured["truncated"] is True
    assert first.structured["next_start_line"] == 2_000 + returned
    assert max(runner.stdout_sizes) <= cap


def test_line_longer_than_the_runner_window_returns_offset_guidance(tmp_path) -> None:
    long_line = b"x" * (200 * 1024) + b"\n"
    content = b"first\nsecond\n" + long_line + b"last\n"
    (tmp_path / "minified.js").write_bytes(content)
    cap = 64 * 1024
    workspace, runner = _capped_workspace(tmp_path, cap)
    window = workspace.bounded_read_limit(4 * 1024 * 1024)

    result = _read(workspace, {"path": "minified.js", "start_line": 3})

    assert result.is_error is True
    assert result.structured == {
        "error": "invalid_arguments",
        "reason": "line_exceeds_read_window",
        "path": "minified.js",
        "start_line": 3,
        "total_bytes": len(content),
        "max_read_bytes": window,
        "offset": 13,
        "max_bytes": window,
    }
    assert f"offset=13 with max_bytes={window}" in result.content
    assert "outcome_unknown" not in result.content

    # Following the guidance works.
    page = _read(workspace, {"path": "minified.js", "offset": 13, "max_bytes": window})
    assert page.is_error is False
    assert _text(page) == "x" * window
    assert page.structured["next_offset"] == 13 + window
    assert max(runner.stdout_sizes) <= cap


def test_line_longer_than_caller_max_bytes_is_still_cut_with_a_hint(tmp_path) -> None:
    (tmp_path / "wide.txt").write_bytes(b"a\n" + b"y" * 5_000 + b"\nz\n")
    workspace, _ = _capped_workspace(tmp_path, 64 * 1024)

    result = _read(workspace, {"path": "wide.txt", "start_line": 2, "max_bytes": 1_000})

    assert result.is_error is False
    assert _text(result) == "y" * 1_000
    assert result.structured["line_truncated"] is True
    assert result.structured["next_start_line"] == 2


def test_byte_reads_larger_than_the_runner_window_return_a_shorter_page(tmp_path) -> None:
    content = b"b" * (100 * 1024)
    (tmp_path / "data.txt").write_bytes(content)
    cap = 32 * 1024
    workspace, runner = _capped_workspace(tmp_path, cap)
    window = guard_read_limit(runner)
    assert window is not None

    raw = asyncio.run(workspace.read_bytes("data.txt"))
    result = _read(workspace, {"path": "data.txt", "max_bytes": 4 * 1024 * 1024})

    assert raw.content == content[:window]
    assert raw.truncated is True
    assert result.is_error is False
    assert result.structured["next_offset"] == window
    assert max(runner.stdout_sizes) <= cap


SMALL = b"alpha\nbeta\n\ngamma \xc3\xa9\ndelta"


@pytest.mark.parametrize(
    "args",
    [
        {"start_line": 1},
        {"start_line": 2, "end_line": 3},
        {"start_line": 5},
        {"start_line": 4, "end_line": 9},
        {"end_line": 1},
        {"start_line": 2, "max_bytes": 6},
        {"start_line": 6},
        {"start_line": 9},
    ],
)
def test_small_file_line_reads_match_across_workspaces(tmp_path, args) -> None:
    (tmp_path / "small.txt").write_bytes(SMALL)
    local = LocalWorkspace(tmp_path, workspace_id="same")
    uncapped = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        workspace_id="same",
        python_executable=sys.executable,
    )
    capped, _ = _capped_workspace(tmp_path, 8 * 1024)
    request = {"path": "small.txt", **args}

    results = [_read(workspace, request) for workspace in (local, uncapped, capped)]

    assert results[1].model_dump() == results[0].model_dump()
    assert results[2].model_dump() == results[0].model_dump()


@pytest.mark.parametrize(
    "content",
    [b"", b"\n", b"one", b"one\n", b"one\ntwo", b"one\ntwo\n", b"\n\n\n", b"a\n" * 70_000],
)
@pytest.mark.parametrize("line_range", [(1, None), (1, 1), (2, None), (2, 3), (3, 3), (5, None)])
def test_guest_and_paging_line_locators_agree(tmp_path, content, line_range) -> None:
    (tmp_path / "f.txt").write_bytes(content)
    start_line, end_line = line_range
    guest = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        workspace_id="guest",
        python_executable=sys.executable,
    )
    local = LocalWorkspace(tmp_path, workspace_id="local")
    assert local.line_locator() is None

    located = [
        asyncio.run(
            locate_workspace_lines(workspace, "f.txt", start_line=start_line, end_line=end_line)
        )
        for workspace in (guest, local)
    ]

    assert located[0] == located[1]
    span = located[0]
    lines = content.splitlines(keepends=True)
    assert span.total_lines == len(lines)
    last = len(lines) if end_line is None else min(end_line, len(lines))
    assert content[span.start_offset : span.end_offset] == b"".join(lines[start_line - 1 : last])


def test_guest_line_locator_reports_missing_files(tmp_path) -> None:
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        workspace_id="guest",
        python_executable=sys.executable,
    )
    (tmp_path / "dir").mkdir()

    for path in ("missing.txt", "dir"):
        with pytest.raises(FileNotFoundError):
            asyncio.run(workspace.locate_lines(path, start_line=1))


def _redacting_read(workspace: Any, secret: str, args: dict[str, Any]) -> Any:
    ctx = ToolContext(
        session_id="sess",
        workspace=workspace,
        invocation_secret_redactor=lambda: SecretRedactor(secret),
    )
    return asyncio.run(ReadFileTool().run(ctx, args))


@pytest.mark.parametrize("capped", [False, True])
def test_line_ending_on_a_secret_prefix_still_advances(tmp_path, capped) -> None:
    (tmp_path / "f.txt").write_text("one\nsafe\n")
    workspace = (
        _capped_workspace(tmp_path, 8 * 1024)[0]
        if capped
        else LocalWorkspace(tmp_path, workspace_id="local")
    )

    result = _redacting_read(
        workspace, "one\ntwo", {"path": "f.txt", "start_line": 1, "end_line": 1}
    )

    assert result.is_error is False
    assert _text(result) == "one\n"
    assert result.structured["end_line"] == 1
    assert result.structured["next_start_line"] is None
    assert result.structured["revision"] is None


def test_redacted_line_shorter_than_its_source_fits_max_bytes(tmp_path) -> None:
    secret = "K" * 2000
    (tmp_path / "f.txt").write_text(f"a\n{secret}\nb\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="local")

    result = _redacting_read(
        workspace, secret, {"path": "f.txt", "start_line": 2, "end_line": 2, "max_bytes": 1000}
    )

    assert _text(result) == f"{REDACTED_SECRET}\n"
    assert result.structured["next_start_line"] is None
    assert result.structured["truncated"] is False


@pytest.mark.parametrize("capped", [False, True])
def test_binary_file_is_reported_for_any_start_line(tmp_path, capped) -> None:
    (tmp_path / "blob.txt").write_bytes(b"\x00\x01binary\nmore\n")
    workspace = (
        _capped_workspace(tmp_path, 8 * 1024)[0]
        if capped
        else LocalWorkspace(tmp_path, workspace_id="local")
    )

    for start_line in (1, 2):
        result = _read(workspace, {"path": "blob.txt", "start_line": start_line})
        assert result.is_error is True
        assert result.structured["binary"] is True


class _AppendingRunner(LocalRunner):
    """Appends a line to a log before every command, like a process writing it."""

    def __init__(self, root: Any) -> None:
        super().__init__(root, inherit_env=False)
        self.log = root / "server.log"
        self.appended = 0

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        self.appended += 1
        with self.log.open("a") as log:
            log.write(f"appended {self.appended}\n")
        return await super().exec(command, **kwargs)


@pytest.mark.parametrize(("start_line", "end_line"), [(1, 5), (50, 55), (98, None)])
def test_line_reads_of_a_growing_log_succeed(tmp_path, start_line, end_line) -> None:
    (tmp_path / "server.log").write_text("".join(f"line {n}\n" for n in range(1, 101)))
    runner = _AppendingRunner(tmp_path)
    workspace = RunnerWorkspace(runner, workspace_id="log", python_executable=sys.executable)
    args: dict[str, Any] = {"path": "server.log", "start_line": start_line}
    if end_line is not None:
        args["end_line"] = end_line

    result = _read(workspace, args)

    assert result.is_error is False
    lines = _text(result).splitlines()
    assert lines[0] == f"line {start_line}"
    if end_line is not None:
        assert lines == [f"line {n}" for n in range(start_line, end_line + 1)]
    assert runner.appended >= 2


def test_line_read_of_a_file_that_shrank_below_the_range_fails(tmp_path, monkeypatch) -> None:
    import cayu.tools.files as files_module

    path = tmp_path / "f.txt"
    path.write_text("".join(f"line {n}\n" for n in range(1, 101)))
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    locate = files_module.locate_workspace_lines

    async def locate_then_truncate(*args: Any, **kwargs: Any) -> Any:
        span = await locate(*args, **kwargs)
        path.write_text("line 1\n")
        return span

    monkeypatch.setattr(files_module, "locate_workspace_lines", locate_then_truncate)

    result = _read(workspace, {"path": "f.txt", "start_line": 50, "end_line": 55})

    assert result.is_error is True
    assert result.structured == {"error": "workspace_file_changed", "path": "f.txt"}


def test_rewrite_after_locating_never_reports_a_whole_file_revision(tmp_path, monkeypatch) -> None:
    import cayu.tools.files as files_module

    path = tmp_path / "f.txt"
    path.write_text("ab\ncd\n")
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    locate = files_module.locate_workspace_lines

    async def locate_then_rewrite(*args: Any, **kwargs: Any) -> Any:
        span = await locate(*args, **kwargs)
        # Same size, one more line: the located span still ends at the file's end.
        path.write_text("a\nb\nc\n")
        return span

    monkeypatch.setattr(files_module, "locate_workspace_lines", locate_then_rewrite)

    result = _read(workspace, {"path": "f.txt", "start_line": 1, "end_line": 2})

    assert result.is_error is False
    assert _text(result) == "a\nb\n"
    assert result.structured["revision"] is None
    assert result.structured["sha256"] is None


def test_wrappers_report_a_deferred_runners_cap_once_it_materializes(tmp_path) -> None:
    from types import SimpleNamespace

    from cayu.environments.deferred import DeferredRunner
    from cayu.runners import LaunchedRunner

    state = SimpleNamespace(materialized_runner=None)
    deferred = DeferredRunner(
        state,
        default_cwd=str(tmp_path),
        configured_candidate=None,
        environment_authority=None,
        isolation="unknown",
    )
    wrapped = LaunchedRunner(deferred, lambda command: command)
    assert wrapped.max_exec_output_bytes is None

    live = LocalRunner(tmp_path, inherit_env=False)
    live.max_exec_output_bytes = 2 * 1024 * 1024
    state.materialized_runner = live

    assert wrapped.max_exec_output_bytes == 2 * 1024 * 1024
    assert RunnerWorkspace(wrapped).bounded_read_limit(4 * 1024 * 1024) == RunnerWorkspace(
        deferred
    ).bounded_read_limit(4 * 1024 * 1024)
