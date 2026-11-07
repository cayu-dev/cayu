"""The invocation workspace handle forwards line reads and text edits.

Every case runs the same tool call through a handle over a runner workspace
with the capabilities and through one over the same workspace without them,
which reads and edits in separate steps, and compares what came back, what
the file holds after, the mutations the handle reported and the commands each
took.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from cayu.runners import ExecCommand, ExecResult, LocalRunner
from cayu.runtime._invocation_secrets import InvocationSecretTracker
from cayu.tools import EditFileTool, ReadFileTool
from cayu.tools._file_lines import locate_workspace_lines
from cayu.tools._resources import InvocationWorkspaceMutationOwner, invocation_workspace_handle
from cayu.tools.base import ToolContext
from cayu.vaults import SecretRedactor
from cayu.vaults.base import ResolvedSecret
from cayu.workspaces import RunnerWorkspace, WorkspaceTextReplacement

SECRET = "handle-secret-canary-0123456789ABCDEF"


class _CountingRunner(LocalRunner):
    """A local runner that counts commands and can run a hook before the next one."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, inherit_env=False)
        self.commands = 0
        self.before_next: Any = None

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        self.commands += 1
        hook, self.before_next = self.before_next, None
        if hook is not None:
            hook()
        return await super().exec(command, **kwargs)


class _SeparateSteps(RunnerWorkspace):
    """The same workspace without one-call line reads, edits or batches."""

    def line_reader(self):
        return None

    def text_editor(self):
        return None

    def file_batch_reader(self):
        return None


class _Recorder:
    def __init__(self) -> None:
        self.operations: list[tuple[str, str, object]] = []

    def __call__(self, method: str, path: str, result: object) -> None:
        self.operations.append((method, path, result))


def _workspace(root: Path, *, separate: bool, **options: Any) -> RunnerWorkspace:
    kind = _SeparateSteps if separate else RunnerWorkspace
    return kind(
        _CountingRunner(root),
        workspace_id="same",
        python_executable=sys.executable,
        excluded_directory_names=("private",),
        **options,
    )


def _handle(
    workspace: RunnerWorkspace,
    tracker: InvocationSecretTracker,
    *,
    recorder: _Recorder | None = None,
    owner: InvocationWorkspaceMutationOwner | None = None,
) -> Any:
    return invocation_workspace_handle(
        workspace,
        redactor_snapshot_provider=tracker.snapshot,
        capture_observer=tracker.record_ambiguous_output_capture,
        mutation_owner=owner,
        direct_mutation_observer=recorder,
    )


def _tracker(*secrets: str) -> InvocationSecretTracker:
    return InvocationSecretTracker(SecretRedactor(*secrets))


def _add_secret(tracker: InvocationSecretTracker, value: str = SECRET) -> None:
    tracker.record(ResolvedSecret(name="token", value=SecretStr(value)))


def _call(tool: Any, handle: Any, tracker: InvocationSecretTracker, args: dict[str, Any]) -> Any:
    ctx = ToolContext(
        session_id="sess", workspace=handle, invocation_secret_snapshot_provider=tracker.snapshot
    )
    try:
        return asyncio.run(tool.run(ctx, args)).model_dump()
    except Exception as exc:
        return ("raised", type(exc).__name__, str(exc))


def _numbered(count: int, width: int = 120) -> bytes:
    return b"".join(
        f"line {n:06d} ".encode().ljust(width - 1, b".") + b"\n" for n in range(1, count + 1)
    )


def _tree(root: Path) -> None:
    root.mkdir()
    (root / "a.py").write_text("".join(f"a = {n}\n" for n in range(1, 41)))
    (root / "big.txt").write_bytes(_numbered(6_000))
    (root / "long.txt").write_text("short\n" + "x" * 5_000 + "\nend\n")
    (root / "no-newline.txt").write_text("one\ntwo\nthree")
    (root / "binary.bin").write_bytes(b"\x00\x01binary\x00\n" * 10)
    (root / "config.txt").write_text(f"one\ntoken={SECRET}\nthree\n")
    (root / "private").mkdir()
    (root / "private" / "key.txt").write_text("hidden\nlines\n")


LINE_READS: dict[str, dict[str, Any]] = {
    "middle": {"path": "a.py", "start_line": 3, "end_line": 4},
    "first lines": {"path": "a.py", "start_line": 1, "end_line": 5},
    "to the end": {"path": "a.py", "start_line": 38},
    "past the end": {"path": "a.py", "start_line": 100},
    "end past the last line": {"path": "a.py", "start_line": 39, "end_line": 50},
    "deep in a large file": {"path": "big.txt", "start_line": 5_000, "end_line": 5_010},
    "cut by max_bytes": {"path": "big.txt", "start_line": 10, "end_line": 40, "max_bytes": 500},
    "line longer than max_bytes": {"path": "long.txt", "start_line": 2, "max_bytes": 100},
    "no final newline": {"path": "no-newline.txt", "start_line": 2},
    "binary": {"path": "binary.bin", "start_line": 2},
    "a secret in the range": {"path": "config.txt", "start_line": 2, "end_line": 3},
    "missing file": {"path": "missing.txt", "start_line": 2},
    "refused path": {"path": "private/key.txt", "start_line": 2},
}


@pytest.mark.parametrize("case", LINE_READS)
@pytest.mark.parametrize("secret_in_scope", [False, True], ids=["no secret", "secret"])
def test_a_line_read_through_the_handle_matches_the_separate_steps(
    tmp_path: Path, case: str, secret_in_scope: bool
) -> None:
    results = []
    commands = []
    for separate in (True, False):
        root = tmp_path / ("separate" if separate else "forwarded")
        _tree(root)
        workspace = _workspace(root, separate=separate)
        tracker = _tracker(*((SECRET,) if secret_in_scope else ()))
        handle = _handle(workspace, tracker)
        results.append(_call(ReadFileTool(), handle, tracker, LINE_READS[case]))
        commands.append(workspace._runner.commands)
        assert tracker.seal_for_publication().unsafe_output is False

    assert results[1] == results[0]
    if secret_in_scope:
        # Line spans count raw bytes, so a secret keeps the separate redacted reads.
        assert commands[1] == commands[0]
    elif case not in {"missing file", "refused path"}:
        assert commands[1] == 1 <= commands[0]
        # Past the end, the separate steps stop after locating the range.
        assert (commands[0] > 1) is (case != "past the end")


TEXT = "alpha\nbeta é\ngamma\nbeta é\n"

EDITS: dict[str, tuple[Any, list[dict[str, Any]], dict[str, Any]]] = {
    "one edit": (TEXT, [{"old_text": "alpha", "new_text": "ALPHA"}], {}),
    "every occurrence": (
        TEXT,
        [
            {"old_text": "beta é", "new_text": "β", "expected_replacements": 2},
            {"old_text": "gamma\n", "new_text": ""},
        ],
        {},
    ),
    "missing match": (TEXT, [{"old_text": "delta", "new_text": "x"}], {}),
    "overlapping edits": (
        TEXT,
        [{"old_text": "alpha\nbe", "new_text": "x"}, {"old_text": "a\nbeta é\ng", "new_text": "y"}],
        {},
    ),
    "stale revision": (TEXT, [{"old_text": "alpha", "new_text": "x"}], {"stale": True}),
    "invalid UTF-8": (b"alpha\xff\n", [{"old_text": "alpha", "new_text": "x"}], {}),
    "larger than max_bytes": (TEXT, [{"old_text": "alpha", "new_text": "x"}], {"max_bytes": 8}),
    "result too large": (
        TEXT,
        [{"old_text": "alpha", "new_text": "a" * 40}],
        {"max_bytes": len(TEXT.encode()) + 10},
    ),
    "missing file": (None, [{"old_text": "alpha", "new_text": "x"}], {}),
    "refused path": (TEXT, [{"old_text": "alpha", "new_text": "x"}], {"path": "private/n.txt"}),
    "a secret in the file": (
        f"token={SECRET}\nalpha\n",
        [{"old_text": "alpha", "new_text": "omega"}],
        {},
    ),
    # Edit text too large to send with one command: read, then replace.
    "large edit text": (
        "é" * 100_000,
        [{"old_text": "é" * 100_000, "new_text": "à" * 100_000}],
        {},
    ),
}


@pytest.mark.parametrize("case", EDITS)
@pytest.mark.parametrize("secret_in_scope", [False, True], ids=["no secret", "secret"])
@pytest.mark.parametrize("owned", [False, True], ids=["direct", "owned"])
def test_an_edit_through_the_handle_matches_the_separate_steps(
    tmp_path: Path, case: str, secret_in_scope: bool, owned: bool
) -> None:
    content, edits, options = EDITS[case]
    path = options.get("path", "notes.txt")
    results = []
    files = []
    operations = []
    commands = []
    for separate in (True, False):
        root = tmp_path / ("separate" if separate else "forwarded")
        root.mkdir()
        (root / "private").mkdir()
        if content is not None:
            (root / path).write_bytes(content if isinstance(content, bytes) else content.encode())
        workspace = _workspace(root, separate=separate)
        tracker = _tracker(*((SECRET,) if secret_in_scope else ()))
        recorder = _Recorder()
        owner = InvocationWorkspaceMutationOwner() if owned else None
        handle = _handle(workspace, tracker, recorder=recorder, owner=owner)
        read = _call(ReadFileTool(), _handle(workspace, tracker), tracker, {"path": path})
        revision = read["structured"].get("revision") if isinstance(read, dict) else None
        if options.get("stale") or revision is None:
            revision = "sha256:" + "1" * 64
        workspace._runner.commands = 0
        args: dict[str, Any] = {"path": path, "expected_revision": revision, "edits": edits}
        if "max_bytes" in options:
            args["max_bytes"] = options["max_bytes"]
        results.append(_call(EditFileTool(), handle, tracker, args))
        files.append((root / path).read_bytes() if (root / path).exists() else None)
        operations.append(recorder.operations)
        commands.append(workspace._runner.commands)
        assert tracker.seal_for_publication().unsafe_output is False

    assert results[1] == results[0]
    assert files[1] == files[0]
    assert operations[1] == operations[0]
    if not secret_in_scope and case == "one edit":
        assert commands == [2, 1]
    else:
        assert commands[1] <= commands[0]


def test_the_handle_offers_line_reads_and_edits_only_when_its_workspace_does(
    tmp_path: Path,
) -> None:
    _tree(tmp_path / "root")
    tracker = _tracker()
    forwarded = _handle(_workspace(tmp_path / "root", separate=False), tracker)
    separate = _handle(_workspace(tmp_path / "root", separate=True), tracker)

    assert forwarded.line_reader() is forwarded
    assert forwarded.text_editor() is forwarded
    assert separate.line_reader() is None
    assert separate.text_editor() is None

    _add_secret(tracker)
    assert forwarded.line_reader() is None
    assert forwarded.text_editor() is None


def _redacted_steps(handle: Any, path: str, *, start_line: int, end_line: int | None) -> Any:
    """What a caller without a line reader reads: the located span, its window and head."""

    span = asyncio.run(
        locate_workspace_lines(handle, path, start_line=start_line, end_line=end_line)
    )
    window = asyncio.run(
        handle.read_bytes(
            path,
            offset=span.start_offset,
            max_bytes=max(1, min(1_000, span.end_offset - span.start_offset)),
        )
    )
    head = asyncio.run(handle.read_bytes(path, max_bytes=8))
    return span, window, head


def test_a_line_read_asked_for_with_a_secret_in_scope_reads_through_the_redacted_steps(
    tmp_path: Path,
) -> None:
    _tree(tmp_path / "root")
    tracker = _tracker()
    workspace = _workspace(tmp_path / "root", separate=False)
    handle = _handle(workspace, tracker)
    assert handle.line_reader() is handle
    # The secret comes into scope after the caller took the line reader.
    _add_secret(tracker)

    read = asyncio.run(
        handle.read_lines("config.txt", start_line=2, end_line=2, max_bytes=1_000, head_bytes=8)
    )

    separate = _handle(_workspace(tmp_path / "root", separate=True), tracker)
    span, window, head = _redacted_steps(separate, "config.txt", start_line=2, end_line=2)
    assert (read.span, read.window, read.head) == (span, window, head)
    assert SECRET.encode() not in read.window.content
    assert tracker.seal_for_publication().unsafe_output is False


def test_a_secret_registered_during_a_line_read_reads_it_again_redacted(tmp_path: Path) -> None:
    _tree(tmp_path / "root")
    tracker = _tracker()
    workspace = _workspace(tmp_path / "root", separate=False)
    handle = _handle(workspace, tracker)
    workspace._runner.before_next = lambda: _add_secret(tracker)

    read = asyncio.run(
        handle.read_lines("config.txt", start_line=2, end_line=3, max_bytes=1_000, head_bytes=8)
    )

    separate_workspace = _workspace(tmp_path / "root", separate=True)
    separate = _handle(separate_workspace, tracker)
    span, window, head = _redacted_steps(separate, "config.txt", start_line=2, end_line=3)
    assert (read.span, read.window, read.head) == (span, window, head)
    assert SECRET.encode() not in read.window.content
    # The forwarded command, then the same redacted steps.
    assert workspace._runner.commands == 1 + separate_workspace._runner.commands
    assert tracker.seal_for_publication().unsafe_output is False


def test_an_edit_asked_for_with_a_secret_in_scope_is_left_to_the_caller(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _tree(root)
    tracker = _tracker()
    workspace = _workspace(root, separate=False)
    handle = _handle(workspace, tracker)
    assert handle.text_editor() is handle
    _add_secret(tracker)
    before = (root / "config.txt").read_bytes()
    revision = asyncio.run(handle.read_bytes("config.txt")).revision

    edited = asyncio.run(
        handle.edit_text(
            "config.txt",
            [WorkspaceTextReplacement(old_text="one", new_text="ONE")],
            expected_revision=revision,
            max_bytes=1_000,
        )
    )

    assert edited.mutation is None
    assert edited.read == asyncio.run(handle.read_bytes("config.txt", max_bytes=1_000))
    assert SECRET.encode() not in edited.read.content
    assert (root / "config.txt").read_bytes() == before


def test_a_secret_registered_during_an_edit_marks_the_output_unsafe(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _tree(root)
    tracker = _tracker()
    recorder = _Recorder()
    workspace = _workspace(root, separate=False)
    handle = _handle(workspace, tracker, recorder=recorder)
    revision = asyncio.run(handle.read_bytes("a.py")).revision
    workspace._runner.before_next = lambda: _add_secret(tracker)

    edited = asyncio.run(
        handle.edit_text(
            "a.py",
            [WorkspaceTextReplacement(old_text="a = 1\n", new_text="a = one\n")],
            expected_revision=revision,
            max_bytes=10_000,
        )
    )

    assert edited.mutation is not None
    assert (root / "a.py").read_text().startswith("a = one\n")
    assert recorder.operations == [("replace_bytes", "a.py", edited.mutation)]
    # The edit's read was taken before the secret came into scope.
    assert tracker.seal_for_publication().unsafe_output is True


def test_an_edit_past_the_workspace_read_limit_is_refused_as_the_separate_steps_refuse_it(
    tmp_path: Path,
) -> None:
    results = []
    files = []
    for separate in (True, False):
        root = tmp_path / ("separate" if separate else "forwarded")
        root.mkdir()
        (root / "notes.txt").write_text("alpha\n" + "x" * 2_000 + "\n")
        workspace = _workspace(root, separate=separate, default_read_limit_bytes=1024)
        tracker = _tracker()
        # The file's current revision, read without the 1 KiB limit.
        revision = asyncio.run(_workspace(root, separate=True).read_bytes("notes.txt")).revision
        assert revision is not None
        args = {
            "path": "notes.txt",
            "expected_revision": revision,
            "edits": [{"old_text": "alpha", "new_text": "omega"}],
            "max_bytes": 4096,
        }
        results.append(_call(EditFileTool(), _handle(workspace, tracker), tracker, args))
        files.append((root / "notes.txt").read_bytes())

    assert results[1] == results[0]
    assert results[0]["structured"]["reason"] == "file_truncated"
    assert files[1] == files[0]


def test_a_failed_replacement_raises_what_the_separate_replace_raises(tmp_path: Path) -> None:
    raised = []
    for separate in (True, False):
        root = tmp_path / ("separate" if separate else "forwarded")
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "n.txt").write_text("alpha\n")
        workspace = _workspace(root, separate=separate)
        tracker = _tracker()
        revision = asyncio.run(workspace.read_bytes("sub/n.txt")).revision
        (root / "sub").chmod(0o500)
        try:
            outcome = _call(
                EditFileTool(),
                _handle(workspace, tracker, owner=InvocationWorkspaceMutationOwner()),
                tracker,
                {
                    "path": "sub/n.txt",
                    "expected_revision": revision,
                    "edits": [{"old_text": "alpha", "new_text": "omega"}],
                },
            )
        finally:
            (root / "sub").chmod(0o700)
        assert isinstance(outcome, tuple), outcome
        raised.append(outcome[1])
        assert (root / "sub" / "n.txt").read_text() == "alpha\n"

    assert raised[1] == raised[0]


def test_a_file_that_grows_during_the_redacted_steps_keeps_its_located_lines(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    _tree(root)
    tracker = _tracker()
    workspace = _workspace(root, separate=False)
    handle = _handle(workspace, tracker)
    runner = workspace._runner
    exec_command = runner.exec

    async def growing(command: ExecCommand, **kwargs: Any) -> ExecResult:
        # The secret comes into scope during the forwarded read; then the file
        # grows before every later command, as a log does.
        if runner.commands == 0:
            _add_secret(tracker)
        else:
            with (root / "a.py").open("a") as log:
                log.write("a = more\n")
        return await exec_command(command, **kwargs)

    runner.exec = growing  # type: ignore[method-assign]

    read = asyncio.run(
        handle.read_lines("a.py", start_line=3, end_line=4, max_bytes=1_000, head_bytes=8)
    )

    lines = (root / "a.py").read_bytes().splitlines(keepends=True)
    start = sum(len(line) for line in lines[:2])
    assert (read.span.start_offset, read.span.end_offset) == (
        start,
        start + len(b"".join(lines[2:4])),
    )
    assert read.window.content.startswith(b"".join(lines[2:4]))
    assert read.span.total_bytes == read.window.total_bytes
