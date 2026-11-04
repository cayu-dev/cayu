"""edit_file through a runner workspace's text editor: one command, same results."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

from cayu.runners import ExecCommand, ExecResult, LocalRunner
from cayu.tools import EditFileTool, ReadFileTool
from cayu.tools.base import ToolContext
from cayu.workspaces import (
    LocalWorkspace,
    RunnerWorkspace,
    WorkspaceMutationResult,
    WorkspaceTextReplacement,
)
from cayu.workspaces import _guest_guard as guest_guard


class _CountingRunner(LocalRunner):
    def __init__(self, root: Path) -> None:
        super().__init__(root, inherit_env=False)
        self.commands = 0

    async def exec(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        self.commands += 1
        return await super().exec(command, **kwargs)


def _run(tool: Any, workspace: Any, args: dict[str, Any]) -> Any:
    return asyncio.run(tool.run(ToolContext(session_id="sess", workspace=workspace), args))


def _revision(workspace: Any, path: str) -> str:
    result = _run(ReadFileTool(), workspace, {"path": path})
    if result.is_error:
        return "sha256:" + "0" * 64
    return result.structured["revision"]


TEXT = "alpha\nbeta é\ngamma\nbeta é\n"

CASES = {
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
    "ambiguous match": (TEXT, [{"old_text": "beta", "new_text": "x"}], {}),
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
    "huge result": (
        "a" * 1000,
        [{"old_text": "a", "new_text": "b" * 60, "expected_replacements": 1000}],
        {"max_bytes": 1024},
    ),
    # Edit text too large to send with one command: read, then replace.
    "large edit text": (
        "é" * 100_000,
        [{"old_text": "é" * 100_000, "new_text": "à" * 100_000}],
        {"commands": 2},
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_runner_text_editor_matches_the_two_step_edit_in_one_command(
    tmp_path: Path, case: str
) -> None:
    content, edits, options = CASES[case]
    results = []
    files = []
    for name in ("local", "runner"):
        root = tmp_path / name
        root.mkdir()
        if content is not None:
            (root / "notes.txt").write_bytes(
                content if isinstance(content, bytes) else content.encode()
            )
        runner = _CountingRunner(root)
        workspace = (
            LocalWorkspace(root, workspace_id="same")
            if name == "local"
            else RunnerWorkspace(runner, workspace_id="same", python_executable=sys.executable)
        )
        revision = _revision(workspace, "notes.txt")
        if options.get("stale"):
            revision = "sha256:" + "1" * 64
        runner.commands = 0
        args: dict[str, Any] = {"path": "notes.txt", "expected_revision": revision, "edits": edits}
        if "max_bytes" in options:
            args["max_bytes"] = options["max_bytes"]
        results.append(_run(EditFileTool(), workspace, args))
        files.append((root / "notes.txt").read_bytes() if content is not None else None)
        if name == "runner":
            assert runner.commands == options.get("commands", 1)

    assert results[1].model_dump() == results[0].model_dump()
    assert files[1] == files[0]


def test_guest_edit_refuses_an_oversized_result_before_building_it() -> None:
    namespace: dict[str, Any] = {}
    exec(guest_guard.GUEST_COMBINED_MODES_SOURCE, namespace)
    apply_text_edits = namespace["apply_text_edits"]

    # Built, this result would take about a gigabyte.
    assert apply_text_edits("a" * 1000, [["a", "b" * (1 << 20), 1000]], 1024) is None
    assert apply_text_edits("a" * 1000, [["a", "b", 999]], 1024) is None
    assert apply_text_edits("a-a", [["a", "bb", 2]], 5) == "bb-bb"
    assert apply_text_edits("a-a", [["a", "bb", 2]], 4) is None


def test_text_editor_refuses_an_edit_without_writing(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text(TEXT)
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False), python_executable=sys.executable
    )
    editor = workspace.text_editor()
    assert editor is workspace
    revision = _revision(workspace, "notes.txt")

    refused = asyncio.run(
        editor.edit_text(
            "notes.txt",
            [WorkspaceTextReplacement("beta", "x")],
            expected_revision=revision,
            max_bytes=1024,
        )
    )
    applied = asyncio.run(
        editor.edit_text(
            "notes.txt",
            [WorkspaceTextReplacement("beta", "x", expected_replacements=2)],
            expected_revision=revision,
            max_bytes=1024,
        )
    )

    assert refused.mutation is None
    assert refused.read.content == TEXT.encode()
    assert refused.read.revision == revision
    assert applied.read.content == TEXT.encode()
    assert type(applied.mutation) is WorkspaceMutationResult
    assert applied.mutation.before_revision == revision
    assert (tmp_path / "notes.txt").read_text() == TEXT.replace("beta", "x")


def test_subclasses_that_change_reads_or_writes_keep_the_separate_steps(tmp_path: Path) -> None:
    class _AuditedWorkspace(RunnerWorkspace):
        async def replace_bytes(self, path: str, content: bytes, *, expected_revision: str):
            return await super().replace_bytes(path, content, expected_revision=expected_revision)

    class _FilteredWorkspace(RunnerWorkspace):
        async def read_bytes(self, path: str, *, offset: int = 0, max_bytes: int | None = None):
            return await super().read_bytes(path, offset=offset, max_bytes=max_bytes)

    runner = LocalRunner(tmp_path, inherit_env=False)

    assert _AuditedWorkspace(runner).text_editor() is None
    assert _AuditedWorkspace(runner).line_reader() is not None
    assert _FilteredWorkspace(runner).text_editor() is None
    assert _FilteredWorkspace(runner).line_reader() is None
    assert LocalWorkspace(tmp_path).text_editor() is None
    assert LocalWorkspace(tmp_path).line_reader() is None
