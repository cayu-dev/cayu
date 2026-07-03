from __future__ import annotations

import asyncio
import sys

import pytest

from cayu.runners import LocalRunner
from cayu.workspaces import (
    LocalWorkspace,
    RunnerWorkspace,
    Workspace,
    matches_list_pattern,
    translate_list_pattern,
    validate_list_pattern,
)


def test_list_pattern_is_anchored_and_star_stays_in_one_segment() -> None:
    assert matches_list_pattern("a.txt", "*.txt") is True
    assert matches_list_pattern("nested/a.txt", "*.txt") is False
    assert matches_list_pattern("nested/a.txt", "nested/*.txt") is True
    assert matches_list_pattern("nested/deep/a.txt", "nested/*.txt") is False
    assert matches_list_pattern("a.txt.bak", "*.txt") is False
    assert matches_list_pattern("prefix-a.txt", "a.txt") is False


def test_list_pattern_double_star_matches_zero_or_more_directories() -> None:
    assert matches_list_pattern("a.txt", "**/*.txt") is True
    assert matches_list_pattern("d/a.txt", "**/*.txt") is True
    assert matches_list_pattern("d/e/a.txt", "**/*.txt") is True
    assert matches_list_pattern("d/a.md", "**/*.txt") is False
    assert matches_list_pattern("d/a.txt", "d/**/*.txt") is True
    assert matches_list_pattern("d/e/a.txt", "d/**/*.txt") is True
    assert matches_list_pattern("other/a.txt", "d/**/*.txt") is False


def test_list_pattern_trailing_double_star_matches_any_remaining_path() -> None:
    assert matches_list_pattern("a.txt", "**") is True
    assert matches_list_pattern("d/e/a.txt", "**") is True
    assert matches_list_pattern("d/a.txt", "d/**") is True
    assert matches_list_pattern("d/e/a.txt", "d/**") is True
    assert matches_list_pattern("d", "d/**") is False
    assert matches_list_pattern("other/a.txt", "d/**") is False


def test_list_pattern_question_mark_and_character_classes() -> None:
    assert matches_list_pattern("a.txt", "?.txt") is True
    assert matches_list_pattern("ab.txt", "?.txt") is False
    assert matches_list_pattern("d/x", "d/?") is True
    assert matches_list_pattern("a1.txt", "a[0-9].txt") is True
    assert matches_list_pattern("ax.txt", "a[0-9].txt") is False
    assert matches_list_pattern("ax.txt", "a[!0-9].txt") is True
    assert matches_list_pattern("a1.txt", "a[!0-9].txt") is False
    assert matches_list_pattern("a^.txt", "a[^x].txt") is True
    assert matches_list_pattern("ax.txt", "a[^x].txt") is True
    assert matches_list_pattern("ay.txt", "a[^x].txt") is False
    assert matches_list_pattern("a\\.txt", "a[\\x].txt") is True
    assert matches_list_pattern("ax.txt", "a[!\\].txt") is True
    assert matches_list_pattern("a\\.txt", "a[!\\].txt") is False
    # An unterminated class is treated as a literal bracket.
    assert matches_list_pattern("a[1.txt", "a[1.txt") is True


def test_list_pattern_ignores_empty_and_dot_segments() -> None:
    assert matches_list_pattern("d/a.txt", "d//a.txt") is True
    assert matches_list_pattern("d/a.txt", "./d/a.txt") is True
    assert translate_list_pattern(".") == r"(?!)"
    assert matches_list_pattern("a.txt", ".") is False


def test_validate_list_pattern_rejects_escapes_and_blank() -> None:
    assert validate_list_pattern("**/*") == "**/*"
    with pytest.raises(ValueError, match="pattern"):
        validate_list_pattern("/abs/*")
    with pytest.raises(ValueError, match="pattern"):
        validate_list_pattern("../*")
    with pytest.raises(ValueError, match="pattern"):
        validate_list_pattern("d/../../*")
    with pytest.raises(ValueError, match="pattern"):
        validate_list_pattern("   ")


_TREE = {
    "root.txt": b"root",
    "root.md": b"md",
    "nested/a.txt": b"a",
    "nested/deep/b.txt": b"b",
    "other/c.log": b"c",
}

_PATTERNS = (
    "**/*",
    "**/*.txt",
    "*.txt",
    "nested/*.txt",
    "nested/**",
    "**",
    "root.?xt",
    "other/*",
)


def _populate(workspace: Workspace) -> None:
    for path, content in _TREE.items():
        asyncio.run(workspace.write_bytes(path, content))


def _listing(workspace: Workspace, pattern: str) -> tuple[str, ...]:
    return asyncio.run(workspace.list(pattern, limit=50)).paths


def test_local_and_runner_backends_share_the_normative_matcher(tmp_path) -> None:
    local_root = tmp_path / "local"
    local_root.mkdir()
    runner_root = tmp_path / "runner"
    runner_root.mkdir()
    local = LocalWorkspace(local_root, workspace_id="local")
    runner = RunnerWorkspace(
        LocalRunner(runner_root, inherit_env=False),
        workspace_id="runner",
        python_executable=sys.executable,
    )
    _populate(local)
    _populate(runner)

    for pattern in _PATTERNS:
        expected = tuple(sorted(path for path in _TREE if matches_list_pattern(path, pattern)))
        assert _listing(local, pattern) == expected, pattern
        assert _listing(runner, pattern) == expected, pattern


def test_local_workspace_list_is_anchored(tmp_path) -> None:
    workspace = LocalWorkspace(tmp_path, workspace_id="local")
    _populate(workspace)

    result = asyncio.run(workspace.list("*.txt"))

    assert result.paths == ("root.txt",)
    assert result.total_count == 1
    assert result.truncated is False
