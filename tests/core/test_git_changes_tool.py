from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

import cayu.tools.git as git_module
from cayu.core.tools import ToolContext
from cayu.runners import LocalRunner
from cayu.tools import GitChangesTool
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import InvocationRunnerHandle
from cayu.tools.git import _workspace_cwd
from cayu.vaults import SecretRedactor
from cayu.workspaces import (
    E2BWorkspace,
    LocalWorkspace,
    MicrosandboxWorkspace,
    RunnerWorkspace,
)


def test_git_secondary_bounds_never_split_redaction_marker() -> None:
    content = "prefix-" + "[REDACTED_SECRET]" + "-suffix"

    bounded, truncated = git_module._bounded_text(content, len("prefix-[REDA"))

    assert truncated is True
    assert "[REDA" not in bounded
    assert len(bounded.encode()) <= len("prefix-[REDA")


def test_git_diff_offset_rejects_position_inside_redaction_marker() -> None:
    content = "prefix-" + "[REDACTED_SECRET]" + "-suffix"

    with pytest.raises(ValueError, match="splits a redaction marker"):
        git_module._page_diff_text(
            content,
            offset=len("prefix-[REDA"),
            maximum=128,
            capture_truncated=False,
        )


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "Cayu Tests")
    (root / "tracked.txt").write_text("before\n")
    (root / "rename-me.txt").write_text("rename\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")


def test_git_changes_status_is_structured_and_pageable(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "tracked.txt").write_text("after\n")
    (tmp_path / "staged.txt").write_text("staged\n")
    (tmp_path / "untracked.txt").write_text("untracked\n")
    _git(tmp_path, "add", "staged.txt")
    _git(tmp_path, "mv", "rename-me.txt", "renamed.txt")
    ctx = ToolContext(session_id="session", runner=LocalRunner(tmp_path))

    first = asyncio.run(GitChangesTool().run(ctx, {"mode": "status", "limit": 2}))
    second = asyncio.run(
        GitChangesTool().run(
            ctx,
            {
                "mode": "status",
                "offset": first.structured["next_offset"],
                "limit": 10,
            },
        )
    )

    changes = [*first.structured["changes"], *second.structured["changes"]]
    assert first.is_error is False
    assert first.structured["truncated"] is True
    assert first.structured["truncation_reasons"] == ["limit"]
    assert {change["path"] for change in changes} == {
        "renamed.txt",
        "staged.txt",
        "tracked.txt",
        "untracked.txt",
    }
    renamed = next(change for change in changes if change["path"] == "renamed.txt")
    assert renamed["original_path"] == "rename-me.txt"


def test_git_changes_summary_and_diff_are_bounded(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "tracked.txt").write_text(
        "".join(f"changed line {index}\n" for index in range(500))
    )
    ctx = ToolContext(session_id="session", runner=LocalRunner(tmp_path))

    summary = asyncio.run(GitChangesTool().run(ctx, {"mode": "summary"}))
    diff = asyncio.run(
        GitChangesTool().run(
            ctx,
            {"mode": "diff", "max_result_bytes": 1024},
        )
    )

    assert summary.is_error is False
    assert summary.structured["changes"][0]["path"] == "tracked.txt"
    assert summary.structured["changes"][0]["additions"] == 500
    assert summary.structured["changes"][0]["deletions"] == 1
    assert len(diff.content.encode()) <= 1024
    assert diff.content.endswith("[git changes truncated]")
    assert "capture_limit" in diff.structured["truncation_reasons"]


def test_git_changes_reports_non_repository_and_rejects_escaping_paths(
    tmp_path: Path,
) -> None:
    ctx = ToolContext(session_id="session", runner=LocalRunner(tmp_path))

    not_repository = asyncio.run(GitChangesTool().run(ctx, {}))
    invalid_path = asyncio.run(GitChangesTool().run(ctx, {"paths": ["../outside"]}))

    assert not_repository.is_error is True
    assert not_repository.structured["error"] == "git_command_failed"
    assert invalid_path.is_error is True
    assert invalid_path.structured == {"error": "invalid_arguments"}


def test_git_changes_diff_does_not_fall_back_to_all_paths_for_an_empty_page(
    tmp_path: Path,
) -> None:
    _repository(tmp_path)
    (tmp_path / "tracked.txt").write_text("secret change\n")
    (tmp_path / "untracked.txt").write_text("untracked\n")
    ctx = ToolContext(session_id="session", runner=LocalRunner(tmp_path))

    past_end = asyncio.run(GitChangesTool().run(ctx, {"mode": "diff", "offset": 100}))
    untracked_only = asyncio.run(
        GitChangesTool().run(
            ctx,
            {"mode": "diff", "paths": ["untracked.txt"]},
        )
    )

    assert "secret change" not in past_end.content
    assert "secret change" not in untracked_only.content
    assert untracked_only.content == "Untracked content omitted:\nuntracked.txt"


def test_git_changes_bounds_structured_status_entries(tmp_path: Path) -> None:
    _repository(tmp_path)
    for index in range(80):
        (tmp_path / f"untracked-{index:03d}-{'x' * 40}.txt").write_text("x")
    ctx = ToolContext(session_id="session", runner=LocalRunner(tmp_path))

    result = asyncio.run(
        GitChangesTool().run(
            ctx,
            {
                "mode": "status",
                "limit": 200,
                "max_result_bytes": 1024,
            },
        )
    )

    assert len(json.dumps(result.structured, separators=(",", ":")).encode()) <= 1024
    assert "structured_result_bytes" in result.structured["truncation_reasons"]
    assert result.structured["next_offset"] == result.structured["returned"]


def test_git_changes_uses_the_bound_workspace_cwd(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    (repository / "tracked.txt").write_text("workspace change\n")
    runner = LocalRunner(tmp_path)
    workspace = RunnerWorkspace(runner, cwd="repo")
    ctx = ToolContext(session_id="session", runner=runner, workspace=workspace)

    result = asyncio.run(GitChangesTool().run(ctx, {}))

    assert result.is_error is False
    assert [change["path"] for change in result.structured["changes"]] == ["tracked.txt"]


@pytest.mark.parametrize("workspace_kind", ["local", "runner"])
def test_git_changes_uses_authenticated_invocation_workspace_cwd(
    tmp_path: Path,
    workspace_kind: str,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    (repository / "tracked.txt").write_text("workspace change\n")
    runner = LocalRunner(tmp_path)
    workspace = (
        LocalWorkspace(repository, workspace_id="local")
        if workspace_kind == "local"
        else RunnerWorkspace(runner, cwd="repo")
    )
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )
    ctx = ToolContext(session_id="session", runner=handle, workspace=workspace)

    result = asyncio.run(GitChangesTool().run(ctx, {}))

    assert result.is_error is False
    assert [change["path"] for change in result.structured["changes"]] == ["tracked.txt"]


def test_git_changes_invocation_handle_rejects_mismatched_workspace(tmp_path: Path) -> None:
    runner_root = tmp_path / "runner"
    workspace_root = tmp_path / "workspace"
    runner_root.mkdir()
    workspace_root.mkdir()
    handle = InvocationRunnerHandle(
        LocalRunner(runner_root),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )
    ctx = ToolContext(
        session_id="session",
        runner=handle,
        workspace=LocalWorkspace(workspace_root, workspace_id="local"),
    )

    result = asyncio.run(GitChangesTool().run(ctx, {}))

    assert result.is_error is True
    assert result.structured == {"error": "workspace_runner_mismatch"}


@pytest.mark.parametrize(
    ("workspace_type", "root"),
    [
        (E2BWorkspace, "/home/user/workspace"),
        (MicrosandboxWorkspace, "/workspace"),
    ],
)
def test_git_changes_invocation_handle_authenticates_remote_workspace_binding(
    tmp_path: Path,
    workspace_type: type[E2BWorkspace] | type[MicrosandboxWorkspace],
    root: str,
) -> None:
    runner = LocalRunner(tmp_path)
    workspace = object.__new__(workspace_type)
    workspace._runner = runner
    workspace.root = root
    handle = InvocationRunnerHandle(
        runner,
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0,
            redactor=SecretRedactor(),
        ),
    )

    cwd = _workspace_cwd(
        ToolContext(
            session_id="session",
            runner=handle,
            workspace=workspace,
        )
    )

    assert cwd == root


def test_git_changes_refuses_upward_repository_discovery(tmp_path: Path) -> None:
    _repository(tmp_path)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (tmp_path / "sibling.txt").write_text("outside\n")
    runner = LocalRunner(workspace_root)
    ctx = ToolContext(session_id="session", runner=runner)

    result = asyncio.run(GitChangesTool().run(ctx, {}))

    assert result.is_error is True
    assert "sibling.txt" not in result.content


def test_git_changes_clears_inherited_repository_selection_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    external_root = tmp_path / "external"
    workspace_root.mkdir()
    external_root.mkdir()
    _repository(workspace_root)
    _repository(external_root)
    (workspace_root / "tracked.txt").write_text("inside\n")
    (external_root / "tracked.txt").write_text("outside\n")
    monkeypatch.setenv("GIT_DIR", str(external_root / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(external_root))
    ctx = ToolContext(
        session_id="session",
        runner=LocalRunner(workspace_root, inherit_env=True),
    )

    result = asyncio.run(GitChangesTool().run(ctx, {"mode": "diff"}))

    assert result.is_error is False
    assert "inside" in result.content
    assert "outside" not in result.content


@pytest.mark.skipif(os.name == "nt", reason="requires a POSIX executable hook")
def test_git_changes_disables_repository_configured_executables(tmp_path: Path) -> None:
    _repository(tmp_path)
    marker = tmp_path / "executed"
    helper = tmp_path / "helper.sh"
    helper.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    helper.chmod(0o755)
    _git(tmp_path, "config", "core.fsmonitor", str(helper))
    _git(tmp_path, "config", "filter.evil.clean", str(helper))
    (tmp_path / ".gitattributes").write_text("tracked.txt filter=evil\n")
    (tmp_path / "tracked.txt").write_text("change\n")
    ctx = ToolContext(session_id="session", runner=LocalRunner(tmp_path))

    result = asyncio.run(GitChangesTool().run(ctx, {"mode": "diff"}))

    assert result.is_error is False
    assert not marker.exists()


@pytest.mark.skipif(os.name == "nt", reason="requires a POSIX executable hook")
def test_git_changes_ignores_global_filter_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    repository = tmp_path / "repo"
    home.mkdir()
    repository.mkdir()
    monkeypatch.setenv("HOME", str(home))
    _repository(repository)
    marker = tmp_path / "global-filter-executed"
    helper = tmp_path / "global-helper.sh"
    helper.write_text(f"#!/bin/sh\ntouch {marker}\ncat\n")
    helper.chmod(0o755)
    _git(repository, "config", "--global", "filter.evil.clean", str(helper))
    (repository / ".gitattributes").write_text("tracked.txt filter=evil\n")
    (repository / "tracked.txt").write_text("change\n")
    ctx = ToolContext(
        session_id="session",
        runner=LocalRunner(repository, inherit_env=True),
    )

    result = asyncio.run(GitChangesTool().run(ctx, {"mode": "diff"}))

    assert result.is_error is False
    assert not marker.exists()


def test_git_changes_numstat_is_filename_safe_and_distinguishes_count_kinds(
    tmp_path: Path,
) -> None:
    _repository(tmp_path)
    tabbed = tmp_path / "tab\tname.txt"
    tabbed.write_text("before\n")
    _git(tmp_path, "add", "tab\tname.txt")
    _git(tmp_path, "commit", "-qm", "tabbed")
    tabbed.write_text("after\n")
    (tmp_path / "untracked.txt").write_text("new\n")
    ctx = ToolContext(session_id="session", runner=LocalRunner(tmp_path))

    result = asyncio.run(GitChangesTool().run(ctx, {"mode": "summary"}))

    changes = {change["path"]: change for change in result.structured["changes"]}
    assert changes["tab\tname.txt"]["count_kind"] == "text"
    assert changes["tab\tname.txt"]["additions"] == 1
    assert changes["tab\tname.txt"]["deletions"] == 1
    assert changes["untracked.txt"]["count_kind"] == "untracked"


def test_git_changes_diff_has_deterministic_byte_continuation(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "tracked.txt").write_text("".join(f"line {index}\n" for index in range(1000)))
    ctx = ToolContext(session_id="session", runner=LocalRunner(tmp_path))

    first = asyncio.run(GitChangesTool().run(ctx, {"mode": "diff", "max_result_bytes": 1024}))
    second = asyncio.run(
        GitChangesTool().run(
            ctx,
            {
                "mode": "diff",
                "max_result_bytes": 1024,
                "diff_offset": first.structured["next_diff_offset"],
            },
        )
    )

    assert first.structured["next_diff_offset"] is not None
    assert second.structured["diff_offset"] == first.structured["next_diff_offset"]
    assert second.content != first.content


def test_git_changes_omits_forced_text_binary_diff(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / ".gitattributes").write_text("*.bin diff\n")
    binary = tmp_path / "payload.bin"
    binary.write_bytes(b"before\x00data")
    _git(tmp_path, "add", ".gitattributes", "payload.bin")
    _git(tmp_path, "commit", "-qm", "binary")
    binary.write_bytes(b"after\x00data")
    ctx = ToolContext(session_id="session", runner=LocalRunner(tmp_path))

    result = asyncio.run(GitChangesTool().run(ctx, {"mode": "diff"}))

    assert result.is_error is False
    assert "\0" not in result.content
    assert result.structured["binary_omitted"] is True
