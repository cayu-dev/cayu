from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from cayu.runners import LocalRunner
from cayu.tools import ExecCommandTool, GitCommandPolicy, ProcessCommandPolicy
from cayu.tools.base import ToolContext


@pytest.mark.parametrize("subcommand", ["add", "status"])
def test_git_command_policy_does_not_execute_index_hooks(tmp_path: Path, subcommand: str) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required for the real repository integration.")
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run([git, "init", "-b", "main"], cwd=repository, check=True, capture_output=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("initial\n", encoding="utf-8")
    subprocess.run(
        [git, "add", "--", "tracked.txt"], cwd=repository, check=True, capture_output=True
    )
    marker = tmp_path / "index-hook-ran"
    hook = repository / ".git" / "hooks" / "post-index-change"
    hook.write_text(f'#!/bin/sh\n: > "{marker}"\n', encoding="utf-8")
    hook.chmod(0o755)
    # Changing the worktree gives both commands a reason to refresh the index.
    tracked.write_text("modified worktree\n", encoding="utf-8")
    policy = GitCommandPolicy(
        process_policy=ProcessCommandPolicy(
            allowed_executables={git},
            allowed_cwds={str(tmp_path)},
            max_timeout_s=60,
        ),
        git_executables={git},
        allowed_repositories={str(repository)},
    )
    runner = LocalRunner(tmp_path)
    tool = ExecCommandTool(policy=policy)
    argv = [git, "--no-pager", "-c", "core.fsmonitor=false", subcommand]
    argv += ["--", "tracked.txt"] if subcommand == "add" else ["--short"]

    async def run() -> None:
        try:
            await tool.run(
                ToolContext(session_id="index-hook-policy", runner=runner),
                {"argv": argv, "cwd": "repo"},
            )
        finally:
            await runner.close()

    asyncio.run(run())

    assert not marker.exists(), "An authorized Git command executed a repository hook"
