from __future__ import annotations

import asyncio

import pytest

from cayu.runners import ExecCommand
from cayu.tools import (
    CommandPolicyDecision,
    CommandRequest,
    GitCommandPolicy,
    ProcessCommandPolicy,
)
from cayu.tools.base import ToolContext


@pytest.mark.parametrize("value", ["²", "9" * 4301], ids=["unicode-digit", "oversized-integer"])
@pytest.mark.parametrize("subcommand", ["log", "diff"])
def test_git_command_policy_denies_invalid_numeric_options(subcommand: str, value: str) -> None:
    policy = GitCommandPolicy(
        process_policy=ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
            max_timeout_s=30,
        ),
        allowed_repositories={"/workspace/repo"},
    )
    prefix = ["git", "--no-pager", "-c", "core.fsmonitor=false"]
    if subcommand == "log":
        argv = [
            *prefix,
            "-c",
            "log.showSignature=false",
            "log",
            "--oneline",
            f"--max-count={value}",
        ]
    else:
        argv = [
            *prefix,
            "-c",
            "core.hooksPath=/dev/null",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            f"--unified={value}",
        ]
    request = CommandRequest(
        command=ExecCommand.process(*argv),
        canonical_cwd="/workspace/repo",
        timeout_s=30,
    )

    result = asyncio.run(policy.evaluate(ToolContext(session_id="numeric-policy"), request))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Git option is not allowed for this subcommand."
