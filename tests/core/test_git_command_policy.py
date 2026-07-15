from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    ExecCommand,
    ExecCommandTool,
    GitCommandPolicy,
    LocalRunner,
    Message,
    ModelStreamEvent,
    ProcessCommandPolicy,
    RunRequest,
    ScriptedModelProvider,
)
from cayu.core.tools import ToolContext
from cayu.tools import GitCommandPolicy as ToolsGitCommandPolicy

_SAFE_PREFIX = ("git", "--no-pager", "-c", "core.fsmonitor=false")
_SAFE_INSPECTION_PREFIX = (
    *_SAFE_PREFIX,
    "-c",
    "log.showSignature=false",
)
_SAFE_COMMIT_PREFIX = (
    *_SAFE_PREFIX,
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "commit.gpgSign=false",
)


def _request(
    *argv: str,
    canonical_cwd: str | None = "/workspace/repo",
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> CommandRequest:
    return CommandRequest(
        command=ExecCommand.process(*argv),
        cwd=None,
        canonical_cwd=canonical_cwd,
        env=env,
        timeout_s=30,
        stdin=stdin,
    )


def _policy() -> GitCommandPolicy:
    return GitCommandPolicy(
        process_policy=ProcessCommandPolicy(
            allowed_executables={"git", "python"},
            allowed_cwds={"/workspace"},
            allowed_env_names={"LANG", "GIT_CONFIG_COUNT"},
            allow_stdin=True,
            max_timeout_s=30,
        ),
        git_executables={"git"},
        allowed_repositories={"/workspace/repo"},
    )


def _evaluate(policy: GitCommandPolicy, request: CommandRequest):
    return asyncio.run(policy.evaluate(ToolContext(session_id="sess_1"), request))


class _RecordingGitPolicy(CommandPolicy):
    def __init__(self, delegate: GitCommandPolicy) -> None:
        self.delegate = delegate
        self.requests: list[CommandRequest] = []

    async def evaluate(
        self,
        ctx: ToolContext,
        request: CommandRequest,
    ) -> CommandPolicyResult:
        self.requests.append(request)
        return await self.delegate.evaluate(ctx, request)


def test_git_command_policy_is_exported_from_public_surfaces() -> None:
    assert ToolsGitCommandPolicy is GitCommandPolicy


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"process_policy": object()}, TypeError, "ProcessCommandPolicy"),
        ({"git_executables": "git"}, TypeError, "iterable of strings"),
        ({"git_executables": set()}, ValueError, "at least one"),
        ({"allowed_repositories": "repo"}, TypeError, "iterable of strings"),
        ({"allowed_repositories": set()}, ValueError, "at least one"),
        ({"allowed_repositories": {"repo"}}, ValueError, "absolute POSIX"),
        (
            {"allowed_repositories": {"/workspace/repo/../other"}},
            ValueError,
            "normalized POSIX",
        ),
        ({"max_commit_message_bytes": True}, TypeError, "must be an integer"),
        ({"max_commit_message_bytes": 0}, ValueError, "greater than zero"),
    ],
)
def test_git_command_policy_rejects_invalid_host_configuration(
    kwargs: dict,
    error: type[Exception],
    message: str,
) -> None:
    defaults: dict[str, Any] = {
        "process_policy": ProcessCommandPolicy(),
        "allowed_repositories": {"/workspace/repo"},
    }
    defaults.update(kwargs)
    with pytest.raises(error, match=message):
        GitCommandPolicy(**defaults)


def test_git_command_policy_allows_documented_status_shape() -> None:
    result = _evaluate(
        _policy(),
        _request(*_SAFE_PREFIX, "status", "--short", "--branch"),
    )

    assert result.decision is CommandPolicyDecision.ALLOW
    assert result.reason is None


@pytest.mark.parametrize(
    "argv",
    [
        (*_SAFE_PREFIX, "status", "--short", "--", "nested/file.py"),
        (*_SAFE_PREFIX, "status", "--", "-leading-dash"),
        (*_SAFE_PREFIX, "ls-files"),
        (*_SAFE_PREFIX, "ls-files", "--cached", "--modified", "--deleted"),
        (*_SAFE_PREFIX, "ls-files", "--others", "--exclude-standard", "--", "src"),
        (*_SAFE_PREFIX, "rev-parse", "--show-toplevel"),
        (*_SAFE_PREFIX, "rev-parse", "--show-prefix"),
        (*_SAFE_PREFIX, "rev-parse", "--is-inside-work-tree"),
        (*_SAFE_PREFIX, "rev-parse", "--git-dir"),
        (*_SAFE_PREFIX, "rev-parse", "--verify", "HEAD~1"),
        (*_SAFE_PREFIX, "branch", "--show-current"),
        (*_SAFE_INSPECTION_PREFIX, "log", "--oneline", "--max-count=20", "HEAD"),
        (
            *_SAFE_INSPECTION_PREFIX,
            "log",
            "--oneline",
            "--decorate=short",
            "main..HEAD",
            "--",
            "src/app.py",
        ),
        (
            *_SAFE_INSPECTION_PREFIX,
            "show",
            "--format=medium",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
        ),
        (
            *_SAFE_INSPECTION_PREFIX,
            "show",
            "--format=medium",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD:src/app.py",
        ),
        (*_SAFE_PREFIX, "cat-file", "-t", "HEAD"),
        (*_SAFE_PREFIX, "cat-file", "-p", "HEAD:src/app.py"),
        (*_SAFE_PREFIX, "diff", "--no-ext-diff", "--no-textconv"),
        (
            *_SAFE_PREFIX,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            "--stat",
            "--",
            "src/app.py",
        ),
        (*_SAFE_PREFIX, "add", "--", "file.txt"),
        (*_SAFE_PREFIX, "add", "--", "nested/file.py", "name with spaces.txt"),
        (*_SAFE_PREFIX, "add", "--", "-leading-dash"),
        (
            *_SAFE_COMMIT_PREFIX,
            "commit",
            "--no-verify",
            "--no-gpg-sign",
            "-m",
            "ordinary local commit",
        ),
    ],
)
def test_git_command_policy_allows_only_documented_local_workflow_matrix(
    argv: tuple[str, ...],
) -> None:
    result = _evaluate(_policy(), _request(*argv))

    assert result.decision is CommandPolicyDecision.ALLOW
    assert result.reason is None


def test_git_command_policy_preserves_general_process_denial() -> None:
    result = _evaluate(_policy(), _request("/usr/bin/git", "status"))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Executable is not allowed by the process policy."


def test_git_command_policy_preserves_general_process_approval_requirement() -> None:
    policy = GitCommandPolicy(
        process_policy=ProcessCommandPolicy(
            approval_required_executables={"git"},
            allowed_cwds={"/workspace"},
        ),
        allowed_repositories={"/workspace/repo"},
    )

    result = _evaluate(policy, _request(*_SAFE_PREFIX, "status"))

    assert result.decision is CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL
    assert result.reason == "Executable requires command approval by the process policy."


def test_git_command_policy_denies_non_git_process_allowed_by_general_policy() -> None:
    result = _evaluate(_policy(), _request("python", "script.py"))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Executable is not a configured Git executable."


def test_git_command_policy_denies_shell_even_when_general_policy_allows_it() -> None:
    policy = GitCommandPolicy(
        process_policy=ProcessCommandPolicy(
            allowed_cwds={"/workspace"},
            shell_decision=CommandPolicyDecision.ALLOW,
        ),
        allowed_repositories={"/workspace/repo"},
    )
    request = _request("git").model_copy(update={"command": ExecCommand.bash("git status")})

    result = _evaluate(policy, request)

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Executable is not a configured Git executable."


@pytest.mark.parametrize(
    "argv",
    [
        ("git", "status"),
        ("git", "--no-p", "-c", "core.fsmonitor=false", "status"),
        ("git", "--no-pager", "status"),
        ("git", "--no-pager", "-c", "alias.status=!sh", "status"),
        ("git", "--no-pager", "--git-dir", "/tmp/repo", "status"),
        ("git", "--no-pager", "--work-tree=/tmp", "status"),
        ("git", "--no-pager", "--namespace", "other", "status"),
        ("git", "--no-pager", "--exec-path=/tmp", "status"),
        ("git", "--no-pager", "--bare", "status"),
        ("git", "--no-pager", "--config-env=x=y", "status"),
        ("git", "--no-pager", "-C"),
        ("git", "-C/workspace/repo", *_SAFE_PREFIX[1:], "status"),
        ("git", "--no-pager", "-ccore.fsmonitor=false", "status"),
        (*_SAFE_PREFIX, "--", "status"),
    ],
)
def test_git_command_policy_rejects_unsupported_or_incomplete_global_options(
    argv: tuple[str, ...],
) -> None:
    result = _evaluate(_policy(), _request(*argv))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason in {
        "Git invocation is missing required safety controls.",
        "Git global option is not allowed.",
        "Git global option is missing its operand.",
    }


@pytest.mark.parametrize(
    "argv",
    [
        (*_SAFE_PREFIX, "push"),
        (*_SAFE_PREFIX, "credential", "fill"),
        (*_SAFE_PREFIX, "reset", "--hard"),
        (*_SAFE_PREFIX, "status", "--ignored"),
    ],
)
def test_git_command_policy_rejects_unsupported_subcommands_and_options(
    argv: tuple[str, ...],
) -> None:
    result = _evaluate(_policy(), _request(*argv))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason in {
        "Git subcommand is not supported by the Git policy.",
        "Git option is not allowed for this subcommand.",
    }


@pytest.mark.parametrize(
    "subcommand",
    [
        "push",
        "pull",
        "fetch",
        "clone",
        "remote",
        "submodule",
        "credential",
        "credential-fill",
        "upload-pack",
        "receive-pack",
        "reset",
        "clean",
        "checkout",
        "switch",
        "restore",
        "rebase",
        "merge",
        "cherry-pick",
        "revert",
        "replace",
        "notes",
        "tag",
        "worktree",
        "reflog",
        "gc",
        "repack",
        "maintenance",
        "config",
        "alias-from-config",
    ],
)
def test_git_command_policy_denies_remote_credential_destructive_and_helper_commands(
    subcommand: str,
) -> None:
    result = _evaluate(_policy(), _request(*_SAFE_PREFIX, subcommand))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Git subcommand is not supported by the Git policy."


@pytest.mark.parametrize(
    "path",
    [
        ".",
        "..",
        "../escape",
        "nested/../../escape",
        "/workspace/repo/file.py",
        ".git/config",
        ".GIT/config",
        "nested/.git/hooks/pre-commit",
        ":(glob)**/*.py",
        ":magic",
        "*.py",
        "file?.py",
        "nested//file.py",
        "a" * 4097,
        "nul\0path",
    ],
)
def test_git_command_policy_rejects_unsafe_or_ambiguous_paths(path: str) -> None:
    result = _evaluate(_policy(), _request(*_SAFE_PREFIX, "add", "--", path))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Git path is not allowed by the Git policy."


def test_git_command_policy_empty_path_is_rejected_by_process_command_shape() -> None:
    with pytest.raises(ValueError, match="argv entries must be non-empty"):
        _request(*_SAFE_PREFIX, "add", "--", "")


def test_git_command_policy_bounds_number_of_explicit_paths() -> None:
    result = _evaluate(
        _policy(),
        _request(*_SAFE_PREFIX, "add", "--", *(f"file-{index}" for index in range(257))),
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Git path is not allowed by the Git policy."


@pytest.mark.parametrize(
    "argv",
    [
        (*_SAFE_PREFIX, "add"),
        (*_SAFE_PREFIX, "add", "file.py"),
        (*_SAFE_PREFIX, "add", "--all"),
        (*_SAFE_PREFIX, "add", "-A"),
        (*_SAFE_PREFIX, "add", "--update"),
        (*_SAFE_PREFIX, "add", "--pathspec-from-file=paths"),
        (*_SAFE_PREFIX, "add", "--pathspec-file-nul"),
        (*_SAFE_PREFIX, "diff", "--no-ext-diff"),
        (*_SAFE_PREFIX, "diff", "--no-textconv"),
        (*_SAFE_PREFIX, "diff", "--ext-diff", "--no-textconv"),
        (*_SAFE_PREFIX, "diff", "--no-ext-diff", "--textconv"),
        (*_SAFE_PREFIX, "status", "--ignored"),
        (*_SAFE_INSPECTION_PREFIX, "log", "--oneline", "--max-count=1001"),
        (*_SAFE_INSPECTION_PREFIX, "log", "--oneline", "-p"),
        (*_SAFE_INSPECTION_PREFIX, "show", "HEAD"),
        (*_SAFE_PREFIX, "cat-file", "--filters", "HEAD:file"),
    ],
)
def test_git_command_policy_denies_options_outside_supported_matrix(
    argv: tuple[str, ...],
) -> None:
    result = _evaluate(_policy(), _request(*argv))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason in {
        "Git option is not allowed for this subcommand.",
        "Git path is not allowed by the Git policy.",
        "Git invocation is missing required safety controls.",
    }


@pytest.mark.parametrize(
    "argv",
    [
        (
            *_SAFE_COMMIT_PREFIX,
            "commit",
            "--no-verify",
            "--no-gpg-sign",
            "--amend",
            "-m",
            "message",
        ),
        (*_SAFE_COMMIT_PREFIX, "commit", "--no-verify", "-m", "message"),
        (*_SAFE_COMMIT_PREFIX, "commit", "--no-gpg-sign", "-m", "message"),
        (*_SAFE_COMMIT_PREFIX, "commit", "--no-verify", "--no-gpg-sign"),
        (
            *_SAFE_COMMIT_PREFIX,
            "commit",
            "--no-verify",
            "--no-gpg-sign",
            "--fixup=HEAD",
            "-m",
            "message",
        ),
        (
            *_SAFE_COMMIT_PREFIX,
            "commit",
            "--no-verify",
            "--no-gpg-sign",
            "-F",
            "message.txt",
        ),
        (
            *_SAFE_PREFIX,
            "commit",
            "--no-verify",
            "--no-gpg-sign",
            "-m",
            "missing safe config",
        ),
    ],
)
def test_git_command_policy_denies_history_rewrite_or_unsafe_commit_shapes(
    argv: tuple[str, ...],
) -> None:
    result = _evaluate(_policy(), _request(*argv))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Git commit shape is not allowed by the Git policy."


def test_git_command_policy_bounds_commit_message_without_disclosing_it() -> None:
    secret = "secret-message"
    policy = GitCommandPolicy(
        process_policy=ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
        ),
        allowed_repositories={"/workspace/repo"},
        max_commit_message_bytes=5,
    )
    result = _evaluate(
        policy,
        _request(
            *_SAFE_COMMIT_PREFIX,
            "commit",
            "--no-verify",
            "--no-gpg-sign",
            "-m",
            secret,
        ),
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Git commit message exceeds the Git policy byte limit."
    assert secret not in result.reason


def test_git_command_policy_completes_local_workflow_and_blocks_seeded_bypasses(
    tmp_path: Path,
) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required for the real repository integration.")
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(git, repository, "init", "-b", "main")
    _git(git, repository, "config", "user.name", "Cayu Test")
    _git(git, repository, "config", "user.email", "cayu@example.invalid")
    (repository / "tracked.txt").write_text("initial\n", encoding="utf-8")
    (repository / "deleted.txt").write_text("delete me\n", encoding="utf-8")
    (repository / "danger.filtered").write_text("initial filtered content\n", encoding="utf-8")
    (repository / ".gitattributes").write_text(
        "*.filtered diff=marker\n",
        encoding="utf-8",
    )
    _git(
        git,
        repository,
        "add",
        "--",
        "tracked.txt",
        "deleted.txt",
        "danger.filtered",
        ".gitattributes",
    )
    _git(git, repository, "commit", "-m", "initial")
    initial_head = _git(git, repository, "rev-parse", "HEAD").strip()

    markers = {
        name: tmp_path / f"{name}-ran"
        for name in (
            "alias",
            "credential",
            "diff",
            "editor",
            "fsmonitor",
            "hook",
            "pager",
            "remote",
            "signing",
            "textconv",
        )
    }
    scripts = {name: _marker_script(tmp_path, name, marker) for name, marker in markers.items()}
    hook = repository / ".git" / "hooks" / "post-commit"
    hook.write_text(scripts["hook"].read_text(encoding="utf-8"), encoding="utf-8")
    hook.chmod(0o755)
    _git(git, repository, "config", "alias.sneaky", f"!{scripts['alias']}")
    _git(
        git,
        repository,
        "config",
        "credential.helper",
        f"!{scripts['credential']}",
    )
    _git(git, repository, "config", "diff.external", str(scripts["diff"]))
    _git(git, repository, "config", "core.pager", str(scripts["pager"]))
    _git(git, repository, "config", "core.editor", str(scripts["editor"]))
    _git(git, repository, "config", "core.fsmonitor", str(scripts["fsmonitor"]))
    _git(git, repository, "config", "gpg.program", str(scripts["signing"]))
    _git(git, repository, "config", "commit.gpgSign", "true")
    _git(git, repository, "config", "log.showSignature", "true")
    _git(git, repository, "config", "diff.marker.textconv", str(scripts["textconv"]))
    _git(git, repository, "config", "protocol.ext.allow", "always")
    remote_url = f"ext::{scripts['remote']}"
    _git(git, repository, "remote", "add", "origin", remote_url)
    (repository / "tracked.txt").write_text("modified but unstaged\n", encoding="utf-8")
    (repository / "deleted.txt").unlink()
    (repository / "new.txt").write_text("new content\n", encoding="utf-8")

    safe = (git, "--no-pager", "-c", "core.fsmonitor=false")
    safe_inspection = (*safe, "-c", "log.showSignature=false")
    safe_commit = (
        *safe,
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "commit.gpgSign=false",
    )
    commands = [
        (*safe, "rev-parse", "--show-toplevel"),
        (*safe, "status", "--porcelain=v1"),
        (*safe, "ls-files", "--cached"),
        (
            *safe,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--",
            "tracked.txt",
        ),
        (*safe, "add", "--", ":(glob)*.filtered"),
        (*safe, "sneaky"),
        (*safe, "credential", "fill"),
        (*safe, "add", "--", "new.txt", "deleted.txt"),
        (
            *safe,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            "--",
            "new.txt",
            "deleted.txt",
        ),
        (*safe, "push", "origin", "main"),
        (
            *safe_commit,
            "commit",
            "--no-verify",
            "--no-gpg-sign",
            "-m",
            "add new file",
        ),
        (*safe, "status", "--short"),
        (*safe_inspection, "log", "--oneline", "--max-count=2", "HEAD"),
        (
            *safe_inspection,
            "show",
            "--format=medium",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD:new.txt",
        ),
    ]
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id=f"git_call_{index}",
                    name="exec_command",
                    arguments={"argv": list(command), "cwd": "repo"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
            for index, command in enumerate(commands)
        ]
        + [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
    )
    policy = _RecordingGitPolicy(
        GitCommandPolicy(
            process_policy=ProcessCommandPolicy(
                allowed_executables={git},
                allowed_cwds={str(tmp_path)},
                max_timeout_s=60,
            ),
            git_executables={git},
            allowed_repositories={str(repository)},
        )
    )
    app = CayuApp()
    runner = LocalRunner(tmp_path)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-git"), runner=runner),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="coding-agent", model="scripted-model"),
        tools=[ExecCommandTool(policy=policy)],
    )
    assert all(not marker.exists() for marker in markers.values())

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                agent_name="coding-agent",
                messages=[Message.text("user", "complete the local Git workflow")],
                max_steps=16,
            ),
        )
    )

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert [(request.cwd, request.canonical_cwd) for request in policy.requests] == [
        ("repo", str(repository))
    ] * len(commands)
    failed = [event for event in events if event.type is EventType.TOOL_CALL_FAILED]
    assert len(failed) == 4
    assert [event.payload["result"]["structured"]["reason"] for event in failed] == [
        "Git path is not allowed by the Git policy.",
        "Git subcommand is not supported by the Git policy.",
        "Git subcommand is not supported by the Git policy.",
        "Git subcommand is not supported by the Git policy.",
    ]
    denial_event = json.dumps(failed[0].model_dump(mode="json"), sort_keys=True)
    assert ":(glob)*.filtered" not in denial_event
    assert all(not marker.exists() for marker in markers.values())
    verify = ("--no-pager", "-c", "core.fsmonitor=false")
    assert _git(git, repository, *verify, "rev-parse", "HEAD^").strip() == initial_head
    assert (
        _git(
            git,
            repository,
            *verify,
            "show",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD:new.txt",
        )
        == "new content\n"
    )
    assert _git(
        git,
        repository,
        *verify,
        "show",
        "--no-ext-diff",
        "--no-textconv",
        "--format=",
        "--name-status",
        "HEAD",
    ) == ("D\tdeleted.txt\nA\tnew.txt\n")
    assert (
        _git(
            git,
            repository,
            *verify,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--cached",
            "--name-only",
        )
        == ""
    )
    assert _git(
        git,
        repository,
        *verify,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--name-only",
        "--",
        "tracked.txt",
    ) == ("tracked.txt\n")
    assert (
        _git(
            git,
            repository,
            *verify,
            "config",
            "--get",
            "remote.origin.url",
        ).strip()
        == remote_url
    )

    denial_tool = ExecCommandTool(policy=policy)
    denial_ctx = ToolContext(session_id="git_denial_state_proof", runner=runner)
    denied_commands = [
        (*safe, "add", "--", ":(glob)*.filtered"),
        (*safe, "sneaky"),
        (*safe, "credential", "fill"),
        (*safe, "push", "origin", "main"),
        (*safe, "reset", "--hard"),
        (git, "--no-pager", "-c", "alias.status=!false", "status"),
    ]
    for command in denied_commands:
        before = _repository_state(repository, markers.values())
        result = asyncio.run(
            denial_tool.run(
                denial_ctx,
                {"argv": list(command), "cwd": "repo"},
            )
        )
        assert result.is_error is True
        assert _repository_state(repository, markers.values()) == before


def _git(git: str, repository: Path, *args: str) -> str:
    completed = subprocess.run(
        [git, *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _repository_state(
    repository: Path,
    markers: Any,
) -> tuple[tuple[tuple[str, bytes], ...], tuple[bool, ...]]:
    files = tuple(
        (path.relative_to(repository).as_posix(), path.read_bytes())
        for path in sorted(repository.rglob("*"))
        if path.is_file()
    )
    return files, tuple(marker.exists() for marker in markers)


def _marker_script(tmp_path: Path, name: str, marker: Path) -> Path:
    script = tmp_path / f"{name}-helper.sh"
    script.write_text(
        f'#!/bin/sh\n: > "{marker}"\nexit 1\n',
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


async def _collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


@pytest.mark.parametrize(
    ("argv", "decision"),
    [
        ((*_SAFE_PREFIX[:1], "-C", "nested", *_SAFE_PREFIX[1:], "status"), "allow"),
        (
            (
                *_SAFE_PREFIX[:1],
                "-C",
                "nested",
                "-C",
                "child",
                *_SAFE_PREFIX[1:],
                "status",
            ),
            "allow",
        ),
        ((*_SAFE_PREFIX[:1], "-C", "../escape", *_SAFE_PREFIX[1:], "status"), "deny"),
        ((*_SAFE_PREFIX[:1], "-C", "nested/../other", *_SAFE_PREFIX[1:], "status"), "deny"),
    ],
)
def test_git_command_policy_processes_repeated_c_structurally(
    argv: tuple[str, ...],
    decision: str,
) -> None:
    result = _evaluate(_policy(), _request(*argv))

    assert result.decision == decision
    if decision == "deny":
        assert result.reason == "Git working-directory change is not allowed."


@pytest.mark.parametrize(
    "argv",
    [
        ("git", "--no-pager", "-ccore.fsmonitor=false", "status"),
        ("git", "-C/workspace/repo/nested", *_SAFE_PREFIX[1:], "status"),
        (*_SAFE_PREFIX, "--", "status"),
    ],
)
def test_git_command_policy_rejects_git_unsupported_global_spellings(
    argv: tuple[str, ...],
) -> None:
    result = _evaluate(_policy(), _request(*argv))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Git global option is not allowed."


@pytest.mark.parametrize("subcommand", ["log", "show"])
def test_git_command_policy_requires_signature_suppression_for_object_inspection(
    subcommand: str,
) -> None:
    result = _evaluate(_policy(), _request(*_SAFE_PREFIX, subcommand))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Git global option is not allowed."


def test_git_command_policy_uses_canonical_cwd_not_requested_spelling() -> None:
    request = _request(*_SAFE_PREFIX, "status").model_copy(
        update={"cwd": "../../outside", "canonical_cwd": "/workspace/repo"}
    )

    result = _evaluate(_policy(), request)

    assert result.decision is CommandPolicyDecision.ALLOW


@pytest.mark.parametrize(
    ("env", "stdin", "reason"),
    [
        (
            {"GIT_CONFIG_COUNT": "1"},
            None,
            "Git environment is not allowed by the Git policy.",
        ),
        ({"LANG": "C"}, "secret", "Git stdin is not allowed by the Git policy."),
    ],
)
def test_git_command_policy_tightens_general_environment_and_stdin_controls(
    env: dict[str, str],
    stdin: str | None,
    reason: str,
) -> None:
    result = _evaluate(
        _policy(),
        _request(*_SAFE_PREFIX, "status", env=env, stdin=stdin),
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == reason


@pytest.mark.parametrize(
    "name",
    [
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_PAGER",
        "GIT_EDITOR",
        "GIT_SSH_COMMAND",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "PAGER",
        "EDITOR",
        "VISUAL",
        "HOME",
        "XDG_CONFIG_HOME",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ],
)
def test_git_command_policy_rejects_git_helper_config_and_proxy_environment(
    name: str,
) -> None:
    secret = "do-not-persist"
    policy = GitCommandPolicy(
        process_policy=ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
            allowed_env_names={name},
        ),
        allowed_repositories={"/workspace/repo"},
    )

    result = _evaluate(
        policy,
        _request(*_SAFE_PREFIX, "status", env={name: secret}),
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Git environment is not allowed by the Git policy."
    assert secret not in result.reason
