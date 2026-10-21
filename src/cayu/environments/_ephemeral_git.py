"""An ephemeral Git baseline of a coding sandbox's workspace, made in the guest.

Coding factories copy a source tree into the guest without its ``.git``. Right
after copy-in they commit the copy to a fresh local repository, so the guest's
own ``git`` (``git_changes`` among others) sees what the session changed. The
repository never leaves the guest: ``.git`` is a protected directory that no
copy crosses. Its configuration turns off hooks, filters, pagers, credentials,
signing and every transport, and is checked before the baseline is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from cayu.runners.base import ExecCommand, Runner

__all__ = [
    "GIT_ENV_REMOVE",
    "EphemeralGitBaseline",
    "ephemeral_git_baseline",
    "git_filter_configuration_is_safe",
    "initialize_ephemeral_git_baseline",
]

# Variables that would point Git at another repository, object store or transport.
GIT_ENV_REMOVE = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_EXTERNAL_DIFF",
    "GIT_INDEX_FILE",
    "GIT_NAMESPACE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_WORK_TREE",
)
_FILTER_OUTPUT_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class EphemeralGitBaseline:
    """The baseline's identity: its commit, index, tracked flags and configuration."""

    head_revision: str
    staged_entries_sha256: str
    tracked_flags_sha256: str
    configuration_sha256: str


async def initialize_ephemeral_git_baseline(runner: Runner, *, cwd: str | None = None) -> None:
    """Commit the workspace at ``cwd`` (the runner's root by default) to a fresh, inert repository.

    No step reads back output that grows with the workspace, so a runner whose
    command output is capped, as a brokered sandbox's is, can baseline any tree
    it holds.
    """

    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_ASKPASS": "/bin/false",
        "GIT_CONFIG_COUNT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "PAGER": "cat",
    }
    commands = (
        ExecCommand.process("rm", "-rf", "--", ".git"),
        ExecCommand.process("git", "init", "-q"),
        ExecCommand.process("git", "config", "--local", "core.hooksPath", "/dev/null"),
        ExecCommand.process("git", "config", "--local", "core.fsmonitor", "false"),
        ExecCommand.process("git", "config", "--local", "core.pager", "cat"),
        ExecCommand.process("git", "config", "--local", "pager.branch", "false"),
        ExecCommand.process("git", "config", "--local", "pager.diff", "false"),
        ExecCommand.process("git", "config", "--local", "pager.status", "false"),
        ExecCommand.process("git", "config", "--local", "credential.helper", "/bin/false"),
        ExecCommand.process("git", "config", "--local", "credential.interactive", "never"),
        ExecCommand.process("git", "config", "--local", "commit.gpgSign", "false"),
        ExecCommand.process("git", "config", "--local", "tag.gpgSign", "false"),
        ExecCommand.process("git", "config", "--local", "protocol.allow", "never"),
        ExecCommand.process("git", "add", "-A", "--", "."),
        ExecCommand.process(
            "git",
            "-c",
            "user.name=Cayu Runtime",
            "-c",
            "user.email=runtime@cayu.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-q",
            "--allow-empty",
            "--no-gpg-sign",
            "--no-verify",
            "-m",
            "Cayu workspace baseline",
        ),
    )
    for command in commands:
        result = await runner.exec_system(
            command,
            cwd=cwd,
            env=environment,
            env_remove=GIT_ENV_REMOVE,
            timeout_s=60,
            output_limit_bytes=64 * 1024,
        )
        if result.exit_code != 0 or result.timed_out:
            raise RuntimeError("The coding sandbox could not establish its ephemeral Git baseline.")
    if not await git_filter_configuration_is_safe(runner, cwd=cwd):
        raise RuntimeError("The coding sandbox's ephemeral Git configuration is unsafe.")


async def ephemeral_git_baseline(runner: Runner) -> EphemeralGitBaseline:
    """Read the identity of the baseline in the runner's workspace.

    Each index listing comes back whole in one command's output, so the
    runner must return output as large as the index.
    """

    async def capture(*argv: str, output_limit_bytes: int = 16 * 1024 * 1024) -> str:
        result = await runner.exec_system(
            ExecCommand.process("git", *argv),
            env={
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_CONFIG_COUNT": "0",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "LC_ALL": "C",
            },
            env_remove=GIT_ENV_REMOVE,
            timeout_s=60,
            output_limit_bytes=output_limit_bytes,
        )
        if (
            result.exit_code != 0
            or result.timed_out
            or result.cancelled
            or result.stdout_truncated
            or result.stderr_truncated
        ):
            raise RuntimeError("The coding sandbox could not verify its ephemeral Git baseline.")
        return result.stdout

    head_revision = (
        await capture("rev-parse", "--verify", "HEAD", output_limit_bytes=1024)
    ).strip()
    staged_entries = await capture("ls-files", "--stage", "-z", "--")
    tracked_flags = await capture("ls-files", "-v", "-z", "--")
    configuration = await capture("config", "--includes", "--null", "--list")
    return EphemeralGitBaseline(
        head_revision=head_revision,
        staged_entries_sha256="sha256:" + sha256(staged_entries.encode("utf-8")).hexdigest(),
        tracked_flags_sha256="sha256:" + sha256(tracked_flags.encode("utf-8")).hexdigest(),
        configuration_sha256="sha256:" + sha256(configuration.encode("utf-8")).hexdigest(),
    )


async def git_filter_configuration_is_safe(runner: Runner, *, cwd: str | None = None) -> bool:
    """True when the repository at ``cwd`` has no include and no clean, smudge or process filter."""

    configured_filters = await runner.exec_system(
        ExecCommand.process(
            "git",
            "config",
            "--includes",
            "--null",
            "--name-only",
            "--get-regexp",
            r"^(include([iI][fF])?\..*|filter\..*\.(clean|smudge|process))$",
        ),
        cwd=cwd,
        env={
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        },
        env_remove=GIT_ENV_REMOVE,
        timeout_s=60,
        output_limit_bytes=_FILTER_OUTPUT_BYTES,
    )
    return not (
        configured_filters.exit_code not in {0, 1}
        or configured_filters.timed_out
        or configured_filters.cancelled
        or configured_filters.stdout_truncated
        or configured_filters.stderr_truncated
        or bool(configured_filters.stdout)
        or configured_filters.exit_code == 0
    )
