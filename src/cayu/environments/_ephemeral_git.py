"""An ephemeral Git baseline of a coding sandbox's workspace, made in the guest.

Coding factories copy a source tree into the guest without its ``.git``. Right
after copy-in they commit the copy to a fresh local repository, so the guest's
own ``git`` (``git_changes`` among others) sees what the session changed. The
repository never leaves the guest: ``.git`` is a protected directory that no
copy crosses. Its configuration turns off hooks, filters, pagers, credentials,
signing and every transport, and is checked before the baseline is trusted.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu.runners.base import ExecCommand, Runner

__all__ = [
    "BASELINE_STEP_TIMEOUT_SECONDS",
    "GIT_ENV_REMOVE",
    "MAX_BASELINE_STEP_TIMEOUT_SECONDS",
    "EphemeralGitBaseline",
    "baseline_step_timeout_seconds",
    "ephemeral_git_baseline",
    "git_filter_configuration_is_safe",
    "initialize_ephemeral_git_baseline",
    "sync_bound_tree_size",
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
#: The baseline of a tree of unknown size gets a minute.
BASELINE_STEP_TIMEOUT_SECONDS = 60
#: The baseline of a known tree gets the minute plus this long per MiB and per
#: thousand files, which ``git add`` hashes and compresses one by one, up to the cap.
_BASELINE_SECONDS_PER_MIB = 0.25
_BASELINE_SECONDS_PER_THOUSAND_FILES = 10.0
MAX_BASELINE_STEP_TIMEOUT_SECONDS = 1_800
# Configuration that would run commands or read another file: includes and filters.
_UNSAFE_CONFIGURATION_PATTERN = r"^(include([iI][fF])?\..*|filter\..*\.(clean|smudge|process))$"
_UNSAFE_CONFIGURATION_QUERY = (
    "git",
    "config",
    "--includes",
    "--null",
    "--name-only",
    "--get-regexp",
)
_QUERY_ENVIRONMENT = {"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"}
_BASELINE_STEPS = (
    ("rm", "-rf", "--", ".git"),
    ("git", "init", "-q"),
    ("git", "config", "--local", "core.hooksPath", "/dev/null"),
    ("git", "config", "--local", "core.fsmonitor", "false"),
    ("git", "config", "--local", "core.pager", "cat"),
    ("git", "config", "--local", "pager.branch", "false"),
    ("git", "config", "--local", "pager.diff", "false"),
    ("git", "config", "--local", "pager.status", "false"),
    ("git", "config", "--local", "credential.helper", "/bin/false"),
    ("git", "config", "--local", "credential.interactive", "never"),
    ("git", "config", "--local", "commit.gpgSign", "false"),
    ("git", "config", "--local", "tag.gpgSign", "false"),
    ("git", "config", "--local", "protocol.allow", "never"),
    ("git", "add", "-A", "--", "."),
    (
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
_UNSAFE_CONFIGURATION_EXIT_CODE = 3
# The steps stop at the first failure. The last line exits 3 unless ``git config``
# finds nothing matching $1, so a configuration that names an include or a filter
# fails the baseline apart from a step that fails.
_BASELINE_SCRIPT = (
    " &&\n".join(shlex.join(step) for step in _BASELINE_STEPS)
    + " || exit 1\n"
    + shlex.join(
        [
            *(f"{name}={value}" for name, value in _QUERY_ENVIRONMENT.items()),
            *_UNSAFE_CONFIGURATION_QUERY,
        ]
    )
    + ' "$1" >/dev/null\n'
    + f'[ "$?" -eq 1 ] || exit {_UNSAFE_CONFIGURATION_EXIT_CODE}\n'
)


def baseline_step_timeout_seconds(
    *, tree_bytes: int | None = None, tree_files: int | None = None
) -> int:
    """How long the baseline of a tree of this size may take, all its steps included.

    A 1 GiB tree of 10,000 files gets 60 + 256 + 100 seconds; nothing gets more
    than ``MAX_BASELINE_STEP_TIMEOUT_SECONDS``.
    """

    for name, value in (("tree_bytes", tree_bytes), ("tree_files", tree_files)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a non-negative integer.")
    extra = (tree_bytes or 0) / (1024 * 1024) * _BASELINE_SECONDS_PER_MIB + (
        tree_files or 0
    ) / 1000 * _BASELINE_SECONDS_PER_THOUSAND_FILES
    return min(MAX_BASELINE_STEP_TIMEOUT_SECONDS, BASELINE_STEP_TIMEOUT_SECONDS + int(extra))


def sync_bound_tree_size(metadata: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """The bytes and files a ``SyncBinding`` bind reports it copied in, where it says."""

    sync = metadata.get("sync_binding")
    sync = sync if isinstance(sync, Mapping) else {}
    copied_bytes, copied_files = sync.get("copied_bytes"), sync.get("copied_files")
    return (
        copied_bytes if type(copied_bytes) is int and copied_bytes >= 0 else None,
        copied_files if type(copied_files) is int and copied_files >= 0 else None,
    )


@dataclass(frozen=True, slots=True)
class EphemeralGitBaseline:
    """The baseline's identity: its commit, index, tracked flags and configuration."""

    head_revision: str
    staged_entries_sha256: str
    tracked_flags_sha256: str
    configuration_sha256: str


async def initialize_ephemeral_git_baseline(
    runner: Runner,
    *,
    cwd: str | None = None,
    tree_bytes: int | None = None,
    tree_files: int | None = None,
    timeout_s: int | None = None,
) -> None:
    """Commit the workspace at ``cwd`` (the runner's root by default) to a fresh, inert repository.

    Every step, from ``git init`` through the commit and the check of its
    configuration, runs in one ``/bin/sh`` command, because each command costs
    a round trip that on a brokered sandbox outweighs the steps themselves. No
    step reads back output that grows with the workspace, so a runner whose
    command output is capped, as a brokered sandbox's is, can baseline any tree
    it holds. The command gets ``timeout_s``, or by default a time that grows
    with the copied tree's ``tree_bytes`` and ``tree_files`` (see
    ``baseline_step_timeout_seconds``). That is the most any one step got when
    the steps ran as separate commands, so the baseline needs no more of a
    sandbox's remaining lifetime than its ``git add`` did; the other steps take
    a fraction of a second.
    """

    if timeout_s is None:
        tree_timeout = baseline_step_timeout_seconds(tree_bytes=tree_bytes, tree_files=tree_files)
    elif type(timeout_s) is not int or timeout_s < 1:
        raise ValueError("timeout_s must be a positive integer.")
    else:
        tree_timeout = timeout_s

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
    result = await runner.exec_system(
        ExecCommand.process(
            "/bin/sh", "-c", _BASELINE_SCRIPT, "cayu-git-baseline", _UNSAFE_CONFIGURATION_PATTERN
        ),
        cwd=cwd,
        env=environment,
        env_remove=GIT_ENV_REMOVE,
        timeout_s=tree_timeout,
        output_limit_bytes=64 * 1024,
    )
    if result.exit_code == _UNSAFE_CONFIGURATION_EXIT_CODE and not (
        result.timed_out or result.cancelled
    ):
        raise RuntimeError("The coding sandbox's ephemeral Git configuration is unsafe.")
    if result.exit_code != 0 or result.timed_out or result.cancelled:
        raise RuntimeError("The coding sandbox could not establish its ephemeral Git baseline.")


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
        ExecCommand.process(*_UNSAFE_CONFIGURATION_QUERY, _UNSAFE_CONFIGURATION_PATTERN),
        cwd=cwd,
        env={
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "0",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            **_QUERY_ENVIRONMENT,
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
