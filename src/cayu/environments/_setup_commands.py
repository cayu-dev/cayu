"""Shell commands a sandbox factory runs before a session starts, shared across backends."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

from cayu.egress._docker_diagnostics import docker_stderr_tail
from cayu.runners.base import ExecCommand, ExecResult, Runner
from cayu.vaults import SecretRedactor

__all__ = [
    "run_setup_commands",
    "setup_command_failure_message",
    "setup_commands_sha256",
    "validate_setup_command_timeout",
    "validated_setup_commands",
]


def validated_setup_commands(commands: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    """Own ``commands`` as a tuple of nonblank shell commands, refusing one bare string."""

    if isinstance(commands, str | bytes):
        raise TypeError(f"{field_name} must be a sequence of shell commands, not one string.")
    owned = tuple(commands)
    if any(type(command) is not str or not command.strip() for command in owned):
        raise ValueError(f"{field_name} entries must be nonblank shell commands.")
    return owned


def validate_setup_command_timeout(value: object) -> int:
    """Return a setup command timeout: a positive whole number of seconds."""

    if type(value) is not int or value <= 0:
        raise ValueError("setup_command_timeout_s must be a positive integer number of seconds.")
    return value


def setup_command_failure_message(
    label: str,
    result: ExecResult,
    *,
    timeout_s: int,
    redactor: SecretRedactor | None = None,
) -> str:
    """Say why a setup command failed, ending with a bounded, masked tail of its output."""

    if result.timed_out:
        reason = f"timed out after {timeout_s} seconds"
    elif result.cancelled:
        reason = "was cancelled"
    else:
        reason = f"exit code {result.exit_code}"
    stderr = docker_stderr_tail(result.stderr, redactor=redactor)
    stdout = docker_stderr_tail(result.stdout, redactor=redactor)
    return f"{label} failed ({reason}); stderr: {stderr}; stdout: {stdout}"


def setup_commands_sha256(commands: Sequence[str]) -> str:
    """The hex digest a factory's configuration fingerprint records for ``commands``."""

    return sha256("\0".join(commands).encode("utf-8")).hexdigest()


async def run_setup_commands(
    runner: Runner,
    commands: Sequence[str],
    *,
    label: str,
    timeout_s: int,
    cwd: str | None = None,
) -> None:
    """Run ``commands`` one at a time in ``runner``, raising on the first that fails.

    A nonzero exit, a timeout or a cancellation fails with
    ``setup_command_failure_message`` under ``label``.
    """

    for command in commands:
        result = await runner.exec(ExecCommand.bash(command), cwd=cwd, timeout_s=timeout_s)
        if result.exit_code != 0 or result.timed_out or result.cancelled:
            raise RuntimeError(setup_command_failure_message(label, result, timeout_s=timeout_s))
