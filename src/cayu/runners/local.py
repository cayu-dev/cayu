from __future__ import annotations

from os import PathLike
from pathlib import Path

from cayu._validation import require_nonblank
from cayu.runners._subprocess import (
    SubprocessCommand,
    copy_runner_env,
    run_subprocess,
)
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
)


class LocalRunner(Runner):
    """Executes local commands with cwd restricted under one root.

    This is not a sandbox. Commands still run with the permissions of the
    current OS user and can access absolute paths allowed by the OS.
    """

    isolation = "local"

    def __init__(self, root: str | Path, *, inherit_env: bool = True) -> None:
        if not isinstance(root, str | PathLike):
            raise TypeError("LocalRunner root must be a string or Path.")
        if not isinstance(inherit_env, bool):
            raise TypeError("LocalRunner inherit_env must be a bool.")
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Runner root does not exist: {root_path}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Runner root is not a directory: {root_path}")
        self.root = root_path
        self.inherit_env = inherit_env

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        if type(command) is not ExecCommand:
            raise TypeError("LocalRunner command must be an ExecCommand.")
        working_dir = self.resolve_cwd(cwd)
        environment = copy_runner_env(env, inherit_env=self.inherit_env)
        subprocess_command = _subprocess_command(command)
        return await run_subprocess(
            subprocess_command,
            cwd=working_dir,
            env=environment,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )

    def resolve_cwd(self, cwd: str | None = None) -> Path:
        if cwd is None:
            return self.root
        cwd = require_nonblank(cwd, "cwd")
        candidate = Path(cwd)
        if candidate.is_absolute():
            raise ValueError("Runner cwd must be relative.")
        resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("Runner cwd escapes the runner root.") from exc
        if not resolved.exists():
            raise FileNotFoundError(f"Runner cwd does not exist: {cwd}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Runner cwd is not a directory: {cwd}")
        return resolved


def _subprocess_command(command: ExecCommand) -> SubprocessCommand:
    if command.kind == "process":
        if command.argv is None:
            raise ValueError("Process commands require argv.")
        return SubprocessCommand(argv=command.argv)
    if command.shell is None:
        raise ValueError("Shell commands require a script.")
    return SubprocessCommand(shell=command.shell)
