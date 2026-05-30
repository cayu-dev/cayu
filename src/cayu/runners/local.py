from __future__ import annotations

import asyncio
import os
import signal
from os import PathLike
from pathlib import Path

from cayu._validation import require_nonblank
from cayu.runners.base import ExecCommand, ExecResult, Runner


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
    ) -> ExecResult:
        if type(command) is not ExecCommand:
            raise TypeError("LocalRunner command must be an ExecCommand.")
        working_dir = self.resolve_cwd(cwd)
        environment = _copy_env(env, inherit_env=self.inherit_env)
        timeout = _validate_timeout(timeout_s)
        standard_input = _validate_stdin(stdin)
        process_options = _subprocess_options()

        try:
            if command.kind == "process":
                if command.argv is None:
                    raise ValueError("Process commands require argv.")
                process = await asyncio.create_subprocess_exec(
                    *command.argv,
                    cwd=str(working_dir),
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **process_options,
                )
            else:
                if command.shell is None:
                    raise ValueError("Shell commands require a script.")
                process = await asyncio.create_subprocess_shell(
                    command.shell,
                    cwd=str(working_dir),
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **process_options,
                )
        except FileNotFoundError:
            command_name = command.argv[0] if command.argv else command.kind
            return ExecResult(
                stderr=f"Command not found: {command_name}",
                exit_code=127,
            )
        except PermissionError:
            command_name = command.argv[0] if command.argv else command.kind
            return ExecResult(
                stderr=f"Command not executable: {command_name}",
                exit_code=126,
            )

        input_bytes = (
            standard_input.encode("utf-8")
            if standard_input is not None
            else None
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_bytes),
                timeout=timeout,
            )
            return ExecResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode if process.returncode is not None else 0,
            )
        except asyncio.TimeoutError:
            _kill_process(process)
            stdout, stderr = await process.communicate()
            return ExecResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=process.returncode if process.returncode is not None else -1,
                timed_out=True,
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


def _copy_env(env: dict[str, str] | None, *, inherit_env: bool) -> dict[str, str]:
    base_env = os.environ.copy() if inherit_env else {}
    if env is None:
        return base_env
    if type(env) is not dict:
        raise TypeError("Runner env must be a dictionary.")
    copied = base_env
    for key, value in env.items():
        if type(key) is not str or not key.strip():
            raise ValueError("Runner env keys must be non-empty strings.")
        if type(value) is not str:
            raise ValueError("Runner env values must be strings.")
        copied[key] = value
    return copied


def _subprocess_options() -> dict[str, bool]:
    if os.name == "posix":
        return {"start_new_session": True}
    return {}


def _kill_process(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    process.kill()


def _validate_timeout(timeout_s: int | None) -> int | None:
    if timeout_s is None:
        return None
    if type(timeout_s) is not int:
        raise TypeError("Runner timeout_s must be an integer.")
    if timeout_s <= 0:
        raise ValueError("Runner timeout_s must be greater than zero.")
    return timeout_s


def _validate_stdin(stdin: str | None) -> str | None:
    if stdin is None:
        return None
    if type(stdin) is not str:
        raise TypeError("Runner stdin must be a string.")
    return stdin
