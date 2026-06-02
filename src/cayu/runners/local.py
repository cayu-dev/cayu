from __future__ import annotations

import asyncio
import os
import signal
from os import PathLike
from pathlib import Path

from cayu._validation import require_nonblank
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
        environment = _copy_env(env, inherit_env=self.inherit_env)
        timeout = _validate_timeout(timeout_s)
        standard_input = _validate_stdin(stdin)
        output_limit = _validate_output_limit(output_limit_bytes)
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

        input_bytes = standard_input.encode("utf-8") if standard_input is not None else None
        stdin_task = asyncio.create_task(_write_stdin(process, input_bytes))
        stdout_task = asyncio.create_task(_read_limited(process.stdout, output_limit))
        stderr_task = asyncio.create_task(_read_limited(process.stderr, output_limit))
        wait_task = asyncio.create_task(process.wait())
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=timeout)
            timed_out = False
        except TimeoutError:
            timed_out = True
            _kill_process(process)
            try:
                await _await_process_exit(wait_task)
            except asyncio.CancelledError:
                await _cleanup_io_tasks(stdin_task, stdout_task, stderr_task)
                raise
        except asyncio.CancelledError:
            _kill_process(process)
            try:
                await _await_process_exit(wait_task)
            finally:
                await _cleanup_io_tasks(stdin_task, stdout_task, stderr_task)
            raise
        finally:
            await _cleanup_io_tasks(stdin_task)

        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
        if timed_out:
            return ExecResult(
                stdout=stdout.content.decode("utf-8", errors="replace"),
                stderr=stderr.content.decode("utf-8", errors="replace"),
                exit_code=process.returncode if process.returncode is not None else -1,
                timed_out=True,
                stdout_truncated=stdout.truncated,
                stderr_truncated=stderr.truncated,
            )
        return ExecResult(
            stdout=stdout.content.decode("utf-8", errors="replace"),
            stderr=stderr.content.decode("utf-8", errors="replace"),
            exit_code=process.returncode if process.returncode is not None else 0,
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
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


async def _await_process_exit(wait_task: asyncio.Task[int]) -> None:
    try:
        await asyncio.shield(wait_task)
    except asyncio.CancelledError:
        await asyncio.shield(wait_task)
        raise


async def _cleanup_io_tasks(*tasks: asyncio.Task) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _write_stdin(
    process: asyncio.subprocess.Process,
    input_bytes: bytes | None,
) -> None:
    if process.stdin is None:
        return
    try:
        if input_bytes is not None:
            process.stdin.write(input_bytes)
            await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()
    except (BrokenPipeError, ConnectionResetError):
        return


class _CapturedOutput:
    def __init__(self, content: bytes, *, truncated: bool) -> None:
        self.content = content
        self.truncated = truncated


async def _read_limited(
    stream: asyncio.StreamReader | None,
    limit: int | None,
) -> _CapturedOutput:
    if stream is None:
        return _CapturedOutput(b"", truncated=False)
    chunks: list[bytes] = []
    captured = 0
    truncated = False
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        if limit is None:
            chunks.append(chunk)
            continue
        remaining = limit - captured
        if remaining > 0:
            chunks.append(chunk[:remaining])
            captured += min(len(chunk), remaining)
        if len(chunk) > remaining:
            truncated = True
    return _CapturedOutput(b"".join(chunks), truncated=truncated)


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


def _validate_output_limit(output_limit_bytes: int | None) -> int | None:
    if output_limit_bytes is None:
        return None
    if type(output_limit_bytes) is not int:
        raise TypeError("Runner output_limit_bytes must be an integer.")
    if output_limit_bytes <= 0:
        raise ValueError("Runner output_limit_bytes must be greater than zero.")
    return output_limit_bytes
