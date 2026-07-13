from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
from pathlib import Path

from cayu.runners.base import ExecResult

# Wall-clock bound for draining captured stdout/stderr after the child has been
# killed. A daemonizing grandchild can inherit the pipe write ends and keep them
# open indefinitely, so the post-kill read tasks would otherwise never see EOF
# and hang forever, defeating the timeout guarantee.
_DRAIN_AFTER_KILL_S = 2.0


class SubprocessCommand:
    """Validated command shape for internal runner subprocess execution."""

    def __init__(
        self,
        *,
        argv: list[str] | None = None,
        shell: str | None = None,
    ) -> None:
        if (argv is None) == (shell is None):
            raise ValueError("SubprocessCommand requires exactly one of argv or shell.")
        if argv is not None:
            if type(argv) is not list:
                raise TypeError("Subprocess argv must be a list.")
            if not argv:
                raise ValueError("Subprocess argv cannot be empty.")
            for item in argv:
                if type(item) is not str or not item.strip():
                    raise ValueError("Subprocess argv entries must be non-empty strings.")
            self.argv = list(argv)
            self.shell = None
            return
        if type(shell) is not str or not shell.strip():
            raise ValueError("Subprocess shell command must be a non-empty string.")
        self.argv = None
        self.shell = shell

    @property
    def command_name(self) -> str:
        if self.argv:
            return self.argv[0]
        return "shell"


async def run_subprocess(
    command: SubprocessCommand,
    *,
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
    stdin: str | None = None,
    output_limit_bytes: int | None = None,
    start_new_session: bool | None = None,
) -> ExecResult:
    """Run a subprocess with bounded output, timeout, and cancellation cleanup."""

    if type(command) is not SubprocessCommand:
        raise TypeError("run_subprocess command must be a SubprocessCommand.")
    timeout = validate_timeout(timeout_s)
    standard_input = validate_stdin(stdin)
    output_limit = validate_output_limit(output_limit_bytes)
    working_dir = _copy_cwd(cwd)
    environment = copy_runner_env(env, inherit_env=False)
    use_new_session = os.name == "posix" if start_new_session is None else start_new_session

    try:
        if command.argv is not None:
            if os.name == "posix":
                process = await asyncio.create_subprocess_exec(
                    *command.argv,
                    cwd=working_dir,
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=use_new_session,
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *command.argv,
                    cwd=working_dir,
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
        else:
            if command.shell is None:
                raise ValueError("Subprocess shell command cannot be None.")
            if os.name == "posix":
                process = await asyncio.create_subprocess_shell(
                    command.shell,
                    cwd=working_dir,
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=use_new_session,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command.shell,
                    cwd=working_dir,
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
    except FileNotFoundError:
        message = f"Command not found: {command.command_name}"
        return ExecResult(
            stderr=message,
            exit_code=127,
            stdout_bytes=0,
            stderr_bytes=len(message.encode("utf-8")),
        )
    except PermissionError:
        message = f"Command not executable: {command.command_name}"
        return ExecResult(
            stderr=message,
            exit_code=126,
            stdout_bytes=0,
            stderr_bytes=len(message.encode("utf-8")),
        )

    input_bytes = standard_input.encode("utf-8") if standard_input is not None else None
    stdout = _CapturedOutput()
    stderr = _CapturedOutput()
    stdin_task = asyncio.create_task(_write_stdin(process, input_bytes))
    stdout_task = asyncio.create_task(_read_limited(process.stdout, output_limit, stdout))
    stderr_task = asyncio.create_task(_read_limited(process.stderr, output_limit, stderr))
    wait_task = asyncio.create_task(process.wait())
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=timeout)
        timed_out = False
    except TimeoutError:
        timed_out = True
        _kill_process(process, process_group=use_new_session)
        try:
            # Bounded (see _await_process_exit): process.wait() only resolves
            # once the captured pipes reach EOF, so a pipe-holding descendant
            # would otherwise hang this await past the wall-clock limit.
            await _await_process_exit(wait_task)
        except asyncio.CancelledError:
            await _cleanup_io_tasks(stdin_task, stdout_task, stderr_task, wait_task)
            raise
    except asyncio.CancelledError:
        _kill_process(process, process_group=use_new_session)
        try:
            await _await_process_exit(wait_task)
        finally:
            await _cleanup_io_tasks(stdin_task, stdout_task, stderr_task)
        raise
    finally:
        await _cleanup_io_tasks(stdin_task)

    if timed_out:
        # The child was killed and its exit already awaited (bounded); bound the
        # output drain too so a daemonizing grandchild that inherited the pipes
        # cannot hold the read tasks open past the wall-clock limit.
        await _bounded_drain(
            process, stdout_task, stderr_task, wait_task, captures=(stdout, stderr)
        )
    else:
        await asyncio.gather(stdout_task, stderr_task)
    return ExecResult(
        stdout=stdout.content.decode("utf-8", errors="replace"),
        stderr=stderr.content.decode("utf-8", errors="replace"),
        exit_code=process.returncode
        if process.returncode is not None
        else (-1 if timed_out else 0),
        timed_out=timed_out,
        stdout_truncated=stdout.truncated,
        stderr_truncated=stderr.truncated,
        stdout_bytes=stdout.total_bytes,
        stderr_bytes=stderr.total_bytes,
    )


def copy_runner_env(env: dict[str, str] | None, *, inherit_env: bool) -> dict[str, str]:
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


def validate_timeout(timeout_s: int | None) -> int | None:
    if timeout_s is None:
        return None
    if type(timeout_s) is not int:
        raise TypeError("Runner timeout_s must be an integer.")
    if timeout_s <= 0:
        raise ValueError("Runner timeout_s must be greater than zero.")
    return timeout_s


def validate_stdin(stdin: str | None) -> str | None:
    if stdin is None:
        return None
    if type(stdin) is not str:
        raise TypeError("Runner stdin must be a string.")
    return stdin


def validate_output_limit(output_limit_bytes: int | None) -> int | None:
    if output_limit_bytes is None:
        return None
    if type(output_limit_bytes) is not int:
        raise TypeError("Runner output_limit_bytes must be an integer.")
    if output_limit_bytes <= 0:
        raise ValueError("Runner output_limit_bytes must be greater than zero.")
    return output_limit_bytes


def _copy_cwd(cwd: Path | str | None) -> str | None:
    if cwd is None:
        return None
    if isinstance(cwd, Path):
        return str(cwd)
    if type(cwd) is str:
        return cwd
    raise TypeError("Subprocess cwd must be a string, Path, or None.")


def _kill_process(process: asyncio.subprocess.Process, *, process_group: bool) -> None:
    if os.name == "posix" and process_group:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    if os.name == "nt" and _taskkill_tree(process.pid):
        return
    process.kill()


def _taskkill_tree(pid: int) -> bool:
    """Kill a Windows process together with its child tree via ``taskkill /T``.

    ``Process.kill`` maps to ``TerminateProcess``, which only reaps the direct
    child, so a spawned grandchild could keep the captured pipes open. Returns
    True when the tree kill was issued successfully.
    """
    try:
        completed = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
    except (OSError, ValueError):
        return False
    return completed.returncode == 0


async def _bounded_drain(
    process: asyncio.subprocess.Process,
    stdout_task: asyncio.Task[None],
    stderr_task: asyncio.Task[None],
    wait_task: asyncio.Task[int],
    *,
    captures: tuple[_CapturedOutput, ...],
) -> None:
    """Await the post-kill exit + read tasks under a wall-clock bound.

    Shielded so the read tasks keep accumulating into their captures until the
    deadline; on timeout the captures are marked truncated and every task is
    cancelled so a pipe-holding grandchild cannot block the caller forever.
    """
    try:
        await asyncio.wait_for(
            asyncio.gather(
                asyncio.shield(stdout_task),
                asyncio.shield(stderr_task),
                asyncio.shield(wait_task),
            ),
            timeout=_DRAIN_AFTER_KILL_S,
        )
    except TimeoutError:
        for capture in captures:
            capture.truncated = True
        await _cleanup_io_tasks(stdout_task, stderr_task, wait_task)
        _detach_process(process)
    except asyncio.CancelledError:
        await _cleanup_io_tasks(stdout_task, stderr_task, wait_task)
        _detach_process(process)
        raise


def _detach_process(process: asyncio.subprocess.Process) -> None:
    """Close the subprocess transport after abandoning a pipe-holding child.

    The direct child is already killed; closing the transport releases our pipe
    ends while the loop is still running, avoiding an "event loop is closed"
    teardown warning if a leaked descendant keeps the write ends open.
    """
    transport = getattr(process, "_transport", None)
    if transport is None:
        return
    with contextlib.suppress(Exception):
        transport.close()


async def _await_process_exit(wait_task: asyncio.Task[int]) -> None:
    """Wait for the killed process to exit, capped by the wall-clock bound.

    asyncio resolves ``process.wait()`` only once every captured pipe reaches
    EOF, so a daemonizing descendant that inherited the write ends would keep it
    pending forever. The bound lets us abandon that wait (leaving ``wait_task``
    to be cancelled by the caller) instead of hanging.
    """
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=_DRAIN_AFTER_KILL_S)
    except TimeoutError:
        return
    except asyncio.CancelledError:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(wait_task), timeout=_DRAIN_AFTER_KILL_S)
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
    """Mutable sink for a captured stream.

    Reads accumulate here in place so that a partially-drained capture remains
    available to the caller even when its read task is cancelled by the bounded
    post-kill drain.
    """

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._captured = 0
        self.total_bytes = 0
        self.truncated = False

    @property
    def content(self) -> bytes:
        return b"".join(self._chunks)

    def append(self, chunk: bytes, *, limit: int | None) -> None:
        self.total_bytes += len(chunk)
        if limit is None:
            self._chunks.append(chunk)
            return
        remaining = limit - self._captured
        if remaining > 0:
            self._chunks.append(chunk[:remaining])
            self._captured += min(len(chunk), remaining)
        if len(chunk) > remaining:
            self.truncated = True


async def _read_limited(
    stream: asyncio.StreamReader | None,
    limit: int | None,
    out: _CapturedOutput,
) -> None:
    if stream is None:
        return
    while True:
        chunk = await stream.read(8192)
        if not chunk:
            break
        out.append(chunk, limit=limit)
