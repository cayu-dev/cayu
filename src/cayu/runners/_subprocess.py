from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
from collections.abc import Iterable
from pathlib import Path

from cayu.runners._redacted_output import RedactedOutputCapture
from cayu.runners.base import ExecResult
from cayu.vaults import SecretRedactor
from cayu.vaults.redaction import _bounded_redacted_head

# Wall-clock bound for draining captured stdout/stderr after the child has been
# killed. A daemonizing grandchild can inherit the pipe write ends and keep them
# open indefinitely, so the post-kill read tasks would otherwise never see EOF
# and hang forever, defeating the timeout guarantee.
_DRAIN_AFTER_KILL_S = 2.0
_TASKKILL_TIMEOUT_S = 2.0
_TASKKILL_ENV_KEYS = (
    "COMSPEC",
    "PATH",
    "PATHEXT",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
)


def remove_runner_env(environment: dict[str, str], keys: Iterable[str]) -> dict[str, str]:
    """Return an environment without explicitly forbidden variables."""

    removed: set[str] = set()
    for key in keys:
        if type(key) is not str or not key:
            raise ValueError("Runner env_remove entries must be non-empty strings.")
        removed.add(key)
    return {key: value for key, value in environment.items() if key not in removed}


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
    output_redactor: SecretRedactor | None = None,
    start_new_session: bool | None = None,
) -> ExecResult:
    """Run a subprocess with bounded output, timeout, and cancellation cleanup."""

    if type(command) is not SubprocessCommand:
        raise TypeError("run_subprocess command must be a SubprocessCommand.")
    timeout = validate_timeout(timeout_s)
    standard_input = validate_stdin(stdin)
    output_limit = validate_output_limit(output_limit_bytes)
    if output_redactor is not None and not isinstance(output_redactor, SecretRedactor):
        raise TypeError("run_subprocess output_redactor must be a SecretRedactor.")
    redactor = output_redactor or SecretRedactor()
    working_dir = _copy_cwd(cwd)
    environment = copy_runner_env(env, inherit_env=False)
    env = None
    use_new_session = os.name == "posix" if start_new_session is None else start_new_session

    try:
        if command.argv is not None:
            if os.name == "posix":
                spawn = asyncio.create_subprocess_exec(
                    *command.argv,
                    cwd=working_dir,
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=use_new_session,
                )
            else:
                spawn = asyncio.create_subprocess_exec(
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
                spawn = asyncio.create_subprocess_shell(
                    command.shell,
                    cwd=working_dir,
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=use_new_session,
                )
            else:
                spawn = asyncio.create_subprocess_shell(
                    command.shell,
                    cwd=working_dir,
                    env=environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
        # The spawn coroutine owns the detached environment now. Drop this
        # frame's direct reference before cancellation can cross the await.
        environment = {}
        process = await spawn
    except FileNotFoundError:
        message = f"Command not found: {command.command_name}"
        return _redact_and_bound_exec_result(
            ExecResult(
                stderr=message,
                exit_code=127,
                stdout_bytes=0,
                stderr_bytes=len(message.encode("utf-8")),
            ),
            redactor=redactor,
            output_limit=output_limit,
        )
    except PermissionError:
        message = f"Command not executable: {command.command_name}"
        return _redact_and_bound_exec_result(
            ExecResult(
                stderr=message,
                exit_code=126,
                stdout_bytes=0,
                stderr_bytes=len(message.encode("utf-8")),
            ),
            redactor=redactor,
            output_limit=output_limit,
        )

    input_bytes = standard_input.encode("utf-8") if standard_input is not None else None
    stdout = RedactedOutputCapture(redactor=redactor, limit=output_limit)
    stderr = RedactedOutputCapture(redactor=redactor, limit=output_limit)
    stdin_task = asyncio.create_task(_write_stdin(process, input_bytes))
    stdout_task = asyncio.create_task(_read_limited(process.stdout, stdout))
    stderr_task = asyncio.create_task(_read_limited(process.stderr, stderr))
    wait_task = asyncio.create_task(process.wait())
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=timeout)
        timed_out = False
    except TimeoutError:
        timed_out = True
        await _kill_timed_out_process(
            process,
            process_group=use_new_session,
            stdin_task=stdin_task,
            stdout_task=stdout_task,
            stderr_task=stderr_task,
            wait_task=wait_task,
        )
    except asyncio.CancelledError:
        await _cleanup_cancelled_process(
            process,
            process_group=use_new_session,
            stdin_task=stdin_task,
            stdout_task=stdout_task,
            stderr_task=stderr_task,
            wait_task=wait_task,
        )
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
        stdout=stdout.text(),
        stderr=stderr.text(),
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


def _redact_and_bound_exec_result(
    result: ExecResult,
    *,
    redactor: SecretRedactor,
    output_limit: int | None,
) -> ExecResult:
    stdout, stdout_redaction_truncated = _redact_and_bound_output(
        result.stdout,
        redactor=redactor,
        output_limit=output_limit,
    )
    stderr, stderr_redaction_truncated = _redact_and_bound_output(
        result.stderr,
        redactor=redactor,
        output_limit=output_limit,
    )
    return result.model_copy(
        update={
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": result.stdout_truncated
            or stdout_redaction_truncated
            or (
                output_limit is not None
                and result.stdout_bytes is not None
                and result.stdout_bytes > output_limit
            ),
            "stderr_truncated": result.stderr_truncated
            or stderr_redaction_truncated
            or (
                output_limit is not None
                and result.stderr_bytes is not None
                and result.stderr_bytes > output_limit
            ),
        }
    )


def _redact_and_bound_output(
    value: str,
    *,
    redactor: SecretRedactor,
    output_limit: int | None,
) -> tuple[str, bool]:
    redacted = redactor.redact_text(value)
    if output_limit is None:
        return redacted, False
    encoded = redacted.encode("utf-8", "replace")
    if len(encoded) <= output_limit:
        return redacted, False

    return (
        _bounded_redacted_head(encoded, max_bytes=output_limit).decode(
            "utf-8",
            "ignore",
        ),
        True,
    )


def _copy_cwd(cwd: Path | str | None) -> str | None:
    if cwd is None:
        return None
    if isinstance(cwd, Path):
        return str(cwd)
    if type(cwd) is str:
        return cwd
    raise TypeError("Subprocess cwd must be a string, Path, or None.")


async def _kill_process(
    process: asyncio.subprocess.Process,
    *,
    process_group: bool,
) -> None:
    if os.name == "posix" and process_group:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
    if os.name == "nt" and await _taskkill_tree(process.pid):
        return
    process.kill()


async def _kill_timed_out_process(
    process: asyncio.subprocess.Process,
    *,
    process_group: bool,
    stdin_task: asyncio.Task[None],
    stdout_task: asyncio.Task[None],
    stderr_task: asyncio.Task[None],
    wait_task: asyncio.Task[int],
) -> None:
    """Kill a timed-out process without letting cancellation strand its I/O tasks."""
    termination_task = asyncio.create_task(
        _kill_process_and_wait(process, process_group=process_group, wait_task=wait_task)
    )
    try:
        cancellation = await _await_task_resisting_cancellation(termination_task)
    except Exception as termination_error:
        cancellation = await _cleanup_io_tasks_resisting_cancellation(
            stdin_task, stdout_task, stderr_task, wait_task
        )
        if cancellation is not None:
            cancellation.add_note(
                "Process termination failed before cancellation interrupted I/O cleanup: "
                f"{type(termination_error).__name__}: {termination_error}"
            )
            raise cancellation from termination_error
        raise
    if cancellation is not None:
        await _cleanup_io_tasks_resisting_cancellation(
            stdin_task, stdout_task, stderr_task, wait_task
        )
        raise cancellation


async def _cleanup_cancelled_process(
    process: asyncio.subprocess.Process,
    *,
    process_group: bool,
    stdin_task: asyncio.Task[None],
    stdout_task: asyncio.Task[None],
    stderr_task: asyncio.Task[None],
    wait_task: asyncio.Task[int],
) -> None:
    """Finish bounded process teardown while preserving the caller's cancellation."""
    termination_task = asyncio.create_task(
        _kill_process_and_wait(process, process_group=process_group, wait_task=wait_task)
    )
    try:
        await _await_task_resisting_cancellation(termination_task)
    except Exception:
        # Cancellation stays authoritative even if the preferred tree cleanup
        # fails unexpectedly. Make one final best-effort direct-child kill.
        with contextlib.suppress(OSError):
            process.kill()
    finally:
        await _cleanup_io_tasks_resisting_cancellation(
            stdin_task, stdout_task, stderr_task, wait_task
        )


async def _kill_process_and_wait(
    process: asyncio.subprocess.Process,
    *,
    process_group: bool,
    wait_task: asyncio.Task[int],
) -> None:
    await _kill_process(process, process_group=process_group)
    # Bounded (see _await_process_exit): process.wait() only resolves once the
    # captured pipes reach EOF, so a pipe-holding descendant cannot hang this.
    await _await_process_exit(wait_task)


async def _await_task_resisting_cancellation(
    task: asyncio.Task[None],
) -> asyncio.CancelledError | None:
    """Let a bounded cleanup task finish, remembering repeated caller cancellation."""
    cancellation: asyncio.CancelledError | None = None
    task_exception: BaseException | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            if cancellation is None:
                cancellation = exc
        except Exception as exc:
            task_exception = exc
            break
    if task_exception is None:
        task_exception = task.exception()
    if task_exception is not None:
        if cancellation is None:
            raise task_exception
        cancellation.add_note(
            "Cleanup task failed while cancellation was pending: "
            f"{type(task_exception).__name__}: {task_exception}"
        )
        cancellation.__cause__ = task_exception
    return cancellation


async def _cleanup_io_tasks_resisting_cancellation(
    *tasks: asyncio.Task,
) -> asyncio.CancelledError | None:
    cleanup_task = asyncio.create_task(_cleanup_io_tasks(*tasks))
    return await _await_task_resisting_cancellation(cleanup_task)


async def _taskkill_tree(pid: int) -> bool:
    """Kill a Windows process together with its child tree via ``taskkill /T``.

    ``Process.kill`` maps to ``TerminateProcess``, which only reaps the direct
    child, so a spawned grandchild could keep the captured pipes open. Returns
    True when the tree kill was issued successfully.
    """
    try:
        completed = await asyncio.wait_for(
            asyncio.to_thread(
                subprocess.run,
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
                env={name: os.environ[name] for name in _TASKKILL_ENV_KEYS if name in os.environ},
                timeout=_TASKKILL_TIMEOUT_S,
            ),
            timeout=_TASKKILL_TIMEOUT_S,
        )
    except (TimeoutError, OSError, ValueError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


async def _bounded_drain(
    process: asyncio.subprocess.Process,
    stdout_task: asyncio.Task[None],
    stderr_task: asyncio.Task[None],
    wait_task: asyncio.Task[int],
    *,
    captures: tuple[RedactedOutputCapture, ...],
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


async def _read_limited(
    stream: asyncio.StreamReader | None,
    out: RedactedOutputCapture,
) -> None:
    if stream is None:
        out.finish_complete()
        return
    try:
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                out.finish_complete()
                return
            out.append(chunk)
    except BaseException:
        out.abort()
        raise
