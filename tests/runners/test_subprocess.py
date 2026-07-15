from __future__ import annotations

import asyncio
import subprocess
import sys
import time

import pytest

import cayu.runners._subprocess as subprocess_module
from cayu.runners._subprocess import (
    SubprocessCommand,
    copy_runner_env,
    run_subprocess,
    validate_output_limit,
    validate_stdin,
    validate_timeout,
)


def test_subprocess_command_accepts_exactly_one_command_shape() -> None:
    assert SubprocessCommand(argv=["python", "--version"]).argv == ["python", "--version"]
    assert SubprocessCommand(shell="echo ok").shell == "echo ok"

    with pytest.raises(ValueError, match="exactly one"):
        SubprocessCommand()

    with pytest.raises(ValueError, match="exactly one"):
        SubprocessCommand(argv=["echo"], shell="echo ok")


def test_subprocess_command_rejects_invalid_argv_and_shell() -> None:
    with pytest.raises(ValueError, match="empty"):
        SubprocessCommand(argv=[])

    with pytest.raises(ValueError, match="non-empty"):
        SubprocessCommand(argv=[" "])

    with pytest.raises(ValueError, match="non-empty"):
        SubprocessCommand(shell=" ")


def test_runner_env_copy_can_inherit_or_isolate_parent_env(monkeypatch) -> None:
    monkeypatch.setenv("CAYU_PARENT_ENV", "visible")

    inherited = copy_runner_env({"CHILD": "set"}, inherit_env=True)
    assert inherited["CAYU_PARENT_ENV"] == "visible"
    assert inherited["CHILD"] == "set"

    isolated = copy_runner_env({"CHILD": "set"}, inherit_env=False)
    assert "CAYU_PARENT_ENV" not in isolated
    assert isolated == {"CHILD": "set"}


def test_run_subprocess_does_not_inherit_parent_env_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CAYU_PARENT_ENV", "hidden")

    isolated = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[
                    sys.executable,
                    "-c",
                    "import os; print(os.environ.get('CAYU_PARENT_ENV', ''))",
                ]
            ),
            cwd=tmp_path,
        )
    )
    inherited = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[
                    sys.executable,
                    "-c",
                    "import os; print(os.environ.get('CAYU_PARENT_ENV', ''))",
                ]
            ),
            cwd=tmp_path,
            env=copy_runner_env(None, inherit_env=True),
        )
    )

    assert isolated.stdout == "\n"
    assert inherited.stdout == "hidden\n"


def test_runner_env_copy_rejects_invalid_env() -> None:
    with pytest.raises(TypeError, match="dictionary"):
        copy_runner_env([], inherit_env=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="keys"):
        copy_runner_env({" ": "bad"}, inherit_env=False)

    with pytest.raises(ValueError, match="values"):
        copy_runner_env({"KEY": 1}, inherit_env=False)  # type: ignore[dict-item]


def test_runner_validation_helpers_reject_invalid_values() -> None:
    with pytest.raises(TypeError, match="timeout_s"):
        validate_timeout("1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        validate_timeout(0)

    with pytest.raises(TypeError, match="stdin"):
        validate_stdin(b"bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="output_limit_bytes"):
        validate_output_limit("1")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="greater than zero"):
        validate_output_limit(0)


def test_run_subprocess_executes_process_and_bounds_output(tmp_path) -> None:
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[
                    sys.executable,
                    "-c",
                    "print('abcdef')",
                ]
            ),
            cwd=tmp_path,
            env={},
            output_limit_bytes=4,
        )
    )

    assert result.stdout == "abcd"
    assert result.stdout_truncated is True
    assert result.stdout_bytes == 7
    assert result.stderr_bytes == 0
    assert result.exit_code == 0


def test_run_subprocess_executes_shell_and_stdin(tmp_path) -> None:
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(shell="cat"),
            cwd=tmp_path,
            env={},
            stdin="hello",
        )
    )

    assert result.stdout == "hello"
    assert result.exit_code == 0


def test_run_subprocess_reports_missing_command(tmp_path) -> None:
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(argv=["cayu-command-that-does-not-exist"]),
            cwd=tmp_path,
            env={},
        )
    )

    assert result.exit_code == 127
    assert "Command not found" in result.stderr
    assert result.stdout_bytes == 0
    assert result.stderr_bytes == len(result.stderr.encode("utf-8"))


def test_run_subprocess_times_out_and_returns_partial_output(tmp_path) -> None:
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(
                argv=[
                    sys.executable,
                    "-c",
                    "import time; print('before', flush=True); time.sleep(5)",
                ]
            ),
            cwd=tmp_path,
            env={},
            timeout_s=1,
        )
    )

    assert result.stdout == "before\n"
    assert result.timed_out is True
    assert result.exit_code != 0


def test_windows_taskkill_runs_off_event_loop_with_timeout(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def slow_taskkill(argv, *, capture_output, check, timeout):
        observed.update(
            argv=argv,
            capture_output=capture_output,
            check=check,
            timeout=timeout,
        )
        time.sleep(0.05)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess_module.subprocess, "run", slow_taskkill)

    async def run() -> tuple[bool, int]:
        task = asyncio.create_task(subprocess_module._taskkill_tree(123))
        ticks = 0
        while not task.done():
            ticks += 1
            await asyncio.sleep(0.005)
        return await task, ticks

    succeeded, ticks = asyncio.run(run())

    assert succeeded is True
    assert ticks > 1
    assert observed == {
        "argv": ["taskkill", "/F", "/T", "/PID", "123"],
        "capture_output": True,
        "check": False,
        "timeout": subprocess_module._TASKKILL_TIMEOUT_S,
    }


def test_windows_taskkill_timeout_falls_back_to_direct_kill(monkeypatch) -> None:
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="taskkill", timeout=2)

    monkeypatch.setattr(subprocess_module.subprocess, "run", timed_out)
    assert asyncio.run(subprocess_module._taskkill_tree(123)) is False

    class FakeProcess:
        pid = 123

        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    async def failed_tree_kill(pid: int) -> bool:
        assert pid == 123
        return False

    process = FakeProcess()
    monkeypatch.setattr(subprocess_module.os, "name", "nt")
    monkeypatch.setattr(subprocess_module, "_taskkill_tree", failed_tree_kill)

    asyncio.run(subprocess_module._kill_process(process, process_group=False))  # type: ignore[arg-type]

    assert process.killed is True


def test_windows_taskkill_timeout_includes_executor_queue_time(monkeypatch) -> None:
    worker_started = asyncio.Event()
    worker_cancelled = asyncio.Event()

    async def queued_to_thread(*_args, **_kwargs):
        worker_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            worker_cancelled.set()
            raise

    monkeypatch.setattr(subprocess_module.asyncio, "to_thread", queued_to_thread)
    monkeypatch.setattr(subprocess_module, "_TASKKILL_TIMEOUT_S", 0.01)

    async def run() -> bool:
        task = asyncio.create_task(subprocess_module._taskkill_tree(123))
        await worker_started.wait()
        return await asyncio.wait_for(task, timeout=0.2)

    assert asyncio.run(run()) is False
    assert worker_cancelled.is_set()


def test_timeout_kill_resists_repeated_cancellation_and_cleans_io_tasks(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    kill_started = asyncio.Event()
    release_kill = asyncio.Event()

    async def slow_kill(process, *, process_group):
        assert process_group is False
        kill_started.set()
        await release_kill.wait()
        process.kill()

    monkeypatch.setattr(subprocess_module, "_kill_process", slow_kill)

    async def run() -> tuple[FakeProcess, tuple[asyncio.Task, ...]]:
        blocker = asyncio.Event()
        io_tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(3))
        wait_task = asyncio.create_task(asyncio.sleep(0, result=0))
        process = FakeProcess()
        cleanup_task = asyncio.create_task(
            subprocess_module._kill_timed_out_process(
                process,  # type: ignore[arg-type]
                process_group=False,
                stdin_task=io_tasks[0],
                stdout_task=io_tasks[1],
                stderr_task=io_tasks[2],
                wait_task=wait_task,
            )
        )
        await kill_started.wait()
        cleanup_task.cancel()
        await asyncio.sleep(0)
        cleanup_task.cancel()
        release_kill.set()
        with pytest.raises(asyncio.CancelledError):
            await cleanup_task
        return process, (*io_tasks, wait_task)

    process, tasks = asyncio.run(run())

    assert process.killed is True
    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks[:3])


def test_cancelled_process_cleanup_resists_second_cancellation(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    kill_started = asyncio.Event()
    release_kill = asyncio.Event()

    async def slow_kill(process, *, process_group):
        assert process_group is False
        kill_started.set()
        await release_kill.wait()
        process.kill()

    monkeypatch.setattr(subprocess_module, "_kill_process", slow_kill)

    async def run() -> tuple[FakeProcess, tuple[asyncio.Task, ...]]:
        blocker = asyncio.Event()
        io_tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(3))
        wait_task = asyncio.create_task(asyncio.sleep(0, result=0))
        process = FakeProcess()

        async def operation() -> None:
            try:
                await blocker.wait()
            except asyncio.CancelledError:
                await subprocess_module._cleanup_cancelled_process(
                    process,  # type: ignore[arg-type]
                    process_group=False,
                    stdin_task=io_tasks[0],
                    stdout_task=io_tasks[1],
                    stderr_task=io_tasks[2],
                    wait_task=wait_task,
                )
                raise

        operation_task = asyncio.create_task(operation())
        await asyncio.sleep(0)
        operation_task.cancel()
        await kill_started.wait()
        operation_task.cancel()
        release_kill.set()
        with pytest.raises(asyncio.CancelledError):
            await operation_task
        return process, (*io_tasks, wait_task)

    process, tasks = asyncio.run(run())

    assert process.killed is True
    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks[:3])


def test_timeout_kill_failure_cleans_io_tasks_and_propagates(monkeypatch) -> None:
    async def failed_termination(process, *, process_group, wait_task):
        assert process_group is False
        raise RuntimeError("termination failed")

    monkeypatch.setattr(subprocess_module, "_kill_process_and_wait", failed_termination)

    async def run() -> tuple[asyncio.Task, ...]:
        blocker = asyncio.Event()
        tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(4))
        with pytest.raises(RuntimeError, match="termination failed"):
            await subprocess_module._kill_timed_out_process(
                object(),  # type: ignore[arg-type]
                process_group=False,
                stdin_task=tasks[0],
                stdout_task=tasks[1],
                stderr_task=tasks[2],
                wait_task=tasks[3],
            )
        return tasks

    tasks = asyncio.run(run())

    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks)


def test_timeout_kill_preserves_cancellation_when_termination_fails(monkeypatch) -> None:
    termination_started = asyncio.Event()
    release_termination = asyncio.Event()

    async def failed_termination(process, *, process_group, wait_task):
        assert process_group is False
        termination_started.set()
        await release_termination.wait()
        raise RuntimeError("termination failed")

    monkeypatch.setattr(subprocess_module, "_kill_process_and_wait", failed_termination)

    async def run() -> tuple[asyncio.CancelledError, tuple[asyncio.Task, ...]]:
        blocker = asyncio.Event()
        tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(4))
        cleanup_task = asyncio.create_task(
            subprocess_module._kill_timed_out_process(
                object(),  # type: ignore[arg-type]
                process_group=False,
                stdin_task=tasks[0],
                stdout_task=tasks[1],
                stderr_task=tasks[2],
                wait_task=tasks[3],
            )
        )
        await termination_started.wait()
        cleanup_task.cancel("caller cancelled")
        await asyncio.sleep(0)
        release_termination.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await cleanup_task
        return exc_info.value, tasks

    cancellation, tasks = asyncio.run(run())

    assert str(cancellation) == "caller cancelled"
    assert isinstance(cancellation.__cause__, RuntimeError)
    assert "termination failed" in "\n".join(cancellation.__notes__)
    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks)


def test_timeout_kill_preserves_cancellation_during_io_cleanup_after_termination_failure(
    monkeypatch,
) -> None:
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_io_tasks = subprocess_module._cleanup_io_tasks

    async def failed_termination(process, *, process_group, wait_task):
        assert process_group is False
        raise RuntimeError("termination failed")

    async def slow_cleanup(*tasks):
        cleanup_started.set()
        await release_cleanup.wait()
        await cleanup_io_tasks(*tasks)

    monkeypatch.setattr(subprocess_module, "_kill_process_and_wait", failed_termination)
    monkeypatch.setattr(subprocess_module, "_cleanup_io_tasks", slow_cleanup)

    async def run() -> tuple[asyncio.CancelledError, tuple[asyncio.Task, ...]]:
        blocker = asyncio.Event()
        tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(4))
        kill_task = asyncio.create_task(
            subprocess_module._kill_timed_out_process(
                object(),  # type: ignore[arg-type]
                process_group=False,
                stdin_task=tasks[0],
                stdout_task=tasks[1],
                stderr_task=tasks[2],
                wait_task=tasks[3],
            )
        )
        await cleanup_started.wait()
        kill_task.cancel("caller cancelled during I/O cleanup")
        await asyncio.sleep(0)
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await kill_task
        return exc_info.value, tasks

    cancellation, tasks = asyncio.run(run())

    assert str(cancellation) == "caller cancelled during I/O cleanup"
    assert isinstance(cancellation.__cause__, RuntimeError)
    assert "termination failed" in "\n".join(cancellation.__notes__)
    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks)


def test_cancelled_process_cleanup_falls_back_when_termination_fails(monkeypatch) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.killed = False

        def kill(self) -> None:
            self.killed = True

    async def failed_termination(process, *, process_group, wait_task):
        assert process_group is False
        raise RuntimeError("termination failed")

    monkeypatch.setattr(subprocess_module, "_kill_process_and_wait", failed_termination)

    async def run() -> tuple[FakeProcess, tuple[asyncio.Task, ...]]:
        blocker = asyncio.Event()
        tasks = tuple(asyncio.create_task(blocker.wait()) for _ in range(4))
        process = FakeProcess()
        await subprocess_module._cleanup_cancelled_process(
            process,  # type: ignore[arg-type]
            process_group=False,
            stdin_task=tasks[0],
            stdout_task=tasks[1],
            stderr_task=tasks[2],
            wait_task=tasks[3],
        )
        return process, tasks

    process, tasks = asyncio.run(run())

    assert process.killed is True
    assert all(task.done() for task in tasks)
    assert all(task.cancelled() for task in tasks)


@pytest.mark.skipif(sys.platform == "win32", reason="posix session semantics")
def test_run_subprocess_bounded_drain_when_child_leaks_pipe(tmp_path) -> None:
    # The child spawns a detached (own-session) grandchild that inherits the
    # captured stdout pipe and outlives the kill, so the stdout read would never
    # see EOF. The bounded post-kill drain must still return promptly.
    child = (
        "import sys, subprocess, time\n"
        "subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(30)'],\n"
        "    start_new_session=True,\n"
        ")\n"
        "print('parent', flush=True)\n"
        "time.sleep(30)\n"
    )
    started = time.monotonic()
    result = asyncio.run(
        run_subprocess(
            SubprocessCommand(argv=[sys.executable, "-c", child]),
            cwd=tmp_path,
            env={},
            timeout_s=1,
        )
    )
    elapsed = time.monotonic() - started

    assert result.timed_out is True
    assert "parent" in result.stdout
    # The grandchild sleeps 30s; without the bounded drain the gather would hang
    # that long. Timeout (1s) + drain bound (2s) plus margin must be well under.
    assert elapsed < 10
    assert result.stdout_truncated is True
