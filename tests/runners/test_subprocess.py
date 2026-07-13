from __future__ import annotations

import asyncio
import sys
import time

import pytest

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
