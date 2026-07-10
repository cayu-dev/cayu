"""Live DockerRunner interruption check.

Requires Docker installed and running locally.

Run:
    PYTHONPATH=src .venv/bin/python examples/docker_interrupt_live.py
"""

from __future__ import annotations

import asyncio
import os

from _live_checks import require_cleanup_artifact, require_equal, require_exec_success
from cayu.runners import DockerRunner, ExecCommand


async def _line_count(runner: DockerRunner, path: str) -> int:
    result = await runner.exec(
        ExecCommand.bash(f"test -f {path} && wc -l < {path} || printf 0"),
        timeout_s=10,
    )
    if result.exit_code != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return int(result.stdout.strip() or "0")


async def _assert_stopped(runner: DockerRunner, path: str) -> None:
    first = await _line_count(runner, path)
    await asyncio.sleep(3)
    second = await _line_count(runner, path)
    print(f"{path} lines_after_interrupt first={first} second={second}")
    if second > first:
        raise RuntimeError(
            f"Interrupted Docker command is still running: {path} grew from {first} to {second}"
        )


async def main() -> None:
    name = os.environ.get("CAYU_DOCKER_NAME", "cayu-docker-interrupt-live")
    docker_path = os.environ.get("CAYU_DOCKER_PATH")
    image = os.environ.get("CAYU_DOCKER_IMAGE", "debian:stable-slim")
    print(f"container_name {name}")
    print(f"image {image}")
    print("creating container")
    runner = await DockerRunner.create(name, image=image, docker_path=docker_path, replace=True)
    print("container ready")
    try:
        cancel_log = "/workspace/cayu-cancel.log"
        cancel_command = ExecCommand.bash(
            f"i=0; while true; do echo cancel-$i >> {cancel_log}; i=$((i+1)); sleep 1; done"
        )
        task = asyncio.create_task(runner.exec(cancel_command))
        await asyncio.sleep(2)
        print("cancelling foreground command")
        task.cancel()
        try:
            await task
            raise RuntimeError("Docker command cancellation did not cancel the task")
        except asyncio.CancelledError as exc:
            artifacts = getattr(exc, "artifacts", [])
            require_cleanup_artifact(artifacts, adapter="docker", action="kill_command")
            print(f"cancel_cleanup_artifacts {artifacts}")

        probe = DockerRunner(
            name,
            default_cwd=runner.default_cwd,
            close_action="none",
            docker_path=runner.docker_path,
        )
        await _assert_stopped(probe, cancel_log)
        after_cancel = await probe.exec(ExecCommand.bash("printf after-cancel"), timeout_s=10)
        require_exec_success(after_cancel, stdout="after-cancel", label="after_cancel")
        print(f"after_cancel stdout={after_cancel.stdout!r} exit_code={after_cancel.exit_code}")

        timeout_log = "/workspace/cayu-timeout.log"
        timeout_command = ExecCommand.bash(
            f"i=0; while true; do echo timeout-$i >> {timeout_log}; i=$((i+1)); sleep 1; done"
        )
        timeout_runner = DockerRunner(
            name,
            default_cwd=runner.default_cwd,
            close_action="none",
            docker_path=runner.docker_path,
        )
        print("running timeout command")
        timeout_result = await timeout_runner.exec(timeout_command, timeout_s=2)
        print(
            "timeout_result "
            f"timed_out={timeout_result.timed_out} "
            f"exit_code={timeout_result.exit_code} "
            f"artifacts={timeout_result.artifacts}"
        )
        require_equal(timeout_result.timed_out, True, "timeout_result timed_out")
        require_cleanup_artifact(timeout_result.artifacts, adapter="docker", action="kill_command")
        timeout_probe = DockerRunner(
            name,
            default_cwd=runner.default_cwd,
            close_action="none",
            docker_path=runner.docker_path,
        )
        await _assert_stopped(timeout_probe, timeout_log)
        after_timeout = await timeout_probe.exec(
            ExecCommand.bash("printf after-timeout"), timeout_s=10
        )
        require_exec_success(after_timeout, stdout="after-timeout", label="after_timeout")
        print(f"after_timeout stdout={after_timeout.stdout!r} exit_code={after_timeout.exit_code}")
    finally:
        print("removing container")
        await runner.kill()
    print("completed")


if __name__ == "__main__":
    asyncio.run(main())
