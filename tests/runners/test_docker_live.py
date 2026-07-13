from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from uuid import uuid4

import pytest
from examples._runner_conformance import verify_bounded_output_drain

from cayu import DockerRunner, ExecCommand

_REQUIRE_DOCKER_RUNNER_ENV_VAR = "CAYU_REQUIRE_DOCKER_RUNNER"


def _docker_path_or_skip() -> str:
    docker_path = os.environ.get("CAYU_DOCKER_PATH") or shutil.which("docker")
    if docker_path is None:
        _docker_unavailable("docker CLI not found")
    try:
        subprocess.run(
            [docker_path, "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        _docker_unavailable(f"docker daemon unavailable: {exc}")
    return docker_path


def _docker_unavailable(reason: str) -> None:
    if os.environ.get(_REQUIRE_DOCKER_RUNNER_ENV_VAR):
        pytest.fail(reason)
    pytest.skip(reason)


def test_real_docker_runner_executes_and_cleans_up_timed_out_command() -> None:
    docker_path = _docker_path_or_skip()
    image = os.environ.get("CAYU_DOCKER_LIVE_IMAGE", "alpine:3.20")
    name = f"cayu-docker-live-{uuid4().hex[:12]}"

    async def run() -> None:
        async with await DockerRunner.create(
            name,
            image=image,
            docker_path=docker_path,
            replace=True,
            close_action="remove",
        ) as runner:
            ok = await runner.exec(ExecCommand.process("sh", "-c", "printf docker-live-ok"))
            assert ok.exit_code == 0
            assert ok.stdout == "docker-live-ok"
            assert ok.timed_out is False

            await verify_bounded_output_drain(runner, adapter="docker")

            timed_out = await runner.exec(ExecCommand.process("sh", "-c", "sleep 30"), timeout_s=1)
            assert timed_out.timed_out is True
            assert timed_out.exit_code != 0
            assert timed_out.artifacts == [
                {
                    "type": "cayu.runner_cleanup.v1",
                    "adapter": "docker",
                    "action": "kill_command",
                    "status": "completed",
                    "timeout_s": 5.0,
                }
            ]

    asyncio.run(run())
