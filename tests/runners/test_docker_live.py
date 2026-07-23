from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
from examples._runner_conformance import verify_bounded_output_drain

from cayu import DockerRunner, ExecCommand, SearchTextTool, ToolContext

_REQUIRE_DOCKER_RUNNER_ENV_VAR = "CAYU_REQUIRE_DOCKER_RUNNER"
_SEARCH_TEXT_IMAGE = "cayu-search-text-live:local"


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


def _search_text_image(docker_path: str) -> str:
    configured = os.environ.get("CAYU_SEARCH_TEXT_DOCKER_IMAGE")
    if configured is not None:
        return configured

    dockerfile = Path(__file__).with_name("search_text_live.Dockerfile")
    try:
        subprocess.run(
            [
                docker_path,
                "build",
                "--file",
                str(dockerfile),
                "--tag",
                _SEARCH_TEXT_IMAGE,
                str(dockerfile.parent),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        _docker_unavailable(f"search-text Docker image unavailable: {exc}")
    return _SEARCH_TEXT_IMAGE


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


def test_search_text_tool_bounds_a_minified_line_in_real_docker_runner() -> None:
    docker_path = _docker_path_or_skip()
    image = _search_text_image(docker_path)
    name = f"cayu-search-text-live-{uuid4().hex[:12]}"

    async def run() -> None:
        async with await DockerRunner.create(
            name,
            image=image,
            docker_path=docker_path,
            replace=True,
            close_action="remove",
        ) as runner:
            fixture = await runner.exec(
                ExecCommand.process(
                    "python3",
                    "-c",
                    (
                        "from pathlib import Path; "
                        "Path('src').mkdir(); "
                        "Path('src/app.js').write_text('MATCH' + 'x' * 1_100_000 + '\\n')"
                    ),
                )
            )
            assert fixture.exit_code == 0

            result = await SearchTextTool(
                max_preview_bytes=80,
                max_result_bytes=200,
            ).run(
                ToolContext(session_id="docker-search-text", runner=runner),
                {"pattern": "MATCH", "mode": "content"},
            )

            assert runner.isolation == "docker"
            assert result.is_error is False
            assert len(result.content.encode("utf-8")) <= 200
            assert result.structured is not None
            match = result.structured["matches"][0]
            assert match["path"] == "src/app.js"
            assert match["line"] == 1
            assert len(match["preview"].encode("utf-8")) <= 80
            assert match["preview"].endswith("[match preview truncated]")
            assert result.structured["stdout_bytes"] < 1_024
            assert result.structured["truncation_reasons"] == ["line"]

    asyncio.run(run())
