from __future__ import annotations

import asyncio
import io
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from examples._runner_conformance import verify_bounded_output_drain

from cayu import (
    DockerImageIdentity,
    DockerRunner,
    DockerTmpfsMount,
    DockerWorkloadRestrictions,
    ExecCommand,
    ImmutableInputStore,
    SearchTextTool,
    ToolContext,
    inspect_local_immutable_input,
)

_REQUIRE_DOCKER_RUNNER_ENV_VAR = "CAYU_REQUIRE_DOCKER_RUNNER"
_SEARCH_TEXT_IMAGE = "cayu-search-text-live:local"

pytestmark = pytest.mark.process


class _FailingBinarySink(io.BytesIO):
    def write(self, content: Any, /) -> int:
        del content
        raise OSError("intentional live Docker sink failure")


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

            stream_payload = b"binary\x00payload\xff"
            stream_output = io.BytesIO()
            streamed = await runner.exec_stream(
                ExecCommand.process("cat"),
                stdin=io.BytesIO(stream_payload),
                stdout=stream_output,
                stdout_limit_bytes=len(stream_payload),
            )
            assert streamed.exit_code == 0
            assert streamed.stdout == ""
            assert streamed.stdout_bytes == len(stream_payload)
            assert stream_output.getvalue() == stream_payload

            marker = f"/tmp/cayu-stream-failure-{uuid4().hex}.pid"
            with pytest.raises(
                OSError, match="intentional live Docker sink failure"
            ) as stream_failure:
                await runner.exec_stream(
                    ExecCommand.process(
                        "sh",
                        "-c",
                        f"echo $$ > {marker}; printf x; while :; do sleep 1; done",
                    ),
                    stdout=_FailingBinarySink(),
                    stdout_limit_bytes=1,
                )
            assert getattr(stream_failure.value, "artifacts", None) == [
                {
                    "type": "cayu.runner_cleanup.v1",
                    "adapter": "docker",
                    "action": "kill_command",
                    "status": "completed",
                    "timeout_s": 5.0,
                }
            ]
            settled = await runner.exec(
                ExecCommand.process(
                    "sh",
                    "-c",
                    (
                        f"if ! test -f {marker}; then exit 0; fi; "
                        f"pid=$(cat {marker}); "
                        'if ! kill -0 "$pid" 2>/dev/null; then exit 0; fi; '
                        "state=$(awk '{print $3}' \"/proc/$pid/stat\"); "
                        'test "$state" = Z'
                    ),
                )
            )
            assert settled.exit_code == 0
            await runner.exec(ExecCommand.process("rm", "-f", marker))

            shell_failure = await runner.exec(
                ExecCommand.bash("printf 'validation failed\\n'; exit 23")
            )
            assert shell_failure.exit_code == 23
            assert shell_failure.stdout == "validation failed\n"
            assert shell_failure.timed_out is False

            process_failure = await runner.exec(ExecCommand.process("sh", "-c", "exit 17"))
            assert process_failure.exit_code == 17
            assert process_failure.timed_out is False

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


def test_100_real_docker_runners_share_one_immutable_input(tmp_path: Path) -> None:
    docker_path = _docker_path_or_skip()
    image = os.environ.get("CAYU_DOCKER_LIVE_IMAGE", "alpine:3.20")
    try:
        image_id = subprocess.run(
            [docker_path, "image", "inspect", "--format", "{{.Id}}", image],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception as exc:
        _docker_unavailable(f"Docker image unavailable for immutable fan-out: {exc}")
    source_root = tmp_path / "source"
    source_root.mkdir()
    expected = b"one physical immutable input\n"
    (source_root / "input.txt").write_bytes(expected)
    source = inspect_local_immutable_input(
        source_root,
        target_path="/opt/cayu/inputs/runtime",
        policy_fingerprint="sha256:" + ("a" * 64),
        runtime_compatibility_fingerprint="sha256:" + ("b" * 64),
        authorization_scope_fingerprint="sha256:" + ("c" * 64),
    )
    store = ImmutableInputStore(tmp_path / "managed")
    restrictions = DockerWorkloadRestrictions(
        pids_limit=16,
        memory_bytes=32 * 1024 * 1024,
        memory_swap_bytes=32 * 1024 * 1024,
        shm_size_bytes=1024 * 1024,
        tmpfs=(
            DockerTmpfsMount(
                target="/tmp",
                size_bytes=1024 * 1024,
                mode=0o1777,
                noexec=False,
            ),
            DockerTmpfsMount(
                target="/workspace",
                size_bytes=1024 * 1024,
                mode=0o750,
                noexec=False,
            ),
        ),
    )
    image_identity = DockerImageIdentity(reference=image, content_digest=image_id)

    async def run() -> None:
        attachments = await asyncio.gather(
            *(
                store.attach(
                    source,
                    attachment_id=f"live-docker:{index}",
                    owner_id=f"live-session:{index}",
                )
                for index in range(100)
            )
        )
        runners: list[DockerRunner] = []
        create_slots = asyncio.Semaphore(10)

        async def create_one(index: int) -> None:
            async with create_slots:
                runner = await DockerRunner.create(
                    f"cayu-immutable-live-{uuid4().hex[:12]}",
                    image=image,
                    image_identity=image_identity,
                    workload_restrictions=restrictions,
                    immutable_input_mounts=(attachments[index].docker_mount(),),
                    network="none",
                    replace=False,
                    close_action="remove",
                    credential_mode="trusted_tool",
                    allow_raw_secret_env=False,
                    cancellation_cleanup="sandbox",
                    timeout_cleanup="sandbox",
                    docker_path=docker_path,
                )
                runners.append(runner)

        try:
            outcomes = await asyncio.gather(
                *(create_one(index) for index in range(100)),
                return_exceptions=True,
            )
            failures = [value for value in outcomes if isinstance(value, BaseException)]
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup("Docker immutable fan-out failed.", failures)
            io_slots = asyncio.Semaphore(10)

            async def read_one(runner: DockerRunner):
                async with io_slots:
                    return await runner.exec(
                        ExecCommand.process(
                            "sh",
                            "-c",
                            'test "$(cat /opt/cayu/inputs/runtime/input.txt)" = '
                            '"one physical immutable input"',
                        )
                    )

            reads = await asyncio.gather(*(read_one(runner) for runner in runners))
            assert all(result.exit_code == 0 for result in reads)
            assert all(
                runner.execution_capability_evidence().claim_for("read_only_host_inputs").state
                == "live_verified"
                for runner in runners
            )
            diagnostic = store.inspect()[0]
            assert diagnostic.reference_count == 100
            assert diagnostic.reuse_count == 99
            assert len(tuple((store.root / "objects").iterdir())) == 1
        finally:
            close_slots = asyncio.Semaphore(10)

            async def close_one(runner: DockerRunner) -> None:
                async with close_slots:
                    await runner.close()

            await asyncio.gather(*(close_one(runner) for runner in runners))
            await asyncio.gather(*(store.release(item.attachment_id) for item in attachments))

    asyncio.run(run())

    diagnostic = store.inspect()[0]
    assert diagnostic.reference_count == 0
    assert diagnostic.cleanup_state == "eligible"


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
