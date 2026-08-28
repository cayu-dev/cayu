from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from cayu import (
    DockerCodingEnvironmentFactory,
    DockerImageIdentity,
    EnvironmentFactoryRequest,
    ExecCommand,
    ExecutionRequirements,
    LocalWorkspace,
)

_REQUIRE_ENV = "CAYU_REQUIRE_DOCKER_CODING"
_IMAGE_ENV = "CAYU_DOCKER_CODING_IMAGE"

pytestmark = pytest.mark.process


def _configuration_or_skip() -> tuple[str, str, str]:
    docker_path = os.environ.get("CAYU_DOCKER_PATH") or shutil.which("docker")
    image = os.environ.get(_IMAGE_ENV)
    if docker_path is None or image is None:
        _unavailable(f"docker CLI and {_IMAGE_ENV} are required")
    try:
        subprocess.run(
            [docker_path, "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        inspected = subprocess.run(
            [docker_path, "image", "inspect", "--format", "{{.Id}}", image],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception as exc:
        _unavailable(f"Docker coding image unavailable: {exc}")
    if not inspected.startswith("sha256:") or len(inspected) != 71:
        _unavailable("Docker image inspection did not return an exact content digest")
    return docker_path, image, inspected


def _unavailable(reason: str) -> None:
    if os.environ.get(_REQUIRE_ENV) == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def test_real_docker_coding_round_trip_is_bounded_and_excludes_host_git(
    tmp_path: Path,
) -> None:
    docker_path, image, image_id = _configuration_or_skip()
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    (tmp_path / ".git" / "objects" / "host-only").write_text(
        "host metadata",
        encoding="utf-8",
    )
    (tmp_path / "edit.txt").write_text("before", encoding="utf-8")
    source = LocalWorkspace(tmp_path, workspace_id="live-docker-source")
    factory = DockerCodingEnvironmentFactory(
        source_workspace=source,
        image_identity=DockerImageIdentity(reference=image, content_digest=image_id),
        docker_path=docker_path,
    )
    request = EnvironmentFactoryRequest(
        session_id="live-docker-coding",
        agent_name="coding-agent",
        environment_name="coding",
        execution_requirements=ExecutionRequirements.trusted(
            real_secret_visibility="non_possession",
            network_access="deny_by_default",
            guest_privilege="unprivileged",
            host_filesystem="isolated",
            cancellation="confirmed",
            cleanup="confirmed",
            required_executables=("git", "python3"),
        ),
    )

    async def run() -> None:
        result = await factory.create(request)
        environment = result.environment
        assert environment.runner is not None
        assert environment.binding is not None
        bound = await environment.binding.bind(
            environment.workspace,
            environment.runner,
            session_id=request.session_id,
            agent_name=request.agent_name,
            environment_name=request.environment_name,
        )
        try:
            inspection = await environment.runner.exec(
                ExecCommand.process(
                    "python3",
                    "-c",
                    "from pathlib import Path; "
                    "assert Path('.git').is_dir(); "
                    "assert not Path('.git/objects/host-only').exists(); "
                    "Path('edit.txt').write_text('after', encoding='utf-8'); "
                    "Path('new.txt').write_text('new', encoding='utf-8')",
                )
            )
            assert inspection.exit_code == 0
            snapshot = await environment.binding.finalize(
                bound,
                outcome="completed",
            )
            assert snapshot is not None
            network_probe = await environment.runner.exec(
                ExecCommand.process(
                    "python3",
                    "-c",
                    "import socket; "
                    "\ntry: socket.create_connection(('1.1.1.1', 53), timeout=0.2)"
                    "\nexcept OSError: raise SystemExit(0)"
                    "\nraise SystemExit(70)",
                )
            )
            assert network_probe.exit_code == 0
            timed_out = await environment.runner.exec(
                ExecCommand.process("python3", "-c", "import time; time.sleep(60)"),
                timeout_s=1,
            )
            assert timed_out.timed_out is True
            assert environment.runner._closed is True
        finally:
            await environment.runner.close()

        cancellation_result = await factory.create(request)
        cancellation_runner = cancellation_result.environment.runner
        assert cancellation_runner is not None
        try:
            task = asyncio.create_task(
                cancellation_runner.exec(
                    ExecCommand.process(
                        "python3",
                        "-c",
                        "import time; time.sleep(60)",
                    )
                )
            )
            await asyncio.sleep(0.2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancellation_runner._closed is True
        finally:
            await cancellation_runner.close()

    asyncio.run(run())

    assert (tmp_path / "edit.txt").read_text(encoding="utf-8") == "after"
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "new"
    assert (tmp_path / ".git" / "objects" / "host-only").read_text(
        encoding="utf-8"
    ) == "host metadata"
