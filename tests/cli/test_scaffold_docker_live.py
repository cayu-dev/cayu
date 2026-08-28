"""Opt-in built-image proof for the generated Docker coding composition."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

from cayu import (
    DockerCodingEnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    ExecCommand,
    InMemorySessionStore,
    InMemoryTaskStore,
    RunCheckTool,
    ScriptedModelProvider,
)
from cayu.cli import main
from cayu.cli.project import project_context
from cayu.core.tools import ToolContext
from cayu.tools import EditFileTool, GitChangesTool, WriteFileTool

_WHEEL_ENV = "CAYU_DOCKER_GENERATED_CODING_WHEEL"
_REQUIRE_ENV = "CAYU_REQUIRE_GENERATED_DOCKER_CODING"
_BASE_IMAGE = (
    "python:3.12.11-slim-bookworm@"
    "sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"
)
_UV_VERSION = "0.9.28"
_DEBIAN_SNAPSHOT = "20250930T000000Z"
_DEBIAN_SUITE = "bookworm"
_GIT_PACKAGE = "1:2.39.5-0+deb12u2"
_RIPGREP_PACKAGE = "13.0.0-4+b2"

pytestmark = pytest.mark.process


def _configuration_or_skip() -> tuple[str, str, Path]:
    docker = os.environ.get("CAYU_DOCKER_PATH") or shutil.which("docker")
    uv = os.environ.get("CAYU_UV_PATH") or shutil.which("uv")
    wheel_raw = os.environ.get(_WHEEL_ENV)
    if docker is None or uv is None or wheel_raw is None:
        _unavailable(f"Docker, uv, and {_WHEEL_ENV} are required")
    wheel = Path(wheel_raw)
    if not wheel.is_file() or wheel.suffix != ".whl":
        _unavailable(f"{_WHEEL_ENV} must select a built wheel")
    try:
        subprocess.run(
            [docker, "info"],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=20,
        )
    except Exception as exc:
        _unavailable(f"Docker is unavailable: {type(exc).__name__}")
    return docker, uv, wheel


def _unavailable(reason: str) -> NoReturn:
    if os.environ.get(_REQUIRE_ENV) == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def test_built_wheel_generated_docker_path_fails_repairs_passes_and_copies_back(
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    docker, uv, wheel = _configuration_or_skip()
    project_parent = tmp_path / "generated"
    project_parent.mkdir()
    assert (
        main(
            [
                "new",
                "live-docker-coder",
                "--composition",
                "coding",
                "--coding-execution",
                "docker",
                "--dir",
                str(project_parent),
            ]
        )
        == 0
    )
    project = project_parent / "live-docker-coder"
    wheel_target = project / ".cayu" / wheel.name
    wheel_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wheel, wheel_target)
    wheel_digest = "sha256:" + hashlib.sha256(wheel_target.read_bytes()).hexdigest()
    image = (
        "cayu-generated-coding-test:"
        + hashlib.sha256(str(project).encode("utf-8")).hexdigest()[:16]
    )
    (project / "docker-coding-build.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "image_reference": image,
                "base_image": _BASE_IMAGE,
                "uv_version": _UV_VERSION,
                "debian_snapshot": _DEBIAN_SNAPSHOT,
                "debian_suite": _DEBIAN_SUITE,
                "git_package": _GIT_PACKAGE,
                "ripgrep_package": _RIPGREP_PACKAGE,
                "cayu_wheel": f".cayu/{wheel.name}",
                "cayu_wheel_sha256": wheel_digest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [uv, "lock"],
        cwd=project,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
        timeout=120,
    )
    built = subprocess.run(
        [sys.executable, "build_coding_image.py"],
        cwd=project,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert built.returncode == 0, (built.stdout + built.stderr)[-4000:]
    image_configuration = json.loads(
        (project / "docker-coding-image.json").read_text(encoding="utf-8")
    )
    assert image_configuration["reference"] == image
    image_id = image_configuration["content_digest"]
    assert isinstance(image_id, str) and image_id.startswith("sha256:")

    def remove_image() -> None:
        subprocess.run(
            [docker, "image", "rm", image],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )

    request.addfinalizer(remove_image)

    spec = importlib.util.spec_from_file_location("live_generated_app", project / "app.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with project_context(project):
        spec.loader.exec_module(module)
        app = module.build_app(
            provider=ScriptedModelProvider([]),
            session_store=InMemorySessionStore(),
            task_store=InMemoryTaskStore(),
        )

    registered_environment = app._environments["coding"]
    factory = registered_environment.factory
    assert isinstance(factory, DockerCodingEnvironmentFactory)
    assert factory.docker_path == docker
    registered_primary = app._agents["live-docker-coder"]
    check_tool = registered_primary.tools["run_check"].tool
    assert isinstance(check_tool, RunCheckTool)
    environment_request = EnvironmentFactoryRequest(
        session_id="generated-live-docker",
        agent_name="live-docker-coder",
        environment_name="coding",
        execution_requirements=registered_primary.execution_requirements,
    )

    async def exercise() -> None:
        result = await factory.create(environment_request)
        environment = result.environment
        assert environment.runner is not None
        assert environment.binding is not None
        container_id = result.metadata["container_id"]
        assert container_id == environment.runner.container_id
        inspected = subprocess.run(
            [docker, "inspect", "--format", "{{json .}}", container_id],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        inspection = json.loads(inspected.stdout)
        host_configuration = inspection["HostConfig"]
        restrictions = factory.restrictions
        assert inspection["Config"]["User"] == restrictions.user
        assert host_configuration["NetworkMode"] == "none"
        assert host_configuration["ReadonlyRootfs"] is True
        assert host_configuration["Privileged"] is False
        assert host_configuration["CapDrop"] == ["ALL"]
        assert not host_configuration["Binds"]
        assert host_configuration["PidsLimit"] == restrictions.pids_limit
        assert host_configuration["Memory"] == restrictions.memory_bytes
        assert host_configuration["MemorySwap"] == restrictions.memory_swap_bytes
        assert host_configuration["CpuPeriod"] == restrictions.cpu_period_us
        assert host_configuration["CpuQuota"] == restrictions.cpu_quota_us
        bound = await environment.binding.bind(
            environment.workspace,
            environment.runner,
            session_id=environment_request.session_id,
            agent_name=environment_request.agent_name,
            environment_name=environment_request.environment_name,
        )
        assert bound.workspace is not None
        context = ToolContext(
            session_id=environment_request.session_id,
            agent_name=environment_request.agent_name,
            environment_name=environment_request.environment_name,
            idempotency_key="generated-live-docker-tools",
            workspace=bound.workspace,
            runner=environment.runner,
        )
        try:
            identity = await environment.runner.exec(ExecCommand.process("id", "-u"))
            assert identity.exit_code == 0
            assert identity.stdout.strip() == "1000"
            network = await environment.runner.exec(
                ExecCommand.process(
                    "python3",
                    "-c",
                    "import socket; "
                    "\ntry: socket.create_connection(('1.1.1.1', 53), timeout=0.2)"
                    "\nexcept OSError: raise SystemExit(0)"
                    "\nraise SystemExit(70)",
                )
            )
            assert network.exit_code == 0

            test_path = "tests/test_project.py"
            observed = await bound.workspace.read_bytes(test_path)
            assert observed.revision is not None
            needle = "def test_generated_project_smoke() -> None:"
            injected = "def test_live_generated_failure() -> None:\n    assert False\n\n\n" + needle
            edited = await EditFileTool().run(
                context,
                {
                    "path": test_path,
                    "expected_revision": observed.revision,
                    "edits": [{"old_text": needle, "new_text": injected}],
                },
            )
            assert edited.is_error is False
            failed = await check_tool.run(context, {"check": "test"})
            assert failed.structured is not None
            assert failed.structured["status"] == "failed"

            current = await bound.workspace.read_bytes(test_path)
            assert current.revision is not None
            repaired = await EditFileTool().run(
                context,
                {
                    "path": test_path,
                    "expected_revision": current.revision,
                    "edits": [{"old_text": "    assert False", "new_text": "    assert True"}],
                },
            )
            assert repaired.is_error is False
            passed = await check_tool.run(context, {"check": "test"})
            assert passed.structured is not None
            assert passed.structured["status"] == "passed", {
                "stdout": passed.structured["stdout"],
                "stderr": passed.structured["stderr"],
            }
            written = await WriteFileTool().run(
                context,
                {
                    "path": "live_copyback.txt",
                    "content": "generated Docker live proof\n",
                    "mode": "create",
                },
            )
            assert written.is_error is False
            changes = await GitChangesTool().run(
                context,
                {"mode": "diff", "scope": "unstaged"},
            )
            assert "test_live_generated_failure" in changes.content
            snapshot = await environment.binding.finalize(bound, outcome="completed")
            assert snapshot is not None
        finally:
            if result.release is not None:
                await result.release(EnvironmentFactoryReleaseAction.DISCARD)
        removed = subprocess.run(
            [docker, "inspect", container_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
        assert removed.returncode != 0

    asyncio.run(exercise())

    assert (project / "live_copyback.txt").read_text(encoding="utf-8") == (
        "generated Docker live proof\n"
    )
    assert "assert True" in (project / "tests/test_project.py").read_text(encoding="utf-8")
