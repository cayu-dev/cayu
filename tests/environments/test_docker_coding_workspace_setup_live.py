from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding_live import _configuration_or_skip
from tests.environments.test_docker_coding_workspace_setup import (
    _CANDIDATE_PATCH,
    _GIT_IDENTITY,
    _UNPATCHED_BASELINE,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.environments.admission import ExecutionRequirements
from cayu.environments.base import EnvironmentSpec
from cayu.environments.docker_coding import DockerCodingEnvironmentFactory
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
)
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelStreamEvent
from cayu.runners.base import ExecCommand
from cayu.runners.docker_workload import DockerImageIdentity
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.requests import RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.workspaces.local import LocalWorkspace

pytestmark = pytest.mark.process


def _factory(source: Path, **kwargs) -> DockerCodingEnvironmentFactory:
    docker_path, image, image_id = _configuration_or_skip()
    architecture = subprocess.check_output(
        [docker_path, "image", "inspect", "--format", "{{.Architecture}}", image], text=True
    ).strip()
    return DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(source),
        toolchain_profile=docker_toolchain_profile(
            image_identity=DockerImageIdentity(reference=image, content_digest=image_id),
            platform_architecture=architecture,
        ),
        docker_path=docker_path,
        **kwargs,
    )


def _container_exists(docker_path: str, container_id: str) -> bool:
    inspected = subprocess.run(
        [docker_path, "container", "inspect", container_id], capture_output=True
    )
    return inspected.returncode == 0


def test_real_docker_setup_commands_amend_the_baseline(tmp_path: Path) -> None:
    (tmp_path / "calc.py").write_text("value = 2\n", encoding="utf-8")
    (tmp_path / "candidate.patch").write_text(_CANDIDATE_PATCH, encoding="utf-8")
    factory = _factory(tmp_path, workspace_setup_commands=_UNPATCHED_BASELINE)
    request = EnvironmentFactoryRequest(
        session_id="live-docker-setup",
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
        runner = environment.runner
        assert runner is not None and environment.binding is not None
        try:
            bound = await environment.binding.bind(
                environment.workspace,
                runner,
                session_id=request.session_id,
                agent_name=request.agent_name,
                environment_name=request.environment_name,
            )

            async def shell(script: str) -> str:
                executed = await runner.exec(ExecCommand.bash(script))
                assert executed.exit_code == 0, executed.stderr
                return executed.stdout

            assert await shell("git show HEAD:calc.py") == "value = 1\n"
            assert await shell("git diff --name-only") == "calc.py\n"
            assert await shell(f"git {_GIT_IDENTITY} stash -q && cat calc.py") == "value = 1\n"
            assert await shell("git stash pop -q && cat calc.py") == "value = 2\n"
            snapshot = await environment.binding.finalize(bound, outcome="completed")
            assert snapshot is not None
        finally:
            await runner.close()

    asyncio.run(run())

    assert (tmp_path / "calc.py").read_text(encoding="utf-8") == "value = 2\n"
    assert not (tmp_path / ".git").exists()


def test_real_docker_failed_setup_command_fails_the_session_and_reaps(tmp_path: Path) -> None:
    docker_path, _, _ = _configuration_or_skip()
    source = tmp_path / "source"
    source.mkdir()
    (source / "code.py").write_text("value = 1\n", encoding="utf-8")
    inner = _factory(source, workspace_setup_commands=("echo setup-broke >&2; exit 7",))
    results = []

    class Factory(EnvironmentFactory):
        @property
        def execution_profile_identity(self):
            return inner.execution_profile_identity

        def construction_admission_candidate(self):
            return inner.construction_admission_candidate()

        def execution_admission_candidate(self, request):
            return inner.execution_admission_candidate(request)

        def allocation_scope(self, request):
            return inner.allocation_scope(request)

        async def recover_finalization_disposal(self, request, state):
            await inner.recover_finalization_disposal(request, state)

        async def create(self, request):
            raise AssertionError("The runtime allocates recoverably")

        async def create_recoverable(self, request, allocation):
            result = await inner.create_recoverable(request, allocation)
            results.append(result)
            return result

        async def reap_allocation(self, request, allocation):
            await inner.reap_allocation(request, allocation)

    class Provider(ModelProvider):
        name = "setup-failure-fixture"

        @property
        def execution_profile_identity(self):
            return ExecutionProfileBehaviorIdentity(
                name=self.name, behavior_version="1", implementation_version="1"
            )

        async def stream(self, request):
            raise AssertionError("A session whose setup failed must not call the model")
            yield ModelStreamEvent.completed()

    async def scenario() -> None:
        store = SQLiteSessionStore(tmp_path / "sessions.db")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(Provider(), default=True)
        app.register_agent(AgentSpec(name="probe", model="fixture"))
        app.register_environment_factory(EnvironmentSpec(name="docker"), Factory(), default=True)
        try:
            async with asyncio.timeout(120):
                async for _ in app.run(
                    RunRequest(
                        agent_name="probe",
                        session_id="setup-failure",
                        messages=[Message.text("user", "finish")],
                        max_steps=1,
                    )
                ):
                    pass
                assert await app.drain_environment_cleanups(timeout_s=30)
            events = await store.load_events("setup-failure")
            failure = next(event for event in events if event.type == "environment.binding.failed")
            assert "workspace setup command failed (exit code 7)" in str(failure.payload)
            assert "setup-broke" in str(failure.payload)
            assert failure.payload["environment_factory_release"]["action"] == "discard"
            assert failure.payload["environment_factory_release"]["completed"] is True
            assert (await store.load("setup-failure")).status == "failed"
            assert len(results) == 1
            assert not _container_exists(docker_path, results[0].metadata["container_id"])
        finally:
            # Test hygiene only: the absence assertion above precedes this fallback.
            for result in results:
                await result.release(EnvironmentFactoryReleaseAction.DISCARD)
            await store.close()

    asyncio.run(scenario())
    assert (source / "code.py").read_text(encoding="utf-8") == "value = 1\n"
