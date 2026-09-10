from __future__ import annotations

import asyncio
import json

import pytest
from tests.docker_toolchain import docker_toolchain_profile
from tests.environments.test_docker_coding import _CONTAINER_ID, _image_identity, _inspection
from tests.runners.test_docker_admission_renewal import _completed_probe_result

from cayu import (
    DockerCodingEnvironmentFactory,
    DockerWorkloadRestrictions,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
)
from cayu.runners import DockerRunner, ExecCommand, ExecResult
from cayu.runners._creation_cleanup import settle_creation_cleanup
from cayu.workspaces import LocalWorkspace


@pytest.mark.anyio
@pytest.mark.parametrize("removal_fails", [False, True])
async def test_factory_never_reconnects_container_owned_by_rollback(
    monkeypatch, tmp_path, removal_fails
):
    restrictions = DockerWorkloadRestrictions()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    allocated = False
    fail_setup = True
    repaired = False
    removals = 0
    inspections = 0
    creates = 0

    async def bounded_rollback(*args, **kwargs):
        await settle_creation_cleanup(*args, **{**kwargs, "timeout_s": 0.05})

    async def docker(command, **kwargs):
        nonlocal allocated, fail_setup, removals, inspections, creates
        args = command.argv[1:]
        if args[:2] == ["container", "ls"]:
            return ExecResult(stdout=_CONTAINER_ID if allocated else "")
        if args[0] == "run":
            creates += 1
            allocated = True
            return ExecResult(stdout=_CONTAINER_ID)
        if args[:2] == ["rm", "-f"]:
            assert args[2] == _CONTAINER_ID
            removals += 1
            cleaning.set()
            await release.wait()
            if removal_fails and not repaired:
                raise PermissionError("removal denied")
            allocated = False
            return ExecResult()
        if args[0] == "exec" and fail_setup:
            fail_setup = False
            raise ConnectionError("transient setup failure")
        if args[0] == "inspect":
            inspections += 1
            return ExecResult(stdout=json.dumps(_inspection(restrictions)))
        if any(".cayu-toolchain-write-probe" in arg for arg in args):
            return _completed_probe_result(args, stdout="linux/amd64\n")
        if "id -u" in args[-1]:
            return _completed_probe_result(args, stdout=restrictions.user)
        return _completed_probe_result(args)

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", docker)
    monkeypatch.setattr("cayu.runners.docker.settle_creation_cleanup", bounded_rollback)
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(
            image_identity=_image_identity(), restrictions=restrictions
        ),
        docker_path="docker",
    )
    request = EnvironmentFactoryRequest(
        session_id="rollback-session", agent_name="agent", environment_name="coding"
    )
    task = asyncio.create_task(factory.create(request))
    try:
        await asyncio.wait_for(cleaning.wait(), 1)
        with pytest.raises(RuntimeError, match="rollback is still pending"):
            await task
        # Also exercise the existing-container path on a later public request.
        with pytest.raises(RuntimeError, match="rollback is still pending"):
            await factory.create(request)
        assert allocated and creates == 1 and inspections == 0 and removals == 1
        exact_handle = DockerRunner(_CONTAINER_ID, docker_path="docker")
        with pytest.raises(RuntimeError, match="rollback is still pending"):
            await exact_handle.exec(ExecCommand.process("true"))
        release.set()
        remaining = await DockerRunner.drain_failed_creations(timeout_s=1)
        if removal_fails:
            assert remaining == 1
            with pytest.raises(RuntimeError, match="rollback is still pending"):
                await factory.create(request)
            repaired = True
            assert await DockerRunner.drain_failed_creations(timeout_s=1) == 0
        else:
            assert remaining == 0
        assert not allocated
        result = await factory.create(request)
        assert allocated and creates == 2
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)
        assert not allocated
    finally:
        repaired = True
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await DockerRunner.drain_failed_creations(timeout_s=1) == 0
