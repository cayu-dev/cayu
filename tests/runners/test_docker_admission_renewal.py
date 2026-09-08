from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.environments.test_docker_coding import (
    _CONTAINER_ID,
    _IMAGE_REFERENCE,
    _image_identity,
    _inspection,
)

from cayu import DockerWorkloadRestrictions, ExecResult
from cayu.runners.docker import DockerRunner, DockerRuntimeConfigurationError


@pytest.mark.parametrize(
    "failure", [None, "network", "image", "probe", "cancel", "seccomp", "mounts"]
)
def test_expired_strict_admission_reprobes_exact_container(monkeypatch, failure):
    restrictions = DockerWorkloadRestrictions()
    calls = []
    renewing = False

    async def dispatch(command, **kwargs):
        del kwargs
        args = command.argv[1:]
        calls.append(args)
        if args[0] == "run":
            return ExecResult(stdout=_CONTAINER_ID)
        if args[0] == "inspect":
            inspection = _inspection(restrictions)
            if renewing and failure == "network":
                inspection["HostConfig"]["NetworkMode"] = "bridge"
            if renewing and failure == "image":
                inspection["Image"] = "sha256:" + "e" * 64
            return ExecResult(stdout=json.dumps(inspection))
        if args[:2] == ["exec", _CONTAINER_ID] and "id -u" in args[-1]:
            if renewing and failure == "cancel":
                raise asyncio.CancelledError("probe cancelled")
            if renewing and failure == "probe":
                return ExecResult(exit_code=1)
            return ExecResult(stdout=restrictions.user)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", dispatch)

    async def scenario():
        nonlocal renewing
        runner = await DockerRunner.create(
            "renewal-test",
            image=_IMAGE_REFERENCE,
            image_identity=_image_identity(),
            workload_restrictions=restrictions,
            required_executables=("python3",),
            network="none",
            replace=False,
            close_action="remove",
            credential_mode="trusted_tool",
            allow_raw_secret_env=False,
            cancellation_cleanup="sandbox",
            timeout_cleanup="sandbox",
            docker_path="/usr/bin/docker",
        )
        try:
            before = len(calls)
            await runner.refresh_execution_admission()
            assert len(calls) == before
            old = runner._runtime_evidence
            assert old is not None
            expired = replace(
                old,
                observed_at=datetime.now(UTC) - timedelta(seconds=301),
                valid_until=datetime.now(UTC) - timedelta(seconds=1),
            )
            if failure == "seccomp":
                expired = replace(expired, seccomp_profile_sha256="sha256:" + "d" * 64)
            if failure == "mounts":
                expired = replace(
                    expired, immutable_input_mounts=(("sha256:" + "d" * 64, "/evidence"),)
                )
            runner._runtime_evidence = expired
            fingerprint = runner.execution_capability_evidence().environment_fingerprint
            renewing = True
            if failure:
                expected = (
                    asyncio.CancelledError
                    if failure == "cancel"
                    else DockerRuntimeConfigurationError
                )
                with pytest.raises(expected):
                    await runner.refresh_execution_admission()
                assert runner._runtime_evidence is expired
            else:
                await asyncio.gather(*(runner.refresh_execution_admission() for _ in range(3)))
                assert runner._runtime_evidence.valid_until > datetime.now(UTC)
                assert runner.execution_capability_evidence().environment_fingerprint == fingerprint
                assert sum(args[0] == "inspect" for args in calls[before:]) == 1
                assert all(_CONTAINER_ID in args for args in calls[before:])
        finally:
            await runner.close()

    asyncio.run(scenario())


def test_unverified_runner_cannot_mint_live_evidence():
    runner = DockerRunner("unverified", docker_path="/usr/bin/docker")
    asyncio.run(runner.refresh_execution_admission())
    assert runner._runtime_evidence is None


@pytest.mark.skipif(
    not os.environ.get("CAYU_DOCKER_ADMISSION_TEST_IMAGE"),
    reason="requires an explicitly selected local immutable Python Docker image",
)
def test_live_strict_container_renews_expired_admission():
    from cayu import DockerCodingToolchainProfile, DockerImageIdentity, ExecCommand
    from cayu.environments.docker_toolchains import (
        ensure_docker_coding_toolchain_runner_admission,
    )
    from cayu.tools._runner import InvocationRunnerHandle

    image = os.environ["CAYU_DOCKER_ADMISSION_TEST_IMAGE"]
    profile = DockerCodingToolchainProfile(
        profile_id="admission-renewal",
        revision="1",
        image_identity=DockerImageIdentity(reference=image, content_digest=image),
        platform_architecture="amd64",
    )

    async def scenario():
        runner = await DockerRunner.create(
            "cayu-renewal-" + uuid4().hex,
            image=image,
            image_identity=profile.image_identity,
            workload_restrictions=profile.restrictions,
            required_executables=profile.required_executables,
            toolchain_profile_fingerprint=profile.fingerprint,
            network="none",
            replace=False,
            close_action="remove",
            credential_mode="trusted_tool",
            allow_raw_secret_env=False,
            cancellation_cleanup="sandbox",
            timeout_cleanup="sandbox",
        )
        try:
            old = runner._runtime_evidence
            assert old is not None
            runner._runtime_evidence = replace(
                old,
                observed_at=datetime.now(UTC) - timedelta(seconds=301),
                valid_until=datetime.now(UTC) - timedelta(seconds=1),
            )
            handle = InvocationRunnerHandle(runner, redactor_snapshot_provider=lambda: None)
            assert (
                await ensure_docker_coding_toolchain_runner_admission(handle, profile=profile)
                is None
            )
            result = await runner.exec(ExecCommand.process("python3", "-c", "print('renewed')"))
            assert result.exit_code == 0
            assert result.stdout.strip() == "renewed"
            assert (
                runner.execution_capability_evidence().environment_fingerprint
                == old.environment_fingerprint
            )
        finally:
            await runner.close()

    asyncio.run(scenario())
