from __future__ import annotations

import asyncio
import json
import os
import re
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

import cayu.runners.docker as docker_module
from cayu import DockerWorkloadRestrictions, ExecResult
from cayu.environments.factory import environment_factory_cleanup_settlement_task
from cayu.runners.docker import DockerRunner, DockerRuntimeConfigurationError

_PROBE_COMPLETION_TOKEN = re.compile(r"cayu-admission-probe-complete-[0-9a-f]{32}")


def _completed_probe_result(
    args: list[str],
    *,
    stdout: str = "",
    guest_exit_code: int = 0,
) -> ExecResult:
    token = next(
        (
            match.group(0)
            for value in args
            if (match := _PROBE_COMPLETION_TOKEN.search(value)) is not None
        ),
        None,
    )
    if token is None:
        return ExecResult(stdout=stdout, exit_code=guest_exit_code)
    return ExecResult(
        stdout=stdout,
        stderr=f"\n{token}:{guest_exit_code}\n",
        exit_code=0,
    )


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
                return _completed_probe_result(args, guest_exit_code=1)
            return _completed_probe_result(args, stdout=restrictions.user)
        if args[0] == "exec":
            return _completed_probe_result(args)
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
                expected = RuntimeError if failure == "cancel" else DockerRuntimeConfigurationError
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


def test_admission_probe_completion_requires_exact_untruncated_receipt() -> None:
    token = "cayu-admission-probe-complete-" + "d" * 32
    marker = f"\n{token}:73\n"
    completed = docker_module._docker_admission_probe_completion(
        ExecResult(
            stderr="probe diagnostic" + marker,
            exit_code=0,
            stderr_bytes=len(("probe diagnostic" + marker).encode()),
        ),
        completion_token=token,
    )

    assert completed is not None
    assert completed.exit_code == 73
    assert completed.stderr == "probe diagnostic"
    assert completed.stderr_bytes == len(b"probe diagnostic")

    ambiguous = (
        ExecResult(exit_code=0),
        ExecResult(stderr="\nwrong-token:73\n", exit_code=0),
        ExecResult(stderr=marker, exit_code=1),
        ExecResult(stderr=marker, exit_code=0, stderr_truncated=True),
        ExecResult(stderr=f"\n{token}:256\n", exit_code=0),
    )
    assert all(
        docker_module._docker_admission_probe_completion(
            result,
            completion_token=token,
        )
        is None
        for result in ambiguous
    )


@pytest.mark.parametrize(
    "failure_mode",
    ["cancel", "timeout", "transport_exception", "transport_result"],
)
def test_final_admission_probe_retains_guest_owner_and_fences_reconnect(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    observed_at = datetime.now(UTC)
    evidence = docker_module._DockerRuntimeEvidence(
        container_id=_CONTAINER_ID,
        image_id="sha256:" + "b" * 64,
        image_reference=_IMAGE_REFERENCE,
        network_mode="none",
        default_cwd="/workspace",
        runtime=None,
        seccomp_profile_sha256=None,
        restrictions=restrictions,
        image_identity=_image_identity(),
        toolchain_profile_fingerprint=None,
        required_executables=(),
        executable_availability=(),
        immutable_input_mounts=(),
        observed_at=observed_at,
        valid_until=observed_at + timedelta(seconds=300),
    )
    runner = DockerRunner(
        "retained-probe",
        image=_IMAGE_REFERENCE,
        default_cwd="/workspace",
        close_action="none",
        docker_path="/usr/bin/docker",
        credential_mode="trusted_tool",
        allow_raw_secret_env=False,
        cancellation_cleanup="sandbox",
        timeout_cleanup="sandbox",
        _container_id=_CONTAINER_ID,
        _runtime_evidence=evidence,
    )
    probe_dispatched = asyncio.Event()
    probe_finished = asyncio.Event()
    cleanup_dispatched = asyncio.Event()
    allow_cleanup = asyncio.Event()
    transport_error = ConnectionError("docker exec transport failed after dispatch")
    failure_active = True
    guest_active = False
    inspect_calls = 0

    async def dispatch(command, **kwargs):
        nonlocal guest_active, inspect_calls
        del kwargs
        args = command.argv[1:]
        if args[0] == "inspect":
            inspect_calls += 1
            return ExecResult(stdout=json.dumps(_inspection(restrictions)))
        if args[0] == "rm":
            return ExecResult()
        if any("read pid process_group" in value for value in args):
            cleanup_dispatched.set()
            await allow_cleanup.wait()
            guest_active = False
            probe_finished.set()
            return ExecResult()
        if args[0] == "exec" and any("id -u" in value for value in args):
            guest_active = True
            probe_dispatched.set()
            if failure_active:
                if failure_mode == "cancel":
                    await probe_finished.wait()
                elif failure_mode == "timeout":
                    return ExecResult(exit_code=-9, timed_out=True)
                elif failure_mode == "transport_exception":
                    raise transport_error
                else:
                    return ExecResult(exit_code=1, stderr="Docker stream disconnected")
            guest_active = False
            return _completed_probe_result(args, stdout=restrictions.user)
        if args[0] == "exec":
            return _completed_probe_result(args)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", dispatch)

    async def scenario() -> None:
        nonlocal failure_active
        collection = asyncio.create_task(runner.collect_execution_admission_candidate())
        await asyncio.wait_for(probe_dispatched.wait(), timeout=10)
        if failure_mode == "cancel":
            collection.cancel("cancel dispatched Docker evidence probe")
            assert collection.cancelling() == 1
            with pytest.raises(asyncio.CancelledError) as raised:
                await collection
            error: BaseException = raised.value
            assert raised.value.args == ("cancel dispatched Docker evidence probe",)
            assert collection.cancelled() is True
        elif failure_mode == "timeout":
            with pytest.raises(DockerRuntimeConfigurationError) as raised:
                await collection
            error = raised.value
            assert raised.value.code == "admission_probe_timed_out"
        elif failure_mode == "transport_exception":
            with pytest.raises(ConnectionError) as raised:
                await collection
            error = raised.value
            assert error is transport_error
        else:
            with pytest.raises(DockerRuntimeConfigurationError) as raised:
                await collection
            error = raised.value
            assert raised.value.code == "admission_probe_completion_unverified"

        settlement = environment_factory_cleanup_settlement_task(error)
        assert settlement is not None
        await asyncio.wait_for(cleanup_dispatched.wait(), timeout=10)
        assert guest_active is True
        inspections_before_reconnect = inspect_calls
        with pytest.raises(RuntimeError, match="still pending"):
            await DockerRunner.reconnect_strict(
                "competing-reconnect",
                container_id=_CONTAINER_ID,
                image_identity=_image_identity(),
                workload_restrictions=restrictions,
                docker_path="/usr/bin/docker",
            )
        assert inspect_calls == inspections_before_reconnect

        failure_active = False
        allow_cleanup.set()
        await asyncio.wait_for(asyncio.shield(settlement), timeout=10)
        await asyncio.sleep(0)
        assert guest_active is False
        reconnected = await DockerRunner.reconnect_strict(
            "settled-reconnect",
            container_id=_CONTAINER_ID,
            image_identity=_image_identity(),
            workload_restrictions=restrictions,
            docker_path="/usr/bin/docker",
        )
        await reconnected.close()

    asyncio.run(scenario())


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
