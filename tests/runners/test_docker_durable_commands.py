from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from datetime import UTC, datetime, timedelta

import pytest

from cayu.runners import docker
from cayu.runners._subprocess import SubprocessCommand, run_subprocess
from cayu.runners.base import ExecCommand
from cayu.runners.docker import DockerRunner, _DockerRuntimeEvidence
from cayu.runners.docker_workload import DockerImageIdentity, DockerWorkloadRestrictions
from cayu.tools._runner import _durable_runner_resource_identity
from cayu.vaults import SecretRedactor

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux guest supervisor")
CONTAINER = "c" * 64


def make_runner(root):
    now = datetime.now(UTC)
    evidence = _DockerRuntimeEvidence(
        container_id=CONTAINER,
        image_id="sha256:" + "a" * 64,
        image_reference="test@sha256:" + "a" * 64,
        network_mode="none",
        default_cwd=str(root),
        runtime=None,
        seccomp_profile_sha256=None,
        restrictions=DockerWorkloadRestrictions(),
        image_identity=DockerImageIdentity(reference="test@sha256:" + "a" * 64),
        toolchain_profile_fingerprint=None,
        required_executables=("python3",),
        executable_availability=(("python3", True),),
        immutable_input_mounts=(),
        observed_at=now,
        valid_until=now + timedelta(hours=1),
    )
    return DockerRunner(
        CONTAINER,
        _container_id=CONTAINER,
        docker_path="/usr/bin/true",
        default_cwd=str(root),
        cancellation_cleanup="sandbox",
        timeout_cleanup="sandbox",
        _runtime_evidence=evidence,
    )


def install_transport(monkeypatch, root):
    """Controlled Docker transport only; run the shipped guest program for real."""
    calls = []
    monkeypatch.setattr(docker, "DOCKER_COMMAND_STATE_DIR", str(root / "receipts"))

    async def transport(command, **kwargs):
        argv = command.argv
        calls.append(tuple(argv))
        assert argv[1] == "exec", "Unexpected mutation outside the requested command"
        guest = argv[argv.index(CONTAINER) + 1 :]
        options = dict(kwargs)
        options["env"] = dict(os.environ)
        if "-w" in argv:
            options["cwd"] = argv[argv.index("-w") + 1]
        if "--env-file" in argv:
            with open(argv[argv.index("--env-file") + 1]) as stream:
                for line in stream:
                    name, value = line.rstrip("\n").split("=", 1)
                    options["env"][name] = value
        return await run_subprocess(SubprocessCommand(argv=guest), **options)

    monkeypatch.setattr(docker, "run_subprocess", transport)
    return calls


def prepare(runner, command):
    from cayu.runners._docker_command_receipt import IDENTITY_FIELDS

    identity = {
        "schema": "cayu.durable_runner_operation.v1",
        **{field: "sha256:" + "b" * 64 for field in IDENTITY_FIELDS},
    }
    identity["runner_resource_identity"] = _durable_runner_resource_identity(runner)
    receipt = runner.prepare_command_receipt(
        identity, command=command, cwd=None, env=None, timeout_s=5, output_limit_bytes=1024
    )
    assert receipt is not None
    return identity, receipt


def test_docker_runner_reopens_authenticated_result_without_redispatch(tmp_path, monkeypatch):
    calls = install_transport(monkeypatch, tmp_path)
    runner = make_runner(tmp_path)
    command = ExecCommand.process(sys.executable, "-c", "print('finished')")
    identity, receipt = prepare(runner, command)

    async def run():
        result = await runner.exec_with_receipt(
            command,
            receipt=receipt,
            redactor=SecretRedactor(),
            cwd=None,
            env=None,
            timeout_s=5,
            stdin=None,
            output_limit_bytes=1024,
        )
        assert result.stdout == "finished\n"
        replacement = make_runner(tmp_path)
        retained = json.loads(json.dumps(receipt))
        recovered = await replacement.observe_command_receipt(identity, retained)
        assert recovered == result
        from tests.docker_toolchain import docker_toolchain_profile

        from cayu import DockerCodingEnvironmentFactory, LocalWorkspace

        factory = DockerCodingEnvironmentFactory(
            source_workspace=LocalWorkspace(tmp_path),
            toolchain_profile=docker_toolchain_profile(
                image_identity=DockerImageIdentity(reference="test@sha256:" + "a" * 64)
            ),
            docker_path="/usr/bin/true",
        )
        before = len(calls)
        assert await factory.observe_command_receipt(identity, retained) == result
        assert len(calls) == before + 1
        assert "-i" not in calls[-1]  # Read only: no supervisor launch or reconnect.
        assert sum("-i" in call for call in calls) == 1
        assert all(receipt["key"] not in str(call) for call in calls)
        retained["key"] = "d" * 64
        assert await replacement.observe_command_receipt(identity, retained) is None

    asyncio.run(run())


def test_large_capture_keeps_ordinary_execution_without_receipt(tmp_path):
    from cayu.runners._docker_command_receipt import MAX_OUTPUT_BYTES

    runner = make_runner(tmp_path)
    command = ExecCommand.process(sys.executable, "-c", "print('large')")
    identity, _ = prepare(runner, command)
    assert (
        runner.prepare_command_receipt(
            identity,
            command=command,
            cwd=None,
            env=None,
            timeout_s=5,
            output_limit_bytes=MAX_OUTPUT_BYTES + 1,
        )
        is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("command", ExecCommand.process(sys.executable, "-c", "print('different')")),
        ("env", {"CHANGED": "1"}),
        ("timeout_s", 4),
        ("output_limit_bytes", 512),
        ("stdin", "unexpected"),
    ],
)
def test_prepared_request_conflict_refuses_before_dispatch(tmp_path, monkeypatch, field, value):
    calls = install_transport(monkeypatch, tmp_path)
    runner = make_runner(tmp_path)
    command = ExecCommand.process(sys.executable, "-c", "print('original')")
    _, receipt = prepare(runner, command)
    arguments = dict(
        command=command,
        receipt=receipt,
        redactor=SecretRedactor(),
        cwd=None,
        env=None,
        timeout_s=5,
        stdin=None,
        output_limit_bytes=1024,
    )
    arguments[field] = value
    with pytest.raises(ValueError, match="prepared authority"):
        asyncio.run(runner.exec_with_receipt(**arguments))
    assert calls == []


def test_runner_redacts_incomplete_secret_prefix_after_signed_capture(tmp_path, monkeypatch):
    install_transport(monkeypatch, tmp_path)
    runner = make_runner(tmp_path)
    secret = "sensitive-secret"
    command = ExecCommand.process(sys.executable, "-c", f"print({'x' * 1018 + secret!r})")
    _, receipt = prepare(runner, command)
    result = asyncio.run(
        runner.exec_with_receipt(
            command,
            receipt=copy.deepcopy(receipt),
            redactor=SecretRedactor(secret),
            cwd=None,
            env=None,
            timeout_s=5,
            stdin=None,
            output_limit_bytes=1024,
        )
    )
    assert "sensit" not in result.stdout
    assert result.stdout_truncated
