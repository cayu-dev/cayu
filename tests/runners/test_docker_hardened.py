"""Hardened Docker containers: workload restrictions without strict evidence."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from tests.runners.test_docker_admission_renewal import _completed_probe_result

from cayu.environments.admission import ExecutionRequirements, evaluate_execution_admission
from cayu.runners.base import ExecResult
from cayu.runners.docker import DockerRunner, DockerRuntimeConfigurationError
from cayu.runners.docker_workload import (
    DockerImageIdentity,
    DockerTmpfsMount,
    DockerWorkloadRestrictions,
)

_CONTAINER_ID = "e" * 64
_NETWORK = "cayu-egress-net-hardened"


def _inspection(restrictions: DockerWorkloadRestrictions, **host_overrides: Any) -> dict:
    args = restrictions.run_args()
    tmpfs = {
        args[index + 1].split(":", 1)[0]: args[index + 1].split(":", 1)[1]
        for index, value in enumerate(args)
        if value == "--tmpfs"
    }
    host = {
        "NetworkMode": _NETWORK,
        "Privileged": False,
        "ReadonlyRootfs": restrictions.read_only_root,
        "PidsLimit": restrictions.pids_limit,
        "Memory": restrictions.memory_bytes,
        "MemorySwap": restrictions.memory_swap_bytes,
        "CpuPeriod": restrictions.cpu_period_us,
        "CpuQuota": restrictions.cpu_quota_us,
        "ShmSize": restrictions.shm_size_bytes,
        "SecurityOpt": ["no-new-privileges"] if restrictions.no_new_privileges else [],
        "CapDrop": ["ALL"],
        "CapAdd": list(restrictions.capability_add),
        "Tmpfs": tmpfs,
        # The egress CA is a read-only bind; hardening does not police host mounts.
        "Binds": [],
        "Mounts": [{"Type": "bind", "Source": "/tmp/ca.pem", "Target": "/etc/cayu/ca.pem"}],
        "Devices": [],
        "DeviceRequests": [],
    }
    host.update(host_overrides)
    return {
        "Id": _CONTAINER_ID,
        "Image": "sha256:" + "f" * 64,
        "Config": {
            "Image": "python:3.12-slim",
            "User": restrictions.user,
            "Env": [f"{name}={value}" for name, value in restrictions.home_environment.items()],
        },
        "HostConfig": host,
    }


def _fake_docker(
    restrictions: DockerWorkloadRestrictions,
    calls: list[list[str]],
    **host_overrides: Any,
):
    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        docker_args = command.argv[1:]
        if docker_args[0] == "run":
            return ExecResult(stdout=_CONTAINER_ID + "\n")
        if docker_args[0] == "inspect":
            return ExecResult(stdout=json.dumps(_inspection(restrictions, **host_overrides)))
        if docker_args[0] == "exec" and "id -u" in docker_args[-1]:
            return _completed_probe_result(docker_args, stdout=restrictions.user)
        return _completed_probe_result(docker_args)

    return fake_run_subprocess


def _create(restrictions: DockerWorkloadRestrictions, **options: Any) -> DockerRunner:
    return asyncio.run(
        DockerRunner.create(
            "hardened",
            image=options.pop("image", "python:3.12-slim"),
            network=_NETWORK,
            ca_mount=options.pop("ca_mount", None),
            close_action="remove",
            credential_mode="virtual_egress",
            env_overlay={"HTTPS_PROXY": "http://cayu-egress:8080"},
            workload_restrictions=restrictions,
            docker_path="/usr/bin/docker",
            **options,
        )
    )


def test_restrictions_harden_a_networked_container_without_strict_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []
    monkeypatch.setattr("cayu.runners.docker.run_subprocess", _fake_docker(restrictions, calls))
    ca = tmp_path / "ca.pem"
    ca.write_text("ca")

    runner = _create(
        restrictions,
        ca_mount=(str(ca), "/etc/cayu/ca.pem"),
        setup_commands=("python3 -m venv /workspace/.venv",),
    )

    run_argv = next(argv for argv in calls if argv[1] == "run")
    for expected in (
        ["--read-only"],
        ["--cap-drop", "ALL"],
        ["--security-opt", "no-new-privileges=true"],
        ["--user", restrictions.user],
        ["--pids-limit", str(restrictions.pids_limit)],
        ["--network", _NETWORK],
    ):
        assert any(run_argv[i : i + len(expected)] == expected for i in range(len(run_argv)))
    # Root without capabilities cannot write the user's tmpfs or signal its
    # processes, so no Cayu helper runs as root in a hardened container.
    execs = [argv for argv in calls if argv[1] == "exec"]
    assert execs
    assert not any(argv[2:4] == ["-u", "root"] for argv in execs)
    assert ["-u", restrictions.user, _CONTAINER_ID, "sh", "-c", "mkdir -p /workspace"] == (
        next(argv for argv in execs if argv[-1] == "mkdir -p /workspace")[2:]
    )
    setup = next(argv for argv in execs if argv[-1] == "python3 -m venv /workspace/.venv")
    assert setup[2:4] == ["-u", restrictions.user]
    assert any(argv[1] == "inspect" for argv in calls)
    assert runner.container_id == _CONTAINER_ID
    assert runner.workload_restrictions == restrictions

    evidence = runner.execution_capability_evidence()
    for capability in ("guest_privilege_containment", "unprivileged_guest"):
        claim = evidence.claim_for(capability)
        assert claim is not None
        assert claim.state == "available"
        assert {detail.name for detail in claim.adapter_details} >= {
            "read_only_root",
            "no_new_privileges",
            "cap_drop_all",
            "pids_limit",
            "memory_bytes",
        }
    # Hardening says nothing about kernel isolation, the network or host files.
    for capability in (
        "untrusted_code_isolation",
        "deny_by_default_network",
        "host_filesystem_isolation",
    ):
        assert evidence.state_for(capability) == "unsupported"
    decision = evaluate_execution_admission(
        candidate="docker",
        requirements=ExecutionRequirements.trusted(guest_privilege="unprivileged"),
        evidence=evidence,
        stage="pre_exposure",
    )
    assert decision.status == "admitted"

    # The strict, evidence-bearing path still demands network='none'.
    pinned = "python@sha256:" + "a" * 64
    with pytest.raises(ValueError, match="requires network='none'"):
        _create(
            restrictions,
            image=pinned,
            image_identity=DockerImageIdentity(reference=pinned),
            replace=False,
        )


def test_hardened_container_drift_fails_closed_and_removes_exact_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "cayu.runners.docker.run_subprocess",
        _fake_docker(restrictions, calls, ReadonlyRootfs=False),
    )

    with pytest.raises(DockerRuntimeConfigurationError) as caught:
        _create(restrictions)

    assert caught.value.code == "read_only_root_drift"
    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_weakened_hardening_claims_no_privilege_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restrictions = DockerWorkloadRestrictions(read_only_root=False, capability_add=("CHOWN",))
    monkeypatch.setattr("cayu.runners.docker.run_subprocess", _fake_docker(restrictions, []))

    evidence = _create(restrictions).execution_capability_evidence()

    claim = evidence.claim_for("guest_privilege_containment")
    assert claim is not None
    assert claim.state == "unsupported"
    assert claim.reason_code == "docker_privilege_restrictions_weakened"


def test_hardened_cwd_must_be_writable_tmpfs(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unexpected_dispatch(*args: Any, **kwargs: Any) -> ExecResult:
        pytest.fail("Invalid hardened configuration reached Docker.")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", unexpected_dispatch)

    with pytest.raises(ValueError, match="inside a configured tmpfs"):
        _create(DockerWorkloadRestrictions(), default_cwd="/srv/app")


def test_hardened_cutover_fence_runs_as_the_restricted_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []
    monkeypatch.setattr("cayu.runners.docker.run_subprocess", _fake_docker(restrictions, calls))
    runner = _create(restrictions)
    calls.clear()

    asyncio.run(runner.fence_guest_processes_for_egress_cutover())

    assert calls[0][1:6] == ["exec", "-u", restrictions.user, _CONTAINER_ID, "python3"]


def test_executable_tmpfs_mounts_say_exec_because_docker_defaults_to_noexec() -> None:
    restrictions = DockerWorkloadRestrictions(
        tmpfs=(
            DockerTmpfsMount(target="/data", size_bytes=1024 * 1024, noexec=True),
            DockerTmpfsMount(target="/tmp", size_bytes=1024 * 1024, mode=0o1777, noexec=False),
        )
    )
    args = restrictions.run_args()
    options = {
        args[index + 1].split(":", 1)[0]: args[index + 1].split(":", 1)[1].split(",")
        for index, value in enumerate(args)
        if value == "--tmpfs"
    }

    assert "exec" in options["/tmp"]
    assert "noexec" not in options["/tmp"]
    assert "noexec" in options["/data"]
    assert "exec" not in options["/data"]


def test_custom_tmpfs_mounts_are_noexec_unless_they_opt_out() -> None:
    # A caller's mount stays non-executable unless it asks otherwise; the
    # default /tmp and /workspace mounts opt out explicitly.
    data = DockerTmpfsMount(target="/data", size_bytes=1024 * 1024)
    venv = DockerTmpfsMount(
        target="/workspace", size_bytes=1024 * 1024 * 1024, mode=0o750, noexec=False
    )
    restrictions = DockerWorkloadRestrictions(
        tmpfs=(DockerTmpfsMount(target="/tmp", size_bytes=1024 * 1024, mode=0o1777), data, venv)
    )

    assert data.noexec is True
    assert data == DockerTmpfsMount(target="/data", size_bytes=1024 * 1024, noexec=True)
    assert all(not default.noexec for default in DockerWorkloadRestrictions().tmpfs)
    args = restrictions.run_args()
    options = {
        args[index + 1].split(":", 1)[0]: args[index + 1].split(":", 1)[1].split(",")
        for index, value in enumerate(args)
        if value == "--tmpfs"
    }
    assert options["/data"][-1] == "noexec"
    assert options["/tmp"][-1] == "noexec"
    assert options["/workspace"][-1] == "exec"


def test_tmpfs_mounts_are_sorted_by_target() -> None:
    tmp = DockerTmpfsMount(target="/tmp", size_bytes=1024 * 1024, mode=0o1777)
    workspace = DockerTmpfsMount(target="/workspace", size_bytes=1024 * 1024)
    cache = DockerTmpfsMount(target="/cache", size_bytes=1024 * 1024)

    restrictions = DockerWorkloadRestrictions(tmpfs=[workspace, tmp, cache])

    assert restrictions.tmpfs == (cache, tmp, workspace)
    assert restrictions == DockerWorkloadRestrictions(tmpfs=(cache, tmp, workspace))
    assert (
        restrictions.fingerprint
        == DockerWorkloadRestrictions(tmpfs=(tmp, cache, workspace)).fingerprint
    )
    with pytest.raises(ValueError, match="targets must be unique"):
        DockerWorkloadRestrictions(tmpfs=(workspace, tmp, workspace))


def test_containers_created_before_explicit_exec_still_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    legacy_tmpfs = {
        target: ",".join(option for option in options.split(",") if option != "exec")
        for target, options in _inspection(restrictions)["HostConfig"]["Tmpfs"].items()
    }
    monkeypatch.setattr(
        "cayu.runners.docker.run_subprocess",
        _fake_docker(restrictions, [], Tmpfs=legacy_tmpfs),
    )

    assert _create(restrictions).workload_restrictions == restrictions

    noexec_tmpfs = {
        target: options.replace("exec", "noexec")
        for target, options in _inspection(restrictions)["HostConfig"]["Tmpfs"].items()
    }
    monkeypatch.setattr(
        "cayu.runners.docker.run_subprocess",
        _fake_docker(restrictions, [], Tmpfs=noexec_tmpfs),
    )
    with pytest.raises(DockerRuntimeConfigurationError) as caught:
        _create(restrictions)
    assert caught.value.code == "tmpfs_options_drift"


@pytest.mark.parametrize("drift", [False, True])
def test_hardened_seccomp_verifies_daemon_json(monkeypatch, tmp_path, drift):
    profile = {"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []}
    path = tmp_path / "seccomp.json"
    path.write_text(json.dumps(profile, indent=2))
    inspected = {**profile, "defaultAction": "SCMP_ACT_ALLOW"} if drift else profile
    restrictions = DockerWorkloadRestrictions()
    calls = []
    monkeypatch.setattr(
        "cayu.runners.docker.run_subprocess",
        _fake_docker(
            restrictions,
            calls,
            SecurityOpt=["no-new-privileges", "seccomp=" + json.dumps(inspected)],
        ),
    )
    if drift:
        with pytest.raises(DockerRuntimeConfigurationError) as caught:
            _create(restrictions, seccomp_profile=str(path))
        assert caught.value.code == "seccomp_profile_drift"
        assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]
    else:
        assert _create(restrictions, seccomp_profile=str(path)).container_id == _CONTAINER_ID


def test_egress_fence_ignores_guest_python_startup_hooks(monkeypatch, tmp_path):
    import os
    import subprocess
    import sys

    hook = tmp_path / "sitecustomize.py"
    hook.write_text("import os; os._exit(0)")
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}
    probe = "print('fence ran')"
    assert subprocess.check_output([sys.executable, "-c", probe], env=environment) == b""
    calls = []

    async def dispatch(command, **kwargs):
        calls.append(command.argv)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", dispatch)
    runner = DockerRunner("guest", docker_path="/usr/bin/docker")
    asyncio.run(runner.fence_guest_processes_for_egress_cutover())
    args = calls[0]
    flags = args[args.index("python3") + 1 : -1]
    assert (
        subprocess.check_output([sys.executable, *flags, probe], env=environment) == b"fence ran\n"
    )
