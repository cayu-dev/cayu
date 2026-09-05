from __future__ import annotations

import asyncio
import json
from itertools import count
from pathlib import Path
from typing import Any

import pytest

from cayu import (
    DockerImageIdentity,
    DockerWorkloadRestrictions,
    ImmutableInputStore,
    inspect_local_immutable_input,
)
from cayu.runners.base import ExecResult
from cayu.runners.docker import DockerRunner, DockerRuntimeConfigurationError

_IMAGE_REFERENCE = "cayu/immutable-input-test@sha256:" + ("c" * 64)
_IMAGE_ID = "sha256:" + ("b" * 64)


def _sha(character: str) -> str:
    return "sha256:" + (character * 64)


def _source(root: Path, *, target_path: str = "/opt/cayu/inputs/runtime"):
    return inspect_local_immutable_input(
        root,
        target_path=target_path,
        policy_fingerprint=_sha("a"),
        runtime_compatibility_fingerprint=_sha("b"),
        authorization_scope_fingerprint=_sha("c"),
    )


def _inspection(
    restrictions: DockerWorkloadRestrictions,
    *,
    container_id: str,
    source_path: str,
    target_path: str,
) -> dict[str, Any]:
    tmpfs: dict[str, str] = {}
    args = restrictions.run_args()
    for index, value in enumerate(args):
        if value == "--tmpfs":
            target, options = args[index + 1].split(":", 1)
            tmpfs[target] = options
    return {
        "Id": container_id,
        "Image": _IMAGE_ID,
        "Config": {
            "Image": _IMAGE_REFERENCE,
            "User": restrictions.user,
            "Env": [f"{name}={value}" for name, value in restrictions.home_environment.items()],
        },
        "HostConfig": {
            "NetworkMode": "none",
            "Privileged": False,
            "ReadonlyRootfs": restrictions.read_only_root,
            "PidsLimit": restrictions.pids_limit,
            "Memory": restrictions.memory_bytes,
            "MemorySwap": restrictions.memory_swap_bytes,
            "CpuPeriod": restrictions.cpu_period_us,
            "CpuQuota": restrictions.cpu_quota_us,
            "ShmSize": restrictions.shm_size_bytes,
            "SecurityOpt": (["no-new-privileges"] if restrictions.no_new_privileges else []),
            "CapDrop": ["ALL"],
            "CapAdd": list(restrictions.capability_add),
            "Tmpfs": tmpfs,
            "Binds": [],
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": source_path,
                    "Target": target_path,
                    "ReadOnly": True,
                }
            ],
            "Devices": [],
            "DeviceRequests": [],
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": source_path,
                "Destination": target_path,
                "RW": False,
            }
        ],
    }


def test_100_docker_environments_share_one_verified_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "runtime.bin").write_bytes(b"immutable-runtime" * 4096)
    source = _source(source_root)
    store = ImmutableInputStore(tmp_path / "managed")
    restrictions = DockerWorkloadRestrictions()
    image_identity = DockerImageIdentity(reference=_IMAGE_REFERENCE, content_digest=_IMAGE_ID)
    identifiers = count(1)
    calls: list[list[str]] = []
    mount_source: str | None = None

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        nonlocal mount_source
        del kwargs
        calls.append(command.argv)
        docker_args = command.argv[1:]
        if docker_args[0] == "run":
            mount_argument = docker_args[docker_args.index("--mount") + 1]
            fields = dict(
                field.split("=", 1) for field in mount_argument.split(",") if "=" in field
            )
            mount_source = fields["source"]
            return ExecResult(stdout=f"{next(identifiers):064x}\n")
        if docker_args[0] == "inspect":
            assert mount_source is not None
            return ExecResult(
                stdout=json.dumps(
                    _inspection(
                        restrictions,
                        container_id=docker_args[-1],
                        source_path=mount_source,
                        target_path=source.projection.target_path,
                    )
                )
            )
        if docker_args[0] == "exec" and "id -u" in docker_args[-1]:
            return ExecResult(stdout=restrictions.user)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    async def run() -> tuple[list[DockerRunner], list[str]]:
        attachments = await asyncio.gather(
            *(
                store.attach(
                    source,
                    attachment_id=f"environment:{index}",
                    owner_id=f"session:{index}",
                )
                for index in range(100)
            )
        )
        runners = await asyncio.gather(
            *(
                DockerRunner.create(
                    f"immutable-input-{index}",
                    image=_IMAGE_REFERENCE,
                    image_identity=image_identity,
                    workload_restrictions=restrictions,
                    immutable_input_mounts=(attachment.docker_mount(),),
                    network="none",
                    replace=False,
                    close_action="remove",
                    credential_mode="trusted_tool",
                    allow_raw_secret_env=False,
                    cancellation_cleanup="sandbox",
                    timeout_cleanup="sandbox",
                    docker_path="/usr/bin/docker",
                )
                for index, attachment in enumerate(attachments)
            )
        )
        evidence_states = [
            runner.execution_capability_evidence().claim_for("read_only_host_inputs").state  # type: ignore[union-attr]
            for runner in runners
        ]
        await asyncio.gather(*(runner.close() for runner in runners))
        await asyncio.gather(*(store.release(item.attachment_id) for item in attachments))
        return list(runners), evidence_states

    runners, evidence_states = asyncio.run(run())
    diagnostic = store.inspect()[0]
    run_calls = [call for call in calls if call[1] == "run"]
    mounted_sources = {
        call[call.index("--mount") + 1].split("source=", 1)[1].split(",", 1)[0]
        for call in run_calls
    }

    assert len(runners) == 100
    assert evidence_states == ["live_verified"] * 100
    assert len(run_calls) == 100
    assert len(mounted_sources) == 1
    assert diagnostic.reference_count == 0
    assert diagnostic.attachment_count == 100
    assert diagnostic.reuse_count == 99
    assert diagnostic.cleanup_state == "eligible"
    assert len(tuple((store.root / "objects").iterdir())) == 1


def test_docker_refuses_exposure_when_root_can_write_immutable_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "input.txt").write_text("content", encoding="utf-8")
    source = _source(source_root)
    store = ImmutableInputStore(tmp_path / "managed")
    attachment = store.attach_sync(source, attachment_id="environment:1", owner_id="session:1")
    restrictions = DockerWorkloadRestrictions()
    container_id = "1" * 64
    calls: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        docker_args = command.argv[1:]
        if docker_args[0] == "run":
            return ExecResult(stdout=container_id)
        if docker_args[0] == "inspect":
            return ExecResult(
                stdout=json.dumps(
                    _inspection(
                        restrictions,
                        container_id=container_id,
                        source_path=str(attachment.materialization_path),
                        target_path=source.projection.target_path,
                    )
                )
            )
        if "cayu-immutable-input-probe" in docker_args:
            return ExecResult(exit_code=73)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(DockerRuntimeConfigurationError) as caught:
        asyncio.run(
            DockerRunner.create(
                "mutable-input",
                image=_IMAGE_REFERENCE,
                image_identity=DockerImageIdentity(
                    reference=_IMAGE_REFERENCE,
                    content_digest=_IMAGE_ID,
                ),
                workload_restrictions=restrictions,
                immutable_input_mounts=(attachment.docker_mount(),),
                network="none",
                replace=False,
                close_action="remove",
                credential_mode="trusted_tool",
                allow_raw_secret_env=False,
                cancellation_cleanup="sandbox",
                timeout_cleanup="sandbox",
                docker_path="/usr/bin/docker",
            )
        )

    assert caught.value.code == "immutable_input_write_not_refused"
    assert calls[-1][1:] == ["rm", "-f", container_id]
