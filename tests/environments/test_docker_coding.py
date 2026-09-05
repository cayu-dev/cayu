from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from tests.docker_toolchain import docker_toolchain_profile

import cayu.environments.docker_coding as docker_coding_module
from cayu import (
    DockerCodingCommandAuthority,
    DockerCodingDependencyInput,
    DockerCodingEnvironmentFactory,
    DockerCodingToolchainError,
    DockerCodingToolchainProfile,
    DockerCodingWorkspaceBinding,
    DockerImageIdentity,
    DockerWorkloadRestrictions,
    DockerWorkspaceTransferLimits,
    EnvironmentAllocationContext,
    EnvironmentAllocationIntent,
    EnvironmentAllocationState,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    ExecCommand,
    ExecutionAdmissionError,
    ExecutionRequirements,
    ImmutableInputProjectionCapability,
    ImmutableInputStore,
    LocalRunner,
    SyncBinding,
    SyncBindingSourceConflictError,
    evaluate_execution_admission,
    inspect_local_immutable_input,
)
from cayu._coding_product_authority import (
    CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY,
    CodingProductSourceCopyAuthority,
)
from cayu.runners.base import ExecResult
from cayu.runners.docker import (
    DockerContainerOwnershipError,
    DockerRunner,
    DockerRuntimeConfigurationError,
)
from cayu.workspaces import LocalWorkspace, RunnerWorkspace, WorkspaceMutationResult
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)

_CONTAINER_ID = "a" * 64
_IMAGE_ID = "sha256:" + ("b" * 64)
_IMAGE_REFERENCE = "cayu/coding@sha256:" + ("c" * 64)


class _TestAllocationContext(EnvironmentAllocationContext):
    def __init__(
        self,
        intent: EnvironmentAllocationIntent,
        *,
        state: EnvironmentAllocationState = EnvironmentAllocationState.UNPREPARED,
    ) -> None:
        self._intent = intent
        self._state = state
        self._acknowledgement: dict[str, Any] | None = None

    @property
    def intent(self) -> EnvironmentAllocationIntent:
        return self._intent

    @property
    def state(self) -> EnvironmentAllocationState:
        return self._state

    @property
    def acknowledged_reconnect_metadata(self) -> dict[str, Any] | None:
        return self._acknowledgement

    async def prepare(
        self,
        provider_metadata: Mapping[str, Any],
    ) -> EnvironmentAllocationIntent:
        self._intent = self._intent.with_provider_metadata(provider_metadata)
        self._state = EnvironmentAllocationState.PREPARED
        return self._intent

    async def mark_dispatched(self) -> None:
        self._state = EnvironmentAllocationState.DISPATCHED

    async def acknowledge(self, reconnect_metadata: Mapping[str, Any]) -> None:
        self._acknowledgement = dict(reconnect_metadata)
        self._state = EnvironmentAllocationState.ACKNOWLEDGED

    async def mark_reaping(self) -> bool:
        self._state = EnvironmentAllocationState.REAPING
        return True

    async def mark_reaped(self) -> None:
        self._state = EnvironmentAllocationState.REAPED


def _image_identity() -> DockerImageIdentity:
    return DockerImageIdentity(reference=_IMAGE_REFERENCE)


def _custom_toolchain_profile(
    *,
    dependency_content: bytes | None = None,
) -> DockerCodingToolchainProfile:
    dependencies = (
        ()
        if dependency_content is None
        else (
            DockerCodingDependencyInput(
                path="Cargo.lock",
                content_sha256="sha256:" + sha256(dependency_content).hexdigest(),
            ),
        )
    )
    return DockerCodingToolchainProfile(
        profile_id="custom-rust",
        revision="1",
        image_identity=_image_identity(),
        platform_architecture="amd64",
        command_authorities=(
            DockerCodingCommandAuthority(
                selector="cargo-check",
                revision="1",
                description="Run the admitted Rust check.",
                exposure="named_check",
                executable="/usr/bin/cargo",
                fixed_arguments=("check",),
                max_arguments=0,
                timeout_seconds=120,
            ),
        ),
        dependency_inputs=dependencies,
    )


def _inspection(
    restrictions: DockerWorkloadRestrictions,
    *,
    network_mode: str = "none",
    immutable_mount: tuple[str, str] | None = None,
) -> dict[str, Any]:
    tmpfs: dict[str, str] = {}
    args = restrictions.run_args()
    for index, value in enumerate(args):
        if value == "--tmpfs":
            target, options = args[index + 1].split(":", 1)
            tmpfs[target] = options
    return {
        "Id": _CONTAINER_ID,
        "Image": _IMAGE_ID,
        "Config": {
            "Image": _IMAGE_REFERENCE,
            "User": restrictions.user,
            "Env": [f"{name}={value}" for name, value in restrictions.home_environment.items()],
        },
        "HostConfig": {
            "NetworkMode": network_mode,
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
            "Mounts": (
                []
                if immutable_mount is None
                else [
                    {
                        "Type": "bind",
                        "Source": immutable_mount[0],
                        "Target": immutable_mount[1],
                        "ReadOnly": True,
                    }
                ]
            ),
            "Devices": [],
            "DeviceRequests": [],
        },
        "Mounts": (
            []
            if immutable_mount is None
            else [
                {
                    "Type": "bind",
                    "Source": immutable_mount[0],
                    "Destination": immutable_mount[1],
                    "RW": False,
                }
            ]
        ),
    }


class _LocalDockerRunner(DockerRunner):
    """Exercise Docker binding control flow against a local test directory."""

    def __init__(self, root: Path) -> None:
        super().__init__(
            "local-docker-test",
            default_cwd="/workspace",
            docker_path="/usr/bin/docker",
            _container_id=_CONTAINER_ID,
        )
        self.local = LocalRunner(root, inherit_env=False)
        self._test_root = root

    def resolve_cwd(self, cwd: str | None = None) -> str:
        del cwd
        return str(self._test_root)

    async def exec(self, command, **kwargs: Any):
        kwargs["cwd"] = None
        return await self.local.exec(command, **kwargs)

    async def exec_stream(self, command, **kwargs: Any):
        kwargs["cwd"] = None
        return await self.local.exec_stream(command, **kwargs)

    async def close(self) -> None:
        await self.local.close()
        self._closed = True


def test_docker_runner_workspace_partial_read_drops_complete_file_identity(
    tmp_path: Path,
) -> None:
    target_root = tmp_path / "target"
    target_root.mkdir()
    (target_root / "candidate.patch").write_bytes(b"0123456789abcdef")
    workspace = RunnerWorkspace(
        _LocalDockerRunner(target_root),
        workspace_id="docker-target",
        python_executable=sys.executable,
    )

    result = asyncio.run(workspace.read_bytes("candidate.patch", max_bytes=4))

    assert result.content == b"0123"
    assert result.total_bytes == 16
    assert result.truncated is True
    assert result.offset == 0
    assert result.revision is None
    assert result.sha256 is None
    assert result.git_mode is None


def _coding_product_test_binding(
    source_root: Path,
    target_root: Path,
) -> tuple[
    _LocalDockerRunner,
    LocalWorkspace,
    DockerCodingWorkspaceBinding,
]:
    runner = _LocalDockerRunner(target_root)
    source = LocalWorkspace(
        source_root,
        workspace_id="source",
        excluded_directory_names=(".cayu", ".git", ".runtime"),
    )
    observation_limits = WorkspaceRevisionObservationLimits()
    admitted = asyncio.run(
        observe_deterministic_workspace(
            source,
            observer="cayu-coding-product-source",
            limits=observation_limits,
        )
    )
    assert admitted.revision is not None
    target = RunnerWorkspace(
        runner,
        workspace_id="target",
        python_executable=sys.executable,
        excluded_directory_names=(".cayu", ".git", ".runtime"),
    )
    binding = DockerCodingWorkspaceBinding(
        target_workspace=target,
        limits=DockerWorkspaceTransferLimits(
            max_file_bytes=1024,
            max_total_bytes=4096,
            max_archive_bytes=64 * 1024,
        ),
        source_copy_authority=CodingProductSourceCopyAuthority(
            request_fingerprint="sha256:" + "a" * 64,
            source_workspace_id=source.id,
            baseline_revision=admitted.revision,
            observation_limits=observation_limits,
        ),
    )
    return runner, source, binding


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX executable modes")
def test_docker_coding_binding_preserves_executable_modes_both_directions(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source_script = source_root / "script.sh"
    source_script.write_bytes(b"#!/bin/sh\n")
    source_script.chmod(0o755)
    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="mode-round-trip")
        target_script = target_root / "script.sh"
        assert target_script.stat().st_mode & 0o777 == 0o755
        target_script.chmod(0o644)
        generated = target_root / "generated.sh"
        generated.write_bytes(b"#!/bin/sh\n")
        generated.chmod(0o755)
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None

    asyncio.run(run())

    assert source_script.stat().st_mode & 0o777 == 0o644
    assert (source_root / "generated.sh").stat().st_mode & 0o777 == 0o755


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX executable modes")
def test_docker_coding_copy_back_rejects_source_mode_drift(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source_script = source_root / "script.sh"
    source_script.write_bytes(b"#!/bin/sh\n")
    source_script.chmod(0o644)
    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="mode-conflict")
        source_script.chmod(0o755)
        (target_root / "script.sh").write_bytes(b"#!/bin/sh\necho changed\n")
        with pytest.raises(SyncBindingSourceConflictError):
            await binding.finalize(bound, outcome="completed")
        binding.abandon(bound)

    asyncio.run(run())

    assert source_script.read_bytes() == b"#!/bin/sh\n"
    assert source_script.stat().st_mode & 0o777 == 0o755


def test_strict_docker_runner_uses_typed_restrictions_and_exact_live_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        if any(".cayu-toolchain-write-probe" in arg for arg in command.argv):
            return ExecResult(stdout="linux/amd64\n")
        docker_args = command.argv[1:]
        if docker_args[0] == "run":
            return ExecResult(stdout=_CONTAINER_ID + "\n")
        if docker_args[0] == "inspect":
            return ExecResult(stdout=json.dumps(_inspection(restrictions)))
        if docker_args[:2] == ["exec", _CONTAINER_ID] and "id -u" in docker_args[-1]:
            return ExecResult(stdout=restrictions.user)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    async def run() -> DockerRunner:
        runner = await DockerRunner.create(
            "strict-coding",
            image_identity=_image_identity(),
            workload_restrictions=restrictions,
            required_executables=("python3", "git"),
            image=_IMAGE_REFERENCE,
            network="none",
            replace=False,
            close_action="remove",
            credential_mode="trusted_tool",
            allow_raw_secret_env=False,
            cancellation_cleanup="sandbox",
            timeout_cleanup="sandbox",
            docker_path="/usr/bin/docker",
        )
        await runner.close()
        return runner

    runner = asyncio.run(run())
    evidence = runner.execution_capability_evidence()
    run_argv = next(call for call in calls if call[1] == "run")

    assert runner.container_id == _CONTAINER_ID
    assert runner.resource_key == ("docker", _CONTAINER_ID)
    assert "--read-only" in run_argv
    assert run_argv[run_argv.index("--network") + 1] == "none"
    assert run_argv[run_argv.index("--cap-drop") + 1] == "ALL"
    assert "--mount" not in run_argv
    assert run_argv[run_argv.index("--entrypoint") + 1] == "sleep"
    assert run_argv[-2:] == [_IMAGE_REFERENCE, "infinity"]
    assert evidence.environment_fingerprint is not None
    assert evidence.image_fingerprint == _image_identity().fingerprint
    assert evidence.claim_for("untrusted_code_isolation").state == "unsupported"  # type: ignore[union-attr]
    assert evidence.claim_for("deny_by_default_network").state == "live_verified"  # type: ignore[union-attr]
    assert evidence.tool_requirements is not None
    assert [item.executable for item in evidence.tool_requirements.executables] == [
        "git",
        "python3",
    ]
    assert all(item.state == "live_verified" for item in evidence.tool_requirements.executables)
    assert calls[-1][1:3] == ["rm", "-f"]
    assert calls[-1][3] == _CONTAINER_ID


def test_weakened_docker_restrictions_do_not_claim_privilege_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restrictions = DockerWorkloadRestrictions(
        read_only_root=False,
        no_new_privileges=False,
        capability_add=("SYS_ADMIN",),
    )

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        if any(".cayu-toolchain-write-probe" in arg for arg in command.argv):
            return ExecResult(stdout="linux/amd64\n")
        docker_args = command.argv[1:]
        if docker_args[0] == "run":
            return ExecResult(stdout=_CONTAINER_ID + "\n")
        if docker_args[0] == "inspect":
            return ExecResult(stdout=json.dumps(_inspection(restrictions)))
        if docker_args[:2] == ["exec", _CONTAINER_ID] and "id -u" in docker_args[-1]:
            return ExecResult(stdout=restrictions.user)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    async def run() -> DockerRunner:
        runner = await DockerRunner.create(
            "weakened-coding",
            image_identity=_image_identity(),
            workload_restrictions=restrictions,
            image=_IMAGE_REFERENCE,
            network="none",
            replace=False,
            close_action="remove",
            credential_mode="trusted_tool",
            allow_raw_secret_env=False,
            cancellation_cleanup="sandbox",
            timeout_cleanup="sandbox",
            docker_path="/usr/bin/docker",
        )
        await runner.close()
        return runner

    evidence = asyncio.run(run()).execution_capability_evidence()
    requirements = ExecutionRequirements.trusted(guest_privilege="unprivileged")
    decision = evaluate_execution_admission(
        candidate="docker",
        requirements=requirements,
        evidence=evidence,
        stage="pre_exposure",
    )

    containment_claim = evidence.claim_for("guest_privilege_containment")
    unprivileged_claim = evidence.claim_for("unprivileged_guest")
    assert containment_claim is not None
    assert unprivileged_claim is not None
    assert containment_claim.state == "unsupported"
    assert unprivileged_claim.state == "unsupported"
    assert decision.status == "refused"


def test_strict_docker_runner_fails_closed_and_cleans_exact_container_on_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        if command.argv[1] == "run":
            return ExecResult(stdout=_CONTAINER_ID)
        if command.argv[1] == "inspect":
            return ExecResult(stdout=json.dumps(_inspection(restrictions, network_mode="bridge")))
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(DockerRuntimeConfigurationError) as caught:
        asyncio.run(
            DockerRunner.create(
                "drifted-coding",
                image_identity=_image_identity(),
                workload_restrictions=restrictions,
                image=_IMAGE_REFERENCE,
                network="none",
                replace=False,
                credential_mode="trusted_tool",
                allow_raw_secret_env=False,
                cancellation_cleanup="sandbox",
                timeout_cleanup="sandbox",
                docker_path="/usr/bin/docker",
            )
        )

    assert caught.value.code == "network_mode_drift"
    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_strict_docker_runner_rejects_changed_local_image_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    image_reference = "cayu-coding:local"
    expected_image_id = "sha256:" + ("d" * 64)
    calls: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        if command.argv[1] == "run":
            return ExecResult(stdout=_CONTAINER_ID)
        if command.argv[1] == "inspect":
            inspection = _inspection(restrictions)
            inspection["Config"]["Image"] = image_reference
            return ExecResult(stdout=json.dumps(inspection))
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(DockerRuntimeConfigurationError) as caught:
        asyncio.run(
            DockerRunner.create(
                "changed-local-image",
                image_identity=DockerImageIdentity(
                    reference=image_reference,
                    content_digest=expected_image_id,
                ),
                workload_restrictions=restrictions,
                image=image_reference,
                network="none",
                replace=False,
                credential_mode="trusted_tool",
                allow_raw_secret_env=False,
                cancellation_cleanup="sandbox",
                timeout_cleanup="sandbox",
                docker_path="/usr/bin/docker",
            )
        )

    assert caught.value.code == "image_content_digest_mismatch"
    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_strict_docker_runner_never_name_cleans_ambiguous_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        return ExecResult(stdout="not-an-exact-container-id")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(DockerContainerOwnershipError):
        asyncio.run(
            DockerRunner.create(
                "ambiguous-coding",
                image_identity=_image_identity(),
                workload_restrictions=DockerWorkloadRestrictions(),
                image=_IMAGE_REFERENCE,
                network="none",
                replace=False,
                credential_mode="trusted_tool",
                allow_raw_secret_env=False,
                cancellation_cleanup="sandbox",
                timeout_cleanup="sandbox",
                docker_path="/usr/bin/docker",
            )
        )

    assert len(calls) == 1
    assert calls[0][1] == "run"


def test_exact_container_owner_is_used_for_timeout_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        if command.argv[1] == "exec":
            return ExecResult(exit_code=-9, timed_out=True)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = DockerRunner(
        "diagnostic-name",
        docker_path="/usr/bin/docker",
        close_action="remove",
        timeout_cleanup="sandbox",
        cancellation_cleanup="sandbox",
        _container_id=_CONTAINER_ID,
    )

    result = asyncio.run(runner.exec(ExecCommand.process("sleep", "999"), timeout_s=1))

    assert result.timed_out is True
    assert runner._closed is True
    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_exact_container_owner_is_used_for_cancellation_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        if command.argv[1] == "exec":
            raise asyncio.CancelledError
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    runner = DockerRunner(
        "diagnostic-name",
        docker_path="/usr/bin/docker",
        close_action="remove",
        timeout_cleanup="sandbox",
        cancellation_cleanup="sandbox",
        _container_id=_CONTAINER_ID,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.exec(ExecCommand.process("sleep", "999")))

    assert runner._closed is True
    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_docker_coding_factory_rejects_untrusted_before_docker_allocation(
    tmp_path: Path,
) -> None:
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
    )
    request = EnvironmentFactoryRequest(
        session_id="session",
        agent_name="agent",
        environment_name="coding",
        execution_requirements=ExecutionRequirements.untrusted(),
    )

    candidate = factory.execution_admission_candidate(request)
    decision = evaluate_execution_admission(
        candidate=candidate.candidate,
        requirements=request.execution_requirements,
        evidence=candidate.evidence,
        stage="pre_create",
    )
    assert decision.status == "refused"
    assert any(refusal.capability == "untrusted_code_isolation" for refusal in decision.refusals)
    with pytest.raises(ExecutionAdmissionError) as caught:
        asyncio.run(factory.create(request))
    assert any(
        refusal.capability == "untrusted_code_isolation"
        for refusal in caught.value.decision.refusals
    )


def test_docker_coding_factory_publicly_constructs_request_bound_binding(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    source = LocalWorkspace(
        source_root,
        workspace_id="factory-source",
        excluded_directory_names=(".cayu", ".git", ".runtime"),
    )
    limits = WorkspaceRevisionObservationLimits()
    observed = asyncio.run(
        observe_deterministic_workspace(
            source,
            observer="cayu-coding-product-source",
            limits=limits,
        )
    )
    assert observed.revision is not None
    authority = CodingProductSourceCopyAuthority(
        request_fingerprint="sha256:" + ("a" * 64),
        source_workspace_id=source.id,
        baseline_revision=observed.revision,
        observation_limits=limits,
    )
    request = EnvironmentFactoryRequest(
        session_id="request-bound-binding",
        agent_name="agent",
        environment_name="coding",
        metadata={CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY: authority.model_dump(mode="json")},
    )
    factory = DockerCodingEnvironmentFactory(
        source_workspace=source,
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
    )
    runner = _LocalDockerRunner(target_root)
    target = RunnerWorkspace(
        runner,
        workspace_id="factory-target",
        excluded_directory_names=(".cayu", ".git", ".runtime"),
    )

    binding = factory.create_workspace_binding(request, target_workspace=target)
    bound = asyncio.run(binding.bind(source, runner, session_id=request.session_id))

    assert isinstance(binding, DockerCodingWorkspaceBinding)
    assert (target_root / "example.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert binding.abandon(bound) is True

    mismatched = replace(
        request,
        metadata={
            CODING_PRODUCT_SOURCE_AUTHORITY_METADATA_KEY: authority.model_copy(
                update={"source_workspace_id": "other-source"}
            ).model_dump(mode="json")
        },
    )
    with pytest.raises(RuntimeError, match="another source workspace"):
        factory.create_workspace_binding(mismatched, target_workspace=target)


def test_toolchain_dependency_drift_refuses_before_docker_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "Cargo.lock").write_bytes(b"changed\n")
    allocation_attempted = False

    async def unexpected_create(*args: Any, **kwargs: Any) -> DockerRunner:
        nonlocal allocation_attempted
        del args, kwargs
        allocation_attempted = True
        raise AssertionError("Docker allocation must not start for a stale toolchain.")

    monkeypatch.setattr(DockerRunner, "create", unexpected_create)
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=_custom_toolchain_profile(dependency_content=b"locked\n"),
    )

    with pytest.raises(DockerCodingToolchainError) as caught:
        asyncio.run(
            factory.create(
                EnvironmentFactoryRequest(
                    session_id="stale",
                    agent_name="agent",
                    environment_name="coding",
                )
            )
        )

    assert caught.value.code == "dependency_inputs_changed"
    assert allocation_attempted is False


def test_explicit_toolchain_platform_drift_cleans_exact_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        docker_args = command.argv[1:]
        if docker_args[0] == "run":
            return ExecResult(stdout=_CONTAINER_ID)
        if docker_args[0] == "inspect":
            return ExecResult(stdout=json.dumps(_inspection(restrictions)))
        if docker_args[:2] == ["exec", _CONTAINER_ID] and "id -u" in docker_args[-1]:
            return ExecResult(stdout=restrictions.user)
        if docker_args[:2] == ["exec", _CONTAINER_ID] and any(
            "platform.system" in item for item in docker_args
        ):
            return ExecResult(stdout="linux/arm64\n")
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=_custom_toolchain_profile(),
        docker_path="/usr/bin/docker",
    )

    with pytest.raises(DockerCodingToolchainError) as caught:
        asyncio.run(
            factory.create(
                EnvironmentFactoryRequest(
                    session_id="platform-drift",
                    agent_name="agent",
                    environment_name="coding",
                )
            )
        )

    assert caught.value.code == "platform_mismatch"
    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_docker_coding_factory_declares_bounded_tools_and_immutable_identity(
    tmp_path: Path,
) -> None:
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(
            image_identity=_image_identity(), required_executables=("ruff",)
        ),
        transfer_limits=DockerWorkspaceTransferLimits(max_files=42),
    )
    candidate = factory.construction_admission_candidate()

    assert candidate.candidate == "docker"
    assert candidate.evidence.environment_fingerprint is not None
    assert candidate.evidence.image_fingerprint == _image_identity().fingerprint
    assert candidate.evidence.tool_requirements is not None
    assert [item.executable for item in candidate.evidence.tool_requirements.executables] == [
        "/usr/bin/ruff",
        "git",
        "python3",
        "rm",
        "sh",
        "sleep",
    ]
    assert all(
        item.state == "declared" for item in candidate.evidence.tool_requirements.executables
    )
    assert factory.execution_profile_identity.implementation_version.startswith("sha256:")
    default_factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
        transfer_limits=DockerWorkspaceTransferLimits(max_files=42),
    )
    assert (
        factory.execution_profile_identity.implementation_version
        != default_factory.execution_profile_identity.implementation_version
    )


def test_docker_coding_factory_binds_immutable_projection_identity(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    input_root = tmp_path / "runtime"
    input_root.mkdir()
    (input_root / "runtime.bin").write_bytes(b"runtime")
    runtime_compatibility = "sha256:" + ("d" * 64)
    immutable_input = inspect_local_immutable_input(
        input_root,
        target_path="/opt/cayu/inputs/runtime",
        policy_fingerprint="sha256:" + ("a" * 64),
        runtime_compatibility_fingerprint=runtime_compatibility,
        authorization_scope_fingerprint="sha256:" + ("c" * 64),
    )
    store = ImmutableInputStore(tmp_path / "managed")
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(workspace_root),
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
        immutable_inputs=(immutable_input,),
        immutable_input_store=store,
        immutable_input_runtime_compatibility_fingerprint=runtime_compatibility,
    )
    default_factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(workspace_root),
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
    )
    claim = factory.construction_admission_candidate().evidence.claim_for("read_only_host_inputs")

    assert factory.immutable_input_capability.capability is (
        ImmutableInputProjectionCapability.SHARED_READ_ONLY
    )
    assert claim is not None
    assert claim.state == "declared"
    assert (
        factory.execution_profile_identity.implementation_version
        != default_factory.execution_profile_identity.implementation_version
    )
    request = EnvironmentFactoryRequest(
        session_id="immutable-session",
        agent_name="agent",
        environment_name="coding",
    )
    first = asyncio.run(factory._attach_immutable_inputs(request))
    recovered_factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(workspace_root),
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
        immutable_inputs=(immutable_input,),
        immutable_input_store=ImmutableInputStore(store.root),
        immutable_input_runtime_compatibility_fingerprint=runtime_compatibility,
    )
    replay = asyncio.run(recovered_factory._attach_immutable_inputs(request))

    assert replay[0].attachment_id == first[0].attachment_id
    assert store.inspect()[0].reference_count == 1
    assert store.inspect()[0].attachment_count == 1
    asyncio.run(recovered_factory._release_immutable_inputs(replay))

    with pytest.raises(ValueError, match="runtime compatibility"):
        DockerCodingEnvironmentFactory(
            source_workspace=LocalWorkspace(workspace_root),
            toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
            immutable_inputs=(immutable_input,),
            immutable_input_store=store,
            immutable_input_runtime_compatibility_fingerprint="sha256:" + ("e" * 64),
        )


def test_docker_coding_binding_closes_runner_before_releasing_immutable_input(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    input_root = tmp_path / "runtime"
    source_root.mkdir()
    target_root.mkdir()
    input_root.mkdir()
    (input_root / "runtime.bin").write_bytes(b"runtime")
    runner, source_workspace, base_binding = _coding_product_test_binding(
        source_root,
        target_root,
    )
    immutable_input = inspect_local_immutable_input(
        input_root,
        target_path="/opt/cayu/inputs/runtime",
        policy_fingerprint="sha256:" + ("a" * 64),
        runtime_compatibility_fingerprint="sha256:" + ("b" * 64),
        authorization_scope_fingerprint="sha256:" + ("c" * 64),
    )
    store = ImmutableInputStore(tmp_path / "managed")
    attachment = store.attach_sync(
        immutable_input,
        attachment_id="docker:binding",
        owner_id="binding-session",
    )
    binding = DockerCodingWorkspaceBinding(
        target_workspace=base_binding._docker_target,
        limits=DockerWorkspaceTransferLimits(),
        immutable_input_store=store,
        immutable_input_attachments=(attachment,),
    )
    closed = False

    async def close_runner() -> None:
        nonlocal closed
        assert store.inspect()[0].reference_count == 1
        closed = True

    monkeypatch.setattr(runner, "close", close_runner)

    async def run() -> None:
        bound = await binding.bind(
            source_workspace,
            runner,
            session_id="binding-session",
        )
        await binding.finalize(bound, outcome="completed")

    asyncio.run(run())

    assert closed is True
    assert store.inspect()[0].reference_count == 0


def test_docker_coding_binding_retries_release_after_runner_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    input_root = tmp_path / "runtime"
    source_root.mkdir()
    target_root.mkdir()
    input_root.mkdir()
    (input_root / "runtime.bin").write_bytes(b"runtime")
    runner, source_workspace, base_binding = _coding_product_test_binding(
        source_root,
        target_root,
    )
    immutable_input = inspect_local_immutable_input(
        input_root,
        target_path="/opt/cayu/inputs/runtime",
        policy_fingerprint="sha256:" + ("a" * 64),
        runtime_compatibility_fingerprint="sha256:" + ("b" * 64),
        authorization_scope_fingerprint="sha256:" + ("c" * 64),
    )
    store = ImmutableInputStore(tmp_path / "managed")
    attachment = store.attach_sync(
        immutable_input,
        attachment_id="docker:release-retry",
        owner_id="binding-session",
    )
    binding = DockerCodingWorkspaceBinding(
        target_workspace=base_binding._docker_target,
        limits=DockerWorkspaceTransferLimits(),
        immutable_input_store=store,
        immutable_input_attachments=(attachment,),
    )
    close_calls = 0
    release_calls = 0
    original_release = store.release

    async def close_runner() -> None:
        nonlocal close_calls
        close_calls += 1

    async def flaky_release(attachment_id: str) -> None:
        nonlocal release_calls
        release_calls += 1
        if release_calls == 1:
            raise OSError("simulated durable registry failure after close")
        await original_release(attachment_id)

    monkeypatch.setattr(runner, "close", close_runner)
    monkeypatch.setattr(store, "release", flaky_release)

    async def run() -> None:
        bound = await binding.bind(
            source_workspace,
            runner,
            session_id="binding-session",
        )
        with pytest.raises(OSError, match="registry failure after close"):
            await binding.finalize(bound, outcome="completed")
        assert binding.abandon(bound) is False
        await binding.finalize(bound, outcome="completed")

    asyncio.run(run())

    assert close_calls == 1
    assert release_calls == 2
    assert store.inspect()[0].reference_count == 0


def test_docker_coding_factory_refuses_weakened_privilege_restrictions(
    tmp_path: Path,
) -> None:
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(
            image_identity=_image_identity(),
            restrictions=DockerWorkloadRestrictions(
                read_only_root=False, no_new_privileges=False, capability_add=("SYS_ADMIN",)
            ),
        ),
    )
    candidate = factory.construction_admission_candidate()
    requirements = ExecutionRequirements.trusted(guest_privilege="unprivileged")
    decision = evaluate_execution_admission(
        candidate=candidate.candidate,
        requirements=requirements,
        evidence=candidate.evidence,
        stage="pre_create",
    )

    containment_claim = candidate.evidence.claim_for("guest_privilege_containment")
    unprivileged_claim = candidate.evidence.claim_for("unprivileged_guest")
    assert containment_claim is not None
    assert unprivileged_claim is not None
    assert containment_claim.state == "unsupported"
    assert unprivileged_claim.state == "unsupported"
    assert decision.status == "refused"


def test_docker_coding_factory_returns_only_exact_final_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        if any(".cayu-toolchain-write-probe" in arg for arg in command.argv):
            return ExecResult(stdout="linux/amd64\n")
        docker_args = command.argv[1:]
        if docker_args[0] == "run":
            return ExecResult(stdout=_CONTAINER_ID)
        if docker_args[0] == "inspect":
            return ExecResult(stdout=json.dumps(_inspection(restrictions)))
        if docker_args[:2] == ["exec", _CONTAINER_ID] and "id -u" in docker_args[-1]:
            return ExecResult(stdout=restrictions.user)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(
            image_identity=_image_identity(), restrictions=restrictions
        ),
        docker_path="/usr/bin/docker",
    )
    request = EnvironmentFactoryRequest(
        session_id="session",
        agent_name="agent",
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
        evidence = result.metadata["execution_capabilities"]
        assert evidence["environment_fingerprint"].startswith("sha256:")
        assert result.metadata["execution_requirements"]["required_executables"] == [
            "git",
            "python3",
            "rm",
            "sh",
            "sleep",
        ]
        assert result.environment.runner is not None
        assert result.environment.binding is not None
        assert result.environment.workspace is factory.source_workspace
        assert result.release is not None
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(run())

    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_docker_coding_factory_reconnects_exact_preserved_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []
    workspace_root = tmp_path / "workspace"
    input_root = tmp_path / "runtime"
    workspace_root.mkdir()
    input_root.mkdir()
    (input_root / "runtime.bin").write_bytes(b"runtime")
    image_identity = _image_identity()
    immutable_input = inspect_local_immutable_input(
        input_root,
        target_path="/opt/cayu/inputs/runtime",
        policy_fingerprint="sha256:" + ("a" * 64),
        runtime_compatibility_fingerprint=image_identity.fingerprint,
        authorization_scope_fingerprint="sha256:" + ("c" * 64),
    )
    store = ImmutableInputStore(tmp_path / "managed")
    mount_source: str | None = None

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        nonlocal mount_source
        del kwargs
        calls.append(command.argv)
        if any(".cayu-toolchain-write-probe" in arg for arg in command.argv):
            return ExecResult(stdout="linux/amd64\n")
        docker_args = command.argv[1:]
        if docker_args[0] == "run":
            mount_argument = docker_args[docker_args.index("--mount") + 1]
            mount_source = mount_argument.split("source=", 1)[1].split(",", 1)[0]
            return ExecResult(stdout=_CONTAINER_ID)
        if docker_args[0] == "inspect":
            assert mount_source is not None
            return ExecResult(
                stdout=json.dumps(
                    _inspection(
                        restrictions,
                        immutable_mount=(
                            mount_source,
                            immutable_input.projection.target_path,
                        ),
                    )
                )
            )
        if docker_args[:2] == ["exec", _CONTAINER_ID] and "id -u" in docker_args[-1]:
            return ExecResult(stdout=restrictions.user)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(workspace_root),
        toolchain_profile=docker_toolchain_profile(
            image_identity=image_identity, restrictions=restrictions
        ),
        docker_path="/usr/bin/docker",
        immutable_inputs=(immutable_input,),
        immutable_input_store=store,
        immutable_input_runtime_compatibility_fingerprint=image_identity.fingerprint,
    )
    create_request = EnvironmentFactoryRequest(
        session_id="reconnect",
        agent_name="agent",
        environment_name="coding",
    )

    async def run() -> None:
        created = await factory.create(create_request)
        assert created.reconnect_metadata["container_id"] == _CONTAINER_ID
        assert len(created.reconnect_metadata["allocation_fingerprint"]) == 64
        assert created.release is not None
        await created.release(EnvironmentFactoryReleaseAction.PRESERVE)
        assert not any(call[1:3] == ["rm", "-f"] for call in calls)
        assert store.inspect()[0].reference_count == 1

        reconnect_request = replace(
            create_request,
            operation=EnvironmentFactoryOperation.RECONNECT,
            reconnect_metadata=created.reconnect_metadata,
        )
        candidate = factory.execution_admission_candidate(reconnect_request)
        assert candidate.evidence.claim_for("reconnect").state == "declared"  # type: ignore[union-attr]
        reconnected = await factory.create(reconnect_request)
        assert reconnected.reconnect_metadata == created.reconnect_metadata
        assert reconnected.environment.runner is not None
        assert reconnected.environment.runner.container_id == _CONTAINER_ID
        assert store.inspect()[0].reference_count == 1
        assert (
            reconnected.environment.runner.execution_capability_evidence()
            .claim_for("reconnect")
            .state
            == "live_verified"
        )
        assert reconnected.release is not None
        await reconnected.release(EnvironmentFactoryReleaseAction.DISCARD)
        assert store.inspect()[0].reference_count == 0

    asyncio.run(run())

    assert sum(call[1] == "run" for call in calls) == 1
    assert sum(call[1] == "inspect" for call in calls) == 2
    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_docker_coding_same_create_request_recovers_one_named_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []
    container_exists = False

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        nonlocal container_exists
        del kwargs
        calls.append(command.argv)
        if any(".cayu-toolchain-write-probe" in arg for arg in command.argv):
            return ExecResult(stdout="linux/amd64\n")
        docker_args = command.argv[1:]
        if docker_args[:2] == ["container", "ls"]:
            return ExecResult(stdout=(_CONTAINER_ID + "\n") if container_exists else "")
        if docker_args[0] == "run":
            container_exists = True
            return ExecResult(stdout=_CONTAINER_ID)
        if docker_args[0] == "inspect":
            return ExecResult(stdout=json.dumps(_inspection(restrictions)))
        if docker_args[:2] == ["exec", _CONTAINER_ID] and "id -u" in docker_args[-1]:
            return ExecResult(stdout=restrictions.user)
        if docker_args[:2] == ["rm", "-f"]:
            container_exists = False
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(
            image_identity=_image_identity(), restrictions=restrictions
        ),
        docker_path="/usr/bin/docker",
    )
    request = EnvironmentFactoryRequest(
        session_id="same-create",
        agent_name="agent",
        environment_name="coding",
    )

    async def run() -> None:
        scope = factory.allocation_scope(request)
        assert scope is not None
        assert scope.provider == "docker"
        first = await factory.create(request)
        second = await factory.create(request)
        assert first.reconnect_metadata == second.reconnect_metadata
        assert first.release is not None
        assert second.release is not None
        await first.release(EnvironmentFactoryReleaseAction.PRESERVE)
        await second.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(run())

    assert sum(call[1] == "run" for call in calls) == 1
    assert sum(call[1] == "inspect" for call in calls) == 2
    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_docker_coding_recoverable_allocation_reuses_dispatched_intent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []
    container_exists = False

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        nonlocal container_exists
        del kwargs
        calls.append(command.argv)
        if any(".cayu-toolchain-write-probe" in arg for arg in command.argv):
            return ExecResult(stdout="linux/amd64\n")
        docker_args = command.argv[1:]
        if docker_args[:2] == ["container", "ls"]:
            return ExecResult(stdout=(_CONTAINER_ID + "\n") if container_exists else "")
        if docker_args[0] == "run":
            container_exists = True
            return ExecResult(stdout=_CONTAINER_ID)
        if docker_args[0] == "inspect":
            return ExecResult(stdout=json.dumps(_inspection(restrictions)))
        if docker_args[:2] == ["exec", _CONTAINER_ID] and "id -u" in docker_args[-1]:
            return ExecResult(stdout=restrictions.user)
        if docker_args[:2] == ["rm", "-f"]:
            container_exists = False
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(
            image_identity=_image_identity(), restrictions=restrictions
        ),
        docker_path="/usr/bin/docker",
    )
    request = EnvironmentFactoryRequest(
        session_id="recoverable-create",
        agent_name="agent",
        environment_name="coding",
    )
    intent = EnvironmentAllocationIntent(
        allocation_id="ealloc_" + ("d" * 32),
        provider="docker",
        adapter_generation="cayu.docker_coding.v11",
        session_id=request.session_id,
        environment_name=request.environment_name,
        requested_operation=EnvironmentFactoryOperation.CREATE,
    )

    async def run() -> None:
        first_context = _TestAllocationContext(intent)
        first = await factory.create_recoverable(request, first_context)
        assert first_context.state is EnvironmentAllocationState.ACKNOWLEDGED
        assert first_context.acknowledged_reconnect_metadata == first.reconnect_metadata

        recovered_context = _TestAllocationContext(
            first_context.intent,
            state=EnvironmentAllocationState.DISPATCHED,
        )
        recovered = await factory.create_recoverable(request, recovered_context)
        assert recovered_context.state is EnvironmentAllocationState.ACKNOWLEDGED
        assert recovered.reconnect_metadata == first.reconnect_metadata
        assert first.release is not None
        assert recovered.release is not None
        await first.release(EnvironmentFactoryReleaseAction.PRESERVE)
        assert await recovered_context.mark_reaping() is True
        await recovered.release(EnvironmentFactoryReleaseAction.DISCARD)
        assert recovered_context.state is EnvironmentAllocationState.REAPED

    asyncio.run(run())

    assert sum(call[1] == "run" for call in calls) == 1
    assert sum(call[1] == "inspect" for call in calls) == 2
    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_docker_coding_reconnect_releases_interrupted_finalize_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    input_root = tmp_path / "runtime"
    workspace_root.mkdir()
    input_root.mkdir()
    (input_root / "runtime.bin").write_bytes(b"runtime")
    image_identity = _image_identity()
    immutable_input = inspect_local_immutable_input(
        input_root,
        target_path="/opt/cayu/inputs/runtime",
        policy_fingerprint="sha256:" + ("a" * 64),
        runtime_compatibility_fingerprint=image_identity.fingerprint,
        authorization_scope_fingerprint="sha256:" + ("c" * 64),
    )
    store = ImmutableInputStore(tmp_path / "managed")
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(workspace_root),
        toolchain_profile=docker_toolchain_profile(image_identity=image_identity),
        docker_path="/usr/bin/docker",
        immutable_inputs=(immutable_input,),
        immutable_input_store=store,
        immutable_input_runtime_compatibility_fingerprint=image_identity.fingerprint,
    )
    create_request = EnvironmentFactoryRequest(
        session_id="interrupted-finalize",
        agent_name="agent",
        environment_name="coding",
    )
    attachment = store.attach_sync(
        immutable_input,
        attachment_id=docker_coding_module._immutable_input_attachment_id(
            create_request,
            immutable_input,
        ),
        owner_id=create_request.session_id,
    )
    store.mark_container_closing_sync((attachment,), container_id=_CONTAINER_ID)
    reconnect_request = replace(
        create_request,
        operation=EnvironmentFactoryOperation.RECONNECT,
        reconnect_metadata=docker_coding_module._docker_coding_reconnect_metadata(
            container_id=_CONTAINER_ID,
            configuration_fingerprint=factory._configuration_fingerprint,
            image_fingerprint=image_identity.fingerprint,
            toolchain_profile_fingerprint=factory.toolchain_profile.fingerprint,
        ),
    )

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        if any(".cayu-toolchain-write-probe" in arg for arg in command.argv):
            return ExecResult(stdout="linux/amd64\n")
        docker_args = command.argv[1:]
        assert docker_args[:2] == ["container", "ls"]
        return ExecResult(stdout="")

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)

    with pytest.raises(RuntimeError, match="cleanup completed"):
        asyncio.run(factory.create(reconnect_request))

    assert ImmutableInputStore(store.root).inspect()[0].reference_count == 0


def test_docker_coding_factory_structurally_refuses_a_missing_final_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    restrictions = DockerWorkloadRestrictions()
    calls: list[list[str]] = []

    async def fake_run_subprocess(command, **kwargs: Any) -> ExecResult:
        del kwargs
        calls.append(command.argv)
        if any(".cayu-toolchain-write-probe" in arg for arg in command.argv):
            return ExecResult(stdout="linux/amd64\n")
        docker_args = command.argv[1:]
        if docker_args[0] == "run":
            return ExecResult(stdout=_CONTAINER_ID)
        if docker_args[0] == "inspect":
            return ExecResult(stdout=json.dumps(_inspection(restrictions)))
        if docker_args[:2] == ["exec", _CONTAINER_ID] and "id -u" in docker_args[-1]:
            return ExecResult(stdout=restrictions.user)
        if docker_args[:2] == ["exec", _CONTAINER_ID] and "command -v git" in docker_args[-1]:
            return ExecResult(exit_code=127)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(
            image_identity=_image_identity(), restrictions=restrictions
        ),
        docker_path="/usr/bin/docker",
    )
    request = EnvironmentFactoryRequest(
        session_id="missing-tool",
        agent_name="agent",
        environment_name="coding",
    )

    with pytest.raises(ExecutionAdmissionError) as caught:
        asyncio.run(factory.create(request))

    refusal = next(item for item in caught.value.decision.refusals if item.executable == "git")
    assert refusal.code == "unsupported_capability"
    assert calls[-1][1:] == ["rm", "-f", _CONTAINER_ID]


def test_revision_aware_sync_refuses_preexisting_source_conflict(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("baseline", encoding="utf-8")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        max_file_bytes=1024,
        max_total_bytes=4096,
        source_conflict_policy="require_revision",
    )

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="conflict")
        await target.write_bytes("a.txt", b"container")
        await source.write_bytes("a.txt", b"external")
        with pytest.raises(SyncBindingSourceConflictError) as caught:
            await binding.finalize(bound, outcome="completed")
        assert caught.value.path == "a.txt"
        assert caught.value.applied_paths == ()
        assert binding.abandon(bound) is True

    asyncio.run(run())
    assert (source_root / "a.txt").read_text(encoding="utf-8") == "external"


def test_revision_aware_sync_reports_partial_publication(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("a0", encoding="utf-8")
    (source_root / "b.txt").write_text("b0", encoding="utf-8")

    class RacingLocalWorkspace(LocalWorkspace):
        race = True

        async def replace_bytes(
            self,
            path: str,
            content: bytes,
            *,
            expected_revision: str,
        ):
            if path == "b.txt" and self.race:
                self.race = False
                await self.write_bytes(path, b"external-race")
            return await super().replace_bytes(
                path,
                content,
                expected_revision=expected_revision,
            )

    source = RacingLocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        max_file_bytes=1024,
        max_total_bytes=4096,
        source_conflict_policy="require_revision",
    )

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="partial-conflict")
        await target.write_bytes("a.txt", b"a1")
        await target.write_bytes("b.txt", b"b1")
        with pytest.raises(SyncBindingSourceConflictError) as caught:
            await binding.finalize(bound, outcome="completed")
        assert caught.value.path == "b.txt"
        assert caught.value.applied_paths == ("a.txt",)
        await source.write_bytes("b.txt", b"b0")
        # The external repair is unconditional, so a new bind is intentionally required;
        # finalization cannot reinterpret that write as Cayu's publication.
        assert binding.abandon(bound) is True

    asyncio.run(run())
    assert (source_root / "a.txt").read_text(encoding="utf-8") == "a1"
    assert (source_root / "b.txt").read_text(encoding="utf-8") == "b0"


def test_revision_aware_sync_recovery_never_overwrites_concurrent_source_change(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("a0", encoding="utf-8")
    (source_root / "b.txt").write_text("b0", encoding="utf-8")

    class FailOnceLocalWorkspace(LocalWorkspace):
        fail_once = True

        async def replace_bytes(
            self,
            path: str,
            content: bytes,
            *,
            expected_revision: str,
        ) -> WorkspaceMutationResult:
            if path == "b.txt" and self.fail_once:
                self.fail_once = False
                raise OSError("injected copy-back failure for b.txt")
            return await super().replace_bytes(
                path,
                content,
                expected_revision=expected_revision,
            )

    source = FailOnceLocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        max_file_bytes=1024,
        max_total_bytes=4096,
        source_conflict_policy="require_revision",
    )

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="initial-owner")
        await target.write_bytes("a.txt", b"a1")
        await target.write_bytes("b.txt", b"b1")
        with pytest.raises(SyncBindingSourceConflictError):
            await binding.finalize(bound, outcome="completed")
        recovery_state = binding._completion_finalization_recovery_state(bound)
        assert recovery_state is not None

        await source.write_bytes("b.txt", b"external-change")
        recovered_binding = SyncBinding(
            target_workspace=target,
            max_file_bytes=1024,
            max_total_bytes=4096,
            source_conflict_policy="require_revision",
        )
        recovered = await recovered_binding._recover_completion_finalization(
            source,
            None,
            session_id="recovered-owner",
            agent_name="agent",
            environment_name="environment",
            recovery_state=recovery_state,
        )
        try:
            with pytest.raises(SyncBindingSourceConflictError) as caught:
                await recovered_binding.finalize(recovered, outcome="completed")
            assert caught.value.path == "b.txt"
            assert caught.value.applied_paths == ("a.txt",)
            retried = await recovered_binding._recover_completion_finalization(
                source,
                None,
                session_id="recovered-owner",
                agent_name="agent",
                environment_name="environment",
                recovery_state=recovery_state,
            )
            assert retried.state_key == recovered.state_key
            with pytest.raises(SyncBindingSourceConflictError):
                await recovered_binding.finalize(retried, outcome="completed")
        finally:
            recovered_binding.abandon(recovered)
            binding.abandon(bound)

    asyncio.run(run())

    assert (source_root / "a.txt").read_text(encoding="utf-8") == "a1"
    assert (source_root / "b.txt").read_text(encoding="utf-8") == "external-change"


def test_docker_coding_recovery_restores_fresh_binding_authority(tmp_path: Path) -> None:
    source_root = tmp_path / "source-docker-recovery"
    target_root = tmp_path / "target-docker-recovery"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "code.py").write_text("value = 1\n", encoding="utf-8")
    runner = _LocalDockerRunner(target_root)
    source = LocalWorkspace(
        source_root,
        workspace_id="docker-recovery-source",
        excluded_directory_names=(".cayu", ".git", ".runtime"),
    )

    def binding() -> DockerCodingWorkspaceBinding:
        return DockerCodingWorkspaceBinding(
            target_workspace=RunnerWorkspace(
                runner,
                workspace_id="docker-recovery-target",
                python_executable=sys.executable,
                excluded_directory_names=(".cayu", ".git", ".runtime"),
            ),
            limits=DockerWorkspaceTransferLimits(
                max_file_bytes=1024,
                max_total_bytes=4096,
                max_archive_bytes=64 * 1024,
            ),
        )

    async def run() -> None:
        original = binding()
        bound = await original.bind(source, runner, session_id="docker-recovery-session")
        assert bound.workspace is not None
        await bound.workspace.write_bytes("code.py", b"value = 2\n")
        recovery_state = original._completion_finalization_recovery_state(bound)
        assert recovery_state is not None
        original.abandon(bound)

        recovered_binding = binding()
        recovered = await recovered_binding._recover_completion_finalization(
            source,
            runner,
            session_id="docker-recovery-session",
            agent_name="agent",
            environment_name="environment",
            recovery_state=recovery_state,
        )
        assert recovered.state_key in recovered_binding._coding_authorities
        await recovered_binding.finalize(recovered, outcome="completed")
        assert recovered_binding._coding_authorities == {}
        assert recovered_binding._states == {}

    asyncio.run(run())
    assert (source_root / "code.py").read_text(encoding="utf-8") == "value = 2\n"


def test_revision_aware_sync_preserves_copyback_process_control(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("before", encoding="utf-8")

    class FatalLocalWorkspace(LocalWorkspace):
        async def replace_bytes(
            self,
            path: str,
            content: bytes,
            *,
            expected_revision: str,
        ):
            del path, content, expected_revision
            raise GeneratorExit("copyback supervisor exited")

    source = FatalLocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        max_file_bytes=1024,
        max_total_bytes=4096,
        source_conflict_policy="require_revision",
    )

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="fatal-copyback")
        await target.write_bytes("a.txt", b"after")
        await binding.finalize(bound, outcome="completed")

    with pytest.raises(GeneratorExit, match="copyback supervisor exited"):
        asyncio.run(run())


def test_revision_aware_sync_records_a_settled_write_before_redelivering_cancellation(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_text("before", encoding="utf-8")
    mutation_settled = asyncio.Event()
    release_result = asyncio.Event()

    class SettlingLocalWorkspace(LocalWorkspace):
        block_once = True

        async def replace_bytes(
            self,
            path: str,
            content: bytes,
            *,
            expected_revision: str,
        ):
            result = await super().replace_bytes(
                path,
                content,
                expected_revision=expected_revision,
            )
            if self.block_once:
                self.block_once = False
                mutation_settled.set()
                await release_result.wait()
            return result

    source = SettlingLocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        max_file_bytes=1024,
        max_total_bytes=4096,
        source_conflict_policy="require_revision",
    )

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="cancelled-copyback")
        await target.write_bytes("a.txt", b"after")
        finalize_task = asyncio.create_task(binding.finalize(bound, outcome="completed"))
        await mutation_settled.wait()
        finalize_task.cancel()
        release_result.set()
        with pytest.raises(asyncio.CancelledError):
            await finalize_task
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None

    asyncio.run(run())
    assert (source_root / "a.txt").read_text(encoding="utf-8") == "after"


def test_runner_workspace_excludes_protected_directories_before_listing_limits(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git" / "objects").mkdir(parents=True)
    for index in range(20):
        (tmp_path / ".git" / "objects" / f"{index}").write_text("x", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("ok", encoding="utf-8")
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        python_executable=sys.executable,
        excluded_directory_names=(".cayu", ".git", ".runtime"),
    )

    result = asyncio.run(workspace.list("**/*", limit=1))

    assert result.paths == ("visible.txt",)
    assert result.truncated is False
    with pytest.raises(ValueError, match="excluded directory"):
        asyncio.run(workspace.read_bytes(".git/objects/0"))


def test_runner_workspace_directory_exclusions_match_local_portable_name_semantics(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "build "
    protected.mkdir()
    (protected / "artifact.txt").write_text("generated\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("source\n", encoding="utf-8")
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        python_executable=sys.executable,
        excluded_directory_names=("build",),
    )

    listing = asyncio.run(workspace.list("**/*", limit=1))
    git_entries = asyncio.run(workspace.list_git_entries(limit=1))

    assert listing.paths == ("visible.txt",)
    assert listing.total_count == 1
    assert listing.truncated is False
    assert tuple(entry.path for entry in git_entries.entries) == ("visible.txt",)
    with pytest.raises(ValueError, match="excluded directory"):
        asyncio.run(workspace.read_bytes("build /artifact.txt"))
    with pytest.raises(ValueError, match="case-insensitively unique"):
        RunnerWorkspace(
            LocalRunner(tmp_path, inherit_env=False),
            python_executable=sys.executable,
            excluded_directory_names=("build", "BUILD."),
        )


def test_runner_workspace_excludes_sensitive_path_patterns_before_dispatch(
    tmp_path: Path,
) -> None:
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "private.pem").write_text("secret\n", encoding="utf-8")
    nested_env = nested / ".env"
    nested_env.mkdir()
    (nested_env / "token").write_text("directory secret\n", encoding="utf-8")
    (tmp_path / "visible.txt").write_text("ok\n", encoding="utf-8")
    workspace = RunnerWorkspace(
        LocalRunner(tmp_path, inherit_env=False),
        python_executable=sys.executable,
        excluded_path_patterns=(".env", "**/.env", "*.pem", "**/*.pem"),
    )

    result = asyncio.run(workspace.list("**/*", limit=1))
    git_entries = asyncio.run(workspace.list_git_entries(limit=1))

    assert result.paths == ("visible.txt",)
    assert result.truncated is False
    assert tuple(entry.path for entry in git_entries.entries) == ("visible.txt",)
    with pytest.raises(ValueError, match="excluded path pattern"):
        asyncio.run(workspace.read_bytes(".env"))
    with pytest.raises(ValueError, match="excluded path pattern"):
        asyncio.run(workspace.write_bytes("nested/private.pem", b"changed"))
    with pytest.raises(ValueError, match="excluded path pattern"):
        asyncio.run(workspace.read_bytes("nested/.env/token"))


def test_docker_coding_binding_uses_ephemeral_git_and_never_copies_protected_paths(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / ".git" / "objects").mkdir(parents=True)
    (source_root / ".git" / "objects" / "host-only").write_text(
        "host git",
        encoding="utf-8",
    )
    (source_root / ".cayu").mkdir()
    (source_root / ".cayu" / "private.txt").write_text("private", encoding="utf-8")
    (source_root / ".runtime").mkdir()
    (source_root / ".runtime" / "state").write_text("state", encoding="utf-8")
    (source_root / ".env").write_text("TOKEN=host-secret\n", encoding="utf-8")
    (source_root / "code.py").write_text("value = 1\n", encoding="utf-8")

    class LocalDockerRunner(DockerRunner):
        def __init__(self, root: Path) -> None:
            super().__init__(
                "local-docker-test",
                default_cwd="/workspace",
                docker_path="/usr/bin/docker",
                _container_id=_CONTAINER_ID,
            )
            self.local = LocalRunner(root, inherit_env=False)
            self._test_root = root

        def resolve_cwd(self, cwd: str | None = None) -> str:
            del cwd
            return str(self._test_root)

        async def exec(self, command, **kwargs: Any):
            kwargs["cwd"] = None
            return await self.local.exec(command, **kwargs)

        async def exec_stream(self, command, **kwargs: Any):
            kwargs["cwd"] = None
            return await self.local.exec_stream(command, **kwargs)

        async def close(self) -> None:
            await self.local.close()
            self._closed = True

    runner = LocalDockerRunner(target_root)
    source = LocalWorkspace(
        source_root,
        workspace_id="source",
        excluded_directory_names=(".cayu", ".git", ".runtime"),
        excluded_path_patterns=(".env", "**/.env"),
    )
    target = RunnerWorkspace(
        runner,
        workspace_id="target",
        python_executable=sys.executable,
        excluded_directory_names=(".cayu", ".git", ".runtime"),
        excluded_path_patterns=(".env", "**/.env"),
    )
    binding = DockerCodingWorkspaceBinding(
        target_workspace=target,
        limits=DockerWorkspaceTransferLimits(
            max_file_bytes=1024,
            max_total_bytes=4096,
            max_archive_bytes=32 * 1024,
        ),
    )

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="docker-binding")
        assert (target_root / ".git").is_dir()
        assert not (target_root / ".git" / "objects" / "host-only").exists()
        assert not (target_root / ".cayu").exists()
        assert not (target_root / ".runtime").exists()
        assert not (target_root / ".env").exists()
        guest_git_config = (target_root / ".git" / "config").read_text(encoding="utf-8")
        assert "hooksPath = /dev/null" in guest_git_config
        assert "fsmonitor = false" in guest_git_config
        assert "interactive = never" in guest_git_config
        assert "allow = never" in guest_git_config
        assert "gpgSign = false" in guest_git_config
        await runner.exec(
            ExecCommand.process(
                sys.executable,
                "-c",
                "from pathlib import Path; "
                "Path('code.py').write_text('value = 2\\n'); "
                "Path('new.py').write_text('new = True\\n')",
            )
        )
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None

    asyncio.run(run())

    assert (source_root / "code.py").read_text(encoding="utf-8") == "value = 2\n"
    assert (source_root / "new.py").read_text(encoding="utf-8") == "new = True\n"
    assert (source_root / ".git" / "objects" / "host-only").read_text(
        encoding="utf-8"
    ) == "host git"
    assert (source_root / ".cayu" / "private.txt").read_text(encoding="utf-8") == "private"
    assert (source_root / ".runtime" / "state").read_text(encoding="utf-8") == "state"
    assert (source_root / ".env").read_text(encoding="utf-8") == "TOKEN=host-secret\n"


def test_docker_coding_binding_rejects_copy_in_after_admitted_source_changes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source_file = source_root / "code.py"
    source_file.write_text("value = 1\n", encoding="utf-8")

    runner, source, binding = _coding_product_test_binding(source_root, target_root)
    source_file.write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="conflicts with the admitted source revision"):
        asyncio.run(binding.bind(source, runner, session_id="stale-copy-in"))


def test_docker_coding_binding_rejects_mutated_ephemeral_git_authority(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source_file = source_root / "code.py"
    source_file.write_text("value = 1\n", encoding="utf-8")

    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="mutated-git-authority")
        result = await runner.exec(
            ExecCommand.process("git", "update-index", "--assume-unchanged", "code.py")
        )
        assert result.exit_code == 0
        (target_root / "code.py").write_text("value = 2\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="ephemeral Git authority changed"):
            await binding.finalize(bound, outcome="completed")
        assert binding.abandon(bound) is True

    asyncio.run(run())

    assert source_file.read_text(encoding="utf-8") == "value = 1\n"


def test_docker_coding_rejects_included_filter_before_filter_execution(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source_file = source_root / "code.py"
    source_file.write_text("value = 1\n", encoding="utf-8")
    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="included-filter")
        (target_root / ".git" / "filter.conf").write_text(
            '[filter "review"]\n\tclean = touch filter-ran; cat\n',
            encoding="utf-8",
        )
        configured = await runner.exec(
            ExecCommand.process("git", "config", "--local", "include.path", "filter.conf")
        )
        assert configured.exit_code == 0
        (target_root / ".gitattributes").write_text("*.py filter=review\n", encoding="utf-8")
        source_file_in_guest = target_root / "code.py"
        source_file_in_guest.write_text("value = 2\n", encoding="utf-8")

        transformed = await docker_coding_module._git_paths_with_transformed_bytes(
            runner,
            paths=("code.py",),
        )
        assert transformed is None
        assert not (target_root / "filter-ran").exists()
        with pytest.raises(RuntimeError, match="ephemeral Git authority changed"):
            await binding.finalize(bound, outcome="completed")
        assert binding.abandon(bound) is True

    asyncio.run(run())

    assert not (target_root / "filter-ran").exists()
    assert source_file.read_text(encoding="utf-8") == "value = 1\n"


def test_docker_coding_binding_captures_exact_final_diff_content(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source_file = source_root / "code.py"
    source_file.write_text("value = 1\n", encoding="utf-8")

    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    async def run() -> dict[str, Any]:
        bound = await binding.bind(source, runner, session_id="exact-final-diff")
        (target_root / "code.py").write_text("value = 2  \n", encoding="utf-8")
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None
        evidence = snapshot.metadata["final_git_evidence"]
        assert type(evidence) is dict
        return evidence

    evidence = asyncio.run(run())

    diff = evidence["diff"]
    assert type(diff) is dict
    assert "+value = 2  \n" in diff["content"]
    assert source_file.read_text(encoding="utf-8") == "value = 2  \n"


def test_docker_coding_binding_marks_git_normalized_byte_changes_partial(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / ".gitattributes").write_text("*.txt text\n", encoding="utf-8")
    source_file = source_root / "normalized.txt"
    source_file.write_bytes(b"stable\n")

    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    async def run() -> dict[str, Any]:
        bound = await binding.bind(source, runner, session_id="normalized-final-diff")
        (target_root / "normalized.txt").write_bytes(b"stable\r\n")
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None
        evidence = snapshot.metadata["final_git_evidence"]
        assert type(evidence) is dict
        return evidence

    evidence = asyncio.run(run())

    summary = evidence["summary"]
    diff = evidence["diff"]
    assert type(summary) is dict
    assert type(diff) is dict
    summary_changes = summary["structured"]["changes"]
    normalized = next(change for change in summary_changes if change["path"] == "normalized.txt")
    assert normalized["count_kind"] == "unknown"
    assert diff["structured"]["truncated"] is True
    assert "workspace_delta_unrepresented" in diff["structured"]["truncation_reasons"]
    assert "normalized.txt" not in diff["content"]
    assert source_file.read_bytes() == b"stable\r\n"


def test_docker_coding_binding_rejects_mixed_text_and_normalized_byte_changes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / ".gitattributes").write_text("*.txt text\n", encoding="utf-8")
    source_file = source_root / "normalized.txt"
    source_file.write_bytes(b"stable\n")

    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    async def run() -> dict[str, Any]:
        bound = await binding.bind(source, runner, session_id="mixed-normalized-final-diff")
        (target_root / "normalized.txt").write_bytes(b"changed\r\n")
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None
        evidence = snapshot.metadata["final_git_evidence"]
        assert type(evidence) is dict
        return evidence

    evidence = asyncio.run(run())

    summary = evidence["summary"]
    diff = evidence["diff"]
    assert type(summary) is dict
    assert type(diff) is dict
    normalized = next(
        change for change in summary["structured"]["changes"] if change["path"] == "normalized.txt"
    )
    assert normalized["count_kind"] == "text"
    assert "normalized.txt" in diff["content"]
    assert diff["structured"]["truncated"] is True
    assert "workspace_delta_unrepresented" in diff["structured"]["truncation_reasons"]
    assert source_file.read_bytes() == b"changed\r\n"


def test_docker_coding_binding_accepts_explicitly_unfiltered_text_bytes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / ".gitattributes").write_text("*.txt -text\n", encoding="utf-8")
    source_file = source_root / "raw.txt"
    source_file.write_bytes(b"stable\n")

    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    async def run() -> dict[str, Any]:
        bound = await binding.bind(source, runner, session_id="raw-final-diff")
        (target_root / "raw.txt").write_bytes(b"changed\r\n")
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None
        evidence = snapshot.metadata["final_git_evidence"]
        assert type(evidence) is dict
        return evidence

    evidence = asyncio.run(run())

    diff = evidence["diff"]
    assert type(diff) is dict
    assert diff["structured"]["truncated"] is False
    assert "+changed\r\n" in diff["content"]
    assert source_file.read_bytes() == b"changed\r\n"


def test_docker_coding_binding_rejects_a_normalized_baseline_preimage(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / ".gitattributes").write_text("*.txt text\n", encoding="utf-8")
    source_file = source_root / "normalized.txt"
    source_file.write_bytes(b"stable\r\n")

    runner, source, binding = _coding_product_test_binding(source_root, target_root)

    async def run() -> dict[str, Any]:
        bound = await binding.bind(source, runner, session_id="normalized-baseline-diff")
        (target_root / "normalized.txt").write_bytes(b"changed\n")
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None
        evidence = snapshot.metadata["final_git_evidence"]
        assert type(evidence) is dict
        return evidence

    evidence = asyncio.run(run())

    diff = evidence["diff"]
    assert type(diff) is dict
    assert "normalized.txt" in diff["content"]
    assert diff["structured"]["truncated"] is True
    assert "workspace_delta_unrepresented" in diff["structured"]["truncation_reasons"]
    assert source_file.read_bytes() == b"changed\n"


def test_docker_coding_binding_failed_abandon_preserves_finalization_authority(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    substituted_root = tmp_path / "substituted"
    source_root.mkdir()
    target_root.mkdir()
    substituted_root.mkdir()
    source_file = source_root / "code.py"
    source_file.write_text("value = 1\n", encoding="utf-8")

    runner, source, binding = _coding_product_test_binding(source_root, target_root)
    substituted = LocalWorkspace(substituted_root, workspace_id="substituted")

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="failed-abandon")
        forged = replace(bound, source_workspace=substituted)
        with pytest.raises(ValueError, match="source workspace"):
            binding.abandon(forged)
        (target_root / "code.py").write_text("value = 2\n", encoding="utf-8")
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None
        assert "final_git_evidence" in snapshot.metadata

    asyncio.run(run())

    assert source_file.read_text(encoding="utf-8") == "value = 2\n"


@pytest.mark.parametrize("sync_back", ["never", "on_success", "always"])
@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled", None])
def test_docker_coding_binding_scopes_ignored_path_guard_to_publication(
    tmp_path: Path,
    sync_back,
    outcome,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (source_root / "code.py").write_text("value = 1\n", encoding="utf-8")

    class LocalDockerRunner(DockerRunner):
        def __init__(self, root: Path) -> None:
            super().__init__(
                "local-docker-test",
                default_cwd="/workspace",
                docker_path="/usr/bin/docker",
                _container_id=_CONTAINER_ID,
            )
            self.local = LocalRunner(root, inherit_env=False)
            self._test_root = root

        def resolve_cwd(self, cwd: str | None = None) -> str:
            del cwd
            return str(self._test_root)

        async def exec(self, command, **kwargs: Any):
            kwargs["cwd"] = None
            return await self.local.exec(command, **kwargs)

        async def exec_stream(self, command, **kwargs: Any):
            kwargs["cwd"] = None
            return await self.local.exec_stream(command, **kwargs)

        async def close(self) -> None:
            await self.local.close()
            self._closed = True

    runner = LocalDockerRunner(target_root)
    source = LocalWorkspace(
        source_root,
        workspace_id="source",
        excluded_directory_names=(".cayu", ".git", ".runtime"),
        excluded_path_patterns=(".env", "**/.env"),
    )
    target = RunnerWorkspace(
        runner,
        workspace_id="target",
        python_executable=sys.executable,
        excluded_directory_names=(".cayu", ".git", ".runtime"),
        excluded_path_patterns=(".env", "**/.env"),
    )
    binding = DockerCodingWorkspaceBinding(
        target_workspace=target,
        limits=DockerWorkspaceTransferLimits(
            max_file_bytes=1024,
            max_total_bytes=4096,
            max_archive_bytes=32 * 1024,
        ),
    )

    async def run() -> None:
        binding.sync_back = sync_back
        bound = await binding.bind(source, runner, session_id="ignored-publication")
        (target_root / "ignored.txt").write_text("hidden mutation\n", encoding="utf-8")
        if sync_back == "always" or (sync_back == "on_success" and outcome == "completed"):
            with pytest.raises(RuntimeError, match="Git would omit a source path"):
                await binding.finalize(bound, outcome=outcome)
        else:
            assert await binding.finalize(bound, outcome=outcome) is None
            with pytest.raises(RuntimeError, match="lost its admitted authority"):
                await binding.finalize(bound, outcome=outcome)
        assert binding.abandon(bound) is True

    asyncio.run(run())

    assert not (source_root / "ignored.txt").exists()


@pytest.mark.parametrize("change", ["missing", "wrong", "duplicate", "unwritable"])
def test_strict_reconnect_rejects_home_contract_drift(monkeypatch, change):
    restrictions = DockerWorkloadRestrictions()
    inspection = _inspection(restrictions)
    if change == "missing":
        inspection["Config"].pop("Env")
    elif change == "wrong":
        inspection["Config"]["Env"][0] = "HOME=/"
    elif change == "duplicate":
        inspection["Config"]["Env"].append("HOME=/")

    async def fake_run_subprocess(command, **kwargs):
        if command.argv[1] == "inspect":
            return ExecResult(stdout=json.dumps(inspection))
        if "id -u" in command.argv[-1]:
            return ExecResult(stdout=restrictions.user)
        if "cayu-home-probe" in command.argv:
            return ExecResult(exit_code=1)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    with pytest.raises(DockerRuntimeConfigurationError, match="home"):
        asyncio.run(
            DockerRunner.reconnect_strict(
                "home-drift",
                container_id=_CONTAINER_ID,
                image_identity=_image_identity(),
                workload_restrictions=restrictions,
                docker_path="/usr/bin/docker",
            )
        )


def test_home_contract_is_bounded_and_part_of_profile_evidence():
    default = DockerWorkloadRestrictions()
    changed = DockerWorkloadRestrictions(home_directory="/tmp/another-home")
    assert changed.fingerprint != default.fingerprint
    profile = docker_toolchain_profile(image_identity=_image_identity())
    other_profile = docker_toolchain_profile(image_identity=_image_identity(), restrictions=changed)
    assert profile.fingerprint != other_profile.fingerprint
    assert profile.evidence()["toolchain_home_environment"] == default.home_environment
    for path in ("/", "/workspace/home", "/tmp/../workspace", "/tmp/home/.."):
        with pytest.raises(ValueError, match="home_directory"):
            DockerWorkloadRestrictions(home_directory=path)
    with pytest.raises(ValueError, match="bounded /tmp"):
        DockerWorkloadRestrictions(tmpfs=default.tmpfs[1:])


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
def test_nonpublishing_coding_binding_still_captures_bounded_git_evidence(
    tmp_path: Path,
    monkeypatch,
    outcome,
) -> None:
    from cayu.environments import docker_coding
    from cayu.tools.git import MAX_GIT_CHANGES_RESULT_BYTES

    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / ".gitignore").write_bytes(b"scratch/\n")
    (source_root / "code.py").write_bytes(b"value = 1\n")
    runner, source, binding = _coding_product_test_binding(source_root, target_root)
    binding.sync_back = "never"
    captured = []
    capture = docker_coding._capture_final_git_evidence

    async def record_capture(bound, authority):
        evidence = await capture(bound, authority)
        captured.append(evidence)
        return evidence

    monkeypatch.setattr(docker_coding, "_capture_final_git_evidence", record_capture)

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="nonpublishing-evidence")
        assert bound.workspace is not None
        await bound.workspace.write_bytes("code.py", b"value = 2\n")
        await bound.workspace.write_bytes("scratch/note.txt", b"fixture")
        assert await binding.finalize(bound, outcome=outcome) is None
        with pytest.raises(RuntimeError, match="lost its admitted authority"):
            await binding.finalize(bound, outcome=outcome)
        assert binding.abandon(bound) is True

    asyncio.run(run())
    assert len(captured) == 1
    evidence = captured[0]
    assert evidence["source_workspace_id"] == source.id
    for mode in ("status", "summary", "diff"):
        assert len(json.dumps(evidence[mode]).encode()) < 2 * MAX_GIT_CHANGES_RESULT_BYTES
    assert "+value = 2" in evidence["diff"]["content"]
    assert evidence["diff"]["structured"]["truncated"] is True
    assert "workspace_delta_unrepresented" in evidence["diff"]["structured"]["truncation_reasons"]
    assert (source_root / "code.py").read_bytes() == b"value = 1\n"
    assert not (source_root / "scratch").exists()


@pytest.mark.parametrize("sync_back", ["never", "always"])
@pytest.mark.parametrize("fault", ["error", "cancel"])
def test_docker_coding_close_failure_retains_exact_generation_for_retry(
    tmp_path: Path,
    monkeypatch,
    sync_back,
    fault,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    foreign_root = tmp_path / "foreign"
    for root in (source_root, target_root, foreign_root):
        root.mkdir()
    (source_root / "code.py").write_bytes(b"value = 1\n")
    runner, source, binding = _coding_product_test_binding(source_root, target_root)
    binding.sync_back = sync_back
    foreign_runner = _LocalDockerRunner(foreign_root)
    close_calls = 0
    original_close = runner.close

    async def close():
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            if fault == "cancel":
                raise asyncio.CancelledError("fixture close cancellation")
            raise OSError("fixture close failure")
        await original_close()

    monkeypatch.setattr(runner, "close", close)

    async def run():
        bound = await binding.bind(source, runner, session_id="close-retry")
        assert binding._completion_requires_successful_finalization(bound) is True
        assert bound.workspace is not None
        await bound.workspace.write_bytes("code.py", b"value = 2\n")
        with pytest.raises(asyncio.CancelledError if fault == "cancel" else OSError):
            await binding.finalize(bound, outcome="completed")
        assert binding.abandon(bound) is False
        assert binding._completion_finalization_recovery_state(bound) is not None
        with pytest.raises(ValueError, match="exact DockerRunner"):
            await binding.finalize(replace(bound, runner=foreign_runner), outcome="completed")
        with pytest.raises(ValueError, match="source workspace"):
            await binding.finalize(
                replace(bound, source_workspace=LocalWorkspace(foreign_root)),
                outcome="completed",
            )
        assert close_calls == 1
        assert foreign_runner._closed is False
        # Publication already finished; retrying disposal must not copy again.
        (source_root / "code.py").write_bytes(b"external edit\n")
        await binding.finalize(bound, outcome="completed")
        assert runner._closed is True
        assert binding.abandon(bound) is True
        assert binding._states == {}
        with pytest.raises(RuntimeError, match="lost its admitted authority"):
            await binding.finalize(bound, outcome="completed")
        assert close_calls == 2

    asyncio.run(run())
    assert (source_root / "code.py").read_bytes() == b"external edit\n"


@pytest.mark.parametrize("fault", ["error", "cancel"])
def test_disposal_checkpoint_failure_retains_container_and_published_snapshot(tmp_path, fault):
    from cayu.environments._finalization_disposal import finalization_disposal_checkpoint

    source_root, target_root = tmp_path / "source", tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "code.py").write_bytes(b"before")
    runner, source, binding = _coding_product_test_binding(source_root, target_root)
    binding.sync_back = "always"
    checkpoint_calls = []

    async def checkpoint(state):
        assert (source_root / "code.py").read_bytes() == b"candidate"
        assert runner._closed is False
        checkpoint_calls.append(state)
        if fault == "cancel":
            raise asyncio.CancelledError("checkpoint cancelled")
        raise OSError("checkpoint unavailable")

    async def run():
        bound = await binding.bind(source, runner, session_id="disposal-checkpoint")
        assert bound.workspace is not None
        await bound.workspace.write_bytes("code.py", b"candidate")
        token = finalization_disposal_checkpoint.set(checkpoint)
        try:
            with pytest.raises(asyncio.CancelledError if fault == "cancel" else OSError):
                await binding.finalize(bound, outcome="completed")
        finally:
            finalization_disposal_checkpoint.reset(token)
        assert runner._closed is False
        assert binding.abandon(bound) is False
        (source_root / "code.py").write_bytes(b"later edit")
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is not None
        assert runner._closed is True
        assert (source_root / "code.py").read_bytes() == b"later edit"
        assert not binding._states

    asyncio.run(run())
    assert len(checkpoint_calls) == 1


@pytest.mark.parametrize("presence", ["present", "absent", "unavailable"])
def test_disposal_recovery_validates_exact_identity_before_docker_access(
    tmp_path,
    monkeypatch,
    presence,
):
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        toolchain_profile=docker_toolchain_profile(image_identity=_image_identity()),
    )
    metadata = docker_coding_module._docker_coding_reconnect_metadata(
        container_id=_CONTAINER_ID,
        configuration_fingerprint=factory._configuration_fingerprint,
        image_fingerprint=factory.image_identity.fingerprint,
        toolchain_profile_fingerprint=factory.toolchain_profile.fingerprint,
    )
    request = EnvironmentFactoryRequest(
        session_id="disposal",
        agent_name="probe",
        environment_name="coding",
        operation=EnvironmentFactoryOperation.RECONNECT,
        reconnect_metadata=metadata,
    )
    state = {
        "version": 1,
        "kind": "docker_coding_disposal",
        "container_id": _CONTAINER_ID,
        "attachment_ids": [],
    }
    probes, closes = [], []

    async def exists(container_id, **kwargs):
        probes.append(container_id)
        if presence == "unavailable":
            raise OSError("daemon unavailable")
        return presence == "present"

    async def close(self):
        closes.append(self.container_id)

    monkeypatch.setattr(DockerRunner, "container_exists", exists)
    monkeypatch.setattr(DockerRunner, "close", close)

    async def run():
        for forged in (
            {**state, "container_id": "f" * 64},
            {**state, "attachment_ids": ["foreign"]},
        ):
            with pytest.raises(ValueError, match="exact allocation"):
                await factory.recover_finalization_disposal(request, forged)
        assert probes == closes == []
        if presence == "unavailable":
            with pytest.raises(OSError, match="daemon unavailable"):
                await factory.recover_finalization_disposal(request, state)
        else:
            await factory.recover_finalization_disposal(request, state)

    asyncio.run(run())
    assert probes == [_CONTAINER_ID]
    assert closes == ([_CONTAINER_ID] if presence == "present" else [])
