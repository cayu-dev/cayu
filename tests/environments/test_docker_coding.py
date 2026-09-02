from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

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
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    ExecCommand,
    ExecutionAdmissionError,
    ExecutionRequirements,
    LocalRunner,
    SyncBinding,
    SyncBindingSourceConflictError,
    evaluate_execution_admission,
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
from cayu.workspaces import LocalWorkspace, RunnerWorkspace
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)

_CONTAINER_ID = "a" * 64
_IMAGE_ID = "sha256:" + ("b" * 64)
_IMAGE_REFERENCE = "cayu/coding@sha256:" + ("c" * 64)


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
        "Config": {"Image": _IMAGE_REFERENCE, "User": restrictions.user},
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
            "Devices": [],
            "DeviceRequests": [],
        },
        "Mounts": [],
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
        image_identity=_image_identity(),
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
        image_identity=_image_identity(),
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
        image_identity=_image_identity(),
        required_executables=("ruff",),
        transfer_limits=DockerWorkspaceTransferLimits(max_files=42),
    )
    candidate = factory.construction_admission_candidate()

    assert candidate.candidate == "docker"
    assert candidate.evidence.environment_fingerprint is not None
    assert candidate.evidence.image_fingerprint == _image_identity().fingerprint
    assert candidate.evidence.tool_requirements is not None
    assert [item.executable for item in candidate.evidence.tool_requirements.executables] == [
        "git",
        "python3",
        "rm",
        "ruff",
        "sh",
        "sleep",
    ]
    assert all(
        item.state == "declared" for item in candidate.evidence.tool_requirements.executables
    )
    assert factory.execution_profile_identity.implementation_version.startswith("sha256:")
    default_factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        image_identity=_image_identity(),
        transfer_limits=DockerWorkspaceTransferLimits(max_files=42),
    )
    assert (
        factory.execution_profile_identity.implementation_version
        != default_factory.execution_profile_identity.implementation_version
    )


def test_docker_coding_factory_refuses_weakened_privilege_restrictions(
    tmp_path: Path,
) -> None:
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        image_identity=_image_identity(),
        restrictions=DockerWorkloadRestrictions(
            read_only_root=False,
            no_new_privileges=False,
            capability_add=("SYS_ADMIN",),
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
        image_identity=_image_identity(),
        restrictions=restrictions,
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


def test_docker_coding_factory_structurally_refuses_a_missing_final_executable(
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
        if docker_args[:2] == ["exec", _CONTAINER_ID] and "command -v git" in docker_args[-1]:
            return ExecResult(exit_code=127)
        return ExecResult()

    monkeypatch.setattr("cayu.runners.docker.run_subprocess", fake_run_subprocess)
    factory = DockerCodingEnvironmentFactory(
        source_workspace=LocalWorkspace(tmp_path),
        image_identity=_image_identity(),
        restrictions=restrictions,
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


def test_docker_coding_binding_refuses_publishable_git_ignored_paths(
    tmp_path: Path,
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
        bound = await binding.bind(source, runner, session_id="ignored-publication")
        (target_root / "ignored.txt").write_text("hidden mutation\n", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Git would omit a source path"):
            await binding.finalize(bound, outcome="completed")
        assert binding.abandon(bound) is True

    asyncio.run(run())

    assert not (source_root / "ignored.txt").exists()
