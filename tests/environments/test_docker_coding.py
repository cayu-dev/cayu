from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from cayu import (
    DockerCodingEnvironmentFactory,
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
from cayu.runners.base import ExecResult
from cayu.runners.docker import (
    DockerContainerOwnershipError,
    DockerRunner,
    DockerRuntimeConfigurationError,
)
from cayu.workspaces import LocalWorkspace, RunnerWorkspace

_CONTAINER_ID = "a" * 64
_IMAGE_ID = "sha256:" + ("b" * 64)
_IMAGE_REFERENCE = "cayu/coding@sha256:" + ("c" * 64)


def _image_identity() -> DockerImageIdentity:
    return DockerImageIdentity(reference=_IMAGE_REFERENCE)


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

    runner = LocalDockerRunner(target_root)
    source = LocalWorkspace(source_root, workspace_id="source")
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
            max_archive_bytes=16 * 1024,
        ),
    )

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="docker-binding")
        assert (target_root / ".git").is_dir()
        assert not (target_root / ".git" / "objects" / "host-only").exists()
        assert not (target_root / ".cayu").exists()
        assert not (target_root / ".runtime").exists()
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
