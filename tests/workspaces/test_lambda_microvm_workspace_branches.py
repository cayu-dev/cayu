"""Durable workspace branches on the retained Lambda MicroVM and Docker filesystems."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

pytest.importorskip("fcntl")

from tests.runners.lambda_microvm_harness import ClientTokenLambdaModel, SupervisorTransport
from tests.workspaces.branch_conformance import (
    verify_atomic_publication,
    verify_branch_isolation_and_net_changes,
    verify_conflict_is_all_or_none,
)
from tests.workspaces.test_runner_workspace_branches import _request

from cayu import LambdaMicroVMRunner
from cayu.runners import DockerRunner, RemoteWorkspaceBranchCapability
from cayu.workspaces import (
    RunnerWorkspace,
    WorkspaceBranchBindingAuthority,
    WorkspaceBranchBindingAuthorityRegistry,
    WorkspaceBranchOutcomeStatus,
)


def _lambda_runner(root: Path) -> LambdaMicroVMRunner:
    return LambdaMicroVMRunner(
        ClientTokenLambdaModel(),
        microvm_id="microvm-branches",
        endpoint="microvm-branches.lambda-microvm.invalid",
        endpoint_transport=SupervisorTransport(root),
        default_cwd=str(root),
        poll_interval_s=0,
    )


def test_lambda_microvm_declares_its_retained_filesystem_exactly(tmp_path: Path) -> None:
    runner = _lambda_runner(tmp_path)

    capability = runner.workspace_capability(RemoteWorkspaceBranchCapability)

    assert capability is not None
    assert capability.resource_key == ("lambda-microvm", "microvm-branches")
    assert capability.allocation_fingerprint == (
        "sha256:" + hashlib.sha256(b"lambda-microvm\0microvm-branches").hexdigest()
    )

    class Wrapped(LambdaMicroVMRunner):
        pass

    wrapped = Wrapped(
        ClientTokenLambdaModel(),
        microvm_id="microvm-branches",
        endpoint="microvm-branches.lambda-microvm.invalid",
        endpoint_transport=SupervisorTransport(tmp_path),
    )
    assert wrapped.workspace_capability(RemoteWorkspaceBranchCapability) is None


def test_docker_declares_branches_only_for_an_exact_container_id() -> None:
    container_id = "c" * 64
    exact = DockerRunner(container_id, _container_id=container_id)
    named = DockerRunner("legacy-name")

    capability = exact.workspace_capability(RemoteWorkspaceBranchCapability)

    assert capability is not None
    assert capability.resource_key == ("docker", container_id)
    assert capability.allocation_fingerprint == (
        "sha256:" + hashlib.sha256(f"docker\0{container_id}".encode()).hexdigest()
    )
    assert named.workspace_capability(RemoteWorkspaceBranchCapability) is None

    class Wrapped(DockerRunner):
        pass

    wrapped = Wrapped(container_id, _container_id=container_id)
    assert wrapped.workspace_capability(RemoteWorkspaceBranchCapability) is None


def test_lambda_microvm_workspace_branches_share_conformance(tmp_path: Path) -> None:
    async def scenario() -> None:
        # Branch state lives beside the workspace root, so the root must sit in
        # an agent-writable directory such as cwd="project" under /workspace.
        root = tmp_path / "project"
        root.mkdir()
        runner = _lambda_runner(tmp_path)
        source = RunnerWorkspace(
            runner,
            cwd="project",
            workspace_id="lambda-branches",
            enable_workspace_branches=True,
            branch_authority_resolver=WorkspaceBranchBindingAuthorityRegistry(
                WorkspaceBranchBindingAuthority(
                    environment_name="sandbox",
                    binding_generation="generation-1",
                    binding_identity="microvm-branches",
                )
            ),
        )
        await source.write_bytes("original.txt", b"original")
        await source.write_bytes("deleted.txt", b"delete-me")

        request = await _request(source)
        first = await source.create_branch(request)
        second = await source.create_branch(request)
        assert first.status is WorkspaceBranchOutcomeStatus.CREATED
        assert second.status is WorkspaceBranchOutcomeStatus.CREATED
        assert first.branch is not None and second.branch is not None
        await verify_branch_isolation_and_net_changes(source, first.branch, second.branch)
        await first.branch.rollback()
        await second.branch.rollback()

        publication = await source.create_branch(await _request(source))
        assert publication.branch is not None
        assert publication.evidence.baseline_revision is not None
        await verify_atomic_publication(
            source, publication.branch, publication.evidence.baseline_revision
        )

        await source.write_bytes("original.txt", b"original")
        await source.write_bytes("deleted.txt", b"delete-me")
        conflict = await source.create_branch(await _request(source))
        assert conflict.branch is not None
        assert conflict.evidence.baseline_revision is not None
        await verify_conflict_is_all_or_none(
            source, conflict.branch, conflict.evidence.baseline_revision
        )
        await conflict.branch.rollback()

    asyncio.run(scenario())
