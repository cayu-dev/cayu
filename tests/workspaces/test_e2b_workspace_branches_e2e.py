"""Opt-in live proof for retained E2B workspace branches."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fcntl")
pytest.importorskip("e2b")

from tests.workspaces.branch_conformance import (
    verify_atomic_publication,
    verify_bound_rollback_and_cleanup,
    verify_branch_isolation_and_net_changes,
    verify_conflict_is_all_or_none,
)
from tests.workspaces.remote_branch_live_support import (
    FileWorkspaceBranchAuthorityProvider,
)

from cayu import (
    E2BRunner,
    RunnerWorkspace,
    WorkspaceBranchAuthority,
    WorkspaceBranchBindingAuthority,
    WorkspaceBranchLimits,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchRecoveryRequest,
    WorkspaceBranchRequest,
)
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_RUN_E2B_WORKSPACE_BRANCH_E2E") != "1" or not os.environ.get("E2B_API_KEY"),
    reason="Set CAYU_RUN_E2B_WORKSPACE_BRANCH_E2E=1 and E2B_API_KEY.",
)


def test_e2b_retained_workspace_branch_recovers_and_publishes(tmp_path: Path) -> None:
    asyncio.run(_drive_live_branch(tmp_path / "binding-authority.json"))


async def _drive_live_branch(authority_path: Path) -> None:
    runner: E2BRunner | None = None
    cleanup: E2BRunner | None = None
    sandbox_id: str | None = None
    binding = WorkspaceBranchBindingAuthority(
        environment_name="e2b-live-branch",
        binding_generation="generation-1",
        binding_identity="binding-1",
    )
    resolver = FileWorkspaceBranchAuthorityProvider(authority_path, initial=binding)
    authority = WorkspaceBranchAuthority(
        session_id="e2b-live-session",
        expected_run_epoch=1,
        environment_name=binding.environment_name,
        binding_generation=binding.binding_generation,
        binding_identity=binding.binding_identity,
        creating_authority="e2b-live-worker-1",
        resource_policy="e2b-live-branch-defaults",
    )
    resolver.authorize_operation(authority)
    try:
        runner = await E2BRunner.create(
            template=os.environ.get("CAYU_E2B_TEMPLATE"),
            sandbox_timeout_s=int(os.environ.get("CAYU_E2B_SANDBOX_TIMEOUT_S", "300")),
            close_action="none",
        )
        sandbox_id = runner.sandbox_id
        source = RunnerWorkspace(
            runner,
            workspace_id="e2b-live-workspace",
            enable_workspace_branches=True,
            branch_authority_resolver=resolver,
        )
        await source.write_bytes("original.txt", b"original")
        await source.write_bytes("deleted.txt", b"delete-me")

        first = await source.create_branch(await _live_request(source))
        second = await source.create_branch(await _live_request(source))
        assert first.branch is not None
        assert second.branch is not None
        await verify_branch_isolation_and_net_changes(source, first.branch, second.branch)
        await first.branch.rollback()
        await second.branch.rollback()

        publication = await source.create_branch(await _live_request(source))
        assert publication.branch is not None
        assert publication.evidence.baseline_revision is not None
        await verify_atomic_publication(
            source,
            publication.branch,
            publication.evidence.baseline_revision,
        )
        await source.write_bytes("original.txt", b"original")
        await source.write_bytes("deleted.txt", b"delete-me")
        await source.delete("nested/new.txt")

        conflict = await source.create_branch(await _live_request(source))
        assert conflict.branch is not None
        assert conflict.evidence.baseline_revision is not None
        await verify_conflict_is_all_or_none(
            source,
            conflict.branch,
            conflict.evidence.baseline_revision,
        )
        await conflict.branch.rollback()
        await source.write_bytes("deleted.txt", b"delete-me")

        await source.delete("original.txt")
        await source.delete("deleted.txt")
        bounded = await source.create_branch(
            await _live_request(source, limits=WorkspaceBranchLimits(max_file_bytes=3))
        )
        assert bounded.branch is not None
        await verify_bound_rollback_and_cleanup(source, bounded.branch)

        observation = await observe_deterministic_workspace(
            source,
            observer="e2b-live-branch-proof",
            limits=WorkspaceRevisionObservationLimits(),
        )
        created = await source.create_branch(
            WorkspaceBranchRequest(
                baseline=observation,
                branch_id="e2b-live-durable-branch",
                idempotency_key="create-1",
                authority=authority,
            )
        )
        assert created.branch is not None
        await created.branch.write_bytes("original.txt", b"published-after-reconnect")
        changes = await created.branch.changes()
        await runner.close()
        runner = None

        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.workspaces.e2b_branch_fresh_process",
            str(authority_path),
            sandbox_id,
            changes.branch_id,
            authority.session_id,
            str(authority.expected_run_epoch),
            authority.binding_generation,
            authority.binding_identity,
            changes.baseline_revision,
            changes.digest,
            cwd=Path(__file__).resolve().parents[2],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _stderr = await asyncio.wait_for(child.communicate(), timeout=180)
        except BaseException:
            child.kill()
            await child.wait()
            raise
        child_stderr = _stderr.decode("utf-8", errors="replace")[-4096:]
        assert child.returncode == 0, f"fresh E2B branch process failed: {child_stderr}"
        assert json.loads(stdout.splitlines()[-1]) == {"status": "committed"}

        cleanup = await E2BRunner.from_existing(
            sandbox_id,
            close_action="kill",
        )
        fresh = RunnerWorkspace(
            cleanup,
            workspace_id="e2b-live-workspace",
            enable_workspace_branches=True,
            branch_authority_resolver=resolver,
        )
        recovery_request = WorkspaceBranchRecoveryRequest(
            branch_id="e2b-live-durable-branch",
            session_id=authority.session_id,
            expected_run_epoch=authority.expected_run_epoch,
            binding_generation=authority.binding_generation,
            binding_identity=authority.binding_identity,
            recovery_id="recover-parent-terminal",
        )
        recovered = await fresh.recover_branch(recovery_request)
        assert recovered.publication is not None
        assert recovered.publication.status is WorkspaceBranchOutcomeStatus.COMMITTED
        assert (await fresh.read_bytes("original.txt")).content == b"published-after-reconnect"
    finally:
        if cleanup is not None:
            await cleanup.close()
        elif runner is not None:
            await runner.close()
        elif sandbox_id is not None:
            cleanup = await E2BRunner.from_existing(sandbox_id, close_action="kill")
            await cleanup.close()


async def _live_request(
    source: RunnerWorkspace,
    *,
    limits: WorkspaceBranchLimits | None = None,
) -> WorkspaceBranchRequest:
    observation = await observe_deterministic_workspace(
        source,
        observer="e2b-live-shared-conformance",
        limits=WorkspaceRevisionObservationLimits(),
    )
    return WorkspaceBranchRequest(
        baseline=observation,
        limits=limits or WorkspaceBranchLimits(),
    )
