from __future__ import annotations

import asyncio
from pathlib import Path

from tests.workspaces.test_durable_local_workspace_branches import (
    _create_session,
    _durable_branch,
)

from cayu.environments import SyncBinding
from cayu.storage import SQLiteSessionStore
from cayu.workspaces import LocalWorkspace, WorkspaceBranchRollbackRequest


def test_session_scoped_branches_can_bind_independent_targets(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        targets = [tmp_path / "target-a", tmp_path / "target-b"]
        for target in targets:
            target.mkdir()
        store = SQLiteSessionStore(tmp_path / "branches.sqlite3")
        branches = []
        bindings = []
        bound_workspaces = []
        try:
            for session in ("session-a", "session-b"):
                await _create_session(store, session_id=session)
                _source, branch, _request = await _durable_branch(
                    root, store, session_id=session, branch_id="proposal"
                )
                branches.append(branch)
            first, second = branches
            assert first._private_root != second._private_root
            await first.write_bytes("proposal.txt", b"proposal from session a")
            await second.write_bytes("proposal.txt", b"proposal from session b")

            for index, (branch, target) in enumerate(zip(branches, targets, strict=True)):
                binding = SyncBinding(
                    target_workspace=LocalWorkspace(target, workspace_id=f"target-{index}"),
                    sync_back="never",
                )
                bindings.append(binding)
                bound_workspaces.append(
                    await binding.bind(branch, None, session_id=f"consumer-{index}")
                )

            assert (targets[0] / "proposal.txt").read_bytes() == b"proposal from session a"
            assert (targets[1] / "proposal.txt").read_bytes() == b"proposal from session b"
            assert first.resource_key != second.resource_key
        finally:
            for binding, bound in zip(bindings, bound_workspaces):  # noqa: B905
                binding.abandon(bound)
            for branch in branches:
                await branch.rollback(
                    WorkspaceBranchRollbackRequest(
                        branch_id=branch.branch_id,
                        idempotency_key="cleanup",
                        expected_run_epoch=0,
                        binding_generation="binding-1",
                    )
                )
            await store.close()

    asyncio.run(scenario())
