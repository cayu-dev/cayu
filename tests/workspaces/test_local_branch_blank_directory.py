from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from tests.workspaces.test_durable_local_workspace_branches import (
    _create_session,
    _recovery_request,
    _workspace,
)

from cayu.storage import SQLiteSessionStore
from cayu.workspaces import (
    LocalWorkspace,
    WorkspaceBranchAuthority,
    WorkspaceBranchDurableState,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchRequest,
    WorkspaceRevisionObservationLimits,
)
from cayu.workspaces.revisions import observe_deterministic_workspace


@pytest.mark.parametrize("durable", [False, True])
def test_blank_source_directory_returns_typed_unsupported(tmp_path: Path, durable: bool) -> None:
    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        # Empty directories do not invalidate a complete file observation.
        # This is a legal POSIX name, but not a branch-addressable path.
        (root / "   ").mkdir()
        store = SQLiteSessionStore(tmp_path / "branches.sqlite3")
        try:
            if durable:
                await _create_session(store)
                source = _workspace(root, store)
            else:
                source = LocalWorkspace(root, workspace_id="workspace-alpha")
            baseline = await observe_deterministic_workspace(
                source,
                observer="durable-branch-tests",
                limits=WorkspaceRevisionObservationLimits(),
            )
            assert baseline.revision is not None
            assert baseline.paths == ()
            request = WorkspaceBranchRequest(
                baseline=baseline,
                **(
                    {
                        "branch_id": "branch-alpha",
                        "idempotency_key": "create-alpha",
                        "authority": WorkspaceBranchAuthority(
                            session_id="branch-session",
                            expected_run_epoch=0,
                            environment_name="local",
                            binding_generation="binding-1",
                            binding_identity="workspace-alpha@binding-1",
                            creating_authority="session-fork:alpha",
                            resource_policy="local-cow-defaults-v1",
                        ),
                    }
                    if durable
                    else {}
                ),
            )

            result = await source.create_branch(request)

            assert result.status is WorkspaceBranchOutcomeStatus.UNSUPPORTED
            assert result.branch is None
            assert (root / "   ").is_dir()
            assert not tuple(tmp_path.glob(".cayu-*"))
            if durable:
                recovered = await source.recover_branch(_recovery_request())
                assert recovered.state is WorkspaceBranchDurableState.FAILED
                assert recovered.evidence.outcome is WorkspaceBranchOutcomeStatus.UNSUPPORTED
        finally:
            await store.close()

    asyncio.run(scenario())
