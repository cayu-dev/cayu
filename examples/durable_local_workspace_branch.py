"""Recover and publish a durable local copy-on-write workspace branch."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from cayu import (
    LocalWorkspace,
    RunRequest,
    SessionIdentity,
    SessionWorkspaceBranchStore,
    SQLiteSessionStore,
    WorkspaceBranchAuthority,
    WorkspaceBranchBindingAuthority,
    WorkspaceBranchBindingAuthorityRegistry,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchPublicationRequest,
    WorkspaceBranchRecoveryRequest,
    WorkspaceBranchRequest,
    WorkspaceRevisionObservationLimits,
)
from cayu.workspaces.revisions import observe_deterministic_workspace


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_root = root / "source"
        source_root.mkdir()
        (source_root / "proposal.txt").write_text("baseline\n")
        database = root / "sessions.sqlite3"
        store = SQLiteSessionStore(database)
        binding_authority = WorkspaceBranchBindingAuthority(
            environment_name="local",
            binding_generation="proposal-binding-1",
            binding_identity="proposal-workspace@1",
        )
        binding_authority_provider = WorkspaceBranchBindingAuthorityRegistry(binding_authority)
        session = await store.create(
            RunRequest(
                agent_name="durable-branch-example",
                session_id="proposal-session",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="example", model="none"),
        )
        source = LocalWorkspace(
            source_root,
            workspace_id="proposal-workspace",
            branch_store=SessionWorkspaceBranchStore(store),
            branch_authority_resolver=binding_authority_provider,
        )
        baseline = await observe_deterministic_workspace(
            source,
            observer="durable-branch-example",
            limits=WorkspaceRevisionObservationLimits(),
        )
        created = await source.create_branch(
            WorkspaceBranchRequest(
                baseline=baseline,
                branch_id="proposal-alpha",
                idempotency_key="create-proposal-alpha",
                authority=WorkspaceBranchAuthority(
                    session_id=session.id,
                    expected_run_epoch=session.run_epoch,
                    environment_name="local",
                    binding_generation="proposal-binding-1",
                    binding_identity="proposal-workspace@1",
                    creating_authority="example",
                    resource_policy="bounded-local-cow-v1",
                ),
            )
        )
        if created.branch is None:
            raise RuntimeError(f"Branch creation failed: {created.status}")
        await created.branch.write_bytes("proposal.txt", b"recovered publication\n")

        # A replacement process reopens the same store and source, then owns recovery.
        recovered_store = SQLiteSessionStore(database)
        replacement = LocalWorkspace(
            source_root,
            workspace_id="proposal-workspace",
            branch_store=SessionWorkspaceBranchStore(recovered_store),
            branch_authority_resolver=binding_authority_provider,
        )
        recovered = await replacement.recover_branch(
            WorkspaceBranchRecoveryRequest(
                branch_id="proposal-alpha",
                session_id=session.id,
                expected_run_epoch=session.run_epoch,
                binding_generation="proposal-binding-1",
                binding_identity="proposal-workspace@1",
                recovery_id="recover-proposal-alpha",
            )
        )
        if recovered.branch is None:
            raise RuntimeError(f"Branch is not publishable: {recovered.state}")
        changes = await recovered.branch.changes()
        publication = await recovered.branch.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=recovered.branch.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
                idempotency_key="publish-proposal-alpha",
                expected_run_epoch=session.run_epoch,
                binding_generation="proposal-binding-1",
            )
        )
        if publication.status is not WorkspaceBranchOutcomeStatus.COMMITTED:
            raise RuntimeError(f"Publication did not commit: {publication.status}")
        assert (source_root / "proposal.txt").read_text() == "recovered publication\n"


if __name__ == "__main__":
    asyncio.run(main())
