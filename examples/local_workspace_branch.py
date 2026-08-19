"""Create, compare, publish, and roll back bounded local workspace branches."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from cayu.environments import DeterministicWorkspaceBinding
from cayu.workspaces import (
    LocalWorkspace,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchPublicationRequest,
    WorkspaceBranchRequest,
)


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "proposal.txt").write_text("baseline\n")
        source = LocalWorkspace(root, workspace_id="proposal-workspace")

        binding = DeterministicWorkspaceBinding()
        bound = await binding.bind(source, None, session_id="branch-example")
        baseline = await binding.observe_revision(bound)
        request = WorkspaceBranchRequest(baseline=baseline)

        first_result = await source.create_branch(request)
        second_result = await source.create_branch(request)
        if (
            first_result.status is not WorkspaceBranchOutcomeStatus.CREATED
            or first_result.branch is None
            or second_result.status is not WorkspaceBranchOutcomeStatus.CREATED
            or second_result.branch is None
        ):
            raise RuntimeError("Local workspace branching was unavailable.")

        first = first_result.branch
        second = second_result.branch
        await first.write_bytes("proposal.txt", b"publish me\n")
        await second.write_bytes("proposal.txt", b"discard me\n")

        changes = await first.changes()
        publication = await first.publish(
            WorkspaceBranchPublicationRequest(
                branch_id=first.branch_id,
                baseline_revision=changes.baseline_revision,
                change_set_digest=changes.digest,
            )
        )
        if publication.status is not WorkspaceBranchOutcomeStatus.COMMITTED:
            raise RuntimeError(f"Publication did not commit: {publication.status}")
        await second.rollback()
        await binding.finalize(bound)

        assert (root / "proposal.txt").read_text() == "publish me\n"


if __name__ == "__main__":
    asyncio.run(main())
