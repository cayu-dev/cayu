from __future__ import annotations

from cayu.workspaces import (
    WorkspaceBranch,
    WorkspaceBranchClosedError,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchPublicationRequest,
    WorkspaceBranchResourceExhaustedError,
)


async def verify_branch_isolation_and_net_changes(
    source,
    first: WorkspaceBranch,
    second: WorkspaceBranch,
) -> None:
    assert (await first.read_bytes("original.txt")).content == b"original"
    assert (await second.read_bytes("original.txt")).content == b"original"

    await first.write_bytes("original.txt", b"first")
    await first.create_bytes("created.txt", b"created")
    await first.delete("deleted.txt")

    assert (await source.read_bytes("original.txt")).content == b"original"
    assert (await second.read_bytes("original.txt")).content == b"original"
    assert "created.txt" not in (await source.list()).paths
    assert "created.txt" not in (await second.list()).paths
    assert (await source.read_bytes("deleted.txt")).content == b"delete-me"

    changes = await first.changes()
    assert [(entry.path, entry.operation) for entry in changes.changes] == [
        ("created.txt", "created"),
        ("deleted.txt", "deleted"),
        ("original.txt", "modified"),
    ]
    assert all(
        entry.before is None or not hasattr(entry.before, "content") for entry in changes.changes
    )

    await first.write_bytes("original.txt", b"original")
    await first.delete("created.txt")
    await first.write_bytes("deleted.txt", b"delete-me")
    assert (await first.changes()).changes == ()


async def verify_atomic_publication(
    source,
    branch: WorkspaceBranch,
    baseline_revision: str,
) -> None:
    await branch.write_bytes("original.txt", b"published")
    await branch.create_bytes("nested/new.txt", b"new")
    await branch.delete("deleted.txt")
    changes = await branch.changes()
    result = await branch.publish(
        WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=baseline_revision,
            change_set_digest=changes.digest,
        )
    )
    assert result.status is WorkspaceBranchOutcomeStatus.COMMITTED
    assert (await source.read_bytes("original.txt")).content == b"published"
    assert (await source.read_bytes("nested/new.txt")).content == b"new"
    assert "deleted.txt" not in (await source.list()).paths


async def verify_conflict_is_all_or_none(
    source,
    branch: WorkspaceBranch,
    baseline_revision: str,
) -> None:
    await branch.write_bytes("original.txt", b"branch-original")
    await branch.write_bytes("deleted.txt", b"branch-deleted")
    changes = await branch.changes()
    await source.write_bytes("deleted.txt", b"source-conflict")
    result = await branch.publish(
        WorkspaceBranchPublicationRequest(
            branch_id=branch.branch_id,
            baseline_revision=baseline_revision,
            change_set_digest=changes.digest,
        )
    )
    assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
    assert tuple(conflict.path for conflict in result.conflicts) == ("deleted.txt",)
    assert (await source.read_bytes("original.txt")).content == b"original"
    assert (await source.read_bytes("deleted.txt")).content == b"source-conflict"


async def verify_bound_rollback_and_cleanup(source, branch: WorkspaceBranch) -> None:
    try:
        await branch.write_bytes("oversized.txt", b"four")
    except WorkspaceBranchResourceExhaustedError as error:
        assert error.detail_code == "file_byte_limit_exceeded"
    else:  # pragma: no cover - a conforming bounded branch must reject this
        raise AssertionError("Workspace branch accepted a file beyond its configured bound.")

    await branch.write_bytes("temporary.txt", b"ok")
    assert [(change.path, change.operation) for change in (await branch.changes()).changes] == [
        ("temporary.txt", "created")
    ]
    first = await branch.rollback()
    second = await branch.rollback()
    assert first == second
    assert first.status is WorkspaceBranchOutcomeStatus.ROLLED_BACK
    assert "temporary.txt" not in (await source.list()).paths
    try:
        await branch.read_bytes("temporary.txt")
    except WorkspaceBranchClosedError:
        pass
    else:  # pragma: no cover - cleanup must close the speculative view
        raise AssertionError("Rolled-back workspace branch remained readable.")
