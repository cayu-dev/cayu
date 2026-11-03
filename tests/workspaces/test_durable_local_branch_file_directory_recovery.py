import asyncio
from pathlib import Path

import pytest
from tests.workspaces.test_durable_local_workspace_branches import (
    _create_session,
    _durable_branch,
    _publication_request,
    _recovery_request,
    _workspace,
)

from cayu.storage import SQLiteSessionStore
from cayu.workspaces import WorkspaceBranchDurableState, WorkspaceBranchOutcomeStatus


@pytest.mark.parametrize("interrupted_after", ["deletion", "parents", "child"])
def test_file_to_directory_publication_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_after: str,
) -> None:
    import cayu.workspaces._local_branch as branch_module

    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "node").write_bytes(b"baseline-file")
        store = SQLiteSessionStore(tmp_path / "branches.sqlite3")
        try:
            await _create_session(store)
            _source, branch, _request = await _durable_branch(root, store)
            await branch.delete("node")
            await branch.write_bytes("node/nested/child.txt", b"child")
            publication = await _publication_request(branch)
            original_create = branch_module.create_regular

            def interrupt_creation(source_root, path, content, **kwargs):
                if interrupted_after == "parents":
                    (source_root / path).parent.mkdir(parents=True)
                elif interrupted_after == "child":
                    original_create(source_root, path, content, **kwargs)
                raise ConnectionError("publication interrupted")

            with monkeypatch.context() as fault:
                fault.setattr(branch_module, "create_regular", interrupt_creation)
                with pytest.raises(ConnectionError, match="publication interrupted"):
                    await branch.publish(publication)

            recovered = await _workspace(root, store).recover_branch(_recovery_request())
            assert recovered.state is WorkspaceBranchDurableState.COMMITTED
            assert (root / "node/nested/child.txt").read_bytes() == b"child"
            replayed = await branch.publish(publication)
            assert replayed.status is WorkspaceBranchOutcomeStatus.COMMITTED
        finally:
            await store.close()

    asyncio.run(scenario())


def test_file_to_directory_publication_preserves_changed_ancestor(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "node").write_bytes(b"baseline-file")
        store = SQLiteSessionStore(tmp_path / "branches.sqlite3")
        try:
            await _create_session(store)
            _source, branch, _request = await _durable_branch(root, store)
            await branch.delete("node")
            await branch.write_bytes("node/nested/child.txt", b"child")
            publication = await _publication_request(branch)
            (root / "node").write_bytes(b"external")

            result = await branch.publish(publication)
            assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
            assert (root / "node").read_bytes() == b"external"

            (root / "node").write_bytes(b"baseline-file")
            retried = await branch.publish(publication)
            assert retried.status is WorkspaceBranchOutcomeStatus.COMMITTED
            assert (root / "node/nested/child.txt").read_bytes() == b"child"
        finally:
            await store.close()

    asyncio.run(scenario())
