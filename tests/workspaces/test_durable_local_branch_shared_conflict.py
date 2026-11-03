import asyncio
from pathlib import Path

from tests.workspaces.test_durable_local_workspace_branches import (
    _create_session,
    _durable_branch,
    _publication_request,
)

from cayu.storage import SQLiteSessionStore
from cayu.workspaces import WorkspaceBranchOutcomeStatus


def test_publication_deduplicates_a_conflicting_source_ancestor(tmp_path: Path):
    async def scenario():
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "node").mkdir()
        (root / "node/a.txt").write_bytes(b"a")
        (root / "node/b.txt").write_bytes(b"b")
        store = SQLiteSessionStore(tmp_path / "branches.sqlite3")
        try:
            await _create_session(store)
            source, branch, _request = await _durable_branch(root, store)
            await branch.delete("node/a.txt")
            await branch.delete("node/b.txt")
            publication = await _publication_request(branch)
            await source.delete("node/a.txt")
            await source.delete("node/b.txt")
            (root / "node").rmdir()
            await source.write_bytes("node", b"external file")

            result = await branch.publish(publication)

            assert result.status is WorkspaceBranchOutcomeStatus.CONFLICTED
            assert [conflict.path for conflict in result.conflicts] == ["node"]
            assert (root / "node").read_bytes() == b"external file"
        finally:
            await store.close()

    asyncio.run(scenario())
