import asyncio
from pathlib import Path

from tests.workspaces.test_durable_local_workspace_branches import (
    _create_session,
    _durable_branch,
    _publication_request,
)

from cayu.storage import SQLiteSessionStore
from cayu.workspaces import WorkspaceBranchOutcomeStatus


def test_durable_publication_accepts_file_to_directory_transition(tmp_path: Path):
    async def scenario():
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "node").write_bytes(b"baseline-file")
        store = SQLiteSessionStore(tmp_path / "branches.sqlite3")
        try:
            await _create_session(store)
            _source, branch, _request = await _durable_branch(root, store)
            await branch.delete("node")
            await branch.write_bytes("node/nested/child.txt", b"child")

            result = await branch.publish(await _publication_request(branch))

            assert result.status is WorkspaceBranchOutcomeStatus.COMMITTED
            assert (root / "node/nested/child.txt").read_bytes() == b"child"
        finally:
            await store.close()

    asyncio.run(scenario())
