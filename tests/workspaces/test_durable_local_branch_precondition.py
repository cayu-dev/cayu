import asyncio
from pathlib import Path

import pytest
from tests.workspaces.test_durable_local_workspace_branches import (
    _create_session,
    _durable_branch,
)

from cayu.storage import SQLiteSessionStore
from cayu.workspaces import WorkspaceBranchLifecycleStatus


def test_rejected_create_does_not_fence_a_durable_branch(tmp_path: Path):
    async def scenario():
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "answer.txt").write_bytes(b"baseline")
        store = SQLiteSessionStore(tmp_path / "branches.sqlite3")
        try:
            await _create_session(store)
            _source, branch, _request = await _durable_branch(root, store)
            with pytest.raises(FileExistsError):
                await branch.create_bytes("answer.txt", b"must not replace")

            assert branch.lifecycle_status is WorkspaceBranchLifecycleStatus.ACTIVE
            assert (await branch.read_bytes("answer.txt")).content == b"baseline"
            await branch.write_bytes("answer.txt", b"valid edit")
        finally:
            await store.close()

    asyncio.run(scenario())
