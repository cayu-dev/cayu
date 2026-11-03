import asyncio
from pathlib import Path

import pytest
from tests.workspaces.test_durable_local_workspace_branches import (
    _binding_authority,
    _create_session,
    _durable_branch,
    _DurableAuthorityProvider,
    _isolate_retained_branch_state,
    _publication_request,
    _recovery_request,
)

from cayu.storage import SQLiteSessionStore
from cayu.workspaces import WorkspaceBranchDurableState, WorkspaceBranchOperationConflict


def test_terminal_cas_loser_does_not_poison_later_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    async def scenario():
        import cayu.workspaces._local_branch as branch_module

        _isolate_retained_branch_state(monkeypatch, branch_module)
        root = tmp_path / "workspace"
        root.mkdir()
        store = SQLiteSessionStore(tmp_path / "branches.sqlite3")
        reached_terminal = asyncio.Event()
        release_terminal = asyncio.Event()
        original_publish = branch_module._publish_or_reconcile_exact_record
        first_task = None

        async def pause_first_terminal(store_arg, **kwargs):
            if (
                asyncio.current_task() is first_task
                and kwargs["updated"].state is WorkspaceBranchDurableState.COMMITTED
            ):
                reached_terminal.set()
                await release_terminal.wait()
            return await original_publish(store_arg, **kwargs)

        monkeypatch.setattr(
            branch_module, "_publish_or_reconcile_exact_record", pause_first_terminal
        )
        try:
            await _create_session(store)
            provider = _DurableAuthorityProvider(_binding_authority())
            source, branch, _request = await _durable_branch(root, store, resolver=provider)
            await branch.write_bytes("answer.txt", b"candidate")
            publication = await _publication_request(branch)
            first_task = asyncio.create_task(branch.publish(publication))
            try:
                await asyncio.wait_for(reached_terminal.wait(), timeout=5)
                # A second owner observes PUBLICATION_PROGRESS, verifies the same
                # source result, and commits it at a newer record revision.
                recovered = await source.recover_branch(_recovery_request())
                assert recovered.state is WorkspaceBranchDurableState.COMMITTED
            finally:
                release_terminal.set()
            # Even if the losing publisher reports its CAS conflict, an already
            # committed successor must not leave an unreconcilable retained owner.
            try:  # noqa: SIM105
                await first_task
            except WorkspaceBranchOperationConflict:
                pass

            retried = await source.recover_branch(_recovery_request(recovery_id="retry"))
            assert retried.state is WorkspaceBranchDurableState.COMMITTED
            provider.replace(
                _binding_authority(generation="binding-2", identity="workspace-alpha@binding-2")
            )
        finally:
            release_terminal.set()
            if first_task is not None and not first_task.done():
                await asyncio.gather(first_task, return_exceptions=True)
            await store.close()

    asyncio.run(scenario())
