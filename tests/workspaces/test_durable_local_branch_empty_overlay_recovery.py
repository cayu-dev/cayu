from __future__ import annotations

import asyncio
import os
import signal
import time
from pathlib import Path

import pytest
from tests.workspaces.test_durable_local_workspace_branches import (
    _create_session,
    _durable_branch,
    _isolate_retained_branch_state,
    _recovery_request,
    _workspace,
)

from cayu.storage import SQLiteSessionStore
from cayu.workspaces import WorkspaceBranchDurableState, WorkspaceBranchRollbackRequest


@pytest.mark.parametrize("restore_baseline", [False, True])
@pytest.mark.parametrize("replacement_path", ["partial", "PARTIAL"])
@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX process-loss recovery")
def test_recovered_overlay_does_not_shadow_files_with_uncommitted_empty_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_baseline: bool, replacement_path: str
) -> None:
    import cayu.workspaces._local_branch as branch_module

    _isolate_retained_branch_state(monkeypatch, branch_module)
    root = tmp_path / "workspace"
    root.mkdir()
    if restore_baseline:
        (root / "partial").write_bytes(b"replacement file")
    database = tmp_path / "branches.sqlite3"
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - the parent checks the crash and recovery
        try:
            import cayu.workspaces._local_guard as guard_module

            async def interrupted_writer() -> None:
                store = SQLiteSessionStore(database)
                await _create_session(store)
                _source, branch, _request = await _durable_branch(root, store)
                await branch.write_bytes("retained.txt", b"acknowledged work")
                if restore_baseline:
                    await branch.delete("partial")

                def crash_before_staging_file(*_args, **_kwargs):
                    # _open_parent(create=True) has already created partial/nested,
                    # but no file or new journal state has been published yet.
                    os._exit(77)

                guard_module._write_temp = crash_before_staging_file
                await branch.write_bytes("partial/nested/leaf.txt", b"unacknowledged")

            asyncio.run(interrupted_writer())
        finally:
            os._exit(90)

    deadline = time.monotonic() + 15
    while True:
        exited_pid, status = os.waitpid(child_pid, os.WNOHANG)
        if exited_pid:
            assert os.waitstatus_to_exitcode(status) == 77
            break
        if time.monotonic() > deadline:
            os.kill(child_pid, signal.SIGKILL)
            os.waitpid(child_pid, 0)
            pytest.fail("writer did not reach the private-write crash boundary")
        time.sleep(0.02)

    async def recover() -> None:
        store = SQLiteSessionStore(database)
        try:
            recovered = await _workspace(root, store).recover_branch(_recovery_request())
            assert recovered.state is WorkspaceBranchDurableState.OPEN
            assert recovered.branch is not None
            branch = recovered.branch
            # Recovery is observational: an active writer could own these
            # directories. Removal belongs to the later guarded file mutation.
            assert (branch._overlay_root / "partial/nested").is_dir()
            assert (await branch.read_bytes("retained.txt")).content == b"acknowledged work"
            assert (await branch.list()).paths == ("retained.txt",)

            # The recovered logical view contains no "partial" directory. Its
            # name must remain usable as a file, just as on a fresh branch.
            await branch.write_bytes(replacement_path, b"replacement file")
            assert (await branch.read_bytes(replacement_path)).content == b"replacement file"
            assert (await branch.read_bytes("retained.txt")).content == b"acknowledged work"
            await branch.rollback(
                WorkspaceBranchRollbackRequest(
                    branch_id=branch.branch_id,
                    idempotency_key="rollback-alpha",
                    expected_run_epoch=0,
                    binding_generation="binding-1",
                )
            )
        finally:
            await store.close()

    asyncio.run(recover())
