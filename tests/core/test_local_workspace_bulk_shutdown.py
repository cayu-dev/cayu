from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cayu.workspaces import LocalWorkspace
from cayu.workspaces import local as local_module


def test_bulk_read_shutdown_does_not_leak_source_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CAYU_WORKSPACE_TRANSFER_PROCESSES", "1")
    if not local_module.transfers_use_helpers():
        pytest.skip("requires bulk transfer helpers")

    (tmp_path / "file.txt").write_bytes(b"content")
    (tmp_path / "empty").mkdir()
    workspace = LocalWorkspace(tmp_path)
    reader = workspace.file_identity_reader()
    assert reader is not None
    acquired = threading.Event()
    return_lease = threading.Event()
    release_callbacks = []
    original_acquire = local_module.acquire_local_source_read_gate

    def pause_before_returning_lease(root):
        release_callbacks.append(original_acquire(root))
        acquired.set()
        assert return_lease.wait(10)
        return release_callbacks[-1]

    monkeypatch.setattr(
        local_module, "acquire_local_source_read_gate", pause_before_returning_lease
    )

    async def unblock_during_shutdown() -> None:
        try:
            await asyncio.Future()
        finally:
            return_lease.set()

    async def start_read() -> None:
        unblock = asyncio.create_task(unblock_during_shutdown())
        read = asyncio.create_task(reader.read_file_identities(("file.txt",), max_bytes=100))
        assert await asyncio.to_thread(acquired.wait, 5)
        assert not read.done() and not unblock.done()
        # asyncio.run now cancels all pending tasks before draining its executor.

    asyncio.run(start_read())

    # Use a new loop as a synchronous caller reusing its workspace would. Pruning
    # takes the same exclusive source gate used by branch capture and publication.
    def prune() -> None:
        asyncio.run(workspace.prune_empty_directories("empty", max_directories=1))

    with ThreadPoolExecutor(max_workers=1) as executor:
        pruning = executor.submit(prune)
        try:
            pruning.result(timeout=2)
        except TimeoutError:
            pytest.fail("Loop shutdown stranded a reader in the source gate; pruning cannot enter")
        finally:
            # Test cleanup: release even on failure so the exclusive waiter can exit.
            for release in release_callbacks:
                release()
    assert not (tmp_path / "empty").exists()
