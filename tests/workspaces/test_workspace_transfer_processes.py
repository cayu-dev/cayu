"""Bulk workspace work runs in helper processes, outside the session's event loop."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from cayu.environments import SyncBinding, SyncBindingSourceConflictError
from cayu.workspaces import LocalWorkspace, WorkspaceFileIdentity
from cayu.workspaces import _transfer_processes as transfer
from cayu.workspaces import local as local_module


@pytest.fixture(autouse=True)
def _helpers_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(transfer.TRANSFER_PROCESSES_ENV, raising=False)


def test_work_runs_in_a_reused_helper_process() -> None:
    async def run() -> tuple[int, int]:
        return (
            await transfer.run_workspace_transfer(os.getpid),
            await transfer.run_workspace_transfer(os.getpid),
        )

    first, second = asyncio.run(run())

    assert first != os.getpid()
    assert first == second


def test_zero_helpers_runs_the_work_in_this_process(monkeypatch) -> None:
    monkeypatch.setenv(transfer.TRANSFER_PROCESSES_ENV, "0")

    assert asyncio.run(transfer.run_workspace_transfer(os.getpid)) == os.getpid()


@pytest.mark.parametrize("value", ["-1", "65", "two"])
def test_the_helper_count_is_checked(monkeypatch, value: str) -> None:
    monkeypatch.setenv(transfer.TRANSFER_PROCESSES_ENV, value)

    with pytest.raises(ValueError, match="CAYU_WORKSPACE_TRANSFER_PROCESSES must be"):
        asyncio.run(transfer.run_workspace_transfer(os.getpid))


def test_errors_and_printed_output_come_back_without_breaking_the_helper() -> None:
    async def run() -> None:
        with pytest.raises(ValueError, match="invalid literal"):
            await transfer.run_workspace_transfer(int, "not a number")
        # What the work prints goes to stderr, not into the answers.
        assert await transfer.run_workspace_transfer(print, "noise") is None
        assert await transfer.run_workspace_transfer(int, "7") == 7

    asyncio.run(run())


def test_a_helper_that_dies_fails_its_call_and_the_next_call_gets_a_new_one() -> None:
    async def run() -> tuple[int, int]:
        before = await transfer.run_workspace_transfer(os.getpid)
        with pytest.raises(RuntimeError, match="exited before it finished"):
            await transfer.run_workspace_transfer(os._exit, 3)
        return before, await transfer.run_workspace_transfer(os.getpid)

    before, after = asyncio.run(run())

    assert after not in {before, os.getpid()}


@pytest.mark.parametrize("helpers", ["2", "0"])
def test_an_open_file_reaches_the_work_as_a_descriptor(
    tmp_path: Path, monkeypatch, helpers: str
) -> None:
    monkeypatch.setenv(transfer.TRANSFER_PROCESSES_ENV, helpers)
    with (tmp_path / "spool").open("w+b") as spool:
        written = asyncio.run(
            transfer.run_workspace_transfer(os.write, transfer.OpenFile(spool.fileno()), b"hello")
        )
        assert written == 5
        # The caller's descriptor stays open and shares the file's position.
        spool.seek(0)
        assert spool.read() == b"hello"


def test_open_files_never_reach_a_later_request(tmp_path: Path, monkeypatch) -> None:
    async def run() -> None:
        # A helper started before the module below existed can't import it.
        await transfer.run_workspace_transfer(os.getpid)
        (tmp_path / "late_module.py").write_text(
            "def write(descriptor):\n    return 0\n", encoding="utf-8"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        import late_module  # type: ignore[import-not-found]

        with (tmp_path / "spool").open("w+b") as spool, pytest.raises(ModuleNotFoundError):
            await transfer.run_workspace_transfer(
                late_module.write, transfer.OpenFile(spool.fileno())
            )
        # The descriptor sent with it was taken off the socket, not left for this call.
        assert await transfer.run_workspace_transfer(int, "7") == 7

    asyncio.run(run())


def test_helpers_never_run_the_programs_main_module(tmp_path: Path) -> None:
    # A script without a __main__ guard would run again in a helper that imported it.
    script = tmp_path / "unguarded.py"
    script.write_text(
        textwrap.dedent(
            """
            import asyncio, os
            from pathlib import Path

            from cayu.workspaces._transfer_processes import run_workspace_transfer

            with Path("runs.txt").open("a") as runs:
                runs.write("run\\n")
            print(asyncio.run(run_workspace_transfer(os.getpid)) != os.getpid())
            """
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True"
    assert (tmp_path / "runs.txt").read_text() == "run\n"


def _tree(root: Path) -> tuple[str, ...]:
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "app.py").write_bytes(b"print('hello')\n")
    (root / "run.sh").write_bytes(b"#!/bin/sh\n")
    (root / "run.sh").chmod(0o755)
    (root / "empty.txt").write_bytes(b"")
    return ("src/app.py", "run.sh", "empty.txt")


def test_local_file_identities_match_what_read_bytes_reports(tmp_path: Path) -> None:
    paths = _tree(tmp_path / "tree")
    workspace = LocalWorkspace(tmp_path / "tree")
    reader = workspace.file_identity_reader()
    assert reader is not None

    async def run() -> None:
        identities = await reader.read_file_identities(paths, max_bytes=1024)
        assert [identity.path for identity in identities] == list(paths)
        for identity in identities:
            read = await workspace.read_bytes(identity.path, max_bytes=1024)
            assert identity == WorkspaceFileIdentity(
                path=identity.path,
                total_bytes=read.total_bytes,
                truncated=read.truncated,
                revision=read.revision,
                sha256=read.sha256,
                git_mode=read.git_mode,
            )
        assert identities[1].git_mode == "100755"

        # A read cut short reports it, as read_bytes does.
        (short,) = await reader.read_file_identities(("src/app.py",), max_bytes=3)
        assert short.truncated
        assert short.revision is None
        assert short.sha256 is None

        # Paths are reported as asked for and read as read_bytes reads them.
        (dotted,) = await reader.read_file_identities(("./src/app.py",), max_bytes=1024)
        assert dotted.path == "./src/app.py"
        assert dotted.revision == identities[0].revision

        with pytest.raises(FileNotFoundError):
            await reader.read_file_identities(("missing.txt",), max_bytes=1024)
        with pytest.raises(ValueError, match="escapes"):
            await reader.read_file_identities(("../outside.txt",), max_bytes=1024)

    asyncio.run(run())


def test_excluded_paths_are_refused_before_any_read(tmp_path: Path) -> None:
    _tree(tmp_path / "tree")
    workspace = LocalWorkspace(tmp_path / "tree", excluded_directory_names=("src",))
    reader = workspace.file_identity_reader()
    assert reader is not None

    with pytest.raises(ValueError, match="excluded directory"):
        asyncio.run(reader.read_file_identities(("src/app.py",), max_bytes=1024))


def test_batched_reads_are_only_offered_where_read_bytes_is_local_workspaces_own(
    tmp_path: Path, monkeypatch
) -> None:
    _tree(tmp_path / "tree")

    class Subclass(LocalWorkspace):
        pass

    assert Subclass(tmp_path / "tree").file_identity_reader() is None
    patched = LocalWorkspace(tmp_path / "tree")
    patched.read_bytes = patched.read_bytes  # type: ignore[method-assign]
    assert patched.file_identity_reader() is None

    original = LocalWorkspace.read_bytes

    async def read_bytes(self, path, **kwargs):  # type: ignore[no-untyped-def]
        return await original(self, path, **kwargs)

    monkeypatch.setattr(LocalWorkspace, "read_bytes", read_bytes)
    assert LocalWorkspace(tmp_path / "tree").file_identity_reader() is None


def test_a_sync_binding_reads_and_packs_a_local_source_in_helper_processes(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _tree(tmp_path / "source")
    target_root = tmp_path / "target"
    target_root.mkdir()
    calls: list[str] = []
    real = transfer.run_in_helper

    async def recording(function, /, *args):  # type: ignore[no-untyped-def]
        calls.append(function.__name__)
        return await real(function, *args)

    monkeypatch.setattr(local_module, "run_in_helper", recording)
    binding = SyncBinding(
        target_workspace=LocalWorkspace(target_root, workspace_id="target"),
        sync_back="on_success",
        source_conflict_policy="require_revision",
        preserve_git_modes=True,
        max_file_bytes=1024 * 1024,
        max_total_bytes=8 * 1024 * 1024,
    )

    async def run() -> None:
        bound = await binding.bind(
            LocalWorkspace(tmp_path / "source", workspace_id="source"),
            None,
            session_id="sess-transfer-processes",
        )
        await binding.finalize(bound, outcome="completed")

    asyncio.run(run())

    for path in paths:
        assert (target_root / path).read_bytes() == (tmp_path / "source" / path).read_bytes()
    assert os.stat(target_root / "run.sh").st_mode & 0o111
    assert "_pack_local_files" in calls
    assert "_read_file_identities" in calls


def test_batched_revision_checks_still_catch_a_source_that_changed(tmp_path: Path) -> None:
    _tree(tmp_path / "source")
    target_root = tmp_path / "target"
    target_root.mkdir()
    binding = SyncBinding(
        target_workspace=LocalWorkspace(target_root, workspace_id="target"),
        sync_back="on_success",
        source_conflict_policy="require_revision",
        max_file_bytes=1024 * 1024,
        max_total_bytes=8 * 1024 * 1024,
    )

    async def run() -> None:
        bound = await binding.bind(
            LocalWorkspace(tmp_path / "source", workspace_id="source"),
            None,
            session_id="sess-transfer-conflict",
        )
        (tmp_path / "source" / "src" / "app.py").write_bytes(b"changed at the source\n")
        (target_root / "src" / "app.py").write_bytes(b"changed in the sandbox\n")
        with pytest.raises(SyncBindingSourceConflictError):
            await binding.finalize(bound, outcome="completed")

    asyncio.run(run())

    assert (tmp_path / "source" / "src" / "app.py").read_bytes() == b"changed at the source\n"


def test_a_batched_read_waiting_on_a_publication_that_fences_fails_instead_of_hanging(
    tmp_path: Path,
) -> None:
    from cayu.workspaces._mutations import fence_local_workspace_source, workspace_source_lock
    from cayu.workspaces.branches import WorkspaceBranchFencedError

    paths = _tree(tmp_path / "tree")
    root = (tmp_path / "tree").resolve()
    reader = LocalWorkspace(root).file_identity_reader()
    assert reader is not None
    publishing = threading.Event()
    fence = threading.Event()

    def publish_and_fence() -> None:
        # A branch publication whose rollback failed: it fences the source and keeps its lock.
        with workspace_source_lock(root, exclusive=True):
            publishing.set()
            fence.wait(10)
            fence_local_workspace_source(root)

    async def run() -> None:
        publisher = threading.Thread(target=publish_and_fence)
        publisher.start()
        try:
            assert await asyncio.to_thread(publishing.wait, 10)
            read = asyncio.ensure_future(reader.read_file_identities(paths, max_bytes=1024))
            await asyncio.sleep(0.2)
            assert not read.done()
            fence.set()
            with pytest.raises(WorkspaceBranchFencedError):
                await asyncio.wait_for(read, 10)
        finally:
            fence.set()
            await asyncio.to_thread(publisher.join, 10)

    asyncio.run(run())


def test_batched_reads_in_a_thread_dont_deadlock_behind_a_waiting_writer(
    tmp_path: Path, monkeypatch
) -> None:
    from cayu.workspaces._mutations import workspace_source_lock

    monkeypatch.setenv(transfer.TRANSFER_PROCESSES_ENV, "0")
    paths = _tree(tmp_path / "tree")
    root = (tmp_path / "tree").resolve()
    reader = LocalWorkspace(root).file_identity_reader()
    assert reader is not None
    real_read = local_module._read_file_locked
    writers: list[threading.Thread] = []

    def read_after_a_writer_queues(*args):  # type: ignore[no-untyped-def]
        if not writers:
            # A writer on the same source, such as a branch publication, starts waiting.
            def write() -> None:
                with workspace_source_lock(root, exclusive=True):
                    pass

            writers.append(threading.Thread(target=write))
            writers[0].start()
            writers[0].join(0.2)
        return real_read(*args)

    monkeypatch.setattr(local_module, "_read_file_locked", read_after_a_writer_queues)

    async def run() -> None:
        identities = await asyncio.wait_for(reader.read_file_identities(paths, max_bytes=1024), 10)
        assert len(identities) == len(paths)

    asyncio.run(run())
    writers[0].join(10)
    assert not writers[0].is_alive()
