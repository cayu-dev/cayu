from __future__ import annotations

import asyncio
import errno
import os
import stat
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from cayu import _filesystem_lock as lock_module
from cayu.artifacts import (
    ArtifactScope,
    ArtifactStoreUnavailableError,
    ArtifactWriteSettlementFailureCode,
    ArtifactWriteSettlementStatus,
    InvalidArtifactIdError,
    LocalArtifactStore,
    artifact_write_settlements,
)
from cayu.artifacts import local as local_module

_ARTIFACT_ID = f"art_{'a' * 32}"


def _put(store: LocalArtifactStore, *, artifact_id: str | None = None):
    return asyncio.run(
        store.put_bytes(
            b"durable-content",
            artifact_id=artifact_id,
            filename="durable.txt",
            content_type="text/plain",
            session_id="sess_durable",
        )
    )


def _descriptor_kind(descriptor: int) -> str:
    mode = os.fstat(descriptor).st_mode
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    return "other"


def _assert_acyclic_exception_graph(error: BaseException) -> list[BaseException]:
    observed: list[BaseException] = []
    active: set[int] = set()

    def visit(current: BaseException) -> None:
        identity = id(current)
        assert identity not in active, "exception cause graph contains a cycle"
        active.add(identity)
        observed.append(current)
        if isinstance(current, BaseExceptionGroup):
            for child in current.exceptions:
                visit(child)
        if current.__cause__ is not None:
            visit(current.__cause__)
        elif current.__context__ is not None and not current.__suppress_context__:
            visit(current.__context__)
        active.remove(identity)

    visit(error)
    identities = [id(item) for item in observed]
    assert len(identities) == len(set(identities)), "exception appears more than once"
    return observed


def test_local_deterministic_retry_binds_the_complete_write_tuple(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    kwargs = {
        "artifact_id": _ARTIFACT_ID,
        "filename": "durable.txt",
        "content_type": "text/plain",
        "scope": ArtifactScope.SESSION,
        "session_id": "sess_durable",
        "agent_name": "assistant",
        "environment_name": "local",
        "metadata": {"alpha": 1, "enabled": False},
    }

    first = asyncio.run(store.put_bytes(b"durable-content", **kwargs))
    replayed = asyncio.run(
        store.put_bytes(
            b"durable-content",
            **{**kwargs, "metadata": {"enabled": False, "alpha": 1}},
        )
    )

    assert replayed == first
    conflicts = (
        (b"changed-content", {}),
        (b"durable-content", {"filename": "changed.txt"}),
        (b"durable-content", {"content_type": "application/octet-stream"}),
        (b"durable-content", {"scope": ArtifactScope.ENVIRONMENT}),
        (b"durable-content", {"session_id": "sess_changed"}),
        (b"durable-content", {"agent_name": "reviewer"}),
        (b"durable-content", {"environment_name": "remote"}),
        (b"durable-content", {"metadata": {"alpha": 1, "enabled": True}}),
    )
    for content, update in conflicts:
        with pytest.raises(ValueError, match="different content or metadata"):
            asyncio.run(store.put_bytes(content, **{**kwargs, **update}))

    assert asyncio.run(store.read_bytes(_ARTIFACT_ID)).metadata == first


def test_local_artifact_put_syncs_files_then_directories(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    observed: list[str] = []
    real_sync = local_module._sync_descriptor

    def record_sync(descriptor: int) -> None:
        observed.append(_descriptor_kind(descriptor))
        real_sync(descriptor)

    monkeypatch.setattr(local_module, "_sync_descriptor", record_sync)

    artifact = _put(store)

    assert observed == [
        "file",
        "file",
        "directory",
        "directory",
    ]
    assert asyncio.run(store.read_bytes(artifact.id)).content == b"durable-content"


def test_artifact_ownership_locks_use_a_bounded_shard_namespace(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    lock_root = tmp_path / "artifact-locks"
    monkeypatch.setattr(
        local_module,
        "_ARTIFACT_LOCK_DIRECTORY_NAME",
        str(lock_root),
    )
    artifact_ids = tuple(f"art_{index:032x}" for index in range(1_024))
    expected_keys = {local_module._artifact_lock_key(artifact_id) for artifact_id in artifact_ids}

    for artifact_id in artifact_ids:
        with local_module._artifact_ownership_lock(root, artifact_id):
            pass

    assert len(expected_keys) <= local_module._ARTIFACT_LOCK_SHARD_COUNT
    assert len(expected_keys) < len(artifact_ids)
    assert len(tuple(lock_root.iterdir())) == len(expected_keys)


def test_generated_deterministic_and_delete_paths_share_one_artifact_lock_key(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    observed_keys: list[str] = []
    real_lock = local_module.cooperative_path_lock

    @contextmanager
    def record_lock(lock_root, relative_path, *, lock_directory_name):
        if lock_directory_name == local_module._ARTIFACT_LOCK_DIRECTORY_NAME:
            observed_keys.append(relative_path)
        with real_lock(
            lock_root,
            relative_path,
            lock_directory_name=lock_directory_name,
        ):
            yield

    monkeypatch.setattr(local_module, "_new_artifact_id", lambda: _ARTIFACT_ID)
    monkeypatch.setattr(local_module, "cooperative_path_lock", record_lock)

    generated = _put(store)
    asyncio.run(store.delete(generated.id))
    deterministic = _put(store, artifact_id=generated.id)

    assert deterministic.id == generated.id
    assert observed_keys == [local_module._artifact_lock_key(_ARTIFACT_ID)] * 3


def test_local_artifact_store_creation_syncs_new_directory_ancestry(monkeypatch, tmp_path):
    root = tmp_path / "parent" / "artifacts"
    expected = {
        local_module._stat_identity(os.stat(path, follow_symlinks=False)) for path in (tmp_path,)
    }
    observed: set[tuple[int, int]] = set()
    real_sync = local_module._sync_descriptor

    def record_sync(descriptor: int) -> None:
        observed.add(local_module._stat_identity(os.fstat(descriptor)))
        real_sync(descriptor)

    monkeypatch.setattr(local_module, "_sync_descriptor", record_sync)

    LocalArtifactStore(root)

    expected.update(
        local_module._stat_identity(os.stat(path, follow_symlinks=False))
        for path in (root, root.parent)
    )
    assert expected <= observed


@pytest.mark.parametrize("failed_sync", range(1, 10))
def test_local_artifact_store_creation_retry_resynchronizes_ambiguous_ancestry(
    monkeypatch,
    tmp_path,
    failed_sync,
):
    root = tmp_path / "parent" / "artifacts"
    real_sync = local_module._sync_descriptor
    calls = 0

    def fail_selected_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == failed_sync:
            raise OSError(f"ancestry sync failure {failed_sync}")
        real_sync(descriptor)

    monkeypatch.setattr(local_module, "_sync_descriptor", fail_selected_sync)
    with pytest.raises(ArtifactStoreUnavailableError, match="root could not be made durable"):
        LocalArtifactStore(root)

    observed: set[tuple[int, int]] = set()

    def record_retry_sync(descriptor: int) -> None:
        observed.add(local_module._stat_identity(os.fstat(descriptor)))
        real_sync(descriptor)

    monkeypatch.setattr(local_module, "_sync_descriptor", record_retry_sync)
    store = LocalArtifactStore(root)
    assert _put(store).id.startswith("art_")

    assert local_module._stat_identity(os.stat(root, follow_symlinks=False)) in observed
    assert not local_module._root_pending_marker(root).exists()


def test_preprovisioned_root_does_not_require_parent_synchronization(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_sync_path = local_module._sync_directory_path

    def reject_parent(path, **kwargs) -> None:
        if path == root.parent:
            raise PermissionError("parent is execute-only")
        real_sync_path(path, **kwargs)

    monkeypatch.setattr(local_module, "_sync_directory_path", reject_parent)

    artifact = _put(store)

    assert asyncio.run(store.read_bytes(artifact.id)).content == b"durable-content"


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory permissions are required.")
def test_preprovisioned_root_beneath_execute_only_ancestor_constructs(tmp_path):
    parent = tmp_path / "execute-only"
    root = parent / "artifacts"
    root.mkdir(parents=True)
    parent.chmod(0o311)
    try:
        store = LocalArtifactStore(root)
        artifact = _put(store)
        assert asyncio.run(store.read_bytes(artifact.id)).content == b"durable-content"
    finally:
        parent.chmod(0o700)


@pytest.mark.parametrize(
    "artifact_id",
    ("not_artifact", f"art_{'a' * 31}\ud800"),
)
def test_delete_validates_artifact_id_before_lock_or_filesystem_access(
    monkeypatch,
    tmp_path,
    artifact_id,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    root.rename(tmp_path / "unavailable-artifacts")

    @contextmanager
    def reject_lock(*_args, **_kwargs):
        pytest.fail("invalid artifact ids must not acquire the filesystem lock")
        yield

    monkeypatch.setattr(local_module, "cooperative_path_lock", reject_lock)

    with pytest.raises(InvalidArtifactIdError):
        asyncio.run(store.delete(artifact_id))


def test_concurrent_local_artifact_root_initialization_converges(monkeypatch, tmp_path):
    root = tmp_path / "parent" / "artifacts"
    entered = threading.Event()
    release = threading.Event()
    real_create_marker = local_module._create_pending_root_marker

    def blocked_create_marker(*args, **kwargs) -> None:
        real_create_marker(*args, **kwargs)
        entered.set()
        assert release.wait(timeout=10)

    monkeypatch.setattr(
        local_module,
        "_create_pending_root_marker",
        blocked_create_marker,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(LocalArtifactStore, root)
        assert entered.wait(timeout=10)
        second = pool.submit(LocalArtifactStore, root)
        assert not second.done()
        release.set()
        first_store = first.result(timeout=10)
        second_store = second.result(timeout=10)

    assert first_store.root == second_store.root == root
    assert not local_module._root_pending_marker(root).exists()


def test_case_alias_waits_for_durable_root_initialization(monkeypatch, tmp_path):
    root = tmp_path / "CaseSensitiveSpelling"
    alias = tmp_path / "casesensitivespelling"
    entered = threading.Event()
    release = threading.Event()
    real_sync = local_module._sync_open_directory
    blocked = False

    def block_first_root_sync(directory_fd, path, expected_identity) -> None:
        nonlocal blocked
        if not blocked and path == root:
            blocked = True
            entered.set()
            assert release.wait(timeout=10)
        real_sync(directory_fd, path, expected_identity)

    monkeypatch.setattr(local_module, "_sync_open_directory", block_first_root_sync)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(LocalArtifactStore, root)
        assert entered.wait(timeout=10)
        if not alias.exists() or os.path.samefile(root, alias) is False:
            release.set()
            first.result(timeout=10)
            pytest.skip("filesystem is case-sensitive")
        assert str(root.resolve()) != str(alias.resolve())
        second = pool.submit(LocalArtifactStore, alias)
        assert not second.done()
        release.set()
        first_store = first.result(timeout=10)
        second_store = second.result(timeout=10)

    assert first_store._root_identity == second_store._root_identity
    assert (
        local_module._root_pending_marker(root).name
        == local_module._root_pending_marker(alias).name
    )
    assert not local_module._root_pending_marker(root).exists()


def test_racing_external_root_creation_is_synchronized_before_acceptance(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    real_create_ancestry = local_module._create_durable_directory_ancestry
    real_sync = local_module._sync_descriptor
    observed: set[tuple[int, int]] = set()

    def create_racing_root(path) -> None:
        real_create_ancestry(path)
        root.mkdir()

    def record_sync(descriptor: int) -> None:
        observed.add(local_module._stat_identity(os.fstat(descriptor)))
        real_sync(descriptor)

    monkeypatch.setattr(
        local_module,
        "_create_durable_directory_ancestry",
        create_racing_root,
    )
    monkeypatch.setattr(local_module, "_sync_descriptor", record_sync)

    store = LocalArtifactStore(root)

    assert store.root == root
    assert local_module._stat_identity(os.stat(root, follow_symlinks=False)) in observed
    assert local_module._stat_identity(os.stat(root.parent, follow_symlinks=False)) in observed


def test_local_artifact_store_rejects_root_replaced_after_durable_initialization(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    original_root = tmp_path / "initialized-artifacts"
    real_initialize = local_module._initialize_durable_store_root

    def initialize_then_replace(path):
        initialized_identity = real_initialize(path)
        path.rename(original_root)
        path.mkdir()
        return initialized_identity

    monkeypatch.setattr(
        local_module,
        "_initialize_durable_store_root",
        initialize_then_replace,
    )

    with pytest.raises(ArtifactStoreUnavailableError, match="changed during durable"):
        LocalArtifactStore(root)

    assert original_root.is_dir()
    assert root.is_dir()


def test_malformed_local_artifact_root_pending_marker_fails_closed(tmp_path):
    root = tmp_path / "artifacts"
    marker = local_module._root_pending_marker(root)
    marker.write_bytes(b"not a Cayu root marker")

    with pytest.raises(ArtifactStoreUnavailableError, match="pending marker is invalid"):
        LocalArtifactStore(root)

    assert marker.read_bytes() == b"not a Cayu root marker"
    assert not root.exists()


def test_durable_descriptor_sync_uses_fsync_off_darwin(monkeypatch, tmp_path):
    path = tmp_path / "content"
    path.write_bytes(b"durable")
    descriptor = os.open(path, os.O_RDONLY)
    observed: list[int] = []
    try:
        monkeypatch.setattr(local_module.sys, "platform", "linux")
        monkeypatch.setattr(local_module.os, "fsync", observed.append)

        local_module._sync_descriptor(descriptor)
    finally:
        os.close(descriptor)

    assert observed == [descriptor]


def test_durable_descriptor_sync_uses_fullfsync_on_darwin(monkeypatch, tmp_path):
    path = tmp_path / "content"
    path.write_bytes(b"durable")
    descriptor = os.open(path, os.O_RDONLY)
    observed: list[tuple[int, int]] = []
    try:
        monkeypatch.setattr(local_module.sys, "platform", "darwin")
        monkeypatch.setattr(local_module, "_DARWIN_FULL_SYNC_COMMAND", 51)
        monkeypatch.setattr(
            local_module,
            "_FCNTL_MODULE",
            SimpleNamespace(fcntl=lambda fd, command: observed.append((fd, command))),
        )
        monkeypatch.setattr(
            local_module.os,
            "fsync",
            lambda _fd: pytest.fail("macOS durability must not use plain fsync"),
        )

        local_module._sync_descriptor(descriptor)
    finally:
        os.close(descriptor)

    assert observed == [(descriptor, 51)]


def test_durable_publication_fails_closed_without_fullfsync_on_darwin(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    monkeypatch.setattr(local_module.sys, "platform", "darwin")
    monkeypatch.setattr(local_module, "_DARWIN_FULL_SYNC_COMMAND", None)

    with pytest.raises(ArtifactStoreUnavailableError, match="directory synchronization"):
        _put(store)


@pytest.mark.parametrize("failed_sync", range(1, 5))
def test_local_artifact_put_fails_closed_at_each_sync_boundary(
    monkeypatch,
    tmp_path,
    failed_sync,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_sync = local_module._sync_descriptor
    calls = 0

    def fail_selected_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == failed_sync:
            raise OSError(f"sync failure {failed_sync}")
        real_sync(descriptor)

    monkeypatch.setattr(local_module, "_sync_descriptor", fail_selected_sync)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store, artifact_id=_ARTIFACT_ID)

    assert isinstance(exc_info.value.__cause__, OSError)
    target = root / _ARTIFACT_ID
    if failed_sync < 4:
        assert not target.exists()
        assert list(root.iterdir()) == []
    else:
        assert (target / "content").read_bytes() == b"durable-content"
        assert asyncio.run(store.read_bytes(_ARTIFACT_ID)).metadata.id == _ARTIFACT_ID


@pytest.mark.parametrize("failed_file", ("content", "metadata.json"))
def test_local_artifact_put_cleans_staging_after_file_write_failure(
    monkeypatch,
    tmp_path,
    failed_file,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_write = local_module._write_artifact_file

    def fail_selected_file(*args, **kwargs) -> None:
        if args[3] == failed_file:
            raise OSError(f"{failed_file} write failed")
        real_write(*args, **kwargs)

    monkeypatch.setattr(local_module, "_write_artifact_file", fail_selected_file)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store, artifact_id=_ARTIFACT_ID)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert list(root.iterdir()) == []


def test_local_staging_identity_failure_requires_reconciliation(monkeypatch, tmp_path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_stat = local_module._stat_directory_entry
    failed = False

    def fail_initial_staging_stat(path, *, parent_fd=None):
        nonlocal failed
        if not failed and ".staging-" in path.name:
            failed = True
            raise OSError("staging identity unavailable")
        return real_stat(path, parent_fd=parent_fd)

    monkeypatch.setattr(local_module, "_stat_directory_entry", fail_initial_staging_stat)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store, artifact_id=_ARTIFACT_ID)

    assert failed
    settlement = artifact_write_settlements(exc_info.value)
    assert len(settlement) == 1
    assert settlement[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert any(".staging-" in entry.name for entry in root.iterdir())


def test_local_unproved_staging_removal_requires_reconciliation(monkeypatch, tmp_path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_write = local_module._write_artifact_file

    def fail_metadata_write(*args, **kwargs) -> None:
        if args[3] == "metadata.json":
            raise OSError("metadata write failed")
        real_write(*args, **kwargs)

    def leave_staging_in_place(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(local_module, "_write_artifact_file", fail_metadata_write)
    monkeypatch.setattr(local_module.shutil, "rmtree", leave_staging_in_place)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store, artifact_id=_ARTIFACT_ID)

    settlement = artifact_write_settlements(exc_info.value)
    assert len(settlement) == 1
    assert settlement[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert settlement[0].failure_codes == (
        ArtifactWriteSettlementFailureCode.MUTATION_FAILED,
        ArtifactWriteSettlementFailureCode.CLEANUP_FAILED,
    )
    assert any(".staging-" in entry.name for entry in root.iterdir())


def test_local_artifact_put_cleans_staging_after_rename_failure(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)

    def fail_rename(*_args, **_kwargs) -> None:
        raise OSError("rename failed")

    monkeypatch.setattr(local_module, "_rename_directory_no_replace", fail_rename)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store, artifact_id=_ARTIFACT_ID)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert list(root.iterdir()) == []


def test_deterministic_retry_resynchronizes_after_ambiguous_publication(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_sync = local_module._sync_descriptor
    calls = 0

    def fail_commit_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("commit acknowledgement lost")
        real_sync(descriptor)

    monkeypatch.setattr(local_module, "_sync_descriptor", fail_commit_sync)
    with pytest.raises(ArtifactStoreUnavailableError):
        _put(store, artifact_id=_ARTIFACT_ID)

    observed: list[str] = []

    def record_retry_sync(descriptor: int) -> None:
        observed.append(_descriptor_kind(descriptor))
        real_sync(descriptor)

    monkeypatch.setattr(local_module, "_sync_descriptor", record_retry_sync)

    artifact = _put(store, artifact_id=_ARTIFACT_ID)

    assert artifact.id == _ARTIFACT_ID
    assert observed[-4:] == ["file", "file", "directory", "directory"]
    assert asyncio.run(store.read_bytes(_ARTIFACT_ID)).content == b"durable-content"


@pytest.mark.parametrize("failed_retry_sync", range(1, 5))
def test_deterministic_retry_fails_closed_at_each_resynchronization_boundary(
    monkeypatch,
    tmp_path,
    failed_retry_sync,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_sync = local_module._sync_descriptor
    initial_syncs = 0

    def fail_initial_commit_sync(descriptor: int) -> None:
        nonlocal initial_syncs
        initial_syncs += 1
        if initial_syncs == 4:
            raise OSError("initial commit acknowledgement lost")
        real_sync(descriptor)

    monkeypatch.setattr(local_module, "_sync_descriptor", fail_initial_commit_sync)
    with pytest.raises(ArtifactStoreUnavailableError):
        _put(store, artifact_id=_ARTIFACT_ID)

    retry_resync_started = False
    retry_syncs = 0
    real_resync = local_module._sync_existing_artifact

    def mark_retry_resync(*args, **kwargs) -> None:
        nonlocal retry_resync_started
        retry_resync_started = True
        real_resync(*args, **kwargs)

    def fail_selected_retry_sync(descriptor: int) -> None:
        nonlocal retry_syncs
        if retry_resync_started:
            retry_syncs += 1
            if retry_syncs == failed_retry_sync:
                raise OSError(f"retry sync failure {failed_retry_sync}")
        real_sync(descriptor)

    monkeypatch.setattr(local_module, "_sync_existing_artifact", mark_retry_resync)
    monkeypatch.setattr(local_module, "_sync_descriptor", fail_selected_retry_sync)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store, artifact_id=_ARTIFACT_ID)
    assert isinstance(exc_info.value.__cause__, OSError)
    assert retry_syncs == failed_retry_sync
    assert asyncio.run(store.read_bytes(_ARTIFACT_ID)).content == b"durable-content"

    monkeypatch.setattr(local_module, "_sync_descriptor", real_sync)
    retry = _put(store, artifact_id=_ARTIFACT_ID)
    assert retry.id == _ARTIFACT_ID


def test_local_artifact_cancellation_waits_for_dispatched_write(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    dispatched = threading.Event()
    release = threading.Event()
    real_rename = local_module._rename_directory_no_replace

    def blocked_rename(*args, **kwargs) -> None:
        dispatched.set()
        assert release.wait(timeout=10)
        real_rename(*args, **kwargs)

    monkeypatch.setattr(local_module, "_rename_directory_no_replace", blocked_rename)

    async def exercise():
        first = asyncio.create_task(
            store.put_bytes(
                b"durable-content",
                artifact_id=_ARTIFACT_ID,
                filename="durable.txt",
                content_type="text/plain",
                session_id="sess_durable",
            )
        )
        assert await asyncio.to_thread(dispatched.wait, 10)
        first.cancel("stop local artifact write")
        assert first.cancelling() == 1
        await asyncio.sleep(0)
        # The shared settlement helper owns and normalizes the delivered request
        # while the physical worker remains in flight, then re-delivers it below.
        assert first.cancelling() == 0
        assert not first.done()

        contender = asyncio.create_task(
            store.put_bytes(
                b"durable-content",
                artifact_id=_ARTIFACT_ID,
                filename="durable.txt",
                content_type="text/plain",
                session_id="sess_durable",
            )
        )
        await asyncio.sleep(0.05)
        assert not contender.done()
        release.set()

        with pytest.raises(
            asyncio.CancelledError,
            match="stop local artifact write",
        ) as raised:
            await first
        assert first.cancelled()
        settlement = artifact_write_settlements(raised.value)
        assert len(settlement) == 1
        assert settlement[0].artifact_id == _ARTIFACT_ID
        assert settlement[0].status is ArtifactWriteSettlementStatus.COMMITTED
        return await contender

    artifact = asyncio.run(exercise())

    assert artifact.id == _ARTIFACT_ID
    assert asyncio.run(store.read_bytes(_ARTIFACT_ID)).content == b"durable-content"


def test_local_child_task_cancellation_keeps_dispatched_thread_owned(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    dispatched = threading.Event()
    release = threading.Event()
    child_tasks: list[asyncio.Task] = []
    real_rename = local_module._rename_directory_no_replace
    real_operation = local_module._run_local_artifact_write

    def blocked_rename(*args, **kwargs) -> None:
        dispatched.set()
        assert release.wait(timeout=10)
        real_rename(*args, **kwargs)

    async def capture_child(*args, **kwargs):
        child = asyncio.current_task()
        assert child is not None
        child_tasks.append(child)
        return await real_operation(*args, **kwargs)

    monkeypatch.setattr(local_module, "_rename_directory_no_replace", blocked_rename)
    monkeypatch.setattr(local_module, "_run_local_artifact_write", capture_child)

    async def exercise():
        put_task = asyncio.create_task(
            store.put_bytes(
                b"durable-content",
                artifact_id=_ARTIFACT_ID,
                filename="durable.txt",
                content_type="text/plain",
                session_id="sess_durable",
            )
        )
        assert await asyncio.to_thread(dispatched.wait, 10)
        assert len(child_tasks) == 1
        child_tasks[0].cancel("supervisor stopped child")
        await asyncio.sleep(0)
        assert not put_task.done()

        contender = asyncio.create_task(
            store.put_bytes(
                b"durable-content",
                artifact_id=_ARTIFACT_ID,
                filename="durable.txt",
                content_type="text/plain",
                session_id="sess_durable",
            )
        )
        await asyncio.sleep(0.05)
        assert not contender.done()
        release.set()

        with pytest.raises(
            RuntimeError,
            match="Local artifact publication was cancelled without caller cancellation",
        ) as raised:
            await put_task
        assert isinstance(raised.value.__cause__, asyncio.CancelledError)
        evidence = artifact_write_settlements(raised.value)
        assert len(evidence) == 1
        assert evidence[0].status is ArtifactWriteSettlementStatus.COMMITTED
        assert evidence[0].failure_codes == (ArtifactWriteSettlementFailureCode.CHILD_CANCELLED,)
        return put_task, await contender

    put_task, contender = asyncio.run(exercise())

    assert not put_task.cancelled()
    assert child_tasks[0].done()
    assert child_tasks[0].cancelling() == 0
    assert not child_tasks[0].cancelled()
    assert contender.id == _ARTIFACT_ID
    assert asyncio.run(store.read_bytes(_ARTIFACT_ID)).content == b"durable-content"


def test_local_child_task_cancellation_preserves_later_worker_failure(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    dispatched = threading.Event()
    release = threading.Event()
    child_tasks: list[asyncio.Task] = []
    publication_error = OSError("publication failed after child cancellation")
    real_operation = local_module._run_local_artifact_write

    def failing_rename(*_args, **_kwargs) -> None:
        dispatched.set()
        assert release.wait(timeout=10)
        raise publication_error

    async def capture_child(*args, **kwargs):
        child = asyncio.current_task()
        assert child is not None
        child_tasks.append(child)
        return await real_operation(*args, **kwargs)

    monkeypatch.setattr(local_module, "_rename_directory_no_replace", failing_rename)
    monkeypatch.setattr(local_module, "_run_local_artifact_write", capture_child)

    async def exercise():
        put_task = asyncio.create_task(
            store.put_bytes(
                b"durable-content",
                artifact_id=_ARTIFACT_ID,
                filename="durable.txt",
                content_type="text/plain",
                session_id="sess_durable",
            )
        )
        assert await asyncio.to_thread(dispatched.wait, 10)
        child_tasks[0].cancel("supervisor stopped child")
        await asyncio.sleep(0)
        assert not put_task.done()
        release.set()
        with pytest.raises(
            ArtifactStoreUnavailableError,
            match="could not write artifact content",
        ) as raised:
            await put_task
        return put_task, raised.value

    put_task, failure = asyncio.run(exercise())

    assert not put_task.cancelled()
    assert child_tasks[0].done()
    assert child_tasks[0].cancelling() == 0
    assert not child_tasks[0].cancelled()
    observed = _assert_acyclic_exception_graph(failure)
    assert sum(error is publication_error for error in observed) == 1
    assert sum(isinstance(error, asyncio.CancelledError) for error in observed) == 1
    evidence = artifact_write_settlements(failure)
    assert len(evidence) == 1
    assert evidence[0].status is ArtifactWriteSettlementStatus.ABSENT
    assert evidence[0].failure_codes == (
        ArtifactWriteSettlementFailureCode.MUTATION_FAILED,
        ArtifactWriteSettlementFailureCode.CHILD_CANCELLED,
    )


def test_local_artifact_cancellation_preserves_worker_failure(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    dispatched = threading.Event()
    release = threading.Event()

    def fail_after_release(*_args, **_kwargs) -> None:
        dispatched.set()
        assert release.wait(timeout=10)
        raise OSError("rename failed during cancellation")

    monkeypatch.setattr(local_module, "_rename_directory_no_replace", fail_after_release)

    async def exercise() -> asyncio.Task:
        task = asyncio.create_task(
            store.put_bytes(
                b"durable-content",
                artifact_id=_ARTIFACT_ID,
                filename="durable.txt",
                content_type="text/plain",
                session_id="sess_durable",
            )
        )
        assert await asyncio.to_thread(dispatched.wait, 10)
        task.cancel("stop failed local artifact write")
        release.set()
        with pytest.raises(asyncio.CancelledError, match="stop failed local artifact write") as exc:
            await task
        assert isinstance(exc.value.__cause__, OSError)
        assert "also failed" in " ".join(exc.value.__notes__)
        settlement = artifact_write_settlements(exc.value)
        assert len(settlement) == 1
        assert settlement[0].status is ArtifactWriteSettlementStatus.ABSENT
        return task

    task = asyncio.run(exercise())

    assert task.cancelled()
    assert list(root.iterdir()) == []


def test_local_cancellation_during_cleanup_waits_for_positive_absence(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    real_write = local_module._write_artifact_file
    real_cleanup = local_module._remove_artifact_directory_if_unchanged

    def fail_metadata(*args, **kwargs) -> None:
        if args[3] == "metadata.json":
            raise OSError("metadata write failed")
        real_write(*args, **kwargs)

    def blocked_cleanup(*args, **kwargs) -> None:
        cleanup_started.set()
        assert release_cleanup.wait(timeout=10)
        real_cleanup(*args, **kwargs)

    monkeypatch.setattr(local_module, "_write_artifact_file", fail_metadata)
    monkeypatch.setattr(
        local_module,
        "_remove_artifact_directory_if_unchanged",
        blocked_cleanup,
    )

    async def exercise() -> asyncio.Task:
        task = asyncio.create_task(
            store.put_bytes(
                b"durable-content",
                artifact_id=_ARTIFACT_ID,
                filename="durable.txt",
                content_type="text/plain",
                session_id="sess_durable",
            )
        )
        assert await asyncio.to_thread(cleanup_started.wait, 10)
        task.cancel("stop during local cleanup")
        await asyncio.sleep(0)
        assert not task.done()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError, match="stop during local cleanup") as raised:
            await task
        evidence = artifact_write_settlements(raised.value)
        assert len(evidence) == 1
        assert evidence[0].status is ArtifactWriteSettlementStatus.ABSENT
        return task

    task = asyncio.run(exercise())

    assert task.cancelled()
    assert not any(".staging-" in entry.name for entry in root.iterdir())


def test_generated_artifact_cancellation_leaves_exact_publication_retryable(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    dispatched = threading.Event()
    release = threading.Event()
    real_rename = local_module._rename_directory_no_replace
    real_lock = local_module.cooperative_path_lock
    observed_lock_keys: list[str] = []
    published_identities = []
    real_write_generated = local_module._write_generated_artifact
    retry_attempted_lock = threading.Event()

    def blocked_rename(*args, **kwargs) -> None:
        dispatched.set()
        assert release.wait(timeout=10)
        real_rename(*args, **kwargs)

    def record_publication(*args, **kwargs):
        identity = real_write_generated(*args, **kwargs)
        published_identities.append(identity)
        return identity

    @contextmanager
    def record_lock(lock_root, relative_path, *, lock_directory_name):
        if lock_directory_name == local_module._ARTIFACT_LOCK_DIRECTORY_NAME:
            observed_lock_keys.append(relative_path)
            if len(observed_lock_keys) == 2:
                retry_attempted_lock.set()
        with real_lock(
            lock_root,
            relative_path,
            lock_directory_name=lock_directory_name,
        ):
            yield

    monkeypatch.setattr(local_module, "_rename_directory_no_replace", blocked_rename)
    monkeypatch.setattr(local_module, "_new_artifact_id", lambda: _ARTIFACT_ID)
    monkeypatch.setattr(local_module, "_write_generated_artifact", record_publication)
    monkeypatch.setattr(local_module, "cooperative_path_lock", record_lock)

    async def exercise():
        task = asyncio.create_task(
            store.put_bytes(
                b"generated-content",
                filename="generated.txt",
                content_type="text/plain",
                session_id="sess_generated",
            )
        )
        assert await asyncio.to_thread(dispatched.wait, 10)
        task.cancel("stop generated local artifact write")
        assert task.cancelling() == 1
        retry_task = asyncio.create_task(
            store.put_bytes(
                b"generated-content",
                artifact_id=_ARTIFACT_ID,
                filename="generated.txt",
                content_type="text/plain",
                session_id="sess_generated",
            )
        )
        assert await asyncio.to_thread(retry_attempted_lock.wait, 10)
        assert not retry_task.done()
        release.set()
        with pytest.raises(
            asyncio.CancelledError,
            match="stop generated local artifact write",
        ):
            await task
        return task, await retry_task

    task, retry = asyncio.run(exercise())

    assert task.cancelled()
    assert len(published_identities) == 1
    published_identity = published_identities[0]
    target = root / _ARTIFACT_ID
    assert local_module._stat_identity(target.stat()) == published_identity

    assert retry.id == _ARTIFACT_ID
    assert local_module._stat_identity(target.stat()) == published_identity
    assert asyncio.run(store.read_bytes(_ARTIFACT_ID)).content == b"generated-content"
    assert len(observed_lock_keys) == 2
    assert len(set(observed_lock_keys)) == 1


def test_generated_artifact_does_not_depend_on_descriptor_duplication(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)

    def fail_dup(_descriptor: int) -> int:
        raise OSError(errno.EMFILE, "descriptor limit reached")

    monkeypatch.setattr(local_module.os, "dup", fail_dup)

    artifact = _put(store)

    assert artifact.id.startswith("art_")
    assert asyncio.run(store.read_bytes(artifact.id)).content == b"durable-content"


def test_generated_artifact_postpublication_failure_leaves_exact_retryable_artifact(
    monkeypatch, tmp_path
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_sync = local_module._sync_open_directory

    def fail_post_publication_sync(directory_fd, path, expected_identity) -> None:
        if path == root and any(
            child.name.startswith("art_") and ".staging-" not in child.name
            for child in root.iterdir()
        ):
            raise OSError("publication failed")
        real_sync(directory_fd, path, expected_identity)

    monkeypatch.setattr(local_module, "_new_artifact_id", lambda: _ARTIFACT_ID)
    monkeypatch.setattr(local_module, "_sync_open_directory", fail_post_publication_sync)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store)
    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "publication failed"

    monkeypatch.setattr(local_module, "_sync_open_directory", real_sync)
    identity = local_module._stat_identity((root / _ARTIFACT_ID).stat())
    retry = _put(store, artifact_id=_ARTIFACT_ID)
    assert retry.id == _ARTIFACT_ID
    assert local_module._stat_identity((root / _ARTIFACT_ID).stat()) == identity


def test_generated_artifact_lock_teardown_failure_leaves_artifact(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_lock = local_module.cooperative_path_lock
    lock_calls = 0

    @contextmanager
    def fail_lock_teardown(*args, **kwargs):
        nonlocal lock_calls
        lock_calls += 1
        fail_this_teardown = lock_calls == 1
        with real_lock(*args, **kwargs):
            yield
        if fail_this_teardown:
            raise OSError("ownership lock teardown failed")

    monkeypatch.setattr(local_module, "_new_artifact_id", lambda: _ARTIFACT_ID)
    monkeypatch.setattr(local_module, "cooperative_path_lock", fail_lock_teardown)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "ownership lock teardown failed"
    assert asyncio.run(store.read_bytes(_ARTIFACT_ID)).content == b"durable-content"


@pytest.mark.skipif(os.name == "nt", reason="This regression exercises POSIX flock teardown.")
def test_real_lock_unlock_failure_closes_lock_and_leaves_generated_artifact(
    monkeypatch,
    tmp_path,
):
    import fcntl

    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_flock = fcntl.flock
    real_close = lock_module.os.close
    failed_lock_descriptor: int | None = None
    closed_descriptors: list[int] = []

    def fail_first_unlock(descriptor: int, operation: int) -> None:
        nonlocal failed_lock_descriptor
        if operation == fcntl.LOCK_UN and failed_lock_descriptor is None:
            failed_lock_descriptor = descriptor
            raise OSError("real ownership lock unlock failed")
        real_flock(descriptor, operation)

    def record_close(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(local_module, "_new_artifact_id", lambda: _ARTIFACT_ID)
    monkeypatch.setattr(fcntl, "flock", fail_first_unlock)
    monkeypatch.setattr(lock_module.os, "close", record_close)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "real ownership lock unlock failed"
    assert failed_lock_descriptor is not None
    assert failed_lock_descriptor in closed_descriptors
    identity = local_module._stat_identity((root / _ARTIFACT_ID).stat())

    retry = _put(store, artifact_id=_ARTIFACT_ID)
    assert retry.id == _ARTIFACT_ID
    assert local_module._stat_identity((root / _ARTIFACT_ID).stat()) == identity
    assert asyncio.run(store.read_bytes(_ARTIFACT_ID)).content == b"durable-content"


@pytest.mark.skipif(os.name == "nt", reason="This regression exercises POSIX flock teardown.")
def test_lock_body_unlock_and_close_failures_preserve_ordered_evidence(
    monkeypatch,
    tmp_path,
):
    import fcntl

    real_flock = fcntl.flock
    real_close = lock_module.os.close
    lock_descriptor: int | None = None
    primary_error = ValueError("lock body failed")
    unlock_error = OSError("lock unlock failed")
    close_error = OSError("lock close failed")

    def fail_cleanup(descriptor: int, operation: int) -> None:
        nonlocal lock_descriptor
        if operation == fcntl.LOCK_EX:
            lock_descriptor = descriptor
            real_flock(descriptor, operation)
            return
        real_flock(descriptor, operation)
        if descriptor == lock_descriptor and operation == fcntl.LOCK_UN:
            raise unlock_error

    def fail_close(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == lock_descriptor:
            raise close_error

    monkeypatch.setattr(fcntl, "flock", fail_cleanup)
    monkeypatch.setattr(lock_module.os, "close", fail_close)

    with (
        pytest.raises(ValueError, match="lock body failed") as exc_info,
        lock_module.cooperative_path_lock(
            tmp_path,
            _ARTIFACT_ID,
            lock_directory_name="cayu-artifact-lock-test",
        ),
    ):
        raise primary_error

    observed = _assert_acyclic_exception_graph(exc_info.value)
    assert observed[0] is primary_error
    assert sum(error is primary_error for error in observed) == 1
    assert sum(error is unlock_error for error in observed) == 1
    assert sum(error is close_error for error in observed) == 1


@pytest.mark.skipif(os.name == "nt", reason="This regression exercises POSIX flock teardown.")
def test_lock_unlock_and_close_failures_keep_unlock_authoritative(monkeypatch, tmp_path):
    import fcntl

    real_flock = fcntl.flock
    real_close = lock_module.os.close
    lock_descriptor: int | None = None
    unlock_error = OSError("lock unlock failed")
    close_error = OSError("lock close failed")

    def fail_unlock(descriptor: int, operation: int) -> None:
        nonlocal lock_descriptor
        if operation == fcntl.LOCK_EX:
            lock_descriptor = descriptor
        real_flock(descriptor, operation)
        if descriptor == lock_descriptor and operation == fcntl.LOCK_UN:
            raise unlock_error

    def fail_close(descriptor: int) -> None:
        real_close(descriptor)
        if descriptor == lock_descriptor:
            raise close_error

    monkeypatch.setattr(fcntl, "flock", fail_unlock)
    monkeypatch.setattr(lock_module.os, "close", fail_close)

    with (
        pytest.raises(OSError, match="lock unlock failed") as exc_info,
        lock_module.cooperative_path_lock(
            tmp_path,
            _ARTIFACT_ID,
            lock_directory_name="cayu-artifact-lock-test",
        ),
    ):
        pass

    assert exc_info.value is unlock_error
    observed = _assert_acyclic_exception_graph(exc_info.value)
    assert sum(error is unlock_error for error in observed) == 1
    assert sum(error is close_error for error in observed) == 1


@pytest.mark.skipif(os.name == "nt", reason="This regression exercises POSIX flock teardown.")
def test_lock_acquisition_failure_still_closes_descriptor(monkeypatch, tmp_path):
    import fcntl

    real_close = lock_module.os.close
    acquisition_error = OSError("lock acquisition failed")
    attempted_descriptor: int | None = None
    closed_descriptors: list[int] = []
    operations: list[int] = []

    def fail_acquisition(descriptor: int, operation: int) -> None:
        nonlocal attempted_descriptor
        attempted_descriptor = descriptor
        operations.append(operation)
        raise acquisition_error

    def record_close(descriptor: int) -> None:
        closed_descriptors.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(fcntl, "flock", fail_acquisition)
    monkeypatch.setattr(lock_module.os, "close", record_close)

    with (
        pytest.raises(OSError, match="lock acquisition failed") as exc_info,
        lock_module.cooperative_path_lock(
            tmp_path,
            _ARTIFACT_ID,
            lock_directory_name="cayu-artifact-lock-test",
        ),
    ):
        pytest.fail("lock body must not run after acquisition failure")

    assert exc_info.value is acquisition_error
    assert attempted_descriptor is not None
    assert attempted_descriptor in closed_descriptors
    assert operations == [fcntl.LOCK_EX]


def test_generated_artifact_child_cancellation_after_publication_leaves_artifact(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_sync = local_module._sync_open_directory
    cancellation_injected = False

    def cancel_after_publication(directory_fd, path, expected_identity) -> None:
        nonlocal cancellation_injected
        if (
            not cancellation_injected
            and path == root
            and any(
                child.name.startswith("art_") and ".staging-" not in child.name
                for child in root.iterdir()
            )
        ):
            cancellation_injected = True
            raise asyncio.CancelledError("worker publication cancelled")
        real_sync(directory_fd, path, expected_identity)

    monkeypatch.setattr(local_module, "_sync_open_directory", cancel_after_publication)

    async def exercise() -> asyncio.Task:
        task = asyncio.create_task(
            store.put_bytes(
                b"generated-content",
                filename="generated.txt",
                content_type="text/plain",
                session_id="sess_generated",
            )
        )
        with pytest.raises(
            RuntimeError,
            match="Local artifact publication was cancelled without caller cancellation",
        ) as exc_info:
            await task
        assert isinstance(exc_info.value.__cause__, asyncio.CancelledError)
        assert task.cancelling() == 0
        assert not task.cancelled()
        return task

    asyncio.run(exercise())

    assert cancellation_injected
    assert len(asyncio.run(store.list(session_id="sess_generated")).artifacts) == 1


def test_generated_artifact_child_cancellation_during_lock_teardown_is_operational(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_lock = local_module.cooperative_path_lock
    lock_calls = 0

    @contextmanager
    def cancel_lock_teardown(*args, **kwargs):
        nonlocal lock_calls
        lock_calls += 1
        cancel_this_teardown = lock_calls == 1
        with real_lock(*args, **kwargs):
            yield
        if cancel_this_teardown:
            raise asyncio.CancelledError("worker lock teardown cancelled")

    monkeypatch.setattr(local_module, "cooperative_path_lock", cancel_lock_teardown)

    async def exercise() -> asyncio.Task:
        task = asyncio.create_task(
            store.put_bytes(
                b"generated-content",
                filename="generated.txt",
                content_type="text/plain",
                session_id="sess_generated",
            )
        )
        with pytest.raises(
            RuntimeError,
            match="Local artifact publication was cancelled without caller cancellation",
        ) as exc_info:
            await task
        assert isinstance(exc_info.value.__cause__, asyncio.CancelledError)
        assert task.cancelling() == 0
        assert not task.cancelled()
        return task

    asyncio.run(exercise())

    assert len(asyncio.run(store.list(session_id="sess_generated")).artifacts) == 1


def test_publication_and_lock_teardown_failures_preserve_both_and_artifact(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_sync = local_module._sync_open_directory
    real_lock = local_module.cooperative_path_lock
    original_errors: list[BaseException] = []
    lock_calls = 0

    def fail_post_publication_sync(directory_fd, path, expected_identity) -> None:
        if path == root and any(
            child.name.startswith("art_") and ".staging-" not in child.name
            for child in root.iterdir()
        ):
            error = OSError("publication failed")
            original_errors.append(error)
            raise error
        real_sync(directory_fd, path, expected_identity)

    @contextmanager
    def fail_lock_teardown(*args, **kwargs):
        nonlocal lock_calls
        lock_calls += 1
        fail_this_teardown = lock_calls == 1
        with real_lock(*args, **kwargs):
            yield
        if fail_this_teardown:
            error = OSError("ownership lock teardown failed")
            original_errors.append(error)
            raise error

    monkeypatch.setattr(local_module, "_sync_open_directory", fail_post_publication_sync)
    monkeypatch.setattr(local_module, "cooperative_path_lock", fail_lock_teardown)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store)

    observed = _assert_acyclic_exception_graph(exc_info.value)
    for original in original_errors:
        assert sum(item is original for item in observed) == 1
    assert len(asyncio.run(store.list(session_id="sess_durable")).artifacts) == 1


def test_typed_postpublication_failure_preserves_exact_evidence(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_sync = local_module._sync_open_directory
    original_errors: list[BaseException] = []

    def fail_typed_publication(directory_fd, path, expected_identity) -> None:
        if path == root and any(
            child.name.startswith("art_") and ".staging-" not in child.name
            for child in root.iterdir()
        ):
            error = ArtifactStoreUnavailableError("typed publication failed")
            original_errors.append(error)
            raise error
        real_sync(directory_fd, path, expected_identity)

    monkeypatch.setattr(local_module, "_sync_open_directory", fail_typed_publication)

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store)

    assert str(exc_info.value) == "typed publication failed"
    observed = _assert_acyclic_exception_graph(exc_info.value)
    assert sum(item is original_errors[0] for item in observed) == 1
    assert exc_info.value is original_errors[0]


def test_cancellation_with_postpublication_failure_is_acyclic(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    rename_entered = threading.Event()
    release_rename = threading.Event()
    real_rename = local_module._rename_directory_no_replace
    real_sync = local_module._sync_open_directory
    original_errors: list[BaseException] = []

    def blocked_rename(*args, **kwargs) -> None:
        rename_entered.set()
        assert release_rename.wait(timeout=10)
        real_rename(*args, **kwargs)

    def fail_typed_publication(directory_fd, path, expected_identity) -> None:
        if path == root and any(
            child.name.startswith("art_") and ".staging-" not in child.name
            for child in root.iterdir()
        ):
            error = ArtifactStoreUnavailableError("typed publication failed")
            original_errors.append(error)
            raise error
        real_sync(directory_fd, path, expected_identity)

    monkeypatch.setattr(local_module, "_rename_directory_no_replace", blocked_rename)
    monkeypatch.setattr(local_module, "_sync_open_directory", fail_typed_publication)

    async def exercise() -> asyncio.Task:
        task = asyncio.create_task(
            store.put_bytes(
                b"generated-content",
                filename="generated.txt",
                content_type="text/plain",
                session_id="sess_generated",
            )
        )
        assert await asyncio.to_thread(rename_entered.wait, 10)
        task.cancel("cancel with cleanup failures")
        release_rename.set()
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel with cleanup failures",
        ) as exc_info:
            await task
        observed = _assert_acyclic_exception_graph(exc_info.value)
        for original in original_errors:
            assert sum(item is original for item in observed) == 1
        return task

    task = asyncio.run(exercise())
    assert task.cancelled()
    assert len(asyncio.run(store.list(session_id="sess_generated")).artifacts) == 1


def test_metadata_failure_and_staging_cleanup_preserve_both_failures(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)

    real_write = local_module._write_artifact_file
    primary_error = OSError("primary write failed")
    cleanup_error = OSError("staging cleanup failed")

    def fail_write(*args, **kwargs) -> None:
        if args[3] == "metadata.json":
            raise primary_error
        real_write(*args, **kwargs)

    def fail_cleanup(*_args, **_kwargs) -> None:
        raise cleanup_error

    monkeypatch.setattr(local_module, "_write_artifact_file", fail_write)
    monkeypatch.setattr(
        local_module,
        "_remove_artifact_directory_if_unchanged",
        fail_cleanup,
    )

    with pytest.raises(ArtifactStoreUnavailableError) as exc_info:
        _put(store, artifact_id=_ARTIFACT_ID)

    observed = _assert_acyclic_exception_graph(exc_info.value)
    assert sum(error is primary_error for error in observed) == 1
    assert sum(error is cleanup_error for error in observed) == 1
    group = exc_info.value.__cause__
    assert isinstance(group, BaseExceptionGroup)
    assert [str(error) for error in group.exceptions] == [
        "primary write failed",
        "staging cleanup failed",
    ]
    settlement = artifact_write_settlements(exc_info.value)
    assert len(settlement) == 1
    assert settlement[0].status is ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED
    assert settlement[0].failure_codes == (
        ArtifactWriteSettlementFailureCode.MUTATION_FAILED,
        ArtifactWriteSettlementFailureCode.CLEANUP_FAILED,
    )


@pytest.mark.parametrize("failed_file_index", (1, 2))
@pytest.mark.parametrize("failure_phase", ("partial-write", "post-flush"))
def test_local_artifact_put_cleans_partial_file_failures(
    monkeypatch,
    tmp_path,
    failed_file_index,
    failure_phase,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    real_fdopen = local_module.os.fdopen
    writable_files = 0
    exercised_files: list[int] = []

    class FailingFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def write(self, content):
            if failure_phase == "partial-write":
                self.wrapped.write(content[: max(1, len(content) // 2)])
                raise OSError("partial file write failed")
            return self.wrapped.write(content)

        def flush(self):
            self.wrapped.flush()
            if failure_phase == "post-flush":
                raise OSError("file flush failed")

    def failing_fdopen(descriptor, mode="r", *args, **kwargs):
        nonlocal writable_files
        opened = real_fdopen(descriptor, mode, *args, **kwargs)
        if "w" not in mode:
            return opened
        writable_files += 1
        exercised_files.append(writable_files)
        return FailingFile(opened) if writable_files == failed_file_index else opened

    monkeypatch.setattr(local_module.os, "fdopen", failing_fdopen)

    with pytest.raises(ArtifactStoreUnavailableError):
        _put(store, artifact_id=_ARTIFACT_ID)

    assert exercised_files == list(range(1, failed_file_index + 1))
    assert list(root.iterdir()) == []


def test_unsupported_directory_sync_rejects_writes_but_preserves_reads(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "artifacts"
    root.mkdir()
    store = LocalArtifactStore(root)
    artifact = _put(store, artifact_id=_ARTIFACT_ID)
    monkeypatch.setattr(local_module, "_SUPPORTS_DURABLE_DIRECTORY_SYNC", False)

    assert asyncio.run(store.read_bytes(artifact.id)).content == b"durable-content"
    with pytest.raises(ArtifactStoreUnavailableError, match="directory synchronization"):
        _put(store)
    assert [path.name for path in root.iterdir()] == [_ARTIFACT_ID]


def test_orphan_staging_directory_is_ignored_and_does_not_block_retry(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    orphan = root / f"{_ARTIFACT_ID}.staging-{'b' * 32}"
    orphan.mkdir()
    (orphan / "content").write_bytes(b"abandoned")
    reopened = LocalArtifactStore(root)

    assert asyncio.run(reopened.list()).artifacts == ()
    artifact = _put(reopened, artifact_id=_ARTIFACT_ID)

    assert artifact.id == _ARTIFACT_ID
    assert orphan.is_dir()
    assert asyncio.run(reopened.read_bytes(_ARTIFACT_ID)).content == b"durable-content"


@pytest.mark.parametrize("content_state", ("missing", "truncated"))
def test_listing_excludes_incomplete_legacy_artifact_pairs(tmp_path, content_state):
    root = tmp_path / content_state
    root.mkdir()
    store = LocalArtifactStore(root)
    artifact = _put(store, artifact_id=_ARTIFACT_ID)
    content_path = root / artifact.id / "content"
    if content_state == "missing":
        content_path.unlink()
    else:
        content_path.write_bytes(b"short")

    reopened = LocalArtifactStore(root)
    listed = asyncio.run(reopened.list())

    assert listed.artifacts == ()
    assert listed.total_count == 0


def test_deterministic_retry_repairs_matching_metadata_only_legacy_pair(tmp_path):
    root = tmp_path / "metadata-only"
    root.mkdir()
    store = LocalArtifactStore(root)
    artifact = _put(store, artifact_id=_ARTIFACT_ID)
    (root / artifact.id / "content").unlink()

    reopened = LocalArtifactStore(root)
    assert asyncio.run(reopened.list()).artifacts == ()

    repaired = _put(reopened, artifact_id=_ARTIFACT_ID)

    assert repaired.id == _ARTIFACT_ID
    assert asyncio.run(reopened.read_bytes(_ARTIFACT_ID)).content == b"durable-content"


@pytest.mark.skipif(os.name == "nt", reason="Local durable publication is POSIX-only.")
@pytest.mark.parametrize(
    "phase",
    (
        "root-during-marker-write",
        "root-after-marker-sync",
        "root-after-root-create",
        "root-after-root-sync",
        "root-after-parent-sync",
        "root-after-marker-remove",
        "root-raced-during-marker-write",
        "root-raced-after-marker-sync",
        "root-raced-after-root-sync",
        "root-raced-after-parent-sync",
        "root-raced-after-marker-remove",
    ),
)
def test_local_artifact_root_creation_recovers_after_process_death(tmp_path, phase):
    root = tmp_path / phase / "artifacts"
    root.parent.mkdir()
    worker = Path(__file__).with_name("_local_durability_worker.py")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

    result = subprocess.run(
        [sys.executable, str(worker), str(root), phase, _ARTIFACT_ID],
        env=environment,
        check=False,
        timeout=20,
    )

    assert result.returncode == 86
    reopened = LocalArtifactStore(root)
    artifact = _put(reopened, artifact_id=_ARTIFACT_ID)
    assert artifact.id == _ARTIFACT_ID
    assert not local_module._root_pending_marker(root).exists()
    assert asyncio.run(reopened.read_bytes(_ARTIFACT_ID)).content == b"durable-content"


@pytest.mark.skipif(os.name == "nt", reason="Local durable publication is POSIX-only.")
@pytest.mark.parametrize(
    ("phase", "published"),
    (
        ("after-content-sync", False),
        ("after-metadata-sync", False),
        ("after-staging-sync", False),
        ("after-publish", True),
        ("after-root-sync", True),
    ),
)
def test_local_artifact_process_death_never_exposes_partial_final_artifact(
    tmp_path,
    phase,
    published,
):
    root = tmp_path / phase
    root.mkdir()
    worker = Path(__file__).with_name("_local_durability_worker.py")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")

    result = subprocess.run(
        [sys.executable, str(worker), str(root), phase, _ARTIFACT_ID],
        env=environment,
        check=False,
        timeout=20,
    )

    assert result.returncode == 86
    reopened = LocalArtifactStore(root)
    target = root / _ARTIFACT_ID
    assert target.exists() is published
    if published:
        assert asyncio.run(reopened.read_bytes(_ARTIFACT_ID)).content == b"durable-content"
    else:
        assert asyncio.run(reopened.list()).artifacts == ()

    retry = _put(reopened, artifact_id=_ARTIFACT_ID)
    assert retry.id == _ARTIFACT_ID
    assert asyncio.run(reopened.read_bytes(_ARTIFACT_ID)).content == b"durable-content"


@pytest.mark.skipif(os.name == "nt", reason="Local durable publication is POSIX-only.")
def test_acknowledged_artifact_survives_writer_process_termination(tmp_path):
    root = tmp_path / "acknowledged"
    root.mkdir()
    worker = Path(__file__).with_name("_local_durability_worker.py")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    process = subprocess.Popen(
        [sys.executable, str(worker), str(root), "acknowledged", _ARTIFACT_ID],
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "acknowledged"
        process.kill()
        assert process.wait(timeout=10) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    reopened = LocalArtifactStore(root)
    assert asyncio.run(reopened.read_bytes(_ARTIFACT_ID)).content == b"durable-content"
