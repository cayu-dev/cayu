"""Filesystem CAS acknowledgement must include exact bytes and named entries."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest

from cayu.snapshots import AgentBundleError, AgentBundleObjectKind, AgentBundleObjectRef, bundles
from cayu.snapshots import _filesystem_publication as publication

CONTENT = b"one exact retained object\n"
REFERENCE = AgentBundleObjectRef(
    digest=sha256(CONTENT).hexdigest(),
    kind=AgentBundleObjectKind.COMPONENT_BLOB,
    schema_id="test.snapshot-object.v1",
    byte_count=len(CONTENT),
)


def object_path(root: Path) -> Path:
    return root / "objects" / REFERENCE.digest[:2] / REFERENCE.digest[2:]


def subprocess_environment(tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }


def put(store, source: Path, method: str) -> None:
    asyncio.run(
        store.put(REFERENCE, CONTENT) if method == "put" else store.put_file(REFERENCE, source)
    )


@pytest.mark.parametrize("method", ["put", "put_file"])
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("cut", ["file", "shard", "objects", "root"])
def test_publication_and_exact_retry_refuse_every_missing_barrier(
    tmp_path, monkeypatch, method, retry, cut
):
    root = tmp_path / "cas"
    store = bundles.FileSystemAgentSnapshotObjectStore(root)
    source = tmp_path / "source"
    source.write_bytes(CONTENT)
    target = object_path(root)
    target.parent.mkdir()
    if retry:
        put(store, source, method)
    actual = os.fsync
    names = {
        publication.identity(path.stat()): name
        for path, name in ((root, "root"), (root / "objects", "objects"), (target.parent, "shard"))
    }

    def fail(descriptor):
        info = os.fstat(descriptor)
        name = names.get(publication.identity(info)) if stat.S_ISDIR(info.st_mode) else "file"
        if name == cut:
            raise OSError("injected CAS publication barrier")
        actual(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="injected CAS publication barrier"):
            put(store, source, method)
    retained = target.stat().st_ino if target.exists() else None
    reopened = bundles.FileSystemAgentSnapshotObjectStore(root)
    put(reopened, source, method)
    assert target.read_bytes() == CONTENT
    if retained is not None:
        assert target.stat().st_ino == retained
    assert asyncio.run(reopened.inventory()).object_digests == (REFERENCE.digest,)


@pytest.mark.parametrize("method", ["put", "put_file"])
def test_exact_retry_flushes_the_existing_file_and_directory_chain(tmp_path, monkeypatch, method):
    root = tmp_path / "cas"
    store = bundles.FileSystemAgentSnapshotObjectStore(root)
    source = tmp_path / "source"
    source.write_bytes(CONTENT)
    put(store, source, method)
    path = object_path(root)
    expected = [publication.identity(p.stat()) for p in (path, path.parent, root / "objects", root)]
    observed = []
    actual = os.fsync

    def sync(descriptor):
        observed.append(publication.identity(os.fstat(descriptor)))
        actual(descriptor)

    monkeypatch.setattr(os, "fsync", sync)
    put(store, source, method)
    assert observed == expected


@pytest.mark.parametrize("changed", ["root", "objects", "shard", "file"])
def test_replay_does_not_acknowledge_a_detached_inode(tmp_path, monkeypatch, changed):
    root = tmp_path / "cas"
    store = bundles.FileSystemAgentSnapshotObjectStore(root)
    asyncio.run(store.put(REFERENCE, CONTENT))
    target = object_path(root)
    trigger = target.parent if changed == "file" else target
    trigger_identity = publication.identity(trigger.stat())
    selected = {"root": root, "objects": root / "objects", "shard": target.parent, "file": target}[
        changed
    ]
    actual = os.fsync
    replaced = False

    def replace_during_barrier(descriptor):
        nonlocal replaced
        actual(descriptor)
        if not replaced and publication.identity(os.fstat(descriptor)) == trigger_identity:
            replaced = True
            selected.rename(tmp_path / "detached")
            if changed == "file":
                selected.write_bytes(CONTENT)
            else:
                selected.mkdir()

    monkeypatch.setattr(os, "fsync", replace_during_barrier)
    with pytest.raises((AgentBundleError, OSError)):
        asyncio.run(store.put(REFERENCE, CONTENT))
    assert replaced


def test_constructor_keeps_alias_path_and_does_not_open_unrelated_ancestor(tmp_path, monkeypatch):
    outer = tmp_path / "traverse"
    root = outer / "cas"
    root.mkdir(parents=True)
    (root / "objects").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    actual = os.fsync
    unrelated = publication.identity(outer.stat())

    def sync(descriptor):
        assert publication.identity(os.fstat(descriptor)) != unrelated
        actual(descriptor)

    monkeypatch.setattr(os, "fsync", sync)
    outer.chmod(0o111)
    try:
        store = bundles.FileSystemAgentSnapshotObjectStore(alias)
        assert store.root == alias
        asyncio.run(store.put(REFERENCE, CONTENT))
        assert asyncio.run(store.get(REFERENCE)) == CONTENT
    finally:
        outer.chmod(0o700)


def test_initializers_and_writers_converge_without_metadata_in_inventory(tmp_path):
    root = tmp_path / "new" / "nested" / "cas"

    def construct_and_write():
        store = bundles.FileSystemAgentSnapshotObjectStore(root)
        asyncio.run(store.put(REFERENCE, CONTENT))
        return asyncio.run(store.inventory())

    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: construct_and_write(), range(4)))
    assert all(result.object_digests == (REFERENCE.digest,) for result in results)
    assert not list(root.rglob(publication.PENDING_FILE))
    assert list(root.rglob(publication.LOCK_FILE))


@pytest.mark.parametrize("level", ["root", "ancestor", "objects"])
def test_initialization_does_not_create_the_target_of_a_broken_alias(tmp_path, level):
    outside = tmp_path / "absent-target"
    root = tmp_path / "cas"
    if level == "objects":
        root.mkdir()
        (root / "objects").symlink_to(outside, target_is_directory=True)
    else:
        root.symlink_to(outside, target_is_directory=True)
        if level == "ancestor":
            root = root / "nested"
    with pytest.raises(FileExistsError):
        bundles.FileSystemAgentSnapshotObjectStore(root)
    assert not outside.exists()


@pytest.mark.parametrize("phase", ["parent_rename", "root_rename", "parent_sync", "marker_remove"])
def test_replacement_process_recovers_interrupted_directory_creation(tmp_path, phase):
    root = tmp_path / "new" / "cas"
    code = r"""
import os, sys, stat
from pathlib import Path
from cayu.snapshots import FileSystemAgentSnapshotObjectStore
from cayu.snapshots import _filesystem_publication as helper
root, phase = Path(sys.argv[1]), sys.argv[2]
rename, sync, unlink = os.rename, os.fsync, os.unlink
def cut_rename(source, destination, **kwargs):
    rename(source, destination, **kwargs)
    if (phase == "parent_rename" and destination == "new") or (phase == "root_rename" and destination == "cas"):
        os._exit(74)
def cut_sync(descriptor):
    info = os.fstat(descriptor)
    if phase == "parent_sync" and root.exists() and stat.S_ISDIR(info.st_mode) and info.st_ino == root.parent.stat().st_ino:
        os._exit(74)
    sync(descriptor)
def cut_unlink(path, **kwargs):
    unlink(path, **kwargs)
    if phase == "marker_remove" and root.exists() and path == helper.PENDING_FILE and os.fstat(kwargs['dir_fd']).st_ino == root.stat().st_ino:
        os._exit(74)
os.rename, os.fsync, os.unlink = cut_rename, cut_sync, cut_unlink
os.supports_dir_fd = set(os.supports_dir_fd) | {cut_rename, cut_unlink}
FileSystemAgentSnapshotObjectStore(root)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(root), phase],
        capture_output=True,
        text=True,
        env=subprocess_environment(tmp_path),
        timeout=30,
    )
    assert result.returncode == 74, result.stderr
    retained = {p: publication.identity(p.stat()) for p in (root.parent, root) if p.exists()}
    store = bundles.FileSystemAgentSnapshotObjectStore(root)
    asyncio.run(store.put(REFERENCE, CONTENT))
    assert all(publication.identity(p.stat()) == expected for p, expected in retained.items())
    assert not list(root.parent.rglob(publication.PENDING_FILE))
    assert asyncio.run(store.inventory()).object_digests == (REFERENCE.digest,)


@pytest.mark.parametrize("method", ["put", "put_file"])
def test_replacement_process_reuses_object_after_rename_without_acknowledgement(tmp_path, method):
    root = tmp_path / "cas"
    bundles.FileSystemAgentSnapshotObjectStore(root)
    source = tmp_path / "source"
    source.write_bytes(CONTENT)
    code = r"""
import asyncio, os, sys
from pathlib import Path
from cayu.snapshots import AgentBundleObjectRef, FileSystemAgentSnapshotObjectStore
root, source, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
reference = AgentBundleObjectRef.model_validate_json(sys.argv[4])
store = FileSystemAgentSnapshotObjectStore(root)
rename = os.rename
def cut_rename(source, destination, **kwargs):
    rename(source, destination, **kwargs)
    if destination == reference.digest[2:]:
        os._exit(74)
os.rename = cut_rename
os.supports_dir_fd = set(os.supports_dir_fd) | {cut_rename}
asyncio.run(store.put(reference, source.read_bytes()) if method == 'put' else store.put_file(reference, source))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            str(root),
            str(source),
            method,
            REFERENCE.model_dump_json(),
        ],
        capture_output=True,
        text=True,
        env=subprocess_environment(tmp_path),
        timeout=30,
    )
    assert result.returncode == 74, result.stderr
    retained = publication.identity(object_path(root).stat())
    reopened = bundles.FileSystemAgentSnapshotObjectStore(root)
    put(reopened, source, method)
    assert publication.identity(object_path(root).stat()) == retained
    assert asyncio.run(reopened.get(REFERENCE)) == CONTENT
    assert asyncio.run(reopened.inventory()).object_digests == (REFERENCE.digest,)


@pytest.mark.parametrize("corruption", ["marker", "lock_symlink", "lock_hardlink"])
def test_pending_initialization_rejects_corrupt_or_substituted_metadata(
    tmp_path, monkeypatch, corruption
):
    root = tmp_path / "cas"
    actual = os.fsync

    def fail_parent(descriptor):
        if root.exists() and publication.identity(os.fstat(descriptor)) == publication.identity(
            tmp_path.stat()
        ):
            raise OSError("uncertain new directory")
        actual(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_parent)
        with pytest.raises(OSError, match="uncertain new directory"):
            bundles.FileSystemAgentSnapshotObjectStore(root)
    if corruption == "marker":
        (root / publication.PENDING_FILE).write_bytes(b"wrong")
    else:
        lock = root / publication.LOCK_FILE
        outside = tmp_path / "outside"
        outside.write_bytes(b"unchanged")
        lock.unlink()
        if corruption == "lock_symlink":
            lock.symlink_to(outside)
        else:
            os.link(outside, lock)
    with pytest.raises(OSError):
        bundles.FileSystemAgentSnapshotObjectStore(root)
    assert (root / publication.PENDING_FILE).exists()
    if corruption != "marker":
        assert outside.read_bytes() == b"unchanged"


def test_unsupported_guards_keep_construction_but_refuse_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(bundles, "_cas_descriptor_guards_available", lambda: False)
    store = bundles.FileSystemAgentSnapshotObjectStore(tmp_path / "cas")
    assert store.root.exists()
    with pytest.raises(AgentBundleError, match="object_store_descriptor_guard_unavailable"):
        asyncio.run(store.put(REFERENCE, CONTENT))


def leave_pending_publication(tmp_path, monkeypatch, level):
    root = tmp_path / "cas"
    pending = root
    store = None
    if level == "objects":
        root.mkdir()
        pending = root / "objects"
    elif level == "shard":
        store = bundles.FileSystemAgentSnapshotObjectStore(root)
        pending = object_path(root).parent
    actual = os.fsync

    def fail_parent(descriptor):
        if (pending / publication.PENDING_FILE).exists() and publication.identity(
            os.fstat(descriptor)
        ) == publication.identity(pending.parent.stat()):
            raise OSError("interrupted directory publication")
        actual(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_parent)
        with pytest.raises(OSError, match="interrupted directory publication"):
            if store is None:
                bundles.FileSystemAgentSnapshotObjectStore(root)
            else:
                asyncio.run(store.put(REFERENCE, CONTENT))
    assert (pending / publication.PENDING_FILE).is_file()
    return root, pending


class RemountedStat:
    """Change only the mount-local device field; retain real inode/timestamps."""

    def __init__(self, original, device):
        self.original = original
        self.st_dev = device

    def __getattr__(self, name):
        return getattr(self.original, name)


def simulate_device_change(monkeypatch, select):
    actual_stat, actual_fstat = os.stat, os.fstat

    def changed(info):
        return RemountedStat(info, info.st_dev + 1000) if select(info) else info

    def remounted_stat(*args, **kwargs):
        return changed(actual_stat(*args, **kwargs))

    monkeypatch.setattr(os, "stat", remounted_stat)
    monkeypatch.setattr(os, "fstat", lambda descriptor: changed(actual_fstat(descriptor)))
    monkeypatch.setattr(os, "supports_dir_fd", set(os.supports_dir_fd) | {remounted_stat})


@pytest.mark.parametrize("level", ["root", "objects", "shard"])
def test_pending_publication_survives_mount_device_renumbering(tmp_path, monkeypatch, level):
    root, pending = leave_pending_publication(tmp_path, monkeypatch, level)
    retained_inode = pending.stat().st_ino
    marker = pending / publication.PENDING_FILE
    original_marker = marker.read_bytes()
    with monkeypatch.context() as patch:
        simulate_device_change(patch, lambda _info: True)
        store = bundles.FileSystemAgentSnapshotObjectStore(root)
        asyncio.run(store.put(REFERENCE, CONTENT))
        assert pending.stat().st_ino == retained_inode
        assert not marker.exists()
        assert asyncio.run(store.get(REFERENCE)) == CONTENT
        assert asyncio.run(store.inventory()).object_digests == (REFERENCE.digest,)
    assert json.loads(original_marker) == {
        "version": 2,
        "parent": pending.parent.stat().st_ino,
        "name": pending.name.casefold(),
        "directory": retained_inode,
    }


def test_pending_publication_rejects_a_different_live_filesystem(tmp_path, monkeypatch):
    root, pending = leave_pending_publication(tmp_path, monkeypatch, "root")
    retained_inode = pending.stat().st_ino
    with monkeypatch.context() as patch:
        simulate_device_change(patch, lambda info: info.st_ino == retained_inode)
        with pytest.raises(OSError, match="crossed filesystems"):
            bundles.FileSystemAgentSnapshotObjectStore(root)
    assert (pending / publication.PENDING_FILE).exists()


@pytest.mark.parametrize("field", ["parent", "directory", "name", "version"])
def test_pending_publication_still_binds_each_persisted_field(tmp_path, monkeypatch, field):
    root, pending = leave_pending_publication(tmp_path, monkeypatch, "root")
    marker = pending / publication.PENDING_FILE
    value = json.loads(marker.read_bytes())
    value[field] = "another-name" if field == "name" else value[field] + 1
    marker.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    with pytest.raises(OSError, match="Invalid snapshot directory publication marker"):
        bundles.FileSystemAgentSnapshotObjectStore(root)
    assert marker.exists()


@pytest.mark.parametrize("method", ["put", "put_file"])
@pytest.mark.parametrize("entry", ["directory", "fifo", "fstat_error", "fdopen_error"])
def test_rejected_object_descriptors_are_always_closed(tmp_path, monkeypatch, method, entry):
    root = tmp_path / "cas"
    store = bundles.FileSystemAgentSnapshotObjectStore(root)
    source = tmp_path / "source"
    source.write_bytes(CONTENT)
    target = object_path(root)
    target.parent.mkdir()
    if entry == "directory":
        target.mkdir()
    elif entry == "fifo":
        os.mkfifo(target)
    else:
        target.write_bytes(CONTENT)
    actual_open, actual_fstat, actual_fdopen = os.open, os.fstat, os.fdopen
    opened = []
    current = None

    def track_open(path, *args, **kwargs):
        nonlocal current
        descriptor = actual_open(path, *args, **kwargs)
        if path == target.name:
            current = descriptor
            opened.append(descriptor)
        return descriptor

    def fail_fstat(descriptor):
        if descriptor == current and entry == "fstat_error":
            raise OSError("injected object fstat failure")
        return actual_fstat(descriptor)

    def fail_fdopen(descriptor, *args, **kwargs):
        if descriptor == current and entry == "fdopen_error":
            raise OSError("injected object wrapper failure")
        return actual_fdopen(descriptor, *args, **kwargs)

    descriptor_directory = Path("/dev/fd")
    initial_count = len(os.listdir(descriptor_directory)) if descriptor_directory.is_dir() else None
    with monkeypatch.context() as patch:
        patch.setattr(os, "open", track_open)
        patch.setattr(os, "fstat", fail_fstat)
        patch.setattr(os, "fdopen", fail_fdopen)
        patch.setattr(os, "supports_dir_fd", set(os.supports_dir_fd) | {track_open})
        for _ in range(20):
            current = None
            if entry in {"directory", "fifo"}:
                with pytest.raises(AgentBundleError, match="object_path_not_regular"):
                    put(store, source, method)
            else:
                with pytest.raises(OSError, match="injected object"):
                    put(store, source, method)
            assert current is not None
            with pytest.raises(OSError) as closed:
                actual_fstat(current)
            assert closed.value.errno == errno.EBADF
    assert len(opened) == 20
    if initial_count is not None:
        assert len(os.listdir(descriptor_directory)) == initial_count
