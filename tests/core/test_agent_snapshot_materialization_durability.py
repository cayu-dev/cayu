from __future__ import annotations

import asyncio
import errno
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.core.test_agent_bundles import _portable_fixture, _TestMaterializationAuthority

from cayu.snapshots.base import (
    AgentSnapshot,
    AgentSnapshotAccess,
    AgentSnapshotCaptureRequest,
    AgentSnapshotComponentKind,
    AgentSnapshotComponentRef,
    AgentSnapshotComponentSelector,
    AgentSnapshotCoordinator,
    AgentSnapshotMaterialization,
    AgentSnapshotMaterializationOperation,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotMaterializedComponent,
    AgentSnapshotTrialStateMode,
    InMemoryAgentSnapshotStore,
)
from cayu.snapshots.bundles import (
    AgentBundleCoordinator,
    AgentBundleError,
    AgentBundleMaterializationRequest,
    AgentBundleMode,
    AgentSnapshotComponentPackage,
    AgentSnapshotMaterializationMode,
    FileSystemAgentSnapshotObjectStore,
    PortableAgentSnapshotComponentProvider,
    agent_snapshot_component_package,
)


@dataclass
class MaterializationFixture:
    provider: PortableAgentSnapshotComponentProvider
    snapshot: AgentSnapshot
    component: AgentSnapshotComponentRef
    request: AgentSnapshotMaterializationRequest
    operation: AgentSnapshotMaterializationOperation

    @property
    def destination(self) -> Path:
        return self.provider.materialization_root / self.request.state_scope_id / "workspace"

    @property
    def stage(self) -> Path:
        return self.destination.with_name(f".workspace.{self.operation.operation_id}.tmp")

    def materialize(self) -> AgentSnapshotMaterializedComponent:
        return asyncio.run(
            self.provider.materialize(self.snapshot, self.component, self.request, self.operation)
        )

    def completed(self, result: AgentSnapshotMaterializedComponent) -> AgentSnapshotMaterialization:
        return AgentSnapshotMaterialization.create(
            progress_id="1" * 64,
            request=self.request,
            created_at=datetime(2026, 10, 22, tzinfo=UTC),
            components=(result,),
        )

    def recover(self, route: str, result: AgentSnapshotMaterializedComponent):
        if route == "retry":
            return self.materialize()
        if route == "operation":
            return asyncio.run(
                self.provider.recover_materialization_operation(
                    self.snapshot, self.component, self.request, self.operation
                )
            )
        assert route == "completed"
        return asyncio.run(
            self.provider.recover(self.snapshot, self.component, result, self.completed(result))
        )


@pytest.fixture
def materialization(tmp_path: Path, request) -> MaterializationFixture:
    collision = getattr(request, "param", None)

    async def build() -> MaterializationFixture:
        providers, objects, subject, scope = _portable_fixture(tmp_path / "source")
        previous = next(p for p in providers if p.kind is AgentSnapshotComponentKind.WORKSPACE)
        files = {
            "nested/deeper/check.py": b"#!/usr/bin/env python3\nprint(42)\n",
            "notes.txt": b"retained notes\n",
        }
        if collision is not None:
            files[collision] = b"declared snapshot bytes\n"
        package, blobs = agent_snapshot_component_package(
            kind=previous.kind,
            provider_id=previous.provider_id,
            component_schema=previous.package.component_schema,
            profile=previous.package.profile,
            payload={"promoted_paths": ["nested/deeper/check.py", "notes.txt"]},
            files=files,
            executable_paths=("nested/deeper/check.py",),
        )
        provider = PortableAgentSnapshotComponentProvider(
            package, blobs, object_store=objects, materialization_root=tmp_path / "materialized"
        )
        providers[providers.index(previous)] = provider
        snapshot = await AgentSnapshotCoordinator(
            providers, store=InMemoryAgentSnapshotStore()
        ).capture(
            AgentSnapshotCaptureRequest(
                capture_request_id="durability-fixture",
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=p.kind)
                    for p in sorted(providers, key=lambda p: str(p.kind))
                ),
            )
        )
        component = next(c for c in snapshot.components if c.kind is provider.kind)
        request = AgentSnapshotMaterializationRequest(
            access=AgentSnapshotAccess(
                snapshot=snapshot.ref,
                binding_id=snapshot.identity_binding.binding_id,
                authority_scope_fingerprint=scope,
            ),
            candidate_id="durability-candidate",
            trial_id="durability-trial",
            state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        )
        # Exercise inline and public object-store copy_to paths in every publication.
        provider.blobs.pop(next(f.digest for f in package.files if f.executable))
        return MaterializationFixture(
            provider,
            snapshot,
            component,
            request,
            AgentSnapshotMaterializationOperation.create(request=request, component=component),
        )

    return asyncio.run(build())


def fd_is_path(descriptor: int, path: Path) -> bool:
    try:
        actual, expected = os.fstat(descriptor), path.stat()
    except FileNotFoundError:
        return False
    return (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)


BARRIERS = (
    "notes.txt",
    "nested/deeper/check.py",
    ".cayu-materialization.json",
    "nested/deeper",
    "nested",
    ".",
    "scope",
    "root",
)
TREE_BARRIERS = BARRIERS[:6]


def barrier_path(fixture: MaterializationFixture, barrier: str, *, retained: bool) -> Path | None:
    tree = fixture.destination if retained or fixture.destination.exists() else fixture.stage
    # Do not confuse initialization or copy_to's pre-chmod flush with the completion fence.
    if not (tree / ".cayu-materialization.json").exists():
        return None
    if barrier == "scope":
        return tree.parent if fixture.destination.exists() else None
    if barrier == "root":
        return fixture.provider.materialization_root if fixture.destination.exists() else None
    return tree / barrier


@pytest.mark.parametrize("barrier", BARRIERS)
def test_fresh_publication_refuses_each_completion_barrier(materialization, monkeypatch, barrier):
    original = os.fsync
    reached = []

    def fail(descriptor):
        target = barrier_path(materialization, barrier, retained=False)
        if target is not None and fd_is_path(descriptor, target):
            reached.append(target)
            raise OSError(errno.EIO, "injected publication failure")
        original(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="injected publication failure"):
            materialization.materialize()
    assert reached
    retained_inode = (
        materialization.destination.stat().st_ino if materialization.destination.exists() else None
    )
    result = materialization.materialize()
    assert result.baseline_fingerprint == materialization.component.logical.fingerprint
    if retained_inode is not None:
        assert materialization.destination.stat().st_ino == retained_inode


@pytest.mark.parametrize("route", ("retry", "operation", "completed"))
@pytest.mark.parametrize("barrier", BARRIERS)
def test_retained_recovery_resynchronizes_each_barrier(
    materialization, monkeypatch, route, barrier
):
    result = materialization.materialize()
    before = {
        p.relative_to(materialization.destination): (p.stat().st_ino, p.read_bytes())
        for p in materialization.destination.rglob("*")
        if p.is_file()
    }
    original = os.fsync
    reached = []

    def fail(descriptor):
        target = barrier_path(materialization, barrier, retained=True)
        if target is not None and fd_is_path(descriptor, target):
            reached.append(target)
            raise OSError(errno.EIO, "retained barrier failure")
        original(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="retained barrier failure"):
            materialization.recover(route, result)
    assert reached
    assert materialization.recover(route, result) == result
    assert before == {
        p.relative_to(materialization.destination): (p.stat().st_ino, p.read_bytes())
        for p in materialization.destination.rglob("*")
        if p.is_file()
    }


def test_final_modes_and_receipt_precede_publication_without_private_tree_metadata(
    materialization, monkeypatch
):
    original_sync, original_rename = os.fsync, os.rename
    observed = set()

    def sync(descriptor):
        for relative in TREE_BARRIERS:
            path = materialization.stage / relative
            if (
                fd_is_path(descriptor, path)
                and (materialization.stage / ".cayu-materialization.json").exists()
            ):
                observed.add(relative)
                if relative == "nested/deeper/check.py":
                    assert stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o700
        original_sync(descriptor)

    def rename(source, destination, *args, **kwargs):
        if source == materialization.stage.name:
            assert observed == set(TREE_BARRIERS)
        return original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(os, "supports_dir_fd", os.supports_dir_fd | {rename})
    result = materialization.materialize()
    assert result.overlay is not None
    assert {
        p.relative_to(materialization.destination).as_posix()
        for p in materialization.destination.rglob("*")
        if p.is_file()
    } == {"notes.txt", "nested/deeper/check.py", ".cayu-materialization.json"}
    assert stat.S_IMODE((materialization.destination / "notes.txt").stat().st_mode) == 0o600


@pytest.mark.parametrize("missing", ("root", "scope", "component", "receipt"))
def test_completed_recovery_never_reconstructs_missing_state(materialization, missing):
    result = materialization.materialize()
    target = {
        "root": materialization.provider.materialization_root,
        "scope": materialization.destination.parent,
        "component": materialization.destination,
        "receipt": materialization.destination / ".cayu-materialization.json",
    }[missing]
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    with pytest.raises(AgentBundleError):
        materialization.recover("completed", result)
    assert not target.exists()


@pytest.mark.parametrize("materialization", [".cayu-materialization.json"], indirect=True)
@pytest.mark.parametrize("route", ["fresh", "retry", "operation", "completed"])
def test_reserved_receipt_file_refuses_before_materialization_effects(
    materialization, monkeypatch, route
):
    from cayu.snapshots import bundles

    result = None
    before = None
    if route != "fresh":
        # Reproduce a retained predecessor outcome: declared bytes were overwritten
        # by the native receipt. Recovery must refuse it without repairing or syncing.
        with monkeypatch.context() as patch:
            patch.setattr(
                materialization.provider,
                "_require_materializable_package",
                lambda: None,
                raising=False,
            )
            result = materialization.materialize()
        before = {
            p.relative_to(materialization.destination): (p.stat().st_ino, p.read_bytes())
            for p in materialization.destination.rglob("*")
            if p.is_file()
        }

    def no_parent(*args, **kwargs):
        pytest.fail("reserved component must refuse before opening or preparing directories")

    monkeypatch.setattr(bundles, "_open_materialization_parent", no_parent)
    with pytest.raises(AgentBundleError, match="materialization_reserved_path"):
        if route == "fresh":
            materialization.materialize()
        else:
            materialization.recover(route, result)
    if before is None:
        assert not materialization.provider.materialization_root.exists()
    else:
        assert before == {
            p.relative_to(materialization.destination): (p.stat().st_ino, p.read_bytes())
            for p in materialization.destination.rglob("*")
            if p.is_file()
        }


@pytest.mark.parametrize(
    "materialization",
    [".cayu-materialization.json/child", ".CAYU-MATERIALIZATION.JSON"],
    indirect=True,
)
def test_reserved_receipt_prefix_and_portable_case_alias_refuse_before_effects(materialization):
    with pytest.raises(AgentBundleError, match="materialization_reserved_path"):
        materialization.materialize()
    assert not materialization.provider.materialization_root.exists()


@pytest.mark.parametrize("code", [errno.ENOENT, errno.ELOOP, errno.ENOTDIR])
def test_materialization_parent_does_not_translate_caller_body_errors(materialization, code):
    from cayu.snapshots import bundles

    problem = OSError(code, "caller body failure")
    with (
        pytest.raises(OSError) as raised,
        bundles._open_materialization_parent(
            materialization.provider.materialization_root,
            materialization.request.state_scope_id,
            materialization.provider.kind.value,
            create=True,
        ),
    ):
        raise problem
    assert raised.value is problem


def test_missing_copy_object_store_is_not_recovery_destination_missing(materialization):
    shutil.rmtree(materialization.provider.object_store.root / "objects")
    with pytest.raises(FileNotFoundError):
        materialization.materialize()


def _observed(path: Path) -> tuple[object, ...]:
    # Everything recovery could change. Not st_atime: reading the file updates it,
    # so it differs whenever the read lands in the next second.
    info = path.lstat()
    return (
        info.st_mode,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        path.read_bytes(),
    )


@pytest.mark.parametrize("mutation", ("content", "executable", "receipt", "extra", "symlink"))
def test_recovery_refuses_mutated_state_without_repair(materialization, mutation):
    result = materialization.materialize()
    target = materialization.destination / "notes.txt"
    if mutation == "content":
        target.write_bytes(b"changed state")
    elif mutation == "executable":
        target.chmod(0o700)
    elif mutation == "receipt":
        target = materialization.destination / ".cayu-materialization.json"
        data = json.loads(target.read_bytes())
        data["operation_id"] = "0" * 64
        target.write_text(json.dumps(data))
    elif mutation == "extra":
        target = materialization.destination / "unlisted.txt"
        target.write_bytes(b"extra")
    else:
        target.unlink()
        target.symlink_to(materialization.destination / "nested/deeper/check.py")
    before = _observed(target)
    with pytest.raises(AgentBundleError):
        materialization.recover("completed", result)
    assert _observed(target) == before


def test_recovery_preserves_permitted_modes_receipt_bytes_and_empty_directories(materialization):
    result = materialization.materialize()
    target = materialization.destination / "notes.txt"
    target.chmod(0o640)
    receipt = materialization.destination / ".cayu-materialization.json"
    retained = b"  " + receipt.read_bytes() + b"\n"
    receipt.write_bytes(retained)
    (materialization.destination / "ordinary-empty").mkdir()
    assert materialization.recover("completed", result) == result
    assert receipt.read_bytes() == retained
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert (materialization.destination / "ordinary-empty").is_dir()


def test_traversal_failure_is_not_an_empty_subtree(materialization, monkeypatch):
    result = materialization.materialize()
    original = os.listdir

    def fail(path):
        if isinstance(path, int) and fd_is_path(path, materialization.destination / "nested"):
            raise OSError(errno.EIO, "directory traversal unavailable")
        return original(path)

    monkeypatch.setattr(os, "listdir", fail)
    monkeypatch.setattr(os, "supports_fd", os.supports_fd | {fail})
    with pytest.raises(OSError, match="directory traversal unavailable"):
        materialization.recover("completed", result)


@pytest.mark.parametrize("route", ("fresh", "operation", "completed"))
def test_unsupported_durability_refuses_before_mutation_but_allows_read_only_verification(
    materialization, monkeypatch, route
):
    result = materialization.materialize() if route == "completed" else None
    before = set(materialization.provider.materialization_root.rglob("*")) if result else set()
    monkeypatch.setattr(os, "supports_fd", os.supports_fd - {os.listdir})
    # The provider can still be constructed and inspect existing native packages.
    assert asyncio.run(
        materialization.provider.verify(materialization.snapshot, materialization.component)
    )
    with pytest.raises(AgentBundleError, match="materialization_durability_unavailable"):
        if route == "fresh":
            materialization.materialize()
        elif route == "operation":
            asyncio.run(
                materialization.provider.recover_materialization_operation(
                    materialization.snapshot,
                    materialization.component,
                    materialization.request,
                    materialization.operation,
                )
            )
        else:
            materialization.recover("completed", result)
    after = (
        set(materialization.provider.materialization_root.rglob("*"))
        if materialization.provider.materialization_root.exists()
        else set()
    )
    assert before == after


def clone_provider(fixture):
    return PortableAgentSnapshotComponentProvider(
        fixture.provider.package,
        fixture.provider.blobs,
        object_store=fixture.provider.object_store,
        materialization_root=fixture.provider.materialization_root,
    )


@pytest.mark.parametrize("target_name", ("root", "scope", "component", "file"))
def test_retained_barrier_refuses_named_inode_replacement(
    materialization, monkeypatch, target_name
):
    result = materialization.materialize()
    target = {
        "root": materialization.provider.materialization_root,
        "scope": materialization.destination.parent,
        "component": materialization.destination,
        "file": materialization.destination / "notes.txt",
    }[target_name]
    moved = target.with_name(target.name + ".displaced")
    original = os.fsync
    changed = False

    def replace(descriptor):
        nonlocal changed
        if not changed and fd_is_path(descriptor, target):
            changed = True
            target.rename(moved)
            if target_name == "file":
                target.write_bytes(moved.read_bytes())
            else:
                target.mkdir()
        original(descriptor)

    monkeypatch.setattr(os, "fsync", replace)
    with pytest.raises((AgentBundleError, OSError)) as failure:
        materialization.recover("completed", result)
    if isinstance(failure.value, OSError):
        assert failure.value.errno == errno.ESTALE
    assert changed
    assert moved.exists()


def test_retained_wrapper_failure_closes_owned_descriptor(materialization, monkeypatch):
    result = materialization.materialize()
    original = os.fdopen
    descriptors = []

    def fail(descriptor, *args, **kwargs):
        if fd_is_path(descriptor, materialization.destination / "notes.txt"):
            descriptors.append(descriptor)
            raise OSError(errno.EIO, "injected wrapper failure")
        return original(descriptor, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", fail)
    for _ in range(10):
        with pytest.raises(OSError, match="injected wrapper failure"):
            materialization.recover("completed", result)
        with pytest.raises(OSError) as closed:
            os.fstat(descriptors[-1])
        assert closed.value.errno == errno.EBADF
    assert len(descriptors) == 10


def test_concurrent_exact_writers_converge_without_recopy(materialization, monkeypatch):
    original = materialization.provider.object_store.copy_to
    calls = []

    async def copy(reference, destination):
        calls.append(reference.digest)
        await original(reference, destination)

    monkeypatch.setattr(materialization.provider.object_store, "copy_to", copy)
    replacement = clone_provider(materialization)

    async def run():
        return await asyncio.gather(
            materialization.provider.materialize(
                materialization.snapshot,
                materialization.component,
                materialization.request,
                materialization.operation,
            ),
            replacement.materialize(
                materialization.snapshot,
                materialization.component,
                materialization.request,
                materialization.operation,
            ),
        )

    first, second = asyncio.run(run())
    assert first == second
    assert len(calls) == 1


def test_cancelled_caller_recovery_settles_dispatched_writer_without_recopy(
    materialization, monkeypatch
):
    original = materialization.provider.object_store.copy_to
    started, release = threading.Event(), threading.Event()
    calls = []

    async def copy(reference, destination):
        calls.append(reference.digest)
        started.set()
        assert release.wait(5), "test must release its own writer"
        await original(reference, destination)

    monkeypatch.setattr(materialization.provider.object_store, "copy_to", copy)
    replacement = clone_provider(materialization)

    async def run():
        task = asyncio.create_task(
            materialization.provider.materialize(
                materialization.snapshot,
                materialization.component,
                materialization.request,
                materialization.operation,
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        return await replacement.recover_materialization_operation(
            materialization.snapshot,
            materialization.component,
            materialization.request,
            materialization.operation,
        )

    result = asyncio.run(run())
    assert len(calls) == 1
    assert materialization.recover("completed", result) == result


def save_vector(fixture, path):
    path.write_text(
        json.dumps(
            {
                "package": fixture.provider.package.model_dump(mode="json"),
                "blobs": {digest: data.hex() for digest, data in fixture.provider.blobs.items()},
                "objects": str(fixture.provider.object_store.root),
                "root": str(fixture.provider.materialization_root),
                "snapshot": fixture.snapshot.model_dump(mode="json"),
                "component": fixture.component.model_dump(mode="json"),
                "request": fixture.request.model_dump(mode="json"),
                "operation": fixture.operation.model_dump(mode="json"),
            }
        )
    )


def load_vector(path):
    data = json.loads(Path(path).read_bytes())
    provider = PortableAgentSnapshotComponentProvider(
        AgentSnapshotComponentPackage.model_validate(data["package"]),
        {digest: bytes.fromhex(content) for digest, content in data["blobs"].items()},
        object_store=FileSystemAgentSnapshotObjectStore(data["objects"]),
        materialization_root=data["root"],
    )
    return MaterializationFixture(
        provider,
        AgentSnapshot.model_validate(data["snapshot"]),
        AgentSnapshotComponentRef.model_validate(data["component"]),
        AgentSnapshotMaterializationRequest.model_validate(data["request"]),
        AgentSnapshotMaterializationOperation.model_validate(data["operation"]),
    )


PROCESS_CUT = r"""
import os, sys
from pathlib import Path
from tests.core.test_agent_snapshot_materialization_durability import load_vector, fd_is_path
fixture = load_vector(sys.argv[1])
cut = sys.argv[2]
original_write, original_rename, original_sync = Path.write_bytes, os.rename, os.fsync

def write(path, data):
    result = original_write(path, data)
    if cut == "receipt" and path == fixture.stage / ".cayu-materialization.json":
        os._exit(74)
    return result

def rename(source, destination, *args, **kwargs):
    result = original_rename(source, destination, *args, **kwargs)
    if cut == "rename" and source == fixture.stage.name:
        os._exit(74)
    return result

def sync(descriptor):
    original_sync(descriptor)
    if cut == "scope" and fixture.destination.exists() and fd_is_path(descriptor, fixture.destination.parent):
        os._exit(74)

Path.write_bytes, os.rename, os.fsync = write, rename, sync
os.supports_dir_fd = os.supports_dir_fd | {rename}
result = fixture.materialize()
print(result.model_dump_json())
"""


def child_environment(tmp_path):
    return {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "TMPDIR": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }


@pytest.mark.parametrize("cut", ("receipt", "rename", "scope"))
def test_replacement_process_settles_exact_materialization_after_cut(
    materialization, tmp_path, cut
):
    vector = tmp_path / "vector.json"
    save_vector(materialization, vector)
    crashed = subprocess.run(
        [sys.executable, "-B", "-c", PROCESS_CUT, str(vector), cut],
        env=child_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert crashed.returncode == 74, crashed.stderr
    retained_inode = (
        materialization.destination.stat().st_ino if materialization.destination.exists() else None
    )
    replacement = subprocess.run(
        [sys.executable, "-B", "-c", PROCESS_CUT, str(vector), "none"],
        env=child_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert replacement.returncode == 0, replacement.stderr
    result = AgentSnapshotMaterializedComponent.model_validate_json(replacement.stdout)
    assert materialization.materialize() == result
    assert (
        json.loads((materialization.destination / ".cayu-materialization.json").read_bytes())[
            "operation_id"
        ]
        == materialization.operation.operation_id
    )
    if retained_inode is not None:
        assert materialization.destination.stat().st_ino == retained_inode


@pytest.mark.parametrize("which", ("root", "scope"))
def test_actual_materialization_retries_interrupted_parent_preparation(
    materialization, monkeypatch, which
):
    target = (
        materialization.provider.materialization_root
        if which == "root"
        else materialization.destination.parent
    )
    original = os.fsync

    def fail(descriptor):
        if (target / ".cayu-snapshot-directory.pending").exists() and fd_is_path(
            descriptor, target.parent
        ):
            raise OSError(errno.EIO, "parent preparation interrupted")
        original(descriptor)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail)
        with pytest.raises(OSError, match="parent preparation interrupted"):
            materialization.materialize()
    assert (target / ".cayu-snapshot-directory.pending").exists()
    assert not materialization.destination.exists()
    materialization.provider = clone_provider(materialization)
    materialization.materialize()
    assert not (target / ".cayu-snapshot-directory.pending").exists()


def test_unsupported_bundle_materialization_refuses_before_authority(
    materialization, tmp_path, monkeypatch
):
    async def run():
        store = InMemoryAgentSnapshotStore()
        await store.save_snapshot(materialization.snapshot)
        authority = _TestMaterializationAuthority()
        coordinator = AgentBundleCoordinator(
            snapshot_store=store,
            object_store=materialization.provider.object_store,
            materialization_authority=authority,
        )
        exported = await coordinator.export(
            operation_id="unsupported-fixture",
            access=materialization.request.access,
            profile=materialization.provider.package.profile,
            destination=tmp_path / "exported",
            mode=AgentBundleMode.FULL,
        )
        request = AgentBundleMaterializationRequest(
            operation_id="unsupported",
            bundle=exported.bundle,
            access=materialization.request.access,
            mode=AgentSnapshotMaterializationMode.MATERIALIZE,
            candidate_id="unsupported",
            trial_id="unsupported",
            state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        )
        monkeypatch.setattr(os, "supports_fd", os.supports_fd - {os.listdir})
        with pytest.raises(AgentBundleError, match="materialization_durability_unavailable"):
            await coordinator.materialize(request, materialization_root=tmp_path / "unsupported")
        assert authority.requests == []
        assert not (tmp_path / "unsupported").exists()

    asyncio.run(run())
