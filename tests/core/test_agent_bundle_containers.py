from __future__ import annotations

import asyncio
import io
import os
import sqlite3
import stat
import threading
import warnings
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest
from tests.core.test_agent_bundles import _portable_fixture

import cayu.agent_bundle_containers as container_module
from cayu.agent_bundle_containers import (
    AGENT_BUNDLE_CONTAINER_MEDIA_TYPE,
    AGENT_BUNDLE_CONTAINER_MIMETYPE_ENTRY,
    inspect_agent_bundle_container,
    pack_agent_bundle,
    unpack_agent_bundle_container,
)
from cayu.agent_bundles import (
    AGENT_BUNDLE_INDEX_FILENAME,
    AgentBundle,
    AgentBundleCoordinator,
    AgentBundleError,
    AgentBundleInventory,
    AgentBundleMode,
    AgentBundleSizeReport,
    AgentSnapshotProfile,
    FileSystemAgentSnapshotObjectStore,
    _canonical_json,
)
from cayu.agent_snapshots import (
    AgentSnapshotAccess,
    AgentSnapshotCaptureRequest,
    AgentSnapshotComponentSelector,
    AgentSnapshotCoordinator,
    AgentSnapshotGCRequest,
    AgentSnapshotProtection,
    InMemoryAgentSnapshotStore,
    SQLiteAgentSnapshotStore,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _run(coroutine):
    return asyncio.run(coroutine)


async def _export_directory(
    root: Path,
    *,
    body_content: bytes = b"def answer():\n    return 42\n",
    mode: AgentBundleMode = AgentBundleMode.FULL,
    destination_inventory: AgentBundleInventory | None = None,
    snapshot_store: InMemoryAgentSnapshotStore | None = None,
):
    providers, object_store, subject, scope = _portable_fixture(
        root / "source",
        body_content=body_content,
    )
    snapshot_store = snapshot_store or InMemoryAgentSnapshotStore()
    snapshot = await AgentSnapshotCoordinator(providers, store=snapshot_store).capture(
        AgentSnapshotCaptureRequest(
            capture_request_id="container-capture",
            subject=subject,
            authority_scope_fingerprint=scope,
            components=tuple(
                AgentSnapshotComponentSelector(kind=kind)
                for kind in sorted((provider.kind for provider in providers), key=str)
            ),
        )
    )
    access = AgentSnapshotAccess(
        snapshot=snapshot.ref,
        binding_id=snapshot.identity_binding.binding_id,
        authority_scope_fingerprint=scope,
    )
    coordinator = AgentBundleCoordinator(
        snapshot_store=snapshot_store,
        object_store=object_store,
    )
    directory = (root / "bundle.d").resolve()
    export = await coordinator.export(
        operation_id="container-directory-export",
        access=access,
        profile=AgentSnapshotProfile.REUSABLE_AGENT,
        destination=directory,
        mode=mode,
        destination_inventory=destination_inventory,
    )
    return (
        coordinator,
        snapshot_store,
        object_store,
        subject,
        scope,
        snapshot,
        access,
        directory,
        export,
    )


def _archive_entries(path: Path) -> tuple[zipfile.ZipInfo, ...]:
    with zipfile.ZipFile(path) as archive:
        return tuple(archive.infolist())


def test_cayu_pack_is_deterministic_and_preserves_bundle_identity(tmp_path: Path) -> None:
    async def scenario() -> None:
        *_, snapshot, _access, directory, export = await _export_directory(tmp_path)
        first = (tmp_path / "first.cayu").resolve()
        second = (tmp_path / "second.cayu").resolve()
        first_receipt = pack_agent_bundle(directory, first)
        second_receipt = pack_agent_bundle(directory, second)

        assert first.read_bytes() == second.read_bytes()
        assert first_receipt == second_receipt
        assert first_receipt.bundle == export.bundle
        assert first_receipt.inspection.snapshot_root == snapshot.snapshot_root
        assert first_receipt.inspection.transport_sha256 == sha256(first.read_bytes()).hexdigest()
        assert first_receipt.inspection.requires_preexisting_objects is False
        entries = _archive_entries(first)
        assert entries[0].filename == AGENT_BUNDLE_CONTAINER_MIMETYPE_ENTRY
        assert entries[1].filename == AGENT_BUNDLE_INDEX_FILENAME
        assert all(entry.compress_type == zipfile.ZIP_STORED for entry in entries)
        assert all(entry.date_time == (1980, 1, 1, 0, 0, 0) for entry in entries)
        with zipfile.ZipFile(first) as archive:
            assert archive.read(entries[0]) == AGENT_BUNDLE_CONTAINER_MEDIA_TYPE.encode("ascii")

    _run(scenario())


def test_pack_unpack_and_unpack_pack_preserve_exact_bytes(tmp_path: Path) -> None:
    async def scenario() -> None:
        *_, directory, export = await _export_directory(tmp_path)
        container = (tmp_path / "agent.cayu").resolve()
        packed = pack_agent_bundle(directory, container)
        unpacked = (tmp_path / "agent.cayu.d").resolve()
        unpacked_receipt = unpack_agent_bundle_container(container, unpacked)
        repacked = (tmp_path / "agent-repacked.cayu").resolve()
        repacked_receipt = pack_agent_bundle(unpacked, repacked)

        assert packed.bundle == unpacked_receipt.bundle == repacked_receipt.bundle == export.bundle
        assert container.read_bytes() == repacked.read_bytes()
        assert (directory / AGENT_BUNDLE_INDEX_FILENAME).read_bytes() == (
            unpacked / AGENT_BUNDLE_INDEX_FILENAME
        ).read_bytes()
        expected_files = sorted(
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        )
        actual_files = sorted(
            path.relative_to(unpacked).as_posix() for path in unpacked.rglob("*") if path.is_file()
        )
        assert expected_files == actual_files
        for relative in expected_files:
            assert (directory / relative).read_bytes() == (unpacked / relative).read_bytes()

    _run(scenario())


def test_pack_and_unpack_leave_preexisting_legacy_staging_paths_untouched(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        *_, directory, _export = await _export_directory(tmp_path)
        container = (tmp_path / "owned-staging.cayu").resolve()
        legacy_pack_staging = container.parent / (
            f".{container.name}.{sha256(str(directory).encode()).hexdigest()}.tmp"
        )
        legacy_pack_staging.write_bytes(b"unowned pack data")

        pack_agent_bundle(directory, container)

        assert legacy_pack_staging.read_bytes() == b"unowned pack data"
        unpacked = (tmp_path / "owned-staging.cayu.d").resolve()
        legacy_unpack_staging = unpacked.parent / (
            f".{unpacked.name}.{sha256(unpacked.name.encode()).hexdigest()}.tmp"
        )
        legacy_unpack_staging.mkdir()
        marker = legacy_unpack_staging / "marker"
        marker.write_bytes(b"unowned unpack data")

        unpack_agent_bundle_container(container, unpacked)

        assert marker.read_bytes() == b"unowned unpack data"

    _run(scenario())


def test_pack_rejects_staging_substitution_after_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        *_, directory, _export = await _export_directory(tmp_path)
        destination = (tmp_path / "substitution.cayu").resolve()
        publish = container_module._publish_file

        def substitute_staging(
            source: Path,
            target: Path,
            *,
            expected_identity: tuple[int, int, int, int],
        ) -> None:
            source.unlink()
            source.write_bytes(b"attacker-controlled replacement")
            publish(source, target, expected_identity=expected_identity)

        monkeypatch.setattr(container_module, "_publish_file", substitute_staging)

        with pytest.raises(AgentBundleError, match="container_staging_changed"):
            pack_agent_bundle(directory, destination)
        assert not destination.exists()

    _run(scenario())


def test_path_inspection_rejects_same_inode_change_with_restored_mtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        *_, first_directory, _first_export = await _export_directory(
            tmp_path / "first",
            body_content=b"def answer():\n    return 41\n",
        )
        *_, second_directory, _second_export = await _export_directory(
            tmp_path / "second",
            body_content=b"def answer():\n    return 42\n",
        )
        first = (tmp_path / "first.cayu").resolve()
        second = (tmp_path / "second.cayu").resolve()
        pack_agent_bundle(first_directory, first)
        pack_agent_bundle(second_directory, second)
        first_bytes = first.read_bytes()
        second_bytes = second.read_bytes()
        assert len(first_bytes) == len(second_bytes)

        target = (tmp_path / "mutable.cayu").resolve()
        target.write_bytes(first_bytes)
        initial_stat = target.stat()
        real_zip_file = container_module.zipfile.ZipFile
        changed = False

        def change_source_then_open(*args, **kwargs):
            nonlocal changed
            if not changed:
                changed = True
                target.write_bytes(second_bytes)
                os.utime(
                    target,
                    ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
                )
            return real_zip_file(*args, **kwargs)

        monkeypatch.setattr(container_module.zipfile, "ZipFile", change_source_then_open)

        with pytest.raises(AgentBundleError, match="container_source_changed"):
            container_module._read_container(target)

    _run(scenario())


def test_coordinator_stream_export_retries_short_writes_before_acknowledging(
    tmp_path: Path,
) -> None:
    class ShortWriteStream(io.BytesIO):
        def write(self, data: bytes) -> int:  # ty: ignore[invalid-method-override]
            accepted = max(1, len(data) // 2)
            return super().write(data[:accepted])

    async def scenario() -> None:
        (
            coordinator,
            snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path)
        destination = ShortWriteStream()
        receipt = await coordinator.export_container(
            operation_id="container-short-write-export",
            access=access,
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=destination,
        )

        payload = destination.getvalue()
        inspection = inspect_agent_bundle_container(io.BytesIO(payload))
        assert len(payload) == receipt.inspection.container_bytes
        assert inspection == receipt.inspection
        assert not snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


@pytest.mark.parametrize("accepted", (None, 0, -1, 2_000_000))
def test_coordinator_stream_export_rejects_invalid_write_counts_and_retains_protection(
    tmp_path: Path,
    accepted: int | None,
) -> None:
    class InvalidWriteStream(io.BytesIO):
        def write(self, _data: bytes) -> int | None:  # ty: ignore[invalid-method-override]
            return accepted

    async def scenario() -> None:
        (
            coordinator,
            snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path / str(accepted))
        with pytest.raises(AgentBundleError, match="container_stream_write_invalid"):
            await coordinator.export_container(
                operation_id=f"container-invalid-write-{accepted}",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=InvalidWriteStream(),
            )
        assert snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_coordinator_stream_export_rolls_back_partial_write_before_exact_retry(
    tmp_path: Path,
) -> None:
    class FailOnceAfterPartialWrite(io.BytesIO):
        failed = False

        def write(self, data: bytes) -> int:  # ty: ignore[invalid-method-override]
            if not self.failed:
                self.failed = True
                super().write(data[:100])
                return len(data) + 1
            return super().write(data)

    async def scenario() -> None:
        (
            coordinator,
            snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path)
        destination = FailOnceAfterPartialWrite()

        with pytest.raises(AgentBundleError, match="container_stream_write_invalid"):
            await coordinator.export_container(
                operation_id="container-partial-write-retry",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )

        assert destination.getvalue() == b""
        assert snapshots._root_is_protected(snapshot.snapshot_root)

        receipt = await coordinator.export_container(
            operation_id="container-partial-write-retry",
            access=access,
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=destination,
        )
        payload = destination.getvalue()
        assert len(payload) == receipt.inspection.container_bytes
        assert inspect_agent_bundle_container(io.BytesIO(payload)) == receipt.inspection
        assert not snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_cancelled_stream_export_settles_and_resets_before_retry(
    tmp_path: Path,
) -> None:
    class BlockFirstWriteStream(io.BytesIO):
        def __init__(self) -> None:
            super().__init__()
            self.claimed = False
            self.blocked = threading.Event()
            self.release = threading.Event()

        def write(self, data: bytes) -> int:  # ty: ignore[invalid-method-override]
            if not self.claimed:
                self.claimed = True
                self.blocked.set()
                if not self.release.wait(timeout=10):
                    raise RuntimeError("timed out waiting to release stream writer")
            return super().write(data)

    async def scenario() -> None:
        (
            coordinator,
            snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path)
        destination = BlockFirstWriteStream()
        first = asyncio.create_task(
            coordinator.export_container(
                operation_id="container-cancelled-stream-retry",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )
        )
        assert await asyncio.to_thread(destination.blocked.wait, 5)

        first.cancel("stop container export")
        await asyncio.sleep(0.05)
        assert not first.done()
        assert snapshots._root_is_protected(snapshot.snapshot_root)

        destination.release.set()
        with pytest.raises(asyncio.CancelledError, match="stop container export"):
            await first

        assert destination.getvalue() == b""
        assert snapshots._root_is_protected(snapshot.snapshot_root)

        receipt = await coordinator.export_container(
            operation_id="container-cancelled-stream-retry",
            access=access,
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=destination,
        )
        payload = destination.getvalue()
        assert len(payload) == receipt.inspection.container_bytes
        assert inspect_agent_bundle_container(io.BytesIO(payload)) == receipt.inspection
        assert not snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_committed_protection_release_wins_racing_stream_cancellation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        database = (tmp_path / "snapshots.db").resolve()
        snapshots = SQLiteAgentSnapshotStore(database)
        (
            coordinator,
            _snapshot_store,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(
            tmp_path,
            snapshot_store=snapshots,  # ty: ignore[invalid-argument-type]
        )
        release_entered = threading.Event()
        release_continue = threading.Event()
        original_release = snapshots._release_snapshot_protection_sync

        def blocking_release(*args, **kwargs):
            release_entered.set()
            if not release_continue.wait(timeout=10):
                raise RuntimeError("timed out waiting to release snapshot protection")
            return original_release(*args, **kwargs)

        monkeypatch.setattr(
            snapshots,
            "_release_snapshot_protection_sync",
            blocking_release,
        )
        destination = io.BytesIO()
        export = asyncio.create_task(
            coordinator.export_container(
                operation_id="container-cancelled-protection-release",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )
        )
        assert await asyncio.to_thread(release_entered.wait, 5)
        committed_payload = destination.getvalue()
        assert inspect_agent_bundle_container(io.BytesIO(committed_payload)).snapshot_root == (
            snapshot.snapshot_root
        )

        export.cancel("cancel during protection release")
        await asyncio.sleep(0.05)
        assert not export.done()

        release_continue.set()
        receipt = await export
        assert destination.getvalue() == committed_payload
        assert receipt.inspection.container_bytes == len(committed_payload)

        with sqlite3.connect(database) as connection:
            active = connection.execute(
                "SELECT 1 FROM cayu_agent_snapshot_protections "
                "WHERE snapshot_root = ? AND released = 0 LIMIT 1",
                (snapshot.snapshot_root,),
            ).fetchone()
        assert active is None

    _run(scenario())


def test_stream_export_release_failure_resets_before_exact_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        (
            coordinator,
            snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path)
        original_release = snapshots.release_snapshot_protection
        release_attempts = 0

        async def fail_once(**kwargs):
            nonlocal release_attempts
            release_attempts += 1
            if release_attempts == 1:
                raise RuntimeError("simulated protection release failure")
            return await original_release(**kwargs)

        monkeypatch.setattr(snapshots, "release_snapshot_protection", fail_once)
        destination = io.BytesIO()
        with pytest.raises(RuntimeError, match="simulated protection release failure"):
            await coordinator.export_container(
                operation_id="container-release-failure-retry",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )

        assert destination.getvalue() == b""
        assert snapshots._root_is_protected(snapshot.snapshot_root)

        receipt = await coordinator.export_container(
            operation_id="container-release-failure-retry",
            access=access,
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=destination,
        )
        payload = destination.getvalue()
        assert len(payload) == receipt.inspection.container_bytes
        assert inspect_agent_bundle_container(io.BytesIO(payload)) == receipt.inspection
        assert not snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_stream_export_rejects_nontransactional_destination_before_writing(
    tmp_path: Path,
) -> None:
    class WriteOnlyStream:
        writes = 0

        def write(self, data: bytes) -> int:
            self.writes += 1
            return len(data)

    async def scenario() -> None:
        (
            coordinator,
            snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path)
        destination = WriteOnlyStream()

        with pytest.raises(
            AgentBundleError,
            match="container_stream_transaction_unsupported",
        ):
            await coordinator.export_container(
                operation_id="container-nontransactional-stream",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,  # type: ignore[arg-type]
            )

        assert destination.writes == 0
        assert snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_stream_export_rejects_nonempty_destination_without_mutating_it(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        (
            coordinator,
            snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path)
        destination = io.BytesIO(b"existing destination bytes")
        destination.seek(0)

        with pytest.raises(
            AgentBundleError,
            match="container_stream_destination_not_empty",
        ):
            await coordinator.export_container(
                operation_id="container-nonempty-stream",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )

        assert destination.getvalue() == b"existing destination bytes"
        assert snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_full_container_imports_identical_root_in_fresh_stores(tmp_path: Path) -> None:
    async def scenario() -> None:
        (
            coordinator,
            _source_snapshots,
            _source_objects,
            subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path)
        container = (tmp_path / "portable-agent.cayu").resolve()
        exported = await coordinator.export_container(
            operation_id="container-export",
            access=access,
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=container,
        )
        destination_snapshots = InMemoryAgentSnapshotStore()
        destination = AgentBundleCoordinator(
            snapshot_store=destination_snapshots,
            object_store=FileSystemAgentSnapshotObjectStore(
                (tmp_path / "destination-objects").resolve()
            ),
        )
        imported = await destination.import_container(
            operation_id="container-import",
            source=container,
            subject=subject.model_copy(update={"agent_id": "container-imported"}),
            authority_scope_fingerprint=_digest("container-destination-scope"),
            owner="container-test",
        )

        assert exported.bundle.snapshot_ref == snapshot.ref
        assert imported.snapshot_ref == snapshot.ref
        loaded = await destination_snapshots.load_snapshot(snapshot.snapshot_root)
        assert loaded is not None
        assert loaded.subject.agent_id == "container-imported"

    _run(scenario())


def test_thin_container_names_dependency_and_requires_destination_inventory(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        *_, full_export = await _export_directory(tmp_path / "full")
        inventory = AgentBundleInventory(
            object_digests=tuple(reference.digest for reference in full_export.bundle.closure[:-1])
        )
        *_, thin_directory, thin_export = await _export_directory(
            tmp_path / "thin",
            mode=AgentBundleMode.THIN,
            destination_inventory=inventory,
        )
        container = (tmp_path / "thin.cayu").resolve()
        receipt = pack_agent_bundle(thin_directory, container)

        assert receipt.bundle == thin_export.bundle
        assert receipt.inspection.mode is AgentBundleMode.THIN
        assert receipt.inspection.requires_preexisting_objects is True
        assert receipt.inspection.destination_inventory_fingerprint == inventory.fingerprint
        assert receipt.inspection.transferred_object_count == len(
            thin_export.bundle.transferred_digests
        )

    _run(scenario())


def test_thin_container_import_requires_the_exact_preexisting_objects(tmp_path: Path) -> None:
    async def scenario() -> None:
        (
            _full_coordinator,
            _full_snapshots,
            _full_objects,
            subject,
            _scope,
            snapshot,
            _full_access,
            full_directory,
            full_export,
        ) = await _export_directory(tmp_path / "full-source")
        full_container = (tmp_path / "full.cayu").resolve()
        pack_agent_bundle(full_directory, full_container)
        inventory = AgentBundleInventory(
            object_digests=tuple(reference.digest for reference in full_export.bundle.closure)
        )
        *_, thin_directory, thin_export = await _export_directory(
            tmp_path / "thin-source",
            mode=AgentBundleMode.THIN,
            destination_inventory=inventory,
        )
        thin_container = (tmp_path / "thin.cayu").resolve()
        pack_agent_bundle(thin_directory, thin_container)
        assert thin_export.bundle.transferred_digests == (
            thin_export.bundle.snapshot_document.digest,
        )

        empty_destination = AgentBundleCoordinator(
            snapshot_store=InMemoryAgentSnapshotStore(),
            object_store=FileSystemAgentSnapshotObjectStore(
                (tmp_path / "empty-destination-objects").resolve()
            ),
        )
        with pytest.raises(AgentBundleError, match="thin_bundle_object_unavailable"):
            await empty_destination.import_container(
                operation_id="thin-container-missing-import",
                source=thin_container,
                subject=subject,
                authority_scope_fingerprint=_digest("thin-container-destination-scope"),
                owner="thin-container-test",
            )

        destination_snapshots = InMemoryAgentSnapshotStore()
        populated_destination = AgentBundleCoordinator(
            snapshot_store=destination_snapshots,
            object_store=FileSystemAgentSnapshotObjectStore(
                (tmp_path / "populated-destination-objects").resolve()
            ),
        )
        await populated_destination.import_container(
            operation_id="full-container-prime-import",
            source=full_container,
            subject=subject,
            authority_scope_fingerprint=_digest("thin-container-destination-scope"),
            owner="thin-container-test",
        )
        imported = await populated_destination.import_container(
            operation_id="thin-container-import",
            source=thin_container,
            subject=subject,
            authority_scope_fingerprint=_digest("thin-container-destination-scope"),
            owner="thin-container-test",
        )
        assert imported.snapshot_ref == snapshot.ref
        assert imported.reused_digests

    _run(scenario())


def test_stream_api_and_forced_zip64_round_trip(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        *_, directory, export = await _export_directory(
            tmp_path,
            body_content=b"x" * (2 * 1024 * 1024 + 17),
        )
        monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1)
        output = io.BytesIO()
        packed = pack_agent_bundle(directory, output)
        source = io.BytesIO(output.getvalue())
        inspection = inspect_agent_bundle_container(source)
        unpacked = (tmp_path / "zip64-unpacked").resolve()
        receipt = unpack_agent_bundle_container(io.BytesIO(output.getvalue()), unpacked)

        assert inspection == packed.inspection == receipt.inspection
        assert receipt.bundle == export.bundle

    _run(scenario())


@pytest.mark.parametrize(
    "entry_count", (zipfile.ZIP_FILECOUNT_LIMIT, zipfile.ZIP_FILECOUNT_LIMIT + 1)
)
def test_zip64_end_record_entry_count_boundary(entry_count: int) -> None:
    class EntryList:
        def __len__(self) -> int:
            return entry_count

    class Archive:
        start_dir = 0

        def infolist(self) -> EntryList:
            return EntryList()

    ordinary_count = min(entry_count, 0xFFFF)
    ordinary_end = container_module._ZIP_END_RECORD.pack(
        container_module._ZIP_END_SIGNATURE,
        0,
        0,
        ordinary_count,
        ordinary_count,
        0,
        0,
        0,
    )
    if entry_count == zipfile.ZIP_FILECOUNT_LIMIT:
        payload = ordinary_end
    else:
        zip64_end = container_module._ZIP64_END_RECORD.pack(
            container_module._ZIP64_END_SIGNATURE,
            44,
            45,
            45,
            0,
            0,
            entry_count,
            entry_count,
            0,
            0,
        )
        locator = container_module._ZIP64_LOCATOR.pack(
            container_module._ZIP64_LOCATOR_SIGNATURE,
            0,
            0,
            1,
        )
        payload = zip64_end + locator + ordinary_end

    assert (
        container_module._validate_end_records(
            io.BytesIO(payload),
            Archive(),  # ty: ignore[invalid-argument-type]
            len(payload),
        )
        == 0
    )


def _rewrite_archive(
    source: Path,
    destination: Path,
    transform,
) -> None:
    with zipfile.ZipFile(source) as archive:
        entries = [(info, archive.read(info)) for info in archive.infolist()]
    transformed = transform(entries)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
            for info, content in transformed:
                archive.writestr(info, content)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("compressed", "container_compression_unsupported"),
        ("mimetype_second", "container_schema_unsupported"),
        ("missing_object", "container_entry_set_or_order_mismatch"),
        ("extra_object", "container_entry_not_regular"),
        ("traversal", "container_entry_name_invalid"),
        ("symlink", "container_entry_not_regular"),
        ("duplicate", "container_entry_overlap_or_gap"),
    ),
)
def test_hostile_container_entries_fail_closed(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    async def scenario() -> None:
        *_, directory, _export = await _export_directory(tmp_path / mutation)
        source = (tmp_path / f"{mutation}-source.cayu").resolve()
        pack_agent_bundle(directory, source)
        hostile = (tmp_path / f"{mutation}.cayu").resolve()

        def transform(entries):
            copied = list(entries)
            if mutation == "compressed":
                copied[2][0].compress_type = zipfile.ZIP_DEFLATED
            elif mutation == "mimetype_second":
                copied[0], copied[1] = copied[1], copied[0]
            elif mutation == "missing_object":
                copied.pop()
            elif mutation == "extra_object":
                info = zipfile.ZipInfo("objects/00/" + "0" * 62)
                copied.append((info, b"extra"))
            elif mutation == "traversal":
                copied[2][0].filename = "../object"
            elif mutation == "symlink":
                copied[2][0].external_attr = (stat.S_IFLNK | 0o777) << 16
            elif mutation == "duplicate":
                copied.append(copied[-1])
            return copied

        _rewrite_archive(source, hostile, transform)
        with pytest.raises(AgentBundleError, match=error):
            inspect_agent_bundle_container(hostile)

    _run(scenario())


def test_trailing_and_sha_invalid_containers_fail_before_destination_publication(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        *_, directory, _export = await _export_directory(tmp_path)
        valid = (tmp_path / "valid.cayu").resolve()
        pack_agent_bundle(directory, valid)

        trailing = (tmp_path / "trailing.cayu").resolve()
        trailing.write_bytes(valid.read_bytes() + b"unexpected")
        with pytest.raises(AgentBundleError, match="container_trailing_data"):
            inspect_agent_bundle_container(trailing)

        corrupted = bytearray(valid.read_bytes())
        with zipfile.ZipFile(valid) as archive:
            object_info = archive.infolist()[2]
            offset = (
                object_info.header_offset + 30 + len(object_info.filename) + len(object_info.extra)
            )
        corrupted[offset] ^= 1
        corrupt_path = (tmp_path / "corrupt.cayu").resolve()
        corrupt_path.write_bytes(corrupted)
        destination = (tmp_path / "must-not-exist").resolve()
        with pytest.raises(AgentBundleError):
            unpack_agent_bundle_container(corrupt_path, destination)
        assert not destination.exists()

    _run(scenario())


def test_redundant_zip64_end_records_are_not_canonical_exact_retries(tmp_path: Path) -> None:
    async def scenario() -> None:
        *_, directory, _export = await _export_directory(tmp_path)
        canonical = (tmp_path / "canonical.cayu").resolve()
        pack_agent_bundle(directory, canonical)
        canonical_bytes = canonical.read_bytes()
        end_offset = len(canonical_bytes) - container_module._ZIP_END_RECORD.size
        (
            signature,
            disk_number,
            central_disk,
            entries_on_disk,
            entry_count,
            central_size,
            central_offset,
            comment_size,
        ) = container_module._ZIP_END_RECORD.unpack_from(canonical_bytes, end_offset)
        assert signature == container_module._ZIP_END_SIGNATURE
        assert (disk_number, central_disk, comment_size) == (0, 0, 0)
        zip64_end = container_module._ZIP64_END_RECORD.pack(
            container_module._ZIP64_END_SIGNATURE,
            44,
            45,
            45,
            0,
            0,
            entries_on_disk,
            entry_count,
            central_size,
            central_offset,
        )
        locator = container_module._ZIP64_LOCATOR.pack(
            container_module._ZIP64_LOCATOR_SIGNATURE,
            0,
            end_offset,
            1,
        )
        noncanonical = (tmp_path / "redundant-zip64.cayu").resolve()
        noncanonical.write_bytes(
            canonical_bytes[:end_offset] + zip64_end + locator + canonical_bytes[end_offset:]
        )

        with pytest.raises(AgentBundleError, match="container_zip64_not_canonical"):
            inspect_agent_bundle_container(noncanonical)
        with pytest.raises(AgentBundleError, match="container_zip64_not_canonical"):
            pack_agent_bundle(directory, noncanonical)

    _run(scenario())


def test_crc_valid_sha_invalid_and_noncanonical_index_fail_closed(tmp_path: Path) -> None:
    async def scenario() -> None:
        *_, directory, _export = await _export_directory(tmp_path)
        valid = (tmp_path / "valid-rewrite.cayu").resolve()
        pack_agent_bundle(directory, valid)

        sha_invalid = (tmp_path / "sha-invalid.cayu").resolve()

        def alter_object(entries):
            copied = list(entries)
            info, content = copied[-1]
            copied[-1] = (info, content[:-1] + bytes([content[-1] ^ 1]))
            return copied

        _rewrite_archive(valid, sha_invalid, alter_object)
        with pytest.raises(
            AgentBundleError,
            match="container_(snapshot_node_invalid|object_integrity_mismatch)",
        ):
            inspect_agent_bundle_container(sha_invalid)

        noncanonical = (tmp_path / "noncanonical-index.cayu").resolve()

        def alter_index(entries):
            copied = list(entries)
            info, content = copied[1]
            copied[1] = (info, content[:-1] + b" \n}")
            return copied

        _rewrite_archive(valid, noncanonical, alter_index)
        with pytest.raises(AgentBundleError, match="bundle_index_not_canonical"):
            inspect_agent_bundle_container(noncanonical)

    _run(scenario())


def test_local_header_size_and_unknown_extra_fail_before_unpack(tmp_path: Path) -> None:
    async def scenario() -> None:
        *_, directory, _export = await _export_directory(tmp_path)
        valid = (tmp_path / "valid-raw.cayu").resolve()
        pack_agent_bundle(directory, valid)

        with zipfile.ZipFile(valid) as archive:
            object_info = archive.infolist()[2]
        mismatched = bytearray(valid.read_bytes())
        declared_size_offset = object_info.header_offset + 22
        declared_size = int.from_bytes(
            mismatched[declared_size_offset : declared_size_offset + 4], "little"
        )
        mismatched[declared_size_offset : declared_size_offset + 4] = (declared_size + 1).to_bytes(
            4, "little"
        )
        mismatch_path = (tmp_path / "local-size-mismatch.cayu").resolve()
        mismatch_path.write_bytes(mismatched)
        destination = (tmp_path / "raw-mutation-destination").resolve()
        with pytest.raises(AgentBundleError):
            unpack_agent_bundle_container(mismatch_path, destination)
        assert not destination.exists()

        unknown_extra = (tmp_path / "unknown-extra.cayu").resolve()

        def add_unknown_extra(entries):
            copied = list(entries)
            copied[2][0].extra += b"\x34\x12\x01\x00x"
            return copied

        _rewrite_archive(valid, unknown_extra, add_unknown_extra)
        with pytest.raises(AgentBundleError, match="container_extra_field_unsupported"):
            inspect_agent_bundle_container(unknown_extra)

    _run(scenario())


def test_concurrent_container_export_and_import_retries_converge(tmp_path: Path) -> None:
    async def scenario() -> None:
        (
            coordinator,
            source_snapshots,
            _source_objects,
            subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path)
        destination = (tmp_path / "convergent.cayu").resolve()

        async def export_once():
            return await coordinator.export_container(
                operation_id="convergent-container-export",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )

        first, second = await asyncio.gather(export_once(), export_once())
        retry = await export_once()
        assert first == second == retry
        assert first.inspection.transport_sha256 == sha256(destination.read_bytes()).hexdigest()
        assert not source_snapshots._root_is_protected(snapshot.snapshot_root)

        destination_snapshots = InMemoryAgentSnapshotStore()
        importer = AgentBundleCoordinator(
            snapshot_store=destination_snapshots,
            object_store=FileSystemAgentSnapshotObjectStore(
                (tmp_path / "convergent-destination-objects").resolve()
            ),
        )

        async def import_once():
            return await importer.import_container(
                operation_id="convergent-container-import",
                source=destination,
                subject=subject,
                authority_scope_fingerprint=_digest("convergent-container-scope"),
                owner="container-convergence-test",
            )

        imported_first, imported_second = await asyncio.gather(import_once(), import_once())
        imported_retry = await import_once()
        assert imported_first == imported_second == imported_retry
        assert imported_first.snapshot_ref == snapshot.ref

    _run(scenario())


def test_concurrent_exact_container_exports_do_not_share_one_released_protection(
    tmp_path: Path,
) -> None:
    class ProtectionIntervalStore(InMemoryAgentSnapshotStore):
        active_exports = 0
        released_while_export_active = False
        replayed_after_shared_protection_release = False
        gc_deleted_roots: tuple[str, ...] = ()

        async def protect_snapshot(
            self,
            protection: AgentSnapshotProtection,
        ) -> AgentSnapshotProtection:
            protected = await super().protect_snapshot(protection)
            if protection.reason == "bundle-container-export-in-progress":
                self.active_exports += 1
                if not self._root_is_protected(protection.access.snapshot.snapshot_root):
                    self.replayed_after_shared_protection_release = True
            return protected

        async def release_snapshot_protection(
            self,
            *,
            operation_id: str,
            access: AgentSnapshotAccess,
            protection_id: str,
        ) -> AgentSnapshotProtection:
            released = await super().release_snapshot_protection(
                operation_id=operation_id,
                access=access,
                protection_id=protection_id,
            )
            if released.reason == "bundle-container-export-in-progress":
                self.active_exports -= 1
                if self.active_exports and not self._root_is_protected(
                    access.snapshot.snapshot_root
                ):
                    self.released_while_export_active = True
                if not self.gc_deleted_roots:
                    plan = await self.plan_snapshot_gc(
                        AgentSnapshotGCRequest(
                            operation_id="gc-between-concurrent-exact-exports",
                            candidates=(access,),
                        )
                    )
                    receipt = await self.execute_snapshot_gc(plan)
                    self.gc_deleted_roots = receipt.deleted_roots
            return released

    async def scenario() -> None:
        snapshots = ProtectionIntervalStore()
        (
            coordinator,
            _snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path, snapshot_store=snapshots)
        destination = (tmp_path / "serialized-exact-export.cayu").resolve()

        async def export_once():
            return await coordinator.export_container(
                operation_id="serialized-exact-container-export",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )

        first, second = await asyncio.gather(export_once(), export_once())

        assert first == second
        assert snapshots.active_exports == 0
        assert snapshots.released_while_export_active is False
        assert snapshots.replayed_after_shared_protection_release is True
        assert snapshots.gc_deleted_roots == (snapshot.snapshot_root,)
        assert not snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_container_export_lost_ack_retry_releases_original_root_protection(
    tmp_path: Path,
) -> None:
    class CrashBeforeReleaseStore(InMemoryAgentSnapshotStore):
        fail_release = True

        async def release_snapshot_protection(
            self,
            *,
            operation_id: str,
            access: AgentSnapshotAccess,
            protection_id: str,
        ) -> AgentSnapshotProtection:
            if self.fail_release:
                self.fail_release = False
                raise RuntimeError("simulated container lost acknowledgement")
            return await super().release_snapshot_protection(
                operation_id=operation_id,
                access=access,
                protection_id=protection_id,
            )

    async def scenario() -> None:
        providers, object_store, subject, scope = _portable_fixture(tmp_path / "lost-ack")
        snapshot_store = CrashBeforeReleaseStore()
        snapshot = await AgentSnapshotCoordinator(providers, store=snapshot_store).capture(
            AgentSnapshotCaptureRequest(
                capture_request_id="container-lost-ack-capture",
                subject=subject,
                authority_scope_fingerprint=scope,
                components=tuple(
                    AgentSnapshotComponentSelector(kind=kind)
                    for kind in sorted((provider.kind for provider in providers), key=str)
                ),
            )
        )
        access = AgentSnapshotAccess(
            snapshot=snapshot.ref,
            binding_id=snapshot.identity_binding.binding_id,
            authority_scope_fingerprint=scope,
        )
        coordinator = AgentBundleCoordinator(
            snapshot_store=snapshot_store,
            object_store=object_store,
        )
        destination = (tmp_path / "lost-ack.cayu").resolve()
        with pytest.raises(RuntimeError, match="simulated container lost acknowledgement"):
            await coordinator.export_container(
                operation_id="container-lost-ack-export",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )
        assert destination.is_file()
        assert snapshot_store._root_is_protected(snapshot.snapshot_root)

        canonical_bytes = destination.read_bytes()
        changed_metadata = bytearray(canonical_bytes)
        with zipfile.ZipFile(io.BytesIO(canonical_bytes)) as archive:
            central_offset = archive.start_dir
        changed_metadata[central_offset + 4 : central_offset + 6] = ((3 << 8) | 45).to_bytes(
            2,
            "little",
        )
        destination.write_bytes(changed_metadata)
        with pytest.raises(AgentBundleError, match="container_central_directory_mismatch"):
            await coordinator.export_container(
                operation_id="container-lost-ack-export",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )
        assert snapshot_store._root_is_protected(snapshot.snapshot_root)

        destination.write_bytes(canonical_bytes)

        recovered = await coordinator.export_container(
            operation_id="container-lost-ack-export",
            access=access,
            profile=AgentSnapshotProfile.REUSABLE_AGENT,
            destination=destination,
        )
        assert recovered.bundle.snapshot_ref == snapshot.ref
        assert not snapshot_store._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_container_export_retry_rejects_changed_valid_bundle_and_retains_protection(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        (
            coordinator,
            snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            directory,
            exported,
        ) = await _export_directory(tmp_path)
        original = exported.bundle
        changed_size_report = AgentBundleSizeReport(
            **{
                **original.size_report.model_dump(mode="python"),
                "materialized_disk_bytes": (original.size_report.materialized_disk_bytes + 1),
            }
        )
        changed = AgentBundle.create(
            snapshot_ref=original.snapshot_ref,
            export_binding_id=original.export_binding_id,
            export_authority_scope_fingerprint=(original.export_authority_scope_fingerprint),
            destination_inventory_fingerprint=(original.destination_inventory_fingerprint),
            profile=original.profile,
            mode=original.mode,
            snapshot_document=original.snapshot_document,
            closure=original.closure,
            transferred_digests=original.transferred_digests,
            external_bindings=original.external_bindings,
            size_report=changed_size_report,
        )
        assert changed.bundle_id != original.bundle_id
        (directory / AGENT_BUNDLE_INDEX_FILENAME).write_bytes(
            _canonical_json(changed, "changed_agent_bundle")
        )
        destination = (tmp_path / "changed-existing.cayu").resolve()
        packed_changed = pack_agent_bundle(directory, destination)
        assert packed_changed.bundle == changed

        with pytest.raises(AgentBundleError, match="container_destination_conflict"):
            await coordinator.export_container(
                operation_id="changed-existing-container-export",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )

        assert inspect_agent_bundle_container(destination).bundle_id == changed.bundle_id
        assert snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_container_publication_failure_leaves_no_final_file_and_retains_protection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        (
            coordinator,
            snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path)
        destination = (tmp_path / "must-not-publish.cayu").resolve()

        def fail_publication(
            _source: Path,
            _destination: Path,
            *,
            expected_identity: tuple[int, int, int, int],
        ) -> None:
            del expected_identity
            raise RuntimeError("simulated final publication failure")

        monkeypatch.setattr(
            "cayu.agent_bundle_containers._publish_file",
            fail_publication,
        )
        with pytest.raises(RuntimeError, match="simulated final publication failure"):
            await coordinator.export_container(
                operation_id="container-publication-failure",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )
        assert not destination.exists()
        assert snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())


def test_cancelled_container_export_retains_protection_without_a_partial_final_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def scenario() -> None:
        (
            coordinator,
            snapshots,
            _objects,
            _subject,
            _scope,
            snapshot,
            access,
            _directory,
            _export,
        ) = await _export_directory(tmp_path)
        destination = (tmp_path / "cancelled.cayu").resolve()

        def cancel_pack(*_args, **_kwargs):
            raise asyncio.CancelledError

        monkeypatch.setattr("cayu.agent_bundle_containers.pack_agent_bundle", cancel_pack)
        with pytest.raises(asyncio.CancelledError):
            await coordinator.export_container(
                operation_id="cancelled-container-export",
                access=access,
                profile=AgentSnapshotProfile.REUSABLE_AGENT,
                destination=destination,
            )
        assert not destination.exists()
        assert snapshots._root_is_protected(snapshot.snapshot_root)

    _run(scenario())
