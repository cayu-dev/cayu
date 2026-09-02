from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cayu.cli import _guarded_tree_publication as publication
from cayu.cli._guarded_tree_publication import (
    DestinationPolicy,
    GuardedTreePublicationError,
    GuardedTreeStage,
    publish_guarded_tree,
    recover_guarded_tree,
    validate_guarded_tree_files,
)

_REQUEST_DIGEST = f"sha256:{'1' * 64}"


def _populate(staging: GuardedTreeStage, *, content: str = "new\n") -> None:
    staging.write_text("nested/value.txt", content)


@pytest.mark.skipif(os.name == "nt", reason="requires a POSIX descriptor")
def test_linux_incarnation_changes_when_inode_generation_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("value", encoding="utf-8")
    descriptor = os.open(target, os.O_RDONLY)
    generations = iter((17, 18))
    monkeypatch.setattr(publication, "_linux_birth_time_ns", lambda *args, **kwargs: 42)
    monkeypatch.setattr(
        publication,
        "_linux_inode_generation",
        lambda *args, **kwargs: next(generations),
    )
    try:
        value = os.fstat(descriptor)
        first = publication._linux_incarnation(
            value,
            path=None,
            descriptor=descriptor,
            dir_fd=None,
            name=None,
        )
        second = publication._linux_incarnation(
            value,
            path=None,
            descriptor=descriptor,
            dir_fd=None,
            name=None,
        )
    finally:
        os.close(descriptor)

    assert first != second
    assert first & ((1 << 128) - 1) == second & ((1 << 128) - 1) == 42


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX fcntl")
def test_linux_zero_inode_generation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fcntl

    monkeypatch.setattr(fcntl, "ioctl", lambda *args, **kwargs: 0)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publication._linux_inode_generation(0, path=None)

    assert exc_info.value.code == "stable_identity_unavailable"


def test_identity_pin_translates_namespace_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("value", encoding="utf-8")
    expected = target.stat(follow_symlinks=False)

    def replaced_open(*_args: object, **_kwargs: object) -> int:
        raise OSError(errno.ELOOP, "namespace entry became a link")

    monkeypatch.setattr(publication.os, "open", replaced_open)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publication._pin_identity_descriptor(
            expected,
            path=target,
            descriptor=None,
            dir_fd=None,
            name=None,
        )

    assert exc_info.value.code == "identity_changed"
    assert isinstance(exc_info.value.__cause__, OSError)


def test_windows_incarnation_distinguishes_recycled_file_id_with_object_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("value", encoding="utf-8")
    expected = target.stat(follow_symlinks=False)
    object_ids = iter((bytes.fromhex("11" * 16), bytes.fromhex("22" * 16)))

    def create_file(*_args: object) -> int:
        return 101

    def close_handle(handle: int) -> int:
        assert handle == 101
        return 1

    def get_file_id(
        handle: int,
        info_class: int,
        output: Any,
        output_size: int,
    ) -> int:
        assert handle == 101
        assert info_class == publication._WINDOWS_FILE_ID_INFO_CLASS
        assert output_size == ctypes.sizeof(publication._WindowsFileIdInfo)
        info = ctypes.cast(
            output,
            ctypes.POINTER(publication._WindowsFileIdInfo),
        ).contents
        info.volume_serial_number = 19
        for index in range(16):
            info.file_id[index] = 0xAB
        return 1

    def create_or_get_object_id(
        handle: int,
        control_code: int,
        _input: object,
        input_size: int,
        output: Any,
        output_size: int,
        returned: Any,
        _overlapped: object,
    ) -> int:
        assert handle == 101
        assert control_code == publication._WINDOWS_FSCTL_CREATE_OR_GET_OBJECT_ID
        assert input_size == 0
        assert output_size == ctypes.sizeof(publication._WindowsObjectIdBuffer)
        result = ctypes.cast(
            output,
            ctypes.POINTER(publication._WindowsObjectIdBuffer),
        ).contents
        for index, byte in enumerate(next(object_ids)):
            result.object_id[index] = byte
        ctypes.cast(returned, ctypes.POINTER(ctypes.c_uint32)).contents.value = output_size
        return 1

    monkeypatch.setattr(
        publication,
        "_windows_file_id_bindings",
        lambda: (object(), create_file, close_handle, get_file_id),
    )
    monkeypatch.setattr(
        publication,
        "_windows_object_id_binding",
        lambda: create_or_get_object_id,
    )

    first = publication._windows_incarnation(
        expected,
        path=target,
        descriptor=None,
        dir_fd=None,
        name=None,
    )
    second = publication._windows_incarnation(
        expected,
        path=target,
        descriptor=None,
        dir_fd=None,
        name=None,
    )

    assert first != second
    assert first >> 128 == second >> 128 == 19
    assert first & ((1 << 128) - 1) == int.from_bytes(bytes.fromhex("11" * 16), "little")
    assert second & ((1 << 128) - 1) == int.from_bytes(bytes.fromhex("22" * 16), "little")


def test_windows_object_identity_fails_closed_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publication, "_windows_object_id_binding", lambda: lambda *_args: 0)
    monkeypatch.setattr(publication, "_windows_last_error", lambda: errno.ENOTSUP)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publication._windows_object_id(101, path=Path("target"))

    assert exc_info.value.code == "stable_identity_unavailable"
    assert isinstance(exc_info.value.__cause__, OSError)


def test_windows_native_bindings_enable_last_error_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def create_file(*_args: object) -> int:
        return 101

    def close_handle(_handle: int) -> int:
        return 1

    def get_file_information(*_args: object) -> int:
        return 1

    def device_io_control(*_args: object) -> int:
        return 1

    kernel32 = SimpleNamespace(
        CreateFileW=create_file,
        CloseHandle=close_handle,
        GetFileInformationByHandleEx=get_file_information,
        DeviceIoControl=device_io_control,
    )

    def load_windows_dll(name: str, *, use_last_error: bool) -> object:
        calls.append((name, use_last_error))
        return kernel32

    monkeypatch.setattr(publication.ctypes, "WinDLL", load_windows_dll, raising=False)
    publication._windows_kernel32_binding.cache_clear()
    publication._windows_file_id_bindings.cache_clear()
    publication._windows_object_id_binding.cache_clear()
    try:
        loaded_kernel32, *_bindings = publication._windows_file_id_bindings()
        object_id_binding = publication._windows_object_id_binding()
    finally:
        publication._windows_object_id_binding.cache_clear()
        publication._windows_file_id_bindings.cache_clear()
        publication._windows_kernel32_binding.cache_clear()

    assert loaded_kernel32 is kernel32
    assert object_id_binding is device_io_control
    assert calls == [("kernel32", True)]


def test_windows_mutation_observation_uses_native_change_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get_file_info(
        handle: int,
        info_class: int,
        output: Any,
        output_size: int,
    ) -> int:
        assert handle == 101
        assert info_class == publication._WINDOWS_FILE_BASIC_INFO_CLASS
        assert output_size == ctypes.sizeof(publication._WindowsFileBasicInfo)
        info = ctypes.cast(
            output,
            ctypes.POINTER(publication._WindowsFileBasicInfo),
        ).contents
        info.change_time = 987_654_321
        return 1

    monkeypatch.setattr(
        publication,
        "_windows_file_id_bindings",
        lambda: (object(), object(), object(), get_file_info),
    )
    value = cast(
        "os.stat_result",
        SimpleNamespace(
            st_size=4,
            st_mode=stat.S_IFREG | 0o644,
            st_mtime_ns=123,
            st_ctime_ns=456,
            st_file_attributes=0,
        ),
    )

    observation = publication._capture_windows_file_mutation_observation(
        value,
        handle=101,
    )

    assert observation.changed_token == 987_654_321
    assert observation.changed_token != value.st_ctime_ns


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX sealing")
def test_tree_seal_captures_regular_file_incarnation_while_descriptor_is_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "value.txt").write_text("value", encoding="utf-8")
    original_capture = publication._capture_stable_identity
    observed_regular_descriptor = False

    def capture_while_pinned(
        value: os.stat_result,
        *,
        path: Path | None = None,
        descriptor: int | None = None,
        dir_fd: int | None = None,
        name: str | None = None,
    ) -> publication._Identity:
        nonlocal observed_regular_descriptor
        if stat.S_ISREG(value.st_mode):
            assert descriptor is not None
            assert os.path.samestat(value, os.fstat(descriptor))
            observed_regular_descriptor = True
        return original_capture(
            value,
            path=path,
            descriptor=descriptor,
            dir_fd=dir_fd,
            name=name,
        )

    root_identity = original_capture(root.stat(follow_symlinks=False), path=root)
    monkeypatch.setattr(publication, "_capture_stable_identity", capture_while_pinned)

    publication._capture_tree_authority(root, expected=root_identity)

    assert observed_regular_descriptor


def _destination_metadata_stem(destination: Path) -> str:
    return publication._publication_metadata_stem(
        publication._publication_metadata_keys(destination.name)
    )


def _metadata_paths(destination: Path) -> tuple[Path, Path]:
    stem = _destination_metadata_stem(destination)
    active = destination.parent / f".cayu-tree-publication-{stem}.jsonl"
    receipt = destination.parent / f".cayu-tree-publication-{stem}-receipt.jsonl"
    return active, receipt


def _active_journal_path(destination: Path) -> Path:
    return _metadata_paths(destination)[0]


def _receipt_path(destination: Path) -> Path:
    return _metadata_paths(destination)[1]


def _pending_journal_paths(destination: Path) -> tuple[Path, ...]:
    active = _active_journal_path(destination)
    return tuple(sorted(active.parent.glob(f"{active.name}.pending-*")))


def _journal_path(destination: Path) -> Path:
    active = _active_journal_path(destination)
    receipt = _receipt_path(destination)
    return active if active.exists() or not receipt.exists() else receipt


def _owned_paths(parent: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in parent.iterdir()
            if path.name.startswith(
                (
                    ".cayu-tree-stage-",
                    ".cayu-tree-backup-",
                    ".cayu-tree-cleanup-",
                    ".cayu-tree-authority-",
                )
            )
        ),
        key=lambda path: path.name,
    )


def _run_crashing_publication(
    destination: Path,
    *,
    phase: str,
    exit_code: int,
    lookup_semantics: str | None = None,
) -> None:
    lookup_override = (
        ""
        if lookup_semantics is None
        else (
            "publication._directory_lookup_semantics = "
            "lambda _parent: publication._DirectoryLookupSemantics"
            f"({lookup_semantics!r})"
        )
    )
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from cayu.cli import _guarded_tree_publication as publication
        from cayu.cli._guarded_tree_publication import DestinationPolicy, publish_guarded_tree

        destination = Path({str(destination)!r})
        {lookup_override}
        def populate(staging):
            staging.write_text('value.txt', 'new\\n')
        def fault(current):
            if current == {phase!r}:
                os._exit({exit_code})
        publication._publication_fault = fault
        publish_guarded_tree(
            destination,
            consumer='test',
            request_digest={_REQUEST_DIGEST!r},
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=populate,
        )
        """
    )
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )
    assert completed.returncode == exit_code


def _run_crashing_replacement_publication(
    destination: Path,
    *,
    phase: str,
    exit_code: int,
) -> None:
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from cayu.cli import _guarded_tree_publication as publication
        from cayu.cli._guarded_tree_publication import DestinationPolicy, publish_guarded_tree

        destination = Path({str(destination)!r})
        def populate(staging):
            staging.write_text('value.txt', 'new\\n')
        def fault(current):
            if current == {phase!r}:
                os._exit({exit_code})
        publication._publication_fault = fault
        publish_guarded_tree(
            destination,
            consumer='test',
            request_digest={_REQUEST_DIGEST!r},
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=populate,
        )
        """
    )
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )
    assert completed.returncode == exit_code


def test_guarded_tree_publication_publishes_absent_destination(tmp_path: Path) -> None:
    destination = tmp_path / "published"

    result = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=_populate,
    )

    assert result.destination == destination
    assert not result.recovered
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()
    assert _owned_paths(tmp_path) == []

    replay = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=lambda _staging: pytest.fail("exact replay must not repopulate"),
    )
    assert replay.recovered


def test_guarded_stage_writer_does_not_write_through_replaced_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    displaced = tmp_path / "displaced-stage"
    replacement: Path | None = None

    def replace_stage_at_population_handoff(phase: str) -> None:
        nonlocal replacement
        if phase != "stage_created":
            return
        replacement = next(tmp_path.glob(".cayu-tree-stage-*"))
        replacement.rename(displaced)
        replacement.mkdir(mode=0o700)

    monkeypatch.setattr(publication, "_publication_fault", replace_stage_at_population_handoff)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert exc_info.value.code == "staging_changed"
    assert replacement is not None
    assert list(replacement.iterdir()) == []
    assert list(displaced.iterdir()) == []
    assert not destination.exists()


def test_guarded_stage_writer_accepts_exact_entry_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publication, "_TREE_ENTRY_LIMIT", 2)
    destination = tmp_path / "published"

    publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=lambda stage: stage.write_text("nested/value.txt", "new\n"),
    )

    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"


def test_posix_tree_sealing_stops_at_remaining_entry_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed: list[str] = []

    @contextmanager
    def excessive_entries(_descriptor: object) -> Iterator[Iterator[SimpleNamespace]]:
        def entries() -> Iterator[SimpleNamespace]:
            for name in ("second", "third"):
                consumed.append(name)
                yield SimpleNamespace(name=name)
            pytest.fail("tree sealing consumed beyond remaining_limit + 1")

        yield entries()

    monkeypatch.setattr(publication, "_TREE_ENTRY_LIMIT", 2)
    monkeypatch.setattr(publication.os, "scandir", excessive_entries)
    entry_budget = publication._TreeEntryBudget(1)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publication._seal_directory_from_fd(
            1,
            root_path=tmp_path,
            prefix=PurePosixPath(),
            digest=hashlib.sha256(),
            flags=0,
            entries=[],
            entry_budget=entry_budget,
            linux_mount_points=frozenset(),
            require_cleanup_access=False,
        )

    assert exc_info.value.code == "tree_limit"
    assert consumed == ["second", "third"]


def test_windows_tree_sealing_stops_at_remaining_entry_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed: list[str] = []

    @contextmanager
    def excessive_entries(_path: object) -> Iterator[Iterator[SimpleNamespace]]:
        def entries() -> Iterator[SimpleNamespace]:
            for name in ("second", "third"):
                consumed.append(name)
                yield SimpleNamespace(name=name)
            pytest.fail("tree sealing consumed beyond remaining_limit + 1")

        yield entries()

    monkeypatch.setattr(publication, "_TREE_ENTRY_LIMIT", 2)
    monkeypatch.setattr(publication.os, "scandir", excessive_entries)
    entry_budget = publication._TreeEntryBudget(1)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publication._seal_windows_directory(
            tmp_path,
            prefix=PurePosixPath(),
            digest=hashlib.sha256(),
            entries=[],
            entry_budget=entry_budget,
            require_cleanup_access=False,
        )

    assert exc_info.value.code == "tree_limit"
    assert consumed == ["second", "third"]


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX sealing")
def test_posix_tree_sealing_shares_one_budget_across_recursive_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "tree"
    nested = root / "0-nested"
    nested.mkdir(parents=True)
    for name in ("a", "b", "c"):
        (nested / name).write_text(name, encoding="utf-8")
    for name in ("x", "y", "z"):
        (root / name).write_text(name, encoding="utf-8")

    expected = publication._capture_stable_identity(
        root.stat(follow_symlinks=False),
        path=root,
    )
    tracked_directories = {
        (value.st_dev, value.st_ino)
        for value in (
            root.stat(follow_symlinks=False),
            nested.stat(follow_symlinks=False),
        )
    }
    consumed: list[str] = []
    scandir = publication.os.scandir

    @contextmanager
    def counted_entries(target: object) -> Iterator[Iterator[Any]]:
        with scandir(cast("Any", target)) as entries:
            value = os.fstat(target) if isinstance(target, int) else os.stat(target)
            tracked = (value.st_dev, value.st_ino) in tracked_directories

            def counted() -> Iterator[Any]:
                for entry in entries:
                    if tracked:
                        consumed.append(entry.name)
                    yield entry

            yield counted()

    monkeypatch.setattr(publication, "_TREE_ENTRY_LIMIT", 4)
    monkeypatch.setattr(publication.os, "scandir", counted_entries)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publication._capture_tree_authority(root, expected=expected)

    assert exc_info.value.code == "tree_limit"
    assert len(consumed) == 5


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX cleanup")
def test_posix_cleanup_stops_at_first_entry_outside_durable_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed: list[str] = []

    @contextmanager
    def changed_entries(_descriptor: object) -> Iterator[Iterator[SimpleNamespace]]:
        def entries() -> Iterator[SimpleNamespace]:
            for name in ("owned-a", "owned-b", "injected"):
                consumed.append(name)
                yield SimpleNamespace(name=name)
            pytest.fail("cleanup consumed beyond the first unauthorized entry")

        yield entries()

    authority = {
        name: cast("publication._CleanupEntry", object()) for name in ("owned-a", "owned-b")
    }
    monkeypatch.setattr(publication.os, "scandir", changed_entries)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publication._remove_directory_contents_from_fd(
            1,
            path=tmp_path,
            flags=0,
            authority=authority,
            linux_mount_points=frozenset(),
        )

    assert exc_info.value.code == "cleanup_conflict"
    assert exc_info.value.paths == ("injected",)
    assert consumed == ["owned-a", "owned-b", "injected"]


@pytest.mark.parametrize(
    ("file_mode", "directory_mode"),
    ((0, 0o755), (0o644, 0o555)),
)
def test_guarded_stage_writer_rejects_unrecoverable_modes_and_settles(
    tmp_path: Path,
    file_mode: int,
    directory_mode: int,
) -> None:
    contents = {"nested/value.txt": b"new\n"}
    with pytest.raises(GuardedTreePublicationError) as validation_exc_info:
        validate_guarded_tree_files(
            contents,
            file_mode=file_mode,
            directory_mode=directory_mode,
        )
    assert validation_exc_info.value.code == "invalid_population"

    destination = tmp_path / "published"
    with pytest.raises(GuardedTreePublicationError) as publication_exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=lambda stage: stage.write_files(
                contents,
                file_mode=file_mode,
                directory_mode=directory_mode,
            ),
        )

    assert publication_exc_info.value.code == "invalid_population"
    assert not destination.exists()
    assert not _active_journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode boundary")
def test_guarded_stage_writer_accepts_minimum_recoverable_posix_modes(tmp_path: Path) -> None:
    destination = tmp_path / "published"

    publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=lambda stage: stage.write_text(
            "nested/value.txt",
            "new\n",
            file_mode=0o400,
            directory_mode=0o700,
        ),
    )

    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"


def test_guarded_stage_writer_rejects_over_depth_tree_and_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publication, "_TREE_DEPTH_LIMIT", 2)
    destination = tmp_path / "published"

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=lambda stage: stage.write_text("one/two/value.txt", "new\n"),
        )

    assert exc_info.value.code == "tree_limit"
    assert not destination.exists()
    assert not _active_journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_replacement_rejects_over_depth_original_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    marker = destination / "one" / "two" / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep\n", encoding="utf-8")
    monkeypatch.setattr(publication, "_TREE_DEPTH_LIMIT", 2)
    populated = False

    def populate(_stage: GuardedTreeStage) -> None:
        nonlocal populated
        populated = True

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=populate,
        )

    assert exc_info.value.code == "tree_limit"
    assert not populated
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert not _active_journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_replacement_rejects_unrepresentable_cleanup_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    monkeypatch.setattr(publication, "_CLEANUP_MANIFEST_LIMIT_BYTES", 1)
    populated = False

    def populate(_stage: GuardedTreeStage) -> None:
        nonlocal populated
        populated = True

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=populate,
        )

    assert exc_info.value.code == "tree_limit"
    assert not populated
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert not _active_journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
def test_guarded_replacement_rejects_unremovable_original_before_mutation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    locked = destination / "locked"
    locked.mkdir(parents=True)
    marker = locked / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    locked.chmod(0o555)
    populated = False

    def populate(_stage: GuardedTreeStage) -> None:
        nonlocal populated
        populated = True

    try:
        with pytest.raises(GuardedTreePublicationError) as exc_info:
            publish_guarded_tree(
                destination,
                consumer="test",
                request_digest=_REQUEST_DIGEST,
                policy=DestinationPolicy.REPLACE_DIRECTORY,
                populate=populate,
            )
    finally:
        locked.chmod(0o755)

    assert exc_info.value.code == "cleanup_unavailable"
    assert not populated
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert not _active_journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-flag semantics")
@pytest.mark.parametrize("flag_source", ("stat", "linux_ioctl"))
@pytest.mark.parametrize("flag_target", ("original_file", "publication_parent"))
def test_guarded_replacement_rejects_cleanup_blocking_flags_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag_source: str,
    flag_target: str,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    target_inode = (
        marker.stat(follow_symlinks=False).st_ino
        if flag_target == "original_file"
        else tmp_path.stat(follow_symlinks=False).st_ino
    )
    populated = False
    if flag_source == "stat":
        inspect_flags = publication._posix_cleanup_blocking_flags

        def report_marker_as_immutable(value: os.stat_result) -> int:
            if value.st_ino == target_inode:
                return 1
            return inspect_flags(value)

        monkeypatch.setattr(
            publication,
            "_posix_cleanup_blocking_flags",
            report_marker_as_immutable,
        )
    else:
        inspect_linux_flags = publication._linux_cleanup_blocking_flags

        def report_marker_as_linux_immutable(descriptor: int) -> int:
            if os.fstat(descriptor).st_ino == target_inode:
                return 1
            return inspect_linux_flags(descriptor)

        monkeypatch.setattr(
            publication,
            "_linux_cleanup_blocking_flags",
            report_marker_as_linux_immutable,
        )

    def populate(_stage: GuardedTreeStage) -> None:
        nonlocal populated
        populated = True

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=populate,
        )

    assert exc_info.value.code == "cleanup_unavailable"
    assert not populated
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert not _active_journal_path(destination).exists()
    assert not list(tmp_path.glob(".cayu-tree-publication-*.pending-*"))
    assert _owned_paths(tmp_path) == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-flag semantics")
def test_guarded_exact_receipt_replay_does_not_require_mutable_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    first = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=_populate,
    )
    parent_inode = tmp_path.stat(follow_symlinks=False).st_ino
    inspect_flags = publication._posix_cleanup_blocking_flags

    def report_parent_as_immutable(value: os.stat_result) -> int:
        if value.st_ino == parent_inode:
            return 1
        return inspect_flags(value)

    monkeypatch.setattr(
        publication,
        "_posix_cleanup_blocking_flags",
        report_parent_as_immutable,
    )

    replay = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=lambda _stage: pytest.fail("exact replay must not populate"),
    )

    assert not first.recovered
    assert replay.recovered
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"


def test_guarded_tree_publication_reports_terminal_boundary_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"

    @contextmanager
    def fail_after_releasing_publication(
        *_args: object,
        **_kwargs: object,
    ) -> Iterator[None]:
        yield
        raise OSError("simulated lock release failure")

    monkeypatch.setattr(
        publication,
        "cooperative_path_lock",
        fail_after_releasing_publication,
    )

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert exc_info.value.code == "boundary_cleanup_failed"
    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "simulated lock release failure"
    assert _journal_path(destination).is_file()

    monkeypatch.undo()
    result = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=lambda _staging: pytest.fail("exact retry must not repopulate"),
    )
    assert result.recovered


def test_guarded_tree_recovery_reports_terminal_boundary_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=_populate,
    )

    @contextmanager
    def fail_after_recovery(*_args: object, **_kwargs: object) -> Iterator[None]:
        yield
        raise OSError("simulated recovery lock release failure")

    monkeypatch.setattr(publication, "cooperative_path_lock", fail_after_recovery)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "boundary_cleanup_failed"
    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "simulated recovery lock release failure"
    assert _journal_path(destination).is_file()


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX signal delivery semantics")
def test_guarded_tree_publication_propagates_interrupt_during_terminal_boundary_cleanup(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    repository_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        f"""
        import os
        import signal
        from contextlib import contextmanager
        from pathlib import Path
        from cayu.cli import _guarded_tree_publication as publication
        from cayu.cli._guarded_tree_publication import DestinationPolicy, publish_guarded_tree

        destination = Path({str(destination)!r})
        real_lock = publication.cooperative_path_lock
        @contextmanager
        def interrupt_after_lock_release(*args, **kwargs):
            with real_lock(*args, **kwargs):
                yield
            os.kill(os.getpid(), signal.SIGINT)
        publication.cooperative_path_lock = interrupt_after_lock_release
        def populate(staging):
            staging.write_text('value.txt', 'new\\n')
        try:
            publish_guarded_tree(
                destination,
                consumer='test',
                request_digest={_REQUEST_DIGEST!r},
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=populate,
            )
        except KeyboardInterrupt:
            raise SystemExit(91)
        raise SystemExit(92)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )

    assert completed.returncode == 91
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()


def test_guarded_tree_publication_replaces_exact_empty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()

    publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=_populate,
    )

    assert (destination / "nested" / "value.txt").is_file()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_replaces_nonempty_directory_when_authorized(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")

    publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.REPLACE_DIRECTORY,
        populate=_populate,
    )

    assert not (destination / "old.txt").exists()
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_rejects_original_content_changed_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    original = destination / "old.txt"
    original.write_text("old\n", encoding="utf-8")
    changed = False

    def change_original_after_staging(phase: str) -> None:
        nonlocal changed
        if phase == "stage_synced":
            changed = True
            original.write_text("operator edit\n", encoding="utf-8")

    monkeypatch.setattr(publication, "_publication_fault", change_original_after_staging)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=_populate,
        )

    assert changed
    assert exc_info.value.code == "publication_conflict"
    assert original.read_text(encoding="utf-8") == "operator edit\n"
    assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_does_not_move_replaced_original_at_rename_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    displaced_original = tmp_path / "displaced-original"
    rename_name_no_replace = publication._rename_name_no_replace

    def replace_original_at_namespace_boundary(
        parent: publication._Parent,
        source: str,
        target: str,
    ) -> None:
        if source == destination.name and target.startswith(".cayu-tree-backup-"):
            destination.rename(displaced_original)
            destination.mkdir()
            (destination / "operator.txt").write_text("keep\n", encoding="utf-8")
        rename_name_no_replace(parent, source, target)

    monkeypatch.setattr(
        publication,
        "_rename_name_no_replace",
        replace_original_at_namespace_boundary,
    )

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=_populate,
        )

    assert exc_info.value.code == "original_destination_changed"
    assert (destination / "operator.txt").read_text(encoding="utf-8") == "keep\n"
    assert (displaced_original / "old.txt").read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(".cayu-tree-backup-*"))
    assert _journal_path(destination).is_file()
    assert isinstance(exc_info.value.__cause__, GuardedTreePublicationError)
    assert set(exc_info.value.__cause__.paths) >= {destination.name}


@pytest.mark.skipif(os.name == "nt", reason="exercises POSIX source-descriptor ownership")
def test_guarded_rename_preserves_primary_conflict_when_source_close_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    expected = publication._capture_parent(source)
    displaced = tmp_path / "displaced"
    rename_name_no_replace = publication._rename_name_no_replace
    close_descriptor = publication.os.close
    failed_close = False
    expected_source_closes = 0
    replaced_source = False

    def replace_source_during_rename(
        parent: publication._Parent,
        source_name: str,
        destination_name: str,
    ) -> None:
        nonlocal replaced_source
        if not replaced_source:
            replaced_source = True
            source.rename(displaced)
            source.mkdir()
        rename_name_no_replace(parent, source_name, destination_name)

    def fail_expected_source_close(descriptor: int) -> None:
        nonlocal expected_source_closes, failed_close
        is_expected = expected.matches(os.fstat(descriptor))
        close_descriptor(descriptor)
        if is_expected:
            expected_source_closes += 1
        # Stable identity capture owns the first short-lived descriptor. The
        # second is the rename operation's pinned source owner.
        if expected_source_closes == 2 and not failed_close:
            failed_close = True
            raise OSError("simulated pinned-source close failure")

    monkeypatch.setattr(publication, "_rename_name_no_replace", replace_source_during_rename)
    expected_parent = publication._capture_parent(tmp_path)
    with publication._pinned_parent(tmp_path, expected=expected_parent) as parent:
        monkeypatch.setattr(publication.os, "close", fail_expected_source_close)
        with pytest.raises(GuardedTreePublicationError) as exc_info:
            publication._rename_no_replace(
                parent,
                source.name,
                "destination",
                expected=expected,
                label="source",
            )

    assert failed_close
    assert exc_info.value.code == "source_changed"
    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "simulated pinned-source close failure"
    assert source.is_dir()
    assert displaced.is_dir()
    assert not (tmp_path / "destination").exists()


def test_guarded_tree_publication_rejects_invalid_policy_before_allocation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=cast("DestinationPolicy", "absent_or_empty"),
            populate=_populate,
        )

    assert exc_info.value.code == "invalid_policy"
    assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_rejects_control_character_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published\nforged-diagnostic"

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert exc_info.value.code == "invalid_destination"
    assert not destination.exists()
    assert not list(tmp_path.glob(".cayu-tree-*"))


@pytest.mark.parametrize("entrance", ("publish", "recover"))
def test_guarded_tree_publication_rejects_ancestor_alias_before_allocation(
    tmp_path: Path,
    entrance: str,
) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    marker = parent / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    destination = child / ".."

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        if entrance == "publish":
            publish_guarded_tree(
                destination,
                consumer="test",
                request_digest=_REQUEST_DIGEST,
                policy=DestinationPolicy.REPLACE_DIRECTORY,
                populate=lambda _stage: pytest.fail("ancestor alias must not populate"),
            )
        else:
            recover_guarded_tree(destination)

    assert exc_info.value.code == "invalid_destination"
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert not list(child.glob(".cayu-tree-*"))


@pytest.mark.parametrize(
    "destination_name",
    ("published.", "published ", "CON", "con.txt", "LPT1.log", "bad:name"),
)
@pytest.mark.skipif(os.name != "nt", reason="Win32 namespace aliases are Windows-specific")
def test_guarded_tree_publication_rejects_win32_aliased_destination_before_keying(
    tmp_path: Path,
    destination_name: str,
) -> None:
    destination = tmp_path / destination_name

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=lambda _stage: pytest.fail("invalid destination must not populate"),
        )

    assert exc_info.value.code == "invalid_destination"
    with pytest.raises(GuardedTreePublicationError) as recovery_exc_info:
        recover_guarded_tree(destination)
    assert recovery_exc_info.value.code == "invalid_destination"
    assert not destination.exists()
    assert not list(tmp_path.glob(".cayu-tree-*"))


@pytest.mark.skipif(os.name == "nt", reason="these names alias or are reserved on Windows")
@pytest.mark.parametrize(
    "destination_name",
    ("published.", "published ", "CON", "con.txt", "LPT1.log", "bad:name", "back\\slash"),
)
def test_guarded_tree_publication_accepts_posix_destination_names(
    tmp_path: Path,
    destination_name: str,
) -> None:
    destination = tmp_path / destination_name

    result = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=_populate,
    )

    assert result.destination == destination
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert recover_guarded_tree(destination) == "published"


def test_guarded_tree_publication_rejects_nonempty_destination_without_mutation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    canary = destination / "owned.txt"
    canary.write_text("keep\n", encoding="utf-8")

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert exc_info.value.code == "destination_not_empty"
    assert canary.read_text(encoding="utf-8") == "keep\n"
    assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_uses_native_destination_case_semantics(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    alias = tmp_path / "PUBLISHED"
    alias.mkdir()
    marker = alias / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")

    if not destination.exists():
        result = publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

        assert result.destination == destination
        assert marker.read_text(encoding="utf-8") == "keep\n"
        assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
        assert _active_journal_path(destination) != _active_journal_path(alias)
        assert recover_guarded_tree(destination) == "published"
        return

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert exc_info.value.code == "destination_case_alias"
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert not _journal_path(destination).exists()


def test_destination_alias_scan_finds_late_alias_without_retaining_unrelated_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed: list[str] = []

    @contextmanager
    def parent_entries(_root: object) -> Iterator[Iterator[SimpleNamespace]]:
        def entries() -> Iterator[SimpleNamespace]:
            for index in range(32):
                name = f"unrelated-{index}"
                consumed.append(name)
                yield SimpleNamespace(name=name)
            consumed.append("PUBLISHED")
            yield SimpleNamespace(name="PUBLISHED")

        yield entries()

    parent = SimpleNamespace(path=tmp_path, descriptor=None, entry_stat=lambda _name: object())
    monkeypatch.setattr(publication.os, "name", "nt")
    monkeypatch.setattr(publication.os, "scandir", parent_entries)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publication._reject_case_alias(cast("publication._Parent", parent), "published")

    assert exc_info.value.code == "destination_case_alias"
    assert exc_info.value.paths == ("PUBLISHED",)
    assert consumed[-1] == "PUBLISHED"


def test_destination_alias_scan_bounds_unrelated_parent_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed: list[str] = []

    @contextmanager
    def unrelated_entries(_root: object) -> Iterator[Iterator[SimpleNamespace]]:
        def entries() -> Iterator[SimpleNamespace]:
            for name in ("unrelated-1", "unrelated-2", "unrelated-3"):
                consumed.append(name)
                yield SimpleNamespace(name=name)
            pytest.fail("destination alias discovery exceeded limit + 1 inspections")

        yield entries()

    parent = SimpleNamespace(path=tmp_path, descriptor=None, entry_stat=lambda _name: None)
    monkeypatch.setattr(publication, "_PARENT_DIRECTORY_CENSUS_LIMIT", 2)
    monkeypatch.setattr(publication.os, "scandir", unrelated_entries)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publication._reject_case_alias(cast("publication._Parent", parent), "published")

    assert exc_info.value.code == "parent_inspection_failed"
    assert consumed == ["unrelated-1", "unrelated-2", "unrelated-3"]


def test_case_sensitive_darwin_lookup_normalizes_canonical_unicode_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_parent = publication._capture_parent(tmp_path)
    monkeypatch.setattr(publication.sys, "platform", "darwin")
    monkeypatch.setattr(publication.os, "pathconf", lambda _root, _name: 1)

    with publication._pinned_parent(tmp_path, expected=expected_parent) as parent:
        semantics = publication._directory_lookup_semantics(parent)
        assert semantics is publication._DirectoryLookupSemantics.UNICODE_NORMALIZED
        composed = publication._destination_name_for_lookup_semantics(
            "caf\N{LATIN SMALL LETTER E WITH ACUTE}", semantics
        )
        decomposed = publication._destination_name_for_lookup_semantics(
            "cafe\N{COMBINING ACUTE ACCENT}", semantics
        )
        differently_cased = publication._destination_name_for_lookup_semantics(
            "Caf\N{LATIN SMALL LETTER E WITH ACUTE}",
            semantics,
        )

    assert composed == decomposed
    assert differently_cased != composed


@pytest.mark.parametrize("entrance", ("publish", "recover"))
def test_interrupted_publication_rejects_canonical_unicode_alias_without_new_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrance: str,
) -> None:
    destination = tmp_path / "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    alias = tmp_path / "cafe\N{COMBINING ACUTE ACCENT}"
    semantics = "unicode_normalized"
    _run_crashing_publication(
        destination,
        phase="stage_synced",
        exit_code=98,
        lookup_semantics=semantics,
    )
    monkeypatch.setattr(
        publication,
        "_directory_lookup_semantics",
        lambda _parent: publication._DirectoryLookupSemantics(semantics),
    )
    journals_before = tuple(tmp_path.glob(".cayu-tree-publication-*.jsonl"))
    owned_before = _owned_paths(tmp_path)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        if entrance == "publish":
            publish_guarded_tree(
                alias,
                consumer="test",
                request_digest=_REQUEST_DIGEST,
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=lambda _stage: pytest.fail("an alias must not acquire a second owner"),
            )
        else:
            recover_guarded_tree(alias)

    assert exc_info.value.code == "invalid_publication_journal"
    assert tuple(tmp_path.glob(".cayu-tree-publication-*.jsonl")) == journals_before
    assert _owned_paths(tmp_path) == owned_before
    assert recover_guarded_tree(destination) == "rolled_back"
    assert _owned_paths(tmp_path) == []


@pytest.mark.parametrize("lookup_semantics", ("unicode_casefolded", "unknown"))
@pytest.mark.parametrize("entrance", ("publish", "recover"))
def test_interrupted_publication_rejects_conservative_alias_without_new_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lookup_semantics: str,
    entrance: str,
) -> None:
    destination = tmp_path / "Published"
    alias = tmp_path / "published"
    _run_crashing_publication(
        destination,
        phase="stage_synced",
        exit_code=96,
        lookup_semantics=lookup_semantics,
    )
    monkeypatch.setattr(
        publication,
        "_directory_lookup_semantics",
        lambda _parent: publication._DirectoryLookupSemantics(lookup_semantics),
    )
    journals_before = tuple(tmp_path.glob(".cayu-tree-publication-*.jsonl"))
    owned_before = _owned_paths(tmp_path)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        if entrance == "publish":
            publish_guarded_tree(
                alias,
                consumer="test",
                request_digest=_REQUEST_DIGEST,
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=lambda _stage: pytest.fail("an alias must not acquire a second owner"),
            )
        else:
            recover_guarded_tree(alias)

    assert exc_info.value.code == "invalid_publication_journal"
    assert tuple(tmp_path.glob(".cayu-tree-publication-*.jsonl")) == journals_before
    assert _owned_paths(tmp_path) == owned_before
    assert not destination.exists()
    assert recover_guarded_tree(destination) == "rolled_back"
    assert _owned_paths(tmp_path) == []


@pytest.mark.parametrize(
    ("initial_semantics", "retry_semantics", "destination_name"),
    (
        ("unknown", "case_sensitive", "Published"),
        ("case_sensitive", "unknown", "Published"),
        (
            "unicode_normalized",
            "unknown",
            "Caf\N{LATIN SMALL LETTER E}\N{COMBINING ACUTE ACCENT}",
        ),
        (
            "unknown",
            "unicode_normalized",
            "Caf\N{LATIN SMALL LETTER E}\N{COMBINING ACUTE ACCENT}",
        ),
        (
            "unicode_normalized",
            "unicode_casefolded",
            "Caf\N{LATIN SMALL LETTER E}\N{COMBINING ACUTE ACCENT}",
        ),
        (
            "unicode_casefolded",
            "unicode_normalized",
            "Caf\N{LATIN SMALL LETTER E}\N{COMBINING ACUTE ACCENT}",
        ),
    ),
)
@pytest.mark.parametrize("entrance", ("publish", "recover"))
def test_interrupted_publication_reuses_owner_across_lookup_semantics_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_semantics: str,
    retry_semantics: str,
    destination_name: str,
    entrance: str,
) -> None:
    destination = tmp_path / destination_name
    active, receipt = _metadata_paths(destination)
    _run_crashing_publication(
        destination,
        phase="stage_synced",
        exit_code=97,
        lookup_semantics=initial_semantics,
    )
    assert active.is_file()
    monkeypatch.setattr(
        publication,
        "_directory_lookup_semantics",
        lambda _parent: publication._DirectoryLookupSemantics(retry_semantics),
    )

    if entrance == "publish":
        result = publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )
        assert result.destination == destination
        assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
        assert receipt.is_file()
    else:
        assert recover_guarded_tree(destination) == "rolled_back"
        assert not destination.exists()
        assert not receipt.exists()

    assert not active.exists()
    assert not tuple(tmp_path.glob(f"{active.name}.pending-*"))
    assert _owned_paths(tmp_path) == []


@pytest.mark.parametrize(
    ("initial_semantics", "retry_semantics"),
    (
        ("unknown", "unicode_normalized"),
        ("unicode_normalized", "unknown"),
        ("unicode_casefolded", "unicode_normalized"),
        ("unicode_normalized", "unicode_casefolded"),
    ),
)
@pytest.mark.parametrize("entrance", ("publish", "recover"))
def test_interrupted_publication_rejects_unicode_alias_across_semantics_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_semantics: str,
    retry_semantics: str,
    entrance: str,
) -> None:
    destination = tmp_path / "Caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    alias = tmp_path / "Cafe\N{COMBINING ACUTE ACCENT}"
    initial_active, _initial_receipt = _metadata_paths(destination)
    retry_active, retry_receipt = _metadata_paths(alias)
    assert initial_active != retry_active
    _run_crashing_publication(
        destination,
        phase="stage_synced",
        exit_code=95,
        lookup_semantics=initial_semantics,
    )
    assert initial_active.is_file()
    assert not retry_active.exists()
    monkeypatch.setattr(
        publication,
        "_directory_lookup_semantics",
        lambda _parent: publication._DirectoryLookupSemantics(retry_semantics),
    )
    journals_before = tuple(tmp_path.glob(".cayu-tree-publication-*.jsonl"))
    owned_before = _owned_paths(tmp_path)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        if entrance == "publish":
            publish_guarded_tree(
                alias,
                consumer="test",
                request_digest=_REQUEST_DIGEST,
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=lambda _stage: pytest.fail("an alias must not acquire a second owner"),
            )
        else:
            recover_guarded_tree(alias)

    assert exc_info.value.code == "invalid_publication_journal"
    assert tuple(tmp_path.glob(".cayu-tree-publication-*.jsonl")) == journals_before
    assert _owned_paths(tmp_path) == owned_before
    assert not retry_active.exists()
    assert not retry_receipt.exists()
    monkeypatch.setattr(
        publication,
        "_directory_lookup_semantics",
        lambda _parent: publication._DirectoryLookupSemantics(initial_semantics),
    )
    assert recover_guarded_tree(destination) == "rolled_back"
    assert _owned_paths(tmp_path) == []


@pytest.mark.parametrize(
    ("initial_semantics", "retry_semantics"),
    (
        ("case_sensitive", "unknown"),
        ("unicode_normalized", "unicode_casefolded"),
    ),
)
@pytest.mark.parametrize("metadata_state", ("active", "receipt", "pending"))
@pytest.mark.parametrize("entrance", ("publish", "recover"))
def test_case_changed_alias_rejects_owner_from_prior_lookup_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_semantics: str,
    retry_semantics: str,
    metadata_state: str,
    entrance: str,
) -> None:
    destination = tmp_path / "Caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    alias = tmp_path / "cafe\N{COMBINING ACUTE ACCENT}"
    initial_active, initial_receipt = _metadata_paths(destination)
    retry_active, retry_receipt = _metadata_paths(alias)
    assert initial_active != retry_active

    current_semantics = initial_semantics
    monkeypatch.setattr(
        publication,
        "_directory_lookup_semantics",
        lambda _parent: publication._DirectoryLookupSemantics(current_semantics),
    )
    monkeypatch.setattr(publication, "_reject_case_alias", lambda _parent, _name: None)
    if metadata_state == "receipt":
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )
        assert initial_receipt.is_file()
    else:
        _run_crashing_publication(
            destination,
            phase=("stage_synced" if metadata_state == "active" else "journal_temp_synced"),
            exit_code=92,
            lookup_semantics=initial_semantics,
        )
        if metadata_state == "active":
            assert initial_active.is_file()
        else:
            assert not initial_active.exists()
            assert len(tuple(tmp_path.glob(f"{initial_active.name}.pending-*"))) == 1

    metadata_before = {
        path.name: path.read_bytes() for path in tmp_path.glob(".cayu-tree-publication-*")
    }
    owned_before = _owned_paths(tmp_path)
    current_semantics = retry_semantics

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        if entrance == "publish":
            publish_guarded_tree(
                alias,
                consumer="test",
                request_digest=_REQUEST_DIGEST,
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=lambda _stage: pytest.fail("an alias must not acquire a second owner"),
            )
        else:
            recover_guarded_tree(alias)

    assert exc_info.value.code == "invalid_publication_journal"
    assert {
        path.name: path.read_bytes() for path in tmp_path.glob(".cayu-tree-publication-*")
    } == metadata_before
    assert _owned_paths(tmp_path) == owned_before
    assert not retry_active.exists()
    assert not retry_receipt.exists()
    assert not tuple(tmp_path.glob(f"{retry_active.name}.pending-*"))

    if metadata_state == "active":
        current_semantics = initial_semantics
        assert recover_guarded_tree(destination) == "rolled_back"
        assert _owned_paths(tmp_path) == []


def test_lookup_semantics_probe_failure_is_typed_and_precedes_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "Published"

    def fail_probe(_descriptor: int) -> int | None:
        raise OSError(errno.EIO, "lookup-semantics probe failed")

    monkeypatch.setattr(publication.sys, "platform", "linux")
    monkeypatch.setattr(
        publication,
        "_linux_incarnation",
        lambda expected, **_kwargs: int(expected.st_ctime_ns),
    )
    monkeypatch.setattr(publication, "_linux_file_flags", fail_probe)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=lambda _stage: pytest.fail("probe failure must precede population"),
        )

    assert exc_info.value.code == "parent_inspection_failed"
    assert not destination.exists()
    assert not list(tmp_path.glob(".cayu-tree-*"))


def test_guarded_tree_publication_rejects_symlink_destination_without_mutation(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    target = tmp_path / "target"
    target.mkdir()
    try:
        destination.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert exc_info.value.code == "unsafe_entry"
    assert destination.is_symlink()
    assert list(target.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX FIFO and symlink replacement")
@pytest.mark.parametrize("replacement", ["fifo", "symlink"])
def test_guarded_tree_recovery_rejects_unsafe_published_replacement_without_blocking(
    tmp_path: Path,
    replacement: str,
) -> None:
    destination = tmp_path / "published"
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.txt").write_text("keep\n", encoding="utf-8")
    publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=_populate,
    )
    shutil.rmtree(destination)
    if replacement == "fifo":
        os.mkfifo(destination)
    else:
        try:
            destination.symlink_to(external, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are unavailable")

    repository_root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from cayu.cli._guarded_tree_publication import (
            GuardedTreePublicationError,
            recover_guarded_tree,
        )

        try:
            recover_guarded_tree(Path({str(destination)!r}))
        except GuardedTreePublicationError as exc:
            print(exc.code)
            raise SystemExit(0)
        raise SystemExit(91)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "publication_conflict"
    assert (external / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    if replacement == "fifo":
        assert stat.S_ISFIFO(destination.stat(follow_symlinks=False).st_mode)
    else:
        assert destination.is_symlink()
    assert _journal_path(destination).is_file()
    assert _receipt_path(destination).is_file()


def test_guarded_tree_publication_rolls_back_populate_failure(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    original_identity = destination.stat()

    def fail_after_write(staging: GuardedTreeStage) -> None:
        staging.write_text("partial.txt", "partial\n")
        raise RuntimeError("populate failed")

    with pytest.raises(RuntimeError, match="populate failed"):
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=fail_after_write,
        )

    assert os.path.samestat(original_identity, destination.stat())
    assert list(destination.iterdir()) == []
    assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_preserves_unsafe_stage_without_following_it(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")

    def populate_with_link(staging: GuardedTreeStage) -> None:
        try:
            (staging._path / "unsafe").symlink_to(external, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are unavailable")

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=populate_with_link,
        )

    assert exc_info.value.code == "unsafe_tree_entry"
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert not destination.exists()
    assert len(_owned_paths(tmp_path)) == 1
    assert _journal_path(destination).is_file()
    assert any("remains recoverable" in note for note in exc_info.value.__notes__)


def test_guarded_tree_publication_rolls_back_when_second_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    original_identity = destination.stat()
    real_rename = publication._rename_no_replace
    calls = 0

    def fail_second_rename(
        parent: publication._Parent,
        source: str,
        target: str,
        *,
        expected: publication._Identity,
        label: str,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second rename failure")
        real_rename(parent, source, target, expected=expected, label=label)

    monkeypatch.setattr(publication, "_rename_no_replace", fail_second_rename)

    with pytest.raises(OSError, match="simulated second rename failure"):
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert os.path.samestat(original_identity, destination.stat())
    assert list(destination.iterdir()) == []
    assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_rolls_back_after_backup_sync_acknowledgement_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    original_identity = destination.stat()
    sync = publication._Parent.sync
    failed = False

    def fail_after_backup_sync(parent: publication._Parent) -> None:
        nonlocal failed
        sync(parent)
        if (
            not failed
            and parent.entry_stat(destination.name) is None
            and any(path.name.startswith(".cayu-tree-backup-") for path in tmp_path.iterdir())
        ):
            failed = True
            raise OSError("simulated backup sync acknowledgement loss")

    monkeypatch.setattr(publication._Parent, "sync", fail_after_backup_sync)

    with pytest.raises(OSError, match="backup sync acknowledgement loss"):
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert failed
    assert os.path.samestat(original_identity, destination.stat())
    assert list(destination.iterdir()) == []
    assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_settles_journal_creation_acknowledgement_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    sync = publication._Parent.sync
    failed = False

    def fail_after_journal_sync(parent: publication._Parent) -> None:
        nonlocal failed
        sync(parent)
        if not failed and _journal_path(destination).exists() and not _owned_paths(tmp_path):
            failed = True
            raise OSError("simulated journal sync acknowledgement loss")

    monkeypatch.setattr(publication._Parent, "sync", fail_after_journal_sync)

    with pytest.raises(OSError, match="journal sync acknowledgement loss"):
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert failed
    assert not destination.exists()
    assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


@pytest.mark.parametrize("existing_destination", [False, True])
def test_guarded_tree_publication_rolls_back_when_finalizer_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_destination: bool,
) -> None:
    destination = tmp_path / "published"
    original_identity: os.stat_result | None = None
    if existing_destination:
        destination.mkdir()
        original_identity = destination.stat()

    def fail_finalizer(
        _parent: publication._Parent,
        _name: str,
        *,
        expected: publication._Identity,
    ) -> None:
        del expected
        raise OSError("simulated finalizer failure")

    monkeypatch.setattr(publication, "_finalize_published_tree", fail_finalizer)

    with pytest.raises(OSError, match="simulated finalizer failure"):
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    if original_identity is None:
        assert not destination.exists()
    else:
        assert os.path.samestat(original_identity, destination.stat())
        assert list(destination.iterdir()) == []
    assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_preserves_conflict_after_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    replacement = destination
    real_fault = publication._publication_fault

    def replace_destination(phase: str) -> None:
        if phase == "original_backed_up":
            replacement.mkdir()
            (replacement / "operator.txt").write_text("keep\n", encoding="utf-8")
            raise RuntimeError("stop after replacement")
        real_fault(phase)

    monkeypatch.setattr(publication, "_publication_fault", replace_destination)

    with pytest.raises(RuntimeError, match="stop after replacement") as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert (destination / "operator.txt").read_text(encoding="utf-8") == "keep\n"
    assert _journal_path(destination).is_file()
    assert len(_owned_paths(tmp_path)) == 2
    assert any("remains recoverable" in note for note in exc_info.value.__notes__)


def test_guarded_tree_publication_preserves_original_changed_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    changed_backup: Path | None = None

    def change_original_backup(phase: str) -> None:
        nonlocal changed_backup
        if phase != "published":
            return
        changed_backup = next(tmp_path.glob(".cayu-tree-backup-*"))
        (changed_backup / "operator.txt").write_text("keep\n", encoding="utf-8")

    monkeypatch.setattr(publication, "_publication_fault", change_original_backup)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=_populate,
        )

    assert exc_info.value.code == "publication_conflict"
    assert changed_backup is not None
    assert (changed_backup / "operator.txt").read_text(encoding="utf-8") == "keep\n"
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()
    assert any("remains recoverable" in note for note in exc_info.value.__notes__)


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX cleanup")
def test_guarded_tree_publication_preserves_same_inode_write_during_cleanup_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    capture_cleanup_entry = publication._capture_posix_cleanup_entry
    read = os.read
    active_entry: tuple[int, str] | None = None
    changed_path: Path | None = None

    def capture_with_mutation_barrier(
        descriptor: int,
        name: str,
        *,
        relative: PurePosixPath,
        value: os.stat_result,
    ) -> publication._CapturedCleanupEntry:
        nonlocal active_entry
        active_entry = (descriptor, name)
        try:
            return capture_cleanup_entry(
                descriptor,
                name,
                relative=relative,
                value=value,
            )
        finally:
            active_entry = None

    def mutate_after_hash(descriptor: int, count: int) -> bytes:
        nonlocal changed_path
        content = read(descriptor, count)
        if content or active_entry is None or changed_path is not None:
            return content
        parent_descriptor, name = active_entry
        writer = os.open(name, os.O_WRONLY, dir_fd=parent_descriptor)
        try:
            os.write(writer, b"new\n")
            os.ftruncate(writer, 4)
            written = os.fstat(writer)
            os.utime(
                writer,
                ns=(written.st_atime_ns, written.st_mtime_ns + 1_000_000_000),
            )
            os.fsync(writer)
        finally:
            os.close(writer)
        cleanup = next(tmp_path.glob(".cayu-tree-cleanup-*"))
        changed_path = cleanup / name
        return content

    monkeypatch.setattr(
        publication,
        "_capture_posix_cleanup_entry",
        capture_with_mutation_barrier,
    )
    monkeypatch.setattr(publication.os, "read", mutate_after_hash)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=_populate,
        )

    assert exc_info.value.code == "cleanup_conflict"
    assert changed_path is not None
    assert changed_path.read_text(encoding="utf-8") == "new\n"
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows cleanup handles")
def test_guarded_tree_publication_preserves_same_file_write_with_restored_mtime_during_windows_cleanup_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    seal_windows_directory = publication._seal_windows_directory
    sync_windows_path = publication._sync_windows_path
    cleanup_scans = 0
    mutate_during_scan = False
    changed_path: Path | None = None

    def seal_with_mutation_barrier(
        path: Path,
        *,
        prefix: PurePosixPath,
        digest: Any,
        entries: list[publication._CleanupEntry],
        entry_budget: publication._TreeEntryBudget,
        require_cleanup_access: bool,
    ) -> None:
        nonlocal cleanup_scans, mutate_during_scan
        is_cleanup_root = path.name.startswith(".cayu-tree-cleanup-") and not prefix.parts
        if is_cleanup_root:
            cleanup_scans += 1
            mutate_during_scan = cleanup_scans == 3
        try:
            seal_windows_directory(
                path,
                prefix=prefix,
                digest=digest,
                entries=entries,
                entry_budget=entry_budget,
                require_cleanup_access=require_cleanup_access,
            )
        finally:
            if is_cleanup_root:
                mutate_during_scan = False

    def mutate_before_final_observation(path: Path, *, directory: bool) -> None:
        nonlocal changed_path
        if mutate_during_scan and not directory and path.name == "old.txt":
            before = path.stat(follow_symlinks=False)
            path.write_text("new\n", encoding="utf-8")
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns),
                follow_symlinks=False,
            )
            changed_path = path
        sync_windows_path(path, directory=directory)

    monkeypatch.setattr(publication, "_seal_windows_directory", seal_with_mutation_barrier)
    monkeypatch.setattr(publication, "_sync_windows_path", mutate_before_final_observation)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=_populate,
        )

    assert exc_info.value.code == "staging_changed"
    assert changed_path is not None
    assert changed_path.read_text(encoding="utf-8") == "new\n"
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX cleanup")
def test_guarded_tree_publication_claims_cleanup_before_deleting_owned_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    remove_contents = publication._remove_directory_contents_from_fd
    replacement_backup: Path | None = None

    def introduce_replacement_after_cleanup_claim(
        descriptor: int,
        *,
        path: Path,
        flags: int,
        authority: dict[str, publication._CleanupEntry] | None = None,
        prefix: PurePosixPath | None = None,
        linux_mount_points: frozenset[str] | None = None,
    ) -> None:
        nonlocal replacement_backup
        if path.name.startswith(".cayu-tree-cleanup-") and replacement_backup is None:
            journal_entry = json.loads(_journal_path(destination).read_text().splitlines()[0])
            replacement_backup = tmp_path / journal_entry["backup_name"]
            replacement_backup.mkdir()
            (replacement_backup / "operator.txt").write_text("keep\n", encoding="utf-8")
        remove_contents(
            descriptor,
            path=path,
            flags=flags,
            authority=authority,
            prefix=PurePosixPath() if prefix is None else prefix,
            linux_mount_points=linux_mount_points,
        )

    monkeypatch.setattr(
        publication,
        "_remove_directory_contents_from_fd",
        introduce_replacement_after_cleanup_claim,
    )

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=_populate,
        )

    assert exc_info.value.code == "cleanup_conflict"
    assert replacement_backup is not None
    assert (replacement_backup / "operator.txt").read_text(encoding="utf-8") == "keep\n"
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX cleanup")
def test_guarded_tree_publication_rejects_same_content_descendant_replacement_after_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    displaced = tmp_path / "displaced-old.txt"
    capture_tree_authority = publication._capture_tree_authority
    replaced = False

    def replace_descendant_before_cleanup_seal(
        path: Path,
        *,
        expected: publication._Identity,
        require_cleanup_access: bool = False,
    ) -> tuple[str, tuple[publication._CleanupEntry, ...]]:
        nonlocal replaced
        if path.name.startswith(".cayu-tree-cleanup-") and not replaced:
            replaced = True
            child = path / "old.txt"
            child.rename(displaced)
            child.write_text("old\n", encoding="utf-8")
        return capture_tree_authority(
            path,
            expected=expected,
            require_cleanup_access=require_cleanup_access,
        )

    monkeypatch.setattr(
        publication,
        "_capture_tree_authority",
        replace_descendant_before_cleanup_seal,
    )

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=_populate,
        )

    assert replaced
    assert exc_info.value.code == "cleanup_conflict"
    cleanup = next(tmp_path.glob(".cayu-tree-cleanup-*"))
    assert (cleanup / "old.txt").read_text(encoding="utf-8") == "old\n"
    assert displaced.read_text(encoding="utf-8") == "old\n"
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _active_journal_path(destination).is_file()
    assert not _receipt_path(destination).exists()


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX cleanup")
def test_guarded_tree_publication_preserves_descendant_replaced_after_cleanup_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    displaced = tmp_path / "displaced-old.txt"
    mark_cleanup_sealed = publication._mark_cleanup_sealed
    replaced = False

    def replace_after_cleanup_seal(*args: object, **kwargs: object) -> None:
        nonlocal replaced
        mark_cleanup_sealed(*args, **kwargs)
        cleanup = next(tmp_path.glob(".cayu-tree-cleanup-*"))
        child = cleanup / "old.txt"
        child.rename(displaced)
        child.write_text("foreign\n", encoding="utf-8")
        replaced = True

    monkeypatch.setattr(publication, "_mark_cleanup_sealed", replace_after_cleanup_seal)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=_populate,
        )

    assert replaced
    assert exc_info.value.code == "cleanup_conflict"
    cleanup = next(tmp_path.glob(".cayu-tree-cleanup-*"))
    assert (cleanup / "old.txt").read_text(encoding="utf-8") == "foreign\n"
    assert displaced.read_text(encoding="utf-8") == "old\n"
    assert not _receipt_path(destination).exists()


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX cleanup")
def test_guarded_tree_recovery_preserves_descendant_replaced_after_cleanup_seal(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    _run_crashing_replacement_publication(
        destination,
        phase="cleanup_sealed",
        exit_code=91,
    )
    cleanup = next(tmp_path.glob(".cayu-tree-cleanup-*"))
    displaced = tmp_path / "displaced-old.txt"
    (cleanup / "old.txt").rename(displaced)
    (cleanup / "old.txt").write_text("foreign\n", encoding="utf-8")

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "cleanup_conflict"
    assert (cleanup / "old.txt").read_text(encoding="utf-8") == "foreign\n"
    assert displaced.read_text(encoding="utf-8") == "old\n"
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert not _receipt_path(destination).exists()


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX cleanup")
def test_guarded_tree_recovery_rejects_reused_descendant_inode_after_cleanup_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    _run_crashing_replacement_publication(
        destination,
        phase="cleanup_sealed",
        exit_code=91,
    )
    cleanup = next(tmp_path.glob(".cayu-tree-cleanup-*"))
    retained = cleanup / "old.txt"
    retained_stat = retained.stat(follow_symlinks=False)
    capture_stable_identity = publication._capture_stable_identity

    def report_reused_inode(
        value: os.stat_result,
        *,
        path: Path | None = None,
        descriptor: int | None = None,
        dir_fd: int | None = None,
        name: str | None = None,
    ) -> publication._Identity:
        identity = capture_stable_identity(
            value,
            path=path,
            descriptor=descriptor,
            dir_fd=dir_fd,
            name=name,
        )
        if (
            identity.device == retained_stat.st_dev
            and identity.inode == retained_stat.st_ino
            and identity.kind == stat.S_IFMT(retained_stat.st_mode)
        ):
            assert identity.incarnation is not None
            return replace(identity, incarnation=identity.incarnation + 1)
        return identity

    monkeypatch.setattr(publication, "_capture_stable_identity", report_reused_inode)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "cleanup_conflict"
    assert retained.read_text(encoding="utf-8") == "old\n"
    assert not _receipt_path(destination).exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mount-boundary contract")
def test_guarded_tree_publication_rejects_mounted_descendant_before_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    mounted = destination / "mounted"
    mounted.mkdir(parents=True)
    (mounted / "external.txt").write_text("keep\n", encoding="utf-8")
    path_is_mount_boundary = publication._path_is_mount_boundary

    def report_nested_mount(
        path: Path,
        *,
        parent_device: int | None = None,
        linux_mount_points: frozenset[str] | None = None,
    ) -> bool:
        return path == mounted or path_is_mount_boundary(
            path,
            parent_device=parent_device,
            linux_mount_points=linux_mount_points,
        )

    monkeypatch.setattr(publication, "_path_is_mount_boundary", report_nested_mount)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=_populate,
        )

    assert exc_info.value.code == "mount_boundary"
    assert (mounted / "external.txt").read_text(encoding="utf-8") == "keep\n"
    assert not _active_journal_path(destination).exists()


def test_guarded_tree_publication_requires_cleanup_namespace_sync_before_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    sync = publication._Parent.sync

    def fail_cleanup_completion_sync(parent: publication._Parent) -> None:
        journal = _active_journal_path(destination)
        if journal.exists() and not list(tmp_path.glob(".cayu-tree-cleanup-*")):
            entries = [json.loads(line) for line in journal.read_text().splitlines()]
            if entries[-1]["phase"] == "cleanup_sealed":
                raise OSError("simulated cleanup namespace sync failure")
        sync(parent)

    monkeypatch.setattr(publication._Parent, "sync", fail_cleanup_completion_sync)

    with pytest.raises(OSError, match="cleanup namespace sync failure"):
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=_populate,
        )

    assert _active_journal_path(destination).is_file()
    assert not _receipt_path(destination).exists()
    entries = [
        json.loads(line) for line in _active_journal_path(destination).read_text().splitlines()
    ]
    assert entries[-1]["phase"] == "cleanup_sealed"

    monkeypatch.undo()
    result = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.REPLACE_DIRECTORY,
        populate=lambda _staging: pytest.fail("cleanup recovery must not repopulate"),
    )

    assert result.recovered
    assert not _active_journal_path(destination).exists()
    assert _receipt_path(destination).is_file()


@pytest.mark.parametrize(
    ("phase", "expected_outcome"),
    [
        ("stage_created", "rolled_back"),
        ("stage_synced", "rolled_back"),
        ("commit_intent_synced", "rolled_back"),
        ("original_backed_up", "rolled_back"),
        ("tree_renamed", "published"),
        ("published", "published"),
        ("cleanup_owned", "published"),
        ("cleanup_sealed", "published"),
        ("settled", "published"),
    ],
)
def test_guarded_tree_publication_recovers_after_process_death_at_every_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    expected_outcome: str,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    _run_crashing_publication(destination, phase=phase, exit_code=73)
    assert _journal_path(destination).is_file()
    finalized: list[Path] = []

    def record_finalization(
        parent: publication._Parent,
        name: str,
        *,
        expected: publication._Identity,
    ) -> None:
        del expected
        finalized.append(parent.path / name)

    monkeypatch.setattr(publication, "_finalize_published_tree", record_finalization)

    assert recover_guarded_tree(destination) == expected_outcome

    if expected_outcome == "published":
        assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    else:
        assert destination.is_dir()
        assert list(destination.iterdir()) == []
    assert finalized == ([destination] if phase == "tree_renamed" else [])
    if expected_outcome == "published":
        assert _journal_path(destination).is_file()
    else:
        assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []

    assert recover_guarded_tree(destination) == (
        "published" if expected_outcome == "published" else None
    )


def test_guarded_tree_recovery_preserves_stage_without_durable_owner_marker(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    _run_crashing_publication(destination, phase="stage_directory_created", exit_code=93)

    stage = next(tmp_path.glob(".cayu-tree-stage-*"))
    assert list(stage.iterdir()) == []

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "publication_conflict"
    assert stage.is_dir()
    assert list(stage.iterdir()) == []
    assert _active_journal_path(destination).is_file()


def test_guarded_tree_publication_preserves_user_created_markerless_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    create_private_stage = publication._create_private_stage
    lookalike: Path | None = None

    def race_stage_creation(
        stage: Path,
        *,
        token: str,
        parent: publication._Parent,
    ) -> None:
        nonlocal lookalike
        lookalike = stage
        stage.mkdir(mode=0o700)
        create_private_stage(stage, token=token, parent=parent)

    monkeypatch.setattr(publication, "_create_private_stage", race_stage_creation)

    with pytest.raises(FileExistsError):
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert lookalike is not None
    assert lookalike.is_dir()
    assert list(lookalike.iterdir()) == []
    assert _active_journal_path(destination).is_file()


@pytest.mark.parametrize("replacement_appears", [False, True])
def test_guarded_tree_publication_binds_prepared_ownership_to_exact_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_appears: bool,
) -> None:
    destination = tmp_path / "published"
    create_private_stage = publication._create_private_stage
    prepared_stage_identity = publication._owned_prepared_stage_identity
    stage_path: Path | None = None
    displaced_stage: Path | None = None
    ownership_checks = 0

    def fail_after_prepared_stage(
        stage: Path,
        *,
        token: str,
        parent: publication._Parent,
    ) -> None:
        nonlocal stage_path
        create_private_stage(stage, token=token, parent=parent)
        stage_path = stage
        raise RuntimeError("simulated failure after prepared-stage ownership")

    def replace_after_ownership_proof(
        parent: publication._Parent,
        stage_name: str,
        *,
        token: str,
    ) -> publication._Identity | None:
        nonlocal displaced_stage, ownership_checks
        ownership_checks += 1
        identity = prepared_stage_identity(parent, stage_name, token=token)
        if identity is None:
            return None
        stage = parent.path / stage_name
        displaced_stage = parent.path / f"{stage_name}-displaced"
        stage.rename(displaced_stage)
        if replacement_appears:
            stage.mkdir(mode=0o700)
            (stage / "user-owned.txt").write_text("keep\n", encoding="utf-8")
        return identity

    monkeypatch.setattr(publication, "_create_private_stage", fail_after_prepared_stage)
    monkeypatch.setattr(
        publication,
        "_owned_prepared_stage_identity",
        replace_after_ownership_proof,
    )

    with pytest.raises(
        RuntimeError,
        match="simulated failure after prepared-stage ownership",
    ) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )

    assert ownership_checks == 1
    assert isinstance(exc_info.value.__cause__, GuardedTreePublicationError)
    assert exc_info.value.__cause__.code == "cleanup_conflict"
    assert stage_path is not None
    assert displaced_stage is not None
    assert displaced_stage.is_dir()
    if replacement_appears:
        assert (stage_path / "user-owned.txt").read_text(encoding="utf-8") == "keep\n"
    else:
        assert not stage_path.exists()
    assert _active_journal_path(destination).is_file()

    with pytest.raises(GuardedTreePublicationError) as recovery_error:
        recover_guarded_tree(destination)

    assert recovery_error.value.code == "publication_conflict"
    assert displaced_stage.is_dir()
    if replacement_appears:
        assert (stage_path / "user-owned.txt").read_text(encoding="utf-8") == "keep\n"
    assert _active_journal_path(destination).is_file()


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX subprocess exit semantics")
def test_guarded_tree_recovery_resumes_prepared_cleanup_after_marker_removal(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from cayu.cli import _guarded_tree_publication as publication
        from cayu.cli._guarded_tree_publication import DestinationPolicy, publish_guarded_tree

        destination = Path({str(destination)!r})
        create_private_stage = publication._create_private_stage
        def fail_after_prepared_stage(stage, *, token, parent):
            create_private_stage(stage, token=token, parent=parent)
            raise RuntimeError('simulated failure after prepared-stage ownership')
        def fault(phase):
            if phase == 'cleanup_entry_removed':
                os._exit(94)
        publication._create_private_stage = fail_after_prepared_stage
        publication._publication_fault = fault
        publish_guarded_tree(
            destination,
            consumer='test',
            request_digest={_REQUEST_DIGEST!r},
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=lambda _staging: None,
        )
        """
    )
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )

    assert completed.returncode == 94
    cleanup = next(tmp_path.glob(".cayu-tree-cleanup-*"))
    assert cleanup.is_dir()
    assert list(cleanup.iterdir()) == []
    assert _active_journal_path(destination).is_file()

    assert recover_guarded_tree(destination) == "rolled_back"

    assert not destination.exists()
    assert not _active_journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []
    assert recover_guarded_tree(destination) is None


@pytest.mark.parametrize("entrance", ("publish", "recover"))
@pytest.mark.parametrize(
    "phase",
    ("journal_temp_created", "journal_temp_written", "journal_temp_synced"),
)
def test_guarded_tree_publication_preserves_initial_pending_journal_after_crash(
    tmp_path: Path,
    entrance: str,
    phase: str,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    _run_crashing_publication(destination, phase=phase, exit_code=93)

    pending = _pending_journal_paths(destination)
    assert len(pending) == 1
    pending_content = pending[0].read_bytes()
    pending_identity = pending[0].stat(follow_symlinks=False)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        if entrance == "publish":
            publish_guarded_tree(
                destination,
                consumer="test",
                request_digest=_REQUEST_DIGEST,
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=lambda _stage: pytest.fail(
                    "unverifiable pending metadata must prevent population"
                ),
            )
        else:
            recover_guarded_tree(destination)

    assert exc_info.value.code == "invalid_publication_journal"
    assert os.path.samestat(pending_identity, pending[0].stat(follow_symlinks=False))
    assert pending[0].read_bytes() == pending_content
    assert not (destination / "nested").exists()


@pytest.mark.parametrize("entrance", ("publish", "recover"))
def test_guarded_tree_publication_preserves_user_created_pending_lookalike(
    tmp_path: Path,
    entrance: str,
) -> None:
    destination = tmp_path / "published"
    active = _active_journal_path(destination)
    pending = active.with_name(f"{active.name}.pending-{'0' * 64}")
    content = b"user-owned pending lookalike\n"
    pending.write_bytes(content)
    pending.chmod(0o600)
    pending_identity = pending.stat(follow_symlinks=False)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        if entrance == "publish":
            publish_guarded_tree(
                destination,
                consumer="test",
                request_digest=_REQUEST_DIGEST,
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=lambda _stage: pytest.fail(
                    "a user-created pending lookalike must prevent population"
                ),
            )
        else:
            recover_guarded_tree(destination)

    assert exc_info.value.code == "invalid_publication_journal"
    assert os.path.samestat(pending_identity, pending.stat(follow_symlinks=False))
    assert pending.read_bytes() == content
    assert not active.exists()
    assert not destination.exists()


def test_pending_metadata_discovery_stops_after_second_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_name = ".cayu-tree-publication-owner.jsonl"
    prefix = f"{final_name}.pending-"

    @contextmanager
    def candidate_entries(_root: object) -> Iterator[Iterator[Path]]:
        def entries() -> Iterator[Path]:
            yield Path(f"{prefix}{'1' * 64}")
            yield Path("unrelated")
            yield Path(f"{prefix}{'2' * 64}")
            pytest.fail("candidate discovery must stop after the second match")

        yield entries()

    expected_parent = publication._capture_parent(tmp_path)
    with publication._pinned_parent(tmp_path, expected=expected_parent) as parent:
        monkeypatch.setattr(publication.os, "scandir", candidate_entries)
        with pytest.raises(GuardedTreePublicationError) as exc_info:
            publication._pending_metadata_candidates(parent, final_name)

    assert exc_info.value.code == "invalid_publication_journal"


def test_pending_metadata_discovery_bounds_unrelated_parent_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumed: list[str] = []

    @contextmanager
    def unrelated_entries(_root: object) -> Iterator[Iterator[SimpleNamespace]]:
        def entries() -> Iterator[SimpleNamespace]:
            for name in ("unrelated-1", "unrelated-2", "unrelated-3"):
                consumed.append(name)
                yield SimpleNamespace(name=name)
            pytest.fail("pending metadata discovery exceeded limit + 1 inspections")

        yield entries()

    expected_parent = publication._capture_parent(tmp_path)
    with publication._pinned_parent(tmp_path, expected=expected_parent) as parent:
        monkeypatch.setattr(publication, "_PARENT_DIRECTORY_CENSUS_LIMIT", 2)
        monkeypatch.setattr(publication.os, "scandir", unrelated_entries)
        with pytest.raises(GuardedTreePublicationError) as exc_info:
            publication._pending_metadata_candidates(parent, "publication.jsonl")

    assert exc_info.value.code == "invalid_publication_journal"
    assert consumed == ["unrelated-1", "unrelated-2", "unrelated-3"]


def test_publication_metadata_discovery_stops_at_its_collision_domain_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    keys = publication._publication_metadata_keys(destination.name)
    collision_key = keys[0]
    first = f".cayu-tree-publication-{collision_key}-{'1' * 32}-{'2' * 32}.jsonl"
    unrelated = f".cayu-tree-publication-{'3' * 32}-{'4' * 32}-{'5' * 32}.jsonl"
    second = f".cayu-tree-publication-{collision_key}-{'6' * 32}-{'7' * 32}-receipt.jsonl"

    @contextmanager
    def candidate_entries(_root: object) -> Iterator[Iterator[Path]]:
        def entries() -> Iterator[Path]:
            yield Path(first)
            yield Path(unrelated)
            yield Path(second)
            pytest.fail("metadata discovery must stop at its collision-domain limit")

        yield entries()

    expected_parent = publication._capture_parent(tmp_path)
    with publication._pinned_parent(tmp_path, expected=expected_parent) as parent:
        monkeypatch.setattr(publication, "_PUBLICATION_METADATA_CENSUS_LIMIT", 1)
        monkeypatch.setattr(publication.os, "scandir", candidate_entries)
        with pytest.raises(GuardedTreePublicationError) as exc_info:
            publication._publication_metadata_candidates(
                parent,
                keys=keys,
                semantics=publication._DirectoryLookupSemantics.UNKNOWN,
            )

    assert exc_info.value.code == "invalid_publication_journal"


@pytest.mark.parametrize("entrance", ("publish", "recover"))
def test_publication_metadata_discovery_bounds_total_parent_census(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrance: str,
) -> None:
    destination = tmp_path / "published"
    consumed: list[str] = []
    populated = False

    @contextmanager
    def unrelated_entries(_root: object) -> Iterator[Iterator[SimpleNamespace]]:
        def entries() -> Iterator[SimpleNamespace]:
            for name in ("unrelated-1", "unrelated-2", "unrelated-3"):
                consumed.append(name)
                yield SimpleNamespace(name=name)
            pytest.fail("publication metadata discovery exceeded limit + 1 inspections")

        yield entries()

    def populate(_stage: GuardedTreeStage) -> None:
        nonlocal populated
        populated = True

    monkeypatch.setattr(publication, "_PARENT_DIRECTORY_CENSUS_LIMIT", 2)
    monkeypatch.setattr(publication.os, "scandir", unrelated_entries)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        if entrance == "publish":
            publish_guarded_tree(
                destination,
                consumer="test",
                request_digest=_REQUEST_DIGEST,
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=populate,
            )
        else:
            recover_guarded_tree(destination)

    assert exc_info.value.code == "invalid_publication_journal"
    assert consumed == ["unrelated-1", "unrelated-2", "unrelated-3"]
    assert not populated
    assert not destination.exists()
    monkeypatch.undo()
    assert not list(tmp_path.glob(".cayu-tree-*"))


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX subprocess exit semantics")
@pytest.mark.parametrize("entrance", ("publish", "recover"))
def test_unrelated_incomplete_metadata_does_not_block_destination_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrance: str,
) -> None:
    stranded_destination = tmp_path / "stranded"
    malformed_destination = tmp_path / "malformed"
    destination = tmp_path / "published"
    _run_crashing_publication(
        stranded_destination,
        phase="journal_temp_created",
        exit_code=91,
        lookup_semantics="unknown",
    )
    pending = _pending_journal_paths(stranded_destination)
    assert len(pending) == 1
    pending_identity = pending[0].stat(follow_symlinks=False)
    pending_content = pending[0].read_bytes()
    malformed = _active_journal_path(malformed_destination)
    malformed.write_bytes(b"not-json\n")
    malformed_identity = malformed.stat(follow_symlinks=False)
    monkeypatch.setattr(
        publication,
        "_directory_lookup_semantics",
        lambda _parent: publication._DirectoryLookupSemantics.UNKNOWN,
    )
    monkeypatch.setattr(publication, "_PUBLICATION_METADATA_CENSUS_LIMIT", 1)

    if entrance == "publish":
        result = publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )
        assert result.destination == destination
        assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    else:
        assert recover_guarded_tree(destination) is None
        assert not destination.exists()

    assert os.path.samestat(pending_identity, pending[0].stat(follow_symlinks=False))
    assert pending[0].read_bytes() == pending_content
    assert os.path.samestat(malformed_identity, malformed.stat(follow_symlinks=False))
    assert malformed.read_bytes() == b"not-json\n"


@pytest.mark.parametrize("entrance", ("publish", "recover"))
@pytest.mark.parametrize("conflict", ("none", "destination", "parent"))
def test_guarded_tree_pending_journal_preserves_caller_authored_authority(
    tmp_path: Path,
    entrance: str,
    conflict: str,
) -> None:
    destination = tmp_path / "published"
    token = "a" * 32
    parent_identity = publication._capture_parent(tmp_path)
    if conflict == "parent":
        parent_identity = publication._Identity(
            device=parent_identity.device + 1,
            inode=parent_identity.inode,
            kind=parent_identity.kind,
            incarnation=parent_identity.incarnation,
        )
    record = publication._Record(
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        predecessor_request_digest=None,
        token=token,
        destination_name="foreign" if conflict == "destination" else destination.name,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        parent_identity=parent_identity,
        original_identity=None,
        original_sha256=None,
        stage_name=f".cayu-tree-stage-{token}",
        stage_identity=None,
        stage_sha256=None,
        backup_name=f".cayu-tree-backup-{token}",
        cleanup_manifest_identity=None,
        cleanup_manifest_sha256=None,
        phase=publication._Phase.PREPARED,
    )
    content, _entry_sha256 = publication._journal_entry(record, previous_sha256=None)
    pending = _active_journal_path(destination).with_name(
        publication._pending_metadata_name(_active_journal_path(destination).name, content)
    )
    pending.write_bytes(content)
    pending.chmod(0o600)
    pending_identity = pending.stat(follow_symlinks=False)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        if entrance == "publish":
            publish_guarded_tree(
                destination,
                consumer="test",
                request_digest=_REQUEST_DIGEST,
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=lambda _stage: pytest.fail("foreign pending journal must not populate"),
            )
        else:
            recover_guarded_tree(destination)

    assert exc_info.value.code == "invalid_publication_journal"
    assert os.path.samestat(pending_identity, pending.stat(follow_symlinks=False))
    assert pending.read_bytes() == content
    assert not _active_journal_path(destination).exists()
    assert not destination.exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_recovery_promotes_one_pending_cleanup_manifest(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    _run_crashing_replacement_publication(
        destination,
        phase="cleanup_manifest_temp_synced",
        exit_code=94,
    )

    pending = list(tmp_path.glob(".cayu-tree-authority-*.json.pending-*"))
    assert len(pending) == 1
    assert not list(tmp_path.glob(".cayu-tree-authority-*.json"))

    assert recover_guarded_tree(destination) == "published"

    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".cayu-tree-authority-*"))
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_recovery_reclaims_cleanup_manifest_crash_after_create(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    _run_crashing_replacement_publication(
        destination,
        phase="cleanup_manifest_temp_created",
        exit_code=95,
    )

    pending = list(tmp_path.glob(".cayu-tree-authority-*.json.pending-*"))
    assert len(pending) == 1
    assert pending[0].read_bytes() == b""

    assert recover_guarded_tree(destination) == "published"

    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".cayu-tree-authority-*"))
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_recovery_syncs_complete_pending_cleanup_manifest_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    _run_crashing_replacement_publication(
        destination,
        phase="cleanup_manifest_temp_written",
        exit_code=99,
    )
    pending = list(tmp_path.glob(".cayu-tree-authority-*.json.pending-*"))
    assert len(pending) == 1
    assert pending[0].read_bytes()
    synchronized: list[str] = []
    sync_pending = publication._sync_exact_pending_metadata

    def record_sync(
        parent: publication._Parent,
        name: str,
        *,
        expected: publication._Identity,
    ) -> None:
        synchronized.append(name)
        sync_pending(parent, name, expected=expected)

    monkeypatch.setattr(publication, "_sync_exact_pending_metadata", record_sync)

    assert recover_guarded_tree(destination) == "published"

    assert synchronized == [pending[0].name]
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob(".cayu-tree-authority-*"))
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_exact_retry_reuses_committed_tree(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    _run_crashing_publication(destination, phase="settled", exit_code=80)

    def must_not_populate(_staging: GuardedTreeStage) -> None:
        pytest.fail("an exact retry must not populate another tree")

    result = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=must_not_populate,
    )

    assert result.recovered
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()
    assert _owned_paths(tmp_path) == []


@pytest.mark.parametrize("metadata_state", ("active", "receipt"))
def test_guarded_tree_publication_exact_retry_rejects_modified_committed_tree(
    tmp_path: Path,
    metadata_state: str,
) -> None:
    destination = tmp_path / "published"
    if metadata_state == "active":
        destination.mkdir()
        _run_crashing_publication(destination, phase="settled", exit_code=80)
    else:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )
    modified = (
        destination / "value.txt"
        if metadata_state == "active"
        else destination / "nested" / "value.txt"
    )
    modified.write_text("operator edit\n", encoding="utf-8")

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=lambda _stage: pytest.fail("a modified exact retry must not repopulate"),
        )

    assert exc_info.value.code == "destination_not_empty"
    assert modified.read_text(encoding="utf-8") == "operator edit\n"


@pytest.mark.parametrize("metadata_state", ("active", "receipt"))
@pytest.mark.parametrize(
    ("original_state", "retry_state"),
    (("absent", "recreated_empty"), ("empty", "absent")),
)
def test_guarded_tree_publication_exact_retry_republishes_safe_changed_root(
    tmp_path: Path,
    metadata_state: str,
    original_state: str,
    retry_state: str,
) -> None:
    destination = tmp_path / "published"
    if original_state == "empty":
        destination.mkdir()
    if metadata_state == "active":
        _run_crashing_publication(destination, phase="settled", exit_code=80)
    else:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )
    shutil.rmtree(destination)
    if retry_state == "recreated_empty":
        destination.mkdir()

    result = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=lambda stage: _populate(stage, content="republished\n"),
    )

    assert not result.recovered
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "republished\n"
    assert not _active_journal_path(destination).exists()
    assert _receipt_path(destination).is_file()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_does_not_retire_receipt_with_private_name_conflict(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=_populate,
    )
    receipt = _receipt_path(destination)
    record = json.loads(receipt.read_text(encoding="ascii").splitlines()[-1])
    shutil.rmtree(destination)
    conflicting_stage = tmp_path / record["stage_name"]
    conflicting_stage.mkdir()
    marker = conflicting_stage / "keep.txt"
    marker.write_text("application-owned\n", encoding="utf-8")

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=lambda _stage: pytest.fail("ambiguous private state must not repopulate"),
        )

    assert exc_info.value.code == "publication_conflict"
    assert receipt.is_file()
    assert marker.read_text(encoding="utf-8") == "application-owned\n"
    assert not destination.exists()


@pytest.mark.parametrize("metadata_state", ("active", "receipt"))
def test_guarded_tree_publication_does_not_retire_metadata_from_another_parent(
    tmp_path: Path,
    metadata_state: str,
) -> None:
    original_parent = tmp_path / "original"
    replacement_parent = tmp_path / "replacement"
    original_parent.mkdir()
    replacement_parent.mkdir()
    original_destination = original_parent / "published"
    replacement_destination = replacement_parent / "published"
    if metadata_state == "active":
        _run_crashing_publication(original_destination, phase="settled", exit_code=80)
        source_metadata = _active_journal_path(original_destination)
        copied_metadata = _active_journal_path(replacement_destination)
    else:
        publish_guarded_tree(
            original_destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=_populate,
        )
        source_metadata = _receipt_path(original_destination)
        copied_metadata = _receipt_path(replacement_destination)
    shutil.copy2(source_metadata, copied_metadata)
    copied_content = copied_metadata.read_bytes()

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            replacement_destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=lambda _stage: pytest.fail(
                "parent-mismatched publication metadata must not repopulate"
            ),
        )

    assert exc_info.value.code == "publication_conflict"
    assert copied_metadata.read_bytes() == copied_content
    assert not replacement_destination.exists()
    assert not list(replacement_parent.glob(".cayu-tree-stage-*"))


def test_guarded_tree_publication_supersedes_terminal_receipt_with_bound_successor(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    successor_digest = f"sha256:{'2' * 64}"

    publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.REPLACE_DIRECTORY,
        populate=lambda staging: _populate(staging, content="first\n"),
    )
    assert _receipt_path(destination).is_file()

    publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=successor_digest,
        predecessor_request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.REPLACE_DIRECTORY,
        populate=lambda staging: _populate(staging, content="second\n"),
    )

    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "second\n"
    assert not _active_journal_path(destination).exists()
    assert _receipt_path(destination).is_file()
    replay = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=successor_digest,
        predecessor_request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.REPLACE_DIRECTORY,
        populate=lambda _staging: pytest.fail("exact successor replay must not repopulate"),
    )
    assert replay.recovered

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=lambda _staging: pytest.fail("stale predecessor must not repopulate"),
        )
    assert exc_info.value.code == "publication_request_conflict"

    with pytest.raises(GuardedTreePublicationError) as stale_successor:
        publish_guarded_tree(
            destination,
            consumer="test",
            request_digest=f"sha256:{'3' * 64}",
            predecessor_request_digest=_REQUEST_DIGEST,
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=lambda _staging: pytest.fail(
                "a successor bound to the superseded receipt must not repopulate"
            ),
        )
    assert stale_successor.value.code == "publication_request_conflict"


def test_guarded_tree_recovery_preserves_post_publish_edits(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    _run_crashing_publication(destination, phase="published", exit_code=84)
    later_edit = destination / "operator.txt"
    later_edit.write_text("keep\n", encoding="utf-8")

    assert recover_guarded_tree(destination) == "published"

    assert later_edit.read_text(encoding="utf-8") == "keep\n"
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_recovery_rejects_reused_published_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "published"
    publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=_populate,
    )
    destination_stat = destination.stat(follow_symlinks=False)
    capture_stable_identity = publication._capture_stable_identity

    def report_reused_inode(
        value: os.stat_result,
        *,
        path: Path | None = None,
        descriptor: int | None = None,
        dir_fd: int | None = None,
        name: str | None = None,
    ) -> publication._Identity:
        identity = capture_stable_identity(
            value,
            path=path,
            descriptor=descriptor,
            dir_fd=dir_fd,
            name=name,
        )
        if (
            identity.device == destination_stat.st_dev
            and identity.inode == destination_stat.st_ino
            and identity.kind == stat.S_IFMT(destination_stat.st_mode)
        ):
            assert identity.incarnation is not None
            return replace(identity, incarnation=identity.incarnation + 1)
        return identity

    monkeypatch.setattr(publication, "_capture_stable_identity", report_reused_inode)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "publication_conflict"
    assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _receipt_path(destination).is_file()


def test_guarded_tree_rejects_non_name_surrogate_windows_reparse_point() -> None:
    value = cast(
        "os.stat_result",
        SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600,
            st_file_attributes=0x400,
            st_reparse_tag=0x8000001A,
        ),
    )

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publication._require_supported_tree_entry(value, path="cache.dat")

    assert exc_info.value.code == "unsafe_tree_entry"


@pytest.mark.parametrize(
    ("consumer", "request_digest", "policy"),
    [
        ("other", _REQUEST_DIGEST, DestinationPolicy.ABSENT_OR_EMPTY),
        ("test", f"sha256:{'2' * 64}", DestinationPolicy.ABSENT_OR_EMPTY),
        ("test", _REQUEST_DIGEST, DestinationPolicy.REPLACE_DIRECTORY),
    ],
)
def test_guarded_tree_publication_conflicting_retry_does_not_repopulate(
    tmp_path: Path,
    consumer: str,
    request_digest: str,
    policy: DestinationPolicy,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    _run_crashing_publication(destination, phase="published", exit_code=81)

    def must_not_populate(_staging: GuardedTreeStage) -> None:
        pytest.fail("a conflicting retry must not populate another tree")

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        publish_guarded_tree(
            destination,
            consumer=consumer,
            request_digest=request_digest,
            policy=policy,
            populate=must_not_populate,
        )

    assert exc_info.value.code == "publication_request_conflict"
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()
    assert len(_owned_paths(tmp_path)) == 1

    result = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=must_not_populate,
    )
    assert result.recovered
    assert _journal_path(destination).is_file()
    assert _owned_paths(tmp_path) == []


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX signal delivery semantics")
def test_guarded_tree_publication_propagates_real_interrupt_after_settlement(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    script = textwrap.dedent(
        f"""
        import os
        import signal
        from pathlib import Path
        from cayu.cli import _guarded_tree_publication as publication
        from cayu.cli._guarded_tree_publication import DestinationPolicy, publish_guarded_tree

        destination = Path({str(destination)!r})
        def populate(staging):
            staging.write_text('value.txt', 'new\\n')
        def fault(phase):
            if phase == 'commit_intent_synced':
                os.kill(os.getpid(), signal.SIGINT)
        publication._publication_fault = fault
        try:
            publish_guarded_tree(
                destination,
                consumer='test',
                request_digest={_REQUEST_DIGEST!r},
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=populate,
            )
        except KeyboardInterrupt:
            raise SystemExit(75)
        raise SystemExit(76)
        """
    )
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )

    assert completed.returncode == 75
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX signal delivery semantics")
def test_guarded_tree_publication_keeps_real_interrupt_authoritative_during_rollback(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    script = textwrap.dedent(
        f"""
        import os
        import signal
        from pathlib import Path
        from cayu.cli import _guarded_tree_publication as publication
        from cayu.cli._guarded_tree_publication import DestinationPolicy, publish_guarded_tree

        destination = Path({str(destination)!r})
        def populate(staging):
            staging.write_text('value.txt', 'partial\\n')
            raise RuntimeError('primary population failure')
        def fault(phase):
            if phase == 'rollback_cleanup_sealed':
                os.kill(os.getpid(), signal.SIGINT)
        publication._publication_fault = fault
        try:
            publish_guarded_tree(
                destination,
                consumer='test',
                request_digest={_REQUEST_DIGEST!r},
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=populate,
            )
        except KeyboardInterrupt as signal_error:
            if not isinstance(signal_error.__cause__, RuntimeError):
                raise SystemExit(77)
            if str(signal_error.__cause__) != 'primary population failure':
                raise SystemExit(78)
            raise SystemExit(75)
        raise SystemExit(76)
        """
    )
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )

    assert completed.returncode == 75
    assert not destination.exists()
    assert recover_guarded_tree(destination) == "rolled_back"
    assert _owned_paths(tmp_path) == []


@pytest.mark.parametrize(
    "process_signal",
    (KeyboardInterrupt(), SystemExit(19), GeneratorExit()),
)
def test_settlement_process_control_remains_authoritative_with_ordered_evidence(
    process_signal: BaseException,
) -> None:
    primary = RuntimeError("primary")
    additional_cleanup = OSError("additional cleanup")
    settlement = BaseExceptionGroup(
        "settlement",
        (additional_cleanup, BaseExceptionGroup("signal", (process_signal,))),
    )

    with pytest.raises(type(process_signal)) as exc_info:
        publication._raise_primary_with_settlement_failure(primary, settlement)

    assert exc_info.value is process_signal
    cause = exc_info.value.__cause__
    assert isinstance(cause, BaseExceptionGroup)
    evidence = list(publication.iter_exception_tree(cause))
    assert sum(candidate is primary for candidate in evidence) == 1
    assert sum(candidate is additional_cleanup for candidate in evidence) == 1
    assert all(candidate is not process_signal for candidate in evidence)


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX signal delivery semantics")
def test_guarded_tree_publication_interrupt_after_commit_retains_exact_receipt(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    script = textwrap.dedent(
        f"""
        import os
        import signal
        from pathlib import Path
        from cayu.cli import _guarded_tree_publication as publication
        from cayu.cli._guarded_tree_publication import DestinationPolicy, publish_guarded_tree

        destination = Path({str(destination)!r})
        def populate(staging):
            staging.write_text('value.txt', 'new\\n')
        def fault(phase):
            if phase == 'published':
                os.kill(os.getpid(), signal.SIGINT)
        publication._publication_fault = fault
        try:
            publish_guarded_tree(
                destination,
                consumer='test',
                request_digest={_REQUEST_DIGEST!r},
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=populate,
            )
        except KeyboardInterrupt:
            raise SystemExit(85)
        raise SystemExit(86)
        """
    )
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )

    assert completed.returncode == 85
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()
    assert _owned_paths(tmp_path) == []

    result = publish_guarded_tree(
        destination,
        consumer="test",
        request_digest=_REQUEST_DIGEST,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        populate=lambda _staging: pytest.fail("exact retry must not repopulate"),
    )
    assert result.recovered
    assert _journal_path(destination).is_file()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows console signal delivery")
def test_guarded_tree_publication_propagates_real_windows_interrupt_after_settlement(
    tmp_path: Path,
) -> None:
    import signal

    destination = tmp_path / "published"
    destination.mkdir()
    ready = tmp_path / "interrupt-ready"
    script = textwrap.dedent(
        f"""
        import signal
        import time
        from pathlib import Path
        from cayu.cli import _guarded_tree_publication as publication
        from cayu.cli._guarded_tree_publication import DestinationPolicy, publish_guarded_tree

        destination = Path({str(destination)!r})
        ready = Path({str(ready)!r})
        signal.signal(signal.SIGBREAK, lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))
        def populate(staging):
            staging.write_text('value.txt', 'new\\n')
        def fault(phase):
            if phase == 'commit_intent_synced':
                ready.write_text('ready\\n', encoding='utf-8')
                while True:
                    time.sleep(0.05)
        publication._publication_fault = fault
        try:
            publish_guarded_tree(
                destination,
                consumer='test',
                request_digest={_REQUEST_DIGEST!r},
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=populate,
            )
        except KeyboardInterrupt:
            raise SystemExit(75)
        raise SystemExit(76)
        """
    )
    repository_root = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        creationflags=cast("int", subprocess.__dict__["CREATE_NEW_PROCESS_GROUP"]),
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            if process.poll() is not None:
                pytest.fail(f"publisher exited before interruption with {process.returncode}")
            if time.monotonic() >= deadline:
                pytest.fail("publisher did not reach the interrupt barrier")
            time.sleep(0.01)
        process.send_signal(cast("int", signal.__dict__["CTRL_BREAK_EVENT"]))
        assert process.wait(timeout=10) == 75
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert not _journal_path(destination).exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_recovery_rejects_changed_staging_authority(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    _run_crashing_publication(destination, phase="stage_synced", exit_code=77)
    journal = _journal_path(destination)
    entries = journal.read_text(encoding="ascii").splitlines()
    previous = json.loads(entries[-1])
    forged = {key: value for key, value in previous.items() if key != "entry_sha256"}
    forged["sequence"] += 1
    forged["previous_sha256"] = previous["entry_sha256"]
    forged["phase"] = "commit_intent"
    forged["stage_identity"][1] += 1
    canonical = json.dumps(
        forged,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    forged["entry_sha256"] = hashlib.sha256(canonical).hexdigest()
    with journal.open("a", encoding="ascii") as output:
        output.write(
            json.dumps(
                forged,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "invalid_publication_journal"
    assert journal.is_file()
    assert destination.is_dir()
    assert len(_owned_paths(tmp_path)) == 1


def test_guarded_tree_recovery_preserves_all_trees_on_post_publish_name_conflict(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    _run_crashing_publication(destination, phase="tree_renamed", exit_code=79)
    journal = _journal_path(destination)
    initial = json.loads(journal.read_text(encoding="ascii").splitlines()[0])
    conflicting_stage = tmp_path / initial["stage_name"]
    conflicting_stage.mkdir()
    marker = conflicting_stage / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    backup = tmp_path / initial["backup_name"]
    backup_identity = backup.stat()

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "publication_conflict"
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
    assert marker.read_text(encoding="utf-8") == "keep\n"
    assert os.path.samestat(backup_identity, backup.stat())
    assert journal.is_file()


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX subprocess exit semantics")
def test_guarded_tree_recovery_resumes_partial_owned_cleanup(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old-a.txt").write_text("a\n", encoding="utf-8")
    (destination / "old-b.txt").write_text("b\n", encoding="utf-8")
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from cayu.cli import _guarded_tree_publication as publication
        from cayu.cli._guarded_tree_publication import DestinationPolicy, publish_guarded_tree

        destination = Path({str(destination)!r})
        removed = False
        original_remove = publication._remove_directory_contents_from_fd
        def stop_during_cleanup(
            descriptor,
            *,
            path,
            flags,
            authority=None,
            prefix=publication.PurePosixPath(),
            linux_mount_points=None,
        ):
            global removed
            if path.name.startswith('.cayu-tree-cleanup-') and not removed:
                removed = True
                with os.scandir(descriptor) as entries:
                    name = next(iter(entries)).name
                os.unlink(name, dir_fd=descriptor)
                os.fsync(descriptor)
                os._exit(78)
            original_remove(
                descriptor,
                path=path,
                flags=flags,
                authority=authority,
                prefix=prefix,
                linux_mount_points=linux_mount_points,
            )
        publication._remove_directory_contents_from_fd = stop_during_cleanup
        def populate(staging):
            staging.write_text('new.txt', 'new\\n')
        publish_guarded_tree(
            destination,
            consumer='test',
            request_digest={_REQUEST_DIGEST!r},
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=populate,
        )
        """
    )
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )

    assert completed.returncode == 78
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert len(list(tmp_path.glob(".cayu-tree-cleanup-*"))) == 1

    assert recover_guarded_tree(destination) == "published"

    assert (destination / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert _journal_path(destination).is_file()
    assert _owned_paths(tmp_path) == []


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows cleanup handles")
def test_guarded_tree_recovery_resumes_partial_owned_windows_cleanup(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old-a.txt").write_text("a\n", encoding="utf-8")
    (destination / "old-b.txt").write_text("b\n", encoding="utf-8")
    script = textwrap.dedent(
        f"""
        import os
        from pathlib import Path
        from cayu.cli import _guarded_tree_publication as publication
        from cayu.cli._guarded_tree_publication import DestinationPolicy, publish_guarded_tree

        destination = Path({str(destination)!r})
        removed = False
        original_delete = publication._delete_windows_entry_by_handle
        def stop_during_cleanup(path, *, expected, authority=None, prefix=None):
            global removed
            if path.name.startswith('.cayu-tree-cleanup-') and not removed:
                removed = True
                child = next(path.iterdir())
                child_identity = publication._capture_stable_identity(
                    child.stat(follow_symlinks=False),
                    path=child,
                )
                original_delete(
                    child,
                    expected=child_identity,
                    authority=authority,
                    prefix=prefix,
                )
                os._exit(78)
            original_delete(
                path,
                expected=expected,
                authority=authority,
                prefix=prefix,
            )
        publication._delete_windows_entry_by_handle = stop_during_cleanup
        def populate(staging):
            staging.write_text('new.txt', 'new\\n')
        publish_guarded_tree(
            destination,
            consumer='test',
            request_digest={_REQUEST_DIGEST!r},
            policy=DestinationPolicy.REPLACE_DIRECTORY,
            populate=populate,
        )
        """
    )
    repository_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )

    assert completed.returncode == 78
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new\n"
    assert len(list(tmp_path.glob(".cayu-tree-cleanup-*"))) == 1

    assert recover_guarded_tree(destination) == "published"

    assert _journal_path(destination).is_file()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_publication_serializes_cross_process_writers(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    ready = tmp_path / "first-ready"
    release = tmp_path / "release-first"
    repository_root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": str(repository_root / "src")}
    first_script = textwrap.dedent(
        f"""
        import time
        from pathlib import Path
        from cayu.cli._guarded_tree_publication import (
            DestinationPolicy,
            publish_guarded_tree,
        )

        destination = Path({str(destination)!r})
        ready = Path({str(ready)!r})
        release = Path({str(release)!r})
        def populate(staging):
            staging.write_text('winner.txt', 'first\\n')
            ready.write_text('ready\\n', encoding='utf-8')
            deadline = time.monotonic() + 10
            while not release.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError('release was not published')
                time.sleep(0.01)
        publish_guarded_tree(
            destination,
            consumer='test',
            request_digest={_REQUEST_DIGEST!r},
            policy=DestinationPolicy.ABSENT_OR_EMPTY,
            populate=populate,
        )
        """
    )
    second_script = textwrap.dedent(
        f"""
        from pathlib import Path
        from cayu.cli._guarded_tree_publication import (
            DestinationPolicy,
            GuardedTreePublicationError,
            publish_guarded_tree,
        )

        destination = Path({str(destination)!r})
        def populate(staging):
            staging.write_text('winner.txt', 'second\\n')
        try:
            publish_guarded_tree(
                destination,
                consumer='other',
                request_digest={f"sha256:{'2' * 64}"!r},
                policy=DestinationPolicy.ABSENT_OR_EMPTY,
                populate=populate,
            )
        except GuardedTreePublicationError as error:
            if error.code == 'publication_request_conflict':
                raise SystemExit(82)
            raise
        raise SystemExit(83)
        """
    )
    first = subprocess.Popen(
        [sys.executable, "-c", first_script],
        cwd=repository_root,
        env=environment,
    )
    second: subprocess.Popen[bytes] | None = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            if first.poll() is not None:
                pytest.fail(f"first publisher exited early with {first.returncode}")
            if time.monotonic() >= deadline:
                pytest.fail("first publisher did not reach its population barrier")
            time.sleep(0.01)
        second = subprocess.Popen(
            [sys.executable, "-c", second_script],
            cwd=repository_root,
            env=environment,
        )
        time.sleep(0.2)
        assert second.poll() is None
        assert len(_owned_paths(tmp_path)) == 1

        release.write_text("release\n", encoding="utf-8")
        assert first.wait(timeout=10) == 0
        assert second.wait(timeout=10) == 82
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)

    assert (destination / "winner.txt").read_text(encoding="utf-8") == "first\n"
    assert _journal_path(destination).is_file()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_recovery_rejects_malformed_journal_without_deleting_it(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    journal = _journal_path(destination)
    journal.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "invalid_publication_journal"
    assert journal.read_text(encoding="utf-8") == "not-json\n"


@pytest.mark.parametrize(
    "content",
    (
        pytest.param(b'{"value":' + (b"1" * 5000) + b"}\n", id="integer-digit-limit"),
        pytest.param(
            b'{"value":' + (b"[" * 2048) + b"0" + (b"]" * 2048) + b"}\n",
            id="nesting-limit",
        ),
    ),
)
def test_guarded_tree_recovery_translates_bounded_json_parser_failures(
    tmp_path: Path,
    content: bytes,
) -> None:
    destination = tmp_path / "published"
    journal = _journal_path(destination)
    journal.write_bytes(content)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "invalid_publication_journal"
    assert journal.read_bytes() == content
    assert not destination.exists()
    assert _owned_paths(tmp_path) == []


def test_guarded_tree_recovery_classifies_invalid_unicode_journal_metadata(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    token = "a" * 32
    record = publication._Record(
        consumer="\ud800",
        request_digest=_REQUEST_DIGEST,
        predecessor_request_digest=None,
        token=token,
        destination_name=destination.name,
        policy=DestinationPolicy.ABSENT_OR_EMPTY,
        parent_identity=publication._capture_parent(tmp_path),
        original_identity=None,
        original_sha256=None,
        stage_name=f".cayu-tree-stage-{token}",
        stage_identity=None,
        stage_sha256=None,
        backup_name=f".cayu-tree-backup-{token}",
        cleanup_manifest_identity=None,
        cleanup_manifest_sha256=None,
        phase=publication._Phase.PREPARED,
    )
    content, _entry_sha256 = publication._journal_entry(record, previous_sha256=None)
    journal = _active_journal_path(destination)
    journal.write_bytes(content)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "invalid_publication_journal"
    assert journal.read_bytes() == content
    assert not destination.exists()
    assert _owned_paths(tmp_path) == []


def _replace_cleanup_manifest_content(
    destination: Path,
    content: bytes,
) -> tuple[Path, Path]:
    manifest = next(destination.parent.glob(".cayu-tree-authority-*.json"))
    manifest.write_bytes(content)
    journal = _active_journal_path(destination)
    journal_entries = [json.loads(line) for line in journal.read_bytes().splitlines()]
    terminal = journal_entries[-1]
    terminal["cleanup_manifest_sha256"] = f"sha256:{hashlib.sha256(content).hexdigest()}"
    terminal_payload = {key: value for key, value in terminal.items() if key != "entry_sha256"}
    terminal["entry_sha256"] = hashlib.sha256(
        json.dumps(
            terminal_payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    journal.write_bytes(
        b"".join(
            json.dumps(
                entry,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            + b"\n"
            for entry in journal_entries
        )
    )
    return manifest, journal


@pytest.mark.parametrize(
    "content",
    (
        pytest.param(b'{"value":' + (b"1" * 5000) + b"}\n", id="integer-digit-limit"),
        pytest.param(
            b'{"value":' + (b"[" * 2048) + b"0" + (b"]" * 2048) + b"}\n",
            id="nesting-limit",
        ),
    ),
)
def test_guarded_tree_cleanup_translates_bounded_json_parser_failures(
    tmp_path: Path,
    content: bytes,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    _run_crashing_replacement_publication(
        destination,
        phase="cleanup_sealed",
        exit_code=99,
    )
    manifest, journal = _replace_cleanup_manifest_content(destination, content)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "cleanup_conflict"
    assert journal.is_file()
    assert manifest.read_bytes() == content
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"


def test_guarded_tree_recovery_classifies_invalid_unicode_cleanup_metadata(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "old.txt").write_text("old\n", encoding="utf-8")
    _run_crashing_replacement_publication(
        destination,
        phase="cleanup_sealed",
        exit_code=95,
    )
    manifest = next(tmp_path.glob(".cayu-tree-authority-*.json"))
    manifest_value = json.loads(manifest.read_bytes())
    manifest_value["entries"][0]["path"] = "\ud800"
    manifest_content = (
        json.dumps(
            manifest_value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    manifest, journal = _replace_cleanup_manifest_content(destination, manifest_content)

    with pytest.raises(GuardedTreePublicationError) as exc_info:
        recover_guarded_tree(destination)

    assert exc_info.value.code == "cleanup_conflict"
    assert journal.is_file()
    assert manifest.is_file()
    assert (destination / "value.txt").read_text(encoding="utf-8") == "new\n"
