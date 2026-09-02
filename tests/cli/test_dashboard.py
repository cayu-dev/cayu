from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from importlib.resources import files
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cayu._server_contract_version import SERVER_CONTRACT_VERSION
from cayu.cli import _guarded_tree_publication as publication_cli
from cayu.cli import _version, main
from cayu.cli import dashboard as dashboard_cli
from cayu.cli.dashboard import DashboardSourceError, eject_dashboard_source


def _packaged_bundle() -> bytes:
    resource_root = files("cayu.data").joinpath("dashboard_source")
    bundles = [resource for resource in resource_root.iterdir() if resource.name.endswith(".zip")]
    assert len(bundles) == 1
    return bundles[0].read_bytes()


def _rewrite_bundle(
    transform: Callable[[zipfile.ZipInfo, bytes], tuple[zipfile.ZipInfo, bytes]],
    *,
    appended: tuple[zipfile.ZipInfo, bytes] | None = None,
) -> bytes:
    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(_packaged_bundle())) as source,
        zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as target,
    ):
        for member in source.infolist():
            rewritten, content = transform(member, source.read(member))
            target.writestr(rewritten, content)
        if appended is not None:
            target.writestr(*appended)
    return output.getvalue()


def _ordinary_member(name: str) -> zipfile.ZipInfo:
    member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    member.create_system = 3
    member.compress_type = zipfile.ZIP_STORED
    member.external_attr = (stat.S_IFREG | 0o644) << 16
    return member


@contextmanager
def _windows_directory_write_handle(path: Path) -> Iterator[None]:
    if os.name != "nt":
        pytest.fail("Windows write handles require Windows")

    import ctypes
    from ctypes import wintypes

    windows_ctypes: Any = ctypes
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    handle = create_file(
        str(path),
        0x40000000,
        0x1 | 0x2 | 0x4,
        None,
        0x3,
        0x00200000 | 0x02000000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise windows_ctypes.WinError(windows_ctypes.get_last_error())
    try:
        yield
    finally:
        if not close_handle(handle):
            raise windows_ctypes.WinError(windows_ctypes.get_last_error())


def test_dashboard_eject_requires_explicit_destination(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["dashboard", "eject"])

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "cayu dashboard eject" in error
    assert "destination" in error.lower()
    assert "invalid choice" not in error


def test_dashboard_eject_rejects_unrecoverable_tree_size_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    monkeypatch.setattr(publication_cli, "_TREE_ENTRY_LIMIT", 1)

    with pytest.raises(DashboardSourceError, match="entry limit"):
        eject_dashboard_source(destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".cayu-tree-*"))


@pytest.mark.skipif(os.name == "nt", reason="colon is reserved by the Win32 namespace")
def test_dashboard_eject_preserves_native_posix_destination_names(tmp_path: Path) -> None:
    destination = tmp_path / "dashboard:source"

    result = eject_dashboard_source(destination)

    assert result.destination == destination
    assert (destination / "package.json").is_file()


def test_dashboard_eject_materializes_version_matched_editable_project(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "my dashboard"

    assert main(["dashboard", "eject", str(destination)]) == 0

    for relative in (
        "README.md",
        "LICENSE",
        "NOTICE",
        "REDISTRIBUTION.md",
        "cayu-dashboard-source.json",
        "package.json",
        "package-lock.json",
        "vite.config.ts",
        "tsconfig.json",
        "biome.json",
        "server-openapi.json",
        "scripts/finalize-third-party-licenses.mjs",
        "scripts/generate-api-types.py",
        "scripts/run-python.mjs",
        "src/main.tsx",
        "src/lib/generated/server-api/types.gen.ts",
        "tests/dashboard-capabilities.test.mjs",
        "third_party/shadcn-ui.LICENSE",
    ):
        assert (destination / relative).is_file(), relative

    manifest = json.loads((destination / "cayu-dashboard-source.json").read_text(encoding="utf-8"))
    assert manifest["cayu_version"] == _version()
    assert manifest["server_contract_version"] == SERVER_CONTRACT_VERSION
    assert manifest["source_digest"].startswith("sha256:")
    package = json.loads((destination / "package.json").read_text(encoding="utf-8"))
    assert "build:package" not in package["scripts"]
    assert not any("../" in command for command in package["scripts"].values())
    assert package["scripts"]["generate:api"] == (
        "node scripts/run-python.mjs scripts/generate-api-types.py"
    )
    assert package["scripts"]["check:api"] == (
        "node scripts/run-python.mjs scripts/generate-api-types.py --check"
    )

    output = capsys.readouterr().out
    assert f"installed Cayu version: {_version()}" in output
    assert f"dashboard source version: {_version()}" in output
    assert f"dashboard server contract: v{SERVER_CONTRACT_VERSION}" in output
    assert f"Project directory: {destination}" in output
    assert f"cd {destination}" not in output
    assert "npm ci" in output
    assert "npm run dev" in output
    assert "npm run build" in output
    assert "DashboardConfig(directory=" in output
    assert "dashboard_dir=" in output
    assert 'Path("dist")' in output
    assert "my dashboard/dist" not in output


def test_dashboard_eject_exact_retry_recovers_published_process_death(tmp_path: Path) -> None:
    destination = tmp_path / "control-plane"
    repository_root = Path(__file__).resolve().parents[2]
    script = f"""
import os
from pathlib import Path
from cayu.cli import _guarded_tree_publication as publication
from cayu.cli.dashboard import eject_dashboard_source

destination = Path({str(destination)!r})
def fault(phase):
    if phase == 'settled':
        os._exit(87)
publication._publication_fault = fault
eject_dashboard_source(destination)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        env={**os.environ, "PYTHONPATH": str(repository_root / "src")},
        check=False,
    )

    assert completed.returncode == 87
    assert (destination / "package.json").is_file()
    assert list(tmp_path.glob(".cayu-tree-publication-*.jsonl"))

    result = eject_dashboard_source(destination)

    assert result.destination == destination
    assert result.manifest.cayu_version == _version()
    assert list(tmp_path.glob(".cayu-tree-publication-*.jsonl"))
    assert not list(tmp_path.glob(".cayu-tree-stage-*"))
    assert not list(tmp_path.glob(".cayu-tree-backup-*"))
    assert not list(tmp_path.glob(".cayu-tree-cleanup-*"))


@pytest.mark.parametrize(
    "destination_state",
    ("unchanged", "modified", "empty", "absent", "recreated_empty"),
)
def test_dashboard_eject_exact_retry_authenticates_published_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    destination_state: str,
) -> None:
    destination = tmp_path / "control-plane"
    eject_dashboard_source(destination)

    if destination_state == "modified":
        (destination / "package.json").write_text("modified\n", encoding="utf-8")
    elif destination_state == "empty":
        for child in destination.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    elif destination_state == "absent":
        shutil.rmtree(destination)
    elif destination_state == "recreated_empty":
        shutil.rmtree(destination)
        destination.mkdir()

    if destination_state in {"unchanged", "modified"}:
        monkeypatch.setattr(
            dashboard_cli,
            "_write_staging_tree",
            lambda *_args, **_kwargs: pytest.fail(
                "an unchanged or modified exact retry must not repopulate"
            ),
        )

    if destination_state == "modified":
        with pytest.raises(DashboardSourceError, match="destination must be empty"):
            eject_dashboard_source(destination)
        assert (destination / "package.json").read_text(encoding="utf-8") == "modified\n"
        return

    result = eject_dashboard_source(destination)

    assert result.destination == destination
    assert (destination / "package.json").is_file()


def test_dashboard_eject_request_identity_binds_raw_manifest_bytes(tmp_path: Path) -> None:
    original_bundle = _packaged_bundle()

    def compact_manifest(
        member: zipfile.ZipInfo,
        content: bytes,
    ) -> tuple[zipfile.ZipInfo, bytes]:
        if member.filename != "cayu-dashboard-source.json":
            return member, content
        manifest = json.loads(content)
        return member, json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    compact_bundle = _rewrite_bundle(compact_manifest)
    with (
        zipfile.ZipFile(io.BytesIO(original_bundle)) as original_archive,
        zipfile.ZipFile(io.BytesIO(compact_bundle)) as compact_archive,
    ):
        original_manifest = original_archive.read("cayu-dashboard-source.json")
        compact_manifest_bytes = compact_archive.read("cayu-dashboard-source.json")

    assert original_manifest != compact_manifest_bytes
    assert json.loads(original_manifest) == json.loads(compact_manifest_bytes)

    destination = tmp_path / "control-plane"
    eject_dashboard_source(destination, bundle_bytes=original_bundle)

    with pytest.raises(
        DashboardSourceError,
        match="terminal publication receipt does not authorize this successor request",
    ):
        eject_dashboard_source(destination, bundle_bytes=compact_bundle)

    assert (destination / "cayu-dashboard-source.json").read_bytes() == original_manifest
    assert not list(tmp_path.glob(".cayu-tree-stage-*"))
    assert not list(tmp_path.glob(".cayu-tree-backup-*"))
    assert not list(tmp_path.glob(".cayu-tree-cleanup-*"))


def test_dashboard_eject_reports_terminal_publication_boundary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"

    @contextmanager
    def fail_after_publication(*_args: object, **_kwargs: object) -> Iterator[None]:
        yield
        raise OSError("simulated publication lock release failure")

    monkeypatch.setattr(publication_cli, "cooperative_path_lock", fail_after_publication)

    with pytest.raises(DashboardSourceError, match="ownership boundary did not settle") as exc_info:
        eject_dashboard_source(destination)

    assert isinstance(exc_info.value.__cause__, publication_cli.GuardedTreePublicationError)
    assert isinstance(exc_info.value.__cause__.__cause__, OSError)
    assert (destination / "package.json").is_file()
    assert list(tmp_path.glob(".cayu-tree-publication-*.jsonl"))


def test_dashboard_eject_refuses_non_empty_destination_without_mutating_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "control-plane"
    destination.mkdir()
    marker = destination / "application-owned.txt"
    marker.write_text("keep me\n", encoding="utf-8")

    assert main(["dashboard", "eject", str(destination)]) == 1

    assert marker.read_text(encoding="utf-8") == "keep me\n"
    assert sorted(path.name for path in destination.iterdir()) == [marker.name]
    error = capsys.readouterr().err
    assert "destination must be empty" in error


def test_dashboard_eject_accepts_an_existing_empty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "control-plane"
    destination.mkdir()

    assert main(["dashboard", "eject", str(destination)]) == 0

    assert (destination / "package.json").is_file()


def test_dashboard_eject_requires_an_existing_destination_parent(tmp_path: Path) -> None:
    destination = tmp_path / "missing-parent" / "control-plane"

    with pytest.raises(DashboardSourceError, match="destination parent must already exist"):
        eject_dashboard_source(destination)

    assert not destination.parent.exists()


def test_dashboard_eject_does_not_create_through_link_appearing_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linked_parent = tmp_path / "linked-parent"
    destination = linked_parent / "nested" / "control-plane"
    target = tmp_path / "target"
    target.mkdir()
    validate_destination = dashboard_cli._validate_destination

    def validate_then_link(path: Path) -> Path:
        validated = validate_destination(path)
        try:
            linked_parent.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are unavailable")
        return validated

    monkeypatch.setattr(dashboard_cli, "_validate_destination", validate_then_link)

    with pytest.raises(DashboardSourceError, match="symbolic link or junction"):
        eject_dashboard_source(destination)

    assert list(target.iterdir()) == []


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX creation")
def test_dashboard_eject_anchors_staging_during_transient_parent_link_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    displaced_parent = tmp_path / "displaced-parent"
    target = tmp_path / "target"
    target.mkdir()
    destination = parent / "control-plane"
    mkdir = os.mkdir
    swapped = False

    def mkdir_during_transient_parent_swap(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal swapped
        candidate_name = Path(os.fsdecode(path)).name
        if not swapped and candidate_name.startswith(".cayu-tree-stage-"):
            swapped = True
            parent.rename(displaced_parent)
            try:
                parent.symlink_to(target, target_is_directory=True)
            except OSError:
                displaced_parent.rename(parent)
                pytest.skip("symlinks are unavailable")
            try:
                mkdir(path, mode=mode, dir_fd=dir_fd)
            finally:
                parent.unlink()
                displaced_parent.rename(parent)
            return
        mkdir(path, mode=mode, dir_fd=dir_fd)

    monkeypatch.setattr(dashboard_cli.os, "mkdir", mkdir_during_transient_parent_swap)

    eject_dashboard_source(destination)

    assert swapped
    assert (destination / "package.json").is_file()
    assert list(target.iterdir()) == []


def test_dashboard_eject_refuses_destination_changed_during_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    destination.mkdir()
    marker = destination / "application-owned.txt"
    write_staging_tree = dashboard_cli._write_staging_tree

    def write_staging_tree_then_mutate_destination(
        staging: Path,
        contents: dict[str, bytes],
        *,
        staging_guard: dashboard_cli._StagingGuard,
    ) -> None:
        write_staging_tree(staging, contents, staging_guard=staging_guard)
        marker.write_text("keep me\n", encoding="utf-8")

    monkeypatch.setattr(
        dashboard_cli,
        "_write_staging_tree",
        write_staging_tree_then_mutate_destination,
    )

    with pytest.raises(DashboardSourceError, match="destination content changed"):
        eject_dashboard_source(destination)

    assert marker.read_text(encoding="utf-8") == "keep me\n"
    assert sorted(path.name for path in destination.iterdir()) == [marker.name]
    assert not list(tmp_path.glob(".cayu-tree-*"))


def test_dashboard_eject_refuses_parent_changed_during_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    destination = parent / "control-plane"
    redirected_parent = tmp_path / "redirected-parent"
    write_staging_tree = dashboard_cli._write_staging_tree

    def write_staging_tree_then_swap_parent(
        staging: Path,
        contents: dict[str, bytes],
        *,
        staging_guard: dashboard_cli._StagingGuard,
    ) -> None:
        write_staging_tree(staging, contents, staging_guard=staging_guard)
        parent.rename(redirected_parent)
        try:
            parent.symlink_to(redirected_parent, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are unavailable")

    monkeypatch.setattr(
        dashboard_cli,
        "_write_staging_tree",
        write_staging_tree_then_swap_parent,
    )

    with pytest.raises(DashboardSourceError, match="publication parent changed"):
        eject_dashboard_source(destination)

    assert not (redirected_parent / "control-plane").exists()


def test_dashboard_eject_does_not_write_through_replaced_staging_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    target_mode = stat.S_IMODE(target.stat().st_mode)
    owned_staging = tmp_path / "owned-staging"
    write_staging_tree = dashboard_cli._write_staging_tree

    def replace_staging_before_first_write(
        staging: Path,
        contents: dict[str, bytes],
        *,
        staging_guard: dashboard_cli._StagingGuard,
    ) -> None:
        staging.rename(owned_staging)
        try:
            staging.symlink_to(target, target_is_directory=True)
        except OSError:
            owned_staging.rename(staging)
            pytest.skip("symlinks are unavailable")
        write_staging_tree(
            staging,
            contents,
            staging_guard=staging_guard,
        )

    monkeypatch.setattr(
        dashboard_cli,
        "_write_staging_tree",
        replace_staging_before_first_write,
    )

    with pytest.raises(DashboardSourceError, match="staging directory changed"):
        eject_dashboard_source(destination)

    assert not destination.exists()
    assert list(target.iterdir()) == []
    assert stat.S_IMODE(target.stat().st_mode) == target_mode
    assert owned_staging.is_dir()
    assert list(owned_staging.iterdir()) == []
    replacement_staging = next(tmp_path.glob(".cayu-tree-stage-*"))
    assert replacement_staging.is_symlink()


def test_dashboard_eject_authenticates_stage_before_callback_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    displaced_stage = tmp_path / "displaced-stage"
    replacement_stage: Path | None = None

    def replace_stage_at_callback_handoff(phase: str) -> None:
        nonlocal replacement_stage
        if phase != "stage_created":
            return
        stages = list(tmp_path.glob(".cayu-tree-stage-*"))
        assert len(stages) == 1
        replacement_stage = stages[0]
        replacement_stage.rename(displaced_stage)
        replacement_stage.mkdir(mode=0o700)

    monkeypatch.setattr(
        publication_cli,
        "_publication_fault",
        replace_stage_at_callback_handoff,
    )

    with pytest.raises(DashboardSourceError, match="staging directory changed"):
        eject_dashboard_source(destination)

    assert replacement_stage is not None
    assert not destination.exists()
    assert displaced_stage.is_dir()
    assert list(displaced_stage.iterdir()) == []
    assert replacement_stage.is_dir()
    assert list(replacement_stage.iterdir()) == []


def test_dashboard_eject_refuses_replaced_staging_tree_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    owned_staging = tmp_path / "owned-staging"
    replacement_marker_name = "replacement-owned.txt"
    replaced = False

    def replace_staging_after_seal(phase: str) -> None:
        nonlocal replaced
        if phase != "stage_synced" or replaced:
            return
        replaced = True
        staging = next(tmp_path.glob(".cayu-tree-stage-*"))
        staging.rename(owned_staging)
        staging.mkdir()
        (staging / replacement_marker_name).write_text("preserve me\n", encoding="utf-8")

    monkeypatch.setattr(publication_cli, "_publication_fault", replace_staging_after_seal)

    with pytest.raises(DashboardSourceError, match="staging identity") as exc_info:
        eject_dashboard_source(destination)

    assert not destination.exists()
    assert replaced
    replacement_staging = next(tmp_path.glob(".cayu-tree-stage-*"))
    assert (replacement_staging / replacement_marker_name).read_text(encoding="utf-8") == (
        "preserve me\n"
    )
    assert (owned_staging / "package.json").is_file()
    assert any("remains recoverable" in note for note in getattr(exc_info.value, "__notes__", ()))


@pytest.mark.parametrize("existing_destination", [False, True])
def test_dashboard_eject_preserves_staging_replaced_inside_final_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_destination: bool,
) -> None:
    destination = tmp_path / "control-plane"
    original_destination: os.stat_result | None = None
    if existing_destination:
        destination.mkdir()
        original_destination = destination.stat()
    owned_staging = tmp_path / "owned-staging"
    replacement_marker_name = "replacement-owned.txt"
    rename_name_no_replace = publication_cli._rename_name_no_replace

    def replace_staging_during_publish(
        parent: publication_cli._Parent,
        source: str,
        target: str,
    ) -> None:
        if source.startswith(".cayu-tree-stage-") and target == destination.name:
            staging = tmp_path / source
            staging.rename(owned_staging)
            staging.mkdir()
            (staging / replacement_marker_name).write_text("preserve me\n", encoding="utf-8")
        rename_name_no_replace(parent, source, target)

    monkeypatch.setattr(
        publication_cli,
        "_rename_name_no_replace",
        replace_staging_during_publish,
    )

    with pytest.raises(DashboardSourceError, match="staging identity changed") as exc_info:
        eject_dashboard_source(destination)

    replacement_staging = next(tmp_path.glob(".cayu-tree-stage-*"))
    assert (replacement_staging / replacement_marker_name).read_text(encoding="utf-8") == (
        "preserve me\n"
    )
    assert (owned_staging / "package.json").is_file()
    if original_destination is None:
        assert not destination.exists()
    else:
        assert os.path.samestat(original_destination, destination.stat())
    assert any("remains recoverable" in note for note in getattr(exc_info.value, "__notes__", ()))
    assert "affected paths" in str(exc_info.value)
    publication_error = exc_info.value.__cause__
    assert isinstance(publication_error, publication_cli.GuardedTreePublicationError)
    settlement_error = publication_error.__cause__
    assert isinstance(settlement_error, publication_cli.GuardedTreePublicationError)
    assert set(settlement_error.paths) >= {destination.name, replacement_staging.name}


@pytest.mark.skipif(os.name == "nt", reason="exercises descriptor-relative POSIX publication")
def test_dashboard_eject_anchors_publication_to_validated_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    displaced_parent = tmp_path / "displaced-parent"
    destination = parent / "control-plane"
    swapped = False

    def replace_parent_after_staging(phase: str) -> None:
        nonlocal swapped
        if not swapped and phase == "stage_synced":
            swapped = True
            parent.rename(displaced_parent)
            parent.mkdir()

    monkeypatch.setattr(
        publication_cli,
        "_publication_fault",
        replace_parent_after_staging,
    )

    with pytest.raises(DashboardSourceError, match="publication parent changed"):
        eject_dashboard_source(destination)

    assert swapped
    assert not destination.exists()
    assert not (displaced_parent / destination.name).exists()
    assert list(parent.iterdir()) == []


def test_dashboard_eject_refuses_link_introduced_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    target = tmp_path / "target"
    target.mkdir()
    introduced_link = False

    def publish_after_introducing_link(phase: str) -> None:
        nonlocal introduced_link
        if phase != "stage_synced":
            return
        try:
            destination.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are unavailable")
        introduced_link = True

    monkeypatch.setattr(
        publication_cli,
        "_publication_fault",
        publish_after_introducing_link,
    )

    with pytest.raises(DashboardSourceError, match="destination appeared"):
        eject_dashboard_source(destination)

    assert introduced_link
    assert destination.is_symlink()
    assert list(target.iterdir()) == []
    assert len(list(tmp_path.glob(".cayu-tree-stage-*"))) == 1


def test_dashboard_eject_refuses_to_clean_replaced_staging_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    owned_staging = tmp_path / "owned-staging"
    replacement_marker_name = "replacement-owned.txt"
    write_staging_tree = dashboard_cli._write_staging_tree

    def replace_staging_before_failure(
        staging: Path,
        contents: dict[str, bytes],
        *,
        staging_guard: dashboard_cli._StagingGuard,
    ) -> None:
        write_staging_tree(staging, contents, staging_guard=staging_guard)
        staging.rename(owned_staging)
        staging.mkdir()
        (staging / replacement_marker_name).write_text("preserve me\n", encoding="utf-8")
        raise OSError("injected write failure")

    monkeypatch.setattr(
        dashboard_cli,
        "_write_staging_tree",
        replace_staging_before_failure,
    )

    with pytest.raises(OSError, match="injected write failure") as exc_info:
        eject_dashboard_source(destination)

    assert not destination.exists()
    replacement_staging = next(tmp_path.glob(".cayu-tree-stage-*"))
    assert (replacement_staging / replacement_marker_name).read_text(encoding="utf-8") == (
        "preserve me\n"
    )
    assert (owned_staging / "package.json").is_file()
    assert any("remains recoverable" in note for note in getattr(exc_info.value, "__notes__", ()))


@pytest.mark.skipif(
    os.name == "nt",
    reason="exercises descriptor-relative POSIX cleanup",
)
def test_dashboard_eject_refuses_cleanup_replacement_after_directory_is_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    owned_cleanup = tmp_path / "owned-cleanup"
    replacement_marker_name = "replacement-owned.txt"
    write_staging_tree = dashboard_cli._write_staging_tree
    remove_directory_contents_from_fd = publication_cli._remove_directory_contents_from_fd
    replaced_cleanup_root = False

    def fail_after_writing(
        staging: Path,
        contents: dict[str, bytes],
        *,
        staging_guard: dashboard_cli._StagingGuard,
    ) -> None:
        write_staging_tree(staging, contents, staging_guard=staging_guard)
        raise OSError("injected write failure")

    def replace_isolated_staging_after_cleanup_is_pinned(
        descriptor: int,
        *,
        path: Path,
        flags: int,
        authority: dict[str, publication_cli._CleanupEntry] | None = None,
        prefix: PurePosixPath | None = None,
        linux_mount_points: frozenset[str] | None = None,
    ) -> None:
        nonlocal replaced_cleanup_root
        if not replaced_cleanup_root:
            replaced_cleanup_root = True
            path.rename(owned_cleanup)
            path.mkdir()
            (path / replacement_marker_name).write_text("preserve me\n", encoding="utf-8")
        remove_directory_contents_from_fd(
            descriptor,
            path=path,
            flags=flags,
            authority=authority,
            prefix=PurePosixPath() if prefix is None else prefix,
            linux_mount_points=linux_mount_points,
        )

    monkeypatch.setattr(dashboard_cli, "_write_staging_tree", fail_after_writing)
    monkeypatch.setattr(
        publication_cli,
        "_remove_directory_contents_from_fd",
        replace_isolated_staging_after_cleanup_is_pinned,
    )

    with pytest.raises(OSError, match="injected write failure") as exc_info:
        eject_dashboard_source(destination)

    assert not destination.exists()
    replacement_cleanup = next(tmp_path.glob(".cayu-tree-cleanup-*"))
    assert (replacement_cleanup / replacement_marker_name).read_text(encoding="utf-8") == (
        "preserve me\n"
    )
    assert owned_cleanup.is_dir()
    assert (owned_cleanup / "package.json").is_file()
    assert any("remains recoverable" in note for note in getattr(exc_info.value, "__notes__", ()))


@pytest.mark.skipif(os.name != "nt", reason="requires a real Windows directory junction")
def test_dashboard_eject_refuses_windows_cleanup_junction_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    external_target = tmp_path / "external-target"
    external_target.mkdir()
    external_marker = external_target / "preserve-me.txt"
    external_marker.write_text("preserve me\n", encoding="utf-8")
    junction_probe = tmp_path / "junction-probe"
    probe_result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/j", str(junction_probe), str(external_target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe_result.returncode != 0:
        pytest.skip(f"directory junctions are unavailable: {probe_result.stderr.strip()}")
    junction_probe.rmdir()

    write_staging_tree = dashboard_cli._write_staging_tree
    deletion_handle = publication_cli._windows_deletion_handle
    displaced_scripts = tmp_path / "owned-cleanup-scripts"
    introduced_junction: Path | None = None

    def fail_after_writing(
        staging: Path,
        contents: dict[str, bytes],
        *,
        staging_guard: dashboard_cli._StagingGuard,
    ) -> None:
        write_staging_tree(staging, contents, staging_guard=staging_guard)
        raise OSError("injected write failure")

    @contextmanager
    def replace_directory_before_cleanup_handle_open(
        path: Path,
        *,
        read_content: bool = False,
    ) -> Iterator[tuple[int, Callable[[], None]]]:
        nonlocal introduced_junction
        if path.name == "scripts" and introduced_junction is None:
            path.rename(displaced_scripts)
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/j", str(path), str(external_target)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                displaced_scripts.rename(path)
                pytest.fail(f"could not create cleanup junction: {result.stderr.strip()}")
            introduced_junction = path
        with deletion_handle(path, read_content=read_content) as deletion_owner:
            yield deletion_owner

    monkeypatch.setattr(dashboard_cli, "_write_staging_tree", fail_after_writing)
    monkeypatch.setattr(
        publication_cli,
        "_windows_deletion_handle",
        replace_directory_before_cleanup_handle_open,
    )

    try:
        with pytest.raises(OSError, match="injected write failure") as exc_info:
            eject_dashboard_source(destination)

        assert introduced_junction is not None
        assert dashboard_cli._is_windows_reparse_point(
            introduced_junction.stat(follow_symlinks=False)
        )
        assert external_marker.read_text(encoding="utf-8") == "preserve me\n"
        assert displaced_scripts.is_dir()
        assert any(
            "remains recoverable" in note for note in getattr(exc_info.value, "__notes__", ())
        )
    finally:
        if introduced_junction is not None:
            try:
                junction_identity = introduced_junction.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if dashboard_cli._is_windows_reparse_point(junction_identity):
                    introduced_junction.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows sharing semantics")
def test_dashboard_eject_preserves_cleanup_with_preexisting_windows_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    write_staging_tree = dashboard_cli._write_staging_tree
    writer_handles = ExitStack()

    def fail_with_writer_held(
        staging: Path,
        contents: dict[str, bytes],
        *,
        staging_guard: dashboard_cli._StagingGuard,
    ) -> None:
        write_staging_tree(staging, contents, staging_guard=staging_guard)
        writer_handles.enter_context(_windows_directory_write_handle(staging))
        raise OSError("injected write failure")

    monkeypatch.setattr(dashboard_cli, "_write_staging_tree", fail_with_writer_held)

    with (
        writer_handles,
        pytest.raises(
            OSError,
            match="injected write failure",
        ) as exc_info,
    ):
        eject_dashboard_source(destination)

    cleanup_tree = next(tmp_path.glob(".cayu-tree-stage-*"))
    assert (cleanup_tree / "package.json").is_file()
    assert not destination.exists()
    assert any("remains recoverable" in note for note in getattr(exc_info.value, "__notes__", ()))


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows sharing semantics")
def test_dashboard_eject_blocks_windows_writer_after_cleanup_is_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    write_staging_tree = dashboard_cli._write_staging_tree
    deletion_handle = publication_cli._windows_deletion_handle
    attempted_write = False

    def fail_after_writing(
        staging: Path,
        contents: dict[str, bytes],
        *,
        staging_guard: dashboard_cli._StagingGuard,
    ) -> None:
        write_staging_tree(staging, contents, staging_guard=staging_guard)
        raise OSError("injected write failure")

    @contextmanager
    def attempt_write_after_cleanup_handle_open(
        path: Path,
        *,
        read_content: bool = False,
    ) -> Iterator[tuple[int, Callable[[], None]]]:
        nonlocal attempted_write
        with deletion_handle(path, read_content=read_content) as deletion_owner:
            if path.name == "scripts" and not attempted_write:
                attempted_write = True
                with pytest.raises(OSError) as exc_info, _windows_directory_write_handle(path):
                    pytest.fail("cleanup handle unexpectedly shared directory writes")
                assert getattr(exc_info.value, "winerror", None) == 32
            yield deletion_owner

    monkeypatch.setattr(dashboard_cli, "_write_staging_tree", fail_after_writing)
    monkeypatch.setattr(
        publication_cli,
        "_windows_deletion_handle",
        attempt_write_after_cleanup_handle_open,
    )

    with pytest.raises(OSError, match="injected write failure") as exc_info:
        eject_dashboard_source(destination)

    assert attempted_write
    assert not destination.exists()
    assert not list(tmp_path.glob(".cayu-tree-stage-*"))
    assert not any(
        "remains recoverable" in note for note in getattr(exc_info.value, "__notes__", ())
    )


def test_dashboard_eject_restores_original_empty_destination_after_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    destination.mkdir(mode=0o700)
    original_stat = destination.stat()
    rename_publication_entry = publication_cli._rename_no_replace

    def fail_staging_publish(
        parent: Any,
        source: str,
        target: str,
        *,
        expected: publication_cli._Identity,
        label: str,
    ) -> None:
        if source.startswith(".cayu-tree-stage-") and target == destination.name:
            raise OSError("injected publish failure")
        rename_publication_entry(parent, source, target, expected=expected, label=label)

    monkeypatch.setattr(publication_cli, "_rename_no_replace", fail_staging_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        eject_dashboard_source(destination)

    assert destination.is_dir()
    assert os.path.samestat(original_stat, destination.stat())
    assert not list(destination.iterdir())
    assert not list(tmp_path.glob(".cayu-tree-stage-*"))
    assert not list(tmp_path.glob(".cayu-tree-backup-*"))
    assert not list(tmp_path.glob(".cayu-tree-publication-*.jsonl"))


@pytest.mark.parametrize("existing_destination", [False, True])
@pytest.mark.skipif(os.name != "nt", reason="requires Windows permission publication")
def test_dashboard_eject_rolls_back_if_permissions_cannot_be_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_destination: bool,
) -> None:
    destination = tmp_path / "control-plane"
    original_destination: os.stat_result | None = None
    if existing_destination:
        destination.mkdir()
        original_destination = destination.stat()

    def fail_permission_restore(
        _parent: publication_cli._Parent,
        _name: str,
        *,
        expected: publication_cli._Identity,
    ) -> None:
        del expected
        raise OSError("injected permission restore failure")

    monkeypatch.setattr(
        publication_cli,
        "_finalize_published_tree",
        fail_permission_restore,
    )

    with pytest.raises(OSError, match="injected permission restore failure"):
        eject_dashboard_source(destination)

    if original_destination is None:
        assert not destination.exists()
    else:
        assert destination.is_dir()
        assert os.path.samestat(original_destination, destination.stat())
        assert not list(destination.iterdir())
    assert not list(tmp_path.glob(".cayu-tree-stage-*"))
    assert not list(tmp_path.glob(".cayu-tree-backup-*"))
    assert not list(tmp_path.glob(".cayu-tree-publication-*.jsonl"))


def test_dashboard_eject_preserves_both_trees_when_cleanup_conflicts_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    destination.mkdir()
    original_stat = destination.stat()
    published_marker = destination / "concurrent-user.txt"
    original_marker_name = "original-user.txt"

    def mutate_both_trees_before_original_cleanup(phase: str) -> None:
        if phase != "published":
            return
        published_marker.write_text("keep published\n", encoding="utf-8")
        backup = next(tmp_path.glob(".cayu-tree-backup-*"))
        (backup / original_marker_name).write_text("keep original\n", encoding="utf-8")

    monkeypatch.setattr(
        publication_cli,
        "_publication_fault",
        mutate_both_trees_before_original_cleanup,
    )

    with pytest.raises(DashboardSourceError, match="original destination changed"):
        eject_dashboard_source(destination)

    assert published_marker.read_text(encoding="utf-8") == "keep published\n"
    assert (destination / "package.json").is_file()
    backups = list(tmp_path.glob(".cayu-tree-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / original_marker_name).read_text(encoding="utf-8") == "keep original\n"
    assert os.path.samestat(original_stat, backups[0].stat())
    assert len(list(tmp_path.glob(".cayu-tree-publication-*.jsonl"))) == 1


def test_dashboard_eject_refuses_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    destination = tmp_path / "control-plane"
    try:
        destination.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    assert main(["dashboard", "eject", str(destination)]) == 1

    assert list(target.iterdir()) == []


def test_dashboard_eject_refuses_symlink_in_destination_path(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    assert main(["dashboard", "eject", str(linked_parent / "control-plane")]) == 1

    assert list(target.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="requires a real Windows directory junction")
def test_dashboard_eject_refuses_junction_in_destination_path(tmp_path: Path) -> None:
    bundle = _packaged_bundle()
    target = tmp_path / "target"
    target.mkdir()
    linked_parent = tmp_path / "linked-parent"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/j", str(linked_parent), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"directory junctions are unavailable: {result.stderr.strip()}")

    try:
        with pytest.raises(DashboardSourceError, match="symbolic link or junction"):
            eject_dashboard_source(
                linked_parent / "control-plane",
                bundle_bytes=bundle,
            )
        assert list(target.iterdir()) == []
    finally:
        linked_parent.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="requires a real Windows directory junction")
def test_dashboard_eject_refuses_junction_introduced_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    target = tmp_path / "target"
    target.mkdir()
    introduced_junction = False

    def publish_after_introducing_junction(phase: str) -> None:
        nonlocal introduced_junction
        if phase != "stage_synced":
            return
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/j", str(destination), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"directory junctions are unavailable: {result.stderr.strip()}")
        introduced_junction = True

    monkeypatch.setattr(
        publication_cli,
        "_publication_fault",
        publish_after_introducing_junction,
    )

    try:
        with pytest.raises(DashboardSourceError, match="destination appeared"):
            eject_dashboard_source(destination)
        assert introduced_junction
        assert list(target.iterdir()) == []
        assert len(list(tmp_path.glob(".cayu-tree-stage-*"))) == 1
    finally:
        if destination.exists():
            destination.rmdir()


def test_dashboard_eject_rechecks_lexical_parent_if_link_appears_during_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _packaged_bundle()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.mkdir()
    displaced_parent = tmp_path / "displaced-parent"
    target = tmp_path / "target"
    target.mkdir()
    destination = linked_parent / "control-plane"
    resolve = Path.resolve
    swapped = False

    def resolve_after_link_appears(path: Path, strict: bool = False) -> Path:
        nonlocal swapped
        if path == destination and not swapped:
            linked_parent.rename(displaced_parent)
            try:
                linked_parent.symlink_to(target, target_is_directory=True)
            except OSError:
                displaced_parent.rename(linked_parent)
                pytest.skip("symlinks are unavailable")
            swapped = True
        return resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve_after_link_appears)

    with pytest.raises(DashboardSourceError, match="symbolic link or junction"):
        eject_dashboard_source(destination, bundle_bytes=bundle)

    assert swapped is True
    assert list(target.iterdir()) == []
    assert list(displaced_parent.iterdir()) == []


def test_dashboard_destination_link_detection_rejects_every_windows_reparse_point() -> None:
    assert dashboard_cli._is_windows_reparse_point(
        cast(
            "os.stat_result",
            SimpleNamespace(st_file_attributes=0x400, st_reparse_tag=0xA0000003),
        )
    )
    assert dashboard_cli._is_windows_reparse_point(
        cast("os.stat_result", SimpleNamespace(st_file_attributes=0x400))
    )
    assert dashboard_cli._is_windows_reparse_point(
        cast(
            "os.stat_result",
            SimpleNamespace(st_file_attributes=0x400, st_reparse_tag=0x8000001A),
        )
    )


@pytest.mark.parametrize(
    ("member_name", "mode", "message"),
    [
        ("/outside.txt", stat.S_IFREG, "unsafe dashboard source archive path"),
        ("../outside.txt", stat.S_IFREG, "unsafe dashboard source archive path"),
        (".", stat.S_IFREG, "unsafe dashboard source archive path"),
        ("C:outside.txt", stat.S_IFREG, "unsafe dashboard source archive path"),
        ("C:/outside.txt", stat.S_IFREG, "unsafe dashboard source archive path"),
        ("safe.txt:stream", stat.S_IFREG, "unsafe dashboard source archive path"),
        (".. /outside.txt", stat.S_IFREG, "unsafe dashboard source archive path"),
        (" ../outside.txt", stat.S_IFREG, "unsafe dashboard source archive path"),
        ("safe./file.txt", stat.S_IFREG, "unsafe dashboard source archive path"),
        ("NUL.txt", stat.S_IFREG, "unsafe dashboard source archive path"),
        ("COM1.log", stat.S_IFREG, "unsafe dashboard source archive path"),
        ("bad?.txt", stat.S_IFREG, "unsafe dashboard source archive path"),
        ("bad\x1f.txt", stat.S_IFREG, "unsafe dashboard source archive path"),
        (
            "PACKAGE.JSON/child.txt",
            stat.S_IFREG,
            "conflicting dashboard source archive paths",
        ),
        ("linked.txt", stat.S_IFLNK, "unsupported dashboard source archive entry type"),
        ("pipe", stat.S_IFIFO, "unsupported dashboard source archive entry type"),
        ("PACKAGE.JSON", stat.S_IFREG, "duplicate dashboard source archive path"),
    ],
)
def test_dashboard_eject_rejects_unsafe_archive_entries_without_partial_output(
    tmp_path: Path,
    member_name: str,
    mode: int,
    message: str,
) -> None:
    member = _ordinary_member(member_name)
    member.external_attr = (mode | 0o644) << 16
    bundle = _rewrite_bundle(lambda info, content: (info, content), appended=(member, b"x"))
    destination = tmp_path / "control-plane"

    with pytest.raises(DashboardSourceError, match=message):
        eject_dashboard_source(destination, bundle_bytes=bundle)

    assert not destination.exists()


def test_dashboard_eject_quotes_control_characters_in_unsafe_path_error(
    tmp_path: Path,
) -> None:
    member = _ordinary_member("bad\nforged.txt")
    bundle = _rewrite_bundle(lambda info, content: (info, content), appended=(member, b"x"))

    with pytest.raises(DashboardSourceError) as exc_info:
        eject_dashboard_source(tmp_path / "control-plane", bundle_bytes=bundle)

    message = str(exc_info.value)
    assert "\\n" in message
    assert "\n" not in message


def test_dashboard_eject_publishes_file_matching_internal_owner_marker_name(
    tmp_path: Path,
) -> None:
    marker_name = publication_cli._OWNER_MARKER
    marker_content = b"application-owned payload\n"
    with zipfile.ZipFile(io.BytesIO(_packaged_bundle())) as source:
        contents = {member.filename: source.read(member) for member in source.infolist()}
    manifest = json.loads(contents.pop("cayu-dashboard-source.json"))
    contents[marker_name] = marker_content
    manifest["files"].append(
        {
            "path": marker_name,
            "size": len(marker_content),
            "sha256": dashboard_cli._sha256(marker_content),
        }
    )
    manifest["files"].sort(key=lambda item: item["path"])
    manifest["source_digest"] = dashboard_cli._contents_digest(contents)
    contents["cayu-dashboard-source.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path, content in contents.items():
            archive.writestr(_ordinary_member(path), content)

    destination = tmp_path / "control-plane"
    eject_dashboard_source(destination, bundle_bytes=output.getvalue())

    assert (destination / marker_name).read_bytes() == marker_content


def test_dashboard_eject_rejects_unsafe_manifest_path_without_partial_output(
    tmp_path: Path,
) -> None:
    def add_windows_reserved_manifest_path(
        member: zipfile.ZipInfo,
        content: bytes,
    ) -> tuple[zipfile.ZipInfo, bytes]:
        if member.filename != "cayu-dashboard-source.json":
            return member, content
        manifest = json.loads(content)
        manifest["files"][0]["path"] = "NUL.txt"
        return member, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    destination = tmp_path / "control-plane"
    with pytest.raises(DashboardSourceError, match="unsafe dashboard source archive path"):
        eject_dashboard_source(
            destination,
            bundle_bytes=_rewrite_bundle(add_windows_reserved_manifest_path),
        )

    assert not destination.exists()


def test_dashboard_eject_rejects_conflicting_manifest_path_without_partial_output(
    tmp_path: Path,
) -> None:
    def add_ancestor_conflict_to_manifest(
        member: zipfile.ZipInfo,
        content: bytes,
    ) -> tuple[zipfile.ZipInfo, bytes]:
        if member.filename != "cayu-dashboard-source.json":
            return member, content
        manifest = json.loads(content)
        ancestor = manifest["files"][0]["path"]
        manifest["files"].append(
            {
                "path": f"{ancestor}/child.txt",
                "sha256": f"sha256:{'0' * 64}",
                "size": 0,
            }
        )
        manifest["files"].sort(key=lambda item: item["path"])
        return member, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    destination = tmp_path / "control-plane"
    with pytest.raises(DashboardSourceError, match="conflicting dashboard source manifest paths"):
        eject_dashboard_source(
            destination,
            bundle_bytes=_rewrite_bundle(add_ancestor_conflict_to_manifest),
        )

    assert not destination.exists()


def test_dashboard_eject_rejects_duplicate_archive_entries_without_partial_output(
    tmp_path: Path,
) -> None:
    package_json = _ordinary_member("package.json")
    with pytest.warns(UserWarning, match="Duplicate name"):
        bundle = _rewrite_bundle(
            lambda info, content: (info, content),
            appended=(package_json, b"duplicate"),
        )
    destination = tmp_path / "control-plane"

    with pytest.raises(DashboardSourceError, match="duplicate dashboard source archive path"):
        eject_dashboard_source(destination, bundle_bytes=bundle)

    assert not destination.exists()


def test_dashboard_eject_rejects_corrupt_bundle_content_without_partial_output(
    tmp_path: Path,
) -> None:
    def corrupt_package_json(
        member: zipfile.ZipInfo,
        content: bytes,
    ) -> tuple[zipfile.ZipInfo, bytes]:
        return member, b"{}\n" if member.filename == "package.json" else content

    destination = tmp_path / "control-plane"
    with pytest.raises(DashboardSourceError, match="dashboard source (size|digest) mismatch"):
        eject_dashboard_source(destination, bundle_bytes=_rewrite_bundle(corrupt_package_json))

    assert not destination.exists()


def test_dashboard_eject_rejects_version_mismatch_without_partial_output(tmp_path: Path) -> None:
    def change_version(
        member: zipfile.ZipInfo,
        content: bytes,
    ) -> tuple[zipfile.ZipInfo, bytes]:
        if member.filename != "cayu-dashboard-source.json":
            return member, content
        manifest = json.loads(content)
        manifest["cayu_version"] = "9.9.9"
        return member, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    destination = tmp_path / "control-plane"
    with pytest.raises(DashboardSourceError, match="dashboard source Cayu version mismatch"):
        eject_dashboard_source(destination, bundle_bytes=_rewrite_bundle(change_version))

    assert not destination.exists()


def test_dashboard_eject_removes_staging_after_filesystem_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"

    def fail_after_partial_write(
        staging: Path,
        _contents: dict[str, bytes],
        *,
        staging_guard: dashboard_cli._StagingGuard,
    ) -> None:
        del staging_guard
        (staging / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise OSError("injected write failure")

    monkeypatch.setattr(dashboard_cli, "_write_staging_tree", fail_after_partial_write)
    with pytest.raises(OSError, match="injected write failure"):
        eject_dashboard_source(destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".cayu-tree-*"))


@pytest.mark.skipif(os.name == "nt", reason="exercises POSIX descriptor ownership")
@pytest.mark.parametrize("failing_mode", [0o644, 0o755])
def test_dashboard_eject_closes_staging_descriptor_after_chmod_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_mode: int,
) -> None:
    destination = tmp_path / "control-plane"
    fchmod = os.fchmod
    failed_descriptor: int | None = None

    def fail_one_staging_chmod(descriptor: int, mode: int) -> None:
        nonlocal failed_descriptor
        if failed_descriptor is None and mode == failing_mode:
            failed_descriptor = descriptor
            raise OSError("injected staging chmod failure")
        fchmod(descriptor, mode)

    monkeypatch.setattr(dashboard_cli.os, "fchmod", fail_one_staging_chmod)

    with pytest.raises(OSError, match="injected staging chmod failure"):
        eject_dashboard_source(destination)

    assert failed_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(failed_descriptor)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cayu-tree-*"))


@pytest.mark.skipif(os.name != "nt", reason="requires Windows DACL inspection")
def test_dashboard_eject_protects_staging_dacl_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    write_staging_tree = dashboard_cli._write_staging_tree
    inspected_staging = False

    def inspect_staging_before_write(
        staging: Path,
        contents: dict[str, bytes],
        *,
        staging_guard: dashboard_cli._StagingGuard,
    ) -> None:
        nonlocal inspected_staging
        dacl_present, dacl_protected = publication_cli._windows_directory_dacl_state(staging)
        assert dacl_present and dacl_protected
        inspected_staging = True
        write_staging_tree(staging, contents, staging_guard=staging_guard)

    monkeypatch.setattr(dashboard_cli, "_write_staging_tree", inspect_staging_before_write)

    eject_dashboard_source(destination)

    assert inspected_staging


@pytest.mark.skipif(os.name != "nt", reason="requires Windows DACL inspection")
def test_dashboard_eject_restores_parent_acl_inheritance_after_publication(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "control-plane"

    eject_dashboard_source(destination)

    publication_cli._assert_windows_directory_dacl_is_inherited(destination)


@pytest.mark.skipif(os.name != "nt", reason="requires Windows namespace sharing semantics")
def test_dashboard_eject_pins_published_identity_during_permission_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "control-plane"
    displaced = tmp_path / "displaced-control-plane"
    restore_inheritance = publication_cli._restore_windows_directory_inheritance
    replacement_attempted = False

    def restore_after_attempting_replacement(path: Path) -> None:
        nonlocal replacement_attempted
        replacement_attempted = True
        with pytest.raises(OSError):
            path.rename(displaced)
        restore_inheritance(path)

    monkeypatch.setattr(
        publication_cli,
        "_restore_windows_directory_inheritance",
        restore_after_attempting_replacement,
    )

    eject_dashboard_source(destination)

    assert replacement_attempted
    assert (destination / "package.json").is_file()
    assert not displaced.exists()


def test_windows_staging_creation_uses_native_acl_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / f".cayu-tree-stage-{'a' * 32}"
    native_creation_used = False

    def create_native_staging(candidate: Path) -> None:
        nonlocal native_creation_used
        native_creation_used = True
        assert candidate == expected
        candidate.mkdir()

    monkeypatch.setattr(publication_cli.os, "name", "nt")
    monkeypatch.setattr(
        publication_cli,
        "_windows_directory_namespace_fence",
        lambda _path: nullcontext(),
    )
    monkeypatch.setattr(
        publication_cli,
        "_create_private_windows_directory",
        create_native_staging,
    )
    monkeypatch.setattr(publication_cli, "_sync_windows_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        publication_cli,
        "_assert_windows_directory_dacl_is_protected",
        lambda _path: None,
    )
    parent = publication_cli._Parent(
        path=tmp_path,
        identity=publication_cli._capture_parent(tmp_path),
        descriptor=None,
    )

    publication_cli._create_private_stage(expected, token="a" * 32, parent=parent)

    assert native_creation_used
    assert (expected / publication_cli._OWNER_MARKER).read_text(encoding="ascii") == "a" * 32


@pytest.mark.skipif(os.name != "nt", reason="requires Windows namespace sharing semantics")
def test_dashboard_eject_fences_windows_parent_during_staging_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    displaced_parent = tmp_path / "displaced-parent"
    destination = parent / "control-plane"
    create_private_staging = publication_cli._create_private_windows_directory
    attempted_parent_rename = False

    def create_after_attempting_parent_rename(candidate: Path) -> OSError | None:
        nonlocal attempted_parent_rename
        attempted_parent_rename = True
        try:
            parent.rename(displaced_parent)
        except OSError:
            pass
        else:
            displaced_parent.rename(parent)
            pytest.fail("the validated destination parent was not fenced")
        return create_private_staging(candidate)

    monkeypatch.setattr(
        publication_cli,
        "_create_private_windows_directory",
        create_after_attempting_parent_rename,
    )

    eject_dashboard_source(destination)

    assert attempted_parent_rename
    assert (destination / "package.json").is_file()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows namespace sharing semantics")
def test_dashboard_eject_fences_windows_parent_during_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    displaced_parent = tmp_path / "displaced-parent"
    destination = parent / "control-plane"
    rename_publication_entry = publication_cli._rename_no_replace
    attempted_parent_rename = False

    def publish_after_attempting_parent_rename(
        publication_parent: Any,
        source: str,
        target: str,
        *,
        expected: publication_cli._Identity,
        label: str,
    ) -> None:
        nonlocal attempted_parent_rename
        if not attempted_parent_rename and target == destination.name:
            attempted_parent_rename = True
            try:
                parent.rename(displaced_parent)
            except OSError:
                pass
            else:
                displaced_parent.rename(parent)
                pytest.fail("the validated destination parent was not fenced")
        rename_publication_entry(
            publication_parent,
            source,
            target,
            expected=expected,
            label=label,
        )

    monkeypatch.setattr(
        publication_cli,
        "_rename_no_replace",
        publish_after_attempting_parent_rename,
    )

    eject_dashboard_source(destination)

    assert attempted_parent_rename
    assert (destination / "package.json").is_file()
