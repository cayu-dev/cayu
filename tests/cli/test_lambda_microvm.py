from __future__ import annotations

import json
import os
import shutil
import tomllib
from pathlib import Path

import pytest

from cayu.cli import lambda_microvm as sidecar_cli
from cayu.cli import main

_SOURCE = Path(__file__).resolve().parents[2] / "examples" / "aws" / "lambda_microvm_sidecar"
_MANIFEST = "cayu-lambda-microvm-sidecar-manifest.json"


def _project_version() -> str:
    with (Path(__file__).resolve().parents[2] / "pyproject.toml").open("rb") as project:
        return tomllib.load(project)["project"]["version"]


def _tree_contents(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy_source(tmp_path: Path) -> Path:
    resource = tmp_path / "resource"
    shutil.copytree(_SOURCE, resource)
    return resource


def test_lambda_microvm_sidecar_export_is_reproducible(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert main(["lambda-microvm", "sidecar", "export", str(first)]) == 0
    first_output = capsys.readouterr()
    assert main(["lambda-microvm", "sidecar", "export", str(second)]) == 0
    second_output = capsys.readouterr()

    assert first_output.err == second_output.err == ""
    assert "content digest: sha256:" in first_output.out
    assert "content digest: sha256:" in second_output.out
    assert _tree_contents(first) == _tree_contents(second)
    manifest = json.loads((first / _MANIFEST).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["artifact_version"] == 1
    assert manifest["cayu_version"] == _project_version()
    assert manifest["protocol_version"] == "1"
    assert manifest["content_digest"] in first_output.out
    assert set(_tree_contents(first)) == {
        _MANIFEST,
        *(item["path"] for item in manifest["files"]),
    }


def test_lambda_microvm_sidecar_export_requires_replace_for_nonempty_destination(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "sidecar"
    destination.mkdir()
    sentinel = destination / "operator-owned.txt"
    sentinel.write_text("keep", encoding="utf-8")

    assert main(["lambda-microvm", "sidecar", "export", str(destination)]) == 1
    failure = capsys.readouterr()
    assert failure.out == ""
    assert "pass --replace" in failure.err
    assert sentinel.read_text(encoding="utf-8") == "keep"

    assert (
        main(
            [
                "lambda-microvm",
                "sidecar",
                "export",
                str(destination),
                "--replace",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert not sentinel.exists()
    assert (destination / _MANIFEST).is_file()


def test_lambda_microvm_sidecar_export_accepts_existing_empty_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "sidecar"
    destination.mkdir()

    result = sidecar_cli._export_sidecar(destination, replace=False)

    assert result.destination == destination
    assert (destination / _MANIFEST).is_file()


@pytest.mark.parametrize("replace", [False, True])
def test_lambda_microvm_sidecar_export_refuses_file_and_symlink_destinations(
    tmp_path: Path,
    replace: bool,
) -> None:
    file_destination = tmp_path / "file"
    file_destination.write_text("operator-owned", encoding="utf-8")
    symlink_destination = tmp_path / "link"
    try:
        symlink_destination.symlink_to(file_destination)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(sidecar_cli._SidecarArtifactError, match="must be a directory"):
        sidecar_cli._export_sidecar(file_destination, replace=replace)
    with pytest.raises(sidecar_cli._SidecarArtifactError, match="must not be a symlink"):
        sidecar_cli._export_sidecar(symlink_destination, replace=replace)
    assert file_destination.read_text(encoding="utf-8") == "operator-owned"


def test_lambda_microvm_sidecar_export_refuses_working_directory_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)

    with pytest.raises(sidecar_cli._SidecarArtifactError, match="current working directory"):
        sidecar_cli._export_sidecar(tmp_path, replace=True)


def test_lambda_microvm_sidecar_export_refuses_home_and_its_ancestors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    for destination in (home, tmp_path):
        with pytest.raises(sidecar_cli._SidecarArtifactError, match="home directory"):
            sidecar_cli._export_sidecar(destination, replace=True)


@pytest.mark.parametrize("mutation", ["modified", "missing", "unexpected", "symlink"])
def test_lambda_microvm_sidecar_export_rejects_corrupt_resources_before_mutation(
    tmp_path: Path,
    mutation: str,
) -> None:
    resource = _copy_source(tmp_path)
    if mutation == "modified":
        (resource / "app.py").write_text("corrupt", encoding="utf-8")
    elif mutation == "missing":
        (resource / "app.py").unlink()
    elif mutation == "unexpected":
        (resource / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    else:
        target = resource / "app.py"
        link = resource / "linked.py"
        try:
            link.symlink_to(target)
        except OSError:
            pytest.skip("symlinks are unavailable")

    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "operator-owned.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(sidecar_cli._SidecarArtifactError):
        sidecar_cli._export_sidecar(
            destination,
            replace=True,
            resource_root=resource,
            expected_cayu_version=_project_version(),
        )
    assert _tree_contents(destination) == {"operator-owned.txt": b"keep"}
    assert list(tmp_path.glob(".destination.cayu-sidecar-*")) == []


def test_lambda_microvm_sidecar_export_rejects_linked_resource_root(
    tmp_path: Path,
) -> None:
    resource = _copy_source(tmp_path)
    linked_resource = tmp_path / "linked-resource"
    try:
        linked_resource.symlink_to(resource, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(sidecar_cli._SidecarArtifactError, match="resource root.*not a link"):
        sidecar_cli._export_sidecar(
            tmp_path / "destination",
            replace=False,
            resource_root=linked_resource,
            expected_cayu_version=_project_version(),
        )


def test_packaged_sidecar_requires_installed_distribution_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "cayu-data"
    shutil.copytree(_SOURCE, data_root / "lambda_microvm_sidecar")
    monkeypatch.setattr(sidecar_cli, "files", lambda package: data_root)

    def missing_version(distribution: str) -> str:
        raise sidecar_cli.importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(sidecar_cli.importlib.metadata, "version", missing_version)

    with pytest.raises(sidecar_cli._SidecarArtifactError, match="distribution metadata"):
        sidecar_cli._sidecar_resource()


def test_lambda_microvm_sidecar_export_rejects_manifest_changed_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = _copy_source(tmp_path)
    manifest_path = resource / _MANIFEST
    original_read_bytes = Path.read_bytes
    manifest_reads = 0

    def change_second_manifest_read(path: Path) -> bytes:
        nonlocal manifest_reads
        content = original_read_bytes(path)
        if path == manifest_path:
            manifest_reads += 1
            if manifest_reads == 2:
                return content + b"\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", change_second_manifest_read)

    with pytest.raises(
        sidecar_cli._SidecarArtifactError,
        match="manifest changed while it was being validated",
    ):
        sidecar_cli._export_sidecar(
            tmp_path / "destination",
            replace=False,
            resource_root=resource,
            expected_cayu_version=_project_version(),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "unsupported sidecar manifest schema"),
        ("artifact_version", 2, "unsupported sidecar artifact"),
        ("cayu_version", "different", "Cayu version mismatch"),
        ("protocol_version", "different", "protocol version mismatch"),
    ],
)
def test_lambda_microvm_sidecar_export_rejects_incompatible_manifest(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    resource = _copy_source(tmp_path)
    manifest_path = resource / _MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(sidecar_cli._SidecarArtifactError, match=message):
        sidecar_cli._export_sidecar(
            tmp_path / "destination",
            replace=False,
            resource_root=resource,
            expected_cayu_version=_project_version(),
        )


def test_lambda_microvm_sidecar_export_rejects_unsafe_and_case_colliding_manifest_paths(
    tmp_path: Path,
) -> None:
    for unsafe_path in ("../outside", "App.py"):
        resource = _copy_source(tmp_path / unsafe_path.replace("/", "_"))
        manifest_path = resource / _MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        duplicate = dict(manifest["files"][0])
        duplicate["path"] = unsafe_path
        manifest["files"].append(duplicate)
        manifest["files"].sort(key=lambda item: item["path"])
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(sidecar_cli._SidecarArtifactError):
            sidecar_cli._export_sidecar(
                tmp_path / "destination",
                replace=False,
                resource_root=resource,
                expected_cayu_version=_project_version(),
            )


def test_lambda_microvm_sidecar_export_cleans_partial_staging_without_touching_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "operator-owned.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def fail_after_partial_write(
        staging: Path,
        artifact: sidecar_cli._ValidatedSidecarArtifact,
    ) -> None:
        del artifact
        (staging / "partial").write_text("partial", encoding="utf-8")
        raise OSError("injected write failure")

    monkeypatch.setattr(sidecar_cli, "_write_staging_tree", fail_after_partial_write)

    with pytest.raises(OSError, match="injected write failure"):
        sidecar_cli._export_sidecar(destination, replace=True)
    assert _tree_contents(destination) == {"operator-owned.txt": b"keep"}
    assert list(tmp_path.glob(".destination.cayu-sidecar-*")) == []


def test_lambda_microvm_sidecar_export_preserves_interruption_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "destination"
    interruption = KeyboardInterrupt("injected interruption")

    def interrupt_after_partial_write(
        staging: Path,
        artifact: sidecar_cli._ValidatedSidecarArtifact,
    ) -> None:
        del artifact
        (staging / "partial").write_text("partial", encoding="utf-8")
        raise interruption

    monkeypatch.setattr(sidecar_cli, "_write_staging_tree", interrupt_after_partial_write)

    with pytest.raises(KeyboardInterrupt) as raised:
        sidecar_cli._export_sidecar(destination, replace=False)
    assert raised.value is interruption
    assert not destination.exists()
    assert list(tmp_path.glob(".destination.cayu-sidecar-*")) == []


def test_lambda_microvm_sidecar_export_preserves_backup_and_reports_it_when_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "operator-owned.txt"
    sentinel.write_text("keep", encoding="utf-8")
    original_rename = Path.rename

    def fail_staging_publish(path: Path, target: Path) -> Path:
        if (
            path.name.startswith(".destination.cayu-sidecar-")
            and "backup" not in path.name
            and Path(target) == destination
        ):
            raise OSError("injected publish failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_publish)

    assert main(["lambda-microvm", "sidecar", "export", str(destination), "--replace"]) == 1
    output = capsys.readouterr()
    backups = list(tmp_path.glob(".destination.cayu-sidecar-backup-*"))
    assert output.out == ""
    assert "error: injected publish failure" in output.err
    assert len(backups) == 1
    assert f"note: the original destination remains at {backups[0]}" in output.err
    assert _tree_contents(backups[0]) == {"operator-owned.txt": b"keep"}
    assert not destination.exists()
    assert list(tmp_path.glob(".destination.cayu-sidecar-*")) == backups


def test_lambda_microvm_sidecar_export_reports_old_destination_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "operator-owned.txt"
    sentinel.write_text("keep", encoding="utf-8")
    original_rmtree = shutil.rmtree

    def fail_backup_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if Path(path).name.startswith(".destination.cayu-sidecar-backup-"):
            raise OSError("injected cleanup failure")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fail_backup_cleanup)

    assert main(["lambda-microvm", "sidecar", "export", str(destination), "--replace"]) == 1
    output = capsys.readouterr()
    backups = list(tmp_path.glob(".destination.cayu-sidecar-backup-*"))
    assert (destination / _MANIFEST).is_file()
    assert len(backups) == 1
    assert _tree_contents(backups[0]) == {"operator-owned.txt": b"keep"}
    assert f"old destination cleanup failed at {backups[0]}" in output.err


def test_lambda_microvm_sidecar_cli_reports_expanduser_errors_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_expanduser = Path.expanduser

    def fail_expanduser(path: Path) -> Path:
        if path == tmp_path / "destination":
            raise RuntimeError("injected expansion failure")
        return original_expanduser(path)

    monkeypatch.setattr(Path, "expanduser", fail_expanduser)

    assert main(["lambda-microvm", "sidecar", "export", str(tmp_path / "destination")]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "error: could not expand destination" in output.err
    assert "Traceback" not in output.err


def test_lambda_microvm_sidecar_cli_rejects_non_utf8_destination_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = "sidecar-\udcff"

    assert main(["lambda-microvm", "sidecar", "export", destination]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "error: destination path must be valid UTF-8\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_lambda_microvm_sidecar_export_normalizes_file_modes(tmp_path: Path) -> None:
    destination = tmp_path / "destination"

    sidecar_cli._export_sidecar(destination, replace=False)

    assert destination.stat().st_mode & 0o777 == 0o755
    assert all(
        path.stat().st_mode & 0o777 == (0o755 if path.is_dir() else 0o644)
        for path in destination.rglob("*")
    )
