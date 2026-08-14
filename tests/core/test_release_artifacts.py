from __future__ import annotations

import io
import json
import runpy
import shutil
import stat
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from cayu.cli.lambda_microvm import _render_manifest
from cayu.runners.aws_lambda_microvm import LAMBDA_MICROVM_PROTOCOL_VERSION

_ROOT = Path(__file__).parents[2]
_SIDECAR_SOURCE = _ROOT / "examples" / "aws" / "lambda_microvm_sidecar"
_COMPILED_DASHBOARD_SOURCE = _ROOT / "src" / "cayu" / "server" / "dashboard"
_MANIFEST = "cayu-lambda-microvm-sidecar-manifest.json"
with (_ROOT / "pyproject.toml").open("rb") as _project_file:
    _VERSION = tomllib.load(_project_file)["project"]["version"]
_WHEEL_DIST_INFO = f"cayu-{_VERSION}.dist-info"
_DASHBOARD_SOURCE_BUNDLE_NAME = f"cayu-dashboard-source-{_VERSION}.zip"
_DASHBOARD_SOURCE_BUNDLE = (
    _ROOT / "src" / "cayu" / "data" / "dashboard_source" / _DASHBOARD_SOURCE_BUNDLE_NAME
)
_WHEEL_DASHBOARD_SOURCE = f"cayu/data/dashboard_source/{_DASHBOARD_SOURCE_BUNDLE_NAME}"
_SDIST_DASHBOARD_SOURCE = f"src/cayu/data/dashboard_source/{_DASHBOARD_SOURCE_BUNDLE_NAME}"

artifact_validator = runpy.run_path(str(_ROOT / "scripts" / "check_release_artifacts.py"))
validate_sdist = artifact_validator["validate_sdist"]
validate_wheel = artifact_validator["validate_wheel"]
validate_sidecar_equivalence = artifact_validator["validate_sidecar_equivalence"]
validate_dashboard_source_equivalence = artifact_validator["validate_dashboard_source_equivalence"]


def _canonical_sidecar() -> dict[str, bytes]:
    return {
        path.relative_to(_SIDECAR_SOURCE).as_posix(): path.read_bytes()
        for path in _SIDECAR_SOURCE.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def _canonical_compiled_dashboard() -> dict[str, bytes]:
    return {
        path.relative_to(_COMPILED_DASHBOARD_SOURCE).as_posix(): path.read_bytes()
        for path in _COMPILED_DASHBOARD_SOURCE.rglob("*")
        if path.is_file()
    }


def _valid_wheel_names(sidecar: dict[str, bytes] | None = None) -> set[str]:
    sidecar = sidecar or _canonical_sidecar()
    compiled_dashboard = _canonical_compiled_dashboard()
    return {
        "cayu/__init__.py",
        "cayu/cli/_targets.py",
        "cayu/cli/__init__.py",
        "cayu/cli/console.py",
        "cayu/data/__init__.py",
        "cayu/data/default_model_catalog.json",
        "cayu/data/default_price_book.json",
        "cayu/guides/application-anatomy.md",
        "cayu/guides/authoring.md",
        "cayu/guides/diagnostics.md",
        "cayu/guides/durable-operations.md",
        "cayu/guides/providers.md",
        "cayu/guides/tool-effects.md",
        _WHEEL_DASHBOARD_SOURCE,
        *{f"cayu/server/dashboard/{name}" for name in compiled_dashboard},
        *{f"{artifact_validator['_WHEEL_SIDECAR_PREFIX']}/{name}" for name in sidecar},
        f"{_WHEEL_DIST_INFO}/METADATA",
        f"{_WHEEL_DIST_INFO}/RECORD",
        f"{_WHEEL_DIST_INFO}/WHEEL",
        f"{_WHEEL_DIST_INFO}/entry_points.txt",
        f"{_WHEEL_DIST_INFO}/licenses/LICENSE",
        f"{_WHEEL_DIST_INFO}/licenses/NOTICE",
    }


def _as_bytes(value: str | bytes) -> bytes:
    return value if isinstance(value, bytes) else value.encode()


def _write_wheel(
    path: Path,
    names: set[str],
    *,
    sidecar: dict[str, bytes] | None = None,
    contents_by_name: dict[str, str | bytes] | None = None,
    third_party_notice: str | None = None,
    symlink_name: str | None = None,
) -> None:
    sidecar = sidecar or _canonical_sidecar()
    compiled_dashboard = _canonical_compiled_dashboard()
    contents_by_name = contents_by_name or {}
    prefix = f"{artifact_validator['_WHEEL_SIDECAR_PREFIX']}/"
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            content: str | bytes = ""
            if name.startswith(prefix) and name.removeprefix(prefix) in sidecar:
                content = sidecar[name.removeprefix(prefix)]
            dashboard_prefix = "cayu/server/dashboard/"
            if (
                name.startswith(dashboard_prefix)
                and name.removeprefix(dashboard_prefix) in compiled_dashboard
            ):
                content = compiled_dashboard[name.removeprefix(dashboard_prefix)]
            if name == _WHEEL_DASHBOARD_SOURCE:
                content = _DASHBOARD_SOURCE_BUNDLE.read_bytes()
            if name == f"{_WHEEL_DIST_INFO}/METADATA":
                content = f"Metadata-Version: 2.4\nName: cayu\nVersion: {_VERSION}\n"
            if name in contents_by_name:
                content = contents_by_name[name]
            if (
                name == "cayu/server/dashboard/THIRD_PARTY_LICENSES.md"
                and third_party_notice is not None
            ):
                content = third_party_notice
            if name == symlink_name:
                member = zipfile.ZipInfo(name)
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(member, "app.py")
            else:
                archive.writestr(name, _as_bytes(content))


def _write_sdist(
    path: Path,
    *,
    sidecar: dict[str, bytes] | None = None,
    additional_names: set[str] | None = None,
    contents_by_name: dict[str, str | bytes] | None = None,
) -> None:
    sidecar = sidecar or _canonical_sidecar()
    compiled_dashboard = _canonical_compiled_dashboard()
    additional_names = additional_names or set()
    contents_by_name = contents_by_name or {}
    prefix = artifact_validator["_SDIST_SIDECAR_PREFIX"]
    names = (
        artifact_validator["_SDIST_REQUIRED"]
        | additional_names
        | {f"{prefix}/{name}" for name in sidecar}
        | {_SDIST_DASHBOARD_SOURCE}
        | {f"src/cayu/server/dashboard/{name}" for name in compiled_dashboard}
    )
    with tarfile.open(path, "w:gz") as archive:
        for relative_name in names:
            content: str | bytes = ""
            if relative_name.startswith(f"{prefix}/"):
                sidecar_name = relative_name.removeprefix(f"{prefix}/")
                if sidecar_name in sidecar:
                    content = sidecar[sidecar_name]
            dashboard_prefix = "src/cayu/server/dashboard/"
            if (
                relative_name.startswith(dashboard_prefix)
                and relative_name.removeprefix(dashboard_prefix) in compiled_dashboard
            ):
                content = compiled_dashboard[relative_name.removeprefix(dashboard_prefix)]
            if relative_name == _SDIST_DASHBOARD_SOURCE:
                content = _DASHBOARD_SOURCE_BUNDLE.read_bytes()
            if relative_name == "src/cayu/server/dashboard/THIRD_PARTY_LICENSES.md":
                content = compiled_dashboard["THIRD_PARTY_LICENSES.md"]
            if relative_name == "PKG-INFO":
                content = f"Metadata-Version: 2.4\nName: cayu\nVersion: {_VERSION}\n"
            if relative_name in contents_by_name:
                content = contents_by_name[relative_name]
            data = _as_bytes(content)
            member = tarfile.TarInfo(f"cayu-{_VERSION}/{relative_name}")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def _sidecar_with_nested_file() -> dict[str, bytes]:
    sidecar = _canonical_sidecar()
    sidecar["support/nested.txt"] = b"nested\n"
    files = {name: content for name, content in sidecar.items() if name != _MANIFEST}
    sidecar[_MANIFEST] = _render_manifest(
        files,
        cayu_version=_VERSION,
        protocol_version=LAMBDA_MICROVM_PROTOCOL_VERSION,
    )
    return sidecar


def _dashboard_bundle_with_manifest_value(field: str, value: str) -> bytes:
    output = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(_DASHBOARD_SOURCE_BUNDLE.read_bytes())) as source,
        zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as target,
    ):
        for member in source.infolist():
            content = source.read(member)
            if member.filename == "cayu-dashboard-source.json":
                manifest = json.loads(content)
                manifest[field] = value
                content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            target.writestr(member, content)
    return output.getvalue()


def test_validate_wheel_requires_application_anatomy_guide(tmp_path: Path) -> None:
    wheel = tmp_path / "cayu.whl"
    names = _valid_wheel_names()
    names.remove("cayu/guides/application-anatomy.md")
    _write_wheel(wheel, names)
    with pytest.raises(ValueError, match=r"missing required wheel files: .*application-anatomy"):
        validate_wheel(wheel)


def test_validate_wheel_requires_provider_compatibility_guide(tmp_path: Path) -> None:
    wheel = tmp_path / "cayu.whl"
    names = _valid_wheel_names() | {"cayu/guides/providers.md"}
    names.remove("cayu/guides/providers.md")
    _write_wheel(wheel, names)

    with pytest.raises(ValueError, match=r"missing required wheel files: .*providers\.md"):
        validate_wheel(wheel)


def test_validate_wheel_requires_durable_operations_guide(tmp_path: Path) -> None:
    wheel = tmp_path / "cayu.whl"
    names = _valid_wheel_names()
    names.remove("cayu/guides/durable-operations.md")
    _write_wheel(wheel, names)

    with pytest.raises(ValueError, match=r"missing required wheel files: .*durable-operations"):
        validate_wheel(wheel)


def test_validate_wheel_requires_sidecar_manifest(tmp_path: Path) -> None:
    wheel = tmp_path / "missing-sidecar.whl"
    names = _valid_wheel_names()
    names.remove(f"{artifact_validator['_WHEEL_SIDECAR_PREFIX']}/{_MANIFEST}")
    _write_wheel(wheel, names)
    with pytest.raises(ValueError, match="missing required wheel files"):
        validate_wheel(wheel)


def test_validate_wheel_requires_dashboard_source_bundle(tmp_path: Path) -> None:
    wheel = tmp_path / "missing-dashboard-source.whl"
    names = _valid_wheel_names()
    names.remove(_WHEEL_DASHBOARD_SOURCE)
    _write_wheel(wheel, names)

    with pytest.raises(ValueError, match="missing required dashboard source bundle"):
        validate_wheel(wheel)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cayu_version", "9.9.9", "dashboard source Cayu version mismatch"),
        ("server_contract_version", "999", "dashboard source server contract mismatch"),
    ],
)
def test_validate_wheel_rejects_dashboard_source_metadata_drift(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    wheel = tmp_path / f"dashboard-{field}-mismatch.whl"
    _write_wheel(
        wheel,
        _valid_wheel_names(),
        contents_by_name={
            _WHEEL_DASHBOARD_SOURCE: _dashboard_bundle_with_manifest_value(field, value)
        },
    )

    with pytest.raises(ValueError, match=message):
        validate_wheel(wheel)


def test_validate_wheel_rejects_dashboard_source_compiled_asset_drift(tmp_path: Path) -> None:
    wheel = tmp_path / "dashboard-assets-mismatch.whl"
    _write_wheel(
        wheel,
        _valid_wheel_names(),
        contents_by_name={"cayu/server/dashboard/index.html": "tampered\n"},
    )

    with pytest.raises(ValueError, match="does not match compiled dashboard assets"):
        validate_wheel(wheel)


def test_validate_release_artifacts_require_identical_dashboard_source_bundles(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="dashboard source bundle differs"):
        validate_dashboard_source_equivalence(
            tmp_path / "source.tar.gz",
            tmp_path / "wheel.whl",
            sdist_bundle=b"source",
            wheel_bundle=b"wheel",
        )


def test_validate_wheel_rejects_manifest_inventory_and_digest_mismatches(tmp_path: Path) -> None:
    prefix = artifact_validator["_WHEEL_SIDECAR_PREFIX"]
    for suffix, content, message in (
        ("unexpected.txt", b"unexpected", "unexpected files"),
        ("app.py", b"corrupt", "size mismatch"),
    ):
        wheel = tmp_path / f"{suffix.replace('/', '-')}.whl"
        names = _valid_wheel_names() | {f"{prefix}/{suffix}"}
        sidecar = _canonical_sidecar()
        sidecar[suffix] = content
        _write_wheel(wheel, names, sidecar=sidecar)
        with pytest.raises(ValueError, match=message):
            validate_wheel(wheel)


def test_validate_release_artifacts_accept_manifest_driven_nested_files_and_zip_dirs(
    tmp_path: Path,
) -> None:
    sidecar = _sidecar_with_nested_file()
    sdist = tmp_path / "cayu.tar.gz"
    wheel = tmp_path / "cayu.whl"
    _write_sdist(sdist, sidecar=sidecar)
    names = _valid_wheel_names(sidecar) | {
        f"{artifact_validator['_WHEEL_SIDECAR_PREFIX']}/support/"
    }
    _write_wheel(wheel, names, sidecar=sidecar)
    source_contents = validate_sdist(sdist)
    validate_sidecar_equivalence(sdist, wheel)
    validate_dashboard_source_equivalence(sdist, wheel)
    assert source_contents.sidecar["support/nested.txt"] == b"nested\n"


def test_validate_sidecar_manifest_version_must_match_package_metadata(tmp_path: Path) -> None:
    wheel = tmp_path / "version-mismatch.whl"
    sidecar = _canonical_sidecar()
    manifest = json.loads(sidecar[_MANIFEST])
    manifest["cayu_version"] = "9.9.9"
    sidecar[_MANIFEST] = json.dumps(manifest).encode()
    _write_wheel(wheel, _valid_wheel_names(sidecar), sidecar=sidecar)
    with pytest.raises(ValueError, match="Cayu version mismatch"):
        validate_wheel(wheel)


def test_validate_wheel_rejects_sidecar_symlinks(tmp_path: Path) -> None:
    wheel = tmp_path / "linked.whl"
    linked_name = f"{artifact_validator['_WHEEL_SIDECAR_PREFIX']}/supervisor.py"
    _write_wheel(wheel, _valid_wheel_names(), symlink_name=linked_name)
    with pytest.raises(ValueError, match=r"wheel must not contain links: .*supervisor\.py"):
        validate_wheel(wheel)


def test_validate_sdist_rejects_other_examples(tmp_path: Path) -> None:
    sdist = tmp_path / "unexpected.tar.gz"
    _write_sdist(sdist, additional_names={"examples/aws/unrelated.py"})
    with pytest.raises(ValueError, match="unexpected source-distribution path"):
        validate_sdist(sdist)


def test_validate_release_artifacts_require_identical_sidecar_sources(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="packaged sidecar differs.*app.py"):
        validate_sidecar_equivalence(
            tmp_path / "source.tar.gz",
            tmp_path / "wheel.whl",
            sdist_contents={"app.py": b"source"},
            wheel_contents={"app.py": b"wheel"},
        )


@pytest.mark.parametrize(
    "required_name",
    ["cayu/guides/tool-effects.md", "cayu/server/dashboard/THIRD_PARTY_LICENSES.md"],
)
def test_validate_wheel_requires_publication_files(tmp_path: Path, required_name: str) -> None:
    wheel = tmp_path / "missing.whl"
    names = _valid_wheel_names()
    names.remove(required_name)
    _write_wheel(wheel, names)
    with pytest.raises(ValueError, match="missing required wheel files"):
        validate_wheel(wheel)


def test_validate_wheel_rejects_incomplete_third_party_license_inventory(tmp_path: Path) -> None:
    wheel = tmp_path / "licenses.whl"
    _write_wheel(wheel, _valid_wheel_names(), third_party_notice="MIT")
    with pytest.raises(ValueError, match="third-party license inventory is incomplete"):
        validate_wheel(wheel)


def test_validate_wheel_rejects_non_public_identifiers(tmp_path: Path) -> None:
    wheel = tmp_path / "non-public.whl"
    private_organization = "vertex" + "kg"
    _write_wheel(
        wheel,
        _valid_wheel_names(),
        contents_by_name={"cayu/__init__.py": f'OWNER = "{private_organization}"\n'},
    )
    with pytest.raises(ValueError, match="non-public identifier included in cayu/__init__.py"):
        validate_wheel(wheel)


def test_validate_sdist_rejects_non_public_identifiers_case_insensitively(
    tmp_path: Path,
) -> None:
    sdist = tmp_path / "non-public.tar.gz"
    internal_application = ("lane" + "-" + "agent").upper()
    _write_sdist(sdist, contents_by_name={"README.md": internal_application})
    with pytest.raises(ValueError, match="non-public identifier included"):
        validate_sdist(sdist)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_validate_archives_reject_non_public_identifiers_in_member_paths(
    tmp_path: Path,
    kind: str,
) -> None:
    identifier = "vertex" + "kg"
    if kind == "wheel":
        archive = tmp_path / "non-public.whl"
        _write_wheel(archive, _valid_wheel_names() | {f"cayu/{identifier}.py"})
        validator = validate_wheel
    else:
        archive = tmp_path / "non-public.tar.gz"
        _write_sdist(archive, additional_names={f"src/cayu/{identifier}.py"})
        validator = validate_sdist
    with pytest.raises(ValueError, match="non-public identifier included in archive path"):
        validator(archive)


def test_validate_sdist_rejects_tests_tree(tmp_path: Path) -> None:
    sdist = tmp_path / "tests-tree.tar.gz"
    _write_sdist(sdist, additional_names={"tests/test_leaked.py"})
    with pytest.raises(ValueError, match="unexpected source-distribution path: tests"):
        validate_sdist(sdist)


def test_validate_wheel_rejects_unexpected_top_level_paths(tmp_path: Path) -> None:
    wheel = tmp_path / "top-level.whl"
    _write_wheel(wheel, _valid_wheel_names() | {"tests/test_leaked.py"})
    with pytest.raises(ValueError, match="unexpected wheel top-level paths: tests"):
        validate_wheel(wheel)


def test_validate_wheel_rejects_duplicate_member_paths(tmp_path: Path) -> None:
    wheel = tmp_path / "duplicate.whl"
    _write_wheel(wheel, _valid_wheel_names())
    with (
        zipfile.ZipFile(wheel, "a") as archive,
        pytest.warns(UserWarning, match="Duplicate name"),
    ):
        archive.writestr("cayu/__init__.py", b"duplicate")

    with pytest.raises(ValueError, match="duplicate member paths"):
        validate_wheel(wheel)


def test_sidecar_manifest_generator_detects_and_repairs_stale_manifest(
    tmp_path: Path,
) -> None:
    generator = runpy.run_path(str(_ROOT / "scripts" / "generate_sidecar_manifest.py"))
    source = tmp_path / "sidecar"
    shutil.copytree(_SIDECAR_SOURCE, source)
    generator["main"].__globals__["SOURCE_ROOT"] = source
    (source / "nested").mkdir()
    (source / "nested" / "new.txt").write_text("new\n", encoding="utf-8")

    assert generator["main"](["--check"]) == 1
    assert generator["main"]([]) == 0
    assert generator["main"](["--check"]) == 0
