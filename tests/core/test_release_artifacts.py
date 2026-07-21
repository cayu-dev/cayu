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

_ROOT = Path(__file__).parents[2]
_SIDECAR_SOURCE = _ROOT / "examples" / "aws" / "lambda_microvm_sidecar"
_MANIFEST = "cayu-lambda-microvm-sidecar-manifest.json"
with (_ROOT / "pyproject.toml").open("rb") as _project_file:
    _VERSION = tomllib.load(_project_file)["project"]["version"]
_WHEEL_DIST_INFO = f"cayu-{_VERSION}.dist-info"

artifact_validator = runpy.run_path(str(_ROOT / "scripts" / "check_release_artifacts.py"))
validate_sdist = artifact_validator["validate_sdist"]
validate_wheel = artifact_validator["validate_wheel"]
validate_sidecar_equivalence = artifact_validator["validate_sidecar_equivalence"]


def _canonical_sidecar() -> dict[str, bytes]:
    return {
        path.relative_to(_SIDECAR_SOURCE).as_posix(): path.read_bytes()
        for path in _SIDECAR_SOURCE.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    }


def _valid_wheel_names(sidecar: dict[str, bytes] | None = None) -> set[str]:
    sidecar = sidecar or _canonical_sidecar()
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
        "cayu/guides/tool-effects.md",
        "cayu/server/dashboard/THIRD_PARTY_LICENSES.md",
        "cayu/server/dashboard/index.html",
        "cayu/server/dashboard/assets/app.js",
        "cayu/server/dashboard/assets/app.css",
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
    contents_by_name = contents_by_name or {}
    prefix = f"{artifact_validator['_WHEEL_SIDECAR_PREFIX']}/"
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            content: str | bytes = contents_by_name.get(name, "")
            if name.startswith(prefix) and name.removeprefix(prefix) in sidecar:
                content = sidecar[name.removeprefix(prefix)]
            if name == "cayu/server/dashboard/THIRD_PARTY_LICENSES.md":
                content = (
                    third_party_notice
                    if third_party_notice is not None
                    else "\n".join(artifact_validator["_THIRD_PARTY_LICENSE_MARKERS"])
                )
            if name == f"{_WHEEL_DIST_INFO}/METADATA":
                content = f"Metadata-Version: 2.4\nName: cayu\nVersion: {_VERSION}\n"
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
    additional_names = additional_names or set()
    contents_by_name = contents_by_name or {}
    prefix = artifact_validator["_SDIST_SIDECAR_PREFIX"]
    names = (
        artifact_validator["_SDIST_REQUIRED"]
        | additional_names
        | {f"{prefix}/{name}" for name in sidecar}
    )
    notice = "\n".join(artifact_validator["_THIRD_PARTY_LICENSE_MARKERS"])
    with tarfile.open(path, "w:gz") as archive:
        for relative_name in names:
            content: str | bytes = contents_by_name.get(relative_name, "")
            if relative_name.startswith(f"{prefix}/"):
                sidecar_name = relative_name.removeprefix(f"{prefix}/")
                if sidecar_name in sidecar:
                    content = sidecar[sidecar_name]
            if relative_name == "src/cayu/server/dashboard/THIRD_PARTY_LICENSES.md":
                content = notice
            if relative_name == "PKG-INFO":
                content = f"Metadata-Version: 2.4\nName: cayu\nVersion: {_VERSION}\n"
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
        protocol_version="1",
    )
    return sidecar


def test_validate_wheel_requires_application_anatomy_guide(tmp_path: Path) -> None:
    wheel = tmp_path / "cayu.whl"
    names = _valid_wheel_names()
    names.remove("cayu/guides/application-anatomy.md")
    _write_wheel(wheel, names)
    with pytest.raises(ValueError, match=r"missing required wheel files: .*application-anatomy"):
        validate_wheel(wheel)


def test_validate_wheel_requires_sidecar_manifest(tmp_path: Path) -> None:
    wheel = tmp_path / "missing-sidecar.whl"
    names = _valid_wheel_names()
    names.remove(f"{artifact_validator['_WHEEL_SIDECAR_PREFIX']}/{_MANIFEST}")
    _write_wheel(wheel, names)
    with pytest.raises(ValueError, match="missing required wheel files"):
        validate_wheel(wheel)


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
    wheel_contents = validate_wheel(wheel)
    validate_sidecar_equivalence(
        sdist,
        wheel,
        sdist_contents=source_contents,
        wheel_contents=wheel_contents,
    )
    assert source_contents["support/nested.txt"] == b"nested\n"


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
