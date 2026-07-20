from __future__ import annotations

import io
import runpy
import tarfile
import zipfile
from pathlib import Path

import pytest

artifact_validator = runpy.run_path(
    str(Path(__file__).parents[2] / "scripts" / "check_release_artifacts.py")
)
validate_sdist = artifact_validator["validate_sdist"]
validate_wheel = artifact_validator["validate_wheel"]

_WHEEL_DIST_INFO = "cayu-0.1.0.dist-info"


def _valid_wheel_names() -> set[str]:
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
        f"{_WHEEL_DIST_INFO}/METADATA",
        f"{_WHEEL_DIST_INFO}/RECORD",
        f"{_WHEEL_DIST_INFO}/WHEEL",
        f"{_WHEEL_DIST_INFO}/entry_points.txt",
        f"{_WHEEL_DIST_INFO}/licenses/LICENSE",
        f"{_WHEEL_DIST_INFO}/licenses/NOTICE",
    }


def _write_wheel(
    path: Path,
    names: set[str],
    *,
    contents_by_name: dict[str, str] | None = None,
    third_party_notice: str | None = None,
) -> None:
    contents_by_name = contents_by_name or {}
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            contents = contents_by_name.get(name, "")
            if name == "cayu/server/dashboard/THIRD_PARTY_LICENSES.md":
                contents = (
                    third_party_notice
                    if third_party_notice is not None
                    else "\n".join(artifact_validator["_THIRD_PARTY_LICENSE_MARKERS"])
                )
            archive.writestr(name, contents)


def _write_sdist(
    path: Path,
    *,
    additional_names: set[str] | None = None,
    contents_by_name: dict[str, str] | None = None,
) -> None:
    additional_names = additional_names or set()
    contents_by_name = contents_by_name or {}
    names = artifact_validator["_SDIST_REQUIRED"] | additional_names
    third_party_notice = "\n".join(artifact_validator["_THIRD_PARTY_LICENSE_MARKERS"])
    with tarfile.open(path, "w:gz") as archive:
        for relative_name in names:
            contents = contents_by_name.get(relative_name, "")
            if relative_name == "src/cayu/server/dashboard/THIRD_PARTY_LICENSES.md":
                contents = third_party_notice
            data = contents.encode()
            member = tarfile.TarInfo(f"cayu-0.1.0/{relative_name}")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def test_validate_wheel_requires_application_anatomy_guide(tmp_path) -> None:
    wheel = tmp_path / "cayu-0.1.0-py3-none-any.whl"
    names = _valid_wheel_names()
    names.remove("cayu/guides/application-anatomy.md")
    _write_wheel(wheel, names)

    with pytest.raises(
        ValueError,
        match=r"missing required wheel files: cayu/guides/application-anatomy\.md",
    ):
        validate_wheel(wheel)


def test_validate_wheel_requires_tool_effect_guide(tmp_path) -> None:
    wheel = tmp_path / "missing-tool-effects-guide.whl"
    names = _valid_wheel_names()
    names.remove("cayu/guides/tool-effects.md")
    _write_wheel(wheel, names)

    with pytest.raises(ValueError, match="missing required wheel files"):
        validate_wheel(wheel)


def test_validate_wheel_requires_third_party_license_inventory(tmp_path) -> None:
    wheel = tmp_path / "missing-third-party-licenses.whl"
    names = _valid_wheel_names()
    names.remove("cayu/server/dashboard/THIRD_PARTY_LICENSES.md")
    _write_wheel(wheel, names)

    with pytest.raises(ValueError, match="missing required wheel files"):
        validate_wheel(wheel)


def test_validate_wheel_rejects_incomplete_third_party_license_inventory(tmp_path) -> None:
    wheel = tmp_path / "incomplete-third-party-licenses.whl"
    _write_wheel(wheel, _valid_wheel_names(), third_party_notice="MIT")

    with pytest.raises(ValueError, match="third-party license inventory is incomplete"):
        validate_wheel(wheel)


def test_validate_wheel_rejects_non_public_identifiers(tmp_path) -> None:
    wheel = tmp_path / "non-public-identifier.whl"
    private_organization = "vertex" + "kg"
    _write_wheel(
        wheel,
        _valid_wheel_names(),
        contents_by_name={"cayu/__init__.py": f'REPOSITORY_OWNER = "{private_organization}"\n'},
    )

    with pytest.raises(
        ValueError,
        match="non-public identifier included in cayu/__init__.py",
    ):
        validate_wheel(wheel)


def test_validate_sdist_rejects_non_public_identifiers_case_insensitively(
    tmp_path,
) -> None:
    sdist = tmp_path / "non-public-identifier.tar.gz"
    internal_application = ("lane" + "-" + "agent").upper()
    _write_sdist(
        sdist,
        contents_by_name={"README.md": f"Internal consumer: {internal_application}\n"},
    )

    with pytest.raises(
        ValueError,
        match=r"non-public identifier included in cayu-0\.1\.0/README\.md",
    ):
        validate_sdist(sdist)


def test_validate_wheel_rejects_non_public_identifiers_in_member_paths(tmp_path) -> None:
    wheel = tmp_path / "non-public-path.whl"
    private_organization = ("vertex" + "kg").upper()
    _write_wheel(
        wheel,
        _valid_wheel_names() | {f"cayu/{private_organization}.py"},
    )

    with pytest.raises(
        ValueError,
        match="non-public identifier included in archive path",
    ):
        validate_wheel(wheel)


def test_validate_sdist_rejects_non_public_identifiers_in_member_paths(tmp_path) -> None:
    sdist = tmp_path / "non-public-path.tar.gz"
    internal_application = "lane" + "-" + "agent"
    _write_sdist(
        sdist,
        additional_names={f"src/cayu/{internal_application}.py"},
    )

    with pytest.raises(
        ValueError,
        match="non-public identifier included in archive path",
    ):
        validate_sdist(sdist)


def test_validate_sdist_rejects_tests_tree(tmp_path) -> None:
    sdist = tmp_path / "cayu-0.1.0.tar.gz"
    files = {
        "LICENSE": "license",
        "NOTICE": "notice",
        "PKG-INFO": "metadata",
        "README.md": "readme",
        "pyproject.toml": "project",
        "src/cayu/__init__.py": "",
        "src/cayu/data/__init__.py": "",
        "src/cayu/data/default_model_catalog.json": "{}",
        "src/cayu/data/default_price_book.json": "{}",
        "tests/test_leaked.py": "",
    }
    with tarfile.open(sdist, "w:gz") as archive:
        for relative_name, contents in files.items():
            data = contents.encode()
            member = tarfile.TarInfo(f"cayu-0.1.0/{relative_name}")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    with pytest.raises(ValueError, match="unexpected source-distribution path: tests"):
        validate_sdist(sdist)


def test_validate_wheel_rejects_unexpected_top_level_paths(tmp_path) -> None:
    wheel = tmp_path / "cayu-0.1.0-py3-none-any.whl"
    names = _valid_wheel_names() | {"tests/test_leaked.py"}
    _write_wheel(wheel, names)

    with pytest.raises(ValueError, match="unexpected wheel top-level paths: tests"):
        validate_wheel(wheel)
