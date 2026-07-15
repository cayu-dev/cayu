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
        "cayu/guides/application-anatomy.md",
        "cayu/guides/authoring.md",
        "cayu/guides/diagnostics.md",
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


def _write_wheel(path: Path, names: set[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, "")


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
