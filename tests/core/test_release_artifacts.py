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


def test_validate_sdist_rejects_tests_tree(tmp_path) -> None:
    sdist = tmp_path / "cayu-0.1.0.tar.gz"
    files = {
        "LICENSE": "license",
        "NOTICE": "notice",
        "PKG-INFO": "metadata",
        "README.md": "readme",
        "pyproject.toml": "project",
        "src/cayu/__init__.py": "",
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
    dist_info = "cayu-0.1.0.dist-info"
    names = {
        "cayu/__init__.py": "",
        "cayu/cli/__init__.py": "",
        "cayu/server/dashboard/index.html": "",
        "cayu/server/dashboard/assets/app.js": "",
        "cayu/server/dashboard/assets/app.css": "",
        f"{dist_info}/METADATA": "",
        f"{dist_info}/RECORD": "",
        f"{dist_info}/WHEEL": "",
        f"{dist_info}/entry_points.txt": "",
        f"{dist_info}/licenses/LICENSE": "",
        f"{dist_info}/licenses/NOTICE": "",
        "tests/test_leaked.py": "",
    }
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, contents in names.items():
            archive.writestr(name, contents)

    with pytest.raises(ValueError, match="unexpected wheel top-level paths: tests"):
        validate_wheel(wheel)
