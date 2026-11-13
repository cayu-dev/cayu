from __future__ import annotations

import importlib.metadata
import json
import stat
import sys
from pathlib import Path

import pytest

from cayu._dashboard_artifacts import (
    PARTS_PACKAGE_NAME,
    parts_tarball_name,
    validate_parts_tarball,
)
from cayu.cli import _dashboard_parts, main
from cayu.cli._dashboard_parts import dashboard_parts_findings, packaged_parts_tarball

VERSION = importlib.metadata.version("cayu")
NAME = parts_tarball_name(VERSION)


def _web(root: Path, dependencies: dict[str, str] | None) -> Path:
    web = root / "web"
    (web / "vendor").mkdir(parents=True)
    manifest: dict[str, object] = {"name": "app", "private": True}
    if dependencies is not None:
        manifest["dependencies"] = dependencies
    (web / "package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return web


def test_the_installed_cayu_ships_its_own_version_of_the_parts() -> None:
    version, name, content = packaged_parts_tarball()

    assert (version, name) == (VERSION, NAME)
    validate_parts_tarball(content, cayu_version=VERSION)


def test_parts_writes_the_tarball_and_points_the_file_spec_at_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    web = _web(
        tmp_path,
        {"react": "^19.2.6", PARTS_PACKAGE_NAME: "file:vendor/cayu-dashboard-parts-0.0.1.tgz"},
    )
    (web / "vendor" / "cayu-dashboard-parts-0.0.1.tgz").write_bytes(b"old")
    (web / "vendor" / "other.tgz").write_bytes(b"keep")
    monkeypatch.chdir(tmp_path)

    assert main(["dashboard", "parts"]) == 0

    assert sorted(path.name for path in (web / "vendor").iterdir()) == sorted([NAME, "other.tgz"])
    assert (web / "vendor" / NAME).read_bytes() == packaged_parts_tarball()[2]
    manifest = json.loads((web / "package.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == {
        "react": "^19.2.6",
        PARTS_PACKAGE_NAME: f"file:vendor/{NAME}",
    }
    assert (web / "package.json").read_text(encoding="utf-8").startswith('{\n  "name"')
    out = capsys.readouterr().out
    assert f"wrote web/vendor/{NAME}" in out
    assert "removed web/vendor/cayu-dashboard-parts-0.0.1.tgz" in out
    assert "run `npm install`" in out


@pytest.mark.parametrize(
    "dependencies", [None, {"react": "^19.2.6"}, {PARTS_PACKAGE_NAME: "^0.19.0"}]
)
def test_parts_leaves_a_package_json_without_a_file_spec_alone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dependencies: dict[str, str] | None,
) -> None:
    web = _web(tmp_path, dependencies)
    before = (web / "package.json").read_bytes()
    monkeypatch.chdir(tmp_path)

    assert main(["dashboard", "parts"]) == 0

    assert (web / "package.json").read_bytes() == before
    assert (web / "vendor" / NAME).is_file()
    assert f"npm install ./vendor/{NAME}" in capsys.readouterr().out


def test_parts_writes_to_an_explicit_destination(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "somewhere" / "vendor"

    assert main(["dashboard", "parts", str(destination)]) == 0

    assert [path.name for path in destination.iterdir()] == [NAME]
    assert f"npm install {destination / NAME}" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_parts_writes_a_tarball_others_can_read(tmp_path: Path) -> None:
    destination = tmp_path / "vendor"

    assert main(["dashboard", "parts", str(destination)]) == 0

    assert stat.S_IMODE((destination / NAME).stat().st_mode) == 0o644


def test_parts_needs_a_web_folder_for_its_default_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["dashboard", "parts"]) == 1

    assert "there is no web/ folder here" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_parts_refuses_when_the_installed_cayu_lacks_its_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_dashboard_parts, "_installed_version", lambda: "9.9.9")

    assert main(["dashboard", "parts", str(tmp_path / "vendor")]) == 1

    assert "does not include @cayu-tech/dashboard-parts" in capsys.readouterr().err
    assert not (tmp_path / "vendor").exists()


def _lock(root: Path, lock: dict[str, object]) -> None:
    (root / "web").mkdir(exist_ok=True)
    (root / "web" / "package-lock.json").write_text(json.dumps(lock), encoding="utf-8")


def _v3(version: str) -> dict[str, object]:
    return {
        "lockfileVersion": 3,
        "packages": {
            "": {"dependencies": {PARTS_PACKAGE_NAME: f"file:vendor/{NAME}"}},
            f"node_modules/{PARTS_PACKAGE_NAME}": {
                "version": version,
                "resolved": f"file:vendor/cayu-dashboard-parts-{version}.tgz",
            },
        },
    }


def test_check_warns_when_the_lockfile_resolves_another_version(tmp_path: Path) -> None:
    _lock(tmp_path, _v3("0.0.1"))

    [finding] = dashboard_parts_findings(tmp_path)

    assert finding.code == "DASHBOARD_PARTS_VERSION_MISMATCH"
    assert finding.severity.value == "warning"
    assert finding.path == "web/package-lock.json"
    assert f"to 0.0.1, but the installed Cayu is {VERSION}" in finding.message
    assert finding.hint is not None and "cayu dashboard parts" in finding.hint


@pytest.mark.parametrize(
    "lock",
    [
        _v3(VERSION),
        {"lockfileVersion": 3, "packages": {"": {}, "node_modules/react": {"version": "19"}}},
        {"lockfileVersion": 1, "dependencies": {PARTS_PACKAGE_NAME: {"version": VERSION}}},
        # npm writes a version 1 lockfile's local tarball as its file: spec.
        {
            "lockfileVersion": 1,
            "dependencies": {PARTS_PACKAGE_NAME: {"version": f"file:vendor/{NAME}"}},
        },
        {
            "lockfileVersion": 1,
            "dependencies": {PARTS_PACKAGE_NAME: {"version": "file:vendor/renamed.tgz"}},
        },
    ],
)
def test_check_is_quiet_when_the_lockfile_matches_or_lacks_the_parts(
    tmp_path: Path, lock: dict[str, object]
) -> None:
    _lock(tmp_path, lock)

    assert dashboard_parts_findings(tmp_path) == ()


def test_check_reads_a_version_1_lockfile_and_ignores_no_or_unreadable_ones(
    tmp_path: Path,
) -> None:
    assert dashboard_parts_findings(tmp_path) == ()
    _lock(tmp_path, {"lockfileVersion": 1, "dependencies": {PARTS_PACKAGE_NAME: {"version": "1"}}})
    assert [item.code for item in dashboard_parts_findings(tmp_path)] == [
        "DASHBOARD_PARTS_VERSION_MISMATCH"
    ]
    _lock(
        tmp_path,
        {
            "lockfileVersion": 1,
            "dependencies": {
                PARTS_PACKAGE_NAME: {"version": "file:vendor/cayu-dashboard-parts-0.0.1.tgz"}
            },
        },
    )
    [finding] = dashboard_parts_findings(tmp_path)
    assert finding.parameters["locked_version"] == "0.0.1"
    (tmp_path / "web" / "package-lock.json").write_text("{", encoding="utf-8")
    assert dashboard_parts_findings(tmp_path) == ()


_FACTORY = """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)
"""


def test_cayu_check_reports_the_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "parts_project:build_app"\n', encoding="utf-8"
    )
    (project / "parts_project.py").write_text(_FACTORY, encoding="utf-8")
    sys.modules.pop("parts_project", None)
    _lock(project, _v3("0.0.1"))
    monkeypatch.chdir(project)

    main(["check", "--json"])

    report = json.loads(capsys.readouterr().out)
    [finding] = [
        item for item in report["diagnostics"] if item["code"] == "DASHBOARD_PARTS_VERSION_MISMATCH"
    ]
    assert finding["severity"] == "warning"
