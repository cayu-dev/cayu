from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import pytest

from cayu._dashboard_artifacts import (
    BUNDLE_MANIFEST_NAME,
    RELEASE_METADATA_FILE,
    render_release_metadata,
)
from cayu._server_contract_version import SERVER_CONTRACT_VERSION
from cayu.cli import main
from cayu.cli._dashboard_upgrade import (
    UpgradeAction,
    compare_versions,
    dashboard_source_findings,
    find_sidecars,
    plan_dashboard_upgrade,
)
from cayu.cli.dashboard import ValidatedDashboardSource, load_dashboard_source

VERSION = importlib.metadata.version("cayu")
OLD = "0.0.1"
SIDECAR_SUFFIX = f".cayu-{VERSION}"

UNEDITED_CHANGED = "src/main.tsx"
EDITED_UNCHANGED = "biome.json"
EDITED_CHANGED = "vite.config.ts"
ADDED = "src/kit/ui.ts"
REMOVED_UNEDITED = "src/legacy/old.ts"
REMOVED_EDITED = "src/removed-edited.ts"
PROJECT_OWN = "src/parts/custom.tsx"


def _sha(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


@pytest.fixture(scope="module")
def release() -> ValidatedDashboardSource:
    return load_dashboard_source()


def _release_files(release: ValidatedDashboardSource) -> dict[str, bytes]:
    return {
        path: content for path, content in release.contents.items() if path != BUNDLE_MANIFEST_NAME
    }


def _old_manifest(
    release: ValidatedDashboardSource, files: dict[str, bytes], *, version: str = OLD
) -> bytes:
    manifest = json.loads(release.contents[BUNDLE_MANIFEST_NAME])
    manifest["cayu_version"] = version
    manifest["files"] = [
        {"path": path, "sha256": _sha(content), "size": len(content)}
        for path, content in sorted(files.items())
    ]
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()


def _materialize(root: Path, contents: dict[str, bytes]) -> None:
    for path, content in contents.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _project(
    tmp_path: Path,
    release: ValidatedDashboardSource,
    *,
    old: dict[str, bytes],
    edits: dict[str, bytes | None],
    version: str = OLD,
) -> Path:
    """An ejected dashboard from a release with ``old`` files, edited with ``edits``."""
    root = tmp_path / "dashboard"
    project = dict(old)
    for path, content in edits.items():
        if content is None:
            project.pop(path, None)
        else:
            project[path] = content
    project[BUNDLE_MANIFEST_NAME] = _old_manifest(release, old, version=version)
    _materialize(root, project)
    return root


def _every_case(tmp_path: Path, release: ValidatedDashboardSource) -> Path:
    old = _release_files(release)
    old[UNEDITED_CHANGED] = b"old main\n"
    old[EDITED_CHANGED] = b"old vite config\n"
    del old[ADDED]
    old[REMOVED_UNEDITED] = b"legacy\n"
    old[REMOVED_EDITED] = b"original\n"
    old[RELEASE_METADATA_FILE] = render_release_metadata(
        cayu_version=OLD, server_contract_version=SERVER_CONTRACT_VERSION
    )
    return _project(
        tmp_path,
        release,
        old=old,
        edits={
            EDITED_UNCHANGED: b"{}\n",
            EDITED_CHANGED: b"my vite config\n",
            REMOVED_EDITED: b"mine\n",
            PROJECT_OWN: b"export {}\n",
        },
    )


def test_upgrade_applies_every_case(
    tmp_path: Path, release: ValidatedDashboardSource, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _every_case(tmp_path, release)

    assert main(["dashboard", "upgrade", str(root)]) == 1

    files = release.contents
    assert (root / UNEDITED_CHANGED).read_bytes() == files[UNEDITED_CHANGED]
    assert (root / EDITED_UNCHANGED).read_bytes() == b"{}\n"
    assert (root / EDITED_CHANGED).read_bytes() == b"my vite config\n"
    sidecar = root / f"{EDITED_CHANGED}{SIDECAR_SUFFIX}"
    assert sidecar.read_bytes() == files[EDITED_CHANGED]
    assert (root / ADDED).read_bytes() == files[ADDED]
    assert not (root / REMOVED_UNEDITED).exists()
    assert not (root / "src/legacy").exists()
    assert (root / REMOVED_EDITED).read_bytes() == b"mine\n"
    assert (root / PROJECT_OWN).read_bytes() == b"export {}\n"
    assert (root / RELEASE_METADATA_FILE).read_bytes() == files[RELEASE_METADATA_FILE]
    assert (root / BUNDLE_MANIFEST_NAME).read_bytes() == files[BUNDLE_MANIFEST_NAME]
    assert not list(root.rglob(".cayu-upgrade-*"))

    output = capsys.readouterr().out
    assert f"Cayu {OLD} -> {VERSION}" in output
    assert "dry run" not in output
    assert f"{EDITED_CHANGED} <- {EDITED_CHANGED}{SIDECAR_SUFFIX}" in output
    assert f"  {REMOVED_EDITED}" in output

    # Run again: the merge is pending while its sidecar is there, which is left as it is.
    sidecar.write_bytes(b"half merged\n")
    assert main(["dashboard", "upgrade", "--json", str(root)]) == 1
    rerun = json.loads(capsys.readouterr().out)
    assert rerun["from_version"] == VERSION
    assert {item["path"]: item["action"] for item in rerun["files"]} == {
        EDITED_UNCHANGED: "keep",
        EDITED_CHANGED: "merge",
    }
    assert rerun["manual"] == [EDITED_CHANGED]
    assert sidecar.read_bytes() == b"half merged\n"

    sidecar.unlink()
    assert main(["dashboard", "upgrade", "--json", str(root)]) == 0
    rerun = json.loads(capsys.readouterr().out)
    assert {item["path"]: item["action"] for item in rerun["files"]} == {
        EDITED_UNCHANGED: "keep",
        EDITED_CHANGED: "keep",
    }
    assert rerun["manual"] == []


def test_dry_run_prints_the_plan_and_writes_nothing(
    tmp_path: Path, release: ValidatedDashboardSource, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _every_case(tmp_path, release)
    before = _tree(root)

    assert main(["dashboard", "upgrade", "--dry-run", str(root)]) == 1

    assert _tree(root) == before
    output = capsys.readouterr().out
    assert "(dry run; nothing written)" in output
    for action, path in (
        ("replace", UNEDITED_CHANGED),
        ("keep", EDITED_UNCHANGED),
        ("merge", EDITED_CHANGED),
        ("add", ADDED),
        ("remove", REMOVED_UNEDITED),
        ("keep_removed", REMOVED_EDITED),
        ("replace", RELEASE_METADATA_FILE),
        ("replace", BUNDLE_MANIFEST_NAME),
    ):
        assert f"  {action:<12} {path} (" in output, (action, path)
    assert PROJECT_OWN not in output
    assert "npm run check:api" in output


def test_json_plan_lists_every_step_for_an_agent(
    tmp_path: Path, release: ValidatedDashboardSource, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _every_case(tmp_path, release)
    before = _tree(root)

    assert main(["dashboard", "upgrade", "--dry-run", "--json", str(root)]) == 1

    assert _tree(root) == before
    plan = json.loads(capsys.readouterr().out)
    assert plan["directory"] == str(root.resolve())
    assert (plan["from_version"], plan["to_version"], plan["applied"]) == (OLD, VERSION, False)
    steps = {item["path"]: item for item in plan["files"]}
    assert {path: item["action"] for path, item in steps.items()} == {
        UNEDITED_CHANGED: "replace",
        EDITED_UNCHANGED: "keep",
        EDITED_CHANGED: "merge",
        ADDED: "add",
        REMOVED_UNEDITED: "remove",
        REMOVED_EDITED: "keep_removed",
        RELEASE_METADATA_FILE: "replace",
        BUNDLE_MANIFEST_NAME: "replace",
    }
    assert steps[EDITED_CHANGED]["sidecar"] == f"{EDITED_CHANGED}{SIDECAR_SUFFIX}"
    assert plan["manual"] == [REMOVED_EDITED, EDITED_CHANGED]
    assert plan["unchanged"] == len(release.contents) - 6


def test_upgrade_without_edits_rewrites_the_release_metadata_and_manifest(
    tmp_path: Path, release: ValidatedDashboardSource, capsys: pytest.CaptureFixture[str]
) -> None:
    old = _release_files(release)
    old[RELEASE_METADATA_FILE] = render_release_metadata(
        cayu_version=OLD, server_contract_version=SERVER_CONTRACT_VERSION
    )
    root = _project(tmp_path, release, old=old, edits={})

    assert main(["dashboard", "upgrade", str(root)]) == 0

    assert _tree(root) == release.contents
    assert "npm run check:api" in capsys.readouterr().out
    assert main(["dashboard", "upgrade", "--json", str(root)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert (plan["files"], plan["manual"]) == ([], [])


def test_an_edited_generated_file_takes_the_release_version(
    tmp_path: Path, release: ValidatedDashboardSource, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _project(
        tmp_path,
        release,
        old=_release_files(release),
        edits={RELEASE_METADATA_FILE: b"// edited\n"},
    )

    assert main(["dashboard", "upgrade", "--json", str(root)]) == 0

    steps = json.loads(capsys.readouterr().out)["files"]
    assert [item["path"] for item in steps] == [BUNDLE_MANIFEST_NAME, RELEASE_METADATA_FILE]
    assert steps[1] == {
        "path": RELEASE_METADATA_FILE,
        "action": "replace",
        "sidecar": None,
        "pending_sidecars": [],
        "replaces_edit": True,
    }
    assert (root / RELEASE_METADATA_FILE).read_bytes() == release.contents[RELEASE_METADATA_FILE]


def test_upgrade_refuses_a_directory_without_a_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "dashboard"
    root.mkdir()
    (root / "package.json").write_text("{}\n", encoding="utf-8")

    assert main(["dashboard", "upgrade", str(root)]) == 2

    assert f"has no {BUNDLE_MANIFEST_NAME}" in capsys.readouterr().err
    assert main(["dashboard", "upgrade", "--json", str(root)]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "DASHBOARD_UPGRADE_REFUSED"
    assert BUNDLE_MANIFEST_NAME in error["message"]
    assert [path.name for path in root.iterdir()] == ["package.json"]


def test_upgrade_refuses_a_dashboard_from_a_newer_cayu(
    tmp_path: Path, release: ValidatedDashboardSource, capsys: pytest.CaptureFixture[str]
) -> None:
    old = _release_files(release)
    old[UNEDITED_CHANGED] = b"from the future\n"
    root = _project(tmp_path, release, old=old, edits={}, version="99.0.0")
    before = _tree(root)

    assert main(["dashboard", "upgrade", str(root)]) == 2

    assert "ejected from Cayu 99.0.0, newer than the installed Cayu" in capsys.readouterr().err
    assert _tree(root) == before


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need privileges on Windows")
def test_upgrade_refuses_a_link_at_a_dashboard_path(
    tmp_path: Path, release: ValidatedDashboardSource, capsys: pytest.CaptureFixture[str]
) -> None:
    old = _release_files(release)
    old[UNEDITED_CHANGED] = b"old main\n"
    root = _project(tmp_path, release, old=old, edits={})
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "src").rename(tmp_path / "moved-src")
    (root / "src").symlink_to(outside, target_is_directory=True)

    assert main(["dashboard", "upgrade", str(root)]) == 2

    assert "is a link" in capsys.readouterr().err
    assert list(outside.iterdir()) == []


def test_plan_treats_a_project_file_at_an_added_path_or_a_deleted_file_as_edits() -> None:
    plan = plan_dashboard_upgrade(
        {"deleted.ts": _sha(b"a"), "deleted-changed.ts": _sha(b"a")},
        {
            "added.ts": b"release",
            "deleted.ts": b"a",
            "deleted-changed.ts": b"b",
            "current.ts": b"same",
        },
        {"added.ts": _sha(b"project"), "current.ts": _sha(b"same")},
        from_version="1.0.0",
        to_version="1.1.0",
    )

    assert [(step.path, step.action, step.sidecar) for step in plan.steps] == [
        ("added.ts", UpgradeAction.MERGE, "added.ts.cayu-1.1.0"),
        ("deleted-changed.ts", UpgradeAction.MERGE, "deleted-changed.ts.cayu-1.1.0"),
        ("deleted.ts", UpgradeAction.KEEP, None),
    ]
    assert plan.unchanged == ("current.ts",)
    assert [step.path for step in plan.manual] == ["added.ts", "deleted-changed.ts"]


def test_a_merge_left_by_an_earlier_upgrade_stays_pending(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    for name in ("main.tsx", "main.tsx.cayu-0.20.0", "main.tsx.cayu-notes", "main.tsx.bak"):
        (tmp_path / "src" / name).write_text("x\n", encoding="utf-8")
    sidecars = find_sidecars(tmp_path, ["src/main.tsx", "src/other.tsx", "absent/x.ts"])
    assert sidecars == {"src/main.tsx": ("src/main.tsx.cayu-0.20.0", "src/main.tsx.cayu-notes")}

    ejected = {"src/main.tsx": _sha(b"0.20")}
    project = {"src/main.tsx": _sha(b"mine")}
    pending = {"src/main.tsx": ("src/main.tsx.cayu-0.20.0",)}

    # 0.21 leaves the file alone: the 0.20 merge is still listed.
    plan = plan_dashboard_upgrade(
        ejected,
        {"src/main.tsx": b"0.20"},
        project,
        from_version="0.20.0",
        to_version="0.21.0",
        sidecars=pending,
    )
    [step] = plan.manual
    assert (step.action, step.sidecar, step.pending_sidecars) == (
        UpgradeAction.MERGE,
        None,
        ("src/main.tsx.cayu-0.20.0",),
    )

    # 0.21 changes it too: both are listed.
    plan = plan_dashboard_upgrade(
        ejected,
        {"src/main.tsx": b"0.21"},
        project,
        from_version="0.20.0",
        to_version="0.21.0",
        sidecars=pending,
    )
    [step] = plan.manual
    assert (step.sidecar, step.pending_sidecars) == (
        "src/main.tsx.cayu-0.21.0",
        ("src/main.tsx.cayu-0.20.0",),
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("0.19.0", "0.19.1", -1),
        ("0.19.1", "0.19.1", 0),
        ("0.20.0", "0.19.10", 1),
        ("1.0", "1.0.0", 0),
        ("1.0.0rc1", "1.0.0", -1),
        ("1.0.0.dev3", "1.0.0a1", -1),
        ("1.0.0.post1", "1.0.0", 1),
        ("1.0.0+local", "1.0.0", 0),
        ("main", "0.19.1", None),
        ("main", "main", 0),
    ],
)
def test_compare_versions(left: str, right: str, expected: int | None) -> None:
    assert compare_versions(left, right) == expected


def _ejected_manifest(root: Path, directory: str, version: str) -> None:
    manifest = root / directory / BUNDLE_MANIFEST_NAME
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"cayu_version": version}), encoding="utf-8")


def test_check_warns_about_an_ejected_dashboard_from_another_release(tmp_path: Path) -> None:
    _ejected_manifest(tmp_path, "dashboard", OLD)
    _ejected_manifest(tmp_path, "apps/control-plane", VERSION)
    _ejected_manifest(tmp_path, "node_modules/pkg", OLD)

    [finding] = dashboard_source_findings(tmp_path)

    assert finding.code == "DASHBOARD_SOURCE_VERSION_MISMATCH"
    assert finding.severity.value == "warning"
    assert finding.path == f"dashboard/{BUNDLE_MANIFEST_NAME}"
    assert f"ejected from Cayu {OLD}, but the installed Cayu is {VERSION}" in finding.message
    assert finding.hint is not None and "`cayu dashboard upgrade dashboard`" in finding.hint
    assert finding.documentation_anchor == (
        "cayu guide diagnostics#dashboard-source-version-mismatch"
    )


def test_check_skips_a_folder_it_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ejected_manifest(tmp_path, "dashboard", OLD)
    (tmp_path / "db").mkdir()
    is_file = Path.is_file

    def locked(path: Path) -> bool:
        if path.parent.name == "db":
            raise PermissionError(13, "Permission denied", str(path))
        return is_file(path)

    monkeypatch.setattr(Path, "is_file", locked)

    assert [item.path for item in dashboard_source_findings(tmp_path)] == [
        f"dashboard/{BUNDLE_MANIFEST_NAME}"
    ]


def test_check_names_a_newer_dashboard_release_to_install(tmp_path: Path) -> None:
    _ejected_manifest(tmp_path, "dashboard", "99.0.0")

    [finding] = dashboard_source_findings(tmp_path)

    assert finding.hint is not None and "install Cayu 99.0.0" in finding.hint
    assert "cayu dashboard upgrade" not in finding.hint


_FACTORY = """from cayu import CayuApp


def build_app():
    return CayuApp(enable_logging=False)
"""


def test_cayu_check_reports_a_stale_ejected_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "upgrade_project:build_app"\n', encoding="utf-8"
    )
    (project / "upgrade_project.py").write_text(_FACTORY, encoding="utf-8")
    sys.modules.pop("upgrade_project", None)
    _ejected_manifest(project, "dashboard", OLD)
    monkeypatch.chdir(project)

    main(["check", "--json"])

    report = json.loads(capsys.readouterr().out)
    [finding] = [
        item
        for item in report["diagnostics"]
        if item["code"] == "DASHBOARD_SOURCE_VERSION_MISMATCH"
    ]
    assert finding["severity"] == "warning"
    assert "cayu dashboard upgrade dashboard" in finding["hint"]
