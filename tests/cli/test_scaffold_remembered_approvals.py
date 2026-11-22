"""`cayu new --with remembered-approvals`: tool authorization grants for the approvals."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_check import check_declared_scaffold_source

PRESETS = ("agent", "long-running", "b2b")
OWNED = (
    "policies/grants.py",
    "policies/REMEMBERED_APPROVALS.md",
    "tests/test_remembered_approvals.py",
)
CHANGED = (
    "app.py",
    "agents/registration.py",
    "configuration/settings.py",
    "configuration/storage.py",
    "policies/tools.py",
)


def _with_grants(preset: str) -> dict[str, str]:
    return project_files("demo", preset=preset, with_capabilities=("remembered-approvals",))


def test_dry_run_lists_the_capability_its_files_and_environment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = ["new", "demo", "--with", "remembered-approvals", "--dir", str(tmp_path)]
    assert main([*command, "--dry-run", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert "remembered-approvals" in plan["capabilities"]
    assert set(OWNED) <= set(plan["files"])
    assert "CAYU_GRANT_OWNERS" in plan["environment"]
    assert not (tmp_path / "demo").exists()


@pytest.mark.parametrize("preset", PRESETS)
def test_every_supported_preset_wires_the_grants(preset: str) -> None:
    files = _with_grants(preset)
    plain = project_files("demo", preset=preset)
    assert set(OWNED) <= set(files)
    assert not set(OWNED) & set(plain)
    for relative in CHANGED:
        assert files[relative] != plain[relative], relative
    scaffold = tomllib.loads(files["pyproject.toml"])["tool"]["cayu"]["scaffold"]
    assert {"approvals", "remembered-approvals"} <= set(scaffold["capabilities"])
    assert "host_remembered_approvals(app, stores.grant_store)" in files["app.py"]
    assert "tool_authorization_grants=session_store is None" in files["configuration/storage.py"]
    assert "approval_gate=APPROVAL_GATE" in files["policies/tools.py"]
    assert 'POLICY_ID = "demo.external-effects"' in files["policies/grants.py"]
    card = files["policies/REMEMBERED_APPROVALS.md"]
    assert "CAYU_GRANT_OWNERS" in card and "never member" in card
    # `configuration/settings.py` stays the project's only environment reader.
    assert "os.environ" not in files["policies/grants.py"]


@pytest.mark.parametrize(
    ("preset", "without", "message"),
    (
        ("service", (), "service_factory"),
        ("coding", (), "not supported by preset 'coding'"),
        ("agent", ("approvals",), "requires excluded capability 'approvals'"),
    ),
)
def test_unsupported_plans_are_refused_before_anything_is_written(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    preset: str,
    without: tuple[str, ...],
    message: str,
) -> None:
    command = ["new", "demo", "--preset", preset, "--with", "remembered-approvals"]
    for name in without:
        command += ["--without", name]
    assert main([*command, "--dir", str(tmp_path)]) != 0
    assert message in capsys.readouterr().err
    assert not (tmp_path / "demo").exists()


@pytest.mark.parametrize("preset", PRESETS)
def test_rendered_files_pass_ruff(tmp_path: Path, preset: str) -> None:
    files = _with_grants(preset)
    for relative in (*CHANGED, "policies/grants.py", "tests/test_remembered_approvals.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[relative], encoding="utf-8")
    # The new modules and the storage seam are formatted; the rest only has to pass checks.
    formatted = ("policies", "tests", "configuration/storage.py")
    for command in (("check", "."), ("format", "--check", *formatted)):
        result = subprocess.run(
            [sys.executable, "-m", "ruff", *command, "--isolated", "--no-cache"],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("selected", (False, True))
def test_check_reports_grant_hosting_that_disagrees_with_the_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], selected: bool
) -> None:
    command = ["new", "demo", "--dir", str(tmp_path), "--ci", "none"]
    if selected:
        command += ["--with", "remembered-approvals"]
    assert main(command) == 0
    capsys.readouterr()
    project = tmp_path / "demo"
    app = project / "app.py"
    source = app.read_text(encoding="utf-8")
    assert all(
        item.code != "SCAFFOLD_PLAN_DRIFT" for item in check_declared_scaffold_source(project)
    )
    hosting = "        host_remembered_approvals(app, stores.grant_store)\n"
    app.write_text(
        source.replace(hosting, "")
        if selected
        else source.replace("        return app\n", hosting + "        return app\n"),
        encoding="utf-8",
    )
    [drift] = [
        item
        for item in check_declared_scaffold_source(project)
        if item.code == "SCAFFOLD_PLAN_DRIFT"
    ]
    assert drift.path == "app.py"
    assert drift.parameters["field"] == "remembered-approvals"


# Each supported preset, and the capabilities the grant files and test touch turned
# off where the preset allows it, alone and together.
COMBINATIONS = (
    ("agent", ()),
    ("long-running", ()),
    ("b2b", ()),
    ("agent", ("--without", "tasks")),
    ("agent", ("--without", "knowledge,memory,evals,observability")),
    ("agent", ("--without", "human-input,artifacts,recovery,web-app")),
    (
        "agent",
        (
            "--without",
            "tasks,knowledge,memory,evals,observability,human-input,artifacts,recovery,web-app",
        ),
    ),
    ("agent", ("--with", "organizations")),
    ("long-running", ("--without", "evals,observability,recovery,web-app")),
    ("b2b", ("--without", "evals,observability,recovery,human-input")),
)


@pytest.mark.parametrize(("preset", "flags"), COMBINATIONS)
def test_generated_project_remembers_an_approval_and_passes_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], preset: str, flags: tuple[str, ...]
) -> None:
    command = ["new", "demo", "--preset", preset, "--with", "remembered-approvals", *flags]
    assert main([*command, "--dir", str(tmp_path), "--ci", "none"]) == 0
    capsys.readouterr()
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CAYU_DATABASE_", "CAYU_REQUIRE_POSTGRES", "CAYU_GRANT_OWNERS"))
    }
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_root, environment.get("PYTHONPATH")))
    )
    for arguments in (
        ("pytest", "-q", "-p", "no:cacheprovider", "tests/test_remembered_approvals.py"),
        ("cayu.cli", "check", "--fail-on", "warning"),
    ):
        result = subprocess.run(
            [sys.executable, "-m", *arguments],
            cwd=tmp_path / "demo",
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stdout + result.stderr
