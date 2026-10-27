"""`cayu new --preset b2b`: organizations, departments and members in the app's own tables,
turned into Cayu resource-access grants."""

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

B2B_FILES = (
    "integrations/sign_in.py",
    "integrations/http.py",
    "migrations/versions/0003_membership_roles.py",
    "tests/test_http_api.py",
    "tests/test_background_revocation.py",
    "domain/organizations.py",
    "database/memberships.py",
    "domain/sharing.py",
    "policies/access.py",
    "operations/demo_seed.py",
    "migrations/versions/0002_organizations.py",
    "tests/test_organizations.py",
)


def test_dry_run_lists_the_preset_with_its_files_and_capabilities(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = ["new", "acme-desk", "--preset", "b2b", "--dir", str(tmp_path), "--dry-run"]
    assert main([*command, "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]

    assert plan["preset"] == "b2b"
    assert {"app-database", "tasks", "artifacts"} <= set(plan["capabilities"])
    assert set(B2B_FILES) <= set(plan["files"])
    assert not (tmp_path / "acme-desk").exists()


def test_the_app_is_built_with_the_membership_policy() -> None:
    files = project_files("acme-desk", preset="b2b")

    assert set(B2B_FILES) <= set(files)
    assert "resource_access_policy=" in files["app.py"]
    assert "from policies.access import build_access_policy" in files["app.py"]
    dependencies = tomllib.loads(files["pyproject.toml"])["project"]["dependencies"]
    assert any(item.startswith("alembic") for item in dependencies)
    assert "class Membership(Base)" in files["database/models.py"]
    assert "## Organizations and sharing" in files["AGENTS.md"]
    serve = tomllib.loads(files["pyproject.toml"])["tool"]["cayu"]["serve"]
    assert serve["routers"] == ["integrations.http:build_router"]
    assert serve["auth"] == "cayu.server.environment_auth:OPERATOR_ONLY_AUTH"
    assert "role: Mapped[str]" in files["database/models.py"]
    # The sharing labels Cayu enforces can't be changed by the people they protect.
    assert "protected_label_keys" in files["policies/access.py"]


def test_every_default_capability_can_also_be_named() -> None:
    from cayu.cli.scaffold_plan import PRESETS, normalize_application_plan

    (b2b,) = [spec for spec in PRESETS if spec.name == "b2b"]
    plan = normalize_application_plan(
        name="acme-desk",
        agent_name="acme_desk",
        preset="b2b",
        with_capabilities=b2b.default_capabilities,
    )
    assert set(b2b.default_capabilities) <= set(plan.capabilities)


def test_the_app_database_capability_cant_be_left_out() -> None:
    from cayu.cli.scaffold_plan import ScaffoldPlanError, normalize_application_plan

    with pytest.raises(ScaffoldPlanError, match="app-database"):
        normalize_application_plan(
            name="acme-desk",
            agent_name="acme_desk",
            preset="b2b",
            without_capabilities=("app-database",),
        )


def test_generated_suite_runs_without_a_database_and_against_postgres(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    _postgres_server_dsn: str,
) -> None:
    assert main(["new", "acme-desk", "--preset", "b2b", "--dir", str(tmp_path), "--json"]) == 0
    capsys.readouterr()
    project = tmp_path / "acme-desk"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CAYU_DATABASE_", "CAYU_TEST_POSTGRES"))
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(Path(__file__).resolve().parents[2] / "src"), environment.get("PYTHONPATH")),
        )
    )

    def generated_suite(extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-v",
                "-p",
                "no:cacheprovider",
                "tests/test_organizations.py",
                "tests/test_http_api.py",
                "tests/test_background_revocation.py",
                "tests/test_app_database.py",
            ],
            cwd=project,
            env={**environment, **extra},
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )

    # Without a database the policy's own tests pass and the PostgreSQL ones skip.
    offline = generated_suite({})
    assert offline.returncode == 0, offline.stdout + offline.stderr
    assert "SKIPPED" in offline.stdout

    online = generated_suite({"CAYU_TEST_POSTGRES_DSN": _postgres_server_dsn})
    assert online.returncode == 0, online.stdout + online.stderr
    assert "SKIPPED" not in online.stdout


def test_a_generated_project_passes_cayu_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["new", "acme-desk", "--preset", "b2b", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    monkeypatch.chdir(tmp_path / "acme-desk")

    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []
