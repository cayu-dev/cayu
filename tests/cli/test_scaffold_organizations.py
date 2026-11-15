"""The `organizations` capability: on the agent preset, and as the b2b preset's own.

Organizations go on top of the web app and the app database. They make the app's people the
organizations' members, and leave the Developer dashboard to the Agent's operators. The b2b
preset is the web app plus organizations (`test_scaffold_b2b.py`, and its recorded tree in
`test_scaffold_app_web.py`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.generate_preset import _declared_plan
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_check import _normalized_declared_plan
from cayu.cli.scaffold_plan import ScaffoldPlanError, normalize_application_plan

ORGANIZATIONS = ("--with", "organizations", "--without", "knowledge,memory")
# What b2b projects recorded before web-app and organizations were capabilities.
LEGACY_B2B = [
    "app-database",
    "approvals",
    "artifacts",
    "evals",
    "human-input",
    "observability",
    "recovery",
    "tasks",
]


def _plan(preset: str = "agent", **choices):
    return normalize_application_plan(
        name="acme-desk", agent_name="acme-desk", preset=preset, **choices
    )


def test_organizations_bring_the_web_app_and_the_app_database() -> None:
    plan = _plan(with_capabilities=("organizations",), without_capabilities=("knowledge", "memory"))
    assert {"organizations", "web-app", "app-database", "tasks", "artifacts"} <= set(
        plan.capabilities
    )


def test_organizations_keep_knowledge_and_memory_scoped_to_each_organization() -> None:
    """The agent preset's knowledge and memory stay on, each organization's its own."""

    plan = _plan(with_capabilities=("organizations",))
    assert {"organizations", "knowledge", "memory"} <= set(plan.capabilities)
    files = project_files("acme-desk", application_plan=plan)

    assert 'session_label_keys=("organization",),' in files["knowledge/retrieval.py"]
    assert 'copy_session_labels=("organization",),' in files["tools/registration.py"]
    assert "MembershipAccessPolicy(memberships(), knowledge=True)" in files["policies/access.py"]
    assert "tests/test_organization_knowledge.py" in files
    for guidance in ("README.md", "AGENTS.md"):
        assert "## Knowledge and memory by organization" in files[guidance]
    for path in (
        "knowledge/retrieval.py",
        "tools/registration.py",
        "policies/access.py",
        "tests/test_organization_knowledge.py",
    ):
        compile(files[path], path, "exec")


def test_organizations_with_knowledge_but_no_memory_expect_no_automatic_recall() -> None:
    plan = _plan(with_capabilities=("organizations",), without_capabilities=("memory",))
    assert "knowledge" in plan.capabilities and "memory" not in plan.capabilities
    test = project_files("acme-desk", application_plan=plan)["tests/test_organization_knowledge.py"]
    assert "assert KEPT not in recalled\n" in test
    assert "assert KEPT in recalled\n" not in test
    compile(test, "tests/test_organization_knowledge.py", "exec")


def test_without_organizations_knowledge_is_generated_as_before() -> None:
    """A project without organizations, or organizations without knowledge, is unchanged."""

    plain = project_files("acme-desk", application_plan=_plan())
    assert "session_label_keys" not in plain["knowledge/retrieval.py"]
    assert "copy_session_labels" not in plain["tools/registration.py"]
    without = project_files(
        "acme-desk",
        application_plan=_plan(
            with_capabilities=("organizations",), without_capabilities=("knowledge", "memory")
        ),
    )
    assert (
        without["policies/access.py"]
        == project_files("acme-desk", preset="b2b")["policies/access.py"]
    )
    assert "knowledge=True" not in without["policies/access.py"]
    assert "tests/test_organization_knowledge.py" not in without
    assert "Knowledge and memory by organization" not in without["README.md"]


def test_organizations_need_the_web_app() -> None:
    with pytest.raises(ScaffoldPlanError) as excinfo:
        _plan(
            with_capabilities=("organizations",),
            without_capabilities=("knowledge", "memory", "web-app"),
        )
    assert excinfo.value.code == "capability_dependency_conflict"
    assert "members sign in to the web app" in str(excinfo.value)


@pytest.mark.parametrize("preset", ["coding", "long-running", "service"])
def test_only_the_agent_and_b2b_presets_have_organizations(preset: str) -> None:
    with pytest.raises(ScaffoldPlanError) as excinfo:
        _plan(preset, with_capabilities=("organizations",))
    assert excinfo.value.code == "unsupported_capability"


def test_b2b_cant_leave_its_organizations_out() -> None:
    with pytest.raises(ScaffoldPlanError) as excinfo:
        _plan("b2b", without_capabilities=("organizations",))
    assert excinfo.value.code == "capability_required"
    assert "choose preset agent" in str(excinfo.value)


def test_the_agent_presets_organizations_are_b2bs() -> None:
    agent = project_files(
        "acme-desk",
        application_plan=_plan(
            with_capabilities=("organizations",), without_capabilities=("knowledge", "memory")
        ),
    )
    b2b = project_files("acme-desk", preset="b2b")

    # The organizations' own files, their routes and their app are the same.
    shared = [
        path
        for path in b2b
        if path.startswith(("domain/", "database/", "policies/access.py", "migrations/"))
        or path.startswith(("integrations/", "operations/demo_seed.py", "web/src/"))
        or path
        in (
            "tests/conftest.py",
            "tests/test_http_api.py",
            "tests/test_generated_routes.py",
            "tests/test_app_database.py",
        )
    ]
    assert len(shared) > 40
    assert {path: agent.get(path) for path in shared} == {path: b2b[path] for path in shared}
    # Its people are the members: the operator app's own files aren't there.
    assert "tests/test_web_app.py" not in agent
    # Its smoke test signs a demonstration person in first, as b2b's does.
    assert agent["web/e2e/smoke.spec.ts"] == b2b["web/e2e/smoke.spec.ts"]
    assert "Sign in" in agent["web/e2e/smoke.spec.ts"]
    assert "web/src/agent/task-runs.tsx" not in agent
    serve = tomllib.loads(agent["pyproject.toml"])["tool"]["cayu"]["serve"]
    assert serve == tomllib.loads(b2b["pyproject.toml"])["tool"]["cayu"]["serve"]
    assert serve["auth"] == "cayu.server.environment_auth:OPERATOR_ONLY_AUTH"
    assert "## Organizations and sharing" in agent["README.md"]
    assert "An agent that several businesses share." in agent["README.md"]
    assert '"name": "web",' in agent["web/package.json"]


def test_the_docker_sandboxs_tests_stay_hermetic_with_organizations() -> None:
    plan = _plan(
        execution="docker",
        with_capabilities=("organizations",),
        without_capabilities=("knowledge", "memory"),
    )
    conftest = project_files("acme-desk", application_plan=plan)["tests/conftest.py"]

    # The sandbox's autouse fixture stays, and the PostgreSQL tests get their database.
    assert 'monkeypatch.setattr(sandbox, "START_CONTAINERS", False)' in conftest
    assert "from test_app_database import SKIPPED, database" in conftest
    # And a run without PostgreSQL still names the tests it left out.
    assert "def pytest_terminal_summary(" in conftest
    assert "\n\n\n\n" not in conftest
    compile(conftest, "tests/conftest.py", "exec")


def test_cayu_new_names_the_organizations_people_and_lists_the_capability(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["new", "acme-desk", *ORGANIZATIONS, "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "organizations' people: http://127.0.0.1:8000/" in out
    assert "operators: http://127.0.0.1:8000/" not in out

    assert main(["new", "--list-presets"]) == 0
    (b2b,) = [line for line in capsys.readouterr().out.splitlines() if " b2b " in line]
    assert "The web app plus organizations" in b2b
    assert main(["new", "--explain", "organizations"]) == 0
    explained = capsys.readouterr().out
    assert "Implies: web-app, app-database" in explained
    assert "Conflicts with" not in explained


def test_an_older_b2b_projects_record_still_reads_as_b2b() -> None:
    contract = {
        "preset": "b2b",
        "provider": "neutral",
        "execution": "none",
        "capabilities": LEGACY_B2B,
    }
    current = _plan("b2b").capabilities

    assert _normalized_declared_plan(contract).capabilities == current
    plan = _declared_plan(contract, name="acme-desk", agent_name="acme_desk", ci="github")
    assert plan.capabilities == current
    # Only b2b reads a record that way: an agent project without them has neither.
    agent = _normalized_declared_plan({**contract, "preset": "agent"})
    assert agent.capabilities == tuple(LEGACY_B2B)


def test_a_generated_project_passes_cayu_check_and_its_tests_on_postgres(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    _postgres_server_dsn: str,
) -> None:
    # With the agent preset's knowledge and memory, scoped to each organization.
    assert main(["new", "acme-desk", "--with", "organizations", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "acme-desk"
    monkeypatch.chdir(project)
    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []

    environment = {
        key: value
        for key, value in os.environ.items()
        # CI requires its own PostgreSQL tests; each run here says what it requires.
        if not key.startswith(("CAYU_DATABASE_", "CAYU_TEST_POSTGRES", "CAYU_REQUIRE_POSTGRES"))
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(Path(__file__).resolve().parents[2] / "src"), environment.get("PYTHONPATH")),
        )
    )
    environment["CAYU_TEST_POSTGRES_DSN"] = _postgres_server_dsn
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "-p",
            "no:cacheprovider",
            "tests/test_organizations.py",
            "tests/test_http_api.py",
            "tests/test_generated_routes.py",
            "tests/test_background_revocation.py",
            "tests/test_app_database.py",
            "tests/test_organization_knowledge.py",
        ],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SKIPPED" not in result.stdout
