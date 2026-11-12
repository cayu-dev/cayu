"""`cayu new --preset b2b`: organizations, departments and members in the app's own tables,
turned into Cayu resource-access grants."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.scaffold import project_files

B2B_FILES = (
    "integrations/sign_in.py",
    "integrations/http/__init__.py",
    "integrations/http/generated/__init__.py",
    "integrations/http/generated/cayu_api.py",
    "integrations/http/generated/inbox.py",
    "integrations/http/generated/overview.py",
    "integrations/http/generated/settings.py",
    "integrations/http/generated/task_runs.py",
    "integrations/http/generated/whoami.py",
    "integrations/http/fixed_settings.py",
    "integrations/http/overview_metrics.py",
    "web/package.json",
    "web/package-lock.json",
    "web/src/main.tsx",
    "web/src/app/layout.tsx",
    "web/src/agent/index.tsx",
    "web/src/generated/overview/index.tsx",
    "web/src/generated/settings/index.tsx",
    "web/src/kit/shadcn/ui.ts",
    "web/static/index.html",
    "migrations/versions/0003_membership_roles.py",
    "tests/test_http_api.py",
    "tests/test_generated_routes.py",
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
    # Its routing table sends the people's interface and new members to this preset's own
    # answers, not to client.js or the service preset.
    assert "| A web UI for the organizations' people |" in files["AGENTS.md"]
    assert "`client.js`. Do not poll" not in files["AGENTS.md"]
    assert "--preset service" not in files["AGENTS.md"]
    serve = tomllib.loads(files["pyproject.toml"])["tool"]["cayu"]["serve"]
    assert serve["routers"] == ["integrations.http:build_router"]
    assert serve["auth"] == "cayu.server.environment_auth:OPERATOR_ONLY_AUTH"
    assert "role: Mapped[str]" in files["database/models.py"]
    # The sharing labels Cayu enforces can't be changed by the people they protect.
    assert "protected_label_keys" in files["policies/access.py"]


def test_the_web_interface_is_copied_with_its_source_and_build() -> None:
    files = project_files("acme-desk", preset="b2b")
    template = Path(__file__).resolve().parents[2] / "src/cayu/cli/templates/b2b_web"
    packaged = {
        f"web/{path.relative_to(template).as_posix()}": path.read_text(encoding="utf-8")
        for path in template.rglob("*")
        if path.is_file()
    }

    # Copied as they are, but for the pages' title: the project's name, which the app shows.
    for page in ("web/index.html", "web/static/index.html"):
        assert "<title>Agent</title>" in packaged[page]
        packaged[page] = packaged[page].replace("<title>Agent</title>", "<title>acme-desk</title>")
    assert packaged == {name: text for name, text in files.items() if name.startswith("web/")}
    # The page names the built files it loads, so they're served as they were built.
    built = {name for name in packaged if name.startswith("web/static/assets/")}
    assert {f"web/static/{name}" for name in _page_assets(files["web/static/index.html"])} == built
    # It starts conversations with the project's agent.
    assert 'AGENT_NAME = "acme-desk"' in files["integrations/http/__init__.py"]


def test_the_web_app_routes_in_the_browser_and_its_build_is_checked() -> None:
    files = project_files("acme-desk", preset="b2b")

    # Its files load from / on every page, and `cayu serve` answers every page there.
    assert 'base: "/"' in files["web/vite.config.ts"]
    serve = tomllib.loads(files["pyproject.toml"])["tool"]["cayu"]["serve"]
    assert serve["ui"] == "web/static"
    assert "integrations/web.py" not in files
    # A build reads only the app's source, never the previous build in web/static, and
    # `check:build` fails when web/static isn't the build of web/src.
    assert '@import "tailwindcss" source("../..");' in files["web/src/kit/shadcn/styles.css"]
    scripts = json.loads(files["web/package.json"])["scripts"]
    assert scripts["check:build"] == "node scripts/check-build.mjs"
    assert "web/scripts/check-build.mjs" in files
    # Installed packages and build output, of web/ and of an ejected dashboard, aren't committed.
    ignored = files[".gitignore"].splitlines()
    assert {"node_modules/", "dist/"} <= set(ignored)


def test_the_web_app_is_the_generated_shell_with_b2bs_pages_as_areas() -> None:
    files = project_files("acme-desk", preset="b2b")
    source = {name: text for name, text in files.items() if name.startswith("web/src/")}

    # Every source file says whose it is, so the generator knows what it may rewrite.
    markers = ("GENERATED.", "THE AGENT'S OWN.", "BUILDING BLOCK.")
    owned = {
        name
        for name, text in source.items()
        if not name.startswith("web/src/kit/") and not any(item in text for item in markers)
    }
    assert owned == set()
    assert all("THE AGENT'S OWN." in text for name, text in source.items() if "/agent/" in name)
    # Every component comes through the one `@kit` alias, which points at the shadcn kit.
    assert '"@kit/*": ["./src/kit/shadcn/*"]' in files["web/tsconfig.json"]
    assert 'const kit = "shadcn";' in files["web/vite.config.ts"]
    assert not [name for name in files if name.startswith("web/src/components/")]
    agent = files["web/src/agent/index.tsx"]
    for area in ("overviewArea(", "conversationsArea", "membersArea", "settingsArea"):
        assert area in agent
    assert "shell: ShellParts = { SignIn, ProfileItems }" in agent
    # Until Cayu ships its Inbox as a part an app imports, the shell links to the Developer
    # dashboard's.
    assert (
        '{ label: "Inbox", icon: InboxIcon, href: "/cayu/inbox" }'
        in files["web/src/app/layout.tsx"]
    )
    # Only the stock shadcn kit ships; nothing names a private package or registry.
    everything = "\n".join(text for name, text in files.items() if name.startswith("web/"))
    for private in ("@cayu-tech", "cayu-design", "@cayu/design", "npm.pkg.github.com"):
        assert private not in everything
    assert "web/.npmrc" not in files


def test_the_generated_web_app_routes_are_composed_with_b2bs_people_and_metrics() -> None:
    files = project_files("acme-desk", preset="b2b")
    generated = {
        name: text
        for name, text in files.items()
        if name.startswith("integrations/http/generated/")
    }

    # The generator owns these, and says so; the Agent's own hooks are its own.
    assert generated and all(text.startswith('"""GENERATED.') for text in generated.values())
    for own in ("fixed_settings.py", "overview_metrics.py", "__init__.py"):
        assert files[f"integrations/http/{own}"].startswith('"""THE AGENT\'S OWN.')
    http = files["integrations/http/__init__.py"]
    assert "build_generated_router(" in http
    assert "overview_metrics=overview_metrics" in http
    assert "fixed_settings=fixed_settings" in http
    # Without an Inbox page of its own, the Overview sends people to the Developer dashboard's.
    assert "inbox_page=" not in http
    assert 'INBOX_PAGE = "/cayu/inbox"' in generated["integrations/http/generated/overview.py"]
    assert "def app_role(" in files["domain/organizations.py"]
    # The Settings page's `cayu cloud settings` commands name the Cloud application, which
    # `cayu cloud init` makes from the project's name, never the specialist's name.
    named = project_files("acme-desk", preset="b2b", agent_name="support")
    http = named["integrations/http/__init__.py"]
    assert 'CLOUD_APPLICATION = "acme-desk"' in http and 'AGENT_NAME = "support"' in http
    assert "application=cloud_application()" in http


def _page_assets(page: str) -> set[str]:
    return set(re.findall(r'(?:src|href)="/(assets/[^"]+)"', page))


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
                "tests/test_generated_routes.py",
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
    assert "people: http://127.0.0.1:8000/\n" in capsys.readouterr().out
    monkeypatch.chdir(tmp_path / "acme-desk")

    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []
