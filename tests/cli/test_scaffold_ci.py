from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from cayu.cli import main, scaffold
from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_ci import CI_WORKFLOW_PATH, generated_ci_choice
from cayu.cli.scaffold_plan import CI_CHOICES, PRESETS, normalize_application_plan

_PINNED_ACTION = re.compile(r"^- uses: [\w.-]+/[\w.-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+$")


def _commands(workflow: str) -> list[str]:
    """The workflow's `run:` commands, in order, leaving out commented lines."""

    return [
        line.strip().removeprefix("- ").removeprefix("run: ")
        for line in workflow.splitlines()
        if re.match(r"^ +(- )?run: ", line)
    ]


@pytest.mark.parametrize(
    ("preset", "execution"),
    [(spec.name, "none") for spec in PRESETS]
    + [("coding", "docker"), ("agent", "docker"), ("agent+app-database", "none")],
)
def test_every_preset_gets_a_pinned_workflow_that_needs_no_secrets(
    preset: str, execution: str
) -> None:
    preset, _, extra = preset.partition("+")
    plan = normalize_application_plan(
        name="app",
        agent_name="app",
        preset=preset,
        execution=execution,
        with_capabilities=(extra,) if extra else (),
    )
    files = project_files("app", application_plan=plan)
    workflow = files[CI_WORKFLOW_PATH]

    assert "\t" not in workflow
    code = [line for line in workflow.splitlines() if not line.lstrip().startswith("#")]
    assert all((len(line) - len(line.lstrip())) % 2 == 0 for line in code)
    uses = [line.strip() for line in code if line.strip().startswith("- uses:")]
    assert uses and all(_PINNED_ACTION.match(line) for line in uses), uses
    # Only a commented step names a secret; the default run needs none.
    assert not any("secrets." in line for line in code)
    commands = _commands(workflow)
    install = "if [ -f uv.lock ]; then uv sync --locked; else uv sync; fi"
    # A Docker coding app needs its image before `cayu check` can build the app.
    docker_coding = (preset, execution) == ("coding", "docker")
    image = ["uv run --no-sync python build_coding_image.py"] if docker_coding else []
    checks = [install, "uv run --no-sync pytest -q", *image, plan.ci_check_command()]
    assert commands[: len(checks)] == checks
    assert not any("cayu eval run" in command for command in commands)
    if "evals" in plan.capabilities:
        assert "      #   run: uv run --no-sync cayu eval run\n" in workflow
    # The service's CI runs the deploy check, which verifies authentication with
    # throwaway local values; every other preset runs the documented check.
    assert ("--deploy" in plan.ci_check_command()) == (preset == "service")
    assert bool(plan.ci_check_environment()) == (preset == "service")
    for name, value in plan.ci_check_environment():
        assert f"          {name}: '{value}'\n" in workflow
    has_web = "web/package.json" in files
    assert has_web == ("web-app" in plan.capabilities)
    # The web app's smoke test runs after the check, against `cayu serve --dev`.
    smoke = "web-app" in plan.capabilities
    assert smoke == ("web/e2e/package.json" in files)
    # The organizations' app reads its own tables: the smoke test gets a migrated database
    # with the demonstration people first.
    organizations = "organizations" in plan.capabilities
    if smoke:
        database = (
            "        env:\n          CAYU_DATABASE_URL: "
            "postgresql://postgres:postgres@localhost:5432/postgres\n"
            if organizations
            else ""
        )
        assert (
            f"        working-directory: web/e2e\n{database}        run: |\n          npm ci\n"
            "          npx playwright install --with-deps chromium\n          npm test\n"
        ) in workflow
    assert ("python -m operations.demo_seed" in workflow) == organizations
    web_commands = ["npm ci", "npm run lint", "npm run typecheck", "npm run check:build"]
    assert commands[len(checks) :] == (["|"] * (organizations + smoke)) + (
        web_commands if has_web else []
    )
    # The tests that need PostgreSQL get a server, and fail rather than skip without one.
    postgres = "app-database" in plan.capabilities
    assert ("    services:\n      # The tests" in workflow) == postgres
    assert ('CAYU_REQUIRE_POSTGRES_TESTS: "1"' in workflow) == postgres
    if postgres:
        assert re.search(r"image: postgres:[\w.-]+@sha256:[0-9a-f]{64}\n", workflow)
    assert ("memory" in plan.capabilities) == ("CAYU_MEMORY_EVIDENCE_KEY" in workflow)
    for guide in ("README.md", "AGENTS.md"):
        assert f"`{CI_WORKFLOW_PATH}` runs `uv sync`" in files[guide]


def test_ci_none_writes_no_workflow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["new", "planned", "--dir", str(tmp_path), "--dry-run", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert CI_WORKFLOW_PATH in plan["files"]
    assert ".github/workflows" in plan["directories"]
    core = next(group for group in plan["file_groups"] if group["name"] == "core")
    assert CI_WORKFLOW_PATH in core["files"]

    command = ["new", "quiet", "--ci", "none", "--dir", str(tmp_path)]
    assert main([*command, "--dry-run", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)["plan"]
    assert CI_WORKFLOW_PATH not in plan["files"]
    assert all(CI_WORKFLOW_PATH not in group["files"] for group in plan["file_groups"])

    assert main(command) == 0
    receipt = capsys.readouterr().out
    project = tmp_path / "quiet"
    assert not (project / ".github").exists()
    assert "  CI:" not in receipt
    for guide in ("README.md", "AGENTS.md"):
        text = (project / guide).read_text(encoding="utf-8")
        # Byte for byte what `cayu new` wrote before `--ci` existed, so preset
        # conversion still recognizes those projects' guidance as generated.
        assert "## Continuous integration" not in text
        assert "--ci" not in text


def test_preset_conversion_keeps_the_projects_ci_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def convert(project: Path) -> dict[str, str]:
        monkeypatch.chdir(project)
        assert main(["generate", "preset", "b2b", "--dry-run", "--json"]) == 0
        planned = json.loads(capsys.readouterr().out)
        assert planned["status"] == "ready", planned
        return {edit["path"]: edit["operation"] for edit in planned["edits"]}

    assert main(["new", "withci", "--dir", str(tmp_path)]) == 0
    assert main(["new", "without", "--ci", "none", "--dir", str(tmp_path)]) == 0
    assert main(["new", "own", "--ci", "none", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    # A project without the generated workflow may add its own; conversion leaves it.
    own = tmp_path / "own" / CI_WORKFLOW_PATH
    own.parent.mkdir(parents=True)
    own.write_text("name: Mine\n", encoding="utf-8")

    # The b2b workflow adds the web app's job.
    assert convert(tmp_path / "withci")[CI_WORKFLOW_PATH] == "update_region"
    assert CI_WORKFLOW_PATH not in convert(tmp_path / "without")
    assert CI_WORKFLOW_PATH not in convert(tmp_path / "own")
    assert own.read_text(encoding="utf-8") == "name: Mine\n"


def test_the_generated_guidance_records_the_ci_choice() -> None:
    for ci in CI_CHOICES:
        plan = normalize_application_plan(name="app", agent_name="app", ci=ci)
        agents = project_files("app", application_plan=plan)["AGENTS.md"]
        assert generated_ci_choice(agents) == ci
    assert generated_ci_choice(None) == "none"


def test_fresh_checkout_without_the_ignored_data_directory_passes_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The coding preset keeps its database under .cayu/, so nothing it runs
    # recreates data/, which git ignores and a CI checkout doesn't have.
    git = shutil.which("git")
    assert git is not None
    # Scaffolding runs git but not ripgrep; the check doesn't need either.
    monkeypatch.setattr(scaffold, "_preflight_coding_commands", lambda *, parent: (git, "rg"))
    assert main(["new", "checkout", "--preset", "coding", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "checkout"
    shutil.rmtree(project / "data")
    monkeypatch.chdir(project)

    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []
    assert not (project / "data").exists()
