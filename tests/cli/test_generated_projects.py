"""The generated projects' matrix: `scripts/check_generated_project.py` and its CI job.

Each case generates a project from the checkout and runs the commands the project's own
generated CI workflow runs. These tests keep the cases, the job's matrix and those commands
in step without generating anything on disk; the job itself runs the projects.
"""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

from cayu.cli.scaffold import project_files
from cayu.cli.scaffold_ci import CI_WORKFLOW_PATH
from cayu.cli.scaffold_plan import PRESETS, WEB_APP

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = runpy.run_path(str(ROOT / "scripts" / "check_generated_project.py"))
CASES = _SCRIPT["cases"]()
project_steps = _SCRIPT["project_steps"]


def _job(workflow: str, name: str) -> str:
    start = workflow.index(f"\n  {name}:\n")
    following = re.search(r"\n  [\w-]+:\n", workflow[start + 1 :])
    return workflow[start : start + 1 + following.start()] if following else workflow[start:]


def _run_commands(workflow: str) -> list[str]:
    """Each `run:` command, a block's one per line, leaving out commented steps."""

    commands: list[str] = []
    lines = workflow.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^( +)(?:- )?run: (.*)$", line)
        if match is None:
            continue
        if match.group(2) != "|":
            commands.append(match.group(2))
            continue
        for block in lines[index + 1 :]:
            if len(block) - len(block.lstrip()) <= len(match.group(1)):
                break
            commands.append(block.strip())
    return commands


def test_every_preset_is_a_case_with_and_without_its_web_app() -> None:
    expected = set()
    for spec in PRESETS:
        expected.add(spec.name)
        if WEB_APP in spec.default_capabilities and spec.name != "b2b":
            expected.add(f"{spec.name}-without-web-app")
    assert set(CASES) == expected | {"agent-organizations"}
    organizations = CASES["agent-organizations"].plan()
    assert {"organizations", "app-database", WEB_APP, "knowledge", "memory"} <= set(
        organizations.capabilities
    )


def test_the_ci_job_runs_every_case_on_the_generated_workflows_node() -> None:
    job = _job((ROOT / ".github/workflows/ci.yml").read_text(), "generated-projects")
    matrix = re.search(r"\n        case:\n((?:          - .+\n)+)", job)
    assert matrix is not None
    assert re.findall(r"^          - (.+)$", matrix.group(1), re.MULTILINE) == list(CASES)
    assert "scripts/run_ci.py --lane generated-project" in job
    assert "needs.verification-scope.outputs.generated_projects == 'true'" in job

    generated = project_files("acme", preset="agent")[CI_WORKFLOW_PATH]
    assert set(re.findall(r'node-version: "(\d+)"', generated)) == set(
        re.findall(r'node-version: "(\d+)"', job)
    )


@pytest.mark.parametrize("name", list(CASES))
def test_a_case_runs_what_its_projects_ci_runs(name: str) -> None:
    case = CASES[name]
    plan = case.plan()
    files = project_files("acme", application_plan=plan)
    steps = project_steps(plan, files.keys(), database="postgresql://matrix")

    assert [step.command for step in steps] == _run_commands(files[CI_WORKFLOW_PATH])
    has_web_app = WEB_APP in plan.capabilities
    assert ("npm run check:build" in [step.command for step in steps]) is has_web_app
    assert ("web/package.json" in files) is has_web_app
