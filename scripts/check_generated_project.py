"""Generate a project from this checkout and run what its own CI runs.

Each case is a `cayu new` choice: every preset as it comes, every preset that has the web app
without it (`--without web-app`; b2b refuses that), and the agent preset with organizations.
For a case, this makes the project, then runs the commands its generated
`.github/workflows/ci.yml` runs, in that order: `uv sync`, its tests, `cayu check`, the web
app's smoke test in a browser, and the web app's install, lint, typecheck and
`npm run check:build`, which builds `web/src` and fails unless `web/static` is that build. A
case with application tables gets a throwaway PostgreSQL server in Docker, as its CI does.

What the files are is the hash manifests' to pin (`tests/fixtures/preset_trees/`,
`tests/fixtures/b2b_tree.json`); this checks that they install, build and pass.

    uv run --no-sync python scripts/check_generated_project.py --list
    uv run --no-sync python scripts/check_generated_project.py agent
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from cayu.cli import dev_postgres
from cayu.cli.scaffold_app_database import APP_DATABASE
from cayu.cli.scaffold_plan import (
    ORGANIZATIONS,
    PRESETS,
    WEB_APP,
    ApplicationPlan,
    ScaffoldPlanError,
    normalize_application_plan,
)

PROJECT = "acme"


@dataclass(frozen=True)
class Case:
    name: str
    preset: str
    with_capabilities: tuple[str, ...] = ()
    without_capabilities: tuple[str, ...] = ()

    def arguments(self) -> list[str]:
        arguments = ["--preset", self.preset]
        if self.with_capabilities:
            arguments += ["--with", ",".join(self.with_capabilities)]
        if self.without_capabilities:
            arguments += ["--without", ",".join(self.without_capabilities)]
        return arguments

    def plan(self) -> ApplicationPlan:
        return normalize_application_plan(
            name=PROJECT,
            agent_name=PROJECT,
            preset=self.preset,
            with_capabilities=self.with_capabilities,
            without_capabilities=self.without_capabilities,
        )


def cases() -> dict[str, Case]:
    found: list[Case] = []
    for spec in PRESETS:
        found.append(Case(spec.name, spec.name))
        if WEB_APP in spec.default_capabilities:
            without = Case(
                f"{spec.name}-without-web-app", spec.name, without_capabilities=(WEB_APP,)
            )
            try:
                without.plan()
            except ScaffoldPlanError:
                continue  # b2b: the app is what it is
            found.append(without)
    # Organizations on the agent preset, with its knowledge and memory scoped to each one.
    found.append(Case("agent-organizations", "agent", with_capabilities=(ORGANIZATIONS,)))
    return {case.name: case for case in found}


@dataclass(frozen=True)
class Step:
    label: str
    command: str
    cwd: str = "."
    env: Mapping[str, str] = field(default_factory=dict)


def project_steps(plan: ApplicationPlan, paths: Collection[str], *, database: str) -> list[Step]:
    """The commands the project's generated CI workflow runs, in its order."""

    postgres = APP_DATABASE in plan.capabilities
    steps = [
        Step("Install", "if [ -f uv.lock ]; then uv sync --locked; else uv sync; fi"),
        Step(
            "Tests",
            "uv run --no-sync pytest -q",
            env={"CAYU_TEST_POSTGRES_DSN": database, "CAYU_REQUIRE_POSTGRES_TESTS": "1"}
            if postgres
            else {},
        ),
        Step("cayu check", plan.ci_check_command(), env=dict(plan.ci_check_environment())),
    ]
    if "web/e2e/package.json" in paths:
        smoke_env = {}
        if ORGANIZATIONS in plan.capabilities:
            smoke_env = {"CAYU_DATABASE_URL": database}
            steps += [
                Step("Demonstration data", command, env=smoke_env)
                for command in (
                    "uv run --no-sync cayu storage migrate",
                    "uv run --no-sync alembic upgrade head",
                    "uv run --no-sync python -m operations.demo_seed",
                )
            ]
        steps += [
            Step("Smoke test install", "npm ci", "web/e2e", smoke_env),
            Step(
                "Smoke test browser",
                "npx playwright install --with-deps chromium",
                "web/e2e",
                smoke_env,
            ),
            Step("Smoke test", "npm test", "web/e2e", smoke_env),
        ]
    if "web/package.json" in paths:
        steps += [
            Step(f"Web app: {command}", command, "web")
            for command in ("npm ci", "npm run lint", "npm run typecheck", "npm run check:build")
        ]
    return steps


@contextmanager
def _postgres(project: Path) -> Iterator[str]:
    """A throwaway PostgreSQL server for the project: `cayu dev postgres up`, then `down`."""

    given = os.environ.get("CAYU_TEST_POSTGRES_DSN", "").strip()
    if given:
        yield given
        return
    server, _ = dev_postgres.start(project, timeout=120)
    try:
        yield server.url
    finally:
        dev_postgres.stop(project)


def _environment(project: Path) -> dict[str, str]:
    # The project's commands see neither this checkout's environment nor its CI settings.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT"}
        and not key.startswith("CAYU_")
    }
    workflow = project / ".github/workflows/ci.yml"
    python = re.search(r'^  UV_PYTHON: "([^"]+)"$', workflow.read_text(), re.MULTILINE)
    if python:
        environment["UV_PYTHON"] = python.group(1)
    return environment


def check(case: Case, parent: Path) -> list[tuple[str, float]]:
    plan = case.plan()
    command = [sys.executable, "-m", "cayu", "new", PROJECT, "--dir", str(parent)]
    print(f"== Generate: cayu new {PROJECT} {shlex.join(case.arguments())}", flush=True)
    subprocess.run([*command, *case.arguments()], check=True)
    project = parent / PROJECT
    paths = {str(path.relative_to(project)) for path in project.rglob("*") if path.is_file()}
    record = tomllib.loads((project / "pyproject.toml").read_text())["tool"]["cayu"]["scaffold"]
    if tuple(record["capabilities"]) != plan.capabilities:
        raise RuntimeError(f"the project records {record['capabilities']}, not {plan.capabilities}")

    timings: list[tuple[str, float]] = []
    database = _postgres(project) if APP_DATABASE in plan.capabilities else _no_database()
    with database as dsn:
        environment = _environment(project)
        for step in project_steps(plan, paths, database=dsn):
            command = step.command
            if platform.system() != "Linux":
                # Playwright installs system packages only on Linux.
                command = command.replace(" --with-deps", "")
            print(f"== {step.label}: {command}  (in {step.cwd})", flush=True)
            started = time.monotonic()
            subprocess.run(
                ["bash", "-eo", "pipefail", "-c", command],
                cwd=project / step.cwd,
                env={**environment, **step.env},
                check=True,
            )
            timings.append((step.label, time.monotonic() - started))
    return timings


@contextmanager
def _no_database() -> Iterator[str]:
    yield ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "cases", nargs="*", metavar="CASE", help="the cases to check (default: every case)"
    )
    parser.add_argument("--list", action="store_true", help="print every case's name")
    parser.add_argument(
        "--dir", type=Path, help="make the projects here (default: a temporary folder)"
    )
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(cases()))
        return 0
    unknown = sorted(set(args.cases) - cases().keys())
    if unknown:
        parser.error(f"unknown cases {unknown}; --list lists them")
    failed = []
    for name in args.cases or cases():
        case = cases()[name]
        started = time.monotonic()
        try:
            if args.dir is not None:
                (args.dir / name).mkdir(parents=True)
                timings = check(case, (args.dir / name).resolve())
            else:
                with tempfile.TemporaryDirectory(prefix=f"cayu-generated-{name}-") as parent:
                    timings = check(case, Path(parent).resolve())
        except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
            print(f"error: {name}: {error}", file=sys.stderr)
            failed.append(name)
            continue
        print(f"== {name} passed in {time.monotonic() - started:.0f}s")
        for label, seconds in timings:
            print(f"   {seconds:6.1f}s  {label}")
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
