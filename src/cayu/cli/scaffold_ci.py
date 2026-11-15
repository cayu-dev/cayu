"""The CI workflow ``cayu new`` writes, selected by ``--ci``.

``--ci github`` (the default) writes a GitHub Actions workflow that runs the
project's tests and ``cayu check`` on a fresh checkout, plus the web app's
checks and its smoke test for a project that has them. It needs no secrets: the generated root
``conftest.py`` gives the tests a throwaway memory evidence key and ``cayu
check`` borrows one. A project with application tables gets a throwaway PostgreSQL server for
the tests that need one, which fail rather than skip without it. ``--ci none`` writes no
workflow.

The workflow is an ordinary project file the application owns. It composes the
same commands a person runs locally, so it can be changed, moved to another CI
service, or deleted without touching anything else.
"""

from __future__ import annotations

from collections.abc import Mapping

from cayu.cli.scaffold_app_database import APP_DATABASE
from cayu.cli.scaffold_plan import ORGANIZATIONS, ApplicationPlan, CiName

CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
_GUIDANCE_MARKER = "\n## Continuous integration\n\n"

# Actions are pinned to commits, as in Cayu's own workflows; the comment names the release.
_CHECKOUT = "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4.3.1"
_SETUP_UV = "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86 # v5.4.2"
_SETUP_NODE = "actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020 # v4.4.0"
_UV_VERSION = "0.9.28"
# Pinned to a digest, as the actions are to commits.
_POSTGRES_IMAGE = (
    "postgres:17-alpine@sha256:d3244e8ce3f165ee3c93d9c4a4b1d0fe48b3a89e2287b717fdcbbde92d405065"
)
_POSTGRES_URL = "postgresql://postgres:postgres@localhost:5432/postgres"

_HEADER = """\
# Runs the tests and `cayu check` on every pull request and every push to main.
# It needs no secrets. `cayu new` wrote it; change it like any other project file.
name: CI

on:
  workflow_dispatch:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

env:
  # Any version that requires-python in pyproject.toml allows.
  UV_PYTHON: "3.13"

jobs:
  check:
    name: Tests and checks
    runs-on: ubuntu-latest
    timeout-minutes: 20
    steps:
"""

_SOURCE_CHECKOUT_NOTE = """\
      # [tool.uv.sources] in pyproject.toml installs Cayu from the local checkout this
      # project was made from, which CI can't reach. Remove that table to install the
      # Cayu release that [project].dependencies pins.
"""

_MEMORY_KEY_NOTE = """\
      # The root conftest.py gives the tests a throwaway memory evidence key, and
      # `cayu check` borrows one, so this job needs no CAYU_MEMORY_EVIDENCE_KEY.
"""

_CODING_IMAGE_STEP = """\
      # `cayu check` builds the app, which needs the coding image. This builds it from
      # the pins in docker-coding-build.json: run `uv run --no-sync python
      # build_coding_image.py --resolve-pins` once, review the pins and commit them.
      - name: Build the coding image
        run: uv run --no-sync python build_coding_image.py
"""

_EVALS_STEP = """\
      # - name: Evals
      #   run: uv run --no-sync cayu eval run
"""

_EVALS_WITH_MEMORY = """\
      # `cayu eval run` isn't part of this job yet. With automatic memory it needs
      # a stable CAYU_MEMORY_EVIDENCE_KEY, which a fresh checkout doesn't have. To
      # run the evals here, store one as a repository secret and uncomment these lines.
"""

_EVALS_WITHOUT_MEMORY = """\
      # `cayu eval run` isn't part of this job yet. To run the evals here too,
      # uncomment these lines.
"""

_WEB_JOB = """
  web:
    name: Web app
    runs-on: ubuntu-latest
    timeout-minutes: 15
    defaults:
      run:
        working-directory: web
    steps:
      - uses: __CHECKOUT__
      - uses: __SETUP_NODE__
        with:
          node-version: "24"
          cache: npm
          cache-dependency-path: web/package-lock.json
      - run: npm ci
      - run: npm run lint
      - run: npm run typecheck
      # Builds web/src into a temporary folder and fails unless the committed
      # web/static, which `cayu serve` serves, is exactly that build.
      - run: npm run check:build
"""


# A throwaway PostgreSQL server in the check job, for a project with application tables.
_POSTGRES_SERVICE = """\
    services:
      # The tests that read and write the application's tables create their own databases
      # here. Nothing on it outlives the job.
      postgres:
        image: __IMAGE__
        env:
          POSTGRES_PASSWORD: postgres
        ports: ["5432:5432"]
        options: >-
          --health-cmd pg_isready --health-interval 5s --health-timeout 5s --health-retries 12
"""

_POSTGRES_TESTS_ENV = """\
        env:
          CAYU_TEST_POSTGRES_DSN: __URL__
          # Fail, rather than skip, a test that needs PostgreSQL and finds no server.
          CAYU_REQUIRE_POSTGRES_TESTS: "1"
"""

# The organizations' app reads its own tables, so its smoke test starts from a migrated
# database with the demonstration organizations and people (operations/demo_seed.py).
_DEMO_DATA_STEP = """\
      - name: Demonstration data for the smoke test
        env:
          CAYU_DATABASE_URL: __URL__
        run: |
          uv run --no-sync cayu storage migrate
          uv run --no-sync alembic upgrade head
          uv run --no-sync python -m operations.demo_seed
"""

_SMOKE_STEP = """\
      # Opens every area of the web app in a browser, served by `cayu serve --dev`
      # (web/e2e/playwright.config.ts).
      - uses: __SETUP_NODE__
        with:
          node-version: "24"
          cache: npm
          cache-dependency-path: web/e2e/package-lock.json
      - name: Web app smoke test
        working-directory: web/e2e
        run: |
          npm ci
          npx playwright install --with-deps chromium
          npm test
"""


def ci_files(plan: ApplicationPlan, files: Mapping[str, str]) -> dict[str, str]:
    """The CI files for ``plan``, given every other file the project gets."""

    if plan.ci == "none":
        return {}
    return {CI_WORKFLOW_PATH: _github_workflow(plan, files)}


def generated_ci_choice(agents_md: str | None) -> CiName:
    """The `--ci` choice a project was generated with, read from its AGENTS.md.

    [tool.cayu.scaffold] doesn't record it, and the workflow file alone can't tell:
    a project may delete the generated one or add its own. The section `ci_guidance`
    writes can: a project made with `--ci none`, or before the option, has none.
    """

    return "github" if agents_md is not None and _GUIDANCE_MARKER in agents_md else "none"


def ci_guidance(plan: ApplicationPlan, files: Mapping[str, str]) -> str:
    """The README and AGENTS.md section that describes the workflow, if there is one."""

    if plan.ci == "none":
        return ""
    web = (
        " A second job checks the web app: `npm ci`, lint, types, and that `web/static` "
        "is the build of `web/src`."
        if _has_web_app(files)
        else ""
    )
    if _has_smoke_test(files):
        web += (
            " After the check, the first job opens every area of the web app in a browser "
            "(`web/e2e`)"
            + (
                ", signed in as a demonstration person."
                if ORGANIZATIONS in plan.capabilities
                else "."
            )
        )
    if APP_DATABASE in plan.capabilities:
        web = (
            " The tests that need PostgreSQL run against a throwaway server in the job, "
            "and fail rather than skip without it (`CAYU_REQUIRE_POSTGRES_TESTS`)." + web
        )
    image = (
        " Before the check it builds the coding image from the pins committed in "
        "`docker-coding-build.json`."
        if _builds_coding_image(files)
        else ""
    )
    deploy = (
        " The deploy check runs with throwaway "
        + " and ".join(f"`{name}`" for name, _value in plan.ci_check_environment())
        + " values, so it proves the service's authentication is wired; a deployment sets "
        "its own."
        if plan.ci_check_environment()
        else ""
    )
    evals = (
        " `cayu eval run` is a commented step: it needs a stable memory evidence key, "
        "which a fresh checkout doesn't have."
        if "evals" in plan.capabilities and "memory" in plan.capabilities
        else " `cayu eval run` is a commented step there; uncomment it to run the evals too."
        if "evals" in plan.capabilities
        else ""
    )
    return (
        _GUIDANCE_MARKER + f"`{CI_WORKFLOW_PATH}` runs `uv sync`, `uv run --no-sync pytest -q` and "
        f"`{plan.ci_check_command()}` on GitHub Actions for every pull request and every "
        f"push to `main`. It needs no secrets.{deploy}{image}{web}{evals} Keep it passing, and add a "
        "command there when the project gains another check.\n"
    )


def _builds_coding_image(files: Mapping[str, str]) -> bool:
    return "build_coding_image.py" in files


def _has_web_app(files: Mapping[str, str]) -> bool:
    return "web/package.json" in files and "web/package-lock.json" in files


def _has_smoke_test(files: Mapping[str, str]) -> bool:
    return "web/e2e/package.json" in files and "web/e2e/package-lock.json" in files


def _github_workflow(plan: ApplicationPlan, files: Mapping[str, str]) -> str:
    steps = [
        f"      - uses: {_CHECKOUT}\n",
        f'      - uses: {_SETUP_UV}\n        with:\n          version: "{_UV_VERSION}"\n',
    ]
    if "[tool.uv.sources]" in files.get("pyproject.toml", ""):
        steps.append(_SOURCE_CHECKOUT_NOTE)
    steps.append(
        "      # With a committed uv.lock, --locked fails when it is out of date.\n"
        "      - name: Install\n"
        "        run: if [ -f uv.lock ]; then uv sync --locked; else uv sync; fi\n"
    )
    if "memory" in plan.capabilities:
        steps.append(_MEMORY_KEY_NOTE)
    postgres = APP_DATABASE in plan.capabilities
    steps.append(
        "      - name: Tests\n        run: uv run --no-sync pytest -q\n"
        + (_POSTGRES_TESTS_ENV.replace("__URL__", _POSTGRES_URL) if postgres else "")
    )
    if _builds_coding_image(files):
        steps.append(_CODING_IMAGE_STEP)
    check = f"      - name: cayu check\n        run: {plan.ci_check_command()}\n"
    environment = plan.ci_check_environment()
    if environment:
        check = (
            "      # Throwaway local credentials that let the check verify the service's\n"
            "      # authentication. A deployment sets its own, as secrets.\n"
            + check
            + "        env:\n"
            + "".join(f"          {name}: {_yaml_string(value)}\n" for name, value in environment)
        )
    steps.append(check)
    if _has_smoke_test(files):
        smoke = _SMOKE_STEP.replace("__SETUP_NODE__", _SETUP_NODE)
        if ORGANIZATIONS in plan.capabilities:
            steps.append(_DEMO_DATA_STEP.replace("__URL__", _POSTGRES_URL))
            smoke = smoke.replace(
                "        working-directory: web/e2e\n",
                "        working-directory: web/e2e\n        env:\n"
                f"          CAYU_DATABASE_URL: {_POSTGRES_URL}\n",
            )
        steps.append(smoke)
    if "evals" in plan.capabilities:
        steps.append(_evals_note(plan))
    header = _HEADER
    if postgres:
        header = header.replace(
            "    timeout-minutes: 20\n",
            "    timeout-minutes: 20\n" + _POSTGRES_SERVICE.replace("__IMAGE__", _POSTGRES_IMAGE),
        )
    workflow = header + "".join(steps)
    if _has_web_app(files):
        workflow += _WEB_JOB.replace("__CHECKOUT__", _CHECKOUT).replace(
            "__SETUP_NODE__", _SETUP_NODE
        )
    return workflow


def _evals_note(plan: ApplicationPlan) -> str:
    if "memory" not in plan.capabilities:
        return _EVALS_WITHOUT_MEMORY + _EVALS_STEP
    return (
        _EVALS_WITH_MEMORY
        + _EVALS_STEP
        + "      #   env:\n"
        + "      #     CAYU_MEMORY_EVIDENCE_KEY: ${{ secrets.CAYU_MEMORY_EVIDENCE_KEY }}\n"
    )


def _yaml_string(value: str) -> str:
    # A single-quoted YAML scalar takes everything literally except a doubled quote.
    return "'" + value.replace("'", "''") + "'"
