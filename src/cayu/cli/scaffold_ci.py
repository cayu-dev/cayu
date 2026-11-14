"""The CI workflow ``cayu new`` writes, selected by ``--ci``.

``--ci github`` (the default) writes a GitHub Actions workflow that runs the
project's tests and ``cayu check`` on a fresh checkout, plus the web app's
checks for a project that has one. It needs no secrets: the generated root
``conftest.py`` gives the tests a throwaway memory evidence key and ``cayu
check`` borrows one. ``--ci none`` writes no workflow.

The workflow is an ordinary project file the application owns. It composes the
same commands a person runs locally, so it can be changed, moved to another CI
service, or deleted without touching anything else.
"""

from __future__ import annotations

from collections.abc import Mapping

from cayu.cli.scaffold_plan import ApplicationPlan, CiName

CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
_GUIDANCE_MARKER = "\n## Continuous integration\n\n"

# Actions are pinned to commits, as in Cayu's own workflows; the comment names the release.
_CHECKOUT = "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5 # v4.3.1"
_SETUP_UV = "astral-sh/setup-uv@d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86 # v5.4.2"
_SETUP_NODE = "actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020 # v4.4.0"
_UV_VERSION = "0.9.28"

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
    image = (
        " Before the check it builds the coding image from the pins committed in "
        "`docker-coding-build.json`."
        if _builds_coding_image(files)
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
        f"`{plan.check_command()}` on GitHub Actions for every pull request and every "
        f"push to `main`. It needs no secrets.{image}{web}{evals} Keep it passing, and add a "
        "command there when the project gains another check.\n"
    )


def _builds_coding_image(files: Mapping[str, str]) -> bool:
    return "build_coding_image.py" in files


def _has_web_app(files: Mapping[str, str]) -> bool:
    return "web/package.json" in files and "web/package-lock.json" in files


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
    steps.append("      - name: Tests\n        run: uv run --no-sync pytest -q\n")
    if _builds_coding_image(files):
        steps.append(_CODING_IMAGE_STEP)
    check = f"      - name: cayu check\n        run: {plan.check_command()}\n"
    environment = plan.check_environment()
    if environment:
        check = (
            "      # Throwaway local credentials that let the check verify the service's\n"
            "      # authentication. A deployment sets its own, as secrets.\n"
            + check
            + "        env:\n"
            + "".join(f"          {name}: {_yaml_string(value)}\n" for name, value in environment)
        )
    steps.append(check)
    if "evals" in plan.capabilities:
        steps.append(_evals_note(plan))
    workflow = _HEADER + "".join(steps)
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
