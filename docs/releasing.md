# Releasing Cayu

This runbook is for Cayu maintainers publishing from the public
`cayu-dev/cayu` repository. Releases use the tag-gated jobs in
`.github/workflows/ci.yml`; branch pushes and pull requests never publish.

## One-time trusted-publisher setup

Complete these steps before pushing the first release tag. Required-reviewer
environments and tag protection can be plan-gated for private repositories, so
make `cayu-dev/cayu` public before configuring them.

1. Merge the release workflow and version bump, then mirror `main` to
   `cayu-dev/cayu`.
2. Make `cayu-dev/cayu` public.
3. In PyPI, add the pending trusted publisher with these exact values:

   | Field | Value |
   | --- | --- |
   | Project | `cayu` |
   | Owner | `cayu-dev` |
   | Repository | `cayu` |
   | Workflow | `ci.yml` |
   | Environment | `pypi` |

   The workflow filename is part of the OIDC identity. Using the obsolete
   `release.yml` name makes every publication fail authentication.
4. In GitHub, configure the `pypi` environment with a required reviewer. Enable
   prevention of self-review and disable administrator bypass where the plan
   permits it. Under **Deployment branches and tags**, select the custom policy
   and allow only tags matching `v*`.
5. Add an active `v*` tag ruleset with **Restrict updates**, **Restrict
   deletions**, and **Block force pushes** enabled and no bypass actors. Together
   these rules block updates, deletion, and non-fast-forward changes. PyPI files
   are immutable, so a published tag must be immutable too.
6. In **Settings → Secrets and variables → Actions → Variables**, create the
   repository variable `PYPI_PUBLISH_ENABLED` with the value `true`. Leave this
   variable absent until steps 3–5 have been verified; an absent or different
   value keeps the publish job disabled even if a release tag is pushed.

Do not push any `v*` tag until the trusted publisher, protected `pypi`
environment, tag ruleset, and `PYPI_PUBLISH_ENABLED` switch are all active.
Referencing an absent environment from a workflow causes GitHub to create it
without protection rules.

## Between releases

Once a `v*` tag has been published, its matching `## vX.Y.Z` release-note
section is part of the immutable release record. Main must not edit that tagged
section; publish corrections or follow-up guidance under `## Unreleased` and in
the next release instead.

Post-release code and migration guidance belongs under exactly one
`## Unreleased` heading. Main must also use an explicit development version
that differs from the latest published artifact. Keep `pyproject.toml`, the
source-tree fallback in `src/cayu/_version.py`, and `uv.lock` synchronized.
Using a prospective `.dev0` identity does not commit the project to that final
release number; release preparation may select and coordinate a different
version.

The release-artifact CI lane runs `scripts/verify_release_state.py` against the
tags already fetched into the checkout. It fails when a tagged section changes
or post-release source reuses a published package version, permits
`## Unreleased` and new untagged sections to evolve, and requires no GitHub or
PyPI access.

## Publish a release

1. Land the coordinated version bump on `main`, confirm the version in
   `pyproject.toml`, and curate the applicable `## Unreleased` material into one
   exact, non-empty `## vX.Y.Z` section in `docs/release-notes.md` for the
   matching tag. The workflow does not generate release notes; it publishes
   that curated section verbatim. A missing, duplicate, or empty section fails
   before GitHub release creation. Regenerate dashboard API metadata, compiled
   assets, and the version-matched editable source bundle after the version
   change:

   ```bash
   cd dashboard
   npm ci
   CAYU_PYTHON=../.venv/bin/python npm run generate:api
   npm run build:package
   cd ..
   uv run python scripts/build_dashboard_source_bundle.py
   ```
2. Run the release-only model-catalog freshness gate before creating the tag:

   ```bash
   uv run python -m maintenance.model_catalog.check
   ```

   Pull-request and `main` CI deliberately skip wall-clock staleness so unchanged
   branches remain deterministic. The tag workflow does not skip it. Do not create
   the tag if this command fails; re-verify the reported records against their
   official sources and land the refreshed catalog first.
3. Create and push the matching tag:

   ```bash
   git switch main
   git pull --ff-only
   version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
   git tag "v${version}"
   git push origin "v${version}"
   ```

4. Wait for the `static`, `test`, `sqlite-cancellation`, `package`,
   `windows-dashboard-artifact`, and `dashboard` jobs to pass on the tagged commit. The
   `package` job checks that the tag matches the project version and uploads the exact
   distribution it validated. Its artifact gate verifies identical dashboard-source bundles in
   the wheel and sdist, ejects from both installed artifacts, installs the locked Node
   dependencies, runs lint/tests/typechecking/API drift checks, reproduces the packaged dashboard
   tree, and browser-smokes a non-root deep link plus a bounded control-plane read against the
   installed server. The Windows artifact job separately verifies native extraction, inherited
   permissions, and the installed API-check workflow.
5. Confirm the environment request names the expected tag and commit, then
   approve the `pypi` deployment.
6. Wait for PyPI publication and the dependent GitHub release to complete.
7. Verify the exact published version:

   ```bash
   version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
   python -m pip install --pre "cayu==${version}"
   cayu version
   ```

If PyPI succeeds but GitHub release creation fails, rerun only the failed job;
do not rerun or recreate the already-published PyPI version.
