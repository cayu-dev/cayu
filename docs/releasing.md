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

## Publish a release

1. Land the version bump on `main` and confirm the version in `pyproject.toml`.
2. Create and push the matching tag:

   ```bash
   git switch main
   git pull --ff-only
   version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
   git tag "v${version}"
   git push origin "v${version}"
   ```

3. Wait for the `static`, `test`, `package`, and `dashboard` jobs to pass on the
   tagged commit. The `package` job checks that the tag matches the project
   version and uploads the exact distribution it validated.
4. Confirm the environment request names the expected tag and commit, then
   approve the `pypi` deployment.
5. Wait for PyPI publication and the dependent GitHub release to complete.
6. Verify the exact published version:

   ```bash
   version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
   python -m pip install --pre "cayu==${version}"
   cayu version
   ```

If PyPI succeeds but GitHub release creation fails, rerun only the failed job;
do not rerun or recreate the already-published PyPI version.
