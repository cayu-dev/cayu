# Releasing Cayu

Prepare releases in `cayu-tech/cayu`; publish from `cayu-dev/cayu`.
Only public `v*` tag pushes publish. Leave PR merges and release tags to a maintainer.

1. Choose an unused version. Update `pyproject.toml`, `src/cayu/_version.py`,
   `uv.lock`, and `dashboard/src/lib/release-metadata.ts`. Add one short,
   matching, non-empty `## vX.Y.Z` section to `docs/release-notes.md`.
2. Regenerate versioned assets:

   ```bash
   uv sync --extra dev --extra server --extra browser
   uv run python scripts/generate_sidecar_manifest.py
   cd dashboard
   npm ci
   CAYU_PYTHON=../.venv/bin/python npm run generate:api
   npm run build:package
   cd ..
   uv run python scripts/build_dashboard_source_bundle.py
   ```

3. Pass CI, `qualification.yml`, and
   `uv run python -m maintenance.model_catalog.check` on the release commit.
   Check note integrity with
   `uv run python scripts/verify_release_state.py --notes docs/release-notes.md`.
   Branch pushes do not start CI; use the PR or `workflow_dispatch`.
4. Sync the approved commit to the public repo and push its matching `vX.Y.Z`
   tag. Verify the tag and commit before approving the `pypi` deployment.
5. Confirm PyPI and GitHub publication, then install `cayu==X.Y.Z` in a clean
   environment and check `cayu version`.

Publication requires the existing PyPI trusted publisher
(`cayu-dev/cayu`, `ci.yml`, environment `pypi`), a required reviewer with
self-approval disabled, and a `v*` tag ruleset blocking updates, deletion, and
non-fast-forward changes. Enable `PYPI_PUBLISH_ENABLED=true` only after these
controls are set. The workflow publishes the artifacts it tested and uses the
matching release-note section verbatim.

Never reuse a published version, move its tag, or edit its tagged release notes.
If PyPI succeeds but GitHub release creation fails, rerun only the failed job.
After release, give private `main` a distinct development version and one
`## Unreleased` section.
