from __future__ import annotations

import re
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).parents[2]
_CI_WORKFLOW = _REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
_TAG_VERIFIER = _REPOSITORY_ROOT / ".github" / "actions" / "verify-release-tag" / "action.yml"
_RELEASE_RUNBOOK = _REPOSITORY_ROOT / "docs" / "releasing.md"
_SIDECAR_VERIFIER = _REPOSITORY_ROOT / "scripts" / "verify_release_sidecar_artifacts.sh"
_COMMIT_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def _job_block(workflow: str, job_name: str) -> str:
    lines = workflow.splitlines()
    marker = f"  {job_name}:"
    start = lines.index(marker)
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("  ")
            and not lines[index].startswith("    ")
            and lines[index].endswith(":")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _action_references(block: str) -> list[str]:
    return re.findall(r"^\s+(?:- )?uses: ([^\s#]+)", block, flags=re.MULTILINE)


def _job_ids(workflow: str) -> set[str]:
    jobs = workflow.split("jobs:\n", 1)[1]
    return set(re.findall(r"^  ([a-z0-9_-]+):$", jobs, flags=re.MULTILINE))


def test_release_jobs_pin_every_external_action_to_immutable_commit() -> None:
    workflow = _CI_WORKFLOW.read_text()
    references = [
        reference
        for job_name in (
            "verification-scope",
            "static",
            "test_shards",
            "test_specialists",
            "sqlite-cancellation",
            "package",
            "dashboard",
            "publish",
            "github-release",
        )
        for reference in _action_references(_job_block(workflow, job_name))
        if not reference.startswith("./")
    ]

    assert references
    assert all(_COMMIT_PIN.fullmatch(reference) for reference in references), references


def test_pull_requests_and_main_run_the_core_workers_with_selected_high_value_gates() -> None:
    workflow = _CI_WORKFLOW.read_text()

    assert _job_ids(workflow) == {
        "verification-scope",
        "static",
        "test_shards",
        "test_specialists",
        "sqlite-cancellation",
        "package",
        "dashboard",
        "publish",
        "github-release",
    }
    for core_job in ("static", "test_shards", "test_specialists"):
        assert "startsWith(github.ref, 'refs/tags/v')" not in _job_block(workflow, core_job)
    scope = _job_block(workflow, "verification-scope")
    assert "python3 scripts/select_ci_jobs.py" in scope
    assert 'if test "$EVENT_NAME" != "pull_request"' in scope
    for output in ("dashboard", "release_artifacts", "sqlite_cancellation"):
        assert f'echo "{output}=true"' in scope
        assert f"{output}: ${{{{ steps.scope.outputs.{output} }}}}" in scope


def test_core_ci_uses_balanced_required_shards_without_coverage() -> None:
    workflow = _CI_WORKFLOW.read_text()
    shards = _job_block(workflow, "test_shards")
    specialists = _job_block(workflow, "test_specialists")

    assert "github.event_name == 'pull_request'" not in shards
    assert "timeout-minutes: 30" in shards
    assert "shard: [1, 2, 3, 4, 5, 6]" in shards
    assert "--splits 6" in shards
    assert '--group "${{ matrix.shard }}"' in shards
    assert "--splitting-algorithm least_duration" in shards
    assert "pytest -q" in shards
    assert "-n 2" not in shards
    assert '-m "not (stress or process or postgres)"' in shards
    assert "--cov" not in shards
    assert "COVERAGE_FILE" not in shards
    assert "uses: actions/upload-artifact@" not in shards

    assert "github.event_name == 'pull_request'" not in specialists
    assert "timeout-minutes: 30" in specialists
    assert "lane: stress-process" in specialists
    assert 'marker: "stress or process"' in specialists
    assert specialists.count('marker: "postgres and not (stress or process)"') == 2
    assert "lane: postgres-conformance-1" in specialists
    assert "lane: postgres-conformance-2" in specialists
    assert '-m "${{ matrix.marker }}"' in specialists
    assert '--splits "${{ matrix.splits }}"' in specialists
    assert '--group "${{ matrix.group }}"' in specialists
    assert "--cov" not in specialists
    assert "COVERAGE_FILE" not in specialists
    assert "uses: actions/upload-artifact@" not in specialists


def test_privileged_jobs_share_release_tag_verifier() -> None:
    workflow = _CI_WORKFLOW.read_text()
    publish = _job_block(workflow, "publish")
    github_release = _job_block(workflow, "github-release")

    for job, operation in (
        (publish, "uses: pypa/gh-action-pypi-publish@"),
        (github_release, "name: Create release with the published artifacts"),
    ):
        checkout = job.index("uses: actions/checkout@")
        download = job.index("uses: actions/download-artifact@")
        verifier = job.index("uses: ./.github/actions/verify-release-tag")
        privileged_operation = job.index(operation)
        assert checkout < download < verifier < privileged_operation
        assert "persist-credentials: false" in job[checkout:download]
        assert "uses: ./.github/actions/verify-release-tag" in job
        assert "gh api" not in job

    verifier = _TAG_VERIFIER.read_text()
    assert 'gh api "repos/$GITHUB_REPOSITORY/commits/$GITHUB_REF_NAME"' in verifier
    assert 'if test "$resolved" != "$GITHUB_SHA"' in verifier


def test_release_runbook_records_external_security_prerequisites() -> None:
    contributing = (_REPOSITORY_ROOT / "CONTRIBUTING.md").read_text()
    runbook = _RELEASE_RUNBOOK.read_text()
    runbook_words = " ".join(runbook.split())

    assert "docs/releasing.md" in contributing
    assert "`ci.yml`" in runbook
    assert "required reviewer" in runbook
    assert "`v*` tag ruleset" in runbook
    assert "updates, deletion, and non-fast-forward changes" in runbook_words
    assert "Do not push any `v*` tag" in runbook
    assert "PYPI_PUBLISH_ENABLED" in runbook
    assert "exact, non-empty `## vX.Y.Z` section" in runbook
    assert "does not generate release notes" in runbook
    assert "`## Unreleased`" in runbook
    assert "must not edit that tagged section" in runbook_words
    assert "development version" in runbook_words
    assert "scripts/verify_release_state.py" in runbook
    assert "0.1.0a1" not in runbook
    assert 'version="$(python -c' in runbook


def test_release_workflow_gates_publish_and_reuses_validated_artifact() -> None:
    workflow = _CI_WORKFLOW.read_text()
    package = _job_block(workflow, "package")
    publish = _job_block(workflow, "publish")
    github_release = _job_block(workflow, "github-release")

    assert "timeout-minutes: 40" in package

    assert 'tags: ["v*"]' in workflow
    assert "!cancelled()" in publish
    assert "!failure()" in publish
    assert "startsWith(github.ref, 'refs/tags/v')" in publish
    assert "vars.PYPI_PUBLISH_ENABLED == 'true'" in publish
    assert (
        "needs: [static, test_shards, test_specialists, sqlite-cancellation, package, "
        "dashboard]" in publish
    )
    assert "if: startsWith(github.ref, 'refs/tags/v')" in github_release
    assert "needs: [publish, package]" in github_release

    assert "prerelease: ${{ steps.release-version.outputs.prerelease }}" in package
    assert "id: release-version" in package
    assert "Version(version).is_prerelease" in package
    release_step = package.index("name: Check tag matches project version")
    release_version = package.index("id: release-version")
    release_run = package.index("run: |", release_version)
    assert "if: startsWith(github.ref, 'refs/tags/v')" in package[release_step:release_run]
    upload = package.index("name: Upload release distribution")
    assert package.index("name: Check installed CLI version") < upload
    assert "if: startsWith(github.ref, 'refs/tags/v')" in package[upload : upload + 160]
    assert "name: release-dist" in package[upload:]
    assert "path: dist/first/" in package[upload:]

    assert "name: release-dist" in publish
    assert "path: dist/" in publish
    assert "uv build" not in publish
    assert "pypa/gh-action-pypi-publish@" in publish

    assert "needs.package.outputs.prerelease" in github_release
    assert "--verify-tag" in github_release
    assert "--prerelease" in github_release
    assert "--latest=false" in github_release


def test_github_release_uses_curated_notes_for_the_exact_tag() -> None:
    github_release = _job_block(_CI_WORKFLOW.read_text(), "github-release")

    assert "python3 scripts/extract_release_notes.py" in github_release
    assert "--notes docs/release-notes.md" in github_release
    assert '--version "$GITHUB_REF_NAME"' in github_release
    assert '--output "$RUNNER_TEMP/release-notes.md"' in github_release
    assert '--notes-file "$RUNNER_TEMP/release-notes.md"' in github_release
    assert "--generate-notes" not in github_release


def test_release_artifact_job_enforces_tagged_note_immutability() -> None:
    package = _job_block(_CI_WORKFLOW.read_text(), "package")

    checkout = package.index("uses: actions/checkout@")
    setup = package.index("uses: astral-sh/setup-uv@")
    assert "fetch-depth: 0" in package[checkout:setup]
    assert "uv run --no-project" in package
    assert "--offline" in package
    assert "--no-python-downloads" in package
    assert "--python 3.11" in package
    assert "python scripts/verify_release_state.py" in package
    assert "--notes docs/release-notes.md" in package


def test_selected_high_value_jobs_preserve_premerge_and_main_contracts() -> None:
    workflow = _CI_WORKFLOW.read_text()
    sqlite = _job_block(workflow, "sqlite-cancellation")
    package = _job_block(workflow, "package")
    dashboard = _job_block(workflow, "dashboard")

    assert "needs: verification-scope" in sqlite
    assert "needs.verification-scope.outputs.sqlite_cancellation == 'true'" in sqlite
    assert "needs: verification-scope" in package
    assert "needs.verification-scope.outputs.release_artifacts == 'true'" in package
    assert "needs: verification-scope" in dashboard
    assert "needs.verification-scope.outputs.dashboard == 'true'" in dashboard

    assert "test_final_workspace_observer_restores_caller_cancellation_requests" in sqlite
    assert "test_delegated_stream_close_counts_checkpoint_cancellation_once" in sqlite
    assert "test_delegated_stream_close_distinguishes_restored_and_late_cancellation" in sqlite
    sidecar_verifier_command = "bash scripts/verify_release_sidecar_artifacts.sh"
    assert package.count(sidecar_verifier_command) == 1

    sidecar_verifier = _SIDECAR_VERIFIER.read_text()
    assert sidecar_verifier.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert sidecar_verifier.count("lambda-microvm sidecar export") == 3
    assert "docker build --platform linux/arm64" in sidecar_verifier
    assert "docker run --rm --platform linux/arm64" in sidecar_verifier

    for preserved_check in (
        "Verify the installed-wheel dashboard-to-local eval journey",
        "Check built-wheel secure public-service contract",
    ):
        check_offset = package.index(preserved_check)
        assert (
            "if: github.event_name != 'pull_request'"
            not in package[check_offset : check_offset + 180]
        )

    sidecar_offset = package.index("Verify installed wheel and source-distribution sidecar exports")
    assert (
        "if: github.event_name != 'pull_request'"
        not in package[sidecar_offset : sidecar_offset + 180]
    )
