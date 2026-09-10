from __future__ import annotations

import re
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).parents[2]
_CI_WORKFLOW = _REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
_CI_RUNNER = _REPOSITORY_ROOT / "scripts" / "run_ci.py"
_PACKAGE_MANIFEST = _REPOSITORY_ROOT / "scripts" / "package_ci_steps.yml"
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
            "test-collection",
            "test_shards",
            "test_specialists",
            "sqlite-cancellation",
            "package-build",
            "package",
            "release-qualification",
            "dashboard",
            "publish",
            "github-release",
        )
        for reference in _action_references(_job_block(workflow, job_name))
        if not reference.startswith("./")
    ]

    assert references
    assert all(_COMMIT_PIN.fullmatch(reference) for reference in references), references


def test_pull_requests_and_manual_runs_keep_selected_high_value_gates() -> None:
    workflow = _CI_WORKFLOW.read_text()

    assert _job_ids(workflow) == {
        "verification-scope",
        "static",
        "test-collection",
        "test_shards",
        "test_specialists",
        "sqlite-cancellation",
        "package-build",
        "package",
        "release-qualification",
        "dashboard",
        "publish",
        "github-release",
    }
    for core_job in ("static", "test-collection", "test_shards", "test_specialists"):
        assert "startsWith(github.ref, 'refs/tags/v')" not in _job_block(workflow, core_job)
    scope = _job_block(workflow, "verification-scope")
    assert "python3 scripts/select_ci_jobs.py" in scope
    assert 'if test "$EVENT_NAME" != "pull_request"' in scope
    for output in ("dashboard", "release_artifacts", "sqlite_cancellation"):
        assert f'echo "{output}=true"' in scope
        assert f"{output}: ${{{{ steps.scope.outputs.{output} }}}}" in scope


def test_core_ci_uses_balanced_required_shards_without_coverage() -> None:
    workflow = _CI_WORKFLOW.read_text()
    runner = _CI_RUNNER.read_text()
    shards = _job_block(workflow, "test_shards")
    specialists = _job_block(workflow, "test_specialists")

    for job in (shards, specialists):
        assert "command -v rg" in job
        assert "sudo apt-get install --yes ripgrep" in job
        assert job.index("sudo apt-get install --yes ripgrep") < job.index("scripts/run_ci.py")

    assert "github.event_name == 'pull_request'" not in shards
    assert "timeout-minutes: 15" in shards
    assert (
        "shard: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]"
        in shards
    )
    assert 'scripts/run_ci.py --lane general --shard "${{ matrix.shard }}"' in shards
    assert '"not (stress or qualification or postgres or browser_docker)"' in runner
    assert '"--splitting-algorithm",\n            "least_duration"' in runner
    assert "_GENERAL_SHARDS = 32" in runner
    assert "-n 2" not in runner
    assert "--cov" not in runner
    assert "COVERAGE_FILE" not in runner
    assert "name: ci-durations-general-${{ matrix.shard }}" in shards

    assert "github.event_name == 'pull_request'" not in specialists
    assert "timeout-minutes: 15" in specialists
    assert "stress-process" not in specialists
    assert "postgres-conformance-8" in specialists
    assert "scripts/run_ci.py --lane specialist" in specialists
    assert '--specialist-lane "${{ matrix.lane }}"' in specialists
    assert '"stress or qualification", 8, group' in runner
    assert '"postgres and not (stress or qualification)", 8, group' in runner
    assert "--cov" not in specialists
    assert "COVERAGE_FILE" not in specialists
    assert "name: ci-durations-${{ matrix.lane }}" in specialists


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
    assert "self-approval disabled" in runbook
    assert "`v*` tag ruleset" in runbook
    assert "updates, deletion, and non-fast-forward changes" in runbook_words
    assert "Leave PR merges and release tags to a maintainer" in runbook
    assert "PYPI_PUBLISH_ENABLED" in runbook
    assert "matching, non-empty `## vX.Y.Z` section" in runbook
    assert "matching release-note section verbatim" in runbook_words
    assert "`## Unreleased`" in runbook
    assert "Never reuse a published version, move its tag, or edit its tagged release notes" in (
        runbook_words
    )
    assert "development version" in runbook_words
    assert "scripts/verify_release_state.py" in runbook
    assert "0.1.0a1" not in runbook


def test_release_workflow_gates_publish_and_reuses_validated_artifact() -> None:
    workflow = _CI_WORKFLOW.read_text()
    package_manifest = _PACKAGE_MANIFEST.read_text()
    package = _job_block(workflow, "package-build")
    publish = _job_block(workflow, "publish")
    github_release = _job_block(workflow, "github-release")

    assert package_manifest.count("scripts/smoke_built_wheel_doctor.py") == 1

    assert "timeout-minutes: 10" in package

    assert 'tags: ["v*"]' in workflow
    assert "!cancelled()" in publish
    assert "!failure()" in publish
    assert "startsWith(github.ref, 'refs/tags/v')" in publish
    assert "vars.PYPI_PUBLISH_ENABLED == 'true'" in publish
    assert (
        "needs: [static, test-collection, test_shards, test_specialists, sqlite-cancellation, package-build, package, "
        "dashboard, release-qualification]" in publish
    )
    assert (
        "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')" in github_release
    )
    assert "needs: [publish, package-build]" in github_release

    assert "prerelease: ${{ steps.release-package.outputs.prerelease }}" in package
    assert "id: release-package" in package
    assert "python3 scripts/run_ci.py --lane package" in package
    assert "publishing=(--publishing)" in package
    assert "Version(version).is_prerelease" in package_manifest
    assert "publishing: true\n    run: |\n      uv run --group nightly" in package_manifest
    upload = package.index("name: Upload release distribution")
    assert package.index("python3 scripts/run_ci.py --lane package") < upload
    assert "if:" not in package[upload:]
    assert "name: release-dist" in package[upload:]
    assert "path: dist/first/" in package[upload:]

    assert "name: release-dist" in publish
    assert "path: dist/" in publish
    assert "uv build" not in publish
    assert "pypa/gh-action-pypi-publish@" in publish

    assert "needs.package-build.outputs.prerelease" in github_release
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
    package = _job_block(_CI_WORKFLOW.read_text(), "package-build")
    package_manifest = _PACKAGE_MANIFEST.read_text()

    checkout = package.index("uses: actions/checkout@")
    setup = package.index("uses: astral-sh/setup-uv@")
    assert "fetch-depth: 0" in package[checkout:setup]
    assert "python3 scripts/run_ci.py --lane package" in package
    assert "uv run --no-project" in package_manifest
    assert "--offline" in package_manifest
    assert "--no-python-downloads" in package_manifest
    assert "--python 3.11" in package_manifest
    assert "python scripts/verify_release_state.py" in package_manifest
    assert "--notes docs/release-notes.md" in package_manifest


def test_selected_high_value_jobs_preserve_premerge_and_manual_contracts() -> None:
    workflow = _CI_WORKFLOW.read_text()
    runner = _CI_RUNNER.read_text()
    package_manifest = _PACKAGE_MANIFEST.read_text()
    sqlite = _job_block(workflow, "sqlite-cancellation")
    package = _job_block(workflow, "package-build")
    dashboard = _job_block(workflow, "dashboard")

    assert "needs: verification-scope" in sqlite
    assert "needs.verification-scope.outputs.sqlite_cancellation == 'true'" in sqlite
    assert "needs: verification-scope" in package
    assert "needs.verification-scope.outputs.release_artifacts == 'true'" in package
    assert "needs: verification-scope" in dashboard
    assert "needs.verification-scope.outputs.dashboard == 'true'" in dashboard

    assert "scripts/run_ci.py --lane sqlite-cancellation" in sqlite
    assert "test_final_workspace_observer_restores_caller_cancellation_requests" in runner
    assert "test_delegated_stream_close_counts_checkpoint_cancellation_once" in runner
    assert "test_delegated_stream_close_distinguishes_restored_and_late_cancellation" in runner
    sidecar_verifier_command = "bash scripts/verify_release_sidecar_artifacts.sh"
    assert package_manifest.count(sidecar_verifier_command) == 1
    checks = _job_block(workflow, "package")
    assert "needs: package-build" in checks
    assert "group: [core, server, dashboard, authoring, coding, docker, service]" in checks
    assert "matrix.group == 'core'" in checks
    assert '["self-hosted","linux","arm64","gcp-ci"]' in checks
    assert "ubuntu-24.04-arm" in checks
    assert "name: release-dist" in checks
    assert "--package-artifacts" in checks
    assert "python3 scripts/run_ci.py --lane package" in package

    sidecar_verifier = _SIDECAR_VERIFIER.read_text()
    assert sidecar_verifier.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert 'mktemp -d "$RUNNER_TEMP/sidecar.XXXXXX"' in sidecar_verifier
    assert sidecar_verifier.count("lambda-microvm sidecar export") == 3
    assert "docker build --platform linux/arm64" in sidecar_verifier
    assert "docker run --rm --platform linux/arm64" in sidecar_verifier
    assert "mktemp -d)" not in package_manifest
    assert package_manifest.count('mktemp -d "$RUNNER_TEMP/') == 7

    for preserved_check in (
        "Verify the installed-wheel dashboard-to-local eval journey",
        "Check built-wheel generated Docker coding contract",
        "Check built-wheel secure public-service contract",
    ):
        check_offset = package_manifest.index(preserved_check)
        assert "publishing: true" not in package_manifest[check_offset : check_offset + 180]

    sidecar_offset = package_manifest.index(
        "Verify installed wheel and source-distribution sidecar exports"
    )
    assert "publishing: true" not in package_manifest[sidecar_offset : sidecar_offset + 180]


def test_ci_triggers_skip_main_pushes_and_cover_draft_transitions() -> None:
    workflow = _CI_WORKFLOW.read_text()
    triggers = workflow.split("on:\n", 1)[1].split("\npermissions:", 1)[0]
    assert "branches:" not in triggers
    assert 'tags: ["v*"]' in triggers
    assert "workflow_dispatch:" in triggers
    assert (
        "types: [opened, synchronize, reopened, ready_for_review, converted_to_draft]" in triggers
    )
    assert "group: ci-${{ github.ref }}" in workflow
    assert "cancel-in-progress: true" in workflow


def test_draft_prs_skip_every_independent_worker_before_allocation() -> None:
    workflow = _CI_WORKFLOW.read_text()
    draft_guard = "    if: github.event_name != 'pull_request' || !github.event.pull_request.draft"
    for name in (
        "verification-scope",
        "static",
        "test-collection",
        "test_shards",
        "test_specialists",
    ):
        block = _job_block(workflow, name)
        assert draft_guard in block
        assert block.index(draft_guard) < block.index("runs-on:")
    for name in ("sqlite-cancellation", "package-build", "dashboard"):
        assert "needs: verification-scope" in _job_block(workflow, name)
    assert "needs: package-build" in _job_block(workflow, "package")


def test_manual_tag_validation_cannot_publish() -> None:
    workflow = _CI_WORKFLOW.read_text()
    for name in ("publish", "github-release"):
        condition = _job_block(workflow, name).split("needs:", 1)[0]
        assert "github.event_name == 'push'" in condition
        assert "startsWith(github.ref, 'refs/tags/v')" in condition
