from __future__ import annotations

import re
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).parents[2]
_CI_WORKFLOW = _REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
_TAG_VERIFIER = _REPOSITORY_ROOT / ".github" / "actions" / "verify-release-tag" / "action.yml"
_RELEASE_RUNBOOK = _REPOSITORY_ROOT / "docs" / "releasing.md"
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


def test_release_jobs_pin_every_external_action_to_immutable_commit() -> None:
    workflow = _CI_WORKFLOW.read_text()
    references = [
        reference
        for job_name in (
            "static",
            "test",
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
    assert "0.1.0a1" not in runbook
    assert 'version="$(python -c' in runbook


def test_release_workflow_gates_publish_and_reuses_validated_artifact() -> None:
    workflow = _CI_WORKFLOW.read_text()
    package = _job_block(workflow, "package")
    publish = _job_block(workflow, "publish")
    github_release = _job_block(workflow, "github-release")

    assert 'tags: ["v*"]' in workflow
    assert "startsWith(github.ref, 'refs/tags/v')" in publish
    assert "vars.PYPI_PUBLISH_ENABLED == 'true'" in publish
    assert "needs: [static, test, package, dashboard]" in publish
    assert "if: startsWith(github.ref, 'refs/tags/v')" in github_release
    assert "needs: [publish, package]" in github_release

    assert "prerelease: ${{ steps.release-version.outputs.prerelease }}" in package
    assert "id: release-version" in package
    assert "Version(version).is_prerelease" in package
    upload = package.index("name: Upload release distribution")
    assert package.index("name: Check installed CLI version") < upload
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
