from __future__ import annotations

import runpy
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_SELECTOR = runpy.run_path(str(_ROOT / "scripts" / "select_ci_jobs.py"))
VerificationScope = _SELECTOR["VerificationScope"]
select_pull_request_jobs = _SELECTOR["select_pull_request_jobs"]


def test_unrelated_pull_request_keeps_expensive_jobs_skipped() -> None:
    assert select_pull_request_jobs(
        [
            "docs/runtime-contracts.md",
            "src/cayu/tools/search.py",
            "tests/core/test_search_text_tool.py",
        ]
    ) == VerificationScope(
        dashboard=False,
        release_artifacts=False,
        sqlite_cancellation=False,
    )


def test_sqlite_regression_path_selects_only_cross_version_lane() -> None:
    expected = VerificationScope(
        dashboard=False,
        release_artifacts=False,
        sqlite_cancellation=True,
    )
    assert select_pull_request_jobs(["tests/examples/test_sqlite_example_paths.py"]) == expected
    assert select_pull_request_jobs(["src/cayu/_task_wait.py"]) == expected
    assert select_pull_request_jobs(["tests/core/test_task_wait.py"]) == expected
    assert select_pull_request_jobs(["tests/core/test_subagent_cancellation.py"]) == expected


def test_runtime_change_selects_sqlite_and_dashboard_contract_lanes() -> None:
    assert select_pull_request_jobs(["src/cayu/runtime/sessions.py"]) == VerificationScope(
        dashboard=True,
        release_artifacts=False,
        sqlite_cancellation=True,
    )


def test_dashboard_contract_changes_select_dashboard_and_release_artifact_lanes() -> None:
    assert select_pull_request_jobs(["dashboard/src/api.ts"]) == VerificationScope(
        dashboard=True,
        release_artifacts=True,
        sqlite_cancellation=False,
    )

    dashboard_only = VerificationScope(
        dashboard=True,
        release_artifacts=False,
        sqlite_cancellation=False,
    )
    assert select_pull_request_jobs(["scripts/generate_dashboard_api_types.py"]) == dashboard_only
    assert select_pull_request_jobs(["src/cayu/artifacts/base.py"]) == dashboard_only
    assert select_pull_request_jobs(["src/cayu/evals/corpus.py"]) == dashboard_only
    assert select_pull_request_jobs([".gitattributes"]) == VerificationScope(
        dashboard=True,
        release_artifacts=True,
        sqlite_cancellation=False,
    )


def test_release_input_selects_only_release_artifact_lane() -> None:
    expected = VerificationScope(
        dashboard=False,
        release_artifacts=True,
        sqlite_cancellation=False,
    )
    assert select_pull_request_jobs(["LICENSE"]) == expected
    assert select_pull_request_jobs(["README.md"]) == expected
    assert select_pull_request_jobs(["docs/release-notes.md"]) == expected
    assert select_pull_request_jobs(["scripts/extract_release_notes.py"]) == expected
    assert select_pull_request_jobs(["scripts/verify_release_state.py"]) == expected
    assert select_pull_request_jobs(
        ["src/cayu/server/dashboard/THIRD_PARTY_LICENSES.md"]
    ) == VerificationScope(
        dashboard=True,
        release_artifacts=True,
        sqlite_cancellation=False,
    )


def test_cloud_cli_changes_select_release_artifact_lane() -> None:
    expected = VerificationScope(
        dashboard=False,
        release_artifacts=True,
        sqlite_cancellation=False,
    )
    assert select_pull_request_jobs(["src/cayu/cli/cloud.py"]) == expected
    assert select_pull_request_jobs(["src/cayu/cli/_cloud_project.py"]) == expected


def test_dependency_or_scope_changes_fail_open() -> None:
    all_jobs = VerificationScope(
        dashboard=True,
        release_artifacts=True,
        sqlite_cancellation=True,
    )
    assert select_pull_request_jobs(["pyproject.toml"]) == all_jobs
    assert select_pull_request_jobs([".github/workflows/ci.yml"]) == all_jobs
    assert select_pull_request_jobs(["scripts/select_ci_jobs.py"]) == all_jobs


def test_github_outputs_are_stable_lowercase_booleans() -> None:
    assert VerificationScope(
        dashboard=True,
        release_artifacts=False,
        sqlite_cancellation=True,
    ).render_github_outputs() == (
        "dashboard=true\nrelease_artifacts=false\nsqlite_cancellation=true"
    )
