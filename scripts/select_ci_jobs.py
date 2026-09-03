from __future__ import annotations

import argparse
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_SCOPE_DEFINITION_PATHS = {
    ".github/workflows/ci.yml",
    "scripts/package_ci_steps.yml",
    "scripts/run_ci.py",
    "scripts/select_ci_jobs.py",
    "tests/core/test_ci_verification_scope.py",
    "tests/core/test_local_ci.py",
}
_SHARED_DEPENDENCY_PATHS = {
    "pyproject.toml",
    "uv.lock",
}
_SQLITE_EXACT_PATHS = {
    "src/cayu/_task_wait.py",
    "tests/core/test_runtime.py",
    "tests/core/test_runtime_event_emission_characterization.py",
    "tests/core/test_runtime_event_writer.py",
    "tests/core/test_subagent_cancellation.py",
    "tests/core/test_task_wait.py",
}
_DASHBOARD_EXACT_PATHS = {
    ".gitattributes",
    "scripts/generate_dashboard_api_types.py",
    "src/cayu/_server_contract_version.py",
    "src/cayu/_validation.py",
}
_RELEASE_EXACT_PATHS = {
    ".gitattributes",
    "LICENSE",
    "NOTICE",
    "README.md",
    "docs/release-notes.md",
    "examples/dashboard_behavior_live.py",
    "examples/evals_judge_calibration_live.py",
    "examples/evals_release_acceptance_live.py",
    "pyproject.toml",
    "scripts/check_release_artifacts.py",
    "scripts/build_dashboard_source_bundle.py",
    "scripts/check_ejected_dashboard_build.py",
    "scripts/extract_release_notes.py",
    "scripts/generate_sidecar_manifest.py",
    "scripts/smoke_ejected_dashboard.py",
    "scripts/smoke_built_wheel_serve.py",
    "scripts/verify_release_sidecar_artifacts.sh",
    "scripts/smoke_built_wheel_doctor.py",
    "scripts/verify_release_state.py",
    "src/cayu/_server_contract_version.py",
    "src/cayu/runtime/system_diagnostics.py",
    "src/cayu/server/dashboard/THIRD_PARTY_LICENSES.md",
    "src/cayu/support_bundles.py",
    "tests/cli/test_scaffold_docker_live.py",
    "uv.lock",
}


@dataclass(frozen=True)
class VerificationScope:
    dashboard: bool
    release_artifacts: bool
    sqlite_cancellation: bool

    def render_github_outputs(self) -> str:
        return "\n".join(
            (
                f"dashboard={str(self.dashboard).lower()}",
                f"release_artifacts={str(self.release_artifacts).lower()}",
                f"sqlite_cancellation={str(self.sqlite_cancellation).lower()}",
            )
        )


def select_pull_request_jobs(changed_paths: Iterable[str]) -> VerificationScope:
    paths = {_normalize_path(path) for path in changed_paths}
    scope_definition_changed = bool(paths & _SCOPE_DEFINITION_PATHS)
    dependency_changed = bool(paths & _SHARED_DEPENDENCY_PATHS)
    return VerificationScope(
        dashboard=scope_definition_changed
        or dependency_changed
        or any(_affects_dashboard(path) for path in paths),
        release_artifacts=scope_definition_changed
        or any(_affects_release_artifacts(path) for path in paths),
        sqlite_cancellation=scope_definition_changed
        or dependency_changed
        or any(_affects_sqlite_cancellation(path) for path in paths),
    )


def _affects_sqlite_cancellation(path: str) -> bool:
    return (
        path in _SQLITE_EXACT_PATHS
        or path.startswith(("src/cayu/runtime/", "src/cayu/storage/", "tests/runtime/"))
        or (
            path.startswith("tests/")
            and any(component in path for component in ("sqlite", "session_store"))
        )
    )


def _affects_dashboard(path: str) -> bool:
    return path in _DASHBOARD_EXACT_PATHS or path.startswith(
        (
            "dashboard/",
            "src/cayu/artifacts/",
            "src/cayu/core/",
            "src/cayu/evals/",
            "src/cayu/runtime/",
            "src/cayu/server/",
            "src/cayu/storage/",
            "tests/server/",
        )
    )


def _affects_release_artifacts(path: str) -> bool:
    if path in _RELEASE_EXACT_PATHS:
        return True
    if path.startswith(
        (
            "dashboard/",
            "maintenance/model_catalog/",
            "src/cayu/cli/",
            "src/cayu/data/",
            "src/cayu/guides/",
        )
    ):
        return True
    return (
        path.startswith("src/cayu/")
        and ("dashboard" in path or "lambda_microvm" in path or "manifest" in path)
    ) or (
        path.startswith("examples/aws/")
        and ("lambda_microvm" in path or path.startswith("examples/aws/lambda_microvm"))
    )


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").removeprefix("./")


def _changed_paths(base_sha: str, head_sha: str) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            f"{base_sha}...{head_sha}",
        ],
        check=True,
        capture_output=True,
    )
    return [path.decode(errors="surrogateescape") for path in completed.stdout.split(b"\0") if path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select expensive CI jobs from pull-request changed paths."
    )
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    args = parser.parse_args(argv)
    scope = select_pull_request_jobs(_changed_paths(args.base_sha, args.head_sha))
    print(scope.render_github_outputs())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
