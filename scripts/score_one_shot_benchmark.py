"""Score a published Cayu fresh-agent benchmark evidence report."""

from __future__ import annotations

import argparse
import json
import stat
import tarfile
import zipfile
from collections import Counter
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

_ARCHETYPES = ("rfp", "research_document", "coding_repository")
_EVIDENCE_LAYERS = ("inspect", "check", "tests", "eval", "process_boundary", "live")
_REQUIRED_PASS_LAYERS = frozenset({"inspect", "check", "tests", "eval", "process_boundary"})
_CASE_REQUIREMENTS = {
    "rfp": frozenset(
        {
            "domain_input",
            "model_decision",
            "explicit_effect_tool",
            "approval_boundary",
            "durable_state",
            "deterministic_test",
            "trajectory_eval",
        }
    ),
    "research_document": frozenset(
        {
            "document_input",
            "artifact_output",
            "citations",
            "deterministic_test",
            "trajectory_eval",
            "omits_unneeded_subsystems",
        }
    ),
    "coding_repository": frozenset(
        {
            "isolated_workspace",
            "narrow_command_policy",
            "patch_artifact",
            "no_delivery_effect",
            "deterministic_test",
            "trajectory_eval",
        }
    ),
}
_SCHEMA_PATH = Path(__file__).parents[1] / "benchmarks" / "one_shot" / "trial-report.schema.json"
_REPOSITORY_ROOT = _SCHEMA_PATH.parents[2]


def score_report(
    report: dict[str, Any],
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    _validate_schema(report)
    _validate_run_controls(report["run"])
    trials = report["trials"]

    violations: list[str] = []
    trial_failures: list[str] = []
    ids: list[str] = []
    submission_claims: dict[str, list[tuple[str, str]]] = {}
    evidence_claims: dict[str, list[tuple[str, str]]] = {}
    by_archetype: dict[str, list[dict[str, Any]]] = {name: [] for name in _ARCHETYPES}
    passed_trials: set[str] = set()
    for trial in trials:
        trial_failure_start = len(trial_failures)
        trial_id = trial["id"]
        ids.append(trial_id)
        archetype = trial["archetype"]
        if archetype not in by_archetype:
            violations.append(f"{trial_id} used unknown archetype {archetype!r}")
            continue
        by_archetype[archetype].append(trial)

        hints = trial["framework_specific_hints"]
        if hints:
            violations.append(f"{trial_id} used {hints} framework-specific human hint(s)")
        private_violations = trial["private_interface_violations"]
        security_violations = trial["security_violations"]
        if private_violations:
            violations.append(f"{trial_id} recorded private-interface violations")
        if trial["public_interfaces_only"] is not True:
            violations.append(f"{trial_id} did not use public Cayu interfaces only")
        if security_violations:
            violations.append(f"{trial_id} recorded security violations")
        if trial["clarification_count"] != len(trial["clarifications"]):
            violations.append(f"{trial_id} clarification_count does not match clarifications")
        failures = trial.get("failures", [])
        _score_failure_references(trial_id, failures, violations=violations)

        submission_passed = _score_submission(
            trial_id,
            submission_path=trial["submission_path"],
            submission_artifact=trial["submission_artifact"],
            artifact_root=artifact_root,
            violations=violations,
        )
        submission_identity = _artifact_identity(trial["submission_path"], artifact_root)
        submission_claims.setdefault(submission_identity, []).append(
            (trial_id, trial["submission_path"])
        )
        claimed_evidence_paths: list[str] = []
        evidence_passed = _score_evidence(
            trial_id,
            trial["evidence"],
            submission_path=trial["submission_path"],
            claimed_paths=claimed_evidence_paths,
            artifact_root=artifact_root,
            violations=violations,
            trial_failures=trial_failures,
        )
        case_passed = _score_case_acceptance(
            trial_id,
            archetype,
            trial["case_acceptance"],
            submission_path=trial["submission_path"],
            claimed_paths=claimed_evidence_paths,
            artifact_root=artifact_root,
            violations=violations,
            trial_failures=trial_failures,
        )
        claimed_identities = [
            _artifact_identity(path, artifact_root) for path in claimed_evidence_paths
        ]
        for path, identity in zip(claimed_evidence_paths, claimed_identities, strict=True):
            evidence_claims.setdefault(identity, []).append((trial_id, path))
        reused_identities = {
            identity for identity, count in Counter(claimed_identities).items() if count > 1
        }
        reused_paths = sorted(
            {
                path
                for path, identity in zip(claimed_evidence_paths, claimed_identities, strict=True)
                if identity in reused_identities
            }
        )
        for path in reused_paths:
            violations.append(f"{trial_id} evidence path is reused by multiple claims: {path}")
        evidence_index_passed = not reused_paths
        if len(trial_failures) == trial_failure_start:
            if trial["first_submission_acceptance_passed"] is False:
                trial_failures.append(f"{trial_id} declared first-submission acceptance failed")
            elif failures:
                trial_failures.append(f"{trial_id} records classified failure(s)")

        passed = (
            trial["first_submission_acceptance_passed"] is True
            and hints == 0
            and trial["public_interfaces_only"] is True
            and not private_violations
            and not security_violations
            and not failures
            and submission_passed
            and evidence_passed
            and case_passed
            and evidence_index_passed
        )
        if passed:
            passed_trials.add(trial_id)

    duplicate_ids = sorted(name for name, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        violations.append(f"duplicate trial ids: {', '.join(duplicate_ids)}")
        passed_trials.difference_update(duplicate_ids)
    reused_trial_ids = _score_cross_trial_reuse(
        "submission path",
        submission_claims,
        violations=violations,
    )
    reused_trial_ids.update(
        _score_cross_trial_reuse(
            "evidence path",
            evidence_claims,
            violations=violations,
        )
    )
    passed_trials.difference_update(reused_trial_ids)
    for trial in trials:
        if trial["id"] not in passed_trials and not trial.get("failures"):
            violations.append(
                f"{trial['id']} failed without classified dispositions and follow-up references"
            )
    passed_count = len(passed_trials)

    archetype_scores: dict[str, dict[str, int | float]] = {}
    for archetype in _ARCHETYPES:
        group = by_archetype[archetype]
        group_passed = sum(1 for trial in group if trial["id"] in passed_trials)
        total = len(group)
        rate = group_passed / total if total else 0.0
        archetype_scores[archetype] = {
            "passed": group_passed,
            "total": total,
            "rate": round(rate, 4),
        }
        if total < 3:
            violations.append(f"{archetype} requires at least 3 published trials; found {total}")
        if total and rate < 2 / 3:
            violations.append(f"{archetype} first-submission pass rate {rate:.1%} is below 66.7%")

    total = len(trials)
    aggregate_rate = passed_count / total if total else 0.0
    if aggregate_rate < 0.8:
        violations.append(
            f"aggregate first-submission pass rate {aggregate_rate:.1%} is below 80.0%"
        )
    return {
        "schema_version": "1",
        "passed": not violations,
        "aggregate": {
            "passed": passed_count,
            "total": total,
            "rate": round(aggregate_rate, 4),
        },
        "archetypes": archetype_scores,
        "trial_failures": trial_failures,
        "violations": violations,
    }


def _artifact_identity(relative_path: str, artifact_root: Path | None) -> str:
    if artifact_root is None:
        return PurePosixPath(relative_path).as_posix()
    return str((artifact_root.resolve() / relative_path).resolve())


def _score_cross_trial_reuse(
    kind: str,
    claims: dict[str, list[tuple[str, str]]],
    *,
    violations: list[str],
) -> set[str]:
    repeated_trial_ids: set[str] = set()
    for entries in claims.values():
        trial_ids = sorted({trial_id for trial_id, _ in entries})
        if len(trial_ids) < 2:
            continue
        repeated_trial_ids.update(trial_ids)
        paths = sorted({path for _, path in entries})
        violations.append(
            f"{kind} is reused across trials {', '.join(trial_ids)}: {', '.join(paths)}"
        )
    return repeated_trial_ids


def _score_failure_references(
    trial_id: str,
    failures: list[dict[str, Any]],
    *,
    violations: list[str],
) -> None:
    root = _REPOSITORY_ROOT.resolve()
    for index, failure in enumerate(failures):
        reference = failure["reference"]
        if reference["kind"] != "repository_path":
            continue
        candidate = (root / reference["value"]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            violations.append(
                f"{trial_id} failure {index} repository reference escapes the Cayu checkout: "
                f"{reference['value']}"
            )
            continue
        if not candidate.is_file():
            violations.append(
                f"{trial_id} failure {index} references a missing repository fixture: "
                f"{reference['value']}"
            )


def _validate_schema(report: dict[str, Any]) -> None:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema).iter_errors(report),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = ".".join(str(part) for part in error.absolute_path) or "report"
    raise ValueError(f"benchmark report schema error at {path}: {error.message}")


def _validate_run_controls(run: dict[str, Any]) -> None:
    cost = run["limits"]["cost"]
    if cost["mode"] != "limited":
        return
    amount = cost["amount"]
    try:
        finite = isfinite(amount)
    except OverflowError:
        finite = isinstance(amount, int)
    if not finite:
        raise ValueError(
            "benchmark report schema error at run.limits.cost.amount: value must be finite"
        )


def _score_evidence(
    trial_id: str,
    evidence: dict[str, dict[str, Any]],
    *,
    submission_path: str,
    claimed_paths: list[str],
    artifact_root: Path | None,
    violations: list[str],
    trial_failures: list[str],
) -> bool:
    passed = True
    for layer in _EVIDENCE_LAYERS:
        item = evidence[layer]
        status = item["status"]
        if layer in _REQUIRED_PASS_LAYERS and status != "passed":
            trial_failures.append(f"{trial_id} required evidence layer {layer} is {status}")
            passed = False
        if status == "passed" and item["exit_code"] != 0:
            violations.append(f"{trial_id} {layer} claims passed with nonzero exit code")
            passed = False
        if status == "failed" and item["exit_code"] in (None, 0):
            violations.append(f"{trial_id} {layer} claims failed without a nonzero exit code")
            passed = False
        if status == "not_run":
            if any(item[field] is not None for field in ("command", "exit_code", "output_path")):
                violations.append(f"{trial_id} {layer} not_run evidence includes execution data")
                passed = False
            if item["note"] is None:
                violations.append(f"{trial_id} {layer} not_run evidence requires a note")
                passed = False
            continue
        if item["command"] is None or item["output_path"] is None:
            violations.append(f"{trial_id} {layer} executed evidence is incomplete")
            passed = False
            continue
        claimed_paths.append(item["output_path"])
        passed = (
            _validate_evidence_location(
                trial_id,
                item["output_path"],
                submission_path=submission_path,
                violations=violations,
            )
            and passed
        )
        if artifact_root is not None:
            passed = (
                _validate_artifact_path(
                    trial_id,
                    item["output_path"],
                    artifact_root=artifact_root,
                    violations=violations,
                    scope_path=f"{submission_path}/evidence",
                )
                and passed
            )
    return passed


def _score_case_acceptance(
    trial_id: str,
    archetype: str,
    acceptance: list[dict[str, Any]],
    *,
    submission_path: str,
    claimed_paths: list[str],
    artifact_root: Path | None,
    violations: list[str],
    trial_failures: list[str],
) -> bool:
    by_id = Counter(item["id"] for item in acceptance)
    expected = _CASE_REQUIREMENTS[archetype]
    observed = frozenset(by_id)
    passed = True
    if observed != expected or any(count != 1 for count in by_id.values()):
        violations.append(
            f"{trial_id} case acceptance ids differ: expected {', '.join(sorted(expected))}"
        )
        passed = False
    for item in acceptance:
        if item["passed"] is not True:
            trial_failures.append(f"{trial_id} case requirement {item['id']} failed")
            passed = False
        claimed_paths.append(item["evidence_path"])
        passed = (
            _validate_evidence_location(
                trial_id,
                item["evidence_path"],
                submission_path=submission_path,
                violations=violations,
            )
            and passed
        )
        if artifact_root is not None:
            passed = (
                _validate_artifact_path(
                    trial_id,
                    item["evidence_path"],
                    artifact_root=artifact_root,
                    violations=violations,
                    scope_path=f"{submission_path}/evidence",
                )
                and passed
            )
    return passed


def _score_submission(
    trial_id: str,
    *,
    submission_path: str,
    submission_artifact: str,
    artifact_root: Path | None,
    violations: list[str],
) -> bool:
    passed = True
    if not _is_relative_path_within(submission_artifact, submission_path):
        violations.append(
            f"{trial_id} first-submission artifact must be inside its submission directory"
        )
        passed = False
    if artifact_root is not None:
        directory_valid = _validate_artifact_path(
            trial_id,
            submission_path,
            artifact_root=artifact_root,
            violations=violations,
            directory=True,
        )
        artifact_valid = _validate_artifact_path(
            trial_id,
            submission_artifact,
            artifact_root=artifact_root,
            violations=violations,
            kind="first-submission artifact",
            scope_path=submission_path,
        )
        passed = directory_valid and artifact_valid and passed
        if artifact_valid:
            passed = (
                _validate_submission_artifact_format(
                    trial_id,
                    submission_artifact,
                    artifact_root=artifact_root,
                    violations=violations,
                )
                and passed
            )
    return passed


def _validate_evidence_location(
    trial_id: str,
    relative_path: str,
    *,
    submission_path: str,
    violations: list[str],
) -> bool:
    evidence_root = (PurePosixPath(submission_path) / "evidence").as_posix()
    if _is_relative_path_within(relative_path, evidence_root):
        return True
    violations.append(
        f"{trial_id} evidence path is outside its submission evidence directory: {relative_path}"
    )
    return False


def _is_relative_path_within(child: str, parent: str) -> bool:
    child_path = PurePosixPath(child)
    parent_path = PurePosixPath(parent)
    if (
        child_path.is_absolute()
        or parent_path.is_absolute()
        or ".." in child_path.parts
        or ".." in parent_path.parts
    ):
        return False
    try:
        child_path.relative_to(parent_path)
    except ValueError:
        return False
    return child_path != parent_path


def _validate_submission_artifact_format(
    trial_id: str,
    relative_path: str,
    *,
    artifact_root: Path,
    violations: list[str],
) -> bool:
    path = (artifact_root.resolve() / relative_path).resolve()
    lower_name = path.name.lower()
    try:
        if lower_name.endswith((".diff", ".patch")):
            if _valid_git_diff(path.read_bytes()):
                return True
        elif lower_name.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                files = {
                    PurePosixPath(item.filename): item.file_size
                    for item in archive.infolist()
                    if not item.is_dir()
                }
                if (
                    files
                    and all(_safe_archive_member(item.filename) for item in archive.infolist())
                    and not any(_zip_member_is_symlink(item) for item in archive.infolist())
                    and _archive_contains_project(files)
                ):
                    return True
        elif lower_name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, mode="r:gz") as archive:
                members = archive.getmembers()
                files = {PurePosixPath(item.name): item.size for item in members if item.isfile()}
                if (
                    files
                    and all(_safe_archive_member(item.name) for item in members)
                    and all(item.isfile() or item.isdir() for item in members)
                    and _archive_contains_project(files)
                ):
                    return True
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        pass
    violations.append(
        f"{trial_id} first-submission artifact is not a non-empty diff or safe archive: "
        f"{relative_path}"
    )
    return False


def _safe_archive_member(name: str) -> bool:
    member = PurePosixPath(name)
    return not member.is_absolute() and ".." not in member.parts


def _valid_git_diff(content: bytes) -> bool:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return False
    return (
        any(line.startswith("diff --git a/") and " b/" in line for line in lines)
        and any(line.startswith("--- ") for line in lines)
        and any(line.startswith("+++ ") for line in lines)
        and any(line.startswith("@@ ") for line in lines)
        and any(
            (line.startswith("+") and not line.startswith("+++"))
            or (line.startswith("-") and not line.startswith("---"))
            for line in lines
        )
    )


def _zip_member_is_symlink(item: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(item.external_attr >> 16)


def _archive_contains_project(files: dict[PurePosixPath, int]) -> bool:
    required = {"AGENTS.md", "app.py", "pyproject.toml"}
    by_parent: dict[PurePosixPath, set[str]] = {}
    for path, size in files.items():
        if size <= 0:
            continue
        by_parent.setdefault(path.parent, set()).add(path.name)
    return any(required <= names for names in by_parent.values())


def _validate_artifact_path(
    trial_id: str,
    relative_path: str,
    *,
    artifact_root: Path,
    violations: list[str],
    directory: bool = False,
    kind: str | None = None,
    scope_path: str | None = None,
) -> bool:
    root = artifact_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        violations.append(f"{trial_id} evidence path escapes the report root: {relative_path}")
        return False
    if scope_path is not None:
        scope = (root / scope_path).resolve()
        try:
            candidate.relative_to(scope)
        except ValueError:
            scoped_kind = kind or "evidence file"
            violations.append(
                f"{trial_id} {scoped_kind} path escapes its required scope: {relative_path}"
            )
            return False
    valid = (
        candidate.is_dir() if directory else candidate.is_file() and candidate.stat().st_size > 0
    )
    if not valid:
        missing_kind = kind or ("submission directory" if directory else "non-empty evidence file")
        violations.append(f"{trial_id} missing {missing_kind}: {relative_path}")
    return valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    score = score_report(report, artifact_root=args.report.parent)
    print(json.dumps(score, indent=2, sort_keys=True))
    return 0 if score["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
