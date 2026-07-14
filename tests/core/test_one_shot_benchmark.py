from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest

scorer = runpy.run_path(str(Path(__file__).parents[2] / "scripts" / "score_one_shot_benchmark.py"))
score_report = scorer["score_report"]
CASE_REQUIREMENTS = scorer["CASE_REQUIREMENTS"]
BENCHMARK_ROOT = Path(__file__).parents[2] / "benchmarks" / "one_shot"
PROMPT_TOOL_ALIGNMENT_EXAMPLE = json.loads(
    (BENCHMARK_ROOT / "prompt-tool-alignment.example.json").read_text(encoding="utf-8")
)


def _run() -> dict:
    return {
        "cayu_version": "0.1.0",
        "wheel_sha256": "0" * 64,
        "source_commit": "abc123",
        "operating_system": "test",
        "python_version": "3.11",
        "agent": "test-agent",
        "model": "test-model",
        "reasoning": "xhigh",
        "agent_configuration": {"profile": "default"},
        "permissions": {"sandbox": "workspace-write"},
        "allowed_tools": ["shell"],
        "network_policy": "disabled",
        "limits": {
            "elapsed_time": {"mode": "no_limit"},
            "cost": {"mode": "no_limit"},
        },
    }


def _evidence(output_path: str, status: str = "passed") -> dict:
    if status == "not_run":
        return {
            "status": "not_run",
            "command": None,
            "exit_code": None,
            "output_path": None,
            "note": "credentials unavailable",
        }
    return {
        "status": status,
        "command": "verify",
        "exit_code": 0 if status == "passed" else 1,
        "output_path": output_path,
        "note": None,
    }


def _trial(archetype: str, number: int, *, passed: bool = True) -> dict:
    trial_id = f"{archetype}-{number}"
    submission = f"submissions/{trial_id}"
    return {
        "id": trial_id,
        "archetype": archetype,
        "submission_path": submission,
        "submission_artifact": f"{submission}/first-submission.diff",
        "first_submission_acceptance_passed": passed,
        "framework_specific_hints": 0,
        "public_interfaces_only": True,
        "private_interface_violations": [],
        "security_violations": [],
        "clarification_count": 1,
        "clarifications": [{"question": "Who uses it?", "answer": "One local user."}],
        "case_acceptance": [
            {
                "id": requirement,
                "passed": passed,
                "evidence_path": (
                    f"{submission}/evidence/case-{requirement}.json"
                    if requirement == "prompt_tool_alignment"
                    else f"{submission}/evidence/case-{requirement}.txt"
                ),
            }
            for requirement in sorted(CASE_REQUIREMENTS[archetype])
        ],
        "evidence": {
            "inspect": _evidence(f"{submission}/evidence/inspect.txt"),
            "check": _evidence(f"{submission}/evidence/check.txt"),
            "tests": _evidence(f"{submission}/evidence/tests.txt"),
            "eval": _evidence(f"{submission}/evidence/eval.txt"),
            "process_boundary": _evidence(f"{submission}/evidence/process-boundary.txt"),
            "live": _evidence("unused", "not_run"),
        },
    }


def _trials() -> list[dict]:
    return [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]


def _materialize_trial(root: Path, trial: dict) -> None:
    artifact = root / trial["submission_artifact"]
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+app\n",
        encoding="utf-8",
    )
    paths = [
        item["output_path"]
        for item in trial["evidence"].values()
        if item["output_path"] is not None
    ]
    paths.extend(item["evidence_path"] for item in trial["case_acceptance"])
    alignment = next(
        (item for item in trial["case_acceptance"] if item["id"] == "prompt_tool_alignment"),
        None,
    )
    for relative in paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if alignment is not None and relative == alignment["evidence_path"]:
            path.write_text(
                json.dumps(_prompt_tool_alignment_artifact(), indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            path.write_text(f"evidence for {relative}\n", encoding="utf-8")


def _prompt_tool_alignment_artifact() -> dict:
    return copy.deepcopy(PROMPT_TOOL_ALIGNMENT_EXAMPLE)


def test_case_requirements_are_loaded_from_the_published_manifest() -> None:
    manifest = json.loads((BENCHMARK_ROOT / "case-requirements.json").read_text(encoding="utf-8"))
    expected = {
        archetype: frozenset(requirements)
        for archetype, requirements in manifest["archetypes"].items()
    }

    assert expected == CASE_REQUIREMENTS


def test_score_report_enforces_the_published_one_shot_gate() -> None:
    trials = [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]

    score = score_report({"schema_version": "1", "run": _run(), "trials": trials})

    assert score["passed"] is True
    assert score["aggregate"] == {"passed": 9, "total": 9, "rate": 1.0}
    assert score["archetypes"]["rfp"] == {"passed": 3, "total": 3, "rate": 1.0}
    assert score["violations"] == []


def test_score_report_requires_distinct_command_selector_evidence() -> None:
    trials = _trials()
    coding_trial = next(trial for trial in trials if trial["id"] == "coding_repository-1")
    coding_trial["case_acceptance"] = [
        item
        for item in coding_trial["case_acceptance"]
        if item["id"] != "workspace_side_effect_containment"
    ]
    coding_trial["failures"] = [
        {
            "classification": "unrealistic_test_boundary",
            "disposition": "regression_fixture",
            "reference": {
                "kind": "repository_path",
                "value": "tests/core/test_one_shot_benchmark.py",
            },
        }
    ]

    score = score_report({"schema_version": "1", "run": _run(), "trials": trials})

    assert any(
        violation.startswith("coding_repository-1 case acceptance ids differ:")
        and "selector_argument_boundary" in violation
        and "workspace_side_effect_containment" in violation
        and "check_outcome_classification" in violation
        and "selector_scope_reporting" in violation
        for violation in score["violations"]
    )


def test_score_report_allows_one_honestly_classified_behavioral_failure() -> None:
    trials = [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]
    next(item for item in trials[0]["case_acceptance"] if item["id"] == "domain_input")[
        "passed"
    ] = False
    trials[0]["failures"] = [
        {
            "classification": "insufficient_module_depth_or_caller_lifecycle",
            "disposition": "authoring_or_diagnostic_improvement",
            "reference": {
                "kind": "pull_request_url",
                "value": "https://github.com/vertexkg/cayu/pull/287",
            },
        }
    ]

    score = score_report({"schema_version": "1", "run": _run(), "trials": trials})

    assert score["passed"] is True
    assert score["aggregate"] == {"passed": 8, "total": 9, "rate": 0.8889}
    assert score["archetypes"]["rfp"] == {"passed": 2, "total": 3, "rate": 0.6667}
    assert score["violations"] == []
    assert score["trial_failures"] == ["rfp-1 case requirement domain_input failed"]


def test_score_report_rejects_selection_bias_hints_and_security_violations() -> None:
    trials = [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]
    trials[0]["framework_specific_hints"] = 1
    trials[1]["security_violations"] = ["external tool used AllowAllToolPolicy"]
    trials[2]["public_interfaces_only"] = False
    trials.pop()

    score = score_report({"schema_version": "1", "run": _run(), "trials": trials})

    assert score["passed"] is False
    assert "coding_repository requires at least 3 published trials; found 2" in score["violations"]
    assert "rfp-1 used 1 framework-specific human hint(s)" in score["violations"]
    assert "rfp-2 recorded security violations" in score["violations"]
    assert "rfp-3 did not use public Cayu interfaces only" in score["violations"]


def test_score_report_validates_schema_and_evidence_artifacts(tmp_path: Path) -> None:
    trials = [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]
    report = {"schema_version": "1", "run": _run(), "trials": trials}

    score = score_report(report, artifact_root=tmp_path)

    assert score["passed"] is False
    assert "rfp-1 missing submission directory: submissions/rfp-1" in score["violations"]
    assert (
        "rfp-1 missing first-submission artifact: submissions/rfp-1/first-submission.diff"
    ) in score["violations"]
    assert (
        "rfp-1 missing non-empty evidence file: submissions/rfp-1/evidence/inspect.txt"
    ) in score["violations"]

    report["run"].pop("wheel_sha256")
    with pytest.raises(ValueError, match="schema error at run"):
        score_report(report)


def test_score_report_requires_an_explicit_reasoning_setting() -> None:
    trials = _trials()
    run = _run()
    run.pop("reasoning")

    with pytest.raises(ValueError, match="reasoning"):
        score_report({"schema_version": "1", "run": run, "trials": trials})


@pytest.mark.parametrize("field", ["agent_configuration", "permissions"])
def test_score_report_rejects_empty_run_configuration(field: str) -> None:
    trials = _trials()
    run = _run()
    run[field] = {}

    with pytest.raises(ValueError, match=field):
        score_report({"schema_version": "1", "run": run, "trials": trials})


def test_score_report_requires_typed_elapsed_and_cost_limits() -> None:
    trials = _trials()
    run = _run()
    run["limits"] = {"unrelated": "x"}

    with pytest.raises(ValueError, match="limits"):
        score_report({"schema_version": "1", "run": run, "trials": trials})


def test_score_report_accepts_typed_finite_run_limits() -> None:
    trials = _trials()
    run = _run()
    run["limits"] = {
        "elapsed_time": {"mode": "limited", "seconds": 900},
        "cost": {"mode": "limited", "amount": 0, "currency": "USD"},
    }

    score = score_report({"schema_version": "1", "run": run, "trials": trials})

    assert score["passed"] is True


def test_score_report_rejects_a_non_finite_cost_limit() -> None:
    trials = _trials()
    run = _run()
    run["limits"] = {
        "elapsed_time": {"mode": "limited", "seconds": 900},
        "cost": {"mode": "limited", "amount": float("inf"), "currency": "USD"},
    }

    with pytest.raises(ValueError, match="finite"):
        score_report({"schema_version": "1", "run": run, "trials": trials})


def test_score_report_accepts_a_large_finite_integer_cost_limit() -> None:
    run = _run()
    run["limits"] = {
        "elapsed_time": {"mode": "limited", "seconds": 900},
        "cost": {"mode": "limited", "amount": 10**1000, "currency": "USD"},
    }

    score = score_report({"schema_version": "1", "run": run, "trials": _trials()})

    assert score["passed"] is True


def test_score_report_requires_failure_classification_for_failed_trials() -> None:
    trial = _trial("rfp", 1, passed=False)

    with pytest.raises(ValueError, match="failures"):
        score_report({"schema_version": "1", "run": _run(), "trials": [trial]})


def test_score_report_accepts_a_classified_failed_trial() -> None:
    trial = _trial("rfp", 1, passed=False)
    for item in trial["case_acceptance"]:
        item["passed"] = True
    trial["failures"] = [
        {
            "classification": "missing_capability",
            "disposition": "linked_capability_issue",
            "reference": {
                "kind": "issue_url",
                "value": "https://github.com/vertexkg/cayu/issues/254",
            },
        },
        {
            "classification": "unrealistic_test_boundary",
            "disposition": "regression_fixture",
            "reference": {
                "kind": "repository_path",
                "value": "tests/core/test_one_shot_benchmark.py",
            },
        },
    ]

    score = score_report({"schema_version": "1", "run": _run(), "trials": [trial]})

    assert score["passed"] is False
    assert not any("failed without classified dispositions" in item for item in score["violations"])
    assert score["trial_failures"] == ["rfp-1 declared first-submission acceptance failed"]


def test_score_report_rejects_an_unverifiable_failure_reference() -> None:
    trial = _trial("rfp", 1, passed=False)
    trial["failures"] = [
        {
            "classification": "missing_capability",
            "disposition": "linked_capability_issue",
            "reference": {"kind": "issue_url", "value": "x"},
        }
    ]

    with pytest.raises(ValueError, match="x"):
        score_report({"schema_version": "1", "run": _run(), "trials": [trial]})


def test_score_report_rejects_a_missing_regression_fixture() -> None:
    trial = _trial("rfp", 1, passed=False)
    trial["failures"] = [
        {
            "classification": "unrealistic_test_boundary",
            "disposition": "regression_fixture",
            "reference": {
                "kind": "repository_path",
                "value": "tests/does-not-exist.py",
            },
        }
    ]

    score = score_report({"schema_version": "1", "run": _run(), "trials": [trial]})

    assert (
        "rfp-1 failure 0 references a missing repository fixture: tests/does-not-exist.py"
        in score["violations"]
    )


def test_score_report_requires_classification_for_a_scored_failure() -> None:
    trials = [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]
    trials[0]["case_acceptance"][0]["passed"] = False

    score = score_report({"schema_version": "1", "run": _run(), "trials": trials})

    assert (
        "rfp-1 failed without classified dispositions and follow-up references"
        in score["violations"]
    )


def test_score_report_accepts_materialized_trial_scoped_evidence(tmp_path: Path) -> None:
    trials = [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]
    for trial in trials:
        _materialize_trial(tmp_path, trial)

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is True


def test_score_report_rejects_empty_submissions_and_shared_dummy_evidence(
    tmp_path: Path,
) -> None:
    trials = [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]
    shared = tmp_path / "evidence" / "output.txt"
    shared.parent.mkdir()
    shared.write_text("dummy\n", encoding="utf-8")
    for trial in trials:
        (tmp_path / trial["submission_path"]).mkdir(parents=True)
        for item in trial["evidence"].values():
            if item["status"] != "not_run":
                item["output_path"] = "evidence/output.txt"
        for item in trial["case_acceptance"]:
            item["evidence_path"] = "evidence/output.txt"

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is False
    assert (
        "rfp-1 first-submission artifact must be inside its submission directory"
        not in score["violations"]
    )
    assert any("missing first-submission artifact" in item for item in score["violations"])
    assert any(
        "evidence path is outside its submission evidence directory" in item
        for item in score["violations"]
    )
    assert any("evidence path is reused by multiple claims" in item for item in score["violations"])


def test_coding_repository_prompt_tool_alignment_cannot_reuse_eval_evidence(
    tmp_path: Path,
) -> None:
    trials = _trials()
    for trial in trials:
        _materialize_trial(tmp_path, trial)
    coding = next(trial for trial in trials if trial["archetype"] == "coding_repository")
    alignment = next(
        item for item in coding["case_acceptance"] if item["id"] == "prompt_tool_alignment"
    )
    alignment["evidence_path"] = coding["evidence"]["eval"]["output_path"]

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is False
    assert any("evidence path is reused by multiple claims" in item for item in score["violations"])


def test_coding_repository_prompt_tool_alignment_rejects_eval_copy_with_trailing_whitespace(
    tmp_path: Path,
) -> None:
    trials = _trials()
    for trial in trials:
        _materialize_trial(tmp_path, trial)
    coding = next(trial for trial in trials if trial["archetype"] == "coding_repository")
    alignment = next(
        item for item in coding["case_acceptance"] if item["id"] == "prompt_tool_alignment"
    )
    eval_path = tmp_path / coding["evidence"]["eval"]["output_path"]
    alignment_path = tmp_path / alignment["evidence_path"]
    alignment_path.write_bytes(eval_path.read_bytes() + b"  \n")

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is False
    assert (
        f"{coding['id']} prompt_tool_alignment evidence duplicates trajectory eval content"
        in score["violations"]
    )


def test_coding_repository_prompt_tool_alignment_rejects_case_eval_evidence_copy(
    tmp_path: Path,
) -> None:
    trials = _trials()
    for trial in trials:
        _materialize_trial(tmp_path, trial)
    coding = next(trial for trial in trials if trial["archetype"] == "coding_repository")
    by_id = {item["id"]: item for item in coding["case_acceptance"]}
    alignment_path = tmp_path / by_id["prompt_tool_alignment"]["evidence_path"]
    trajectory_path = tmp_path / by_id["trajectory_eval"]["evidence_path"]
    trajectory_output = {"status": "passed", "cases": [{"id": "review"}]}
    trajectory_path.write_text(
        json.dumps(trajectory_output, separators=(",", ":")),
        encoding="utf-8",
    )
    alignment_path.write_text(json.dumps(trajectory_output, indent=2), encoding="utf-8")

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is False
    assert (
        f"{coding['id']} prompt_tool_alignment evidence duplicates trajectory eval content"
        in score["violations"]
    )


def test_coding_repository_prompt_tool_alignment_rejects_arbitrary_text(
    tmp_path: Path,
) -> None:
    trials = _trials()
    for trial in trials:
        _materialize_trial(tmp_path, trial)
    coding = next(trial for trial in trials if trial["archetype"] == "coding_repository")
    alignment = next(
        item for item in coding["case_acceptance"] if item["id"] == "prompt_tool_alignment"
    )
    (tmp_path / alignment["evidence_path"]).write_text(
        "trajectory passed\n",
        encoding="utf-8",
    )

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is False
    assert f"{coding['id']} prompt_tool_alignment artifact is not valid JSON" in score["violations"]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("agent_name_type", "schema error"),
        ("registered_duplicate", "schema error"),
        ("workflow_surrounding_whitespace", "nonblank names without surrounding whitespace"),
        ("workflow_not_registered", "workflow tools are not registered for agent reviewer"),
        ("failed_check", "schema error"),
        ("alignment_diagnostic", "contains alignment diagnostic"),
    ],
)
def test_coding_repository_prompt_tool_alignment_validates_structured_fields(
    tmp_path: Path,
    mutation: str,
    expected: str,
) -> None:
    trials = _trials()
    for trial in trials:
        _materialize_trial(tmp_path, trial)
    coding = next(trial for trial in trials if trial["archetype"] == "coding_repository")
    alignment = next(
        item for item in coding["case_acceptance"] if item["id"] == "prompt_tool_alignment"
    )
    artifact = copy.deepcopy(_prompt_tool_alignment_artifact())
    if mutation == "agent_name_type":
        artifact["agent"]["name"] = 7
    elif mutation == "registered_duplicate":
        artifact["agent"]["registered_tool_names"] = ["read_source", "read_source"]
    elif mutation == "workflow_surrounding_whitespace":
        artifact["agent"]["workflow_tool_names"] = [" read_source"]
    elif mutation == "workflow_not_registered":
        artifact["agent"]["workflow_tool_names"] = ["missing_tool"]
    elif mutation == "failed_check":
        artifact["check"]["exit_code"] = 1
    elif mutation == "alignment_diagnostic":
        artifact["check"]["result"]["diagnostics"] = [
            {"code": "AGENT_WORKFLOW_TOOL_NOT_REGISTERED"}
        ]
    else:  # pragma: no cover - parameter list owns the cases
        raise AssertionError(mutation)
    (tmp_path / alignment["evidence_path"]).write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is False
    assert any(
        expected in violation
        for violation in score["violations"]
        if "prompt_tool_alignment" in violation
    )


def test_coding_repository_prompt_tool_alignment_rejects_symlink_to_eval_copy(
    tmp_path: Path,
) -> None:
    trials = _trials()
    for trial in trials:
        _materialize_trial(tmp_path, trial)
    coding = next(trial for trial in trials if trial["archetype"] == "coding_repository")
    by_id = {item["id"]: item for item in coding["case_acceptance"]}
    eval_path = tmp_path / coding["evidence"]["eval"]["output_path"]
    alignment_path = tmp_path / by_id["prompt_tool_alignment"]["evidence_path"]
    copied_eval = alignment_path.with_name("copied-eval.txt")
    copied_eval.write_bytes(eval_path.read_bytes())
    alignment_path.unlink()
    alignment_path.symlink_to(copied_eval)

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is False
    assert (
        f"{coding['id']} prompt_tool_alignment evidence duplicates trajectory eval content"
        in score["violations"]
    )


def test_coding_repository_requires_prompt_tool_alignment_evidence() -> None:
    trials = _trials()
    coding = next(trial for trial in trials if trial["archetype"] == "coding_repository")
    coding["case_acceptance"] = [
        item for item in coding["case_acceptance"] if item["id"] != "prompt_tool_alignment"
    ]

    score = score_report({"schema_version": "1", "run": _run(), "trials": trials})

    assert score["passed"] is False
    assert any(
        "missing case requirements: prompt_tool_alignment" in item
        for item in score["trial_failures"]
    )


def test_score_report_rejects_submission_and_evidence_reused_across_trials(
    tmp_path: Path,
) -> None:
    trials = [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]
    for trial in trials:
        _materialize_trial(tmp_path, trial)
    original = trials[0]
    for repeated in trials[1:3]:
        repeated["submission_path"] = original["submission_path"]
        repeated["submission_artifact"] = original["submission_artifact"]
        repeated["evidence"] = original["evidence"]
        repeated["case_acceptance"] = original["case_acceptance"]

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is False
    assert score["aggregate"] == {"passed": 6, "total": 9, "rate": 0.6667}
    assert score["archetypes"]["rfp"] == {"passed": 0, "total": 3, "rate": 0.0}
    assert (
        "submission path is reused across trials rfp-1, rfp-2, rfp-3: submissions/rfp-1"
        in score["violations"]
    )
    assert any(
        item.startswith("evidence path is reused across trials rfp-1, rfp-2, rfp-3:")
        for item in score["violations"]
    )


def test_score_report_rejects_submission_artifact_symlink_escape(tmp_path: Path) -> None:
    trials = [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]
    for trial in trials:
        _materialize_trial(tmp_path, trial)
    escaped = tmp_path / "shared.diff"
    escaped.write_text("diff --git a/a b/a\n", encoding="utf-8")
    artifact = tmp_path / trials[0]["submission_artifact"]
    artifact.unlink()
    artifact.symlink_to(escaped)

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is False
    assert (
        "rfp-1 first-submission artifact path escapes its required scope: "
        "submissions/rfp-1/first-submission.diff"
    ) in score["violations"]


def test_score_report_rejects_a_dummy_first_submission_diff(tmp_path: Path) -> None:
    trials = [
        _trial(archetype, number)
        for archetype in ("rfp", "research_document", "coding_repository")
        for number in range(1, 4)
    ]
    for trial in trials:
        _materialize_trial(tmp_path, trial)
    artifact = tmp_path / trials[0]["submission_artifact"]
    artifact.write_text("diff --git a/app.py b/app.py\n", encoding="utf-8")

    score = score_report(
        {"schema_version": "1", "run": _run(), "trials": trials},
        artifact_root=tmp_path,
    )

    assert score["passed"] is False
    assert (
        "rfp-1 first-submission artifact is not a non-empty diff or safe archive: "
        "submissions/rfp-1/first-submission.diff"
    ) in score["violations"]
