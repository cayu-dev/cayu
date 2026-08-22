from __future__ import annotations

import json
from pathlib import Path

from tests.evals.eval_store_conformance import captured_result_for_corpus

from cayu import (
    AgentSpec,
    CorpusTarget,
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalPlan,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    FinalOutputEqualsAssertionSpec,
    Message,
    ModelJudgeAssertionSpec,
    ModelJudgeTarget,
    ModelStreamEvent,
    RunInputSpec,
    RunRequest,
    ScriptedModelProvider,
    TrialRequestSpec,
    captured_evaluation_result_to_json,
    eval_corpus_to_json,
    load_corpus_execution_result,
    load_eval_corpus,
)
from cayu.cli import main
from cayu.runtime.app import CayuApp


def _source() -> EvaluationSourceIdentityV1:
    return EvaluationSourceIdentityV1(
        application_release_id="captured-release",
        app_manifest_schema_version="7",
        app_manifest_fingerprint="a" * 64,
        evidence_revision="sha256:" + "e" * 64,
    )


def _corpus(*, suite_id: str = "refunds", case_id: str = "refund-approved"):
    suite = EvalSuiteSpec.create(
        id=suite_id,
        name=suite_id.title(),
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=30),
    )
    case = EvalCaseSpec.create(
        id=case_id,
        suite_id=suite_id,
        name=case_id.title(),
        source=_source(),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Review refund."),)),
        assertions=(FinalOutputEqualsAssertionSpec(id="answer", expected="Approved"),),
    )
    return EvalCorpusDocument.create(
        target_key="refund-agent",
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        suites=(suite,),
        cases=(case,),
    )


def _corpus_eval_plan(*, output: str = "Approved") -> EvalPlan:
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta(output),
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    return EvalPlan(
        corpus_target=CorpusTarget(
            key="refund-agent",
            app=app,
            request_base=RunRequest(agent_name="agent", messages=[], max_steps=1),
            bootstrap_messages=(Message.text("system", "Follow policy."),),
            application_release_id="release-2026-08-06",
        )
    )


def build_corpus_eval_plan() -> EvalPlan:
    return _corpus_eval_plan()


def build_failing_corpus_eval_plan() -> EvalPlan:
    return _corpus_eval_plan(output="Denied")


def _model_judge_eval_plan_and_corpus() -> tuple[EvalPlan, EvalCorpusDocument]:
    candidate_app = CayuApp(enable_logging=False)
    candidate_app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("Approved"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    candidate_app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    judge_app = CayuApp(enable_logging=False)
    judge_app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta('{"score": 0.9, "rationale": "correct"}'),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    judge_app.register_agent(AgentSpec(name="judge", model="judge-model"))
    judge = ModelJudgeTarget(
        key="quality-judge",
        app=judge_app,
        agent_name="judge",
    )
    target = CorpusTarget(
        key="refund-agent",
        app=candidate_app,
        request_base=RunRequest(agent_name="agent", messages=[], max_steps=1),
        application_release_id="release-2026-08-06",
        model_judges=(judge,),
    )
    suite = EvalSuiteSpec.create(
        id="refunds",
        name="Refunds",
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=30),
    )
    case = EvalCaseSpec.create(
        id="refund-quality",
        suite_id=suite.id,
        name="Refund quality",
        source=_source(),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Review refund."),)),
        assertions=(
            ModelJudgeAssertionSpec(
                id="quality",
                evaluator_key=judge.key,
                rubric="Score correctness.",
                rubric_version="quality-v1",
                threshold=0.8,
            ),
        ),
    )
    return (
        EvalPlan(corpus_target=target),
        EvalCorpusDocument.create(
            target_key=target.key,
            evidence_policy=target.evidence_policy,
            suites=(suite,),
            cases=(case,),
        ),
    )


def build_model_judge_eval_plan() -> EvalPlan:
    plan, _ = _model_judge_eval_plan_and_corpus()
    return plan


def test_eval_run_executes_downloaded_corpus_and_writes_safe_json_and_html(
    tmp_path: Path,
    monkeypatch,
) -> None:
    corpus_path = tmp_path / "corpus.json"
    result_path = tmp_path / "result.json"
    html_path = tmp_path / "result.html"
    corpus_path.write_text(eval_corpus_to_json(_corpus()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "eval",
            "run",
            f"{__name__}:build_corpus_eval_plan",
            "--corpus",
            str(corpus_path),
            "--output",
            str(result_path),
            "--html-output",
            str(html_path),
        ]
    )

    result = load_corpus_execution_result(result_path)
    assert exit_code == 0
    assert result.run.status == "passed"
    assert result.run.suite_id == "refunds"
    assert result.target.application_release_id == "release-2026-08-06"
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")
    serialized = result_path.read_text(encoding="utf-8")
    assert "session_id" not in serialized
    assert '"final_output":' not in serialized
    assert '"text": "Approved"' in serialized


def test_eval_run_cli_executes_portable_model_judge_through_corpus_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _, corpus = _model_judge_eval_plan_and_corpus()
    corpus_path = tmp_path / "model-judge-corpus.json"
    result_path = tmp_path / "model-judge-result.json"
    corpus_path.write_text(eval_corpus_to_json(corpus), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "eval",
            "run",
            f"{__name__}:build_model_judge_eval_plan",
            "--corpus",
            str(corpus_path),
            "--output",
            str(result_path),
        ]
    )

    result = load_corpus_execution_result(result_path)
    detail = result.run.cases[0].trials[0].assertions[0].detail
    assert exit_code == 0
    assert result.run.score == 0.9
    assert detail.kind == "model_judge"
    assert detail.diagnostic == "judgment_recorded"
    serialized = result_path.read_text(encoding="utf-8")
    assert "judge_output" not in serialized
    assert "rationale" not in serialized


def test_eval_report_and_compare_accept_dashboard_corpus_results_with_stable_exits(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.json"
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    report_path = tmp_path / "report.html"
    comparison_path = tmp_path / "comparison.json"
    corpus_path.write_text(eval_corpus_to_json(_corpus()), encoding="utf-8")

    assert (
        main(
            [
                "eval",
                "run",
                f"{__name__}:build_corpus_eval_plan",
                "--corpus",
                str(corpus_path),
                "--output",
                str(baseline_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "eval",
                "run",
                f"{__name__}:build_failing_corpus_eval_plan",
                "--corpus",
                str(corpus_path),
                "--output",
                str(current_path),
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "eval",
                "report",
                str(current_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "eval",
                "compare",
                str(baseline_path),
                str(current_path),
                "--json",
                "--output",
                str(comparison_path),
            ]
        )
        == 1
    )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["compatibility"]["comparable"] is True
    assert [item["kind"] for item in comparison["regressions"]] == [
        "status",
        "score",
        "status",
        "score",
    ]
    assert "Cayu Eval Comparison" not in report_path.read_text(encoding="utf-8")
    assert "Cayu Eval Report" in report_path.read_text(encoding="utf-8")


def test_eval_report_and_compare_round_trip_captured_and_fresh_results(
    tmp_path: Path,
) -> None:
    corpus = _corpus()
    corpus_path = tmp_path / "corpus.json"
    fresh_path = tmp_path / "fresh.json"
    captured_path = tmp_path / "captured.json"
    captured_json_path = tmp_path / "captured-report.json"
    captured_html_path = tmp_path / "captured-report.html"
    comparison_json_path = tmp_path / "comparison.json"
    comparison_html_path = tmp_path / "comparison.html"
    reverse_comparison_json_path = tmp_path / "reverse-comparison.json"
    corpus_path.write_text(eval_corpus_to_json(corpus), encoding="utf-8")

    assert (
        main(
            [
                "eval",
                "run",
                f"{__name__}:build_corpus_eval_plan",
                "--corpus",
                str(corpus_path),
                "--output",
                str(fresh_path),
            ]
        )
        == 0
    )
    fresh = load_corpus_execution_result(fresh_path)
    captured = captured_result_for_corpus(corpus, fresh)
    captured_path.write_text(captured_evaluation_result_to_json(captured), encoding="utf-8")

    assert (
        main(
            [
                "eval",
                "report",
                str(captured_path),
                "--json",
                "--output",
                str(captured_json_path),
            ]
        )
        == 0
    )
    assert captured_json_path.read_text(encoding="utf-8") == captured_path.read_text(
        encoding="utf-8"
    )
    assert (
        main(
            [
                "eval",
                "report",
                str(captured_path),
                "--output",
                str(captured_html_path),
            ]
        )
        == 0
    )
    assert "Cayu Captured Eval Report" in captured_html_path.read_text(encoding="utf-8")

    assert (
        main(
            [
                "eval",
                "compare",
                str(captured_path),
                str(fresh_path),
                "--json",
                "--output",
                str(comparison_json_path),
            ]
        )
        == 0
    )
    comparison = json.loads(comparison_json_path.read_text(encoding="utf-8"))
    assert comparison["compatibility"]["comparable"] is True
    assert comparison["regressions"] == []
    assert comparison["baseline"]["result_revision"] == captured.revision
    assert comparison["current"]["result_revision"] == fresh.revision

    assert (
        main(
            [
                "eval",
                "compare",
                str(captured_path),
                str(fresh_path),
                "--html",
                "--output",
                str(comparison_html_path),
            ]
        )
        == 0
    )
    assert "Cayu Eval Comparison" in comparison_html_path.read_text(encoding="utf-8")

    assert (
        main(
            [
                "eval",
                "compare",
                str(fresh_path),
                str(captured_path),
                "--json",
                "--output",
                str(reverse_comparison_json_path),
            ]
        )
        == 0
    )
    reverse_comparison = json.loads(reverse_comparison_json_path.read_text(encoding="utf-8"))
    assert reverse_comparison["compatibility"]["comparable"] is True
    assert reverse_comparison["regressions"] == []
    assert reverse_comparison["baseline"]["result_revision"] == fresh.revision
    assert reverse_comparison["current"]["result_revision"] == captured.revision


def test_eval_compare_returns_two_and_typed_reasons_for_incomparable_corpora(
    tmp_path: Path,
) -> None:
    baseline_corpus = tmp_path / "baseline-corpus.json"
    current_corpus = tmp_path / "current-corpus.json"
    baseline_result = tmp_path / "baseline-result.json"
    current_result = tmp_path / "current-result.json"
    comparison_path = tmp_path / "comparison.json"
    baseline_corpus.write_text(eval_corpus_to_json(_corpus()), encoding="utf-8")
    current_corpus.write_text(
        eval_corpus_to_json(_corpus(case_id="refund-changed")),
        encoding="utf-8",
    )
    target = f"{__name__}:build_corpus_eval_plan"

    for corpus_path, result_path in (
        (baseline_corpus, baseline_result),
        (current_corpus, current_result),
    ):
        assert (
            main(
                [
                    "eval",
                    "run",
                    target,
                    "--corpus",
                    str(corpus_path),
                    "--output",
                    str(result_path),
                ]
            )
            == 0
        )

    assert (
        main(
            [
                "eval",
                "compare",
                str(baseline_result),
                str(current_result),
                "--json",
                "--output",
                str(comparison_path),
            ]
        )
        == 2
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["compatibility"]["comparable"] is False
    assert comparison["compatibility"]["reasons"] == [
        "corpus_revision_mismatch",
        "case_contract_mismatch",
        "assertion_contract_mismatch",
    ]
    assert comparison["cases"] == []
    assert comparison["regressions"] == []


def test_eval_corpus_validate_inspect_and_atomic_merge_commands(
    tmp_path: Path,
    capsys,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    merged = tmp_path / "merged.json"
    inspection = tmp_path / "inspection.json"
    first.write_text(eval_corpus_to_json(_corpus()), encoding="utf-8")
    second.write_text(
        eval_corpus_to_json(_corpus(suite_id="accounts", case_id="account-approved")),
        encoding="utf-8",
    )

    assert main(["eval", "validate", str(first)]) == 0
    assert "Valid eval corpus sha256:" in capsys.readouterr().out
    assert (
        main(
            [
                "eval",
                "inspect",
                str(first),
                "--json",
                "--output",
                str(inspection),
            ]
        )
        == 0
    )
    assert json.loads(inspection.read_text(encoding="utf-8"))["suite_count"] == 1
    assert main(["eval", "merge", str(merged), str(first), str(second)]) == 0
    capsys.readouterr()
    assert tuple(suite.id for suite in load_eval_corpus(merged).suites) == (
        "accounts",
        "refunds",
    )


def test_eval_run_requires_explicit_suite_for_multi_suite_corpus(
    tmp_path: Path,
    capsys,
) -> None:
    first = _corpus()
    second = _corpus(suite_id="accounts", case_id="account-approved")
    corpus_path = tmp_path / "multi.json"
    from cayu import merge_eval_corpora

    corpus_path.write_text(
        eval_corpus_to_json(merge_eval_corpora((first, second))),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "eval",
            "run",
            f"{__name__}:build_corpus_eval_plan",
            "--corpus",
            str(corpus_path),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["error"]["code"] == "EVAL_COMMAND_FAILED"
    assert "--suite is required" in payload["error"]["message"]


def test_corpus_commands_reject_output_paths_that_would_overwrite_inputs(tmp_path, capsys):
    corpus_path = tmp_path / "evals.json"
    original = eval_corpus_to_json(_corpus())
    corpus_path.write_text(original, encoding="utf-8")

    assert (
        main(
            [
                "eval",
                "validate",
                str(corpus_path),
                "--output",
                str(corpus_path),
            ]
        )
        == 2
    )
    assert "must not overwrite corpus input" in capsys.readouterr().err
    assert corpus_path.read_text(encoding="utf-8") == original

    assert (
        main(
            [
                "eval",
                "run",
                f"{__name__}:build_corpus_eval_plan",
                "--corpus",
                str(corpus_path),
                "--output",
                str(corpus_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert "must not overwrite --corpus" in captured.out + captured.err
    assert corpus_path.read_text(encoding="utf-8") == original

    destination = tmp_path / "merged.json"
    assert (
        main(
            [
                "eval",
                "merge",
                str(destination),
                str(corpus_path),
                "--output",
                str(destination),
            ]
        )
        == 2
    )
    assert "must not overwrite merge destination" in capsys.readouterr().err
    assert not destination.exists()


def test_eval_run_rejects_colliding_json_and_html_outputs_before_execution(
    tmp_path,
    capsys,
):
    corpus_path = tmp_path / "evals.json"
    output_path = tmp_path / "result"
    corpus_path.write_text(eval_corpus_to_json(_corpus()), encoding="utf-8")

    assert (
        main(
            [
                "eval",
                "run",
                f"{__name__}:build_corpus_eval_plan",
                "--corpus",
                str(corpus_path),
                "--output",
                str(output_path),
                "--html-output",
                str(output_path),
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert "must use different files" in captured.out + captured.err
    assert not output_path.exists()
