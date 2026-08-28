from __future__ import annotations

import asyncio

import pytest
from tests.evals.test_structured_model_judge import _judgment, _rubric, _target
from tests.server.test_server_eval_scenarios import _AUTH_HEADERS, _server

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import (
    AgentSpec,
    CayuApp,
    CorpusUserMessageSpec,
    EvalCaseDraftV2,
    EvalJudgeCalibrationCriterionLabelV1,
    EvalJudgeCalibrationDraftV1,
    EvalJudgeEvidenceSelectionV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDraftV2,
    ModelJudgeTarget,
    ModelStreamEvent,
    PublicJudgeReferenceDraftV1,
    RunInputSpec,
    ScriptedModelProvider,
    StructuredModelJudgeAssertionDraftV1,
    StructuredModelJudgeAssertionSpec,
    StructuredRubricDraftV1,
    model_judge_profile,
)
from cayu.storage.evals_sqlite import SQLiteEvalStore


def _judge_with_trials(count: int) -> tuple[ModelJudgeTarget, ScriptedModelProvider]:
    script = (
        ModelStreamEvent.text_delta(_judgment()),
        ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            }
        ),
    )
    provider = ScriptedModelProvider([script for _ in range(count)])
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="judge", model="judge-model"))
    return (
        ModelJudgeTarget(
            key="quality-judge",
            label="Quality judge",
            app=app,
            agent_name="judge",
        ),
        provider,
    )


def _draft(judge: ModelJudgeTarget, *, trials: int = 2) -> EvalJudgeCalibrationDraftV1:
    profile = model_judge_profile(judge)
    return EvalJudgeCalibrationDraftV1(
        id="known-refund-answer",
        target_key="refund-agent",
        assertion=StructuredModelJudgeAssertionSpec(
            id="answer-quality",
            judge_profile_key=profile.key,
            judge_profile_revision=profile.revision,
            rubric=_rubric(),
            threshold="0.6",
            evidence=EvalJudgeEvidenceSelectionV1(),
        ),
        evidence_source_id="reviewed-refund-fixture",
        task="Can I get a refund?",
        final_output="Refunds are available within 30 days.",
        human_criteria=(
            EvalJudgeCalibrationCriterionLabelV1(
                criterion_id="correctness",
                score="1",
            ),
            EvalJudgeCalibrationCriterionLabelV1(
                criterion_id="usefulness",
                score="0.5",
            ),
        ),
        trials=trials,
    )


def test_preview_and_run_calibrate_fixed_evidence_without_candidate_execution(tmp_path) -> None:
    judge, judge_provider = _judge_with_trials(2)
    target, candidate_provider = _target(judge)
    path = tmp_path / "evals.db"
    store = SQLiteEvalStore(path)
    draft = _draft(judge)
    try:
        with TestClient(_server(target, store)) as client:
            assert (
                client.post(
                    "/api/evals/judge-calibrations/preview",
                    json={"draft": draft.model_dump(mode="json")},
                ).status_code
                == 401
            )
            preview = client.post(
                "/api/evals/judge-calibrations/preview",
                headers=_AUTH_HEADERS,
                json={"draft": draft.model_dump(mode="json")},
            )
            assert preview.status_code == 200
            previewed = preview.json()
            assert previewed["ready"] is True
            assert previewed["diagnostics"] == []
            assert previewed["work"]["judge_calls"] == 2
            assert previewed["candidate_route_relation"] == "independent_model"
            assert previewed["definition"]["evidence"]["provenance"] == {
                "schema_version": 1,
                "kind": "operator_supplied",
                "source_id": "reviewed-refund-fixture",
            }
            assert judge_provider.requests == []
            assert candidate_provider.requests == []

            run = client.post(
                "/api/evals/judge-calibrations",
                headers=_AUTH_HEADERS,
                json={
                    "run_id": "calibration-fixed-answer",
                    "expected_definition_revision": previewed["definition"]["revision"],
                    "definition": previewed["definition"],
                },
            )
            assert run.status_code == 201
            report = run.json()["report"]
            assert len(report["trials"]) == 2
            assert all(item["pass_agreement"] is True for item in report["trials"])
            assert len(judge_provider.requests) == 2
            assert candidate_provider.requests == []

            loaded = client.get(
                f"/api/evals/judge-calibrations/{report['revision']}",
                headers=_AUTH_HEADERS,
            )
            assert loaded.status_code == 200
            assert loaded.json() == report

            changed_draft = draft.model_copy(
                update={"final_output": "Refunds are never available."}
            )
            changed_preview = client.post(
                "/api/evals/judge-calibrations/preview",
                headers=_AUTH_HEADERS,
                json={"draft": changed_draft.model_dump(mode="json")},
            )
            assert changed_preview.status_code == 200
            changed_definition = changed_preview.json()["definition"]
            conflict = client.post(
                "/api/evals/judge-calibrations",
                headers=_AUTH_HEADERS,
                json={
                    "run_id": "calibration-fixed-answer",
                    "expected_definition_revision": changed_definition["revision"],
                    "definition": changed_definition,
                },
            )
            assert conflict.status_code == 409
            assert conflict.json() == {
                "detail": "Judge calibration run ID is bound to different reviewed input."
            }
            assert len(judge_provider.requests) == 2
            assert candidate_provider.requests == []

        reopened = SQLiteEvalStore(path)
        try:
            with TestClient(_server(target, reopened)) as restarted_client:
                recovered = restarted_client.post(
                    "/api/evals/judge-calibrations",
                    headers=_AUTH_HEADERS,
                    json={
                        "run_id": "calibration-fixed-answer",
                        "expected_definition_revision": previewed["definition"]["revision"],
                        "definition": previewed["definition"],
                    },
                )
                assert recovered.status_code == 201
                assert recovered.json()["report"] == report
                assert len(judge_provider.requests) == 2
                assert candidate_provider.requests == []
        finally:
            asyncio.run(reopened.close())
    finally:
        asyncio.run(store.close())


def test_structured_suite_preview_save_and_reload_share_the_server_compiled_contract(
    tmp_path,
) -> None:
    judge, judge_provider = _judge_with_trials(1)
    target, candidate_provider = _target(judge)
    profile = model_judge_profile(judge)
    assertion = StructuredModelJudgeAssertionDraftV1(
        id="answer-quality",
        judge_profile_key=profile.key,
        judge_profile_revision=profile.revision,
        rubric=StructuredRubricDraftV1.from_rubric(_rubric()),
        reference=PublicJudgeReferenceDraftV1(
            id="refund-policy",
            expected_answer="Refunds are available within 30 days.",
        ),
        threshold="0.6",
    )
    draft = EvalSuiteDraftV2(
        id="refund-quality",
        target_key=target.key,
        name="Refund quality",
        cases=(
            EvalCaseDraftV2(
                id="refund-answer",
                name="Refund answer",
                stimulus=EvalSimpleInputStimulusV1(
                    input=RunInputSpec(
                        messages=(CorpusUserMessageSpec(text="Can I get a refund?"),)
                    )
                ),
                assertions=(assertion,),
            ),
        ),
    )
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": draft.model_dump(mode="json")},
            )
            assert preview.status_code == 200
            body = preview.json()
            assert body["ready"] is True
            compiled = body["suite"]["cases"][0]["assertions"][0]
            assert compiled["rubric"]["revision"].startswith("sha256:")
            assert compiled["reference"]["revision"].startswith("sha256:")

            saved = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": body["suite"]["revision"],
                    "suite": body["suite"],
                },
            )
            assert saved.status_code == 201
            assert saved.json()["suite"] == body["suite"]

            loaded = client.get(
                f"/api/evals/suites/{body['suite']['revision']}",
                headers=_AUTH_HEADERS,
            )
            assert loaded.status_code == 200
            assert loaded.json() == body["suite"]
        assert judge_provider.requests == []
        assert candidate_provider.requests == []
    finally:
        asyncio.run(store.close())
