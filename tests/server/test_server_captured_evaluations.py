from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.server.test_server_evals import _execution_profile_revision
from tests.server.test_server_evaluation_promotion import (
    _AUTH_HEADERS,
    _SESSION_ID,
    _authenticate,
    _seed_app,
)

from cayu.evals import (
    CapturedEvaluationCandidateV1,
    CapturedEvaluationResultV1,
    EvalCaseSpec,
    ModelJudgeAssertionSpec,
)
from cayu.evals.store import EvalResultRecord
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    _create_project_control_plane_context,
)
from cayu.runtime import default_price_book
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.invocation import InvocationOriginTrust, SessionExecutionSource
from cayu.server import DashboardConfig, ServerConfig, create_server
from cayu.server.routes import _eval_result_record_matches_document
from cayu.storage.evals_sqlite import SQLiteEvalStore


def _captured_draft(candidate: dict) -> dict:
    return {
        "expected_baseline_revision": candidate["revision"],
        "suite": {
            "id": candidate["suite"]["id"],
            "name": candidate["suite"]["name"],
            "description": candidate["suite"]["description"],
        },
        "case": {
            "id": candidate["case"]["id"],
            "suite_id": candidate["case"]["suite_id"],
            "name": candidate["case"]["name"],
            "description": candidate["case"]["description"],
            "assertions": json.loads(json.dumps(candidate["case"]["assertions"])),
        },
    }


def test_click_to_evaluate_saves_catalogs_baselines_and_exports_without_runnable_input(
    tmp_path,
) -> None:
    app = asyncio.run(_seed_app())
    store = SQLiteEvalStore(tmp_path / "cayu.db")
    context = _create_project_control_plane_context(
        project_root=Path(__file__).resolve().parents[2],
        project_id="captured-workflow",
        configured_release_id="release-current",
        eval_store=store,
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
    )
    server = create_server(
        app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
        ),
        project_context=context,
    )

    preview_url = f"/api/evals/sessions/{_SESSION_ID}/evaluation/preview"
    with TestClient(server) as client:
        readiness = client.get("/api/contract", headers=_AUTH_HEADERS).json()["capabilities"][
            "evals_readiness"
        ]
        assert readiness["captured_evaluation"] == {
            "state": "ready",
            "reason_code": None,
        }
        assert readiness["captured_result_persistence"] == {
            "state": "ready",
            "reason_code": None,
        }
        assert readiness["scenario_conversion"] == {
            "state": "ready",
            "reason_code": None,
        }

        preview = client.post(preview_url, headers=_AUTH_HEADERS, json={})
        assert preview.status_code == 200
        initial = preview.json()
        assert initial["candidate"]["case"]["input"] is None
        assert initial["captured_score"]["status"] == "passed"
        assert initial["runnable_conversion"] == {
            "available": True,
            "reason_code": None,
        }
        assert initial["scenario_conversion"]["available"] is True
        assert initial["scenario_conversion"]["diagnostics"] == []
        scenario = initial["scenario_conversion"]["scenario"]
        assert scenario["schema_version"] == 2
        assert scenario["target_key"] == initial["candidate"]["target_key"]
        assert [event["kind"] for event in scenario["events"]] == ["initial"]
        assert scenario["events"][0]["input"]["messages"][0]["content"] == [
            {"type": "text", "text": "promote this completed run"}
        ]

        target_key = initial["candidate"]["target_key"]
        assert (
            client.get(
                "/api/evals/results",
                headers=_AUTH_HEADERS,
                params={"target_key": target_key},
            ).json()["items"]
            == []
        )
        assert (
            client.get(
                "/api/evals/corpora",
                headers=_AUTH_HEADERS,
                params={"target_key": target_key},
            ).json()["items"]
            == []
        )

        draft = _captured_draft(initial["candidate"])
        draft["case"]["assertions"].append(
            {
                "id": "answer-text",
                "kind": "final_output_contains",
                "expected": "captured answer",
            }
        )
        edited = client.post(preview_url, headers=_AUTH_HEADERS, json={"draft": draft})
        assert edited.status_code == 200
        reviewed = edited.json()
        assert reviewed["captured_score"]["status"] == "passed"
        assert len(reviewed["captured_score"]["assertions"]) == 2

        # Preview is deliberately side-effect free, including after edits.
        assert (
            client.get(
                "/api/evals/results",
                headers=_AUTH_HEADERS,
                params={"target_key": target_key},
            ).json()["items"]
            == []
        )
        assert (
            client.get(
                "/api/evals/corpora",
                headers=_AUTH_HEADERS,
                params={"target_key": target_key},
            ).json()["items"]
            == []
        )

        unsupported_draft = _captured_draft(initial["candidate"])
        unsupported_draft["case"]["assertions"] = [
            {
                "id": "judge-answer",
                "kind": "model_judge",
                "evaluator_key": "quality-judge",
                "rubric": "The answer is useful.",
                "rubric_version": "v1",
                "threshold": 0.8,
                "include_transcript": False,
            }
        ]
        unsupported_preview = client.post(
            preview_url,
            headers=_AUTH_HEADERS,
            json={"draft": unsupported_draft},
        )
        assert unsupported_preview.status_code == 400
        assert unsupported_preview.json()["detail"]["code"] == "candidate_rejected"

        # Export must pass through the same scorer as preview/save. A validly
        # revisioned candidate cannot bypass captured-evidence limitations.
        baseline = CapturedEvaluationCandidateV1.model_validate_json(
            json.dumps(initial["candidate"])
        )
        unsupported_case = EvalCaseSpec.create(
            id=baseline.case.id,
            suite_id=baseline.suite.id,
            name=baseline.case.name,
            description=baseline.case.description,
            source=baseline.case.source,
            input=None,
            assertions=(
                ModelJudgeAssertionSpec(
                    id="judge-answer",
                    evaluator_key="quality-judge",
                    rubric="The answer is useful.",
                    rubric_version="v1",
                    threshold=0.8,
                    include_transcript=False,
                ),
            ),
        )
        unsupported_candidate = CapturedEvaluationCandidateV1.create(
            target_key=baseline.target_key,
            source=baseline.source,
            evidence_policy=baseline.evidence_policy,
            pricing_profile=baseline.pricing_profile,
            evidence=baseline.evidence,
            suite=baseline.suite,
            case=unsupported_case,
        )
        unsupported_export = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/export",
            headers=_AUTH_HEADERS,
            json={
                "expected_candidate_revision": unsupported_candidate.revision,
                "candidate": unsupported_candidate.model_dump(mode="json"),
            },
        )
        assert unsupported_export.status_code == 400
        assert unsupported_export.json()["detail"]["code"] == "candidate_rejected"

        candidate = reviewed["candidate"]
        save = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/save",
            headers=_AUTH_HEADERS,
            json={
                "expected_candidate_revision": candidate["revision"],
                "candidate": candidate,
            },
        )
        assert save.status_code == 201
        saved = save.json()
        result_revision = saved["record"]["revision"]
        corpus_revision = saved["record"]["corpus_revision"]
        target_key = saved["record"]["target"]["target_key"]
        saved_record = EvalResultRecord.model_validate(saved["record"])
        saved_result = CapturedEvaluationResultV1.model_validate(saved["result"])
        assert _eval_result_record_matches_document(saved_record, saved_result)
        assert not _eval_result_record_matches_document(
            saved_record.model_copy(update={"suite_id": "different-suite"}),
            saved_result,
        )
        assert not _eval_result_record_matches_document(
            saved_record.model_copy(update={"document_bytes": saved_record.document_bytes + 1}),
            saved_result,
        )

        repeated_save = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/save",
            headers=_AUTH_HEADERS,
            json={
                "expected_candidate_revision": candidate["revision"],
                "candidate": candidate,
            },
        )
        assert repeated_save.status_code == 201
        assert repeated_save.json() == saved

        catalog = client.get(
            "/api/evals/results",
            headers=_AUTH_HEADERS,
            params={"target_key": target_key},
        )
        assert catalog.status_code == 200
        assert [item["revision"] for item in catalog.json()["items"]] == [result_revision]
        assert (
            client.get(
                "/api/evals/results",
                headers=_AUTH_HEADERS,
                params={"target_key": target_key, "origin": "captured_session"},
            ).json()["items"][0]["revision"]
            == result_revision
        )
        assert (
            client.get(
                "/api/evals/results",
                headers=_AUTH_HEADERS,
                params={"target_key": target_key, "origin": "fresh_execution"},
            ).json()["items"]
            == []
        )

        detail = client.get(
            f"/api/evals/results/{result_revision}",
            headers=_AUTH_HEADERS,
        )
        assert detail.status_code == 200
        assert detail.json()["result"] == saved["result"]
        assert detail.json()["baseline"] is None

        captured_json_report = client.get(
            f"/api/evals/results/{result_revision}/report.json",
            headers=_AUTH_HEADERS,
        )
        assert captured_json_report.status_code == 200
        assert captured_json_report.content.endswith(b"\n")
        assert captured_json_report.json() == saved["result"]
        assert (
            captured_json_report.headers["content-disposition"]
            == f'attachment; filename="{result_revision.removeprefix("sha256:")}.eval-result.json"'
        )
        captured_html_report = client.get(
            f"/api/evals/results/{result_revision}/report.html",
            headers=_AUTH_HEADERS,
        )
        assert captured_html_report.status_code == 200
        assert b"Cayu Captured Eval Report" in captured_html_report.content
        assert _SESSION_ID.encode() not in captured_html_report.content

        spoofed_actor = client.post(
            f"/api/evals/results/{result_revision}/baseline",
            headers=_AUTH_HEADERS,
            json={
                "result_revision": result_revision,
                "expected_generation": 0,
                "operation_id": "sha256:" + "0" * 64,
                "actor_id": "spoofed-operator",
            },
        )
        assert spoofed_actor.status_code == 422

        baseline = client.post(
            f"/api/evals/results/{result_revision}/baseline",
            headers=_AUTH_HEADERS,
            json={
                "result_revision": result_revision,
                "expected_generation": 0,
                "operation_id": "sha256:" + "1" * 64,
            },
        )
        assert baseline.status_code == 200
        assert baseline.json()["baseline"]["updated_by"] == "eval-operator"
        assert baseline.json()["baseline"]["generation"] == 1

        exported = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/export",
            headers=_AUTH_HEADERS,
            json={
                "expected_candidate_revision": candidate["revision"],
                "candidate": candidate,
            },
        )
        assert exported.status_code == 200
        assert exported.json()["revision"] == corpus_revision
        assert exported.json()["cases"][0]["input"] is None

        rejected_run = client.post(
            "/api/evals/runs",
            headers={**_AUTH_HEADERS, "Idempotency-Key": "captured-only-run"},
            json={
                "corpus_revision": corpus_revision,
                "suite_id": saved["record"]["suite_id"],
                "expected_execution_profile_revision": _execution_profile_revision(
                    client,
                    target_key,
                ),
                "max_concurrency": 1,
            },
        )
        assert rejected_run.status_code == 409
        assert "no runnable input" in rejected_run.json()["detail"]

    asyncio.run(context.close())


def test_captured_preview_remains_usable_when_scenario_source_was_redacted(tmp_path) -> None:
    app = asyncio.run(_seed_app(secret="promote this completed run"))
    context = _create_project_control_plane_context(
        project_root=Path(__file__).resolve().parents[2],
        project_id="captured-redacted-scenario",
        configured_release_id="release-current",
        eval_store=SQLiteEvalStore(tmp_path / "cayu.db"),
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
    )
    server = create_server(
        app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
        ),
        project_context=context,
    )

    with TestClient(server) as client:
        preview = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/preview",
            headers=_AUTH_HEADERS,
            json={},
        )

        assert preview.status_code == 200
        body = preview.json()
        assert body["captured_score"]["status"] == "passed"
        assert body["candidate"]["case"]["input"] is None
        assert body["runnable_conversion"] == {
            "available": True,
            "reason_code": None,
        }
        assert body["scenario_conversion"]["available"] is False
        assert body["scenario_conversion"]["scenario"] is None
        assert {
            diagnostic["code"] for diagnostic in body["scenario_conversion"]["diagnostics"]
        } == {"source_payload_redacted"}

    asyncio.run(context.close())


def test_baseline_rejects_an_authenticated_actor_that_cannot_be_published(tmp_path) -> None:
    app = asyncio.run(_seed_app(secret="eval-operator"))
    context = _create_project_control_plane_context(
        project_root=Path(__file__).resolve().parents[2],
        project_id="captured-actor-boundary",
        configured_release_id="release-current",
        eval_store=SQLiteEvalStore(tmp_path / "cayu.db"),
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
    )
    server = create_server(
        app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
        ),
        project_context=context,
    )

    with TestClient(server) as client:
        preview = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/preview",
            headers=_AUTH_HEADERS,
            json={},
        )
        assert preview.status_code == 200
        candidate = preview.json()["candidate"]
        saved = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/save",
            headers=_AUTH_HEADERS,
            json={
                "expected_candidate_revision": candidate["revision"],
                "candidate": candidate,
            },
        )
        assert saved.status_code == 201
        result_revision = saved.json()["record"]["revision"]

        baseline = client.post(
            f"/api/evals/results/{result_revision}/baseline",
            headers=_AUTH_HEADERS,
            json={
                "result_revision": result_revision,
                "expected_generation": 0,
                "operation_id": "sha256:" + "2" * 64,
            },
        )
        assert baseline.status_code == 422
        assert baseline.json()["detail"] == (
            "The authenticated baseline actor cannot cross the public boundary."
        )

    asyncio.run(context.close())


def test_reviewed_simple_session_launches_one_fresh_trial_with_http_operator_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    app = asyncio.run(_seed_app())
    store = SQLiteEvalStore(tmp_path / "cayu.db")
    context = _create_project_control_plane_context(
        project_root=Path(__file__).resolve().parents[2],
        project_id="fresh-captured-workflow",
        configured_release_id="release-current",
        eval_store=store,
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
    )
    server = create_server(
        app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
        ),
        project_context=context,
    )

    with TestClient(server) as client:
        preview = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/preview",
            headers=_AUTH_HEADERS,
            json={},
        )
        assert preview.status_code == 200
        candidate = preview.json()["candidate"]
        request = {
            "expected_candidate_revision": candidate["revision"],
            "candidate": candidate,
            "expected_execution_profile_revision": _execution_profile_revision(
                client,
                candidate["target_key"],
            ),
            "trial_request": {"trials": 1, "timeout_seconds": 30},
            "max_concurrency": 1,
            "max_steps": 3,
            "limits": {
                "max_total_tokens": 100,
                "max_tool_calls": 2,
                "max_elapsed_seconds": 30,
                "scope": "run",
            },
        }
        launch_url = f"/api/evals/sessions/{_SESSION_ID}/evaluation/launch"
        launched = client.post(
            launch_url,
            headers={**_AUTH_HEADERS, "Idempotency-Key": "fresh-trial-one"},
            json=request,
        )
        assert launched.status_code == 202
        body = launched.json()
        assert (
            body["captured"]["record"]["corpus_revision"] == body["run"]["spec"]["corpus_revision"]
        )
        assert body["captured"]["result"]["score"]["candidate_revision"] != candidate["revision"]
        invocation = body["run"]["spec"]["invocation"]
        execution_profile = invocation.pop("execution_profile")
        admission_request_revision = invocation.pop("admission_request_revision")
        assert invocation == {
            "schema_version": 1,
            "source": "http_run",
            "origin": {
                "trust": "server_verified",
                "subject": "eval-operator",
                "tenant": None,
            },
            "max_steps": 3,
            "limits": {
                "max_input_tokens": None,
                "max_output_tokens": None,
                "max_total_tokens": 100,
                "max_tool_calls": 2,
                "max_elapsed_seconds": 30,
                "scope": "run",
            },
            "cost_budget": None,
        }
        assert execution_profile["profile_revision"].startswith("sha256:")
        assert admission_request_revision.startswith("sha256:")
        captured_result_revision = body["captured"]["record"]["revision"]
        selected_baseline = client.post(
            f"/api/evals/results/{captured_result_revision}/baseline",
            headers=_AUTH_HEADERS,
            json={
                "result_revision": captured_result_revision,
                "expected_generation": 0,
                "operation_id": "sha256:" + "3" * 64,
            },
        )
        assert selected_baseline.status_code == 200
        replay = client.post(
            launch_url,
            headers={**_AUTH_HEADERS, "Idempotency-Key": "fresh-trial-one"},
            json=request,
        )
        assert replay.status_code == 202
        assert replay.json()["run"]["spec"]["run_id"] == body["run"]["spec"]["run_id"]

        run_id = body["run"]["spec"]["run_id"]
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            run = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS).json()
            if run["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        assert run["status"] == "completed"
        assert run["result"]["status"] == "passed"
        fresh_result_revision = run["result"]["revision"]
        fresh_run_result = client.get(
            f"/api/evals/runs/{run_id}/result",
            headers=_AUTH_HEADERS,
        )
        assert fresh_run_result.status_code == 200
        assert fresh_run_result.json()["baseline"]["result_revision"] == captured_result_revision

        fresh_detail = client.get(
            f"/api/evals/results/{fresh_result_revision}",
            headers=_AUTH_HEADERS,
        )
        assert fresh_detail.status_code == 200
        assert fresh_detail.json()["baseline"]["result_revision"] == captured_result_revision
        comparison = client.post(
            "/api/evals/result-comparisons",
            headers=_AUTH_HEADERS,
            json={
                "baseline_result_revision": captured_result_revision,
                "current_result_revision": fresh_result_revision,
                "score_tolerance": 0,
            },
        )
        assert comparison.status_code == 200
        comparison_body = comparison.json()
        assert comparison_body["baseline"]["origin"] == "captured_session"
        assert comparison_body["current"]["origin"] == "fresh_execution"
        assert comparison_body["comparison"]["compatibility"]["comparable"] is True
        assert comparison_body["comparison"]["regressions"] == []

        fresh_json_report = client.get(
            f"/api/evals/results/{fresh_result_revision}/report.json",
            headers=_AUTH_HEADERS,
        )
        assert fresh_json_report.status_code == 200
        assert fresh_json_report.json()["revision"] == fresh_result_revision

        sessions = asyncio.run(app.session_store.list_sessions()).sessions
        fresh = next(session for session in sessions if session.id != _SESSION_ID)
        assert fresh.invocation.source is SessionExecutionSource.HTTP_RUN
        assert fresh.invocation.origin.trust is InvocationOriginTrust.SERVER_VERIFIED
        assert fresh.invocation.origin.subject == "eval-operator"
        assert fresh.invocation.origin.tenant is None

        too_many_trials = client.post(
            launch_url,
            headers={**_AUTH_HEADERS, "Idempotency-Key": "fresh-trial-scale"},
            json={
                **request,
                "trial_request": {"trials": 2, "timeout_seconds": 30},
            },
        )
        assert too_many_trials.status_code == 400
        assert "execution-profile trial limit" in too_many_trials.json()["detail"]

        missing_server_pricing = client.post(
            launch_url,
            headers={**_AUTH_HEADERS, "Idempotency-Key": "fresh-trial-cost"},
            json={
                **request,
                "cost_budget": {"max_estimated_cost": "1.00", "currency": "USD"},
            },
        )
        assert missing_server_pricing.status_code == 400
        assert "target or bounds" in missing_server_pricing.json()["detail"]

        def unavailable_profile(*, model: str) -> None:
            del model
            raise RuntimeError("provider temporarily unavailable")

        monkeypatch.setattr(app.get_provider(), "preflight_model_target", unavailable_profile)
        replayed_while_unavailable = client.post(
            launch_url,
            headers={**_AUTH_HEADERS, "Idempotency-Key": "fresh-trial-one"},
            json=request,
        )
        assert replayed_while_unavailable.status_code == 202
        assert (
            replayed_while_unavailable.json()["run"]["spec"]["run_id"]
            == body["run"]["spec"]["run_id"]
        )

    asyncio.run(context.close())


@pytest.mark.parametrize(
    ("priced_target", "requested_currency", "published_currencies"),
    (
        (True, "EUR", ["USD"]),
        (False, "USD", []),
    ),
)
def test_fresh_launch_rejects_incompatible_cost_budget_before_writing(
    tmp_path,
    priced_target: bool,
    requested_currency: str,
    published_currencies: list[str],
) -> None:
    app = asyncio.run(_seed_app())
    store = SQLiteEvalStore(tmp_path / "cayu.db")
    context = _create_project_control_plane_context(
        project_root=Path(__file__).resolve().parents[2],
        project_id="fresh-cost-preflight",
        configured_release_id="release-current",
        eval_store=store,
        store_backend="sqlite",
        store_source="project",
        access=ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION,
    )
    pricing = (
        PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="promotion-provider",
                    model="promotion-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                    currency="USD",
                ),
            )
        )
        if priced_target
        else default_price_book()
    )
    server = create_server(
        app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(runtime_config={"priceBook": pricing}),
        ),
        project_context=context,
    )

    with TestClient(server) as client:
        target = client.get("/api/evals/targets", headers=_AUTH_HEADERS).json()["items"][0]
        assert target["cost_budget_available"] is bool(published_currencies)
        assert target["cost_budget_currencies"] == published_currencies
        preview = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/preview",
            headers=_AUTH_HEADERS,
            json={},
        )
        assert preview.status_code == 200
        candidate = preview.json()["candidate"]

        rejected = client.post(
            f"/api/evals/sessions/{_SESSION_ID}/evaluation/launch",
            headers={**_AUTH_HEADERS, "Idempotency-Key": "fresh-cost-eur"},
            json={
                "expected_candidate_revision": candidate["revision"],
                "candidate": candidate,
                "expected_execution_profile_revision": target["execution_profile"]["revision"],
                "cost_budget": {
                    "max_estimated_cost": "1.00",
                    "currency": requested_currency,
                },
            },
        )

        assert rejected.status_code == 400
        assert "target or bounds" in rejected.json()["detail"]
        for resource in ("corpora", "results", "runs"):
            response = client.get(
                f"/api/evals/{resource}",
                headers=_AUTH_HEADERS,
                params={"target_key": target["target_key"]},
            )
            assert response.status_code == 200
            assert response.json()["items"] == []
        sessions = asyncio.run(app.session_store.list_sessions()).sessions
        assert [session.id for session in sessions] == [_SESSION_ID]

    asyncio.run(context.close())
