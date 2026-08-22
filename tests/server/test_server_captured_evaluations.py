from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.server.test_server_evaluation_promotion import (
    _AUTH_HEADERS,
    _SESSION_ID,
    _authenticate,
    _seed_app,
)

from cayu.evals import (
    CapturedEvaluationCandidateV1,
    EvalCaseSpec,
    ModelJudgeAssertionSpec,
)
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    _create_project_control_plane_context,
)
from cayu.server import DashboardConfig, ServerConfig, create_server
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

        preview = client.post(preview_url, headers=_AUTH_HEADERS, json={})
        assert preview.status_code == 200
        initial = preview.json()
        assert initial["candidate"]["case"]["input"] is None
        assert initial["captured_score"]["status"] == "passed"
        assert initial["runnable_conversion"] == {
            "available": True,
            "reason_code": None,
        }

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
                "max_concurrency": 1,
            },
        )
        assert rejected_run.status_code == 409
        assert "no runnable input" in rejected_run.json()["detail"]

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
