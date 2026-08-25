from __future__ import annotations

import asyncio

import pytest
from tests.server.test_server_eval_scenarios import (
    _AUTH_HEADERS,
    _scenario,
    _server,
    _target,
)

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    FinalOutputContainsAssertionSpec,
    RootStatusAssertionSpec,
    RunInputSpec,
)
from cayu.evals.suite_authoring import (
    EvalCaseDraftV1,
    EvalScenarioStimulusV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDocumentV1,
    EvalSuiteDraftV1,
)
from cayu.storage.evals_sqlite import SQLiteEvalStore


def _simple_case() -> EvalCaseDraftV1:
    return EvalCaseDraftV1(
        id="refund-request",
        name="Refund request",
        stimulus=EvalSimpleInputStimulusV1(
            input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Refund invoice 123."),))
        ),
        assertions=(
            RootStatusAssertionSpec(id="completed", expected="completed"),
            FinalOutputContainsAssertionSpec(id="mentions-refund", expected="refund"),
        ),
    )


def _draft(*cases: EvalCaseDraftV1) -> EvalSuiteDraftV1:
    return EvalSuiteDraftV1(
        id="refund-regressions",
        target_key="assistant.default",
        name="Refund regressions",
        cases=cases or (_simple_case(),),
    )


def test_suite_preview_save_catalog_and_download_are_target_scoped(tmp_path) -> None:
    target, _, provider = _target(tmp_path)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            assert (
                client.post(
                    "/api/evals/suites/preview",
                    json={"draft": _draft().model_dump(mode="json")},
                ).status_code
                == 401
            )
            preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": _draft().model_dump(mode="json")},
            )
            assert preview.status_code == 200
            assert preview.json()["ready"] is True
            assert preview.json()["diagnostics"] == []
            assert preview.json()["full_selection"]["mode"] == "full_suite"
            suite = EvalSuiteDocumentV1.model_validate(preview.json()["suite"])

            stale = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": "sha256:" + "0" * 64,
                    "suite": suite.model_dump(mode="json"),
                },
            )
            assert stale.status_code == 409
            assert stale.json() == {
                "detail": "Authored eval suite changed after the reviewed revision."
            }

            saved = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": suite.revision,
                    "suite": suite.model_dump(mode="json"),
                },
            )
            assert saved.status_code == 201
            assert saved.json()["entry"]["revision"] == suite.revision

            catalog = client.get("/api/evals/suites", headers=_AUTH_HEADERS)
            assert catalog.status_code == 200
            assert [item["revision"] for item in catalog.json()["items"]] == [suite.revision]
            loaded = client.get(
                f"/api/evals/suites/{suite.revision}",
                headers=_AUTH_HEADERS,
            )
            assert loaded.status_code == 200
            assert loaded.json() == suite.model_dump(mode="json")
            downloaded = client.get(
                f"/api/evals/suites/{suite.revision}/download",
                headers=_AUTH_HEADERS,
            )
            assert downloaded.status_code == 200
            assert downloaded.content.endswith(b"\n")
            assert EvalSuiteDocumentV1.model_validate_json(downloaded.content) == suite
        assert provider.requests == []
    finally:
        asyncio.run(store.close())


def test_suite_preview_reports_exact_scenario_readiness_before_save(tmp_path) -> None:
    target, _, _ = _target(tmp_path)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    scenario = _scenario()
    scenario_case = EvalCaseDraftV1(
        id="approval-flow",
        name="Approval flow",
        stimulus=EvalScenarioStimulusV1(
            scenario_id=scenario.id,
            scenario_revision=scenario.revision,
        ),
        assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
    )
    try:
        with TestClient(_server(target, store)) as client:
            unavailable = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": _draft(scenario_case).model_dump(mode="json")},
            )
            assert unavailable.status_code == 200
            assert unavailable.json()["ready"] is False
            assert unavailable.json()["diagnostics"] == [
                {
                    "code": "scenario_unavailable",
                    "case_id": "approval-flow",
                    "message": "The exact scenario revision is unavailable.",
                }
            ]
            rejected = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": unavailable.json()["suite"]["revision"],
                    "suite": unavailable.json()["suite"],
                },
            )
            assert rejected.status_code == 409
            assert rejected.json() == {"detail": "Authored eval suite is not ready to save."}

            scenario_save = client.post(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                },
            )
            assert scenario_save.status_code == 201
            ready = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": _draft(scenario_case).model_dump(mode="json")},
            )
            assert ready.status_code == 200
            assert ready.json()["ready"] is True
            assert ready.json()["diagnostics"] == []
    finally:
        asyncio.run(store.close())


def test_suite_preview_reports_an_unpublished_target_without_execution(tmp_path) -> None:
    target, _, provider = _target(tmp_path)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            response = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={
                    "draft": _draft()
                    .model_copy(update={"target_key": "assistant.unpublished"})
                    .model_dump(mode="json")
                },
            )

            assert response.status_code == 200
            assert response.json()["ready"] is False
            assert response.json()["diagnostics"] == [
                {
                    "code": "target_unavailable",
                    "case_id": None,
                    "message": "The authored suite target is not currently published.",
                }
            ]
        assert provider.requests == []
    finally:
        asyncio.run(store.close())
