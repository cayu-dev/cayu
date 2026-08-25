from __future__ import annotations

import asyncio
import time

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

from cayu import ModelStreamEvent, ScriptedModelProvider
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


def test_authored_suite_full_and_subset_launch_use_existing_durable_runners(tmp_path) -> None:
    provider = ScriptedModelProvider(
        [
            (
                ModelStreamEvent.text_delta("refund current scenario result"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
            (
                ModelStreamEvent.text_delta("refund current scenario result"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        ]
    )
    target, _, _ = _target(tmp_path, provider)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    scenario = _scenario()
    scenario_case = EvalCaseDraftV1(
        id="retained-scenario",
        name="Retained scenario",
        stimulus=EvalScenarioStimulusV1(
            scenario_id=scenario.id,
            scenario_revision=scenario.revision,
        ),
        assertions=(
            RootStatusAssertionSpec(id="completed", expected="completed"),
            FinalOutputContainsAssertionSpec(
                id="scenario-result",
                expected="current scenario result",
            ),
        ),
    )
    try:
        with TestClient(_server(target, store)) as client:
            saved_scenario = client.post(
                "/api/evals/scenarios",
                headers=_AUTH_HEADERS,
                json={
                    "expected_scenario_revision": scenario.revision,
                    "scenario": scenario.model_dump(mode="json"),
                    "settings": {"trials": 1, "max_concurrency": 1},
                },
            )
            assert saved_scenario.status_code == 201
            suite_preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": _draft(_simple_case(), scenario_case).model_dump(mode="json")},
            )
            assert suite_preview.status_code == 200
            suite = EvalSuiteDocumentV1.model_validate(suite_preview.json()["suite"])
            saved_suite = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": suite.revision,
                    "suite": suite.model_dump(mode="json"),
                },
            )
            assert saved_suite.status_code == 201

            assert (
                client.post(
                    f"/api/evals/suites/{suite.revision}/runs/preview",
                    json={},
                ).status_code
                == 401
            )
            subset = client.post(
                f"/api/evals/suites/{suite.revision}/runs/preview",
                headers=_AUTH_HEADERS,
                json={"case_ids": ["refund-request"]},
            )
            assert subset.status_code == 200
            assert subset.json()["ready"] is True
            assert subset.json()["selection"]["mode"] == "subset"
            assert subset.json()["launches"] == [
                {
                    "kind": "simple_input",
                    "case_ids": ["refund-request"],
                    "scenario_revision": None,
                }
            ]

            full = client.post(
                f"/api/evals/suites/{suite.revision}/runs/preview",
                headers=_AUTH_HEADERS,
                json={},
            )
            assert full.status_code == 200
            assert full.json()["ready"] is True
            assert full.json()["selection"]["mode"] == "full_suite"
            assert [item["kind"] for item in full.json()["launches"]] == [
                "simple_input",
                "scenario",
            ]

            launched = client.post(
                f"/api/evals/suites/{suite.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "authored-suite-full-1"},
                json={},
            )
            assert launched.status_code == 202
            body = launched.json()
            assert body["selection"]["revision"] == full.json()["selection"]["revision"]
            assert [item["kind"] for item in body["runs"]] == ["simple_input", "scenario"]
            for admitted in body["runs"]:
                invocation = admitted["run"]["spec"]["invocation"]
                assert invocation["authored_suite_revision"] == suite.revision
                assert (
                    invocation["authored_suite_selection_revision"]
                    == full.json()["selection"]["revision"]
                )
            scenario_invocation = body["runs"][1]["run"]["spec"]["invocation"]["scenario"]
            assert scenario_invocation["scenario_revision"] == scenario.revision
            assert scenario_invocation["authored_suite_revision"] == suite.revision
            assert scenario_invocation["authored_case_revision"] == next(
                case.revision for case in suite.cases if case.id == "retained-scenario"
            )

            replayed = client.post(
                f"/api/evals/suites/{suite.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "authored-suite-full-1"},
                json={},
            )
            assert replayed.status_code == 202
            assert [item["run"]["spec"]["run_id"] for item in replayed.json()["runs"]] == [
                item["run"]["spec"]["run_id"] for item in body["runs"]
            ]
            changed_selection = client.post(
                f"/api/evals/suites/{suite.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "authored-suite-full-1"},
                json={"case_ids": ["refund-request"]},
            )
            assert changed_selection.status_code == 409
            assert changed_selection.json() == {
                "detail": "Idempotency-Key is already bound to another eval run request."
            }

            terminal_runs = {}
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                for item in body["runs"]:
                    run_id = item["run"]["spec"]["run_id"]
                    current = client.get(
                        f"/api/evals/runs/{run_id}",
                        headers=_AUTH_HEADERS,
                    )
                    assert current.status_code == 200
                    if current.json()["status"] in {"completed", "failed", "cancelled"}:
                        terminal_runs[run_id] = current.json()
                if len(terminal_runs) == len(body["runs"]):
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Authored suite runs did not terminalize.")

            assert {run["status"] for run in terminal_runs.values()} == {"completed"}
            for item in body["runs"]:
                run_id = item["run"]["spec"]["run_id"]
                result = client.get(
                    f"/api/evals/runs/{run_id}/result",
                    headers=_AUTH_HEADERS,
                )
                assert result.status_code == 200
                assert result.json()["result"]["run"]["status"] == "passed"
                assert [
                    case["case_id"] for case in result.json()["result"]["run"]["cases"]
                ] == item["case_ids"]
            scenario_run_id = body["runs"][1]["run"]["spec"]["run_id"]
            assert (
                terminal_runs[scenario_run_id]["scenario_progress"]["trials"][0]["phase"]
                == "completed"
            )

            first_spec = body["runs"][0]["run"]["spec"]
            independent = client.post(
                "/api/evals/runs",
                headers={
                    **_AUTH_HEADERS,
                    "Idempotency-Key": "authored-suite-full-1",
                },
                json={
                    "corpus_revision": first_spec["corpus_revision"],
                    "suite_id": first_spec["suite_id"],
                    "max_concurrency": 1,
                },
            )
            assert independent.status_code == 202
            assert independent.json()["spec"]["run_id"] not in {
                item["run"]["spec"]["run_id"] for item in body["runs"]
            }
    finally:
        asyncio.run(store.close())
