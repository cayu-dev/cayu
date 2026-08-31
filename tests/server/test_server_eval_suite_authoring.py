from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import pytest
from tests.server.test_server_eval_scenarios import (
    _AUTH_HEADERS,
    _authenticate,
    _scenario,
    _server,
    _target,
)

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu import (
    EvalExecutionProfilePolicyV1,
    ModelPrice,
    ModelStreamEvent,
    PriceBook,
    ScriptedModelProvider,
)
from cayu.evals.corpus import (
    ArtifactAssertionSpec,
    CorpusUserMessageSpec,
    EvaluationEvidencePolicySpec,
    FinalOutputContainsAssertionSpec,
    ProcessEventAssertionSpec,
    ProcessEventsInOrderAssertionSpec,
    RootStatusAssertionSpec,
    RunInputSpec,
    ToolArgumentsContainAssertionSpec,
    ToolResultContainsAssertionSpec,
)
from cayu.evals.suite_authoring import (
    EvalCaseDraftV1,
    EvalCaseDraftV2,
    EvalScenarioStimulusV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDocumentV1,
    EvalSuiteDocumentV3,
    EvalSuiteDraftV1,
    EvalSuiteDraftV3,
    EvalSuiteTrialRequestDraftV3,
)
from cayu.server import DashboardConfig, EvalsConfig, ServerConfig, create_server
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
            ProcessEventAssertionSpec(
                id="started-once",
                event="session_started",
                min_count=1,
                max_count=1,
            ),
            ProcessEventsInOrderAssertionSpec(
                id="session-lifecycle",
                events=("session_started", "session_completed"),
            ),
        ),
    )


def _draft(*cases: EvalCaseDraftV1) -> EvalSuiteDraftV1:
    return EvalSuiteDraftV1(
        id="refund-regressions",
        target_key="assistant.default",
        name="Refund regressions",
        cases=cases or (_simple_case(),),
    )


def test_v3_authored_suite_executes_all_trials_and_applies_pass_threshold(tmp_path) -> None:
    provider = ScriptedModelProvider(
        [
            (
                ModelStreamEvent.text_delta("refund accepted"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
            (
                ModelStreamEvent.text_delta("request denied"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
            (
                ModelStreamEvent.text_delta("refund confirmed"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        ]
    )
    target, _, _ = _target(tmp_path, provider)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    draft = EvalSuiteDraftV3(
        id="reliable-refunds",
        target_key=target.key,
        name="Reliable refunds",
        trial_request=EvalSuiteTrialRequestDraftV3(
            trials=3,
            minimum_passed_trials=2,
            max_concurrency=2,
            timeout_seconds=30,
        ),
        cases=(EvalCaseDraftV2.model_validate(_simple_case().model_dump(mode="python")),),
    )
    server = create_server(
        target.app,
        config=ServerConfig.protected(
            _authenticate,
            dashboard=DashboardConfig(enabled=False),
            evals=EvalsConfig(
                target=target,
                store=store,
                execution_profile_policy=EvalExecutionProfilePolicyV1(
                    reset_strategy="application_managed",
                    isolation_revision="sha256:" + "a" * 64,
                    max_trials=3,
                    max_concurrency=2,
                ),
                poll_interval_seconds=0.02,
                lease_seconds=5,
                shutdown_grace_seconds=2,
            ),
        ),
    )
    try:
        with TestClient(server) as client:
            preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": draft.model_dump(mode="json")},
            )
            assert preview.status_code == 200
            suite = preview.json()["suite"]
            assert suite["schema_version"] == 3
            policy = suite["suite"]["trial_request"]["trial_policy"]
            assert policy["minimum_passed_trials"] == 2

            saved = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": suite["revision"],
                    "suite": suite,
                },
            )
            assert saved.status_code == 201
            launch_preview = client.post(
                f"/api/evals/suites/{suite['revision']}/runs/preview",
                headers=_AUTH_HEADERS,
                json={},
            )
            assert launch_preview.status_code == 200
            reviewed = launch_preview.json()
            assert reviewed["ready"] is True
            assert reviewed["exposure"]["candidate_trials"] == 3
            assert reviewed["exposure"]["max_concurrency"] == 2
            assert reviewed["exposure"]["maximum_candidate_model_steps"] == 24
            assert reviewed["exposure"]["maximum_candidate_total_tokens"] is None

            stale_exposure = client.post(
                f"/api/evals/suites/{suite['revision']}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "reliable-refunds-stale"},
                json={
                    "expected_exposure_revision": "sha256:" + "f" * 64,
                    "expected_execution_profiles": [
                        {
                            "case_ids": item["case_ids"],
                            "execution_profile_revision": item["execution_profile_revision"],
                        }
                        for item in reviewed["launches"]
                    ],
                },
            )
            assert stale_exposure.status_code == 409
            assert "exposure changed" in stale_exposure.json()["detail"]

            launched = client.post(
                f"/api/evals/suites/{suite['revision']}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "reliable-refunds-1"},
                json={
                    "expected_exposure_revision": reviewed["exposure"]["revision"],
                    "expected_execution_profiles": [
                        {
                            "case_ids": item["case_ids"],
                            "execution_profile_revision": item["execution_profile_revision"],
                        }
                        for item in reviewed["launches"]
                    ],
                },
            )
            assert launched.status_code == 202
            run_id = launched.json()["runs"][0]["run"]["spec"]["run_id"]

            deadline = time.monotonic() + 5
            run = None
            while time.monotonic() < deadline:
                response = client.get(f"/api/evals/runs/{run_id}", headers=_AUTH_HEADERS)
                assert response.status_code == 200
                run = response.json()
                if run["status"] in {"completed", "error", "cancelled"}:
                    break
                time.sleep(0.02)
            assert run is not None
            assert run["status"] == "completed"
            result = client.get(
                f"/api/evals/runs/{run_id}/result",
                headers=_AUTH_HEADERS,
            )
            assert result.status_code == 200
            published = result.json()["result"]["run"]
            assert [trial["status"] for trial in published["cases"][0]["trials"]] == [
                "passed",
                "failed",
                "passed",
            ]
            assert published["cases"][0]["status"] == "passed"
            assert published["cases"][0]["reliability"]["passed_trials"] == 2
            assert published["accepted_exposure"]["revision"] == reviewed["exposure"]["revision"]
    finally:
        asyncio.run(store.close())


def test_authored_simple_launch_binds_a_reviewed_candidate_cost_budget(tmp_path) -> None:
    target, _, _ = _target(tmp_path)
    target = target.model_copy(
        update={
            "price_book": PriceBook(
                price_book_version="authored-suite-test-v1",
                generated_at="2026-08-31T00:00:00Z",
                prices=(
                    ModelPrice.fixed(
                        provider_name="scripted",
                        model="scenario-model",
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("2"),
                    ),
                ),
            )
        }
    )
    store = SQLiteEvalStore(tmp_path / "evals.db")
    try:
        with TestClient(_server(target, store)) as client:
            preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": _draft().model_dump(mode="json")},
            )
            assert preview.status_code == 200
            suite = preview.json()["suite"]
            saved = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": suite["revision"],
                    "suite": suite,
                },
            )
            assert saved.status_code == 201
            budget = {"max_estimated_cost": "0.25", "currency": "USD"}
            reviewed_response = client.post(
                f"/api/evals/suites/{suite['revision']}/runs/preview",
                headers=_AUTH_HEADERS,
                json={"cost_budget": budget},
            )
            assert reviewed_response.status_code == 200
            reviewed = reviewed_response.json()
            assert reviewed["ready"] is True
            assert reviewed["exposure"]["execution_profiles"][0]["candidate_cost_budget"] == {
                "amount": "0.25",
                "currency": "USD",
            }
            assert reviewed["exposure"]["candidate_cost"]["unavailable_reason"] == (
                "candidate_cost_not_hard_bounded"
            )

            launch_body = {
                "cost_budget": budget,
                "expected_exposure_revision": reviewed["exposure"]["revision"],
                "expected_execution_profiles": [
                    {
                        "case_ids": item["case_ids"],
                        "execution_profile_revision": item["execution_profile_revision"],
                    }
                    for item in reviewed["launches"]
                ],
            }
            changed = client.post(
                f"/api/evals/suites/{suite['revision']}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "changed-authored-cost"},
                json={
                    **launch_body,
                    "cost_budget": {"max_estimated_cost": "0.50", "currency": "USD"},
                },
            )
            assert changed.status_code == 409
            assert "exposure changed" in changed.json()["detail"]

            launched = client.post(
                f"/api/evals/suites/{suite['revision']}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "reviewed-authored-cost"},
                json=launch_body,
            )
            assert launched.status_code == 202
            invocation = launched.json()["runs"][0]["run"]["spec"]["invocation"]
            assert invocation["cost_budget"] == budget
    finally:
        asyncio.run(store.close())


def test_authored_suite_run_preview_rejects_scale_above_current_profile(tmp_path) -> None:
    target, _, _ = _target(tmp_path)
    store = SQLiteEvalStore(tmp_path / "evals.db")
    draft = EvalSuiteDraftV3(
        id="repeated-without-isolation",
        target_key=target.key,
        name="Repeated without isolation",
        trial_request=EvalSuiteTrialRequestDraftV3(
            trials=2,
            minimum_passed_trials=1,
            max_concurrency=1,
            timeout_seconds=30,
        ),
        cases=(EvalCaseDraftV2.model_validate(_simple_case().model_dump(mode="python")),),
    )
    try:
        with TestClient(_server(target, store)) as client:
            preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": draft.model_dump(mode="json")},
            )
            assert preview.status_code == 200
            suite = preview.json()["suite"]
            saved = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": suite["revision"],
                    "suite": suite,
                },
            )
            assert saved.status_code == 201

            launch_preview = client.post(
                f"/api/evals/suites/{suite['revision']}/runs/preview",
                headers=_AUTH_HEADERS,
                json={},
            )
            assert launch_preview.status_code == 200
            assert launch_preview.json()["ready"] is False
            assert launch_preview.json()["exposure"] is None
            assert [item["code"] for item in launch_preview.json()["diagnostics"]] == [
                "trial_policy_exceeds_execution_profile"
            ]
    finally:
        asyncio.run(store.close())


def test_authored_suite_run_preview_explains_missing_tool_evidence_authority(tmp_path) -> None:
    target, _, _ = _target(tmp_path)
    target = target.model_copy(
        update={
            "evidence_policy": EvaluationEvidencePolicySpec.create(include_tool_arguments=False)
        }
    )
    store = SQLiteEvalStore(tmp_path / "evals.db")
    result_case = _simple_case().model_copy(
        update={
            "assertions": (
                RootStatusAssertionSpec(id="completed", expected="completed"),
                ToolArgumentsContainAssertionSpec(
                    id="lookup-arguments",
                    tool_name="lookup",
                    expected_subset={"query": "cayu"},
                ),
                ToolResultContainsAssertionSpec(
                    id="lookup-result",
                    tool_name="lookup",
                    expected_subset={"structured": {"status": "ok"}},
                ),
            )
        }
    )
    try:
        with TestClient(_server(target, store)) as client:
            preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": _draft(result_case).model_dump(mode="json")},
            )
            assert preview.status_code == 200
            suite = EvalSuiteDocumentV1.model_validate(preview.json()["suite"])
            saved = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": suite.revision,
                    "suite": suite.model_dump(mode="json"),
                },
            )
            assert saved.status_code == 201

            launch_preview = client.post(
                f"/api/evals/suites/{suite.revision}/runs/preview",
                headers=_AUTH_HEADERS,
                json={},
            )

            assert launch_preview.status_code == 200
            assert launch_preview.json()["ready"] is False
            assert launch_preview.json()["diagnostics"] == [
                {
                    "code": "tool_result_evidence_unavailable",
                    "case_id": result_case.id,
                    "message": (
                        "This case requires retained public-safe tool results, but the selected "
                        "target does not publish result evidence. Choose a target profile with "
                        "result retention or remove the result assertion."
                    ),
                },
                {
                    "code": "tool_argument_evidence_unavailable",
                    "case_id": result_case.id,
                    "message": (
                        "This case requires public tool arguments, but the selected target does "
                        "not publish argument evidence. Choose a compatible target profile or "
                        "remove the argument assertion."
                    ),
                },
            ]
    finally:
        asyncio.run(store.close())


def test_authored_suite_run_preview_explains_missing_artifact_text_authority(tmp_path) -> None:
    target, _, _ = _target(tmp_path)
    target = target.model_copy(update={"evidence_policy": EvaluationEvidencePolicySpec.standard()})
    store = SQLiteEvalStore(tmp_path / "evals.db")
    artifact_case = _simple_case().model_copy(
        update={
            "assertions": (
                RootStatusAssertionSpec(id="completed", expected="completed"),
                ArtifactAssertionSpec(
                    id="artifact-text",
                    filename="report.json",
                    content_type="application/json",
                    text_contains='"status":"ready"',
                ),
            )
        }
    )
    try:
        with TestClient(_server(target, store)) as client:
            preview = client.post(
                "/api/evals/suites/preview",
                headers=_AUTH_HEADERS,
                json={"draft": _draft(artifact_case).model_dump(mode="json")},
            )
            assert preview.status_code == 200
            suite = EvalSuiteDocumentV1.model_validate(preview.json()["suite"])
            saved = client.post(
                "/api/evals/suites",
                headers=_AUTH_HEADERS,
                json={
                    "expected_suite_revision": suite.revision,
                    "suite": suite.model_dump(mode="json"),
                },
            )
            assert saved.status_code == 201

            launch_preview = client.post(
                f"/api/evals/suites/{suite.revision}/runs/preview",
                headers=_AUTH_HEADERS,
                json={},
            )

            assert launch_preview.status_code == 200
            assert launch_preview.json()["ready"] is False
            assert launch_preview.json()["diagnostics"] == [
                {
                    "code": "artifact_text_evidence_unavailable",
                    "case_id": artifact_case.id,
                    "message": (
                        "This case requires retained public-safe artifact text, but the selected "
                        "target does not publish artifact text evidence. Choose a compatible "
                        "target profile or remove the text expectation."
                    ),
                }
            ]
    finally:
        asyncio.run(store.close())


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


def test_authored_suite_full_and_subset_launch_use_existing_durable_runners(
    tmp_path,
    monkeypatch,
) -> None:
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
        server = create_server(
            target.app,
            config=ServerConfig.protected(
                _authenticate,
                dashboard=DashboardConfig(enabled=False),
                evals=EvalsConfig(
                    target=target,
                    store=store,
                    execution_profile_policy=EvalExecutionProfilePolicyV1(
                        reset_strategy="application_managed",
                        isolation_revision="sha256:" + "a" * 64,
                        max_concurrency=2,
                    ),
                    poll_interval_seconds=0.02,
                    lease_seconds=5,
                    shutdown_grace_seconds=2,
                ),
            ),
        )
        with TestClient(server) as client:
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
                json={
                    "draft": EvalSuiteDraftV3(
                        id="refund-regressions",
                        target_key="assistant.default",
                        name="Refund regressions",
                        trial_request=EvalSuiteTrialRequestDraftV3(
                            trials=1,
                            minimum_passed_trials=1,
                            max_concurrency=2,
                            timeout_seconds=300,
                        ),
                        cases=(
                            EvalCaseDraftV2.model_validate(
                                _simple_case().model_dump(mode="python")
                            ),
                            EvalCaseDraftV2.model_validate(scenario_case.model_dump(mode="python")),
                        ),
                    ).model_dump(mode="json")
                },
            )
            assert suite_preview.status_code == 200
            suite = EvalSuiteDocumentV3.model_validate(suite_preview.json()["suite"])
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
            assert len(subset.json()["launches"]) == 1
            subset_launch = subset.json()["launches"][0]
            assert subset_launch["kind"] == "simple_input"
            assert subset_launch["case_ids"] == ["refund-request"]
            assert subset_launch["scenario_revision"] is None
            assert subset_launch["execution_profile_revision"].startswith("sha256:")

            full = client.post(
                f"/api/evals/suites/{suite.revision}/runs/preview",
                headers=_AUTH_HEADERS,
                json={},
            )
            assert full.status_code == 200
            assert full.json()["ready"] is True
            assert full.json()["exposure"]["candidate_trials"] == 2
            assert full.json()["exposure"]["judge_evaluations"] == 0
            assert full.json()["exposure"]["max_concurrency"] == 2
            assert full.json()["exposure"]["candidate_cost"] == {
                "state": "unavailable",
                "totals": [],
                "unavailable_reason": "no_candidate_cost_ceiling",
                "pricing_profile_fingerprints": [],
            }
            assert full.json()["selection"]["mode"] == "full_suite"
            assert [item["kind"] for item in full.json()["launches"]] == [
                "simple_input",
                "scenario",
            ]
            launch_body = {
                "expected_exposure_revision": full.json()["exposure"]["revision"],
                "expected_execution_profiles": [
                    {
                        "case_ids": item["case_ids"],
                        "execution_profile_revision": item["execution_profile_revision"],
                    }
                    for item in full.json()["launches"]
                ],
            }

            launched = client.post(
                f"/api/evals/suites/{suite.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "authored-suite-full-1"},
                json=launch_body,
            )
            assert launched.status_code == 202
            body = launched.json()
            assert body["selection"]["revision"] == full.json()["selection"]["revision"]
            assert [item["kind"] for item in body["runs"]] == ["simple_input", "scenario"]
            launch_revisions = set()
            launch_lanes = set()
            for admitted in body["runs"]:
                invocation = admitted["run"]["spec"]["invocation"]
                assert invocation["authored_suite_revision"] == suite.revision
                assert (
                    invocation["authored_suite_selection_revision"]
                    == full.json()["selection"]["revision"]
                )
                launch_revisions.add(invocation["authored_suite_launch_revision"])
                launch_lanes.add(invocation["authored_suite_launch_lane"])
            assert len(launch_revisions) == 1
            assert next(iter(launch_revisions)).startswith("sha256:")
            assert launch_lanes == {0, 1}
            assert {admitted["run"]["spec"]["max_concurrency"] for admitted in body["runs"]} == {1}
            scenario_invocation = body["runs"][1]["run"]["spec"]["invocation"]["scenario"]
            assert scenario_invocation["scenario_revision"] == scenario.revision
            assert scenario_invocation["authored_suite_revision"] == suite.revision
            assert scenario_invocation["authored_case_revision"] == next(
                case.revision for case in suite.cases if case.id == "retained-scenario"
            )

            replayed = client.post(
                f"/api/evals/suites/{suite.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "authored-suite-full-1"},
                json=launch_body,
            )
            assert replayed.status_code == 202
            assert [item["run"]["spec"]["run_id"] for item in replayed.json()["runs"]] == [
                item["run"]["spec"]["run_id"] for item in body["runs"]
            ]
            changed_selection = client.post(
                f"/api/evals/suites/{suite.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "authored-suite-full-1"},
                json={
                    "case_ids": ["refund-request"],
                    "expected_exposure_revision": subset.json()["exposure"]["revision"],
                    "expected_execution_profiles": [
                        {
                            "case_ids": subset_launch["case_ids"],
                            "execution_profile_revision": subset_launch[
                                "execution_profile_revision"
                            ],
                        }
                    ],
                },
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
                    "expected_execution_profile_revision": full.json()["launches"][0][
                        "execution_profile_revision"
                    ],
                    "max_concurrency": 1,
                },
            )
            assert independent.status_code == 202
            assert independent.json()["spec"]["run_id"] not in {
                item["run"]["spec"]["run_id"] for item in body["runs"]
            }

            def unavailable_profile(*, model: str) -> None:
                del model
                raise RuntimeError("provider temporarily unavailable")

            monkeypatch.setattr(provider, "preflight_model_target", unavailable_profile)
            replayed_while_unavailable = client.post(
                f"/api/evals/suites/{suite.revision}/runs",
                headers={**_AUTH_HEADERS, "Idempotency-Key": "authored-suite-full-1"},
                json=launch_body,
            )
            assert replayed_while_unavailable.status_code == 202
            assert [
                item["run"]["spec"]["run_id"] for item in replayed_while_unavailable.json()["runs"]
            ] == [item["run"]["spec"]["run_id"] for item in body["runs"]]
    finally:
        asyncio.run(store.close())
