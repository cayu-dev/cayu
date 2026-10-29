from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient
from tests.evals.test_process_local_comparison import prepared
from tests.server.test_server_eval_suite_authoring import _simple_case
from tests.server.test_server_eval_suite_launch import _server
from tests.server.test_server_evals import _AUTH_HEADERS, _wait_for_terminal

from cayu.evals.read_views import run_view_id
from cayu.evals.suite_authoring import EvalSuiteDraftV1, compile_eval_suite_draft
from cayu.storage.evals_sqlite import SQLiteEvalStore


@pytest.mark.parametrize("declared,tool_count", [(False, 1), (True, 1), (False, 300)])
def test_http_suite_runs_explain_identity_drift_after_store_reopen(tmp_path, declared, tool_count):
    run_ids = []
    for generation in range(2):
        target, _ = prepared(declared=declared, tool_count=tool_count)
        suite = compile_eval_suite_draft(
            EvalSuiteDraftV1(
                id="restart",
                name="Restart",
                target_key=target.key,
                cases=(_simple_case(),),
            )
        )
        store = SQLiteEvalStore(tmp_path / "evals.db")
        try:
            asyncio.run(store.save_authored_suite(suite, redact_json=target.app.redact_json))
            with TestClient(_server(target, store)) as client:
                root = f"/api/evals/suites/{suite.revision}/runs"
                preview = client.post(f"{root}/preview", headers=_AUTH_HEADERS, json={})
                assert preview.status_code == 200, preview.text
                body = preview.json()
                launched = client.post(
                    root,
                    headers={**_AUTH_HEADERS, "Idempotency-Key": f"generation-{generation}"},
                    json={
                        "expected_exposure_revision": body["exposure"]["revision"],
                        "expected_execution_profiles": [
                            {
                                "case_ids": item["case_ids"],
                                "execution_profile_revision": item["execution_profile_revision"],
                            }
                            for item in body["launches"]
                        ],
                    },
                )
                assert launched.status_code == 202, launched.text
                run_id = launched.json()["runs"][0]["run"]["spec"]["run_id"]
                terminal = _wait_for_terminal(client, run_id)
                assert terminal["status"] == "completed", terminal
                run_ids.append(run_id)
                if generation:
                    response = client.get(
                        f"/api/evals/run-views/{run_view_id('run', run_id)}",
                        headers=_AUTH_HEADERS,
                        params={"compared_with": run_view_id("run", run_ids[0])},
                    )
                    assert response.status_code == 200, response.text
                    comparison = response.json()["comparison"]
                    if declared:
                        assert comparison["state"] == "compared", comparison
                        assert comparison["compatibility"]["process_local_components"] == []
                    else:
                        assert comparison["state"] == "incompatible", comparison
                        assert (
                            "process_local_execution_identity"
                            in comparison["compatibility"]["reasons"]
                        )
                        assert comparison["compatibility"]["process_local_components"] == (
                            ["tool:agent/lookup"]
                            if tool_count == 1
                            else [f"tool:agent/t{i:04d}" for i in range(8)]
                        )
                        assert comparison["compatibility"][
                            "process_local_components_truncated"
                        ] == (tool_count > 8)
        finally:
            asyncio.run(store.close())


def test_current_dashboard_memory_report_fixture_is_accepted_after_profile_diagnostics(tmp_path):
    from pathlib import Path

    target, _ = prepared()
    source = (
        Path(__file__).resolve().parents[2]
        / "examples/fixtures/dashboard-memory-report-request.json"
    )
    store = SQLiteEvalStore(tmp_path / "memory-fixture.db")
    try:
        with TestClient(_server(target, store)) as client:
            response = client.post(
                "/api/evals/memory-reports",
                headers={**_AUTH_HEADERS, "Content-Type": "application/json"},
                content=source.read_bytes(),
            )
            assert response.status_code == 200, response.text
    finally:
        asyncio.run(store.close())
