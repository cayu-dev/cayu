from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from tests.server.test_server_evals import _AUTH_HEADERS, _server

from cayu.evals.benchmark_campaign import (
    admit_benchmark_campaign,
    prepare_benchmark_campaign,
    resume_benchmark_campaign,
)
from cayu.evals.benchmark_package import load_benchmark_package
from cayu.evals.benchmark_synthetic import (
    build_synthetic_benchmark_plan,
    write_synthetic_benchmark_package,
)
from cayu.server.contracts import EvalResultResponse
from cayu.storage.evals_sqlite import SQLiteEvalStore


def test_dashboard_campaign_result_links_match_private_native_checkpoints(tmp_path):
    plan = build_synthetic_benchmark_plan(tmp_path / "target")
    directory = tmp_path / "campaign"

    async def prepare():
        loaded = load_benchmark_package(write_synthetic_benchmark_package(tmp_path / "package"))
        prepared = await prepare_benchmark_campaign(loaded, plan, case_ids=["attachment"])
        await admit_benchmark_campaign(prepared, directory)
        await resume_benchmark_campaign(directory, plan)
        return prepared.campaign

    campaign = asyncio.run(prepare())
    store = SQLiteEvalStore(directory / "evals.sqlite3")
    target = plan.corpus_target
    calls = len(target.app.get_provider().requests)
    try:
        with TestClient(
            _server(target, store, execution_profile_policy=plan.execution_profile_policy)
        ) as client:
            route = f"/api/evals/runs/{campaign.runs[0].spec.id}"
            assert client.get(route + "/result").status_code == 401
            response = client.get(route + "/result", headers=_AUTH_HEADERS)
            assert response.status_code == 200, response.text
            document = response.json()
            assert len(document["trial_evidence_links"]) == 1
            link = document["trial_evidence_links"][0]
            trial = document["result"]["run"]["cases"][0]["trials"][0]
            assert link["source_trial_revision"] == trial["source_trial_revision"]
            assert (
                client.get(f"/api/sessions/{link['session_id']}", headers=_AUTH_HEADERS).status_code
                == 200
            )
            report = client.get(route + "/report.html", headers=_AUTH_HEADERS)
            assert report.status_code == 200
            assert link["session_id"] not in report.text
            document["trial_evidence_links"][0]["source_trial_revision"] = "a" * 64
            with pytest.raises(ValueError, match="does not match"):
                EvalResultResponse.model_validate_json(json.dumps(document))
        assert len(target.app.get_provider().requests) == calls
    finally:
        asyncio.run(store.close())
        asyncio.run(target.app.session_store.close())
