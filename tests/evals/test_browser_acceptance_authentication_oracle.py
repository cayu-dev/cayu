"""Fixture-owned authentication evidence is distinct from a model's requested URL."""

import asyncio
from dataclasses import replace

import httpx
import pytest
from tests.evals.test_browser_acceptance_execution import _plan

from cayu.evals.browser_acceptance import (
    BrowserAcceptanceCaseCategory,
    BrowserAcceptanceCaseV1,
    BrowserAcceptanceDiagnosticV1,
    BrowserAcceptanceManifestV1,
    BrowserAcceptanceSemanticOracle,
    BrowserAcceptanceState,
    _semantic_state,
    run_browser_acceptance,
)
from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
from cayu.evals.runner import EvalPlan


def test_account_counter_requires_exact_cookie_not_a_route_or_substring():
    with BrowserAcceptanceFixtureV1() as fixture, httpx.Client(trust_env=False) as client:
        for cookie in (
            "",
            "other_cayu_fixture_session=active",
            "cayu_fixture_session=active-extra",
            "unrelated=cayu_fixture_session=active",
            "cayu_fixture_session=expired",
        ):
            response = client.get(
                fixture.upstream_origin + "/auth/account", headers={"Cookie": cookie}
            )
            assert response.status_code == 401
            assert fixture.authenticated_request_count() == 0
        assert fixture.request_counts()["/auth/account"] == 5
        response = client.get(
            fixture.upstream_origin + "/auth/account",
            headers={"Cookie": "other=value; cayu_fixture_session=active"},
        )
        assert response.status_code == 200
        assert fixture.authenticated_request_count() == 1
        assert "cayu_fixture_session" not in response.text


@pytest.mark.parametrize("count", [None, 0, 1, 2])
def test_authentication_oracle_survives_diagnostic_serialization(count):
    case = BrowserAcceptanceCaseV1.build(
        case_id="profile-restoration-oracle",
        category=BrowserAcceptanceCaseCategory.SUCCESS,
        expected_state=BrowserAcceptanceState.PASSED,
        semantic_oracle=BrowserAcceptanceSemanticOracle.OBSERVATION,
        operations=("navigate",),
        oracle_parameters={"required_authenticated_requests": 1},
    )
    diagnostic = BrowserAcceptanceDiagnosticV1(
        state="captured",
        fixture_authenticated_request_count=count,
        operations=(
            {
                "sequence": 1,
                "invocation_revision": "sha256:" + "1" * 64,
                "operation": "navigate",
                "state": "terminal",
                "allocation_disposition": "retired",
            },
        ),
    )
    reconstructed = BrowserAcceptanceDiagnosticV1.model_validate_json(diagnostic.model_dump_json())
    assert _semantic_state(
        case, reconstructed, public_operations=frozenset({"navigate"})
    ).value == ("passed" if count == 1 else "failed")


@pytest.mark.parametrize("count", [True, -1, "1", (1 << 20) + 1])
def test_authentication_counter_rejects_malformed_or_unbounded_evidence(count):
    with pytest.raises(ValueError):
        BrowserAcceptanceDiagnosticV1(state="captured", fixture_authenticated_request_count=count)


def test_prior_authenticated_request_cannot_satisfy_a_later_trial(tmp_path):
    with BrowserAcceptanceFixtureV1() as fixture, httpx.Client(trust_env=False) as client:
        assert (
            client.get(
                fixture.upstream_origin + "/auth/account",
                headers={"Cookie": "cayu_fixture_session=active"},
            ).status_code
            == 200
        )
        source = _plan(tmp_path, fixture)
        original = source.manifest.cases[0]
        case = BrowserAcceptanceCaseV1.build(
            **{
                **original.model_dump(mode="python", exclude={"revision"}),
                "oracle_parameters": {
                    **original.oracle_parameters,
                    "required_authenticated_requests": 1,
                },
            }
        )
        manifest = BrowserAcceptanceManifestV1.build(
            **{
                **source.manifest.model_dump(mode="python", exclude={"revision"}),
                "limits": source.manifest.limits,
                "cases": (case, *source.manifest.cases[1:]),
            }
        )
        executable = source.eval_plan.suite.cases[0].model_copy(
            update={"metadata": {"browser_acceptance_case_revision": case.revision}}
        )
        plan = replace(
            source,
            manifest=manifest,
            eval_plan=EvalPlan(
                app=source.eval_plan.app,
                suite=source.eval_plan.suite.model_copy(update={"cases": [executable]}),
            ),
        )
        report = asyncio.run(run_browser_acceptance(plan, deterministic_fixture=fixture))
        assert report.rows[0].diagnostic.fixture_authenticated_request_count == 0
        assert report.rows[0].semantic_state.value == "failed"
