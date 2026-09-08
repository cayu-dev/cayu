"""Operator scorecard characterization, not a live handoff acceptance proof."""

import asyncio
from dataclasses import replace

import pytest
from tests.core.test_browser_control_audit import acquired_record
from tests.evals.test_browser_acceptance_execution import _plan

from cayu import InMemorySessionStore, ScriptedModelProvider, SQLiteSessionStore
from cayu.evals.browser_acceptance import (
    BrowserAcceptanceCaseCategory,
    BrowserAcceptanceCaseV1,
    BrowserAcceptanceDiagnosticV1,
    BrowserAcceptanceManifestV1,
    BrowserAcceptanceOperatorEvidenceV1,
    BrowserAcceptanceSemanticOracle,
    BrowserAcceptanceState,
    _project_operator_record,
    _semantic_state,
    run_browser_acceptance,
)
from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
from cayu.evals.runner import EvalPlan
from cayu.runtime.browser_control import BrowserControlPageAudit


def _settled_record():
    acquired = acquired_record()
    return acquired.model_copy(
        update={
            "state": "closed",
            "control_epoch": 3,
            "revision": 10,
            "lease_until_ms": None,
            "handback_audit": BrowserControlPageAudit(
                request_id=acquired.request.request_id,
                phase="handed_back",
                control_epoch=3,
                locations=acquired.acquisition_audit.locations,
            ),
            "settled_input_sequence": 2,
            "operator_page_operations": ({"page_id": "page", "operations": 2},),
        }
    )


def _case():
    return BrowserAcceptanceCaseV1.build(
        case_id="operator-handoff-oracle",
        category=BrowserAcceptanceCaseCategory.SUCCESS,
        expected_state=BrowserAcceptanceState.PASSED,
        semantic_oracle=BrowserAcceptanceSemanticOracle.OBSERVATION,
        operations=("navigate", "observe", "close"),
        oracle_parameters={
            "required_operations": ["navigate", "observe", "close"],
            "required_operator_inputs": 2,
            "allocation_disposition": "retired",
        },
    )


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"state": "operator_controlled"},
        {"state": "control_uncertain"},
        {"acquired_epoch": None},
        {"handed_back_epoch": 4},
        {"control_epoch": 4},
        {"acquisition_revision": None},
        {"handback_revision": None},
        {"fresh_observation_revision": None},
        {"settled_inputs": 1},
        {"input_pending": True},
        {"fresh_observation_required": True},
        {"sensitive_entry_pending": True},
        {"mutation_uncertain": True},
        {"browser_session_revision": "sha256:" + "d" * 64},
    ],
)
def test_operator_oracle_requires_complete_matching_durable_evidence(change):
    proof = _project_operator_record(_settled_record()).model_copy(
        update={"fresh_observation_revision": "sha256:" + "e" * 64}
    )
    altered = BrowserAcceptanceOperatorEvidenceV1.model_validate(
        {**proof.model_dump(mode="python"), **(change or {})}
    )
    diagnostic = BrowserAcceptanceDiagnosticV1.model_validate(
        {
            "state": "captured",
            "operator": altered,
            "operations": tuple(
                {
                    "sequence": index + 1,
                    "invocation_revision": "sha256:" + str(index + 1) * 64,
                    "operation": operation,
                    "state": "terminal",
                    "allocation_disposition": "retired" if operation == "close" else "live",
                    "browser_session_revision": proof.browser_session_revision,
                }
                for index, operation in enumerate(_case().operations)
            ),
        }
    )
    reconstructed = BrowserAcceptanceDiagnosticV1.model_validate_json(diagnostic.model_dump_json())
    assert _semantic_state(_case(), reconstructed, public_operations=frozenset()).value == (
        "passed" if change is None else "failed"
    )
    if change is None:
        absent = reconstructed.model_copy(update={"operator": None})
        assert _semantic_state(_case(), absent, public_operations=frozenset()).value == "failed"


def test_operator_projection_keeps_private_control_labels_out_of_report():
    record = _settled_record()
    projected = _project_operator_record(record)
    encoded = projected.model_dump_json()
    for value in (
        record.identity.browser_session_id,
        record.identity.worker_instance_id,
        record.request.operator.operator_session_id,
        record.request.request_id,
        record.acquisition_audit.locations[0].origin,
    ):
        assert value not in encoded


@pytest.mark.parametrize("field", ["control_epoch", "settled_inputs", "acquired_epoch"])
def test_operator_projection_does_not_accept_boolean_counters(field):
    values = _project_operator_record(_settled_record()).model_dump(mode="python")
    values[field] = True
    with pytest.raises(ValueError):
        BrowserAcceptanceOperatorEvidenceV1.model_validate(values)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_completed_trial_without_operator_evidence_cannot_pass_handoff(tmp_path, backend):
    async def scenario(fixture):
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "sessions.sqlite")
        )
        try:
            original = _plan(tmp_path, fixture, session_store=store)
            case = original.manifest.cases[0]
            case = BrowserAcceptanceCaseV1.build(
                **{
                    **case.model_dump(mode="python", exclude={"revision"}),
                    "oracle_parameters": {
                        **case.oracle_parameters,
                        "required_operator_inputs": 2,
                        "required_operator_observation_id": "missing-fresh-observation",
                    },
                }
            )
            manifest = BrowserAcceptanceManifestV1.build(
                **{
                    **original.manifest.model_dump(mode="python", exclude={"revision"}),
                    "limits": original.manifest.limits,
                    "cases": (case, *original.manifest.cases[1:]),
                }
            )
            suite = original.eval_plan.suite
            app = original.eval_plan.app
            assert suite is not None and app is not None
            executable = suite.cases[0].model_copy(
                update={"metadata": {"browser_acceptance_case_revision": case.revision}}
            )
            plan = replace(
                original,
                manifest=manifest,
                eval_plan=EvalPlan(app=app, suite=suite.model_copy(update={"cases": [executable]})),
            )
            report = await run_browser_acceptance(plan, deterministic_fixture=fixture)
            row = report.rows[0]
            assert row.observed_state.value == "passed"
            assert row.conformance.value == "incomplete"
            assert row.diagnostic.error_code == "diagnostic_projection_failed"
            assert row.diagnostic.operator is None
            assert report.aggregate.overall_status.value != "passed"
            provider = app.get_provider("scripted")
            assert isinstance(provider, ScriptedModelProvider)
            assert len(provider.requests) == 2
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))
