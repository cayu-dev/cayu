from __future__ import annotations

import asyncio
import json

import pytest
from tests.evals.test_workflow_eval_target import _NoChildWorkflow, _register_app, _suite, _target

from cayu._validation import DurableValueError
from cayu.cli import main
from cayu.evals import _exception_diagnostics as diagnostics
from cayu.evals.assertions import FinalOutputContains
from cayu.evals.reporting import load_eval_run, write_eval_run_json
from cayu.evals.runner import run_workflow_eval_suite
from cayu.failure_evidence import FailureEvidence
from cayu.workflows.models import ParallelStepError, StepError, StepFailure
from cayu.workflows.workflow import WorkflowBase


def _payload(summary):
    return json.loads(summary.split(": ", 1)[1])


@pytest.mark.parametrize("error", [ValueError("secret"), StepError("secret")])
def test_generic_and_typed_failures_do_not_invent_causes(error):
    summary, evidence = diagnostics.workflow_exception_diagnostic(
        error, phase="execution", stage="execution"
    )
    assert "secret" not in summary
    assert _payload(summary)["exception_type"] == type(error).__name__
    assert evidence.session_id is None
    assert evidence.classification == ("unknown" if isinstance(error, StepError) else "failure")


def test_typed_child_reference_survives():
    error = StepError(
        "secret",
        evidence=FailureEvidence(
            classification="failure", session_id="child", run_epoch=2, terminal_event_id="terminal"
        ),
    )
    summary, evidence = diagnostics.workflow_exception_diagnostic(
        error, phase="execution", stage="execution"
    )
    assert evidence.session_id == "child"
    assert _payload(summary)["failure_evidence"]["terminal_event_id"] == "terminal"


@pytest.mark.parametrize(
    "changes",
    [
        {"code": "secret", "path": "$/secret", "limit": True},
        {"path": "$/" + "1" * 600, "limit": 2**100, "observed_lower_bound": 2**101},
        {"code": [], "path": {}, "limit": "secret", "observed_lower_bound": []},
    ],
)
def test_mutated_durable_attributes_are_safe(changes):
    error = DurableValueError("json_value_too_large", "secret", limit=10, observed_lower_bound=11)
    error.__dict__.update(changes)
    summary, _ = diagnostics.workflow_exception_diagnostic(
        error, phase="evidence_preparation", stage="result_projection"
    )
    assert "secret" not in summary
    assert len(summary) < 2000
    assert _payload(summary)["durable_value"]["limit"] is None


class _HostileError(ValueError):
    def __str__(self):
        raise AssertionError("secret str")

    @property
    def execution_deadline(self):
        raise RuntimeError("secret projection")


def test_projection_failure_keeps_original_type_and_durable_details(monkeypatch):
    def broken(exc):
        raise RuntimeError("secret projection")

    monkeypatch.setattr(diagnostics, "exception_evidence", broken)
    error = DurableValueError("json_value_too_large", "secret", limit=10, observed_lower_bound=11)
    summary, evidence = diagnostics.workflow_exception_diagnostic(
        error, phase="evidence_preparation", stage="result_projection"
    )
    payload = _payload(summary)
    assert payload["projection"] == "unavailable"
    assert payload["exception_type"] == "DurableValueError"
    assert payload["durable_value"]["limit"] == 10
    assert evidence is None
    assert "secret" not in summary


def test_execution_projection_failure_does_not_become_preparation_failure():
    class BrokenWorkflow(WorkflowBase):
        spec = _NoChildWorkflow.spec

        async def run(self, session_id):
            yield await self.context(session_id).start()
            raise _HostileError("secret")

    result = asyncio.run(
        run_workflow_eval_suite(
            _target(_register_app(), BrokenWorkflow), _suite(FinalOutputContains("done"))
        )
    )
    trial = result.cases[0].trials[0]
    assert trial.execution_status == "failed"
    payload = _payload(trial.error)
    assert payload["phase"] == "execution"
    assert payload["exception_type"] == "_HostileError"
    assert payload["projection"] == "unavailable"
    assert "secret" not in trial.error


@pytest.mark.parametrize("frame", ["workflow session", "workflow event record"])
def test_root_frame_bound_survives_saved_report_and_cli(tmp_path, monkeypatch, capsys, frame):
    from cayu import _validation

    canonical = _validation.canonical_durable_json_bytes

    def bounded(value, field_name, **kwargs):
        if field_name == frame:
            kwargs["max_bytes"] = 128
        return canonical(value, field_name, **kwargs)

    monkeypatch.setattr(_validation, "canonical_durable_json_bytes", bounded)
    result = asyncio.run(
        run_workflow_eval_suite(
            _target(_register_app(), _NoChildWorkflow), _suite(FinalOutputContains("done"))
        )
    )
    trial = result.cases[0].trials[0]
    payload = _payload(trial.error)
    assert trial.execution_status == "completed"
    assert trial.failure_evidence is None
    assert payload["phase"] == "evidence_preparation"
    assert payload["stage"] == "result_projection"
    assert payload["durable_value"]["code"] == "json_value_too_large"
    assert payload["durable_value"]["limit"] == 128
    assert payload["durable_value"]["observed_lower_bound"] > 128
    report = tmp_path / "report.json"
    write_eval_run_json(result, report)
    assert load_eval_run(report).cases[0].trials[0].error == trial.error
    assert main(["eval", "report", str(report), "--format", "json"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert _payload(printed["cases"][0]["trials"][0]["error"]) == payload


def test_handled_control_transfer_is_not_a_terminal_failure():
    class HandledWorkflow(WorkflowBase):
        spec = _NoChildWorkflow.spec

        async def run(self, session_id):
            ctx = self.context(session_id)
            yield await ctx.start()
            try:
                raise _HostileError("secret control transfer")
            except _HostileError:
                pass
            yield await ctx.completed({"answer": "done"})

    result = asyncio.run(
        run_workflow_eval_suite(
            _target(_register_app(), HandledWorkflow), _suite(FinalOutputContains("done"))
        )
    )
    trial = result.cases[0].trials[0]
    assert trial.execution_status == "completed"
    assert trial.error is None
    assert trial.failure_evidence is None


def test_oversized_projected_metadata_cannot_hide_original_exception(monkeypatch):
    monkeypatch.setattr(
        diagnostics,
        "exception_evidence",
        lambda exc: FailureEvidence(run_epoch=2**100),
    )
    summary, evidence = diagnostics.workflow_exception_diagnostic(
        ValueError("secret"), phase="execution", stage="execution"
    )
    assert _payload(summary)["projection"] == "unavailable"
    assert _payload(summary)["exception_type"] == "ValueError"
    assert evidence is None


def test_parallel_failure_keeps_each_child_without_selecting_a_root_cause():
    error = ParallelStepError(
        [
            StepFailure(
                error="secret",
                error_type="ValueError",
                evidence=FailureEvidence(classification="failure", session_id=child),
            )
            for child in ("left", "right")
        ]
    )
    summary, evidence = diagnostics.workflow_exception_diagnostic(
        error, phase="execution", stage="execution"
    )
    assert evidence.session_id is None
    assert [branch.session_id for branch in evidence.branch_failures] == ["left", "right"]
    assert _payload(summary)["exception_type"] == "ParallelStepError"
    assert "secret" not in summary
