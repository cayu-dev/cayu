from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime

import pytest

from cayu import (
    EnvironmentFactoryRequest,
    EvalAssertionResult,
    EvalCaseResult,
    EvalOutcome,
    EvalRun,
    EvalStatus,
    EvalTrialResult,
    EventType,
    SessionUsageSummary,
    run_eval_plan,
)

_CASE_IDS = [
    "tool_roundtrip",
    "workspace_roundtrip",
    "context_observability",
    "knowledge_tool_roundtrip",
    "subagent_roundtrip",
    "usage_accounting",
    "budget_interrupt",
]
_LIVE_CREDENTIAL_ENV = {
    "ANTHROPIC_API_KEY",
    "E2B_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
}
_WORKSPACE_FILE = "runtime-acceptance/workspace-roundtrip.txt"


def _assert_acceptance_passed(result: EvalRun) -> None:
    # This suite is hermetic. Report the concrete case/trial failure instead of
    # pytest's truncated EvalRun repr, without dumping outputs or trajectories.
    if result.status is EvalStatus.PASSED:
        return
    raise AssertionError(
        "Runtime acceptance failed:\n"
        + json.dumps(
            [
                {
                    "case_id": case.case_id,
                    "status": case.status.value,
                    "error": case.error,
                    "unavailable_reason": case.unavailable_reason,
                    "trials": [
                        {
                            "trial": trial.trial_number,
                            "status": trial.status.value,
                            "error": trial.error,
                            "unavailable_reason": trial.unavailable_reason,
                            "duration_ms": trial.duration_ms,
                            "events_count": trial.events_count,
                            "evidence_complete": trial.evidence_complete,
                            "assertions": [
                                {
                                    "name": assertion.name,
                                    "outcome": assertion.outcome.value,
                                    "message": assertion.message,
                                }
                                for assertion in trial.assertions
                                if assertion.outcome is not EvalOutcome.PASSED
                            ],
                        }
                        for trial in case.trials
                        if trial.status is not EvalStatus.PASSED
                    ],
                }
                for case in result.cases
                if case.status is not EvalStatus.PASSED
            ],
            indent=2,
        )
    )


@pytest.mark.parametrize("status", [EvalStatus.ERROR, EvalStatus.FAILED, EvalStatus.UNAVAILABLE])
def test_acceptance_failure_reports_case_and_trial_evidence(status):
    now = datetime.now(UTC)
    trial = EvalTrialResult(
        trial_number=1,
        status=status,
        error="synthetic execution failure" if status is EvalStatus.ERROR else None,
        unavailable_reason="synthetic missing evidence"
        if status is EvalStatus.UNAVAILABLE
        else None,
        score=0.0 if status is EvalStatus.FAILED else None,
        evidence_complete=status is EvalStatus.FAILED,
        session_id="synthetic-session" if status is EvalStatus.FAILED else None,
        usage_summary=(
            SessionUsageSummary(session_id="synthetic-session").model_dump(mode="json")
            if status is EvalStatus.FAILED
            else None
        ),
        assertions=(
            EvalAssertionResult(
                name="synthetic-check",
                outcome=EvalOutcome(status.value),
                score=0.0 if status is EvalStatus.FAILED else None,
                message="synthetic assertion detail",
            ),
        ),
        final_output="output-must-not-be-dumped",
        started_at=now,
        completed_at=now,
    )
    case = EvalCaseResult.from_trials(case_id="synthetic-case", trials=(trial,))
    result = EvalRun(
        suite_id="synthetic-suite",
        status=status,
        score=case.score,
        cases=(case,),
        started_at=now,
        completed_at=now,
    )
    with pytest.raises(AssertionError, match="Runtime acceptance failed") as raised:
        _assert_acceptance_passed(result)
    detail = json.loads(str(raised.value).split("\n", 1)[1])
    assert detail[0]["case_id"] == "synthetic-case"
    assert detail[0]["trials"][0]["error"] == trial.error
    assert detail[0]["trials"][0]["unavailable_reason"] == trial.unavailable_reason
    assert detail[0]["trials"][0]["assertions"][0]["message"] == "synthetic assertion detail"
    assert "output-must-not-be-dumped" not in str(raised.value)


@pytest.mark.parametrize("max_concurrency", [1, 2])
def test_internal_runtime_acceptance_plan_is_hermetic_and_isolated(
    monkeypatch, max_concurrency
) -> None:
    from cayu.evals.internal.runtime_acceptance import build

    environ_type = type(os.environ)
    original_getitem = environ_type.__getitem__
    original_contains = environ_type.__contains__

    def guarded_getitem(environ, key):
        if key in _LIVE_CREDENTIAL_ENV:
            raise AssertionError(f"internal eval read live credential {key}")
        return original_getitem(environ, key)

    def guarded_contains(environ, key):
        if key in _LIVE_CREDENTIAL_ENV:
            raise AssertionError(f"internal eval inspected live credential {key}")
        return original_contains(environ, key)

    monkeypatch.setattr(environ_type, "__getitem__", guarded_getitem)
    monkeypatch.setattr(environ_type, "__contains__", guarded_contains)

    async def run():
        plan = await build()
        result = await run_eval_plan(
            plan,
            max_concurrency=max_concurrency,
            case_timeout_seconds=20,
            retain_trajectory=True,
        )
        _assert_acceptance_passed(result)
        cases_by_id = {case.case_id: case for case in result.cases}
        workspace_session_id = cases_by_id["workspace_roundtrip"].trials[0].session_id
        tool_session_id = cases_by_id["tool_roundtrip"].trials[0].session_id
        assert workspace_session_id is not None
        assert tool_session_id is not None

        factory = plan.app.get_environment_factory()
        workspace_environment = await factory.create(
            EnvironmentFactoryRequest(
                session_id=workspace_session_id,
                agent_name="runtime_acceptance_workspace",
                environment_name="runtime-acceptance-local",
            )
        )
        other_environment = await factory.create(
            EnvironmentFactoryRequest(
                session_id=tool_session_id,
                agent_name="runtime_acceptance_tool",
                environment_name="runtime-acceptance-local",
            )
        )
        return plan, result, workspace_environment, other_environment

    plan, result, workspace_environment, other_environment = asyncio.run(run())

    assert plan.suite.id == "cayu-internal-runtime-acceptance-v1"
    assert [case.id for case in plan.suite.cases] == _CASE_IDS
    assert [case.case_id for case in result.cases] == _CASE_IDS
    assert all(case.status is EvalStatus.PASSED for case in result.cases)

    budget_trial = next(case for case in result.cases if case.case_id == "budget_interrupt").trials[
        0
    ]
    assert budget_trial.evidence_complete is True
    assert budget_trial.trajectory is not None
    assert budget_trial.trajectory.events[-1].type is EventType.SESSION_INTERRUPTED
    assert all(assertion.outcome is EvalOutcome.PASSED for assertion in budget_trial.assertions)

    assert workspace_environment.environment.workspace is not None
    assert other_environment.environment.workspace is not None
    written = asyncio.run(workspace_environment.environment.workspace.read_bytes(_WORKSPACE_FILE))
    assert written.content == b"isolated workspace content"
    with pytest.raises(FileNotFoundError, match="Workspace file not found"):
        asyncio.run(other_environment.environment.workspace.read_bytes(_WORKSPACE_FILE))
