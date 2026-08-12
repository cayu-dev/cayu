from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from cayu import (
    CayuApp,
    EvalAssertionResult,
    EvalCase,
    EvalCaseComparison,
    EvalCaseResult,
    EvalContext,
    EvalOutcome,
    EvalPlan,
    EvalRun,
    EvalRunComparison,
    EvalStatus,
    EvalSuite,
    EvalTrialResult,
    Message,
    MessageRole,
    RunRequest,
    StructuredOutputError,
    StructuredOutputValidation,
    ToolCallPart,
    Trajectory,
    TrajectoryProbes,
    run_eval_case,
    run_eval_suite,
)
from cayu.evals.models import WorkspaceFileProbe

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _unavailable_trial(**updates: Any) -> EvalTrialResult:
    values: dict[str, Any] = {
        "trial_number": 1,
        "status": EvalStatus.UNAVAILABLE,
        "unavailable_reason": "evidence unavailable",
        "started_at": _NOW,
        "completed_at": _NOW,
    }
    values.update(updates)
    return EvalTrialResult(**values)


def _case_result(
    *,
    case_id: str = "case",
    authored_session_id: str | None = None,
) -> EvalCaseResult:
    return EvalCaseResult.from_trials(
        case_id=case_id,
        authored_session_id=authored_session_id,
        trials=[_unavailable_trial()],
        started_at=_NOW,
        completed_at=_NOW,
    )


def _eval_run(*, run_id: str = "run", suite_id: str = "suite") -> EvalRun:
    return EvalRun(
        run_id=run_id,
        suite_id=suite_id,
        status=EvalStatus.UNAVAILABLE,
        score=None,
        cases=(_case_result(),),
        started_at=_NOW,
        completed_at=_NOW,
    )


def _run_request(*, messages: list[Message] | None = None) -> RunRequest:
    return RunRequest(
        agent_name="agent",
        messages=[Message.text("user", "test")] if messages is None else messages,
    )


def _eval_case(*, request: RunRequest | None = None) -> EvalCase:
    return EvalCase(id="case", request=_run_request() if request is None else request)


@dataclass(frozen=True)
class _PortableTextBoundary:
    owner: str
    field: str
    capture: Callable[[str], Any]

    @property
    def id(self) -> str:
        return f"{self.owner}.{self.field}"


_PORTABLE_TEXT_BOUNDARIES = (
    _PortableTextBoundary(
        "EvalAssertionResult",
        "message",
        lambda value: EvalAssertionResult(
            name="assertion",
            outcome=EvalOutcome.UNAVAILABLE,
            message=value,
        ),
    ),
    _PortableTextBoundary(
        "EvalAssertionResult",
        "assertion_revision",
        lambda value: EvalAssertionResult(
            name="assertion",
            assertion_revision=value,
            outcome=EvalOutcome.UNAVAILABLE,
        ),
    ),
    _PortableTextBoundary(
        "EvalTrialResult",
        "error",
        lambda value: _unavailable_trial(
            status=EvalStatus.ERROR,
            unavailable_reason=None,
            error=value,
        ),
    ),
    _PortableTextBoundary(
        "EvalTrialResult",
        "unavailable_reason",
        lambda value: _unavailable_trial(unavailable_reason=value),
    ),
    _PortableTextBoundary(
        "EvalTrialResult",
        "final_output",
        lambda value: _unavailable_trial(final_output=value),
    ),
    _PortableTextBoundary(
        "EvalTrialResult",
        "session_id",
        lambda value: _unavailable_trial(session_id=value),
    ),
    _PortableTextBoundary(
        "EvalCaseResult",
        "case_id",
        lambda value: _case_result(case_id=value),
    ),
    _PortableTextBoundary(
        "EvalCaseResult",
        "authored_session_id",
        lambda value: _case_result(authored_session_id=value),
    ),
    _PortableTextBoundary(
        "EvalRun",
        "run_id",
        lambda value: _eval_run(run_id=value),
    ),
    _PortableTextBoundary(
        "EvalRun",
        "suite_id",
        lambda value: _eval_run(suite_id=value),
    ),
    _PortableTextBoundary(
        "Trajectory",
        "final_output",
        lambda value: Trajectory(final_output=value),
    ),
    _PortableTextBoundary(
        "TrajectoryProbes",
        "workspace_unavailable_paths",
        lambda value: TrajectoryProbes(
            workspace_available=True,
            workspace_unavailable_paths=(value,),
        ),
    ),
    _PortableTextBoundary(
        "TrajectoryProbes",
        "workspace_files key",
        lambda value: TrajectoryProbes(
            workspace_available=True,
            workspace_files={value: None},
        ),
    ),
)


def test_eval_portable_text_conformance_covers_every_issue_boundary() -> None:
    assert {(case.owner, case.field) for case in _PORTABLE_TEXT_BOUNDARIES} == {
        ("EvalAssertionResult", "message"),
        ("EvalAssertionResult", "assertion_revision"),
        ("EvalTrialResult", "error"),
        ("EvalTrialResult", "unavailable_reason"),
        ("EvalTrialResult", "final_output"),
        ("EvalTrialResult", "session_id"),
        ("EvalCaseResult", "case_id"),
        ("EvalCaseResult", "authored_session_id"),
        ("EvalRun", "run_id"),
        ("EvalRun", "suite_id"),
        ("Trajectory", "final_output"),
        ("TrajectoryProbes", "workspace_unavailable_paths"),
        ("TrajectoryProbes", "workspace_files key"),
    }


@pytest.mark.parametrize("case", _PORTABLE_TEXT_BOUNDARIES, ids=lambda case: case.id)
@pytest.mark.parametrize("invalid_text", ["nul\x00text", "surrogate\ud800text"])
def test_eval_evidence_rejects_nonportable_text(
    case: _PortableTextBoundary,
    invalid_text: str,
) -> None:
    with pytest.raises(ValueError):
        case.capture(invalid_text)


@pytest.mark.parametrize("case", _PORTABLE_TEXT_BOUNDARIES, ids=lambda case: case.id)
def test_eval_evidence_accepts_ordinary_unicode(case: _PortableTextBoundary) -> None:
    case.capture("Zażółć gęślą jaźń 😀")


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("entry_point", ["case", "suite"])
def test_eval_execution_rejects_non_finite_timeouts_before_execution(
    timeout: float,
    entry_point: str,
) -> None:
    app = CayuApp()
    case = _eval_case()

    with pytest.raises(ValueError, match="finite positive number"):
        if entry_point == "case":
            asyncio.run(
                run_eval_case(
                    app,
                    case,
                    suite_id="suite",
                    timeout_seconds=timeout,
                )
            )
        else:
            asyncio.run(
                run_eval_suite(
                    app,
                    EvalSuite(id="suite", cases=[case]),
                    case_timeout_seconds=timeout,
                )
            )


def test_eval_suite_detaches_single_already_validated_case() -> None:
    source_case = _eval_case()
    source_case.metadata["nested"] = {"status": "original"}

    suite = EvalSuite(
        id="suite",
        cases=source_case,  # ty: ignore[invalid-argument-type] -- accepted singleton input
    )
    retained_dump = suite.model_dump(mode="python")

    source_case.id = "changed"
    source_case.request.agent_name = "changed"
    source_case.metadata["nested"]["status"] = "changed"

    assert suite.cases[0] is not source_case
    assert suite.cases[0].id == "case"
    assert suite.cases[0].request.agent_name == "agent"
    assert suite.cases[0].metadata == {"nested": {"status": "original"}}
    assert suite.model_dump(mode="python") == retained_dump


def test_eval_suite_and_plan_normalize_case_subclass_from_iterable() -> None:
    class MutableEvalCase(EvalCase):
        pass

    source_case = MutableEvalCase(id="case", request=_run_request())
    source_case.metadata["nested"] = {"status": "original"}

    suite = EvalSuite(id="suite", cases=[source_case])
    plan = EvalPlan(app=CayuApp(), suite=suite)
    suite_dump = suite.model_dump(mode="python")
    assert plan.suite is not None
    plan_dump = plan.suite.model_dump(mode="python")

    source_case.id = "changed"
    source_case.request.agent_name = "changed"
    source_case.metadata["nested"]["status"] = "changed"

    assert type(suite.cases[0]) is EvalCase
    assert suite.cases[0] is not source_case
    assert suite.cases[0].id == "case"
    assert suite.cases[0].request.agent_name == "agent"
    assert suite.cases[0].metadata == {"nested": {"status": "original"}}
    assert suite.model_dump(mode="python") == suite_dump
    assert type(plan.suite.cases[0]) is EvalCase
    assert plan.suite.cases[0] is not source_case
    assert plan.suite.model_dump(mode="python") == plan_dump


def test_eval_suite_detaches_already_validated_cases_transitively() -> None:
    source_case = _eval_case(
        request=_run_request(
            messages=[
                Message(
                    role=MessageRole.ASSISTANT,
                    content=(
                        ToolCallPart(
                            tool_call_id="call-1",
                            tool_name="lookup",
                            arguments={"query": {"status": "original"}},
                        ),
                    ),
                )
            ]
        )
    )
    source_case.metadata["nested"] = {"status": "original"}

    suite = EvalSuite(id="suite", cases=[source_case])

    source_case.id = "changed"
    source_case.request.agent_name = "changed"
    source_case.request.messages.append(Message.text("user", "changed"))
    source_case.metadata["nested"]["status"] = "changed"
    source_part = source_case.request.messages[0].content[0]
    assert isinstance(source_part, ToolCallPart)
    source_part.arguments["query"]["status"] = "changed"

    retained = suite.cases[0]
    retained_part = retained.request.messages[0].content[0]
    assert retained is not source_case
    assert retained.id == "case"
    assert retained.request.agent_name == "agent"
    assert len(retained.request.messages) == 1
    assert retained.metadata == {"nested": {"status": "original"}}
    assert isinstance(retained_part, ToolCallPart)
    assert retained_part.arguments == {"query": {"status": "original"}}


def test_eval_run_comparison_detaches_already_validated_case_comparisons() -> None:
    source = EvalCaseComparison(
        case_id="case",
        baseline_status=EvalStatus.PASSED,
        current_status=EvalStatus.FAILED,
        regressions=("status regressed",),
    )

    comparison = EvalRunComparison(
        baseline_run_id="baseline",
        current_run_id="current",
        baseline_suite_id="suite",
        current_suite_id="suite",
        baseline_status=EvalStatus.PASSED,
        current_status=EvalStatus.FAILED,
        cases=(source,),
    )

    source.case_id = "changed"
    source.regressions = ("changed",)

    assert comparison.cases[0] is not source
    assert comparison.cases[0].case_id == "case"
    assert comparison.cases[0].regressions == ("status regressed",)


def _comparison_input(model_type: type, field_name: str, value: object) -> dict[str, object]:
    if model_type is EvalCaseComparison:
        return {"case_id": "case", field_name: value}
    return {
        "baseline_run_id": "baseline",
        "current_run_id": "current",
        "baseline_suite_id": "suite",
        "current_suite_id": "suite",
        "baseline_status": EvalStatus.PASSED,
        "current_status": EvalStatus.PASSED,
        field_name: value,
    }


@pytest.mark.parametrize("model_type", [EvalCaseComparison, EvalRunComparison])
@pytest.mark.parametrize("field_name", ["baseline_score", "current_score"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_eval_comparison_scores_reject_nonfinite_values(
    model_type: type,
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        model_type.model_validate(_comparison_input(model_type, field_name, value))


@pytest.mark.parametrize("model_type", [EvalCaseComparison, EvalRunComparison])
@pytest.mark.parametrize("field_name", ["baseline_score", "current_score"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_eval_comparison_scores_reject_out_of_range_values(
    model_type: type,
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        model_type.model_validate(_comparison_input(model_type, field_name, value))


@pytest.mark.parametrize("model_type", [EvalCaseComparison, EvalRunComparison])
@pytest.mark.parametrize("field_name", ["baseline_score", "current_score"])
@pytest.mark.parametrize("value", [True, "0.5"])
def test_eval_comparison_scores_preserve_strict_type_validation(
    model_type: type,
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        model_type.model_validate(_comparison_input(model_type, field_name, value))


@pytest.mark.parametrize("model_type", [EvalCaseComparison, EvalRunComparison])
@pytest.mark.parametrize("field_name", ["baseline_score", "current_score"])
@pytest.mark.parametrize("value", [None, 0.0, 1.0])
def test_eval_comparison_scores_accept_none_and_range_boundaries(
    model_type: type,
    field_name: str,
    value: float | None,
) -> None:
    comparison = model_type.model_validate(_comparison_input(model_type, field_name, value))

    assert getattr(comparison, field_name) == value


def test_eval_plan_detaches_suite_but_preserves_application_reference() -> None:
    app = CayuApp()
    source_suite = EvalSuite(id="suite", cases=[_eval_case()])
    source_suite.metadata["nested"] = {"status": "original"}

    plan = EvalPlan(app=app, suite=source_suite)

    source_suite.id = "changed"
    source_suite.cases[0].id = "changed"
    source_suite.metadata["nested"]["status"] = "changed"

    assert plan.app is app
    assert plan.suite is not source_suite
    assert plan.suite is not None
    assert plan.suite.id == "suite"
    assert plan.suite.cases[0].id == "case"
    assert plan.suite.metadata == {"nested": {"status": "original"}}


def test_structured_output_validation_detaches_already_validated_errors() -> None:
    source = StructuredOutputError(path="$.answer", message="original", schema_path="$.type")

    validation = StructuredOutputValidation(valid=False, errors=[source])

    source.message = "changed"

    assert validation.errors[0] is not source
    assert validation.errors[0].message == "original"


def test_eval_context_detaches_trajectory_and_metadata_transitively() -> None:
    source = Trajectory(
        final_output="original",
        probes=TrajectoryProbes(
            workspace_available=True,
            workspace_files={"result.txt": b"original"},
            workspace_file_stats={
                "result.txt": WorkspaceFileProbe(
                    total_bytes=8,
                    sha256="0682c5f2076f099c34cfdd15a9e063849ed437a49677e6fcc5b4198c76575be5",
                    truncated=False,
                )
            },
        ),
        metadata={"nested": {"status": "original"}},
    )
    context_metadata = {"nested": {"status": "original"}}

    context = EvalContext(
        trajectory=source,
        suite_id="suite",
        case_id="case",
        metadata=context_metadata,
    )

    source.final_output = "changed"
    source.probes.workspace_files["result.txt"] = b"changed"
    source.metadata["nested"]["status"] = "changed"
    context_metadata["nested"]["status"] = "changed"

    assert context.trajectory is not source
    assert context.final_output == "original"
    assert context.probes.workspace_files == {"result.txt": b"original"}
    assert context.trajectory.metadata == {"nested": {"status": "original"}}
    assert context.metadata == {"nested": {"status": "original"}}
