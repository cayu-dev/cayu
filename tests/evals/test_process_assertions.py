from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError
from tests._session_provenance import fixture_session_invocation

from cayu.core.agents import AgentSpec
from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.evals.corpus import (
    EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS,
    ChildStatusAssertionSpec,
    EvaluationEvidencePolicySpec,
    ProcessEventAssertionSpec,
    ProcessEventsInOrderAssertionSpec,
)
from cayu.evals.evidence import project_assertion_evidence_view
from cayu.evals.models import EvalOutcome, EvalStatus, Trajectory
from cayu.evals.portable_assertions import compile_assertion_spec
from cayu.evals.portable_evaluation import evaluate_assertion_spec
from cayu.evals.published import (
    PublishedAssertionResult,
    PublishedProcessEventDetail,
    PublishedProcessEventsInOrderDetail,
    _published_detail,
)
from cayu.evals.result_presentation import _present_assertion
from cayu.evals.runner import EvalCase, EvalSuite, run_eval_suite
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import RunRequest, Session, SessionStatus
from cayu.runtime.usage import session_usage_summary


def _session(session_id: str, status: SessionStatus, *, parent_id: str | None = None) -> Session:
    return Session(
        id=session_id,
        agent_name="agent",
        provider_name="fixture",
        model="fixture-model",
        causal_budget_id="budget",
        parent_session_id=parent_id,
        invocation=fixture_session_invocation(session_id, parent_session_id=parent_id),
        status=status,
    )


def _trajectory(
    events: tuple[Event, ...],
    *,
    children: tuple[Trajectory, ...] = (),
    children_incomplete: bool = False,
) -> Trajectory:
    session_id = "root"
    return Trajectory(
        session=_session(session_id, SessionStatus.COMPLETED),
        events=events,
        usage_summary=session_usage_summary(session_id, list(events)),
        children=children,
        children_incomplete=children_incomplete,
    )


def _process_trajectory(*, duplicate_approval: bool = False) -> Trajectory:
    session_id = "root"
    approval = Event(
        type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
        session_id=session_id,
        tool_name="private-tool",
        payload={"approval_id": "private-approval", "reason": "private reason"},
    )
    events = (
        Event(type=EventType.SESSION_STARTED, session_id=session_id),
        Event(type="custom.application.audit", session_id=session_id, payload={"secret": "x"}),
        approval,
        *(
            (
                Event(
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=session_id,
                    tool_name="private-tool",
                    payload={"approval_id": "second-private-approval"},
                ),
            )
            if duplicate_approval
            else ()
        ),
        Event(
            type=EventType.TOOL_CALL_APPROVED,
            session_id=session_id,
            tool_name="private-tool",
            payload={"approval_id": "private-approval", "actor": "private-actor"},
        ),
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id=session_id,
            tool_name="private-tool",
            payload={"tool_call_id": "private-call"},
        ),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id=session_id,
            tool_name="private-tool",
            payload={"tool_call_id": "private-call"},
        ),
        Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
    )
    return _trajectory(events)


def _evidence(trajectory: Trajectory):
    return project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        trajectory,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )


class _FailingProvider(ModelProvider):
    name = "failing"

    async def stream(self, request):
        if request is not None:
            raise RuntimeError("model exploded")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _failing_app() -> CayuApp:
    app = CayuApp(enable_logging=False)
    app.register_provider(_FailingProvider(), default=True)
    app.register_agent(AgentSpec(name="agent", model="fixture-model"))
    return app


def _compiled_process_assertion(app: CayuApp, spec):
    return compile_assertion_spec(
        spec,
        app=app,
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
        trusted_pricing=None,
    )


def _run_failing_process_case(spec):
    app = _failing_app()
    assertion = _compiled_process_assertion(app, spec)
    suite = EvalSuite(
        id="process-failure",
        cases=(
            EvalCase(
                id="failure",
                request=RunRequest(
                    agent_name="agent",
                    messages=(Message.text("user", "fail"),),
                ),
                assertions=(assertion,),
            ),
        ),
    )
    return asyncio.run(run_eval_suite(app, suite))


def test_process_event_specs_reject_raw_and_custom_event_names() -> None:
    with pytest.raises(ValidationError, match="Input should be"):
        ProcessEventAssertionSpec.model_validate(
            {"id": "raw", "event": "tool.call.approval_requested"}
        )
    with pytest.raises(ValidationError, match="Input should be"):
        ProcessEventAssertionSpec.model_validate({"id": "custom", "event": "custom.audit"})
    with pytest.raises(ValidationError, match="greater than or equal"):
        ProcessEventAssertionSpec(
            id="range",
            event="tool_approval_requested",
            min_count=2,
            max_count=1,
        )
    with pytest.raises(ValidationError, match="at least 1 item"):
        ProcessEventsInOrderAssertionSpec(id="order", events=())


def test_projection_retains_only_closed_payload_free_process_facts() -> None:
    evidence = _evidence(_process_trajectory())

    assert evidence.process_event_evidence_state == "complete"
    assert evidence.process_events == (
        "session_started",
        "tool_approval_requested",
        "tool_approved",
        "tool_call_started",
        "tool_call_completed",
        "session_completed",
    )
    document = evidence.model_dump(mode="json")
    assert "private-approval" not in str(document)
    assert "private reason" not in str(document)
    assert "private-actor" not in str(document)
    assert "custom.application.audit" not in str(document)


def test_process_event_count_supports_required_forbidden_and_ranges() -> None:
    evidence = _evidence(_process_trajectory())

    required = evaluate_assertion_spec(
        ProcessEventAssertionSpec(id="required", event="tool_approval_requested"), evidence
    )
    forbidden = evaluate_assertion_spec(
        ProcessEventAssertionSpec(
            id="forbidden",
            event="tool_approval_denied",
            min_count=0,
            max_count=0,
        ),
        evidence,
    )
    mismatch = evaluate_assertion_spec(
        ProcessEventAssertionSpec(
            id="twice",
            event="tool_approval_requested",
            min_count=2,
            max_count=2,
        ),
        evidence,
    )

    assert required.outcome is EvalOutcome.PASSED
    assert required.metadata["count"] == 1
    assert forbidden.outcome is EvalOutcome.PASSED
    assert mismatch.outcome is EvalOutcome.FAILED


@pytest.mark.parametrize(
    "spec",
    [
        ProcessEventAssertionSpec(id="failed", event="session_failed"),
        ProcessEventAssertionSpec(id="completed", event="session_completed"),
        ProcessEventAssertionSpec(id="interrupted", event="session_interrupted"),
        ProcessEventsInOrderAssertionSpec(
            id="failure-lifecycle",
            events=("session_started", "session_failed"),
        ),
    ],
)
def test_terminal_process_assertions_evaluate_failed_sessions(spec) -> None:
    compiled = _compiled_process_assertion(CayuApp(enable_logging=False), spec)

    assert compiled.evaluates_failed_session is True


@pytest.mark.parametrize(
    "spec",
    [
        ProcessEventAssertionSpec(id="started", event="session_started"),
        ProcessEventAssertionSpec(id="approval", event="tool_approval_requested"),
        ProcessEventsInOrderAssertionSpec(
            id="approval-protocol",
            events=("tool_approval_requested", "tool_approved"),
        ),
    ],
)
def test_nonterminal_process_assertions_preserve_failed_run_errors(spec) -> None:
    compiled = _compiled_process_assertion(CayuApp(enable_logging=False), spec)
    assert compiled.evaluates_failed_session is False

    result = _run_failing_process_case(spec)
    trial = result.cases[0].trials[0]

    assert result.status is EvalStatus.ERROR
    assert trial.status is EvalStatus.ERROR
    assert trial.error is not None
    assert "model exploded" in trial.error
    assert trial.assertions[0].outcome is EvalOutcome.ERROR


@pytest.mark.parametrize(
    "spec",
    [
        ProcessEventAssertionSpec(id="failed", event="session_failed"),
        ProcessEventsInOrderAssertionSpec(
            id="failure-lifecycle",
            events=("session_started", "session_failed"),
        ),
    ],
)
def test_explicit_failed_terminal_process_assertions_can_score_failed_runs(spec) -> None:
    result = _run_failing_process_case(spec)
    trial = result.cases[0].trials[0]

    assert result.status is EvalStatus.PASSED
    assert trial.status is EvalStatus.PASSED
    assert trial.error is None
    assert trial.assertions[0].outcome is EvalOutcome.PASSED


def test_completed_process_assertion_turns_failed_run_into_candidate_mismatch() -> None:
    result = _run_failing_process_case(
        ProcessEventAssertionSpec(id="completed", event="session_completed")
    )
    trial = result.cases[0].trials[0]

    assert result.status is EvalStatus.FAILED
    assert trial.status is EvalStatus.FAILED
    assert trial.error is None
    assert trial.assertions[0].outcome is EvalOutcome.FAILED


def test_process_order_is_exact_after_filtering_to_selected_fact_kinds() -> None:
    spec = ProcessEventsInOrderAssertionSpec(
        id="approval-protocol",
        events=(
            "tool_approval_requested",
            "tool_approved",
            "tool_call_started",
            "tool_call_completed",
        ),
    )

    passed = evaluate_assertion_spec(spec, _evidence(_process_trajectory()))
    extra = evaluate_assertion_spec(spec, _evidence(_process_trajectory(duplicate_approval=True)))

    assert passed.outcome is EvalOutcome.PASSED
    assert passed.metadata["actual"] == list(spec.events)
    assert extra.outcome is EvalOutcome.FAILED
    assert extra.metadata["actual"][:2] == [
        "tool_approval_requested",
        "tool_approval_requested",
    ]


def test_bounded_prefix_and_missing_root_evidence_are_unavailable() -> None:
    session_id = "root"
    events = (
        *(
            Event(type=EventType.TOOL_CALL_APPROVAL_REQUESTED, session_id=session_id)
            for _ in range(EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS + 1)
        ),
        Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
    )
    limited = _evidence(_trajectory(events))
    absent = project_assertion_evidence_view(
        CayuApp(enable_logging=False),
        Trajectory(),
        evidence_policy=EvaluationEvidencePolicySpec.standard(),
    )
    spec = ProcessEventAssertionSpec(id="approval", event="tool_approval_requested")

    assert limited.process_event_evidence_state == "limit_exceeded"
    assert len(limited.process_events) == EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS
    assert evaluate_assertion_spec(spec, limited).outcome is EvalOutcome.UNAVAILABLE
    assert absent.process_event_evidence_state == "unavailable"
    assert evaluate_assertion_spec(spec, absent).outcome is EvalOutcome.UNAVAILABLE


def test_interrupted_child_status_is_typed_and_incomplete_capture_stays_unavailable() -> None:
    child_id = "child"
    child_events = (Event(type=EventType.SESSION_INTERRUPTED, session_id=child_id),)
    child = Trajectory(
        session=_session(child_id, SessionStatus.INTERRUPTED, parent_id="root"),
        events=child_events,
        usage_summary=session_usage_summary(child_id, list(child_events)),
    )
    root_events = (Event(type=EventType.SESSION_COMPLETED, session_id="root"),)
    spec = ChildStatusAssertionSpec(id="child", expected="interrupted")

    complete = evaluate_assertion_spec(
        spec,
        _evidence(_trajectory(root_events, children=(child,))),
    )
    incomplete = evaluate_assertion_spec(
        spec,
        _evidence(
            _trajectory(
                root_events,
                children=(child,),
                children_incomplete=True,
            )
        ),
    )

    assert complete.outcome is EvalOutcome.PASSED
    assert incomplete.outcome is EvalOutcome.UNAVAILABLE


def test_published_process_details_retain_only_safe_typed_observations() -> None:
    evidence = _evidence(_process_trajectory())
    count_spec = ProcessEventAssertionSpec(
        id="approval",
        event="tool_approval_requested",
        min_count=1,
        max_count=1,
    )
    order_spec = ProcessEventsInOrderAssertionSpec(
        id="protocol",
        events=("tool_approval_requested", "tool_approved"),
    )

    count_detail = _published_detail(
        count_spec,
        evaluate_assertion_spec(count_spec, evidence),
    )
    order_detail = _published_detail(
        order_spec,
        evaluate_assertion_spec(order_spec, evidence),
    )

    assert count_detail == PublishedProcessEventDetail(
        event="tool_approval_requested",
        min_count=1,
        max_count=1,
        matching_count=1,
    )
    assert order_detail == PublishedProcessEventsInOrderDetail(
        expected=("tool_approval_requested", "tool_approved"),
        actual_count=2,
        matched=True,
    )
    assert "private" not in str(count_detail.model_dump(mode="json"))
    assert "private" not in str(order_detail.model_dump(mode="json"))

    presentation = _present_assertion(
        PublishedAssertionResult(
            assertion_id=order_spec.id,
            assertion_revision=evaluate_assertion_spec(order_spec, evidence).assertion_revision,
            outcome="passed",
            score=1.0,
            code="passed",
            message="Assertion passed.",
            detail=order_detail,
        )
    )
    assert presentation.process == order_detail
    assert presentation.tool_json is None
