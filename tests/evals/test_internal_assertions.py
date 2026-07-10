from __future__ import annotations

import asyncio

import pytest

from cayu import (
    ChildSessionCompleted,
    EvalContext,
    Event,
    EventPayloadContains,
    EventType,
    Session,
    SessionStatus,
    SessionUsageSummary,
    Trajectory,
    UsageMetrics,
    UsageRecorded,
)


def _context(trajectory: Trajectory) -> EvalContext:
    return EvalContext(trajectory=trajectory, suite_id="suite", case_id="case", metadata={})


def _session(session_id: str, agent_name: str, status: SessionStatus) -> Session:
    return Session(
        id=session_id,
        agent_name=agent_name,
        provider_name="scripted",
        model="test-model",
        causal_budget_id="budget",
        status=status,
    )


def test_event_payload_contains_counts_nested_matches_and_reports_observations_on_failure():
    observed_payloads = [
        {"usage": {"total": 7, "input": 4}, "source": "provider"},
        {"usage": {"total": 7}, "source": "fallback"},
        {"usage": {"total": 8}},
    ]
    context = _context(
        Trajectory(
            events=(
                *(
                    Event(type=EventType.CONTEXT_COUNTED, session_id="session", payload=payload)
                    for payload in observed_payloads
                ),
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="session",
                    payload={"usage": {"total": 7}},
                ),
            )
        )
    )

    assertion = EventPayloadContains(
        EventType.CONTEXT_COUNTED,
        {"usage": {"total": 7}},
        min_count=2,
    )
    result = asyncio.run(assertion.evaluate(context))

    assert result.passed is True
    assert result.metadata == {
        "expected": {"usage": {"total": 7}},
        "event_type": "context.counted",
        "matching_count": 2,
    }

    mismatch = asyncio.run(
        EventPayloadContains(
            EventType.CONTEXT_COUNTED,
            {"usage": {"total": 7}},
            min_count=3,
        ).evaluate(context)
    )
    assert mismatch.passed is False
    assert "at least 3" in mismatch.message
    assert "got 2" in mismatch.message
    assert mismatch.metadata == {
        "expected": {"usage": {"total": 7}},
        "event_type": "context.counted",
        "matching_count": 2,
        "observed_payloads": observed_payloads,
    }


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (
            lambda: EventPayloadContains(EventType.MODEL_COMPLETED, []),
            TypeError,
            "expected must be a mapping",
        ),
        (
            lambda: EventPayloadContains(EventType.MODEL_COMPLETED, {}, min_count=True),
            TypeError,
            "min_count must be an integer",
        ),
        (
            lambda: EventPayloadContains(EventType.MODEL_COMPLETED, {}, min_count=-1),
            ValueError,
            "min_count must be non-negative",
        ),
    ],
)
def test_event_payload_contains_validates_inputs(factory, error, message):
    with pytest.raises(error, match=message):
        factory()


def test_usage_recorded_requires_a_summary_meeting_the_token_minimum():
    recorded = _context(
        Trajectory(
            usage_summary=SessionUsageSummary(
                session_id="session",
                usage=UsageMetrics(total_tokens=12),
            )
        )
    )

    result = asyncio.run(UsageRecorded(min_total_tokens=10).evaluate(recorded))

    assert result.passed is True
    assert result.metadata == {"total_tokens": 12, "minimum": 10}

    too_few = asyncio.run(UsageRecorded(min_total_tokens=13).evaluate(recorded))
    assert too_few.passed is False
    assert "12" in too_few.message
    assert "13" in too_few.message
    assert too_few.metadata == {"total_tokens": 12, "minimum": 13}

    missing = asyncio.run(UsageRecorded().evaluate(_context(Trajectory())))
    assert missing.passed is False
    assert "no usage summary" in missing.message.lower()


@pytest.mark.parametrize(
    ("minimum", "error", "message"),
    [
        (True, TypeError, "min_total_tokens must be an integer"),
        (-1, ValueError, "min_total_tokens must be non-negative"),
    ],
)
def test_usage_recorded_validates_the_token_minimum(minimum, error, message):
    with pytest.raises(error, match=message):
        UsageRecorded(min_total_tokens=minimum)


def test_child_session_completed_filters_direct_children_and_reports_incomplete_capture():
    completed_helper = Trajectory(
        session=_session("child-completed", "helper", SessionStatus.COMPLETED)
    )
    running_helper = Trajectory(session=_session("child-running", "helper", SessionStatus.RUNNING))
    completed_other = Trajectory(
        session=_session("child-other", "other", SessionStatus.COMPLETED),
        children=(
            Trajectory(session=_session("grandchild-completed", "helper", SessionStatus.COMPLETED)),
        ),
    )
    context = _context(
        Trajectory(
            children=(completed_helper, running_helper, completed_other, Trajectory()),
            children_incomplete=True,
        )
    )

    result = asyncio.run(ChildSessionCompleted(agent_name="helper", min_count=1).evaluate(context))

    assert result.passed is True
    assert result.metadata["matching_session_ids"] == ["child-completed"]
    assert result.metadata["matching_count"] == 1
    assert result.metadata["observed_children"] == [
        {"session_id": "child-completed", "status": "completed", "agent_name": "helper"},
        {"session_id": "child-running", "status": "running", "agent_name": "helper"},
        {"session_id": "child-other", "status": "completed", "agent_name": "other"},
        {"session_id": None, "status": None, "agent_name": None},
    ]

    assert asyncio.run(ChildSessionCompleted(min_count=2).evaluate(context)).passed is True

    mismatch = asyncio.run(
        ChildSessionCompleted(agent_name="helper", min_count=2).evaluate(context)
    )
    assert mismatch.passed is False
    assert "incomplete" in mismatch.message.lower()
    assert mismatch.metadata["matching_session_ids"] == ["child-completed"]
    assert mismatch.metadata["children_incomplete"] is True


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (
            lambda: ChildSessionCompleted(agent_name="  "),
            ValueError,
            "agent_name must be a non-empty string",
        ),
        (
            lambda: ChildSessionCompleted(min_count=True),
            TypeError,
            "min_count must be an integer",
        ),
        (
            lambda: ChildSessionCompleted(min_count=-1),
            ValueError,
            "min_count must be non-negative",
        ),
    ],
)
def test_child_session_completed_validates_filters(factory, error, message):
    with pytest.raises(error, match=message):
        factory()
