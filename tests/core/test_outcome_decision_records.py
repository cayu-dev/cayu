"""Failed and interrupted outcomes record what they owe as version-2 decisions."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from cayu.events import (
    Event,
    EventType,
    event_nested_payload_authority_is_runtime_generated,
    event_payload_authority_is_runtime_generated,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
    runtime_payload_authority_paths,
)
from cayu.runtime._outcome_decision_settlement import owed_terminal_event, owed_turn_event
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalOutcome,
    build_owed_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_event_id,
    owed_decision_event_id,
)
from cayu.sessions._paired_writes import (
    checkpoint_settling_owed_decision,
    owed_decision_from_install_mutation,
    owed_decision_install_mutation,
    owed_decision_of_status,
    require_owed_outcome_kept,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY
from cayu.sessions.records import Session, SessionStatus
from cayu.storage import migrations as m

_IDENTITY = {"session_id": "s", "session_instance_id": "i", "run_epoch": 2, "interaction_id": "x"}
_OBSERVED_AT = datetime(2026, 11, 9, tzinfo=UTC)
_STATUS = {
    InvocationTerminalOutcome.FAILED: SessionStatus.FAILED,
    InvocationTerminalOutcome.INTERRUPTED: SessionStatus.INTERRUPTED,
}
_INTERACTION_EVENT = {
    InvocationTerminalOutcome.FAILED: EventType.INTERACTION_FAILED,
    InvocationTerminalOutcome.INTERRUPTED: EventType.INTERACTION_PAUSED,
}
_SESSION_EVENT = {
    InvocationTerminalOutcome.FAILED: EventType.SESSION_FAILED,
    InvocationTerminalOutcome.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}
_OUTCOMES = sorted(_STATUS, key=str)


def _decided(outcome: InvocationTerminalOutcome, **overrides: Any):
    arguments: dict[str, Any] = {
        **_IDENTITY,
        "outcome": outcome,
        "profile_interaction_id": "x",
        "execution_profile_fingerprint": "a" * 64,
        "interaction_event_id": "interaction-ended",
        "terminal_event_id": invocation_terminal_event_id(
            outcome=outcome, **_IDENTITY, source_id="interaction-ended", event_kind="session"
        ),
        "observed_at": _OBSERVED_AT,
        "terminal_payload": {"approval_id": "approval", "approval": {"approval_id": "approval"}},
        "turn_completed_payload": {"status": _STATUS[outcome].value, "interaction_ids": ["x"]},
        "terminal_payload_authority": ("approval_id",),
        "terminal_nested_payload_authority": (("approval", "approval_id"),),
    }
    arguments.update(overrides)
    return build_owed_invocation_terminal_decision(**arguments)


def _interaction_event(outcome: InvocationTerminalOutcome, **updates: Any) -> Event:
    fields: dict[str, Any] = {
        "id": "interaction-ended",
        "type": _INTERACTION_EVENT[outcome],
        "session_id": "s",
        "interaction_id": "x",
        "timestamp": _OBSERVED_AT,
    }
    fields.update(updates)
    return Event(**fields)


@pytest.mark.parametrize("outcome", _OUTCOMES)
def test_failed_and_interrupted_decisions_round_trip_as_version_two(
    outcome: InvocationTerminalOutcome,
) -> None:
    decision = _decided(outcome)
    stored = decision.model_dump(mode="json")

    assert decision.schema_version == 2
    assert decision.decision_id.startswith("invocation-terminal-decision:v2:")
    assert (
        invocation_terminal_decision_from_checkpoint(
            {INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: stored}
        )
        == decision
    )
    assert owed_decision_event_id(decision, "turn_completed") == (
        f"{decision.decision_id}:turn_completed"
    )
    with pytest.raises(ValueError, match="Only version-2 decisions"):
        owed_decision_event_id(decision, "task_completed")


@pytest.mark.parametrize("outcome", _OUTCOMES)
@pytest.mark.parametrize(
    "overrides",
    [
        {"task_id": "t", "task_result_payload": {"session_id": "s"}},
        {"terminal_event_id": "another-terminal-event"},
        {"turn_completed_payload": None},
    ],
    ids=["task", "foreign-terminal-event", "no-turn"],
)
def test_failed_and_interrupted_decisions_carry_only_their_own_authority(
    outcome: InvocationTerminalOutcome, overrides: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        _decided(outcome, **overrides)


@pytest.mark.parametrize("outcome", _OUTCOMES)
def test_a_terminal_transition_installs_only_the_decision_of_its_own_outcome(
    outcome: InvocationTerminalOutcome,
) -> None:
    decision = _decided(outcome)
    mutation = owed_decision_install_mutation(decision)
    event = _interaction_event(outcome)

    assert owed_decision_from_install_mutation(mutation, event=event, to_status=_STATUS[outcome])
    other = next(status for status in _STATUS.values() if status is not _STATUS[outcome])
    for wrong_event, status in (
        (event, other),
        (event, SessionStatus.COMPLETED),
        (_interaction_event(outcome, id="another"), _STATUS[outcome]),
        (
            _interaction_event(outcome, timestamp=datetime(2026, 11, 10, tzinfo=UTC)),
            _STATUS[outcome],
        ),
    ):
        with pytest.raises(ValueError, match="belongs to another interaction outcome"):
            owed_decision_from_install_mutation(mutation, event=wrong_event, to_status=status)


@pytest.mark.parametrize("outcome", _OUTCOMES)
def test_release_settles_an_owed_outcome_only_after_its_own_session_event(
    outcome: InvocationTerminalOutcome,
) -> None:
    decision = _decided(outcome)
    checkpoint = {INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: decision.model_dump(mode="json")}
    published = Event(id=decision.terminal_event_id, type=_SESSION_EVENT[outcome], session_id="s")

    for terminal_event in (
        None,
        published.model_copy(update={"id": "another"}),
        published.model_copy(update={"type": EventType.SESSION_COMPLETED}),
    ):
        with pytest.raises(SessionRunFenced, match="held until its terminal outcome"):
            checkpoint_settling_owed_decision(checkpoint, terminal_event=terminal_event)
    assert checkpoint_settling_owed_decision(checkpoint, terminal_event=published) == {}


@pytest.mark.parametrize("outcome", _OUTCOMES)
def test_a_session_keeps_its_decided_status_while_its_outcome_is_owed(
    outcome: InvocationTerminalOutcome,
) -> None:
    decision = _decided(outcome)
    checkpoint = {INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: decision.model_dump(mode="json")}
    status = _STATUS[outcome]

    assert owed_decision_of_status(checkpoint, status) == decision
    assert owed_decision_of_status(checkpoint, SessionStatus.COMPLETED) is None
    with pytest.raises(SessionRunFenced, match="keeps its terminal status"):
        require_owed_outcome_kept(status, SessionStatus.RUNNING, checkpoint)
    require_owed_outcome_kept(status, status, checkpoint)
    require_owed_outcome_kept(SessionStatus.RUNNING, SessionStatus.FAILED, checkpoint)


def test_rebuilt_events_carry_the_payload_authority_their_producer_attested() -> None:
    template = event_with_runtime_nested_payload_authority(
        event_with_runtime_payload_authority(
            Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id="s",
                payload={"approval_id": "approval", "approval": {"approval_id": "approval"}},
            ),
            "approval_id",
        ),
        ("approval", "approval_id"),
    )
    fields, paths = runtime_payload_authority_paths(template)
    decision = _decided(
        InvocationTerminalOutcome.INTERRUPTED,
        terminal_payload=template.payload,
        terminal_payload_authority=fields,
        terminal_nested_payload_authority=paths,
    )
    session = Session.model_construct(id="s", agent_name="agent", environment_name=None)

    rebuilt = owed_terminal_event(decision, session)

    assert (rebuilt.id, rebuilt.type, rebuilt.timestamp) == (
        decision.terminal_event_id,
        EventType.SESSION_INTERRUPTED,
        _OBSERVED_AT,
    )
    assert event_payload_authority_is_runtime_generated(
        rebuilt, field_name="approval_id", value="approval"
    )
    assert event_nested_payload_authority_is_runtime_generated(
        rebuilt, path=("approval", "approval_id"), value="approval"
    )
    turn = owed_turn_event(decision, session)
    assert (turn.id, turn.payload) == (
        f"{decision.decision_id}:turn_completed",
        decision.turn_completed_payload,
    )


def test_failed_and_interrupted_decisions_fence_old_binaries() -> None:
    current = m.SchemaState(revision=145, compatible_from=145)
    with pytest.raises(m.SchemaTooNew):
        m.validate(current, app_latest=144, app_min_supported=144)
    with pytest.raises(m.SchemaTooOld):
        m.validate(m.SchemaState(revision=144, compatible_from=144))
    assert (
        m.validate_migration_input(m.SchemaState(revision=144, compatible_from=144))[0].revision
        == 145
    )
