"""The completed terminal decision is a version-2 record beside unchanged version-1 records."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from cayu.events import Event, EventType
from cayu.sessions import _invocation_lifecycle as invocation_lifecycle_module
from cayu.sessions._execution_profile_checkpoint import ActiveInvocationExecutionProfile
from cayu.sessions._interaction_transitions import InteractionTransitionSpec
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalDecision,
    InvocationTerminalOutcome,
    _decision_digest,
    _decision_id,
    build_invocation_terminal_decision,
    build_owed_invocation_terminal_decision,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_event_id,
    owed_decision_event_id,
)
from cayu.sessions._paired_writes import (
    checkpoint_settling_owed_decision,
    owed_decision_install_mutation,
    require_owed_decision_authority,
    unsettled_owed_decision,
)
from cayu.sessions._terminal_evidence import (
    SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    queued_dispatch_terminal_event_id,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.checkpoints import (
    INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY,
)
from cayu.sessions.records import Session, SessionStatus

_IDENTITY = {"session_id": "s", "session_instance_id": "i", "run_epoch": 2, "interaction_id": "x"}
_OBSERVED_AT = datetime(2026, 9, 7, tzinfo=UTC)
_OPERATION = "c" * 64


def _interrupted() -> InvocationTerminalDecision:
    return build_invocation_terminal_decision(
        outcome=InvocationTerminalOutcome.INTERRUPTED,
        **_IDENTITY,
        profile_interaction_id="x",
        execution_profile_fingerprint="a" * 64,
        interaction_event_id=invocation_terminal_event_id(
            outcome=InvocationTerminalOutcome.INTERRUPTED,
            **_IDENTITY,
            source_id="r",
            event_kind="interaction",
        ),
        terminal_event_id=invocation_terminal_event_id(
            outcome=InvocationTerminalOutcome.INTERRUPTED,
            **_IDENTITY,
            source_id="r",
            event_kind="session",
        ),
        observed_at=_OBSERVED_AT,
        terminal_payload={"k": 1},
        interruption_request_id="r",
    )


def _failed() -> InvocationTerminalDecision:
    return build_invocation_terminal_decision(
        outcome=InvocationTerminalOutcome.FAILED,
        **_IDENTITY,
        profile_interaction_id="x",
        execution_profile_fingerprint="a" * 64,
        interaction_event_id="f:interaction_failed",
        terminal_event_id="f:session_failed",
        observed_at=_OBSERVED_AT,
        terminal_payload={},
        task_id="t",
        runtime_task_failure_id="f",
        task_terminalization_request_sha256="b" * 64,
        task_error_payload={"error": "e"},
        turn_completed_payload={"status": "failed"},
    )


def _completed(**overrides: Any) -> InvocationTerminalDecision:
    arguments: dict[str, Any] = {
        **_IDENTITY,
        "profile_interaction_id": "x",
        "execution_profile_fingerprint": "a" * 64,
        "interaction_event_id": "interaction-completed",
        "terminal_event_id": invocation_terminal_event_id(
            outcome=InvocationTerminalOutcome.COMPLETED,
            **_IDENTITY,
            source_id="interaction-completed",
            event_kind="session",
        ),
        "observed_at": _OBSERVED_AT,
        "terminal_payload": {},
        "turn_completed_payload": {"status": "completed"},
        "task_id": "t",
        "task_result_payload": {"session_id": "s"},
    }
    arguments.update(overrides)
    return build_owed_invocation_terminal_decision(
        outcome=InvocationTerminalOutcome.COMPLETED, **arguments
    )


@pytest.mark.parametrize(
    ("build", "decision_id", "record_digest"),
    [
        (
            _interrupted,
            "invocation-terminal-decision:v1:"
            "5c29a7b6a92d60d392612a1563784c46947a65be4f6e3a72138e592605c4abd3",
            "6954f9c94c36a6d661c213182293fd36ec73b0dd3c8ffccb9d5f92c64d66c526",
        ),
        (
            _failed,
            "invocation-terminal-decision:v1:"
            "d3104db677286f0318ae4421cc6e8affa7b64f98c1c038a602f0494bfdf6248d",
            "36161f05bb10dabd764dca029bcef59b014deb42ea3a09f2aa291ebd6150db18",
        ),
    ],
    ids=["interrupted", "failed"],
)
def test_version_one_records_keep_their_bytes_identity_and_digest(
    build, decision_id: str, record_digest: str
) -> None:
    decision = build()
    stored = decision.model_dump(mode="json")

    assert (decision.schema_version, decision.decision_id, decision.record_digest) == (
        1,
        decision_id,
        record_digest,
    )
    assert "task_result_payload" not in stored
    reloaded = invocation_terminal_decision_from_checkpoint(
        {INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: stored}
    )
    assert reloaded is not None and reloaded.model_dump(mode="json") == stored


@pytest.mark.parametrize("task_id", ["t", None])
def test_completed_decision_round_trips_as_version_two(task_id: str | None) -> None:
    decision = _completed(
        task_id=task_id, task_result_payload=None if task_id is None else {"session_id": "s"}
    )
    stored = decision.model_dump(mode="json")

    assert decision.schema_version == 2
    assert decision.decision_id.startswith("invocation-terminal-decision:v2:")
    assert ("task_result_payload" in stored) is (task_id is not None)
    assert (
        invocation_terminal_decision_from_checkpoint(
            {INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: stored}
        )
        == decision
    )
    assert owed_decision_event_id(decision, "task_completed") == (
        f"{decision.decision_id}:task_completed"
    )
    with pytest.raises(ValueError, match="Only version-2 decisions"):
        owed_decision_event_id(_failed(), "task_completed")


def test_completed_decision_adopts_a_queued_run_operation_event_id() -> None:
    decision = _completed(
        terminal_event_id=queued_dispatch_terminal_event_id(_OPERATION),
        terminal_payload={SESSION_RUN_OPERATION_ID_PAYLOAD_KEY: _OPERATION},
    )

    assert decision.terminal_event_id == queued_dispatch_terminal_event_id(_OPERATION)


def _resealed(decision: InvocationTerminalDecision, **updates: Any) -> dict[str, Any]:
    payload = decision.model_dump(mode="json")
    payload.update(updates)
    identity = {key: value for key, value in payload.items() if key != "record_digest"}
    identity.pop("decision_id")
    payload["decision_id"] = _decision_id(identity)
    payload.pop("record_digest")
    payload["record_digest"] = _decision_digest(payload)
    return payload


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": 1},
        {"turn_completed_payload": None},
        {"task_result_payload": None},
        {"task_error_payload": {"error": "e"}},
        {"interaction_event_id": None, "predecessor_interaction_event_id": "earlier"},
        {"terminal_event_id": "another-terminal-event"},
    ],
    ids=[
        "version-one",
        "no-turn-payload",
        "task-without-result",
        "failure-evidence",
        "no-interaction-event",
        "foreign-terminal-event",
    ],
)
def test_completed_decision_rejects_incomplete_or_foreign_authority(
    updates: dict[str, Any],
) -> None:
    with pytest.raises(RuntimeError, match="invalid"):
        invocation_terminal_decision_from_checkpoint(
            {INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: _resealed(_completed(), **updates)}
        )


@pytest.mark.parametrize("build", [_interrupted, _failed], ids=["interrupted", "failed"])
def test_failed_and_interrupted_records_cannot_use_version_two(build) -> None:
    with pytest.raises(RuntimeError, match="invalid"):
        invocation_terminal_decision_from_checkpoint(
            {INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: _resealed(build(), schema_version=2)}
        )


def _completed_session() -> tuple[Session, ActiveInvocationExecutionProfile]:
    decision = _completed()
    session = Session.model_construct(id="s", instance_id="i", status=SessionStatus.COMPLETED)
    profile = ActiveInvocationExecutionProfile.model_construct(
        session_id="s",
        interaction_id="x",
        run_epoch=3,
        profile=SimpleNamespace(fingerprint=decision.execution_profile_fingerprint),
    )
    return session, profile


def test_admission_refuses_an_owed_completed_predecessor() -> None:
    session, profile = _completed_session()

    with pytest.raises(SessionRunFenced, match="unsettled predecessor"):
        invocation_lifecycle_module._checkpoint_after_predecessor_terminal_decision(
            session,
            {INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: _completed().model_dump(mode="json")},
            expected_active_profile=profile,
        )


def _interaction_completed(**updates: Any) -> Event:
    fields: dict[str, Any] = {
        "id": "interaction-completed",
        "type": EventType.INTERACTION_COMPLETED,
        "session_id": "s",
        "interaction_id": "x",
        "timestamp": _OBSERVED_AT,
    }
    fields.update(updates)
    return Event(**fields)


def test_completion_transition_installs_only_its_own_completed_decision() -> None:
    mutation = owed_decision_install_mutation(_completed())
    spec = InteractionTransitionSpec(
        event=_interaction_completed(),
        from_statuses=(SessionStatus.RUNNING,),
        to_status=SessionStatus.COMPLETED,
        only_if_no_queued_messages=True,
        checkpoint_mutation=mutation,
    )

    assert spec.checkpoint_mutation == mutation
    for event, conditional in (
        (_interaction_completed(id="another-interaction-completed"), True),
        (_interaction_completed(timestamp=datetime(2026, 9, 8, tzinfo=UTC)), True),
        (_interaction_completed(), False),
    ):
        with pytest.raises(ValueError):
            InteractionTransitionSpec(
                event=event,
                from_statuses=(SessionStatus.RUNNING,),
                to_status=SessionStatus.COMPLETED,
                only_if_no_queued_messages=conditional,
                checkpoint_mutation=mutation,
            )
    with pytest.raises(ValueError, match="only install its own decision"):
        InteractionTransitionSpec(
            event=_interaction_completed(),
            from_statuses=(SessionStatus.RUNNING,),
            to_status=SessionStatus.COMPLETED,
            only_if_no_queued_messages=True,
            checkpoint_mutation={
                "operations": [
                    {
                        "key": "another_key",
                        "expected_value_digest": None,
                        "action": "set",
                        "value": {},
                    }
                ]
            },
        )
    with pytest.raises(ValueError, match="Only a version-2 decision"):
        owed_decision_install_mutation(_failed())


def test_release_settles_a_completed_decision_only_after_its_terminal_event() -> None:
    decision = _completed()
    checkpoint = {INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: decision.model_dump(mode="json")}
    session_completed = Event(
        id=decision.terminal_event_id, type=EventType.SESSION_COMPLETED, session_id="s"
    )

    for terminal_event in (
        None,
        session_completed.model_copy(update={"id": "another-terminal-event"}),
        session_completed.model_copy(update={"type": EventType.SESSION_FAILED}),
    ):
        with pytest.raises(SessionRunFenced, match="held until its terminal outcome"):
            checkpoint_settling_owed_decision(checkpoint, terminal_event=terminal_event)
    assert checkpoint_settling_owed_decision(checkpoint, terminal_event=session_completed) == {}
    interrupted = {
        INVOCATION_TERMINAL_DECISION_CHECKPOINT_KEY: _interrupted().model_dump(mode="json")
    }
    assert checkpoint_settling_owed_decision(interrupted, terminal_event=None) is interrupted
    assert unsettled_owed_decision(interrupted) is None


@pytest.mark.parametrize("run_epoch", [2, 3], ids=["owner", "another-epoch"])
def test_store_installs_only_the_completing_invocations_decision(run_epoch: int) -> None:
    session = Session.model_construct(id="s", instance_id="i", run_epoch=run_epoch)
    profile = ActiveInvocationExecutionProfile.model_construct(
        session_id="s",
        interaction_id="x",
        run_epoch=run_epoch,
        profile=SimpleNamespace(fingerprint="a" * 64),
    )

    def install() -> None:
        require_owed_decision_authority(
            to_status=SessionStatus.COMPLETED,
            checkpoint_mutation=owed_decision_install_mutation(_completed()),
            event=_interaction_completed(),
            session=session,
            checkpoint=None,
            active_profile=profile,
        )

    if run_epoch == _IDENTITY["run_epoch"]:
        install()
    else:
        with pytest.raises(SessionRunFenced, match="belong to the invocation it ends"):
            install()
