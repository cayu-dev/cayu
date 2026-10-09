"""A replay presents exactly the idempotency key its interrupted attempt used."""

from __future__ import annotations

import pytest

from cayu.events import Event, EventType
from cayu.runtime._abandoned_session_recovery import (
    _abandoned_execution,
    abandoned_running_execution,
)
from cayu.runtime._tool_call_replay import started_dispatch_identities
from cayu.runtime._tool_identity import tool_idempotency_key
from cayu.sessions.base import Session, SessionStatus


def _started(call_id: str, *, approval_id=None, input_id=None, key=None) -> Event:
    payload: dict[str, object] = {
        "tool_call_id": call_id,
        "idempotency_key": key
        or tool_idempotency_key(
            session_id="s",
            tool_call_id=call_id,
            tool_round_id="round",
            approval_id=approval_id,
            pause_id=input_id,
        ),
    }
    if approval_id is not None:
        payload["approval_id"] = approval_id
    if input_id is not None:
        payload["input_id"] = input_id
    return Event(type=EventType.TOOL_CALL_STARTED, session_id="s", payload=payload)


def test_identities_carry_the_pause_and_approval_of_the_original_dispatch() -> None:
    events = [
        _started("plain"),
        _started("after-pause", input_id="input-1"),
        _started("after-approval", approval_id="approval-1"),
    ]

    identities = started_dispatch_identities(
        events,
        session_id="s",
        tool_round_id="round",
        call_ids=("plain", "after-pause", "after-approval"),
    )

    assert identities == {
        "plain": (None, None),
        "after-pause": (None, "input-1"),
        "after-approval": ("approval-1", None),
    }
    # Each identity reproduces the key the interrupted attempt recorded.
    for event in events:
        call_id = event.payload["tool_call_id"]
        approval_id, input_id = identities[call_id]
        assert event.payload["idempotency_key"] == tool_idempotency_key(
            session_id="s",
            tool_call_id=call_id,
            tool_round_id="round",
            approval_id=approval_id,
            pause_id=input_id,
        )


@pytest.mark.parametrize(
    "events",
    [
        [],
        [_started("call", key="cayu-tool:v1:" + "0" * 64)],
        [
            _started("call", input_id="input-1", key=None).model_copy(
                update={
                    "payload": {
                        "tool_call_id": "call",
                        "idempotency_key": tool_idempotency_key(
                            session_id="s",
                            tool_call_id="call",
                            tool_round_id="round",
                            pause_id="input-1",
                        ),
                    }
                }
            )
        ],
    ],
    ids=["no-start-event", "unreproducible-key", "pause-identity-missing"],
)
def test_unprovable_identity_refuses_the_replay(events: list[Event]) -> None:
    assert (
        started_dispatch_identities(
            events, session_id="s", tool_round_id="round", call_ids=("call",)
        )
        is None
    )


def test_only_the_taken_over_session_may_elect_a_replay() -> None:
    def session(session_id: str, instance_id: str, run_epoch: int) -> Session:
        return Session.model_construct(
            id=session_id,
            instance_id=instance_id,
            run_epoch=run_epoch,
            status=SessionStatus.RUNNING,
        )

    taken_over = session("s", "incarnation-1", 1)
    with _abandoned_execution(taken_over, replays_tool_calls=True):
        # Recovery fences the run, so the session it elects for is a later epoch.
        assert abandoned_running_execution(session("s", "incarnation-1", 2))
        assert not abandoned_running_execution(session("other", "incarnation-1", 1))
        assert not abandoned_running_execution(session("s", "incarnation-2", 1))
    with _abandoned_execution(taken_over):
        assert not abandoned_running_execution(taken_over)
    assert not abandoned_running_execution(taken_over)
