"""Characterization of wait identity; public lifecycle coverage lives separately."""

import asyncio

import pytest
from pydantic import ValidationError

from cayu.core.events import Event, EventType
from cayu.core.messages import Message
from cayu.runtime._event_projection import prepare_new_runtime_event, project_runtime_event
from cayu.runtime._foreground_child_wait import (
    ForegroundChildResumeRequest,
    ForegroundChildTerminal,
    ForegroundChildWait,
    event_with_foreground_child_wait_authority,
    foreground_child_state_from_checkpoint,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import ResumeRequest
from cayu.vaults.redaction import SecretRedactor


def _wait_payload():
    return {
        "parent_effect": {
            "session_id": "parent",
            "session_instance_id": "parent-incarnation",
            "source_run_epoch": 3,
            "interaction_id": "parent-interaction",
            "model_step_id": "step",
            "model_attempt_id": "attempt",
            "tool_round_id": "round",
            "tool_call_id": "call",
            "agent_name": "parent-agent",
            "tool_name": "subagent",
            "idempotency_key": "runtime-spawn",
            "execution_profile_fingerprint": "a" * 64,
            "schema_digest": "b" * 64,
            "arguments_digest": "c" * 64,
        },
        "child_session_id": "child",
        "child_session_instance_id": "child-incarnation",
        "child_interaction_id": "child-interaction",
        "child_spawn_fingerprint": "sha256:" + "d" * 64,
        "child_action_kind": "tool_approval",
        "child_action_id": "approval",
        "child_action_run_epoch": 4,
        "revision": 1,
    }


def test_wait_round_trip_retains_complete_parent_effect_identity():
    wait = ForegroundChildWait.model_validate(_wait_payload())
    reconstructed = ForegroundChildWait.model_validate_json(wait.model_dump_json())
    assert reconstructed == wait
    assert reconstructed.parent_effect.model_dump() == wait.parent_effect.model_dump()
    assert reconstructed.delegated_action_reference() == {
        "child_session_id": "child",
        "action_kind": "tool_approval",
        "action_id": "approval",
        "status": "waiting_on_child_action",
    }


@pytest.mark.parametrize(
    "event_type", [EventType.SESSION_INTERRUPTED, EventType.SESSION_DELEGATED_ACTION_UPDATED]
)
def test_wait_discovery_identity_requires_runtime_publisher(event_type):
    payload = _wait_payload()
    payload["child_session_id"] = "private-child-prefix:generated-id"
    wait = ForegroundChildWait.model_validate(payload)
    redactor = SecretRedactor("private-child-prefix")
    event = Event(
        type=event_type,
        session_id=wait.parent_effect.session_id,
        payload={
            "interruption_type": "waiting_on_child_action",
            **wait.delegated_action_reference(),
        },
    )
    with pytest.raises(ValueError, match="child_session_id contains a workload secret"):
        prepare_new_runtime_event(event, redactor=redactor)
    attested = event_with_foreground_child_wait_authority(event, wait)
    prepared = prepare_new_runtime_event(attested, redactor=redactor)
    assert prepared.payload["child_session_id"] == wait.child_session_id
    public = project_runtime_event(prepared, sequence=1, redactor=redactor)
    assert "private-child-prefix" not in public.model_dump_json()
    # Serialization must not grant the raw public entrance publisher provenance.
    copied = Event.model_validate_json(event.model_dump_json())
    with pytest.raises(ValueError, match="child_session_id contains a workload secret"):
        prepare_new_runtime_event(copied, redactor=redactor)
    wrong_parent = event.model_copy(update={"session_id": "another-parent"})
    with pytest.raises(ValueError, match="authenticated wait"):
        event_with_foreground_child_wait_authority(wrong_parent, wait)


def test_private_child_continuation_cannot_enter_public_resume():
    async def scenario():
        app = CayuApp(enable_logging=False)
        request = ForegroundChildResumeRequest(session_id="parent", messages=[])
        with pytest.raises(TypeError):
            _ = [event async for event in app.resume(request)]
        assert await app.session_store.load("parent") is None

    asyncio.run(scenario())


def test_public_and_private_resume_have_distinct_input_contracts():
    with pytest.raises(ValidationError):
        ResumeRequest(session_id="parent", messages=[])
    with pytest.raises(ValidationError):
        ForegroundChildResumeRequest(
            session_id="parent", messages=[Message.text("user", "new input")]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("child_released_run_epoch", True),
        ("child_released_run_epoch", 4),
        ("event_id", ""),
        ("event_type", "session.resumed"),
        ("event_digest", "unknown"),
    ],
)
def test_terminal_selection_rejects_ambiguous_or_pre_pause_identity(field, value):
    payload = {
        "wait": _wait_payload(),
        "child_released_run_epoch": 6,
        "event_id": "terminal",
        "event_type": "session.completed",
        "event_digest": "e" * 64,
    }
    selected = ForegroundChildTerminal.model_validate(payload)
    assert ForegroundChildTerminal.model_validate_json(selected.model_dump_json()) == selected
    payload[field] = value
    with pytest.raises(ValidationError):
        ForegroundChildTerminal.model_validate(payload)


def test_checkpoint_rejects_orphan_conflicting_and_null_terminal_authority():
    wait = ForegroundChildWait.model_validate(_wait_payload())
    terminal = ForegroundChildTerminal(
        wait=wait,
        child_released_run_epoch=6,
        event_id="terminal",
        event_type="session.completed",
        event_digest="e" * 64,
    )
    state = {
        "foreground_child_wait": wait.model_dump(mode="json"),
        "foreground_child_terminal": terminal.model_dump(mode="json"),
    }
    assert foreground_child_state_from_checkpoint(state) == (wait, terminal)
    with pytest.raises(RuntimeError, match="no exact retained wait"):
        foreground_child_state_from_checkpoint(
            {"foreground_child_terminal": state["foreground_child_terminal"]}
        )
    with pytest.raises(ValidationError):
        foreground_child_state_from_checkpoint({"foreground_child_wait": None})
    with pytest.raises(ValidationError):
        foreground_child_state_from_checkpoint({**state, "foreground_child_terminal": None})
    changed_wait = wait.model_copy(update={"child_action_id": "another-action"})
    with pytest.raises(RuntimeError, match="no exact retained wait"):
        foreground_child_state_from_checkpoint(
            {**state, "foreground_child_wait": changed_wait.model_dump(mode="json")}
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1.0),
        ("schema_version", 2),
        ("child_session_id", ""),
        ("child_action_id", True),
        ("child_interaction_id", None),
        ("child_spawn_fingerprint", "unknown"),
        ("child_action_kind", "future-action"),
        ("child_action_run_epoch", True),
        ("child_action_run_epoch", -1),
        ("revision", True),
        ("revision", 0),
        ("question", "child-content-canary"),
    ],
)
def test_wait_rejects_ambiguous_identity_and_child_content(field, value):
    payload = _wait_payload()
    payload[field] = value
    with pytest.raises(ValidationError) as raised:
        ForegroundChildWait.model_validate(payload)
    assert "child-content-canary" not in str(raised.value)
