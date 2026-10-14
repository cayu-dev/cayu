"""Admission copies and validation remain independent of store implementations."""

from dataclasses import FrozenInstanceError

import pytest

from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions import _interaction_transitions as rules


def started(**overrides):
    fields = dict(
        type=EventType.INTERACTION_STARTED,
        session_id="session",
        interaction_id="interaction",
        payload={"nested": {"value": "original"}},
    )
    return Event(**(fields | overrides))


@pytest.mark.parametrize("mode", ["create", "start", "continue"])
@pytest.mark.parametrize("defer", [False, True])
def test_admission_detaches_inputs_and_freezes_fields(mode, defer):
    event = started()
    messages = [Message.text("user", "original")]
    if mode == "create":
        admission = rules._copy_optional_interaction_admission(
            "session", event, messages, defer_transcript=defer
        )
    else:
        admission = rules._copy_transition_interaction_admission(
            "session",
            event if mode == "start" else None,
            messages,
            continued_interaction_id="interaction" if mode == "continue" else None,
            defer_interaction_source=defer,
        )
    assert admission is not None
    assert admission.interaction_id == "interaction"
    assert admission.defer_source is defer
    assert admission.source_messages[0] is not messages[0]
    assert isinstance(admission.source_messages, tuple)
    with pytest.raises(FrozenInstanceError):
        admission.interaction_id = "changed"
    event.payload["nested"]["value"] = "changed"
    event.interaction_id = "changed"
    messages[0] = Message.text("user", "changed")
    messages.clear()
    assert admission.source_messages == (Message.text("user", "original"),)
    if mode == "continue":
        assert admission.started_event is None
    else:
        assert admission.started_event is not None
        assert admission.started_event.interaction_id == "interaction"
        assert admission.started_event.payload == {"nested": {"value": "original"}}


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"session_id": "other"}, "different session"),
        ({"interaction_id": None}, "must have an interaction_id"),
        ({"type": EventType.SESSION_STARTED}, "must be interaction.started"),
    ],
)
def test_admission_rejects_invalid_start_identity(changes, error):
    with pytest.raises(ValueError, match=error):
        rules._copy_optional_interaction_admission(
            "session", started(**changes), [], defer_transcript=False
        )


@pytest.mark.parametrize(
    "changes, error_type, error",
    [
        ({"continued_interaction_id": "interaction"}, ValueError, "mutually exclusive"),
        ({"source_messages": None}, ValueError, "required for interaction admission"),
        (
            {"started_event": None, "source_messages": []},
            ValueError,
            "require an interaction identity",
        ),
        (
            {"started_event": None, "source_messages": None, "defer_interaction_source": True},
            ValueError,
            "require an interaction identity",
        ),
        (
            {"started_event": None, "continued_interaction_id": " padded "},
            ValueError,
            "continued_interaction_id",
        ),
        ({"defer_interaction_source": 1}, TypeError, "must be a bool"),
        ({"source_messages": ()}, TypeError, "must be a list"),
        ({"source_messages": [object()]}, TypeError, "Message"),
    ],
)
def test_admission_rejects_invalid_transition_inputs(changes, error_type, error):
    fields = dict(
        session_id="session",
        started_event=started(),
        source_messages=[],
        continued_interaction_id=None,
        defer_interaction_source=False,
    )
    with pytest.raises(error_type, match=error):
        rules._copy_transition_interaction_admission(**(fields | changes))


@pytest.mark.parametrize("event, messages", [(None, []), (started(), None)])
def test_creation_admission_requires_event_and_messages_together(event, messages):
    with pytest.raises(ValueError, match="supplied together"):
        rules._copy_optional_interaction_admission(
            "session", event, messages, defer_transcript=True
        )


def test_absent_admission_does_not_create_an_interaction():
    assert (
        rules._copy_optional_interaction_admission("session", None, None, defer_transcript=True)
        is None
    )
    assert (
        rules._copy_transition_interaction_admission(
            "session", None, None, continued_interaction_id=None, defer_interaction_source=False
        )
        is None
    )
