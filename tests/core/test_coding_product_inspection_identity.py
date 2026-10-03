"""Private call identity correlation must not depend on public presentation aliases."""

import pytest

from cayu import Event, EventType
from cayu.coding_products import _inspection_tool_call_ordinals


@pytest.mark.parametrize(
    "field", ["model_step_id", "model_attempt_id", "tool_round_id", "tool_call_id"]
)
@pytest.mark.parametrize("replacement", ["different", None, True, ""])
def test_inspection_call_correlation_binds_every_private_field(field, replacement):
    identity = dict(
        model_step_id="step", model_attempt_id="attempt", tool_round_id="round", tool_call_id="call"
    )
    started = Event(
        type=EventType.TOOL_CALL_STARTED,
        session_id="session",
        tool_name="read_file",
        payload=identity,
    )
    finished = Event(
        type=EventType.TOOL_CALL_COMPLETED,
        session_id="session",
        tool_name="read_file",
        payload=identity,
    )
    assert _inspection_tool_call_ordinals((started, finished), session_id="session") == (0, 0)
    finished.payload[field] = replacement
    assert _inspection_tool_call_ordinals((started, finished), session_id="session") == (
        0,
        1 if replacement == "different" else None,
    )
    assert _inspection_tool_call_ordinals((started,), session_id="another") == (None,)
