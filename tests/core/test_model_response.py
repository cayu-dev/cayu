from __future__ import annotations

import pytest

from cayu.providers import ModelStreamEvent
from cayu.providers.response import ModelResponse


def test_response_is_detached_ordered_and_reconstructable():
    events = [
        ModelStreamEvent.thinking("reason"),
        ModelStreamEvent.text_delta("hello"),
        ModelStreamEvent.tool_call(name="first", arguments={"x": [1]}, id="a"),
        ModelStreamEvent.text_delta(" world"),
        ModelStreamEvent.tool_call(name="second", arguments={}, id="b"),
        ModelStreamEvent.completed({"finish_reason": "tool_calls", "usage": {"output_tokens": 8}}),
    ]
    response = ModelResponse(events=events)
    events[1].delta = "changed"
    events[-1].payload["usage"]["output_tokens"] = 999
    assert response.text == "hello world"
    assert response.thinking == "reason"
    assert [call["name"] for call in response.tool_calls] == ["first", "second"]
    response.tool_calls[0]["arguments"]["x"].append(2)
    assert response.tool_calls[0]["arguments"]["x"] == [1]
    assert response.payload["usage"]["output_tokens"] == 8
    assert response.completion.finish_reason == "tool_calls"
    assert ModelResponse.model_validate_json(response.model_dump_json()) == response


@pytest.mark.parametrize(
    "events",
    [
        [],
        [ModelStreamEvent.text_delta("partial")],
        [ModelStreamEvent.completed(), ModelStreamEvent.completed()],
        [ModelStreamEvent.completed(), ModelStreamEvent.text_delta("late")],
        [ModelStreamEvent.error("failed"), ModelStreamEvent.completed()],
    ],
)
def test_response_cannot_fabricate_success_from_incomplete_or_failed_stream(events):
    with pytest.raises(ValueError, match="exactly one final completion"):
        ModelResponse(events=events)
