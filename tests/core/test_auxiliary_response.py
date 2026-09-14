from __future__ import annotations

import json
import warnings

import pytest

from cayu.providers import ModelStreamEvent
from cayu.runtime._auxiliary_response import AuxiliaryResponseCollector


def test_response_collector_exact_byte_bound_and_detachment():
    events = [ModelStreamEvent.text_delta("héllo"), ModelStreamEvent.completed()]
    size = len(
        json.dumps(
            {"events": [e.model_dump(mode="json") for e in events]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    )
    collector = AuxiliaryResponseCollector(max_bytes=size)
    for event in events:
        collector.add(event)
    events[0].delta = "changed"
    assert collector.finish().text == "héllo"
    response = collector.finish()
    response.events[0].delta = "also changed"
    assert collector.finish().text == "héllo"
    too_small = AuxiliaryResponseCollector(max_bytes=size - 1)
    too_small.add(ModelStreamEvent.text_delta("héllo"))
    with pytest.raises(ValueError):
        too_small.add(ModelStreamEvent.completed())
    with pytest.raises(ValueError, match="did not complete"):
        too_small.finish()


@pytest.mark.parametrize(
    "event",
    [
        ModelStreamEvent.error("provider failure"),
        ModelStreamEvent.tool_call(name="nested", arguments={}, id="call"),
    ],
)
def test_response_collector_cannot_resume_after_rejection(event):
    collector = AuxiliaryResponseCollector(max_bytes=1024)
    with pytest.raises(ValueError, match="unsupported"):
        collector.add(event)
    with pytest.raises(ValueError, match="already terminal"):
        collector.add(ModelStreamEvent.completed())
    with pytest.raises(ValueError, match="did not complete"):
        collector.finish()


def test_response_collector_rejects_incomplete_and_postterminal_output():
    collector = AuxiliaryResponseCollector(max_bytes=1024)
    collector.add(ModelStreamEvent.text_delta("partial"))
    with pytest.raises(ValueError, match="did not complete"):
        collector.finish()
    collector.add(ModelStreamEvent.completed())
    with pytest.raises(ValueError, match="already terminal"):
        collector.add(ModelStreamEvent.text_delta("late"))
    with pytest.raises(ValueError, match="did not complete"):
        collector.finish()


def test_response_collector_accepts_bounded_error_only_as_failed_terminal():
    collector = AuxiliaryResponseCollector(max_bytes=1024)
    collector.add(ModelStreamEvent.text_delta("partial"))
    event = ModelStreamEvent.error("provider refused")
    event.payload["usage"] = {"input_tokens": 3}
    accepted = collector.accept_error(event)
    event.payload["usage"]["input_tokens"] = 999
    assert accepted.payload["usage"] == {"input_tokens": 3}
    with pytest.raises(ValueError, match="already terminal"):
        collector.accept_error(ModelStreamEvent.error("duplicate"))
    with pytest.raises(ValueError, match="did not complete"):
        collector.finish()
    small = AuxiliaryResponseCollector(max_bytes=20)
    with pytest.raises(ValueError):
        small.accept_error(ModelStreamEvent.error("too large"))
    with pytest.raises(ValueError, match="already terminal"):
        small.add(ModelStreamEvent.completed())


def test_response_collector_checks_hostile_values_before_serializing(capsys, caplog):
    class Hostile:
        def __repr__(self):
            raise AssertionError("secret-canary")

        __str__ = __repr__

    event = ModelStreamEvent.text_delta("valid")
    event.payload["hostile"] = Hostile()
    collector = AuxiliaryResponseCollector(max_bytes=1024)
    with warnings.catch_warnings(record=True) as caught, pytest.raises(ValueError) as failure:
        collector.add(event)
    assert "secret-canary" not in str(failure.value)
    assert not caught
    captured = capsys.readouterr()
    assert "secret-canary" not in captured.out + captured.err + caplog.text


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "1024", 8 * 1024 * 1024 + 1])
def test_response_collector_rejects_invalid_limits(limit):
    with pytest.raises(ValueError, match="Response byte limit"):
        AuxiliaryResponseCollector(max_bytes=limit)


@pytest.mark.parametrize("reason", ["tool_calls", "error"])
def test_response_collector_rejects_non_successful_completion(reason):
    collector = AuxiliaryResponseCollector(max_bytes=1024)
    with pytest.raises(ValueError, match="tool calls or an error"):
        collector.add(ModelStreamEvent.completed({"finish_reason": reason}))
    with pytest.raises(ValueError, match="did not complete"):
        collector.finish()


def test_response_collector_bounds_even_empty_event_accumulation():
    collector = AuxiliaryResponseCollector(max_bytes=1024)
    for _ in range(1024):
        try:
            collector.add(ModelStreamEvent.text_delta(""))
        except ValueError:
            break
    else:
        pytest.fail("empty events bypassed the retained response limit")
    with pytest.raises(ValueError, match="did not complete"):
        collector.finish()
