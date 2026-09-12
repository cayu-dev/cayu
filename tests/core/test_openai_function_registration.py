"""Native registration controls; synthetic sequences do not attribute incident wire order."""

import json
from copy import deepcopy

import pytest
from tests.core import test_openai_function_ordering as function
from tests.core import test_openai_search_ordering as search
from tests.providers._responses_ordering import run_ordering_attempts

from cayu import EventType


def other_added(kind, index=0):
    if kind == "web_search_call":
        return search.added(index)
    item = {"type": kind, "id": f"fixture-{kind}", "status": "in_progress"}
    if kind == "message":
        item.update(role="assistant", content=[])
    else:
        item["summary"] = []
    return {"type": "response.output_item.added", "output_index": index, "item": item}


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", ["api", "subscription"])
@pytest.mark.parametrize("kind", ["reasoning", "web_search_call", "message"])
@pytest.mark.parametrize("operation", ["added", "delta", "done"])
async def test_active_nonfunction_index_has_exact_type_diagnostic(
    tmp_path, adapter, kind, operation
):
    incoming = {
        "added": function.added(),
        "delta": function.delta(),
        "done": function.arguments_done(),
    }[operation]
    raw = [function.created(), other_added(kind), incoming]
    events, durable, calls = await run_ordering_attempts(tmp_path, adapter, [raw, raw])
    assert not calls
    assert not any(e.type == EventType.TOOL_CALL_STARTED for e in durable)
    errors = [e.payload for e in durable if e.type == EventType.MODEL_ERROR]
    assert len(errors) == 2
    for error in errors:
        assert error["provider_protocol_reason"] == "function_call_output_index_type_mismatch"
        types = json.loads(error["provider_protocol_stream_item_types"])
        assert types[-1][2] == kind
        trace = json.loads(error["provider_protocol_stream_trace"])
        assert trace[-1][3] == "pending"
        assert (
            error["provider_protocol_native_structure"]
            == error["provider_protocol_transport_structure"]
        )
        assert error["provider_protocol_native_structure_truncated"] == 0
        fields = {k: v for k, v in error.items() if k.startswith("provider_protocol_")}
        assert "fixture-" not in repr(fields)
        assert "safe" not in repr(fields)
        assert "fc_0" not in repr(fields)
    assert errors[-1]["retry_disposition"] == "unknown_provider_attempt_cap"
    assert errors[-1]["effective_max_attempts"] == 2
    assert events[-1].type == EventType.SESSION_FAILED


def mixed_stream():
    reasoning = other_added("reasoning", 2)
    reasoning_done = deepcopy(reasoning)
    reasoning_done["type"] = "response.output_item.done"
    reasoning_done["item"]["status"] = "completed"
    first = function.done(0)
    second = function.done(3)
    first["item"]["arguments"] = '{"text":"first"}'
    second["item"]["arguments"] = "{}"
    terminal = function.terminal(())
    terminal["response"]["output"] = [
        first["item"],
        search.done(1)["item"],
        reasoning_done["item"],
        second["item"],
        search.done(4)["item"],
    ]
    return [
        function.created(),
        function.added(0),
        search.added(1),
        reasoning,
        function.added(3),
        search.added(4),
        function.delta(0, '{"text":'),
        search.lifecycle(4),
        function.delta(3, ""),
        search.lifecycle(1, "in_progress"),
        {"type": "response.reasoning_summary_text.delta", "output_index": 2, "delta": "check"},
        function.delta(0, '"first"}'),
        search.lifecycle(1),
        search.lifecycle(1),
        function.arguments_done(3, "{}"),
        second,
        search.lifecycle(4, "completed"),
        search.done(4),
        function.arguments_done(0, '{"text":"first"}'),
        first,
        search.lifecycle(1, "completed"),
        search.done(1),
        reasoning_done,
        terminal,
    ]


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", ["api", "subscription"])
@pytest.mark.parametrize("retry", [False, True])
async def test_mixed_native_retry_preserves_two_functions_and_searches(tmp_path, adapter, retry):
    raw = mixed_stream()
    attempts = [raw, [function.terminal(())]]
    if retry:
        # Reuse the same IDs after abandoning partial arguments and a search.
        attempts.insert(
            0,
            [
                function.created(),
                function.added(0),
                search.added(1),
                function.delta(0, "abandoned"),
                function.arguments_done(9),
            ],
        )
    events, durable, calls = await run_ordering_attempts(tmp_path, adapter, attempts)
    assert sorted(calls, key=len) == [{}, {"text": "first"}]
    started = [e for e in durable if e.type == EventType.TOOL_CALL_STARTED]
    assert len(started) == 2
    assert {(e.payload["tool_call_id"], e.tool_name) for e in started} == {
        ("call_0", "echo"),
        ("call_3", "echo"),
    }
    completed = [
        e.payload
        for e in durable
        if e.type == EventType.MODEL_HOSTED_TOOL_CALL and e.payload["status"] == "completed"
    ]
    assert {e["call_id"] for e in completed} == {
        search.added(1)["item"]["id"],
        search.added(4)["item"]["id"],
    }
    assert len(completed) == 2
    assert all(e["action"]["sources"][0]["url"] == "https://example.com/" for e in completed)
    errors = [e.payload for e in durable if e.type == EventType.MODEL_ERROR]
    assert len(errors) == int(retry)
    if retry:
        assert errors[0]["provider_protocol_reason"] == (
            "function_call_arguments_done_arrived_before_output_item_added"
        )
        assert errors[0]["retry_disposition"] == "retry_scheduled"
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
@pytest.mark.parametrize("adapter", ["api", "subscription"])
async def test_shifted_function_identity_stays_visible_without_remapping(tmp_path, adapter):
    shifted = {**function.arguments_done(1), "item_id": function.added()["item"]["id"]}
    raw = [function.created(), function.added(), shifted]
    _events, durable, calls = await run_ordering_attempts(tmp_path, adapter, [raw, raw])
    assert not calls
    for event in durable:
        if event.type != EventType.MODEL_ERROR:
            continue
        error = event.payload
        assert error["provider_protocol_reason"] == (
            "function_call_arguments_done_arrived_before_output_item_added"
        )
        rows = json.loads(error["provider_protocol_native_structure"])
        assert rows == json.loads(error["provider_protocol_transport_structure"])
        assert rows[1][3] == 0 and rows[2][3] == 1
        assert rows[1][-1] == rows[2][-1] != 0
