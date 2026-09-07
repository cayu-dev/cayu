"""Synthetic function-call controls for #1498, never incident wire data."""

from copy import deepcopy

import pytest

from cayu.providers import ModelStreamEventType
from cayu.providers.openai import OpenAIProtocolError, openai_stream_events


def added(index=0):
    return {
        "type": "response.output_item.added",
        "output_index": index,
        "item": {
            "type": "function_call",
            "id": f"fc_{index}",
            "call_id": f"call_{index}",
            "name": "echo",
            "arguments": "",
            "status": "in_progress",
        },
    }


def delta(index=0, value='{"text":"safe"}'):
    return {
        "type": "response.function_call_arguments.delta",
        "output_index": index,
        "item_id": f"fc_{index}",
        "delta": value,
    }


def arguments_done(index=0, value='{"text":"safe"}'):
    return {
        "type": "response.function_call_arguments.done",
        "output_index": index,
        "item_id": f"fc_{index}",
        "arguments": value,
    }


def done(index=0):
    item = {
        **added(index)["item"],
        "arguments": arguments_done(index)["arguments"],
        "status": "completed",
    }
    return {"type": "response.output_item.done", "output_index": index, "item": item}


def terminal(indexes=(0,)):
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_safe",
            "model": "gpt-5.6",
            "status": "completed",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "output": [done(i)["item"] for i in indexes],
        },
    }


def created():
    return {"type": "response.created", "response": {"id": "resp_safe"}}


def normal():
    return [created(), added(), delta(), arguments_done(), done(), terminal()]


async def parse(raw, seen):
    async def source():
        for event in raw:
            yield deepcopy(event)

    async for event in openai_stream_events(source()):
        seen.append(event)


FAILURES = [
    (
        [created(), arguments_done()],
        "function_call_arguments_done_arrived_before_output_item_added",
    ),
    (
        [created(), added(), arguments_done(1)],
        "function_call_arguments_done_arrived_before_output_item_added",
    ),
    (
        [created(), added(), arguments_done(), arguments_done()],
        "function_call_arguments_done_was_repeated",
    ),
    (
        [created(), added(), arguments_done(), delta()],
        "function_call_arguments_delta_arrived_after_arguments_done",
    ),
    (
        [created(), added(), arguments_done(), added()],
        "function_call_output_item_added_was_repeated",
    ),
    (
        [created(), added(), delta(), arguments_done(value='{"text":"changed"}')],
        "function_call_arguments_done_conflicts_with_streamed_arguments",
    ),
    (
        [created(), added(), {**arguments_done(), "name": "other"}],
        "function_call_arguments_done_name_mismatch",
    ),
    (
        [created(), added(), {**arguments_done(), "call_id": "other"}],
        "function_call_arguments_done_call_id_mismatch",
    ),
    (
        [created(), added(), {**arguments_done(), "response_id": "other"}],
        "stream_emitted_conflicting_response_identities",
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize("raw,reason", FAILURES)
async def test_function_ordering_failures(raw, reason):
    seen = []
    with pytest.raises(OpenAIProtocolError) as caught:
        await parse(raw, seen)
    assert caught.value.reason_code == reason
    assert not [e for e in seen if e.type == ModelStreamEventType.COMPLETED]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case",
    [
        "normal",
        "omitted_status",
        "empty_delta",
        "interleaved",
        "duplicate_item_done",
        "no_deltas",
        "empty_object",
        "terminal_fallback",
        "tail",
    ],
)
async def test_valid_function_streams(case):
    raw = normal()
    count = 1
    expected = {"text": "safe"}
    if case == "omitted_status":
        for event in raw:
            if "item" in event:
                event["item"].pop("status", None)
        raw[-1]["response"]["output"][0].pop("status", None)
    elif case == "empty_delta":
        raw[2] = delta(value="")
    elif case == "interleaved":
        raw = [
            created(),
            added(),
            added(1),
            delta(1),
            delta(),
            arguments_done(1),
            arguments_done(),
            done(1),
            done(),
            terminal((0, 1)),
        ]
        count = 2
    elif case == "duplicate_item_done":
        raw.insert(-1, done())
    elif case == "no_deltas":
        del raw[2]
    elif case == "empty_object":
        value = "{}"
        raw[2] = delta(value=value)
        raw[3] = arguments_done(value=value)
        raw[4]["item"]["arguments"] = value
        raw[5]["response"]["output"][0]["arguments"] = value
        expected = {}
    elif case == "terminal_fallback":
        raw[-1]["response"]["output"] = []
    elif case == "tail":
        raw.append(arguments_done())  # Completed responses close before reading a tail.
    seen = []
    await parse(raw, seen)
    calls = [e.payload for e in seen if e.type == ModelStreamEventType.TOOL_CALL]
    assert len(calls) == count
    assert {c["id"] for c in calls} == {f"call_{i}" for i in range(count)}
    assert all(c["name"] == "echo" and c["arguments"] == expected for c in calls)
    assert len([e for e in seen if e.type == ModelStreamEventType.COMPLETED]) == 1


def extra_failures():
    cases = [
        (
            [created(), added(), arguments_done(value="")],
            "streaming_function_call_is_missing_arguments",
        )
    ]
    message = {
        "type": "message",
        "id": "msg",
        "role": "assistant",
        "status": "in_progress",
        "content": [],
    }
    message_added = {"type": "response.output_item.added", "output_index": 0, "item": message}
    message_done = {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {**message, "status": "completed"},
    }
    for raw in [
        [created(), message_added, added()],
        [created(), added(), message_added],
        [created(), added(), arguments_done(), message_done],
    ]:
        cases.append((raw, "function_call_output_index_type_mismatch"))
    for kind in [delta, arguments_done]:
        cases.append(
            (
                [created(), added(), {**kind(), "item_id": "other"}],
                f"function_call_arguments_{'delta' if kind is delta else 'done'}_item_id_mismatch",
            )
        )
    for field, value, reason in [
        (
            "arguments",
            '{"text":"changed"}',
            "function_call_arguments_done_conflicts_with_streamed_arguments",
        ),
        ("name", "other", "function_call_arguments_done_name_mismatch"),
    ]:
        raw = normal()
        raw[3][field] = value
        raw[4]["item"][field] = value
        raw[5]["response"]["output"][0][field] = value
        cases.append((raw, reason))
    for field in ["id", "call_id", "name", "arguments"]:
        raw = normal()
        raw[-2]["item"][field] = "{}" if field == "arguments" else "other"
        cases.append((raw, "function_call_output_item_done_conflicts_with_streamed_arguments"))
        raw = normal()
        raw[-1]["response"]["output"][0][field] = "{}" if field == "arguments" else "other"
        cases.append((raw, "terminal_function_call_evidence_conflicts_with_lifecycle_evidence"))
    raw = normal()
    raw[-1]["response"]["output"] = [
        {"type": "message", "id": "msg", "role": "assistant", "status": "completed", "content": []}
    ]
    cases.append((raw, "terminal_response_omitted_completed_function_call_evidence"))
    raw = normal()
    raw[-1]["response"]["id"] = "other"
    cases.append((raw, "stream_emitted_conflicting_response_identities"))
    for field in ["id", "call_id"]:
        duplicate = added(1)
        duplicate["item"][field] = added()["item"][field]
        cases.append(([created(), added(), duplicate], "function_call_identity_was_reused"))
    cases.append(
        (
            [created(), added(), done()],
            "function_call_output_item_done_arrived_before_arguments_completion",
        )
    )
    cases.append(
        (
            [created(), added(), delta(), terminal()],
            "streaming_response_completed_with_unfinished_function_calls",
        )
    )
    cases.append(
        ([created(), added(), delta()], "streaming_response_ended_before_response_completed")
    )
    incomplete = {
        "type": "response.incomplete",
        "response": {
            "id": "resp_safe",
            "status": "incomplete",
            "output": [],
            "incomplete_details": {"reason": "max_output_tokens"},
        },
    }
    cases.append(
        (
            [created(), added(), incomplete, arguments_done()],
            "stream_event_arrived_after_terminal_response",
        )
    )
    return cases


@pytest.mark.anyio
@pytest.mark.parametrize("raw,reason", extra_failures())
async def test_identity_and_terminal_conflicts(raw, reason):
    seen = []
    with pytest.raises(OpenAIProtocolError) as caught:
        await parse(raw, seen)
    assert caught.value.reason_code == reason


async def run_sse(tmp_path, attempts):
    import httpx
    from tests.providers._responses_sse import ChunkedSSE

    from cayu import (
        AgentSpec,
        CayuApp,
        Message,
        OpenAIProvider,
        RetryPolicy,
        RunRequest,
        SQLiteSessionStore,
        Tool,
        ToolResult,
        ToolSpec,
    )
    from cayu.providers import HttpxOpenAITransport

    executed = []

    class Echo(Tool):
        spec = ToolSpec(
            name="echo",
            description="Synthetic echo",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        )

        async def run(self, ctx, args):
            executed.append(args)
            return ToolResult(content="ok")

    requests = []

    async def handler(request):
        assert len(requests) < len(attempts), "unexpected dispatch"
        raw = attempts[len(requests)]
        requests.append(request)
        return httpx.Response(
            200, stream=ChunkedSSE(raw), headers={"content-type": "text/event-stream"}
        )

    transport = HttpxOpenAITransport()
    database = tmp_path / "function-ordering.sqlite3"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport._client._client = client
        app = CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False)
        app.register_provider(
            OpenAIProvider(api_key="synthetic-key", transport=transport), default=True
        )
        app.register_agent(AgentSpec(name="assistant", model="gpt-5.6"), tools=[Echo()])
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="ordering",
                    messages=[Message.text("user", "go")],
                    retry_policy=RetryPolicy(
                        max_attempts=5, max_unknown_attempts=2, initial_delay_s=0
                    ),
                )
            )
        ]
    durable = await SQLiteSessionStore(database).load_events("ordering")
    assert len(requests) == len(attempts)
    return events, durable, executed


@pytest.mark.anyio
@pytest.mark.parametrize("raw,reason", FAILURES + extra_failures()[:-1])
async def test_native_sse_failures_cannot_execute_and_preserve_retry_receipts(
    tmp_path, raw, reason
):
    import json

    from cayu import EventType

    events, durable, executed = await run_sse(tmp_path, [raw, raw])
    assert executed == []
    assert not [
        e for e in durable if e.type in {EventType.TOOL_CALL_STARTED, EventType.MODEL_COMPLETED}
    ]
    errors = [e.payload for e in durable if e.type == EventType.MODEL_ERROR]
    public_errors = [e.payload for e in events if e.type == EventType.MODEL_ERROR]
    assert len(errors) == len(public_errors) == 2
    for error, public in zip(errors, public_errors, strict=True):
        assert error["provider_error_type"] == "protocol_error"
        assert error["provider_protocol_reason"] == reason
        fields = {k: v for k, v in error.items() if k.startswith("provider_protocol_")}
        assert fields == {k: v for k, v in public.items() if k.startswith("provider_protocol_")}
        trace = json.loads(fields["provider_protocol_stream_trace"])
        assert trace[0][0] == 1
        assert len(trace) <= 16
        assert "safe" not in repr(fields)
        assert "fc_0" not in repr(fields) and "call_0" not in repr(fields)
    started = [e.payload for e in durable if e.type == EventType.MODEL_STARTED]
    assert len(started) == 2
    assert len({e["model_attempt_id"] for e in started}) == 2
    assert len({e["model_step_id"] for e in started}) == 1
    assert [e["model_attempt_id"] for e in errors] == [e["model_attempt_id"] for e in started]
    assert [e["model_step_id"] for e in errors] == [e["model_step_id"] for e in started]
    assert [e["attempt"] for e in errors] == [1, 2]
    assert errors[-1]["effective_max_attempts"] == 2
    assert errors[-1]["retry_disposition"] == "unknown_provider_attempt_cap"
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.anyio
@pytest.mark.parametrize("omit_status", [False, True])
async def test_retry_state_is_fresh_and_valid_call_executes_once(tmp_path, omit_status):
    from cayu import EventType

    # Same synthetic item/call identities; an abandoned partial call cannot contaminate retry.
    bad = [created(), added(), delta(value="partial"), arguments_done(1)]
    good = normal()
    if omit_status:
        for event in good:
            if "item" in event:
                event["item"].pop("status", None)
        good[-1]["response"]["output"][0].pop("status", None)
    events, durable, executed = await run_sse(tmp_path, [bad, good, [terminal(())]])
    assert executed == [{"text": "safe"}]
    assert len([e for e in durable if e.type == EventType.MODEL_ERROR]) == 1
    assert len([e for e in durable if e.type == EventType.TOOL_CALL_STARTED]) == 1
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case,state,relation",
    [
        ("orphan", "absent", "unregistered"),
        ("repeated", "completed", "matches"),
        ("mismatch", "pending", "differs"),
    ],
)
async def test_trace_locates_registration_state(case, state, relation):
    import json

    from cayu.providers._openai_protocol import protocol_exception_fields

    raw = [created(), arguments_done()]
    if case == "repeated":
        raw = [created(), added(), arguments_done(), arguments_done()]
    elif case == "mismatch":
        raw = [created(), added(), {**arguments_done(), "item_id": "different"}]
    with pytest.raises(OpenAIProtocolError) as caught:
        await parse(raw, [])
    fields = protocol_exception_fields(caught.value)
    assert json.loads(fields["provider_protocol_stream_trace"])[-1][3:5] == [state, relation]


@pytest.mark.anyio
async def test_function_trace_is_bounded_and_contains_no_content(tmp_path):
    import json

    from cayu import EventType

    canary = "synthetic-sensitive-" + "x" * 5000
    raw = [created(), added(), *[delta(value=canary) for _ in range(20)], arguments_done(1)]
    _events, durable, executed = await run_sse(tmp_path, [raw, raw])
    assert executed == []
    for event in durable:
        if event.type != EventType.MODEL_ERROR:
            continue
        payload = event.payload
        trace = payload["provider_protocol_stream_trace"]
        assert payload["provider_protocol_stream_trace_truncated"] == 1
        assert len(json.loads(trace)) == 16
        assert len(trace.encode()) < 4096
        assert canary not in repr(payload)
        assert json.loads(trace)[-1][3:5] == ["absent", "unregistered"]


@pytest.mark.anyio
async def test_incomplete_tail_cannot_execute(tmp_path):
    from cayu import EventType

    raw, _reason = extra_failures()[-1]
    _events, durable, executed = await run_sse(tmp_path, [raw])
    assert executed == []
    assert not [e for e in durable if e.type == EventType.TOOL_CALL_STARTED]


@pytest.mark.anyio
async def test_interleaved_native_sse_executes_each_call_once(tmp_path):
    from cayu import EventType

    raw = [
        created(),
        added(),
        added(1),
        delta(1),
        delta(),
        arguments_done(1),
        arguments_done(),
        done(1),
        done(),
        terminal((0, 1)),
    ]
    events, durable, executed = await run_sse(tmp_path, [raw, [terminal(())]])
    assert executed == [{"text": "safe"}, {"text": "safe"}]
    assert len([e for e in durable if e.type == EventType.TOOL_CALL_STARTED]) == 2
    assert events[-1].type == EventType.SESSION_COMPLETED
