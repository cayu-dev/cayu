"""Credential-free SSE controls for #1495; these do not reproduce incident wire data."""

import json
from copy import deepcopy

import httpx
import pytest
from tests.providers._responses_sse import ChunkedSSE

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    OpenAIProvider,
    OpenAIWebSearch,
    RetryPolicy,
    RunRequest,
    SQLiteSessionStore,
)
from cayu.providers import HttpxOpenAITransport
from cayu.providers._openai_protocol import protocol_exception_fields
from cayu.providers._openai_search_trace import SearchStreamDiagnostic, SearchStreamTrace
from cayu.providers.openai import OpenAIProtocolError, _OpenAIBackgroundOperationAdapter

SECRET = "sk-synthetic-secret"


def added(index=0):
    return {
        "type": "response.output_item.added",
        "output_index": index,
        "item": {"type": "web_search_call", "id": f"{SECRET}-ws-{index}", "status": "in_progress"},
    }


def lifecycle(index=0, status="searching"):
    return {
        "type": f"response.web_search_call.{status}",
        "output_index": index,
        "item_id": f"{SECRET}-ws-{index}",
    }


def done(index=0):
    item = deepcopy(added(index)["item"])
    item.update(
        status="completed",
        action={
            "type": "search",
            "query": "synthetic query",
            "sources": [{"type": "url", "url": "https://example.com/", "title": "Example"}],
        },
    )
    return {"type": "response.output_item.done", "output_index": index, "item": item}


def created():
    return {"type": "response.created", "response": {"id": f"{SECRET}-response"}}


def terminal(indexes=(0,)):
    return {
        "type": "response.completed",
        "response": {
            "id": f"{SECRET}-response",
            "model": "gpt-5.6",
            "status": "completed",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "output": [
                *[done(index)["item"] for index in indexes],
                {
                    "type": "message",
                    "id": "msg_synthetic",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Done."}],
                },
            ],
        },
    }


def normal():
    return [
        created(),
        added(),
        lifecycle(status="in_progress"),
        lifecycle(),
        lifecycle(status="completed"),
        done(),
        terminal(),
    ]


async def run_sse(tmp_path, attempts):
    calls = []

    async def handler(request):
        assert len(calls) < len(attempts), "unexpected provider dispatch"
        raw = attempts[len(calls)]
        calls.append(request)
        return httpx.Response(
            200,
            stream=ChunkedSSE(raw),
            headers={
                "content-type": "text/event-stream",
            },
        )

    transport = HttpxOpenAITransport()
    database = tmp_path / "ordering.sqlite3"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport._client._client = client
        app = CayuApp(session_store=SQLiteSessionStore(database), enable_logging=False)
        app.register_provider(OpenAIProvider(api_key=SECRET, transport=transport), default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="gpt-5.6"), hosted_tools=[OpenAIWebSearch()]
        )
        events = [
            event
            async for event in app.run(
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
    assert len(calls) == len(attempts)
    assert not [event for event in durable if event.type == EventType.TOOL_CALL_STARTED]
    return events, durable


def diagnostics(events):
    return [event.payload for event in events if event.type == EventType.MODEL_ERROR]


@pytest.mark.anyio
@pytest.mark.parametrize("case", ["normal", "interleaved", "repeated", "missing_id", "tail"])
async def test_valid_hosted_search_sse(tmp_path, case):
    raw = normal()
    if case == "interleaved":
        raw = [
            created(),
            added(),
            added(1),
            lifecycle(1),
            lifecycle(),
            done(1),
            done(),
            terminal((0, 1)),
        ]
    elif case == "repeated":
        raw[4:4] = [lifecycle(), lifecycle(status="in_progress")]
    elif case == "missing_id":
        del raw[3]["item_id"]  # Existing compatibility remains explicit.
    elif case == "tail":
        raw.append(lifecycle())  # Accepted response.completed closes the stream.
    events, durable = await run_sse(tmp_path, [raw])
    assert not diagnostics(durable)
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len([e for e in durable if e.type == EventType.MODEL_COMPLETED]) == 1
    hosted = [e.payload for e in durable if e.type == EventType.MODEL_HOSTED_TOOL_CALL]
    finished = [e for e in hosted if e["status"] == "completed"]
    assert len(finished) == (2 if case == "interleaved" else 1)
    assert all(e["action"]["sources"][0]["url"] == "https://example.com/" for e in finished)


FAILURES = [
    (
        "orphan",
        [created(), lifecycle()],
        "web_search_lifecycle_arrived_before_output_item_added",
        "absent",
        "unregistered",
    ),
    (
        "index",
        [created(), added(), lifecycle(1)],
        "web_search_lifecycle_arrived_before_output_item_added",
        "absent",
        "unregistered",
    ),
    (
        "late",
        [created(), added(), done(), lifecycle(status="completed")],
        "web_search_lifecycle_arrived_after_output_item_done",
        "completed",
        "matches",
    ),
    (
        "duplicate_done",
        [created(), added(), done(), done()],
        "web_search_call_output_item_done_was_repeated",
        "completed",
        "matches",
    ),
    (
        "duplicate_added",
        [created(), added(), added()],
        "web_search_call_output_item_added_was_repeated",
        "pending",
        "matches",
    ),
    (
        "mismatch",
        [created(), added(), {**lifecycle(), "item_id": "different"}],
        "web_search_lifecycle_item_id_mismatch",
        "pending",
        "differs",
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize(("case", "raw", "reason", "state", "relation"), FAILURES)
async def test_ordering_diagnostics_survive_native_sse_retry_and_sqlite(
    tmp_path,
    case,
    raw,
    reason,
    state,
    relation,
):
    events, durable = await run_sse(tmp_path, [raw, raw])
    errors = diagnostics(durable)
    assert len(errors) == 2
    public_errors = diagnostics(events)
    assert len(public_errors) == 2
    assert [
        {k: v for k, v in error.items() if k.startswith("provider_protocol_")} for error in errors
    ] == [
        {k: v for k, v in error.items() if k.startswith("provider_protocol_")}
        for error in public_errors
    ]
    traces = []
    for error in errors:
        assert error["provider_error_type"] == "protocol_error"
        assert error["provider_protocol_reason"] == reason
        assert error["provider_protocol_stream_boundary"] == "native_adapter"
        trace = json.loads(error["provider_protocol_stream_trace"])
        traces.append(trace)
        assert trace[-1][3:5] == [state, relation]
        assert trace[0][0] == 1  # Fresh attempt-local trace.
        fields = {k: v for k, v in error.items() if k.startswith("provider_protocol_")}
        assert SECRET not in repr(fields)
        assert "synthetic query" not in repr(fields)
        assert "example.com" not in repr(fields)
    assert traces[0] == traces[1]
    assert [row[1] for row in traces[0]] == [event["type"] for event in raw]
    assert errors[-1]["retry_disposition"] == "unknown_provider_attempt_cap"
    assert not [e for e in durable if e.type == EventType.MODEL_COMPLETED]
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.anyio
async def test_failed_attempt_then_valid_stream_has_isolated_registration(tmp_path):
    events, durable = await run_sse(tmp_path, [[created(), added(), lifecycle(1)], normal()])
    assert len(diagnostics(durable)) == 1
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len([e for e in durable if e.type == EventType.MODEL_COMPLETED]) == 1


@pytest.mark.anyio
async def test_truncated_stream_preserves_unknown_hosted_outcome(tmp_path):
    raw = [created(), added(), lifecycle()]
    events, durable = await run_sse(tmp_path, [raw, raw])
    assert all(
        e["provider_protocol_reason"] == "streaming_response_ended_before_response_completed"
        for e in diagnostics(durable)
    )
    hosted = [e.payload["status"] for e in durable if e.type == EventType.MODEL_HOSTED_TOOL_CALL]
    assert hosted == ["in_progress", "searching", "outcome_unknown"] * 2
    assert events[-1].type == EventType.SESSION_FAILED


def test_trace_bounds_and_untrusted_values():
    trace = SearchStreamTrace()
    for _ in range(30):
        trace.record({"type": SECRET, "output_index": 10**100, "item_id": SECRET}, {}, {}, None)
    error = OpenAIProtocolError("safe", stream_diagnostic=trace.snapshot())
    fields = protocol_exception_fields(error)
    rows = json.loads(fields["provider_protocol_stream_trace"])
    assert len(rows) == 16
    assert rows[0][0] == 15
    assert fields["provider_protocol_stream_trace_truncated"] == 1
    assert all(row[1:3] == ["other", -1] for row in rows)
    assert SECRET not in repr(fields)
    assert len(fields["provider_protocol_stream_trace"]) < 4096
    for value in [
        SECRET,
        object(),
        SearchStreamDiagnostic(((1, SECRET, 0, "absent", "missing", "missing"),), False),
    ]:
        error.stream_diagnostic = value
        assert "provider_protocol_stream_trace" not in protocol_exception_fields(error)


@pytest.mark.anyio
async def test_bounded_trace_survives_native_sse_error(tmp_path):
    raw = [created(), added(), *[lifecycle() for _ in range(30)], lifecycle(1)]
    _, durable = await run_sse(tmp_path, [raw, normal()])
    error = diagnostics(durable)[0]
    assert error["provider_protocol_stream_trace_truncated"] == 1
    rows = json.loads(error["provider_protocol_stream_trace"])
    assert len(rows) == 16
    assert rows[-1] == [
        33,
        "response.web_search_call.searching",
        1,
        "absent",
        "unregistered",
        "missing",
    ]


def test_trace_correlates_response_and_missing_item_without_retaining_id():
    trace = SearchStreamTrace()
    trace.record(
        {**lifecycle(), "response_id": SECRET},
        {0: (lifecycle()["item_id"], "searching")},
        {},
        SECRET,
    )
    trace.record(
        {"type": "response.web_search_call.searching", "output_index": 0},
        {0: (SECRET, "searching")},
        {},
        SECRET,
    )
    trace.record({"type": "response.created", "response": {"id": "different"}}, {}, {}, SECRET)
    rows = trace.snapshot().entries
    assert rows[0][4:] == ("matches", "matches")
    assert rows[1][4:] == ("missing", "missing")
    assert rows[2][-1] == "differs"
    assert SECRET not in repr(rows)


def test_trace_rejects_hostile_string_subclass_without_invoking_it():
    class HostileString(str):
        def __hash__(self):
            raise AssertionError("untrusted string hashed")

        def strip(self):
            raise AssertionError("untrusted string stripped")

        def __eq__(self, other):
            raise AssertionError("untrusted string compared")

    trace = SearchStreamTrace()
    trace.record(
        {"type": HostileString("response.created"), "item_id": HostileString(SECRET)}, {}, {}, None
    )
    assert trace.snapshot().entries[0][1:] == ("other", -1, "absent", "invalid", "missing")
    error = OpenAIProtocolError(
        "safe",
        stream_diagnostic=SearchStreamDiagnostic(
            ((1, HostileString("response.created"), 0, "absent", "missing", "missing"),),
            False,
        ),
    )
    assert "provider_protocol_stream_trace" not in protocol_exception_fields(error)


def test_background_safe_exception_preserves_only_validated_trace():
    provider = OpenAIProvider(api_key=SECRET)
    adapter = _OpenAIBackgroundOperationAdapter(provider)
    trace = SearchStreamTrace()
    trace.record(lifecycle(), {}, {}, None)
    error = OpenAIProtocolError(
        SECRET,
        reason_code="web_search_lifecycle_arrived_before_output_item_added",
        stream_diagnostic=trace.snapshot(),
    )
    safe = adapter._safe_failure(error)
    assert protocol_exception_fields(safe) == protocol_exception_fields(error)
    assert SECRET not in str(safe)
    error.stream_diagnostic = SearchStreamDiagnostic(
        ((1, SECRET, 0, "absent", "missing", "missing"),),
        False,
    )
    safe = adapter._safe_failure(error)
    assert safe.stream_diagnostic is None
    assert "provider_protocol_stream_trace" not in protocol_exception_fields(safe)


def test_item_relation_uses_adapter_whitespace_normalization():
    trace = SearchStreamTrace()
    trace.record(
        {**lifecycle(), "item_id": "  ws_safe  "}, {0: ("ws_safe", "in_progress")}, {}, None
    )
    assert trace.snapshot().entries[0][4] == "matches"
