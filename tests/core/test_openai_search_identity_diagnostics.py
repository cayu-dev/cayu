"""Synthetic boundary evidence, never provider fault attribution or reconciliation."""

import json
from dataclasses import replace

import pytest
from tests.core.test_openai_search_ordering import (
    SECRET,
    added,
    created,
    diagnostics,
    done,
    lifecycle,
    normal,
    run_sse,
)

from cayu import EventType
from cayu.providers._openai_protocol import protocol_exception_fields
from cayu.providers._openai_search_trace import (
    ResponseStructureTrace,
    SearchStreamTrace,
)
from cayu.providers.openai import OpenAIProtocolError
from cayu.providers.openai_subscription import _safe_subscription_error_event


@pytest.mark.anyio
@pytest.mark.parametrize(
    "case", ["shifted", "unknown", "completed", "missing", "invalid", "duplicate"]
)
async def test_identity_evidence_crosses_http_api_and_durable_boundaries(tmp_path, case):
    end = {**lifecycle(4), "item_id": added()["item"]["id"], "sequence_number": 19}
    raw = [created(), added(), added(2), lifecycle(2), lifecycle()]
    expected = "pending_elsewhere"
    if case == "unknown":
        end["item_id"] = SECRET + "-unrelated"
        expected = "unknown"
    elif case == "completed":
        raw.append(done())
        expected = "completed_elsewhere"
    elif case == "missing":
        del end["item_id"]
        expected = "missing"
    elif case == "invalid":
        end["item_id"] = {"secret": SECRET}
        expected = "invalid"
    elif case == "duplicate":
        end = added(4)
        end["item"]["id"] = added()["item"]["id"]
    raw.append(end)
    retry = json.loads(json.dumps(raw).replace(SECRET, SECRET + "-retry"))
    events, durable = await run_sse(tmp_path, [raw, retry])
    errors = diagnostics(durable)
    assert len(errors) == 2
    assert [
        {k: v for k, v in error.items() if k.startswith("provider_protocol_")}
        for error in diagnostics(events)
    ] == [
        {k: v for k, v in error.items() if k.startswith("provider_protocol_")} for error in errors
    ]
    for error in errors:
        identities = json.loads(error["provider_protocol_stream_identities"])
        assert identities[-1][1] == expected
        assert identities[-1][2] == (0 if expected.endswith("elsewhere") else -1)
        native = json.loads(error["provider_protocol_native_structure"])
        transport = json.loads(error["provider_protocol_transport_structure"])
        assert native == transport
        assert native[0][0] == 1
        if case != "duplicate":
            assert native[-1][5:7] == ["present", 19]
        if case in {"shifted", "completed", "duplicate"}:
            assert native[-1][-1] == native[1][-1] == 1
        if case == "unknown":
            assert native[-1][-1] == 3
        assert json.loads(error["provider_protocol_stream_item_types"])[1][1] == "web_search_call"
        fields = {k: v for k, v in error.items() if k.startswith("provider_protocol_")}
        assert SECRET not in repr(fields)
        assert "synthetic query" not in repr(fields)
    assert (
        errors[0]["provider_protocol_native_structure"]
        == errors[1]["provider_protocol_native_structure"]
    )
    assert errors[-1]["retry_disposition"] == "unknown_provider_attempt_cap"
    assert events[-1].type == EventType.SESSION_FAILED


@pytest.mark.parametrize(
    "pending,completed,expected",
    [
        ({0: (SECRET, "searching"), 2: (SECRET, "searching")}, {}, "ambiguous"),
        ({0: (SECRET, "searching")}, {2: {"id": SECRET, "type": "web_search_call"}}, "ambiguous"),
        ({}, {0: {"id": SECRET, "type": "web_search_call"}}, "completed_here"),
        ({0: (SECRET, "searching")}, {}, "pending_here"),
    ],
)
def test_cross_identity_relationships(pending, completed, expected):
    trace = SearchStreamTrace()
    trace.record({**lifecycle(), "item_id": SECRET}, pending, completed, None)
    assert trace.snapshot().identities[-1][1] == expected


@pytest.mark.parametrize(
    "value,state",
    [
        (None, "invalid"),
        (True, "invalid"),
        (-1, "invalid"),
        (SECRET, "invalid"),
        (10**100, "out_of_range"),
        (0, "present"),
    ],
)
def test_upstream_sequence_is_untrusted_and_separate_from_ordinal(value, state):
    trace = ResponseStructureTrace()
    trace.record({"sequence_number": value})
    trace.record({})
    rows = trace.snapshot().entries
    assert rows[0][0] == 1
    assert rows[0][5:7] == (state, value if state == "present" else -1)
    assert rows[1][5:7] == ("missing", -1)


def test_alias_exhaustion_is_bounded_without_reuse_and_attempt_local():
    trace = ResponseStructureTrace()
    for index in range(129):
        trace.record({"item_id": f"{SECRET}-{index}"})
    trace.record({"item_id": f"{SECRET}-0"})
    snapshot = trace.snapshot()
    assert len(snapshot.entries) == 16
    assert snapshot.truncated and snapshot.exhausted
    assert snapshot.entries[-2][-2:] == ("exhausted", 0)
    assert snapshot.entries[-1][-2:] == ("present", 1)
    assert len(trace._aliases) == 128
    assert SECRET not in repr(vars(trace))
    fresh = ResponseStructureTrace()
    fresh.record({"item_id": f"{SECRET}-128"})
    assert fresh.snapshot().entries[-1][-2:] == ("present", 1)
    trace.record({"item_id": "x" * 1025})
    assert trace.snapshot().entries[-1][-2:] == ("oversized", 0)


def test_new_fields_revalidated_before_subscription_and_exception_projection():
    trace = SearchStreamTrace()
    trace.record(lifecycle(), {}, {}, None)
    good = trace.snapshot()
    error = OpenAIProtocolError(SECRET, stream_diagnostic=good)
    fields = protocol_exception_fields(error)
    projected = _safe_subscription_error_event(error, None, provider_name="openai-subscription")
    assert all(projected.payload[key] == value for key, value in fields.items())
    assert SECRET not in repr(projected)
    row = good.structure.entries[0]
    for bad in [
        replace(good, identities=((1, SECRET, -1),)),
        replace(good, identities=((True, "unknown", -1),)),
        replace(good, structure=replace(good.structure, entries=((1, SECRET, *row[2:]),))),
        replace(good, transport=replace(good.structure, entries=(row * 2,))),
        replace(good, structure=replace(good.structure, exhausted=SECRET)),
        replace(good, transport=replace(good.structure, entries=good.structure.entries * 17)),
    ]:
        error.stream_diagnostic = bad
        assert "provider_protocol_native_structure" not in protocol_exception_fields(error)
        assert (
            "provider_protocol_native_structure"
            not in _safe_subscription_error_event(
                error, None, provider_name="openai-subscription"
            ).payload
        )


@pytest.mark.anyio
async def test_malformed_identity_at_registered_index_keeps_diagnostic_and_acceptance(tmp_path):
    raw = [created(), added(), {**lifecycle(), "item_id": [SECRET]}]
    events, durable = await run_sse(tmp_path, [raw, normal()])
    error = diagnostics(durable)[0]
    assert error["provider_protocol_reason"] == "stream_field_must_be_a_string"
    assert json.loads(error["provider_protocol_stream_identities"])[-1][1] == "invalid"
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["search", "function"])
async def test_mixed_stream_retains_function_and_search_evidence(tmp_path, failure):
    from tests.core.test_openai_function_ordering import added as function_added
    from tests.core.test_openai_function_ordering import arguments_done

    end = lifecycle() if failure == "search" else arguments_done(2)
    raw = [created(), function_added(2), added(), {**end, "item_id": [SECRET]}]
    events, durable = await run_sse(tmp_path, [raw, normal()])
    error = diagnostics(durable)[0]
    assert error["provider_protocol_reason"] == "stream_field_must_be_a_string"
    assert json.loads(error["provider_protocol_stream_identities"])[-1][1] == "invalid"
    native = json.loads(error["provider_protocol_native_structure"])
    assert native == json.loads(error["provider_protocol_transport_structure"])
    assert native[-1][-2:] == ["invalid", 0]
    # Registration columns include a known conflicting item at the received index.
    trace = json.loads(error["provider_protocol_stream_trace"])
    types = json.loads(error["provider_protocol_stream_item_types"])
    assert trace[-1][3] == "pending"
    assert types[-1][2] == ("function_call" if failure == "function" else "web_search_call")
    fields = {k: v for k, v in error.items() if k.startswith("provider_protocol_")}
    assert fields == {
        k: v for k, v in diagnostics(events)[0].items() if k.startswith("provider_protocol_")
    }
    assert SECRET not in repr(fields)
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.anyio
async def test_decoded_objects_preserved_and_boundary_change_is_visible():
    import httpx
    from tests.providers._responses_sse import ChunkedSSE

    from cayu.providers import HttpxOpenAITransport
    from cayu.providers.openai import openai_stream_events

    raw = [created(), added(), lifecycle()]
    transport = HttpxOpenAITransport()

    async def handler(request):
        return httpx.Response(200, stream=ChunkedSSE(raw))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport._client._client = client
        decoded = [
            event
            async for event in transport.stream_response_events(
                url="https://example.com/v1/responses",
                headers={},
                payload={},
                timeout_s=30,
                transport_idle_timeout_s=30,
                protocol_idle_timeout_s=30,
                semantic_progress_timeout_s=30,
                absolute_stream_timeout_s=60,
            )
        ]
    assert decoded == raw  # Metadata never changes decoded JSON objects.
    decoded[-1]["output_index"] = 7

    async def changed():
        for event in decoded:
            yield event

    with pytest.raises(OpenAIProtocolError) as caught:
        _ = [event async for event in openai_stream_events(changed())]
    fields = protocol_exception_fields(caught.value)
    native = json.loads(fields["provider_protocol_native_structure"])
    wire = json.loads(fields["provider_protocol_transport_structure"])
    assert native[:-1] == wire[:-1]
    assert native[-1][3] == 7 and wire[-1][3] == 0
    assert native[-1][-1] == wire[-1][-1] == 1


def test_hostile_structure_fields_and_forged_transport_metadata():
    from cayu.providers._http import _trusted_sse_response_structure

    class HostileString(str):
        def __hash__(self):
            raise AssertionError("hashed untrusted subclass")

        def strip(self):
            raise AssertionError("stripped untrusted subclass")

    trace = SearchStreamTrace()
    trace.record(
        {
            "type": HostileString(SECRET),
            "item_id": HostileString(SECRET),
            "sequence_number": HostileString(SECRET),
            "item": {"id": HostileString(SECRET), "type": HostileString(SECRET)},
        },
        {},
        {},
        None,
    )
    good = trace.snapshot()
    assert good.structure.entries[-1][1:] == (
        "other",
        "missing",
        -1,
        "other",
        "invalid",
        -1,
        "invalid",
        0,
    )
    assert _trusted_sse_response_structure({"_response_structure": good.structure}) is None
    row = good.structure.entries[0]
    error = OpenAIProtocolError(
        "safe",
        stream_diagnostic=replace(
            good,
            structure=replace(good.structure, entries=((row[0], HostileString(SECRET), *row[2:]),)),
        ),
    )
    assert "provider_protocol_native_structure" not in protocol_exception_fields(error)
