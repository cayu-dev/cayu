"""Synthetic offset diagnosis; these fixtures do not attribute incident wire data."""

import json
from dataclasses import replace

import pytest
from tests.core.test_openai_provider import RecordingTransport

from cayu import Message, ModelRequest, OpenAIProvider
from cayu.providers import ModelStreamEventType, OpenAIProtocolError, openai_response_events
from cayu.providers._openai_citation_offsets import CitationOffsetDiagnostic, citation_offset_fields
from cayu.providers._openai_protocol import protocol_exception_fields
from cayu.providers.openai import _OpenAIBackgroundOperationAdapter, _url_citation_event

CASES = [
    ({"start_index": 0}, "missing_endpoint", "integer", "absent"),
    ({"end_index": 2}, "missing_endpoint", "absent", "integer"),
    ({"start_index": None, "end_index": 2}, "missing_endpoint", "null", "integer"),
    ({"start_index": 0, "end_index": None}, "missing_endpoint", "integer", "null"),
    ({"start_index": True, "end_index": 2}, "non_integer_endpoint", "boolean", "integer"),
    ({"start_index": 0, "end_index": 2.0}, "non_integer_endpoint", "integer", "number"),
    ({"start_index": "sk-secret", "end_index": 2}, "non_integer_endpoint", "string", "integer"),
    (
        {"start_index": 0, "end_index": {"secret": "opaque"}},
        "non_integer_endpoint",
        "integer",
        "object",
    ),
    ({"start_index": [], "end_index": 2}, "non_integer_endpoint", "array", "integer"),
    ({"start_index": -1, "end_index": 2}, "negative_start", "integer", "integer"),
    ({"start_index": 2, "end_index": 2}, "empty_range", "integer", "integer"),
    ({"start_index": 2, "end_index": 1}, "reversed_range", "integer", "integer"),
    ({"start_index": 0, "end_index": 7}, "end_out_of_bounds", "integer", "integer"),
    ({"start_index": 0, "end_index": 10**1000}, "end_out_of_bounds", "integer", "integer"),
]


def annotation(offsets):
    return {"type": "url_citation", "url": "https://example.com", "title": "Source", **offsets}


def response(offsets):
    return {
        "id": "resp_synthetic",
        "model": "gpt-5.6",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "prefix "},
                    {
                        "type": "output_text",
                        "text": "é😀e\u0301中文",
                        "annotations": [annotation(offsets)],
                    },
                ],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }


def stream(offsets, incremental):
    raw = []
    if incremental:
        for i, part in enumerate(response(offsets)["output"][0]["content"]):
            raw.append(
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": i,
                    "delta": part["text"],
                }
            )
            for a in part.get("annotations", []):
                raw.append(
                    {
                        "type": "response.output_text.annotation.added",
                        "output_index": 0,
                        "content_index": i,
                        "annotation": a,
                    }
                )
    return [*raw, {"type": "response.completed", "response": response(offsets)}]


@pytest.mark.parametrize("offsets,condition,start_kind,end_kind", CASES)
def test_completed_offset_diagnosis(offsets, condition, start_kind, end_kind):
    with pytest.raises(OpenAIProtocolError) as caught:
        openai_response_events(response(offsets))
    fields = protocol_exception_fields(caught.value)
    assert fields["provider_protocol_reason"] == "citation_has_invalid_text_offsets"
    assert fields["provider_protocol_citation_condition"] == condition
    assert fields["provider_protocol_citation_start_kind"] == start_kind
    assert fields["provider_protocol_citation_end_kind"] == end_kind
    assert fields["provider_protocol_citation_text_length"] == 6
    assert fields["provider_protocol_citation_text_offset"] == 7
    assert "sk-secret" not in json.dumps(fields) + repr(vars(caught.value))
    assert "opaque" not in json.dumps(fields) + repr(vars(caught.value))
    assert len(json.dumps(fields)) < 1000


@pytest.mark.anyio
@pytest.mark.parametrize("offsets,condition,start_kind,end_kind", CASES)
async def test_stream_offset_diagnosis(offsets, condition, start_kind, end_kind):
    provider = OpenAIProvider(
        api_key="sk-secret",
        transport=RecordingTransport(stream_events=[stream(offsets, True)]),
    )
    events = [
        e
        async for e in provider.stream(
            ModelRequest(model="gpt-5.6", messages=[Message.text("user", "go")])
        )
    ]
    error = next(e for e in events if e.type == ModelStreamEventType.ERROR)
    assert error.payload["provider_protocol_citation_condition"] == condition
    assert error.payload["provider_protocol_citation_text_length"] == 6
    assert error.payload["provider_protocol_citation_text_offset"] == 7
    assert not any(e.type == ModelStreamEventType.COMPLETED for e in events)
    assert "sk-secret" not in json.dumps(error.payload)
    await provider.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("incremental", [False, True])
async def test_valid_unicode_multipart_completion_and_usage(incremental):
    offsets = {"start_index": 1, "end_index": 6}
    provider = OpenAIProvider(
        api_key="offline",
        transport=RecordingTransport(stream_events=[stream(offsets, incremental)]),
    )
    events = [
        e
        async for e in provider.stream(
            ModelRequest(model="gpt-5.6", messages=[Message.text("user", "go")])
        )
    ]
    if not incremental:
        events = openai_response_events(response(offsets))
    assert not any(e.type == ModelStreamEventType.ERROR for e in events)
    text = "".join(e.delta for e in events if e.type == ModelStreamEventType.TEXT_DELTA)
    citations = [e.payload for e in events if e.type == ModelStreamEventType.CITATION]
    assert len(citations) == 1
    assert text[citations[0]["start_index"] : citations[0]["end_index"]] == "😀e\u0301中文"
    assert events[-1].type == ModelStreamEventType.COMPLETED
    assert events[-1].payload["usage"]["input_tokens"] == 3
    assert events[-1].payload["usage"]["output_tokens"] == 4
    await provider.aclose()


@pytest.mark.parametrize(
    "offsets", [{}, {"start_index": None, "end_index": None}, {"start_index": 0, "end_index": 6}]
)
def test_existing_optional_pair_and_valid_range(offsets):
    event = _url_citation_event(annotation(offsets), text="Source", path="synthetic")
    assert event.type == ModelStreamEventType.CITATION


def test_diagnostic_projection_revalidates_and_background_preserves():
    diagnostic = CitationOffsetDiagnostic.invalid_offsets(
        {"start_index": 0, "end_index": 10**1000}, text_length=6, text_offset=7
    )
    assert (
        citation_offset_fields(diagnostic)["provider_protocol_citation_end_index_status"]
        == "outside_diagnostic_bounds"
    )
    for field, value in [
        ("condition", "secret"),
        ("start_kind", []),
        ("text_length", True),
        ("start_index", "sk-secret"),
        ("end_index", 10**1000),
    ]:
        assert citation_offset_fields(replace(diagnostic, **{field: value})) == {}
    provider = OpenAIProvider(api_key="offline")
    error = OpenAIProtocolError(
        "private text",
        reason_code="citation_has_invalid_text_offsets",
        citation_diagnostic=diagnostic,
    )
    safe = _OpenAIBackgroundOperationAdapter(provider)._safe_failure(error)
    assert protocol_exception_fields(safe) == protocol_exception_fields(error)


@pytest.mark.anyio
async def test_offset_diagnostics_survive_retry_and_durable_projection(tmp_path):
    from tests.core.test_openai_search_ordering import run_sse

    from cayu import EventType

    invalid = stream({"start_index": 0, "end_index": 7}, True)
    events, durable = await run_sse(tmp_path, [invalid, invalid])
    assert events[-1].type == EventType.SESSION_FAILED
    errors = [e for e in durable if e.type == EventType.MODEL_ERROR]
    assert len(errors) == 2
    assert all(
        e.payload["provider_protocol_citation_condition"] == "end_out_of_bounds" for e in errors
    )
    assert not any(e.type == EventType.MODEL_COMPLETED for e in durable)
