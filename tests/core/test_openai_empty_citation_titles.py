"""Blank optional citation metadata must not abort valid responses."""

import pytest
from tests.core.test_openai_provider import RecordingTransport

from cayu import (
    AgentSpec,
    CayuApp,
    CitationPart,
    EventType,
    Message,
    ModelRequest,
    OpenAIProvider,
    ProviderStatePart,
    RunRequest,
    SQLiteSessionStore,
)
from cayu.providers import OpenAIProtocolError, build_openai_payload, openai_response_events
from cayu.providers.openai import _url_citation_event


@pytest.mark.anyio
@pytest.mark.parametrize("title", [None, "", " \t\n", "Reference"])
@pytest.mark.parametrize("streamed", [False, True])
async def test_optional_title_survives_response_persistence_and_replay(tmp_path, title, streamed):
    annotation = {
        "type": "url_citation",
        "url": "https://example.com/reference",
        "start_index": 0,
        "end_index": 6,
    }
    if title is not None:
        annotation["title"] = title
    response = {
        "id": "resp_citation",
        "model": "gpt-5.6",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Source",
                        "annotations": [annotation],
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    assert openai_response_events(response)[-1].type.value == "completed"
    raw = []
    if streamed:
        raw = [
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "delta": "Source",
            },
            {
                "type": "response.output_text.annotation.added",
                "output_index": 0,
                "content_index": 0,
                "annotation": annotation,
            },
        ]
    raw.append({"type": "response.completed", "response": response})
    database = tmp_path / "citations.sqlite"
    store = SQLiteSessionStore(database)
    transport = RecordingTransport(stream_events=[raw])
    app = CayuApp(session_store=store, enable_logging=False)

    class TerminalResponseProvider(OpenAIProvider):
        async def stream(self, request):
            for event in openai_response_events(response):
                yield event

    provider_type = OpenAIProvider if streamed else TerminalResponseProvider
    app.register_provider(provider_type(api_key="offline", transport=transport), default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.6"))
    events = [
        e
        async for e in app.run(
            RunRequest(
                agent_name="assistant", session_id="citation", messages=[Message.text("user", "go")]
            )
        )
    ]
    assert not [e for e in events if e.type in {EventType.MODEL_ERROR, EventType.MODEL_RETRY}]
    assert len(transport.calls) == int(streamed)
    await store.close()
    reopened = SQLiteSessionStore(database)
    transcript = await reopened.load_transcript("citation")
    parts = [p for m in transcript for p in m.content if isinstance(p, CitationPart)]
    assert len(parts) == 1
    assert parts[0].title == (title if title and title.strip() else None)
    assert parts[0].url == annotation["url"]
    assert (parts[0].start_index, parts[0].end_index) == (0, 6)
    durable = await reopened.load_events("citation")
    assert len([e for e in durable if e.type == EventType.MODEL_CITATION]) == 1
    for neutral in [False, True]:
        messages = [
            m.model_copy(
                update={
                    "content": [
                        p for p in m.content if not (neutral and isinstance(p, ProviderStatePart))
                    ]
                }
            )
            for m in transcript
        ]
        payload = build_openai_payload(
            ModelRequest(model="gpt-5.6", messages=messages),
            reasoning_state="server" if neutral else "inline",
            chain=not neutral,
        )
        annotations = [
            a
            for i in payload["input"]
            if i.get("role") == "assistant"
            for c in i["content"]
            for a in c.get("annotations", [])
        ]
        assert len(annotations) == 1
        # Opaque provider state preserves wire metadata; parsing replay agrees
        # with the normalized transcript and neutral reconstruction.
        replayed = _url_citation_event(annotations[0], text="Source", path="replay")
        assert replayed.payload.get("title") == parts[0].title
        assert annotations[0]["url"] == parts[0].url
        assert (annotations[0]["start_index"], annotations[0]["end_index"]) == (0, 6)
    await reopened.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("title", 42),
        ("title", []),
        ("title", "x" * 1025),
        ("title", " " * 1025),
        ("url", "javascript:alert(1)"),
        ("url", 42),
        ("start_index", True),
        ("start_index", -1),
        ("end_index", 7),
        ("end_index", None),
    ],
)
def test_blank_title_preserves_strict_validation(field, value):
    annotation = {
        "type": "url_citation",
        "url": "https://example.com/",
        "title": "",
        "start_index": 0,
        "end_index": 6,
    }
    annotation[field] = value
    with pytest.raises(OpenAIProtocolError):
        _url_citation_event(annotation, text="Source", path="synthetic")
