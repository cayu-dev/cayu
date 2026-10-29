"""The session reader over a Cloud Agent's own Cayu API, and what it adds to `CayuApi`."""

from __future__ import annotations

import asyncio
import gzip
import json

import httpx
import pytest

from cayu.cli import _cloud_agent_api as agent_api
from cayu.cli._agent_session_reader import AgentSessionReader
from cayu.cli._cloud_agent_api import CayuApi, CayuApiError
from cayu.sessions._call_inspection import InspectionWindow
from cayu.sessions.event_queries import EventQuery

_SESSIONS = {"sessions": [], "next_cursor": None, "total_count": 0}


def _api(handler) -> CayuApi:
    return CayuApi(
        "https://agent.example/cayu/",
        authorization="Basic secret",
        description="The Agent bug-hunter",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_client_bounds_answers(monkeypatch) -> None:
    monkeypatch.setattr(agent_api, "MAX_RESPONSE_BYTES", 64)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"events": ["x" * 100]})

    with pytest.raises(CayuApiError, match="larger than the CLI reads") as raised:
        _api(handler).get("/sessions/s/events", resource="session events")
    assert raised.value.category == "too_large"


def test_client_reads_compressed_answers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=gzip.compress(json.dumps(_SESSIONS).encode()),
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )

    assert _api(handler).get("/sessions", resource="sessions") == _SESSIONS


def test_client_never_repeats_a_detail_that_carries_a_credential() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "bad token Bearer sk-live-abcdefghijklmnop"})

    with pytest.raises(CayuApiError) as raised:
        _api(handler).get("/sessions", resource="sessions")
    assert "sk-live" not in str(raised.value)


def test_reader_sends_only_set_query_parameters_as_strings() -> None:
    seen: list[httpx.QueryParams] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params)
        return httpx.Response(200, json=_SESSIONS)

    from cayu.sessions.queries import SessionQuery

    asyncio.run(
        AgentSessionReader(_api(handler)).list_sessions(
            SessionQuery(limit=5, labels={"team": "core", "tier": "1"})
        )
    )

    params = seen[-1]
    assert params["limit"] == "5"
    assert "offset" not in params and "cursor" not in params
    assert params.get_list("label") == ["team=core", "tier=1"]


def test_reader_names_views_an_older_agent_does_not_serve() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sessions"):
            return httpx.Response(200, json=_SESSIONS)
        if request.url.path.endswith("/missing/tool-calls"):
            return httpx.Response(404, json={"detail": "Session not found"})
        return httpx.Response(404, json={"detail": "Not Found"})

    api = _api(handler)
    api.get("/sessions", resource="sessions")  # settles the API base, as a list does first
    reader = AgentSessionReader(api)
    with pytest.raises(CayuApiError, match="doesn't serve") as raised:
        asyncio.run(reader.tool_calls("s", InspectionWindow(), offset=0, limit=10))
    assert raised.value.category == "unsupported"
    with pytest.raises(ValueError, match="Session not found: missing"):
        asyncio.run(reader.tool_calls("missing", InspectionWindow(), offset=0, limit=10))


def test_reader_bounds_multi_type_event_reads_at_one_high_water_mark() -> None:
    def event(sequence: int, event_type: str) -> dict:
        return {
            "sequence": sequence,
            "id": f"e{sequence}",
            "type": event_type,
            "session_id": "s",
            "payload": {},
            "timestamp": "2026-10-28T12:00:00+00:00",
        }

    asked: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sessions"):
            return httpx.Response(200, json=_SESSIONS)
        params = dict(request.url.params)
        asked.append(params)
        if params.get("order_by") == "sequence_desc":
            return httpx.Response(200, json={"events": [event(4, "custom.b")], "has_more": True})
        if params["event_type"] == "custom.a":
            # A later append (sequence 7) must not be read past the high-water mark.
            return httpx.Response(200, json={"events": [event(1, "custom.a")], "has_more": False})
        return httpx.Response(200, json={"events": [event(3, "custom.b")], "has_more": False})

    reader = AgentSessionReader(_api(handler))
    records = asyncio.run(
        reader.query_events(
            EventQuery(session_id="s", event_types=("custom.a", "custom.b"), limit=10)
        )
    )

    assert [record.sequence for record in records] == [1, 3]
    assert {params.get("before_sequence") for params in asked[1:]} == {"5"}
