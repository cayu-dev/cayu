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


def test_reader_names_rollups_an_older_agent_does_not_serve() -> None:
    from cayu.sessions._rollup import SessionRollupQuery

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_SESSIONS)
        return httpx.Response(405, json={"detail": "Method Not Allowed"})

    reader = AgentSessionReader(_api(handler))
    with pytest.raises(CayuApiError, match="doesn't serve session rollups yet"):
        asyncio.run(reader.rollup_page(SessionRollupQuery()))


def test_reader_filters_the_creation_window_when_an_older_agent_ignores_it() -> None:
    from datetime import UTC, datetime

    from cayu.sessions.queries import SessionQuery

    def row(session_id: str, created_at: str) -> dict:
        return {"id": session_id, "status": "completed", "created_at": created_at}

    pages = {
        None: {
            "sessions": [
                row("new", "2026-10-28T00:00:00+00:00"),
                row("in", "2026-10-27T01:00:00+00:00"),
            ],
            "next_cursor": "c1",
            "total_count": 9,
        },
        "c1": {
            "sessions": [row("in2", "2026-10-27T00:30:00+00:00")],
            "next_cursor": None,
            "total_count": 9,
        },
    }
    asked: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params.get("limit") == "1" and "created_since" not in params:
            return httpx.Response(200, json=_SESSIONS)
        asked.append(params)
        return httpx.Response(200, json=pages[params.get("cursor")])

    reader = AgentSessionReader(_api(handler))
    page = asyncio.run(
        reader.list_sessions(
            SessionQuery(
                created_since=datetime(2026, 10, 27, tzinfo=UTC),
                created_before=datetime(2026, 10, 28, tzinfo=UTC),
                limit=5,
            )
        )
    )

    assert [item["id"] for item in page.rows] == ["in", "in2"]
    assert page.total_count is None and page.next_cursor is None
    assert asked[0]["created_since"] == "2026-10-27T00:00:00+00:00"
    assert asked[1]["cursor"] == "c1" and asked[1]["limit"] == "4"


def test_reader_applies_the_offset_after_filtering_the_creation_window() -> None:
    from datetime import UTC, datetime

    from cayu.sessions.queries import SessionQuery

    ordered = [
        {"id": "a", "status": "completed", "created_at": "2026-10-20T00:00:00+00:00"},
        {"id": "b", "status": "completed", "created_at": "2026-10-27T01:00:00+00:00"},
        {"id": "c", "status": "completed", "created_at": "2026-10-27T02:00:00+00:00"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if params.get("limit") == "1" and "created_since" not in params:
            return httpx.Response(200, json=_SESSIONS)
        # An older Agent: ignores the window, pages by offset or cursor.
        start = int(params.get("cursor") or params.get("offset") or 0)
        stop = start + int(params["limit"])
        cursor = str(stop) if stop < len(ordered) else None
        return httpx.Response(200, json={"sessions": ordered[start:stop], "next_cursor": cursor})

    reader = AgentSessionReader(_api(handler))
    page = asyncio.run(
        reader.list_sessions(
            SessionQuery(created_since=datetime(2026, 10, 27, tzinfo=UTC), offset=1, limit=1)
        )
    )

    assert [item["id"] for item in page.rows] == ["c"]


def test_post_falls_back_past_another_apps_route_at_the_mount() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        if request.url.path.startswith("/cayu/"):
            return httpx.Response(405, json={"detail": "Method Not Allowed"})
        return httpx.Response(200, json={"ok": True})

    assert _api(handler).post("/sessions/rollup", {}, resource="session rollups") == {"ok": True}
    assert seen == ["POST /cayu/api/sessions/rollup", "POST /api/sessions/rollup"]


def test_reader_prices_usage_on_the_agent_and_falls_back_without_cost() -> None:
    from cayu.budgets.pricing import default_price_book
    from cayu.budgets.usage import build_aggregate_usage_metrics

    usage = {
        "session_id": "s",
        "calls": [],
        "aggregate": {
            "model_calls": 0,
            "usage": build_aggregate_usage_metrics().model_dump(mode="json"),
        },
    }
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sessions"):
            return httpx.Response(200, json=_SESSIONS)
        methods.append(request.method)
        if request.method == "POST":
            assert "price_book_version" in request.content.decode()
            return httpx.Response(405, json={"detail": "Method Not Allowed"})
        return httpx.Response(200, json=usage)

    reader = AgentSessionReader(_api(handler))
    result = asyncio.run(
        reader.model_calls(
            "s", InspectionWindow(), offset=0, limit=10, pricing=default_price_book()
        )
    )

    # Both API bases refuse the POST before the GET falls back to unpriced calls.
    assert methods == ["POST", "POST", "GET"]
    assert result["calls"] == [] and result.get("price_book") is None


def test_reader_never_trusts_an_agents_total_for_a_creation_window() -> None:
    from datetime import UTC, datetime

    from cayu.sessions.queries import SessionQuery

    rows = [
        {"id": "new1", "status": "completed", "created_at": "2026-10-28T01:00:00+00:00"},
        {"id": "new2", "status": "completed", "created_at": "2026-10-28T00:30:00+00:00"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        # An older Agent: ignores the window and counts every session.
        return httpx.Response(200, json={"sessions": rows, "next_cursor": "c", "total_count": 3})

    reader = AgentSessionReader(_api(handler))
    page = asyncio.run(
        reader.list_sessions(
            SessionQuery(created_since=datetime(2026, 10, 27, tzinfo=UTC), limit=2)
        )
    )

    assert [item["id"] for item in page.rows] == ["new1", "new2"]
    assert page.total_count is None
