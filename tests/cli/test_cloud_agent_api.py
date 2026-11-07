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


def test_client_reads_the_message_of_a_structured_refusal() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/app"):
            return httpx.Response(200, json={})
        refusal = {"code": "session_not_found", "message": "Session missing was not found."}
        return httpx.Response(404, json={"detail": refusal})

    api = _api(handler)
    api.get("/app", resource="agent")
    # A route's own refusal, not a route the server doesn't serve.
    with pytest.raises(CayuApiError) as raised:
        api.post("/evals/sessions/missing/evaluation/preview", {}, resource="evaluation")
    assert raised.value.category == "not_found"
    assert raised.value.detail == "Session missing was not found."


def test_a_refused_request_is_forbidden_with_its_reason_not_a_rejected_login() -> None:
    answers = {
        "/cayu/api/knowledge": httpx.Response(403, json={"detail": "Knowledge access denied."}),
        "/cayu/api/inbox": httpx.Response(
            403, json={"detail": {"code": "role_forbidden", "message": "Your role can't use this."}}
        ),
        "/cayu/api/tasks": httpx.Response(403, text="Cross-site request refused."),
        "/cayu/api/artifacts": httpx.Response(
            403, json={"detail": {"code": "sk-live-abcdefghijklmnop", "message": "No."}}
        ),
        "/cayu/api/sessions": httpx.Response(401, json={"detail": "Not authenticated"}),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return answers[request.url.path]

    def refusal(path: str) -> CayuApiError:
        with pytest.raises(CayuApiError) as raised:
            _api(handler).get(path, resource=path.strip("/"))
        return raised.value

    denied = refusal("/knowledge")
    assert (denied.category, denied.status_code, denied.detail, denied.code) == (
        "forbidden",
        403,
        "Knowledge access denied.",
        None,
    )
    assert str(denied) == (
        "The Agent bug-hunter refused the knowledge request: Knowledge access denied."
    )
    role = refusal("/inbox")
    assert (role.category, role.code) == ("forbidden", "role_forbidden")
    assert str(role).endswith(": Your role can't use this. (role_forbidden)")
    # A refusal that isn't Cayu's JSON (the router's own) still says it was refused.
    plain = refusal("/tasks")
    assert (plain.category, plain.detail, plain.code) == ("forbidden", None, None)
    assert str(plain) == "The Agent bug-hunter refused the tasks request."
    # A code that looks like a token is screened out like a reason would be.
    token = refusal("/artifacts")
    assert (token.category, token.detail, token.code) == ("forbidden", "No.", None)
    assert "sk-live" not in str(token)
    login = refusal("/sessions")
    assert (login.category, login.status_code) == ("credentials_rejected", 401)


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


def test_reader_pages_newest_first_reads_back_from_each_page() -> None:
    from cayu.sessions.event_queries import EventOrder

    stored = {
        "custom.a": [1, 4, 6],
        "custom.b": [2, 5],
    }
    asked: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/sessions"):
            return httpx.Response(200, json=_SESSIONS)
        params = dict(request.url.params)
        asked.append(params)
        before = int(params.get("before_sequence") or 1_000)
        if "event_type" not in params:
            newest = max(max(sequences) for sequences in stored.values())
            matching = [newest] if newest < before else []
        else:
            matching = sorted((n for n in stored[params["event_type"]] if n < before), reverse=True)
        # This Agent answers at most two events a page.
        limit = min(int(params["limit"]), 2)
        events = [
            {
                "sequence": n,
                "id": f"e{n}",
                "type": params.get("event_type", "custom.a"),
                "session_id": "s",
                "payload": {},
                "timestamp": "2026-10-28T12:00:00+00:00",
            }
            for n in matching[:limit]
        ]
        return httpx.Response(200, json={"events": events, "has_more": len(matching) > limit})

    reader = AgentSessionReader(_api(handler))
    records = asyncio.run(
        reader.query_events(
            EventQuery(
                session_id="s",
                event_types=("custom.a", "custom.b"),
                order_by=EventOrder.SEQUENCE_DESC,
                limit=4,
            )
        )
    )

    assert [record.sequence for record in records] == [6, 5, 4, 2]
    assert all(params["order_by"] == "sequence_desc" for params in asked)
    # Every type is read below one high-water mark; a type's next page continues below the
    # oldest event of its previous page.
    pages_of_a = [params for params in asked if params.get("event_type") == "custom.a"]
    assert [params["before_sequence"] for params in pages_of_a] == ["7", "4"]


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


def test_reader_asks_an_agent_that_cant_price_rollups_again_without_pricing() -> None:
    from cayu.budgets.pricing import default_price_book
    from cayu.sessions._rollup import SessionRollupQuery

    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "pricing" in body:
            return httpx.Response(
                422,
                json={
                    "detail": "Invalid control-plane request: body field pricing is not allowed."
                },
            )
        return httpx.Response(200, json={"groups": [], "sessions_read": 0, "next_cursor": None})

    reader = AgentSessionReader(_api(handler))
    priced = SessionRollupQuery(pricing=default_price_book())

    asyncio.run(reader.rollup_page(priced))
    asyncio.run(reader.rollup_page(priced))

    assert reader.rollup_pricing_unsupported is True
    # The first page tried the price book once; every later page goes straight to unpriced.
    # The key itself is left out: an older Agent rejects it whatever its value.
    assert ["pricing" in body for body in bodies] == [True, False, False]


def test_client_reads_created_and_accepted_answers() -> None:
    seen: list[tuple[str, str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("idempotency-key")))
        if request.url.path.endswith("/evals/corpora"):
            return httpx.Response(201, json={"revision": "sha256:corpus"})
        return httpx.Response(202, json={"status": "queued"})

    api = _api(handler)

    # An eval corpus import answers 201 Created, an eval run admission 202 Accepted.
    assert api.post("/evals/corpora", {}, resource="eval corpus import") == {
        "revision": "sha256:corpus"
    }
    assert api.mutate("/evals/runs", {}, resource="eval run", operation_id="admit-1") == {
        "status": "queued"
    }
    assert seen == [
        ("POST", "/cayu/api/evals/corpora", None),
        ("POST", "/cayu/api/evals/runs", "admit-1"),
    ]


def test_accepted_mutation_without_a_readable_receipt_is_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"name": "agent"})
        return httpx.Response(202, content=b"accepted")

    with pytest.raises(CayuApiError, match="no readable receipt") as raised:
        _api(handler).mutate("/evals/runs", {}, resource="eval run", operation_id="admit-1")
    assert raised.value.category == "outcome_unknown"


def test_other_successful_statuses_are_still_not_answers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    with pytest.raises(CayuApiError, match="answered HTTP 204") as raised:
        _api(handler).get("/sessions", resource="sessions")
    assert raised.value.category == "server_error"


def _waking_api(
    wakes: int,
    *,
    wake_timeout_seconds: float,
    notices: list[str],
    offers_wake_option: bool = True,
) -> CayuApi:
    """An Agent whose router answers 503 with Retry-After ``wakes`` times, then its sessions."""

    answered = {"count": 0}
    clock = {"now": 0.0}

    def handler(request: httpx.Request) -> httpx.Response:
        answered["count"] += 1
        if answered["count"] <= wakes:
            return httpx.Response(503, headers={"retry-after": "5"})
        return httpx.Response(200, json=_SESSIONS)

    def sleep(seconds: float) -> None:
        clock["now"] += seconds

    return CayuApi(
        "https://agent.example/cayu/",
        authorization="Basic secret",
        description="The Agent bug-hunter",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        wake_timeout_seconds=wake_timeout_seconds,
        sleep=sleep,
        monotonic=lambda: clock["now"],
        notice=notices.append,
        offers_wake_option=offers_wake_option,
    )


def test_a_read_waits_for_the_agent_it_woke_and_says_so_once() -> None:
    notices: list[str] = []
    # A wake of about two minutes, within the default bound.
    api = _waking_api(
        24, wake_timeout_seconds=agent_api.DEFAULT_WAKE_TIMEOUT_SECONDS, notices=notices
    )

    assert api.get("/sessions", resource="sessions") == _SESSIONS
    assert notices == ["The Agent bug-hunter is waking up; waiting up to 3 minutes."]
    api.get("/sessions", resource="sessions")
    assert len(notices) == 1


def test_a_wake_past_the_bound_or_no_wait_fails_and_says_how_to_wait() -> None:
    notices: list[str] = []
    with pytest.raises(CayuApiError, match="didn't answer within 30 seconds") as late:
        _waking_api(100, wake_timeout_seconds=30, notices=notices).get(
            "/sessions", resource="sessions"
        )
    assert late.value.category == "starting" and "--wake-timeout-seconds" in str(late.value)

    quiet: list[str] = []
    with pytest.raises(CayuApiError, match="is starting") as at_once:
        _waking_api(1, wake_timeout_seconds=0, notices=quiet).get("/sessions", resource="sessions")
    assert at_once.value.category == "starting" and quiet == []
    assert "--wake-timeout-seconds" in str(at_once.value)


def test_a_command_without_the_wake_options_is_not_told_to_pass_them() -> None:
    with pytest.raises(CayuApiError, match="didn't answer within 30 seconds") as late:
        _waking_api(100, wake_timeout_seconds=30, notices=[], offers_wake_option=False).get(
            "/sessions", resource="sessions"
        )
    assert late.value.category == "starting"
    assert "--wake-timeout-seconds" not in str(late.value)


def test_the_wake_options_parse_into_one_bound() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    agent_api.add_wake_options(parser)
    assert parser.parse_args([]).wake_timeout_seconds == agent_api.DEFAULT_WAKE_TIMEOUT_SECONDS
    assert parser.parse_args(["--no-wait"]).wake_timeout_seconds == 0
    assert parser.parse_args(["--wake-timeout-seconds", "45"]).wake_timeout_seconds == 45
    with pytest.raises(SystemExit):
        parser.parse_args(["--no-wait", "--wake-timeout-seconds", "45"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--wake-timeout-seconds", "-1"])
