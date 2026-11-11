"""`cayu cloud service logs`: the query it sends, how it prints lines, `--follow`, and a Cloud
from before line times."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from cayu.cli import _cloud_api, _cloud_service_logs, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient

_AGENTS = httpx.Response(200, json={"items": [{"id": "my-agent", "name": "My Agent"}]})
_LOGS = "/v1/applications/my-agent/service/logs"


def _line(line_id: str, message: str, *, process: str | None = "worker") -> dict[str, Any]:
    return {
        "id": line_id,
        "timestamp": "2026-10-29T04:06:00.123+00:00",
        "process": process,
        "release": "dep_one",
        "instance": "task-1",
        "message": message,
    }


def _page(
    *lines: dict[str, Any],
    next_cursor: str | None = "c-end",
    more: bool = False,
    searched: tuple[str, str] | None = None,
) -> httpx.Response:
    body: dict[str, Any] = {
        "items": [line["message"] for line in lines],
        "lines": list(lines),
        "next_cursor": next_cursor,
        "more": more,
    }
    if searched is not None:
        body["searched"] = {"from": searched[0], "to": searched[1]}
    return httpx.Response(200, json=body)


class _Cloud:
    """Answers the Agent list, and the log route with ``pages`` in turn; an exception in
    ``pages`` is raised instead, as Ctrl-C would be during a request."""

    def __init__(self, pages: list[httpx.Response | BaseException]) -> None:
        self.pages = pages
        self.queries: list[dict[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/applications":
            return _AGENTS
        assert request.url.path == _LOGS
        self.queries.append(dict(request.url.params))
        answer = self.pages.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[..., _Cloud]:
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def install(*pages: httpx.Response | BaseException) -> _Cloud:
        fake = _Cloud(list(pages))
        real_client = httpx.Client
        monkeypatch.setattr(
            _cloud_api.httpx,
            "Client",
            lambda **kwargs: real_client(transport=httpx.MockTransport(fake), **kwargs),
        )
        monkeypatch.setattr(
            cloud_cli,
            "_cloud_client",
            lambda *_args, **_kwargs: CloudApiClient(
                api_url="https://cloud.example", api_key="login-token"
            ),
        )
        return fake

    return install


def _logs(*options: str) -> int:
    return main(["cloud", "service", "logs", "--application", "my-agent", *options])


def test_each_line_prints_with_its_time_process_and_release(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(
            _line("1", '"GET / HTTP/1.1" 404', process="web"),
            _line("2", "AttributeError: 'list' object has no attribute 'items'"),
            _line("3", "written before\x1b[2J processes were named", process=None),
        )
    )

    assert _logs() == 0

    assert capsys.readouterr().out.splitlines() == [
        '2026-10-29T04:06:00.123Z web dep_one "GET / HTTP/1.1" 404',
        "2026-10-29T04:06:00.123Z worker dep_one AttributeError: 'list' object has no attribute "
        "'items'",
        "2026-10-29T04:06:00.123Z - dep_one written before?[2J processes were named",
    ]
    # Asking for a page is what tells a Cloud to answer with timed lines.
    assert fake.queries == [{"limit": "100"}]


def test_a_schedules_full_name_is_printed(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    process = "schedule:" + "a" * 63
    cloud(_page(_line("1", "nightly ran", process=process)))

    assert _logs() == 0

    assert capsys.readouterr().out.split(" ")[1] == process


def test_options_choose_the_lines_cloud_returns(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(_page(_line("1", "Traceback (most recent call last):"), next_cursor=None))

    assert (
        _logs(
            "--process",
            "worker",
            "--release",
            "dep_one",
            "--since",
            "2026-10-28T21:00:00-07:00",
            "--until",
            "2026-10-28T21:10:00-07:00",
            "--grep",
            "Traceback",
            "--limit",
            "500",
        )
        == 0
    )

    assert fake.queries == [
        {
            "process": "worker",
            "release": "dep_one",
            "since": "2026-10-29T04:00:00+00:00",
            "until": "2026-10-29T04:10:00+00:00",
            "grep": "Traceback",
            "limit": "500",
        }
    ]
    assert "Traceback (most recent call last):" in capsys.readouterr().out


def test_since_takes_a_span_back_from_now(cloud: Callable[..., _Cloud]) -> None:
    fake = cloud(_page())

    assert _logs("--since", "90m") == 0

    since = datetime.fromisoformat(fake.queries[0]["since"])
    assert abs(datetime.now(UTC) - timedelta(minutes=90) - since) < timedelta(minutes=1)


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        (["--since", "yesterday"], "neither a span"),
        (["--limit", "5000"], "between 1 and 1000"),
        (["--search-for", "soon"], "neither 0 nor a span"),
    ],
)
def test_options_cayu_cant_read_are_refused_before_asking(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    reason: str,
) -> None:
    fake = cloud()

    assert _logs(*options) == 2

    assert reason in json.loads(capsys.readouterr().out)["error"]["message"]
    assert fake.queries == []


def test_cloud_names_what_it_refuses(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    refusal = {"code": "log_query_invalid", "message": "process is web, worker, ... not 'db'."}
    cloud(httpx.Response(422, json={"detail": refusal}))

    assert _logs("--process", "db") == 2

    error = capsys.readouterr().err
    assert "process is web, worker, ... not 'db'." in error


def test_json_prints_the_envelope_with_the_page(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(_page(_line("1", "task 7 started"), more=True, next_cursor="c1"))

    assert _logs("--json", "--search-for", "0") == 0

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is True
    assert envelope["operation"] == "service.logs"
    assert envelope["result"]["lines"][0]["process"] == "worker"
    assert envelope["result"]["next_cursor"] == "c1"


def test_a_page_with_more_after_it_says_how_to_go_on(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(_line("1", "task 7 started"), more=True, next_cursor="c1"),
        _page(_line("2", "task 8 started")),
    )

    assert _logs("--since", "2h", "--search-for", "0") == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "More lines follow; go on with --cursor c1."
    )
    assert _logs("--cursor", "c1") == 0
    assert "task 8 started" in capsys.readouterr().out
    assert fake.queries[1] == {"cursor": "c1", "limit": "100"}


_DAY = (
    ("2026-11-05T17:00:00.000+00:00", "2026-11-05T23:00:00.000+00:00"),
    ("2026-11-05T23:00:00.000+00:00", "2026-11-06T05:00:00.000+00:00"),
    ("2026-11-06T05:00:00.000+00:00", "2026-11-06T17:00:00.000+00:00"),
)


def test_a_wide_search_follows_short_pages_until_it_has_read_the_window(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    # Cloud answers each page with what it searched in a few seconds: none, then one match.
    fake = cloud(
        _page(more=True, next_cursor="c1", searched=_DAY[0]),
        _page(
            _line("1", "decision_outcome_unknown"), more=True, next_cursor="c2", searched=_DAY[1]
        ),
        _page(_line("2", "decision_outcome_unknown"), next_cursor=None, searched=_DAY[2]),
    )

    assert _logs("--since", "24h", "--grep", "decision_outcome_unknown", "--json") == 0

    result = json.loads(capsys.readouterr().out)["result"]
    assert [line["id"] for line in result["lines"]] == ["1", "2"]
    assert result["items"] == ["decision_outcome_unknown"] * 2
    assert result["searched"] == {"from": _DAY[0][0], "to": _DAY[2][1]}
    assert result["next_cursor"] is None
    assert result["more"] is False
    # Each page goes on from the one before, for the lines still missing from the limit.
    assert [query.get("cursor") for query in fake.queries] == [None, "c1", "c2"]
    assert [query["limit"] for query in fake.queries] == ["100", "100", "99"]
    assert all(query["grep"] == "decision_outcome_unknown" for query in fake.queries)
    assert "since" not in fake.queries[1]


def test_a_search_stops_at_its_limit(cloud: Callable[..., _Cloud]) -> None:
    fake = cloud(
        _page(_line("1", "a"), more=True, next_cursor="c1", searched=_DAY[0]),
        _page(_line("2", "b"), more=True, next_cursor="c2", searched=_DAY[1]),
    )

    assert _logs("--since", "24h", "--limit", "2") == 0

    assert len(fake.queries) == 2
    assert fake.pages == []


def test_a_search_out_of_time_says_which_span_it_searched() -> None:
    pages = [
        {
            "items": [],
            "lines": [],
            "next_cursor": f"c{n}",
            "more": True,
            "searched": {"from": start, "to": end},
        }
        for n, (start, end) in enumerate(_DAY[:2], start=1)
    ]

    class Client:
        def request(self, method: str, path: str, **_kwargs: object) -> dict[str, Any]:
            if path == "/v1/applications":
                return {"items": [{"id": "my-agent", "name": "My Agent"}]}
            clock[0] += 50
            return pages.pop(0)

    clock = [0.0]
    options = dict.fromkeys(("process", "release", "until", "limit", "cursor"))
    result = cloud_cli._service(
        SimpleNamespace(
            application="my-agent",
            service_command="logs",
            follow=False,
            json=False,
            since=datetime(2026, 11, 5, 17, tzinfo=UTC),
            grep="decision_outcome_unknown",
            search_for=60.0,
            **options,
        ),
        client=Client(),
        monotonic=lambda: clock[0],
    )

    # The second page ended past the minute: the search stops there.
    assert pages == []
    assert result["result"]["searched"] == {"from": _DAY[0][0], "to": _DAY[1][1]}
    assert _cloud_service_logs.render_logs(result) == (
        "Searched 2026-11-05T17:00:00.000Z to 2026-11-06T05:00:00.000Z; more may follow. "
        "Go on with --cursor c2."
    )


def test_ctrl_c_stops_a_search_after_the_last_page_it_merged(
    cloud: Callable[..., _Cloud],
) -> None:
    fake = cloud(
        _page(_line("1", "a"), more=True, next_cursor="c1", searched=_DAY[0]),
        _page(_line("2", "b"), more=True, next_cursor="c2", searched=_DAY[1]),
        KeyboardInterrupt(),
    )
    options = dict.fromkeys(("process", "release", "until", "grep", "limit", "cursor"))

    result = cloud_cli._service(
        SimpleNamespace(
            application="my-agent",
            service_command="logs",
            follow=False,
            json=True,
            since=datetime(2026, 11, 5, 17, tzinfo=UTC),
            search_for=60.0,
            **options,
        ),
        client=CloudApiClient(api_url="https://cloud.example", api_key="login-token"),
    )

    page = result["result"]
    # It goes on after the second page, whose lines it holds.
    assert [line["id"] for line in page["lines"]] == ["1", "2"]
    assert (page["next_cursor"], page["more"]) == ("c2", True)
    assert page["searched"] == {"from": _DAY[0][0], "to": _DAY[1][1]}
    assert len(fake.queries) == 3


def test_follow_prints_each_new_line_once_until_interrupted(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(_line("1", "task 7 started"), more=True, next_cursor="c1"),
        # A line Cloud gives again at a page boundary is printed once.
        _page(_line("1", "task 7 started"), _line("2", "Traceback"), more=True, next_cursor="c2"),
        KeyboardInterrupt(),
    )

    assert _logs("--process", "worker", "--since", "1h", "--follow") == 0

    printed = capsys.readouterr().out.splitlines()
    assert [line.split(" ", 3)[3] for line in printed] == ["task 7 started", "Traceback"]
    assert [query.get("cursor") for query in fake.queries] == [None, "c1", "c2"]
    # The cursor goes on from where the last page ended; the start isn't asked again.
    assert all("since" not in query for query in fake.queries[1:])
    assert {query["process"] for query in fake.queries} == {"worker"}


def test_ctrl_c_during_the_first_request_ends_follow_quietly(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(KeyboardInterrupt())

    assert _logs("--follow", "--json", "--cursor", "c0") == 0

    output = capsys.readouterr()
    [envelope] = [json.loads(line) for line in output.out.splitlines()]
    # Nothing was printed, so it goes on from where it was asked to.
    assert envelope["result"]["next_cursor"] == "c0"
    assert output.err == ""


def test_ctrl_c_while_a_page_is_written_goes_on_from_that_page(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Client:
        def __init__(self) -> None:
            self.pages = [
                {"items": [], "lines": [_line("1", "a")], "next_cursor": "c1", "more": True},
                {"items": [], "lines": [_line("2", "b")], "next_cursor": "c2", "more": True},
            ]

        def request(self, method: str, path: str, **_kwargs: object) -> dict[str, Any]:
            if path == "/v1/applications":
                return {"items": [{"id": "my-agent", "name": "My Agent"}]}
            return self.pages.pop(0)

    written: list[str] = []

    def write(text: str, **_kwargs: object) -> None:
        if written:
            raise KeyboardInterrupt
        written.append(text)

    monkeypatch.setattr(_cloud_service_logs, "print", write, raising=False)
    options = dict.fromkeys(("process", "release", "since", "until", "grep", "limit", "cursor"))
    result = cloud_cli._service(
        SimpleNamespace(
            application="my-agent", service_command="logs", follow=True, json=True, **options
        ),
        client=Client(),
        sleep=lambda _: None,
    )

    assert [json.loads(text)["result"]["items"] for text in written] == [["a"]]
    # The second page was never written: going on from the first's cursor prints it again.
    assert result["result"]["next_cursor"] == "c1"


def test_follow_waits_between_caught_up_polls_and_stops_on_ctrl_c(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Client:
        def __init__(self) -> None:
            self.pages = [
                {"items": [], "lines": [], "next_cursor": "c1", "more": False},
                {
                    "items": ["nightly ran"],
                    "lines": [_line("1", "nightly ran", process="schedule:nightly")],
                    "next_cursor": "c2",
                    "more": False,
                },
            ]

        def request(self, method: str, path: str, **_kwargs: object) -> dict[str, Any]:
            if path == "/v1/applications":
                return {"items": [{"id": "my-agent", "name": "My Agent"}]}
            return self.pages.pop(0)

    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)
        if len(waits) == 2:
            raise KeyboardInterrupt

    options = dict.fromkeys(("process", "release", "since", "until", "grep", "limit", "cursor"))
    result = cloud_cli._service(
        SimpleNamespace(
            application="my-agent", service_command="logs", follow=True, json=True, **options
        ),
        client=Client(),
        sleep=sleep,
    )

    assert waits == [2.0, 2.0]
    [envelope] = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert envelope["result"]["items"] == ["nightly ran"]
    # Nothing is left to print, only where to go on.
    assert result == {
        "operation": "service.logs",
        "result": {"items": [], "lines": [], "next_cursor": "c2", "more": False},
    }


def test_follow_ends_once_until_has_been_read(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(_line("1", "task 7 started"), more=True, next_cursor="c1"),
        _page(_line("2", "task 8 started"), next_cursor=None),
    )

    assert _logs("--until", "2026-10-29T04:10:00Z", "--follow") == 0

    assert len(capsys.readouterr().out.splitlines()) == 2
    assert len(fake.queries) == 2


def test_an_older_cloud_prints_its_untimed_lines(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(httpx.Response(200, json={"items": ['"GET / HTTP/1.1" 404', "worker tick"]}))

    assert _logs() == 0

    assert capsys.readouterr().out.splitlines() == [
        '"GET / HTTP/1.1" 404',
        "worker tick",
        "This Cayu Cloud doesn't report line times; upgrade it for the rest.",
    ]


@pytest.mark.parametrize(
    ("options", "named"),
    [
        (["--process", "worker"], "--process"),
        (["--limit", "5"], "--limit"),
        (["--follow"], "--follow"),
    ],
)
def test_an_older_cloud_refuses_what_it_would_ignore(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    named: str,
) -> None:
    cloud(httpx.Response(200, json={"items": ["worker tick"]}))

    assert _logs("--json", *options) == 2

    error = json.loads(capsys.readouterr().out)["error"]
    assert error["category"] == "service_logs_filters_unsupported"
    assert named in error["message"]


def test_structured_event_is_exact_and_decodes_json(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    messages = [
        '{"event":"worker.done.extra"}',
        '{"event":"other","message":"worker.done"}',
        ' { "ev\\u0065nt" : "worker.done", "count": 2 } ',
        '{"event":"worker.done"',
        '[{"event":"worker.done"}]',
        '{"event":"worker.done","value":NaN}',
        '{"event":"worker.done","value":Infinity}',
        "worker.done",
    ]
    fake = cloud(_page(*(_line(str(n), message) for n, message in enumerate(messages))))

    assert _logs("--since", "1h", "--event", "worker.done", "--json") == 0

    result = json.loads(capsys.readouterr().out)["result"]
    assert result["items"] == [messages[2]]
    assert [line["id"] for line in result["lines"]] == ["2"]
    assert "event" not in fake.queries[0]


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("value=1", ["number", "decimal"]),
        ('value="1"', ["string"]),
        ("value=true", ["boolean"]),
        ("value=null", ["null"]),
        ("value=hello=there", ["text"]),
        ("a.b=yes", ["literal"]),
    ],
)
def test_fields_match_json_scalar_types_and_literal_keys(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    expression: str,
    expected: list[str],
) -> None:
    values = {
        "number": 1,
        "decimal": 1.0,
        "string": "1",
        "boolean": True,
        "null": None,
        "text": "hello=there",
        "object": {"a": 1},
        "array": [1],
    }
    fake = cloud(
        _page(
            *(_line(key, json.dumps({"value": value})) for key, value in values.items()),
            _line("missing", "{}"),
            _line("literal", '{"a.b":"yes"}'),
            _line("nested", '{"a":{"b":"yes"}}'),
        )
    )

    assert _logs("--since", "1h", "--field", expression, "--json") == 0

    assert [
        line["id"] for line in json.loads(capsys.readouterr().out)["result"]["lines"]
    ] == expected
    assert "field" not in fake.queries[0]


def test_structured_filters_compose_and_search_past_full_nonmatching_pages(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    wanted = '{"event":"done","status":"ok","count":2}'
    fake = cloud(
        _page(
            _line("0", '{"event":"done","status":"bad","count":2}'),
            more=True,
            next_cursor="c1",
            searched=_DAY[0],
        ),
        _page(_line("1", wanted), more=True, next_cursor="c2", searched=_DAY[1]),
    )
    assert (
        _logs(
            "--since",
            "1h",
            "--event",
            "done",
            "--field",
            "status=ok",
            "--field",
            "count=2",
            "--process",
            "worker",
            "--release",
            "dep_one",
            "--grep",
            "done",
            "--limit",
            "1",
            "--json",
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["items"] == [wanted]
    assert result["next_cursor"] == "c2"
    assert [query.get("cursor") for query in fake.queries] == [None, "c1"]
    assert all(query["limit"] == "1" and query["grep"] == "done" for query in fake.queries)


def test_default_structured_search_retains_newest_matches_across_whole_window(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(
            _line("1", '{"event":"done"}'),
            _line("2", '{"event":"done"}'),
            more=True,
            next_cursor="c1",
            searched=_DAY[0],
        ),
        _page(
            _line("3", '{"event":"done"}'),
            _line("4", '{"event":"other"}'),
            next_cursor="end",
            searched=_DAY[1],
        ),
    )
    assert _logs("--event", "done", "--limit", "2", "--json") == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert [line["id"] for line in result["lines"]] == ["2", "3"]
    assert result["more"] is False
    assert result["next_cursor"] == "end"
    first = fake.queries[0]
    assert datetime.fromisoformat(first["until"]) - datetime.fromisoformat(
        first["since"]
    ) == timedelta(days=7)
    assert [query["limit"] for query in fake.queries] == ["1000", "1000"]


def test_future_until_does_not_move_default_lookback_into_the_future(
    cloud: Callable[..., _Cloud],
) -> None:
    fake = cloud(_page())
    before = datetime.now(UTC)
    assert _logs("--event", "done", "--until", "2099-01-01T00:00:00Z") == 0
    upper = datetime.fromisoformat(fake.queries[0]["until"])
    assert before <= upper <= datetime.now(UTC)
    assert upper - datetime.fromisoformat(fake.queries[0]["since"]) == timedelta(days=7)


def test_partial_newest_scan_says_newer_matches_may_remain(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(_line("1", '{"event":"done"}'), more=True, next_cursor="c1", searched=_DAY[0])
    )
    assert _logs("--event", "done", "--search-for", "0") == 0
    text = capsys.readouterr().out
    assert "newer matches may remain" in text
    assert "--cursor c1" in text
    assert len(fake.queries) == 1


@pytest.mark.parametrize("limit", [1, 2])
def test_newest_follow_keeps_raw_page_overlap_outside_display_tail(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str], limit: int
) -> None:
    fake = cloud(
        _page(
            *(_line(name, '{"event":"done"}') for name in ("old", "middle", "new")),
            more=True,
            next_cursor="c1",
        ),
        _page(
            *(_line(name, '{"event":"done"}') for name in ("old", "new", "later")),
            more=True,
            next_cursor="c2",
        ),
        _page(_line("old", '{"event":"done"}'), next_cursor=None),
    )
    assert (
        _logs("--event", "done", "--limit", str(limit), "--search-for", "0", "--follow", "--json")
        == 0
    )
    envelopes = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["id"] for envelope in envelopes for line in envelope["result"]["lines"]] == [
        *["middle", "new"][-limit:],
        "later",
    ]
    assert [query.get("cursor") for query in fake.queries] == [None, "c1", "c2"]


@pytest.mark.parametrize("limit", [1, 1000])
def test_newest_tail_bounds_and_boundary_duplicates(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    limit: int,
) -> None:
    records = [_line(str(index), '{"event":"done"}') for index in range(1000)]
    cloud(
        _page(*records, more=True, next_cursor="c1"),
        _page(records[0], _line("1000", '{"event":"done"}')),
    )
    assert _logs("--event", "done", "--limit", str(limit), "--json") == 0
    lines = json.loads(capsys.readouterr().out)["result"]["lines"]
    assert [line["id"] for line in lines] == [str(index) for index in range(1001 - limit, 1001)]


def test_structured_follow_advances_nonmatching_pages_and_keeps_original_until(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(_line("1", '{"event":"other"}'), next_cursor="c1", more=True),
        _page(_line("2", '{"event":"done"}'), next_cursor="c2", more=True),
        _page(_line("2", '{"event":"done"}'), next_cursor=None),
    )
    assert (
        _logs(
            "--since",
            "1h",
            "--until",
            "2099-01-01T00:00:00Z",
            "--event",
            "done",
            "--follow",
            "--json",
        )
        == 0
    )
    envelopes = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["id"] for envelope in envelopes for line in envelope["result"]["lines"]] == ["2"]
    assert [query.get("cursor") for query in fake.queries] == [None, "c1", "c2"]
    assert all(query["until"] == "2099-01-01T00:00:00+00:00" for query in fake.queries)


@pytest.mark.parametrize("expression", ["missing", "=value", "a=[]", "a={}", "a=NaN", "a=Infinity"])
def test_invalid_field_filters_fail_before_request(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    expression: str,
) -> None:
    fake = cloud()
    assert _logs("--field", expression) == 2
    message = json.loads(capsys.readouterr().out)["error"]["message"]
    assert "--field" in message
    assert "unrecognized" not in message
    assert fake.queries == []


def test_cursor_structured_search_keeps_first_matches_without_implicit_window(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(_page(_line("1", '{"event":"done"}'), more=True, next_cursor="c2"))
    assert _logs("--cursor", "c1", "--event", "done", "--limit", "1", "--json") == 0
    assert fake.queries == [{"cursor": "c1", "limit": "1"}]
    assert json.loads(capsys.readouterr().out)["result"]["next_cursor"] == "c2"


def test_newest_scan_budget_retains_latest_matches_and_coverage(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Advance time at real request dispatch, rather than expiring before the search starts.
    clock = [0.0]
    original = _Cloud.__call__

    def dispatch(self: _Cloud, request: httpx.Request) -> httpx.Response:
        result = original(self, request)
        if request.url.path == _LOGS:
            clock[0] += 50
        return result

    monkeypatch.setattr(_Cloud, "__call__", dispatch)
    fake = cloud(
        _page(_line("1", '{"event":"done"}'), more=True, next_cursor="c1", searched=_DAY[0]),
        _page(_line("2", '{"event":"done"}'), more=True, next_cursor="c2", searched=_DAY[1]),
    )
    from cayu.cli.cloud import _build_parser

    arguments = _build_parser().parse_args(
        [
            "service",
            "logs",
            "--application",
            "my-agent",
            "--event",
            "done",
            "--limit",
            "1",
            "--search-for",
            "60s",
        ]
    )
    result = cloud_cli._service(
        arguments,
        client=CloudApiClient(api_url="https://cloud.example", api_key="login-token"),
        monotonic=lambda: clock[0],
    )
    assert len(fake.queries) == 2
    assert [line["id"] for line in result["result"]["lines"]] == ["2"]
    assert result["result"]["searched"] == {"from": _DAY[0][0], "to": _DAY[1][1]}
    assert "newer matches may remain" in result["warnings"][0]


def test_unknown_coverage_and_truncated_messages_are_not_upgraded(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    truncated = {**_line("bad", '{"event":"done"}'), "truncated": True}
    cloud(
        _page(truncated, more=True, next_cursor="c1", searched=_DAY[0]),
        _page(_line("1", '{"event":"done"}'), more=True, next_cursor="c2"),
        _page(_line("2", '{"event":"done"}'), searched=_DAY[2]),
    )
    assert _logs("--event", "done", "--json") == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert [line["id"] for line in result["lines"]] == ["1", "2"]
    assert result["searched"] is None


@pytest.mark.parametrize("phase", ["scan", "follow", "follow_cursor", "overlap"])
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX SIGINT delivery")
def test_structured_follow_over_real_http_preserves_sigint_checkpoint(
    tmp_path: Path,
    phase: str,
) -> None:
    requests: list[dict[str, list[str]]] = []
    waiting, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            target = urlsplit(self.path)
            if target.path == "/v1/applications":
                body = {"items": [{"id": "my-agent", "name": "My Agent"}]}
            else:
                assert target.path == _LOGS
                requests.append(parse_qs(target.query))
                number = len(requests)
                blocked = {"scan": 2, "follow": 3, "follow_cursor": 4, "overlap": 3}[phase]
                if number == blocked:
                    waiting.set()
                    release.wait(20)
                    return
                if phase == "overlap":
                    response = _page(
                        *(
                            _line(identity, '{"event":"done"}')
                            for identity in (("old", "new") if number == 1 else ("old", "next"))
                        ),
                        more=True,
                        next_cursor=f"c{number}",
                    )
                elif number == 1:
                    response = _page(
                        _line("old", '{"event":"done"}'),
                        more=True,
                        next_cursor="c1",
                        searched=_DAY[0],
                    )
                elif number == 2:
                    response = _page(
                        _line("new", '{"event":"done"}'), next_cursor=None, searched=_DAY[1]
                    )
                else:
                    # Same inclusive integer-millisecond bounds as Cloud's LogQuery.
                    upper = datetime.fromisoformat(requests[0]["until"][0])
                    after = datetime.fromisoformat(requests[-1]["since"][0])
                    until = datetime.fromisoformat(requests[-1]["until"][0])
                    assert until - after <= timedelta(days=7)
                    end_ms = int(upper.timestamp() * 1000)
                    candidates = [(end_ms, "old"), (end_ms, "new"), (end_ms + 1, "next")]
                    lines = [
                        _line(identity, '{"event":"done"}')
                        for milliseconds, identity in candidates
                        if milliseconds >= int(after.timestamp() * 1000)
                    ]
                    response = _page(*lines, next_cursor="live", more=True)
                body = response.json()
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = {
        **os.environ,
        "CAYU_CLOUD_CONFIG": str(tmp_path / "config.json"),
        "CAYU_CLOUD_API_URL": f"http://127.0.0.1:{server.server_port}",
        "CAYU_CLOUD_API_KEY": "local-test-token",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
    }
    for name in ("CAYU_CLOUD_CONTEXT", "CAYU_CLOUD_API_KEY_FILE"):
        environment.pop(name, None)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cayu",
            "cloud",
            "service",
            "logs",
            "--application",
            "my-agent",
            "--event",
            "done",
            "--limit",
            "1",
            "--until",
            "2099-01-01T00:00:00Z",
            "--follow",
            "--json",
            *(["--search-for", "0"] if phase == "overlap" else []),
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if not waiting.wait(15):
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(f"CLI did not reach the {phase} request: {stdout!r} {stderr!r}")
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0
        assert stderr == ""
        envelopes = [json.loads(line) for line in stdout.splitlines()]
        assert [line["id"] for envelope in envelopes for line in envelope["result"]["lines"]] == {
            "scan": ["old"],
            "follow": ["new"],
            "follow_cursor": ["new", "next"],
            "overlap": ["new", "next"],
        }[phase]
        assert (
            envelopes[-1]["result"]["next_cursor"]
            == {"scan": "c1", "follow": None, "follow_cursor": "live", "overlap": "c2"}[phase]
        )
        assert all("event" not in query and "field" not in query for query in requests)
        if phase == "follow":
            upper = datetime.fromisoformat(requests[0]["until"][0])
            after = upper + timedelta(milliseconds=1)
            assert datetime.fromisoformat(requests[-1]["since"][0]) == after
            assert datetime.fromisoformat(requests[-1]["until"][0]) == after + timedelta(days=7)
            assert envelopes[-1]["result"]["next_since"] == after.isoformat()
        elif phase == "follow_cursor":
            assert requests[-1]["cursor"] == ["live"]
            assert requests[-1]["until"] == ["2099-01-01T00:00:00+00:00"]
            assert "next_since" not in envelopes[-1]["result"]
        elif phase == "overlap":
            assert [query.get("cursor") for query in requests] == [None, ["c1"], ["c2"]]
    finally:
        release.set()
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
