"""`cayu cloud service logs`: the query it sends, how it prints lines, `--follow`, and a Cloud
from before line times."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    *lines: dict[str, Any], next_cursor: str | None = "c-end", more: bool = False
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "items": [line["message"] for line in lines],
            "lines": list(lines),
            "next_cursor": next_cursor,
            "more": more,
        },
    )


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

    assert _logs("--json") == 0

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

    assert _logs("--since", "2h") == 0
    assert capsys.readouterr().out.splitlines()[-1] == (
        "More lines follow; go on with --cursor c1."
    )
    assert _logs("--cursor", "c1") == 0
    assert "task 8 started" in capsys.readouterr().out
    assert fake.queries[1] == {"cursor": "c1", "limit": "100"}


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
