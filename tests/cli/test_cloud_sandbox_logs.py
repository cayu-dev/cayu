"""`cayu cloud sandboxes logs`: the route and query it asks, how it prints the broker's lines,
paging, `--follow`, and what a refusal means."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from cayu.cli import _cloud_api, _cloud_sandbox_logs, main
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient

_AGENTS = httpx.Response(200, json={"items": [{"id": "acme-scout", "name": "Acme Scout"}]})
_SANDBOX = "sbx_0123456789abcdef0123456789abcdef"
_ONE = f"/v1/applications/acme-scout/sandboxes/{_SANDBOX}/logs"
_AGENT = "/v1/applications/acme-scout/sandboxes/logs"


def _line(line_id: str, event: str, **fields: object) -> dict[str, Any]:
    return {
        "id": line_id,
        "timestamp": "2026-11-08T17:02:11.408+00:00",
        "level": "warning",
        "event": event,
        "sandbox_id": _SANDBOX,
        "fields": fields,
    }


def _page(
    *lines: dict[str, Any],
    next_cursor: str | None = None,
    more: bool = False,
    searched: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={"lines": list(lines), "next_cursor": next_cursor, "more": more, "searched": searched},
    )


def _refusal(status: int, code: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": code, "message": "no", "detail": None}})


class _Cloud:
    """Answers the Agent list, and each log read with ``pages`` in turn; an exception in
    ``pages`` is raised instead, as Ctrl-C would be during a request."""

    def __init__(self, pages: list[httpx.Response | BaseException]) -> None:
        self.pages = pages
        self.reads: list[tuple[str, dict[str, str]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.endswith("/logs"):
            assert request.url.path == "/v1/applications", request.url
            return _AGENTS
        self.reads.append((request.url.raw_path.decode().split("?")[0], dict(request.url.params)))
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


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(_cloud_sandbox_logs.time, "sleep", slept.append)
    return slept


def _logs(*options: str) -> int:
    return main(["cloud", "sandboxes", "logs", "--application", "Acme Scout", *options])


def test_one_sandboxs_lines_print_with_their_event_and_fields(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(
            _line(
                "1", "sandbox_egress_refused", host="pypi.org", reason="connect_failed", status=502
            ),
            _line(
                "2",
                "sandbox_preflight_failed",
                attempt=1,
                retrying=True,
                proxy_reason=None,
                message="proxy said no\x1b[2J at [url]",
            ),
        )
    )

    assert _logs(_SANDBOX) == 0

    assert capsys.readouterr().out.splitlines() == [
        f"2026-11-08T17:02:11.408Z warning sandbox_egress_refused {_SANDBOX} host=pypi.org "
        "reason=connect_failed status=502",
        f"2026-11-08T17:02:11.408Z warning sandbox_preflight_failed {_SANDBOX} attempt=1 "
        'retrying=true proxy_reason=null message="proxy said no\\u001b[2J at [url]"',
    ]
    assert fake.reads == [(_ONE, {"limit": "100"})]


def test_without_a_sandbox_it_reads_the_agents_lines_with_the_query_asked(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(_page(_line("1", "sandbox_ready")))

    assert (
        _logs(
            "--since",
            "2026-11-08T17:00:00Z",
            "--until",
            "2026-11-08T18:00:00+00:00",
            "--event",
            "sandbox_ready",
            "--limit",
            "20",
        )
        == 0
    )

    assert fake.reads == [
        (
            _AGENT,
            {
                "since": "2026-11-08T17:00:00+00:00",
                "until": "2026-11-08T18:00:00+00:00",
                "event": "sandbox_ready",
                "limit": "20",
            },
        )
    ]
    assert capsys.readouterr().out.startswith("2026-11-08T17:02:11.408Z warning sandbox_ready")


def test_a_page_cut_short_goes_on_from_its_cursor_without_the_start(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(_line("1", "sandbox_ready"), next_cursor="c1", more=True),
        # A line Cloud gives again at a page boundary is printed once.
        _page(_line("1", "sandbox_ready"), _line("2", "sandbox_destroyed"), next_cursor=None),
    )

    assert _logs(_SANDBOX, "--since", "1h", "--event", "sandbox_ready", "--json") == 0

    envelope = json.loads(capsys.readouterr().out)
    assert envelope["operation"] == "sandboxes.logs"
    result = envelope["result"]
    assert [line["id"] for line in result["lines"]] == ["1", "2"]
    assert (result["sandbox_id"], result["next_cursor"], result["schema_version"]) == (
        _SANDBOX,
        None,
        1,
    )
    assert "since" in fake.reads[0][1]
    assert fake.reads[1][1] == {"cursor": "c1", "event": "sandbox_ready", "limit": "99"}


def test_text_says_how_to_go_on(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(_line("1", "sandbox_ready"), next_cursor="c1", more=True),
        # Cloud's read budget ran out before any line: read on at once.
        _page(next_cursor="c2", more=True),
        _page(_line("2", "sandbox_destroyed"), next_cursor="c3", more=True),
    )

    assert _logs(_SANDBOX, "--limit", "1") == 0
    assert capsys.readouterr().out.splitlines()[-1] == "More lines follow; go on with --cursor c1."

    assert _logs(_SANDBOX, "--cursor", "c1", "--limit", "1") == 0
    assert capsys.readouterr().out.splitlines()[-1] == "More lines follow; go on with --cursor c3."
    assert [query.get("cursor") for _, query in fake.reads] == [None, "c1", "c2"]


def test_a_relative_until_goes_on_at_the_time_it_resolved_to(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    # `--until 1h` is another time when run again, so the hint and the result carry the bound
    # the first page was read to, and going on with it repeats the exact query.
    fake = cloud(
        _page(_line("1", "sandbox_ready"), next_cursor="c1", more=True),
        _page(_line("1", "sandbox_ready"), next_cursor="c1", more=True),
    )

    assert _logs(_SANDBOX, "--until", "1h", "--limit", "1") == 0
    hint = capsys.readouterr().out.splitlines()[-1]
    until = fake.reads[0][1]["until"]
    assert hint == f"More lines follow; go on with --cursor c1 --until {until}."

    assert _logs(_SANDBOX, "--until", "1h", "--limit", "1", "--json") == 0
    result = json.loads(capsys.readouterr().out)["result"]
    assert result["until"] == fake.reads[1][1]["until"]


def test_a_live_sandboxs_read_says_where_later_lines_go_on(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(_page(_line("1", "sandbox_ready"), next_cursor="c1"))

    assert _logs(_SANDBOX) == 0

    assert (
        capsys.readouterr().out.splitlines()[-1]
        == "Later lines go on from --cursor c1, or use --follow."
    )


def test_a_cursor_and_a_start_together_are_refused_before_any_request(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud()

    assert _logs(_SANDBOX, "--cursor", "c1", "--since", "1h", "--json") != 0

    assert fake.reads == []
    assert "not allowed with argument --cursor" in capsys.readouterr().out


def test_follow_prints_each_line_once_and_ends_with_the_read(
    cloud: Callable[..., _Cloud], sleeps: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        _page(_line("1", "sandbox_ready"), next_cursor="c1"),
        _page(_line("1", "sandbox_ready"), _line("2", "sandbox_egress_refused"), next_cursor="c2"),
        # The sandbox was destroyed 15 minutes ago: its read ends.
        _page(_line("3", "sandbox_destroyed"), next_cursor=None),
    )

    assert _logs(_SANDBOX, "--follow", "--until", "2026-11-08T18:00:00Z") == 0

    printed = capsys.readouterr().out.splitlines()
    assert [line.split(" ")[2] for line in printed] == [
        "sandbox_ready",
        "sandbox_egress_refused",
        "sandbox_destroyed",
    ]
    assert [query.get("cursor") for _, query in fake.reads] == [None, "c1", "c2"]
    assert {query["until"] for _, query in fake.reads} == {"2026-11-08T18:00:00+00:00"}
    assert sleeps == [2.0, 2.0]


def test_follow_until_ctrl_c_says_where_to_go_on(
    cloud: Callable[..., _Cloud], sleeps: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        _page(_line("1", "sandbox_ready"), next_cursor="c1", more=True),
        KeyboardInterrupt(),
    )

    assert _logs("--follow", "--json") == 0

    envelopes = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["id"] for line in envelopes[0]["result"]["lines"]] == ["1"]
    assert envelopes[-1]["result"]["next_cursor"] == "c1"
    assert envelopes[-1]["result"]["lines"] == []
    # A page cut short is read on at once.
    assert sleeps == []


@pytest.mark.parametrize(
    ("status", "code", "message"),
    [
        (503, "sandbox_logs_unavailable", "doesn't keep it where Cloud can read it"),
        (422, "log_query_invalid", "the span is at most 7 days"),
        (404, "not_found", "no such sandbox for this Agent"),
    ],
)
def test_a_refusal_says_what_it_means(
    cloud: Callable[..., _Cloud],
    capsys: pytest.CaptureFixture[str],
    status: int,
    code: str,
    message: str,
) -> None:
    cloud(_refusal(status, code))

    assert _logs(_SANDBOX, "--json") != 0

    envelope = json.loads(capsys.readouterr().out)
    assert message in envelope["error"]["message"]


def test_an_invalid_page_is_refused(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(httpx.Response(200, json={"lines": "nope", "next_cursor": None, "more": False}))

    assert _logs(_SANDBOX, "--json") != 0

    assert "invalid" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_follow_in_text_says_where_to_go_on_after_ctrl_c(
    cloud: Callable[..., _Cloud], sleeps: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(_page(_line("1", "sandbox_ready"), next_cursor="c1"), KeyboardInterrupt())

    assert _logs(_SANDBOX, "--follow") == 0

    printed = capsys.readouterr().out.splitlines()
    assert printed[0].split(" ")[2] == "sandbox_ready"
    assert printed[-1] == "Go on with --cursor c1."


def test_follow_goes_on_after_the_agents_newest_lines_without_the_start(
    cloud: Callable[..., _Cloud], sleeps: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    # Cloud answers the newest lines whole (`more` false) with a cursor after them.
    fake = cloud(
        _page(_line("1", "sandbox_ready"), next_cursor="c1"),
        _page(_line("2", "sandbox_destroyed"), next_cursor="c2"),
        KeyboardInterrupt(),
    )

    assert _logs("--follow", "--event", "sandbox_destroyed") == 0

    assert [(path, query) for path, query in fake.reads] == [
        (_AGENT, {"event": "sandbox_destroyed", "limit": "100"}),
        (_AGENT, {"event": "sandbox_destroyed", "cursor": "c1", "limit": "100"}),
        (_AGENT, {"event": "sandbox_destroyed", "cursor": "c2", "limit": "100"}),
    ]
    assert sleeps == [2.0, 2.0]
    assert capsys.readouterr().out.splitlines()[-1] == "Go on with --cursor c2."


def test_reading_on_stops_at_the_search_deadline(
    cloud: Callable[..., _Cloud],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clock = iter([0.0, 1.0, _cloud_sandbox_logs.SEARCH_SECONDS + 1])
    monkeypatch.setattr(_cloud_sandbox_logs.time, "monotonic", lambda: next(clock))
    fake = cloud(
        _page(next_cursor="c1", more=True),
        _page(_line("1", "sandbox_ready"), next_cursor="c2", more=True),
    )

    assert _logs(_SANDBOX) == 0

    assert len(fake.reads) == 2
    assert capsys.readouterr().out.splitlines()[-1] == "More lines follow; go on with --cursor c2."


def test_ctrl_c_while_reading_on_keeps_what_was_read_and_the_span_searched(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        _page(
            _line("1", "sandbox_ready"),
            next_cursor="c1",
            more=True,
            searched={"from": "2026-11-08T16:00:00+00:00", "to": "2026-11-08T16:30:00+00:00"},
        ),
        _page(
            _line("2", "sandbox_egress_refused"),
            next_cursor="c2",
            more=True,
            searched={"from": "2026-11-08T16:30:00+00:00", "to": "2026-11-08T17:00:00+00:00"},
        ),
        KeyboardInterrupt(),
    )

    assert _logs(_SANDBOX, "--json") == 0

    result = json.loads(capsys.readouterr().out)["result"]
    assert [line["id"] for line in result["lines"]] == ["1", "2"]
    assert result["next_cursor"] == "c2"
    assert result["searched"] == {
        "from": "2026-11-08T16:00:00+00:00",
        "to": "2026-11-08T17:00:00+00:00",
    }


def test_a_long_cursor_from_cloud_is_taken_and_a_bad_event_refused(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cursor = "e" * 2000
    fake = cloud(_page())

    assert _logs(_SANDBOX, "--cursor", cursor) == 0
    assert fake.reads[0][1]["cursor"] == cursor

    assert _logs(_SANDBOX, "--event", "egress refused", "--json") != 0
    assert "is not a log event name" in capsys.readouterr().out
    assert len(fake.reads) == 1


def test_a_login_that_cant_see_the_agent_is_told_so(
    cloud: Callable[..., _Cloud], capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(httpx.Response(403, json={"detail": {"code": "agent_not_yours"}}))

    assert _logs("--json") != 0

    assert (
        "can't read the Agent's sandboxes"
        in json.loads(capsys.readouterr().out)["error"]["message"]
    )
