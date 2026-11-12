"""Write once: the marker format and why no value can end it early, reading markers back from
the trusted login only, and a comment or an issue looked up before it is written, through
races, lost answers, dropped labels, page limits and read-only mode, against the fake
GitHub."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from cayu.connections import ConnectionResponse
from cayu.connectors import github
from cayu.connectors.github import markers as markers_module
from cayu.connectors.github.testing import Call, FakeGitHub, reply

REPO = "acme/widgets"
HERALD = "cayu-herald[bot]"
SCOUT = "cayu-scout[bot]"
SOMEONE = "mallory"
COMMENTS = f"/repos/{REPO}/issues/7/comments"
ISSUES = f"/repos/{REPO}/issues"
LABEL = "cayu-scout"
run = asyncio.run

Fakes = Callable[..., FakeGitHub]


@pytest.fixture
def now() -> list[datetime]:
    """The fakes' clock, which the helpers' waits move on."""

    return [datetime(2026, 11, 12, 9, tzinfo=UTC)]


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch, now: list[datetime]) -> list[float]:
    """Every wait the helpers make, recorded instead of slept, moving the fakes' clock on."""

    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)
        now[0] += timedelta(seconds=seconds)

    monkeypatch.setattr(markers_module, "_sleep", record)
    return waited


@pytest.fixture
def fakes(now: list[datetime]) -> Iterator[Fakes]:
    """Fake GitHubs with ``acme/widgets`` and its issue 7, writing as ``login`` on the ``now``
    clock; each must have answered every call it was sent."""

    made: list[FakeGitHub] = []

    def make(login: str = HERALD, **settings: Any) -> FakeGitHub:
        fake = FakeGitHub(login=login, clock=lambda: now[0], **settings)
        fake.add_issue(REPO, 7, title="Digests", author="octocat")
        made.append(fake)
        return fake

    yield make
    for fake in made:
        assert not fake.unanswered


def _comments(fake: FakeGitHub) -> list[Any]:
    return fake.repository(REPO).issues[7].comments


def _comment_once(fake: FakeGitHub, key: str = "herald/digest/1", **kwargs: Any) -> Any:
    gh = kwargs.pop("gh", None) or fake.client(read_retries=0)
    body = kwargs.pop("body", "Today's digest.")
    login = kwargs.pop("login", HERALD)
    return run(github.comment_once(gh, REPO, 7, body, github.Marker(key), login=login, **kwargs))


def _file(fake: FakeGitHub, key: str = "scout/finding/f1", **kwargs: Any) -> Any:
    kwargs.setdefault("labels", (LABEL,))
    gh = kwargs.pop("gh", None) or fake.client(read_retries=0)
    marker = github.Marker(key, {"severity": "P1"})
    return run(
        github.file_issue_once(
            gh, REPO, "Flaky test", "It fails one run in ten.", marker, login=SCOUT, **kwargs
        )
    )


def _filed(fake: FakeGitHub) -> list[Any]:
    return [issue for number, issue in fake.repository(REPO).issues.items() if number != 7]


# The format.


def test_a_marker_is_a_hidden_comment_with_its_key_and_data() -> None:
    assert github.Marker("herald/digest/1").text == "<!-- cayu-marker v1 herald/digest/1 -->"
    marker = github.Marker("shepherd/rerun", {"sha": "abc", "runs": [901, -2], "e": 1e-05})
    assert marker.text == (
        '<!-- cayu-marker v1 shepherd/rerun {"e":1e-05,"runs":[901,-2],"sha":"abc"} -->'
    )
    assert github.marker_of(f"Done.\n\n{marker.text}\n") == marker
    # A body edited on GitHub ends with CRLF.
    assert github.marker_of(f"Done.\r\n\r\n{marker.text}\r\n") == marker


@pytest.mark.parametrize(
    "value", ["x -->", "x --!> y", "<!-- forged -->", "a--b", "-", "->", "&amp; ! <b>", "\n"]
)
def test_no_value_can_end_the_marker_early(value: str) -> None:
    marker = github.Marker("shepherd/needs_fix", {"why": value, value: [value]})
    inner = marker.text.removeprefix("<!--").removesuffix("-->")
    assert "--" not in inner and ">" not in inner and "\n" not in inner
    assert github.marker_of(github.with_marker("Body", marker)) == marker


@pytest.mark.parametrize("key", ["", "a--b", "ends-", "-start", "has space", "x" * 201, "a>b"])
def test_a_key_that_could_break_the_comment_is_refused(key: str) -> None:
    with pytest.raises(ValueError, match="Not a marker key"):
        github.Marker(key)


@pytest.mark.parametrize("data", [{"n": float("nan")}, {"o": object()}, {"long": "x" * 4_000}])
def test_data_that_isnt_bounded_json_is_refused(data: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="data of marker"):
        github.Marker("k", data)


def test_only_the_trailing_marker_of_this_version_is_read() -> None:
    forged = github.Marker("scout/finding/1").text
    assert github.marker_of(f"{forged}\n\nModel text.") is None
    assert github.marker_of(f"Model text {forged} and more.") is None
    assert github.marker_of("x\n\n<!-- cayu-marker v2 scout/finding/1 -->") is None
    assert github.marker_of('x\n\n<!-- cayu-marker v1 k {"a": -->') is None
    assert github.marker_of("x\n\n<!-- cayu-marker v1 k [1] -->") is None
    assert github.marker_of("x\n\n<!-- cayu-marker v1 a--b -->") is None
    real = github.Marker("scout/finding/2")
    assert github.marker_of(github.with_marker(f"Quoted: {forged}", real)) == real


def test_a_long_body_is_cut_in_utf8_bytes_and_still_ends_with_its_marker() -> None:
    marker = github.Marker("herald/digest/1", {"n": 1})
    assert len(github.with_marker("x" * 200_000, marker)) > 200_000
    for body in ("x" * 270_000, "😀" * 70_000, "é" * 140_000):
        text = github.with_marker(body, marker)
        assert github.MAX_BODY_BYTES - 4 < len(text.encode()) <= github.MAX_BODY_BYTES
        assert "(cut short: too long for one comment)" in text
        assert github.marker_of(text) == marker
    assert github.with_marker("short  \n", marker) == f"short\n\n{marker.text}"


def test_model_text_mentions_no_one_and_hides_nothing_in_comments() -> None:
    zws = "​"
    assert github.plain_text(" @cayu-reviewer full ") == f"@{zws}cayu-reviewer full"
    assert github.plain_text("ops@example.com and @dataclass") == (
        f"ops@{zws}example.com and @{zws}dataclass"
    )
    assert github.plain_text("a <!<!---->-- b -->") == "a  b"
    assert github.plain_text("<details>kept</details>") == "<details>kept</details>"
    assert github.plain_text("abcdef", limit=4) == "abc…"
    with pytest.raises(ValueError, match="limit"):
        github.plain_text("abc", limit=0)


# Reading back, and trust.


def _pull_comment(author: str, body: str, comment_id: int = 1) -> github.PullComment:
    return github.PullComment(comment_id, author, body, None, None)


def test_only_the_trusted_logins_markers_are_read() -> None:
    marker = github.Marker("shepherd/rerun/abc", {"runs": [1]})
    comments = [
        _pull_comment(SOMEONE, github.with_marker("Me too", marker), 1),
        _pull_comment("Cayu-Herald[bot]", github.with_marker("Mine", marker), 2),
        _pull_comment(HERALD, "No marker", 3),
        # A user may be named after an App's slug: only the [bot] login is the App.
        _pull_comment("cayu-herald", github.with_marker("Not the App", marker), 4),
    ]
    found = github.trusted_markers(comments, login=HERALD)
    assert [(item.marker, item.comment.id) for item in found] == [(marker, 2)]
    with pytest.raises(ValueError, match="login"):
        github.trusted_markers(comments, login=" ")


def test_a_legacy_format_is_read_from_the_trusted_login_only() -> None:
    def legacy(body: str) -> github.Marker | None:
        return github.Marker("shepherd/introduce") if "<!-- cayu-shepherd" in body else None

    old = '<!-- cayu-shepherd {"action": "introduce"} -->'
    comments = [_pull_comment(SOMEONE, old, 1), _pull_comment(HERALD, old, 2)]
    found = github.trusted_markers(comments, login=HERALD, legacy=legacy)
    assert [item.comment.id for item in found] == [2]


def test_an_app_connection_trusts_only_its_bot_login() -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ).decode()
    app = github.app_connections("1234", pem, permissions={"issues": "write"})
    gh = github.GitHubClient(app)
    with pytest.raises(ValueError, match=r"GitHubAppConnections\.login\(\)"):
        run(github.comment_once(gh, REPO, 7, "x", github.Marker("k"), login="cayu-herald"))


# Comment once.


def test_a_comment_is_posted_once_and_found_after(fakes: Fakes) -> None:
    fake = fakes()
    first = _comment_once(fake)
    assert first.outcome is github.WriteOutcome.CREATED
    assert _comments(fake)[0].body.endswith("\n\n<!-- cayu-marker v1 herald/digest/1 -->")
    again = _comment_once(fake)
    assert (again.outcome, again.id, again.url) == (
        github.WriteOutcome.FOUND,
        first.id,
        first.url,
    )
    assert len(fake.writes()) == 1


def test_a_trusted_comment_ending_in_crlf_is_found(fakes: Fakes) -> None:
    fake = fakes()
    marked = github.with_marker("Edited on GitHub", github.Marker("herald/digest/1"))
    fake.add_comment(REPO, 7, marked.replace("\n", "\r\n") + "\r\n", author=HERALD)
    assert _comment_once(fake).outcome is github.WriteOutcome.FOUND


def test_another_logins_comment_with_the_marker_doesnt_stop_the_post(fakes: Fakes) -> None:
    fake = fakes()
    marked = github.with_marker("Copied", github.Marker("herald/digest/1"))
    fake.add_comment(REPO, 7, marked, author=SOMEONE)
    assert _comment_once(fake).outcome is github.WriteOutcome.CREATED


def test_marker_text_quoted_inside_a_trusted_comment_doesnt_count(fakes: Fakes) -> None:
    fake = fakes()
    forged = github.Marker("herald/digest/1").text
    body = github.with_marker(f"Model said {forged}", github.Marker("other"))
    fake.add_comment(REPO, 7, body, author=HERALD)
    assert _comment_once(fake).outcome is github.WriteOutcome.CREATED


def test_since_narrows_the_lookup_to_comments_updated_since(fakes: Fakes) -> None:
    fake = fakes()
    _comment_once(fake, since=datetime(2026, 11, 5, tzinfo=UTC))
    assert fake.requests[0].query["since"] == "2026-11-05T00:00:00Z"
    with pytest.raises(ValueError, match="timezone"):
        _comment_once(fake, since=datetime(2026, 11, 5))


def test_update_edits_the_found_comment_in_place_only_when_it_differs(fakes: Fakes) -> None:
    fake = fakes()
    first = _comment_once(fake, body="Draft")
    same = _comment_once(fake, body="Draft", update=True)
    assert same.outcome is github.WriteOutcome.FOUND
    changed = _comment_once(fake, body="Final", update=True)
    assert (changed.outcome, changed.id) == (github.WriteOutcome.UPDATED, first.id)
    assert [(call.method, call.path) for call in fake.writes()] == [
        ("POST", COMMENTS),
        ("PATCH", f"/repos/{REPO}/issues/comments/{first.id}"),
    ]
    assert _comments(fake)[0].body.startswith("Final\n\n")


def test_raced_writers_converge_on_the_earliest_and_see_the_rest(fakes: Fakes) -> None:
    fake = fakes()
    marker = github.Marker("herald/digest/1")
    earliest = fake.add_comment(REPO, 7, github.with_marker("One", marker), author=HERALD)
    later = fake.add_comment(REPO, 7, github.with_marker("Two", marker), author=HERALD)
    found = _comment_once(fake)
    assert (found.outcome, found.id, found.duplicates) == (
        github.WriteOutcome.FOUND,
        earliest.id,
        (later.id,),
    )
    # Reported, never deleted.
    assert fake.writes() == []


class _Yielding:
    """The fake behind a connection that lets other coroutines run before each call, as a
    network round trip does, so writers that don't take turns interleave."""

    def __init__(self, fake: FakeGitHub) -> None:
        self.fake = fake

    async def request(self, *args: Any, **kwargs: Any) -> ConnectionResponse:
        await asyncio.sleep(0)
        return await self.fake.request(*args, **kwargs)


def test_one_clients_writers_take_turns(fakes: Fakes) -> None:
    fake = fakes()
    gh = github.GitHubClient(_Yielding(fake), read_retries=0)

    async def three() -> list[Any]:
        return await asyncio.gather(
            *(
                github.comment_once(gh, REPO, 7, "Digest", github.Marker("k"), login=HERALD)
                for _ in range(3)
            )
        )

    outcomes = sorted(written.outcome for written in run(three()))
    assert outcomes == ["created", "found", "found"]
    assert len(fake.writes()) == 1
    assert gh._write_once.current() == {}


def test_a_lookup_cut_short_by_its_page_limit_writes_nothing(fakes: Fakes) -> None:
    fake = fakes(page_size=2)
    for _ in range(3):
        fake.add_comment(REPO, 7, "Chatter", author=SOMEONE)
    with pytest.raises(github.GitHubLookupIncomplete, match="known=") as raised:
        _comment_once(fake, max_pages=1)
    assert "since" not in str(raised.value)
    assert fake.writes() == []
    # Found on the first page, the limit doesn't matter.
    early = fakes(page_size=2)
    early.add_comment(REPO, 7, github.with_marker("x", github.Marker("k")), author=HERALD)
    for _ in range(3):
        early.add_comment(REPO, 7, "Chatter", author=SOMEONE)
    assert _comment_once(early, "k", max_pages=1).outcome is github.WriteOutcome.FOUND


def _landing_then_lost(fake: FakeGitHub) -> Callable[[Call, Any], ConnectionResponse]:
    """A write that takes effect on GitHub, whose answer is then lost."""

    def handler(call: Call, groups: Any) -> ConnectionResponse:
        document = call.json()
        if groups:
            fake.add_comment(REPO, int(groups[0]), document["body"], author=fake.login)
        else:
            fake.add_issue(
                REPO,
                title=document["title"],
                body=document["body"],
                labels=document.get("labels", []),
                author=fake.login,
            )
        return reply(502, {"message": "Server Error"})

    return handler


def test_a_comment_whose_answer_was_lost_is_looked_up_not_resent(
    fakes: Fakes, waits: list[float]
) -> None:
    fake = fakes()
    fake.route("POST", r"issues/(\d+)/comments", _landing_then_lost(fake))
    written = _comment_once(fake)
    assert (written.outcome, written.id) == (github.WriteOutcome.CREATED, _comments(fake)[0].id)
    assert len(fake.writes()) == 1
    assert waits == [1.0]


def test_a_lost_answer_is_looked_for_until_the_listing_shows_it(
    fakes: Fakes, waits: list[float]
) -> None:
    fake = fakes(SCOUT, listing_lag=timedelta(seconds=6))
    fake.route("POST", r"issues", _landing_then_lost(fake))
    filed = _file(fake)
    assert (filed.outcome, filed.number) == (github.WriteOutcome.CREATED, _filed(fake)[0].number)
    assert waits == [1.0, 2.0, 4.0]
    assert len(fake.writes()) == 1


def test_a_lost_answer_never_listed_raises_after_about_20_seconds(
    fakes: Fakes, waits: list[float]
) -> None:
    fake = fakes(SCOUT, listing_lag=timedelta(minutes=5))
    fake.route("POST", r"issues", _landing_then_lost(fake))
    with pytest.raises(github.GitHubUnavailable) as raised:
        _file(fake)
    assert raised.value.ambiguous is True
    assert sum(waits) == 20.0
    assert len(fake.writes()) == 1


def test_a_lost_write_that_didnt_land_raises_without_resending(fakes: Fakes) -> None:
    fake = fakes()
    fake.fail("POST", COMMENTS, status=502)
    with pytest.raises(github.GitHubUnavailable) as raised:
        _comment_once(fake)
    assert raised.value.ambiguous is True
    assert len(fake.writes()) == 1
    assert _comments(fake) == []


def test_a_read_only_client_looks_up_and_reports_the_refused_write(fakes: Fakes) -> None:
    fake = fakes()
    gh = fake.client(read_retries=0).read_only_view()
    written = _comment_once(fake, gh=gh)
    assert (written.outcome, written.id) == (github.WriteOutcome.REFUSED, None)
    assert [call.method for call in fake.requests] == ["GET"]
    assert gh.refused == (github.RefusedWrite("POST", COMMENTS, REPO),)


def test_a_permission_refusal_on_the_write_is_raised(fakes: Fakes) -> None:
    fake = fakes()
    fake.fail("POST", COMMENTS, status=403, body={"message": "Resource not accessible"})
    with pytest.raises(github.GitHubPermissionDenied):
        _comment_once(fake)


# File an issue once.


def test_an_issue_is_filed_once_through_the_issues_listing(fakes: Fakes) -> None:
    fake = fakes(SCOUT)
    filed = _file(fake)
    assert (filed.outcome, filed.labelled) == (github.WriteOutcome.CREATED, True)
    assert [issue.labels for issue in _filed(fake)] == [[LABEL]]
    assert filed.number == _filed(fake)[0].number
    lookup = fake.requests[0]
    assert lookup.path == ISSUES
    assert {name: lookup.query[name] for name in ("state", "creator", "sort", "direction")} == {
        "state": "all",
        "creator": SCOUT,
        "sort": "created",
        "direction": "asc",
    }
    assert "labels" not in lookup.query
    again = _file(fake)
    assert (again.outcome, again.number) == (github.WriteOutcome.FOUND, filed.number)
    assert len(fake.writes()) == 1
    assert all(not call.path.startswith("/search") for call in fake.requests)


def test_an_old_untouched_issue_is_still_found(fakes: Fakes) -> None:
    fake = fakes(SCOUT)
    month_ago = datetime.now(UTC) - timedelta(days=30)
    body = github.with_marker("Filed long ago", github.Marker("scout/finding/f1"))
    old = fake.add_issue(
        REPO, body=body, author=SCOUT, labels=[LABEL], created_at=month_ago, updated_at=month_ago
    )
    found = _file(fake)
    assert (found.outcome, found.number) == (github.WriteOutcome.FOUND, old.number)
    assert fake.writes() == []


def test_issue_lookup_skips_pull_requests_and_other_authors(fakes: Fakes) -> None:
    fake = fakes(SCOUT)
    marked = github.with_marker("x", github.Marker("scout/finding/f1"))
    fake.add_issue(REPO, body=marked, author=SOMEONE, labels=[LABEL])
    fake.add_pull(REPO, body=marked, author=SCOUT, labels=[LABEL])
    assert _file(fake).outcome is github.WriteOutcome.CREATED


def test_the_author_is_checked_even_if_github_ignored_the_creator_filter(fakes: Fakes) -> None:
    fake = fakes(SCOUT)
    marked = github.with_marker("x", github.Marker("scout/finding/f1"))

    def unfiltered(call: Call, groups: Any) -> ConnectionResponse:
        return reply(200, [{"number": 3, "body": marked, "user": {"login": SOMEONE}}])

    fake.route("GET", r"issues", unfiltered)
    assert _file(fake).outcome is github.WriteOutcome.CREATED


def test_an_issue_filed_without_its_labels_is_found_and_says_so(fakes: Fakes) -> None:
    fake = fakes(SCOUT)

    def without_labels(call: Call, groups: Any) -> ConnectionResponse:
        document = call.json()
        issue = fake.add_issue(REPO, title=document["title"], body=document["body"], author=SCOUT)
        return reply(201, {"number": issue.number, "html_url": "", "labels": []})

    fake.route("POST", r"issues", without_labels)
    first = _file(fake)
    assert (first.outcome, first.labelled) == (github.WriteOutcome.CREATED, False)
    second = _file(fake)
    assert (second.outcome, second.number, second.labelled) == (
        github.WriteOutcome.FOUND,
        first.number,
        False,
    )
    assert len(fake.writes()) == 1
    marked = github.with_marker("x", github.Marker("other"))
    fake.add_issue(REPO, body=marked, author=SCOUT, labels=[LABEL])
    assert _file(fake, "other").labelled is True


def test_the_earliest_issue_is_found_whatever_its_labels(fakes: Fakes) -> None:
    fake = fakes(SCOUT)
    marked = github.with_marker("x", github.Marker("scout/finding/f1"))
    earliest = fake.add_issue(REPO, body=marked, author=SCOUT)
    later = fake.add_issue(REPO, body=marked, author=SCOUT, labels=[LABEL])
    found = _file(fake)
    assert (found.outcome, found.number, found.duplicates, found.labelled) == (
        github.WriteOutcome.FOUND,
        earliest.number,
        (later.number,),
        False,
    )
    assert [call.method for call in fake.requests] == ["GET"]


def test_an_issue_lookup_fails_closed_at_its_page_limit(fakes: Fakes) -> None:
    fake = fakes(SCOUT, page_size=1)
    for _ in range(2):
        fake.add_issue(REPO, body="An unmarked issue", author=SCOUT)
    with pytest.raises(github.GitHubLookupIncomplete, match="max_pages"):
        _file(fake, max_pages=1)
    assert fake.writes() == []


# The listing lags a new write.


def test_a_quick_repeat_on_one_client_finds_its_write_before_the_listing_does(
    fakes: Fakes,
) -> None:
    fake = fakes(SCOUT, listing_lag=timedelta(seconds=15))
    gh = fake.client(read_retries=0)
    first = _file(fake, gh=gh)
    again = _file(fake, gh=gh)
    assert (again.outcome, again.number) == (github.WriteOutcome.FOUND, first.number)
    herald = fakes(listing_lag=timedelta(seconds=15))
    comments = herald.client(read_retries=0)
    posted = _comment_once(herald, gh=comments)
    assert _comment_once(herald, gh=comments).id == posted.id
    assert len(fake.writes()) == len(herald.writes()) == 1


def test_a_new_process_finds_a_just_filed_issue_by_its_known_number(fakes: Fakes) -> None:
    fake = fakes(SCOUT, listing_lag=timedelta(seconds=15))
    first = _file(fake)
    # A fresh client: no memory, and the listing doesn't show the issue yet.
    again = _file(fake, known=first.number)
    assert (again.outcome, again.number) == (github.WriteOutcome.FOUND, first.number)
    assert [call.path for call in fake.requests][-2:] == [f"{ISSUES}/{first.number}", ISSUES]
    assert len(fake.writes()) == 1
    # Without it, that window can file a duplicate, which a later call reports.
    assert _file(fake).outcome is github.WriteOutcome.CREATED


def test_a_recalled_issue_doesnt_hide_an_earlier_one_or_its_duplicates(fakes: Fakes) -> None:
    fake = fakes(SCOUT)
    marked = github.with_marker("x", github.Marker("scout/finding/f1"))
    earliest = fake.add_issue(REPO, body=marked, author=SCOUT)
    later = fake.add_issue(REPO, body=marked, author=SCOUT)
    gh = fake.client(read_retries=0)
    for _ in range(2):  # by known=, then from this client's memory of it
        found = _file(fake, gh=gh, known=later.number)
        assert (found.outcome, found.number, found.duplicates) == (
            github.WriteOutcome.FOUND,
            earliest.number,
            (later.number,),
        )


def test_a_recalled_comment_doesnt_hide_an_earlier_one(fakes: Fakes) -> None:
    fake = fakes()
    marker = github.Marker("herald/digest/1")
    earliest = fake.add_comment(REPO, 7, github.with_marker("early", marker), author=HERALD)
    later = fake.add_comment(REPO, 7, github.with_marker("later", marker), author=HERALD)
    edited = _comment_once(fake, known=later.id, body="replacement", update=True)
    assert (edited.outcome, edited.id, edited.duplicates) == (
        github.WriteOutcome.UPDATED,
        earliest.id,
        (later.id,),
    )
    assert _comments(fake)[1].body.startswith("later")


def test_a_lagging_listing_still_finds_the_recalled_write_and_its_elders(fakes: Fakes) -> None:
    fake = fakes(SCOUT, listing_lag=timedelta(seconds=15))
    marked = github.with_marker("x", github.Marker("scout/finding/f1"))
    older = fake.add_issue(
        REPO, body=marked, author=SCOUT, created_at=fake.clock() - timedelta(minutes=1)
    )
    fresh = _file(fake, known=None)
    assert (fresh.outcome, fresh.number) == (github.WriteOutcome.FOUND, older.number)
    newest = fake.add_issue(REPO, body=marked, author=SCOUT)
    found = _file(fake, known=newest.number)
    assert (found.outcome, found.number, found.duplicates) == (
        github.WriteOutcome.FOUND,
        older.number,
        (newest.number,),
    )


@pytest.mark.parametrize(
    "answer",
    [
        reply(410, {"message": "This issue was deleted"}),
        reply(301, None, {"location": "https://api.github.com/repositories/9/issues/1"}),
        reply(
            200,
            {
                "number": 99,
                "body": github.with_marker("x", github.Marker("scout/finding/f1")),
                "user": {"login": SCOUT},
                "html_url": "https://github.com/acme/elsewhere/issues/1",
            },
        ),
    ],
    ids=["deleted", "transferred", "elsewhere"],
)
def test_a_known_issue_gone_from_the_repository_is_skipped_and_forgotten(
    fakes: Fakes, answer: ConnectionResponse
) -> None:
    fake = fakes(SCOUT)
    gh = fake.client(read_retries=0)
    first = _file(fake, gh=gh)
    fake.repository(REPO).issues.pop(first.number)
    fake.route("GET", r"issues/(\d+)", lambda call, groups: answer)
    again = _file(fake, gh=gh, known=99)
    assert again.outcome is github.WriteOutcome.CREATED
    assert gh._written_once == {("acme/widgets", None, "scout/finding/f1"): again.number}


def test_a_known_comment_is_read_directly_and_must_still_be_ours(fakes: Fakes) -> None:
    fake = fakes(listing_lag=timedelta(seconds=15))
    posted = _comment_once(fake)
    assert _comment_once(fake, known=posted.id).outcome is github.WriteOutcome.FOUND
    changed = _comment_once(fake, known=posted.id, body="Edited", update=True)
    assert (changed.outcome, changed.id) == (github.WriteOutcome.UPDATED, posted.id)
    assert len(fake.writes()) == 2


@pytest.mark.parametrize("case", ["other author", "other key", "other thread", "deleted"])
def test_a_known_id_that_isnt_this_write_is_ignored(fakes: Fakes, case: str) -> None:
    fake = fakes()
    fake.add_issue(REPO, 8)
    marker = github.Marker("other" if case == "other key" else "herald/digest/1")
    author = SOMEONE if case == "other author" else HERALD
    comment = fake.add_comment(
        REPO, 8 if case == "other thread" else 7, github.with_marker("x", marker), author=author
    )
    known = comment.id
    if case == "deleted":
        _comments(fake).remove(comment)
    assert _comment_once(fake, known=known).outcome is github.WriteOutcome.CREATED


def test_an_issue_whose_answer_was_lost_is_found_not_filed_again(fakes: Fakes) -> None:
    fake = fakes(SCOUT)
    fake.route("POST", r"issues", _landing_then_lost(fake))
    filed = _file(fake)
    assert (filed.outcome, filed.number) == (github.WriteOutcome.CREATED, _filed(fake)[0].number)
    assert len(fake.writes()) == 1
