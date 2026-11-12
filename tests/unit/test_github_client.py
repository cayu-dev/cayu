"""The GitHub client: each call through a slot, paging, errors, rate limits, read-only mode
and GraphQL answers with errors."""

from __future__ import annotations

import asyncio
import gc
import json
import time
import weakref
from collections.abc import Callable, Mapping
from typing import Any

import httpx
import pytest
from tests.connector_broker_support import Answer, FakeBroker, Refusal, Sent

from cayu.connections import CloudConnectionBroker, ConnectionResponse, Query
from cayu.connectors import github
from cayu.connectors.github import client as client_module

REPO = "acme/widgets"
API = "https://api.github.com"


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every wait the client makes, recorded instead of slept."""

    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(client_module, "_sleep", record)
    return waited


def _gh(broker: FakeBroker, **kwargs: Any) -> github.GitHubClient:
    return github.GitHubClient(broker.client, **kwargs)


def _answers(*answers: Answer | Refusal) -> Callable[[Sent], Answer | Refusal]:
    """A connector answering in turn, then repeating its last answer."""

    queue = list(answers)
    return lambda _: queue.pop(0) if len(queue) > 1 else queue[0]


async def _workflow(gh: github.GitHubClient) -> tuple[list[int], dict[str, Any]]:
    """The same code on Cayu Cloud and on a laptop: list open pulls, comment on the first."""

    pulls = await gh.list(f"/repos/{REPO}/pulls", query={"state": "open"})
    numbers = [pull["number"] for pull in pulls.items]
    comment = await gh.post(
        f"/repos/{REPO}/issues/{numbers[0]}/comments", json={"body": "Checked."}
    )
    return numbers, comment


def _pulls_and_comments(method: str, path: str) -> tuple[int, Any]:
    if method == "GET" and path == f"/repos/{REPO}/pulls":
        return 200, [{"number": 7}, {"number": 9}]
    if method == "POST" and path == f"/repos/{REPO}/issues/7/comments":
        return 201, {"id": 1, "body": "Checked."}
    return 404, {"message": "Not Found"}


def test_a_workflow_lists_pulls_and_comments_through_the_broker() -> None:
    def connector(sent: Sent) -> Answer:
        status, body = _pulls_and_comments(sent.method, sent.path)
        return Answer(status=status, body=body)

    broker = FakeBroker(connector)
    numbers, comment = asyncio.run(_workflow(_gh(broker)))

    assert numbers == [7, 9]
    assert comment == {"id": 1, "body": "Checked."}
    listed, posted = broker.sent
    assert listed.slot == posted.slot == "github"
    assert listed.query == {"per_page": "100", "state": "open"}
    assert listed.headers == {
        "accept": "application/vnd.github+json",
        "x-cayu-github-repository": REPO,
        "x-github-api-version": "2022-11-28",
    }
    assert posted.headers["content-type"] == "application/json"
    assert posted.json() == {"body": "Checked."}


def test_the_same_workflow_runs_on_a_laptop_with_a_token() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status, body = _pulls_and_comments(request.method, request.url.path)
        return httpx.Response(status, json=body)

    local = github.token_connections("ghp_test", transport=httpx.MockTransport(handler))
    numbers, _ = asyncio.run(_workflow(github.GitHubClient(local)))

    assert numbers == [7, 9]
    assert [str(request.url) for request in seen] == [
        f"{API}/repos/{REPO}/pulls?per_page=100&state=open",
        f"{API}/repos/{REPO}/issues/7/comments",
    ]
    assert all(request.headers["authorization"] == "Bearer ghp_test" for request in seen)
    assert all("x-cayu-github-repository" not in request.headers for request in seen)


def test_without_a_client_outside_cayu_cloud_the_client_says_what_to_give(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAYU_CONNECTIONS_URL", raising=False)
    with pytest.raises(ValueError, match="connections_from_environment"):
        github.GitHubClient()


def _paged(pages: list[list[int]], *, link: Callable[[int], str] | None = None) -> FakeBroker:
    """A listing of ``pages``, each linking to the next by the repository's id, as GitHub
    does."""

    def next_link(page: int) -> str:
        return f'<{API}/repositories/1296269/pulls?state=all&per_page=100&page={page}>; rel="next"'

    def connector(sent: Sent) -> Answer:
        page = int(sent.query.get("page", ["1"])[0])
        headers = {"link": (link or next_link)(page + 1)} if page < len(pages) else {}
        return Answer(body=[{"number": n} for n in pages[page - 1]], headers=headers)

    return FakeBroker(connector)


def test_list_follows_next_links_across_pages_for_the_same_repository() -> None:
    broker = _paged([[1, 2], [3], [4]])
    page = asyncio.run(_gh(broker).list(f"/repos/{REPO}/pulls", query={"state": "all"}))

    assert page == github.Page(
        items=({"number": 1}, {"number": 2}, {"number": 3}, {"number": 4}), complete=True
    )
    assert [sent.path for sent in broker.sent] == [f"/repos/{REPO}/pulls"] * 3
    assert [sent.headers["x-cayu-github-repository"] for sent in broker.sent] == [REPO] * 3
    assert broker.sent[2].query == {"state": ["all"], "per_page": ["100"], "page": ["3"]}


def test_list_says_when_max_pages_or_keep_going_stopped_it() -> None:
    capped = _paged([[1], [2], [3]])
    page = asyncio.run(_gh(capped).list(f"/repos/{REPO}/pulls", max_pages=2))
    assert page == github.Page(items=({"number": 1}, {"number": 2}), complete=False)
    assert len(capped.sent) == 2

    stopped = _paged([[1], [2], [3]])
    page = asyncio.run(
        _gh(stopped).list(f"/repos/{REPO}/pulls", keep_going=lambda items: items[-1]["number"] < 2)
    )
    assert page.items == ({"number": 1}, {"number": 2})
    assert page.complete is False

    async def every() -> list[int]:
        return [
            item["number"]
            async for item in _gh(_paged([[1], [2], [3]])).paginate(f"/repos/{REPO}/pulls")
        ]

    assert asyncio.run(every()) == [1, 2, 3]


def test_a_next_link_off_the_api_or_to_another_listing_is_not_followed() -> None:
    for link in (
        f'<https://evil.example/repos/{REPO}/pulls?page=2>; rel="next"',
        f'<{API}/repos/acme/other/pulls?page=2>; rel="next"',
        f'<{API}/repositories/1/issues?page=2>; rel="next"',
    ):
        broker = _paged([[1], [2]], link=lambda _, link=link: link)
        with pytest.raises(github.GitHubError, match="isn't the same listing"):
            asyncio.run(_gh(broker).list(f"/repos/{REPO}/pulls"))
        assert len(broker.sent) == 1


def test_list_reads_items_under_a_key_and_refuses_another_shape() -> None:
    broker = FakeBroker(lambda _: Answer(body={"total_count": 1, "check_runs": [{"id": 5}]}))
    gh = _gh(broker)
    path = f"/repos/{REPO}/commits/abc/check-runs"
    assert asyncio.run(gh.list(path, item_key="check_runs")).items == ({"id": 5},)
    with pytest.raises(github.GitHubError, match="isn't a list"):
        asyncio.run(gh.list(path))


def test_errors_carry_githubs_words_status_and_request_id() -> None:
    broker = FakeBroker(
        _answers(
            Answer(
                status=404,
                body={"message": "Not Found", "documentation_url": "https://docs.github.com/x"},
                headers={"x-github-request-id": "ABC:1"},
            ),
            Answer(
                status=403,
                body={"message": "Resource not accessible by integration"},
                headers={"x-accepted-github-permissions": "issues=write"},
            ),
            Answer(
                status=422,
                body={
                    "message": "Validation Failed",
                    "errors": [{"resource": "Label", "field": "name", "code": "already_exists"}],
                },
            ),
            Answer(status=400, body={"message": "x" * 5_000}),
        )
    )
    gh = _gh(broker)

    with pytest.raises(github.GitHubNotFound) as missing:
        asyncio.run(gh.get(f"/repos/{REPO}/pulls/1"))
    assert str(missing.value) == (
        f"GitHub answered 404 to GET /repos/{REPO}/pulls/1 (request ABC:1): Not Found; "
        "see https://docs.github.com/x"
    )
    assert (missing.value.status, missing.value.request_id) == (404, "ABC:1")

    with pytest.raises(github.GitHubPermissionDenied) as denied:
        asyncio.run(gh.post(f"/repos/{REPO}/issues/1/comments", json={"body": "x"}))
    assert denied.value.accepted_permissions == "issues=write"
    assert "Resource not accessible by integration" in str(denied.value)

    with pytest.raises(github.GitHubConflict) as conflict:
        asyncio.run(gh.post(f"/repos/{REPO}/labels", json={"name": "bug"}))
    assert conflict.value.codes == ("already_exists",)
    assert "Label.name already_exists" in str(conflict.value)

    with pytest.raises(github.GitHubError) as long:
        asyncio.run(gh.get(f"/repos/{REPO}"))
    assert len(str(long.value)) == 1_000


def test_a_primary_rate_limit_is_waited_until_reset_and_resent_once(waits: list[float]) -> None:
    reset = str(int(time.time()) + 30)
    broker = FakeBroker(
        _answers(
            Answer(
                status=403,
                body={"message": "API rate limit exceeded"},
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": reset},
            ),
            Answer(body={"number": 1}),
        )
    )
    assert asyncio.run(_gh(broker).get(f"/repos/{REPO}/pulls/1")) == {"number": 1}
    assert len(broker.sent) == 2
    [wait] = waits
    assert 25 <= wait <= 31


def test_a_secondary_limit_on_a_write_is_waited_and_resent_once_then_raised(
    waits: list[float],
) -> None:
    limited = Answer(
        status=429,
        body={"message": "You have exceeded a secondary rate limit."},
        headers={"retry-after": "3"},
    )
    broker = FakeBroker(_answers(limited))
    with pytest.raises(github.GitHubRateLimited) as raised:
        asyncio.run(_gh(broker).post(f"/repos/{REPO}/issues/1/comments", json={"body": "x"}))
    assert (raised.value.secondary, raised.value.retry_after) == (True, 3.0)
    assert waits == [3.0]
    assert len(broker.sent) == 2

    worded = FakeBroker(
        _answers(Answer(status=403, body={"message": "secondary rate limit"}), Answer(body={}))
    )
    asyncio.run(_gh(worded).get(f"/repos/{REPO}"))
    assert waits[-1] == 60.0


def test_a_rate_limit_longer_than_the_wait_allowed_raises_without_resending(
    waits: list[float],
) -> None:
    broker = FakeBroker(_answers(Answer(status=429, headers={"retry-after": "3600"})))
    with pytest.raises(github.GitHubRateLimited) as raised:
        asyncio.run(_gh(broker, max_rate_limit_wait_seconds=10).get(f"/repos/{REPO}"))
    assert raised.value.retry_after == 3600.0
    assert waits == []
    assert len(broker.sent) == 1


def test_a_server_failure_is_retried_for_reads_and_ambiguous_for_writes(
    waits: list[float],
) -> None:
    flaky = FakeBroker(_answers(Answer(status=502), Answer(status=503), Answer(body={"ok": 1})))
    assert asyncio.run(_gh(flaky).get(f"/repos/{REPO}")) == {"ok": 1}
    assert waits == [0.5, 1.0]

    failing = FakeBroker(_answers(Answer(status=502)))
    with pytest.raises(github.GitHubUnavailable) as read:
        asyncio.run(_gh(failing, read_retries=1).get(f"/repos/{REPO}"))
    assert read.value.ambiguous is False
    assert len(failing.sent) == 2

    write = FakeBroker(_answers(Answer(status=502)))
    with pytest.raises(github.GitHubUnavailable) as written:
        asyncio.run(_gh(write).post(f"/repos/{REPO}/issues/1/comments", json={"body": "x"}))
    assert written.value.ambiguous is True
    assert len(write.sent) == 1


def test_broker_refusals_are_typed_and_its_busy_signal_is_waited_once(
    waits: list[float],
) -> None:
    refused = FakeBroker(lambda _: Refusal(403, "scope_not_allowed", "Not in the binding."))
    with pytest.raises(github.GitHubConnectionRefused) as raised:
        asyncio.run(_gh(refused).post(f"/repos/{REPO}/merges", json={}))
    assert raised.value.code == "scope_not_allowed"
    assert "Not in the binding." in str(raised.value)

    busy = FakeBroker(
        _answers(Refusal(429, "too_many_calls", headers={"retry-after": "2"}), Answer(body=[]))
    )
    assert asyncio.run(_gh(busy).get(f"/repos/{REPO}/pulls")) == []
    assert waits == [2.0]

    down = FakeBroker(_answers(Refusal(502, "upstream_failed"), Answer(body={})))
    assert asyncio.run(_gh(down).get(f"/repos/{REPO}")) == {}


def test_read_only_refuses_writes_before_sending_and_lists_them() -> None:
    broker = FakeBroker(lambda sent: Answer(body={"data": {"viewer": {"login": "x"}}}))
    gh = _gh(broker, read_only=True)

    async def dry_run() -> None:
        await gh.get(f"/repos/{REPO}/pulls/1")
        with pytest.raises(github.GitHubWriteRefused, match="refused POST"):
            await gh.post(f"/repos/{REPO}/issues/1/comments", json={"body": "x"})
        with pytest.raises(github.GitHubWriteRefused, match="a GraphQL mutation on acme/widgets"):
            await gh.graphql(
                "# reads first\nmutation Move($id: ID!) { updateRefs(input: {}) { clientMutationId } }",
                repository=REPO,
            )
        await gh.graphql(
            'query { viewer { login } } # not a mutation\n# "mutation"',
            repository=REPO,
        )

    asyncio.run(dry_run())
    assert gh.refused == (
        github.RefusedWrite("POST", f"/repos/{REPO}/issues/1/comments", REPO),
        github.RefusedWrite("POST", "/graphql", REPO),
    )
    assert [sent.method for sent in broker.sent] == ["GET", "POST"]


def test_a_read_only_view_shares_the_slot_and_keeps_its_own_record() -> None:
    broker = FakeBroker(lambda _: Answer(status=201, body={}))
    gh = _gh(broker, slot="github-read")
    view = gh.read_only_view()
    with pytest.raises(github.GitHubWriteRefused):
        asyncio.run(view.delete(f"/repos/{REPO}/git/refs/heads/x"))
    asyncio.run(gh.put(f"/repos/{REPO}/pulls/1/merge", json={"sha": "a"}))
    assert (view.slot, view.read_only, gh.read_only) == ("github-read", True, False)
    assert len(view.refused) == 1 and gh.refused == ()
    assert [sent.slot for sent in broker.sent] == ["github-read"]


def test_is_mutation_ignores_strings_and_comments() -> None:
    assert github.is_mutation("mutation { addComment(input: {}) { clientMutationId } }")
    assert github.is_mutation('query A { a } mutation B { b(x: "q") { c } }')
    assert not github.is_mutation('query { search(query: "mutation") { issueCount } }')
    assert not github.is_mutation('query { a(x: """ a\n mutation """) } # mutation')
    assert not github.is_mutation("query { repository { mutationTesting: name } }")


_ROLLUP = (
    "repository",
    "pullRequest",
    "commits",
    "nodes",
    0,
    "commit",
    "statusCheckRollup",
    "contexts",
    "nodes",
    2,
)


def _forbidden_check() -> dict[str, Any]:
    """GitHub's answer when one check run of a pull isn't readable with this credential."""

    return {
        "data": {"repository": {"pullRequest": {"number": 12, "commits": {"nodes": []}}}},
        "errors": [
            {
                "type": "FORBIDDEN",
                "path": list(_ROLLUP),
                "message": "Resource not accessible by integration",
            }
        ],
    }


def test_forbidden_on_one_check_run_fails_the_answer_unless_accepted() -> None:
    broker = FakeBroker(lambda _: Answer(body=_forbidden_check()))
    gh = _gh(broker)
    query = 'query($n: Int!) { repository(owner: "acme", name: "widgets") { pullRequest(number: $n) { number } } }'

    with pytest.raises(github.GitHubGraphQLFailed) as failed:
        asyncio.run(gh.graphql(query, {"n": 12}, repository=REPO))
    assert failed.value.data == _forbidden_check()["data"]
    assert failed.value.errors == (
        github.GraphQLError("FORBIDDEN", "Resource not accessible by integration", _ROLLUP),
    )
    assert "FORBIDDEN at repository.pullRequest" in str(failed.value)

    answer = asyncio.run(
        gh.graphql(
            query, {"n": 12}, repository=REPO, accept=github.forbidden_under("statusCheckRollup")
        )
    )
    assert answer.data["repository"]["pullRequest"]["number"] == 12
    assert [error.type for error in answer.accepted] == ["FORBIDDEN"]

    sent = broker.sent[0]
    assert (sent.method, sent.path) == ("POST", "/graphql")
    assert sent.headers["x-cayu-github-repository"] == REPO
    assert sent.json() == {"query": query, "variables": {"n": 12}}


def test_an_accepted_filter_leaves_every_other_error_failing() -> None:
    body = _forbidden_check()
    body["errors"].append({"type": "FORBIDDEN", "path": ["repository", "issues"], "message": "no"})
    gh = _gh(FakeBroker(lambda _: Answer(body=body)))
    with pytest.raises(github.GitHubGraphQLFailed, match="FORBIDDEN at repository.issues"):
        asyncio.run(
            gh.graphql(
                "query { a }", repository=REPO, accept=github.forbidden_under("statusCheckRollup")
            )
        )

    under = github.errors_under("repository", "pullRequest", types={"FORBIDDEN"})
    assert under(github.GraphQLError("FORBIDDEN", "", _ROLLUP))
    assert not under(github.GraphQLError("NOT_FOUND", "", _ROLLUP))
    assert not under(github.GraphQLError("FORBIDDEN", "", ("repository", "issues")))
    assert github.errors_under("p12")(github.GraphQLError(None, "", ("p12", "commits")))


def test_a_null_repository_is_not_found_and_no_data_fails() -> None:
    missing = {
        "data": {"repository": None},
        "errors": [{"type": "NOT_FOUND", "path": ["repository"], "message": "Could not resolve"}],
    }
    gh = _gh(FakeBroker(_answers(Answer(body=missing), Answer(body={"data": None, "errors": []}))))
    with pytest.raises(github.GitHubNotFound, match="acme/widgets isn't there"):
        asyncio.run(gh.graphql("query { repository { id } }", repository=REPO))
    with pytest.raises(github.GitHubGraphQLFailed, match="no data") as failed:
        asyncio.run(gh.graphql("query { repository { id } }", repository=REPO))
    assert failed.value.data is None


def test_a_graphql_query_is_retried_after_a_server_failure_and_a_mutation_is_not() -> None:
    query_broker = FakeBroker(_answers(Answer(status=502), Answer(body={"data": {"a": 1}})))
    assert asyncio.run(_gh(query_broker).graphql("{ a }", repository=REPO)).data == {"a": 1}
    assert len(query_broker.sent) == 2

    mutation_broker = FakeBroker(_answers(Answer(status=502)))
    with pytest.raises(github.GitHubUnavailable) as raised:
        asyncio.run(_gh(mutation_broker).graphql("mutation { a }", repository=REPO))
    assert raised.value.ambiguous is True
    assert len(mutation_broker.sent) == 1
    with pytest.raises(ValueError, match="owner/name"):
        asyncio.run(_gh(mutation_broker).graphql("{ a }", repository="acme"))


def test_text_follows_a_log_redirect_without_the_credential() -> None:
    location = "https://productionresultssa3.blob.core.windows.net/logs/job.txt?sig=secret"
    broker = FakeBroker(lambda _: Answer(status=302, headers={"location": location}))
    downloads: list[httpx.Request] = []

    def blob(request: httpx.Request) -> httpx.Response:
        downloads.append(request)
        return httpx.Response(200, content=b"step 1\nError: boom\n")

    gh = _gh(broker, transport=httpx.MockTransport(blob))
    text = asyncio.run(gh.text(f"/repos/{REPO}/actions/jobs/11/logs"))

    assert text == "step 1\nError: boom\n"
    [download] = downloads
    assert str(download.url) == location
    assert "authorization" not in download.headers
    assert broker.sent[0].headers["accept"] == "application/vnd.github.raw+json"


def test_text_keeps_the_end_of_a_long_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "MAX_TEXT_BYTES", 10)
    location = "https://pipelines.actions.githubusercontent.com/x/logs"
    broker = FakeBroker(lambda _: Answer(status=302, headers={"location": location}))
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=b"0123456789abcdef"))
    assert asyncio.run(
        _gh(broker, transport=transport).text("/repos/acme/widgets/actions/jobs/1/logs")
    ) == ("6789abcdef")


def test_redirects_elsewhere_are_refused_without_naming_the_target() -> None:
    elsewhere = "https://evil.example/steal?token=secret"
    broker = FakeBroker(
        _answers(
            Answer(status=302, headers={"location": elsewhere}),
            Answer(status=301, headers={"location": f"{API}/repositories/1"}),
        )
    )
    gh = _gh(broker)
    with pytest.raises(github.GitHubError, match="away from its log hosts") as away:
        asyncio.run(gh.text(f"/repos/{REPO}/actions/jobs/1/logs"))
    assert "secret" not in str(away.value)
    with pytest.raises(github.GitHubError, match="renamed or moved") as moved:
        asyncio.run(gh.get(f"/repos/{REPO}"))
    assert moved.value.status == 301


def test_paths_are_encoded_and_must_name_a_repository_well() -> None:
    broker = FakeBroker(lambda _: Answer(body={}))
    gh = _gh(broker)
    asyncio.run(gh.get(f"/repos/{REPO}/git/ref/heads/feature/é b"))
    assert broker.sent[0].path == f"/repos/{REPO}/git/ref/heads/feature/%C3%A9%20b"
    for path in ("repos/acme/widgets", "/repos/acme", "/repos/ac me/widgets/pulls"):
        with pytest.raises(ValueError):
            asyncio.run(gh.get(path))
    asyncio.run(gh.get("/rate_limit"))
    assert "x-cayu-github-repository" not in broker.sent[-1].headers


class _Counting:
    """A connection client that records how many calls it has in flight at most."""

    def __init__(self) -> None:
        self.active = 0
        self.most = 0

    async def request(
        self,
        slot: str,
        method: str,
        path: str,
        *,
        query: Query | None = None,
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
    ) -> ConnectionResponse:
        self.active += 1
        self.most = max(self.most, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ConnectionResponse(200, (), json.dumps({"path": path}).encode())


def test_calls_in_flight_are_capped() -> None:
    counting = _Counting()
    gh = github.GitHubClient(counting)

    async def many() -> None:
        await asyncio.gather(*(gh.get(f"/repos/{REPO}/pulls/{n}") for n in range(20)))

    asyncio.run(many())
    assert counting.most == 6


def test_a_comment_ended_by_a_lone_carriage_return_doesnt_hide_a_mutation() -> None:
    for document in (
        "# x\rmutation { a }",
        "# x\r\nmutation { a }",
        '{ a(s: "q\r") } mutation { b }',
    ):
        assert github.is_mutation(document), document

    broker = FakeBroker(lambda _: Answer(body={"data": {"a": 1}}))
    gh = _gh(broker, read_only=True)
    with pytest.raises(github.GitHubWriteRefused):
        asyncio.run(gh.graphql("# read\rmutation { a }", repository=REPO))
    assert broker.sent == [] and len(gh.refused) == 1

    failing = FakeBroker(_answers(Answer(status=502), Answer(body={"data": {"a": 1}})))
    with pytest.raises(github.GitHubUnavailable) as raised:
        asyncio.run(_gh(failing).graphql("# read\rmutation { a }", repository=REPO))
    assert raised.value.ambiguous is True
    assert len(failing.sent) == 1


def test_a_failed_log_download_keeps_no_trace_of_the_signed_url() -> None:
    location = "https://productionresultssa3.blob.core.windows.net/logs/job.txt?sig=secret"
    broker = FakeBroker(lambda _: Answer(status=302, headers={"location": location}))

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    gh = _gh(broker, transport=httpx.MockTransport(unreachable))
    with pytest.raises(github.GitHubUnavailable, match="ConnectError") as raised:
        asyncio.run(gh.text(f"/repos/{REPO}/actions/jobs/11/logs"))
    assert raised.value.__cause__ is None and raised.value.__context__ is None
    assert "secret" not in str(raised.value)


def test_paginate_stops_at_a_next_link_that_repeats_the_page() -> None:
    link = f'<{API}/repos/{REPO}/pulls?page=2>; rel="next"'
    broker = FakeBroker(lambda _: Answer(body=[{"number": 1}], headers={"link": link}))

    async def every() -> list[Any]:
        return [item async for item in _gh(broker).paginate(f"/repos/{REPO}/pulls")]

    with pytest.raises(github.GitHubError, match="repeats a page already read"):
        asyncio.run(every())
    assert len(broker.sent) == 2


def test_dot_names_are_not_repositories_and_a_literal_percent_is_encoded() -> None:
    broker = FakeBroker(lambda _: Answer(body={"data": {"a": 1}}))
    gh = _gh(broker)
    for repository in ("acme/.", "acme/.."):
        with pytest.raises(ValueError, match="owner/name"):
            asyncio.run(gh.graphql("{ a }", repository=repository))
        with pytest.raises(ValueError, match="owner/name"):
            asyncio.run(gh.get(f"/repos/{repository}/pulls"))
    asyncio.run(gh.get(f"/repos/{REPO}/git/ref/heads/100%-done"))
    asyncio.run(gh.get(f"/repos/{REPO}/contents/a%20b"))
    assert [sent.path for sent in broker.sent] == [
        f"/repos/{REPO}/git/ref/heads/100%25-done",
        f"/repos/{REPO}/contents/a%20b",
    ]


def _graphql_limited(reset: int) -> Answer:
    return Answer(
        body={"errors": [{"type": "RATE_LIMITED", "message": "API rate limit exceeded"}]},
        headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset)},
    )


def test_a_graphql_rate_limit_in_a_200_is_waited_out_or_raised(waits: list[float]) -> None:
    soon = FakeBroker(
        _answers(_graphql_limited(int(time.time()) + 20), Answer(body={"data": {"a": 1}}))
    )
    assert asyncio.run(_gh(soon).graphql("{ a }", repository=REPO)).data == {"a": 1}
    assert len(soon.sent) == 2
    [wait] = waits
    assert 15 <= wait <= 21

    later = FakeBroker(_answers(_graphql_limited(int(time.time()) + 3600)))
    with pytest.raises(github.GitHubRateLimited, match="API rate limit exceeded") as raised:
        asyncio.run(_gh(later).graphql("{ a }", repository=REPO))
    assert raised.value.secondary is False
    assert (raised.value.retry_after or 0) > 3500
    assert len(later.sent) == 1


def test_an_accepted_null_repository_is_the_callers_to_read() -> None:
    missing = {
        "data": {"repository": None},
        "errors": [{"type": "NOT_FOUND", "path": ["repository"], "message": "Could not resolve"}],
    }
    gh = _gh(FakeBroker(lambda _: Answer(body=missing)))
    answer = asyncio.run(
        gh.graphql(
            "query { repository { id } }",
            repository=REPO,
            accept=github.errors_under("repository", types={"NOT_FOUND"}),
        )
    )
    assert answer.data == {"repository": None}
    assert [error.type for error in answer.accepted] == ["NOT_FOUND"]


def test_one_client_serves_one_event_loop_after_another() -> None:
    counting = _Counting()
    gh = github.GitHubClient(counting)

    async def many() -> None:
        await asyncio.gather(*(gh.get(f"/repos/{REPO}/pulls/{n}") for n in range(20)))

    asyncio.run(many())
    asyncio.run(many())
    assert counting.most == 6


def test_paginate_refuses_a_cycle_back_to_a_page_already_read() -> None:
    def connector(sent: Sent) -> Answer:
        page = sent.query.get("page", ["1"])[0]
        following = "1" if page == "2" else "2"
        link = f'<{API}/repos/{REPO}/pulls?per_page=100&page={following}>; rel="next"'
        return Answer(body=[{"page": page}], headers={"link": link})

    broker = FakeBroker(connector)

    async def every() -> list[Any]:
        return [item async for item in _gh(broker).paginate(f"/repos/{REPO}/pulls")]

    with pytest.raises(github.GitHubError, match="repeats a page already read"):
        asyncio.run(every())
    assert [sent.query.get("page") for sent in broker.sent] == [None, ["2"], ["1"]]


_TWO_COMMENTS = "mutation { a: addComment(input: {}) { clientMutationId } b: addComment(input: {}) { clientMutationId } }"


def test_a_rate_limited_mutation_is_resent_only_when_none_of_it_ran(waits: list[float]) -> None:
    partly = FakeBroker(
        lambda _: Answer(
            body={
                "data": {"a": {"clientMutationId": None}, "b": None},
                "errors": [{"type": "RATE_LIMITED", "path": ["b"], "message": "slow down"}],
            },
            headers={"x-ratelimit-reset": str(int(time.time()) + 5)},
        )
    )
    with pytest.raises(github.GitHubGraphQLFailed, match="RATE_LIMITED at b") as raised:
        asyncio.run(_gh(partly).graphql(_TWO_COMMENTS, repository=REPO))
    assert raised.value.ambiguous is True
    assert raised.value.data == {"a": {"clientMutationId": None}, "b": None}
    assert len(partly.sent) == 1 and waits == []

    nothing_ran = FakeBroker(
        _answers(
            Answer(body={"errors": [{"type": "RATE_LIMITED", "message": "slow down"}]}),
            Answer(body={"data": {"a": {}, "b": {}}}),
        )
    )
    assert asyncio.run(_gh(nothing_ran).graphql(_TWO_COMMENTS, repository=REPO)).data == {
        "a": {},
        "b": {},
    }
    assert len(nothing_ran.sent) == 2


def test_a_client_keeps_no_closed_event_loop_alive() -> None:
    gh = github.GitHubClient(_Counting())
    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

    async def many() -> None:
        loops.append(weakref.ref(asyncio.get_running_loop()))
        await asyncio.gather(*(gh.get(f"/repos/{REPO}/pulls/{n}") for n in range(20)))

    for _ in range(3):
        asyncio.run(many())
    gc.collect()
    assert [loop() is None for loop in loops[:-1]] == [True, True]


def _owner_paged(path: str, by_id: str) -> FakeBroker:
    """A two-page listing at ``path`` whose next link names its owner by id, as GitHub does."""

    def connector(sent: Sent) -> Answer:
        if sent.query.get("page") == ["2"]:
            return Answer(body=[{"id": 2}])
        link = f'<{API}{by_id}?per_page=100&page=2>; rel="next"'
        return Answer(body=[{"id": 1}], headers={"link": link})

    return FakeBroker(connector)


def test_listings_page_through_links_naming_their_owner_by_id() -> None:
    for path, by_id in (
        ("/orgs/acme/repos", "/organizations/9919/repos"),
        ("/users/octocat/repos", "/user/583231/repos"),
        ("/orgs/acme/teams/core/members", "/organizations/9919/team/42/members"),
        (f"/repos/{REPO}/issues/12/comments", "/repositories/1296269/issues/12/comments"),
    ):
        broker = _owner_paged(path, by_id)
        page = asyncio.run(_gh(broker).list(path))
        assert page == github.Page(items=({"id": 1}, {"id": 2}), complete=True)
        assert [sent.path for sent in broker.sent] == [path, path]

    for path, other in (
        ("/orgs/acme/repos", "/organizations/9919/members"),
        ("/orgs/acme/teams/core/members", "/organizations/9919/team/42/repos"),
        ("/orgs/acme/repos", "/orgs/other/repos"),
    ):
        broker = _owner_paged(path, other)
        with pytest.raises(github.GitHubError, match="isn't the same listing"):
            asyncio.run(_gh(broker).list(path))


_COMMENTS = f"/repos/{REPO}/issues/1/comments"


@pytest.mark.parametrize(
    ("failure", "error", "ambiguous"),
    [
        (Answer(status=500), github.GitHubUnavailable, True),
        (Answer(status=502), github.GitHubUnavailable, True),
        (Refusal(502, "upstream_failed"), github.GitHubUnavailable, True),
        (Refusal(502, "upstream_response_too_large"), github.GitHubConnectionRefused, True),
        (Refusal(502, "a_code_this_release_doesnt_know"), github.GitHubConnectionRefused, True),
        (Refusal(422, "scope_not_allowed"), github.GitHubConnectionRefused, False),
        (Refusal(409, "connection_not_ready"), github.GitHubConnectionRefused, False),
        (
            Refusal(429, "too_many_calls", headers={"retry-after": "2"}),
            github.GitHubRateLimited,
            False,
        ),
        (Refusal(429, "too_many_calls"), github.GitHubRateLimited, False),
        (Answer(status=429, headers={"retry-after": "5"}), github.GitHubRateLimited, False),
        (Answer(status=409, body={"message": "Conflict"}), github.GitHubConflict, False),
    ],
)
def test_without_rate_limit_waits_a_failed_write_is_sent_once_and_ambiguous_when_it_may_have_run(
    failure: Answer | Refusal, error: type[github.GitHubError], ambiguous: bool
) -> None:
    broker = FakeBroker(_answers(failure, Answer(status=201, body={"id": 1})))
    gh = _gh(broker, max_rate_limit_wait_seconds=0)
    with pytest.raises(error) as raised:
        asyncio.run(gh.post(_COMMENTS, json={"body": "x"}))
    assert type(raised.value) is error
    assert raised.value.ambiguous is ambiguous
    assert len(broker.sent) == 1


def test_without_rate_limit_waits_a_write_github_refused_unexecuted_is_resent_once(
    waits: list[float],
) -> None:
    past = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(time.time()) - 5)}
    for limited in (Answer(status=429, headers={"retry-after": "0"}), Answer(403, headers=past)):
        broker = FakeBroker(_answers(limited, Answer(status=201, body={"id": 1})))
        gh = _gh(broker, max_rate_limit_wait_seconds=0)
        assert asyncio.run(gh.post(_COMMENTS, json={"body": "x"})) == {"id": 1}
        assert len(broker.sent) == 2

        twice = FakeBroker(_answers(limited))
        with pytest.raises(github.GitHubRateLimited):
            asyncio.run(_gh(twice, max_rate_limit_wait_seconds=0).post(_COMMENTS, json={}))
        assert len(twice.sent) == 2
    assert waits == [0.0] * 4


def test_an_answer_too_large_for_a_read_is_refused_unambiguous_and_not_retried() -> None:
    broker = FakeBroker(_answers(Refusal(502, "upstream_response_too_large")))
    with pytest.raises(
        github.GitHubConnectionRefused, match="GET .* was answered too large"
    ) as raised:
        asyncio.run(_gh(broker).get(f"/repos/{REPO}/pulls"))
    assert (raised.value.code, raised.value.ambiguous) == ("upstream_response_too_large", False)
    assert len(broker.sent) == 1


@pytest.mark.parametrize(
    "answer",
    [
        Answer(body={"data": None, "errors": [{"message": "Something went wrong"}]}),
        Answer(body={"errors": [{"type": "INTERNAL", "message": "Timeout"}]}),
        Answer(body=b"<html>busy</html>"),
    ],
)
def test_a_failed_mutation_is_ambiguous_even_without_data_and_a_query_isnt(answer: Answer) -> None:
    mutation = FakeBroker(_answers(answer))
    with pytest.raises(github.GitHubGraphQLFailed) as raised:
        asyncio.run(
            _gh(mutation, max_rate_limit_wait_seconds=0).graphql(
                "mutation { markPullRequestReadyForReview(input: {}) { clientMutationId } }",
                repository=REPO,
            )
        )
    assert raised.value.ambiguous is True
    assert len(mutation.sent) == 1

    query = FakeBroker(_answers(answer))
    with pytest.raises(github.GitHubGraphQLFailed) as raised:
        asyncio.run(_gh(query, read_retries=0).graphql("{ viewer { login } }", repository=REPO))
    assert raised.value.ambiguous is False


def test_a_slot_failure_is_worded_by_what_happened_to_the_call() -> None:
    def failure(refusal: Refusal, method: str) -> str:
        broker = FakeBroker(_answers(refusal))
        with pytest.raises(github.GitHubError) as raised:
            asyncio.run(_gh(broker, read_retries=0).request(method, f"/repos/{REPO}/pulls"))
        return str(raised.value)

    too_large = Refusal(502, "upstream_response_too_large")
    assert f"POST /repos/{REPO}/pulls" in failure(too_large, "POST")
    assert "may have sent POST" in failure(too_large, "POST")
    assert f"GET /repos/{REPO}/pulls was answered too large" in failure(too_large, "GET")
    assert "got no answer to GET" in failure(Refusal(502, "upstream_failed"), "GET")
    unsent = failure(Refusal(422, "path_not_allowed"), "POST")
    assert "didn't send POST" in unsent and "may have sent" not in unsent


# The login a client writes as, and Cayu Cloud's GitHub App refusals.

_APP_SLOT = {
    "slot": "github",
    "connection": "triage-app",
    "connector": "github-app",
    "status": "ready",
    "scopes": ["issues:write", "metadata:read"],
    "account": {"id": "123456", "name": "cayu-triage[bot]"},
}
#: Cayu Cloud's refusal of a call to a repository its GitHub App isn't installed on.
_NOT_INSTALLED = Refusal(
    422,
    "credential_unavailable",
    f"The GitHub App isn't installed on {REPO}, or can't see it; install it on the "
    "repository on GitHub.",
)


def test_login_on_cayu_cloud_is_the_slots_account_read_once() -> None:
    broker = FakeBroker(_answers(Answer()), slots=[{**_APP_SLOT, "slot": "other"}, _APP_SLOT])
    gh = _gh(broker)

    assert asyncio.run(gh.login()) == "cayu-triage[bot]"
    assert asyncio.run(gh.login()) == "cayu-triage[bot]"
    assert broker.listed == 1
    assert broker.sent == []


def test_login_on_cayu_cloud_says_when_the_slot_or_its_account_is_missing() -> None:
    with pytest.raises(github.GitHubConnectionRefused) as unbound:
        asyncio.run(_gh(FakeBroker(_answers(Answer()), slots=[])).login())
    assert unbound.value.code == "connection_unbound"
    assert (
        "`cayu cloud connections bind github CONNECTION --application AGENT`, and for a GitHub "
        "App `--repository OWNER/NAME`"
    ) in str(unbound.value)
    assert unbound.value.path == "/v1/agent-connections"

    for account in (None, {"id": "1", "name": ""}):
        slots = [{**_APP_SLOT, "account": account}]
        with pytest.raises(ValueError, match="lists no account for the connection of slot github"):
            asyncio.run(_gh(FakeBroker(_answers(Answer()), slots=slots)).login())


def test_login_with_a_token_is_its_owner() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={"login": "octocat", "type": "User"})

    gh = github.GitHubClient(
        github.token_connections("ghp_test", transport=httpx.MockTransport(handler))
    )
    assert asyncio.run(gh.login()) == "octocat"
    assert asyncio.run(gh.read_only_view().login()) == "octocat"
    assert seen == ["/user"]


def test_login_of_an_installation_token_is_the_one_named_or_asked_for() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(403, json={"message": "Resource not accessible by integration"})

    transport = httpx.MockTransport(handler)
    named = github.token_connections("ghs_test", login="github-actions[bot]", transport=transport)
    assert asyncio.run(github.GitHubClient(named).login()) == "github-actions[bot]"
    assert seen == []

    unnamed = github.GitHubClient(github.token_connections("ghs_test", transport=transport))
    with pytest.raises(ValueError, match=r"token_connections\(token, login=\.\.\.\)"):
        asyncio.run(unnamed.login())
    assert seen == ["/user"]
    with pytest.raises(ValueError, match="login, when given"):
        github.token_connections("ghs_test", login=" ")


def test_login_of_a_revoked_token_raises_githubs_refusal() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(401, json={"message": "Bad credentials"})
    )
    gh = github.GitHubClient(github.token_connections("ghp_revoked", transport=transport))
    with pytest.raises(github.GitHubPermissionDenied, match="Bad credentials") as raised:
        asyncio.run(gh.login())
    assert raised.value.status == 401


def test_login_of_a_fake_github_is_whom_it_writes_as() -> None:
    from cayu.connectors.github.testing import FakeGitHub

    assert asyncio.run(github.GitHubClient(FakeGitHub(login="herald[bot]")).login()) == (
        "herald[bot]"
    )


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_an_app_not_installed_on_cayu_cloud_is_raised_unsent_and_not_retried(method: str) -> None:
    broker = FakeBroker(_answers(_NOT_INSTALLED))

    with pytest.raises(github.GitHubAppNotInstalled) as raised:
        asyncio.run(_gh(broker).request(method, f"/repos/{REPO}/issues"))

    assert raised.value.repository == REPO
    assert raised.value.ambiguous is False
    assert (raised.value.method, raised.value.path) == (method, f"/repos/{REPO}/issues")
    assert str(raised.value).endswith(f"{method} /repos/{REPO}/issues wasn't carried out.")
    assert len(broker.sent) == 1


def test_an_app_not_installed_on_cayu_cloud_fails_a_graphql_query_unsent() -> None:
    broker = FakeBroker(_answers(_NOT_INSTALLED))

    with pytest.raises(github.GitHubAppNotInstalled) as raised:
        asyncio.run(_gh(broker).graphql("{ viewer { login } }", repository=REPO))

    assert (raised.value.repository, raised.value.path) == (REPO, "/graphql")
    assert len(broker.sent) == 1


def test_any_other_unavailable_credential_is_still_retried_for_a_read() -> None:
    broker = FakeBroker(_answers(Refusal(422, "credential_unavailable", "GitHub 5xx.")))

    with pytest.raises(github.GitHubUnavailable) as raised:
        asyncio.run(_gh(broker, read_retries=1).get(f"/repos/{REPO}/issues"))

    assert not isinstance(raised.value, github.GitHubAppNotInstalled)
    assert len(broker.sent) == 2


def _listing_broker(handler: Callable[[httpx.Request], httpx.Response]) -> CloudConnectionBroker:
    return CloudConnectionBroker(
        url="https://api.example.test/v1/agent-connections",
        token="cayu_conn_test",
        transport=httpx.MockTransport(handler),
    )


def test_a_refused_listing_is_refused_and_an_unreachable_one_unavailable() -> None:
    refused = httpx.Response(
        401, json={"detail": {"code": "agent_credential_invalid", "message": "No."}}
    )
    gh = github.GitHubClient(_listing_broker(lambda _: refused))
    with pytest.raises(github.GitHubConnectionRefused) as raised:
        asyncio.run(gh.login())
    assert raised.value.code == "agent_credential_invalid"
    assert (raised.value.status, raised.value.ambiguous) == (401, False)
    assert raised.value.path == "/v1/agent-connections"
    assert "didn't list the slot github: No." in str(raised.value)

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    for handler in (unreachable, lambda _: httpx.Response(503)):
        with pytest.raises(github.GitHubUnavailable) as down:
            asyncio.run(github.GitHubClient(_listing_broker(handler)).login())
        assert down.value.ambiguous is False


def test_concurrent_first_logins_list_the_slot_once_for_the_client_and_its_views() -> None:
    broker = FakeBroker(_answers(Answer()), slots=[_APP_SLOT])
    gh = _gh(broker)
    view = gh.read_only_view()

    async def both() -> list[str]:
        return list(await asyncio.gather(gh.login(), view.login(), gh.login()))

    assert asyncio.run(both()) == ["cayu-triage[bot]"] * 3
    assert broker.listed == 1
