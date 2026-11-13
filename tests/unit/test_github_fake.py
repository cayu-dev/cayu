"""The fake GitHub an application's tests import, driven through the real ``GitHubClient``:
paging, writes read back, GitHub's refusals, reruns, injected failures, read-only mode, the
repository header and GraphQL answers."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from cayu.connections import ConnectionCallError, ConnectionResponse
from cayu.connectors import github
from cayu.connectors.github import REPOSITORY_HEADER
from cayu.connectors.github import client as client_module
from cayu.connectors.github import pulls as github_pulls
from cayu.connectors.github.testing import Call, FakeGitHub, reply

REPO = "acme/widgets"
BOT = "cayu-shepherd[bot]"
run = asyncio.run


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every wait the client makes, recorded instead of slept."""

    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(client_module, "_sleep", record)
    return waited


def _fake(**settings: Any) -> FakeGitHub:
    fake = FakeGitHub(login=BOT, **settings)
    fake.add_repository(REPO)
    return fake


def test_a_listing_is_paged_with_githubs_links_and_read_whole() -> None:
    fake = _fake(page_size=2)
    for number in range(1, 6):
        fake.add_pull(REPO, number, title=f"Change {number}")
    fake.add_pull(REPO, 6, state="closed")

    page = run(fake.client().list(f"/repos/{REPO}/pulls", query={"state": "open"}))

    assert page.complete
    assert [pull["number"] for pull in page.items] == [5, 4, 3, 2, 1]
    assert [call.query.get("page") for call in fake.requests] == [None, "2", "3"]
    assert all(call.query["state"] == "open" for call in fake.requests)


def test_a_capped_listing_says_it_stopped_early() -> None:
    fake = _fake(page_size=2)
    for number in range(1, 6):
        fake.add_issue(REPO, number)

    page = run(fake.client().list(f"/repos/{REPO}/issues", max_pages=2))

    assert not page.complete
    assert len(page.items) == 4


def test_every_call_names_its_repository_in_the_header() -> None:
    fake = _fake()
    fake.add_pull(REPO, 7)
    fake.on_graphql("Viewer", lambda variables, repository: {"data": {"viewer": {"login": "x"}}})
    gh = fake.client()

    run(gh.get(f"/repos/{REPO}/pulls/7"))
    run(gh.graphql("query Viewer { viewer { login } }", repository=REPO))

    assert [call.repository for call in fake.requests] == [REPO, REPO]
    assert all(call.headers["x-github-api-version"] == "2022-11-28" for call in fake.requests)


def test_a_call_whose_header_names_another_repository_is_refused() -> None:
    fake = _fake()

    with pytest.raises(ConnectionCallError) as refused:
        run(
            fake.request(
                "github", "GET", f"/repos/{REPO}/pulls", headers={REPOSITORY_HEADER: "a/b"}
            )
        )
    with pytest.raises(ConnectionCallError):
        run(fake.request("github", "POST", "/graphql", body=b"{}"))

    assert refused.value.code == "headers_invalid"


def test_a_comment_is_written_under_the_login_and_read_back() -> None:
    fake = _fake()
    fake.add_pull(REPO, 7)
    fake.add_comment(REPO, 7, "Looks risky.", author="alice")
    gh = fake.client()

    posted = run(gh.post(f"/repos/{REPO}/issues/7/comments", json={"body": "Rerunning CI."}))
    comments = run(gh.list(f"/repos/{REPO}/issues/7/comments")).items

    assert [(c["user"]["login"], c["user"]["type"], c["body"]) for c in comments] == [
        ("alice", "User", "Looks risky."),
        (BOT, "Bot", "Rerunning CI."),
    ]
    assert posted["id"] == comments[1]["id"]
    assert posted["html_url"].endswith(f"/issues/7#issuecomment-{posted['id']}")
    assert [(call.method, call.path) for call in fake.writes()] == [
        ("POST", f"/repos/{REPO}/issues/7/comments")
    ]


def test_comments_since_a_time_and_an_edited_comment() -> None:
    now = datetime(2026, 11, 12, 8, 0, tzinfo=UTC)
    clock = [now - timedelta(days=2)]
    fake = _fake(clock=lambda: clock[0])
    fake.add_issue(REPO, 3)
    old = fake.add_comment(REPO, 3, "old")
    clock[0] = now
    fake.add_comment(REPO, 3, "new")
    gh = fake.client()

    since = run(gh.get(f"/repos/{REPO}/issues/3/comments", query={"since": "2026-11-11T00:00:00Z"}))
    run(gh.patch(f"/repos/{REPO}/issues/comments/{old.id}", json={"body": "edited"}))
    again = run(gh.get(f"/repos/{REPO}/issues/3/comments", query={"since": "2026-11-11T00:00:00Z"}))

    assert [comment["body"] for comment in since] == ["new"]
    assert [comment["body"] for comment in again] == ["edited", "new"]


def test_an_issue_is_created_labelled_and_closed() -> None:
    fake = _fake()
    fake.add_pull(REPO, 1)
    gh = fake.client()

    created = run(gh.post(f"/repos/{REPO}/issues", json={"title": "Flaky test", "labels": ["ci"]}))
    number = created["number"]
    run(gh.post(f"/repos/{REPO}/issues/{number}/labels", json={"labels": ["triaged", "ci"]}))
    run(gh.delete(f"/repos/{REPO}/issues/{number}/labels/ci"))
    labelled = run(gh.list(f"/repos/{REPO}/issues", query={"labels": "triaged"})).items
    run(gh.patch(f"/repos/{REPO}/issues/{number}", json={"state": "closed"}))
    still_open = run(gh.list(f"/repos/{REPO}/issues")).items

    assert number == 2
    assert created["user"]["login"] == BOT
    assert [(item["number"], [label["name"] for label in item["labels"]]) for item in labelled] == [
        (2, ["triaged"])
    ]
    assert [(item["number"], "pull_request" in item) for item in still_open] == [(1, True)]


def test_a_label_with_a_slash_is_removed_by_its_encoded_name() -> None:
    fake = _fake()
    fake.add_issue(REPO, 1, author=BOT)
    gh = fake.client()

    run(gh.post(f"/repos/{REPO}/issues/1/labels", json={"labels": ["area/runtime", "ci"]}))
    run(gh.delete(f"/repos/{REPO}/issues/1/labels/area%2Fruntime"))

    assert fake.requests[-1].path == f"/repos/{REPO}/issues/1/labels/area%2Fruntime"
    assert fake.repositories[REPO].issues[1].labels == ["ci"]


def test_issues_are_listed_by_their_creator_ignoring_case() -> None:
    fake = _fake()
    fake.add_issue(REPO, 1, author=BOT)
    fake.add_issue(REPO, 2, author="octocat")
    gh = fake.client()

    def created_by(login: str) -> list[int]:
        query = {"state": "all", "creator": login}
        return [item["number"] for item in run(gh.list(f"/repos/{REPO}/issues", query=query)).items]

    assert created_by(BOT) == [1]
    assert created_by(BOT.upper()) == [1]
    assert created_by("someone-else") == []
    assert not fake.unanswered


def test_listings_show_a_new_item_only_after_the_listing_lag() -> None:
    now = [datetime(2026, 11, 12, 9, tzinfo=UTC)]
    fake = FakeGitHub(login=BOT, clock=lambda: now[0], listing_lag=timedelta(seconds=5))
    fake.add_repository(REPO)
    gh = fake.client()
    issue = run(gh.post(f"/repos/{REPO}/issues", json={"title": "New"}))
    comment = run(gh.post(f"/repos/{REPO}/issues/1/comments", json={"body": "Hi"}))

    def listed() -> tuple[list[int], list[int]]:
        issues = run(gh.list(f"/repos/{REPO}/issues", query={"state": "all"})).items
        comments = run(gh.list(f"/repos/{REPO}/issues/1/comments")).items
        return [item["number"] for item in issues], [item["id"] for item in comments]

    assert listed() == ([], [])
    # A direct read finds both at once.
    assert run(gh.get(f"/repos/{REPO}/issues/{issue['number']}"))["number"] == 1
    read = run(gh.get(f"/repos/{REPO}/issues/comments/{comment['id']}"))
    assert read["issue_url"].endswith(f"/repos/{REPO}/issues/1")
    now[0] += timedelta(seconds=5)
    assert listed() == ([1], [comment["id"]])
    assert not fake.unanswered


def test_an_invalid_write_is_refused_as_github_does() -> None:
    fake = _fake()
    fake.add_issue(REPO, 1)
    gh = fake.client()

    with pytest.raises(github.GitHubConflict) as missing_body:
        run(gh.post(f"/repos/{REPO}/issues/1/comments", json={}))
    with pytest.raises(github.GitHubNotFound):
        run(gh.delete(f"/repos/{REPO}/issues/1/labels/absent"))

    assert missing_body.value.status == 422
    assert missing_body.value.codes == ("missing_field",)
    assert fake.repositories[REPO].issues[1].comments == []


def test_a_merge_at_a_stale_head_is_refused_and_one_at_the_head_merges() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7)
    gh = fake.client()

    with pytest.raises(github.GitHubConflict) as stale:
        run(gh.put(f"/repos/{REPO}/pulls/7/merge", json={"sha": "0" * 40}))
    open_after = run(gh.get(f"/repos/{REPO}/pulls/7"))
    merged = run(gh.put(f"/repos/{REPO}/pulls/7/merge", json={"sha": pull.head_sha}))
    after = run(gh.get(f"/repos/{REPO}/pulls/7"))
    with pytest.raises(github.GitHubError) as again:
        run(gh.put(f"/repos/{REPO}/pulls/7/merge", json={"sha": pull.head_sha}))

    assert stale.value.status == 409
    assert "Head branch was modified" in str(stale.value)
    assert (open_after["state"], open_after["merged"]) == ("open", False)
    assert merged["merged"] is True
    assert (after["state"], after["merged"], after["merged_by"]["login"]) == ("closed", True, BOT)
    assert after["merge_commit_sha"] == merged["sha"]
    assert again.value.status == 405


def test_a_branch_update_checks_the_expected_head_and_leaves_it() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7, behind_by=3)
    gh = fake.client()
    path = f"/repos/{REPO}/pulls/7/update-branch"

    with pytest.raises(github.GitHubConflict) as stale:
        run(gh.put(path, json={"expected_head_sha": "0" * 40}))
    accepted = run(gh.request("PUT", path, json={"expected_head_sha": pull.head_sha}))
    behind = run(gh.get(f"/repos/{REPO}/compare/main...{pull.head_sha}"))

    assert stale.value.status == 422
    assert accepted.status == 202
    assert run(gh.get(f"/repos/{REPO}/pulls/7"))["head"]["sha"] == pull.head_sha
    assert behind["behind_by"] == 3


@pytest.mark.parametrize(
    ("settings", "sha", "status", "text"),
    [
        ({"draft": True, "mergeable": False}, "stale", 405, "Pull Request is still a draft"),
        ({"draft": True, "state": "closed"}, None, 405, "Pull Request is still a draft"),
        ({"state": "closed", "mergeable": False}, None, 405, "Pull Request is not mergeable"),
        ({"mergeable": False}, "stale", 405, "Pull Request has merge conflicts"),
        (
            {"mergeable": False, "behind_by": 1, "requires_up_to_date": True},
            "stale",
            405,
            "Pull Request has merge conflicts",
        ),
        (
            {"behind_by": 1, "requires_up_to_date": True},
            "stale",
            405,
            "Head branch is out of date. Review and try the merge again.",
        ),
        (
            {"behind_by": 1},
            "stale",
            409,
            "Head branch was modified. Review and try the merge again.",
        ),
    ],
)
def test_a_merge_is_refused_in_githubs_order_with_its_body(
    settings: dict[str, Any], sha: str | None, status: int, text: str
) -> None:
    fake = _fake()
    fake.add_pull(REPO, 7, **settings)
    body = json.dumps({"sha": "0" * 40} if sha else {}).encode()

    answer = run(
        fake.request(
            "github",
            "PUT",
            f"/repos/{REPO}/pulls/7/merge",
            headers={REPOSITORY_HEADER: REPO},
            body=body,
        )
    )

    assert answer.status == status
    assert json.loads(answer.body) == {
        "message": text,
        "documentation_url": "https://docs.github.com/rest",
        "status": str(status),
    }


def test_a_merge_while_mergeability_reads_null_is_accepted() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7, mergeable_unknown_reads=5)
    gh = fake.client()

    merged = run(gh.put(f"/repos/{REPO}/pulls/7/merge", json={"sha": pull.head_sha}))
    after = run(gh.get(f"/repos/{REPO}/pulls/7"))

    assert merged["message"] == "Pull Request successfully merged"
    assert (after["mergeable"], after["mergeable_state"]) == (None, "unknown")


@pytest.mark.parametrize(
    ("settings", "expected", "text"),
    [
        ({"mergeable": False}, "stale", "expected head sha didn\u2019t match current head ref."),
        ({"mergeable": False}, None, "merge conflict between base and head"),
        ({"state": "closed"}, None, "merge conflict between base and head"),
        ({"behind_by": 0}, None, "There are no new commits on the base branch."),
    ],
)
def test_a_branch_update_is_refused_in_githubs_order(
    settings: dict[str, Any], expected: str | None, text: str
) -> None:
    fake = _fake()
    fake.add_pull(REPO, 7, **{"behind_by": 2, **settings})
    asked = {"expected_head_sha": "0" * 40} if expected else {}

    with pytest.raises(github.GitHubConflict) as refused:
        run(fake.client().put(f"/repos/{REPO}/pulls/7/update-branch", json=asked))

    assert refused.value.status == 422
    assert f": {text}; see https://docs.github.com/rest" in str(refused.value)


@pytest.mark.parametrize(("private", "default"), [(True, "none"), (False, "read")])
def test_a_permission_is_a_collaborators_or_the_repositorys_default(
    private: bool, default: str
) -> None:
    fake = FakeGitHub(login=BOT)
    fake.add_repository(REPO, private=private, permissions={"alice": "write"})
    fake.users.add("bob")
    gh = fake.client()

    def permission(login: str) -> tuple[str, str]:
        answer = run(gh.get(f"/repos/{REPO}/collaborators/{login}/permission"))
        return answer["permission"], answer["role_name"]

    assert permission("alice") == ("write", "write")
    assert permission("bob") == (default, "" if default == "none" else default)
    with pytest.raises(github.GitHubNotFound, match="ghost is not a user"):
        permission("ghost")


def test_mergeability_is_unknown_on_the_first_read_after_a_push() -> None:
    fake = _fake()
    fake.add_pull(REPO, 7, mergeable=False, mergeable_unknown_reads=1)
    gh = fake.client()

    first = run(gh.get(f"/repos/{REPO}/pulls/7"))
    second = run(gh.get(f"/repos/{REPO}/pulls/7"))

    assert (first["mergeable"], first["mergeable_state"]) == (None, "unknown")
    assert (second["mergeable"], second["mergeable_state"]) == (False, "dirty")


def test_a_rerun_of_failed_jobs_starts_the_next_attempt() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7)
    ci = fake.add_workflow_run(REPO, pull.head_sha, conclusion="failure")
    fake.add_check_run(REPO, pull.head_sha, "lint", run=ci)
    failed = fake.add_check_run(REPO, pull.head_sha, "tests", run=ci, conclusion="failure")
    gh = fake.client()
    checks = f"/repos/{REPO}/commits/{pull.head_sha}/check-runs"

    run(gh.post(f"/repos/{REPO}/actions/runs/{ci.id}/rerun-failed-jobs"))
    runs = run(
        gh.list(
            f"/repos/{REPO}/actions/runs",
            query={"head_sha": pull.head_sha},
            item_key="workflow_runs",
        )
    )
    latest = run(gh.list(checks, item_key="check_runs")).items
    every = run(gh.list(checks, query={"filter": "all"}, item_key="check_runs")).items
    with pytest.raises(github.GitHubPermissionDenied):
        run(gh.post(f"/repos/{REPO}/actions/runs/{ci.id}/rerun-failed-jobs"))
    fake.finish_run(REPO, ci.id, "success")
    finished = run(gh.list(checks, item_key="check_runs")).items
    with pytest.raises(github.GitHubPermissionDenied, match="This workflow run cannot be retried"):
        run(gh.post(f"/repos/{REPO}/actions/runs/{ci.id}/rerun-failed-jobs"))

    [rerun] = runs.items
    assert (rerun["run_attempt"], rerun["status"], rerun["conclusion"]) == (2, "queued", None)
    assert [(c["name"], c["status"]) for c in latest] == [
        ("lint", "completed"),
        ("tests", "queued"),
    ]
    assert failed.id not in {c["id"] for c in latest}
    assert len(every) == 3
    assert {c["conclusion"] for c in finished} == {"success"}


def test_a_jobs_log_is_read_through_its_redirect_without_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_network(*args: object, **kwargs: object) -> httpx.Response:
        raise AssertionError("A test reached the network.")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    fake = _fake()
    fake.set_log(REPO, 42, "line 1\nFAILED test_cache\n")
    gh = github.GitHubClient(fake)

    text = run(gh.text(f"/repos/{REPO}/actions/jobs/42/logs"))

    assert text.endswith("FAILED test_cache\n")
    with pytest.raises(github.GitHubNotFound):
        run(gh.text(f"/repos/{REPO}/actions/jobs/43/logs"))


def test_an_injected_rate_limit_is_waited_out(waits: list[float]) -> None:
    fake = _fake()
    fake.add_pull(REPO, 7)
    fake.rate_limit("GET", "/repos/*/pulls/7", retry_after=0)
    fake.rate_limit("POST", "*/comments", retry_after=5, secondary=True)
    gh = fake.client()

    pull = run(gh.get(f"/repos/{REPO}/pulls/7"))
    run(gh.post(f"/repos/{REPO}/issues/7/comments", json={"body": "hi"}))

    assert pull["number"] == 7
    assert waits == [0.0, 5.0]
    assert [call.method for call in fake.requests] == ["GET", "GET", "POST", "POST"]
    assert len(fake.repositories[REPO].issues[7].comments) == 1


@pytest.mark.parametrize("year", [2020, 2030])
def test_an_injected_rate_limit_resets_in_wall_time_whatever_the_fakes_clock(
    waits: list[float], year: int
) -> None:
    fake = _fake(clock=lambda: datetime(year, 1, 1, tzinfo=UTC))
    fake.add_issue(REPO, 1)
    fake.rate_limit(path=f"/repos/{REPO}/issues/1", retry_after=5)
    gh = fake.client()

    issue = run(gh.get(f"/repos/{REPO}/issues/1"))
    fake.rate_limit(path=f"/repos/{REPO}/issues/1", retry_after=600)
    with pytest.raises(github.GitHubRateLimited) as limited:
        run(gh.get(f"/repos/{REPO}/issues/1"))

    assert issue["number"] == 1
    assert len(waits) == 1 and 3 <= waits[0] <= 5
    assert not limited.value.secondary
    assert limited.value.retry_after is not None and 590 <= limited.value.retry_after <= 600


def test_server_failures_and_timeouts_are_retried_for_reads_only() -> None:
    fake = _fake()
    fake.add_issue(REPO, 1)
    fake.fail("GET", f"/repos/{REPO}/issues/1", status=502)
    fake.refuse("upstream_failed", "GET", f"/repos/{REPO}/issues/1")
    fake.fail("POST", status=500)
    gh = fake.client()

    issue = run(gh.get(f"/repos/{REPO}/issues/1"))
    with pytest.raises(github.GitHubUnavailable) as write:
        run(gh.post(f"/repos/{REPO}/issues/1/comments", json={"body": "hi"}))

    assert issue["number"] == 1
    assert write.value.ambiguous
    assert fake.repositories[REPO].issues[1].comments == []


def test_an_injected_status_and_a_brokers_refusal_reach_the_caller_typed() -> None:
    fake = _fake()
    fake.add_issue(REPO, 1)
    fake.fail("GET", f"/repos/{REPO}/issues/1", status=404, body={"message": "Not Found"})
    fake.refuse("scope_not_allowed", "POST")
    gh = fake.client()

    with pytest.raises(github.GitHubNotFound):
        run(gh.get(f"/repos/{REPO}/issues/1"))
    with pytest.raises(github.GitHubConnectionRefused) as refused:
        run(gh.post(f"/repos/{REPO}/issues", json={"title": "x"}))

    assert refused.value.code == "scope_not_allowed"
    assert run(gh.get(f"/repos/{REPO}/issues/1"))["number"] == 1


def test_a_read_only_client_never_reaches_the_fake() -> None:
    fake = _fake()
    fake.add_pull(REPO, 7)
    gh = fake.client(read_only=True)

    with pytest.raises(github.GitHubWriteRefused):
        run(gh.post(f"/repos/{REPO}/issues/7/comments", json={"body": "hi"}))
    with pytest.raises(github.GitHubWriteRefused):
        run(gh.graphql("mutation Close { closePullRequest { clientMutationId } }", repository=REPO))

    assert fake.requests == []
    assert [write.path for write in gh.refused] == [f"/repos/{REPO}/issues/7/comments", "/graphql"]


def test_graphql_is_answered_per_named_operation() -> None:
    fake = _fake()
    seen: list[tuple[dict[str, Any], str]] = []

    def checks(variables: Any, repository: str) -> dict[str, Any]:
        seen.append((dict(variables), repository))
        rollup = {"statusCheckRollup": None}
        error = {
            "type": "FORBIDDEN",
            "message": "no",
            "path": ["repository", "pullRequest", "statusCheckRollup"],
        }
        return {"data": {"repository": {"pullRequest": rollup}}, "errors": [error]}

    fake.on_graphql("PullChecks", checks)
    fake.on_graphql("Label", lambda variables, repository: {"data": {"addLabelsToLabelable": {}}})
    gh = fake.client()
    query = "# the checks\nquery PullChecks($number: Int!) { repository { pullRequest } }"

    answer = run(
        gh.graphql(
            query,
            {"number": 7},
            repository=REPO,
            accept=github.forbidden_under("statusCheckRollup"),
        )
    )
    with pytest.raises(github.GitHubGraphQLFailed):
        run(gh.graphql(query, {"number": 7}, repository=REPO))
    run(gh.graphql("mutation Label { addLabelsToLabelable { clientMutationId } }", repository=REPO))
    with pytest.raises(NotImplementedError, match="on_graphql"):
        run(gh.graphql("query Unknown { viewer { login } }", repository=REPO))

    assert [error.type for error in answer.accepted] == ["FORBIDDEN"]
    assert seen[0] == ({"number": 7}, REPO)
    assert [call.path for call in fake.writes()] == ["/graphql"]


def test_an_unknown_repository_is_not_found_and_an_unknown_route_is_not_guessed() -> None:
    fake = _fake()
    gh = fake.client()

    with pytest.raises(github.GitHubNotFound):
        run(gh.get("/repos/acme/missing/pulls"))
    with pytest.raises(NotImplementedError, match="route"):
        run(gh.get(f"/repos/{REPO}/deployments"))
    with pytest.raises(NotImplementedError, match="created"):
        run(gh.get(f"/repos/{REPO}/actions/runs", query={"created": ">=2026-11-01"}))

    assert [call.path for call in fake.unanswered] == [
        f"/repos/{REPO}/deployments",
        f"/repos/{REPO}/actions/runs",
    ]


def test_an_unanswered_call_is_kept_even_when_the_application_swallows_it() -> None:
    fake = _fake()

    async def log_or_placeholder() -> str:
        try:
            return await fake.client().text(f"/repos/{REPO}/actions/runs/1/logs")
        except Exception as error:  # an application's fall-back
            return f"(log unavailable: {type(error).__name__})"

    assert run(log_or_placeholder()) == "(log unavailable: NotImplementedError)"
    assert [call.path for call in fake.unanswered] == [f"/repos/{REPO}/actions/runs/1/logs"]


def test_a_write_body_field_the_fake_doesnt_model_is_refused() -> None:
    fake = _fake()
    fake.add_issue(REPO, 1)

    with pytest.raises(NotImplementedError, match="assignees"):
        run(fake.client().patch(f"/repos/{REPO}/issues/1", json={"assignees": ["alice"]}))

    assert len(fake.unanswered) == 1
    assert fake.repositories[REPO].issues[1].state == "open"


def test_compare_status_and_behind_only_under_branch_protection() -> None:
    fake = _fake()
    behind = fake.add_pull(REPO, 1, behind_by=2, ahead_by=0)
    both = fake.add_pull(REPO, 2, behind_by=2, ahead_by=1, requires_up_to_date=True)
    level = fake.add_pull(REPO, 3, ahead_by=0)
    gh = fake.client()

    def status(head: str, base: str = "main") -> str:
        return run(gh.get(f"/repos/{REPO}/compare/{base}...{head}"))["status"]

    assert [status(behind.head_sha), status(both.head_sha), status(level.head_sha)] == [
        "behind",
        "diverged",
        "identical",
    ]
    assert status("main") == "identical"
    assert run(gh.get(f"/repos/{REPO}/pulls/1"))["mergeable_state"] == "clean"
    assert run(gh.get(f"/repos/{REPO}/pulls/2"))["mergeable_state"] == "behind"


def test_owner_and_name_match_ignoring_case_and_a_zero_page_size_is_the_default() -> None:
    fake = FakeGitHub(login=BOT)
    fake.add_repository("Acme/Widgets")
    for number in range(1, 36):
        fake.add_issue("acme/widgets", number)

    page = run(fake.client().list(f"/repos/{REPO}/issues", query={"per_page": "0"}))

    assert page.complete and len(page.items) == 35
    assert [call.query.get("page") for call in fake.requests] == [None, "2"]


def test_a_test_adds_a_route_the_fake_lacks() -> None:
    fake = _fake()

    def deployments(call: Call, groups: tuple[str, ...]) -> ConnectionResponse:
        return reply(200, [{"id": 1, "environment": call.query.get("environment")}])

    fake.route("GET", r"deployments", deployments)
    found = run(fake.client().get(f"/repos/{REPO}/deployments", query={"environment": "prod"}))

    assert found == [{"id": 1, "environment": "prod"}]


def test_a_check_run_listing_reads_a_pulls_head_by_branch_and_reviews_by_author() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7, head_ref="scout/fix-7")
    fake.add_check_run(
        REPO, pull.head_sha, "cayu-reviewer", app_slug="cayu-reviewer", output_title="5/5"
    )
    fake.add_review(REPO, 7, "cayu-reviewer[bot]", state="APPROVED", body="5/5")
    gh = fake.client()

    checks = run(gh.get(f"/repos/{REPO}/commits/scout/fix-7/check-runs"))
    reviews = run(gh.get(f"/repos/{REPO}/pulls/7/reviews"))

    assert [(c["app"]["slug"], c["output"]["title"]) for c in checks["check_runs"]] == [
        ("cayu-reviewer", "5/5")
    ]
    assert [(r["user"]["login"], r["commit_id"]) for r in reviews] == [
        ("cayu-reviewer[bot]", pull.head_sha)
    ]


def test_a_pull_listing_filters_its_head_by_owner_and_branch() -> None:
    fake = _fake()
    fake.add_pull(REPO, 7, head_ref="fix")
    gh = fake.client()

    def numbers(head: str) -> list[int]:
        return [
            pull["number"] for pull in run(gh.get(f"/repos/{REPO}/pulls", query={"head": head}))
        ]

    assert numbers("acme:fix") == [7]
    assert numbers("ACME:fix") == [7]
    assert numbers("someone-else:fix") == []
    assert numbers("acme:other") == []
    with pytest.raises(NotImplementedError, match="owner:branch"):
        numbers("fix")
    assert len(fake.unanswered) == 1


def test_pulls_sorted_by_update_default_to_oldest_first_and_issues_to_newest() -> None:
    now = datetime(2026, 11, 12, tzinfo=UTC)
    fake = _fake()
    first = fake.add_pull(REPO, 1, updated_at=now)
    fake.add_pull(REPO, 2, updated_at=now - timedelta(days=1))
    gh = fake.client()

    def numbers(path: str, **query: str) -> list[int]:
        return [item["number"] for item in run(gh.get(f"/repos/{REPO}/{path}", query=query))]

    assert numbers("pulls", sort="updated") == [2, 1]
    assert numbers("pulls", sort="updated", direction="desc") == [1, 2]
    assert numbers("pulls") == [2, 1]
    assert numbers("issues", sort="updated") == [1, 2]
    assert first.number == 1


def test_a_reopened_issue_says_so() -> None:
    fake = _fake()
    fake.add_issue(REPO, 5)
    gh = fake.client()

    run(gh.patch(f"/repos/{REPO}/issues/5", json={"state": "closed", "state_reason": "completed"}))
    reopened = run(gh.patch(f"/repos/{REPO}/issues/5", json={"state": "open"}))

    assert (reopened["state"], reopened["state_reason"]) == ("open", "reopened")


def test_a_since_with_no_offset_is_read_as_utc() -> None:
    now = datetime(2026, 11, 12, 8, 0, tzinfo=UTC)
    fake = _fake(clock=lambda: now)
    fake.add_issue(REPO, 1)
    fake.add_comment(REPO, 1, "hi")
    gh = fake.client()

    issues = run(gh.get(f"/repos/{REPO}/issues", query={"since": "2026-11-12T07:59:00"}))
    later = run(gh.get(f"/repos/{REPO}/issues/1/comments", query={"since": "2026-11-12T08:01:00"}))

    assert [issue["number"] for issue in issues] == [1]
    assert later == []


def test_labels_are_added_from_an_array_and_kept_once() -> None:
    fake = _fake()
    fake.add_issue(REPO, 1)
    gh = fake.client()
    path = f"/repos/{REPO}/issues/1/labels"

    added = run(gh.post(path, json=["bug", "bug"]))
    run(gh.put(path, json={"labels": ["ready", "ready"]}))
    run(gh.delete(f"{path}/ready"))

    assert [label["name"] for label in added] == ["bug"]
    assert fake.repositories[REPO].issues[1].labels == []


def test_a_commits_combined_status_takes_each_contexts_latest() -> None:
    now = datetime(2026, 11, 12, tzinfo=UTC)
    fake = _fake()
    pull = fake.add_pull(REPO, 7)
    fake.add_status(REPO, pull.head_sha, "deploy", state="failure", created_at=now)
    fake.add_status(REPO, pull.head_sha, "deploy", state="success", created_at=now + timedelta(1))
    fake.add_status(REPO, pull.head_sha, "lint", state="pending", created_at=now)
    gh = fake.client()

    combined = run(gh.get(f"/repos/{REPO}/commits/{pull.head_sha}/status"))
    empty = run(gh.get(f"/repos/{REPO}/commits/{'0' * 40}/status"))

    assert combined["state"] == "pending"
    assert [(s["context"], s["state"]) for s in combined["statuses"]] == [
        ("deploy", "success"),
        ("lint", "pending"),
    ]
    assert (empty["state"], empty["statuses"]) == ("pending", [])


def test_the_pull_request_read_model_reads_a_pull_from_the_fake() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7, behind_by=2, labels=["ready"], author="cayu-scout[bot]")
    ci = fake.add_workflow_run(REPO, pull.head_sha, conclusion="failure")
    fake.add_check_run(REPO, pull.head_sha, "tests", run=ci, conclusion="failure")
    fake.add_status(REPO, pull.head_sha, "deploy", state="success")
    fake.add_review(REPO, 7, "cayu-reviewer[bot]", state="APPROVED")
    fake.add_comment(REPO, 7, "Looks risky.", author="alice")

    facts = run(github_pulls.read_pull(fake.client(read_only=True), REPO, 7))

    assert (facts.state, facts.labels, facts.author) == ("open", ("ready",), "cayu-scout[bot]")
    assert (facts.behind_by, facts.ahead_by, facts.gaps) == (2, 1, ())
    assert facts.checks.state == "failed"
    assert facts.checks.failed_names == ("tests",)
    assert [review.author for review in facts.reviews] == ["cayu-reviewer[bot]"]
    assert [comment.author for comment in facts.comments or ()] == ["alice"]
    assert not fake.unanswered


def test_the_compare_query_resolves_branches_but_not_pull_request_refs() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7, head_ref="fix", behind_by=3, ahead_by=2)
    query = """query PullRequestCompare(
      $base: String!, $head: String!, $branch: String!, $pull: String!
    ) {
      repository(owner: "acme", name: "widgets") {
        ref(qualifiedName: $base) { compare(headRef: $head) { aheadBy behindBy } }
        branch: ref(qualifiedName: $branch) { target { oid } }
        pull: ref(qualifiedName: $pull) { target { oid } }
      }
    }"""

    def compared(head: str) -> dict[str, Any]:
        variables = {"base": "refs/heads/main", "head": head}
        variables |= {"branch": "refs/heads/fix", "pull": "refs/pull/7/head"}
        return run(fake.client().graphql(query, variables, repository=REPO)).data["repository"]

    # As GitHub's GraphQL answers: a pull request's ref doesn't resolve, though a compare takes
    # it, the head commit or the head branch.
    for head in (pull.head_sha, "refs/pull/7/head", "fix"):
        assert compared(head) == {
            "ref": {"compare": {"aheadBy": 2, "behindBy": 3}},
            "branch": {"target": {"oid": pull.head_sha}},
            "pull": None,
        }
    assert compared("b" * 40)["ref"] == {"compare": None}


def test_the_compare_query_takes_the_pull_on_the_compared_base() -> None:
    fake = _fake()
    fake.add_pull(REPO, 7, head_sha="c" * 40, behind_by=1)
    fake.add_pull(REPO, 8, head_sha="c" * 40, base="release", behind_by=4)

    facts = run(github_pulls.read_pull(fake.client(read_only=True), REPO, 8))

    assert (facts.behind_by, facts.ahead_by, facts.gaps) == (4, 1, ())


def test_the_pull_request_listing_reads_its_answer_from_on_graphql() -> None:
    fake = _fake()
    node = {
        "number": 7,
        "title": "Fix the cache",
        "url": f"https://github.com/{REPO}/pull/7",
        "state": "OPEN",
        "isDraft": False,
        "createdAt": "2026-11-12T00:00:00Z",
        "updatedAt": "2026-11-12T00:00:00Z",
        "mergedAt": None,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "CLEAN",
        "baseRefName": "main",
        "headRefName": "fix",
        "headRefOid": "a" * 40,
        "author": {"__typename": "Bot", "login": "cayu-scout"},
        "mergedBy": None,
        "labels": {"nodes": [{"name": "ready"}]},
        "assignees": {"nodes": []},
        "reviewRequests": {"nodes": []},
        "reviews": {"pageInfo": {"hasPreviousPage": False}, "nodes": []},
        "commits": {"nodes": []},
    }
    page = {"pageInfo": {"hasNextPage": False, "endCursor": None}, "nodes": [node]}
    fake.on_graphql(
        "PullRequestFacts",
        lambda variables, repository: {"data": {"repository": {"pullRequests": page}}},
    )

    listed = run(github_pulls.list_pulls(fake.client(), REPO))

    assert listed.complete
    assert [(p.number, p.author, p.labels) for p in listed.pulls] == [
        (7, "cayu-scout[bot]", ("ready",))
    ]
    assert not fake.unanswered
