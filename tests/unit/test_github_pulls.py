"""The pull request read model: every fact of one pull request through REST, a listing through
GraphQL, what the check words mean, and a fact that can't be read reported on its own."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import pytest
from tests.connector_broker_support import Answer, FakeBroker, Refusal, Sent

from cayu.connectors import github
from cayu.connectors.github import pulls as pulls_module
from cayu.connectors.github.pulls import CheckState, PullCheck, PullChecks

REPO = "acme/widgets"
SHA = "a" * 40
ROOT = f"/repos/{REPO}"
REVIEWER = "cayu-reviewer"
FORBIDDEN_CHECK = "Resource not accessible by integration"
GRAPHQL = "/graphql"
TOO_LARGE = Refusal(502, "upstream_response_too_large", "The answer is over 5 MiB.")


def _compared(compare: dict[str, int] | None, head: str = SHA) -> dict[str, Any]:
    """GraphQL's compare of the base branch with the pull request's head, and the commit that
    head named."""

    return {
        "data": {
            "repository": {
                "ref": None if compare is None else {"compare": compare},
                "head": {"target": {"oid": head}},
            }
        }
    }


def _graphql_error(kind: str, *path: str | int) -> dict[str, Any]:
    return {"type": kind, "message": f"{kind} here", "path": list(path)}


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every wait the read model makes, recorded instead of slept."""

    waited: list[float] = []

    async def record(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(pulls_module, "_sleep", record)
    return waited


def _at(day: int, hour: int = 9) -> str:
    return f"2026-11-{day:02d}T{hour:02d}:00:00Z"


def _when(day: int, hour: int = 9) -> datetime:
    return datetime(2026, 11, day, hour, tzinfo=UTC)


def _pull(**overrides: Any) -> dict[str, Any]:
    return {
        "number": 7,
        "title": "Add widgets",
        "html_url": f"https://github.com/{REPO}/pull/7",
        "state": "open",
        "draft": False,
        "user": {"login": "cayu-scout[bot]", "type": "Bot"},
        "labels": [{"name": "P0"}, {"name": "review: 4/5"}],
        "base": {"ref": "release/1.x"},
        "head": {"ref": "scout/widgets", "sha": SHA},
        "created_at": _at(10),
        "updated_at": _at(11),
        "merged_at": None,
        "merged_by": None,
        "assignees": [{"login": "ana"}],
        "requested_reviewers": [{"login": "ben"}],
        "requested_teams": [{"slug": "core"}],
        "mergeable": True,
        "mergeable_state": "clean",
        **overrides,
    }


def _check_run(
    run_id: int, name: str, conclusion: str | None, *, app: str = "github-actions", suite: int = 11
) -> dict[str, Any]:
    return {
        "id": run_id,
        "name": name,
        "status": "completed" if conclusion else "in_progress",
        "conclusion": conclusion,
        "app": {"slug": app},
        "check_suite": {"id": suite},
        "html_url": f"https://github.com/{REPO}/runs/{run_id}",
        "started_at": _at(11, 8),
        "completed_at": _at(11, 9) if conclusion else None,
        "output": {"title": None, "summary": None},
    }


def _routes(**changes: Answer | Refusal | Callable[[Sent], Answer]) -> dict[str, Any]:
    """Every GET a full read sends, by path, with GitHub's answers."""

    reviewer_check = _check_run(4, "Review", "success", app=REVIEWER, suite=12)
    reviewer_check["output"] = {"title": "5/5 · ready", "summary": "**Merge readiness: 5/5**"}
    routes: dict[str, Any] = {
        f"{ROOT}/pulls/7": Answer(body=_pull()),
        GRAPHQL: Answer(body=_compared({"aheadBy": 2, "behindBy": 3})),
        f"{ROOT}/git/commits/{SHA}": Answer(body={"committer": {"date": _at(11, 8)}}),
        f"{ROOT}/commits/{SHA}/check-runs": Answer(
            body={
                "total_count": 4,
                "check_runs": [
                    _check_run(1, "test", "failure"),
                    _check_run(2, "lint", "success"),
                    # A workflow job may take the reviewer's name: it is still CI.
                    _check_run(3, REVIEWER, "success"),
                    reviewer_check,
                ],
            }
        ),
        f"{ROOT}/actions/runs": Answer(
            body={
                "workflow_runs": [
                    {
                        "id": 901,
                        "check_suite_id": 11,
                        "run_attempt": 2,
                        "name": "CI",
                        "html_url": f"https://github.com/{REPO}/actions/runs/901",
                    }
                ]
            }
        ),
        f"{ROOT}/commits/{SHA}/status": Answer(
            body={
                "state": "pending",
                "statuses": [
                    {"context": "ci/legacy", "state": "success", "target_url": None},
                    {"context": "deploy/preview", "state": "pending", "target_url": None},
                ],
            }
        ),
        f"{ROOT}/pulls/7/reviews": Answer(
            body=[
                _review(1, "cayu-reviewer[bot]", "COMMENTED", _at(11, 10), body="3/5"),
                _review(2, "ana", "APPROVED", _at(11, 11)),
                _review(3, "cayu-reviewer[bot]", "COMMENTED", _at(11, 12), body="4/5"),
                _review(4, "me", "PENDING", None),
            ]
        ),
        f"{ROOT}/issues/7/comments": Answer(
            body=[
                {
                    "id": 50,
                    "user": None,
                    "body": "@cayu-reviewer review",
                    "created_at": _at(11, 13),
                    "updated_at": _at(11, 13),
                    "html_url": "https://github.com/c/50",
                }
            ]
        ),
    }
    routes.update(changes)
    return routes


def _review(
    review_id: int, login: str, state: str, at: str | None, *, body: str = ""
) -> dict[str, Any]:
    return {
        "id": review_id,
        "user": {"login": login},
        "state": state,
        "body": body,
        "commit_id": SHA,
        "submitted_at": at,
        "html_url": f"https://github.com/r/{review_id}",
    }


def _rest(routes: Mapping[str, Any]) -> FakeBroker:
    def connector(sent: Sent) -> Answer | Refusal:
        found = routes.get(sent.path)
        if found is None:
            return Answer(status=404, body={"message": "Not Found"})
        return found(sent) if callable(found) else found

    return FakeBroker(connector)


def _read(broker: FakeBroker, **kwargs: Any) -> github.PullRequestFacts:
    gh = github.GitHubClient(broker.client, read_retries=0)
    return asyncio.run(github.read_pull(gh, REPO, 7, **kwargs))


def test_read_pull_answers_every_fact_through_rest() -> None:
    broker = _rest(_routes())
    facts = _read(broker, reviewer_app=REVIEWER)

    assert (facts.number, facts.title, facts.author) == (7, "Add widgets", "cayu-scout[bot]")
    assert facts.state is github.PullState.OPEN and not facts.draft
    assert facts.labels == ("P0", "review: 4/5")
    assert (facts.base, facts.head_ref, facts.head_sha) == ("release/1.x", "scout/widgets", SHA)
    assert facts.head_committed_at == _when(11, 8)
    assert (facts.created_at, facts.updated_at, facts.merged_at) == (_when(10), _when(11), None)
    assert facts.assignees == ("ana",) and facts.requested_reviewers == ("ben",)
    assert facts.requested_teams == ("core",)
    assert (facts.mergeable, facts.mergeable_state) == (True, "clean")
    assert (facts.behind_by, facts.ahead_by) == (3, 2)
    assert facts.gaps == ()

    checks = facts.checks
    assert [check.name for check in checks.ci] == [
        "test",
        "lint",
        REVIEWER,
        "ci/legacy",
        "deploy/preview",
    ]
    assert [check.name for check in checks.reviewer] == ["Review"]
    assert checks.reviewer[0].title == "5/5 · ready"
    failed = checks.failed
    assert [check.name for check in failed] == ["test"] and checks.failed_names == ("test",)
    assert failed[0].id == 1 and failed[0].app == "github-actions"
    assert failed[0].workflow_run == github.CheckWorkflowRun(
        901, 2, "CI", f"https://github.com/{REPO}/actions/runs/901"
    )
    assert [check.name for check in checks.running] == ["deploy/preview"]
    assert checks.complete and checks.state is CheckState.FAILED

    assert [(review.id, review.author) for review in facts.reviews] == [
        (1, "cayu-reviewer[bot]"),
        (2, "ana"),
        (3, "cayu-reviewer[bot]"),
    ]
    # The reviewer's comment-only reviews stand for nothing; ana's approval stands.
    assert [review.id for review in facts.latest_reviews] == [2]
    assert facts.comments is not None
    assert [(comment.author, comment.body) for comment in facts.comments] == [
        ("ghost", "@cayu-reviewer review")
    ]

    assert {sent.method for sent in broker.sent if sent.path != GRAPHQL} == {"GET"}
    assert {sent.headers["x-cayu-github-repository"] for sent in broker.sent} == {REPO}
    by_path = {sent.path: sent for sent in broker.sent}
    compare = by_path[GRAPHQL].json()
    assert "query PullRequestCompare" in compare["query"]
    assert compare["variables"]["base"] == "refs/heads/release/1.x"
    assert compare["variables"]["head"] == "refs/pull/7/head"
    assert by_path[f"{ROOT}/commits/{SHA}/check-runs"].query["per_page"] == "30"
    assert by_path[f"{ROOT}/actions/runs"].query["head_sha"] == SHA


def test_without_a_reviewer_app_every_check_is_ci_and_comments_can_be_left_out() -> None:
    broker = _rest(_routes())
    facts = _read(broker, comments=False)

    assert facts.checks.reviewer == ()
    assert "Review" in [check.name for check in facts.checks.ci]
    assert facts.comments is None
    assert f"{ROOT}/issues/7/comments" not in [sent.path for sent in broker.sent]


def test_a_read_only_client_reads_a_whole_pull() -> None:
    gh = github.GitHubClient(_rest(_routes()).client).read_only_view()
    facts = asyncio.run(github.read_pull(gh, REPO, 7))

    assert facts.number == 7 and gh.refused == ()


def test_unknown_mergeability_is_read_again_once_after_the_wait(waits: list[float]) -> None:
    answers = iter([_pull(mergeable=None, mergeable_state="unknown"), _pull()])
    broker = _rest(_routes(**{f"{ROOT}/pulls/7": lambda _: Answer(body=next(answers))}))
    facts = _read(broker)

    assert (facts.mergeable, facts.mergeable_state) == (True, "clean")
    assert waits == [github.MERGEABLE_WAIT_SECONDS]
    assert [sent.path for sent in broker.sent].count(f"{ROOT}/pulls/7") == 2


def test_mergeability_still_unknown_after_the_wait_stays_unknown(waits: list[float]) -> None:
    unknown = Answer(body=_pull(mergeable=None, mergeable_state="unknown"))
    broker = _rest(_routes(**{f"{ROOT}/pulls/7": unknown}))
    facts = _read(broker, mergeable_wait_seconds=5)

    assert (facts.mergeable, facts.mergeable_state) == (None, "unknown")
    assert waits == [5]
    assert [sent.path for sent in broker.sent].count(f"{ROOT}/pulls/7") == 2


@pytest.mark.parametrize(
    ("pull", "wait"),
    [
        (_pull(state="closed", mergeable=None), 2.0),
        (_pull(mergeable=None), 0),
    ],
)
def test_a_closed_pull_or_no_wait_is_not_read_again(
    waits: list[float], pull: dict[str, Any], wait: float
) -> None:
    broker = _rest(_routes(**{f"{ROOT}/pulls/7": Answer(body=pull)}))
    facts = _read(broker, mergeable_wait_seconds=wait)

    assert facts.mergeable is None and waits == []


def test_a_merged_pull_reads_as_merged_and_a_dirty_one_as_conflicting() -> None:
    merged = _pull(state="closed", merged_at=_at(12), merged_by={"login": "ana"})
    facts = _read(_rest(_routes(**{f"{ROOT}/pulls/7": Answer(body=merged)})))
    assert facts.state is github.PullState.MERGED
    assert (facts.merged_at, facts.merged_by) == (_when(12), "ana")

    dirty = _pull(mergeable=True, mergeable_state="dirty")
    facts = _read(_rest(_routes(**{f"{ROOT}/pulls/7": Answer(body=dirty)})))
    assert (facts.mergeable, facts.mergeable_state) == (False, "dirty")


def test_a_fact_this_credential_cant_read_is_a_gap_and_the_rest_stands() -> None:
    denied = Answer(status=403, body={"message": FORBIDDEN_CHECK})
    passing = Answer(
        body={"check_runs": [_check_run(1, "test", "success"), _check_run(2, "lint", "success")]}
    )
    routes = _routes(
        **{
            f"{ROOT}/actions/runs": denied,
            f"{ROOT}/commits/{SHA}/status": denied,
            f"{ROOT}/commits/{SHA}/check-runs": passing,
        }
    )
    routes[GRAPHQL] = Answer(body=_compared(None))
    facts = _read(_rest(routes))

    assert [gap.fact for gap in facts.gaps] == ["behind_by", "workflow_runs", "checks"]
    assert FORBIDDEN_CHECK in facts.gaps[1].reason
    assert (facts.behind_by, facts.ahead_by) == (None, None)
    assert all(check.workflow_run is None for check in facts.checks.ci)
    assert not facts.checks.complete
    assert facts.checks.state is CheckState.UNKNOWN
    assert facts.head_committed_at == _when(11, 8)


def test_a_failure_stands_when_some_checks_were_not_read() -> None:
    routes = _routes(**{f"{ROOT}/commits/{SHA}/status": Answer(status=403, body={})})
    facts = _read(_rest(routes))

    assert not facts.checks.complete and facts.checks.state is CheckState.FAILED


def test_check_runs_on_several_pages_are_all_read() -> None:
    def page(sent: Sent) -> Answer:
        link = f'<https://api.github.com{ROOT}/commits/{SHA}/check-runs?page=2>; rel="next"'
        if sent.query.get("page") == ["2"]:
            return Answer(body={"check_runs": [_check_run(2, "lint", "success")]})
        return Answer(
            body={"check_runs": [_check_run(1, "test", "success")]}, headers={"link": link}
        )

    facts = _read(_rest(_routes(**{f"{ROOT}/commits/{SHA}/check-runs": page})))

    assert [check.name for check in facts.checks.ci][:2] == ["test", "lint"]
    assert facts.checks.complete


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (Answer(status=500, body={"message": "Server Error"}), github.GitHubUnavailable),
        (
            Answer(status=429, body={"message": "slow down"}, headers={"retry-after": "600"}),
            github.GitHubRateLimited,
        ),
    ],
)
def test_an_outage_or_rate_limit_still_fails_the_read(
    failure: Answer, error: type[github.GitHubError]
) -> None:
    with pytest.raises(error):
        _read(_rest(_routes(**{f"{ROOT}/pulls/7/reviews": failure})))


def test_a_compare_of_a_head_that_moved_during_the_read_is_a_gap() -> None:
    moved = Answer(body=_compared({"aheadBy": 1, "behindBy": 0}, head="b" * 40))
    facts = _read(_rest(_routes(**{GRAPHQL: moved})))

    assert (facts.behind_by, facts.ahead_by) == (None, None)
    assert [gap.fact for gap in facts.gaps] == ["behind_by"]
    assert "moved" in facts.gaps[0].reason and SHA in facts.gaps[0].reason


def test_a_missing_base_branch_is_a_gap_on_the_compare() -> None:
    missing = _compared(None)
    missing["errors"] = [_graphql_error("NOT_FOUND", "repository", "ref")]
    facts = _read(_rest(_routes(**{GRAPHQL: Answer(body=missing)})))

    assert facts.behind_by is None and [gap.fact for gap in facts.gaps] == ["behind_by"]


@pytest.mark.parametrize(
    ("kind", "error"),
    [("INTERNAL", github.GitHubGraphQLFailed), ("RATE_LIMITED", github.GitHubRateLimited)],
)
def test_a_github_failure_on_the_compare_still_fails_the_read(
    kind: str, error: type[github.GitHubError]
) -> None:
    failed = _compared(None)
    failed["errors"] = [_graphql_error(kind, "repository", "ref", "compare")]
    broker = _rest(_routes(**{GRAPHQL: Answer(body=failed)}))
    gh = github.GitHubClient(broker.client, read_retries=0, max_rate_limit_wait_seconds=0)
    with pytest.raises(error):
        asyncio.run(github.read_pull(gh, REPO, 7))


def test_an_answer_over_the_slots_size_limit_is_a_gap_on_that_fact() -> None:
    routes = _routes(
        **{
            GRAPHQL: TOO_LARGE,
            f"{ROOT}/commits/{SHA}/check-runs": TOO_LARGE,
            f"{ROOT}/pulls/7/reviews": TOO_LARGE,
            f"{ROOT}/issues/7/comments": TOO_LARGE,
        }
    )
    facts = _read(_rest(routes))

    assert [gap.fact for gap in facts.gaps] == ["behind_by", "checks", "reviews", "comments"]
    assert "upstream_response_too_large" in facts.gaps[0].reason
    assert (facts.behind_by, facts.reviews, facts.comments) == (None, (), None)
    assert facts.checks.state is CheckState.UNKNOWN
    assert facts.head_committed_at == _when(11, 8)


def test_a_comment_only_review_leaves_an_approval_standing() -> None:
    reviews = Answer(
        body=[
            _review(1, "alice", "APPROVED", _at(11, 10)),
            _review(2, "alice", "COMMENTED", _at(11, 11)),
            _review(3, "bob", "CHANGES_REQUESTED", _at(11, 12)),
            _review(4, "bob", "DISMISSED", _at(11, 13)),
        ]
    )
    facts = _read(_rest(_routes(**{f"{ROOT}/pulls/7/reviews": reviews})))

    assert [review.id for review in facts.reviews] == [1, 2, 3, 4]
    assert [(review.author, review.state) for review in facts.latest_reviews] == [
        ("alice", "APPROVED"),
        ("bob", "DISMISSED"),
    ]


def test_a_job_kept_from_an_earlier_attempt_names_its_run_without_an_attempt() -> None:
    kept = _check_run(5, "build", "success", suite=10)
    kept["details_url"] = f"https://github.com/{REPO}/actions/runs/901/job/5"
    rerun = _check_run(1, "test", "success")
    runs = {f"{ROOT}/commits/{SHA}/check-runs": Answer(body={"check_runs": [kept, rerun]})}
    build, test = _read(_rest(_routes(**runs))).checks.ci[:2]

    assert build.workflow_run == github.CheckWorkflowRun(
        901, None, "CI", f"https://github.com/{REPO}/actions/runs/901"
    )
    assert test.workflow_run is not None and test.workflow_run.attempt == 2


def test_the_pull_itself_missing_fails_the_read() -> None:
    routes = _routes()
    del routes[f"{ROOT}/pulls/7"]
    with pytest.raises(github.GitHubNotFound):
        _read(_rest(routes))


def _ci(name: str, status: str = "completed", conclusion: str | None = "success") -> PullCheck:
    return PullCheck(name=name, status=status, conclusion=conclusion)


@pytest.mark.parametrize(
    ("checks", "state"),
    [
        (PullChecks(), CheckState.NONE),
        (PullChecks(reviewer=(_ci("review", "in_progress", None),)), CheckState.NONE),
        (
            PullChecks((_ci("a"), _ci("b", conclusion="skipped"), _ci("c", conclusion="neutral"))),
            CheckState.GREEN,
        ),
        (PullChecks((_ci("a"), _ci("b", "queued", None))), CheckState.RUNNING),
        (
            PullChecks((_ci("a", conclusion="cancelled"), _ci("b", "queued", None))),
            CheckState.FAILED,
        ),
        (PullChecks((_ci("a", conclusion="stale"),)), CheckState.FAILED),
        (PullChecks((_ci("a", conclusion="action_required"),)), CheckState.FAILED),
        (PullChecks((_ci("a"),), complete=False), CheckState.UNKNOWN),
        (PullChecks(complete=False), CheckState.UNKNOWN),
        (PullChecks((_ci("a", conclusion="timed_out"),), complete=False), CheckState.FAILED),
    ],
)
def test_what_the_check_words_mean(checks: PullChecks, state: CheckState) -> None:
    assert checks.state is state


def test_read_pull_refuses_a_bad_number_or_wait() -> None:
    gh = github.GitHubClient(_rest({}).client)
    with pytest.raises(ValueError, match="positive"):
        asyncio.run(github.read_pull(gh, REPO, 0))
    with pytest.raises(ValueError, match="negative"):
        asyncio.run(github.read_pull(gh, REPO, 7, mergeable_wait_seconds=-1))


# The GraphQL listing.


def _actor(login: str, kind: str = "User") -> dict[str, str]:
    return {"__typename": kind, "login": login}


def _context_run(name: str, conclusion: str | None, app: str = "github-actions") -> dict[str, Any]:
    return {
        "__typename": "CheckRun",
        "databaseId": 100 + len(name),
        "name": name,
        "status": "COMPLETED" if conclusion else "IN_PROGRESS",
        "conclusion": conclusion,
        "title": None,
        "detailsUrl": f"https://github.com/{REPO}/actions/runs/901/job/{100 + len(name)}",
        "startedAt": _at(11, 8),
        "completedAt": _at(11, 9) if conclusion else None,
        "checkSuite": {
            "app": {"slug": app},
            "workflowRun": {"databaseId": 901, "url": "https://run"}
            if app == "github-actions"
            else None,
        },
    }


def _node(number: int, updated: str = _at(11), **overrides: Any) -> dict[str, Any]:
    contexts = overrides.pop(
        "contexts", [_context_run("test", "SUCCESS"), _context_run("Review", "SUCCESS", REVIEWER)]
    )
    return {
        "number": number,
        "title": f"Pull {number}",
        "url": f"https://github.com/{REPO}/pull/{number}",
        "state": "OPEN",
        "isDraft": False,
        "createdAt": _at(10),
        "updatedAt": updated,
        "mergedAt": None,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "CLEAN",
        "baseRefName": "main",
        "headRefName": f"feature/{number}",
        "headRefOid": SHA,
        "author": _actor("cayu-scout", "Bot"),
        "mergedBy": None,
        "labels": {"nodes": [{"name": "P0"}]},
        "assignees": {"nodes": [_actor("ana")]},
        "reviewRequests": {
            "nodes": [
                {"requestedReviewer": _actor("cayu-reviewer", "Bot")},
                {"requestedReviewer": {"__typename": "Team", "slug": "core"}},
            ]
        },
        "reviews": {
            "pageInfo": {"hasPreviousPage": False},
            "nodes": [
                {
                    "databaseId": 70,
                    "state": "COMMENTED",
                    "submittedAt": _at(11, 10),
                    "url": "https://github.com/r/70",
                    "author": _actor("cayu-reviewer", "Bot"),
                    "commit": {"oid": SHA},
                }
            ],
        },
        "commits": {
            "nodes": [
                {
                    "commit": {
                        "oid": SHA,
                        "committedDate": _at(11, 8),
                        "statusCheckRollup": {
                            "contexts": {
                                "totalCount": len(contexts),
                                "pageInfo": {"hasNextPage": False},
                                "nodes": contexts,
                            }
                        },
                    }
                }
            ]
        },
        **overrides,
    }


def _page(nodes: list[dict[str, Any]], *, more: bool = False, cursor: str = "c1") -> dict[str, Any]:
    return {
        "repository": {
            "pullRequests": {
                "pageInfo": {"hasNextPage": more, "endCursor": cursor},
                "nodes": nodes,
            }
        }
    }


def _graphql(pages: list[dict[str, Any]], mergeable: dict[str, Any] | None = None) -> FakeBroker:
    """GitHub's GraphQL API: the listing's pages in turn, and the mergeable re-read."""

    queue = list(pages)

    def connector(sent: Sent) -> Answer:
        assert (sent.method, sent.path) == ("POST", "/graphql")
        if "query Mergeable" in sent.json()["query"]:
            return Answer(body={"data": {"repository": mergeable or {}}})
        return Answer(body=queue.pop(0))

    return FakeBroker(connector)


def _list(broker: FakeBroker, **kwargs: Any) -> github.PullRequestList:
    gh = github.GitHubClient(broker.client, read_retries=0)
    return asyncio.run(github.list_pulls(gh, REPO, **kwargs))


def _queries(broker: FakeBroker) -> list[dict[str, Any]]:
    return [sent.json() for sent in broker.sent]


def test_list_pulls_reads_open_pulls_with_bot_logins_as_rest_writes_them() -> None:
    broker = _graphql([{"data": _page([_node(7)])}])
    listing = _list(broker, reviewer_app=REVIEWER)

    assert listing.complete
    (pull,) = listing.pulls
    assert pull.author == "cayu-scout[bot]"
    assert pull.state is github.PullState.OPEN
    assert pull.requested_reviewers == ("cayu-reviewer[bot]",)
    assert pull.requested_teams == ("core",) and pull.assignees == ("ana",)
    assert [(review.author, review.body) for review in pull.reviews] == [
        ("cayu-reviewer[bot]", None)
    ]
    assert (pull.mergeable, pull.mergeable_state) == (True, "clean")
    assert pull.head_committed_at == _when(11, 8)
    assert (pull.behind_by, pull.comments) == (None, None)
    assert [check.name for check in pull.checks.ci] == ["test"]
    assert [check.name for check in pull.checks.reviewer] == ["Review"]
    assert pull.checks.ci[0].workflow_run == github.CheckWorkflowRun(901, None, "", "https://run")
    assert pull.checks.state is CheckState.GREEN and pull.gaps == ()
    (query,) = _queries(broker)
    assert query["variables"]["states"] == ["OPEN"] and query["variables"]["first"] == 20
    # Small enough for GitHub's timeouts and the slot's 5 MiB: no check summaries or workflow
    # names, and the viewer's own pending review left out.
    assert "summary" not in query["query"] and "workflow {" not in query["query"]
    assert "states: [APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED]" in query["query"]
    assert query["variables"]["owner"] == "acme" and query["variables"]["name"] == "widgets"


def test_one_forbidden_check_run_leaves_only_that_pulls_checks_unknown() -> None:
    hidden = _node(7, contexts=[_context_run("test", "SUCCESS"), None])
    answer = {
        "data": _page([hidden, _node(9)]),
        "errors": [
            {
                "type": "FORBIDDEN",
                "message": FORBIDDEN_CHECK,
                "path": [
                    "repository",
                    "pullRequests",
                    "nodes",
                    0,
                    "commits",
                    "nodes",
                    0,
                    "commit",
                    "statusCheckRollup",
                    "contexts",
                    "nodes",
                    1,
                ],
            }
        ],
    }
    first, second = _list(_graphql([answer])).pulls

    assert first.checks.state is CheckState.UNKNOWN
    assert [gap.fact for gap in first.gaps] == ["checks"]
    assert "FORBIDDEN" in first.gaps[0].reason
    assert second.checks.state is CheckState.GREEN and second.gaps == ()


def _forbidden(*path: str | int) -> dict[str, Any]:
    nodes = ["repository", "pullRequests", "nodes", 0, "commits", "nodes", 0, "commit"]
    return {"type": "FORBIDDEN", "message": FORBIDDEN_CHECK, "path": [*nodes, *path]}


def test_without_actions_read_the_checks_stand_and_their_runs_are_a_gap() -> None:
    green = _context_run("test", "SUCCESS")
    green["checkSuite"]["workflowRun"] = None
    path = ("statusCheckRollup", "contexts", "nodes", 0, "checkSuite", "workflowRun")
    answer = {"data": _page([_node(7, contexts=[green])]), "errors": [_forbidden(*path)]}
    (pull,) = _list(_graphql([answer])).pulls

    assert pull.checks.complete and pull.checks.state is CheckState.GREEN
    assert [gap.fact for gap in pull.gaps] == ["workflow_runs"]
    assert "actions: read" in pull.gaps[0].reason
    # The check's page still names its run.
    assert pull.checks.ci[0].workflow_run == github.CheckWorkflowRun(901)


def test_a_forbidden_rollup_leaves_the_checks_unknown() -> None:
    node = _node(7)
    node["commits"]["nodes"][0]["commit"]["statusCheckRollup"] = None
    answer = {"data": _page([node]), "errors": [_forbidden("statusCheckRollup")]}
    (pull,) = _list(_graphql([answer])).pulls

    assert pull.checks.state is CheckState.UNKNOWN and [gap.fact for gap in pull.gaps] == ["checks"]


def test_any_other_graphql_error_fails_the_listing() -> None:
    answer = {
        "data": _page([_node(7)]),
        "errors": [{"type": "INTERNAL", "message": "Something went wrong", "path": ["repository"]}],
    }
    with pytest.raises(github.GitHubGraphQLFailed):
        _list(_graphql([answer]))


def test_status_contexts_count_as_ci_and_truncated_checks_as_unknown() -> None:
    contexts = [
        {
            "__typename": "StatusContext",
            "context": "ci/legacy",
            "state": "PENDING",
            "targetUrl": None,
        },
        {"__typename": "StatusContext", "context": "ci/old", "state": "ERROR", "targetUrl": None},
    ]
    node = _node(7, contexts=contexts)
    pull = _list(_graphql([{"data": _page([node])}])).pulls[0]
    assert pull.checks.state is CheckState.FAILED and pull.checks.failed_names == ("ci/old",)

    node = _node(7, contexts=[_context_run("test", "SUCCESS")])
    node["commits"]["nodes"][0]["commit"]["statusCheckRollup"]["contexts"]["totalCount"] = 140
    pull = _list(_graphql([{"data": _page([node])}])).pulls[0]
    assert pull.checks.state is CheckState.UNKNOWN
    assert pull.gaps[0].reason == "GitHub has 140 checks on the head; 1 were read."


def test_a_pull_with_no_ci_reads_none_and_a_deleted_author_ghost() -> None:
    node = _node(7, author=None)
    node["commits"]["nodes"][0]["commit"]["statusCheckRollup"] = None
    pull = _list(_graphql([{"data": _page([node])}])).pulls[0]

    assert pull.checks.state is CheckState.NONE and pull.author == "ghost"


def test_updated_since_stops_at_the_first_page_that_reaches_past_it() -> None:
    pages = [
        {"data": _page([_node(5, _at(15)), _node(4, _at(14))], more=True, cursor="c1")},
        {"data": _page([_node(3, _at(13)), _node(1, _at(11))], more=True, cursor="c2")},
    ]
    broker = _graphql(pages)
    listing = _list(broker, states=None, updated_since=_when(12))

    assert [pull.number for pull in listing.pulls] == [5, 4, 3] and listing.complete
    first, second = _queries(broker)
    assert first["variables"]["states"] is None and first["variables"]["cursor"] is None
    assert second["variables"]["cursor"] == "c1"


def test_max_pages_stops_a_listing_and_says_so() -> None:
    broker = _graphql([{"data": _page([_node(5)], more=True)}])
    listing = _list(broker, max_pages=1)

    assert [pull.number for pull in listing.pulls] == [5] and not listing.complete


def test_unknown_mergeability_in_a_listing_is_read_again_together(waits: list[float]) -> None:
    unknown = {"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"}
    merged = _node(8, state="MERGED", mergedBy=_actor("ana"), mergedAt=_at(12), **unknown)
    pages = [{"data": _page([_node(7, **unknown), merged, _node(9, **unknown)])}]
    broker = _graphql(
        pages,
        mergeable={
            "p7": {"mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY"},
            "p9": {"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"},
        },
    )
    seven, eight, nine = _list(broker).pulls

    assert (seven.mergeable, seven.mergeable_state) == (False, "dirty")
    assert eight.state is github.PullState.MERGED and eight.merged_by == "ana"
    assert eight.mergeable is None and nine.mergeable is None
    assert waits == [github.MERGEABLE_WAIT_SECONDS]
    reread = _queries(broker)[1]["query"]
    assert "p7: pullRequest(number: 7)" in reread and "p9: pullRequest(number: 9)" in reread
    assert "number: 8" not in reread


def test_the_mergeable_reread_asks_for_at_most_fifty_pulls_a_query(waits: list[float]) -> None:
    unknown = {"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"}
    nodes = [_node(number, **unknown) for number in range(1, 52)]
    broker = _graphql([{"data": _page(nodes)}])
    listing = _list(broker)

    assert len(listing.pulls) == 51 and all(pull.mergeable is None for pull in listing.pulls)
    rereads = [query["query"] for query in _queries(broker)[1:]]
    assert [reread.count("pullRequest(number:") for reread in rereads] == [50, 1]
    assert waits == [github.MERGEABLE_WAIT_SECONDS]


def test_a_pull_gone_before_the_mergeable_reread_stays_unknown(waits: list[float]) -> None:
    unknown = {"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"}

    def connector(sent: Sent) -> Answer:
        if "query Mergeable" in sent.json()["query"]:
            gone = _graphql_error("NOT_FOUND", "repository", "p7")
            return Answer(body={"data": {"repository": {"p7": None}}, "errors": [gone]})
        return Answer(body={"data": _page([_node(7, **unknown)])})

    (pull,) = _list(FakeBroker(connector)).pulls
    assert pull.mergeable is None


def test_a_github_failure_on_the_mergeable_reread_fails_the_listing(waits: list[float]) -> None:
    unknown = {"mergeable": "UNKNOWN", "mergeStateStatus": "UNKNOWN"}

    def connector(sent: Sent) -> Answer:
        if "query Mergeable" in sent.json()["query"]:
            failed = _graphql_error("INTERNAL", "repository", "p7", "mergeable")
            return Answer(body={"data": {"repository": {"p7": None}}, "errors": [failed]})
        return Answer(body={"data": _page([_node(7, **unknown)])})

    with pytest.raises(github.GitHubGraphQLFailed):
        _list(FakeBroker(connector))


def test_list_pulls_refuses_a_naive_since_or_no_pages() -> None:
    broker = _graphql([])
    with pytest.raises(ValueError, match="timezone-aware"):
        _list(broker, updated_since=datetime(2026, 11, 12))
    with pytest.raises(ValueError, match="max_pages"):
        _list(broker, max_pages=0)
    assert broker.sent == []
