"""One pull request's facts as a typed record: its state, head, mergeability, CI checks with
their workflow run attempts, the reviewer App's check kept apart, reviews and comments."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Collection, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any, TypeVar

from cayu.connectors.github._errors import (
    GitHubConnectionRefused,
    GitHubError,
    GitHubNotFound,
    GitHubPermissionDenied,
)
from cayu.connectors.github._graphql import (
    ErrorFilter,
    GraphQLError,
    errors_under,
    forbidden_under,
)
from cayu.connectors.github.client import GitHubClient, Page

#: How long ``read_pull`` and ``list_pulls`` wait before reading ``mergeable`` again when GitHub
#: hasn't worked it out yet: it starts on the first read, and is usually ready a moment later.
MERGEABLE_WAIT_SECONDS = 2.0
#: A check's conclusions that pass; any other conclusion of a finished check fails.
PASSING_CONCLUSIONS = frozenset({"success", "neutral", "skipped"})
#: What a deleted account's login reads as, in REST's words.
_GHOST_LOGIN = "ghost"
#: Pull requests a ``list_pulls`` page asks GraphQL for (GitHub times out heavier queries), and
#: the most pages it reads.
_LIST_PAGE_SIZE = 20
_DEFAULT_LIST_PAGES = 10
#: The latest reviews ``list_pulls`` reads of each pull request.
_LIST_REVIEWS = 40
#: Check runs a ``read_pull`` page asks for: each can carry 128 KiB of output, under the
#: slot's 5 MiB answer.
_CHECK_RUNS_PAGE = "30"
#: Pull requests the mergeable re-read asks GraphQL for at once.
_MERGEABLE_BATCH = 50
#: The slot's refusal of an answer over its size limit: that one fact is left out.
_TOO_LARGE = "upstream_response_too_large"
#: A GitHub Actions check run's page: its workflow run and job.
_ACTIONS_JOB_URL = re.compile(r"/actions/runs/([0-9]+)/job/[0-9]+")
_FINISHED = "completed"
_STATUS_FINISHED = frozenset({"success", "failure", "error"})
_GRAPHQL_MERGEABLE: dict[object, bool] = {"MERGEABLE": True, "CONFLICTING": False}
_NODES_PATH = ("repository", "pullRequests", "nodes")
#: The GraphQL errors that mean a fact is missing or this credential can't read it; any other
#: error (a rate limit, a timeout, GitHub's own failure) fails the read.
_MISSING_OR_FORBIDDEN = frozenset({"NOT_FOUND", "FORBIDDEN"})
#: The review states that stand until the reviewer submits another of them; a comment-only
#: review leaves an approval or a request for changes standing, as in GitHub's own decision.
_STANDING_REVIEW_STATES = frozenset({"APPROVED", "CHANGES_REQUESTED", "DISMISSED"})
_COMPARE_QUERY = """
query PullRequestCompare($owner: String!, $name: String!, $base: String!, $head: String!) {
  repository(owner: $owner, name: $name) {
    ref(qualifiedName: $base) { compare(headRef: $head) { aheadBy behindBy } }
  }
}
"""
_PULLS_QUERY = """
query PullRequestFacts(
  $owner: String!, $name: String!, $first: Int!, $reviews: Int!, $cursor: String,
  $states: [PullRequestState!]
) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      first: $first, after: $cursor, states: $states, orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      pageInfo { hasNextPage endCursor }
      nodes { ...PullRequestFacts }
    }
  }
}
fragment Actor on Actor { __typename login }
fragment PullRequestFacts on PullRequest {
  number title url state isDraft createdAt updatedAt mergedAt mergeable mergeStateStatus
  baseRefName headRefName headRefOid
  author { ...Actor }
  mergedBy { ...Actor }
  labels(first: 100) { nodes { name } }
  assignees(first: 50) { nodes { __typename login } }
  reviewRequests(first: 50) {
    nodes {
      requestedReviewer {
        __typename
        ... on User { login }
        ... on Bot { login }
        ... on Mannequin { login }
        ... on Team { slug }
      }
    }
  }
  reviews(last: $reviews, states: [APPROVED, CHANGES_REQUESTED, COMMENTED, DISMISSED]) {
    pageInfo { hasPreviousPage }
    nodes { databaseId state submittedAt url author { ...Actor } commit { oid } }
  }
  commits(last: 1) {
    nodes {
      commit {
        oid committedDate
        statusCheckRollup {
          contexts(first: 100) {
            totalCount
            pageInfo { hasNextPage }
            nodes {
              __typename
              ... on CheckRun {
                databaseId name status conclusion title detailsUrl startedAt completedAt
                checkSuite { app { slug } workflowRun { databaseId url } }
              }
              ... on StatusContext { context state targetUrl createdAt }
            }
          }
        }
      }
    }
  }
}
"""
_sleep = asyncio.sleep
_Value = TypeVar("_Value")


class PullState(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"


class CheckState(StrEnum):
    """One word for a head commit's CI, over its CI checks (check runs and commit statuses; the
    reviewer App's checks are apart), decided in this order:

    - ``failed``: a finished check concluded other than ``success``, ``neutral`` or ``skipped``
      (``cancelled``, ``timed_out``, a status's ``failure`` or ``error``, and also
      ``action_required``, which waits on a person, and ``stale``, a check GitHub gave up on).
      A failure stands even when some checks weren't read. Each check keeps its
      ``conclusion``, so a caller can tell one waiting on a person from one that broke before
      rerunning it.
    - ``unknown``: some checks weren't read (a check this credential can't see, or more than
      were listed), so one left out might fail.
    - ``none``: no CI check reported on the head. Whether that is green or a wait is the
      caller's rule.
    - ``running``: a check hasn't finished.
    - ``green``: every check finished and passed.
    """

    GREEN = "green"
    FAILED = "failed"
    RUNNING = "running"
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CheckWorkflowRun:
    """The GitHub Actions workflow run a check belongs to. ``attempt`` is the run attempt the
    check ran in (1, 2, ... after each rerun); None where it isn't known: in ``list_pulls``, and
    for a job an earlier attempt ran that a rerun of the failed jobs kept. ``name`` and ``url``
    are empty where they weren't read."""

    id: int
    attempt: int | None = None
    name: str = ""
    url: str = ""


@dataclass(frozen=True)
class PullCheck:
    """A check run on the head commit, or a commit status (``app`` None).

    ``status`` is ``completed`` once it finished (else ``queued``, ``in_progress``,
    ``pending``, ...); ``conclusion`` is set only then. A GitHub Actions check run's ``id`` is
    its job's id. ``summary`` is read by ``read_pull`` only.
    """

    name: str
    status: str
    conclusion: str | None = None
    app: str | None = None
    id: int | None = None
    workflow_run: CheckWorkflowRun | None = None
    title: str | None = None
    summary: str | None = None
    url: str = ""
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def finished(self) -> bool:
        return self.status == _FINISHED

    @property
    def failed(self) -> bool:
        return self.finished and self.conclusion not in PASSING_CONCLUSIONS


@dataclass(frozen=True)
class PullChecks:
    """The head commit's checks: ``ci``, and ``reviewer``, the check runs of the reviewer App
    named to the read, told apart by the App that made them and not by name. ``complete`` is
    false when some checks weren't read (see ``PullRequestFacts.gaps``)."""

    ci: tuple[PullCheck, ...] = ()
    reviewer: tuple[PullCheck, ...] = ()
    complete: bool = True

    @property
    def failed(self) -> tuple[PullCheck, ...]:
        return tuple(check for check in self.ci if check.failed)

    @property
    def running(self) -> tuple[PullCheck, ...]:
        return tuple(check for check in self.ci if not check.finished)

    @property
    def failed_names(self) -> tuple[str, ...]:
        return tuple(sorted({check.name for check in self.failed}))

    @property
    def state(self) -> CheckState:
        if self.failed:
            return CheckState.FAILED
        if not self.complete:
            return CheckState.UNKNOWN
        if not self.ci:
            return CheckState.NONE
        if self.running:
            return CheckState.RUNNING
        return CheckState.GREEN


@dataclass(frozen=True)
class PullReview:
    """A submitted review; ``state`` is GitHub's (``APPROVED``, ``CHANGES_REQUESTED``,
    ``COMMENTED``, ``DISMISSED``). ``body`` is None where it wasn't read (``list_pulls``)."""

    id: int | None
    author: str
    state: str
    body: str | None
    commit_id: str | None
    submitted_at: datetime | None
    url: str = ""


@dataclass(frozen=True)
class PullComment:
    """A comment on the pull request's conversation."""

    id: int | None
    author: str
    body: str
    created_at: datetime | None
    updated_at: datetime | None
    url: str = ""


@dataclass(frozen=True)
class FactGap:
    """A fact read in part or not at all, and why: ``fact`` is ``checks``, ``workflow_runs``,
    ``behind_by``, ``head_committed_at``, ``mergeable``, ``reviews`` or ``comments``."""

    fact: str
    reason: str


@dataclass(frozen=True)
class PullRequestFacts:
    """What GitHub says about one pull request, read once.

    Logins are as GitHub's REST API writes them: a GitHub App's ends in ``[bot]``, a deleted
    account is ``ghost``. ``mergeable`` is None while GitHub hasn't worked it out, even after
    reading it again; a ``dirty`` ``mergeable_state`` is False. ``behind_by``, ``ahead_by`` and
    ``comments`` are None where they weren't read; ``gaps`` says what was read in part and why.
    ``reviews`` and ``comments`` are oldest first.
    """

    repository: str
    number: int
    title: str
    url: str
    author: str
    state: PullState
    draft: bool
    labels: tuple[str, ...]
    base: str
    head_ref: str
    head_sha: str
    head_committed_at: datetime | None
    created_at: datetime | None
    updated_at: datetime | None
    merged_at: datetime | None
    merged_by: str | None
    assignees: tuple[str, ...]
    requested_reviewers: tuple[str, ...]
    requested_teams: tuple[str, ...]
    mergeable: bool | None
    mergeable_state: str
    behind_by: int | None
    ahead_by: int | None
    checks: PullChecks
    reviews: tuple[PullReview, ...]
    comments: tuple[PullComment, ...] | None
    gaps: tuple[FactGap, ...] = ()

    @property
    def latest_reviews(self) -> tuple[PullReview, ...]:
        """Each reviewer's standing review, as GitHub decides it: their latest that approves,
        requests changes or was dismissed, in submission order. A
        later comment-only review leaves it standing; ``reviews`` has every review."""

        latest = {
            review.author: review
            for review in self.reviews
            if review.state in _STANDING_REVIEW_STATES
        }
        return tuple(sorted(latest.values(), key=self.reviews.index))


@dataclass(frozen=True)
class PullRequestList:
    """A listing's pull requests; ``complete`` is false when ``max_pages`` stopped it."""

    pulls: tuple[PullRequestFacts, ...]
    complete: bool


async def read_pull(
    gh: GitHubClient,
    repository: str,
    number: int,
    *,
    reviewer_app: str | None = None,
    mergeable_wait_seconds: float = MERGEABLE_WAIT_SECONDS,
    comments: bool = True,
) -> PullRequestFacts:
    """Every fact about pull request ``number`` of ``repository``, through GitHub's REST API.

    An open pull request whose ``mergeable`` GitHub hasn't worked out is read again once after
    ``mergeable_wait_seconds`` (0: never). ``reviewer_app`` is the slug of the GitHub App whose
    check runs are the reviewer's (``PullChecks.reviewer``). The head's commit time, its compare
    with the base (GraphQL's, which carries no files), its check runs, workflow runs and
    statuses are each a fact of their own: one this credential may not read (403) or can't find
    (404) is left out with a ``FactGap``, and the rest stands. So is any fact, reviews and
    comments included, whose answer is over the slot's size limit. Only reads (GET and GraphQL
    queries), so a read-only client reads it all.
    """

    if number < 1:
        raise ValueError(f"A pull request number is positive: {number}.")
    if mergeable_wait_seconds < 0:
        raise ValueError("mergeable_wait_seconds can't be negative.")
    root = f"/repos/{repository}"
    pull = await gh.get(f"{root}/pulls/{number}")
    if pull.get("state") == "open" and pull.get("mergeable") is None and mergeable_wait_seconds:
        await _sleep(mergeable_wait_seconds)
        pull = await gh.get(f"{root}/pulls/{number}")
    head_sha = pull["head"]["sha"]
    base = pull["base"]["ref"]
    commit = f"{root}/commits/{head_sha}"
    compared, committed, check_runs, runs, statuses, reviews, conversation = await _all(
        _fact("behind_by", _compare(gh, repository, base, head_sha)),
        _fact("head_committed_at", gh.get(f"{root}/git/commits/{head_sha}")),
        _fact(
            "checks",
            gh.list(
                f"{commit}/check-runs", query={"per_page": _CHECK_RUNS_PAGE}, item_key="check_runs"
            ),
        ),
        _fact(
            "workflow_runs",
            gh.list(f"{root}/actions/runs", query={"head_sha": head_sha}, item_key="workflow_runs"),
        ),
        _fact("checks", gh.list(f"{commit}/status", item_key="statuses")),
        _fact("reviews", gh.list(f"{root}/pulls/{number}/reviews"), missing=False),
        _fact("comments", gh.list(f"{root}/issues/{number}/comments"), missing=False)
        if comments
        else _nothing(),
    )
    gaps: list[FactGap] = []
    for fact in (compared, committed, check_runs, runs, statuses, reviews, conversation):
        if isinstance(fact, FactGap):
            gaps.append(fact)
    for page, fact, what in (
        (check_runs, "checks", "check runs"),
        (runs, "workflow_runs", "workflow runs"),
        (statuses, "checks", "commit statuses"),
        (reviews, "reviews", "reviews"),
        (conversation, "comments", "comments"),
    ):
        if isinstance(page, Page) and not page.complete:
            gaps.append(FactGap(fact, f"GitHub has more {what} than were read."))
    suites = {run.get("check_suite_id"): run for run in _items(runs) if run.get("check_suite_id")}
    by_id = {run.get("id"): run for run in _items(runs)}
    ci, reviewer = _split(
        [_rest_check_run(item, suites, by_id) for item in _items(check_runs)], reviewer_app
    )
    ci.extend(_status(item) for item in _items(statuses))
    checks_complete = not any(gap.fact == "checks" for gap in gaps)
    numbers = compared if isinstance(compared, Mapping) else {}
    return PullRequestFacts(
        repository=repository,
        number=int(pull["number"]),
        title=pull.get("title") or "",
        url=pull.get("html_url") or "",
        author=_rest_login(pull.get("user")),
        state=PullState.MERGED
        if pull.get("merged_at")
        else PullState(pull.get("state") or PullState.OPEN),
        draft=bool(pull.get("draft")),
        labels=tuple(label.get("name") or "" for label in pull.get("labels") or ()),
        base=base,
        head_ref=pull["head"].get("ref") or "",
        head_sha=head_sha,
        head_committed_at=_time((committed.get("committer") or {}).get("date"))
        if isinstance(committed, Mapping)
        else None,
        created_at=_time(pull.get("created_at")),
        updated_at=_time(pull.get("updated_at")),
        merged_at=_time(pull.get("merged_at")),
        merged_by=_rest_login(pull["merged_by"]) if pull.get("merged_by") else None,
        assignees=tuple(_rest_login(user) for user in pull.get("assignees") or ()),
        requested_reviewers=tuple(
            _rest_login(user) for user in pull.get("requested_reviewers") or ()
        ),
        requested_teams=tuple(team.get("slug") or "" for team in pull.get("requested_teams") or ()),
        mergeable=_mergeable(pull.get("mergeable"), pull.get("mergeable_state")),
        mergeable_state=(pull.get("mergeable_state") or "unknown").lower(),
        behind_by=_count(numbers.get("behindBy")),
        ahead_by=_count(numbers.get("aheadBy")),
        checks=PullChecks(tuple(ci), tuple(reviewer), checks_complete),
        reviews=tuple(
            sorted(
                (_rest_review(item) for item in _items(reviews) if item.get("state") != "PENDING"),
                key=_submitted,
            )
        ),
        comments=None
        if conversation is None or isinstance(conversation, FactGap)
        else tuple(_rest_comment(item) for item in _items(conversation)),
        gaps=tuple(gaps),
    )


async def list_pulls(
    gh: GitHubClient,
    repository: str,
    *,
    states: Collection[PullState] | None = (PullState.OPEN,),
    updated_since: datetime | None = None,
    reviewer_app: str | None = None,
    max_pages: int = _DEFAULT_LIST_PAGES,
    mergeable_wait_seconds: float = MERGEABLE_WAIT_SECONDS,
) -> PullRequestList:
    """A repository's pull requests in ``states`` (None: all), most recently updated first,
    through one GraphQL query a page of 20 (at most ``max_pages`` pages).

    ``updated_since`` (timezone-aware) keeps only those updated after it and stops at the first
    page that reaches past it. Open pull requests whose ``mergeable`` GitHub hasn't worked out
    are read again together, once, after ``mergeable_wait_seconds`` (0: never). A check run this
    credential can't read (``FORBIDDEN``) leaves that pull request's checks incomplete, with a
    ``FactGap``; one whose workflow run it can't read (no ``actions: read``) leaves only a
    ``workflow_runs`` gap. Any other error fails the listing. Not read here: ``behind_by``,
    ``ahead_by``, ``comments``, review bodies, check summaries, workflow names and run attempts
    (``read_pull`` reads them).
    """

    if max_pages < 1:
        raise ValueError("max_pages is at least 1.")
    if mergeable_wait_seconds < 0:
        raise ValueError("mergeable_wait_seconds can't be negative.")
    if updated_since is not None and updated_since.tzinfo is None:
        raise ValueError("updated_since is a timezone-aware datetime.")
    owner, _, name = repository.partition("/")
    variables: dict[str, object] = {
        "owner": owner,
        "name": name,
        "first": _LIST_PAGE_SIZE,
        "reviews": _LIST_REVIEWS,
        "states": None if states is None else [PullState(state).name for state in states],
        "cursor": None,
    }
    pulls: list[PullRequestFacts] = []
    for _ in range(max_pages):
        answer = await gh.graphql(
            _PULLS_QUERY,
            variables,
            repository=repository,
            accept=forbidden_under("statusCheckRollup"),
        )
        listing = answer.data["repository"]["pullRequests"]
        forbidden = _forbidden_by_node(answer.accepted)
        page = [
            _graphql_pull(repository, node, reviewer_app, forbidden.get(index, (0, 0)))
            for index, node in enumerate(listing.get("nodes") or ())
            if node
        ]
        recent = [pull for pull in page if _updated_after(pull, updated_since)]
        pulls.extend(recent)
        info = listing.get("pageInfo") or {}
        if not info.get("hasNextPage") or len(recent) < len(page):
            break
        variables["cursor"] = info.get("endCursor")
    else:
        return PullRequestList(await _settled(gh, repository, pulls, mergeable_wait_seconds), False)
    return PullRequestList(await _settled(gh, repository, pulls, mergeable_wait_seconds), True)


def _mergeable(value: object, state: object) -> bool | None:
    """GitHub's ``mergeable``, False when its merge state is ``dirty`` (a conflict)."""

    if isinstance(state, str) and state.lower() == "dirty":
        return False
    return value if isinstance(value, bool) else None


def _split(
    checks: list[PullCheck], reviewer_app: str | None
) -> tuple[list[PullCheck], list[PullCheck]]:
    """CI checks, and the check runs ``reviewer_app`` made."""

    ci = [check for check in checks if reviewer_app is None or check.app != reviewer_app]
    reviewer = [check for check in checks if reviewer_app is not None and check.app == reviewer_app]
    return ci, reviewer


async def _fact(name: str, read: Awaitable[_Value], *, missing: bool = True) -> _Value | FactGap:
    """A fact's read, or the gap it leaves when its answer is over the slot's size limit or,
    when ``missing``, when this credential can't read or find it."""

    try:
        return await read
    except GitHubError as error:
        too_large = isinstance(error, GitHubConnectionRefused) and error.code == _TOO_LARGE
        if too_large or (missing and isinstance(error, GitHubPermissionDenied | GitHubNotFound)):
            return FactGap(name, error.message)
        raise


async def _compare(
    gh: GitHubClient, repository: str, base: str, head_sha: str
) -> Mapping[str, Any] | FactGap:
    """GraphQL's compare of the base branch with ``head_sha``, the head commit the record
    describes: its ``aheadBy`` and ``behindBy``, without files. The compare names the commit
    itself, which a fork's pull request makes readable in the base repository too, so a head
    that moves during the read leaves these counts about the head the record names; GraphQL
    can't resolve the ``refs/pull/N/head`` ref to a commit (it answers ``null``)."""

    owner, _, name = repository.partition("/")
    answer = await gh.graphql(
        _COMPARE_QUERY,
        {"owner": owner, "name": name, "base": f"refs/heads/{base}", "head": head_sha},
        repository=repository,
        accept=_missing_or_forbidden_under(("repository", "ref")),
    )
    compared = (answer.data["repository"].get("ref") or {}).get("compare")
    if answer.accepted or not isinstance(compared, dict):
        said = "; ".join(error.message for error in answer.accepted)
        return FactGap("behind_by", said or f"GitHub has no compare of {base} with the head.")
    return compared


def _missing_or_forbidden_under(*prefixes: tuple[str, ...]) -> ErrorFilter:
    """Accept a ``NOT_FOUND`` or ``FORBIDDEN`` error under one of ``prefixes``, a fact that is
    missing or this credential can't read; nothing else."""

    accepted = [errors_under(*prefix, types=_MISSING_OR_FORBIDDEN) for prefix in prefixes]
    return lambda error: any(accept(error) for accept in accepted)


async def _nothing() -> None:
    return None


async def _all(*reads: Awaitable[Any]) -> list[Any]:
    """Every read's answer; the first failure raises once all have finished."""

    answers = await asyncio.gather(*reads, return_exceptions=True)
    for answer in answers:
        if isinstance(answer, BaseException):
            raise answer
    return answers


def _items(found: object) -> tuple[Any, ...]:
    return found.items if isinstance(found, Page) else ()


def _rest_check_run(
    item: Mapping[str, Any],
    suites: Mapping[Any, Mapping[str, Any]],
    runs: Mapping[Any, Mapping[str, Any]],
) -> PullCheck:
    output = item.get("output") or {}
    run = suites.get((item.get("check_suite") or {}).get("id"))
    return PullCheck(
        name=item.get("name") or "",
        status=item.get("status") or "",
        conclusion=item.get("conclusion"),
        app=(item.get("app") or {}).get("slug"),
        id=item.get("id"),
        workflow_run=_rest_run(item, run, runs),
        title=output.get("title"),
        summary=output.get("summary"),
        url=item.get("html_url") or item.get("details_url") or "",
        started_at=_time(item.get("started_at")),
        completed_at=_time(item.get("completed_at")),
    )


def _rest_run(
    item: Mapping[str, Any], run: Mapping[str, Any] | None, runs: Mapping[Any, Mapping[str, Any]]
) -> CheckWorkflowRun | None:
    """A check run's workflow run: by its check suite, which is one attempt's, so the attempt is
    the check's; else by the run its page names, a job a rerun kept from an earlier attempt,
    whose attempt isn't known."""

    if run is not None:
        return CheckWorkflowRun(
            int(run["id"]), run.get("run_attempt"), run.get("name") or "", run.get("html_url") or ""
        )
    run_id = _actions_run_id(item.get("details_url"))
    if run_id is None:
        return None
    listed = runs.get(run_id) or {}
    return CheckWorkflowRun(run_id, None, listed.get("name") or "", listed.get("html_url") or "")


def _actions_run_id(url: object) -> int | None:
    found = _ACTIONS_JOB_URL.search(url) if isinstance(url, str) else None
    return int(found.group(1)) if found else None


def _status_check(context: str, state: str, url: str | None, at: object) -> PullCheck:
    """A commit status as a check: finished once it is ``success``, ``failure`` or ``error``."""

    state = state.lower()
    finished = state in _STATUS_FINISHED
    return PullCheck(
        name=context,
        status=_FINISHED if finished else "pending",
        conclusion=state if finished else None,
        url=url or "",
        started_at=_time(at),
    )


def _status(item: Mapping[str, Any]) -> PullCheck:
    return _status_check(
        item.get("context") or "",
        item.get("state") or "",
        item.get("target_url"),
        item.get("created_at"),
    )


def _rest_review(item: Mapping[str, Any]) -> PullReview:
    return PullReview(
        id=item.get("id"),
        author=_rest_login(item.get("user")),
        state=item.get("state") or "",
        body=item.get("body") or "",
        commit_id=item.get("commit_id"),
        submitted_at=_time(item.get("submitted_at")),
        url=item.get("html_url") or "",
    )


def _rest_comment(item: Mapping[str, Any]) -> PullComment:
    return PullComment(
        id=item.get("id"),
        author=_rest_login(item.get("user")),
        body=item.get("body") or "",
        created_at=_time(item.get("created_at")),
        updated_at=_time(item.get("updated_at")),
        url=item.get("html_url") or "",
    )


def _rest_login(user: Any) -> str:
    return (user.get("login") if isinstance(user, dict) else None) or _GHOST_LOGIN


def _graphql_login(actor: Any) -> str:
    """An actor's login as REST writes it: GraphQL names a GitHub App's bot without ``[bot]``."""

    if not isinstance(actor, dict) or not actor.get("login"):
        return _GHOST_LOGIN
    login = str(actor["login"])
    if actor.get("__typename") == "Bot" and not login.endswith("[bot]"):
        return f"{login}[bot]"
    return login


def _submitted(review: PullReview) -> float:
    return review.submitted_at.timestamp() if review.submitted_at else float("inf")


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


async def _settled(
    gh: GitHubClient, repository: str, pulls: list[PullRequestFacts], wait: float
) -> tuple[PullRequestFacts, ...]:
    """The pull requests, those open whose ``mergeable`` was unknown read again once, up to
    ``_MERGEABLE_BATCH`` in a query. The answer is taken only for the head the record
    describes: a pull request whose head moved since keeps ``mergeable`` unknown, with a gap."""

    unsettled = [p.number for p in pulls if p.state is PullState.OPEN and p.mergeable is None]
    if not unsettled or not wait:
        return tuple(pulls)
    await _sleep(wait)
    owner, _, name = repository.partition("/")
    found: dict[str, Any] = {}
    for start in range(0, len(unsettled), _MERGEABLE_BATCH):
        fields = " ".join(
            f"p{number}: pullRequest(number: {number}) {{ headRefOid mergeable mergeStateStatus }}"
            for number in unsettled[start : start + _MERGEABLE_BATCH]
        )
        answer = await gh.graphql(
            "query Mergeable($owner: String!, $name: String!) {"
            f" repository(owner: $owner, name: $name) {{ {fields} }} }}",
            {"owner": owner, "name": name},
            repository=repository,
            accept=_under_alias,
        )
        found.update(answer.data["repository"])
    settled = []
    for pull in pulls:
        node = found.get(f"p{pull.number}")
        if node and node.get("headRefOid") != pull.head_sha:
            moved = FactGap(
                "mergeable",
                f"The head moved to {node.get('headRefOid') or 'nothing'} while {pull.head_sha} "
                "was read; its mergeability wasn't taken.",
            )
            pull = replace(pull, gaps=(*pull.gaps, moved))
        elif node:
            pull = replace(
                pull,
                mergeable=_mergeable(
                    _GRAPHQL_MERGEABLE.get(node.get("mergeable")), node.get("mergeStateStatus")
                ),
                mergeable_state=(node.get("mergeStateStatus") or "unknown").lower(),
            )
        settled.append(pull)
    return tuple(settled)


def _under_alias(error: GraphQLError) -> bool:
    """A pull request of the mergeable re-read that is gone or unreadable, which leaves it
    unknown; any other error fails the listing."""

    return (
        error.type in _MISSING_OR_FORBIDDEN
        and error.path[:1] == ("repository",)
        and len(error.path) > 1
    )


def _forbidden_by_node(accepted: tuple[GraphQLError, ...]) -> dict[int, tuple[int, int]]:
    """For each pull request of a page, by its index: how many of its checks this credential
    can't see, and how many of its checks' workflow runs (they need ``actions: read``). An
    error on a check's workflow run leaves the check read; any other, on the rollup, a check
    or the App that made it (which tells the reviewer's check apart), hides a check."""

    found: dict[int, tuple[int, int]] = {}
    for error in accepted:
        path = error.path
        index = path[3] if path[:3] == _NODES_PATH and len(path) > 3 else None
        if not isinstance(index, int):
            continue
        hidden, runs = found.get(index, (0, 0))
        if "workflowRun" in path:
            runs += 1
        else:
            hidden += 1
        found[index] = (hidden, runs)
    return found


def _updated_after(pull: PullRequestFacts, since: datetime | None) -> bool:
    return since is None or (pull.updated_at is not None and pull.updated_at > since)


def _graphql_pull(
    repository: str,
    node: Mapping[str, Any],
    reviewer_app: str | None,
    forbidden: tuple[int, int],
) -> PullRequestFacts:
    commits = (node.get("commits") or {}).get("nodes") or [{}]
    commit = (commits[-1] or {}).get("commit") or {}
    contexts = (commit.get("statusCheckRollup") or {}).get("contexts") or {}
    nodes = contexts.get("nodes") or []
    read = [context for context in nodes if context]
    total = contexts.get("totalCount") or 0
    gaps = []
    hidden, runs = forbidden
    unreadable = max(hidden, len(nodes) - len(read))
    if unreadable:
        gaps.append(
            FactGap(
                "checks",
                f"{unreadable} check(s) on the head this credential can't read (FORBIDDEN).",
            )
        )
    if (contexts.get("pageInfo") or {}).get("hasNextPage") or total > len(nodes):
        gaps.append(
            FactGap("checks", f"GitHub has {total} checks on the head; {len(nodes)} were read.")
        )
    if runs:
        gaps.append(
            FactGap(
                "workflow_runs",
                f"{runs} check(s)' workflow run this credential can't read (FORBIDDEN; it needs "
                "actions: read).",
            )
        )
    reviews = node.get("reviews") or {}
    if (reviews.get("pageInfo") or {}).get("hasPreviousPage"):
        gaps.append(FactGap("reviews", f"Only the latest {_LIST_REVIEWS} reviews were read."))
    ci, reviewer = _split([_graphql_check(context) for context in read], reviewer_app)
    requested = [
        (request or {}).get("requestedReviewer") or {}
        for request in (node.get("reviewRequests") or {}).get("nodes") or ()
    ]
    state = (node.get("state") or "OPEN").lower()
    return PullRequestFacts(
        repository=repository,
        number=int(node["number"]),
        title=node.get("title") or "",
        url=node.get("url") or "",
        author=_graphql_login(node.get("author")),
        state=PullState(state),
        draft=bool(node.get("isDraft")),
        labels=tuple(
            label["name"] for label in (node.get("labels") or {}).get("nodes") or () if label
        ),
        base=node.get("baseRefName") or "",
        head_ref=node.get("headRefName") or "",
        head_sha=node.get("headRefOid") or commit.get("oid") or "",
        head_committed_at=_time(commit.get("committedDate")),
        created_at=_time(node.get("createdAt")),
        updated_at=_time(node.get("updatedAt")),
        merged_at=_time(node.get("mergedAt")),
        merged_by=_graphql_login(node["mergedBy"]) if node.get("mergedBy") else None,
        assignees=tuple(
            _graphql_login(user) for user in (node.get("assignees") or {}).get("nodes") or ()
        ),
        requested_reviewers=tuple(
            _graphql_login(reviewer) for reviewer in requested if reviewer.get("login")
        ),
        requested_teams=tuple(reviewer["slug"] for reviewer in requested if reviewer.get("slug")),
        mergeable=_mergeable(
            _GRAPHQL_MERGEABLE.get(node.get("mergeable")), node.get("mergeStateStatus")
        ),
        mergeable_state=(node.get("mergeStateStatus") or "unknown").lower(),
        behind_by=None,
        ahead_by=None,
        checks=PullChecks(tuple(ci), tuple(reviewer), all(gap.fact != "checks" for gap in gaps)),
        reviews=tuple(
            sorted(
                (_graphql_review(review) for review in reviews.get("nodes") or () if review),
                key=_submitted,
            )
        ),
        comments=None,
        gaps=tuple(gaps),
    )


def _graphql_check(context: Mapping[str, Any]) -> PullCheck:
    if context.get("__typename") == "StatusContext":
        return _status_check(
            context.get("context") or "",
            context.get("state") or "",
            context.get("targetUrl"),
            context.get("createdAt"),
        )
    suite = context.get("checkSuite") or {}
    run = suite.get("workflowRun") or {}
    conclusion = context.get("conclusion")
    return PullCheck(
        name=context.get("name") or "",
        status=(context.get("status") or "").lower(),
        conclusion=conclusion.lower() if isinstance(conclusion, str) else None,
        app=(suite.get("app") or {}).get("slug"),
        id=context.get("databaseId"),
        workflow_run=CheckWorkflowRun(id=int(run["databaseId"]), url=run.get("url") or "")
        if run.get("databaseId")
        else _listed_run(context.get("detailsUrl")),
        title=context.get("title"),
        url=context.get("detailsUrl") or "",
        started_at=_time(context.get("startedAt")),
        completed_at=_time(context.get("completedAt")),
    )


def _listed_run(url: object) -> CheckWorkflowRun | None:
    """The workflow run a check's page names, when its own wasn't readable."""

    run_id = _actions_run_id(url)
    return None if run_id is None else CheckWorkflowRun(run_id)


def _graphql_review(review: Mapping[str, Any]) -> PullReview:
    return PullReview(
        id=review.get("databaseId"),
        author=_graphql_login(review.get("author")),
        state=review.get("state") or "",
        body=None,
        commit_id=(review.get("commit") or {}).get("oid"),
        submitted_at=_time(review.get("submittedAt")),
        url=review.get("url") or "",
    )
