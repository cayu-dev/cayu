"""A fake GitHub for an application's tests: a ``ConnectionClient`` answering from data.

``FakeGitHub`` takes the place of Cayu Cloud's broker or ``app_connections``, so a test runs the
real ``GitHubClient`` (paging, typed errors, rate limits, read-only mode) against GitHub-shaped
answers. Repositories, pull requests, issues, comments, reviews, check runs, workflow runs and
releases are plain objects a test sets up and may change; the writes an application makes take
effect as GitHub's do, and every call is recorded in ``requests``.

It answers the REST routes agents use (pulls, merges, branch updates, compares, commits, check
runs, workflow runs and their reruns and logs, issues, comments, labels, releases and
collaborator permissions). Any other route, query filter or body field raises
``NotImplementedError`` rather than being guessed or ignored, and is kept in ``unanswered``, which
a test asserts is empty in case the application caught the error; ``route()`` adds a route. It
has no GraphQL engine: it answers ``read_pull``'s ``PullRequestCompare`` query from the pulls, and
``on_graphql(name, answer)`` answers any other named operation, as the test writes it. Refusal texts and orders that GitHub's documentation doesn't give were checked
against GitHub itself, and say so where they are made.
"""

from __future__ import annotations

import fnmatch
import hashlib
import itertools
import json
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar
from urllib.parse import unquote, urlencode

import httpx

from cayu.connections import ConnectionCallError, ConnectionResponse, Query
from cayu.connections.client import checked_headers
from cayu.connectors.github._graphql import is_mutation
from cayu.connectors.github.client import (
    API_URL,
    DEFAULT_SLOT,
    GRAPHQL_PATH,
    REPOSITORY_HEADER,
    GitHubClient,
)

#: Who the fake's writes are by, unless ``FakeGitHub(login=...)`` names another.
DEFAULT_LOGIN = "cayu-agent[bot]"
#: GitHub's page size when a listing names none, and the most it allows.
DEFAULT_PER_PAGE = 30
MAX_PER_PAGE = 100
#: Where a job's log is redirected: a host ``GitHubClient.text()`` follows.
LOG_URL = "https://pipelines.actions.githubusercontent.com/fake-logs/"
_LOG_PATH = "/fake-logs/"
#: Where the fake's error bodies point, as GitHub's point at the call's documentation.
DOCUMENTATION_URL = "https://docs.github.com/rest"
_Item = TypeVar("_Item", bound="Issue")
_PAGING = frozenset({"per_page", "page"})
_PASSED = frozenset({"success", "neutral", "skipped"})
_OPERATION = re.compile(r"^\s*(?:query|mutation)\s+([A-Za-z_][A-Za-z0-9_]*)")
_COMMENT = re.compile(r"#[^\n\r]*")
#: A ``ref(qualifiedName: $variable) { ... }`` field of a GraphQL document, with its alias and
#: the variable a ``compare(headRef: $variable)`` inside it names.
_REF_FIELD = re.compile(
    r"(?:(\w+)\s*:\s*)?\bref\(\s*qualifiedName:\s*\$(\w+)\s*\)\s*\{"
    r"([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
)
_COMPARE_HEAD = re.compile(r"compare\(\s*headRef:\s*\$(\w+)\s*\)")
_COMPARE_OPERATION = "PullRequestCompare"
_PULL_REF = re.compile(r"refs/pull/(\d+)/head")


def _iso(when: datetime) -> str:
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _user(login: str) -> dict[str, Any]:
    return {"login": login, "type": "Bot" if login.endswith("[bot]") else "User"}


def reply(
    status: int = 200, body: Any = None, headers: Mapping[str, str] | None = None
) -> ConnectionResponse:
    """An answer as GitHub gives it: ``body`` as JSON (bytes and text as they are)."""

    if isinstance(body, bytes):
        content = body
    elif isinstance(body, str):
        content = body.encode()
    else:
        content = b"" if body is None else json.dumps(body).encode()
    sent = {"content-type": "application/json; charset=utf-8", **(headers or {})}
    return ConnectionResponse(status, tuple((k.lower(), v) for k, v in sent.items()), content)


def _error(status: int, message: str, **extra: Any) -> ConnectionResponse:
    """An error body shaped as GitHub's: its ``message``, a ``documentation_url`` and the
    ``status`` as a string."""

    shape = {"documentation_url": DOCUMENTATION_URL, "status": str(status)}
    return reply(status, {"message": message, **extra, **shape})


def _invalid(resource: str, field_name: str, code: str = "missing_field") -> ConnectionResponse:
    errors = [{"resource": resource, "code": code, "field": field_name}]
    return _error(422, "Validation Failed", errors=errors)


@dataclass(frozen=True)
class Call:
    """One request the fake was sent, whatever it answered."""

    method: str
    path: str
    query: Mapping[str, str]
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body) if self.body else None

    @property
    def repository(self) -> str | None:
        """The repository the call named in its ``x-cayu-github-repository`` header."""

        return self.headers.get(REPOSITORY_HEADER)

    @property
    def write(self) -> bool:
        """Whether GitHub would change something: any method but GET and HEAD, except a
        GraphQL query."""

        if self.method in ("GET", "HEAD"):
            return False
        if self.path == GRAPHQL_PATH:
            return is_mutation(str((self.json() or {}).get("query", "")))
        return True


@dataclass
class Comment:
    id: int
    body: str
    author: str
    created_at: datetime
    updated_at: datetime


@dataclass
class Issue:
    """An issue; ``PullRequest`` is one too, as on GitHub, sharing its numbers."""

    number: int
    title: str = ""
    body: str = ""
    author: str = "octocat"
    state: str = "open"
    state_reason: str | None = None
    labels: list[str] = field(default_factory=list)
    comments: list[Comment] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None


@dataclass
class Review:
    id: int
    author: str
    state: str = "COMMENTED"
    body: str = ""
    commit_id: str = ""
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PullRequest(Issue):
    """``mergeable`` is what GitHub answers once it has computed it; the first
    ``mergeable_unknown_reads`` reads of the pull answer null, as GitHub does after a push.
    ``merge_method`` is that of the merge that merged it."""

    head_sha: str = ""
    head_ref: str = ""
    base: str = "main"
    draft: bool = False
    mergeable: bool = True
    mergeable_unknown_reads: int = 0
    behind_by: int = 0
    ahead_by: int = 1
    #: Branch protection requires the branch to be up to date: only then does a pull that is
    #: behind its base answer ``mergeable_state: "behind"`` (else ``"clean"``), and is a merge
    #: of it refused until the branch is updated.
    requires_up_to_date: bool = False
    committed_at: datetime = field(default_factory=lambda: datetime.now(UTC) - timedelta(hours=1))
    merged: bool = False
    merged_by: str | None = None
    merge_commit_sha: str | None = None
    merge_method: str | None = None
    reviews: list[Review] = field(default_factory=list)


@dataclass
class CheckRun:
    """A check run; for GitHub Actions its ``id`` is also the job's, whose log ``set_log``
    gives. A rerun supersedes it with a new one, which ``filter=all`` still lists."""

    id: int
    head_sha: str
    name: str
    status: str = "completed"
    conclusion: str | None = "success"
    app_slug: str = "github-actions"
    check_suite_id: int = 0
    output_title: str | None = None
    output_summary: str | None = None
    superseded: bool = False


@dataclass
class WorkflowRun:
    id: int
    head_sha: str
    name: str = "CI"
    head_branch: str = "main"
    event: str = "pull_request"
    status: str = "completed"
    conclusion: str | None = "success"
    run_attempt: int = 1
    check_suite_id: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class CommitStatus:
    """A commit status, as a CI service outside GitHub Actions posts one on a commit."""

    id: int
    sha: str
    context: str
    state: str = "success"
    description: str = ""
    target_url: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Release:
    id: int
    tag_name: str
    name: str = ""
    draft: bool = False
    prerelease: bool = False
    published_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class Repository:
    full_name: str
    id: int
    default_branch: str = "main"
    private: bool = True
    issues: dict[int, Issue] = field(default_factory=dict)
    check_runs: list[CheckRun] = field(default_factory=list)
    workflow_runs: list[WorkflowRun] = field(default_factory=list)
    releases: list[Release] = field(default_factory=list)
    statuses: list[CommitStatus] = field(default_factory=list)
    logs: dict[int, str] = field(default_factory=dict)
    #: Each collaborator's permission (``admin``, ``write``, ``read``); anyone else's is
    #: ``read`` on a public repository and ``none`` on a private one.
    permissions: dict[str, str] = field(default_factory=dict)

    @property
    def pulls(self) -> list[PullRequest]:
        return [item for item in self.issues.values() if isinstance(item, PullRequest)]


@dataclass
class _Fault:
    method: str
    path: str
    times: int
    answer: Callable[[], ConnectionResponse]


#: Answers a route ``route()`` adds: from the call and its pattern's groups.
Handler = Callable[[Call, tuple[str, ...]], ConnectionResponse]
GraphQLAnswerer = Callable[[Mapping[str, Any], str], Mapping[str, Any]]


class FakeGitHub:
    """GitHub in memory, behind a connection slot: give it where a ``ConnectionClient`` goes,
    or take ``client()``. Writes are by ``login``; listings answer ``page_size`` items a page
    at most, with GitHub's next-page links, so a test of paging needs only a few items.
    ``listing_lag`` leaves an issue or comment out of the issues and comments listings until
    it is that old by ``clock``, as GitHub's listings show a new one seconds late; reading it
    directly finds it at once."""

    def __init__(
        self,
        *,
        login: str = DEFAULT_LOGIN,
        slot: str = DEFAULT_SLOT,
        page_size: int = MAX_PER_PAGE,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        listing_lag: timedelta = timedelta(0),
    ) -> None:
        if page_size < 1:
            raise ValueError("page_size is at least 1.")
        self.login = login
        self.slot = slot
        self.page_size = page_size
        self.clock = clock
        self.listing_lag = listing_lag
        #: Each repository by its lowercased ``owner/name``.
        self.repositories: dict[str, Repository] = {}
        #: Logins that are GitHub users besides the fake's own, collaborators and authors.
        self.users: set[str] = set()
        #: Every call, oldest first, including those a fault answered.
        self.requests: list[Call] = []
        #: Every call the fake couldn't answer (an unknown route, query filter, body field or
        #: GraphQL operation), also raised as ``NotImplementedError``. An application may catch
        #: that, so a test asserts this is empty.
        self.unanswered: list[Call] = []
        #: Serves the job logs a log read is redirected to; ``GitHubClient`` takes it from its
        #: connection client, so a log download never leaves the test.
        self.download_transport = self._download_transport()
        self._ids = itertools.count(1001)
        self._faults: list[_Fault] = []
        self._graphql: dict[str, GraphQLAnswerer] = {}
        self._routes: list[tuple[str, re.Pattern[str], Callable[[_Served], ConnectionResponse]]]
        self._routes = list(_ROUTES)

    def client(self, **settings: Any) -> GitHubClient:
        """A ``GitHubClient`` on this fake; ``GitHubClient(fake)`` is the same."""

        return GitHubClient(self, self.slot, **settings)

    def listed(self, created_at: datetime) -> bool:
        """Whether an item made at ``created_at`` shows in listings yet (``listing_lag``)."""

        return self.clock() - created_at >= self.listing_lag

    def writes(self) -> list[Call]:
        """Every call that would change something on GitHub, oldest first."""

        return [call for call in self.requests if call.write]

    # ---------------------------------------------------------------- state

    def add_repository(self, full_name: str, **settings: Any) -> Repository:
        repository = Repository(full_name, next(self._ids), **settings)
        self.repositories[full_name.lower()] = repository
        return repository

    def repository(self, full_name: str) -> Repository:
        """The repository named ``full_name`` (owner and name are matched ignoring case, as
        GitHub does), added if it isn't there yet."""

        found = self.repositories.get(full_name.lower())
        return found if found is not None else self.add_repository(full_name)

    def add_issue(self, repository: str, number: int | None = None, **settings: Any) -> Issue:
        return self._add(repository, Issue, number, settings)

    def add_pull(self, repository: str, number: int | None = None, **settings: Any) -> PullRequest:
        """A pull request; its head is ``head_sha`` (by default made from the number) on
        ``head_ref``."""

        pull = self._add(repository, PullRequest, number, settings)
        pull.head_sha = (
            pull.head_sha or hashlib.sha1(f"{repository}#{pull.number}".encode()).hexdigest()
        )
        pull.head_ref = pull.head_ref or f"change-{pull.number}"
        return pull

    def add_comment(
        self, repository: str, number: int, body: str, *, author: str = "octocat"
    ) -> Comment:
        now = self.clock()
        comment = Comment(next(self._ids), body, author, now, now)
        self._issue(repository, number).comments.append(comment)
        return comment

    def add_review(self, repository: str, number: int, author: str, **settings: Any) -> Review:
        pull = self._issue(repository, number)
        if not isinstance(pull, PullRequest):
            raise ValueError(f"{repository}#{number} isn't a pull request.")
        settings.setdefault("commit_id", pull.head_sha)
        settings.setdefault("submitted_at", self.clock())
        review = Review(next(self._ids), author, **settings)
        pull.reviews.append(review)
        return review

    def add_workflow_run(self, repository: str, head_sha: str, **settings: Any) -> WorkflowRun:
        settings.setdefault("created_at", self.clock())
        run = WorkflowRun(next(self._ids), head_sha, **settings)
        run.check_suite_id = run.check_suite_id or next(self._ids)
        self.repository(repository).workflow_runs.append(run)
        return run

    def add_check_run(
        self,
        repository: str,
        head_sha: str,
        name: str,
        *,
        run: WorkflowRun | None = None,
        **settings: Any,
    ) -> CheckRun:
        """A check run; one of ``run``'s jobs when given (a rerun of the run reruns it)."""

        if run is not None:
            settings["check_suite_id"] = run.check_suite_id
        check = CheckRun(next(self._ids), head_sha, name, **settings)
        self.repository(repository).check_runs.append(check)
        return check

    def add_status(self, repository: str, sha: str, context: str, **settings: Any) -> CommitStatus:
        """A commit status on ``sha``; a later one of the same ``context`` replaces it in the
        commit's combined status."""

        settings.setdefault("created_at", self.clock())
        status = CommitStatus(next(self._ids), sha, context, **settings)
        self.repository(repository).statuses.append(status)
        return status

    def add_release(self, repository: str, tag_name: str, **settings: Any) -> Release:
        settings.setdefault("published_at", self.clock())
        release = Release(next(self._ids), tag_name, **settings)
        self.repository(repository).releases.append(release)
        return release

    def set_log(self, repository: str, job_id: int, text: str) -> None:
        self.repository(repository).logs[job_id] = text

    def finish_run(self, repository: str, run_id: int, conclusion: str = "success") -> None:
        """Play GitHub's part after a rerun: the run and its unfinished checks complete."""

        found = self.repository(repository)
        run = next(run for run in found.workflow_runs if run.id == run_id)
        run.status, run.conclusion = "completed", conclusion
        for check in found.check_runs:
            if check.check_suite_id == run.check_suite_id and check.status != "completed":
                check.status, check.conclusion = "completed", conclusion

    # ------------------------------------------------------------- behaviour

    def fail(
        self,
        method: str = "*",
        path: str = "*",
        *,
        status: int = 502,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
        times: int = 1,
    ) -> None:
        """The next ``times`` calls matching ``method`` and the glob ``path`` (matched against
        the path without its query) answer ``status`` instead of being served. Of the faults
        set up, the first one set up that matches answers."""

        answer = body if body is not None else {"message": f"Injected {status}"}
        self._fault(method, path, times, lambda: reply(status, answer, headers))

    def rate_limit(
        self,
        method: str = "*",
        path: str = "*",
        *,
        retry_after: int = 0,
        secondary: bool = False,
        times: int = 1,
    ) -> None:
        """The next ``times`` matching calls hit a rate limit: the primary one (403, none left,
        reset in ``retry_after`` seconds) or a secondary one (403 with ``Retry-After``)."""

        def answer() -> ConnectionResponse:
            if secondary:
                message = "You have exceeded a secondary rate limit. Please wait a few minutes."
                return reply(403, {"message": message}, {"retry-after": str(retry_after)})
            # GitHub's reset is a wall-clock time, whatever the fake's clock says.
            reset = int(time.time()) + retry_after
            limit = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset)}
            return reply(403, {"message": "API rate limit exceeded."}, limit)

        self._fault(method, path, times, answer)

    def refuse(
        self,
        code: str,
        method: str = "*",
        path: str = "*",
        *,
        status: int | None = None,
        times: int = 1,
    ) -> None:
        """The next ``times`` matching calls fail at the slot's host with ``code``:
        ``upstream_failed`` when GitHub gave no answer (a timeout), or a refusal that never
        reached GitHub (``scope_not_allowed``, ``too_many_calls``, ...)."""

        def answer() -> ConnectionResponse:
            raise ConnectionCallError(
                f"Refused ({code}).", code=code, status=status, retry_after=0.0
            )

        self._fault(method, path, times, answer)

    def route(self, method: str, pattern: str, handler: Handler) -> None:
        """Answer a repository's ``method`` ``/repos/{owner}/{name}/<pattern>`` (a regular
        expression) with ``handler(call, groups)``, ahead of the built-in routes."""

        def answer(served: _Served) -> ConnectionResponse:
            return handler(served.call, served.groups)

        self._routes.insert(0, (method.upper(), re.compile(pattern), answer))

    def on_graphql(self, operation: str, answer: GraphQLAnswerer) -> None:
        """Answer the GraphQL operation named ``operation`` (``query Name(...)``) with
        ``answer(variables, repository)``: the whole answer, ``{"data": ..., "errors": ...}``.
        The operation is the document's first definition, so a document that starts with a
        ``fragment`` isn't answered. An answer given here takes the place of the fake's own
        ``PullRequestCompare`` answer."""

        self._graphql[operation] = answer

    def _download_transport(self) -> httpx.MockTransport:
        def serve(request: httpx.Request) -> httpx.Response:
            repository_id, _, job = request.url.path.removeprefix(_LOG_PATH).partition("/")
            logs = next(
                (
                    found.logs
                    for found in self.repositories.values()
                    if str(found.id) == repository_id
                ),
                {},
            )
            text = logs.get(int(job)) if job.isdigit() else None
            return httpx.Response(404) if text is None else httpx.Response(200, text=text)

        return httpx.MockTransport(serve)

    # --------------------------------------------------------- the slot side

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
        if slot != self.slot:
            raise ConnectionCallError(
                f"The slot {slot} isn't configured.", code="connection_unbound"
            )
        sent = checked_headers(path, headers, body)
        flat = {
            name: value if isinstance(value, str) else value[-1]
            for name, value in (query or {}).items()
        }
        call = Call(method.upper(), path, flat, sent, body)
        self.requests.append(call)
        named = _repository_named(path)
        claimed = (call.repository or "").lower()
        if (named is not None and claimed != named.lower()) or (
            path == GRAPHQL_PATH and not claimed
        ):
            raise ConnectionCallError(
                f"{REPOSITORY_HEADER} must name the call's repository.", code="headers_invalid"
            )
        for fault in self._faults:
            if fault.method in ("*", call.method) and fnmatch.fnmatchcase(path, fault.path):
                fault.times -= 1
                if fault.times == 0:
                    self._faults.remove(fault)
                return fault.answer()
        if path == GRAPHQL_PATH and call.method == "POST":
            return self._answer_graphql(call)
        if named is None:
            raise self.unanswerable(call, f"FakeGitHub doesn't answer {call.method} {path}.")
        repository = self.repositories.get(named.lower())
        if repository is None:
            return _not_found()
        rest = path.removeprefix(f"/repos/{named}").removeprefix("/")
        for route_method, pattern, handler in self._routes:
            matched = pattern.fullmatch(rest)
            if route_method == call.method and matched is not None:
                groups = tuple(unquote(group) for group in matched.groups())
                return handler(_Served(self, repository, call, groups))
        raise self.unanswerable(
            call, f"FakeGitHub doesn't answer {call.method} {path}; add it with route()."
        )

    def unanswerable(self, call: Call, message: str) -> NotImplementedError:
        """The error for a call the fake can't answer, recorded in ``unanswered``."""

        self.unanswered.append(call)
        return NotImplementedError(message)

    # -------------------------------------------------------------- helpers

    def _fault(
        self, method: str, path: str, times: int, answer: Callable[[], ConnectionResponse]
    ) -> None:
        if times < 1:
            raise ValueError("times is at least 1.")
        self._faults.append(_Fault(method.upper(), path, times, answer))

    def _add(
        self, repository: str, kind: type[Any], number: int | None, settings: dict[str, Any]
    ) -> Any:
        found = self.repository(repository)
        number = number if number is not None else max(found.issues, default=0) + 1
        if number in found.issues:
            raise ValueError(f"{repository}#{number} exists already.")
        settings.setdefault("created_at", self.clock())
        item = kind(number, **settings)
        found.issues[number] = item
        return item

    def _issue(self, repository: str, number: int) -> Issue:
        return self.repository(repository).issues[number]

    def _answer_graphql(self, call: Call) -> ConnectionResponse:
        document = call.json() or {}
        matched = _OPERATION.match(_COMMENT.sub(" ", str(document.get("query", ""))))
        operation = matched.group(1) if matched else None
        answer = self._graphql.get(operation or "")
        if answer is None and operation == _COMPARE_OPERATION:
            return reply(200, self._compare_refs(document, call.repository or ""))
        if answer is None:
            raise self.unanswerable(
                call,
                "FakeGitHub has no answer for this GraphQL operation; name it "
                "(query Name(...)) and register one with on_graphql().",
            )
        return reply(200, dict(answer(document.get("variables") or {}, call.repository or "")))

    def _compare_refs(self, document: Mapping[str, Any], repository: str) -> dict[str, Any]:
        """``read_pull``'s ``PullRequestCompare`` query: each ``ref(qualifiedName:)`` field of
        the document answered from the pulls, with its ``target`` and its ``compare(headRef:)``
        when asked. As checked against GitHub, GraphQL resolves a branch but answers ``null``
        for a pull request's ``refs/pull/N/head``, though a compare takes that ref, a branch or
        a commit as its head."""

        variables = document.get("variables") or {}
        found = self.repositories.get(repository.lower())
        if found is None:
            missing = f"Could not resolve to a Repository with the name '{repository}'."
            error = {"type": "NOT_FOUND", "path": ["repository"], "message": missing}
            return {"data": {"repository": None}, "errors": [error]}
        data: dict[str, Any] = {}
        for alias, name, fields in _REF_FIELD.findall(str(document.get("query", ""))):
            qualified = str(variables.get(name, ""))
            target = _ref_target(found, qualified)
            if target is None:
                data[alias or "ref"] = None
                continue
            answered: dict[str, Any] = {}
            if re.search(r"\btarget\b", fields):
                answered["target"] = {"oid": target}
            head = _COMPARE_HEAD.search(fields)
            if head:
                base = qualified.removeprefix("refs/heads/")
                answered["compare"] = _ref_compare(found, base, str(variables.get(head[1], "")))
            data[alias or "ref"] = answered
        return {"data": {"repository": data}}

    def _page(
        self,
        call: Call,
        repository: Repository,
        items: list[Any],
        key: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> ConnectionResponse:
        """One page of ``items`` as GitHub answers it, with its next-page link, which names
        the repository by its id as GitHub's do. A ``per_page`` that isn't a positive number
        reads as GitHub's default, 30, and one over 100 as 100."""

        rest = "/".join(call.path.split("/")[4:])
        size = min(_positive(call.query.get("per_page"), DEFAULT_PER_PAGE), self.page_size)
        number = _positive(call.query.get("page"), 1, cap=None)
        chunk = items[(number - 1) * size : number * size]
        headers = {}
        if number * size < len(items):
            following = urlencode({**call.query, "page": str(number + 1)})
            headers["link"] = (
                f'<{API_URL}/repositories/{repository.id}/{rest}?{following}>; rel="next"'
            )
        body: Any = (
            chunk if key is None else {**(extra or {}), "total_count": len(items), key: chunk}
        )
        return reply(200, body, headers)


def _repository_named(path: str) -> str | None:
    parts = path.split("/")
    return "/".join(parts[2:4]) if len(parts) >= 4 and parts[1] == "repos" else None


def _positive(value: str | None, default: int, cap: int | None = MAX_PER_PAGE) -> int:
    number = int(value) if value is not None and value.isdigit() else 0
    if number < 1:
        return default
    return number if cap is None else min(number, cap)


def _when(value: str) -> datetime:
    """A ``since`` time; one with no offset is UTC, as GitHub reads it."""

    when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


def _sorted(r: _Served, items: Iterable[_Item], *, pulls: bool = False) -> list[_Item]:
    """``items`` in the listing's order. Issues default to newest first; pulls too when sorted
    by ``created``, and oldest first for any other sort, as GitHub documents."""

    key = r.call.query.get("sort", "created")
    if key not in ("created", "updated"):
        raise r.fake.unanswerable(r.call, f"FakeGitHub doesn't sort by {key}.")
    default = "desc" if not pulls or key == "created" else "asc"
    return sorted(
        items,
        key=lambda item: (getattr(item, f"{key}_at") or item.created_at, item.number),
        reverse=r.call.query.get("direction", default) == "desc",
    )


def _html(repository: Repository, kind: str, number: int) -> str:
    return f"https://github.com/{repository.full_name}/{kind}/{number}"


def _issue_body(repository: Repository, issue: Issue) -> dict[str, Any]:
    kind = "pull" if isinstance(issue, PullRequest) else "issues"
    body: dict[str, Any] = {
        "number": issue.number,
        "title": issue.title,
        "body": issue.body,
        "state": issue.state,
        "state_reason": issue.state_reason,
        "user": _user(issue.author),
        "labels": [{"name": name, "color": "ededed"} for name in issue.labels],
        "comments": len(issue.comments),
        "html_url": _html(repository, kind, issue.number),
        "created_at": _iso(issue.created_at),
        "updated_at": _iso(issue.updated_at or issue.created_at),
    }
    if isinstance(issue, PullRequest):
        body["pull_request"] = {"html_url": body["html_url"]}
    return body


def _pull_body(repository: Repository, pull: PullRequest, *, full: bool) -> dict[str, Any]:
    """A pull as a listing answers it, or (``full``) as its own read does, with mergeability."""

    body = _issue_body(repository, pull)
    del body["pull_request"], body["comments"], body["state_reason"]
    branch = {"repo": {"full_name": repository.full_name}}
    body.update(
        draft=pull.draft,
        base={"ref": pull.base, **branch},
        head={"ref": pull.head_ref, "sha": pull.head_sha, **branch},
        merge_commit_sha=pull.merge_commit_sha,
    )
    if full:
        # GitHub reads a closed pull's mergeability as null, and a new head's as null until it
        # has computed it.
        unknown = pull.state != "open" or pull.mergeable_unknown_reads > 0
        pull.mergeable_unknown_reads = max(0, pull.mergeable_unknown_reads - 1)
        behind = pull.behind_by > 0 and pull.requires_up_to_date
        state = "dirty" if not pull.mergeable else "behind" if behind else "clean"
        body.update(
            merged=pull.merged,
            merged_by=_user(pull.merged_by) if pull.merged_by else None,
            mergeable=None if unknown else pull.mergeable,
            mergeable_state="unknown" if unknown else "draft" if pull.draft else state,
        )
    return body


def _comment_body(repository: Repository, number: int, comment: Comment) -> dict[str, Any]:
    return {
        "id": comment.id,
        "body": comment.body,
        "user": _user(comment.author),
        "html_url": f"{_html(repository, 'issues', number)}#issuecomment-{comment.id}",
        "issue_url": f"{API_URL}/repos/{repository.full_name}/issues/{number}",
        "created_at": _iso(comment.created_at),
        "updated_at": _iso(comment.updated_at),
    }


def _check_body(check: CheckRun) -> dict[str, Any]:
    return {
        "id": check.id,
        "name": check.name,
        "head_sha": check.head_sha,
        "status": check.status,
        "conclusion": check.conclusion,
        "app": {"slug": check.app_slug},
        "check_suite": {"id": check.check_suite_id},
        "output": {"title": check.output_title, "summary": check.output_summary},
    }


def _run_body(repository: Repository, run: WorkflowRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "name": run.name,
        "head_sha": run.head_sha,
        "head_branch": run.head_branch,
        "event": run.event,
        "status": run.status,
        "conclusion": run.conclusion,
        "run_attempt": run.run_attempt,
        "check_suite_id": run.check_suite_id,
        "html_url": f"https://github.com/{repository.full_name}/actions/runs/{run.id}",
        "created_at": _iso(run.created_at),
    }


@dataclass(frozen=True)
class _Served:
    """One call to a built-in route: the fake, the repository, the call and the route's
    groups."""

    fake: FakeGitHub
    repository: Repository
    call: Call
    groups: tuple[str, ...]

    def filters(self, *allowed: str) -> Mapping[str, str]:
        """The call's query, once the fake answers each of its filters: it never ignores one."""

        unknown = set(self.call.query) - _PAGING - set(allowed)
        if unknown:
            names = ", ".join(sorted(unknown))
            message = f"FakeGitHub doesn't filter {self.call.method} {self.call.path} by {names}."
            raise self.fake.unanswerable(self.call, message)
        return self.call.query

    def asked(self, *allowed: str) -> dict[str, Any]:
        """The call's JSON body, once the fake models each of its fields: it never ignores
        one."""

        answer = self.call.json()
        asked = answer if isinstance(answer, dict) else {}
        unknown = set(asked) - set(allowed)
        if unknown:
            names = ", ".join(sorted(unknown))
            message = f"FakeGitHub doesn't model {names} in {self.call.method} {self.call.path}."
            raise self.fake.unanswerable(self.call, message)
        return asked

    def item(self, kind: type[Issue] = Issue) -> Any:
        """The issue or pull the route's first group numbers, if it is one of ``kind``."""

        found = self.repository.issues.get(int(self.groups[0]))
        return found if isinstance(found, kind) else None

    def page(
        self, items: list[Any], key: str | None = None, extra: Mapping[str, Any] | None = None
    ) -> ConnectionResponse:
        return self.fake._page(self.call, self.repository, items, key, extra)


def _not_found() -> ConnectionResponse:
    return _error(404, "Not Found")


def _list_pulls(r: _Served) -> ConnectionResponse:
    query = r.filters("state", "base", "head", "sort", "direction")
    state = query.get("state", "open")
    owner = r.repository.full_name.split("/")[0].lower()
    head = query.get("head")
    if head is not None and ":" not in head:
        raise r.fake.unanswerable(r.call, "FakeGitHub filters pulls by head as owner:branch.")
    pulls = [
        pull
        for pull in r.repository.pulls
        if state in ("all", pull.state)
        and query.get("base", pull.base) == pull.base
        # A pull's head is a branch of the repository itself: the fake has no forks.
        and (head is None or head.lower().partition(":")[0] == owner)
        and (head is None or head.partition(":")[2] == pull.head_ref)
    ]
    found = _sorted(r, pulls, pulls=True)
    return r.page([_pull_body(r.repository, pull, full=False) for pull in found])


def _get_pull(r: _Served) -> ConnectionResponse:
    pull = r.item(PullRequest)
    return _not_found() if pull is None else reply(200, _pull_body(r.repository, pull, full=True))


def _edit_issue(r: _Served) -> ConnectionResponse:
    """``PATCH`` an issue: its title, body, state and labels."""

    issue = r.item()
    if issue is None:
        return _not_found()
    change = r.asked("title", "body", "state", "state_reason", "labels")
    state = change.get("state", issue.state)
    if state not in ("open", "closed"):
        return _invalid("Issue", "state", "invalid")
    issue.title = change.get("title", issue.title)
    issue.body = change.get("body", issue.body)
    reopened = issue.state == "closed" and state == "open"
    issue.state = state
    issue.state_reason = change.get("state_reason", "reopened" if reopened else issue.state_reason)
    issue.labels = list(dict.fromkeys(change.get("labels", issue.labels)))
    issue.updated_at = r.fake.clock()
    return reply(200, _issue_body(r.repository, issue))


def _merge(r: _Served) -> ConnectionResponse:
    """Merge a pull at ``sha``. As checked against GitHub: a draft is refused first, then a
    closed pull, then one with conflicts (each 405, with GitHub's text), and only then a stale
    ``sha`` (409); a merge while ``mergeable`` still reads null is accepted.

    A pull that is behind under ``requires_up_to_date`` is refused with 405 too, after the
    conflicts and before the ``sha``, as GitHub refuses every merge it can't make before it
    compares the head; that text and that place in the order are unverified against GitHub,
    whose documentation gives only the 405."""

    pull = r.item(PullRequest)
    if pull is None:
        return _not_found()
    asked = r.asked("sha", "merge_method")
    sha = asked.get("sha")
    if pull.draft:
        return _error(405, "Pull Request is still a draft")
    if pull.state != "open":
        return _error(405, "Pull Request is not mergeable")
    if not pull.mergeable:
        return _error(405, "Pull Request has merge conflicts")
    if pull.behind_by > 0 and pull.requires_up_to_date:
        return _error(405, "Head branch is out of date. Review and try the merge again.")
    if sha is not None and sha != pull.head_sha:
        return _error(409, "Head branch was modified. Review and try the merge again.")
    pull.merge_commit_sha = hashlib.sha1(f"merge {pull.head_sha}".encode()).hexdigest()
    pull.state, pull.merged, pull.merged_by = "closed", True, r.fake.login
    pull.merge_method = asked.get("merge_method", "merge")
    pull.updated_at = r.fake.clock()
    merged = {"sha": pull.merge_commit_sha, "merged": True}
    return reply(200, {**merged, "message": "Pull Request successfully merged"})


def _update_branch(r: _Served) -> ConnectionResponse:
    """GitHub accepts the update (202) and merges the base in later: the head is unchanged
    when the next read comes, so a test that wants the new head sets it. As checked against
    GitHub, a stale ``expected_head_sha`` is refused first (422), then a closed pull or one with
    conflicts (422), then one that isn't behind (422), each with GitHub's text."""

    pull = r.item(PullRequest)
    if pull is None:
        return _not_found()
    expected = r.asked("expected_head_sha").get("expected_head_sha")
    if expected is not None and expected != pull.head_sha:
        return _error(422, "expected head sha didn\u2019t match current head ref.")
    if pull.state != "open" or not pull.mergeable:
        return _error(422, "merge conflict between base and head")
    if pull.behind_by == 0:
        return _error(422, "There are no new commits on the base branch.")
    url = f"{API_URL}/repos/{r.repository.full_name}/pulls/{pull.number}"
    return reply(202, {"message": "Updating pull request branch.", "url": url})


def _list_reviews(r: _Served) -> ConnectionResponse:
    pull = r.item(PullRequest)
    if pull is None:
        return _not_found()
    r.filters()
    reviews = [
        {
            "id": review.id,
            "user": _user(review.author),
            "state": review.state,
            "body": review.body,
            "commit_id": review.commit_id,
            "submitted_at": _iso(review.submitted_at),
        }
        for review in pull.reviews
    ]
    return r.page(reviews)


def _head(repository: Repository, ref: str) -> PullRequest | None:
    """The pull whose head is ``ref``: its sha, or its branch."""

    return next((pull for pull in repository.pulls if ref in (pull.head_sha, pull.head_ref)), None)


def _ref_target(repository: Repository, qualified: str) -> str | None:
    """The commit GraphQL's ``ref(qualifiedName:)`` names: a pull's head branch names its head,
    the default branch or a pull's base a commit of its own; anything else, ``refs/pull/N/head``
    included, is ``None``."""

    if not qualified.startswith("refs/heads/"):
        return None
    branch = qualified.removeprefix("refs/heads/")
    pull = next((pull for pull in repository.pulls if pull.head_ref == branch), None)
    if pull is not None:
        return pull.head_sha
    if branch == repository.default_branch or any(p.base == branch for p in repository.pulls):
        return hashlib.sha1(f"{repository.full_name}@{branch}".encode()).hexdigest()
    return None


def _ref_compare(repository: Repository, base: str, head: str) -> dict[str, int] | None:
    """GraphQL's compare of branch ``base`` with ``head`` (a commit, branch or pull request
    ref), from the pull's ``behind_by`` and ``ahead_by``; ``None`` for a head of no pull on
    ``base``."""

    if head.removeprefix("refs/heads/") == base:
        return {"aheadBy": 0, "behindBy": 0}
    number = _PULL_REF.fullmatch(head)
    branch = head.removeprefix("refs/heads/")
    pull = next(
        (
            pull
            for pull in repository.pulls
            if pull.base == base
            and (
                pull.number == int(number[1])
                if number
                else branch in (pull.head_sha, pull.head_ref)
            )
        ),
        None,
    )
    if pull is None:
        return None
    return {"aheadBy": pull.ahead_by, "behindBy": pull.behind_by}


def _compare(r: _Served) -> ConnectionResponse:
    """A pull's head against its base, from the pull's ``behind_by`` and ``ahead_by``; a ref
    against itself is identical."""

    base, head = r.groups
    pull = _head(r.repository, head)
    if base == head:
        behind = ahead = 0
    elif pull is None or pull.base != base:
        return _not_found()
    else:
        behind, ahead = pull.behind_by, pull.ahead_by
    status = {(False, False): "identical", (False, True): "ahead", (True, False): "behind"}
    counts = {"behind_by": behind, "ahead_by": ahead, "total_commits": ahead}
    return reply(200, {"status": status.get((behind > 0, ahead > 0), "diverged"), **counts})


def _get_commit(r: _Served) -> ConnectionResponse:
    pull = _head(r.repository, r.groups[0])
    if pull is None:
        return _not_found()
    when = {"date": _iso(pull.committed_at)}
    return reply(200, {"sha": pull.head_sha, "author": when, "committer": when})


def _list_checks(r: _Served) -> ConnectionResponse:
    query = r.filters("check_name", "status", "filter")
    pull = _head(r.repository, r.groups[0])
    sha = pull.head_sha if pull is not None else r.groups[0]
    every = query.get("filter", "latest") == "all"
    checks = [
        _check_body(check)
        for check in r.repository.check_runs
        if check.head_sha == sha
        and (every or not check.superseded)
        and query.get("check_name", check.name) == check.name
        and query.get("status", check.status) == check.status
    ]
    return r.page(checks, "check_runs")


def _combined_status(r: _Served) -> ConnectionResponse:
    """A commit's combined status: the latest status of each context, newest first, and their
    state as GitHub documents it: ``failure`` when one is ``failure`` or ``error``, else
    ``pending`` when one is pending or there are none, else ``success``."""

    r.filters()
    pull = _head(r.repository, r.groups[0])
    sha = pull.head_sha if pull is not None else r.groups[0]
    latest: dict[str, CommitStatus] = {}
    for status in sorted(r.repository.statuses, key=lambda status: (status.created_at, status.id)):
        if status.sha == sha:
            latest[status.context] = status
    states = {status.state for status in latest.values()}
    if states & {"failure", "error"}:
        state = "failure"
    elif not states or "pending" in states:
        state = "pending"
    else:
        state = "success"
    statuses = [
        {
            "id": status.id,
            "context": status.context,
            "state": status.state,
            "description": status.description,
            "target_url": status.target_url,
            "created_at": _iso(status.created_at),
            "updated_at": _iso(status.created_at),
        }
        for status in sorted(
            latest.values(), key=lambda status: (status.created_at, status.id), reverse=True
        )
    ]
    return r.page(statuses, "statuses", {"state": state, "sha": sha})


def _list_runs(r: _Served) -> ConnectionResponse:
    query = r.filters("head_sha", "branch", "status", "event")
    runs = [
        _run_body(r.repository, run)
        for run in sorted(r.repository.workflow_runs, key=lambda run: (run.created_at, run.id))
        if query.get("head_sha", run.head_sha) == run.head_sha
        and query.get("branch", run.head_branch) == run.head_branch
        and query.get("event", run.event) == run.event
        and query.get("status", run.status) in (run.status, run.conclusion)
    ]
    return r.page(runs[::-1], "workflow_runs")


def _get_run(r: _Served) -> ConnectionResponse:
    run = next((run for run in r.repository.workflow_runs if run.id == int(r.groups[0])), None)
    return _not_found() if run is None else reply(200, _run_body(r.repository, run))


def _rerun(r: _Served) -> ConnectionResponse:
    """A rerun starts the run's next attempt: each job rerun gets a new check run, queued, and
    the one it replaces is listed only with ``filter=all``. ``finish_run`` completes it. A run
    still going, or a rerun of failed jobs when none failed, is refused with 403 and GitHub's
    text, as checked against GitHub."""

    run = next((run for run in r.repository.workflow_runs if run.id == int(r.groups[0])), None)
    if run is None:
        return _not_found()
    r.asked()
    if run.status != "completed":
        return _error(403, "This workflow is already running")
    failed_only = r.call.path.endswith("/rerun-failed-jobs")
    if failed_only and run.conclusion in _PASSED:
        return _error(403, "This workflow run cannot be retried")
    run.run_attempt += 1
    run.status, run.conclusion = "queued", None
    for check in list(r.repository.check_runs):
        if check.check_suite_id != run.check_suite_id or check.superseded:
            continue
        if failed_only and check.conclusion in _PASSED:
            continue
        again = replace(check, id=next(r.fake._ids), status="queued", conclusion=None)
        check.superseded = True
        r.repository.check_runs.append(again)
    return reply(201, {})


def _job_log(r: _Served) -> ConnectionResponse:
    """A log read answers 302 to a signed download, as GitHub's does, which the fake's
    ``download_transport`` serves."""

    if int(r.groups[0]) not in r.repository.logs:
        return _not_found()
    location = f"{LOG_URL}{r.repository.id}/{r.groups[0]}?sig=fake"
    return reply(302, None, {"location": location})


def _list_issues(r: _Served) -> ConnectionResponse:
    query = r.filters("state", "labels", "since", "creator", "sort", "direction")
    state = query.get("state", "open")
    wanted = [label for label in query.get("labels", "").split(",") if label]
    since = _when(query["since"]) if "since" in query else None
    issues = [
        issue
        for issue in r.repository.issues.values()
        if state in ("all", issue.state)
        and all(label in issue.labels for label in wanted)
        and query.get("creator", issue.author).casefold() == issue.author.casefold()
        and (since is None or (issue.updated_at or issue.created_at) >= since)
        and r.fake.listed(issue.created_at)
    ]
    return r.page([_issue_body(r.repository, issue) for issue in _sorted(r, issues)])


def _create_issue(r: _Served) -> ConnectionResponse:
    asked = r.asked("title", "body", "labels")
    if not asked.get("title"):
        return _invalid("Issue", "title")
    issue = r.fake.add_issue(
        r.repository.full_name,
        title=asked["title"],
        body=asked.get("body") or "",
        labels=list(asked.get("labels", [])),
        author=r.fake.login,
    )
    return reply(201, _issue_body(r.repository, issue))


def _get_issue(r: _Served) -> ConnectionResponse:
    issue = r.item()
    return _not_found() if issue is None else reply(200, _issue_body(r.repository, issue))


def _list_comments(r: _Served) -> ConnectionResponse:
    issue = r.item()
    if issue is None:
        return _not_found()
    query = r.filters("since")
    since = _when(query["since"]) if "since" in query else None
    comments = [
        _comment_body(r.repository, issue.number, comment)
        for comment in issue.comments
        if (since is None or comment.updated_at >= since) and r.fake.listed(comment.created_at)
    ]
    return r.page(comments)


def _create_comment(r: _Served) -> ConnectionResponse:
    issue = r.item()
    if issue is None:
        return _not_found()
    body = r.asked("body").get("body")
    if not body:
        return _invalid("IssueComment", "body")
    comment = r.fake.add_comment(r.repository.full_name, issue.number, body, author=r.fake.login)
    issue.updated_at = comment.created_at
    return reply(201, _comment_body(r.repository, issue.number, comment))


def _get_comment(r: _Served) -> ConnectionResponse:
    for issue in r.repository.issues.values():
        for comment in issue.comments:
            if comment.id == int(r.groups[0]):
                return reply(200, _comment_body(r.repository, issue.number, comment))
    return _not_found()


def _edit_comment(r: _Served) -> ConnectionResponse:
    for issue in r.repository.issues.values():
        for comment in issue.comments:
            if comment.id == int(r.groups[0]):
                body = r.asked("body").get("body")
                if not body:
                    return _invalid("IssueComment", "body")
                comment.body, comment.updated_at = body, r.fake.clock()
                return reply(200, _comment_body(r.repository, issue.number, comment))
    return _not_found()


def _labels(r: _Served) -> ConnectionResponse:
    """Add labels (``POST``), set them (``PUT``) or remove one (``DELETE .../labels/NAME``).
    A label list is ``{"labels": [...]}`` or the array itself, and an issue has each label
    once."""

    issue = r.item()
    if issue is None:
        return _not_found()
    if r.call.method == "DELETE":
        if r.groups[1] not in issue.labels:
            return _error(404, "Label does not exist")
        issue.labels.remove(r.groups[1])
    else:
        given = r.call.json()
        asked = given if isinstance(given, list) else r.asked("labels").get("labels")
        if not isinstance(asked, list) or not all(isinstance(name, str) for name in asked):
            return _invalid("Label", "labels")
        kept = [] if r.call.method == "PUT" else issue.labels
        issue.labels = list(dict.fromkeys(kept + asked))
    issue.updated_at = r.fake.clock()
    return reply(200, [{"name": name, "color": "ededed"} for name in issue.labels])


def _list_releases(r: _Served) -> ConnectionResponse:
    r.filters()
    releases = [
        {
            "id": release.id,
            "tag_name": release.tag_name,
            "name": release.name or release.tag_name,
            "draft": release.draft,
            "prerelease": release.prerelease,
            "published_at": _iso(release.published_at),
        }
        for release in sorted(r.repository.releases, key=lambda release: release.published_at)
    ]
    return r.page(releases[::-1])


def _permission(r: _Served) -> ConnectionResponse:
    """A collaborator's permission. As checked against GitHub, anyone else's is ``read`` on a
    public repository and ``none`` on a private one, and a login that isn't a user is 404."""

    login = r.groups[0]
    if not _is_user(r, login):
        return _error(404, f"{login} is not a user")
    default = "none" if r.repository.private else "read"
    permission = r.repository.permissions.get(login, default)
    role = "" if permission == "none" else permission
    return reply(200, {"permission": permission, "role_name": role, "user": _user(login)})


def _is_user(r: _Served, login: str) -> bool:
    """Whether ``login`` is a user: one of ``users``, the fake's own login, a collaborator, or
    the author of anything in the repository."""

    found = r.repository
    authors = {item.author for item in found.issues.values()}
    authors |= {comment.author for item in found.issues.values() for comment in item.comments}
    authors |= {review.author for pull in found.pulls for review in pull.reviews}
    known = r.fake.users | {r.fake.login} | set(found.permissions) | authors
    return login in known


#: The REST routes the fake answers under ``/repos/{owner}/{name}``: a method, the rest of the
#: path as a regular expression, and how it is answered.
_ROUTES: tuple[tuple[str, re.Pattern[str], Callable[[_Served], ConnectionResponse]], ...] = tuple(
    (method, re.compile(pattern), handler)
    for method, pattern, handler in (
        ("GET", r"pulls", _list_pulls),
        ("GET", r"pulls/(\d+)", _get_pull),
        ("PUT", r"pulls/(\d+)/merge", _merge),
        ("PUT", r"pulls/(\d+)/update-branch", _update_branch),
        ("GET", r"pulls/(\d+)/reviews", _list_reviews),
        ("GET", r"compare/(.+)\.\.\.(.+)", _compare),
        ("GET", r"git/commits/([^/]+)", _get_commit),
        ("GET", r"commits/(.+)/check-runs", _list_checks),
        ("GET", r"commits/(.+)/status", _combined_status),
        ("GET", r"actions/runs", _list_runs),
        ("GET", r"actions/runs/(\d+)", _get_run),
        ("POST", r"actions/runs/(\d+)/rerun", _rerun),
        ("POST", r"actions/runs/(\d+)/rerun-failed-jobs", _rerun),
        ("GET", r"actions/jobs/(\d+)/logs", _job_log),
        ("GET", r"issues", _list_issues),
        ("POST", r"issues", _create_issue),
        ("GET", r"issues/(\d+)", _get_issue),
        ("PATCH", r"issues/(\d+)", _edit_issue),
        ("GET", r"issues/(\d+)/comments", _list_comments),
        ("POST", r"issues/(\d+)/comments", _create_comment),
        ("GET", r"issues/comments/(\d+)", _get_comment),
        ("PATCH", r"issues/comments/(\d+)", _edit_comment),
        ("POST", r"issues/(\d+)/labels", _labels),
        ("PUT", r"issues/(\d+)/labels", _labels),
        ("DELETE", r"issues/(\d+)/labels/(.+)", _labels),
        ("GET", r"releases", _list_releases),
        ("GET", r"collaborators/([^/]+)/permission", _permission),
    )
)
