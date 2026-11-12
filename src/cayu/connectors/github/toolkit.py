"""GitHub as tools for the model, on the client and the pull request read model."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Collection, Mapping
from typing import Any, ClassVar

from cayu.connections import ConnectionCallError, ConnectionClient
from cayu.connectors import _common
from cayu.connectors.github._errors import (
    GitHubAppNotInstalled,
    GitHubConflict,
    GitHubConnectionRefused,
    GitHubError,
    GitHubNotFound,
    GitHubPermissionDenied,
    GitHubRateLimited,
    GitHubUnavailable,
)
from cayu.connectors.github.client import (
    DEFAULT_SLOT,
    GitHubClient,
    _checked_repository,
    _LogDownloadFailed,
)
from cayu.connectors.github.markers import plain_text
from cayu.connectors.github.pulls import (
    PullCheck,
    PullRequestFacts,
    PullState,
    list_pulls,
    read_pull,
)
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy

#: The longest comment GitHub accepts.
MAX_COMMENT_CHARACTERS = 65_536

READ_PULL = "github_read_pull"
LIST_PULLS = "github_list_pulls"
READ_ISSUE = "github_read_issue"
READ_JOB_LOG = "github_read_job_log"
COMMENT = "github_comment"
RERUN_FAILED_JOBS = "github_rerun_failed_jobs"
UPDATE_BRANCH = "github_update_branch"
MERGE_PULL = "github_merge_pull"

_DEFAULT_PULLS = 20
_MAX_PULLS = 100
#: Pull requests ``list_pulls`` answers a page.
_PULLS_PAGE = 20
_DEFAULT_COMMENTS = 20
#: The page size the newest comments are read in: small enough that a page of the longest
#: comments GitHub takes stays under the slot's answer limit.
_COMMENTS_PAGE = 25
_TOO_LARGE = "upstream_response_too_large"
_MAX_COMMENTS = 50
_DEFAULT_LOG_LINES = 200
_MAX_LOG_LINES = 2_000
#: The most of a log's tail the model reads, in characters.
_LOG_CHARACTERS = 40_000
#: The failed, running and reviewer checks ``github_read_pull`` lists of each; the
#: ``*_total`` fields count them all.
_LISTED_CHECKS = 30
_CHECK_TITLE_CHARACTERS = 200
_CHECK_NAME_CHARACTERS = 200
#: How long a comment or a check's text may be, tried in turn until the answer fits.
_TEXT_LIMITS = (2_000, 500, 100, 20)
_ISSUE_BODY_CHARACTERS = 8_000
_SUMMARY_CHARACTERS = 500
_SHORT_SHA = 7
#: GitHub's answer to a merge it can't make as things stand.
_NOT_MERGEABLE = 405
_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
#: Control characters other than tab and line breaks, shown as escapes in a log's text.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MERGE_METHODS = ("merge", "squash", "rebase")
_STATES = {
    "open": (PullState.OPEN,),
    "closed": (PullState.CLOSED,),
    "merged": (PullState.MERGED,),
    "all": None,
}

_NUMBER = {"type": "integer", "minimum": 1, "description": "The pull request's or issue's number."}
_HEAD_SHA = {
    "type": "string",
    "description": "The pull request's head commit, in full, as github_read_pull answered it. "
    "GitHub refuses the call if the head has moved since.",
}


def tools(
    client: ConnectionClient | None = None,
    slot: str = DEFAULT_SLOT,
    *,
    include_writes: bool = False,
    write_slot: str | None = None,
    repositories: Collection[str] | None = None,
    reviewer_app: str | None = None,
    plain_comments: bool = True,
) -> list[Tool]:
    """The GitHub read tools on ``slot``, and with ``include_writes=True`` the comment, rerun,
    branch update and merge tools too (pair them with ``approval_policy()``), calling through
    ``client`` (default: Cayu Cloud's broker). The write tools use ``write_slot`` (default:
    ``slot``), so the read tools can be bound to a connection that can only read.

    ``repositories`` (``owner/name``) are the only ones the tools act on; without it, any the
    connection reaches. ``reviewer_app`` is the slug of the GitHub App whose check runs
    ``github_read_pull`` and ``github_list_pulls`` keep apart from CI. ``github_comment``
    posts the model's text through ``plain_text`` (no mentions, no hidden HTML comments) unless
    ``plain_comments=False``.
    """

    resolved = _common.resolved_client(client)
    gh = None if resolved is None else GitHubClient(resolved, slot)
    allowed = None if repositories is None else _allowed(repositories)
    kinds: tuple[type[_GitHubTool], ...] = (_ReadPull, _ListPulls, _ReadIssue, _ReadJobLog)
    reads = None if gh is None else gh.read_only_view()
    made: list[Tool] = [kind(reads, allowed, reviewer_app) for kind in kinds]
    if include_writes is True:
        writes = gh
        if resolved is not None and write_slot is not None and write_slot != slot:
            writes = GitHubClient(resolved, write_slot)
        made.append(_Comment(writes, allowed, reviewer_app, plain=plain_comments))
        kinds = (_RerunFailedJobs, _UpdateBranch, _MergePull)
        made.extend(kind(writes, allowed, reviewer_app) for kind in kinds)
    return made


def approval_policy(inner: ToolPolicy | None = None) -> ToolPolicy:
    """Ask for approval before a comment, a rerun, a branch update or a merge, naming the
    repository, the number and the head commit; every other decision is ``inner``'s (default:
    allow)."""

    return _common.approval_policy(inner, _REASONS)


def review_fields() -> dict[str, tuple[str, ...]]:
    """The arguments an approver sees, for ``ToolArgumentReview``."""

    return {
        COMMENT: ("repository", "number", "body"),
        RERUN_FAILED_JOBS: ("repository", "run_id", "head_sha"),
        UPDATE_BRANCH: ("repository", "number", "head_sha"),
        MERGE_PULL: ("repository", "number", "head_sha", "merge_method"),
    }


def _target(arguments: Mapping[str, Any]) -> str:
    repository = _common.reason_argument(arguments.get("repository"))
    return f"{repository}#{_common.reason_argument(arguments.get('number'))}"


def _head(arguments: Mapping[str, Any]) -> str:
    return _common.reason_argument(str(arguments.get("head_sha"))[:_SHORT_SHA])


def _merge_reason(arguments: Mapping[str, Any]) -> str:
    method = arguments.get("merge_method")
    how = f" ({_common.reason_argument(method)})" if method else ""
    return f"Merge {_target(arguments)} at head {_head(arguments)}{how}."


_REASONS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    COMMENT: lambda arguments: f"Comment on {_target(arguments)}.",
    RERUN_FAILED_JOBS: lambda arguments: (
        f"Rerun the failed jobs of workflow run {_common.reason_argument(arguments.get('run_id'))}"
        f" in {_common.reason_argument(arguments.get('repository'))}"
        + (f" at head {_head(arguments)}." if arguments.get("head_sha") else ".")
    ),
    UPDATE_BRANCH: lambda arguments: (
        f"Update the branch of {_target(arguments)} at head {_head(arguments)}."
    ),
    MERGE_PULL: _merge_reason,
}


def _allowed(repositories: Collection[str]) -> dict[str, str]:
    """Each allowed repository by its lowercased name, as GitHub matches names."""

    if isinstance(repositories, str):
        raise TypeError("repositories is a collection of owner/name strings, not one string.")
    names = [_checked_repository(name) for name in repositories]
    if not names:
        raise ValueError("repositories names at least one repository.")
    return {name.lower(): name for name in names}


class _InvalidArguments(ValueError):
    pass


class _Failed(Exception):
    def __init__(self, result: ToolResult) -> None:
        super().__init__(result.content)
        self.result = result


class _GitHubTool(Tool):
    NAME: ClassVar[str]
    DESCRIPTION: ClassVar[str]
    PROPERTIES: ClassVar[dict[str, Any]]
    REQUIRED: ClassVar[tuple[str, ...]]
    EFFECT: ClassVar[ToolEffect] = ToolEffect.NONE
    #: How to tell whether a write that failed ambiguous took effect.
    CHECK: ClassVar[str] = "check whether it did"

    def __init__(
        self, gh: GitHubClient | None, allowed: dict[str, str] | None, reviewer_app: str | None
    ) -> None:
        repository: dict[str, Any] = {"type": "string", "description": "owner/name."}
        if allowed is not None:
            repository["enum"] = sorted(allowed.values())
        super().__init__(
            ToolSpec(
                name=self.NAME,
                description=self.DESCRIPTION,
                input_schema={
                    "type": "object",
                    "properties": {"repository": repository, **self.PROPERTIES},
                    "required": ["repository", *self.REQUIRED],
                    "additionalProperties": False,
                },
                effect=self.EFFECT,
                parallel_safe=self.EFFECT is ToolEffect.NONE,
                max_terminal_payload_bytes=_common.MAX_TERMINAL_PAYLOAD_BYTES,
            )
        )
        self._gh = gh
        self._allowed = allowed
        self._reviewer_app = reviewer_app

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if self._gh is None:
            return _common.no_client_result()
        try:
            return await self._run(self._gh, self._repository(args), args)
        except _Failed as failed:
            return failed.result
        except _InvalidArguments as error:
            return _common.invalid_arguments(str(error))
        except GitHubError as error:
            return _error_result(error, self._gh.slot, self.CHECK)

    async def _run(self, gh: GitHubClient, repository: str, args: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def _repository(self, args: Mapping[str, Any]) -> str:
        repository = args.get("repository")
        try:
            if not isinstance(repository, str):
                raise ValueError(repository)
            repository = _checked_repository(repository)
        except ValueError:
            raise _InvalidArguments("repository is owner/name.") from None
        if self._allowed is None:
            return repository
        found = self._allowed.get(repository.lower())
        if found is None:
            raise _Failed(
                _common.error_result(
                    "repository_not_allowed",
                    f"These tools don't act on {repository}.",
                    f"Use one of: {', '.join(sorted(self._allowed.values()))}.",
                )
            )
        return found


def _error_result(error: GitHubError, slot: str, check: str) -> ToolResult:
    """A failed GitHub call as the model reads it: its code, GitHub's words and what to do. A
    write that may have taken effect (``ambiguous``) is advised to ``check`` first, never to
    retry."""

    message = error.message
    if error.ambiguous:
        if isinstance(error, GitHubConnectionRefused):
            code = error.code
        elif isinstance(error, GitHubUnavailable):
            code = "upstream_failed"
        else:
            code = "github_error"
        return _common.ambiguous_write_result(code, message, check)
    if isinstance(error, GitHubRateLimited):
        return _common.rate_limited_result(message, error.retry_after)
    if isinstance(error, GitHubConnectionRefused):
        refused = ConnectionCallError(message, code=error.code, status=error.status)
        return _common.call_error_result(refused, slot)
    if isinstance(error, GitHubNotFound):
        return _common.error_result(
            "github_not_found",
            message,
            "Check the repository and number; this connection may not see it.",
        )
    if isinstance(error, GitHubPermissionDenied):
        return _common.error_result(
            "github_permission_denied",
            message,
            "The connection lacks the permission; an admin must grant it. Don't retry.",
            accepted_permissions=error.accepted_permissions,
        )
    if isinstance(error, GitHubAppNotInstalled):
        return _common.error_result(
            "github_app_not_installed",
            message,
            "An admin must install the GitHub App on the repository.",
        )
    if isinstance(error, GitHubConflict):
        return _common.error_result(
            "github_conflict",
            message,
            "GitHub refused it as things stand; read the pull request or issue again first.",
            codes=list(error.codes) or None,
        )
    if isinstance(error, GitHubUnavailable):
        return _common.error_result("upstream_failed", message, _common.TEMPORARY)
    return _common.error_result("github_error", message, status=error.status)


def _integer(
    args: Mapping[str, Any], name: str, default: int, maximum: int, minimum: int = 1
) -> int:
    value = args.get(name, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise _InvalidArguments(f"{name} must be an integer from {minimum} to {maximum}.")
    return value


def _number(args: Mapping[str, Any], name: str = "number") -> int:
    value = args.get(name)
    if type(value) is not int or value < 1:
        raise _InvalidArguments(f"{name} must be a positive integer.")
    return value


def _sha(args: Mapping[str, Any]) -> str:
    value = args.get("head_sha")
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise _InvalidArguments(
            "head_sha is the full head commit: 40 (or 64) lowercase hex characters."
        )
    return value


def _text(args: Mapping[str, Any], name: str, maximum: int) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise _InvalidArguments(f"{name} is a nonblank string of at most {maximum} characters.")
    return value


def _fitting(build: Callable[[int], dict[str, Any]]) -> ToolResult:
    """The answer ``build(limit)`` makes with the longest text limit that fits."""

    for limit in _TEXT_LIMITS:
        result = _common.answer_result(_storable(build(limit)))
        if _common.answer_fits(result):
            return result
    raise _too_large()


def _fitted(answer: dict[str, Any]) -> ToolResult:
    """``answer`` when it fits."""

    result = _common.answer_result(_storable(answer))
    if _common.answer_fits(result):
        return result
    raise _too_large()


def _storable(value: Any) -> Any:
    """``value`` with every string a tool result can store: a NUL GitHub's text may hold is
    written as ``\\x00``, and a lone surrogate as ``?``."""

    if isinstance(value, str):
        text = value.replace("\x00", "\\x00")
        return text.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {key: _storable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_storable(item) for item in value]
    return value


def _too_large() -> _Failed:
    return _Failed(
        _common.error_result(
            "answer_too_large",
            "The answer is too large to give.",
            "Ask for fewer items with a smaller limit.",
        )
    )


def _when(value: Any) -> str | None:
    return None if value is None else value.isoformat()


def _shortened(text: str | None, limit: int, answer: dict[str, Any], name: str) -> None:
    if not text:
        return
    answer[name], cut = _common.shortened(text, limit)
    if cut:
        answer[f"{name}_shortened"] = True


def _comment(comment: Mapping[str, Any], limit: int) -> dict[str, Any]:
    user = comment.get("user")
    answer: dict[str, Any] = {
        "id": comment.get("id"),
        "author": user.get("login") if isinstance(user, dict) else None,
        "created_at": comment.get("created_at"),
    }
    _shortened(str(comment.get("body") or ""), limit, answer, "body")
    return answer


def _capped(text: str, cap: int, limit: int) -> str:
    """``text`` cut to its own ``cap``, and to the answer's text ``limit`` when that is less."""

    return _common.shortened(text, min(cap, limit))[0]


def _failed_check(check: PullCheck, limit: int) -> dict[str, Any]:
    answer: dict[str, Any] = {"name": _name(check.name, limit), "conclusion": check.conclusion}
    if check.app == "github-actions" and check.id is not None:
        answer["job_id"] = check.id
    if check.workflow_run is not None:
        answer["run_id"] = check.workflow_run.id
        answer["attempt"] = check.workflow_run.attempt
    if check.title:
        answer["title"] = _capped(check.title, _CHECK_TITLE_CHARACTERS, limit)
    if check.url:
        answer["url"] = check.url
    return answer


def _name(name: str, limit: int) -> str:
    return _capped(name, _CHECK_NAME_CHARACTERS, limit)


def _reviewer_check(check: PullCheck, limit: int) -> dict[str, Any]:
    seen: dict[str, Any] = {"name": _name(check.name, limit), "status": check.status}
    seen["conclusion"] = check.conclusion
    seen["title"] = (
        None if check.title is None else _capped(check.title, _CHECK_TITLE_CHARACTERS, limit)
    )
    _shortened(check.summary, min(_SUMMARY_CHARACTERS, limit), seen, "summary")
    return seen


def _checks(facts: PullRequestFacts, limit: int) -> dict[str, Any]:
    """The head's checks, each list at most ``_LISTED_CHECKS`` long with its total, their
    texts cut to ``limit``."""

    checks = facts.checks
    answer: dict[str, Any] = {
        "state": checks.state.value,
        "complete": checks.complete,
        "passed": sum(1 for check in checks.ci if check.finished and not check.failed),
    }
    running = [_name(name, limit) for name in sorted({check.name for check in checks.running})]
    for kind, items in (
        ("failed", [_failed_check(check, limit) for check in checks.failed]),
        ("running", running),
        ("reviewer", [_reviewer_check(check, limit) for check in checks.reviewer]),
    ):
        answer[kind] = items[:_LISTED_CHECKS]
        answer[f"{kind}_total"] = len(items)
        if len(items) > _LISTED_CHECKS:
            answer[f"{kind}_note"] = (
                f"Only the first {_LISTED_CHECKS} are listed; {kind}_total counts all."
            )
    return answer


def _reviews_complete(facts: PullRequestFacts) -> bool:
    """Whether every review was read, so no standing review can be missing."""

    return not any(gap.fact == "reviews" for gap in facts.gaps)


def _pull(facts: PullRequestFacts, limit: int) -> dict[str, Any]:
    """What the model reads of a pull request, without its comments, its check texts cut to
    ``limit``."""

    return {
        "repository": facts.repository,
        "number": facts.number,
        "title": facts.title,
        "url": facts.url,
        "author": facts.author,
        "state": facts.state.value,
        "draft": facts.draft,
        "labels": list(facts.labels),
        "base": facts.base,
        "head_ref": facts.head_ref,
        "head_sha": facts.head_sha,
        "head_committed_at": _when(facts.head_committed_at),
        "updated_at": _when(facts.updated_at),
        "mergeable": facts.mergeable,
        "mergeable_state": facts.mergeable_state,
        "behind_by": facts.behind_by,
        "ahead_by": facts.ahead_by,
        "checks": _checks(facts, limit),
        "reviews": [
            {
                "author": review.author,
                "state": review.state,
                "commit_id": review.commit_id,
                "submitted_at": _when(review.submitted_at),
            }
            for review in facts.latest_reviews
        ],
        "reviews_complete": _reviews_complete(facts),
        "gaps": [{"fact": gap.fact, "reason": gap.reason} for gap in facts.gaps],
    }


class _ReadPull(_GitHubTool):
    NAME = READ_PULL
    DESCRIPTION = (
        "Read a pull request's facts: state, labels, head commit, mergeability, how far behind "
        "its base, CI checks (failed ones with their job_id and run_id), standing reviews and "
        f"its newest {_DEFAULT_COMMENTS} comments. gaps names any fact that couldn't be read."
    )
    PROPERTIES = {"number": _NUMBER}
    REQUIRED = ("number",)

    async def _run(self, gh: GitHubClient, repository: str, args: dict[str, Any]) -> ToolResult:
        number = _number(args)
        facts = await read_pull(
            gh, repository, number, reviewer_app=self._reviewer_app, comments=False
        )
        # The conversation's tail, read from its last pages, however long it is.
        total = (await gh.get(f"/repos/{repository}/issues/{number}")).get("comments")
        newest, gap = await _newest_comments(gh, repository, number, total, _DEFAULT_COMMENTS)
        gaps = [] if gap is None else [gap]

        def build(limit: int) -> dict[str, Any]:
            pull = _pull(facts, limit)
            pull["gaps"].extend(gaps)
            return {
                **pull,
                "comments_total": total,
                "comments": [_comment(comment, limit) for comment in newest],
            }

        return _fitting(build)


class _ListPulls(_GitHubTool):
    NAME = LIST_PULLS
    DESCRIPTION = (
        "List a repository's pull requests, most recently updated first: number, title, "
        "author, labels, head commit, mergeability, CI state with failed check names and "
        "standing reviews. Read one with github_read_pull for more."
    )
    PROPERTIES = {
        "state": {"type": "string", "enum": list(_STATES), "description": "Default open."},
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MAX_PULLS,
            "description": f"How many to return (default {_DEFAULT_PULLS}).",
        },
    }
    REQUIRED = ()

    async def _run(self, gh: GitHubClient, repository: str, args: dict[str, Any]) -> ToolResult:
        state = args.get("state", "open")
        if state not in _STATES:
            raise _InvalidArguments(f"state is one of {', '.join(_STATES)}.")
        limit = _integer(args, "limit", _DEFAULT_PULLS, _MAX_PULLS)
        listed = await list_pulls(
            gh,
            repository,
            states=_STATES[state],
            reviewer_app=self._reviewer_app,
            max_pages=math.ceil(limit / _PULLS_PAGE),
        )
        pulls = [
            {
                "number": facts.number,
                "title": facts.title,
                "author": facts.author,
                "state": facts.state.value,
                "draft": facts.draft,
                "labels": list(facts.labels),
                "base": facts.base,
                "head_ref": facts.head_ref,
                "head_sha": facts.head_sha,
                "updated_at": _when(facts.updated_at),
                "mergeable": facts.mergeable,
                "checks": facts.checks.state.value,
                "failed_checks": list(facts.checks.failed_names),
                "reviews": [
                    {"author": review.author, "state": review.state}
                    for review in facts.latest_reviews
                ],
                "reviews_complete": _reviews_complete(facts),
                "gaps": [{"fact": gap.fact, "reason": gap.reason} for gap in facts.gaps],
            }
            for facts in listed.pulls[:limit]
        ]
        complete = listed.complete and len(listed.pulls) <= limit
        return _fitted({"repository": repository, "pulls": pulls, "complete": complete})


class _ReadIssue(_GitHubTool):
    NAME = READ_ISSUE
    DESCRIPTION = (
        "Read an issue, or a pull request's conversation, by number: title, state, labels, "
        "body and its newest comments."
    )
    PROPERTIES = {
        "number": _NUMBER,
        "comments": {
            "type": "integer",
            "minimum": 0,
            "maximum": _MAX_COMMENTS,
            "description": f"How many of the newest comments to return (default "
            f"{_DEFAULT_COMMENTS}).",
        },
    }
    REQUIRED = ("number",)

    async def _run(self, gh: GitHubClient, repository: str, args: dict[str, Any]) -> ToolResult:
        number = _number(args)
        wanted = _integer(args, "comments", _DEFAULT_COMMENTS, _MAX_COMMENTS, minimum=0)
        issue = await gh.get(f"/repos/{repository}/issues/{number}")
        total = issue.get("comments")
        newest, gap = await _newest_comments(gh, repository, number, total, wanted)
        user = issue.get("user")

        def build(limit: int) -> dict[str, Any]:
            answer: dict[str, Any] = {
                "repository": repository,
                "number": issue.get("number"),
                "title": issue.get("title"),
                "url": issue.get("html_url"),
                "state": issue.get("state"),
                "state_reason": issue.get("state_reason"),
                "author": user.get("login") if isinstance(user, dict) else None,
                "labels": [label.get("name") for label in issue.get("labels") or ()],
                "is_pull_request": "pull_request" in issue,
                "comments_total": total,
                "comments": [_comment(comment, limit) for comment in newest],
                "gaps": [] if gap is None else [gap],
            }
            _shortened(issue.get("body"), _ISSUE_BODY_CHARACTERS, answer, "body")
            return answer

        return _fitting(build)


async def _newest_comments(
    gh: GitHubClient, repository: str, number: int, total: object, wanted: int
) -> tuple[list[Any], dict[str, str] | None]:
    """The newest ``wanted`` of ``total`` comments, from the last pages of the listing without
    reading the pages before them; or none, and the gap that says why, when a page is over the
    slot's answer limit."""

    if not wanted or type(total) is not int or total < 1:
        return [], None
    path = f"/repos/{repository}/issues/{number}/comments"
    page = math.ceil(total / _COMMENTS_PAGE)
    newest: list[Any] = []
    while page >= 1 and len(newest) < wanted:
        query = {"per_page": str(_COMMENTS_PAGE), "page": str(page)}
        try:
            newest = list(await gh.get(path, query=query) or ()) + newest
        except GitHubConnectionRefused as error:
            if error.code != _TOO_LARGE:
                raise
            return [], {"fact": "comments", "reason": error.message}
        page -= 1
    return newest[-wanted:], None


class _ReadJobLog(_GitHubTool):
    NAME = READ_JOB_LOG
    DESCRIPTION = (
        "Read the end of a GitHub Actions job's log, by the job_id github_read_pull gives a "
        "failed check."
    )
    PROPERTIES = {
        "job_id": {"type": "integer", "minimum": 1},
        "lines": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MAX_LOG_LINES,
            "description": f"How many of the last lines to return (default {_DEFAULT_LOG_LINES}).",
        },
    }
    REQUIRED = ("job_id",)

    async def _run(self, gh: GitHubClient, repository: str, args: dict[str, Any]) -> ToolResult:
        job_id = _number(args, "job_id")
        lines = _integer(args, "lines", _DEFAULT_LOG_LINES, _MAX_LOG_LINES)
        try:
            log = await gh.text(f"/repos/{repository}/actions/jobs/{job_id}/logs")
        except _LogDownloadFailed as error:
            raise _Failed(
                _common.error_result(
                    "log_unavailable",
                    error.message,
                    "The log can't be downloaded from here; don't retry. Open the check's url "
                    "to read it.",
                )
            ) from None
        shown = _CONTROL.sub(lambda found: f"\\x{ord(found[0]):02x}", _ANSI.sub("", log))
        every = _storable(shown).splitlines()
        tail = "\n".join(every[-lines:])
        truncated = len(every) > lines or len(tail) > _LOG_CHARACTERS
        tail = tail[-_LOG_CHARACTERS:]
        while True:
            # Cut by the answer's encoded size: escaping and non-ASCII text make it larger
            # than its characters.
            result = _common.answer_result(
                {"repository": repository, "job_id": job_id, "log": tail, "truncated": truncated}
            )
            if _common.answer_fits(result):
                return result
            tail, truncated = tail[len(tail) - len(tail) // 2 :], True


class _Comment(_GitHubTool):
    NAME = COMMENT
    DESCRIPTION = (
        "Comment on a pull request or an issue, in GitHub Markdown. It may need a person's "
        "approval first."
    )
    PROPERTIES = {
        "number": _NUMBER,
        "body": {"type": "string", "maxLength": MAX_COMMENT_CHARACTERS},
    }
    REQUIRED = ("number", "body")
    EFFECT = ToolEffect.EXTERNAL
    CHECK = "read the pull request's or issue's latest comments"

    def __init__(self, *settings: Any, plain: bool) -> None:
        super().__init__(*settings)
        self._plain = plain

    async def _run(self, gh: GitHubClient, repository: str, args: dict[str, Any]) -> ToolResult:
        number = _number(args)
        body = _text(args, "body", MAX_COMMENT_CHARACTERS)
        if self._plain:
            body = plain_text(body, limit=MAX_COMMENT_CHARACTERS)
        made = await gh.post(f"/repos/{repository}/issues/{number}/comments", json={"body": body})
        return _common.answer_result(
            {
                "repository": repository,
                "number": number,
                "id": made.get("id"),
                "url": made.get("html_url"),
                "created_at": made.get("created_at"),
            }
        )


class _RerunFailedJobs(_GitHubTool):
    NAME = RERUN_FAILED_JOBS
    DESCRIPTION = (
        "Rerun the failed jobs of a GitHub Actions workflow run, by the run_id github_read_pull "
        "gives a failed check. It may need a person's approval first."
    )
    PROPERTIES = {
        "run_id": {"type": "integer", "minimum": 1},
        "head_sha": {
            "type": "string",
            "description": "The head commit the run is for; when given, a run of another "
            "commit isn't rerun.",
        },
    }
    REQUIRED = ("run_id",)
    EFFECT = ToolEffect.EXTERNAL
    CHECK = "read the run's attempts"

    async def _run(self, gh: GitHubClient, repository: str, args: dict[str, Any]) -> ToolResult:
        run_id = _number(args, "run_id")
        run = f"/repos/{repository}/actions/runs/{run_id}"
        head_sha = None if args.get("head_sha") is None else _sha(args)
        if head_sha is not None:
            ran_on = (await gh.get(run)).get("head_sha")
            if ran_on != head_sha:
                raise _Failed(
                    _common.error_result(
                        "github_conflict",
                        f"Workflow run {run_id} ran on {ran_on}, not on head_sha {head_sha}.",
                        "Read the pull request again for its current runs.",
                    )
                )
        await gh.post(f"{run}/rerun-failed-jobs")
        return _common.answer_result(
            {"repository": repository, "run_id": run_id, "head_sha": head_sha, "rerun": True}
        )


class _UpdateBranch(_GitHubTool):
    NAME = UPDATE_BRANCH
    DESCRIPTION = (
        "Merge a pull request's base branch into its head branch, when the head is still "
        "head_sha. GitHub does it shortly after accepting. It may need a person's approval first."
    )
    PROPERTIES = {"number": _NUMBER, "head_sha": _HEAD_SHA}
    REQUIRED = ("number", "head_sha")
    EFFECT = ToolEffect.EXTERNAL
    CHECK = "read the pull request's head"

    async def _run(self, gh: GitHubClient, repository: str, args: dict[str, Any]) -> ToolResult:
        number = _number(args)
        head_sha = _sha(args)
        await gh.put(
            f"/repos/{repository}/pulls/{number}/update-branch",
            json={"expected_head_sha": head_sha},
        )
        return _common.answer_result(
            {"repository": repository, "number": number, "head_sha": head_sha, "accepted": True}
        )


class _MergePull(_GitHubTool):
    NAME = MERGE_PULL
    DESCRIPTION = (
        "Merge a pull request, only if its head is still head_sha. merge_method defaults to "
        "a merge commit. It may need a person's approval first."
    )
    PROPERTIES = {
        "number": _NUMBER,
        "head_sha": _HEAD_SHA,
        "merge_method": {"type": "string", "enum": list(_MERGE_METHODS)},
    }
    REQUIRED = ("number", "head_sha")
    EFFECT = ToolEffect.EXTERNAL
    CHECK = "read whether the pull request was merged"

    async def _run(self, gh: GitHubClient, repository: str, args: dict[str, Any]) -> ToolResult:
        number = _number(args)
        asked: dict[str, Any] = {"sha": _sha(args)}
        method = args.get("merge_method")
        if method is not None:
            if method not in _MERGE_METHODS:
                raise _InvalidArguments(f"merge_method is one of {', '.join(_MERGE_METHODS)}.")
            asked["merge_method"] = method
        try:
            merged = await gh.put(f"/repos/{repository}/pulls/{number}/merge", json=asked)
        except GitHubError as error:
            if error.status != _NOT_MERGEABLE:
                raise
            raise _Failed(
                _common.error_result(
                    "github_not_mergeable",
                    error.message,
                    "Read the pull request: it may be a draft or closed, have conflicts, be "
                    "behind its base, or wait on required checks or reviews.",
                )
            ) from None
        return _common.answer_result(
            {
                "repository": repository,
                "number": number,
                "merged": merged.get("merged") is True,
                "sha": merged.get("sha"),
            }
        )
