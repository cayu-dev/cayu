"""What a GitHub call can fail with, and GitHub's own words for why."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from cayu.connections import ConnectionResponse

#: The longest error text kept from GitHub's answer.
MAX_ERROR_CHARACTERS = 1_000


class GitHubError(RuntimeError):
    """A GitHub call that failed: the request (``method``, ``path``), GitHub's ``status`` and
    ``message``, its ``x-github-request-id`` and the ``codes`` of its ``errors[]``.

    ``ambiguous`` is true when a write may have taken effect before it failed (a 5xx or a lost
    answer): don't resend it unless resending is safe.
    """

    def __init__(
        self,
        message: str,
        *,
        method: str,
        path: str,
        status: int | None = None,
        request_id: str | None = None,
        codes: tuple[str, ...] = (),
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.method = method
        self.path = path
        self.status = status
        self.request_id = request_id
        self.codes = codes
        self.ambiguous = ambiguous


class GitHubNotFound(GitHubError):
    """404, or a GraphQL ``repository`` that is null: missing, or not readable with this
    credential."""


class GitHubPermissionDenied(GitHubError):
    """401, or a 403 that isn't a rate limit. ``accepted_permissions`` is GitHub's
    ``x-accepted-github-permissions``: what the call needs, when GitHub said."""

    def __init__(self, message: str, *, accepted_permissions: str | None = None, **kwargs: Any):
        super().__init__(message, **kwargs)
        self.accepted_permissions = accepted_permissions


class GitHubRateLimited(GitHubError):
    """A primary rate limit (``secondary=False``) or a secondary one; ``retry_after`` is how
    many seconds to wait, when known."""

    def __init__(
        self, message: str, *, retry_after: float | None, secondary: bool, **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.secondary = secondary


class GitHubConflict(GitHubError):
    """409 or 422: GitHub refused the change as things stand, such as a merge or branch update
    whose expected head moved, or a label that already exists (see ``codes``)."""


class GitHubUnavailable(GitHubError):
    """A 5xx, or no answer from GitHub or the connection; ``ambiguous`` for a write."""


class GitHubWriteRefused(GitHubError):
    """A read-only client refused a write before sending it."""


class GitHubGraphQLFailed(GitHubError):
    """A GraphQL answer with errors the caller didn't accept, or with no ``data``; ``data`` is
    whatever GitHub did answer."""

    def __init__(
        self,
        message: str,
        *,
        errors: tuple[Any, ...],
        data: Mapping[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.errors = errors
        self.data = data


class GitHubConnectionRefused(GitHubError):
    """The slot or its host refused to send the call; ``code`` names why (``scope_not_allowed``,
    ``connection_unbound``, ...)."""

    def __init__(self, message: str, *, code: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.code = code


class GitHubAppNotInstalled(GitHubError):
    """The GitHub App isn't installed on ``repository``."""

    def __init__(self, message: str, *, repository: str, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.repository = repository


def response_error(
    response: ConnectionResponse, method: str, path: str, *, ambiguous: bool = False
) -> GitHubError:
    """The typed error for a GitHub answer that isn't a success. A rate limit is told apart
    by the caller, which knows how long to wait."""

    words, codes = github_words(response.body)
    request_id = response.header("x-github-request-id")
    message = error_text(f"GitHub answered {response.status} to {method} {path}", words, request_id)
    fields: dict[str, Any] = {
        "method": method,
        "path": path,
        "status": response.status,
        "request_id": request_id,
        "codes": codes,
    }
    status = response.status
    if status in (401, 403):
        return GitHubPermissionDenied(
            message,
            accepted_permissions=response.header("x-accepted-github-permissions"),
            **fields,
        )
    if status == 404:
        return GitHubNotFound(message, **fields)
    if status in (409, 422):
        return GitHubConflict(message, **fields)
    if status >= 500:
        return GitHubUnavailable(message, ambiguous=ambiguous, **fields)
    if 300 <= status < 400:
        moved = " (the repository may have been renamed or moved; call it by its new name)"
        return GitHubError(message + moved if status in (301, 307, 308) else message, **fields)
    return GitHubError(message, **fields)


def github_words(body: bytes) -> tuple[str, tuple[str, ...]]:
    """GitHub's ``message``, each ``errors[]`` entry's message or code and its
    ``documentation_url``, as one line; and the ``errors[]`` codes."""

    try:
        answer = json.loads(body)
    except ValueError:
        return "", ()
    if not isinstance(answer, dict):
        return "", ()
    parts = [answer["message"]] if isinstance(answer.get("message"), str) else []
    codes: list[str] = []
    errors = answer.get("errors")
    for entry in errors if isinstance(errors, list) else ():
        if isinstance(entry, str):
            parts.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        code = entry.get("code")
        if isinstance(code, str):
            codes.append(code)
        said = entry.get("message")
        if isinstance(said, str):
            parts.append(said)
        elif isinstance(code, str):
            named = ".".join(
                str(entry[key]) for key in ("resource", "field") if isinstance(entry.get(key), str)
            )
            parts.append(f"{named} {code}".strip())
    if isinstance(answer.get("documentation_url"), str):
        parts.append(f"see {answer['documentation_url']}")
    return "; ".join(parts), tuple(codes)


def error_text(prefix: str, words: str, request_id: str | None) -> str:
    """One error line, the request id before GitHub's words so a cut never loses it."""

    text = prefix + (f" (request {request_id})" if request_id else "")
    text += f": {words}" if words else "."
    if len(text) > MAX_ERROR_CHARACTERS:
        text = text[: MAX_ERROR_CHARACTERS - 1] + "…"
    return text
