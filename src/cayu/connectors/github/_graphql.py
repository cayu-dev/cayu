"""GraphQL answers: their errors, which of them a caller accepts, and mutations told apart from
queries."""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any

#: A block string, a string, or a comment: none of them can name the operation. A comment
#: and a string end at any line terminator, a lone carriage return included.
_NOT_SYNTAX = re.compile(r'"""(?:\\"""|[^"]|"(?!""))*"""|"(?:\\.|[^"\\\n\r])*"|#[^\n\r]*')
_MUTATION = re.compile(r"\bmutation\b")


@dataclass(frozen=True)
class GraphQLError:
    """One entry of a GraphQL answer's ``errors``: its ``type`` (``FORBIDDEN``,
    ``NOT_FOUND``, ...), when GitHub gave one, and where in ``data`` it applies."""

    type: str | None
    message: str
    path: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class GraphQLAnswer:
    """A GraphQL answer's ``data``, and the errors the caller accepted with it."""

    data: Mapping[str, Any]
    accepted: tuple[GraphQLError, ...] = ()


ErrorFilter = Callable[[GraphQLError], bool]


def forbidden_under(*fields: str) -> ErrorFilter:
    """Accept a ``FORBIDDEN`` error anywhere under one of ``fields`` (as the answer names
    them: an alias, where the query gives one), for example
    ``forbidden_under("statusCheckRollup")`` for a check run this credential can't read, which
    otherwise fails the whole answer."""

    if not fields:
        raise ValueError("forbidden_under names at least one field.")
    named = frozenset(fields)
    return lambda error: error.type == "FORBIDDEN" and any(step in named for step in error.path)


def errors_under(*path_prefix: str | int, types: Collection[str] | None = None) -> ErrorFilter:
    """Accept an error whose path starts with ``path_prefix`` (an alias, a field, an index),
    of one of ``types`` when they are given."""

    if not path_prefix:
        raise ValueError("errors_under names a path prefix.")
    prefix = tuple(path_prefix)
    wanted = None if types is None else frozenset(types)
    return lambda error: (
        error.path[: len(prefix)] == prefix and (wanted is None or error.type in wanted)
    )


def is_mutation(document: str) -> bool:
    """Whether a GraphQL document has a mutation: its strings and comments are ignored, and any
    other ``mutation`` word counts, so a read is never taken for a write."""

    return _MUTATION.search(_NOT_SYNTAX.sub(" ", document)) is not None


def graphql_errors(errors: Any) -> tuple[GraphQLError, ...]:
    if not isinstance(errors, list):
        return ()
    parsed = []
    for entry in errors:
        if not isinstance(entry, dict):
            parsed.append(GraphQLError(None, str(entry)))
            continue
        extensions = entry.get("extensions")
        kind = entry.get("type")
        if kind is None and isinstance(extensions, dict):
            kind = extensions.get("type", extensions.get("code"))
        path = entry.get("path")
        parsed.append(
            GraphQLError(
                kind if isinstance(kind, str) else None,
                str(entry.get("message", "")),
                tuple(step for step in path if isinstance(step, str | int))
                if isinstance(path, list)
                else (),
            )
        )
    return tuple(parsed)


def repository_missing(data: Mapping[str, Any] | None, errors: tuple[GraphQLError, ...]) -> bool:
    """Whether the answer's ``repository`` is null and the errors left, of which there is at
    least one, are all about that: the repository is missing or this credential can't read it.
    A caller that accepted those errors reads the null itself."""

    return (
        data is not None
        and "repository" in data
        and data["repository"] is None
        and bool(errors)
        and all(error.type == "NOT_FOUND" and error.path[:1] == ("repository",) for error in errors)
    )
