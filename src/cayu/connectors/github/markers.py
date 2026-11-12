"""Writes that happen once however often a task restarts: a comment or an issue ending with a
hidden marker, looked up first among what the caller's own login wrote."""

from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from cayu.connectors.github._errors import (
    GitHubError,
    GitHubLookupIncomplete,
    GitHubNotFound,
    GitHubWriteRefused,
)
from cayu.connectors.github.client import DEFAULT_MAX_PAGES, GitHubClient
from cayu.connectors.github.pulls import PullComment, _rest_comment

#: The longest body a marked write sends, in UTF-8 bytes: a little under the 262,144 GitHub
#: takes (its refusal speaks of 65,536 characters, but it counts bytes).
MAX_BODY_BYTES = 262_000
#: The waits before each look for a write whose answer was lost: about 20 seconds in all, past
#: the 15 seconds GitHub's issues listing has been seen to lag a just-filed issue.
_RELOOKUP_WAITS_SECONDS = (1.0, 2.0, 4.0, 8.0, 5.0)
#: The writes one client remembers, newest kept, so it finds them again before its listings do.
_REMEMBERED_WRITES = 1_024
#: What a direct read of a write answers once it was deleted (410) or moved (a redirect).
_GONE = frozenset({301, 302, 307, 308, 410})
#: What ends a GitHub App's login, and no user's: an App connection's markers are trusted only
#: under such a login.
_BOT_SUFFIX = "[bot]"
#: The format's name and version, as each marker starts.
_FORMAT = "cayu-marker"
_VERSION = "v1"
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/=+-]{0,199}")
#: The most characters a marker's encoded data may take.
_MAX_DATA_CHARACTERS = 4_000
#: The trailing marker of a body; a reader skips any version but its own.
_TRAILING = re.compile(r"<!-- cayu-marker (v[0-9]+) (\S+)(?: (\{[^\n]*\}))? -->\Z")
#: A JSON string as ``json.dumps`` writes it.
_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
#: What a string in a marker's data never holds as itself: with these escaped, nothing in the
#: data can end the HTML comment (``-->``, ``--!>``) or put ``--`` in it.
_ESCAPED = str.maketrans({character: f"\\u{ord(character):04x}" for character in "-<>!&"})
_CUT_NOTE = "\n\n(cut short: too long for one comment)"
_ELLIPSIS = "…"
#: Follows each ``@`` of model text: GitHub links no mention through it, and it shows nothing.
_ZERO_WIDTH_SPACE = "\u200b"
_sleep = asyncio.sleep


class WriteOutcome(StrEnum):
    CREATED = "created"
    FOUND = "found"
    UPDATED = "updated"
    #: A read-only client refused the write; it is in the client's ``refused``.
    REFUSED = "refused"


@dataclass(frozen=True)
class Marker:
    """What must happen once (``key``), and data the writer keeps with it.

    ``key`` is letters, digits and ``. _ : / = + -``, at most 200 characters, with no ``--``
    and not ending in ``-``; prefix it with the application's name by convention. ``data`` is
    a JSON object, at most 4,000 characters encoded.
    """

    key: str
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _KEY.fullmatch(self.key) or "--" in self.key or self.key.endswith("-"):
            raise ValueError(f"Not a marker key: {self.key!r}.")
        if len(self._encoded_data()) > _MAX_DATA_CHARACTERS:
            raise ValueError(f"The data of marker {self.key} is over 4,000 characters.")

    @property
    def text(self) -> str:
        """The hidden HTML comment: ``<!-- cayu-marker v1 KEY {data} -->``."""

        data = self._encoded_data()
        return f"<!-- {_FORMAT} {_VERSION} {self.key}{f' {data}' if data else ''} -->"

    def _encoded_data(self) -> str:
        if not self.data:
            return ""
        try:
            encoded = json.dumps(
                dict(self.data), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"The data of marker {self.key} isn't JSON: {error}") from None
        return _JSON_STRING.sub(lambda found: found.group(0).translate(_ESCAPED), encoded)


@dataclass(frozen=True)
class MarkedComment:
    """A comment the trusted login wrote, and its marker."""

    marker: Marker
    comment: PullComment


@dataclass(frozen=True)
class WrittenOnce:
    """What a write-once call did. ``id`` and ``url`` are the comment's or issue's (None when
    ``REFUSED``); ``number`` is the issue's. ``duplicates`` are later writes with the same
    marker by the same login, from writers that raced: comments' ids, issues' numbers. They
    are reported, never deleted: that would be a second write, which a read-only or narrowly
    bound caller can't make. ``labelled`` is false when the issue lacks one of the ``labels``
    asked for: GitHub drops labels the writer can't set, and people remove them."""

    outcome: WriteOutcome
    id: int | None
    url: str | None
    number: int | None = None
    duplicates: tuple[int, ...] = ()
    labelled: bool = True


#: Reads an application's earlier marker format from a body, while it moves to this one.
LegacyReader = Callable[[str], Marker | None]
#: A write-once lookup: the matching items, earliest first, and whether the listing was whole.
_Lookup = Callable[[], Awaitable[tuple[list[Mapping[str, Any]], bool]]]
#: What a write-once call writes: the repository, the issue's number (None for a new issue),
#: and the marker's key.
_Target = tuple[str, int | None, str]


def with_marker(body: str, marker: Marker) -> str:
    """``body`` ending with ``marker`` on its own line, within ``MAX_BODY_BYTES`` of UTF-8: an
    over-long body is cut with a note, never inside a character, and the marker still ends
    it."""

    suffix = f"\n\n{marker.text}"
    body = body.rstrip()
    room = MAX_BODY_BYTES - len(suffix.encode())
    encoded = body.encode()
    if len(encoded) > room:
        cut = encoded[: room - len(_CUT_NOTE.encode())].decode("utf-8", "ignore")
        body = cut.rstrip() + _CUT_NOTE
    return body + suffix


def marker_of(body: str) -> Marker | None:
    """The marker a body ends with; None when it ends with none, or with another version's.
    Marker text earlier in the body is never read: a writer's marker is always last. Trailing
    whitespace, such as the ``\\r\\n`` of a body edited on GitHub, is ignored."""

    found = _TRAILING.search(body.rstrip())
    if found is None or found.group(1) != _VERSION:
        return None
    try:
        data = json.loads(found.group(3)) if found.group(3) else {}
        return Marker(found.group(2), data) if isinstance(data, dict) else None
    except ValueError:
        return None


def trusted_markers(
    comments: Iterable[PullComment], *, login: str, legacy: LegacyReader | None = None
) -> tuple[MarkedComment, ...]:
    """The markers of the comments ``login`` wrote, in the comments' order; any other author's
    are never read. ``legacy`` reads a body that ends with no marker of this format.

    Trust is the comment's author: someone who can edit others' comments on the repository (a
    maintainer) can change or remove a marker in one, and what they leave is read as written.
    Take ``login`` from ``await gh.login()``, never by hand: an App's is ``<slug>[bot]``, which
    no user can be named, while a user may be named the slug alone.
    """

    trusted = _login(login)
    found = []
    for comment in comments:
        if comment.author.casefold() != trusted:
            continue
        marker = _read(comment.body, legacy)
        if marker is not None:
            found.append(MarkedComment(marker, comment))
    return tuple(found)


def plain_text(text: str, *, limit: int | None = None) -> str:
    """Text a model wrote, safer to post: every ``@`` is followed by a zero-width space, so it
    mentions no one (a mention can start a bot's command) while an address or ``@dataclass``
    still reads as written; the HTML comment delimiters ``<!--`` and ``-->`` are removed, so
    nothing is hidden and what a comment held shows as text; it is cut to ``limit`` characters
    with ``…``. Other HTML, such as ``<details>``, is left as it is."""

    if limit is not None and limit < 1:
        raise ValueError("limit is at least 1.")
    cleaned = text
    while "<!--" in cleaned or "-->" in cleaned:
        cleaned = cleaned.replace("<!--", "").replace("-->", "")
    cleaned = cleaned.strip().replace("@", "@" + _ZERO_WIDTH_SPACE)
    if limit is None or len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + _ELLIPSIS


async def comment_once(
    gh: GitHubClient,
    repository: str,
    number: int,
    body: str,
    marker: Marker,
    *,
    login: str,
    known: int | None = None,
    update: bool = False,
    since: datetime | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
    legacy: LegacyReader | None = None,
) -> WrittenOnce:
    """Comment on issue or pull request ``number`` once: ``body`` ending with ``marker``.

    The comment is looked for first: ``login``'s comments in the thread's listing, with
    ``known`` (the id an earlier call answered, which the caller kept) and the one this client
    last wrote read directly, as the listing may not show them yet. Of all those with
    ``marker.key``, the earliest is the one. Found, it is left (``FOUND``), or with
    ``update`` edited in place when its body differs (``UPDATED``); else the comment is posted
    (``CREATED``). The others, from writers that raced, are reported in ``duplicates`` and
    left. A listing that
    stopped at ``max_pages`` without finding it raises ``GitHubLookupIncomplete``. A post
    whose answer was lost is looked for again for about 20 seconds, never resent.

    ``since`` reads only comments updated since then, so it is safe only when the marked
    comment can't be older, as for a key naming its own day: an older one, never edited, isn't
    read, and the comment is posted again.
    """

    trusted = await _trusted_login(gh, login)
    path = f"/repos/{repository}/issues/{number}/comments"
    text = with_marker(body, marker)
    query = {} if since is None else {"since": _iso(since)}
    target: _Target = (repository.lower(), number, marker.key)

    async def lookup() -> tuple[list[Mapping[str, Any]], bool]:
        page = await gh.list(path, query=query, max_pages=max_pages)
        return _matching(page.items, marker, trusted, legacy), page.complete

    def on_thread(item: Mapping[str, Any]) -> bool:
        return str(item.get("issue_url") or "").endswith(f"/issues/{number}")

    async with _one_writer(gh, target):
        recalled = await _recalled(
            gh,
            target,
            known,
            lambda comment_id: f"/repos/{repository}/issues/comments/{comment_id}",
            on_thread,
            marker,
            trusted,
            legacy,
        )
        listed, complete = await lookup()
        found = _merged(listed, recalled)
        if found:
            first, duplicates = found[0], _ids(found[1:], "id")
            _remember(gh, target, first["id"])
            if not update or (first.get("body") or "").rstrip() == text:
                return _written(WriteOutcome.FOUND, first, duplicates)
            comment_path = f"/repos/{repository}/issues/comments/{first['id']}"
            edited = await _write(gh.patch(comment_path, json={"body": text}), lookup=None)
            if edited is None:
                return WrittenOnce(WriteOutcome.REFUSED, None, None, duplicates=duplicates)
            return _written(WriteOutcome.UPDATED, edited, duplicates)
        _check_complete(complete, path, marker)
        written = await _write(gh.post(path, json={"body": text}), lookup=lookup)
        if written is None:
            return WrittenOnce(WriteOutcome.REFUSED, None, None)
        _remember(gh, target, written["id"])
    return _written(WriteOutcome.CREATED, written, ())


async def file_issue_once(
    gh: GitHubClient,
    repository: str,
    title: str,
    body: str,
    marker: Marker,
    *,
    login: str,
    labels: Iterable[str] = (),
    known: int | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
    legacy: LegacyReader | None = None,
) -> WrittenOnce:
    """File an issue once: ``body`` ending with ``marker``, labelled ``labels``.

    The issue is looked for first: the issues ``login`` opened, open or closed, through the
    issues listing filtered by creator (not the search API), with ``known`` (the number an
    earlier call answered, which the caller kept) and the one this client last filed read
    directly, as the listing may not show them yet; one deleted or moved is skipped. Of all
    those with ``marker.key``, the earliest is the one (``FOUND``), the others reported in
    ``duplicates``; else the issue is filed (``CREATED``). Labels only tag a new issue: the lookup doesn't depend on them,
    and ``labelled`` says whether the issue has them all. A listing that stopped at
    ``max_pages`` without finding it raises ``GitHubLookupIncomplete``. A filing whose answer
    was lost is looked for again for about 20 seconds, never resent.

    GitHub's listing shows a just-filed issue only seconds later (3 to 15 have been seen): a
    process that may file the same issue again that soon after another process filed it
    keeps the ``number`` it was answered and passes it as ``known``.
    """

    trusted = await _trusted_login(gh, login)
    labels = tuple(labels)
    path = f"/repos/{repository}/issues"
    target: _Target = (repository.lower(), None, marker.key)
    # GitHub filters by creator, a bot's login included; the author is checked again here.
    query = {"state": "all", "creator": login.strip(), "sort": "created", "direction": "asc"}

    async def lookup() -> tuple[list[Mapping[str, Any]], bool]:
        page = await gh.list(path, query=query, max_pages=max_pages)
        issues = [item for item in page.items if _is_issue(item)]
        return _matching(issues, marker, trusted, legacy), page.complete

    def in_repository(issue: Mapping[str, Any]) -> bool:
        """An issue still here: not a pull request, nor one moved to another repository."""

        place = f"/{repository}/issues/{issue.get('number')}".lower()
        return _is_issue(issue) and str(issue.get("html_url") or "").lower().endswith(place)

    def kept_all(issue: Mapping[str, Any]) -> bool:
        kept = {label.get("name") for label in issue.get("labels") or [] if isinstance(label, dict)}
        return kept.issuperset(labels)

    async with _one_writer(gh, target):
        recalled = await _recalled(
            gh,
            target,
            known,
            lambda issue_number: f"/repos/{repository}/issues/{issue_number}",
            in_repository,
            marker,
            trusted,
            legacy,
        )
        listed, complete = await lookup()
        found = _merged(listed, recalled)
        if found:
            first, duplicates = found[0], _ids(found[1:], "number")
            _remember(gh, target, first["number"])
            return _written(WriteOutcome.FOUND, first, duplicates, labelled=kept_all(first))
        _check_complete(complete, path, marker)
        issue: dict[str, object] = {"title": title, "body": with_marker(body, marker)}
        if labels:
            issue["labels"] = list(labels)
        written = await _write(gh.post(path, json=issue), lookup=lookup)
        if written is None:
            return WrittenOnce(WriteOutcome.REFUSED, None, None)
        _remember(gh, target, written["number"])
    return _written(WriteOutcome.CREATED, written, (), labelled=kept_all(written))


@asynccontextmanager
async def _one_writer(gh: GitHubClient, target: _Target) -> AsyncIterator[None]:
    """This client's turn to write ``target`` once: its coroutines take turns, so they never
    race each other. Other processes, and other clients, can still race it."""

    locks = gh._write_once.get()
    entry = locks.setdefault(target, [asyncio.Lock(), 0])
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if not entry[1]:
            del locks[target]


async def _recalled(
    gh: GitHubClient,
    target: _Target,
    known: int | None,
    path_of: Callable[[int], str],
    belongs: Callable[[Mapping[str, Any]], bool],
    marker: Marker,
    trusted: str,
    legacy: LegacyReader | None,
) -> list[Mapping[str, Any]]:
    """The writes ``known`` names and this client remembers for ``target``, read directly
    (GitHub answers a direct read at once, while its listings lag): those still there, where
    ``belongs`` says, by ``trusted``, with ``marker.key``. One deleted (404, 410), moved
    (a redirect) or no longer this write is skipped, and forgotten."""

    remembered = gh._written_once.get(target)
    recalled = []
    for candidate in dict.fromkeys(item for item in (known, remembered) if item is not None):
        try:
            item = await gh.get(path_of(candidate))
        except GitHubError as error:
            if not isinstance(error, GitHubNotFound) and error.status not in _GONE:
                raise
            item = None
        if isinstance(item, dict) and belongs(item) and _matching([item], marker, trusted, legacy):
            recalled.append(item)
        elif candidate == remembered:
            gh._written_once.pop(target, None)
    return recalled


def _merged(
    listed: list[Mapping[str, Any]], recalled: list[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """The listing's matches with the recalled writes it doesn't show yet, earliest first."""

    merged = {_order(item): item for item in [*recalled, *listed]}
    return [merged[key] for key in sorted(merged)]


def _order(item: Mapping[str, Any]) -> tuple[int, int]:
    """Where a write comes among others: issues by number, comments by id."""

    return item.get("number") or 0, item.get("id") or 0


def _remember(gh: GitHubClient, target: _Target, written: int) -> None:
    remembered: OrderedDict[_Target, int] = gh._written_once
    remembered[target] = written
    remembered.move_to_end(target)
    while len(remembered) > _REMEMBERED_WRITES:
        remembered.popitem(last=False)


async def _write(sending: Awaitable[Any], *, lookup: _Lookup | None) -> Mapping[str, Any] | None:
    """The written item; None when a read-only client refused it. A write whose answer was
    lost (``ambiguous``) is looked for after each of ``_RELOOKUP_WAITS_SECONDS``, as GitHub's
    listings show a new item seconds late: found, it is the written item; never found, the
    write's own error stands, and it is never resent."""

    try:
        return await sending
    except GitHubWriteRefused:
        return None
    except GitHubError as error:
        if not error.ambiguous or lookup is None:
            raise
        for wait in _RELOOKUP_WAITS_SECONDS:
            await _sleep(wait)
            try:
                found, _ = await lookup()
            except GitHubError:
                raise error from None
            if found:
                return found[0]
        raise


def _is_issue(item: Mapping[str, Any]) -> bool:
    return "pull_request" not in item


def _matching(
    items: Iterable[Any], marker: Marker, trusted: str, legacy: LegacyReader | None
) -> list[Mapping[str, Any]]:
    """The items ``trusted`` wrote whose marker has ``marker.key``, earliest first: issues by
    number, comments by id."""

    found = [
        item
        for item in items
        if isinstance(item, dict)
        and (marked := trusted_markers([_rest_comment(item)], login=trusted, legacy=legacy))
        and marked[0].marker.key == marker.key
    ]
    return sorted(found, key=_order)


def _read(body: str, legacy: LegacyReader | None) -> Marker | None:
    marker = marker_of(body)
    if marker is None and legacy is not None:
        return legacy(body)
    return marker


def _check_complete(complete: bool, path: str, marker: Marker) -> None:
    if not complete:
        raise GitHubLookupIncomplete(
            f"The lookup for marker {marker.key} read {path} up to its page limit without "
            "finding it; nothing was written. Raise max_pages, or pass the id or number an "
            "earlier call answered as known=.",
            method="GET",
            path=path,
        )


def _written(
    outcome: WriteOutcome,
    item: Mapping[str, Any],
    duplicates: tuple[int, ...],
    *,
    labelled: bool = True,
) -> WrittenOnce:
    number = item.get("number")
    return WrittenOnce(
        outcome,
        item.get("id"),
        item.get("html_url"),
        number=number if isinstance(number, int) else None,
        duplicates=duplicates,
        labelled=labelled,
    )


def _ids(items: Iterable[Mapping[str, Any]], name: str) -> tuple[int, ...]:
    return tuple(item[name] for item in items if isinstance(item.get(name), int))


def _login(login: str) -> str:
    if not login.strip():
        raise ValueError("login is the GitHub login whose markers are trusted.")
    return login.strip().casefold()


async def _trusted_login(gh: GitHubClient, login: str) -> str:
    """``login``, checked against what ``gh`` writes as: on a GitHub App's slot, exactly the
    App's own ``<slug>[bot]``, never a user's bare slug nor another App's bot."""

    trusted = _login(login)
    if await gh._writes_as_app() and (
        not trusted.endswith(_BOT_SUFFIX) or trusted != _login(await gh.login())
    ):
        raise ValueError(
            f"This GitHub App slot writes as its own <slug>[bot], not {login!r}: take "
            "login from `await gh.login()`."
        )
    return trusted


def _iso(moment: datetime) -> str:
    if moment.tzinfo is None:
        raise ValueError("since is a timezone-aware time.")
    return moment.isoformat().replace("+00:00", "Z")
