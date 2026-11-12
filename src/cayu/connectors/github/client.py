"""GitHub's REST and GraphQL APIs from code, through a connection slot."""

from __future__ import annotations

import asyncio
import copy
import json as json_module
import re
import time
import weakref
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar
from urllib.parse import parse_qs, quote, urlsplit

import httpx

from cayu.connections import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_BODY_BYTES,
    CloudConnectionBroker,
    ConnectionCallError,
    ConnectionClient,
    ConnectionResponse,
    Query,
    retry_after_seconds,
)
from cayu.connections.client import kept_open
from cayu.connectors.github._errors import (
    GitHubAppNotInstalled,
    GitHubConnectionRefused,
    GitHubError,
    GitHubGraphQLFailed,
    GitHubNotFound,
    GitHubPermissionDenied,
    GitHubRateLimited,
    GitHubUnavailable,
    GitHubWriteRefused,
    error_text,
    github_words,
    response_error,
)
from cayu.connectors.github._graphql import (
    ErrorFilter,
    GraphQLAnswer,
    graphql_errors,
    is_mutation,
    repository_missing,
)

DEFAULT_SLOT = "github"
#: GitHub's REST API on github.com; a GitHub Enterprise Server's is ``https://HOST/api/v3``
#: (its GraphQL API, at ``/api/graphql``, isn't reachable through such a slot yet).
API_URL = "https://api.github.com"
#: The repository (``owner/name``) a call is about: the slot's host picks the credential for
#: it and drops the header, so GitHub never sees it. ``app_connections`` and
#: ``token_connections`` drop it; Cayu Cloud's GitHub connector does too.
REPOSITORY_HEADER = "x-cayu-github-repository"
API_VERSION = "2022-11-28"
JSON_MEDIA_TYPE = "application/vnd.github+json"
#: What ``text()`` asks for: a file's raw content; a log's redirect is the same either way.
RAW_MEDIA_TYPE = "application/vnd.github.raw+json"
GRAPHQL_PATH = "/graphql"
RATE_LIMIT_PATH = "/rate_limit"
#: Where a token's owner is read: the login a token connection writes as.
_USER_PATH = "/user"
#: GitHub's words refusing ``GET /user`` to an installation token, which is no user's.
_NO_USER = "Resource not accessible by integration"
#: Cayu Cloud's broker route listing the Agent's slots, which a failed listing names.
_LISTING_PATH = "/v1/agent-connections"
#: Cayu Cloud's connector for a GitHub App, whose slots write as the App's ``<slug>[bot]``.
_APP_CONNECTOR = "github-app"
DEFAULT_PAGE_SIZE = 100
DEFAULT_MAX_PAGES = 50
#: Calls one client has in flight at once, under the broker's eight per Agent.
MAX_CONCURRENT_CALLS = 6
#: The most of a downloaded log ``text()`` keeps: its end.
MAX_TEXT_BYTES = MAX_BODY_BYTES
#: Where GitHub redirects a log read; ``text()`` follows a redirect only to these hosts.
LOG_HOSTS = re.compile(
    r"(?:[a-z0-9-]+\.)*actions\.githubusercontent\.com|productionresultssa\d+\.blob\.core\.windows\.net"
)

_READS = frozenset({"GET", "HEAD"})
_REPOSITORY = re.compile(r"[A-Za-z0-9-]{1,39}/[A-Za-z0-9._-]{1,100}")
#: What stays as given in a path: separators, sub-delimiters and existing escapes.
_PATH_SAFE = "/%:@!$&'()*+,;=-._~"
#: A ``%`` that doesn't start an escape: a literal one, encoded as ``%25``.
_LITERAL_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")
_LINK = re.compile(r'<([^>]*)>\s*;\s*rel="([^"]*)"')
#: GitHub asks for at least a minute's wait after a limit that names no time.
_LIMIT_WAIT_SECONDS = 60.0
#: The wait after the broker's own ``too_many_calls`` when it names no time.
_BUSY_SECONDS = 1.0
_BACKOFF_SECONDS = 0.5
#: Retried for reads: GitHub, or the slot's credential for the call, couldn't be reached.
_TRANSIENT_CODES = frozenset({"upstream_failed", "credential_unavailable"})
#: What Cayu Cloud's broker says in its ``credential_unavailable`` refusal of a call to a
#: repository the GitHub App isn't installed on.
_NOT_INSTALLED = "isn't installed on"
_sleep = asyncio.sleep
_Bound = TypeVar("_Bound")
#: A page's items (``GitHubClient.list`` hides the builtin in the class's scope).
_Items = list[Any]


class _LoopBound(Generic[_Bound]):
    """An object of the running event loop, such as a lock, made anew when the loop changes:
    a client outlives one ``asyncio.run`` and never keeps an old loop alive. One loop at a time
    uses it; a client isn't shared between threads running their own loops."""

    def __init__(self, make: Callable[[], _Bound]) -> None:
        self._make = make
        self._loop: weakref.ref[asyncio.AbstractEventLoop] | None = None
        self._value: _Bound | None = None

    def get(self) -> _Bound:
        loop = asyncio.get_running_loop()
        if self._value is None or self._loop is None or self._loop() is not loop:
            self._loop = weakref.ref(loop)
            self._value = self._make()
        return self._value

    def current(self) -> _Bound | None:
        """The object of the loop that last used it, if any."""

        return self._value


class GitHubResponse(ConnectionResponse):
    """GitHub's answer to ``GitHubClient.request``."""

    def json(self) -> Any:
        """The body as JSON; None when it is empty."""

        return json_module.loads(self.body) if self.body else None


@dataclass(frozen=True)
class Page:
    """A listing's items, and whether they are all of it: ``complete`` is false when
    ``max_pages`` or ``keep_going`` stopped it before the last page."""

    items: tuple[Any, ...]
    complete: bool


@dataclass(frozen=True)
class RefusedWrite:
    """A write a read-only client refused: its method, path, and the repository it was for."""

    method: str
    path: str
    repository: str | None = None


class _LogDownloadFailed(GitHubUnavailable):
    """The direct download of a redirected log failed: the log's host didn't answer or refused
    it, such as where this process can't reach GitHub's log storage."""


class GitHubClient:
    """GitHub through ``slot`` of ``client`` (default: Cayu Cloud's broker).

    ``api_url`` is the slot's API root, against which next-page links are checked.
    ``read_only=True`` refuses every write before sending it and records it in ``refused``.
    A rate limit is waited out and the call resent once when the wait is at most
    ``max_rate_limit_wait_seconds``; a read that fails with a 5xx or no answer is retried up to
    ``read_retries`` times. A write is sent again only after GitHub refused it unexecuted (that
    rate limit, or a 401 an App slot renews its token for), at most once for each; one that
    fails after it may have reached GitHub is raised with ``ambiguous`` set. ``transport`` is for the one direct
    download ``text()`` makes when GitHub redirects a log read; by default it is the connection
    client's own ``download_transport``, when it has one, as a test's fake GitHub does.
    """

    def __init__(
        self,
        client: ConnectionClient | None = None,
        slot: str = DEFAULT_SLOT,
        *,
        api_url: str = API_URL,
        read_only: bool = False,
        max_rate_limit_wait_seconds: float = 60.0,
        read_retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        resolved = client if client is not None else CloudConnectionBroker.from_environment()
        if resolved is None:
            raise ValueError(
                "This process has no connection client: outside Cayu Cloud, give one, such as "
                "github.connections_from_environment(permissions=...)."
            )
        api = urlsplit(api_url)
        if api.scheme not in ("https", "http") or not api.hostname or api.query:
            raise ValueError("api_url is an http(s) URL with no query.")
        if read_retries < 0 or max_rate_limit_wait_seconds < 0:
            raise ValueError("read_retries and max_rate_limit_wait_seconds can't be negative.")
        self._client = resolved
        self._slot = slot
        self._api = (api.scheme, api.netloc.lower(), api.path.rstrip("/"))
        self._read_only = read_only
        self._max_wait = max_rate_limit_wait_seconds
        self._read_retries = read_retries
        offered = getattr(resolved, "download_transport", None)
        if transport is None and isinstance(offered, httpx.AsyncBaseTransport):
            transport = offered
        self._transport = kept_open(transport)
        self._cap = _LoopBound(lambda: asyncio.Semaphore(MAX_CONCURRENT_CALLS))
        #: The write-once helpers' locks, by what they write once, so this client's own
        #: coroutines take turns; shared with a ``read_only_view``.
        self._write_once: _LoopBound[dict[Any, Any]] = _LoopBound(dict)
        #: What the write-once helpers last wrote with this client, by what they wrote once.
        self._written_once: OrderedDict[Any, int] = OrderedDict()
        self._refused: list[RefusedWrite] = []
        #: What this client learns of its slot once (``login``, and ``listed``: the slot as
        #: Cayu Cloud's broker lists it), shared with a ``read_only_view``; one read at a time.
        self._known: dict[str, Any] = {}
        self._learning = _LoopBound(asyncio.Lock)

    def __repr__(self) -> str:
        return f"GitHubClient(slot={self._slot!r}, read_only={self._read_only})"

    @property
    def slot(self) -> str:
        return self._slot

    @property
    def read_only(self) -> bool:
        return self._read_only

    @property
    def refused(self) -> tuple[RefusedWrite, ...]:
        """Every write this read-only client refused, oldest first, for a dry run's report."""

        return tuple(self._refused)

    async def login(self) -> str:
        """The login this client's writes are by: the ``login`` to trust in ``comment_once``,
        ``file_issue_once`` and ``trusted_markers``.

        On Cayu Cloud it is the account of the slot's connection, as the broker lists it: a
        GitHub App's ``<slug>[bot]``. Off it, a connection client that knows its own login
        answers it (``GitHubAppConnections.login()``, a fake's ``login``), and any other is a
        token's, whose owner ``GET /user`` names. Read once per client and its read-only views:
        after the slot is bound to another connection, a new client reads the new login.
        """

        login = self._known.get("login")
        if login is None:
            if isinstance(self._client, CloudConnectionBroker):
                login = _account_login(await self._slot_listed(self._client), self._slot)
            else:
                async with self._learning.get():
                    login = self._known.get("login") or await self._own_login()
            self._known["login"] = login
        return login

    def read_only_view(self) -> GitHubClient:
        """The same slot and settings, refusing writes, with its own ``refused``; both share the
        cap on calls in flight."""

        view = copy.copy(self)
        view._read_only = True
        view._refused = []
        return view

    async def request(
        self,
        method: str,
        path: str,
        *,
        query: Query | None = None,
        json: object = None,
        accept: str = JSON_MEDIA_TYPE,
        redirect: Literal["refuse", "return"] = "refuse",
    ) -> GitHubResponse:
        """One REST call; GitHub's answer when it is a success (or a redirect, with
        ``redirect="return"``), else the typed ``GitHubError``.

        A path under ``/repos/{owner}/{name}/`` names its repository to the slot; no other path
        names one. An App slot needs one for every call, so ``/rate_limit`` isn't available
        through this client on an App slot yet: it raises ``GitHubConnectionRefused``.
        """

        method = method.upper()
        path = _encoded(path)
        repository = _repository_in(path)
        if method not in _READS:
            self._check_write(method, path, repository)
        headers = {"accept": accept}
        if repository is not None:
            headers[REPOSITORY_HEADER] = repository
        body = b""
        if json is not None:
            body = json_module.dumps(json, ensure_ascii=False).encode()
            headers["content-type"] = "application/json"
        response = await self._send(
            method, path, query, headers, body, retry_failures=method in _READS
        )
        if 200 <= response.status < 300 or (redirect == "return" and 300 <= response.status < 400):
            return response
        raise response_error(response, method, path)

    async def get(self, path: str, *, query: Query | None = None) -> Any:
        return (await self.request("GET", path, query=query)).json()

    async def post(self, path: str, *, json: object = None) -> Any:
        return (await self.request("POST", path, json=json)).json()

    async def put(self, path: str, *, json: object = None) -> Any:
        return (await self.request("PUT", path, json=json)).json()

    async def patch(self, path: str, *, json: object = None) -> Any:
        return (await self.request("PATCH", path, json=json)).json()

    async def delete(self, path: str, *, json: object = None) -> Any:
        return (await self.request("DELETE", path, json=json)).json()

    async def text(
        self, path: str, *, query: Query | None = None, accept: str = RAW_MEDIA_TYPE
    ) -> str:
        """A text answer, such as a file's content or a job's log. GitHub answers a log read
        with a redirect to a signed URL: it is downloaded directly, with no credential, only
        from ``LOG_HOSTS``, keeping its last ``MAX_TEXT_BYTES``."""

        response = await self.request("GET", path, query=query, accept=accept, redirect="return")
        if response.status < 300:
            return response.body.decode("utf-8", "replace")
        location = response.header("location") or ""
        target = urlsplit(location)
        if target.scheme != "https" or not LOG_HOSTS.fullmatch(target.hostname or ""):
            raise GitHubError(
                f"GitHub redirected GET {path} away from its log hosts; not followed.",
                method="GET",
                path=path,
                status=response.status,
                request_id=response.header("x-github-request-id"),
            )
        return await self._download(location, path)

    def paginate(
        self,
        path: str,
        *,
        query: Query | None = None,
        item_key: str | None = None,
        max_pages: int | None = None,
    ) -> AsyncIterator[Any]:
        """Each item of a listing, following GitHub's next-page links: the answer is a list,
        or an object whose ``item_key`` is one. ``max_pages`` stops early without saying so;
        ``list()`` says."""

        return self._items(path, query, item_key, max_pages)

    async def list(
        self,
        path: str,
        *,
        query: Query | None = None,
        item_key: str | None = None,
        max_pages: int = DEFAULT_MAX_PAGES,
        keep_going: Callable[[_Items], bool] | None = None,
    ) -> Page:
        """A listing's items, up to ``max_pages`` pages; ``keep_going`` sees each page's items
        and stops after a page it answers false for. ``complete`` says whether every page was
        read."""

        if max_pages < 1:
            raise ValueError("max_pages is at least 1.")
        items: _Items = []
        pages = 0
        async with aclosing(self._pages(path, query, item_key)) as listing:
            async for page, more in listing:
                items.extend(page)
                pages += 1
                if not more:
                    return Page(tuple(items), True)
                if pages >= max_pages or (keep_going is not None and not keep_going(page)):
                    return Page(tuple(items), False)
        return Page(tuple(items), True)

    async def graphql(
        self,
        query: str,
        variables: Mapping[str, object] | None = None,
        *,
        repository: str,
        accept: ErrorFilter | None = None,
    ) -> GraphQLAnswer:
        """A GraphQL query or mutation about ``repository`` (``owner/name``).

        Any error in the answer fails it (``GitHubGraphQLFailed``, with GitHub's ``data`` and
        ``errors``) unless ``accept`` takes it: accepted errors come back in
        ``GraphQLAnswer.accepted`` with the ``data``, so a caller can read the rest of a
        partial answer. A null ``repository`` raises ``GitHubNotFound``, unless ``accept``
        took its error. An answer whose errors are a rate limit (``RATE_LIMITED``, in a 200) is
        waited out and sent once more like a REST call's, else raises ``GitHubRateLimited``;
        a mutation is sent again only when none of it ran (no ``data``). Any other failure of a
        mutation is ``ambiguous``: GitHub answers errors, or no ``data``, after part of one ran,
        as when it times out.
        """

        repository = _checked_repository(repository)
        mutation = is_mutation(query)
        if mutation:
            self._check_write("POST", GRAPHQL_PATH, repository)
        body = json_module.dumps(
            {"query": query, "variables": dict(variables or {})}, ensure_ascii=False
        ).encode()
        headers = {
            "accept": JSON_MEDIA_TYPE,
            "content-type": "application/json",
            REPOSITORY_HEADER: repository,
        }
        waited = False
        while True:
            response = await self._send(
                "POST", GRAPHQL_PATH, None, headers, body, retry_failures=not mutation
            )
            if not 200 <= response.status < 300:
                raise response_error(response, "POST", GRAPHQL_PATH)
            try:
                return _graphql_answer(response, repository, accept, mutation=mutation)
            except GitHubRateLimited as limited:
                wait = limited.retry_after or 0.0
                if waited or wait > self._max_wait:
                    raise
                waited = True
                await _sleep(wait)

    async def _items(
        self, path: str, query: Query | None, item_key: str | None, max_pages: int | None
    ) -> AsyncGenerator[Any, None]:
        pages = 0
        async with aclosing(self._pages(path, query, item_key)) as listing:
            async for page, _ in listing:
                for item in page:
                    yield item
                pages += 1
                if max_pages is not None and pages >= max_pages:
                    return

    async def _pages(
        self, path: str, query: Query | None, item_key: str | None
    ) -> AsyncGenerator[tuple[_Items, bool], None]:
        """Each page's items, and whether a next page follows."""

        sent: Query = {"per_page": str(DEFAULT_PAGE_SIZE), **(query or {})}
        read = {_query_key(sent)}
        encoded = _encoded(path)
        while True:
            response = await self.request("GET", path, query=sent)
            following = _next_link(response.header("link"))
            yield _page_items(response, path, item_key), following is not None
            if following is None:
                return
            sent = self._next_query(following, encoded)
            key = _query_key(sent)
            if key in read:
                raise GitHubError(
                    f"GitHub's next-page link for GET {path} repeats a page already read; "
                    "not followed.",
                    method="GET",
                    path=path,
                )
            read.add(key)

    def _next_query(self, link: str, path: str) -> Query:
        """A next-page link's query, once the link is to the same listing on the slot's API.

        GitHub names a repository's listing by the repository's id in its links
        (``/repositories/1296269/pulls``); the next page keeps the path it was asked for, so it
        goes through the slot for the same repository.
        """

        target = urlsplit(link)
        scheme, host, prefix = self._api
        linked = target.path[len(prefix) :] if target.path.startswith(prefix + "/") else ""
        if (
            target.scheme != scheme
            or target.netloc.lower() != host
            or not _same_listing(linked, path)
        ):
            raise GitHubError(
                f"GitHub's next-page link for GET {path} isn't the same listing on "
                f"{scheme}://{host}{prefix}; not followed.",
                method="GET",
                path=path,
            )
        return parse_qs(target.query, keep_blank_values=True)

    async def _writes_as_app(self) -> bool:
        """Whether this slot is a GitHub App's: a local App connection, or a Cayu Cloud slot
        bound to a GitHub App connection."""

        # Imported here: the credentials module builds on this one.
        from cayu.connectors.github.credentials import GitHubAppConnections

        if isinstance(self._client, GitHubAppConnections):
            return True
        if isinstance(self._client, CloudConnectionBroker):
            return (await self._slot_listed(self._client)).get("connector") == _APP_CONNECTOR
        return False

    async def _slot_listed(self, broker: CloudConnectionBroker) -> Mapping[str, Any]:
        """This client's slot as ``broker``, its connection client, lists it (its
        ``connector`` and ``account``), read once; ``connection_unbound`` when the Agent has no
        such slot."""

        async with self._learning.get():
            listed = self._known.get("listed")
            if listed is None:
                listed = self._known["listed"] = await self._read_slot(broker)
        return listed

    async def _read_slot(self, broker: CloudConnectionBroker) -> Mapping[str, Any]:
        try:
            items = await broker.slots()
        except ConnectionCallError as error:
            message = f"Cayu Cloud's broker didn't list the slot {self._slot}: {error}" + (
                f" ({error.code})" if error.code else ""
            )
            fields: dict[str, Any] = {
                "method": "GET",
                "path": _LISTING_PATH,
                "status": error.status,
            }
            if error.code is None or error.code in _TRANSIENT_CODES:
                raise GitHubUnavailable(message, **fields) from None
            raise GitHubConnectionRefused(message, code=error.code, **fields) from None
        listed = next(
            (item for item in items if isinstance(item, dict) and item.get("slot") == self._slot),
            None,
        )
        if listed is None:
            raise GitHubConnectionRefused(
                f"The slot {self._slot} isn't bound to a connection on Cayu Cloud: bind it with "
                f"`cayu cloud connections bind {self._slot} CONNECTION --application AGENT`, "
                "and for a GitHub App `--repository OWNER/NAME`.",
                code="connection_unbound",
                method="GET",
                path=_LISTING_PATH,
            )
        return listed

    async def _own_login(self) -> str:
        """The login of a connection client off Cayu Cloud: its own, else the token's owner."""

        known: Any = getattr(self._client, "login", None)
        if isinstance(known, str):
            return known
        if callable(known):
            try:
                return str(await known())
            except ConnectionCallError as error:
                raise self._refusal(error, "GET", "/app", write=False) from None
        try:
            user = await self.get(_USER_PATH)
        except GitHubPermissionDenied as denied:
            # An installation token, such as Actions' GITHUB_TOKEN, is no user's; any other
            # refusal (a revoked token's 401) is GitHub's own.
            if denied.status != 403 or _NO_USER not in denied.message:
                raise
            raise ValueError(
                f"GitHub answered {denied.status} to GET {_USER_PATH}: this token is no "
                "user's, such as an installation token. Name its login: "
                "token_connections(token, login=...), 'github-actions[bot]' for GitHub "
                "Actions' GITHUB_TOKEN."
            ) from None
        login = user.get("login") if isinstance(user, dict) else None
        if not isinstance(login, str) or not login:
            raise GitHubError(
                f"GitHub's answer to GET {_USER_PATH} names no login.",
                method="GET",
                path=_USER_PATH,
            )
        return login

    def _check_write(self, method: str, path: str, repository: str | None) -> None:
        if not self._read_only:
            return
        self._refused.append(RefusedWrite(method, path, repository))
        what = "a GraphQL mutation" if path == GRAPHQL_PATH else f"{method} {path}"
        raise GitHubWriteRefused(
            f"Read-only: refused {what}" + (f" on {repository}." if repository else "."),
            method=method,
            path=path,
        )

    async def _send(
        self,
        method: str,
        path: str,
        query: Query | None,
        headers: Mapping[str, str],
        body: bytes,
        *,
        retry_failures: bool,
    ) -> GitHubResponse:
        """GitHub's answer, once it is neither a rate limit nor a failure to retry: a rate
        limit is waited out once, and a failure retried only when ``retry_failures``."""

        sent = {**headers, "x-github-api-version": API_VERSION}
        repository = headers.get(REPOSITORY_HEADER)
        waited = False
        failures = 0
        while True:
            try:
                async with self._cap.get():
                    answer = await self._client.request(
                        self._slot, method, path, query=query, headers=sent, body=body
                    )
            except ConnectionCallError as error:
                if (
                    repository is not None
                    and error.code == "credential_unavailable"
                    and _NOT_INSTALLED in str(error)
                ):
                    # Cayu Cloud's broker refused it before sending, unretried: installing the
                    # App is what changes the answer.
                    raise GitHubAppNotInstalled(
                        f"{error} {method} {path} wasn't carried out.",
                        repository=repository,
                        method=method,
                        path=path,
                        status=error.status,
                    ) from None
                failure = self._refusal(error, method, path, write=not retry_failures)
            except GitHubAppNotInstalled as missing:
                # Raised while minting the call's token, before the call was sent: it names the
                # call, not the App's own.
                raise GitHubAppNotInstalled(
                    f"{missing.message} {method} {path} wasn't carried out.",
                    repository=missing.repository,
                    method=method,
                    path=path,
                    status=missing.status,
                    request_id=missing.request_id,
                ) from None
            else:
                response = GitHubResponse(answer.status, answer.headers, answer.body)
                failure = _limited_or_unavailable(
                    response, method, path, ambiguous=not retry_failures
                )
                if failure is None:
                    return response
            if isinstance(failure, GitHubRateLimited):
                wait = _BUSY_SECONDS if failure.retry_after is None else failure.retry_after
                if waited or wait > self._max_wait:
                    raise failure
                waited = True
                await _sleep(wait)
                continue
            if (
                isinstance(failure, GitHubUnavailable)
                and retry_failures
                and failures < self._read_retries
            ):
                await _sleep(_BACKOFF_SECONDS * 2**failures)
                failures += 1
                continue
            raise failure

    def _refusal(
        self, error: ConnectionCallError, method: str, path: str, *, write: bool
    ) -> GitHubError:
        """The typed error for a call the slot failed: ambiguous for a write that may have
        reached GitHub (``ConnectionCallError.ambiguous``)."""

        code = error.code
        ambiguous = write and error.ambiguous
        call = f"{method} {path}"
        if ambiguous:
            what = f"The slot {self._slot} may have sent {call}"
        elif not error.ambiguous:
            what = f"The slot {self._slot} didn't send {call}"
        elif code == "upstream_response_too_large":
            what = f"{call} was answered too large for the slot {self._slot}"
        else:
            what = f"The slot {self._slot} got no answer to {call}"
        message = f"{what}: {error}" + (f" ({code})" if code else "")
        if code == "too_many_calls":
            return GitHubRateLimited(
                message,
                retry_after=error.retry_after,
                secondary=False,
                method=method,
                path=path,
                status=error.status,
            )
        if code is None or code in _TRANSIENT_CODES:
            return GitHubUnavailable(
                message, method=method, path=path, status=error.status, ambiguous=ambiguous
            )
        return GitHubConnectionRefused(
            message, code=code, method=method, path=path, status=error.status, ambiguous=ambiguous
        )

    async def _download(self, location: str, path: str) -> str:
        """A redirected log, fetched with no credential; never naming its signed URL."""

        kept = bytearray()
        try:
            async with (
                httpx.AsyncClient(
                    transport=self._transport, timeout=DEFAULT_TIMEOUT_SECONDS
                ) as client,
                client.stream("GET", location) as response,
            ):
                if response.status_code != 200:
                    raise _LogDownloadFailed(
                        f"The log download for GET {path} answered {response.status_code}.",
                        method="GET",
                        path=path,
                        status=response.status_code,
                    )
                async for chunk in response.aiter_bytes():
                    kept.extend(chunk)
                    if len(kept) > MAX_TEXT_BYTES:
                        del kept[: len(kept) - MAX_TEXT_BYTES]
        except httpx.HTTPError as error:
            failure = type(error).__name__
        else:
            return bytes(kept).decode("utf-8", "replace")
        # Raised outside the handler, so neither its cause nor its context holds the request,
        # whose URL carries the signature.
        raise _LogDownloadFailed(
            f"The log download for GET {path} failed ({failure}).", method="GET", path=path
        )


def _account_login(listed: Mapping[str, Any], slot: str) -> str:
    """The login of the account Cayu Cloud lists for the slot's connection."""

    account = listed.get("account")
    name = account.get("name") if isinstance(account, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Cayu Cloud lists no account for the connection of slot {slot}, so the login it "
            "writes as isn't known: its connector names none, or this Cayu Cloud predates "
            "listing it."
        )
    return name.strip()


def _limited_or_unavailable(
    response: ConnectionResponse, method: str, path: str, *, ambiguous: bool
) -> GitHubError | None:
    """The rate limit or server failure this answer is, if it is one."""

    if response.status >= 500:
        return response_error(response, method, path, ambiguous=ambiguous)
    limit = _rate_limit(response)
    if limit is None:
        return None
    words, codes = github_words(response.body)
    request_id = response.header("x-github-request-id")
    return GitHubRateLimited(
        error_text(f"GitHub rate-limited {method} {path}", words, request_id),
        retry_after=limit[0],
        secondary=limit[1],
        method=method,
        path=path,
        status=response.status,
        request_id=request_id,
        codes=codes,
    )


def _rate_limit(response: ConnectionResponse) -> tuple[float, bool] | None:
    """How long to wait, and whether it is a secondary limit, when GitHub's answer is a rate
    limit: a 429, or a 403 with no requests left, a ``Retry-After`` or a secondary limit."""

    if response.status not in (403, 429):
        return None
    words, _ = github_words(response.body)
    primary = response.header("x-ratelimit-remaining") == "0"
    retry_after = retry_after_seconds(response.header("retry-after"))
    if (
        response.status == 403
        and not primary
        and retry_after is None
        and "secondary rate limit" not in words.lower()
    ):
        return None
    if retry_after is None and primary:
        retry_after = _reset_wait(response)
    return (_LIMIT_WAIT_SECONDS if retry_after is None else retry_after), not primary


def _reset_wait(response: ConnectionResponse) -> float | None:
    """Seconds until GitHub's ``x-ratelimit-reset``; None when it doesn't say."""

    reset = response.header("x-ratelimit-reset") or ""
    if not (reset.isascii() and reset.isdigit()):
        return None
    return max(0.0, int(reset) - time.time())


def _graphql_answer(
    response: GitHubResponse, repository: str, accept: ErrorFilter | None, *, mutation: bool
) -> GraphQLAnswer:
    request_id = response.header("x-github-request-id")
    fields: dict[str, Any] = {
        "method": "POST",
        "path": GRAPHQL_PATH,
        "status": response.status,
        "request_id": request_id,
    }
    try:
        answer = json_module.loads(response.body)
    except ValueError:
        answer = None
    if not isinstance(answer, dict):
        raise GitHubGraphQLFailed(
            "GitHub's GraphQL answer isn't a JSON object.",
            errors=(),
            data=None,
            ambiguous=mutation,
            **fields,
        )
    data = answer.get("data") if isinstance(answer.get("data"), dict) else None
    errors = graphql_errors(answer.get("errors"))
    taken = [accept is not None and accept(error) for error in errors]
    left = tuple(error for error, took in zip(errors, taken, strict=True) if not took)
    if any(error.type == "RATE_LIMITED" for error in left) and (data is None or not mutation):
        # Nothing of a mutation ran when it has no data, and a query is safe to send again.
        wait = _reset_wait(response)
        raise GitHubRateLimited(
            error_text(
                "GitHub rate-limited POST /graphql",
                "; ".join(error.message for error in left if error.type == "RATE_LIMITED"),
                request_id,
            ),
            retry_after=_LIMIT_WAIT_SECONDS if wait is None else wait,
            secondary=False,
            **fields,
        )
    if repository_missing(data, left):
        raise GitHubNotFound(
            f"The repository {repository} isn't there, or this credential can't read it.",
            **fields,
        )
    if left or data is None:
        said = "; ".join(
            f"{error.type or 'ERROR'} at {'.'.join(map(str, error.path)) or '(root)'}: "
            f"{error.message}"
            for error in left
        )
        raise GitHubGraphQLFailed(
            error_text("GitHub's GraphQL answer has errors", said or "no data", request_id),
            errors=errors,
            data=data,
            ambiguous=mutation,
            **fields,
        )
    # No error is left unaccepted here, so every one of them was accepted.
    return GraphQLAnswer(data, errors)


def _page_items(response: GitHubResponse, path: str, item_key: str | None) -> _Items:
    answer = response.json()
    items = answer.get(item_key) if item_key is not None and isinstance(answer, dict) else answer
    if not isinstance(items, list):
        shape = f"an object with a list under {item_key!r}" if item_key else "a list"
        raise GitHubError(
            f"GitHub's answer to GET {path} isn't {shape}.",
            method="GET",
            path=path,
            status=response.status,
            request_id=response.header("x-github-request-id"),
        )
    return items


def _next_link(header: str | None) -> str | None:
    for url, relations in _LINK.findall(header or ""):
        if "next" in relations.split():
            return url
    return None


def _query_key(query: Query) -> frozenset[tuple[str, tuple[str, ...]]]:
    return frozenset(
        (name, (value,) if isinstance(value, str) else tuple(value))
        for name, value in query.items()
    )


def _same_listing(linked: str, path: str) -> bool:
    """Whether a next-page link's path is the listing ``path``: the same path, or GitHub's name
    for it by id, which spells the owner, repository or team as numbers
    (``/repositories/1296269/pulls``, ``/organizations/9919/team/42/members``). The segments
    after the link's last id are then the listing itself, and ``path`` must end with them."""

    if linked == path:
        return True
    segments = linked.split("/")
    ids = [index for index, segment in enumerate(segments) if segment.isdigit()]
    if not ids:
        return False
    listing = segments[ids[-1] + 1 :]
    return bool(listing) and path.split("/")[-len(listing) :] == listing


def _encoded(path: str) -> str:
    """``path`` percent-encoded where a URL needs it; an existing escape (``%2F``) is kept, and
    any other ``%`` is a literal one."""

    if not path.startswith("/"):
        raise ValueError(f"A GitHub path starts with /: {path!r}.")
    return quote(_LITERAL_PERCENT.sub("%25", path), safe=_PATH_SAFE)


def _repository_in(path: str) -> str | None:
    """The ``owner/name`` a ``/repos/`` path is about; None for any other path."""

    if not path.startswith("/repos/"):
        return None
    return _checked_repository("/".join(path.split("/")[2:4]))


def _checked_repository(repository: str) -> str:
    if not _REPOSITORY.fullmatch(repository) or repository.split("/")[1] in (".", ".."):
        raise ValueError(f"A repository is owner/name: {repository!r}.")
    return repository
