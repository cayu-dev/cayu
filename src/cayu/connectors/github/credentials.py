"""Local credentials for a GitHub slot, off Cayu Cloud: a token, or a GitHub App's
installation tokens."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections import OrderedDict
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from cayu.connections import (
    CloudConnectionBroker,
    ConnectionCallError,
    ConnectionClient,
    ConnectionResponse,
    LocalConnections,
    LocalSlot,
    Query,
)
from cayu.connectors.github._errors import GitHubAppNotInstalled, error_text, github_words
from cayu.connectors.github.client import (
    API_URL,
    API_VERSION,
    DEFAULT_SLOT,
    GRAPHQL_PATH,
    JSON_MEDIA_TYPE,
    RATE_LIMIT_PATH,
    REPOSITORY_HEADER,
    _checked_repository,
    _LoopBound,
    _rate_limit,
)

TOKEN_ENVIRONMENT = "GITHUB_TOKEN"
APP_ID_ENVIRONMENT = "GITHUB_APP_ID"
APP_PRIVATE_KEY_ENVIRONMENT = "GITHUB_APP_PRIVATE_KEY"
#: An installation token is replaced this long before GitHub says it expires.
TOKEN_REFRESH_SECONDS = 300
#: The repositories whose installation and token one App client keeps.
MAX_CACHED_REPOSITORIES = 256
#: An App's sign-in token is issued a minute in the past, for clock drift, and lasts ten.
_JWT_BACKDATE_SECONDS = 60
_JWT_LIFETIME_SECONDS = 540
if TYPE_CHECKING:
    import httpx

_LEVELS = frozenset({"read", "write", "admin"})
_now = time.time


def token_connections(
    token: str,
    *,
    slot: str = DEFAULT_SLOT,
    api_url: str = API_URL,
    transport: httpx.AsyncBaseTransport | None = None,
) -> LocalConnections:
    """``slot`` straight to GitHub with ``token`` (a personal access token, or Actions'
    ``GITHUB_TOKEN``): every call is made in the token owner's name."""

    if not token.strip():
        raise ValueError("The GitHub token must not be empty.")
    return _TokenConnections(
        {slot: LocalSlot(api_url, {"authorization": f"Bearer {token.strip()}"})},
        transport=transport,
    )


def app_connections(
    app_id: str,
    private_key: str,
    *,
    permissions: Mapping[str, str],
    slot: str = DEFAULT_SLOT,
    api_url: str = API_URL,
    repositories: Collection[str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> GitHubAppConnections:
    """``slot`` straight to GitHub as a GitHub App: each call carries an installation token for
    its one repository with exactly ``permissions`` (``{"pull_requests": "write", ...}``), as a
    Cayu Cloud binding would. ``repositories``, when given, are the only ones it calls."""

    return GitHubAppConnections(
        app_id,
        private_key,
        permissions=permissions,
        slot=slot,
        api_url=api_url,
        repositories=repositories,
        transport=transport,
    )


def connections_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    permissions: Mapping[str, str],
    slot: str = DEFAULT_SLOT,
) -> ConnectionClient | None:
    """Cayu Cloud's broker when this process has one; else ``GITHUB_TOKEN``; else the App of
    ``GITHUB_APP_ID`` and ``GITHUB_APP_PRIVATE_KEY`` with ``permissions``; else None.

    A token and an App both set is an error rather than a choice: they sign comments in
    different names, and a marker trusted by its author's login depends on which.
    """

    environ = os.environ if environ is None else environ
    broker = CloudConnectionBroker.from_environment(environ)
    if broker is not None:
        return broker
    token = environ.get(TOKEN_ENVIRONMENT, "").strip()
    app_id = environ.get(APP_ID_ENVIRONMENT, "").strip()
    private_key = environ.get(APP_PRIVATE_KEY_ENVIRONMENT, "").strip()
    if token and (app_id or private_key):
        raise ValueError(
            f"Set {TOKEN_ENVIRONMENT}, or {APP_ID_ENVIRONMENT} with "
            f"{APP_PRIVATE_KEY_ENVIRONMENT}, not both: they act in different names."
        )
    if token:
        return token_connections(token, slot=slot)
    if not (app_id or private_key):
        return None
    if not (app_id and private_key):
        raise ValueError(
            f"{APP_ID_ENVIRONMENT} and {APP_PRIVATE_KEY_ENVIRONMENT} must be set together."
        )
    if "\n" not in private_key:
        private_key = private_key.replace("\\n", "\n")
    return app_connections(app_id, private_key, permissions=permissions, slot=slot)


class _TokenConnections(LocalConnections):
    """A token's slot, dropping the repository header: GitHub never sees it."""

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
        sent = {
            name: value
            for name, value in (headers or {}).items()
            if name.lower() != REPOSITORY_HEADER
        }
        return await super().request(slot, method, path, query=query, headers=sent, body=body)


@dataclass
class _Token:
    value: str = field(repr=False)
    expires_at: float


@dataclass
class _Repository:
    #: The running event loop's lock, so a client outlives one loop.
    locks: _LoopBound[asyncio.Lock] = field(
        default_factory=lambda: _LoopBound(asyncio.Lock), repr=False
    )
    installation: int | None = None
    token: _Token | None = field(default=None, repr=False)

    def lock(self) -> asyncio.Lock:
        return self.locks.get()

    def busy(self) -> bool:
        lock = self.locks.current()
        return lock is not None and lock.locked()


class GitHubAppConnections:
    """A ``ConnectionClient`` that calls GitHub as a GitHub App (see ``app_connections``).

    Each call names its repository in ``REPOSITORY_HEADER``, which ``GitHubClient`` sets and
    this client drops before sending. The installation and token are kept per repository, for
    at most ``MAX_CACHED_REPOSITORIES``; a token is minted once however many calls miss it at
    once, replaced ``TOKEN_REFRESH_SECONDS`` before it expires, and dropped and minted again
    once when GitHub answers 401. It calls only a repository's own API
    (``/repos/{owner}/{name}/...``), ``/graphql`` and ``/rate_limit``, as Cayu Cloud's GitHub
    connector is to; each names the repository whose token it uses. ``GitHubClient`` names none
    for ``/rate_limit`` yet, so it can't read that path through this slot.
    Neither the private key nor a token is ever in a repr or an error.
    """

    def __init__(
        self,
        app_id: str,
        private_key: str,
        *,
        permissions: Mapping[str, str],
        slot: str = DEFAULT_SLOT,
        api_url: str = API_URL,
        repositories: Collection[str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        app_id = str(app_id).strip()
        if not app_id or any(character.isspace() for character in app_id):
            raise ValueError("The GitHub App id must be its id or client id.")
        self._app_id = app_id
        self._key = _private_key(private_key)
        self._permissions = _checked_permissions(permissions)
        self._repositories = (
            None
            if repositories is None
            else frozenset(_checked_repository(name).lower() for name in repositories)
        )
        self._slot = slot
        self._api_url = api_url
        self._github = LocalConnections({slot: LocalSlot(api_url)}, transport=transport)
        self._entries: OrderedDict[str, _Repository] = OrderedDict()
        self._login: str | None = None

    def __repr__(self) -> str:
        return (
            f"GitHubAppConnections(app_id={self._app_id!r}, slot={self._slot!r}, "
            f"api_url={self._api_url!r})"
        )

    @property
    def permissions(self) -> Mapping[str, str]:
        return dict(self._permissions)

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
        if slot != self._slot:
            raise ConnectionCallError(
                f"The slot {slot} isn't configured.", code="connection_unbound"
            )
        sent = {name.lower(): value for name, value in (headers or {}).items()}
        if path not in (GRAPHQL_PATH, RATE_LIMIT_PATH) and not path.startswith("/repos/"):
            raise ConnectionCallError(
                "A GitHub App slot calls only a repository's own API (/repos/{owner}/{name}/...), "
                "/graphql and /rate_limit.",
                code="path_not_allowed",
            )
        named = sent.pop(REPOSITORY_HEADER, None)
        if named is None and path == RATE_LIMIT_PATH:
            raise ConnectionCallError(
                f"A GitHub App slot answers {RATE_LIMIT_PATH} only for a repository named in "
                f"{REPOSITORY_HEADER}, which GitHubClient doesn't send for it yet.",
                code="path_not_allowed",
            )
        repository = self._repository(named, path)
        token = await self._token(repository)
        response = await self._send(method, path, query, sent, body, token)
        if response.status == 401:
            self._drop(repository, token)
            response = await self._send(
                method, path, query, sent, body, await self._token(repository)
            )
        return response

    async def login(self) -> str:
        """The App's bot login, ``<slug>[bot]``: the author of every comment it writes."""

        if self._login is None:
            response = await self._as_app("GET", "/app")
            if response.status != 200:
                raise _app_failure(response, "the App's own details")
            slug = _json_field(response, "slug")
            if not isinstance(slug, str) or not slug:
                raise ConnectionCallError("GitHub's answer for the App has no slug.")
            self._login = f"{slug}[bot]"
        return self._login

    def _repository(self, named: str | None, path: str) -> str:
        if named is None:
            raise ConnectionCallError(
                f"A GitHub App call names its repository in {REPOSITORY_HEADER}.",
                code="path_not_allowed",
            )
        try:
            repository = _checked_repository(named)
        except ValueError as error:
            raise ConnectionCallError(str(error), code="headers_invalid") from None
        in_path = "/".join(path.split("/")[2:4]) if path.startswith("/repos/") else None
        if in_path is not None and in_path.lower() != repository.lower():
            raise ConnectionCallError(
                f"The path isn't about {repository}, the call's repository.",
                code="path_not_allowed",
            )
        if self._repositories is not None and repository.lower() not in self._repositories:
            raise ConnectionCallError(
                f"The repository {repository} isn't one this slot may call.",
                code="path_not_allowed",
            )
        return repository

    async def _send(
        self,
        method: str,
        path: str,
        query: Query | None,
        headers: Mapping[str, str],
        body: bytes,
        token: str,
    ) -> ConnectionResponse:
        return await self._github.request(
            self._slot,
            method,
            path,
            query=query,
            headers={**headers, "authorization": f"Bearer {token}"},
            body=body,
        )

    async def _token(self, repository: str) -> str:
        entry = self._entry(repository.lower())
        async with entry.lock():
            if entry.token is not None and entry.token.expires_at - _now() > TOKEN_REFRESH_SECONDS:
                return entry.token.value
            cached = entry.installation
            installation = cached if cached is not None else await self._installation(repository)
            entry.installation = installation
            try:
                entry.token = await self._mint(installation, repository)
            except GitHubAppNotInstalled:
                # The installation was removed, and perhaps the App installed again: look it
                # up once more, unless it was just looked up.
                entry.installation = None
                if cached is None:
                    raise
                installation = await self._installation(repository)
                entry.installation = installation
                entry.token = await self._mint(installation, repository)
            return entry.token.value

    def _entry(self, key: str) -> _Repository:
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return entry
        entry = self._entries[key] = _Repository()
        # The oldest entries go first, never one a call is minting for: that call's waiters
        # would otherwise find a fresh entry and mint again.
        idle = [name for name, kept in self._entries.items() if not kept.busy() and name != key]
        for name in idle[: max(0, len(self._entries) - MAX_CACHED_REPOSITORIES)]:
            del self._entries[name]
        return entry

    def _drop(self, repository: str, token: str) -> None:
        """Forget a token GitHub refused, and the installation it came from, unless another
        call already replaced it."""

        entry = self._entries.get(repository.lower())
        if entry is not None and entry.token is not None and entry.token.value == token:
            entry.token = None
            entry.installation = None

    async def _installation(self, repository: str) -> int:
        path = f"/repos/{repository}/installation"
        response = await self._as_app("GET", path)
        if response.status == 404:
            raise _not_installed(response, self._app_id, repository, "GET", path)
        installation = _json_field(response, "id") if response.status == 200 else None
        if not isinstance(installation, int):
            raise _app_failure(response, f"the App's installation on {repository}")
        return installation

    async def _mint(self, installation: int, repository: str) -> _Token:
        path = f"/app/installations/{installation}/access_tokens"
        response = await self._as_app(
            "POST",
            path,
            {"repositories": [repository.split("/")[1]], "permissions": self._permissions},
        )
        if response.status == 404:
            raise _not_installed(response, self._app_id, repository, "POST", path)
        value = _json_field(response, "token") if response.status == 201 else None
        expires = _json_field(response, "expires_at") if response.status == 201 else None
        if not isinstance(value, str) or not isinstance(expires, str):
            raise _app_failure(response, f"an installation token for {repository}")
        try:
            expires_at = datetime.fromisoformat(expires).timestamp()
        except ValueError:
            raise ConnectionCallError(
                "GitHub's installation token has an unreadable expiry."
            ) from None
        return _Token(value, expires_at)

    async def _as_app(self, method: str, path: str, document: object = None) -> ConnectionResponse:
        """A call signed as the App itself, with a fresh JWT."""

        headers = {
            "accept": JSON_MEDIA_TYPE,
            "authorization": f"Bearer {self._jwt()}",
            "x-github-api-version": API_VERSION,
        }
        body = b""
        if document is not None:
            headers["content-type"] = "application/json"
            body = json.dumps(document).encode()
        return await self._github.request(self._slot, method, path, headers=headers, body=body)

    def _jwt(self) -> str:
        now = int(_now())
        header = _base64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
        claims = {
            "iat": now - _JWT_BACKDATE_SECONDS,
            "exp": now + _JWT_LIFETIME_SECONDS,
            "iss": self._app_id,
        }
        signed = f"{header}.{_base64url(json.dumps(claims).encode())}"
        signature = self._key.sign(signed.encode(), padding.PKCS1v15(), hashes.SHA256())
        return f"{signed}.{_base64url(signature)}"


def _private_key(text: str) -> rsa.RSAPrivateKey:
    try:
        key = serialization.load_pem_private_key(text.encode(), password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm):
        raise ValueError(
            "The GitHub App private key isn't an unencrypted PEM private key."
        ) from None
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError("The GitHub App private key isn't an RSA key.")
    return key


def _checked_permissions(permissions: Mapping[str, str]) -> dict[str, str]:
    """The App permissions an installation token carries; never empty, which would mean all
    of the installation's."""

    if not permissions:
        raise ValueError("Name the App permissions the slot's tokens carry.")
    checked = {}
    for name, level in permissions.items():
        if not name.replace("_", "").isalpha() or not name.islower() or level not in _LEVELS:
            raise ValueError(f"An App permission is a name and read, write or admin: {name!r}.")
        checked[name] = level
    return checked


def _json_field(response: ConnectionResponse, name: str) -> object:
    try:
        answer = json.loads(response.body)
    except ValueError:
        return None
    return answer.get(name) if isinstance(answer, dict) else None


def _not_installed(
    response: ConnectionResponse, app_id: str, repository: str, method: str, path: str
) -> GitHubAppNotInstalled:
    return GitHubAppNotInstalled(
        f"The GitHub App {app_id} isn't installed on {repository}, or can't see it.",
        repository=repository,
        method=method,
        path=path,
        status=response.status,
        request_id=response.header("x-github-request-id"),
    )


def _app_failure(response: ConnectionResponse, what: str) -> ConnectionCallError:
    """GitHub's refusal of the App's own call, with its words and any rate-limit wait."""

    words, _ = github_words(response.body)
    message = error_text(
        f"GitHub answered {response.status} for {what}",
        words,
        response.header("x-github-request-id"),
    )
    limit = _rate_limit(response)
    if limit is not None:
        code = "too_many_calls"
    elif response.status >= 500:
        code = "upstream_failed"
    elif response.status == 422:
        code = "scope_not_allowed"
    else:
        code = "credential_invalid"
    return ConnectionCallError(
        message,
        code=code,
        status=response.status,
        retry_after=None if limit is None else limit[0],
    )


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()
