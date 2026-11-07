"""Calls to an application's connections: the outside systems it reaches through named slots.

An application names what it needs by slot (``warehouse``, ``crm``) and never holds the
credential behind it when a host serves the slot for it. A ``ConnectionClient`` sends one HTTP
request through a slot and answers the connector's response:

* ``CloudConnectionBroker`` sends it to Cayu Cloud's connection broker, which attaches the
  Organization's credential bound to the slot (``docs/adr/0003-connections-runtime-contract.md``).
  ``from_environment`` configures it from ``CAYU_CONNECTIONS_URL`` and the Agent's own
  ``CAYU_CONNECTIONS_TOKEN``, both of which only Cayu Cloud sets.
* ``LocalConnections`` sends it straight to each slot's own host with headers the application
  gives, for a laptop or a test.

The connector's status is the answer's, whatever it is; ``ConnectionCallError`` is raised only
when the request couldn't be sent: the slot is unbound, the host refused to send it, or the
connector couldn't be reached.
"""

from __future__ import annotations

import base64
import contextlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import unquote

import httpx

URL_ENVIRONMENT = "CAYU_CONNECTIONS_URL"
#: The Agent's credential for the connection broker, and for nothing else.
TOKEN_ENVIRONMENT = "CAYU_CONNECTIONS_TOKEN"
DEFAULT_TIMEOUT_SECONDS = 70.0
#: The largest body one call carries, as Cayu Cloud's broker limits it.
MAX_BODY_BYTES = 5 * 1024 * 1024
_ENCODED_SEPARATOR = re.compile(r"%(2f|5c)", re.IGNORECASE)
_HOP_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
#: A slot name: 1-63 lowercase letters, digits or interior hyphens.
_SLOT = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

Query = Mapping[str, str | Sequence[str]]


@dataclass(frozen=True)
class ConnectionResponse:
    status: int
    #: Lowercase names, in the order the connector sent them.
    headers: tuple[tuple[str, str], ...]
    body: bytes

    def header(self, name: str) -> str | None:
        return next((value for key, value in self.headers if key == name.lower()), None)


class ConnectionCallError(Exception):
    """The request wasn't sent through the slot; ``code`` names why, when the host said."""

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None):
        super().__init__(message)
        self.code = code
        self.status = status


class ConnectionClient(Protocol):
    async def request(
        self,
        slot: str,
        method: str,
        path: str,
        *,
        query: Query | None = None,
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
    ) -> ConnectionResponse: ...


class CloudConnectionBroker:
    """Cayu Cloud's connection broker, authenticated as the Agent."""

    def __init__(
        self,
        *,
        url: str,
        token: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not url.startswith(("https://", "http://")):
            raise ValueError("The connection broker URL must be http(s).")
        if not token:
            raise ValueError("The connection broker token must not be empty.")
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._transport = _kept_open(transport)
        self._timeout = timeout_seconds

    def __repr__(self) -> str:
        return f"CloudConnectionBroker(url={self._url!r})"

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> CloudConnectionBroker | None:
        """The broker Cayu Cloud bound this process to; None outside Cayu Cloud."""

        environ = os.environ if environ is None else environ
        url = environ.get(URL_ENVIRONMENT, "").strip()
        if not url:
            return None
        token = environ.get(TOKEN_ENVIRONMENT, "").strip()
        if not token:
            raise ValueError(f"{URL_ENVIRONMENT} is set without {TOKEN_ENVIRONMENT}.")
        return cls(url=url, token=token)

    async def slots(self) -> list[dict[str, Any]]:
        """The Agent's bound slots and each connection's state."""

        answer = await self._call("GET", "")
        items = answer.get("items")
        return items if isinstance(items, list) else []

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
        answer = await self._call(
            "POST",
            f"/{_slot(slot)}/request",
            json={
                "method": method.upper(),
                "path": path,
                "query": {
                    name: value if isinstance(value, str) else list(value)
                    for name, value in (query or {}).items()
                },
                "headers": _checked(path, headers, body),
                "body": base64.b64encode(body).decode(),
            },
        )
        try:
            return ConnectionResponse(
                status=int(answer["status"]),
                headers=tuple((str(name), str(value)) for name, value in answer["headers"]),
                body=base64.b64decode(answer["body"], validate=True),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConnectionCallError("The connection broker's answer is malformed.") from exc

    async def _call(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout, headers=self._headers
            ) as client:
                response = await client.request(method, f"{self._url}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise ConnectionCallError(
                f"The connection broker is unreachable ({type(exc).__name__})."
            ) from exc
        if not 200 <= response.status_code < 300:
            code = message = None
            with contextlib.suppress(ValueError, AttributeError, TypeError):
                detail = response.json().get("detail")
                code, message = detail.get("code"), detail.get("message")
            raise ConnectionCallError(
                message
                if isinstance(message, str)
                else f"The broker answered {response.status_code}.",
                code=code if isinstance(code, str) else None,
                status=response.status_code,
            )
        try:
            answer = response.json()
        except ValueError as exc:
            raise ConnectionCallError("The connection broker answered with invalid JSON.") from exc
        if not isinstance(answer, dict):
            raise ConnectionCallError("The connection broker's answer is malformed.")
        return answer


@dataclass(frozen=True)
class LocalSlot:
    """Where a slot's requests go locally, and the headers that carry its credential."""

    base_url: str
    #: Never in its repr: it carries the credential.
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)


class LocalConnections:
    """Each slot straight to its own host, for local runs and tests."""

    def __init__(
        self,
        slots: Mapping[str, LocalSlot],
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._slots = dict(slots)
        self._transport = _kept_open(transport)
        self._timeout = timeout_seconds

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
        configured = self._slots.get(slot)
        if configured is None:
            raise ConnectionCallError(
                f"The slot {slot} isn't configured.", code="connection_unbound"
            )
        # The slot's own headers win, whatever case either side names them in.
        # The slot's host alone names where its credential goes: never the application's `Host`,
        # nor the hop-by-hop headers of its own connection, as Cayu Cloud's broker drops them.
        sent = {
            name: value
            for name, value in _checked(path, headers, body).items()
            if name not in _HOP_HEADERS
        }
        sent.update({name.lower(): value for name, value in configured.headers.items()})
        url = httpx.URL(configured.base_url.rstrip("/") + path)
        if url.host != httpx.URL(configured.base_url).host:
            raise ConnectionCallError(
                f"The path leaves the slot {slot}'s host.", code="path_invalid"
            )
        params = [
            (name, each)
            for name, value in (query or {}).items()
            for each in ([value] if isinstance(value, str) else value)
        ]
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout
            ) as client:
                response = await client.request(
                    method.upper(),
                    url,
                    params=params,
                    headers=sent,
                    content=body,
                    follow_redirects=False,
                )
        except httpx.HTTPError as exc:
            raise ConnectionCallError(
                f"The slot {slot}'s host is unreachable ({type(exc).__name__}).",
                code="upstream_failed",
            ) from exc
        return ConnectionResponse(
            status=response.status_code,
            headers=tuple((name.lower(), value) for name, value in response.headers.multi_items()),
            body=response.content,
        )


def _checked(path: str, headers: Mapping[str, str] | None, body: bytes) -> dict[str, str]:
    """The request's headers, lowercased, once its path and body are ones the broker sends:
    refused here as Cayu Cloud's broker refuses them, before anything leaves the process."""

    decoded = unquote(path)
    if (
        not path.startswith("/")
        or "//" in path
        or any(segment in {".", ".."} for segment in decoded.split("/"))
        or any(character in decoded for character in "?#\\\x00")
        or _ENCODED_SEPARATOR.search(path)
    ):
        raise ConnectionCallError(
            "A path starts with /, has no . or .. segment, empty segment, encoded separator, "
            "backslash or query.",
            code="path_invalid",
        )
    if len(body) > MAX_BODY_BYTES:
        raise ConnectionCallError(
            f"A body is at most {MAX_BODY_BYTES} bytes.", code="body_too_large"
        )
    lowered: dict[str, str] = {}
    for name, value in (headers or {}).items():
        key = name.lower()
        if key in lowered:
            raise ConnectionCallError(f"The header {key} is given twice.", code="headers_invalid")
        lowered[key] = value
    return lowered


def _slot(slot: str) -> str:
    """The slot's name, once it is one: never a name that would change the broker's route."""

    if not _SLOT.fullmatch(slot):
        raise ConnectionCallError(
            "A slot name is 1-63 lowercase letters, digits or interior hyphens.",
            code="slot_invalid",
        )
    return slot


class _KeptOpen(httpx.AsyncBaseTransport):
    """An application's transport, shared by every call: closing one call's client leaves it
    open for the others, and its owner closes it."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        return None


def _kept_open(transport: httpx.AsyncBaseTransport | None) -> httpx.AsyncBaseTransport | None:
    return None if transport is None else _KeptOpen(transport)
