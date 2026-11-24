"""Read Cayu Cloud's People directory from an Agent, with the Agent's own Cloud identity.

On Cayu Cloud every Agent process holds its attention credential
(``CAYU_ATTENTION_HANDOFF_TOKEN``), and a Cloud that serves the People directory to
Agents binds ``CAYU_PEOPLE_URL`` beside it::

    GET $CAYU_PEOPLE_URL
    Authorization: Bearer $CAYU_ATTENTION_HANDOFF_TOKEN
    -> {"application_id": "...", "items": [{"user_id": "...", "role": "admin",
        "name": "...", "email": "...", "on_agent": true}]}

``items`` is everyone in the Agent's Organization, with their Organization role and
whether they work with this Agent. The parts compose on their own:

- :class:`CloudPeopleDirectory` reads that directory.
- :class:`StaticPeopleDirectory` is a fixed list, for tests and for ``cayu serve --dev``:
  :meth:`StaticPeopleDirectory.from_pyproject` reads the people of
  ``[tool.cayu.serve.identity_proxy]``, the same people the development identity proxy
  signs in as.
- :func:`people_directory` picks one: Cloud's when its address and credential are bound,
  otherwise the stand-in it is given. On Cayu Cloud without ``CAYU_PEOPLE_URL`` it raises
  :class:`PeopleDirectoryUnsupported` rather than silently using the stand-in.

The directory only reads. Who may see or change a record stays the application's
decision; the signed-in person comes from the request (``operator_sign_in``), and the
directory says who else is in the Organization.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx

from cayu._development_people import parse_identity_proxy_table
from cayu.human_attention_handoff import TOKEN_ENVIRONMENT
from cayu.human_attention_handoff import URL_ENVIRONMENT as _ATTENTION_URL_ENVIRONMENT

#: Where a Cloud that serves the People directory to Agents binds its address.
PEOPLE_URL_ENVIRONMENT = "CAYU_PEOPLE_URL"
_REQUEST_TIMEOUT_SECONDS = 10.0
_MAX_PEOPLE = 10_000


@dataclass(frozen=True)
class Person:
    """One person in the Organization.

    ``role`` is their Organization role (``admin``, ``developer`` or ``member`` on
    Cayu Cloud). ``on_agent`` says whether they are one of this Agent's people.
    """

    user_id: str
    role: str
    name: str | None = None
    email: str | None = None
    on_agent: bool = False


class PeopleDirectoryError(Exception):
    """The directory could not be read."""


class PeopleDirectoryUnavailable(PeopleDirectoryError):
    """The directory exists but could not be read now; ``code`` names why when Cloud said."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class PeopleDirectoryUnsupported(PeopleDirectoryError):
    """This Cayu Cloud doesn't serve the People directory to Agents."""


@runtime_checkable
class PeopleDirectory(Protocol):
    """Who is in the Organization, as an Agent may read it."""

    async def list_people(self) -> tuple[Person, ...]: ...


class StaticPeopleDirectory:
    """A fixed directory: the local stand-in for Cloud's, and a test double."""

    def __init__(self, people: Iterable[Person]) -> None:
        listed = tuple(people)
        seen: set[str] = set()
        for person in listed:
            if not isinstance(person, Person):
                raise TypeError("StaticPeopleDirectory takes Person values.")
            if person.user_id in seen:
                raise ValueError(f"Person {person.user_id!r} is listed twice.")
            seen.add(person.user_id)
        self._people = listed

    @classmethod
    def from_pyproject(cls, path: str | Path = "pyproject.toml") -> StaticPeopleDirectory:
        """The people of ``[tool.cayu.serve.identity_proxy]``, all working with the Agent.

        A missing file or table gives an empty directory; an invalid table raises
        ``ValueError`` as ``cayu check`` does.
        """

        try:
            with Path(path).open("rb") as file:
                document = tomllib.load(file)
        except FileNotFoundError:
            return cls(())
        table = document.get("tool", {}).get("cayu", {}).get("serve", {}).get("identity_proxy")
        if table is None:
            return cls(())
        people, _ = parse_identity_proxy_table(table)
        return cls(
            Person(
                user_id=person.subject,
                role=person.role,
                name=person.name,
                email=person.email,
                on_agent=True,
            )
            for person in people
        )

    async def list_people(self) -> tuple[Person, ...]:
        return self._people


class CloudPeopleDirectory:
    """Cayu Cloud's People directory, read with the Agent's own attention credential.

    Each call reads Cloud afresh; cache in the application if it reads often. Raises
    :class:`PeopleDirectoryUnsupported` when Cloud doesn't serve the read (HTTP 404) and
    :class:`PeopleDirectoryUnavailable` for a refused credential, an unreadable directory
    or a network failure. The credential is never included in an error.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = _REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        url = url.strip()
        if not url.startswith(("https://", "http://")):
            raise ValueError("CloudPeopleDirectory needs an http(s) URL.")
        if not token.strip():
            raise ValueError("CloudPeopleDirectory needs the Agent's attention credential.")
        self._url = url
        self._token = token.strip()
        self._transport = transport
        self._timeout_s = timeout_s

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] = os.environ,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> CloudPeopleDirectory:
        """The directory Cloud bound for this process.

        Raises :class:`PeopleDirectoryUnsupported` when ``CAYU_PEOPLE_URL`` or the
        attention credential isn't bound: outside Cayu Cloud, or on a Cayu Cloud that
        doesn't serve the People directory to Agents.
        """

        url = environ.get(PEOPLE_URL_ENVIRONMENT, "").strip()
        token = environ.get(TOKEN_ENVIRONMENT, "").strip()
        if not url or not token:
            missing = PEOPLE_URL_ENVIRONMENT if not url else TOKEN_ENVIRONMENT
            raise PeopleDirectoryUnsupported(
                f"{missing} is not set. Reading the People directory needs a Cayu Cloud that "
                "serves it to Agents; outside Cayu Cloud use StaticPeopleDirectory."
            )
        return cls(url, token, transport=transport)

    async def list_people(self) -> tuple[Person, ...]:
        async with httpx.AsyncClient(
            timeout=self._timeout_s, transport=self._transport, follow_redirects=False
        ) as client:
            try:
                response = await client.get(
                    self._url, headers={"Authorization": f"Bearer {self._token}"}
                )
            except httpx.HTTPError as exc:
                raise PeopleDirectoryUnavailable(
                    f"Cayu Cloud's People directory could not be reached ({type(exc).__name__})."
                ) from None
        if response.status_code == 404:
            raise PeopleDirectoryUnsupported(
                "This Cayu Cloud doesn't serve the People directory to Agents (HTTP 404). "
                "It needs a Cayu Cloud that supports reading People with the Agent's credential."
            )
        if response.status_code != 200:
            code = _error_code(response)
            raise PeopleDirectoryUnavailable(
                f"Cayu Cloud's People directory answered HTTP {response.status_code}"
                + (f" ({code})." if code else "."),
                code=code,
            )
        return _parse_people(response)


def people_directory(
    environ: Mapping[str, str] = os.environ,
    *,
    local: PeopleDirectory | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> PeopleDirectory:
    """Cloud's directory when it is bound, otherwise ``local``.

    ``local`` defaults to :meth:`StaticPeopleDirectory.from_pyproject` in the current
    directory. On Cayu Cloud (its attention handoff is bound) without ``CAYU_PEOPLE_URL``
    this raises :class:`PeopleDirectoryUnsupported` instead of using the stand-in, so a
    deployed Agent never reads development people.
    """

    if environ.get(PEOPLE_URL_ENVIRONMENT, "").strip():
        return CloudPeopleDirectory.from_environment(environ, transport=transport)
    if (
        environ.get(TOKEN_ENVIRONMENT, "").strip()
        or environ.get(_ATTENTION_URL_ENVIRONMENT, "").strip()
    ):
        raise PeopleDirectoryUnsupported(
            f"{PEOPLE_URL_ENVIRONMENT} is not set on this Cayu Cloud. Reading the People "
            "directory needs a Cayu Cloud that serves it to Agents."
        )
    return StaticPeopleDirectory.from_pyproject() if local is None else local


async def agent_people(directory: PeopleDirectory) -> tuple[Person, ...]:
    """The people who work with this Agent."""

    return tuple(person for person in await directory.list_people() if person.on_agent)


async def find_person(directory: PeopleDirectory, user_id: str) -> Person | None:
    """The person with ``user_id``, or ``None`` when they aren't in the Organization."""

    return next((p for p in await directory.list_people() if p.user_id == user_id), None)


def _error_code(response: httpx.Response) -> str | None:
    try:
        body = response.json()
    except ValueError:
        return None
    detail = body.get("detail") if isinstance(body, dict) else None
    code = detail.get("code") if isinstance(detail, dict) else None
    return code if isinstance(code, str) and len(code) <= 100 else None


def _parse_people(response: httpx.Response) -> tuple[Person, ...]:
    def malformed() -> PeopleDirectoryUnavailable:
        return PeopleDirectoryUnavailable(
            "Cayu Cloud's People directory answered with a body this Runtime can't read."
        )

    try:
        body = response.json()
    except ValueError:
        raise malformed() from None
    items = body.get("items") if isinstance(body, dict) else None
    if not isinstance(items, list) or len(items) > _MAX_PEOPLE:
        raise malformed()
    people: list[Person] = []
    for item in items:
        if not isinstance(item, dict):
            raise malformed()
        user_id, role = item.get("user_id"), item.get("role")
        name, email = item.get("name"), item.get("email")
        if not isinstance(user_id, str) or not user_id or not isinstance(role, str):
            raise malformed()
        if not all(value is None or isinstance(value, str) for value in (name, email)):
            raise malformed()
        people.append(
            Person(
                user_id=user_id,
                role=role,
                name=name,
                email=email,
                on_agent=item.get("on_agent") is True,
            )
        )
    return tuple(people)
