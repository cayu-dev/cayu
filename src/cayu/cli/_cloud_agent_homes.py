"""Where each Agent deploys: the Cayu Cloud install and Organization it was deployed to.

`cayu cloud deploy` records an Agent's home the first time it deploys from this machine, and
every command that changes an Agent compares the Cloud and Organization it acts in with that
record before it changes anything. A command for the Agent from another Organization or Cloud
is refused, unless it names that Organization with ``--organization``, so a deploy can't go
somewhere else just because another login or context happens to be selected.

The records are local: one private JSON file per Agent slug under ``agents/`` in the Cayu Cloud
config directory. Nothing here talks to Cloud except :func:`acting_organization`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_private_state import write_private_json

AGENTS_DIRECTORY = "agents"
_SCHEMA_VERSION = 1
# The slugs Cloud accepts (`cayu.cli._cloud_project.is_application_slug`), so one names a file.
_SLUG = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
# What an Organization id may be (WorkOS's `org_...`), so one is safe to print and to put in a
# command; Cloud's other answers are refused.
_ORGANIZATION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _printable_name(value: object) -> str | None:
    """An Organization name fit to print: its Organization's admins choose it, so one with
    control characters (a terminal escape, say) is dropped and the id names it instead."""

    if isinstance(value, str) and value.strip() and value.isprintable() and len(value) <= 200:
        return value
    return None


@dataclass(frozen=True, slots=True)
class Organization:
    """A Cayu Cloud install and one of its Organizations; ``name`` is None when not known."""

    api_url: str
    id: str
    name: str | None

    def describe(self) -> str:
        named = self.id if self.name is None else f"{self.name} ({self.id})"
        return f"Organization {named} on {self.api_url}"

    def public(self) -> dict[str, Any]:
        return {"api_url": self.api_url, "id": self.id, "name": self.name}


@dataclass(frozen=True, slots=True)
class AgentHome:
    """Where ``application`` deploys, as this machine recorded it."""

    application: str
    organization: Organization
    recorded_at: str


class AgentOrganizationError(CloudApiError):
    """A command refused because it would act in another Organization than intended."""

    def __init__(self, category: str, message: str, *, details: dict[str, object]) -> None:
        super().__init__(category, message)
        self.details = details


def acting_organization(client: CloudApiClient, *, api_url: str) -> Organization:
    """The Organization ``client``'s credential acts in at ``api_url``, from Cloud itself.

    `GET /v1/me` names it for a login and an Organization API key alike. Its name comes from
    `GET /v1/me/organizations`, which only a signed-in person can read; without it the name is
    None and the id still identifies the Organization.
    """

    identity = client.request("GET", "/v1/me")
    organization_id = identity.get("organization_id")
    if not isinstance(organization_id, str) or _ORGANIZATION_ID.fullmatch(organization_id) is None:
        raise CloudApiError(
            "api_response_invalid", "Cayu Cloud didn't say which Organization this acts in."
        )
    name = None
    if identity.get("authentication") != "api_key":
        try:
            listing = client.request("GET", "/v1/me/organizations")
        except CloudApiError:
            listing = {}
        items = listing.get("items")
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict) and item.get("id") == organization_id:
                name = _printable_name(item.get("name"))
                break
    return Organization(api_url=api_url, id=organization_id, name=name)


def require_intended_organization(intended: str | None, acting: Organization) -> None:
    """Refuse when ``--organization`` names another Organization than the one acted in."""

    if intended is None or intended == acting.id:
        return
    raise AgentOrganizationError(
        "organization_mismatch",
        f"This command would act in {acting.describe()}, not in Organization {intended}. "
        f"Switch with `cayu cloud organizations use {intended}` (or "
        f"`cayu cloud login --organization {intended}`), then run it again.",
        details={"acting_organization": acting.public(), "intended_organization_id": intended},
    )


def find_agent_home(config_directory: Path, reference: str) -> AgentHome | None:
    """The recorded home of the Agent ``reference`` names (its slug, or the name it was
    deployed with), or None when this machine never recorded one.

    It only explains a reference Cloud doesn't resolve: a command checks the Agent Cloud
    resolves with :func:`require_agent_home`. A record it can't read is passed over here.
    """

    directory = config_directory / AGENTS_DIRECTORY
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return None
    for path in paths:
        try:
            home = _read_agent_home(path)
            payload = json.loads(path.read_text())
        except (AgentOrganizationError, OSError, ValueError):
            continue
        if home is not None and reference in {home.application, payload.get("name")}:
            return home
    return None


def require_agent_home(
    config_directory: Path,
    application: str,
    acting: Organization,
    *,
    intended: str | None,
    login_command: Callable[[str], str],
) -> AgentHome | None:
    """Refuse to change Agent ``application`` (its slug, which is its id) outside the Cloud and
    Organization it deploys to. A record that can't be read refuses too.

    ``intended`` is the command's ``--organization``: naming the Organization acted in (which
    :func:`require_intended_organization` has checked) is how a person says they mean to act
    on the Agent there anyway. ``login_command(api_url)`` is the command that signs in to an
    install. The record, or None when there is none, is returned.
    """

    home = (
        _read_agent_home(config_directory / AGENTS_DIRECTORY / f"{application}.json")
        if _SLUG.fullmatch(application) is not None
        else None
    )
    if home is None or _same(home.organization, acting) or intended == acting.id:
        return home
    recorded = home.organization
    if recorded.api_url == acting.api_url:
        switch = (
            f"Switch with `cayu cloud organizations use {recorded.id}` (or "
            f"`cayu cloud login --organization {recorded.id}`)"
        )
    else:
        switch = (
            f"Sign in there with `{login_command(recorded.api_url)} --organization {recorded.id}`"
        )
    raise AgentOrganizationError(
        "agent_organization_mismatch",
        f"Agent {home.application} deploys to {recorded.describe()}, but this command would act "
        f"in {acting.describe()}. {switch}, or pass `--organization {acting.id}` to act on "
        f"{home.application} in {acting.name or acting.id} anyway.",
        details={
            "acting_organization": acting.public(),
            "agent_organization": recorded.public(),
            "application": home.application,
        },
    )


def remember_agent_home(
    config_directory: Path,
    application: str,
    acting: Organization,
    *,
    name: str | None = None,
) -> str | None:
    """Record that ``application`` deploys to ``acting``, after a deploy there succeeded.

    Returns the warning to show when that is news: the Agent had no record yet (an Agent
    deployed before this machine kept records, or a new one), or it moved there from another
    Organization or Cloud with ``--organization``. Returns None when the record already said so.
    A record that can't be written is a warning too; the deploy itself succeeded.
    """

    path = config_directory / AGENTS_DIRECTORY / f"{application}.json"
    try:
        previous = _read_agent_home(path)
    except AgentOrganizationError:
        previous = None
    if previous is not None and _same(previous.organization, acting):
        return None
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "application": application,
        "api_url": acting.api_url,
        "organization_id": acting.id,
        "organization_name": acting.name,
        "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if name is not None:
        payload["name"] = name
    try:
        _create_private_directories(config_directory, path.parent)
        write_private_json(path, payload)
    except (OSError, ValueError):
        return (
            f"Agent {application} deployed to {acting.describe()}, but this machine couldn't "
            f"record that in {path}, so a later command for it from another Organization isn't "
            "refused."
        )
    if previous is None:
        return (
            f"Recorded that Agent {application} deploys to {acting.describe()}. From now on a "
            f"command that changes {application} from another Organization or Cayu Cloud is "
            f"refused unless it passes `--organization`."
        )
    return (
        f"Agent {application} now deploys to {acting.describe()}; it deployed to "
        f"{previous.organization.describe()} before."
    )


def _same(left: Organization, right: Organization) -> bool:
    return left.api_url == right.api_url and left.id == right.id


def _read_agent_home(path: Path) -> AgentHome | None:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _unreadable(path) from exc
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise _unreadable(path) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
        raise _unreadable(path)
    application = payload.get("application")
    api_url = payload.get("api_url")
    organization_id = payload.get("organization_id")
    name = payload.get("organization_name")
    recorded_at = payload.get("recorded_at")
    if (
        not isinstance(application, str)
        or path.name != f"{application}.json"
        or not isinstance(api_url, str)
        or not api_url
        or not api_url.isprintable()
        or " " in api_url
        or not isinstance(organization_id, str)
        or _ORGANIZATION_ID.fullmatch(organization_id) is None
        or not (name is None or _printable_name(name) == name)
        or not isinstance(recorded_at, str)
    ):
        raise _unreadable(path)
    return AgentHome(
        application=application,
        organization=Organization(api_url=api_url, id=organization_id, name=name),
        recorded_at=recorded_at,
    )


def _unreadable(path: Path) -> AgentOrganizationError:
    return AgentOrganizationError(
        "agent_record_invalid",
        f"The record of where this Agent deploys, {path}, can't be read. Delete it; the next "
        "successful `cayu cloud deploy` records it again.",
        details={"path": str(path)},
    )


def _create_private_directories(config_directory: Path, directory: Path) -> None:
    """Create the config directory and ``agents/`` in it, each private to the user."""

    config_directory.parent.mkdir(parents=True, exist_ok=True)
    for level in (config_directory, directory):
        level.mkdir(mode=0o700, exist_ok=True)
