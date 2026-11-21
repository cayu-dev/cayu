"""Where connector tools come from: Cayu Cloud's catalog of a toolset, and where its tools run.

Cayu Cloud serves a connector package's tools by toolset (``cayu.connector-tools/v1``): an Agent
reads the catalog with its connections token and runs each tool on its own binding; a local
process reads it as the developer signed in with ``cayu cloud`` and runs the tools on the
binding of the Agent its project deploys to. Offline, a saved snapshot or the last catalog read
gives the tools' declarations; ``CAYU_CONNECTOR_TOOLS=snapshot`` pins a process (a test, CI) to
them, with no network.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import string
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx

from cayu.connections.client import (
    _SLOT,
    TOKEN_ENVIRONMENT,
    URL_ENVIRONMENT,
    ConnectionCallError,
    broker_answer,
)
from cayu.connectors import _common

if TYPE_CHECKING:
    from cayu.cli._cloud_api import CloudApiClient

#: The wire contract this Runtime reads and calls.
CONNECTOR_TOOLS_API = "cayu.connector-tools/v1"
#: ``snapshot`` pins the process to the project's saved catalogs, then the cache, with no network:
#: for tests and CI.
MODE_ENVIRONMENT = "CAYU_CONNECTOR_TOOLS"
#: How long a starting Agent keeps asking an unreachable Cayu Cloud for a catalog.
STARTUP_RETRY_SECONDS = 20.0
_FIRST_RETRY_SECONDS = 0.5
CATALOG_TIMEOUT_SECONDS = 10.0
#: How long one run waits: Cloud's 25-second deadline, with room for the way there and back.
RUN_TIMEOUT_SECONDS = 30.0
#: Where `cayu cloud connectors tools TOOLSET --save` keeps a project's snapshot.
SNAPSHOT_DIRECTORY = Path(".cayu") / "connector-tools"
_PROJECT_MANIFEST = "cayu-cloud.toml"
#: A toolset id, as Cayu Cloud names them: its tools' names start with it and ``_``.
TOOLSET_PATTERN = re.compile(r"^[a-z][a-z0-9]{0,31}$")
_EFFECTS = frozenset({"none", "idempotent", "external"})
_MAX_DESCRIPTION = 1024
_MAX_APPROVAL = 512
#: Cloud's digest of a tool: SHA-256, in hex.
_DIGEST = re.compile(r"[0-9a-fA-F]{64}")
_MAX_SCHEMA_BYTES = 16 * 1024
_MAX_TOOLS = 128
_TRANSIENT_STATUSES = frozenset({408, 425, 429})

logger = logging.getLogger("cayu.connections")

#: The transport every catalog read and run uses; None is HTTPX's own.
_transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None
_sleep: Callable[[float], None] = time.sleep
_clock: Callable[[], float] = time.monotonic


class ConnectorToolsError(RuntimeError):
    """Connector tools couldn't be loaded; ``code`` says why (``connector_catalog_unreachable``,
    ``connector_catalog_refused``, ``connector_catalog_invalid``, or Cayu Cloud's own).
    Provisional."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{message} ({code})")
        self.code = code


class _CloudUnreachable(ConnectorToolsError):
    """Cayu Cloud didn't answer the catalog: a member process falls back to a saved one."""


class CredentialUnavailable(Exception):
    """The credential a run carries couldn't be had or renewed (a login that can't refresh)."""


class RunTargetUnavailable(Exception):
    """A local run has no Agent's bindings to use: ``code`` says why
    (``agent_not_found``, ``agent_ambiguous`` or ``connection_failed``)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolDescriptor:
    """One tool of a toolset, as Cayu Cloud describes it."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    effect: Literal["none", "idempotent", "external"]
    scopes: tuple[str, ...]
    #: The approval reason's template, naming arguments (``"Post to {channel}."``).
    approval: str | None
    max_answer_bytes: int
    #: Cayu Cloud's digest of what the Agent's policy and slot were built from; sent back on
    #: every run, so a changed tool is refused (``tool_changed``) until the Agent restarts.
    digest: str


@dataclass(frozen=True)
class Catalog:
    """A toolset's tools as Cayu Cloud offers them (``cayu.connector-tools/v1``)."""

    toolset: str
    package: str
    version: str
    connectors: tuple[str, ...]
    default_slot: str
    description: str
    tools: tuple[ToolDescriptor, ...]
    #: The catalog as Cloud answered it, which a snapshot or the cache keeps.
    document: Mapping[str, Any]


@dataclass(frozen=True)
class Bearer:
    """The credential a request to Cayu Cloud carries, renewed once when Cloud refuses it."""

    token: Callable[[], str]
    renew: Callable[[str], str] | None = None


@dataclass(frozen=True)
class Origin:
    """Where a toolset's catalog came from and where its tools run."""

    kind: Literal["agent", "member", "snapshot"]
    #: Cayu Cloud's URL; None offline.
    cloud_url: str | None
    #: The Agent whose bindings the tools use, when known.
    agent: str | None = None
    #: Where ``{slot}/tools/{tool}`` runs a tool, found when a tool first runs (it may ask
    #: Cloud which Agent the project names); None, or answering None, when no run can be sent.
    run_url: Callable[[], str | None] | None = None
    bearer: Bearer | None = None
    #: The file an offline catalog came from.
    path: Path | None = None


@dataclass(frozen=True)
class Loaded:
    """A toolset's catalog, None when Cayu Cloud no longer offers it, and its origin."""

    toolset: str
    catalog: Catalog | None
    origin: Origin


#: Each toolset read by this process: a process reads a toolset's catalog once.
_read: dict[str, Loaded] = {}


def load(toolset: str) -> Loaded:
    """The toolset's catalog, from the first source that applies (see ``cayu.connections``)."""

    if not isinstance(toolset, str) or not TOOLSET_PATTERN.fullmatch(toolset):
        raise ValueError(f"{toolset!r} isn't a toolset id.")
    kept = _read.get(toolset)
    if kept is None:
        kept = _read[toolset] = _loaded(toolset, os.environ)
    return kept


def summary() -> str | None:
    """One line per source of the toolsets this process read: which versions, from which Cayu
    Cloud and for which Agent, so the Cloud in use (staging or production) is visible."""

    groups: dict[tuple[str, str | None], list[str]] = {}
    for each in _read.values():
        origin = each.origin
        if origin.path is None:
            where = f"from {origin.cloud_url}"
        elif origin.kind == "member" and origin.cloud_url is None:
            where = f"from the saved catalog {origin.path} (the Cayu Cloud sign-in can't be used)"
        elif origin.cloud_url is None:
            where = (
                f"from the saved catalog {origin.path} (offline: runs answer no_connection_client)"
            )
        else:
            where = f"from the saved catalog {origin.path} ({origin.cloud_url} was unreachable)"
        named = (
            f"{each.toolset} (unavailable)"
            if each.catalog is None
            else f"{each.toolset} {each.catalog.version}"
        )
        groups.setdefault((where, origin.agent), []).append(named)
    if not groups:
        return None
    return "\n".join(
        f"Connector tools: {', '.join(names)} {where}"
        + (f", Agent {agent}" if agent is not None else "")
        for (where, agent), names in groups.items()
    )


def _loaded(toolset: str, environ: Mapping[str, str]) -> Loaded:
    mode = environ.get(MODE_ENVIRONMENT, "").strip()
    if mode not in {"", "snapshot"}:
        raise ValueError(f"{MODE_ENVIRONMENT} is unset or snapshot, not {mode!r}.")
    if mode == "snapshot":
        offline = _offline(toolset)
        if offline is None:
            raise ValueError(
                f"{MODE_ENVIRONMENT}=snapshot, and the connector tools {toolset} have no saved "
                f"catalog: save one with `cayu cloud connectors tools {toolset} --save`."
            )
        return offline
    url = environ.get(URL_ENVIRONMENT, "").strip().rstrip("/")
    if url:
        token = environ.get(TOKEN_ENVIRONMENT, "").strip()
        if not token:
            raise ValueError(f"{URL_ENVIRONMENT} is set without {TOKEN_ENVIRONMENT}.")
        origin = Origin(
            kind="agent",
            cloud_url=url,
            agent=environ.get("CAYU_CLOUD_APPLICATION_ID") or None,
            run_url=lambda: url,
            bearer=Bearer(token=lambda: token),
        )
        return Loaded(toolset, _fetched(f"{url}/toolsets/{toolset}", origin, patient=True), origin)
    from cayu.cli._cloud_api import CloudApiError
    from cayu.cli._cloud_auth import CloudAuthError

    try:
        member = _Member.signed_in()
    except (CloudAuthError, CloudApiError) as unusable:
        # A saved login that can't be refreshed, or an API key that can't be read: the saved
        # catalog serves, and its tools sign in again when they run.
        if isinstance(unusable, CloudApiError) and unusable.category != "api_key_unavailable":
            raise
        offline = _offline(toolset)
        if offline is None or offline.catalog is None:
            raise
        logger.warning(
            "The Cayu Cloud sign-in can't be used (%s); the connector tools %s come from the "
            "saved catalog %s.",
            unusable.category,
            toolset,
            offline.origin.path,
        )
        agent = linked_agent()
        lazy = _Member(None)
        origin = Origin(
            kind="member",
            cloud_url=None,
            agent=agent,
            run_url=_MemberRuns(lazy, agent),
            bearer=lazy.bearer(),
            path=offline.origin.path,
        )
        return Loaded(toolset, offline.catalog, origin)
    if member is not None:
        client = member.client()
        agent = linked_agent()
        origin = Origin(
            kind="member",
            cloud_url=client.api_url,
            agent=agent,
            run_url=_MemberRuns(member, agent),
            bearer=member.bearer(),
        )
        url = f"{client.api_url}/v1/connector-tools/{toolset}"
        try:
            catalog = _fetched(url, origin, patient=False)
        except (_CloudUnreachable, CredentialUnavailable) as unreachable:
            offline = _offline(toolset)
            if offline is None or offline.catalog is None:
                raise
            logger.warning(
                "Cayu Cloud at %s can't be read (%s); the connector tools %s come from the "
                "saved catalog %s.",
                client.api_url,
                unreachable,
                toolset,
                offline.origin.path,
            )
            return Loaded(
                toolset,
                offline.catalog,
                dataclasses.replace(origin, path=offline.origin.path),
            )
        if catalog is not None:
            save(cache_path(toolset), catalog.document)
        return Loaded(toolset, catalog, origin)
    offline = _offline(toolset)
    if offline is not None:
        return offline
    raise ValueError(
        f"The connector tools {toolset} have no catalog: run in Cayu Cloud "
        f"({URL_ENVIRONMENT} and {TOKEN_ENVIRONMENT}), sign in with `cayu cloud login` (or set "
        "CAYU_CLOUD_API_KEY or select a context), or save a snapshot with `cayu cloud "
        f"connectors tools {toolset} --save`."
    )


class _Member:
    """The developer's Cayu Cloud client, signed in when first needed: a process that started
    without a usable sign-in signs in again when a tool runs."""

    def __init__(self, client: CloudApiClient | None) -> None:
        self._client = client

    @classmethod
    def signed_in(cls) -> _Member | None:
        """The client `cayu cloud` commands use; None when nothing signs in."""

        from cayu.cli import cloud as cloud_cli

        client = cloud_cli.member_client(timeout_seconds=CATALOG_TIMEOUT_SECONDS)
        return None if client is None else cls(client)

    def client(self) -> CloudApiClient:
        if self._client is None:
            from cayu.cli._cloud_api import CloudApiError
            from cayu.cli._cloud_auth import CloudAuthError

            try:
                signed_in = _Member.signed_in()
            except (CloudAuthError, CloudApiError) as exc:
                raise CredentialUnavailable(str(exc)) from exc
            if signed_in is None:
                raise CredentialUnavailable("Nothing signs in to Cayu Cloud here.")
            self._client = signed_in.client()
        return self._client

    def _token(self) -> str:
        client = self.client()
        return client.api_key if client.api_key_provider is None else client.api_key_provider()

    def _renewed(self, rejected: str) -> str:
        client = self.client()
        renew = client.rejected_api_key_provider
        return rejected if renew is None else renew(rejected)

    def bearer(self) -> Bearer:
        return Bearer(token=self._token, renew=self._renewed)


class _MemberRuns:
    """Where a local process runs tools: the binding of the Agent its project names, found
    when a tool first runs, so a process starts even while Cloud is unreachable."""

    def __init__(self, member: _Member, agent: str | None) -> None:
        self._member = member
        self._agent = agent
        self._url: str | None = None

    def __call__(self) -> str | None:
        if self._url is None and self._agent is not None:
            from cayu.cli import cloud as cloud_cli
            from cayu.cli._cloud_api import CloudApiError
            from cayu.cli._cloud_auth import CloudAuthError

            client = self._member.client()
            try:
                application_id = cloud_cli.resolve_application_id(client, self._agent)
            except CloudAuthError as exc:
                raise CredentialUnavailable(str(exc)) from exc
            except CloudApiError as exc:
                if exc.category == "application_not_found":
                    raise RunTargetUnavailable(
                        "agent_not_found",
                        f"Cayu Cloud at {client.api_url} has no Agent {self._agent}, which "
                        "this project's cayu-cloud.toml names.",
                    ) from exc
                if exc.category == "application_ambiguous":
                    raise RunTargetUnavailable(
                        "agent_ambiguous",
                        f"Several Agents in Cayu Cloud answer to {self._agent}, which this "
                        "project's cayu-cloud.toml names.",
                    ) from exc
                raise RunTargetUnavailable(
                    "connection_failed",
                    f"Cayu Cloud couldn't say which Agent this project names ({exc.category}).",
                ) from exc
            self._url = f"{client.api_url}/v1/applications/{application_id}/connections"
        return self._url


def _offline(toolset: str) -> Loaded | None:
    """The project's saved catalog of ``toolset``, else the cached one; None when neither is."""

    for path in (snapshot_path(toolset), cache_path(toolset)):
        if path is None or not path.is_file():
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConnectorToolsError(
                "connector_catalog_invalid", f"The saved catalog {path} can't be read."
            ) from exc
        catalog = catalog_from(document, toolset)
        return Loaded(toolset, catalog, Origin(kind="snapshot", cloud_url=None, path=path))
    return None


def _fetched(url: str, origin: Origin, *, patient: bool) -> Catalog | None:
    """The catalog at ``url``; None when Cloud no longer offers the toolset. A ``patient``
    reader (an Agent) asks again while Cloud is unreachable, for up to
    ``STARTUP_RETRY_SECONDS``, honouring a ``Retry-After``; any other gives up at once."""

    assert origin.bearer is not None and origin.cloud_url is not None
    toolset = url.rsplit("/", 1)[-1]
    deadline = _clock() + STARTUP_RETRY_SECONDS
    delay = _FIRST_RETRY_SECONDS
    while True:
        try:
            document = get(url, origin.bearer)
        except ConnectionCallError as error:
            if error.code == "connector_unavailable":
                logger.warning(
                    "connector_tools_unavailable: Cayu Cloud at %s offers no toolset %s; "
                    "starting without its tools.",
                    origin.cloud_url,
                    toolset,
                )
                return None
            if not _transient(error):
                raise ConnectorToolsError(
                    error.code or "connector_catalog_refused",
                    f"Cayu Cloud at {origin.cloud_url} refused the catalog of {toolset}: {error}",
                ) from None
            remaining = deadline - _clock()
            if not patient or remaining <= 0:
                raise _CloudUnreachable(
                    "connector_catalog_unreachable",
                    f"Cayu Cloud at {origin.cloud_url} didn't answer the catalog of {toolset}"
                    + (f" within {STARTUP_RETRY_SECONDS:g} seconds" if patient else "")
                    + f": {error}",
                ) from None
            _sleep(min(max(delay, error.retry_after or 0.0), remaining))
            delay *= 2
            continue
        return catalog_from(document, toolset)


def _transient(error: ConnectionCallError) -> bool:
    return error.status is None or error.status >= 500 or error.status in _TRANSIENT_STATUSES


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def get(url: str, bearer: Bearer) -> dict[str, Any]:
    """Cayu Cloud's JSON object at ``url``; its refusal as a ``ConnectionCallError``."""

    transport = _transport if isinstance(_transport, httpx.BaseTransport) else None
    try:
        with httpx.Client(
            transport=transport, timeout=CATALOG_TIMEOUT_SECONDS, follow_redirects=False
        ) as client:
            token = _credential_now(bearer.token)
            response = client.get(url, headers=_headers(token))
            if response.status_code == 401 and bearer.renew is not None:
                renewed = _credential_now(bearer.renew, token)
                if renewed != token:
                    response = client.get(url, headers=_headers(renewed))
    except httpx.HTTPError as exc:
        raise ConnectionCallError(f"Cayu Cloud is unreachable ({type(exc).__name__}).") from exc
    return broker_answer(response)


async def post(url: str, bearer: Bearer, body: Mapping[str, Any]) -> dict[str, Any]:
    """Cayu Cloud's JSON answer to ``body`` at ``url``; its refusal as a
    ``ConnectionCallError``."""

    transport = _transport if isinstance(_transport, httpx.AsyncBaseTransport) else None
    try:
        async with httpx.AsyncClient(
            transport=transport, timeout=RUN_TIMEOUT_SECONDS, follow_redirects=False
        ) as client:
            token = await _credential(bearer.token)
            response = await client.post(url, headers=_headers(token), json=body)
            if response.status_code == 401 and bearer.renew is not None:
                renewed = await _credential(bearer.renew, token)
                if renewed != token:
                    response = await client.post(url, headers=_headers(renewed), json=body)
    except httpx.HTTPError as exc:
        raise ConnectionCallError(f"Cayu Cloud is unreachable ({type(exc).__name__}).") from exc
    return broker_answer(response)


def _credential_now(source: Callable[..., str], *arguments: str) -> str:
    """A credential from ``source``; any failure to have it is ``CredentialUnavailable``."""

    try:
        return source(*arguments)
    except CredentialUnavailable:
        raise
    except Exception as exc:
        raise CredentialUnavailable(str(exc)) from exc


async def _credential(source: Callable[..., str], *arguments: str) -> str:
    """A credential from ``source``, read off the event loop (a login may refresh over the
    network); any failure to have it is ``CredentialUnavailable``."""

    return await asyncio.to_thread(_credential_now, source, *arguments)


def catalog_from(document: object, toolset: str) -> Catalog:
    """``document`` as a toolset's catalog, once it is one this Runtime can serve."""

    try:
        return _catalog(document, toolset)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConnectorToolsError(
            "connector_catalog_invalid", f"The catalog of {toolset} is malformed: {exc}"
        ) from None


def _catalog(given: object, toolset: str) -> Catalog:
    if not isinstance(given, dict):
        raise TypeError("it isn't a JSON object")
    document = cast("dict[str, Any]", given)
    if document.get("api") != CONNECTOR_TOOLS_API:
        raise ValueError(f"its api isn't {CONNECTOR_TOOLS_API}")
    if document.get("toolset") != toolset:
        raise ValueError("it is another toolset's")
    connectors = _strings(document["connectors"], "connectors")
    default_slot = _text(document["default_slot"], "default_slot")
    if not connectors or not all(_SLOT.fullmatch(item) for item in (*connectors, default_slot)):
        raise ValueError("its connectors or default_slot aren't connector or slot names")
    tools = document["tools"]
    if not isinstance(tools, list) or len(tools) > _MAX_TOOLS:
        raise ValueError(f"its tools aren't a list of at most {_MAX_TOOLS}")
    described = tuple(_tool(item, toolset) for item in tools)
    if len({tool.name for tool in described}) != len(described):
        raise ValueError("it names a tool twice")
    return Catalog(
        toolset=toolset,
        package=_text(document["package"], "package"),
        version=_text(document["version"], "version"),
        connectors=connectors,
        default_slot=default_slot,
        description=_text(document.get("description") or toolset, "description"),
        tools=described,
        document=document,
    )


def _tool(given: object, toolset: str) -> ToolDescriptor:
    if not isinstance(given, dict):
        raise TypeError("a tool isn't a JSON object")
    item = cast("dict[str, Any]", given)
    name = _text(item["name"], "a tool's name")
    if not re.fullmatch(rf"{re.escape(toolset)}_[a-z0-9_]{{1,56}}", name):
        raise ValueError(f"the tool name {name!r} isn't {toolset}_ and 1-56 of [a-z0-9_]")
    description = _text(item["description"], f"{name}'s description")
    schema = item["input_schema"]
    if (
        not isinstance(schema, dict)
        or schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or len(json.dumps(schema).encode()) > _MAX_SCHEMA_BYTES
        or len(description) > _MAX_DESCRIPTION
    ):
        raise ValueError(
            f"{name}'s description is over {_MAX_DESCRIPTION} characters, or its input schema "
            f"isn't an object of at most {_MAX_SCHEMA_BYTES} bytes with additionalProperties "
            "false"
        )
    properties = schema.get("properties", {})
    effect = item["effect"]
    if effect not in _EFFECTS:
        raise ValueError(f"{name}'s effect isn't one of {', '.join(sorted(_EFFECTS))}")
    approval = item.get("approval")
    if approval is not None:
        if len(_text(approval, f"{name}'s approval")) > _MAX_APPROVAL:
            raise ValueError(f"{name}'s approval is over {_MAX_APPROVAL} characters")
        unknown = [
            field
            for field in placeholders(approval)
            if not isinstance(properties, dict) or field not in properties
        ]
        if unknown:
            raise ValueError(f"{name}'s approval names {unknown[0]}, which isn't an argument")
    elif effect != "none":
        raise ValueError(f"{name} has an effect and no approval")
    answer = item["max_answer_bytes"]
    if type(answer) is not int or not 0 < answer <= _common.ANSWER_BUDGET_BYTES:
        raise ValueError(f"{name}'s max_answer_bytes isn't 1-{_common.ANSWER_BUDGET_BYTES}")
    return ToolDescriptor(
        name=name,
        description=description,
        input_schema=schema,
        effect=effect,
        scopes=_strings(item.get("scopes", []), f"{name}'s scopes"),
        approval=approval,
        max_answer_bytes=answer,
        digest=_digest(item["digest"], name),
    )


def placeholders(template: str) -> tuple[str, ...]:
    """The argument names an approval template quotes (``{channel}``); a template with any
    other replacement field is refused."""

    names = []
    for _, field, spec, conversion in string.Formatter().parse(template):
        if field is None:
            continue
        if not field.isidentifier() or spec or conversion:
            raise ValueError(f"the approval template field {{{field}}} isn't an argument name")
        names.append(field)
    return tuple(names)


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"{name}'s digest isn't a SHA-256 in hex")
    return value


def _text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} isn't text")
    return value


def _strings(value: object, what: str) -> tuple[str, ...]:
    strings = (
        tuple(item for item in value if isinstance(item, str)) if isinstance(value, list) else ()
    )
    if not isinstance(value, list) or len(strings) != len(value):
        raise ValueError(f"{what} isn't a list of strings")
    return strings


def project_root() -> Path:
    """The nearest folder at or above the working directory with a ``cayu-cloud.toml`` or a
    ``pyproject.toml``; the working directory when none has."""

    directory = Path.cwd().resolve()
    for candidate in (directory, *directory.parents):
        if (candidate / _PROJECT_MANIFEST).is_file() or (candidate / "pyproject.toml").is_file():
            return candidate
    return directory


def linked_agent() -> str | None:
    """The Agent the project deploys to: its ``cayu-cloud.toml``'s ``application``."""

    path = project_root() / _PROJECT_MANIFEST
    try:
        application = tomllib.loads(path.read_text(encoding="utf-8")).get("application")
    except (OSError, ValueError):
        return None
    return application if isinstance(application, str) and application else None


def cache_path(toolset: str) -> Path | None:
    """Where the last catalog of ``toolset`` read here is kept: the user's cache directory,
    never the project's."""

    from cayu.cli._release_check import user_cache_directory

    directory = user_cache_directory()
    return None if directory is None else directory / "connector-tools" / f"{toolset}.json"


def snapshot_path(toolset: str) -> Path:
    """Where the project keeps its saved catalog of ``toolset``."""

    return project_root() / SNAPSHOT_DIRECTORY / f"{toolset}.json"


def save(path: Path | None, document: Mapping[str, Any]) -> bool:
    """Write a catalog to ``path``, whole or not at all; whether it was written."""

    if path is None:
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        partial.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", "utf-8")
        partial.replace(path)
    except OSError:
        logger.warning("The connector tools catalog couldn't be kept at %s.", path)
        return False
    return True
