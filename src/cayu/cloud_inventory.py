"""Register a Release's tools and knowledge with Cayu Cloud and report their use.

This is the Runtime side of Cloud's ``cayu.agent-registry/v1`` contract. The outbound
credential is used only for this Agent's registry routes. Only allowlisted declarations
cross this boundary: no prompt, argument schema, argument, result, provider option,
metadata value, knowledge text or conversation-derived knowledge. The registration also
carries the Release's managed settings schema: the settings its parts declared, with the
code's values as defaults (``docs/managed-settings.md``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from cayu._version import __version__
from cayu.cloud_reporting import _store_identity
from cayu.events import EventType
from cayu.knowledge.records import KnowledgeActorType, KnowledgeStatus, KnowledgeVisibility
from cayu.knowledge.search import KnowledgeListQuery, encode_knowledge_list_cursor
from cayu.knowledge.usage import knowledge_usage_contribution
from cayu.sessions.event_queries import EventOrder, EventQuery
from cayu.tool_inventory import (
    APPROVALS as _APPROVALS,
)
from cayu.tool_inventory import (
    EFFECTS as _EFFECTS,
)
from cayu.tool_inventory import (
    TOOL_USAGE_EVENT_TYPES,
    ToolSourceMetadata,
    classify_tool_source,
    tool_usage_contribution,
)
from cayu.tool_inventory import (
    ToolApproval as _Approval,
)
from cayu.tool_inventory import (
    ToolEffect as _Effect,
)
from cayu.tool_inventory import (
    ToolSource as CloudToolSource,
)
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.applications import CayuApp

_ModelT = TypeVar("_ModelT", bound=BaseModel)

logger = logging.getLogger(__name__)
CLOUD_REGISTRY_CONTRACT = "cayu.agent-registry/v1"
CLOUD_REGISTRY_TOKEN_ENV = "CAYU_CLOUD_REGISTRY_TOKEN"
# Cloud's limits (agent-registry-v1). Cloud refuses longer values instead of truncating
# them, so Runtime fits every value before sending it.
MAX_BODY_BYTES = 1024 * 1024
MAX_TOOLS = 500
MAX_KNOWLEDGE_ENTRIES = 500
MAX_AGENTS = 100
MAX_NAME = 128
MAX_ENTRY_ID_BYTES = 256
MAX_DESCRIPTION = 2_000
MAX_TITLE = 300
MAX_VERSION = 64
# Knowledge is listed a page at a time; a store with more shared entries than this is
# reported as truncated instead of being scanned to its end.
_KNOWLEDGE_PAGE = 100
_MAX_KNOWLEDGE_SCANNED = 10_000
_USAGE_PAGE = 100
_MAX_RESPONSE_BYTES = 64 * 1024
_RETRY_SECONDS = 30.0
# Knowledge everyone using the Agent shares. Per-user, per-session and per-task entries
# hold one person's or one conversation's knowledge and never leave the Runtime.
_SHARED_VISIBILITIES = (
    KnowledgeVisibility.GLOBAL,
    KnowledgeVisibility.ORGANIZATION,
    KnowledgeVisibility.PROJECT,
    KnowledgeVisibility.WORKSPACE,
)
# Application-authored entries that Cayu's curator or maintenance pipeline wrote from
# conversation evidence or a model's proposal carry these metadata keys.
_DERIVED_KNOWLEDGE_METADATA_KEYS = frozenset({"cayu_curator", "cayu_knowledge_maintenance"})
_USAGE_EVENT_TYPES = (*TOOL_USAGE_EVENT_TYPES, EventType.KNOWLEDGE_LOOKUP)
# Cloud allows newlines and tabs in descriptions and titles, never other control characters.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_LINE_BREAKS = re.compile(r"[\x00-\x1f\x7f]")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class CloudTool(_Model):
    name: str
    description: str | None = None
    agents: tuple[str, ...]
    source: CloudToolSource
    effect: _Effect
    approval: _Approval


class CloudKnowledgeEntry(_Model):
    id: str
    title: str | None = None
    kind: str
    # Only active entries an application registered are ever reported.
    status: Literal["active"] = "active"
    created_by: Literal["app"] = "app"
    updated_at: datetime


class CloudKnowledge(_Model):
    available: bool
    truncated: bool
    entries: tuple[CloudKnowledgeEntry, ...] = ()


class CloudRuntime(_Model):
    version: str = __version__


class CloudRegistration(_Model):
    contract: str = CLOUD_REGISTRY_CONTRACT
    runtime: CloudRuntime = Field(default_factory=CloudRuntime)
    # The session store's identity: Cloud's cursor counts this store's sequence.
    store_id: str | None = None
    start_cursor: int | None = Field(default=None, ge=0)
    # A cursor Cloud answered that this store can't reach (its newest event is below it).
    replaces_cursor: int | None = Field(default=None, ge=0)
    agents: tuple[str, ...]
    tools: tuple[CloudTool, ...]
    knowledge: CloudKnowledge
    # The Release's cayu.managed-settings-schema/v1 document (docs/managed-settings.md).
    # A plain mapping, so that encoding keeps each setting's null default.
    settings_schema: dict[str, Any] | None = None


class CloudToolUsage(_Model):
    name: str
    calls: int = 0
    failures: int = 0
    approved: int = 0
    rejected: int = 0
    last_used_at: datetime | None = None


class CloudKnowledgeUsage(_Model):
    id: str
    lookups: int = 0
    last_used_at: datetime | None = None


class CloudUsageReport(_Model):
    contract: str = CLOUD_REGISTRY_CONTRACT
    store_id: str | None = None
    after: int | None = Field(ge=0)
    through: int = Field(ge=1)
    tools: tuple[CloudToolUsage, ...] = ()
    knowledge: tuple[CloudKnowledgeUsage, ...] = ()


def _encode(value: BaseModel) -> bytes:
    return json.dumps(
        value.model_dump(mode="json", exclude_none=True), separators=(",", ":")
    ).encode()


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _text(value: str | None, limit: int) -> str | None:
    """Fit free text to Cloud's rules, or None when nothing printable is left."""
    if value is None:
        return None
    cleaned = _CONTROL.sub(" ", value).strip()[:limit].strip()
    return cleaned or None


def _line(value: str | None, limit: int) -> str | None:
    if value is None:
        return None
    cleaned = _LINE_BREAKS.sub(" ", value).strip()
    return cleaned if cleaned and len(cleaned) <= limit else None


@dataclass(frozen=True)
class _Registered:
    """What one registration declared, so use is counted only for it."""

    tools: frozenset[str] = frozenset()
    # Knowledge lookups name (namespace, entry id); Cloud keys entries by id alone.
    knowledge: frozenset[tuple[str, str]] = frozenset()


async def _registered_knowledge(
    app: CayuApp,
    sanitize: Callable[[str | None], str | None],
) -> tuple[CloudKnowledge, dict[str, str]]:
    stores = [(app.knowledge_store, app.knowledge_access_scope)]
    stores.extend(
        (
            registration.environment.knowledge_store,
            registration.environment.knowledge_access_scope,
        )
        for _, registration in sorted(app._environment_registry.registrations.items())
        if registration.factory is None
    )
    available = False
    truncated = False
    scanned = 0
    entries: list[CloudKnowledgeEntry] = []
    # Each reported entry id's namespace.
    namespaces: dict[str, str] = {}
    now = datetime.now(UTC)
    for store, scope in stores:
        if store is None:
            continue
        available = True
        cursor = None
        while not truncated:
            result = await store.list_entries(
                KnowledgeListQuery(
                    cursor=cursor,
                    limit=_KNOWLEDGE_PAGE,
                    max_bytes=MAX_BODY_BYTES,
                    statuses=[KnowledgeStatus.ACTIVE],
                    visibilities=list(_SHARED_VISIBILITIES),
                    include_expired=False,
                ),
                access_scope=scope,
            )
            for item in result.entries:
                entry = item.entry
                scanned += 1
                # Only what application code registered: never a model-written memory,
                # a person's entry, or knowledge Cayu derived from conversations.
                if (
                    entry.created_by_type is not KnowledgeActorType.APP
                    or entry.status is not KnowledgeStatus.ACTIVE
                    or entry.visibility not in _SHARED_VISIBILITIES
                    or (entry.expires_at is not None and _aware(entry.expires_at) <= now)
                    or _DERIVED_KNOWLEDGE_METADATA_KEYS.intersection(entry.metadata)
                    or entry.id in namespaces
                    or sanitize(entry.id) != entry.id
                    or len(entry.id.encode()) > MAX_ENTRY_ID_BYTES
                    or _LINE_BREAKS.search(entry.id)
                ):
                    continue
                kind = _line(sanitize(entry.kind), MAX_NAME)
                if kind is None:
                    continue
                if len(entries) == MAX_KNOWLEDGE_ENTRIES:
                    truncated = True
                    break
                namespaces[entry.id] = entry.namespace
                entries.append(
                    CloudKnowledgeEntry(
                        id=entry.id,
                        title=entry.title,
                        kind=kind,
                        updated_at=_aware(entry.updated_at),
                    )
                )
            if truncated or not result.truncated:
                break
            if scanned >= _MAX_KNOWLEDGE_SCANNED:
                truncated = True
                break
            if not result.entries:
                raise ValueError("Knowledge registration cannot advance its bounded page.")
            next_cursor = encode_knowledge_list_cursor(result.entries[-1].entry)
            if next_cursor == cursor:
                raise ValueError("Knowledge registration cursor did not advance.")
            cursor = next_cursor
    knowledge = CloudKnowledge(available=available, truncated=truncated, entries=tuple(entries))
    return knowledge, namespaces


async def _build_registration(
    app: CayuApp,
    *,
    project_root: Path | None,
    start_cursor: int | None,
    transport_redactor: SecretRedactor | None = None,
    store_id: str | None = None,
    replaces_cursor: int | None = None,
    settings_schema: bool = True,
) -> tuple[CloudRegistration, _Registered]:
    from cayu.mcp.tools import McpToolAdapter
    from cayu.runtime.manifest import describe_inventory_tools

    def sanitize(value: str | None) -> str | None:
        if value is None:
            return None
        redacted = app.redact_json(value)
        assert isinstance(redacted, str)
        return transport_redactor.redact_text(redacted) if transport_redactor else redacted

    source_metadata = ToolSourceMetadata()
    declarations = describe_inventory_tools(app, project_root=project_root)
    agents: list[str] = []
    tools: dict[str, CloudTool] = {}
    registrations = sorted(app._agent_registry.registrations.items())
    for agent_name, registration in registrations:
        name = _line(sanitize(agent_name), MAX_NAME)
        if name is None or len(agents) == MAX_AGENTS:
            continue
        agents.append(name)
        declared: list[tuple[str, CloudTool]] = []
        for tool, (tool_name, registered) in zip(
            declarations[agent_name], sorted(registration.tools.items()), strict=True
        ):
            description = (
                registered.tool.definition.description
                if isinstance(registered.tool, McpToolAdapter)
                else registered.description
            )
            declared.append(
                (
                    tool_name,
                    CloudTool(
                        name=tool_name,
                        description=description,
                        agents=(name,),
                        source=classify_tool_source(
                            tool,
                            registered.tool,
                            sanitize,
                            source_metadata=source_metadata,
                            name_limit=MAX_NAME,
                            version_limit=MAX_VERSION,
                        ),
                        effect=_EFFECTS.get(tool.effect, "external"),
                        approval=_APPROVALS.get(tool.policy_coverage, "unknown"),
                    ),
                )
            )
        for hosted in registration.hosted_tools:
            declared.append(
                (
                    hosted.type,
                    CloudTool(
                        name=hosted.type,
                        description="Model-provider hosted tool",
                        agents=(name,),
                        source=CloudToolSource(
                            kind="provider", provider=hosted.type.partition("_")[0]
                        ),
                        effect="none",
                        # Cayu never asks a person before a provider-hosted call.
                        approval="never",
                    ),
                )
            )
        for tool_name, tool in declared:
            if _line(sanitize(tool_name), MAX_NAME) != tool_name:
                continue
            existing = tools.get(tool_name)
            if existing is not None:
                tools[tool_name] = existing.model_copy(
                    update={
                        "agents": tuple(dict.fromkeys((*existing.agents, name))),
                        "approval": existing.approval
                        if existing.approval == tool.approval
                        else "unknown",
                        "effect": max(
                            (existing.effect, tool.effect),
                            key={"none": 0, "idempotent": 1, "external": 2}.__getitem__,
                        ),
                        "source": existing.source
                        if existing.source == tool.source
                        else CloudToolSource(kind="unknown"),
                        "description": existing.description
                        if existing.description == tool.description
                        else None,
                    }
                )
            elif len(tools) < MAX_TOOLS:
                tools[tool_name] = tool
    knowledge, namespaces = await _registered_knowledge(app, sanitize)
    registration_value = _redact(
        app,
        CloudRegistration(
            store_id=store_id,
            start_cursor=start_cursor,
            replaces_cursor=replaces_cursor,
            agents=tuple(agents),
            tools=tuple(tools.values()),
            knowledge=knowledge,
        ),
    )
    if transport_redactor is not None:
        registration_value = CloudRegistration.model_validate(
            transport_redactor.redact_json(registration_value.model_dump(mode="json"))
        )
    if settings_schema:
        # Added after redaction so that Cloud's recomputed digest matches; declaring a
        # setting already refuses one that contains a secret.
        registration_value = registration_value.model_copy(
            update={"settings_schema": app.managed_settings_schema()}
        )
    registration_value = _fit_registration(registration_value)
    return registration_value, _Registered(
        tools=frozenset(tool.name for tool in registration_value.tools),
        knowledge=frozenset(
            (namespaces[entry.id], entry.id)
            for entry in registration_value.knowledge.entries
            if entry.id in namespaces
        ),
    )


def _fit_registration(registration: CloudRegistration) -> CloudRegistration:
    """Keep the body under Cloud's limit: drop knowledge entries first, then shorten
    descriptions, never sending a body Cloud would refuse on every retry."""
    # Redaction can lengthen text, so the per-value limits apply again here.
    registration = registration.model_copy(
        update={
            "tools": tuple(
                tool.model_copy(update={"description": _text(tool.description, MAX_DESCRIPTION)})
                for tool in registration.tools
            ),
            "knowledge": registration.knowledge.model_copy(
                update={
                    "entries": tuple(
                        entry.model_copy(update={"title": _text(entry.title, MAX_TITLE)})
                        for entry in registration.knowledge.entries
                    )
                }
            ),
        }
    )
    if len(_encode(registration)) <= MAX_BODY_BYTES:
        return registration
    entries = registration.knowledge.entries
    while entries:
        entries = entries[: len(entries) // 2]
        candidate = registration.model_copy(
            update={
                "knowledge": registration.knowledge.model_copy(
                    update={"entries": entries, "truncated": True}
                )
            }
        )
        if len(_encode(candidate)) <= MAX_BODY_BYTES:
            return candidate
    registration = registration.model_copy(
        update={
            "knowledge": registration.knowledge.model_copy(
                update={"entries": (), "truncated": True}
            )
        }
    )
    limit = MAX_DESCRIPTION
    while len(_encode(registration)) > MAX_BODY_BYTES:
        limit //= 2
        registration = registration.model_copy(
            update={
                "tools": tuple(
                    tool.model_copy(
                        update={"description": _text(tool.description, limit) if limit else None}
                    )
                    for tool in registration.tools
                )
            }
        )
        if limit == 0:
            break
    # Associations can dominate the body even without descriptions: every tool
    # may belong to every specialist. Keep a deterministic prefix that fits;
    # _build_registration derives its usage allowlist from this final inventory.
    tools = registration.tools
    lower, upper = 0, len(tools)
    while lower < upper:
        count = (lower + upper + 1) // 2
        candidate = registration.model_copy(update={"tools": tools[:count]})
        if len(_encode(candidate)) <= MAX_BODY_BYTES:
            lower = count
        else:
            upper = count - 1
    return registration.model_copy(update={"tools": tools[:lower]})


async def cloud_registration(
    app: CayuApp, *, project_root: Path | None = None, start_cursor: int | None = None
) -> CloudRegistration:
    """Project public declarations; never export a whole app or knowledge record.

    Knowledge is limited to active, unexpired, shared entries that application code
    created. Model-written memories, people's entries, per-user, per-session and per-task
    entries, and entries Cayu's curator or maintenance pipeline derived are never listed.
    Environment factories are not materialized for discovery.
    """
    registration, _ = await _build_registration(
        app, project_root=project_root, start_cursor=start_cursor
    )
    return registration


def _redact(app: CayuApp, value: _ModelT) -> _ModelT:
    return type(value).model_validate(app.redact_json(value.model_dump(mode="json")))


async def _session_store_identity(app: CayuApp) -> str | None:
    """The session store's identity, or None when it has none (a revision-124 database)."""
    try:
        return await _store_identity(app)
    except NotImplementedError:
        return None


async def newest_event_sequence(app: CayuApp) -> int:
    """The newest durable event sequence, where a starting process's use begins."""
    newest = await app.session_store.query_events(
        EventQuery(limit=1, order_by=EventOrder.SEQUENCE_DESC)
    )
    return newest[0].sequence if newest else 0


@dataclass
class _Totals:
    calls: int = 0
    failures: int = 0
    approved: int = 0
    rejected: int = 0
    lookups: int = 0
    last_used_at: datetime | None = None

    def used(self, at: datetime) -> None:
        at = _aware(at)
        if self.last_used_at is None or at > self.last_used_at:
            self.last_used_at = at


async def _cloud_usage_report(
    app: CayuApp,
    registered: _Registered,
    *,
    after: int | None,
    limit: int = _USAGE_PAGE,
) -> tuple[CloudUsageReport | None, bool]:
    """Count the registered tools' and entries' use in the next events after ``after``.

    Counts are totals over ``(after, through]``, so the body is bounded by what was
    registered, never by how many events the window holds.
    """
    events = await app.session_store.query_events(
        EventQuery(
            after_sequence=after or 0,
            limit=limit,
            order_by=EventOrder.SEQUENCE_ASC,
            event_types=_USAGE_EVENT_TYPES,
        )
    )
    if not events:
        return None, False
    tools: dict[str, _Totals] = {}
    knowledge: dict[str, _Totals] = {}
    for record in events:
        event = record.event
        if event.type == EventType.KNOWLEDGE_LOOKUP:
            contribution = knowledge_usage_contribution(event)
            if contribution is None:
                continue
            namespace, ids = contribution
            for entry_id in ids:
                if (namespace, entry_id) in registered.knowledge:
                    totals = knowledge.setdefault(entry_id, _Totals())
                    totals.lookups += 1
                    totals.used(event.timestamp)
            continue
        contribution = tool_usage_contribution(event)
        if contribution is None or contribution.name not in registered.tools:
            continue
        totals = tools.setdefault(contribution.name, _Totals())
        totals.calls += contribution.calls
        totals.failures += contribution.failures
        totals.approved += contribution.approved
        totals.rejected += contribution.rejected
        if contribution.calls:
            totals.used(event.timestamp)
    report = CloudUsageReport(
        after=after,
        through=events[-1].sequence,
        tools=tuple(
            CloudToolUsage(
                name=name,
                calls=value.calls,
                failures=value.failures,
                approved=value.approved,
                rejected=value.rejected,
                last_used_at=value.last_used_at,
            )
            for name, value in tools.items()
        ),
        knowledge=tuple(
            CloudKnowledgeUsage(id=entry_id, lookups=value.lookups, last_used_at=value.last_used_at)
            for entry_id, value in knowledge.items()
        ),
    )
    return report, len(events) == limit


class CloudInventoryConfig(_Model):
    api_url: str
    application_id: str
    release_id: str
    token: SecretStr

    @model_validator(mode="after")
    def validate_destination(self) -> CloudInventoryConfig:
        try:
            parsed = urlsplit(self.api_url)
            valid_url = (
                parsed.scheme == "https"
                and parsed.hostname
                and not parsed.username
                and not parsed.password
                and not parsed.query
                and not parsed.fragment
            )
        except ValueError:
            valid_url = False
        token = self.token.get_secret_value()
        if (
            not valid_url
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", self.application_id)
            or not self.release_id
            or _LINE_BREAKS.search(self.release_id)
            or not token
            or any(ord(c) < 33 or ord(c) > 126 for c in token)
        ):
            raise ValueError(
                "Cloud registration requires an HTTPS API URL, application id, Release id "
                "and registry token."
            )
        return self

    @classmethod
    def from_environment(
        cls, *, release_id: str, environ: Mapping[str, str] | None = None
    ) -> CloudInventoryConfig | None:
        source = os.environ if environ is None else environ
        token = source.get(CLOUD_REGISTRY_TOKEN_ENV)
        if token is None:
            return None
        return cls(
            api_url=source.get("CAYU_CLOUD_API_URL", "").strip().rstrip("/"),
            application_id=source.get("CAYU_CLOUD_APPLICATION_ID", ""),
            release_id=release_id,
            token=SecretStr(token),
        )


class CloudRegistryRefused(Exception):
    """Cloud's definite refusal: its error code and detail, never a response body."""

    def __init__(self, status: int, code: str | None, detail: Mapping[str, Any]) -> None:
        super().__init__(f"Cayu Cloud refused the agent registry request ({status}).")
        self.status = status
        self.code = code
        self.detail = dict(detail)


class _ReporterStopped(Exception):
    pass


class CloudInventoryPublisher:
    """One lifecycle-owned publisher; network failures never stop agent work."""

    def __init__(
        self,
        app: CayuApp,
        config: CloudInventoryConfig,
        *,
        client: httpx.AsyncClient,
        project_root: Path | None = None,
    ) -> None:
        self._app = app
        self._config = config
        self._client = client
        self._project_root = project_root
        self._start_cursor: int | None = None
        self._registered: _Registered | None = None
        self._store_id: str | None = None
        # Where this store's use is counted to. Cloud keeps a cursor per session store, so it
        # is always in this store's sequence.
        self._cursor: int | None = None
        # A replacement Cloud hasn't answered yet: the cursor replaced and the start sent, kept
        # so a retry doesn't move the start past events written in between.
        self._replacement: tuple[int, int] | None = None
        # Cloud refuses an invalid settings schema with 422 and keeps the rest of the
        # registration; later registrations leave the schema out.
        self._settings_schema_refused = False

    def _route(self, suffix: str = "") -> str:
        return (
            f"{self._config.api_url}/v1/applications/"
            f"{quote(self._config.application_id, safe='')}/agent-registry/releases/"
            f"{quote(self._config.release_id, safe='')}{suffix}"
        )

    async def _send(self, method: str, url: str, value: BaseModel) -> dict[str, Any]:
        # The transport credential is not a workload secret, so redact it here too.
        redactor = SecretRedactor(self._config.token)
        sanitized = type(value).model_validate(redactor.redact_json(value.model_dump(mode="json")))
        body = _encode(sanitized)
        if len(body) > MAX_BODY_BYTES:
            raise ValueError("Cloud registry body exceeds its byte limit.")
        async with self._client.stream(
            method,
            url,
            content=body,
            headers={
                "Authorization": f"Bearer {self._config.token.get_secret_value()}",
                "Content-Type": "application/json",
            },
            follow_redirects=False,
        ) as response:
            raw = bytearray()
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > _MAX_RESPONSE_BYTES:
                    break
        try:
            answer = json.loads(bytes(raw)) if len(raw) <= _MAX_RESPONSE_BYTES else {}
        except ValueError:
            answer = {}
        if not isinstance(answer, dict):
            answer = {}
        if response.is_success:
            return answer
        error = answer.get("error")
        error = error if isinstance(error, dict) else {}
        code = error.get("code")
        detail = error.get("detail")
        raise CloudRegistryRefused(
            response.status_code,
            code if isinstance(code, str) else None,
            detail if isinstance(detail, dict) else {},
        )

    @staticmethod
    def _cursor_from(value: object) -> int | None:
        return value if type(value) is int and value >= 0 else None

    async def register(
        self, *, start_cursor: int | None = None, replaces_cursor: int | None = None
    ) -> None:
        """Register this Release and its session store, and follow the cursor Cloud answers.

        The start is where this store's use is counted from if Cloud keeps no cursor for the
        store: the cursor this process's reports reached, else the newest event when it
        started. Cloud keeps the store's cursor unless ``replaces_cursor`` names it.
        """
        if self._start_cursor is None:
            self._start_cursor = await newest_event_sequence(self._app)
        if start_cursor is None:
            start_cursor = self._start_cursor if self._cursor is None else self._cursor
        store_id = await _session_store_identity(self._app)
        while True:
            registration, registered = await _build_registration(
                self._app,
                project_root=self._project_root,
                start_cursor=start_cursor,
                transport_redactor=SecretRedactor(self._config.token),
                store_id=store_id,
                replaces_cursor=replaces_cursor,
                settings_schema=not self._settings_schema_refused,
            )
            try:
                answer = await self._send("PUT", self._route(), registration)
            except CloudRegistryRefused as refused:
                if refused.status != 422 or registration.settings_schema is None:
                    raise
                logger.warning(
                    "Cayu Cloud refused this Release's managed settings schema; "
                    "registering without it, so Cloud can't manage its Agent settings."
                )
                self._settings_schema_refused = True
                continue
            break
        self._registered = registered
        self._store_id = store_id
        self._cursor = self._cursor_from(answer.get("usage_cursor"))

    async def _replace_unreachable_cursor(self) -> bool:
        """Hand back a cursor past this store's newest event; return whether Cloud took it.

        Cloud's cursor came from a report of this store's sequence, so a store whose newest
        event, read after Cloud answered, is below it holds another history: restored from a
        backup, or started over without an identity. Counting resumes from its newest event.
        """
        cursor = self._cursor
        if cursor is None:
            return False
        if self._replacement is None or self._replacement[0] != cursor:
            newest = await newest_event_sequence(self._app)
            if cursor <= newest:
                return False
            self._replacement = (cursor, newest)
        _, start = self._replacement
        await self.register(start_cursor=start, replaces_cursor=cursor)
        self._replacement = None
        return self._cursor is None or self._cursor <= start

    async def publish_once(self) -> bool:
        """Register if needed, then report one window of use; return whether more is ready."""
        try:
            if self._registered is None:
                await self.register()
            elif (identity := await _session_store_identity(self._app)) != self._store_id:
                # The database was replaced under this process: count the new store from now.
                # Its start is kept until Cloud answers, so a retry doesn't skip events.
                self._cursor = await newest_event_sequence(self._app)
                self._store_id = identity
                self._registered = None
                # A pending rewind belongs only to the store that requested it.
                self._replacement = None
                await self.register()
            assert self._registered is not None
            report, more = await _cloud_usage_report(
                self._app, self._registered, after=self._cursor
            )
            if report is None:
                # Nothing new, or Cloud's cursor is past everything this store holds.
                return await self._replace_unreachable_cursor()
            report = report.model_copy(update={"store_id": self._store_id})
            answer = await self._send("POST", self._route("/usage"), report)
            self._cursor = self._cursor_from(answer.get("usage_cursor"))
            if self._cursor is None:
                self._cursor = report.through
            return more
        except CloudRegistryRefused as refused:
            if refused.status == 409 and refused.code == "usage_cursor":
                # Another process or an earlier lost answer moved Cloud's cursor.
                self._cursor = self._cursor_from(refused.detail.get("cursor"))
                return True
            if refused.status == 409 and refused.code in {"not_registered", "store_not_registered"}:
                # Cloud forgot the registration or this store's cursor: register again, from
                # where this store's use was counted.
                self._registered = None
                return True
            if refused.status == 409 and refused.code == "release_not_current":
                # A newer Release counts this Agent's use; report again later.
                return False
            if refused.status in {401, 404, 413, 422}:
                raise _ReporterStopped from None
            raise

    async def run(self) -> None:
        while True:
            delay = _RETRY_SECONDS
            try:
                if await self.publish_once():
                    delay = 0.0
            except _ReporterStopped:
                logger.warning(
                    "Cayu Cloud refused this Agent's registry credential or registration; "
                    "tool and knowledge reporting stopped until the service restarts."
                )
                return
            except Exception:
                # Never log credentials, response bodies, exception strings or URLs.
                logger.warning("Cayu Cloud registry reporting failed; retrying in 30 seconds.")
            await asyncio.sleep(delay)


@asynccontextmanager
async def cloud_inventory_lifespan(
    app: CayuApp, *, project_root: Path | None = None
) -> AsyncIterator[None]:
    """Register on server startup when Cloud binds the Agent's registry credential."""
    if os.environ.get(CLOUD_REGISTRY_TOKEN_ENV) is None:
        yield
        return
    if app.release_identity is None:
        raise ValueError("Cloud registration requires CAYU_RELEASE_ID or app.release_identity.")
    config = CloudInventoryConfig.from_environment(release_id=app.release_identity.id)
    assert config is not None
    async with httpx.AsyncClient(timeout=10.0, trust_env=False) as client:
        publisher = CloudInventoryPublisher(app, config, client=client, project_root=project_root)
        task = asyncio.create_task(publisher.run(), name="cayu-cloud-inventory")
        try:
            yield
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
