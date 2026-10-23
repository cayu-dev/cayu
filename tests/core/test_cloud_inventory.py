"""The Cloud agent-registry boundary: public declarations and counted use only."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr, ValidationError
from tests.core.test_cloud_reporting import PRICE, _app, _Provider, _respond, _start
from tests.core.test_mcp import FakeMcpSession, _fake_server_spec, _fake_tool_definitions

from cayu import AgentSpec, CayuApp, Event, EventType
from cayu.cloud_inventory import (
    CLOUD_REGISTRY_CONTRACT,
    MAX_BODY_BYTES,
    CloudInventoryConfig,
    CloudInventoryPublisher,
    CloudKnowledge,
    CloudKnowledgeEntry,
    CloudRegistration,
    CloudTool,
    CloudToolSource,
    _build_registration,
    _cloud_usage_report,
    _encode,
    _fit_registration,
    cloud_registration,
)
from cayu.context.base import _context_recall_telemetry_publisher_scope
from cayu.knowledge.records import (
    KnowledgeActorType,
    KnowledgeEntry,
    KnowledgeStatus,
    KnowledgeVisibility,
)
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.search import KnowledgeQuery
from cayu.knowledge.usage import knowledge_lookup_scope
from cayu.mcp import McpInitializeResult, McpToolset
from cayu.memory.recall import KnowledgeRecallSource
from cayu.releases import ReleaseIdentity
from cayu.storage.knowledge_memory import InMemoryKnowledgeStore
from cayu.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.knowledge import SearchKnowledgeTool
from cayu.vaults.redaction import SecretRedactor

SCOPE = KnowledgeAccessScope.privileged()
RELEASE = ReleaseIdentity(id="release-a")
SECRET = "private-credential-canary"
TOKEN = "cayu_reg_scoped-registry-token-canary"
# The fields agent-registry-v1 accepts; Cloud refuses a body with any other.
REGISTRATION_KEYS = {
    "contract",
    "runtime",
    "agents",
    "tools",
    "knowledge",
    "store_id",
    "start_cursor",
    "replaces_cursor",
}
# Only the publisher knows the store and the cursor Cloud answered.
_PUBLISHER_KEYS = {"store_id", "replaces_cursor"}
TOOL_KEYS = {"name", "description", "agents", "source", "effect", "approval"}
ENTRY_KEYS = {"id", "title", "kind", "status", "created_by", "updated_at"}
USAGE_KEYS = {"contract", "store_id", "after", "through", "tools", "knowledge"}


class _Tool(Tool):
    def __init__(self, name: str = "lookup"):
        super().__init__(
            ToolSpec(
                name=name,
                description=f"Find invoices {SECRET}",
                input_schema={"type": "object", "properties": {"secret-schema-canary": {}}},
            )
        )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="done")


def _config(release_id: str = RELEASE.id) -> CloudInventoryConfig:
    return CloudInventoryConfig(
        api_url="https://cloud.example",
        application_id="app-1",
        release_id=release_id,
        token=SecretStr(TOKEN),
    )


async def _knowledge_app() -> tuple[CayuApp, InMemoryKnowledgeStore]:
    store = InMemoryKnowledgeStore(access_scope=SCOPE)
    past = datetime.now(UTC) - timedelta(days=1)
    entries = [
        KnowledgeEntry(
            id="seed",
            text="document-body-canary",
            title=f"Month-end checklist {SECRET}",
            kind="procedure",
            metadata={"summary": "metadata-canary"},
        ),
        # The reviewer's case: a memory the model wrote from a user's chat.
        KnowledgeEntry(
            id="memory",
            text="Alice told me about her diagnosis.",
            title="Alice's HIV diagnosis from chat",
            created_by_type=KnowledgeActorType.MODEL,
            status=KnowledgeStatus.PENDING,
        ),
        KnowledgeEntry(
            id="kept-memory",
            text="kept",
            title="kept-memory-canary",
            created_by_type=KnowledgeActorType.MODEL,
        ),
        KnowledgeEntry(
            id="person", text="p", title="person-canary", created_by_type=KnowledgeActorType.USER
        ),
        KnowledgeEntry(id="deleted", text="d", title="user-deleted-title"),
        KnowledgeEntry(id="expired", text="e", title="expired-canary", expires_at=past),
        KnowledgeEntry(
            id="archived", text="a", title="archived-canary", status=KnowledgeStatus.ARCHIVED
        ),
        KnowledgeEntry(
            id="personal",
            text="u",
            title="personal-canary",
            visibility=KnowledgeVisibility.USER,
        ),
        KnowledgeEntry(
            id="session",
            text="s",
            title="session-canary",
            visibility=KnowledgeVisibility.SESSION,
        ),
        KnowledgeEntry(
            id="curated",
            text="c",
            title="curated-canary",
            metadata={"cayu_curator": {"evaluator_verdict": "accept"}},
        ),
    ]
    for entry in entries:
        await store.create_entry(entry)
    await store.delete_entry("deleted", expected_revision=1)
    app = CayuApp(
        knowledge_store=store,
        knowledge_access_scope=SCOPE,
        release_identity=RELEASE,
        secret_redactor=SecretRedactor(SECRET),
        enable_logging=False,
    )
    app.register_agent(
        AgentSpec(
            name="billing",
            model="model",
            system_prompt="prompt-canary",
            metadata={"private": "agent-metadata-canary"},
            provider_options={"private": "provider-option-canary"},
        ),
        tools=[_Tool()],
    )
    app.register_agent(AgentSpec(name="audit", model="model"), tools=[_Tool()])
    return app, store


def test_registration_reports_only_active_shared_knowledge_the_app_registered():
    async def scenario():
        app, _ = await _knowledge_app()
        registration = await cloud_registration(app, project_root=Path.cwd(), start_cursor=7)
        encoded = _encode(registration).decode()
        for forbidden in (
            SECRET,
            "Alice",
            "kept-memory-canary",
            "person-canary",
            "user-deleted-title",
            "expired-canary",
            "archived-canary",
            "personal-canary",
            "session-canary",
            "curated-canary",
            "prompt-canary",
            "agent-metadata-canary",
            "provider-option-canary",
            "document-body-canary",
            "metadata-canary",
            "secret-schema-canary",
        ):
            assert forbidden not in encoded
        body = json.loads(encoded)
        assert set(body) == REGISTRATION_KEYS - _PUBLISHER_KEYS
        assert body["contract"] == CLOUD_REGISTRY_CONTRACT == "cayu.agent-registry/v1"
        assert body["start_cursor"] == 7
        assert body["agents"] == ["audit", "billing"]
        # One tool name is one Cloud tool, listing every specialist that has it.
        (tool,) = body["tools"]
        assert set(tool) == TOOL_KEYS
        assert tool["agents"] == ["audit", "billing"]
        assert tool["source"] == {"kind": "agent"}
        assert tool["approval"] == "never"
        assert tool["effect"] == "external"
        assert body["knowledge"]["available"] is True
        assert body["knowledge"]["truncated"] is False
        (entry,) = body["knowledge"]["entries"]
        assert set(entry) == ENTRY_KEYS
        assert entry["id"] == "seed"
        assert entry["title"].startswith("Month-end checklist ")
        assert entry["status"] == "active"
        assert entry["created_by"] == "app"
        assert datetime.fromisoformat(entry["updated_at"]).tzinfo is not None
        await app.aclose()

    asyncio.run(scenario())


def test_registration_without_a_knowledge_store_is_unavailable():
    async def scenario():
        app = CayuApp(release_identity=RELEASE, enable_logging=False)
        registration = await cloud_registration(app)
        assert registration.knowledge == CloudKnowledge(available=False, truncated=False)
        assert json.loads(_encode(registration))["knowledge"]["entries"] == []
        await app.aclose()

    asyncio.run(scenario())


def test_registration_caps_knowledge_at_cloud_limit_without_losing_equal_timestamps():
    async def scenario():
        store = InMemoryKnowledgeStore(access_scope=SCOPE)
        timestamp = datetime(2026, 10, 21, tzinfo=UTC)
        for number in range(503):
            await store.create_entry(
                KnowledgeEntry(
                    id=f"entry-{number:03}",
                    text="private body",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
        app = CayuApp(
            knowledge_store=store,
            knowledge_access_scope=SCOPE,
            release_identity=RELEASE,
            enable_logging=False,
        )
        registration = await cloud_registration(app)
        ids = [entry.id for entry in registration.knowledge.entries]
        assert len(ids) == len(set(ids)) == 500
        assert registration.knowledge.truncated is True
        await app.aclose()

    asyncio.run(scenario())


def test_an_oversized_registration_is_fit_under_cloud_body_limit():
    # JSON escapes each of these characters to six bytes, so this is several MiB.
    description = "é" * 2_000
    registration = CloudRegistration(
        agents=("billing",),
        tools=tuple(
            CloudTool(
                name=f"tool_{number}",
                description=description,
                agents=("billing",),
                source=CloudToolSource(kind="agent"),
                effect="none",
                approval="never",
            )
            for number in range(500)
        ),
        knowledge=CloudKnowledge(
            available=True,
            truncated=False,
            entries=tuple(
                CloudKnowledgeEntry(
                    id=f"entry-{number}", title="t" * 300, kind="fact", updated_at=datetime.now(UTC)
                )
                for number in range(500)
            ),
        ),
    )
    assert len(_encode(registration)) > MAX_BODY_BYTES
    fitted = _fit_registration(registration)
    assert len(_encode(fitted)) <= MAX_BODY_BYTES
    assert len(fitted.tools) == 500
    assert fitted.knowledge.truncated is True


def test_external_package_and_mcp_server_provenance(monkeypatch):
    monkeypatch.setattr(_Tool, "__module__", "invoice_tools.lookup")
    monkeypatch.setattr(
        "cayu.cloud_inventory.metadata.packages_distributions",
        lambda: {"invoice_tools": ["invoice-tools"]},
    )
    monkeypatch.setattr("cayu.cloud_inventory.metadata.version", lambda name: "1.2.3")

    async def scenario():
        definitions = _fake_tool_definitions("echo")
        session = FakeMcpSession(
            definitions=definitions,
            initialize_result=McpInitializeResult(
                protocol_version="2025-06-18",
                capabilities={"tools": {}},
                instructions="mcp-server-prompt-canary",
            ),
        )
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "inventory-test"}),
            session=session,
            definitions=definitions,
        )
        app = CayuApp(release_identity=RELEASE, enable_logging=False)
        app.register_agent(
            AgentSpec(name="billing", model="model"), tools=[_Tool()], mcp_toolsets=(toolset,)
        )
        registration = await cloud_registration(app)
        tools = {tool.name: tool for tool in registration.tools}
        assert tools["lookup"].source == CloudToolSource(
            kind="package", package="invoice-tools", version="1.2.3"
        )
        mcp = [tool for tool in tools.values() if tool.source.kind == "mcp"]
        assert len(mcp) == 1
        assert json.loads(_encode(mcp[0].source)) == {
            "kind": "mcp",
            "server": toolset.server.name,
        }
        assert "mcp-server-prompt-canary" not in _encode(registration).decode()
        await app.aclose()

    asyncio.run(scenario())


async def _usage_app(store=None, *, knowledge_store=None):
    if knowledge_store is None:
        app = _app(store=store, release=RELEASE)
    else:
        app = CayuApp(
            session_store=store,
            knowledge_store=knowledge_store,
            knowledge_access_scope=SCOPE,
            release_identity=RELEASE,
            enable_logging=False,
        )
        app.register_provider(
            _Provider(name=PRICE.provider_name, response_factory=_respond), default=True
        )
        app.register_agent(AgentSpec(name="writer", model=PRICE.model))
        app.register_agent(AgentSpec(name="keeper", model=PRICE.model), tools=[_Tool("record")])
    await _start(app, "session", "writer")
    for kind, tool in (
        (EventType.TOOL_CALL_STARTED, "record"),
        (EventType.TOOL_CALL_COMPLETED, "record"),
        (EventType.TOOL_CALL_FAILED, "record"),
        (EventType.TOOL_CALL_APPROVED, "record"),
        (EventType.TOOL_CALL_APPROVAL_DENIED, "record"),
        (EventType.TOOL_CALL_COMPLETED, "unregistered_tool"),
    ):
        await app.session_store.append_event(
            "session",
            Event(
                type=kind,
                session_id="session",
                agent_name="writer",
                tool_name=tool,
                payload={"arguments": "argument-canary", "result": "result-canary"},
            ),
        )
    await app.session_store.append_event(
        "session",
        Event(
            type=EventType.KNOWLEDGE_LOOKUP,
            session_id="session",
            agent_name="writer",
            payload={"namespace": "default", "entry_ids": ["seed", "seed", "memory"]},
        ),
    )
    return app


async def _seeded_store() -> InMemoryKnowledgeStore:
    store = InMemoryKnowledgeStore(access_scope=SCOPE)
    await store.create_entry(KnowledgeEntry(id="seed", text="t", title="Seed"))
    await store.create_entry(
        KnowledgeEntry(
            id="memory",
            text="m",
            title="memory-title-canary",
            created_by_type=KnowledgeActorType.MODEL,
        )
    )
    return store


def test_usage_counts_registered_tools_and_entries_over_a_window():
    async def scenario():
        app = await _usage_app(knowledge_store=await _seeded_store())
        _, registered = await _build_registration(app, project_root=None, start_cursor=None)
        assert registered.knowledge == frozenset({("default", "seed")})
        report, more = await _cloud_usage_report(app, registered, after=None)
        assert report is not None and more is False
        body = json.loads(_encode(report))
        assert set(body) == USAGE_KEYS - {"after", "store_id"}
        (tool,) = body["tools"]
        assert {
            key: tool[key] for key in ("name", "calls", "failures", "approved", "rejected")
        } == {
            "name": "record",
            "calls": 2,
            "failures": 1,
            "approved": 1,
            "rejected": 1,
        }
        # The model's memory isn't registered, so its id never leaves through usage either.
        assert [(entry["id"], entry["lookups"]) for entry in body["knowledge"]] == [("seed", 1)]
        text = _encode(report).decode()
        for forbidden in ("argument-canary", "result-canary", "memory", "unregistered_tool"):
            assert forbidden not in text
        again, _ = await _cloud_usage_report(app, registered, after=report.through)
        assert again is None
        await app.aclose()

    asyncio.run(scenario())


def test_usage_report_is_bounded_by_what_was_registered_not_by_events():
    async def scenario():
        app = _app(release=RELEASE)
        await _start(app, "session", "writer")
        ids = [f"{number:032x}" for number in range(100)]
        for _ in range(100):
            await app.session_store.append_event(
                "session",
                Event(
                    type=EventType.KNOWLEDGE_LOOKUP,
                    session_id="session",
                    payload={"namespace": "default", "entry_ids": ids},
                ),
            )
        _, registered = await _build_registration(app, project_root=None, start_cursor=None)
        registered = type(registered)(
            tools=registered.tools, knowledge=frozenset(("default", value) for value in ids)
        )
        report, _ = await _cloud_usage_report(app, registered, after=None)
        assert report is not None
        assert len(report.knowledge) == 100
        assert {entry.lookups for entry in report.knowledge} == {100}
        assert len(_encode(report)) < MAX_BODY_BYTES // 20
        await app.aclose()

    asyncio.run(scenario())


class _Cloud:
    """agent-registry-v1's write routes, enough to drive the reporter."""

    def __init__(self) -> None:
        # Each session store's cursor, and the cursor its last replacement replaced.
        self.cursors: dict[str | None, int | None] = {}
        self.replaced: dict[str | None, int] = {}
        # The store that registered last; ``cursor`` is its cursor.
        self.store_id: str | None = None
        self.lose_next_answer = False
        self.current: str | None = None
        self.registered: dict[str, dict] = {}
        self.applied: list[dict] = []
        self.requests: list[httpx.Request] = []
        self.next_errors: list[httpx.Response] = []

    @property
    def cursor(self) -> int | None:
        return self.cursors.get(self.store_id)

    @cursor.setter
    def cursor(self, value: int | None) -> None:
        self.cursors[self.store_id] = value

    @staticmethod
    def _error(status: int, code: str, **detail) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"code": code, "message": "refused", "detail": detail}}
        )

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        assert TOKEN.encode() not in request.content
        if self.next_errors:
            return self.next_errors.pop(0)
        prefix = "/v1/applications/app-1/agent-registry/releases/"
        assert request.url.path.startswith(prefix)
        release, _, route = request.url.path.removeprefix(prefix).partition("/")
        body = json.loads(request.content)
        assert body["contract"] == CLOUD_REGISTRY_CONTRACT
        if request.method == "PUT":
            assert route == ""
            assert set(body) <= REGISTRATION_KEYS
            self.registered[release] = body
            self.current = release
            store, replaces = body.get("store_id"), body.get("replaces_cursor")
            if self.cursors.get(store) is None:
                self.cursors[store] = body.get("start_cursor")
            elif replaces == self.cursors[store] and replaces != self.replaced.get(store):
                self.cursors[store] = body.get("start_cursor")
                self.replaced[store] = replaces
            self.store_id = store
            return httpx.Response(200, json={"release_id": release, "usage_cursor": self.cursor})
        assert request.method == "POST" and route == "usage"
        assert set(body) <= USAGE_KEYS
        if release not in self.registered:
            return self._error(409, "not_registered")
        if release != self.current:
            return self._error(409, "release_not_current", release_id=self.current)
        store = body.get("store_id")
        if store not in self.cursors:
            return self._error(409, "store_not_registered")
        cursor = self.cursors[store]
        if cursor is not None and body.get("after") != cursor:
            return self._error(409, "usage_cursor", cursor=cursor)
        self.applied.append(body)
        self.cursors[store] = body["through"]
        if self.lose_next_answer:
            self.lose_next_answer = False
            return httpx.Response(503)
        return httpx.Response(200, json={"usage_cursor": body["through"]})


def _calls(cloud: _Cloud) -> int:
    return sum(tool["calls"] for body in cloud.applied for tool in body.get("tools", []))


def test_reporter_starts_counting_at_its_start_and_follows_cloud_cursor():
    async def scenario():
        app = await _usage_app()
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            # Events before the process started are not counted: Cloud starts there.
            assert await publisher.publish_once() is False
            assert cloud.applied == []
            await app.session_store.append_event(
                "session",
                Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", tool_name="record"),
            )
            await publisher.publish_once()
            assert _calls(cloud) == 1
            # Another process reported the same window: Cloud names its cursor and the
            # reporter counts again from there, so nothing is counted twice.
            await app.session_store.append_event(
                "session",
                Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", tool_name="record"),
            )
            newest = (await app.session_store.query_events())[-1].sequence
            cloud.cursor = newest
            assert await publisher.publish_once() is True
            assert await publisher.publish_once() is False
            assert _calls(cloud) == 1
            registrations = sum(request.method == "PUT" for request in cloud.requests)
            assert registrations == 1
        await app.aclose()

    asyncio.run(scenario())


def test_reporter_registers_again_when_cloud_forgot_and_backs_off_when_not_current():
    async def scenario():
        app = await _usage_app()
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            await publisher.publish_once()
            await app.session_store.append_event(
                "session",
                Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", tool_name="record"),
            )
            # A newer Release registered: back off, without registering again.
            cloud.current = "release-b"
            assert await publisher.publish_once() is False
            assert sum(request.method == "PUT" for request in cloud.requests) == 1
            assert cloud.applied == []
            # Cloud lost this Release's registration: register again and report.
            cloud.registered.clear()
            assert await publisher.publish_once() is True
            await publisher.publish_once()
            assert sum(request.method == "PUT" for request in cloud.requests) == 2
            assert _calls(cloud) == 1
        await app.aclose()

    asyncio.run(scenario())


def test_reporter_retries_server_errors_and_stops_on_a_refused_credential(caplog):
    async def scenario():
        app = await _usage_app()
        cloud = _Cloud()
        cloud.next_errors.append(httpx.Response(503, text="response-secret-canary"))
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            with pytest.raises(Exception, match="refused") as raised:
                await publisher.publish_once()
            assert "response-secret-canary" not in str(raised.value)
            await publisher.publish_once()
            assert cloud.current == RELEASE.id
            cloud.next_errors.append(
                httpx.Response(401, json={"error": {"code": "unauthenticated", "detail": {}}})
            )
            await app.session_store.append_event(
                "session",
                Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", tool_name="record"),
            )
            with caplog.at_level(logging.WARNING, logger="cayu.cloud_inventory"):
                await asyncio.wait_for(publisher.run(), timeout=10)
            assert "reporting stopped" in caplog.text
            assert TOKEN not in caplog.text
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "updates",
    [
        {"api_url": "http://cloud.example"},
        {"api_url": "https://user:secret@cloud.example"},
        {"api_url": "https://cloud.example/?token=secret"},
        {"application_id": "../other"},
        {"application_id": ".."},
        {"release_id": ""},
        {"token": SecretStr("token\r\nheader")},
    ],
)
def test_registry_credential_is_destination_bound(updates):
    values = _config().model_dump()
    values.update(updates)
    with pytest.raises(ValidationError):
        CloudInventoryConfig(**values)
    assert TOKEN not in repr(_config())
    assert CloudInventoryConfig.from_environment(release_id="r", environ={}) is None


def test_search_and_automatic_recall_record_ids_only():
    async def scenario():
        store = InMemoryKnowledgeStore(access_scope=SCOPE)
        await store.create_entry(KnowledgeEntry(id="entry", text="private invoice document"))
        observed = []

        async def observe(namespace, ids):
            observed.append((namespace, ids))

        ctx = ToolContext(session_id="session", knowledge_store=store, knowledge_access_scope=SCOPE)
        with knowledge_lookup_scope(observe):
            result = await SearchKnowledgeTool().run(ctx, {"query": "invoice"})
        assert not result.is_error
        assert observed == [("default", ("entry",))]
        telemetry = []

        async def publish(value):
            telemetry.append(value)

        source = KnowledgeRecallSource(store)
        with _context_recall_telemetry_publisher_scope(publish):
            await source._observed_search(KnowledgeQuery(text="invoice"), access_scope=SCOPE)
        assert len(telemetry) == 1
        assert telemetry[0].event_type == EventType.KNOWLEDGE_LOOKUP
        assert telemetry[0].payload == {"namespace": "default", "entry_ids": ["entry"]}
        assert "invoice" not in json.dumps(telemetry[0].payload)

    asyncio.run(scenario())


def test_runtime_search_is_counted_for_a_registered_environment_entry():
    from cayu import Message, RunRequest
    from cayu.environments import Environment, EnvironmentSpec
    from cayu.messages import MessageRole
    from cayu.providers import ModelStreamEvent

    def respond(request):
        if any(message.role == MessageRole.TOOL for message in request.messages):
            return [ModelStreamEvent.completed({"finish_reason": "stop"})]
        return [
            ModelStreamEvent.tool_call(
                id="search", name="search_knowledge", arguments={"query": "invoice"}
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]

    async def scenario():
        store = InMemoryKnowledgeStore(access_scope=SCOPE)
        await store.create_entry(KnowledgeEntry(id="entry", text="private invoice document"))
        app = CayuApp(release_identity=RELEASE, enable_logging=False)
        app.register_provider(_Provider(name="test", response_factory=respond), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"), knowledge_store=store, knowledge_access_scope=SCOPE
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="billing", model="model"), tools=[SearchKnowledgeTool()])
        _, registered = await _build_registration(app, project_root=None, start_cursor=None)
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="billing",
                    session_id="lookup-session",
                    messages=[Message.text("user", "Find invoice")],
                )
            )
        ]
        assert any(event.type == EventType.KNOWLEDGE_LOOKUP for event in events)
        report, _ = await _cloud_usage_report(app, registered, after=None)
        assert report is not None
        assert [(entry.id, entry.lookups) for entry in report.knowledge] == [("entry", 1)]
        assert [(tool.name, tool.calls) for tool in report.tools] == [("search_knowledge", 1)]
        assert "private invoice document" not in _encode(report).decode()
        await app.aclose()

    asyncio.run(scenario())


def test_publisher_does_not_follow_redirects_or_log_private_failure_details(caplog):
    async def scenario():
        app = CayuApp(release_identity=RELEASE, enable_logging=False)
        attempted = asyncio.Event()
        requests = []

        async def redirect(request):
            requests.append(request)
            attempted.set()
            return httpx.Response(307, headers={"Location": "https://other.example/private-canary"})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(redirect), follow_redirects=True
        ) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            task = asyncio.create_task(publisher.run())
            try:
                await asyncio.wait_for(attempted.wait(), timeout=10)
                await asyncio.sleep(0)
                assert len(requests) == 1
                assert "retrying in 30 seconds" in caplog.text
                assert "private-canary" not in caplog.text
                assert TOKEN not in caplog.text
                assert "other.example" not in caplog.text
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        await app.aclose()

    asyncio.run(scenario())


def test_a_restarted_process_continues_from_cloud_cursor(sqlite_resources):
    from cayu.storage.sqlite import SQLiteSessionStore

    async def scenario():
        async with sqlite_resources as resources:
            path = resources.path("cloud-usage.sqlite")
            store = resources.own(SQLiteSessionStore(path))
            cloud = _Cloud()
            first_app = await _usage_app(store)
            async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
                await CloudInventoryPublisher(first_app, _config(), client=client).publish_once()
                await first_app.session_store.append_event(
                    "session",
                    Event(
                        type=EventType.TOOL_CALL_COMPLETED, session_id="session", tool_name="record"
                    ),
                )
                await first_app.aclose()
                await store.close()
                reopened = resources.own(SQLiteSessionStore(path))
                restarted = _app(store=reopened, release=RELEASE)
                publisher = CloudInventoryPublisher(restarted, _config(), client=client)
                await publisher.publish_once()
                assert _calls(cloud) == 1
                await publisher.publish_once()
                assert _calls(cloud) == 1
                await restarted.aclose()

    asyncio.run(scenario())


def _record_call() -> Event:
    return Event(type=EventType.TOOL_CALL_COMPLETED, session_id="session", tool_name="record")


def test_a_store_that_starts_over_lower_is_counted_from_its_own_start():
    # The reviewer's scenario: the Agent restarts on a new in-memory store whose sequence is
    # below the cursor Cloud reached on the old one.
    async def scenario():
        cloud = _Cloud()
        first = await _usage_app()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            old = CloudInventoryPublisher(first, _config(), client=client)
            await old.publish_once()
            for _ in range(15):
                await first.session_store.append_event("session", _record_call())
            assert await old.publish_once() is False
            assert _calls(cloud) == 15
            first_store = await first.session_store.read_store_identity()
            reached = cloud.cursors[first_store]
            restarted = await _usage_app()
            started = (await restarted.session_store.query_events())[-1].sequence
            assert reached is not None and started < reached
            new = CloudInventoryPublisher(restarted, _config(), client=client)
            await new.publish_once()
            assert cloud.cursor == started
            for _ in range(3):
                await restarted.session_store.append_event("session", _record_call())
            await new.publish_once()
            assert _calls(cloud) == 18
            # The old process, still serving its own store, keeps counting it; a report whose
            # answer was lost is not counted again.
            await first.session_store.append_event("session", _record_call())
            cloud.lose_next_answer = True
            with pytest.raises(Exception, match="refused"):
                await old.publish_once()
            assert await old.publish_once() is True
            assert await old.publish_once() is False
            assert _calls(cloud) == 19
            await restarted.session_store.append_event("session", _record_call())
            for _ in range(2):
                await new.publish_once()
                await old.publish_once()
            assert _calls(cloud) == 20
            # Cloud dropped the old store's cursor: it registers again from where it counted.
            del cloud.cursors[first_store]
            await first.session_store.append_event("session", _record_call())
            assert await old.publish_once() is True
            await old.publish_once()
            assert _calls(cloud) == 21
            assert sum(request.method == "PUT" for request in cloud.requests) == 3
        await first.aclose()
        await restarted.aclose()

    asyncio.run(scenario())


def test_a_stale_publisher_on_a_shared_store_does_not_recount_after_another_store():
    async def scenario():
        cloud = _Cloud()
        app = await _usage_app()
        other = await _usage_app()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            ahead = CloudInventoryPublisher(app, _config(), client=client)
            behind = CloudInventoryPublisher(app, _config(), client=client)
            await ahead.publish_once()
            await behind.publish_once()
            await app.session_store.append_event("session", _record_call())
            await ahead.publish_once()
            assert _calls(cloud) == 1
            # Another store registers; the publisher still at the older cursor reports next.
            await CloudInventoryPublisher(other, _config(), client=client).publish_once()
            for _ in range(3):
                await behind.publish_once()
            assert _calls(cloud) == 1
        await app.aclose()
        await other.aclose()

    asyncio.run(scenario())


def test_a_database_replaced_under_the_process_is_counted_as_a_new_store():
    async def scenario():
        app = await _usage_app()
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            await publisher.publish_once()
            replaced = "c" * 32

            async def new_identity() -> str:
                return replaced

            app.session_store.read_store_identity = new_identity  # type: ignore[method-assign]
            newest = (await app.session_store.query_events())[-1].sequence
            # The first registration of the new store fails; calls arrive before the retry.
            cloud.next_errors.append(httpx.Response(503))
            with pytest.raises(Exception, match="refused"):
                await publisher.publish_once()
            await app.session_store.append_event("session", _record_call())
            await publisher.publish_once()
            registration = json.loads(cloud.requests[-2].content)
            assert (registration["store_id"], registration["start_cursor"]) == (replaced, newest)
            assert _calls(cloud) == 1
            await app.session_store.append_event("session", _record_call())
            await publisher.publish_once()
            assert _calls(cloud) == 2
            assert json.loads(cloud.requests[-1].content)["store_id"] == replaced
        await app.aclose()

    asyncio.run(scenario())


def test_a_restored_store_hands_back_a_cursor_it_cannot_reach():
    async def scenario():
        app = await _usage_app()
        cloud = _Cloud()
        newest = (await app.session_store.query_events())[-1].sequence
        # Restored from a backup: the same store, which Cloud had counted further.
        cloud.store_id = await app.session_store.read_store_identity()
        cloud.cursor = newest + 50
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            # The first replacement fails; a call arrives before the retry, which keeps the start.
            await publisher.register()
            cloud.next_errors.append(httpx.Response(503))
            with pytest.raises(Exception, match="refused"):
                await publisher.publish_once()
            await app.session_store.append_event("session", _record_call())
            assert await publisher.publish_once() is True
            assert cloud.cursor == newest
            replaced = json.loads(cloud.requests[-1].content)
            assert (replaced["start_cursor"], replaced["replaces_cursor"]) == (newest, newest + 50)
            await publisher.publish_once()
            assert _calls(cloud) == 1
            for _ in range(2):
                await app.session_store.append_event("session", _record_call())
            await publisher.publish_once()
            assert _calls(cloud) == 3
            assert await publisher.publish_once() is False
            # The first registration, the failed replacement and its retry.
            assert sum(request.method == "PUT" for request in cloud.requests) == 3
        await app.aclose()

    asyncio.run(scenario())


def test_pending_cursor_replacement_does_not_rewind_another_store():
    async def scenario():
        app = await _usage_app()
        cloud = _Cloud()
        newest = (await app.session_store.query_events())[-1].sequence
        cloud.store_id = await app.session_store.read_store_identity()
        cloud.cursor = newest + 50
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            await publisher.register()
            cloud.next_errors.append(httpx.Response(503))
            with pytest.raises(Exception, match="refused"):
                await publisher.publish_once()
            # Replace restored A with B, whose events Cloud already counted through the
            # same cursor. A's failed rewind must never be retried against B.
            replaced = "b" * 32

            async def new_identity() -> str:
                return replaced

            app.session_store.read_store_identity = new_identity  # type: ignore[method-assign]
            for _ in range(50):
                await app.session_store.append_event("session", _record_call())
            cloud.cursors[replaced] = newest + 50
            await publisher.publish_once()
            await publisher.publish_once()
            assert cloud.cursors[replaced] == newest + 50
            assert _calls(cloud) == 0
            registrations = [
                json.loads(request.content) for request in cloud.requests if request.method == "PUT"
            ]
            assert all(
                "replaces_cursor" not in body
                for body in registrations
                if body.get("store_id") == replaced
            )
            await app.session_store.append_event("session", _record_call())
            await publisher.publish_once()
            assert _calls(cloud) == 1
        await app.aclose()

    asyncio.run(scenario())


def test_a_store_without_an_identity_registers_without_one():
    async def scenario():
        app = await _usage_app()

        async def no_identity() -> str:
            raise NotImplementedError

        app.session_store.read_store_identity = no_identity  # type: ignore[method-assign]
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            publisher = CloudInventoryPublisher(app, _config(), client=client)
            await publisher.publish_once()
            await app.session_store.append_event("session", _record_call())
            await publisher.publish_once()
            assert _calls(cloud) == 1
            assert all("store_id" not in json.loads(r.content) for r in cloud.requests)
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("secret", [SECRET, TOKEN])
@pytest.mark.parametrize("control", ["", "\x01"])
def test_registration_redacts_complete_titles_before_normalizing_and_truncating(secret, control):
    async def scenario():
        store = InMemoryKnowledgeStore(access_scope=SCOPE)
        value = secret + control
        await store.create_entry(
            KnowledgeEntry(id="boundary", text="safe", title="x" * (301 - len(value)) + value)
        )
        app = CayuApp(
            knowledge_store=store,
            knowledge_access_scope=SCOPE,
            release_identity=RELEASE,
            secret_redactor=SecretRedactor(value) if secret == SECRET else None,
            enable_logging=False,
        )
        config = _config().model_copy(update={"token": SecretStr(value)})
        # Inspect the registration before transmission so a control character in
        # the synthetic credential never needs to enter an HTTP header.
        registration, _ = await _build_registration(
            app,
            project_root=None,
            start_cursor=None,
            transport_redactor=SecretRedactor(config.token),
        )
        title = registration.knowledge.entries[0].title
        assert title is not None
        assert secret[:-1] not in title
        assert "[REDACTED_SECRET]" in title
        assert len(title) <= 300
        await app.aclose()

    asyncio.run(scenario())


def test_same_name_tools_do_not_claim_one_specialists_policy_for_everyone():
    from cayu.tools import AlwaysRequireApprovalToolPolicy

    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_agent(AgentSpec(name="a", model="model"), tools=[_Tool()])
        app.register_agent(
            AgentSpec(name="b", model="model"),
            tools=[_Tool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        registration = await cloud_registration(app)
        (tool,) = registration.tools
        assert tool.agents == ("a", "b")
        assert tool.approval == "unknown"
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("secret", [SECRET, TOKEN])
def test_publisher_redacts_mcp_descriptions_before_the_length_limit(secret):
    async def scenario():
        definitions = _fake_tool_definitions("echo")
        definitions = tuple(
            item.model_copy(update={"description": "x" * (2001 - len(secret)) + secret})
            for item in definitions
        )
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "boundary-test"}),
            session=FakeMcpSession(definitions=definitions),
            definitions=definitions,
        )
        app = CayuApp(
            release_identity=RELEASE, secret_redactor=SecretRedactor(SECRET), enable_logging=False
        )
        app.register_agent(AgentSpec(name="billing", model="model"), mcp_toolsets=(toolset,))
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            await CloudInventoryPublisher(app, _config(), client=client).register()
        description = cloud.registered[RELEASE.id]["tools"][0]["description"]
        assert secret[:-1] not in description
        assert "[REDACTED_SECRET]" in description
        assert len(description) <= 2000
        await app.aclose()

    asyncio.run(scenario())


def test_registration_keeps_event_identifiers_instead_of_manifest_display_names():
    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="billing-0xdeadbeef", model="model"),
            tools=[_Tool("lookup-0xdeadbeef")],
        )
        registration, registered = await _build_registration(
            app, project_root=None, start_cursor=None
        )
        assert registration.agents == ("billing-0xdeadbeef",)
        assert registration.tools[0].name == "lookup-0xdeadbeef"
        assert registered.tools == frozenset({"lookup-0xdeadbeef"})
        await app.aclose()

    asyncio.run(scenario())


def test_registration_fits_when_specialist_associations_dominate_the_body():
    agents = tuple(f"agent_{i:03}_" + "a" * 118 for i in range(100))
    registration = CloudRegistration(
        agents=agents,
        tools=tuple(
            CloudTool(
                name=f"tool_{i}",
                agents=agents,
                source=CloudToolSource(kind="agent"),
                effect="none",
                approval="never",
            )
            for i in range(500)
        ),
        knowledge=CloudKnowledge(available=False, truncated=False),
    )
    fitted = _fit_registration(registration)
    assert len(_encode(fitted)) <= MAX_BODY_BYTES
    assert 0 < len(fitted.tools) < 500
    assert fitted.tools == registration.tools[: len(fitted.tools)]
    assert all(set(tool.agents) <= set(fitted.agents) for tool in fitted.tools)
    assert (
        len(
            _encode(
                fitted.model_copy(update={"tools": registration.tools[: len(fitted.tools) + 1]})
            )
        )
        > MAX_BODY_BYTES
    )


def test_registration_usage_allowlist_matches_the_fitted_tool_inventory(monkeypatch):
    monkeypatch.setattr("cayu.cloud_inventory.MAX_BODY_BYTES", 1000)

    async def scenario():
        app = CayuApp(enable_logging=False)
        for i in range(8):
            app.register_agent(
                AgentSpec(name=f"specialist_{i}_" + "a" * 20, model="model"),
                tools=[_Tool(f"tool_{j}") for j in range(8)],
            )
        registration, registered = await _build_registration(
            app, project_root=None, start_cursor=None
        )
        assert 0 < len(registration.tools) < 8
        assert len(_encode(registration)) <= 1000
        assert registered.tools == frozenset(tool.name for tool in registration.tools)
        await app.aclose()

    asyncio.run(scenario())
