"""Kept knowledge discovery and revision-fenced retirement over HTTP."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server import ServerConfig, create_server
from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeStatus,
)


def test_kept_knowledge_lists_active_and_retires_exact_revision():
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(
        [
            KnowledgeEntry(id="kept", text="Kept entry", status=KnowledgeStatus.ACTIVE),
            KnowledgeEntry(id="pending", text="Pending entry", status=KnowledgeStatus.PENDING),
        ],
        access_scope=scope,
    )
    with TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    ) as client:
        response = client.get("/api/knowledge")
        assert response.status_code == 200
        body = response.json()
        assert [row["entry_id"] for row in body["entries"]] == ["kept"]
        assert body["entries"][0]["kept_by"] is None
        assert body["entries"][0]["kept_at"] is None
        assert body["next_cursor"] is None
        refused = client.post("/api/knowledge/kept/retire", json={"expected_revision": 2})
        assert refused.status_code == 409
        assert asyncio.run(store.get_entry("kept")).status is KnowledgeStatus.ACTIVE
        retired = client.post("/api/knowledge/kept/retire", json={"expected_revision": 1})
        assert retired.status_code == 200
        assert retired.json()["status"] == "archived"
        assert retired.json()["revision"] == 2
        assert client.get("/api/knowledge").json()["entries"] == []
        assert client.get("/api/knowledge/pending").json()["entries"][0]["entry_id"] == "pending"
        assert (
            client.post("/api/knowledge/kept/retire", json={"expected_revision": 1}).status_code
            == 409
        )


@pytest.fixture(params=["memory", "sqlite"])
def attribution_store(request, tmp_path):
    from cayu.storage import SQLiteKnowledgeStore

    store = (
        InMemoryKnowledgeStore()
        if request.param == "memory"
        else SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")
    )
    yield store
    close = getattr(store, "close", None)
    if close is not None:
        asyncio.run(close())


def test_kept_attribution_is_discovered_by_exact_native_revision(attribution_store):
    scope = KnowledgeAccessScope.privileged()
    store = attribution_store
    asyncio.run(
        store.create_entry(
            KnowledgeEntry(
                id="proposal", text="Approved knowledge", status=KnowledgeStatus.PENDING
            ),
            access_scope=scope,
        )
    )
    with TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    ) as client:
        assert (
            client.post(
                "/api/knowledge/proposal/approve", headers={"Idempotency-Key": "approval"}
            ).status_code
            == 200
        )
        receipt = asyncio.run(
            store.load_entry_activation_receipt("proposal", revision=2, access_scope=scope)
        )
        assert receipt.operation_id == "approval"
        assert receipt.entry_revision == 2
        assert (
            asyncio.run(
                store.load_entry_activation_receipt("proposal", revision=1, access_scope=scope)
            )
            is None
        )
        hidden = KnowledgeAccessScope(allowed_namespaces=["other"])
        assert (
            asyncio.run(
                store.load_entry_activation_receipt("proposal", revision=2, access_scope=hidden)
            )
            is None
        )


def test_kept_http_attribution_redaction_precedes_preview_truncation(
    attribution_store, caplog, capsys, recwarn
):
    from cayu.server import AuthContext
    from cayu.vaults.redaction import SecretRedactor

    secret = "knowledge-preview-secret-canary"
    scope = KnowledgeAccessScope.privileged()
    store = attribution_store
    asyncio.run(
        store.create_entry(
            KnowledgeEntry(
                id="reviewed", text="x" * 490 + secret + " end", status=KnowledgeStatus.PENDING
            ),
            access_scope=scope,
        )
    )
    app = CayuApp(
        knowledge_store=store,
        knowledge_access_scope=scope,
        enable_logging=False,
        secret_redactor=SecretRedactor(secret),
    )
    with TestClient(
        create_server(app, config=ServerConfig.protected(lambda _: AuthContext(subject="reviewer")))
    ) as client:
        pending = client.get("/api/inbox").json()["items"]
        assert len(pending) == 1
        assert secret[:9] not in pending[0]["detail"]
        assert (
            client.post(
                "/api/knowledge/reviewed/approve", headers={"Idempotency-Key": "keep-public"}
            ).status_code
            == 200
        )
        result = client.get("/api/knowledge?max_bytes=500")
        assert result.status_code == 200, result.text
        row = result.json()["entries"][0]
        assert row["text_preview"] == pending[0]["detail"]
        assert row["kept_by"] == "reviewer"
        assert row["kept_at"] is not None
        assert row["kept_by_kind"] == "reviewer"
        assert secret[:9] not in result.text
        assert row["lookups"] == "0"
        assert row["specialist"] is None
        captured = capsys.readouterr()
        assert secret not in captured.out + captured.err + caplog.text
        assert all(secret not in str(warning.message) for warning in recwarn)


def test_signed_member_reads_but_cannot_retire_kept_knowledge(attribution_store):
    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    key = ec.generate_private_key(ec.SECP256R1())
    scope = KnowledgeAccessScope.privileged()
    store = attribution_store
    asyncio.run(store.create_entry(KnowledgeEntry(id="kept", text="Kept"), access_scope=scope))
    with TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.protected(_auth(FakeJwks(key))),
        )
    ) as client:
        member = _as(_token(key, role="member"))
        assert client.get("/api/knowledge", headers=member).status_code == 200
        assert (
            client.post(
                "/api/knowledge/kept/retire", headers=member, json={"expected_revision": 1}
            ).status_code
            == 403
        )
        assert asyncio.run(store.get_entry("kept", access_scope=scope)).revision == 1
        assert (
            client.post(
                "/api/knowledge/kept/retire",
                headers=_as(_token(key)),
                json={"expected_revision": 1},
            ).status_code
            == 200
        )


def test_limited_kept_pages_seal_hidden_positions_and_withhold_usage(attribution_store):
    from cayu.server import AuthContext

    scope = KnowledgeAccessScope.privileged()
    store = attribution_store
    for entry_id in ("hidden", "visible"):
        asyncio.run(
            store.create_entry(KnowledgeEntry(id=entry_id, text=entry_id), access_scope=scope)
        )

    async def access(person, item):
        return item.knowledge.entry_id == "visible"

    def auth(request):
        return AuthContext(subject=request.headers.get("X-Person", "reader"))

    with TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.protected(auth),
            inbox_access=access,
        )
    ) as client:
        cursor = None
        seen = []
        pages = []
        for _ in range(3):
            params = {"limit": 1}
            if cursor is not None:
                params["cursor"] = cursor
            result = client.get("/api/knowledge", params=params)
            assert result.status_code == 200, result.text
            body = result.json()
            pages.append(body)
            seen.extend(row["entry_id"] for row in body["entries"])
            assert body["total_count"] is None
            assert body["usage_unavailable_reason"] == "access_limited"
            assert body["usage_through_sequence"] is None
            assert all(row["lookups"] is None for row in body["entries"])
            cursor = body["next_cursor"]
            if cursor is None:
                break
            assert "hidden" not in cursor
            assert (
                client.get(
                    "/api/knowledge", params={"cursor": cursor}, headers={"X-Person": "other"}
                ).status_code
                == 422
            )
        assert seen == ["visible"]
        assert len(pages) == 2
        assert any(not page["entries"] for page in pages)
        assert (
            client.post("/api/knowledge/hidden/retire", json={"expected_revision": 1}).status_code
            == 404
        )
        assert (
            client.post("/api/knowledge/hidden/retire", json={"expected_revision": 999}).status_code
            == 404
        )
        assert (
            asyncio.run(store.get_entry("hidden", access_scope=scope)).status
            is KnowledgeStatus.ACTIVE
        )


def test_retirement_cannot_archive_revision_changed_after_authorization(attribution_store):
    scope = KnowledgeAccessScope.privileged()
    store = attribution_store
    asyncio.run(store.create_entry(KnowledgeEntry(id="kept", text="First"), access_scope=scope))
    original = store.transition_entry_status
    attempted = False

    async def concurrent_change(entry_id, **kwargs):
        nonlocal attempted
        if not attempted:
            attempted = True
            # A concurrent native lifecycle change wins before this request's CAS.
            await original(
                entry_id,
                expected_revision=1,
                access_scope=scope,
                from_status=KnowledgeStatus.ACTIVE,
                to_status=KnowledgeStatus.ARCHIVED,
            )
        return await original(entry_id, **kwargs)

    store.transition_entry_status = concurrent_change
    with TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    ) as client:
        result = client.post("/api/knowledge/kept/retire", json={"expected_revision": 1})
        assert result.status_code == 409, result.text
        retained = asyncio.run(store.get_entry("kept", access_scope=scope))
        assert retained.revision == 2
        assert retained.status is KnowledgeStatus.ARCHIVED
        assert retained.text == "First"


def test_postgres_kept_attribution_and_retirement_through_http(postgres_dsn):
    import httpx

    from cayu.storage import PostgresKnowledgeStore
    from cayu.storage.migrations import SchemaMode

    async def scenario():
        scope = KnowledgeAccessScope.privileged()
        store = PostgresKnowledgeStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        app = CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False)
        try:
            await store.create_entry(
                KnowledgeEntry(id="reviewed", text="Durable", status=KnowledgeStatus.PENDING),
                access_scope=scope,
            )
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                approval = await client.post(
                    "/api/knowledge/reviewed/approve", headers={"Idempotency-Key": "kept-pg"}
                )
                assert approval.status_code == 200, approval.text
                listing = await client.get("/api/knowledge")
                assert listing.status_code == 200, listing.text
                row = listing.json()["entries"][0]
                assert row["revision"] == 2
                assert row["kept_at"] is not None
                assert row["kept_by_kind"] == "reviewer"
                receipt = await store.load_entry_activation_receipt(
                    "reviewed", revision=2, access_scope=scope
                )
                assert receipt.operation_id == "kept-pg"
                retired = await client.post(
                    "/api/knowledge/reviewed/retire", json={"expected_revision": 2}
                )
                assert retired.status_code == 200, retired.text
                assert retired.json()["revision"] == 3
                assert (await client.get("/api/knowledge")).json()["entries"] == []
                replay = await client.post(
                    "/api/knowledge/reviewed/retire", json={"expected_revision": 2}
                )
                assert replay.status_code == 409
                assert (await store.get_entry("reviewed", access_scope=scope)).text == "Durable"
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["unsupported", "source_incomplete"])
def test_kept_usage_never_publishes_zero_for_unavailable_source(failure):
    from cayu.sessions.event_queries import EventQueryResultTooLarge

    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore([KnowledgeEntry(id="kept", text="Kept")], access_scope=scope)
    app = CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False)

    async def unavailable(*args, **kwargs):
        if failure == "unsupported":
            raise NotImplementedError
        raise EventQueryResultTooLarge(kwargs["max_bytes"])

    app.session_store.fold_event_snapshot = unavailable
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        result = client.get("/api/knowledge")
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["usage_unavailable_reason"] == failure
        assert body["usage_through_sequence"] is None
        assert body["entries"][0]["lookups"] is None
        assert body["entries"][0]["last_used_at"] is None


def test_kept_source_requires_positive_registered_model_and_session_evidence():
    import httpx
    from tests.core.test_cloud_inventory import SCOPE, _usage_app

    from cayu.knowledge.records import KnowledgeActorType

    async def scenario():
        store = InMemoryKnowledgeStore()
        app = await _usage_app(knowledge_store=store)
        try:
            for entry_id, actor, source, creator in (
                ("model", KnowledgeActorType.MODEL, "session", "writer"),
                ("application", KnowledgeActorType.APP, "session", "writer"),
                ("missing", KnowledgeActorType.MODEL, "gone", "writer"),
                ("wrong-agent", KnowledgeActorType.MODEL, "session", "keeper"),
            ):
                await store.create_entry(
                    KnowledgeEntry(
                        id=entry_id,
                        text="Entry",
                        created_by_type=actor,
                        created_by=creator,
                        source_type="tool",
                        source_id=source,
                        metadata={
                            "reviewer": "forged",
                            "kept_at": "forged",
                            "specialist": "forged",
                        },
                    ),
                    access_scope=SCOPE,
                )
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                result = await client.get("/api/knowledge")
                assert result.status_code == 200, result.text
                rows = {row["entry_id"]: row for row in result.json()["entries"]}
                assert rows["model"]["specialist"] == "writer"
                assert rows["model"]["source_session_id"] == app.project_session_id_for_exposure(
                    "session"
                )
                assert rows["application"]["specialist"] is None
                for name in ("application", "missing", "wrong-agent"):
                    assert rows[name]["source_session_id"] is None
                    assert rows[name]["source_id"] is None
                assert all(row["kept_by"] is None for row in rows.values())
                assert all(row["kept_at"] is None for row in rows.values())
                assert "forged" not in result.text
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_kept_receipt_attribution_survives_sqlite_reopen(tmp_path):
    from cayu.storage import SQLiteKnowledgeStore

    scope = KnowledgeAccessScope.privileged()
    path = tmp_path / "reopened.sqlite"
    store = SQLiteKnowledgeStore(path)
    asyncio.run(
        store.create_entry(
            KnowledgeEntry(id="reviewed", text="Durable", status=KnowledgeStatus.PENDING),
            access_scope=scope,
        )
    )
    with TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    ) as client:
        assert (
            client.post(
                "/api/knowledge/reviewed/approve", headers={"Idempotency-Key": "reopen"}
            ).status_code
            == 200
        )
        response = client.get("/api/knowledge")
        assert response.status_code == 200, response.text
        before = response.json()["entries"][0]
    asyncio.run(store.close())
    reopened = SQLiteKnowledgeStore(path)
    try:
        with TestClient(
            create_server(
                CayuApp(
                    knowledge_store=reopened, knowledge_access_scope=scope, enable_logging=False
                ),
                config=ServerConfig.local_development(),
            )
        ) as client:
            result = client.get("/api/knowledge")
            assert result.status_code == 200, result.text
            after = result.json()["entries"][0]
            assert after["kept_at"] == before["kept_at"]
            assert after["kept_by"] == before["kept_by"]
            assert after["kept_by_kind"] == "reviewer"
            assert (
                client.post(
                    "/api/knowledge/reviewed/retire", json={"expected_revision": 2}
                ).status_code
                == 200
            )
    finally:
        asyncio.run(reopened.close())


def test_optional_receipt_discovery_is_explicitly_unavailable():
    from cayu.knowledge.base import KnowledgeStore

    class WithoutDiscovery(InMemoryKnowledgeStore):
        load_entry_activation_receipt = KnowledgeStore.load_entry_activation_receipt

    scope = KnowledgeAccessScope.privileged()
    store = WithoutDiscovery([KnowledgeEntry(id="kept", text="Kept")], access_scope=scope)
    with TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    ) as client:
        result = client.get("/api/knowledge")
        assert result.status_code == 200, result.text
        row = result.json()["entries"][0]
        assert row["attribution_unavailable_reason"] == "unsupported"
        assert row["kept_by"] is None
        assert row["kept_at"] is None
        assert row["kept_by_kind"] is None


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("requested_revision", [1, 2])
def test_retirement_cannot_use_a_revision_not_authorized_by_item_gate(
    backend, requested_revision, request, tmp_path
):
    from uuid import uuid4

    import httpx

    from cayu.server import AuthContext
    from cayu.storage import PostgresKnowledgeStore, SQLiteKnowledgeStore
    from cayu.storage.migrations import SchemaMode

    entry_id = "kept-" + uuid4().hex
    scope = KnowledgeAccessScope.privileged()
    store = (
        InMemoryKnowledgeStore()
        if backend == "memory"
        else SQLiteKnowledgeStore(tmp_path / "gate.sqlite")
        if backend == "sqlite"
        else PostgresKnowledgeStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    )

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        allowed_revisions = []

        async def access(person, item):
            allowed_revisions.append(item.knowledge.revision)
            entered.set()
            await release.wait()
            return item.knowledge.revision == 1

        initial = await store.create_entry(
            KnowledgeEntry(id=entry_id, text="Allowed"), access_scope=scope
        )
        app = CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False)
        server = create_server(
            app,
            config=ServerConfig.protected(lambda _: AuthContext(subject="reader")),
            inbox_access=access,
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                response_task = asyncio.create_task(
                    client.post(
                        f"/api/knowledge/{entry_id}/retire",
                        json={"expected_revision": requested_revision},
                    )
                )
                gate_task = asyncio.create_task(entered.wait())
                done, _ = await asyncio.wait(
                    [response_task, gate_task], return_when=asyncio.FIRST_COMPLETED
                )
                if gate_task in done:
                    await store.append_entry_revision(
                        initial.model_copy(update={"revision": 2, "text": "Denied successor"}),
                        expected_revision=1,
                        access_scope=scope,
                    )
                    release.set()
                else:
                    gate_task.cancel()
                    await asyncio.gather(gate_task, return_exceptions=True)
                response = await response_task
                current = await store.get_entry(entry_id, access_scope=scope)
                assert response.status_code == 409, response.text
                assert current.status is KnowledgeStatus.ACTIVE
                assert current.revision in (1, 2)
                assert allowed_revisions in ([], [1])
        finally:
            release.set()
            await app.aclose()
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


@pytest.mark.parametrize("session_backend", ["memory", "sqlite"])
def test_kept_source_uses_one_bounded_identity_read_for_repeated_large_session(
    session_backend, tmp_path
):
    import httpx
    from tests.core.test_cloud_inventory import SCOPE, _usage_app

    from cayu.knowledge.records import KnowledgeActorType

    async def scenario():
        from cayu.storage import SQLiteSessionStore

        store = InMemoryKnowledgeStore()
        sessions = (
            SQLiteSessionStore(tmp_path / "source.sqlite") if session_backend == "sqlite" else None
        )
        app = await _usage_app(store=sessions, knowledge_store=store)
        try:
            await app.session_store.update_metadata("session", {"large": "x" * 200_000})
            await app.session_store.update_labels(
                "session", {f"key-{i}": "value" for i in range(200)}
            )
            for name in ("one", "two"):
                await store.create_entry(
                    KnowledgeEntry(
                        id=name,
                        text="Small",
                        created_by_type=KnowledgeActorType.MODEL,
                        created_by="writer",
                        source_type="tool",
                        source_id="session",
                    ),
                    access_scope=SCOPE,
                )
            calls = []
            inspect = app.session_store.inspect_identity

            async def bounded(session_id):
                calls.append(session_id)
                return await inspect(session_id)

            async def forbidden_load(*args, **kwargs):
                raise AssertionError("Source attribution hydrated the full session")

            app.session_store.inspect_identity = bounded
            app.session_store.load = forbidden_load
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                result = await client.get("/api/knowledge", params={"max_bytes": 100})
                assert result.status_code == 200, result.text
                assert len(result.json()["entries"]) == 2
                assert all(
                    row["source_session_id"] == app.project_session_id_for_exposure("session")
                    for row in result.json()["entries"]
                )
                assert calls == ["session"]
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_postgres_kept_usage_counts_committed_lookup_behind_open_transaction(postgres_dsn):
    import httpx
    import psycopg
    from psycopg.types.json import Jsonb
    from tests.core.test_cloud_inventory import SCOPE, _usage_app

    from cayu.events import Event, EventType
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def scenario():
        sessions = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        knowledge = InMemoryKnowledgeStore()
        app = await _usage_app(store=sessions, knowledge_store=knowledge)
        await knowledge.create_entry(
            KnowledgeEntry(id="usage-race", text="Exact"), access_scope=SCOPE
        )
        server = create_server(app, config=ServerConfig.local_development())

        async def lookup():
            await sessions.append_event(
                "session",
                Event(
                    type=EventType.KNOWLEDGE_LOOKUP,
                    session_id="session",
                    payload={"namespace": "default", "entry_ids": ["usage-race", "usage-race"]},
                ),
            )

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                await lookup()
                normal = await client.get("/api/knowledge")
                assert normal.status_code == 200, normal.text
                assert normal.json()["entries"][0]["lookups"] == "1"
                async with (
                    await psycopg.AsyncConnection.connect(postgres_dsn) as older,
                    await psycopg.AsyncConnection.connect(postgres_dsn) as blocker,
                ):
                    await older.execute("SELECT pg_current_xact_id()")
                    await blocker.execute("SELECT pg_current_xact_id()")
                    # Newer transaction commits a lookup, then older xid inserts
                    # a higher sequence while blocker holds the xmin boundary.
                    await lookup()
                    event = Event(type=EventType.SESSION_STARTED, session_id="session")
                    await older.execute(
                        "INSERT INTO cayu_events (session_id, session_order, event_id, event_type, timestamp, payload, event) "
                        "SELECT %s, COALESCE(MAX(session_order), 0) + 1, %s, %s, %s, %s, %s FROM cayu_events WHERE session_id = %s",
                        (
                            "session",
                            event.id,
                            str(event.type),
                            event.timestamp,
                            Jsonb(event.payload),
                            Jsonb(event.model_dump(mode="json")),
                            "session",
                        ),
                    )
                    await older.commit()
                    result = await client.get("/api/knowledge")
                    assert result.status_code == 200, result.text
                    assert result.json()["usage_unavailable_reason"] is None
                    assert result.json()["entries"][0]["lookups"] == "2"
                    assert result.json()["entries"][0]["last_used_at"] is not None
                    await blocker.rollback()
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("configured_key", [False, True])
@pytest.mark.parametrize("invalid_kind", ["malformed", "expired"])
def test_kept_cursor_error_names_knowledge_listing(configured_key, invalid_kind, monkeypatch):
    from types import SimpleNamespace

    from cayu.server import _inbox_cursor

    now = [1_000.0]
    monkeypatch.setattr(_inbox_cursor, "time", SimpleNamespace(time=lambda: now[0]))
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(
        [KnowledgeEntry(id=name, text=name) for name in ("first", "second")], access_scope=scope
    )
    with TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.local_development(),
            inbox_cursor_key=b"K" * 32 if configured_key else None,
        )
    ) as client:
        cursor = "not-a-cursor"
        if invalid_kind == "expired":
            first = client.get("/api/knowledge", params={"limit": 1})
            assert first.status_code == 200
            cursor = first.json()["next_cursor"]
            assert cursor is not None
            now[0] += 901
        result = client.get("/api/knowledge", params={"cursor": cursor})
        assert result.status_code == 422
        assert "Restart the knowledge listing." in result.json()["detail"]
        assert "inbox listing" not in result.json()["detail"]
        assert result.headers["cache-control"] == "private, no-store"


def test_kept_entry_leaves_out_a_chunk_cut_inside_a_multibyte_character(monkeypatch):
    from cayu.server import _kept_knowledge

    # A 64-byte read window; the entry's one chunk is 65 bytes and ends in a 3-byte
    # character, so the store's UTF-8 cut returns 62 bytes, short of the window.
    monkeypatch.setattr(_kept_knowledge, "_CHUNK_READ_BYTES", 64)
    scope = KnowledgeAccessScope.privileged()
    store = InMemoryKnowledgeStore(
        [KnowledgeEntry(id="kept", text="a" * 62 + "€", status=KnowledgeStatus.ACTIVE)],
        access_scope=scope,
    )
    with TestClient(
        create_server(
            CayuApp(knowledge_store=store, knowledge_access_scope=scope, enable_logging=False),
            config=ServerConfig.local_development(),
        )
    ) as client:
        response = client.get("/api/knowledge/kept/kept")
        assert response.status_code == 200
        assert response.json()["chunks"] == []
