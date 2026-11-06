"""Kept knowledge pages read the store a fixed number of times, whatever their size."""

from __future__ import annotations

import asyncio
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, datetime

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("psycopg")

import httpx
import psycopg

from cayu.events import Event, EventType
from cayu.knowledge.records import KnowledgeActorType, KnowledgeEntry, KnowledgeStatus
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.server import ServerConfig, create_server
from cayu.sessions.knowledge_usage import fold_knowledge_lookup_totals
from cayu.storage import PostgresKnowledgeStore, PostgresSessionStore
from cayu.storage.migrations import SchemaMode

SCOPE = KnowledgeAccessScope.privileged()


@pytest.fixture
def fresh_dsn(postgres_dsn):
    """A database of its own for each test, since every scenario seeds the same IDs."""

    from uuid import uuid4

    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    database = f"cayu_kept_{uuid4().hex}"
    with psycopg.connect(postgres_dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        yield make_conninfo(postgres_dsn, dbname=database)
    finally:
        with psycopg.connect(postgres_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )


@contextmanager
def counted_statements():
    """Count every statement any Postgres cursor sends while the block runs."""

    counts: Counter[str] = Counter()
    original = psycopg.AsyncCursor.execute

    async def execute(self, query, params=None, **kwargs):
        counts[" ".join(str(query).split())[:60]] += 1
        return await original(self, query, params, **kwargs)

    psycopg.AsyncCursor.execute = execute
    try:
        yield counts
    finally:
        psycopg.AsyncCursor.execute = original


async def _postgres_app(dsn: str):
    from tests.core.test_cloud_inventory import _usage_app

    sessions = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
    knowledge = PostgresKnowledgeStore(dsn, schema_mode=SchemaMode.CREATE)
    app = await _usage_app(store=sessions, knowledge_store=knowledge)
    return app, sessions, knowledge


def _lookup(session_id: str, namespace: object, entry_ids: object, **fields) -> Event:
    return Event(
        type=EventType.KNOWLEDGE_LOOKUP,
        session_id=session_id,
        payload={"namespace": namespace, "entry_ids": entry_ids},
        **fields,
    )


def test_postgres_kept_page_store_reads_do_not_grow_with_page_size(fresh_dsn):
    from tests.core.test_cloud_reporting import _start

    async def scenario():
        app, sessions, knowledge = await _postgres_app(fresh_dsn)
        try:
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                for index in range(12):
                    # Every entry names its own source session, so a per-entry
                    # identity read would show up in the count.
                    source = f"source-{index:02d}"
                    await _start(app, source, "keeper" if index == 5 else "writer")
                    await knowledge.create_entry(
                        KnowledgeEntry(
                            id=f"entry-{index:02d}",
                            text=f"Kept entry {index}",
                            status=(
                                KnowledgeStatus.ACTIVE
                                if index % 4 == 3
                                else KnowledgeStatus.PENDING
                            ),
                            created_by_type=KnowledgeActorType.MODEL,
                            created_by="writer",
                            source_type="tool",
                            source_id="missing-session" if index == 7 else source,
                        ),
                        access_scope=SCOPE,
                    )
                    if index % 4 != 3:
                        approved = await client.post(
                            f"/api/knowledge/entry-{index:02d}/approve",
                            headers={"Idempotency-Key": f"approve-{index}"},
                        )
                        assert approved.status_code == 200, approved.text
                    await sessions.append_event(
                        source, _lookup(source, "default", [f"entry-{index:02d}"] * 2)
                    )

                async def page(limit: int) -> tuple[dict, int]:
                    with counted_statements() as counts:
                        response = await client.get("/api/knowledge", params={"limit": limit})
                    assert response.status_code == 200, response.text
                    return response.json(), sum(counts.values())

                await page(1)  # Open the pools and caches before counting.
                small, small_count = await page(3)
                large, large_count = await page(12)
                assert small_count == large_count
                assert len(small["entries"]) == 3
                assert len(large["entries"]) == 12
                assert large["entries"][:3] == small["entries"]
                rows = {row["entry_id"]: row for row in large["entries"]}
                for index in range(12):
                    row = rows[f"entry-{index:02d}"]
                    assert row["lookups"] == "1"
                    assert row["last_used_at"] is not None
                    if index % 4 == 3:
                        assert row["attribution_unavailable_reason"] == "not_recorded"
                        assert row["kept_by_kind"] is None
                    else:
                        assert row["kept_by_kind"] == "reviewer"
                        assert row["kept_at"] is not None
                    expected = (
                        None
                        if index in {5, 7}
                        else app.project_session_id_for_exposure(f"source-{index:02d}")
                    )
                    assert row["source_session_id"] == expected
                assert large["usage_unavailable_reason"] is None
                assert large["usage_through_sequence"] > 0
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_postgres_batched_receipts_match_single_revision_reads(fresh_dsn):
    async def scenario():
        app, _sessions, knowledge = await _postgres_app(fresh_dsn)
        try:
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                for name, namespace in (("shown", "default"), ("other", "elsewhere")):
                    await knowledge.create_entry(
                        KnowledgeEntry(
                            id=f"receipt-{name}",
                            namespace=namespace,
                            text=name,
                            status=KnowledgeStatus.PENDING,
                        ),
                        access_scope=SCOPE,
                    )
                    approved = await client.post(
                        f"/api/knowledge/receipt-{name}/approve",
                        headers={"Idempotency-Key": f"receipt-{name}"},
                    )
                    assert approved.status_code == 200, approved.text
            await knowledge.create_entry(
                KnowledgeEntry(id="receipt-none", text="No receipt"), access_scope=SCOPE
            )
            revisions = [
                ("receipt-shown", 2),
                ("receipt-other", 2),
                ("receipt-shown", 1),
                ("receipt-none", 1),
                ("receipt-absent", 1),
            ]
            limited = KnowledgeAccessScope(allowed_namespaces=["default"])
            for scope in (SCOPE, limited):
                batched = await knowledge.load_entry_activation_receipts(
                    revisions, access_scope=scope
                )
                assert list(batched) == revisions
                for entry_id, revision in revisions:
                    single = await knowledge.load_entry_activation_receipt(
                        entry_id, revision=revision, access_scope=scope
                    )
                    assert batched[(entry_id, revision)] == single
            assert (
                await knowledge.load_entry_activation_receipts(revisions, access_scope=limited)
            )[("receipt-other", 2)] is None
            shown = (await knowledge.load_entry_activation_receipts(revisions, access_scope=SCOPE))[
                ("receipt-other", 2)
            ]
            assert shown is not None and shown.operation_id == "receipt-other"
            assert await knowledge.load_entry_activation_receipts([], access_scope=SCOPE) == {}
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_postgres_lookup_totals_match_the_retained_event_fold(fresh_dsn):
    async def scenario():
        app, sessions, _knowledge = await _postgres_app(fresh_dsn)
        try:
            naive = datetime(2030, 1, 2, 3, 4, 5)
            for event in (
                _lookup("session", "default", ["one", "one", "two"]),
                _lookup("session", "default", ["two", 1, {"id": "one"}]),
                _lookup("session", "other", ["one"]),
                _lookup("session", 7, ["one"]),
                _lookup("session", "default", "one"),
                _lookup("session", "default", ["1"]),
                _lookup("session", "default", ["two"], timestamp=naive),
            ):
                await sessions.append_event("session", event)
            await sessions.append_event(
                "session",
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="session",
                    payload={"namespace": "default", "entry_ids": ["one"]},
                ),
            )
            identities = {
                ("default", "one"),
                ("default", "two"),
                ("other", "one"),
                ("default", "1"),
                ("default", "unused"),
                ("default", "seed"),
            }
            native = await sessions.knowledge_lookup_totals(identities)
            folded = await fold_knowledge_lookup_totals(sessions, identities)
            assert native == folded
            totals, through = native
            assert totals[("default", "one")][0] == 1
            # A naive timestamp is UTC, as the fold reads it.
            assert totals[("default", "two")] == (3, naive.replace(tzinfo=UTC))
            assert totals[("default", "unused")] == (0, None)
            assert through > 0
            assert await sessions.knowledge_lookup_totals(set()) == ({}, through)
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_postgres_batched_identities_match_single_session_reads(fresh_dsn):
    from tests.core.test_cloud_reporting import _start

    from cayu.sessions.inspection import SESSION_INSPECTION_LABEL_LIMIT

    async def scenario():
        app, sessions, _knowledge = await _postgres_app(fresh_dsn)
        try:
            await _start(app, "labelled", "writer")
            await sessions.update_labels(
                "labelled",
                {f"key-{index:03d}": "value" for index in range(SESSION_INSPECTION_LABEL_LIMIT)},
            )
            # More labels than one inspection returns, as an older writer could leave.
            async with await psycopg.AsyncConnection.connect(fresh_dsn) as connection:
                await connection.execute(
                    "INSERT INTO cayu_session_labels (session_id, key, value) "
                    "VALUES ('labelled', 'a-first', 'x'), ('labelled', 'zz-last', 'y')"
                )
            await _start(app, "plain", "keeper")
            wanted = ["labelled", "missing", "plain", "session", "labelled"]
            batched = await sessions.inspect_identities(wanted)
            assert list(batched) == ["labelled", "plain", "session"]
            for session_id, identity in batched.items():
                assert identity == await sessions.inspect_identity(session_id)
            assert batched["labelled"].labels_truncated
            assert len(batched["labelled"].labels) == SESSION_INSPECTION_LABEL_LIMIT
            assert batched["labelled"].label_count == SESSION_INSPECTION_LABEL_LIMIT + 2
            assert "a-first" in batched["labelled"].labels
            assert "zz-last" not in batched["labelled"].labels
            assert await sessions.inspect_identities([]) == {}
            assert await sessions.inspect_identities(["missing"]) == {}
        finally:
            await app.aclose()

    asyncio.run(scenario())
