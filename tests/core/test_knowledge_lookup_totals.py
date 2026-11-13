"""Stored knowledge lookup totals equal a recompute from the events, whatever happened.

SQLite and Postgres keep ``cayu_knowledge_lookups`` and
``cayu_knowledge_lookup_totals`` with triggers on ``cayu_events``. For random
histories of lookups, malformed lookups, payload rewrites and session deletions,
the stored totals must equal ``recompute_knowledge_lookup_totals`` (the
reference fold over the events), ``knowledge_lookup_totals`` must equal
``fold_knowledge_lookup_totals``, and a migration must rebuild them from the
events retained before it.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
from typing import LiteralString, cast
from uuid import uuid4

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from cayu import Event, EventType, Message, RunRequest
from cayu.sessions.knowledge_usage import (
    check_knowledge_lookup_totals,
    fold_knowledge_lookup_totals,
    recompute_knowledge_lookup_totals,
)
from cayu.sessions.records import SessionIdentity
from cayu.storage._knowledge_lookup_schema import KNOWLEDGE_LOOKUP_TOTALS_REVISION
from cayu.storage.migrations import SchemaMode, SchemaTooOld
from cayu.storage.sqlite import SQLiteSessionStore

_ENTRIES = ("a", "b", "c")
_TABLES = ("cayu_knowledge_lookup_totals", "cayu_knowledge_lookups")


class _Backend:
    def __init__(self, name: str, tmp_path, request) -> None:
        self.name = name
        if name == "sqlite":
            self.path = tmp_path / "lookups.sqlite"
        else:
            self.dsn = request.getfixturevalue("postgres_dsn")

    async def open(self, schema_mode: SchemaMode = SchemaMode.CREATE):
        if self.name == "sqlite":
            return SQLiteSessionStore(self.path, schema_mode=schema_mode)
        from cayu.storage.postgres import PostgresSessionStore

        store = PostgresSessionStore(self.dsn, schema_mode=schema_mode)
        try:
            await store.ensure_schema()
        except BaseException:
            await store.close()
            raise
        return store

    async def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        if self.name == "sqlite":
            connection = sqlite3.connect(self.path)
            try:
                rows = [tuple(row) for row in connection.execute(sql, params).fetchall()]
                connection.commit()
                return rows
            finally:
                connection.close()
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.dsn, autocommit=True) as conn:
            cursor = await conn.execute(cast("LiteralString", sql.replace("?", "%s")), params)
            return [tuple(row) for row in await cursor.fetchall()] if cursor.description else []


@pytest.fixture(params=["sqlite", "postgres"])
def backend(request, tmp_path) -> _Backend:
    return _Backend(request.param, tmp_path, request)


def _lookup(session_id: str, namespace: object, entry_ids: object) -> Event:
    return Event(
        type=EventType.KNOWLEDGE_LOOKUP,
        session_id=session_id,
        payload={"namespace": namespace, "entry_ids": entry_ids},
    )


async def _create(store, session_id: str) -> None:
    await store.create(
        RunRequest(
            session_id=session_id, agent_name="assistant", messages=[Message.text("user", "hi")]
        ),
        identity=SessionIdentity(provider_name="anthropic", model="claude-x"),
    )


def _scoped(totals: dict, namespaces: set[str]) -> dict:
    return {key: value for key, value in totals.items() if key[0] in namespaces}


_ids = st.lists(st.sampled_from([*_ENTRIES, 7, None]), max_size=4)
_step = st.one_of(
    st.tuples(st.just("lookup"), st.integers(0, 2), st.integers(0, 1), _ids, st.integers(1, 3)),
    st.tuples(st.just("malformed"), st.integers(0, 2)),
    st.tuples(st.just("rewrite"), st.integers(0, 2), _ids),
    st.tuples(st.just("delete"), st.integers(0, 2)),
)


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(history=st.lists(_step, min_size=1, max_size=10))
def test_stored_totals_equal_a_recompute_for_random_histories(backend, history) -> None:
    run = uuid4().hex[:8]
    namespaces = [f"{run}-0", f"{run}-1"]

    async def scenario():
        store = await backend.open()
        try:
            sessions = [f"{run}-s{index}" for index in range(3)]
            live: set[str] = set()
            for step in history:
                session_id = sessions[step[1]]
                if session_id not in live and step[0] != "delete":
                    await _create(store, session_id)
                    live.add(session_id)
                if step[0] == "lookup":
                    _, _, namespace, ids, repeat = step
                    await store.append_events(
                        session_id,
                        [_lookup(session_id, namespaces[namespace], ids) for _ in range(repeat)],
                    )
                elif step[0] == "malformed":
                    await store.append_events(
                        session_id,
                        [
                            _lookup(session_id, 5, list(_ENTRIES)),
                            _lookup(session_id, namespaces[0], "a"),
                        ],
                    )
                elif step[0] == "rewrite":
                    # A rewritten payload moves the event's contribution with it.
                    rewritten = json.dumps({"namespace": namespaces[1], "entry_ids": step[2]})
                    latest = (
                        "WHERE sequence = (SELECT MAX(sequence) FROM cayu_events "
                        "WHERE session_id = ? AND event_type = 'knowledge.lookup')"
                    )
                    if backend.name == "sqlite":
                        await backend.execute(
                            f"UPDATE cayu_events SET payload_json = ? {latest}",
                            (rewritten, session_id),
                        )
                    else:
                        await backend.execute(
                            "UPDATE cayu_events SET payload = ?::jsonb, "
                            f"event = jsonb_set(event, '{{payload}}', ?::jsonb) {latest}",
                            (rewritten, rewritten, session_id),
                        )
                elif session_id in live:
                    await store.delete_session(session_id)
                    live.discard(session_id)
            stored = _scoped(await store.stored_knowledge_lookup_totals(), set(namespaces))
            recomputed, _ = await recompute_knowledge_lookup_totals(store)
            wanted = {(namespace, entry) for namespace in namespaces for entry in _ENTRIES}
            native, _ = await store.knowledge_lookup_totals(wanted)
            folded, _ = await fold_knowledge_lookup_totals(store, wanted)
            for session_id in live:
                await store.delete_session(session_id)
            leftover = _scoped(await store.stored_knowledge_lookup_totals(), set(namespaces))
            return stored, _scoped(recomputed, set(namespaces)), native, folded, leftover
        finally:
            await store.close()

    stored, recomputed, native, folded, leftover = asyncio.run(scenario())
    assert stored == recomputed
    assert native == folded
    assert leftover == {}


def test_migration_rebuilds_totals_from_retained_lookups(backend) -> None:
    session_id = "migrate-" + uuid4().hex
    namespace = session_id

    async def scenario():
        store = await backend.open()
        try:
            await _create(store, session_id)
            await store.append_events(
                session_id,
                [
                    _lookup(session_id, namespace, ["a", "a", "b"]),
                    _lookup(session_id, namespace, ["a"]),
                ],
            )
            before = _scoped(await store.stored_knowledge_lookup_totals(), {namespace})
        finally:
            await store.close()
        on_events = "" if backend.name == "sqlite" else " ON cayu_events"
        on_lookups = "" if backend.name == "sqlite" else " ON cayu_knowledge_lookups"
        for trigger in ("insert", "delete", "update"):
            await backend.execute(f"DROP TRIGGER cayu_knowledge_lookups_{trigger}{on_events}")
        if backend.name == "sqlite":
            for trigger in ("add", "remove"):
                await backend.execute(f"DROP TRIGGER cayu_knowledge_lookup_totals_{trigger}")
        else:
            for trigger in ("fold", "unfold"):
                await backend.execute(f"DROP TRIGGER cayu_knowledge_lookups_{trigger}{on_lookups}")
        for table in _TABLES:
            await backend.execute(f"DROP TABLE {table}")
        if backend.name == "sqlite":
            await backend.execute("DROP TABLE cayu_knowledge_entry_order")
            await backend.execute("DROP INDEX idx_cayu_checkpoints_pending_action_counts")
        else:
            await backend.execute("DROP TABLE cayu_knowledge_entry_order")
            await backend.execute("DROP TABLE cayu_knowledge_lookup_removals")
        # Back to the revision before the lookup totals.
        await backend.execute(
            f"DELETE FROM cayu_schema_migrations WHERE revision >= {KNOWLEDGE_LOOKUP_TOTALS_REVISION}"
        )
        with pytest.raises(SchemaTooOld, match="requires >= "):
            await (await backend.open()).close()
        store = await backend.open(SchemaMode.MIGRATE)
        try:
            after = _scoped(await store.stored_knowledge_lookup_totals(), {namespace})
            mismatches = [
                mismatch
                for mismatch in (await check_knowledge_lookup_totals(store)).mismatches
                if mismatch.namespace == namespace
            ]
            await store.delete_session(session_id)
        finally:
            await store.close()
        return before, after, mismatches

    before, after, mismatches = asyncio.run(scenario())
    assert set(before) == {(namespace, "a"), (namespace, "b")}
    assert (before[(namespace, "a")].lookups, before[(namespace, "a")].sessions) == (2, 1)
    assert after == before
    assert mismatches == []


def test_postgres_deletes_take_no_totals_lock_before_commit(postgres_dsn) -> None:
    """A transaction deleting lookups one statement at a time can't block a writer.

    Deleting ``b``'s lookup and then ``a``'s, while another session's lookup of
    both commits in between, used to lock ``totals(b)`` early and deadlock with
    that writer's fold of ``a`` then ``b``. Totals now change only at commit.
    """

    import psycopg

    from cayu.storage.postgres import PostgresSessionStore

    namespace = "locks-" + uuid4().hex[:8]
    deleting, writing = f"{namespace}-deleting", f"{namespace}-writing"

    async def scenario():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await store.ensure_schema()
            for session_id in (deleting, writing):
                await _create(store, session_id)
            await store.append_events(
                deleting,
                [_lookup(deleting, namespace, ["b"]), _lookup(deleting, namespace, ["a"])],
            )
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
                cursor = await conn.execute(
                    "SELECT sequence FROM cayu_events WHERE session_id = %s "
                    "AND event_type = 'knowledge.lookup' ORDER BY sequence",
                    (deleting,),
                )
                first, second = [row[0] for row in await cursor.fetchall()]
                await conn.execute("DELETE FROM cayu_events WHERE sequence = %s", (first,))
                # Under the old unfold this waited on totals(b) forever.
                await asyncio.wait_for(
                    store.append_events(writing, [_lookup(writing, namespace, ["a", "b"])]),
                    timeout=10,
                )
                await conn.execute("DELETE FROM cayu_events WHERE sequence = %s", (second,))
                await conn.commit()
            stored = _scoped(await store.stored_knowledge_lookup_totals(), {namespace})
            check = await check_knowledge_lookup_totals(store)
            for session_id in (deleting, writing):
                await store.delete_session(session_id)
            return stored, [m for m in check.mismatches if m.namespace == namespace]
        finally:
            await store.close()

    stored, mismatches = asyncio.run(scenario())
    assert {key: (total.lookups, total.sessions) for key, total in stored.items()} == {
        (namespace, "a"): (1, 1),
        (namespace, "b"): (1, 1),
    }
    assert mismatches == []


def test_cli_check_reports_a_stored_count_that_differs(tmp_path) -> None:
    path = tmp_path / "check.sqlite"

    async def seed():
        store = SQLiteSessionStore(path)
        try:
            await _create(store, "checked")
            await store.append_events("checked", [_lookup("checked", "default", ["a", "b"])])
        finally:
            await store.close()

    asyncio.run(seed())

    def check() -> tuple[int, dict]:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "cayu",
                "knowledge",
                "lookups",
                "check",
                "--sqlite",
                str(path),
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode, json.loads(completed.stdout)

    code, body = check()
    assert code == 0, body
    assert body["matches"] is True
    assert body["entries_stored"] == 2
    connection = sqlite3.connect(path)
    connection.execute("UPDATE cayu_knowledge_lookup_totals SET lookups = 5 WHERE entry_id = 'a'")
    connection.commit()
    connection.close()
    code, body = check()
    assert code == 1
    [mismatch] = body["mismatches"]
    assert mismatch["entry_id"] == "a"
    assert mismatch["stored"]["lookups"] == 5
    assert mismatch["recomputed"]["lookups"] == 1
