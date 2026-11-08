"""SQL stores keep the invocation lifecycle receipt ledger out of the checkpoint row.

The checkpoint every store read returns still carries the whole ledger, so each
case compares a store with the new layout to the same store keeping the ledger
inline, as before, and checks what each one stored.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu._validation import measure_durable_json
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.checkpoints import INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY
from cayu.storage import SQLiteSessionStore
from cayu.storage import _sqlite_records as sqlite_records
from cayu.storage import postgres as postgres_store
from cayu.storage._receipt_ledger_rows import LEDGER_REFERENCE_RECORD_TYPE
from cayu.tools.base import ToolContext

LEDGER = INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY


class _NoteTool(Tool):
    spec = ToolSpec(
        name="note",
        parallel_safe=False,
        max_terminal_payload_bytes=1024,
        # A stable identity, so a later process can resume the session.
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:receipt-ledger-rows:note", behavior_version="1", implementation_version="1"
        ),
        input_schema={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content=f"noted {args['index']}")


def _batches(rounds: int) -> list[list[ModelStreamEvent]]:
    return [
        [
            ModelStreamEvent.tool_call(id=f"call-{index}", name="note", arguments={"index": index}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
        for index in range(rounds)
    ] + [
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({"finish_reason": "stop"})]
    ]


@contextmanager
def _ledger_kept_inline(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Store receipt ledgers inline inside checkpoint rows."""

    with monkeypatch.context() as patch:
        patch.setattr(sqlite_records, "store_receipt_ledger", lambda _c, _s, checkpoint: checkpoint)

        async def inline(_cur: Any, _session_id: str, checkpoint: dict[str, Any]) -> Any:
            return checkpoint

        patch.setattr(postgres_store, "_store_receipt_ledger", inline)
        yield


def _store(backend: str, request: pytest.FixtureRequest, tmp_path: Any, name: str) -> Any:
    if backend == "sqlite":
        return SQLiteSessionStore(tmp_path / f"{name}.sqlite")
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    return PostgresSessionStore(
        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
    )


async def _run(store: Any, session_id: str, rounds: int, *, resume: bool = False) -> list[str]:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider(_batches(rounds)), default=True)
    app.register_agent(AgentSpec(name="worker", model="scripted-model"), tools=[_NoteTool()])
    if resume:
        request = ResumeRequest(
            session_id=session_id, max_steps=rounds + 1, messages=[Message.text("user", "again")]
        )
        return [str(event.type) async for event in app.resume(request)]
    run = RunRequest(
        session_id=session_id,
        agent_name="worker",
        max_steps=rounds + 1,
        messages=[Message.text("user", "take notes")],
    )
    return [str(event.type) async for event in app.run(run)]


async def _stored(store: Any, session_id: str) -> tuple[dict[str, Any], list[tuple[str, Any]]]:
    """The checkpoint row as stored, and the session's ledger rows."""

    if isinstance(store, SQLiteSessionStore):

        def read(connection: sqlite3.Connection) -> Any:
            state = connection.execute(
                "SELECT state_json FROM cayu_checkpoints WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            rows = connection.execute(
                "SELECT record_sha256, ledger_json FROM cayu_invocation_receipt_ledgers "
                "WHERE session_id = ? ORDER BY record_sha256",
                (session_id,),
            ).fetchall()
            return json.loads(state), [(row[0], json.loads(row[1])) for row in rows]

        return await store._run_read(read)
    async with store._connection() as conn, conn.cursor() as cur:
        await cur.execute("SELECT state FROM cayu_checkpoints WHERE session_id = %s", (session_id,))
        state = (await cur.fetchone())[0]
        await cur.execute(
            "SELECT record_sha256, ledger FROM cayu_invocation_receipt_ledgers "
            "WHERE session_id = %s ORDER BY record_sha256",
            (session_id,),
        )
        return state, [(row[0], row[1]) for row in await cur.fetchall()]


def _receipt_shape(checkpoint: dict[str, Any]) -> list[tuple[str, str, str]]:
    ledger = checkpoint[LEDGER]
    return [
        (receipt["kind"], receipt["result_session"]["status"], receipt["command_identity"])
        for receipt in ledger["receipts"]
    ]


def _without_volatile(checkpoint: dict[str, Any]) -> list[str]:
    return sorted(checkpoint)


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_the_checkpoint_row_refers_to_a_ledger_kept_in_its_own_row(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> Any:
        before = _store(backend, request, tmp_path, "before")
        try:
            with _ledger_kept_inline(monkeypatch):
                before_events = await _run(before, f"inline-{backend}", 2)
            before_checkpoint = await before.load_checkpoint(f"inline-{backend}")
            before_row, before_ledgers = await _stored(before, f"inline-{backend}")
        finally:
            await before.close()
        after = _store(backend, request, tmp_path, "after")
        try:
            after_events = await _run(after, f"rows-{backend}", 2)
            after_checkpoint = await after.load_checkpoint(f"rows-{backend}")
            after_row, after_ledgers = await _stored(after, f"rows-{backend}")
            page = await after.load_checkpoints([f"rows-{backend}"])
            fields = await after.load_checkpoint_fields([f"rows-{backend}"], [LEDGER])
        finally:
            await after.close()
        return (
            (before_events, before_checkpoint, before_row, before_ledgers),
            (after_events, after_checkpoint, after_row, after_ledgers),
            page,
            fields,
        )

    before, after, page, fields = asyncio.run(scenario())
    before_events, before_checkpoint, before_row, before_ledgers = before
    after_events, after_checkpoint, after_row, after_ledgers = after
    session_id = f"rows-{backend}"

    # The session ran the same, and every read returns the same logical checkpoint.
    assert after_events == before_events
    assert after_events[-1] == EventType.SESSION_COMPLETED
    assert _without_volatile(after_checkpoint) == _without_volatile(before_checkpoint)
    assert [shape[:2] for shape in _receipt_shape(after_checkpoint)] == [
        shape[:2] for shape in _receipt_shape(before_checkpoint)
    ]
    assert page == {session_id: after_checkpoint}
    assert fields == {session_id: {LEDGER: after_checkpoint[LEDGER]}}

    # Before, the row held the ledger; now it holds a reference to its own row.
    assert before_row[LEDGER] == before_checkpoint[LEDGER]
    assert before_ledgers == []
    ledger = after_checkpoint[LEDGER]
    ledger_bytes, ledger_nodes = measure_durable_json(ledger, "ledger")
    assert after_row[LEDGER] == {
        "record_type": LEDGER_REFERENCE_RECORD_TYPE,
        "schema_version": 1,
        "record_sha256": ledger["record_sha256"],
        "ledger_bytes": ledger_bytes,
        "ledger_nodes": ledger_nodes,
    }
    # The version the row refers to, and at most the one before it.
    assert (ledger["record_sha256"], ledger) in after_ledgers
    assert len(after_ledgers) <= 2
    assert {key: value for key, value in after_row.items() if key != LEDGER} == {
        key: value for key, value in after_checkpoint.items() if key != LEDGER
    }
    assert len(json.dumps(after_row)) < len(json.dumps(before_row)) - len(json.dumps(ledger)) / 2


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_a_row_written_with_the_ledger_inline_is_read_and_moved_out_by_the_next_write(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = f"legacy-{backend}"

    async def scenario() -> Any:
        store = _store(backend, request, tmp_path, "legacy")
        try:
            with _ledger_kept_inline(monkeypatch):
                await _run(store, session_id, 1)
            inline_row, inline_ledgers = await _stored(store, session_id)
            inline_checkpoint = await store.load_checkpoint(session_id)
            events = await _run(store, session_id, 1, resume=True)
            row, ledgers = await _stored(store, session_id)
            checkpoint = await store.load_checkpoint(session_id)
        finally:
            await store.close()
        return inline_row, inline_ledgers, inline_checkpoint, events, row, ledgers, checkpoint

    inline_row, inline_ledgers, inline_checkpoint, events, row, ledgers, checkpoint = asyncio.run(
        scenario()
    )

    assert inline_ledgers == [] and inline_row[LEDGER] == inline_checkpoint[LEDGER]
    assert events[-1] == EventType.SESSION_COMPLETED
    assert row[LEDGER]["record_type"] == LEDGER_REFERENCE_RECORD_TYPE
    assert (checkpoint[LEDGER]["record_sha256"], checkpoint[LEDGER]) in ledgers
    assert len(ledgers) <= 2
    # The second run's receipts were added to the first run's, as with an inline ledger.
    first = {identity for *_, identity in _receipt_shape(inline_checkpoint)}
    assert first < {identity for *_, identity in _receipt_shape(checkpoint)}


@pytest.mark.parametrize("commit", [False, True], ids=["rolled back", "committed"])
def test_a_ledger_row_written_without_its_checkpoint_row_changes_nothing_read(
    tmp_path: Any, commit: bool
) -> None:
    """A write that stores a ledger row and then fails, or never writes its checkpoint row.

    Rolled back, both rows are as they were. Committed, the checkpoint row still
    refers to the version it did, which a new version never removes. Either way
    a new process reads the same checkpoint and continues the session.
    """

    session_id = "atomic"
    path = tmp_path / "atomic.sqlite"

    async def scenario() -> Any:
        store = SQLiteSessionStore(path)
        try:
            await _run(store, session_id, 1)
            earlier = await store.load_checkpoint(session_id)
            # Two more turns, so the earliest ledger version is gone from its rows.
            await _run(store, session_id, 1, resume=True)
            await _run(store, session_id, 1, resume=True)
            row, ledgers = await _stored(store, session_id)
            assert earlier[LEDGER]["record_sha256"] not in {digest for digest, _ in ledgers}
            checkpoint = await store.load_checkpoint(session_id)
            assert earlier[LEDGER] != checkpoint[LEDGER]

            def write_ledger_only(connection: sqlite3.Connection) -> None:
                connection.execute("BEGIN IMMEDIATE")
                sqlite_records.checkpoint_row(connection, session_id, earlier, earlier_time)
                if commit:
                    connection.commit()
                else:
                    connection.rollback()

            from datetime import UTC, datetime

            earlier_time = datetime.now(UTC)
            await store._run_write(write_ledger_only)
            after_row, after_ledgers = await _stored(store, session_id)
        finally:
            await store.close()
        reopened = SQLiteSessionStore(path)
        try:
            reopened_checkpoint = await reopened.load_checkpoint(session_id)
            events = await _run(reopened, session_id, 1, resume=True)
            final_checkpoint = await reopened.load_checkpoint(session_id)
            _, final_ledgers = await _stored(reopened, session_id)
        finally:
            await reopened.close()
        return (
            row,
            ledgers,
            checkpoint,
            earlier,
            after_row,
            after_ledgers,
            reopened_checkpoint,
            events,
            final_checkpoint,
            final_ledgers,
        )

    (
        row,
        ledgers,
        checkpoint,
        earlier,
        after_row,
        after_ledgers,
        reopened_checkpoint,
        events,
        final_checkpoint,
        final_ledgers,
    ) = asyncio.run(scenario())

    assert after_row == row
    digests = {digest for digest, _ in after_ledgers}
    if commit:
        assert digests == {
            checkpoint[LEDGER]["record_sha256"],
            earlier[LEDGER]["record_sha256"],
        }
    else:
        assert after_ledgers == ledgers
    assert reopened_checkpoint == checkpoint
    assert events[-1] == EventType.SESSION_COMPLETED
    assert {digest for digest, _ in final_ledgers} >= {final_checkpoint[LEDGER]["record_sha256"]}
    assert len(final_ledgers) <= 2


def test_a_reference_without_its_ledger_row_fails_closed(tmp_path: Any) -> None:
    async def scenario() -> None:
        store = SQLiteSessionStore(tmp_path / "missing.sqlite")
        try:
            await _run(store, "missing", 1)

            def drop(connection: sqlite3.Connection) -> None:
                connection.execute("DELETE FROM cayu_invocation_receipt_ledgers")
                connection.commit()

            await store._run_write(drop)
            with pytest.raises(RuntimeError, match="receipt ledger is missing"):
                await store.load_checkpoint("missing")
        finally:
            await store.close()

    asyncio.run(scenario())


def test_a_session_keeps_at_most_two_ledger_versions(tmp_path: Any) -> None:
    async def scenario() -> list[Counter[str]]:
        store = SQLiteSessionStore(tmp_path / "versions.sqlite")
        seen = []
        try:
            for turn in range(3):
                await _run(store, "versions", 1, resume=turn > 0)
                _, ledgers = await _stored(store, "versions")
                seen.append(Counter(digest for digest, _ in ledgers))
        finally:
            await store.close()
        return seen

    for versions in asyncio.run(scenario()):
        assert 1 <= len(versions) <= 2


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_a_write_that_fits_only_without_its_ledger_is_refused(
    backend: str, request: pytest.FixtureRequest, tmp_path: Any
) -> None:
    """A generic transform never sees the ledger, but the whole checkpoint must still fit."""

    from cayu._validation import DURABLE_DOCUMENT_LIMITS, DurableValueError, measure_durable_json

    session_id = f"limits-{backend}"

    async def scenario() -> Any:
        store = _store(backend, request, tmp_path, "limits")
        try:
            await _run(store, session_id, 1)
            checkpoint = await store.load_checkpoint(session_id)
            row, _ = await _stored(store, session_id)
            row_bytes, _ = measure_durable_json(row, "checkpoint")
            ledger_bytes, _ = measure_durable_json(checkpoint[LEDGER], "ledger")
            # Ordinary state that fits beside the reference, but not beside the ledger.
            padding = "x" * (DURABLE_DOCUMENT_LIMITS.max_bytes - row_bytes - ledger_bytes // 2)

            def grow(_session: Any, state: dict[str, Any] | None) -> dict[str, Any]:
                assert state is not None
                return {**state, "padding": padding}

            with pytest.raises(DurableValueError) as refused:
                await store.transform_checkpoint(session_id, grow)
            return refused.value.code, checkpoint, await store.load_checkpoint(session_id)
        finally:
            await store.close()

    code, before, after = asyncio.run(scenario())
    assert code == "json_value_too_large"
    assert after == before


def test_selected_text_fields_are_read_beside_a_referenced_ledger(tmp_path: Any) -> None:
    async def scenario() -> Any:
        store = SQLiteSessionStore(tmp_path / "fields.sqlite")
        try:
            await _run(store, "fields", 1)

            def note(_session: Any, state: dict[str, Any] | None) -> dict[str, Any]:
                assert state is not None
                return {**state, "note": "hello"}

            await store.transform_checkpoint("fields", note)
            checkpoint = await store.load_checkpoint("fields")
            text = await store.load_checkpoint_fields(["fields"], ["note"])
            both = await store.load_checkpoint_fields(["fields"], ["note", LEDGER])
            return checkpoint, text, both
        finally:
            await store.close()

    checkpoint, text, both = asyncio.run(scenario())
    assert text == {"fields": {"note": "hello"}}
    assert both == {"fields": {"note": "hello", LEDGER: checkpoint[LEDGER]}}


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("mode", ["create", "validate"])
def test_session_startup_rejects_a_schema_without_the_receipt_ledger(
    backend: str, mode: str, request: pytest.FixtureRequest, tmp_path: Any
) -> None:
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode, SchemaTooOld

    async def scenario() -> None:
        store = _store(backend, request, tmp_path, "old-schema")
        await store.load("absent")
        await store.close()
        if backend == "sqlite":
            path = tmp_path / "old-schema.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute("DROP TABLE cayu_invocation_receipt_ledgers")
                connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 144")
                connection.execute("PRAGMA user_version = 143")
            with pytest.raises(SchemaTooOld):
                SQLiteSessionStore(path, schema_mode=SchemaMode(mode))
        else:
            import psycopg

            dsn = request.getfixturevalue("postgres_dsn")
            async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as connection:
                await connection.execute("DROP TABLE cayu_invocation_receipt_ledgers")
                await connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 144")
            old_store = PostgresSessionStore(dsn, schema_mode=SchemaMode(mode))
            try:
                with pytest.raises(SchemaTooOld):
                    await old_store.load("absent")
            finally:
                await old_store.close()
                restored = PostgresSessionStore(dsn, schema_mode=SchemaMode.MIGRATE)
                try:
                    await restored.load("absent")
                finally:
                    await restored.close()

    asyncio.run(scenario())
