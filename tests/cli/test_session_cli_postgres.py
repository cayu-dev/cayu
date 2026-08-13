from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from cayu import PostgresSessionStore, SQLiteSessionStore
from cayu.cli import main
from cayu.core import Event, EventType, Message
from cayu.runtime import RunRequest, SessionIdentity, SessionStatus, SessionStore
from cayu.storage.migrations import SchemaMode


def _pricing_evidence(*, model: str) -> dict[str, object]:
    return {
        "provider_name": "fake",
        "model": model,
        "match": "exact",
        "provenance": {
            "source": "application",
            "url": "application://test-price-book",
            "as_of": "2026-07-25",
        },
        "effective_from": None,
        "effective_through": None,
        "tier_max_input_tokens": None,
    }


async def _drop_postgres_schema(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    async with await psycopg.AsyncConnection.connect(dsn) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_%'"
            )
            for (table_name,) in await cursor.fetchall():
                await cursor.execute(
                    sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier(table_name))
                )
        await connection.commit()


async def _seed(store: SessionStore) -> None:
    timestamp = datetime(2026, 7, 20, 18, 0, tzinfo=UTC)
    mixed_labels = {
        f"{prefix}{index:03d}": "value"
        for prefix in ("-", "A", "a", "Á", "ä", "Ω")
        for index in range(35)
    }
    await store.create(
        RunRequest(
            agent_name="parity-agent",
            session_id="sess_parity",
            environment_name="test",
            labels={"suite": "parity", **mixed_labels},
            messages=[Message.text("user", "inspect")],
        ),
        identity=SessionIdentity(provider_name="fake", model="requested-model"),
    )
    await store.append_transcript_messages(
        "sess_parity",
        [
            Message.text("user", "inspect"),
            Message.tool_call(
                tool_call_id="call-1",
                tool_name="search",
                arguments={"query": "cayu"},
            ),
            Message.tool_result(
                tool_call_id="call-1",
                tool_name="search",
                content="found",
                structured={"returned": 1},
            ),
        ],
    )
    await store.create(
        RunRequest(
            agent_name="parity-agent",
            session_id="sess_parity_newer",
            environment_name="test",
            labels={"suite": "parity"},
            messages=[Message.text("user", "newer")],
        ),
        identity=SessionIdentity(provider_name="fake", model="requested-model"),
    )
    for event in (
        Event(
            type=EventType.BUDGET_RESERVED,
            session_id="sess_parity",
            timestamp=timestamp - timedelta(seconds=1),
            payload={
                "accepted": True,
                "reservation_id": "reservation-1",
                "scope": "session",
                "key": "sess_parity",
                "window": "lifetime",
                "currency": "USD",
                "maximum": "1.00",
                "action": "interrupt",
                "requested": "0.25",
            },
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="sess_parity",
            timestamp=timestamp,
            payload={
                "transcript_cursor": 2,
                "usage_metrics": {
                    "provider_name": "fake",
                    "requested_model": "requested-model",
                    "model": "resolved-model",
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "total_tokens": 12,
                    "cache": {
                        "cached_input_tokens": 8,
                        "uncached_input_tokens": 2,
                    },
                },
            },
        ),
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="sess_parity",
            agent_name="parity-agent",
            environment_name="test",
            tool_name="search",
            timestamp=timestamp + timedelta(seconds=1),
            payload={
                "tool_call_id": "call-1",
                "tool_round_id": "round-1",
                "arguments": {"query": "cayu"},
            },
        ),
        Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id="sess_parity",
            agent_name="parity-agent",
            environment_name="test",
            tool_name="search",
            timestamp=timestamp + timedelta(seconds=2),
            payload={
                "tool_call_id": "call-1",
                "tool_round_id": "round-1",
                "result": {
                    "content": "found",
                    "structured": {"returned": 1},
                    "artifacts": [],
                    "is_error": False,
                },
            },
        ),
        Event(
            type=EventType.BUDGET_RECONCILED,
            session_id="sess_parity",
            timestamp=timestamp + timedelta(seconds=3),
            payload={
                "reservation_id": "reservation-1",
                "reserved_amount": "0.25",
                "actual_amount": "0.01",
                "pricing": _pricing_evidence(model="resolved-model"),
            },
        ),
        Event(
            type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
            session_id="sess_parity",
            tool_name="deploy",
            timestamp=timestamp + timedelta(seconds=4),
            payload={
                "approval": {
                    "approval_id": "approval-2",
                    "tool_call_id": "call-2",
                    "tool_name": "deploy",
                    "arguments": {"environment": "test"},
                    "agent_name": "parity-agent",
                    "tool_calls": [
                        {
                            "tool_call_id": "call-2",
                            "tool_name": "deploy",
                            "arguments": {"environment": "test"},
                            "policy_decision": "require_approval",
                            "reason": None,
                            "metadata": {},
                            "active_taint_labels": [],
                        }
                    ],
                }
            },
        ),
        Event(
            type="custom.oversized",
            session_id="sess_parity",
            timestamp=timestamp + timedelta(seconds=5),
            payload={"blob": "x" * 2048},
        ),
    ):
        await store.append_event("sess_parity", event)
    await store.checkpoint(
        "sess_parity",
        {
            "pending_tool_approval": {
                "approval_id": "approval-2",
                "tool_call_id": "call-2",
                "tool_name": "deploy",
                "arguments": {"environment": "test"},
                "agent_name": "parity-agent",
                "publish_arguments": True,
                "tool_calls": [
                    {
                        "tool_call_id": "call-2",
                        "tool_name": "deploy",
                        "arguments": {"environment": "test"},
                        "policy_decision": "require_approval",
                        "reason": None,
                        "metadata": {},
                        "active_taint_labels": [],
                    }
                ],
            }
        },
    )
    await store.update_status("sess_parity", SessionStatus.INTERRUPTED)


def _without_backend_timestamps(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(payload))
    if command == "list":
        for session in copied["sessions"]:
            for key in ("created_at", "updated_at", "last_activity_at"):
                session.pop(key)
    elif command == "show":
        for key in ("created_at", "updated_at", "last_activity_at"):
            copied["session"].pop(key)
    return copied


def test_session_commands_have_sqlite_postgres_semantic_parity(
    tmp_path: Path,
    postgres_dsn: str,
    capsys,
) -> None:
    sqlite_path = tmp_path / "cayu.db"

    async def prepare() -> None:
        await _drop_postgres_schema(postgres_dsn)
        sqlite_store = SQLiteSessionStore(sqlite_path)
        postgres_store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _seed(sqlite_store)
            await _seed(postgres_store)
        finally:
            await sqlite_store.close()
            await postgres_store.close()

    asyncio.run(prepare())
    commands = (
        ("list", ["list", "--label", "suite=parity"]),
        ("show", ["show", "sess_parity"]),
        ("usage", ["usage", "sess_parity"]),
        ("tools", ["tools", "sess_parity"]),
        ("events", ["events", "sess_parity"]),
        ("transcript", ["transcript", "sess_parity", "--sizes"]),
    )
    try:
        for name, command in commands:
            assert (
                main(
                    [
                        "session",
                        *command,
                        "--sqlite",
                        str(sqlite_path),
                        "--json",
                    ]
                )
                == 0
            )
            sqlite_payload = json.loads(capsys.readouterr().out)
            assert (
                main(
                    [
                        "session",
                        *command,
                        "--postgres",
                        postgres_dsn,
                        "--json",
                    ]
                )
                == 0
            )
            postgres_payload = json.loads(capsys.readouterr().out)
            assert _without_backend_timestamps(name, sqlite_payload) == _without_backend_timestamps(
                name, postgres_payload
            )
            if name == "show":
                assert sqlite_payload["session"]["label_count"] == 211
                assert len(sqlite_payload["session"]["labels"]) == 200
                assert sqlite_payload["session"]["labels_truncated"] is True
    finally:
        asyncio.run(_drop_postgres_schema(postgres_dsn))


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_session_tools_conflicting_approval_and_execution_is_unavailable_across_backends(
    backend: str,
    tmp_path: Path,
    request,
    monkeypatch,
    capsys,
) -> None:
    from cayu.cli import session as session_cli

    sqlite_path = tmp_path / "conflicting-tool-evidence.db"
    postgres_dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    session_id = "sess_conflicting_tool_evidence"
    observed_at = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    normal_identity = {
        "model_step_id": f"mstep_{'1' * 32}",
        "model_attempt_id": f"matt_{'2' * 32}",
        "tool_round_id": f"tround_{'3' * 32}",
    }
    conflicting_identity = {
        "model_step_id": f"mstep_{'4' * 32}",
        "model_attempt_id": f"matt_{'5' * 32}",
        "tool_round_id": f"tround_{'6' * 32}",
    }
    shared_call_id = "provider-reused-call-id"
    approval_id = "approval-conflict"

    async def seed(store: SessionStore) -> None:
        await store.create(
            RunRequest(
                agent_name="inspector",
                session_id=session_id,
                messages=[Message.text("user", "inspect conflicting tool evidence")],
            ),
            identity=SessionIdentity(provider_name="fake", model="model"),
        )
        await store.append_events(
            session_id,
            [
                Event(type="custom.before", session_id=session_id),
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="read_file",
                    timestamp=observed_at,
                    payload={
                        **normal_identity,
                        "tool_call_id": shared_call_id,
                        "arguments": {"path": "README.md"},
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=session_id,
                    tool_name="deploy",
                    timestamp=observed_at + timedelta(seconds=1),
                    payload={
                        **conflicting_identity,
                        "approval_id": approval_id,
                        "tool_call_id": shared_call_id,
                        "approval": {
                            **conflicting_identity,
                            "approval_id": approval_id,
                            "tool_call_id": shared_call_id,
                            "tool_name": "deploy",
                            "arguments": {"environment": "production"},
                            "tool_calls": [
                                {
                                    "tool_call_id": shared_call_id,
                                    "tool_name": "deploy",
                                    "arguments": {"environment": "production"},
                                    "policy_evidence": "authoritative",
                                    "policy_decision": "require_approval",
                                }
                            ],
                        },
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    tool_name="deploy",
                    timestamp=observed_at + timedelta(seconds=2),
                    payload={
                        **conflicting_identity,
                        "approval_id": approval_id,
                        "tool_call_id": shared_call_id,
                        "arguments": {"environment": "production"},
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_APPROVAL_DENIED,
                    session_id=session_id,
                    tool_name="deploy",
                    timestamp=observed_at + timedelta(seconds=3),
                    payload={
                        **conflicting_identity,
                        "approval_id": approval_id,
                        "tool_call_id": shared_call_id,
                        "approval_required": True,
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id=session_id,
                    tool_name="read_file",
                    timestamp=observed_at + timedelta(seconds=4),
                    payload={
                        **normal_identity,
                        "tool_call_id": shared_call_id,
                        "result": {
                            "content": "ordinary result",
                            "structured": {"returned": 1},
                            "artifacts": [],
                            "is_error": False,
                        },
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id=session_id,
                    tool_name="deploy",
                    timestamp=observed_at + timedelta(seconds=5),
                    payload={
                        **conflicting_identity,
                        "approval_id": approval_id,
                        "tool_call_id": shared_call_id,
                        "result": {
                            "content": "must not be exposed",
                            "structured": {"returned": 9, "truncated": True},
                            "artifacts": [{"size_bytes": 17}],
                            "is_error": False,
                        },
                    },
                ),
                Event(type="custom.after", session_id=session_id),
            ],
        )

    async def prepare() -> None:
        if postgres_dsn is None:
            store: SessionStore = SQLiteSessionStore(sqlite_path)
        else:
            await _drop_postgres_schema(postgres_dsn)
            store = PostgresSessionStore(
                postgres_dsn,
                min_size=1,
                max_size=4,
                schema_mode=SchemaMode.CREATE,
            )
        try:
            await seed(store)
        finally:
            await store.close()

    asyncio.run(prepare())
    monkeypatch.setattr(session_cli, "_EVENT_QUERY_PAGE_SIZE", 2)
    target = (
        ["--sqlite", str(sqlite_path)] if postgres_dsn is None else ["--postgres", postgres_dsn]
    )

    try:
        base_command = [
            "session",
            "tools",
            session_id,
            *target,
            "--after-sequence",
            "1",
            "--before-sequence",
            "8",
            "--limit",
            "1",
            "--json",
        ]
        assert main(base_command) == 0
        first_page = json.loads(capsys.readouterr().out)
        assert first_page["event_window"] == {
            "after_sequence": 1,
            "before_sequence": 8,
        }
        assert first_page["total_calls"] == 2
        assert first_page["has_more"] is True
        assert first_page["next_offset"] == 1
        assert first_page["calls"][0]["model_step_id"] == normal_identity["model_step_id"]
        assert first_page["calls"][0]["status"] == "success"

        assert main([*base_command, "--offset", "1"]) == 0
        output = capsys.readouterr().out
        second_page = json.loads(output)
        assert "must not be exposed" not in output
        assert second_page["total_calls"] == 2
        assert second_page["has_more"] is False
        assert second_page["next_offset"] is None
        assert len(second_page["calls"]) == 1
        conflicting = second_page["calls"][0]
        assert conflicting["model_step_id"] == conflicting_identity["model_step_id"]
        assert conflicting["tool_call_id"] == shared_call_id
        assert conflicting["status"] == "unavailable"
        assert conflicting["approval_state"] == "unavailable"
        for field in (
            "completed_at",
            "duration_ms",
            "rendered_content_bytes",
            "structured_result_bytes",
            "artifact_bytes",
            "returned",
            "truncated",
        ):
            assert conflicting[field] is None
    finally:
        if postgres_dsn is not None:
            asyncio.run(_drop_postgres_schema(postgres_dsn))


def test_postgres_read_only_store_uses_transaction_scoped_protection(
    postgres_dsn: str,
) -> None:
    from psycopg.errors import ReadOnlySqlTransaction

    async def exercise() -> None:
        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()

        reader = PostgresSessionStore(
            postgres_dsn,
            schema_mode=SchemaMode.VALIDATE,
            read_only=True,
        )
        try:
            await reader.ensure_schema()
            async with reader._pool.connection() as connection:
                result = await connection.execute("SHOW default_transaction_read_only")
                assert await result.fetchone() == ("off",)

            with pytest.raises(ReadOnlySqlTransaction):
                await reader.create(
                    RunRequest(
                        agent_name="reader",
                        session_id="must-not-write",
                        messages=[Message.text("user", "inspect")],
                    ),
                    identity=SessionIdentity(provider_name="fake", model="model"),
                )
        finally:
            await reader.close()
            await _drop_postgres_schema(postgres_dsn)

    asyncio.run(exercise())
