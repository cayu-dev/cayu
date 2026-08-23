from __future__ import annotations

import asyncio
import base64
import contextvars
import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
from tests.core._event_projection_support import private_events_for_public_events

from cayu import CHECKPOINT_SCHEMA_VERSION_KEY, SQLiteSessionStore, SQLiteTaskStore
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runtime import (
    CayuApp,
    EnqueueSessionMessageRequest,
    EventOrder,
    EventQuery,
    ForkSessionRequest,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    ResumeRequest,
    RunRequest,
    Session,
    SessionAggregateFilter,
    SessionIdentity,
    SessionInspectionSummary,
    SessionMessageDeliveryMode,
    SessionQuery,
    SessionStatus,
    TranscriptQuery,
    UsageRollupQuery,
)
from cayu.runtime.aggregates import AggregateUsageMetrics
from cayu.runtime.checkpoints import CURRENT_CHECKPOINT_SCHEMA_VERSION
from cayu.runtime.sessions import (
    TRANSCRIPT_SEARCH_TOKENIZER_VERSION,
    BudgetReservationIdentityConflict,
    ModelCompletionStageRequest,
    PendingActionQuery,
    fork_session_invocation,
)
from cayu.storage import _session_store_sql as session_store_sql
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema_migrations
from cayu.storage import sqlite as sqlite_storage


def _tool_round_identity_payload() -> dict[str, str]:
    return {
        "model_step_id": f"mstep_{'1' * 32}",
        "model_attempt_id": f"matt_{'2' * 32}",
        "tool_round_id": f"tround_{'3' * 32}",
    }


def test_read_only_session_store_does_not_create_missing_database(tmp_path) -> None:
    missing = tmp_path / "missing" / "data" / "cayu.db"

    with pytest.raises(sqlite3.OperationalError):
        SQLiteSessionStore(
            missing,
            schema_mode=schema_migrations.SchemaMode.VALIDATE,
            read_only=True,
        )
    assert not missing.exists()
    assert not missing.parent.exists()


def test_sqlite_workflow_replay_query_uses_step_and_attempt_indexes(tmp_path) -> None:
    db_path = tmp_path / "workflow-replay.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))
    query = EventQuery(
        session_id="workflow-run",
        workflow_name="workflow",
        workflow_step_id="step",
        workflow_attempt_fenced=True,
        event_type=EventType.WORKFLOW_STEP_COMPLETED,
        order_by=EventOrder.SEQUENCE_DESC,
        limit=1,
    )
    dialect = session_store_sql.SessionStoreSqlDialect(
        placeholder="?",
        contains_style="sqlite_nocase_like",
        datetime_param=lambda value: value,
    )
    plan = session_store_sql.build_event_query_sql(query, dialect=dialect)
    exact_query = query.model_copy(
        update={
            "workflow_attempt_fenced": False,
            "workflow_attempt_id": "attempt",
        }
    )
    exact_plan = session_store_sql.build_event_query_sql(exact_query, dialect=dialect)

    connection = sqlite3.connect(db_path)
    try:
        details = [
            row[3]
            for row in connection.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT cayu_events.sequence FROM cayu_events "
                "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                f"{plan.where_sql} "
                f"ORDER BY cayu_events.sequence {plan.order_direction} LIMIT ?",
                (*plan.params, query.limit),
            )
        ]
        exact_details = [
            row[3]
            for row in connection.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT cayu_events.sequence FROM cayu_events "
                "JOIN cayu_sessions ON cayu_sessions.id = cayu_events.session_id "
                f"{exact_plan.where_sql} "
                f"ORDER BY cayu_events.sequence {exact_plan.order_direction} LIMIT ?",
                (*exact_plan.params, exact_query.limit),
            )
        ]
    finally:
        connection.close()

    assert any("idx_cayu_events_workflow_step_replay" in detail for detail in details)
    assert any("idx_cayu_events_workflow_attempt_marker" in detail for detail in details)
    assert any("idx_cayu_events_workflow_step_attempt" in detail for detail in exact_details)


def test_sqlite_migrate_repairs_missing_workflow_replay_index(tmp_path) -> None:
    db_path = tmp_path / "workflow-replay-repair.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP INDEX idx_cayu_events_workflow_step_replay")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="Required Cayu SQLite index is missing"):
        SQLiteSessionStore(db_path)

    repaired = SQLiteSessionStore(
        db_path,
        schema_mode=schema_migrations.SchemaMode.MIGRATE,
    )
    asyncio.run(_close(repaired))

    validated = SQLiteSessionStore(db_path)
    asyncio.run(_close(validated))


def test_sqlite_interaction_transcript_query_uses_persisted_absolute_order(
    tmp_path,
) -> None:
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_indexed_transcript",
                messages=[],
            ),
            identity=_identity(),
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("user", "first"), Message.text("assistant", "one")],
            interaction_id="interaction-one",
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("user", "other")],
            interaction_id="interaction-two",
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("assistant", "two")],
            interaction_id="interaction-one",
        )
        page = await store.query_transcript(
            TranscriptQuery(
                session_id=session.id,
                interaction_id="interaction-one",
                limit=10,
            )
        )
        assert [record.index for record in page.records] == [0, 1, 3]
        await _close(store)

    asyncio.run(run())

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT session_order
            FROM cayu_transcript_messages
            WHERE session_id = ?
            ORDER BY session_order
            """,
            ("sess_indexed_transcript",),
        ).fetchall()
        plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT session_order, message_json
            FROM cayu_transcript_messages
            WHERE session_id = ?
              AND interaction_id = ?
            ORDER BY session_order
            LIMIT 100
            """,
            ("sess_indexed_transcript", "interaction-one"),
        ).fetchall()
    finally:
        connection.close()

    assert [row[0] for row in rows] == [1, 2, 3, 4]
    rendered_plan = " ".join(str(column) for row in plan for column in row)
    assert "idx_cayu_transcript_interaction_order" in rendered_plan
    assert "USE TEMP B-TREE" not in rendered_plan


def test_sqlite_revision_twenty_six_rejects_populated_session_database(
    tmp_path,
) -> None:
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def create_transcript_data() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_transcript_order_migration",
                messages=[],
            ),
            identity=_identity(),
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("user", "first")],
            interaction_id="interaction-one",
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("user", "other")],
            interaction_id="interaction-two",
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("assistant", "second")],
            interaction_id="interaction-one",
        )
        await _close(store)

    asyncio.run(create_transcript_data())

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            DROP TRIGGER cayu_reject_explicit_transcript_order;
            DROP TRIGGER cayu_assign_transcript_order;
            DROP INDEX idx_cayu_transcript_interaction_order;
            DROP INDEX idx_cayu_transcript_session_order;
            ALTER TABLE cayu_transcript_messages DROP COLUMN session_order;
            ALTER TABLE cayu_sessions DROP COLUMN transcript_seq;
            DELETE FROM cayu_schema_migrations WHERE revision >= 26;
            PRAGMA user_version = 25;
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match="clean prerelease break",
    ):
        SQLiteSessionStore(
            db_path,
            schema_mode=schema_migrations.SchemaMode.MIGRATE,
        )

    connection = sqlite3.connect(db_path)
    try:
        transcript_count = connection.execute(
            "SELECT COUNT(*) FROM cayu_transcript_messages WHERE session_id = ?",
            ("sess_transcript_order_migration",),
        ).fetchone()[0]
        latest_revision = connection.execute(
            "SELECT MAX(revision) FROM cayu_schema_migrations"
        ).fetchone()[0]
        session_columns = {row[1] for row in connection.execute("PRAGMA table_info(cayu_sessions)")}
        transcript_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cayu_transcript_messages)")
        }
    finally:
        connection.close()

    assert transcript_count == 3
    assert latest_revision == 25
    assert "transcript_seq" not in session_columns
    assert "session_order" not in transcript_columns


def test_read_only_session_store_loads_existing_state_and_rejects_writes(tmp_path) -> None:
    path = tmp_path / "data" / "cayu.db"

    async def exercise() -> None:
        writer = SQLiteSessionStore(path)
        try:
            created = await writer.create(
                RunRequest(agent_name="reader", messages=[Message.text("user", "hello")]),
                identity=SessionIdentity(provider_name="test", model="model"),
            )
        finally:
            await writer.close()

        reader = SQLiteSessionStore(
            path,
            schema_mode=schema_migrations.SchemaMode.VALIDATE,
            read_only=True,
        )
        try:
            assert await reader.load(created.id) == created
            with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
                await reader.create(
                    RunRequest(
                        agent_name="reader",
                        messages=[Message.text("user", "do not write")],
                    ),
                    identity=SessionIdentity(provider_name="test", model="model"),
                )
        finally:
            await reader.close()

    asyncio.run(exercise())


def test_read_only_session_store_reads_while_wal_writer_remains_open(tmp_path) -> None:
    path = tmp_path / "data" / "cayu.db"

    async def exercise() -> None:
        writer = SQLiteSessionStore(path)
        reader = None
        try:
            created = await writer.create(
                RunRequest(
                    agent_name="writer",
                    session_id="sess_live_reader",
                    messages=[Message.text("user", "before")],
                ),
                identity=SessionIdentity(provider_name="test", model="model"),
            )
            await writer.append_transcript_messages(
                created.id,
                [Message.text("assistant", "visible through wal")],
            )

            reader = SQLiteSessionStore(
                path,
                schema_mode=schema_migrations.SchemaMode.VALIDATE,
                read_only=True,
            )

            loaded = await reader.load(created.id)
            assert loaded is not None
            assert loaded.id == created.id
            transcript = await reader.load_transcript(created.id)
            assert len(transcript) == 1
            assert transcript[0].content[0].text == "visible through wal"
        finally:
            if reader is not None:
                await reader.close()
            await writer.close()

    asyncio.run(exercise())


def test_session_inspection_summary_conforms_to_native_event_and_usage_aggregates(
    tmp_path,
) -> None:
    path = tmp_path / "data" / "cayu.db"

    class BoundedInspectionStore(SQLiteSessionStore):
        async def load(self, session_id: str) -> Session | None:
            raise AssertionError("inspect_summary must not materialize session metadata")

    async def exercise() -> None:
        store = BoundedInspectionStore(path)
        timestamp = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
        try:
            await store.create(
                RunRequest(
                    agent_name="inspector",
                    session_id="sess_inspection_conformance",
                    labels={
                        "inspection": "conformance",
                        **{f"label_{index:03d}": "value" for index in range(205)},
                    },
                    metadata={"customer_payload": "x" * 1_000_000},
                    messages=[Message.text("user", "inspect")],
                ),
                identity=SessionIdentity(provider_name="fake", model="model"),
            )
            for event in (
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_inspection_conformance",
                    timestamp=timestamp,
                    payload={
                        "usage_metrics": {
                            "input_tokens": MAX_DURABLE_JSON_INTEGER,
                            "output_tokens": 0,
                            "total_tokens": MAX_DURABLE_JSON_INTEGER,
                        }
                    },
                ),
                Event(
                    type=EventType.MODEL_COMPLETED,
                    session_id="sess_inspection_conformance",
                    timestamp=timestamp + timedelta(milliseconds=500),
                    payload={
                        "usage_metrics": {
                            "input_tokens": 1,
                            "output_tokens": 2,
                            "total_tokens": 3,
                        }
                    },
                ),
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="sess_inspection_conformance",
                    timestamp=timestamp + timedelta(seconds=1),
                    tool_name="read_file",
                    payload={"tool_call_id": "call-1", "arguments": {}},
                ),
            ):
                await store.append_event("sess_inspection_conformance", event)

            inspection = await store.inspect_summary("sess_inspection_conformance")
            event_summary = await store.summarize_events("sess_inspection_conformance")
            usage = await store.aggregate_usage(
                UsageRollupQuery(
                    start_at=timestamp - timedelta(seconds=1),
                    end_at=timestamp + timedelta(days=1),
                    sessions=SessionAggregateFilter(labels={"inspection": "conformance"}),
                )
            )

            assert inspection.events.record_count == event_summary.total_events
            assert inspection.session.label_count == 206
            assert len(inspection.session.labels) == 200
            assert inspection.session.labels["inspection"] == "conformance"
            assert inspection.session.labels_truncated is True
            assert inspection.model_calls == usage.totals.model_steps
            assert inspection.tool_calls == usage.totals.tool_calls
            assert inspection.model_calls_with_usage == usage.totals.model_steps_with_usage
            assert isinstance(inspection.usage.usage, AggregateUsageMetrics)
            assert inspection.usage.usage == usage.totals.usage
            assert inspection.usage.usage.input_tokens == MAX_DURABLE_JSON_INTEGER + 1
            assert (
                SessionInspectionSummary.model_validate(inspection.model_dump(mode="python"))
                == inspection
            )
            assert (
                SessionInspectionSummary.model_validate_json(inspection.model_dump_json())
                == inspection
            )
        finally:
            await store.close()

    asyncio.run(exercise())


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(
        self,
        events: list[ModelStreamEvent] | list[list[ModelStreamEvent]],
    ) -> None:
        if events and isinstance(events[0], list):
            self.event_batches = events  # type: ignore[assignment]
        else:
            self.event_batches = [events]  # type: ignore[list-item]
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest):
        self.requests.append(request)
        batch_index = len(self.requests) - 1
        if batch_index >= len(self.event_batches):
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


async def _close(store: SQLiteSessionStore) -> None:
    await store.close()


def test_sqlite_pending_action_query_uses_persisted_projection_not_original_payload(
    tmp_path,
) -> None:
    db_path = tmp_path / "pending_projection.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run() -> None:
        session_id = "persisted_pending_projection"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_event(
            session_id,
            Event(
                id="persisted_pending_projection_event",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id=session_id,
                agent_name="assistant",
                tool_name="deploy",
                payload={
                    **_tool_round_identity_payload(),
                    "approval_id": "persisted_pending_projection_approval",
                    "tool_call_id": "persisted_pending_projection_call",
                    "approval": {
                        "approval_id": "persisted_pending_projection_approval",
                        **_tool_round_identity_payload(),
                        "tool_call_id": "persisted_pending_projection_call",
                        "tool_name": "deploy",
                        "arguments": {},
                        "agent_name": "assistant",
                        "tool_calls": [
                            {
                                "tool_call_id": "persisted_pending_projection_call",
                                "tool_name": "deploy",
                                "arguments": {},
                                "policy_decision": None,
                                "reason": None,
                                "metadata": {},
                                "active_taint_labels": [],
                            }
                        ],
                    },
                },
            ),
        )
        await store.checkpoint(
            session_id,
            {
                "pending_tool_approval": {
                    "approval_id": "persisted_pending_projection_approval",
                    **_tool_round_identity_payload(),
                    "tool_call_id": "persisted_pending_projection_call",
                    "tool_name": "deploy",
                    "arguments": {},
                    "agent_name": "assistant",
                    "publish_arguments": True,
                    "tool_calls": [
                        {
                            "tool_call_id": "persisted_pending_projection_call",
                            "tool_name": "deploy",
                            "arguments": {},
                            "policy_decision": None,
                            "reason": None,
                            "metadata": {},
                            "active_taint_labels": [],
                        }
                    ],
                }
            },
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)

        connection = sqlite3.connect(db_path)
        try:
            connection.execute(
                "UPDATE cayu_events SET payload_json = ? WHERE session_id = ?",
                ("not-json-and-intentionally-unbounded-from-the-query", session_id),
            )
            connection.commit()
        finally:
            connection.close()

        result = await store.query_pending_actions(PendingActionQuery(session_id=session_id))
        assert len(result.actions) == 1
        assert result.actions[0].approval_id == "persisted_pending_projection_approval"
        await store.close()

    asyncio.run(run())


async def _collect_app_events(events) -> list[Event]:
    return [event async for event in events]


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _public_authority_codec(
    *, active_key_id: str = "primary", retained: bool = False, primary_key_byte: int = 1
) -> PublicAuthorityAliasCodec:
    keys = {
        "primary": SecretStr(
            base64.urlsafe_b64encode(bytes([primary_key_byte]) * 32).decode("ascii").rstrip("=")
        )
    }
    if retained:
        keys["rotated"] = SecretStr(
            base64.urlsafe_b64encode(bytes([2]) * 32).decode("ascii").rstrip("=")
        )
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id=active_key_id, keys=keys)
    )


def test_sqlite_public_authority_aliases_are_authenticated_scoped_and_durable(
    tmp_path,
) -> None:
    db_path = tmp_path / "public-authority.sqlite"
    codec = _public_authority_codec(retained=True)
    store = SQLiteSessionStore(db_path, public_authority_alias_codec=codec)
    session_alias = codec.encode("private-session", field_name="session_id")
    interaction_alias = codec.encode(
        "private-interaction",
        field_name="interaction_id",
        session_id="private-session",
    )

    async def register() -> None:
        await store.register_public_authority_alias(
            session_alias,
            field_name="session_id",
            private_value="private-session",
        )
        # Registration is idempotent and interaction identities are scoped to
        # their private session rather than a process-local public alias.
        await store.register_public_authority_alias(
            session_alias,
            field_name="session_id",
            private_value="private-session",
        )
        await store.register_public_authority_alias(
            interaction_alias,
            field_name="interaction_id",
            private_value="private-interaction",
            scope_session_id="private-session",
        )
        assert (
            await store.resolve_public_authority_alias(
                interaction_alias,
                field_name="interaction_id",
                scope_session_id="different-session",
            )
            is None
        )
        await store.close()

    asyncio.run(register())

    reopened = SQLiteSessionStore(db_path, public_authority_alias_codec=codec)

    async def resolve_after_restart() -> None:
        assert (
            await reopened.resolve_public_authority_alias(
                session_alias,
                field_name="session_id",
            )
            == "private-session"
        )
        assert (
            await reopened.resolve_public_authority_alias(
                interaction_alias,
                field_name="interaction_id",
                scope_session_id="private-session",
            )
            == "private-interaction"
        )
        await reopened.close()

    asyncio.run(resolve_after_restart())


def test_sqlite_public_authority_alias_registration_fails_closed(tmp_path) -> None:
    db_path = tmp_path / "public-authority-invalid.sqlite"
    codec = _public_authority_codec()
    alias = codec.encode("private-session", field_name="session_id")
    unconfigured = SQLiteSessionStore(db_path)

    async def assert_unconfigured() -> None:
        with pytest.raises(ValueError, match="store-configured provenance"):
            await unconfigured.register_public_authority_alias(
                alias,
                field_name="session_id",
                private_value="private-session",
            )
        await unconfigured.close()

    asyncio.run(assert_unconfigured())

    store = SQLiteSessionStore(db_path, public_authority_alias_codec=codec)

    async def assert_invalid() -> None:
        with pytest.raises(ValueError, match="store-configured provenance"):
            await store.register_public_authority_alias(
                alias,
                field_name="session_id",
                private_value="different-private-session",
            )
        with pytest.raises(ValueError, match="field-mismatched"):
            await store.resolve_public_authority_alias(
                alias,
                field_name="interaction_id",
                scope_session_id="private-session",
            )
        with pytest.raises(ValueError, match="must not have a session scope"):
            await store.resolve_public_authority_alias(
                alias,
                field_name="session_id",
                scope_session_id="private-session",
            )
        with pytest.raises(ValueError, match="require a private session scope"):
            await store.resolve_public_authority_alias(
                codec.encode("interaction", field_name="interaction_id", session_id="session"),
                field_name="interaction_id",
            )
        await store.close()

    asyncio.run(assert_invalid())


def test_sqlite_public_authority_aliases_allow_key_rotation_and_reject_conflicts(
    tmp_path,
) -> None:
    db_path = tmp_path / "public-authority-rotation.sqlite"
    first_codec = _public_authority_codec()
    rotated_codec = _public_authority_codec(active_key_id="rotated", retained=True)
    first_alias = first_codec.encode("private-session", field_name="session_id")
    rotated_alias = rotated_codec.encode("private-session", field_name="session_id")
    store = SQLiteSessionStore(db_path, public_authority_alias_codec=rotated_codec)

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            INSERT INTO cayu_public_authority_aliases (
                field_name, scope_session_id, public_alias, private_value
            ) VALUES (?, ?, ?, ?)
            """,
            ("session_id", "", first_alias, "conflicting-private-session"),
        )
        connection.commit()
    finally:
        connection.close()

    async def run() -> None:
        with pytest.raises(ValueError, match="conflicts with existing private authority"):
            await store.register_public_authority_alias(
                first_alias,
                field_name="session_id",
                private_value="private-session",
            )
        await store.register_public_authority_alias(
            rotated_alias,
            field_name="session_id",
            private_value="private-session",
        )
        assert (
            await store.resolve_public_authority_alias(
                rotated_alias,
                field_name="session_id",
            )
            == "private-session"
        )
        await store.close()

    asyncio.run(run())

    with pytest.raises(ValueError, match="codec is required"):
        SQLiteSessionStore(db_path)
    with pytest.raises(ValueError, match="key material conflicts"):
        SQLiteSessionStore(
            db_path,
            public_authority_alias_codec=_public_authority_codec(primary_key_byte=9),
        )


def test_sqlite_public_authority_aliases_reject_retired_keys(tmp_path) -> None:
    db_path = tmp_path / "public-authority-retirement.sqlite"
    first_codec = _public_authority_codec()
    retained_codec = _public_authority_codec(active_key_id="rotated", retained=True)
    rotated_key = SecretStr(base64.urlsafe_b64encode(bytes([2]) * 32).decode("ascii").rstrip("="))
    retired_codec = retained_codec.rotated(
        active_key_id="rotated",
        key=rotated_key,
        retire_key_ids=("primary",),
    )
    private_session_id = "private-session"
    old_alias = first_codec.encode(private_session_id, field_name="session_id")
    active_alias = retained_codec.encode(private_session_id, field_name="session_id")

    first_store = SQLiteSessionStore(
        db_path,
        public_authority_alias_codec=first_codec,
    )

    async def seed() -> None:
        await first_store.register_public_authority_alias(
            old_alias,
            field_name="session_id",
            private_value=private_session_id,
        )
        await first_store.close()

    asyncio.run(seed())

    retained_store = SQLiteSessionStore(
        db_path,
        public_authority_alias_codec=retained_codec,
    )

    async def retain() -> None:
        await retained_store.register_public_authority_alias(
            active_alias,
            field_name="session_id",
            private_value=private_session_id,
        )
        assert (
            await retained_store.resolve_public_authority_alias(
                old_alias,
                field_name="session_id",
            )
            == private_session_id
        )
        await retained_store.close()

    asyncio.run(retain())

    retired_store = SQLiteSessionStore(
        db_path,
        public_authority_alias_codec=retired_codec,
    )

    async def reject_retired() -> None:
        assert (
            await retired_store.resolve_public_authority_alias(
                old_alias,
                field_name="session_id",
            )
            is None
        )
        assert (
            await retired_store.resolve_public_authority_alias(
                active_alias,
                field_name="session_id",
            )
            == private_session_id
        )
        await retired_store.close()

    asyncio.run(reject_retired())


def test_sqlite_public_authority_rotation_fences_stale_workers(tmp_path) -> None:
    db_path = tmp_path / "public-authority-stale-worker.sqlite"
    first_codec = _public_authority_codec()
    rotated_codec = _public_authority_codec(active_key_id="rotated", retained=True)
    stale_store = SQLiteSessionStore(db_path, public_authority_alias_codec=first_codec)

    async def seed() -> None:
        await stale_store.create(
            RunRequest(agent_name="assistant", session_id="before-rotation", messages=[]),
            identity=_identity(),
        )
        await stale_store.append_transcript_messages(
            "before-rotation",
            [Message.text("user", "before rotation")],
        )
        await stale_store.append_event(
            "before-rotation",
            Event(
                id="before-rotation-completed",
                type=EventType.SESSION_COMPLETED,
                session_id="before-rotation",
            ),
        )
        await stale_store.update_status("before-rotation", SessionStatus.COMPLETED)

    asyncio.run(seed())
    staged_codec = _public_authority_codec(active_key_id="primary", retained=True)
    staged_store = SQLiteSessionStore(db_path, public_authority_alias_codec=staged_codec)

    async def stage() -> None:
        with pytest.raises(RuntimeError, match="configuration is stale"):
            await stale_store.create(
                RunRequest(agent_name="assistant", session_id="stale-staged-write", messages=[]),
                identity=_identity(),
            )
        await staged_store.create(
            RunRequest(agent_name="assistant", session_id="during-staging", messages=[]),
            identity=_identity(),
        )

    asyncio.run(stage())
    rotated_store = SQLiteSessionStore(db_path, public_authority_alias_codec=rotated_codec)

    async def assert_fenced() -> None:
        with pytest.raises(RuntimeError, match="configuration is stale"):
            await stale_store.create(
                RunRequest(agent_name="assistant", session_id="stale-write", messages=[]),
                identity=_identity(),
            )
        with pytest.raises(RuntimeError, match="configuration is stale"):
            await stale_store.load("before-rotation")
        with pytest.raises(RuntimeError, match="configuration is stale"):
            await stale_store.list_sessions()
        with pytest.raises(RuntimeError, match="configuration is stale"):
            await stale_store.list_sessions_with_pending_interruption_cascade()
        with pytest.raises(RuntimeError, match="configuration is stale"):
            await stale_store.query_transcript(
                TranscriptQuery(session_id="before-rotation", limit=10)
            )
        with pytest.raises(RuntimeError, match="configuration is stale"):
            await stale_store.summarize_events("before-rotation")
        with pytest.raises(RuntimeError, match="configuration is stale"):
            await stale_store.summarize_outcome("before-rotation")

        rotated_alias = rotated_codec.encode("before-rotation", field_name="session_id")
        assert (
            await rotated_store.resolve_public_authority_alias(
                rotated_alias,
                field_name="session_id",
            )
            == "before-rotation"
        )
        staged_write_rotated_alias = rotated_codec.encode(
            "during-staging",
            field_name="session_id",
        )
        assert (
            await rotated_store.resolve_public_authority_alias(
                staged_write_rotated_alias,
                field_name="session_id",
            )
            == "during-staging"
        )
        await stale_store.close()
        await staged_store.close()
        await rotated_store.close()

    asyncio.run(assert_fenced())

    with pytest.raises(ValueError, match="retired.*cannot become active"):
        SQLiteSessionStore(db_path, public_authority_alias_codec=first_codec)


def test_sqlite_rotation_fence_is_atomic_with_identity_write_transaction(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "public-authority-atomic-fence.sqlite"
    first_codec = _public_authority_codec()
    rotated_codec = _public_authority_codec(active_key_id="rotated", retained=True)
    stale_store = SQLiteSessionStore(db_path, public_authority_alias_codec=first_codec)
    precheck_completed = threading.Event()
    allow_write = threading.Event()

    async def scenario() -> None:
        await stale_store.create(
            RunRequest(agent_name="assistant", session_id="session", messages=[]),
            identity=_identity(),
        )
        original_check = stale_store._require_current_public_authority_configuration

        def pause_after_precheck(connection: sqlite3.Connection) -> None:
            original_check(connection)
            precheck_completed.set()
            if not allow_write.wait(timeout=5):
                raise AssertionError("rotation race was not released")

        monkeypatch.setattr(
            stale_store,
            "_require_current_public_authority_configuration",
            pause_after_precheck,
        )
        append_task = asyncio.create_task(
            stale_store.append_event(
                "session",
                Event(
                    id="must-not-commit",
                    type=EventType.SESSION_COMPLETED,
                    session_id="session",
                ),
            )
        )
        assert await asyncio.to_thread(precheck_completed.wait, 5)
        rotated_store = SQLiteSessionStore(
            db_path,
            public_authority_alias_codec=rotated_codec,
        )
        allow_write.set()
        with pytest.raises(sqlite3.IntegrityError, match="stale public authority"):
            await append_task
        assert await rotated_store.load_events("session") == []
        await rotated_store.close()
        await stale_store.close()

    asyncio.run(scenario())


def test_sqlite_public_authority_aliases_follow_writes_and_backfill_all_sources(
    tmp_path,
) -> None:
    db_path = tmp_path / "public-authority-backfill.sqlite"
    unconfigured = SQLiteSessionStore(db_path)

    async def write_legacy_sources() -> None:
        await unconfigured.create(
            RunRequest(
                agent_name="assistant",
                session_id="legacy-private-session",
                messages=[],
            ),
            identity=_identity(),
        )
        await unconfigured.append_event(
            "legacy-private-session",
            Event(
                id="legacy-event-only",
                type=EventType.INTERACTION_COMPLETED,
                session_id="legacy-private-session",
                interaction_id="legacy-event-interaction",
            ),
        )
        await unconfigured.append_transcript_messages(
            "legacy-private-session",
            [Message.text("user", "legacy")],
            interaction_id="legacy-transcript-interaction",
        )
        await unconfigured.append_event(
            "legacy-private-session",
            Event(
                id="legacy-turn-only",
                type=EventType.TURN_COMPLETED,
                session_id="legacy-private-session",
                payload={"interaction_ids": ["legacy-nested-interaction"]},
            ),
        )
        await unconfigured.close()

    asyncio.run(write_legacy_sources())

    connection = sqlite3.connect(db_path)
    try:
        connection.create_function(
            "cayu_public_authority_alias",
            3,
            lambda _value, _field, _scope: None,
        )
        connection.create_function("cayu_public_authority_active_key_id", 0, lambda: None)
        connection.create_function("cayu_public_authority_keyring_fingerprint", 0, lambda: None)
        connection.create_function(
            "cayu_public_authority_aliases",
            3,
            lambda _value, _field, _scope: "[]",
        )
        connection.execute(
            """
            INSERT INTO cayu_events (
                session_id, event_id, event_type, timestamp, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "legacy-private-session",
                "legacy-malformed-nested-turn",
                str(EventType.TURN_COMPLETED),
                datetime.now(UTC).isoformat(),
                '{"interaction_ids":["","   "]}',
            ),
        )
        connection.commit()
    finally:
        connection.close()

    codec = _public_authority_codec(active_key_id="rotated", retained=True)
    configured = SQLiteSessionStore(db_path, public_authority_alias_codec=codec)

    async def assert_backfill_and_new_write() -> None:
        for session_alias in codec.aliases(
            "legacy-private-session",
            field_name="session_id",
        ):
            assert (
                await configured.resolve_public_authority_alias(
                    session_alias,
                    field_name="session_id",
                )
                == "legacy-private-session"
            )
        for private_interaction in (
            "legacy-event-interaction",
            "legacy-transcript-interaction",
            "legacy-nested-interaction",
        ):
            aliases = codec.aliases(
                private_interaction,
                field_name="interaction_id",
                session_id="legacy-private-session",
            )
            for alias in aliases:
                assert (
                    await configured.resolve_public_authority_alias(
                        alias,
                        field_name="interaction_id",
                        scope_session_id="legacy-private-session",
                    )
                    == private_interaction
                )

        await configured.create(
            RunRequest(
                agent_name="assistant",
                session_id="new-private-session",
                messages=[],
            ),
            identity=_identity(),
        )
        new_alias = codec.encode("new-private-session", field_name="session_id")
        assert (
            await configured.resolve_public_authority_alias(
                new_alias,
                field_name="session_id",
            )
            == "new-private-session"
        )
        await configured.close()

    asyncio.run(assert_backfill_and_new_write())


def test_sqlite_session_store_persists_sessions_events_and_checkpoints(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run_store_operations() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sqlite",
                environment_name="local-dev",
                messages=[Message.text("user", "hi")],
                metadata={"project_id": 123},
            ),
            identity=SessionIdentity(
                provider_name="anthropic",
                model="claude-test",
                runtime_name="cayu",
                runtime_version="test-version",
            ),
        )
        assert session.status == SessionStatus.PENDING
        assert session.provider_name == "anthropic"
        assert session.model == "claude-test"
        assert session.runtime_name == "cayu"
        assert session.runtime_version == "test-version"

        await store.update_status("sess_sqlite", SessionStatus.RUNNING)
        await store.append_event(
            "sess_sqlite",
            Event(
                type=EventType.SESSION_STARTED,
                session_id="sess_sqlite",
                agent_name="assistant",
                environment_name="local-dev",
                payload={"step": 1},
            ),
        )
        await store.append_event(
            "sess_sqlite",
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="sess_sqlite",
                agent_name="assistant",
                environment_name="local-dev",
                payload={"finish_reason": "stop"},
            ),
        )
        await store.append_transcript_messages(
            "sess_sqlite",
            [
                Message.text("user", "hi"),
                Message.text("assistant", "hello"),
            ],
        )
        await store.checkpoint(
            "sess_sqlite",
            {"messages": [{"role": "user", "content": "hi"}], "step": 1},
        )
        await _close(store)

    asyncio.run(run_store_operations())

    reopened = SQLiteSessionStore(db_path)

    async def assert_reopened_state() -> None:
        session = await reopened.load("sess_sqlite")
        events = await reopened.load_events("sess_sqlite")
        transcript = await reopened.load_transcript("sess_sqlite")
        checkpoint = await reopened.load_checkpoint("sess_sqlite")

        assert session is not None
        assert session.agent_name == "assistant"
        assert session.environment_name == "local-dev"
        assert session.status == SessionStatus.RUNNING
        assert session.metadata == {"project_id": 123}
        assert [event.type for event in events] == [
            EventType.SESSION_STARTED,
            EventType.MODEL_COMPLETED,
        ]
        assert [event.payload for event in events] == [
            {"step": 1},
            {"finish_reason": "stop"},
        ]
        assert [message.role for message in transcript] == ["user", "assistant"]
        assert [message.content[0].text for message in transcript] == ["hi", "hello"]
        assert checkpoint == {
            "messages": [{"role": "user", "content": "hi"}],
            "step": 1,
        }
        await _close(reopened)

    asyncio.run(assert_reopened_state())


def test_sqlite_session_store_atomically_appends_transcript_and_transforms_checkpoint(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_atomic_transcript_checkpoint",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.checkpoint(
            "sess_atomic_transcript_checkpoint",
            {"pending_tool_approval": {"approval_id": "approval_1"}},
        )
        await store.append_transcript_messages_and_transform_checkpoint(
            "sess_atomic_transcript_checkpoint",
            [Message.text("assistant", "done")],
            lambda _session, _checkpoint: {"closed": True},
        )
        await _close(store)

    asyncio.run(run_store_operations())

    reopened = SQLiteSessionStore(db_path)

    async def assert_reopened_state() -> None:
        transcript = await reopened.load_transcript("sess_atomic_transcript_checkpoint")
        checkpoint = await reopened.load_checkpoint("sess_atomic_transcript_checkpoint")

        assert [message.role for message in transcript] == ["assistant"]
        assert transcript[0].content[0].text == "done"
        assert checkpoint == {"closed": True}
        await _close(reopened)

    asyncio.run(assert_reopened_state())


def test_sqlite_checkpoint_transforms_run_off_the_event_loop(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")

    async def run_store_operations() -> tuple[int, list[int]]:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_checkpoint_transform_thread",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        event_loop_thread = threading.get_ident()
        transform_threads: list[int] = []

        def transform(_session, checkpoint):
            transform_threads.append(threading.get_ident())
            return {} if checkpoint is None else checkpoint

        await store.transform_checkpoint("sess_checkpoint_transform_thread", transform)
        await store.append_transcript_messages_and_transform_checkpoint(
            "sess_checkpoint_transform_thread",
            [Message.text("assistant", "done")],
            transform,
        )
        await _close(store)
        return event_loop_thread, transform_threads

    event_loop_thread, transform_threads = asyncio.run(run_store_operations())

    assert len(transform_threads) == 2
    assert all(thread_id != event_loop_thread for thread_id in transform_threads)


def test_sqlite_session_store_atomically_transitions_status_and_checkpoint(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_atomic_status_checkpoint",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        session = await store.transition_status_and_checkpoint(
            "sess_atomic_status_checkpoint",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.INTERRUPTING,
            checkpoint_transform=lambda _session, checkpoint: {
                **({} if checkpoint is None else checkpoint),
                "pending_session_interrupt": {"reason": "operator stop"},
            },
        )
        checkpoint = await store.load_checkpoint("sess_atomic_status_checkpoint")
        await _close(store)
        return session, checkpoint

    session, checkpoint = asyncio.run(run_store_operations())

    assert session.status == SessionStatus.INTERRUPTING
    assert checkpoint == {"pending_session_interrupt": {"reason": "operator stop"}}


def test_sqlite_session_store_rejects_stale_atomic_status_checkpoint_transition(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    first = SQLiteSessionStore(db_path)
    second = SQLiteSessionStore(db_path)

    async def run_store_operations() -> None:
        await first.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_stale_atomic_status_checkpoint",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await second.update_status(
            "sess_stale_atomic_status_checkpoint",
            SessionStatus.RUNNING,
        )

        with pytest.raises(ValueError, match="Session status transition not allowed"):
            await first.transition_status_and_checkpoint(
                "sess_stale_atomic_status_checkpoint",
                from_statuses={SessionStatus.PENDING},
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=lambda _session, checkpoint: {
                    **({} if checkpoint is None else checkpoint),
                    "pending_session_interrupt": {"reason": "operator stop"},
                },
            )

        session = await first.load("sess_stale_atomic_status_checkpoint")
        checkpoint = await first.load_checkpoint("sess_stale_atomic_status_checkpoint")
        await first.close()
        await second.close()
        return session, checkpoint

    session, checkpoint = asyncio.run(run_store_operations())

    assert session is not None
    assert session.status == SessionStatus.RUNNING
    assert checkpoint is None


def test_sqlite_session_store_locks_checkpoint_during_atomic_status_checkpoint_transition(
    tmp_path,
):
    db_path = tmp_path / "sessions.sqlite"
    first = SQLiteSessionStore(db_path)

    async def run_store_operations() -> None:
        await first.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_atomic_checkpoint_lock",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await first.checkpoint("sess_atomic_checkpoint_lock", {"existing": True})

        def transform(_session: Session, checkpoint: dict | None) -> dict:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                connection = sqlite3.connect(db_path, timeout=0)
                try:
                    connection.execute(
                        """
                        INSERT INTO cayu_checkpoints (session_id, state_json, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(session_id) DO UPDATE SET
                            state_json = excluded.state_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            "sess_atomic_checkpoint_lock",
                            '{"external": true}',
                            "2026-01-01T00:00:00+00:00",
                        ),
                    )
                    connection.commit()
                finally:
                    connection.close()
            return {
                **({} if checkpoint is None else checkpoint),
                "pending_session_interrupt": {"reason": "operator stop"},
            }

        session = await first.transition_status_and_checkpoint(
            "sess_atomic_checkpoint_lock",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.INTERRUPTING,
            checkpoint_transform=transform,
        )
        checkpoint = await first.load_checkpoint("sess_atomic_checkpoint_lock")
        await first.close()
        return session, checkpoint

    session, checkpoint = asyncio.run(run_store_operations())

    assert session.status == SessionStatus.INTERRUPTING
    assert checkpoint == {
        "existing": True,
        "pending_session_interrupt": {"reason": "operator stop"},
    }


def test_sqlite_session_store_atomic_status_checkpoint_returns_written_snapshot(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    first = SQLiteSessionStore(db_path)
    second = SQLiteSessionStore(db_path)

    async def run_store_operations() -> tuple[Session, Session | None]:
        await first.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_atomic_return_snapshot",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        returned = await first.transition_status_and_checkpoint(
            "sess_atomic_return_snapshot",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.INTERRUPTING,
            checkpoint_transform=lambda _session, checkpoint: {
                **({} if checkpoint is None else checkpoint),
                "pending_session_interrupt": {"reason": "operator stop"},
            },
        )
        await second.update_status("sess_atomic_return_snapshot", SessionStatus.INTERRUPTED)
        loaded = await first.load("sess_atomic_return_snapshot")
        await first.close()
        await second.close()
        return returned, loaded

    returned, loaded = asyncio.run(run_store_operations())

    assert returned.status == SessionStatus.INTERRUPTING
    assert loaded is not None
    assert loaded.status == SessionStatus.INTERRUPTED


def test_sqlite_session_store_persists_forked_session_state(tmp_path):
    db_path = tmp_path / "forks.sqlite"
    store = SQLiteSessionStore(db_path)
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run_operations() -> None:
        await _collect_app_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_sqlite_fork_source",
                    messages=[Message.text("user", "first request")],
                )
            )
        )
        await store.checkpoint("sess_sqlite_fork_source", {"context_compaction": {}})
        events = await _collect_app_events(
            app.fork_session(
                ForkSessionRequest(
                    source_session_id="sess_sqlite_fork_source",
                    session_id="sess_sqlite_fork_child",
                )
            )
        )
        assert [event.type for event in events] == [EventType.SESSION_FORKED]
        await _close(store)

    asyncio.run(run_operations())

    reopened = SQLiteSessionStore(db_path)

    async def assert_persisted() -> None:
        fork = await reopened.load("sess_sqlite_fork_child")
        assert fork is not None
        assert fork.parent_session_id == "sess_sqlite_fork_source"
        assert fork.status == SessionStatus.COMPLETED
        transcript = await reopened.load_transcript("sess_sqlite_fork_child")
        assert [message.content[0].text for message in transcript] == [
            "first request",
            "first answer",
        ]
        checkpoint = await reopened.load_checkpoint("sess_sqlite_fork_child")
        assert checkpoint == {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
            "context_compaction": {},
        }
        children = (
            await reopened.list_sessions(SessionQuery(parent_session_id="sess_sqlite_fork_source"))
        ).sessions
        assert [session.id for session in children] == ["sess_sqlite_fork_child"]
        events = await reopened.load_events("sess_sqlite_fork_child")
        assert [event.type for event in events] == [
            EventType.SESSION_FORKED,
            EventType.TARGETED_TOOL_GRANT_FORK_RESET,
        ]
        await _close(reopened)

    asyncio.run(assert_persisted())


def test_sqlite_session_store_persists_run_request_parent_session_id(tmp_path):
    db_path = tmp_path / "run-parent.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sqlite_run_parent",
                messages=[Message.text("user", "parent")],
            ),
            identity=_identity(),
        )
        child = await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_sqlite_run_child",
                parent_session_id="sess_sqlite_run_parent",
                causal_budget_id="job_sqlite_run_parent",
                messages=[Message.text("user", "child")],
            ),
            identity=_identity(),
        )
        assert child.parent_session_id == "sess_sqlite_run_parent"
        await _close(store)

    asyncio.run(run_operations())

    reopened = SQLiteSessionStore(db_path)

    async def assert_persisted() -> None:
        child = await reopened.load("sess_sqlite_run_child")
        assert child is not None
        assert child.parent_session_id == "sess_sqlite_run_parent"
        assert child.causal_budget_id == "job_sqlite_run_parent"
        children = (
            await reopened.list_sessions(SessionQuery(parent_session_id="sess_sqlite_run_parent"))
        ).sessions
        assert [session.id for session in children] == ["sess_sqlite_run_child"]
        await _close(reopened)

    asyncio.run(assert_persisted())


def test_sqlite_session_store_rejects_fork_status_mismatch(tmp_path):
    store = SQLiteSessionStore(tmp_path / "fork-status.sqlite")

    async def run_operations() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sqlite_fork_status_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="Fork status must match"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_sqlite_fork_status_child",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.RUNNING,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=None,
                checkpoint_transform=None,
            )
        await _close(store)

    asyncio.run(run_operations())


def test_sqlite_session_store_rejects_fork_provider_mismatch(tmp_path):
    store = SQLiteSessionStore(tmp_path / "fork-provider.sqlite")

    async def run_operations() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sqlite_fork_provider_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)

        with pytest.raises(ValueError, match="Fork provider_name must match"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_sqlite_fork_provider_child",
                    agent_name="assistant",
                    provider_name="other",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                expected_source_run_epoch=source.run_epoch,
                transcript_cursor=None,
                checkpoint_transform=None,
            )
        await _close(store)

    asyncio.run(run_operations())


def test_sqlite_session_store_transforms_current_checkpoint_during_fork(tmp_path):
    store = SQLiteSessionStore(tmp_path / "fork-checkpoint.sqlite")

    async def run_operations() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sqlite_fork_checkpoint_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        await store.checkpoint(source.id, {"version": 2})

        await store.create_fork(
            source_session_id=source.id,
            fork=Session(
                id="sess_sqlite_fork_checkpoint_child",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                parent_session_id=source.id,
                invocation=fork_session_invocation(source),
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            expected_source_run_epoch=source.run_epoch,
            transcript_cursor=None,
            checkpoint_transform=lambda _session, checkpoint: {
                "copied_version": checkpoint["version"] if checkpoint else None
            },
        )

        assert await store.load_checkpoint("sess_sqlite_fork_checkpoint_child") == {
            "copied_version": 2
        }
        await _close(store)

    asyncio.run(run_operations())


def test_sqlite_session_store_fork_reads_checkpoint_inside_write_transaction(tmp_path):
    db_path = tmp_path / "fork-transaction.sqlite"
    store = SQLiteSessionStore(db_path)
    concurrent_write_errors: list[str] = []

    async def run_operations() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sqlite_fork_tx_source",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.update_status(source.id, SessionStatus.COMPLETED)
        await store.checkpoint(source.id, {"version": 2})

        def transform(_session: Session, checkpoint: dict | None) -> dict:
            assert checkpoint == {"version": 2}
            connection = sqlite3.connect(db_path, timeout=0)
            try:
                with (
                    pytest.raises(sqlite3.OperationalError, match="database is locked"),
                    connection,
                ):
                    connection.execute(
                        """
                        UPDATE cayu_checkpoints
                        SET state_json = ?
                        WHERE session_id = ?
                        """,
                        ('{"version":99}', source.id),
                    )
            except AssertionError:
                concurrent_write_errors.append("checkpoint write was not locked")
            finally:
                connection.close()
            return {"copied_version": checkpoint["version"] if checkpoint else None}

        await store.create_fork(
            source_session_id=source.id,
            fork=Session(
                id="sess_sqlite_fork_tx_child",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                parent_session_id=source.id,
                invocation=fork_session_invocation(source),
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            expected_source_run_epoch=source.run_epoch,
            transcript_cursor=None,
            checkpoint_transform=transform,
        )

        assert concurrent_write_errors == []
        assert await store.load_checkpoint(source.id) == {"version": 2}
        assert await store.load_checkpoint("sess_sqlite_fork_tx_child") == {"copied_version": 2}
        await _close(store)

    asyncio.run(run_operations())


def test_sqlite_session_store_exposes_queryable_event_identity_columns(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run_store_operations() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_query_columns",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.append_event(
            "sess_query_columns",
            Event(
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="sess_query_columns",
                agent_name="assistant",
                environment_name="local-dev",
                tool_name="read_file",
                payload={"path": "README.md"},
            ),
        )
        await _close(store)

    asyncio.run(run_store_operations())

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT session_id, event_type, agent_name, environment_name,
                   tool_name, payload_json
            FROM cayu_events
            WHERE tool_name = ?
            """,
            ("read_file",),
        ).fetchone()
    finally:
        connection.close()

    assert dict(row) == {
        "session_id": "sess_query_columns",
        "event_type": EventType.TOOL_CALL_COMPLETED,
        "agent_name": "assistant",
        "environment_name": "local-dev",
        "tool_name": "read_file",
        "payload_json": '{"path":"README.md"}',
    }


def test_sqlite_session_store_rejects_duplicate_sessions_and_mismatched_events(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")

    async def run_store_operations() -> None:
        request = RunRequest(
            agent_name="assistant",
            session_id="sess_duplicate",
            messages=[Message.text("user", "hi")],
        )
        await store.create(request, identity=_identity())

        with pytest.raises(ValueError, match="Session already exists"):
            await store.create(request, identity=_identity())

        with pytest.raises(ValueError, match="Event session_id"):
            await store.append_event(
                "sess_duplicate",
                Event(
                    type=EventType.SESSION_STARTED,
                    session_id="other_session",
                ),
            )

        event = Event(
            id="event_duplicate",
            type=EventType.SESSION_STARTED,
            session_id="sess_duplicate",
        )
        await store.append_event("sess_duplicate", event)
        with pytest.raises(ValueError, match="Event already exists"):
            await store.append_event("sess_duplicate", event)

        with pytest.raises(KeyError, match="Session not found"):
            await store.load_events("missing_session")

        await _close(store)

    asyncio.run(run_store_operations())


def test_sqlite_session_store_validate_mode_fails_fast_on_uninitialized(tmp_path):
    # validate-at-startup (ADR 0001 Q4): a store opened in validate mode against an
    # empty database fails fast instead of silently creating the schema.
    db_path = tmp_path / "sessions.sqlite"
    with pytest.raises(schema_migrations.SchemaUninitialized):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)


def test_sqlite_revision_52_rejects_a_conflicting_targeted_grant_index(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite"
    creator = SQLiteSessionStore(db_path)
    asyncio.run(_close(creator))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP INDEX idx_cayu_targeted_tool_grants_interaction")
        connection.execute(
            "CREATE INDEX idx_cayu_targeted_tool_grants_interaction "
            "ON cayu_targeted_tool_grants(grant_id)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="targeted-grant contention contract"):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 52")
        connection.execute("PRAGMA user_version = 51")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="targeted-grant contention contract"):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    connection = sqlite3.connect(db_path)
    try:
        recorded = connection.execute(
            "SELECT COUNT(*) FROM cayu_schema_migrations WHERE revision = 52"
        ).fetchone()
        version = connection.execute("PRAGMA user_version").fetchone()
    finally:
        connection.close()
    assert recorded == (0,)
    assert version == (51,)


def test_sqlite_revision_52_rejects_a_populated_pre_grant_session_store(tmp_path) -> None:
    db_path = tmp_path / "pre-targeted-grants.sqlite"
    creator = SQLiteSessionStore(db_path)

    async def create_session() -> None:
        await creator.create(
            RunRequest(agent_name="assistant", session_id="existing", messages=[]),
            identity=_identity(),
        )
        await _close(creator)

    asyncio.run(create_session())

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE cayu_targeted_tool_grant_uses")
        connection.execute("DROP TABLE cayu_targeted_tool_grants")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 52")
        connection.execute("PRAGMA user_version = 51")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(schema_migrations.SchemaTooOld, match="clean prerelease break"):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (51,)
        assert connection.execute(
            "SELECT MAX(revision) FROM cayu_schema_migrations"
        ).fetchone() == (51,)
        assert connection.execute("SELECT id FROM cayu_sessions").fetchall() == [("existing",)]
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name LIKE 'cayu_targeted_tool_grant%'"
            ).fetchall()
            == []
        )
    finally:
        connection.close()


def test_sqlite_revision_52_requires_the_targeted_use_lookup_index(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite"
    creator = SQLiteSessionStore(db_path)
    asyncio.run(_close(creator))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP INDEX idx_cayu_targeted_tool_grant_uses_grant")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="targeted-grant contention contract"):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)


def test_sqlite_revision_52_rejects_a_missing_targeted_grant_table(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite"
    creator = SQLiteSessionStore(db_path)
    asyncio.run(_close(creator))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE cayu_targeted_tool_grant_uses")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="targeted-grant durability contract"):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)


def test_sqlite_memory_evidence_schema_validation_fails_closed(tmp_path) -> None:
    db_path = tmp_path / "malformed-memory-evidence.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP INDEX idx_cayu_context_exposures_interaction_step_page")
        connection.execute(
            """
            CREATE INDEX idx_cayu_context_exposures_interaction_step_page
            ON cayu_context_exposures(
                session_id, interaction_id, model_step_id, created_at, exposure_id
            )
            WHERE state = 'planned'
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="revision-51 memory evidence contract"):
        SQLiteSessionStore(
            db_path,
            schema_mode=schema_migrations.SchemaMode.VALIDATE,
        )


@pytest.mark.parametrize(
    "unique_index_ddl",
    (
        "CREATE UNIQUE INDEX unexpected_memory_receipt_session_unique "
        "ON cayu_recall_receipts(session_id)",
        "CREATE UNIQUE INDEX unexpected_memory_exposure_session_unique "
        "ON cayu_context_exposures(session_id)",
        "CREATE UNIQUE INDEX unexpected_memory_item_receipt_unique "
        "ON cayu_recall_item_exposures(receipt_id)",
    ),
)
def test_sqlite_memory_evidence_schema_rejects_standalone_unique_indexes(
    tmp_path,
    unique_index_ddl: str,
) -> None:
    db_path = tmp_path / "overrestricted-memory-evidence.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(unique_index_ddl)
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="revision-51 memory evidence contract"):
        SQLiteSessionStore(
            db_path,
            schema_mode=schema_migrations.SchemaMode.VALIDATE,
        )


def test_sqlite_memory_evidence_combined_scope_pages_use_composite_indexes(tmp_path) -> None:
    db_path = tmp_path / "memory-evidence-page-plan.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        receipt_details = [
            row[3]
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM cayu_recall_receipts
                WHERE session_id = ? AND interaction_id = ? AND model_step_id = ?
                  AND (created_at, receipt_id) > (?, ?)
                ORDER BY created_at, receipt_id COLLATE BINARY
                LIMIT ?
                """,
                (
                    "session",
                    "interaction",
                    "mstep_" + "1" * 32,
                    "2026-08-22T00:00:00.000000+00:00",
                    "receipt",
                    51,
                ),
            )
        ]
        exposure_details = [
            row[3]
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT * FROM cayu_context_exposures
                WHERE session_id = ? AND interaction_id = ? AND model_step_id = ?
                  AND (created_at, exposure_id) > (?, ?)
                ORDER BY created_at, exposure_id COLLATE BINARY
                LIMIT ?
                """,
                (
                    "session",
                    "interaction",
                    "mstep_" + "1" * 32,
                    "2026-08-22T00:00:00.000000+00:00",
                    "exposure",
                    51,
                ),
            )
        ]
    finally:
        connection.close()

    assert any(
        "idx_cayu_recall_receipts_interaction_step_page" in detail for detail in receipt_details
    )
    assert any(
        "idx_cayu_context_exposures_interaction_step_page" in detail for detail in exposure_details
    )


def test_sqlite_latest_migrates_queue_and_event_side_effect_handoff(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 19")
        connection.execute("DROP TRIGGER IF EXISTS cayu_protect_undelivered_event_side_effects")
        connection.execute("DROP TABLE cayu_mcp_manifest_baselines")
        connection.execute("DROP TABLE cayu_persisted_event_side_effects")
        connection.execute("DROP TABLE cayu_session_message_queue")
        connection.execute("DROP INDEX idx_cayu_sessions_parent_created_id")
        connection.execute("DROP INDEX idx_cayu_tasks_session_created_id")
        connection.execute("DROP INDEX idx_cayu_tasks_parent_created_id")
        connection.execute("PRAGMA user_version = 18")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match=rf"requires >= {sqlite_storage._SQLITE_SESSION_MIN_REQUIRED_REVISION}",
    ):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match=rf"requires >= {sqlite_storage._SQLITE_TASK_MIN_REQUIRED_REVISION}",
    ):
        SQLiteTaskStore(
            db_path,
            schema_mode=schema_migrations.SchemaMode.VALIDATE,
        )

    migrated = SQLiteSessionStore(
        db_path,
        schema_mode=schema_migrations.SchemaMode.MIGRATE,
    )
    asyncio.run(_close(migrated))

    connection = sqlite3.connect(db_path)
    try:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_session_message_queue'"
        ).fetchone()
        revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 19"
        ).fetchone()
        handoff_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_persisted_event_side_effects'"
        ).fetchone()
        handoff_revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 20"
        ).fetchone()
        billing_revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 21"
        ).fetchone()
        manifest_baseline_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_mcp_manifest_baselines'"
        ).fetchone()
        manifest_revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 22"
        ).fetchone()
        topology_revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 24"
        ).fetchone()
        topology_index = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_cayu_sessions_parent_created_id'"
        ).fetchone()
        task_topology_revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 27"
        ).fetchone()
        public_authority_revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 28"
        ).fetchone()
        public_authority_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_public_authority_aliases'"
        ).fetchone()
        public_authority_keys_table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_public_authority_alias_keys'"
        ).fetchone()
        input_contract_proof_revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 31"
        ).fetchone()
        input_contract_proof_column = connection.execute(
            'SELECT name, type, "notnull", dflt_value '
            "FROM pragma_table_info('cayu_events') "
            "WHERE name = 'input_contract_runtime_owned'"
        ).fetchone()
        file_attestation_proof_revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 54"
        ).fetchone()
        file_attestation_proof_column = connection.execute(
            'SELECT name, type, "notnull", dflt_value '
            "FROM pragma_table_info('cayu_events') "
            "WHERE name = 'file_attachment_attestations_runtime_owned'"
        ).fetchone()
        delayed_task_revision = connection.execute(
            "SELECT kind, compatible_from FROM cayu_schema_migrations WHERE revision = 34"
        ).fetchone()
        delayed_task_column = connection.execute(
            'SELECT name, type, "notnull", dflt_value '
            "FROM pragma_table_info('cayu_tasks') WHERE name = 'available_at'"
        ).fetchone()
        delayed_task_index = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_cayu_tasks_claim_availability'"
        ).fetchone()
        task_session_topology_index = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_cayu_tasks_session_created_id'"
        ).fetchone()
        task_parent_topology_index = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_cayu_tasks_parent_created_id'"
        ).fetchone()
        legacy_writer_trigger = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'cayu_events_enqueue_persisted_side_effect'"
        ).fetchone()
        retention_guard = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'cayu_protect_undelivered_event_side_effects'"
        ).fetchone()
    finally:
        connection.close()
    assert table == ("cayu_session_message_queue",)
    assert revision == ("breaking", 19)
    assert handoff_table == ("cayu_persisted_event_side_effects",)
    assert handoff_revision == ("additive", 19)
    assert billing_revision == ("breaking", 21)
    assert manifest_baseline_table == ("cayu_mcp_manifest_baselines",)
    assert manifest_revision == ("breaking", 22)
    assert topology_revision == ("additive", 23)
    assert topology_index == ("idx_cayu_sessions_parent_created_id",)
    assert task_topology_revision == ("additive", 26)
    assert public_authority_revision == ("breaking", 28)
    assert public_authority_table == ("cayu_public_authority_aliases",)
    assert public_authority_keys_table == ("cayu_public_authority_alias_keys",)
    assert input_contract_proof_revision == ("breaking", 31)
    assert input_contract_proof_column == (
        "input_contract_runtime_owned",
        "INTEGER",
        1,
        "0",
    )
    assert file_attestation_proof_revision == ("breaking", 54)
    assert file_attestation_proof_column == (
        "file_attachment_attestations_runtime_owned",
        "INTEGER",
        1,
        "0",
    )
    assert delayed_task_revision == ("breaking", 34)
    assert delayed_task_column == ("available_at", "TEXT", 0, None)
    assert delayed_task_index == ("idx_cayu_tasks_claim_availability",)
    assert task_session_topology_index == ("idx_cayu_tasks_session_created_id",)
    assert task_parent_topology_index == ("idx_cayu_tasks_parent_created_id",)
    assert legacy_writer_trigger is None
    assert retention_guard == ("cayu_protect_undelivered_event_side_effects",)


def test_sqlite_profiled_dispatch_stores_reject_revision_thirty_nine(tmp_path) -> None:
    db_path = tmp_path / "pre-profiled-dispatch.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 40")
        connection.execute("DROP INDEX idx_cayu_checkpoints_queued_dispatch_run")
        connection.execute("DROP INDEX idx_cayu_checkpoints_queued_dispatch_receipts")
        connection.execute("PRAGMA user_version = 39")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match=rf"requires >= {sqlite_storage._SQLITE_SESSION_MIN_REQUIRED_REVISION}",
    ):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)
    with pytest.raises(schema_migrations.SchemaTooOld, match="requires >= 49"):
        SQLiteTaskStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)


def test_sqlite_session_store_rejects_populated_revision_thirteen_database(tmp_path):
    # Revision 26 intentionally refuses to infer interaction attribution for
    # populated prerelease databases, including databases that start farther back.
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def create() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sqlite_rev12",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await _close(store)

    asyncio.run(create())

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 14")
        connection.execute("PRAGMA user_version = 13")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match=rf"requires >= {sqlite_storage._SQLITE_SESSION_MIN_REQUIRED_REVISION}",
    ):
        SQLiteSessionStore(db_path)

    with pytest.raises(schema_migrations.SchemaTooOld, match="clean prerelease break"):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (13,)
        assert connection.execute(
            "SELECT MAX(revision) FROM cayu_schema_migrations"
        ).fetchone() == (13,)
        assert connection.execute(
            "SELECT id FROM cayu_sessions WHERE id = ?",
            ("sess_sqlite_rev12",),
        ).fetchone() == ("sess_sqlite_rev12",)
    finally:
        connection.close()


def test_sqlite_session_store_rejects_populated_revision_fourteen_database(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def create() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_sqlite_rev14",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await _close(store)

    asyncio.run(create())

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 15")
        connection.execute("DROP INDEX idx_cayu_checkpoints_pending_interruption_cascade")
        connection.execute("PRAGMA user_version = 14")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match=rf"requires >= {sqlite_storage._SQLITE_SESSION_MIN_REQUIRED_REVISION}",
    ):
        SQLiteSessionStore(db_path)

    with pytest.raises(schema_migrations.SchemaTooOld, match="clean prerelease break"):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (14,)
        assert connection.execute(
            "SELECT MAX(revision) FROM cayu_schema_migrations"
        ).fetchone() == (14,)
        assert connection.execute(
            "SELECT id FROM cayu_sessions WHERE id = ?",
            ("sess_sqlite_rev14",),
        ).fetchone() == ("sess_sqlite_rev14",)
        index = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_cayu_checkpoints_pending_interruption_cascade'"
        ).fetchone()
    finally:
        connection.close()
    assert index is None


def test_sqlite_session_store_rejects_populated_pre_invocation_database(tmp_path) -> None:
    db_path = tmp_path / "pre-invocation.sqlite"
    store = SQLiteSessionStore(db_path)

    async def create() -> None:
        await store.create(
            RunRequest(agent_name="assistant", session_id="existing", messages=[]),
            identity=_identity(),
        )
        await _close(store)

    asyncio.run(create())

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 36")
        connection.execute("PRAGMA user_version = 35")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match="requires invocation provenance for every session",
    ):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (35,)
        assert connection.execute(
            "SELECT MAX(revision) FROM cayu_schema_migrations"
        ).fetchone() == (35,)
    finally:
        connection.close()


def test_sqlite_revision_seventeen_requires_session_operation_migration(tmp_path) -> None:
    db_path = tmp_path / "revision-17-session-operations.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 18")
        connection.execute("DROP TABLE cayu_session_operations")
        connection.execute("PRAGMA user_version = 17")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        schema_migrations.SchemaTooOld,
        match=rf"requires >= {sqlite_storage._SQLITE_SESSION_MIN_REQUIRED_REVISION}",
    ):
        SQLiteSessionStore(db_path)

    migrated = SQLiteSessionStore(
        db_path,
        schema_mode=schema_migrations.SchemaMode.MIGRATE,
    )
    asyncio.run(_close(migrated))

    connection = sqlite3.connect(db_path)
    try:
        operation_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_session_operations'"
        ).fetchone()
        revision = connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()

    assert operation_table is not None
    assert revision == schema_migrations.LATEST_REVISION


def test_sqlite_revision_seventeen_rejects_conflicting_same_name_index(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 17")
        connection.execute("DROP INDEX idx_cayu_checkpoints_pending_control_action")
        connection.execute("DROP INDEX idx_cayu_events_pending_action_barrier")
        connection.execute("DROP INDEX idx_cayu_events_pending_action_lookup")
        connection.execute(
            "CREATE INDEX idx_cayu_events_pending_action_lookup "
            "ON cayu_events(session_id, sequence)"
        )
        connection.execute("PRAGMA user_version = 16")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="conflicts with Cayu revision 17"):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    connection = sqlite3.connect(db_path)
    try:
        recorded = connection.execute(
            "SELECT COUNT(*) FROM cayu_schema_migrations WHERE revision = 17"
        ).fetchone()
    finally:
        connection.close()
    assert recorded == (0,)


def test_sqlite_revision_seventeen_validation_checks_exact_index_definition(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP INDEX idx_cayu_events_pending_action_lookup")
        connection.execute(
            "CREATE INDEX idx_cayu_events_pending_action_lookup "
            "ON cayu_events(session_id, event_type, sequence)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="conflicts with Cayu revision 17"):
        SQLiteSessionStore(db_path)

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP INDEX idx_cayu_events_pending_action_lookup")
        connection.commit()
    finally:
        connection.close()

    repaired = SQLiteSessionStore(
        db_path,
        schema_mode=schema_migrations.SchemaMode.MIGRATE,
    )
    asyncio.run(_close(repaired))

    validated = SQLiteSessionStore(db_path)
    asyncio.run(_close(validated))


def test_sqlite_revision_seventeen_migrate_repairs_missing_recorded_index(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP INDEX idx_cayu_events_pending_action_lookup")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="Required Cayu SQLite index is missing"):
        SQLiteSessionStore(db_path)

    repaired = SQLiteSessionStore(
        db_path,
        schema_mode=schema_migrations.SchemaMode.MIGRATE,
    )
    asyncio.run(_close(repaired))

    connection = sqlite3.connect(db_path)
    try:
        definition = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'index' AND name = 'idx_cayu_events_pending_action_lookup'"
        ).fetchone()
    finally:
        connection.close()
    assert definition is not None
    assert "event_type" in definition[0]
    assert "IS NOT NULL" in definition[0]


def test_sqlite_revision_twenty_three_requires_the_unique_reservation_index(
    tmp_path,
) -> None:
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP INDEX idx_cayu_events_budget_reservation_identity")
        connection.execute(
            "CREATE INDEX idx_cayu_events_budget_reservation_identity "
            "ON cayu_events(json_extract(payload_json, '$.reservation_id')) "
            "WHERE event_type = 'budget.reserved'"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="reservation identity contract"):
        SQLiteSessionStore(db_path)

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP INDEX idx_cayu_events_budget_reservation_identity")
        connection.commit()
    finally:
        connection.close()

    repaired = SQLiteSessionStore(
        db_path,
        schema_mode=schema_migrations.SchemaMode.MIGRATE,
    )
    asyncio.run(_close(repaired))

    connection = sqlite3.connect(db_path)
    try:
        definition = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'index' "
            "AND name = 'idx_cayu_events_budget_reservation_identity'"
        ).fetchone()
    finally:
        connection.close()
    assert definition is not None
    assert "CREATE UNIQUE INDEX" in definition[0]
    assert "reservation_id" in definition[0]
    assert "budget.reserved" in definition[0]


def test_sqlite_recorded_revision_twenty_three_fails_closed_without_registry(
    tmp_path,
) -> None:
    db_path = tmp_path / "sessions.sqlite"

    async def seed() -> None:
        store = SQLiteSessionStore(db_path)
        session = await store.create(
            RunRequest(
                session_id="sess_registry_repair",
                agent_name="assistant",
                messages=[Message.text("user", "seed")],
            ),
            identity=_identity(),
        )
        reservation_id = "bres_registry_repair"
        reserved = Event(
            type=EventType.BUDGET_RESERVED,
            session_id=session.id,
            payload={"reservation_id": reservation_id},
        )
        await store.append_event(session.id, reserved)
        reserved_claim = await store.claim_persisted_event_side_effect(
            session_id=session.id,
            event_id=reserved.id,
        )
        assert reserved_claim is not None
        await store.mark_persisted_event_side_effect_delivered(reserved_claim)
        released = Event(
            type=EventType.BUDGET_RESERVATION_RELEASED,
            session_id=session.id,
            payload={"reservation_id": reservation_id},
        )
        await store.append_event(session.id, released)
        released_claim = await store.claim_persisted_event_side_effect(
            session_id=session.id,
            event_id=released.id,
        )
        assert released_claim is not None
        await store.mark_persisted_event_side_effect_delivered(released_claim)
        await store.delete_session(session.id)
        await _close(store)

    asyncio.run(seed())

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE cayu_budget_reservation_identities")
        connection.execute(
            "CREATE TABLE cayu_budget_reservation_identities ("
            "reservation_id TEXT PRIMARY KEY, publication_id TEXT NOT NULL)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="reservation identity contract"):
        SQLiteSessionStore(db_path)

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE cayu_budget_reservation_identities")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="permanent reservation ownership registry"):
        SQLiteSessionStore(
            db_path,
            schema_mode=schema_migrations.SchemaMode.MIGRATE,
        )

    connection = sqlite3.connect(db_path)
    try:
        registry = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'cayu_budget_reservation_identities'"
        ).fetchone()
    finally:
        connection.close()
    assert registry is None


def test_sqlite_reservation_claim_is_atomic_across_store_instances(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite"

    async def exercise() -> None:
        first = SQLiteSessionStore(db_path)
        second = SQLiteSessionStore(db_path)
        stores = (first, second)
        publication_ids = ("evt_first_publication", "evt_second_publication")
        try:
            session = await first.create(
                RunRequest(
                    session_id="sess_cross_store_claim",
                    agent_name="assistant",
                    messages=[Message.text("user", "claim")],
                ),
                identity=_identity(),
            )
            results = await asyncio.gather(
                *(
                    store.claim_budget_reservation_identity(
                        reservation_id="bres_cross_store_claim",
                        publication_session_id=session.id,
                        publication_id=publication_id,
                    )
                    for store, publication_id in zip(stores, publication_ids, strict=True)
                ),
                return_exceptions=True,
            )

            winners = [index for index, result in enumerate(results) if result is None]
            conflicts = [
                result
                for result in results
                if isinstance(result, BudgetReservationIdentityConflict)
            ]
            assert len(winners) == 1
            assert len(conflicts) == 1

            winner = winners[0]
            await stores[winner].claim_budget_reservation_identity(
                reservation_id="bres_cross_store_claim",
                publication_session_id=session.id,
                publication_id=publication_ids[winner],
            )
        finally:
            await asyncio.gather(first.close(), second.close())

    asyncio.run(exercise())


def test_sqlite_session_store_coexists_with_foreign_app_tables(tmp_path):
    # The cayu_ prefix (ADR 0001 Decision 5) means an app's own unprefixed tables in
    # the same database no longer block initialization — they simply coexist.
    db_path = tmp_path / "sessions.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()

    store = SQLiteSessionStore(db_path)

    async def assert_initialized() -> None:
        assert (await store.list_sessions()).sessions == []
        await _close(store)

    asyncio.run(assert_initialized())


def test_sqlite_session_store_initializes_new_unversioned_database(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def assert_initialized() -> None:
        sessions = (await store.list_sessions()).sessions
        assert sessions == []
        await _close(store)

    asyncio.run(assert_initialized())

    connection = sqlite3.connect(db_path)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()

    # user_version now mirrors the ADR 0001 schema revision (the cross-backend
    # source of truth is the cayu_schema_migrations table).
    assert version == schema_migrations.LATEST_REVISION


def test_sqlite_transcript_search_trigger_shape_fails_closed_in_migrate_mode(
    tmp_path,
) -> None:
    db_path = tmp_path / "transcript-search-invalid.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            DROP TRIGGER cayu_transcript_messages_fts_insert;
            CREATE TRIGGER cayu_transcript_messages_fts_insert
            AFTER INSERT ON cayu_transcript_messages
            BEGIN
                SELECT 1;
            END;
            """
        )
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="maintenance triggers conflict"):
        SQLiteSessionStore(db_path)

    with pytest.raises(RuntimeError, match="maintenance triggers conflict"):
        SQLiteSessionStore(
            db_path,
            schema_mode=schema_migrations.SchemaMode.MIGRATE,
        )

    connection = sqlite3.connect(db_path)
    try:
        trigger_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'cayu_transcript_messages_fts_insert'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert "SELECT 1" in trigger_sql


def test_sqlite_revision_forty_six_rejects_populated_transcript_database_without_mutation(
    tmp_path,
) -> None:
    db_path = tmp_path / "pre-transcript-search.sqlite"
    store = SQLiteSessionStore(db_path)

    async def create_old_transcript() -> None:
        session = await store.create(
            RunRequest(
                agent_name="recall",
                session_id="pre-transcript-search",
                messages=[],
            ),
            identity=_identity(),
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("user", "visible")],
            interaction_id="old-interaction",
        )
        await _close(store)

    asyncio.run(create_old_transcript())

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            DROP TRIGGER cayu_transcript_messages_fts_insert;
            DROP TRIGGER cayu_transcript_messages_fts_delete;
            DROP TRIGGER cayu_transcript_messages_fts_update;
            DROP TRIGGER cayu_transcript_messages_search_document_insert;
            DROP TRIGGER cayu_transcript_messages_search_document_update;
            DROP TABLE cayu_transcript_messages_fts;
            DROP TABLE cayu_transcript_search_configuration;
            ALTER TABLE cayu_transcript_messages
                DROP COLUMN transcript_search_document;
            DELETE FROM cayu_schema_migrations WHERE revision >= 46;
            PRAGMA user_version = 45;
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(schema_migrations.SchemaTooOld, match="does not backfill"):
        SQLiteSessionStore(
            db_path,
            schema_mode=schema_migrations.SchemaMode.MIGRATE,
        )

    connection = sqlite3.connect(db_path)
    try:
        revision = connection.execute("SELECT MAX(revision) FROM cayu_schema_migrations").fetchone()
        version = connection.execute("PRAGMA user_version").fetchone()
        transcript_count = connection.execute(
            "SELECT COUNT(*) FROM cayu_transcript_messages "
            "WHERE session_id = 'pre-transcript-search'"
        ).fetchone()
        transcript_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cayu_transcript_messages)")
        }
        fts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_transcript_messages_fts'"
        ).fetchone()
    finally:
        connection.close()

    assert revision == (45,)
    assert version == (45,)
    assert transcript_count == (1,)
    assert "transcript_search_document" not in transcript_columns
    assert fts is None


def test_sqlite_revision_forty_six_migrates_empty_transcript_database(tmp_path) -> None:
    db_path = tmp_path / "empty-pre-transcript-search.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(
            """
            DROP TRIGGER cayu_transcript_messages_fts_insert;
            DROP TRIGGER cayu_transcript_messages_fts_delete;
            DROP TRIGGER cayu_transcript_messages_fts_update;
            DROP TRIGGER cayu_transcript_messages_search_document_insert;
            DROP TRIGGER cayu_transcript_messages_search_document_update;
            DROP TABLE cayu_transcript_messages_fts;
            DROP TABLE cayu_transcript_search_configuration;
            ALTER TABLE cayu_transcript_messages
                DROP COLUMN transcript_search_document;
            DELETE FROM cayu_schema_migrations WHERE revision >= 46;
            PRAGMA user_version = 45;
            """
        )
        connection.commit()
    finally:
        connection.close()

    migrated = SQLiteSessionStore(
        db_path,
        schema_mode=schema_migrations.SchemaMode.MIGRATE,
    )
    asyncio.run(_close(migrated))

    connection = sqlite3.connect(db_path)
    try:
        revision = connection.execute("SELECT MAX(revision) FROM cayu_schema_migrations").fetchone()
        version = connection.execute("PRAGMA user_version").fetchone()
        transcript_column = {
            row[1]: (row[2], row[3])
            for row in connection.execute("PRAGMA table_info(cayu_transcript_messages)")
        }["transcript_search_document"]
        fts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_transcript_messages_fts'"
        ).fetchone()
        tokenizer_configuration = connection.execute(
            "SELECT singleton, tokenizer_version FROM cayu_transcript_search_configuration"
        ).fetchone()
    finally:
        connection.close()

    assert revision == (schema_migrations.LATEST_REVISION,)
    assert version == (schema_migrations.LATEST_REVISION,)
    assert transcript_column == ("TEXT", 1)
    assert fts is not None
    assert tokenizer_configuration == (1, TRANSCRIPT_SEARCH_TOKENIZER_VERSION)


def test_sqlite_transcript_tokenizer_identity_mismatch_fails_closed(tmp_path) -> None:
    db_path = tmp_path / "transcript-tokenizer-mismatch.sqlite"
    store = SQLiteSessionStore(db_path)
    asyncio.run(_close(store))

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "UPDATE cayu_transcript_search_configuration "
            "SET tokenizer_version = 'incompatible-tokenizer'"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="tokenizer identity conflicts"):
        SQLiteSessionStore(
            db_path,
            schema_mode=schema_migrations.SchemaMode.MIGRATE,
        )

    connection = sqlite3.connect(db_path)
    try:
        marker = connection.execute(
            "SELECT tokenizer_version FROM cayu_transcript_search_configuration"
        ).fetchone()
    finally:
        connection.close()
    assert marker == ("incompatible-tokenizer",)


def test_sqlite_session_store_migrates_revision_one_database_to_latest_schema(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(sqlite_support._BASELINE_DDL)
        connection.execute(sqlite_support._MIGRATIONS_TABLE_DDL)
        connection.execute("DROP TABLE cayu_session_labels")
        connection.execute("DROP TABLE cayu_event_watcher_state")
        connection.execute(
            "INSERT INTO cayu_schema_migrations "
            "(revision, kind, compatible_from, checksum, applied_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, str(schema_migrations.RevisionKind.BREAKING), 1, None, "2026-01-01T00:00:00+00:00"),
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    store = SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    async def assert_migrated() -> None:
        created = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_migrated_labels",
                labels={"owner": "org_123"},
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        loaded = await store.load(created.id)
        assert loaded is not None
        assert loaded.labels == {"owner": "org_123"}
        await _close(store)

    asyncio.run(assert_migrated())

    connection = sqlite3.connect(db_path)
    try:
        label_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_session_labels'"
        ).fetchone()
        watcher_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_event_watcher_state'"
        ).fetchone()
        knowledge_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'cayu_knowledge_entries'"
        ).fetchone()
        knowledge_fts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_knowledge_chunks_fts'"
        ).fetchone()
        transcript_fts = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'cayu_transcript_messages_fts'"
        ).fetchone()
        revisions = connection.execute(
            "SELECT revision, compatible_from FROM cayu_schema_migrations ORDER BY revision"
        ).fetchall()
        task_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cayu_tasks)").fetchall()
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()

    assert label_table is not None
    assert watcher_table is not None
    assert knowledge_table is not None
    assert knowledge_fts is not None
    assert transcript_fts is not None
    assert {
        "worker_id",
        "lease_expires_at",
        "status_reason",
        "status_payload_json",
        "invocation_json",
        "retry_series_json",
    }.issubset(task_columns)
    # The explicit catalog guards compatibility-floor regressions as new
    # additive and breaking revisions are appended.
    assert revisions == [(rev.revision, rev.compatible_from) for rev in schema_migrations.REVISIONS]
    assert revisions == [
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 8),
        (9, 9),
        (10, 10),
        (11, 10),
        (12, 10),
        (13, 10),
        (14, 10),
        (15, 10),
        (16, 10),
        (17, 17),
        (18, 18),
        (19, 19),
        (20, 19),
        (21, 21),
        (22, 22),
        (23, 23),
        (24, 23),
        (25, 25),
        (26, 26),
        (27, 26),
        (28, 28),
        (29, 28),
        (30, 28),
        (31, 31),
        (32, 31),
        (33, 31),
        (34, 34),
        (35, 35),
        (36, 36),
        (37, 37),
        (38, 37),
        (39, 39),
        (40, 40),
        (41, 41),
        (42, 42),
        (43, 43),
        (44, 44),
        (45, 45),
        (46, 46),
        (47, 47),
        (48, 48),
        (49, 49),
        (50, 50),
        (51, 50),
        (52, 52),
        (53, 52),
        (54, 54),
    ]
    assert version == schema_migrations.LATEST_REVISION


def test_sqlite_revision_forty_one_rejects_populated_knowledge_receipt_database(
    tmp_path,
    monkeypatch,
) -> None:
    # This test intentionally boots historical revision-40/41 binaries. The
    # This historical test narrows the requirement to its revision-40 boundary.
    monkeypatch.setattr(sqlite_storage, "_SQLITE_SESSION_MIN_REQUIRED_REVISION", 40)
    db_path = tmp_path / "pre-knowledge-access-snapshot.sqlite"
    revisions = schema_migrations.REVISIONS
    schema_migrations.REVISIONS = tuple(
        revision for revision in revisions if revision.revision <= 40
    )
    try:
        store = SQLiteSessionStore(db_path)
        asyncio.run(_close(store))
    finally:
        schema_migrations.REVISIONS = revisions

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            INSERT INTO cayu_knowledge_publication_receipts (
                operation_id,
                entry_id,
                request_sha256,
                entry_created_at,
                entry_updated_at,
                committed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "op_existing",
                "entry_existing",
                "a" * 64,
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    revisions = schema_migrations.REVISIONS
    schema_migrations.REVISIONS = tuple(
        revision for revision in revisions if revision.revision <= 41
    )
    try:
        with pytest.raises(
            schema_migrations.SchemaTooOld,
            match="cannot infer one for existing receipts",
        ):
            SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)
    finally:
        schema_migrations.REVISIONS = revisions

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (40,)
        assert connection.execute(
            "SELECT MAX(revision) FROM cayu_schema_migrations"
        ).fetchone() == (40,)
        assert "access_snapshot_json" not in {
            row[1]
            for row in connection.execute("PRAGMA table_info(cayu_knowledge_publication_receipts)")
        }
    finally:
        connection.close()


def test_sqlite_migrate_recovers_from_a_crashed_partial_revision(tmp_path):
    # Simulate a crash that applied revision 4's ADD COLUMN steps but died before
    # recording the revision (the exact wedge the atomic/idempotent hardening
    # closes): the recorded revision is still 1 but cayu_tasks already has the
    # revision-4 columns. A re-run must not fail with "duplicate column name".
    db_path = tmp_path / "sessions.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(sqlite_support._BASELINE_DDL)
        connection.execute(sqlite_support._MIGRATIONS_TABLE_DDL)
        connection.execute(
            "INSERT INTO cayu_schema_migrations "
            "(revision, kind, compatible_from, checksum, applied_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, str(schema_migrations.RevisionKind.BREAKING), 1, None, "2026-01-01T00:00:00+00:00"),
        )
        # Partial revision-4 application: columns added, revision not yet recorded.
        connection.execute("ALTER TABLE cayu_tasks ADD COLUMN worker_id TEXT")
        connection.execute("ALTER TABLE cayu_tasks ADD COLUMN lease_expires_at TEXT")
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    # Must not raise; the idempotent ADD COLUMN skips the already-present columns.
    store = SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    async def _use() -> None:
        created = await store.create(
            RunRequest(agent_name="assistant", messages=[Message.text("user", "hi")]),
            identity=_identity(),
        )
        assert await store.load(created.id) is not None
        await _close(store)

    asyncio.run(_use())

    connection = sqlite3.connect(db_path)
    try:
        revisions = connection.execute(
            "SELECT revision FROM cayu_schema_migrations ORDER BY revision"
        ).fetchall()
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()

    assert [row[0] for row in revisions] == [rev.revision for rev in schema_migrations.REVISIONS]
    assert version == schema_migrations.LATEST_REVISION


def test_sqlite_migrate_revision_is_atomic_on_failure(tmp_path):
    # If a revision's data hook raises, the whole revision rolls back: no columns,
    # no recorded revision, user_version unchanged (crash cannot wedge migrate).
    db_path = tmp_path / "sessions.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(sqlite_support._BASELINE_DDL)
        connection.execute(sqlite_support._MIGRATIONS_TABLE_DDL)
        connection.execute(
            "INSERT INTO cayu_schema_migrations "
            "(revision, kind, compatible_from, checksum, applied_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, str(schema_migrations.RevisionKind.BREAKING), 1, None, "2026-01-01T00:00:00+00:00"),
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()

    def _boom(_conn: sqlite3.Connection) -> None:
        raise RuntimeError("simulated crash during revision 4")

    connection = sqlite_support.connect(db_path)
    try:
        original = dict(sqlite_support._MIGRATION_HOOKS)
        sqlite_support._MIGRATION_HOOKS[4] = _boom
        try:
            with pytest.raises(RuntimeError, match="simulated crash"):
                sqlite_support._apply_pending(
                    connection, sqlite_support.read_schema_state(connection)
                )
        finally:
            sqlite_support._MIGRATION_HOOKS.clear()
            sqlite_support._MIGRATION_HOOKS.update(original)

        # Revision 4's ADD COLUMN was rolled back atomically with the failed hook.
        task_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cayu_tasks)").fetchall()
        }
        assert "worker_id" not in task_columns
        # Revisions 2 and 3 (which precede 4 and have no hook) committed cleanly.
        recorded = {
            row[0]
            for row in connection.execute("SELECT revision FROM cayu_schema_migrations").fetchall()
        }
        assert recorded == {1, 2, 3}
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    finally:
        connection.close()


def test_sqlite_session_store_filters_session_label_selectors(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")

    async def assert_selectors() -> None:
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_sqlite_selector_invoice",
                labels={"owner": "org_123", "project": "ap_q2", "workflow": "invoice"},
                messages=[Message.text("user", "invoice")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="builder",
                session_id="sess_sqlite_selector_research",
                labels={"owner": "org_123", "project": "research"},
                messages=[Message.text("user", "research")],
            ),
            identity=_identity(),
        )
        await store.create(
            RunRequest(
                agent_name="reviewer",
                session_id="sess_sqlite_selector_unowned",
                labels={"project": "ap_q2"},
                messages=[Message.text("user", "review")],
            ),
            identity=_identity(),
        )

        exists = (
            await store.list_sessions(
                SessionQuery(
                    label_selectors=[{"key": "workflow", "operator": "exists"}],
                    order_by="created_at_asc",
                )
            )
        ).sessions
        in_selector = (
            await store.list_sessions(
                SessionQuery(
                    label_selectors=[
                        {"key": "project", "operator": "in", "values": ["ap_q2", "research"]}
                    ],
                    order_by="created_at_asc",
                )
            )
        ).sessions
        not_in = (
            await store.list_sessions(
                SessionQuery(
                    labels={"owner": "org_123"},
                    label_selectors=[
                        {"key": "project", "operator": "not_in", "values": ["research"]}
                    ],
                    order_by="created_at_asc",
                )
            )
        ).sessions
        not_exists = (
            await store.list_sessions(
                SessionQuery(
                    label_selectors=[{"key": "owner", "operator": "not_exists"}],
                    order_by="created_at_asc",
                )
            )
        ).sessions

        assert [session.id for session in exists] == ["sess_sqlite_selector_invoice"]
        assert [session.id for session in in_selector] == [
            "sess_sqlite_selector_invoice",
            "sess_sqlite_selector_research",
            "sess_sqlite_selector_unowned",
        ]
        assert [session.id for session in not_in] == ["sess_sqlite_selector_invoice"]
        assert [session.id for session in not_exists] == ["sess_sqlite_selector_unowned"]
        await _close(store)

    asyncio.run(assert_selectors())


def test_cayu_app_can_use_sqlite_session_store(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    app = CayuApp(session_store=store)
    app.register_provider(
        FakeProvider(
            [
                ModelStreamEvent.text_delta("hello"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run_app() -> None:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_runtime_sqlite",
                    messages=[Message.text("user", "hi")],
                )
            )
        ]
        persisted_events = await store.load_events("sess_runtime_sqlite")
        private_events = await private_events_for_public_events(store, events)
        session = await store.load("sess_runtime_sqlite")

        assert [event.type for event in events] == [
            EventType.INTERACTION_STARTED,
            EventType.SESSION_STARTED,
            EventType.TOOL_EXPOSURE_RECORDED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.INTERACTION_COMPLETED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]
        assert persisted_events == private_events
        assert session is not None
        assert session.status == SessionStatus.COMPLETED
        assert session.provider_name == "fake"
        assert session.model == "fake-model"
        assert session.runtime_name == "cayu"
        await _close(store)

    asyncio.run(run_app())


def test_cayu_app_can_resume_with_sqlite_session_store(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.text_delta("first answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.text_delta("second answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def run_app() -> None:
        await _collect_app_events(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_resume_sqlite",
                    messages=[Message.text("user", "first request")],
                )
            )
        )
        resume_events = await _collect_app_events(
            app.resume(
                ResumeRequest(
                    session_id="sess_resume_sqlite",
                    messages=[Message.text("user", "second request")],
                )
            )
        )
        transcript = await store.load_transcript("sess_resume_sqlite")
        persisted_events = await store.load_events("sess_resume_sqlite")
        session = await store.load("sess_resume_sqlite")

        assert [event.type for event in resume_events] == [
            EventType.INTERACTION_STARTED,
            EventType.SESSION_RESUMED,
            EventType.TOOL_EXPOSURE_RECORDED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.INTERACTION_COMPLETED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]
        assert [message.content[0].text for message in provider.requests[1].messages] == [
            "first request",
            "first answer",
            "second request",
        ]
        assert [message.content[0].text for message in transcript] == [
            "first request",
            "first answer",
            "second request",
            "second answer",
        ]
        assert [event.type for event in persisted_events] == [
            EventType.INTERACTION_STARTED,
            EventType.SESSION_STARTED,
            EventType.TOOL_EXPOSURE_RECORDED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.INTERACTION_COMPLETED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_COMPLETED,
            EventType.SESSION_EXECUTION_PROFILE_DECIDED,
            EventType.INTERACTION_STARTED,
            EventType.SESSION_RESUMED,
            EventType.TOOL_EXPOSURE_RECORDED,
            EventType.REQUEST_FOOTPRINT_RECORDED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.INTERACTION_COMPLETED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]
        assert session is not None
        assert session.status == SessionStatus.COMPLETED
        await _close(store)

    asyncio.run(run_app())


def test_sqlite_session_store_rejects_incompatibly_new_database(tmp_path):
    # A database migrated past a breaking revision this binary doesn't understand
    # (compatible_from floor above the app's latest) fails fast (ADR 0001 Decision 7).
    db_path = tmp_path / "newer.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "CREATE TABLE cayu_schema_migrations ("
            "revision INTEGER PRIMARY KEY, kind TEXT NOT NULL, "
            "compatible_from INTEGER NOT NULL, checksum TEXT, applied_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO cayu_schema_migrations VALUES "
            "(999, 'breaking', 999, NULL, '2026-01-01T00:00:00+00:00')"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(schema_migrations.SchemaTooNew):
        SQLiteSessionStore(db_path)


def test_sqlite_session_store_reads_through_dedicated_read_only_connection(tmp_path):
    db_path = tmp_path / "read_only.sqlite"
    store = SQLiteSessionStore(db_path)

    # File-backed stores query through a dedicated read-only connection so
    # reads run off the event loop without contending with the writer.
    assert store._read_connection is not store._connection
    assert store._read_lock is not store._lock
    with pytest.raises(sqlite3.OperationalError):
        store._read_connection.execute(
            "INSERT INTO cayu_session_labels (session_id, key, value) VALUES ('x', 'k', 'v')"
        )

    async def run() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_read_only",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.append_event(
            "sess_read_only",
            Event(
                type=EventType.SESSION_STARTED,
                session_id="sess_read_only",
                agent_name="assistant",
                payload={"step": 1},
            ),
        )
        await store.checkpoint("sess_read_only", {"step": 1})

        # Writes on the writer connection are immediately visible to reads on
        # the read-only connection.
        session = await store.load("sess_read_only")
        assert session is not None
        assert session.agent_name == "assistant"
        events = await store.load_events("sess_read_only")
        assert [event.type for event in events] == [EventType.SESSION_STARTED]
        assert await store.load_checkpoint("sess_read_only") == {"step": 1}
        await _close(store)

    asyncio.run(run())


def test_sqlite_session_store_in_memory_shares_single_connection():
    store = SQLiteSessionStore(":memory:")

    # An in-memory database is private to its connection, so the read path
    # falls back to the writer connection and its lock.
    assert store._read_connection is store._connection
    assert store._read_lock is store._lock

    async def run() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_memory_shared",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        session = await store.load("sess_memory_shared")
        assert session is not None
        await _close(store)

    asyncio.run(run())


def test_sqlite_off_thread_writer_retains_connection_ownership_during_cancellation(
    tmp_path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "writer-cancellation.sqlite")
    worker_started = threading.Event()
    release_worker = threading.Event()
    follower_started = threading.Event()

    def blocked_write(connection: sqlite3.Connection) -> str:
        worker_started.set()
        if not release_worker.wait(timeout=5):
            raise TimeoutError("test did not release the SQLite writer")
        connection.execute("SELECT 1")
        return "first"

    def follower_write(connection: sqlite3.Connection) -> str:
        follower_started.set()
        connection.execute("SELECT 1")
        return "second"

    async def run() -> None:
        owner = asyncio.create_task(store._run_write(blocked_write))
        await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=2)

        owner.cancel("first cancellation")
        await asyncio.sleep(0)
        assert not owner.done()
        owner.cancel("second cancellation")
        await asyncio.sleep(0)
        follower = asyncio.create_task(store._run_write(follower_write))
        close_task = asyncio.create_task(store.close())
        await asyncio.sleep(0)

        assert owner.cancelling() == 2
        assert store._lock.locked()
        assert not owner.done()
        assert not follower_started.is_set()
        assert not follower.done()
        assert not close_task.done()

        release_worker.set()
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await owner
        assert cancellation.value.args == ("first cancellation",)
        assert owner.cancelled()
        assert await follower == "second"
        await close_task

    asyncio.run(run())


def test_sqlite_pending_cancellation_retains_off_thread_connection_ownership(
    tmp_path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "pending-cancellation.sqlite")
    worker_started = threading.Event()
    release_worker = threading.Event()
    follower_started = threading.Event()

    def blocked_write(connection: sqlite3.Connection) -> None:
        worker_started.set()
        if not release_worker.wait(timeout=5):
            raise TimeoutError("test did not release the pending-cancellation worker")
        connection.execute("SELECT 1")

    def follower_write(connection: sqlite3.Connection) -> None:
        follower_started.set()
        connection.execute("SELECT 2")

    async def run() -> None:
        async def start_with_pending_cancellation() -> None:
            current_task = asyncio.current_task()
            assert current_task is not None
            current_task.cancel("pending before SQLite ownership")
            await store._run_write(blocked_write)

        owner = asyncio.create_task(start_with_pending_cancellation())
        await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=2)
        follower = asyncio.create_task(store._run_write(follower_write))
        close_task = asyncio.create_task(store.close())
        await asyncio.sleep(0)

        assert store._lock.locked()
        assert not owner.done()
        assert not follower_started.is_set()
        assert not follower.done()
        assert not close_task.done()

        release_worker.set()
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await owner
        assert cancellation.value.args == ("pending before SQLite ownership",)
        await follower
        await close_task

    asyncio.run(run())


def test_sqlite_task_sweep_cannot_end_off_thread_connection_ownership(
    tmp_path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "task-sweep-cancellation.sqlite")
    worker_started = threading.Event()
    release_worker = threading.Event()
    follower_started = threading.Event()

    def blocked_write(connection: sqlite3.Connection) -> None:
        worker_started.set()
        if not release_worker.wait(timeout=5):
            raise TimeoutError("test did not release the task-sweep worker")
        connection.execute("SELECT 1")

    def follower_write(connection: sqlite3.Connection) -> None:
        follower_started.set()
        connection.execute("SELECT 2")

    async def run() -> None:
        owner = asyncio.create_task(store._run_write(blocked_write))
        await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=2)

        current_task = asyncio.current_task()
        assert current_task is not None
        child_tasks = asyncio.all_tasks() - {current_task, owner}
        owner.cancel("shutdown sweep")
        for child_task in child_tasks:
            child_task.cancel("shutdown sweep")

        follower = asyncio.create_task(store._run_write(follower_write))
        close_task = asyncio.create_task(store.close())
        await asyncio.sleep(0)

        assert store._lock.locked()
        assert not owner.done()
        assert not follower_started.is_set()
        assert not follower.done()
        assert not close_task.done()

        release_worker.set()
        with pytest.raises(asyncio.CancelledError, match="shutdown sweep"):
            await owner
        await follower
        await close_task

    asyncio.run(run())


def test_sqlite_off_thread_reader_retains_connection_ownership_during_cancellation(
    tmp_path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "reader-cancellation.sqlite")
    worker_started = threading.Event()
    release_worker = threading.Event()
    follower_started = threading.Event()

    def blocked_read(connection: sqlite3.Connection) -> int:
        worker_started.set()
        if not release_worker.wait(timeout=5):
            raise TimeoutError("test did not release the SQLite reader")
        return connection.execute("SELECT 1").fetchone()[0]

    def follower_read(connection: sqlite3.Connection) -> int:
        follower_started.set()
        return connection.execute("SELECT 2").fetchone()[0]

    async def run() -> None:
        owner = asyncio.create_task(store._run_read(blocked_read))
        await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=2)

        # The file-backed writer uses a different physical connection and must
        # remain independent of the occupied read connection.
        writer_value = await store._run_write(
            lambda connection: connection.execute("SELECT 3").fetchone()[0]
        )
        assert writer_value == 3

        owner.cancel()
        follower = asyncio.create_task(store._run_read(follower_read))
        await asyncio.sleep(0)
        close_task = asyncio.create_task(store.close())
        await asyncio.sleep(0)

        assert store._read_lock.locked()
        assert owner.done()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert not follower_started.is_set()
        assert not follower.done()
        assert not close_task.done()

        release_worker.set()
        assert await follower == 2
        await close_task

    asyncio.run(run())


def test_sqlite_off_thread_worker_failure_and_cancellation_remain_observable(
    tmp_path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "worker-failure.sqlite")

    class WorkerFailure(RuntimeError):
        pass

    ordinary_failure = WorkerFailure("ordinary worker failure")

    def fail_ordinary(_connection: sqlite3.Connection) -> None:
        raise ordinary_failure

    def cancel_worker(_connection: sqlite3.Connection) -> None:
        raise asyncio.CancelledError("worker cancellation")

    worker_started = threading.Event()
    release_worker = threading.Event()
    cancelled_failure = WorkerFailure("worker failed after cancellation")
    context_marker = contextvars.ContextVar("sqlite_worker_context")

    def fail_after_cancellation(_connection: sqlite3.Connection) -> None:
        worker_started.set()
        if not release_worker.wait(timeout=5):
            raise TimeoutError("test did not release the failing SQLite worker")
        raise cancelled_failure

    async def run() -> None:
        context_marker.set("caller context")
        observed_context = await store._run_write(lambda _connection: context_marker.get("missing"))
        assert observed_context == "caller context"

        with pytest.raises(WorkerFailure) as raised:
            await store._run_write(fail_ordinary)
        assert raised.value is ordinary_failure

        with pytest.raises(asyncio.CancelledError, match="worker cancellation"):
            await store._run_write(cancel_worker)
        current_task = asyncio.current_task()
        assert current_task is not None
        assert current_task.cancelling() == 0

        owner = asyncio.create_task(store._run_write(fail_after_cancellation))
        await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=2)
        owner.cancel()
        release_worker.set()
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await owner
        assert cancellation.value.__cause__ is cancelled_failure
        assert any(
            "SQLite worker failed while caller cancellation was pending" in note
            for note in cancellation.value.__notes__
        )
        assert owner.cancelled()
        await store.close()

    asyncio.run(run())


def test_sqlite_in_memory_read_cancellation_serializes_shared_writer_connection() -> None:
    store = SQLiteSessionStore(":memory:")
    worker_started = threading.Event()
    release_worker = threading.Event()
    writer_started = threading.Event()

    def blocked_read(connection: sqlite3.Connection) -> int:
        worker_started.set()
        if not release_worker.wait(timeout=5):
            raise TimeoutError("test did not release the in-memory SQLite reader")
        return connection.execute("SELECT 1").fetchone()[0]

    def write(connection: sqlite3.Connection) -> None:
        writer_started.set()
        connection.execute("CREATE TABLE cancellation_serialization (value INTEGER)")

    async def run() -> None:
        owner = asyncio.create_task(store._run_read(blocked_read))
        await asyncio.wait_for(asyncio.to_thread(worker_started.wait), timeout=2)
        owner.cancel()
        writer = asyncio.create_task(store._run_write(write))
        await asyncio.sleep(0)

        assert store._read_lock is store._lock
        assert owner.done()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert not writer_started.is_set()
        assert not writer.done()

        release_worker.set()
        await writer
        await store.close()

    asyncio.run(run())


def test_sqlite_connect_rejects_read_only_in_memory_database():
    from pathlib import Path

    with pytest.raises(ValueError, match="file-backed"):
        sqlite_support.connect(Path(":memory:"), read_only=True)


def _make_event(session_id: str, *, seq: int, timestamp) -> Event:
    return Event(
        type=EventType.TOOL_CALL_COMPLETED,
        session_id=session_id,
        agent_name="assistant",
        tool_name="read_file",
        timestamp=timestamp,
        payload={"n": seq},
    )


def test_sqlite_events_reconstructed_from_columns_without_event_json(tmp_path):
    from cayu.runtime import EventQuery

    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_events",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        original = _make_event(session.id, seq=1, timestamp=datetime(2026, 1, 1, tzinfo=UTC))
        await store.append_event(session.id, original)

        loaded = await store.load_events(session.id)
        assert loaded == [original]

        records = await store.query_events(EventQuery(session_id=session.id))
        assert [record.event for record in records] == [original]

        summary = await store.summarize_events(session.id)
        assert summary.latest_event is not None
        assert summary.latest_event.event == original
        await _close(store)

    asyncio.run(run())

    # The redundant event_json column must be absent from a freshly created DB.
    connection = sqlite3.connect(db_path)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cayu_events)").fetchall()
        }
    finally:
        connection.close()
    assert "event_json" not in columns
    assert "payload_json" in columns


def test_sqlite_prune_events_bounds_growth(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_prune",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        old = Event(
            id="evt_pruned_interaction",
            type=EventType.INTERACTION_STARTED,
            session_id=session.id,
            interaction_id="interaction-pruned",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        new = _make_event(session.id, seq=2, timestamp=datetime(2026, 3, 1, tzinfo=UTC))
        await store.append_events(session.id, [old, new])
        latest = await store.query_latest_interaction_events(session.id, limit=10)
        assert [record.event.id for record in latest] == [old.id]

        # Retention cannot erase an event whose required side effects are still
        # pending, failed, leased, or dead-lettered.
        assert (
            await store.prune_events(before=datetime(2026, 2, 1, tzinfo=UTC), session_id=session.id)
            == 0
        )
        old_claim = await store.claim_persisted_event_side_effect(
            session_id=session.id,
            event_id=old.id,
        )
        assert old_claim is not None
        await store.mark_persisted_event_side_effect_delivered(old_claim)
        # The latest open lifecycle record remains recovery evidence even after
        # its external handoff has completed.
        assert (
            await store.prune_events(
                before=datetime(2026, 2, 1, tzinfo=UTC),
                session_id=session.id,
            )
            == 0
        )
        closed = Event(
            id="evt_pruned_interaction_completed",
            type=EventType.INTERACTION_COMPLETED,
            session_id=session.id,
            interaction_id="interaction-pruned",
            timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        )
        await store.append_event(session.id, closed)
        closed_claim = await store.claim_persisted_event_side_effect(
            session_id=session.id,
            event_id=closed.id,
        )
        assert closed_claim is not None
        await store.mark_persisted_event_side_effect_delivered(closed_claim)
        deleted = await store.prune_events(
            before=datetime(2026, 2, 1, tzinfo=UTC), session_id=session.id
        )
        assert deleted == 2
        remaining = await store.load_events(session.id)
        assert [event.payload for event in remaining] == [{"n": 2}]
        assert await store.query_latest_interaction_events(session.id, limit=10) == []

        # Unknown session is rejected; wrong-type cutoff is rejected.
        with pytest.raises(KeyError):
            await store.prune_events(before=datetime(2026, 2, 1, tzinfo=UTC), session_id="missing")
        with pytest.raises(TypeError):
            await store.prune_events(before="2026-02-01")  # type: ignore[arg-type]

        # A store-wide prune (no session_id) drops the rest.
        new_claim = await store.claim_persisted_event_side_effect(
            session_id=session.id,
            event_id=new.id,
        )
        assert new_claim is not None
        await store.mark_persisted_event_side_effect_delivered(new_claim)
        assert await store.prune_events(before=datetime(2026, 4, 1, tzinfo=UTC)) == 1
        assert await store.load_events(session.id) == []
        await _close(store)

    asyncio.run(run())


def test_sqlite_interaction_transition_replay_survives_event_retention(tmp_path):
    db_path = tmp_path / "sessions.sqlite"

    async def run() -> None:
        store = SQLiteSessionStore(db_path)
        session_id = "sess_transition_receipt_retention"
        interaction_id = "interaction-transition-receipt-retention"
        started = Event(
            id="evt_transition_receipt_retention_started",
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id=interaction_id,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        failed = Event(
            id="evt_transition_receipt_retention_failed",
            type=EventType.INTERACTION_FAILED,
            session_id=session_id,
            interaction_id=interaction_id,
            timestamp=datetime(2026, 1, 2, tzinfo=UTC),
        )
        try:
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "start")],
                ),
                identity=_identity(),
                interaction_started_event=started,
                interaction_source_messages=[Message.text("user", "start")],
            )
            published = await store.publish_interaction_transition(
                session_id,
                event=failed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.FAILED,
            )
            assert published.status_changed is True

            for event in (started, failed):
                claim = await store.claim_persisted_event_side_effect(
                    session_id=session_id,
                    event_id=event.id,
                )
                assert claim is not None
                await store.mark_persisted_event_side_effect_delivered(claim)
            assert (
                await store.prune_events(
                    before=datetime(2026, 2, 1, tzinfo=UTC),
                    session_id=session_id,
                )
                == 2
            )
            assert await store.load_events(session_id) == []
            await _close(store)

            store = SQLiteSessionStore(db_path)
            replayed = await store.publish_interaction_transition(
                session_id,
                event=failed,
                from_statuses={SessionStatus.RUNNING},
                to_status=SessionStatus.FAILED,
            )
            assert replayed.replayed is True
            assert replayed.status_changed is True
            assert replayed.event == failed
            assert replayed.session.status is SessionStatus.FAILED
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_maintenance_preserves_active_model_completion_stage(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_active_stage_retention",
                messages=[],
            ),
            identity=_identity(),
        )
        messages = [Message.text("user", f"m{index}") for index in range(3)]
        await store.append_transcript_messages(session.id, messages)
        event = Event(
            type=EventType.MODEL_STARTED,
            session_id=session.id,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            payload={"step": 1},
        )
        await store.append_event(session.id, event)
        claim = await store.claim_persisted_event_side_effect(
            session_id=session.id,
            event_id=event.id,
        )
        assert claim is not None
        await store.mark_persisted_event_side_effect_delivered(claim)
        running = await store.transition_status(
            session.id,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        await store.prepare_model_completion_stage(
            session.id,
            request=ModelCompletionStageRequest(
                stage_id="retention-stage",
                logical_step_id="retention-step",
                dispatch_ordinal=0,
                intent={"request_fingerprint": "retention"},
            ),
            expected_statuses={SessionStatus.RUNNING},
            expected_run_epoch=running.run_epoch,
            expected_transcript_cursor=len(messages),
        )

        assert (
            await store.prune_events(
                before=datetime(2026, 2, 1, tzinfo=UTC),
                session_id=session.id,
            )
            == 0
        )
        assert await store.compact_transcript(session.id, keep_last=0) == 0
        assert await store.load_events(session.id) == [event]
        assert await store.load_transcript(session.id) == messages
        await _close(store)

    asyncio.run(run())


def test_sqlite_maintenance_preserves_pending_tool_round(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_pending_round_retention",
                messages=[],
            ),
            identity=_identity(),
        )
        messages = [Message.text("user", f"m{index}") for index in range(3)]
        await store.append_transcript_messages(session.id, messages)
        round_id = "retention-round"
        tool_call_id = "retention-call"
        await store.checkpoint(
            session.id,
            {
                "pending_tool_round": {
                    "tool_round_id": round_id,
                    "agent_name": "assistant",
                    "tool_calls": [
                        {
                            "tool_call_id": tool_call_id,
                            "tool_name": "read_file",
                            "arguments": {},
                        }
                    ],
                }
            },
        )
        event = Event(
            type=EventType.TOOL_CALL_COMPLETED,
            session_id=session.id,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            tool_name="read_file",
            payload={
                "tool_call_id": tool_call_id,
                "tool_round_id": round_id,
                "result": {
                    "content": "done",
                    "structured": None,
                    "artifacts": [],
                    "is_error": False,
                },
            },
        )
        await store.append_event(session.id, event)
        claim = await store.claim_persisted_event_side_effect(
            session_id=session.id,
            event_id=event.id,
        )
        assert claim is not None
        await store.mark_persisted_event_side_effect_delivered(claim)

        assert (
            await store.prune_events(
                before=datetime(2026, 2, 1, tzinfo=UTC),
                session_id=session.id,
            )
            == 0
        )
        assert await store.compact_transcript(session.id, keep_last=0) == 0
        assert await store.load_events(session.id) == [event]
        assert await store.load_transcript(session.id) == messages
        await _close(store)

    asyncio.run(run())


def test_sqlite_revision_twenty_protects_handoffs_from_revision_nineteen_prune(
    tmp_path,
) -> None:
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def prepare() -> tuple[str, str]:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_rolling_prune",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        event = _make_event(
            session.id,
            seq=1,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        )
        await store.append_event(session.id, event)
        await _close(store)
        return session.id, event.id

    session_id, event_id = asyncio.run(prepare())

    connection = sqlite3.connect(db_path)
    try:
        # This is the revision-19 retention shape: it knows nothing about the
        # revision-20 handoff table. The database trigger must preserve the row.
        connection.execute(
            "DELETE FROM cayu_events WHERE timestamp < ?",
            ("2026-02-01T00:00:00+00:00",),
        )
        connection.commit()
        assert connection.execute(
            "SELECT event_id FROM cayu_events WHERE session_id = ?",
            (session_id,),
        ).fetchall() == [(event_id,)]
    finally:
        connection.close()

    reopened = SQLiteSessionStore(db_path)

    async def deliver() -> None:
        claim = await reopened.claim_persisted_event_side_effect(
            session_id=session_id,
            event_id=event_id,
        )
        assert claim is not None
        await reopened.mark_persisted_event_side_effect_delivered(claim)
        await _close(reopened)

    asyncio.run(deliver())

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            "DELETE FROM cayu_events WHERE timestamp < ?",
            ("2026-02-01T00:00:00+00:00",),
        )
        connection.commit()
        assert connection.execute("SELECT event_id FROM cayu_events").fetchall() == []
    finally:
        connection.close()


def test_sqlite_side_effect_deadlines_start_after_write_lock_acquisition(
    tmp_path,
    monkeypatch,
) -> None:
    class StoreClock(datetime):
        current = datetime(2026, 7, 16, 10, 0, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    db_path = tmp_path / "side-effect-deadlines.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_side_effect_deadlines",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        event = _make_event(session.id, seq=1, timestamp=StoreClock.current)
        await store.append_event(session.id, event)
        monkeypatch.setattr("cayu.storage.sqlite.datetime", StoreClock)
        begin_seen = threading.Event()
        store._connection.set_trace_callback(
            lambda statement: begin_seen.set() if statement == "BEGIN IMMEDIATE" else None
        )
        blocker = sqlite3.connect(db_path)

        try:
            blocker.execute("BEGIN IMMEDIATE")
            claim_task = asyncio.create_task(
                store.claim_persisted_event_side_effect(
                    session_id=session.id,
                    event_id=event.id,
                    lease_seconds=30,
                )
            )
            assert await asyncio.to_thread(begin_seen.wait, 1)
            StoreClock.current += timedelta(minutes=1)
            blocker.commit()
            claim = await claim_task
            assert claim is not None
            assert claim.lease_expires_at == StoreClock.current + timedelta(seconds=30)
            assert (
                await store.claim_persisted_event_side_effect(
                    session_id=session.id,
                    event_id=event.id,
                )
                is None
            )

            begin_seen.clear()
            blocker.execute("BEGIN IMMEDIATE")
            failure_task = asyncio.create_task(
                store.mark_persisted_event_side_effect_failed(
                    claim,
                    error="try later",
                    max_attempts=3,
                    retry_delay_seconds=30,
                )
            )
            assert await asyncio.to_thread(begin_seen.wait, 1)
            StoreClock.current += timedelta(minutes=1)
            blocker.commit()
            failed = await failure_task
            assert failed.updated_at == StoreClock.current
            assert failed.next_attempt_at == StoreClock.current + timedelta(seconds=30)
            assert (
                await store.claim_persisted_event_side_effect(
                    session_id=session.id,
                    event_id=event.id,
                )
                is None
            )
        finally:
            blocker.rollback()
            blocker.close()
            store._connection.set_trace_callback(None)
            await _close(store)

    asyncio.run(run())


def test_sqlite_handoff_protection_allows_session_delete_cascade(tmp_path) -> None:
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_handoff_delete",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        await store.append_event(
            session.id,
            _make_event(session.id, seq=1, timestamp=datetime(2026, 1, 1, tzinfo=UTC)),
        )
        await store.delete_session(session.id)
        await _close(store)

    asyncio.run(run())

    connection = sqlite3.connect(db_path)
    try:
        assert connection.execute("SELECT id FROM cayu_sessions").fetchall() == []
        assert connection.execute("SELECT event_id FROM cayu_events").fetchall() == []
        assert (
            connection.execute("SELECT event_id FROM cayu_persisted_event_side_effects").fetchall()
            == []
        )
    finally:
        connection.close()


def test_sqlite_compact_transcript_keeps_recent_messages(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact",
                messages=[Message.text("user", "hi")],
            ),
            identity=_identity(),
        )
        messages = [Message.text("user", f"m{index}") for index in range(5)]
        await store.append_transcript_messages(session.id, messages)

        deleted = await store.compact_transcript(session.id, keep_last=2)
        assert deleted == 3
        kept = await store.load_transcript(session.id)
        assert [message.content[0].text for message in kept] == ["m3", "m4"]
        page = await store.query_transcript(TranscriptQuery(session_id=session.id, limit=10))
        assert [record.index for record in page.records] == [3, 4]

        await store.append_transcript_messages(
            session.id,
            [Message.text("assistant", "after compaction")],
        )
        appended = await store.query_transcript(TranscriptQuery(session_id=session.id, limit=10))
        assert [record.index for record in appended.records] == [3, 4, 5]

        # keep_last larger than the transcript deletes nothing.
        assert await store.compact_transcript(session.id, keep_last=10) == 0

        with pytest.raises(ValueError):
            await store.compact_transcript(session.id, keep_last=-1)
        with pytest.raises(KeyError):
            await store.compact_transcript("missing", keep_last=1)
        await _close(store)

    asyncio.run(run())


def test_sqlite_partial_fork_uses_absolute_cursor_after_transcript_retention(
    tmp_path,
) -> None:
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")

    async def run() -> None:
        source = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_retained_fork_source",
                messages=[],
            ),
            identity=_identity(),
        )
        await store.append_transcript_messages(
            source.id,
            [Message.text("user", f"m{index}") for index in range(5)],
        )
        source = await store.update_status(source.id, SessionStatus.COMPLETED)
        assert await store.compact_transcript(source.id, keep_last=2) == 3

        await store.create_fork(
            source_session_id=source.id,
            fork=Session(
                id="sess_retained_fork_child",
                agent_name="assistant",
                provider_name="fake",
                model="fake-model",
                parent_session_id=source.id,
                invocation=fork_session_invocation(source),
                status=SessionStatus.COMPLETED,
            ),
            source_statuses={SessionStatus.COMPLETED},
            transcript_cursor=4,
            checkpoint_transform=None,
            expected_source_run_epoch=source.run_epoch,
        )

        transcript = await store.load_transcript("sess_retained_fork_child")
        assert [message.content[0].text for message in transcript] == ["m3"]

        with pytest.raises(ValueError, match="transcript_cursor is greater"):
            await store.create_fork(
                source_session_id=source.id,
                fork=Session(
                    id="sess_retained_fork_overflow",
                    agent_name="assistant",
                    provider_name="fake",
                    model="fake-model",
                    parent_session_id=source.id,
                    invocation=fork_session_invocation(source),
                    status=SessionStatus.COMPLETED,
                ),
                source_statuses={SessionStatus.COMPLETED},
                transcript_cursor=6,
                checkpoint_transform=None,
                expected_source_run_epoch=source.run_epoch,
            )
        await _close(store)

    asyncio.run(run())


def test_sqlite_transcript_cursor_remains_monotonic_after_retention(tmp_path):
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite")

    async def run() -> None:
        session = await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_compact_cursor",
                messages=[],
            ),
            identity=_identity(),
        )
        running = await store.transition_status(
            session.id,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        await store.append_transcript_messages(
            session.id,
            [Message.text("user", f"m{index}") for index in range(5)],
        )
        assert await store.compact_transcript(session.id, keep_last=2) == 3

        accepted = await store.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id=session.id,
                idempotency_key="after-retention",
                content="continue",
                delivery_mode=SessionMessageDeliveryMode.NEXT_TURN,
            )
        )
        assert accepted.message.accepted_transcript_cursor == 5

        delivered = await store.deliver_queued_session_messages(
            session.id,
            include_on_idle=False,
            delivery_id="delivery-after-retention",
            interaction_id="interaction-after-retention",
        )
        assert delivered.messages[0].delivered_transcript_cursor == 6
        page = await store.query_transcript(TranscriptQuery(session_id=session.id, limit=10))
        assert [record.index for record in page.records] == [3, 4, 5]

        await store.publish_checkpoint_and_events(
            session.id,
            checkpoint_transform=lambda _session, _checkpoint: {"cursor": 6},
            events=[],
            expected_statuses={SessionStatus.RUNNING},
            expected_run_epoch=running.run_epoch,
            expected_transcript_cursor=6,
        )
        await _close(store)

    asyncio.run(run())
