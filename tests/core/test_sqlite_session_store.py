from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import UTC, datetime

import pytest

from cayu import SQLiteSessionStore
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runtime import (
    CayuApp,
    ForkSessionRequest,
    ResumeRequest,
    RunRequest,
    Session,
    SessionIdentity,
    SessionQuery,
    SessionStatus,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import migrations as schema_migrations


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


async def _collect_app_events(events) -> list[Event]:
    return [event async for event in events]


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


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
        assert checkpoint == {"context_compaction": {}}
        children = (
            await reopened.list_sessions(SessionQuery(parent_session_id="sess_sqlite_fork_source"))
        ).sessions
        assert [session.id for session in children] == ["sess_sqlite_fork_child"]
        events = await reopened.load_events("sess_sqlite_fork_child")
        assert [event.type for event in events] == [EventType.SESSION_FORKED]
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


def test_sqlite_session_store_persists_updated_session_model_across_reopen(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(db_path)

    async def create_and_update() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_reopen_model",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="initial-model"),
        )
        await store.update_model("sess_reopen_model", "upgraded-model")
        await _close(store)

    asyncio.run(create_and_update())

    reopened = SQLiteSessionStore(db_path)

    async def assert_reopened() -> None:
        loaded = await reopened.load("sess_reopen_model")
        assert loaded is not None
        assert loaded.model == "upgraded-model"
        await _close(reopened)

    asyncio.run(assert_reopened())


def test_sqlite_session_store_validate_mode_fails_fast_on_uninitialized(tmp_path):
    # validate-at-startup (ADR 0001 Q4): a store opened in validate mode against an
    # empty database fails fast instead of silently creating the schema.
    db_path = tmp_path / "sessions.sqlite"
    with pytest.raises(schema_migrations.SchemaUninitialized):
        SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.VALIDATE)


def test_sqlite_session_store_revision_thirteen_requires_run_fencing_migration(tmp_path):
    # Revision 14 adds the activity timestamp and run epoch required for safe
    # stalled-session recovery, so revision 13 databases must migrate before use.
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

    with pytest.raises(schema_migrations.SchemaTooOld, match="requires >= 15"):
        SQLiteSessionStore(db_path)

    reopened = SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    async def assert_compatible() -> None:
        loaded = await reopened.load("sess_sqlite_rev12")
        assert loaded is not None
        await _close(reopened)

    asyncio.run(assert_compatible())


def test_sqlite_session_store_revision_fourteen_requires_cascade_index_migration(tmp_path):
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

    with pytest.raises(schema_migrations.SchemaTooOld, match="requires >= 15"):
        SQLiteSessionStore(db_path)

    reopened = SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    async def assert_compatible() -> None:
        loaded = await reopened.load("sess_sqlite_rev14")
        assert loaded is not None
        await _close(reopened)

    asyncio.run(assert_compatible())

    connection = sqlite3.connect(db_path)
    try:
        index = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' "
            "AND name = 'idx_cayu_checkpoints_pending_interruption_cascade'"
        ).fetchone()
    finally:
        connection.close()
    assert index is not None


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
    assert {
        "worker_id",
        "lease_expires_at",
        "status_reason",
        "status_payload_json",
    }.issubset(task_columns)
    # Revisions 2-7 and 11-16 are additive and keep the prior compatibility floor.
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
    ]
    assert version == schema_migrations.LATEST_REVISION


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
        session = await store.load("sess_runtime_sqlite")

        assert [event.type for event in events] == [
            EventType.SESSION_STARTED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_COMPLETED,
        ]
        assert persisted_events == events
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
            EventType.SESSION_RESUMED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
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
            EventType.SESSION_STARTED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
            EventType.TURN_COMPLETED,
            EventType.SESSION_COMPLETED,
            EventType.SESSION_RESUMED,
            EventType.MODEL_STARTED,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_COMPLETED,
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


def test_sqlite_migrate_drops_legacy_event_json_column(tmp_path):
    db_path = tmp_path / "sessions.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        # Recreate the pre-revision-9 cayu_events shape (with event_json), plus
        # the minimal sessions table the FK/reads need, recorded at revision 8.
        connection.executescript(sqlite_support._BASELINE_DDL)
        connection.execute("DROP TABLE cayu_events")
        connection.execute(
            """
            CREATE TABLE cayu_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES cayu_sessions(id) ON DELETE CASCADE,
                event_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                agent_name TEXT,
                environment_name TEXT,
                workflow_name TEXT,
                tool_name TEXT,
                payload_json TEXT NOT NULL,
                event_json TEXT NOT NULL,
                UNIQUE(session_id, event_id)
            )
            """
        )
        connection.execute(sqlite_support._MIGRATIONS_TABLE_DDL)
        for rev in schema_migrations.REVISIONS:
            if rev.revision > 8:
                continue
            connection.execute(
                "INSERT INTO cayu_schema_migrations "
                "(revision, kind, compatible_from, checksum, applied_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    rev.revision,
                    str(rev.kind),
                    rev.compatible_from,
                    None,
                    "2026-01-01T00:00:00+00:00",
                ),
            )
        connection.execute("PRAGMA user_version = 8")
        # A pre-existing event row (with the redundant event_json populated).
        connection.execute(
            """
            INSERT INTO cayu_sessions (
                id, agent_name, provider_name, model, parent_session_id,
                causal_budget_id, runtime_name, runtime_version, environment_name,
                status, created_at, updated_at, last_activity_at, run_epoch, metadata_json
            ) VALUES ('sess_legacy', 'assistant', 'fake', 'fake-model', NULL,
                'sess_legacy', 'cayu', NULL, NULL, 'pending',
                '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00', 0, '{}')
            """
        )
        connection.execute(
            """
            INSERT INTO cayu_events (
                session_id, event_id, event_type, timestamp, agent_name,
                environment_name, workflow_name, tool_name, payload_json, event_json
            ) VALUES ('sess_legacy', 'evt_1', 'tool.call.completed',
                '2026-01-01T00:00:00+00:00', 'assistant', NULL, NULL, 'read_file',
                '{"n":1}',
                '{"type":"tool.call.completed","session_id":"sess_legacy","id":"evt_1","timestamp":"2026-01-01T00:00:00+00:00","agent_name":"assistant","environment_name":null,"workflow_name":null,"tool_name":"read_file","payload":{"n":1}}')
            """
        )
        connection.commit()
    finally:
        connection.close()

    store = SQLiteSessionStore(db_path, schema_mode=schema_migrations.SchemaMode.MIGRATE)

    async def run() -> None:
        events = await store.load_events("sess_legacy")
        assert len(events) == 1
        assert events[0].id == "evt_1"
        assert events[0].payload == {"n": 1}
        assert events[0].tool_name == "read_file"
        await _close(store)

    asyncio.run(run())

    connection = sqlite3.connect(db_path)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cayu_events)").fetchall()
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    finally:
        connection.close()
    assert "event_json" not in columns
    assert version == schema_migrations.LATEST_REVISION


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
        old = _make_event(session.id, seq=1, timestamp=datetime(2026, 1, 1, tzinfo=UTC))
        new = _make_event(session.id, seq=2, timestamp=datetime(2026, 3, 1, tzinfo=UTC))
        await store.append_events(session.id, [old, new])

        deleted = await store.prune_events(
            before=datetime(2026, 2, 1, tzinfo=UTC), session_id=session.id
        )
        assert deleted == 1
        remaining = await store.load_events(session.id)
        assert [event.payload for event in remaining] == [{"n": 2}]

        # Unknown session is rejected; wrong-type cutoff is rejected.
        with pytest.raises(KeyError):
            await store.prune_events(before=datetime(2026, 2, 1, tzinfo=UTC), session_id="missing")
        with pytest.raises(TypeError):
            await store.prune_events(before="2026-02-01")  # type: ignore[arg-type]

        # A store-wide prune (no session_id) drops the rest.
        assert await store.prune_events(before=datetime(2026, 4, 1, tzinfo=UTC)) == 1
        assert await store.load_events(session.id) == []
        await _close(store)

    asyncio.run(run())


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

        # keep_last larger than the transcript deletes nothing.
        assert await store.compact_transcript(session.id, keep_last=10) == 0

        with pytest.raises(ValueError):
            await store.compact_transcript(session.id, keep_last=-1)
        with pytest.raises(KeyError):
            await store.compact_transcript("missing", keep_last=1)
        await _close(store)

    asyncio.run(run())
