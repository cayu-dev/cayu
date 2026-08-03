from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from tests.core.postgres_contention_support import drop_cayu_tables

import cayu.storage.sqlite as sqlite_store_module
from cayu import (
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS,
    Event,
    EventRecord,
    EventType,
    InMemorySessionStore,
    Message,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
    SessionStore,
    SQLiteSessionStore,
    TerminalPublicationMarker,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
    TranscriptRecord,
)
from cayu._validation import compact_json_utf8_size
from cayu.runtime._terminal_evidence import SESSION_RUN_OPERATION_ID_PAYLOAD_KEY
from cayu.runtime.sessions import (
    _assemble_terminal_session_evidence,
    _classify_terminal_session_evidence_records,
)


def _session(
    status: SessionStatus = SessionStatus.COMPLETED,
    *,
    run_epoch: int = 1,
) -> Session:
    now = datetime(2026, 8, 3, tzinfo=UTC)
    return Session(
        id="session-evidence",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        causal_budget_id="session-evidence",
        status=status,
        created_at=now,
        updated_at=now,
        last_activity_at=now,
        run_epoch=run_epoch,
        metadata={"purpose": "conformance"},
    )


def _event_record(
    sequence: int,
    event_type: EventType,
    *,
    interaction_id: str | None = None,
    operation_id: str | None = None,
    payload: dict | None = None,
) -> EventRecord:
    event_payload = {} if payload is None else dict(payload)
    if operation_id is not None:
        event_payload[SESSION_RUN_OPERATION_ID_PAYLOAD_KEY] = operation_id
    return EventRecord(
        sequence=sequence,
        event=Event(
            id=f"event-{sequence}",
            type=event_type,
            session_id="session-evidence",
            interaction_id=interaction_id,
            timestamp=datetime(2026, 8, 3, 0, 0, sequence, tzinfo=UTC),
            payload=event_payload,
        ),
    )


def _complete_records(*, operation_id: str | None = None) -> tuple[EventRecord, ...]:
    return (
        _event_record(
            1,
            EventType.INTERACTION_STARTED,
            interaction_id="interaction-1",
        ),
        _event_record(
            2,
            EventType.INTERACTION_COMPLETED,
            interaction_id="interaction-1",
        ),
        _event_record(
            3,
            EventType.SESSION_COMPLETED,
            operation_id=operation_id,
        ),
    )


def _transcript() -> tuple[TranscriptRecord, ...]:
    return (
        TranscriptRecord(
            index=0,
            interaction_id=None,
            message=Message.text("system", "You are helpful."),
        ),
        TranscriptRecord(
            index=1,
            interaction_id="interaction-1",
            message=Message.text("user", "Answer precisely."),
        ),
        TranscriptRecord(
            index=2,
            interaction_id="interaction-1",
            message=Message.text("assistant", "Done."),
        ),
    )


def _classify(
    *,
    session: Session | None = None,
    marker: TerminalPublicationMarker | None = None,
    records: tuple[EventRecord, ...],
    initial_transcript_pending: bool = False,
    pending_session_interrupt: bool = False,
) -> EventRecord:
    selected_session = _session() if session is None else session
    return _classify_terminal_session_evidence_records(
        session_id=selected_session.id,
        status=selected_session.status,
        run_epoch=selected_session.run_epoch,
        marker=marker,
        newest_evidence_records=records,
        initial_transcript_pending=initial_transcript_pending,
        pending_session_interrupt=pending_session_interrupt,
    )


def _assert_error_code(
    expected: TerminalSessionEvidenceErrorCode,
    callback,
) -> TerminalSessionEvidenceError:
    with pytest.raises(TerminalSessionEvidenceError) as captured:
        callback()
    assert captured.value.code is expected
    return captured.value


async def _create_terminal_session(
    store: SessionStore,
    *,
    status: SessionStatus = SessionStatus.COMPLETED,
    terminal_payload: dict | None = None,
) -> tuple[str, str]:
    session_id = "memory-terminal-evidence"
    interaction_id = "memory-interaction"
    user_message = Message.text("user", "Give a concise answer.")
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[user_message],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
        interaction_started_event=Event(
            id="memory-interaction-started",
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id=interaction_id,
        ),
        interaction_source_messages=[user_message],
    )
    await store.replace_initial_transcript_messages(
        session_id,
        [user_message],
        [Message.text("system", "Be precise."), user_message],
        interaction_id=interaction_id,
    )
    await store.append_transcript_messages(
        session_id,
        [Message.text("assistant", "A concise answer.")],
        interaction_id=interaction_id,
    )
    interaction_terminal_type = (
        EventType.INTERACTION_COMPLETED
        if status == SessionStatus.COMPLETED
        else EventType.INTERACTION_FAILED
    )
    session_terminal_type = (
        EventType.SESSION_COMPLETED
        if status == SessionStatus.COMPLETED
        else EventType.SESSION_FAILED
    )
    await store.publish_interaction_transition(
        session_id,
        event=Event(
            id=f"memory-{status.value}-interaction",
            type=interaction_terminal_type,
            session_id=session_id,
            interaction_id=interaction_id,
        ),
        from_statuses={SessionStatus.RUNNING},
        to_status=status,
    )
    await store.append_event(
        session_id,
        Event(
            id=f"memory-{status.value}-session",
            type=session_terminal_type,
            session_id=session_id,
            payload={} if terminal_payload is None else terminal_payload,
        ),
    )
    return session_id, interaction_id


async def _reset_postgres(dsn: str) -> None:
    await drop_cayu_tables(dsn)
    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as connection:
        async with connection.cursor() as cursor:
            for table in (
                "cayu_public_authority_aliases",
                "cayu_public_authority_alias_keys",
                "cayu_public_authority_alias_config",
            ):
                await cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await connection.commit()


def test_terminal_session_evidence_limits_publish_bounded_defaults_and_hard_caps() -> None:
    limits = TerminalSessionEvidenceLimits()

    assert limits == TerminalSessionEvidenceLimits(
        max_events=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
        max_transcript_records=(TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS),
        max_record_bytes=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES,
        max_total_bytes=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES,
    )
    assert limits.max_events < TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS
    assert limits.max_transcript_records < TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS
    assert limits.max_record_bytes < TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES
    assert limits.max_total_bytes < TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES

    for field, value in (
        ("max_events", TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS + 1),
        (
            "max_transcript_records",
            TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS + 1,
        ),
        ("max_record_bytes", TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES + 1),
        ("max_total_bytes", TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES + 1),
        ("max_events", True),
    ):
        with pytest.raises(ValidationError):
            TerminalSessionEvidenceLimits(**{field: value})


@pytest.mark.parametrize(
    ("status", "code"),
    (
        (SessionStatus.PENDING, TerminalSessionEvidenceErrorCode.SESSION_NOT_TERMINAL),
        (SessionStatus.RUNNING, TerminalSessionEvidenceErrorCode.SESSION_NOT_TERMINAL),
        (
            SessionStatus.INTERRUPTING,
            TerminalSessionEvidenceErrorCode.SESSION_NOT_TERMINAL,
        ),
        (SessionStatus.INTERRUPTED, TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED),
    ),
)
def test_terminal_evidence_rejects_unsupported_session_statuses(
    status: SessionStatus,
    code: TerminalSessionEvidenceErrorCode,
) -> None:
    _assert_error_code(
        code,
        lambda: _classify(
            session=_session(status),
            records=(_event_record(1, EventType.SESSION_INTERRUPTED),),
        ),
    )


def test_terminal_evidence_classification_accepts_completed_and_failed_sessions() -> None:
    completed = _event_record(4, EventType.SESSION_COMPLETED)
    failed = _event_record(5, EventType.SESSION_FAILED)

    assert _classify(records=(completed,)) == completed
    assert _classify(session=_session(SessionStatus.FAILED), records=(failed,)) == failed


def test_terminal_evidence_classification_rejects_missing_conflicting_and_duplicate_events() -> (
    None
):
    _assert_error_code(
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_MISSING,
        lambda: _classify(records=(_event_record(4, EventType.SESSION_RESUMED),)),
    )
    _assert_error_code(
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_CONFLICT,
        lambda: _classify(records=(_event_record(4, EventType.SESSION_FAILED),)),
    )
    _assert_error_code(
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_DUPLICATE,
        lambda: _classify(
            records=(
                _event_record(5, EventType.SESSION_COMPLETED),
                _event_record(4, EventType.SESSION_COMPLETED),
            )
        ),
    )


def test_terminal_evidence_classification_binds_a_pending_publication_marker() -> None:
    marker = TerminalPublicationMarker(operation_id="operation-2", run_epoch=2)
    current = _event_record(
        5,
        EventType.SESSION_COMPLETED,
        operation_id="operation-2",
    )
    prior = _event_record(
        4,
        EventType.SESSION_FAILED,
        operation_id="operation-1",
    )

    assert (
        _classify(
            session=_session(run_epoch=2),
            marker=marker,
            records=(current, prior),
        )
        == current
    )
    _assert_error_code(
        TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_CONFLICT,
        lambda: _classify(
            session=_session(run_epoch=2),
            marker=marker,
            records=(
                _event_record(
                    5,
                    EventType.SESSION_COMPLETED,
                    operation_id="different-operation",
                ),
            ),
        ),
    )
    _assert_error_code(
        TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_CONFLICT,
        lambda: _classify(
            session=_session(run_epoch=3),
            marker=marker,
            records=(current,),
        ),
    )


def test_terminal_evidence_classification_rejects_incomplete_publication_authority() -> None:
    terminal = _event_record(3, EventType.SESSION_COMPLETED)

    _assert_error_code(
        TerminalSessionEvidenceErrorCode.INITIAL_TRANSCRIPT_INCOMPLETE,
        lambda: _classify(records=(terminal,), initial_transcript_pending=True),
    )
    _assert_error_code(
        TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_CONFLICT,
        lambda: _classify(records=(terminal,), pending_session_interrupt=True),
    )


def test_terminal_evidence_assembly_returns_exact_detached_boundaries() -> None:
    session = _session()
    events = _complete_records()
    transcript = _transcript()
    result = _assemble_terminal_session_evidence(
        session=session,
        marker=None,
        terminal_record=events[-1],
        events=events,
        transcript=transcript,
        limits=TerminalSessionEvidenceLimits(),
    )

    expected_session_bytes = compact_json_utf8_size(session.model_dump(mode="json"))
    expected_event_bytes = sum(
        compact_json_utf8_size(record.model_dump(mode="json")) for record in events
    )
    expected_transcript_bytes = sum(
        compact_json_utf8_size(record.model_dump(mode="json")) for record in transcript
    )
    assert result.session == session
    assert result.events == events
    assert result.transcript == transcript
    assert result.terminal_event == events[-1]
    assert result.lifecycle_events == events
    assert result.initial_interaction_id == "interaction-1"
    assert result.boundary.first_event_sequence == 1
    assert result.boundary.terminal_event_sequence == 3
    assert result.boundary.transcript_end_index_exclusive == 3
    assert result.boundary.run_epoch == 1
    assert result.boundary.lifecycle_event_sequences == (1, 2, 3)
    assert result.boundary.session_bytes == expected_session_bytes
    assert result.boundary.event_bytes == expected_event_bytes
    assert result.boundary.transcript_bytes == expected_transcript_bytes
    assert result.boundary.total_bytes == (
        expected_session_bytes + expected_event_bytes + expected_transcript_bytes
    )

    session.metadata["purpose"] = "mutated"
    events[-1].event.payload["late"] = True
    assert result.session.metadata == {"purpose": "conformance"}
    assert result.terminal_event.event.payload == {}


def test_terminal_evidence_assembly_preserves_a_matching_marker_in_its_byte_budget() -> None:
    session = _session(run_epoch=2)
    events = _complete_records(operation_id="operation-2")
    marker = TerminalPublicationMarker(operation_id="operation-2", run_epoch=2)
    terminal = _classify(
        session=session,
        marker=marker,
        records=(events[-1],),
    )
    result = _assemble_terminal_session_evidence(
        session=session,
        marker=marker,
        terminal_record=terminal,
        events=events,
        transcript=_transcript(),
        limits=TerminalSessionEvidenceLimits(),
    )

    marker_bytes = compact_json_utf8_size(marker.model_dump(mode="json"))
    assert result.terminal_publication_marker == marker
    assert result.boundary.total_bytes == (
        result.boundary.session_bytes
        + result.boundary.event_bytes
        + result.boundary.transcript_bytes
        + marker_bytes
    )


def test_terminal_evidence_assembly_enforces_independent_count_limits() -> None:
    events = _complete_records()
    transcript = _transcript()

    error = _assert_error_code(
        TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
        lambda: _assemble_terminal_session_evidence(
            session=_session(),
            marker=None,
            terminal_record=events[-1],
            events=events,
            transcript=transcript,
            limits=TerminalSessionEvidenceLimits(max_events=2),
        ),
    )
    assert (error.limit, error.observed) == (2, 3)

    error = _assert_error_code(
        TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
        lambda: _assemble_terminal_session_evidence(
            session=_session(),
            marker=None,
            terminal_record=events[-1],
            events=events,
            transcript=transcript,
            limits=TerminalSessionEvidenceLimits(max_transcript_records=2),
        ),
    )
    assert (error.limit, error.observed) == (2, 3)


def test_terminal_evidence_assembly_enforces_record_and_total_byte_limits() -> None:
    events = _complete_records()
    transcript = _transcript()
    session = _session()
    record_sizes = [
        compact_json_utf8_size(session.model_dump(mode="json")),
        *(compact_json_utf8_size(record.model_dump(mode="json")) for record in events),
        *(compact_json_utf8_size(record.model_dump(mode="json")) for record in transcript),
    ]
    exact_total = sum(record_sizes)

    error = _assert_error_code(
        TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
        lambda: _assemble_terminal_session_evidence(
            session=session,
            marker=None,
            terminal_record=events[-1],
            events=events,
            transcript=transcript,
            limits=TerminalSessionEvidenceLimits(max_record_bytes=max(record_sizes) - 1),
        ),
    )
    assert error.observed == max(record_sizes)

    error = _assert_error_code(
        TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
        lambda: _assemble_terminal_session_evidence(
            session=session,
            marker=None,
            terminal_record=events[-1],
            events=events,
            transcript=transcript,
            limits=TerminalSessionEvidenceLimits(max_total_bytes=exact_total - 1),
        ),
    )
    assert (error.limit, error.observed) == (exact_total - 1, exact_total)


def test_terminal_evidence_assembly_rejects_torn_or_unattributed_boundaries() -> None:
    events = _complete_records()

    _assert_error_code(
        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT,
        lambda: _assemble_terminal_session_evidence(
            session=_session(),
            marker=None,
            terminal_record=events[-1],
            events=events[:-1],
            transcript=_transcript(),
            limits=TerminalSessionEvidenceLimits(),
        ),
    )
    _assert_error_code(
        TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT,
        lambda: _assemble_terminal_session_evidence(
            session=_session(),
            marker=None,
            terminal_record=events[-1],
            events=events,
            transcript=(
                TranscriptRecord(
                    index=0,
                    interaction_id="unknown-interaction",
                    message=Message.text("user", "not attributable"),
                ),
            ),
            limits=TerminalSessionEvidenceLimits(),
        ),
    )


def test_terminal_evidence_errors_never_embed_session_or_record_content() -> None:
    error = TerminalSessionEvidenceError(
        TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
        limit=1024,
        observed=4096,
    )

    assert str(error) == "Terminal-session evidence exceeds the total-byte limit. Limit: 1024."
    assert "4096" not in str(error)


@pytest.mark.parametrize(
    "status",
    (SessionStatus.COMPLETED, SessionStatus.FAILED),
)
def test_in_memory_terminal_evidence_returns_one_atomic_terminal_prefix(
    status: SessionStatus,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session_id, interaction_id = await _create_terminal_session(
            store,
            status=status,
        )
        await store.append_event(
            session_id,
            Event(
                id="post-terminal-hook",
                type=EventType.HOOK_COMPLETED,
                session_id=session_id,
                payload={"phase": "after_terminal"},
            ),
        )

        evidence = await store.load_terminal_session_evidence(session_id)

        expected_terminal_type = (
            EventType.SESSION_COMPLETED
            if status == SessionStatus.COMPLETED
            else EventType.SESSION_FAILED
        )
        assert store.supports_terminal_session_evidence is True
        assert evidence.session.status is status
        assert evidence.terminal_event.event.type == expected_terminal_type
        assert evidence.terminal_event.event.id == f"memory-{status.value}-session"
        assert evidence.events[-1] == evidence.terminal_event
        assert all(record.event.id != "post-terminal-hook" for record in evidence.events)
        assert [record.index for record in evidence.transcript] == [0, 1, 2]
        assert [record.interaction_id for record in evidence.transcript] == [
            None,
            interaction_id,
            interaction_id,
        ]
        assert evidence.initial_interaction_id == interaction_id
        assert evidence.boundary.event_count == 3
        assert evidence.boundary.transcript_count == 3

        evidence.terminal_event.event.payload["mutated"] = True
        reread = await store.load_terminal_session_evidence(session_id)
        assert "mutated" not in reread.terminal_event.event.payload

    asyncio.run(run())


def test_in_memory_terminal_evidence_returns_typed_missing_and_limit_errors() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        with pytest.raises(TerminalSessionEvidenceError) as missing:
            await store.load_terminal_session_evidence("missing-session")
        assert missing.value.code is TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND

        session_id, _ = await _create_terminal_session(store)
        with pytest.raises(TerminalSessionEvidenceError) as events:
            await store.load_terminal_session_evidence(
                session_id,
                limits=TerminalSessionEvidenceLimits(max_events=2),
            )
        assert events.value.code is TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED
        assert (events.value.limit, events.value.observed) == (2, 3)

        with pytest.raises(TerminalSessionEvidenceError) as transcript:
            await store.load_terminal_session_evidence(
                session_id,
                limits=TerminalSessionEvidenceLimits(max_transcript_records=2),
            )
        assert transcript.value.code is TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED
        assert (transcript.value.limit, transcript.value.observed) == (2, 3)

    asyncio.run(run())


@pytest.mark.parametrize(
    "status",
    (SessionStatus.COMPLETED, SessionStatus.FAILED),
)
def test_sqlite_terminal_evidence_survives_restart_with_the_exact_boundary(
    tmp_path,
    status: SessionStatus,
) -> None:
    async def run() -> None:
        path = tmp_path / f"terminal-evidence-{status.value}.sqlite"
        store = SQLiteSessionStore(path)
        session_id, interaction_id = await _create_terminal_session(store, status=status)
        await store.append_event(
            session_id,
            Event(
                id="sqlite-post-terminal-hook",
                type=EventType.HOOK_COMPLETED,
                session_id=session_id,
            ),
        )
        before_restart = await store.load_terminal_session_evidence(session_id)
        await store.close()

        reopened = SQLiteSessionStore(path)
        try:
            after_restart = await reopened.load_terminal_session_evidence(session_id)
            assert reopened.supports_terminal_session_evidence is True
            assert after_restart == before_restart
            assert after_restart.initial_interaction_id == interaction_id
            assert after_restart.boundary.event_count == 3
            assert after_restart.boundary.transcript_count == 3
            assert all(
                record.event.id != "sqlite-post-terminal-hook" for record in after_restart.events
            )
        finally:
            await reopened.close()

    asyncio.run(run())


def test_sqlite_terminal_evidence_preflights_counts_and_stored_record_lengths(
    tmp_path,
) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "terminal-evidence-limits.sqlite")
        try:
            session_id, _ = await _create_terminal_session(store)
            with pytest.raises(TerminalSessionEvidenceError) as events:
                await store.load_terminal_session_evidence(
                    session_id,
                    limits=TerminalSessionEvidenceLimits(max_events=2),
                )
            assert events.value.code is TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED
            assert (events.value.limit, events.value.observed) == (2, 3)

            with pytest.raises(TerminalSessionEvidenceError) as transcript:
                await store.load_terminal_session_evidence(
                    session_id,
                    limits=TerminalSessionEvidenceLimits(max_transcript_records=2),
                )
            assert (
                transcript.value.code is TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED
            )

            with pytest.raises(TerminalSessionEvidenceError) as record:
                await store.load_terminal_session_evidence(
                    session_id,
                    limits=TerminalSessionEvidenceLimits(max_record_bytes=32),
                )
            assert record.value.code is TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED

            with pytest.raises(TerminalSessionEvidenceError) as total:
                await store.load_terminal_session_evidence(
                    session_id,
                    limits=TerminalSessionEvidenceLimits(max_total_bytes=128),
                )
            assert total.value.code is TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_terminal_evidence_rejects_an_oversized_terminal_before_hydration(
    tmp_path,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "terminal-evidence-oversized.sqlite")
        try:
            session_id, _ = await _create_terminal_session(
                store,
                terminal_payload={"diagnostic": "x" * 2048},
            )
            hydrated_rows = 0
            original = sqlite_store_module._event_from_row

            def spy(row):
                nonlocal hydrated_rows
                hydrated_rows += 1
                return original(row)

            monkeypatch.setattr(sqlite_store_module, "_event_from_row", spy)
            with pytest.raises(TerminalSessionEvidenceError) as captured:
                await store.load_terminal_session_evidence(
                    session_id,
                    limits=TerminalSessionEvidenceLimits(max_record_bytes=1024),
                )
            assert captured.value.code is TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED
            assert hydrated_rows == 0
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "status",
    (SessionStatus.COMPLETED, SessionStatus.FAILED),
)
def test_postgres_terminal_evidence_survives_restart_with_the_exact_boundary(
    postgres_dsn: str,
    status: SessionStatus,
    monkeypatch,
) -> None:
    async def run() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        await _reset_postgres(postgres_dsn)
        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        session_id, interaction_id = await _create_terminal_session(store, status=status)
        await store.append_event(
            session_id,
            Event(
                id="postgres-post-terminal-hook",
                type=EventType.HOOK_COMPLETED,
                session_id=session_id,
            ),
        )
        before_restart = await store.load_terminal_session_evidence(session_id)
        await store.close()

        reopened = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
            read_only=True,
        )
        try:
            after_restart = await reopened.load_terminal_session_evidence(session_id)
            assert reopened.supports_terminal_session_evidence is True
            assert after_restart == before_restart
            assert after_restart.initial_interaction_id == interaction_id
            assert after_restart.boundary.event_count == 3
            assert after_restart.boundary.transcript_count == 3
            assert all(
                record.event.id != "postgres-post-terminal-hook" for record in after_restart.events
            )

            with pytest.raises(TerminalSessionEvidenceError) as events:
                await reopened.load_terminal_session_evidence(
                    session_id,
                    limits=TerminalSessionEvidenceLimits(max_events=2),
                )
            assert events.value.code is TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED

            import cayu.storage.postgres as postgres_store_module

            hydrated_json_values = 0
            original_json_obj = postgres_store_module._json_obj

            def json_obj_spy(value):
                nonlocal hydrated_json_values
                hydrated_json_values += 1
                return original_json_obj(value)

            monkeypatch.setattr(postgres_store_module, "_json_obj", json_obj_spy)
            with pytest.raises(TerminalSessionEvidenceError) as record:
                await reopened.load_terminal_session_evidence(
                    session_id,
                    limits=TerminalSessionEvidenceLimits(max_record_bytes=32),
                )
            assert record.value.code is TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED
            assert hydrated_json_values == 0
        finally:
            await reopened.close()

    asyncio.run(run())
