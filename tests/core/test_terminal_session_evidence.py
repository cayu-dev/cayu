from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from tests._session_provenance import fixture_session_invocation
from tests.core.postgres_contention_support import drop_cayu_tables

import cayu.runtime.sessions as sessions_module
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
    EventQuery,
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
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
    TranscriptRecord,
)
from cayu._validation import compact_json_utf8_size
from cayu.runtime._terminal_evidence import (
    SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    TERMINAL_EVIDENCE_QUERY_LIMIT,
    classify_current_terminal_evidence,
)
from cayu.runtime.sessions import (
    RunnerObservedEventIdentity,
    _assemble_terminal_session_evidence,
    _classify_terminal_session_evidence_records,
    _event_with_session_run_operation,
    _SessionRunOperation,
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
        invocation=fixture_session_invocation("session-evidence"),
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
    session_id: str = "memory-terminal-evidence",
    interaction_id: str = "memory-interaction",
    status: SessionStatus = SessionStatus.COMPLETED,
    terminal_payload: dict | None = None,
    session_metadata: dict | None = None,
    publish_terminal: bool = True,
) -> tuple[str, str]:
    user_message = Message.text("user", "Give a concise answer.")
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[user_message],
            metadata={} if session_metadata is None else session_metadata,
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
    if publish_terminal:
        await store.append_event(
            session_id,
            Event(
                id=f"{session_id}-{status.value}-session",
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
                "cayu_targeted_tool_grant_uses",
                "cayu_targeted_tool_grants",
                "cayu_public_authority_aliases",
                "cayu_public_authority_alias_keys",
                "cayu_public_authority_alias_config",
            ):
                await cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await connection.commit()


async def _create_running_session(
    store: SessionStore,
    *,
    session_id: str,
    interaction_id: str,
) -> None:
    message = Message.text("user", "Exercise terminal evidence.")
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[message],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
        interaction_started_event=Event(
            id=f"{session_id}-interaction-started",
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id=interaction_id,
        ),
        interaction_source_messages=[message],
    )
    await store.replace_initial_transcript_messages(
        session_id,
        [message],
        [message],
        interaction_id=interaction_id,
    )


async def _create_interrupted_session(
    store: SessionStore,
    *,
    session_id: str,
    interaction_id: str,
    parent_session_id: str | None = None,
    terminal_payload: dict | None = None,
) -> tuple[RunnerObservedEventIdentity, ...]:
    message = Message.text("user", "Exercise runner-owned interrupted evidence.")
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            parent_session_id=parent_session_id,
            messages=[message],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
        interaction_started_event=Event(
            id=f"{session_id}-interaction-started",
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id=interaction_id,
        ),
        interaction_source_messages=[message],
    )
    await store.replace_initial_transcript_messages(
        session_id,
        [message],
        [message],
        interaction_id=interaction_id,
    )
    await store.publish_interaction_transition(
        session_id,
        event=Event(
            id=f"{session_id}-interaction-interrupted",
            type=EventType.INTERACTION_INTERRUPTED,
            session_id=session_id,
            interaction_id=interaction_id,
        ),
        from_statuses={SessionStatus.RUNNING},
        to_status=SessionStatus.INTERRUPTED,
    )
    await store.append_event(
        session_id,
        Event(
            id=f"{session_id}-session-interrupted",
            type=EventType.SESSION_INTERRUPTED,
            session_id=session_id,
            payload={} if terminal_payload is None else terminal_payload,
        ),
    )
    records = await store.query_events(EventQuery(session_id=session_id))
    return tuple(
        RunnerObservedEventIdentity(
            session_id=session_id,
            sequence=record.sequence,
            event_type=record.event.type,
        )
        for record in records
    )


async def _expect_store_error(
    store: SessionStore,
    session_id: str,
    expected: TerminalSessionEvidenceErrorCode,
) -> None:
    with pytest.raises(TerminalSessionEvidenceError) as captured:
        await store.load_terminal_session_evidence(session_id)
    assert captured.value.code is expected


def _assert_exact_snapshot_bytes(evidence: TerminalSessionEvidence) -> None:
    assert evidence.boundary.total_bytes == compact_json_utf8_size(evidence.model_dump(mode="json"))


async def _exercise_store_rejection_contract(store: SessionStore, *, prefix: str) -> None:
    await _expect_store_error(
        store,
        f"{prefix}-absent",
        TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND,
    )

    active_id = f"{prefix}-active"
    await _create_running_session(
        store,
        session_id=active_id,
        interaction_id=f"{active_id}-interaction",
    )
    await _expect_store_error(
        store,
        active_id,
        TerminalSessionEvidenceErrorCode.SESSION_NOT_TERMINAL,
    )

    interrupted_id = f"{prefix}-interrupted"
    interrupted_interaction_id = f"{interrupted_id}-interaction"
    await _create_running_session(
        store,
        session_id=interrupted_id,
        interaction_id=interrupted_interaction_id,
    )
    await store.publish_interaction_transition(
        interrupted_id,
        event=Event(
            id=f"{interrupted_id}-interaction-terminal",
            type=EventType.INTERACTION_INTERRUPTED,
            session_id=interrupted_id,
            interaction_id=interrupted_interaction_id,
        ),
        from_statuses={SessionStatus.RUNNING},
        to_status=SessionStatus.INTERRUPTED,
    )
    await store.append_event(
        interrupted_id,
        Event(
            id=f"{interrupted_id}-session-terminal",
            type=EventType.SESSION_INTERRUPTED,
            session_id=interrupted_id,
        ),
    )
    await _expect_store_error(
        store,
        interrupted_id,
        TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED,
    )

    missing_id, _ = await _create_terminal_session(
        store,
        session_id=f"{prefix}-missing",
        interaction_id=f"{prefix}-missing-interaction",
        publish_terminal=False,
    )
    await _expect_store_error(
        store,
        missing_id,
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_MISSING,
    )

    conflict_id, _ = await _create_terminal_session(
        store,
        session_id=f"{prefix}-conflict",
        interaction_id=f"{prefix}-conflict-interaction",
        publish_terminal=False,
    )
    await store.append_event(
        conflict_id,
        Event(
            id=f"{conflict_id}-failed",
            type=EventType.SESSION_FAILED,
            session_id=conflict_id,
        ),
    )
    await _expect_store_error(
        store,
        conflict_id,
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_CONFLICT,
    )

    duplicate_id, _ = await _create_terminal_session(
        store,
        session_id=f"{prefix}-duplicate",
        interaction_id=f"{prefix}-duplicate-interaction",
        publish_terminal=False,
    )
    await store.append_events(
        duplicate_id,
        [
            Event(
                id=f"{duplicate_id}-completed-{index}",
                type=EventType.SESSION_COMPLETED,
                session_id=duplicate_id,
            )
            for index in range(2)
        ],
    )
    await _expect_store_error(
        store,
        duplicate_id,
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_DUPLICATE,
    )

    buried_conflict_id, _ = await _create_terminal_session(
        store,
        session_id=f"{prefix}-buried-conflict",
        interaction_id=f"{prefix}-buried-conflict-interaction",
        publish_terminal=False,
    )
    await store.append_events(
        buried_conflict_id,
        [
            Event(
                id=f"{buried_conflict_id}-completed-first",
                type=EventType.SESSION_COMPLETED,
                session_id=buried_conflict_id,
                payload={SESSION_RUN_OPERATION_ID_PAYLOAD_KEY: "operation-a"},
            ),
            Event(
                id=f"{buried_conflict_id}-failed",
                type=EventType.SESSION_FAILED,
                session_id=buried_conflict_id,
                payload={SESSION_RUN_OPERATION_ID_PAYLOAD_KEY: "operation-b"},
            ),
            Event(
                id=f"{buried_conflict_id}-completed-last",
                type=EventType.SESSION_COMPLETED,
                session_id=buried_conflict_id,
                payload={SESSION_RUN_OPERATION_ID_PAYLOAD_KEY: "operation-a"},
            ),
        ],
    )
    await _expect_store_error(
        store,
        buried_conflict_id,
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_CONFLICT,
    )

    buried_duplicate_id, _ = await _create_terminal_session(
        store,
        session_id=f"{prefix}-buried-duplicate",
        interaction_id=f"{prefix}-buried-duplicate-interaction",
        publish_terminal=False,
    )
    await store.append_events(
        buried_duplicate_id,
        [
            Event(
                id=f"{buried_duplicate_id}-completed-{index}",
                type=EventType.SESSION_COMPLETED,
                session_id=buried_duplicate_id,
                payload={SESSION_RUN_OPERATION_ID_PAYLOAD_KEY: operation_id},
            )
            for index, operation_id in enumerate(("operation-a", "operation-b", "operation-a"))
        ],
    )
    await _expect_store_error(
        store,
        buried_duplicate_id,
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_DUPLICATE,
    )


async def _exercise_runner_owned_interrupted_evidence(
    store: SessionStore,
    *,
    prefix: str,
) -> None:
    root_id = f"{prefix}-runner-interrupted-root"
    observed = await _create_interrupted_session(
        store,
        session_id=root_id,
        interaction_id=f"{root_id}-interaction",
    )

    assert store.supports_runner_owned_interrupted_evidence is True
    with pytest.raises(TerminalSessionEvidenceError) as ordinary:
        await store.load_terminal_session_evidence(root_id)
    assert ordinary.value.code is TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED

    evidence = await store.load_runner_owned_interrupted_evidence(
        root_id,
        observed_events=observed,
    )
    assert evidence.session.status is SessionStatus.INTERRUPTED
    assert evidence.events[-1].event.type is EventType.SESSION_INTERRUPTED
    _assert_exact_snapshot_bytes(evidence)

    forged = list(observed)
    forged[-1] = RunnerObservedEventIdentity(
        session_id=root_id,
        sequence=forged[-1].sequence,
        event_type=EventType.SESSION_FAILED,
    )
    with pytest.raises(TerminalSessionEvidenceError) as mismatched:
        await store.load_runner_owned_interrupted_evidence(
            root_id,
            observed_events=tuple(forged),
        )
    assert mismatched.value.code is TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT

    with pytest.raises(TerminalSessionEvidenceError) as missing_proof:
        await store.load_runner_owned_interrupted_evidence(root_id)
    assert missing_proof.value.code is TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
    with pytest.raises(TerminalSessionEvidenceError) as ambiguous_proof:
        await store.load_runner_owned_interrupted_evidence(
            root_id,
            observed_events=observed,
            expected_parent_session_id="another-parent",
        )
    assert ambiguous_proof.value.code is TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT

    child_id = f"{prefix}-runner-interrupted-child"
    await _create_interrupted_session(
        store,
        session_id=child_id,
        interaction_id=f"{child_id}-interaction",
        parent_session_id=root_id,
    )
    child_evidence = await store.load_runner_owned_interrupted_evidence(
        child_id,
        expected_parent_session_id=root_id,
    )
    assert child_evidence.session.parent_session_id == root_id
    assert child_evidence.session.status is SessionStatus.INTERRUPTED
    with pytest.raises(TerminalSessionEvidenceError) as wrong_parent:
        await store.load_runner_owned_interrupted_evidence(
            child_id,
            expected_parent_session_id="not-the-runner-owned-parent",
        )
    assert wrong_parent.value.code is TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT


async def _exercise_marker_repair(store: SessionStore, *, prefix: str) -> None:
    session_id, _ = await _create_terminal_session(
        store,
        session_id=f"{prefix}-marker-repair",
        interaction_id=f"{prefix}-initial-interaction",
    )
    operation_id = f"{prefix}-operation"

    def mark_operation(session: Session, checkpoint: dict | None) -> dict:
        updated = {} if checkpoint is None else dict(checkpoint)
        updated["session_run_operation"] = {
            "version": 1,
            "operation_id": operation_id,
            "run_epoch": session.run_epoch + 1,
        }
        return updated

    resumed = await store.transition_status_and_checkpoint(
        session_id,
        from_statuses={SessionStatus.COMPLETED},
        to_status=SessionStatus.RUNNING,
        checkpoint_transform=mark_operation,
    )
    expected_run_epoch = resumed.run_epoch
    assert expected_run_epoch >= 1
    interaction_id = f"{prefix}-resumed-interaction"
    await store.append_events(
        session_id,
        [
            Event(
                id=f"{prefix}-session-resumed",
                type=EventType.SESSION_RESUMED,
                session_id=session_id,
            ),
            Event(
                id=f"{prefix}-interaction-resumed",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            ),
        ],
    )
    await store.append_transcript_messages(
        session_id,
        [Message.text("user", "Try again."), Message.text("assistant", "Recovered.")],
        interaction_id=interaction_id,
    )
    await store.publish_interaction_transition(
        session_id,
        event=Event(
            id=f"{prefix}-resumed-interaction-completed",
            type=EventType.INTERACTION_COMPLETED,
            session_id=session_id,
            interaction_id=interaction_id,
        ),
        from_statuses={SessionStatus.RUNNING},
        to_status=SessionStatus.COMPLETED,
    )
    terminal_event = _event_with_session_run_operation(
        Event(
            id=f"{prefix}-resumed-session-completed",
            type=EventType.SESSION_COMPLETED,
            session_id=session_id,
        ),
        _SessionRunOperation(
            operation_id=operation_id,
            run_epoch=expected_run_epoch,
        ),
    )
    await store.append_event(session_id, terminal_event)

    pending_cleanup = await store.load_terminal_session_evidence(session_id)
    assert pending_cleanup.terminal_publication_marker == TerminalPublicationMarker(
        operation_id=operation_id,
        run_epoch=expected_run_epoch,
    )
    assert pending_cleanup.boundary.run_epoch == expected_run_epoch
    assert pending_cleanup.terminal_event.event.id == terminal_event.id
    _assert_exact_snapshot_bytes(pending_cleanup)

    def clear_operation(_session: Session, checkpoint: dict | None) -> dict:
        updated = {} if checkpoint is None else dict(checkpoint)
        updated.pop("session_run_operation", None)
        return updated

    await store.transform_checkpoint(session_id, clear_operation)
    repaired = await store.load_terminal_session_evidence(session_id)
    assert repaired.terminal_publication_marker is None
    assert repaired.events == pending_cleanup.events
    assert repaired.transcript == pending_cleanup.transcript
    assert repaired.boundary.run_epoch == pending_cleanup.boundary.run_epoch
    _assert_exact_snapshot_bytes(repaired)


async def _exercise_atomic_write_races(store: SessionStore, *, prefix: str) -> None:
    session_id, interaction_id = await _create_terminal_session(
        store,
        session_id=f"{prefix}-terminal-race",
        interaction_id=f"{prefix}-terminal-race-interaction",
        publish_terminal=False,
    )

    async def read_while_terminalizing():
        try:
            return await store.load_terminal_session_evidence(session_id)
        except TerminalSessionEvidenceError as exc:
            return exc.code

    async def publish_terminal() -> None:
        await store.append_event(
            session_id,
            Event(
                id=f"{prefix}-terminal-race-completed",
                type=EventType.SESSION_COMPLETED,
                session_id=session_id,
            ),
        )

    results = await asyncio.gather(
        *(read_while_terminalizing() for _ in range(4)),
        publish_terminal(),
        *(read_while_terminalizing() for _ in range(4)),
    )
    for result in results:
        if result is None:
            continue
        if isinstance(result, TerminalSessionEvidenceErrorCode):
            assert result is TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_MISSING
            continue
        assert result.events[-1] == result.terminal_event
        assert result.boundary.event_count == len(result.events)

    before_append = await store.load_terminal_session_evidence(session_id)

    async def append_transcript() -> None:
        await store.append_transcript_messages(
            session_id,
            [Message.text("assistant", "Trailing durable diagnostic.")],
            interaction_id=interaction_id,
        )

    transcript_results = await asyncio.gather(
        *(store.load_terminal_session_evidence(session_id) for _ in range(4)),
        append_transcript(),
        *(store.load_terminal_session_evidence(session_id) for _ in range(4)),
    )
    after_append = await store.load_terminal_session_evidence(session_id)
    assert after_append.boundary.transcript_count == before_append.boundary.transcript_count + 1
    for result in transcript_results:
        if result is None:
            continue
        assert result in (before_append, after_append)


async def _exercise_unpaginated_snapshot(store: SessionStore, *, prefix: str) -> str:
    session_id = f"{prefix}-unpaginated"
    interaction_id = f"{prefix}-unpaginated-interaction"
    await _create_running_session(
        store,
        session_id=session_id,
        interaction_id=interaction_id,
    )
    await store.append_events(
        session_id,
        [
            Event(
                id=f"{prefix}-bulk-event-{index}",
                type="custom.evidence.pagination",
                session_id=session_id,
                interaction_id=interaction_id,
            )
            for index in range(257)
        ],
    )
    await store.append_transcript_messages(
        session_id,
        [Message.text("assistant", f"record-{index}") for index in range(257)],
        interaction_id=interaction_id,
    )
    await store.publish_interaction_transition(
        session_id,
        event=Event(
            id=f"{prefix}-bulk-interaction-completed",
            type=EventType.INTERACTION_COMPLETED,
            session_id=session_id,
            interaction_id=interaction_id,
        ),
        from_statuses={SessionStatus.RUNNING},
        to_status=SessionStatus.COMPLETED,
    )
    await store.append_event(
        session_id,
        Event(
            id=f"{prefix}-bulk-session-completed",
            type=EventType.SESSION_COMPLETED,
            session_id=session_id,
        ),
    )

    evidence = await store.load_terminal_session_evidence(session_id)
    assert evidence.boundary.event_count == 260
    assert evidence.boundary.transcript_count == 258
    assert evidence.events[-2].event.id == f"{prefix}-bulk-interaction-completed"
    assert evidence.transcript[-1].message == Message.text("assistant", "record-256")
    _assert_exact_snapshot_bytes(evidence)
    return session_id


async def _exercise_separator_dense_canonical_limits(
    store: SessionStore,
    *,
    prefix: str,
) -> tuple[str, int, int, int]:
    """Prove backend transport formatting cannot redefine canonical limits."""

    session_id, _ = await _create_terminal_session(
        store,
        session_id=f"{prefix}-separator-dense",
        interaction_id=f"{prefix}-separator-dense-interaction",
        terminal_payload={"values": [0] * 350_000},
    )
    evidence = await store.load_terminal_session_evidence(session_id)
    canonical_record_bytes = compact_json_utf8_size(evidence.terminal_event.model_dump(mode="json"))
    canonical_total_bytes = evidence.boundary.total_bytes
    assert canonical_record_bytes < TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES
    assert canonical_total_bytes < TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES

    tight = await store.load_terminal_session_evidence(
        session_id,
        limits=TerminalSessionEvidenceLimits(
            max_record_bytes=canonical_record_bytes,
            max_total_bytes=canonical_total_bytes,
        ),
    )
    assert tight == evidence
    return (
        session_id,
        evidence.terminal_event.sequence,
        canonical_record_bytes,
        canonical_total_bytes,
    )


async def _create_scientific_notation_evidence(
    store: SessionStore,
    *,
    prefix: str,
) -> tuple[str, TerminalSessionEvidence, int]:
    """Create portable evidence whose JSONB transport exceeds a 3:2 ratio."""

    session_id, _ = await _create_terminal_session(
        store,
        session_id=f"{prefix}-scientific-notation",
        interaction_id=f"{prefix}-scientific-notation-interaction",
        terminal_payload={"values": [1e-7] * 1_000},
    )
    evidence = await store.load_terminal_session_evidence(session_id)
    canonical_record_bytes = compact_json_utf8_size(evidence.terminal_event.model_dump(mode="json"))
    assert canonical_record_bytes < TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES
    return session_id, evidence, canonical_record_bytes


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
    assert TerminalSessionEvidenceLimits(max_transcript_records=0).max_transcript_records == 0

    for field, value in (
        ("max_events", TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS + 1),
        (
            "max_transcript_records",
            TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS + 1,
        ),
        ("max_record_bytes", TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES + 1),
        ("max_total_bytes", TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES + 1),
        ("max_transcript_records", -1),
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
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_CONFLICT,
        lambda: _classify(
            records=(
                _event_record(5, EventType.SESSION_COMPLETED),
                _event_record(4, EventType.SESSION_FAILED),
            )
        ),
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
    assert _classify(records=(current, prior)) == current
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
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_CONFLICT,
        lambda: _classify(
            session=_session(run_epoch=2),
            marker=marker,
            records=(
                current,
                _event_record(
                    4,
                    EventType.SESSION_FAILED,
                    operation_id="operation-2",
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


@pytest.mark.parametrize(
    "lifecycle_event_type",
    (
        EventType.SESSION_STARTED,
        EventType.SESSION_RESUMED,
        EventType.SESSION_FORKED,
    ),
)
def test_terminal_evidence_marker_never_crosses_the_newest_lifecycle(
    lifecycle_event_type: EventType,
) -> None:
    marker = TerminalPublicationMarker(operation_id="operation-2", run_epoch=2)
    lifecycle = _event_record(5, lifecycle_event_type)
    older_matching_terminal = _event_record(
        4,
        EventType.SESSION_COMPLETED,
        operation_id=marker.operation_id,
    )

    _assert_error_code(
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_MISSING,
        lambda: _classify(
            session=_session(run_epoch=marker.run_epoch),
            marker=marker,
            records=(lifecycle, older_matching_terminal),
        ),
    )


def test_terminal_evidence_scopes_interruption_identity_after_the_lifecycle() -> None:
    interruption_request_id = "interrupt-after-resume"
    classification = classify_current_terminal_evidence(
        evidence_events=(
            _event_record(5, EventType.SESSION_RESUMED).event,
            _event_record(
                4,
                EventType.SESSION_INTERRUPTED,
                payload={"interruption_request_id": interruption_request_id},
            ).event,
        ),
        expected_event_type=EventType.SESSION_INTERRUPTED,
        run_operation_id=None,
        interruption_request_id=interruption_request_id,
    )

    assert classification.events == ()
    assert classification.latest_lifecycle_event_type is EventType.SESSION_RESUMED


def test_terminal_evidence_preserves_strict_current_operation_classification() -> None:
    operation_id = "current-operation"
    current = _event_record(
        6,
        EventType.SESSION_COMPLETED,
        operation_id=operation_id,
    ).event
    duplicate = _event_record(
        5,
        EventType.SESSION_COMPLETED,
        operation_id=operation_id,
    ).event
    conflicting = _event_record(
        4,
        EventType.SESSION_FAILED,
        operation_id=operation_id,
    ).event
    another_operation = _event_record(
        3,
        EventType.SESSION_COMPLETED,
        operation_id="another-operation",
    ).event

    duplicates = classify_current_terminal_evidence(
        evidence_events=(current, duplicate),
        expected_event_type=EventType.SESSION_COMPLETED,
        run_operation_id=operation_id,
        interruption_request_id=None,
    )
    assert duplicates.events == (current, duplicate)

    conflict = classify_current_terminal_evidence(
        evidence_events=(current, conflicting),
        expected_event_type=EventType.SESSION_COMPLETED,
        run_operation_id=operation_id,
        interruption_request_id=None,
    )
    assert conflict.events == (current, conflicting)

    multiple_operations = classify_current_terminal_evidence(
        evidence_events=(current, another_operation),
        expected_event_type=EventType.SESSION_COMPLETED,
        run_operation_id=operation_id,
        interruption_request_id=None,
    )
    assert multiple_operations.events == (current,)


def test_terminal_evidence_rejects_input_beyond_the_store_query_bound() -> None:
    events = tuple(
        _event_record(sequence, EventType.SESSION_COMPLETED).event
        for sequence in range(1, TERMINAL_EVIDENCE_QUERY_LIMIT + 2)
    )

    with pytest.raises(ValueError, match="exceeds its bounded query"):
        classify_current_terminal_evidence(
            evidence_events=events,
            expected_event_type=EventType.SESSION_COMPLETED,
            run_operation_id=None,
            interruption_request_id=None,
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
    assert result.boundary.terminal_publication_marker_bytes == 0
    assert result.boundary.total_bytes == compact_json_utf8_size(result.model_dump(mode="json"))

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
    assert result.boundary.terminal_publication_marker_bytes == marker_bytes
    assert result.boundary.total_bytes == compact_json_utf8_size(result.model_dump(mode="json"))


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
    exact_total = _assemble_terminal_session_evidence(
        session=session,
        marker=None,
        terminal_record=events[-1],
        events=events,
        transcript=transcript,
        limits=TerminalSessionEvidenceLimits(),
    ).boundary.total_bytes

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

    exact = _assemble_terminal_session_evidence(
        session=session,
        marker=None,
        terminal_record=events[-1],
        events=events,
        transcript=transcript,
        limits=TerminalSessionEvidenceLimits(max_total_bytes=exact_total),
    )
    assert exact.boundary.total_bytes == exact_total


def test_terminal_evidence_public_validation_recomputes_derived_boundary_bytes() -> None:
    events = _complete_records()
    result = _assemble_terminal_session_evidence(
        session=_session(),
        marker=None,
        terminal_record=events[-1],
        events=events,
        transcript=_transcript(),
        limits=TerminalSessionEvidenceLimits(),
    )
    serialized = result.model_dump(mode="json")

    assert TerminalSessionEvidence.model_validate(serialized) == result

    serialized["boundary"]["total_bytes"] = 1
    with pytest.raises(ValidationError, match="boundary metadata"):
        TerminalSessionEvidence.model_validate(serialized)


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

    contradictory_prefix = (
        _event_record(
            1,
            EventType.INTERACTION_STARTED,
            interaction_id="interaction-1",
        ),
        _event_record(2, EventType.SESSION_COMPLETED, operation_id="operation-a"),
        _event_record(3, EventType.SESSION_FAILED, operation_id="operation-b"),
        _event_record(4, EventType.SESSION_COMPLETED, operation_id="operation-a"),
    )
    _assert_error_code(
        TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_CONFLICT,
        lambda: _assemble_terminal_session_evidence(
            session=_session(run_epoch=2),
            marker=None,
            terminal_record=contradictory_prefix[-1],
            events=contradictory_prefix,
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
        assert evidence.terminal_event.event.id == f"{session_id}-{status.value}-session"
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


def test_in_memory_terminal_evidence_returns_typed_missing_and_limit_errors(
    monkeypatch,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        with pytest.raises(TerminalSessionEvidenceError) as missing:
            await store.load_terminal_session_evidence("missing-session")
        assert missing.value.code is TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND

        session_id, _ = await _create_terminal_session(store)
        sized_records = 0
        original_record_bytes = sessions_module._terminal_session_evidence_record_bytes

        def record_bytes_spy(value):
            nonlocal sized_records
            sized_records += 1
            return original_record_bytes(value)

        monkeypatch.setattr(
            sessions_module,
            "_terminal_session_evidence_record_bytes",
            record_bytes_spy,
        )
        with pytest.raises(TerminalSessionEvidenceError) as events:
            await store.load_terminal_session_evidence(
                session_id,
                limits=TerminalSessionEvidenceLimits(max_events=2),
            )
        assert events.value.code is TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED
        assert (events.value.limit, events.value.observed) == (2, 3)
        assert sized_records == 0

        with pytest.raises(TerminalSessionEvidenceError) as transcript:
            await store.load_terminal_session_evidence(
                session_id,
                limits=TerminalSessionEvidenceLimits(max_transcript_records=2),
            )
        assert transcript.value.code is TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED
        assert (transcript.value.limit, transcript.value.observed) == (2, 3)
        assert sized_records == 0

        with pytest.raises(TerminalSessionEvidenceError) as total:
            await store.load_terminal_session_evidence(
                session_id,
                limits=TerminalSessionEvidenceLimits(max_total_bytes=1),
            )
        assert total.value.code is TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED
        assert sized_records == 1

    asyncio.run(run())


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
def test_builtin_terminal_evidence_acceptance_matrix(tmp_path, backend: str) -> None:
    async def run() -> None:
        store: SessionStore
        if backend == "memory":
            store = InMemorySessionStore()
        else:
            store = SQLiteSessionStore(tmp_path / "terminal-evidence-acceptance.sqlite")
        try:
            await _exercise_store_rejection_contract(store, prefix=backend)
            await _exercise_runner_owned_interrupted_evidence(store, prefix=backend)
            await _exercise_marker_repair(store, prefix=backend)
            await _exercise_atomic_write_races(store, prefix=backend)
            await _exercise_unpaginated_snapshot(store, prefix=backend)
            await _exercise_separator_dense_canonical_limits(store, prefix=backend)
            (
                scientific_id,
                scientific,
                canonical_record_bytes,
            ) = await _create_scientific_notation_evidence(store, prefix=backend)
            tight_scientific = await store.load_terminal_session_evidence(
                scientific_id,
                limits=TerminalSessionEvidenceLimits(
                    max_record_bytes=canonical_record_bytes,
                    max_total_bytes=scientific.boundary.total_bytes,
                ),
            )
            assert tight_scientific == scientific
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

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


def test_sqlite_runner_interrupted_evidence_rejects_oversized_data_before_hydration(
    tmp_path,
    monkeypatch,
) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "runner-interrupted-oversized.sqlite")
        try:
            session_id = "sqlite-runner-interrupted-oversized"
            observed = await _create_interrupted_session(
                store,
                session_id=session_id,
                interaction_id=f"{session_id}-interaction",
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
                await store.load_runner_owned_interrupted_evidence(
                    session_id,
                    observed_events=observed,
                    limits=TerminalSessionEvidenceLimits(max_record_bytes=1024),
                )
            assert captured.value.code is TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED
            assert hydrated_rows == 0
        finally:
            await store.close()

    asyncio.run(run())


def test_sqlite_terminal_evidence_queries_use_bounded_ordering_indexes(tmp_path) -> None:
    async def run() -> None:
        store = SQLiteSessionStore(tmp_path / "terminal-evidence-query-plan.sqlite")
        try:
            session_id, _ = await _create_terminal_session(store)
            evidence = await store.load_terminal_session_evidence(session_id)

            def explain(connection):
                event_rows = connection.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT sequence
                    FROM cayu_events
                    WHERE session_id = ? AND sequence <= ?
                    ORDER BY sequence ASC
                    LIMIT ?
                    """,
                    (session_id, evidence.boundary.terminal_event_sequence, 101),
                ).fetchall()
                transcript_rows = connection.execute(
                    """
                    EXPLAIN QUERY PLAN
                    SELECT session_order
                    FROM cayu_transcript_messages
                    WHERE session_id = ?
                    ORDER BY session_order ASC
                    LIMIT ?
                    """,
                    (session_id, 101),
                ).fetchall()
                return (
                    " ".join(str(row[3]) for row in event_rows),
                    " ".join(str(row[3]) for row in transcript_rows),
                )

            event_plan, transcript_plan = await store._run_read(explain)
            assert "idx_cayu_events_session_sequence" in event_plan
            assert "idx_cayu_transcript_session_order" in transcript_plan
            assert "USE TEMP B-TREE FOR ORDER BY" not in event_plan
            assert "USE TEMP B-TREE FOR ORDER BY" not in transcript_plan
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
            assert record.value.code is TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED
            assert hydrated_json_values == 0
        finally:
            await reopened.close()

    asyncio.run(run())


def test_postgres_terminal_evidence_acceptance_matrix(postgres_dsn: str) -> None:
    async def run() -> None:
        from cayu import PostgresSessionStore
        from cayu.storage.migrations import SchemaMode

        await _reset_postgres(postgres_dsn)
        store = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await _exercise_store_rejection_contract(store, prefix="postgres")
            await _exercise_runner_owned_interrupted_evidence(store, prefix="postgres")
            await _exercise_marker_repair(store, prefix="postgres")
            await _exercise_atomic_write_races(store, prefix="postgres")
            await _exercise_unpaginated_snapshot(store, prefix="postgres")
            (
                _,
                terminal_sequence,
                canonical_record_bytes,
                _,
            ) = await _exercise_separator_dense_canonical_limits(
                store,
                prefix="postgres",
            )
            async with store._connection() as connection, connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT octet_length(event::text) + 1
                           + octet_length(sequence::text)
                    FROM cayu_events
                    WHERE sequence = %s
                    """,
                    (terminal_sequence,),
                )
                transport_row = await cursor.fetchone()
            assert transport_row is not None
            assert int(transport_row[0]) > canonical_record_bytes

            (
                scientific_id,
                scientific,
                scientific_record_bytes,
            ) = await _create_scientific_notation_evidence(
                store,
                prefix="postgres",
            )
            scientific_transport_limit = (
                scientific_record_bytes + (scientific_record_bytes + 1) // 2
            )
            with pytest.raises(TerminalSessionEvidenceError) as scientific_transport:
                await store.load_terminal_session_evidence(
                    scientific_id,
                    limits=TerminalSessionEvidenceLimits(
                        max_record_bytes=scientific_record_bytes,
                        max_total_bytes=scientific.boundary.total_bytes,
                    ),
                )
            assert (
                scientific_transport.value.code
                is TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED
            )
            async with store._connection() as connection, connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT octet_length(event::text) + 1
                           + octet_length(sequence::text)
                    FROM cayu_events
                    WHERE sequence = %s
                    """,
                    (scientific.terminal_event.sequence,),
                )
                scientific_transport_row = await cursor.fetchone()
            assert scientific_transport_row is not None
            assert int(scientific_transport_row[0]) > scientific_transport_limit
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_terminal_evidence_preflight_bounds_whitespace_before_hydration(
    postgres_dsn: str,
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
        try:
            oversized_whitespace = " " * 2_000_025
            event_session_id, _ = await _create_terminal_session(
                store,
                session_id="postgres-whitespace-event-preflight",
                terminal_payload={"diagnostic": oversized_whitespace},
            )
            transcript_session_id, transcript_interaction_id = await _create_terminal_session(
                store,
                session_id="postgres-whitespace-transcript-preflight",
            )
            await store.append_transcript_messages(
                transcript_session_id,
                [Message.text("assistant", f"x{oversized_whitespace}")],
                interaction_id=transcript_interaction_id,
            )
            metadata_session_id, _ = await _create_terminal_session(
                store,
                session_id="postgres-whitespace-metadata-preflight",
                session_metadata={"diagnostic": oversized_whitespace},
            )
            interrupted_session_id = "postgres-whitespace-interrupted-preflight"
            interrupted_observed = await _create_interrupted_session(
                store,
                session_id=interrupted_session_id,
                interaction_id=f"{interrupted_session_id}-interaction",
                terminal_payload={"diagnostic": oversized_whitespace},
            )

            import cayu.storage.postgres as postgres_store_module

            hydrated_json_values = 0
            original_json_obj = postgres_store_module._json_obj

            def json_obj_spy(value):
                nonlocal hydrated_json_values
                hydrated_json_values += 1
                return original_json_obj(value)

            monkeypatch.setattr(postgres_store_module, "_json_obj", json_obj_spy)
            for session_id in (
                event_session_id,
                transcript_session_id,
                metadata_session_id,
            ):
                hydrated_json_values = 0
                with pytest.raises(TerminalSessionEvidenceError) as captured:
                    await store.load_terminal_session_evidence(session_id)
                assert (
                    captured.value.code is TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED
                )
                assert hydrated_json_values == 0

            hydrated_json_values = 0
            with pytest.raises(TerminalSessionEvidenceError) as interrupted:
                await store.load_runner_owned_interrupted_evidence(
                    interrupted_session_id,
                    observed_events=interrupted_observed,
                )
            assert (
                interrupted.value.code is TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED
            )
            assert hydrated_json_values == 0
        finally:
            await store.close()

    asyncio.run(run())


def test_postgres_terminal_evidence_queries_use_bounded_ordering_indexes(
    postgres_dsn: str,
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
        try:
            session_id, _ = await _create_terminal_session(store)
            evidence = await store.load_terminal_session_evidence(session_id)
            async with store._connection() as connection, connection.cursor() as cursor:
                await cursor.execute("SET LOCAL enable_seqscan = off")
                await cursor.execute("SET LOCAL enable_bitmapscan = off")
                await cursor.execute(
                    """
                    EXPLAIN (FORMAT JSON)
                    SELECT sequence
                    FROM cayu_events
                    WHERE session_id = %s AND sequence <= %s
                    ORDER BY sequence ASC
                    LIMIT %s
                    """,
                    (session_id, evidence.boundary.terminal_event_sequence, 101),
                )
                event_plan = str((await cursor.fetchone())[0])
                await cursor.execute(
                    """
                    EXPLAIN (FORMAT JSON)
                    SELECT session_order
                    FROM cayu_transcript_messages
                    WHERE session_id = %s
                    ORDER BY session_order ASC
                    LIMIT %s
                    """,
                    (session_id, 101),
                )
                transcript_plan = str((await cursor.fetchone())[0])

            assert "idx_cayu_events_session_sequence" in event_plan
            assert "idx_cayu_transcript_session_order" in transcript_plan
        finally:
            await store.close()

    asyncio.run(run())
