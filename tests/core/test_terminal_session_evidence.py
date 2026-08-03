from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

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
    Message,
    Session,
    SessionStatus,
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
    return _classify_terminal_session_evidence_records(
        session=_session() if session is None else session,
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
