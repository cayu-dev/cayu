"""Bounded terminal-session snapshots and their shared validation rules."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from itertools import pairwise
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu._validation import MAX_DURABLE_JSON_INTEGER, compact_json_utf8_size
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions._terminal_evidence import (
    _SESSION_RUN_OPERATION_CHECKPOINT_KEY,
    SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    TERMINAL_EVENT_TYPES,
    TERMINAL_EVIDENCE_EVENT_TYPES,
    TERMINAL_EVIDENCE_QUERY_LIMIT,
    TERMINAL_LIFECYCLE_EVENT_TYPES,
    _session_run_operation_from_checkpoint,
    classify_current_terminal_evidence,
)
from cayu.sessions.records import (
    EventRecord,
    RunnerObservedEventIdentity,
    Session,
    SessionStatus,
    TranscriptRecord,
    copy_session,
)

TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS = 10_000
TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS = 5_000
TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES = 1024 * 1024
TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES = 32 * 1024 * 1024
TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS = 100_000
TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS = 50_000
TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES = 8 * 1024 * 1024
TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES = 128 * 1024 * 1024


_TERMINAL_SESSION_EVIDENCE_INTERACTION_LIFECYCLE_EVENT_TYPES = frozenset(
    {
        EventType.INTERACTION_STARTED,
        EventType.INTERACTION_RESUMED,
        EventType.INTERACTION_PAUSED,
        EventType.INTERACTION_COMPLETED,
        EventType.INTERACTION_FAILED,
        EventType.INTERACTION_INTERRUPTED,
    }
)


_TERMINAL_SESSION_EVIDENCE_LIFECYCLE_EVENT_TYPES = frozenset(
    {
        *TERMINAL_EVIDENCE_EVENT_TYPES,
        *_TERMINAL_SESSION_EVIDENCE_INTERACTION_LIFECYCLE_EVENT_TYPES,
    }
)


class TerminalSessionEvidenceLimits(BaseModel):
    """Caller-selected ceilings for one exact terminal-session snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_events: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
        ge=1,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS,
    )
    max_transcript_records: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS,
        ge=0,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS,
    )
    max_record_bytes: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES,
        ge=1,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
    )
    max_total_bytes: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES,
        ge=1,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES,
    )


class TerminalSessionEvidenceErrorCode(StrEnum):
    """Stable reason why a terminal-session snapshot cannot be returned."""

    SESSION_NOT_FOUND = "session_not_found"
    SESSION_NOT_TERMINAL = "session_not_terminal"
    SESSION_INTERRUPTED = "session_interrupted"
    INITIAL_TRANSCRIPT_INCOMPLETE = "initial_transcript_incomplete"
    TERMINAL_EVENT_MISSING = "terminal_event_missing"
    TERMINAL_EVENT_CONFLICT = "terminal_event_conflict"
    TERMINAL_EVENT_DUPLICATE = "terminal_event_duplicate"
    TERMINAL_PUBLICATION_MARKER_INVALID = "terminal_publication_marker_invalid"
    TERMINAL_PUBLICATION_MARKER_CONFLICT = "terminal_publication_marker_conflict"
    EVIDENCE_INCONSISTENT = "evidence_inconsistent"
    EVENT_LIMIT_EXCEEDED = "event_limit_exceeded"
    TRANSCRIPT_LIMIT_EXCEEDED = "transcript_limit_exceeded"
    RECORD_BYTES_EXCEEDED = "record_bytes_exceeded"
    TOTAL_BYTES_EXCEEDED = "total_bytes_exceeded"
    TRANSPORT_BYTES_EXCEEDED = "transport_bytes_exceeded"


_TERMINAL_SESSION_EVIDENCE_ERROR_MESSAGES = {
    TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND: "Session does not exist.",
    TerminalSessionEvidenceErrorCode.SESSION_NOT_TERMINAL: (
        "Session has not reached a supported terminal state."
    ),
    TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED: (
        "Interrupted sessions are not eligible terminal evidence."
    ),
    TerminalSessionEvidenceErrorCode.INITIAL_TRANSCRIPT_INCOMPLETE: (
        "The authoritative initial transcript was not fully published."
    ),
    TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_MISSING: (
        "The current run has no matching terminal event."
    ),
    TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_CONFLICT: (
        "The current terminal event contradicts the session status."
    ),
    TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_DUPLICATE: (
        "The current run has duplicate terminal events."
    ),
    TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID: (
        "The terminal-publication marker is invalid."
    ),
    TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_CONFLICT: (
        "The terminal-publication marker contradicts the terminal event."
    ),
    TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT: (
        "Terminal-session evidence contains inconsistent boundaries."
    ),
    TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED: (
        "Terminal-session evidence exceeds the event-count limit."
    ),
    TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED: (
        "Terminal-session evidence exceeds the transcript-count limit."
    ),
    TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED: (
        "Terminal-session evidence contains an oversized record."
    ),
    TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED: (
        "Terminal-session evidence exceeds the total-byte limit."
    ),
    TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED: (
        "Terminal-session evidence exceeds a backend transport safety limit."
    ),
}


class TerminalSessionEvidenceError(RuntimeError):
    """Bounded typed failure from ``load_terminal_session_evidence``."""

    def __init__(
        self,
        code: TerminalSessionEvidenceErrorCode,
        *,
        limit: int | None = None,
        observed: int | None = None,
    ) -> None:
        if not isinstance(code, TerminalSessionEvidenceErrorCode):
            raise TypeError("code must be a TerminalSessionEvidenceErrorCode.")
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit must be a non-negative integer or None.")
        if observed is not None and (type(observed) is not int or observed < 0):
            raise ValueError("observed must be a non-negative integer or None.")
        self.code = code
        self.limit = limit
        self.observed = observed
        detail = _TERMINAL_SESSION_EVIDENCE_ERROR_MESSAGES[code]
        if limit is not None:
            detail = f"{detail} Limit: {limit}."
        super().__init__(detail)


class TerminalPublicationMarker(BaseModel):
    """Bounded projection of a still-pending terminal-publication operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str
    run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return require_clean_nonblank(value, "operation_id")


class TerminalSessionEvidenceBoundary(BaseModel):
    """Completeness and working-set metadata for a terminal evidence snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    first_event_sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    terminal_event_sequence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    transcript_end_index_exclusive: StrictInt = Field(
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    event_count: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    transcript_count: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    session_bytes: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    event_bytes: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    transcript_bytes: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    terminal_publication_marker_bytes: StrictInt = Field(
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    largest_record_bytes: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    total_bytes: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    lifecycle_event_sequences: tuple[StrictInt, ...] = ()

    @field_validator("lifecycle_event_sequences")
    @classmethod
    def validate_lifecycle_event_sequences(cls, value) -> tuple[int, ...]:
        sequences = tuple(value)
        if any(
            type(sequence) is not int or sequence < 1 or sequence > MAX_DURABLE_JSON_INTEGER
            for sequence in sequences
        ):
            raise ValueError("Lifecycle event sequences must be positive durable integers.")
        if any(left >= right for left, right in pairwise(sequences)):
            raise ValueError("Lifecycle event sequences must be strictly increasing.")
        return sequences


class TerminalSessionEvidence(BaseModel):
    """One exact, bounded store snapshot through a terminal run.

    Ordinary terminal-evidence reads remain completed/failed-only. A store may
    also return this model for an interrupted session through the narrower
    runner-owned operation, which requires an exact emitted sequence/type match
    inside the same snapshot.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    session: Session
    events: tuple[EventRecord, ...]
    transcript: tuple[TranscriptRecord, ...]
    terminal_publication_marker: TerminalPublicationMarker | None = None
    boundary: TerminalSessionEvidenceBoundary

    @field_validator("session")
    @classmethod
    def copy_session(cls, value: Session) -> Session:
        return copy_session(value)

    @field_validator("events")
    @classmethod
    def copy_events(cls, value) -> tuple[EventRecord, ...]:
        return tuple(record.model_copy(deep=True) for record in value)

    @field_validator("transcript")
    @classmethod
    def copy_transcript(cls, value) -> tuple[TranscriptRecord, ...]:
        return tuple(record.model_copy(deep=True) for record in value)

    @model_validator(mode="after")
    def validate_boundaries(self) -> TerminalSessionEvidence:
        if not self.events:
            raise ValueError("Terminal evidence requires at least one event.")
        try:
            _validate_terminal_session_evidence_content(
                session=self.session,
                marker=self.terminal_publication_marker,
                terminal_record=self.events[-1],
                events=self.events,
                transcript=self.transcript,
                allow_interrupted=True,
            )
            expected_boundary = _measure_terminal_session_evidence(
                session=self.session,
                marker=self.terminal_publication_marker,
                events=self.events,
                transcript=self.transcript,
                limits=None,
            )
        except TerminalSessionEvidenceError as exc:
            raise ValueError(str(exc)) from None
        if self.boundary != expected_boundary:
            raise ValueError("Terminal evidence boundary metadata does not match its records.")
        return self

    @property
    def terminal_event(self) -> EventRecord:
        """Return the matching terminal event at the snapshot boundary."""

        return self.events[-1]

    @property
    def lifecycle_events(self) -> tuple[EventRecord, ...]:
        """Return invocation and interaction lifecycle records in durable order."""

        return tuple(
            record
            for record in self.events
            if record.event.type in _TERMINAL_SESSION_EVIDENCE_LIFECYCLE_EVENT_TYPES
        )

    @property
    def initial_interaction_id(self) -> str | None:
        """Return the first admitted interaction identity, when recorded."""

        return next(
            (
                record.event.interaction_id
                for record in self.events
                if record.event.type == EventType.INTERACTION_STARTED
            ),
            None,
        )


def copy_terminal_session_evidence(
    evidence: TerminalSessionEvidence,
) -> TerminalSessionEvidence:
    """Detach and fully revalidate one typed terminal-evidence snapshot.

    Event authority is in-process provenance carried outside the serialized
    model fields. Rebuilding the snapshot through ``model_dump`` would erase
    that authority and make an authentic store result indistinguishable from a
    caller-reconstructed document. Reconstruct each durable model explicitly
    instead: ``EventRecord`` delegates to ``copy_event``, which validates the
    public event fields while retaining authority already present on the exact
    typed event. Serialized evidence cannot manufacture that provenance.
    """

    if type(evidence) is not TerminalSessionEvidence:
        raise TypeError("Terminal evidence copy requires TerminalSessionEvidence.")
    if type(evidence.session) is not Session:
        raise TypeError("Terminal evidence session must be a Session.")

    events: list[EventRecord] = []
    for record in evidence.events:
        if type(record) is not EventRecord or type(record.event) is not Event:
            raise TypeError("Terminal evidence events must be EventRecord values.")
        events.append(EventRecord(sequence=record.sequence, event=record.event))

    transcript: list[TranscriptRecord] = []
    for record in evidence.transcript:
        if type(record) is not TranscriptRecord or type(record.message) is not Message:
            raise TypeError("Terminal evidence transcript must contain TranscriptRecord values.")
        transcript.append(
            TranscriptRecord(
                index=record.index,
                interaction_id=record.interaction_id,
                message=record.message,
            )
        )

    marker = evidence.terminal_publication_marker
    if marker is not None:
        if type(marker) is not TerminalPublicationMarker:
            raise TypeError(
                "Terminal evidence publication marker must be a TerminalPublicationMarker."
            )
        marker = TerminalPublicationMarker(
            operation_id=marker.operation_id,
            run_epoch=marker.run_epoch,
        )

    boundary = evidence.boundary
    if type(boundary) is not TerminalSessionEvidenceBoundary:
        raise TypeError("Terminal evidence boundary must be a TerminalSessionEvidenceBoundary.")
    copied_boundary = TerminalSessionEvidenceBoundary(
        first_event_sequence=boundary.first_event_sequence,
        terminal_event_sequence=boundary.terminal_event_sequence,
        transcript_end_index_exclusive=boundary.transcript_end_index_exclusive,
        run_epoch=boundary.run_epoch,
        event_count=boundary.event_count,
        transcript_count=boundary.transcript_count,
        session_bytes=boundary.session_bytes,
        event_bytes=boundary.event_bytes,
        transcript_bytes=boundary.transcript_bytes,
        terminal_publication_marker_bytes=boundary.terminal_publication_marker_bytes,
        largest_record_bytes=boundary.largest_record_bytes,
        total_bytes=boundary.total_bytes,
        lifecycle_event_sequences=boundary.lifecycle_event_sequences,
    )
    return TerminalSessionEvidence(
        session=evidence.session,
        events=tuple(events),
        transcript=tuple(transcript),
        terminal_publication_marker=marker,
        boundary=copied_boundary,
    )


_TERMINAL_PUBLICATION_EVENT_TYPE_BY_STATUS = {
    SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
    SessionStatus.FAILED: EventType.SESSION_FAILED,
    SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
}


def _copy_terminal_session_evidence_limits(
    limits: TerminalSessionEvidenceLimits | None,
) -> TerminalSessionEvidenceLimits:
    if limits is None:
        return TerminalSessionEvidenceLimits()
    if type(limits) is not TerminalSessionEvidenceLimits:
        raise TypeError("limits must be a TerminalSessionEvidenceLimits or None.")
    return limits.model_copy(deep=True)


def _terminal_session_evidence_marker_from_checkpoint(
    checkpoint: dict[str, Any] | None,
) -> TerminalPublicationMarker | None:
    if checkpoint is not None and _SESSION_RUN_OPERATION_CHECKPOINT_KEY in checkpoint:
        raw_marker = checkpoint[_SESSION_RUN_OPERATION_CHECKPOINT_KEY]
        if type(raw_marker) is not dict or type(raw_marker.get("version")) is not int:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
            )
    try:
        operation = _session_run_operation_from_checkpoint(checkpoint)
    except (TypeError, ValueError) as exc:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_INVALID
        ) from exc
    if operation is None:
        return None
    return TerminalPublicationMarker(
        operation_id=operation.operation_id,
        run_epoch=operation.run_epoch,
    )


def _copy_runner_observed_event_identities(
    session_id: str,
    value: tuple[RunnerObservedEventIdentity, ...],
    *,
    limits: TerminalSessionEvidenceLimits,
) -> tuple[RunnerObservedEventIdentity, ...]:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(value) is not tuple or not value:
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)
    if len(value) > limits.max_events:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
            limit=limits.max_events,
            observed=len(value),
        )
    copied: list[RunnerObservedEventIdentity] = []
    identity_total_bytes = 0
    for identity in value:
        if type(identity) is not RunnerObservedEventIdentity:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            )
        if (
            identity.session_id != session_id
            or type(identity.sequence) is not int
            or identity.sequence < 1
            or identity.sequence > MAX_DURABLE_JSON_INTEGER
        ):
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            )
        try:
            event_type = Event(type=identity.event_type, session_id=session_id).type
        except (TypeError, ValueError) as exc:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            ) from exc
        copied.append(
            RunnerObservedEventIdentity(
                session_id=session_id,
                sequence=identity.sequence,
                event_type=event_type,
            )
        )
        identity_bytes = len(str(event_type).encode("utf-8")) + len(
            str(identity.sequence).encode("ascii")
        )
        if identity_bytes > limits.max_record_bytes:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
                limit=limits.max_record_bytes,
                observed=identity_bytes,
            )
        identity_total_bytes += identity_bytes
        if identity_total_bytes > limits.max_total_bytes:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
                limit=limits.max_total_bytes,
                observed=identity_total_bytes,
            )
    for left, right in pairwise(copied):
        if left.sequence is None or right.sequence is None or left.sequence >= right.sequence:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            )
    return tuple(copied)


def _copy_runner_owned_interruption_proof(
    session_id: str,
    *,
    observed_events: tuple[RunnerObservedEventIdentity, ...] | None,
    expected_parent_session_id: str | None,
    limits: TerminalSessionEvidenceLimits,
    required: bool,
) -> tuple[tuple[RunnerObservedEventIdentity, ...] | None, str | None]:
    """Validate the mutually exclusive root/descendant ownership proof."""

    if not required:
        if observed_events is not None or expected_parent_session_id is not None:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            )
        return None, None
    if (observed_events is None) == (expected_parent_session_id is None):
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)
    if observed_events is not None:
        return (
            _copy_runner_observed_event_identities(
                session_id,
                observed_events,
                limits=limits,
            ),
            None,
        )
    if expected_parent_session_id is None:
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)
    try:
        parent_session_id = require_clean_nonblank(
            expected_parent_session_id,
            "expected_parent_session_id",
        )
    except (TypeError, ValueError) as exc:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
        ) from exc
    if parent_session_id == session_id:
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)
    return None, parent_session_id


def _validate_runner_observed_event_identity_snapshot(
    expected: tuple[RunnerObservedEventIdentity, ...],
    actual: Sequence[RunnerObservedEventIdentity],
) -> None:
    if tuple(actual) != expected:
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)


def _terminal_session_evidence_expected_event_type(
    status: SessionStatus,
    *,
    allow_interrupted: bool = False,
) -> EventType:
    if status == SessionStatus.INTERRUPTED and not allow_interrupted:
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED)
    expected = _TERMINAL_PUBLICATION_EVENT_TYPE_BY_STATUS.get(status)
    allowed = {EventType.SESSION_COMPLETED, EventType.SESSION_FAILED}
    if allow_interrupted:
        allowed.add(EventType.SESSION_INTERRUPTED)
    if expected not in allowed:
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.SESSION_NOT_TERMINAL)
    return expected


def _classify_terminal_session_evidence_records(
    *,
    session_id: str,
    status: SessionStatus,
    run_epoch: int,
    marker: TerminalPublicationMarker | None,
    newest_evidence_records: Sequence[EventRecord],
    initial_transcript_pending: bool,
    pending_session_interrupt: bool,
    allow_interrupted: bool = False,
) -> EventRecord:
    """Validate the bounded current-run evidence and return its terminal record.

    ``newest_evidence_records`` contains at most the two newest session
    lifecycle/terminal records in descending durable-sequence order. Two are
    sufficient to distinguish a current terminal record, a newer lifecycle
    boundary, and a duplicate current-run publication.
    """

    expected_event_type = _terminal_session_evidence_expected_event_type(
        status,
        allow_interrupted=allow_interrupted,
    )
    if len(newest_evidence_records) > TERMINAL_EVIDENCE_QUERY_LIMIT:
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)
    if any(left.sequence <= right.sequence for left, right in pairwise(newest_evidence_records)):
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)
    if any(record.event.session_id != session_id for record in newest_evidence_records):
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)
    if initial_transcript_pending:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.INITIAL_TRANSCRIPT_INCOMPLETE
        )
    if pending_session_interrupt:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_CONFLICT
        )
    if marker is not None and marker.run_epoch != run_epoch:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_CONFLICT
        )

    events_after_lifecycle: list[EventRecord] = []
    for record in newest_evidence_records:
        if record.event.type in {
            EventType.SESSION_STARTED,
            EventType.SESSION_RESUMED,
            EventType.SESSION_FORKED,
        }:
            break
        events_after_lifecycle.append(record)

    current_operation_id = (
        marker.operation_id
        if marker is not None
        else next(
            (
                record.event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
                for record in events_after_lifecycle
            ),
            None,
        )
    )
    if any(
        record.event.type != expected_event_type
        and (record.event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY) == current_operation_id)
        for record in events_after_lifecycle
    ):
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_CONFLICT)
    if (
        marker is not None
        and events_after_lifecycle
        and events_after_lifecycle[0].event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
        != marker.operation_id
    ):
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_CONFLICT
        )

    try:
        classification = classify_current_terminal_evidence(
            evidence_events=tuple(record.event for record in newest_evidence_records),
            expected_event_type=expected_event_type,
            run_operation_id=None if marker is None else marker.operation_id,
            interruption_request_id=None,
        )
    except (TypeError, ValueError) as exc:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
        ) from exc
    if classification.run_operation_conflict:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_CONFLICT
        )
    if not classification.events:
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_MISSING)
    if len(classification.events) > 1:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_DUPLICATE
        )
    terminal_event_id = classification.events[0].id
    for record in newest_evidence_records:
        if record.event.id == terminal_event_id:
            return record
    raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)


def _terminal_session_evidence_record_bytes(value: BaseModel) -> int:
    return compact_json_utf8_size(value.model_dump(mode="json"))


def _validate_terminal_session_evidence_content(
    *,
    session: Session,
    marker: TerminalPublicationMarker | None,
    terminal_record: EventRecord,
    events: Sequence[EventRecord],
    transcript: Sequence[TranscriptRecord],
    allow_interrupted: bool = False,
) -> None:
    """Prove that the complete bounded prefix represents one terminal run."""

    expected_event_type = _terminal_session_evidence_expected_event_type(
        session.status,
        allow_interrupted=allow_interrupted,
    )
    if not events or (
        events[-1].sequence != terminal_record.sequence
        or events[-1].event.id != terminal_record.event.id
    ):
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)
    if any(left.sequence >= right.sequence for left, right in pairwise(events)) or any(
        record.event.session_id != session.id for record in events
    ):
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)
    if any(record.index != index for index, record in enumerate(transcript)):
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)

    current_run_start = 0
    for index, record in enumerate(events):
        if record.event.type in TERMINAL_LIFECYCLE_EVENT_TYPES:
            current_run_start = index + 1
    current_terminal_records = tuple(
        record for record in events[current_run_start:] if record.event.type in TERMINAL_EVENT_TYPES
    )
    if not current_terminal_records:
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_MISSING)
    if any(record.event.type != expected_event_type for record in current_terminal_records):
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_CONFLICT)
    if len(current_terminal_records) > 1:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TERMINAL_EVENT_DUPLICATE
        )
    current_terminal = current_terminal_records[0]
    if (
        current_terminal.sequence != terminal_record.sequence
        or current_terminal.event.id != terminal_record.event.id
    ):
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)

    operation_id = current_terminal.event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
    if operation_id is not None:
        try:
            operation_id = require_clean_nonblank(
                operation_id,
                "terminal event session_run_operation_id",
            )
        except (TypeError, ValueError) as exc:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            ) from exc
    if marker is not None and (
        marker.run_epoch != session.run_epoch or operation_id != marker.operation_id
    ):
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TERMINAL_PUBLICATION_MARKER_CONFLICT
        )

    lifecycle_interaction_ids = {
        record.event.interaction_id
        for record in events
        if record.event.type in _TERMINAL_SESSION_EVIDENCE_INTERACTION_LIFECYCLE_EVENT_TYPES
        and record.event.interaction_id is not None
    }
    if any(
        record.interaction_id is not None and record.interaction_id not in lifecycle_interaction_ids
        for record in transcript
    ):
        raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)


def _terminal_session_evidence_checked_record_bytes(
    value: BaseModel,
    *,
    limits: TerminalSessionEvidenceLimits,
    prior_total_bytes: int,
) -> tuple[int, int]:
    record_bytes = _terminal_session_evidence_record_bytes(value)
    if record_bytes > limits.max_record_bytes:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
            limit=limits.max_record_bytes,
            observed=record_bytes,
        )
    total_bytes = prior_total_bytes + record_bytes
    if total_bytes > limits.max_total_bytes:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
            limit=limits.max_total_bytes,
            observed=total_bytes,
        )
    return record_bytes, total_bytes


_TERMINAL_SESSION_EVIDENCE_TOP_LEVEL_FIELDS = (
    "session",
    "events",
    "transcript",
    "terminal_publication_marker",
    "boundary",
)


_TERMINAL_SESSION_EVIDENCE_TOP_LEVEL_ENVELOPE_BYTES = (
    2
    + len(_TERMINAL_SESSION_EVIDENCE_TOP_LEVEL_FIELDS)
    - 1
    + sum(
        compact_json_utf8_size(field_name) + 1
        for field_name in _TERMINAL_SESSION_EVIDENCE_TOP_LEVEL_FIELDS
    )
)


def _terminal_session_evidence_array_bytes(record_bytes: int, record_count: int) -> int:
    return 2 + record_bytes + max(0, record_count - 1)


def _terminal_session_evidence_boundary_with_exact_total(
    *,
    session: Session,
    marker_bytes: int,
    event_count: int,
    transcript_count: int,
    session_bytes: int,
    event_bytes: int,
    transcript_bytes: int,
    largest_record_bytes: int,
    lifecycle_event_sequences: tuple[int, ...],
    first_event_sequence: int,
    terminal_event_sequence: int,
) -> TerminalSessionEvidenceBoundary:
    fixed_snapshot_bytes = (
        _TERMINAL_SESSION_EVIDENCE_TOP_LEVEL_ENVELOPE_BYTES
        + session_bytes
        + _terminal_session_evidence_array_bytes(event_bytes, event_count)
        + _terminal_session_evidence_array_bytes(transcript_bytes, transcript_count)
        + (marker_bytes if marker_bytes else len("null"))
    )
    total_bytes = 1
    for _ in range(len(str(MAX_DURABLE_JSON_INTEGER)) + 1):
        boundary = TerminalSessionEvidenceBoundary(
            first_event_sequence=first_event_sequence,
            terminal_event_sequence=terminal_event_sequence,
            transcript_end_index_exclusive=transcript_count,
            run_epoch=session.run_epoch,
            event_count=event_count,
            transcript_count=transcript_count,
            session_bytes=session_bytes,
            event_bytes=event_bytes,
            transcript_bytes=transcript_bytes,
            terminal_publication_marker_bytes=marker_bytes,
            largest_record_bytes=largest_record_bytes,
            total_bytes=total_bytes,
            lifecycle_event_sequences=lifecycle_event_sequences,
        )
        measured_total_bytes = fixed_snapshot_bytes + _terminal_session_evidence_record_bytes(
            boundary
        )
        if measured_total_bytes == total_bytes:
            return boundary
        if measured_total_bytes > MAX_DURABLE_JSON_INTEGER:
            raise TerminalSessionEvidenceError(
                TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT
            )
        total_bytes = measured_total_bytes
    raise TerminalSessionEvidenceError(TerminalSessionEvidenceErrorCode.EVIDENCE_INCONSISTENT)


def _measure_terminal_session_evidence(
    *,
    session: Session,
    marker: TerminalPublicationMarker | None,
    events: Sequence[EventRecord],
    transcript: Sequence[TranscriptRecord],
    limits: TerminalSessionEvidenceLimits | None,
) -> TerminalSessionEvidenceBoundary:
    record_total_bytes = 0

    def measure_record(value: BaseModel) -> int:
        nonlocal record_total_bytes
        if limits is None:
            record_bytes = _terminal_session_evidence_record_bytes(value)
            record_total_bytes += record_bytes
            return record_bytes
        record_bytes, record_total_bytes = _terminal_session_evidence_checked_record_bytes(
            value,
            limits=limits,
            prior_total_bytes=record_total_bytes,
        )
        return record_bytes

    session_bytes = measure_record(session)
    largest_record_bytes = session_bytes
    event_bytes = 0
    for record in events:
        record_bytes = measure_record(record)
        event_bytes += record_bytes
        largest_record_bytes = max(largest_record_bytes, record_bytes)
    transcript_bytes = 0
    for record in transcript:
        record_bytes = measure_record(record)
        transcript_bytes += record_bytes
        largest_record_bytes = max(largest_record_bytes, record_bytes)
    marker_bytes = 0
    if marker is not None:
        marker_bytes = measure_record(marker)
        largest_record_bytes = max(largest_record_bytes, marker_bytes)

    lifecycle_event_sequences = tuple(
        record.sequence
        for record in events
        if record.event.type in _TERMINAL_SESSION_EVIDENCE_LIFECYCLE_EVENT_TYPES
    )
    boundary = _terminal_session_evidence_boundary_with_exact_total(
        session=session,
        marker_bytes=marker_bytes,
        event_count=len(events),
        transcript_count=len(transcript),
        session_bytes=session_bytes,
        event_bytes=event_bytes,
        transcript_bytes=transcript_bytes,
        largest_record_bytes=largest_record_bytes,
        lifecycle_event_sequences=lifecycle_event_sequences,
        first_event_sequence=events[0].sequence,
        terminal_event_sequence=events[-1].sequence,
    )
    if limits is not None and boundary.total_bytes > limits.max_total_bytes:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
            limit=limits.max_total_bytes,
            observed=boundary.total_bytes,
        )
    return boundary


def _assemble_terminal_session_evidence(
    *,
    session: Session,
    marker: TerminalPublicationMarker | None,
    terminal_record: EventRecord,
    events: Sequence[EventRecord],
    transcript: Sequence[TranscriptRecord],
    limits: TerminalSessionEvidenceLimits,
    allow_interrupted: bool = False,
) -> TerminalSessionEvidence:
    """Apply the backend-neutral completeness and exact-byte contract."""

    limits = _copy_terminal_session_evidence_limits(limits)
    if len(events) > limits.max_events:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
            limit=limits.max_events,
            observed=len(events),
        )
    if len(transcript) > limits.max_transcript_records:
        raise TerminalSessionEvidenceError(
            TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
            limit=limits.max_transcript_records,
            observed=len(transcript),
        )
    _validate_terminal_session_evidence_content(
        session=session,
        marker=marker,
        terminal_record=terminal_record,
        events=events,
        transcript=transcript,
        allow_interrupted=allow_interrupted,
    )
    boundary = _measure_terminal_session_evidence(
        session=session,
        marker=marker,
        events=events,
        transcript=transcript,
        limits=limits,
    )
    # The complete content and its derived boundary were validated and sized
    # above. Construct a detached result without asking the public model
    # validator to repeat an O(snapshot) byte walk.
    return TerminalSessionEvidence.model_construct(
        session=copy_session(session),
        events=tuple(record.model_copy(deep=True) for record in events),
        transcript=tuple(record.model_copy(deep=True) for record in transcript),
        terminal_publication_marker=(None if marker is None else marker.model_copy(deep=True)),
        boundary=boundary.model_copy(deep=True),
    )
