"""Private, bounded indexed backing for terminal evidence; never load external spools."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import sqlite3
import tempfile
import threading
import time
from collections.abc import Awaitable, Iterator, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar, cast, overload

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cayu._validation import canonical_durable_json_bytes
from cayu.core.events import Event, EventType
from cayu.runtime.sessions import (
    _TERMINAL_SESSION_EVIDENCE_LIFECYCLE_EVENT_TYPES,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
    EventRecord,
    Session,
    TerminalPublicationMarker,
    TerminalSessionEvidence,
    TerminalSessionEvidenceBoundary,
    TranscriptRecord,
    _measure_terminal_session_evidence,
    _validate_terminal_session_evidence_content,
)

T = TypeVar("T", bound=BaseModel)


_settle_evidence_reads: ContextVar[bool] = ContextVar("cayu_settle_evidence_reads", default=False)


@contextmanager
def _settled_evidence_reads():
    """Keep admission ownership until native read workers physically settle."""
    token = _settle_evidence_reads.set(True)
    try:
        yield
    finally:
        _settle_evidence_reads.reset(token)


def _settled_evidence_reads_required() -> bool:
    return _settle_evidence_reads.get()


class IncrementalEvidenceLimits(BaseModel):
    """Work and disk limits, independent of the eager materialization ceiling."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    max_root_bytes: StrictInt = Field(default=4_194_304, ge=1, le=16_777_216)
    max_root_events: StrictInt = Field(default=10_000, ge=1, le=10_000)
    max_sessions: StrictInt = Field(default=100, ge=1, le=500)
    max_depth: StrictInt = Field(default=32, ge=1, le=32)
    max_events: StrictInt = Field(default=1_000_000, ge=1, le=100_000_000)
    max_transcript_records: StrictInt = Field(default=100_000, ge=0, le=10_000_000)
    max_record_bytes: StrictInt = Field(
        default=1_048_576, ge=1, le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES
    )
    max_total_bytes: StrictInt = Field(default=268_435_456, ge=1, le=4_294_967_296)
    max_spill_bytes: StrictInt = Field(default=536_870_912, ge=12288, le=8_589_934_592)
    batch_records: StrictInt = Field(default=4, ge=1, le=256)
    max_lifecycle_bytes: StrictInt = Field(default=2_097_152, ge=1, le=16_777_216)
    max_lifecycle_records: StrictInt = Field(default=10_000, ge=1, le=100_000)
    max_seconds: StrictInt = Field(default=300, ge=1, le=3600)


class IncrementalEvidenceError(RuntimeError):
    """Payload-free resource or capability failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class _Records(Sequence[T], Generic[T]):
    def __init__(self, owner: EvidenceSpool, kind: str, indices: range) -> None:
        self.owner, self.kind, self.indices = owner, kind, indices

    def __len__(self) -> int:
        return len(self.indices)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> _Records[T]: ...

    def __getitem__(self, index: int | slice) -> T | _Records[T]:
        if isinstance(index, slice):
            return _Records(self.owner, self.kind, self.indices[index])
        self.owner.check()
        row = self.owner._db.execute(
            "SELECT CASE WHEN length(payload)<=? THEN payload END, "
            "CASE WHEN length(tag)=32 THEN tag END FROM records WHERE kind=? AND position=?",
            (self.owner._max_encoded_record_bytes, self.kind, self.indices[index]),
        ).fetchone()
        if row is None:
            raise IncrementalEvidenceError("spill_record_missing")
        return cast("T", self.owner._decode(self.kind, self.indices[index], row[0], row[1]))

    def __iter__(self) -> Iterator[T]:
        self.owner.check()
        if not self.indices:
            return
        if self.indices.step != 1:
            for index in range(len(self)):
                yield self[index]
            return
        cursor = self.owner._db.execute(
            "SELECT position, CASE WHEN length(payload)<=? THEN payload END, "
            "CASE WHEN length(tag)=32 THEN tag END FROM records WHERE kind=? "
            "AND position>=? AND position<? ORDER BY position",
            (
                self.owner._max_encoded_record_bytes,
                self.kind,
                self.indices.start,
                self.indices.stop,
            ),
        )
        expected = self.indices.start
        try:
            for position, payload, tag in cursor:
                self.owner.check()
                if position != expected:
                    raise IncrementalEvidenceError("spill_record_missing")
                expected += 1
                yield cast("T", self.owner._decode(self.kind, position, payload, tag))
            if expected != self.indices.stop:
                raise IncrementalEvidenceError("spill_record_missing")
        finally:
            if not self.owner._closed:
                cursor.close()


def _event_authority(event: Event) -> dict:
    return {
        "id_origin": event._id_origin,
        "runtime_generated_id": event._runtime_generated_id,
        "payload": [list(item) for item in sorted(event._runtime_payload_authority)],
        "nested_payload": [
            [list(path), value] for path, value in sorted(event._runtime_nested_payload_authority)
        ],
        "envelope": [list(item) for item in sorted(event._runtime_envelope_authority)],
    }


def _update_record_digest(digest, value: BaseModel, data: bytes) -> None:
    digest.update(len(data).to_bytes(8, "big"))
    digest.update(data)
    if isinstance(value, EventRecord):
        authority = canonical_durable_json_bytes(
            _event_authority(value.event), "event authority seal"
        )
        digest.update(len(authority).to_bytes(8, "big"))
        digest.update(authority)


def _incremental_evidence_sha256(evidence: TerminalSessionEvidence) -> str:
    """Reference seal over an eager fixture, including native authority proofs."""
    digest = hashlib.sha256(b"cayu.incremental-terminal-evidence.v1\0")
    for records in (evidence.events, evidence.transcript):
        for record in records:
            data = canonical_durable_json_bytes(record.model_dump(mode="json"), "evidence record")
            _update_record_digest(digest, record, data)
    for record in (evidence.session, evidence.terminal_publication_marker):
        if record is not None:
            data = canonical_durable_json_bytes(record.model_dump(mode="json"), "evidence record")
            _update_record_digest(digest, record, data)
    return digest.hexdigest()


class EvidenceSpool:
    """Owned snapshot with one-record reads and a bounded SQLite page cache.

    Use only as a context manager. The database transaction ends before returning
    the sealed snapshot. No producer queue or source cursor survives capture.
    """

    def __init__(self, limits: IncrementalEvidenceLimits) -> None:
        self.limits = IncrementalEvidenceLimits.model_validate(limits.model_dump())
        self._directory = tempfile.TemporaryDirectory(prefix="cayu-evidence-")
        self.path = Path(self._directory.name) / "evidence.sqlite3"
        database = None
        try:
            self._db = sqlite3.connect(self.path, check_same_thread=False)
            self._db.execute("PRAGMA page_size=4096")
            self._db.execute("PRAGMA journal_mode=OFF")
            self._db.execute("PRAGMA cache_size=-256")
            self._db.execute("PRAGMA temp_store=FILE")
            self._db.execute(f"PRAGMA max_page_count={limits.max_spill_bytes // 4096}")
            self._db.execute(
                "CREATE TABLE records(kind TEXT, position INTEGER, payload BLOB, identity TEXT, tag BLOB, "
                "PRIMARY KEY(kind, position), UNIQUE(kind, identity)) WITHOUT ROWID"
            )
        except (OSError, sqlite3.Error) as exc:
            database = getattr(self, "_db", None)
            if database is not None:
                database.close()
            self._directory.cleanup()
            raise IncrementalEvidenceError("spill_initialization_failed") from exc
        self._counts = {"event": 0, "transcript": 0}
        self._key = secrets.token_bytes(32)
        self._max_encoded_record_bytes = 3 * limits.max_record_bytes + 4096
        self._lifecycle_count = 0
        self._lifecycle_bytes = 0
        self.processed_bytes = 0
        self.peak_record_bytes = 0
        self.peak_page_transport_bytes = 0
        self.peak_page_records = 0
        self.spill_records_read = 0
        self._digest = hashlib.sha256(b"cayu.incremental-terminal-evidence.v1\0")
        self._deadline = time.monotonic() + limits.max_seconds
        self._cancelled = threading.Event()
        self._closed = False
        self.phase = "snapshot_copy"
        self.session: Session
        self.terminal_publication_marker: TerminalPublicationMarker | None
        self.boundary: TerminalSessionEvidenceBoundary
        self.evidence_sha256: str

    def check(self) -> None:
        if self._closed or self._cancelled.is_set():
            raise IncrementalEvidenceError("capture_cancelled")
        if time.monotonic() >= self._deadline:
            raise IncrementalEvidenceError("capture_deadline_exceeded")

    def should_interrupt(self) -> bool:
        return self._cancelled.is_set() or time.monotonic() >= self._deadline

    def cancel(self) -> None:
        self._cancelled.set()

    def _measure(self, value: BaseModel) -> bytes:
        self.check()
        data = canonical_durable_json_bytes(value.model_dump(mode="json"), "evidence record")
        size = len(data)
        if size > self.limits.max_record_bytes:
            raise IncrementalEvidenceError("record_bytes_exceeded")
        self.processed_bytes += size
        self.peak_record_bytes = max(self.peak_record_bytes, size)
        if self.processed_bytes > self.limits.max_total_bytes:
            raise IncrementalEvidenceError("total_bytes_exceeded")
        _update_record_digest(self._digest, value, data)
        return data

    def observe_page(self, *, records: int, transport_bytes: int) -> None:
        self.check()
        if records > self.limits.batch_records:
            raise IncrementalEvidenceError("page_records_exceeded")
        if transport_bytes > self.limits.batch_records * (
            (3 * self.limits.max_record_bytes + 1) // 2
        ):
            raise IncrementalEvidenceError("page_transport_exceeded")
        self.peak_page_records = max(self.peak_page_records, records)
        self.peak_page_transport_bytes = max(self.peak_page_transport_bytes, transport_bytes)

    def _mac(self, kind: str, position: int, payload: bytes) -> bytes:
        signature = hmac.new(self._key, digestmod="sha256")
        signature.update(kind.encode("ascii"))
        signature.update(position.to_bytes(8, "big"))
        signature.update(payload)
        return signature.digest()

    def _decode(
        self, kind: str, position: int, payload: bytes | None, tag: bytes | None
    ) -> EventRecord | TranscriptRecord:
        self.check()
        if (
            payload is None
            or tag is None
            or not hmac.compare_digest(tag, self._mac(kind, position, payload))
        ):
            raise IncrementalEvidenceError("spill_integrity_rejected")
        document = json.loads(payload)
        self.spill_records_read += 1
        if kind == "transcript":
            return TranscriptRecord.model_validate(document["record"])
        fields = document["record"]["event"]
        fields["timestamp"] = datetime.fromisoformat(fields["timestamp"])
        # Custom namespaces were already validated by the native writer.
        with suppress(ValueError):
            fields["type"] = EventType(fields["type"])
        # Only this object's bounded, typed writer can authenticate these bytes.
        # Restore the already-validated record without repeated payload validation.
        event = Event.model_construct(**fields)
        authority = document["authority"]
        event._id_origin = authority["id_origin"]
        event._runtime_generated_id = authority["runtime_generated_id"]
        event._runtime_payload_authority = frozenset(tuple(item) for item in authority["payload"])
        event._runtime_nested_payload_authority = frozenset(
            (tuple(path), value) for path, value in authority["nested_payload"]
        )
        event._runtime_envelope_authority = frozenset(tuple(item) for item in authority["envelope"])
        return EventRecord.model_construct(sequence=document["record"]["sequence"], event=event)

    def append(self, kind: str, record: EventRecord | TranscriptRecord) -> None:
        data = self._measure(record)
        count = self._counts[kind]
        limit = self.limits.max_events if kind == "event" else self.limits.max_transcript_records
        if count >= limit:
            raise IncrementalEvidenceError(f"{kind}_count_exceeded")
        if (
            isinstance(record, EventRecord)
            and record.event.type in _TERMINAL_SESSION_EVIDENCE_LIFECYCLE_EVENT_TYPES
        ):
            self._lifecycle_count += 1
            self._lifecycle_bytes += len(data)
            if (
                self._lifecycle_count > self.limits.max_lifecycle_records
                or self._lifecycle_bytes > self.limits.max_lifecycle_bytes
            ):
                raise IncrementalEvidenceError("lifecycle_bookkeeping_exceeded")
        if (kind == "event" and type(record) is not EventRecord) or (
            kind == "transcript" and type(record) is not TranscriptRecord
        ):
            raise TypeError("Spill records must match their declared kind.")
        if isinstance(record, EventRecord):
            record = EventRecord(sequence=record.sequence, event=record.event)
        else:
            record = TranscriptRecord.model_validate(record.model_dump(mode="python"))
        packet = {"record": record.model_dump(mode="json")}
        if isinstance(record, EventRecord):
            packet["authority"] = _event_authority(record.event)
        encoded = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > self._max_encoded_record_bytes:
            raise IncrementalEvidenceError("spill_record_bytes_exceeded")
        try:
            self._db.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                (
                    kind,
                    count,
                    encoded,
                    record.event.id if isinstance(record, EventRecord) else str(record.index),
                    self._mac(kind, count, encoded),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise IncrementalEvidenceError("duplicate_record_identity") from exc
        except sqlite3.OperationalError as exc:
            raise IncrementalEvidenceError("spill_write_failed") from exc
        self._counts[kind] += 1

    @property
    def events(self) -> Sequence[EventRecord]:
        return _Records(self, "event", range(self._counts["event"]))

    @property
    def transcript(self) -> Sequence[TranscriptRecord]:
        return _Records(self, "transcript", range(self._counts["transcript"]))

    def stage(
        self,
        session: Session,
        marker: TerminalPublicationMarker | None,
        terminal_record: EventRecord,
        event_count: int,
        transcript_count: int,
    ) -> None:
        if self._counts != {"event": event_count, "transcript": transcript_count}:
            raise IncrementalEvidenceError("snapshot_count_mismatch")
        self.session = session.model_copy(deep=True)
        self.terminal_publication_marker = marker
        self._measure(session)
        if marker is not None:
            self._measure(marker)
        self._terminal_record = terminal_record

    def seal(self) -> None:
        self.phase = "snapshot_validation"
        session = self.session
        marker = self.terminal_publication_marker
        terminal_record = self._terminal_record
        _validate_terminal_session_evidence_content(
            session=session,
            marker=marker,
            terminal_record=terminal_record,
            events=self.events,
            transcript=self.transcript,
        )
        self.boundary = _measure_terminal_session_evidence(
            session=session,
            marker=marker,
            events=self.events,
            transcript=self.transcript,
            limits=None,
        )
        if self.boundary.total_bytes > self.limits.max_total_bytes:
            raise IncrementalEvidenceError("total_bytes_exceeded")
        self._db.commit()
        self.evidence_sha256 = self._digest.hexdigest()
        self.phase = "snapshot_complete"

    async def seal_async(self) -> None:
        await self.run_owned(asyncio.to_thread(self.seal))

    async def run_owned(self, operation: Awaitable[object]) -> None:
        """Do not release the backing while a cancelled worker still owns it."""
        task = asyncio.ensure_future(operation)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            self.cancel()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if task.done() and not task.cancelled():
                task.exception()
            raise

    def __enter__(self) -> EvidenceSpool:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._db.close()
            self._directory.cleanup()


class IncrementalEvidenceAdmission:
    """Explicit shared fail-fast admission; no hidden queue or retry scheduler.

    Share one instance across all captures on a worker. The payload envelope is
    serialized bytes, not an assertion about Python allocator overhead or RSS.
    """

    def __init__(
        self,
        *,
        max_captures: int = 8,
        max_buffer_bytes: int = 512 * 1024 * 1024,
        max_spill_bytes: int = 4 * 1024 * 1024 * 1024,
    ) -> None:
        for value in (max_captures, max_buffer_bytes, max_spill_bytes):
            if type(value) is not int or value < 1:
                raise ValueError("Admission budgets must be positive integers.")
        self.max_captures = max_captures
        self.max_buffer_bytes = max_buffer_bytes
        self.max_spill_bytes = max_spill_bytes
        self.active_captures = 0
        self.reserved_buffer_bytes = 0
        self.reserved_spill_bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def buffer_envelope(limits: IncrementalEvidenceLimits) -> int:
        # Two adjacent transport pages, decoded/serialized current records,
        # metadata records and the private SQLite page cache. PostgreSQL has a
        # separate 3:2 transport guard. Object/allocator expansion is measured RSS.
        return (
            (3 * limits.batch_records + 8) * limits.max_record_bytes
            + 256 * 1024
            + limits.max_lifecycle_records * 64
            + 2 * limits.max_lifecycle_bytes
            + 500 * 16_384
            + 8 * limits.max_root_bytes
        )

    def reserve(self, limits: IncrementalEvidenceLimits) -> _EvidenceReservation:
        limits = IncrementalEvidenceLimits.model_validate(limits.model_dump())
        buffered = self.buffer_envelope(limits)
        with self._lock:
            if (
                self.active_captures >= self.max_captures
                or self.reserved_buffer_bytes + buffered > self.max_buffer_bytes
                or self.reserved_spill_bytes + limits.max_spill_bytes > self.max_spill_bytes
            ):
                raise IncrementalEvidenceError("aggregate_admission_rejected")
            self.active_captures += 1
            self.reserved_buffer_bytes += buffered
            self.reserved_spill_bytes += limits.max_spill_bytes
        return _EvidenceReservation(self, buffered, limits.max_spill_bytes)


class _EvidenceReservation:
    def __init__(self, admission: IncrementalEvidenceAdmission, buffered: int, spill: int) -> None:
        self._admission, self._buffered, self._spill = admission, buffered, spill
        self._closed = False

    def __enter__(self) -> _EvidenceReservation:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        admission = self._admission
        with admission._lock:
            if not self._closed:
                self._closed = True
                admission.active_captures -= 1
                admission.reserved_buffer_bytes -= self._buffered
                admission.reserved_spill_bytes -= self._spill
