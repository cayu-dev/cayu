"""Bounded, consistent per-session state for portable exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from cayu._validation import MAX_DURABLE_JSON_INTEGER, canonical_durable_json_bytes
from cayu.core import EventType
from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint
from cayu.runtime.checkpoints import decode_runtime_checkpoint
from cayu.runtime.sessions import (
    DeferredInteractionInput,
    EventRecord,
    Session,
    TranscriptRecord,
)
from cayu.runtime.tool_grants import TargetedToolGrantStateSnapshot

SESSION_EXPORT_PAGE_SIZE = 256


class SessionExportLimits(BaseModel):
    """Explicit output ceilings; an oversized session is rejected, never truncated."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    max_bytes: StrictInt = Field(default=64 * 1024 * 1024, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    max_record_bytes: StrictInt = Field(default=8 * 1024 * 1024, ge=1, le=MAX_DURABLE_JSON_INTEGER)


class SessionExportTooLarge(ValueError):
    """One session cannot be exported within the caller's explicit byte bounds."""


class SessionExportBoundary(BaseModel):
    """Source-store positions captured together with every export-owned component."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    event_sequences: tuple[StrictInt, ...]
    through_sequence: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)


def validate_export_ownership(
    session: Session,
    events: list,
    deferred: DeferredInteractionInput | None,
    grants: TargetedToolGrantStateSnapshot,
) -> None:
    if any(record.session_id != session.id for record in grants.records):
        raise ValueError("Export grant state belongs to another session.")
    if deferred is not None and not any(
        event.type == EventType.INTERACTION_STARTED
        and event.interaction_id == deferred.interaction_id
        for event in events
    ):
        raise ValueError("Export deferred input has no matching interaction admission.")


def validate_export_boundary(
    *,
    session: Session,
    events: list,
    transcript_records: list[TranscriptRecord],
    checkpoint: dict[str, Any] | None,
    boundary: SessionExportBoundary,
) -> None:
    sequences = boundary.event_sequences
    if len(sequences) != len(events) or any(
        type(sequence) is not int or not 1 <= sequence <= MAX_DURABLE_JSON_INTEGER
        for sequence in sequences
    ):
        raise ValueError("Export event positions do not match its event prefix.")
    if any(right <= left for left, right in pairwise(sequences)):
        raise ValueError("Export event sequences must be strictly increasing.")
    if boundary.through_sequence != (sequences[-1] if sequences else 0):
        raise ValueError("Export event watermark does not match its included prefix.")
    by_id = {}
    for event in events:
        if event.session_id != session.id or event.id in by_id:
            raise ValueError("Export events have conflicting session or event identities.")
        by_id[event.id] = event
    previous_index = -1
    for record in transcript_records:
        if not previous_index < record.index < boundary.transcript_cursor:
            raise ValueError("Export transcript records exceed or conflict with its cursor.")
        previous_index = record.index
    if checkpoint is None:
        return

    def cursor(value: object) -> None:
        if value is not None and (
            type(value) is not int or not 0 <= value <= boundary.transcript_cursor
        ):
            raise ValueError("Export checkpoint references a transcript cursor beyond its prefix.")

    # These fields are runtime-owned cursor contracts. Arbitrary application
    # payloads and provider/tool results remain opaque, even if their keys look
    # like runtime references.
    cursor(checkpoint.get("compacted_transcript_cursor"))
    for root, fields in {
        "context_compaction": (
            "compacted_transcript_cursor",
            "previous_compacted_transcript_cursor",
        ),
        "usage_triggered_context": ("last_transcript_cursor",),
        "pending_tool_round": ("source_transcript_cursor",),
    }.items():
        value = checkpoint.get(root)
        if isinstance(value, dict):
            for field in fields:
                cursor(value.get(field))
    pointer = model_step_publication_from_checkpoint(checkpoint)
    if pointer is not None:
        cursor(pointer.source_transcript_cursor)
        cursor(pointer.transcript_end_cursor)
        completion = by_id.get(pointer.completion_event_id)
        if completion is None or completion.type != EventType.MODEL_COMPLETED:
            raise ValueError("Export checkpoint references an absent model completion event.")
    operations = checkpoint.get("session_operations")
    if isinstance(operations, dict) and isinstance(operations.get("records"), dict):
        for operation in operations["records"].values():
            if not isinstance(operation, dict):
                raise ValueError("Export checkpoint operation record is malformed.")
            cursor(operation.get("source_transcript_cursor"))
            cursor(operation.get("result_transcript_cursor"))
            event_ids = operation.get("event_ids", [])
            if type(event_ids) is not list or any(
                type(event_id) is not str or event_id not in by_id for event_id in event_ids
            ):
                raise ValueError("Export checkpoint operation references an absent event.")


@dataclass(frozen=True)
class SessionExportSnapshot:
    session: Session
    events: tuple[EventRecord, ...]
    transcript_records: tuple[TranscriptRecord, ...]
    checkpoint: dict[str, Any] | None
    deferred_interaction_input: DeferredInteractionInput | None
    targeted_tool_grant_state: TargetedToolGrantStateSnapshot
    boundary: SessionExportBoundary

    def document(self) -> dict[str, Any]:
        checkpoint = decode_runtime_checkpoint(self.checkpoint, session_id=self.session.id)
        validate_export_ownership(
            self.session,
            [record.event for record in self.events],
            self.deferred_interaction_input,
            self.targeted_tool_grant_state,
        )
        validate_export_boundary(
            session=self.session,
            events=[record.event for record in self.events],
            transcript_records=list(self.transcript_records),
            checkpoint=checkpoint,
            boundary=self.boundary,
        )
        return {
            "type": "session",
            "format_version": 2,
            "snapshot": self.boundary.model_dump(mode="json"),
            "session": self.session.model_dump(mode="json"),
            "events": [record.event.model_dump(mode="json") for record in self.events],
            "transcript_records": [
                record.model_dump(mode="json") for record in self.transcript_records
            ],
            "checkpoint": checkpoint,
            "deferred_interaction_input": (
                None
                if self.deferred_interaction_input is None
                else self.deferred_interaction_input.model_dump(mode="json")
            ),
            "targeted_tool_grant_state": self.targeted_tool_grant_state.model_dump(mode="json"),
        }


class SessionExportBuilder:
    """Charge each component before retaining it; store readers feed bounded pages."""

    def __init__(self, limits: SessionExportLimits | None) -> None:
        if limits is not None and type(limits) is not SessionExportLimits:
            raise TypeError("limits must be SessionExportLimits.")
        self.limits = (
            SessionExportLimits()
            if limits is None
            else SessionExportLimits.model_validate(limits.model_dump())
        )
        self.size = 0
        self.events: list[EventRecord] = []
        self.transcript: list[TranscriptRecord] = []

    def preflight_bytes(self, total_bytes: int, largest_record: int) -> None:
        if largest_record > self.limits.max_record_bytes:
            raise SessionExportTooLarge("Session export source exceeds max_record_bytes.")
        if total_bytes > self.limits.max_bytes:
            raise SessionExportTooLarge("Session export source exceeds max_bytes.")

    def charge(self, value: Any) -> None:
        encoded = canonical_durable_json_bytes(value, "session export component")
        size = len(encoded)
        if size > self.limits.max_record_bytes:
            raise SessionExportTooLarge("Session export exceeds max_record_bytes.")
        self.size += size
        if self.size > self.limits.max_bytes:
            raise SessionExportTooLarge("Session export exceeds max_bytes.")

    def event(self, record: EventRecord) -> None:
        self.charge(record.model_dump(mode="json"))
        self.events.append(record.model_copy(deep=True))

    def message(self, record: TranscriptRecord) -> None:
        self.charge(record.model_dump(mode="json"))
        self.transcript.append(record.model_copy(deep=True))

    def finish(
        self,
        *,
        session: Session,
        transcript_cursor: int,
        checkpoint: dict[str, Any] | None,
        deferred: DeferredInteractionInput | None,
        grants: TargetedToolGrantStateSnapshot,
        grants_charged: bool = False,
    ) -> SessionExportSnapshot:
        checkpoint = decode_runtime_checkpoint(checkpoint, session_id=session.id)
        self.charge(session.model_dump(mode="json"))
        self.charge(checkpoint)
        self.charge(None if deferred is None else deferred.model_dump(mode="json"))
        if not grants_charged:
            self.charge(grants.model_dump(mode="json"))
        snapshot = SessionExportSnapshot(
            session=session.model_copy(deep=True),
            events=tuple(self.events),
            transcript_records=tuple(self.transcript),
            checkpoint=checkpoint,
            deferred_interaction_input=deferred,
            targeted_tool_grant_state=grants,
            boundary=SessionExportBoundary(
                event_sequences=tuple(record.sequence for record in self.events),
                through_sequence=self.events[-1].sequence if self.events else 0,
                transcript_cursor=transcript_cursor,
            ),
        )
        if (
            len(
                json.dumps(snapshot.document(), ensure_ascii=False, allow_nan=False).encode("utf-8")
            )
            + 1
            > self.limits.max_bytes
        ):
            raise SessionExportTooLarge("Session export exceeds max_bytes.")
        return snapshot
