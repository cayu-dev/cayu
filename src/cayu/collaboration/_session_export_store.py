"""Exact private session-export publication authority and retained records.

Scopes enclose only runtime-owned store calls, never application callbacks. Their
immutable bytes survive off-thread dispatch without becoming caller capabilities.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, StrictBool, StrictInt, model_validator

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._contracts import ContractValue, InitiatorBinding
from cayu.collaboration._permits import PermitCommand
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration._session_export_bounds import initiator_bytes
from cayu.collaboration.exports import (
    ExportDigest,
    SessionExportAuthorization,
    SessionExportConflict,
    SessionExportNamespace,
    SessionExportReceipt,
    SessionExportRequest,
    SessionExportSettlementReceipt,
)
from cayu.vaults.redaction import SecretRedactor

if TYPE_CHECKING:
    from cayu.events import Event
    from cayu.sessions.base import Session, TranscriptRecord

ROOT_KEY = "session_exports"
OPERATION_PREFIX = "session-export:"
NAMESPACE_KEY = OPERATION_PREFIX + "namespace"


def encoded(value: object) -> bytes:
    return canonical_bounded_durable_json_bytes(
        value, "session export", max_bytes=64 * 1024, max_nodes=8192, max_nesting=64
    )


def digest(value: object) -> str:
    return sha256(encoded(value)).hexdigest()


def source_digest(records: Sequence[TranscriptRecord]) -> str:
    return digest([record.model_dump(mode="json", warnings=False) for record in records])


def operation_key(operation: ContractValue) -> str:
    return (
        OPERATION_PREFIX + sha256(contract_bytes(operation, redactor=SecretRedactor())).hexdigest()
    )


class ExportRoot(ContractValue):
    namespace: SessionExportNamespace
    export_count: StrictInt = Field(ge=0, le=1024)
    pending_count: StrictInt = Field(ge=0, le=64)
    retained_bytes: StrictInt = Field(ge=0, le=64 * 1024 * 1024)
    admission_count: StrictInt = Field(default=0, ge=0, le=64)


class ExportAdmission(ContractValue):
    request: SessionExportRequest
    authorization: SessionExportAuthorization
    source_commitment: ExportDigest
    # Source-owner-only CAS evidence; never part of the public export intent.
    validation_source_commitment: ExportDigest | None = None
    permit: PermitCommand
    settled: StrictBool = False

    @model_validator(mode="after")
    def exact_permit(self) -> ExportAdmission:
        if (self.request.source_selection == "assistant_visible_text_v1") != (
            self.validation_source_commitment is not None
        ):
            raise ValueError("Export admission requires exact source validation evidence.")
        mandate = self.authorization.mandate
        permit = self.permit.intent.request
        if (
            mandate is None
            or mandate.chain.entries[-1].participant != permit.participant
            or permit.source_operation != self.request.ref.operation
            or permit.target.kind != "session_export"
            or permit.target.object_id != operation_key(self.request.ref.operation)
            or permit.target.incarnation != self.request.ref.session_instance_id
            or permit.target_state != "future"
            or permit.effect_scope != "source_export"
            or permit.required_settlement != "exclusion"
        ):
            raise ValueError("Participant export admission authority conflicts.")
        return self


class ExportPreparation(ContractValue):
    admission: ExportAdmission
    state: Literal["prepared", "excluded"] = "prepared"
    reserved_bytes: StrictInt = Field(ge=1, le=64 * 1024 * 1024)
    excluded_by: InitiatorBinding | None = None
    exclusion_mandate_commitment: ExportDigest | None = None

    @model_validator(mode="after")
    def exact_exclusion(self) -> ExportPreparation:
        if (self.state == "excluded") != (self.excluded_by is not None):
            raise ValueError("Export preparation exclusion authority conflicts.")
        if self.state == "prepared" and self.admission.settled:
            raise ValueError("Prepared export cannot discharge its participant admission.")
        if self.excluded_by is not None:
            initiator_bytes(self.excluded_by)
        if (self.excluded_by is not None and self.excluded_by.mandate is not None) != (
            self.exclusion_mandate_commitment is not None
        ):
            raise ValueError("Exclusion must bind its initiating mandate.")
        return self


class ExportRecord(ContractValue):
    receipt: SessionExportReceipt
    payload_json: str
    state: Literal["pending", "released", "retired"] = "pending"
    settlement: SessionExportSettlementReceipt | None = None
    reserved_bytes: StrictInt = Field(ge=1, le=64 * 1024 * 1024)
    admission: ExportAdmission | None = None

    @model_validator(mode="after")
    def consistent_evidence(self) -> ExportRecord:
        expected = self.receipt.expected
        initiator_bytes(expected.initiator)
        authorization = expected.intent.authorization
        payload = json.loads(self.payload_json)
        if (
            type(payload) is not dict
            or encoded(payload).decode() != self.payload_json
            or len(encoded(payload)) > 8192
            or digest(payload) != expected.intent.output_commitment
            or expected.initiator != authorization.initiating_identity()
        ):
            raise ValueError("Conflicting export evidence.")
        if (expected.initiator.participant is not None) != (self.admission is not None):
            raise ValueError("Participant export requires exact retained admission.")
        if self.admission is not None and (
            self.admission.request != expected.intent.request
            or self.admission.authorization != authorization
            or self.admission.source_commitment != expected.intent.source_commitment
            or self.admission.permit.intent.request.target.owner != expected.source
        ):
            raise ValueError("Export receipt conflicts with participant admission.")
        release = expected.intent.release_receipt
        if release is not None and (
            release.expected.source_owner != expected.source
            or set(payload) != {"text"}
            or type(payload["text"]) is not str
            or sha256(payload["text"].encode("utf-8")).hexdigest()
            != release.expected.request.text_commitment
        ):
            raise ValueError("Payload conflicts with exact reviewed content.")
        if self.state == "pending":
            if self.settlement is not None:
                raise ValueError("Pending export carries terminal evidence.")
        elif (
            self.settlement is None
            or self.settlement.request.request != expected.intent.request
            or self.settlement.request.mode
            != {"released": "release", "retired": "retire"}[self.state]
            or (
                self.settlement.acceptance is not None
                and self.settlement.acceptance.export_receipt != self.receipt
            )
        ):
            raise ValueError("Conflicting settlement evidence.")
        return self


class SettlementRecord(ContractValue):
    settlement: SessionExportSettlementReceipt


@dataclass(frozen=True)
class ExportMutation:
    session_id: str
    session_instance_id: str
    expected_root: bytes | None
    desired_root: bytes
    records: tuple[tuple[str, bytes], ...]
    source_indices: tuple[int, ...] = ()
    source_commitment: str | None = None
    events: tuple[bytes, ...] = ()


_MUTATION: ContextVar[ExportMutation | None] = ContextVar("session_export_mutation", default=None)
_READ: ContextVar[str | None] = ContextVar("session_export_read", default=None)


@contextmanager
def mutation_scope(mutation: ExportMutation) -> Iterator[None]:
    token = _MUTATION.set(mutation)
    try:
        yield
    finally:
        _MUTATION.reset(token)


@contextmanager
def read_scope(session_id: str) -> Iterator[None]:
    token = _READ.set(session_id)
    try:
        yield
    finally:
        _READ.reset(token)


def require_operation_key_access(key: str, *, read: bool) -> None:
    if not key.startswith(OPERATION_PREFIX):
        return
    mutation = _MUTATION.get()
    if read and _READ.get() is not None:
        return
    if mutation is None or key not in dict(mutation.records):
        raise SessionExportConflict()


def require_operation_record_owner(key: str, record: object) -> None:
    if not key.startswith(OPERATION_PREFIX):
        return
    require_operation_key_access(key, read=False)
    mutation = _MUTATION.get()
    assert mutation is not None
    if dict(mutation.records)[key] != encoded(record):
        raise SessionExportConflict()


def checkpoint_visible(*, session_id: str) -> bool:
    mutation = _MUTATION.get()
    return _READ.get() == session_id or (mutation is not None and mutation.session_id == session_id)


def require_event_publication(event: Event) -> None:
    from cayu.events import EventType, validate_session_export_event

    if event.type not in {
        EventType.SESSION_EXPORT_PUBLISHED,
        EventType.SESSION_EXPORT_RELEASED,
        EventType.SESSION_EXPORT_RETIRED,
    }:
        return
    validate_session_export_event(event)
    if event._trusted_export_history == encoded(event.model_dump(mode="json", warnings=False)):
        return
    mutation = _MUTATION.get()
    if (
        mutation is None
        or event.session_id != mutation.session_id
        or encoded(event.model_dump(mode="json")) not in mutation.events
    ):
        raise SessionExportConflict()


def restore_export_history_event(event: Event) -> Event:
    """Qualify exact history only at the trusted JSONL import boundary."""
    from cayu.events import EventType, copy_event, validate_session_export_event

    if event.type not in {
        EventType.SESSION_EXPORT_PUBLISHED,
        EventType.SESSION_EXPORT_RELEASED,
        EventType.SESSION_EXPORT_RETIRED,
    }:
        return event
    validate_session_export_event(event)
    copied = copy_event(event)
    copied._trusted_export_history = encoded(copied.model_dump(mode="json", warnings=False))
    return copied


def import_history_checkpoint(
    checkpoint: dict[str, Any] | None, *, session: Session
) -> dict[str, Any] | None:
    """History restore cannot transfer a source-owned export namespace.

    JSONL does not carry its private operation records. Refuse unresolved
    obligations rather than copying an unusable root or silently releasing them.
    Settled history keeps its events, but acquires no payload/replay authority.
    """
    if checkpoint is None or ROOT_KEY not in checkpoint:
        return checkpoint
    root = ExportRoot.model_validate(checkpoint[ROOT_KEY])
    if (
        root.namespace.session_id != session.id
        or root.namespace.session_instance_id != session.instance_id
        or root.pending_count != 0
        or root.admission_count != 0
    ):
        raise SessionExportConflict()
    return {key: value for key, value in checkpoint.items() if key != ROOT_KEY}


def project_checkpoint_root(
    current: Mapping[str, Any] | None,
    replacement: Mapping[str, Any],
    *,
    session_id: str,
) -> dict[str, Any] | None:
    before = None if current is None else current.get(ROOT_KEY)
    after = replacement.get(ROOT_KEY)
    mutation = _MUTATION.get()
    if mutation is not None and mutation.session_id == session_id:
        if (None if before is None else encoded(before)) != mutation.expected_root:
            raise SessionExportConflict()
        if after is None or encoded(after) != mutation.desired_root:
            raise SessionExportConflict()
        return json.loads(mutation.desired_root)
    if after is not None and (before is None or encoded(after) != encoded(before)):
        raise SessionExportConflict()
    if before is None:
        return None
    root = ExportRoot.model_validate(before)
    if root.namespace.session_id != session_id:
        raise SessionExportConflict()
    return root.model_dump(mode="json")


def selected_transcript_indices(*, session_id: str) -> tuple[int, ...]:
    mutation = _MUTATION.get()
    return () if mutation is None or mutation.session_id != session_id else mutation.source_indices


def validate_publication(
    *,
    session: Session,
    current_checkpoint: Mapping[str, Any] | None,
    proposed_checkpoint: Mapping[str, Any],
    operation_records: Mapping[str, Any],
    selected_transcript_rows: tuple[TranscriptRecord, ...],
    events: Sequence[Event],
) -> None:
    mutation = _MUTATION.get()
    if mutation is None:
        for key in operation_records:
            require_operation_record_owner(key, operation_records[key])
        return
    if session.id != mutation.session_id or session.instance_id != mutation.session_instance_id:
        raise SessionExportConflict()
    project_checkpoint_root(current_checkpoint, proposed_checkpoint, session_id=session.id)
    if {key: encoded(value) for key, value in operation_records.items()} != dict(mutation.records):
        raise SessionExportConflict()
    if tuple(encoded(event.model_dump(mode="json")) for event in events) != mutation.events:
        raise SessionExportConflict()
    if mutation.source_commitment is not None and (
        tuple(record.index for record in selected_transcript_rows) != mutation.source_indices
        or source_digest(selected_transcript_rows) != mutation.source_commitment
    ):
        raise SessionExportConflict()


def require_erasure_quiescence(
    *,
    session: Session,
    checkpoint: Mapping[str, Any] | None,
    export_records: Mapping[str, dict[str, Any]],
) -> None:
    raw_root = None if checkpoint is None else checkpoint.get(ROOT_KEY)
    owned = {
        key: value for key, value in export_records.items() if key.startswith(OPERATION_PREFIX)
    }
    if raw_root is None:
        if owned:
            raise SessionExportConflict()
        return
    root = ExportRoot.model_validate(raw_root)
    if (
        root.namespace.session_id != session.id
        or root.namespace.session_instance_id != session.instance_id
        or owned.get(NAMESPACE_KEY) != root.namespace.model_dump(mode="json")
    ):
        raise SessionExportConflict()
    records = [
        ExportRecord.model_validate(raw)
        for key, raw in owned.items()
        if key != NAMESPACE_KEY and "receipt" in raw
    ]
    settlements = {
        key: SettlementRecord.model_validate(raw).settlement
        for key, raw in owned.items()
        if key != NAMESPACE_KEY and "settlement" in raw and "receipt" not in raw
    }
    preparations = [
        ExportPreparation.model_validate(raw)
        for key, raw in owned.items()
        if key != NAMESPACE_KEY and "admission" in raw and "receipt" not in raw
    ]
    if len(records) + len(settlements) + len(preparations) + 1 != len(owned):
        raise SessionExportConflict()
    if (
        len(records) + len(preparations) != root.export_count
        or sum(record.state == "pending" for record in records)
        + sum(record.state == "prepared" for record in preparations)
        != root.pending_count
        or sum(record.reserved_bytes for record in (*records, *preparations)) != root.retained_bytes
        or sum(
            record.admission is not None and not record.admission.settled
            for record in (*records, *preparations)
        )
        != root.admission_count
        or any((record.state == "pending") != (record.settlement is None) for record in records)
        or len(settlements) != sum(record.settlement is not None for record in records)
        or any(
            record.settlement is not None
            and settlements.get(operation_key(record.settlement.request.operation))
            != record.settlement
            for record in records
        )
    ):
        raise SessionExportConflict()
    if root.pending_count or root.admission_count:
        raise SessionExportConflict()
