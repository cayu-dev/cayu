"""Retention attestations that keep compacted runtime-publication receipts verifiable.

A runtime-publication receipt binds the exact transcript segment and events it
published by SHA-256 digest, and every receipt load re-checks that binding.
Storage-retention compaction replaces large tool-output bodies in those events
and transcript messages, so the original digests no longer match. Before it
rewrites anything, compaction verifies the receipt's material against the
receipt, and it then records one attestation per affected receipt: the
receipt's original digests and the digests of the compacted material. A
receipt load accepts compacted material only when an attestation bound to that
exact receipt names its digests; any other change still fails closed.

Attestations are runtime-owned session-operation records under a reserved key
namespace, so they share the session's lifetime and are deleted with it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import MAX_DURABLE_JSON_ENVELOPE_NESTING
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, copy_event
from cayu.messages import Message, detach_message
from cayu.sessions import authority as session_authority_rules
from cayu.sessions._operation_records import (
    RETENTION_COMPACTION_ATTESTATION_OPERATION_KEY_PREFIX,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationReceipt,
    _runtime_publication_event_digest,
)

RETENTION_COMPACTION_ATTESTATION_RECORD_TYPE = "cayu.retention-compaction-attestation"
RETENTION_COMPACTION_ATTESTATION_SCHEMA_VERSION = 1


def _require_digest(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


@dataclass(frozen=True)
class ReceiptMaterial:
    """The durable transcript segment and events one receipt binds."""

    transcript: list[Message]
    transcript_interaction_ids: list[str | None]
    appended_events: list[Event]
    referenced_events: list[Event]


class RuntimePublicationMaterialDigests(BaseModel):
    """The digests a runtime-publication receipt binds for its material."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transcript_digest: str
    events_digest: str
    #: One digest per referenced event, in the receipt's reference order.
    referenced_event_digests: tuple[str, ...] = ()

    @field_validator("transcript_digest", "events_digest")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _require_digest(value, info.field_name)

    @field_validator("referenced_event_digests", mode="before")
    @classmethod
    def validate_referenced(cls, value: Any) -> tuple[str, ...]:
        if type(value) not in (list, tuple):
            raise TypeError("referenced_event_digests must be a list of digests.")
        return tuple(_require_digest(item, "referenced_event_digests item") for item in value)


class RetentionCompactionAttestation(BaseModel):
    """One receipt's original and compacted material digests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.retention-compaction-attestation"] = (
        RETENTION_COMPACTION_ATTESTATION_RECORD_TYPE
    )
    schema_version: Literal[1] = RETENTION_COMPACTION_ATTESTATION_SCHEMA_VERSION
    session_id: str
    publication_id: str = Field(max_length=256)
    receipt: RuntimePublicationMaterialDigests
    compacted: RuntimePublicationMaterialDigests
    audit_id: str = Field(max_length=256)
    recorded_at: datetime

    @field_validator("session_id", "publication_id", "audit_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("recorded_at")
    @classmethod
    def normalize_recorded_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded_at must include a timezone.")
        return value.astimezone(UTC)

    def binds(self, receipt: RuntimePublicationReceipt) -> bool:
        """Whether this attestation was written for exactly ``receipt``."""

        return (
            self.session_id == receipt.session_id
            and self.publication_id == receipt.publication_id
            and self.receipt == receipt_material_digests(receipt)
        )


def retention_attestation_storage_key(publication_id: str) -> str:
    """The session-operation key of a receipt's retention attestation."""

    publication_id = require_clean_nonblank(publication_id, "publication_id")
    return (
        RETENTION_COMPACTION_ATTESTATION_OPERATION_KEY_PREFIX
        + sha256(publication_id.encode("utf-8")).hexdigest()
    )


def receipt_material_digests(
    receipt: RuntimePublicationReceipt,
) -> RuntimePublicationMaterialDigests:
    return RuntimePublicationMaterialDigests(
        transcript_digest=receipt.transcript_digest,
        events_digest=receipt.events_digest,
        referenced_event_digests=tuple(
            reference.event_digest for reference in receipt.referenced_events
        ),
    )


def runtime_publication_material_digests(
    *,
    transcript_messages: Iterable[Message],
    appended_events: Iterable[Event],
    durable_referenced_events: Iterable[Event],
) -> RuntimePublicationMaterialDigests:
    """Digest durable material exactly as receipt validation does."""

    transcript = tuple(detach_message(message) for message in transcript_messages)
    events = tuple(copy_event(event) for event in appended_events)
    return RuntimePublicationMaterialDigests(
        transcript_digest=session_authority_rules._canonical_runtime_publication_digest(
            [message.model_dump(mode="json") for message in transcript],
            max_nesting=MAX_DURABLE_JSON_ENVELOPE_NESTING,
        ),
        events_digest=session_authority_rules._canonical_runtime_publication_digest(
            [event.model_dump(mode="json") for event in events]
        ),
        referenced_event_digests=tuple(
            _runtime_publication_event_digest(event) for event in durable_referenced_events
        ),
    )


def parse_retention_attestation(
    record: Any, receipt: RuntimePublicationReceipt
) -> RetentionCompactionAttestation | None:
    """Return the attestation when ``record`` is well formed and bound to ``receipt``."""

    try:
        attestation = RetentionCompactionAttestation.model_validate(record)
    except (TypeError, ValueError):
        return None
    return attestation if attestation.binds(receipt) else None


__all__ = [
    "RETENTION_COMPACTION_ATTESTATION_RECORD_TYPE",
    "ReceiptMaterial",
    "RetentionCompactionAttestation",
    "RuntimePublicationMaterialDigests",
    "parse_retention_attestation",
    "receipt_material_digests",
    "retention_attestation_storage_key",
    "runtime_publication_material_digests",
]
