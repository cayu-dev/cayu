"""Immutable, historical-only context-view contracts.

These values deliberately use canonical JSON text for retained projections.  The
text is detached and bounded before it crosses the SessionStore boundary; it is
data, never an executable session or an authority token.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_bounded_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.artifacts.attachments import (
    FileAttachment,
    file_attachment_from_payload,
    same_file_attachment_reference,
)
from cayu.artifacts.resources import (
    ResourceAcquisitionReceipt,
    ResourcePreparationReceipt,
    ResourceTransferReceipt,
    resource_operation_digest,
)
from cayu.collaboration._contracts import ContractValue, Generation, Identifier, OwnerRef
from cayu.collaboration.participants import ParticipantRef
from cayu.messages import FilePart, Message, ToolResultPart
from cayu.sessions.base import RunRequest, copy_run_request

CONTEXT_VIEW_CONTRACT_VERSION = 1
CONTEXT_VIEW_STORE_VERSION = 1
CONTEXT_VIEW_MAX_JSON_BYTES = 8 * 1024 * 1024
CONTEXT_VIEW_MAX_MESSAGES = 512
# Store-enforced ceilings. Request limits may be narrower, but no caller can
# grow one owner's durable publication/retention namespace without bound.
CONTEXT_VIEW_MAX_PUBLICATIONS_PER_OWNER = 4096
CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER = 4096
CONTEXT_VIEW_MAX_LIFECYCLE_EVENTS_PER_VIEW = 4096
CONTEXT_VIEW_EXPIRY_BATCH_SIZE = 256


class ContextViewProjectionSource(ContractValue):
    """Detached whole-turn input; JSON strings cannot carry mutable live state."""

    source_session_id: Identifier
    source_session_instance_id: Identifier
    participant_json: StrictStr
    interaction_id: Identifier
    boundary_id: Identifier
    completion_event_id: Identifier
    source_transcript_cursor: StrictInt = Field(ge=0)
    transcript_cursor: StrictInt = Field(ge=0)
    messages_json: StrictStr
    compaction_json: StrictStr
    historical_ancestry_json: StrictStr

    @field_validator(
        "participant_json", "messages_json", "compaction_json", "historical_ancestry_json"
    )
    @classmethod
    def validate_source_json(cls, value: str, info) -> str:
        return _canonical_json_text(value, info.field_name, max_bytes=CONTEXT_VIEW_MAX_JSON_BYTES)

    @property
    def commitment(self) -> str:
        return (
            "sha256:"
            + sha256(
                canonical_bounded_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "context projection source",
                    max_bytes=CONTEXT_VIEW_MAX_JSON_BYTES,
                    max_nodes=8192,
                    max_nesting=64,
                )
            ).hexdigest()
        )


class ContextViewExtensionProjection(ContractValue):
    """Producer attestation that a projection belongs to the supplied snapshot."""

    source_commitment: StrictStr
    projection_json: StrictStr

    @field_validator("projection_json")
    @classmethod
    def validate_history(cls, value: str) -> str:
        return _historical_extension_json(value)


@dataclass(frozen=True, slots=True)
class ContextViewExtensionRegistration:
    """Application-owned producer for one explicitly registered projection.

    The callable runs before the store transaction and must return a boundary-
    qualified projection (or ``None`` for the explicit-absence state). The store only
    receives the resulting typed record; it never executes an extension.
    """

    extension: str
    schema_version: int
    project: Callable[[ContextViewProjectionSource], ContextViewExtensionProjection | None]

    def __post_init__(self) -> None:
        if type(self.extension) is not str or not self.extension.strip():
            raise ValueError("Context-view extension names must be non-empty.")
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("Context-view extension schema versions must be positive integers.")
        if not callable(self.project):
            raise TypeError("Context-view extension project must be callable.")


@dataclass(frozen=True, slots=True)
class ParticipantSessionCreationRequest:
    """Exact creation intent carried across retries and lost acknowledgements."""

    request: RunRequest
    creation_key: str
    metadata_json: str | None = None

    def __post_init__(self) -> None:
        if type(self.request) is not RunRequest:
            raise TypeError("Participant session creation requires a RunRequest.")
        copied = copy_run_request(self.request)
        key = require_durable_clean_nonblank(self.creation_key, "creation_key")
        if len(key.encode("utf-8")) > 256:
            raise ValueError("creation_key must be at most 256 UTF-8 bytes.")
        object.__setattr__(self, "request", copied)
        object.__setattr__(self, "creation_key", key)
        if self.metadata_json is not None:
            object.__setattr__(
                self,
                "metadata_json",
                _canonical_json_text(
                    self.metadata_json, "participant creation metadata", max_bytes=256 * 1024
                ),
            )

    @property
    def request_commitment(self) -> str:
        material = self.request.model_dump(mode="json")
        if self.metadata_json is not None:
            material = {"request": material, "metadata": json.loads(self.metadata_json)}
        return (
            "sha256:"
            + sha256(
                canonical_bounded_durable_json_bytes(
                    material,
                    "participant_session_request",
                    max_bytes=8 * 1024 * 1024,
                    max_nodes=8192,
                    max_nesting=64,
                )
            ).hexdigest()
        )


@dataclass(frozen=True, slots=True)
class RecipientSessionCreationRequest:
    """Authenticated, inert recipient creation intent.

    This is deliberately distinct from root participant-session creation.  A
    FORK carries one exact, already selected historical view; it never carries
    a live session or a fallback request.
    """

    request: RunRequest
    creation_key: str
    recipient: ParticipantRef
    mode: Literal["fresh", "fork"] = "fresh"
    selected_view: ContextViewSelectionReceipt | None = None
    resource_transfers: tuple[ResourceTransferReceipt, ...] = ()
    preparation_receipts: tuple[ResourcePreparationReceipt, ...] = ()

    def __post_init__(self) -> None:
        if type(self.request) is not RunRequest:
            raise TypeError("Recipient creation requires a RunRequest.")
        if type(self.recipient) is not ParticipantRef:
            raise TypeError("Recipient creation requires a ParticipantRef.")
        object.__setattr__(self, "recipient", ParticipantRef.model_validate(self.recipient))
        key = require_durable_clean_nonblank(self.creation_key, "creation_key")
        # Reserve the ten bytes used by participant_request's "recipient:" namespace.
        if len(key.encode("utf-8")) > 246:
            raise ValueError("creation_key must be at most 246 UTF-8 bytes.")
        if self.mode not in {"fresh", "fork"}:
            raise ValueError("Recipient creation mode must be fresh or fork.")
        if self.mode == "fresh" and self.selected_view is not None:
            raise ValueError("Fresh creation cannot include a selected view.")
        if self.mode == "fork":
            if type(self.selected_view) is not ContextViewSelectionReceipt:
                raise ValueError("Fork creation requires an exact selected view receipt.")
            try:
                # model_copy(update=...) can bypass nested model validation.
                # ContractValue snapshots known fields without invoking any
                # serializer and revalidates even already-typed instances.
                selected_view = ContextViewSelectionReceipt.model_validate(self.selected_view)
            except (TypeError, ValueError) as exc:
                raise ValueError("Fork creation requires a valid selected view receipt.") from exc
            object.__setattr__(self, "selected_view", selected_view)
            if self.selected_view.state not in {"adopted", "transferred"}:
                raise ValueError("Fork creation requires an adopted or transferred view.")
        transfers = tuple(self.resource_transfers)
        if any(type(item) is not ResourceTransferReceipt for item in transfers):
            raise TypeError("Recipient resource transfers require typed receipts.")
        transfers = tuple(ResourceTransferReceipt.model_validate(item) for item in transfers)
        transfer_ids = tuple(item.operation_digest for item in transfers)
        if len(set(transfer_ids)) != len(transfer_ids):
            raise ValueError("Resource transfer receipts must be unique.")
        if any(item.stage != "accepted" for item in transfers):
            raise ValueError("Only accepted resource transfers can be retained by a recipient.")
        preparations = tuple(self.preparation_receipts)
        if any(type(item) is not ResourcePreparationReceipt for item in preparations):
            raise TypeError("Recipient preparation evidence requires typed receipts.")
        preparations = tuple(
            ResourcePreparationReceipt.model_validate(item) for item in preparations
        )
        expected_preparations = tuple(resource_operation_digest(item.command) for item in transfers)
        if tuple(item.operation_digest for item in preparations) != expected_preparations:
            raise ValueError("Recipient preparation evidence does not match transfers.")
        object.__setattr__(self, "request", copy_run_request(self.request))
        object.__setattr__(self, "creation_key", key)
        object.__setattr__(self, "resource_transfers", transfers)
        object.__setattr__(self, "preparation_receipts", preparations)

    @property
    def metadata_json(self) -> str:
        material = {
            "mode": self.mode,
            "original_request_commitment": ParticipantSessionCreationRequest(
                request=self.request, creation_key=self.creation_key
            ).request_commitment,
            "recipient": self.recipient.model_dump(mode="json"),
            "selected_view": None
            if self.selected_view is None
            else self.selected_view.model_dump(mode="json"),
            "resource_transfers": [
                item.model_dump(mode="json") for item in self.resource_transfers
            ],
            "preparation_receipts": [
                item.model_dump(mode="json") for item in self.preparation_receipts
            ],
        }
        return canonical_bounded_durable_json_bytes(
            material,
            "recipient creation metadata",
            max_bytes=512 * 1024,
            max_nodes=8192,
            max_nesting=64,
        ).decode()

    @property
    def participant_request(self) -> ParticipantSessionCreationRequest:
        request = self.request
        if self.mode == "fork":
            assert self.selected_view is not None
            try:
                messages = [
                    Message.model_validate(item)
                    for item in json.loads(self.selected_view.view.messages_json)
                ]
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("Selected context-view messages are unavailable.") from exc
            # Retained history precedes the new first input. Neither may erase
            # the other, including in exact readback and lost-ACK replay.
            request = request.model_copy(
                update={"messages": [*messages, *request.messages]}, deep=True
            )
        return ParticipantSessionCreationRequest(
            request=request,
            creation_key="recipient:" + self.creation_key,
            metadata_json=self.metadata_json,
        )

    @property
    def input_artifact_ids(self) -> tuple[str, ...]:
        """Return every artifact referenced by the complete child input."""
        return tuple(sorted({item.artifact_id for item in self.input_attachments}))

    @property
    def input_attachments(self) -> tuple[FileAttachment, ...]:
        """Keep full resolution identity until retained-material validation."""
        messages = self.participant_request.request.messages
        references: dict[str, FileAttachment] = {}
        for message in messages:
            for part in message.content:
                if type(part) is FilePart:
                    payloads = (part.attachment,)
                elif type(part) is ToolResultPart:
                    payloads = tuple(part.artifacts)
                else:
                    continue
                for payload in payloads:
                    attachment = file_attachment_from_payload(payload)
                    if attachment is not None:
                        previous = references.get(attachment.artifact_id)
                        if previous is not None and not same_file_attachment_reference(
                            previous, attachment
                        ):
                            raise ValueError("Recipient attachment references conflict.")
                        references[attachment.artifact_id] = attachment
                    elif type(part) is FilePart:
                        raise ValueError("Recipient file attachment is unsupported.")
        return tuple(references.values())


class RecipientSessionCreationReceipt(ContractValue):
    """Immutable receipt for one inert recipient-owned child."""

    session_id: Identifier
    session_instance_id: Identifier
    recipient: ParticipantRef
    mode: Literal["fresh", "fork"]
    creation_key: StrictStr
    request_commitment: StrictStr
    source_view_commitment: StrictStr | None = None
    resource_transfer_commitments: tuple[StrictStr, ...] = ()
    resource_preparation_commitments: tuple[StrictStr, ...] = ()
    participant_receipt: ParticipantSessionCreationReceipt
    receipt_commitment: StrictStr

    @model_validator(mode="after")
    def validate_receipt(self) -> RecipientSessionCreationReceipt:
        if self.participant_receipt.binding.participant != self.recipient:
            raise ValueError("Recipient receipt ownership conflicts with its binding.")
        if self.participant_receipt.binding.session_id != self.session_id:
            raise ValueError("Recipient receipt session conflicts with its binding.")
        metadata_text = self.participant_receipt.recipient_metadata_json
        if metadata_text is None:
            raise ValueError("Recipient receipt is missing durable creation metadata.")
        try:
            metadata = json.loads(metadata_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("Recipient receipt metadata is invalid.") from exc
        if (
            metadata.get("mode") != self.mode
            or metadata.get("recipient") != self.recipient.model_dump(mode="json")
            or tuple(
                item.get("operation_digest")
                for item in metadata.get("resource_transfers", ())
                if isinstance(item, dict)
            )
            != tuple(self.resource_transfer_commitments)
            or tuple(
                item.get("operation_digest")
                for item in metadata.get("preparation_receipts", ())
                if isinstance(item, dict)
            )
            != tuple(self.resource_preparation_commitments)
        ):
            raise ValueError("Recipient receipt metadata conflicts with its tuple.")
        material = self.model_dump(mode="json", exclude={"receipt_commitment"})
        expected = (
            "sha256:"
            + sha256(
                canonical_bounded_durable_json_bytes(
                    material,
                    "recipient receipt",
                    max_bytes=512 * 1024,
                    max_nodes=8192,
                    max_nesting=64,
                )
            ).hexdigest()
        )
        if self.receipt_commitment != expected:
            raise ValueError("Recipient receipt commitment does not match its material.")
        return self


def _canonical_json_text(value: object, field_name: str, *, max_bytes: int) -> str:
    if type(value) is str:
        text = value
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must contain canonical JSON.") from exc
    else:
        parsed = copy_durable_json_object(value, field_name)
        text = canonical_bounded_durable_json_bytes(
            parsed,
            field_name,
            max_bytes=max_bytes,
            max_nodes=8192,
            max_nesting=64,
        ).decode("utf-8")
        return text
    canonical = canonical_bounded_durable_json_bytes(
        parsed,
        field_name,
        max_bytes=max_bytes,
        max_nodes=8192,
        max_nesting=64,
    ).decode("utf-8")
    if text != canonical:
        raise ValueError(f"{field_name} must be canonical JSON.")
    return text


def json_commitment(value: str, field_name: str = "context view material") -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be JSON text.")
    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


class ContextViewLimits(ContractValue):
    """Finite hard ceilings for one context-view owner."""

    max_views: StrictInt = Field(default=256, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    max_pins: StrictInt = Field(default=256, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    max_view_bytes: StrictInt = Field(default=8 * 1024 * 1024, ge=1, le=64 * 1024 * 1024)
    max_retained_bytes: StrictInt = Field(default=64 * 1024 * 1024, ge=1, le=256 * 1024 * 1024)
    max_lifetime_seconds: StrictInt = Field(default=7 * 24 * 60 * 60, ge=1, le=365 * 24 * 60 * 60)


class ContextViewExtensionRecord(ContractValue):
    """One registered extension's detached historical projection."""

    extension: Identifier
    schema_version: Generation
    state: Literal["present", "absent"]
    projection_json: str | None = None
    commitment: StrictStr
    source_commitment: StrictStr

    @field_validator("projection_json")
    @classmethod
    def validate_projection(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _historical_extension_json(value)

    @model_validator(mode="after")
    def validate_state(self) -> ContextViewExtensionRecord:
        if (self.state == "present") != (self.projection_json is not None):
            raise ValueError("Extension presence must match its projection.")
        expected = (
            "sha256:" + sha256((self.projection_json or "absent").encode("utf-8")).hexdigest()
        )
        if self.commitment != expected:
            raise ValueError("Extension commitment does not match its projection.")
        if self.projection_json is not None:
            _validate_historical_extension_projection(
                json.loads(self.projection_json), self.extension
            )
        return self


def _extension_schema_commitment(records: tuple[ContextViewExtensionRecord, ...]) -> str:
    """Selection compatibility is independent of projected data and boundary."""
    schemas = sorted((record.extension, record.schema_version) for record in records)
    if len({name for name, _version in schemas}) != len(schemas):
        raise ValueError("Context-view extension names must be unique.")
    material = canonical_bounded_durable_json_bytes(
        [{"extension": name, "schema_version": version} for name, version in schemas],
        "context extension schemas",
        max_bytes=256 * 1024,
        max_nodes=8192,
        max_nesting=64,
    )
    return "sha256:" + sha256(material).hexdigest()


def project_context_view_extensions(
    registrations: tuple[ContextViewExtensionRegistration, ...],
    source: ContextViewProjectionSource,
) -> tuple[tuple[ContextViewExtensionRecord, ...], str]:
    """Project every registered extension before publication.

    The complete set is produced or the operation fails.  Producers receive the
    immutable boundary-qualified source; only detached bounded canonical JSON
    crosses the publication boundary.
    """

    if type(registrations) is not tuple or any(
        type(item) is not ContextViewExtensionRegistration for item in registrations
    ):
        raise TypeError("Context-view extensions must be a typed registration tuple.")
    if type(source) is not ContextViewProjectionSource:
        raise TypeError("Context-view projection requires a boundary-qualified source.")
    source = ContextViewProjectionSource.model_validate(source)
    source_commitment = source.commitment
    records: list[ContextViewExtensionRecord] = []
    for registration in registrations:
        projected = registration.project(source.model_copy(deep=True))
        if projected is None:
            records.append(
                ContextViewExtensionRecord(
                    extension=registration.extension,
                    schema_version=registration.schema_version,
                    state="absent",
                    commitment="sha256:" + sha256(b"absent").hexdigest(),
                    source_commitment=source_commitment,
                )
            )
            continue
        if type(projected) is not ContextViewExtensionProjection:
            raise TypeError("Extensions must return a boundary-qualified historical projection.")
        projected = ContextViewExtensionProjection.model_validate(projected)
        if projected.source_commitment != source_commitment:
            raise ValueError("Context extension projection belongs to another boundary.")
        projection_json = _canonical_json_text(
            projected.projection_json,
            f"context extension {registration.extension}",
            max_bytes=256 * 1024,
        )
        _validate_historical_extension_projection(
            json.loads(projection_json), registration.extension
        )
        records.append(
            ContextViewExtensionRecord(
                extension=registration.extension,
                schema_version=registration.schema_version,
                state="present",
                projection_json=projection_json,
                commitment=json_commitment(projection_json),
                source_commitment=source_commitment,
            )
        )
    canonical_bounded_durable_json_bytes(
        [record.model_dump(mode="json") for record in records],
        "context extension set",
        max_bytes=256 * 1024,
        max_nodes=8192,
        max_nesting=64,
    )
    return tuple(records), _extension_schema_commitment(tuple(records))


def _historical_extension_json(value: str) -> str:
    if len(value.encode("utf-8")) > 256 * 1024:
        raise ValueError("Context extension projection exceeds its byte limit.")
    try:
        parsed = json.loads(value)
    except (ValueError, RecursionError):
        raise ValueError("Context extension requires bounded historical JSON.") from None
    _validate_historical_extension_projection(parsed, "projection")
    return _canonical_json_text(value, "context extension projection", max_bytes=256 * 1024)


_HISTORICAL_EXTENSION_TEXT_FIELDS = frozenset(
    {"label", "text", "note", "summary", "provider_name", "model"}
)


def _validate_historical_extension_projection(
    value: object, extension: str, *, depth: int = 0
) -> None:
    """Accept only historical records, never arbitrary authority-shaped JSON.

    Text is inert even when it describes an operation. Structured records have a
    closed grammar: textual fields, a scalar value, and nested historical items.
    Unknown keys and object-valued scalar fields fail closed on reconstruction.
    """
    if depth > 64 or type(value) is not dict:
        raise ValueError(f"Context extension {extension} requires a historical record.")
    for key, child in value.items():
        if key in _HISTORICAL_EXTENSION_TEXT_FIELDS and type(child) is str:
            continue
        if key == "value" and (child is None or type(child) in {str, bool, int, float}):
            continue
        if key == "items" and type(child) is list:
            for item in child:
                _validate_historical_extension_projection(item, extension, depth=depth + 2)
            continue
        raise ValueError(
            f"Context extension {extension} contains unsupported live-authority "
            "or non-historical material."
        )


def validate_context_view_lifecycle_capacity(
    event_count: int, unsettled_count: int, *, additional_slots: int = 0
) -> None:
    """Every unsettled pin reserves one terminal event in the same quota.

    Selection reserves a slot; adoption/transfer consumes a new slot without
    spending the terminal reservation. Release/expiry exchanges its reservation
    for evidence, so cleanup remains possible when admission reaches the ceiling.
    """
    if (
        event_count + unsettled_count + additional_slots
        > CONTEXT_VIEW_MAX_LIFECYCLE_EVENTS_PER_VIEW
    ):
        raise OverflowError("Context-view lifecycle evidence quota exceeded.")


def validate_context_view_lifecycle_storage(
    event: ContextViewLifecycleEvent, row: Mapping[str, Any]
) -> ContextViewLifecycleEvent:
    """Reject contradictory indexed and reconstructed ownership evidence."""
    expected = {
        "event_id": event.event_id,
        "operation_key": event.operation_key,
        "selection_key": event.selection_key,
        "view_id": event.view_id,
        "state": event.state,
        "owner_scope": event.owner.application_scope,
        "owner_id": event.owner.owner_id,
        "owner_incarnation": event.owner.incarnation,
        "pin_commitment": event.pin_commitment,
        "ownership_revision": event.ownership_revision,
    }
    if any(row[key] != value for key, value in expected.items()):
        raise ValueError("Context-view lifecycle indexes conflict with their event.")
    return event


def validate_context_view_manifest_storage(
    manifest: ContextViewManifest,
    *,
    view_id: str,
    owner_scope: str,
    owner_id: str,
    owner_incarnation: str,
    source_session_id: str,
    source_session_instance_id: str,
    transcript_cursor: int,
    projection_schema: str,
    extension_set_commitment: str,
) -> ContextViewManifest:
    """Cross-check denormalized indexed identity against the signed manifest."""

    expected = (
        manifest.view_id == view_id
        and manifest.source_owner.application_scope == owner_scope
        and manifest.source_owner.owner_id == owner_id
        and manifest.source_owner.incarnation == owner_incarnation
        and manifest.source_session_id == source_session_id
        and manifest.source_session_instance_id == source_session_instance_id
        and manifest.transcript_cursor == transcript_cursor
        and manifest.projection_schema == projection_schema
        and manifest.extension_set_commitment == extension_set_commitment
    )
    if not expected:
        raise ValueError("Context-view indexed identity conflicts with its manifest.")
    return manifest


def validate_context_view_receipt_storage(
    receipt: ContextViewSelectionReceipt,
    *,
    selection_key: str,
    view_id: str,
    owner_scope: str,
    owner_id: str,
    owner_incarnation: str,
    state: str,
    pin_commitment: str,
    ownership_revision: int,
) -> ContextViewSelectionReceipt:
    """Cross-check denormalized selection identity against its receipt."""

    owner = receipt.owner
    if not (
        receipt.selection_key == selection_key
        and receipt.view.view_id == view_id
        and owner.application_scope == owner_scope
        and owner.owner_id == owner_id
        and owner.incarnation == owner_incarnation
        and receipt.state == state
        and receipt.pin_commitment == pin_commitment
        and receipt.ownership_revision == ownership_revision
    ):
        raise ValueError("Context-view selection indexes conflict with its receipt.")
    return receipt


class ParticipantSessionBinding(ContractValue):
    """Immutable historical participant ownership evidence for one session."""

    application_scope: Identifier
    participant: ParticipantRef
    session_id: Identifier
    session_instance_id: Identifier
    lifecycle_revision: Generation
    configuration_revision: Generation
    admission_generation: Generation
    creator_commitment: StrictStr
    authorization_commitment: StrictStr
    initial_input_commitment: StrictStr
    request_commitment: StrictStr
    execution_profile_commitment: StrictStr
    historical_definition_json: StrictStr
    creation_key: StrictStr
    schema_version: Literal[1] = 1

    @field_validator("historical_definition_json")
    @classmethod
    def validate_definition(cls, value: str) -> str:
        return _canonical_json_text(value, "historical definition", max_bytes=256 * 1024)


class ParticipantSessionCreationReceipt(ContractValue):
    """Replayable receipt for an inert participant-owned session creation."""

    binding: ParticipantSessionBinding
    requested_session_id: Identifier | None
    initial_input_commitment: StrictStr
    execution_profile_json: StrictStr
    receipt_commitment: StrictStr
    recipient_metadata_json: str | None = None
    schema_version: Literal[1] = 1

    @field_validator("execution_profile_json")
    @classmethod
    def validate_profile(cls, value: str) -> str:
        return _canonical_json_text(value, "execution profile", max_bytes=256 * 1024)

    @model_validator(mode="after")
    def validate_receipt(self) -> ParticipantSessionCreationReceipt:
        if self.initial_input_commitment != self.binding.initial_input_commitment:
            raise ValueError("Receipt initial-input commitment conflicts with its binding.")
        material = self.model_dump(mode="json", exclude={"receipt_commitment"})
        if material.get("recipient_metadata_json") is None:
            material.pop("recipient_metadata_json", None)
        expected = (
            "sha256:"
            + sha256(
                canonical_bounded_durable_json_bytes(
                    material,
                    "participant session creation receipt",
                    max_bytes=512 * 1024,
                    max_nodes=8192,
                    max_nesting=64,
                )
            ).hexdigest()
        )
        if self.receipt_commitment != expected:
            raise ValueError("Receipt commitment does not match its material.")
        return self


@dataclass(frozen=True, slots=True)
class ParticipantSessionExecutionRequest:
    """Authenticated, exact activation intent for one inert participant session."""

    request: RunRequest
    session_instance_id: str
    execution_key: str

    def __post_init__(self) -> None:
        if type(self.request) is not RunRequest:
            raise TypeError("Participant session execution requires a RunRequest.")
        copied = copy_run_request(self.request)
        instance_id = require_durable_clean_nonblank(
            self.session_instance_id, "session_instance_id"
        )
        if len(instance_id.encode("utf-8")) > 256:
            raise ValueError("session_instance_id must be at most 256 UTF-8 bytes.")
        key = require_durable_clean_nonblank(self.execution_key, "execution_key")
        if len(key.encode("utf-8")) > 256:
            raise ValueError("execution_key must be at most 256 UTF-8 bytes.")
        object.__setattr__(self, "request", copied)
        object.__setattr__(self, "session_instance_id", instance_id)
        object.__setattr__(self, "execution_key", key)

    @property
    def request_commitment(self) -> str:
        material = self.request.model_dump(mode="json")
        return self._commitment(material)

    @property
    def creation_request_commitment(self) -> str:
        """Commitment used when creation intentionally omitted a session ID."""

        material = self.request.model_copy(update={"session_id": None}).model_dump(mode="json")
        return self._commitment(material)

    @staticmethod
    def _commitment(material: object) -> str:
        return (
            "sha256:"
            + sha256(
                canonical_bounded_durable_json_bytes(
                    material,
                    "participant_session_execution_request",
                    max_bytes=8 * 1024 * 1024,
                    max_nodes=8192,
                    max_nesting=64,
                )
            ).hexdigest()
        )


class ContextViewPublicationRequest(ContractValue):
    """Authenticated request to publish one completed session boundary."""

    source_session_id: Identifier
    source_session_instance_id: Identifier
    view_id: Identifier
    interaction_id: Identifier
    boundary_id: Identifier
    projection_schema: Identifier
    publication_key: StrictStr
    resource_receipts: tuple[ResourceAcquisitionReceipt, ...] = ()

    @field_validator("publication_key")
    @classmethod
    def validate_publication_key(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "publication_key")
        if len(value.encode("utf-8")) > 512:
            raise ValueError("publication_key must be at most 512 UTF-8 bytes.")
        return value


class ContextViewManifest(ContractValue):
    """Positive schema for one completed, historical-only context boundary."""

    schema_version: Literal[1] = 1
    source_owner: OwnerRef
    participant: ParticipantRef
    source_session_id: Identifier
    source_session_instance_id: Identifier
    view_id: Identifier
    interaction_id: Identifier
    boundary_id: Identifier
    completion_event_id: Identifier
    transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    projection_schema: Identifier
    extension_set_commitment: StrictStr
    messages_json: StrictStr
    application_context_json: StrictStr | None = None
    extensions: tuple[ContextViewExtensionRecord, ...] = ()
    historical_ancestry_json: StrictStr
    causal_budget_ancestry_json: StrictStr
    resource_references_json: StrictStr | None = None
    compaction_json: StrictStr | None = None
    messages_commitment: StrictStr
    application_context_commitment: StrictStr
    manifest_commitment: StrictStr

    @field_validator(
        "messages_json",
        "historical_ancestry_json",
        "causal_budget_ancestry_json",
        "resource_references_json",
        "compaction_json",
        "application_context_json",
    )
    @classmethod
    def validate_json_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _canonical_json_text(value, info.field_name, max_bytes=CONTEXT_VIEW_MAX_JSON_BYTES)

    @field_validator("messages_json")
    @classmethod
    def validate_messages(cls, value: str) -> str:
        parsed = json.loads(value)
        if type(parsed) is not list or len(parsed) > CONTEXT_VIEW_MAX_MESSAGES:
            raise ValueError("A context view must contain a bounded message list.")
        return value

    @model_validator(mode="after")
    def validate_commitments(self) -> ContextViewManifest:
        if self.extensions:
            if self.compaction_json is None:
                raise ValueError("Extensions require a retained compaction boundary.")
            compaction = json.loads(self.compaction_json)
            source = ContextViewProjectionSource(
                source_session_id=self.source_session_id,
                source_session_instance_id=self.source_session_instance_id,
                participant_json=canonical_bounded_durable_json_bytes(
                    self.participant.model_dump(mode="json"),
                    "participant",
                    max_bytes=8192,
                    max_nodes=64,
                ).decode("utf-8"),
                interaction_id=self.interaction_id,
                boundary_id=self.boundary_id,
                completion_event_id=self.completion_event_id,
                source_transcript_cursor=compaction["input_frontier"],
                transcript_cursor=self.transcript_cursor,
                messages_json=self.messages_json,
                compaction_json=self.compaction_json,
                historical_ancestry_json=self.historical_ancestry_json,
            )
            if any(
                extension.source_commitment != source.commitment for extension in self.extensions
            ):
                raise ValueError("Context extensions conflict with the manifest boundary.")
        if self.application_context_commitment != json_commitment(
            self.application_context_json or "null", "application context"
        ):
            raise ValueError("Application-context commitment does not match its material.")
        if self.messages_commitment != json_commitment(self.messages_json, "messages"):
            raise ValueError("Message commitment does not match retained material.")
        extension_material = [extension.model_dump(mode="json") for extension in self.extensions]
        canonical_bounded_durable_json_bytes(
            extension_material,
            "context extension set",
            max_bytes=256 * 1024,
            max_nodes=8192,
            max_nesting=64,
        )
        expected_set = _extension_schema_commitment(self.extensions)
        if self.extension_set_commitment != expected_set:
            raise ValueError("Extension-set commitment does not match retained extensions.")
        material = self.model_dump(mode="json", exclude={"manifest_commitment"})
        expected_manifest = (
            "sha256:"
            + sha256(
                canonical_bounded_durable_json_bytes(
                    material,
                    "context view manifest",
                    max_bytes=CONTEXT_VIEW_MAX_JSON_BYTES,
                    max_nodes=8192,
                    max_nesting=64,
                )
            ).hexdigest()
        )
        if self.manifest_commitment != expected_manifest:
            raise ValueError("Context-view manifest commitment does not match its material.")
        return self


def require_independent_context_view_material(
    manifest: ContextViewManifest,
    *,
    qualified_resources: tuple[ResourceAcquisitionReceipt, ...] | None = None,
) -> None:
    """Prove this retained value needs no mutable source material for readback.

    Called under the source mutation owner, not merely by public preflight.
    Unsupported transforms and resource references remain fenced.
    """

    manifest = ContextViewManifest.model_validate(manifest.model_dump(mode="json", warnings=False))
    compaction = json.loads(manifest.compaction_json or "null")
    messages = json.loads(manifest.messages_json)

    references = json.loads(manifest.resource_references_json or "[]")
    if type(references) is not list:
        raise ValueError("Pinned context-view resource references are invalid.")
    try:
        receipts = tuple(ResourceAcquisitionReceipt.model_validate(item) for item in references)
        required = context_view_artifact_ids([Message.model_validate(item) for item in messages])
    except (TypeError, ValueError):
        raise ValueError("Pinned context-view material is invalid.") from None
    if not set(required) <= {item for receipt in receipts for item in receipt.material_ids}:
        raise ValueError("Pinned context-view resources are not independently retained.")
    if qualified_resources is not None and any(
        receipt not in qualified_resources for receipt in receipts
    ):
        raise ValueError("Pinned context-view material resource references are not qualified.")
    if (
        manifest.projection_schema != "whole-turn.v1"
        or type(compaction) is not dict
        or set(compaction)
        != {"state", "input_frontier", "retained_output_frontier", "retained_suffix_frontier"}
        or compaction["state"] != "uncompacted"
        or any(
            type(compaction[name]) is not int
            for name in ("input_frontier", "retained_output_frontier", "retained_suffix_frontier")
        )
        or not 0 <= compaction["input_frontier"] <= manifest.transcript_cursor
        or compaction["retained_output_frontier"] != manifest.transcript_cursor
        or compaction["retained_suffix_frontier"] != manifest.transcript_cursor
        or len(messages) != manifest.transcript_cursor - compaction["input_frontier"]
    ):
        raise ValueError("Pinned context-view material is unavailable for compaction.")


def context_view_artifact_ids(messages: list[Message]) -> tuple[str, ...]:
    return tuple(sorted({item.artifact_id for item in context_view_attachments(messages)}))


def context_view_attachments(messages: list[Message]) -> tuple[FileAttachment, ...]:
    """Only typed local file attachments may cross the historical boundary.

    Other resource-shaped payloads remain unsupported, including resource
    markers embedded in arbitrary tool data. References carry data, not grants.
    """
    references: dict[str, FileAttachment] = {}

    def has_resource(value):
        if isinstance(value, dict):
            return any(
                (
                    key
                    in {"attachment", "attachments", "resource_reference", "resource_references"}
                    and child is not None
                )
                or has_resource(child)
                for key, child in value.items()
            )
        return isinstance(value, list) and any(has_resource(child) for child in value)

    for message in messages:
        for part in message.content:
            value = part.model_dump(mode="json", warnings=False)
            if type(part) is FilePart:
                payloads = (value.pop("attachment"),)
            elif type(part) is ToolResultPart:
                payloads = tuple(value.pop("artifacts"))
            else:
                payloads = ()
            if has_resource(value):
                raise ValueError("Context-view material contains unsupported resource references.")
            for payload in payloads:
                attachment = file_attachment_from_payload(payload)
                if attachment is None:
                    raise ValueError(
                        "Context-view material contains unsupported resource references."
                    )
                previous = references.get(attachment.artifact_id)
                if previous is not None and not same_file_attachment_reference(
                    previous, attachment
                ):
                    raise ValueError("Context-view attachment references conflict.")
                references[attachment.artifact_id] = attachment
    return tuple(references.values())


class ContextViewSelectionRequest(ContractValue):
    """Exact selection intent; source ownership is checked by SessionStore."""

    source_owner: OwnerRef
    source_session_id: Identifier
    source_session_instance_id: Identifier
    selector: Literal["exact", "latest", "minimum"]
    exact_view_id: Identifier | None = None
    minimum_transcript_cursor: StrictInt | None = Field(default=None, ge=0)
    projection_schema: Identifier
    extension_set_commitment: StrictStr
    limits: ContextViewLimits
    selection_key: StrictStr

    @field_validator("selection_key")
    @classmethod
    def validate_selection_key(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "selection_key")
        if len(value.encode("utf-8")) > 512:
            raise ValueError("selection_key must be at most 512 UTF-8 bytes.")
        return value

    @model_validator(mode="after")
    def validate_selector(self) -> ContextViewSelectionRequest:
        if self.selector == "exact" and self.exact_view_id is None:
            raise ValueError("Exact selection requires a view ID.")
        if self.selector != "exact" and self.exact_view_id is not None:
            raise ValueError("Only exact selection may carry a view ID.")
        if self.selector == "minimum" and self.minimum_transcript_cursor is None:
            raise ValueError("Minimum selection requires a transcript cursor.")
        return self


def context_view_manifest_bytes(manifest: ContextViewManifest) -> int:
    """Return the canonical bounded size used by every retention admission."""

    if type(manifest) is not ContextViewManifest:
        raise TypeError("Context-view size requires a typed manifest.")
    return len(
        canonical_bounded_durable_json_bytes(
            manifest.model_dump(mode="json"),
            "context view manifest",
            max_bytes=CONTEXT_VIEW_MAX_JSON_BYTES,
            max_nodes=8192,
            max_nesting=64,
        )
    )


class ContextViewSelectionReceipt(ContractValue):
    selection_key: StrictStr
    view: ContextViewManifest
    owner: OwnerRef
    state: Literal["selected", "adopted", "transferred", "released", "expired"]
    pin_commitment: StrictStr
    expires_at_ms: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    ownership_revision: StrictInt = Field(default=1, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    owner_participant: ParticipantRef | None = None
    schema_version: Literal[1] = 1


class ContextViewOwnershipRequest(ContractValue):
    """One fenced ownership transition for a selected context-view pin."""

    selection_key: StrictStr
    view_id: Identifier
    pin_commitment: StrictStr
    expected_state: Literal["selected", "adopted", "transferred"]
    expected_revision: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    operation: Literal["adopt", "transfer", "release"]
    current_owner: OwnerRef
    current_participant: ParticipantRef | None = None
    destination_owner: OwnerRef | None = None
    destination_participant: ParticipantRef | None = None
    operation_key: StrictStr

    @field_validator("selection_key", "operation_key", "pin_commitment")
    @classmethod
    def validate_nonblank_keys(cls, value: str, info) -> str:
        value = require_durable_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 512:
            raise ValueError(f"{info.field_name} must be at most 512 UTF-8 bytes.")
        if info.field_name == "operation_key" and value.startswith("expiry:"):
            raise ValueError(
                "Context-view operation keys may not use the reserved expiry namespace."
            )
        return value

    @model_validator(mode="after")
    def validate_transition(self) -> ContextViewOwnershipRequest:
        if self.operation in {"adopt", "transfer"} and self.destination_owner is None:
            raise ValueError("Ownership adoption and transfer require a destination owner.")
        if self.operation == "release" and self.destination_owner is not None:
            raise ValueError("Ownership release cannot carry a destination owner.")
        if (
            self.destination_participant is not None
            and self.destination_owner != self.destination_participant.owner
        ):
            raise ValueError("Destination participant owner conflicts with destination owner.")
        if (
            self.current_participant is not None
            and self.current_owner != self.current_participant.owner
        ):
            raise ValueError("Current participant owner conflicts with current owner.")
        if self.operation == "adopt" and self.expected_state != "selected":
            raise ValueError("Adoption requires a selected pin.")
        if self.operation == "transfer" and self.expected_state not in {"adopted", "transferred"}:
            raise ValueError("Transfer requires an adopted or transferred pin.")
        return self


class ContextViewLifecycleEvent(ContractValue):
    """Durable, public-safe evidence for one ownership transition."""

    event_id: Identifier
    operation_key: StrictStr
    selection_key: StrictStr
    view_id: Identifier
    state: Literal["adopted", "transferred", "released", "expired"]
    owner: OwnerRef
    owner_participant: ParticipantRef | None = None
    pin_commitment: StrictStr
    ownership_revision: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    status: Literal["committed"] = "committed"


class ContextViewReadback(ContractValue):
    """Bounded historical readback; never an executable session object."""

    view: ContextViewManifest
    historical_only: StrictBool = True
    schema_version: Literal[1] = 1

    @model_validator(mode="after")
    def require_historical_marker(self) -> ContextViewReadback:
        if self.historical_only is not True:
            raise ValueError("Context-view readback must remain historical-only.")
        return self
