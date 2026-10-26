"""Shared runtime-publication preparation, exact receipts and replay validation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from cayu._validation import MAX_DURABLE_JSON_ENVELOPE_NESTING, copy_durable_json_object
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, copy_event
from cayu.messages import Message, detach_message
from cayu.sessions import _checkpoint_publication as checkpoint_publication
from cayu.sessions import _operation_records as operation_record_rules
from cayu.sessions import _runtime_publication as publication_contracts
from cayu.sessions import authority as session_authority_rules
from cayu.sessions import records as session_record_rules
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationCheckpointOperation,
    RuntimePublicationMutation,
    SessionRuntimePublicationConflict,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions._retention_attestation import (
    ReceiptMaterial,
    RetentionCompactionAttestation,
)
from cayu.sessions._runtime_publication import (
    RUNTIME_PUBLICATION_RECORD_TYPE,
    RUNTIME_PUBLICATION_SCHEMA_VERSION,
    RuntimePublicationReceipt,
    RuntimePublicationRequest,
)
from cayu.sessions._workspace_observation_publication import (
    _validate_workspace_observation_publication,
)
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.records import Session, SessionStatus


@dataclass(frozen=True)
class _PreparedRuntimePublication:
    session_id: str
    storage_key: str
    request: RuntimePublicationRequest
    request_digest: str
    transcript_payloads: tuple[dict[str, Any], ...]
    event_payloads: tuple[dict[str, Any], ...]
    transcript_digest: str
    events_digest: str
    expected_statuses: frozenset[SessionStatus] | None
    expected_run_epoch: int | None
    expected_transcript_cursor: int | None
    checkpoint_codec: checkpoint_publication._RuntimePublicationCheckpointCodec | None = None


def runtime_publication_request_digest(
    session_id: str,
    request: RuntimePublicationRequest,
    *,
    include_checkpoint_schema: bool = False,
) -> str:
    """Return the exact digest a store must bind to a runtime publication."""

    if type(include_checkpoint_schema) is not bool:
        raise TypeError("include_checkpoint_schema must be a boolean.")
    prepared_request = request
    if include_checkpoint_schema:
        if any(
            operation.key == CHECKPOINT_SCHEMA_VERSION_KEY
            for operation in request.mutation.operations
        ):
            raise ValueError("Runtime publication request already carries a schema operation.")
        prepared_request = RuntimePublicationRequest(
            publication_id=request.publication_id,
            kind=request.kind,
            interaction_id=request.interaction_id,
            intent=request.intent,
            mutation=RuntimePublicationMutation(
                operations=(
                    *request.mutation.operations,
                    RuntimePublicationCheckpointOperation(
                        key=CHECKPOINT_SCHEMA_VERSION_KEY,
                        expected_value_digest=runtime_publication_checkpoint_value_digest(
                            CURRENT_CHECKPOINT_SCHEMA_VERSION
                        ),
                        action="set",
                        value=CURRENT_CHECKPOINT_SCHEMA_VERSION,
                    ),
                )
            ),
            transcript_messages=request.transcript_messages,
            events=request.events,
            operation_record_mutations=request.operation_record_mutations,
            referenced_events=request.referenced_events,
            argument_continuity=request.argument_continuity,
        )
    return _prepare_runtime_publication(
        session_id,
        prepared_request,
        expected_statuses=None,
        expected_run_epoch=None,
        expected_transcript_cursor=None,
    ).request_digest


def _prepare_runtime_publication(
    session_id: str,
    request: RuntimePublicationRequest,
    *,
    expected_statuses: set[SessionStatus] | None,
    expected_run_epoch: int | None,
    expected_transcript_cursor: int | None,
) -> _PreparedRuntimePublication:
    session_id = require_clean_nonblank(session_id, "session_id")
    if type(request) is not RuntimePublicationRequest:
        raise TypeError("Runtime publication requires a RuntimePublicationRequest.")
    try:
        copied_request = RuntimePublicationRequest(
            publication_id=request.publication_id,
            kind=request.kind,
            interaction_id=request.interaction_id,
            intent=request.intent,
            mutation=request.mutation,
            transcript_messages=request.transcript_messages,
            events=request.events,
            operation_record_mutations=request.operation_record_mutations,
            referenced_events=request.referenced_events,
            argument_continuity=request.argument_continuity,
        )
    except AttributeError as exc:
        raise ValueError("Runtime publication request is malformed.") from exc

    _validate_workspace_observation_publication(copied_request, session_id=session_id)
    if copied_request.kind in {"approval-close", "user-input-close"}:
        from cayu.sessions._foreground_child_checkpoint import (
            FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY,
        )

        continuation = next(
            (
                operation
                for operation in copied_request.mutation.operations
                if operation.key == FOREGROUND_CHILD_POST_ACTION_CONTINUATION_KEY
            ),
            None,
        )
        expected_digest = copied_request.intent.get("post_action_continuation_digest")
        if continuation is not None:
            if (
                continuation.action != "set"
                or expected_digest
                != runtime_publication_checkpoint_value_digest(continuation.value)
            ):
                raise ValueError("Action-close continuation conflicts with its publication intent.")
        elif expected_digest is not None:
            raise ValueError("Action-close continuation digest requires a continuation.")
        parent_continuation = next(
            (
                op
                for op in copied_request.mutation.operations
                if op.key == "foreground_parent_continuation"
            ),
            None,
        )
        expected_parent = copied_request.intent.get("foreground_parent_continuation")
        if parent_continuation is not None:
            if parent_continuation.action != "set" or parent_continuation.value != expected_parent:
                raise ValueError("Gate parent continuation conflicts with its close receipt.")
        elif expected_parent is not None:
            raise ValueError("Gate close receipt requires its parent continuation mutation.")
    if copied_request.argument_continuity is not None and copied_request.kind != "tool-round":
        raise ValueError("Private argument continuity requires a tool-round publication.")
    operation_record_rules._validate_session_operation_record_keys(
        {mutation.key: mutation.value for mutation in copied_request.operation_record_mutations}
    )

    appended_event_ids = {event.id for event in copied_request.events}
    referenced_event_ids = set(
        publication_contracts._runtime_publication_referenced_event_ids(
            copied_request.referenced_events
        )
    )
    overlap = appended_event_ids & referenced_event_ids
    if overlap:
        raise ValueError(
            f"Runtime publication cannot both append and reference event id {min(overlap)}."
        )
    for event in copied_request.events:
        if event.session_id != session_id:
            raise ValueError("Runtime publication event session_id does not match target session.")
        if event.interaction_id != copied_request.interaction_id:
            raise ValueError(
                "Runtime publication event interaction_id does not match its transcript "
                "attribution."
            )

    allowed_statuses = (
        None
        if expected_statuses is None
        else frozenset(
            session_record_rules._validate_status_set(expected_statuses, "expected_statuses")
        )
    )
    expected_run_epoch = session_authority_rules._validate_optional_runtime_publication_fence(
        expected_run_epoch,
        "expected_run_epoch",
    )
    expected_transcript_cursor = (
        session_authority_rules._validate_optional_runtime_publication_fence(
            expected_transcript_cursor,
            "expected_transcript_cursor",
        )
    )
    transcript_payloads = tuple(
        message.model_dump(mode="json") for message in copied_request.transcript_messages
    )
    event_payloads = tuple(event.model_dump(mode="json") for event in copied_request.events)
    transcript_digest = session_authority_rules._canonical_runtime_publication_digest(
        list(transcript_payloads), max_nesting=MAX_DURABLE_JSON_ENVELOPE_NESTING
    )
    events_digest = session_authority_rules._canonical_runtime_publication_digest(
        list(event_payloads)
    )
    request_digest = session_authority_rules._canonical_runtime_publication_digest(
        {
            "session_id": session_id,
            "publication_id": copied_request.publication_id,
            "kind": copied_request.kind,
            "interaction_id": copied_request.interaction_id,
            "intent": copied_request.intent,
            "mutation": copied_request.mutation.model_dump(mode="json"),
            "transcript_messages": list(transcript_payloads),
            "events": list(event_payloads),
            "referenced_events": [
                reference.model_dump(mode="json") for reference in copied_request.referenced_events
            ],
            **(
                {"argument_continuity": copied_request.argument_continuity.model_dump(mode="json")}
                if copied_request.argument_continuity is not None
                else {}
            ),
        },
        max_nesting=MAX_DURABLE_JSON_ENVELOPE_NESTING,
    )
    return _PreparedRuntimePublication(
        session_id=session_id,
        storage_key=publication_contracts._runtime_publication_storage_key(
            copied_request.publication_id
        ),
        request=copied_request,
        request_digest=request_digest,
        transcript_payloads=transcript_payloads,
        event_payloads=event_payloads,
        transcript_digest=transcript_digest,
        events_digest=events_digest,
        expected_statuses=allowed_statuses,
        expected_run_epoch=expected_run_epoch,
        expected_transcript_cursor=expected_transcript_cursor,
        checkpoint_codec=checkpoint_publication._active_runtime_publication_checkpoint_codec(),
    )


def _next_runtime_publication_timestamp(session: Session) -> datetime:
    now = datetime.now(UTC)
    persisted = max(session.updated_at.astimezone(UTC), session.last_activity_at.astimezone(UTC))
    if now <= persisted:
        return persisted + timedelta(microseconds=1)
    return now


def _build_runtime_publication_receipt(
    prepared: _PreparedRuntimePublication,
    *,
    source_session: Session,
    checkpoint: dict[str, Any] | None,
    transcript_start_cursor: int,
    published_at: datetime,
) -> RuntimePublicationReceipt:
    request = prepared.request
    receipt_payload: dict[str, Any] = {
        "record_type": RUNTIME_PUBLICATION_RECORD_TYPE,
        "schema_version": RUNTIME_PUBLICATION_SCHEMA_VERSION,
        "session_id": prepared.session_id,
        "publication_id": request.publication_id,
        "kind": request.kind,
        "interaction_id": request.interaction_id,
        "intent": request.intent,
        "request_digest": prepared.request_digest,
        "checkpoint_digest": session_authority_rules._canonical_runtime_publication_digest(
            checkpoint
        ),
        "transcript_digest": prepared.transcript_digest,
        "events_digest": prepared.events_digest,
        "source_status": str(source_session.status),
        "source_run_epoch": source_session.run_epoch,
        "transcript_start_cursor": transcript_start_cursor,
        "transcript_end_cursor": transcript_start_cursor + len(request.transcript_messages),
        "appended_event_ids": [event.id for event in request.events],
        "referenced_events": [
            reference.model_dump(mode="json") for reference in request.referenced_events
        ],
        "published_at": published_at.isoformat().replace("+00:00", "Z"),
    }
    publication_digest = session_authority_rules._canonical_runtime_publication_digest(
        receipt_payload
    )
    return RuntimePublicationReceipt(
        **receipt_payload,
        publication_digest=publication_digest,
    )


def _runtime_publication_receipt_record(
    receipt: RuntimePublicationReceipt,
) -> dict[str, Any]:
    return copy_durable_json_object(
        receipt.model_dump(mode="json"),
        "runtime_publication_receipt",
    )


def _reconstruct_runtime_publication_receipt(
    record: dict[str, Any],
    *,
    storage_key: str,
    session_id: str,
    publication_id: str,
    request_digest: str | None = None,
) -> RuntimePublicationReceipt:
    try:
        if type(record) is not dict:
            raise TypeError
        if set(record) != set(RuntimePublicationReceipt.model_fields):
            raise ValueError
        string_fields = (
            "record_type",
            "session_id",
            "publication_id",
            "kind",
            "request_digest",
            "publication_digest",
            "checkpoint_digest",
            "transcript_digest",
            "events_digest",
            "source_status",
            "published_at",
        )
        if any(type(record[field]) is not str for field in string_fields):
            raise TypeError
        if type(record["schema_version"]) is not int:
            raise TypeError
        if type(record["intent"]) is not dict:
            raise TypeError
        if record["interaction_id"] is not None and type(record["interaction_id"]) is not str:
            raise TypeError
        for field in (
            "source_run_epoch",
            "transcript_start_cursor",
            "transcript_end_cursor",
        ):
            if type(record[field]) is not int:
                raise TypeError
        if type(record["appended_event_ids"]) is not list:
            raise TypeError
        if type(record["referenced_events"]) is not list:
            raise TypeError
        receipt = RuntimePublicationReceipt.model_validate(record)
        if receipt.record_type != RUNTIME_PUBLICATION_RECORD_TYPE:
            raise ValueError
        if receipt.schema_version != RUNTIME_PUBLICATION_SCHEMA_VERSION:
            raise ValueError
        if receipt.session_id != session_id or receipt.publication_id != publication_id:
            raise ValueError
        if (
            publication_contracts._runtime_publication_storage_key(receipt.publication_id)
            != storage_key
        ):
            raise ValueError
        digest_payload = receipt.model_dump(mode="json", exclude={"publication_digest"})
        if (
            session_authority_rules._canonical_runtime_publication_digest(digest_payload)
            != receipt.publication_digest
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication receipt is malformed or conflicts with its key."
        ) from exc
    if request_digest is not None and receipt.request_digest != request_digest:
        raise SessionRuntimePublicationConflict(
            "The runtime publication id was already used for a different request."
        )
    return receipt


def _validate_runtime_publication_replay_receipt(
    receipt: RuntimePublicationReceipt,
    prepared: _PreparedRuntimePublication,
) -> None:
    request = prepared.request
    if (
        receipt.kind != request.kind
        or receipt.interaction_id != request.interaction_id
        or not session_authority_rules._runtime_publication_json_equal(
            receipt.intent, request.intent
        )
        or receipt.transcript_digest != prepared.transcript_digest
        or receipt.events_digest != prepared.events_digest
        or receipt.appended_event_ids != tuple(event.id for event in request.events)
        or receipt.referenced_events != request.referenced_events
        or receipt.transcript_end_cursor - receipt.transcript_start_cursor
        != len(request.transcript_messages)
    ):
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication receipt conflicts with its request material."
        )


def _validate_runtime_publication_receipt_material(
    receipt: RuntimePublicationReceipt,
    material: ReceiptMaterial,
    compaction_attestation: RetentionCompactionAttestation | None = None,
) -> None:
    """Validate loaded receipt material; see the function below."""

    _validate_runtime_publication_durable_material(
        receipt,
        transcript_messages=material.transcript,
        transcript_interaction_ids=material.transcript_interaction_ids,
        appended_events=material.appended_events,
        durable_referenced_events=material.referenced_events,
        compaction_attestation=compaction_attestation,
    )


def _validate_runtime_publication_durable_material(
    receipt: RuntimePublicationReceipt,
    *,
    transcript_messages: Iterable[Message],
    transcript_interaction_ids: Iterable[str | None],
    appended_events: Iterable[Event],
    durable_referenced_events: Iterable[Event],
    compaction_attestation: RetentionCompactionAttestation | None = None,
) -> None:
    """Fail closed when a receipt no longer describes its committed durable material.

    Checkpoints intentionally are not re-read here: later legitimate checkpoint
    transforms replace that mutable snapshot. ``checkpoint_digest`` remains
    commit-time evidence, while transcript/event records are immutable material
    that must continue to match on every load and replay. The one exception is
    storage-retention compaction of large tool-output bodies: a digest that
    differs from the receipt is accepted only when ``compaction_attestation``
    is bound to this exact receipt and names the compacted digest.
    """

    if compaction_attestation is not None and not compaction_attestation.binds(receipt):
        compaction_attestation = None

    try:
        transcript = tuple(detach_message(message) for message in transcript_messages)
        interaction_ids = tuple(transcript_interaction_ids)
        events = tuple(copy_event(event) for event in appended_events)
        durable_references = tuple(copy_event(event) for event in durable_referenced_events)
    except (AttributeError, KeyError, TypeError, UnicodeError, ValueError) as exc:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication material is malformed."
        ) from exc
    expected_transcript_count = receipt.transcript_end_cursor - receipt.transcript_start_cursor
    if len(transcript) != expected_transcript_count:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication transcript segment is missing or truncated."
        )
    if len(interaction_ids) != expected_transcript_count or any(
        interaction_id != receipt.interaction_id for interaction_id in interaction_ids
    ):
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication transcript attribution conflicts with its receipt."
        )
    transcript_digest = session_authority_rules._canonical_runtime_publication_digest(
        [message.model_dump(mode="json") for message in transcript],
        max_nesting=MAX_DURABLE_JSON_ENVELOPE_NESTING,
    )
    if transcript_digest != receipt.transcript_digest and (
        compaction_attestation is None
        or transcript_digest != compaction_attestation.compacted.transcript_digest
    ):
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication transcript segment conflicts with its receipt."
        )

    if tuple(event.id for event in events) != receipt.appended_event_ids:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication event batch is missing or reordered."
        )
    events_digest = session_authority_rules._canonical_runtime_publication_digest(
        [event.model_dump(mode="json") for event in events]
    )
    if events_digest != receipt.events_digest and (
        compaction_attestation is None
        or events_digest != compaction_attestation.compacted.events_digest
    ):
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication event batch conflicts with its receipt."
        )
    if any(event.interaction_id != receipt.interaction_id for event in events):
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication event attribution conflicts with its receipt."
        )

    referenced_event_ids = publication_contracts._runtime_publication_referenced_event_ids(
        receipt.referenced_events
    )
    if tuple(event.id for event in durable_references) != referenced_event_ids:
        raise SessionRuntimePublicationConflict(
            "The durable runtime publication references a missing event."
        )
    compacted_references = (
        None
        if compaction_attestation is None
        else compaction_attestation.compacted.referenced_event_digests
    )
    for index, (reference, durable_event) in enumerate(
        zip(
            receipt.referenced_events,
            durable_references,
            strict=True,
        )
    ):
        event_digest = publication_contracts._runtime_publication_event_digest(durable_event)
        if event_digest != reference.event_digest and (
            compacted_references is None
            or index >= len(compacted_references)
            or event_digest != compacted_references[index]
        ):
            raise SessionRuntimePublicationConflict(
                "The durable runtime publication referenced event content conflicts with "
                "its receipt."
            )
        if durable_event.interaction_id != receipt.interaction_id:
            raise SessionRuntimePublicationConflict(
                "The durable runtime publication referenced event attribution conflicts "
                "with its receipt."
            )
