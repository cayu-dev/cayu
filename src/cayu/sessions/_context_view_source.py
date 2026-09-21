"""Store-owned capture of one historical publication boundary.

Native stores call these pure validators inside their consistent read boundary.
No extension producer or user callback runs under that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cayu.events import Event, EventType
from cayu.runtime._model_completion_publication import (
    ModelStepPublicationCheckpoint,
    model_step_publication_from_checkpoint,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    execution_profile_from_session_metadata,
)
from cayu.sessions.base import (
    RuntimePublicationReceipt,
    Session,
    ToolCallPart,
    ToolResultPart,
    TranscriptRecord,
)
from cayu.sessions.context_views import CONTEXT_VIEW_MAX_MESSAGES, ParticipantSessionBinding


@dataclass(frozen=True, slots=True)
class ContextViewPublicationSource:
    """Detached internal evidence, never an execution permit or public readback."""

    session: Session
    binding: ParticipantSessionBinding
    pointer: ModelStepPublicationCheckpoint
    completion_event: Event
    records: tuple[TranscriptRecord, ...]
    execution_profile: ExecutionProfileIdentity
    transcript_end_cursor: int


def completed_boundary(
    session: Session | None,
    binding: ParticipantSessionBinding | None,
    checkpoint: dict[str, Any] | None,
) -> ModelStepPublicationCheckpoint:
    if session is None:
        raise LookupError("The source session incarnation is unavailable.")
    if binding is None or (binding.session_id, binding.session_instance_id) != (
        session.id,
        session.instance_id,
    ):
        raise PermissionError("The source session has no exact participant binding.")
    pointer = model_step_publication_from_checkpoint(checkpoint)
    if pointer is None or not (
        pointer.assistant_message_published or pointer.assistant_message_deferred
    ):
        raise ValueError("A context view requires a completed assistant turn.")
    if pointer.transcript_end_cursor - pointer.source_transcript_cursor > CONTEXT_VIEW_MAX_MESSAGES:
        raise ValueError("The completed-turn transcript boundary exceeds its limit.")
    if pointer.tool_round_id is not None:
        from cayu.runtime._tool_round_recovery import pending_tool_round_from_checkpoint

        if pending_tool_round_from_checkpoint(checkpoint) is not None:
            raise ValueError("A partial tool round cannot be published as a context view.")
    return pointer


def publication_frontier(
    pointer: ModelStepPublicationCheckpoint, receipt: RuntimePublicationReceipt | None
) -> int:
    """A deferred model pointer is completed only by its exact round receipt."""
    if pointer.tool_round_id is None:
        return pointer.transcript_end_cursor
    if (
        receipt is None
        or receipt.kind not in {"tool-round", "approval-close", "user-input-close"}
        or (
            receipt.kind == "tool-round"
            and (
                receipt.publication_id != f"tool-round:{pointer.tool_round_id}"
                or receipt.intent.get("round_id") != pointer.tool_round_id
            )
        )
        or receipt.intent.get("tool_round_id") != pointer.tool_round_id
        or receipt.intent.get("model_step_id") != pointer.logical_step_id
        or receipt.transcript_start_cursor != pointer.transcript_end_cursor
        or receipt.transcript_end_cursor != pointer.source_transcript_cursor + 2
    ):
        raise ValueError("The completed tool round has no exact publication receipt.")
    return receipt.transcript_end_cursor


def closed_round_publication_id(
    pointer: ModelStepPublicationCheckpoint, events: tuple[Event, ...]
) -> str:
    """Locate an exact closure receipt; the event alone never grants publication."""
    if len(events) != 1:
        raise ValueError("The completed tool round has no unique closure evidence.")
    event = events[0]
    payload = event.payload
    if (
        event.type is not EventType.SESSION_CHECKPOINTED
        or payload.get("tool_round_id") != pointer.tool_round_id
        or payload.get("model_step_id") != pointer.logical_step_id
    ):
        raise ValueError("The completed tool round has conflicting closure evidence.")
    if payload.get("cleared") is True:
        kind, identity = "approval-close", payload.get("approval_id")
    elif payload.get("transition") == "answered":
        kind, identity = "user-input-close", payload.get("input_id")
    else:
        raise ValueError("The completed tool round has no terminal closure evidence.")
    if type(identity) is not str or not identity:
        raise ValueError("The completed tool round has no closure identity.")
    return f"{kind}:{identity}"


def capture_source(
    session: Session,
    binding: ParticipantSessionBinding,
    checkpoint: dict[str, Any] | None,
    pointer: ModelStepPublicationCheckpoint,
    completion: Event | None,
    records: tuple[TranscriptRecord, ...],
    tool_receipt: RuntimePublicationReceipt | None = None,
) -> ContextViewPublicationSource:
    if (
        completion is None
        or completion.type is not EventType.MODEL_COMPLETED
        or completion.id != pointer.completion_event_id
        or completion.session_id != session.id
        or completion.interaction_id is None
    ):
        raise ValueError("The completed-turn checkpoint has no authoritative completion event.")
    end = publication_frontier(pointer, tool_receipt)
    if tuple(record.index for record in records) != tuple(
        range(pointer.source_transcript_cursor, end)
    ):
        raise ValueError("The completed-turn transcript boundary is not contiguous.")
    if tool_receipt is not None:
        calls = tuple(part for part in records[0].message.content if isinstance(part, ToolCallPart))
        results = records[1].message.content
        if (
            tool_receipt.session_id != session.id
            or tool_receipt.interaction_id != completion.interaction_id
            or not calls
            or records[0].message.role != "assistant"
            or records[1].message.role != "tool"
            or any(record.interaction_id != completion.interaction_id for record in records)
            or tool_receipt.intent.get("tool_call_ids") != [part.tool_call_id for part in calls]
            or any(
                (part.model_step_id, part.model_attempt_id, part.tool_round_id)
                != (
                    pointer.logical_step_id,
                    tool_receipt.intent.get("model_attempt_id"),
                    pointer.tool_round_id,
                )
                for part in calls
            )
            or any(not isinstance(part, ToolResultPart) for part in results)
            or [
                (part.tool_call_id, part.tool_name)
                for part in results
                if isinstance(part, ToolResultPart)
            ]
            != [(part.tool_call_id, part.tool_name) for part in calls]
        ):
            raise ValueError(
                "The completed tool round conflicts with its transcript or interaction."
            )
    fingerprint = completion.payload.get("execution_profile_fingerprint")
    if type(fingerprint) is not str or not fingerprint:
        raise ValueError("The completed turn has no execution-profile evidence.")
    from cayu.runtime._invocation_lifecycle import (
        _invocation_lifecycle_receipt_ledger_from_checkpoint,
    )

    ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
    matching = [
        receipt
        for receipt in ledger.receipts
        if (
            receipt.session_id == session.id
            and receipt.session_instance_id == session.instance_id
            and receipt.active_profile.interaction_id == completion.interaction_id
            and receipt.active_profile.profile.fingerprint == fingerprint
        )
    ]
    if matching:
        profile = matching[0].active_profile.profile
        historical_session = matching[0].result_session
    else:
        # A native producer can qualify an event without lifecycle history, but
        # only by positively matching its immutable completion fingerprint.
        profile = execution_profile_from_session_metadata(session.metadata)
        historical_session = session
        if profile is None or profile.fingerprint != fingerprint:
            raise ValueError("The completed turn's historical execution profile is unavailable.")
    return ContextViewPublicationSource(
        session=historical_session.model_copy(deep=True),
        binding=binding.model_copy(deep=True),
        pointer=pointer.model_copy(deep=True),
        completion_event=completion.model_copy(deep=True),
        records=tuple(record.model_copy(deep=True) for record in records),
        execution_profile=profile.model_copy(deep=True),
        transcript_end_cursor=end,
    )
