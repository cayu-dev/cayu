"""Read-only native selection of an exact released participant whole turn.

Constructing or reconstructing this identity grants no writer or execution right.
The receiving owner recaptures it before admission; subsequent execution retains
its ordinary participant, profile, input and writer gates.
"""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Annotated

from pydantic import Field, StrictInt, StrictStr, field_validator

from cayu._validation import MAX_DURABLE_JSON_INTEGER, canonical_bounded_durable_json_bytes
from cayu.collaboration._contracts import MAX_DEPTH, MAX_NODES, ContractValue, Identifier
from cayu.collaboration.participants import ParticipantRef, VersionOne

if TYPE_CHECKING:
    from cayu.sessions._context_view_source import CompletedTurnSnapshot

Digest = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
Cursor = Annotated[StrictInt, Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)]


def require_continuation_selection_store(store) -> None:
    """Check qualification at consumers, including overrides that delegate reads."""
    version = type(store).__dict__.get("recipient_continuation_selection_version")
    if type(version) is not int or version != 1:
        raise NotImplementedError("Recipient continuation selection is not qualified.")


class RecipientContinuationSelection(ContractValue):
    """Bounded identity data, not continuation admission or current authorization."""

    schema_version: VersionOne = 1
    session_id: Identifier
    session_instance_id: Identifier
    participant: ParticipantRef
    run_epoch: Annotated[StrictInt, Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)]
    participant_binding_sha256: Digest
    checkpoint_sha256: Digest
    release_identity: Annotated[StrictStr, Field(min_length=1, max_length=1024)]
    release_sha256: Digest
    interaction_id: Identifier
    model_step_id: Identifier
    completion_event_id: Identifier
    transcript_end_cursor: Cursor
    execution_profile_json: StrictStr

    @field_validator("execution_profile_json")
    @classmethod
    def profile_snapshot(cls, value: str) -> str:
        from cayu.collaboration.prepared_admission import prepared_profile

        prepared_profile(value)
        return value


def select_continuation(snapshot: CompletedTurnSnapshot) -> RecipientContinuationSelection:
    """Qualify one coherent native snapshot with existing lifecycle validators."""
    from cayu.approvals.user_input import user_input_lifecycle_authority_from_checkpoint
    from cayu.collaboration.prepared_admission import MAX_PREPARED_PROFILE_BYTES
    from cayu.runtime._approval_support import pending_approval_from_checkpoint
    from cayu.runtime._invocation_lifecycle import (
        _require_released_invocation_command_receipt,
        invocation_checkpoint_state_sha256,
    )
    from cayu.runtime._session_continuation_store import (
        require_continuation_selection_quiescence,
    )
    from cayu.runtime._tool_round_recovery import pending_tool_round_from_checkpoint
    from cayu.runtime.execution_profiles import (
        active_invocation_execution_profile_from_checkpoint,
        execution_profile_from_session_metadata,
    )
    from cayu.sessions.base import SessionStatus

    session = snapshot.current_session
    source = snapshot.publication
    checkpoint = snapshot.checkpoint
    if (
        session.status is not SessionStatus.COMPLETED
        or snapshot.has_queued_input
        or snapshot.has_closure_owner
        or snapshot.has_active_model_stage
        or snapshot.current_transcript_cursor != source.transcript_end_cursor
    ):
        raise ValueError("Recipient continuation requires a quiescent completed turn.")
    require_continuation_selection_quiescence(checkpoint, session=session)
    pending_input, resolution = user_input_lifecycle_authority_from_checkpoint(
        checkpoint, current_run_epoch=session.run_epoch, runtime_session=session
    )
    if (
        pending_input is not None
        or resolution is not None
        or pending_approval_from_checkpoint(checkpoint) is not None
        or pending_tool_round_from_checkpoint(checkpoint, runtime_session=session) is not None
    ):
        raise ValueError("Recipient continuation cannot bypass pending recovery or human input.")
    active = active_invocation_execution_profile_from_checkpoint(checkpoint)
    profile = execution_profile_from_session_metadata(session.metadata)
    if active is None or profile is None:
        raise ValueError("Recipient continuation requires positive invocation evidence.")
    release = _require_released_invocation_command_receipt(
        session,
        checkpoint,
        session_id=session.id,
        session_instance_id=session.instance_id,
        active_profile=active,
    )
    if (
        active.interaction_id != source.completion_event.interaction_id
        or active.profile != source.execution_profile
        or profile != active.profile
        or source.binding.session_id != session.id
        or source.binding.session_instance_id != session.instance_id
    ):
        raise ValueError("Recipient continuation frontier conflicts with its current profile.")
    assert source.completion_event.interaction_id is not None
    return RecipientContinuationSelection(
        session_id=session.id,
        session_instance_id=session.instance_id,
        participant=source.binding.participant,
        run_epoch=session.run_epoch,
        participant_binding_sha256=sha256(
            canonical_bounded_durable_json_bytes(
                source.binding.model_dump(mode="json"),
                "recipient participant binding",
                max_bytes=512 * 1024,
                max_nodes=8192,
                max_nesting=64,
            )
        ).hexdigest(),
        checkpoint_sha256=invocation_checkpoint_state_sha256(checkpoint),
        release_identity=release.command_identity,
        release_sha256=release.record_sha256,
        interaction_id=source.completion_event.interaction_id,
        model_step_id=source.pointer.logical_step_id,
        completion_event_id=source.completion_event.id,
        transcript_end_cursor=source.transcript_end_cursor,
        execution_profile_json=canonical_bounded_durable_json_bytes(
            profile.model_dump(mode="json"),
            "prepared profile",
            max_bytes=MAX_PREPARED_PROFILE_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        ).decode("utf-8"),
    )
