"""Construct exact interaction-start evidence for session admission."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import uuid4

from cayu._validation import require_clean_nonblank
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._policy_wire import decode as decode_policy_evidence
from cayu.sessions._interaction_transitions import (
    QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY,
    QueuedInteractionProfileHandoff,
    queued_interaction_profile_handoff_evidence,
)
from cayu.sessions.interactions import (
    InteractionStatus,
    InteractionSummaryEvidence,
)
from cayu.sessions.records import Session
from cayu.tools.grants import (
    PreparedTargetedToolGrant,
    prepared_targeted_tool_grant_batch_fingerprint,
)


def build_interaction_started_event(
    *,
    clock: Callable[[], datetime],
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    interaction_id: str,
    targeted_tool_grants: tuple[PreparedTargetedToolGrant, ...] = (),
    queued_profile_handoff: QueuedInteractionProfileHandoff | None = None,
    policy_evidence: bytes | None = None,
    event_id: str | None = None,
    started_at: datetime | None = None,
) -> Event:
    return build_interaction_started_event_from_identity(
        clock=clock,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        interaction_id=interaction_id,
        event_id=event_id,
        started_at=started_at,
        targeted_tool_grants=targeted_tool_grants,
        queued_profile_handoff=queued_profile_handoff,
        policy_evidence=policy_evidence,
    )


def build_interaction_started_event_from_identity(
    *,
    clock: Callable[[], datetime],
    session_id: str,
    agent_name: str,
    environment_name: str | None,
    interaction_id: str,
    event_id: str | None = None,
    started_at: datetime | None = None,
    targeted_tool_grants: tuple[PreparedTargetedToolGrant, ...] = (),
    queued_profile_handoff: QueuedInteractionProfileHandoff | None = None,
    policy_evidence: bytes | None = None,
) -> Event:
    event_id = str(uuid4()) if event_id is None else require_clean_nonblank(event_id, "event_id")
    started_at = clock() if started_at is None else started_at
    evidence = InteractionSummaryEvidence(
        status=InteractionStatus.ACTIVE,
        start_event_id=event_id,
        started_at=started_at,
        targeted_tool_grant_count=(len(targeted_tool_grants) if targeted_tool_grants else None),
        targeted_tool_grant_batch_fingerprint=(
            prepared_targeted_tool_grant_batch_fingerprint(targeted_tool_grants)
            if targeted_tool_grants
            else None
        ),
    )
    return event_with_runtime_payload_authority(
        event_with_runtime_envelope_authority(
            event_with_runtime_generated_id(
                Event(
                    id=event_id,
                    type=EventType.INTERACTION_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                    timestamp=started_at,
                    agent_name=agent_name,
                    environment_name=environment_name,
                    payload={
                        **evidence.model_dump(mode="json"),
                        **(
                            {}
                            if policy_evidence is None
                            else {"model_policy": decode_policy_evidence(policy_evidence)}
                        ),
                        **(
                            {}
                            if queued_profile_handoff is None
                            else {
                                QUEUED_INTERACTION_PROFILE_HANDOFF_PAYLOAD_KEY: (
                                    queued_interaction_profile_handoff_evidence(
                                        queued_profile_handoff
                                    )
                                )
                            }
                        ),
                    },
                )
            ),
            "session_id",
            "interaction_id",
        ),
        "start_event_id",
    )
