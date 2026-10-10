"""Runtime-authoritative events for context construction and compaction."""

from __future__ import annotations

from cayu._validation import (
    copy_json_value,
)
from cayu.context.base import (
    ContextCompactionTelemetry,
    ContextRecallTelemetry,
    sanitize_context_compaction_telemetry,
)
from cayu.events import (
    Event,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ModelStepIdentity,
    copy_model_attempt_identity,
    copy_model_step_identity,
    strip_runtime_owned_execution_identity,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._model_event_authority import _event_with_model_identity_authority
from cayu.sessions.records import Session


def _context_observation_event(event: Event) -> Event:
    """Attest the runtime identities shared by context-observation events."""

    return event_with_runtime_payload_authority(
        event,
        "observation_id",
        "model_step_id",
        "model_attempt_id",
    )


def _context_compaction_telemetry_event(
    *,
    telemetry: ContextCompactionTelemetry,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    execution_identity: ModelStepIdentity | ModelAttemptIdentity | None = None,
    execution_profile: ExecutionProfileIdentity | None = None,
) -> Event:
    if type(telemetry) is not ContextCompactionTelemetry:
        raise TypeError(
            "Context compaction telemetry must be ContextCompactionTelemetry instances."
        )
    sanitized = sanitize_context_compaction_telemetry(telemetry)
    payload = copy_json_value(sanitized.payload, "payload")
    strip_runtime_owned_execution_identity(payload)
    if type(execution_identity) is ModelAttemptIdentity:
        payload.update(copy_model_attempt_identity(execution_identity).payload())
    elif type(execution_identity) is ModelStepIdentity:
        payload.update(copy_model_step_identity(execution_identity).payload())
    elif execution_identity is not None:
        raise TypeError("Context compaction execution identity has an unsupported type.")
    event = Event(
        type=sanitized.event_type,
        session_id=session.id,
        agent_name=registered_agent.spec.name,
        environment_name=environment_name,
        payload=payload,
    )
    event = (
        event
        if execution_identity is None
        else _event_with_model_identity_authority(event, execution_identity)
    )
    return event_with_execution_profile_authority(event, execution_profile)


def _context_recall_telemetry_event(
    *,
    telemetry: ContextRecallTelemetry,
    session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
    environment_name: str | None,
    model_step_identity: ModelStepIdentity,
    execution_profile: ExecutionProfileIdentity | None = None,
) -> Event:
    if type(telemetry) is not ContextRecallTelemetry:
        raise TypeError("Context recall telemetry must be ContextRecallTelemetry instances.")
    payload = copy_json_value(telemetry.payload, "payload")
    strip_runtime_owned_execution_identity(payload)
    payload.update(copy_model_step_identity(model_step_identity).payload())
    return event_with_execution_profile_authority(
        Event(
            type=telemetry.event_type,
            session_id=session.id,
            agent_name=registered_agent.spec.name,
            environment_name=environment_name,
            payload=payload,
        ),
        execution_profile,
    )
