"""Shared session profile admission, rejection and model-transition rules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_json_value,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.events import Event, EventType, copy_event
from cayu.execution_profiles import (
    EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS,
    ExecutionProfileComponentClass,
    ExecutionProfileDecision,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    changed_execution_profile_components,
    copy_execution_profile_decision,
    direct_tool_capability_ceiling_component,
    execution_profile_changes_authority,
    execution_profile_provider_target_component,
    execution_profile_runtime_component,
)
from cayu.messages import Message, ProviderStatePart, ThinkingPart
from cayu.runtime._model_target import project_portable_transcript
from cayu.sessions import authority as session_authority_rules
from cayu.sessions import event_delivery as session_event_rules
from cayu.sessions import records as session_record_rules
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    execution_profile_from_session_metadata,
    execution_profile_metadata_after_adoption,
)
from cayu.sessions._model_failover import (
    MODEL_TARGET_PROJECTION_METADATA_KEY,
    MODEL_TARGET_PROJECTION_RECORD_TYPE,
    MODEL_TARGET_PROJECTION_SCHEMA_VERSION,
    ModelTarget,
)
from cayu.sessions.authority import (
    CheckpointValueAuthority,
    SessionRunFenced,
    checkpoint_value_authority,
)
from cayu.sessions.invocation import SessionInvocationBinding
from cayu.sessions.records import (
    RUNTIME_BUILD_PROVENANCE_METADATA_KEY,
    Session,
    SessionRuntimeIdentity,
    SessionStatus,
    SessionStatusConflict,
)
from cayu.sessions.requests import ResumeRequest, copy_resume_request
from cayu.sessions.transcript_input import session_input_messages_sha256
from cayu.tools.exposure import (
    ToolCapabilityCeiling,
    session_metadata_after_tool_capability_ceiling_narrowing,
    session_metadata_with_tool_capability_ceiling,
    tool_capability_ceiling_from_session_metadata,
)
from cayu.vaults.redaction import SecretRedactor


def session_model_projection_cursor(session: Session) -> int:
    """Return the durable prefix whose provider-native state is invalidated."""

    raw = session.metadata.get(MODEL_TARGET_PROJECTION_METADATA_KEY)
    if raw is None:
        return 0
    expected_keys = {
        "record_type",
        "schema_version",
        "provider_name",
        "model",
        "transcript_cursor",
    }
    if type(raw) is not dict or set(raw) != expected_keys:
        raise ValueError("Session model-target projection metadata is malformed.")
    if (
        raw.get("record_type") != MODEL_TARGET_PROJECTION_RECORD_TYPE
        or raw.get("schema_version") != MODEL_TARGET_PROJECTION_SCHEMA_VERSION
        or raw.get("provider_name") != session.provider_name
        or raw.get("model") != session.model
    ):
        raise ValueError("Session model-target projection metadata conflicts with the session.")
    cursor = raw.get("transcript_cursor")
    if type(cursor) is not int or not 0 <= cursor <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Session model-target projection cursor is malformed.")
    return cursor


@dataclass(frozen=True)
class SessionModelTransition:
    """Runtime-owned material committed with a clean-boundary model switch."""

    target: ModelTarget
    event: Event
    source_transcript_digest: str
    source_transcript_cursor: int


def execution_profile_adoption_request_fingerprint(
    request: ResumeRequest,
    *,
    redactor: SecretRedactor,
) -> str:
    """Bind an adoption idempotency key to the complete admitted resume request."""

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("Execution-profile adoption fingerprint requires a SecretRedactor.")
    copied = copy_resume_request(request)
    if copied.profile_adoption is None:
        raise ValueError("Execution-profile adoption request fingerprint requires adoption intent.")
    document = copied.model_dump(mode="json", warnings=False)
    adoption_document = document.get("profile_adoption")
    if type(adoption_document) is not dict:
        raise ValueError("Execution-profile adoption intent is not a durable object.")
    redactor.require_no_secret_keys(
        adoption_document,
        field_name="profile_adoption",
        match_short_substrings=True,
    )
    if redactor.redact_json_values(adoption_document) != adoption_document:
        raise ValueError("Execution-profile adoption audit fields contain a workload secret.")
    loop_policy_identities: list[dict[str, str]] = []
    for policy in copied.loop_policies:
        replay_identity = policy.adoption_replay_identity
        if replay_identity is None:
            raise ValueError(
                "Request loop policies used with explicit execution-profile adoption must "
                "provide a stable adoption_replay_identity."
            )
        replay_identity = require_clean_nonblank(
            replay_identity,
            "loop_policies.adoption_replay_identity",
        )
        if redactor.redact_text(replay_identity) != replay_identity:
            raise ValueError(
                "loop_policies.adoption_replay_identity contains a workload secret and "
                "cannot be used as durable replay authority."
            )
        if len(replay_identity) > EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS:
            raise ValueError(
                "loop_policies.adoption_replay_identity must not exceed "
                f"{EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS} characters."
            )
        loop_policy_identities.append(
            {
                "name": require_clean_nonblank(policy.name, "loop_policies.name"),
                "implementation": (
                    f"{require_clean_nonblank(type(policy).__module__, 'loop_policies.module')}:"
                    f"{require_clean_nonblank(type(policy).__qualname__, 'loop_policies.qualname')}"
                ),
                "adoption_replay_identity": replay_identity,
            }
        )
    document["loop_policies"] = loop_policy_identities
    return sha256(
        canonical_durable_json_bytes(
            document,
            "execution_profile_adoption_resume_request",
        )
    ).hexdigest()


def _copy_optional_execution_profile(
    profile: ExecutionProfileIdentity | None,
) -> ExecutionProfileIdentity | None:
    if profile is None:
        return None
    if type(profile) is not ExecutionProfileIdentity:
        raise TypeError("execution_profile must be an ExecutionProfileIdentity.")
    return ExecutionProfileIdentity.model_validate(profile.model_dump(mode="json"))


def _copy_optional_execution_profile_decision(
    decision: ExecutionProfileDecision | None,
) -> ExecutionProfileDecision | None:
    if decision is None:
        return None
    if type(decision) is not ExecutionProfileDecision:
        raise TypeError("execution_profile_decision must be an ExecutionProfileDecision.")
    return copy_execution_profile_decision(decision)


def _validate_execution_profile_admission(
    session: Session,
    *,
    candidate_profile: ExecutionProfileIdentity | None,
    model_transition: SessionModelTransition | None,
    decision: ExecutionProfileDecision | None,
) -> dict[str, Any] | None:
    if candidate_profile is None:
        if decision is not None:
            raise ValueError("An execution-profile decision requires a candidate profile.")
        return None
    expected_profile = execution_profile_from_session_metadata(session.metadata)
    changed = changed_execution_profile_components(expected_profile, candidate_profile)
    if decision is not None:
        if decision.expected_profile != expected_profile:
            raise SessionStatusConflict(
                "Session execution-profile expectation changed before admission."
            )
        if decision.candidate_profile != candidate_profile:
            raise ValueError("Execution-profile decision has a conflicting candidate profile.")
        if decision.changed_component_classes != changed:
            raise ValueError("Execution-profile decision has conflicting changed components.")
        if decision.event.session_id != session.id:
            raise ValueError("Execution-profile decision event has a conflicting session.")
        if (
            decision.event.agent_name != session.agent_name
            or decision.event.environment_name != session.environment_name
        ):
            raise ValueError("Execution-profile decision event has conflicting authority.")
        if decision.kind not in {
            ExecutionProfileDecisionKind.EXACT_REUSE,
            ExecutionProfileDecisionKind.COMPATIBLE_REUSE,
            ExecutionProfileDecisionKind.ADOPTED,
        }:
            raise ValueError("Only accepted execution-profile decisions can admit work.")
        if (
            decision.kind is ExecutionProfileDecisionKind.COMPATIBLE_REUSE
            and execution_profile_changes_authority(changed)
        ):
            raise ValueError("Generic compatible reuse cannot change execution authority.")
        provider_target_changed = ExecutionProfileComponentClass.PROVIDER_TARGET in changed
        if provider_target_changed != (model_transition is not None):
            raise ValueError(
                "Provider-target profile changes require the matching model transition."
            )
        if decision.kind is ExecutionProfileDecisionKind.ADOPTED:
            return execution_profile_metadata_after_adoption(
                session.metadata,
                candidate_profile,
            )
        return None
    if model_transition is None:
        if changed:
            changed_names = ", ".join(component.value for component in changed)
            raise SessionStatusConflict(
                "Session execution profile changed during admission: " + changed_names
            )
        return None
    if changed != (ExecutionProfileComponentClass.PROVIDER_TARGET,):
        raise SessionStatusConflict(
            "Model-target adoption must change exactly the provider-target profile component."
        )
    return execution_profile_metadata_after_adoption(
        session.metadata,
        candidate_profile,
    )


def _session_metadata_after_runtime_identity_adoption(
    session: Session,
    identity: SessionRuntimeIdentity | None,
    *,
    model_transition: SessionModelTransition | None,
    execution_profile_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Atomically compose adopted runtime provenance with profile metadata."""

    if identity is None:
        return execution_profile_metadata
    if execution_profile_metadata is None:
        raise ValueError("Runtime identity adoption requires adopted profile metadata.")
    target_provider_name = (
        session.provider_name if model_transition is None else model_transition.target.provider_name
    )
    target_model = session.model if model_transition is None else model_transition.target.model
    profile = execution_profile_from_session_metadata(execution_profile_metadata)
    provider_component = execution_profile_provider_target_component(
        target_provider_name,
        target_model,
    )
    if profile.component(ExecutionProfileComponentClass.PROVIDER_TARGET) != provider_component:
        raise ValueError("Adopted runtime metadata conflicts with the target provider identity.")
    runtime_component = execution_profile_runtime_component(
        identity.runtime_name,
        identity.runtime_version,
        identity.runtime_build_provenance,
    )
    if profile.component(ExecutionProfileComponentClass.RUNTIME) != runtime_component:
        raise ValueError("Adopted runtime metadata conflicts with the target runtime identity.")
    copied = copy_durable_json_value(execution_profile_metadata, "session.metadata")
    copied[RUNTIME_BUILD_PROVENANCE_METADATA_KEY] = identity.runtime_build_provenance.model_dump(
        mode="json"
    )
    return copied


def _session_metadata_after_tool_capability_ceiling_admission(
    session: Session,
    ceiling: ToolCapabilityCeiling | None,
    *,
    transition_metadata: dict[str, Any] | None,
    require_existing_ceiling: bool = False,
) -> dict[str, Any] | None:
    """Validate a ceiling CAS against the stored session and compose metadata."""

    if type(require_existing_ceiling) is not bool:
        raise TypeError("require_existing_ceiling must be a bool.")
    if ceiling is None and not require_existing_ceiling:
        return transition_metadata
    current_ceiling = tool_capability_ceiling_from_session_metadata(session.metadata)
    narrowed_metadata: dict[str, Any]
    if ceiling is None:
        admitted_ceiling = current_ceiling
    else:
        narrowed_metadata = session_metadata_after_tool_capability_ceiling_narrowing(
            session.metadata,
            ceiling,
        )
        admitted_ceiling = ceiling
    base = session.metadata if transition_metadata is None else transition_metadata
    admitted_profile = execution_profile_from_session_metadata(base)
    if admitted_profile.component(
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS
    ) != direct_tool_capability_ceiling_component(admitted_ceiling.tool_names):
        raise ValueError("Admitted execution profile conflicts with its tool capability ceiling.")
    if ceiling is None:
        return transition_metadata
    if transition_metadata is None and current_ceiling == ceiling:
        return None
    narrowed_ceiling = tool_capability_ceiling_from_session_metadata(narrowed_metadata)
    return session_metadata_with_tool_capability_ceiling(
        base,
        narrowed_ceiling,
    )


def _prepare_execution_profile_rejection(
    session_id: str,
    *,
    expected_statuses: set[SessionStatus],
    expected_run_epoch: int,
    expected_profile: ExecutionProfileIdentity,
    candidate_profile: ExecutionProfileIdentity,
    event: Event,
    decision: ExecutionProfileDecision | None = None,
) -> tuple[
    str,
    set[SessionStatus],
    int,
    ExecutionProfileIdentity,
    ExecutionProfileIdentity,
    Event,
]:
    session_id = require_clean_nonblank(session_id, "session_id")
    statuses = session_record_rules._validate_status_set(expected_statuses, "expected_statuses")
    if type(expected_run_epoch) is not int:
        raise TypeError("expected_run_epoch must be an integer.")
    if not 0 <= expected_run_epoch <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("expected_run_epoch exceeds the durable integer limit.")
    expected = ExecutionProfileIdentity.model_validate(expected_profile.model_dump(mode="json"))
    candidate = ExecutionProfileIdentity.model_validate(candidate_profile.model_dump(mode="json"))
    changed = changed_execution_profile_components(expected, candidate)
    if not changed:
        raise ValueError("Execution-profile rejection requires a changed candidate.")
    _, copied_events = session_event_rules._copy_session_event_batch(session_id, [event])
    copied_event = copied_events[0]
    copied_decision = _copy_optional_execution_profile_decision(decision)
    if copied_decision is not None:
        if copied_decision.kind not in {
            ExecutionProfileDecisionKind.MIGRATION_REQUIRED,
            ExecutionProfileDecisionKind.REJECTED,
        }:
            raise ValueError("Only non-admitting decisions can use profile rejection.")
        if (
            copied_decision.expected_profile != expected
            or copied_decision.candidate_profile != candidate
            or copied_decision.event != copied_event
        ):
            raise ValueError("Execution-profile rejection decision is inconsistent.")
        return (
            session_id,
            statuses,
            expected_run_epoch,
            expected,
            candidate,
            copied_event,
        )
    expected_payload = {
        "expected_profile_fingerprint": expected.fingerprint,
        "candidate_profile_fingerprint": candidate.fingerprint,
        "changed_component_classes": [component.value for component in changed],
    }
    if copied_event.type is not EventType.SESSION_EXECUTION_PROFILE_REJECTED:
        raise ValueError("Execution-profile rejection event has the wrong type.")
    if copied_event.interaction_id is not None:
        raise ValueError("Execution-profile rejection cannot belong to an admitted interaction.")
    if copied_event.payload != expected_payload:
        raise ValueError("Execution-profile rejection event payload does not match its profiles.")
    return (
        session_id,
        statuses,
        expected_run_epoch,
        expected,
        candidate,
        copied_event,
    )


def _validate_execution_profile_rejection_session(
    session: Session,
    *,
    checkpoint: dict[str, Any] | None,
    expected_session_instance_id: str | None = None,
    expected_statuses: set[SessionStatus],
    expected_run_epoch: int,
    expected_profile: ExecutionProfileIdentity,
    event: Event,
    expected_active_invocation_profile_authority: CheckpointValueAuthority | None = None,
) -> None:
    if expected_session_instance_id is not None:
        expected_session_instance_id = SessionInvocationBinding.validate_session_instance_id(
            expected_session_instance_id
        )
        if session.instance_id != expected_session_instance_id:
            raise SessionRunFenced(
                "Execution-profile rejection belongs to another session incarnation."
            )
    if session.status not in expected_statuses:
        raise SessionStatusConflict(
            f"Session status no longer permits profile rejection: {session.status}."
        )
    if session.run_epoch != expected_run_epoch:
        raise SessionRunFenced(
            f"Session run epoch changed before profile rejection: expected "
            f"{expected_run_epoch}, current {session.run_epoch}."
        )
    if expected_active_invocation_profile_authority is None:
        current_profile = execution_profile_from_session_metadata(session.metadata)
        if current_profile != expected_profile:
            raise RuntimeError("Session execution-profile expectation changed concurrently.")
    else:
        if type(expected_active_invocation_profile_authority) is not CheckpointValueAuthority:
            raise TypeError(
                "expected_active_invocation_profile_authority must be a CheckpointValueAuthority."
            )
        current_active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if current_active_profile is None or current_active_profile.profile != expected_profile:
            raise RuntimeError("Active invocation execution profile changed concurrently.")
        current_authority = checkpoint_value_authority(
            current_active_profile.model_dump(mode="json"),
            "active_invocation_execution_profile",
        )
        if current_authority != expected_active_invocation_profile_authority:
            raise RuntimeError("Active invocation execution profile changed concurrently.")
    if event.agent_name != session.agent_name or event.environment_name != session.environment_name:
        raise ValueError("Execution-profile rejection event does not match session authority.")


def _execution_profile_rejection_events_equivalent(left: Event, right: Event) -> bool:
    """Compare one idempotent rejection request while retaining its first timestamp."""

    left_document = left.model_dump(mode="json")
    right_document = right.model_dump(mode="json")
    left_document.pop("timestamp", None)
    right_document.pop("timestamp", None)
    return left_document == right_document


def _copy_session_model_transition(
    session_id: str,
    transition: SessionModelTransition | None,
    *,
    interaction_id: str | None,
    interaction_is_new: bool,
) -> SessionModelTransition | None:
    if transition is None:
        return None
    if type(transition) is not SessionModelTransition:
        raise TypeError("model_transition must be a SessionModelTransition.")
    if not interaction_is_new or interaction_id is None:
        raise ValueError("A model transition requires a newly admitted interaction.")
    if type(transition.target) is not ModelTarget:
        raise TypeError("model_transition.target must be a ModelTarget.")
    event = copy_event(transition.event)
    if event.type != EventType.SESSION_MODEL_SWITCHED:
        raise ValueError("model_transition.event must be session.model.switched.")
    if event.session_id != session_id:
        raise ValueError("model_transition.event belongs to a different session.")
    if event.interaction_id != interaction_id:
        raise ValueError("model_transition.event belongs to a different interaction.")
    source_digest = transition.source_transcript_digest
    session_authority_rules._require_raw_sha256_digest(source_digest)
    source_cursor = transition.source_transcript_cursor
    if type(source_cursor) is not int or not 0 <= source_cursor <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("model_transition.source_transcript_cursor is malformed.")
    return SessionModelTransition(
        target=ModelTarget(
            provider_name=transition.target.provider_name,
            model=transition.target.model,
        ),
        event=event,
        source_transcript_digest=source_digest,
        source_transcript_cursor=source_cursor,
    )


def _validate_session_model_transition(
    session: Session,
    current_transcript: Sequence[Message],
    current_transcript_cursor: int,
    transition: SessionModelTransition,
) -> None:
    if session_input_messages_sha256(tuple(current_transcript)) != (
        transition.source_transcript_digest
    ):
        raise SessionStatusConflict(
            "Session transcript changed while the model transition was being prepared."
        )
    if current_transcript_cursor != transition.source_transcript_cursor:
        raise SessionStatusConflict(
            "Session transcript cursor changed while the model transition was being prepared."
        )
    project_portable_transcript(list(current_transcript))
    target = transition.target
    if (session.provider_name, session.model) == (target.provider_name, target.model):
        raise ValueError("A model transition must change the provider or model.")
    provider_state_parts = sum(
        type(part) is ProviderStatePart
        for message in current_transcript
        for part in message.content
    )
    thinking_parts = sum(
        type(part) is ThinkingPart for message in current_transcript for part in message.content
    )
    expected_payload = {
        "source_provider_name": session.provider_name,
        "source_model": session.model,
        "target_provider_name": target.provider_name,
        "target_model": target.model,
        "provider_changed": session.provider_name != target.provider_name,
        "model_changed": session.model != target.model,
        "provider_state_parts_dropped": provider_state_parts,
        "thinking_parts_dropped": thinking_parts,
        "source_transcript_cursor": transition.source_transcript_cursor,
        "cache_state_dropped": True,
        "full_transcript_projection": True,
    }
    if transition.event.payload != expected_payload:
        raise ValueError("The model-transition event conflicts with its durable transition.")


def _session_metadata_after_model_transition(
    session: Session,
    transition: SessionModelTransition,
    *,
    execution_profile_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _session_metadata_with_model_projection(
        (session.metadata if execution_profile_metadata is None else execution_profile_metadata),
        target=transition.target,
        transcript_cursor=transition.source_transcript_cursor,
    )


def _session_metadata_with_model_projection(
    metadata: dict[str, Any],
    *,
    target: ModelTarget,
    transcript_cursor: int,
) -> dict[str, Any]:
    """Replace runtime-owned model projection authority on a metadata copy."""

    if type(target) is not ModelTarget:
        raise TypeError("target must be a ModelTarget.")
    if type(transcript_cursor) is not int or not 0 <= transcript_cursor <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError("Model-target projection cursor exceeds the durable integer limit.")
    copied = copy_durable_json_object(metadata, "session.metadata")
    copied.pop(MODEL_TARGET_PROJECTION_METADATA_KEY, None)
    if transcript_cursor == 0:
        return copied
    copied[MODEL_TARGET_PROJECTION_METADATA_KEY] = {
        "record_type": MODEL_TARGET_PROJECTION_RECORD_TYPE,
        "schema_version": MODEL_TARGET_PROJECTION_SCHEMA_VERSION,
        "provider_name": target.provider_name,
        "model": target.model,
        "transcript_cursor": transcript_cursor,
    }
    return copy_durable_json_object(copied, "session.metadata")
