"""Session-fork evidence, execution-profile contracts and prompt rules."""

from __future__ import annotations

import traceback as traceback_module
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_json_value,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.approvals.actors import ResolutionActor, copy_resolution_actor, resolution_actor_payload
from cayu.events import Event, EventType, copy_event
from cayu.execution_profiles import (
    EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS,
    EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    changed_execution_profile_components,
    direct_tool_capability_ceiling_component,
    inherited_execution_profile_component_changes,
)
from cayu.messages import Message, MessageRole, detach_message
from cayu.sessions import event_delivery as session_event_rules
from cayu.sessions import records as session_record_rules
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_matches_session_epoch,
    execution_profile_baseline_from_session_metadata,
    execution_profile_from_session_metadata,
)
from cayu.sessions._model_failover import MODEL_FAILOVER_CHECKPOINT_KEY, ModelFailoverSelection
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.sessions.invocation import (
    SessionExecutionSource,
    SessionInvocation,
    inherited_session_invocation,
)
from cayu.sessions.records import CheckpointTransform, Session, SessionStatus, copy_session
from cayu.sessions.transcript_queries import ForkTranscriptValidator
from cayu.tools.exposure import tool_capability_ceiling_from_session_metadata

FORK_EXECUTION_PROFILE_METADATA_KEY = "cayu:fork_execution_profile"
FORK_SOURCE_SNAPSHOT_METADATA_KEY = "cayu:fork_source_snapshot"
FORK_EXECUTION_PROFILE_RECORD_TYPE = "cayu.session-fork-execution-profile"
FORK_EXECUTION_PROFILE_ORDINARY_SCHEMA_VERSION = 1
FORK_EXECUTION_PROFILE_EXACT_SOURCE_SCHEMA_VERSION = 2
PROMPT_ANATOMY_TRANSITION_METADATA_KEY = "cayu:prompt_anatomy_transition"
PROMPT_ANATOMY_TRANSITION_RECORD_TYPE = "cayu.prompt-anatomy-transition"
PROMPT_ANATOMY_TRANSITION_SCHEMA_VERSION = 1


class ForkSystemPromptPolicy(StrEnum):
    """Select which body's system prompt becomes authoritative in a session fork."""

    INHERIT_SOURCE = "inherit_source"
    CURRENT_AGENT = "current_agent"


class ForkExecutionProfileSelection(StrEnum):
    """Select the immutable execution-profile baseline for a child session."""

    INHERIT_PARENT = "inherit_parent"
    CURRENT_CHILD = "current_child"


class ForkExecutionProfileSource(StrEnum):
    """Durable parent authority from which the fork baseline was selected."""

    SESSION_EXPECTED = "session_expected"
    ACTIVE_INVOCATION = "active_invocation"


class ForkSourceSnapshot(BaseModel):
    """Exact, caller-visible authority for one safe session-fork source."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    source_session_id: str
    source_instance_fingerprint: str
    status: SessionStatus
    run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    transcript_cursor: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    transcript_sha256: str
    checkpoint_sha256: str
    execution_profile_fingerprint: str
    causal_budget_id: str

    @field_validator("source_session_id", "causal_budget_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "source_instance_fingerprint",
        "transcript_sha256",
        "checkpoint_sha256",
        "execution_profile_fingerprint",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: SessionStatus) -> SessionStatus:
        if value not in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            raise ValueError("A fork source snapshot requires a terminal session status.")
        return value


def fork_source_state_sha256(snapshot: ForkSourceSnapshot) -> str:
    """Bind secret-independent exact-source continuity into one durable digest.

    Source-session and causal-budget identities are intentionally excluded. They
    are independently authenticated by the keyed fork-request identity and the
    child relationship, while including them here would turn this public
    commitment into an offline oracle for identifiers hidden by redaction.
    """

    if type(snapshot) is not ForkSourceSnapshot:
        raise TypeError("snapshot must be a ForkSourceSnapshot.")
    snapshot_document = snapshot.model_dump(mode="json", warnings=False)
    snapshot_document.pop("source_session_id")
    snapshot_document.pop("causal_budget_id")
    return sha256(
        canonical_durable_json_bytes(
            {
                "record_type": "cayu.fork-source-state",
                "schema_version": 1,
                **snapshot_document,
            },
            "fork_source.state",
        )
    ).hexdigest()


class ForkExecutionProfileDecisionRecord(BaseModel):
    """Bounded accepted policy evidence stored with a profiled fork."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: ExecutionProfileDecisionKind
    policy_identity: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS)
    policy_reason: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS)
    authority_decision: ExecutionProfileAuthorityDecision
    actor: ResolutionActor
    reason: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS)
    idempotency_identity: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS)
    adoption_request_fingerprint: str
    event_id: str

    @field_validator(
        "policy_identity",
        "policy_reason",
        "reason",
        "idempotency_identity",
        "adoption_request_fingerprint",
        "event_id",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("actor", mode="before")
    @classmethod
    def copy_actor(cls, value: object) -> ResolutionActor:
        if isinstance(value, ResolutionActor):
            copied = copy_resolution_actor(value)
            if copied is None:
                raise ValueError("actor is required for a fork profile decision.")
            return copied
        return ResolutionActor.model_validate(value)

    @model_validator(mode="after")
    def validate_accepted_decision(self) -> ForkExecutionProfileDecisionRecord:
        if self.kind not in {
            ExecutionProfileDecisionKind.EXACT_REUSE,
            ExecutionProfileDecisionKind.COMPATIBLE_REUSE,
            ExecutionProfileDecisionKind.ADOPTED,
        }:
            raise ValueError("A profiled fork can store only an accepted profile decision.")
        if (
            self.kind is ExecutionProfileDecisionKind.EXACT_REUSE
            and self.authority_decision is not ExecutionProfileAuthorityDecision.NOT_REQUIRED
        ):
            raise ValueError("Exact fork profile reuse cannot carry an authority decision.")
        if (
            self.kind
            in {
                ExecutionProfileDecisionKind.COMPATIBLE_REUSE,
                ExecutionProfileDecisionKind.ADOPTED,
            }
            and self.authority_decision is ExecutionProfileAuthorityDecision.DENIED
        ):
            raise ValueError("A denied authority decision cannot admit a fork profile.")
        return self


class SessionForkEnvironmentAllocationOwner(BaseModel):
    """One canonical environment-allocation owner inherited by a fork."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    environment_name: str
    owner_session_id: str

    @field_validator("environment_name", "owner_session_id")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class SessionForkProfileRelationship(BaseModel):
    """Immutable, content-bound profile authority for one session fork."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.session-fork-execution-profile"] = FORK_EXECUTION_PROFILE_RECORD_TYPE
    schema_version: Literal[1, 2]
    request_sha256: str
    source_state_sha256: str | None = None
    source_session_id: str
    child_session_id: str
    child_agent_name: str
    child_provider_name: str
    child_model: str
    child_environment_name: str | None = None
    source_status: SessionStatus
    source_run_epoch: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    source_profile_source: ForkExecutionProfileSource
    source_profile: ExecutionProfileIdentity
    source_active_interaction_id: str | None = None
    source_active_run_epoch: StrictInt | None = Field(
        default=None, ge=1, le=MAX_DURABLE_JSON_INTEGER
    )
    transcript_cursor: StrictInt | None = Field(
        default=None,
        ge=0,
        le=MAX_DURABLE_JSON_INTEGER,
    )
    copy_checkpoint: StrictBool
    system_prompt_policy: ForkSystemPromptPolicy
    selection: ForkExecutionProfileSelection
    selected_profile: ExecutionProfileIdentity
    model_failover_candidate_index: StrictInt | None = Field(
        default=None, ge=0, le=7, exclude_if=lambda value: value is None
    )
    source_environment_allocation_owners: tuple[SessionForkEnvironmentAllocationOwner, ...]
    initial_invocation_request_sha256: str | None = None
    initial_dispatch_id: str | None = None
    initial_invocation_profile: ExecutionProfileIdentity | None = None
    decision: ForkExecutionProfileDecisionRecord | None = None
    fork_event_id: str

    @field_validator(
        "request_sha256",
        "source_session_id",
        "child_session_id",
        "child_agent_name",
        "child_provider_name",
        "child_model",
        "fork_event_id",
    )
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator(
        "child_environment_name",
        "source_active_interaction_id",
        "initial_dispatch_id",
    )
    @classmethod
    def validate_optional_identity_text(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("source_environment_allocation_owners", mode="before")
    @classmethod
    def copy_environment_allocation_owners(
        cls,
        value: object,
    ) -> tuple[SessionForkEnvironmentAllocationOwner, ...]:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError("source_environment_allocation_owners must be a sequence.")
        owners = tuple(
            SessionForkEnvironmentAllocationOwner.model_validate(
                item.model_dump(mode="json")
                if isinstance(item, SessionForkEnvironmentAllocationOwner)
                else item
            )
            for item in value
        )
        environment_names = tuple(owner.environment_name for owner in owners)
        if environment_names != tuple(sorted(set(environment_names))):
            raise ValueError(
                "source_environment_allocation_owners must be sorted by unique "
                "environment_name values."
            )
        return owners

    @field_validator(
        "source_profile",
        "selected_profile",
        "initial_invocation_profile",
        mode="before",
    )
    @classmethod
    def copy_profile(cls, value: object) -> ExecutionProfileIdentity | None:
        if value is None:
            return None
        if isinstance(value, ExecutionProfileIdentity):
            value = value.model_dump(mode="json")
        return ExecutionProfileIdentity.model_validate(value)

    @model_validator(mode="after")
    def validate_relationship(self) -> SessionForkProfileRelationship:
        selected_binding = self.selected_profile.model_failover
        if (selected_binding is None) != (self.model_failover_candidate_index is None):
            raise ValueError("Fork routing origin conflicts with its selected profile.")
        if selected_binding is not None and (
            self.model_failover_candidate_index is None
            or self.model_failover_candidate_index >= len(selected_binding.plan.candidates)
            or (
                self.selection is ForkExecutionProfileSelection.CURRENT_CHILD
                and self.model_failover_candidate_index != 0
            )
        ):
            raise ValueError("Fork routing origin is outside its configured selection.")
        if len(self.request_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.request_sha256
        ):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest.")
        exact_source = self.schema_version == FORK_EXECUTION_PROFILE_EXACT_SOURCE_SCHEMA_VERSION
        if exact_source != (self.source_state_sha256 is not None):
            raise ValueError(
                "Fork profile schema version conflicts with its exact-source commitment."
            )
        if not self.copy_checkpoint and self.source_environment_allocation_owners:
            raise ValueError("Fork allocation-owner evidence requires copied checkpoint state.")
        if self.child_session_id in {
            owner.owner_session_id for owner in self.source_environment_allocation_owners
        }:
            raise ValueError("A fork cannot inherit environment allocation state from itself.")
        if self.source_state_sha256 is not None and (
            len(self.source_state_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_state_sha256)
        ):
            raise ValueError("source_state_sha256 must be a lowercase SHA-256 digest.")
        initial_authority_presence = (
            self.initial_invocation_request_sha256 is not None,
            self.initial_dispatch_id is not None,
            self.initial_invocation_profile is not None,
        )
        if any(initial_authority_presence) and not all(initial_authority_presence):
            raise ValueError(
                "Fork initial invocation authority requires request, dispatch, and profile."
            )
        if self.initial_invocation_request_sha256 is not None and not exact_source:
            raise ValueError("Fork initial invocation authority requires an exact source.")
        if self.initial_invocation_request_sha256 is not None and (
            len(self.initial_invocation_request_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.initial_invocation_request_sha256
            )
        ):
            raise ValueError(
                "initial_invocation_request_sha256 must be a lowercase SHA-256 digest."
            )
        active = self.source_profile_source is ForkExecutionProfileSource.ACTIVE_INVOCATION
        if active != (self.source_active_interaction_id is not None):
            raise ValueError("Active parent profile authority requires an interaction identity.")
        if active != (self.source_active_run_epoch is not None):
            raise ValueError("Active parent profile authority requires a run epoch.")
        if self.selection is ForkExecutionProfileSelection.INHERIT_PARENT:
            selected_changes = inherited_execution_profile_component_changes(
                self.source_profile,
                self.selected_profile,
            )
            selected_profile_changed_outside_ceiling = (
                self.selected_profile != self.source_profile
                and any(
                    component != ExecutionProfileComponentClass.TOOL_VIEW_GRANTS
                    for component in selected_changes
                )
            )
            if self.decision is not None or selected_profile_changed_outside_ceiling:
                raise ValueError("Inherited fork profile authority is inconsistent.")
            if self.initial_invocation_profile is not None:
                allowed_initial_changes = {
                    ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
                    ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
                    ExecutionProfileComponentClass.FINALIZATION,
                    ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
                }
                if any(
                    component not in allowed_initial_changes
                    for component in inherited_execution_profile_component_changes(
                        self.source_profile,
                        self.initial_invocation_profile,
                    )
                ):
                    raise ValueError(
                        "Inherited fork initial invocation changes unrelated authority."
                    )
        else:
            if (
                self.decision is None
                or self.decision.adoption_request_fingerprint != self.request_sha256
            ):
                raise ValueError(
                    "Current-child fork profile authority requires its exact request decision."
                )
            if (
                self.initial_invocation_profile is not None
                and self.initial_invocation_profile != self.selected_profile
            ):
                raise ValueError(
                    "Current-child fork initial invocation conflicts with its selection."
                )
            changed = changed_execution_profile_components(
                self.source_profile,
                self.selected_profile,
            )
            if self.decision.kind is ExecutionProfileDecisionKind.EXACT_REUSE:
                if changed:
                    raise ValueError("Exact current-child reuse cannot change profile components.")
            elif not changed and not (
                self.decision.kind is ExecutionProfileDecisionKind.ADOPTED
                and self.decision.authority_decision is ExecutionProfileAuthorityDecision.AUTHORIZED
            ):
                raise ValueError(
                    "A non-exact current-child decision requires changed profile components."
                )
            if self.decision.kind is ExecutionProfileDecisionKind.COMPATIBLE_REUSE and any(
                component
                in {
                    ExecutionProfileComponentClass.DIRECT_TOOLS,
                    ExecutionProfileComponentClass.PROVIDER_TARGET,
                }
                for component in changed
            ):
                raise ValueError(
                    "Compatible current-child reuse cannot change provider or tool authority."
                )
            if (
                self.decision.kind is ExecutionProfileDecisionKind.ADOPTED
                and ExecutionProfileComponentClass.DIRECT_TOOLS in changed
                and self.decision.authority_decision
                is not ExecutionProfileAuthorityDecision.AUTHORIZED
            ):
                raise ValueError("Current-child direct-tool adoption requires explicit authority.")
        return self


class ProfiledSessionForkResult(BaseModel):
    """Store-atomic child session and its ordered durable evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session: Session
    events: tuple[Event, ...]

    @field_validator("session", mode="before")
    @classmethod
    def copy_result_session(cls, value: Session) -> Session:
        return copy_session(value)

    @field_validator("events", mode="before")
    @classmethod
    def copy_result_events(cls, value: object) -> tuple[Event, ...]:
        if type(value) not in {list, tuple}:
            raise TypeError("Profiled fork events must be a list or tuple.")
        events = cast("list[Event] | tuple[Event, ...]", value)
        if not 2 <= len(events) <= 3:
            raise ValueError("Profiled fork evidence must contain two or three events.")
        return tuple(copy_event(event) for event in events)


def copy_profiled_session_fork_result(
    result: ProfiledSessionForkResult,
) -> ProfiledSessionForkResult:
    """Detach one extension-owned profiled-fork acknowledgement fail closed."""

    if type(result) is not ProfiledSessionForkResult:
        raise TypeError("Profiled fork publication returned an invalid result type.")
    state = object.__getattribute__(result, "__dict__")
    if type(state) is not dict or set(state) != {"session", "events"}:
        raise TypeError("Profiled fork publication returned malformed result state.")
    session = state["session"]
    events = state["events"]
    if type(session) is not Session or type(events) is not tuple or not 2 <= len(events) <= 3:
        raise TypeError("Profiled fork publication returned malformed result fields.")
    return ProfiledSessionForkResult(
        session=copy_session(session),
        events=tuple(copy_event(event) for event in events),
    )


def session_fork_profile_relationship(
    session: Session,
) -> SessionForkProfileRelationship | None:
    """Load and authenticate the immutable profile relationship of one child."""

    if type(session) is not Session:
        raise TypeError("session must be a Session.")
    raw = session.metadata.get(FORK_EXECUTION_PROFILE_METADATA_KEY)
    if raw is None:
        return None
    try:
        relationship = SessionForkProfileRelationship.model_validate(
            copy_durable_json_value(raw, "fork_execution_profile")
        )
    except Exception as exc:
        raise ValueError("Session fork execution-profile metadata is malformed.") from exc
    if relationship.child_session_id != session.id or (
        session.parent_session_id is not None
        and relationship.source_session_id != session.parent_session_id
    ):
        raise ValueError("Session fork execution-profile metadata conflicts with lineage.")
    baseline = execution_profile_baseline_from_session_metadata(session.metadata)
    if baseline != relationship.selected_profile:
        raise ValueError("Session fork execution-profile metadata conflicts with its baseline.")
    return relationship


def session_prompt_anatomy_transition(
    session: Session,
) -> PromptAnatomyTransitionReceipt | None:
    """Return verified durable prompt-succession evidence for one descendant."""

    if type(session) is not Session:
        raise TypeError("session must be a Session.")
    raw = session.metadata.get(PROMPT_ANATOMY_TRANSITION_METADATA_KEY)
    if raw is None:
        return None
    try:
        receipt = PromptAnatomyTransitionReceipt.model_validate(raw)
    except Exception as exc:
        raise ValueError("Session prompt-anatomy transition metadata is malformed.") from exc
    fork_relationship = session_fork_profile_relationship(session)
    creation_agent_name = (
        session.agent_name if fork_relationship is None else fork_relationship.child_agent_name
    )
    creation_environment_name = (
        session.environment_name
        if fork_relationship is None
        else fork_relationship.child_environment_name
    )
    creation_provider_name = (
        session.provider_name
        if fork_relationship is None
        else fork_relationship.child_provider_name
    )
    creation_model = session.model if fork_relationship is None else fork_relationship.child_model
    creation_source_session_id = (
        session.parent_session_id
        if fork_relationship is None
        else fork_relationship.source_session_id
    )
    if (
        receipt.descendant_session_id != session.id
        or receipt.source_session_id != creation_source_session_id
        or receipt.child_agent_name != creation_agent_name
        or receipt.child_environment_name != creation_environment_name
        or receipt.provider_name != creation_provider_name
        or receipt.model != creation_model
        or receipt.source_provider_name != receipt.provider_name
        or receipt.model_target_changed != (receipt.source_model != receipt.model)
        or receipt.portability_preflight
        != (
            "provider_portable_transcript_preflight"
            if receipt.model_target_changed
            else "context_messages_validated"
        )
    ):
        raise ValueError("Session prompt-anatomy transition conflicts with session identity.")
    return receipt


class PromptAnatomyTransitionReceipt(BaseModel):
    """Durable prompt-succession evidence that never contains prompt text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_type: Literal["cayu.prompt-anatomy-transition"] = PROMPT_ANATOMY_TRANSITION_RECORD_TYPE
    schema_version: Literal[1] = PROMPT_ANATOMY_TRANSITION_SCHEMA_VERSION
    transition_id: str
    request_sha256: str
    source_session_id: str
    descendant_session_id: str
    source_status: SessionStatus
    source_transcript_cursor: int
    source_prompt_sha256: str
    child_prompt_sha256: str
    source_system_prompt_retained: StrictBool
    child_agent_name: str
    child_environment_name: str
    copy_checkpoint: StrictBool
    source_provider_name: str
    source_model: str
    provider_name: str
    model: str
    model_target_changed: StrictBool
    portability_preflight: Literal[
        "context_messages_validated",
        "provider_portable_transcript_preflight",
    ]
    portability_verified: Literal[True]
    inherited_taint_labels: tuple[str, ...]
    policy: Literal[ForkSystemPromptPolicy.CURRENT_AGENT] = ForkSystemPromptPolicy.CURRENT_AGENT

    @classmethod
    def create(
        cls,
        *,
        request_sha256: str,
        source_session_id: str,
        descendant_session_id: str,
        source_status: SessionStatus,
        source_transcript_cursor: int,
        source_prompt_sha256: str,
        child_prompt_sha256: str,
        child_agent_name: str,
        child_environment_name: str,
        copy_checkpoint: bool,
        source_provider_name: str,
        source_model: str,
        provider_name: str,
        model: str,
        model_target_changed: bool,
        portability_preflight: Literal[
            "context_messages_validated",
            "provider_portable_transcript_preflight",
        ],
        inherited_taint_labels: tuple[str, ...],
    ) -> PromptAnatomyTransitionReceipt:
        """Build one canonical receipt and bind its digest to the same material."""

        payload = {
            "record_type": PROMPT_ANATOMY_TRANSITION_RECORD_TYPE,
            "schema_version": PROMPT_ANATOMY_TRANSITION_SCHEMA_VERSION,
            "request_sha256": request_sha256,
            "source_session_id": source_session_id,
            "descendant_session_id": descendant_session_id,
            "source_status": source_status.value,
            "source_transcript_cursor": source_transcript_cursor,
            "source_prompt_sha256": source_prompt_sha256,
            "child_prompt_sha256": child_prompt_sha256,
            "source_system_prompt_retained": False,
            "child_agent_name": child_agent_name,
            "child_environment_name": child_environment_name,
            "copy_checkpoint": copy_checkpoint,
            "source_provider_name": source_provider_name,
            "source_model": source_model,
            "provider_name": provider_name,
            "model": model,
            "model_target_changed": model_target_changed,
            "portability_preflight": portability_preflight,
            "portability_verified": True,
            "inherited_taint_labels": list(inherited_taint_labels),
            "policy": ForkSystemPromptPolicy.CURRENT_AGENT.value,
        }
        transition_id = sha256(
            canonical_durable_json_bytes(payload, "prompt_anatomy_transition")
        ).hexdigest()
        return cls.model_validate({"transition_id": transition_id, **payload})

    @field_validator(
        "transition_id",
        "request_sha256",
        "source_session_id",
        "descendant_session_id",
        "child_agent_name",
        "source_provider_name",
        "source_model",
        "provider_name",
        "model",
    )
    @classmethod
    def validate_nonblank_receipt_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("child_environment_name")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        return require_clean_nonblank(value, "child_environment_name")

    @field_validator(
        "transition_id",
        "request_sha256",
        "source_prompt_sha256",
        "child_prompt_sha256",
    )
    @classmethod
    def validate_prompt_digest(cls, value: str, info) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"{info.field_name} must be a lowercase SHA-256 digest.")
        return value

    @field_validator("inherited_taint_labels")
    @classmethod
    def validate_inherited_taint_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        labels = tuple(require_clean_nonblank(label, "inherited_taint_labels") for label in value)
        if labels != tuple(sorted(set(labels))):
            raise ValueError("inherited_taint_labels must be sorted and unique.")
        return labels

    @model_validator(mode="after")
    def validate_exact_transition_authority(self) -> PromptAnatomyTransitionReceipt:
        if self.source_system_prompt_retained or self.copy_checkpoint:
            raise ValueError(
                "Prompt-anatomy succession must replace the source prompt without checkpoint state."
            )
        if self.source_status not in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.INTERRUPTED,
        }:
            raise ValueError("Prompt-anatomy succession source status is not forkable.")
        if self.source_provider_name != self.provider_name:
            raise ValueError("Prompt-anatomy succession cannot change providers.")
        if self.model_target_changed != (self.source_model != self.model):
            raise ValueError("Prompt-anatomy succession model-target evidence is inconsistent.")
        expected_preflight = (
            "provider_portable_transcript_preflight"
            if self.model_target_changed
            else "context_messages_validated"
        )
        if self.portability_preflight != expected_preflight:
            raise ValueError("Prompt-anatomy succession portability evidence is inconsistent.")
        payload = self.model_dump(mode="json", exclude={"transition_id"})
        expected_transition_id = sha256(
            canonical_durable_json_bytes(payload, "prompt_anatomy_transition")
        ).hexdigest()
        if self.transition_id != expected_transition_id:
            raise ValueError("Prompt-anatomy transition_id does not bind the exact receipt.")
        return self


@dataclass(frozen=True, slots=True)
class ForkSystemPromptReplacement:
    """Store-boundary instruction to replace every inherited system message."""

    message: Message | None


def apply_fork_system_prompt_replacement(
    messages: list[Message],
    interaction_ids: list[str | None],
    replacement: ForkSystemPromptReplacement | None,
) -> tuple[list[Message], list[str | None]]:
    """Install one authoritative system message while preserving non-system history."""

    if len(messages) != len(interaction_ids):
        raise RuntimeError("Fork transcript attribution does not match its messages.")
    if replacement is None:
        return messages, interaction_ids
    replacement_message = replacement.message
    if replacement_message is not None:
        if (
            type(replacement_message) is not Message
            or replacement_message.role != MessageRole.SYSTEM
        ):
            raise TypeError("Fork prompt replacement must be a system Message.")
        replacement_message = detach_message(replacement_message)
    retained_messages: list[Message] = []
    retained_interaction_ids: list[str | None] = []
    for message, interaction_id in zip(messages, interaction_ids, strict=True):
        if message.role == MessageRole.SYSTEM:
            continue
        retained_messages.append(message)
        retained_interaction_ids.append(interaction_id)
    if replacement_message is not None:
        retained_messages.insert(0, replacement_message)
        retained_interaction_ids.insert(0, None)
    messages.clear()
    interaction_ids.clear()
    return retained_messages, retained_interaction_ids


class SessionForkSourceNotFound(KeyError):
    """A fork's source disappeared before the atomic child-creation boundary."""


class SessionForkActiveModelStageConflict(ValueError):
    """A fork was rejected because its source still owns an active model stage."""


def fork_session_invocation(source_session: Session) -> SessionInvocation:
    """Build the only valid invocation provenance for a direct session fork."""

    if type(source_session) is not Session:
        raise TypeError("Fork invocation provenance requires a Session.")
    return inherited_session_invocation(
        source_session.invocation,
        source=SessionExecutionSource.FORK,
    )


def _prepare_session_fork_request(
    *,
    source_session_id: str,
    fork: Session,
    source_statuses: set[SessionStatus],
    transcript_cursor: int | None,
) -> tuple[str, Session, set[SessionStatus], int | None]:
    source_session_id = require_clean_nonblank(source_session_id, "source_session_id")
    fork = copy_session(fork)
    allowed_statuses = session_record_rules._validate_status_set(source_statuses, "source_statuses")
    if fork.parent_session_id != source_session_id:
        raise ValueError("Fork parent_session_id must match source_session_id.")
    if transcript_cursor is not None and transcript_cursor < 0:
        raise ValueError("transcript_cursor must be greater than or equal to 0.")
    return source_session_id, fork, allowed_statuses, transcript_cursor


def _validate_session_fork_source(
    *,
    source_session: Session | None,
    source_session_id: str,
    fork: Session,
    allowed_statuses: set[SessionStatus],
    expected_source_run_epoch: int,
    profile_relationship: SessionForkProfileRelationship | None = None,
) -> Session:
    from cayu._resource_access_errors import ResourceAccessDenied
    from cayu.resource_access import current_binding, current_creation_bounds

    access_bounds = current_creation_bounds()
    if access_bounds is not None:
        source_session = access_bounds.require_read(source_session)
        if not access_bounds.matches(fork.labels, "create") or not access_bounds.matches(
            fork.labels
        ):
            raise ResourceAccessDenied()
        if source_session.invocation.resource_access != current_binding():
            raise ResourceAccessDenied()
    if source_session is None:
        raise SessionForkSourceNotFound("Fork source session was not found.")
    if source_session.status not in allowed_statuses:
        raise ValueError(f"Source session status is not forkable: {source_session.status}")
    if source_session.run_epoch != expected_source_run_epoch:
        raise ValueError(
            "Source session changed while the fork was being prepared: "
            f"run_epoch {source_session.run_epoch} != {expected_source_run_epoch}"
        )
    if fork.status != source_session.status:
        raise ValueError(
            "Fork status must match source session status: "
            f"{fork.status} != {source_session.status}"
        )
    provider_changed = fork.provider_name != source_session.provider_name
    profiled_provider_change = (
        profile_relationship is not None
        and profile_relationship.selection is ForkExecutionProfileSelection.CURRENT_CHILD
    )
    if provider_changed and not profiled_provider_change:
        raise ValueError(
            "Fork provider_name must match source session provider_name: "
            f"{fork.provider_name} != {source_session.provider_name}"
        )
    if fork.invocation != fork_session_invocation(source_session):
        raise ValueError("Fork invocation provenance conflicts with its source session.")
    return source_session


def effective_fork_source_execution_profile(
    source_session: Session,
    source_checkpoint: Mapping[str, Any] | None,
) -> tuple[
    ForkExecutionProfileSource,
    ExecutionProfileIdentity,
    ActiveInvocationExecutionProfile | None,
]:
    """Resolve the exact parent profile authority visible at the fork boundary."""

    source_session = copy_session(source_session)
    active = active_invocation_execution_profile_from_checkpoint(source_checkpoint)
    if active is not None:
        if not active_invocation_execution_profile_matches_session_epoch(
            active,
            session_id=source_session.id,
            run_epoch=source_session.run_epoch,
        ):
            raise ValueError(
                "Source active invocation execution profile conflicts with its run epoch."
            )
        return ForkExecutionProfileSource.ACTIVE_INVOCATION, active.profile, active
    return (
        ForkExecutionProfileSource.SESSION_EXPECTED,
        execution_profile_from_session_metadata(source_session.metadata),
        None,
    )


def _copy_profiled_fork_authority(
    *,
    fork: Session,
    relationship: SessionForkProfileRelationship,
    events: list[Event],
) -> tuple[SessionForkProfileRelationship, list[Event]]:
    if type(relationship) is not SessionForkProfileRelationship:
        raise TypeError("relationship must be a SessionForkProfileRelationship.")
    copied_relationship = SessionForkProfileRelationship.model_validate(
        relationship.model_dump(mode="json")
    )
    _, copied_events = session_event_rules._copy_session_event_batch(fork.id, events)
    if not copied_events:
        raise ValueError("A profiled fork requires durable fork evidence.")
    return copied_relationship, copied_events


def _validate_profiled_fork_authority(
    *,
    source_session: Session,
    source_checkpoint: Mapping[str, Any] | None,
    fork: Session,
    relationship: SessionForkProfileRelationship,
    events: Sequence[Event],
    transcript_cursor: int | None,
    checkpoint_transform: CheckpointTransform | None,
    system_prompt_replacement: ForkSystemPromptReplacement | None,
    transcript_validator: ForkTranscriptValidator | None,
) -> None:
    source_ceiling = tool_capability_ceiling_from_session_metadata(source_session.metadata)
    fork_ceiling = tool_capability_ceiling_from_session_metadata(fork.metadata)
    if not frozenset(fork_ceiling.tool_names) <= frozenset(source_ceiling.tool_names):
        raise ValueError("A fork cannot widen its source tool capability ceiling.")
    source_kind, source_profile, active = effective_fork_source_execution_profile(
        source_session,
        source_checkpoint,
    )
    if relationship.selected_profile.model_failover is not None:
        from cayu.sessions._execution_profile_checkpoint import model_failover_progress_for_session

        source_selection = model_failover_progress_for_session(
            session=source_session,
            execution_profile=source_profile,
            checkpoint=(
                None
                if source_checkpoint is None
                else copy_durable_json_object(source_checkpoint, "checkpoint")
            ),
        )
        expected_index = (
            source_selection.candidate_index
            if relationship.selection is ForkExecutionProfileSelection.INHERIT_PARENT
            and source_selection is not None
            else 0
        )
        if relationship.model_failover_candidate_index != expected_index:
            raise ValueError("Fork routing origin conflicts with its exact source selection.")
    if source_profile.component(
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS
    ) != direct_tool_capability_ceiling_component(source_ceiling.tool_names):
        raise ValueError("Fork source profile conflicts with its tool capability ceiling.")
    if relationship.selected_profile.component(
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS
    ) != direct_tool_capability_ceiling_component(fork_ceiling.tool_names):
        raise ValueError("Fork child profile conflicts with its tool capability ceiling.")
    if (
        relationship.source_session_id != source_session.id
        or relationship.child_session_id != fork.id
        or relationship.child_agent_name != fork.agent_name
        or relationship.child_provider_name != fork.provider_name
        or relationship.child_model != fork.model
        or relationship.child_environment_name != fork.environment_name
        or relationship.source_status != source_session.status
        or relationship.source_run_epoch != source_session.run_epoch
        or relationship.source_profile_source != source_kind
        or relationship.source_profile != source_profile
        or relationship.source_active_interaction_id
        != (None if active is None else active.interaction_id)
        or relationship.source_active_run_epoch != (None if active is None else active.run_epoch)
        or relationship.transcript_cursor != transcript_cursor
    ):
        raise ValueError("Fork execution-profile relationship conflicts with its source.")
    current_agent_prompt = relationship.system_prompt_policy is ForkSystemPromptPolicy.CURRENT_AGENT
    if current_agent_prompt != (system_prompt_replacement is not None):
        raise ValueError("Fork prompt operation conflicts with its profile relationship.")
    if checkpoint_transform is None:
        raise ValueError("A profiled fork requires live checkpoint validation.")
    if transcript_validator is None:
        raise ValueError("A profiled fork requires atomic transcript validation.")
    validate_profiled_fork_evidence(
        fork=fork,
        relationship=relationship,
        events=events,
    )


def _profiled_fork_authority_validation_error(error: Exception) -> ValueError:
    """Detach a failed transactional profile check from private checkpoint state."""

    if error.__traceback__ is not None:
        traceback_module.clear_frames(error.__traceback__)
    return ValueError("Fork source no longer has the expected durable execution-profile identity.")


def _prepare_profiled_fork_checkpoint_result(
    *,
    supports_model_failover: bool,
    fork: Session,
    transcript_cursor: int,
    relationship: SessionForkProfileRelationship,
    source_checkpoint_present: bool,
    copied_checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate copied state and initialize the child's own routing origin."""

    if not relationship.copy_checkpoint:
        if copied_checkpoint is not None:
            raise ValueError("Fork copied checkpoint state contrary to its profile relationship.")
    elif source_checkpoint_present and copied_checkpoint is None:
        raise ValueError("Fork discarded checkpoint state contrary to its profile relationship.")
    checkpoint = (
        None
        if copied_checkpoint is None
        else copy_durable_json_object(copied_checkpoint, "checkpoint")
    )
    binding = relationship.selected_profile.model_failover
    if binding is None:
        return checkpoint
    if supports_model_failover is not True:
        raise NotImplementedError("Store does not attest atomic model failover forks.")
    if checkpoint is not None and MODEL_FAILOVER_CHECKPOINT_KEY in checkpoint:
        raise ValueError("Fork checkpoint cannot copy source routing authority.")
    index = relationship.model_failover_candidate_index
    if index is None or fork.run_epoch != 0:
        raise ValueError("Fork routing origin lost its creation authority.")
    origin = ModelFailoverSelection(
        session_id=fork.id,
        session_instance_id=fork.instance_id,
        execution_profile_fingerprint=relationship.selected_profile.fingerprint,
        plan=binding.plan,
        candidate_index=index,
        origin_id=relationship.request_sha256,
        source_run_epoch=0,
        source_transcript_cursor=transcript_cursor,
        projection_cursor=0,
    )
    updated = {} if checkpoint is None else checkpoint
    updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
    updated[MODEL_FAILOVER_CHECKPOINT_KEY] = origin.payload()
    return updated


def validate_profiled_fork_evidence(
    *,
    fork: Session,
    relationship: SessionForkProfileRelationship,
    events: Sequence[Event],
) -> None:
    """Validate immutable child baseline, relationship, and ordered evidence."""

    if type(fork) is not Session:
        raise TypeError("fork must be a Session.")
    if type(relationship) is not SessionForkProfileRelationship:
        raise TypeError("relationship must be a SessionForkProfileRelationship.")
    fork_ceiling = tool_capability_ceiling_from_session_metadata(fork.metadata)
    if relationship.selected_profile.component(
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS
    ) != direct_tool_capability_ceiling_component(fork_ceiling.tool_names):
        raise ValueError("Fork child profile conflicts with its tool capability ceiling.")
    stored_relationship = session_fork_profile_relationship(fork)
    if (
        stored_relationship != relationship
        or relationship.child_session_id != fork.id
        or (
            fork.parent_session_id is not None
            and relationship.source_session_id != fork.parent_session_id
        )
    ):
        raise ValueError("Fork execution-profile relationship conflicts with child metadata.")
    if relationship.selection is ForkExecutionProfileSelection.CURRENT_CHILD:
        if len(events) != 3 or relationship.decision is None:
            raise ValueError(
                "Current-child fork evidence requires decision, fork, and grant-reset events."
            )
        decision_event, fork_event, grant_reset_event = events
        if (
            decision_event.type != EventType.SESSION_EXECUTION_PROFILE_DECIDED
            or decision_event.id != relationship.decision.event_id
            or decision_event.payload.get("decision") != relationship.decision.kind.value
            or decision_event.payload.get("expected_profile")
            != relationship.source_profile.model_dump(mode="json")
            or decision_event.payload.get("candidate_profile")
            != relationship.selected_profile.model_dump(mode="json")
            or decision_event.payload.get("changed_component_classes")
            != [
                component.value
                for component in changed_execution_profile_components(
                    relationship.source_profile,
                    relationship.selected_profile,
                )
            ]
            or decision_event.payload.get("policy_identity")
            != relationship.decision.policy_identity
            or decision_event.payload.get("policy_reason") != relationship.decision.policy_reason
            or decision_event.payload.get("authority_decision")
            != relationship.decision.authority_decision.value
            or decision_event.payload.get("idempotency_identity")
            != relationship.decision.idempotency_identity
            or decision_event.payload.get("actor")
            != resolution_actor_payload(relationship.decision.actor)
            or decision_event.payload.get("reason") != relationship.decision.reason
            or decision_event.payload.get("adoption_request_fingerprint")
            != relationship.decision.adoption_request_fingerprint
        ):
            raise ValueError("Fork profile decision event conflicts with its relationship.")
    else:
        if len(events) != 2:
            raise ValueError("Inherited fork evidence requires fork and grant-reset events.")
        fork_event, grant_reset_event = events
    if (
        fork_event.type != EventType.SESSION_FORKED
        or fork_event.id != relationship.fork_event_id
        or fork_event.timestamp != fork.created_at
        or any(event.session_id != fork.id for event in events)
    ):
        raise ValueError("Fork event evidence conflicts with its relationship.")
    if (
        grant_reset_event.type is not EventType.TARGETED_TOOL_GRANT_FORK_RESET
        or grant_reset_event.timestamp != fork.created_at
        or grant_reset_event.payload.get("schema_version") != 1
        or grant_reset_event.payload.get("source_session_id") != relationship.source_session_id
        or grant_reset_event.payload.get("source_interaction_id")
        != relationship.source_active_interaction_id
        or grant_reset_event.payload.get("inherited_grant_count") != 0
        or grant_reset_event.payload.get("inherited_reference_count") != 0
    ):
        raise ValueError("Fork targeted-grant reset evidence is inconsistent.")
    payload = fork_event.payload
    selected_index = relationship.model_failover_candidate_index
    if payload.get("model_failover_candidate_index") != selected_index or (
        selected_index is not None
        and type(payload.get("model_failover_candidate_index")) is not int
    ):
        raise ValueError("Fork event conflicts with its initial model selection.")
    exact_source_snapshot_event_fields = {
        "source_instance_fingerprint",
        "source_run_epoch",
        "source_transcript_cursor",
        "source_transcript_sha256",
        "source_checkpoint_sha256",
        "source_execution_profile_fingerprint",
    }
    exact_source_event_fields = exact_source_snapshot_event_fields | {
        "source_environment_allocation_owners"
    }
    source_environment_allocation_owners = [
        owner.model_dump(mode="json") for owner in relationship.source_environment_allocation_owners
    ]
    raw_source_snapshot = fork.metadata.get(FORK_SOURCE_SNAPSHOT_METADATA_KEY)
    if raw_source_snapshot is None:
        if (
            relationship.source_state_sha256 is not None
            or exact_source_snapshot_event_fields.intersection(payload)
        ):
            raise ValueError("Fork event exact-source evidence conflicts with child metadata.")
    else:
        try:
            source_snapshot = ForkSourceSnapshot.model_validate(
                copy_durable_json_value(raw_source_snapshot, "fork_source_snapshot")
            )
        except (TypeError, ValueError):
            raise ValueError("Fork source snapshot metadata is malformed.") from None
        if (
            not exact_source_event_fields.issubset(payload)
            or relationship.source_state_sha256 is None
            or fork_source_state_sha256(source_snapshot) != relationship.source_state_sha256
            or source_snapshot.source_session_id != relationship.source_session_id
            or source_snapshot.status is not relationship.source_status
            or source_snapshot.run_epoch != relationship.source_run_epoch
            or source_snapshot.execution_profile_fingerprint
            != relationship.source_profile.fingerprint
            or source_snapshot.causal_budget_id != fork.causal_budget_id
            or payload.get("source_instance_fingerprint")
            != source_snapshot.source_instance_fingerprint
            or payload.get("source_run_epoch") != source_snapshot.run_epoch
            or payload.get("source_transcript_cursor") != source_snapshot.transcript_cursor
            or payload.get("source_transcript_sha256") != source_snapshot.transcript_sha256
            or payload.get("source_checkpoint_sha256") != source_snapshot.checkpoint_sha256
            or payload.get("source_execution_profile_fingerprint")
            != source_snapshot.execution_profile_fingerprint
            or payload.get("source_environment_allocation_owners")
            != source_environment_allocation_owners
        ):
            raise ValueError("Fork event exact-source evidence conflicts with child metadata.")
    if (
        payload.get("source_session_id") != relationship.source_session_id
        or payload.get("source_status") != relationship.source_status.value
        or payload.get("parent_session_id") != relationship.source_session_id
        or payload.get("causal_budget_id") != fork.causal_budget_id
        or payload.get("agent_name") != relationship.child_agent_name
        or payload.get("provider_name") != relationship.child_provider_name
        or payload.get("model") != relationship.child_model
        or payload.get("environment_name") != relationship.child_environment_name
        or payload.get("transcript_cursor") != relationship.transcript_cursor
        or payload.get("copy_checkpoint") is not relationship.copy_checkpoint
        or payload.get("system_prompt_policy") != relationship.system_prompt_policy.value
        or payload.get("execution_profile_selection") != relationship.selection.value
        or payload.get("selected_profile_fingerprint") != relationship.selected_profile.fingerprint
        or payload.get("source_profile_fingerprint") != relationship.source_profile.fingerprint
        or payload.get("source_environment_allocation_owners")
        != source_environment_allocation_owners
        or payload.get("fork_request_sha256") != relationship.request_sha256
        or payload.get("initial_invocation_request_sha256")
        != relationship.initial_invocation_request_sha256
        or payload.get("initial_dispatch_id") != relationship.initial_dispatch_id
        or payload.get("initial_invocation_profile_fingerprint")
        != (
            None
            if relationship.initial_invocation_profile is None
            else relationship.initial_invocation_profile.fingerprint
        )
    ):
        raise ValueError("Fork event payload conflicts with its profile relationship.")
