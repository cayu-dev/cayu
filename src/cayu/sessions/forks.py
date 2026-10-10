"""Session-fork evidence, execution-profile contracts and prompt rules."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from hashlib import sha256
from typing import Literal, cast

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
    copy_durable_json_value,
)
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.approvals.actors import ResolutionActor, copy_resolution_actor
from cayu.events import Event, copy_event
from cayu.execution_profiles import (
    EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS,
    EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileComponentClass,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    changed_execution_profile_components,
    inherited_execution_profile_component_changes,
)
from cayu.sessions._execution_profile_checkpoint import (
    execution_profile_baseline_from_session_metadata,
)
from cayu.sessions.records import Session, SessionStatus, copy_session

FORK_EXECUTION_PROFILE_METADATA_KEY = "cayu:fork_execution_profile"
FORK_EXECUTION_PROFILE_RECORD_TYPE = "cayu.session-fork-execution-profile"
FORK_EXECUTION_PROFILE_ORDINARY_SCHEMA_VERSION = 1
FORK_EXECUTION_PROFILE_EXACT_SOURCE_SCHEMA_VERSION = 2


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
