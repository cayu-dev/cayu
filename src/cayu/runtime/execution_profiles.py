"""Durable, redacted execution-profile identities for session admission."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

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
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.core.events import Event, copy_event, event_with_runtime_payload_authority
from cayu.runtime.approvals import (
    ResolutionActor,
    copy_resolution_actor,
    resolution_actor_payload,
)
from cayu.runtime.checkpoints import ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY
from cayu.runtime.tool_catalogue import (
    TOOL_CATALOGUE_MAX_TOOLS,
    validate_canonical_tool_id,
    validate_tool_catalogue_revision,
    validate_tool_descriptor_version,
)

EXECUTION_PROFILE_SCHEMA_VERSION = 4
_EXECUTION_PROFILE_RECORD_SCHEMA_VERSION = 1
_ACTIVE_INVOCATION_EXECUTION_PROFILE_SCHEMA_VERSION = 1
EXECUTION_PROFILE_METADATA_KEY = "cayu:execution_profile"
EXECUTION_PROFILE_FINGERPRINT_FIELD = "execution_profile_fingerprint"
_EXECUTION_PROFILE_RECORD_TYPE = "cayu.execution-profile"
_ACTIVE_INVOCATION_EXECUTION_PROFILE_RECORD_TYPE = "cayu.active-invocation-execution-profile"
EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS = 4096
EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS = 256


class ExecutionProfileComponentClass(StrEnum):
    """Stable classes of execution authority represented by a profile."""

    RUNTIME = "runtime"
    PROVIDER_TARGET = "provider_target"
    DURABLE_SYSTEM_PROJECTION = "durable_system_projection"
    DIRECT_TOOLS = "direct_tools"
    TOOL_IMPLEMENTATIONS = "tool_implementations"
    TOOL_VIEW_GRANTS = "tool_view_grants"
    EXECUTION_POLICIES = "execution_policies"
    INVOCATION_POLICIES = "invocation_policies"
    RUNTIME_HOOKS = "runtime_hooks"
    EXECUTION_ENVIRONMENT = "execution_environment"
    EFFECT_AUTHORITY = "effect_authority"
    CONTEXT_SELECTION = "context_selection"
    AUTOMATIC_RECALL = "automatic_recall"
    CONTEXT_COMPACTION = "context_compaction"
    LIVE_STATE_PROJECTION = "live_state_projection"
    PROVIDER_ADAPTER = "provider_adapter"
    PROVIDER_REQUEST_POLICY = "provider_request_policy"
    APPLICATION_BUDGET_POLICY = "application_budget_policy"
    INVOCATION_BUDGET_POLICY = "invocation_budget_policy"
    STRUCTURED_OUTPUT = "structured_output"
    FINALIZATION = "finalization"


_SCHEMA_V1_COMPONENT_CLASSES = frozenset(
    {
        ExecutionProfileComponentClass.RUNTIME,
        ExecutionProfileComponentClass.PROVIDER_TARGET,
        ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION,
        ExecutionProfileComponentClass.DIRECT_TOOLS,
    }
)
_SCHEMA_V2_COMPONENT_CLASSES = frozenset(
    {
        *_SCHEMA_V1_COMPONENT_CLASSES,
        ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
        ExecutionProfileComponentClass.EXECUTION_POLICIES,
        ExecutionProfileComponentClass.INVOCATION_POLICIES,
        ExecutionProfileComponentClass.RUNTIME_HOOKS,
        ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT,
        ExecutionProfileComponentClass.EFFECT_AUTHORITY,
    }
)
_SCHEMA_V4_COMPONENT_CLASSES = frozenset(
    {
        *_SCHEMA_V2_COMPONENT_CLASSES,
        ExecutionProfileComponentClass.CONTEXT_SELECTION,
        ExecutionProfileComponentClass.AUTOMATIC_RECALL,
        ExecutionProfileComponentClass.CONTEXT_COMPACTION,
        ExecutionProfileComponentClass.LIVE_STATE_PROJECTION,
        ExecutionProfileComponentClass.PROVIDER_ADAPTER,
        ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
        ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY,
        ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
        ExecutionProfileComponentClass.FINALIZATION,
    }
)
_SCHEMA_COMPONENT_CLASSES = {
    1: _SCHEMA_V1_COMPONENT_CLASSES,
    2: _SCHEMA_V2_COMPONENT_CLASSES,
    4: _SCHEMA_V4_COMPONENT_CLASSES,
}


class ExecutionProfileIdentityStrength(StrEnum):
    """How strongly one component is identified."""

    APPLICATION_VERSIONED = "application_versioned"
    STRUCTURAL = "structural"
    PROCESS_LOCAL = "process_local"
    UNAVAILABLE = "unavailable"


_AUTHORITY_COMPONENT_CLASSES = frozenset(
    {
        ExecutionProfileComponentClass.DIRECT_TOOLS,
        ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
        ExecutionProfileComponentClass.EXECUTION_POLICIES,
        ExecutionProfileComponentClass.INVOCATION_POLICIES,
        ExecutionProfileComponentClass.RUNTIME_HOOKS,
        ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT,
        ExecutionProfileComponentClass.EFFECT_AUTHORITY,
        ExecutionProfileComponentClass.CONTEXT_SELECTION,
        ExecutionProfileComponentClass.AUTOMATIC_RECALL,
        ExecutionProfileComponentClass.CONTEXT_COMPACTION,
        ExecutionProfileComponentClass.LIVE_STATE_PROJECTION,
        ExecutionProfileComponentClass.PROVIDER_ADAPTER,
        ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
        ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY,
        ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
        ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
        ExecutionProfileComponentClass.FINALIZATION,
    }
)


def execution_profile_changes_authority(
    component_classes: Iterable[ExecutionProfileComponentClass],
) -> bool:
    """Return whether a difference can change governed execution authority."""

    return any(component in _AUTHORITY_COMPONENT_CLASSES for component in component_classes)


def event_with_execution_profile_authority(
    event: Event,
    profile: ExecutionProfileIdentity | None,
) -> Event:
    """Bind runtime evidence to the exact admitted invocation profile."""

    if profile is None:
        return event
    if type(profile) is not ExecutionProfileIdentity:
        raise TypeError("profile must be an ExecutionProfileIdentity or None.")
    return event_with_execution_profile_fingerprint_authority(event, profile.fingerprint)


def event_with_execution_profile_fingerprint_authority(
    event: Event,
    fingerprint: str | None,
) -> Event:
    """Bind runtime evidence when only a validated profile reference remains."""

    if fingerprint is None:
        return event
    if (
        type(fingerprint) is not str
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("fingerprint must be a lowercase SHA-256 digest.")
    payload = dict(event.payload)
    payload[EXECUTION_PROFILE_FINGERPRINT_FIELD] = fingerprint
    return event_with_runtime_payload_authority(
        event.model_copy(update={"payload": payload}),
        EXECUTION_PROFILE_FINGERPRINT_FIELD,
    )


class ExecutionProfileIdentityAvailability(StrEnum):
    """Whether a component can be compared at the admission boundary."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ExecutionProfileDecisionKind(StrEnum):
    """Typed outcome of one execution-profile admission decision."""

    EXACT_REUSE = "exact_reuse"
    COMPATIBLE_REUSE = "compatible_reuse"
    ADOPTED = "adopted"
    MIGRATION_REQUIRED = "migration_required"
    REJECTED = "rejected"


class ExecutionProfilePolicyAction(StrEnum):
    """Application decision for one non-equal execution profile."""

    COMPATIBLE_REUSE = "compatible_reuse"
    ADOPT = "adopt"
    MIGRATION_REQUIRED = "migration_required"
    REJECT = "reject"


class ExecutionProfileAuthorityDecision(StrEnum):
    """Distinct authorization for a potentially authority-broadening change."""

    NOT_REQUIRED = "not_required"
    AUTHORIZED = "authorized"
    DENIED = "denied"


class ExecutionProfileAdoptionIntent(BaseModel):
    """Explicit caller intent to adopt the current application's profile."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    idempotency_key: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS)
    reason: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS)
    requested_by: ResolutionActor

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "idempotency_key")

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return require_durable_nonblank(value, "reason")

    @field_validator("requested_by")
    @classmethod
    def copy_requested_by(cls, value: ResolutionActor) -> ResolutionActor:
        copied = copy_resolution_actor(value)
        if copied is None:
            raise ValueError("requested_by is required for execution-profile adoption.")
        if copied.source is None:
            raise ValueError(
                "requested_by.source is required for execution-profile adoption provenance."
            )
        return copied


def copy_execution_profile_adoption_intent(
    intent: ExecutionProfileAdoptionIntent,
) -> ExecutionProfileAdoptionIntent:
    """Copy caller-owned adoption intent without serializing unvalidated fields."""

    if type(intent) is not ExecutionProfileAdoptionIntent:
        raise TypeError("Execution-profile adoption requires ExecutionProfileAdoptionIntent.")
    requested_by = copy_resolution_actor(intent.requested_by)
    if requested_by is None:
        raise ValueError("requested_by is required for execution-profile adoption.")
    return ExecutionProfileAdoptionIntent(
        idempotency_key=intent.idempotency_key,
        reason=intent.reason,
        requested_by=requested_by,
    )


class ExecutionProfilePolicyRequest(BaseModel):
    """Bounded application-policy input for one profile difference."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    expected_profile: ExecutionProfileIdentity
    candidate_profile: ExecutionProfileIdentity
    changed_component_classes: tuple[ExecutionProfileComponentClass, ...]
    intent: ExecutionProfileAdoptionIntent | None = None
    authority_review_required: StrictBool = False
    source_provider_name: str
    source_model: str
    target_provider_name: str
    target_model: str

    @field_validator(
        "session_id",
        "source_provider_name",
        "source_model",
        "target_provider_name",
        "target_model",
    )
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("expected_profile", "candidate_profile", mode="before")
    @classmethod
    def copy_profile(cls, value: object) -> ExecutionProfileIdentity:
        if isinstance(value, ExecutionProfileIdentity):
            value = value.model_dump(mode="json")
        return ExecutionProfileIdentity.model_validate(value)

    @field_validator("intent", mode="before")
    @classmethod
    def copy_intent(cls, value: object) -> ExecutionProfileAdoptionIntent | None:
        if value is None:
            return None
        if isinstance(value, ExecutionProfileAdoptionIntent):
            return copy_execution_profile_adoption_intent(value)
        return ExecutionProfileAdoptionIntent.model_validate(value)

    @model_validator(mode="after")
    def validate_changed_components(self) -> ExecutionProfilePolicyRequest:
        expected = changed_execution_profile_components(
            self.expected_profile,
            self.candidate_profile,
        )
        if self.changed_component_classes != expected:
            raise ValueError("changed_component_classes do not match the supplied profiles.")
        if execution_profile_changes_authority(expected) and not self.authority_review_required:
            raise ValueError(
                "Authority-changing components require execution-profile authority review."
            )
        return self


class ExecutionProfilePolicyResult(BaseModel):
    """Defensively copied result returned by an application policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    action: ExecutionProfilePolicyAction
    reason: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS)
    authority_decision: ExecutionProfileAuthorityDecision = (
        ExecutionProfileAuthorityDecision.NOT_REQUIRED
    )

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return require_durable_nonblank(value, "reason")


class ExecutionProfilePolicy(ABC):
    """Application-owned compatibility and adoption authority."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return a stable, versioned, non-secret policy identity."""

    @abstractmethod
    async def decide(
        self,
        request: ExecutionProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        """Classify one non-equal profile before governed work begins."""


class ExecutionProfileDecision(BaseModel):
    """Complete runtime-owned decision committed at profile admission."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    kind: ExecutionProfileDecisionKind
    expected_profile: ExecutionProfileIdentity
    candidate_profile: ExecutionProfileIdentity
    changed_component_classes: tuple[ExecutionProfileComponentClass, ...]
    policy_identity: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS)
    policy_reason: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS)
    authority_decision: ExecutionProfileAuthorityDecision
    idempotency_identity: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS)
    adoption_request_fingerprint: str | None = None
    actor: ResolutionActor | None = None
    reason: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS)
    event: Event

    @field_validator("policy_identity", "idempotency_identity")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("adoption_request_fingerprint")
    @classmethod
    def validate_adoption_request_fingerprint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("adoption_request_fingerprint must be a lowercase SHA-256 digest.")
        return value

    @field_validator("policy_reason", "reason")
    @classmethod
    def validate_reason_text(cls, value: str, info) -> str:
        return require_durable_nonblank(value, info.field_name)

    @field_validator("expected_profile", "candidate_profile", mode="before")
    @classmethod
    def copy_decision_profile(cls, value: object) -> ExecutionProfileIdentity:
        if isinstance(value, ExecutionProfileIdentity):
            value = value.model_dump(mode="json")
        return ExecutionProfileIdentity.model_validate(value)

    @field_validator("actor")
    @classmethod
    def copy_actor(cls, value: ResolutionActor | None) -> ResolutionActor | None:
        return copy_resolution_actor(value)

    @field_validator("event", mode="before")
    @classmethod
    def copy_decision_event(cls, value: object) -> Event:
        if isinstance(value, Event):
            return copy_event(value)
        return Event.model_validate(value)

    @model_validator(mode="after")
    def validate_decision(self) -> ExecutionProfileDecision:
        changed = changed_execution_profile_components(
            self.expected_profile,
            self.candidate_profile,
        )
        if changed != self.changed_component_classes:
            raise ValueError("Execution-profile decision changed components are inconsistent.")
        if self.kind is ExecutionProfileDecisionKind.EXACT_REUSE and changed:
            raise ValueError("Exact profile reuse cannot contain changed components.")
        unmodeled_authority_decision = self.kind in {
            ExecutionProfileDecisionKind.MIGRATION_REQUIRED,
            ExecutionProfileDecisionKind.REJECTED,
        } or (
            self.kind is ExecutionProfileDecisionKind.ADOPTED
            and self.authority_decision is ExecutionProfileAuthorityDecision.AUTHORIZED
        )
        if (
            self.kind is not ExecutionProfileDecisionKind.EXACT_REUSE
            and not changed
            and not unmodeled_authority_decision
        ):
            raise ValueError("A non-exact profile decision requires changed components.")
        if (
            self.kind is ExecutionProfileDecisionKind.EXACT_REUSE
            and self.authority_decision is not ExecutionProfileAuthorityDecision.NOT_REQUIRED
        ):
            raise ValueError("Exact profile reuse cannot carry an authority decision.")
        if (
            self.kind
            in {
                ExecutionProfileDecisionKind.COMPATIBLE_REUSE,
                ExecutionProfileDecisionKind.ADOPTED,
            }
            and self.authority_decision is ExecutionProfileAuthorityDecision.DENIED
        ):
            raise ValueError("A denied authority decision cannot admit an execution profile.")
        if self.kind is ExecutionProfileDecisionKind.COMPATIBLE_REUSE and any(
            component in _AUTHORITY_COMPONENT_CLASSES
            or component is ExecutionProfileComponentClass.PROVIDER_TARGET
            for component in changed
        ):
            raise ValueError(
                "Compatible reuse cannot change execution or persistent provider authority."
            )
        if (
            self.kind is ExecutionProfileDecisionKind.ADOPTED
            and any(component in _AUTHORITY_COMPONENT_CLASSES for component in changed)
            and self.authority_decision is not ExecutionProfileAuthorityDecision.AUTHORIZED
        ):
            raise ValueError("Execution-authority adoption requires explicit authority.")
        if self.kind is ExecutionProfileDecisionKind.ADOPTED:
            if self.actor is None:
                raise ValueError("Adopted execution profiles require an attributable actor.")
            if self.actor.source is None:
                raise ValueError("Adopted execution profiles require an actor provenance source.")
        expected_type = (
            "session.execution_profile.rejected"
            if self.kind is ExecutionProfileDecisionKind.REJECTED
            else "session.execution_profile.decided"
        )
        if str(self.event.type) != expected_type:
            raise ValueError("Execution-profile decision event has the wrong type.")
        if self.event.interaction_id is not None:
            raise ValueError("Execution-profile decisions cannot belong to an interaction.")
        if self.event.payload != execution_profile_decision_payload(
            kind=self.kind,
            expected_profile=self.expected_profile,
            candidate_profile=self.candidate_profile,
            changed_component_classes=self.changed_component_classes,
            policy_identity=self.policy_identity,
            policy_reason=self.policy_reason,
            authority_decision=self.authority_decision,
            idempotency_identity=self.idempotency_identity,
            adoption_request_fingerprint=self.adoption_request_fingerprint,
            actor=self.actor,
            reason=self.reason,
        ):
            raise ValueError("Execution-profile decision event payload is inconsistent.")
        return self


class ExecutionProfileComponentIdentity(BaseModel):
    """Redacted identity for one typed execution-profile component."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    component_class: ExecutionProfileComponentClass
    strength: ExecutionProfileIdentityStrength
    availability: ExecutionProfileIdentityAvailability
    fingerprint: str | None

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("fingerprint must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_availability(self) -> ExecutionProfileComponentIdentity:
        unavailable = self.availability is ExecutionProfileIdentityAvailability.UNAVAILABLE
        if unavailable != (self.fingerprint is None):
            raise ValueError("Unavailable components must omit their fingerprint.")
        if unavailable != (self.strength is ExecutionProfileIdentityStrength.UNAVAILABLE):
            raise ValueError("Unavailable components must use unavailable identity strength.")
        return self


class ExecutionProfileIdentity(BaseModel):
    """Versioned, redacted identity frozen before a session can execute."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1, 2, 4] = EXECUTION_PROFILE_SCHEMA_VERSION
    fingerprint: str
    components: tuple[ExecutionProfileComponentIdentity, ...]

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("fingerprint must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_components(self) -> ExecutionProfileIdentity:
        classes = tuple(component.component_class for component in self.components)
        required_classes = tuple(sorted(_SCHEMA_COMPONENT_CLASSES[self.schema_version], key=str))
        if classes != required_classes:
            raise ValueError(
                "Execution-profile components must contain every required class exactly once "
                "in sorted order."
            )
        expected = _profile_fingerprint(
            self.components,
            schema_version=self.schema_version,
        )
        if self.fingerprint != expected:
            raise ValueError("Execution-profile fingerprint does not match its components.")
        return self

    def component(
        self,
        component_class: ExecutionProfileComponentClass,
    ) -> ExecutionProfileComponentIdentity:
        for component in self.components:
            if component.component_class is component_class:
                return component
        raise KeyError(f"Execution profile has no {component_class.value} component.")


class ActiveInvocationExecutionProfile(BaseModel):
    """Durable profile authority for one active interaction and run epoch."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["cayu.active-invocation-execution-profile"] = (
        _ACTIVE_INVOCATION_EXECUTION_PROFILE_RECORD_TYPE
    )
    schema_version: Literal[1] = _ACTIVE_INVOCATION_EXECUTION_PROFILE_SCHEMA_VERSION
    session_id: str
    interaction_id: str
    run_epoch: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    profile: ExecutionProfileIdentity

    @field_validator("session_id", "interaction_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("profile", mode="before")
    @classmethod
    def copy_profile(cls, value: object) -> ExecutionProfileIdentity:
        if isinstance(value, ExecutionProfileIdentity):
            value = value.model_dump(mode="json")
        return ExecutionProfileIdentity.model_validate(value)


class _ExecutionProfileAdmissionRequestRejected(RuntimeError):
    """Private signal for deterministic request rejection before admission."""


class ExecutionProfileMismatchError(RuntimeError):
    """Raised after durable evidence rejects a changed execution profile."""

    def __init__(
        self,
        *,
        session_id: str,
        expected_profile_fingerprint: str,
        candidate_profile_fingerprint: str,
        changed_component_classes: tuple[ExecutionProfileComponentClass, ...],
    ) -> None:
        self.session_id = session_id
        self.expected_profile_fingerprint = expected_profile_fingerprint
        self.candidate_profile_fingerprint = candidate_profile_fingerprint
        self.changed_component_classes = changed_component_classes
        changed = ", ".join(component.value for component in changed_component_classes) or (
            "decision-bearing authority outside the structural profile"
        )
        super().__init__(self._message(session_id=session_id, changed=changed))

    def _message(self, *, session_id: str, changed: str) -> str:
        return (
            f"Session {session_id} execution profile changed in: {changed}. "
            "Start a new session or use an explicit profile-adoption flow."
        )


class ExecutionProfileAdoptionRejected(ExecutionProfileMismatchError):
    """Raised after policy durably rejects an explicit adoption request."""

    def _message(self, *, session_id: str, changed: str) -> str:
        return (
            f"Session {session_id} execution-profile adoption was rejected for changes in: "
            f"{changed}. Inspect the durable decision evidence or start a new session."
        )


class ExecutionProfileMigrationRequired(ExecutionProfileMismatchError):
    """Raised after policy records that an explicit migration is required."""

    def _message(self, *, session_id: str, changed: str) -> str:
        return (
            f"Session {session_id} requires an execution-profile migration before changes in: "
            f"{changed}. No resumed work was admitted."
        )


class ExecutionProfilePolicyError(RuntimeError):
    """Raised when configured profile policy cannot produce an authoritative decision."""


class ExecutionProfileRejectionResult(BaseModel):
    """Durable rejection event plus whether an exact prior write was replayed."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    event: Event
    replayed: StrictBool = False

    @field_validator("event", mode="before")
    @classmethod
    def copy_rejection_event(cls, value: Event) -> Event:
        return copy_event(value)


def copy_execution_profile_policy_result(
    result: ExecutionProfilePolicyResult,
) -> ExecutionProfilePolicyResult:
    """Defensively validate one result returned across the policy trust boundary."""

    if type(result) is not ExecutionProfilePolicyResult:
        raise TypeError("Execution-profile policies must return ExecutionProfilePolicyResult.")
    return ExecutionProfilePolicyResult(
        action=result.action,
        reason=result.reason,
        authority_decision=result.authority_decision,
    )


def copy_execution_profile_decision(
    decision: ExecutionProfileDecision,
) -> ExecutionProfileDecision:
    """Copy a runtime decision without erasing its event authority provenance."""

    if type(decision) is not ExecutionProfileDecision:
        raise TypeError("decision must be an ExecutionProfileDecision.")
    return ExecutionProfileDecision(
        kind=decision.kind,
        expected_profile=decision.expected_profile,
        candidate_profile=decision.candidate_profile,
        changed_component_classes=decision.changed_component_classes,
        policy_identity=decision.policy_identity,
        policy_reason=decision.policy_reason,
        authority_decision=decision.authority_decision,
        idempotency_identity=decision.idempotency_identity,
        adoption_request_fingerprint=decision.adoption_request_fingerprint,
        actor=decision.actor,
        reason=decision.reason,
        event=copy_event(decision.event),
    )


def execution_profile_decision_payload(
    *,
    kind: ExecutionProfileDecisionKind,
    expected_profile: ExecutionProfileIdentity,
    candidate_profile: ExecutionProfileIdentity,
    changed_component_classes: tuple[ExecutionProfileComponentClass, ...],
    policy_identity: str,
    policy_reason: str,
    authority_decision: ExecutionProfileAuthorityDecision,
    idempotency_identity: str,
    adoption_request_fingerprint: str | None = None,
    actor: ResolutionActor | None,
    reason: str,
) -> dict[str, Any]:
    """Build the complete bounded durable evidence payload for one decision."""

    payload = {
        "decision": kind.value,
        "expected_profile": expected_profile.model_dump(mode="json"),
        "candidate_profile": candidate_profile.model_dump(mode="json"),
        "changed_component_classes": [component.value for component in changed_component_classes],
        "actor": resolution_actor_payload(actor),
        "policy_identity": policy_identity,
        "policy_reason": policy_reason,
        "authority_decision": authority_decision.value,
        "reason": reason,
        "idempotency_identity": idempotency_identity,
    }
    if adoption_request_fingerprint is not None:
        payload["adoption_request_fingerprint"] = adoption_request_fingerprint
    return payload


def _aggregate_identity_strength(
    *,
    process_local: bool,
    application_versioned: bool,
) -> ExecutionProfileIdentityStrength:
    if process_local:
        return ExecutionProfileIdentityStrength.PROCESS_LOCAL
    if application_versioned:
        return ExecutionProfileIdentityStrength.APPLICATION_VERSIONED
    return ExecutionProfileIdentityStrength.STRUCTURAL


def direct_tool_capability_ceiling_component(
    tool_names: Iterable[str],
) -> ExecutionProfileComponentIdentity:
    """Build the canonical profile component for one direct-tool ceiling."""

    names = tuple(require_durable_clean_nonblank(name, "tool_name") for name in tool_names)
    if len(names) != len(set(names)):
        raise ValueError("tool_names must contain unique values.")
    return _available_component(
        ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
        ExecutionProfileIdentityStrength.STRUCTURAL,
        {
            "view_kind": "direct",
            "generation": 1,
            "grant_baseline": list(names),
        },
    )


def _canonical_direct_tool_material(
    direct_tools: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if isinstance(direct_tools, str | bytes | bytearray | Mapping | BaseModel):
        raise TypeError("direct_tools must be an iterable of canonical tool identities.")
    material: list[dict[str, str]] = []
    for index, item in enumerate(direct_tools):
        if index >= TOOL_CATALOGUE_MAX_TOOLS:
            raise ValueError(
                f"direct_tools cannot contain more than {TOOL_CATALOGUE_MAX_TOOLS} items."
            )
        if not isinstance(item, Mapping):
            raise TypeError(f"direct_tools[{index}] must be a mapping.")
        if set(item) != {"tool_id", "descriptor_version"}:
            raise ValueError(
                f"direct_tools[{index}] must contain exactly tool_id and descriptor_version."
            )
        material.append(
            {
                "tool_id": validate_canonical_tool_id(
                    item["tool_id"],
                    f"direct_tools[{index}].tool_id",
                ),
                "descriptor_version": validate_tool_descriptor_version(
                    item["descriptor_version"],
                    f"direct_tools[{index}].descriptor_version",
                ),
            }
        )
    tool_ids = tuple(item["tool_id"] for item in material)
    if len(tool_ids) != len(set(tool_ids)):
        raise ValueError("direct_tools must contain unique tool_id values.")
    return material


def build_execution_profile_identity(
    *,
    runtime_name: str,
    runtime_version: str | None,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None,
    direct_tools: Iterable[Mapping[str, Any]],
    tool_catalogue_revision: str,
    tool_implementations: Iterable[Mapping[str, Any]] | None = None,
    tool_implementations_process_local: bool = False,
    tool_implementations_application_versioned: bool = False,
    tool_view_grants: Mapping[str, Any] | None = None,
    execution_policies: Mapping[str, Any] | None = None,
    execution_policies_process_local: bool = False,
    execution_policies_application_versioned: bool = False,
    invocation_policies: Iterable[Mapping[str, Any]] = (),
    invocation_policies_process_local: bool = False,
    invocation_policies_application_versioned: bool = False,
    runtime_hooks: Iterable[Mapping[str, Any]] = (),
    runtime_hooks_process_local: bool = False,
    runtime_hooks_application_versioned: bool = False,
    execution_environment: Mapping[str, Any] | None = None,
    execution_environment_process_local: bool = False,
    execution_environment_application_versioned: bool = False,
    effect_authority: Mapping[str, Any] | None = None,
    context_selection: Mapping[str, Any] | None = None,
    context_selection_process_local: bool = False,
    context_selection_application_versioned: bool = False,
    automatic_recall: Mapping[str, Any] | None = None,
    automatic_recall_process_local: bool = False,
    automatic_recall_application_versioned: bool = False,
    context_compaction: Mapping[str, Any] | None = None,
    context_compaction_process_local: bool = False,
    context_compaction_application_versioned: bool = False,
    live_state_projection: Mapping[str, Any] | None = None,
    provider_adapter: Mapping[str, Any] | None = None,
    provider_adapter_process_local: bool = False,
    provider_adapter_application_versioned: bool = False,
    provider_request_policy: Mapping[str, Any] | None = None,
    provider_request_policy_process_local: bool = False,
    provider_request_policy_application_versioned: bool = False,
    application_budget_policy: Mapping[str, Any] | None = None,
    invocation_budget_policy: Mapping[str, Any] | None = None,
    structured_output: Mapping[str, Any] | None = None,
    finalization: Mapping[str, Any] | None = None,
) -> ExecutionProfileIdentity:
    """Build a profile without retaining raw prompts, schemas, or tool names."""

    direct_tool_material = _canonical_direct_tool_material(direct_tools)
    tool_catalogue_revision = validate_tool_catalogue_revision(
        tool_catalogue_revision,
        "tool_catalogue_revision",
    )
    direct_tool_identity_material = {
        "kind": "cayu:catalogued-direct-tools",
        "version": 1,
        "catalogue_revision": tool_catalogue_revision,
        "tools": direct_tool_material,
    }
    if tool_view_grants is None and direct_tool_material:
        raise ValueError("tool_view_grants must be provided when direct_tools is non-empty.")
    implementation_material = (
        [{"tool_id": item["tool_id"]} for item in direct_tool_material]
        if tool_implementations is None
        else list(tool_implementations)
    )
    components = (
        _available_component(
            ExecutionProfileComponentClass.DIRECT_TOOLS,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            direct_tool_identity_material,
        ),
        _available_component(
            ExecutionProfileComponentClass.TOOL_IMPLEMENTATIONS,
            _aggregate_identity_strength(
                process_local=tool_implementations_process_local,
                application_versioned=tool_implementations_application_versioned,
            ),
            implementation_material,
        ),
        _available_component(
            ExecutionProfileComponentClass.TOOL_VIEW_GRANTS,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            (
                {
                    "view_kind": "direct",
                    "generation": 1,
                    "grant_baseline": [],
                }
                if tool_view_grants is None
                else tool_view_grants
            ),
        ),
        _available_component(
            ExecutionProfileComponentClass.EXECUTION_POLICIES,
            _aggregate_identity_strength(
                process_local=execution_policies_process_local,
                application_versioned=execution_policies_application_versioned,
            ),
            (
                {
                    "tool_policy": "cayu:allow-all-tool-policy:v1",
                    "command_policies": [],
                    "loop_policies": [],
                }
                if execution_policies is None
                else execution_policies
            ),
        ),
        _available_component(
            ExecutionProfileComponentClass.INVOCATION_POLICIES,
            _aggregate_identity_strength(
                process_local=invocation_policies_process_local,
                application_versioned=invocation_policies_application_versioned,
            ),
            list(invocation_policies),
        ),
        _available_component(
            ExecutionProfileComponentClass.RUNTIME_HOOKS,
            _aggregate_identity_strength(
                process_local=runtime_hooks_process_local,
                application_versioned=runtime_hooks_application_versioned,
            ),
            list(runtime_hooks),
        ),
        _available_component(
            ExecutionProfileComponentClass.EXECUTION_ENVIRONMENT,
            _aggregate_identity_strength(
                process_local=execution_environment_process_local,
                application_versioned=execution_environment_application_versioned,
            ),
            (
                {
                    "environment": None,
                    "execution_requirements": {
                        "code_trust": "trusted",
                        "real_secret_visibility": "allowed",
                        "network_access": "unrestricted",
                        "guest_privilege": "unrestricted",
                        "host_filesystem": "unrestricted",
                        "cancellation": "best_effort",
                        "cleanup": "best_effort",
                        "durability": "ephemeral",
                        "minimum_evidence": "declared",
                        "evidence_overrides": [],
                    },
                }
                if execution_environment is None
                else execution_environment
            ),
        ),
        _available_component(
            ExecutionProfileComponentClass.EFFECT_AUTHORITY,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            (
                {
                    "tool_effects": [
                        {
                            "name": item.get("name"),
                            "effect": item.get("effect"),
                            "workspace_mutation": bool(item.get("workspace_mutation", False)),
                        }
                        for item in direct_tool_material
                    ],
                    "credential_authority": "none",
                    "egress_authority": "unrestricted",
                }
                if effect_authority is None
                else effect_authority
            ),
        ),
        _available_component(
            ExecutionProfileComponentClass.CONTEXT_SELECTION,
            _aggregate_identity_strength(
                process_local=context_selection_process_local,
                application_versioned=context_selection_application_versioned,
            ),
            {"kind": "cayu:default-context-selection:v1"}
            if context_selection is None
            else context_selection,
        ),
        _available_component(
            ExecutionProfileComponentClass.AUTOMATIC_RECALL,
            _aggregate_identity_strength(
                process_local=automatic_recall_process_local,
                application_versioned=automatic_recall_application_versioned,
            ),
            {"kind": "none", "version": 1} if automatic_recall is None else automatic_recall,
        ),
        _available_component(
            ExecutionProfileComponentClass.CONTEXT_COMPACTION,
            _aggregate_identity_strength(
                process_local=context_compaction_process_local,
                application_versioned=context_compaction_application_versioned,
            ),
            {"kind": "none", "version": 1} if context_compaction is None else context_compaction,
        ),
        _available_component(
            ExecutionProfileComponentClass.LIVE_STATE_PROJECTION,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"kind": "none", "version": 1}
            if live_state_projection is None
            else live_state_projection,
        ),
        _available_component(
            ExecutionProfileComponentClass.PROVIDER_ADAPTER,
            _aggregate_identity_strength(
                process_local=provider_adapter_process_local,
                application_versioned=provider_adapter_application_versioned,
            ),
            {"kind": "provider-defaults", "version": 1}
            if provider_adapter is None
            else provider_adapter,
        ),
        _available_component(
            ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
            _aggregate_identity_strength(
                process_local=provider_request_policy_process_local,
                application_versioned=provider_request_policy_application_versioned,
            ),
            {"kind": "provider-defaults", "version": 1}
            if provider_request_policy is None
            else provider_request_policy,
        ),
        _available_component(
            ExecutionProfileComponentClass.APPLICATION_BUDGET_POLICY,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"limit_ids": []} if application_budget_policy is None else application_budget_policy,
        ),
        _available_component(
            ExecutionProfileComponentClass.INVOCATION_BUDGET_POLICY,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"limit_ids": []} if invocation_budget_policy is None else invocation_budget_policy,
        ),
        _available_component(
            ExecutionProfileComponentClass.STRUCTURED_OUTPUT,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"kind": "none", "version": 1} if structured_output is None else structured_output,
        ),
        _available_component(
            ExecutionProfileComponentClass.FINALIZATION,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"kind": "cayu:default-finalization:v1"} if finalization is None else finalization,
        ),
        _available_component(
            ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"system_prompt": durable_system_prompt},
        ),
        _available_component(
            ExecutionProfileComponentClass.PROVIDER_TARGET,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            {"provider_name": provider_name, "model": model},
        ),
        (
            _available_component(
                ExecutionProfileComponentClass.RUNTIME,
                ExecutionProfileIdentityStrength.STRUCTURAL,
                {"runtime_name": runtime_name, "runtime_version": runtime_version},
            )
            if runtime_version is not None
            else _unavailable_component(ExecutionProfileComponentClass.RUNTIME)
        ),
    )
    sorted_components = tuple(sorted(components, key=lambda component: component.component_class))
    return ExecutionProfileIdentity(
        fingerprint=_profile_fingerprint(
            sorted_components,
            schema_version=EXECUTION_PROFILE_SCHEMA_VERSION,
        ),
        components=sorted_components,
    )


def changed_execution_profile_components(
    expected: ExecutionProfileIdentity,
    candidate: ExecutionProfileIdentity,
) -> tuple[ExecutionProfileComponentClass, ...]:
    """Return bounded component classes that changed or cannot be verified."""

    expected_by_class = {item.component_class: item for item in expected.components}
    candidate_by_class = {item.component_class: item for item in candidate.components}
    classes = sorted(set(expected_by_class) | set(candidate_by_class), key=str)
    return tuple(
        component_class
        for component_class in classes
        if (
            expected_by_class.get(component_class) != candidate_by_class.get(component_class)
            or expected_by_class[component_class].availability
            is ExecutionProfileIdentityAvailability.UNAVAILABLE
            or candidate_by_class[component_class].availability
            is ExecutionProfileIdentityAvailability.UNAVAILABLE
        )
    )


def unavailable_execution_profile_components(
    profile: ExecutionProfileIdentity,
) -> tuple[ExecutionProfileComponentClass, ...]:
    """Return required component classes with no deterministic identity."""

    return tuple(
        component.component_class
        for component in profile.components
        if component.availability is ExecutionProfileIdentityAvailability.UNAVAILABLE
    )


def execution_profile_with_component(
    profile: ExecutionProfileIdentity,
    component: ExecutionProfileComponentIdentity,
) -> ExecutionProfileIdentity:
    """Return a profile with one durable component identity replaced."""

    by_class = {item.component_class: item for item in profile.components}
    by_class[component.component_class] = component
    components = tuple(sorted(by_class.values(), key=lambda item: item.component_class))
    return ExecutionProfileIdentity(
        schema_version=profile.schema_version,
        fingerprint=_profile_fingerprint(
            components,
            schema_version=profile.schema_version,
        ),
        components=components,
    )


def execution_profile_with_durable_system_projection_digest(
    profile: ExecutionProfileIdentity,
    digest: str,
) -> ExecutionProfileIdentity:
    """Bind a canonical durable-system message digest without raw prompt text."""

    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("durable system projection digest must be a lowercase SHA-256 digest.")
    return execution_profile_with_component(
        profile,
        ExecutionProfileComponentIdentity(
            component_class=ExecutionProfileComponentClass.DURABLE_SYSTEM_PROJECTION,
            strength=ExecutionProfileIdentityStrength.STRUCTURAL,
            availability=ExecutionProfileIdentityAvailability.AVAILABLE,
            fingerprint=digest,
        ),
    )


def execution_profile_session_metadata(
    profile: ExecutionProfileIdentity,
) -> dict[str, Any]:
    """Return the bounded runtime-owned record stored with a new session."""

    dumped = profile.model_dump(mode="json")
    return {
        "record_type": _EXECUTION_PROFILE_RECORD_TYPE,
        "schema_version": _EXECUTION_PROFILE_RECORD_SCHEMA_VERSION,
        "baseline": dumped,
        "expected": dumped,
    }


def execution_profile_metadata_after_adoption(
    metadata: Mapping[str, Any],
    profile: ExecutionProfileIdentity,
) -> dict[str, Any]:
    """Advance the expected profile while retaining the immutable baseline."""

    copied = copy_durable_json_value(dict(metadata), "session.metadata")
    current = copied.get(EXECUTION_PROFILE_METADATA_KEY)
    if type(current) is not dict:
        raise ValueError("Session has no durable execution-profile identity.")
    # Validate the complete record before retaining its immutable baseline.
    execution_profile_from_session_metadata(copied)
    current["expected"] = profile.model_dump(mode="json")
    copied[EXECUTION_PROFILE_METADATA_KEY] = current
    return copied


def execution_profile_from_session_metadata(
    metadata: Mapping[str, Any],
) -> ExecutionProfileIdentity:
    """Load the current expected profile, failing closed on absent/malformed state."""

    raw = metadata.get(EXECUTION_PROFILE_METADATA_KEY)
    if type(raw) is not dict:
        raise ValueError("Session has no durable execution-profile identity.")
    if set(raw) != {"record_type", "schema_version", "baseline", "expected"}:
        raise ValueError("Session execution-profile metadata is malformed.")
    if (
        raw["record_type"] != _EXECUTION_PROFILE_RECORD_TYPE
        or raw["schema_version"] != _EXECUTION_PROFILE_RECORD_SCHEMA_VERSION
    ):
        raise ValueError("Session execution-profile metadata version is unsupported.")
    # Revalidate both identities after every backend round trip. The immutable
    # baseline is durable audit authority even though admission returns the
    # current expectation. No raw component material is stored in either value.
    ExecutionProfileIdentity.model_validate(
        copy_durable_json_value(raw["baseline"], "execution_profile.baseline")
    )
    return ExecutionProfileIdentity.model_validate(
        copy_durable_json_value(raw["expected"], "execution_profile.expected")
    )


def execution_profile_baseline_from_session_metadata(
    metadata: Mapping[str, Any],
) -> ExecutionProfileIdentity:
    """Load the immutable creation baseline from one session profile record."""

    raw = metadata.get(EXECUTION_PROFILE_METADATA_KEY)
    if type(raw) is not dict:
        raise ValueError("Session has no durable execution-profile identity.")
    if set(raw) != {"record_type", "schema_version", "baseline", "expected"}:
        raise ValueError("Session execution-profile metadata is malformed.")
    if (
        raw["record_type"] != _EXECUTION_PROFILE_RECORD_TYPE
        or raw["schema_version"] != _EXECUTION_PROFILE_RECORD_SCHEMA_VERSION
    ):
        raise ValueError("Session execution-profile metadata version is unsupported.")
    # Validate the mutable expectation too. A malformed sibling field must not
    # be bypassed merely because a caller asks only for the immutable baseline.
    ExecutionProfileIdentity.model_validate(
        copy_durable_json_value(raw["expected"], "execution_profile.expected")
    )
    return ExecutionProfileIdentity.model_validate(
        copy_durable_json_value(raw["baseline"], "execution_profile.baseline")
    )


def active_invocation_execution_profile_from_checkpoint(
    checkpoint: Mapping[str, Any] | None,
) -> ActiveInvocationExecutionProfile | None:
    """Load the active invocation profile, failing closed on malformed authority."""

    if checkpoint is None:
        return None
    raw = checkpoint.get(ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY)
    if raw is None:
        return None
    return ActiveInvocationExecutionProfile.model_validate(
        copy_durable_json_value(raw, "active_invocation_execution_profile")
    )


def active_invocation_execution_profile_matches_session_epoch(
    snapshot: ActiveInvocationExecutionProfile,
    *,
    session_id: str,
    run_epoch: int,
) -> bool:
    """Return whether active authority can continue from this durable epoch."""

    permitted_epochs = {run_epoch}
    if run_epoch > 0:
        # Releasing a completed, paused run fences that epoch by advancing the
        # session once. The invocation snapshot continues to own the open
        # interaction until a continuation atomically rebinds it.
        permitted_epochs.add(run_epoch - 1)
    return snapshot.session_id == session_id and snapshot.run_epoch in permitted_epochs


def active_invocation_execution_profile_is_released(
    snapshot: ActiveInvocationExecutionProfile,
    *,
    session_id: str,
    run_epoch: int,
) -> bool:
    """Whether trailing work released the exact invocation's durable run fence."""

    return (
        snapshot.session_id == session_id and run_epoch > 0 and snapshot.run_epoch == run_epoch - 1
    )


def checkpoint_with_active_invocation_execution_profile(
    checkpoint: Mapping[str, Any] | None,
    *,
    session_id: str,
    interaction_id: str,
    run_epoch: int,
    profile: ExecutionProfileIdentity,
    expected: ActiveInvocationExecutionProfile | None = None,
) -> dict[str, Any]:
    """Bind one profile to an interaction epoch with optional CAS authority."""

    current = active_invocation_execution_profile_from_checkpoint(checkpoint)
    if expected is not None and current != expected:
        raise RuntimeError("Active invocation execution profile changed before it was claimed.")
    snapshot = ActiveInvocationExecutionProfile(
        session_id=session_id,
        interaction_id=interaction_id,
        run_epoch=run_epoch,
        profile=profile,
    )
    updated = {} if checkpoint is None else copy_durable_json_value(dict(checkpoint), "checkpoint")
    updated[ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY] = snapshot.model_dump(mode="json")
    return updated


def _available_component(
    component_class: ExecutionProfileComponentClass,
    strength: ExecutionProfileIdentityStrength,
    material: Any,
) -> ExecutionProfileComponentIdentity:
    fingerprint = sha256(
        canonical_durable_json_bytes(material, f"execution_profile.{component_class.value}")
    ).hexdigest()
    return ExecutionProfileComponentIdentity(
        component_class=component_class,
        strength=strength,
        availability=ExecutionProfileIdentityAvailability.AVAILABLE,
        fingerprint=fingerprint,
    )


def _unavailable_component(
    component_class: ExecutionProfileComponentClass,
) -> ExecutionProfileComponentIdentity:
    return ExecutionProfileComponentIdentity(
        component_class=component_class,
        strength=ExecutionProfileIdentityStrength.UNAVAILABLE,
        availability=ExecutionProfileIdentityAvailability.UNAVAILABLE,
        fingerprint=None,
    )


def _profile_fingerprint(
    components: tuple[ExecutionProfileComponentIdentity, ...],
    *,
    schema_version: int,
) -> str:
    material = {
        "schema_version": schema_version,
        "components": [component.model_dump(mode="json") for component in components],
    }
    return sha256(canonical_durable_json_bytes(material, "execution_profile")).hexdigest()
