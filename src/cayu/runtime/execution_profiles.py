"""Durable, redacted execution-profile identities for session admission."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_value,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.core.events import Event, copy_event
from cayu.runtime.approvals import (
    ResolutionActor,
    copy_resolution_actor,
    resolution_actor_payload,
)

EXECUTION_PROFILE_SCHEMA_VERSION = 1
EXECUTION_PROFILE_METADATA_KEY = "cayu:execution_profile"
_EXECUTION_PROFILE_RECORD_TYPE = "cayu.execution-profile"
EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS = 4096
EXECUTION_PROFILE_ADOPTION_ID_MAX_CHARS = 256


class ExecutionProfileComponentClass(StrEnum):
    """Stable classes of execution authority represented by a profile."""

    RUNTIME = "runtime"
    PROVIDER_TARGET = "provider_target"
    DURABLE_SYSTEM_PROJECTION = "durable_system_projection"
    DIRECT_TOOLS = "direct_tools"


class ExecutionProfileIdentityStrength(StrEnum):
    """How strongly one component is identified."""

    APPLICATION_VERSIONED = "application_versioned"
    STRUCTURAL = "structural"
    UNAVAILABLE = "unavailable"


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
        if self.authority_review_required != (
            ExecutionProfileComponentClass.DIRECT_TOOLS in expected
        ):
            raise ValueError(
                "authority_review_required does not match the changed profile authority."
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
        if self.kind is not ExecutionProfileDecisionKind.EXACT_REUSE and not changed:
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
            component
            in {
                ExecutionProfileComponentClass.DIRECT_TOOLS,
                ExecutionProfileComponentClass.PROVIDER_TARGET,
            }
            for component in changed
        ):
            raise ValueError(
                "Compatible reuse cannot change direct-tool or persistent provider authority."
            )
        if (
            self.kind is ExecutionProfileDecisionKind.ADOPTED
            and ExecutionProfileComponentClass.DIRECT_TOOLS in changed
            and self.authority_decision is not ExecutionProfileAuthorityDecision.AUTHORIZED
        ):
            raise ValueError("Direct-tool adoption requires explicit authority.")
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

    schema_version: Literal[1] = EXECUTION_PROFILE_SCHEMA_VERSION
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
        required_classes = tuple(sorted(ExecutionProfileComponentClass, key=str))
        if classes != required_classes:
            raise ValueError(
                "Execution-profile components must contain every required class exactly once "
                "in sorted order."
            )
        expected = _profile_fingerprint(self.components)
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
        changed = ", ".join(component.value for component in changed_component_classes)
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


def build_execution_profile_identity(
    *,
    runtime_name: str,
    runtime_version: str | None,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None,
    direct_tools: Iterable[Mapping[str, Any]],
) -> ExecutionProfileIdentity:
    """Build a profile without retaining raw prompts, schemas, or tool names."""

    components = (
        _available_component(
            ExecutionProfileComponentClass.DIRECT_TOOLS,
            ExecutionProfileIdentityStrength.STRUCTURAL,
            list(direct_tools),
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
                ExecutionProfileIdentityStrength.APPLICATION_VERSIONED,
                {"runtime_name": runtime_name, "runtime_version": runtime_version},
            )
            if runtime_version is not None
            else _unavailable_component(ExecutionProfileComponentClass.RUNTIME)
        ),
    )
    sorted_components = tuple(sorted(components, key=lambda component: component.component_class))
    return ExecutionProfileIdentity(
        fingerprint=_profile_fingerprint(sorted_components),
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
        fingerprint=_profile_fingerprint(components),
        components=components,
    )


def execution_profile_session_metadata(
    profile: ExecutionProfileIdentity,
) -> dict[str, Any]:
    """Return the bounded runtime-owned record stored with a new session."""

    dumped = profile.model_dump(mode="json")
    return {
        "record_type": _EXECUTION_PROFILE_RECORD_TYPE,
        "schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
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
        or raw["schema_version"] != EXECUTION_PROFILE_SCHEMA_VERSION
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
) -> str:
    material = {
        "schema_version": EXECUTION_PROFILE_SCHEMA_VERSION,
        "components": [component.model_dump(mode="json") for component in components],
    }
    return sha256(canonical_durable_json_bytes(material, "execution_profile")).hexdigest()
