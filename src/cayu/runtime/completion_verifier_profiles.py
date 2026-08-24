"""Durable execution-profile authority for completion verifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from hashlib import sha256
from typing import Literal

from pydantic import Field, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    revalidate_model_input,
    revalidate_model_inputs,
)
from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.runtime.approvals import ResolutionActor, copy_resolution_actor
from cayu.runtime.execution_profiles import (
    EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileIdentityStrength,
    ExecutionProfilePolicyResult,
    copy_execution_profile_adoption_intent,
)
from cayu.runtime.work_contracts import (
    CompletionVerifierRef,
    FrozenWorkContractModel,
    WorkCompletionConflict,
    WorkContractRef,
    copy_work_contract_ref,
    normalize_utc_datetime,
    validate_work_completion_idempotency_key,
    validate_work_completion_linked_id,
)

COMPLETION_VERIFIER_PROFILE_SCHEMA_VERSION = 1
COMPLETION_VERIFIER_PROFILE_COMPONENT_MAX_ITEMS = 64
COMPLETION_VERIFIER_PROFILE_CHANGED_COMPONENT_MAX_ITEMS = (
    COMPLETION_VERIFIER_PROFILE_COMPONENT_MAX_ITEMS * 2 + 1
)
COMPLETION_VERIFIER_PROFILE_TEXT_MAX_CHARS = 256
_ADAPTER_COMPONENT_ID = "adapter"


def _sha256_digest(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


class CompletionVerifierProfileComponentDeclaration(FrozenWorkContractModel):
    """Application-declared identity input for one verifier dependency."""

    component_id: str = Field(max_length=COMPLETION_VERIFIER_PROFILE_TEXT_MAX_CHARS)
    identity: ExecutionProfileBehaviorIdentity

    @field_validator("component_id")
    @classmethod
    def validate_component_id(cls, value: str) -> str:
        value = require_durable_clean_nonblank(value, "component_id")
        if value == _ADAPTER_COMPONENT_ID:
            raise ValueError("component_id 'adapter' is reserved for the verifier adapter.")
        return value

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value: object) -> ExecutionProfileBehaviorIdentity:
        if type(value) is not ExecutionProfileBehaviorIdentity:
            raise TypeError("identity must be an ExecutionProfileBehaviorIdentity.")
        copied = copy_execution_profile_behavior_identity(value)
        if copied is None:  # pragma: no cover - excluded by the exact type check
            raise ValueError("identity is required.")
        return copied


class CompletionVerifierProfileComponentIdentity(FrozenWorkContractModel):
    """Bounded fingerprint-only authority for one verifier component."""

    component_id: str = Field(max_length=COMPLETION_VERIFIER_PROFILE_TEXT_MAX_CHARS)
    strength: Literal[ExecutionProfileIdentityStrength.APPLICATION_VERSIONED] = (
        ExecutionProfileIdentityStrength.APPLICATION_VERSIONED
    )
    fingerprint: str

    @field_validator("component_id")
    @classmethod
    def validate_component_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "component_id")

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return _sha256_digest(value, "fingerprint")


class CompletionVerifierExecutionProfile(FrozenWorkContractModel):
    """Immutable verifier-specific profile, distinct from a session profile."""

    schema_version: Literal[1] = COMPLETION_VERIFIER_PROFILE_SCHEMA_VERSION
    verifier: CompletionVerifierRef
    components: tuple[CompletionVerifierProfileComponentIdentity, ...] = Field(
        min_length=1,
        max_length=COMPLETION_VERIFIER_PROFILE_COMPONENT_MAX_ITEMS + 1,
    )
    fingerprint: str

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("verifier", mode="before")
    @classmethod
    def copy_verifier(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionVerifierRef)

    @field_validator("components", mode="before")
    @classmethod
    def copy_components(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            CompletionVerifierProfileComponentIdentity,
            maximum=COMPLETION_VERIFIER_PROFILE_COMPONENT_MAX_ITEMS + 1,
            field_name="components",
        )

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return _sha256_digest(value, "fingerprint")

    @model_validator(mode="after")
    def validate_canonical_profile(self) -> CompletionVerifierExecutionProfile:
        component_ids = tuple(component.component_id for component in self.components)
        if component_ids != tuple(sorted(set(component_ids))):
            raise ValueError("components must contain unique component IDs in canonical order.")
        if _ADAPTER_COMPONENT_ID not in component_ids:
            raise ValueError("components must contain the verifier adapter identity.")
        if self.fingerprint != completion_verifier_profile_fingerprint(
            verifier=self.verifier,
            components=self.components,
        ):
            raise ValueError("fingerprint conflicts with the verifier profile material.")
        return self


class CompletionVerifierProfileAdoptionDecision(FrozenWorkContractModel):
    """Durable authorization for one changed verifier profile."""

    expected_profile_fingerprint: str
    candidate_profile_fingerprint: str
    changed_component_ids: tuple[str, ...] = Field(
        min_length=1,
        max_length=COMPLETION_VERIFIER_PROFILE_CHANGED_COMPONENT_MAX_ITEMS,
    )
    policy_identity: str = Field(max_length=COMPLETION_VERIFIER_PROFILE_TEXT_MAX_CHARS)
    authority_decision: Literal[ExecutionProfileAuthorityDecision.AUTHORIZED]
    idempotency_key: str
    requested_by: ResolutionActor
    reason: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS)
    policy_reason: str = Field(max_length=EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS)
    request_sha256: str

    @field_validator(
        "expected_profile_fingerprint",
        "candidate_profile_fingerprint",
        "request_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_digest(value, info.field_name)

    @field_validator("changed_component_ids")
    @classmethod
    def validate_changed_component_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        copied_items: list[str] = []
        for item in value:
            copied_item = require_durable_clean_nonblank(item, "changed_component_ids")
            if len(copied_item) > COMPLETION_VERIFIER_PROFILE_TEXT_MAX_CHARS:
                raise ValueError(
                    "changed_component_ids values must not exceed "
                    f"{COMPLETION_VERIFIER_PROFILE_TEXT_MAX_CHARS} characters."
                )
            copied_items.append(copied_item)
        copied = tuple(copied_items)
        if copied != tuple(sorted(set(copied))):
            raise ValueError("changed_component_ids must contain unique values in canonical order.")
        return copied

    @field_validator("policy_identity")
    @classmethod
    def validate_identity_text(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("reason", "policy_reason")
    @classmethod
    def validate_reason_text(cls, value: str, info) -> str:
        return require_durable_nonblank(value, info.field_name)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return validate_work_completion_idempotency_key(value)

    @field_validator("requested_by", mode="before")
    @classmethod
    def copy_requested_by(cls, value: object) -> ResolutionActor:
        if type(value) is not ResolutionActor:
            raise TypeError("requested_by must be a ResolutionActor.")
        copied = copy_resolution_actor(value)
        if copied is None:  # pragma: no cover - excluded by the exact type check
            raise ValueError("requested_by is required.")
        if copied.source is None:
            raise ValueError("requested_by.source is required for verifier-profile adoption.")
        if copied.claims:
            raise ValueError(
                "Verifier-profile adoption evidence cannot persist authorization claims."
            )
        return copied


class CompletionVerifierProfilePreparationRequest(FrozenWorkContractModel):
    """Exact insert-only request for one proposal's verifier profile."""

    proposal_id: str
    task_id: str
    attempt_id: str
    attempt_request_sha256: str
    source_execution_profile_fingerprint: str
    proposal_request_sha256: str
    contract: WorkContractRef
    profile: CompletionVerifierExecutionProfile
    expected_prior_proposal_id: str | None = None
    expected_prior_profile_fingerprint: str | None = None
    adoption: CompletionVerifierProfileAdoptionDecision | None = None

    @field_validator("proposal_id", "task_id", "attempt_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return validate_work_completion_linked_id(value, info.field_name)

    @field_validator(
        "attempt_request_sha256",
        "source_execution_profile_fingerprint",
        "proposal_request_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_digest(value, info.field_name)

    @field_validator("expected_prior_profile_fingerprint")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        return (
            None if value is None else _sha256_digest(value, "expected_prior_profile_fingerprint")
        )

    @field_validator("expected_prior_proposal_id")
    @classmethod
    def validate_optional_prior_proposal_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_work_completion_linked_id(value, "expected_prior_proposal_id")

    @field_validator("contract", mode="before")
    @classmethod
    def copy_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @field_validator("profile", mode="before")
    @classmethod
    def copy_profile(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionVerifierExecutionProfile)

    @field_validator("adoption", mode="before")
    @classmethod
    def copy_adoption(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not CompletionVerifierProfileAdoptionDecision:
            raise TypeError("adoption must be a CompletionVerifierProfileAdoptionDecision or None.")
        return _copy_adoption_decision(value)

    @model_validator(mode="after")
    def validate_adoption_shape(self) -> CompletionVerifierProfilePreparationRequest:
        if (self.expected_prior_proposal_id is None) != (
            self.expected_prior_profile_fingerprint is None
        ):
            raise ValueError(
                "Prior verifier-profile proposal and fingerprint must be supplied together."
            )
        if self.expected_prior_profile_fingerprint is None and self.adoption is not None:
            raise ValueError("Initial verifier profiles cannot carry adoption evidence.")
        if self.expected_prior_profile_fingerprint is not None:
            if self.expected_prior_profile_fingerprint == self.profile.fingerprint:
                if self.adoption is not None:
                    raise ValueError("Exact verifier-profile reuse cannot carry adoption evidence.")
            elif self.adoption is None:
                raise ValueError("Changed verifier profiles require adoption evidence.")
        if self.adoption is not None and (
            self.adoption.expected_profile_fingerprint != self.expected_prior_profile_fingerprint
            or self.adoption.candidate_profile_fingerprint != self.profile.fingerprint
        ):
            raise ValueError("Verifier-profile adoption evidence conflicts with the transition.")
        return self


class CompletionVerifierProfileRecord(CompletionVerifierProfilePreparationRequest):
    """Immutable durable profile authority prepared before verifier dispatch."""

    request_sha256: str
    prepared_at: datetime

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _sha256_digest(value, "request_sha256")

    @field_validator("prepared_at")
    @classmethod
    def normalize_prepared_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "prepared_at")


class CompletionVerifierProfilePolicyRequest(FrozenWorkContractModel):
    """Application-policy input for one non-equal verifier profile."""

    task_id: str
    proposal_id: str
    attempt_id: str
    expected_profile: CompletionVerifierExecutionProfile
    candidate_profile: CompletionVerifierExecutionProfile
    changed_component_ids: tuple[str, ...]
    intent: ExecutionProfileAdoptionIntent

    @field_validator("task_id", "proposal_id", "attempt_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return validate_work_completion_linked_id(value, info.field_name)

    @field_validator("expected_profile", "candidate_profile", mode="before")
    @classmethod
    def copy_profile(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionVerifierExecutionProfile)

    @field_validator("intent", mode="before")
    @classmethod
    def copy_intent(cls, value: object) -> ExecutionProfileAdoptionIntent:
        if type(value) is not ExecutionProfileAdoptionIntent:
            raise TypeError("intent must be an ExecutionProfileAdoptionIntent.")
        return copy_execution_profile_adoption_intent(value)

    @model_validator(mode="after")
    def validate_changed_components(self) -> CompletionVerifierProfilePolicyRequest:
        expected = changed_completion_verifier_profile_components(
            self.expected_profile,
            self.candidate_profile,
        )
        if self.changed_component_ids != expected:
            raise ValueError("changed_component_ids do not match the supplied verifier profiles.")
        if not expected:
            raise ValueError("Verifier-profile policy requires a non-equal profile.")
        return self


class CompletionVerifierProfilePolicy(ABC):
    """Application-owned authorization for verifier-profile transitions."""

    @property
    @abstractmethod
    def identity(self) -> str:
        """Return a stable, versioned, nonsecret policy identity."""

    @abstractmethod
    async def decide(
        self,
        request: CompletionVerifierProfilePolicyRequest,
    ) -> ExecutionProfilePolicyResult:
        """Authorize or reject one changed verifier profile."""


def _behavior_identity_fingerprint(identity: ExecutionProfileBehaviorIdentity) -> str:
    copied = copy_execution_profile_behavior_identity(identity)
    if copied is None:  # pragma: no cover - excluded by the public type
        raise ValueError("identity is required.")
    return sha256(
        canonical_durable_json_bytes(
            copied.model_dump(mode="json", warnings=False),
            "completion_verifier_profile_component",
        )
    ).hexdigest()


def completion_verifier_profile_fingerprint(
    *,
    verifier: CompletionVerifierRef,
    components: tuple[CompletionVerifierProfileComponentIdentity, ...],
) -> str:
    material = {
        "schema_version": COMPLETION_VERIFIER_PROFILE_SCHEMA_VERSION,
        "verifier": verifier.model_dump(mode="json", warnings=False),
        "components": [item.model_dump(mode="json", warnings=False) for item in components],
    }
    return sha256(
        canonical_durable_json_bytes(material, "completion_verifier_execution_profile")
    ).hexdigest()


def build_completion_verifier_execution_profile(
    *,
    verifier: CompletionVerifierRef,
    adapter_identity: ExecutionProfileBehaviorIdentity,
    component_declarations: tuple[CompletionVerifierProfileComponentDeclaration, ...] = (),
) -> CompletionVerifierExecutionProfile:
    copied_verifier = revalidate_model_input(verifier, CompletionVerifierRef)
    if type(adapter_identity) is not ExecutionProfileBehaviorIdentity:
        raise TypeError("adapter_identity must be an ExecutionProfileBehaviorIdentity.")
    copied_adapter = copy_execution_profile_behavior_identity(adapter_identity)
    if copied_adapter is None:  # pragma: no cover - excluded by the exact type check
        raise ValueError("adapter_identity is required.")
    copied_declarations = tuple(
        CompletionVerifierProfileComponentDeclaration.model_validate(item)
        for item in component_declarations
    )
    component_ids = tuple(item.component_id for item in copied_declarations)
    if component_ids != tuple(sorted(set(component_ids))):
        raise ValueError(
            "component_declarations must contain unique component IDs in canonical order."
        )
    components = tuple(
        sorted(
            (
                CompletionVerifierProfileComponentIdentity(
                    component_id=_ADAPTER_COMPONENT_ID,
                    fingerprint=_behavior_identity_fingerprint(copied_adapter),
                ),
                *(
                    CompletionVerifierProfileComponentIdentity(
                        component_id=item.component_id,
                        fingerprint=_behavior_identity_fingerprint(item.identity),
                    )
                    for item in copied_declarations
                ),
            ),
            key=lambda item: item.component_id,
        )
    )
    fingerprint = completion_verifier_profile_fingerprint(
        verifier=copied_verifier,
        components=components,
    )
    return CompletionVerifierExecutionProfile(
        verifier=copied_verifier,
        components=components,
        fingerprint=fingerprint,
    )


def changed_completion_verifier_profile_components(
    expected: CompletionVerifierExecutionProfile,
    candidate: CompletionVerifierExecutionProfile,
) -> tuple[str, ...]:
    expected_by_id = {item.component_id: item for item in expected.components}
    candidate_by_id = {item.component_id: item for item in candidate.components}
    return tuple(
        sorted(
            component_id
            for component_id in expected_by_id.keys() | candidate_by_id.keys()
            if expected_by_id.get(component_id) != candidate_by_id.get(component_id)
        )
    )


def completion_verifier_profile_preparation_request_sha256(
    value: CompletionVerifierProfilePreparationRequest,
) -> str:
    copied = copy_completion_verifier_profile_preparation_request(value)
    return sha256(
        canonical_durable_json_bytes(
            copied.model_dump(mode="json", warnings=False),
            "completion_verifier_profile_preparation",
        )
    ).hexdigest()


def require_completion_verifier_profile_transition(
    value: CompletionVerifierProfilePreparationRequest,
    prior: CompletionVerifierProfileRecord | None,
) -> None:
    """Validate complete prior-profile and adoption authority at the store seam."""

    request = copy_completion_verifier_profile_preparation_request(value)
    if prior is not None:
        prior = copy_completion_verifier_profile_record(prior)
    expected_proposal_id = None if prior is None else prior.proposal_id
    expected_fingerprint = None if prior is None else prior.profile.fingerprint
    if (
        request.expected_prior_proposal_id != expected_proposal_id
        or request.expected_prior_profile_fingerprint != expected_fingerprint
    ):
        raise WorkCompletionConflict(
            "Completion-verifier profile conflicts with the prior task profile."
        ) from None
    adoption = request.adoption
    if prior is None or prior.profile == request.profile:
        if adoption is not None:
            raise WorkCompletionConflict(
                "Verifier-profile reuse cannot carry adoption evidence."
            ) from None
        return
    changed_components = changed_completion_verifier_profile_components(
        prior.profile,
        request.profile,
    )
    if adoption is None or adoption.changed_component_ids != changed_components:
        raise WorkCompletionConflict(
            "Changed verifier profile requires exact durable adoption evidence."
        ) from None


def copy_completion_verifier_execution_profile(
    value: CompletionVerifierExecutionProfile,
) -> CompletionVerifierExecutionProfile:
    if type(value) is not CompletionVerifierExecutionProfile:
        raise TypeError("value must be a CompletionVerifierExecutionProfile.")
    return CompletionVerifierExecutionProfile(
        schema_version=value.schema_version,
        verifier=CompletionVerifierRef(
            verifier_id=value.verifier.verifier_id,
            version=value.verifier.version,
            kind=value.verifier.kind,
            configuration_fingerprint=value.verifier.configuration_fingerprint,
        ),
        components=tuple(
            CompletionVerifierProfileComponentIdentity(
                component_id=component.component_id,
                strength=component.strength,
                fingerprint=component.fingerprint,
            )
            for component in value.components
        ),
        fingerprint=value.fingerprint,
    )


def _copy_adoption_decision(
    value: CompletionVerifierProfileAdoptionDecision | None,
) -> CompletionVerifierProfileAdoptionDecision | None:
    if value is None:
        return None
    if type(value) is not CompletionVerifierProfileAdoptionDecision:
        raise TypeError("adoption must be a CompletionVerifierProfileAdoptionDecision.")
    actor = copy_resolution_actor(value.requested_by)
    if actor is None:  # pragma: no cover - required by the model
        raise ValueError("requested_by is required.")
    return CompletionVerifierProfileAdoptionDecision(
        expected_profile_fingerprint=value.expected_profile_fingerprint,
        candidate_profile_fingerprint=value.candidate_profile_fingerprint,
        changed_component_ids=tuple(value.changed_component_ids),
        policy_identity=value.policy_identity,
        authority_decision=value.authority_decision,
        idempotency_key=value.idempotency_key,
        requested_by=actor,
        reason=value.reason,
        policy_reason=value.policy_reason,
        request_sha256=value.request_sha256,
    )


def _copy_preparation_fields(
    value: CompletionVerifierProfilePreparationRequest,
) -> dict[str, object]:
    contract = copy_work_contract_ref(value.contract)
    if contract is None:  # pragma: no cover - required by the model
        raise ValueError("contract is required.")
    return {
        "proposal_id": value.proposal_id,
        "task_id": value.task_id,
        "attempt_id": value.attempt_id,
        "attempt_request_sha256": value.attempt_request_sha256,
        "source_execution_profile_fingerprint": value.source_execution_profile_fingerprint,
        "proposal_request_sha256": value.proposal_request_sha256,
        "contract": contract,
        "profile": copy_completion_verifier_execution_profile(value.profile),
        "expected_prior_proposal_id": value.expected_prior_proposal_id,
        "expected_prior_profile_fingerprint": value.expected_prior_profile_fingerprint,
        "adoption": _copy_adoption_decision(value.adoption),
    }


def copy_completion_verifier_profile_preparation_request(
    value: CompletionVerifierProfilePreparationRequest,
) -> CompletionVerifierProfilePreparationRequest:
    if type(value) is not CompletionVerifierProfilePreparationRequest:
        raise TypeError("value must be a CompletionVerifierProfilePreparationRequest.")
    return CompletionVerifierProfilePreparationRequest.model_validate(
        _copy_preparation_fields(value)
    )


def copy_completion_verifier_profile_record(
    value: CompletionVerifierProfileRecord,
) -> CompletionVerifierProfileRecord:
    if type(value) is not CompletionVerifierProfileRecord:
        raise TypeError("value must be a CompletionVerifierProfileRecord.")
    return CompletionVerifierProfileRecord.model_validate(
        {
            **_copy_preparation_fields(value),
            "request_sha256": value.request_sha256,
            "prepared_at": value.prepared_at,
        }
    )


def completion_verifier_profile_record_from_preparation(
    value: CompletionVerifierProfilePreparationRequest,
    *,
    request_sha256: str,
    prepared_at: datetime,
) -> CompletionVerifierProfileRecord:
    """Construct a durable record without a lossy nested-model round trip."""

    copied = copy_completion_verifier_profile_preparation_request(value)
    return CompletionVerifierProfileRecord(
        proposal_id=copied.proposal_id,
        task_id=copied.task_id,
        attempt_id=copied.attempt_id,
        attempt_request_sha256=copied.attempt_request_sha256,
        source_execution_profile_fingerprint=copied.source_execution_profile_fingerprint,
        proposal_request_sha256=copied.proposal_request_sha256,
        contract=copied.contract,
        profile=copied.profile,
        expected_prior_proposal_id=copied.expected_prior_proposal_id,
        expected_prior_profile_fingerprint=copied.expected_prior_profile_fingerprint,
        adoption=copied.adoption,
        request_sha256=request_sha256,
        prepared_at=prepared_at,
    )


def completion_verifier_profile_record_from_document(
    value: object,
) -> CompletionVerifierProfileRecord:
    """Reconstruct one typed profile from a decoded durable JSON document."""

    if type(value) is not dict:
        raise TypeError("Durable completion-verifier profile must be a JSON object.")
    document = dict(value)
    adoption_document = document.get("adoption")
    if adoption_document is not None:
        if type(adoption_document) is not dict:
            raise TypeError("Durable verifier-profile adoption must be a JSON object.")
        adoption_values = dict(adoption_document)
        actor_document = adoption_values.get("requested_by")
        if type(actor_document) is not dict:
            raise TypeError("Durable verifier-profile adoption actor must be a JSON object.")
        adoption_values["requested_by"] = ResolutionActor.model_validate(actor_document)
        document["adoption"] = CompletionVerifierProfileAdoptionDecision.model_validate(
            adoption_values
        )
    return CompletionVerifierProfileRecord.model_validate(document)


def completion_verifier_profile_adoption_request_sha256(
    value: CompletionVerifierProfilePolicyRequest,
    *,
    policy_identity: str,
) -> str:
    policy_identity = require_durable_clean_nonblank(policy_identity, "policy_identity")
    copied = copy_completion_verifier_profile_policy_request(value)
    material = copied.model_dump(mode="json", warnings=False)
    material["policy_identity"] = policy_identity
    return sha256(
        canonical_durable_json_bytes(material, "completion_verifier_profile_adoption")
    ).hexdigest()


def copy_completion_verifier_profile_policy_request(
    value: CompletionVerifierProfilePolicyRequest,
) -> CompletionVerifierProfilePolicyRequest:
    """Detach a policy request without serializing caller-owned model state."""

    if type(value) is not CompletionVerifierProfilePolicyRequest:
        raise TypeError("value must be a CompletionVerifierProfilePolicyRequest.")
    return CompletionVerifierProfilePolicyRequest(
        task_id=value.task_id,
        proposal_id=value.proposal_id,
        attempt_id=value.attempt_id,
        expected_profile=copy_completion_verifier_execution_profile(value.expected_profile),
        candidate_profile=copy_completion_verifier_execution_profile(value.candidate_profile),
        changed_component_ids=tuple(value.changed_component_ids),
        intent=copy_execution_profile_adoption_intent(value.intent),
    )


__all__ = [
    "COMPLETION_VERIFIER_PROFILE_CHANGED_COMPONENT_MAX_ITEMS",
    "COMPLETION_VERIFIER_PROFILE_COMPONENT_MAX_ITEMS",
    "COMPLETION_VERIFIER_PROFILE_SCHEMA_VERSION",
    "CompletionVerifierExecutionProfile",
    "CompletionVerifierProfileAdoptionDecision",
    "CompletionVerifierProfileComponentDeclaration",
    "CompletionVerifierProfileComponentIdentity",
    "CompletionVerifierProfilePolicy",
    "CompletionVerifierProfilePolicyRequest",
    "CompletionVerifierProfilePreparationRequest",
    "CompletionVerifierProfileRecord",
    "build_completion_verifier_execution_profile",
    "changed_completion_verifier_profile_components",
    "completion_verifier_profile_adoption_request_sha256",
    "completion_verifier_profile_fingerprint",
    "completion_verifier_profile_preparation_request_sha256",
    "completion_verifier_profile_record_from_document",
    "completion_verifier_profile_record_from_preparation",
    "copy_completion_verifier_execution_profile",
    "copy_completion_verifier_profile_policy_request",
    "copy_completion_verifier_profile_preparation_request",
    "copy_completion_verifier_profile_record",
    "require_completion_verifier_profile_transition",
]
