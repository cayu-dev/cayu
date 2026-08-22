from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from itertools import islice
from typing import Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._clock import normalize_utc_datetime
from cayu._validation import (
    FrozenJsonDict,
    FrozenJsonList,
    JsonUtf8SizeCounter,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    revalidate_model_input,
    revalidate_model_inputs,
)

WORK_CONTRACT_MAX_CRITERIA = 64
WORK_CONTRACT_MAX_CONSTRAINTS = 64
WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS = 128
WORK_CONTRACT_MAX_EVIDENCE_REFERENCES = 256
WORK_CONTRACT_MAX_BYTES = 256 * 1024
WORK_COMPLETION_PROPOSAL_MAX_BYTES = 256 * 1024
WORK_COMPLETION_DECISION_MAX_BYTES = 512 * 1024
# Reserve room for Cayu-owned decision, claim, worker, and verifier authority so
# every accepted adapter outcome remains publishable as a full decision.  Six
# request/verifier identities can each consume 256 UTF-8 bytes, while the
# persisted task identity can consume 2,048 and the attempt/contract identities
# another 256 each. JSON control escaping can expand one byte to six, for a
# 24 KiB worst-case expansion; 32 KiB also covers fingerprints, timestamps,
# field names, versions, and punctuation.
WORK_COMPLETION_VERIFIER_DECISION_MAX_BYTES = WORK_COMPLETION_DECISION_MAX_BYTES - (32 * 1024)
WORK_COMPLETION_APPLICATION_MAX_BYTES = 512 * 1024
WORK_COMPLETION_APPLICATION_MAX_ITEMS = 16_384
WORK_CONTRACT_TASK_MAX_BYTES = 1024 * 1024
WORK_CONTRACT_TASK_MAX_ITEMS = 32_768
# Initial snapshots leave room beneath the complete persisted-task ceiling for
# store-owned lifecycle fields. Two maximally escaped linked identities consume
# less than 24 KiB; 64 KiB also covers timestamps and fixed decision status.
WORK_CONTRACT_TASK_LIFECYCLE_HEADROOM_BYTES = 64 * 1024
WORK_CONTRACT_TASK_LIFECYCLE_HEADROOM_ITEMS = 64
WORK_CONTRACT_TASK_CREATION_MAX_BYTES = (
    WORK_CONTRACT_TASK_MAX_BYTES - WORK_CONTRACT_TASK_LIFECYCLE_HEADROOM_BYTES
)
WORK_CONTRACT_TASK_CREATION_MAX_ITEMS = (
    WORK_CONTRACT_TASK_MAX_ITEMS - WORK_CONTRACT_TASK_LIFECYCLE_HEADROOM_ITEMS
)
WORK_COMPLETION_APPLICATION_RECEIPT_MAX_BYTES = WORK_CONTRACT_TASK_MAX_BYTES + 16 * 1024
WORK_COMPLETION_APPLICATION_RECEIPT_MAX_ITEMS = WORK_CONTRACT_TASK_MAX_ITEMS + 64
WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES = 32
WORK_CONTRACT_TEXT_MAX_BYTES = 16 * 1024
WORK_CONTRACT_IDENTIFIER_MAX_BYTES = 256
WORK_COMPLETION_LINKED_ID_MAX_BYTES = 2048
WORK_EVIDENCE_REFERENCE_ID_MAX_BYTES = 2048
WORK_COMPLETION_IDEMPOTENCY_KEY_MAX_BYTES = 256
WORK_VERIFICATION_LEASE_MAX_SECONDS = 3600

_LOWER_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z", flags=re.ASCII)
_CANONICAL_CODE = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}\Z", flags=re.ASCII)


class WorkContractConflict(ValueError):
    """A stable work-contract identity is already bound to different content."""


class WorkCompletionConflict(ValueError):
    """A stable work-completion identity is already bound to another operation."""


class CompletionVerificationClaimLost(ValueError):
    """A verifier no longer owns the live claim for a completion proposal."""


class TaskCompletionDecisionRequired(ValueError):
    """A contracted task cannot complete without an accepted durable decision."""


class CompletionVerdict(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    NEEDS_REVIEW = "needs_review"


class CompletionRejectionAction(StrEnum):
    CONTINUE = "continue"
    INTERRUPT = "interrupt"


class CompletionVerifierKind(StrEnum):
    DETERMINISTIC = "deterministic"
    PROVIDER = "provider"


class CriterionOutcomeStatus(StrEnum):
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    UNVERIFIABLE = "unverifiable"


class CompletionSatisfactionBasis(StrEnum):
    EVIDENCE = "evidence"
    VERIFIER_ASSERTION = "verifier_assertion"


class FrozenWorkContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )


def require_bounded_work_completion_document(
    value: object,
    field_name: str,
    *,
    max_bytes: int,
    max_items: int,
) -> None:
    """Reject a durable completion representation before mutation or persistence."""

    remaining_items = max_items
    pending = [value]
    while pending:
        current = pending.pop()
        remaining_items -= 1
        if remaining_items < 0:
            raise ValueError(f"{field_name} must contain at most {max_items} JSON values.")
        if type(current) is dict:
            pending.extend(cast("dict[object, object]", current).values())
        elif type(current) is FrozenJsonDict:
            pending.extend(current.values())
        elif type(current) is list:
            pending.extend(cast("list[object]", current))
        elif type(current) is FrozenJsonList:
            pending.extend(current)
    counter = JsonUtf8SizeCounter(max_bytes, canonical_durable_numbers=True)
    # Validated completion models are already nesting-bounded; preserve the
    # canonical durable validator as the final authority for malformed use.
    with suppress(RecursionError):
        counter.value(value)
    if counter.exceeded_limit:
        raise ValueError(f"{field_name} must not exceed {max_bytes} bytes.")
    encoded = canonical_durable_json_bytes(value, field_name)
    if len(encoded) > max_bytes:
        raise ValueError(f"{field_name} must not exceed {max_bytes} bytes.")


def preflight_work_completion_document(
    value: object,
    field_name: str,
    *,
    max_bytes: int,
    max_items: int,
) -> None:
    """Bound plain caller JSON before a defensive copy allocates its full shape."""

    remaining_items = max_items
    pending: list[tuple[object, Iterator[object] | None]] = [(value, None)]
    plain_json = True
    active_container_ids: set[int] = set()
    while pending:
        current, children = pending.pop()
        if children is not None:
            try:
                child = next(children)
            except StopIteration:
                active_container_ids.remove(id(current))
                continue
            pending.append((current, children))
            pending.append((child, None))
            continue
        remaining_items -= 1
        if remaining_items < 0:
            raise ValueError(f"{field_name} must contain at most {max_items} JSON values.")
        if type(current) is dict:
            mapping = cast("dict[object, object]", current)
            container_id = id(current)
            if container_id in active_container_ids:
                # The durable copier owns circular-reference diagnostics. The
                return
            active_container_ids.add(container_id)
            if any(type(key) is not str for key in mapping):
                plain_json = False
            pending.append((current, iter(mapping.values())))
        elif type(current) is FrozenJsonDict:
            mapping = current
            container_id = id(current)
            if container_id in active_container_ids:
                return
            active_container_ids.add(container_id)
            if any(type(key) is not str for key in mapping):
                plain_json = False
            pending.append((current, iter(mapping.values())))
        elif type(current) is list:
            sequence = cast("list[object]", current)
            container_id = id(current)
            if container_id in active_container_ids:
                return
            active_container_ids.add(container_id)
            pending.append((current, iter(sequence)))
        elif type(current) is FrozenJsonList:
            sequence = current
            container_id = id(current)
            if container_id in active_container_ids:
                return
            active_container_ids.add(container_id)
            pending.append((current, iter(sequence)))
        elif type(current) not in {str, int, float, bool, type(None)}:
            # The durable-value validator owns invalid or extension-controlled
            # types and its stable diagnostics. Do not invoke their methods here.
            plain_json = False
    if not plain_json:
        return
    counter = JsonUtf8SizeCounter(max_bytes, canonical_durable_numbers=True)
    try:
        counter.value(value)
    except RecursionError:
        # The durable copier supplies the stable nesting error.
        return
    if counter.exceeded_limit:
        raise ValueError(f"{field_name} must not exceed {max_bytes} bytes.")


def _bounded_identifier(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if _utf8_length_exceeds(value, WORK_CONTRACT_IDENTIFIER_MAX_BYTES):
        raise ValueError(
            f"{field_name} must not exceed {WORK_CONTRACT_IDENTIFIER_MAX_BYTES} UTF-8 bytes."
        )
    return value


def _bounded_reference_id(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if _utf8_length_exceeds(value, WORK_EVIDENCE_REFERENCE_ID_MAX_BYTES):
        raise ValueError(
            f"{field_name} must not exceed {WORK_EVIDENCE_REFERENCE_ID_MAX_BYTES} UTF-8 bytes."
        )
    return value


def validate_work_completion_linked_id(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if _utf8_length_exceeds(value, WORK_COMPLETION_LINKED_ID_MAX_BYTES):
        raise ValueError(
            f"{field_name} must not exceed {WORK_COMPLETION_LINKED_ID_MAX_BYTES} UTF-8 bytes "
            "when bound to a work contract."
        )
    return value


def _bounded_text(value: str, field_name: str) -> str:
    value = require_durable_nonblank(value, field_name)
    if _utf8_length_exceeds(value, WORK_CONTRACT_TEXT_MAX_BYTES):
        raise ValueError(
            f"{field_name} must not exceed {WORK_CONTRACT_TEXT_MAX_BYTES} UTF-8 bytes."
        )
    return value


def _canonical_code(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if _CANONICAL_CODE.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase canonical code using letters, digits, '.', ':', "
            "'_', or '-'."
        )
    return value


def _sha256_digest(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if _LOWER_HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def validate_work_completion_idempotency_key(value: str) -> str:
    value = require_durable_clean_nonblank(value, "idempotency_key")
    if _utf8_length_exceeds(value, WORK_COMPLETION_IDEMPOTENCY_KEY_MAX_BYTES):
        raise ValueError(
            "idempotency_key must not exceed "
            f"{WORK_COMPLETION_IDEMPOTENCY_KEY_MAX_BYTES} UTF-8 bytes."
        )
    return value


def _utf8_length_exceeds(value: str, maximum: int) -> bool:
    """Bound hostile post-construction strings before allocating their full encoding."""

    return len(value) > maximum or len(value.encode("utf-8")) > maximum


def _canonical_unique_strings(
    value: tuple[str, ...],
    *,
    field_name: str,
    maximum: int,
) -> tuple[str, ...]:
    if len(value) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} values.")
    copied = tuple(_bounded_identifier(item, field_name) for item in value)
    if copied != tuple(sorted(set(copied))):
        raise ValueError(f"{field_name} must contain unique values in canonical order.")
    return copied


class WorkContractRef(FrozenWorkContractModel):
    contract_id: str
    version: StrictInt = Field(ge=1)
    fingerprint: str

    @field_validator("contract_id")
    @classmethod
    def validate_contract_id(cls, value: str) -> str:
        return _bounded_identifier(value, "contract_id")

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return _sha256_digest(value, "fingerprint")


class CompletionVerifierRef(FrozenWorkContractModel):
    verifier_id: str
    version: str
    kind: CompletionVerifierKind = CompletionVerifierKind.DETERMINISTIC
    configuration_fingerprint: str

    @field_validator("verifier_id", "version")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identifier(value, info.field_name)

    @field_validator("configuration_fingerprint")
    @classmethod
    def validate_configuration_fingerprint(cls, value: str) -> str:
        return _sha256_digest(value, "configuration_fingerprint")


class CompletionContinuationPolicy(FrozenWorkContractModel):
    rejection_action: CompletionRejectionAction = CompletionRejectionAction.INTERRUPT
    max_attempts: StrictInt = Field(default=3, ge=1, le=100)
    max_repeated_gap_count: StrictInt = Field(default=2, ge=1, le=100)


class WorkEvidenceRequirement(FrozenWorkContractModel):
    requirement_id: str
    kind: str
    description: str

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str) -> str:
        return _bounded_identifier(value, "requirement_id")

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        return _canonical_code(value, "kind")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _bounded_text(value, "description")


class WorkCriterion(FrozenWorkContractModel):
    criterion_id: str
    ordinal: StrictInt = Field(ge=1, le=WORK_CONTRACT_MAX_CRITERIA)
    description: str
    evidence_requirement_ids: tuple[str, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
    )

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, value: str) -> str:
        return _bounded_identifier(value, "criterion_id")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _bounded_text(value, "description")

    @field_validator("evidence_requirement_ids")
    @classmethod
    def validate_evidence_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_unique_strings(
            value,
            field_name="evidence_requirement_ids",
            maximum=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
        )


class WorkConstraint(FrozenWorkContractModel):
    constraint_id: str
    description: str
    evidence_requirement_ids: tuple[str, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
    )

    @field_validator("constraint_id")
    @classmethod
    def validate_constraint_id(cls, value: str) -> str:
        return _bounded_identifier(value, "constraint_id")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        return _bounded_text(value, "description")

    @field_validator("evidence_requirement_ids")
    @classmethod
    def validate_evidence_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_unique_strings(
            value,
            field_name="evidence_requirement_ids",
            maximum=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
        )


class WorkContractDraft(FrozenWorkContractModel):
    contract_id: str
    version: StrictInt = Field(ge=1)
    supersedes: WorkContractRef | None = None
    objective: str
    criteria: tuple[WorkCriterion, ...] = Field(
        min_length=1,
        max_length=WORK_CONTRACT_MAX_CRITERIA,
    )
    constraints: tuple[WorkConstraint, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_CONSTRAINTS,
    )
    evidence_requirements: tuple[WorkEvidenceRequirement, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
    )
    verifier: CompletionVerifierRef
    continuation_policy: CompletionContinuationPolicy = Field(
        default_factory=CompletionContinuationPolicy
    )

    @field_validator("contract_id")
    @classmethod
    def validate_contract_id(cls, value: str) -> str:
        return _bounded_identifier(value, "contract_id")

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, value: str) -> str:
        return _bounded_text(value, "objective")

    @field_validator("supersedes", "verifier", "continuation_policy", mode="before")
    @classmethod
    def copy_nested_model(cls, value: object) -> object:
        return revalidate_model_input(
            value,
            WorkContractRef,
            CompletionVerifierRef,
            CompletionContinuationPolicy,
        )

    @field_validator("criteria", mode="before")
    @classmethod
    def copy_criteria(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            WorkCriterion,
            maximum=WORK_CONTRACT_MAX_CRITERIA,
            field_name="criteria",
        )

    @field_validator("constraints", mode="before")
    @classmethod
    def copy_constraints(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            WorkConstraint,
            maximum=WORK_CONTRACT_MAX_CONSTRAINTS,
            field_name="constraints",
        )

    @field_validator("evidence_requirements", mode="before")
    @classmethod
    def copy_evidence_requirements(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            WorkEvidenceRequirement,
            maximum=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
            field_name="evidence_requirements",
        )

    @model_validator(mode="after")
    def validate_contract_definition(self) -> WorkContractDraft:
        if tuple(item.ordinal for item in self.criteria) != tuple(range(1, len(self.criteria) + 1)):
            raise ValueError("criteria must use unique contiguous ordinals starting at 1.")
        criterion_ids = tuple(item.criterion_id for item in self.criteria)
        if len(set(criterion_ids)) != len(criterion_ids):
            raise ValueError("criteria must use unique criterion_id values.")
        constraint_ids = tuple(item.constraint_id for item in self.constraints)
        if constraint_ids != tuple(sorted(set(constraint_ids))):
            raise ValueError("constraints must use unique IDs in canonical order.")
        requirement_ids = tuple(item.requirement_id for item in self.evidence_requirements)
        if requirement_ids != tuple(sorted(set(requirement_ids))):
            raise ValueError("evidence_requirements must use unique IDs in canonical order.")
        known_requirements = set(requirement_ids)
        for criterion in self.criteria:
            if not set(criterion.evidence_requirement_ids).issubset(known_requirements):
                raise ValueError("criteria reference an unknown evidence requirement.")
        for constraint in self.constraints:
            if not set(constraint.evidence_requirement_ids).issubset(known_requirements):
                raise ValueError("constraints reference an unknown evidence requirement.")
        subject_evidence_assignment_counts = tuple(
            len(subject.evidence_requirement_ids) for subject in (*self.criteria, *self.constraints)
        )
        if any(
            count > WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES
            for count in subject_evidence_assignment_counts
        ):
            raise ValueError(
                "Each work-contract criterion or constraint may assign at most "
                f"{WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES} evidence requirements."
            )
        if sum(subject_evidence_assignment_counts) > WORK_CONTRACT_MAX_EVIDENCE_REFERENCES:
            raise ValueError(
                "Work-contract evidence assignments must not exceed "
                f"{WORK_CONTRACT_MAX_EVIDENCE_REFERENCES} in aggregate."
            )
        assigned_requirements = {
            requirement_id
            for subject in (*self.criteria, *self.constraints)
            for requirement_id in subject.evidence_requirement_ids
        }
        if assigned_requirements != known_requirements:
            raise ValueError(
                "Every evidence requirement must be assigned to a criterion or constraint."
            )
        if self.version == 1:
            if self.supersedes is not None:
                raise ValueError("Work-contract version 1 cannot supersede another version.")
        elif (
            self.supersedes is None
            or self.supersedes.contract_id != self.contract_id
            or self.supersedes.version != self.version - 1
        ):
            raise ValueError(
                "Later work-contract versions must supersede the immediately preceding version."
            )
        encoded = _work_contract_definition_bytes(self)
        if len(encoded) > WORK_CONTRACT_MAX_BYTES:
            raise ValueError(f"Work contract must not exceed {WORK_CONTRACT_MAX_BYTES} bytes.")
        return self


class WorkContract(WorkContractDraft):
    fingerprint: str

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint_format(cls, value: str) -> str:
        return _sha256_digest(value, "fingerprint")

    @model_validator(mode="after")
    def validate_fingerprint_matches_definition(self) -> WorkContract:
        if self.fingerprint != work_contract_fingerprint(self):
            raise ValueError("Work-contract fingerprint conflicts with its canonical definition.")
        return self

    def reference(self) -> WorkContractRef:
        return WorkContractRef(
            contract_id=self.contract_id,
            version=self.version,
            fingerprint=self.fingerprint,
        )


def _work_contract_definition_document(contract: WorkContractDraft) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract_id": contract.contract_id,
        "version": contract.version,
        "supersedes": (
            None
            if contract.supersedes is None
            else contract.supersedes.model_dump(mode="json", warnings=False)
        ),
        "objective": contract.objective,
        "criteria": [item.model_dump(mode="json", warnings=False) for item in contract.criteria],
        "constraints": [
            item.model_dump(mode="json", warnings=False) for item in contract.constraints
        ],
        "evidence_requirements": [
            item.model_dump(mode="json", warnings=False) for item in contract.evidence_requirements
        ],
        "verifier": contract.verifier.model_dump(mode="json", warnings=False),
        "continuation_policy": contract.continuation_policy.model_dump(mode="json", warnings=False),
    }


def _work_contract_definition_bytes(contract: WorkContractDraft) -> bytes:
    return canonical_durable_json_bytes(
        _work_contract_definition_document(contract),
        "work_contract",
    )


def work_contract_fingerprint(contract: WorkContractDraft) -> str:
    if type(contract) not in {WorkContractDraft, WorkContract}:
        raise TypeError("Work-contract fingerprints require a WorkContractDraft or WorkContract.")
    return sha256(_work_contract_definition_bytes(contract)).hexdigest()


def work_contract_from_draft(draft: WorkContractDraft) -> WorkContract:
    draft = copy_work_contract_draft(draft)
    document = _work_contract_definition_document(draft)
    document.pop("schema_version")
    return WorkContract.model_validate(
        {
            **document,
            "fingerprint": sha256(_work_contract_definition_bytes(draft)).hexdigest(),
        }
    )


def copy_work_contract_draft(value: WorkContractDraft) -> WorkContractDraft:
    if type(value) is not WorkContractDraft:
        raise TypeError("Work-contract creation requires a WorkContractDraft.")
    return WorkContractDraft.model_validate(_copy_work_contract_definition(value))


def copy_work_contract(value: WorkContract) -> WorkContract:
    if type(value) is not WorkContract:
        raise TypeError("Published work contracts must be WorkContract instances.")
    return WorkContract.model_validate(
        {
            **_copy_work_contract_definition(value),
            "fingerprint": value.fingerprint,
        }
    )


def _copy_work_contract_definition(value: WorkContractDraft) -> dict[str, object]:
    return {
        "contract_id": value.contract_id,
        "version": value.version,
        "supersedes": _copy_bounded_model_input(value.supersedes, WorkContractRef, 3),
        "objective": value.objective,
        "criteria": _copy_bounded_items(
            value.criteria,
            _copy_work_criterion,
            maximum=WORK_CONTRACT_MAX_CRITERIA,
            field_name="criteria",
        ),
        "constraints": _copy_bounded_items(
            value.constraints,
            _copy_work_constraint,
            maximum=WORK_CONTRACT_MAX_CONSTRAINTS,
            field_name="constraints",
        ),
        "evidence_requirements": _copy_bounded_items(
            value.evidence_requirements,
            _copy_work_evidence_requirement,
            maximum=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
            field_name="evidence_requirements",
        ),
        "verifier": _copy_bounded_model_input(value.verifier, CompletionVerifierRef, 4),
        "continuation_policy": _copy_bounded_model_input(
            value.continuation_policy,
            CompletionContinuationPolicy,
            3,
        ),
    }


def copy_work_contract_ref(value: WorkContractRef | None) -> WorkContractRef | None:
    if value is None:
        return None
    if type(value) is not WorkContractRef:
        raise TypeError("work_contract must be a WorkContractRef.")
    return cast("WorkContractRef", revalidate_model_input(value, WorkContractRef))


class WorkEvidenceReference(FrozenWorkContractModel):
    kind: str
    reference_id: str
    requirement_id: str | None = None
    version: str | None = None
    digest: str | None = None
    available: StrictBool = True
    unavailable_reason: str | None = None

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        return _canonical_code(value, "kind")

    @field_validator("reference_id")
    @classmethod
    def validate_reference_id(cls, value: str) -> str:
        return _bounded_reference_id(value, "reference_id")

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_identifier(value, "requirement_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_identifier(value, "version")

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _sha256_digest(value, "digest")

    @field_validator("unavailable_reason")
    @classmethod
    def validate_unavailable_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _canonical_code(value, "unavailable_reason")

    @model_validator(mode="after")
    def validate_availability(self) -> WorkEvidenceReference:
        if self.available and self.unavailable_reason is not None:
            raise ValueError("Available evidence cannot carry an unavailable reason.")
        if not self.available and self.unavailable_reason is None:
            raise ValueError("Unavailable evidence requires an unavailable reason.")
        return self


def _evidence_reference_key(value: WorkEvidenceReference) -> tuple[object, ...]:
    return (
        value.kind,
        value.reference_id,
        value.requirement_id or "",
        value.version or "",
        value.digest or "",
        value.available,
        value.unavailable_reason or "",
    )


def _evidence_reference_identity(value: WorkEvidenceReference) -> tuple[str, str, str]:
    return (
        value.kind,
        value.reference_id,
        value.version or "",
    )


def _evidence_reference_representation(
    value: WorkEvidenceReference,
) -> tuple[str, bool, str]:
    return (
        value.digest or "",
        value.available,
        value.unavailable_reason or "",
    )


def _validate_evidence_references(
    value: tuple[WorkEvidenceReference, ...],
    *,
    field_name: str,
) -> tuple[WorkEvidenceReference, ...]:
    if len(value) > WORK_CONTRACT_MAX_EVIDENCE_REFERENCES:
        raise ValueError(
            f"{field_name} must contain at most {WORK_CONTRACT_MAX_EVIDENCE_REFERENCES} values."
        )
    keys = tuple(_evidence_reference_key(item) for item in value)
    if keys != tuple(sorted(set(keys))):
        raise ValueError(f"{field_name} must contain unique values in canonical order.")
    return value


class CompletionResultReference(FrozenWorkContractModel):
    kind: str
    reference_id: str
    digest: str

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        return _canonical_code(value, "kind")

    @field_validator("reference_id")
    @classmethod
    def validate_reference_id(cls, value: str) -> str:
        return _bounded_reference_id(value, "reference_id")

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _sha256_digest(value, "digest")


class WorkAttemptCreate(FrozenWorkContractModel):
    attempt_id: str
    task_id: str
    session_id: str
    contract: WorkContractRef
    execution_profile_fingerprint: str
    worker_id: str | None = None

    @field_validator("attempt_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identifier(value, info.field_name)

    @field_validator("task_id", "session_id")
    @classmethod
    def validate_inherited_identity(cls, value: str, info) -> str:
        # These identities are inherited from Task and RunRequest. Use the
        # shared linked-identity bound rather than the work-contract-local
        # 256-byte operation-identity limit.
        return validate_work_completion_linked_id(value, info.field_name)

    @field_validator("worker_id")
    @classmethod
    def validate_worker_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_identifier(value, "worker_id")

    @field_validator("execution_profile_fingerprint")
    @classmethod
    def validate_execution_profile_fingerprint(cls, value: str) -> str:
        return _sha256_digest(value, "execution_profile_fingerprint")

    @field_validator("contract", mode="before")
    @classmethod
    def copy_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)


class WorkAttempt(WorkAttemptCreate):
    ordinal: StrictInt = Field(ge=1)
    request_sha256: str
    started_at: datetime

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _sha256_digest(value, "request_sha256")

    @field_validator("started_at")
    @classmethod
    def normalize_started_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "started_at")


class CompletionProposalCreate(FrozenWorkContractModel):
    proposal_id: str
    attempt_id: str
    result: CompletionResultReference
    evidence_references: tuple[WorkEvidenceReference, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
    )

    @field_validator("proposal_id", "attempt_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identifier(value, info.field_name)

    @field_validator("result", mode="before")
    @classmethod
    def copy_result(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionResultReference)

    @field_validator("evidence_references", mode="before")
    @classmethod
    def copy_evidence_references(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            WorkEvidenceReference,
            maximum=WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
            field_name="evidence_references",
        )

    @field_validator("evidence_references")
    @classmethod
    def validate_evidence_references(
        cls, value: tuple[WorkEvidenceReference, ...]
    ) -> tuple[WorkEvidenceReference, ...]:
        return _validate_evidence_references(value, field_name="evidence_references")

    @model_validator(mode="after")
    def validate_proposal_size(self) -> CompletionProposalCreate:
        encoded = canonical_durable_json_bytes(
            self.model_dump(mode="json", warnings=False),
            "completion_proposal",
        )
        if len(encoded) > WORK_COMPLETION_PROPOSAL_MAX_BYTES:
            raise ValueError(
                f"Completion proposal must not exceed {WORK_COMPLETION_PROPOSAL_MAX_BYTES} bytes."
            )
        return self


class CompletionProposal(CompletionProposalCreate):
    task_id: str
    contract: WorkContractRef
    request_sha256: str
    proposed_at: datetime

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return validate_work_completion_linked_id(value, "task_id")

    @field_validator("contract", mode="before")
    @classmethod
    def copy_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _sha256_digest(value, "request_sha256")

    @field_validator("proposed_at")
    @classmethod
    def normalize_proposed_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "proposed_at")


class CompletionVerificationClaimRequest(FrozenWorkContractModel):
    claim_id: str
    proposal_id: str
    worker_id: str
    execution_owner_id: str | None = None
    verifier: CompletionVerifierRef
    lease_seconds: StrictInt = Field(default=300, ge=1, le=WORK_VERIFICATION_LEASE_MAX_SECONDS)
    execution_timeout_seconds: StrictFloat | None = Field(
        default=None,
        gt=0,
        le=WORK_VERIFICATION_LEASE_MAX_SECONDS,
    )

    @field_validator("claim_id", "proposal_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identifier(value, info.field_name)

    @field_validator("execution_owner_id")
    @classmethod
    def validate_execution_owner_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_identifier(value, "execution_owner_id")

    @field_validator("verifier", mode="before")
    @classmethod
    def copy_verifier(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionVerifierRef)

    @model_validator(mode="after")
    def validate_execution_timeout_within_lease(self) -> CompletionVerificationClaimRequest:
        if (
            self.execution_timeout_seconds is not None
            and self.execution_timeout_seconds >= self.lease_seconds
        ):
            raise ValueError("execution_timeout_seconds must be shorter than lease_seconds.")
        return self


class CompletionVerificationClaim(FrozenWorkContractModel):
    claim_id: str
    proposal_id: str
    worker_id: str
    execution_owner_id: str | None = None
    execution_timeout_seconds: StrictFloat | None = Field(
        default=None,
        gt=0,
        le=WORK_VERIFICATION_LEASE_MAX_SECONDS,
    )
    verifier: CompletionVerifierRef
    attempt_number: StrictInt = Field(ge=1)
    request_sha256: str
    claimed_at: datetime
    lease_expires_at: datetime

    @field_validator("claim_id", "proposal_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identifier(value, info.field_name)

    @field_validator("execution_owner_id")
    @classmethod
    def validate_execution_owner_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_identifier(value, "execution_owner_id")

    @field_validator("verifier", mode="before")
    @classmethod
    def copy_verifier(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionVerifierRef)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _sha256_digest(value, "request_sha256")

    @field_validator("claimed_at", "lease_expires_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime, info) -> datetime:
        return normalize_utc_datetime(value, info.field_name)

    @model_validator(mode="after")
    def validate_lease_window(self) -> CompletionVerificationClaim:
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("Verification claim expiry must follow its claim time.")
        return self


class _CompletionOutcome(FrozenWorkContractModel):
    status: CriterionOutcomeStatus
    reason_code: str
    satisfaction_basis: CompletionSatisfactionBasis | None = None
    evidence_references: tuple[WorkEvidenceReference, ...] = Field(
        default=(),
        max_length=WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES,
    )
    summary: str | None = None

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str) -> str:
        return _canonical_code(value, "reason_code")

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, "summary")

    @field_validator("evidence_references", mode="before")
    @classmethod
    def copy_evidence_references(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            WorkEvidenceReference,
            maximum=WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES,
            field_name="evidence_references",
        )

    @field_validator("evidence_references")
    @classmethod
    def validate_evidence_references(
        cls, value: tuple[WorkEvidenceReference, ...]
    ) -> tuple[WorkEvidenceReference, ...]:
        return _validate_evidence_references(value, field_name="evidence_references")

    @model_validator(mode="after")
    def validate_satisfaction_basis(self) -> _CompletionOutcome:
        if self.status is CriterionOutcomeStatus.SATISFIED:
            if self.satisfaction_basis is None:
                raise ValueError("Satisfied outcomes require an explicit satisfaction basis.")
            if self.satisfaction_basis is CompletionSatisfactionBasis.EVIDENCE and not any(
                reference.available for reference in self.evidence_references
            ):
                raise ValueError("Evidence-backed outcomes require available evidence.")
        elif self.satisfaction_basis is not None:
            raise ValueError("Unresolved outcomes cannot carry a satisfaction basis.")
        return self


class CompletionCriterionOutcome(_CompletionOutcome):
    criterion_id: str

    @field_validator("criterion_id")
    @classmethod
    def validate_criterion_id(cls, value: str) -> str:
        return _bounded_identifier(value, "criterion_id")


class CompletionConstraintOutcome(_CompletionOutcome):
    constraint_id: str

    @field_validator("constraint_id")
    @classmethod
    def validate_constraint_id(cls, value: str) -> str:
        return _bounded_identifier(value, "constraint_id")


class CompletionGap(FrozenWorkContractModel):
    criterion_id: str | None = None
    constraint_id: str | None = None
    code: str
    evidence_requirement_ids: tuple[str, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
    )
    summary: str | None = None

    @field_validator("criterion_id", "constraint_id")
    @classmethod
    def validate_subject_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_identifier(value, info.field_name)

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        return _canonical_code(value, "code")

    @field_validator("evidence_requirement_ids")
    @classmethod
    def validate_evidence_requirement_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_unique_strings(
            value,
            field_name="evidence_requirement_ids",
            maximum=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
        )

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _bounded_text(value, "summary")

    @model_validator(mode="after")
    def validate_exact_subject(self) -> CompletionGap:
        if (self.criterion_id is None) == (self.constraint_id is None):
            raise ValueError("Completion gaps must identify exactly one criterion or constraint.")
        return self

    def subject_key(self) -> tuple[str, str]:
        if self.criterion_id is not None:
            return ("criterion", self.criterion_id)
        if self.constraint_id is None:  # pragma: no cover - enforced by model validation
            raise AssertionError("Completion gap has no subject.")
        return ("constraint", self.constraint_id)


def _validate_completion_decision_outcome(
    *,
    verdict: CompletionVerdict,
    criterion_outcomes: tuple[CompletionCriterionOutcome, ...],
    constraint_outcomes: tuple[CompletionConstraintOutcome, ...],
    gaps: tuple[CompletionGap, ...],
    evidence_references: tuple[WorkEvidenceReference, ...],
) -> None:
    """Validate the verifier-owned portion shared by proposed and durable decisions."""

    criterion_ids = tuple(item.criterion_id for item in criterion_outcomes)
    if len(set(criterion_ids)) != len(criterion_ids):
        raise ValueError("criterion_outcomes must contain each criterion at most once.")
    constraint_ids = tuple(item.constraint_id for item in constraint_outcomes)
    if len(set(constraint_ids)) != len(constraint_ids):
        raise ValueError("constraint_outcomes must contain each constraint at most once.")
    gap_keys = tuple(
        (
            0 if item.criterion_id is not None else 1,
            item.criterion_id or item.constraint_id,
            item.code,
            item.evidence_requirement_ids,
        )
        for item in gaps
    )
    if gap_keys != tuple(sorted(set(gap_keys))):
        raise ValueError("gaps must contain unique values in canonical order.")
    all_outcomes = (*criterion_outcomes, *constraint_outcomes)
    evidence_by_identity: dict[tuple[str, str, str], tuple[str, bool, str]] = {}
    evidence_groups = (
        evidence_references,
        *(outcome.evidence_references for outcome in all_outcomes),
    )
    for evidence_group in evidence_groups:
        for reference in evidence_group:
            identity = _evidence_reference_identity(reference)
            representation = _evidence_reference_representation(reference)
            prior_representation = evidence_by_identity.setdefault(identity, representation)
            if prior_representation != representation:
                raise ValueError(
                    "Completion decision contains conflicting representations of the same evidence."
                )
    all_satisfied = all(item.status is CriterionOutcomeStatus.SATISFIED for item in all_outcomes)
    unresolved_subjects = {
        ("criterion", item.criterion_id)
        for item in criterion_outcomes
        if item.status is not CriterionOutcomeStatus.SATISFIED
    } | {
        ("constraint", item.constraint_id)
        for item in constraint_outcomes
        if item.status is not CriterionOutcomeStatus.SATISFIED
    }
    gap_subjects = {item.subject_key() for item in gaps}
    if verdict is CompletionVerdict.ACCEPTED:
        if not all_satisfied or gaps:
            raise ValueError(
                "Accepted decisions require all criteria and constraints satisfied with no gaps."
            )
    else:
        if all_satisfied or not gaps:
            raise ValueError("Non-accepted decisions require an unresolved outcome and a gap.")
        if gap_subjects != unresolved_subjects:
            raise ValueError(
                "Non-accepted decisions require gaps for exactly the unresolved outcomes."
            )
    evidence_reference_count = len(evidence_references) + sum(
        len(outcome.evidence_references) for outcome in all_outcomes
    )
    if evidence_reference_count > WORK_CONTRACT_MAX_EVIDENCE_REFERENCES:
        raise ValueError("Completion decision contains too many aggregate evidence references.")


class CompletionVerifierDecision(FrozenWorkContractModel):
    """Verifier-owned outcome before Cayu binds durable decision authority."""

    verdict: CompletionVerdict
    criterion_outcomes: tuple[CompletionCriterionOutcome, ...] = Field(
        min_length=1,
        max_length=WORK_CONTRACT_MAX_CRITERIA,
    )
    constraint_outcomes: tuple[CompletionConstraintOutcome, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_CONSTRAINTS,
    )
    gaps: tuple[CompletionGap, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_CRITERIA + WORK_CONTRACT_MAX_CONSTRAINTS,
    )
    evidence_references: tuple[WorkEvidenceReference, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
    )

    @field_validator("criterion_outcomes", mode="before")
    @classmethod
    def copy_criterion_outcomes(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            CompletionCriterionOutcome,
            maximum=WORK_CONTRACT_MAX_CRITERIA,
            field_name="criterion_outcomes",
        )

    @field_validator("constraint_outcomes", mode="before")
    @classmethod
    def copy_constraint_outcomes(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            CompletionConstraintOutcome,
            maximum=WORK_CONTRACT_MAX_CONSTRAINTS,
            field_name="constraint_outcomes",
        )

    @field_validator("gaps", mode="before")
    @classmethod
    def copy_gaps(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            CompletionGap,
            maximum=WORK_CONTRACT_MAX_CRITERIA + WORK_CONTRACT_MAX_CONSTRAINTS,
            field_name="gaps",
        )

    @field_validator("evidence_references", mode="before")
    @classmethod
    def copy_evidence_references(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            WorkEvidenceReference,
            maximum=WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
            field_name="evidence_references",
        )

    @field_validator("evidence_references")
    @classmethod
    def validate_evidence_references(
        cls, value: tuple[WorkEvidenceReference, ...]
    ) -> tuple[WorkEvidenceReference, ...]:
        return _validate_evidence_references(value, field_name="evidence_references")

    @model_validator(mode="after")
    def validate_verdict_shape(self) -> CompletionVerifierDecision:
        _validate_completion_decision_outcome(
            verdict=self.verdict,
            criterion_outcomes=self.criterion_outcomes,
            constraint_outcomes=self.constraint_outcomes,
            gaps=self.gaps,
            evidence_references=self.evidence_references,
        )
        encoded = canonical_durable_json_bytes(
            self.model_dump(mode="json", warnings=False),
            "completion_verifier_decision",
        )
        if len(encoded) > WORK_COMPLETION_VERIFIER_DECISION_MAX_BYTES:
            raise ValueError(
                "Completion verifier decision must not exceed "
                f"{WORK_COMPLETION_VERIFIER_DECISION_MAX_BYTES} bytes."
            )
        return self


class CompletionDecisionCreate(FrozenWorkContractModel):
    decision_id: str
    proposal_id: str
    claim_id: str
    worker_id: str
    verifier: CompletionVerifierRef
    decision_version: Literal[1] = 1
    verdict: CompletionVerdict
    criterion_outcomes: tuple[CompletionCriterionOutcome, ...] = Field(
        min_length=1,
        max_length=WORK_CONTRACT_MAX_CRITERIA,
    )
    constraint_outcomes: tuple[CompletionConstraintOutcome, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_CONSTRAINTS,
    )
    gaps: tuple[CompletionGap, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_CRITERIA + WORK_CONTRACT_MAX_CONSTRAINTS,
    )
    evidence_references: tuple[WorkEvidenceReference, ...] = Field(
        default=(),
        max_length=WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
    )

    @field_validator("decision_id", "proposal_id", "claim_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identifier(value, info.field_name)

    @field_validator("decision_version", mode="before")
    @classmethod
    def validate_decision_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("decision_version must be the integer 1.")
        return value

    @field_validator("verifier", mode="before")
    @classmethod
    def copy_verifier(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionVerifierRef)

    @field_validator("criterion_outcomes", mode="before")
    @classmethod
    def copy_criterion_outcomes(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            CompletionCriterionOutcome,
            maximum=WORK_CONTRACT_MAX_CRITERIA,
            field_name="criterion_outcomes",
        )

    @field_validator("constraint_outcomes", mode="before")
    @classmethod
    def copy_constraint_outcomes(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            CompletionConstraintOutcome,
            maximum=WORK_CONTRACT_MAX_CONSTRAINTS,
            field_name="constraint_outcomes",
        )

    @field_validator("gaps", mode="before")
    @classmethod
    def copy_gaps(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            CompletionGap,
            maximum=WORK_CONTRACT_MAX_CRITERIA + WORK_CONTRACT_MAX_CONSTRAINTS,
            field_name="gaps",
        )

    @field_validator("evidence_references", mode="before")
    @classmethod
    def copy_evidence_references(cls, value: object) -> object:
        return revalidate_model_inputs(
            value,
            WorkEvidenceReference,
            maximum=WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
            field_name="evidence_references",
        )

    @field_validator("evidence_references")
    @classmethod
    def validate_evidence_references(
        cls, value: tuple[WorkEvidenceReference, ...]
    ) -> tuple[WorkEvidenceReference, ...]:
        return _validate_evidence_references(value, field_name="evidence_references")

    @model_validator(mode="after")
    def validate_verdict_shape(self) -> CompletionDecisionCreate:
        _validate_completion_decision_outcome(
            verdict=self.verdict,
            criterion_outcomes=self.criterion_outcomes,
            constraint_outcomes=self.constraint_outcomes,
            gaps=self.gaps,
            evidence_references=self.evidence_references,
        )
        encoded = canonical_durable_json_bytes(
            self.model_dump(mode="json", warnings=False),
            "completion_decision",
        )
        if len(encoded) > WORK_COMPLETION_DECISION_MAX_BYTES:
            raise ValueError(
                f"Completion decision must not exceed {WORK_COMPLETION_DECISION_MAX_BYTES} bytes."
            )
        return self


def validate_completion_decision_contract(
    contract: WorkContract,
    request: CompletionDecisionCreate,
) -> None:
    """Require one decision to cover exactly the frozen contract authority."""

    expected_criteria = tuple(item.criterion_id for item in contract.criteria)
    observed_criteria = tuple(item.criterion_id for item in request.criterion_outcomes)
    if observed_criteria != expected_criteria:
        raise WorkCompletionConflict(
            "Completion decision must cover every contract criterion exactly once in order."
        )
    expected_constraints = tuple(item.constraint_id for item in contract.constraints)
    observed_constraints = tuple(item.constraint_id for item in request.constraint_outcomes)
    if observed_constraints != expected_constraints:
        raise WorkCompletionConflict(
            "Completion decision must cover every contract constraint exactly once in order."
        )

    requirements = {item.requirement_id: item for item in contract.evidence_requirements}
    all_outcomes = (*request.criterion_outcomes, *request.constraint_outcomes)
    all_references = (
        *request.evidence_references,
        *(reference for outcome in all_outcomes for reference in outcome.evidence_references),
    )
    for reference in all_references:
        if reference.requirement_id is None:
            continue
        requirement = requirements.get(reference.requirement_id)
        if requirement is None or requirement.kind != reference.kind:
            raise WorkCompletionConflict(
                "Completion decision evidence conflicts with the frozen contract requirements."
            )

    subject_requirements: dict[tuple[str, str], frozenset[str]] = {
        ("criterion", item.criterion_id): frozenset(item.evidence_requirement_ids)
        for item in contract.criteria
    }
    subject_requirements.update(
        {
            ("constraint", item.constraint_id): frozenset(item.evidence_requirement_ids)
            for item in contract.constraints
        }
    )
    outcomes_with_subjects = tuple(
        (("criterion", item.criterion_id), item) for item in request.criterion_outcomes
    ) + tuple((("constraint", item.constraint_id), item) for item in request.constraint_outcomes)
    for subject, outcome in outcomes_with_subjects:
        required = subject_requirements[subject]
        bound = {
            reference.requirement_id
            for reference in outcome.evidence_references
            if reference.requirement_id is not None
        }
        if not bound.issubset(required):
            raise WorkCompletionConflict(
                "Completion outcome cites evidence assigned to another contract outcome."
            )
        if outcome.status is not CriterionOutcomeStatus.SATISFIED:
            continue
        available = {
            reference.requirement_id
            for reference in outcome.evidence_references
            if reference.available and reference.requirement_id is not None
        }
        if required and (
            outcome.satisfaction_basis is not CompletionSatisfactionBasis.EVIDENCE
            or not required.issubset(available)
        ):
            raise WorkCompletionConflict(
                "Satisfied completion outcomes must carry every available required evidence item."
            )

    for gap in request.gaps:
        required = subject_requirements.get(gap.subject_key())
        if required is None or not set(gap.evidence_requirement_ids).issubset(required):
            raise WorkCompletionConflict(
                "Completion decision contains a gap outside the frozen contract."
            )


class CompletionDecision(CompletionDecisionCreate):
    task_id: str
    attempt_id: str
    contract: WorkContractRef
    claim_authority_sha256: str
    request_sha256: str
    gap_fingerprint: str
    decided_at: datetime

    @field_validator("attempt_id")
    @classmethod
    def validate_parent_identity(cls, value: str, info) -> str:
        return _bounded_identifier(value, info.field_name)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return validate_work_completion_linked_id(value, "task_id")

    @field_validator("contract", mode="before")
    @classmethod
    def copy_contract(cls, value: object) -> object:
        return revalidate_model_input(value, WorkContractRef)

    @field_validator("claim_authority_sha256", "request_sha256", "gap_fingerprint")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_digest(value, info.field_name)

    @field_validator("decided_at")
    @classmethod
    def normalize_decided_at(cls, value: datetime) -> datetime:
        return normalize_utc_datetime(value, "decided_at")


class CompletionDecisionApplicationRequest(FrozenWorkContractModel):
    task_id: str
    decision_id: str
    idempotency_key: str
    result: dict[str, object] | None = None
    result_reference: CompletionResultReference | None = None

    @field_validator("decision_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identifier(value, info.field_name)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return validate_work_completion_linked_id(value, "task_id")

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str) -> str:
        return validate_work_completion_idempotency_key(value)

    @field_validator("result", mode="before")
    @classmethod
    def copy_result(cls, value: object) -> object:
        if value is None:
            return None
        preflight_work_completion_document(
            value,
            "Completion decision application result",
            max_bytes=WORK_COMPLETION_APPLICATION_MAX_BYTES,
            max_items=WORK_COMPLETION_APPLICATION_MAX_ITEMS,
        )
        return copy_durable_json_object(value, "result")

    @field_validator("result_reference", mode="before")
    @classmethod
    def copy_result_reference(cls, value: object) -> object:
        if value is None:
            return None
        return revalidate_model_input(value, CompletionResultReference)

    @model_validator(mode="after")
    def validate_result_binding(self) -> CompletionDecisionApplicationRequest:
        if (self.result is None) != (self.result_reference is None):
            raise ValueError(
                "A task result and its verified result reference must be supplied together."
            )
        require_bounded_work_completion_document(
            self.model_dump(mode="json", warnings=False),
            "Completion decision application",
            max_bytes=WORK_COMPLETION_APPLICATION_MAX_BYTES,
            max_items=WORK_COMPLETION_APPLICATION_MAX_ITEMS,
        )
        if (
            self.result is not None
            and self.result_reference is not None
            and self.result_reference.digest != completion_result_sha256(self.result)
        ):
            raise ValueError("Task result content conflicts with its verified result digest.")
        return self


def _copy_model(value: object, model_type: type[BaseModel], message: str):
    if type(value) is not model_type:
        raise TypeError(message)
    return revalidate_model_input(value, model_type)


def copy_work_attempt_create(value: WorkAttemptCreate) -> WorkAttemptCreate:
    if type(value) is not WorkAttemptCreate:
        raise TypeError("Attempts require a WorkAttemptCreate request.")
    return WorkAttemptCreate.model_validate(_copy_work_attempt_definition(value))


def copy_work_attempt(value: WorkAttempt) -> WorkAttempt:
    if type(value) is not WorkAttempt:
        raise TypeError("Work-attempt lookups must return a WorkAttempt.")
    return WorkAttempt.model_validate(
        {
            **_copy_work_attempt_definition(value),
            "ordinal": value.ordinal,
            "request_sha256": value.request_sha256,
            "started_at": value.started_at,
        }
    )


def _copy_work_attempt_definition(value: WorkAttemptCreate) -> dict[str, object]:
    return {
        "attempt_id": value.attempt_id,
        "task_id": value.task_id,
        "session_id": value.session_id,
        "contract": _copy_bounded_model_input(value.contract, WorkContractRef, 3),
        "execution_profile_fingerprint": value.execution_profile_fingerprint,
        "worker_id": value.worker_id,
    }


def copy_completion_proposal_create(value: CompletionProposalCreate) -> CompletionProposalCreate:
    if type(value) is not CompletionProposalCreate:
        raise TypeError("Completion proposals require a CompletionProposalCreate request.")
    return CompletionProposalCreate.model_validate(
        {
            "proposal_id": value.proposal_id,
            "attempt_id": value.attempt_id,
            "result": _copy_bounded_model_input(value.result, CompletionResultReference, 3),
            "evidence_references": _copy_bounded_items(
                value.evidence_references,
                _copy_work_evidence_reference,
                maximum=WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
                field_name="evidence_references",
            ),
        }
    )


def copy_completion_proposal(value: CompletionProposal) -> CompletionProposal:
    if type(value) is not CompletionProposal:
        raise TypeError("Completion proposal lookups must return a CompletionProposal.")
    return CompletionProposal.model_validate(
        {
            "proposal_id": value.proposal_id,
            "attempt_id": value.attempt_id,
            "result": _copy_bounded_model_input(value.result, CompletionResultReference, 3),
            "evidence_references": _copy_bounded_items(
                value.evidence_references,
                _copy_work_evidence_reference,
                maximum=WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
                field_name="evidence_references",
            ),
            "task_id": value.task_id,
            "contract": _copy_bounded_model_input(value.contract, WorkContractRef, 3),
            "request_sha256": value.request_sha256,
            "proposed_at": value.proposed_at,
        }
    )


def copy_completion_verification_claim_request(
    value: CompletionVerificationClaimRequest,
) -> CompletionVerificationClaimRequest:
    if type(value) is not CompletionVerificationClaimRequest:
        raise TypeError("Verification claims require a CompletionVerificationClaimRequest.")
    return CompletionVerificationClaimRequest.model_validate(
        _copy_completion_verification_claim_definition(value)
    )


def copy_completion_verification_claim(
    value: CompletionVerificationClaim,
) -> CompletionVerificationClaim:
    if type(value) is not CompletionVerificationClaim:
        raise TypeError("Verification-claim lookups must return a CompletionVerificationClaim.")
    return CompletionVerificationClaim.model_validate(
        {
            "claim_id": value.claim_id,
            "proposal_id": value.proposal_id,
            "worker_id": value.worker_id,
            "execution_owner_id": value.execution_owner_id,
            "execution_timeout_seconds": value.execution_timeout_seconds,
            "verifier": _copy_bounded_model_input(value.verifier, CompletionVerifierRef, 4),
            "attempt_number": value.attempt_number,
            "request_sha256": value.request_sha256,
            "claimed_at": value.claimed_at,
            "lease_expires_at": value.lease_expires_at,
        }
    )


def _copy_completion_verification_claim_definition(
    value: CompletionVerificationClaimRequest,
) -> dict[str, object]:
    return {
        "claim_id": value.claim_id,
        "proposal_id": value.proposal_id,
        "worker_id": value.worker_id,
        "execution_owner_id": value.execution_owner_id,
        "execution_timeout_seconds": value.execution_timeout_seconds,
        "verifier": _copy_bounded_model_input(value.verifier, CompletionVerifierRef, 4),
        "lease_seconds": value.lease_seconds,
    }


def copy_completion_verifier_decision(
    value: CompletionVerifierDecision,
) -> CompletionVerifierDecision:
    if type(value) is not CompletionVerifierDecision:
        raise TypeError("Completion verifiers must return a CompletionVerifierDecision.")
    # Do not feed the model instance to the generic copier: a post-construction
    # mutation or model_construct() can otherwise make it recursively copy an
    # arbitrarily large collection before field validators enforce their item
    # bounds.  Each collection is capped before any nested model is rebuilt.
    return CompletionVerifierDecision.model_validate(
        {
            "verdict": value.verdict,
            "criterion_outcomes": _copy_bounded_items(
                value.criterion_outcomes,
                _copy_completion_criterion_outcome,
                maximum=WORK_CONTRACT_MAX_CRITERIA,
                field_name="criterion_outcomes",
            ),
            "constraint_outcomes": _copy_bounded_items(
                value.constraint_outcomes,
                _copy_completion_constraint_outcome,
                maximum=WORK_CONTRACT_MAX_CONSTRAINTS,
                field_name="constraint_outcomes",
            ),
            "gaps": _copy_bounded_items(
                value.gaps,
                _copy_completion_gap,
                maximum=WORK_CONTRACT_MAX_CRITERIA + WORK_CONTRACT_MAX_CONSTRAINTS,
                field_name="gaps",
            ),
            "evidence_references": _copy_bounded_items(
                value.evidence_references,
                _copy_work_evidence_reference,
                maximum=WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
                field_name="evidence_references",
            ),
        }
    )


def _copy_bounded_items(
    value: object,
    copier: Callable[[object], object],
    *,
    maximum: int,
    field_name: str,
) -> object:
    if (
        value is None
        or isinstance(value, (str, bytes, bytearray, Mapping, BaseModel))
        or not isinstance(value, Iterable)
    ):
        return value
    items = tuple(islice(value, maximum + 1))
    if len(items) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} values.")
    return tuple(copier(item) for item in items)


def _copy_bounded_model_input(
    value: object,
    model_type: type[BaseModel],
    maximum_fields: int,
) -> object:
    if isinstance(value, model_type):
        return revalidate_model_input(value, model_type)
    if not isinstance(value, Mapping):
        return value
    mapping = cast("Mapping[object, object]", value)
    keys = tuple(islice(mapping, maximum_fields + 1))
    if len(keys) > maximum_fields:
        raise ValueError(
            f"{model_type.__name__} input must contain at most {maximum_fields} fields."
        )
    return {key: mapping[key] for key in keys}


def _copy_completion_criterion_outcome(value: object) -> object:
    if not isinstance(value, CompletionCriterionOutcome):
        return _copy_bounded_model_input(value, CompletionCriterionOutcome, 6)
    return CompletionCriterionOutcome.model_validate(
        {
            "criterion_id": value.criterion_id,
            "status": value.status,
            "reason_code": value.reason_code,
            "satisfaction_basis": value.satisfaction_basis,
            "evidence_references": _copy_bounded_items(
                value.evidence_references,
                _copy_work_evidence_reference,
                maximum=WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES,
                field_name="evidence_references",
            ),
            "summary": value.summary,
        }
    )


def _copy_completion_constraint_outcome(value: object) -> object:
    if not isinstance(value, CompletionConstraintOutcome):
        return _copy_bounded_model_input(value, CompletionConstraintOutcome, 6)
    return CompletionConstraintOutcome.model_validate(
        {
            "constraint_id": value.constraint_id,
            "status": value.status,
            "reason_code": value.reason_code,
            "satisfaction_basis": value.satisfaction_basis,
            "evidence_references": _copy_bounded_items(
                value.evidence_references,
                _copy_work_evidence_reference,
                maximum=WORK_COMPLETION_OUTCOME_MAX_EVIDENCE_REFERENCES,
                field_name="evidence_references",
            ),
            "summary": value.summary,
        }
    )


def _copy_completion_gap(value: object) -> object:
    if not isinstance(value, CompletionGap):
        return _copy_bounded_model_input(value, CompletionGap, 5)
    return CompletionGap.model_validate(
        {
            "criterion_id": value.criterion_id,
            "constraint_id": value.constraint_id,
            "code": value.code,
            "evidence_requirement_ids": _copy_bounded_items(
                value.evidence_requirement_ids,
                lambda item: item,
                maximum=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
                field_name="evidence_requirement_ids",
            ),
            "summary": value.summary,
        }
    )


def copy_completion_decision_create(value: CompletionDecisionCreate) -> CompletionDecisionCreate:
    if type(value) is not CompletionDecisionCreate:
        raise TypeError("Completion decisions require a CompletionDecisionCreate request.")
    return CompletionDecisionCreate.model_validate(_copy_completion_decision_definition(value))


def copy_completion_decision(value: CompletionDecision) -> CompletionDecision:
    if type(value) is not CompletionDecision:
        raise TypeError("Completion decision lookups must return a CompletionDecision.")
    return CompletionDecision.model_validate(
        {
            **_copy_completion_decision_definition(value),
            "task_id": value.task_id,
            "attempt_id": value.attempt_id,
            "contract": _copy_bounded_model_input(value.contract, WorkContractRef, 3),
            "claim_authority_sha256": value.claim_authority_sha256,
            "request_sha256": value.request_sha256,
            "gap_fingerprint": value.gap_fingerprint,
            "decided_at": value.decided_at,
        }
    )


def _copy_completion_decision_definition(
    value: CompletionDecisionCreate,
) -> dict[str, object]:
    return {
        "decision_id": value.decision_id,
        "proposal_id": value.proposal_id,
        "claim_id": value.claim_id,
        "worker_id": value.worker_id,
        "verifier": _copy_bounded_model_input(value.verifier, CompletionVerifierRef, 4),
        "decision_version": value.decision_version,
        "verdict": value.verdict,
        "criterion_outcomes": _copy_bounded_items(
            value.criterion_outcomes,
            _copy_completion_criterion_outcome,
            maximum=WORK_CONTRACT_MAX_CRITERIA,
            field_name="criterion_outcomes",
        ),
        "constraint_outcomes": _copy_bounded_items(
            value.constraint_outcomes,
            _copy_completion_constraint_outcome,
            maximum=WORK_CONTRACT_MAX_CONSTRAINTS,
            field_name="constraint_outcomes",
        ),
        "gaps": _copy_bounded_items(
            value.gaps,
            _copy_completion_gap,
            maximum=WORK_CONTRACT_MAX_CRITERIA + WORK_CONTRACT_MAX_CONSTRAINTS,
            field_name="gaps",
        ),
        "evidence_references": _copy_bounded_items(
            value.evidence_references,
            _copy_work_evidence_reference,
            maximum=WORK_CONTRACT_MAX_EVIDENCE_REFERENCES,
            field_name="evidence_references",
        ),
    }


def _copy_work_criterion(value: object) -> object:
    if not isinstance(value, WorkCriterion):
        return _copy_bounded_model_input(value, WorkCriterion, 4)
    return WorkCriterion.model_validate(
        {
            "criterion_id": value.criterion_id,
            "ordinal": value.ordinal,
            "description": value.description,
            "evidence_requirement_ids": _copy_bounded_items(
                value.evidence_requirement_ids,
                lambda item: item,
                maximum=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
                field_name="evidence_requirement_ids",
            ),
        }
    )


def _copy_work_constraint(value: object) -> object:
    if not isinstance(value, WorkConstraint):
        return _copy_bounded_model_input(value, WorkConstraint, 3)
    return WorkConstraint.model_validate(
        {
            "constraint_id": value.constraint_id,
            "description": value.description,
            "evidence_requirement_ids": _copy_bounded_items(
                value.evidence_requirement_ids,
                lambda item: item,
                maximum=WORK_CONTRACT_MAX_EVIDENCE_REQUIREMENTS,
                field_name="evidence_requirement_ids",
            ),
        }
    )


def _copy_work_evidence_requirement(value: object) -> object:
    if not isinstance(value, WorkEvidenceRequirement):
        return _copy_bounded_model_input(value, WorkEvidenceRequirement, 3)
    return WorkEvidenceRequirement(
        requirement_id=value.requirement_id,
        kind=value.kind,
        description=value.description,
    )


def _copy_work_evidence_reference(value: object) -> object:
    return _copy_bounded_model_input(value, WorkEvidenceReference, 7)


def copy_completion_decision_application_request(
    value: CompletionDecisionApplicationRequest,
) -> CompletionDecisionApplicationRequest:
    if type(value) is CompletionDecisionApplicationRequest and (
        value.result is not None and type(value.result) is not dict
    ):
        del value
        raise TypeError("Completion decision application result must be a plain dict.")
    if type(value) is CompletionDecisionApplicationRequest and (
        value.result_reference is not None
        and type(value.result_reference) is not CompletionResultReference
    ):
        del value
        raise TypeError(
            "Completion decision application result_reference must be a CompletionResultReference."
        )
    if type(value) is CompletionDecisionApplicationRequest and value.result is not None:
        preflight_work_completion_document(
            value.result,
            "Completion decision application result",
            max_bytes=WORK_COMPLETION_APPLICATION_MAX_BYTES,
            max_items=WORK_COMPLETION_APPLICATION_MAX_ITEMS,
        )
    return cast(
        "CompletionDecisionApplicationRequest",
        _copy_model(
            value,
            CompletionDecisionApplicationRequest,
            "Decision application requires a CompletionDecisionApplicationRequest.",
        ),
    )


def _request_sha256(value: BaseModel, field_name: str) -> str:
    return sha256(
        canonical_durable_json_bytes(
            value.model_dump(mode="json", warnings=False),
            field_name,
        )
    ).hexdigest()


def completion_result_sha256(value: dict[str, object]) -> str:
    copied = copy_durable_json_object(value, "result")
    return sha256(canonical_durable_json_bytes(copied, "completion_result")).hexdigest()


def work_attempt_request_sha256(value: WorkAttemptCreate) -> str:
    return _request_sha256(copy_work_attempt_create(value), "work_attempt")


def completion_proposal_request_sha256(value: CompletionProposalCreate) -> str:
    return _request_sha256(copy_completion_proposal_create(value), "completion_proposal")


def completion_verification_claim_request_sha256(
    value: CompletionVerificationClaimRequest,
) -> str:
    copied = copy_completion_verification_claim_request(value)
    material = copied.model_dump(mode="json", warnings=False)
    # Earlier claims predate runtime execution-owner generations and bounded
    # adapter execution. Preserve their exact canonical request material so
    # durable custom-store claims and completed decisions remain replayable.
    if copied.execution_owner_id is None:
        material.pop("execution_owner_id", None)
    if copied.execution_timeout_seconds is None:
        material.pop("execution_timeout_seconds", None)
    return sha256(
        canonical_durable_json_bytes(material, "completion_verification_claim")
    ).hexdigest()


def completion_verification_claim_authority_sha256(
    value: CompletionVerificationClaim,
) -> str:
    """Bind a decision to the complete final durable verification claim."""

    copied = copy_completion_verification_claim(value)
    return sha256(
        canonical_durable_json_bytes(
            copied.model_dump(mode="json", warnings=False),
            "completion_verification_claim_authority",
        )
    ).hexdigest()


def completion_decision_request_sha256(value: CompletionDecisionCreate) -> str:
    return _request_sha256(copy_completion_decision_create(value), "completion_decision")


def completion_decision_application_request_sha256(
    value: CompletionDecisionApplicationRequest,
) -> str:
    return _request_sha256(
        copy_completion_decision_application_request(value),
        "completion_decision_application",
    )


def completion_gap_fingerprint(value: CompletionDecisionCreate) -> str:
    value = copy_completion_decision_create(value)
    material = {
        "schema_version": 3,
        "criterion_outcomes": [
            {
                "criterion_id": outcome.criterion_id,
                "status": outcome.status.value,
                "reason_code": outcome.reason_code,
            }
            for outcome in value.criterion_outcomes
            if outcome.status is not CriterionOutcomeStatus.SATISFIED
        ],
        "constraint_outcomes": [
            {
                "constraint_id": outcome.constraint_id,
                "status": outcome.status.value,
                "reason_code": outcome.reason_code,
            }
            for outcome in value.constraint_outcomes
            if outcome.status is not CriterionOutcomeStatus.SATISFIED
        ],
        "gaps": [
            {
                "criterion_id": gap.criterion_id,
                "constraint_id": gap.constraint_id,
                "code": gap.code,
                "evidence_requirement_ids": list(gap.evidence_requirement_ids),
            }
            for gap in value.gaps
        ],
    }
    return sha256(canonical_durable_json_bytes(material, "completion_gaps")).hexdigest()


__all__ = [
    "CompletionConstraintOutcome",
    "CompletionContinuationPolicy",
    "CompletionCriterionOutcome",
    "CompletionDecision",
    "CompletionDecisionApplicationRequest",
    "CompletionDecisionCreate",
    "CompletionGap",
    "CompletionProposal",
    "CompletionProposalCreate",
    "CompletionRejectionAction",
    "CompletionResultReference",
    "CompletionSatisfactionBasis",
    "CompletionVerdict",
    "CompletionVerificationClaim",
    "CompletionVerificationClaimLost",
    "CompletionVerificationClaimRequest",
    "CompletionVerifierDecision",
    "CompletionVerifierKind",
    "CompletionVerifierRef",
    "CriterionOutcomeStatus",
    "TaskCompletionDecisionRequired",
    "WorkAttempt",
    "WorkAttemptCreate",
    "WorkCompletionConflict",
    "WorkConstraint",
    "WorkContract",
    "WorkContractConflict",
    "WorkContractDraft",
    "WorkContractRef",
    "WorkCriterion",
    "WorkEvidenceReference",
    "WorkEvidenceRequirement",
    "completion_decision_application_request_sha256",
    "completion_decision_request_sha256",
    "completion_gap_fingerprint",
    "completion_proposal_request_sha256",
    "completion_result_sha256",
    "completion_verification_claim_authority_sha256",
    "completion_verification_claim_request_sha256",
    "copy_completion_decision",
    "copy_completion_decision_application_request",
    "copy_completion_decision_create",
    "copy_completion_proposal",
    "copy_completion_proposal_create",
    "copy_completion_verification_claim",
    "copy_completion_verification_claim_request",
    "copy_completion_verifier_decision",
    "copy_work_attempt",
    "copy_work_attempt_create",
    "copy_work_contract",
    "copy_work_contract_draft",
    "copy_work_contract_ref",
    "validate_completion_decision_contract",
    "validate_work_completion_idempotency_key",
    "validate_work_completion_linked_id",
    "work_attempt_request_sha256",
    "work_contract_fingerprint",
    "work_contract_from_draft",
]
