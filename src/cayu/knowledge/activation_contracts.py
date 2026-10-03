"""Backend-independent contracts for knowledge activation and governed approval."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_object, require_finite
from cayu._validation import require_durable_clean_nonblank as require_clean_nonblank
from cayu.knowledge.records import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeStatus,
    _copy_entry_chunks,
    _copy_entry_evidence,
    _knowledge_activation_identity,
    _knowledge_entry_id,
    _knowledge_publication_operation_id,
    _next_knowledge_revision,
    _validate_knowledge_revision,
    copy_knowledge_chunk,
    copy_knowledge_entry,
    copy_knowledge_evidence,
)
from cayu.knowledge.scopes import (
    KnowledgeAccessScope,
    _knowledge_access_snapshot,
    _KnowledgeAccessSnapshot,
    copy_knowledge_access_scope,
    knowledge_access_scope_sha256,
)

_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")

MAX_KNOWLEDGE_ACTIVATION_ANNOTATION_BYTES = 16_384
MAX_KNOWLEDGE_ACTIVATION_CHUNKS = 10_000
MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS = 10_000
MAX_KNOWLEDGE_ACTIVATION_EVALUATOR_RESULT_BYTES = 65_536
MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES = 1_048_576
MAX_KNOWLEDGE_ACTIVATION_RECEIPT_BYTES = 1_114_112


class KnowledgeGovernanceMode(StrEnum):
    """Application-selected authority model for one high-level knowledge write."""

    REVIEWED = "reviewed"
    POLICY_AUTOMATIC = "policy_automatic"
    AUTONOMOUS = "autonomous"


class KnowledgeActivationDisposition(StrEnum):
    """Application-policy outcome for one exact activation request."""

    ACTIVATE = "activate"
    ROUTE_TO_REVIEW = "route_to_review"
    REJECT = "reject"


class KnowledgeActivationSource(StrEnum):
    """High-level boundary that requested an activation decision."""

    CURATOR = "curator"
    MODEL_TOOL = "model_tool"
    REVIEW_APPROVAL = "review_approval"


def _knowledge_activation_schema_version(value: object) -> int:
    if type(value) is not int or value != 1:
        raise ValueError("`schema_version` must be the integer 1.")
    return value


def _knowledge_activation_revision(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"`{field_name}` must be an integer.")
    _validate_knowledge_revision(value, field_name)
    return value


class KnowledgeGovernanceConfig(BaseModel):
    """Host-owned mode, policy identity, and execution bound for knowledge authority."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )

    schema_version: Literal[1] = 1
    mode: KnowledgeGovernanceMode = KnowledgeGovernanceMode.REVIEWED
    policy_identity: str | None = None
    policy_version: str | None = None
    policy_timeout_seconds: float = 30.0

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> int:
        return _knowledge_activation_schema_version(value)

    @field_validator("policy_identity", "policy_version", mode="before")
    @classmethod
    def validate_optional_identity(cls, value: object, info) -> str | None:
        if value is None:
            return None
        return _knowledge_activation_identity(value, info.field_name)

    @field_validator("policy_timeout_seconds", mode="before")
    @classmethod
    def validate_timeout(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("`policy_timeout_seconds` must be a number.")
        value = require_finite(float(value), "policy_timeout_seconds")
        if value <= 0.0 or value > 3_600.0:
            raise ValueError("`policy_timeout_seconds` must be between 0 and 3600 seconds.")
        return value

    @model_validator(mode="after")
    def validate_policy_authority(self) -> KnowledgeGovernanceConfig:
        has_identity = self.policy_identity is not None
        has_version = self.policy_version is not None
        if has_identity != has_version:
            raise ValueError(
                "Knowledge governance policy identity and version must be configured together."
            )
        if self.mode is KnowledgeGovernanceMode.REVIEWED and has_identity:
            raise ValueError("Reviewed governance cannot configure an automatic policy.")
        if self.mode is not KnowledgeGovernanceMode.REVIEWED and not has_identity:
            raise ValueError("Automatic and autonomous governance require an explicit policy.")
        return self


class KnowledgeActivationRequest(BaseModel):
    """Copied bounded material presented to one application activation policy."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )

    schema_version: Literal[1] = 1
    operation_id: str
    mode: KnowledgeGovernanceMode
    source: KnowledgeActivationSource
    candidate_entry: KnowledgeEntry
    chunks: tuple[KnowledgeChunk, ...]
    evidence: tuple[KnowledgeEvidence, ...] = ()
    expected_revision: int | None = None
    target_revision: int
    access_scope: KnowledgeAccessScope
    evaluator_identity: str | None = None
    evaluator_result: dict[str, Any] | None = None
    evaluator_decision_sha256: str | None = None
    forbidden_authority_identities: tuple[str, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> int:
        return _knowledge_activation_schema_version(value)

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return _knowledge_publication_operation_id(value)

    @field_validator("candidate_entry", mode="before")
    @classmethod
    def copy_entry(cls, value: object) -> object:
        if type(value) is KnowledgeEntry:
            return copy_knowledge_entry(value)
        return value

    @field_validator("chunks", mode="before")
    @classmethod
    def copy_chunks(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise TypeError("`chunks` must be a list or tuple.")
        if len(value) > MAX_KNOWLEDGE_ACTIVATION_CHUNKS:
            raise ValueError(
                f"`chunks` cannot contain more than {MAX_KNOWLEDGE_ACTIVATION_CHUNKS} records."
            )
        return tuple(
            copy_knowledge_chunk(item) if type(item) is KnowledgeChunk else item for item in value
        )

    @field_validator("evidence", mode="before")
    @classmethod
    def copy_evidence(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise TypeError("`evidence` must be a list or tuple.")
        if len(value) > MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS:
            raise ValueError(
                "`evidence` cannot contain more than "
                f"{MAX_KNOWLEDGE_ACTIVATION_EVIDENCE_RECORDS} records."
            )
        return tuple(
            copy_knowledge_evidence(item) if type(item) is KnowledgeEvidence else item
            for item in value
        )

    @field_validator("expected_revision", mode="before")
    @classmethod
    def validate_expected_revision(cls, value: object) -> int | None:
        if value is None:
            return None
        return _knowledge_activation_revision(value, "expected_revision")

    @field_validator("target_revision", mode="before")
    @classmethod
    def validate_target_revision(cls, value: object) -> int:
        return _knowledge_activation_revision(value, "target_revision")

    @field_validator("access_scope", mode="before")
    @classmethod
    def copy_access_scope(cls, value: object) -> object:
        if type(value) is KnowledgeAccessScope:
            return copy_knowledge_access_scope(value)
        return value

    @field_validator("evaluator_decision_sha256")
    @classmethod
    def validate_sha256(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError(f"`{info.field_name}` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("evaluator_identity", mode="before")
    @classmethod
    def validate_evaluator_identity(cls, value: object) -> str | None:
        if value is None:
            return None
        return _knowledge_activation_identity(value, "evaluator_identity")

    @field_validator("evaluator_result", mode="before")
    @classmethod
    def copy_evaluator_result(cls, value: object) -> object:
        if value is None:
            return None
        copied = copy_durable_json_object(value, "evaluator_result")
        if len(canonical_durable_json_bytes(copied, "evaluator_result")) > (
            MAX_KNOWLEDGE_ACTIVATION_EVALUATOR_RESULT_BYTES
        ):
            raise ValueError("`evaluator_result` exceeds its canonical byte limit.")
        return copied

    @field_validator("forbidden_authority_identities", mode="before")
    @classmethod
    def copy_forbidden_identities(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise TypeError("`forbidden_authority_identities` must be a list or tuple.")
        copied: list[str] = []
        for item in value:
            if type(item) is not str:
                raise TypeError("Forbidden authority identities must be strings.")
            clean = _knowledge_activation_identity(item, "forbidden_authority_identities")
            if clean not in copied:
                copied.append(clean)
        if len(copied) > 16:
            raise ValueError("At most 16 forbidden authority identities may be supplied.")
        return tuple(copied)

    @model_validator(mode="after")
    def validate_exact_material(self) -> KnowledgeActivationRequest:
        expected_target = 1 if self.expected_revision is None else self.expected_revision + 1
        if self.target_revision != expected_target:
            raise ValueError("`target_revision` must follow `expected_revision`.")
        if self.candidate_entry.status is not KnowledgeStatus.PENDING:
            raise ValueError("Activation policy must receive a pending candidate projection.")
        candidate_revision = self.candidate_entry.revision
        if self.source is KnowledgeActivationSource.REVIEW_APPROVAL:
            if self.mode is not KnowledgeGovernanceMode.REVIEWED:
                raise ValueError("Review approval requires reviewed governance.")
            if self.expected_revision is None or candidate_revision != self.expected_revision:
                raise ValueError("Review approval must bind the current pending revision.")
        elif candidate_revision != self.target_revision:
            raise ValueError("Generated activation candidates must bind the target revision.")
        _copy_entry_chunks(self.candidate_entry.id, candidate_revision, list(self.chunks))
        _copy_entry_evidence(
            self.candidate_entry.id,
            candidate_revision,
            list(self.evidence),
            chunks=list(self.chunks),
        )
        evaluator_fields = (
            self.evaluator_identity,
            self.evaluator_result,
            self.evaluator_decision_sha256,
        )
        if any(value is None for value in evaluator_fields) != all(
            value is None for value in evaluator_fields
        ):
            raise ValueError(
                "Evaluator identity, result, and decision fingerprint must appear together."
            )
        if (
            self.evaluator_result is not None
            and self.evaluator_decision_sha256
            != sha256(
                canonical_durable_json_bytes(
                    self.evaluator_result,
                    "knowledge activation evaluator result",
                )
            ).hexdigest()
        ):
            raise ValueError("Evaluator result does not match its decision fingerprint.")
        request_bytes = canonical_durable_json_bytes(
            self.model_dump(mode="json"),
            "knowledge activation request",
        )
        if len(request_bytes) > MAX_KNOWLEDGE_ACTIVATION_REQUEST_BYTES:
            raise ValueError("Knowledge activation request exceeds its canonical byte limit.")
        return self

    @property
    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "knowledge activation request",
            )
        ).hexdigest()

    @property
    def access_scope_sha256(self) -> str:
        """Return the canonical identity of the copied policy-visible scope."""

        return knowledge_access_scope_sha256(self.access_scope)


class KnowledgeActivationDecision(BaseModel):
    """Exact application-owned disposition for one activation request."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )

    schema_version: Literal[1] = 1
    request_sha256: str
    disposition: KnowledgeActivationDisposition
    policy_identity: str
    policy_version: str
    code: str
    annotations: dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> int:
        return _knowledge_activation_schema_version(value)

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("`request_sha256` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("policy_identity", "policy_version", "code", mode="before")
    @classmethod
    def validate_identity(cls, value: object, info) -> str:
        return _knowledge_activation_identity(value, info.field_name)

    @field_validator("annotations", mode="before")
    @classmethod
    def copy_annotations(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "annotations")
        if len(canonical_durable_json_bytes(copied, "annotations")) > (
            MAX_KNOWLEDGE_ACTIVATION_ANNOTATION_BYTES
        ):
            raise ValueError("Activation annotations exceed their canonical byte limit.")
        return copied

    @property
    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "knowledge activation decision",
            )
        ).hexdigest()


class KnowledgeActivationAuthority(BaseModel):
    """Validated request/decision pair handed to the mechanical store boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )

    request: KnowledgeActivationRequest
    decision: KnowledgeActivationDecision

    @field_validator("request", mode="before")
    @classmethod
    def copy_request(cls, value: object) -> object:
        if type(value) is KnowledgeActivationRequest:
            return value.model_copy(deep=True)
        return value

    @field_validator("decision", mode="before")
    @classmethod
    def copy_decision(cls, value: object) -> object:
        if type(value) is KnowledgeActivationDecision:
            return value.model_copy(deep=True)
        return value

    @model_validator(mode="after")
    def validate_binding(self) -> KnowledgeActivationAuthority:
        if self.decision.request_sha256 != self.request.fingerprint:
            raise ValueError("Activation decision does not bind its exact request.")
        if self.decision.policy_identity in self.request.forbidden_authority_identities:
            raise ValueError("A generator, evaluator, or model cannot authorize activation.")
        if (
            self.request.mode is KnowledgeGovernanceMode.REVIEWED
            and self.request.source is not KnowledgeActivationSource.REVIEW_APPROVAL
            and self.decision.disposition is not KnowledgeActivationDisposition.ROUTE_TO_REVIEW
        ):
            raise ValueError("Reviewed governance can only route generated knowledge to review.")
        return self


class KnowledgeActivationReceipt(BaseModel):
    """Immutable store-authored attribution for one governed revision operation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )

    schema_version: Literal[1] = 1
    operation_id: str
    entry_id: str
    entry_revision: int
    expected_revision: int | None
    publication_request_sha256: str
    authority: KnowledgeActivationAuthority
    committed_at: datetime
    replayed: bool = False

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> int:
        return _knowledge_activation_schema_version(value)

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return _knowledge_publication_operation_id(value)

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator("entry_revision", mode="before")
    @classmethod
    def validate_entry_revision(cls, value: object) -> int:
        return _knowledge_activation_revision(value, "entry_revision")

    @field_validator("expected_revision", mode="before")
    @classmethod
    def validate_expected_revision(cls, value: object) -> int | None:
        if value is None:
            return None
        return _knowledge_activation_revision(value, "expected_revision")

    @field_validator("publication_request_sha256")
    @classmethod
    def validate_publication_request_sha256(cls, value: str) -> str:
        if type(value) is not str or _SHA256_HEX_RE.fullmatch(value) is None:
            raise ValueError("`publication_request_sha256` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("committed_at")
    @classmethod
    def validate_committed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`committed_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("replayed", mode="before")
    @classmethod
    def validate_replayed(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("`replayed` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_exact_binding(self) -> KnowledgeActivationReceipt:
        request = self.authority.request
        if (
            self.operation_id != request.operation_id
            or self.entry_id != request.candidate_entry.id
            or self.entry_revision != request.target_revision
            or self.expected_revision != request.expected_revision
        ):
            raise ValueError("Activation receipt does not bind its exact operation and revision.")
        if self.authority.decision.disposition is KnowledgeActivationDisposition.REJECT:
            raise ValueError("Rejected activation decisions cannot have publication receipts.")
        receipt_bytes = canonical_durable_json_bytes(
            self.model_dump(mode="json"),
            "knowledge activation receipt",
        )
        if len(receipt_bytes) > MAX_KNOWLEDGE_ACTIVATION_RECEIPT_BYTES:
            raise ValueError("Knowledge activation receipt exceeds its canonical byte limit.")
        return self


class KnowledgeReviewApproval(BaseModel):
    """Exact activated revision and durable reviewed-approval attribution."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )

    entry: KnowledgeEntry
    receipt: KnowledgeActivationReceipt

    @model_validator(mode="after")
    def validate_binding(self) -> KnowledgeReviewApproval:
        if (
            self.entry.id != self.receipt.entry_id
            or self.entry.revision != self.receipt.entry_revision
            or self.entry.status is not KnowledgeStatus.ACTIVE
            or self.receipt.authority.request.source
            is not KnowledgeActivationSource.REVIEW_APPROVAL
            or self.receipt.authority.decision.disposition
            is not KnowledgeActivationDisposition.ACTIVATE
        ):
            raise ValueError("Reviewed approval does not bind one active review successor.")
        return self


class KnowledgeActivationConflict(RuntimeError):
    """An idempotent activation operation conflicts with durable state."""

    def __init__(self, reason: str) -> None:
        self.reason = require_clean_nonblank(reason, "reason")
        super().__init__("Knowledge activation conflicts with durable state.")


def copy_knowledge_activation_request(
    request: KnowledgeActivationRequest,
) -> KnowledgeActivationRequest:
    if type(request) is not KnowledgeActivationRequest:
        raise TypeError("KnowledgeActivationRequest instances must not be subclasses.")
    return KnowledgeActivationRequest.model_validate(request.model_dump(mode="python"))


def copy_knowledge_activation_decision(
    decision: KnowledgeActivationDecision,
) -> KnowledgeActivationDecision:
    if type(decision) is not KnowledgeActivationDecision:
        raise TypeError("KnowledgeActivationDecision instances must not be subclasses.")
    return KnowledgeActivationDecision.model_validate(decision.model_dump(mode="python"))


def copy_knowledge_activation_authority(
    authority: KnowledgeActivationAuthority,
) -> KnowledgeActivationAuthority:
    if type(authority) is not KnowledgeActivationAuthority:
        raise TypeError("KnowledgeActivationAuthority instances must not be subclasses.")
    return KnowledgeActivationAuthority.model_validate(authority.model_dump(mode="python"))


def copy_knowledge_activation_receipt(
    receipt: KnowledgeActivationReceipt,
    *,
    replayed: bool | None = None,
) -> KnowledgeActivationReceipt:
    if type(receipt) is not KnowledgeActivationReceipt:
        raise TypeError("KnowledgeActivationReceipt instances must not be subclasses.")
    return KnowledgeActivationReceipt(
        schema_version=receipt.schema_version,
        operation_id=receipt.operation_id,
        entry_id=receipt.entry_id,
        entry_revision=receipt.entry_revision,
        expected_revision=receipt.expected_revision,
        publication_request_sha256=receipt.publication_request_sha256,
        authority=copy_knowledge_activation_authority(receipt.authority),
        committed_at=receipt.committed_at,
        replayed=receipt.replayed if replayed is None else replayed,
    )


def _knowledge_activation_receipt_json(receipt: KnowledgeActivationReceipt) -> str:
    """Return the exact canonical receipt document used by durable backends."""

    copied = copy_knowledge_activation_receipt(receipt)
    return canonical_durable_json_bytes(
        copied.model_dump(mode="json"),
        "knowledge activation receipt",
    ).decode("utf-8")


def copy_knowledge_review_approval(approval: KnowledgeReviewApproval) -> KnowledgeReviewApproval:
    if type(approval) is not KnowledgeReviewApproval:
        raise TypeError("KnowledgeReviewApproval instances must not be subclasses.")
    return KnowledgeReviewApproval(
        entry=copy_knowledge_entry(approval.entry),
        receipt=copy_knowledge_activation_receipt(approval.receipt),
    )


def prepare_knowledge_activation_request(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    *,
    evidence: list[KnowledgeEvidence] | None = None,
    access_scope: KnowledgeAccessScope,
    operation_id: str,
    governance_mode: KnowledgeGovernanceMode,
    source: KnowledgeActivationSource,
    expected_revision: int | None = None,
    evaluator_identity: str | None = None,
    evaluator_result: dict[str, Any] | None = None,
    evaluator_decision_sha256: str | None = None,
    forbidden_authority_identities: tuple[str, ...] = (),
) -> KnowledgeActivationRequest:
    """Copy the exact bounded candidate material presented to activation policy."""

    if type(access_scope) is not KnowledgeAccessScope:
        raise TypeError("`access_scope` must be a KnowledgeAccessScope.")
    candidate = copy_knowledge_entry(entry)
    if candidate.status is not KnowledgeStatus.PENDING:
        candidate = candidate.model_copy(update={"status": KnowledgeStatus.PENDING})
    copied_chunks = _copy_entry_chunks(candidate.id, candidate.revision, chunks)
    copied_evidence = _copy_entry_evidence(
        candidate.id,
        candidate.revision,
        evidence or [],
        chunks=copied_chunks,
    )
    target_revision = (
        1 if expected_revision is None else _next_knowledge_revision(expected_revision)
    )
    return KnowledgeActivationRequest(
        operation_id=operation_id,
        mode=governance_mode,
        source=source,
        candidate_entry=candidate,
        chunks=tuple(copied_chunks),
        evidence=tuple(copied_evidence),
        expected_revision=expected_revision,
        target_revision=target_revision,
        access_scope=access_scope,
        evaluator_identity=evaluator_identity,
        evaluator_result=evaluator_result,
        evaluator_decision_sha256=evaluator_decision_sha256,
        forbidden_authority_identities=forbidden_authority_identities,
    )


_MAX_KNOWLEDGE_ACTIVATION_RETIREMENT_BYTES = 1_048_576


_MAX_KNOWLEDGE_ACTIVATION_RETIREMENT_TIME = datetime(
    9999,
    12,
    31,
    23,
    59,
    59,
    999999,
    tzinfo=UTC,
)


class _KnowledgeActivationRetirement(BaseModel):
    """Content-free final authority retained when governed knowledge is pruned."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    entry_id: str
    entry_revision: int
    access_snapshot: _KnowledgeAccessSnapshot
    retired_at: datetime

    @field_validator("entry_id")
    @classmethod
    def validate_entry_id(cls, value: str) -> str:
        return _knowledge_entry_id(value)

    @field_validator("entry_revision")
    @classmethod
    def validate_entry_revision(cls, value: int) -> int:
        _validate_knowledge_revision(value, "entry_revision")
        return value

    @field_validator("access_snapshot", mode="before")
    @classmethod
    def copy_access_snapshot(cls, value: object) -> object:
        if type(value) is _KnowledgeAccessSnapshot:
            return value.model_copy(deep=True)
        return value

    @field_validator("retired_at")
    @classmethod
    def validate_retired_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`retired_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_document_size(self) -> _KnowledgeActivationRetirement:
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge activation retirement",
                )
            )
            > _MAX_KNOWLEDGE_ACTIVATION_RETIREMENT_BYTES
        ):
            raise ValueError(
                "Knowledge activation retirement authority exceeds its canonical byte limit."
            )
        return self


def _knowledge_activation_retirement_json(retirement: _KnowledgeActivationRetirement) -> str:
    if type(retirement) is not _KnowledgeActivationRetirement:
        raise TypeError("retirement must be a _KnowledgeActivationRetirement.")
    return canonical_durable_json_bytes(
        retirement.model_dump(mode="json"),
        "knowledge activation retirement",
    ).decode("utf-8")


def _parse_knowledge_activation_retirement_json(value: str) -> _KnowledgeActivationRetirement:
    if type(value) is not str:
        raise TypeError("Knowledge activation retirement must be JSON text.")
    return _KnowledgeActivationRetirement.model_validate_json(value)


def _knowledge_activation_retirement(
    entry: KnowledgeEntry,
    *,
    retired_at: datetime,
) -> _KnowledgeActivationRetirement:
    return _KnowledgeActivationRetirement(
        entry_id=entry.id,
        entry_revision=entry.revision,
        access_snapshot=_knowledge_access_snapshot(entry),
        retired_at=retired_at,
    )


def _require_knowledge_activation_retirement_capacity(entry: KnowledgeEntry) -> None:
    """Prove one governed successor can always preserve final access authority."""

    _knowledge_activation_retirement(
        entry,
        retired_at=_MAX_KNOWLEDGE_ACTIVATION_RETIREMENT_TIME,
    )
