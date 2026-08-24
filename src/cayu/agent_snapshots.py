"""Portable logical snapshots for reproducible stateful agent evaluation.

An :class:`AgentSnapshot` is a content-addressed manifest of exact logical
identities.  It is intentionally not a database export, filesystem copy, live
process checkpoint, provider continuation, or activation request.  Component
owners retain the authority and implementation needed to capture, verify, and
materialize the references they publish here.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank

if TYPE_CHECKING:
    from cayu.evals.models import Trajectory
    from cayu.runtime.execution_profiles import ExecutionProfileIdentity
    from cayu.runtime.manifest import AppManifest
    from cayu.workspaces.revisions import WorkspaceRevisionObservation

AGENT_SNAPSHOT_SCHEMA_VERSION = 2
AGENT_SNAPSHOT_MAX_COMPONENTS = 32
AGENT_SNAPSHOT_MAX_LIMITATIONS = 64
AGENT_SNAPSHOT_MAX_BYTES = 1024 * 1024
AGENT_SNAPSHOT_RECORD_TYPE = "cayu.agent-snapshot"
AGENT_SNAPSHOT_TRIAL_METADATA_KEY = "agent_snapshot_trial"

_SHA256_CHARS = frozenset("0123456789abcdef")
_SAFE_SOURCE_REF_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:._-"
)
_CONSISTENCY_RANK: dict[AgentSnapshotConsistency, int] = {}


def _clean(value: str, field_name: str, *, max_chars: int = 512) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value) > max_chars:
        raise ValueError(f"{field_name} must be at most {max_chars} characters.")
    return value


def _sha256_hex(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _content_sha256(value: object, field_name: str) -> str:
    return sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _ordered_unique_text(
    value: object,
    field_name: str,
    *,
    max_items: int = AGENT_SNAPSHOT_MAX_LIMITATIONS,
) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field_name} must be an ordered array.")
    copied_items: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str:
            raise ValueError(f"{field_name}[{index}] must be a string.")
        copied_items.append(_clean(item, f"{field_name}[{index}]", max_chars=256))
    copied = tuple(copied_items)
    if len(copied) > max_items:
        raise ValueError(f"{field_name} exceeds its item limit.")
    if copied != tuple(sorted(set(copied))):
        raise ValueError(f"{field_name} must be unique and sorted.")
    return copied


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


class _SnapshotModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )


class AgentSnapshotComponentKind(StrEnum):
    BODY = "body"
    EXECUTION_PROFILE = "execution_profile"
    MEMORY = "memory"
    SESSION = "session"
    WORKSPACE = "workspace"
    ENVIRONMENT = "environment"
    ARTIFACTS = "artifacts"
    POLICIES = "policies"


class AgentSnapshotConsistency(StrEnum):
    TRANSACTIONAL = "transactional"
    FRONTIER_CONSISTENT = "frontier_consistent"
    BEST_EFFORT = "best_effort"
    INCONSISTENT = "inconsistent"


_CONSISTENCY_RANK.update(
    {
        AgentSnapshotConsistency.INCONSISTENT: 0,
        AgentSnapshotConsistency.BEST_EFFORT: 1,
        AgentSnapshotConsistency.FRONTIER_CONSISTENT: 2,
        AgentSnapshotConsistency.TRANSACTIONAL: 3,
    }
)


class AgentSnapshotCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class AgentSnapshotRedaction(StrEnum):
    NONE = "none"
    BOUNDED_PROJECTION = "bounded_projection"
    REDACTED = "redacted"
    TRUNCATED = "truncated"


class AgentSnapshotMaterializationCapability(StrEnum):
    REFERENCE_ONLY = "reference_only"
    REPLAYABLE = "replayable"
    RESTORABLE = "restorable"
    UNAVAILABLE = "unavailable"


class AgentSnapshotTrialStateMode(StrEnum):
    RESET_EACH_TRIAL = "reset_each_trial"
    ACCUMULATE_WITHIN_CANDIDATE = "accumulate_within_candidate"


class AgentSnapshotOverlayKind(StrEnum):
    MEMORY = "memory"
    WORKSPACE = "workspace"


class AgentSnapshotLearningDisposition(StrEnum):
    ISOLATED = "isolated"
    QUARANTINED = "quarantined"
    READ_ONLY = "read_only"
    EXPLICITLY_UNDER_EVALUATION = "explicitly_under_evaluation"


class AgentSnapshotTerminalDisposition(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    OUTCOME_UNKNOWN = "outcome_unknown"


class AgentSnapshotLogicalRef(_SnapshotModel):
    """Exact logical identity with an optional bounded local drill-down alias."""

    fingerprint: StrictStr
    revision: StrictStr | None = Field(default=None, max_length=512)
    frontier: StrictStr | None = Field(default=None, max_length=512)
    scope_fingerprint: StrictStr | None = None
    source_ref: StrictStr | None = Field(default=None, max_length=512)

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("scope_fingerprint")
    @classmethod
    def validate_scope_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("revision", "frontier")
    @classmethod
    def validate_optional_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean(value, info.field_name)

    @field_validator("source_ref")
    @classmethod
    def validate_safe_source_ref(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = _clean(value, "source_ref")
        if not value.startswith("cayu-ref:"):
            raise ValueError("source_ref must be a bounded cayu-ref alias.")
        if value == "cayu-ref:":
            raise ValueError("source_ref must identify a bounded cayu-ref alias.")
        if ".." in value or any(character not in _SAFE_SOURCE_REF_CHARS for character in value):
            raise ValueError("source_ref cannot contain a pathname, URL authority, or query.")
        return value

    @model_validator(mode="after")
    def validate_exact_identity(self) -> AgentSnapshotLogicalRef:
        if self.revision is None and self.frontier is None:
            raise ValueError("A logical reference requires an exact revision or frontier.")
        return self

    def identity_material(self) -> dict[str, object]:
        """Return relocation-stable fingerprint input.

        ``source_ref`` is intentionally excluded: a physical/package alias may
        change while the same logical content and frontiers remain equivalent.
        """

        return {
            "fingerprint": self.fingerprint,
            "revision": self.revision,
            "frontier": self.frontier,
            "scope_fingerprint": self.scope_fingerprint,
        }


class AgentSnapshotSubject(_SnapshotModel):
    agent_id: StrictStr = Field(max_length=256)
    application_id: StrictStr = Field(max_length=256)
    project_id: StrictStr = Field(max_length=256)
    body_release: AgentSnapshotLogicalRef

    @field_validator("agent_id", "application_id", "project_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)


class AgentSnapshotExecutionProfileComponent(_SnapshotModel):
    name: StrictStr = Field(max_length=256)
    fingerprint: StrictStr | None = None
    availability: Literal["available", "unavailable"]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @model_validator(mode="after")
    def validate_availability(self) -> AgentSnapshotExecutionProfileComponent:
        if (self.availability == "available") != (self.fingerprint is not None):
            raise ValueError("Execution-profile availability contradicts its fingerprint.")
        return self


class AgentSnapshotExecutionProfileRef(_SnapshotModel):
    schema_version: StrictInt = Field(ge=1)
    fingerprint: StrictStr
    components: tuple[AgentSnapshotExecutionProfileComponent, ...]

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("components", mode="before")
    @classmethod
    def validate_components_are_ordered(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("components must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_component_order(self) -> AgentSnapshotExecutionProfileRef:
        names = tuple(component.name for component in self.components)
        if not names or names != tuple(sorted(set(names))):
            raise ValueError("Execution-profile components must be nonempty, unique, and sorted.")
        return self


class MemoryStateRef(_SnapshotModel):
    """Typed identity of Cayu knowledge and memory at one bounded frontier."""

    schema_version: Literal[1] = 1
    fingerprint: StrictStr
    knowledge: AgentSnapshotLogicalRef | None = None
    transcript_evidence: AgentSnapshotLogicalRef | None = None
    artifact_evidence: AgentSnapshotLogicalRef | None = None
    work_context: AgentSnapshotLogicalRef | None = None
    recall_policy: AgentSnapshotLogicalRef | None = None
    admission_policy: AgentSnapshotLogicalRef | None = None
    context_projection_policy: AgentSnapshotLogicalRef | None = None
    interaction_focus: AgentSnapshotLogicalRef | None = None
    recall_receipts: AgentSnapshotLogicalRef | None = None
    context_exposures: AgentSnapshotLogicalRef | None = None
    index_readiness: AgentSnapshotLogicalRef | None = None
    learning_disposition: AgentSnapshotLearningDisposition
    limitations: tuple[StrictStr, ...] = ()

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("limitations", mode="before")
    @classmethod
    def validate_limitations(cls, value: object) -> tuple[str, ...]:
        return _ordered_unique_text(value, "limitations")

    @model_validator(mode="after")
    def validate_fingerprint_matches(self) -> MemoryStateRef:
        if not any(
            getattr(self, field_name) is not None
            for field_name in (
                "knowledge",
                "transcript_evidence",
                "artifact_evidence",
                "work_context",
                "recall_policy",
                "admission_policy",
                "context_projection_policy",
                "interaction_focus",
                "recall_receipts",
                "context_exposures",
                "index_readiness",
            )
        ):
            raise ValueError("MemoryStateRef requires at least one declared memory component.")
        if self.fingerprint != _content_sha256(self._identity_material(), "memory_state"):
            raise ValueError("MemoryStateRef fingerprint does not match its logical frontiers.")
        return self

    def _identity_material(self) -> dict[str, object]:
        material: dict[str, object] = {
            "schema_version": self.schema_version,
            "learning_disposition": self.learning_disposition.value,
            "limitations": list(self.limitations),
        }
        for field_name in (
            "knowledge",
            "transcript_evidence",
            "artifact_evidence",
            "work_context",
            "recall_policy",
            "admission_policy",
            "context_projection_policy",
            "interaction_focus",
            "recall_receipts",
            "context_exposures",
            "index_readiness",
        ):
            reference = cast("AgentSnapshotLogicalRef | None", getattr(self, field_name))
            material[field_name] = None if reference is None else reference.identity_material()
        return material

    def _references(self) -> tuple[AgentSnapshotLogicalRef, ...]:
        return tuple(
            reference
            for field_name in (
                "knowledge",
                "transcript_evidence",
                "artifact_evidence",
                "work_context",
                "recall_policy",
                "admission_policy",
                "context_projection_policy",
                "interaction_focus",
                "recall_receipts",
                "context_exposures",
                "index_readiness",
            )
            if (reference := cast("AgentSnapshotLogicalRef | None", getattr(self, field_name)))
            is not None
        )

    @classmethod
    def create(
        cls,
        *,
        knowledge: AgentSnapshotLogicalRef | None = None,
        transcript_evidence: AgentSnapshotLogicalRef | None = None,
        artifact_evidence: AgentSnapshotLogicalRef | None = None,
        work_context: AgentSnapshotLogicalRef | None = None,
        recall_policy: AgentSnapshotLogicalRef | None = None,
        admission_policy: AgentSnapshotLogicalRef | None = None,
        context_projection_policy: AgentSnapshotLogicalRef | None = None,
        interaction_focus: AgentSnapshotLogicalRef | None = None,
        recall_receipts: AgentSnapshotLogicalRef | None = None,
        context_exposures: AgentSnapshotLogicalRef | None = None,
        index_readiness: AgentSnapshotLogicalRef | None = None,
        learning_disposition: AgentSnapshotLearningDisposition,
        limitations: tuple[str, ...] = (),
    ) -> MemoryStateRef:
        references = {
            "knowledge": knowledge,
            "transcript_evidence": transcript_evidence,
            "artifact_evidence": artifact_evidence,
            "work_context": work_context,
            "recall_policy": recall_policy,
            "admission_policy": admission_policy,
            "context_projection_policy": context_projection_policy,
            "interaction_focus": interaction_focus,
            "recall_receipts": recall_receipts,
            "context_exposures": context_exposures,
            "index_readiness": index_readiness,
        }
        material: dict[str, object] = {
            "schema_version": 1,
            "learning_disposition": learning_disposition.value,
            "limitations": list(limitations),
        }
        for field_name in (
            "knowledge",
            "transcript_evidence",
            "artifact_evidence",
            "work_context",
            "recall_policy",
            "admission_policy",
            "context_projection_policy",
            "interaction_focus",
            "recall_receipts",
            "context_exposures",
            "index_readiness",
        ):
            reference = references[field_name]
            material[field_name] = None if reference is None else reference.identity_material()
        return cls(
            fingerprint=_content_sha256(material, "memory_state"),
            knowledge=knowledge,
            transcript_evidence=transcript_evidence,
            artifact_evidence=artifact_evidence,
            work_context=work_context,
            recall_policy=recall_policy,
            admission_policy=admission_policy,
            context_projection_policy=context_projection_policy,
            interaction_focus=interaction_focus,
            recall_receipts=recall_receipts,
            context_exposures=context_exposures,
            index_readiness=index_readiness,
            learning_disposition=learning_disposition,
            limitations=limitations,
        )


class AgentSnapshotComponentRef(_SnapshotModel):
    kind: AgentSnapshotComponentKind
    provider_id: StrictStr = Field(max_length=256)
    logical: AgentSnapshotLogicalRef
    consistency: AgentSnapshotConsistency
    completeness: AgentSnapshotCompleteness
    redaction: AgentSnapshotRedaction
    materialization: AgentSnapshotMaterializationCapability
    required: StrictBool = True
    consistency_group: StrictStr | None = Field(default=None, max_length=256)
    limitations: tuple[StrictStr, ...] = ()

    @field_validator("provider_id", "consistency_group")
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("limitations", mode="before")
    @classmethod
    def validate_limitations(cls, value: object) -> tuple[str, ...]:
        return _ordered_unique_text(value, "limitations")

    @model_validator(mode="after")
    def validate_statuses(self) -> AgentSnapshotComponentRef:
        unavailable = self.completeness is AgentSnapshotCompleteness.UNAVAILABLE
        if unavailable != (
            self.materialization is AgentSnapshotMaterializationCapability.UNAVAILABLE
        ):
            raise ValueError("Unavailable completeness and materialization must agree.")
        if self.consistency is AgentSnapshotConsistency.TRANSACTIONAL and (
            self.consistency_group is None
        ):
            raise ValueError("Transactional components require a shared consistency group.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "provider_id": self.provider_id,
            "logical": self.logical.identity_material(),
            "consistency": self.consistency.value,
            "consistency_group": self.consistency_group,
            "completeness": self.completeness.value,
            "redaction": self.redaction.value,
            "materialization": self.materialization.value,
            "required": self.required,
            "limitations": list(self.limitations),
        }


class AgentSnapshotAuthorityRef(_SnapshotModel):
    identity: AgentSnapshotLogicalRef
    candidate_visible: Literal[False] = False


class AgentSnapshot(_SnapshotModel):
    """Strict portable manifest for one bounded logical agent state."""

    record_type: Literal["cayu.agent-snapshot"] = AGENT_SNAPSHOT_RECORD_TYPE
    schema_version: Literal[2] = AGENT_SNAPSHOT_SCHEMA_VERSION
    fingerprint: StrictStr
    capture_request_id: StrictStr = Field(max_length=256)
    captured_at: datetime
    subject: AgentSnapshotSubject
    authority_scope_fingerprint: StrictStr
    execution_profile: AgentSnapshotExecutionProfileRef
    memory_state: MemoryStateRef | None = None
    components: tuple[AgentSnapshotComponentRef, ...] = Field(
        max_length=AGENT_SNAPSHOT_MAX_COMPONENTS
    )
    consistency: AgentSnapshotConsistency
    evaluator: AgentSnapshotAuthorityRef | None = None
    promotion_authority: AgentSnapshotAuthorityRef | None = None
    parent_snapshot_fingerprint: StrictStr | None = None
    lineage: tuple[StrictStr, ...] = ()
    exclusions: tuple[StrictStr, ...] = (
        "credentials",
        "hidden_evaluator_truth",
        "promotion_activation",
        "provider_continuation_state",
        "unrelated_records",
    )
    limitations: tuple[StrictStr, ...] = ()

    @field_validator("fingerprint", "authority_scope_fingerprint", "parent_snapshot_fingerprint")
    @classmethod
    def validate_digests(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("capture_request_id")
    @classmethod
    def validate_request_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("captured_at")
    @classmethod
    def validate_captured_at(cls, value: datetime) -> datetime:
        return _utc(value, "captured_at")

    @field_validator("components", mode="before")
    @classmethod
    def validate_component_sequence(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("components must be an ordered array.")
        return value

    @field_validator("lineage")
    @classmethod
    def validate_lineage(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 64:
            raise ValueError("lineage exceeds its retained depth.")
        for fingerprint in value:
            _sha256_hex(fingerprint, "lineage")
        if len(value) != len(set(value)):
            raise ValueError("lineage cannot repeat a snapshot fingerprint.")
        return value

    @field_validator("exclusions", "limitations", mode="before")
    @classmethod
    def validate_sorted_codes(cls, value: object, info) -> tuple[str, ...]:
        return _ordered_unique_text(value, info.field_name)

    @model_validator(mode="after")
    def validate_manifest(self) -> AgentSnapshot:
        kinds = tuple(component.kind for component in self.components)
        expected_kinds = tuple(sorted(set(kinds), key=str))
        if kinds != expected_kinds:
            raise ValueError("Snapshot components must be unique and sorted by kind.")
        required_kinds = {
            AgentSnapshotComponentKind.BODY,
            AgentSnapshotComponentKind.EXECUTION_PROFILE,
        }
        if not required_kinds.issubset(kinds):
            raise ValueError("AgentSnapshot requires body and execution-profile components.")
        if any(
            component.required and component.completeness is AgentSnapshotCompleteness.UNAVAILABLE
            for component in self.components
        ):
            raise ValueError("Required snapshot components cannot be unavailable.")
        body = self.component(AgentSnapshotComponentKind.BODY)
        if body.logical.fingerprint != self.subject.body_release.fingerprint:
            raise ValueError("Body component does not match the declared body release.")
        profile = self.component(AgentSnapshotComponentKind.EXECUTION_PROFILE)
        if profile.logical.fingerprint != self.execution_profile.fingerprint:
            raise ValueError("Execution-profile component does not match its typed reference.")
        memory_components = tuple(
            component
            for component in self.components
            if component.kind is AgentSnapshotComponentKind.MEMORY
        )
        if (self.memory_state is None) != (not memory_components):
            raise ValueError("MemoryStateRef presence must match the memory component.")
        if self.memory_state is not None and (
            memory_components[0].logical.fingerprint != self.memory_state.fingerprint
        ):
            raise ValueError("Memory component does not match MemoryStateRef.")
        if self.memory_state is not None and any(
            reference.scope_fingerprint not in {None, self.authority_scope_fingerprint}
            for reference in self.memory_state._references()
        ):
            raise ValueError("MemoryStateRef cannot broaden the snapshot authority scope.")
        expected_consistency = agent_snapshot_consistency(self.components)
        if self.consistency is not expected_consistency:
            raise ValueError("Snapshot consistency does not match its component evidence.")
        if self.parent_snapshot_fingerprint is None and self.lineage:
            raise ValueError("Snapshot lineage requires a direct parent fingerprint.")
        if self.parent_snapshot_fingerprint is not None and (
            not self.lineage or self.lineage[-1] != self.parent_snapshot_fingerprint
        ):
            raise ValueError("Snapshot lineage must end at its direct parent.")
        if self.fingerprint != _content_sha256(self.identity_material(), "agent_snapshot"):
            raise ValueError("AgentSnapshot fingerprint does not match its logical contents.")
        encoded = canonical_durable_json_bytes(self.model_dump(mode="json"), "agent_snapshot")
        if len(encoded) > AGENT_SNAPSHOT_MAX_BYTES:
            raise ValueError("AgentSnapshot exceeds its portable byte limit.")
        return self

    def component(self, kind: AgentSnapshotComponentKind) -> AgentSnapshotComponentRef:
        for component in self.components:
            if component.kind is kind:
                return component
        raise KeyError(f"Snapshot has no {kind.value} component.")

    def identity_material(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "subject": {
                "agent_id": self.subject.agent_id,
                "application_id": self.subject.application_id,
                "project_id": self.subject.project_id,
                "body_release": self.subject.body_release.identity_material(),
            },
            "authority_scope_fingerprint": self.authority_scope_fingerprint,
            "execution_profile": self.execution_profile.model_dump(mode="json"),
            "memory_state": (
                None if self.memory_state is None else self.memory_state._identity_material()
            ),
            "components": [component.identity_material() for component in self.components],
            "consistency": self.consistency.value,
            "evaluator": (
                None if self.evaluator is None else self.evaluator.identity.identity_material()
            ),
            "promotion_authority": (
                None
                if self.promotion_authority is None
                else self.promotion_authority.identity.identity_material()
            ),
            "parent_snapshot_fingerprint": self.parent_snapshot_fingerprint,
            "lineage": list(self.lineage),
            "exclusions": list(self.exclusions),
            "limitations": list(self.limitations),
        }

    @classmethod
    def create(
        cls,
        *,
        capture_request_id: str,
        captured_at: datetime,
        subject: AgentSnapshotSubject,
        authority_scope_fingerprint: str,
        execution_profile: AgentSnapshotExecutionProfileRef,
        components: Iterable[AgentSnapshotComponentRef],
        memory_state: MemoryStateRef | None = None,
        evaluator: AgentSnapshotAuthorityRef | None = None,
        promotion_authority: AgentSnapshotAuthorityRef | None = None,
        parent_snapshot_fingerprint: str | None = None,
        lineage: tuple[str, ...] = (),
        exclusions: tuple[str, ...] = (
            "credentials",
            "hidden_evaluator_truth",
            "promotion_activation",
            "provider_continuation_state",
            "unrelated_records",
        ),
        limitations: tuple[str, ...] = (),
    ) -> AgentSnapshot:
        ordered = tuple(sorted(components, key=lambda item: str(item.kind)))
        consistency = agent_snapshot_consistency(ordered)
        provisional = cls.model_construct(
            fingerprint="0" * 64,
            capture_request_id=capture_request_id,
            captured_at=captured_at,
            subject=subject,
            authority_scope_fingerprint=authority_scope_fingerprint,
            execution_profile=execution_profile,
            memory_state=memory_state,
            components=ordered,
            consistency=consistency,
            evaluator=evaluator,
            promotion_authority=promotion_authority,
            parent_snapshot_fingerprint=parent_snapshot_fingerprint,
            lineage=lineage,
            exclusions=exclusions,
            limitations=limitations,
        )
        return cls(
            fingerprint=_content_sha256(provisional.identity_material(), "agent_snapshot"),
            capture_request_id=capture_request_id,
            captured_at=captured_at,
            subject=subject,
            authority_scope_fingerprint=authority_scope_fingerprint,
            execution_profile=execution_profile,
            memory_state=memory_state,
            components=ordered,
            consistency=consistency,
            evaluator=evaluator,
            promotion_authority=promotion_authority,
            parent_snapshot_fingerprint=parent_snapshot_fingerprint,
            lineage=lineage,
            exclusions=exclusions,
            limitations=limitations,
        )


def agent_snapshot_consistency(
    components: Iterable[AgentSnapshotComponentRef],
) -> AgentSnapshotConsistency:
    items = tuple(components)
    if not items:
        raise ValueError("Snapshot consistency requires at least one component.")
    weakest = min(items, key=lambda item: _CONSISTENCY_RANK[item.consistency]).consistency
    if weakest is not AgentSnapshotConsistency.TRANSACTIONAL:
        return weakest
    groups = {component.consistency_group for component in items}
    if len(groups) == 1 and None not in groups:
        return AgentSnapshotConsistency.TRANSACTIONAL
    return AgentSnapshotConsistency.FRONTIER_CONSISTENT


def execution_profile_snapshot_ref(
    profile: ExecutionProfileIdentity,
) -> AgentSnapshotExecutionProfileRef:
    """Project Cayu's existing redacted execution profile into snapshot form."""

    from cayu.runtime.execution_profiles import ExecutionProfileIdentity

    if type(profile) is not ExecutionProfileIdentity:
        raise TypeError("profile must be an exact ExecutionProfileIdentity.")
    validated = ExecutionProfileIdentity.model_validate(profile.model_dump(mode="json"))
    return AgentSnapshotExecutionProfileRef(
        schema_version=validated.schema_version,
        fingerprint=validated.fingerprint,
        components=tuple(
            AgentSnapshotExecutionProfileComponent(
                name=component.component_class.value,
                fingerprint=component.fingerprint,
                availability=component.availability.value,
            )
            for component in validated.components
        ),
    )


def app_body_snapshot_ref(
    manifest: AppManifest,
    *,
    application_release_id: str,
    agent_id: str,
    source_ref: str | None = None,
) -> AgentSnapshotLogicalRef:
    """Bind one registered agent body to an AppManifest and release identity."""

    from cayu.runtime.manifest import AppManifest

    if type(manifest) is not AppManifest:
        raise TypeError("manifest must be an exact AppManifest.")
    validated = AppManifest.model_validate(manifest.model_dump(mode="json"))
    release = _clean(application_release_id, "application_release_id", max_chars=256)
    agent = _clean(agent_id, "agent_id", max_chars=256)
    if agent not in {item.name for item in validated.agents}:
        raise ValueError("AppManifest does not contain the declared agent body.")
    material = {
        "app_manifest_schema_version": validated.schema_version,
        "app_manifest_fingerprint": validated.fingerprint,
        "application_release_id": release,
        "agent_id": agent,
    }
    return AgentSnapshotLogicalRef(
        fingerprint=_content_sha256(material, "agent_body"),
        revision=f"application-release:{release}",
        frontier=f"app-manifest:{validated.fingerprint}",
        source_ref=source_ref,
    )


def workspace_snapshot_ref(
    observation: WorkspaceRevisionObservation,
    *,
    scope_fingerprint: str,
    source_ref: str | None = None,
) -> AgentSnapshotLogicalRef:
    """Project a complete workspace observation without retaining a pathname."""

    from cayu.workspaces.revisions import (
        WorkspaceRevisionObservation,
        WorkspaceRevisionObservationStatus,
    )

    if type(observation) is not WorkspaceRevisionObservation:
        raise TypeError("observation must be an exact WorkspaceRevisionObservation.")
    validated = WorkspaceRevisionObservation.model_validate(observation.model_dump(mode="json"))
    if validated.status is not WorkspaceRevisionObservationStatus.SUPPORTED:
        raise ValueError("Workspace snapshot identity requires a complete supported observation.")
    assert validated.revision is not None
    material = {
        "revision": validated.revision,
        "head_revision": validated.head_revision,
        "path_scope": validated.path_scope,
        "paths": [path.model_dump(mode="json", exclude_none=True) for path in validated.paths],
    }
    return AgentSnapshotLogicalRef(
        fingerprint=_content_sha256(material, "workspace_snapshot"),
        revision=f"workspace:{validated.revision}",
        frontier=(
            None if validated.head_revision is None else f"workspace-head:{validated.head_revision}"
        ),
        scope_fingerprint=scope_fingerprint,
        source_ref=source_ref,
    )


def trajectory_snapshot_ref(
    trajectory: Trajectory,
    *,
    scope_fingerprint: str,
    source_ref: str | None = None,
) -> AgentSnapshotLogicalRef:
    """Hash a strict retained trajectory while keeping its content provider-owned."""

    from cayu.evals.models import Trajectory

    if type(trajectory) is not Trajectory:
        raise TypeError("trajectory must be an exact Trajectory.")
    validated = Trajectory.model_validate(trajectory.model_dump(mode="json"))
    digest = _content_sha256(validated.model_dump(mode="json"), "session_trajectory")
    return AgentSnapshotLogicalRef(
        fingerprint=digest,
        revision=f"trajectory:{digest}",
        scope_fingerprint=scope_fingerprint,
        source_ref=source_ref,
    )


class AgentSnapshotComponentSelector(_SnapshotModel):
    kind: AgentSnapshotComponentKind
    required: StrictBool = True


class AgentSnapshotCaptureRequest(_SnapshotModel):
    capture_request_id: StrictStr = Field(max_length=256)
    subject: AgentSnapshotSubject
    authority_scope_fingerprint: StrictStr
    components: tuple[AgentSnapshotComponentSelector, ...]
    required_consistency: AgentSnapshotConsistency = AgentSnapshotConsistency.FRONTIER_CONSISTENT
    session_id: StrictStr | None = Field(default=None, max_length=512)
    environment_name: StrictStr | None = Field(default=None, max_length=256)
    evaluator: AgentSnapshotAuthorityRef | None = None
    promotion_authority: AgentSnapshotAuthorityRef | None = None
    parent_snapshot_fingerprint: StrictStr | None = None
    lineage: tuple[StrictStr, ...] = ()

    @field_validator("capture_request_id", "session_id", "environment_name")
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        max_chars = 256 if info.field_name != "session_id" else 512
        return _clean(value, info.field_name, max_chars=max_chars)

    @field_validator("authority_scope_fingerprint", "parent_snapshot_fingerprint")
    @classmethod
    def validate_digest(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("components", mode="before")
    @classmethod
    def validate_component_input(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("components must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> AgentSnapshotCaptureRequest:
        kinds = tuple(selector.kind for selector in self.components)
        if kinds != tuple(sorted(set(kinds), key=str)):
            raise ValueError("Capture component selectors must be unique and sorted.")
        required = {
            AgentSnapshotComponentKind.BODY,
            AgentSnapshotComponentKind.EXECUTION_PROFILE,
        }
        if not required.issubset(kinds):
            raise ValueError("Capture requires body and execution-profile selectors.")
        if any(not selector.required for selector in self.components if selector.kind in required):
            raise ValueError("Body and execution-profile selectors are always required.")
        if self.required_consistency is AgentSnapshotConsistency.INCONSISTENT:
            raise ValueError("A capture cannot request inconsistent state as its requirement.")
        if self.parent_snapshot_fingerprint is None and self.lineage:
            raise ValueError("Capture lineage requires a parent snapshot fingerprint.")
        return self


class AgentSnapshotComponentCapture(_SnapshotModel):
    component: AgentSnapshotComponentRef
    execution_profile: AgentSnapshotExecutionProfileRef | None = None
    memory_state: MemoryStateRef | None = None

    @model_validator(mode="after")
    def validate_typed_payload(self) -> AgentSnapshotComponentCapture:
        if (self.component.kind is AgentSnapshotComponentKind.EXECUTION_PROFILE) != (
            self.execution_profile is not None
        ):
            raise ValueError("Only execution-profile capture carries its typed reference.")
        if (self.component.kind is AgentSnapshotComponentKind.MEMORY) != (
            self.memory_state is not None
        ):
            raise ValueError("Only memory capture carries MemoryStateRef.")
        if self.execution_profile is not None and (
            self.component.logical.fingerprint != self.execution_profile.fingerprint
        ):
            raise ValueError("Execution-profile capture identities conflict.")
        if self.memory_state is not None and (
            self.component.logical.fingerprint != self.memory_state.fingerprint
        ):
            raise ValueError("Memory capture identities conflict.")
        return self


class AgentSnapshotOverlayRef(_SnapshotModel):
    kind: AgentSnapshotOverlayKind
    overlay_id: StrictStr = Field(max_length=256)
    fingerprint: StrictStr
    baseline_fingerprint: StrictStr
    candidate_id: StrictStr = Field(max_length=256)
    state_scope_id: StrictStr = Field(max_length=256)
    source_ref: StrictStr | None = Field(default=None, max_length=512)

    @field_validator("overlay_id", "candidate_id", "state_scope_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("fingerprint", "baseline_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("source_ref")
    @classmethod
    def validate_source_ref(cls, value: str | None) -> str | None:
        return AgentSnapshotLogicalRef.validate_safe_source_ref(value)

    @model_validator(mode="after")
    def validate_fingerprint_matches(self) -> AgentSnapshotOverlayRef:
        material = self.identity_material()
        material.pop("fingerprint")
        if self.fingerprint != _content_sha256(material, "snapshot_overlay"):
            raise ValueError("Overlay fingerprint does not match its isolation identity.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "overlay_id": self.overlay_id,
            "fingerprint": self.fingerprint,
            "baseline_fingerprint": self.baseline_fingerprint,
            "candidate_id": self.candidate_id,
            "state_scope_id": self.state_scope_id,
        }

    @classmethod
    def create(
        cls,
        *,
        kind: AgentSnapshotOverlayKind,
        overlay_id: str,
        baseline_fingerprint: str,
        candidate_id: str,
        state_scope_id: str,
        source_ref: str | None = None,
    ) -> AgentSnapshotOverlayRef:
        material = {
            "kind": kind.value,
            "overlay_id": overlay_id,
            "baseline_fingerprint": baseline_fingerprint,
            "candidate_id": candidate_id,
            "state_scope_id": state_scope_id,
        }
        return cls(
            kind=kind,
            overlay_id=overlay_id,
            fingerprint=_content_sha256(material, "snapshot_overlay"),
            baseline_fingerprint=baseline_fingerprint,
            candidate_id=candidate_id,
            state_scope_id=state_scope_id,
            source_ref=source_ref,
        )


class AgentSnapshotMaterializedComponent(_SnapshotModel):
    kind: AgentSnapshotComponentKind
    baseline_fingerprint: StrictStr
    capability: AgentSnapshotMaterializationCapability
    materialization_ref: StrictStr | None = Field(default=None, max_length=512)
    overlay: AgentSnapshotOverlayRef | None = None
    limitations: tuple[StrictStr, ...] = ()

    @field_validator("baseline_fingerprint")
    @classmethod
    def validate_baseline(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("materialization_ref")
    @classmethod
    def validate_materialization_ref(cls, value: str | None) -> str | None:
        return AgentSnapshotLogicalRef.validate_safe_source_ref(value)

    @field_validator("limitations", mode="before")
    @classmethod
    def validate_limitations(cls, value: object) -> tuple[str, ...]:
        return _ordered_unique_text(value, "limitations")

    @model_validator(mode="after")
    def validate_shape(self) -> AgentSnapshotMaterializedComponent:
        expects_overlay = self.kind in {
            AgentSnapshotComponentKind.MEMORY,
            AgentSnapshotComponentKind.WORKSPACE,
        }
        if self.overlay is not None and not expects_overlay:
            raise ValueError("Only memory and workspace components expose overlays.")
        expected_overlay_kind = {
            AgentSnapshotComponentKind.MEMORY: AgentSnapshotOverlayKind.MEMORY,
            AgentSnapshotComponentKind.WORKSPACE: AgentSnapshotOverlayKind.WORKSPACE,
        }.get(self.kind)
        if self.overlay is not None and self.overlay.kind is not expected_overlay_kind:
            raise ValueError("Materialized component overlay kind does not match its component.")
        if self.capability is AgentSnapshotMaterializationCapability.UNAVAILABLE and (
            self.materialization_ref is not None or self.overlay is not None
        ):
            raise ValueError("Unavailable materialization cannot expose a handle or overlay.")
        return self

    def identity_material(self) -> dict[str, object]:
        material = self.model_dump(
            mode="json",
            exclude={"materialization_ref", "overlay"},
        )
        material["overlay"] = None if self.overlay is None else self.overlay.identity_material()
        return material


class AgentSnapshotMaterializationRequest(_SnapshotModel):
    snapshot_fingerprint: StrictStr
    candidate_id: StrictStr = Field(max_length=256)
    trial_id: StrictStr = Field(max_length=256)
    state_mode: AgentSnapshotTrialStateMode

    @field_validator("snapshot_fingerprint")
    @classmethod
    def validate_snapshot_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("candidate_id", "trial_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @property
    def state_scope_id(self) -> str:
        material = {
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "candidate_id": self.candidate_id,
            "trial_id": (
                self.trial_id
                if self.state_mode is AgentSnapshotTrialStateMode.RESET_EACH_TRIAL
                else None
            ),
            "state_mode": self.state_mode.value,
        }
        return _content_sha256(material, "snapshot_state_scope")


class AgentSnapshotMaterializationOperation(_SnapshotModel):
    """Stable provider-owned operation announced before any component effect."""

    record_type: Literal["cayu.agent-snapshot-materialization-operation"] = (
        "cayu.agent-snapshot-materialization-operation"
    )
    schema_version: Literal[1] = 1
    operation_id: StrictStr
    snapshot_fingerprint: StrictStr
    candidate_id: StrictStr = Field(max_length=256)
    state_scope_id: StrictStr = Field(max_length=256)
    state_mode: AgentSnapshotTrialStateMode
    component_kind: AgentSnapshotComponentKind
    provider_id: StrictStr = Field(max_length=256)
    baseline_fingerprint: StrictStr
    capability: AgentSnapshotMaterializationCapability

    @field_validator("operation_id", "snapshot_fingerprint", "baseline_fingerprint")
    @classmethod
    def validate_fingerprints(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("candidate_id", "state_scope_id", "provider_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @model_validator(mode="after")
    def validate_operation_id(self) -> AgentSnapshotMaterializationOperation:
        if self.operation_id != _content_sha256(
            self.identity_material(), "snapshot_materialization_operation"
        ):
            raise ValueError("Materialization operation id does not match its identity.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "candidate_id": self.candidate_id,
            "state_scope_id": self.state_scope_id,
            "state_mode": self.state_mode.value,
            "component_kind": self.component_kind.value,
            "provider_id": self.provider_id,
            "baseline_fingerprint": self.baseline_fingerprint,
            "capability": self.capability.value,
        }

    @classmethod
    def create(
        cls,
        *,
        request: AgentSnapshotMaterializationRequest,
        component: AgentSnapshotComponentRef,
    ) -> AgentSnapshotMaterializationOperation:
        provisional = cls.model_construct(
            operation_id="0" * 64,
            snapshot_fingerprint=request.snapshot_fingerprint,
            candidate_id=request.candidate_id,
            state_scope_id=request.state_scope_id,
            state_mode=request.state_mode,
            component_kind=component.kind,
            provider_id=component.provider_id,
            baseline_fingerprint=component.logical.fingerprint,
            capability=component.materialization,
        )
        return cls(
            operation_id=_content_sha256(
                provisional.identity_material(), "snapshot_materialization_operation"
            ),
            snapshot_fingerprint=request.snapshot_fingerprint,
            candidate_id=request.candidate_id,
            state_scope_id=request.state_scope_id,
            state_mode=request.state_mode,
            component_kind=component.kind,
            provider_id=component.provider_id,
            baseline_fingerprint=component.logical.fingerprint,
            capability=component.materialization,
        )


class AgentSnapshotMaterialization(_SnapshotModel):
    record_type: Literal["cayu.agent-snapshot-materialization"] = (
        "cayu.agent-snapshot-materialization"
    )
    schema_version: Literal[2] = 2
    fingerprint: StrictStr
    progress_id: StrictStr
    snapshot_fingerprint: StrictStr
    candidate_id: StrictStr = Field(max_length=256)
    state_scope_id: StrictStr = Field(max_length=256)
    state_mode: AgentSnapshotTrialStateMode
    created_at: datetime
    components: tuple[AgentSnapshotMaterializedComponent, ...]

    @field_validator("fingerprint", "progress_id", "snapshot_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("candidate_id", "state_scope_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _utc(value, "created_at")

    @model_validator(mode="after")
    def validate_materialization(self) -> AgentSnapshotMaterialization:
        kinds = tuple(component.kind for component in self.components)
        if kinds != tuple(sorted(set(kinds), key=str)):
            raise ValueError("Materialized components must be unique and sorted.")
        for component in self.components:
            if component.overlay is not None and (
                component.overlay.candidate_id != self.candidate_id
                or component.overlay.state_scope_id != self.state_scope_id
            ):
                raise ValueError("Component overlay escaped its candidate state scope.")
        if self.fingerprint != _content_sha256(
            self.identity_material(), "snapshot_materialization"
        ):
            raise ValueError("Materialization fingerprint does not match its contents.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "progress_id": self.progress_id,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "candidate_id": self.candidate_id,
            "state_scope_id": self.state_scope_id,
            "state_mode": self.state_mode.value,
            "components": [component.identity_material() for component in self.components],
        }

    @classmethod
    def create(
        cls,
        *,
        progress_id: str,
        request: AgentSnapshotMaterializationRequest,
        created_at: datetime,
        components: Iterable[AgentSnapshotMaterializedComponent],
    ) -> AgentSnapshotMaterialization:
        ordered = tuple(sorted(components, key=lambda item: str(item.kind)))
        provisional = cls.model_construct(
            fingerprint="0" * 64,
            progress_id=progress_id,
            snapshot_fingerprint=request.snapshot_fingerprint,
            candidate_id=request.candidate_id,
            state_scope_id=request.state_scope_id,
            state_mode=request.state_mode,
            created_at=created_at,
            components=ordered,
        )
        return cls(
            fingerprint=_content_sha256(
                provisional.identity_material(), "snapshot_materialization"
            ),
            progress_id=progress_id,
            snapshot_fingerprint=request.snapshot_fingerprint,
            candidate_id=request.candidate_id,
            state_scope_id=request.state_scope_id,
            state_mode=request.state_mode,
            created_at=created_at,
            components=ordered,
        )


class AgentSnapshotMaterializationProgress(_SnapshotModel):
    """Durable scope plan and compare-and-set component progress."""

    record_type: Literal["cayu.agent-snapshot-materialization-progress"] = (
        "cayu.agent-snapshot-materialization-progress"
    )
    schema_version: Literal[1] = 1
    progress_id: StrictStr
    snapshot_fingerprint: StrictStr
    candidate_id: StrictStr = Field(max_length=256)
    state_scope_id: StrictStr = Field(max_length=256)
    state_mode: AgentSnapshotTrialStateMode
    created_at: datetime
    operations: tuple[AgentSnapshotMaterializationOperation, ...]
    revision: StrictInt = Field(ge=0)
    active_operation_id: StrictStr | None = None
    components: tuple[AgentSnapshotMaterializedComponent, ...] = ()
    materialization_fingerprint: StrictStr | None = None

    @field_validator(
        "progress_id",
        "snapshot_fingerprint",
        "active_operation_id",
        "materialization_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("candidate_id", "state_scope_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _utc(value, "created_at")

    @model_validator(mode="after")
    def validate_progress(self) -> AgentSnapshotMaterializationProgress:
        operation_kinds = tuple(operation.component_kind for operation in self.operations)
        if operation_kinds != tuple(sorted(set(operation_kinds), key=str)):
            raise ValueError("Materialization operations must be unique and sorted.")
        for operation in self.operations:
            if (
                operation.snapshot_fingerprint != self.snapshot_fingerprint
                or operation.candidate_id != self.candidate_id
                or operation.state_scope_id != self.state_scope_id
                or operation.state_mode is not self.state_mode
            ):
                raise ValueError("Materialization operation escaped its durable scope.")
        completed_kinds = tuple(component.kind for component in self.components)
        if completed_kinds != tuple(sorted(set(completed_kinds), key=str)):
            raise ValueError("Completed materialization components must be unique and sorted.")
        operations_by_kind = {operation.component_kind: operation for operation in self.operations}
        for component in self.components:
            operation = operations_by_kind.get(component.kind)
            if operation is None:
                raise ValueError("Completed component has no announced operation.")
            if (
                component.baseline_fingerprint != operation.baseline_fingerprint
                or component.capability is not operation.capability
            ):
                raise ValueError("Completed component differs from its announced operation.")
            if component.overlay is not None and (
                component.overlay.candidate_id != self.candidate_id
                or component.overlay.state_scope_id != self.state_scope_id
            ):
                raise ValueError("Completed component overlay escaped its durable scope.")
        operation_ids = {operation.operation_id for operation in self.operations}
        if self.active_operation_id is not None:
            if self.active_operation_id not in operation_ids:
                raise ValueError("Active materialization operation is not in the durable plan.")
            active_kind = next(
                operation.component_kind
                for operation in self.operations
                if operation.operation_id == self.active_operation_id
            )
            if active_kind in completed_kinds:
                raise ValueError("Completed materialization operation cannot remain active.")
        complete = len(self.components) == len(self.operations)
        if self.materialization_fingerprint is not None and not complete:
            raise ValueError("Only complete progress can bind a materialization fingerprint.")
        if complete and self.active_operation_id is not None:
            raise ValueError("Complete materialization progress cannot remain active.")
        if self.progress_id != _content_sha256(
            self.identity_material(), "snapshot_materialization_progress"
        ):
            raise ValueError("Materialization progress id does not match its immutable plan.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "candidate_id": self.candidate_id,
            "state_scope_id": self.state_scope_id,
            "state_mode": self.state_mode.value,
            "operations": [operation.identity_material() for operation in self.operations],
        }

    @classmethod
    def create(
        cls,
        *,
        request: AgentSnapshotMaterializationRequest,
        created_at: datetime,
        components: Iterable[AgentSnapshotComponentRef],
    ) -> AgentSnapshotMaterializationProgress:
        operations = tuple(
            sorted(
                (
                    AgentSnapshotMaterializationOperation.create(
                        request=request,
                        component=component,
                    )
                    for component in components
                ),
                key=lambda operation: str(operation.component_kind),
            )
        )
        provisional = cls.model_construct(
            progress_id="0" * 64,
            snapshot_fingerprint=request.snapshot_fingerprint,
            candidate_id=request.candidate_id,
            state_scope_id=request.state_scope_id,
            state_mode=request.state_mode,
            created_at=created_at,
            operations=operations,
            revision=0,
            active_operation_id=None,
            components=(),
            materialization_fingerprint=None,
        )
        return cls(
            progress_id=_content_sha256(
                provisional.identity_material(), "snapshot_materialization_progress"
            ),
            snapshot_fingerprint=request.snapshot_fingerprint,
            candidate_id=request.candidate_id,
            state_scope_id=request.state_scope_id,
            state_mode=request.state_mode,
            created_at=created_at,
            operations=operations,
            revision=0,
        )


class AgentSnapshotTrialBinding(_SnapshotModel):
    record_type: Literal["cayu.agent-snapshot-trial"] = "cayu.agent-snapshot-trial"
    schema_version: Literal[1] = 1
    fingerprint: StrictStr
    snapshot_fingerprint: StrictStr
    materialization_fingerprint: StrictStr
    candidate_id: StrictStr = Field(max_length=256)
    case_id: StrictStr = Field(max_length=256)
    trial_id: StrictStr = Field(max_length=256)
    evaluator_fingerprint: StrictStr
    memory_overlay_fingerprint: StrictStr | None = None
    workspace_overlay_fingerprint: StrictStr | None = None
    created_at: datetime

    @field_validator(
        "fingerprint",
        "snapshot_fingerprint",
        "materialization_fingerprint",
        "evaluator_fingerprint",
        "memory_overlay_fingerprint",
        "workspace_overlay_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("candidate_id", "case_id", "trial_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _utc(value, "created_at")

    @model_validator(mode="after")
    def validate_fingerprint_matches(self) -> AgentSnapshotTrialBinding:
        if self.fingerprint != _content_sha256(self.identity_material(), "snapshot_trial"):
            raise ValueError("Trial fingerprint does not match its lineage.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "materialization_fingerprint": self.materialization_fingerprint,
            "candidate_id": self.candidate_id,
            "case_id": self.case_id,
            "trial_id": self.trial_id,
            "evaluator_fingerprint": self.evaluator_fingerprint,
            "memory_overlay_fingerprint": self.memory_overlay_fingerprint,
            "workspace_overlay_fingerprint": self.workspace_overlay_fingerprint,
        }

    @classmethod
    def create(
        cls,
        *,
        materialization: AgentSnapshotMaterialization,
        case_id: str,
        trial_id: str,
        evaluator_fingerprint: str,
        created_at: datetime,
    ) -> AgentSnapshotTrialBinding:
        overlays = {
            component.overlay.kind: component.overlay.fingerprint
            for component in materialization.components
            if component.overlay is not None
        }
        memory_overlay_fingerprint = overlays.get(AgentSnapshotOverlayKind.MEMORY)
        workspace_overlay_fingerprint = overlays.get(AgentSnapshotOverlayKind.WORKSPACE)
        provisional = cls.model_construct(
            fingerprint="0" * 64,
            snapshot_fingerprint=materialization.snapshot_fingerprint,
            materialization_fingerprint=materialization.fingerprint,
            candidate_id=materialization.candidate_id,
            case_id=case_id,
            trial_id=trial_id,
            evaluator_fingerprint=evaluator_fingerprint,
            memory_overlay_fingerprint=memory_overlay_fingerprint,
            workspace_overlay_fingerprint=workspace_overlay_fingerprint,
            created_at=created_at,
        )
        return cls(
            fingerprint=_content_sha256(provisional.identity_material(), "snapshot_trial"),
            snapshot_fingerprint=materialization.snapshot_fingerprint,
            materialization_fingerprint=materialization.fingerprint,
            candidate_id=materialization.candidate_id,
            case_id=case_id,
            trial_id=trial_id,
            evaluator_fingerprint=evaluator_fingerprint,
            memory_overlay_fingerprint=memory_overlay_fingerprint,
            workspace_overlay_fingerprint=workspace_overlay_fingerprint,
            created_at=created_at,
        )

    def session_metadata(self) -> dict[str, object]:
        """Return bounded machine-readable lineage for an ordinary Cayu run."""

        return {
            AGENT_SNAPSHOT_TRIAL_METADATA_KEY: {
                "schema_version": self.schema_version,
                "fingerprint": self.fingerprint,
                "snapshot_fingerprint": self.snapshot_fingerprint,
                "materialization_fingerprint": self.materialization_fingerprint,
                "candidate_id": self.candidate_id,
                "case_id": self.case_id,
                "trial_id": self.trial_id,
                "evaluator_fingerprint": self.evaluator_fingerprint,
                "memory_overlay_fingerprint": self.memory_overlay_fingerprint,
                "workspace_overlay_fingerprint": self.workspace_overlay_fingerprint,
            }
        }


class AgentSnapshotResultBinding(_SnapshotModel):
    record_type: Literal["cayu.agent-snapshot-result"] = "cayu.agent-snapshot-result"
    schema_version: Literal[1] = 1
    fingerprint: StrictStr
    trial_fingerprint: StrictStr
    session_id: StrictStr = Field(max_length=512)
    terminal_disposition: AgentSnapshotTerminalDisposition
    runtime_evidence_fingerprint: StrictStr
    eval_result_revision: StrictStr
    memory_evidence_fingerprint: StrictStr | None = None
    usage_fingerprint: StrictStr | None = None
    cost_fingerprint: StrictStr | None = None
    recorded_at: datetime

    @field_validator(
        "fingerprint",
        "trial_fingerprint",
        "runtime_evidence_fingerprint",
        "eval_result_revision",
        "memory_evidence_fingerprint",
        "usage_fingerprint",
        "cost_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(cls, value: datetime) -> datetime:
        return _utc(value, "recorded_at")

    @model_validator(mode="after")
    def validate_fingerprint_matches(self) -> AgentSnapshotResultBinding:
        if self.fingerprint != _content_sha256(self.identity_material(), "snapshot_result"):
            raise ValueError("Result fingerprint does not match its evidence lineage.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "trial_fingerprint": self.trial_fingerprint,
            "session_id": self.session_id,
            "terminal_disposition": self.terminal_disposition.value,
            "runtime_evidence_fingerprint": self.runtime_evidence_fingerprint,
            "eval_result_revision": self.eval_result_revision,
            "memory_evidence_fingerprint": self.memory_evidence_fingerprint,
            "usage_fingerprint": self.usage_fingerprint,
            "cost_fingerprint": self.cost_fingerprint,
        }

    @classmethod
    def create(
        cls,
        *,
        trial: AgentSnapshotTrialBinding,
        session_id: str,
        terminal_disposition: AgentSnapshotTerminalDisposition,
        runtime_evidence_fingerprint: str,
        eval_result_revision: str,
        recorded_at: datetime,
        memory_evidence_fingerprint: str | None = None,
        usage_fingerprint: str | None = None,
        cost_fingerprint: str | None = None,
    ) -> AgentSnapshotResultBinding:
        provisional = cls.model_construct(
            fingerprint="0" * 64,
            trial_fingerprint=trial.fingerprint,
            session_id=session_id,
            terminal_disposition=terminal_disposition,
            runtime_evidence_fingerprint=runtime_evidence_fingerprint,
            eval_result_revision=eval_result_revision,
            memory_evidence_fingerprint=memory_evidence_fingerprint,
            usage_fingerprint=usage_fingerprint,
            cost_fingerprint=cost_fingerprint,
            recorded_at=recorded_at,
        )
        return cls(
            fingerprint=_content_sha256(provisional.identity_material(), "snapshot_result"),
            trial_fingerprint=trial.fingerprint,
            session_id=session_id,
            terminal_disposition=terminal_disposition,
            runtime_evidence_fingerprint=runtime_evidence_fingerprint,
            eval_result_revision=eval_result_revision,
            memory_evidence_fingerprint=memory_evidence_fingerprint,
            usage_fingerprint=usage_fingerprint,
            cost_fingerprint=cost_fingerprint,
            recorded_at=recorded_at,
        )


class AgentSnapshotCaptureError(RuntimeError):
    def __init__(self, code: str, *, component: AgentSnapshotComponentKind | None = None) -> None:
        self.code = _clean(code, "code", max_chars=256)
        self.component = component
        detail = self.code if component is None else f"{component.value}:{self.code}"
        super().__init__(f"Agent snapshot capture failed closed ({detail}).")


class AgentSnapshotVerificationError(RuntimeError):
    pass


class AgentSnapshotMaterializationError(RuntimeError):
    pass


class AgentSnapshotStoreConflict(RuntimeError):
    pass


class AgentSnapshotComponentProvider(ABC):
    """Application/component-owned capture and materialization adapter."""

    kind: AgentSnapshotComponentKind
    provider_id: str

    @abstractmethod
    async def capture(
        self,
        request: AgentSnapshotCaptureRequest,
        selector: AgentSnapshotComponentSelector,
    ) -> AgentSnapshotComponentCapture:
        """Capture one exact, bounded logical component reference."""

    @abstractmethod
    async def verify(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
    ) -> bool:
        """Return whether the exact component remains valid and authorized."""

    @abstractmethod
    async def materialize(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        """Materialize the exact, durably announced operation once.

        The provider must bind all effects to ``operation.operation_id`` in its
        own durable authority so concurrent or recovered calls cannot create a
        second logical materialization.
        """

    @abstractmethod
    async def recover_materialization_operation(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        """Safely finish or recover one outcome-unknown announced operation.

        Implementations must use ``operation.operation_id`` as durable
        idempotency authority. They must fail closed when they cannot prove
        whether an earlier invocation took effect; the coordinator never calls
        ``materialize`` again for an operation already recorded as active.
        """

    async def recover(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        materialized: AgentSnapshotMaterializedComponent,
        materialization: AgentSnapshotMaterialization,
    ) -> AgentSnapshotMaterializedComponent:
        """Recover an existing materialization without repeating candidate effects."""

        return materialized


class AgentSnapshotStore(ABC):
    @abstractmethod
    async def save_snapshot(self, snapshot: AgentSnapshot) -> AgentSnapshot:
        pass

    @abstractmethod
    async def load_snapshot(self, fingerprint: str) -> AgentSnapshot | None:
        pass

    @abstractmethod
    async def save_materialization(
        self, materialization: AgentSnapshotMaterialization
    ) -> AgentSnapshotMaterialization:
        pass

    @abstractmethod
    async def load_materialization(self, fingerprint: str) -> AgentSnapshotMaterialization | None:
        pass

    @abstractmethod
    async def load_materialization_for_scope(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterialization | None:
        pass

    @abstractmethod
    async def begin_materialization(
        self,
        progress: AgentSnapshotMaterializationProgress,
    ) -> AgentSnapshotMaterializationProgress:
        """Atomically bind a scope to its immutable component operation plan."""

    @abstractmethod
    async def load_materialization_progress_for_scope(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterializationProgress | None:
        pass

    @abstractmethod
    async def claim_materialization_operation(
        self,
        progress: AgentSnapshotMaterializationProgress,
        operation_id: str,
    ) -> AgentSnapshotMaterializationProgress:
        """CAS an unstarted operation to active before provider effects."""

    @abstractmethod
    async def complete_materialization_operation(
        self,
        progress: AgentSnapshotMaterializationProgress,
        operation_id: str,
        component: AgentSnapshotMaterializedComponent,
    ) -> AgentSnapshotMaterializationProgress:
        """CAS one exact provider result into durable progress."""

    @abstractmethod
    async def finalize_materialization(
        self,
        progress: AgentSnapshotMaterializationProgress,
        materialization: AgentSnapshotMaterialization,
    ) -> AgentSnapshotMaterialization:
        """Atomically publish a completed materialization for its scope."""

    @abstractmethod
    async def save_trial(self, trial: AgentSnapshotTrialBinding) -> AgentSnapshotTrialBinding:
        pass

    @abstractmethod
    async def load_trial(self, fingerprint: str) -> AgentSnapshotTrialBinding | None:
        pass

    @abstractmethod
    async def save_result(self, result: AgentSnapshotResultBinding) -> AgentSnapshotResultBinding:
        pass

    @abstractmethod
    async def load_result(self, fingerprint: str) -> AgentSnapshotResultBinding | None:
        pass


def _scope_key(
    value: AgentSnapshotMaterializationRequest | AgentSnapshotMaterializationProgress,
) -> tuple[str, str, str, str]:
    return (
        value.snapshot_fingerprint,
        value.candidate_id,
        value.state_scope_id,
        value.state_mode.value,
    )


def _validate_store_record(value: BaseModel) -> BaseModel:
    try:
        return type(value).model_validate(value.model_dump(mode="json"))
    except Exception as error:
        raise AgentSnapshotStoreConflict(
            "Record fingerprint does not match its logical identity."
        ) from error


def _validate_store_response(
    value: object,
    model: type[BaseModel],
    operation: str,
) -> BaseModel:
    if type(value) is not model:
        raise AgentSnapshotStoreConflict(f"Store returned an invalid {operation} record type.")
    try:
        return model.model_validate(value.model_dump(mode="json"))
    except Exception as error:
        raise AgentSnapshotStoreConflict(
            f"Store returned an invalid {operation} record."
        ) from error


def _same_record_identity(first: BaseModel, second: BaseModel) -> bool:
    first_identity = getattr(first, "identity_material", None)
    second_identity = getattr(second, "identity_material", None)
    if not callable(first_identity) or not callable(second_identity):
        return first == second
    return first_identity() == second_identity()


def _claim_progress(
    progress: AgentSnapshotMaterializationProgress,
    operation_id: str,
) -> AgentSnapshotMaterializationProgress:
    _sha256_hex(operation_id, "operation_id")
    if progress.materialization_fingerprint is not None:
        raise AgentSnapshotStoreConflict("Materialization scope is already complete.")
    if progress.active_operation_id is not None:
        raise AgentSnapshotStoreConflict("Another materialization operation is already active.")
    operations = {operation.operation_id: operation for operation in progress.operations}
    operation = operations.get(operation_id)
    if operation is None:
        raise AgentSnapshotStoreConflict("Materialization operation is outside the scope plan.")
    if operation.component_kind in {component.kind for component in progress.components}:
        raise AgentSnapshotStoreConflict("Materialization operation is already complete.")
    return AgentSnapshotMaterializationProgress.model_validate(
        progress.model_copy(
            update={
                "revision": progress.revision + 1,
                "active_operation_id": operation_id,
            }
        ).model_dump(mode="json")
    )


def _require_initial_progress(progress: AgentSnapshotMaterializationProgress) -> None:
    if (
        progress.revision != 0
        or progress.active_operation_id is not None
        or progress.components
        or progress.materialization_fingerprint is not None
    ):
        raise AgentSnapshotStoreConflict(
            "A new materialization scope requires empty revision-zero progress."
        )


def _complete_progress(
    progress: AgentSnapshotMaterializationProgress,
    operation_id: str,
    component: AgentSnapshotMaterializedComponent,
) -> AgentSnapshotMaterializationProgress:
    _sha256_hex(operation_id, "operation_id")
    if progress.active_operation_id != operation_id:
        raise AgentSnapshotStoreConflict("Materialization operation is not the active claim.")
    operation = next(
        (item for item in progress.operations if item.operation_id == operation_id),
        None,
    )
    if operation is None or operation.component_kind is not component.kind:
        raise AgentSnapshotStoreConflict("Materialized component differs from its active claim.")
    components = tuple(sorted((*progress.components, component), key=lambda item: str(item.kind)))
    return AgentSnapshotMaterializationProgress.model_validate(
        progress.model_copy(
            update={
                "revision": progress.revision + 1,
                "active_operation_id": None,
                "components": components,
            }
        ).model_dump(mode="json")
    )


def _final_progress(
    progress: AgentSnapshotMaterializationProgress,
    materialization: AgentSnapshotMaterialization,
) -> AgentSnapshotMaterializationProgress:
    if progress.active_operation_id is not None or len(progress.components) != len(
        progress.operations
    ):
        raise AgentSnapshotStoreConflict("Materialization progress is not complete.")
    if (
        materialization.progress_id != progress.progress_id
        or materialization.snapshot_fingerprint != progress.snapshot_fingerprint
        or materialization.candidate_id != progress.candidate_id
        or materialization.state_scope_id != progress.state_scope_id
        or materialization.state_mode is not progress.state_mode
        or tuple(component.identity_material() for component in materialization.components)
        != tuple(component.identity_material() for component in progress.components)
    ):
        raise AgentSnapshotStoreConflict(
            "Materialization differs from its durable component progress."
        )
    return AgentSnapshotMaterializationProgress.model_validate(
        progress.model_copy(
            update={
                "revision": progress.revision + 1,
                "materialization_fingerprint": materialization.fingerprint,
            }
        ).model_dump(mode="json")
    )


def _require_progress_successor(
    previous: AgentSnapshotMaterializationProgress,
    successor: AgentSnapshotMaterializationProgress,
) -> None:
    if (
        successor.progress_id != previous.progress_id
        or _scope_key(successor) != _scope_key(previous)
        or not _same_record_identity(successor, previous)
        or successor.revision <= previous.revision
    ):
        raise AgentSnapshotStoreConflict(
            "Materialization progress refresh is not a monotonic successor."
        )
    successor_components = {component.kind: component for component in successor.components}
    for component in previous.components:
        retained = successor_components.get(component.kind)
        if retained is None or retained.identity_material() != component.identity_material():
            raise AgentSnapshotStoreConflict(
                "Materialization progress refresh removed completed evidence."
            )
    if (
        previous.materialization_fingerprint is not None
        and successor.materialization_fingerprint != previous.materialization_fingerprint
    ):
        raise AgentSnapshotStoreConflict(
            "Materialization progress refresh changed its final identity."
        )
    if previous.active_operation_id is not None:
        active_operation = next(
            operation
            for operation in previous.operations
            if operation.operation_id == previous.active_operation_id
        )
        if active_operation.component_kind not in successor_components:
            raise AgentSnapshotStoreConflict(
                "Materialization progress refresh discarded an active operation outcome."
            )


def _require_scope_materialization(
    request: AgentSnapshotMaterializationRequest,
    progress: AgentSnapshotMaterializationProgress,
    materialization: AgentSnapshotMaterialization,
) -> None:
    try:
        if (
            _scope_key(progress) != _scope_key(request)
            or progress.materialization_fingerprint != materialization.fingerprint
        ):
            raise AgentSnapshotStoreConflict(
                "Materialization pointer differs from its durable scope."
            )
        _final_progress(progress, materialization)
    except AgentSnapshotStoreConflict as error:
        raise AgentSnapshotStoreConflict(
            "Materialization scope index does not match its durable records."
        ) from error


def _validated_sqlite_progress_row(
    row: sqlite3.Row,
    *,
    expected_scope: tuple[str, str, str, str] | None = None,
) -> AgentSnapshotMaterializationProgress:
    try:
        progress = AgentSnapshotMaterializationProgress.model_validate_json(row["document"])
    except Exception as error:
        raise AgentSnapshotStoreConflict(
            "Materialization scope contains invalid progress."
        ) from error
    stored_scope = (
        row["snapshot_fingerprint"],
        row["candidate_id"],
        row["state_scope_id"],
        row["state_mode"],
    )
    if (
        _scope_key(progress) != stored_scope
        or (expected_scope is not None and stored_scope != expected_scope)
        or row["progress_id"] != progress.progress_id
        or row["revision"] != progress.revision
        or row["materialization_fingerprint"] != progress.materialization_fingerprint
    ):
        raise AgentSnapshotStoreConflict(
            "Materialization scope index does not match its durable progress."
        )
    return progress


class InMemoryAgentSnapshotStore(AgentSnapshotStore):
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], BaseModel] = {}
        self._materialization_progress: dict[
            tuple[str, str, str, str], AgentSnapshotMaterializationProgress
        ] = {}

    def _save(self, kind: str, value: BaseModel, fingerprint: str) -> BaseModel:
        validated = _validate_store_record(value)
        if getattr(validated, "fingerprint", None) != fingerprint:
            raise AgentSnapshotStoreConflict(
                "Record fingerprint does not match its content-addressed key."
            )
        key = (kind, fingerprint)
        existing = self._records.get(key)
        if existing is not None:
            if not _same_record_identity(existing, validated):
                raise AgentSnapshotStoreConflict(
                    "Record fingerprint is already bound to another logical identity."
                )
            return type(value).model_validate(existing.model_dump(mode="json"))
        self._records[key] = validated
        return type(value).model_validate(validated.model_dump(mode="json"))

    def _load(self, kind: str, fingerprint: str, model: type[BaseModel]) -> BaseModel | None:
        _sha256_hex(fingerprint, "fingerprint")
        existing = self._records.get((kind, fingerprint))
        if existing is None:
            return None
        try:
            validated = model.model_validate(existing.model_dump(mode="json"))
        except Exception as error:
            raise AgentSnapshotStoreConflict(
                "Stored record is invalid for its fingerprint."
            ) from error
        if getattr(validated, "fingerprint", None) != fingerprint:
            raise AgentSnapshotStoreConflict(
                "Stored record fingerprint does not match its content-addressed key."
            )
        return validated

    async def save_snapshot(self, snapshot: AgentSnapshot) -> AgentSnapshot:
        return cast("AgentSnapshot", self._save("snapshot", snapshot, snapshot.fingerprint))

    async def load_snapshot(self, fingerprint: str) -> AgentSnapshot | None:
        return cast("AgentSnapshot | None", self._load("snapshot", fingerprint, AgentSnapshot))

    async def save_materialization(
        self, materialization: AgentSnapshotMaterialization
    ) -> AgentSnapshotMaterialization:
        validated = cast("AgentSnapshotMaterialization", _validate_store_record(materialization))
        progress = next(
            (
                item
                for item in self._materialization_progress.values()
                if item.progress_id == validated.progress_id
            ),
            None,
        )
        if progress is None:
            raise AgentSnapshotStoreConflict(
                "Materialization has no durable pre-effect progress record."
            )
        if progress.materialization_fingerprint is not None:
            if progress.materialization_fingerprint != validated.fingerprint:
                raise AgentSnapshotStoreConflict(
                    "Materialization scope is already finalized with another identity."
                )
            _final_progress(progress, validated)
            existing = await self.load_materialization(validated.fingerprint)
            if existing is None or not _same_record_identity(existing, validated):
                raise AgentSnapshotStoreConflict(
                    "Finalized materialization record is missing or conflicting."
                )
            return existing
        return await self.finalize_materialization(progress, validated)

    async def load_materialization(self, fingerprint: str) -> AgentSnapshotMaterialization | None:
        return cast(
            "AgentSnapshotMaterialization | None",
            self._load("materialization", fingerprint, AgentSnapshotMaterialization),
        )

    async def load_materialization_for_scope(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterialization | None:
        if type(request) is not AgentSnapshotMaterializationRequest:
            raise TypeError("request must be an AgentSnapshotMaterializationRequest.")
        progress = self._materialization_progress.get(_scope_key(request))
        if progress is None or progress.materialization_fingerprint is None:
            return None
        fingerprint = progress.materialization_fingerprint
        materialization = await self.load_materialization(fingerprint)
        if materialization is None:
            raise AgentSnapshotStoreConflict(
                "Materialization scope points to a missing in-memory record."
            )
        _require_scope_materialization(request, progress, materialization)
        return materialization

    async def begin_materialization(
        self,
        progress: AgentSnapshotMaterializationProgress,
    ) -> AgentSnapshotMaterializationProgress:
        validated = cast("AgentSnapshotMaterializationProgress", _validate_store_record(progress))
        key = _scope_key(validated)
        existing = self._materialization_progress.get(key)
        if existing is not None:
            if existing.progress_id != validated.progress_id or not _same_record_identity(
                existing, validated
            ):
                raise AgentSnapshotStoreConflict(
                    "Materialization scope is already bound to another operation plan."
                )
            return AgentSnapshotMaterializationProgress.model_validate(
                existing.model_dump(mode="json")
            )
        _require_initial_progress(validated)
        self._materialization_progress[key] = validated
        return AgentSnapshotMaterializationProgress.model_validate(
            validated.model_dump(mode="json")
        )

    async def load_materialization_progress_for_scope(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterializationProgress | None:
        if type(request) is not AgentSnapshotMaterializationRequest:
            raise TypeError("request must be an AgentSnapshotMaterializationRequest.")
        progress = self._materialization_progress.get(_scope_key(request))
        if progress is None:
            return None
        return AgentSnapshotMaterializationProgress.model_validate(progress.model_dump(mode="json"))

    def _current_progress(
        self,
        expected: AgentSnapshotMaterializationProgress,
    ) -> AgentSnapshotMaterializationProgress:
        validated = cast("AgentSnapshotMaterializationProgress", _validate_store_record(expected))
        current = self._materialization_progress.get(_scope_key(validated))
        if (
            current is None
            or current.progress_id != validated.progress_id
            or current.revision != validated.revision
            or current != validated
        ):
            raise AgentSnapshotStoreConflict("Materialization progress compare-and-set failed.")
        return current

    async def claim_materialization_operation(
        self,
        progress: AgentSnapshotMaterializationProgress,
        operation_id: str,
    ) -> AgentSnapshotMaterializationProgress:
        current = self._current_progress(progress)
        updated = _claim_progress(current, operation_id)
        self._materialization_progress[_scope_key(updated)] = updated
        return AgentSnapshotMaterializationProgress.model_validate(updated.model_dump(mode="json"))

    async def complete_materialization_operation(
        self,
        progress: AgentSnapshotMaterializationProgress,
        operation_id: str,
        component: AgentSnapshotMaterializedComponent,
    ) -> AgentSnapshotMaterializationProgress:
        validated = AgentSnapshotMaterializedComponent.model_validate(
            component.model_dump(mode="json")
        )
        current = self._current_progress(progress)
        updated = _complete_progress(current, operation_id, validated)
        self._materialization_progress[_scope_key(updated)] = updated
        return AgentSnapshotMaterializationProgress.model_validate(updated.model_dump(mode="json"))

    async def finalize_materialization(
        self,
        progress: AgentSnapshotMaterializationProgress,
        materialization: AgentSnapshotMaterialization,
    ) -> AgentSnapshotMaterialization:
        validated = cast("AgentSnapshotMaterialization", _validate_store_record(materialization))
        expected = cast("AgentSnapshotMaterializationProgress", _validate_store_record(progress))
        try:
            current = self._current_progress(expected)
        except AgentSnapshotStoreConflict as error:
            current = self._materialization_progress.get(_scope_key(expected))
            if (
                current is None
                or current.progress_id != expected.progress_id
                or current.materialization_fingerprint != validated.fingerprint
            ):
                raise
            expected_final = _final_progress(expected, validated)
            if current != expected_final:
                raise AgentSnapshotStoreConflict(
                    "Materialization finalization compare-and-set failed."
                ) from error
            existing = await self.load_materialization(validated.fingerprint)
            if existing is None or not _same_record_identity(existing, validated):
                raise AgentSnapshotStoreConflict(
                    "Finalized materialization record is missing or conflicting."
                ) from error
            return existing
        if current.materialization_fingerprint is not None:
            if current.materialization_fingerprint != validated.fingerprint:
                raise AgentSnapshotStoreConflict(
                    "Materialization scope is already finalized with another identity."
                )
            existing = await self.load_materialization(validated.fingerprint)
            if existing is None or not _same_record_identity(existing, validated):
                raise AgentSnapshotStoreConflict(
                    "Finalized materialization record is missing or conflicting."
                )
            return existing
        updated = _final_progress(current, validated)
        stored = cast(
            "AgentSnapshotMaterialization",
            self._save("materialization", validated, validated.fingerprint),
        )
        self._materialization_progress[_scope_key(updated)] = updated
        return stored

    async def save_trial(self, trial: AgentSnapshotTrialBinding) -> AgentSnapshotTrialBinding:
        return cast("AgentSnapshotTrialBinding", self._save("trial", trial, trial.fingerprint))

    async def load_trial(self, fingerprint: str) -> AgentSnapshotTrialBinding | None:
        return cast(
            "AgentSnapshotTrialBinding | None",
            self._load("trial", fingerprint, AgentSnapshotTrialBinding),
        )

    async def save_result(self, result: AgentSnapshotResultBinding) -> AgentSnapshotResultBinding:
        return cast("AgentSnapshotResultBinding", self._save("result", result, result.fingerprint))

    async def load_result(self, fingerprint: str) -> AgentSnapshotResultBinding | None:
        return cast(
            "AgentSnapshotResultBinding | None",
            self._load("result", fingerprint, AgentSnapshotResultBinding),
        )


class SQLiteAgentSnapshotStore(AgentSnapshotStore):
    """Small durable journal for manifests and evaluation lineage records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._write_lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_records (
                    record_kind TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    document TEXT NOT NULL,
                    PRIMARY KEY (record_kind, fingerprint)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_materialization_progress (
                    snapshot_fingerprint TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    state_scope_id TEXT NOT NULL,
                    state_mode TEXT NOT NULL,
                    progress_id TEXT NOT NULL UNIQUE,
                    revision INTEGER NOT NULL,
                    document TEXT NOT NULL,
                    materialization_fingerprint TEXT,
                    PRIMARY KEY (
                        snapshot_fingerprint,
                        candidate_id,
                        state_scope_id,
                        state_mode
                    )
                )
                """
            )

    def _save_sync(self, kind: str, fingerprint: str, value: BaseModel) -> str:
        validated = _validate_store_record(value)
        if getattr(validated, "fingerprint", None) != fingerprint:
            raise AgentSnapshotStoreConflict(
                "Record fingerprint does not match its content-addressed key."
            )
        document = json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO cayu_agent_snapshot_records "
                "(record_kind, fingerprint, document) VALUES (?, ?, ?)",
                (kind, fingerprint, document),
            )
            row = connection.execute(
                "SELECT document FROM cayu_agent_snapshot_records "
                "WHERE record_kind = ? AND fingerprint = ?",
                (kind, fingerprint),
            ).fetchone()
        if row is None:
            raise RuntimeError("SQLite failed to persist the snapshot record.")
        stored_document = cast("str", row["document"])
        try:
            stored = type(value).model_validate_json(stored_document)
        except Exception as error:
            raise AgentSnapshotStoreConflict(
                "Stored record is invalid for its fingerprint."
            ) from error
        if getattr(stored, "fingerprint", None) != fingerprint:
            raise AgentSnapshotStoreConflict(
                "Stored record fingerprint does not match its content-addressed key."
            )
        if not _same_record_identity(stored, validated):
            raise AgentSnapshotStoreConflict(
                "Record fingerprint is already bound to another logical identity."
            )
        return stored_document

    def _load_sync(
        self,
        kind: str,
        fingerprint: str,
        model: type[BaseModel],
    ) -> str | None:
        _sha256_hex(fingerprint, "fingerprint")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT document FROM cayu_agent_snapshot_records "
                "WHERE record_kind = ? AND fingerprint = ?",
                (kind, fingerprint),
            ).fetchone()
        if row is None:
            return None
        document = cast("str", row["document"])
        try:
            stored = model.model_validate_json(document)
        except Exception as error:
            raise AgentSnapshotStoreConflict(
                "Stored record is invalid for its fingerprint."
            ) from error
        if getattr(stored, "fingerprint", None) != fingerprint:
            raise AgentSnapshotStoreConflict(
                "Stored record fingerprint does not match its content-addressed key."
            )
        return document

    def _begin_materialization_sync(
        self,
        progress: AgentSnapshotMaterializationProgress,
    ) -> str:
        validated = cast("AgentSnapshotMaterializationProgress", _validate_store_record(progress))
        _require_initial_progress(validated)
        document = json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        scope_values = _scope_key(validated)
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO cayu_agent_snapshot_materialization_progress "
                "(snapshot_fingerprint, candidate_id, state_scope_id, state_mode, "
                "progress_id, revision, document, materialization_fingerprint) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (*scope_values, validated.progress_id, validated.revision, document),
            )
            row = connection.execute(
                "SELECT snapshot_fingerprint, candidate_id, state_scope_id, state_mode, "
                "progress_id, revision, document, materialization_fingerprint "
                "FROM cayu_agent_snapshot_materialization_progress "
                "WHERE snapshot_fingerprint = ? AND candidate_id = ? "
                "AND state_scope_id = ? AND state_mode = ?",
                scope_values,
            ).fetchone()
        if row is None:
            raise RuntimeError("SQLite failed to persist materialization progress.")
        stored_document = cast("str", row["document"])
        stored = _validated_sqlite_progress_row(row, expected_scope=scope_values)
        if row["progress_id"] != validated.progress_id or not _same_record_identity(
            stored, validated
        ):
            raise AgentSnapshotStoreConflict(
                "Materialization scope is already bound to another operation plan."
            )
        return stored_document

    def _load_materialization_progress_for_scope_sync(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> str | None:
        scope_values = _scope_key(request)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_fingerprint, candidate_id, state_scope_id, state_mode, "
                "progress_id, revision, document, materialization_fingerprint "
                "FROM cayu_agent_snapshot_materialization_progress "
                "WHERE snapshot_fingerprint = ? AND candidate_id = ? "
                "AND state_scope_id = ? AND state_mode = ?",
                scope_values,
            ).fetchone()
        if row is None:
            return None
        _validated_sqlite_progress_row(row, expected_scope=scope_values)
        return cast("str", row["document"])

    def _transition_materialization_progress_sync(
        self,
        expected: AgentSnapshotMaterializationProgress,
        updated: AgentSnapshotMaterializationProgress,
    ) -> str:
        expected = cast("AgentSnapshotMaterializationProgress", _validate_store_record(expected))
        updated = cast("AgentSnapshotMaterializationProgress", _validate_store_record(updated))
        if (
            expected.progress_id != updated.progress_id
            or _scope_key(expected) != _scope_key(updated)
            or updated.revision != expected.revision + 1
        ):
            raise AgentSnapshotStoreConflict("Invalid materialization progress transition.")
        document = json.dumps(
            updated.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expected_document = json.dumps(
            expected.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE cayu_agent_snapshot_materialization_progress "
                "SET revision = ?, document = ?, materialization_fingerprint = ? "
                "WHERE snapshot_fingerprint = ? AND candidate_id = ? "
                "AND state_scope_id = ? AND state_mode = ? "
                "AND progress_id = ? AND revision = ? AND document = ? "
                "AND materialization_fingerprint IS ?",
                (
                    updated.revision,
                    document,
                    updated.materialization_fingerprint,
                    *_scope_key(expected),
                    expected.progress_id,
                    expected.revision,
                    expected_document,
                    expected.materialization_fingerprint,
                ),
            )
            if cursor.rowcount != 1:
                raise AgentSnapshotStoreConflict("Materialization progress compare-and-set failed.")
        return document

    def _claim_materialization_operation_sync(
        self,
        progress: AgentSnapshotMaterializationProgress,
        operation_id: str,
    ) -> str:
        return self._transition_materialization_progress_sync(
            progress,
            _claim_progress(progress, operation_id),
        )

    def _complete_materialization_operation_sync(
        self,
        progress: AgentSnapshotMaterializationProgress,
        operation_id: str,
        component: AgentSnapshotMaterializedComponent,
    ) -> str:
        return self._transition_materialization_progress_sync(
            progress,
            _complete_progress(progress, operation_id, component),
        )

    def _finalize_materialization_sync(
        self,
        progress: AgentSnapshotMaterializationProgress,
        materialization: AgentSnapshotMaterialization,
    ) -> str:
        validated = cast("AgentSnapshotMaterialization", _validate_store_record(materialization))
        progress = cast("AgentSnapshotMaterializationProgress", _validate_store_record(progress))
        with self._connect() as connection:
            current_row = connection.execute(
                "SELECT snapshot_fingerprint, candidate_id, state_scope_id, state_mode, "
                "progress_id, revision, document, materialization_fingerprint "
                "FROM cayu_agent_snapshot_materialization_progress "
                "WHERE snapshot_fingerprint = ? AND candidate_id = ? "
                "AND state_scope_id = ? AND state_mode = ?",
                _scope_key(progress),
            ).fetchone()
        if current_row is None:
            raise AgentSnapshotStoreConflict("Materialization scope progress is missing.")
        current = _validated_sqlite_progress_row(
            current_row,
            expected_scope=_scope_key(progress),
        )
        if progress.materialization_fingerprint is not None:
            if current != progress or progress.materialization_fingerprint != validated.fingerprint:
                raise AgentSnapshotStoreConflict(
                    "Materialization scope is already finalized with another identity."
                )
            existing_document = self._load_sync(
                "materialization",
                validated.fingerprint,
                AgentSnapshotMaterialization,
            )
            if existing_document is None:
                raise AgentSnapshotStoreConflict("Finalized materialization record is missing.")
            existing = AgentSnapshotMaterialization.model_validate_json(existing_document)
            if not _same_record_identity(existing, validated):
                raise AgentSnapshotStoreConflict(
                    "Finalized materialization record has a conflicting identity."
                )
            return existing_document
        updated = _final_progress(progress, validated)
        expected_progress_document = json.dumps(
            progress.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        progress_document = json.dumps(
            updated.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        materialization_document = json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO cayu_agent_snapshot_records "
                "(record_kind, fingerprint, document) VALUES (?, ?, ?)",
                ("materialization", validated.fingerprint, materialization_document),
            )
            record_row = connection.execute(
                "SELECT document FROM cayu_agent_snapshot_records "
                "WHERE record_kind = 'materialization' AND fingerprint = ?",
                (validated.fingerprint,),
            ).fetchone()
            if record_row is None:
                raise RuntimeError("SQLite failed to persist the materialization record.")
            stored_document = cast("str", record_row["document"])
            try:
                stored = AgentSnapshotMaterialization.model_validate_json(stored_document)
            except Exception as error:
                raise AgentSnapshotStoreConflict(
                    "Stored materialization is invalid for its fingerprint."
                ) from error
            if not _same_record_identity(stored, validated):
                raise AgentSnapshotStoreConflict(
                    "Materialization fingerprint is bound to another logical identity."
                )
            cursor = connection.execute(
                "UPDATE cayu_agent_snapshot_materialization_progress "
                "SET revision = ?, document = ?, materialization_fingerprint = ? "
                "WHERE snapshot_fingerprint = ? AND candidate_id = ? "
                "AND state_scope_id = ? AND state_mode = ? "
                "AND progress_id = ? AND revision = ? AND document = ? "
                "AND materialization_fingerprint IS NULL",
                (
                    updated.revision,
                    progress_document,
                    validated.fingerprint,
                    *_scope_key(progress),
                    progress.progress_id,
                    progress.revision,
                    expected_progress_document,
                ),
            )
            if cursor.rowcount != 1:
                current_row = connection.execute(
                    "SELECT snapshot_fingerprint, candidate_id, state_scope_id, state_mode, "
                    "progress_id, revision, document, materialization_fingerprint "
                    "FROM cayu_agent_snapshot_materialization_progress "
                    "WHERE snapshot_fingerprint = ? AND candidate_id = ? "
                    "AND state_scope_id = ? AND state_mode = ?",
                    _scope_key(progress),
                ).fetchone()
                if current_row is None:
                    raise AgentSnapshotStoreConflict(
                        "Materialization finalization compare-and-set failed."
                    )
                current = _validated_sqlite_progress_row(
                    current_row,
                    expected_scope=_scope_key(progress),
                )
                if current != updated:
                    raise AgentSnapshotStoreConflict(
                        "Materialization finalization compare-and-set failed."
                    )
        return stored_document

    def _save_materialization_sync(
        self,
        materialization: AgentSnapshotMaterialization,
    ) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT snapshot_fingerprint, candidate_id, state_scope_id, state_mode, "
                "progress_id, revision, document, materialization_fingerprint "
                "FROM cayu_agent_snapshot_materialization_progress "
                "WHERE progress_id = ?",
                (materialization.progress_id,),
            ).fetchone()
        if row is None:
            raise AgentSnapshotStoreConflict(
                "Materialization has no durable pre-effect progress record."
            )
        progress = _validated_sqlite_progress_row(row)
        if progress.materialization_fingerprint is not None:
            if progress.materialization_fingerprint != materialization.fingerprint:
                raise AgentSnapshotStoreConflict(
                    "Materialization scope is already finalized with another identity."
                )
            validated = cast(
                "AgentSnapshotMaterialization", _validate_store_record(materialization)
            )
            _final_progress(progress, validated)
            document = self._load_sync(
                "materialization",
                materialization.fingerprint,
                AgentSnapshotMaterialization,
            )
            if document is None:
                raise AgentSnapshotStoreConflict("Finalized materialization record is missing.")
            existing = AgentSnapshotMaterialization.model_validate_json(document)
            if not _same_record_identity(existing, validated):
                raise AgentSnapshotStoreConflict(
                    "Finalized materialization record has a conflicting identity."
                )
            return document
        return self._finalize_materialization_sync(progress, materialization)

    def _load_materialization_for_scope_sync(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT progress.snapshot_fingerprint, progress.candidate_id, "
                "progress.state_scope_id, progress.state_mode, progress.progress_id, "
                "progress.revision, progress.document, "
                "progress.materialization_fingerprint, "
                "record.document AS materialization_document "
                "FROM cayu_agent_snapshot_materialization_progress AS progress "
                "LEFT JOIN cayu_agent_snapshot_records AS record "
                "ON record.record_kind = 'materialization' "
                "AND record.fingerprint = progress.materialization_fingerprint "
                "WHERE progress.snapshot_fingerprint = ? AND progress.candidate_id = ? "
                "AND progress.state_scope_id = ? AND progress.state_mode = ?",
                _scope_key(request),
            ).fetchone()
        if row is None:
            return None
        progress = _validated_sqlite_progress_row(row, expected_scope=_scope_key(request))
        if progress.materialization_fingerprint is None:
            return None
        materialization_document = row["materialization_document"]
        if materialization_document is None:
            raise AgentSnapshotStoreConflict(
                "Materialization scope points to a missing durable record."
            )
        try:
            materialization = AgentSnapshotMaterialization.model_validate_json(
                materialization_document
            )
        except Exception as error:
            raise AgentSnapshotStoreConflict(
                "Materialization scope points to an invalid durable record."
            ) from error
        _require_scope_materialization(request, progress, materialization)
        return cast("str", materialization_document)

    async def _save(
        self,
        kind: str,
        fingerprint: str,
        value: BaseModel,
        model: type[BaseModel],
    ) -> BaseModel:
        document = await asyncio.to_thread(self._save_sync, kind, fingerprint, value)
        return model.model_validate_json(document)

    async def _load(
        self,
        kind: str,
        fingerprint: str,
        model: type[BaseModel],
    ) -> BaseModel | None:
        document = await asyncio.to_thread(self._load_sync, kind, fingerprint, model)
        return None if document is None else model.model_validate_json(document)

    async def save_snapshot(self, snapshot: AgentSnapshot) -> AgentSnapshot:
        return cast(
            "AgentSnapshot",
            await self._save("snapshot", snapshot.fingerprint, snapshot, AgentSnapshot),
        )

    async def load_snapshot(self, fingerprint: str) -> AgentSnapshot | None:
        return cast(
            "AgentSnapshot | None", await self._load("snapshot", fingerprint, AgentSnapshot)
        )

    async def save_materialization(
        self, materialization: AgentSnapshotMaterialization
    ) -> AgentSnapshotMaterialization:
        document = await asyncio.to_thread(
            self._save_materialization_sync,
            materialization,
        )
        return AgentSnapshotMaterialization.model_validate_json(document)

    async def load_materialization(self, fingerprint: str) -> AgentSnapshotMaterialization | None:
        return cast(
            "AgentSnapshotMaterialization | None",
            await self._load("materialization", fingerprint, AgentSnapshotMaterialization),
        )

    async def load_materialization_for_scope(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterialization | None:
        if type(request) is not AgentSnapshotMaterializationRequest:
            raise TypeError("request must be an AgentSnapshotMaterializationRequest.")
        document = await asyncio.to_thread(
            self._load_materialization_for_scope_sync,
            request,
        )
        return (
            None if document is None else AgentSnapshotMaterialization.model_validate_json(document)
        )

    async def begin_materialization(
        self,
        progress: AgentSnapshotMaterializationProgress,
    ) -> AgentSnapshotMaterializationProgress:
        document = await asyncio.to_thread(self._begin_materialization_sync, progress)
        return AgentSnapshotMaterializationProgress.model_validate_json(document)

    async def load_materialization_progress_for_scope(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterializationProgress | None:
        if type(request) is not AgentSnapshotMaterializationRequest:
            raise TypeError("request must be an AgentSnapshotMaterializationRequest.")
        document = await asyncio.to_thread(
            self._load_materialization_progress_for_scope_sync,
            request,
        )
        return (
            None
            if document is None
            else AgentSnapshotMaterializationProgress.model_validate_json(document)
        )

    async def claim_materialization_operation(
        self,
        progress: AgentSnapshotMaterializationProgress,
        operation_id: str,
    ) -> AgentSnapshotMaterializationProgress:
        document = await asyncio.to_thread(
            self._claim_materialization_operation_sync,
            progress,
            operation_id,
        )
        return AgentSnapshotMaterializationProgress.model_validate_json(document)

    async def complete_materialization_operation(
        self,
        progress: AgentSnapshotMaterializationProgress,
        operation_id: str,
        component: AgentSnapshotMaterializedComponent,
    ) -> AgentSnapshotMaterializationProgress:
        validated = AgentSnapshotMaterializedComponent.model_validate(
            component.model_dump(mode="json")
        )
        document = await asyncio.to_thread(
            self._complete_materialization_operation_sync,
            progress,
            operation_id,
            validated,
        )
        return AgentSnapshotMaterializationProgress.model_validate_json(document)

    async def finalize_materialization(
        self,
        progress: AgentSnapshotMaterializationProgress,
        materialization: AgentSnapshotMaterialization,
    ) -> AgentSnapshotMaterialization:
        document = await asyncio.to_thread(
            self._finalize_materialization_sync,
            progress,
            materialization,
        )
        return AgentSnapshotMaterialization.model_validate_json(document)

    async def save_trial(self, trial: AgentSnapshotTrialBinding) -> AgentSnapshotTrialBinding:
        return cast(
            "AgentSnapshotTrialBinding",
            await self._save("trial", trial.fingerprint, trial, AgentSnapshotTrialBinding),
        )

    async def load_trial(self, fingerprint: str) -> AgentSnapshotTrialBinding | None:
        return cast(
            "AgentSnapshotTrialBinding | None",
            await self._load("trial", fingerprint, AgentSnapshotTrialBinding),
        )

    async def save_result(self, result: AgentSnapshotResultBinding) -> AgentSnapshotResultBinding:
        return cast(
            "AgentSnapshotResultBinding",
            await self._save("result", result.fingerprint, result, AgentSnapshotResultBinding),
        )

    async def load_result(self, fingerprint: str) -> AgentSnapshotResultBinding | None:
        return cast(
            "AgentSnapshotResultBinding | None",
            await self._load("result", fingerprint, AgentSnapshotResultBinding),
        )


class AgentSnapshotCoordinator:
    """Capture, verify, materialize, and recover through component owners."""

    def __init__(
        self,
        providers: Iterable[AgentSnapshotComponentProvider],
        *,
        store: AgentSnapshotStore | None = None,
        clock=None,
    ) -> None:
        items = tuple(providers)
        if not items:
            raise ValueError("AgentSnapshotCoordinator requires component providers.")
        by_kind: dict[AgentSnapshotComponentKind, AgentSnapshotComponentProvider] = {}
        for provider in items:
            if not isinstance(provider, AgentSnapshotComponentProvider):
                raise TypeError("providers must contain AgentSnapshotComponentProvider values.")
            if provider.kind in by_kind:
                raise ValueError(f"Duplicate snapshot provider for {provider.kind.value}.")
            _clean(provider.provider_id, "provider_id", max_chars=256)
            by_kind[provider.kind] = provider
        self._providers = by_kind
        self.store = store if store is not None else InMemoryAgentSnapshotStore()
        if not isinstance(self.store, AgentSnapshotStore):
            raise TypeError("store must be an AgentSnapshotStore.")
        self._clock = clock if clock is not None else lambda: datetime.now(UTC)
        self._materialization_locks: dict[str, asyncio.Lock] = {}

    async def capture(self, request: AgentSnapshotCaptureRequest) -> AgentSnapshot:
        if type(request) is not AgentSnapshotCaptureRequest:
            raise TypeError("request must be an AgentSnapshotCaptureRequest.")
        captures: list[AgentSnapshotComponentCapture] = []
        for selector in request.components:
            provider = self._providers.get(selector.kind)
            if provider is None:
                raise AgentSnapshotCaptureError("provider_unavailable", component=selector.kind)
            try:
                captured = await provider.capture(request, selector)
            except Exception as error:
                raise AgentSnapshotCaptureError(
                    "provider_capture_failed", component=selector.kind
                ) from error
            if type(captured) is not AgentSnapshotComponentCapture:
                raise AgentSnapshotCaptureError("invalid_provider_result", component=selector.kind)
            try:
                captured = AgentSnapshotComponentCapture.model_validate(
                    captured.model_dump(mode="json")
                )
            except Exception as error:
                raise AgentSnapshotCaptureError(
                    "invalid_provider_result", component=selector.kind
                ) from error
            if captured.component.kind is not selector.kind:
                raise AgentSnapshotCaptureError("component_kind_mismatch", component=selector.kind)
            if captured.component.provider_id != provider.provider_id:
                raise AgentSnapshotCaptureError(
                    "provider_identity_mismatch", component=selector.kind
                )
            if captured.component.required != selector.required:
                raise AgentSnapshotCaptureError(
                    "component_requirement_mismatch", component=selector.kind
                )
            if captured.component.logical.scope_fingerprint not in {
                None,
                request.authority_scope_fingerprint,
            }:
                raise AgentSnapshotCaptureError(
                    "authority_scope_broadened", component=selector.kind
                )
            if selector.required and (
                captured.component.completeness is AgentSnapshotCompleteness.UNAVAILABLE
            ):
                raise AgentSnapshotCaptureError(
                    "required_component_unavailable", component=selector.kind
                )
            captures.append(captured)
        profile = next(
            (
                capture.execution_profile
                for capture in captures
                if capture.execution_profile is not None
            ),
            None,
        )
        if profile is None:
            raise AgentSnapshotCaptureError(
                "execution_profile_unavailable",
                component=AgentSnapshotComponentKind.EXECUTION_PROFILE,
            )
        memory_state = next(
            (capture.memory_state for capture in captures if capture.memory_state is not None),
            None,
        )
        snapshot = AgentSnapshot.create(
            capture_request_id=request.capture_request_id,
            captured_at=self._now(),
            subject=request.subject,
            authority_scope_fingerprint=request.authority_scope_fingerprint,
            execution_profile=profile,
            memory_state=memory_state,
            components=(capture.component for capture in captures),
            evaluator=request.evaluator,
            promotion_authority=request.promotion_authority,
            parent_snapshot_fingerprint=request.parent_snapshot_fingerprint,
            lineage=request.lineage,
        )
        if (
            _CONSISTENCY_RANK[snapshot.consistency]
            < _CONSISTENCY_RANK[request.required_consistency]
        ):
            raise AgentSnapshotCaptureError("required_consistency_unavailable")
        await self.verify(snapshot)
        stored = cast(
            "AgentSnapshot",
            _validate_store_response(
                await self.store.save_snapshot(snapshot),
                AgentSnapshot,
                "snapshot save",
            ),
        )
        if stored.fingerprint != snapshot.fingerprint or not _same_record_identity(
            stored, snapshot
        ):
            raise AgentSnapshotStoreConflict("Snapshot save returned another logical identity.")
        return stored

    async def verify(self, snapshot: AgentSnapshot) -> AgentSnapshot:
        if type(snapshot) is not AgentSnapshot:
            raise TypeError("snapshot must be an AgentSnapshot.")
        validated = AgentSnapshot.model_validate(snapshot.model_dump(mode="json"))
        for component in validated.components:
            provider = self._providers.get(component.kind)
            if provider is None:
                if component.required or (
                    component.completeness is not AgentSnapshotCompleteness.UNAVAILABLE
                ):
                    raise AgentSnapshotVerificationError(
                        f"Provider unavailable for {component.kind.value} component."
                    )
                continue
            if provider.provider_id != component.provider_id:
                raise AgentSnapshotVerificationError(
                    f"Provider identity changed for {component.kind.value} component."
                )
            try:
                verified = await provider.verify(validated, component)
            except Exception as error:
                raise AgentSnapshotVerificationError(
                    f"Verification failed for {component.kind.value} component."
                ) from error
            if type(verified) is not bool or not verified:
                raise AgentSnapshotVerificationError(
                    f"Verification rejected {component.kind.value} component."
                )
        return validated

    async def materialize(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterialization:
        if type(request) is not AgentSnapshotMaterializationRequest:
            raise TypeError("request must be an AgentSnapshotMaterializationRequest.")
        lock = self._materialization_locks.setdefault(request.state_scope_id, asyncio.Lock())
        async with lock:
            return await self._materialize_or_resume(request)

    async def _materialize_or_resume(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterialization:
        loaded_snapshot = await self.store.load_snapshot(request.snapshot_fingerprint)
        if loaded_snapshot is None:
            raise AgentSnapshotMaterializationError("Starting snapshot is unavailable.")
        snapshot = cast(
            "AgentSnapshot",
            _validate_store_response(loaded_snapshot, AgentSnapshot, "snapshot load"),
        )
        if snapshot.fingerprint != request.snapshot_fingerprint:
            raise AgentSnapshotMaterializationError("Starting snapshot identity changed.")
        await self.verify(snapshot)
        planned_components: list[AgentSnapshotComponentRef] = []
        for component in snapshot.components:
            provider = self._providers.get(component.kind)
            if provider is None:
                if component.required or (
                    component.completeness is not AgentSnapshotCompleteness.UNAVAILABLE
                ):
                    raise AgentSnapshotMaterializationError(
                        f"Provider unavailable for {component.kind.value} component."
                    )
                continue
            planned_components.append(component)
        expected_progress = AgentSnapshotMaterializationProgress.create(
            request=request,
            created_at=self._now(),
            components=planned_components,
        )
        progress = cast(
            "AgentSnapshotMaterializationProgress",
            _validate_store_response(
                await self.store.begin_materialization(expected_progress),
                AgentSnapshotMaterializationProgress,
                "materialization begin",
            ),
        )
        if progress.progress_id != expected_progress.progress_id or not _same_record_identity(
            progress, expected_progress
        ):
            raise AgentSnapshotStoreConflict(
                "Materialization scope is bound to a snapshot-inconsistent operation plan."
            )
        return await self._resume_materialization(snapshot, request, progress)

    async def _resume_materialization(
        self,
        snapshot: AgentSnapshot,
        request: AgentSnapshotMaterializationRequest,
        progress: AgentSnapshotMaterializationProgress,
    ) -> AgentSnapshotMaterialization:
        while True:
            if progress.materialization_fingerprint is not None:
                recovered = await self.recover_materialization(progress.materialization_fingerprint)
                _require_scope_materialization(request, progress, recovered)
                return recovered
            completed_kinds = {component.kind for component in progress.components}
            if len(completed_kinds) == len(progress.operations):
                value = AgentSnapshotMaterialization.create(
                    progress_id=progress.progress_id,
                    request=request,
                    created_at=progress.created_at,
                    components=progress.components,
                )
                expected_final_progress = _final_progress(progress, value)
                try:
                    finalized = cast(
                        "AgentSnapshotMaterialization",
                        _validate_store_response(
                            await self.store.finalize_materialization(progress, value),
                            AgentSnapshotMaterialization,
                            "materialization finalization",
                        ),
                    )
                    if finalized.fingerprint != value.fingerprint or not _same_record_identity(
                        finalized, value
                    ):
                        raise AgentSnapshotStoreConflict(
                            "Materialization finalization returned another logical identity."
                        )
                    recovered = await self.recover_materialization(finalized.fingerprint)
                    _require_scope_materialization(
                        request,
                        expected_final_progress,
                        recovered,
                    )
                    return recovered
                except AgentSnapshotStoreConflict:
                    refreshed_value = await self.store.load_materialization_progress_for_scope(
                        request
                    )
                    if refreshed_value is None:
                        raise
                    refreshed = cast(
                        "AgentSnapshotMaterializationProgress",
                        _validate_store_response(
                            refreshed_value,
                            AgentSnapshotMaterializationProgress,
                            "materialization progress load",
                        ),
                    )
                    _require_progress_successor(progress, refreshed)
                    progress = refreshed
                    continue

            recovering = progress.active_operation_id is not None
            if recovering:
                operation = next(
                    item
                    for item in progress.operations
                    if item.operation_id == progress.active_operation_id
                )
            else:
                operation = next(
                    item
                    for item in progress.operations
                    if item.component_kind not in completed_kinds
                )
                expected_claim = _claim_progress(progress, operation.operation_id)
                try:
                    claimed = cast(
                        "AgentSnapshotMaterializationProgress",
                        _validate_store_response(
                            await self.store.claim_materialization_operation(
                                progress,
                                operation.operation_id,
                            ),
                            AgentSnapshotMaterializationProgress,
                            "materialization claim",
                        ),
                    )
                    if claimed != expected_claim:
                        raise AgentSnapshotStoreConflict(
                            "Materialization claim returned an unexpected durable state."
                        )
                    progress = claimed
                except AgentSnapshotStoreConflict:
                    refreshed_value = await self.store.load_materialization_progress_for_scope(
                        request
                    )
                    if refreshed_value is None:
                        raise
                    refreshed = cast(
                        "AgentSnapshotMaterializationProgress",
                        _validate_store_response(
                            refreshed_value,
                            AgentSnapshotMaterializationProgress,
                            "materialization progress load",
                        ),
                    )
                    _require_progress_successor(progress, refreshed)
                    progress = refreshed
                    continue

            component = snapshot.component(operation.component_kind)
            provider = self._providers.get(operation.component_kind)
            if provider is None or provider.provider_id != operation.provider_id:
                raise AgentSnapshotMaterializationError(
                    f"Provider unavailable for {operation.component_kind.value} materialization."
                )
            try:
                if recovering:
                    result = await provider.recover_materialization_operation(
                        snapshot,
                        component,
                        request,
                        operation,
                    )
                else:
                    result = await provider.materialize(
                        snapshot,
                        component,
                        request,
                        operation,
                    )
            except Exception as error:
                action = "recover" if recovering else "materialize"
                raise AgentSnapshotMaterializationError(
                    f"Provider failed to {action} {component.kind.value} component."
                ) from error
            if type(result) is not AgentSnapshotMaterializedComponent:
                raise AgentSnapshotMaterializationError(
                    f"Provider returned invalid {component.kind.value} materialization."
                )
            try:
                result = AgentSnapshotMaterializedComponent.model_validate(
                    result.model_dump(mode="json")
                )
            except Exception as error:
                raise AgentSnapshotMaterializationError(
                    f"Provider returned invalid {component.kind.value} materialization."
                ) from error
            self._validate_materialized_component(
                snapshot,
                component,
                candidate_id=request.candidate_id,
                state_scope_id=request.state_scope_id,
                result=result,
            )
            expected_completion = _complete_progress(
                progress,
                operation.operation_id,
                result,
            )
            try:
                completed = cast(
                    "AgentSnapshotMaterializationProgress",
                    _validate_store_response(
                        await self.store.complete_materialization_operation(
                            progress,
                            operation.operation_id,
                            result,
                        ),
                        AgentSnapshotMaterializationProgress,
                        "materialization completion",
                    ),
                )
                if completed != expected_completion:
                    raise AgentSnapshotStoreConflict(
                        "Materialization completion returned an unexpected durable state."
                    )
                progress = completed
            except AgentSnapshotStoreConflict:
                refreshed_value = await self.store.load_materialization_progress_for_scope(request)
                if refreshed_value is None:
                    raise
                refreshed = cast(
                    "AgentSnapshotMaterializationProgress",
                    _validate_store_response(
                        refreshed_value,
                        AgentSnapshotMaterializationProgress,
                        "materialization progress load",
                    ),
                )
                _require_progress_successor(progress, refreshed)
                durable = next(
                    (
                        item
                        for item in refreshed.components
                        if item.kind is operation.component_kind
                    ),
                    None,
                )
                if durable is None or durable.identity_material() != result.identity_material():
                    raise
                progress = refreshed

    async def recover_materialization(
        self,
        fingerprint: str,
    ) -> AgentSnapshotMaterialization:
        _sha256_hex(fingerprint, "fingerprint")
        materialization = await self.store.load_materialization(fingerprint)
        if materialization is None:
            raise AgentSnapshotMaterializationError("Materialization is unavailable.")
        try:
            materialization = AgentSnapshotMaterialization.model_validate(
                materialization.model_dump(mode="json")
            )
        except Exception as error:
            raise AgentSnapshotMaterializationError("Materialization record is invalid.") from error
        if materialization.fingerprint != fingerprint:
            raise AgentSnapshotMaterializationError("Materialization identity changed.")
        snapshot = await self.store.load_snapshot(materialization.snapshot_fingerprint)
        if snapshot is None:
            raise AgentSnapshotMaterializationError("Starting snapshot is unavailable.")
        if snapshot.fingerprint != materialization.snapshot_fingerprint:
            raise AgentSnapshotMaterializationError("Starting snapshot identity changed.")
        await self.verify(snapshot)
        recovered: list[AgentSnapshotMaterializedComponent] = []
        by_kind = {component.kind: component for component in materialization.components}
        expected_components = self._validate_materialization_against_snapshot(
            snapshot,
            materialization,
        )
        for component in expected_components:
            existing = by_kind[component.kind]
            provider = self._providers.get(component.kind)
            assert provider is not None
            try:
                value = await provider.recover(
                    snapshot,
                    component,
                    existing,
                    materialization,
                )
            except Exception as error:
                raise AgentSnapshotMaterializationError(
                    f"Provider failed to recover {component.kind.value} component."
                ) from error
            if type(value) is not AgentSnapshotMaterializedComponent:
                raise AgentSnapshotMaterializationError(
                    f"Provider returned invalid {component.kind.value} recovery."
                )
            try:
                value = AgentSnapshotMaterializedComponent.model_validate(
                    value.model_dump(mode="json")
                )
            except Exception as error:
                raise AgentSnapshotMaterializationError(
                    f"Provider returned invalid {component.kind.value} recovery."
                ) from error
            self._validate_materialized_component(
                snapshot,
                component,
                candidate_id=materialization.candidate_id,
                state_scope_id=materialization.state_scope_id,
                result=value,
            )
            recovered.append(value)
        # Accumulating candidates derive the same scope without a trial id. A
        # reset-each-trial materialization cannot be reconstructed from a made-up
        # trial id, so retain and directly validate its durable identity instead.
        if materialization.state_mode is AgentSnapshotTrialStateMode.RESET_EACH_TRIAL:
            candidate = AgentSnapshotMaterialization.model_validate(
                materialization.model_copy(update={"components": tuple(recovered)}).model_dump(
                    mode="json"
                )
            )
        else:
            candidate = AgentSnapshotMaterialization.create(
                progress_id=materialization.progress_id,
                request=AgentSnapshotMaterializationRequest(
                    snapshot_fingerprint=materialization.snapshot_fingerprint,
                    candidate_id=materialization.candidate_id,
                    trial_id="recovery",
                    state_mode=materialization.state_mode,
                ),
                created_at=materialization.created_at,
                components=recovered,
            )
        if candidate.fingerprint != materialization.fingerprint:
            raise AgentSnapshotMaterializationError(
                "Recovered component identities changed the materialization."
            )
        return candidate

    async def begin_trial(
        self,
        materialization: AgentSnapshotMaterialization,
        *,
        case_id: str,
        trial_id: str,
        evaluator_fingerprint: str,
    ) -> AgentSnapshotTrialBinding:
        if type(materialization) is not AgentSnapshotMaterialization:
            raise TypeError("materialization must be an AgentSnapshotMaterialization.")
        materialization = AgentSnapshotMaterialization.model_validate(
            materialization.model_dump(mode="json")
        )
        loaded = await self.store.load_materialization(materialization.fingerprint)
        if loaded is None:
            raise AgentSnapshotMaterializationError(
                "Trial requires the exact durable materialization."
            )
        stored = cast(
            "AgentSnapshotMaterialization",
            _validate_store_response(
                loaded,
                AgentSnapshotMaterialization,
                "materialization load",
            ),
        )
        if stored.fingerprint != materialization.fingerprint or not _same_record_identity(
            stored, materialization
        ):
            raise AgentSnapshotMaterializationError(
                "Trial requires the exact durable materialization."
            )
        loaded_snapshot = await self.store.load_snapshot(stored.snapshot_fingerprint)
        if loaded_snapshot is None:
            raise AgentSnapshotMaterializationError("Starting snapshot is unavailable.")
        snapshot = cast(
            "AgentSnapshot",
            _validate_store_response(loaded_snapshot, AgentSnapshot, "snapshot load"),
        )
        if snapshot.fingerprint != stored.snapshot_fingerprint:
            raise AgentSnapshotMaterializationError("Starting snapshot identity changed.")
        await self.verify(snapshot)
        self._validate_materialization_against_snapshot(snapshot, stored)
        trial_request = AgentSnapshotMaterializationRequest(
            snapshot_fingerprint=stored.snapshot_fingerprint,
            candidate_id=stored.candidate_id,
            trial_id=trial_id,
            state_mode=stored.state_mode,
        )
        if trial_request.state_scope_id != stored.state_scope_id:
            raise AgentSnapshotMaterializationError(
                "Trial identity does not match the materialization state scope."
            )
        if snapshot.evaluator is None:
            raise AgentSnapshotMaterializationError(
                "Starting snapshot does not declare an evaluator identity."
            )
        if snapshot.evaluator.identity.fingerprint != evaluator_fingerprint:
            raise AgentSnapshotMaterializationError(
                "Trial evaluator differs from the starting snapshot."
            )
        trial = AgentSnapshotTrialBinding.create(
            materialization=stored,
            case_id=case_id,
            trial_id=trial_id,
            evaluator_fingerprint=evaluator_fingerprint,
            created_at=self._now(),
        )
        saved = cast(
            "AgentSnapshotTrialBinding",
            _validate_store_response(
                await self.store.save_trial(trial),
                AgentSnapshotTrialBinding,
                "trial save",
            ),
        )
        if saved.fingerprint != trial.fingerprint or not _same_record_identity(saved, trial):
            raise AgentSnapshotStoreConflict("Trial save returned another logical identity.")
        return saved

    async def record_result(
        self,
        result: AgentSnapshotResultBinding,
    ) -> AgentSnapshotResultBinding:
        if type(result) is not AgentSnapshotResultBinding:
            raise TypeError("result must be an AgentSnapshotResultBinding.")
        result = AgentSnapshotResultBinding.model_validate(result.model_dump(mode="json"))
        loaded_trial = await self.store.load_trial(result.trial_fingerprint)
        if loaded_trial is None:
            raise AgentSnapshotMaterializationError(
                "Result requires its exact durable trial binding."
            )
        trial = cast(
            "AgentSnapshotTrialBinding",
            _validate_store_response(loaded_trial, AgentSnapshotTrialBinding, "trial load"),
        )
        if trial.fingerprint != result.trial_fingerprint:
            raise AgentSnapshotMaterializationError(
                "Result requires its exact durable trial binding."
            )
        saved = cast(
            "AgentSnapshotResultBinding",
            _validate_store_response(
                await self.store.save_result(result),
                AgentSnapshotResultBinding,
                "result save",
            ),
        )
        if saved.fingerprint != result.fingerprint or not _same_record_identity(saved, result):
            raise AgentSnapshotStoreConflict("Result save returned another logical identity.")
        return saved

    def _validate_materialization_against_snapshot(
        self,
        snapshot: AgentSnapshot,
        materialization: AgentSnapshotMaterialization,
    ) -> tuple[AgentSnapshotComponentRef, ...]:
        expected_components = tuple(
            component
            for component in snapshot.components
            if self._providers.get(component.kind) is not None
        )
        by_kind = {component.kind: component for component in materialization.components}
        if set(by_kind) != {component.kind for component in expected_components}:
            raise AgentSnapshotMaterializationError(
                "Materialization components differ from the verified snapshot plan."
            )
        for component in expected_components:
            self._validate_materialized_component(
                snapshot,
                component,
                candidate_id=materialization.candidate_id,
                state_scope_id=materialization.state_scope_id,
                result=by_kind[component.kind],
            )
        return expected_components

    def _validate_materialized_component(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        *,
        candidate_id: str,
        state_scope_id: str,
        result: AgentSnapshotMaterializedComponent,
    ) -> None:
        if result.kind is not component.kind:
            raise AgentSnapshotMaterializationError("Materialized component kind changed.")
        if result.baseline_fingerprint != component.logical.fingerprint:
            raise AgentSnapshotMaterializationError("Materialized component baseline changed.")
        if result.capability is not component.materialization:
            raise AgentSnapshotMaterializationError(
                "Materialized capability differs from the verified manifest."
            )
        if component.required and (
            result.capability is AgentSnapshotMaterializationCapability.UNAVAILABLE
        ):
            raise AgentSnapshotMaterializationError(
                f"Required {component.kind.value} component is unavailable."
            )
        if result.overlay is not None:
            expected_overlay_kind = {
                AgentSnapshotComponentKind.MEMORY: AgentSnapshotOverlayKind.MEMORY,
                AgentSnapshotComponentKind.WORKSPACE: AgentSnapshotOverlayKind.WORKSPACE,
            }.get(component.kind)
            if result.overlay.kind is not expected_overlay_kind:
                raise AgentSnapshotMaterializationError(
                    "Materialized overlay kind differs from its component."
                )
            if result.overlay.candidate_id != candidate_id:
                raise AgentSnapshotMaterializationError("Overlay candidate identity changed.")
            if result.overlay.state_scope_id != state_scope_id:
                raise AgentSnapshotMaterializationError("Overlay state scope changed.")
            if result.overlay.baseline_fingerprint != component.logical.fingerprint:
                raise AgentSnapshotMaterializationError("Overlay baseline identity changed.")
        if (
            component.kind
            in {
                AgentSnapshotComponentKind.MEMORY,
                AgentSnapshotComponentKind.WORKSPACE,
            }
            and result.capability is AgentSnapshotMaterializationCapability.RESTORABLE
            and (result.overlay is None)
        ):
            raise AgentSnapshotMaterializationError(
                f"Restorable {component.kind.value} requires a private overlay."
            )
        if snapshot.authority_scope_fingerprint != (
            component.logical.scope_fingerprint or snapshot.authority_scope_fingerprint
        ):
            raise AgentSnapshotMaterializationError("Component authority scope changed.")

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("AgentSnapshot clock must return a datetime.")
        return _utc(value, "clock")


def agent_snapshot_to_json(snapshot: AgentSnapshot) -> str:
    validated = AgentSnapshot.model_validate(snapshot.model_dump(mode="json"))
    return json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def agent_snapshot_from_json(document: str | bytes) -> AgentSnapshot:
    if not isinstance(document, str | bytes):
        raise TypeError("document must be JSON text or bytes.")
    if len(document.encode("utf-8") if isinstance(document, str) else document) > (
        AGENT_SNAPSHOT_MAX_BYTES
    ):
        raise ValueError("AgentSnapshot exceeds its portable byte limit.")
    return AgentSnapshot.model_validate_json(document)


__all__ = [
    "AGENT_SNAPSHOT_MAX_BYTES",
    "AGENT_SNAPSHOT_RECORD_TYPE",
    "AGENT_SNAPSHOT_SCHEMA_VERSION",
    "AGENT_SNAPSHOT_TRIAL_METADATA_KEY",
    "AgentSnapshot",
    "AgentSnapshotAuthorityRef",
    "AgentSnapshotCaptureError",
    "AgentSnapshotCaptureRequest",
    "AgentSnapshotCompleteness",
    "AgentSnapshotComponentCapture",
    "AgentSnapshotComponentKind",
    "AgentSnapshotComponentProvider",
    "AgentSnapshotComponentRef",
    "AgentSnapshotComponentSelector",
    "AgentSnapshotConsistency",
    "AgentSnapshotCoordinator",
    "AgentSnapshotExecutionProfileComponent",
    "AgentSnapshotExecutionProfileRef",
    "AgentSnapshotLearningDisposition",
    "AgentSnapshotLogicalRef",
    "AgentSnapshotMaterialization",
    "AgentSnapshotMaterializationCapability",
    "AgentSnapshotMaterializationError",
    "AgentSnapshotMaterializationOperation",
    "AgentSnapshotMaterializationProgress",
    "AgentSnapshotMaterializationRequest",
    "AgentSnapshotMaterializedComponent",
    "AgentSnapshotOverlayKind",
    "AgentSnapshotOverlayRef",
    "AgentSnapshotRedaction",
    "AgentSnapshotResultBinding",
    "AgentSnapshotStore",
    "AgentSnapshotStoreConflict",
    "AgentSnapshotSubject",
    "AgentSnapshotTerminalDisposition",
    "AgentSnapshotTrialBinding",
    "AgentSnapshotTrialStateMode",
    "AgentSnapshotVerificationError",
    "InMemoryAgentSnapshotStore",
    "MemoryStateRef",
    "SQLiteAgentSnapshotStore",
    "agent_snapshot_consistency",
    "agent_snapshot_from_json",
    "agent_snapshot_to_json",
    "app_body_snapshot_ref",
    "execution_profile_snapshot_ref",
    "trajectory_snapshot_ref",
    "workspace_snapshot_ref",
]
