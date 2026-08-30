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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
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

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)

if TYPE_CHECKING:
    from cayu.evals.models import Trajectory
    from cayu.runtime.execution_profiles import ExecutionProfileIdentity
    from cayu.runtime.manifest import AppManifest
    from cayu.workspaces.revisions import WorkspaceRevisionObservation

AGENT_SNAPSHOT_SCHEMA_VERSION = 3
AGENT_SNAPSHOT_MAX_COMPONENTS = 32
AGENT_SNAPSHOT_MAX_LIMITATIONS = 64
AGENT_SNAPSHOT_MAX_BYTES = 1024 * 1024
AGENT_SNAPSHOT_RECORD_TYPE = "cayu.agent-snapshot"
AGENT_SNAPSHOT_NODE_RECORD_TYPE = "cayu.agent-snapshot-node"
AGENT_SNAPSHOT_NODE_SCHEMA_VERSION = 1
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
    ROLE_PROFILE = "role_profile"
    TOOL_CATALOGUE = "tool_catalogue"
    TOOL_EXPOSURE_POLICY = "tool_exposure_policy"
    KNOWLEDGE = "knowledge"
    MEMORY = "memory"
    SESSION = "session"
    WORK_CONTEXT = "work_context"
    RECALL_POLICY = "recall_policy"
    CONTEXT_PROJECTION_POLICY = "context_projection_policy"
    LEARNING_POLICY = "learning_policy"
    WORKSPACE = "workspace"
    ENVIRONMENT = "environment"
    EXTERNAL_BINDINGS = "external_bindings"
    ARTIFACTS = "artifacts"
    STANDING_POLICY = "standing_policy"
    POLICIES = "policies"


class AgentSnapshotNodeKind(StrEnum):
    MANIFEST = "manifest"
    COMPONENT = "component"


class AgentSnapshotRetentionClass(StrEnum):
    TRANSIENT = "transient"
    RUN_EVIDENCE = "run_evidence"
    CANDIDATE = "candidate"
    CHAMPION = "champion"
    RELEASE = "release"
    LEGAL_HOLD = "legal_hold"


class AgentSnapshotProtectionKind(StrEnum):
    ACTIVE = "active"
    OUTCOME_UNKNOWN = "outcome_unknown"
    IMPORTING = "importing"
    EXPORTING = "exporting"
    MATERIALIZING = "materializing"


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

    def merkle_material(self) -> dict[str, object]:
        """Return the typed memory/knowledge content without authorization scope."""

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
            material[field_name] = (
                None
                if reference is None
                else {
                    "fingerprint": reference.fingerprint,
                    "revision": reference.revision,
                    "frontier": reference.frontier,
                }
            )
        return material

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

    def merkle_material(self) -> dict[str, object]:
        """Return content identity without location or authorization metadata."""

        return {
            "kind": self.kind.value,
            "provider_id": self.provider_id,
            "logical": {
                "fingerprint": self.logical.fingerprint,
                "revision": self.logical.revision,
                "frontier": self.logical.frontier,
            },
            "consistency": self.consistency.value,
            "consistency_group": self.consistency_group,
            "completeness": self.completeness.value,
            "redaction": self.redaction.value,
            "materialization": self.materialization.value,
            "required": self.required,
            "limitations": list(self.limitations),
        }


class AgentSnapshotNodeChild(_SnapshotModel):
    relation: AgentSnapshotComponentKind
    node_kind: AgentSnapshotNodeKind
    schema_id: StrictStr = Field(max_length=256)
    digest: StrictStr

    @field_validator("schema_id")
    @classmethod
    def validate_schema_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)


class AgentSnapshotNode(_SnapshotModel):
    """One strict content-addressed node in an AgentSnapshot Merkle DAG."""

    record_type: Literal["cayu.agent-snapshot-node"] = AGENT_SNAPSHOT_NODE_RECORD_TYPE
    schema_version: Literal[1] = AGENT_SNAPSHOT_NODE_SCHEMA_VERSION
    node_kind: AgentSnapshotNodeKind
    schema_id: StrictStr = Field(max_length=256)
    payload: dict[str, object]
    children: tuple[AgentSnapshotNodeChild, ...] = ()
    digest: StrictStr

    @field_validator("schema_id")
    @classmethod
    def validate_schema_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("payload", mode="before")
    @classmethod
    def validate_payload(cls, value: object) -> dict[str, object]:
        return copy_durable_json_object(value, "agent_snapshot_node.payload")

    @field_validator("children", mode="before")
    @classmethod
    def validate_children_array(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("children must be an ordered array.")
        return value

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @model_validator(mode="after")
    def validate_envelope(self) -> AgentSnapshotNode:
        child_keys = tuple(
            (child.relation.value, child.node_kind.value, child.schema_id, child.digest)
            for child in self.children
        )
        if child_keys != tuple(sorted(set(child_keys))):
            raise ValueError("Merkle children must be unique and canonically ordered.")
        relations = tuple(child.relation for child in self.children)
        if len(relations) != len(set(relations)):
            raise ValueError("Merkle child relations must be unique within one node.")
        if self.node_kind is AgentSnapshotNodeKind.COMPONENT and self.children:
            raise ValueError("Component nodes cannot contain child nodes in schema version 1.")
        if self.digest != _content_sha256(self.identity_material(), "agent_snapshot_node"):
            raise ValueError("AgentSnapshotNode digest does not match its canonical envelope.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "node_kind": self.node_kind.value,
            "schema_id": self.schema_id,
            "payload": self.payload,
            "children": [child.model_dump(mode="json") for child in self.children],
        }

    @classmethod
    def create(
        cls,
        *,
        node_kind: AgentSnapshotNodeKind,
        schema_id: str,
        payload: dict[str, object],
        children: Iterable[AgentSnapshotNodeChild] = (),
    ) -> AgentSnapshotNode:
        ordered = tuple(
            sorted(
                children,
                key=lambda child: (
                    child.relation.value,
                    child.node_kind.value,
                    child.schema_id,
                    child.digest,
                ),
            )
        )
        provisional = cls.model_construct(
            node_kind=node_kind,
            schema_id=schema_id,
            payload=copy_durable_json_object(payload, "agent_snapshot_node.payload"),
            children=ordered,
            digest="0" * 64,
        )
        return cls(
            node_kind=node_kind,
            schema_id=schema_id,
            payload=payload,
            children=ordered,
            digest=_content_sha256(provisional.identity_material(), "agent_snapshot_node"),
        )


class AgentSnapshotRef(_SnapshotModel):
    snapshot_root: StrictStr

    @field_validator("snapshot_root")
    @classmethod
    def validate_snapshot_root(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)


class AgentSnapshotIdentityBinding(_SnapshotModel):
    """Immutable authorization-scoped registration of one logical agent at one root."""

    record_type: Literal["cayu.agent-snapshot-identity-binding"] = (
        "cayu.agent-snapshot-identity-binding"
    )
    schema_version: Literal[1] = 1
    binding_id: StrictStr
    subject: AgentSnapshotSubject
    snapshot: AgentSnapshotRef
    authority_scope_fingerprint: StrictStr

    @field_validator("binding_id", "authority_scope_fingerprint")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @model_validator(mode="after")
    def validate_binding(self) -> AgentSnapshotIdentityBinding:
        if self.binding_id != _content_sha256(
            self.identity_material(), "agent_snapshot_identity_binding"
        ):
            raise ValueError("AgentSnapshot identity binding does not match its contents.")
        return self

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
            "snapshot": self.snapshot.model_dump(mode="json"),
            "authority_scope_fingerprint": self.authority_scope_fingerprint,
        }

    @classmethod
    def create(
        cls,
        *,
        subject: AgentSnapshotSubject,
        snapshot: AgentSnapshotRef,
        authority_scope_fingerprint: str,
    ) -> AgentSnapshotIdentityBinding:
        provisional = cls.model_construct(
            binding_id="0" * 64,
            subject=subject,
            snapshot=snapshot,
            authority_scope_fingerprint=authority_scope_fingerprint,
        )
        return cls(
            binding_id=_content_sha256(
                provisional.identity_material(), "agent_snapshot_identity_binding"
            ),
            subject=subject,
            snapshot=snapshot,
            authority_scope_fingerprint=authority_scope_fingerprint,
        )


class AgentSnapshotAccess(_SnapshotModel):
    """Principal-derived constraints for one authorized snapshot registration."""

    snapshot: AgentSnapshotRef
    binding_id: StrictStr
    authority_scope_fingerprint: StrictStr

    @field_validator("binding_id", "authority_scope_fingerprint")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)


class AgentSnapshotPutReceipt(_SnapshotModel):
    record_type: Literal["cayu.agent-snapshot-put-receipt"] = "cayu.agent-snapshot-put-receipt"
    schema_version: Literal[1] = 1
    receipt_id: StrictStr
    snapshot: AgentSnapshotRef
    binding_id: StrictStr
    node_digests: tuple[StrictStr, ...]

    @field_validator("receipt_id", "binding_id")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("node_digests", mode="before")
    @classmethod
    def validate_node_digests_input(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("node_digests must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_receipt(self) -> AgentSnapshotPutReceipt:
        for digest in self.node_digests:
            _sha256_hex(digest, "node_digests")
        if not self.node_digests or self.node_digests != tuple(sorted(set(self.node_digests))):
            raise ValueError("node_digests must be nonempty, unique, and sorted.")
        if self.snapshot.snapshot_root not in self.node_digests:
            raise ValueError("Put receipt must include its root manifest node.")
        material = self.identity_material()
        if self.receipt_id != _content_sha256(material, "agent_snapshot_put_receipt"):
            raise ValueError("AgentSnapshot put receipt does not match its contents.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "snapshot": self.snapshot.model_dump(mode="json"),
            "binding_id": self.binding_id,
            "node_digests": list(self.node_digests),
        }

    @classmethod
    def create(
        cls,
        *,
        snapshot: AgentSnapshotRef,
        binding_id: str,
        node_digests: Iterable[str],
    ) -> AgentSnapshotPutReceipt:
        ordered = tuple(sorted(set(node_digests)))
        provisional = cls.model_construct(
            receipt_id="0" * 64,
            snapshot=snapshot,
            binding_id=binding_id,
            node_digests=ordered,
        )
        return cls(
            receipt_id=_content_sha256(
                provisional.identity_material(), "agent_snapshot_put_receipt"
            ),
            snapshot=snapshot,
            binding_id=binding_id,
            node_digests=ordered,
        )


class AgentSnapshotClosureInspection(_SnapshotModel):
    snapshot: AgentSnapshotRef
    root_manifest_bytes: StrictInt = Field(ge=1)
    logical_closure_bytes: StrictInt = Field(ge=1)
    unique_stored_bytes: StrictInt = Field(ge=0)
    shared_bytes: StrictInt = Field(ge=0)
    object_count: StrictInt = Field(ge=1)
    node_digests: tuple[StrictStr, ...]
    unresolved_external_bindings: tuple[StrictStr, ...] = ()

    @field_validator("node_digests", mode="before")
    @classmethod
    def validate_node_digests_input(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("node_digests must be an ordered array.")
        return value

    @field_validator("unresolved_external_bindings", mode="before")
    @classmethod
    def validate_unresolved(cls, value: object) -> tuple[str, ...]:
        return _ordered_unique_text(value, "unresolved_external_bindings")

    @model_validator(mode="after")
    def validate_metrics(self) -> AgentSnapshotClosureInspection:
        for digest in self.node_digests:
            _sha256_hex(digest, "node_digests")
        if self.node_digests != tuple(sorted(set(self.node_digests))):
            raise ValueError("node_digests must be unique and sorted.")
        if self.snapshot.snapshot_root not in self.node_digests:
            raise ValueError("Closure inspection must include its root node.")
        if self.object_count != len(self.node_digests):
            raise ValueError("object_count must equal the enumerated closure size.")
        if self.unique_stored_bytes + self.shared_bytes != self.logical_closure_bytes:
            raise ValueError("Unique and shared bytes must partition logical closure bytes.")
        return self


class AgentSnapshotPinRequest(_SnapshotModel):
    operation_id: StrictStr = Field(max_length=256)
    access: AgentSnapshotAccess
    owner: StrictStr = Field(max_length=256)
    reason: StrictStr = Field(max_length=256)
    retention_class: AgentSnapshotRetentionClass

    @field_validator("operation_id", "owner", "reason")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class AgentSnapshotPinReceipt(_SnapshotModel):
    record_type: Literal["cayu.agent-snapshot-pin-receipt"] = "cayu.agent-snapshot-pin-receipt"
    schema_version: Literal[1] = 1
    receipt_id: StrictStr
    operation_id: StrictStr = Field(max_length=256)
    pin_id: StrictStr
    snapshot: AgentSnapshotRef
    binding_id: StrictStr
    owner: StrictStr = Field(max_length=256)
    reason: StrictStr = Field(max_length=256)
    retention_class: AgentSnapshotRetentionClass

    @field_validator("receipt_id", "pin_id", "binding_id")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("operation_id", "owner", "reason")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @model_validator(mode="after")
    def validate_receipt(self) -> AgentSnapshotPinReceipt:
        pin_material = {
            "operation_id": self.operation_id,
            "snapshot": self.snapshot.model_dump(mode="json"),
            "binding_id": self.binding_id,
            "owner": self.owner,
            "reason": self.reason,
            "retention_class": self.retention_class.value,
        }
        if self.pin_id != _content_sha256(pin_material, "agent_snapshot_pin"):
            raise ValueError("AgentSnapshot pin_id does not match its pin identity.")
        if self.receipt_id != _content_sha256(
            self.identity_material(), "agent_snapshot_pin_receipt"
        ):
            raise ValueError("AgentSnapshot pin receipt does not match its contents.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "pin_id": self.pin_id,
            "snapshot": self.snapshot.model_dump(mode="json"),
            "binding_id": self.binding_id,
            "owner": self.owner,
            "reason": self.reason,
            "retention_class": self.retention_class.value,
        }

    @classmethod
    def from_request(cls, request: AgentSnapshotPinRequest) -> AgentSnapshotPinReceipt:
        pin_material = {
            "operation_id": request.operation_id,
            "snapshot": request.access.snapshot.model_dump(mode="json"),
            "binding_id": request.access.binding_id,
            "owner": request.owner,
            "reason": request.reason,
            "retention_class": request.retention_class.value,
        }
        pin_id = _content_sha256(pin_material, "agent_snapshot_pin")
        provisional = cls.model_construct(
            receipt_id="0" * 64,
            operation_id=request.operation_id,
            pin_id=pin_id,
            snapshot=request.access.snapshot,
            binding_id=request.access.binding_id,
            owner=request.owner,
            reason=request.reason,
            retention_class=request.retention_class,
        )
        return cls(
            receipt_id=_content_sha256(
                provisional.identity_material(), "agent_snapshot_pin_receipt"
            ),
            operation_id=request.operation_id,
            pin_id=pin_id,
            snapshot=request.access.snapshot,
            binding_id=request.access.binding_id,
            owner=request.owner,
            reason=request.reason,
            retention_class=request.retention_class,
        )


class AgentSnapshotReleaseRequest(_SnapshotModel):
    operation_id: StrictStr = Field(max_length=256)
    access: AgentSnapshotAccess
    pin_id: StrictStr
    owner: StrictStr = Field(max_length=256)
    reason: StrictStr = Field(max_length=256)

    @field_validator("pin_id")
    @classmethod
    def validate_pin_id(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("operation_id", "owner", "reason")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json")


class AgentSnapshotReleaseReceipt(_SnapshotModel):
    record_type: Literal["cayu.agent-snapshot-release-receipt"] = (
        "cayu.agent-snapshot-release-receipt"
    )
    schema_version: Literal[1] = 1
    receipt_id: StrictStr
    operation_id: StrictStr = Field(max_length=256)
    pin_id: StrictStr
    snapshot: AgentSnapshotRef
    binding_id: StrictStr
    owner: StrictStr = Field(max_length=256)
    reason: StrictStr = Field(max_length=256)

    @field_validator("receipt_id", "pin_id", "binding_id")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("operation_id", "owner", "reason")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @model_validator(mode="after")
    def validate_receipt(self) -> AgentSnapshotReleaseReceipt:
        if self.receipt_id != _content_sha256(
            self.identity_material(), "agent_snapshot_release_receipt"
        ):
            raise ValueError("AgentSnapshot release receipt does not match its contents.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "pin_id": self.pin_id,
            "snapshot": self.snapshot.model_dump(mode="json"),
            "binding_id": self.binding_id,
            "owner": self.owner,
            "reason": self.reason,
        }

    @classmethod
    def from_request(cls, request: AgentSnapshotReleaseRequest) -> AgentSnapshotReleaseReceipt:
        provisional = cls.model_construct(
            receipt_id="0" * 64,
            operation_id=request.operation_id,
            pin_id=request.pin_id,
            snapshot=request.access.snapshot,
            binding_id=request.access.binding_id,
            owner=request.owner,
            reason=request.reason,
        )
        return cls(
            receipt_id=_content_sha256(
                provisional.identity_material(), "agent_snapshot_release_receipt"
            ),
            operation_id=request.operation_id,
            pin_id=request.pin_id,
            snapshot=request.access.snapshot,
            binding_id=request.access.binding_id,
            owner=request.owner,
            reason=request.reason,
        )


class AgentSnapshotProtection(_SnapshotModel):
    protection_id: StrictStr
    operation_id: StrictStr = Field(max_length=256)
    access: AgentSnapshotAccess
    kind: AgentSnapshotProtectionKind
    owner: StrictStr = Field(max_length=256)
    reason: StrictStr = Field(max_length=256)

    @field_validator("protection_id")
    @classmethod
    def validate_protection_id(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("operation_id", "owner", "reason")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @model_validator(mode="after")
    def validate_protection(self) -> AgentSnapshotProtection:
        if self.protection_id != _content_sha256(
            self.identity_material(), "agent_snapshot_protection"
        ):
            raise ValueError("AgentSnapshot protection does not match its contents.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "access": self.access.model_dump(mode="json"),
            "kind": self.kind.value,
            "owner": self.owner,
            "reason": self.reason,
        }

    @classmethod
    def create(
        cls,
        *,
        operation_id: str,
        access: AgentSnapshotAccess,
        kind: AgentSnapshotProtectionKind,
        owner: str,
        reason: str,
    ) -> AgentSnapshotProtection:
        provisional = cls.model_construct(
            protection_id="0" * 64,
            operation_id=operation_id,
            access=access,
            kind=kind,
            owner=owner,
            reason=reason,
        )
        return cls(
            protection_id=_content_sha256(
                provisional.identity_material(), "agent_snapshot_protection"
            ),
            operation_id=operation_id,
            access=access,
            kind=kind,
            owner=owner,
            reason=reason,
        )


class AgentSnapshotGCRequest(_SnapshotModel):
    operation_id: StrictStr = Field(max_length=256)
    candidates: tuple[AgentSnapshotAccess, ...] = Field(min_length=1, max_length=1024)

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("candidates", mode="before")
    @classmethod
    def validate_candidates_input(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("candidates must be an ordered array.")
        return value

    @model_validator(mode="after")
    def validate_candidates(self) -> AgentSnapshotGCRequest:
        keys = tuple(_snapshot_access_key(candidate) for candidate in self.candidates)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("GC candidates must be unique and canonically ordered.")
        return self


class AgentSnapshotGCPlan(_SnapshotModel):
    plan_id: StrictStr
    operation_id: StrictStr = Field(max_length=256)
    authorized_candidates: tuple[AgentSnapshotAccess, ...]
    collectable_roots: tuple[StrictStr, ...]
    blocked_roots: tuple[StrictStr, ...]
    node_digests_to_delete: tuple[StrictStr, ...]
    retained_shared_node_digests: tuple[StrictStr, ...]
    bytes_to_delete: StrictInt = Field(ge=0)

    @field_validator("plan_id")
    @classmethod
    def validate_plan_id(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("authorized_candidates", mode="before")
    @classmethod
    def validate_authorized_candidates_input(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise ValueError("authorized_candidates must be an ordered array.")
        return value

    @field_validator(
        "collectable_roots",
        "blocked_roots",
        "node_digests_to_delete",
        "retained_shared_node_digests",
        mode="before",
    )
    @classmethod
    def validate_digest_arrays(cls, value: object, info) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError(f"{info.field_name} must be an ordered array.")
        copied_items: list[str] = []
        for digest in value:
            if type(digest) is not str:
                raise ValueError(f"{info.field_name} must contain strings.")
            copied_items.append(_sha256_hex(digest, info.field_name))
        copied = tuple(copied_items)
        if copied != tuple(sorted(set(copied))):
            raise ValueError(f"{info.field_name} must be unique and sorted.")
        return copied

    @model_validator(mode="after")
    def validate_plan(self) -> AgentSnapshotGCPlan:
        candidate_keys = tuple(
            _snapshot_access_key(access) for access in self.authorized_candidates
        )
        if not candidate_keys or candidate_keys != tuple(sorted(set(candidate_keys))):
            raise ValueError("GC authorized candidates must be nonempty, unique, and sorted.")
        if set(self.collectable_roots) & set(self.blocked_roots):
            raise ValueError("A GC root cannot be both collectable and blocked.")
        candidate_roots = {
            candidate.snapshot.snapshot_root for candidate in self.authorized_candidates
        }
        if candidate_roots != set(self.collectable_roots) | set(self.blocked_roots):
            raise ValueError("GC root disposition must cover every authorized candidate root.")
        if set(self.node_digests_to_delete) & set(self.retained_shared_node_digests):
            raise ValueError("A GC node cannot be both deleted and retained.")
        if self.plan_id != _content_sha256(self.identity_material(), "agent_snapshot_gc_plan"):
            raise ValueError("AgentSnapshot GC plan does not match its contents.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "authorized_candidates": [
                candidate.model_dump(mode="json") for candidate in self.authorized_candidates
            ],
            "collectable_roots": list(self.collectable_roots),
            "blocked_roots": list(self.blocked_roots),
            "node_digests_to_delete": list(self.node_digests_to_delete),
            "retained_shared_node_digests": list(self.retained_shared_node_digests),
            "bytes_to_delete": self.bytes_to_delete,
        }


class AgentSnapshotGCReceipt(_SnapshotModel):
    receipt_id: StrictStr
    plan_id: StrictStr
    operation_id: StrictStr = Field(max_length=256)
    deleted_roots: tuple[StrictStr, ...]
    deleted_node_digests: tuple[StrictStr, ...]
    deleted_bytes: StrictInt = Field(ge=0)

    @field_validator("receipt_id", "plan_id")
    @classmethod
    def validate_digests(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @field_validator("deleted_roots", "deleted_node_digests", mode="before")
    @classmethod
    def validate_digest_arrays(cls, value: object, info) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError(f"{info.field_name} must be an ordered array.")
        copied_items: list[str] = []
        for digest in value:
            if type(digest) is not str:
                raise ValueError(f"{info.field_name} must contain strings.")
            copied_items.append(_sha256_hex(digest, info.field_name))
        copied = tuple(copied_items)
        if copied != tuple(sorted(set(copied))):
            raise ValueError(f"{info.field_name} must be unique and sorted.")
        return copied

    @model_validator(mode="after")
    def validate_receipt(self) -> AgentSnapshotGCReceipt:
        if self.receipt_id != _content_sha256(
            self.identity_material(), "agent_snapshot_gc_receipt"
        ):
            raise ValueError("AgentSnapshot GC receipt does not match its contents.")
        return self

    def identity_material(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "operation_id": self.operation_id,
            "deleted_roots": list(self.deleted_roots),
            "deleted_node_digests": list(self.deleted_node_digests),
            "deleted_bytes": self.deleted_bytes,
        }


class AgentSnapshotAuthorityRef(_SnapshotModel):
    identity: AgentSnapshotLogicalRef
    candidate_visible: Literal[False] = False


class AgentSnapshot(_SnapshotModel):
    """Strict portable manifest for one bounded logical agent state."""

    record_type: Literal["cayu.agent-snapshot"] = AGENT_SNAPSHOT_RECORD_TYPE
    schema_version: Literal[3] = AGENT_SNAPSHOT_SCHEMA_VERSION
    snapshot_root: StrictStr
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

    @field_validator("snapshot_root", "authority_scope_fingerprint", "parent_snapshot_fingerprint")
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
        if any(
            component.logical.scope_fingerprint not in {None, self.authority_scope_fingerprint}
            for component in self.components
        ):
            raise ValueError("Snapshot components cannot broaden the authority scope.")
        if self.subject.body_release.scope_fingerprint not in {
            None,
            self.authority_scope_fingerprint,
        }:
            raise ValueError("Snapshot body registration cannot broaden the authority scope.")
        body = self.component(AgentSnapshotComponentKind.BODY)
        if (
            body.logical.fingerprint,
            body.logical.revision,
            body.logical.frontier,
        ) != (
            self.subject.body_release.fingerprint,
            self.subject.body_release.revision,
            self.subject.body_release.frontier,
        ):
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
        if self.snapshot_root != self.root_node().digest:
            raise ValueError("AgentSnapshot snapshot_root does not match its Merkle manifest.")
        encoded = canonical_durable_json_bytes(self.model_dump(mode="json"), "agent_snapshot")
        if len(encoded) > AGENT_SNAPSHOT_MAX_BYTES:
            raise ValueError("AgentSnapshot exceeds its portable byte limit.")
        return self

    def component(self, kind: AgentSnapshotComponentKind) -> AgentSnapshotComponentRef:
        for component in self.components:
            if component.kind is kind:
                return component
        raise KeyError(f"Snapshot has no {kind.value} component.")

    @property
    def fingerprint(self) -> str:
        """Deprecated read-only spelling retained for v2 source compatibility."""

        return self.snapshot_root

    @property
    def ref(self) -> AgentSnapshotRef:
        return AgentSnapshotRef(snapshot_root=self.snapshot_root)

    @property
    def identity_binding(self) -> AgentSnapshotIdentityBinding:
        return AgentSnapshotIdentityBinding.create(
            subject=self.subject,
            snapshot=self.ref,
            authority_scope_fingerprint=self.authority_scope_fingerprint,
        )

    def component_node(self, component: AgentSnapshotComponentRef) -> AgentSnapshotNode:
        payload = component.merkle_material()
        if component.kind is AgentSnapshotComponentKind.EXECUTION_PROFILE:
            payload["execution_profile"] = self.execution_profile.model_dump(mode="json")
        if component.kind is AgentSnapshotComponentKind.MEMORY:
            payload["memory_state"] = (
                None if self.memory_state is None else self.memory_state.merkle_material()
            )
        return AgentSnapshotNode.create(
            node_kind=AgentSnapshotNodeKind.COMPONENT,
            schema_id=f"cayu.agent-snapshot.component.{component.kind.value}.v1",
            payload=payload,
        )

    def component_nodes(self) -> tuple[AgentSnapshotNode, ...]:
        return tuple(self.component_node(component) for component in self.components)

    def root_node(self) -> AgentSnapshotNode:
        children = tuple(
            AgentSnapshotNodeChild(
                relation=component.kind,
                node_kind=AgentSnapshotNodeKind.COMPONENT,
                schema_id=node.schema_id,
                digest=node.digest,
            )
            for component, node in zip(self.components, self.component_nodes(), strict=True)
        )
        return AgentSnapshotNode.create(
            node_kind=AgentSnapshotNodeKind.MANIFEST,
            schema_id="cayu.agent-snapshot.manifest.v3",
            payload=self.identity_material(),
            children=children,
        )

    def merkle_nodes(self) -> tuple[AgentSnapshotNode, ...]:
        components = self.component_nodes()
        return (*components, self.root_node())

    def identity_material(self) -> dict[str, object]:
        """Return content-only root payload.

        Logical registration, authority scope, capture metadata, evaluator and
        promotion authority, and provenance deliberately live outside the root.
        """

        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "consistency": self.consistency.value,
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
            snapshot_root="0" * 64,
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
            snapshot_root=provisional.root_node().digest,
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
    access: AgentSnapshotAccess
    candidate_id: StrictStr = Field(max_length=256)
    trial_id: StrictStr = Field(max_length=256)
    state_mode: AgentSnapshotTrialStateMode
    state_partition_fingerprint: StrictStr | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("state_partition_fingerprint")
    @classmethod
    def validate_state_partition_fingerprint(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("candidate_id", "trial_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name, max_chars=256)

    @property
    def snapshot_fingerprint(self) -> str:
        return self.access.snapshot.snapshot_root

    @classmethod
    def derive_state_scope_id(
        cls,
        *,
        snapshot_fingerprint: str,
        candidate_id: str,
        trial_id: str,
        state_mode: AgentSnapshotTrialStateMode,
        state_partition_fingerprint: str | None = None,
    ) -> str:
        snapshot_fingerprint = _sha256_hex(snapshot_fingerprint, "snapshot_fingerprint")
        candidate_id = _clean(candidate_id, "candidate_id", max_chars=256)
        trial_id = _clean(trial_id, "trial_id", max_chars=256)
        if not isinstance(state_mode, AgentSnapshotTrialStateMode):
            raise TypeError("state_mode must be an AgentSnapshotTrialStateMode.")
        if state_partition_fingerprint is not None:
            state_partition_fingerprint = _sha256_hex(
                state_partition_fingerprint,
                "state_partition_fingerprint",
            )
        material: dict[str, object] = {
            "snapshot_fingerprint": snapshot_fingerprint,
            "candidate_id": candidate_id,
            "trial_id": (
                trial_id if state_mode is AgentSnapshotTrialStateMode.RESET_EACH_TRIAL else None
            ),
            "state_mode": state_mode.value,
        }
        if state_partition_fingerprint is not None:
            material["state_partition_fingerprint"] = state_partition_fingerprint
        return _content_sha256(material, "snapshot_state_scope")

    @property
    def state_scope_id(self) -> str:
        return self.derive_state_scope_id(
            snapshot_fingerprint=self.snapshot_fingerprint,
            candidate_id=self.candidate_id,
            trial_id=self.trial_id,
            state_mode=self.state_mode,
            state_partition_fingerprint=self.state_partition_fingerprint,
        )


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
    state_partition_fingerprint: StrictStr | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    component_kind: AgentSnapshotComponentKind
    provider_id: StrictStr = Field(max_length=256)
    baseline_fingerprint: StrictStr
    capability: AgentSnapshotMaterializationCapability

    @field_validator(
        "operation_id",
        "snapshot_fingerprint",
        "state_partition_fingerprint",
        "baseline_fingerprint",
    )
    @classmethod
    def validate_fingerprints(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
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
        material: dict[str, object] = {
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
        if self.state_partition_fingerprint is not None:
            material["state_partition_fingerprint"] = self.state_partition_fingerprint
        return material

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
            state_partition_fingerprint=request.state_partition_fingerprint,
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
            state_partition_fingerprint=request.state_partition_fingerprint,
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
    state_partition_fingerprint: StrictStr | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    created_at: datetime
    components: tuple[AgentSnapshotMaterializedComponent, ...]

    @field_validator(
        "fingerprint",
        "progress_id",
        "snapshot_fingerprint",
        "state_partition_fingerprint",
    )
    @classmethod
    def validate_fingerprint(cls, value: str | None, info) -> str | None:
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
        material: dict[str, object] = {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "progress_id": self.progress_id,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "candidate_id": self.candidate_id,
            "state_scope_id": self.state_scope_id,
            "state_mode": self.state_mode.value,
            "components": [component.identity_material() for component in self.components],
        }
        if self.state_partition_fingerprint is not None:
            material["state_partition_fingerprint"] = self.state_partition_fingerprint
        return material

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
            state_partition_fingerprint=request.state_partition_fingerprint,
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
            state_partition_fingerprint=request.state_partition_fingerprint,
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
    state_partition_fingerprint: StrictStr | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    created_at: datetime
    operations: tuple[AgentSnapshotMaterializationOperation, ...]
    revision: StrictInt = Field(ge=0)
    active_operation_id: StrictStr | None = None
    components: tuple[AgentSnapshotMaterializedComponent, ...] = ()
    materialization_fingerprint: StrictStr | None = None

    @field_validator(
        "progress_id",
        "snapshot_fingerprint",
        "state_partition_fingerprint",
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
                or operation.state_partition_fingerprint != self.state_partition_fingerprint
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
        material: dict[str, object] = {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "snapshot_fingerprint": self.snapshot_fingerprint,
            "candidate_id": self.candidate_id,
            "state_scope_id": self.state_scope_id,
            "state_mode": self.state_mode.value,
            "operations": [operation.identity_material() for operation in self.operations],
        }
        if self.state_partition_fingerprint is not None:
            material["state_partition_fingerprint"] = self.state_partition_fingerprint
        return material

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
            state_partition_fingerprint=request.state_partition_fingerprint,
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
            state_partition_fingerprint=request.state_partition_fingerprint,
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


class AgentSnapshotAuthorizationError(RuntimeError):
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
    async def put_snapshot(
        self,
        snapshot: AgentSnapshot,
        binding: AgentSnapshotIdentityBinding,
    ) -> AgentSnapshotPutReceipt:
        """Atomically store one root closure and one authorized logical binding."""

    @abstractmethod
    async def load_identity_binding(self, binding_id: str) -> AgentSnapshotIdentityBinding | None:
        pass

    @abstractmethod
    async def get_snapshot(self, access: AgentSnapshotAccess) -> AgentSnapshot:
        """Load a snapshot only after validating its logical binding and scope."""

    @abstractmethod
    async def enumerate_snapshot_closure(
        self, access: AgentSnapshotAccess
    ) -> tuple[AgentSnapshotNode, ...]:
        pass

    @abstractmethod
    async def inspect_snapshot(self, access: AgentSnapshotAccess) -> AgentSnapshotClosureInspection:
        pass

    @abstractmethod
    async def pin_snapshot(self, request: AgentSnapshotPinRequest) -> AgentSnapshotPinReceipt:
        pass

    @abstractmethod
    async def release_snapshot_pin(
        self, request: AgentSnapshotReleaseRequest
    ) -> AgentSnapshotReleaseReceipt:
        pass

    @abstractmethod
    async def protect_snapshot(
        self, protection: AgentSnapshotProtection
    ) -> AgentSnapshotProtection:
        pass

    @abstractmethod
    async def release_snapshot_protection(
        self,
        *,
        operation_id: str,
        access: AgentSnapshotAccess,
        protection_id: str,
    ) -> AgentSnapshotProtection:
        pass

    @abstractmethod
    async def plan_snapshot_gc(self, request: AgentSnapshotGCRequest) -> AgentSnapshotGCPlan:
        pass

    @abstractmethod
    async def execute_snapshot_gc(self, plan: AgentSnapshotGCPlan) -> AgentSnapshotGCReceipt:
        pass

    @abstractmethod
    async def save_snapshot(self, snapshot: AgentSnapshot) -> AgentSnapshot:
        pass

    @abstractmethod
    async def load_snapshot(self, snapshot_root: str) -> AgentSnapshot | None:
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


def _snapshot_model_bytes(value: BaseModel, field_name: str) -> bytes:
    return canonical_durable_json_bytes(value.model_dump(mode="json"), field_name)


def _snapshot_model_json(value: BaseModel) -> str:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_snapshot_binding(
    snapshot: AgentSnapshot,
    binding: AgentSnapshotIdentityBinding,
) -> tuple[AgentSnapshot, AgentSnapshotIdentityBinding]:
    try:
        validated_snapshot = AgentSnapshot.model_validate(snapshot.model_dump(mode="json"))
        validated_binding = AgentSnapshotIdentityBinding.model_validate(
            binding.model_dump(mode="json")
        )
    except Exception as error:
        raise AgentSnapshotStoreConflict(
            "Snapshot fingerprint/root put contains an invalid record."
        ) from error
    if validated_binding.snapshot.snapshot_root != validated_snapshot.snapshot_root:
        raise AgentSnapshotStoreConflict("Snapshot binding names another root.")
    if validated_binding.authority_scope_fingerprint != (
        validated_snapshot.authority_scope_fingerprint
    ):
        raise AgentSnapshotStoreConflict("Snapshot binding names another authority scope.")
    if validated_binding.subject != validated_snapshot.subject:
        raise AgentSnapshotStoreConflict("Snapshot binding names another logical subject.")
    return validated_snapshot, validated_binding


def _validate_snapshot_access(
    access: AgentSnapshotAccess,
    binding: AgentSnapshotIdentityBinding | None,
) -> AgentSnapshotIdentityBinding:
    if type(access) is not AgentSnapshotAccess:
        raise TypeError("access must be an AgentSnapshotAccess.")
    validated_access = AgentSnapshotAccess.model_validate(access.model_dump(mode="json"))
    if binding is None:
        raise AgentSnapshotAuthorizationError("Snapshot identity binding is unavailable.")
    try:
        validated_binding = AgentSnapshotIdentityBinding.model_validate(
            binding.model_dump(mode="json")
        )
    except Exception as error:
        raise AgentSnapshotAuthorizationError("Snapshot identity binding is invalid.") from error
    if validated_binding.binding_id != validated_access.binding_id:
        raise AgentSnapshotAuthorizationError("Snapshot access names another binding.")
    if validated_binding.snapshot != validated_access.snapshot:
        raise AgentSnapshotAuthorizationError("Snapshot access names another root.")
    if validated_binding.authority_scope_fingerprint != (
        validated_access.authority_scope_fingerprint
    ):
        raise AgentSnapshotAuthorizationError("Snapshot access broadens authority scope.")
    return validated_binding


def _verify_snapshot_nodes(
    snapshot: AgentSnapshot,
    nodes: dict[str, AgentSnapshotNode],
) -> tuple[AgentSnapshotNode, ...]:
    """Verify and enumerate one closure in deterministic root-first order."""

    validated_snapshot = AgentSnapshot.model_validate(snapshot.model_dump(mode="json"))
    root = validated_snapshot.snapshot_root
    ordered: list[AgentSnapshotNode] = []
    visited: set[str] = set()
    active: set[str] = set()

    def visit(
        digest: str,
        *,
        expected_kind: AgentSnapshotNodeKind,
        expected_schema: str,
    ) -> None:
        if digest in active:
            raise AgentSnapshotVerificationError("AgentSnapshot Merkle closure contains a cycle.")
        if digest in visited:
            node = nodes.get(digest)
            if node is None:
                raise AgentSnapshotVerificationError(
                    "AgentSnapshot Merkle closure is missing a child node."
                )
            if node.node_kind is not expected_kind or node.schema_id != expected_schema:
                raise AgentSnapshotVerificationError(
                    "AgentSnapshot Merkle child kind or schema does not match its edge."
                )
            return
        raw = nodes.get(digest)
        if raw is None:
            raise AgentSnapshotVerificationError(
                "AgentSnapshot Merkle closure is missing a child node."
            )
        try:
            node = AgentSnapshotNode.model_validate(raw.model_dump(mode="json"))
        except Exception as error:
            raise AgentSnapshotVerificationError(
                "AgentSnapshot Merkle closure contains a corrupt node."
            ) from error
        if node.digest != digest:
            raise AgentSnapshotVerificationError(
                "AgentSnapshot Merkle node does not match its content-addressed key."
            )
        if node.node_kind is not expected_kind or node.schema_id != expected_schema:
            raise AgentSnapshotVerificationError(
                "AgentSnapshot Merkle child kind or schema does not match its edge."
            )
        active.add(digest)
        ordered.append(node)
        for child in node.children:
            visit(
                child.digest,
                expected_kind=child.node_kind,
                expected_schema=child.schema_id,
            )
            child_node = nodes[child.digest]
            if child.node_kind is AgentSnapshotNodeKind.COMPONENT and (
                child_node.payload.get("kind") != child.relation.value
            ):
                raise AgentSnapshotVerificationError(
                    "AgentSnapshot Merkle child relation does not match its payload kind."
                )
        active.remove(digest)
        visited.add(digest)

    visit(
        root,
        expected_kind=AgentSnapshotNodeKind.MANIFEST,
        expected_schema="cayu.agent-snapshot.manifest.v3",
    )
    expected_root = validated_snapshot.root_node()
    if nodes[root] != expected_root:
        raise AgentSnapshotVerificationError(
            "AgentSnapshot manifest node differs from the stored snapshot document."
        )
    return tuple(ordered)


def _snapshot_unresolved_bindings(snapshot: AgentSnapshot) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{component.kind.value}:{component.logical.fingerprint}"
            for component in snapshot.components
            if component.materialization
            in {
                AgentSnapshotMaterializationCapability.REFERENCE_ONLY,
                AgentSnapshotMaterializationCapability.UNAVAILABLE,
            }
            or component.kind is AgentSnapshotComponentKind.EXTERNAL_BINDINGS
        )
    )


def _snapshot_access_key(access: AgentSnapshotAccess) -> tuple[str, str, str]:
    return (
        access.snapshot.snapshot_root,
        access.binding_id,
        access.authority_scope_fingerprint,
    )


def _snapshot_accesses_by_root(
    accesses: Iterable[AgentSnapshotAccess],
) -> dict[str, tuple[AgentSnapshotAccess, ...]]:
    grouped: dict[str, list[AgentSnapshotAccess]] = {}
    for access in accesses:
        grouped.setdefault(access.snapshot.snapshot_root, []).append(access)
    return {
        root: tuple(sorted(items, key=_snapshot_access_key))
        for root, items in sorted(grouped.items())
    }


def _create_gc_plan(
    *,
    operation_id: str,
    authorized_candidates: Iterable[AgentSnapshotAccess],
    collectable_roots: Iterable[str],
    blocked_roots: Iterable[str],
    node_digests_to_delete: Iterable[str],
    retained_shared_node_digests: Iterable[str],
    bytes_to_delete: int,
) -> AgentSnapshotGCPlan:
    values = {
        "operation_id": operation_id,
        "authorized_candidates": tuple(sorted(authorized_candidates, key=_snapshot_access_key)),
        "collectable_roots": tuple(sorted(set(collectable_roots))),
        "blocked_roots": tuple(sorted(set(blocked_roots))),
        "node_digests_to_delete": tuple(sorted(set(node_digests_to_delete))),
        "retained_shared_node_digests": tuple(sorted(set(retained_shared_node_digests))),
        "bytes_to_delete": bytes_to_delete,
    }
    provisional = AgentSnapshotGCPlan.model_construct(
        plan_id="0" * 64,
        operation_id=values["operation_id"],
        authorized_candidates=values["authorized_candidates"],
        collectable_roots=values["collectable_roots"],
        blocked_roots=values["blocked_roots"],
        node_digests_to_delete=values["node_digests_to_delete"],
        retained_shared_node_digests=values["retained_shared_node_digests"],
        bytes_to_delete=values["bytes_to_delete"],
    )
    return AgentSnapshotGCPlan(
        plan_id=_content_sha256(provisional.identity_material(), "agent_snapshot_gc_plan"),
        **values,
    )


def _create_gc_receipt(
    *,
    plan: AgentSnapshotGCPlan,
    deleted_roots: Iterable[str],
    deleted_node_digests: Iterable[str],
    deleted_bytes: int,
) -> AgentSnapshotGCReceipt:
    values = {
        "plan_id": plan.plan_id,
        "operation_id": plan.operation_id,
        "deleted_roots": tuple(sorted(set(deleted_roots))),
        "deleted_node_digests": tuple(sorted(set(deleted_node_digests))),
        "deleted_bytes": deleted_bytes,
    }
    provisional = AgentSnapshotGCReceipt.model_construct(
        receipt_id="0" * 64,
        plan_id=values["plan_id"],
        operation_id=values["operation_id"],
        deleted_roots=values["deleted_roots"],
        deleted_node_digests=values["deleted_node_digests"],
        deleted_bytes=values["deleted_bytes"],
    )
    return AgentSnapshotGCReceipt(
        receipt_id=_content_sha256(provisional.identity_material(), "agent_snapshot_gc_receipt"),
        **values,
    )


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
    if type(first) is AgentSnapshot and type(second) is AgentSnapshot:
        return first.snapshot_root == second.snapshot_root
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
        or materialization.state_partition_fingerprint != progress.state_partition_fingerprint
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
        self._snapshot_lock = threading.RLock()
        self._snapshot_nodes: dict[str, AgentSnapshotNode] = {}
        self._snapshot_root_nodes: dict[str, tuple[str, ...]] = {}
        self._snapshot_node_roots: dict[str, set[str]] = {}
        self._snapshot_bindings: dict[str, AgentSnapshotIdentityBinding] = {}
        self._snapshot_binding_documents: dict[str, AgentSnapshot] = {}
        self._snapshot_put_receipts: dict[str, AgentSnapshotPutReceipt] = {}
        self._snapshot_pins: dict[str, AgentSnapshotPinReceipt] = {}
        self._snapshot_released_pins: set[str] = set()
        self._snapshot_root_pins: dict[str, set[str]] = {}
        self._snapshot_protections: dict[str, AgentSnapshotProtection] = {}
        self._snapshot_released_protections: set[str] = set()
        self._snapshot_root_protections: dict[str, set[str]] = {}
        self._snapshot_operations: dict[tuple[str, str], tuple[str, BaseModel]] = {}
        self._snapshot_gc_plans: dict[str, AgentSnapshotGCPlan] = {}
        self._snapshot_gc_requests: dict[str, AgentSnapshotGCRequest] = {}
        self._snapshot_gc_receipts: dict[str, AgentSnapshotGCReceipt] = {}

    def _snapshot_operation_replay(
        self,
        kind: str,
        operation_id: str,
        request_material: object,
        model: type[BaseModel],
    ) -> BaseModel | None:
        request_digest = _content_sha256(request_material, f"{kind}_request")
        existing = self._snapshot_operations.get((kind, operation_id))
        if existing is None:
            return None
        existing_digest, receipt = existing
        if existing_digest != request_digest:
            raise AgentSnapshotStoreConflict(
                f"Snapshot {kind} operation_id is already bound to another request."
            )
        return model.model_validate(receipt.model_dump(mode="json"))

    def _record_snapshot_operation(
        self,
        kind: str,
        operation_id: str,
        request_material: object,
        receipt: BaseModel,
    ) -> None:
        self._snapshot_operations[(kind, operation_id)] = (
            _content_sha256(request_material, f"{kind}_request"),
            receipt,
        )

    def _binding_for_access(self, access: AgentSnapshotAccess) -> AgentSnapshotIdentityBinding:
        binding = self._snapshot_bindings.get(access.binding_id)
        return _validate_snapshot_access(access, binding)

    def _closure_for_access(
        self, access: AgentSnapshotAccess
    ) -> tuple[AgentSnapshot, tuple[AgentSnapshotNode, ...]]:
        self._binding_for_access(access)
        snapshot = self._snapshot_binding_documents.get(access.binding_id)
        if snapshot is None:
            raise AgentSnapshotVerificationError(
                "Snapshot binding points to a missing manifest document."
            )
        indexed = self._snapshot_root_nodes.get(access.snapshot.snapshot_root)
        if indexed is None:
            raise AgentSnapshotVerificationError("AgentSnapshot root closure is unavailable.")
        nodes = {
            digest: node
            for digest in indexed
            if (node := self._snapshot_nodes.get(digest)) is not None
        }
        closure = _verify_snapshot_nodes(snapshot, nodes)
        reachable = tuple(sorted(node.digest for node in closure))
        if reachable != indexed:
            raise AgentSnapshotVerificationError(
                "AgentSnapshot closure index differs from reachable nodes."
            )
        return snapshot, closure

    async def put_snapshot(
        self,
        snapshot: AgentSnapshot,
        binding: AgentSnapshotIdentityBinding,
    ) -> AgentSnapshotPutReceipt:
        validated_snapshot, validated_binding = _validate_snapshot_binding(snapshot, binding)
        nodes = validated_snapshot.merkle_nodes()
        node_by_digest = {node.digest: node for node in nodes}
        closure = _verify_snapshot_nodes(validated_snapshot, node_by_digest)
        node_digests = tuple(sorted(node.digest for node in closure))
        receipt = AgentSnapshotPutReceipt.create(
            snapshot=validated_snapshot.ref,
            binding_id=validated_binding.binding_id,
            node_digests=node_digests,
        )
        with self._snapshot_lock:
            existing_receipt = self._snapshot_put_receipts.get(validated_binding.binding_id)
            if existing_receipt is not None:
                if existing_receipt != receipt:
                    raise AgentSnapshotStoreConflict(
                        "Snapshot binding is already bound to another put receipt."
                    )
                self._closure_for_access(
                    AgentSnapshotAccess(
                        snapshot=validated_snapshot.ref,
                        binding_id=validated_binding.binding_id,
                        authority_scope_fingerprint=(validated_binding.authority_scope_fingerprint),
                    )
                )
                return AgentSnapshotPutReceipt.model_validate(
                    existing_receipt.model_dump(mode="json")
                )
            for node in closure:
                existing_node = self._snapshot_nodes.get(node.digest)
                if existing_node is not None and existing_node != node:
                    raise AgentSnapshotStoreConflict(
                        "Snapshot node digest is already bound to another envelope."
                    )
            existing_closure = self._snapshot_root_nodes.get(validated_snapshot.snapshot_root)
            if existing_closure is not None and existing_closure != node_digests:
                raise AgentSnapshotStoreConflict(
                    "Snapshot root is already bound to another closure."
                )
            existing_binding = self._snapshot_bindings.get(validated_binding.binding_id)
            if existing_binding is not None and existing_binding != validated_binding:
                raise AgentSnapshotStoreConflict(
                    "Snapshot binding id is already bound to another registration."
                )
            self._save("snapshot", validated_snapshot, validated_snapshot.snapshot_root)
            for node in closure:
                self._snapshot_nodes[node.digest] = node
            if existing_closure is None:
                self._snapshot_root_nodes[validated_snapshot.snapshot_root] = node_digests
                for digest in node_digests:
                    self._snapshot_node_roots.setdefault(digest, set()).add(
                        validated_snapshot.snapshot_root
                    )
            self._snapshot_bindings[validated_binding.binding_id] = validated_binding
            self._snapshot_binding_documents[validated_binding.binding_id] = validated_snapshot
            self._snapshot_put_receipts[validated_binding.binding_id] = receipt
            return AgentSnapshotPutReceipt.model_validate(receipt.model_dump(mode="json"))

    async def load_identity_binding(self, binding_id: str) -> AgentSnapshotIdentityBinding | None:
        _sha256_hex(binding_id, "binding_id")
        binding = self._snapshot_bindings.get(binding_id)
        if binding is None:
            return None
        validated = AgentSnapshotIdentityBinding.model_validate(binding.model_dump(mode="json"))
        if validated.binding_id != binding_id:
            raise AgentSnapshotStoreConflict(
                "Stored snapshot binding does not match its content-addressed key."
            )
        return validated

    async def get_snapshot(self, access: AgentSnapshotAccess) -> AgentSnapshot:
        with self._snapshot_lock:
            snapshot, _ = self._closure_for_access(access)
            return AgentSnapshot.model_validate(snapshot.model_dump(mode="json"))

    async def enumerate_snapshot_closure(
        self, access: AgentSnapshotAccess
    ) -> tuple[AgentSnapshotNode, ...]:
        with self._snapshot_lock:
            _, closure = self._closure_for_access(access)
            return tuple(
                AgentSnapshotNode.model_validate(node.model_dump(mode="json")) for node in closure
            )

    async def inspect_snapshot(self, access: AgentSnapshotAccess) -> AgentSnapshotClosureInspection:
        with self._snapshot_lock:
            snapshot, closure = self._closure_for_access(access)
            sizes = {
                node.digest: len(_snapshot_model_bytes(node, "agent_snapshot_node"))
                for node in closure
            }
            unique = sum(
                size
                for digest, size in sizes.items()
                if len(self._snapshot_node_roots.get(digest, set())) == 1
            )
            shared = sum(sizes.values()) - unique
            return AgentSnapshotClosureInspection(
                snapshot=access.snapshot,
                root_manifest_bytes=len(
                    _snapshot_model_bytes(snapshot.root_node(), "agent_snapshot_root_manifest")
                ),
                logical_closure_bytes=sum(sizes.values()),
                unique_stored_bytes=unique,
                shared_bytes=shared,
                object_count=len(closure),
                node_digests=tuple(sorted(sizes)),
                unresolved_external_bindings=_snapshot_unresolved_bindings(snapshot),
            )

    async def pin_snapshot(self, request: AgentSnapshotPinRequest) -> AgentSnapshotPinReceipt:
        if type(request) is not AgentSnapshotPinRequest:
            raise TypeError("request must be an AgentSnapshotPinRequest.")
        validated = AgentSnapshotPinRequest.model_validate(request.model_dump(mode="json"))
        with self._snapshot_lock:
            replay = self._snapshot_operation_replay(
                "pin",
                validated.operation_id,
                validated.identity_material(),
                AgentSnapshotPinReceipt,
            )
            if replay is not None:
                return cast("AgentSnapshotPinReceipt", replay)
            self._closure_for_access(validated.access)
            receipt = AgentSnapshotPinReceipt.from_request(validated)
            existing = self._snapshot_pins.get(receipt.pin_id)
            if existing is not None and (
                existing.snapshot != receipt.snapshot
                or existing.binding_id != receipt.binding_id
                or existing.owner != receipt.owner
                or existing.reason != receipt.reason
                or existing.retention_class is not receipt.retention_class
            ):
                raise AgentSnapshotStoreConflict("Snapshot pin identity conflicts.")
            self._snapshot_pins[receipt.pin_id] = receipt
            self._snapshot_root_pins.setdefault(receipt.snapshot.snapshot_root, set()).add(
                receipt.pin_id
            )
            self._record_snapshot_operation(
                "pin", validated.operation_id, validated.identity_material(), receipt
            )
            return receipt

    async def release_snapshot_pin(
        self, request: AgentSnapshotReleaseRequest
    ) -> AgentSnapshotReleaseReceipt:
        if type(request) is not AgentSnapshotReleaseRequest:
            raise TypeError("request must be an AgentSnapshotReleaseRequest.")
        validated = AgentSnapshotReleaseRequest.model_validate(request.model_dump(mode="json"))
        with self._snapshot_lock:
            replay = self._snapshot_operation_replay(
                "release",
                validated.operation_id,
                validated.identity_material(),
                AgentSnapshotReleaseReceipt,
            )
            if replay is not None:
                return cast("AgentSnapshotReleaseReceipt", replay)
            self._closure_for_access(validated.access)
            pin = self._snapshot_pins.get(validated.pin_id)
            if pin is None:
                raise AgentSnapshotStoreConflict("Snapshot pin is unavailable.")
            if pin.snapshot != validated.access.snapshot or pin.binding_id != (
                validated.access.binding_id
            ):
                raise AgentSnapshotAuthorizationError(
                    "Snapshot release names a pin outside its binding."
                )
            if pin.owner != validated.owner:
                raise AgentSnapshotAuthorizationError(
                    "Snapshot release owner does not own the pin."
                )
            receipt = AgentSnapshotReleaseReceipt.from_request(validated)
            self._snapshot_released_pins.add(pin.pin_id)
            self._snapshot_root_pins.get(pin.snapshot.snapshot_root, set()).discard(pin.pin_id)
            self._record_snapshot_operation(
                "release", validated.operation_id, validated.identity_material(), receipt
            )
            return receipt

    async def protect_snapshot(
        self, protection: AgentSnapshotProtection
    ) -> AgentSnapshotProtection:
        if type(protection) is not AgentSnapshotProtection:
            raise TypeError("protection must be an AgentSnapshotProtection.")
        validated = AgentSnapshotProtection.model_validate(protection.model_dump(mode="json"))
        with self._snapshot_lock:
            replay = self._snapshot_operation_replay(
                "protect",
                validated.operation_id,
                validated.model_dump(mode="json"),
                AgentSnapshotProtection,
            )
            if replay is not None:
                return cast("AgentSnapshotProtection", replay)
            self._closure_for_access(validated.access)
            existing = self._snapshot_protections.get(validated.protection_id)
            if existing is not None and existing != validated:
                raise AgentSnapshotStoreConflict("Snapshot protection identity conflicts.")
            self._snapshot_protections[validated.protection_id] = validated
            self._snapshot_root_protections.setdefault(
                validated.access.snapshot.snapshot_root, set()
            ).add(validated.protection_id)
            self._record_snapshot_operation(
                "protect",
                validated.operation_id,
                validated.model_dump(mode="json"),
                validated,
            )
            return validated

    async def release_snapshot_protection(
        self,
        *,
        operation_id: str,
        access: AgentSnapshotAccess,
        protection_id: str,
    ) -> AgentSnapshotProtection:
        operation_id = _clean(operation_id, "operation_id", max_chars=256)
        protection_id = _sha256_hex(protection_id, "protection_id")
        request_material = {
            "operation_id": operation_id,
            "access": access.model_dump(mode="json"),
            "protection_id": protection_id,
        }
        with self._snapshot_lock:
            replay = self._snapshot_operation_replay(
                "unprotect",
                operation_id,
                request_material,
                AgentSnapshotProtection,
            )
            if replay is not None:
                return cast("AgentSnapshotProtection", replay)
            self._closure_for_access(access)
            protection = self._snapshot_protections.get(protection_id)
            if protection is None:
                raise AgentSnapshotStoreConflict("Snapshot protection is unavailable.")
            if protection.access != access:
                raise AgentSnapshotAuthorizationError(
                    "Snapshot protection is outside the supplied binding."
                )
            self._snapshot_released_protections.add(protection_id)
            self._snapshot_root_protections.get(
                protection.access.snapshot.snapshot_root, set()
            ).discard(protection_id)
            self._record_snapshot_operation("unprotect", operation_id, request_material, protection)
            return protection

    def _root_is_protected(self, snapshot_root: str) -> bool:
        return bool(self._snapshot_root_pins.get(snapshot_root)) or bool(
            self._snapshot_root_protections.get(snapshot_root)
        )

    async def plan_snapshot_gc(self, request: AgentSnapshotGCRequest) -> AgentSnapshotGCPlan:
        if type(request) is not AgentSnapshotGCRequest:
            raise TypeError("request must be an AgentSnapshotGCRequest.")
        validated = AgentSnapshotGCRequest.model_validate(request.model_dump(mode="json"))
        request_digest = _content_sha256(
            validated.model_dump(mode="json"), "agent_snapshot_gc_request"
        )
        with self._snapshot_lock:
            existing_plan = self._snapshot_gc_plans.get(validated.operation_id)
            if existing_plan is not None:
                existing_request = self._snapshot_gc_requests[existing_plan.plan_id]
                if (
                    _content_sha256(
                        existing_request.model_dump(mode="json"), "agent_snapshot_gc_request"
                    )
                    != request_digest
                ):
                    raise AgentSnapshotStoreConflict(
                        "Snapshot GC operation_id is already bound to another request."
                    )
                return AgentSnapshotGCPlan.model_validate(existing_plan.model_dump(mode="json"))
            blocked: list[str] = []
            collectable: list[str] = []
            for access in validated.candidates:
                self._closure_for_access(access)
            accesses_by_root = _snapshot_accesses_by_root(validated.candidates)
            for root, accesses in accesses_by_root.items():
                authorized_binding_ids = {access.binding_id for access in accesses}
                stored_binding_ids = {
                    binding_id
                    for binding_id, binding in self._snapshot_bindings.items()
                    if binding.snapshot.snapshot_root == root
                }
                if self._root_is_protected(root) or authorized_binding_ids != stored_binding_ids:
                    blocked.append(root)
                else:
                    collectable.append(root)
            collectable_set = set(collectable)
            candidate_nodes = {
                digest for root in collectable for digest in self._snapshot_root_nodes[root]
            }
            deleting = {
                digest
                for digest in candidate_nodes
                if self._snapshot_node_roots.get(digest, set()) <= collectable_set
            }
            retained = candidate_nodes - deleting
            bytes_to_delete = sum(
                len(_snapshot_model_bytes(self._snapshot_nodes[digest], "agent_snapshot_node"))
                for digest in deleting
            )
            plan = _create_gc_plan(
                operation_id=validated.operation_id,
                authorized_candidates=validated.candidates,
                collectable_roots=collectable,
                blocked_roots=blocked,
                node_digests_to_delete=deleting,
                retained_shared_node_digests=retained,
                bytes_to_delete=bytes_to_delete,
            )
            self._snapshot_gc_plans[validated.operation_id] = plan
            self._snapshot_gc_requests[plan.plan_id] = validated
            return plan

    async def execute_snapshot_gc(self, plan: AgentSnapshotGCPlan) -> AgentSnapshotGCReceipt:
        if type(plan) is not AgentSnapshotGCPlan:
            raise TypeError("plan must be an AgentSnapshotGCPlan.")
        validated = AgentSnapshotGCPlan.model_validate(plan.model_dump(mode="json"))
        with self._snapshot_lock:
            existing_receipt = self._snapshot_gc_receipts.get(validated.plan_id)
            if existing_receipt is not None:
                return AgentSnapshotGCReceipt.model_validate(
                    existing_receipt.model_dump(mode="json")
                )
            stored = self._snapshot_gc_plans.get(validated.operation_id)
            if stored != validated:
                raise AgentSnapshotStoreConflict("Snapshot GC plan was not durably prepared.")
            accesses_by_root = _snapshot_accesses_by_root(validated.authorized_candidates)
            for root in validated.collectable_roots:
                accesses = accesses_by_root[root]
                for access in accesses:
                    self._closure_for_access(access)
                authorized_binding_ids = {access.binding_id for access in accesses}
                stored_binding_ids = {
                    binding_id
                    for binding_id, binding in self._snapshot_bindings.items()
                    if binding.snapshot.snapshot_root == root
                }
                if authorized_binding_ids != stored_binding_ids:
                    raise AgentSnapshotStoreConflict(
                        "Snapshot GC binding reachability changed after planning."
                    )
                if self._root_is_protected(root):
                    raise AgentSnapshotStoreConflict(
                        "Snapshot GC root gained protection after planning."
                    )
                if root not in self._snapshot_root_nodes:
                    raise AgentSnapshotStoreConflict(
                        "Snapshot GC root disappeared before execution."
                    )
            collectable = set(validated.collectable_roots)
            actual_deleting = {
                digest
                for root in collectable
                for digest in self._snapshot_root_nodes[root]
                if self._snapshot_node_roots.get(digest, set()) <= collectable
            }
            if actual_deleting != set(validated.node_digests_to_delete):
                raise AgentSnapshotStoreConflict("Snapshot GC reachability changed after planning.")
            deleted_bytes = sum(
                len(_snapshot_model_bytes(self._snapshot_nodes[digest], "agent_snapshot_node"))
                for digest in actual_deleting
            )
            if deleted_bytes != validated.bytes_to_delete:
                raise AgentSnapshotStoreConflict("Snapshot GC byte plan changed.")
            for root in collectable:
                for digest in self._snapshot_root_nodes.pop(root):
                    roots = self._snapshot_node_roots[digest]
                    roots.discard(root)
                    if not roots:
                        self._snapshot_node_roots.pop(digest)
                self._records.pop(("snapshot", root), None)
                self._snapshot_root_pins.pop(root, None)
                self._snapshot_root_protections.pop(root, None)
            for digest in actual_deleting:
                self._snapshot_nodes.pop(digest, None)
            binding_ids = tuple(
                binding_id
                for binding_id, binding in self._snapshot_bindings.items()
                if binding.snapshot.snapshot_root in collectable
            )
            for binding_id in binding_ids:
                self._snapshot_bindings.pop(binding_id, None)
                self._snapshot_binding_documents.pop(binding_id, None)
                self._snapshot_put_receipts.pop(binding_id, None)
            receipt = _create_gc_receipt(
                plan=validated,
                deleted_roots=collectable,
                deleted_node_digests=actual_deleting,
                deleted_bytes=deleted_bytes,
            )
            self._snapshot_gc_receipts[validated.plan_id] = receipt
            return receipt

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
        await self.put_snapshot(snapshot, snapshot.identity_binding)
        return await self.get_snapshot(
            AgentSnapshotAccess(
                snapshot=snapshot.ref,
                binding_id=snapshot.identity_binding.binding_id,
                authority_scope_fingerprint=(snapshot.identity_binding.authority_scope_fingerprint),
            )
        )

    async def load_snapshot(self, snapshot_root: str) -> AgentSnapshot | None:
        return cast("AgentSnapshot | None", self._load("snapshot", snapshot_root, AgentSnapshot))

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

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @contextmanager
    def _write_connection(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:  # noqa: SIM117 - acquire before opening SQLite
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                yield connection

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_connection() as connection:
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_nodes (
                    digest TEXT PRIMARY KEY,
                    node_kind TEXT NOT NULL,
                    schema_id TEXT NOT NULL,
                    byte_count INTEGER NOT NULL,
                    document TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_roots (
                    snapshot_root TEXT PRIMARY KEY,
                    manifest_document TEXT NOT NULL,
                    manifest_bytes INTEGER NOT NULL,
                    closure_bytes INTEGER NOT NULL,
                    object_count INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_root_nodes (
                    snapshot_root TEXT NOT NULL,
                    node_digest TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY (snapshot_root, node_digest),
                    UNIQUE (snapshot_root, ordinal)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_bindings (
                    binding_id TEXT PRIMARY KEY,
                    snapshot_root TEXT NOT NULL,
                    authority_scope_fingerprint TEXT NOT NULL,
                    binding_document TEXT NOT NULL,
                    snapshot_document TEXT NOT NULL,
                    put_receipt_document TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_pins (
                    pin_id TEXT PRIMARY KEY,
                    snapshot_root TEXT NOT NULL,
                    binding_id TEXT NOT NULL,
                    document TEXT NOT NULL,
                    released INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_protections (
                    protection_id TEXT PRIMARY KEY,
                    snapshot_root TEXT NOT NULL,
                    binding_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    document TEXT NOT NULL,
                    released INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_lifecycle_operations (
                    operation_kind TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    response_model TEXT NOT NULL,
                    response_document TEXT NOT NULL,
                    PRIMARY KEY (operation_kind, operation_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_gc_plans (
                    operation_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL UNIQUE,
                    request_digest TEXT NOT NULL,
                    request_document TEXT NOT NULL,
                    plan_document TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cayu_agent_snapshot_gc_receipts (
                    plan_id TEXT PRIMARY KEY,
                    receipt_document TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS cayu_agent_snapshot_root_nodes_by_node "
                "ON cayu_agent_snapshot_root_nodes (node_digest, snapshot_root)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS cayu_agent_snapshot_bindings_by_root "
                "ON cayu_agent_snapshot_bindings (snapshot_root)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS cayu_agent_snapshot_active_pins_by_root "
                "ON cayu_agent_snapshot_pins (snapshot_root, released)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS cayu_agent_snapshot_active_protections_by_root "
                "ON cayu_agent_snapshot_protections (snapshot_root, released)"
            )

    def _load_access_closure_sync(
        self,
        connection: sqlite3.Connection,
        access: AgentSnapshotAccess,
    ) -> tuple[
        AgentSnapshot,
        AgentSnapshotIdentityBinding,
        tuple[AgentSnapshotNode, ...],
        sqlite3.Row,
    ]:
        validated_access = AgentSnapshotAccess.model_validate(access.model_dump(mode="json"))
        binding_row = connection.execute(
            "SELECT snapshot_root, authority_scope_fingerprint, binding_document, "
            "snapshot_document FROM cayu_agent_snapshot_bindings WHERE binding_id = ?",
            (validated_access.binding_id,),
        ).fetchone()
        if binding_row is None:
            raise AgentSnapshotAuthorizationError("Snapshot identity binding is unavailable.")
        binding_document = cast("str", binding_row["binding_document"])
        try:
            binding = AgentSnapshotIdentityBinding.model_validate_json(binding_document)
        except Exception as error:
            raise AgentSnapshotAuthorizationError(
                "Snapshot identity binding is invalid."
            ) from error
        if binding_document != _snapshot_model_json(binding):
            raise AgentSnapshotAuthorizationError(
                "Snapshot identity binding document is not canonical."
            )
        _validate_snapshot_access(validated_access, binding)
        if (
            binding_row["snapshot_root"] != binding.snapshot.snapshot_root
            or binding_row["authority_scope_fingerprint"] != binding.authority_scope_fingerprint
        ):
            raise AgentSnapshotAuthorizationError(
                "Snapshot identity binding columns contradict its document."
            )
        snapshot_document = cast("str", binding_row["snapshot_document"])
        try:
            snapshot = AgentSnapshot.model_validate_json(snapshot_document)
        except Exception as error:
            raise AgentSnapshotVerificationError(
                "Snapshot binding manifest document is invalid."
            ) from error
        if snapshot_document != _snapshot_model_json(snapshot):
            raise AgentSnapshotVerificationError(
                "Snapshot binding manifest document is not canonical."
            )
        if snapshot.snapshot_root != validated_access.snapshot.snapshot_root:
            raise AgentSnapshotVerificationError("Snapshot binding manifest names another root.")
        root_row = connection.execute(
            "SELECT manifest_document, manifest_bytes, closure_bytes, object_count "
            "FROM cayu_agent_snapshot_roots WHERE snapshot_root = ?",
            (validated_access.snapshot.snapshot_root,),
        ).fetchone()
        if root_row is None:
            raise AgentSnapshotVerificationError("AgentSnapshot root closure is unavailable.")
        root_manifest_document = cast("str", root_row["manifest_document"])
        try:
            stored_root_manifest = AgentSnapshotNode.model_validate_json(root_manifest_document)
        except Exception as error:
            raise AgentSnapshotVerificationError(
                "AgentSnapshot root manifest node is invalid."
            ) from error
        if root_manifest_document != _snapshot_model_json(stored_root_manifest):
            raise AgentSnapshotVerificationError(
                "AgentSnapshot root manifest node is not canonical."
            )
        rows = connection.execute(
            "SELECT link.node_digest, link.ordinal, node.node_kind, node.schema_id, "
            "node.byte_count, node.document "
            "FROM cayu_agent_snapshot_root_nodes AS link "
            "LEFT JOIN cayu_agent_snapshot_nodes AS node ON node.digest = link.node_digest "
            "WHERE link.snapshot_root = ? ORDER BY link.ordinal",
            (validated_access.snapshot.snapshot_root,),
        ).fetchall()
        nodes: dict[str, AgentSnapshotNode] = {}
        indexed: list[str] = []
        for ordinal, row in enumerate(rows):
            if row["ordinal"] != ordinal:
                raise AgentSnapshotVerificationError(
                    "AgentSnapshot closure index has a non-contiguous ordinal."
                )
            digest = cast("str", row["node_digest"])
            indexed.append(digest)
            if row["document"] is None:
                continue
            try:
                node_document = cast("str", row["document"])
                node = AgentSnapshotNode.model_validate_json(node_document)
            except Exception as error:
                raise AgentSnapshotVerificationError(
                    "AgentSnapshot Merkle closure contains a corrupt node."
                ) from error
            if (
                node.digest != digest
                or row["node_kind"] != node.node_kind.value
                or row["schema_id"] != node.schema_id
                or row["byte_count"] != len(_snapshot_model_bytes(node, "agent_snapshot_node"))
                or node_document != _snapshot_model_json(node)
            ):
                raise AgentSnapshotVerificationError(
                    "AgentSnapshot node columns contradict its envelope."
                )
            nodes[digest] = node
        closure = _verify_snapshot_nodes(snapshot, nodes)
        if stored_root_manifest != nodes[snapshot.snapshot_root]:
            raise AgentSnapshotVerificationError(
                "AgentSnapshot root manifest row differs from its content-addressed node."
            )
        if tuple(indexed) != tuple(sorted(node.digest for node in closure)):
            raise AgentSnapshotVerificationError(
                "AgentSnapshot closure index differs from reachable nodes."
            )
        if root_row["object_count"] != len(closure) or root_row["closure_bytes"] != sum(
            len(_snapshot_model_bytes(node, "agent_snapshot_node")) for node in closure
        ):
            raise AgentSnapshotVerificationError(
                "AgentSnapshot root metrics contradict its closure."
            )
        if root_row["manifest_bytes"] != len(
            _snapshot_model_bytes(nodes[snapshot.snapshot_root], "agent_snapshot_root_manifest")
        ):
            raise AgentSnapshotVerificationError(
                "AgentSnapshot root manifest byte count is invalid."
            )
        return snapshot, binding, closure, root_row

    def _put_snapshot_sync(
        self,
        snapshot: AgentSnapshot,
        binding: AgentSnapshotIdentityBinding,
    ) -> str:
        validated_snapshot, validated_binding = _validate_snapshot_binding(snapshot, binding)
        closure = _verify_snapshot_nodes(
            validated_snapshot,
            {node.digest: node for node in validated_snapshot.merkle_nodes()},
        )
        node_digests = tuple(sorted(node.digest for node in closure))
        receipt = AgentSnapshotPutReceipt.create(
            snapshot=validated_snapshot.ref,
            binding_id=validated_binding.binding_id,
            node_digests=node_digests,
        )
        snapshot_document = _snapshot_model_json(validated_snapshot)
        binding_document = _snapshot_model_json(validated_binding)
        receipt_document = _snapshot_model_json(receipt)
        root_manifest = validated_snapshot.root_node()
        root_manifest_document = _snapshot_model_json(root_manifest)
        manifest_bytes = len(_snapshot_model_bytes(root_manifest, "agent_snapshot_root_manifest"))
        closure_bytes = sum(
            len(_snapshot_model_bytes(node, "agent_snapshot_node")) for node in closure
        )
        with self._write_connection() as connection:
            existing_binding_row = connection.execute(
                "SELECT binding_document, snapshot_document, put_receipt_document "
                "FROM cayu_agent_snapshot_bindings WHERE binding_id = ?",
                (validated_binding.binding_id,),
            ).fetchone()
            if existing_binding_row is not None:
                try:
                    existing_binding = AgentSnapshotIdentityBinding.model_validate_json(
                        cast("str", existing_binding_row["binding_document"])
                    )
                    existing_snapshot = AgentSnapshot.model_validate_json(
                        cast("str", existing_binding_row["snapshot_document"])
                    )
                    existing_receipt = AgentSnapshotPutReceipt.model_validate_json(
                        cast("str", existing_binding_row["put_receipt_document"])
                    )
                except Exception as error:
                    raise AgentSnapshotStoreConflict(
                        "Stored snapshot binding is invalid."
                    ) from error
                if (
                    not _same_record_identity(existing_binding, validated_binding)
                    or existing_snapshot.snapshot_root != validated_snapshot.snapshot_root
                    or existing_receipt != receipt
                    or existing_binding_row["put_receipt_document"]
                    != _snapshot_model_json(existing_receipt)
                ):
                    raise AgentSnapshotStoreConflict(
                        "Snapshot binding is already bound to another put."
                    )
                self._load_access_closure_sync(
                    connection,
                    AgentSnapshotAccess(
                        snapshot=validated_snapshot.ref,
                        binding_id=validated_binding.binding_id,
                        authority_scope_fingerprint=(validated_binding.authority_scope_fingerprint),
                    ),
                )
                return cast("str", existing_binding_row["put_receipt_document"])
            for node in closure:
                document = _snapshot_model_json(node)
                byte_count = len(_snapshot_model_bytes(node, "agent_snapshot_node"))
                connection.execute(
                    "INSERT OR IGNORE INTO cayu_agent_snapshot_nodes "
                    "(digest, node_kind, schema_id, byte_count, document) VALUES (?, ?, ?, ?, ?)",
                    (node.digest, node.node_kind.value, node.schema_id, byte_count, document),
                )
                row = connection.execute(
                    "SELECT node_kind, schema_id, byte_count, document "
                    "FROM cayu_agent_snapshot_nodes WHERE digest = ?",
                    (node.digest,),
                ).fetchone()
                if row is None or (
                    row["node_kind"] != node.node_kind.value
                    or row["schema_id"] != node.schema_id
                    or row["byte_count"] != byte_count
                    or row["document"] != document
                ):
                    raise AgentSnapshotStoreConflict(
                        "Snapshot node digest is already bound to another envelope."
                    )
            root_row = connection.execute(
                "SELECT manifest_document, manifest_bytes, closure_bytes, object_count "
                "FROM cayu_agent_snapshot_roots WHERE snapshot_root = ?",
                (validated_snapshot.snapshot_root,),
            ).fetchone()
            if root_row is None:
                connection.execute(
                    "INSERT INTO cayu_agent_snapshot_roots "
                    "(snapshot_root, manifest_document, manifest_bytes, closure_bytes, object_count) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        validated_snapshot.snapshot_root,
                        root_manifest_document,
                        manifest_bytes,
                        closure_bytes,
                        len(closure),
                    ),
                )
                for ordinal, digest in enumerate(node_digests):
                    connection.execute(
                        "INSERT INTO cayu_agent_snapshot_root_nodes "
                        "(snapshot_root, node_digest, ordinal) VALUES (?, ?, ?)",
                        (validated_snapshot.snapshot_root, digest, ordinal),
                    )
            else:
                indexed = tuple(
                    row["node_digest"]
                    for row in connection.execute(
                        "SELECT node_digest FROM cayu_agent_snapshot_root_nodes "
                        "WHERE snapshot_root = ? ORDER BY ordinal",
                        (validated_snapshot.snapshot_root,),
                    ).fetchall()
                )
                try:
                    stored_manifest = AgentSnapshotNode.model_validate_json(
                        cast("str", root_row["manifest_document"])
                    )
                except Exception as error:
                    raise AgentSnapshotStoreConflict("Stored root manifest is invalid.") from error
                if (
                    indexed != node_digests
                    or stored_manifest != root_manifest
                    or root_row["manifest_document"] != root_manifest_document
                    or root_row["manifest_bytes"] != manifest_bytes
                    or root_row["closure_bytes"] != closure_bytes
                    or root_row["object_count"] != len(closure)
                ):
                    raise AgentSnapshotStoreConflict(
                        "Snapshot root is already bound to another closure."
                    )
            connection.execute(
                "INSERT INTO cayu_agent_snapshot_bindings "
                "(binding_id, snapshot_root, authority_scope_fingerprint, binding_document, "
                "snapshot_document, put_receipt_document) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    validated_binding.binding_id,
                    validated_snapshot.snapshot_root,
                    validated_binding.authority_scope_fingerprint,
                    binding_document,
                    snapshot_document,
                    receipt_document,
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO cayu_agent_snapshot_records "
                "(record_kind, fingerprint, document) VALUES ('snapshot', ?, ?)",
                (validated_snapshot.snapshot_root, snapshot_document),
            )
            record = connection.execute(
                "SELECT document FROM cayu_agent_snapshot_records "
                "WHERE record_kind = 'snapshot' AND fingerprint = ?",
                (validated_snapshot.snapshot_root,),
            ).fetchone()
            if record is None:
                raise AgentSnapshotStoreConflict("Snapshot manifest record was not stored.")
            try:
                stored = AgentSnapshot.model_validate_json(cast("str", record["document"]))
            except Exception as error:
                raise AgentSnapshotStoreConflict(
                    "Stored snapshot manifest record is invalid."
                ) from error
            if stored.snapshot_root != validated_snapshot.snapshot_root:
                raise AgentSnapshotStoreConflict(
                    "Stored snapshot manifest record names another root."
                )
        return receipt_document

    async def put_snapshot(
        self,
        snapshot: AgentSnapshot,
        binding: AgentSnapshotIdentityBinding,
    ) -> AgentSnapshotPutReceipt:
        document = await asyncio.to_thread(self._put_snapshot_sync, snapshot, binding)
        return AgentSnapshotPutReceipt.model_validate_json(document)

    async def load_identity_binding(self, binding_id: str) -> AgentSnapshotIdentityBinding | None:
        binding_id = _sha256_hex(binding_id, "binding_id")

        def load() -> str | None:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT snapshot_root, authority_scope_fingerprint, binding_document "
                    "FROM cayu_agent_snapshot_bindings "
                    "WHERE binding_id = ?",
                    (binding_id,),
                ).fetchone()
            if row is None:
                return None
            document = cast("str", row["binding_document"])
            try:
                binding = AgentSnapshotIdentityBinding.model_validate_json(document)
            except Exception as error:
                raise AgentSnapshotStoreConflict("Stored snapshot binding is invalid.") from error
            if (
                binding.binding_id != binding_id
                or row["snapshot_root"] != binding.snapshot.snapshot_root
                or row["authority_scope_fingerprint"] != binding.authority_scope_fingerprint
                or document != _snapshot_model_json(binding)
            ):
                raise AgentSnapshotStoreConflict(
                    "Stored snapshot binding columns contradict its canonical document."
                )
            return document

        document = await asyncio.to_thread(load)
        if document is None:
            return None
        return AgentSnapshotIdentityBinding.model_validate_json(document)

    def _get_snapshot_sync(self, access: AgentSnapshotAccess) -> str:
        with self._connection() as connection:
            snapshot, _, _, _ = self._load_access_closure_sync(connection, access)
            return _snapshot_model_json(snapshot)

    async def get_snapshot(self, access: AgentSnapshotAccess) -> AgentSnapshot:
        return AgentSnapshot.model_validate_json(
            await asyncio.to_thread(self._get_snapshot_sync, access)
        )

    def _enumerate_snapshot_closure_sync(self, access: AgentSnapshotAccess) -> tuple[str, ...]:
        with self._connection() as connection:
            _, _, closure, _ = self._load_access_closure_sync(connection, access)
            return tuple(_snapshot_model_json(node) for node in closure)

    async def enumerate_snapshot_closure(
        self, access: AgentSnapshotAccess
    ) -> tuple[AgentSnapshotNode, ...]:
        documents = await asyncio.to_thread(self._enumerate_snapshot_closure_sync, access)
        return tuple(AgentSnapshotNode.model_validate_json(document) for document in documents)

    def _inspect_snapshot_sync(self, access: AgentSnapshotAccess) -> str:
        with self._connection() as connection:
            snapshot, _, closure, root_row = self._load_access_closure_sync(connection, access)
            sizes = {
                node.digest: len(_snapshot_model_bytes(node, "agent_snapshot_node"))
                for node in closure
            }
            unique = 0
            for digest, size in sizes.items():
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM cayu_agent_snapshot_root_nodes "
                    "WHERE node_digest = ?",
                    (digest,),
                ).fetchone()
                if row is not None and row["count"] == 1:
                    unique += size
            inspection = AgentSnapshotClosureInspection(
                snapshot=access.snapshot,
                root_manifest_bytes=cast("int", root_row["manifest_bytes"]),
                logical_closure_bytes=cast("int", root_row["closure_bytes"]),
                unique_stored_bytes=unique,
                shared_bytes=cast("int", root_row["closure_bytes"]) - unique,
                object_count=len(closure),
                node_digests=tuple(sorted(sizes)),
                unresolved_external_bindings=_snapshot_unresolved_bindings(snapshot),
            )
            return _snapshot_model_json(inspection)

    async def inspect_snapshot(self, access: AgentSnapshotAccess) -> AgentSnapshotClosureInspection:
        return AgentSnapshotClosureInspection.model_validate_json(
            await asyncio.to_thread(self._inspect_snapshot_sync, access)
        )

    def _sqlite_operation_replay(
        self,
        connection: sqlite3.Connection,
        *,
        kind: str,
        operation_id: str,
        request_material: object,
        model: type[BaseModel],
    ) -> BaseModel | None:
        request_digest = _content_sha256(request_material, f"{kind}_request")
        row = connection.execute(
            "SELECT request_digest, response_model, response_document "
            "FROM cayu_agent_snapshot_lifecycle_operations "
            "WHERE operation_kind = ? AND operation_id = ?",
            (kind, operation_id),
        ).fetchone()
        if row is None:
            return None
        if row["request_digest"] != request_digest or row["response_model"] != model.__name__:
            raise AgentSnapshotStoreConflict(
                f"Snapshot {kind} operation_id is already bound to another request."
            )
        document = cast("str", row["response_document"])
        try:
            response = model.model_validate_json(document)
        except Exception as error:
            raise AgentSnapshotStoreConflict(
                f"Snapshot {kind} operation receipt is invalid."
            ) from error
        if document != _snapshot_model_json(response):
            raise AgentSnapshotStoreConflict(f"Snapshot {kind} operation receipt is not canonical.")
        return response

    def _record_sqlite_operation(
        self,
        connection: sqlite3.Connection,
        *,
        kind: str,
        operation_id: str,
        request_material: object,
        response: BaseModel,
    ) -> None:
        connection.execute(
            "INSERT INTO cayu_agent_snapshot_lifecycle_operations "
            "(operation_kind, operation_id, request_digest, response_model, response_document) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                kind,
                operation_id,
                _content_sha256(request_material, f"{kind}_request"),
                type(response).__name__,
                _snapshot_model_json(response),
            ),
        )

    def _pin_snapshot_sync(self, request: AgentSnapshotPinRequest) -> str:
        validated = AgentSnapshotPinRequest.model_validate(request.model_dump(mode="json"))
        with self._write_connection() as connection:
            replay = self._sqlite_operation_replay(
                connection,
                kind="pin",
                operation_id=validated.operation_id,
                request_material=validated.identity_material(),
                model=AgentSnapshotPinReceipt,
            )
            if replay is not None:
                return _snapshot_model_json(replay)
            self._load_access_closure_sync(connection, validated.access)
            receipt = AgentSnapshotPinReceipt.from_request(validated)
            connection.execute(
                "INSERT OR IGNORE INTO cayu_agent_snapshot_pins "
                "(pin_id, snapshot_root, binding_id, document, released) "
                "VALUES (?, ?, ?, ?, 0)",
                (
                    receipt.pin_id,
                    receipt.snapshot.snapshot_root,
                    receipt.binding_id,
                    _snapshot_model_json(receipt),
                ),
            )
            row = connection.execute(
                "SELECT document FROM cayu_agent_snapshot_pins WHERE pin_id = ?",
                (receipt.pin_id,),
            ).fetchone()
            if row is None:
                raise AgentSnapshotStoreConflict("Snapshot pin was not stored.")
            try:
                stored = AgentSnapshotPinReceipt.model_validate_json(cast("str", row["document"]))
            except Exception as error:
                raise AgentSnapshotStoreConflict("Stored snapshot pin is invalid.") from error
            if (
                stored.snapshot != receipt.snapshot
                or stored.binding_id != receipt.binding_id
                or stored.owner != receipt.owner
                or stored.reason != receipt.reason
                or stored.retention_class is not receipt.retention_class
            ):
                raise AgentSnapshotStoreConflict("Snapshot pin identity conflicts.")
            self._record_sqlite_operation(
                connection,
                kind="pin",
                operation_id=validated.operation_id,
                request_material=validated.identity_material(),
                response=receipt,
            )
            return _snapshot_model_json(receipt)

    async def pin_snapshot(self, request: AgentSnapshotPinRequest) -> AgentSnapshotPinReceipt:
        return AgentSnapshotPinReceipt.model_validate_json(
            await asyncio.to_thread(self._pin_snapshot_sync, request)
        )

    def _release_snapshot_pin_sync(self, request: AgentSnapshotReleaseRequest) -> str:
        validated = AgentSnapshotReleaseRequest.model_validate(request.model_dump(mode="json"))
        with self._write_connection() as connection:
            replay = self._sqlite_operation_replay(
                connection,
                kind="release",
                operation_id=validated.operation_id,
                request_material=validated.identity_material(),
                model=AgentSnapshotReleaseReceipt,
            )
            if replay is not None:
                return _snapshot_model_json(replay)
            self._load_access_closure_sync(connection, validated.access)
            row = connection.execute(
                "SELECT snapshot_root, binding_id, document FROM cayu_agent_snapshot_pins "
                "WHERE pin_id = ?",
                (validated.pin_id,),
            ).fetchone()
            if row is None:
                raise AgentSnapshotStoreConflict("Snapshot pin is unavailable.")
            try:
                pin = AgentSnapshotPinReceipt.model_validate_json(cast("str", row["document"]))
            except Exception as error:
                raise AgentSnapshotStoreConflict("Stored snapshot pin is invalid.") from error
            if (
                pin.snapshot != validated.access.snapshot
                or pin.binding_id != validated.access.binding_id
            ):
                raise AgentSnapshotAuthorizationError(
                    "Snapshot release names a pin outside its binding."
                )
            if pin.owner != validated.owner:
                raise AgentSnapshotAuthorizationError(
                    "Snapshot release owner does not own the pin."
                )
            receipt = AgentSnapshotReleaseReceipt.from_request(validated)
            connection.execute(
                "UPDATE cayu_agent_snapshot_pins SET released = 1 WHERE pin_id = ?",
                (pin.pin_id,),
            )
            self._record_sqlite_operation(
                connection,
                kind="release",
                operation_id=validated.operation_id,
                request_material=validated.identity_material(),
                response=receipt,
            )
            return _snapshot_model_json(receipt)

    async def release_snapshot_pin(
        self, request: AgentSnapshotReleaseRequest
    ) -> AgentSnapshotReleaseReceipt:
        return AgentSnapshotReleaseReceipt.model_validate_json(
            await asyncio.to_thread(self._release_snapshot_pin_sync, request)
        )

    def _protect_snapshot_sync(self, protection: AgentSnapshotProtection) -> str:
        validated = AgentSnapshotProtection.model_validate(protection.model_dump(mode="json"))
        material = validated.model_dump(mode="json")
        with self._write_connection() as connection:
            replay = self._sqlite_operation_replay(
                connection,
                kind="protect",
                operation_id=validated.operation_id,
                request_material=material,
                model=AgentSnapshotProtection,
            )
            if replay is not None:
                return _snapshot_model_json(replay)
            self._load_access_closure_sync(connection, validated.access)
            document = _snapshot_model_json(validated)
            connection.execute(
                "INSERT OR IGNORE INTO cayu_agent_snapshot_protections "
                "(protection_id, snapshot_root, binding_id, kind, document, released) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (
                    validated.protection_id,
                    validated.access.snapshot.snapshot_root,
                    validated.access.binding_id,
                    validated.kind.value,
                    document,
                ),
            )
            row = connection.execute(
                "SELECT document FROM cayu_agent_snapshot_protections WHERE protection_id = ?",
                (validated.protection_id,),
            ).fetchone()
            if row is None or row["document"] != document:
                raise AgentSnapshotStoreConflict("Snapshot protection identity conflicts.")
            self._record_sqlite_operation(
                connection,
                kind="protect",
                operation_id=validated.operation_id,
                request_material=material,
                response=validated,
            )
            return document

    async def protect_snapshot(
        self, protection: AgentSnapshotProtection
    ) -> AgentSnapshotProtection:
        return AgentSnapshotProtection.model_validate_json(
            await asyncio.to_thread(self._protect_snapshot_sync, protection)
        )

    def _release_snapshot_protection_sync(
        self,
        operation_id: str,
        access: AgentSnapshotAccess,
        protection_id: str,
    ) -> str:
        operation_id = _clean(operation_id, "operation_id", max_chars=256)
        protection_id = _sha256_hex(protection_id, "protection_id")
        request_material = {
            "operation_id": operation_id,
            "access": access.model_dump(mode="json"),
            "protection_id": protection_id,
        }
        with self._write_connection() as connection:
            replay = self._sqlite_operation_replay(
                connection,
                kind="unprotect",
                operation_id=operation_id,
                request_material=request_material,
                model=AgentSnapshotProtection,
            )
            if replay is not None:
                return _snapshot_model_json(replay)
            self._load_access_closure_sync(connection, access)
            row = connection.execute(
                "SELECT document FROM cayu_agent_snapshot_protections WHERE protection_id = ?",
                (protection_id,),
            ).fetchone()
            if row is None:
                raise AgentSnapshotStoreConflict("Snapshot protection is unavailable.")
            try:
                protection = AgentSnapshotProtection.model_validate_json(
                    cast("str", row["document"])
                )
            except Exception as error:
                raise AgentSnapshotStoreConflict(
                    "Stored snapshot protection is invalid."
                ) from error
            if protection.access != access:
                raise AgentSnapshotAuthorizationError(
                    "Snapshot protection is outside the supplied binding."
                )
            connection.execute(
                "UPDATE cayu_agent_snapshot_protections SET released = 1 WHERE protection_id = ?",
                (protection_id,),
            )
            self._record_sqlite_operation(
                connection,
                kind="unprotect",
                operation_id=operation_id,
                request_material=request_material,
                response=protection,
            )
            return _snapshot_model_json(protection)

    async def release_snapshot_protection(
        self,
        *,
        operation_id: str,
        access: AgentSnapshotAccess,
        protection_id: str,
    ) -> AgentSnapshotProtection:
        return AgentSnapshotProtection.model_validate_json(
            await asyncio.to_thread(
                self._release_snapshot_protection_sync,
                operation_id,
                access,
                protection_id,
            )
        )

    def _root_is_protected_sql(self, connection: sqlite3.Connection, snapshot_root: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM cayu_agent_snapshot_pins "
            "WHERE snapshot_root = ? AND released = 0 LIMIT 1",
            (snapshot_root,),
        ).fetchone()
        if row is not None:
            return True
        return (
            connection.execute(
                "SELECT 1 FROM cayu_agent_snapshot_protections "
                "WHERE snapshot_root = ? AND released = 0 LIMIT 1",
                (snapshot_root,),
            ).fetchone()
            is not None
        )

    def _plan_snapshot_gc_sync(self, request: AgentSnapshotGCRequest) -> str:
        validated = AgentSnapshotGCRequest.model_validate(request.model_dump(mode="json"))
        request_document = _snapshot_model_json(validated)
        request_digest = _content_sha256(
            validated.model_dump(mode="json"), "agent_snapshot_gc_request"
        )
        with self._write_connection() as connection:
            row = connection.execute(
                "SELECT plan_id, request_digest, plan_document "
                "FROM cayu_agent_snapshot_gc_plans WHERE operation_id = ?",
                (validated.operation_id,),
            ).fetchone()
            if row is not None:
                if row["request_digest"] != request_digest:
                    raise AgentSnapshotStoreConflict(
                        "Snapshot GC operation_id is already bound to another request."
                    )
                return cast("str", row["plan_document"])
            blocked: list[str] = []
            collectable: list[str] = []
            for access in validated.candidates:
                self._load_access_closure_sync(connection, access)
            accesses_by_root = _snapshot_accesses_by_root(validated.candidates)
            for root, accesses in accesses_by_root.items():
                authorized_binding_ids = {access.binding_id for access in accesses}
                stored_binding_ids = {
                    row["binding_id"]
                    for row in connection.execute(
                        "SELECT binding_id FROM cayu_agent_snapshot_bindings "
                        "WHERE snapshot_root = ?",
                        (root,),
                    ).fetchall()
                }
                if (
                    self._root_is_protected_sql(connection, root)
                    or authorized_binding_ids != stored_binding_ids
                ):
                    blocked.append(root)
                else:
                    collectable.append(root)
            collectable_set = set(collectable)
            candidate_nodes: set[str] = set()
            for root in collectable:
                candidate_nodes.update(
                    row["node_digest"]
                    for row in connection.execute(
                        "SELECT node_digest FROM cayu_agent_snapshot_root_nodes "
                        "WHERE snapshot_root = ?",
                        (root,),
                    ).fetchall()
                )
            deleting: set[str] = set()
            retained: set[str] = set()
            for digest in candidate_nodes:
                roots = {
                    row["snapshot_root"]
                    for row in connection.execute(
                        "SELECT snapshot_root FROM cayu_agent_snapshot_root_nodes "
                        "WHERE node_digest = ?",
                        (digest,),
                    ).fetchall()
                }
                (deleting if roots <= collectable_set else retained).add(digest)
            bytes_to_delete = 0
            for digest in deleting:
                node_row = connection.execute(
                    "SELECT byte_count FROM cayu_agent_snapshot_nodes WHERE digest = ?",
                    (digest,),
                ).fetchone()
                if node_row is None:
                    raise AgentSnapshotVerificationError(
                        "AgentSnapshot GC candidate contains a missing node."
                    )
                bytes_to_delete += cast("int", node_row["byte_count"])
            plan = _create_gc_plan(
                operation_id=validated.operation_id,
                authorized_candidates=validated.candidates,
                collectable_roots=collectable,
                blocked_roots=blocked,
                node_digests_to_delete=deleting,
                retained_shared_node_digests=retained,
                bytes_to_delete=bytes_to_delete,
            )
            plan_document = _snapshot_model_json(plan)
            connection.execute(
                "INSERT INTO cayu_agent_snapshot_gc_plans "
                "(operation_id, plan_id, request_digest, request_document, plan_document) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    validated.operation_id,
                    plan.plan_id,
                    request_digest,
                    request_document,
                    plan_document,
                ),
            )
            return plan_document

    async def plan_snapshot_gc(self, request: AgentSnapshotGCRequest) -> AgentSnapshotGCPlan:
        return AgentSnapshotGCPlan.model_validate_json(
            await asyncio.to_thread(self._plan_snapshot_gc_sync, request)
        )

    def _execute_snapshot_gc_sync(self, plan: AgentSnapshotGCPlan) -> str:
        validated = AgentSnapshotGCPlan.model_validate(plan.model_dump(mode="json"))
        with self._write_connection() as connection:
            receipt_row = connection.execute(
                "SELECT receipt_document FROM cayu_agent_snapshot_gc_receipts WHERE plan_id = ?",
                (validated.plan_id,),
            ).fetchone()
            if receipt_row is not None:
                return cast("str", receipt_row["receipt_document"])
            plan_row = connection.execute(
                "SELECT plan_id, plan_document FROM cayu_agent_snapshot_gc_plans "
                "WHERE operation_id = ?",
                (validated.operation_id,),
            ).fetchone()
            if plan_row is None:
                raise AgentSnapshotStoreConflict("Snapshot GC plan was not durably prepared.")
            try:
                stored = AgentSnapshotGCPlan.model_validate_json(
                    cast("str", plan_row["plan_document"])
                )
            except Exception as error:
                raise AgentSnapshotStoreConflict("Stored snapshot GC plan is invalid.") from error
            if stored != validated or plan_row["plan_id"] != validated.plan_id:
                raise AgentSnapshotStoreConflict("Snapshot GC plan differs from durable state.")
            collectable = set(validated.collectable_roots)
            accesses_by_root = _snapshot_accesses_by_root(validated.authorized_candidates)
            for root in collectable:
                accesses = accesses_by_root[root]
                for access in accesses:
                    self._load_access_closure_sync(connection, access)
                authorized_binding_ids = {access.binding_id for access in accesses}
                stored_binding_ids = {
                    row["binding_id"]
                    for row in connection.execute(
                        "SELECT binding_id FROM cayu_agent_snapshot_bindings "
                        "WHERE snapshot_root = ?",
                        (root,),
                    ).fetchall()
                }
                if authorized_binding_ids != stored_binding_ids:
                    raise AgentSnapshotStoreConflict(
                        "Snapshot GC binding reachability changed after planning."
                    )
                if self._root_is_protected_sql(connection, root):
                    raise AgentSnapshotStoreConflict(
                        "Snapshot GC root gained protection after planning."
                    )
                if (
                    connection.execute(
                        "SELECT 1 FROM cayu_agent_snapshot_roots WHERE snapshot_root = ?",
                        (root,),
                    ).fetchone()
                    is None
                ):
                    raise AgentSnapshotStoreConflict(
                        "Snapshot GC root disappeared before execution."
                    )
            candidate_nodes: set[str] = set()
            for root in collectable:
                candidate_nodes.update(
                    row["node_digest"]
                    for row in connection.execute(
                        "SELECT node_digest FROM cayu_agent_snapshot_root_nodes "
                        "WHERE snapshot_root = ?",
                        (root,),
                    ).fetchall()
                )
            actual_deleting: set[str] = set()
            for digest in candidate_nodes:
                roots = {
                    row["snapshot_root"]
                    for row in connection.execute(
                        "SELECT snapshot_root FROM cayu_agent_snapshot_root_nodes "
                        "WHERE node_digest = ?",
                        (digest,),
                    ).fetchall()
                }
                if roots <= collectable:
                    actual_deleting.add(digest)
            if actual_deleting != set(validated.node_digests_to_delete):
                raise AgentSnapshotStoreConflict("Snapshot GC reachability changed after planning.")
            deleted_bytes = 0
            for digest in actual_deleting:
                row = connection.execute(
                    "SELECT byte_count FROM cayu_agent_snapshot_nodes WHERE digest = ?",
                    (digest,),
                ).fetchone()
                if row is None:
                    raise AgentSnapshotStoreConflict("Snapshot GC node disappeared.")
                deleted_bytes += cast("int", row["byte_count"])
            if deleted_bytes != validated.bytes_to_delete:
                raise AgentSnapshotStoreConflict("Snapshot GC byte plan changed.")
            for root in collectable:
                connection.execute(
                    "DELETE FROM cayu_agent_snapshot_bindings WHERE snapshot_root = ?",
                    (root,),
                )
                connection.execute(
                    "DELETE FROM cayu_agent_snapshot_records "
                    "WHERE record_kind = 'snapshot' AND fingerprint = ?",
                    (root,),
                )
                connection.execute(
                    "DELETE FROM cayu_agent_snapshot_root_nodes WHERE snapshot_root = ?",
                    (root,),
                )
                connection.execute(
                    "DELETE FROM cayu_agent_snapshot_roots WHERE snapshot_root = ?",
                    (root,),
                )
            for digest in actual_deleting:
                remaining = connection.execute(
                    "SELECT 1 FROM cayu_agent_snapshot_root_nodes WHERE node_digest = ? LIMIT 1",
                    (digest,),
                ).fetchone()
                if remaining is None:
                    connection.execute(
                        "DELETE FROM cayu_agent_snapshot_nodes WHERE digest = ?",
                        (digest,),
                    )
            receipt = _create_gc_receipt(
                plan=validated,
                deleted_roots=collectable,
                deleted_node_digests=actual_deleting,
                deleted_bytes=deleted_bytes,
            )
            document = _snapshot_model_json(receipt)
            connection.execute(
                "INSERT INTO cayu_agent_snapshot_gc_receipts (plan_id, receipt_document) "
                "VALUES (?, ?)",
                (validated.plan_id, document),
            )
            return document

    async def execute_snapshot_gc(self, plan: AgentSnapshotGCPlan) -> AgentSnapshotGCReceipt:
        return AgentSnapshotGCReceipt.model_validate_json(
            await asyncio.to_thread(self._execute_snapshot_gc_sync, plan)
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
        with self._write_connection() as connection:
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
        with self._connection() as connection:
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
        with self._write_connection() as connection:
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
        with self._connection() as connection:
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
        with self._write_connection() as connection:
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
        with self._connection() as connection:
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
        with self._write_connection() as connection:
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
        with self._connection() as connection:
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
        with self._connection() as connection:
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
        await self.put_snapshot(snapshot, snapshot.identity_binding)
        return await self.get_snapshot(
            AgentSnapshotAccess(
                snapshot=snapshot.ref,
                binding_id=snapshot.identity_binding.binding_id,
                authority_scope_fingerprint=(snapshot.identity_binding.authority_scope_fingerprint),
            )
        )

    async def load_snapshot(self, snapshot_root: str) -> AgentSnapshot | None:
        return cast(
            "AgentSnapshot | None", await self._load("snapshot", snapshot_root, AgentSnapshot)
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
        snapshot = cast(
            "AgentSnapshot",
            _validate_store_response(
                await self.store.get_snapshot(request.access),
                AgentSnapshot,
                "authorized snapshot load",
            ),
        )
        if snapshot.fingerprint != request.snapshot_fingerprint:
            raise AgentSnapshotMaterializationError("Starting snapshot identity changed.")
        operation_digest = _content_sha256(
            request.model_dump(mode="json"), "agent_snapshot_materialization_protection"
        )
        protection = AgentSnapshotProtection.create(
            operation_id=f"materialize:{operation_digest}",
            access=request.access,
            kind=AgentSnapshotProtectionKind.MATERIALIZING,
            owner="agent-snapshot-coordinator",
            reason="materialization-in-progress",
        )
        await self.store.protect_snapshot(protection)
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
        result = await self._resume_materialization(snapshot, request, progress)
        await self.store.release_snapshot_protection(
            operation_id=f"materialized:{operation_digest}",
            access=request.access,
            protection_id=protection.protection_id,
        )
        return result

    async def _resume_materialization(
        self,
        snapshot: AgentSnapshot,
        request: AgentSnapshotMaterializationRequest,
        progress: AgentSnapshotMaterializationProgress,
    ) -> AgentSnapshotMaterialization:
        while True:
            if progress.materialization_fingerprint is not None:
                recovered = await self.recover_materialization(
                    progress.materialization_fingerprint,
                    access=request.access,
                )
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
                    recovered = await self.recover_materialization(
                        finalized.fingerprint,
                        access=request.access,
                    )
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
        *,
        access: AgentSnapshotAccess,
    ) -> AgentSnapshotMaterialization:
        _sha256_hex(fingerprint, "fingerprint")
        if type(access) is not AgentSnapshotAccess:
            raise TypeError("access must be an AgentSnapshotAccess.")
        access = AgentSnapshotAccess.model_validate(access.model_dump(mode="json"))
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
        if materialization.snapshot_fingerprint != access.snapshot.snapshot_root:
            raise AgentSnapshotAuthorizationError(
                "Materialization access belongs to another snapshot."
            )
        snapshot = await self.store.get_snapshot(access)
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
                    access=access,
                    candidate_id=materialization.candidate_id,
                    trial_id="recovery",
                    state_mode=materialization.state_mode,
                    state_partition_fingerprint=(materialization.state_partition_fingerprint),
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
            access=AgentSnapshotAccess(
                snapshot=snapshot.ref,
                binding_id=snapshot.identity_binding.binding_id,
                authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
            ),
            candidate_id=stored.candidate_id,
            trial_id=trial_id,
            state_mode=stored.state_mode,
            state_partition_fingerprint=stored.state_partition_fingerprint,
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
    "AGENT_SNAPSHOT_NODE_RECORD_TYPE",
    "AGENT_SNAPSHOT_NODE_SCHEMA_VERSION",
    "AGENT_SNAPSHOT_RECORD_TYPE",
    "AGENT_SNAPSHOT_SCHEMA_VERSION",
    "AGENT_SNAPSHOT_TRIAL_METADATA_KEY",
    "AgentSnapshot",
    "AgentSnapshotAccess",
    "AgentSnapshotAuthorityRef",
    "AgentSnapshotAuthorizationError",
    "AgentSnapshotCaptureError",
    "AgentSnapshotCaptureRequest",
    "AgentSnapshotClosureInspection",
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
    "AgentSnapshotGCPlan",
    "AgentSnapshotGCReceipt",
    "AgentSnapshotGCRequest",
    "AgentSnapshotIdentityBinding",
    "AgentSnapshotLearningDisposition",
    "AgentSnapshotLogicalRef",
    "AgentSnapshotMaterialization",
    "AgentSnapshotMaterializationCapability",
    "AgentSnapshotMaterializationError",
    "AgentSnapshotMaterializationOperation",
    "AgentSnapshotMaterializationProgress",
    "AgentSnapshotMaterializationRequest",
    "AgentSnapshotMaterializedComponent",
    "AgentSnapshotNode",
    "AgentSnapshotNodeChild",
    "AgentSnapshotNodeKind",
    "AgentSnapshotOverlayKind",
    "AgentSnapshotOverlayRef",
    "AgentSnapshotPinReceipt",
    "AgentSnapshotPinRequest",
    "AgentSnapshotProtection",
    "AgentSnapshotProtectionKind",
    "AgentSnapshotPutReceipt",
    "AgentSnapshotRedaction",
    "AgentSnapshotRef",
    "AgentSnapshotReleaseReceipt",
    "AgentSnapshotReleaseRequest",
    "AgentSnapshotResultBinding",
    "AgentSnapshotRetentionClass",
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
