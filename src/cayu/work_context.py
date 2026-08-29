"""Durable work-context and recall-checkpoint contracts.

``AgentWorkContext`` is application/runtime-owned task state.  It is not prompt
text, workflow authority, or the ephemeral ``RecallSituation``.  A checkpoint
records only which bounded freshness frontier was processed against one exact
work context and access-policy identity; it is not provider-exposure evidence.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator, model_validator

from cayu._clock import utc_clock
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    FrozenJsonDict,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    freeze_json_value,
    require_durable_clean_nonblank,
    thaw_json_value,
)

if TYPE_CHECKING:
    from cayu.recall_processing import AgentRecallProcessingResult

AGENT_WORK_CONTEXT_SCHEMA_VERSION = "cayu.agent_work_context.v1"
AGENT_WORK_CONTEXT_PUBLICATION_SCHEMA_VERSION = "cayu.agent_work_context_publication_receipt.v1"
AGENT_RECALL_CHECKPOINT_SCHEMA_VERSION = "cayu.agent_recall_checkpoint.v1"
AGENT_RECALL_DELIVERY_SCHEMA_VERSION = "cayu.agent_recall_delivery.v1"
AGENT_RECALL_DELIVERY_RECORD_SCHEMA_VERSION = "cayu.agent_recall_delivery_record.v1"
AGENT_RECALL_DELIVERY_CLAIM_SCHEMA_VERSION = "cayu.agent_recall_delivery_claim.v1"
AGENT_RECALL_DELIVERY_RELEASE_SCHEMA_VERSION = "cayu.agent_recall_delivery_release.v1"
AGENT_RECALL_DELIVERY_ACKNOWLEDGEMENT_SCHEMA_VERSION = (
    "cayu.agent_recall_delivery_acknowledgement.v1"
)

MAX_AGENT_WORK_CONTEXT_REVISION = 2_147_483_647
MAX_AGENT_WORK_CONTEXT_ID_BYTES = 512
MAX_AGENT_WORK_CONTEXT_GOAL_BYTES = 32_000
MAX_AGENT_WORK_CONTEXT_VALUE_BYTES = 4_096
MAX_AGENT_WORK_CONTEXT_VALUES = 128
MAX_AGENT_WORK_CONTEXT_BYTES = 256_000
MAX_AGENT_RECALL_DELIVERY_BYTES = 2_000_000
MAX_AGENT_RECALL_DELIVERY_LEASE_SECONDS = 86_400.0

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_CONTEXT_COLLECTION_FIELDS = (
    "scope_ids",
    "entity_ids",
    "artifact_ids",
    "repository_paths",
    "code_symbols",
    "planned_action_ids",
)


class _WorkContextModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        strict=True,
        validate_default=True,
    )

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        payload = self.model_dump(mode="python", round_trip=True)
        if update is not None:
            payload.update(update)
        return type(self).model_validate(payload)


def _bounded_identity(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > MAX_AGENT_WORK_CONTEXT_ID_BYTES:
        raise ValueError(
            f"`{field_name}` must be at most {MAX_AGENT_WORK_CONTEXT_ID_BYTES} UTF-8 bytes."
        )
    return value


def _bounded_value(value: str, field_name: str) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > MAX_AGENT_WORK_CONTEXT_VALUE_BYTES:
        raise ValueError(
            f"`{field_name}` must be at most {MAX_AGENT_WORK_CONTEXT_VALUE_BYTES} UTF-8 bytes."
        )
    return value


def _bounded_goal(value: str) -> str:
    value = require_durable_clean_nonblank(value, "goal")
    if len(value.encode("utf-8")) > MAX_AGENT_WORK_CONTEXT_GOAL_BYTES:
        raise ValueError(f"`goal` must be at most {MAX_AGENT_WORK_CONTEXT_GOAL_BYTES} UTF-8 bytes.")
    return value


def _ordered_unique_values(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"`{field_name}` must be a sequence of strings.")
    if len(value) > MAX_AGENT_WORK_CONTEXT_VALUES:
        raise ValueError(
            f"`{field_name}` cannot contain more than {MAX_AGENT_WORK_CONTEXT_VALUES} values."
        )
    copied: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str:
            raise ValueError(f"`{field_name}[{index}]` must be a string.")
        copied.append(_bounded_value(item, f"{field_name}[{index}]"))
    if len(copied) != len(set(copied)):
        raise ValueError(f"`{field_name}` cannot contain duplicates.")
    return tuple(sorted(copied))


def _copy_context_values(value: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"`{field_name}` must be a sequence of strings.")
    if len(value) > MAX_AGENT_WORK_CONTEXT_VALUES:
        raise ValueError(
            f"`{field_name}` cannot contain more than {MAX_AGENT_WORK_CONTEXT_VALUES} values."
        )
    return tuple(value)


def _sha256_hex(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise ValueError(f"`{field_name}` must be a lowercase SHA-256 digest.")
    return value


def _positive_revision(value: int, field_name: str) -> int:
    if type(value) is not int or not 1 <= value <= MAX_AGENT_WORK_CONTEXT_REVISION:
        raise ValueError(f"`{field_name}` must be between 1 and {MAX_AGENT_WORK_CONTEXT_REVISION}.")
    return value


def _sequence(value: int, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"`{field_name}` must be a non-negative durable JSON integer.")
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"`{field_name}` must be timezone-aware.")
    return value.astimezone(UTC)


class _AgentWorkContextContent(_WorkContextModel):
    task_id: str
    goal: str
    scope_ids: tuple[str, ...] = ()
    workflow_id: str | None = None
    workflow_phase: str | None = None
    workflow_iteration: int | None = None
    entity_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    repository_paths: tuple[str, ...] = ()
    code_symbols: tuple[str, ...] = ()
    planned_action_ids: tuple[str, ...] = ()

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        return _bounded_identity(value, "task_id")

    @field_validator("goal")
    @classmethod
    def validate_goal(cls, value: str) -> str:
        return _bounded_goal(value)

    @field_validator(*_CONTEXT_COLLECTION_FIELDS, mode="before")
    @classmethod
    def validate_collections(cls, value: object, info) -> tuple[str, ...]:
        return _ordered_unique_values(value, info.field_name)

    @field_validator("workflow_id", "workflow_phase")
    @classmethod
    def validate_optional_value(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_value(value, info.field_name)

    @field_validator("workflow_iteration")
    @classmethod
    def validate_workflow_iteration(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if type(value) is not int or not 0 <= value <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError("`workflow_iteration` must be a non-negative durable JSON integer.")
        return value

    @model_validator(mode="after")
    def validate_workflow_fields(self) -> _AgentWorkContextContent:
        if self.workflow_id is None and (
            self.workflow_phase is not None or self.workflow_iteration is not None
        ):
            raise ValueError("`workflow_phase` and `workflow_iteration` require `workflow_id`.")
        if self.workflow_phase is None and self.workflow_iteration is not None:
            raise ValueError("`workflow_iteration` requires `workflow_phase`.")
        payload = {
            "schema_version": AGENT_WORK_CONTEXT_SCHEMA_VERSION,
            "content": self.model_dump(mode="json"),
        }
        if len(canonical_durable_json_bytes(payload, "agent work context content")) > (
            MAX_AGENT_WORK_CONTEXT_BYTES
        ):
            raise ValueError("Agent work context content exceeds its serialized byte limit.")
        return self

    def content_fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                {
                    "schema_version": AGENT_WORK_CONTEXT_SCHEMA_VERSION,
                    "content": self.model_dump(mode="json"),
                },
                "agent work context content",
            )
        ).hexdigest()


class AgentWorkContext(_AgentWorkContextContent):
    """One immutable numbered revision of application-owned task state."""

    schema_version: Literal["cayu.agent_work_context.v1"] = AGENT_WORK_CONTEXT_SCHEMA_VERSION
    revision: int
    content_sha256: str
    operation_id: str
    published_by: str
    published_at: datetime

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: int) -> int:
        return _positive_revision(value, "revision")

    @field_validator("content_sha256")
    @classmethod
    def validate_content_sha256(cls, value: str) -> str:
        return _sha256_hex(value, "content_sha256")

    @field_validator("operation_id", "published_by")
    @classmethod
    def validate_publication_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("published_at")
    @classmethod
    def validate_published_at(cls, value: datetime) -> datetime:
        return _utc(value, "published_at")

    @model_validator(mode="after")
    def validate_content_identity(self) -> AgentWorkContext:
        content = _AgentWorkContextContent.model_validate(
            {
                field_name: getattr(self, field_name)
                for field_name in _AgentWorkContextContent.model_fields
            }
        )
        if self.content_sha256 != content.content_fingerprint():
            raise ValueError("`content_sha256` does not match the work-context content.")
        return self

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        goal: str,
        revision: int,
        operation_id: str,
        published_by: str,
        published_at: datetime,
        scope_ids: Sequence[str] = (),
        workflow_id: str | None = None,
        workflow_phase: str | None = None,
        workflow_iteration: int | None = None,
        entity_ids: Sequence[str] = (),
        artifact_ids: Sequence[str] = (),
        repository_paths: Sequence[str] = (),
        code_symbols: Sequence[str] = (),
        planned_action_ids: Sequence[str] = (),
    ) -> AgentWorkContext:
        content = _AgentWorkContextContent(
            task_id=task_id,
            goal=goal,
            scope_ids=_copy_context_values(scope_ids, "scope_ids"),
            workflow_id=workflow_id,
            workflow_phase=workflow_phase,
            workflow_iteration=workflow_iteration,
            entity_ids=_copy_context_values(entity_ids, "entity_ids"),
            artifact_ids=_copy_context_values(artifact_ids, "artifact_ids"),
            repository_paths=_copy_context_values(repository_paths, "repository_paths"),
            code_symbols=_copy_context_values(code_symbols, "code_symbols"),
            planned_action_ids=_copy_context_values(
                planned_action_ids,
                "planned_action_ids",
            ),
        )
        return cls(
            **content.model_dump(mode="python"),
            revision=revision,
            content_sha256=content.content_fingerprint(),
            operation_id=operation_id,
            published_by=published_by,
            published_at=published_at,
        )

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "agent work context",
            )
        ).hexdigest()


class AgentWorkContextPublicationReceipt(_WorkContextModel):
    """Immutable evidence for one changed or semantic no-change publication."""

    schema_version: Literal["cayu.agent_work_context_publication_receipt.v1"] = (
        AGENT_WORK_CONTEXT_PUBLICATION_SCHEMA_VERSION
    )
    operation_id: str
    request_sha256: str
    expected_revision: int | None
    requested_content_sha256: str
    changed: bool
    context: AgentWorkContext
    committed_at: datetime

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return _bounded_identity(value, "operation_id")

    @field_validator("request_sha256", "requested_content_sha256")
    @classmethod
    def validate_sha256(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("expected_revision")
    @classmethod
    def validate_expected_revision(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return _positive_revision(value, "expected_revision")

    @field_validator("changed", mode="before")
    @classmethod
    def validate_changed(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("`changed` must be a boolean.")
        return value

    @field_validator("committed_at")
    @classmethod
    def validate_committed_at(cls, value: datetime) -> datetime:
        return _utc(value, "committed_at")

    @model_validator(mode="after")
    def validate_result(self) -> AgentWorkContextPublicationReceipt:
        if self.requested_content_sha256 != self.context.content_sha256:
            raise ValueError("Publication result content does not match the request.")
        if not self.changed and self.expected_revision is None:
            raise ValueError("No-change publication requires an expected revision.")
        expected_result_revision = (
            1 if self.expected_revision is None else self.expected_revision + int(self.changed)
        )
        if self.context.revision != expected_result_revision:
            raise ValueError("Publication result revision is inconsistent.")
        if self.changed and self.context.operation_id != self.operation_id:
            raise ValueError("Changed publication must retain its operation identity.")
        if not self.changed and self.context.operation_id == self.operation_id:
            raise ValueError("No-change publication requires a distinct operation identity.")
        return self

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "agent work context publication receipt",
            )
        ).hexdigest()


class AgentRecallCheckpointMode(StrEnum):
    FULL_INDEX = "full_index"
    DELTA = "delta"


class AgentRecallCheckpointKey(_WorkContextModel):
    agent_id: str
    task_id: str
    knowledge_namespace: str
    access_policy_sha256: str

    @field_validator("agent_id", "task_id", "knowledge_namespace")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("access_policy_sha256")
    @classmethod
    def validate_access_policy_sha256(cls, value: str) -> str:
        return _sha256_hex(value, "access_policy_sha256")

    def sort_key(self) -> tuple[str, str, str, str]:
        return (
            self.agent_id,
            self.task_id,
            self.knowledge_namespace,
            self.access_policy_sha256,
        )

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                {
                    "schema_version": AGENT_RECALL_CHECKPOINT_SCHEMA_VERSION,
                    "key": self.model_dump(mode="json"),
                },
                "agent recall checkpoint key",
            )
        ).hexdigest()


class AgentRecallCheckpoint(_WorkContextModel):
    """Processed freshness frontier for one exact agent/task/access view."""

    schema_version: Literal["cayu.agent_recall_checkpoint.v1"] = (
        AGENT_RECALL_CHECKPOINT_SCHEMA_VERSION
    )
    agent_id: str
    task_id: str
    knowledge_namespace: str
    access_policy_sha256: str
    revision: int
    work_context_revision: int
    work_context_sha256: str
    knowledge_sequence: int
    index_readiness_sequence: int
    knowledge_high_water_sequence: int
    index_readiness_high_water_sequence: int
    processing_mode: AgentRecallCheckpointMode
    processing_id: str
    operation_id: str
    updated_by: str
    updated_at: datetime

    @field_validator("agent_id", "task_id", "knowledge_namespace")
    @classmethod
    def validate_key_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("processing_id", "operation_id", "updated_by")
    @classmethod
    def validate_processing_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("access_policy_sha256", "work_context_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("revision", "work_context_revision")
    @classmethod
    def validate_positive_revision(cls, value: int, info) -> int:
        return _positive_revision(value, info.field_name)

    @field_validator(
        "knowledge_sequence",
        "index_readiness_sequence",
        "knowledge_high_water_sequence",
        "index_readiness_high_water_sequence",
    )
    @classmethod
    def validate_sequence(cls, value: int, info) -> int:
        return _sequence(value, info.field_name)

    @field_validator("processing_mode", mode="before")
    @classmethod
    def validate_processing_mode(cls, value: object) -> AgentRecallCheckpointMode:
        if isinstance(value, AgentRecallCheckpointMode):
            return value
        if type(value) is str:
            return AgentRecallCheckpointMode(value)
        raise ValueError("`processing_mode` must be an AgentRecallCheckpointMode.")

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime) -> datetime:
        return _utc(value, "updated_at")

    @model_validator(mode="after")
    def validate_captured_frontiers(self) -> AgentRecallCheckpoint:
        if self.knowledge_sequence > self.knowledge_high_water_sequence:
            raise ValueError("`knowledge_sequence` cannot exceed `knowledge_high_water_sequence`.")
        if self.index_readiness_sequence > self.index_readiness_high_water_sequence:
            raise ValueError(
                "`index_readiness_sequence` cannot exceed `index_readiness_high_water_sequence`."
            )
        return self

    def key(self) -> AgentRecallCheckpointKey:
        return AgentRecallCheckpointKey(
            agent_id=self.agent_id,
            task_id=self.task_id,
            knowledge_namespace=self.knowledge_namespace,
            access_policy_sha256=self.access_policy_sha256,
        )

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "agent recall checkpoint",
            )
        ).hexdigest()


class AgentRecallDeliveryState(StrEnum):
    """Durable handoff state; acknowledgement is not provider exposure."""

    PENDING = "pending"
    CLAIMED = "claimed"
    ACKNOWLEDGED = "acknowledged"


class AgentRecallDeliveryEvidenceKind(StrEnum):
    """Kind of durable downstream evidence that accepted a staged result."""

    RECALL_RECEIPT = "recall_receipt"
    CONTEXT_EXPOSURE = "context_exposure"
    APPLICATION_HANDOFF = "application_handoff"


class AgentRecallDelivery(_WorkContextModel):
    """Immutable materialized recall staged beside its committed checkpoint."""

    schema_version: Literal["cayu.agent_recall_delivery.v1"] = AGENT_RECALL_DELIVERY_SCHEMA_VERSION
    delivery_id: str
    agent_id: str
    task_id: str
    knowledge_namespace: str
    access_policy_sha256: str
    work_context_revision: int
    work_context_sha256: str
    processing_id: str
    operation_id: str
    processing_result_sha256: str
    processing_result: Mapping[str, Any]
    checkpoint: AgentRecallCheckpoint
    checkpoint_sha256: str
    processing_mode: AgentRecallCheckpointMode
    expected_checkpoint_revision: int | None
    staged_by: str
    staged_at: datetime

    @field_validator(
        "delivery_id",
        "agent_id",
        "task_id",
        "knowledge_namespace",
        "processing_id",
        "operation_id",
        "staged_by",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator(
        "access_policy_sha256",
        "work_context_sha256",
        "processing_result_sha256",
        "checkpoint_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("work_context_revision")
    @classmethod
    def validate_work_context_revision(cls, value: int) -> int:
        return _positive_revision(value, "work_context_revision")

    @field_validator("expected_checkpoint_revision")
    @classmethod
    def validate_expected_checkpoint_revision(cls, value: int | None) -> int | None:
        if value is not None:
            _positive_revision(value, "expected_checkpoint_revision")
        return value

    @field_validator("processing_result", mode="before")
    @classmethod
    def copy_processing_result(cls, value: object) -> dict[str, Any]:
        from cayu.recall_processing import AgentRecallProcessingResult

        if type(value) is AgentRecallProcessingResult:
            value = value.model_dump(mode="json")
        copied = copy_durable_json_object(value, "processing_result")
        result = AgentRecallProcessingResult.model_validate_json(
            canonical_durable_json_bytes(copied, "processing_result")
        )
        return result.model_dump(mode="json")

    @field_validator("processing_result")
    @classmethod
    def freeze_processing_result(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        frozen = freeze_json_value(dict(value))
        if type(frozen) is not FrozenJsonDict:  # pragma: no cover - defensive invariant
            raise AssertionError("Agent recall processing result did not freeze as an object.")
        return frozen

    @field_serializer("processing_result")
    def serialize_processing_result(self, value: Mapping[str, Any]) -> dict[str, Any]:
        thawed = thaw_json_value(value)
        if type(thawed) is not dict:  # pragma: no cover - defensive invariant
            raise AssertionError("Agent recall processing result did not thaw as an object.")
        return thawed

    @field_validator("checkpoint", mode="before")
    @classmethod
    def copy_checkpoint(cls, value: object, info) -> AgentRecallCheckpoint:
        if type(value) is AgentRecallCheckpoint:
            return copy_agent_recall_checkpoint(value)
        if info.mode == "json":
            return AgentRecallCheckpoint.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall delivery checkpoint")
            )
        return AgentRecallCheckpoint.model_validate(value)

    @field_validator("processing_mode", mode="before")
    @classmethod
    def validate_processing_mode(cls, value: object) -> AgentRecallCheckpointMode:
        if isinstance(value, AgentRecallCheckpointMode):
            return value
        if type(value) is str:
            return AgentRecallCheckpointMode(value)
        raise ValueError("processing_mode must be an AgentRecallCheckpointMode.")

    @field_validator("staged_at")
    @classmethod
    def validate_staged_at(cls, value: datetime) -> datetime:
        return _utc(value, "staged_at")

    @model_validator(mode="after")
    def validate_processing_authority(self) -> AgentRecallDelivery:
        from cayu.recall_processing import AgentRecallProcessingMode, AgentRecallProcessingResult

        result = AgentRecallProcessingResult.model_validate_json(
            canonical_durable_json_bytes(self.processing_result, "processing_result")
        )
        checkpoint = result.proposed_checkpoint
        if result.mode is AgentRecallProcessingMode.NO_WORK or checkpoint is None:
            raise ValueError("Only processing work with a checkpoint proposal can be staged.")
        if result.fingerprint() != self.processing_result_sha256:
            raise ValueError("Processing result fingerprint does not match its staged payload.")
        if checkpoint != self.checkpoint:
            raise ValueError("Staged checkpoint does not match the processing result proposal.")
        if self.checkpoint_sha256 != checkpoint.fingerprint():
            raise ValueError("Staged checkpoint fingerprint does not match its exact proposal.")
        if self.processing_mode is not checkpoint.processing_mode:
            raise ValueError("Staged processing mode does not match its checkpoint proposal.")
        if self.staged_at < checkpoint.updated_at:
            raise ValueError("A staged delivery cannot predate its checkpoint proposal.")
        expected_revision = (
            1
            if self.expected_checkpoint_revision is None
            else self.expected_checkpoint_revision + 1
        )
        if checkpoint.revision != expected_revision:
            raise ValueError("Staged checkpoint revision does not follow its expected revision.")
        if (
            self.agent_id != result.agent_id
            or self.task_id != result.task_id
            or self.knowledge_namespace != result.knowledge_namespace
            or self.access_policy_sha256 != result.access_policy_sha256
            or self.work_context_revision != result.work_context_revision
            or self.work_context_sha256 != result.work_context_sha256
            or self.processing_id != result.processing_id
            or self.operation_id != result.operation_id
            or self.operation_id != checkpoint.operation_id
        ):
            raise ValueError("Staged delivery authority conflicts with its processing result.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "agent recall delivery",
                )
            )
            > MAX_AGENT_RECALL_DELIVERY_BYTES
        ):
            raise ValueError(
                f"Agent recall delivery must be at most {MAX_AGENT_RECALL_DELIVERY_BYTES} bytes."
            )
        return self

    @classmethod
    def from_processing_result(
        cls,
        result: AgentRecallProcessingResult,
        *,
        delivery_id: str,
        expected_checkpoint_revision: int | None,
        staged_by: str,
        staged_at: datetime,
    ) -> AgentRecallDelivery:
        from cayu.recall_processing import AgentRecallProcessingResult

        if type(result) is not AgentRecallProcessingResult:
            raise TypeError("result must be an AgentRecallProcessingResult.")
        copied = result.model_copy(deep=True)
        checkpoint = copied.proposed_checkpoint
        if checkpoint is None:
            raise ValueError("Processing result has no checkpoint proposal to stage.")
        return cls(
            delivery_id=delivery_id,
            agent_id=copied.agent_id,
            task_id=copied.task_id,
            knowledge_namespace=copied.knowledge_namespace,
            access_policy_sha256=copied.access_policy_sha256,
            work_context_revision=copied.work_context_revision,
            work_context_sha256=copied.work_context_sha256,
            processing_id=copied.processing_id,
            operation_id=copied.operation_id,
            processing_result_sha256=copied.fingerprint(),
            processing_result=copied.model_dump(mode="json"),
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint.fingerprint(),
            processing_mode=checkpoint.processing_mode,
            expected_checkpoint_revision=expected_checkpoint_revision,
            staged_by=staged_by,
            staged_at=staged_at,
        )

    def materialized_result(self) -> AgentRecallProcessingResult:
        from cayu.recall_processing import AgentRecallProcessingResult

        return AgentRecallProcessingResult.model_validate_json(
            canonical_durable_json_bytes(self.processing_result, "processing_result")
        )

    def key(self) -> AgentRecallCheckpointKey:
        return self.checkpoint.key()

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "agent recall delivery",
            )
        ).hexdigest()


class AgentRecallDeliveryClaim(_WorkContextModel):
    """Lease-fenced authority over one staged delivery."""

    schema_version: Literal["cayu.agent_recall_delivery_claim.v1"] = (
        AGENT_RECALL_DELIVERY_CLAIM_SCHEMA_VERSION
    )
    delivery_id: str
    processing_result_sha256: str
    claim_id: str
    worker_id: str
    attempt: int
    state_revision: int
    claimed_at: datetime
    lease_expires_at: datetime

    @field_validator("delivery_id", "claim_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("processing_result_sha256")
    @classmethod
    def validate_result_digest(cls, value: str) -> str:
        return _sha256_hex(value, "processing_result_sha256")

    @field_validator("attempt")
    @classmethod
    def validate_attempt(cls, value: int) -> int:
        return _positive_revision(value, "attempt")

    @field_validator("state_revision")
    @classmethod
    def validate_state_revision(cls, value: int) -> int:
        return _positive_revision(value, "state_revision")

    @field_validator("claimed_at", "lease_expires_at")
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        return _utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_lease(self) -> AgentRecallDeliveryClaim:
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("Delivery claim lease must expire after it is claimed.")
        return self


class AgentRecallDeliveryAcknowledgement(_WorkContextModel):
    """Durable downstream handoff evidence; not provider-exposure evidence."""

    schema_version: Literal["cayu.agent_recall_delivery_acknowledgement.v1"] = (
        AGENT_RECALL_DELIVERY_ACKNOWLEDGEMENT_SCHEMA_VERSION
    )
    acknowledgement_id: str
    delivery_id: str
    processing_result_sha256: str
    claim_id: str
    worker_id: str
    attempt: int
    evidence_kind: AgentRecallDeliveryEvidenceKind
    evidence_ref: str
    acknowledged_at: datetime

    @field_validator(
        "acknowledgement_id",
        "delivery_id",
        "claim_id",
        "worker_id",
        "evidence_ref",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("processing_result_sha256")
    @classmethod
    def validate_result_digest(cls, value: str) -> str:
        return _sha256_hex(value, "processing_result_sha256")

    @field_validator("evidence_kind", mode="before")
    @classmethod
    def validate_evidence_kind(cls, value: object) -> AgentRecallDeliveryEvidenceKind:
        if isinstance(value, AgentRecallDeliveryEvidenceKind):
            return value
        if type(value) is str:
            return AgentRecallDeliveryEvidenceKind(value)
        raise ValueError("evidence_kind must be an AgentRecallDeliveryEvidenceKind.")

    @field_validator("attempt")
    @classmethod
    def validate_attempt(cls, value: int) -> int:
        return _positive_revision(value, "attempt")

    @field_validator("acknowledged_at")
    @classmethod
    def validate_acknowledged_at(cls, value: datetime) -> datetime:
        return _utc(value, "acknowledged_at")


class AgentRecallDeliveryRelease(_WorkContextModel):
    """Immutable evidence that one claim returned its stage for retry."""

    schema_version: Literal["cayu.agent_recall_delivery_release.v1"] = (
        AGENT_RECALL_DELIVERY_RELEASE_SCHEMA_VERSION
    )
    release_id: str
    delivery_id: str
    processing_result_sha256: str
    claim_id: str
    worker_id: str
    attempt: int
    claim_state_revision: int
    reason: str
    released_at: datetime

    @field_validator("release_id", "delivery_id", "claim_id", "worker_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("processing_result_sha256")
    @classmethod
    def validate_result_digest(cls, value: str) -> str:
        return _sha256_hex(value, "processing_result_sha256")

    @field_validator("attempt", "claim_state_revision")
    @classmethod
    def validate_positive_counter(cls, value: int, info) -> int:
        return _positive_revision(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        return _bounded_value(value, "reason")

    @field_validator("released_at")
    @classmethod
    def validate_released_at(cls, value: datetime) -> datetime:
        return _utc(value, "released_at")

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "agent recall delivery release",
            )
        ).hexdigest()


class AgentRecallDeliveryRecord(_WorkContextModel):
    """Current delivery state plus its immutable staged payload and acknowledgement."""

    schema_version: Literal["cayu.agent_recall_delivery_record.v1"] = (
        AGENT_RECALL_DELIVERY_RECORD_SCHEMA_VERSION
    )
    delivery: AgentRecallDelivery
    state: AgentRecallDeliveryState = AgentRecallDeliveryState.PENDING
    state_revision: int = 0
    attempt: int = 0
    claim: AgentRecallDeliveryClaim | None = None
    release: AgentRecallDeliveryRelease | None = None
    acknowledgement: AgentRecallDeliveryAcknowledgement | None = None
    updated_at: datetime

    @field_validator("delivery", mode="before")
    @classmethod
    def copy_delivery(cls, value: object, info) -> AgentRecallDelivery:
        if type(value) is AgentRecallDelivery:
            return copy_agent_recall_delivery(value)
        if info.mode == "json":
            return AgentRecallDelivery.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall delivery")
            )
        return AgentRecallDelivery.model_validate(value)

    @field_validator("state", mode="before")
    @classmethod
    def validate_state(cls, value: object) -> AgentRecallDeliveryState:
        if isinstance(value, AgentRecallDeliveryState):
            return value
        if type(value) is str:
            return AgentRecallDeliveryState(value)
        raise ValueError("state must be an AgentRecallDeliveryState.")

    @field_validator("state_revision", "attempt")
    @classmethod
    def validate_nonnegative_counter(cls, value: int, info) -> int:
        return _sequence(value, info.field_name)

    @field_validator("claim", mode="before")
    @classmethod
    def copy_claim(cls, value: object, info) -> AgentRecallDeliveryClaim | None:
        if value is None:
            return None
        if type(value) is AgentRecallDeliveryClaim:
            return copy_agent_recall_delivery_claim(value)
        if info.mode == "json":
            return AgentRecallDeliveryClaim.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall delivery claim")
            )
        return AgentRecallDeliveryClaim.model_validate(value)

    @field_validator("release", mode="before")
    @classmethod
    def copy_release(cls, value: object, info) -> AgentRecallDeliveryRelease | None:
        if value is None:
            return None
        if type(value) is AgentRecallDeliveryRelease:
            return copy_agent_recall_delivery_release(value)
        if info.mode == "json":
            return AgentRecallDeliveryRelease.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall delivery release")
            )
        return AgentRecallDeliveryRelease.model_validate(value)

    @field_validator("acknowledgement", mode="before")
    @classmethod
    def copy_acknowledgement(
        cls,
        value: object,
        info,
    ) -> AgentRecallDeliveryAcknowledgement | None:
        if value is None:
            return None
        if type(value) is AgentRecallDeliveryAcknowledgement:
            return copy_agent_recall_delivery_acknowledgement(value)
        if info.mode == "json":
            return AgentRecallDeliveryAcknowledgement.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall delivery acknowledgement")
            )
        return AgentRecallDeliveryAcknowledgement.model_validate(value)

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime) -> datetime:
        return _utc(value, "updated_at")

    @model_validator(mode="after")
    def validate_lifecycle(self) -> AgentRecallDeliveryRecord:
        if self.updated_at < self.delivery.staged_at:
            raise ValueError("Delivery state cannot predate its immutable stage.")
        if self.state is AgentRecallDeliveryState.PENDING:
            initial = (
                self.state_revision == 0
                and self.attempt == 0
                and self.claim is None
                and self.release is None
            )
            released = (
                self.state_revision > 0
                and self.attempt > 0
                and self.claim is not None
                and self.release is not None
            )
            if self.acknowledgement is not None or not (initial or released):
                raise ValueError(
                    "Pending delivery must be initial or retain exact release evidence."
                )
        elif self.state is AgentRecallDeliveryState.CLAIMED:
            if self.claim is None or self.release is not None or self.acknowledgement is not None:
                raise ValueError("Claimed delivery requires exactly one live claim.")
        elif self.claim is None or self.release is not None or self.acknowledgement is None:
            raise ValueError("Acknowledged delivery requires its final claim and evidence.")
        if self.claim is not None and (
            self.claim.delivery_id != self.delivery.delivery_id
            or self.claim.processing_result_sha256 != self.delivery.processing_result_sha256
            or self.claim.attempt != self.attempt
            or self.claim.claimed_at < self.delivery.staged_at
            or self.updated_at < self.claim.claimed_at
            or (
                self.state is AgentRecallDeliveryState.CLAIMED
                and self.claim.state_revision != self.state_revision
            )
            or (
                self.state is not AgentRecallDeliveryState.CLAIMED
                and self.claim.state_revision + 1 != self.state_revision
            )
        ):
            raise ValueError("Delivery claim conflicts with current staged state.")
        if self.release is not None and (
            self.claim is None
            or self.release.delivery_id != self.delivery.delivery_id
            or self.release.processing_result_sha256 != self.delivery.processing_result_sha256
            or self.release.claim_id != self.claim.claim_id
            or self.release.worker_id != self.claim.worker_id
            or self.release.attempt != self.claim.attempt
            or self.release.claim_state_revision != self.claim.state_revision
            or self.release.released_at < self.claim.claimed_at
            or self.release.released_at >= self.claim.lease_expires_at
            or self.updated_at < self.release.released_at
        ):
            raise ValueError("Delivery release conflicts with its retained claim.")
        if self.acknowledgement is not None and (
            self.claim is None
            or (
                self.acknowledgement.delivery_id != self.delivery.delivery_id
                or self.acknowledgement.processing_result_sha256
                != self.delivery.processing_result_sha256
                or self.acknowledgement.claim_id != self.claim.claim_id
                or self.acknowledgement.worker_id != self.claim.worker_id
                or self.acknowledgement.attempt != self.claim.attempt
                or self.acknowledgement.acknowledged_at < self.claim.claimed_at
                or self.acknowledgement.acknowledged_at >= self.claim.lease_expires_at
                or self.updated_at < self.acknowledgement.acknowledged_at
            )
        ):
            raise ValueError("Delivery acknowledgement conflicts with its final claim.")
        return self


class AgentRecallDeliveryConflict(RuntimeError):
    """A staged-delivery operation lost or reused exact transition authority."""

    def __init__(self, code: str) -> None:
        self.code = require_durable_clean_nonblank(code, "code")
        super().__init__(f"Agent recall delivery conflicted ({self.code}).")


class AgentWorkContextConflict(RuntimeError):
    """A publication or checkpoint could not satisfy its exact authority tuple."""

    def __init__(self, code: str) -> None:
        self.code = require_durable_clean_nonblank(code, "code")
        super().__init__(f"Agent work-context operation conflicted ({self.code}).")


def copy_agent_work_context(value: AgentWorkContext) -> AgentWorkContext:
    if type(value) is not AgentWorkContext:
        raise TypeError("value must be an AgentWorkContext.")
    return AgentWorkContext.model_validate(value.model_dump(mode="python"))


def copy_agent_work_context_publication_receipt(
    value: AgentWorkContextPublicationReceipt,
) -> AgentWorkContextPublicationReceipt:
    if type(value) is not AgentWorkContextPublicationReceipt:
        raise TypeError("value must be an AgentWorkContextPublicationReceipt.")
    return AgentWorkContextPublicationReceipt.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_checkpoint(value: AgentRecallCheckpoint) -> AgentRecallCheckpoint:
    if type(value) is not AgentRecallCheckpoint:
        raise TypeError("value must be an AgentRecallCheckpoint.")
    return AgentRecallCheckpoint.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_checkpoint_key(
    value: AgentRecallCheckpointKey,
) -> AgentRecallCheckpointKey:
    if type(value) is not AgentRecallCheckpointKey:
        raise TypeError("value must be an AgentRecallCheckpointKey.")
    return AgentRecallCheckpointKey.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_delivery(value: AgentRecallDelivery) -> AgentRecallDelivery:
    if type(value) is not AgentRecallDelivery:
        raise TypeError("value must be an AgentRecallDelivery.")
    return AgentRecallDelivery.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_delivery_claim(
    value: AgentRecallDeliveryClaim,
) -> AgentRecallDeliveryClaim:
    if type(value) is not AgentRecallDeliveryClaim:
        raise TypeError("value must be an AgentRecallDeliveryClaim.")
    return AgentRecallDeliveryClaim.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_delivery_acknowledgement(
    value: AgentRecallDeliveryAcknowledgement,
) -> AgentRecallDeliveryAcknowledgement:
    if type(value) is not AgentRecallDeliveryAcknowledgement:
        raise TypeError("value must be an AgentRecallDeliveryAcknowledgement.")
    return AgentRecallDeliveryAcknowledgement.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_delivery_release(
    value: AgentRecallDeliveryRelease,
) -> AgentRecallDeliveryRelease:
    if type(value) is not AgentRecallDeliveryRelease:
        raise TypeError("value must be an AgentRecallDeliveryRelease.")
    return AgentRecallDeliveryRelease.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_delivery_record(
    value: AgentRecallDeliveryRecord,
) -> AgentRecallDeliveryRecord:
    if type(value) is not AgentRecallDeliveryRecord:
        raise TypeError("value must be an AgentRecallDeliveryRecord.")
    return AgentRecallDeliveryRecord.model_validate(value.model_dump(mode="python"))


def _validate_delivery_lease_seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("lease_seconds must be a number.")
    result = float(value)
    if not 0 < result <= MAX_AGENT_RECALL_DELIVERY_LEASE_SECONDS:
        raise ValueError(
            "lease_seconds must be greater than zero and at most "
            f"{MAX_AGENT_RECALL_DELIVERY_LEASE_SECONDS}."
        )
    return result


def agent_recall_delivery_claim_request_sha256(
    key: AgentRecallCheckpointKey,
    *,
    claim_id: str,
    worker_id: str,
    lease_seconds: float,
) -> str:
    key = copy_agent_recall_checkpoint_key(key)
    claim_id = _bounded_identity(claim_id, "claim_id")
    worker_id = _bounded_identity(worker_id, "worker_id")
    lease_seconds = _validate_delivery_lease_seconds(lease_seconds)
    return sha256(
        canonical_durable_json_bytes(
            {
                "key": key.model_dump(mode="json"),
                "claim_id": claim_id,
                "worker_id": worker_id,
                "lease_seconds": lease_seconds,
            },
            "agent recall delivery claim request",
        )
    ).hexdigest()


def _claim_agent_recall_delivery_record(
    record: AgentRecallDeliveryRecord,
    *,
    claim_id: str,
    worker_id: str,
    lease_seconds: float,
    now: datetime,
) -> AgentRecallDeliveryRecord:
    record = copy_agent_recall_delivery_record(record)
    claim_id = _bounded_identity(claim_id, "claim_id")
    worker_id = _bounded_identity(worker_id, "worker_id")
    lease_seconds = _validate_delivery_lease_seconds(lease_seconds)
    now = _utc(now, "now")
    if record.state is AgentRecallDeliveryState.ACKNOWLEDGED:
        raise AgentRecallDeliveryConflict("delivery_already_acknowledged")
    if record.state is AgentRecallDeliveryState.CLAIMED:
        assert record.claim is not None
        if record.claim.claim_id == claim_id and record.claim.worker_id == worker_id:
            return record
        if record.claim.lease_expires_at > now:
            raise AgentRecallDeliveryConflict("delivery_claimed")
    state_revision = record.state_revision + 1
    claim = AgentRecallDeliveryClaim(
        delivery_id=record.delivery.delivery_id,
        processing_result_sha256=record.delivery.processing_result_sha256,
        claim_id=claim_id,
        worker_id=worker_id,
        attempt=record.attempt + 1,
        state_revision=state_revision,
        claimed_at=now,
        lease_expires_at=now + timedelta(seconds=lease_seconds),
    )
    return AgentRecallDeliveryRecord(
        delivery=record.delivery,
        state=AgentRecallDeliveryState.CLAIMED,
        state_revision=state_revision,
        attempt=claim.attempt,
        claim=claim,
        release=None,
        updated_at=now,
    )


def _require_current_delivery_claim(
    record: AgentRecallDeliveryRecord,
    claim: AgentRecallDeliveryClaim,
    *,
    now: datetime,
    require_live: bool = True,
) -> None:
    claim = copy_agent_recall_delivery_claim(claim)
    now = _utc(now, "now")
    if record.state is not AgentRecallDeliveryState.CLAIMED or record.claim != claim:
        raise AgentRecallDeliveryConflict("stale_delivery_claim")
    if require_live and claim.lease_expires_at <= now:
        raise AgentRecallDeliveryConflict("expired_delivery_claim")


def _require_replayable_delivery_claim_attempt(
    record: AgentRecallDeliveryRecord,
    *,
    claim_id: str,
    worker_id: str,
    attempt: int,
    now: datetime,
) -> None:
    """Fence a claim replay after expiry or a newer ownership attempt."""

    now = _utc(now, "now")
    current = record.claim
    if (
        current is None
        or current.claim_id != claim_id
        or current.worker_id != worker_id
        or current.attempt != attempt
    ):
        raise AgentRecallDeliveryConflict("claim_replay_superseded")
    if record.state is AgentRecallDeliveryState.CLAIMED and current.lease_expires_at <= now:
        raise AgentRecallDeliveryConflict("expired_delivery_claim")


def _renew_agent_recall_delivery_record(
    record: AgentRecallDeliveryRecord,
    claim: AgentRecallDeliveryClaim,
    *,
    lease_seconds: float,
    now: datetime,
) -> AgentRecallDeliveryRecord:
    record = copy_agent_recall_delivery_record(record)
    claim = copy_agent_recall_delivery_claim(claim)
    lease_seconds = _validate_delivery_lease_seconds(lease_seconds)
    now = _utc(now, "now")
    if (
        record.state is AgentRecallDeliveryState.CLAIMED
        and record.claim is not None
        and record.claim.claim_id == claim.claim_id
        and record.claim.worker_id == claim.worker_id
        and record.claim.attempt == claim.attempt
        and record.claim.state_revision == claim.state_revision + 1
    ):
        if record.claim.lease_expires_at != record.updated_at + timedelta(seconds=lease_seconds):
            raise AgentRecallDeliveryConflict("renewal_reused")
        if record.claim.lease_expires_at <= now:
            raise AgentRecallDeliveryConflict("expired_delivery_claim")
        return record
    _require_current_delivery_claim(record, claim, now=now)
    state_revision = record.state_revision + 1
    renewed = claim.model_copy(
        update={
            "state_revision": state_revision,
            "lease_expires_at": now + timedelta(seconds=lease_seconds),
        }
    )
    return record.model_copy(
        update={
            "state_revision": state_revision,
            "claim": renewed,
            "updated_at": now,
        }
    )


def _release_agent_recall_delivery_record(
    record: AgentRecallDeliveryRecord,
    claim: AgentRecallDeliveryClaim,
    *,
    release_id: str,
    reason: str,
    released_at: datetime,
    now: datetime,
) -> AgentRecallDeliveryRecord:
    record = copy_agent_recall_delivery_record(record)
    claim = copy_agent_recall_delivery_claim(claim)
    now = _utc(now, "now")
    requested = _agent_recall_delivery_release(
        claim,
        release_id=release_id,
        reason=reason,
        released_at=released_at,
    )
    if record.state is AgentRecallDeliveryState.PENDING and record.release == requested:
        return record
    _require_current_delivery_claim(record, claim, now=now)
    if requested.released_at < claim.claimed_at:
        raise AgentRecallDeliveryConflict("release_predates_claim")
    if requested.released_at >= claim.lease_expires_at:
        raise AgentRecallDeliveryConflict("release_outside_claim_lease")
    if requested.released_at > now:
        raise AgentRecallDeliveryConflict("release_from_future")
    return record.model_copy(
        update={
            "state": AgentRecallDeliveryState.PENDING,
            "state_revision": record.state_revision + 1,
            "release": requested,
            "updated_at": max(now, requested.released_at),
        }
    )


def _agent_recall_delivery_release(
    claim: AgentRecallDeliveryClaim,
    *,
    release_id: str,
    reason: str,
    released_at: datetime,
) -> AgentRecallDeliveryRelease:
    claim = copy_agent_recall_delivery_claim(claim)
    return AgentRecallDeliveryRelease(
        release_id=release_id,
        delivery_id=claim.delivery_id,
        processing_result_sha256=claim.processing_result_sha256,
        claim_id=claim.claim_id,
        worker_id=claim.worker_id,
        attempt=claim.attempt,
        claim_state_revision=claim.state_revision,
        reason=reason,
        released_at=released_at,
    )


def _acknowledge_agent_recall_delivery_record(
    record: AgentRecallDeliveryRecord,
    claim: AgentRecallDeliveryClaim,
    *,
    acknowledgement_id: str,
    evidence_kind: AgentRecallDeliveryEvidenceKind,
    evidence_ref: str,
    acknowledged_at: datetime,
    now: datetime,
) -> AgentRecallDeliveryRecord:
    record = copy_agent_recall_delivery_record(record)
    claim = copy_agent_recall_delivery_claim(claim)
    acknowledgement_id = _bounded_identity(acknowledgement_id, "acknowledgement_id")
    if not isinstance(evidence_kind, AgentRecallDeliveryEvidenceKind):
        raise TypeError("evidence_kind must be an AgentRecallDeliveryEvidenceKind.")
    evidence_ref = _bounded_identity(evidence_ref, "evidence_ref")
    acknowledged_at = _utc(acknowledged_at, "acknowledged_at")
    now = _utc(now, "now")
    requested = AgentRecallDeliveryAcknowledgement(
        acknowledgement_id=acknowledgement_id,
        delivery_id=claim.delivery_id,
        processing_result_sha256=claim.processing_result_sha256,
        claim_id=claim.claim_id,
        worker_id=claim.worker_id,
        attempt=claim.attempt,
        evidence_kind=evidence_kind,
        evidence_ref=evidence_ref,
        acknowledged_at=acknowledged_at,
    )
    if record.state is AgentRecallDeliveryState.ACKNOWLEDGED:
        if record.acknowledgement != requested:
            raise AgentRecallDeliveryConflict("acknowledgement_reused")
        return record
    _require_current_delivery_claim(record, claim, now=now)
    if acknowledged_at < claim.claimed_at:
        raise AgentRecallDeliveryConflict("acknowledgement_predates_claim")
    if acknowledged_at >= claim.lease_expires_at:
        raise AgentRecallDeliveryConflict("acknowledgement_outside_claim_lease")
    if acknowledged_at > now:
        raise AgentRecallDeliveryConflict("acknowledgement_from_future")
    state_revision = record.state_revision + 1
    return record.model_copy(
        update={
            "state": AgentRecallDeliveryState.ACKNOWLEDGED,
            "state_revision": state_revision,
            "acknowledgement": requested,
            "updated_at": max(now, acknowledged_at),
        }
    )


def agent_work_context_publication_request_sha256(
    context: AgentWorkContext,
    expected_revision: int | None,
) -> str:
    context = copy_agent_work_context(context)
    if expected_revision is not None:
        _positive_revision(expected_revision, "expected_revision")
    return sha256(
        canonical_durable_json_bytes(
            {
                "context": context.model_dump(mode="json"),
                "expected_revision": expected_revision,
            },
            "agent work context publication request",
        )
    ).hexdigest()


def validate_agent_work_context_publication(
    context: AgentWorkContext,
    expected_revision: int | None,
    current: AgentWorkContext | None,
) -> None:
    if expected_revision is not None:
        _positive_revision(expected_revision, "expected_revision")
    expected_context_revision = 1 if expected_revision is None else expected_revision + 1
    if context.revision != expected_context_revision:
        raise AgentWorkContextConflict("invalid_successor_revision")
    if current is None:
        if expected_revision is not None:
            raise AgentWorkContextConflict("unknown_task")
        return
    if expected_revision is None or current.revision != expected_revision:
        raise AgentWorkContextConflict("stale_context_revision")
    if current.task_id != context.task_id:
        raise AgentWorkContextConflict("task_mismatch")


def validate_agent_recall_checkpoint_work_context(
    checkpoint: AgentRecallCheckpoint,
    referenced: AgentWorkContext | None,
    current: AgentWorkContext | None,
) -> None:
    if referenced is None:
        raise AgentWorkContextConflict("unknown_work_context")
    if (
        referenced.task_id != checkpoint.task_id
        or referenced.revision != checkpoint.work_context_revision
    ):
        raise AgentWorkContextConflict("work_context_identity_mismatch")
    if referenced.content_sha256 != checkpoint.work_context_sha256:
        raise AgentWorkContextConflict("work_context_hash_mismatch")
    if (
        current is None
        or current.task_id != checkpoint.task_id
        or current.revision != checkpoint.work_context_revision
        or current.content_sha256 != checkpoint.work_context_sha256
    ):
        raise AgentWorkContextConflict("stale_work_context_revision")


def validate_agent_recall_checkpoint_advance(
    checkpoint: AgentRecallCheckpoint,
    expected_revision: int | None,
    current: AgentRecallCheckpoint | None,
) -> None:
    if expected_revision is not None:
        _positive_revision(expected_revision, "expected_revision")
    expected_checkpoint_revision = 1 if expected_revision is None else expected_revision + 1
    if checkpoint.revision != expected_checkpoint_revision:
        raise AgentWorkContextConflict("invalid_checkpoint_revision")
    if current is None:
        if expected_revision is not None:
            raise AgentWorkContextConflict("unknown_checkpoint")
        if checkpoint.processing_mode is not AgentRecallCheckpointMode.FULL_INDEX:
            raise AgentWorkContextConflict("initial_checkpoint_requires_full_index")
        return
    if expected_revision is None or current.revision != expected_revision:
        raise AgentWorkContextConflict("stale_checkpoint_revision")
    if current.key() != checkpoint.key():
        raise AgentWorkContextConflict("checkpoint_key_mismatch")
    if checkpoint.work_context_revision < current.work_context_revision:
        raise AgentWorkContextConflict("work_context_revision_regression")
    if (
        checkpoint.work_context_revision == current.work_context_revision
        and checkpoint.work_context_sha256 != current.work_context_sha256
    ):
        raise AgentWorkContextConflict("work_context_revision_identity_mismatch")
    context_changed = (
        current.work_context_revision != checkpoint.work_context_revision
        or current.work_context_sha256 != checkpoint.work_context_sha256
    )
    if not context_changed:
        if checkpoint.knowledge_sequence < current.knowledge_sequence:
            raise AgentWorkContextConflict("knowledge_sequence_regression")
        if checkpoint.index_readiness_sequence < current.index_readiness_sequence:
            raise AgentWorkContextConflict("index_sequence_regression")
    if checkpoint.knowledge_high_water_sequence < current.knowledge_high_water_sequence:
        raise AgentWorkContextConflict("knowledge_high_water_regression")
    if checkpoint.index_readiness_high_water_sequence < current.index_readiness_high_water_sequence:
        raise AgentWorkContextConflict("index_high_water_regression")
    if context_changed:
        if checkpoint.processing_mode is not AgentRecallCheckpointMode.FULL_INDEX:
            raise AgentWorkContextConflict("changed_context_requires_full_index")
        return
    if checkpoint.processing_mode is AgentRecallCheckpointMode.DELTA and (
        checkpoint.knowledge_sequence == current.knowledge_sequence
        and checkpoint.index_readiness_sequence == current.index_readiness_sequence
    ):
        raise AgentWorkContextConflict("delta_checkpoint_has_no_progress")


class AgentWorkContextStore(ABC):
    """Narrow durable store for work contexts and freshness checkpoints."""

    @abstractmethod
    async def publish_work_context(
        self,
        context: AgentWorkContext,
        *,
        expected_revision: int | None,
    ) -> AgentWorkContextPublicationReceipt:
        """Create, append, or record an idempotent semantic no-change."""

    @abstractmethod
    async def load_work_context(
        self,
        task_id: str,
        *,
        revision: int | None = None,
    ) -> AgentWorkContext | None:
        """Load the current or one exact immutable task-context revision."""

    @abstractmethod
    async def load_work_context_publication(
        self,
        operation_id: str,
    ) -> AgentWorkContextPublicationReceipt | None:
        """Load immutable idempotency evidence for a context publication."""

    @abstractmethod
    async def advance_recall_checkpoint(
        self,
        checkpoint: AgentRecallCheckpoint,
        *,
        expected_revision: int | None,
    ) -> AgentRecallCheckpoint:
        """Advance one exact processing frontier through compare-and-swap."""

    @abstractmethod
    async def load_recall_checkpoint(
        self,
        key: AgentRecallCheckpointKey,
        *,
        revision: int | None = None,
    ) -> AgentRecallCheckpoint | None:
        """Load the current or one exact immutable checkpoint revision."""

    @abstractmethod
    async def stage_recall_delivery(
        self,
        delivery: AgentRecallDelivery,
    ) -> AgentRecallDeliveryRecord:
        """Atomically stage exact materialized recall and advance its checkpoint."""

    @abstractmethod
    async def load_recall_delivery(
        self,
        delivery_id: str,
    ) -> AgentRecallDeliveryRecord | None:
        """Load one exact staged delivery and its current handoff state."""

    @abstractmethod
    async def claim_recall_delivery(
        self,
        key: AgentRecallCheckpointKey,
        *,
        claim_id: str,
        worker_id: str,
        lease_seconds: float,
    ) -> AgentRecallDeliveryRecord | None:
        """Claim the oldest unacknowledged delivery for one exact checkpoint key."""

    @abstractmethod
    async def renew_recall_delivery(
        self,
        claim: AgentRecallDeliveryClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallDeliveryRecord:
        """Renew one exact live delivery claim."""

    @abstractmethod
    async def release_recall_delivery(
        self,
        claim: AgentRecallDeliveryClaim,
        *,
        release_id: str,
        reason: str,
        released_at: datetime,
    ) -> AgentRecallDeliveryRecord:
        """Release one exact delivery claim for deterministic retry."""

    @abstractmethod
    async def acknowledge_recall_delivery(
        self,
        claim: AgentRecallDeliveryClaim,
        *,
        acknowledgement_id: str,
        evidence_kind: AgentRecallDeliveryEvidenceKind,
        evidence_ref: str,
        acknowledged_at: datetime,
    ) -> AgentRecallDeliveryRecord:
        """Acknowledge durable downstream acceptance without claiming exposure."""

    async def close(self) -> None:
        """Release store-owned resources, if any."""

        return None


class InMemoryAgentWorkContextStore(AgentWorkContextStore):
    """Copy-safe reference implementation of the durable contract."""

    def __init__(self, *, clock=None) -> None:
        self._clock = utc_clock(clock)
        self._lock = asyncio.Lock()
        self._contexts: dict[str, dict[int, AgentWorkContext]] = {}
        self._context_heads: dict[str, int] = {}
        self._context_publications: dict[str, AgentWorkContextPublicationReceipt] = {}
        self._checkpoint_revisions: dict[
            tuple[str, str, str, str], dict[int, AgentRecallCheckpoint]
        ] = {}
        self._checkpoint_operations: dict[str, tuple[tuple[str, str, str, str], int]] = {}
        self._checkpoint_heads: dict[tuple[str, str, str, str], int] = {}
        self._recall_deliveries: dict[str, AgentRecallDeliveryRecord] = {}
        self._delivery_by_checkpoint: dict[tuple[tuple[str, str, str, str], int], str] = {}
        self._delivery_by_operation: dict[str, str] = {}
        self._delivery_queues: dict[tuple[str, str, str, str], deque[str]] = {}
        self._delivery_claims: dict[str, tuple[str, str, str, int]] = {}
        self._delivery_releases: dict[str, AgentRecallDeliveryRelease] = {}
        self._delivery_acknowledgements: dict[str, str] = {}

    async def publish_work_context(
        self,
        context: AgentWorkContext,
        *,
        expected_revision: int | None,
    ) -> AgentWorkContextPublicationReceipt:
        context = copy_agent_work_context(context)
        request_sha256 = agent_work_context_publication_request_sha256(
            context,
            expected_revision,
        )
        async with self._lock:
            replay = self._context_publications.get(context.operation_id)
            if replay is not None:
                if replay.request_sha256 != request_sha256:
                    raise AgentWorkContextConflict("publication_operation_reused")
                return copy_agent_work_context_publication_receipt(replay)
            revisions = self._contexts.get(context.task_id)
            current_revision = self._context_heads.get(context.task_id)
            current = (
                None
                if revisions is None or current_revision is None
                else revisions[current_revision]
            )
            validate_agent_work_context_publication(context, expected_revision, current)
            changed = current is None or current.content_sha256 != context.content_sha256
            result = context if changed else current
            assert result is not None
            receipt = AgentWorkContextPublicationReceipt(
                operation_id=context.operation_id,
                request_sha256=request_sha256,
                expected_revision=expected_revision,
                requested_content_sha256=context.content_sha256,
                changed=changed,
                context=result,
                committed_at=_utc(self._clock(), "clock result"),
            )
            if changed:
                self._contexts.setdefault(context.task_id, {})[context.revision] = context
                self._context_heads[context.task_id] = context.revision
            self._context_publications[context.operation_id] = receipt
            return copy_agent_work_context_publication_receipt(receipt)

    async def load_work_context(
        self,
        task_id: str,
        *,
        revision: int | None = None,
    ) -> AgentWorkContext | None:
        task_id = _bounded_identity(task_id, "task_id")
        if revision is not None:
            _positive_revision(revision, "revision")
        async with self._lock:
            revisions = self._contexts.get(task_id)
            if not revisions:
                return None
            resolved_revision = self._context_heads[task_id] if revision is None else revision
            context = revisions.get(resolved_revision)
            return None if context is None else copy_agent_work_context(context)

    async def load_work_context_publication(
        self,
        operation_id: str,
    ) -> AgentWorkContextPublicationReceipt | None:
        operation_id = _bounded_identity(operation_id, "operation_id")
        async with self._lock:
            receipt = self._context_publications.get(operation_id)
            return None if receipt is None else copy_agent_work_context_publication_receipt(receipt)

    async def advance_recall_checkpoint(
        self,
        checkpoint: AgentRecallCheckpoint,
        *,
        expected_revision: int | None,
    ) -> AgentRecallCheckpoint:
        if expected_revision is not None:
            _positive_revision(expected_revision, "expected_revision")
        checkpoint = copy_agent_recall_checkpoint(checkpoint)
        key = checkpoint.key().sort_key()
        async with self._lock:
            replay = self._checkpoint_operations.get(checkpoint.operation_id)
            if replay is not None:
                replay_key, replay_revision = replay
                stored = self._checkpoint_revisions[replay_key][replay_revision]
                if (
                    replay_key != key
                    or stored != checkpoint
                    or expected_revision != (None if replay_revision == 1 else replay_revision - 1)
                ):
                    raise AgentWorkContextConflict("checkpoint_operation_reused")
                return copy_agent_recall_checkpoint(stored)
            self._advance_recall_checkpoint_unlocked(
                checkpoint,
                expected_revision=expected_revision,
            )
            return copy_agent_recall_checkpoint(checkpoint)

    def _advance_recall_checkpoint_unlocked(
        self,
        checkpoint: AgentRecallCheckpoint,
        *,
        expected_revision: int | None,
    ) -> None:
        key = checkpoint.key().sort_key()
        task_contexts = self._contexts.get(checkpoint.task_id)
        work_context = (
            None if task_contexts is None else task_contexts.get(checkpoint.work_context_revision)
        )
        current_work_context_revision = self._context_heads.get(checkpoint.task_id)
        current_work_context = (
            None
            if task_contexts is None or current_work_context_revision is None
            else task_contexts[current_work_context_revision]
        )
        validate_agent_recall_checkpoint_work_context(
            checkpoint,
            work_context,
            current_work_context,
        )
        revisions = self._checkpoint_revisions.get(key)
        current_revision = self._checkpoint_heads.get(key)
        current = (
            None if revisions is None or current_revision is None else revisions[current_revision]
        )
        validate_agent_recall_checkpoint_advance(
            checkpoint,
            expected_revision,
            current,
        )
        self._checkpoint_revisions.setdefault(key, {})[checkpoint.revision] = checkpoint
        self._checkpoint_operations[checkpoint.operation_id] = (key, checkpoint.revision)
        self._checkpoint_heads[key] = checkpoint.revision

    async def stage_recall_delivery(
        self,
        delivery: AgentRecallDelivery,
    ) -> AgentRecallDeliveryRecord:
        delivery = copy_agent_recall_delivery(delivery)
        key = delivery.key().sort_key()
        async with self._lock:
            existing = self._recall_deliveries.get(delivery.delivery_id)
            if existing is not None:
                if existing.delivery != delivery:
                    raise AgentRecallDeliveryConflict("delivery_id_reused")
                return copy_agent_recall_delivery_record(existing)
            occupied = self._delivery_by_checkpoint.get((key, delivery.checkpoint.revision))
            if occupied is not None:
                raise AgentRecallDeliveryConflict("checkpoint_delivery_exists")
            operation_delivery = self._delivery_by_operation.get(delivery.operation_id)
            if operation_delivery is not None:
                raise AgentRecallDeliveryConflict("delivery_operation_reused")
            if delivery.operation_id in self._checkpoint_operations:
                raise AgentRecallDeliveryConflict("checkpoint_committed_without_delivery")
            if delivery.staged_at > _utc(self._clock(), "clock result"):
                raise AgentRecallDeliveryConflict("delivery_staged_in_future")
            self._advance_recall_checkpoint_unlocked(
                delivery.checkpoint,
                expected_revision=delivery.expected_checkpoint_revision,
            )
            record = AgentRecallDeliveryRecord(
                delivery=delivery,
                updated_at=delivery.staged_at,
            )
            self._recall_deliveries[delivery.delivery_id] = record
            self._delivery_by_checkpoint[(key, delivery.checkpoint.revision)] = delivery.delivery_id
            self._delivery_by_operation[delivery.operation_id] = delivery.delivery_id
            self._delivery_queues.setdefault(key, deque()).append(delivery.delivery_id)
            return copy_agent_recall_delivery_record(record)

    async def load_recall_delivery(
        self,
        delivery_id: str,
    ) -> AgentRecallDeliveryRecord | None:
        delivery_id = _bounded_identity(delivery_id, "delivery_id")
        async with self._lock:
            record = self._recall_deliveries.get(delivery_id)
            return None if record is None else copy_agent_recall_delivery_record(record)

    async def claim_recall_delivery(
        self,
        key: AgentRecallCheckpointKey,
        *,
        claim_id: str,
        worker_id: str,
        lease_seconds: float,
    ) -> AgentRecallDeliveryRecord | None:
        key = copy_agent_recall_checkpoint_key(key)
        claim_id = _bounded_identity(claim_id, "claim_id")
        worker_id = _bounded_identity(worker_id, "worker_id")
        request_sha256 = agent_recall_delivery_claim_request_sha256(
            key,
            claim_id=claim_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )
        async with self._lock:
            replayed_claim = self._delivery_claims.get(claim_id)
            if replayed_claim is not None:
                (
                    replayed_request_sha256,
                    replayed_delivery_id,
                    replayed_worker_id,
                    replayed_attempt,
                ) = replayed_claim
                if replayed_request_sha256 != request_sha256:
                    raise AgentRecallDeliveryConflict("claim_id_reused")
                replayed_record = self._recall_deliveries[replayed_delivery_id]
                _require_replayable_delivery_claim_attempt(
                    replayed_record,
                    claim_id=claim_id,
                    worker_id=replayed_worker_id,
                    attempt=replayed_attempt,
                    now=max(_utc(self._clock(), "clock result"), replayed_record.updated_at),
                )
                return copy_agent_recall_delivery_record(replayed_record)
            queue_key = key.sort_key()
            self._trim_delivery_queue_unlocked(queue_key)
            queue = self._delivery_queues.get(queue_key)
            if not queue:
                return None
            current = self._recall_deliveries[queue[0]]
            now = max(_utc(self._clock(), "clock result"), current.updated_at)
            if (
                current.state is AgentRecallDeliveryState.CLAIMED
                and current.claim is not None
                and current.claim.lease_expires_at > now
                and (current.claim.claim_id != claim_id or current.claim.worker_id != worker_id)
            ):
                return None
            claimed = _claim_agent_recall_delivery_record(
                current,
                claim_id=claim_id,
                worker_id=worker_id,
                lease_seconds=lease_seconds,
                now=now,
            )
            self._recall_deliveries[current.delivery.delivery_id] = claimed
            assert claimed.claim is not None
            self._delivery_claims[claim_id] = (
                request_sha256,
                current.delivery.delivery_id,
                claimed.claim.worker_id,
                claimed.claim.attempt,
            )
            return copy_agent_recall_delivery_record(claimed)

    async def renew_recall_delivery(
        self,
        claim: AgentRecallDeliveryClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallDeliveryRecord:
        claim = copy_agent_recall_delivery_claim(claim)
        _validate_delivery_lease_seconds(lease_seconds)
        async with self._lock:
            current = self._recall_deliveries.get(claim.delivery_id)
            if current is None:
                raise AgentRecallDeliveryConflict("unknown_delivery")
            now = max(_utc(self._clock(), "clock result"), current.updated_at)
            renewed = _renew_agent_recall_delivery_record(
                current,
                claim,
                lease_seconds=lease_seconds,
                now=now,
            )
            self._recall_deliveries[claim.delivery_id] = renewed
            return copy_agent_recall_delivery_record(renewed)

    async def release_recall_delivery(
        self,
        claim: AgentRecallDeliveryClaim,
        *,
        release_id: str,
        reason: str,
        released_at: datetime,
    ) -> AgentRecallDeliveryRecord:
        claim = copy_agent_recall_delivery_claim(claim)
        release_id = _bounded_identity(release_id, "release_id")
        requested = _agent_recall_delivery_release(
            claim,
            release_id=release_id,
            reason=reason,
            released_at=released_at,
        )
        async with self._lock:
            current = self._recall_deliveries.get(claim.delivery_id)
            if current is None:
                raise AgentRecallDeliveryConflict("unknown_delivery")
            replayed_release = self._delivery_releases.get(release_id)
            if replayed_release is not None:
                if replayed_release != requested:
                    raise AgentRecallDeliveryConflict("release_id_reused")
                if current.release != replayed_release:
                    raise AgentRecallDeliveryConflict("release_replay_superseded")
                return copy_agent_recall_delivery_record(current)
            now = max(_utc(self._clock(), "clock result"), current.updated_at)
            released = _release_agent_recall_delivery_record(
                current,
                claim,
                release_id=release_id,
                reason=reason,
                released_at=released_at,
                now=now,
            )
            self._recall_deliveries[claim.delivery_id] = released
            assert released.release is not None
            self._delivery_releases[release_id] = released.release
            return copy_agent_recall_delivery_record(released)

    async def acknowledge_recall_delivery(
        self,
        claim: AgentRecallDeliveryClaim,
        *,
        acknowledgement_id: str,
        evidence_kind: AgentRecallDeliveryEvidenceKind,
        evidence_ref: str,
        acknowledged_at: datetime,
    ) -> AgentRecallDeliveryRecord:
        claim = copy_agent_recall_delivery_claim(claim)
        acknowledgement_id = _bounded_identity(acknowledgement_id, "acknowledgement_id")
        async with self._lock:
            current = self._recall_deliveries.get(claim.delivery_id)
            if current is None:
                raise AgentRecallDeliveryConflict("unknown_delivery")
            occupied_delivery_id = self._delivery_acknowledgements.get(acknowledgement_id)
            if occupied_delivery_id is not None and occupied_delivery_id != claim.delivery_id:
                raise AgentRecallDeliveryConflict("acknowledgement_reused")
            now = max(_utc(self._clock(), "clock result"), current.updated_at)
            acknowledged = _acknowledge_agent_recall_delivery_record(
                current,
                claim,
                acknowledgement_id=acknowledgement_id,
                evidence_kind=evidence_kind,
                evidence_ref=evidence_ref,
                acknowledged_at=acknowledged_at,
                now=now,
            )
            self._recall_deliveries[claim.delivery_id] = acknowledged
            self._delivery_acknowledgements[acknowledgement_id] = claim.delivery_id
            self._trim_delivery_queue_unlocked(acknowledged.delivery.key().sort_key())
            return copy_agent_recall_delivery_record(acknowledged)

    def _trim_delivery_queue_unlocked(self, key: tuple[str, str, str, str]) -> None:
        queue = self._delivery_queues.get(key)
        if queue is None:
            return
        while queue:
            record = self._recall_deliveries[queue[0]]
            if record.state is not AgentRecallDeliveryState.ACKNOWLEDGED:
                return
            queue.popleft()
        self._delivery_queues.pop(key, None)

    async def load_recall_checkpoint(
        self,
        key: AgentRecallCheckpointKey,
        *,
        revision: int | None = None,
    ) -> AgentRecallCheckpoint | None:
        key = copy_agent_recall_checkpoint_key(key)
        if revision is not None:
            _positive_revision(revision, "revision")
        async with self._lock:
            revisions = self._checkpoint_revisions.get(key.sort_key())
            if not revisions:
                return None
            resolved_revision = (
                self._checkpoint_heads[key.sort_key()] if revision is None else revision
            )
            checkpoint = revisions.get(resolved_revision)
            return None if checkpoint is None else copy_agent_recall_checkpoint(checkpoint)


__all__ = [
    "AGENT_RECALL_CHECKPOINT_SCHEMA_VERSION",
    "AGENT_RECALL_DELIVERY_ACKNOWLEDGEMENT_SCHEMA_VERSION",
    "AGENT_RECALL_DELIVERY_CLAIM_SCHEMA_VERSION",
    "AGENT_RECALL_DELIVERY_RECORD_SCHEMA_VERSION",
    "AGENT_RECALL_DELIVERY_RELEASE_SCHEMA_VERSION",
    "AGENT_RECALL_DELIVERY_SCHEMA_VERSION",
    "AGENT_WORK_CONTEXT_PUBLICATION_SCHEMA_VERSION",
    "AGENT_WORK_CONTEXT_SCHEMA_VERSION",
    "MAX_AGENT_RECALL_DELIVERY_BYTES",
    "MAX_AGENT_RECALL_DELIVERY_LEASE_SECONDS",
    "MAX_AGENT_WORK_CONTEXT_BYTES",
    "MAX_AGENT_WORK_CONTEXT_GOAL_BYTES",
    "MAX_AGENT_WORK_CONTEXT_ID_BYTES",
    "MAX_AGENT_WORK_CONTEXT_REVISION",
    "MAX_AGENT_WORK_CONTEXT_VALUES",
    "MAX_AGENT_WORK_CONTEXT_VALUE_BYTES",
    "AgentRecallCheckpoint",
    "AgentRecallCheckpointKey",
    "AgentRecallCheckpointMode",
    "AgentRecallDelivery",
    "AgentRecallDeliveryAcknowledgement",
    "AgentRecallDeliveryClaim",
    "AgentRecallDeliveryConflict",
    "AgentRecallDeliveryEvidenceKind",
    "AgentRecallDeliveryRecord",
    "AgentRecallDeliveryRelease",
    "AgentRecallDeliveryState",
    "AgentWorkContext",
    "AgentWorkContextConflict",
    "AgentWorkContextPublicationReceipt",
    "AgentWorkContextStore",
    "InMemoryAgentWorkContextStore",
]
