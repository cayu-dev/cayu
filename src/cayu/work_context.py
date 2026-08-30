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
AGENT_RECALL_CHECKPOINT_SCHEMA_VERSION = "cayu.agent_recall_checkpoint.v2"
DEFAULT_AGENT_RECALL_CHECKPOINT_STREAM_ID = "cayu.agent_recall.inline.v1"
AGENT_RECALL_DELIVERY_SCHEMA_VERSION = "cayu.agent_recall_delivery.v1"
AGENT_RECALL_DELIVERY_RECORD_SCHEMA_VERSION = "cayu.agent_recall_delivery_record.v1"
AGENT_RECALL_DELIVERY_CLAIM_SCHEMA_VERSION = "cayu.agent_recall_delivery_claim.v1"
AGENT_RECALL_DELIVERY_RELEASE_SCHEMA_VERSION = "cayu.agent_recall_delivery_release.v1"
AGENT_RECALL_DELIVERY_ACKNOWLEDGEMENT_SCHEMA_VERSION = (
    "cayu.agent_recall_delivery_acknowledgement.v1"
)
AGENT_RECALL_SUBSCRIPTION_SCHEMA_VERSION = "cayu.agent_recall_subscription.v1"
AGENT_RECALL_SUBSCRIPTION_PUBLICATION_SCHEMA_VERSION = (
    "cayu.agent_recall_subscription_publication_receipt.v1"
)
AGENT_RECALL_SUBSCRIPTION_RECORD_SCHEMA_VERSION = "cayu.agent_recall_subscription_record.v1"
AGENT_RECALL_SUBSCRIPTION_CLAIM_SCHEMA_VERSION = "cayu.agent_recall_subscription_claim.v1"
AGENT_RECALL_SUBSCRIPTION_RELEASE_SCHEMA_VERSION = "cayu.agent_recall_subscription_release.v1"
AGENT_RECALL_SUBSCRIPTION_EVALUATION_SCHEMA_VERSION = "cayu.agent_recall_subscription_evaluation.v1"
AGENT_RECALL_SUBSCRIPTION_WAKE_SCHEMA_VERSION = "cayu.agent_recall_subscription_wake.v1"
AGENT_RECALL_SUBSCRIPTION_WAKE_CLAIM_SCHEMA_VERSION = "cayu.agent_recall_subscription_wake_claim.v1"
AGENT_RECALL_SUBSCRIPTION_WAKE_RELEASE_SCHEMA_VERSION = (
    "cayu.agent_recall_subscription_wake_release.v1"
)
AGENT_RECALL_SUBSCRIPTION_WAKE_ACKNOWLEDGEMENT_SCHEMA_VERSION = (
    "cayu.agent_recall_subscription_wake_acknowledgement.v1"
)

MAX_AGENT_WORK_CONTEXT_REVISION = 2_147_483_647
MAX_AGENT_WORK_CONTEXT_ID_BYTES = 512
MAX_AGENT_WORK_CONTEXT_GOAL_BYTES = 32_000
MAX_AGENT_WORK_CONTEXT_VALUE_BYTES = 4_096
MAX_AGENT_WORK_CONTEXT_VALUES = 128
MAX_AGENT_WORK_CONTEXT_BYTES = 256_000
MAX_AGENT_RECALL_DELIVERY_BYTES = 2_000_000
MAX_AGENT_RECALL_DELIVERY_LEASE_SECONDS = 86_400.0
MAX_AGENT_RECALL_SUBSCRIPTION_BYTES = 512_000
MAX_AGENT_RECALL_SUBSCRIPTION_INTERVAL_SECONDS = 604_800.0
MAX_AGENT_RECALL_SUBSCRIPTION_QUERY_BYTES = 8_192
MAX_AGENT_RECALL_SUBSCRIPTION_PRIORITY = 1_000

_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_CONTEXT_COLLECTION_FIELDS = (
    "scope_ids",
    "entity_ids",
    "artifact_ids",
    "repository_paths",
    "code_symbols",
    "planned_action_ids",
)
_AGENT_RECALL_FACET_ASPECT_PREFIX = "cayu.agent_recall_facet.v1"
_AGENT_RECALL_SUBSCRIPTION_CHECKPOINT_STREAM_PREFIX = "cayu.agent_recall.subscription_checkpoint.v1"


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


def agent_recall_facet_aspect(field_name: str, value: str) -> str:
    """Encode one typed work-context facet as an exact knowledge aspect."""

    if field_name not in _CONTEXT_COLLECTION_FIELDS:
        raise ValueError(f"Unsupported agent recall facet field {field_name!r}.")
    value = _bounded_value(value, field_name)
    digest = sha256(
        canonical_durable_json_bytes(
            {"field": field_name, "value": value},
            "agent recall facet",
        )
    ).hexdigest()
    return f"{_AGENT_RECALL_FACET_ASPECT_PREFIX}:{field_name}:{digest}"


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
    checkpoint_stream_id: str = DEFAULT_AGENT_RECALL_CHECKPOINT_STREAM_ID

    @field_validator("agent_id", "task_id", "knowledge_namespace", "checkpoint_stream_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("access_policy_sha256")
    @classmethod
    def validate_access_policy_sha256(cls, value: str) -> str:
        return _sha256_hex(value, "access_policy_sha256")

    def sort_key(self) -> tuple[str, str, str, str, str]:
        return (
            self.agent_id,
            self.task_id,
            self.knowledge_namespace,
            self.access_policy_sha256,
            self.checkpoint_stream_id,
        )

    def authority_sort_key(self) -> tuple[str, str, str, str]:
        """Return the selector shared by all checkpoint streams for this authority."""

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

    schema_version: Literal["cayu.agent_recall_checkpoint.v2"] = (
        AGENT_RECALL_CHECKPOINT_SCHEMA_VERSION
    )
    agent_id: str
    task_id: str
    knowledge_namespace: str
    access_policy_sha256: str
    checkpoint_stream_id: str = DEFAULT_AGENT_RECALL_CHECKPOINT_STREAM_ID
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

    @field_validator(
        "agent_id",
        "task_id",
        "knowledge_namespace",
        "checkpoint_stream_id",
    )
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
            checkpoint_stream_id=self.checkpoint_stream_id,
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


class AgentRecallSubscriptionStatus(StrEnum):
    """Application-owned subscription lifecycle."""

    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class AgentRecallSubscriptionRunState(StrEnum):
    """Whether one active subscription is available or runner-claimed."""

    DUE = "due"
    CLAIMED = "claimed"


class AgentRecallSubscriptionEvaluationOutcome(StrEnum):
    """Durable result of one bounded subscription evaluation."""

    NO_WORK = "no_work"
    SILENT = "silent"
    WAKE = "wake"


class AgentRecallSubscriptionWakeState(StrEnum):
    """Independent task-scheduler handoff state for one admitted wake."""

    PENDING = "pending"
    CLAIMED = "claimed"
    ACKNOWLEDGED = "acknowledged"


class AgentRecallSubscription(_WorkContextModel):
    """One immutable revision of an idle-task recall subscription."""

    schema_version: Literal["cayu.agent_recall_subscription.v1"] = (
        AGENT_RECALL_SUBSCRIPTION_SCHEMA_VERSION
    )
    subscription_id: str
    agent_id: str
    task_id: str
    knowledge_namespace: str
    access_policy_sha256: str
    work_context_revision: int
    work_context_sha256: str
    query: str | None = None
    scope_ids: tuple[str, ...] = ()
    entity_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    repository_paths: tuple[str, ...] = ()
    code_symbols: tuple[str, ...] = ()
    planned_action_ids: tuple[str, ...] = ()
    admission_policy: Mapping[str, Any]
    admission_policy_sha256: str
    priority: int = 0
    minimum_interval_seconds: float
    expires_at: datetime
    status: AgentRecallSubscriptionStatus
    revision: int
    operation_id: str
    published_by: str
    published_at: datetime

    @field_validator(
        "subscription_id",
        "agent_id",
        "task_id",
        "knowledge_namespace",
        "operation_id",
        "published_by",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("access_policy_sha256", "work_context_sha256", "admission_policy_sha256")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _sha256_hex(value, info.field_name)

    @field_validator("work_context_revision", "revision")
    @classmethod
    def validate_revision(cls, value: int, info) -> int:
        return _positive_revision(value, info.field_name)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_durable_clean_nonblank(value, "query")
        if len(value.encode("utf-8")) > MAX_AGENT_RECALL_SUBSCRIPTION_QUERY_BYTES:
            raise ValueError(
                f"`query` must be at most {MAX_AGENT_RECALL_SUBSCRIPTION_QUERY_BYTES} UTF-8 bytes."
            )
        return value

    @field_validator(*_CONTEXT_COLLECTION_FIELDS, mode="before")
    @classmethod
    def validate_filter_collections(cls, value: object, info) -> tuple[str, ...]:
        return _ordered_unique_values(value, info.field_name)

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, value: int) -> int:
        if type(value) is not int or not 0 <= value <= MAX_AGENT_RECALL_SUBSCRIPTION_PRIORITY:
            raise ValueError(
                f"`priority` must be between 0 and {MAX_AGENT_RECALL_SUBSCRIPTION_PRIORITY}."
            )
        return value

    @field_validator("admission_policy", mode="before")
    @classmethod
    def copy_admission_policy(cls, value: object) -> dict[str, Any]:
        from cayu.memory import AutomaticRecallPolicy

        if type(value) is AutomaticRecallPolicy:
            value = value.model_dump(mode="json")
        copied = copy_durable_json_object(value, "admission_policy")
        policy = AutomaticRecallPolicy.model_validate_json(
            canonical_durable_json_bytes(copied, "admission_policy")
        )
        return policy.model_dump(mode="json")

    @field_validator("admission_policy")
    @classmethod
    def freeze_admission_policy(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        frozen = freeze_json_value(dict(value))
        if type(frozen) is not FrozenJsonDict:  # pragma: no cover - defensive invariant
            raise AssertionError("Recall subscription policy did not freeze as an object.")
        return frozen

    @field_serializer("admission_policy")
    def serialize_admission_policy(self, value: Mapping[str, Any]) -> dict[str, Any]:
        thawed = thaw_json_value(value)
        if type(thawed) is not dict:  # pragma: no cover - defensive invariant
            raise AssertionError("Recall subscription policy did not thaw as an object.")
        return thawed

    @field_validator("minimum_interval_seconds")
    @classmethod
    def validate_minimum_interval(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("`minimum_interval_seconds` must be a number.")
        result = float(value)
        if not 0 < result <= MAX_AGENT_RECALL_SUBSCRIPTION_INTERVAL_SECONDS:
            raise ValueError(
                "`minimum_interval_seconds` must be greater than zero and at most "
                f"{MAX_AGENT_RECALL_SUBSCRIPTION_INTERVAL_SECONDS}."
            )
        return result

    @field_validator("expires_at", "published_at")
    @classmethod
    def validate_datetime(cls, value: datetime, info) -> datetime:
        return _utc(value, info.field_name)

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, value: object) -> AgentRecallSubscriptionStatus:
        if isinstance(value, AgentRecallSubscriptionStatus):
            return value
        if type(value) is str:
            return AgentRecallSubscriptionStatus(value)
        raise ValueError("status must be an AgentRecallSubscriptionStatus.")

    @model_validator(mode="after")
    def validate_authority(self) -> AgentRecallSubscription:
        from cayu.memory import AutomaticRecallPolicy
        from cayu.recall import RECALL_MAX_QUERY_BYTES
        from cayu.storage.memory import KnowledgeQuery, KnowledgeSearchMode

        policy = AutomaticRecallPolicy.model_validate_json(
            canonical_durable_json_bytes(self.admission_policy, "admission_policy")
        )
        if policy.fingerprint() != self.admission_policy_sha256:
            raise ValueError("Admission-policy fingerprint does not match its exact payload.")
        if self.expires_at <= self.published_at:
            raise ValueError("`expires_at` must follow `published_at`.")
        has_exact_facets = any(
            getattr(self, field_name) for field_name in _CONTEXT_COLLECTION_FIELDS
        )
        if self.query is None and not has_exact_facets:
            raise ValueError("A subscription requires a query or at least one exact facet.")
        if self.query is not None and not has_exact_facets:
            try:
                KnowledgeQuery(text=self.query, mode=KnowledgeSearchMode.KEYWORD)
            except ValueError as exc:
                raise ValueError(
                    "A subscription query without exact facets must contain at least one "
                    "lexical search token."
                ) from exc
        if len(self.retrieval_query().encode("utf-8")) > RECALL_MAX_QUERY_BYTES:
            raise ValueError("Subscription retrieval input exceeds the recall query byte limit.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "agent recall subscription",
                )
            )
            > MAX_AGENT_RECALL_SUBSCRIPTION_BYTES
        ):
            raise ValueError("Subscription exceeds its serialized byte limit.")
        return self

    @classmethod
    def create(
        cls,
        *,
        subscription_id: str,
        agent_id: str,
        work_context: AgentWorkContext,
        knowledge_namespace: str,
        access_policy_sha256: str,
        admission_policy: Any,
        minimum_interval_seconds: float,
        expires_at: datetime,
        revision: int,
        operation_id: str,
        published_by: str,
        published_at: datetime,
        query: str | None = None,
        scope_ids: Sequence[str] = (),
        entity_ids: Sequence[str] = (),
        artifact_ids: Sequence[str] = (),
        repository_paths: Sequence[str] = (),
        code_symbols: Sequence[str] = (),
        planned_action_ids: Sequence[str] = (),
        priority: int = 0,
        status: AgentRecallSubscriptionStatus = AgentRecallSubscriptionStatus.ACTIVE,
    ) -> AgentRecallSubscription:
        from cayu.memory import AutomaticRecallPolicy

        context = copy_agent_work_context(work_context)
        if type(admission_policy) is not AutomaticRecallPolicy:
            raise TypeError("admission_policy must be an AutomaticRecallPolicy.")
        policy = AutomaticRecallPolicy.model_validate(admission_policy.model_dump(mode="python"))
        return cls(
            subscription_id=subscription_id,
            agent_id=agent_id,
            task_id=context.task_id,
            knowledge_namespace=knowledge_namespace,
            access_policy_sha256=access_policy_sha256,
            work_context_revision=context.revision,
            work_context_sha256=context.content_sha256,
            query=query,
            scope_ids=_copy_context_values(scope_ids, "scope_ids"),
            entity_ids=_copy_context_values(entity_ids, "entity_ids"),
            artifact_ids=_copy_context_values(artifact_ids, "artifact_ids"),
            repository_paths=_copy_context_values(repository_paths, "repository_paths"),
            code_symbols=_copy_context_values(code_symbols, "code_symbols"),
            planned_action_ids=_copy_context_values(
                planned_action_ids,
                "planned_action_ids",
            ),
            admission_policy=policy.model_dump(mode="json"),
            admission_policy_sha256=policy.fingerprint(),
            priority=priority,
            minimum_interval_seconds=minimum_interval_seconds,
            expires_at=expires_at,
            status=status,
            revision=revision,
            operation_id=operation_id,
            published_by=published_by,
            published_at=published_at,
        )

    def checkpoint_key(self) -> AgentRecallCheckpointKey:
        return AgentRecallCheckpointKey(
            agent_id=self.agent_id,
            task_id=self.task_id,
            knowledge_namespace=self.knowledge_namespace,
            access_policy_sha256=self.access_policy_sha256,
            checkpoint_stream_id=self.checkpoint_stream_id(),
        )

    def checkpoint_stream_id(self) -> str:
        """Return the cursor identity for this subscription's retrieval definition."""

        digest = sha256(
            canonical_durable_json_bytes(
                {
                    "subscription_id": self.subscription_id,
                    "situation_sha256": self.situation_sha256(),
                },
                "agent recall subscription checkpoint stream",
            )
        ).hexdigest()
        return f"{_AGENT_RECALL_SUBSCRIPTION_CHECKPOINT_STREAM_PREFIX}:{digest}"

    def policy(self):
        from cayu.memory import AutomaticRecallPolicy

        return AutomaticRecallPolicy.model_validate_json(
            canonical_durable_json_bytes(self.admission_policy, "admission_policy")
        )

    def retrieval_query(self) -> str:
        return self.query or "exact knowledge facet subscription"

    def facet_aspect_groups(self) -> tuple[tuple[str, ...], ...]:
        """Return OR-within/AND-across indexed aspect groups for this subscription."""

        return tuple(
            tuple(agent_recall_facet_aspect(field_name, value) for value in values)
            for field_name in _CONTEXT_COLLECTION_FIELDS
            if (values := getattr(self, field_name))
        )

    def recall_situation(self, access_scope: Any, *, current_time: datetime):
        """Build the exact recurring processor input for this subscription."""

        from cayu.recall import RecallSituation
        from cayu.storage.memory import (
            KnowledgeAccessScope,
            copy_knowledge_access_scope,
            knowledge_access_scope_sha256,
        )

        if type(access_scope) is not KnowledgeAccessScope:
            raise TypeError("access_scope must be a KnowledgeAccessScope.")
        scope = copy_knowledge_access_scope(access_scope)
        if (
            scope.allow_all_namespaces
            or tuple(scope.allowed_namespaces) != (self.knowledge_namespace,)
            or knowledge_access_scope_sha256(scope) != self.access_policy_sha256
        ):
            raise AgentRecallSubscriptionConflict("subscription_access_scope_mismatch")
        return RecallSituation(
            query=self.retrieval_query(),
            knowledge_access_scope=scope,
            knowledge_namespace=self.knowledge_namespace,
            knowledge_aspect_groups=self.facet_aspect_groups(),
            knowledge_filter_only=self.query is None,
            current_time=_utc(current_time, "current_time"),
        )

    def situation_sha256(self) -> str:
        from cayu.recall import RecallSituation
        from cayu.recall_processing import agent_recall_situation_input_sha256

        return agent_recall_situation_input_sha256(
            RecallSituation(
                query=self.retrieval_query(),
                knowledge_namespace=self.knowledge_namespace,
                knowledge_aspect_groups=self.facet_aspect_groups(),
                knowledge_filter_only=self.query is None,
                current_time=self.published_at,
            )
        )

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "agent recall subscription",
            )
        ).hexdigest()


class AgentRecallSubscriptionPublicationReceipt(_WorkContextModel):
    """Immutable idempotency evidence for one subscription revision."""

    schema_version: Literal["cayu.agent_recall_subscription_publication_receipt.v1"] = (
        AGENT_RECALL_SUBSCRIPTION_PUBLICATION_SCHEMA_VERSION
    )
    operation_id: str
    request_sha256: str
    expected_revision: int | None
    subscription: AgentRecallSubscription
    committed_at: datetime

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return _bounded_identity(value, "operation_id")

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _sha256_hex(value, "request_sha256")

    @field_validator("expected_revision")
    @classmethod
    def validate_expected_revision(cls, value: int | None) -> int | None:
        if value is not None:
            _positive_revision(value, "expected_revision")
        return value

    @field_validator("subscription", mode="before")
    @classmethod
    def copy_subscription(cls, value: object, info) -> AgentRecallSubscription:
        if type(value) is AgentRecallSubscription:
            return copy_agent_recall_subscription(value)
        if info.mode == "json":
            return AgentRecallSubscription.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall subscription")
            )
        return AgentRecallSubscription.model_validate(value)

    @field_validator("committed_at")
    @classmethod
    def validate_committed_at(cls, value: datetime) -> datetime:
        return _utc(value, "committed_at")

    @model_validator(mode="after")
    def validate_publication(self) -> AgentRecallSubscriptionPublicationReceipt:
        expected = 1 if self.expected_revision is None else self.expected_revision + 1
        if (
            self.operation_id != self.subscription.operation_id
            or self.subscription.revision != expected
        ):
            raise ValueError("Subscription publication authority is inconsistent.")
        if self.request_sha256 != agent_recall_subscription_publication_request_sha256(
            self.subscription,
            self.expected_revision,
        ):
            raise ValueError("Subscription publication request fingerprint is inconsistent.")
        if self.committed_at < self.subscription.published_at:
            raise ValueError("Subscription publication cannot commit before it was published.")
        return self


class AgentRecallSubscriptionClaim(_WorkContextModel):
    """One store-clock lease over a due subscription revision."""

    schema_version: Literal["cayu.agent_recall_subscription_claim.v1"] = (
        AGENT_RECALL_SUBSCRIPTION_CLAIM_SCHEMA_VERSION
    )
    subscription_id: str
    subscription_revision: int
    subscription_sha256: str
    claim_id: str
    runner_id: str
    attempt: int
    state_revision: int
    claimed_at: datetime
    lease_expires_at: datetime

    @field_validator("subscription_id", "claim_id", "runner_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("subscription_revision", "attempt", "state_revision")
    @classmethod
    def validate_revision(cls, value: int, info) -> int:
        return _positive_revision(value, info.field_name)

    @field_validator("subscription_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _sha256_hex(value, "subscription_sha256")

    @field_validator("claimed_at", "lease_expires_at")
    @classmethod
    def validate_datetime(cls, value: datetime, info) -> datetime:
        return _utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_lease(self) -> AgentRecallSubscriptionClaim:
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("Subscription claim lease must expire after it is claimed.")
        return self


class AgentRecallSubscriptionRelease(_WorkContextModel):
    """Immutable evidence returning an evaluation claim for retry."""

    schema_version: Literal["cayu.agent_recall_subscription_release.v1"] = (
        AGENT_RECALL_SUBSCRIPTION_RELEASE_SCHEMA_VERSION
    )
    release_id: str
    subscription_id: str
    subscription_revision: int
    subscription_sha256: str
    claim_id: str
    runner_id: str
    attempt: int
    claim_state_revision: int
    reason: str
    released_at: datetime

    @field_validator("release_id", "subscription_id", "claim_id", "runner_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("subscription_revision", "attempt", "claim_state_revision")
    @classmethod
    def validate_revision(cls, value: int, info) -> int:
        return _positive_revision(value, info.field_name)

    @field_validator("subscription_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _sha256_hex(value, "subscription_sha256")

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
                "agent recall subscription release",
            )
        ).hexdigest()


class AgentRecallSubscriptionRecord(_WorkContextModel):
    """Current scheduling state for one immutable subscription head."""

    schema_version: Literal["cayu.agent_recall_subscription_record.v1"] = (
        AGENT_RECALL_SUBSCRIPTION_RECORD_SCHEMA_VERSION
    )
    subscription: AgentRecallSubscription
    run_state: AgentRecallSubscriptionRunState = AgentRecallSubscriptionRunState.DUE
    state_revision: int = 0
    attempt: int = 0
    claim: AgentRecallSubscriptionClaim | None = None
    release: AgentRecallSubscriptionRelease | None = None
    next_evaluation_at: datetime
    last_evaluation_id: str | None = None
    updated_at: datetime

    @field_validator("subscription", mode="before")
    @classmethod
    def copy_subscription(cls, value: object, info) -> AgentRecallSubscription:
        if type(value) is AgentRecallSubscription:
            return copy_agent_recall_subscription(value)
        if info.mode == "json":
            return AgentRecallSubscription.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall subscription")
            )
        return AgentRecallSubscription.model_validate(value)

    @field_validator("run_state", mode="before")
    @classmethod
    def validate_run_state(cls, value: object) -> AgentRecallSubscriptionRunState:
        if isinstance(value, AgentRecallSubscriptionRunState):
            return value
        if type(value) is str:
            return AgentRecallSubscriptionRunState(value)
        raise ValueError("run_state must be an AgentRecallSubscriptionRunState.")

    @field_validator("state_revision", "attempt")
    @classmethod
    def validate_counter(cls, value: int, info) -> int:
        return _sequence(value, info.field_name)

    @field_validator("claim", mode="before")
    @classmethod
    def copy_claim(cls, value: object, info) -> AgentRecallSubscriptionClaim | None:
        if value is None:
            return None
        if type(value) is AgentRecallSubscriptionClaim:
            return copy_agent_recall_subscription_claim(value)
        if info.mode == "json":
            return AgentRecallSubscriptionClaim.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall subscription claim")
            )
        return AgentRecallSubscriptionClaim.model_validate(value)

    @field_validator("release", mode="before")
    @classmethod
    def copy_release(cls, value: object, info) -> AgentRecallSubscriptionRelease | None:
        if value is None:
            return None
        if type(value) is AgentRecallSubscriptionRelease:
            return copy_agent_recall_subscription_release(value)
        if info.mode == "json":
            return AgentRecallSubscriptionRelease.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall subscription release")
            )
        return AgentRecallSubscriptionRelease.model_validate(value)

    @field_validator("next_evaluation_at", "updated_at")
    @classmethod
    def validate_datetime(cls, value: datetime, info) -> datetime:
        return _utc(value, info.field_name)

    @field_validator("last_evaluation_id")
    @classmethod
    def validate_optional_identity(cls, value: str | None) -> str | None:
        return None if value is None else _bounded_identity(value, "last_evaluation_id")

    @model_validator(mode="after")
    def validate_lifecycle(self) -> AgentRecallSubscriptionRecord:
        if self.updated_at < self.subscription.published_at:
            raise ValueError("Subscription state cannot predate its definition.")
        if self.run_state is AgentRecallSubscriptionRunState.CLAIMED:
            if self.claim is None or self.release is not None:
                raise ValueError("Claimed subscription state requires exactly one live claim.")
        elif self.release is not None:
            if self.claim is None:
                raise ValueError("Released subscription state must retain its claim.")
        elif self.claim is not None:
            raise ValueError("Due subscription state cannot retain an unreleased claim.")
        if self.claim is not None and (
            self.claim.subscription_id != self.subscription.subscription_id
            or self.claim.subscription_revision != self.subscription.revision
            or self.claim.subscription_sha256 != self.subscription.fingerprint()
            or self.claim.attempt != self.attempt
            or self.claim.claimed_at < self.subscription.published_at
            or self.updated_at < self.claim.claimed_at
            or (
                self.run_state is AgentRecallSubscriptionRunState.CLAIMED
                and self.claim.state_revision != self.state_revision
            )
            or (self.release is not None and self.claim.state_revision + 1 != self.state_revision)
        ):
            raise ValueError("Subscription claim conflicts with current durable state.")
        if self.release is not None and (
            self.claim is None
            or self.release.subscription_id != self.claim.subscription_id
            or self.release.subscription_revision != self.claim.subscription_revision
            or self.release.subscription_sha256 != self.claim.subscription_sha256
            or self.release.claim_id != self.claim.claim_id
            or self.release.runner_id != self.claim.runner_id
            or self.release.attempt != self.claim.attempt
            or self.release.claim_state_revision != self.claim.state_revision
            or self.release.released_at < self.claim.claimed_at
            or self.release.released_at >= self.claim.lease_expires_at
            or self.updated_at < self.release.released_at
        ):
            raise ValueError("Subscription release conflicts with its retained claim.")
        return self


class AgentRecallSubscriptionEvaluation(_WorkContextModel):
    """Immutable evidence for one atomic subscription evaluation commit."""

    schema_version: Literal["cayu.agent_recall_subscription_evaluation.v1"] = (
        AGENT_RECALL_SUBSCRIPTION_EVALUATION_SCHEMA_VERSION
    )
    evaluation_id: str
    request_sha256: str
    subscription_id: str
    subscription_revision: int
    subscription_sha256: str
    claim_id: str
    runner_id: str
    attempt: int
    claim_state_revision: int
    processing_id: str
    processing_operation_id: str
    processing_result_sha256: str
    contribution_sha256: str | None
    outcome: AgentRecallSubscriptionEvaluationOutcome
    checkpoint_sha256: str | None = None
    checkpoint_revision: int | None = None
    delivery_id: str | None = None
    committed_at: datetime
    next_evaluation_at: datetime

    @field_validator(
        "evaluation_id",
        "subscription_id",
        "claim_id",
        "runner_id",
        "processing_id",
        "processing_operation_id",
        "delivery_id",
    )
    @classmethod
    def validate_identity(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_identity(value, info.field_name)

    @field_validator(
        "request_sha256",
        "subscription_sha256",
        "processing_result_sha256",
        "contribution_sha256",
        "checkpoint_sha256",
    )
    @classmethod
    def validate_digest(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_hex(value, info.field_name)

    @field_validator("subscription_revision", "attempt", "claim_state_revision")
    @classmethod
    def validate_positive(cls, value: int, info) -> int:
        return _positive_revision(value, info.field_name)

    @field_validator("checkpoint_revision")
    @classmethod
    def validate_optional_revision(cls, value: int | None) -> int | None:
        if value is not None:
            _positive_revision(value, "checkpoint_revision")
        return value

    @field_validator("outcome", mode="before")
    @classmethod
    def validate_outcome(cls, value: object) -> AgentRecallSubscriptionEvaluationOutcome:
        if isinstance(value, AgentRecallSubscriptionEvaluationOutcome):
            return value
        if type(value) is str:
            return AgentRecallSubscriptionEvaluationOutcome(value)
        raise ValueError("outcome must be an AgentRecallSubscriptionEvaluationOutcome.")

    @field_validator("committed_at", "next_evaluation_at")
    @classmethod
    def validate_datetime(cls, value: datetime, info) -> datetime:
        return _utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_outcome_evidence(self) -> AgentRecallSubscriptionEvaluation:
        if self.next_evaluation_at <= self.committed_at:
            raise ValueError("Next subscription evaluation must follow its commit.")
        if self.outcome is AgentRecallSubscriptionEvaluationOutcome.NO_WORK:
            if any(
                value is not None
                for value in (
                    self.contribution_sha256,
                    self.checkpoint_sha256,
                    self.checkpoint_revision,
                    self.delivery_id,
                )
            ):
                raise ValueError("No-work subscription evaluation cannot publish recall state.")
        elif self.contribution_sha256 is None or self.checkpoint_sha256 is None:
            raise ValueError("Processed subscription evaluation requires admission and checkpoint.")
        elif self.checkpoint_revision is None:
            raise ValueError("Processed subscription evaluation requires checkpoint revision.")
        elif self.outcome is AgentRecallSubscriptionEvaluationOutcome.WAKE:
            if self.delivery_id is None:
                raise ValueError("Wake evaluation requires a staged delivery.")
        elif self.delivery_id is not None:
            raise ValueError("Silent evaluation cannot publish a staged delivery.")
        return self


class AgentRecallSubscriptionWakeClaim(_WorkContextModel):
    """Store-clock lease over one pending task-scheduler wake signal."""

    schema_version: Literal["cayu.agent_recall_subscription_wake_claim.v1"] = (
        AGENT_RECALL_SUBSCRIPTION_WAKE_CLAIM_SCHEMA_VERSION
    )
    wake_id: str
    delivery_id: str
    claim_id: str
    runner_id: str
    attempt: int
    state_revision: int
    claimed_at: datetime
    lease_expires_at: datetime

    @field_validator("wake_id", "delivery_id", "claim_id", "runner_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("attempt", "state_revision")
    @classmethod
    def validate_revision(cls, value: int, info) -> int:
        return _positive_revision(value, info.field_name)

    @field_validator("claimed_at", "lease_expires_at")
    @classmethod
    def validate_datetime(cls, value: datetime, info) -> datetime:
        return _utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_lease(self) -> AgentRecallSubscriptionWakeClaim:
        if self.lease_expires_at <= self.claimed_at:
            raise ValueError("Wake claim lease must expire after it is claimed.")
        return self


class AgentRecallSubscriptionWakeRelease(_WorkContextModel):
    """Immutable evidence returning one scheduler wake for retry."""

    schema_version: Literal["cayu.agent_recall_subscription_wake_release.v1"] = (
        AGENT_RECALL_SUBSCRIPTION_WAKE_RELEASE_SCHEMA_VERSION
    )
    release_id: str
    wake_id: str
    delivery_id: str
    claim_id: str
    runner_id: str
    attempt: int
    claim_state_revision: int
    reason: str
    released_at: datetime

    @field_validator("release_id", "wake_id", "delivery_id", "claim_id", "runner_id")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("attempt", "claim_state_revision")
    @classmethod
    def validate_revision(cls, value: int, info) -> int:
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
                "agent recall subscription wake release",
            )
        ).hexdigest()


class AgentRecallSubscriptionWakeAcknowledgement(_WorkContextModel):
    """Evidence that an application scheduler accepted one wake request."""

    schema_version: Literal["cayu.agent_recall_subscription_wake_acknowledgement.v1"] = (
        AGENT_RECALL_SUBSCRIPTION_WAKE_ACKNOWLEDGEMENT_SCHEMA_VERSION
    )
    acknowledgement_id: str
    wake_id: str
    delivery_id: str
    claim_id: str
    runner_id: str
    attempt: int
    acknowledged_at: datetime

    @field_validator(
        "acknowledgement_id",
        "wake_id",
        "delivery_id",
        "claim_id",
        "runner_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("attempt")
    @classmethod
    def validate_attempt(cls, value: int) -> int:
        return _positive_revision(value, "attempt")

    @field_validator("acknowledged_at")
    @classmethod
    def validate_acknowledged_at(cls, value: datetime) -> datetime:
        return _utc(value, "acknowledged_at")


class AgentRecallSubscriptionWake(_WorkContextModel):
    """Independent scheduler handoff state beside one exact staged delivery."""

    schema_version: Literal["cayu.agent_recall_subscription_wake.v1"] = (
        AGENT_RECALL_SUBSCRIPTION_WAKE_SCHEMA_VERSION
    )
    wake_id: str
    subscription: AgentRecallSubscription
    evaluation: AgentRecallSubscriptionEvaluation
    delivery: AgentRecallDelivery
    state: AgentRecallSubscriptionWakeState = AgentRecallSubscriptionWakeState.PENDING
    state_revision: int = 0
    attempt: int = 0
    claim: AgentRecallSubscriptionWakeClaim | None = None
    release: AgentRecallSubscriptionWakeRelease | None = None
    acknowledgement: AgentRecallSubscriptionWakeAcknowledgement | None = None
    updated_at: datetime

    @field_validator("wake_id")
    @classmethod
    def validate_wake_id(cls, value: str) -> str:
        return _bounded_identity(value, "wake_id")

    @field_validator("state", mode="before")
    @classmethod
    def validate_state(cls, value: object) -> AgentRecallSubscriptionWakeState:
        if isinstance(value, AgentRecallSubscriptionWakeState):
            return value
        if type(value) is str:
            return AgentRecallSubscriptionWakeState(value)
        raise ValueError("state must be an AgentRecallSubscriptionWakeState.")

    @field_validator("state_revision", "attempt")
    @classmethod
    def validate_counter(cls, value: int, info) -> int:
        return _sequence(value, info.field_name)

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime) -> datetime:
        return _utc(value, "updated_at")

    @model_validator(mode="after")
    def validate_linkage_and_lifecycle(self) -> AgentRecallSubscriptionWake:
        if (
            self.wake_id != self.evaluation.evaluation_id
            or self.evaluation.outcome is not AgentRecallSubscriptionEvaluationOutcome.WAKE
            or self.evaluation.subscription_id != self.subscription.subscription_id
            or self.evaluation.subscription_revision != self.subscription.revision
            or self.evaluation.subscription_sha256 != self.subscription.fingerprint()
            or self.evaluation.delivery_id != self.delivery.delivery_id
            or self.evaluation.processing_result_sha256 != self.delivery.processing_result_sha256
            or self.evaluation.checkpoint_sha256 != self.delivery.checkpoint.fingerprint()
            or self.evaluation.checkpoint_revision != self.delivery.checkpoint.revision
            or self.delivery.agent_id != self.subscription.agent_id
            or self.delivery.task_id != self.subscription.task_id
            or self.delivery.knowledge_namespace != self.subscription.knowledge_namespace
            or self.delivery.access_policy_sha256 != self.subscription.access_policy_sha256
            or self.delivery.work_context_revision != self.subscription.work_context_revision
            or self.delivery.work_context_sha256 != self.subscription.work_context_sha256
            or self.updated_at < self.evaluation.committed_at
        ):
            raise ValueError("Subscription wake conflicts with its evaluation or delivery.")
        if self.state is AgentRecallSubscriptionWakeState.PENDING:
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
                raise ValueError("Pending wake must be initial or retain release evidence.")
        elif self.state is AgentRecallSubscriptionWakeState.CLAIMED:
            if self.claim is None or self.release is not None or self.acknowledgement is not None:
                raise ValueError("Claimed wake requires exactly one live scheduler claim.")
        elif self.claim is None or self.release is not None or self.acknowledgement is None:
            raise ValueError("Acknowledged wake requires its final claim and evidence.")
        if self.claim is not None and (
            self.claim.wake_id != self.wake_id
            or self.claim.delivery_id != self.delivery.delivery_id
            or self.claim.attempt != self.attempt
            or self.claim.claimed_at < self.evaluation.committed_at
            or self.updated_at < self.claim.claimed_at
            or (
                self.state is AgentRecallSubscriptionWakeState.CLAIMED
                and self.claim.state_revision != self.state_revision
            )
            or (
                self.state is not AgentRecallSubscriptionWakeState.CLAIMED
                and self.claim.state_revision + 1 != self.state_revision
            )
        ):
            raise ValueError("Wake claim conflicts with current durable state.")
        if self.release is not None and (
            self.claim is None
            or self.release.wake_id != self.claim.wake_id
            or self.release.delivery_id != self.claim.delivery_id
            or self.release.claim_id != self.claim.claim_id
            or self.release.runner_id != self.claim.runner_id
            or self.release.attempt != self.claim.attempt
            or self.release.claim_state_revision != self.claim.state_revision
            or self.release.released_at < self.claim.claimed_at
            or self.release.released_at >= self.claim.lease_expires_at
            or self.updated_at < self.release.released_at
        ):
            raise ValueError("Wake release conflicts with its retained claim.")
        if self.acknowledgement is not None and (
            self.claim is None
            or self.acknowledgement.wake_id != self.claim.wake_id
            or self.acknowledgement.delivery_id != self.claim.delivery_id
            or self.acknowledgement.claim_id != self.claim.claim_id
            or self.acknowledgement.runner_id != self.claim.runner_id
            or self.acknowledgement.attempt != self.claim.attempt
            or self.acknowledgement.acknowledged_at < self.claim.claimed_at
            or self.acknowledgement.acknowledged_at >= self.claim.lease_expires_at
            or self.updated_at < self.acknowledgement.acknowledged_at
        ):
            raise ValueError("Wake acknowledgement conflicts with its final claim.")
        return self


class AgentRecallSubscriptionConflict(RuntimeError):
    """A subscription operation lost or reused exact transition authority."""

    def __init__(self, code: str) -> None:
        self.code = require_durable_clean_nonblank(code, "code")
        super().__init__(f"Agent recall subscription conflicted ({self.code}).")


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


def copy_agent_recall_subscription(
    value: AgentRecallSubscription,
) -> AgentRecallSubscription:
    if type(value) is not AgentRecallSubscription:
        raise TypeError("value must be an AgentRecallSubscription.")
    return AgentRecallSubscription.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_subscription_publication_receipt(
    value: AgentRecallSubscriptionPublicationReceipt,
) -> AgentRecallSubscriptionPublicationReceipt:
    if type(value) is not AgentRecallSubscriptionPublicationReceipt:
        raise TypeError("value must be an AgentRecallSubscriptionPublicationReceipt.")
    return AgentRecallSubscriptionPublicationReceipt.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_subscription_claim(
    value: AgentRecallSubscriptionClaim,
) -> AgentRecallSubscriptionClaim:
    if type(value) is not AgentRecallSubscriptionClaim:
        raise TypeError("value must be an AgentRecallSubscriptionClaim.")
    return AgentRecallSubscriptionClaim.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_subscription_release(
    value: AgentRecallSubscriptionRelease,
) -> AgentRecallSubscriptionRelease:
    if type(value) is not AgentRecallSubscriptionRelease:
        raise TypeError("value must be an AgentRecallSubscriptionRelease.")
    return AgentRecallSubscriptionRelease.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_subscription_record(
    value: AgentRecallSubscriptionRecord,
) -> AgentRecallSubscriptionRecord:
    if type(value) is not AgentRecallSubscriptionRecord:
        raise TypeError("value must be an AgentRecallSubscriptionRecord.")
    return AgentRecallSubscriptionRecord.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_subscription_evaluation(
    value: AgentRecallSubscriptionEvaluation,
) -> AgentRecallSubscriptionEvaluation:
    if type(value) is not AgentRecallSubscriptionEvaluation:
        raise TypeError("value must be an AgentRecallSubscriptionEvaluation.")
    return AgentRecallSubscriptionEvaluation.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_subscription_wake_claim(
    value: AgentRecallSubscriptionWakeClaim,
) -> AgentRecallSubscriptionWakeClaim:
    if type(value) is not AgentRecallSubscriptionWakeClaim:
        raise TypeError("value must be an AgentRecallSubscriptionWakeClaim.")
    return AgentRecallSubscriptionWakeClaim.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_subscription_wake_release(
    value: AgentRecallSubscriptionWakeRelease,
) -> AgentRecallSubscriptionWakeRelease:
    if type(value) is not AgentRecallSubscriptionWakeRelease:
        raise TypeError("value must be an AgentRecallSubscriptionWakeRelease.")
    return AgentRecallSubscriptionWakeRelease.model_validate(value.model_dump(mode="python"))


def copy_agent_recall_subscription_wake_acknowledgement(
    value: AgentRecallSubscriptionWakeAcknowledgement,
) -> AgentRecallSubscriptionWakeAcknowledgement:
    if type(value) is not AgentRecallSubscriptionWakeAcknowledgement:
        raise TypeError("value must be an AgentRecallSubscriptionWakeAcknowledgement.")
    return AgentRecallSubscriptionWakeAcknowledgement.model_validate(
        value.model_dump(mode="python")
    )


def copy_agent_recall_subscription_wake(
    value: AgentRecallSubscriptionWake,
) -> AgentRecallSubscriptionWake:
    if type(value) is not AgentRecallSubscriptionWake:
        raise TypeError("value must be an AgentRecallSubscriptionWake.")
    return AgentRecallSubscriptionWake.model_validate(value.model_dump(mode="python"))


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


def agent_recall_subscription_publication_request_sha256(
    subscription: AgentRecallSubscription,
    expected_revision: int | None,
) -> str:
    subscription = copy_agent_recall_subscription(subscription)
    if expected_revision is not None:
        _positive_revision(expected_revision, "expected_revision")
    return sha256(
        canonical_durable_json_bytes(
            {
                "subscription": subscription.model_dump(mode="json"),
                "expected_revision": expected_revision,
            },
            "agent recall subscription publication request",
        )
    ).hexdigest()


def validate_agent_recall_subscription_publication(
    subscription: AgentRecallSubscription,
    expected_revision: int | None,
    current: AgentRecallSubscription | None,
    work_context: AgentWorkContext | None,
) -> None:
    subscription = copy_agent_recall_subscription(subscription)
    if expected_revision is not None:
        _positive_revision(expected_revision, "expected_revision")
    expected = 1 if expected_revision is None else expected_revision + 1
    if subscription.revision != expected:
        raise AgentRecallSubscriptionConflict("invalid_successor_revision")
    if (
        work_context is None
        or work_context.task_id != subscription.task_id
        or work_context.revision != subscription.work_context_revision
        or work_context.content_sha256 != subscription.work_context_sha256
    ):
        raise AgentRecallSubscriptionConflict("stale_work_context")
    if any(
        not set(getattr(subscription, field_name)).issubset(getattr(work_context, field_name))
        for field_name in _CONTEXT_COLLECTION_FIELDS
    ):
        raise AgentRecallSubscriptionConflict("subscription_facet_outside_work_context")
    if current is None:
        if expected_revision is not None:
            raise AgentRecallSubscriptionConflict("unknown_subscription")
        return
    if expected_revision is None or current.revision != expected_revision:
        raise AgentRecallSubscriptionConflict("stale_subscription_revision")
    if (
        current.subscription_id != subscription.subscription_id
        or current.agent_id != subscription.agent_id
        or current.task_id != subscription.task_id
        or current.knowledge_namespace != subscription.knowledge_namespace
        or current.access_policy_sha256 != subscription.access_policy_sha256
    ):
        raise AgentRecallSubscriptionConflict("subscription_authority_changed")
    if subscription.published_at < current.published_at:
        raise AgentRecallSubscriptionConflict("subscription_publication_time_regression")
    if (
        current.status is AgentRecallSubscriptionStatus.CANCELLED
        and subscription.status is not AgentRecallSubscriptionStatus.CANCELLED
    ):
        raise AgentRecallSubscriptionConflict("cancelled_subscription_is_terminal")


def agent_recall_subscription_claim_request_sha256(
    key: AgentRecallCheckpointKey,
    *,
    claim_id: str,
    runner_id: str,
    lease_seconds: float,
) -> str:
    key = copy_agent_recall_checkpoint_key(key)
    claim_id = _bounded_identity(claim_id, "claim_id")
    runner_id = _bounded_identity(runner_id, "runner_id")
    lease_seconds = _validate_delivery_lease_seconds(lease_seconds)
    return sha256(
        canonical_durable_json_bytes(
            {
                "key": key.model_dump(mode="json"),
                "claim_id": claim_id,
                "runner_id": runner_id,
                "lease_seconds": lease_seconds,
            },
            "agent recall subscription claim request",
        )
    ).hexdigest()


def _claim_agent_recall_subscription_record(
    record: AgentRecallSubscriptionRecord,
    *,
    claim_id: str,
    runner_id: str,
    lease_seconds: float,
    now: datetime,
) -> AgentRecallSubscriptionRecord:
    record = copy_agent_recall_subscription_record(record)
    claim_id = _bounded_identity(claim_id, "claim_id")
    runner_id = _bounded_identity(runner_id, "runner_id")
    lease_seconds = _validate_delivery_lease_seconds(lease_seconds)
    now = _utc(now, "now")
    subscription = record.subscription
    if subscription.status is not AgentRecallSubscriptionStatus.ACTIVE:
        raise AgentRecallSubscriptionConflict("subscription_inactive")
    if subscription.expires_at <= now:
        raise AgentRecallSubscriptionConflict("subscription_expired")
    if record.next_evaluation_at > now:
        raise AgentRecallSubscriptionConflict("subscription_not_due")
    if record.run_state is AgentRecallSubscriptionRunState.CLAIMED:
        assert record.claim is not None
        if record.claim.claim_id == claim_id and record.claim.runner_id == runner_id:
            return record
        if record.claim.lease_expires_at > now:
            raise AgentRecallSubscriptionConflict("subscription_claimed")
    state_revision = record.state_revision + 1
    claim = AgentRecallSubscriptionClaim(
        subscription_id=subscription.subscription_id,
        subscription_revision=subscription.revision,
        subscription_sha256=subscription.fingerprint(),
        claim_id=claim_id,
        runner_id=runner_id,
        attempt=record.attempt + 1,
        state_revision=state_revision,
        claimed_at=now,
        lease_expires_at=now + timedelta(seconds=lease_seconds),
    )
    return record.model_copy(
        update={
            "run_state": AgentRecallSubscriptionRunState.CLAIMED,
            "state_revision": state_revision,
            "attempt": claim.attempt,
            "claim": claim,
            "release": None,
            "updated_at": now,
        }
    )


def _require_current_subscription_claim(
    record: AgentRecallSubscriptionRecord,
    claim: AgentRecallSubscriptionClaim,
    *,
    now: datetime,
) -> None:
    claim = copy_agent_recall_subscription_claim(claim)
    now = _utc(now, "now")
    if record.run_state is not AgentRecallSubscriptionRunState.CLAIMED or record.claim != claim:
        raise AgentRecallSubscriptionConflict("stale_subscription_claim")
    if claim.lease_expires_at <= now:
        raise AgentRecallSubscriptionConflict("expired_subscription_claim")


def _renew_agent_recall_subscription_record(
    record: AgentRecallSubscriptionRecord,
    claim: AgentRecallSubscriptionClaim,
    *,
    lease_seconds: float,
    now: datetime,
) -> AgentRecallSubscriptionRecord:
    record = copy_agent_recall_subscription_record(record)
    claim = copy_agent_recall_subscription_claim(claim)
    lease_seconds = _validate_delivery_lease_seconds(lease_seconds)
    now = _utc(now, "now")
    if (
        record.run_state is AgentRecallSubscriptionRunState.CLAIMED
        and record.claim is not None
        and record.claim.claim_id == claim.claim_id
        and record.claim.runner_id == claim.runner_id
        and record.claim.attempt == claim.attempt
        and record.claim.state_revision == claim.state_revision + 1
    ):
        if record.claim.lease_expires_at != record.updated_at + timedelta(seconds=lease_seconds):
            raise AgentRecallSubscriptionConflict("renewal_reused")
        if record.claim.lease_expires_at <= now:
            raise AgentRecallSubscriptionConflict("expired_subscription_claim")
        return record
    _require_current_subscription_claim(record, claim, now=now)
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


def _agent_recall_subscription_release(
    claim: AgentRecallSubscriptionClaim,
    *,
    release_id: str,
    reason: str,
    released_at: datetime,
) -> AgentRecallSubscriptionRelease:
    claim = copy_agent_recall_subscription_claim(claim)
    return AgentRecallSubscriptionRelease(
        release_id=release_id,
        subscription_id=claim.subscription_id,
        subscription_revision=claim.subscription_revision,
        subscription_sha256=claim.subscription_sha256,
        claim_id=claim.claim_id,
        runner_id=claim.runner_id,
        attempt=claim.attempt,
        claim_state_revision=claim.state_revision,
        reason=reason,
        released_at=released_at,
    )


def _release_agent_recall_subscription_record(
    record: AgentRecallSubscriptionRecord,
    claim: AgentRecallSubscriptionClaim,
    *,
    release_id: str,
    reason: str,
    released_at: datetime,
    now: datetime,
) -> AgentRecallSubscriptionRecord:
    record = copy_agent_recall_subscription_record(record)
    claim = copy_agent_recall_subscription_claim(claim)
    now = _utc(now, "now")
    requested = _agent_recall_subscription_release(
        claim,
        release_id=release_id,
        reason=reason,
        released_at=released_at,
    )
    if record.run_state is AgentRecallSubscriptionRunState.DUE and record.release == requested:
        return record
    _require_current_subscription_claim(record, claim, now=now)
    if requested.released_at < claim.claimed_at:
        raise AgentRecallSubscriptionConflict("release_predates_claim")
    if requested.released_at >= claim.lease_expires_at:
        raise AgentRecallSubscriptionConflict("release_outside_claim_lease")
    if requested.released_at > now:
        raise AgentRecallSubscriptionConflict("release_from_future")
    return record.model_copy(
        update={
            "run_state": AgentRecallSubscriptionRunState.DUE,
            "state_revision": record.state_revision + 1,
            "release": requested,
            "updated_at": max(now, requested.released_at),
        }
    )


def agent_recall_subscription_evaluation_request_sha256(
    claim: AgentRecallSubscriptionClaim,
    result: AgentRecallProcessingResult,
    *,
    evaluation_id: str,
    delivery_id: str | None,
    staged_by: str,
    evaluated_at: datetime,
) -> str:
    from cayu.recall_processing import AgentRecallProcessingResult

    claim = copy_agent_recall_subscription_claim(claim)
    if type(result) is not AgentRecallProcessingResult:
        raise TypeError("result must be an AgentRecallProcessingResult.")
    evaluation_id = _bounded_identity(evaluation_id, "evaluation_id")
    if delivery_id is not None:
        delivery_id = _bounded_identity(delivery_id, "delivery_id")
    staged_by = _bounded_identity(staged_by, "staged_by")
    evaluated_at = _utc(evaluated_at, "evaluated_at")
    return sha256(
        canonical_durable_json_bytes(
            {
                "claim": claim.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
                "evaluation_id": evaluation_id,
                "delivery_id": delivery_id,
                "staged_by": staged_by,
                "evaluated_at": evaluated_at.isoformat(),
            },
            "agent recall subscription evaluation request",
        )
    ).hexdigest()


def _prepare_agent_recall_subscription_evaluation(
    record: AgentRecallSubscriptionRecord,
    claim: AgentRecallSubscriptionClaim,
    result: AgentRecallProcessingResult,
    current_work_context: AgentWorkContext | None,
    *,
    evaluation_id: str,
    delivery_id: str | None,
    staged_by: str,
    evaluated_at: datetime,
    now: datetime,
) -> tuple[
    AgentRecallSubscriptionEvaluation,
    AgentRecallDelivery | None,
    AgentRecallSubscriptionRecord,
]:
    from cayu.memory import admit_recall
    from cayu.recall_processing import AgentRecallProcessingMode, AgentRecallProcessingResult

    record = copy_agent_recall_subscription_record(record)
    claim = copy_agent_recall_subscription_claim(claim)
    if type(result) is not AgentRecallProcessingResult:
        raise TypeError("result must be an AgentRecallProcessingResult.")
    result = result.model_copy(deep=True)
    evaluation_id = _bounded_identity(evaluation_id, "evaluation_id")
    staged_by = _bounded_identity(staged_by, "staged_by")
    evaluated_at = _utc(evaluated_at, "evaluated_at")
    now = _utc(now, "now")
    _require_current_subscription_claim(record, claim, now=now)
    subscription = record.subscription
    if subscription.status is not AgentRecallSubscriptionStatus.ACTIVE:
        raise AgentRecallSubscriptionConflict("inactive_subscription")
    if subscription.expires_at <= now:
        raise AgentRecallSubscriptionConflict("expired_subscription")
    if (
        current_work_context is None
        or current_work_context.task_id != subscription.task_id
        or current_work_context.revision != subscription.work_context_revision
        or current_work_context.content_sha256 != subscription.work_context_sha256
    ):
        raise AgentRecallSubscriptionConflict("stale_work_context")
    if evaluated_at > now:
        raise AgentRecallSubscriptionConflict("evaluation_from_future")
    if evaluated_at < claim.claimed_at:
        raise AgentRecallSubscriptionConflict("evaluation_predates_claim")
    if (
        result.agent_id != subscription.agent_id
        or result.task_id != subscription.task_id
        or result.knowledge_namespace != subscription.knowledge_namespace
        or result.access_policy_sha256 != subscription.access_policy_sha256
        or result.checkpoint_stream_id != subscription.checkpoint_stream_id()
        or result.work_context_revision != subscription.work_context_revision
        or result.work_context_sha256 != subscription.work_context_sha256
        or result.situation_sha256 != subscription.situation_sha256()
    ):
        raise AgentRecallSubscriptionConflict("evaluation_authority_mismatch")
    request_sha256 = agent_recall_subscription_evaluation_request_sha256(
        claim,
        result,
        evaluation_id=evaluation_id,
        delivery_id=delivery_id,
        staged_by=staged_by,
        evaluated_at=evaluated_at,
    )
    contribution_sha256: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_revision: int | None = None
    delivery: AgentRecallDelivery | None = None
    if result.mode is AgentRecallProcessingMode.NO_WORK:
        outcome = AgentRecallSubscriptionEvaluationOutcome.NO_WORK
        if delivery_id is not None:
            raise AgentRecallSubscriptionConflict("no_work_delivery")
    else:
        if result.retry_required or result.proposed_checkpoint is None or result.recall is None:
            raise AgentRecallSubscriptionConflict("evaluation_retry_required")
        contribution = admit_recall(result.recall, subscription.policy())
        contribution_sha256 = sha256(
            canonical_durable_json_bytes(
                contribution.model_dump(mode="json"),
                "agent recall subscription contribution",
            )
        ).hexdigest()
        checkpoint = result.proposed_checkpoint
        checkpoint_sha256 = checkpoint.fingerprint()
        checkpoint_revision = checkpoint.revision
        should_wake = contribution.focus is not None or contribution.offer is not None
        outcome = (
            AgentRecallSubscriptionEvaluationOutcome.WAKE
            if should_wake
            else AgentRecallSubscriptionEvaluationOutcome.SILENT
        )
        if should_wake:
            if delivery_id is None:
                raise AgentRecallSubscriptionConflict("wake_delivery_required")
            expected_checkpoint_revision = (
                None if checkpoint.revision == 1 else checkpoint.revision - 1
            )
            delivery = AgentRecallDelivery.from_processing_result(
                result,
                delivery_id=delivery_id,
                expected_checkpoint_revision=expected_checkpoint_revision,
                staged_by=staged_by,
                staged_at=evaluated_at,
            )
        elif delivery_id is not None:
            raise AgentRecallSubscriptionConflict("silent_delivery_forbidden")
    next_evaluation_at = max(now, evaluated_at) + timedelta(
        seconds=subscription.minimum_interval_seconds
    )
    evaluation = AgentRecallSubscriptionEvaluation(
        evaluation_id=evaluation_id,
        request_sha256=request_sha256,
        subscription_id=subscription.subscription_id,
        subscription_revision=subscription.revision,
        subscription_sha256=subscription.fingerprint(),
        claim_id=claim.claim_id,
        runner_id=claim.runner_id,
        attempt=claim.attempt,
        claim_state_revision=claim.state_revision,
        processing_id=result.processing_id,
        processing_operation_id=result.operation_id,
        processing_result_sha256=result.fingerprint(),
        contribution_sha256=contribution_sha256,
        outcome=outcome,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_revision=checkpoint_revision,
        delivery_id=None if delivery is None else delivery.delivery_id,
        committed_at=max(now, evaluated_at),
        next_evaluation_at=next_evaluation_at,
    )
    updated_record = record.model_copy(
        update={
            "run_state": AgentRecallSubscriptionRunState.DUE,
            "state_revision": record.state_revision + 1,
            "claim": None,
            "release": None,
            "next_evaluation_at": next_evaluation_at,
            "last_evaluation_id": evaluation_id,
            "updated_at": evaluation.committed_at,
        }
    )
    return evaluation, delivery, updated_record


def agent_recall_subscription_wake_claim_request_sha256(
    key: AgentRecallCheckpointKey,
    *,
    claim_id: str,
    runner_id: str,
    lease_seconds: float,
) -> str:
    key = copy_agent_recall_checkpoint_key(key)
    claim_id = _bounded_identity(claim_id, "claim_id")
    runner_id = _bounded_identity(runner_id, "runner_id")
    lease_seconds = _validate_delivery_lease_seconds(lease_seconds)
    return sha256(
        canonical_durable_json_bytes(
            {
                "key": key.model_dump(mode="json"),
                "claim_id": claim_id,
                "runner_id": runner_id,
                "lease_seconds": lease_seconds,
            },
            "agent recall subscription wake claim request",
        )
    ).hexdigest()


def _claim_agent_recall_subscription_wake(
    wake: AgentRecallSubscriptionWake,
    *,
    claim_id: str,
    runner_id: str,
    lease_seconds: float,
    now: datetime,
) -> AgentRecallSubscriptionWake:
    wake = copy_agent_recall_subscription_wake(wake)
    claim_id = _bounded_identity(claim_id, "claim_id")
    runner_id = _bounded_identity(runner_id, "runner_id")
    lease_seconds = _validate_delivery_lease_seconds(lease_seconds)
    now = _utc(now, "now")
    if wake.state is AgentRecallSubscriptionWakeState.ACKNOWLEDGED:
        raise AgentRecallSubscriptionConflict("wake_already_acknowledged")
    if wake.state is AgentRecallSubscriptionWakeState.CLAIMED:
        assert wake.claim is not None
        if wake.claim.claim_id == claim_id and wake.claim.runner_id == runner_id:
            return wake
        if wake.claim.lease_expires_at > now:
            raise AgentRecallSubscriptionConflict("wake_claimed")
    state_revision = wake.state_revision + 1
    claim = AgentRecallSubscriptionWakeClaim(
        wake_id=wake.wake_id,
        delivery_id=wake.delivery.delivery_id,
        claim_id=claim_id,
        runner_id=runner_id,
        attempt=wake.attempt + 1,
        state_revision=state_revision,
        claimed_at=now,
        lease_expires_at=now + timedelta(seconds=lease_seconds),
    )
    return wake.model_copy(
        update={
            "state": AgentRecallSubscriptionWakeState.CLAIMED,
            "state_revision": state_revision,
            "attempt": claim.attempt,
            "claim": claim,
            "release": None,
            "updated_at": now,
        }
    )


def _require_replayable_subscription_wake_claim(
    wake: AgentRecallSubscriptionWake,
    *,
    claim_id: str,
    runner_id: str,
    attempt: int,
    now: datetime,
) -> None:
    claim = wake.claim
    if (
        wake.state is not AgentRecallSubscriptionWakeState.CLAIMED
        or claim is None
        or claim.claim_id != claim_id
        or claim.runner_id != runner_id
        or claim.attempt != attempt
    ):
        raise AgentRecallSubscriptionConflict("wake_claim_replay_superseded")
    if claim.lease_expires_at <= _utc(now, "now"):
        raise AgentRecallSubscriptionConflict("expired_wake_claim")


def _require_current_subscription_wake_claim(
    wake: AgentRecallSubscriptionWake,
    claim: AgentRecallSubscriptionWakeClaim,
    *,
    now: datetime,
) -> None:
    claim = copy_agent_recall_subscription_wake_claim(claim)
    if wake.state is not AgentRecallSubscriptionWakeState.CLAIMED or wake.claim != claim:
        raise AgentRecallSubscriptionConflict("stale_wake_claim")
    if claim.lease_expires_at <= _utc(now, "now"):
        raise AgentRecallSubscriptionConflict("expired_wake_claim")


def _renew_agent_recall_subscription_wake(
    wake: AgentRecallSubscriptionWake,
    claim: AgentRecallSubscriptionWakeClaim,
    *,
    lease_seconds: float,
    now: datetime,
) -> AgentRecallSubscriptionWake:
    wake = copy_agent_recall_subscription_wake(wake)
    claim = copy_agent_recall_subscription_wake_claim(claim)
    lease_seconds = _validate_delivery_lease_seconds(lease_seconds)
    now = _utc(now, "now")
    if (
        wake.state is AgentRecallSubscriptionWakeState.CLAIMED
        and wake.claim is not None
        and wake.claim.claim_id == claim.claim_id
        and wake.claim.runner_id == claim.runner_id
        and wake.claim.attempt == claim.attempt
        and wake.claim.state_revision == claim.state_revision + 1
    ):
        if wake.claim.lease_expires_at != wake.updated_at + timedelta(seconds=lease_seconds):
            raise AgentRecallSubscriptionConflict("wake_renewal_reused")
        if wake.claim.lease_expires_at <= now:
            raise AgentRecallSubscriptionConflict("expired_wake_claim")
        return wake
    _require_current_subscription_wake_claim(wake, claim, now=now)
    state_revision = wake.state_revision + 1
    renewed = claim.model_copy(
        update={
            "state_revision": state_revision,
            "lease_expires_at": now + timedelta(seconds=lease_seconds),
        }
    )
    return wake.model_copy(
        update={
            "state_revision": state_revision,
            "claim": renewed,
            "updated_at": now,
        }
    )


def _agent_recall_subscription_wake_release(
    claim: AgentRecallSubscriptionWakeClaim,
    *,
    release_id: str,
    reason: str,
    released_at: datetime,
) -> AgentRecallSubscriptionWakeRelease:
    claim = copy_agent_recall_subscription_wake_claim(claim)
    return AgentRecallSubscriptionWakeRelease(
        release_id=release_id,
        wake_id=claim.wake_id,
        delivery_id=claim.delivery_id,
        claim_id=claim.claim_id,
        runner_id=claim.runner_id,
        attempt=claim.attempt,
        claim_state_revision=claim.state_revision,
        reason=reason,
        released_at=released_at,
    )


def _release_agent_recall_subscription_wake(
    wake: AgentRecallSubscriptionWake,
    claim: AgentRecallSubscriptionWakeClaim,
    *,
    release_id: str,
    reason: str,
    released_at: datetime,
    now: datetime,
) -> AgentRecallSubscriptionWake:
    wake = copy_agent_recall_subscription_wake(wake)
    claim = copy_agent_recall_subscription_wake_claim(claim)
    requested = _agent_recall_subscription_wake_release(
        claim,
        release_id=release_id,
        reason=reason,
        released_at=released_at,
    )
    now = _utc(now, "now")
    if wake.state is AgentRecallSubscriptionWakeState.PENDING and wake.release == requested:
        return wake
    _require_current_subscription_wake_claim(wake, claim, now=now)
    if requested.released_at < claim.claimed_at:
        raise AgentRecallSubscriptionConflict("wake_release_predates_claim")
    if requested.released_at >= claim.lease_expires_at:
        raise AgentRecallSubscriptionConflict("wake_release_outside_claim_lease")
    if requested.released_at > now:
        raise AgentRecallSubscriptionConflict("wake_release_from_future")
    return wake.model_copy(
        update={
            "state": AgentRecallSubscriptionWakeState.PENDING,
            "state_revision": wake.state_revision + 1,
            "release": requested,
            "updated_at": max(now, requested.released_at),
        }
    )


def _acknowledge_agent_recall_subscription_wake(
    wake: AgentRecallSubscriptionWake,
    claim: AgentRecallSubscriptionWakeClaim,
    *,
    acknowledgement_id: str,
    acknowledged_at: datetime,
    now: datetime,
) -> AgentRecallSubscriptionWake:
    wake = copy_agent_recall_subscription_wake(wake)
    claim = copy_agent_recall_subscription_wake_claim(claim)
    requested = AgentRecallSubscriptionWakeAcknowledgement(
        acknowledgement_id=acknowledgement_id,
        wake_id=claim.wake_id,
        delivery_id=claim.delivery_id,
        claim_id=claim.claim_id,
        runner_id=claim.runner_id,
        attempt=claim.attempt,
        acknowledged_at=acknowledged_at,
    )
    now = _utc(now, "now")
    if wake.state is AgentRecallSubscriptionWakeState.ACKNOWLEDGED:
        if wake.acknowledgement != requested:
            raise AgentRecallSubscriptionConflict("wake_already_acknowledged")
        return wake
    _require_current_subscription_wake_claim(wake, claim, now=now)
    if requested.acknowledged_at < claim.claimed_at:
        raise AgentRecallSubscriptionConflict("wake_acknowledgement_predates_claim")
    if requested.acknowledged_at >= claim.lease_expires_at:
        raise AgentRecallSubscriptionConflict("wake_acknowledgement_outside_claim_lease")
    if requested.acknowledged_at > now:
        raise AgentRecallSubscriptionConflict("wake_acknowledgement_from_future")
    return wake.model_copy(
        update={
            "state": AgentRecallSubscriptionWakeState.ACKNOWLEDGED,
            "state_revision": wake.state_revision + 1,
            "acknowledgement": requested,
            "updated_at": max(now, requested.acknowledged_at),
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

    @abstractmethod
    async def publish_recall_subscription(
        self,
        subscription: AgentRecallSubscription,
        *,
        expected_revision: int | None,
    ) -> AgentRecallSubscriptionPublicationReceipt:
        """Create or replace one exact idle-task subscription revision."""

    @abstractmethod
    async def load_recall_subscription(
        self,
        subscription_id: str,
        *,
        revision: int | None = None,
    ) -> AgentRecallSubscription | None:
        """Load the current or one exact immutable subscription revision."""

    @abstractmethod
    async def claim_due_recall_subscription(
        self,
        key: AgentRecallCheckpointKey,
        *,
        claim_id: str,
        runner_id: str,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionRecord | None:
        """Claim the oldest due subscription in one authorized checkpoint scope."""

    @abstractmethod
    async def renew_recall_subscription(
        self,
        claim: AgentRecallSubscriptionClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionRecord:
        """Renew one exact live subscription-evaluation claim."""

    @abstractmethod
    async def release_recall_subscription(
        self,
        claim: AgentRecallSubscriptionClaim,
        *,
        release_id: str,
        reason: str,
        released_at: datetime,
    ) -> AgentRecallSubscriptionRecord:
        """Release one exact evaluation claim without advancing recall progress."""

    @abstractmethod
    async def commit_recall_subscription_evaluation(
        self,
        claim: AgentRecallSubscriptionClaim,
        result: AgentRecallProcessingResult,
        *,
        evaluation_id: str,
        delivery_id: str | None,
        staged_by: str,
        evaluated_at: datetime,
    ) -> AgentRecallSubscriptionEvaluation:
        """Atomically commit silent progress or progress plus one staged wake."""

    @abstractmethod
    async def load_recall_subscription_evaluation(
        self,
        evaluation_id: str,
    ) -> AgentRecallSubscriptionEvaluation | None:
        """Load immutable subscription evaluation evidence."""

    @abstractmethod
    async def claim_recall_subscription_wake(
        self,
        key: AgentRecallCheckpointKey,
        *,
        claim_id: str,
        runner_id: str,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionWake | None:
        """Claim the oldest pending task-scheduler wake created by a subscription."""

    @abstractmethod
    async def load_recall_subscription_wake(
        self,
        wake_id: str,
    ) -> AgentRecallSubscriptionWake | None:
        """Load one scheduler wake without claiming its staged delivery."""

    @abstractmethod
    async def renew_recall_subscription_wake(
        self,
        claim: AgentRecallSubscriptionWakeClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionWake:
        """Renew one exact live scheduler-wake claim."""

    @abstractmethod
    async def release_recall_subscription_wake(
        self,
        claim: AgentRecallSubscriptionWakeClaim,
        *,
        release_id: str,
        reason: str,
        released_at: datetime,
    ) -> AgentRecallSubscriptionWake:
        """Release a scheduler wake for retry without changing its delivery."""

    @abstractmethod
    async def acknowledge_recall_subscription_wake(
        self,
        claim: AgentRecallSubscriptionWakeClaim,
        *,
        acknowledgement_id: str,
        acknowledged_at: datetime,
    ) -> AgentRecallSubscriptionWake:
        """Record scheduler acceptance and make the staged delivery claimable."""

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
            tuple[str, str, str, str, str], dict[int, AgentRecallCheckpoint]
        ] = {}
        self._checkpoint_operations: dict[str, tuple[tuple[str, str, str, str, str], int]] = {}
        self._checkpoint_heads: dict[tuple[str, str, str, str, str], int] = {}
        self._recall_deliveries: dict[str, AgentRecallDeliveryRecord] = {}
        self._delivery_by_checkpoint: dict[tuple[tuple[str, str, str, str, str], int], str] = {}
        self._delivery_by_operation: dict[str, str] = {}
        self._delivery_queues: dict[tuple[str, str, str, str, str], deque[str]] = {}
        self._delivery_claims: dict[str, tuple[str, str, str, int]] = {}
        self._delivery_releases: dict[str, AgentRecallDeliveryRelease] = {}
        self._delivery_acknowledgements: dict[str, str] = {}
        self._subscription_revisions: dict[str, dict[int, AgentRecallSubscription]] = {}
        self._subscription_heads: dict[str, int] = {}
        self._subscription_publications: dict[str, AgentRecallSubscriptionPublicationReceipt] = {}
        self._subscription_states: dict[str, AgentRecallSubscriptionRecord] = {}
        self._subscription_ids_by_key: dict[tuple[str, str, str, str], set[str]] = {}
        self._subscription_claims: dict[str, tuple[str, str, str, int]] = {}
        self._subscription_releases: dict[str, AgentRecallSubscriptionRelease] = {}
        self._subscription_evaluations: dict[str, AgentRecallSubscriptionEvaluation] = {}
        self._subscription_evaluation_by_processing_operation: dict[str, str] = {}
        self._subscription_evaluation_by_delivery: dict[str, str] = {}
        self._subscription_wakes: dict[str, AgentRecallSubscriptionWake] = {}
        self._subscription_wake_queues: dict[tuple[str, str, str, str], deque[str]] = {}
        self._pending_subscription_wake_ids: dict[str, set[str]] = {}
        self._subscription_wake_claims: dict[str, tuple[str, str, str, int]] = {}
        self._subscription_wake_releases: dict[str, AgentRecallSubscriptionWakeRelease] = {}
        self._subscription_wake_acknowledgements: dict[str, str] = {}

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
            if (
                checkpoint.operation_id in self._delivery_by_operation
                or checkpoint.operation_id in self._subscription_evaluation_by_processing_operation
            ):
                raise AgentWorkContextConflict("checkpoint_operation_reused")
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
            if delivery.operation_id in self._subscription_evaluation_by_processing_operation:
                raise AgentRecallDeliveryConflict("delivery_operation_reused")
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

    async def publish_recall_subscription(
        self,
        subscription: AgentRecallSubscription,
        *,
        expected_revision: int | None,
    ) -> AgentRecallSubscriptionPublicationReceipt:
        subscription = copy_agent_recall_subscription(subscription)
        request_sha256 = agent_recall_subscription_publication_request_sha256(
            subscription,
            expected_revision,
        )
        async with self._lock:
            replay = self._subscription_publications.get(subscription.operation_id)
            if replay is not None:
                if replay.request_sha256 != request_sha256:
                    raise AgentRecallSubscriptionConflict("publication_operation_reused")
                return copy_agent_recall_subscription_publication_receipt(replay)
            revisions = self._subscription_revisions.get(subscription.subscription_id)
            current_revision = self._subscription_heads.get(subscription.subscription_id)
            current = (
                None
                if revisions is None or current_revision is None
                else revisions[current_revision]
            )
            contexts = self._contexts.get(subscription.task_id)
            context_revision = self._context_heads.get(subscription.task_id)
            work_context = (
                None if contexts is None or context_revision is None else contexts[context_revision]
            )
            now = _utc(self._clock(), "clock result")
            if subscription.published_at > now:
                raise AgentRecallSubscriptionConflict("publication_from_future")
            validate_agent_recall_subscription_publication(
                subscription,
                expected_revision,
                current,
                work_context,
            )
            receipt = AgentRecallSubscriptionPublicationReceipt(
                operation_id=subscription.operation_id,
                request_sha256=request_sha256,
                expected_revision=expected_revision,
                subscription=subscription,
                committed_at=now,
            )
            prior_state = self._subscription_states.get(subscription.subscription_id)
            state_revision = 0 if prior_state is None else prior_state.state_revision + 1
            state = AgentRecallSubscriptionRecord(
                subscription=subscription,
                state_revision=state_revision,
                attempt=0 if prior_state is None else prior_state.attempt,
                next_evaluation_at=max(now, subscription.published_at),
                updated_at=now,
            )
            self._subscription_revisions.setdefault(subscription.subscription_id, {})[
                subscription.revision
            ] = subscription
            self._subscription_heads[subscription.subscription_id] = subscription.revision
            self._subscription_publications[subscription.operation_id] = receipt
            self._subscription_states[subscription.subscription_id] = state
            self._subscription_ids_by_key.setdefault(
                subscription.checkpoint_key().authority_sort_key(),
                set(),
            ).add(subscription.subscription_id)
            return copy_agent_recall_subscription_publication_receipt(receipt)

    async def load_recall_subscription(
        self,
        subscription_id: str,
        *,
        revision: int | None = None,
    ) -> AgentRecallSubscription | None:
        subscription_id = _bounded_identity(subscription_id, "subscription_id")
        if revision is not None:
            _positive_revision(revision, "revision")
        async with self._lock:
            revisions = self._subscription_revisions.get(subscription_id)
            if not revisions:
                return None
            resolved = self._subscription_heads[subscription_id] if revision is None else revision
            subscription = revisions.get(resolved)
            return None if subscription is None else copy_agent_recall_subscription(subscription)

    async def claim_due_recall_subscription(
        self,
        key: AgentRecallCheckpointKey,
        *,
        claim_id: str,
        runner_id: str,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionRecord | None:
        key = copy_agent_recall_checkpoint_key(key)
        request_sha256 = agent_recall_subscription_claim_request_sha256(
            key,
            claim_id=claim_id,
            runner_id=runner_id,
            lease_seconds=lease_seconds,
        )
        async with self._lock:
            replay = self._subscription_claims.get(claim_id)
            if replay is not None:
                replay_request, subscription_id, replay_runner, replay_attempt = replay
                if replay_request != request_sha256:
                    raise AgentRecallSubscriptionConflict("claim_id_reused")
                record = self._subscription_states[subscription_id]
                current = record.claim
                now = max(_utc(self._clock(), "clock result"), record.updated_at)
                if (
                    record.run_state is not AgentRecallSubscriptionRunState.CLAIMED
                    or current is None
                    or current.claim_id != claim_id
                    or current.runner_id != replay_runner
                    or current.attempt != replay_attempt
                ):
                    raise AgentRecallSubscriptionConflict("claim_replay_superseded")
                if current.lease_expires_at <= now:
                    raise AgentRecallSubscriptionConflict("expired_subscription_claim")
                return copy_agent_recall_subscription_record(record)
            now = _utc(self._clock(), "clock result")
            candidates: list[AgentRecallSubscriptionRecord] = []
            for subscription_id in self._subscription_ids_by_key.get(
                key.authority_sort_key(),
                (),
            ):
                record = self._subscription_states[subscription_id]
                subscription = record.subscription
                contexts = self._contexts.get(subscription.task_id)
                current_context_revision = self._context_heads.get(subscription.task_id)
                current_context = (
                    None
                    if contexts is None or current_context_revision is None
                    else contexts[current_context_revision]
                )
                if (
                    current_context is None
                    or current_context.revision != subscription.work_context_revision
                    or current_context.content_sha256 != subscription.work_context_sha256
                    or subscription.status is not AgentRecallSubscriptionStatus.ACTIVE
                    or subscription.expires_at <= now
                    or record.next_evaluation_at > now
                    or self._subscription_has_pending_wake_unlocked(subscription.subscription_id)
                    or (
                        record.run_state is AgentRecallSubscriptionRunState.CLAIMED
                        and record.claim is not None
                        and record.claim.lease_expires_at > now
                    )
                ):
                    continue
                candidates.append(record)
            if not candidates:
                return None
            current = min(
                candidates,
                key=lambda item: (
                    item.next_evaluation_at,
                    -item.subscription.priority,
                    item.subscription.subscription_id,
                ),
            )
            effective_now = max(now, current.updated_at)
            claimed = _claim_agent_recall_subscription_record(
                current,
                claim_id=claim_id,
                runner_id=runner_id,
                lease_seconds=lease_seconds,
                now=effective_now,
            )
            self._subscription_states[current.subscription.subscription_id] = claimed
            assert claimed.claim is not None
            self._subscription_claims[claim_id] = (
                request_sha256,
                claimed.subscription.subscription_id,
                claimed.claim.runner_id,
                claimed.claim.attempt,
            )
            return copy_agent_recall_subscription_record(claimed)

    async def renew_recall_subscription(
        self,
        claim: AgentRecallSubscriptionClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionRecord:
        claim = copy_agent_recall_subscription_claim(claim)
        async with self._lock:
            current = self._subscription_states.get(claim.subscription_id)
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_subscription")
            renewed = _renew_agent_recall_subscription_record(
                current,
                claim,
                lease_seconds=lease_seconds,
                now=max(_utc(self._clock(), "clock result"), current.updated_at),
            )
            self._subscription_states[claim.subscription_id] = renewed
            return copy_agent_recall_subscription_record(renewed)

    async def release_recall_subscription(
        self,
        claim: AgentRecallSubscriptionClaim,
        *,
        release_id: str,
        reason: str,
        released_at: datetime,
    ) -> AgentRecallSubscriptionRecord:
        claim = copy_agent_recall_subscription_claim(claim)
        release_id = _bounded_identity(release_id, "release_id")
        requested = _agent_recall_subscription_release(
            claim,
            release_id=release_id,
            reason=reason,
            released_at=released_at,
        )
        async with self._lock:
            current = self._subscription_states.get(claim.subscription_id)
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_subscription")
            replay = self._subscription_releases.get(release_id)
            if replay is not None:
                if replay != requested:
                    raise AgentRecallSubscriptionConflict("release_id_reused")
                if current.release != replay:
                    raise AgentRecallSubscriptionConflict("release_replay_superseded")
                return copy_agent_recall_subscription_record(current)
            released = _release_agent_recall_subscription_record(
                current,
                claim,
                release_id=release_id,
                reason=reason,
                released_at=released_at,
                now=max(_utc(self._clock(), "clock result"), current.updated_at),
            )
            self._subscription_states[claim.subscription_id] = released
            assert released.release is not None
            self._subscription_releases[release_id] = released.release
            return copy_agent_recall_subscription_record(released)

    async def commit_recall_subscription_evaluation(
        self,
        claim: AgentRecallSubscriptionClaim,
        result: AgentRecallProcessingResult,
        *,
        evaluation_id: str,
        delivery_id: str | None,
        staged_by: str,
        evaluated_at: datetime,
    ) -> AgentRecallSubscriptionEvaluation:
        from cayu.recall_processing import AgentRecallProcessingResult

        claim = copy_agent_recall_subscription_claim(claim)
        if type(result) is not AgentRecallProcessingResult:
            raise TypeError("result must be an AgentRecallProcessingResult.")
        request_sha256 = agent_recall_subscription_evaluation_request_sha256(
            claim,
            result,
            evaluation_id=evaluation_id,
            delivery_id=delivery_id,
            staged_by=staged_by,
            evaluated_at=evaluated_at,
        )
        async with self._lock:
            replay = self._subscription_evaluations.get(evaluation_id)
            if replay is not None:
                if replay.request_sha256 != request_sha256:
                    raise AgentRecallSubscriptionConflict("evaluation_id_reused")
                return copy_agent_recall_subscription_evaluation(replay)
            occupied_evaluation = self._subscription_evaluation_by_processing_operation.get(
                result.operation_id
            )
            if occupied_evaluation is not None:
                raise AgentRecallSubscriptionConflict("processing_operation_reused")
            if (
                result.operation_id in self._checkpoint_operations
                or result.operation_id in self._delivery_by_operation
            ):
                raise AgentRecallSubscriptionConflict("processing_operation_reused")
            current = self._subscription_states.get(claim.subscription_id)
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_subscription")
            evaluation, delivery, updated = _prepare_agent_recall_subscription_evaluation(
                current,
                claim,
                result,
                (
                    None
                    if self._context_heads.get(current.subscription.task_id) is None
                    else self._contexts[current.subscription.task_id][
                        self._context_heads[current.subscription.task_id]
                    ]
                ),
                evaluation_id=evaluation_id,
                delivery_id=delivery_id,
                staged_by=staged_by,
                evaluated_at=evaluated_at,
                now=max(_utc(self._clock(), "clock result"), current.updated_at),
            )
            if delivery is not None:
                key = delivery.key().sort_key()
                if delivery.delivery_id in self._recall_deliveries:
                    raise AgentRecallSubscriptionConflict("delivery_id_reused")
                if (key, delivery.checkpoint.revision) in self._delivery_by_checkpoint:
                    raise AgentRecallSubscriptionConflict("checkpoint_delivery_exists")
                if delivery.operation_id in self._delivery_by_operation:
                    raise AgentRecallSubscriptionConflict("delivery_operation_reused")
                if delivery.operation_id in self._checkpoint_operations:
                    raise AgentRecallSubscriptionConflict("checkpoint_committed_without_delivery")
                self._advance_recall_checkpoint_unlocked(
                    delivery.checkpoint,
                    expected_revision=delivery.expected_checkpoint_revision,
                )
                delivery_record = AgentRecallDeliveryRecord(
                    delivery=delivery,
                    updated_at=delivery.staged_at,
                )
                self._recall_deliveries[delivery.delivery_id] = delivery_record
                self._delivery_by_checkpoint[(key, delivery.checkpoint.revision)] = (
                    delivery.delivery_id
                )
                self._delivery_by_operation[delivery.operation_id] = delivery.delivery_id
                self._subscription_evaluation_by_delivery[delivery.delivery_id] = evaluation_id
                self._subscription_wakes[evaluation_id] = AgentRecallSubscriptionWake(
                    wake_id=evaluation_id,
                    subscription=current.subscription,
                    evaluation=evaluation,
                    delivery=delivery,
                    updated_at=evaluation.committed_at,
                )
                self._subscription_wake_queues.setdefault(
                    delivery.key().authority_sort_key(),
                    deque(),
                ).append(evaluation_id)
                self._pending_subscription_wake_ids.setdefault(
                    current.subscription.subscription_id,
                    set(),
                ).add(evaluation_id)
            elif result.proposed_checkpoint is not None:
                expected_revision = (
                    None
                    if result.proposed_checkpoint.revision == 1
                    else result.proposed_checkpoint.revision - 1
                )
                self._advance_recall_checkpoint_unlocked(
                    result.proposed_checkpoint,
                    expected_revision=expected_revision,
                )
            self._subscription_evaluations[evaluation_id] = evaluation
            self._subscription_evaluation_by_processing_operation[result.operation_id] = (
                evaluation_id
            )
            self._subscription_states[claim.subscription_id] = updated
            return copy_agent_recall_subscription_evaluation(evaluation)

    async def load_recall_subscription_evaluation(
        self,
        evaluation_id: str,
    ) -> AgentRecallSubscriptionEvaluation | None:
        evaluation_id = _bounded_identity(evaluation_id, "evaluation_id")
        async with self._lock:
            evaluation = self._subscription_evaluations.get(evaluation_id)
            return (
                None
                if evaluation is None
                else copy_agent_recall_subscription_evaluation(evaluation)
            )

    async def claim_recall_subscription_wake(
        self,
        key: AgentRecallCheckpointKey,
        *,
        claim_id: str,
        runner_id: str,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionWake | None:
        key = copy_agent_recall_checkpoint_key(key)
        request_sha256 = agent_recall_subscription_wake_claim_request_sha256(
            key,
            claim_id=claim_id,
            runner_id=runner_id,
            lease_seconds=lease_seconds,
        )
        async with self._lock:
            replay = self._subscription_wake_claims.get(claim_id)
            if replay is not None:
                replay_request, wake_id, replay_runner, replay_attempt = replay
                if replay_request != request_sha256:
                    raise AgentRecallSubscriptionConflict("wake_claim_id_reused")
                wake = self._subscription_wakes[wake_id]
                _require_replayable_subscription_wake_claim(
                    wake,
                    claim_id=claim_id,
                    runner_id=replay_runner,
                    attempt=replay_attempt,
                    now=max(_utc(self._clock(), "clock result"), wake.updated_at),
                )
                return copy_agent_recall_subscription_wake(wake)
            queue_key = key.authority_sort_key()
            self._trim_subscription_wake_queue_unlocked(queue_key)
            queue = self._subscription_wake_queues.get(queue_key)
            if not queue:
                return None
            clock_now = _utc(self._clock(), "clock result")
            candidates: list[AgentRecallSubscriptionWake] = []
            for wake_id in queue:
                wake = self._subscription_wakes[wake_id]
                if wake.state is AgentRecallSubscriptionWakeState.PENDING or (
                    wake.state is AgentRecallSubscriptionWakeState.CLAIMED
                    and wake.claim is not None
                    and wake.claim.lease_expires_at <= max(clock_now, wake.updated_at)
                ):
                    candidates.append(wake)
            if not candidates:
                return None
            current = min(
                candidates,
                key=lambda wake: (wake.evaluation.committed_at, wake.wake_id),
            )
            now = max(clock_now, current.updated_at)
            claimed = _claim_agent_recall_subscription_wake(
                current,
                claim_id=claim_id,
                runner_id=runner_id,
                lease_seconds=lease_seconds,
                now=now,
            )
            self._subscription_wakes[current.wake_id] = claimed
            assert claimed.claim is not None
            self._subscription_wake_claims[claim_id] = (
                request_sha256,
                current.wake_id,
                claimed.claim.runner_id,
                claimed.claim.attempt,
            )
            return copy_agent_recall_subscription_wake(claimed)

    async def load_recall_subscription_wake(
        self,
        wake_id: str,
    ) -> AgentRecallSubscriptionWake | None:
        wake_id = _bounded_identity(wake_id, "wake_id")
        async with self._lock:
            wake = self._subscription_wakes.get(wake_id)
            return None if wake is None else copy_agent_recall_subscription_wake(wake)

    async def renew_recall_subscription_wake(
        self,
        claim: AgentRecallSubscriptionWakeClaim,
        *,
        lease_seconds: float,
    ) -> AgentRecallSubscriptionWake:
        claim = copy_agent_recall_subscription_wake_claim(claim)
        async with self._lock:
            current = self._subscription_wakes.get(claim.wake_id)
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_wake")
            renewed = _renew_agent_recall_subscription_wake(
                current,
                claim,
                lease_seconds=lease_seconds,
                now=max(_utc(self._clock(), "clock result"), current.updated_at),
            )
            self._subscription_wakes[claim.wake_id] = renewed
            return copy_agent_recall_subscription_wake(renewed)

    async def release_recall_subscription_wake(
        self,
        claim: AgentRecallSubscriptionWakeClaim,
        *,
        release_id: str,
        reason: str,
        released_at: datetime,
    ) -> AgentRecallSubscriptionWake:
        claim = copy_agent_recall_subscription_wake_claim(claim)
        requested = _agent_recall_subscription_wake_release(
            claim,
            release_id=release_id,
            reason=reason,
            released_at=released_at,
        )
        async with self._lock:
            current = self._subscription_wakes.get(claim.wake_id)
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_wake")
            replay = self._subscription_wake_releases.get(requested.release_id)
            if replay is not None:
                if replay != requested:
                    raise AgentRecallSubscriptionConflict("wake_release_id_reused")
                if current.release != replay:
                    raise AgentRecallSubscriptionConflict("wake_release_replay_superseded")
                return copy_agent_recall_subscription_wake(current)
            released = _release_agent_recall_subscription_wake(
                current,
                claim,
                release_id=requested.release_id,
                reason=requested.reason,
                released_at=requested.released_at,
                now=max(_utc(self._clock(), "clock result"), current.updated_at),
            )
            self._subscription_wakes[claim.wake_id] = released
            self._subscription_wake_releases[requested.release_id] = requested
            return copy_agent_recall_subscription_wake(released)

    async def acknowledge_recall_subscription_wake(
        self,
        claim: AgentRecallSubscriptionWakeClaim,
        *,
        acknowledgement_id: str,
        acknowledged_at: datetime,
    ) -> AgentRecallSubscriptionWake:
        claim = copy_agent_recall_subscription_wake_claim(claim)
        acknowledgement_id = _bounded_identity(acknowledgement_id, "acknowledgement_id")
        async with self._lock:
            current = self._subscription_wakes.get(claim.wake_id)
            if current is None:
                raise AgentRecallSubscriptionConflict("unknown_wake")
            occupied = self._subscription_wake_acknowledgements.get(acknowledgement_id)
            if occupied is not None and occupied != claim.wake_id:
                raise AgentRecallSubscriptionConflict("wake_acknowledgement_id_reused")
            acknowledged = _acknowledge_agent_recall_subscription_wake(
                current,
                claim,
                acknowledgement_id=acknowledgement_id,
                acknowledged_at=acknowledged_at,
                now=max(_utc(self._clock(), "clock result"), current.updated_at),
            )
            self._subscription_wakes[claim.wake_id] = acknowledged
            self._subscription_wake_acknowledgements[acknowledgement_id] = claim.wake_id
            delivery_key = acknowledged.delivery.key().sort_key()
            authority_key = acknowledged.delivery.key().authority_sort_key()
            if current.state is not AgentRecallSubscriptionWakeState.ACKNOWLEDGED:
                self._delivery_queues.setdefault(delivery_key, deque()).append(
                    acknowledged.delivery.delivery_id
                )
                pending_wakes = self._pending_subscription_wake_ids[
                    acknowledged.subscription.subscription_id
                ]
                pending_wakes.remove(acknowledged.wake_id)
                if not pending_wakes:
                    self._pending_subscription_wake_ids.pop(
                        acknowledged.subscription.subscription_id,
                        None,
                    )
            self._trim_subscription_wake_queue_unlocked(authority_key)
            return copy_agent_recall_subscription_wake(acknowledged)

    def _subscription_has_pending_wake_unlocked(self, subscription_id: str) -> bool:
        return bool(self._pending_subscription_wake_ids.get(subscription_id))

    def _trim_subscription_wake_queue_unlocked(
        self,
        key: tuple[str, str, str, str],
    ) -> None:
        queue = self._subscription_wake_queues.get(key)
        if queue is None:
            return
        retained = deque(
            wake_id
            for wake_id in queue
            if self._subscription_wakes[wake_id].state
            is not AgentRecallSubscriptionWakeState.ACKNOWLEDGED
        )
        queue.clear()
        queue.extend(retained)
        if not queue:
            self._subscription_wake_queues.pop(key, None)

    def _trim_delivery_queue_unlocked(self, key: tuple[str, str, str, str, str]) -> None:
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
    "AGENT_RECALL_SUBSCRIPTION_CLAIM_SCHEMA_VERSION",
    "AGENT_RECALL_SUBSCRIPTION_EVALUATION_SCHEMA_VERSION",
    "AGENT_RECALL_SUBSCRIPTION_PUBLICATION_SCHEMA_VERSION",
    "AGENT_RECALL_SUBSCRIPTION_RECORD_SCHEMA_VERSION",
    "AGENT_RECALL_SUBSCRIPTION_RELEASE_SCHEMA_VERSION",
    "AGENT_RECALL_SUBSCRIPTION_SCHEMA_VERSION",
    "AGENT_RECALL_SUBSCRIPTION_WAKE_ACKNOWLEDGEMENT_SCHEMA_VERSION",
    "AGENT_RECALL_SUBSCRIPTION_WAKE_CLAIM_SCHEMA_VERSION",
    "AGENT_RECALL_SUBSCRIPTION_WAKE_RELEASE_SCHEMA_VERSION",
    "AGENT_RECALL_SUBSCRIPTION_WAKE_SCHEMA_VERSION",
    "AGENT_WORK_CONTEXT_PUBLICATION_SCHEMA_VERSION",
    "AGENT_WORK_CONTEXT_SCHEMA_VERSION",
    "DEFAULT_AGENT_RECALL_CHECKPOINT_STREAM_ID",
    "MAX_AGENT_RECALL_DELIVERY_BYTES",
    "MAX_AGENT_RECALL_DELIVERY_LEASE_SECONDS",
    "MAX_AGENT_RECALL_SUBSCRIPTION_BYTES",
    "MAX_AGENT_RECALL_SUBSCRIPTION_INTERVAL_SECONDS",
    "MAX_AGENT_RECALL_SUBSCRIPTION_PRIORITY",
    "MAX_AGENT_RECALL_SUBSCRIPTION_QUERY_BYTES",
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
    "AgentRecallSubscription",
    "AgentRecallSubscriptionClaim",
    "AgentRecallSubscriptionConflict",
    "AgentRecallSubscriptionEvaluation",
    "AgentRecallSubscriptionEvaluationOutcome",
    "AgentRecallSubscriptionPublicationReceipt",
    "AgentRecallSubscriptionRecord",
    "AgentRecallSubscriptionRelease",
    "AgentRecallSubscriptionRunState",
    "AgentRecallSubscriptionStatus",
    "AgentRecallSubscriptionWake",
    "AgentRecallSubscriptionWakeAcknowledgement",
    "AgentRecallSubscriptionWakeClaim",
    "AgentRecallSubscriptionWakeRelease",
    "AgentRecallSubscriptionWakeState",
    "AgentWorkContext",
    "AgentWorkContextConflict",
    "AgentWorkContextPublicationReceipt",
    "AgentWorkContextStore",
    "InMemoryAgentWorkContextStore",
    "agent_recall_facet_aspect",
]
