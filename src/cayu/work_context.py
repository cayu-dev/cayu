"""Durable work-context and recall-checkpoint contracts.

``AgentWorkContext`` is application/runtime-owned task state.  It is not prompt
text, workflow authority, or the ephemeral ``RecallSituation``.  A checkpoint
records only which bounded freshness frontier was processed against one exact
work context and access-policy identity; it is not provider-exposure evidence.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._clock import utc_clock
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
)

AGENT_WORK_CONTEXT_SCHEMA_VERSION = "cayu.agent_work_context.v1"
AGENT_WORK_CONTEXT_PUBLICATION_SCHEMA_VERSION = "cayu.agent_work_context_publication_receipt.v1"
AGENT_RECALL_CHECKPOINT_SCHEMA_VERSION = "cayu.agent_recall_checkpoint.v1"

MAX_AGENT_WORK_CONTEXT_REVISION = 2_147_483_647
MAX_AGENT_WORK_CONTEXT_ID_BYTES = 512
MAX_AGENT_WORK_CONTEXT_GOAL_BYTES = 32_000
MAX_AGENT_WORK_CONTEXT_VALUE_BYTES = 4_096
MAX_AGENT_WORK_CONTEXT_VALUES = 128
MAX_AGENT_WORK_CONTEXT_BYTES = 256_000

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
            task_contexts = self._contexts.get(checkpoint.task_id)
            work_context = (
                None
                if task_contexts is None
                else task_contexts.get(checkpoint.work_context_revision)
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
                None
                if revisions is None or current_revision is None
                else revisions[current_revision]
            )
            validate_agent_recall_checkpoint_advance(
                checkpoint,
                expected_revision,
                current,
            )
            self._checkpoint_revisions.setdefault(key, {})[checkpoint.revision] = checkpoint
            self._checkpoint_operations[checkpoint.operation_id] = (key, checkpoint.revision)
            self._checkpoint_heads[key] = checkpoint.revision
            return copy_agent_recall_checkpoint(checkpoint)

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
    "AGENT_WORK_CONTEXT_PUBLICATION_SCHEMA_VERSION",
    "AGENT_WORK_CONTEXT_SCHEMA_VERSION",
    "MAX_AGENT_WORK_CONTEXT_BYTES",
    "MAX_AGENT_WORK_CONTEXT_GOAL_BYTES",
    "MAX_AGENT_WORK_CONTEXT_ID_BYTES",
    "MAX_AGENT_WORK_CONTEXT_REVISION",
    "MAX_AGENT_WORK_CONTEXT_VALUES",
    "MAX_AGENT_WORK_CONTEXT_VALUE_BYTES",
    "AgentRecallCheckpoint",
    "AgentRecallCheckpointKey",
    "AgentRecallCheckpointMode",
    "AgentWorkContext",
    "AgentWorkContextConflict",
    "AgentWorkContextPublicationReceipt",
    "AgentWorkContextStore",
    "InMemoryAgentWorkContextStore",
]
