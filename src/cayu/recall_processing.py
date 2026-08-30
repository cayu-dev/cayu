"""Checkpoint-aware cross-agent knowledge recall.

The processor in this module evaluates a bounded freshness frontier and returns
an immutable checkpoint proposal. It never persists that proposal, injects
provider context, or claims that any candidate was exposed.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
)
from cayu.recall import (
    RECALL_MAX_LINEAGE_BYTES_PER_RECORD,
    RECALL_MAX_LINEAGE_CANDIDATES,
    RECALL_MAX_LINEAGE_LINKS_PER_RECORD,
    RECALL_MAX_RESULT_BYTES,
    RECALL_MAX_WORK_CONTEXT_BYTES,
    KnowledgeFrontierRecallSource,
    KnowledgeRecallSource,
    KnowledgeRevisionRecallSource,
    RecallEngine,
    RecallEngineConfig,
    RecallResult,
    RecallSituation,
    RecallSourceStatus,
)
from cayu.retrieval import WeightedReciprocalRankFusionConfig
from cayu.storage.memory import (
    MAX_KNOWLEDGE_RELATION_BYTES,
    MAX_KNOWLEDGE_REVISION_SEARCH_REFS,
    KnowledgeAccessScope,
    KnowledgeChange,
    KnowledgeChangeBatch,
    KnowledgeEmbeddingIdentity,
    KnowledgeIndexReadiness,
    KnowledgeIndexReadinessBatch,
    KnowledgeIndexState,
    KnowledgeRevisionRef,
    KnowledgeSearchMode,
    KnowledgeStore,
    copy_knowledge_access_scope,
    copy_knowledge_change,
    copy_knowledge_index_readiness,
    copy_knowledge_revision_refs,
    knowledge_access_scope_sha256,
)
from cayu.work_context import (
    MAX_AGENT_WORK_CONTEXT_ID_BYTES,
    AgentRecallCheckpoint,
    AgentRecallCheckpointMode,
    AgentWorkContext,
    copy_agent_recall_checkpoint,
    copy_agent_work_context,
)

AGENT_RECALL_PROCESSING_SCHEMA_VERSION = "cayu.agent_recall_processing.v3"


class _AgentRecallModel(BaseModel):
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


def _sequence(value: int, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_DURABLE_JSON_INTEGER:
        raise ValueError(f"`{field_name}` must be a non-negative durable JSON integer.")
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"`{field_name}` must be timezone-aware.")
    return value.astimezone(UTC)


class AgentRecallProcessingMode(StrEnum):
    FULL_INDEX = "full_index"
    DELTA = "delta"
    NO_WORK = "no_work"


class AgentRecallFrontier(_AgentRecallModel):
    """A caller-replayable pair of unchanged-context delta high-water marks."""

    knowledge_sequence: int
    index_readiness_sequence: int

    @field_validator("knowledge_sequence", "index_readiness_sequence")
    @classmethod
    def validate_sequence(cls, value: int, info) -> int:
        return _sequence(value, info.field_name)

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "agent recall frontier",
            )
        ).hexdigest()


class AgentRecallProcessorConfig(_AgentRecallModel):
    """Hard work and output bounds for one processor invocation."""

    knowledge_change_limit: int = 100
    index_readiness_limit: int = 100
    candidate_limit: int = 20
    source_max_bytes: int = 64_000
    max_record_bytes: int = 8_000
    semantic_timeout_seconds: float = 1.0
    lineage_limit: int = 0
    lineage_candidate_limit: int = 10
    lineage_max_bytes: int = 8_192

    @field_validator("knowledge_change_limit", "index_readiness_limit")
    @classmethod
    def validate_page_limit(cls, value: int, info) -> int:
        if type(value) is not int or not 1 <= value <= MAX_KNOWLEDGE_REVISION_SEARCH_REFS:
            raise ValueError(
                f"`{info.field_name}` must be between 1 and {MAX_KNOWLEDGE_REVISION_SEARCH_REFS}."
            )
        return value

    @field_validator("candidate_limit")
    @classmethod
    def validate_candidate_limit(cls, value: int) -> int:
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError("`candidate_limit` must be between 1 and 100.")
        return value

    @field_validator("source_max_bytes", "max_record_bytes")
    @classmethod
    def validate_result_byte_bound(cls, value: int, info) -> int:
        if type(value) is not int or not 1 <= value <= RECALL_MAX_RESULT_BYTES:
            raise ValueError(
                f"`{info.field_name}` must be between 1 and {RECALL_MAX_RESULT_BYTES}."
            )
        return value

    @field_validator("lineage_limit")
    @classmethod
    def validate_lineage_limit(cls, value: int) -> int:
        if type(value) is not int or not 0 <= value <= RECALL_MAX_LINEAGE_LINKS_PER_RECORD:
            raise ValueError(
                f"`lineage_limit` must be between 0 and {RECALL_MAX_LINEAGE_LINKS_PER_RECORD}."
            )
        return value

    @field_validator("lineage_candidate_limit")
    @classmethod
    def validate_lineage_candidate_limit(cls, value: int) -> int:
        if type(value) is not int or not 1 <= value <= RECALL_MAX_LINEAGE_CANDIDATES:
            raise ValueError(
                f"`lineage_candidate_limit` must be between 1 and {RECALL_MAX_LINEAGE_CANDIDATES}."
            )
        return value

    @field_validator("lineage_max_bytes")
    @classmethod
    def validate_lineage_max_bytes(cls, value: int) -> int:
        if (
            type(value) is not int
            or not MAX_KNOWLEDGE_RELATION_BYTES <= value <= RECALL_MAX_LINEAGE_BYTES_PER_RECORD
        ):
            raise ValueError(
                "`lineage_max_bytes` must be between "
                f"{MAX_KNOWLEDGE_RELATION_BYTES} and "
                f"{RECALL_MAX_LINEAGE_BYTES_PER_RECORD}."
            )
        return value

    @field_validator("semantic_timeout_seconds", mode="before")
    @classmethod
    def validate_semantic_timeout(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("`semantic_timeout_seconds` must be a number.")
        result = float(value)
        if not 0 < result <= 60:
            raise ValueError("`semantic_timeout_seconds` must be greater than 0 and at most 60.")
        return result

    @model_validator(mode="after")
    def validate_combined_revision_bound(self) -> AgentRecallProcessorConfig:
        if self.knowledge_change_limit + self.index_readiness_limit > (
            MAX_KNOWLEDGE_REVISION_SEARCH_REFS
        ):
            raise ValueError(
                "The combined change and readiness limits cannot exceed the exact-revision "
                "search bound."
            )
        if self.max_record_bytes > self.source_max_bytes:
            raise ValueError("`max_record_bytes` cannot exceed `source_max_bytes`.")
        if self.lineage_candidate_limit * self.lineage_max_bytes > RECALL_MAX_RESULT_BYTES:
            raise ValueError(
                "The lineage candidate and per-record byte bounds cannot reserve more than "
                f"{RECALL_MAX_RESULT_BYTES} bytes."
            )
        return self


class AgentRecallProcessingRequest(_AgentRecallModel):
    """One side-effect-free full-index or delta recall request."""

    agent_id: str
    work_context: AgentWorkContext
    situation: RecallSituation
    checkpoint_stream_id: str
    checkpoint: AgentRecallCheckpoint | None = None
    frontier: AgentRecallFrontier | None = None
    processing_id: str
    operation_id: str
    updated_by: str
    updated_at: datetime

    @field_validator(
        "agent_id",
        "checkpoint_stream_id",
        "processing_id",
        "operation_id",
        "updated_by",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("work_context", mode="before")
    @classmethod
    def copy_work_context(cls, value: object, info) -> AgentWorkContext:
        if type(value) is AgentWorkContext:
            return copy_agent_work_context(value)
        if info.mode == "json":
            return AgentWorkContext.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall work context")
            )
        return AgentWorkContext.model_validate(value)

    @field_validator("situation", mode="before")
    @classmethod
    def copy_situation(cls, value: object, info) -> RecallSituation:
        if type(value) is RecallSituation:
            return value.model_copy(deep=True)
        if info.mode == "json":
            return RecallSituation.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall situation")
            )
        return RecallSituation.model_validate(value)

    @field_validator("checkpoint", mode="before")
    @classmethod
    def copy_checkpoint(
        cls,
        value: object,
        info,
    ) -> AgentRecallCheckpoint | None:
        if value is None:
            return None
        if type(value) is AgentRecallCheckpoint:
            return copy_agent_recall_checkpoint(value)
        if info.mode == "json":
            return AgentRecallCheckpoint.model_validate_json(
                canonical_durable_json_bytes(value, "agent recall checkpoint")
            )
        return AgentRecallCheckpoint.model_validate(value)

    @field_validator("frontier", mode="before")
    @classmethod
    def copy_frontier(cls, value: object) -> AgentRecallFrontier | None:
        if value is None:
            return None
        if type(value) is AgentRecallFrontier:
            return value.model_copy(deep=True)
        return AgentRecallFrontier.model_validate(value)

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime) -> datetime:
        return _utc(value, "updated_at")

    @model_validator(mode="after")
    def validate_authority(self) -> AgentRecallProcessingRequest:
        scope = self.situation.knowledge_access_scope
        if scope is None:
            raise ValueError("Checkpoint-aware recall requires an explicit knowledge access scope.")
        if self.situation.work_context is not None:
            raise ValueError(
                "`situation.work_context` must be omitted; it is derived from `work_context`."
            )
        if scope.allow_all_namespaces or tuple(scope.allowed_namespaces) != (
            self.situation.knowledge_namespace,
        ):
            raise ValueError(
                "Checkpoint-aware recall requires an exact single-namespace access scope."
            )
        checkpoint = self.checkpoint
        access_policy_sha256 = knowledge_access_scope_sha256(scope)
        if checkpoint is not None:
            if (
                checkpoint.agent_id != self.agent_id
                or checkpoint.task_id != self.work_context.task_id
                or checkpoint.knowledge_namespace != self.situation.knowledge_namespace
                or checkpoint.access_policy_sha256 != access_policy_sha256
                or checkpoint.checkpoint_stream_id != self.checkpoint_stream_id
            ):
                raise ValueError("The checkpoint key conflicts with the recall request authority.")
            if checkpoint.work_context_revision > self.work_context.revision:
                raise ValueError("The checkpoint references a future work-context revision.")
            if (
                checkpoint.work_context_revision == self.work_context.revision
                and checkpoint.work_context_sha256 != self.work_context.content_sha256
            ):
                raise ValueError("The checkpoint conflicts with the current work-context identity.")
        if (
            self.frontier is not None
            and checkpoint is not None
            and (
                self.frontier.knowledge_sequence < checkpoint.knowledge_sequence
                or self.frontier.index_readiness_sequence < checkpoint.index_readiness_sequence
                or self.frontier.knowledge_sequence < checkpoint.knowledge_high_water_sequence
                or self.frontier.index_readiness_sequence
                < checkpoint.index_readiness_high_water_sequence
            )
        ):
            raise ValueError("A replay frontier cannot regress the checkpoint frontier.")
        if self.frontier is not None and (
            checkpoint is None
            or checkpoint.work_context_revision != self.work_context.revision
            or checkpoint.work_context_sha256 != self.work_context.content_sha256
        ):
            raise ValueError(
                "A replay frontier is valid only for an unchanged checkpoint work context."
            )
        return self


def agent_recall_situation_input_sha256(situation: RecallSituation) -> str:
    """Fingerprint retrieval-shaping input, excluding clock and access authority."""

    if type(situation) is not RecallSituation:
        raise TypeError("situation must be a RecallSituation.")
    copied = situation.model_copy(deep=True)
    return sha256(
        canonical_durable_json_bytes(
            copied.model_dump(
                mode="json",
                exclude={"current_time", "knowledge_access_scope"},
            ),
            "agent recall situation input",
        )
    ).hexdigest()


class AgentRecallProcessingResult(_AgentRecallModel):
    """Immutable retrieval outcome and optional, unapplied checkpoint proposal."""

    schema_version: Literal["cayu.agent_recall_processing.v3"] = (
        AGENT_RECALL_PROCESSING_SCHEMA_VERSION
    )
    mode: AgentRecallProcessingMode
    reason: str
    agent_id: str
    task_id: str
    knowledge_namespace: str
    work_context_revision: int
    work_context_sha256: str
    processing_id: str
    operation_id: str
    access_policy_sha256: str
    checkpoint_stream_id: str
    situation_sha256: str
    frontier: AgentRecallFrontier
    processed_frontier: AgentRecallFrontier
    knowledge_event_count: int
    index_readiness_event_count: int
    knowledge_events: tuple[KnowledgeChange, ...] = ()
    index_readiness_events: tuple[KnowledgeIndexReadiness, ...] = ()
    eligible_revisions: tuple[KnowledgeRevisionRef, ...] = ()
    recall: RecallResult | None = None
    proposed_checkpoint: AgentRecallCheckpoint | None = None
    frontier_complete: bool
    retry_required: bool = False

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: object) -> AgentRecallProcessingMode:
        if isinstance(value, AgentRecallProcessingMode):
            return value
        if type(value) is str:
            return AgentRecallProcessingMode(value)
        raise ValueError("`mode` must be an AgentRecallProcessingMode.")

    @field_validator(
        "reason",
        "agent_id",
        "task_id",
        "knowledge_namespace",
        "checkpoint_stream_id",
        "processing_id",
        "operation_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_identity(value, info.field_name)

    @field_validator("access_policy_sha256", "situation_sha256")
    @classmethod
    def validate_sha256(cls, value: str, info) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"`{info.field_name}` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("work_context_revision")
    @classmethod
    def validate_work_context_revision(cls, value: int) -> int:
        if type(value) is not int or not 1 <= value <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError("`work_context_revision` must be a positive durable JSON integer.")
        return value

    @field_validator("work_context_sha256")
    @classmethod
    def validate_work_context_sha256(cls, value: str) -> str:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("`work_context_sha256` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("knowledge_event_count", "index_readiness_event_count")
    @classmethod
    def validate_event_count(cls, value: int, info) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"`{info.field_name}` must be a non-negative integer.")
        return value

    @field_validator("knowledge_events", mode="before")
    @classmethod
    def copy_knowledge_events(cls, value) -> tuple[KnowledgeChange, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, (list, tuple)):
            raise ValueError("`knowledge_events` must be a sequence.")
        return tuple(
            copy_knowledge_change(
                item if type(item) is KnowledgeChange else KnowledgeChange.model_validate(item)
            )
            for item in value
        )

    @field_validator("index_readiness_events", mode="before")
    @classmethod
    def copy_index_readiness_events(cls, value) -> tuple[KnowledgeIndexReadiness, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, (list, tuple)):
            raise ValueError("`index_readiness_events` must be a sequence.")
        copied: list[KnowledgeIndexReadiness] = []
        for item in value:
            if type(item) is KnowledgeIndexReadiness:
                readiness = item
            else:
                if type(item) is not dict:
                    raise ValueError("`index_readiness_events` must contain readiness records.")
                payload = dict(item)
                identity = payload.get("identity")
                if type(identity) is not KnowledgeEmbeddingIdentity:
                    payload["identity"] = KnowledgeEmbeddingIdentity.model_validate(identity)
                readiness = KnowledgeIndexReadiness.model_validate(payload)
            copied.append(copy_knowledge_index_readiness(readiness))
        return tuple(copied)

    @field_validator("eligible_revisions", mode="before")
    @classmethod
    def copy_revisions(cls, value) -> tuple[KnowledgeRevisionRef, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, (list, tuple)):
            raise ValueError("`eligible_revisions` must be a sequence.")
        return copy_knowledge_revision_refs(
            tuple(
                item
                if type(item) is KnowledgeRevisionRef
                else KnowledgeRevisionRef.model_validate(item)
                for item in value
            )
        )

    @field_validator("frontier_complete", "retry_required", mode="before")
    @classmethod
    def validate_boolean(cls, value: object, info) -> bool:
        if type(value) is not bool:
            raise ValueError(f"`{info.field_name}` must be a boolean.")
        return value

    @model_validator(mode="after")
    def validate_result_shape(self) -> AgentRecallProcessingResult:
        if self.knowledge_event_count != len(self.knowledge_events):
            raise ValueError("`knowledge_event_count` must match `knowledge_events`.")
        if self.index_readiness_event_count != len(self.index_readiness_events):
            raise ValueError("`index_readiness_event_count` must match `index_readiness_events`.")
        if self.knowledge_event_count + self.index_readiness_event_count > (
            MAX_KNOWLEDGE_REVISION_SEARCH_REFS
        ):
            raise ValueError("Processing evidence exceeds the exact-revision search bound.")
        knowledge_sequences = [event.sequence for event in self.knowledge_events]
        readiness_sequences = [event.sequence for event in self.index_readiness_events]
        if knowledge_sequences != sorted(set(knowledge_sequences)):
            raise ValueError("Knowledge processing evidence must have unique ascending sequences.")
        if readiness_sequences != sorted(set(readiness_sequences)):
            raise ValueError("Readiness processing evidence must have unique ascending sequences.")
        if knowledge_sequences and knowledge_sequences[-1] > self.frontier.knowledge_sequence:
            raise ValueError("Knowledge processing evidence exceeds its captured frontier.")
        if readiness_sequences and readiness_sequences[-1] > self.frontier.index_readiness_sequence:
            raise ValueError("Readiness processing evidence exceeds its captured frontier.")
        if (
            self.processed_frontier.knowledge_sequence > self.frontier.knowledge_sequence
            or self.processed_frontier.index_readiness_sequence
            > self.frontier.index_readiness_sequence
        ):
            raise ValueError("The processed frontier cannot exceed the captured frontier.")
        expected_complete = self.processed_frontier == self.frontier
        if self.frontier_complete != expected_complete:
            raise ValueError("`frontier_complete` must reflect the processed frontier.")
        if self.mode is AgentRecallProcessingMode.NO_WORK:
            if self.recall is not None or self.proposed_checkpoint is not None:
                raise ValueError("No-work results cannot retrieve or propose checkpoint state.")
            if (
                self.eligible_revisions
                or self.knowledge_events
                or self.index_readiness_events
                or (self.knowledge_event_count or self.index_readiness_event_count)
            ):
                raise ValueError("No-work results cannot contain delta eligibility evidence.")
            if not self.frontier_complete or self.retry_required:
                raise ValueError("No-work results must be complete and cannot require retry.")
        elif self.recall is None:
            raise ValueError("Full-index and delta results require a recall result.")
        semantic_retry_evidence = False
        if self.recall is not None:
            knowledge_diagnostic = next(
                (source for source in self.recall.sources if source.source == "knowledge"),
                None,
            )
            if knowledge_diagnostic is None:
                raise ValueError("Processing recall evidence requires a knowledge diagnostic.")
            if knowledge_diagnostic.failure_code is not None:
                semantic_retry_evidence = bool(
                    set(knowledge_diagnostic.failure_code.split("+"))
                    & {"semantic_failed", "semantic_timeout"}
                )
        if self.retry_required != semantic_retry_evidence:
            raise ValueError("`retry_required` must reflect semantic retry evidence.")
        if self.mode is AgentRecallProcessingMode.FULL_INDEX and (
            self.eligible_revisions
            or self.knowledge_events
            or self.index_readiness_events
            or self.knowledge_event_count
            or self.index_readiness_event_count
        ):
            raise ValueError("Full-index results cannot contain delta eligibility evidence.")
        if (
            self.mode is not AgentRecallProcessingMode.NO_WORK
            and self.proposed_checkpoint is None
            and not self.retry_required
        ):
            raise ValueError("Completed recall work requires a checkpoint proposal.")
        if self.retry_required and self.proposed_checkpoint is not None:
            replayable_semantic_revisions = {
                (event.identity.entry_id, event.identity.entry_revision)
                for event in self.index_readiness_events
                if event.state is KnowledgeIndexState.READY
                and event.sequence > self.proposed_checkpoint.index_readiness_sequence
            }
            eligible_revision_keys = {
                (reference.entry_id, reference.revision) for reference in self.eligible_revisions
            }
            if (
                self.mode is AgentRecallProcessingMode.FULL_INDEX
                or not eligible_revision_keys
                or not eligible_revision_keys <= replayable_semantic_revisions
            ):
                raise ValueError(
                    "Retrieval retry progress requires retained READY evidence for every "
                    "eligible revision."
                )
        expected_revisions = copy_knowledge_revision_refs(
            tuple(
                KnowledgeRevisionRef(
                    entry_id=event.entry_id,
                    revision=event.entry_revision,
                )
                for event in self.knowledge_events
            )
            + tuple(
                KnowledgeRevisionRef(
                    entry_id=event.identity.entry_id,
                    revision=event.identity.entry_revision,
                )
                for event in self.index_readiness_events
                if event.state is KnowledgeIndexState.READY
            )
        )
        if self.eligible_revisions != expected_revisions:
            raise ValueError("Eligible revisions conflict with their source-event evidence.")
        checkpoint = self.proposed_checkpoint
        if checkpoint is not None:
            expected_mode = (
                AgentRecallCheckpointMode.FULL_INDEX
                if self.mode is AgentRecallProcessingMode.FULL_INDEX
                else AgentRecallCheckpointMode.DELTA
            )
            if (
                checkpoint.processing_mode is not expected_mode
                or checkpoint.agent_id != self.agent_id
                or checkpoint.task_id != self.task_id
                or checkpoint.knowledge_namespace != self.knowledge_namespace
                or checkpoint.work_context_revision != self.work_context_revision
                or checkpoint.work_context_sha256 != self.work_context_sha256
                or checkpoint.processing_id != self.processing_id
                or checkpoint.operation_id != self.operation_id
                or checkpoint.access_policy_sha256 != self.access_policy_sha256
                or checkpoint.checkpoint_stream_id != self.checkpoint_stream_id
                or checkpoint.knowledge_high_water_sequence != self.frontier.knowledge_sequence
                or checkpoint.index_readiness_high_water_sequence
                != self.frontier.index_readiness_sequence
            ):
                raise ValueError("The checkpoint proposal conflicts with processing evidence.")
            if (
                checkpoint.knowledge_sequence != self.processed_frontier.knowledge_sequence
                or checkpoint.index_readiness_sequence
                != self.processed_frontier.index_readiness_sequence
            ):
                raise ValueError("Checkpoint progress conflicts with the processed frontier.")
        return self

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "agent recall processing result",
            )
        ).hexdigest()


class AgentRecallProcessingError(RuntimeError):
    """A required freshness source could not be evaluated safely."""

    def __init__(self, code: str) -> None:
        self.code = require_durable_clean_nonblank(code, "code")
        super().__init__(f"Agent recall processing failed ({self.code}).")


def agent_work_context_recall_text(context: AgentWorkContext) -> str:
    """Render deterministic bounded retrieval text from durable work context."""

    context = copy_agent_work_context(context)
    lines = [f"goal: {context.goal}"]
    if context.workflow_id is not None:
        lines.append(f"workflow: {context.workflow_id}")
    if context.workflow_phase is not None:
        lines.append(f"workflow phase: {context.workflow_phase}")
    if context.workflow_iteration is not None:
        lines.append(f"workflow iteration: {context.workflow_iteration}")
    collections = (
        ("scope", context.scope_ids),
        ("entity", context.entity_ids),
        ("artifact", context.artifact_ids),
        ("repository path", context.repository_paths),
        ("code symbol", context.code_symbols),
        ("planned action", context.planned_action_ids),
    )
    lines.extend(f"{label}: {item}" for label, values in collections for item in values)
    rendered = "\n".join(lines)
    encoded = rendered.encode("utf-8")
    if len(encoded) <= RECALL_MAX_WORK_CONTEXT_BYTES:
        return rendered
    return encoded[:RECALL_MAX_WORK_CONTEXT_BYTES].decode("utf-8", errors="ignore")


class AgentRecallProcessor:
    """Select and evaluate a full-index, delta, or explicit no-work recall lane."""

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        *,
        fusion_config: WeightedReciprocalRankFusionConfig,
        engine_config: RecallEngineConfig | None = None,
        config: AgentRecallProcessorConfig | None = None,
    ) -> None:
        if not isinstance(knowledge_store, KnowledgeStore):
            raise TypeError("knowledge_store must be a KnowledgeStore.")
        if type(fusion_config) is not WeightedReciprocalRankFusionConfig:
            raise TypeError("fusion_config must be a WeightedReciprocalRankFusionConfig.")
        self._store = knowledge_store
        self._fusion_config = fusion_config.model_copy(deep=True)
        self._engine_config = (
            RecallEngineConfig()
            if engine_config is None
            else RecallEngineConfig.model_validate(engine_config.model_dump(mode="python"))
        )
        self._config = (
            AgentRecallProcessorConfig() if config is None else config.model_copy(deep=True)
        )
        RecallEngine(
            (self._knowledge_source(),),
            fusion_config=self._fusion_config,
            config=self._engine_config,
        )

    async def process(
        self,
        request: AgentRecallProcessingRequest,
    ) -> AgentRecallProcessingResult:
        if type(request) is not AgentRecallProcessingRequest:
            raise TypeError("request must be an AgentRecallProcessingRequest.")
        request = request.model_copy(deep=True)
        checkpoint = request.checkpoint
        context_changed = checkpoint is None or (
            checkpoint.work_context_revision != request.work_context.revision
            or checkpoint.work_context_sha256 != request.work_context.content_sha256
        )
        knowledge_after = 0 if checkpoint is None else checkpoint.knowledge_sequence
        readiness_after = 0 if checkpoint is None else checkpoint.index_readiness_sequence
        scope = request.situation.knowledge_access_scope
        assert scope is not None
        scope = copy_knowledge_access_scope(scope)

        try:
            raw_changes = await self._store.read_changes(
                after_sequence=(0 if context_changed else knowledge_after),
                limit=(1 if context_changed else self._config.knowledge_change_limit),
                access_scope=scope,
            )
        except NotImplementedError as exc:
            raise AgentRecallProcessingError("knowledge_changes_unavailable") from exc
        except Exception as exc:
            raise AgentRecallProcessingError("knowledge_changes_failed") from exc
        if type(raw_changes) is not KnowledgeChangeBatch:
            raise TypeError("KnowledgeStore.read_changes() must return a KnowledgeChangeBatch.")
        changes = raw_changes.model_copy(deep=True)
        try:
            raw_readiness = await self._store.read_index_readiness(
                after_sequence=(0 if context_changed else readiness_after),
                limit=(1 if context_changed else self._config.index_readiness_limit),
                access_scope=scope,
            )
        except NotImplementedError as exc:
            raise AgentRecallProcessingError("index_readiness_unavailable") from exc
        except Exception as exc:
            raise AgentRecallProcessingError("index_readiness_failed") from exc
        if type(raw_readiness) is not KnowledgeIndexReadinessBatch:
            raise TypeError(
                "KnowledgeStore.read_index_readiness() must return a KnowledgeIndexReadinessBatch."
            )
        readiness = raw_readiness.model_copy(deep=True)

        frontier = self._resolve_frontier(
            request.frontier,
            changes,
            readiness,
            minimum=AgentRecallFrontier(
                knowledge_sequence=(
                    0 if checkpoint is None else checkpoint.knowledge_high_water_sequence
                ),
                index_readiness_sequence=(
                    0 if checkpoint is None else checkpoint.index_readiness_high_water_sequence
                ),
            ),
        )
        if checkpoint is not None and (
            frontier.knowledge_sequence < checkpoint.knowledge_high_water_sequence
            or frontier.index_readiness_sequence < checkpoint.index_readiness_high_water_sequence
        ):
            raise AgentRecallProcessingError("captured_frontier_regression")

        if not context_changed and (
            frontier.knowledge_sequence == knowledge_after
            and frontier.index_readiness_sequence == readiness_after
        ):
            return AgentRecallProcessingResult(
                mode=AgentRecallProcessingMode.NO_WORK,
                reason="work_context_and_freshness_unchanged",
                agent_id=request.agent_id,
                task_id=request.work_context.task_id,
                knowledge_namespace=request.situation.knowledge_namespace,
                work_context_revision=request.work_context.revision,
                work_context_sha256=request.work_context.content_sha256,
                processing_id=request.processing_id,
                operation_id=request.operation_id,
                access_policy_sha256=knowledge_access_scope_sha256(scope),
                checkpoint_stream_id=request.checkpoint_stream_id,
                situation_sha256=agent_recall_situation_input_sha256(request.situation),
                frontier=frontier,
                processed_frontier=frontier,
                knowledge_event_count=0,
                index_readiness_event_count=0,
                frontier_complete=True,
            )

        if context_changed:
            mode = AgentRecallProcessingMode.FULL_INDEX
            eligible_revisions: tuple[KnowledgeRevisionRef, ...] = ()
            selected_changes: tuple[KnowledgeChange, ...] = ()
            selected_readiness: tuple[KnowledgeIndexReadiness, ...] = ()
            knowledge_events = 0
            readiness_events = 0
            knowledge_processed = frontier.knowledge_sequence
            readiness_processed = frontier.index_readiness_sequence
            source: KnowledgeRecallSource = self._knowledge_source(frontier=frontier)
            reason = "initial_checkpoint" if checkpoint is None else "work_context_changed"
        else:
            mode = AgentRecallProcessingMode.DELTA
            selected_changes = tuple(
                change
                for change in changes.changes
                if change.sequence <= frontier.knowledge_sequence
            )
            selected_readiness = tuple(
                event
                for event in readiness.readiness
                if event.sequence <= frontier.index_readiness_sequence
            )
            eligible_revisions = copy_knowledge_revision_refs(
                tuple(
                    KnowledgeRevisionRef(
                        entry_id=change.entry_id,
                        revision=change.entry_revision,
                    )
                    for change in selected_changes
                )
                + tuple(
                    KnowledgeRevisionRef(
                        entry_id=event.identity.entry_id,
                        revision=event.identity.entry_revision,
                    )
                    for event in selected_readiness
                    if event.state is KnowledgeIndexState.READY
                )
            )
            knowledge_events = len(selected_changes)
            readiness_events = len(selected_readiness)
            knowledge_processed = self._processed_sequence(
                after_sequence=knowledge_after,
                target_sequence=frontier.knowledge_sequence,
                next_after_sequence=changes.next_after_sequence,
                truncated=changes.truncated,
            )
            readiness_processed = self._processed_sequence(
                after_sequence=readiness_after,
                target_sequence=frontier.index_readiness_sequence,
                next_after_sequence=readiness.next_after_sequence,
                truncated=readiness.truncated,
            )
            source = self._knowledge_source(eligible_revisions, frontier=frontier)
            reason = "freshness_frontier_advanced"

        situation = request.situation.model_copy(
            update={"work_context": agent_work_context_recall_text(request.work_context)},
            deep=True,
        )
        recall = await RecallEngine(
            (source,),
            fusion_config=self._fusion_config,
            config=self._engine_config,
        ).recall(situation)

        semantic_retry = self._semantic_retry_required(recall)
        if semantic_retry:
            readiness_processed = readiness_after
        processed_frontier = AgentRecallFrontier(
            knowledge_sequence=knowledge_processed,
            index_readiness_sequence=readiness_processed,
        )
        frontier_complete = processed_frontier == frontier
        retry_required = semantic_retry
        replayable_semantic_revisions = {
            (event.identity.entry_id, event.identity.entry_revision)
            for event in selected_readiness
            if event.state is KnowledgeIndexState.READY
        }
        eligible_revision_keys = {
            (reference.entry_id, reference.revision) for reference in eligible_revisions
        }
        semantic_retry_can_advance = (
            semantic_retry
            and mode is AgentRecallProcessingMode.DELTA
            and bool(eligible_revision_keys)
            and eligible_revision_keys <= replayable_semantic_revisions
        )
        proposed_checkpoint = (
            None
            if semantic_retry and not semantic_retry_can_advance
            else self._proposed_checkpoint(
                request,
                mode=mode,
                scope=scope,
                frontier=frontier,
                knowledge_processed=knowledge_processed,
                readiness_processed=readiness_processed,
            )
        )
        if (
            mode is AgentRecallProcessingMode.DELTA
            and checkpoint is not None
            and proposed_checkpoint is not None
            and (
                proposed_checkpoint.knowledge_sequence == checkpoint.knowledge_sequence
                and proposed_checkpoint.index_readiness_sequence
                == checkpoint.index_readiness_sequence
            )
        ):
            proposed_checkpoint = None

        return AgentRecallProcessingResult(
            mode=mode,
            reason=reason,
            agent_id=request.agent_id,
            task_id=request.work_context.task_id,
            knowledge_namespace=request.situation.knowledge_namespace,
            work_context_revision=request.work_context.revision,
            work_context_sha256=request.work_context.content_sha256,
            processing_id=request.processing_id,
            operation_id=request.operation_id,
            access_policy_sha256=knowledge_access_scope_sha256(scope),
            checkpoint_stream_id=request.checkpoint_stream_id,
            situation_sha256=agent_recall_situation_input_sha256(request.situation),
            frontier=frontier,
            processed_frontier=processed_frontier,
            knowledge_event_count=knowledge_events,
            index_readiness_event_count=readiness_events,
            knowledge_events=selected_changes,
            index_readiness_events=selected_readiness,
            eligible_revisions=eligible_revisions,
            recall=recall,
            proposed_checkpoint=proposed_checkpoint,
            frontier_complete=frontier_complete,
            retry_required=retry_required,
        )

    def _knowledge_source(
        self,
        revisions: tuple[KnowledgeRevisionRef, ...] | None = None,
        *,
        frontier: AgentRecallFrontier | None = None,
    ) -> KnowledgeRecallSource:
        if revisions is None and frontier is not None:
            return KnowledgeFrontierRecallSource(
                self._store,
                knowledge_sequence=frontier.knowledge_sequence,
                index_readiness_sequence=frontier.index_readiness_sequence,
                candidate_limit=self._config.candidate_limit,
                max_bytes=self._config.source_max_bytes,
                max_record_bytes=self._config.max_record_bytes,
                semantic_timeout_seconds=self._config.semantic_timeout_seconds,
                lineage_limit=self._config.lineage_limit,
                lineage_candidate_limit=self._config.lineage_candidate_limit,
                lineage_max_bytes=self._config.lineage_max_bytes,
            )
        if revisions is None:
            return KnowledgeRecallSource(
                self._store,
                candidate_limit=self._config.candidate_limit,
                max_bytes=self._config.source_max_bytes,
                max_record_bytes=self._config.max_record_bytes,
                semantic_timeout_seconds=self._config.semantic_timeout_seconds,
                lineage_limit=self._config.lineage_limit,
                lineage_candidate_limit=self._config.lineage_candidate_limit,
                lineage_max_bytes=self._config.lineage_max_bytes,
            )
        return KnowledgeRevisionRecallSource(
            self._store,
            revisions,
            knowledge_sequence=(None if frontier is None else frontier.knowledge_sequence),
            index_readiness_sequence=(
                None if frontier is None else frontier.index_readiness_sequence
            ),
            candidate_limit=self._config.candidate_limit,
            max_bytes=self._config.source_max_bytes,
            max_record_bytes=self._config.max_record_bytes,
            semantic_timeout_seconds=self._config.semantic_timeout_seconds,
            lineage_limit=self._config.lineage_limit,
            lineage_candidate_limit=self._config.lineage_candidate_limit,
            lineage_max_bytes=self._config.lineage_max_bytes,
        )

    @staticmethod
    def _resolve_frontier(
        requested: AgentRecallFrontier | None,
        changes: KnowledgeChangeBatch,
        readiness: KnowledgeIndexReadinessBatch,
        *,
        minimum: AgentRecallFrontier,
    ) -> AgentRecallFrontier:
        observed = AgentRecallFrontier(
            knowledge_sequence=max(
                changes.high_water_sequence,
                minimum.knowledge_sequence,
            ),
            index_readiness_sequence=max(
                readiness.high_water_sequence,
                minimum.index_readiness_sequence,
            ),
        )
        if requested is None:
            return observed
        if (
            requested.knowledge_sequence > observed.knowledge_sequence
            or requested.index_readiness_sequence > observed.index_readiness_sequence
        ):
            raise AgentRecallProcessingError("replay_frontier_not_observable")
        return requested.model_copy(deep=True)

    @staticmethod
    def _processed_sequence(
        *,
        after_sequence: int,
        target_sequence: int,
        next_after_sequence: int,
        truncated: bool,
    ) -> int:
        if target_sequence <= after_sequence:
            return after_sequence
        if truncated and next_after_sequence < target_sequence:
            return next_after_sequence
        return target_sequence

    def _proposed_checkpoint(
        self,
        request: AgentRecallProcessingRequest,
        *,
        mode: AgentRecallProcessingMode,
        scope: KnowledgeAccessScope,
        frontier: AgentRecallFrontier,
        knowledge_processed: int,
        readiness_processed: int,
    ) -> AgentRecallCheckpoint:
        checkpoint = request.checkpoint
        return AgentRecallCheckpoint(
            agent_id=request.agent_id,
            task_id=request.work_context.task_id,
            knowledge_namespace=request.situation.knowledge_namespace,
            access_policy_sha256=knowledge_access_scope_sha256(scope),
            checkpoint_stream_id=request.checkpoint_stream_id,
            revision=1 if checkpoint is None else checkpoint.revision + 1,
            work_context_revision=request.work_context.revision,
            work_context_sha256=request.work_context.content_sha256,
            knowledge_sequence=knowledge_processed,
            index_readiness_sequence=readiness_processed,
            knowledge_high_water_sequence=frontier.knowledge_sequence,
            index_readiness_high_water_sequence=frontier.index_readiness_sequence,
            processing_mode=(
                AgentRecallCheckpointMode.FULL_INDEX
                if mode is AgentRecallProcessingMode.FULL_INDEX
                else AgentRecallCheckpointMode.DELTA
            ),
            processing_id=request.processing_id,
            operation_id=request.operation_id,
            updated_by=request.updated_by,
            updated_at=request.updated_at,
        )

    def _semantic_retry_required(self, recall: RecallResult) -> bool:
        if KnowledgeSearchMode.SEMANTIC not in self._store.supported_search_modes():
            return False
        knowledge = next(source for source in recall.sources if source.source == "knowledge")
        if knowledge.status is RecallSourceStatus.COMPLETE or knowledge.failure_code is None:
            return False
        reasons = set(knowledge.failure_code.split("+"))
        return bool(reasons & {"semantic_failed", "semantic_timeout"})


__all__ = [
    "AGENT_RECALL_PROCESSING_SCHEMA_VERSION",
    "AgentRecallFrontier",
    "AgentRecallProcessingError",
    "AgentRecallProcessingMode",
    "AgentRecallProcessingRequest",
    "AgentRecallProcessingResult",
    "AgentRecallProcessor",
    "AgentRecallProcessorConfig",
    "agent_work_context_recall_text",
]
