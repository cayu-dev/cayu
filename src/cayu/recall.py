from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from itertools import islice
from typing import Any, ClassVar, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from cayu._validation import (
    FrozenJsonDict,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_json_value,
    freeze_json_value,
    require_finite,
    thaw_json_value,
)
from cayu._validation import (
    require_durable_clean_nonblank as require_clean_nonblank,
)
from cayu._validation import require_durable_nonblank as require_nonblank
from cayu.retrieval import (
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    FusedRetrievalCandidate,
    RankedRetrievalChannel,
    RankedRetrievalHit,
    RetrievalCandidateIdentity,
    RetrievalFusionDiagnostics,
    RetrievalFusionResult,
    RetrievalFusionStrategy,
    WeightedReciprocalRankFusion,
    WeightedReciprocalRankFusionConfig,
)
from cayu.runtime.sessions import (
    MAX_SESSION_ID_BYTES,
    TRANSCRIPT_SEARCH_INDEX_VERSION,
    TRANSCRIPT_SEARCH_MAX_BYTES,
    TRANSCRIPT_SEARCH_MAX_SCAN_LIMIT,
    TRANSCRIPT_SEARCH_MIN_MAX_BYTES,
    SessionStore,
    TranscriptSearchQuery,
    encode_transcript_search_cursor,
)
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_NAMESPACE,
    KnowledgeAccessScope,
    KnowledgeHit,
    KnowledgeIndexCoverage,
    KnowledgeQuery,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStore,
    copy_knowledge_access_scope,
)

RECALL_ENGINE_VERSION = "cayu.recall.v1"
KNOWLEDGE_LEXICAL_CHANNEL = "knowledge.lexical"
KNOWLEDGE_SEMANTIC_CHANNEL = "knowledge.semantic"
TRANSCRIPT_LEXICAL_CHANNEL = "transcript.lexical"
RECALL_MAX_RECENT_CONVERSATION_ITEMS = 20
RECALL_MAX_RECENT_CONVERSATION_BYTES = 32_000
RECALL_MAX_WORK_CONTEXT_BYTES = 32_000
RECALL_MAX_QUERY_BYTES = 8_192
RECALL_MAX_RESULT_BYTES = 1_000_000
RECALL_MAX_CONTINUATIONS = 100
RECALL_MAX_CONTINUATION_BYTES = 4_096
_RECALL_MAX_SOURCES = 32
_RECALL_MAX_CHANNELS = 100
_RECALL_MAX_NAME_BYTES = 256
_SHORT_FOLLOWUP_MAX_TERMS = 4


class RecallSourceStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class RecallSourceUnavailable(RuntimeError):
    """A required source could not produce an access-safe bounded result."""

    def __init__(self, source: str, code: str) -> None:
        self.source = require_clean_nonblank(source, "source")
        self.code = require_clean_nonblank(code, "code")
        super().__init__(f"Required recall source {self.source!r} is unavailable ({self.code}).")


class RecallSituation(BaseModel):
    """Immutable per-boundary retrieval input; never provider-visible by itself."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    query: str
    recent_conversation: tuple[str, ...] = ()
    work_context: str | None = None
    knowledge_access_scope: KnowledgeAccessScope | None = None
    knowledge_namespace: str = DEFAULT_KNOWLEDGE_NAMESPACE
    transcript_session_ids: tuple[str, ...] = ()
    continuations: Mapping[str, str] = Field(default_factory=dict)
    current_time: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = require_nonblank(value, "query")
        if len(value.encode("utf-8")) > RECALL_MAX_QUERY_BYTES:
            raise ValueError(f"`query` must be at most {RECALL_MAX_QUERY_BYTES} UTF-8 bytes.")
        return value

    @field_validator("recent_conversation", mode="before")
    @classmethod
    def validate_recent_conversation(cls, value) -> tuple[str, ...]:
        if isinstance(value, str | bytes):
            raise ValueError("`recent_conversation` must be a sequence of strings.")
        try:
            items = list(islice(value, RECALL_MAX_RECENT_CONVERSATION_ITEMS + 1))
        except TypeError as exc:
            raise ValueError("`recent_conversation` must be a sequence of strings.") from exc
        if len(items) > RECALL_MAX_RECENT_CONVERSATION_ITEMS:
            raise ValueError("`recent_conversation` exceeds its interaction-item bound.")
        result: list[str] = []
        total_bytes = 0
        for index, item in enumerate(items):
            if type(item) is not str:
                raise ValueError(f"`recent_conversation[{index}]` must be a string.")
            text = require_nonblank(item, f"recent_conversation[{index}]")
            total_bytes += len(text.encode("utf-8"))
            if total_bytes > RECALL_MAX_RECENT_CONVERSATION_BYTES:
                raise ValueError("`recent_conversation` exceeds its UTF-8 byte bound.")
            result.append(text)
        return tuple(result)

    @field_validator("work_context")
    @classmethod
    def validate_work_context(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_nonblank(value, "work_context")
        if len(value.encode("utf-8")) > RECALL_MAX_WORK_CONTEXT_BYTES:
            raise ValueError(
                f"`work_context` must be at most {RECALL_MAX_WORK_CONTEXT_BYTES} UTF-8 bytes."
            )
        return value

    @field_validator("knowledge_access_scope", mode="before")
    @classmethod
    def copy_access_scope(cls, value) -> KnowledgeAccessScope | None:
        if value is None:
            return None
        if type(value) is KnowledgeAccessScope:
            return copy_knowledge_access_scope(value)
        return KnowledgeAccessScope.model_validate(value)

    @field_validator("knowledge_access_scope")
    @classmethod
    def freeze_access_scope(
        cls,
        value: KnowledgeAccessScope | None,
    ) -> KnowledgeAccessScope | None:
        if value is None:
            return None
        object.__setattr__(value, "allowed_namespaces", tuple(value.allowed_namespaces))
        object.__setattr__(value, "required_labels", FrozenJsonDict(value.required_labels))
        object.__setattr__(value, "allowed_visibilities", tuple(value.allowed_visibilities))
        object.__setattr__(
            value,
            "allowed_source_types",
            None if value.allowed_source_types is None else tuple(value.allowed_source_types),
        )
        object.__setattr__(
            value,
            "allowed_source_ids",
            None if value.allowed_source_ids is None else tuple(value.allowed_source_ids),
        )
        object.__setattr__(value, "allowed_statuses", tuple(value.allowed_statuses))
        return value

    @field_serializer("knowledge_access_scope")
    def serialize_access_scope(
        self,
        value: KnowledgeAccessScope | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "allowed_namespaces": list(value.allowed_namespaces),
            "allow_all_namespaces": value.allow_all_namespaces,
            "required_labels": dict(value.required_labels),
            "allowed_visibilities": [str(item) for item in value.allowed_visibilities],
            "allowed_source_types": (
                None if value.allowed_source_types is None else list(value.allowed_source_types)
            ),
            "allowed_source_ids": (
                None if value.allowed_source_ids is None else list(value.allowed_source_ids)
            ),
            "allowed_statuses": [str(item) for item in value.allowed_statuses],
            "include_expired": value.include_expired,
        }

    @field_validator("knowledge_namespace")
    @classmethod
    def validate_namespace(cls, value: str) -> str:
        return require_clean_nonblank(value, "knowledge_namespace")

    @field_validator("transcript_session_ids", mode="before")
    @classmethod
    def validate_transcript_session_ids(cls, value) -> tuple[str, ...]:
        if isinstance(value, str | bytes):
            raise ValueError("`transcript_session_ids` must be a sequence.")
        try:
            items = list(islice(value, 101))
        except TypeError as exc:
            raise ValueError("`transcript_session_ids` must be a sequence.") from exc
        if len(items) > 100:
            raise ValueError("`transcript_session_ids` cannot contain more than 100 ids.")
        cleaned: list[str] = []
        for item in items:
            session_id = require_clean_nonblank(item, "transcript_session_ids")
            if len(session_id.encode("utf-8")) > MAX_SESSION_ID_BYTES:
                raise ValueError(
                    f"Transcript session ids must be at most {MAX_SESSION_ID_BYTES} UTF-8 bytes."
                )
            cleaned.append(session_id)
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("`transcript_session_ids` cannot contain duplicates.")
        return tuple(sorted(cleaned))

    @field_validator("continuations", mode="before")
    @classmethod
    def copy_continuations(cls, value) -> dict[str, str]:
        copied = copy_json_value(value, "continuations")
        if type(copied) is not dict:
            raise ValueError("`continuations` must be an object.")
        if len(copied) > RECALL_MAX_CONTINUATIONS:
            raise ValueError(
                f"`continuations` cannot contain more than {RECALL_MAX_CONTINUATIONS} cursors."
            )
        result: dict[str, str] = {}
        for raw_channel, raw_cursor in copied.items():
            if type(raw_channel) is not str or type(raw_cursor) is not str:
                raise ValueError("Recall continuation channels and cursors must be strings.")
            channel = require_clean_nonblank(raw_channel, "continuation channel")
            cursor = require_clean_nonblank(raw_cursor, "continuation cursor")
            if len(cursor.encode("utf-8")) > RECALL_MAX_CONTINUATION_BYTES:
                raise ValueError(
                    "Recall continuation cursors cannot exceed "
                    f"{RECALL_MAX_CONTINUATION_BYTES} UTF-8 bytes."
                )
            result[channel] = cursor
        return {channel: result[channel] for channel in sorted(result)}

    @field_validator("continuations")
    @classmethod
    def freeze_continuations(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return FrozenJsonDict(value)

    @field_serializer("continuations")
    def serialize_continuations(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @field_validator("current_time")
    @classmethod
    def validate_current_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`current_time` must be timezone-aware.")
        return value.astimezone(UTC)

    def retrieval_text(self) -> str:
        """Resolve short follow-ups from bounded caller-supplied current context."""

        terms = self.query.split()
        if len(terms) > _SHORT_FOLLOWUP_MAX_TERMS:
            return self.query
        context: list[str] = []
        if self.work_context is not None:
            context.append(self.work_context)
        context.extend(self.recent_conversation[-2:])
        context.append(self.query)
        text = "\n".join(context)
        encoded = text.encode("utf-8")
        if len(encoded) <= RECALL_MAX_QUERY_BYTES:
            return text
        return encoded[-RECALL_MAX_QUERY_BYTES:].decode("utf-8", errors="ignore")

    def fingerprint(self) -> str:
        return sha256(
            canonical_durable_json_bytes(
                self.model_dump(mode="json"),
                "recall situation",
            )
        ).hexdigest()

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


class RecallRecord(BaseModel):
    """Bounded representation and exact locator for one canonical candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    identity: RetrievalCandidateIdentity
    representation: str
    text: str
    text_complete: bool
    content_hash: str
    locator: Mapping[str, Any]

    @field_validator("identity", mode="before")
    @classmethod
    def copy_identity(cls, value):
        if type(value) is RetrievalCandidateIdentity:
            return value.model_dump(mode="python")
        return value

    @field_validator("representation")
    @classmethod
    def validate_representation(cls, value: str) -> str:
        return require_clean_nonblank(value, "representation")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return require_nonblank(value, "text")

    @field_validator("text_complete", mode="before")
    @classmethod
    def validate_text_complete(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`text_complete` must be a boolean.")
        return value

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        value = require_clean_nonblank(value, "content_hash")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("`content_hash` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("locator", mode="before")
    @classmethod
    def copy_locator(cls, value) -> dict[str, Any]:
        return copy_durable_json_object(value, "locator")

    @field_validator("locator")
    @classmethod
    def freeze_locator(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        frozen = freeze_json_value(dict(value))
        if type(frozen) is not FrozenJsonDict:  # pragma: no cover - defensive invariant
            raise AssertionError("Recall locator did not freeze as an object.")
        return frozen

    @field_serializer("locator")
    def serialize_locator(self, value: Mapping[str, Any]) -> dict[str, Any]:
        thawed = thaw_json_value(value)
        if type(thawed) is not dict:  # pragma: no cover - defensive invariant
            raise AssertionError("Recall locator did not thaw as an object.")
        return thawed


class RecallSourceResult(BaseModel):
    """Validated source output before cross-source fusion."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    source: str
    channels: tuple[RankedRetrievalChannel, ...]
    records: tuple[RecallRecord, ...]
    coverage_complete: bool
    partial_reason: str | None = None

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return require_clean_nonblank(value, "source")

    @field_validator("channels", mode="before")
    @classmethod
    def copy_channels(cls, value) -> tuple[RankedRetrievalChannel, ...]:
        return tuple(
            RankedRetrievalChannel.model_validate(
                channel.model_dump(mode="python")
                if type(channel) is RankedRetrievalChannel
                else channel
            )
            for channel in value
        )

    @field_validator("records", mode="before")
    @classmethod
    def copy_records(cls, value) -> tuple[RecallRecord, ...]:
        return tuple(
            RecallRecord.model_validate(
                record.model_dump(mode="python") if type(record) is RecallRecord else record
            )
            for record in value
        )

    @field_validator("coverage_complete", mode="before")
    @classmethod
    def validate_coverage_complete(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`coverage_complete` must be a boolean.")
        return value

    @field_validator("partial_reason")
    @classmethod
    def validate_partial_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "partial_reason")

    @model_validator(mode="after")
    def validate_source_result(self) -> RecallSourceResult:
        channel_names = [channel.channel for channel in self.channels]
        if len(channel_names) != len(set(channel_names)):
            raise ValueError("A recall source cannot repeat a channel identity.")
        records = {record.identity.sort_key(): record for record in self.records}
        if len(records) != len(self.records):
            raise ValueError("A recall source cannot repeat a candidate record.")
        for channel in self.channels:
            for hit in channel.hits:
                record = records.get(hit.identity.sort_key())
                if record is None:
                    raise ValueError("Every ranked recall hit requires a candidate record.")
                if (
                    record.representation != hit.representation
                    or record.content_hash != hit.content_hash
                ):
                    raise ValueError("Ranked hit material conflicts with its candidate record.")
        hit_identities = {
            hit.identity.sort_key() for channel in self.channels for hit in channel.hits
        }
        if hit_identities != set(records):
            raise ValueError("Recall records must correspond exactly to ranked channel hits.")
        if self.coverage_complete and self.partial_reason is not None:
            raise ValueError("Complete source coverage cannot carry a partial reason.")
        if not self.coverage_complete and self.partial_reason is None:
            raise ValueError("Partial source coverage requires a bounded reason.")
        return self


class RecallSource(ABC):
    """Trusted read-only extension that returns bounded independently ranked lanes."""

    name: ClassVar[str]
    channel_names: ClassVar[tuple[str, ...]]
    continuation_channels: ClassVar[tuple[str, ...]] = ()

    def __init__(self, *, required: bool, candidate_limit: int) -> None:
        if type(required) is not bool:
            raise TypeError("required must be a boolean.")
        if type(candidate_limit) is not int or not 1 <= candidate_limit <= 100:
            raise ValueError("candidate_limit must be between 1 and 100.")
        self.required = required
        self.candidate_limit = candidate_limit

    @abstractmethod
    async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
        """Return source-owned channels after hard source filtering."""


class RecallEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    source_timeout_seconds: float = 2.0
    overall_timeout_seconds: float = 5.0
    max_parallel_sources: int = 4
    max_source_result_bytes: int = 256_000
    max_result_bytes: int = 128_000
    engine_version: str = RECALL_ENGINE_VERSION

    @field_validator("source_timeout_seconds", "overall_timeout_seconds", mode="before")
    @classmethod
    def validate_timeout(cls, value, info) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"`{info.field_name}` must be a number.")
        value = require_finite(float(value), info.field_name)
        if value <= 0 or value > 60:
            raise ValueError(f"`{info.field_name}` must be greater than 0 and at most 60.")
        return value

    @field_validator("max_parallel_sources", mode="before")
    @classmethod
    def validate_parallelism(cls, value) -> int:
        if type(value) is not int or not 1 <= value <= 32:
            raise ValueError("`max_parallel_sources` must be between 1 and 32.")
        return value

    @field_validator("max_source_result_bytes", "max_result_bytes", mode="before")
    @classmethod
    def validate_byte_limits(cls, value, info) -> int:
        if type(value) is not int or not 1 <= value <= RECALL_MAX_RESULT_BYTES:
            raise ValueError(
                f"`{info.field_name}` must be between 1 and {RECALL_MAX_RESULT_BYTES}."
            )
        return value

    @field_validator("engine_version")
    @classmethod
    def validate_engine_version(cls, value: str) -> str:
        if value != RECALL_ENGINE_VERSION:
            raise ValueError(f"`engine_version` must be {RECALL_ENGINE_VERSION!r}.")
        return value


class RecallSourceDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    source: str
    required: bool
    status: RecallSourceStatus
    channels: tuple[str, ...]
    failure_code: str | None = None

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        return require_clean_nonblank(value, "source")

    @field_validator("required", mode="before")
    @classmethod
    def validate_required(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`required` must be a boolean.")
        return value

    @field_validator("channels", mode="before")
    @classmethod
    def validate_channels(cls, value) -> tuple[str, ...]:
        if isinstance(value, str | bytes):
            raise ValueError("`channels` must be a sequence.")
        channels = tuple(require_clean_nonblank(item, "channels") for item in value)
        if not channels or len(channels) != len(set(channels)):
            raise ValueError("`channels` must contain unique channel names.")
        return tuple(sorted(channels))

    @field_validator("failure_code")
    @classmethod
    def validate_failure_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "failure_code")

    @model_validator(mode="after")
    def validate_status(self) -> RecallSourceDiagnostic:
        if (self.status is RecallSourceStatus.COMPLETE) != (self.failure_code is None):
            raise ValueError("Only incomplete source diagnostics carry a failure code.")
        return self


class RecallCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    fused: FusedRetrievalCandidate
    record: RecallRecord

    @field_validator("fused", mode="before")
    @classmethod
    def copy_fused(cls, value) -> FusedRetrievalCandidate:
        return FusedRetrievalCandidate.model_validate(
            value.model_dump(mode="python") if type(value) is FusedRetrievalCandidate else value
        )

    @field_validator("record", mode="before")
    @classmethod
    def copy_record(cls, value) -> RecallRecord:
        return RecallRecord.model_validate(
            value.model_dump(mode="python") if type(value) is RecallRecord else value
        )

    @model_validator(mode="after")
    def validate_identity(self) -> RecallCandidate:
        if self.fused.identity != self.record.identity:
            raise ValueError("Fused candidate identity conflicts with its recall record.")
        return self


class RecallResult(BaseModel):
    """Cross-source retrieval output; not admission, context, or exposure evidence."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
    )

    engine_version: str
    situation_sha256: str
    candidates: tuple[RecallCandidate, ...]
    fusion: RetrievalFusionDiagnostics
    sources: tuple[RecallSourceDiagnostic, ...]
    continuations: Mapping[str, str] = Field(default_factory=dict)
    truncated: bool
    omitted_by_result_bytes: int = 0

    @field_validator("engine_version")
    @classmethod
    def validate_engine_version(cls, value: str) -> str:
        if value != RECALL_ENGINE_VERSION:
            raise ValueError(f"`engine_version` must be {RECALL_ENGINE_VERSION!r}.")
        return value

    @field_validator("situation_sha256")
    @classmethod
    def validate_situation_hash(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("`situation_sha256` must be a lowercase SHA-256 digest.")
        return value

    @field_validator("candidates", mode="before")
    @classmethod
    def copy_candidates(cls, value) -> tuple[RecallCandidate, ...]:
        return tuple(
            RecallCandidate.model_validate(
                candidate.model_dump(mode="python")
                if type(candidate) is RecallCandidate
                else candidate
            )
            for candidate in value
        )

    @field_validator("fusion", mode="before")
    @classmethod
    def copy_fusion(cls, value) -> RetrievalFusionDiagnostics:
        return RetrievalFusionDiagnostics.model_validate(
            value.model_dump(mode="python") if type(value) is RetrievalFusionDiagnostics else value
        )

    @field_validator("sources", mode="before")
    @classmethod
    def copy_sources(cls, value) -> tuple[RecallSourceDiagnostic, ...]:
        return tuple(
            RecallSourceDiagnostic.model_validate(
                source.model_dump(mode="python")
                if type(source) is RecallSourceDiagnostic
                else source
            )
            for source in value
        )

    @field_validator("continuations", mode="before")
    @classmethod
    def copy_continuations(cls, value) -> dict[str, str]:
        copied = copy_json_value(value, "continuations")
        if type(copied) is not dict:
            raise ValueError("`continuations` must be an object.")
        if len(copied) > RECALL_MAX_CONTINUATIONS:
            raise ValueError("Recall results contain too many continuation cursors.")
        result: dict[str, str] = {}
        for raw_channel, raw_cursor in copied.items():
            if type(raw_channel) is not str or type(raw_cursor) is not str:
                raise ValueError("Recall continuation channels and cursors must be strings.")
            channel = require_clean_nonblank(raw_channel, "continuation channel")
            cursor = require_clean_nonblank(raw_cursor, "continuation cursor")
            if len(cursor.encode("utf-8")) > RECALL_MAX_CONTINUATION_BYTES:
                raise ValueError("Recall continuation cursor exceeds its byte bound.")
            result[channel] = cursor
        return {channel: result[channel] for channel in sorted(result)}

    @field_validator("continuations")
    @classmethod
    def freeze_continuations(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return FrozenJsonDict(value)

    @field_serializer("continuations")
    def serialize_continuations(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @field_validator("truncated", mode="before")
    @classmethod
    def validate_truncated(cls, value) -> bool:
        if type(value) is not bool:
            raise ValueError("`truncated` must be a boolean.")
        return value

    @field_validator("omitted_by_result_bytes", mode="before")
    @classmethod
    def validate_omitted(cls, value) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("`omitted_by_result_bytes` must be a non-negative integer.")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> RecallResult:
        identities = [candidate.fused.identity.sort_key() for candidate in self.candidates]
        if len(identities) != len(set(identities)):
            raise ValueError("Recall results cannot repeat a candidate identity.")
        if self.omitted_by_result_bytes and not self.truncated:
            raise ValueError("Byte-omitted recall results must be truncated.")
        if tuple(self.continuations) != self.fusion.continuation_channels:
            raise ValueError("Recall continuations conflict with fusion diagnostics.")
        source_names = [source.source for source in self.sources]
        if (
            not source_names
            or source_names != sorted(source_names)
            or len(source_names) != len(set(source_names))
        ):
            raise ValueError("Recall source diagnostics must be unique and canonically ordered.")
        source_channels = [channel for source in self.sources for channel in source.channels]
        fusion_channels = [channel.channel for channel in self.fusion.channels]
        if (
            fusion_channels != sorted(fusion_channels)
            or len(fusion_channels) != len(set(fusion_channels))
            or len(source_channels) != len(set(source_channels))
            or set(source_channels) != set(fusion_channels)
        ):
            raise ValueError("Recall source channels conflict with fusion diagnostics.")
        expected_truncated = (
            self.fusion.truncated
            or any(source.status is not RecallSourceStatus.COMPLETE for source in self.sources)
            or self.omitted_by_result_bytes > 0
        )
        if self.truncated != expected_truncated:
            raise ValueError("Recall truncation conflicts with its recorded diagnostics.")
        if len(self.candidates) + self.omitted_by_result_bytes != (
            self.fusion.returned_candidate_count
        ):
            raise ValueError("Recall byte omissions conflict with fusion candidate counts.")
        return self


@dataclass(frozen=True)
class _SourceRegistration:
    source: RecallSource
    name: str
    channel_names: tuple[str, ...]
    continuation_channels: tuple[str, ...]
    required: bool
    candidate_limit: int


@dataclass(frozen=True)
class _SourceExecution:
    registration: _SourceRegistration
    result: RecallSourceResult | None = None
    failure_code: str | None = None


class RecallEngine:
    """Run bounded sources concurrently and fuse their recorded ranks deterministically."""

    def __init__(
        self,
        sources: Sequence[RecallSource],
        *,
        fusion_config: WeightedReciprocalRankFusionConfig,
        fusion_strategy: RetrievalFusionStrategy | None = None,
        config: RecallEngineConfig | None = None,
    ) -> None:
        if isinstance(sources, RecallSource):
            sources = (sources,)
        try:
            copied_sources = tuple(islice(sources, _RECALL_MAX_SOURCES + 1))
        except TypeError as exc:
            raise TypeError("sources must be a sequence of RecallSource instances.") from exc
        if not copied_sources or any(
            not isinstance(source, RecallSource) for source in copied_sources
        ):
            raise TypeError("sources must contain at least one RecallSource instance.")
        if len(copied_sources) > _RECALL_MAX_SOURCES:
            raise ValueError(
                f"RecallEngine cannot register more than {_RECALL_MAX_SOURCES} sources."
            )
        registrations: list[_SourceRegistration] = []
        for source in copied_sources:
            name = require_clean_nonblank(source.name, "RecallSource.name")
            if len(name.encode("utf-8")) > _RECALL_MAX_NAME_BYTES:
                raise ValueError("RecallSource.name exceeds its UTF-8 byte bound.")
            if isinstance(source.channel_names, str | bytes):
                raise ValueError("RecallSource.channel_names must be a non-empty tuple.")
            try:
                raw_channel_names = tuple(islice(source.channel_names, _RECALL_MAX_CHANNELS + 1))
            except TypeError as exc:
                raise ValueError(
                    "RecallSource.channel_names must be a non-empty sequence."
                ) from exc
            if not raw_channel_names or len(raw_channel_names) > _RECALL_MAX_CHANNELS:
                raise ValueError("RecallSource.channel_names exceeds its count bound.")
            channel_names = tuple(
                require_clean_nonblank(channel, "RecallSource.channel_names")
                for channel in raw_channel_names
            )
            if any(
                len(channel.encode("utf-8")) > _RECALL_MAX_NAME_BYTES for channel in channel_names
            ):
                raise ValueError("A recall channel name exceeds its UTF-8 byte bound.")
            if len(channel_names) != len(set(channel_names)):
                raise ValueError("A RecallSource cannot repeat a channel name.")
            if isinstance(source.continuation_channels, str | bytes):
                raise ValueError("RecallSource.continuation_channels must be a tuple.")
            try:
                raw_continuation_channels = tuple(
                    islice(source.continuation_channels, _RECALL_MAX_CHANNELS + 1)
                )
            except TypeError as exc:
                raise ValueError("RecallSource.continuation_channels must be a sequence.") from exc
            if len(raw_continuation_channels) > _RECALL_MAX_CHANNELS:
                raise ValueError("RecallSource.continuation_channels exceeds its count bound.")
            continuation_channels = tuple(
                require_clean_nonblank(channel, "RecallSource.continuation_channels")
                for channel in raw_continuation_channels
            )
            if len(continuation_channels) != len(set(continuation_channels)):
                raise ValueError("A RecallSource cannot repeat a continuation channel.")
            if not set(continuation_channels).issubset(channel_names):
                raise ValueError("RecallSource continuation channels must belong to that source.")
            if type(source.required) is not bool:
                raise TypeError("RecallSource.required must be a boolean.")
            if type(source.candidate_limit) is not int or not 1 <= source.candidate_limit <= 100:
                raise ValueError("RecallSource.candidate_limit must be between 1 and 100.")
            registrations.append(
                _SourceRegistration(
                    source=source,
                    name=name,
                    channel_names=channel_names,
                    continuation_channels=continuation_channels,
                    required=source.required,
                    candidate_limit=source.candidate_limit,
                )
            )
        names = [registration.name for registration in registrations]
        if len(names) != len(set(names)):
            raise ValueError("Recall source names must be unique.")
        channel_names = [
            channel for registration in registrations for channel in registration.channel_names
        ]
        if len(channel_names) > _RECALL_MAX_CHANNELS:
            raise ValueError(
                f"RecallEngine cannot register more than {_RECALL_MAX_CHANNELS} channels."
            )
        if len(channel_names) != len(set(channel_names)):
            raise ValueError("Recall channel names must be globally unique.")
        if type(fusion_config) is not WeightedReciprocalRankFusionConfig:
            raise TypeError("fusion_config must be a WeightedReciprocalRankFusionConfig.")
        self._fusion_config = WeightedReciprocalRankFusionConfig.model_validate(
            fusion_config.model_dump(mode="python")
        )
        if set(channel_names) != set(self._fusion_config.channel_weights):
            raise ValueError("Recall sources must exactly own the configured fusion channels.")
        if any(
            registration.candidate_limit > self._fusion_config.max_candidates_per_channel
            for registration in registrations
        ):
            raise ValueError("A recall source candidate limit exceeds the fusion ceiling.")
        self._sources = tuple(sorted(registrations, key=lambda item: item.name))
        self._fusion_strategy = fusion_strategy or WeightedReciprocalRankFusion()
        if not isinstance(self._fusion_strategy, RetrievalFusionStrategy):
            raise TypeError("fusion_strategy must be a RetrievalFusionStrategy.")
        raw_strategy_version = getattr(self._fusion_strategy, "strategy_version", None)
        if type(raw_strategy_version) is not str:
            raise TypeError("RetrievalFusionStrategy.strategy_version must be a string.")
        strategy_version = require_clean_nonblank(
            raw_strategy_version,
            "RetrievalFusionStrategy.strategy_version",
        )
        if len(strategy_version.encode("utf-8")) > _RECALL_MAX_NAME_BYTES:
            raise ValueError("RetrievalFusionStrategy.strategy_version exceeds its byte bound.")
        if strategy_version != self._fusion_config.strategy_version:
            raise ValueError(
                "The fusion strategy identity must match fusion_config.strategy_version."
            )
        if config is not None and type(config) is not RecallEngineConfig:
            raise TypeError("config must be a RecallEngineConfig or None.")
        self._config = RecallEngineConfig.model_validate(
            (config or RecallEngineConfig()).model_dump(mode="python")
        )

    async def recall(self, situation: RecallSituation) -> RecallResult:
        if type(situation) is not RecallSituation:
            raise TypeError("situation must be a RecallSituation.")
        situation = RecallSituation.model_validate(situation.model_dump(mode="python"))
        continuation_channels = {
            channel
            for registration in self._sources
            for channel in registration.continuation_channels
        }
        unknown_continuations = set(situation.continuations) - continuation_channels
        if unknown_continuations:
            raise ValueError(
                "Recall situation contains a continuation for an unknown channel or a "
                "non-pageable channel."
            )
        semaphore = asyncio.Semaphore(self._config.max_parallel_sources)

        async def run_source(registration: _SourceRegistration) -> _SourceExecution:
            try:
                async with semaphore:
                    async with asyncio.timeout(self._config.source_timeout_seconds):
                        result = await registration.source.retrieve(situation)
                return _SourceExecution(registration=registration, result=result)
            except TimeoutError:
                return _SourceExecution(registration=registration, failure_code="timeout")
            except NotImplementedError:
                return _SourceExecution(registration=registration, failure_code="unsupported")
            except Exception:
                return _SourceExecution(registration=registration, failure_code="failed")

        try:
            async with asyncio.timeout(self._config.overall_timeout_seconds):
                executions = await asyncio.gather(
                    *(run_source(registration) for registration in self._sources)
                )
        except TimeoutError as exc:
            raise RecallSourceUnavailable("recall_engine", "overall_timeout") from exc
        required_failure = next(
            (
                execution
                for execution in executions
                if execution.failure_code is not None and execution.registration.required
            ),
            None,
        )
        if required_failure is not None:
            failure_code = required_failure.failure_code
            if failure_code is None:  # pragma: no cover - selected by the predicate above
                raise RuntimeError("Required recall source failure lost its bounded code.")
            raise RecallSourceUnavailable(
                required_failure.registration.name,
                failure_code,
            )

        channels: list[RankedRetrievalChannel] = []
        records_by_identity: dict[tuple[str, str, str], RecallRecord] = {}
        source_diagnostics: list[RecallSourceDiagnostic] = []
        for execution in executions:
            registration = execution.registration
            if execution.result is None:
                channels.extend(
                    RankedRetrievalChannel(
                        channel=channel_name,
                        index_version="unavailable",
                        candidate_limit=registration.candidate_limit,
                        truncated=True,
                        continuation=situation.continuations.get(channel_name),
                    )
                    for channel_name in registration.channel_names
                )
                source_diagnostics.append(
                    RecallSourceDiagnostic(
                        source=registration.name,
                        required=registration.required,
                        status=RecallSourceStatus.UNAVAILABLE,
                        channels=registration.channel_names,
                        failure_code=execution.failure_code,
                    )
                )
                continue
            result = RecallSourceResult.model_validate(execution.result.model_dump(mode="python"))
            result_size = len(
                canonical_durable_json_bytes(
                    result.model_dump(mode="json"),
                    "recall source result",
                )
            )
            if result_size > self._config.max_source_result_bytes:
                raise ValueError("Recall source result exceeded the configured byte ceiling.")
            if result.source != registration.name:
                raise ValueError("Recall source result identity conflicts with its registration.")
            if {channel.channel for channel in result.channels} != set(registration.channel_names):
                raise ValueError("Recall source returned an unexpected channel set.")
            for channel in sorted(result.channels, key=lambda item: item.channel):
                if channel.candidate_limit > registration.candidate_limit:
                    raise ValueError("Recall source exceeded its registered candidate limit.")
                if (
                    channel.continuation is not None
                    and channel.channel not in registration.continuation_channels
                ):
                    raise ValueError(
                        "Recall source returned a continuation for a non-pageable channel."
                    )
                hit_continuations = tuple(hit.continuation for hit in channel.hits)
                if channel.channel in registration.continuation_channels:
                    if any(continuation is None for continuation in hit_continuations):
                        raise ValueError(
                            "Pageable recall channels must provide a continuation after every hit."
                        )
                    if any(
                        len(continuation.encode("utf-8")) > RECALL_MAX_CONTINUATION_BYTES
                        for continuation in hit_continuations
                        if continuation is not None
                    ) or (
                        channel.continuation is not None
                        and len(channel.continuation.encode("utf-8"))
                        > RECALL_MAX_CONTINUATION_BYTES
                    ):
                        raise ValueError("A pageable channel cursor exceeds its byte bound.")
                    if (
                        channel.continuation is not None
                        and hit_continuations
                        and channel.continuation != hit_continuations[-1]
                    ):
                        raise ValueError(
                            "A pageable channel cursor must follow its final returned hit."
                        )
                    if (
                        not hit_continuations
                        and channel.continuation != situation.continuations.get(channel.channel)
                    ):
                        raise ValueError(
                            "A pageable channel cannot advance its cursor without a hit."
                        )
                elif any(continuation is not None for continuation in hit_continuations):
                    raise ValueError(
                        "Non-pageable recall channels cannot provide per-hit continuations."
                    )
                channels.append(channel)
            for record in result.records:
                key = record.identity.sort_key()
                existing = records_by_identity.get(key)
                if existing is not None and existing != record:
                    raise ValueError("Recall sources disagree on canonical candidate material.")
                records_by_identity[key] = record
            source_diagnostics.append(
                RecallSourceDiagnostic(
                    source=registration.name,
                    required=registration.required,
                    status=(
                        RecallSourceStatus.COMPLETE
                        if result.coverage_complete
                        else RecallSourceStatus.PARTIAL
                    ),
                    channels=tuple(channel.channel for channel in result.channels),
                    failure_code=result.partial_reason,
                )
            )

        raw_fusion = self._fusion_strategy.fuse(tuple(channels), self._fusion_config)
        if type(raw_fusion) is not RetrievalFusionResult:
            raise TypeError("RetrievalFusionStrategy must return a RetrievalFusionResult.")
        fusion = RetrievalFusionResult.model_validate(raw_fusion.model_dump(mode="python"))
        if type(self._fusion_strategy) is not WeightedReciprocalRankFusion:
            _validate_custom_fusion_result(
                fusion,
                channels=tuple(channels),
                config=self._fusion_config,
            )
        fused_candidates: list[RecallCandidate] = []
        for fused in fusion.candidates:
            record = records_by_identity.get(fused.identity.sort_key())
            if record is None:
                raise ValueError("Fused candidate has no validated recall record.")
            fused_candidates.append(RecallCandidate(fused=fused, record=record))
        partial_sources = any(
            diagnostic.status is not RecallSourceStatus.COMPLETE
            for diagnostic in source_diagnostics
        )
        situation_sha256 = situation.fingerprint()
        pageable_channels = {
            channel
            for registration in self._sources
            for channel in registration.continuation_channels
        }

        def safe_continuations(candidate_count: int) -> dict[str, str]:
            visible = {
                candidate.fused.identity.sort_key()
                for candidate in fused_candidates[:candidate_count]
            }
            safe: dict[str, str] = {}
            for channel in sorted(channels, key=lambda item: item.channel):
                if channel.channel not in pageable_channels:
                    continue
                ranked_hits = sorted(channel.hits, key=lambda hit: hit.rank)
                consumed = 0
                for hit in ranked_hits:
                    if hit.identity.sort_key() not in visible:
                        break
                    consumed += 1
                if consumed == len(ranked_hits):
                    cursor = channel.continuation
                elif consumed:
                    cursor = ranked_hits[consumed - 1].continuation
                else:
                    cursor = situation.continuations.get(channel.channel)
                if cursor is not None:
                    safe[channel.channel] = cursor
            return safe

        def diagnostics_with_continuations(
            continuations: Mapping[str, str],
        ) -> RetrievalFusionDiagnostics:
            payload = fusion.diagnostics.model_dump(mode="python")
            payload["continuation_channels"] = tuple(sorted(continuations))
            return RetrievalFusionDiagnostics.model_validate(payload)

        def result_with_prefix(candidate_count: int) -> RecallResult:
            omitted_by_bytes = len(fused_candidates) - candidate_count
            continuations = safe_continuations(candidate_count)
            return RecallResult(
                engine_version=self._config.engine_version,
                situation_sha256=situation_sha256,
                candidates=tuple(fused_candidates[:candidate_count]),
                fusion=diagnostics_with_continuations(continuations),
                sources=tuple(source_diagnostics),
                continuations=continuations,
                truncated=(fusion.diagnostics.truncated or partial_sources or omitted_by_bytes > 0),
                omitted_by_result_bytes=omitted_by_bytes,
            )

        def serialized_result_bytes(result: RecallResult) -> int:
            return len(
                canonical_durable_json_bytes(
                    result.model_dump(mode="json"),
                    "recall result",
                )
            )

        bounded_result = result_with_prefix(0)
        metadata_only_bytes = serialized_result_bytes(bounded_result)
        if metadata_only_bytes > self._config.max_result_bytes:
            raise ValueError("Recall result metadata exceeded the configured byte ceiling.")

        def durable_size(value: Any, field_name: str) -> int:
            return len(canonical_durable_json_bytes(value, field_name))

        candidate_prefix_bytes = [0]
        for candidate in fused_candidates:
            candidate_prefix_bytes.append(
                candidate_prefix_bytes[-1]
                + durable_size(candidate.model_dump(mode="json"), "recall candidate")
            )

        def array_size(item_bytes: int, item_count: int) -> int:
            return 2 if item_count == 0 else item_bytes + item_count + 1

        position_by_identity = {
            candidate.fused.identity.sort_key(): position
            for position, candidate in enumerate(fused_candidates, start=1)
        }
        continuation_events: dict[int, list[tuple[str, str | None]]] = {}
        for channel in sorted(channels, key=lambda item: item.channel):
            if channel.channel not in pageable_channels:
                continue
            ranked_hits = sorted(channel.hits, key=lambda hit: hit.rank)
            threshold = 0
            for hit_index, hit in enumerate(ranked_hits, start=1):
                position = position_by_identity.get(hit.identity.sort_key())
                if position is None:
                    break
                threshold = max(threshold, position)
                cursor = channel.continuation if hit_index == len(ranked_hits) else hit.continuation
                continuation_events.setdefault(threshold, []).append((channel.channel, cursor))

        current_continuations = safe_continuations(0)
        continuation_pair_bytes: dict[str, int] = {}
        continuation_pair_total = 0
        continuation_channel_total = 0
        encoded_channel_bytes: dict[str, int] = {}

        def channel_bytes(channel: str) -> int:
            cached = encoded_channel_bytes.get(channel)
            if cached is None:
                cached = durable_size(channel, "recall continuation channel")
                encoded_channel_bytes[channel] = cached
            return cached

        def continuation_pair_size(channel: str, cursor: str) -> int:
            return channel_bytes(channel) + 1 + durable_size(cursor, "recall continuation cursor")

        for channel, cursor in current_continuations.items():
            pair_bytes = continuation_pair_size(channel, cursor)
            continuation_pair_bytes[channel] = pair_bytes
            continuation_pair_total += pair_bytes
            continuation_channel_total += channel_bytes(channel)

        fusion_without_continuations = fusion.diagnostics.model_dump(mode="json")
        fusion_without_continuations["continuation_channels"] = []
        empty_fusion_bytes = durable_size(
            fusion_without_continuations,
            "recall fusion diagnostics",
        )
        normalized_payload = bounded_result.model_dump(mode="json")
        normalized_payload.update(
            {
                "candidates": [],
                "continuations": {},
                "fusion": fusion_without_continuations,
                "omitted_by_result_bytes": 0,
                "truncated": False,
            }
        )
        fixed_result_bytes = durable_size(
            normalized_payload,
            "normalized recall result",
        ) - sum(
            (
                2,  # empty candidates array
                2,  # empty continuations object
                empty_fusion_bytes,
                1,  # integer zero
                5,  # JSON false
            )
        )

        def current_continuation_object_bytes() -> int:
            count = len(current_continuations)
            return 2 if count == 0 else continuation_pair_total + count + 1

        def current_continuation_channel_bytes() -> int:
            return array_size(continuation_channel_total, len(current_continuations))

        def exact_result_bytes(candidate_count: int) -> int:
            omitted = len(fused_candidates) - candidate_count
            truncated = fusion.diagnostics.truncated or partial_sources or omitted > 0
            fusion_bytes = empty_fusion_bytes - 2 + current_continuation_channel_bytes()
            return sum(
                (
                    fixed_result_bytes,
                    array_size(candidate_prefix_bytes[candidate_count], candidate_count),
                    current_continuation_object_bytes(),
                    fusion_bytes,
                    durable_size(omitted, "recall omitted candidate count"),
                    4 if truncated else 5,
                )
            )

        best_candidate_count = 0
        best_result_bytes = metadata_only_bytes
        for candidate_count in range(1, len(fused_candidates) + 1):
            for channel, cursor in continuation_events.get(candidate_count, ()):
                existing_pair_bytes = continuation_pair_bytes.pop(channel, None)
                if existing_pair_bytes is not None:
                    continuation_pair_total -= existing_pair_bytes
                    continuation_channel_total -= channel_bytes(channel)
                    current_continuations.pop(channel, None)
                if cursor is not None:
                    pair_bytes = continuation_pair_size(channel, cursor)
                    continuation_pair_bytes[channel] = pair_bytes
                    continuation_pair_total += pair_bytes
                    continuation_channel_total += channel_bytes(channel)
                    current_continuations[channel] = cursor
            result_bytes = exact_result_bytes(candidate_count)
            if result_bytes <= self._config.max_result_bytes:
                best_candidate_count = candidate_count
                best_result_bytes = result_bytes

        bounded_result = result_with_prefix(best_candidate_count)
        if serialized_result_bytes(bounded_result) != best_result_bytes:
            raise RuntimeError("Recall result byte accounting diverged from serialization.")
        return bounded_result


class KnowledgeRecallSource(RecallSource):
    """Revision-exact lexical and optional semantic lanes from one KnowledgeStore."""

    name = "knowledge"
    channel_names = (KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL)

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        required: bool = True,
        candidate_limit: int = 20,
        max_bytes: int = 64_000,
        max_record_bytes: int = 8_000,
        semantic_timeout_seconds: float = 1.0,
    ) -> None:
        if not isinstance(store, KnowledgeStore):
            raise TypeError("store must be a KnowledgeStore.")
        super().__init__(required=required, candidate_limit=candidate_limit)
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer.")
        if type(max_record_bytes) is not int or not 1 <= max_record_bytes <= max_bytes:
            raise ValueError("max_record_bytes must be positive and cannot exceed max_bytes.")
        if isinstance(semantic_timeout_seconds, bool) or not isinstance(
            semantic_timeout_seconds, int | float
        ):
            raise ValueError("semantic_timeout_seconds must be a number.")
        semantic_timeout_seconds = require_finite(
            float(semantic_timeout_seconds),
            "semantic_timeout_seconds",
        )
        if not 0 < semantic_timeout_seconds <= 60:
            raise ValueError("semantic_timeout_seconds must be greater than 0 and at most 60.")
        self._store = store
        self._max_bytes = max_bytes
        self._max_record_bytes = max_record_bytes
        self._semantic_timeout_seconds = semantic_timeout_seconds

    async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
        if situation.knowledge_access_scope is None:
            raise ValueError("Knowledge recall requires an explicit KnowledgeAccessScope.")
        text = situation.retrieval_text()
        lexical_query = KnowledgeQuery(
            text=text,
            namespace=situation.knowledge_namespace,
            mode=KnowledgeSearchMode.KEYWORD,
            limit=self.candidate_limit,
            max_bytes=self._max_bytes,
        )
        semantic: KnowledgeSearchResult | None = None
        semantic_failure: str | None = None
        semantic_supported = KnowledgeSearchMode.SEMANTIC in self._store.supported_search_modes()
        semantic_task: asyncio.Task[tuple[KnowledgeSearchResult | None, str | None]] | None = None
        if semantic_supported:
            semantic_query = KnowledgeQuery.model_validate(
                {
                    **lexical_query.model_dump(mode="python"),
                    "mode": KnowledgeSearchMode.SEMANTIC,
                }
            )

            async def retrieve_semantic() -> tuple[KnowledgeSearchResult | None, str | None]:
                try:
                    async with asyncio.timeout(self._semantic_timeout_seconds):
                        result = await self._store.search(
                            semantic_query,
                            access_scope=situation.knowledge_access_scope,
                        )
                except TimeoutError:
                    return None, "semantic_timeout"
                except Exception:
                    return None, "semantic_failed"
                return result, None

            semantic_task = asyncio.create_task(
                retrieve_semantic(),
                name="cayu-knowledge-semantic-recall",
            )
        try:
            lexical = await self._store.search(
                lexical_query,
                access_scope=situation.knowledge_access_scope,
            )
        except BaseException:
            if semantic_task is not None:
                semantic_task.cancel()
                with suppress(BaseException):
                    await semantic_task
            raise
        if semantic_task is not None:
            semantic, semantic_failure = await semantic_task

        records: dict[tuple[str, str, str], RecallRecord] = {}
        lexical_channel = self._channel_from_result(
            lexical,
            channel=KNOWLEDGE_LEXICAL_CHANNEL,
            index_version="cayu.knowledge.lexical.v1",
            records=records,
            coverage_complete=True,
        )
        partial_reasons: list[str] = []
        if lexical.truncated:
            partial_reasons.append("lexical_truncated")
        if semantic is None:
            semantic_channel = RankedRetrievalChannel(
                channel=KNOWLEDGE_SEMANTIC_CHANNEL,
                index_version=("unsupported" if not semantic_supported else "unavailable"),
                candidate_limit=self.candidate_limit,
                truncated=True,
            )
            partial_reasons.append(semantic_failure or "semantic_unsupported")
        else:
            semantic_index_complete = bool(semantic.index_coverage) and all(
                coverage.complete for coverage in semantic.index_coverage
            )
            semantic_channel = self._channel_from_result(
                semantic,
                channel=KNOWLEDGE_SEMANTIC_CHANNEL,
                index_version=_knowledge_coverage_version(semantic.index_coverage),
                records=records,
                coverage_complete=semantic_index_complete,
            )
            if not semantic_index_complete:
                partial_reasons.append("semantic_index_partial")
            if semantic.truncated:
                partial_reasons.append("semantic_truncated")
        return RecallSourceResult(
            source=self.name,
            channels=(lexical_channel, semantic_channel),
            records=tuple(records[key] for key in sorted(records)),
            coverage_complete=not partial_reasons,
            partial_reason=(None if not partial_reasons else "+".join(partial_reasons)),
        )

    def _channel_from_result(
        self,
        result: KnowledgeSearchResult,
        *,
        channel: str,
        index_version: str,
        records: dict[tuple[str, str, str], RecallRecord],
        coverage_complete: bool,
    ) -> RankedRetrievalChannel:
        ranked: list[RankedRetrievalHit] = []
        for fallback_rank, hit in enumerate(result.hits, start=1):
            record = _knowledge_recall_record(hit, max_text_bytes=self._max_record_bytes)
            key = record.identity.sort_key()
            existing = records.get(key)
            if existing is not None and existing != record:
                raise ValueError("Knowledge lanes disagree on exact candidate material.")
            records[key] = record
            ranked.append(
                RankedRetrievalHit(
                    identity=record.identity,
                    rank=hit.rank or fallback_rank,
                    representation=record.representation,
                    content_hash=record.content_hash,
                    explanations=(() if hit.reason is None else (hit.reason,)),
                    raw_score=hit.score,
                    features={"current_revision": 1.0},
                )
            )
        return RankedRetrievalChannel(
            channel=channel,
            index_version=index_version,
            candidate_limit=self.candidate_limit,
            hits=tuple(ranked),
            truncated=result.truncated or not coverage_complete,
        )


class TranscriptRecallSource(RecallSource):
    """Authoritative transcript narrative lane over explicit session scope."""

    name = "transcript"
    channel_names = (TRANSCRIPT_LEXICAL_CHANNEL,)
    continuation_channels = (TRANSCRIPT_LEXICAL_CHANNEL,)

    def __init__(
        self,
        store: SessionStore,
        *,
        required: bool = False,
        candidate_limit: int = 20,
        max_bytes: int = 64_000,
        max_records_scanned: int = 10_000,
    ) -> None:
        if not isinstance(store, SessionStore):
            raise TypeError("store must be a SessionStore.")
        super().__init__(required=required, candidate_limit=candidate_limit)
        if not store.supports_transcript_search:
            raise ValueError("SessionStore does not advertise transcript search support.")
        if (
            type(max_bytes) is not int
            or not TRANSCRIPT_SEARCH_MIN_MAX_BYTES <= max_bytes <= TRANSCRIPT_SEARCH_MAX_BYTES
        ):
            raise ValueError(
                "max_bytes must be between "
                f"{TRANSCRIPT_SEARCH_MIN_MAX_BYTES} and {TRANSCRIPT_SEARCH_MAX_BYTES}."
            )
        if (
            type(max_records_scanned) is not int
            or not 1 <= max_records_scanned <= TRANSCRIPT_SEARCH_MAX_SCAN_LIMIT
        ):
            raise ValueError(
                f"max_records_scanned must be between 1 and {TRANSCRIPT_SEARCH_MAX_SCAN_LIMIT}."
            )
        self._store = store
        self._max_bytes = max_bytes
        self._max_records_scanned = max_records_scanned

    async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
        if not situation.transcript_session_ids:
            return RecallSourceResult(
                source=self.name,
                channels=(
                    RankedRetrievalChannel(
                        channel=TRANSCRIPT_LEXICAL_CHANNEL,
                        index_version=TRANSCRIPT_SEARCH_INDEX_VERSION,
                        candidate_limit=self.candidate_limit,
                    ),
                ),
                records=(),
                coverage_complete=True,
            )
        result = await self._store.search_transcript(
            TranscriptSearchQuery(
                text=situation.retrieval_text(),
                session_ids=situation.transcript_session_ids,
                limit=self.candidate_limit,
                max_bytes=self._max_bytes,
                max_records_scanned=self._max_records_scanned,
                cursor=situation.continuations.get(TRANSCRIPT_LEXICAL_CHANNEL),
            )
        )
        records: list[RecallRecord] = []
        ranked: list[RankedRetrievalHit] = []
        for rank, hit in enumerate(result.hits, start=1):
            identity = RetrievalCandidateIdentity(
                record_type="transcript_message",
                record_id=f"{hit.session_id}:{hit.transcript_index}",
                revision=hit.content_hash,
            )
            record = RecallRecord(
                identity=identity,
                representation="transcript_text",
                text=hit.text,
                text_complete=hit.text_complete,
                content_hash=hit.content_hash,
                locator={
                    "session_id": hit.session_id,
                    "interaction_id": hit.interaction_id,
                    "transcript_index": hit.transcript_index,
                    "text_part_indexes": list(hit.text_part_indexes),
                },
            )
            records.append(record)
            ranked.append(
                RankedRetrievalHit(
                    identity=identity,
                    rank=rank,
                    representation=record.representation,
                    content_hash=record.content_hash,
                    explanations=("authoritative transcript text match",),
                    raw_score=hit.raw_score,
                    features={"authoritative_evidence": 1.0},
                    continuation=encode_transcript_search_cursor(
                        result.query,
                        raw_score=int(hit.raw_score or 0),
                        session_id=hit.session_id,
                        transcript_index=hit.transcript_index,
                    ),
                )
            )
        channel = RankedRetrievalChannel(
            channel=TRANSCRIPT_LEXICAL_CHANNEL,
            index_version=result.index_version,
            candidate_limit=self.candidate_limit,
            hits=tuple(ranked),
            truncated=result.truncated,
            continuation=result.next_cursor,
        )
        return RecallSourceResult(
            source=self.name,
            channels=(channel,),
            records=tuple(records),
            coverage_complete=result.coverage_complete,
            partial_reason=(None if result.coverage_complete else "scan_limit"),
        )


def _knowledge_recall_record(
    hit: KnowledgeHit,
    *,
    max_text_bytes: int,
) -> RecallRecord:
    if hit.chunk is None:
        identity = RetrievalCandidateIdentity(
            record_type="knowledge_entry",
            record_id=hit.entry.id,
            revision=str(hit.entry.revision),
        )
        representation = "entry_text"
        text = hit.entry.text
        locator = {
            "entry_id": hit.entry.id,
            "entry_revision": hit.entry.revision,
        }
    else:
        identity = RetrievalCandidateIdentity(
            record_type="knowledge_chunk",
            record_id=hit.chunk.id,
            revision=str(hit.entry.revision),
        )
        representation = "chunk_text"
        text = hit.chunk.text
        locator = {
            "entry_id": hit.entry.id,
            "entry_revision": hit.entry.revision,
            "chunk_id": hit.chunk.id,
            "chunk_index": hit.chunk.chunk_index,
        }
    encoded = text.encode("utf-8")
    preview = encoded[:max_text_bytes].decode("utf-8", errors="ignore")
    if not preview:
        raise ValueError("Knowledge recall record cannot fit its configured byte bound.")
    return RecallRecord(
        identity=identity,
        representation=representation,
        text=preview,
        text_complete=len(preview.encode("utf-8")) == len(encoded),
        content_hash=sha256(encoded).hexdigest(),
        locator=locator,
    )


def _knowledge_coverage_version(coverage: Sequence[KnowledgeIndexCoverage]) -> str:
    if not coverage:
        return "unreported"
    payload = [item.model_dump(mode="json") for item in coverage]
    return (
        "sha256:"
        + sha256(canonical_durable_json_bytes(payload, "knowledge semantic coverage")).hexdigest()
    )


def _validate_custom_fusion_result(
    result: RetrievalFusionResult,
    *,
    channels: tuple[RankedRetrievalChannel, ...],
    config: WeightedReciprocalRankFusionConfig,
) -> None:
    """Reject custom rankings that alter validated evidence or coverage facts."""

    reference_config = config.model_copy(
        update={"strategy_version": WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION}
    )
    reference = WeightedReciprocalRankFusion().fuse(channels, reference_config)
    expected_diagnostics_payload = reference.diagnostics.model_dump(mode="python")
    expected_diagnostics_payload.update(
        {
            "strategy_version": config.strategy_version,
            "configuration_version": config.configuration_version,
            "configuration_sha256": config.fingerprint(),
        }
    )
    expected_diagnostics = RetrievalFusionDiagnostics.model_validate(expected_diagnostics_payload)
    if result.diagnostics != expected_diagnostics:
        raise ValueError("Custom recall fusion returned inconsistent diagnostics.")
    identities = [candidate.identity.sort_key() for candidate in result.candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("Custom recall fusion repeated a candidate identity.")

    hits_by_identity: dict[
        tuple[str, str, str],
        list[tuple[RankedRetrievalChannel, RankedRetrievalHit]],
    ] = {}
    for channel in channels:
        for hit in channel.hits:
            hits_by_identity.setdefault(hit.identity.sort_key(), []).append((channel, hit))
    for candidate in result.candidates:
        matches = hits_by_identity.get(candidate.identity.sort_key())
        if not matches:
            raise ValueError("Custom recall fusion returned an unknown candidate identity.")
        expected_features = matches[0][1].features
        if any(hit.features != expected_features for _, hit in matches):
            raise ValueError("Recall channels disagree on canonical candidate features.")
        if candidate.features != expected_features:
            raise ValueError("Custom recall fusion altered canonical candidate features.")
        if candidate.best_rank != min(hit.rank for _, hit in matches) or (
            candidate.channel_count != len({channel.channel for channel, _ in matches})
        ):
            raise ValueError("Custom recall fusion altered candidate rank provenance.")
        expected_matches = sorted(
            (
                channel.channel,
                channel.index_version,
                hit.rank,
                hit.representation,
                hit.content_hash,
                hit.explanations,
                hit.raw_score,
            )
            for channel, hit in matches
        )
        actual_matches = sorted(
            (
                match.channel,
                match.index_version,
                match.rank,
                match.representation,
                match.content_hash,
                match.explanations,
                match.raw_score,
            )
            for match in candidate.matches
        )
        if actual_matches != expected_matches:
            raise ValueError("Custom recall fusion altered candidate match provenance.")
        require_finite(candidate.score, "custom fused candidate score")
        require_finite(
            candidate.reciprocal_rank_score,
            "custom fused reciprocal-rank score",
        )
        require_finite(candidate.feature_adjustment, "custom fused feature adjustment")


__all__ = [
    "KNOWLEDGE_LEXICAL_CHANNEL",
    "KNOWLEDGE_SEMANTIC_CHANNEL",
    "RECALL_ENGINE_VERSION",
    "TRANSCRIPT_LEXICAL_CHANNEL",
    "KnowledgeRecallSource",
    "RecallCandidate",
    "RecallEngine",
    "RecallEngineConfig",
    "RecallRecord",
    "RecallResult",
    "RecallSituation",
    "RecallSource",
    "RecallSourceDiagnostic",
    "RecallSourceResult",
    "RecallSourceStatus",
    "RecallSourceUnavailable",
    "TranscriptRecallSource",
]
