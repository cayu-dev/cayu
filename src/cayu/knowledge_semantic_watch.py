"""Policy-governed semantic-watch evaluation over exact knowledge recall.

This module deliberately separates three concerns: ``RecallEngine`` supplies
bounded match evidence, an application policy decides whether that evidence is
actionable, and ``KnowledgeStore`` durably records the resulting route.  It does
not inject context, dispatch work, or authorize a tool or external effect.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from itertools import islice
from threading import Lock
from typing import Any, Literal, Protocol

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
    FrozenJsonList,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    freeze_json_value,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    thaw_json_value,
)
from cayu.recall import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    RECALL_ENGINE_VERSION,
    RECALL_MAX_KNOWLEDGE_GROUPED_ASPECT_BYTES,
    RECALL_MAX_QUERY_BYTES,
    RecallEngine,
    RecallResult,
    RecallSituation,
    RecallSourceDiagnostic,
    RecallSourceStatus,
    RecallSourceUnavailable,
)
from cayu.retrieval import RetrievalFusionDiagnostics
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_NAMESPACE,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeRevisionRef,
    KnowledgeStatus,
    KnowledgeStore,
    copy_knowledge_access_scope,
    knowledge_access_scope_sha256,
)

MAX_KNOWLEDGE_SEMANTIC_WATCH_OBSERVATION_BYTES = RECALL_MAX_QUERY_BYTES
MAX_KNOWLEDGE_SEMANTIC_WATCH_ANNOTATION_BYTES = 4_096
MAX_KNOWLEDGE_SEMANTIC_WATCH_POLICY_REQUEST_BYTES = 256_000
MAX_KNOWLEDGE_SEMANTIC_WATCH_RECEIPT_BYTES = 384_000
MAX_KNOWLEDGE_SEMANTIC_WATCH_CANDIDATES = 20
MAX_KNOWLEDGE_SEMANTIC_WATCH_ASPECT_GROUPS = 6
MAX_KNOWLEDGE_SEMANTIC_WATCH_ASPECTS_PER_GROUP = 128

_IDENTITY_MAX_BYTES = 256
_MAX_REASONS_PER_CHANNEL_MATCH = 16
_SHA256_HEX = frozenset("0123456789abcdef")
_MAX_RETAINED_POLICY_TASKS = 256
_RETAINED_POLICY_TASKS: set[asyncio.Task[object]] = set()
_RETAINED_POLICY_TASKS_LOCK = Lock()


def _clean(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"`{field_name}` must be a string.")
    clean = require_durable_clean_nonblank(value, field_name)
    if len(clean.encode("utf-8")) > _IDENTITY_MAX_BYTES:
        raise ValueError(f"`{field_name}` must be at most {_IDENTITY_MAX_BYTES} UTF-8 bytes.")
    return clean


def _sha256(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"`{field_name}` must be lowercase SHA-256 hex.")
    return value


def _fingerprint(value: object, field_name: str) -> str:
    return sha256(canonical_durable_json_bytes(value, field_name)).hexdigest()


def _max_candidates(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_KNOWLEDGE_SEMANTIC_WATCH_CANDIDATES:
        raise ValueError(
            f"`max_candidates` must be between 1 and {MAX_KNOWLEDGE_SEMANTIC_WATCH_CANDIDATES}."
        )
    return value


def _aspect_groups(value: object) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("`knowledge_aspect_groups` must be a sequence of groups.")
    if len(value) > MAX_KNOWLEDGE_SEMANTIC_WATCH_ASPECT_GROUPS:
        raise ValueError("`knowledge_aspect_groups` exceeds its group bound.")
    groups: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    total_bytes = 0
    for group_index, group in enumerate(value):
        if isinstance(group, str | bytes) or not isinstance(group, Sequence) or not group:
            raise ValueError(
                f"`knowledge_aspect_groups[{group_index}]` must be a non-empty sequence."
            )
        if len(group) > MAX_KNOWLEDGE_SEMANTIC_WATCH_ASPECTS_PER_GROUP:
            raise ValueError("A semantic-watch aspect group exceeds its value bound.")
        copied = tuple(_clean(item, "knowledge_aspect_groups item") for item in group)
        total_bytes += sum(len(item.encode("utf-8")) for item in copied)
        if total_bytes > RECALL_MAX_KNOWLEDGE_GROUPED_ASPECT_BYTES:
            raise ValueError(
                "`knowledge_aspect_groups` exceeds the recall aggregate UTF-8 byte bound."
            )
        if len(copied) != len(set(copied)) or copied in seen:
            raise ValueError("Semantic-watch aspect groups cannot contain duplicates.")
        seen.add(copied)
        groups.append(copied)
    return tuple(groups)


def _required_channels(value: object) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("`required_channels` must be a sequence.")
    if not 1 <= len(value) <= 2:
        raise ValueError("`required_channels` must select one or two knowledge recall lanes.")
    channels = tuple(sorted({_clean(item, "required_channels item") for item in value}))
    supported = {KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL}
    if not channels or len(channels) != len(value) or not set(channels) <= supported:
        raise ValueError("`required_channels` must select unique knowledge recall lanes.")
    return channels


class _WatchModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )


class KnowledgeSemanticWatchDisposition(StrEnum):
    """Application-policy disposition for one bounded watch evaluation."""

    IGNORE = "ignore"
    EMIT = "emit"
    ROUTE_TO_REVIEW = "route_to_review"


class KnowledgeSemanticWatchConfig(_WatchModel):
    """Application-owned identities and bounds for one watch profile."""

    schema_version: Literal[1] = 1
    watch_identity: str
    watch_version: str
    recall_profile_identity: str
    recall_profile_version: str
    policy_identity: str
    policy_version: str
    policy_timeout_seconds: float = 5.0
    max_candidates: int = 10
    knowledge_namespace: str = DEFAULT_KNOWLEDGE_NAMESPACE
    knowledge_aspect_groups: tuple[tuple[str, ...], ...] = ()
    required_channels: tuple[str, ...] = (KNOWLEDGE_LEXICAL_CHANNEL,)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator(
        "watch_identity",
        "watch_version",
        "recall_profile_identity",
        "recall_profile_version",
        "policy_identity",
        "policy_version",
        "knowledge_namespace",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("policy_timeout_seconds", mode="before")
    @classmethod
    def validate_timeout(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("`policy_timeout_seconds` must be a number.")
        result = float(value)
        if not 0 < result <= 60 or result != result or result in {float("inf"), float("-inf")}:
            raise ValueError("`policy_timeout_seconds` must be greater than 0 and at most 60.")
        return result

    @field_validator("max_candidates", mode="before")
    @classmethod
    def validate_max_candidates(cls, value: object) -> int:
        return _max_candidates(value)

    @field_validator("knowledge_aspect_groups", mode="before")
    @classmethod
    def validate_aspect_groups(cls, value: object) -> tuple[tuple[str, ...], ...]:
        return _aspect_groups(value)

    @field_validator("required_channels", mode="before")
    @classmethod
    def validate_required_channels(cls, value: object) -> tuple[str, ...]:
        return _required_channels(value)

    @model_validator(mode="after")
    def validate_authority_separation(self) -> KnowledgeSemanticWatchConfig:
        if self.policy_identity in {self.watch_identity, self.recall_profile_identity}:
            raise ValueError("Watch and recall identities cannot authorize their own matches.")
        return self


class KnowledgeSemanticWatchInvocation(_WatchModel):
    """Durable-safe identity for an observation before recall or policy execution."""

    schema_version: Literal[1] = 1
    operation_id: str
    observation_id: str
    observation_source_type: str
    observation_source_id: str
    observation_sha256: str
    observation_bytes: int
    watch_identity: str
    watch_version: str
    recall_profile_identity: str
    recall_profile_version: str
    policy_identity: str
    policy_version: str
    knowledge_namespace: str
    knowledge_aspect_groups: tuple[tuple[str, ...], ...]
    required_channels: tuple[str, ...]
    max_candidates: int
    access_scope: KnowledgeAccessScope

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator(
        "operation_id",
        "observation_id",
        "observation_source_type",
        "observation_source_id",
        "watch_identity",
        "watch_version",
        "recall_profile_identity",
        "recall_profile_version",
        "policy_identity",
        "policy_version",
        "knowledge_namespace",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("observation_sha256")
    @classmethod
    def validate_observation_sha256(cls, value: str) -> str:
        return _sha256(value, "observation_sha256")

    @field_validator("observation_bytes", mode="before")
    @classmethod
    def validate_observation_bytes(cls, value: object) -> int:
        if (
            type(value) is not int
            or not 1 <= value <= MAX_KNOWLEDGE_SEMANTIC_WATCH_OBSERVATION_BYTES
        ):
            raise ValueError("`observation_bytes` is outside the semantic-watch input bound.")
        return value

    @field_validator("max_candidates", mode="before")
    @classmethod
    def validate_max_candidates(cls, value: object) -> int:
        return _max_candidates(value)

    @field_validator("knowledge_aspect_groups", mode="before")
    @classmethod
    def validate_aspect_groups(cls, value: object) -> tuple[tuple[str, ...], ...]:
        return _aspect_groups(value)

    @field_validator("required_channels", mode="before")
    @classmethod
    def validate_required_channels(cls, value: object) -> tuple[str, ...]:
        return _required_channels(value)

    @field_validator("access_scope", mode="before")
    @classmethod
    def copy_access_scope(cls, value: object) -> object:
        if type(value) is KnowledgeAccessScope:
            return value.model_dump(mode="python", warnings=False)
        return value

    @field_validator("access_scope")
    @classmethod
    def freeze_access_scope(cls, value: KnowledgeAccessScope) -> KnowledgeAccessScope:
        object.__setattr__(value, "allowed_namespaces", FrozenJsonList(value.allowed_namespaces))
        object.__setattr__(value, "required_labels", FrozenJsonDict(value.required_labels))
        object.__setattr__(
            value,
            "allowed_visibilities",
            FrozenJsonList(value.allowed_visibilities),
        )
        object.__setattr__(
            value,
            "allowed_source_types",
            (
                None
                if value.allowed_source_types is None
                else FrozenJsonList(value.allowed_source_types)
            ),
        )
        object.__setattr__(
            value,
            "allowed_source_ids",
            (
                None
                if value.allowed_source_ids is None
                else FrozenJsonList(value.allowed_source_ids)
            ),
        )
        object.__setattr__(value, "allowed_statuses", FrozenJsonList(value.allowed_statuses))
        return value

    @field_serializer("access_scope")
    def serialize_access_scope(self, value: KnowledgeAccessScope) -> dict[str, Any]:
        return copy_knowledge_access_scope(value).model_dump(mode="json", warnings=False)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-semantic-watch-invocation.v1",
                "invocation": self.model_dump(mode="json"),
            },
            "knowledge semantic watch invocation fingerprint",
        )

    @property
    def access_scope_sha256(self) -> str:
        return knowledge_access_scope_sha256(self.access_scope)


class KnowledgeSemanticWatchChannelMatch(_WatchModel):
    """Safe rank evidence from one independently ranked recall lane."""

    channel: str
    index_version: str
    rank: int
    reasons: tuple[str, ...]
    raw_score: float | None = None

    @field_validator("channel", "index_version")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("rank", mode="before")
    @classmethod
    def validate_rank(cls, value: object) -> int:
        if type(value) is not int or value < 1:
            raise ValueError("`rank` must be a positive integer.")
        return value

    @field_validator("reasons", mode="before")
    @classmethod
    def validate_reasons(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("`reasons` must be a sequence.")
        if not 1 <= len(value) <= _MAX_REASONS_PER_CHANNEL_MATCH:
            raise ValueError(
                "`reasons` must contain between 1 and "
                f"{_MAX_REASONS_PER_CHANNEL_MATCH} safe reasons."
            )
        reasons = tuple(_clean(item, "reasons item") for item in value)
        if len(reasons) != len(set(reasons)):
            raise ValueError("`reasons` cannot contain duplicates.")
        return reasons

    @field_validator("raw_score", mode="before")
    @classmethod
    def validate_raw_score(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("`raw_score` must be a number or None.")
        score = float(value)
        if score != score or score in {float("inf"), float("-inf")}:
            raise ValueError("`raw_score` must be finite.")
        return score


class KnowledgeSemanticWatchCandidate(_WatchModel):
    """Durable-safe exact knowledge match presented to application policy."""

    reference: KnowledgeRevisionRef
    record_type: Literal["knowledge_entry", "knowledge_chunk"]
    record_id: str
    representation: str
    content_hash: str
    fused_rank: int
    fused_score: float
    best_rank: int
    channel_matches: tuple[KnowledgeSemanticWatchChannelMatch, ...]

    @field_validator("reference", mode="before")
    @classmethod
    def copy_reference(cls, value: object) -> object:
        if type(value) is KnowledgeRevisionRef:
            return value.model_dump(mode="python", warnings=False)
        return value

    @field_validator("record_id", "representation")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("content_hash")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        return _sha256(value, "content_hash")

    @field_validator("fused_rank", "best_rank", mode="before")
    @classmethod
    def validate_rank(cls, value: object, info) -> int:
        if type(value) is not int or value < 1:
            raise ValueError(f"`{info.field_name}` must be a positive integer.")
        return value

    @field_validator("fused_score", mode="before")
    @classmethod
    def validate_score(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("`fused_score` must be a number.")
        score = float(value)
        if score != score or score in {float("inf"), float("-inf")}:
            raise ValueError("`fused_score` must be finite.")
        return score

    @field_validator("channel_matches", mode="before")
    @classmethod
    def copy_channel_matches(cls, value: object) -> tuple[KnowledgeSemanticWatchChannelMatch, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("`channel_matches` must be a sequence.")
        if not 1 <= len(value) <= 2:
            raise ValueError("`channel_matches` must contain one or two knowledge recall lanes.")
        matches = tuple(
            KnowledgeSemanticWatchChannelMatch.model_validate(
                item.model_dump(mode="python", warnings=False)
                if type(item) is KnowledgeSemanticWatchChannelMatch
                else item
            )
            for item in value
        )
        if not matches or len(matches) != len({item.channel for item in matches}):
            raise ValueError("Channel matches must contain unique lane evidence.")
        supported = {KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL}
        if not {item.channel for item in matches} <= supported:
            raise ValueError("Semantic-watch candidates accept only knowledge recall lanes.")
        return tuple(sorted(matches, key=lambda item: item.channel))

    @model_validator(mode="after")
    def validate_reference_identity(self) -> KnowledgeSemanticWatchCandidate:
        if self.record_type == "knowledge_entry" and self.record_id != self.reference.entry_id:
            raise ValueError("Knowledge-entry watch identity conflicts with its revision.")
        return self


class KnowledgeSemanticWatchEvidence(_WatchModel):
    """Safe bounded projection of one exact recall result."""

    schema_version: Literal[1] = 1
    engine_version: str
    situation_sha256: str
    fusion: RetrievalFusionDiagnostics
    sources: tuple[RecallSourceDiagnostic, ...]
    required_channels: tuple[str, ...]
    candidates: tuple[KnowledgeSemanticWatchCandidate, ...]
    recall_candidate_count: int
    omitted_candidate_count: int
    complete: bool
    truncation_reasons: tuple[str, ...] = ()

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("engine_version")
    @classmethod
    def validate_engine_version(cls, value: str) -> str:
        if value != RECALL_ENGINE_VERSION:
            raise ValueError(f"`engine_version` must be {RECALL_ENGINE_VERSION!r}.")
        return value

    @field_validator("situation_sha256")
    @classmethod
    def validate_situation_sha256(cls, value: str) -> str:
        return _sha256(value, "situation_sha256")

    @field_validator("fusion", mode="before")
    @classmethod
    def copy_fusion(cls, value: object) -> object:
        if type(value) is RetrievalFusionDiagnostics:
            return value.model_dump(mode="python", warnings=False)
        return value

    @field_validator("sources", mode="before")
    @classmethod
    def copy_sources(cls, value: object) -> tuple[RecallSourceDiagnostic, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("`sources` must be a sequence.")
        return tuple(
            RecallSourceDiagnostic.model_validate(
                item.model_dump(mode="python", warnings=False)
                if type(item) is RecallSourceDiagnostic
                else item
            )
            for item in value
        )

    @field_validator("required_channels", mode="before")
    @classmethod
    def validate_required_channels(cls, value: object) -> tuple[str, ...]:
        return _required_channels(value)

    @field_validator("candidates", mode="before")
    @classmethod
    def copy_candidates(cls, value: object) -> tuple[KnowledgeSemanticWatchCandidate, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("`candidates` must be a sequence.")
        candidates = tuple(
            KnowledgeSemanticWatchCandidate.model_validate(
                item.model_dump(mode="python", warnings=False)
                if type(item) is KnowledgeSemanticWatchCandidate
                else item
            )
            for item in value
        )
        if len(candidates) > MAX_KNOWLEDGE_SEMANTIC_WATCH_CANDIDATES:
            raise ValueError("Semantic-watch evidence exceeds its candidate bound.")
        return candidates

    @field_validator("recall_candidate_count", "omitted_candidate_count", mode="before")
    @classmethod
    def validate_counts(cls, value: object, info) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"`{info.field_name}` must be a non-negative integer.")
        return value

    @field_validator("complete", mode="before")
    @classmethod
    def validate_complete(cls, value: object) -> bool:
        if type(value) is not bool:
            raise ValueError("`complete` must be a boolean.")
        return value

    @field_validator("truncation_reasons", mode="before")
    @classmethod
    def validate_truncation_reasons(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            raise ValueError("`truncation_reasons` must be a sequence.")
        reasons = tuple(sorted({_clean(item, "truncation_reasons item") for item in value}))
        if len(reasons) > 64:
            raise ValueError("Semantic-watch evidence has too many truncation reasons.")
        return reasons

    @model_validator(mode="after")
    def validate_evidence(self) -> KnowledgeSemanticWatchEvidence:
        if self.recall_candidate_count != len(self.candidates) + self.omitted_candidate_count:
            raise ValueError("Semantic-watch candidate counts do not reconcile.")
        if (
            self.recall_candidate_count != self.fusion.unique_candidate_count
            or self.fusion.returned_candidate_count + self.fusion.omitted_candidate_count
            != self.fusion.unique_candidate_count
            or len(self.candidates) > self.fusion.returned_candidate_count
            or self.fusion.omitted_candidate_count > self.omitted_candidate_count
        ):
            raise ValueError("Semantic-watch candidate counts conflict with fusion diagnostics.")
        if self.fusion.truncated != bool(self.fusion.truncation_reasons):
            raise ValueError("Semantic-watch fusion truncation diagnostics do not reconcile.")
        if not set(self.fusion.truncation_reasons).issubset(self.truncation_reasons):
            raise ValueError("Semantic-watch evidence omitted fusion truncation reasons.")
        ranks = tuple(candidate.fused_rank for candidate in self.candidates)
        if ranks != tuple(range(1, len(ranks) + 1)):
            raise ValueError("Semantic-watch candidates must retain contiguous fused ranks.")
        identities = tuple(
            (item.record_type, item.record_id, item.reference.entry_id, item.reference.revision)
            for item in self.candidates
        )
        if len(identities) != len(set(identities)):
            raise ValueError("Semantic-watch evidence cannot repeat a recall record.")
        channel_names = tuple(item.channel for item in self.fusion.channels)
        supported_channels = {KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL}
        if (
            not channel_names
            or channel_names != tuple(sorted(channel_names))
            or len(channel_names) != len(set(channel_names))
            or not set(channel_names) <= supported_channels
        ):
            raise ValueError("Semantic-watch fusion channels must be unique and ordered.")
        channel_diagnostics = {item.channel: item for item in self.fusion.channels}
        for diagnostic in channel_diagnostics.values():
            if (
                diagnostic.candidate_limit < 1
                or diagnostic.hit_count < 0
                or diagnostic.unique_candidate_count != diagnostic.hit_count
                or diagnostic.hit_count > diagnostic.candidate_limit
                or (diagnostic.continuation_available and not diagnostic.truncated)
            ):
                raise ValueError("Semantic-watch channel diagnostics are inconsistent.")
        continuation_channels = tuple(self.fusion.continuation_channels)
        if continuation_channels != tuple(sorted(set(continuation_channels))) or any(
            channel not in channel_diagnostics
            or not channel_diagnostics[channel].continuation_available
            for channel in continuation_channels
        ):
            raise ValueError("Semantic-watch continuation diagnostics are inconsistent.")
        source_names = tuple(item.source for item in self.sources)
        source_channels = tuple(channel for item in self.sources for channel in item.channels)
        if (
            source_names != tuple(sorted(source_names))
            or len(source_names) != len(set(source_names))
            or len(source_channels) != len(set(source_channels))
            or set(source_channels) != set(channel_diagnostics)
            or any(
                source.status is not RecallSourceStatus.COMPLETE
                and not any(channel_diagnostics[channel].truncated for channel in source.channels)
                for source in self.sources
            )
        ):
            raise ValueError("Semantic-watch source diagnostics conflict with fusion coverage.")
        for candidate in self.candidates:
            if candidate.best_rank != min(match.rank for match in candidate.channel_matches):
                raise ValueError("Semantic-watch candidate best rank conflicts with its lanes.")
            for match in candidate.channel_matches:
                diagnostic = channel_diagnostics.get(match.channel)
                if (
                    diagnostic is None
                    or match.index_version != diagnostic.index_version
                    or match.rank > diagnostic.candidate_limit
                ):
                    raise ValueError("Semantic-watch candidate lane evidence is inconsistent.")
        lane_ranks = tuple(
            (match.channel, match.rank)
            for candidate in self.candidates
            for match in candidate.channel_matches
        )
        if len(lane_ranks) != len(set(lane_ranks)):
            raise ValueError("Semantic-watch candidate lane ranks must be unique.")
        projected_channel_counts = {
            channel: sum(
                match.channel == channel
                for candidate in self.candidates
                for match in candidate.channel_matches
            )
            for channel in channel_diagnostics
        }
        if any(
            projected_channel_counts[channel] > diagnostic.hit_count
            for channel, diagnostic in channel_diagnostics.items()
        ):
            raise ValueError("Semantic-watch candidate lanes exceed their hit diagnostics.")
        expected_complete = self.omitted_candidate_count == 0 and all(
            channel in channel_diagnostics and not channel_diagnostics[channel].truncated
            for channel in self.required_channels
        )
        if self.complete != expected_complete:
            raise ValueError("Semantic-watch completeness conflicts with its diagnostics.")
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "contract": "cayu.knowledge-semantic-watch-evidence.v1",
                "evidence": self.model_dump(mode="json"),
            },
            "knowledge semantic watch evidence fingerprint",
        )


def _require_evidence_invocation_binding(
    invocation: KnowledgeSemanticWatchInvocation,
    evidence: KnowledgeSemanticWatchEvidence,
) -> None:
    if evidence.required_channels != invocation.required_channels:
        raise ValueError("Semantic-watch evidence conflicts with its required recall lanes.")
    if len(evidence.candidates) > invocation.max_candidates:
        raise ValueError("Semantic-watch evidence exceeds its invocation candidate bound.")


def _invocation_matches_config(
    invocation: KnowledgeSemanticWatchInvocation,
    config: KnowledgeSemanticWatchConfig,
) -> bool:
    return (
        invocation.watch_identity == config.watch_identity
        and invocation.watch_version == config.watch_version
        and invocation.recall_profile_identity == config.recall_profile_identity
        and invocation.recall_profile_version == config.recall_profile_version
        and invocation.policy_identity == config.policy_identity
        and invocation.policy_version == config.policy_version
        and invocation.knowledge_namespace == config.knowledge_namespace
        and invocation.knowledge_aspect_groups == config.knowledge_aspect_groups
        and invocation.required_channels == config.required_channels
        and invocation.max_candidates == config.max_candidates
    )


class KnowledgeSemanticWatchRequest(_WatchModel):
    """Copied bounded policy input; observation text is never persisted by Cayu."""

    schema_version: Literal[1] = 1
    invocation: KnowledgeSemanticWatchInvocation
    observation_text: str
    recall_situation: RecallSituation
    evidence: KnowledgeSemanticWatchEvidence

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("invocation", "evidence", mode="before")
    @classmethod
    def copy_models(cls, value: object) -> object:
        if isinstance(value, KnowledgeSemanticWatchInvocation | KnowledgeSemanticWatchEvidence):
            return value.model_dump(mode="python", warnings=False)
        return value

    @field_validator("recall_situation", mode="before")
    @classmethod
    def copy_recall_situation(cls, value: object) -> object:
        if type(value) is RecallSituation:
            return value.model_dump(mode="python", warnings=False)
        return value

    @field_validator("observation_text")
    @classmethod
    def validate_observation_text(cls, value: str) -> str:
        if type(value) is not str:
            raise ValueError("`observation_text` must be a string.")
        text = require_durable_nonblank(value, "observation_text")
        if len(text.encode("utf-8")) > MAX_KNOWLEDGE_SEMANTIC_WATCH_OBSERVATION_BYTES:
            raise ValueError("Semantic-watch observation exceeds its UTF-8 byte bound.")
        return text

    @model_validator(mode="after")
    def validate_binding(self) -> KnowledgeSemanticWatchRequest:
        encoded = self.observation_text.encode("utf-8")
        if (
            len(encoded) != self.invocation.observation_bytes
            or sha256(encoded).hexdigest() != self.invocation.observation_sha256
        ):
            raise ValueError("Semantic-watch observation conflicts with its invocation.")
        expected_situation = RecallSituation(
            query=self.observation_text,
            knowledge_access_scope=self.invocation.access_scope,
            knowledge_namespace=self.invocation.knowledge_namespace,
            knowledge_aspect_groups=self.invocation.knowledge_aspect_groups,
            current_time=self.recall_situation.current_time,
        )
        if self.recall_situation != expected_situation:
            raise ValueError("Semantic-watch recall situation conflicts with its invocation.")
        if self.evidence.situation_sha256 != expected_situation.fingerprint():
            raise ValueError("Semantic-watch evidence conflicts with its recall situation.")
        _require_evidence_invocation_binding(self.invocation, self.evidence)
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge semantic watch request",
                )
            )
            > MAX_KNOWLEDGE_SEMANTIC_WATCH_POLICY_REQUEST_BYTES
        ):
            raise ValueError("Semantic-watch policy request exceeds its byte ceiling.")
        return self

    @property
    def fingerprint(self) -> str:
        # The invocation already binds the observation hash and length.  Excluding
        # raw observation text lets the persistence boundary reconstruct this
        # identity without retaining private input material.
        return knowledge_semantic_watch_request_fingerprint(self.invocation, self.evidence)


class KnowledgeSemanticWatchDecision(_WatchModel):
    """Exact application-policy route for one watch request."""

    schema_version: Literal[1] = 1
    request_sha256: str
    disposition: KnowledgeSemanticWatchDisposition
    policy_identity: str
    policy_version: str
    code: str
    annotations: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("request_sha256")
    @classmethod
    def validate_request_sha256(cls, value: str) -> str:
        return _sha256(value, "request_sha256")

    @field_validator("policy_identity", "policy_version", "code")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _clean(value, info.field_name)

    @field_validator("annotations", mode="before")
    @classmethod
    def copy_annotations(cls, value: object) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "annotations")
        if len(canonical_durable_json_bytes(copied, "annotations")) > (
            MAX_KNOWLEDGE_SEMANTIC_WATCH_ANNOTATION_BYTES
        ):
            raise ValueError("Semantic-watch annotations exceed their byte ceiling.")
        return copied

    @field_validator("annotations")
    @classmethod
    def freeze_annotations(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        frozen = freeze_json_value(dict(value))
        if type(frozen) is not FrozenJsonDict:  # pragma: no cover - defensive invariant
            raise AssertionError("Semantic-watch annotations did not freeze as an object.")
        return frozen

    @field_serializer("annotations")
    def serialize_annotations(self, value: Mapping[str, Any]) -> dict[str, Any]:
        thawed = thaw_json_value(value)
        if type(thawed) is not dict:  # pragma: no cover - defensive invariant
            raise AssertionError("Semantic-watch annotations did not thaw as an object.")
        return thawed


class KnowledgeSemanticWatchAuthority(_WatchModel):
    """Durable-safe request evidence and accepted application decision."""

    schema_version: Literal[1] = 1
    invocation: KnowledgeSemanticWatchInvocation
    evidence: KnowledgeSemanticWatchEvidence
    decision: KnowledgeSemanticWatchDecision

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("invocation", "evidence", "decision", mode="before")
    @classmethod
    def copy_models(cls, value: object) -> object:
        if isinstance(
            value,
            KnowledgeSemanticWatchInvocation
            | KnowledgeSemanticWatchEvidence
            | KnowledgeSemanticWatchDecision,
        ):
            return value.model_dump(mode="python", warnings=False)
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> KnowledgeSemanticWatchAuthority:
        _require_evidence_invocation_binding(self.invocation, self.evidence)
        expected = knowledge_semantic_watch_request_fingerprint(self.invocation, self.evidence)
        if self.decision.request_sha256 != expected:
            raise ValueError("Semantic-watch decision does not bind its request.")
        if self.decision.policy_identity in {
            self.invocation.watch_identity,
            self.invocation.recall_profile_identity,
        }:
            raise ValueError("Watch or recall output cannot authorize its own route.")
        if (
            self.decision.policy_identity != self.invocation.policy_identity
            or self.decision.policy_version != self.invocation.policy_version
        ):
            raise ValueError("Semantic-watch decision conflicts with its policy profile.")
        if not self.evidence.complete and (
            self.decision.disposition is not KnowledgeSemanticWatchDisposition.ROUTE_TO_REVIEW
        ):
            raise ValueError("Incomplete watch evidence can only route to review.")
        if (
            self.decision.disposition is KnowledgeSemanticWatchDisposition.EMIT
            and not self.evidence.candidates
        ):
            raise ValueError("A semantic-watch signal requires exact match evidence.")
        return self


class KnowledgeSemanticWatchReceipt(_WatchModel):
    """Store-authored immutable outcome for one exact watch invocation."""

    schema_version: Literal[1] = 1
    operation_id: str
    invocation_sha256: str
    request_sha256: str
    authority: KnowledgeSemanticWatchAuthority
    committed_at: datetime
    replayed: bool = False

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("`schema_version` must be the integer 1.")
        return value

    @field_validator("operation_id")
    @classmethod
    def validate_operation_id(cls, value: str) -> str:
        return _clean(value, "operation_id")

    @field_validator("invocation_sha256", "request_sha256")
    @classmethod
    def validate_sha256(cls, value: str, info) -> str:
        return _sha256(value, info.field_name)

    @field_validator("authority", mode="before")
    @classmethod
    def copy_authority(cls, value: object) -> object:
        if type(value) is KnowledgeSemanticWatchAuthority:
            return value.model_dump(mode="python", warnings=False)
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
    def validate_receipt(self) -> KnowledgeSemanticWatchReceipt:
        if (
            self.operation_id != self.authority.invocation.operation_id
            or self.invocation_sha256 != self.authority.invocation.fingerprint
            or self.request_sha256 != self.authority.decision.request_sha256
        ):
            raise ValueError("Semantic-watch receipt does not bind its authority.")
        if (
            len(
                canonical_durable_json_bytes(
                    self.model_dump(mode="json"),
                    "knowledge semantic watch receipt",
                )
            )
            > MAX_KNOWLEDGE_SEMANTIC_WATCH_RECEIPT_BYTES
        ):
            raise ValueError("Semantic-watch receipt exceeds its byte ceiling.")
        return self


class KnowledgeSemanticWatchPolicy(Protocol):
    """Application authority for one copied, bounded watch request."""

    async def decide_semantic_watch(
        self,
        request: KnowledgeSemanticWatchRequest,
    ) -> KnowledgeSemanticWatchDecision: ...


class KnowledgeSemanticWatchPolicyError(RuntimeError):
    """Watch evaluation failed closed before durable routing."""

    def __init__(self, code: str) -> None:
        self.code = _clean(code, "code")
        super().__init__("Knowledge semantic-watch policy failed closed.")


class KnowledgeSemanticWatchConflict(ValueError):
    """A watch operation conflicts with an immutable durable outcome."""

    def __init__(self, code: str) -> None:
        self.code = _clean(code, "code")
        super().__init__(f"Knowledge semantic-watch conflict: {self.code}.")


def knowledge_semantic_watch_request_fingerprint(
    invocation: KnowledgeSemanticWatchInvocation,
    evidence: KnowledgeSemanticWatchEvidence,
) -> str:
    if type(invocation) is not KnowledgeSemanticWatchInvocation:
        raise TypeError("invocation must be a KnowledgeSemanticWatchInvocation.")
    if type(evidence) is not KnowledgeSemanticWatchEvidence:
        raise TypeError("evidence must be KnowledgeSemanticWatchEvidence.")
    return _fingerprint(
        {
            "contract": "cayu.knowledge-semantic-watch-request.v1",
            "invocation": invocation.model_dump(mode="json"),
            "evidence": evidence.model_dump(mode="json"),
        },
        "knowledge semantic watch request fingerprint",
    )


def copy_knowledge_semantic_watch_invocation(
    invocation: KnowledgeSemanticWatchInvocation,
) -> KnowledgeSemanticWatchInvocation:
    if type(invocation) is not KnowledgeSemanticWatchInvocation:
        raise TypeError("Semantic-watch invocations must not be subclasses.")
    return KnowledgeSemanticWatchInvocation.model_validate(
        invocation.model_dump(mode="python", warnings=False)
    )


def copy_knowledge_semantic_watch_evidence(
    evidence: KnowledgeSemanticWatchEvidence,
) -> KnowledgeSemanticWatchEvidence:
    if type(evidence) is not KnowledgeSemanticWatchEvidence:
        raise TypeError("Semantic-watch evidence must not be subclassed.")
    return KnowledgeSemanticWatchEvidence.model_validate(
        evidence.model_dump(mode="python", warnings=False)
    )


def copy_knowledge_semantic_watch_request(
    request: KnowledgeSemanticWatchRequest,
) -> KnowledgeSemanticWatchRequest:
    if type(request) is not KnowledgeSemanticWatchRequest:
        raise TypeError("Semantic-watch requests must not be subclasses.")
    return KnowledgeSemanticWatchRequest.model_validate(
        request.model_dump(mode="python", warnings=False)
    )


def copy_knowledge_semantic_watch_decision(
    decision: KnowledgeSemanticWatchDecision,
) -> KnowledgeSemanticWatchDecision:
    if type(decision) is not KnowledgeSemanticWatchDecision:
        raise TypeError("Semantic-watch decisions must not be subclasses.")
    return KnowledgeSemanticWatchDecision.model_validate(
        decision.model_dump(mode="python", warnings=False)
    )


def copy_knowledge_semantic_watch_authority(
    authority: KnowledgeSemanticWatchAuthority,
) -> KnowledgeSemanticWatchAuthority:
    if type(authority) is not KnowledgeSemanticWatchAuthority:
        raise TypeError("Semantic-watch authorities must not be subclasses.")
    return KnowledgeSemanticWatchAuthority.model_validate(
        authority.model_dump(mode="python", warnings=False)
    )


def copy_knowledge_semantic_watch_receipt(
    receipt: KnowledgeSemanticWatchReceipt,
    *,
    replayed: bool | None = None,
) -> KnowledgeSemanticWatchReceipt:
    if type(receipt) is not KnowledgeSemanticWatchReceipt:
        raise TypeError("Semantic-watch receipts must not be subclasses.")
    values = receipt.model_dump(mode="python", exclude={"replayed"}, warnings=False)
    return KnowledgeSemanticWatchReceipt(
        **values,
        replayed=receipt.replayed if replayed is None else replayed,
    )


def require_knowledge_semantic_watch_authority_records(
    authority: KnowledgeSemanticWatchAuthority,
    records: Sequence[tuple[KnowledgeEntry, Sequence[KnowledgeChunk]]],
    *,
    now: datetime,
) -> KnowledgeSemanticWatchAuthority:
    """Bind candidate identities to exact current, active store material."""

    copied = copy_knowledge_semantic_watch_authority(authority)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware.")
    now = now.astimezone(UTC)
    indexed: dict[tuple[str, int], tuple[KnowledgeEntry, tuple[KnowledgeChunk, ...]]] = {}
    for raw_entry, raw_chunks in records:
        if type(raw_entry) is not KnowledgeEntry:
            raise TypeError("Semantic-watch records must contain exact KnowledgeEntry values.")
        if isinstance(raw_chunks, str | bytes) or not isinstance(raw_chunks, Sequence):
            raise TypeError("Semantic-watch record chunks must be a sequence.")
        entry = KnowledgeEntry.model_validate(raw_entry.model_dump(mode="python", warnings=False))
        chunks = tuple(
            KnowledgeChunk.model_validate(
                chunk.model_dump(mode="python", warnings=False)
                if type(chunk) is KnowledgeChunk
                else chunk
            )
            for chunk in raw_chunks
        )
        key = (entry.id, entry.revision)
        if key in indexed:
            raise ValueError("Semantic-watch record material cannot repeat a revision.")
        indexed[key] = (entry, chunks)

    expected = {
        (candidate.reference.entry_id, candidate.reference.revision)
        for candidate in copied.evidence.candidates
    }
    if set(indexed) != expected:
        raise KnowledgeSemanticWatchConflict("candidate_stale")
    for entry, chunks in indexed.values():
        if (
            entry.status is not KnowledgeStatus.ACTIVE
            or (entry.expires_at is not None and entry.expires_at <= now)
            or entry.namespace != copied.invocation.knowledge_namespace
            or any(
                not set(entry.aspects).intersection(group)
                for group in copied.invocation.knowledge_aspect_groups
            )
        ):
            raise KnowledgeSemanticWatchConflict("candidate_stale")
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        if len(chunks_by_id) != len(chunks):
            raise KnowledgeSemanticWatchConflict("candidate_stale")
        for candidate in copied.evidence.candidates:
            if candidate.reference != KnowledgeRevisionRef(
                entry_id=entry.id,
                revision=entry.revision,
            ):
                continue
            if candidate.record_type == "knowledge_entry":
                valid = (
                    candidate.record_id == entry.id
                    and candidate.representation == "entry_text"
                    and candidate.content_hash == sha256(entry.text.encode("utf-8")).hexdigest()
                )
            else:
                chunk = chunks_by_id.get(candidate.record_id)
                valid = (
                    chunk is not None
                    and chunk.entry_id == entry.id
                    and chunk.entry_revision == entry.revision
                    and candidate.representation == "chunk_text"
                    and candidate.content_hash == sha256(chunk.text.encode("utf-8")).hexdigest()
                )
            if not valid:
                raise KnowledgeSemanticWatchConflict("candidate_stale")
    return copied


def prepare_knowledge_semantic_watch_invocation(
    *,
    operation_id: str,
    observation_id: str,
    observation_source_type: str,
    observation_source_id: str,
    observation_text: str,
    access_scope: KnowledgeAccessScope,
    config: KnowledgeSemanticWatchConfig,
) -> KnowledgeSemanticWatchInvocation:
    if type(config) is not KnowledgeSemanticWatchConfig:
        raise TypeError("config must be a KnowledgeSemanticWatchConfig.")
    if type(access_scope) is not KnowledgeAccessScope:
        raise TypeError("access_scope must be a KnowledgeAccessScope.")
    if type(observation_text) is not str:
        raise TypeError("observation_text must be a string.")
    text = require_durable_nonblank(observation_text, "observation_text")
    encoded = text.encode("utf-8")
    return KnowledgeSemanticWatchInvocation(
        operation_id=operation_id,
        observation_id=observation_id,
        observation_source_type=observation_source_type,
        observation_source_id=observation_source_id,
        observation_sha256=sha256(encoded).hexdigest(),
        observation_bytes=len(encoded),
        watch_identity=config.watch_identity,
        watch_version=config.watch_version,
        recall_profile_identity=config.recall_profile_identity,
        recall_profile_version=config.recall_profile_version,
        policy_identity=config.policy_identity,
        policy_version=config.policy_version,
        knowledge_namespace=config.knowledge_namespace,
        knowledge_aspect_groups=config.knowledge_aspect_groups,
        required_channels=config.required_channels,
        max_candidates=config.max_candidates,
        access_scope=copy_knowledge_access_scope(access_scope),
    )


def project_knowledge_semantic_watch_evidence(
    result: RecallResult,
    *,
    max_candidates: int,
    required_channels: Sequence[str] = (KNOWLEDGE_LEXICAL_CHANNEL,),
) -> KnowledgeSemanticWatchEvidence:
    if type(result) is not RecallResult:
        raise TypeError("result must be a RecallResult.")
    if type(max_candidates) is not int or not (
        1 <= max_candidates <= MAX_KNOWLEDGE_SEMANTIC_WATCH_CANDIDATES
    ):
        raise ValueError("max_candidates is outside the semantic-watch bound.")
    required = _required_channels(required_channels)
    selected = tuple(islice(result.candidates, max_candidates))
    projected: list[KnowledgeSemanticWatchCandidate] = []
    for fused_rank, candidate in enumerate(selected, start=1):
        identity = candidate.record.identity
        locator: Mapping[str, Any] = candidate.record.locator
        entry_id = locator.get("entry_id")
        entry_revision = locator.get("entry_revision")
        if (
            identity.record_type not in {"knowledge_entry", "knowledge_chunk"}
            or type(entry_id) is not str
            or type(entry_revision) is not int
            or identity.revision != str(entry_revision)
        ):
            raise ValueError("Semantic watches accept only exact knowledge recall candidates.")
        record_type: Literal["knowledge_entry", "knowledge_chunk"] = (
            "knowledge_entry" if identity.record_type == "knowledge_entry" else "knowledge_chunk"
        )
        projected.append(
            KnowledgeSemanticWatchCandidate(
                reference=KnowledgeRevisionRef(entry_id=entry_id, revision=entry_revision),
                record_type=record_type,
                record_id=identity.record_id,
                representation=candidate.record.representation,
                content_hash=candidate.record.content_hash,
                fused_rank=fused_rank,
                fused_score=candidate.fused.score,
                best_rank=candidate.fused.best_rank,
                channel_matches=tuple(
                    KnowledgeSemanticWatchChannelMatch(
                        channel=match.channel,
                        index_version=match.index_version,
                        rank=match.rank,
                        reasons=match.explanations,
                        raw_score=match.raw_score,
                    )
                    for match in candidate.fused.matches
                ),
            )
        )
    watch_omitted = len(result.candidates) - len(projected)
    omitted = watch_omitted + result.omitted_by_result_bytes + result.fusion.omitted_candidate_count
    reasons = list(result.fusion.truncation_reasons)
    if result.omitted_by_result_bytes:
        reasons.append("recall_result_bytes")
    if watch_omitted:
        reasons.append("watch_candidate_limit")
    channel_diagnostics = {item.channel: item for item in result.fusion.channels}
    return KnowledgeSemanticWatchEvidence(
        engine_version=result.engine_version,
        situation_sha256=result.situation_sha256,
        fusion=result.fusion,
        sources=result.sources,
        required_channels=required,
        candidates=tuple(projected),
        recall_candidate_count=result.fusion.unique_candidate_count,
        omitted_candidate_count=omitted,
        complete=(
            not omitted
            and all(
                channel in channel_diagnostics and not channel_diagnostics[channel].truncated
                for channel in required
            )
        ),
        truncation_reasons=tuple(reasons),
    )


def _observe_policy_task(task: asyncio.Task[object]) -> None:
    with _RETAINED_POLICY_TASKS_LOCK:
        _RETAINED_POLICY_TASKS.discard(task)
    if not task.cancelled():
        with suppress(BaseException):
            task.exception()


def _start_policy_task(
    decide: Callable[[KnowledgeSemanticWatchRequest], Awaitable[object]],
    request: KnowledgeSemanticWatchRequest,
) -> asyncio.Task[object] | None:
    async def invoke() -> object:
        return await decide(request)

    with _RETAINED_POLICY_TASKS_LOCK:
        if len(_RETAINED_POLICY_TASKS) >= _MAX_RETAINED_POLICY_TASKS:
            return None
        task = asyncio.create_task(invoke(), name="cayu-knowledge-semantic-watch-policy")
        _RETAINED_POLICY_TASKS.add(task)
    task.add_done_callback(_observe_policy_task)
    return task


async def _invoke_policy(
    decide: Callable[[KnowledgeSemanticWatchRequest], Awaitable[object]],
    request: KnowledgeSemanticWatchRequest,
    *,
    timeout_seconds: float,
) -> object:
    await asyncio.sleep(0)
    task = _start_policy_task(decide, request)
    if task is None:
        raise KnowledgeSemanticWatchPolicyError("policy_capacity_exhausted")
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    try:
        await asyncio.wait({task}, timeout=timeout_seconds)
    except asyncio.CancelledError:
        task.cancel("Knowledge semantic-watch policy was cancelled by its caller.")
        raise
    if not task.done() or loop.time() >= deadline:
        if not task.done():
            task.cancel("Knowledge semantic-watch policy exceeded its deadline.")
        raise KnowledgeSemanticWatchPolicyError("policy_timed_out")
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:
        task.cancel("Knowledge semantic-watch policy was cancelled by its caller.")
        raise
    if loop.time() >= deadline:
        raise KnowledgeSemanticWatchPolicyError("policy_timed_out")
    if task.cancelled():
        raise KnowledgeSemanticWatchPolicyError("policy_failed")
    try:
        return task.result()
    except asyncio.CancelledError:
        raise KnowledgeSemanticWatchPolicyError("policy_failed") from None
    except Exception:
        raise KnowledgeSemanticWatchPolicyError("policy_failed") from None


async def decide_knowledge_semantic_watch(
    request: KnowledgeSemanticWatchRequest,
    *,
    config: KnowledgeSemanticWatchConfig,
    policy: KnowledgeSemanticWatchPolicy | None,
) -> KnowledgeSemanticWatchAuthority:
    copied_request = copy_knowledge_semantic_watch_request(request)
    if type(config) is not KnowledgeSemanticWatchConfig:
        raise TypeError("config must be a KnowledgeSemanticWatchConfig.")
    if not _invocation_matches_config(copied_request.invocation, config):
        raise KnowledgeSemanticWatchPolicyError("policy_request_invalid")
    try:
        decide = getattr(policy, "decide_semantic_watch", None)
    except Exception:
        raise KnowledgeSemanticWatchPolicyError("policy_failed") from None
    if not callable(decide):
        raise KnowledgeSemanticWatchPolicyError("policy_missing")
    raw = await _invoke_policy(
        decide,
        copy_knowledge_semantic_watch_request(copied_request),
        timeout_seconds=config.policy_timeout_seconds,
    )
    if type(raw) is not KnowledgeSemanticWatchDecision:
        raise KnowledgeSemanticWatchPolicyError("policy_output_invalid")
    try:
        decision = copy_knowledge_semantic_watch_decision(raw)
        if (
            decision.policy_identity != config.policy_identity
            or decision.policy_version != config.policy_version
        ):
            raise ValueError("Policy identity does not match the watch configuration.")
        return KnowledgeSemanticWatchAuthority(
            invocation=copied_request.invocation,
            evidence=copied_request.evidence,
            decision=decision,
        )
    except (TypeError, ValueError):
        raise KnowledgeSemanticWatchPolicyError("policy_output_invalid") from None


class KnowledgeSemanticWatchEvaluator:
    """Explicit orchestration over existing recall and application policy."""

    def __init__(
        self,
        store: KnowledgeStore,
        recall_engine: RecallEngine,
        *,
        config: KnowledgeSemanticWatchConfig,
        policy: KnowledgeSemanticWatchPolicy | None,
    ) -> None:
        if not isinstance(store, KnowledgeStore):
            raise TypeError("store must be a KnowledgeStore.")
        if not isinstance(recall_engine, RecallEngine):
            raise TypeError("recall_engine must be a RecallEngine.")
        if type(config) is not KnowledgeSemanticWatchConfig:
            raise TypeError("config must be a KnowledgeSemanticWatchConfig.")
        self._store = store
        self._recall_engine = recall_engine
        self._config = KnowledgeSemanticWatchConfig.model_validate(
            config.model_dump(mode="python", warnings=False)
        )
        self._policy = policy

    @property
    def config(self) -> KnowledgeSemanticWatchConfig:
        return KnowledgeSemanticWatchConfig.model_validate(
            self._config.model_dump(mode="python", warnings=False)
        )

    async def evaluate(
        self,
        *,
        operation_id: str,
        observation_id: str,
        observation_source_type: str,
        observation_source_id: str,
        observation_text: str,
        access_scope: KnowledgeAccessScope | None = None,
    ) -> KnowledgeSemanticWatchReceipt:
        scope = self._store.bound_access_scope() if access_scope is None else access_scope
        if type(scope) is not KnowledgeAccessScope:
            raise KnowledgeSemanticWatchPolicyError("access_scope_missing")
        scope = copy_knowledge_access_scope(scope)
        invocation = prepare_knowledge_semantic_watch_invocation(
            operation_id=operation_id,
            observation_id=observation_id,
            observation_source_type=observation_source_type,
            observation_source_id=observation_source_id,
            observation_text=observation_text,
            access_scope=scope,
            config=self._config,
        )
        existing = await self._store.load_semantic_watch_receipt(
            invocation.operation_id,
            access_scope=scope,
        )
        if existing is not None:
            copied = copy_knowledge_semantic_watch_receipt(existing, replayed=True)
            if copied.authority.invocation != invocation:
                raise KnowledgeSemanticWatchConflict("operation_reuse")
            self._require_policy_identity(copied.authority.decision)
            return copied

        try:
            situation = RecallSituation(
                query=observation_text,
                knowledge_access_scope=scope,
                knowledge_namespace=self._config.knowledge_namespace,
                knowledge_aspect_groups=self._config.knowledge_aspect_groups,
            )
            result = await self._recall_engine.recall(situation)
            if result.situation_sha256 != situation.fingerprint():
                raise ValueError("Recall result conflicts with the submitted situation.")
            evidence = project_knowledge_semantic_watch_evidence(
                result,
                max_candidates=self._config.max_candidates,
                required_channels=self._config.required_channels,
            )
        except RecallSourceUnavailable:
            raise KnowledgeSemanticWatchPolicyError("recall_unavailable") from None
        except asyncio.CancelledError:
            raise
        except Exception:
            raise KnowledgeSemanticWatchPolicyError("recall_failed") from None
        try:
            request = KnowledgeSemanticWatchRequest(
                invocation=invocation,
                observation_text=observation_text,
                recall_situation=situation,
                evidence=evidence,
            )
        except (TypeError, ValueError):
            raise KnowledgeSemanticWatchPolicyError("policy_request_invalid") from None
        authority = await decide_knowledge_semantic_watch(
            request,
            config=self._config,
            policy=self._policy,
        )
        receipt = await self._store.record_semantic_watch_outcome(
            authority,
            access_scope=scope,
        )
        return copy_knowledge_semantic_watch_receipt(receipt)

    def _require_policy_identity(self, decision: KnowledgeSemanticWatchDecision) -> None:
        if (
            decision.policy_identity != self._config.policy_identity
            or decision.policy_version != self._config.policy_version
        ):
            raise KnowledgeSemanticWatchConflict("operation_reuse")


async def load_knowledge_semantic_watch_receipt(
    store: KnowledgeStore,
    *,
    operation_id: str,
    access_scope: KnowledgeAccessScope | None = None,
) -> KnowledgeSemanticWatchReceipt | None:
    receipt = await store.load_semantic_watch_receipt(
        operation_id,
        access_scope=access_scope,
    )
    return None if receipt is None else copy_knowledge_semantic_watch_receipt(receipt)


__all__ = [
    "MAX_KNOWLEDGE_SEMANTIC_WATCH_ANNOTATION_BYTES",
    "MAX_KNOWLEDGE_SEMANTIC_WATCH_CANDIDATES",
    "MAX_KNOWLEDGE_SEMANTIC_WATCH_OBSERVATION_BYTES",
    "MAX_KNOWLEDGE_SEMANTIC_WATCH_POLICY_REQUEST_BYTES",
    "MAX_KNOWLEDGE_SEMANTIC_WATCH_RECEIPT_BYTES",
    "KnowledgeSemanticWatchAuthority",
    "KnowledgeSemanticWatchCandidate",
    "KnowledgeSemanticWatchChannelMatch",
    "KnowledgeSemanticWatchConfig",
    "KnowledgeSemanticWatchConflict",
    "KnowledgeSemanticWatchDecision",
    "KnowledgeSemanticWatchDisposition",
    "KnowledgeSemanticWatchEvaluator",
    "KnowledgeSemanticWatchEvidence",
    "KnowledgeSemanticWatchInvocation",
    "KnowledgeSemanticWatchPolicy",
    "KnowledgeSemanticWatchPolicyError",
    "KnowledgeSemanticWatchReceipt",
    "KnowledgeSemanticWatchRequest",
    "copy_knowledge_semantic_watch_authority",
    "copy_knowledge_semantic_watch_decision",
    "copy_knowledge_semantic_watch_evidence",
    "copy_knowledge_semantic_watch_invocation",
    "copy_knowledge_semantic_watch_receipt",
    "copy_knowledge_semantic_watch_request",
    "decide_knowledge_semantic_watch",
    "knowledge_semantic_watch_request_fingerprint",
    "load_knowledge_semantic_watch_receipt",
    "prepare_knowledge_semantic_watch_invocation",
    "project_knowledge_semantic_watch_evidence",
    "require_knowledge_semantic_watch_authority_records",
]
