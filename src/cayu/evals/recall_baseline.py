from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import ceil
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_label_map,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)
from cayu.core.messages import Message, MessageRole
from cayu.evals.memory_baseline import MemoryRetrievalAccessSpec
from cayu.memory import AutomaticRecallPolicy, admit_recall
from cayu.recall import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    RECALL_ENGINE_VERSION,
    TRANSCRIPT_LEXICAL_CHANNEL,
    KnowledgeRecallSource,
    RecallCandidate,
    RecallEngine,
    RecallSituation,
    RecallSource,
    RecallSourceResult,
    RecallSourceStatus,
    RecallSourceUnavailable,
    TranscriptRecallSource,
)
from cayu.retrieval import (
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    RetrievalCandidateIdentity,
    WeightedReciprocalRankFusionConfig,
)
from cayu.runtime.sessions import (
    TRANSCRIPT_SEARCH_MAX_BYTES,
    TRANSCRIPT_SEARCH_MAX_SCAN_LIMIT,
    TRANSCRIPT_SEARCH_MIN_MAX_BYTES,
    RunRequest,
    SessionIdentity,
    SessionQuery,
    SessionStore,
)
from cayu.storage.memory import (
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeStatus,
    KnowledgeStore,
)

RECALL_BASELINE_CORPUS_SCHEMA_VERSION = "cayu.recall_baseline_corpus.v2"
RECALL_BASELINE_RESULT_SCHEMA_VERSION = "cayu.recall_baseline_result.v2"
_MAX_CORPUS_BYTES = 4 * 1024 * 1024
_MAX_CORPUS_RECORDS = 10_000
_BASELINE_TIME = datetime(2026, 8, 21, 7, 0, tzinfo=UTC)


class RecallBaselineKnowledgeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    revisions: tuple[str, ...]
    namespace: str
    labels: dict[str, str] = Field(default_factory=dict)

    @field_validator("id", "namespace")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("revisions")
    @classmethod
    def validate_revisions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("Knowledge baseline entries require at least one revision.")
        return tuple(
            require_durable_nonblank(text, f"revisions[{index}]")
            for index, text in enumerate(value)
        )

    @field_validator("labels", mode="before")
    @classmethod
    def copy_labels(cls, value) -> dict[str, str]:
        return copy_label_map(value, "labels")


class RecallBaselineTranscriptMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    interaction_id: str
    role: MessageRole
    text: str

    @field_validator("interaction_id")
    @classmethod
    def validate_interaction_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "interaction_id")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: MessageRole) -> MessageRole:
        if value not in {MessageRole.USER, MessageRole.ASSISTANT}:
            raise ValueError("Recall baseline transcripts allow only narrative roles.")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return require_durable_nonblank(value, "text")


class RecallBaselineTranscript(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: str
    messages: tuple[RecallBaselineTranscriptMessage, ...]

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "session_id")

    @field_validator("messages")
    @classmethod
    def validate_messages(
        cls,
        value: tuple[RecallBaselineTranscriptMessage, ...],
    ) -> tuple[RecallBaselineTranscriptMessage, ...]:
        if not value:
            raise ValueError("Recall baseline transcripts cannot be empty.")
        return value


class RecallBaselineExpectedIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    record_type: Literal["knowledge_chunk", "knowledge_entry", "transcript_message"]
    record_id: str
    revision: str | None = None
    locator: dict[str, Any] = Field(default_factory=dict)
    expected_disposition: Literal["inject", "offer", "silent"] | None = None

    @field_validator("record_id")
    @classmethod
    def validate_record_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "record_id")

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "revision")

    @field_validator("locator", mode="before")
    @classmethod
    def copy_locator(cls, value) -> dict[str, Any]:
        return copy_durable_json_object(value, "locator")

    @model_validator(mode="after")
    def validate_locator(self) -> RecallBaselineExpectedIdentity:
        if self.record_type == "knowledge_chunk":
            expected_keys = {"entry_id", "entry_revision", "chunk_id", "chunk_index"}
            if set(self.locator) != expected_keys or self.locator["chunk_id"] != self.record_id:
                raise ValueError("Knowledge chunk expectations require one exact chunk locator.")
            if self.revision != str(self.locator["entry_revision"]):
                raise ValueError("Knowledge chunk revision and locator must agree.")
        elif self.record_type == "knowledge_entry":
            if set(self.locator) != {"entry_id", "entry_revision"}:
                raise ValueError("Knowledge entry expectations require one exact entry locator.")
            if self.locator["entry_id"] != self.record_id or self.revision != str(
                self.locator["entry_revision"]
            ):
                raise ValueError("Knowledge entry identity and locator must agree.")
        else:
            expected_keys = {
                "session_id",
                "interaction_id",
                "transcript_index",
                "text_part_indexes",
            }
            if set(self.locator) != expected_keys or self.record_id != (
                f"{self.locator['session_id']}:{self.locator['transcript_index']}"
            ):
                raise ValueError("Transcript expectations require one exact message locator.")
        return self


class RecallBaselineCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    id: str
    language: str
    query: str
    recent_conversation: tuple[str, ...] = ()
    work_context: str | None = None
    namespace: str
    access: MemoryRetrievalAccessSpec
    transcript_session_ids: tuple[str, ...] = ()
    relevant: tuple[RecallBaselineExpectedIdentity, ...]
    forbidden: tuple[RecallBaselineExpectedIdentity, ...] = ()
    expect_partial_coverage: bool = True
    cutoff: int = 5
    transcript_max_bytes: int = 64_000
    transcript_max_records_scanned: int = 10_000
    admission_max_injected_items: int = 5

    @field_validator("id", "language", "namespace")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        return require_durable_nonblank(value, "query")

    @field_validator("work_context")
    @classmethod
    def validate_work_context(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_nonblank(value, "work_context")

    @field_validator("cutoff", mode="before")
    @classmethod
    def validate_cutoff(cls, value) -> int:
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError("Recall baseline cutoff must be between 1 and 100.")
        return value

    @field_validator("admission_max_injected_items", mode="before")
    @classmethod
    def validate_admission_max_injected_items(cls, value) -> int:
        if type(value) is not int or not 1 <= value <= 50:
            raise ValueError(
                "Recall baseline admission_max_injected_items must be between 1 and 50."
            )
        return value

    @field_validator(
        "transcript_max_bytes",
        "transcript_max_records_scanned",
        mode="before",
    )
    @classmethod
    def validate_transcript_bound(cls, value, info) -> int:
        if type(value) is not int:
            raise ValueError(f"Recall baseline {info.field_name} must be an integer.")
        minimum, maximum = (
            (TRANSCRIPT_SEARCH_MIN_MAX_BYTES, TRANSCRIPT_SEARCH_MAX_BYTES)
            if info.field_name == "transcript_max_bytes"
            else (1, TRANSCRIPT_SEARCH_MAX_SCAN_LIMIT)
        )
        if not minimum <= value <= maximum:
            raise ValueError(
                f"Recall baseline {info.field_name} must be between {minimum} and {maximum}."
            )
        return value

    @model_validator(mode="after")
    def validate_expectations(self) -> RecallBaselineCase:
        if not self.relevant:
            raise ValueError("Recall baseline cases require relevant identities.")
        relevant = {(item.record_type, item.record_id) for item in self.relevant}
        forbidden = {(item.record_type, item.record_id) for item in self.forbidden}
        if len(relevant) != len(self.relevant):
            raise ValueError("Recall baseline relevant identities must be unique.")
        if relevant & forbidden:
            raise ValueError("Relevant and forbidden recall identities cannot overlap.")
        if any(item.expected_disposition is None for item in self.relevant):
            raise ValueError("Relevant identities require an admission disposition.")
        if any(item.expected_disposition is not None for item in self.forbidden):
            raise ValueError("Forbidden identities cannot declare an admission disposition.")
        return self


class RecallBaselineCorpus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.recall_baseline_corpus.v2"]
    corpus_revision: str
    origin: Literal["hermetic_public", "external_private"]
    knowledge: tuple[RecallBaselineKnowledgeEntry, ...]
    transcripts: tuple[RecallBaselineTranscript, ...]
    cases: tuple[RecallBaselineCase, ...]

    @field_validator("corpus_revision")
    @classmethod
    def validate_corpus_revision(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "corpus_revision")

    @model_validator(mode="after")
    def validate_corpus(self) -> RecallBaselineCorpus:
        total = len(self.knowledge) + len(self.transcripts) + len(self.cases)
        if total == 0 or total > _MAX_CORPUS_RECORDS:
            raise ValueError(
                f"Recall baseline corpus must contain 1..{_MAX_CORPUS_RECORDS} records."
            )
        knowledge_ids = [entry.id for entry in self.knowledge]
        session_ids = [transcript.session_id for transcript in self.transcripts]
        case_ids = [case.id for case in self.cases]
        if len(knowledge_ids) != len(set(knowledge_ids)):
            raise ValueError("Recall baseline knowledge ids must be unique.")
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("Recall baseline session ids must be unique.")
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise ValueError("Recall baseline case ids must be non-empty and unique.")
        known_sessions = set(session_ids)
        if any(
            session_id not in known_sessions
            for case in self.cases
            for session_id in case.transcript_session_ids
        ):
            raise ValueError("Recall baseline case references an unknown permitted session.")
        return self


class RecallBaselineCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    case_id: str
    language: str
    selected_identities: tuple[str, ...]
    candidate_count: int
    recall_at_k: float
    false_result_count: int
    stale_knowledge_count: int
    authorization_leak_count: int
    locator_correct_count: int
    locator_evaluated_count: int
    partial_source_count: int
    source_statuses: dict[str, RecallSourceStatus]
    source_failure_codes: dict[str, str | None]
    channel_index_versions: dict[str, str]
    continuation_channels: tuple[str, ...]
    false_complete: bool
    truncated: bool
    result_bytes: int
    latency_ms: float
    injected_identities: tuple[str, ...]
    offered_identities: tuple[str, ...]
    silent_identities: tuple[str, ...]
    correct_injected_count: int
    correct_offered_count: int
    correct_silent_count: int
    false_injection_count: int
    stale_injection_count: int
    unauthorized_injection_count: int
    injected_source_diversity: int
    contribution_bytes: int
    estimated_contribution_tokens: int
    admission_truncated: bool
    admission_latency_ms: float


class RecallBaselineMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    case_count: int
    recall_at_k: float
    false_result_rate: float
    stale_knowledge_rate: float
    authorization_leak_rate: float
    locator_correctness: float
    false_complete_rate: float
    truncated_case_count: int
    partial_source_count: int
    total_candidate_count: int
    mean_candidate_count: float
    result_bytes: int
    estimated_result_tokens: int
    latency_p50_ms: float
    latency_p95_ms: float
    injected_precision: float
    offer_precision: float
    silent_precision: float
    false_injection_rate: float
    stale_injection_rate: float
    unauthorized_injection_rate: float
    mean_injected_source_diversity: float
    contribution_bytes: int
    estimated_contribution_tokens: int
    admission_truncated_case_count: int
    admission_latency_p50_ms: float
    admission_latency_p95_ms: float
    required_source_failure_closed: bool


class RecallBaselineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal["cayu.recall_baseline_result.v2"]
    corpus_revision: str
    corpus_origin: Literal["hermetic_public", "external_private"]
    backend: str
    configuration: dict[str, Any]
    metrics: RecallBaselineMetrics
    cases: tuple[RecallBaselineCaseResult, ...]

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "backend")

    @field_validator("configuration", mode="before")
    @classmethod
    def copy_configuration(cls, value) -> dict[str, Any]:
        return copy_durable_json_object(value, "configuration")


class _RequiredFailureProbeSource(RecallSource):
    name = "required_failure_probe"
    channel_names = ("required_failure_probe.lexical",)

    def __init__(self) -> None:
        super().__init__(required=True, candidate_limit=1)

    async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
        raise RuntimeError("hermetic required-source failure")


def load_recall_baseline_corpus(path: str | Path) -> RecallBaselineCorpus:
    corpus_path = Path(path)
    try:
        with corpus_path.open("rb") as stream:
            raw = stream.read(_MAX_CORPUS_BYTES + 1)
        if len(raw) > _MAX_CORPUS_BYTES:
            raise ValueError(f"Recall baseline corpus exceeds {_MAX_CORPUS_BYTES} bytes.")
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load recall baseline corpus: {exc}") from exc
    return RecallBaselineCorpus.model_validate(payload)


async def run_recall_baseline(
    corpus: RecallBaselineCorpus,
    knowledge_store: KnowledgeStore,
    session_store: SessionStore,
    *,
    backend: str,
    configuration: dict[str, Any] | None = None,
) -> RecallBaselineResult:
    """Seed empty built-in stores and measure deterministic cross-source recall."""

    if type(corpus) is not RecallBaselineCorpus:
        raise TypeError("corpus must be a RecallBaselineCorpus.")
    corpus = RecallBaselineCorpus.model_validate(corpus.model_dump(mode="json"))
    if not isinstance(knowledge_store, KnowledgeStore):
        raise TypeError("knowledge_store must implement KnowledgeStore.")
    if not isinstance(session_store, SessionStore):
        raise TypeError("session_store must implement SessionStore.")
    if not session_store.supports_transcript_search:
        raise ValueError("session_store must support transcript search.")
    backend = require_durable_clean_nonblank(backend, "backend")
    recorded_configuration = copy_durable_json_object(
        {} if configuration is None else configuration,
        "configuration",
    )
    fusion_configuration_version = "public-recall-admission-baseline-v2"
    admission_policies = {
        case.id: AutomaticRecallPolicy(
            calibration_version="public-recall-admission-v1",
            fusion_strategy_version=WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
            fusion_configuration_version=fusion_configuration_version,
            minimum_inject_score=0.0162,
            minimum_offer_score=0.016,
            max_evaluated_candidates=30,
            max_injected_items=case.admission_max_injected_items,
            max_offered_items=10,
            max_candidate_text_bytes=32_000,
            max_focus_bytes=96_000,
            max_offer_bytes=48_000,
            max_total_bytes=192_000,
        )
        for case in corpus.cases
    }
    required_configuration = {
        "engine_version": RECALL_ENGINE_VERSION,
        "fusion_configuration_version": fusion_configuration_version,
        "fusion_strategy_version": WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
        "channel_weights": {
            KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
            KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
            TRANSCRIPT_LEXICAL_CHANNEL: 1.0,
        },
        "cutoffs_by_case": {case.id: case.cutoff for case in corpus.cases},
        "transcript_budgets_by_case": {
            case.id: {
                "max_bytes": case.transcript_max_bytes,
                "max_records_scanned": case.transcript_max_records_scanned,
            }
            for case in corpus.cases
        },
        "source_set": ["knowledge", "transcript"],
        "token_estimator": "ceil(utf8_json_bytes/4)",
        "admission_policy_by_case": {
            case_id: policy.model_dump(mode="json")
            for case_id, policy in admission_policies.items()
        },
    }
    for key, value in required_configuration.items():
        if key in recorded_configuration and recorded_configuration[key] != value:
            raise ValueError(f"configuration cannot override runner-owned field {key!r}.")
        recorded_configuration[key] = value

    privileged = KnowledgeAccessScope.privileged()
    existing_knowledge = await knowledge_store.list_entries(
        KnowledgeListQuery(statuses=list(KnowledgeStatus), include_expired=True, limit=1),
        access_scope=privileged,
    )
    existing_sessions = await session_store.list_sessions(SessionQuery(limit=1))
    if existing_knowledge.entries or existing_sessions.sessions:
        raise ValueError("Recall baseline requires empty knowledge and session stores.")

    current_revisions: dict[str, int] = {}
    expected_content_hashes: dict[tuple[str, str], str] = {}
    for corpus_entry in corpus.knowledge:
        entry = KnowledgeEntry(
            id=corpus_entry.id,
            text=f"Canonical knowledge record {corpus_entry.id} revision 1.",
            namespace=corpus_entry.namespace,
            labels=dict(corpus_entry.labels),
            created_at=_BASELINE_TIME,
            updated_at=_BASELINE_TIME,
        )
        created = await knowledge_store.create_entry(
            entry,
            chunks=[
                KnowledgeChunk(
                    id=f"{entry.id}:r1:0",
                    entry_id=entry.id,
                    entry_revision=1,
                    chunk_index=0,
                    text=corpus_entry.revisions[0],
                )
            ],
            access_scope=privileged,
        )
        for revision, text in enumerate(corpus_entry.revisions[1:], start=2):
            created = await knowledge_store.append_entry_revision(
                created.model_copy(
                    update={
                        "revision": revision,
                        "text": (f"Canonical knowledge record {created.id} revision {revision}."),
                        "updated_at": _BASELINE_TIME + timedelta(seconds=revision - 1),
                    }
                ),
                chunks=[
                    KnowledgeChunk(
                        id=f"{created.id}:r{revision}:0",
                        entry_id=created.id,
                        entry_revision=revision,
                        chunk_index=0,
                        text=text,
                    )
                ],
                expected_revision=revision - 1,
                access_scope=privileged,
            )
        current_revisions[created.id] = created.revision
        expected_content_hashes[("knowledge_entry", created.id)] = sha256(
            created.text.encode("utf-8")
        ).hexdigest()
        for revision, text in enumerate(corpus_entry.revisions, start=1):
            expected_content_hashes[("knowledge_chunk", f"{created.id}:r{revision}:0")] = sha256(
                text.encode("utf-8")
            ).hexdigest()

    for transcript in corpus.transcripts:
        await session_store.create(
            RunRequest(
                agent_name="recall-baseline",
                session_id=transcript.session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="hermetic", model="none"),
        )
        for transcript_index, message in enumerate(transcript.messages):
            await session_store.append_transcript_messages(
                transcript.session_id,
                [Message.text(message.role, message.text)],
                interaction_id=message.interaction_id,
            )
            expected_content_hashes[
                ("transcript_message", f"{transcript.session_id}:{transcript_index}")
            ] = sha256(message.text.encode("utf-8")).hexdigest()

    case_results: list[RecallBaselineCaseResult] = []
    selected_total = 0
    false_results = 0
    stale_knowledge = 0
    knowledge_selected = 0
    authorization_leaks = 0
    authorization_checks = 0
    locator_correct = 0
    locator_evaluated = 0
    false_complete = 0
    injected_predictions = 0
    offered_predictions = 0
    silent_predictions = 0
    correct_injected = 0
    correct_offered = 0
    correct_silent = 0
    false_injections = 0
    stale_injections = 0
    knowledge_injections = 0
    unauthorized_injections = 0
    for case in corpus.cases:
        fusion_config = WeightedReciprocalRankFusionConfig(
            configuration_version=required_configuration["fusion_configuration_version"],
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
                TRANSCRIPT_LEXICAL_CHANNEL: 1.0,
            },
            max_candidates_per_channel=case.cutoff,
            fused_head_limit=case.cutoff,
        )
        engine = RecallEngine(
            (
                KnowledgeRecallSource(knowledge_store, candidate_limit=case.cutoff),
                TranscriptRecallSource(
                    session_store,
                    candidate_limit=case.cutoff,
                    max_bytes=case.transcript_max_bytes,
                    max_records_scanned=case.transcript_max_records_scanned,
                ),
            ),
            fusion_config=fusion_config,
        )
        started_ns = perf_counter_ns()
        result = await engine.recall(
            RecallSituation(
                query=case.query,
                recent_conversation=case.recent_conversation,
                work_context=case.work_context,
                knowledge_access_scope=case.access.to_scope(),
                knowledge_namespace=case.namespace,
                transcript_session_ids=case.transcript_session_ids,
                current_time=_BASELINE_TIME,
            )
        )
        latency_ms = (perf_counter_ns() - started_ns) / 1_000_000
        admission_started_ns = perf_counter_ns()
        contribution = admit_recall(result, admission_policies[case.id])
        admission_latency_ms = (perf_counter_ns() - admission_started_ns) / 1_000_000
        selected = tuple(result.candidates[: case.cutoff])
        selected_total += len(selected)
        selected_keys = {
            (candidate.record.identity.record_type, candidate.record.identity.record_id)
            for candidate in selected
        }
        relevant_keys = {(item.record_type, item.record_id) for item in case.relevant}
        forbidden_keys = {(item.record_type, item.record_id) for item in case.forbidden}
        matched_relevant = selected_keys & relevant_keys
        case_false = len(selected_keys - relevant_keys)
        case_leaks = len(selected_keys & forbidden_keys)
        case_stale = 0
        case_locator_correct = 0
        relevant_by_key: dict[tuple[str, str], RecallBaselineExpectedIdentity] = {
            (item.record_type, item.record_id): item for item in case.relevant
        }
        for candidate in selected:
            identity = candidate.record.identity
            if identity.record_type in {"knowledge_chunk", "knowledge_entry"}:
                knowledge_selected += 1
                entry_id = candidate.record.locator["entry_id"]
                if int(identity.revision) != current_revisions[entry_id]:
                    case_stale += 1
            expected = relevant_by_key.get((identity.record_type, identity.record_id))
            if expected is None:
                continue
            locator_evaluated += 1
            expected_locator = expected.locator
            revision_correct = expected.revision is None or identity.revision == expected.revision
            locator_matches = candidate.record.locator == expected_locator
            expected_hash = expected_content_hashes[(identity.record_type, identity.record_id)]
            hash_correct = candidate.record.content_hash == expected_hash and (
                identity.record_type != "transcript_message" or identity.revision == expected_hash
            )
            if revision_correct and locator_matches and hash_correct:
                case_locator_correct += 1

        injected_candidates = (
            ()
            if contribution.focus is None
            else tuple(item.candidate for item in contribution.focus.items)
        )
        injected_keys = {
            (candidate.record.identity.record_type, candidate.record.identity.record_id)
            for candidate in injected_candidates
        }
        offered_identities = (
            ()
            if contribution.offer is None
            else tuple(item.identity for item in contribution.offer.items)
        )
        offered_keys = {
            (identity.record_type, identity.record_id) for identity in offered_identities
        }
        silent_candidates = tuple(
            candidate
            for candidate in selected
            if (
                candidate.record.identity.record_type,
                candidate.record.identity.record_id,
            )
            not in injected_keys | offered_keys
        )
        expected_dispositions: dict[
            tuple[str, str],
            Literal["inject", "offer", "silent"] | None,
        ] = {
            (item.record_type, item.record_id): item.expected_disposition for item in case.relevant
        }
        case_correct_injected = sum(
            expected_dispositions.get(key) == "inject" for key in injected_keys
        )
        case_correct_offered = sum(
            expected_dispositions.get(key) == "offer" for key in offered_keys
        )
        silent_keys = {
            (candidate.record.identity.record_type, candidate.record.identity.record_id)
            for candidate in silent_candidates
        }
        case_correct_silent = sum(expected_dispositions.get(key) == "silent" for key in silent_keys)
        case_false_injections = len(injected_keys) - case_correct_injected
        case_stale_injections = 0
        for candidate in injected_candidates:
            if candidate.record.identity.record_type not in {
                "knowledge_chunk",
                "knowledge_entry",
            }:
                continue
            knowledge_injections += 1
            entry_id = candidate.record.locator["entry_id"]
            if int(candidate.record.identity.revision) != current_revisions[entry_id]:
                case_stale_injections += 1
        case_unauthorized_injections = len(injected_keys & forbidden_keys)
        injected_predictions += len(injected_keys)
        offered_predictions += len(offered_keys)
        silent_predictions += len(silent_keys)
        correct_injected += case_correct_injected
        correct_offered += case_correct_offered
        correct_silent += case_correct_silent
        false_injections += case_false_injections
        stale_injections += case_stale_injections
        unauthorized_injections += case_unauthorized_injections
        locator_correct += case_locator_correct
        authorization_checks += len(forbidden_keys)
        for forbidden in case.forbidden:
            if forbidden.record_type not in {"knowledge_chunk", "knowledge_entry"}:
                continue
            authorization_checks += 2
            entry_id = forbidden.locator["entry_id"]
            scope = case.access.to_scope()
            if await knowledge_store.get_entry(entry_id, access_scope=scope) is not None:
                case_leaks += 1
            if await knowledge_store.read_chunks(entry_id, access_scope=scope):
                case_leaks += 1
        false_results += case_false
        stale_knowledge += case_stale
        authorization_leaks += case_leaks
        partial_source_count = sum(
            source.status is not RecallSourceStatus.COMPLETE for source in result.sources
        )
        case_false_complete = case.expect_partial_coverage and partial_source_count == 0
        false_complete += case_false_complete
        result_bytes = len(
            canonical_durable_json_bytes(result.model_dump(mode="json"), "recall baseline result")
        )
        contribution_bytes = len(
            canonical_durable_json_bytes(
                contribution.model_dump(mode="json"),
                "automatic recall baseline contribution",
            )
        )
        injected_source_types = {
            _candidate_source_type(candidate) for candidate in injected_candidates
        }
        case_results.append(
            RecallBaselineCaseResult(
                case_id=case.id,
                language=case.language,
                selected_identities=tuple(
                    ":".join(candidate.record.identity.sort_key()) for candidate in selected
                ),
                candidate_count=result.fusion.unique_candidate_count,
                recall_at_k=len(matched_relevant) / len(relevant_keys),
                false_result_count=case_false,
                stale_knowledge_count=case_stale,
                authorization_leak_count=case_leaks,
                locator_correct_count=case_locator_correct,
                locator_evaluated_count=len(matched_relevant),
                partial_source_count=partial_source_count,
                source_statuses={source.source: source.status for source in result.sources},
                source_failure_codes={
                    source.source: source.failure_code for source in result.sources
                },
                channel_index_versions={
                    channel.channel: channel.index_version for channel in result.fusion.channels
                },
                continuation_channels=result.fusion.continuation_channels,
                false_complete=case_false_complete,
                truncated=result.truncated,
                result_bytes=result_bytes,
                latency_ms=latency_ms,
                injected_identities=tuple(
                    _identity_label(candidate.record.identity) for candidate in injected_candidates
                ),
                offered_identities=tuple(
                    _identity_label(identity) for identity in offered_identities
                ),
                silent_identities=tuple(
                    _identity_label(candidate.record.identity) for candidate in silent_candidates
                ),
                correct_injected_count=case_correct_injected,
                correct_offered_count=case_correct_offered,
                correct_silent_count=case_correct_silent,
                false_injection_count=case_false_injections,
                stale_injection_count=case_stale_injections,
                unauthorized_injection_count=case_unauthorized_injections,
                injected_source_diversity=len(injected_source_types),
                contribution_bytes=contribution_bytes,
                estimated_contribution_tokens=ceil(contribution_bytes / 4),
                admission_truncated=contribution.diagnostics.admission_truncated,
                admission_latency_ms=admission_latency_ms,
            )
        )

    result_bytes = sum(case.result_bytes for case in case_results)
    latencies = [case.latency_ms for case in case_results]
    admission_latencies = [case.admission_latency_ms for case in case_results]
    total_candidates = sum(case.candidate_count for case in case_results)
    contribution_bytes = sum(case.contribution_bytes for case in case_results)
    required_source_failure_closed = await _required_source_failure_closed()
    metrics = RecallBaselineMetrics(
        case_count=len(case_results),
        recall_at_k=_mean(case.recall_at_k for case in case_results),
        false_result_rate=false_results / selected_total if selected_total else 0.0,
        stale_knowledge_rate=(stale_knowledge / knowledge_selected if knowledge_selected else 0.0),
        authorization_leak_rate=(
            authorization_leaks / authorization_checks if authorization_checks else 0.0
        ),
        locator_correctness=(locator_correct / locator_evaluated if locator_evaluated else 0.0),
        false_complete_rate=false_complete / len(case_results),
        truncated_case_count=sum(case.truncated for case in case_results),
        partial_source_count=sum(case.partial_source_count for case in case_results),
        total_candidate_count=total_candidates,
        mean_candidate_count=total_candidates / len(case_results),
        result_bytes=result_bytes,
        estimated_result_tokens=ceil(result_bytes / 4),
        latency_p50_ms=_nearest_rank_percentile(latencies, 0.50),
        latency_p95_ms=_nearest_rank_percentile(latencies, 0.95),
        injected_precision=_precision(correct_injected, injected_predictions),
        offer_precision=_precision(correct_offered, offered_predictions),
        silent_precision=_precision(correct_silent, silent_predictions),
        false_injection_rate=(
            false_injections / injected_predictions if injected_predictions else 0.0
        ),
        stale_injection_rate=(
            stale_injections / knowledge_injections if knowledge_injections else 0.0
        ),
        unauthorized_injection_rate=(
            unauthorized_injections / injected_predictions if injected_predictions else 0.0
        ),
        mean_injected_source_diversity=_mean(
            case.injected_source_diversity for case in case_results
        ),
        contribution_bytes=contribution_bytes,
        estimated_contribution_tokens=ceil(contribution_bytes / 4),
        admission_truncated_case_count=sum(case.admission_truncated for case in case_results),
        admission_latency_p50_ms=_nearest_rank_percentile(admission_latencies, 0.50),
        admission_latency_p95_ms=_nearest_rank_percentile(admission_latencies, 0.95),
        required_source_failure_closed=required_source_failure_closed,
    )
    return RecallBaselineResult(
        schema_version=RECALL_BASELINE_RESULT_SCHEMA_VERSION,
        corpus_revision=corpus.corpus_revision,
        corpus_origin=corpus.origin,
        backend=backend,
        configuration=recorded_configuration,
        metrics=metrics,
        cases=tuple(case_results),
    )


async def _required_source_failure_closed() -> bool:
    engine = RecallEngine(
        (_RequiredFailureProbeSource(),),
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version="public-recall-required-source-probe-v1",
            channel_weights={"required_failure_probe.lexical": 1.0},
            max_candidates_per_channel=1,
            fused_head_limit=1,
        ),
    )
    try:
        await engine.recall(RecallSituation(query="required-source-failure-probe"))
    except RecallSourceUnavailable as exc:
        return exc.source == _RequiredFailureProbeSource.name and exc.code == "failed"
    return False


def _identity_label(identity: RetrievalCandidateIdentity) -> str:
    return ":".join(identity.sort_key())


def _candidate_source_type(candidate: RecallCandidate) -> str:
    record_type = candidate.record.identity.record_type
    if record_type in {"knowledge_chunk", "knowledge_entry"}:
        return "knowledge"
    if record_type == "transcript_message":
        return "transcript"
    return record_type


def _precision(correct: int, predicted: int) -> float:
    return correct / predicted if predicted else 1.0


def _mean(values) -> float:
    copied = list(values)
    return sum(copied) / len(copied) if copied else 0.0


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(ceil(percentile * len(ordered)) - 1, 0)]


__all__ = [
    "RECALL_BASELINE_CORPUS_SCHEMA_VERSION",
    "RECALL_BASELINE_RESULT_SCHEMA_VERSION",
    "RecallBaselineCase",
    "RecallBaselineCaseResult",
    "RecallBaselineCorpus",
    "RecallBaselineExpectedIdentity",
    "RecallBaselineKnowledgeEntry",
    "RecallBaselineMetrics",
    "RecallBaselineResult",
    "RecallBaselineTranscript",
    "RecallBaselineTranscriptMessage",
    "load_recall_baseline_corpus",
    "run_recall_baseline",
]
