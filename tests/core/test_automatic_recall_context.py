from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from decimal import Decimal
from hashlib import sha256
from typing import Any

import pytest

from cayu import (
    BeforeStopContext,
    BeforeStopDecision,
    CayuApp,
    CayuConfig,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ForkSessionRequest,
    IncompleteSessionRecoveryRequest,
    LoopPolicy,
    ModelStreamEvent,
    ResumeRequest,
    RunDefaults,
    ScriptedModelProvider,
    StructuredOutputSpec,
)
from cayu._exception_groups import iter_exception_tree
from cayu._validation import canonical_durable_json_bytes
from cayu.core.agents import AgentSpec
from cayu.core.messages import (
    FilePart,
    Message,
    MessageRole,
    TextPart,
    copy_message,
    copy_message_part,
)
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.embeddings import (
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)
from cayu.memory import (
    AutomaticRecallPolicy,
    MemoryDeltaPolicy,
    MemoryDeltaRefreshDisposition,
    MemoryDeltaRefreshOutcome,
    MemoryDeltaTriggerKind,
    MemoryReanchorRefreshDisposition,
)
from cayu.memory_evidence import (
    ContextExposureEvidenceKind,
    ContextExposureState,
    RecallEvidenceQuery,
)
from cayu.providers import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelStreamDeadlineError,
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.providers.base import ModelRequest
from cayu.providers.deadlines import (
    ProviderDeadlineKind,
    ProviderStreamDeadlineEvidence,
    ProviderStreamDeadlines,
)
from cayu.recall import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    TRANSCRIPT_LEXICAL_CHANNEL,
)
from cayu.retrieval import (
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    WeightedReciprocalRankFusionConfig,
)
from cayu.runtime._checkpoint_redaction import require_secret_free_durable_object
from cayu.runtime._memory_evidence import (
    MemoryEvidenceItemReference,
    MemoryEvidenceKey,
    _request_includes_exact_memory_manifests,
    memory_evidence_key,
    memory_evidence_key_scope,
    recall_receipt_document_sha256,
    recall_receipt_manifest_binding_hmac_sha256,
    recover_context_exposure,
)
from cayu.runtime.budgets import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    InMemoryBudgetLedger,
)
from cayu.runtime.context import (
    CheckpointCompactionContextPolicy,
    ContextBuildError,
    ContextBuildResult,
    ContextCompactionTelemetry,
    ContextPolicy,
    ContextRecallTelemetry,
    ContextRequest,
    RecentTurnsContextPolicy,
    RuntimeManagedContextPolicy,
    TranscriptDigestCompactor,
    _context_secret_redactor_scope,
)
from cayu.runtime.context_counting import ContextCountingConfig, ContextCountingMode
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.memory_context import (
    _AUTOMATIC_RECALL_NOTICE,
    AutomaticRecallContextPolicy,
    AutomaticRecallSourceConfig,
    _message_digest,
    _redacted_locator_json,
    _render_projection,
)
from cayu.runtime.request_footprints import RequestFootprintConfig
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.sessions import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.runtime.structured_output import STRUCTURED_OUTPUT_TOOL_NAME
from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
from cayu.storage.memory import (
    InMemoryEmbeddingKnowledgeStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeIndexReadinessUpdate,
    KnowledgeIndexState,
    KnowledgeSearchMode,
    KnowledgeStatus,
    KnowledgeStore,
    knowledge_chunk_embedding_identity,
)
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _CountingKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self, *, access_scope: KnowledgeAccessScope) -> None:
        super().__init__(access_scope=access_scope)
        self.search_count = 0
        self.frontier_search_count = 0
        self.revision_search_count = 0
        self.change_read_count = 0
        self.readiness_read_count = 0

    async def search(self, query, *, access_scope=None):
        self.search_count += 1
        return await super().search(query, access_scope=access_scope)

    async def search_at_frontier(self, query, **kwargs):
        self.frontier_search_count += 1
        return await super().search_at_frontier(query, **kwargs)

    async def search_revisions(self, query, revision_refs, **kwargs):
        self.revision_search_count += 1
        return await super().search_revisions(query, revision_refs, **kwargs)

    async def read_changes(self, **kwargs):
        self.change_read_count += 1
        return await super().read_changes(**kwargs)

    async def read_index_readiness(self, **kwargs):
        self.readiness_read_count += 1
        return await super().read_index_readiness(**kwargs)


class _SemanticTestEmbeddingProvider(TextEmbeddingProvider):
    name = "automatic-recall-semantic-test"

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(
                    index=index,
                    vector=[
                        1.0
                        if "auth" in text.casefold() or "credential" in text.casefold()
                        else 0.0,
                        1.0 if "release" in text.casefold() else 0.0,
                        0.0,
                    ],
                )
                for index, text in enumerate(request.texts)
            ],
        )


class _RetryableSemanticKnowledgeStore(InMemoryEmbeddingKnowledgeStore):
    def __init__(self, *, access_scope: KnowledgeAccessScope) -> None:
        super().__init__(
            embedding_provider=_SemanticTestEmbeddingProvider(),
            embedding_model="automatic-recall-semantic-test",
            embedding_dimensions=3,
            access_scope=access_scope,
        )
        self.fail_revision_semantic = False
        self.revision_search_count = 0

    async def search_revisions(self, query, revision_refs, **kwargs):
        self.revision_search_count += 1
        if self.fail_revision_semantic and query.mode is KnowledgeSearchMode.SEMANTIC:
            raise RuntimeError("temporary semantic search failure")
        return await super().search_revisions(query, revision_refs, **kwargs)


@pytest.fixture(autouse=True)
def _direct_context_memory_evidence_key():
    with memory_evidence_key_scope(MemoryEvidenceKey(key_id="test-memory-key", key=b"m" * 32)):
        yield


class _CountingSessionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.transcript_search_count = 0
        self.context_exposure_list_count = 0
        self.recall_item_exposure_load_count = 0

    async def search_transcript(self, query):
        self.transcript_search_count += 1
        return await super().search_transcript(query)

    async def list_context_exposures(self, query):
        self.context_exposure_list_count += 1
        return await super().list_context_exposures(query)

    async def load_recall_item_exposures(self, session_id, exposure_id):
        self.recall_item_exposure_load_count += 1
        return await super().load_recall_item_exposures(session_id, exposure_id)


class _DispatchEvidenceFailingSessionStore(_CountingSessionStore):
    invocation_lifecycle_command_version = 1

    async def transition_context_exposure(self, session_id, exposure_id, request):
        if request.state is ContextExposureState.DISPATCH_STARTED:
            raise RuntimeError("context exposure dispatch persistence failed")
        return await super().transition_context_exposure(session_id, exposure_id, request)


class _StageDispatchReceiptFailingSessionStore(_CountingSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self._terminal_transition_failures_remaining = 2

    async def mark_model_completion_stage_dispatched(
        self,
        session_id,
        *,
        stage,
        consume_child_session_notifications=True,
    ):
        del session_id, stage
        raise RuntimeError("model dispatch receipt persistence failed")

    async def transition_context_exposure(self, session_id, exposure_id, request):
        if (
            request.state is ContextExposureState.FAILED
            and self._terminal_transition_failures_remaining
        ):
            self._terminal_transition_failures_remaining -= 1
            raise RuntimeError("initial context exposure terminal persistence failed")
        return await super().transition_context_exposure(session_id, exposure_id, request)


class _ExposureCreationReconciliationFailingSessionStore(_CountingSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self._creation_ack_failures_remaining = 2
        self._creation_readback_failures_remaining = 1

    async def create_context_exposure(self, exposure, item_exposures=()):
        persisted = await super().create_context_exposure(exposure, item_exposures)
        if self._creation_ack_failures_remaining:
            self._creation_ack_failures_remaining -= 1
            raise RuntimeError("context exposure creation acknowledgement lost")
        return persisted

    async def load_context_exposure(self, session_id, exposure_id):
        if self._creation_readback_failures_remaining:
            self._creation_readback_failures_remaining -= 1
            raise RuntimeError("context exposure creation readback unavailable")
        return await super().load_context_exposure(session_id, exposure_id)


class _EvidenceAcknowledgementLosingSessionStore(_CountingSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self._lose_receipt_ack = True
        self._lose_exposure_ack = True
        self._lose_transition_acks = {
            ContextExposureState.PREPARED,
            ContextExposureState.DISPATCH_STARTED,
        }

    async def create_recall_receipt(self, receipt):
        persisted = await super().create_recall_receipt(receipt)
        if self._lose_receipt_ack:
            self._lose_receipt_ack = False
            raise RuntimeError("recall receipt acknowledgement lost")
        return persisted

    async def create_context_exposure(self, exposure, item_exposures=()):
        persisted = await super().create_context_exposure(exposure, item_exposures)
        if self._lose_exposure_ack:
            self._lose_exposure_ack = False
            raise RuntimeError("context exposure acknowledgement lost")
        return persisted

    async def transition_context_exposure(self, session_id, exposure_id, request):
        persisted = await super().transition_context_exposure(
            session_id,
            exposure_id,
            request,
        )
        if request.state in self._lose_transition_acks:
            self._lose_transition_acks.remove(request.state)
            raise RuntimeError("context exposure transition acknowledgement lost")
        return persisted


class _TimeoutBeforeAcknowledgementScriptedProvider(ScriptedModelProvider):
    async def stream(self, request):
        self._consume_batch(request)
        if False:  # pragma: no cover - keeps this an async generator
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
        raise TimeoutError("provider response boundary timed out")


class _RecordingCountScriptedProvider(ScriptedModelProvider):
    def __init__(self, events) -> None:
        super().__init__(events)
        self.count_requests: list[ModelRequest] = []

    async def count_input_tokens(self, request: ModelRequest) -> None:
        self.count_requests.append(
            ModelRequest(
                model=request.model,
                messages=request.messages,
                tools=request.tools,
                hosted_tools=request.hosted_tools,
                options=request.options,
            )
        )
        return None


class _FirstRequestRaisingScriptedProvider(ScriptedModelProvider):
    def __init__(
        self,
        failure: Exception,
        recovery_events: tuple[ModelStreamEvent, ...] = (),
    ) -> None:
        completed = ModelStreamEvent.completed({"finish_reason": "stop"})
        super().__init__([[completed], list(recovery_events or (completed,))])
        self.failure = failure

    async def stream(self, request):
        events = self._consume_batch(request)
        if len(self.requests) == 1:
            raise self.failure
        for event in events:
            yield event


class _ProviderEffectThenRaisingScriptedProvider(ScriptedModelProvider):
    def __init__(self, failure: Exception) -> None:
        super().__init__(
            [
                ModelStreamEvent.text_delta("partial provider output"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        self.failure = failure

    async def stream(self, request):
        events = self._consume_batch(request)
        yield events[0]
        raise self.failure


class _MalformedAcknowledgementScriptedProvider(ScriptedModelProvider):
    async def stream(self, request):
        self._consume_batch(request)
        yield {"type": "text_delta", "payload": {"text": "not a typed event"}}


class _BlockingBeforeAcknowledgementScriptedProvider(ScriptedModelProvider):
    def __init__(self) -> None:
        super().__init__([ModelStreamEvent.completed({"finish_reason": "stop"})])
        self.started = asyncio.Event()

    async def stream(self, request):
        self._consume_batch(request)
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")
        yield  # pragma: no cover


class _MemoryRecoveryProcessLoss(BaseException):
    pass


class _MemoryRecoveryOperationAdapter(ProviderOperationAdapter):
    def __init__(self) -> None:
        self.status = ProviderOperationStatus.IN_PROGRESS
        self.state = ProviderOperationState(
            operation_id="automatic_recall_recovery_operation",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": 0},
        )
        self.start_calls = 0
        self.retrieve_calls = 0

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        del request
        self.start_calls += 1

        async def events() -> AsyncIterator[ModelStreamEvent]:
            raise _MemoryRecoveryProcessLoss("worker lost after operation publication")
            yield  # pragma: no cover

        return ProviderOperationConnection(
            state=self.state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        assert state == self.state
        self.retrieve_calls += 1
        return ProviderOperationSnapshot(
            state=self.state,
            status=self.status,
            events=(
                ModelStreamEvent.text_delta("Friday"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            )
            if self.status is ProviderOperationStatus.COMPLETED
            else (),
        )

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        del state
        raise AssertionError("terminal recovery must not reconnect")

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        del state
        raise AssertionError("recovery must not cancel the completed operation")


class _MemoryRecoveryOperationProvider(ModelProvider):
    name = "automatic-recall-recovery"

    def __init__(self) -> None:
        self.adapter = _MemoryRecoveryOperationAdapter()

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:automatic-recall-recovery-provider",
            behavior_version="1",
            implementation_version="1",
        )

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return ProviderOperationMode.BACKGROUND

    @property
    def provider_operations(self) -> ProviderOperationAdapter:
        return self.adapter

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        raise AssertionError("background provider must not use synchronous streaming")
        yield  # pragma: no cover


class _CorruptingAutomaticRecallEvidencePolicy(AutomaticRecallContextPolicy):
    corrupted_field: str

    async def build_with_checkpoint(self, request, *, checkpoint):
        result = await super().build_with_checkpoint(request, checkpoint=checkpoint)
        assert result.checkpoint is not None
        corrupted = json.loads(json.dumps(result.checkpoint))
        corrupted["automatic_recall"][self.corrupted_field] = "0" * 64
        return result.model_copy(update={"checkpoint": corrupted})


class _ReceiptDigestCorruptingAutomaticRecallPolicy(_CorruptingAutomaticRecallEvidencePolicy):
    corrupted_field = "receipt_document_sha256"


class _ReceiptManifestBindingCorruptingAutomaticRecallPolicy(
    _CorruptingAutomaticRecallEvidencePolicy
):
    corrupted_field = "receipt_manifest_binding_hmac_sha256"


class _ReceiptIdentityRemovingAutomaticRecallPolicy(AutomaticRecallContextPolicy):
    async def build_with_checkpoint(self, request, *, checkpoint):
        result = await super().build_with_checkpoint(request, checkpoint=checkpoint)
        assert result.checkpoint is not None
        corrupted = json.loads(json.dumps(result.checkpoint))
        del corrupted["automatic_recall"]["receipt_id"]
        return result.model_copy(update={"checkpoint": corrupted})


class _StripAutomaticMemoryPart(ContextPolicy):
    async def build(self, request: ContextRequest) -> list[Message]:
        result: list[Message] = []
        for message in request.messages:
            if message.role is not MessageRole.USER:
                result.append(copy_message(message))
                continue
            result.append(
                Message(
                    role=MessageRole.USER,
                    content=tuple(
                        copy_message_part(part)
                        for part in message.content
                        if not (
                            type(part) is TextPart
                            and part.text.startswith('<cayu_automatic_memory version="2">')
                        )
                    ),
                )
            )
        return result


class _RemoveUserAnchor(ContextPolicy):
    async def build(self, request: ContextRequest) -> list[Message]:
        return [
            copy_message(message)
            for message in request.messages
            if message.role is not MessageRole.USER
        ]


class _SummarizeAndRemoveUserAnchor(ContextPolicy):
    async def build(self, request: ContextRequest) -> list[Message]:
        source_text = "\n".join(
            part.text
            for message in request.messages
            for part in message.content
            if type(part) is TextPart
        )
        return [Message.text("assistant", f"Compacted transcript:\n{source_text}")]


class _RemoveAnchorAfterFirstBoundary(ContextPolicy):
    def __init__(self) -> None:
        self.build_count = 0

    async def build(self, request: ContextRequest) -> list[Message]:
        self.build_count += 1
        if self.build_count == 1:
            return [copy_message(message) for message in request.messages]
        return [
            Message.text(
                MessageRole.USER,
                "Atlas release evidence?",
            )
        ]


class _ContinueOnceBeforeStop(LoopPolicy):
    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        if context.step == 1:
            return BeforeStopDecision.continue_with(
                Message.text("user", "Correct the final answer."),
                reason="test continuation",
            )
        return BeforeStopDecision.complete("second step is final")


class _PublishAtlasEvidenceTool(Tool):
    spec = ToolSpec(
        name="publish_atlas_evidence",
        description="Publish new Atlas evidence.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    def __init__(self, knowledge: InMemoryKnowledgeStore) -> None:
        super().__init__()
        self._knowledge = knowledge

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        await self._knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-runtime-delta",
                namespace="project:cayu",
                text="New Atlas release evidence says Saturday after the final review.",
            )
        )
        return ToolResult(content="Published the new Atlas evidence.")


class _NoopMemoryBoundaryTool(Tool):
    spec = ToolSpec(
        name="complete_memory_boundary",
        description="Complete one model boundary without changing knowledge.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="Boundary completed.")


class _SupersedeAtlasMemoryTool(Tool):
    spec = ToolSpec(
        name="supersede_atlas_memory",
        description="Supersede the Atlas knowledge revision.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    def __init__(self, knowledge: InMemoryKnowledgeStore) -> None:
        super().__init__()
        self._knowledge = knowledge

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        current = await self._knowledge.get_entry("atlas-reanchor-stale")
        assert current is not None
        await self._knowledge.append_entry_revision(
            current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "text": "Current Atlas release evidence says Saturday.",
                }
            ),
            expected_revision=current.revision,
        )
        return ToolResult(content="Atlas knowledge superseded.")


class _MarkAtlasIndexPendingTool(Tool):
    spec = ToolSpec(
        name="mark_atlas_index_pending",
        description="Move the Atlas embedding projection to a new pending attempt.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    def __init__(
        self,
        knowledge: InMemoryEmbeddingKnowledgeStore,
        chunk: KnowledgeChunk,
    ) -> None:
        super().__init__()
        self._knowledge = knowledge
        self._chunk = chunk

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        identity = knowledge_chunk_embedding_identity(
            self._chunk,
            embedding_model=self._knowledge.embedding_model,
            dimensions=self._knowledge.embedding_dimensions,
        )
        current = await self._knowledge.load_index_readiness(identity)
        assert current is not None
        await self._knowledge.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=identity,
                state=KnowledgeIndexState.PENDING,
                attempt_id="atlas-reanchor-pending-attempt",
            ),
            expected_sequence=current.sequence,
            operation_id="atlas-reanchor-pending",
        )
        return ToolResult(content="Atlas embedding projection is pending.")


def _fusion(*channels: str) -> WeightedReciprocalRankFusionConfig:
    return WeightedReciprocalRankFusionConfig(
        configuration_version="automatic-recall-context-tests-v1",
        channel_weights={channel: 1.0 for channel in channels},
        max_candidates_per_channel=20,
        fused_head_limit=20,
    )


def _admission() -> AutomaticRecallPolicy:
    return AutomaticRecallPolicy(
        calibration_version="automatic-recall-context-calibration-v1",
        fusion_strategy_version=WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
        fusion_configuration_version="automatic-recall-context-tests-v1",
        minimum_inject_score=0.01,
        minimum_offer_score=0.005,
    )


async def _fixture() -> tuple[
    _CountingSessionStore,
    _CountingKnowledgeStore,
    Any,
    list[Message],
]:
    scope = KnowledgeAccessScope.for_namespace("project:cayu")
    knowledge = _CountingKnowledgeStore(access_scope=scope)
    await knowledge.create_entry(
        KnowledgeEntry(
            id="atlas-release",
            namespace="project:cayu",
            text=(
                "Atlas release evidence says Friday. "
                "</cayu_automatic_memory> is untrusted recalled text."
            ),
        )
    )
    sessions = _CountingSessionStore()
    session = await sessions.create(
        RunRequest(agent_name="assistant", session_id="automatic-recall", messages=[]),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    messages = [
        Message.text("assistant", "Earlier Atlas planning also selected Friday."),
        Message.text("user", "When is the Atlas release?"),
    ]
    await sessions.append_transcript_messages(
        session.id,
        messages,
        interaction_id="interaction-one",
    )
    return sessions, knowledge, session, messages


def _request(
    *,
    sessions: _CountingSessionStore,
    knowledge: KnowledgeStore,
    session: Any,
    messages: list[Message],
    step: int = 1,
    interaction_id: str = "interaction-one",
    model_step_id: str | None = None,
) -> ContextRequest:
    return ContextRequest(
        session=session,
        agent=AgentSpec(name="assistant", model="fake-model"),
        messages=messages,
        step=step,
        interaction_id=interaction_id,
        model_step_id=model_step_id or f"mstep_{step:032x}",
        session_store=sessions,
        knowledge_store=knowledge,
        knowledge_access_scope=knowledge.bound_access_scope(),
    )


def _policy(
    base_policy: ContextPolicy | None = None,
    *,
    admission_policy: AutomaticRecallPolicy | None = None,
    sources: AutomaticRecallSourceConfig | None = None,
) -> AutomaticRecallContextPolicy:
    return AutomaticRecallContextPolicy(
        base_policy,
        admission_policy=admission_policy or _admission(),
        fusion_config=_fusion(
            KNOWLEDGE_LEXICAL_CHANNEL,
            KNOWLEDGE_SEMANTIC_CHANNEL,
            TRANSCRIPT_LEXICAL_CHANNEL,
        ),
        sources=sources or AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
    )


def _manifest(result) -> str:
    user = next(message for message in result.messages if message.role is MessageRole.USER)
    assert type(user.content[0]) is TextPart
    return user.content[0].text


def _provider_manifest(messages: list[Message]) -> str:
    return next(
        part.text
        for message in messages
        if message.role is MessageRole.USER
        for part in message.content
        if type(part) is TextPart and part.text.startswith('<cayu_automatic_memory version="2">')
    )


def test_automatic_recall_freezes_projection_across_tool_and_runtime_user_rounds() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = _policy()
        original = [copy_message(message) for message in messages]

        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        first_manifest = _manifest(first)
        assert first_manifest.count('<cayu_automatic_memory version="2">') == 1
        assert first_manifest.count("</cayu_automatic_memory>") == 1
        assert "\\u003c/cayu_automatic_memory\\u003e" in first_manifest
        assert messages == original
        assert first.checkpoint is not None
        assert first.checkpoint["automatic_recall"]["anchor_transcript_index"] == 1
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1
        assert knowledge.change_read_count == 0
        assert knowledge.readiness_read_count == 0

        tool_round = [*messages, Message.text("assistant", "tool round continuation")]
        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=tool_round,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
        assert _manifest(second) == first_manifest
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

        repair = Message.text("user", "Return the required structured value.")
        repair_round = [*tool_round, repair]
        repair_checkpoint = dict(first.checkpoint)
        repair_checkpoint["runtime_authored_user_message"] = {
            "version": 1,
            "anchor_transcript_index": len(repair_round) - 1,
            "user_message_sha256": _message_digest(repair),
        }
        third = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=repair_round,
                step=3,
            ),
            checkpoint=repair_checkpoint,
        )
        assert _manifest(third) == first_manifest
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1
        assert third.checkpoint is not None
        assert third.checkpoint["automatic_recall"]["runtime_authored_anchors"] == [
            {
                "anchor_transcript_index": len(repair_round) - 1,
                "user_message_sha256": _message_digest(repair),
            }
        ]

        next_user = Message.text("user", "What did Atlas planning decide?")
        next_round = [*repair_round, next_user]
        await sessions.append_transcript_messages(
            session.id,
            [repair, next_user],
            interaction_id="interaction-two",
        )
        fourth = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=next_round,
                step=4,
                interaction_id="interaction-two",
            ),
            checkpoint=third.checkpoint,
        )
        assert fourth.checkpoint is not None
        assert fourth.checkpoint["automatic_recall"]["anchor_transcript_index"] == (
            len(next_round) - 1
        )
        assert fourth.checkpoint["automatic_recall"]["runtime_authored_anchors"] == []
        assert knowledge.search_count == 2
        assert sessions.transcript_search_count == 2

    asyncio.run(run())


def test_memory_delta_appends_new_revision_once_without_changing_base_focus() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(),
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        base_manifest = _manifest(first)
        assert '<cayu_memory_delta version="2"' not in base_manifest

        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-new-evidence",
                namespace="project:cayu",
                text="New Atlas release evidence says Saturday after the final review.",
            )
        )
        boundary_messages = [*messages, Message.text("assistant", "Tool round completed.")]
        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=boundary_messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
        assert second.checkpoint is not None
        user = next(message for message in second.messages if message.role is MessageRole.USER)
        text_parts = [part.text for part in user.content if type(part) is TextPart]
        assert text_parts[0] == base_manifest
        assert text_parts[1].startswith('<cayu_memory_delta version="2" sequence="1">')
        assert "Saturday" in text_parts[1]
        delta_state = second.checkpoint["automatic_recall"]["delta_state"]
        assert [item["sequence"] for item in delta_state["deltas"]] == [1]
        assert [item["disposition"] for item in delta_state["refresh_outcomes"]] == [
            MemoryDeltaRefreshDisposition.DELTA_APPENDED.value
        ]
        assert knowledge.frontier_search_count == 1
        assert knowledge.revision_search_count == 1

        third = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=boundary_messages,
                step=3,
            ),
            checkpoint=second.checkpoint,
        )
        assert third.checkpoint is not None
        assert third.messages == second.messages
        assert [
            item["disposition"]
            for item in third.checkpoint["automatic_recall"]["delta_state"]["refresh_outcomes"]
        ] == [
            MemoryDeltaRefreshDisposition.DELTA_APPENDED.value,
            MemoryDeltaRefreshDisposition.FRONTIER_UNCHANGED.value,
        ]
        assert knowledge.revision_search_count == 1
        receipts = (
            await sessions.list_recall_receipts(RecallEvidenceQuery(session_id=session.id))
        ).items
        assert len(receipts) == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "relevance",
    [
        "rank_only.v1",
        "cayu.query_concepts.v1",
        "cayu.query_concepts.v2",
        "cayu.query_concepts.v3",
        "cayu.query_concepts.v4",
    ],
)
def test_reanchor_policy_requires_validated_independent_relevance(relevance: str) -> None:
    admission = AutomaticRecallPolicy.model_validate(
        {**_admission().model_dump(mode="python"), "relevance_policy": relevance}
    )
    policy = AutomaticRecallContextPolicy(
        admission_policy=admission,
        fusion_config=_fusion(
            KNOWLEDGE_LEXICAL_CHANNEL,
            KNOWLEDGE_SEMANTIC_CHANNEL,
            TRANSCRIPT_LEXICAL_CHANNEL,
        ),
        sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
        delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=True),
    )
    restored = policy._reanchor_admission_policy()
    assert policy.admission_policy == admission
    assert restored.relevance_policy == "cayu.query_concepts.v2"
    assert restored.relevance_text_version is not None
    assert restored == AutomaticRecallPolicy.model_validate(restored.model_dump(mode="python"))


def test_reanchor_policy_has_no_exposure_or_recall_work_while_projection_is_retained() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=True),
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=[*messages, Message.text("assistant", "Tool round completed.")],
                step=2,
            ),
            checkpoint=first.checkpoint,
        )

        assert second.checkpoint is not None
        assert (
            second.checkpoint["automatic_recall"]["delta_state"]["reanchor_refresh_outcomes"] == []
        )
        assert sessions.context_exposure_list_count == 0
        assert sessions.recall_item_exposure_load_count == 0
        assert knowledge.revision_search_count == 0

    asyncio.run(run())


def test_memory_delta_rejects_a_non_injecting_admission_mode() -> None:
    with pytest.raises(ValueError, match="mode that injects strong matches"):
        AutomaticRecallContextPolicy(
            admission_policy=_admission().model_copy(update={"mode": "offer"}),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            delta_policy=MemoryDeltaPolicy(),
        )


def test_memory_delta_reuses_one_boundary_plan_and_bounds_frontier_checks() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(
                max_refreshes_per_interaction=2,
                max_deltas_per_interaction=2,
            ),
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-first-boundary",
                namespace="project:cayu",
                text="New Atlas boundary evidence says Saturday.",
            )
        )
        boundary_messages = [*messages, Message.text("assistant", "Boundary.")]
        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=boundary_messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
        assert second.checkpoint is not None
        reads_after_second = (knowledge.change_read_count, knowledge.readiness_read_count)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-after-boundary",
                namespace="project:cayu",
                text="Later Atlas boundary evidence says Sunday.",
            )
        )

        exact_retry = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=boundary_messages,
                step=2,
            ),
            checkpoint=second.checkpoint,
        )
        assert exact_retry.checkpoint is None
        assert exact_retry.messages == second.messages
        assert (knowledge.change_read_count, knowledge.readiness_read_count) == reads_after_second
        with pytest.raises(ContextBuildError, match="another interaction"):
            await policy.build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=boundary_messages,
                    step=2,
                    interaction_id="interaction-other",
                ),
                checkpoint=second.checkpoint,
            )
        assert (knowledge.change_read_count, knowledge.readiness_read_count) == reads_after_second

        third = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=[*boundary_messages, Message.text("assistant", "Next boundary.")],
                step=3,
            ),
            checkpoint=second.checkpoint,
        )
        assert third.checkpoint is not None
        assert len(third.checkpoint["automatic_recall"]["delta_state"]["deltas"]) == 2
        reads_after_third = (knowledge.change_read_count, knowledge.readiness_read_count)

        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-after-refresh-budget",
                namespace="project:cayu",
                text="Atlas evidence beyond the refresh budget says Monday.",
            )
        )
        fourth = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=[*boundary_messages, Message.text("assistant", "Final boundary.")],
                step=4,
            ),
            checkpoint=third.checkpoint,
        )
        assert fourth.checkpoint is None
        assert (knowledge.change_read_count, knowledge.readiness_read_count) == reads_after_third

    asyncio.run(run())


def test_memory_delta_ranking_isolated_from_transcript_candidates() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        fusion = WeightedReciprocalRankFusionConfig(
            configuration_version="memory-delta-source-isolation-v1",
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
                TRANSCRIPT_LEXICAL_CHANNEL: 100.0,
            },
            max_candidates_per_channel=20,
            fused_head_limit=20,
        )
        admission = AutomaticRecallPolicy(
            calibration_version="memory-delta-source-isolation-v1",
            fusion_strategy_version=fusion.strategy_version,
            fusion_configuration_version=fusion.configuration_version,
            minimum_inject_score=0.0001,
            minimum_offer_score=0.0001,
            max_evaluated_candidates=1,
            max_injected_items=1,
            max_offered_items=1,
        )
        policy = AutomaticRecallContextPolicy(
            admission_policy=admission,
            fusion_config=fusion,
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(),
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        assert sessions.transcript_search_count == 1
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-source-isolated-delta",
                namespace="project:cayu",
                text="Atlas release evidence says Saturday.",
            )
        )

        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=[*messages, Message.text("assistant", "Tool completed.")],
                step=2,
            ),
            checkpoint=first.checkpoint,
        )

        assert second.checkpoint is not None
        delta_state = second.checkpoint["automatic_recall"]["delta_state"]
        assert len(delta_state["deltas"]) == 1
        assert delta_state["refresh_outcomes"][0]["disposition"] == (
            MemoryDeltaRefreshDisposition.DELTA_APPENDED.value
        )
        assert sessions.transcript_search_count == 1
        assert knowledge.revision_search_count == 1

    asyncio.run(run())


def test_memory_delta_retries_transient_partial_semantic_recall() -> None:
    async def run() -> None:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _RetryableSemanticKnowledgeStore(access_scope=scope)
        sessions = _CountingSessionStore()
        session = await sessions.create(
            RunRequest(agent_name="assistant", session_id="semantic-delta-retry", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [Message.text("user", "How should auth work?")]
        await sessions.append_transcript_messages(
            session.id,
            messages,
            interaction_id="interaction-one",
        )
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL),
            sources=AutomaticRecallSourceConfig(
                include_transcript=False,
                transcript_required=False,
                knowledge_namespace="project:cayu",
            ),
            delta_policy=MemoryDeltaPolicy(),
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        entry = KnowledgeEntry(
            id="credential-design",
            namespace="project:cayu",
            text="Credential broker design.",
        )
        await knowledge.create_entry(
            entry,
            [
                KnowledgeChunk(
                    id="credential-design-chunk",
                    entry_id=entry.id,
                    entry_revision=entry.revision,
                    chunk_index=0,
                    text=entry.text,
                )
            ],
        )
        worker = await knowledge.process_embedding_changes(
            "semantic-delta-retry-consumer",
            "semantic-delta-retry-worker",
            access_scope=scope,
        )
        assert worker.acknowledged_changes == 1
        knowledge.fail_revision_semantic = True

        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
        assert second.checkpoint is not None
        partial_state = second.checkpoint["automatic_recall"]["delta_state"]
        assert partial_state["deltas"] == []
        assert partial_state["refresh_outcomes"][0]["disposition"] == (
            MemoryDeltaRefreshDisposition.RECALL_INCOMPLETE.value
        )
        assert partial_state["knowledge_sequence"] == partial_state["initial_knowledge_sequence"]
        assert (
            partial_state["index_readiness_sequence"]
            == partial_state["initial_index_readiness_sequence"]
        )

        knowledge.fail_revision_semantic = False
        third = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
                step=3,
            ),
            checkpoint=second.checkpoint,
        )

        assert third.checkpoint is not None
        recovered_state = third.checkpoint["automatic_recall"]["delta_state"]
        assert len(recovered_state["deltas"]) == 1
        assert [item["disposition"] for item in recovered_state["refresh_outcomes"]] == [
            MemoryDeltaRefreshDisposition.RECALL_INCOMPLETE.value,
            MemoryDeltaRefreshDisposition.DELTA_APPENDED.value,
        ]
        assert knowledge.revision_search_count == 4

    asyncio.run(run())


def test_memory_delta_records_item_budget_exhaustion() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL),
            sources=AutomaticRecallSourceConfig(
                include_transcript=False,
                transcript_required=False,
                knowledge_namespace="project:cayu",
            ),
            delta_policy=MemoryDeltaPolicy(max_items_per_delta=1, max_cumulative_items=1),
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-over-item-budget",
                namespace="project:cayu",
                text="Atlas release evidence says Saturday.",
            )
        )

        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )

        assert second.checkpoint is not None
        state = second.checkpoint["automatic_recall"]["delta_state"]
        assert state["deltas"] == []
        outcome = MemoryDeltaRefreshOutcome.model_validate(state["refresh_outcomes"][0])
        assert outcome.disposition is MemoryDeltaRefreshDisposition.ITEM_BUDGET_EXHAUSTED
        assert outcome.eligible_item_count == outcome.omitted_item_count == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    ("delta_policy", "expected_disposition"),
    [
        (
            MemoryDeltaPolicy(max_delta_bytes=1, max_cumulative_bytes=1),
            MemoryDeltaRefreshDisposition.BYTE_BUDGET_EXHAUSTED,
        ),
        (
            MemoryDeltaPolicy(max_delta_estimated_tokens=1),
            MemoryDeltaRefreshDisposition.TOKEN_BUDGET_EXHAUSTED,
        ),
    ],
)
def test_memory_delta_records_projection_budget_exhaustion(
    delta_policy: MemoryDeltaPolicy,
    expected_disposition: MemoryDeltaRefreshDisposition,
) -> None:
    async def run() -> None:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        sessions = _CountingSessionStore()
        session = await sessions.create(
            RunRequest(agent_name="assistant", session_id="delta-byte-budget", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [Message.text("user", "When is the Atlas release?")]
        await sessions.append_transcript_messages(
            session.id,
            messages,
            interaction_id="interaction-one",
        )
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL),
            sources=AutomaticRecallSourceConfig(
                include_transcript=False,
                transcript_required=False,
                knowledge_namespace="project:cayu",
            ),
            delta_policy=delta_policy,
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-over-byte-budget",
                namespace="project:cayu",
                text="Atlas release evidence says Saturday.",
            )
        )

        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )

        assert second.checkpoint is not None
        state = second.checkpoint["automatic_recall"]["delta_state"]
        assert state["deltas"] == []
        outcome = MemoryDeltaRefreshOutcome.model_validate(state["refresh_outcomes"][0])
        assert outcome.disposition is expected_disposition
        assert outcome.eligible_item_count == outcome.omitted_item_count == 1

    asyncio.run(run())


def test_memory_delta_rejects_base_projection_over_cumulative_budget() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(max_delta_bytes=1, max_cumulative_bytes=1),
        )

        with pytest.raises(ContextBuildError) as captured:
            await policy.build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=messages,
                ),
                checkpoint=None,
            )

        assert isinstance(captured.value.__cause__, ValueError)
        assert "base automatic-memory projection" in str(captured.value.__cause__)

    asyncio.run(run())


def test_memory_delta_checkpoint_rejects_active_policy_limit_violation() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(
                max_refreshes_per_interaction=1,
                max_deltas_per_interaction=1,
            ),
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
        assert second.checkpoint is not None
        corrupted = json.loads(json.dumps(second.checkpoint))
        state = corrupted["automatic_recall"]["delta_state"]
        model_step_id = f"mstep_{999:032x}"
        state["refresh_outcomes"].append(
            MemoryDeltaRefreshOutcome(
                interaction_id="interaction-one",
                ordinal=2,
                model_step_id=model_step_id,
                disposition=MemoryDeltaRefreshDisposition.FRONTIER_UNCHANGED,
                previous_knowledge_sequence=state["knowledge_sequence"],
                observed_knowledge_sequence=state["knowledge_sequence"],
                previous_index_readiness_sequence=state["index_readiness_sequence"],
                observed_index_readiness_sequence=state["index_readiness_sequence"],
            ).model_dump(mode="json")
        )
        state["last_evaluated_model_step_id"] = model_step_id

        with pytest.raises(ContextBuildError, match="checkpoint is invalid"):
            await policy.build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=messages,
                    step=3,
                ),
                checkpoint=corrupted,
            )

    asyncio.run(run())


def test_memory_delta_uses_the_sqlite_frontier_and_exact_revision_path(tmp_path) -> None:
    async def run() -> None:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = SQLiteKnowledgeStore(tmp_path / "memory-delta.sqlite", access_scope=scope)
        sessions = _CountingSessionStore()
        session = await sessions.create(
            RunRequest(agent_name="assistant", session_id="sqlite-memory-delta", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [Message.text("user", "When is the Atlas release?")]
        await sessions.append_transcript_messages(
            session.id,
            messages,
            interaction_id="interaction-sqlite",
        )
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(),
        )

        first = await policy.build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=messages,
                step=1,
                interaction_id="interaction-sqlite",
                model_step_id="mstep_00000000000000000000000000000001",
                session_store=sessions,
                knowledge_store=knowledge,
                knowledge_access_scope=scope,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        assert first.checkpoint["automatic_recall"]["projection"] is None
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-sqlite-delta",
                namespace="project:cayu",
                text="New Atlas release evidence says Saturday.",
            )
        )
        second = await policy.build_with_checkpoint(
            ContextRequest(
                session=session,
                agent=AgentSpec(name="assistant", model="fake-model"),
                messages=[*messages, Message.text("assistant", "Tool boundary.")],
                step=2,
                interaction_id="interaction-sqlite",
                model_step_id="mstep_00000000000000000000000000000002",
                session_store=sessions,
                knowledge_store=knowledge,
                knowledge_access_scope=scope,
            ),
            checkpoint=first.checkpoint,
        )
        assert second.checkpoint is not None
        delta = second.checkpoint["automatic_recall"]["delta_state"]["deltas"][0]
        assert (
            delta["projection"]["base_receipt_id"]
            == first.checkpoint["automatic_recall"]["receipt_id"]
        )
        locator = json.loads(delta["projection"]["items"][0]["locator_json"])
        assert locator["entry_id"] == "atlas-sqlite-delta"
        await knowledge.close()

    asyncio.run(run())


def test_runtime_exposes_base_and_delta_receipts_for_the_exact_tool_round() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-runtime-base",
                namespace="project:cayu",
                text="Initial Atlas release evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_publish_atlas_evidence",
                        name="publish_atlas_evidence",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("Saturday"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(),
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=policy,
            tools=[_PublishAtlasEvidenceTool(knowledge)],
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-memory-delta",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        first_parts = [
            part.text
            for message in provider.requests[0].messages
            for part in message.content
            if type(part) is TextPart
        ]
        second_parts = [
            part.text
            for message in provider.requests[1].messages
            for part in message.content
            if type(part) is TextPart
        ]
        base = next(text for text in first_parts if text.startswith("<cayu_automatic_memory"))
        assert base in second_parts
        assert any(text.startswith("<cayu_memory_delta") for text in second_parts)

        receipts = (
            await sessions.list_recall_receipts(
                RecallEvidenceQuery(session_id="automatic-recall-memory-delta")
            )
        ).items
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-memory-delta")
            )
        ).items
        assert len(receipts) == 2
        assert len(exposures) == 2
        assert len(exposures[0].receipt_ids) == 1
        assert exposures[1].receipt_ids == tuple(receipt.receipt_id for receipt in receipts)
        links = await sessions.load_recall_item_exposures(
            "automatic-recall-memory-delta",
            exposures[1].exposure_id,
        )
        assert {link.receipt_id for link in links} == set(exposures[1].receipt_ids)
        delta_receipt_id = receipts[1].receipt_id
        assert {
            link.selection_reason.value for link in links if link.receipt_id == delta_receipt_id
        } == {"newly_relevant"}

    asyncio.run(run())


def test_runtime_reanchors_a_current_previously_exposed_revision_after_projection_loss() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-reanchor",
                namespace="project:cayu",
                text="Verified Atlas release evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_complete_memory_boundary",
                        name="complete_memory_boundary",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("Friday"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        policy = AutomaticRecallContextPolicy(
            _RemoveAnchorAfterFirstBoundary(),
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=True),
        )
        footprint_config = RequestFootprintConfig(
            fingerprint_key_id="test-memory-key",
            fingerprint_key="automatic-recall-test-key-material",
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=footprint_config,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=policy,
            tools=[_NoopMemoryBoundaryTool()],
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-reanchor",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        first_memory = [
            part.text
            for message in provider.requests[0].messages
            for part in message.content
            if type(part) is TextPart and part.text.startswith("<cayu_automatic_memory")
        ]
        second_memory = [
            part.text
            for message in provider.requests[1].messages
            for part in message.content
            if type(part) is TextPart and part.text.startswith("<cayu_memory_delta")
        ]
        assert len(first_memory) == len(second_memory) == 1
        assert "Friday" in second_memory[0]
        assert "reanchored_current_revision" in second_memory[0]
        assert not any(
            type(part) is TextPart and part.text.startswith("<cayu_automatic_memory")
            for message in provider.requests[1].messages
            for part in message.content
        )

        checkpoint = await sessions.load_checkpoint("automatic-recall-reanchor")
        assert checkpoint is not None
        delta_state = checkpoint["automatic_recall"]["delta_state"]
        assert delta_state["original_projection_suppressed"] is True
        assert len(delta_state["suppressed_manifest_sha256s"]) == 1
        assert len(delta_state["deltas"]) == 1
        trigger = delta_state["deltas"][0]["trigger"]
        assert trigger["kind"] == (
            MemoryDeltaTriggerKind.PROJECTION_REMOVED_BY_CONTEXT_POLICY.value
        )
        assert trigger["minimum_boundary_distance"] == 1
        assert [item["disposition"] for item in delta_state["reanchor_refresh_outcomes"]] == [
            MemoryReanchorRefreshDisposition.DELTA_APPENDED.value
        ]

        receipts = (
            await sessions.list_recall_receipts(
                RecallEvidenceQuery(session_id="automatic-recall-reanchor")
            )
        ).items
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-reanchor")
            )
        ).items
        assert len(receipts) == len(exposures) == 2
        assert exposures[0].provider_exposure_proven is True
        reanchor_links = await sessions.load_recall_item_exposures(
            "automatic-recall-reanchor",
            exposures[1].exposure_id,
        )
        assert {link.selection_reason.value for link in reanchor_links} == {
            "reanchored_current_revision"
        }

        recovered_session = await sessions.load("automatic-recall-reanchor")
        snapshot = await sessions.load_transcript_snapshot("automatic-recall-reanchor")
        assert recovered_session is not None
        reads_before_retry = (
            sessions.context_exposure_list_count,
            sessions.recall_item_exposure_load_count,
            knowledge.revision_search_count,
        )
        with memory_evidence_key_scope(memory_evidence_key(footprint_config)):
            exact_retry = await policy.build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=recovered_session,
                    messages=[record.message for record in snapshot.records],
                    step=2,
                    interaction_id=checkpoint["automatic_recall"]["interaction_id"],
                    model_step_id=trigger["model_step_id"],
                ),
                checkpoint=checkpoint,
            )

        assert exact_retry.checkpoint is None
        assert tuple(
            part.text
            for message in exact_retry.messages
            for part in message.content
            if type(part) is TextPart and part.text.startswith("<cayu_memory_delta")
        ) == tuple(second_memory)
        assert (
            sessions.context_exposure_list_count,
            sessions.recall_item_exposure_load_count,
            knowledge.revision_search_count,
        ) == reads_before_retry

    asyncio.run(run())


def test_runtime_does_not_reanchor_a_superseded_revision_after_projection_loss() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-reanchor-stale",
                namespace="project:cayu",
                text="Obsolete Atlas release evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_supersede_atlas_memory",
                        name="supersede_atlas_memory",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("No stale memory was projected."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=AutomaticRecallContextPolicy(
                _RemoveAnchorAfterFirstBoundary(),
                admission_policy=_admission(),
                fusion_config=_fusion(
                    KNOWLEDGE_LEXICAL_CHANNEL,
                    KNOWLEDGE_SEMANTIC_CHANNEL,
                    TRANSCRIPT_LEXICAL_CHANNEL,
                ),
                sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
                delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=True),
            ),
            tools=[_SupersedeAtlasMemoryTool(knowledge)],
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-reanchor-stale",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        assert not any(
            type(part) is TextPart
            and (
                part.text.startswith("<cayu_automatic_memory")
                or part.text.startswith("<cayu_memory_delta")
            )
            for message in provider.requests[1].messages
            for part in message.content
        )
        checkpoint = await sessions.load_checkpoint("automatic-recall-reanchor-stale")
        assert checkpoint is not None
        delta_state = checkpoint["automatic_recall"]["delta_state"]
        assert all(item["projection"] is None for item in delta_state["deltas"])
        assert delta_state["reanchor_refresh_outcomes"][-1]["disposition"] == (
            MemoryReanchorRefreshDisposition.NO_CURRENT_RELEVANT_ITEM.value
        )

    asyncio.run(run())


def test_runtime_does_not_reanchor_while_the_exact_revision_index_is_pending() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _RetryableSemanticKnowledgeStore(access_scope=scope)
        entry = KnowledgeEntry(
            id="atlas-reanchor-pending",
            namespace="project:cayu",
            text="Verified Atlas release evidence says Friday.",
        )
        chunk = KnowledgeChunk(
            id="atlas-reanchor-pending-chunk",
            entry_id=entry.id,
            entry_revision=entry.revision,
            chunk_index=0,
            text=entry.text,
        )
        await knowledge.create_entry(entry, [chunk])
        worker = await knowledge.process_embedding_changes(
            "atlas-reanchor-pending-consumer",
            "atlas-reanchor-pending-worker",
            access_scope=scope,
        )
        assert worker.acknowledged_changes == 1
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_mark_atlas_index_pending",
                        name="mark_atlas_index_pending",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("No stale memory was projected."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=AutomaticRecallContextPolicy(
                _RemoveAnchorAfterFirstBoundary(),
                admission_policy=_admission(),
                fusion_config=_fusion(
                    KNOWLEDGE_LEXICAL_CHANNEL,
                    KNOWLEDGE_SEMANTIC_CHANNEL,
                    TRANSCRIPT_LEXICAL_CHANNEL,
                ),
                sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
                delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=True),
            ),
            tools=[_MarkAtlasIndexPendingTool(knowledge, chunk)],
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-reanchor-pending",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        assert not any(
            type(part) is TextPart
            and (
                part.text.startswith("<cayu_automatic_memory")
                or part.text.startswith("<cayu_memory_delta")
            )
            for message in provider.requests[1].messages
            for part in message.content
        )
        checkpoint = await sessions.load_checkpoint("automatic-recall-reanchor-pending")
        assert checkpoint is not None
        delta_state = checkpoint["automatic_recall"]["delta_state"]
        assert all(item["projection"] is None for item in delta_state["deltas"])
        assert delta_state["reanchor_refresh_outcomes"][-1]["disposition"] == (
            MemoryReanchorRefreshDisposition.RECALL_INCOMPLETE.value
        )

    asyncio.run(run())


def test_runtime_reproves_reanchor_per_model_step_and_expires_the_previous_projection() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-reanchor-repeat",
                namespace="project:cayu",
                text="Verified Atlas release evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_complete_memory_boundary_1",
                        name="complete_memory_boundary",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="call_complete_memory_boundary_2",
                        name="complete_memory_boundary",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("Friday"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=AutomaticRecallContextPolicy(
                _RemoveAnchorAfterFirstBoundary(),
                admission_policy=_admission(),
                fusion_config=_fusion(
                    KNOWLEDGE_LEXICAL_CHANNEL,
                    KNOWLEDGE_SEMANTIC_CHANNEL,
                    TRANSCRIPT_LEXICAL_CHANNEL,
                ),
                sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
                delta_policy=MemoryDeltaPolicy(
                    reanchor_on_projection_loss=True,
                    max_reanchors_per_item=2,
                ),
            ),
            tools=[_NoopMemoryBoundaryTool()],
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-reanchor-repeat",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 3
        second_deltas = [
            part.text
            for message in provider.requests[1].messages
            for part in message.content
            if type(part) is TextPart and part.text.startswith("<cayu_memory_delta")
        ]
        third_deltas = [
            part.text
            for message in provider.requests[2].messages
            for part in message.content
            if type(part) is TextPart and part.text.startswith("<cayu_memory_delta")
        ]
        assert len(second_deltas) == len(third_deltas) == 1
        assert 'sequence="1"' in second_deltas[0]
        assert 'sequence="2"' in third_deltas[0]
        assert second_deltas[0] not in third_deltas

        checkpoint = await sessions.load_checkpoint("automatic-recall-reanchor-repeat")
        assert checkpoint is not None
        delta_state = checkpoint["automatic_recall"]["delta_state"]
        assert len(delta_state["deltas"]) == 2
        assert delta_state["deltas"][0]["projection"] is None
        assert delta_state["deltas"][1]["projection"] is not None
        assert list(delta_state["reanchor_identity_counts"].values()) == [2]
        assert [item["disposition"] for item in delta_state["reanchor_refresh_outcomes"]] == [
            MemoryReanchorRefreshDisposition.DELTA_APPENDED.value,
            MemoryReanchorRefreshDisposition.DELTA_APPENDED.value,
        ]

    asyncio.run(run())


@pytest.mark.parametrize("corruption", ["exposure_scope", "item_parent", "item_material"])
def test_runtime_rejects_contradictory_reanchor_exposure_evidence(corruption: str) -> None:
    class ContradictoryEvidenceStore(_CountingSessionStore):
        invocation_lifecycle_command_version = 1

        async def list_context_exposures(self, query):
            page = await super().list_context_exposures(query)
            if corruption != "exposure_scope" or not page.items:
                return page
            items = (
                page.items[0].model_copy(update={"interaction_id": "foreign-interaction"}),
                *page.items[1:],
            )
            return page.model_copy(update={"items": items})

        async def load_recall_item_exposures(self, session_id, exposure_id):
            items = await super().load_recall_item_exposures(session_id, exposure_id)
            if not items:
                return items
            if corruption == "item_material":
                return (
                    items[0].model_copy(update={"content_sha256": "0" * 64}),
                    *items[1:],
                )
            if corruption != "item_parent":
                return items
            return (
                items[0].model_copy(update={"exposure_id": "foreign-exposure"}),
                *items[1:],
            )

    async def run() -> None:
        sessions = ContradictoryEvidenceStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id=f"atlas-reanchor-{corruption}",
                namespace="project:cayu",
                text=f"Verified Atlas {corruption} evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id=f"call_reanchor_{corruption}",
                        name="complete_memory_boundary",
                        arguments={},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=AutomaticRecallContextPolicy(
                _RemoveAnchorAfterFirstBoundary(),
                admission_policy=_admission(),
                fusion_config=_fusion(
                    KNOWLEDGE_LEXICAL_CHANNEL,
                    KNOWLEDGE_SEMANTIC_CHANNEL,
                    TRANSCRIPT_LEXICAL_CHANNEL,
                ),
                sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
                delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=True),
            ),
            tools=[_NoopMemoryBoundaryTool()],
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=f"automatic-recall-contradictory-{corruption}",
                    messages=[Message.text("user", f"What is the Atlas {corruption} evidence?")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_FAILED
        assert len(provider.requests) == 1

    asyncio.run(run())


def test_memory_delta_admits_a_new_revision_but_not_a_superseded_archived_revision() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(),
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        current = await knowledge.get_entry("atlas-release")
        assert current is not None
        revised = await knowledge.append_entry_revision(
            current.model_copy(
                update={
                    "revision": 2,
                    "text": "Revised Atlas release evidence says Saturday.",
                }
            ),
            expected_revision=1,
        )
        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=[*messages, Message.text("assistant", "First boundary.")],
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
        assert second.checkpoint is not None
        delta = second.checkpoint["automatic_recall"]["delta_state"]["deltas"][0]
        assert delta["projection"]["items"][0]["identity"]["revision"] == "2"

        revision_three = await knowledge.append_entry_revision(
            revised.model_copy(
                update={
                    "revision": 3,
                    "text": "Transient Atlas release evidence says Sunday.",
                }
            ),
            expected_revision=2,
        )
        await knowledge.transition_entry_status(
            revision_three.id,
            expected_revision=3,
            from_status=KnowledgeStatus.ACTIVE,
            to_status=KnowledgeStatus.ARCHIVED,
        )
        third = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=[*messages, Message.text("assistant", "Second boundary.")],
                step=3,
            ),
            checkpoint=second.checkpoint,
        )
        assert third.checkpoint is not None
        assert len(third.checkpoint["automatic_recall"]["delta_state"]["deltas"]) == 1
        rendered = "\n".join(
            part.text
            for message in third.messages
            for part in message.content
            if type(part) is TextPart
        )
        assert "Transient Atlas" not in rendered

    asyncio.run(run())


def test_memory_delta_checkpoint_rejects_reordered_or_detached_delta_state() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(),
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-checkpoint-delta",
                namespace="project:cayu",
                text="New Atlas checkpoint evidence says Saturday.",
            )
        )
        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=[*messages, Message.text("assistant", "Boundary.")],
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
        assert second.checkpoint is not None
        reordered = json.loads(json.dumps(second.checkpoint))
        reordered["automatic_recall"]["delta_state"]["deltas"][0]["sequence"] = 2
        detached_policy = json.loads(json.dumps(second.checkpoint))
        detached_policy["automatic_recall"]["delta_state"]["policy_sha256"] = "0" * 64
        missing_identity = json.loads(json.dumps(second.checkpoint))
        missing_identity["automatic_recall"]["delta_state"]["emitted_identity_hmac_sha256s"].pop()
        detached_emission_ledger = json.loads(json.dumps(second.checkpoint))
        detached_emission_ledger["automatic_recall"]["delta_state"][
            "emission_ledger_hmac_sha256"
        ] = "0" * 64
        false_suppression = json.loads(json.dumps(second.checkpoint))
        false_suppression["automatic_recall"]["delta_state"]["original_projection_suppressed"] = (
            True
        )

        for corrupted in (
            reordered,
            detached_policy,
            missing_identity,
            detached_emission_ledger,
            false_suppression,
        ):
            with pytest.raises(ContextBuildError, match="checkpoint is invalid"):
                await policy.build_with_checkpoint(
                    _request(
                        sessions=sessions,
                        knowledge=knowledge,
                        session=session,
                        messages=[*messages, Message.text("assistant", "Boundary.")],
                        step=3,
                    ),
                    checkpoint=corrupted,
                )

    asyncio.run(run())


def test_memory_delta_checkpoint_authenticates_suppressed_emission_accounting() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            _RemoveAnchorAfterFirstBoundary(),
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(),
        )
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        boundary_messages = [*messages, Message.text("assistant", "Boundary.")]
        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=boundary_messages,
                step=2,
            ),
            checkpoint=first.checkpoint,
        )
        assert second.checkpoint is not None
        delta_state = second.checkpoint["automatic_recall"]["delta_state"]
        assert delta_state["original_projection_suppressed"] is True
        assert delta_state["base_emitted_bytes"] > 0

        corrupted = json.loads(json.dumps(second.checkpoint))
        corrupted_delta_state = corrupted["automatic_recall"]["delta_state"]
        corrupted_delta_state["base_emitted_bytes"] -= 1
        with pytest.raises(ContextBuildError, match="checkpoint is invalid"):
            await policy.build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=boundary_messages,
                    step=3,
                ),
                checkpoint=corrupted,
            )

    asyncio.run(run())


@pytest.mark.parametrize("corruption", ["remove", "alter", "append"])
def test_reanchor_checkpoint_authenticates_unsuccessful_work(corruption: str) -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            _RemoveAnchorAfterFirstBoundary(),
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=True),
        )
        request = _request(
            sessions=sessions, knowledge=knowledge, session=session, messages=messages
        )
        first = await policy.build_with_checkpoint(request, checkpoint=None)
        second_request = request.model_copy(update={"step": 2, "model_step_id": f"mstep_{2:032x}"})
        second = await policy.build_with_checkpoint(second_request, checkpoint=first.checkpoint)
        assert second.checkpoint is not None
        state = second.checkpoint["automatic_recall"]["delta_state"]
        assert state["reanchor_refresh_outcomes"][0]["disposition"] == "no_acknowledged_exposure"
        corrupted = json.loads(json.dumps(second.checkpoint))
        outcomes = corrupted["automatic_recall"]["delta_state"]["reanchor_refresh_outcomes"]
        if corruption == "remove":
            outcomes.clear()
        elif corruption == "alter":
            outcomes[0]["disposition"] = "no_context_anchor"
        else:
            outcomes.append({**outcomes[0], "ordinal": 2, "model_step_id": f"mstep_{3:032x}"})
        with pytest.raises(ContextBuildError, match="checkpoint is invalid"):
            await policy.build_with_checkpoint(second_request, checkpoint=corrupted)
        # A real retry preserves the signed decision and performs no more work.
        before = sessions.context_exposure_list_count
        retry = await policy.build_with_checkpoint(second_request, checkpoint=second.checkpoint)
        assert retry.checkpoint is None
        assert sessions.context_exposure_list_count == before

    asyncio.run(run())


@pytest.mark.parametrize("reanchor", [False, True])
@pytest.mark.parametrize("failure_kind", ["wrapped", "raw", "cancelled"])
def test_refresh_failure_preserves_completed_context_checkpoint(
    monkeypatch, reanchor: bool, failure_kind: str
) -> None:
    completed_compaction = ContextCompactionTelemetry(
        event_type=EventType.CONTEXT_COMPACTION_COMPLETED,
        payload={"compactor": "checkpoint-test"},
    )
    base_recall = ContextRecallTelemetry(
        event_type=EventType.AUTOMATIC_RECALL_COMPLETED,
        payload={"operation": "base-context"},
    )
    refresh_started = ContextRecallTelemetry(
        event_type=EventType.AUTOMATIC_RECALL_STARTED,
        payload={"operation": "refresh"},
    )

    class CheckpointingPolicy(RuntimeManagedContextPolicy):
        async def build_with_checkpoint(self, request, *, checkpoint):
            messages = (
                [Message.text("user", "Atlas release evidence?")]
                if reanchor and request.step == 2
                else request.messages
            )
            return ContextBuildResult(
                messages=messages,
                checkpoint={**(checkpoint or {}), "completed_context_step": request.step},
                checkpoint_event_payload={"checkpoint": "completed_context_step"},
                compaction_telemetry=[completed_compaction],
                recall_telemetry=[base_recall],
            )

    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            CheckpointingPolicy(),
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=reanchor),
        )
        request = _request(
            sessions=sessions, knowledge=knowledge, session=session, messages=messages
        )
        first = await policy.build_with_checkpoint(request, checkpoint=None)

        original_error = (
            asyncio.CancelledError("cancelled")
            if failure_kind == "cancelled"
            else OSError("unavailable")
        )

        async def fail_refresh(*args, **kwargs):
            kwargs["recorded_telemetry"].append(refresh_started)
            if failure_kind != "wrapped":
                raise original_error
            raise ContextBuildError(
                "Refresh unavailable.",
                compaction_telemetry=[],
                recall_telemetry=kwargs["recorded_telemetry"],
                cause=original_error,
            )

        monkeypatch.setattr(
            policy, "_maybe_append_reanchor" if reanchor else "_maybe_append_delta", fail_refresh
        )
        if failure_kind == "cancelled":
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await policy.build_with_checkpoint(
                    request.model_copy(update={"step": 2, "model_step_id": f"mstep_{2:032x}"}),
                    checkpoint=first.checkpoint,
                )
            assert cancelled.value is original_error
            return
        with pytest.raises(ContextBuildError) as captured:
            await policy.build_with_checkpoint(
                request.model_copy(update={"step": 2, "model_step_id": f"mstep_{2:032x}"}),
                checkpoint=first.checkpoint,
            )
        assert captured.value.checkpoint is not None
        assert captured.value.checkpoint["completed_context_step"] == 2
        assert captured.value.checkpoint_event_payload == {"checkpoint": "completed_context_step"}
        assert captured.value.checkpoint["automatic_recall"] == first.checkpoint["automatic_recall"]
        assert captured.value.cause is original_error
        assert captured.value.compaction_telemetry == (completed_compaction,)
        assert captured.value.recall_telemetry == (base_recall, refresh_started)

    asyncio.run(run())


@pytest.mark.parametrize("reanchor", [False, True])
@pytest.mark.parametrize("failed_read", ["read_changes", "read_index_readiness"])
def test_runtime_frontier_read_failure_persists_completed_context_checkpoint(
    monkeypatch, reanchor: bool, failed_read: str
) -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        knowledge = _CountingKnowledgeStore(
            access_scope=KnowledgeAccessScope.for_namespace("project:cayu")
        )
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-frontier-failure",
                namespace="project:cayu",
                text="Verified Atlas release evidence says Friday.",
            )
        )
        failed_reads = 0

        async def fail_read(*args, **kwargs):
            nonlocal failed_reads
            failed_reads += 1
            raise OSError("Frontier storage is unavailable.")

        class CheckpointingPolicy(RuntimeManagedContextPolicy):
            async def build_with_checkpoint(self, request, *, checkpoint):
                if request.step == 2:
                    monkeypatch.setattr(knowledge, failed_read, fail_read)
                return ContextBuildResult(
                    messages=(
                        [Message.text("user", "Atlas release evidence?")]
                        if reanchor and request.step == 2
                        else request.messages
                    ),
                    checkpoint={**(checkpoint or {}), "completed_context_step": request.step},
                    checkpoint_event_payload={"checkpoint": "completed_context_step"},
                )

        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_frontier_failure", name="complete_memory_boundary", arguments={}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=AutomaticRecallContextPolicy(
                CheckpointingPolicy(),
                admission_policy=_admission(),
                fusion_config=_fusion(
                    KNOWLEDGE_LEXICAL_CHANNEL,
                    KNOWLEDGE_SEMANTIC_CHANNEL,
                    TRANSCRIPT_LEXICAL_CHANNEL,
                ),
                sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
                delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=reanchor),
            ),
            tools=[_NoopMemoryBoundaryTool()],
        )
        session_id = f"frontier-failure-{reanchor}-{failed_read}"
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_FAILED
        assert failed_reads == 1
        assert len(provider.requests) == 1
        checkpoint = await sessions.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["completed_context_step"] == 2
        delta_state = checkpoint["automatic_recall"]["delta_state"]
        assert delta_state["refresh_outcomes"] == []
        assert delta_state["reanchor_refresh_outcomes"] == []
        assert delta_state["deltas"] == []
        receipts = await sessions.list_recall_receipts(RecallEvidenceQuery(session_id=session_id))
        assert len(receipts.items) == 1

    asyncio.run(run())


@pytest.mark.parametrize("reanchor", [False, True])
def test_projection_loss_does_not_consume_a_new_frontier_delta(reanchor: bool) -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = AutomaticRecallContextPolicy(
            _RemoveAnchorAfterFirstBoundary(),
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
            delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=reanchor),
        )
        request = _request(
            sessions=sessions, knowledge=knowledge, session=session, messages=messages
        )
        first = await policy.build_with_checkpoint(request, checkpoint=None)
        assert first.checkpoint is not None
        previous = first.checkpoint["automatic_recall"]["delta_state"]
        await knowledge.create_entry(
            KnowledgeEntry(
                id="simultaneous-frontier-advance",
                namespace="project:cayu",
                text="Atlas release evidence now says Saturday.",
            )
        )
        reads_before = knowledge.revision_search_count
        second = await policy.build_with_checkpoint(
            request.model_copy(update={"step": 2, "model_step_id": f"mstep_{2:032x}"}),
            checkpoint=first.checkpoint,
        )
        assert second.checkpoint is not None
        current = second.checkpoint["automatic_recall"]["delta_state"]
        assert current["original_projection_suppressed"] is True
        assert current["deltas"] == []
        assert current["knowledge_sequence"] == previous["knowledge_sequence"]
        assert current["refresh_outcomes"] == previous["refresh_outcomes"]
        assert current["emitted_identity_hmac_sha256s"] == previous["emitted_identity_hmac_sha256s"]
        assert knowledge.revision_search_count == reads_before
        receipts = await sessions.list_recall_receipts(RecallEvidenceQuery(session_id=session.id))
        assert len(receipts.items) == 1

    asyncio.run(run())


def test_blank_real_user_interaction_expires_the_previous_recall_frame() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = _policy()
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None

        blank = Message(
            role=MessageRole.USER,
            content=(
                FilePart(
                    attachment={
                        "type": "cayu.file_attachment.v1",
                        "artifact_id": "blank-interaction-file",
                        "kind": "document",
                        "filename": "question.pdf",
                        "content_type": "application/pdf",
                        "size_bytes": 1,
                    }
                ),
            ),
        )
        second = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=[*messages, blank],
                step=2,
            ),
            checkpoint=first.checkpoint,
        )

        assert second.checkpoint is not None
        assert "automatic_recall" not in second.checkpoint
        assert all(
            not (
                type(part) is TextPart
                and part.text.startswith('<cayu_automatic_memory version="2">')
            )
            for message in second.messages
            for part in message.content
        )
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

    asyncio.run(run())


def test_automatic_recall_reapplies_or_suppresses_without_running_recall_twice() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        stripping = _policy(_StripAutomaticMemoryPart())
        retained = await stripping.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert _manifest(retained).startswith('<cayu_automatic_memory version="2">')
        assert retained.checkpoint is not None
        assert "automatic_recall" in retained.checkpoint

        removing = _policy(_RemoveUserAnchor())
        removed = await removing.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=retained.checkpoint,
        )
        assert all(message.role is not MessageRole.USER for message in removed.messages)
        assert removed.checkpoint is not None
        assert removed.checkpoint["automatic_recall"]["projection"] is None

        repeated = await removing.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
                step=2,
            ),
            checkpoint=removed.checkpoint,
        )
        assert repeated.checkpoint is None
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

        summarizing = _policy(_SummarizeAndRemoveUserAnchor())
        summarized = await summarizing.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=retained.checkpoint,
        )
        summarized_text = "\n".join(
            part.text
            for message in summarized.messages
            for part in message.content
            if type(part) is TextPart
        )
        assert "Runtime-recalled reference evidence follows" not in summarized_text
        assert "Atlas release evidence says Friday" not in summarized_text
        assert all(
            not (
                type(part) is TextPart
                and part.text.startswith('<cayu_automatic_memory version="2">')
            )
            for message in summarized.messages
            for part in message.content
        )
        assert summarized.checkpoint is not None
        assert summarized.checkpoint["automatic_recall"]["projection"] is None

    asyncio.run(run())


def test_automatic_recall_reuses_one_frame_across_checkpoint_compaction() -> None:
    async def run() -> None:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-compaction",
                namespace="project:cayu",
                text="Atlas compaction evidence says Friday.",
            )
        )
        sessions = _CountingSessionStore()
        session = await sessions.create(
            RunRequest(agent_name="assistant", session_id="recall-compaction", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [
            Message.text("user", "Earlier Atlas planning question."),
            Message.text("assistant", "Earlier Atlas planning answer."),
            Message.text("user", "When is the Atlas release?"),
        ]
        await sessions.append_transcript_messages(
            session.id,
            messages,
            interaction_id="recall-compaction-interaction",
        )
        policy = _policy(
            CheckpointCompactionContextPolicy(
                compactor=TranscriptDigestCompactor(max_summary_chars=2_000),
                max_user_turns=1,
                compact_after_messages=1,
            )
        )
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=messages,
            step=1,
            interaction_id="recall-compaction-interaction",
            model_step_id="mstep_00000000000000000000000000000001",
            session_store=sessions,
            knowledge_store=knowledge,
            knowledge_access_scope=scope,
        )

        first = await policy.build_with_checkpoint(request, checkpoint=None)
        first_manifest = _manifest(first)

        assert first.checkpoint is not None
        assert "context_compaction" in first.checkpoint
        assert first.checkpoint["automatic_recall"]["projection"] is not None
        assert "Atlas compaction evidence says Friday" not in json.dumps(
            first.checkpoint["context_compaction"]
        )
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

        second = await policy.build_with_checkpoint(
            request.model_copy(update={"step": 2}),
            checkpoint=first.checkpoint,
        )

        assert _manifest(second) == first_manifest
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

    asyncio.run(run())


def test_automatic_recall_rejects_invalid_frozen_state_without_recalling() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = _policy()
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        corrupted = json.loads(json.dumps(first.checkpoint))
        corrupted["automatic_recall"]["policy_sha256"] = "0" * 64

        with pytest.raises(ContextBuildError, match="checkpoint is invalid"):
            await policy.build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=messages,
                    step=2,
                ),
                checkpoint=corrupted,
            )

        malformed_projection = json.loads(json.dumps(first.checkpoint))
        state = malformed_projection["automatic_recall"]
        state["projection"]["focus"]["items"][0]["fused_rank"] = "1"
        projection_bytes = canonical_durable_json_bytes(
            state["projection"],
            "test malformed projection",
        )
        manifest = _render_projection(state["projection"])
        assert manifest is not None
        state["projection_sha256"] = sha256(projection_bytes).hexdigest()
        state["manifest_sha256"] = sha256(manifest.encode("utf-8")).hexdigest()
        state["projected_bytes"] = len(manifest.encode("utf-8"))

        with pytest.raises(ContextBuildError, match="checkpoint is invalid"):
            await policy.build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=messages,
                    step=2,
                ),
                checkpoint=malformed_projection,
            )

        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

    asyncio.run(run())


def test_automatic_recall_rejects_invalid_runtime_user_marker_without_recalling() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        policy = _policy()
        first = await policy.build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )
        assert first.checkpoint is not None
        next_user = Message.text("user", "Did the Atlas release date change?")
        next_messages = [*messages, next_user]
        malformed = json.loads(json.dumps(first.checkpoint))
        malformed["runtime_authored_user_message"] = {
            "version": 1,
            "anchor_transcript_index": len(next_messages) - 1,
            "user_message_sha256": _message_digest(next_user),
            "unexpected": True,
        }

        with pytest.raises(
            ContextBuildError,
            match="runtime-authored user-message checkpoint is invalid",
        ):
            await policy.build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=next_messages,
                    step=2,
                ),
                checkpoint=malformed,
            )

        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

    asyncio.run(run())


def test_automatic_recall_readmits_a_well_formed_frame_after_policy_change() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        request = _request(
            sessions=sessions,
            knowledge=knowledge,
            session=session,
            messages=messages,
        )
        first = await _policy().build_with_checkpoint(request, checkpoint=None)
        assert first.checkpoint is not None
        first_state = first.checkpoint["automatic_recall"]
        continuation = Message.text("user", "Return the required structured value.")
        continued_messages = [*messages, continuation]
        continued_checkpoint = json.loads(json.dumps(first.checkpoint))
        continued_checkpoint["runtime_authored_user_message"] = {
            "version": 1,
            "anchor_transcript_index": len(continued_messages) - 1,
            "user_message_sha256": _message_digest(continuation),
        }

        changed_admission = _admission().model_copy(
            update={"calibration_version": "automatic-recall-context-calibration-v2"}
        )
        changed_policy = _policy(admission_policy=changed_admission)
        second = await changed_policy.build_with_checkpoint(
            request.model_copy(update={"messages": continued_messages, "step": 2}),
            checkpoint=continued_checkpoint,
        )

        assert second.checkpoint is not None
        second_state = second.checkpoint["automatic_recall"]
        assert first_state["policy_sha256"] != second_state["policy_sha256"]
        assert second_state["policy_sha256"] == changed_admission.fingerprint()
        assert first_state["configuration_sha256"] != second_state["configuration_sha256"]
        assert second_state["configuration_sha256"] == changed_policy.configuration_fingerprint()
        assert second_state["runtime_authored_anchors"] == [
            {
                "anchor_transcript_index": len(continued_messages) - 1,
                "user_message_sha256": _message_digest(continuation),
            }
        ]
        assert knowledge.search_count == 2
        assert sessions.transcript_search_count == 2
        assert [telemetry.event_type for telemetry in second.recall_telemetry] == [
            EventType.AUTOMATIC_RECALL_STARTED,
            EventType.AUTOMATIC_RECALL_COMPLETED,
            EventType.AUTOMATIC_RECALL_ADMITTED,
        ]

    asyncio.run(run())


def test_automatic_recall_readmits_after_non_admission_configuration_change() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        request = _request(
            sessions=sessions,
            knowledge=knowledge,
            session=session,
            messages=messages,
        )
        first_policy = _policy()
        first = await first_policy.build_with_checkpoint(request, checkpoint=None)
        assert first.checkpoint is not None
        first_state = first.checkpoint["automatic_recall"]

        changed_policy = _policy(
            sources=first_policy.sources.model_copy(update={"knowledge_candidate_limit": 10})
        )
        second = await changed_policy.build_with_checkpoint(
            request.model_copy(update={"step": 2}),
            checkpoint=first.checkpoint,
        )

        assert second.checkpoint is not None
        second_state = second.checkpoint["automatic_recall"]
        assert second_state["policy_sha256"] == first_state["policy_sha256"]
        assert second_state["configuration_sha256"] != first_state["configuration_sha256"]
        assert second_state["configuration_sha256"] == changed_policy.configuration_fingerprint()
        assert knowledge.search_count == 2
        assert sessions.transcript_search_count == 2
        assert [telemetry.event_type for telemetry in second.recall_telemetry] == [
            EventType.AUTOMATIC_RECALL_STARTED,
            EventType.AUTOMATIC_RECALL_COMPLETED,
            EventType.AUTOMATIC_RECALL_ADMITTED,
        ]

    asyncio.run(run())


def test_excluded_source_limits_do_not_constrain_enabled_channels() -> None:
    AutomaticRecallContextPolicy(
        admission_policy=_admission(),
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version="automatic-recall-context-tests-v1",
            channel_weights={TRANSCRIPT_LEXICAL_CHANNEL: 1.0},
            max_candidates_per_channel=5,
            fused_head_limit=5,
        ),
        sources=AutomaticRecallSourceConfig(
            include_knowledge=False,
            include_transcript=True,
            knowledge_required=False,
            transcript_candidate_limit=5,
        ),
    )


def test_frozen_automatic_recall_projection_is_secret_redacted_before_checkpoint() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        secret = "Friday"

        with _context_secret_redactor_scope(SecretRedactor(secret)):
            result = await _policy().build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=messages,
                ),
                checkpoint=None,
            )

        assert result.checkpoint is not None
        serialized_checkpoint = json.dumps(result.checkpoint, sort_keys=True)
        assert secret not in serialized_checkpoint
        assert REDACTED_SECRET in serialized_checkpoint
        assert secret not in _manifest(result)

    asyncio.run(run())


def test_frozen_projection_survives_distinct_identities_redacted_to_one_value() -> None:
    async def run() -> None:
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        secret_ids = ("private-atlas-one", "private-atlas-two")
        for entry_id, suffix in zip(secret_ids, ("Friday", "weekend"), strict=True):
            await knowledge.create_entry(
                KnowledgeEntry(
                    id=entry_id,
                    namespace="project:cayu",
                    text=f"Atlas release planning mentions {suffix}.",
                )
            )
        sessions = _CountingSessionStore()
        session = await sessions.create(
            RunRequest(agent_name="assistant", session_id="redacted-identities", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        messages = [Message.text("user", "What does Atlas release planning mention?")]
        await sessions.append_transcript_messages(
            session.id,
            messages,
            interaction_id="interaction-one",
        )

        with _context_secret_redactor_scope(SecretRedactor(secret_ids)):
            first = await _policy().build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=messages,
                ),
                checkpoint=None,
            )
            assert first.checkpoint is not None
            projection = first.checkpoint["automatic_recall"]["projection"]
            assert projection is not None
            assert [item["identity"]["record_id"] for item in projection["focus"]["items"]].count(
                REDACTED_SECRET
            ) == 2

            second = await _policy().build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=messages,
                    step=2,
                ),
                checkpoint=first.checkpoint,
            )

        assert _manifest(second) == _manifest(first)
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

    asyncio.run(run())


def test_recall_locator_redaction_happens_before_json_encoding() -> None:
    secret = 'private"locator'

    serialized = _redacted_locator_json(
        {"entry_id": secret, secret: "value"},
        redactor=SecretRedactor(secret),
    )
    parsed = json.loads(serialized)

    assert secret not in str(parsed)
    assert REDACTED_SECRET in str(parsed)


def test_runtime_protocol_values_survive_secret_value_collisions() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        redactor = SecretRedactor(
            [
                "offer",
                "complete",
                "calibrated_strong_match",
                "knowledge",
                KNOWLEDGE_LEXICAL_CHANNEL,
                _AUTOMATIC_RECALL_NOTICE,
            ]
        )

        with _context_secret_redactor_scope(redactor):
            result = await _policy().build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=messages,
                ),
                checkpoint=None,
            )

        assert result.checkpoint is not None
        safe_checkpoint = require_secret_free_durable_object(
            result.checkpoint,
            redactor=redactor,
            field_name="automatic recall checkpoint",
        )
        projection = safe_checkpoint["automatic_recall"]["projection"]
        assert projection["notice"] == _AUTOMATIC_RECALL_NOTICE
        assert projection["mode"] == "offer_and_strong_matches"
        assert "complete" in {source["status"] for source in projection["sources"]}
        assert projection["sources"][0]["source"] == "knowledge"
        assert KNOWLEDGE_LEXICAL_CHANNEL in projection["sources"][0]["channels"]
        assert projection["focus"]["items"][0]["identity"]["record_type"] == ("knowledge_entry")
        assert projection["focus"]["items"][0]["selection_reason"] == ("calibrated_strong_match")

    asyncio.run(run())


def test_automatic_recall_evidence_bindings_survive_secret_value_collisions() -> None:
    async def run() -> None:
        sessions, knowledge, session, messages = await _fixture()
        result = await _policy().build_with_checkpoint(
            _request(
                sessions=sessions,
                knowledge=knowledge,
                session=session,
                messages=messages,
            ),
            checkpoint=None,
        )

        assert result.checkpoint is not None
        state = result.checkpoint["automatic_recall"]
        redactor = SecretRedactor(
            [
                state["receipt_id"],
                state["receipt_document_sha256"][:16],
                state["receipt_manifest_binding_hmac_sha256"][:16],
            ]
        )
        safe_checkpoint = require_secret_free_durable_object(
            result.checkpoint,
            redactor=redactor,
            field_name="automatic recall checkpoint",
        )

        safe_state = safe_checkpoint["automatic_recall"]
        assert safe_state["receipt_id"] == state["receipt_id"]
        assert safe_state["receipt_document_sha256"] == state["receipt_document_sha256"]
        assert (
            safe_state["receipt_manifest_binding_hmac_sha256"]
            == (state["receipt_manifest_binding_hmac_sha256"])
        )

    asyncio.run(run())


def test_final_request_rejects_duplicate_or_altered_automatic_memory_envelopes() -> None:
    manifest = (
        '<cayu_automatic_memory version="2">\n'
        '{"notice":"trusted runtime envelope"}\n'
        "</cayu_automatic_memory>"
    )
    manifest_sha256 = sha256(manifest.encode("utf-8")).hexdigest()
    delta_manifest = (
        '<cayu_memory_delta version="2" sequence="1">\n'
        '{"notice":"trusted runtime delta"}\n'
        "</cayu_memory_delta>"
    )
    delta_manifest_sha256 = sha256(delta_manifest.encode("utf-8")).hexdigest()
    exact = ModelRequest(
        model="fake-model",
        messages=[Message(role="user", content=(TextPart(text=manifest),))],
    )
    removed = ModelRequest(
        model="fake-model",
        messages=[Message.text("user", "Memory projection removed.")],
    )
    duplicate = ModelRequest(
        model="fake-model",
        messages=[
            Message(
                role="user",
                content=(TextPart(text=manifest), TextPart(text=manifest)),
            )
        ],
    )
    altered = ModelRequest(
        model="fake-model",
        messages=[
            Message(
                role="user",
                content=(
                    TextPart(text=manifest),
                    TextPart(
                        text=(
                            '<cayu_automatic_memory version="2">\n'
                            '{"notice":"altered envelope"}\n'
                            "</cayu_automatic_memory>"
                        )
                    ),
                ),
            )
        ],
    )

    def reference(digest: str | None) -> MemoryEvidenceItemReference:
        return MemoryEvidenceItemReference(
            receipt_id="receipt-one",
            receipt_document_sha256="a" * 64,
            receipt_manifest_binding_hmac_sha256="b" * 64,
            manifest_sha256=digest,
        )

    assert _request_includes_exact_memory_manifests(exact, (reference(manifest_sha256),)) == (True,)
    with pytest.raises(RuntimeError, match="memory changed"):
        _request_includes_exact_memory_manifests(removed, (reference(manifest_sha256),))
    assert _request_includes_exact_memory_manifests(removed, (reference(None),)) == (False,)
    with pytest.raises(RuntimeError, match="memory changed"):
        _request_includes_exact_memory_manifests(duplicate, (reference(manifest_sha256),))
    with pytest.raises(RuntimeError, match="memory changed"):
        _request_includes_exact_memory_manifests(altered, (reference(manifest_sha256),))
    with pytest.raises(RuntimeError, match="memory changed"):
        _request_includes_exact_memory_manifests(altered, (reference(None),))

    exact_sequence = ModelRequest(
        model="fake-model",
        messages=[
            Message(
                role="user",
                content=(
                    TextPart(text=manifest),
                    TextPart(text=delta_manifest),
                    TextPart(text="Original question."),
                ),
            )
        ],
    )
    references = (reference(manifest_sha256), reference(delta_manifest_sha256))
    assert _request_includes_exact_memory_manifests(exact_sequence, references) == (True, True)
    reversed_sequence = exact_sequence.model_copy(
        update={
            "messages": [
                Message(
                    role="user",
                    content=(TextPart(text=delta_manifest), TextPart(text=manifest)),
                )
            ]
        }
    )
    with pytest.raises(RuntimeError, match="memory changed"):
        _request_includes_exact_memory_manifests(reversed_sequence, references)
    moved_sequence = exact_sequence.model_copy(
        update={
            "messages": [
                Message(
                    role="user",
                    content=(
                        TextPart(text="Original question."),
                        TextPart(text=manifest),
                        TextPart(text=delta_manifest),
                    ),
                )
            ]
        }
    )
    with pytest.raises(RuntimeError, match="memory moved"):
        _request_includes_exact_memory_manifests(moved_sequence, references)


def test_transcript_automatic_recall_excludes_the_anchoring_user_message() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        session = await sessions.create(
            RunRequest(agent_name="assistant", session_id="cutoff", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        current = Message.text("user", "Unique current-only evidence")
        await sessions.append_transcript_messages(
            session.id,
            [current],
            interaction_id="interaction-current",
        )
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(TRANSCRIPT_LEXICAL_CHANNEL),
            sources=AutomaticRecallSourceConfig(
                include_knowledge=False,
                include_transcript=True,
                knowledge_required=False,
                transcript_required=True,
            ),
        )
        request = ContextRequest(
            session=session,
            agent=AgentSpec(name="assistant", model="fake-model"),
            messages=[current],
            step=1,
            interaction_id="interaction-current",
            model_step_id="mstep_00000000000000000000000000000002",
            session_store=sessions,
        )

        result = await policy.build_with_checkpoint(request, checkpoint=None)

        assert result.messages == [current]
        assert result.checkpoint is not None
        assert result.checkpoint["automatic_recall"]["projection"] is None
        assert sessions.transcript_search_count == 1

    asyncio.run(run())


def test_runtime_publishes_one_atomic_automatic_recall_outcome_without_content() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        secret_text = "Atlas private release evidence"
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-runtime",
                namespace="project:cayu",
                text=secret_text,
            )
        )
        provider = _RecordingCountScriptedProvider(
            [
                [
                    ModelStreamEvent.text_delta("Friday"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        footprint = RequestFootprintConfig(
            fingerprint_key_id="test-memory-key",
            fingerprint_key="automatic-recall-test-key-material",
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=footprint,
            context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                knowledge_store=knowledge,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-events",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]
        event_types = [event.type for event in events]

        assert event_types.count(EventType.AUTOMATIC_RECALL_STARTED) == 1
        assert event_types.count(EventType.AUTOMATIC_RECALL_COMPLETED) == 1
        assert event_types.count(EventType.AUTOMATIC_RECALL_ADMITTED) == 1
        assert event_types.index(EventType.AUTOMATIC_RECALL_STARTED) < event_types.index(
            EventType.AUTOMATIC_RECALL_COMPLETED
        )
        assert event_types.index(EventType.AUTOMATIC_RECALL_COMPLETED) < event_types.index(
            EventType.AUTOMATIC_RECALL_ADMITTED
        )
        assert event_types.index(EventType.AUTOMATIC_RECALL_ADMITTED) < event_types.index(
            EventType.SESSION_CHECKPOINTED
        )
        recall_events = [
            event
            for event in events
            if event.type
            in {
                EventType.AUTOMATIC_RECALL_STARTED,
                EventType.AUTOMATIC_RECALL_COMPLETED,
                EventType.AUTOMATIC_RECALL_ADMITTED,
            }
        ]
        completed = next(
            event for event in recall_events if event.type is EventType.AUTOMATIC_RECALL_COMPLETED
        )
        admitted = next(
            event for event in recall_events if event.type is EventType.AUTOMATIC_RECALL_ADMITTED
        )
        assert {
            "policy_sha256",
            "configuration_sha256",
            "situation_sha256",
            "recall_candidate_count",
            "evaluated_candidate_count",
            "source_statuses",
            "duration_seconds",
        } <= set(completed.payload)
        assert {
            "contribution_sha256",
            "manifest_sha256",
            "focused_item_count",
            "offered_item_count",
            "silent_item_count",
        } <= set(admitted.payload)
        assert secret_text not in str([event.payload for event in recall_events])
        assert "When is Atlas released?" not in str([event.payload for event in recall_events])
        checkpoint = await sessions.load_checkpoint("automatic-recall-events")
        assert checkpoint is not None
        assert checkpoint["automatic_recall"]["projection"] is not None
        receipts = (
            await sessions.list_recall_receipts(
                RecallEvidenceQuery(session_id="automatic-recall-events")
            )
        ).items
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-events")
            )
        ).items
        assert len(receipts) == 1
        assert len(exposures) == 1
        assert len(provider.count_requests) == 1
        assert provider.count_requests[0] == provider.requests[0]
        receipt = receipts[0]
        exposure = exposures[0]
        assert receipt.receipt_id == checkpoint["automatic_recall"]["receipt_id"]
        assert (
            recall_receipt_document_sha256(receipt)
            == (checkpoint["automatic_recall"]["receipt_document_sha256"])
        )
        evidence_key = memory_evidence_key(footprint)
        assert evidence_key is not None
        assert checkpoint["automatic_recall"][
            "receipt_manifest_binding_hmac_sha256"
        ] == recall_receipt_manifest_binding_hmac_sha256(
            receipt_document_sha256=checkpoint["automatic_recall"]["receipt_document_sha256"],
            manifest_sha256=checkpoint["automatic_recall"]["manifest_sha256"],
            key=evidence_key,
        )
        assert receipt.eligible_count == (
            receipt.admitted_count
            + receipt.offered_count
            + receipt.silent_count
            + receipt.omitted_count
        )
        assert exposure.receipt_ids == (receipt.receipt_id,)
        assert exposure.state is ContextExposureState.COMPLETED
        assert [transition.state for transition in exposure.transitions] == [
            ContextExposureState.PLANNED,
            ContextExposureState.PREPARED,
            ContextExposureState.DISPATCH_STARTED,
            ContextExposureState.ACKNOWLEDGED,
            ContextExposureState.COMPLETED,
        ]
        assert {
            fingerprint.key_id
            for fingerprint in (
                receipt.situation_fingerprint,
                receipt.frontier_fingerprint,
                exposure.composition_fingerprint,
                exposure.request_contract_fingerprint,
            )
        } == {"test-memory-key"}
        item_exposures = await sessions.load_recall_item_exposures(
            exposure.session_id,
            exposure.exposure_id,
        )
        assert len(item_exposures) == len(receipt.items)
        assert all(item.provider_representation_sha256 is not None for item in item_exposures)
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

    asyncio.run(run())


def test_runtime_dispatches_after_wrapped_policy_suppresses_recalled_content() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-suppressed-exposure",
                namespace="project:cayu",
                text="Atlas suppressed evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("No retained memory context."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(_SummarizeAndRemoveUserAnchor()),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-suppressed-exposure",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 1
        assert not any(
            type(part) is TextPart and part.text.startswith('<cayu_automatic_memory version="2">')
            for message in provider.requests[0].messages
            for part in message.content
        )
        receipts = (
            await sessions.list_recall_receipts(
                RecallEvidenceQuery(session_id="automatic-recall-suppressed-exposure")
            )
        ).items
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-suppressed-exposure")
            )
        ).items
        assert len(receipts) == 1
        assert len(exposures) == 1
        assert exposures[0].state is ContextExposureState.COMPLETED
        assert (
            await sessions.load_recall_item_exposures(
                exposures[0].session_id,
                exposures[0].exposure_id,
            )
            == ()
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("policy_type", "expected_error"),
    [
        (
            _ReceiptDigestCorruptingAutomaticRecallPolicy,
            "does not match its durable receipt",
        ),
        (
            _ReceiptManifestBindingCorruptingAutomaticRecallPolicy,
            "receipt-to-manifest binding is invalid",
        ),
        (
            _ReceiptIdentityRemovingAutomaticRecallPolicy,
            "Automatic-recall memory-evidence checkpoint is malformed",
        ),
    ],
    ids=["receipt-document", "receipt-manifest-binding", "missing-receipt-identity"],
)
def test_runtime_rejects_checkpoint_detached_from_its_durable_recall_receipt(
    policy_type,
    expected_error: str,
) -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-detached-receipt",
                namespace="project:cayu",
                text="Atlas receipt evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("must not dispatch"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        policy = policy_type(
            admission_policy=_admission(),
            fusion_config=_fusion(
                KNOWLEDGE_LEXICAL_CHANNEL,
                KNOWLEDGE_SEMANTIC_CHANNEL,
                TRANSCRIPT_LEXICAL_CHANNEL,
            ),
            sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=policy,
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-detached-receipt",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert provider.requests == []
        assert events[-1].type is EventType.SESSION_FAILED
        assert expected_error in events[-1].payload["error"]
        receipts = (
            await sessions.list_recall_receipts(
                RecallEvidenceQuery(session_id="automatic-recall-detached-receipt")
            )
        ).items
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-detached-receipt")
            )
        ).items
        assert len(receipts) == 1
        assert exposures == ()

    asyncio.run(run())


def test_runtime_records_distinct_exposures_for_automatic_recall_retry() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-retry",
                namespace="project:cayu",
                text="Atlas retry evidence says Friday.",
            )
        )
        retry_error = ModelProviderError(
            "provider temporarily unavailable",
            provider="scripted",
            status_code=503,
            retryable=True,
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.error(str(retry_error), cause=retry_error),
                    ModelStreamEvent.completed({"finish_reason": "error"}),
                ],
                [
                    ModelStreamEvent.text_delta("Friday"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=sessions,
            config=CayuConfig(
                run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
            ),
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-retry",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        assert _provider_manifest(provider.requests[0].messages) == _provider_manifest(
            provider.requests[1].messages
        )
        receipts = (
            await sessions.list_recall_receipts(
                RecallEvidenceQuery(session_id="automatic-recall-retry")
            )
        ).items
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-retry")
            )
        ).items
        assert len(receipts) == 1
        assert len(exposures) == 2
        assert [exposure.state for exposure in exposures] == [
            ContextExposureState.FAILED,
            ContextExposureState.COMPLETED,
        ]
        assert [transition.state for transition in exposures[0].transitions] == [
            ContextExposureState.PLANNED,
            ContextExposureState.PREPARED,
            ContextExposureState.DISPATCH_STARTED,
            ContextExposureState.FAILED,
        ]
        assert exposures[0].provider_exposure_proven is False
        assert len({exposure.exposure_id for exposure in exposures}) == 2
        assert len({exposure.model_attempt_id for exposure in exposures}) == 2
        assert len({exposure.provider_attempt_id for exposure in exposures}) == 2
        assert len({exposure.composition_fingerprint.digest for exposure in exposures}) == 1
        assert {exposure.receipt_ids for exposure in exposures} == {(receipts[0].receipt_id,)}
        recovered_terminal = await recover_context_exposure(
            store=sessions,
            session_id="automatic-recall-retry",
            stage_id="completed-failed-model-stage",
            stage_intent={
                "model_step_id": exposures[0].model_step_id,
                "model_attempt_id": exposures[0].model_attempt_id,
                "provider_name": exposures[0].provider_name,
                "requested_model": exposures[0].model_name,
                "context_exposure": {
                    "exposure_id": exposures[0].exposure_id,
                    "provider_attempt_id": exposures[0].provider_attempt_id,
                },
            },
            state=ContextExposureState.COMPLETED,
            evidence_kind=ContextExposureEvidenceKind.RECOVERY_COMPLETION,
            evidence_ref="model-stage:completed-failed-model-stage:completed",
        )
        assert recovered_terminal is not None
        assert recovered_terminal.state is ContextExposureState.FAILED
        assert knowledge.search_count == 1

    asyncio.run(run())


def test_runtime_records_stream_deadline_exposure_as_indeterminate() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-stream-deadline",
                namespace="project:cayu",
                text="Atlas stream deadline evidence says Friday.",
            )
        )
        deadline = ModelStreamDeadlineError(
            provider="scripted",
            evidence=ProviderStreamDeadlineEvidence(
                deadline_kind=ProviderDeadlineKind.SEMANTIC_IDLE,
                configured_timeout_s=30,
                elapsed_s=31,
                last_progress_kind=None,
                last_progress_elapsed_s=None,
                last_progress_at=None,
            ),
        )
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.error(str(deadline), cause=deadline),
                ModelStreamEvent.completed({"finish_reason": "error"}),
            ]
        )
        app = CayuApp(
            session_store=sessions,
            config=CayuConfig(
                run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
            ),
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        with pytest.raises(ModelStreamDeadlineError):
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-stream-deadline",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            ):
                pass

        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-stream-deadline")
            )
        ).items
        assert len(provider.requests) == 1
        assert len(exposures) == 1
        assert exposures[0].state is ContextExposureState.INDETERMINATE
        assert [transition.state for transition in exposures[0].transitions] == [
            ContextExposureState.PLANNED,
            ContextExposureState.PREPARED,
            ContextExposureState.DISPATCH_STARTED,
            ContextExposureState.INDETERMINATE,
        ]
        assert (
            exposures[0].transitions[-1].evidence_kind
            is ContextExposureEvidenceKind.AMBIGUOUS_TRANSPORT
        )

    asyncio.run(run())


def test_runtime_preserves_stream_deadline_when_exposure_terminalization_fails() -> None:
    class IndeterminateExposureFailingStore(_CountingSessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.transition_failures = [
                RuntimeError("context exposure deadline terminalization failed"),
                RuntimeError("context exposure deadline terminalization replay failed"),
            ]
            self.failure_count = 0

        async def transition_context_exposure(self, session_id, exposure_id, request):
            if request.state is ContextExposureState.INDETERMINATE and self.failure_count < len(
                self.transition_failures
            ):
                failure = self.transition_failures[self.failure_count]
                self.failure_count += 1
                raise failure
            return await super().transition_context_exposure(session_id, exposure_id, request)

    class BlockingDeadlineProvider(ModelProvider):
        name = "automatic-recall-deadline"

        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        @property
        def stream_deadlines(self) -> ProviderStreamDeadlines:
            return ProviderStreamDeadlines(
                transport_idle_timeout_s=1,
                protocol_idle_timeout_s=1,
                semantic_progress_timeout_s=0.01,
                absolute_stream_timeout_s=1,
            )

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            await asyncio.Event().wait()
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        sessions = IndeterminateExposureFailingStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-stream-deadline-terminalization",
                namespace="project:cayu",
                text="Atlas stream deadline evidence says Friday.",
            )
        )
        provider = BlockingDeadlineProvider()
        session_id = "automatic-recall-deadline-terminalization-failure"
        app = CayuApp(
            session_store=sessions,
            config=CayuConfig(
                run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
            ),
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        with pytest.raises(ExceptionGroup) as caught:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            ):
                pass

        leaves = [
            failure
            for failure in iter_exception_tree(caught.value)
            if not isinstance(failure, BaseExceptionGroup)
        ]
        deadlines = [failure for failure in leaves if isinstance(failure, ModelStreamDeadlineError)]
        assert len(leaves) == 2
        assert len(deadlines) == 1
        assert sum(failure is sessions.transition_failures[0] for failure in leaves) == 1
        assert sessions.failure_count == 2
        assert len(provider.requests) == 1
        active = await sessions.load_active_model_completion_stage(session_id)
        assert active is not None and active.stage.state == "in_flight"
        exposures = (
            await sessions.list_context_exposures(RecallEvidenceQuery(session_id=session_id))
        ).items
        assert len(exposures) == 1
        assert exposures[0].state is ContextExposureState.DISPATCH_STARTED

    asyncio.run(run())


@pytest.mark.parametrize("overflow_anchor", ["changed", "removed", "retained"])
def test_runtime_recovers_reanchor_overflow_without_reopening_memory_budgets(
    overflow_anchor: str,
) -> None:
    class BoundaryPolicy(RuntimeManagedContextPolicy):
        def __init__(self, *, overflow: bool = False) -> None:
            self.overflow = overflow

        async def build_with_checkpoint(self, request, *, checkpoint):
            if request.step == 1:
                messages = request.messages
            elif self.overflow and overflow_anchor == "removed":
                messages = [Message.text("assistant", "Continue using the available context.")]
            else:
                padding = "" if self.overflow and overflow_anchor == "changed" else " " * 200
                messages = [Message.text("user", padding + "Atlas release evidence?")]
            return ContextBuildResult(
                messages=messages,
                checkpoint={**(checkpoint or {}), "overflow_context_selected": self.overflow},
                checkpoint_event_payload={"checkpoint": "overflow_context_selected"},
            )

    async def run() -> None:
        sessions = _CountingSessionStore()
        knowledge = _CountingKnowledgeStore(
            access_scope=KnowledgeAccessScope.for_namespace("project:cayu")
        )
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-reanchor-overflow",
                namespace="project:cayu",
                text="Verified Atlas release evidence says Friday.",
            )
        )
        session_id = f"reanchor-overflow-{overflow_anchor}"
        rejected_checkpoint = None
        work_at_dispatch = []

        class OverflowProvider(ScriptedModelProvider):
            async def stream(self, request):
                nonlocal rejected_checkpoint
                batch = self._consume_batch(request)
                work_at_dispatch.append(
                    (knowledge.revision_search_count, sessions.context_exposure_list_count)
                )
                if len(self.requests) == 2:
                    rejected_checkpoint = await sessions.load_checkpoint(session_id)
                    raise ModelContextOverflowError(
                        "context too large",
                        provider="scripted",
                        status_code=400,
                        error_code="context_length_exceeded",
                    )
                for event in batch:
                    yield event

        provider = OverflowProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call_reanchor_overflow", name="complete_memory_boundary", arguments={}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                [
                    ModelStreamEvent.text_delta("Completed."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        def policy(*, overflow: bool) -> AutomaticRecallContextPolicy:
            return AutomaticRecallContextPolicy(
                BoundaryPolicy(overflow=overflow),
                admission_policy=_admission(),
                fusion_config=_fusion(
                    KNOWLEDGE_LEXICAL_CHANNEL,
                    KNOWLEDGE_SEMANTIC_CHANNEL,
                    TRANSCRIPT_LEXICAL_CHANNEL,
                ),
                sources=AutomaticRecallSourceConfig(knowledge_namespace="project:cayu"),
                delta_policy=MemoryDeltaPolicy(reanchor_on_projection_loss=True),
            )

        overflow_policy = policy(overflow=True)
        footprint = RequestFootprintConfig(
            fingerprint_key_id="test-memory-key",
            fingerprint_key="automatic-recall-test-key-material",
        )
        app = CayuApp(session_store=sessions, request_footprint=footprint, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=policy(overflow=False),
            context_overflow_policy=overflow_policy,
            tools=[_NoopMemoryBoundaryTool()],
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 3
        assert work_at_dispatch[1] == work_at_dispatch[2]
        assert rejected_checkpoint is not None
        checkpoint = await sessions.load_checkpoint(session_id)
        assert checkpoint is not None
        assert checkpoint["overflow_context_selected"] is True
        before = rejected_checkpoint["automatic_recall"]["delta_state"]
        after = checkpoint["automatic_recall"]["delta_state"]
        assert len(before["deltas"]) == len(after["deltas"]) == 1
        assert {k: v for k, v in before.items() if k != "deltas"} == {
            k: v for k, v in after.items() if k != "deltas"
        }
        placement_fields = {
            "receipt_manifest_binding_hmac_sha256",
            "projection_sha256",
            "manifest_sha256",
            "projection",
            "projected_bytes",
            "projected_estimated_tokens",
        }
        original, current = before["deltas"][0], after["deltas"][0]
        assert {k: v for k, v in original.items() if k not in placement_fields} == {
            k: v for k, v in current.items() if k not in placement_fields
        }
        assert original["projection"] is not None
        assert (current["projection"] is not None) == (overflow_anchor == "retained")
        final_manifests = [
            part.text
            for message in provider.requests[-1].messages
            for part in message.content
            if type(part) is TextPart and part.text.startswith("<cayu_memory_delta")
        ]
        assert len(final_manifests) == (1 if overflow_anchor == "retained" else 0)
        receipts = await sessions.list_recall_receipts(RecallEvidenceQuery(session_id=session_id))
        assert len(receipts.items) == 2

        # Reload the signed checkpoint at the same boundary: no recall, new
        # receipt, reopened budget, or resurrection of a suppressed placement.
        session = await sessions.load(session_id)
        snapshot = await sessions.load_transcript_snapshot(session_id)
        work_before_retry = (knowledge.revision_search_count, sessions.context_exposure_list_count)
        with memory_evidence_key_scope(memory_evidence_key(footprint)):
            retry = await overflow_policy.build_with_checkpoint(
                _request(
                    sessions=sessions,
                    knowledge=knowledge,
                    session=session,
                    messages=[record.message for record in snapshot.records],
                    step=2,
                    interaction_id=checkpoint["automatic_recall"]["interaction_id"],
                    model_step_id=current["trigger"]["model_step_id"],
                ),
                checkpoint=checkpoint,
            )
        assert retry.checkpoint == checkpoint  # The wrapped policy returns its checkpoint.
        assert work_before_retry == (
            knowledge.revision_search_count,
            sessions.context_exposure_list_count,
        )
        retry_manifests = [
            part.text
            for message in retry.messages
            for part in message.content
            if type(part) is TextPart and part.text.startswith("<cayu_memory_delta")
        ]
        assert retry_manifests == final_manifests

    asyncio.run(run())


def test_runtime_rebuilds_automatic_recall_exposure_after_context_overflow() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-overflow",
                namespace="project:cayu",
                text="Atlas overflow evidence says Friday.",
            )
        )
        overflow = ModelContextOverflowError(
            "context too large",
            provider="scripted",
            status_code=400,
            error_code="context_length_exceeded",
        )
        provider = _FirstRequestRaisingScriptedProvider(
            overflow,
            (
                ModelStreamEvent.text_delta("Friday"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
            context_overflow_policy=_policy(RecentTurnsContextPolicy(max_user_turns=1)),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-overflow",
                    messages=[
                        Message.text("user", "Old Atlas question one."),
                        Message.text("assistant", "Old answer one."),
                        Message.text("user", "Old Atlas question two."),
                        Message.text("assistant", "Old answer two."),
                        Message.text("user", "When is Atlas released?"),
                    ],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 2
        assert len(provider.requests[1].messages) < len(provider.requests[0].messages)
        assert _provider_manifest(provider.requests[0].messages) == _provider_manifest(
            provider.requests[1].messages
        )
        receipts = (
            await sessions.list_recall_receipts(
                RecallEvidenceQuery(session_id="automatic-recall-overflow")
            )
        ).items
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-overflow")
            )
        ).items
        assert len(receipts) == 1
        assert [exposure.state for exposure in exposures] == [
            ContextExposureState.FAILED,
            ContextExposureState.COMPLETED,
        ]
        assert len({exposure.composition_fingerprint.digest for exposure in exposures}) == 2
        assert {exposure.receipt_ids for exposure in exposures} == {(receipts[0].receipt_id,)}
        assert knowledge.search_count == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    ("grouped", "provider_effect_observed", "terminal_state"),
    [
        (False, False, ContextExposureState.FAILED),
        (True, False, ContextExposureState.FAILED),
        (False, True, ContextExposureState.INDETERMINATE),
        (True, True, ContextExposureState.INDETERMINATE),
    ],
    ids=[
        "single-pre-effect",
        "grouped-pre-effect",
        "single-after-effect",
        "grouped-after-effect",
    ],
)
def test_runtime_settles_raised_authentication_failure_with_truthful_provider_effect(
    grouped: bool,
    provider_effect_observed: bool,
    terminal_state: ContextExposureState,
) -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-authentication-rejection",
                namespace="project:cayu",
                text="Atlas authentication evidence says Friday.",
            )
        )
        authentication_failure = ModelProviderError(
            "authentication failed",
            provider="scripted",
            status_code=401,
            error_type="authentication_error",
            retryable=False,
        )
        raised_failure = (
            ExceptionGroup("provider authentication failed", [authentication_failure])
            if grouped
            else authentication_failure
        )
        provider = (
            _ProviderEffectThenRaisingScriptedProvider(raised_failure)
            if provider_effect_observed
            else _FirstRequestRaisingScriptedProvider(raised_failure)
        )
        session_id = "automatic-recall-authentication-failure"
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_FAILED
        assert len(provider.requests) == 1
        assert await sessions.load_active_model_completion_stage(session_id) is None
        exposures = (
            await sessions.list_context_exposures(RecallEvidenceQuery(session_id=session_id))
        ).items
        assert len(exposures) == 1
        assert [transition.state for transition in exposures[0].transitions] == [
            ContextExposureState.PLANNED,
            ContextExposureState.PREPARED,
            ContextExposureState.DISPATCH_STARTED,
            *([ContextExposureState.ACKNOWLEDGED] if provider_effect_observed else []),
            terminal_state,
        ]

    asyncio.run(run())


def test_runtime_fails_closed_before_provider_when_dispatch_evidence_cannot_persist() -> None:
    async def run() -> None:
        class RecordingBudgetLedger(InMemoryBudgetLedger):
            def __init__(self) -> None:
                super().__init__()
                self.dispatch_calls = 0

            async def mark_dispatched(self, **kwargs):
                self.dispatch_calls += 1
                return await super().mark_dispatched(**kwargs)

        sessions = _DispatchEvidenceFailingSessionStore()
        ledger = RecordingBudgetLedger()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-evidence-failure",
                namespace="project:cayu",
                text="Atlas evidence says Friday.",
            )
        )
        provider = _RecordingCountScriptedProvider(
            [
                ModelStreamEvent.text_delta("must not dispatch"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(
            session_store=sessions,
            budget_ledger=ledger,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=Decimal("1"),
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name="scripted",
                                    model="fake-model",
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("1"),
                                ),
                            )
                        ),
                        reservation=BudgetReservation(
                            max_input_tokens=1_000,
                            max_output_tokens=1_000,
                        ),
                    ),
                )
            ),
            context_counting=ContextCountingConfig(mode=ContextCountingMode.OBSERVE),
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-evidence-failure",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert provider.requests == []
        assert provider.count_requests == []
        assert ledger.dispatch_calls == 0
        assert events[-1].type is EventType.SESSION_FAILED
        assert events[-1].payload["error"] == ("context exposure dispatch persistence failed")
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-evidence-failure")
            )
        ).items
        assert len(exposures) == 1
        assert exposures[0].state is ContextExposureState.FAILED

    asyncio.run(run())


def test_runtime_replays_exact_memory_evidence_after_store_acknowledgement_loss() -> None:
    async def run() -> None:
        sessions = _EvidenceAcknowledgementLosingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-evidence-replay",
                namespace="project:cayu",
                text="Atlas evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("Friday"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-evidence-replay",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len(provider.requests) == 1
        receipts = (
            await sessions.list_recall_receipts(
                RecallEvidenceQuery(session_id="automatic-recall-evidence-replay")
            )
        ).items
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-evidence-replay")
            )
        ).items
        assert len(receipts) == 1
        assert len(exposures) == 1
        assert exposures[0].state is ContextExposureState.COMPLETED

    asyncio.run(run())


def test_runtime_recovers_original_memory_exposure_after_background_process_loss() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-background-recovery",
                namespace="project:cayu",
                text="Atlas background recovery evidence says Friday.",
            )
        )
        provider = _MemoryRecoveryOperationProvider()
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        with pytest.raises(_MemoryRecoveryProcessLoss):
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="automatic-recall-background-recovery",
                        messages=[Message.text("user", "When is Atlas released?")],
                    )
                )
            ]

        before = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-background-recovery")
            )
        ).items
        assert len(before) == 1
        assert before[0].state is ContextExposureState.ACKNOWLEDGED

        provider.adapter.status = ProviderOperationStatus.COMPLETED
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="automatic-recall-background-recovery",
                inactive_for_seconds=0,
            )
        )

        receipts = (
            await sessions.list_recall_receipts(
                RecallEvidenceQuery(session_id="automatic-recall-background-recovery")
            )
        ).items
        after = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-background-recovery")
            )
        ).items
        assert len(receipts) == 1
        assert len(after) == 1
        assert after[0].exposure_id == before[0].exposure_id
        assert after[0].state is ContextExposureState.COMPLETED
        assert after[0].receipt_ids == (receipts[0].receipt_id,)
        assert provider.adapter.start_calls == 1
        assert provider.adapter.retrieve_calls == 1
        assert knowledge.search_count == 1

    asyncio.run(run())


def test_runtime_closes_exposure_when_model_dispatch_receipt_cannot_persist() -> None:
    async def run() -> None:
        sessions = _StageDispatchReceiptFailingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-model-dispatch-receipt",
                namespace="project:cayu",
                text="Atlas model dispatch receipt evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("must not dispatch"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-model-dispatch-receipt",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_FAILED
        assert provider.requests == []
        assert (
            await sessions.load_active_model_completion_stage(
                "automatic-recall-model-dispatch-receipt"
            )
            is None
        )
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-model-dispatch-receipt")
            )
        ).items
        assert len(exposures) == 1
        assert [transition.state for transition in exposures[0].transitions] == [
            ContextExposureState.PLANNED,
            ContextExposureState.PREPARED,
            ContextExposureState.DISPATCH_STARTED,
            ContextExposureState.FAILED,
        ]

    asyncio.run(run())


def test_runtime_closes_partially_created_exposure_when_reconciliation_fails() -> None:
    async def run() -> None:
        sessions = _ExposureCreationReconciliationFailingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-exposure-creation-reconciliation",
                namespace="project:cayu",
                text="Atlas exposure creation evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("must not dispatch"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-exposure-creation-reconciliation",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert provider.requests == []
        assert events[-1].type is EventType.SESSION_FAILED
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-exposure-creation-reconciliation")
            )
        ).items
        assert len(exposures) == 1
        assert [transition.state for transition in exposures[0].transitions] == [
            ContextExposureState.PLANNED,
            ContextExposureState.FAILED,
        ]

    asyncio.run(run())


@pytest.mark.parametrize(
    "provider",
    [
        _TimeoutBeforeAcknowledgementScriptedProvider(
            [ModelStreamEvent.completed({"finish_reason": "stop"})]
        ),
        _MalformedAcknowledgementScriptedProvider(
            [ModelStreamEvent.completed({"finish_reason": "stop"})]
        ),
    ],
    ids=["timeout", "malformed-event"],
)
def test_runtime_requires_typed_provider_event_before_acknowledgement(provider) -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-timeout",
                namespace="project:cayu",
                text="Atlas timeout evidence says Friday.",
            )
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-timeout",
                    messages=[Message.text("user", "When is Atlas released?")],
                    # This test proves the state of one ambiguous dispatch, not
                    # the runtime's independent provider retry behavior.
                    retry_policy=RetryPolicy(max_attempts=1),
                )
            )
        ]

        assert events[-1].type is EventType.SESSION_FAILED
        assert len(provider.requests) == 1
        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-timeout")
            )
        ).items
        assert len(exposures) == 1
        assert [transition.state for transition in exposures[0].transitions] == [
            ContextExposureState.PLANNED,
            ContextExposureState.PREPARED,
            ContextExposureState.DISPATCH_STARTED,
            ContextExposureState.INDETERMINATE,
        ]

    asyncio.run(run())


def test_runtime_records_unconfirmed_stream_cancellation_as_indeterminate() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-cancellation",
                namespace="project:cayu",
                text="Atlas cancellation evidence says Friday.",
            )
        )
        provider = _BlockingBeforeAcknowledgementScriptedProvider()
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        async def collect() -> list[Any]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="automatic-recall-cancellation",
                        messages=[Message.text("user", "When is Atlas released?")],
                    )
                )
            ]

        task = asyncio.create_task(collect())
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        exposures = (
            await sessions.list_context_exposures(
                RecallEvidenceQuery(session_id="automatic-recall-cancellation")
            )
        ).items
        assert len(exposures) == 1
        assert exposures[0].state is ContextExposureState.INDETERMINATE
        assert ContextExposureState.ACKNOWLEDGED not in {
            transition.state for transition in exposures[0].transitions
        }

    asyncio.run(run())


def test_runtime_requires_keyed_evidence_configuration_before_automatic_recall() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-missing-key",
                namespace="project:cayu",
                text="Atlas evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("must not dispatch"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=sessions, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-missing-key",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]

        assert provider.requests == []
        assert knowledge.search_count == 0
        assert events[-1].type is EventType.SESSION_FAILED
        assert "keyed request-footprint configuration" in events[-1].payload["error"]
        assert not (
            await sessions.list_recall_receipts(
                RecallEvidenceQuery(session_id="automatic-recall-missing-key")
            )
        ).items

    asyncio.run(run())


def test_fork_discards_source_recall_frame_and_recalls_for_child_interaction() -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-fork",
                namespace="project:cayu",
                text="Atlas fork evidence says Friday.",
            )
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("Friday"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
                [
                    ModelStreamEvent.text_delta("Still Friday"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        _ = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="automatic-recall-fork-source",
                    messages=[Message.text("user", "When is Atlas released?")],
                )
            )
        ]
        source_checkpoint = await sessions.load_checkpoint("automatic-recall-fork-source")
        assert source_checkpoint is not None
        assert "automatic_recall" in source_checkpoint

        fork_events = [
            event
            async for event in app.fork_session(
                ForkSessionRequest(
                    source_session_id="automatic-recall-fork-source",
                    session_id="automatic-recall-fork-child",
                )
            )
        ]
        assert fork_events[-1].type is EventType.SESSION_FORKED
        child_checkpoint = await sessions.load_checkpoint("automatic-recall-fork-child")
        assert child_checkpoint is not None
        assert "automatic_recall" not in child_checkpoint

        child_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="automatic-recall-fork-child",
                    messages=[Message.text("user", "Is Atlas still released Friday?")],
                )
            )
        ]
        assert child_events[-1].type is EventType.SESSION_COMPLETED
        assert [event.type for event in child_events].count(EventType.AUTOMATIC_RECALL_STARTED) == 1
        assert len(provider.requests) == 2
        assert knowledge.search_count == 2
        assert sessions.transcript_search_count == 2
        child_checkpoint = await sessions.load_checkpoint("automatic-recall-fork-child")
        assert child_checkpoint is not None
        assert child_checkpoint["automatic_recall"]["session_id"] == ("automatic-recall-fork-child")

    asyncio.run(run())


@pytest.mark.parametrize("continuation_kind", ["structured_repair", "before_stop"])
def test_runtime_reuses_frozen_recall_for_runtime_authored_user_continuations(
    continuation_kind: str,
) -> None:
    async def run() -> None:
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-continuation",
                namespace="project:cayu",
                text="Atlas continuation evidence says Friday.",
            )
        )
        if continuation_kind == "structured_repair":
            provider = ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("not json"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                    [
                        ModelStreamEvent.tool_call(
                            id="call_structured_repair",
                            name=STRUCTURED_OUTPUT_TOOL_NAME,
                            arguments={"output": {"answer": "Friday"}},
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                ]
            )
            request_updates: dict[str, Any] = {
                "structured_output": StructuredOutputSpec(
                    json_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    max_retries=1,
                )
            }
        else:
            provider = ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("draft"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("final"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            )
            request_updates = {
                "max_steps": 2,
                "loop_policies": (_ContinueOnceBeforeStop(),),
            }
        app = CayuApp(
            session_store=sessions,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(),
        )

        request = RunRequest(
            agent_name="assistant",
            session_id=f"automatic-recall-{continuation_kind}",
            messages=[Message.text("user", "When is Atlas released?")],
            **request_updates,
        )
        events = [event async for event in app.run(request)]

        assert len(provider.requests) == 2
        assert _provider_manifest(provider.requests[0].messages) == _provider_manifest(
            provider.requests[1].messages
        )
        latest_user = provider.requests[1].messages[-1]
        assert latest_user.role is MessageRole.USER
        assert all(
            not (
                type(part) is TextPart
                and part.text.startswith('<cayu_automatic_memory version="2">')
            )
            for part in latest_user.content
        )
        event_types = [event.type for event in events]
        assert event_types.count(EventType.AUTOMATIC_RECALL_STARTED) == 1
        assert event_types.count(EventType.AUTOMATIC_RECALL_ADMITTED) == 1
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1
        checkpoint = await sessions.load_checkpoint(request.session_id or "")
        assert checkpoint is not None
        assert len(checkpoint["automatic_recall"]["runtime_authored_anchors"]) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "query,expected", [("Weather tomorrow?", "weather"), ("And in production?", "atlas-release")]
)
def test_real_context_resolves_query_without_assistant_contamination(query, expected):
    async def run():
        sessions, knowledge, session, previous = await _fixture()
        await knowledge.create_entry(
            KnowledgeEntry(id="weather", namespace="project:cayu", text="Weather tomorrow sunny.")
        )
        messages = [
            *previous,
            Message.text("user", "Atlas release"),
            Message.text("assistant", "irrelevant picnic tables " * 100),
            Message.text("user", query),
        ]
        await sessions.append_transcript_messages(
            session.id, messages[len(previous) :], interaction_id="query-test"
        )
        policy = AutomaticRecallContextPolicy(
            admission_policy=_admission(),
            fusion_config=_fusion(KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL),
            sources=AutomaticRecallSourceConfig(
                knowledge_namespace="project:cayu", include_transcript=False
            ),
        )
        first = await policy.build_with_checkpoint(
            _request(sessions=sessions, knowledge=knowledge, session=session, messages=messages),
            checkpoint=None,
        )
        projection = first.checkpoint["automatic_recall"]["projection"]
        receipt = await sessions.load_recall_receipt(
            session.id, first.checkpoint["automatic_recall"]["receipt_id"]
        )
        assert receipt.query_resolution.decision == (
            "independent_query" if expected == "weather" else "resolved_followup"
        )
        assert not receipt.query_resolution.context_clipped
        assert not receipt.query_resolution.query_clipped

        assert {
            json.loads(item["locator_json"])["entry_id"] for item in projection["focus"]["items"]
        } == {expected}
        replay = await policy.build_with_checkpoint(
            _request(
                sessions=sessions, knowledge=knowledge, session=session, messages=messages, step=2
            ),
            checkpoint=first.checkpoint,
        )
        assert replay.checkpoint is None  # Unchanged checkpoints are omitted.
        assert _provider_manifest(replay.messages) == _provider_manifest(first.messages)
        assert knowledge.search_count == 1

    asyncio.run(run())


def test_presentation_transition_rejects_previous_frozen_checkpoint_version():
    async def run():
        sessions, knowledge, session, messages = await _fixture()
        policy = _policy()
        request = _request(
            sessions=sessions, knowledge=knowledge, session=session, messages=messages
        )
        first = await policy.build_with_checkpoint(request, checkpoint=None)
        checkpoint = json.loads(json.dumps(first.checkpoint))
        assert checkpoint["automatic_recall"]["version"] == 5
        checkpoint["automatic_recall"]["version"] = 4
        with pytest.raises(ContextBuildError, match="checkpoint is invalid"):
            await policy.build_with_checkpoint(request, checkpoint=checkpoint)
        assert knowledge.search_count == 1

    asyncio.run(run())
