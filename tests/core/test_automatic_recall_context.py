from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from typing import Any

import pytest

from cayu import (
    BeforeStopContext,
    BeforeStopDecision,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ForkSessionRequest,
    IncompleteSessionRecoveryRequest,
    LoopPolicy,
    ModelStreamEvent,
    ResumeRequest,
    ScriptedModelProvider,
    StructuredOutputSpec,
)
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
from cayu.memory import AutomaticRecallPolicy
from cayu.memory_evidence import (
    ContextExposureEvidenceKind,
    ContextExposureState,
    RecallEvidenceQuery,
)
from cayu.providers import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.providers.base import ModelRequest
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
    MemoryEvidenceKey,
    _request_includes_exact_frozen_manifest,
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
    ContextPolicy,
    ContextRequest,
    RecentTurnsContextPolicy,
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
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope, KnowledgeEntry
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _CountingKnowledgeStore(InMemoryKnowledgeStore):
    def __init__(self, *, access_scope: KnowledgeAccessScope) -> None:
        super().__init__(access_scope=access_scope)
        self.search_count = 0

    async def search(self, query, *, access_scope=None):
        self.search_count += 1
        return await super().search(query, access_scope=access_scope)


@pytest.fixture(autouse=True)
def _direct_context_memory_evidence_key():
    with memory_evidence_key_scope(MemoryEvidenceKey(key_id="test-memory-key", key=b"m" * 32)):
        yield


class _CountingSessionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.transcript_search_count = 0

    async def search_transcript(self, query):
        self.transcript_search_count += 1
        return await super().search_transcript(query)


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

    async def mark_model_completion_stage_dispatched(self, session_id, *, stage):
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
                            and part.text.startswith('<cayu_automatic_memory version="1">')
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


class _ContinueOnceBeforeStop(LoopPolicy):
    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        if context.step == 1:
            return BeforeStopDecision.continue_with(
                Message.text("user", "Correct the final answer."),
                reason="test continuation",
            )
        return BeforeStopDecision.complete("second step is final")


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
    knowledge: _CountingKnowledgeStore,
    session: Any,
    messages: list[Message],
    step: int = 1,
) -> ContextRequest:
    return ContextRequest(
        session=session,
        agent=AgentSpec(name="assistant", model="fake-model"),
        messages=messages,
        step=step,
        interaction_id=f"interaction-{step}",
        model_step_id="mstep_00000000000000000000000000000000",
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
        if type(part) is TextPart and part.text.startswith('<cayu_automatic_memory version="1">')
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
        assert first_manifest.count('<cayu_automatic_memory version="1">') == 1
        assert first_manifest.count("</cayu_automatic_memory>") == 1
        assert "\\u003c/cayu_automatic_memory\\u003e" in first_manifest
        assert messages == original
        assert first.checkpoint is not None
        assert first.checkpoint["automatic_recall"]["anchor_transcript_index"] == 1
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

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
                and part.text.startswith('<cayu_automatic_memory version="1">')
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
        assert _manifest(retained).startswith('<cayu_automatic_memory version="1">')
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
                and part.text.startswith('<cayu_automatic_memory version="1">')
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
        '<cayu_automatic_memory version="1">\n'
        '{"notice":"trusted runtime envelope"}\n'
        "</cayu_automatic_memory>"
    )
    manifest_sha256 = sha256(manifest.encode("utf-8")).hexdigest()
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
                            '<cayu_automatic_memory version="1">\n'
                            '{"notice":"altered envelope"}\n'
                            "</cayu_automatic_memory>"
                        )
                    ),
                ),
            )
        ],
    )

    assert _request_includes_exact_frozen_manifest(exact, manifest_sha256) is True
    assert _request_includes_exact_frozen_manifest(removed, manifest_sha256) is False
    assert _request_includes_exact_frozen_manifest(removed, None) is False
    with pytest.raises(RuntimeError, match="manifest changed"):
        _request_includes_exact_frozen_manifest(duplicate, manifest_sha256)
    with pytest.raises(RuntimeError, match="manifest changed"):
        _request_includes_exact_frozen_manifest(altered, manifest_sha256)
    with pytest.raises(RuntimeError, match="manifest changed"):
        _request_includes_exact_frozen_manifest(altered, None)


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
            type(part) is TextPart and part.text.startswith('<cayu_automatic_memory version="1">')
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
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
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
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
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
        await asyncio.wait_for(provider.started.wait(), timeout=2)
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
                and part.text.startswith('<cayu_automatic_memory version="1">')
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
