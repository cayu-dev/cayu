from __future__ import annotations

import asyncio
import json
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
    ForkSessionRequest,
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
from cayu.runtime.context import (
    CheckpointCompactionContextPolicy,
    ContextBuildError,
    ContextPolicy,
    ContextRequest,
    TranscriptDigestCompactor,
    _context_secret_redactor_scope,
)
from cayu.runtime.memory_context import (
    _AUTOMATIC_RECALL_NOTICE,
    AutomaticRecallContextPolicy,
    AutomaticRecallSourceConfig,
    _message_digest,
    _redacted_locator_json,
    _render_projection,
)
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


class _CountingSessionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.transcript_search_count = 0

    async def search_transcript(self, query):
        self.transcript_search_count += 1
        return await super().search_transcript(query)


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
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.text_delta("Friday"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        app = CayuApp(session_store=sessions, enable_logging=False)
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
        assert knowledge.search_count == 1
        assert sessions.transcript_search_count == 1

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
