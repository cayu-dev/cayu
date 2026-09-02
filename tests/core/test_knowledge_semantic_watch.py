from __future__ import annotations

import asyncio
import sqlite3
from copy import deepcopy
from typing import TypedDict, cast

import pytest
from examples.knowledge_semantic_watch import main as semantic_watch_example

import cayu
from cayu.knowledge_semantic_watch import (
    MAX_KNOWLEDGE_SEMANTIC_WATCH_POLICY_REQUEST_BYTES,
    KnowledgeSemanticWatchAuthority,
    KnowledgeSemanticWatchConfig,
    KnowledgeSemanticWatchConflict,
    KnowledgeSemanticWatchDecision,
    KnowledgeSemanticWatchDisposition,
    KnowledgeSemanticWatchEvaluator,
    KnowledgeSemanticWatchEvidence,
    KnowledgeSemanticWatchPolicyError,
    KnowledgeSemanticWatchRequest,
    knowledge_semantic_watch_request_fingerprint,
    prepare_knowledge_semantic_watch_invocation,
)
from cayu.recall import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    RECALL_MAX_KNOWLEDGE_GROUPED_ASPECT_BYTES,
    KnowledgeRecallSource,
    RecallEngine,
    RecallSituation,
    RecallSource,
)
from cayu.retrieval import WeightedReciprocalRankFusionConfig
from cayu.storage.knowledge_sqlite import SQLiteKnowledgeStore
from cayu.storage.memory import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeStore,
)
from cayu.storage.migrations import SchemaMode


def _engine(store: KnowledgeStore) -> RecallEngine:
    return RecallEngine(
        (KnowledgeRecallSource(store),),
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version="semantic-watch-tests-v1",
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
            },
            max_candidates_per_channel=20,
            fused_head_limit=20,
        ),
    )


def _config(**updates: object) -> KnowledgeSemanticWatchConfig:
    values: dict[str, object] = {
        "watch_identity": "watch",
        "watch_version": "1",
        "recall_profile_identity": "recall-profile",
        "recall_profile_version": "1",
        "policy_identity": "application-policy",
        "policy_version": "1",
    }
    values.update(updates)
    return KnowledgeSemanticWatchConfig.model_validate(values)


class _EvaluationValues(TypedDict):
    operation_id: str
    observation_id: str
    observation_source_type: str
    observation_source_id: str
    observation_text: str


def _evaluation_values(
    *,
    operation_id: str = "watch-operation",
    observation_text: str = "Atlas deployment is approaching Friday.",
) -> _EvaluationValues:
    return {
        "operation_id": operation_id,
        "observation_id": "observation",
        "observation_source_type": "event",
        "observation_source_id": "event-1",
        "observation_text": observation_text,
    }


class _DecisionPolicy:
    def __init__(
        self,
        disposition: KnowledgeSemanticWatchDisposition,
        *,
        request_sha256: str | None = None,
        policy_identity: str = "application-policy",
    ) -> None:
        self.disposition = disposition
        self.request_sha256 = request_sha256
        self.policy_identity = policy_identity
        self.calls = 0

    async def decide_semantic_watch(
        self,
        request: KnowledgeSemanticWatchRequest,
    ) -> KnowledgeSemanticWatchDecision:
        self.calls += 1
        return KnowledgeSemanticWatchDecision(
            request_sha256=self.request_sha256 or request.fingerprint,
            disposition=self.disposition,
            policy_identity=self.policy_identity,
            policy_version="1",
            code="test-decision",
        )


class _MalformedObjectPolicy:
    async def decide_semantic_watch(self, request: KnowledgeSemanticWatchRequest):
        del request
        return {"disposition": "emit"}


class _RaisingPolicy:
    async def decide_semantic_watch(self, request: KnowledgeSemanticWatchRequest):
        del request
        raise RuntimeError("private policy failure")


class _RaisingPolicyLookup:
    @property
    def decide_semantic_watch(self):
        raise RuntimeError("private policy lookup failure")


class _OversizedPolicy:
    async def decide_semantic_watch(
        self,
        request: KnowledgeSemanticWatchRequest,
    ) -> KnowledgeSemanticWatchDecision:
        return KnowledgeSemanticWatchDecision.model_construct(
            schema_version=1,
            request_sha256=request.fingerprint,
            disposition=KnowledgeSemanticWatchDisposition.EMIT,
            policy_identity="application-policy",
            policy_version="1",
            code="oversized",
            annotations={"unsafe": "x" * 5_000},
        )


class _RequiredFailingSource(RecallSource):
    name = "knowledge"
    channel_names = (KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL)

    def __init__(self) -> None:
        super().__init__(required=True, candidate_limit=20)

    async def retrieve(self, situation):
        del situation
        raise RuntimeError("private recall failure")


class _OversizedEvidenceEngine(RecallEngine):
    async def recall(self, situation):
        result = await super().recall(situation)
        source = result.sources[0].model_copy(
            update={
                "source": "x" * (MAX_KNOWLEDGE_SEMANTIC_WATCH_POLICY_REQUEST_BYTES + 1),
            }
        )
        return result.model_copy(update={"sources": (source,)})


class _MismatchedSituationEngine(RecallEngine):
    async def recall(self, situation):
        mismatched = situation.model_copy(update={"query": "different recall situation"})
        return await super().recall(mismatched)


async def _prepared_store() -> InMemoryKnowledgeStore:
    store = InMemoryKnowledgeStore(access_scope=KnowledgeAccessScope.privileged())
    await store.create_entry(
        KnowledgeEntry(
            id="atlas-release",
            text="Atlas deploys on Friday. private-knowledge-material",
        )
    )
    return store


def test_semantic_watch_supports_lexical_only_no_match_and_safe_receipts() -> None:
    async def run() -> None:
        store = await _prepared_store()
        emit = _DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT)
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=_config(),
            policy=emit,
        )
        receipt = await evaluator.evaluate(**_evaluation_values())
        serialized = receipt.model_dump_json(warnings=False)
        assert receipt.authority.evidence.complete is True
        assert receipt.authority.evidence.truncation_reasons == ("channel:knowledge.semantic",)
        assert receipt.authority.evidence.required_channels == (KNOWLEDGE_LEXICAL_CHANNEL,)
        channel_match = receipt.authority.evidence.candidates[0].channel_matches[0]
        assert channel_match.raw_score is not None
        assert channel_match.reasons == ("entry text match",)
        assert "private-knowledge-material" not in serialized
        assert "Atlas deployment is approaching Friday." not in serialized

        ignore = _DecisionPolicy(KnowledgeSemanticWatchDisposition.IGNORE)
        no_match = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=_config(),
            policy=ignore,
        )
        ignored = await no_match.evaluate(
            **_evaluation_values(
                operation_id="no-match-operation",
                observation_text="unrelatedquasarword",
            )
        )
        assert ignored.authority.evidence.candidates == ()
        assert ignored.authority.decision.disposition is KnowledgeSemanticWatchDisposition.IGNORE

    asyncio.run(run())


def test_semantic_watch_preserves_valid_negative_feature_adjusted_fusion_scores() -> None:
    async def run() -> None:
        store = await _prepared_store()
        engine = RecallEngine(
            (KnowledgeRecallSource(store),),
            fusion_config=WeightedReciprocalRankFusionConfig(
                configuration_version="semantic-watch-negative-feature-v1",
                channel_weights={
                    KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                    KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
                },
                feature_weights={"current_revision": -1.0},
                max_candidates_per_channel=20,
                fused_head_limit=20,
            ),
        )
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            engine,
            config=_config(),
            policy=_DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT),
        )
        receipt = await evaluator.evaluate(**_evaluation_values())
        assert receipt.authority.evidence.candidates[0].fused_score < 0

    asyncio.run(run())


def test_semantic_watch_evidence_cross_binds_fusion_counts_and_lane_identity() -> None:
    async def run() -> None:
        store = await _prepared_store()
        await store.create_entry(
            KnowledgeEntry(
                id="atlas-release-secondary",
                text="Atlas also deploys on Friday.",
            )
        )
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=_config(),
            policy=_DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT),
        )
        receipt = await evaluator.evaluate(**_evaluation_values())
        payload = receipt.authority.evidence.model_dump(mode="python", warnings=False)

        conflicting_counts = deepcopy(payload)
        conflicting_counts["fusion"]["unique_candidate_count"] += 1
        conflicting_counts["fusion"]["omitted_candidate_count"] += 1
        with pytest.raises(ValueError, match="conflict with fusion diagnostics"):
            KnowledgeSemanticWatchEvidence.model_validate(conflicting_counts)

        conflicting_lane = deepcopy(payload)
        conflicting_lane["candidates"][0]["channel_matches"][0]["index_version"] = "forged"
        with pytest.raises(ValueError, match="lane evidence is inconsistent"):
            KnowledgeSemanticWatchEvidence.model_validate(conflicting_lane)

        duplicate_lane_rank = deepcopy(payload)
        assert len(duplicate_lane_rank["candidates"]) == 2
        first_match = duplicate_lane_rank["candidates"][0]["channel_matches"][0]
        second_candidate = duplicate_lane_rank["candidates"][1]
        second_match = second_candidate["channel_matches"][0]
        second_match["rank"] = first_match["rank"]
        second_candidate["best_rank"] = min(
            match["rank"] for match in second_candidate["channel_matches"]
        )
        with pytest.raises(ValueError, match="lane ranks must be unique"):
            KnowledgeSemanticWatchEvidence.model_validate(duplicate_lane_rank)

        reasonless_lane = deepcopy(payload)
        reasonless_lane["candidates"][0]["channel_matches"][0]["reasons"] = []
        with pytest.raises(ValueError, match="reasons.*between 1 and"):
            KnowledgeSemanticWatchEvidence.model_validate(reasonless_lane)

    asyncio.run(run())


def test_semantic_watch_rejects_evidence_from_a_different_recall_situation() -> None:
    async def run() -> None:
        store = await _prepared_store()
        await store.create_entry(
            KnowledgeEntry(
                id="atlas-release-secondary",
                text="Atlas also deploys on Friday.",
            )
        )
        scope = store.bound_access_scope()
        assert scope is not None
        config = _config()
        observation_text = _evaluation_values()["observation_text"]
        invocation = prepare_knowledge_semantic_watch_invocation(
            operation_id="mismatched-situation-request",
            observation_id="observation",
            observation_source_type="event",
            observation_source_id="event-1",
            observation_text=observation_text,
            access_scope=scope,
            config=config,
        )
        situation = RecallSituation(
            query=observation_text,
            knowledge_access_scope=scope,
            knowledge_namespace=config.knowledge_namespace,
        )
        result = await _engine(store).recall(situation)
        evidence = cayu.project_knowledge_semantic_watch_evidence(
            result,
            max_candidates=config.max_candidates,
        )
        assert len(evidence.candidates) == 2
        mismatched = situation.model_copy(update={"query": "different recall situation"})
        with pytest.raises(ValueError, match="recall situation conflicts"):
            KnowledgeSemanticWatchRequest(
                invocation=invocation,
                observation_text=observation_text,
                recall_situation=mismatched,
                evidence=evidence,
            )

        evidence_payload = evidence.model_dump(mode="python", warnings=False)
        evidence_payload["situation_sha256"] = "0" * 64
        mismatched_evidence = KnowledgeSemanticWatchEvidence.model_validate(evidence_payload)
        with pytest.raises(ValueError, match="evidence conflicts"):
            KnowledgeSemanticWatchRequest(
                invocation=invocation,
                observation_text=observation_text,
                recall_situation=situation,
                evidence=mismatched_evidence,
            )

        hybrid_invocation = type(invocation).model_validate(
            {
                **invocation.model_dump(mode="python", warnings=False),
                "required_channels": (
                    KNOWLEDGE_LEXICAL_CHANNEL,
                    KNOWLEDGE_SEMANTIC_CHANNEL,
                ),
            }
        )
        with pytest.raises(ValueError, match="required recall lanes"):
            KnowledgeSemanticWatchRequest(
                invocation=hybrid_invocation,
                observation_text=observation_text,
                recall_situation=situation,
                evidence=evidence,
            )
        hybrid_decision = KnowledgeSemanticWatchDecision(
            request_sha256=knowledge_semantic_watch_request_fingerprint(
                hybrid_invocation,
                evidence,
            ),
            disposition=KnowledgeSemanticWatchDisposition.EMIT,
            policy_identity="application-policy",
            policy_version="1",
            code="mismatched-required-lanes",
        )
        with pytest.raises(ValueError, match="required recall lanes"):
            KnowledgeSemanticWatchAuthority(
                invocation=hybrid_invocation,
                evidence=evidence,
                decision=hybrid_decision,
            )

        bounded_invocation = type(invocation).model_validate(
            {
                **invocation.model_dump(mode="python", warnings=False),
                "max_candidates": 1,
            }
        )
        with pytest.raises(ValueError, match="candidate bound"):
            KnowledgeSemanticWatchRequest(
                invocation=bounded_invocation,
                observation_text=observation_text,
                recall_situation=situation,
                evidence=evidence,
            )
        bounded_decision = KnowledgeSemanticWatchDecision(
            request_sha256=knowledge_semantic_watch_request_fingerprint(
                bounded_invocation,
                evidence,
            ),
            disposition=KnowledgeSemanticWatchDisposition.EMIT,
            policy_identity="application-policy",
            policy_version="1",
            code="mismatched-candidate-bound",
        )
        with pytest.raises(ValueError, match="candidate bound"):
            KnowledgeSemanticWatchAuthority(
                invocation=bounded_invocation,
                evidence=evidence,
                decision=bounded_decision,
            )

        request = KnowledgeSemanticWatchRequest(
            invocation=invocation,
            observation_text=observation_text,
            recall_situation=situation,
            evidence=evidence,
        )
        mismatched_policy = _DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT)
        with pytest.raises(KnowledgeSemanticWatchPolicyError) as mismatched_profile:
            await cayu.decide_knowledge_semantic_watch(
                request,
                config=_config(watch_identity="different-watch"),
                policy=mismatched_policy,
            )
        assert mismatched_profile.value.code == "policy_request_invalid"
        assert mismatched_policy.calls == 0

        policy = _DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT)
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            _MismatchedSituationEngine(
                (KnowledgeRecallSource(store),),
                fusion_config=WeightedReciprocalRankFusionConfig(
                    configuration_version="semantic-watch-mismatched-situation-v1",
                    channel_weights={
                        KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                        KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
                    },
                    max_candidates_per_channel=20,
                    fused_head_limit=20,
                ),
            ),
            config=config,
            policy=policy,
        )
        with pytest.raises(KnowledgeSemanticWatchPolicyError) as failure:
            await evaluator.evaluate(
                **_evaluation_values(operation_id="mismatched-situation-evaluation")
            )
        assert failure.value.code == "recall_failed"
        assert policy.calls == 0
        assert await store.load_semantic_watch_receipt("mismatched-situation-evaluation") is None

    asyncio.run(run())


def test_semantic_watch_decision_annotations_are_recursively_immutable() -> None:
    decision = KnowledgeSemanticWatchDecision(
        request_sha256="0" * 64,
        disposition=KnowledgeSemanticWatchDisposition.IGNORE,
        policy_identity="application-policy",
        policy_version="1",
        code="immutable-annotations",
        annotations={"nested": ["safe-code"]},
    )

    with pytest.raises(TypeError):
        cast("dict[str, object]", decision.annotations)["changed"] = True
    with pytest.raises(TypeError):
        decision.annotations["nested"].append("changed")
    assert decision.model_dump(mode="json")["annotations"] == {"nested": ["safe-code"]}


def test_semantic_watch_invocation_access_scope_is_recursively_immutable() -> None:
    invocation = prepare_knowledge_semantic_watch_invocation(
        operation_id="immutable-scope-operation",
        observation_id="immutable-scope-observation",
        observation_source_type="test",
        observation_source_id="immutable-scope-source",
        observation_text="Atlas deployment Friday",
        access_scope=KnowledgeAccessScope.for_namespace(
            "project:atlas",
            required_labels={"environment": "production"},
            allowed_source_types=["deployment"],
            allowed_source_ids=["deployment-42"],
        ),
        config=_config(knowledge_namespace="project:atlas"),
    )
    fingerprint = invocation.fingerprint

    frozen_sequences = (
        invocation.access_scope.allowed_namespaces,
        invocation.access_scope.allowed_visibilities,
        invocation.access_scope.allowed_source_types,
        invocation.access_scope.allowed_source_ids,
        invocation.access_scope.allowed_statuses,
    )
    for values in frozen_sequences:
        assert values is not None
        with pytest.raises(TypeError):
            cast("list[object]", values).append(values[0])
    with pytest.raises(TypeError):
        invocation.access_scope.required_labels["environment"] = "development"

    assert invocation.fingerprint == fingerprint
    copied = type(invocation).model_validate(invocation.model_dump(mode="python", warnings=False))
    assert copied == invocation
    assert copied.fingerprint == fingerprint


def test_semantic_watch_requires_review_for_incomplete_required_lane() -> None:
    async def run() -> None:
        store = await _prepared_store()
        hybrid = _config(required_channels=(KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL))
        invalid_emit = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=hybrid,
            policy=_DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT),
        )
        with pytest.raises(KnowledgeSemanticWatchPolicyError) as invalid:
            await invalid_emit.evaluate(**_evaluation_values())
        assert invalid.value.code == "policy_output_invalid"
        assert await store.load_semantic_watch_receipt("watch-operation") is None

        review = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=hybrid,
            policy=_DecisionPolicy(KnowledgeSemanticWatchDisposition.ROUTE_TO_REVIEW),
        )
        receipt = await review.evaluate(**_evaluation_values())
        assert receipt.authority.evidence.complete is False
        assert (
            receipt.authority.decision.disposition
            is KnowledgeSemanticWatchDisposition.ROUTE_TO_REVIEW
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("policy", "expected_code"),
    (
        (None, "policy_missing"),
        (
            _DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT, request_sha256="0" * 64),
            "policy_output_invalid",
        ),
        (
            _DecisionPolicy(
                KnowledgeSemanticWatchDisposition.EMIT,
                policy_identity="different-policy",
            ),
            "policy_output_invalid",
        ),
        (_MalformedObjectPolicy(), "policy_output_invalid"),
        (_RaisingPolicy(), "policy_failed"),
        (_RaisingPolicyLookup(), "policy_failed"),
        (_OversizedPolicy(), "policy_output_invalid"),
    ),
)
def test_semantic_watch_policy_failures_create_no_receipt(policy, expected_code: str) -> None:
    async def run() -> None:
        store = await _prepared_store()
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=_config(),
            policy=policy,
        )
        with pytest.raises(KnowledgeSemanticWatchPolicyError) as failure:
            await evaluator.evaluate(**_evaluation_values())
        assert failure.value.code == expected_code
        assert await store.load_semantic_watch_receipt("watch-operation") is None

    asyncio.run(run())


def test_semantic_watch_required_recall_failure_creates_no_receipt() -> None:
    async def run() -> None:
        store = await _prepared_store()
        engine = RecallEngine(
            (_RequiredFailingSource(),),
            fusion_config=WeightedReciprocalRankFusionConfig(
                configuration_version="semantic-watch-failing-source-v1",
                channel_weights={
                    KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                    KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
                },
                max_candidates_per_channel=20,
                fused_head_limit=20,
            ),
        )
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            engine,
            config=_config(),
            policy=_DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT),
        )
        with pytest.raises(KnowledgeSemanticWatchPolicyError) as failure:
            await evaluator.evaluate(**_evaluation_values())
        assert failure.value.code == "recall_unavailable"
        assert await store.load_semantic_watch_receipt("watch-operation") is None

    asyncio.run(run())


def test_semantic_watch_rejects_an_oversized_projected_policy_request() -> None:
    async def run() -> None:
        store = await _prepared_store()
        engine = _OversizedEvidenceEngine(
            (KnowledgeRecallSource(store),),
            fusion_config=WeightedReciprocalRankFusionConfig(
                configuration_version="semantic-watch-oversized-request-v1",
                channel_weights={
                    KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                    KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
                },
                max_candidates_per_channel=20,
                fused_head_limit=20,
            ),
        )
        policy = _DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT)
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            engine,
            config=_config(),
            policy=policy,
        )
        with pytest.raises(KnowledgeSemanticWatchPolicyError) as failure:
            await evaluator.evaluate(**_evaluation_values())
        assert failure.value.code == "policy_request_invalid"
        assert policy.calls == 0
        assert await store.load_semantic_watch_receipt("watch-operation") is None

    asyncio.run(run())


def test_semantic_watch_rejects_self_authorizing_profiles() -> None:
    with pytest.raises(ValueError, match="cannot authorize"):
        _config(policy_identity="watch")
    with pytest.raises(ValueError, match="cannot authorize"):
        _config(policy_identity="recall-profile")


def test_semantic_watch_rejects_aspects_above_the_recall_aggregate_bound() -> None:
    groups = tuple(
        tuple(f"{chr(97 + group_index)}{item_index}{'x' * 216}" for item_index in range(100))
        for group_index in range(6)
    )
    assert (
        sum(len(item.encode("utf-8")) for group in groups for item in group)
        > RECALL_MAX_KNOWLEDGE_GROUPED_ASPECT_BYTES
    )

    with pytest.raises(ValueError, match="recall aggregate UTF-8 byte bound"):
        _config(knowledge_aspect_groups=groups)


def test_semantic_watch_timeout_and_cancellation_create_no_receipt() -> None:
    class WaitingPolicy:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def decide_semantic_watch(
            self,
            request: KnowledgeSemanticWatchRequest,
        ) -> KnowledgeSemanticWatchDecision:
            self.started.set()
            await self.release.wait()
            return KnowledgeSemanticWatchDecision(
                request_sha256=request.fingerprint,
                disposition=KnowledgeSemanticWatchDisposition.EMIT,
                policy_identity="application-policy",
                policy_version="1",
                code="released",
            )

    async def run() -> None:
        timeout_store = await _prepared_store()
        timeout_policy = WaitingPolicy()
        timeout_evaluator = KnowledgeSemanticWatchEvaluator(
            timeout_store,
            _engine(timeout_store),
            config=_config(policy_timeout_seconds=0.001),
            policy=timeout_policy,
        )
        with pytest.raises(KnowledgeSemanticWatchPolicyError) as timeout:
            await timeout_evaluator.evaluate(**_evaluation_values())
        assert timeout.value.code == "policy_timed_out"
        assert await timeout_store.load_semantic_watch_receipt("watch-operation") is None

        cancelled_store = await _prepared_store()
        cancelled_policy = WaitingPolicy()
        cancelled_evaluator = KnowledgeSemanticWatchEvaluator(
            cancelled_store,
            _engine(cancelled_store),
            config=_config(),
            policy=cancelled_policy,
        )
        evaluation = asyncio.create_task(cancelled_evaluator.evaluate(**_evaluation_values()))
        await cancelled_policy.started.wait()
        evaluation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await evaluation
        assert await cancelled_store.load_semantic_watch_receipt("watch-operation") is None

    asyncio.run(run())


def test_semantic_watch_recovers_from_lost_commit_acknowledgement() -> None:
    class AckLossStore(InMemoryKnowledgeStore):
        def __init__(self) -> None:
            super().__init__(access_scope=KnowledgeAccessScope.privileged())
            self.lose_ack = True

        async def record_semantic_watch_outcome(self, authority, *, access_scope=None):
            receipt = await super().record_semantic_watch_outcome(
                authority,
                access_scope=access_scope,
            )
            if self.lose_ack:
                self.lose_ack = False
                raise RuntimeError("lost acknowledgement")
            return receipt

    async def run() -> None:
        store = AckLossStore()
        await store.create_entry(
            KnowledgeEntry(id="atlas-release", text="Atlas deploys on Friday.")
        )
        policy = _DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT)
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=_config(),
            policy=policy,
        )
        with pytest.raises(RuntimeError, match="lost acknowledgement"):
            await evaluator.evaluate(**_evaluation_values())
        recovered = await evaluator.evaluate(**_evaluation_values())
        assert recovered.replayed is True
        assert policy.calls == 1

    asyncio.run(run())


def test_semantic_watch_rejects_a_candidate_that_turns_stale_before_commit() -> None:
    async def run() -> None:
        store = await _prepared_store()

        class StalingPolicy:
            async def decide_semantic_watch(
                self,
                request: KnowledgeSemanticWatchRequest,
            ) -> KnowledgeSemanticWatchDecision:
                current = await store.get_entry("atlas-release")
                assert current is not None
                await store.append_entry_revision(
                    current.model_copy(
                        update={
                            "revision": current.revision + 1,
                            "text": "The release moved to Monday.",
                        }
                    ),
                    expected_revision=current.revision,
                )
                return KnowledgeSemanticWatchDecision(
                    request_sha256=request.fingerprint,
                    disposition=KnowledgeSemanticWatchDisposition.EMIT,
                    policy_identity="application-policy",
                    policy_version="1",
                    code="now-stale",
                )

        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=_config(),
            policy=StalingPolicy(),
        )
        with pytest.raises(KnowledgeSemanticWatchConflict) as stale:
            await evaluator.evaluate(**_evaluation_values())
        assert stale.value.code == "candidate_stale"
        assert await store.load_semantic_watch_receipt("watch-operation") is None

    asyncio.run(run())


def test_semantic_watch_store_rejects_forged_candidate_material() -> None:
    async def run() -> None:
        store = await _prepared_store()
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=_config(),
            policy=_DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT),
        )
        receipt = await evaluator.evaluate(**_evaluation_values())
        invocation = receipt.authority.invocation.model_copy(
            update={"operation_id": "forged-material-operation"}
        )
        evidence_payload = receipt.authority.evidence.model_dump(mode="python", warnings=False)
        evidence_payload["candidates"][0]["content_hash"] = "0" * 64
        evidence = KnowledgeSemanticWatchEvidence.model_validate(evidence_payload)
        decision = KnowledgeSemanticWatchDecision(
            request_sha256=knowledge_semantic_watch_request_fingerprint(invocation, evidence),
            disposition=KnowledgeSemanticWatchDisposition.EMIT,
            policy_identity="application-policy",
            policy_version="1",
            code="forged-material",
        )
        authority = KnowledgeSemanticWatchAuthority(
            invocation=invocation,
            evidence=evidence,
            decision=decision,
        )

        with pytest.raises(KnowledgeSemanticWatchConflict) as stale:
            await store.record_semantic_watch_outcome(authority)
        assert stale.value.code == "candidate_stale"
        assert await store.load_semantic_watch_receipt("forged-material-operation") is None

    asyncio.run(run())


def test_semantic_watch_replay_preserves_committed_historical_attribution() -> None:
    async def run() -> None:
        store = await _prepared_store()
        policy = _DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT)
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=_config(),
            policy=policy,
        )
        first = await evaluator.evaluate(**_evaluation_values())
        current = await store.get_entry("atlas-release")
        assert current is not None
        await store.append_entry_revision(
            current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "text": "The release moved to Monday.",
                }
            ),
            expected_revision=current.revision,
        )
        replay = await evaluator.evaluate(**_evaluation_values())
        assert replay.replayed is True
        assert replay.authority.evidence.candidates[0].reference.revision == 1
        assert replay.model_copy(update={"replayed": False}) == first
        assert policy.calls == 1

    asyncio.run(run())


def test_public_semantic_watch_exports_are_stable() -> None:
    for name in (
        "KnowledgeSemanticWatchAuthority",
        "KnowledgeSemanticWatchCandidate",
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
        "decide_knowledge_semantic_watch",
        "load_knowledge_semantic_watch_receipt",
        "prepare_knowledge_semantic_watch_invocation",
        "project_knowledge_semantic_watch_evidence",
    ):
        assert name in cayu.__all__
        assert getattr(cayu, name) is not None


def test_deterministic_semantic_watch_example(capsys) -> None:
    asyncio.run(semantic_watch_example())
    assert capsys.readouterr().out == (
        "{\n"
        '  "disposition": "emit",\n'
        '  "exact_revisions": [\n'
        "    {\n"
        '      "entry_id": "release-window",\n'
        '      "revision": 1\n'
        "    }\n"
        "  ],\n"
        '  "provider_calls": 0,\n'
        '  "replayed": true\n'
        "}\n"
    )


def test_sqlite_revision_78_adds_empty_watch_storage_without_inference(tmp_path) -> None:
    database = tmp_path / "revision-78-semantic-watch.sqlite"

    async def seed() -> None:
        store = SQLiteKnowledgeStore(
            database,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        try:
            await store.create_entry(
                KnowledgeEntry(
                    id="historical-entry",
                    text="A historical observation must not be evaluated during migration.",
                )
            )
        finally:
            await store.close()

    asyncio.run(seed())
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE cayu_knowledge_semantic_watch_receipts")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision = 78")
        connection.execute("PRAGMA user_version = 77")
        connection.commit()

    migrated = SQLiteKnowledgeStore(
        database,
        schema_mode=SchemaMode.MIGRATE,
        access_scope=KnowledgeAccessScope.privileged(),
    )

    async def verify() -> None:
        try:
            assert await migrated.get_entry("historical-entry") is not None
            assert (
                migrated._connection.execute(
                    "SELECT COUNT(*) FROM cayu_knowledge_semantic_watch_receipts"
                ).fetchone()[0]
                == 0
            )
        finally:
            await migrated.close()

    asyncio.run(verify())


def test_sqlite_revision_78_rejects_malformed_watch_storage(tmp_path) -> None:
    database = tmp_path / "revision-78-malformed-semantic-watch.sqlite"
    store = SQLiteKnowledgeStore(database)
    asyncio.run(store.close())
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE cayu_knowledge_semantic_watch_receipts")
        connection.execute(
            "CREATE TABLE cayu_knowledge_semantic_watch_receipts (operation_id TEXT PRIMARY KEY)"
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="semantic-watch receipt contract"):
        SQLiteKnowledgeStore(database)


def test_sqlite_semantic_watch_reopens_and_detects_corrupt_indexes(tmp_path) -> None:
    database = tmp_path / "semantic-watch-reopen.sqlite"

    async def publish() -> None:
        store = SQLiteKnowledgeStore(
            database,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        try:
            await store.create_entry(
                KnowledgeEntry(id="atlas-release", text="Atlas deploys on Friday.")
            )
            evaluator = KnowledgeSemanticWatchEvaluator(
                store,
                _engine(store),
                config=_config(),
                policy=_DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT),
            )
            await evaluator.evaluate(**_evaluation_values())
        finally:
            await store.close()

    asyncio.run(publish())

    async def replay() -> None:
        store = SQLiteKnowledgeStore(
            database,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        policy = _DecisionPolicy(KnowledgeSemanticWatchDisposition.IGNORE)
        try:
            evaluator = KnowledgeSemanticWatchEvaluator(
                store,
                _engine(store),
                config=_config(),
                policy=policy,
            )
            receipt = await evaluator.evaluate(**_evaluation_values())
            assert receipt.replayed is True
            assert receipt.authority.decision.disposition is KnowledgeSemanticWatchDisposition.EMIT
            assert policy.calls == 0
        finally:
            await store.close()

    asyncio.run(replay())
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE cayu_knowledge_semantic_watch_receipts "
            "SET invocation_sha256 = ? WHERE operation_id = ?",
            ("0" * 64, "watch-operation"),
        )
        connection.commit()

    async def reject_corruption() -> None:
        store = SQLiteKnowledgeStore(
            database,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        try:
            with pytest.raises(KnowledgeSemanticWatchConflict) as malformed:
                await store.load_semantic_watch_receipt("watch-operation")
            assert malformed.value.code == "malformed_receipt"
        finally:
            await store.close()

    asyncio.run(reject_corruption())


def test_in_memory_semantic_watch_detects_corrupt_indexes() -> None:
    async def run() -> None:
        store = await _prepared_store()
        evaluator = KnowledgeSemanticWatchEvaluator(
            store,
            _engine(store),
            config=_config(),
            policy=_DecisionPolicy(KnowledgeSemanticWatchDisposition.EMIT),
        )
        receipt = await evaluator.evaluate(**_evaluation_values())
        store._semantic_watch_receipts[receipt.operation_id] = receipt.model_copy(
            update={"invocation_sha256": "0" * 64}
        )

        with pytest.raises(KnowledgeSemanticWatchConflict) as malformed:
            await store.load_semantic_watch_receipt(receipt.operation_id)
        assert malformed.value.code == "malformed_receipt"

    asyncio.run(run())


async def _drop_postgres_schema(postgres_dsn: str) -> None:
    import psycopg
    from psycopg import sql

    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = current_schema() AND tablename LIKE 'cayu_%'"
            )
            for (table,) in await cursor.fetchall():
                await cursor.execute(
                    sql.SQL("DROP TABLE {} CASCADE").format(sql.Identifier(str(table)))
                )
        await connection.commit()


def test_postgres_revision_78_adds_empty_watch_storage_without_inference(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu.storage.postgres import PostgresKnowledgeStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        try:
            await creator.create_entry(
                KnowledgeEntry(
                    id="historical-entry",
                    text="Historical material must not become a semantic-watch outcome.",
                )
            )
        finally:
            await creator.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_knowledge_semantic_watch_receipts")
                await cursor.execute("DELETE FROM cayu_schema_migrations WHERE revision = 78")
            await connection.commit()

        migrated = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.MIGRATE,
            access_scope=KnowledgeAccessScope.privileged(),
        )
        try:
            await migrated.ensure_schema()
            assert await migrated.get_entry("historical-entry") is not None
            async with migrated._pool.connection() as connection, connection.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) FROM cayu_knowledge_semantic_watch_receipts")
                assert await cursor.fetchone() == (0,)
        finally:
            await migrated.close()

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_postgres_schema(postgres_dsn))


def test_postgres_revision_78_rejects_malformed_watch_storage(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        import psycopg

        from cayu.storage.postgres import PostgresKnowledgeStore

        await _drop_postgres_schema(postgres_dsn)
        creator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await creator.ensure_schema()
        finally:
            await creator.close()
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("DROP TABLE cayu_knowledge_semantic_watch_receipts")
                await cursor.execute(
                    "CREATE TABLE cayu_knowledge_semantic_watch_receipts "
                    "(operation_id TEXT PRIMARY KEY)"
                )
            await connection.commit()

        validator = PostgresKnowledgeStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            with pytest.raises(RuntimeError, match="semantic-watch receipt contract"):
                await validator.ensure_schema()
        finally:
            await validator.close()

    try:
        asyncio.run(run())
    finally:
        asyncio.run(_drop_postgres_schema(postgres_dsn))
