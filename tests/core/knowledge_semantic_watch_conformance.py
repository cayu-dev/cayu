from __future__ import annotations

import asyncio
from typing import TypedDict

from cayu.knowledge_semantic_watch import (
    KnowledgeSemanticWatchConfig,
    KnowledgeSemanticWatchConflict,
    KnowledgeSemanticWatchDecision,
    KnowledgeSemanticWatchDisposition,
    KnowledgeSemanticWatchEvaluator,
    KnowledgeSemanticWatchReceipt,
    KnowledgeSemanticWatchRequest,
)
from cayu.recall import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    KnowledgeRecallSource,
    RecallEngine,
)
from cayu.retrieval import WeightedReciprocalRankFusionConfig
from cayu.storage.memory import (
    KnowledgeAccessDenied,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeRevisionRef,
    KnowledgeStore,
)


def _engine(store: KnowledgeStore) -> RecallEngine:
    return RecallEngine(
        (KnowledgeRecallSource(store),),
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version="semantic-watch-conformance-v1",
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
            },
            max_candidates_per_channel=20,
            fused_head_limit=20,
        ),
    )


def _config() -> KnowledgeSemanticWatchConfig:
    return KnowledgeSemanticWatchConfig(
        watch_identity="release-warning",
        watch_version="1",
        recall_profile_identity="release-warning-recall",
        recall_profile_version="1",
        policy_identity="application-release-policy",
        policy_version="1",
    )


class _EvaluationValues(TypedDict):
    operation_id: str
    observation_id: str
    observation_source_type: str
    observation_source_id: str
    observation_text: str


class _Policy:
    def __init__(self) -> None:
        self.calls = 0

    async def decide_semantic_watch(
        self,
        request: KnowledgeSemanticWatchRequest,
    ) -> KnowledgeSemanticWatchDecision:
        self.calls += 1
        return KnowledgeSemanticWatchDecision(
            request_sha256=request.fingerprint,
            disposition=(
                KnowledgeSemanticWatchDisposition.EMIT
                if request.evidence.candidates
                else KnowledgeSemanticWatchDisposition.IGNORE
            ),
            policy_identity="application-release-policy",
            policy_version="1",
            code="release-evidence" if request.evidence.candidates else "no-match",
        )


class _ConcurrentPolicy(_Policy):
    def __init__(self) -> None:
        super().__init__()
        self._both_started = asyncio.Event()

    async def decide_semantic_watch(
        self,
        request: KnowledgeSemanticWatchRequest,
    ) -> KnowledgeSemanticWatchDecision:
        self.calls += 1
        call = self.calls
        if call == 2:
            self._both_started.set()
        await asyncio.wait_for(self._both_started.wait(), timeout=2)
        return KnowledgeSemanticWatchDecision(
            request_sha256=request.fingerprint,
            disposition=(
                KnowledgeSemanticWatchDisposition.EMIT
                if request.evidence.candidates
                else KnowledgeSemanticWatchDisposition.IGNORE
            ),
            policy_identity="application-release-policy",
            policy_version="1",
            code=f"concurrent-decision-{call}",
        )


async def assert_knowledge_semantic_watch_conformance(store: KnowledgeStore) -> None:
    entry = await store.create_entry(
        KnowledgeEntry(
            id="semantic-watch-release",
            text="Project Atlas releases on Friday after the migration window.",
        )
    )
    policy = _Policy()
    evaluator = KnowledgeSemanticWatchEvaluator(
        store,
        _engine(store),
        config=_config(),
        policy=policy,
    )
    values: _EvaluationValues = {
        "operation_id": "semantic-watch-evaluation",
        "observation_id": "deployment-observation",
        "observation_source_type": "deployment",
        "observation_source_id": "deployment-42",
        "observation_text": "Atlas migration is scheduled before Friday.",
    }
    first = await evaluator.evaluate(**values)
    assert first.replayed is False
    assert first.authority.decision.disposition is KnowledgeSemanticWatchDisposition.EMIT
    assert [item.reference for item in first.authority.evidence.candidates] == [
        KnowledgeRevisionRef(entry_id=entry.id, revision=entry.revision)
    ]
    assert values["observation_text"] not in first.model_dump_json(warnings=False)
    assert policy.calls == 1

    replay = await evaluator.evaluate(**values)
    assert replay.replayed is True
    assert replay.model_copy(update={"replayed": False}) == first
    assert policy.calls == 1

    conflict_values: _EvaluationValues = {
        **values,
        "observation_text": "A conflicting operation observation.",
    }
    try:
        await evaluator.evaluate(**conflict_values)
    except KnowledgeSemanticWatchConflict as exc:
        assert exc.code == "operation_reuse"
    else:  # pragma: no cover - conformance assertion
        raise AssertionError("Conflicting semantic-watch operation reuse must fail.")
    assert policy.calls == 1

    concurrent_policy = _ConcurrentPolicy()
    concurrent_evaluator = KnowledgeSemanticWatchEvaluator(
        store,
        _engine(store),
        config=_config(),
        policy=concurrent_policy,
    )
    concurrent_values: _EvaluationValues = {
        **values,
        "operation_id": "semantic-watch-concurrent",
        "observation_id": "concurrent-observation",
    }
    left, right = await asyncio.gather(
        concurrent_evaluator.evaluate(**concurrent_values),
        concurrent_evaluator.evaluate(**concurrent_values),
    )
    assert isinstance(left, KnowledgeSemanticWatchReceipt)
    assert isinstance(right, KnowledgeSemanticWatchReceipt)
    assert left.model_copy(update={"replayed": False}) == right.model_copy(
        update={"replayed": False}
    )
    assert {left.replayed, right.replayed} == {False, True}
    assert concurrent_policy.calls == 2

    conflicting_policy = _ConcurrentPolicy()
    conflicting_evaluator = KnowledgeSemanticWatchEvaluator(
        store,
        _engine(store),
        config=_config(),
        policy=conflicting_policy,
    )
    conflicting_values: _EvaluationValues = {
        **values,
        "operation_id": "semantic-watch-concurrent-conflict",
        "observation_id": "concurrent-conflict-observation",
    }
    left_conflicting_values: _EvaluationValues = {
        **conflicting_values,
        "observation_text": "Atlas migration is scheduled before Friday.",
    }
    right_conflicting_values: _EvaluationValues = {
        **conflicting_values,
        "observation_text": "A different observation reuses the operation.",
    }
    conflict_results = await asyncio.gather(
        conflicting_evaluator.evaluate(**left_conflicting_values),
        conflicting_evaluator.evaluate(**right_conflicting_values),
        return_exceptions=True,
    )
    assert (
        sum(isinstance(result, KnowledgeSemanticWatchReceipt) for result in conflict_results) == 1
    )
    conflicts = [
        result for result in conflict_results if isinstance(result, KnowledgeSemanticWatchConflict)
    ]
    assert len(conflicts) == 1
    assert conflicts[0].code == "operation_reuse"
    assert conflicting_policy.calls == 2

    loaded = await store.load_semantic_watch_receipt(values["operation_id"])
    assert loaded == first

    class _StalingPolicy:
        async def decide_semantic_watch(
            self,
            request: KnowledgeSemanticWatchRequest,
        ) -> KnowledgeSemanticWatchDecision:
            current = await store.get_entry(entry.id)
            assert current is not None
            await store.append_entry_revision(
                current.model_copy(
                    update={
                        "revision": current.revision + 1,
                        "text": "Atlas release moved to Monday.",
                    }
                ),
                expected_revision=current.revision,
            )
            return KnowledgeSemanticWatchDecision(
                request_sha256=request.fingerprint,
                disposition=KnowledgeSemanticWatchDisposition.EMIT,
                policy_identity="application-release-policy",
                policy_version="1",
                code="stale-before-commit",
            )

    staling_evaluator = KnowledgeSemanticWatchEvaluator(
        store,
        _engine(store),
        config=_config(),
        policy=_StalingPolicy(),
    )
    stale_values: _EvaluationValues = {
        **values,
        "operation_id": "semantic-watch-stale",
        "observation_id": "stale-observation",
    }
    try:
        await staling_evaluator.evaluate(**stale_values)
    except KnowledgeSemanticWatchConflict as exc:
        assert exc.code == "candidate_stale"
    else:  # pragma: no cover - conformance assertion
        raise AssertionError("A stale semantic-watch candidate must not commit.")


async def assert_knowledge_semantic_watch_scope_conformance(store: KnowledgeStore) -> None:
    scope = KnowledgeAccessScope.for_namespace("semantic-watch:allowed")
    other_scope = KnowledgeAccessScope.for_namespace("semantic-watch:other")
    await store.create_entry(
        KnowledgeEntry(
            id="semantic-watch-scoped-entry",
            text="The scoped Atlas release is Friday.",
            namespace="semantic-watch:allowed",
        ),
        access_scope=scope,
    )
    evaluator = KnowledgeSemanticWatchEvaluator(
        store,
        _engine(store),
        config=_config().model_copy(update={"knowledge_namespace": "semantic-watch:allowed"}),
        policy=_Policy(),
    )
    receipt = await evaluator.evaluate(
        operation_id="semantic-watch-scoped-operation",
        observation_id="semantic-watch-scoped-observation",
        observation_source_type="deployment",
        observation_source_id="deployment-scoped",
        observation_text="Atlas release Friday",
        access_scope=scope,
    )
    assert (
        await store.load_semantic_watch_receipt(
            receipt.operation_id,
            access_scope=scope,
        )
        == receipt
    )
    assert (
        await store.load_semantic_watch_receipt(
            receipt.operation_id,
            access_scope=other_scope,
        )
        is None
    )
    try:
        await store.record_semantic_watch_outcome(
            receipt.authority,
            access_scope=other_scope,
        )
    except KnowledgeAccessDenied:
        pass
    else:  # pragma: no cover - conformance assertion
        raise AssertionError("A different access scope must not replay a watch outcome.")


__all__ = [
    "assert_knowledge_semantic_watch_conformance",
    "assert_knowledge_semantic_watch_scope_conformance",
]
