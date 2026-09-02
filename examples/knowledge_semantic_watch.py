"""Evaluate one observation against exact knowledge without a provider or worker."""

from __future__ import annotations

import asyncio
import json
from typing import TypedDict

from cayu import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeRecallSource,
    KnowledgeSemanticWatchConfig,
    KnowledgeSemanticWatchDecision,
    KnowledgeSemanticWatchDisposition,
    KnowledgeSemanticWatchEvaluator,
    RecallEngine,
    WeightedReciprocalRankFusionConfig,
)


class ReleasePolicy:
    """Example application authority; retrieval itself never emits the signal."""

    async def decide_semantic_watch(self, request):
        matched = bool(request.evidence.candidates)
        return KnowledgeSemanticWatchDecision(
            request_sha256=request.fingerprint,
            disposition=(
                KnowledgeSemanticWatchDisposition.EMIT
                if matched
                else KnowledgeSemanticWatchDisposition.IGNORE
            ),
            policy_identity="example.release-policy",
            policy_version="1",
            code="release_window_match" if matched else "no_release_window_match",
        )


class _EvaluationValues(TypedDict):
    operation_id: str
    observation_id: str
    observation_source_type: str
    observation_source_id: str
    observation_text: str


async def main() -> None:
    scope = KnowledgeAccessScope.for_namespace("example:semantic-watch")
    store = InMemoryKnowledgeStore(access_scope=scope)
    await store.create_entry(
        KnowledgeEntry(
            id="release-window",
            text="Atlas deployment Friday",
            namespace="example:semantic-watch",
        )
    )
    engine = RecallEngine(
        (KnowledgeRecallSource(store),),
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version="example.semantic-watch-recall.v1",
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
        config=KnowledgeSemanticWatchConfig(
            watch_identity="example.release-watch",
            watch_version="1",
            recall_profile_identity="example.release-recall",
            recall_profile_version="1",
            policy_identity="example.release-policy",
            policy_version="1",
            knowledge_namespace="example:semantic-watch",
        ),
        policy=ReleasePolicy(),
    )
    values: _EvaluationValues = {
        "operation_id": "release-observation-42",
        "observation_id": "deployment-42",
        "observation_source_type": "deployment",
        "observation_source_id": "deployment-42",
        "observation_text": "Atlas deployment Friday",
    }
    receipt = await evaluator.evaluate(**values)
    replay = await evaluator.evaluate(**values)
    print(
        json.dumps(
            {
                "disposition": receipt.authority.decision.disposition.value,
                "exact_revisions": [
                    {
                        "entry_id": candidate.reference.entry_id,
                        "revision": candidate.reference.revision,
                    }
                    for candidate in receipt.authority.evidence.candidates
                ],
                "provider_calls": 0,
                "replayed": replay.replayed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
