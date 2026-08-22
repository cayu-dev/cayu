"""Credential-free bounded recall and calibrated admission across memory sources.

Recall produces retrieval evidence; admission separately chooses bounded memory
focus, reference-only offers, and silent candidates. Runtime applications use
``AutomaticRecallContextPolicy`` to freeze this decision for one interaction.
"""

from __future__ import annotations

import asyncio

from cayu import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    TRANSCRIPT_LEXICAL_CHANNEL,
    WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeRecallSource,
    RecallEngine,
    RecallSituation,
    TranscriptRecallSource,
    WeightedReciprocalRankFusionConfig,
)
from cayu.core.messages import Message, MessageRole
from cayu.memory import AutomaticRecallContributor, AutomaticRecallPolicy
from cayu.runtime import InMemorySessionStore, RunRequest, SessionIdentity


async def main() -> None:
    scope = KnowledgeAccessScope.for_namespace("acme")
    knowledge = InMemoryKnowledgeStore()
    await knowledge.create_entry(
        KnowledgeEntry(
            id="deployment-port",
            namespace="acme",
            text="The deployment gateway listens on port 8443.",
        ),
        access_scope=scope,
    )

    sessions = InMemorySessionStore()
    session = await sessions.create(
        RunRequest(agent_name="operator", session_id="deployment-session", messages=[]),
        identity=SessionIdentity(provider_name="hermetic", model="none"),
    )
    await sessions.append_transcript_messages(
        session.id,
        [Message.text(MessageRole.USER, "We agreed to use deployment port 8443.")],
        interaction_id="deployment-decision",
    )

    engine = RecallEngine(
        (
            KnowledgeRecallSource(knowledge, candidate_limit=5),
            TranscriptRecallSource(sessions, candidate_limit=5),
        ),
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version="example-v1",
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
                TRANSCRIPT_LEXICAL_CHANNEL: 1.0,
            },
            max_candidates_per_channel=5,
            fused_head_limit=5,
        ),
    )
    contributor = AutomaticRecallContributor(
        engine,
        AutomaticRecallPolicy(
            calibration_version="example-calibration-v1",
            fusion_strategy_version=WEIGHTED_RECIPROCAL_RANK_FUSION_VERSION,
            fusion_configuration_version="example-v1",
            minimum_inject_score=0.0162,
            minimum_offer_score=0.016,
        ),
    )
    contribution = await contributor.contribute(
        RecallSituation(
            query="Which deployment port should we use?",
            knowledge_access_scope=scope,
            knowledge_namespace="acme",
            transcript_session_ids=(session.id,),
        )
    )

    print("memory focus:")
    for item in () if contribution.focus is None else contribution.focus.items:
        record = item.candidate.record
        print(" ", record.identity.record_type, record.text, dict(record.locator))
    print("reference offers:")
    for item in () if contribution.offer is None else contribution.offer.items:
        print(" ", item.identity.record_type, dict(item.locator), item.reason)
    print("silent candidates:", contribution.diagnostics.silent_count)
    print("source coverage:", [(item.source, item.status) for item in contribution.sources])


if __name__ == "__main__":
    asyncio.run(main())
