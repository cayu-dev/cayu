"""Credential-free bounded recall across canonical knowledge and transcripts.

Recall returns retrieval evidence. It does not inject candidates into a model
request; an application-owned context policy must make that separate decision.
"""

from __future__ import annotations

import asyncio

from cayu import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    TRANSCRIPT_LEXICAL_CHANNEL,
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
    result = await engine.recall(
        RecallSituation(
            query="Which deployment port should we use?",
            knowledge_access_scope=scope,
            knowledge_namespace="acme",
            transcript_session_ids=(session.id,),
        )
    )

    for candidate in result.candidates:
        record = candidate.record
        print(record.identity.record_type, record.text, dict(record.locator))
    print("source coverage:", [(item.source, item.status) for item in result.sources])


if __name__ == "__main__":
    asyncio.run(main())
