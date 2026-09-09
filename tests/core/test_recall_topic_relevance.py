from __future__ import annotations

import pytest

from cayu.recall_relevance import (
    TOPIC_RELEVANCE_TEXT_VERSION,
    TOPIC_RELEVANCE_VERSION,
    query_concept_eligibility,
)


@pytest.mark.parametrize(
    "phrase",
    [
        "artifact signing key alias",
        "database replication status indicator",
        "customer profile field mapping",
        "compiler string value representation",
    ],
)
@pytest.mark.parametrize("location", ["text", "title"])
def test_factual_schema_terms_survive_output_instructions(phrase, location):
    query = f"What is the {phrase}? Return only JSON with keys value and status. Do not guess."
    text = f"{phrase}: configured."
    kwargs = {"text": text} if location == "text" else {"text": "Configured.", "title": phrase}
    assert (
        query_concept_eligibility(query, **kwargs, version="cayu.query_concepts.v3")[0]
        == "low_relevance"
    )
    assert query_concept_eligibility(query, **kwargs, version=TOPIC_RELEVANCE_VERSION) == (
        "eligible",
        "query_phrase_support",
    )


@pytest.mark.parametrize(
    "instruction,unrelated",
    [
        ("Return only JSON with keys action reason status.", "Action reason status: picnic."),
        (
            "Use status known when supported by current project records.",
            "Current project records describe a picnic.",
        ),
        (
            "Use status unknown when evidence is missing.",
            "Evidence is missing from the picnic report.",
        ),
        ("Return JSON with keys value status unknown.", "Value status unknown at the picnic."),
        ("in JSON with keys action reason status", "Action reason status: picnic."),
        ("as YAML with fields action reason status", "Action reason status: picnic."),
        (
            "Return an object with fields action reason status.",
            "Fields action reason status describe a picnic.",
        ),
        (
            "Return a JSON object with fields action reason status.",
            "Fields action reason status describe a picnic.",
        ),
        ("Include fields action reason status.", "Fields action reason status describe a picnic."),
    ],
)
def test_schema_instructions_do_not_supply_topic_support(instruction, unrelated):
    query = "Explain release rollback safeguards and credential rotation " + instruction
    assert (
        query_concept_eligibility(query, unrelated, version=TOPIC_RELEVANCE_VERSION)[0]
        == "low_relevance"
    )
    assert query_concept_eligibility(
        query,
        "Release rollback safeguards: restore the healthy image.",
        version=TOPIC_RELEVANCE_VERSION,
    ) == ("eligible", "query_phrase_support")


@pytest.mark.parametrize(
    "instruction",
    [
        "Be concise and clear.",
        "Answer in short simple sentences.",
        "Write short, clear complete sentences, friendly professional tone.",
        "Use status known when supported by current project records.",
    ],
)
@pytest.mark.parametrize("joiner", [". ", ", ", " and ", " then ", "\u2028"])
def test_topic_policy_retains_delivery_clause_guards(instruction, joiner):
    query = "Explain release rollback safeguards and credential rotation" + joiner + instruction
    for text in (instruction + " Picnic planning.", "Clear complete sentences about a picnic."):
        assert (
            query_concept_eligibility(query, text, version=TOPIC_RELEVANCE_VERSION)[0]
            == "low_relevance"
        )


@pytest.mark.parametrize("boundary", [". ", "; ", "\n", "\u2028", ",\n"])
def test_factual_topic_after_delivery_sentence_is_not_erased(boundary):
    query = (
        "Use status known when supported by current records"
        + boundary
        + "What is the artifact signing key alias?"
    )
    assert (
        query_concept_eligibility(
            query, "Artifact signing key alias: production-signer.", version=TOPIC_RELEVANCE_VERSION
        )[0]
        == "eligible"
    )


def test_topic_policy_identity_round_trips():
    from test_memory_admission import _policy

    policy = _policy(relevance_policy=TOPIC_RELEVANCE_VERSION)
    assert policy.relevance_text_version == TOPIC_RELEVANCE_TEXT_VERSION
    assert type(policy).model_validate_json(policy.model_dump_json()) == policy
    assert policy.fingerprint() != _policy(relevance_policy="cayu.query_concepts.v3").fingerprint()


@pytest.mark.parametrize(
    "question,text",
    [
        (
            "How should we handle errors in JSON schema validation pipelines?",
            "JSON schema validation pipelines reject malformed documents.",
        ),
        (
            "How should we use short lived credential rotation tokens?",
            "Credential rotation tokens expire after ten minutes.",
        ),
        (
            "Explain signing validation using JSON schema validation pipelines.",
            "JSON schema validation pipelines reject malformed documents.",
        ),
        (
            "How should we use short-lived credential rotation tokens?",
            "Credential rotation tokens expire after ten minutes.",
        ),
    ],
)
@pytest.mark.parametrize("location", ["body", "title"])
def test_technical_topics_are_not_delivery_suffixes(question, text, location):
    query = question + " Return only JSON with keys value and status. Do not guess."
    kwargs = {"text": text} if location == "body" else {"text": "Configured.", "title": text}
    for version in ("cayu.query_concepts.v3", TOPIC_RELEVANCE_VERSION):
        assert query_concept_eligibility(query, **kwargs, version=version)[0] == "eligible"


@pytest.mark.parametrize(
    "question",
    [
        "How should we return an object with fields describing database replication protocols?",
        "Which components include fields required for database replication protocols?",
        "Explain how to return an object with fields for database replication protocols.",
    ],
)
def test_embedded_schema_questions_preserve_factual_topics(question):
    query = question + " Return only JSON with keys value and status. Do not guess."
    assert (
        query_concept_eligibility(
            query,
            "Database replication protocols control replication.",
            version=TOPIC_RELEVANCE_VERSION,
        )[0]
        == "eligible"
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "query,text,focused",
    [
        (
            "What is the artifact signing key alias? Return JSON with keys value and status. Do not guess.",
            "Artifact signing key alias: production-signer.",
            True,
        ),
        (
            "How should we handle errors in JSON schema validation pipelines? Return only JSON with keys value and status. Do not guess.",
            "JSON schema validation pipelines reject malformed documents.",
            True,
        ),
        (
            "How should we use short lived credential rotation tokens? Return only JSON with keys value and status. Do not guess.",
            "Credential rotation tokens expire after ten minutes.",
            True,
        ),
        (
            "Explain deployment rollback safeguards and credential rotation. Return an object with fields action reason status.",
            "Fields action reason status describe the picnic schedule.",
            False,
        ),
        (
            "Explain deployment rollback safeguards and credential rotation. Include fields action reason status.",
            "Fields action reason status describe the picnic schedule.",
            False,
        ),
    ],
)
def test_factual_schema_terms_reach_real_backend_admission(tmp_path, backend, query, text, focused):
    import asyncio

    from test_memory_admission import _policy

    from cayu.memory import admit_recall
    from cayu.recall import KnowledgeRecallSource, RecallEngine, RecallSituation
    from cayu.retrieval import WeightedReciprocalRankFusionConfig
    from cayu.storage import (
        InMemoryKnowledgeStore,
        KnowledgeAccessScope,
        KnowledgeEntry,
        SQLiteKnowledgeStore,
    )

    async def check():
        scope = KnowledgeAccessScope.for_namespace("default")
        store = (
            InMemoryKnowledgeStore(access_scope=scope)
            if backend == "memory"
            else SQLiteKnowledgeStore(tmp_path / "topic.sqlite", access_scope=scope)
        )
        try:
            await store.create_entry(KnowledgeEntry(id="signing", text=text))
            policy = _policy(
                relevance_policy=TOPIC_RELEVANCE_VERSION,
                minimum_inject_score=0.01,
                minimum_offer_score=0.005,
            )
            engine = RecallEngine(
                (KnowledgeRecallSource(store),),
                fusion_config=WeightedReciprocalRankFusionConfig(
                    configuration_version=policy.fusion_configuration_version,
                    channel_weights={"knowledge.lexical": 1.0, "knowledge.semantic": 1.0},
                ),
            )
            result = await engine.recall(
                RecallSituation(
                    query=query,
                    knowledge_access_scope=scope,
                )
            )
            contribution = admit_recall(result, policy)
            assert len(result.candidates) == 1
            assert (contribution.focus is not None) is focused
            assert contribution.diagnostics.candidate_decisions[0].reason == (
                "query_phrase_support" if focused else "weak_query_support"
            )
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(check())
