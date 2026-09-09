from __future__ import annotations

import asyncio

import pytest

from cayu.memory import admit_recall
from cayu.recall import KnowledgeRecallSource, RecallEngine, RecallSituation
from cayu.recall_relevance import (
    PHRASE_RELEVANCE_TEXT_VERSION,
    PHRASE_RELEVANCE_VERSION,
    query_concept_eligibility,
)
from cayu.retrieval import WeightedReciprocalRankFusionConfig
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    SQLiteKnowledgeStore,
)

QUERY = (
    "Explain the release rollback procedure. Return only JSON with keys action and reason. "
    "Keep the answer brief and report unknown if there is no authorized evidence."
)

DELIVERY_INSTRUCTIONS = (
    "Be concise and clear.",
    "Please be concise and clear.",
    "Answer in short simple sentences.",
    "Write short simple sentences.",
    "Keep the tone friendly calm and professional.",
    "Keep the answer short simple and clear.",
    "Make the response short clear and concise.",
    "Never make up facts.",
    "Do not invent missing details.",
)


@pytest.mark.parametrize("comma", [",", "\uff0c"])
@pytest.mark.parametrize(
    "prefix",
    [
        "Explain deployment rollback safeguards. Write short",
        "Explain deployment rollback safeguards and write short",
        "Explain deployment rollback safeguards, write short",
    ],
)
def test_delivery_context_survives_comma_fragments(comma, prefix):
    query = prefix + comma + " clear complete sentences" + comma + " friendly professional tone."
    for unrelated in (
        "Clear complete sentences about deployment picnics.",
        "Friendly professional tone at deployment picnics.",
    ):
        assert (
            query_concept_eligibility(query, unrelated, version=PHRASE_RELEVANCE_VERSION)[0]
            == "low_relevance"
        )
    assert query_concept_eligibility(
        query,
        "Deployment rollback safeguards: restore the healthy image.",
        version=PHRASE_RELEVANCE_VERSION,
    ) == ("eligible", "query_phrase_support")


@pytest.mark.parametrize("boundary", [". ", "; ", "\n", "\u2028", ",\n", "\uff0c\u2029"])
def test_new_sentence_or_line_resets_delivery_context(boundary):
    query = (
        "Write short, clear complete sentences"
        + boundary
        + "Explain deployment rollback safeguards and credential rotation."
    )
    assert query_concept_eligibility(
        query,
        "Deployment rollback safeguards: restore the healthy image.",
        version=PHRASE_RELEVANCE_VERSION,
    ) == ("eligible", "query_phrase_support")


@pytest.mark.parametrize("joiner", [". ", ", ", ", and ", " and ", " then ", " please "])
@pytest.mark.parametrize(
    "instruction", ["be concise and clear", "answer in short simple sentences"]
)
def test_appended_delivery_commands_do_not_supply_topic_phrases(joiner, instruction):
    query = "Explain deployment rollback safeguards and credential rotation" + joiner + instruction
    unrelated = instruction + " when discussing deployment picnics."
    assert (
        query_concept_eligibility(query, unrelated, version=PHRASE_RELEVANCE_VERSION)[0]
        == "low_relevance"
    )
    assert query_concept_eligibility(
        query,
        "Deployment rollback safeguards: restore the healthy image.",
        version=PHRASE_RELEVANCE_VERSION,
    ) == ("eligible", "query_phrase_support")


@pytest.mark.parametrize("instruction", DELIVERY_INSTRUCTIONS)
def test_delivery_clause_cannot_manufacture_topic_support(instruction):
    query = (
        "Explain deployment rollback safeguards, credential rotation and request deadlines. "
        + instruction
    )
    unrelated = instruction + " The deployment picnic is tomorrow."
    for version in ("cayu.query_concepts.v2", PHRASE_RELEVANCE_VERSION):
        assert query_concept_eligibility(query, unrelated, version=version)[0] == "low_relevance"
    assert query_concept_eligibility(
        query,
        "Deployment rollback safeguards: restore the healthy image.",
        version=PHRASE_RELEVANCE_VERSION,
    ) == ("eligible", "query_phrase_support")


@pytest.mark.parametrize(
    "boundary",
    [
        "\n",
        "\r",
        "\r\n",
        "\v",
        "\f",
        "\x1c",
        "\x1d",
        "\x1e",
        "\x85",
        "\u2028",
        "\u2029",
        "\u3002",
        "\uff01",
        "\uff1f",
        "\uff1b",
    ],
)
@pytest.mark.parametrize("location", ["query", "body", "title"])
def test_unicode_boundaries_do_not_join_phrases(boundary, location):
    split = "release rollback" + boundary + "procedure"
    query = (
        QUERY if location != "query" else split + ". Return only JSON with keys action and reason."
    )
    text = (
        split
        if location == "body"
        else "release rollback procedure"
        if location == "query"
        else "unrelated picnic"
    )
    title = split if location == "title" else None
    assert (
        query_concept_eligibility(query, text, title=title, version=PHRASE_RELEVANCE_VERSION)[0]
        == "low_relevance"
    )


@pytest.mark.parametrize(
    "query",
    [
        "Return the approved maintenance window. Answer in short simple sentences.",
        "Return the approved maintenance window in JSON with a brief explanation of the source.",
        "Please provide the approved maintenance window in short simple sentences.",
    ],
)
def test_returning_a_fact_is_not_a_delivery_clause(query):
    assert query_concept_eligibility(
        query, "Approved maintenance window: midnight.", version=PHRASE_RELEVANCE_VERSION
    ) == ("eligible", "query_phrase_support")


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "query,text",
    [
        (
            "Explain deployment rollback safeguards. Be concise and clear.",
            "Be concise and clear when discussing deployment picnics.",
        ),
        (
            "Explain deployment rollback safeguards. Write short, clear complete sentences.",
            "Clear complete sentences about deployment picnics.",
        ),
        (QUERY, "release rollback\u2028procedure for a picnic"),
    ],
)
def test_reproduced_false_positives_are_not_injected(tmp_path, backend, query, text):
    from test_memory_admission import _policy

    async def check():
        scope = KnowledgeAccessScope.for_namespace("default")
        store = (
            InMemoryKnowledgeStore(access_scope=scope)
            if backend == "memory"
            else SQLiteKnowledgeStore(tmp_path / "negative.sqlite", access_scope=scope)
        )
        try:
            await store.create_entry(KnowledgeEntry(id="unrelated", text=text))
            policy = _policy(
                relevance_policy=PHRASE_RELEVANCE_VERSION,
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
            result = await engine.recall(RecallSituation(query=query, knowledge_access_scope=scope))
            assert len(result.candidates) == 1
            contribution = admit_recall(result, policy)
            assert contribution.focus is None and contribution.offer is None
            assert contribution.diagnostics.candidate_decisions[0].outcome == "low_relevance"
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(check())


@pytest.mark.parametrize(
    "text,title",
    [
        ("Release rollback procedure: restore the healthy image.", None),
        ("Restore the healthy image.", "Release rollback procedure"),
    ],
)
def test_phrase_support_survives_delivery_instructions_without_changing_v2(text, title):
    assert (
        query_concept_eligibility(QUERY, text, title=title, version="cayu.query_concepts.v2")[0]
        == "low_relevance"
    )
    assert query_concept_eligibility(
        QUERY, text, title=title, version=PHRASE_RELEVANCE_VERSION
    ) == ("eligible", "query_phrase_support")


@pytest.mark.parametrize(
    "query,text",
    [
        (QUERY, "The release picnic takes place at noon."),
        (QUERY, "Release rollback is a phrase. Our picnic procedure is simple."),
        (QUERY, "Return only JSON with keys action and reason."),
        (QUERY, "action reason keys output json return"),
        (QUERY, "release release rollback rollback"),
        (QUERY, "release rollback. Procedure for a picnic."),
        (QUERY, "release rollback\nprocedure unrelated picnic"),
        ("alpha. beta gamma extra unrelated question words", "alpha beta gamma"),
        ("alpha JSON beta gamma extra unrelated question words", "alpha beta gamma"),
        ("alpha beta gamma extra unrelated question words", "alpha JSON beta gamma"),
        (
            "src/cache.py rollback procedure and many unrelated question words",
            "src/caches.py rollback procedure",
        ),
        (
            "cache-42 rollback procedure and many unrelated question words",
            "caches-42 rollback procedure",
        ),
        ("cache caches caching many unrelated question words", "cache caches caching"),
    ],
)
def test_partial_generic_disconnected_or_repeated_support_does_not_admit(query, text):
    assert (
        query_concept_eligibility(query, text, version=PHRASE_RELEVANCE_VERSION)[0]
        == "low_relevance"
    )


def test_phrase_cannot_join_title_to_body():
    assert (
        query_concept_eligibility(
            QUERY, "procedure", title="release rollback", version=PHRASE_RELEVANCE_VERSION
        )[0]
        == "low_relevance"
    )


def test_casefold_expansion_preserves_sentence_boundaries():
    # Casefold expands capital dotted I; match offsets must use the folded text.
    query = "İ irrelevant. alpha beta. gamma extra unrelated question words"
    assert (
        query_concept_eligibility(query, "alpha beta gamma", version=PHRASE_RELEVANCE_VERSION)[0]
        == "low_relevance"
    )


def test_phrase_policy_identity_is_distinct_and_round_trips():
    from test_memory_admission import _policy

    policy = _policy(relevance_policy=PHRASE_RELEVANCE_VERSION)
    assert policy.relevance_text_version == PHRASE_RELEVANCE_TEXT_VERSION
    assert type(policy).model_validate_json(policy.model_dump_json()) == policy
    assert policy.fingerprint() != _policy(relevance_policy="cayu.query_concepts.v2").fingerprint()


def test_runtime_delivers_phrase_supported_record_and_persists_its_reason():
    from test_automatic_recall_context import (
        _admission,
        _CountingKnowledgeStore,
        _CountingSessionStore,
        _policy,
        _provider_manifest,
        _RecordingCountScriptedProvider,
    )

    from cayu import AgentSpec, CayuApp, Environment, EnvironmentSpec, Message, RunRequest
    from cayu.memory_evidence import RecallEvidenceQuery
    from cayu.providers import ModelStreamEvent
    from cayu.runtime.request_footprints import RequestFootprintConfig

    async def check():
        sessions = _CountingSessionStore()
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = _CountingKnowledgeStore(access_scope=scope)
        text = "Release rollback procedure: restore the healthy image."
        await knowledge.create_entry(
            KnowledgeEntry(id="rollback", namespace="project:cayu", text=text)
        )
        provider = _RecordingCountScriptedProvider(
            [
                [
                    ModelStreamEvent.text_delta("Restore the healthy image."),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        app = CayuApp(
            session_store=sessions,
            enable_logging=False,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="phrase-test",
                fingerprint_key="phrase-test-key-material-32-bytes-minimum",
            ),
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                knowledge_store=knowledge,
                knowledge_access_scope=scope,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            context_policy=_policy(
                admission_policy=_admission().model_copy(
                    update={"relevance_policy": PHRASE_RELEVANCE_VERSION}
                )
            ),
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="phrase-runtime",
                    messages=[Message.text("user", QUERY)],
                )
            )
        ]
        assert str(events[-1].type) == "session.completed"
        assert text in _provider_manifest(provider.requests[0].messages)
        receipts = await sessions.list_recall_receipts(
            RecallEvidenceQuery(session_id="phrase-runtime")
        )
        assert len(receipts.items) == 1
        assert receipts.items[0].candidate_decisions[0].reason == "query_phrase_support"
        assert (
            type(receipts.items[0]).model_validate_json(receipts.items[0].model_dump_json())
            == receipts.items[0]
        )

    asyncio.run(check())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recall_keeps_current_scoped_phrase_and_rejects_irrelevant_candidates(tmp_path, backend):
    from test_memory_admission import _policy

    async def check():
        store = (
            InMemoryKnowledgeStore()
            if backend == "memory"
            else SQLiteKnowledgeStore(tmp_path / "knowledge.sqlite")
        )
        privileged = KnowledgeAccessScope.privileged()
        scope = KnowledgeAccessScope.for_namespace("operations")
        old = KnowledgeEntry(
            id="procedure",
            namespace="operations",
            text="Release rollback procedure: use the retired image.",
        )
        current = old.model_copy(
            update={"revision": 2, "text": "Release rollback procedure: restore the healthy image."}
        )
        try:
            await store.create_entry(old, access_scope=privileged)
            await store.append_entry_revision(current, expected_revision=1, access_scope=privileged)
            await store.create_entry(
                KnowledgeEntry(id="foreign", namespace="other", text=current.text),
                access_scope=privileged,
            )
            await store.create_entry(
                KnowledgeEntry(
                    id="picnic", namespace="operations", text="The release picnic has a procedure."
                ),
                access_scope=privileged,
            )
            policy = _policy(
                relevance_policy=PHRASE_RELEVANCE_VERSION,
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
                    query=QUERY, knowledge_access_scope=scope, knowledge_namespace="operations"
                )
            )
            contribution = admit_recall(result, policy)
            assert contribution.focus is not None
            locators = [item.candidate.record.locator for item in contribution.focus.items]
            assert len(locators) == 1
            assert locators[0]["entry_id"] == "procedure"
            assert int(locators[0]["entry_revision"]) == 2
            assert any(
                item.reason == "query_phrase_support"
                for item in contribution.diagnostics.candidate_decisions
            )
            assert (
                admit_recall(result, _policy(relevance_policy="cayu.query_concepts.v2")).focus
                is None
            )
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(check())
