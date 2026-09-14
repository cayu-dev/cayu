"""Native recovery of canonical knowledge first delivered by a custom source."""

from __future__ import annotations

import asyncio
from hashlib import sha256

import pytest
from test_automatic_recall_context import _admission, _CountingSessionStore, _fusion

from cayu import (
    AgentSpec,
    AutomaticRecallContextPolicy,
    AutomaticRecallSourceConfig,
    AutomaticRecallSourceDescriptor,
    AutomaticRecallSourceRegistration,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    MemoryDeltaPolicy,
    Message,
    ModelStreamEvent,
    RequestFootprintConfig,
    RunRequest,
    ScriptedModelProvider,
)
from cayu.context.base import (
    CheckpointCompactionContextPolicy,
    CompactionResult,
    ContextPolicy,
    TranscriptDigestCompactor,
)
from cayu.memory.evidence import RecallEvidenceQuery
from cayu.memory.recall import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    KnowledgeRecallSource,
    RecallSource,
    RecallSourceResult,
)
from cayu.messages import TextPart, copy_message
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope, KnowledgeEntry
from cayu.tools.base import Tool, ToolResult, ToolSpec


class _CanonicalSource(RecallSource):
    name = "curated"
    channel_names = ("curated.lexical",)

    def __init__(self, store, *, tampered=False):
        super().__init__(required=True, candidate_limit=2)
        self.store = store
        self.tampered = tampered

    async def retrieve(self, situation):
        # A custom catalogue routes an ambiguous phrase to a canonical record.
        result = await KnowledgeRecallSource(self.store, candidate_limit=2).retrieve(
            situation.model_copy(update={"query": "Atlas release evidence"})
        )
        records = result.records
        channels = result.channels[:1]
        if self.tampered:
            altered = "Unverified Atlas release evidence says Sunday."
            digest = sha256(altered.encode()).hexdigest()
            records = tuple(
                r.model_copy(update={"text": altered, "content_hash": digest}) for r in records
            )
            channels = tuple(
                c.model_copy(
                    update={
                        "hits": tuple(h.model_copy(update={"content_hash": digest}) for h in c.hits)
                    }
                )
                for c in channels
            )
        return RecallSourceResult(
            source=self.name,
            channels=tuple(
                c.model_copy(update={"channel": self.channel_names[0]}) for c in channels
            ),
            records=records,
            coverage_complete=True,
        )


class _ReplaceAnchor(ContextPolicy):
    def __init__(self, *, irrelevant=False):
        self.calls = 0
        self.irrelevant = irrelevant

    async def build(self, request):
        self.calls += 1
        if self.calls == 1:
            return [copy_message(m) for m in request.messages]
        return [
            Message.text(
                "user", "Orion billing invoice?" if self.irrelevant else "Atlas release evidence?"
            )
        ]


class _Boundary(Tool):
    spec = ToolSpec(
        name="boundary",
        description="Inspect unrelated state.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    def __init__(self, store, *, supersede=False, large_result=False):
        super().__init__()
        self.store = store
        self.supersede = supersede
        self.large_result = large_result

    async def run(self, ctx, args):
        if self.supersede:
            entry = await self.store.get_entry("atlas")
            await self.store.append_entry_revision(
                entry.model_copy(
                    update={"revision": 2, "text": "Atlas release evidence says Saturday."}
                ),
                expected_revision=1,
            )
        return ToolResult(
            content="Inspection finished." + (" inventory row" * 1500 if self.large_result else "")
        )


class _CompactTranscript(TranscriptDigestCompactor):
    def __init__(self):
        super().__init__()
        self.inputs = []

    async def compact(self, request):
        self.inputs.append(request)
        return CompactionResult(
            summary="Atlas release evidence? An unrelated inventory inspection finished.",
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(
                sha256(request.existing_summary.encode()).hexdigest()
                if request.existing_summary is not None
                else None
            ),
        )


async def _run(
    *,
    enabled=True,
    tampered=False,
    supersede=False,
    irrelevant=False,
    real_compaction=False,
    reanchor_relevance="cayu.query_concepts.v5",
):
    sessions = _CountingSessionStore()
    scope = KnowledgeAccessScope.for_namespace("project:cayu")
    store = InMemoryKnowledgeStore(access_scope=scope)
    await store.create_entry(
        KnowledgeEntry(
            id="atlas",
            namespace="project:cayu",
            title="Atlas release evidence",
            text="Atlas release evidence says Friday.",
        )
    )
    constructions = []

    async def factory(context):
        constructions.append(context)
        return _CanonicalSource(context.knowledge_store, tampered=tampered)

    compactor = _CompactTranscript()
    base = (
        CheckpointCompactionContextPolicy(
            compactor=compactor,
            compact_after_estimated_context_tokens=2000,
            max_recent_context_tokens=600,
        )
        if real_compaction
        else _ReplaceAnchor(irrelevant=irrelevant)
    )
    policy = AutomaticRecallContextPolicy(
        base,
        admission_policy=_admission(),
        fusion_config=_fusion(
            KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL, "curated.lexical"
        ),
        sources=AutomaticRecallSourceConfig(
            include_transcript=False, knowledge_namespace="project:cayu"
        ),
        custom_sources=(
            AutomaticRecallSourceRegistration(
                descriptor=AutomaticRecallSourceDescriptor(
                    name="curated",
                    channel_names=("curated.lexical",),
                    candidate_limit=2,
                    configuration_version="curated-v1",
                ),
                factory=factory,
            ),
        ),
        delta_policy=MemoryDeltaPolicy(
            refresh_on_knowledge_change=False,
            reanchor_on_projection_loss=True,
            reanchor_relevance_policy=reanchor_relevance,
        )
        if enabled
        else None,
    )
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="inspect", name="boundary", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Finished."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=sessions,
        request_footprint=RequestFootprintConfig(
            fingerprint_key_id="test-recovery",
            fingerprint_key="custom-source-recovery-test-key-material",
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), knowledge_store=store), default=True
    )
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        context_policy=policy,
        tools=[_Boundary(store, supersede=supersede, large_result=real_compaction)],
    )
    messages = [Message.text("user", "Launch schedule?")]
    if real_compaction:
        # Compaction retains the session's first user request; the vulnerable
        # recall anchor belongs to a later customer message in that session.
        messages = [
            Message.text("user", "Inspect the inventory."),
            Message.text("assistant", "Inventory located."),
            *messages,
        ]
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="custom-recovery",
                messages=messages,
            )
        )
    ]
    assert events[-1].type == EventType.SESSION_COMPLETED
    assert len(provider.requests) == 2
    if real_compaction:
        assert compactor.inputs
        assert any(e.type == EventType.CONTEXT_COMPACTION_COMPLETED for e in events)
        assert all(
            "Friday" not in str(r.messages) and "<cayu_automatic_memory" not in str(r.messages)
            for r in compactor.inputs
        )
    first = [
        p.text
        for m in provider.requests[0].messages
        for p in m.content
        if type(p) is TextPart and p.text.startswith("<cayu_automatic_memory")
    ]
    assert len(first) == 1
    assert ("Sunday" if tampered else "Friday") in first[0]
    assert len(constructions) == 1  # Recovery must never rerun the custom factory.
    second = [
        p.text
        for m in provider.requests[1].messages
        for p in m.content
        if type(p) is TextPart and p.text.startswith("<cayu_memory_delta")
    ]
    checkpoint = await sessions.load_checkpoint("custom-recovery")
    if enabled:
        assert checkpoint["automatic_recall"]["delta_state"]["refresh_outcomes"] == []
    return second, checkpoint, sessions


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("real_compaction", [False, True])
def test_custom_canonical_knowledge_recovery_after_anchor_loss(enabled, real_compaction):
    async def run():
        second, checkpoint, sessions = await _run(enabled=enabled, real_compaction=real_compaction)
        assert bool(second) is enabled
        if enabled:
            assert "Friday" in second[0] and "reanchored_current_revision" in second[0]
            exposures = (
                await sessions.list_context_exposures(
                    RecallEvidenceQuery(session_id="custom-recovery")
                )
            ).items
            assert len(exposures) == 2 and all(e.provider_exposure_proven for e in exposures)
            assert (
                checkpoint["automatic_recall"]["delta_state"]["reanchor_refresh_outcomes"][-1][
                    "disposition"
                ]
                == "delta_appended"
            )

    asyncio.run(run())


@pytest.mark.parametrize("value", [0, 1, None, "false"])
def test_change_refresh_switch_requires_a_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        MemoryDeltaPolicy(refresh_on_knowledge_change=value)


def test_default_relevance_remains_conservative_for_a_broad_compaction_summary():
    async def run():
        second, checkpoint, _ = await _run(
            real_compaction=True, reanchor_relevance="cayu.query_concepts.v2"
        )
        assert second == []
        assert (
            checkpoint["automatic_recall"]["delta_state"]["reanchor_refresh_outcomes"][-1][
                "disposition"
            ]
            == "no_current_relevant_item"
        )

    asyncio.run(run())


def test_recovery_relevance_selection_changes_configuration_and_not_base_admission():
    from test_automatic_recall_sources import _policy, _registration

    default = MemoryDeltaPolicy(refresh_on_knowledge_change=False, reanchor_on_projection_loss=True)
    first = _policy(_registration(), delta_policy=default)
    second = _policy(
        _registration(),
        delta_policy=default.model_copy(
            update={"reanchor_relevance_policy": "cayu.query_concepts.v5"}
        ),
    )
    assert first.admission_policy == second.admission_policy
    assert first.configuration_fingerprint() != second.configuration_fingerprint()
    assert first._reanchor_admission_policy().relevance_policy == "cayu.query_concepts.v2"
    assert second._reanchor_admission_policy().relevance_policy == "cayu.query_concepts.v5"
    assert second._reanchor_admission_policy().relevance_text_version is not None


def test_recovery_only_policy_is_explicit_versioned_and_requires_a_trigger():
    with pytest.raises(ValueError, match="At least one"):
        MemoryDeltaPolicy(refresh_on_knowledge_change=False)
    default = MemoryDeltaPolicy(reanchor_on_projection_loss=True)
    recovery = default.model_copy(update={"refresh_on_knowledge_change": False})
    assert recovery.policy_version == "cayu.memory_delta_policy.v3"
    assert default.fingerprint() != recovery.fingerprint()
    assert MemoryDeltaPolicy.model_validate(recovery.model_dump()) == recovery


@pytest.mark.parametrize("value", ["rank_only.v1", "cayu.query_concepts.v1", "unknown", None])
def test_recovery_cannot_disable_independent_relevance(value):
    with pytest.raises(ValueError):
        MemoryDeltaPolicy(reanchor_relevance_policy=value)


def test_recovery_never_follows_an_opaque_custom_locator(monkeypatch):
    original = _CanonicalSource.retrieve

    async def opaque(self, situation):
        result = await original(self, situation)
        records = tuple(
            r.model_copy(
                update={
                    "identity": r.identity.model_copy(update={"record_type": "external"}),
                    "representation": "external_text",
                    "locator": {"uri": "https://example.invalid/private-record"},
                }
            )
            for r in result.records
        )
        by_id = {r.identity.record_id: r for r in records}
        return result.model_copy(
            update={
                "records": records,
                "channels": tuple(
                    c.model_copy(
                        update={
                            "hits": tuple(
                                h.model_copy(
                                    update={
                                        "identity": by_id[h.identity.record_id].identity,
                                        "representation": "external_text",
                                    }
                                )
                                for h in c.hits
                            )
                        }
                    )
                    for c in result.channels
                ),
            }
        )

    monkeypatch.setattr(_CanonicalSource, "retrieve", opaque)

    async def run():
        second, checkpoint, _ = await _run()
        assert second == []
        assert (
            checkpoint["automatic_recall"]["delta_state"]["reanchor_refresh_outcomes"][-1][
                "disposition"
            ]
            == "no_acknowledged_exposure"
        )

    asyncio.run(run())


@pytest.mark.parametrize("case", ["tampered", "supersede", "irrelevant"])
def test_custom_recovery_does_not_restore_changed_or_unsupported_evidence(case):
    async def run():
        second, checkpoint, _ = await _run(**{case: True})
        assert second == []
        assert (
            checkpoint["automatic_recall"]["delta_state"]["reanchor_refresh_outcomes"][-1][
                "disposition"
            ]
            == "no_current_relevant_item"
        )

    asyncio.run(run())
