"""Portable model projection must not invalidate automatic-recall anchors."""

import asyncio

import pytest
from tests.core.test_automatic_recall_context import _policy
from tests.core.test_session_model_switch import _NamedProvider

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.context.footprints import RequestFootprintConfig
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore, ModelTarget
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope, KnowledgeEntry
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("opaque_first_answer", [False, True])
def test_model_switch_preserves_recall_anchor(backend, opaque_first_answer, tmp_path):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "recall-switch.sqlite")
        )
        scope = KnowledgeAccessScope.for_namespace("project:cayu")
        knowledge = InMemoryKnowledgeStore(access_scope=scope)
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-release",
                namespace="project:cayu",
                text="Atlas release is Friday.",
            )
        )
        first = [] if opaque_first_answer else [ModelStreamEvent.text_delta("Friday")]
        first.append(
            ModelStreamEvent.completed(
                {
                    "provider_state": [{"provider": "source", "state": {"id": "response-1"}}],
                }
            )
        )
        source = _NamedProvider(
            "source",
            [
                first,
                [
                    ModelStreamEvent.text_delta("Friday confirmed"),
                    ModelStreamEvent.completed(),
                ],
            ],
        )
        target = _NamedProvider(
            "target",
            [
                [
                    ModelStreamEvent.text_delta("target answer"),
                    ModelStreamEvent.completed(),
                ]
            ],
        )
        app = CayuApp(
            session_store=store,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(source, default=True)
        app.register_provider(target)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                knowledge_store=knowledge,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="source-model"), context_policy=_policy()
        )
        session_id = "recall-portable-anchor"
        try:
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "When is Atlas released?")],
                    )
                )
            ]
            assert initial[-1].type is EventType.SESSION_COMPLETED, initial[-1].payload
            second = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "Confirm the Atlas release date.")],
                    )
                )
            ]
            assert second[-1].type is EventType.SESSION_COMPLETED, second[-1].payload
            assert len(source.requests) == 2
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint["automatic_recall"]["anchor_transcript_index"] == 2
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "Confirm the release on the target model.")],
                        target=ModelTarget(provider_name="target", model="target-model"),
                    )
                )
            ]
            failures = [
                event.payload for event in resumed if event.type is EventType.SESSION_FAILED
            ]
            assert not failures, failures
            assert resumed[-1].type is EventType.SESSION_COMPLETED
            assert len(target.requests) == 1
        finally:
            await app.aclose()
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
