"""Finishing must preserve recall checkpoint coordinates when withdrawing native state."""

from __future__ import annotations

import asyncio
import json

import pytest
from tests.core.test_automatic_recall_context import _policy
from tests.core.test_run_limit_finish import answer, spec
from tests.core.test_session_model_switch import _NamedProvider

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.budgets.run_limits import RunLimits
from cayu.context import MessageWindowContextPolicy
from cayu.context.footprints import RequestFootprintConfig
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.messages import ProviderStatePart
from cayu.providers import OpenAIProvider
from cayu.providers.base import ModelProviderError, ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.sessions import InMemorySessionStore, ModelFailoverPolicy, ModelTarget
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope, KnowledgeEntry
from cayu.storage.sqlite import SQLiteSessionStore


class _ReasoningThenAnswerTransport:
    """Use the built-in OpenAI adapter, substituting only HTTP response frames."""

    def __init__(self, *, opaque_only):
        self.opaque_only = opaque_only
        self.calls = []

    async def stream_response_events(self, *, payload, **kwargs):
        self.calls.append(payload)
        if len(self.calls) == 1:
            output = [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "status": "completed",
                    "summary": [],
                    "encrypted_content": "opaque-test-reasoning",
                }
            ]
            if not self.opaque_only:
                yield {"type": "response.output_text.delta", "delta": "Still working"}
                output.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Still working", "annotations": []}
                        ],
                    }
                )
        else:
            yield {"type": "response.output_text.delta", "delta": '{"answer":42}'}
            output = [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": '{"answer":42}', "annotations": []}
                    ],
                }
            ]
        yield {
            "type": "response.completed",
            "response": {
                "id": f"resp_{len(self.calls)}",
                "model": "gpt-5.6",
                "status": "completed",
                "output": output,
                "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            },
        }


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("opaque_only", [False, True], ids=["text-control", "opaque-only"])
def test_finishing_preserves_recall_anchor_after_provider_only_history(
    tmp_path, backend, opaque_only
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "finish-recall.sqlite")
        )
        knowledge = InMemoryKnowledgeStore(
            access_scope=KnowledgeAccessScope.for_namespace("project:cayu")
        )
        await knowledge.create_entry(
            KnowledgeEntry(
                id="example-widget",
                namespace="project:cayu",
                text="The example widget is blue.",
            )
        )
        transport = _ReasoningThenAnswerTransport(opaque_only=opaque_only)
        provider = OpenAIProvider(api_key="test-key", transport=transport)
        app = CayuApp(
            session_store=store,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge), default=True
        )
        app.register_agent(AgentSpec(name="assistant", model="gpt-5.6"), context_policy=_policy())
        session_id = "finish-recall-anchor"
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "What colour is the example widget?")],
                        structured_output=spec("native"),
                        max_steps=2,
                        limits=RunLimits(on_limit="finish"),
                    )
                )
            ]
            transcript = await store.load_transcript(session_id)
            assert transcript[2].role == "user"  # Runtime-authored structured-output repair.
            if opaque_only:
                assert all(part.type == "provider_state" for part in transcript[1].content)
            assert len(transport.calls) == 2, (
                "Withdrawing native state must not invalidate the runtime-authored repair "
                f"anchor: terminal={events[-1].type}, payload={events[-1].payload}"
            )
            assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
            assert events[-1].type is EventType.SESSION_INTERRUPTED, events[-1].payload
        finally:
            await app.aclose()
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


def test_finishing_context_policy_counts_only_projected_history():
    class RecordingWindow(MessageWindowContextPolicy):
        def __init__(self):
            super().__init__(max_messages=2)
            self.seen = []

        async def build(self, request):
            self.seen.append(list(request.messages))
            return await super().build(request)

    async def scenario():
        transport = _ReasoningThenAnswerTransport(opaque_only=True)
        policy = RecordingWindow()
        app = CayuApp(enable_logging=False)
        app.register_provider(OpenAIProvider(api_key="test-key", transport=transport), default=True)
        app.register_agent(AgentSpec(name="assistant", model="gpt-5.6"), context_policy=policy)
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="finish-projected-window",
                        messages=[Message.text("user", "What colour is the example widget?")],
                        structured_output=spec("native"),
                        max_steps=2,
                        limits=RunLimits(on_limit="finish"),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_INTERRUPTED, events[-1].payload
            # The opaque-only assistant row is withdrawn before the window counts
            # messages, so the question still fits beside the repair message.
            assert [message.role for message in policy.seen[-1]] == ["user", "user"]
            assert not any(
                isinstance(part, ProviderStatePart)
                for message in policy.seen[-1]
                for part in message.content
            )
            assert "What colour is the example widget?" in json.dumps(transport.calls[-1]["input"])
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_finishing_fallback_composes_recall_anchor_remaps(tmp_path, backend):
    class Source(_NamedProvider):
        async def stream(self, request):
            if self.requests:
                self.requests.append(request)
                raise ModelProviderError(
                    "unavailable", provider=self.name, retryable=True, status_code=503
                )
            async for event in super().stream(request):
                yield event

    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "finish-fallback.sqlite")
        )
        knowledge = InMemoryKnowledgeStore(
            access_scope=KnowledgeAccessScope.for_namespace("project:cayu")
        )
        await knowledge.create_entry(
            KnowledgeEntry(
                id="example-widget",
                namespace="project:cayu",
                text="The example widget is blue.",
            )
        )
        source = Source(
            "source",
            [
                [
                    ModelStreamEvent.completed(
                        {"provider_state": [{"provider": "source", "state": {"id": "response-1"}}]}
                    )
                ]
            ],
        )
        backup = _NamedProvider("backup", [answer("tool")])
        app = CayuApp(
            session_store=store,
            request_footprint=RequestFootprintConfig(
                fingerprint_key_id="test-memory-key",
                fingerprint_key="automatic-recall-test-key-material",
            ),
            enable_logging=False,
        )
        app.register_provider(source, default=True)
        app.register_provider(backup)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="source-model"), context_policy=_policy()
        )
        session_id = "finish-fallback-recall"
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "What colour is the example widget?")],
                        structured_output=spec("tool"),
                        max_steps=2,
                        limits=RunLimits(on_limit="finish"),
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="backup-model"),),
                            max_total_attempts=2,
                        ),
                    )
                )
            ]
            transcript = await store.load_transcript(session_id)
            assert transcript[2].role == "user"  # Runtime-authored structured-output repair.
            assert all(part.type == "provider_state" for part in transcript[1].content)
            assert len(source.requests) == 2 and len(backup.requests) == 1, events[-1].payload
            assert any(event.type is EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
            assert events[-1].type is EventType.SESSION_INTERRUPTED, events[-1].payload
        finally:
            await app.aclose()
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
