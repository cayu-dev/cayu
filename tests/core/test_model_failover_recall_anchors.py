"""Fallback context and durable recall anchors must use the same history."""

import asyncio

import pytest
from tests.core.test_automatic_recall_context import _policy
from tests.core.test_session_model_switch import _NamedProvider

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.context.base import (
    RuntimeManagedContextPolicy,
    _attach_context_build_termination_diagnostics,
    context_build_termination_checkpoint_error,
)
from cayu.context.footprints import RequestFootprintConfig
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.providers.base import ModelProviderError, ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.sessions import InMemorySessionStore, ModelFailoverPolicy, ModelTarget
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope, KnowledgeEntry
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["success", "repeated", "refused", "cancelled"])
def test_failover_preserves_recall_anchors(tmp_path, backend, outcome):
    cancellation = asyncio.CancelledError("test fallback compaction cancelled")

    class CancelFallbackContext(RuntimeManagedContextPolicy):
        def __init__(self):
            self.policy = _policy()

        async def build_with_checkpoint(self, request, *, checkpoint):
            result = await self.policy.build_with_checkpoint(request, checkpoint=checkpoint)
            if request.agent.model == "backup-model":
                progress_checkpoint = result.checkpoint or checkpoint
                assert progress_checkpoint["automatic_recall"]["anchor_transcript_index"] == 3
                # Use the same completed-prefix carrier as multi-pass compaction.
                _attach_context_build_termination_diagnostics(
                    cancellation,
                    compaction_telemetry=result.compaction_telemetry,
                    checkpoint=progress_checkpoint,
                    checkpoint_event_payload=result.checkpoint_event_payload,
                )
                raise cancellation
            return result

    class RefuseFallback:
        async def _prepare_model_completion_stage_with_failover(self, *args, admission, **kwargs):
            if outcome == "refused" and admission.transition == "fallback":
                raise RuntimeError("test fallback preparation refused")
            return await super()._prepare_model_completion_stage_with_failover(
                *args, admission=admission, **kwargs
            )

    class MemoryStore(RefuseFallback, InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        model_failover_stage_version = 1

    class SQLiteStore(RefuseFallback, SQLiteSessionStore):
        invocation_lifecycle_command_version = 1
        model_failover_stage_version = 1

    class Source(_NamedProvider):
        async def stream(self, request):
            if len(self.requests) == 2:
                self.requests.append(request)
                raise ModelProviderError(
                    "unavailable", provider=self.name, retryable=True, status_code=503
                )
            async for event in super().stream(request):
                yield event

    class Backup(_NamedProvider):
        async def stream(self, request):
            checkpoint = await store.load_checkpoint(session_id)
            # By provider dispatch both the route boundary and anchors have
            # committed, not merely the candidate's detached context copy.
            assert checkpoint["model_failover"]["projection_cursor"] > 0
            assert checkpoint["automatic_recall"]["anchor_transcript_index"] == 3 + 2 * len(
                self.requests
            )
            if outcome == "repeated" and self.name == "backup":
                self.requests.append(request)
                raise ModelProviderError(
                    "unavailable", provider=self.name, retryable=True, status_code=503
                )
            async for event in super().stream(request):
                yield event

    async def scenario():
        nonlocal store
        store = MemoryStore() if backend == "memory" else SQLiteStore(tmp_path / "recall.sqlite")
        knowledge = InMemoryKnowledgeStore(
            access_scope=KnowledgeAccessScope.for_namespace("project:cayu")
        )
        await knowledge.create_entry(
            KnowledgeEntry(
                id="atlas-release", namespace="project:cayu", text="Atlas release is Friday."
            )
        )
        source = Source(
            "source",
            [
                [
                    ModelStreamEvent.completed(
                        {"provider_state": [{"provider": "source", "state": {"id": "response-1"}}]}
                    )
                ],
                [ModelStreamEvent.text_delta("Friday confirmed"), ModelStreamEvent.completed()],
                [],
                [ModelStreamEvent.text_delta("Still Friday"), ModelStreamEvent.completed()],
            ],
        )
        backup = Backup(
            "backup",
            [[ModelStreamEvent.text_delta("backup answer"), ModelStreamEvent.completed()]] * 2,
        )
        final = Backup(
            "final",
            [[ModelStreamEvent.text_delta("final answer"), ModelStreamEvent.completed()]] * 2,
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
        app.register_provider(backup)
        app.register_provider(final)
        app.register_environment(
            Environment(EnvironmentSpec(name="local"), knowledge_store=knowledge), default=True
        )
        app.register_agent(
            AgentSpec(name="assistant", model="source-model"),
            context_policy=CancelFallbackContext() if outcome == "cancelled" else _policy(),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "When is Atlas released?")],
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(
                                ModelTarget(provider_name="backup", model="backup-model"),
                                ModelTarget(provider_name="final", model="final-model"),
                            ),
                            max_total_attempts=3,
                        ),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            for index, text in enumerate(
                ("Confirm the Atlas release date.", "Confirm the release again.")
            ):
                stream = app.resume(
                    ResumeRequest(session_id=session_id, messages=[Message.text("user", text)])
                )
                if outcome == "cancelled" and index == 1:
                    with pytest.raises(asyncio.CancelledError) as caught:
                        events = [event async for event in stream]
                    assert caught.value is cancellation
                else:
                    events = [event async for event in stream]
            checkpoint = await store.load_checkpoint(session_id)
            assert len(source.requests) == 3
            if outcome in {"refused", "cancelled"}:
                if outcome == "refused":
                    assert events[-1].type is EventType.SESSION_FAILED
                    assert "test fallback preparation refused" in str(events[-1].payload)
                else:
                    progress = context_build_termination_checkpoint_error(cancellation)
                    assert progress.checkpoint["automatic_recall"]["anchor_transcript_index"] == 4
                assert not backup.requests
                assert checkpoint["model_failover"]["candidate_index"] == 0
                assert checkpoint["model_failover"]["projection_cursor"] == 0
                assert checkpoint["automatic_recall"]["anchor_transcript_index"] == 4
                if outcome == "cancelled":
                    events = [
                        event
                        async for event in app.resume(
                            ResumeRequest(
                                session_id=session_id, messages=[Message.text("user", "Thanks.")]
                            )
                        )
                    ]
                    assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
            else:
                assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
                assert len(backup.requests) == 1
                assert len(final.requests) == (1 if outcome == "repeated" else 0)
                assert checkpoint["automatic_recall"]["anchor_transcript_index"] == 3
                # A later invocation must load the already-projected positions.
                events = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id=session_id, messages=[Message.text("user", "Thanks.")]
                        )
                    )
                ]
                assert events[-1].type is EventType.SESSION_COMPLETED, events[-1].payload
        finally:
            await app.aclose()
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    store = None
    session_id = "failover-recall"
    asyncio.run(scenario())
