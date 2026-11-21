"""Resume must preflight every newly admitted fallback before altering the session."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_execution_profiles import RecordingExecutionProfilePolicy

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.context.structured_output import NativeStructuredOutputUnsupported, StructuredOutputSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers.base import ModelProviderError, ModelStreamEvent
from cayu.providers.retry_policy import RetryPolicy
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyResult,
)
from cayu.sessions import InMemorySessionStore, ModelFailoverPolicy, ModelTarget
from cayu.storage import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_resume_rejects_non_native_fallback_before_admission(tmp_path, backend):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "fallback-preflight.sqlite")
        )

        def primary_response(_request):
            if len(primary.requests) == 1:
                return [
                    ModelStreamEvent.text_delta('{"answer":"first"}'),
                    ModelStreamEvent.completed(),
                ]
            raise ModelProviderError(
                "service unavailable", provider="primary", status_code=503, retryable=True
            )

        primary = ScriptedModelProvider(
            name="primary",
            response_factory=primary_response,
            supports_native_structured_output=True,
        )
        fallback = ScriptedModelProvider(
            [ModelStreamEvent.text_delta("ordinary text"), ModelStreamEvent.completed()],
            name="fallback",
            supports_native_structured_output=False,
        )
        app = CayuApp(
            session_store=store,
            execution_profile_policy=RecordingExecutionProfilePolicy(
                ExecutionProfilePolicyResult(
                    action=ExecutionProfilePolicyAction.ADOPT,
                    reason="Authorize the requested fallback plan.",
                    authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                )
            ),
            enable_logging=False,
        )
        app.register_provider(primary, default=True)
        app.register_provider(fallback)
        app.register_agent(AgentSpec(name="agent", model="small"))
        spec = StructuredOutputSpec(
            strategy="native",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            max_retries=0,
        )
        retry = RetryPolicy(max_attempts=1, initial_delay_s=0)
        try:
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="fallback-preflight",
                        messages=[Message.text("user", "first")],
                        structured_output=spec,
                        retry_policy=retry,
                    )
                )
            ]
            assert initial[-1].type is EventType.SESSION_COMPLETED
            before = await store.load("fallback-preflight")
            assert before.status is SessionStatus.COMPLETED
            before_transcript = await store.load_transcript(before.id)
            with pytest.raises(NativeStructuredOutputUnsupported):
                async for _event in app.resume(
                    ResumeRequest(
                        session_id=before.id,
                        messages=[Message.text("user", "second")],
                        structured_output=spec,
                        retry_policy=retry,
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="fallback", model="large"),),
                            max_total_attempts=2,
                        ),
                        profile_adoption=ExecutionProfileAdoptionIntent(
                            idempotency_key="add-fallback",
                            reason="Add a fallback for provider outages.",
                            requested_by=ResolutionActor(
                                subject="maintainer", source=ResolutionActorSource.REQUEST
                            ),
                        ),
                    )
                ):
                    pass
            current = await store.load(before.id)
            assert current == before
            assert await store.load_transcript(before.id) == before_transcript
            assert len(primary.requests) == 1
            assert fallback.requests == []
        finally:
            await app.aclose()
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
