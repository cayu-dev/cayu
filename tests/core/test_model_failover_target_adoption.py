"""Explicit target adoption leaves the old failover route behind atomically."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_execution_profiles import RecordingExecutionProfilePolicy
from tests.core.test_model_failover_recovery import _RecoveryProvider
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelFailoverPolicy,
    ModelTarget,
    ResumeRequest,
    RunRequest,
)
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyResult,
    execution_profile_from_session_metadata,
)
from cayu.runtime.retry_policy import RetryPolicy


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_explicit_target_adoption_removes_settled_failover_route(monkeypatch, tmp_path, backend):

    def application(store):
        app = CayuApp(
            session_store=store,
            execution_profile_policy=RecordingExecutionProfilePolicy(
                ExecutionProfilePolicyResult(
                    action=ExecutionProfilePolicyAction.ADOPT,
                    reason="Reviewed explicit target change.",
                    authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                )
            ),
            enable_logging=False,
        )
        primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        return app, primary, backup

    async def scenario():
        path = tmp_path / "target-adoption.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
        retry = RetryPolicy(max_attempts=1, initial_delay_s=0)
        try:
            app, primary, backup = application(store)
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="source",
                        messages=[Message.text("user", "first")],
                        retry_policy=retry,
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                            max_total_attempts=2,
                        ),
                    )
                )
            ]
            assert initial[-1].type is EventType.SESSION_COMPLETED
            prior = await store.load_checkpoint("source")
            assert prior is not None
            old_stage = prior["model_failover"]["stage_id"]
            request = ResumeRequest(
                session_id="source",
                messages=[Message.text("user", "second")],
                target=ModelTarget(provider_name="backup", model="large"),
                retry_policy=retry,
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key="explicit-target",
                    reason="Adopt one explicit target.",
                    requested_by=ResolutionActor(
                        subject="maintainer", source=ResolutionActorSource.REQUEST
                    ),
                ),
            )
            events = [event async for event in app.resume(request)]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(primary.requests) == 1 and len(backup.requests) == 2
            checkpoint = await store.load_checkpoint("source")
            session = await store.load("source")
            assert checkpoint is not None and "model_failover" not in checkpoint
            assert session is not None and session.provider_name == "backup"
            assert execution_profile_from_session_metadata(session.metadata).model_failover is None
            assert await store.load_active_model_completion_stage("source") is None
            retained = await store.load_model_completion_stage("source", old_stage)
            assert retained is not None and retained.state == "completed"
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
            app, primary, backup = application(store)
            replay = [event async for event in app.resume(request)]
            assert [event.id for event in replay] == [
                event.id
                for event in events
                if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
            ]
            assert not primary.requests and not backup.requests
            assert await store.load_checkpoint("source") == checkpoint
            assert await store.load("source") == session
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_explicit_adoption_adds_failover_to_single_provider_session(monkeypatch, tmp_path, backend):

    def application(store):
        app = CayuApp(
            session_store=store,
            execution_profile_policy=RecordingExecutionProfilePolicy(
                ExecutionProfilePolicyResult(
                    action=ExecutionProfilePolicyAction.ADOPT,
                    reason="Reviewed addition of a fallback target.",
                    authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                )
            ),
            enable_logging=False,
        )
        original, fallback = _RecoveryProvider("original"), _RecoveryProvider("fallback")
        app.register_provider(original, default=True)
        app.register_provider(fallback)
        app.register_agent(AgentSpec(name="agent", model="small"))
        return app, original, fallback

    async def scenario():
        path = tmp_path / "add-policy.sqlite"
        store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
        retry = RetryPolicy(max_attempts=1)
        try:
            app, original, fallback = application(store)
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="source",
                        messages=[Message.text("user", "first")],
                        retry_policy=retry,
                    )
                )
            ]
            assert initial[-1].type is EventType.SESSION_COMPLETED
            checkpoint = await store.load_checkpoint("source")
            assert checkpoint is not None and "model_failover" not in checkpoint
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
            app, original, fallback = application(store)
            request = ResumeRequest(
                session_id="source",
                messages=[Message.text("user", "second")],
                retry_policy=retry,
                failover=ModelFailoverPolicy(
                    fallbacks=(ModelTarget(provider_name="fallback", model="large"),),
                    max_total_attempts=2,
                ),
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key="add-fallback",
                    reason="Add reviewed fallback policy.",
                    requested_by=ResolutionActor(
                        subject="maintainer", source=ResolutionActorSource.REQUEST
                    ),
                ),
            )
            events = [event async for event in app.resume(request)]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(original.requests) == 1 and not fallback.requests
            selections = [
                event for event in events if event.type is EventType.MODEL_FAILOVER_SELECTED
            ]
            assert len(selections) == 1
            assert selections[0].payload["reason"] == "initial"
            assert selections[0].payload["provider"] == "original"
            assert selections[0].payload["previous_stage_id"] is None
            checkpoint = await store.load_checkpoint("source")
            session = await store.load("source")
            assert checkpoint is not None and session is not None
            progress = checkpoint["model_failover"]
            assert progress["candidate_index"] == 0
            assert progress["attempts_used"] == progress["generation"] == 1
            assert progress["session_instance_id"] == session.instance_id
            profile = execution_profile_from_session_metadata(session.metadata)
            assert profile.model_failover is not None
            assert progress["execution_profile_fingerprint"] == profile.fingerprint
            assert await store.load_active_model_completion_stage("source") is None
            if isinstance(store, _StageSQLiteStore):
                await store.close()
                store = _StageSQLiteStore(path)
            app, original, fallback = application(store)
            replay = [event async for event in app.resume(request)]
            assert [event.id for event in replay] == [
                event.id
                for event in events
                if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
            ]
            assert not original.requests and not fallback.requests
            assert await store.load_checkpoint("source") == checkpoint
            assert await store.load("source") == session
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())
