"""Public profile adoption retains selected routing without reusing attempts."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_execution_profiles import RecordingExecutionProfilePolicy
from tests.core.test_model_failover_recovery import _RecoveryProvider
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    IncompleteSessionRecoveryRequest,
    Message,
    ModelFailoverPolicy,
    ModelTarget,
    ResumeRequest,
    RunRequest,
)
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.context.base import MessageWindowContextPolicy
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyResult,
    execution_profile_from_session_metadata,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions.base import SessionStatus, SessionStore


class _LoseAdmissionAcknowledgement(SessionStore):
    invocation_lifecycle_command_version = 1
    lose_admission_acknowledgement = False
    lost_acknowledgements = 0

    async def transition_status_and_checkpoint(self, session_id, **kwargs):
        result = await super().transition_status_and_checkpoint(session_id, **kwargs)
        if self.lose_admission_acknowledgement and kwargs["to_status"] is SessionStatus.RUNNING:
            self.lose_admission_acknowledgement = False
            self.lost_acknowledgements += 1
            raise OSError("admission committed but acknowledgement was lost")
        return result


class _LostAdmissionMemory(_LoseAdmissionAcknowledgement, _StageMemoryStore):
    model_failover_stage_version = 1
    invocation_lifecycle_command_version = 1


class _LostAdmissionSQLite(_LoseAdmissionAcknowledgement, _StageSQLiteStore):
    model_failover_stage_version = 1
    invocation_lifecycle_command_version = 1


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("boundary", ["normal", "close", "cancel", "lost_ack"])
@pytest.mark.parametrize("change", ["context", "attempt_cap", "chain"])
def test_public_profile_adoption_preserves_selected_fallback(
    monkeypatch, tmp_path, backend, boundary, change, request
):
    session_id = f"adoption-{uuid4().hex}"
    database = tmp_path / "adoption.sqlite"
    if backend == "postgres":
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        dsn = request.getfixturevalue("postgres_dsn")

        class StagePostgresStore(_LoseAdmissionAcknowledgement, PostgresSessionStore):
            model_failover_stage_version = 1
            invocation_lifecycle_command_version = 1

    def fresh_store():
        if backend == "postgres":
            return StagePostgresStore(dsn, schema_mode=SchemaMode.CREATE)
        if backend == "sqlite":
            return _LostAdmissionSQLite(database)
        return _LostAdmissionMemory()

    def application(store, *, revision):
        app = CayuApp(
            session_store=store,
            execution_profile_policy=(
                RecordingExecutionProfilePolicy(
                    ExecutionProfilePolicyResult(
                        action=ExecutionProfilePolicyAction.ADOPT,
                        reason="Reviewed context-policy change.",
                        authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                    )
                )
                if revision
                else None
            ),
            enable_logging=False,
        )
        primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_provider(_RecoveryProvider("third"))
        app.register_agent(
            AgentSpec(name="agent", model="small"),
            context_policy=MessageWindowContextPolicy(
                max_messages=4 + revision if change == "context" else 4
            ),
        )
        return app, primary, backup

    def requested_policy(revision):
        if change == "context":
            return None
        return ModelFailoverPolicy(
            fallbacks=(
                ModelTarget(provider_name="backup", model="large"),
                *(
                    (ModelTarget(provider_name="third", model=f"extra-{revision}"),)
                    if change == "chain"
                    else ()
                ),
            ),
            max_total_attempts=2 + revision,
        )

    async def scenario():
        store = fresh_store()
        retry = RetryPolicy(max_attempts=1, initial_delay_s=0)
        try:
            app, primary, backup = application(store, revision=0)
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id=session_id,
                        messages=[Message.text("user", "first")],
                        retry_policy=retry,
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                            max_total_attempts=2,
                        ),
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(primary.requests) == len(backup.requests) == 1
            original = await store.load_checkpoint(session_id)
            assert original is not None and original["model_failover"]["candidate_index"] == 1
            if backend != "memory":
                await store.close()
                store = fresh_store()
            app, primary, backup = application(store, revision=1)
            if boundary == "lost_ack":
                store.lose_admission_acknowledgement = True
            request = ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "second")],
                retry_policy=retry,
                failover=requested_policy(1),
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key="adopt-context",
                    reason="Use reviewed context policy.",
                    requested_by=ResolutionActor(
                        subject="maintainer", source=ResolutionActorSource.REQUEST
                    ),
                ),
            )
            if boundary in {"close", "cancel"}:
                stream = app.resume(request)
                admitted = asyncio.Event()
                caught_cancellation = []

                async def consume_admission():
                    try:
                        async for event in stream:
                            if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED:
                                admitted.set()
                                if boundary == "cancel":
                                    await asyncio.Event().wait()
                                return
                    except asyncio.CancelledError:
                        caught_cancellation.append(True)
                        raise
                    finally:
                        await stream.aclose()

                consumer = asyncio.create_task(consume_admission())
                await asyncio.wait_for(admitted.wait(), timeout=30)
                if boundary == "cancel":
                    consumer.cancel()
                    assert consumer.cancelling() == 1
                    with pytest.raises(asyncio.CancelledError):
                        await consumer
                    assert consumer.cancelled() and consumer.cancelling() == 1
                    assert caught_cancellation == [True]
                else:
                    await consumer
                assert not primary.requests and not backup.requests
                selected = await store.load_checkpoint(session_id)
                assert selected is not None
                assert selected["model_failover"]["state"] == "selected"
                assert selected["model_failover"]["candidate_index"] == (
                    0 if change == "chain" else 1
                )
                assert "stage_id" not in selected["model_failover"]
                assert await store.load_active_model_completion_stage(session_id) is None
                if backend != "memory":
                    await store.close()
                    store = fresh_store()
                # Abandoning the admission stream leaves a recoverable running
                # invocation. Repair it under its admitted profile before
                # requesting a different profile; never bypass that fence.
                repair_app, repair_primary, repair_backup = application(store, revision=1)
                await repair_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
                assert not repair_primary.requests and not repair_backup.requests
                repaired = await store.load_checkpoint(session_id)
                assert repaired is not None
                assert repaired["model_failover"] == selected["model_failover"]
                app, primary, backup = application(store, revision=2)
                request = ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "third")],
                    retry_policy=retry,
                    failover=requested_policy(2),
                    profile_adoption=ExecutionProfileAdoptionIntent(
                        idempotency_key="adopt-context-again",
                        reason="Use the next reviewed context policy.",
                        requested_by=ResolutionActor(
                            subject="maintainer", source=ResolutionActorSource.REQUEST
                        ),
                    ),
                )
            continued = [event async for event in app.resume(request)]
            assert continued[-1].type is EventType.SESSION_COMPLETED
            if boundary == "lost_ack":
                assert store.lost_acknowledgements == 1
            assert len(primary.requests) == (1 if change == "chain" else 0)
            assert len(backup.requests) == 1
            checkpoint = await store.load_checkpoint(session_id)
            session = await store.load(session_id)
            assert checkpoint is not None and session is not None
            progress = checkpoint["model_failover"]
            profile = execution_profile_from_session_metadata(session.metadata)
            assert progress["execution_profile_fingerprint"] == profile.fingerprint
            assert progress["candidate_index"] == 1
            assert progress["attempts_used"] == (2 if change == "chain" else 1)
            assert progress["stage_id"] != original["model_failover"]["stage_id"]
            assert await store.load_active_model_completion_stage(session.id) is None
            if backend != "memory":
                await store.close()
                store = fresh_store()
            replay_app, replay_primary, replay_backup = application(
                store, revision=2 if boundary in {"close", "cancel"} else 1
            )
            replay = [event async for event in replay_app.resume(request)]
            decisions = [
                event
                for event in continued
                if event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
            ]
            assert [event.id for event in replay] == [event.id for event in decisions]
            assert len(primary.requests) == (1 if change == "chain" else 0)
            assert len(backup.requests) == 1
            assert not replay_primary.requests and not replay_backup.requests
            assert await store.load_checkpoint(session_id) == checkpoint
            assert await store.load(session_id) == session
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(scenario())
