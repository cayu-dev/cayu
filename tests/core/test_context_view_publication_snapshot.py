from __future__ import annotations

import asyncio
import json
import threading

import pytest

from cayu.agents import AgentSpec
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.context import MessageWindowContextPolicy
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._model_completion_publication import model_step_publication_from_checkpoint
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import execution_profile_from_session_metadata
from cayu.sessions.base import InMemorySessionStore, Message, ResumeRequest, RunRequest
from cayu.sessions.context_views import (
    ContextViewPublicationRequest,
    ParticipantSessionCreationRequest,
    ParticipantSessionExecutionRequest,
)
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "scenario", ["adoption", "snapshot_race", "deleted_source", "replaced_source"]
)
def test_publication_uses_one_historical_boundary(
    backend, scenario, tmp_path, request, monkeypatch
):
    from tests.core.test_participant_identity import CONTEXT, app, create, registration

    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def run():
        if backend == "memory":
            sessions = InMemorySessionStore()
            collaboration = InMemoryCollaborationStore()
        elif backend == "sqlite":
            from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore

            sessions = SQLiteSessionStore(tmp_path / "snapshot.sqlite")
            collaboration = SQLiteCollaborationStore(tmp_path / "participants.sqlite")
        else:
            from cayu.storage.collaboration_postgres import PostgresCollaborationStore
            from cayu.storage.migrations import SchemaMode
            from cayu.storage.postgres import PostgresSessionStore

            sessions = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)
            collaboration = PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE)

        provider_entered, provider_release = asyncio.Event(), asyncio.Event()

        class Provider(ScriptedModelProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="snapshot-test-provider", behavior_version="1", implementation_version="1"
                )

            async def stream(self, request):
                if scenario == "adoption" and self.requests:
                    provider_entered.set()
                    await provider_release.wait()
                async for event in super().stream(request):
                    yield event

        provider = Provider(
            [
                [
                    ModelStreamEvent.text_delta(text),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
                for text in ("historical first", "new second")
            ]
        )
        reg = registration()
        value = app(collaboration, reg, session_store=sessions)
        value.register_provider(provider, default=True)
        value.register_agent(AgentSpec(name="reviewer", model="model"))
        pending = None
        extra_sessions = None
        release_read = threading.Event()
        try:
            initialized = await value.initialize_collaboration()
            _, created = await create(value, initialized)
            participant = created.participants[0].reference
            creation = ParticipantSessionCreationRequest(
                RunRequest(agent_name="reviewer", messages=[Message.text("user", "first")]),
                f"snapshot-creation-{scenario}-{backend}",
            )
            session, _ = await value.create_participant_session(
                creation, participant=participant, context=CONTEXT
            )
            execution = ParticipantSessionExecutionRequest(
                request=creation.request.model_copy(update={"session_id": session.id}),
                session_instance_id=session.instance_id,
                execution_key="snapshot-execution",
            )
            events = [
                event
                async for event in value.execute_participant_session(
                    execution, participant=participant, context=CONTEXT
                )
            ]
            assert any(event.type is EventType.SESSION_COMPLETED for event in events)
            original = await sessions.load(session.id)
            profile = execution_profile_from_session_metadata(original.metadata)
            pointer = model_step_publication_from_checkpoint(
                await sessions.load_checkpoint(session.id)
            )
            completion = next(
                event
                for event in await sessions.load_events(session.id)
                if event.id == pointer.completion_event_id
            )
            publication = ContextViewPublicationRequest(
                source_session_id=session.id,
                source_session_instance_id=session.instance_id,
                view_id=f"snapshot-view-{scenario}-{backend}",
                interaction_id=completion.interaction_id,
                boundary_id=pointer.logical_step_id,
                publication_key=f"snapshot-publication-{scenario}-{backend}",
                projection_schema="whole-turn.v1",
            )

            async def resume(application=value, adoption=None):
                return [
                    event
                    async for event in application.resume(
                        ResumeRequest(
                            session_id=session.id,
                            messages=[Message.text("user", "second")],
                            profile_adoption=adoption,
                        ),
                        context=CONTEXT,
                    )
                ]

            if scenario == "adoption":
                from tests.core.test_execution_profiles import RecordingExecutionProfilePolicy

                from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
                from cayu.runtime.execution_profiles import (
                    ExecutionProfileAdoptionIntent,
                    ExecutionProfileAuthorityDecision,
                    ExecutionProfilePolicyAction,
                    ExecutionProfilePolicyResult,
                )

                adopted = app(
                    collaboration,
                    reg,
                    session_store=sessions,
                    execution_profile_policy=RecordingExecutionProfilePolicy(
                        ExecutionProfilePolicyResult(
                            action=ExecutionProfilePolicyAction.ADOPT,
                            reason="reviewed",
                            authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                        )
                    ),
                )
                adopted.register_provider(provider, default=True)
                adopted.register_agent(
                    AgentSpec(name="reviewer", model="model"),
                    context_policy=MessageWindowContextPolicy(max_messages=50),
                )
                await adopted.initialize_collaboration()
                intent = ExecutionProfileAdoptionIntent(
                    idempotency_key="snapshot-adoption",
                    reason="reviewed",
                    requested_by=ResolutionActor(
                        subject="maintainer", source=ResolutionActorSource.REQUEST
                    ),
                )
                pending = asyncio.create_task(resume(adopted, intent))
                await asyncio.wait_for(provider_entered.wait(), 30)
                current = await sessions.load(session.id)
                assert (
                    execution_profile_from_session_metadata(current.metadata).fingerprint
                    != profile.fingerprint
                )
                manifest = await value.publish_completed_context_view(
                    publication, participant=participant, context=CONTEXT
                )
                provider_release.set()
                await pending
            elif scenario == "snapshot_race":
                # Pause after the session read but before the checkpoint read.
                # A second connection completes another real public turn.
                read_entered = threading.Event()
                continuation = value
                if backend == "sqlite":
                    import cayu.storage.sqlite as sqlite_module

                    extra_sessions = SQLiteSessionStore(tmp_path / "snapshot.sqlite")
                    continuation = app(collaboration, reg, session_store=extra_sessions)
                    continuation.register_provider(provider, default=True)
                    continuation.register_agent(AgentSpec(name="reviewer", model="model"))
                    await continuation.initialize_collaboration()
                    original_load = sqlite_module._load_session
                    armed = True

                    def load(connection, session_id):
                        nonlocal armed
                        result = original_load(connection, session_id)
                        if armed and threading.current_thread() is not threading.main_thread():
                            armed = False
                            read_entered.set()
                            assert release_read.wait(30)
                        return result

                    monkeypatch.setattr(sqlite_module, "_load_session", load)
                elif backend == "postgres":
                    original_load = sessions._load

                    async def load(cur, session_id):
                        result = await original_load(cur, session_id)
                        if asyncio.current_task().get_name() == "publication-snapshot":
                            read_entered.set()
                            assert await asyncio.to_thread(release_read.wait, 30)
                        return result

                    monkeypatch.setattr(sessions, "_load", load)
                else:
                    # Memory holds its single native lock without an await;
                    # advance immediately after capture instead.
                    original_capture = sessions.capture_context_view_publication_source

                    async def capture(session_id):
                        result = await original_capture(session_id)
                        read_entered.set()
                        assert await asyncio.to_thread(release_read.wait, 30)
                        return result

                    monkeypatch.setattr(
                        sessions, "capture_context_view_publication_source", capture
                    )
                pending = asyncio.create_task(
                    value.publish_completed_context_view(
                        publication, participant=participant, context=CONTEXT
                    ),
                    name="publication-snapshot",
                )
                assert await asyncio.to_thread(read_entered.wait, 30)
                continued = await resume(continuation)
                assert any(event.type is EventType.SESSION_COMPLETED for event in continued)
                release_read.set()
                manifest = await pending
            else:
                original_capture = sessions.capture_context_view_publication_source

                async def capture(session_id):
                    result = await original_capture(session_id)
                    await sessions.delete_session(session_id)
                    if scenario == "replaced_source":
                        from cayu.sessions.base import SessionIdentity

                        replacement = await sessions.create(
                            RunRequest(agent_name="reviewer", session_id=session_id, messages=[]),
                            identity=SessionIdentity(
                                provider_name=original.provider_name, model=original.model
                            ),
                        )
                        assert replacement.instance_id != session.instance_id
                    return result

                monkeypatch.setattr(sessions, "capture_context_view_publication_source", capture)
                with pytest.raises(LookupError, match="incarnation"):
                    await value.publish_completed_context_view(
                        publication, participant=participant, context=CONTEXT
                    )
                assert (
                    await sessions.lookup_context_view_publication(publication.publication_key)
                    is None
                )
                return

            assert "historical first" in manifest.messages_json
            assert "new second" not in manifest.messages_json
            assert manifest.completion_event_id == completion.id
            assert json.loads(manifest.historical_ancestry_json)[
                "execution_profile"
            ] == profile.model_dump(mode="json")
            assert (
                await value.publish_completed_context_view(
                    publication, participant=participant, context=CONTEXT
                )
                == manifest
            )
        finally:
            release_read.set()
            provider_release.set()
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            if backend != "memory":
                if extra_sessions is not None:
                    await extra_sessions.close()
                await sessions.close()
                await collaboration.close()

    asyncio.run(run())
