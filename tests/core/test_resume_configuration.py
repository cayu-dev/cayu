"""Ordinary continuation inherits only verified, profile-bound loop controls."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileMismatchError,
    Message,
    ModelStreamEvent,
    ResolutionActor,
    ResolutionActorSource,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
)
from cayu.runtime._model_step_executor import model_completion_recovery_context_from_stage
from cayu.sessions._model_completion_publication import model_step_publication_from_checkpoint
from cayu.sessions.base import InMemorySessionStore


def application(store, **kwargs):
    app = CayuApp(session_store=store, enable_logging=False, **kwargs)
    provider = ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("ok"), ModelStreamEvent.completed()]]
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="support", model="scripted"))
    return app, provider


async def start(app):
    return [
        event
        async for event in app.run(
            RunRequest(
                session_id="conversation",
                agent_name="support",
                messages=[Message.text("user", "first")],
                max_steps=18,
                limits=RunLimits(max_tool_calls=7),
                retry_policy=RetryPolicy(max_attempts=2),
            )
        )
    ]


async def resume(app, **controls):
    return [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id="conversation", messages=[Message.text("user", "again")], **controls
            )
        )
    ]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("override", [None, "max_steps", "limits", "retry_policy"])
def test_resume_inheritance_and_explicit_changes(sqlite_resources, backend, override):
    async def run():
        async with sqlite_resources as resources:
            store = (
                InMemorySessionStore()
                if backend == "memory"
                else resources.own(SQLiteSessionStore(resources.path()))
            )
            app, _ = application(store)
            await start(app)
            restarted, provider = application(store)
            controls = {
                "max_steps": 64,
                "limits": RunLimits(),
                "retry_policy": RetryPolicy(),
            }
            if override:
                with pytest.raises(ExecutionProfileMismatchError) as error:
                    await resume(restarted, **{override: controls[override]})
                assert override in " ".join(error.value.__notes__)
                assert not provider.requests
            else:
                events = await resume(restarted)
                assert events[-1].type == "session.completed"
                pointer = model_step_publication_from_checkpoint(
                    await store.load_checkpoint("conversation")
                )
                stage = await store.load_model_completion_stage("conversation", pointer.stage_id)
                context = model_completion_recovery_context_from_stage(stage)
                assert context.max_steps == 18
                assert context.limits.max_tool_calls == 7
                assert context.retry_policy.max_attempts == 2
                assert len(provider.requests) == 1

    asyncio.run(run())


def test_resume_nondefault_controls_in_new_process(tmp_path):
    script = """
import asyncio, sys
from cayu import SQLiteSessionStore
from tests.core.test_resume_configuration import application, start, resume
async def run():
    store = SQLiteSessionStore(sys.argv[1])
    try:
        app, _ = application(store)
        events = await (start(app) if sys.argv[2] == 'start' else resume(app))
        assert events[-1].type == 'session.completed'
    finally:
        await store.close()
asyncio.run(run())
"""
    for action in ("start", "resume"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "conversation.sqlite"), action],
            env={**os.environ, "PYTHONPATH": "src:."},
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("revoked", [False, True])
def test_inherited_resume_still_checks_revoked_resource_access(sqlite_resources, revoked):
    from tests.core.test_resource_execution_access import Policy

    from cayu.sessions.access import SessionAccessDenied, SessionAccessScope

    async def run():
        async with sqlite_resources as resources:
            store = resources.own(SQLiteSessionStore(resources.path()))
            policy = Policy()
            app, _ = application(store, resource_access_policy=policy)
            rules = policy.current.read
            policy.current = SessionAccessScope(read=rules, create=rules, execute=rules)
            access = await app.access("alice")
            _ = [
                event
                async for event in access.run(
                    RunRequest(
                        session_id="conversation",
                        agent_name="support",
                        messages=[Message.text("user", "first")],
                        max_steps=18,
                        labels={"organization": "acme"},
                    )
                )
            ]
            restarted, provider = application(store, resource_access_policy=policy)
            if revoked:
                policy.current = SessionAccessScope()
                with pytest.raises(SessionAccessDenied):
                    await resume(restarted)
                assert not provider.requests
            else:
                events = await resume(restarted)
                assert events[-1].type == "session.completed"
                assert len(provider.requests) == 1

    asyncio.run(run())


@pytest.mark.parametrize("damage", ["legacy", "max_steps", "interaction", "profile"])
def test_resume_does_not_inherit_unbound_or_legacy_defaults(monkeypatch, damage):
    async def run():
        store = InMemorySessionStore()
        app, _ = application(store)
        await start(app)
        original = store.load_model_completion_stage

        async def historical_stage(session_id, stage_id):
            stage = await original(session_id, stage_id)
            if stage is None:
                return None
            intent = dict(stage.intent)
            context = dict(intent["recovery_context"])
            if damage == "legacy":
                # Foreground records written before configuration inheritance
                # omitted these controls; their schema defaults are not evidence.
                for name in ("max_steps", "limits", "retry_policy"):
                    context.pop(name, None)
            elif damage == "max_steps":
                context["max_steps"] = 64
            elif damage == "interaction":
                context["interaction_id"] = "another-interaction"
            else:
                context["execution_profile_fingerprint"] = "f" * 64
            intent["recovery_context"] = context
            return stage.model_copy(update={"intent": intent})

        monkeypatch.setattr(store, "load_model_completion_stage", historical_stage)
        restarted, provider = application(store)
        with pytest.raises(ExecutionProfileMismatchError) as error:
            await resume(restarted)
        assert not getattr(error.value, "__notes__", [])
        assert not provider.requests

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("legacy", [False, True])
def test_adoption_replay_preserves_request_before_inheritance(
    sqlite_resources, monkeypatch, backend, legacy
):
    async def run():
        async with sqlite_resources as resources:
            store = (
                InMemorySessionStore()
                if backend == "memory"
                else resources.own(SQLiteSessionStore(resources.path()))
            )
            app, _ = application(store)
            if legacy:
                _ = [
                    event
                    async for event in app.run(
                        RunRequest(
                            session_id="conversation",
                            agent_name="support",
                            messages=[Message.text("user", "first")],
                        )
                    )
                ]
                pointer = model_step_publication_from_checkpoint(
                    await store.load_checkpoint("conversation")
                )
                original = store.load_model_completion_stage

                async def historical_stage(session_id, stage_id):
                    stage = await original(session_id, stage_id)
                    if stage is None or stage_id != pointer.stage_id:
                        return stage
                    intent = dict(stage.intent)
                    context = dict(intent["recovery_context"])
                    # Only the predecessor predates persisted foreground controls.
                    for name in ("max_steps", "limits", "retry_policy"):
                        context.pop(name, None)
                    intent["recovery_context"] = context
                    return stage.model_copy(update={"intent": intent})

                monkeypatch.setattr(store, "load_model_completion_stage", historical_stage)
            else:
                await start(app)

            request = ResumeRequest(
                session_id="conversation",
                messages=[Message.text("user", "adopt")],
                profile_adoption=ExecutionProfileAdoptionIntent(
                    idempotency_key="deployment-v1",
                    reason="Continue this deployment.",
                    requested_by=ResolutionActor(
                        subject="operator", source=ResolutionActorSource.REQUEST
                    ),
                ),
            )
            adopting_app, adopting_provider = application(store)
            applied = [event async for event in adopting_app.resume(request)]
            assert applied[-1].type == "session.completed"
            assert len(adopting_provider.requests) == 1
            persisted = await store.load_events("conversation")
            assert (
                sum(event.type == "session.execution_profile.decided" for event in persisted) == 1
            )

            restarted_app, replay_provider = application(store)
            replayed = [event async for event in restarted_app.resume(request)]
            assert [event.type for event in replayed] == ["session.execution_profile.decided"]
            assert await store.load_events("conversation") == persisted
            assert not replay_provider.requests
            for changes in (
                {"messages": [Message.text("user", "different input")]},
                {"retry_policy": RetryPolicy(max_attempts=1)},
            ):
                with pytest.raises(ValueError, match="idempotency key"):
                    _ = [
                        event
                        async for event in restarted_app.resume(request.model_copy(update=changes))
                    ]
            assert not replay_provider.requests

    asyncio.run(run())
