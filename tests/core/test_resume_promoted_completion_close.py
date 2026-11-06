"""Closing at a recovered model event must settle the admitted resume."""

import asyncio

import pytest
from tests.core.test_model_completion_recovery import (
    _RecordingProvider,
    _register_runtime,
    _stage_completed_model_boundary,
)

from cayu import EventType, Message, ResumeRequest, SessionStatus
from cayu.providers import ModelStreamEvent
from cayu.sessions import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("close_at_model", [False, True], ids=["consume-all", "close"])
def test_resume_promoted_model_close_settles_admitted_resume(tmp_path, backend, close_at_model):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "promoted-close.sqlite")
        )
        provider = _RecordingProvider(
            [
                [ModelStreamEvent.text_delta("continued"), ModelStreamEvent.completed()],
            ]
        )
        # Use the existing recovery fixture: a durable terminal model response
        # whose transcript/event publication has not yet been promoted.
        staged = await _stage_completed_model_boundary(
            store,
            session_id="promoted-close",
            provider_name=provider.name,
        )
        await store.release_run_fence(staged.session.id)
        await store.update_status(staged.session.id, SessionStatus.INTERRUPTED)
        app = _register_runtime(store, provider)
        try:
            stream = app.resume(
                ResumeRequest(
                    session_id=staged.session.id,
                    messages=[Message.text("user", "continue")],
                )
            )
            seen = []
            try:
                async for event in stream:
                    seen.append(event)
                    if close_at_model and event.type is EventType.MODEL_COMPLETED:
                        break
            finally:
                await stream.aclose()

            assert any(event.type is EventType.MODEL_COMPLETED for event in seen)
            if close_at_model:
                assert provider.requests == []
                assert not any(event.type is EventType.SESSION_RESUMED for event in seen)
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            session = await store.load(staged.session.id)
            owner = await store._inspect_session_execution_owner(staged.session.id)
            expected = SessionStatus.INTERRUPTED if close_at_model else SessionStatus.COMPLETED
            retry_failure = None
            retried = []
            if close_at_model:
                try:
                    retried = [
                        event
                        async for event in app.resume(
                            ResumeRequest(
                                session_id=staged.session.id,
                                messages=[Message.text("user", "try again after closing")],
                            )
                        )
                    ]
                except Exception as error:
                    retry_failure = type(error).__name__
            assert session.status is expected, (session.status, owner.state, retry_failure)
            if close_at_model:
                assert retry_failure is None
                assert retried[-1].type is EventType.SESSION_COMPLETED
        finally:
            await app.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
