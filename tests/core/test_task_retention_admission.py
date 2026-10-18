"""Task-retention failures cannot strand an admitted native invocation."""

import asyncio

import pytest
from tests.core.test_session_execution_presence import _app, _stores
from tests.runtime.runtime_provider_support import VersionedFakeProvider

from cayu import EventType, Message, RunRequest, SessionStatus
from cayu.providers import ModelStreamEvent
from cayu.sessions import ResumeRequest
from cayu.tasks import InMemoryTaskStore
from cayu.tasks.store import TaskStore


def test_closure_only_task_store_does_not_require_retention_epochs():
    class ClosureOnlyStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True
        supports_session_retention_epochs = False
        load_session_retention_epoch = TaskStore.load_session_retention_epoch
        advance_session_retention_epoch = TaskStore.advance_session_retention_epoch

    async def scenario():
        from cayu.sessions import InMemorySessionStore

        tasks = ClosureOnlyStore()
        provider = VersionedFakeProvider(
            [[ModelStreamEvent.completed({"finish_reason": "stop"})] for _ in range(2)]
        )
        app = _app(InMemorySessionStore(), provider, task_store=tasks)
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="closure-only",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
            events = [
                event
                async for event in app.resume(
                    ResumeRequest(session_id="closure-only", messages=[Message.text("user", "go")])
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("entrance", ["run", "resume"])
@pytest.mark.parametrize(
    "operation", ["load_session_retention_epoch", "advance_session_retention_epoch"]
)
@pytest.mark.parametrize("failure_kind", ["error", "cancel"])
def test_failed_retention_admission_is_interrupted_and_retryable(
    backend, entrance, operation, failure_kind, request, sqlite_resources, monkeypatch
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (sessions, _):
            tasks = InMemoryTaskStore()
            provider = VersionedFakeProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})] for _ in range(2)]
            )
            app = _app(sessions, provider, task_store=tasks)
            sid = "failed-retention"
            try:
                if entrance == "resume":
                    _ = [
                        event
                        async for event in app.run(
                            RunRequest(
                                agent_name="assistant",
                                session_id=sid,
                                messages=[Message.text("user", "go")],
                            )
                        )
                    ]
                previous_requests = len(provider.requests)
                original = getattr(tasks, operation)
                failure = (
                    OSError("task store unavailable")
                    if failure_kind == "error"
                    else asyncio.CancelledError()
                )

                async def fail(*args, **kwargs):
                    stored = await sessions.load(sid)
                    assert stored.status is SessionStatus.RUNNING
                    raise failure

                monkeypatch.setattr(tasks, operation, fail)
                stream = (
                    app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id=sid,
                            messages=[Message.text("user", "go")],
                        )
                    )
                    if entrance == "run"
                    else app.resume(
                        ResumeRequest(session_id=sid, messages=[Message.text("user", "go")])
                    )
                )
                with pytest.raises(type(failure)) as raised:
                    _ = [event async for event in stream]
                assert raised.value is failure
                assert len(provider.requests) == previous_requests
                assert (await sessions.load(sid)).status is SessionStatus.INTERRUPTED
                assert any(
                    event.type is EventType.SESSION_INTERRUPTED
                    for event in await sessions.load_events(sid)
                )
                monkeypatch.setattr(tasks, operation, original)
                if entrance == "run":
                    # Admission failed before the prepared initial transcript
                    # was published; that session remains inspectable, and its
                    # existing fail-closed resume rule still requires a new ID.
                    with pytest.raises(RuntimeError, match="authoritative initial transcript"):
                        _ = [
                            event
                            async for event in app.resume(
                                ResumeRequest(
                                    session_id=sid, messages=[Message.text("user", "retry")]
                                )
                            )
                        ]
                    retry = app.run(
                        RunRequest(
                            agent_name="assistant",
                            session_id="retry-retention",
                            messages=[Message.text("user", "go")],
                        )
                    )
                else:
                    retry = app.resume(
                        ResumeRequest(session_id=sid, messages=[Message.text("user", "retry")])
                    )
                events = [event async for event in retry]
                assert events[-1].type is EventType.SESSION_COMPLETED
                assert len(provider.requests) == previous_requests + 1
            finally:
                await app.aclose()

    asyncio.run(scenario())
