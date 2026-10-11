"""Interruption operations compose independently of the session engine."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_recovery_composition import _run_without_owners

from cayu import AgentSpec, CayuApp, EventType, RunRequest
from cayu._exception_groups import iter_exception_tree
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import SessionIdentity, SessionStatus
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.workspaces.observation_recovery import (
    workspace_observation_pending_cancellation_requests,
)


def test_interruption_cleanup_imports_without_application_or_engine() -> None:
    _run_without_owners(
        ("cayu.applications", "cayu.runtime._session_engine"),
        "from cayu.runtime._session_interruption_cleanup import SessionInterruptionCleanup\n",
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_interruption_cleanup_settles_after_preparation_failure_without_engine(
    backend, sqlite_resources, monkeypatch
) -> None:
    async def scenario() -> None:
        async with sqlite_resources as resources:
            store = (
                InMemorySessionStore()
                if backend == "memory"
                else resources.own(SQLiteSessionStore(resources.path()))
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_agent(AgentSpec(name="assistant", model="fake"))
            registered_agent = app._get_registered_agent("assistant")
            owner = app._interruption_cleanup
            monkeypatch.delattr(app, "_session_engine")
            app.session_store = InMemorySessionStore()
            session = await store.create(
                RunRequest(agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            original = GeneratorExit("caller closed the stream")
            preparation_failure = RuntimeError("preparation failed")

            async def prepare() -> None:
                raise preparation_failure

            events, propagated = await owner.handle_session_interrupted_preserving_failure(
                authoritative_failure=original,
                session=session,
                registered_agent=registered_agent,
                registered_environment=None,
                environment_name=None,
                prepare_interruption=prepare,
            )
            leaves = list(iter_exception_tree(propagated))
            assert sum(candidate is original for candidate in leaves) == 1
            assert sum(candidate is preparation_failure for candidate in leaves) == 1
            assert [event.type for event in events] == [EventType.SESSION_INTERRUPTED]
            persisted = await store.load(session.id)
            assert persisted is not None
            assert persisted.status is SessionStatus.INTERRUPTED
            records = await store.query_events(EventQuery(session_id=session.id))
            assert [record.event.id for record in records] == [event.id for event in events]
            assert await app.session_store.load(session.id) is None
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("child_cancelled", [False, True])
def test_interruption_finalizer_retains_later_control_and_drains_all_steps(
    child_cancelled, monkeypatch
) -> None:
    async def scenario() -> None:
        app = CayuApp(enable_logging=False)
        owner = app._interruption_cleanup
        monkeypatch.delattr(app, "_session_engine")
        entered = asyncio.Event()
        release = asyncio.Event()
        finished: list[str] = []
        original = GeneratorExit("original stream close")
        cleanup_failure = (
            asyncio.CancelledError("child cancelled itself")
            if child_cancelled
            else RuntimeError("cleanup failed")
        )

        async def first_step() -> None:
            entered.set()
            await release.wait()
            finished.append("first")
            raise cleanup_failure

        async def second_step() -> None:
            finished.append("second")

        operation = asyncio.create_task(
            owner.settle_session_finalizer_preserving_interruption_failure(
                authoritative_failure=original,
                steps=(("first cleanup", first_step), ("second cleanup", second_step)),
            )
        )
        async with asyncio.timeout(5):
            await entered.wait()
            operation.cancel("later caller cancellation")
            while operation.cancelling():
                await asyncio.sleep(0)
            assert not operation.done()
            release.set()
            propagated = await operation
        assert finished == ["first", "second"]
        failures = list(iter_exception_tree(propagated))
        assert sum(candidate is original for candidate in failures) == 1
        assert any(
            isinstance(candidate, asyncio.CancelledError)
            and str(candidate) == "later caller cancellation"
            for candidate in failures
        )
        if child_cancelled:
            assert any(
                isinstance(candidate, RuntimeError) and "first cleanup" in str(candidate)
                for candidate in failures
            )
        else:
            assert sum(candidate is cleanup_failure for candidate in failures) == 1
        assert workspace_observation_pending_cancellation_requests(propagated) == 1
        assert operation.cancelling() == 0
        await app.aclose()

    asyncio.run(scenario())
