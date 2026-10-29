"""Interruption operations compose independently of the session engine."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_recovery_composition import _run_without_owners

from cayu import AgentSpec, CayuApp, EventType, RunRequest
from cayu._exception_groups import iter_exception_tree
from cayu.sessions import InterruptSessionRequest
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
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


def test_session_interruption_imports_without_application_or_engine() -> None:
    _run_without_owners(
        ("cayu.applications", "cayu.runtime._session_engine"),
        "from cayu.runtime._session_interruption import SessionInterruption\n",
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_interruption_and_replay_work_without_engine(
    backend, sqlite_resources, monkeypatch, request
) -> None:
    async def scenario() -> None:
        async with sqlite_resources as resources:
            if backend == "postgres":
                from cayu.storage.migrations import SchemaMode
                from cayu.storage.postgres import PostgresSessionStore

                store = PostgresSessionStore(
                    request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
                )
            elif backend == "sqlite":
                store = resources.own(SQLiteSessionStore(resources.path()))
            else:
                store = InMemorySessionStore()
            app = CayuApp(session_store=store, enable_logging=False)
            try:
                app.register_agent(AgentSpec(name="assistant", model="fake"))
                session = await store.create(
                    RunRequest(agent_name="assistant", messages=[]),
                    identity=SessionIdentity(provider_name="fake", model="fake"),
                )
                monkeypatch.delattr(app, "_session_engine")
                stop = InterruptSessionRequest(session_id=session.id, reason="operator stop")
                first = [event async for event in app.interrupt_session(stop)]
                # Settle the cascade the interrupt started; replays must not reopen it.
                assert await app.drain_background_interruptions()
                app.session_store = InMemorySessionStore()
                direct_replay = [
                    event async for event in app._session_interruption.interrupt_session(stop)
                ]
                assert await app.session_store.load(session.id) is None
                # Public event projection deliberately uses the current public store.
                app.session_store = store
                replay = [event async for event in app.interrupt_session(stop)]
                assert [event.type for event in first] == [EventType.SESSION_INTERRUPTED]
                assert [event.id for event in replay] == [event.id for event in first]
                assert first[0].payload["reason"] == "operator stop"
                persisted = await store.load(session.id)
                assert persisted is not None
                assert persisted.status is SessionStatus.INTERRUPTED
                records = await store.query_events(
                    EventQuery(session_id=session.id, event_type=EventType.SESSION_INTERRUPTED)
                )
                assert len(records) == 1
                assert records[0].event.id == direct_replay[0].id
                # The interruption's child cascade runs in the background and can still be
                # running on a slow machine; wait for it before reading its settled status.
                assert await app._background_interruption_coordinator.drain(timeout_s=10)
                assert await app.interruption_cascade_status(session.id) == "none"
            finally:
                await app.aclose()
                if backend == "postgres":
                    await store.close()

    asyncio.run(scenario())


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
