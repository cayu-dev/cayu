"""Retained crash-leftover teardown is part of environment drain's inventory."""

from __future__ import annotations

import asyncio

from tests.core.test_deferred_environments import DeferredLocalFactory, _app, _turns

from cayu import Message, RunRequest, run_to_completion
from cayu.environments.factory import attach_environment_factory_cleanup_settlement_task
from cayu.sessions import InMemorySessionStore


def test_environment_drain_waits_for_kept_crash_teardown(tmp_path):
    async def scenario():
        allow_teardown = asyncio.Event()
        teardown = None

        class Factory(DeferredLocalFactory):
            async def release_deferred_materialization(self, request):
                nonlocal teardown

                async def finish_removal():
                    await allow_teardown.wait()

                teardown = asyncio.create_task(finish_removal())
                error = ConnectionError("container removal acknowledgement was lost")
                attach_environment_factory_cleanup_settlement_task(error, teardown)
                raise error

        (tmp_path / "source").mkdir()
        (tmp_path / "target").mkdir()
        store = InMemorySessionStore()
        factory = Factory(tmp_path / "source", tmp_path / "target")
        app = _app(factory, store, _turns())
        try:
            outcome = await run_to_completion(
                app,
                RunRequest(
                    agent_name="agent",
                    session_id="kept-crash-teardown-drain",
                    messages=[Message.text("user", "go")],
                ),
            )
            assert outcome.ok
            session = await store.load(outcome.session_id)
            assert session is not None
            # The same best-effort entrance is called by incomplete-session
            # recovery after it settles a crashed deferred run.
            released = await app._environment_lifecycle.release_deferred_materialization(
                session=session,
                registered_agent=app._get_registered_agent(session.agent_name),
                registered_environment=app._get_registered_environment_for_session(
                    session.environment_name
                ),
            )
            assert not released
            assert teardown is not None and not teardown.done()
            # Verify this is retained by the lifecycle, rather than an unowned
            # task invented outside the code being tested.
            kept = app._environment_lifecycle._leftover_teardown._tasks
            assert teardown in kept[(session.id, session.environment_name)]
            drained = await app.drain_environment_cleanups(timeout_s=0.05)
            assert not teardown.done()
            assert drained is False, (
                "Environment drain reported quiescence with retained crash teardown still running"
            )
        finally:
            allow_teardown.set()
            if teardown is not None:
                await teardown
            assert await app.drain_environment_cleanups(timeout_s=2)
            await app.close_idle_environment_resources()

    asyncio.run(scenario())
