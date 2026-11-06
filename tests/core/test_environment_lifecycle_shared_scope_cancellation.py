"""Ending a session must also end the shared allocation scope it owns."""

import asyncio

import pytest
from tests.environments.test_shared_allocations import _app

from cayu import EventType, Message, RunRequest


@pytest.mark.parametrize("cancel_diagnostic", [False, True])
def test_cancelled_terminal_diagnostic_still_releases_owned_shared_scope(
    tmp_path, monkeypatch, cancel_diagnostic
):
    async def scenario():
        app, factory = _app(tmp_path, turns=1, allocation_scope="parent")
        registry = app._environment_lifecycle.shared_allocations
        emit = app._event_writer.emit
        publication_started = asyncio.Event()

        async def block_diagnostic(event):
            if cancel_diagnostic and event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED:
                publication_started.set()
                await asyncio.Event().wait()
            return await emit(event)

        monkeypatch.setattr(app._event_writer, "emit", block_diagnostic)

        async def collect():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="planner",
                        session_id="scope-owner",
                        messages=[Message.text("user", "run")],
                    )
                )
            ]

        run = asyncio.create_task(collect())
        try:
            if cancel_diagnostic:
                await asyncio.wait_for(publication_started.wait(), timeout=10)
                run.cancel("caller stopped diagnostic publication")
                with pytest.raises(asyncio.CancelledError):
                    await run
            else:
                assert (await run)[-1].type is EventType.SESSION_COMPLETED
            assert await app.drain_environment_cleanups(timeout_s=2)
            assert not registry.live_keys(), (
                "The ended session's detached shared allocation was never released",
                factory.finalizations(),
            )
        finally:
            if not run.done():
                run.cancel()
                await asyncio.gather(run, return_exceptions=True)
            await app.close_idle_environment_resources()

    asyncio.run(scenario())
