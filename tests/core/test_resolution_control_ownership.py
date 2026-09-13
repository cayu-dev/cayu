"""Resolution control ownership is nested, cancellable and task-specific."""

import asyncio

import pytest

from cayu.runtime._session_control import SessionControl
from cayu.runtime.sessions import InMemorySessionStore


def test_control_ownership_excludes_only_the_current_control_task():
    async def scenario():
        control = SessionControl[object](session_store=InMemorySessionStore())
        entered = asyncio.Event()
        cancellation_counts = []

        async def competing_control():
            try:
                with control.active_control_ownership("session"):
                    entered.set()
                    await asyncio.Event().wait()
            except asyncio.CancelledError:
                task = asyncio.current_task()
                assert task is not None
                cancellation_counts.append(task.cancelling())
                raise

        with control.active_control_ownership("session"):
            with control.active_control_ownership("session"):
                assert control.has_active_tasks("session")
            assert control.has_active_tasks("session")
            assert not control.has_active_tasks("session", exclude_current_control_task=True)
            competitor = asyncio.create_task(competing_control())
            try:
                await asyncio.wait_for(entered.wait(), 1)
                assert control.has_active_tasks("session", exclude_current_control_task=True)
                competitor.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await competitor
                assert competitor.cancelled() and competitor.cancelling() == 1
                assert cancellation_counts == [1]
                assert control.has_active_tasks("session")
                assert not control.has_active_tasks("session", exclude_current_control_task=True)
            finally:
                if not competitor.done():
                    competitor.cancel()
                await asyncio.gather(competitor, return_exceptions=True)
        assert not control.has_active_tasks("session")

    asyncio.run(scenario())
