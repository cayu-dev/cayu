"""Qualification observation cannot hide a worker failure or renew admission."""

import asyncio

import pytest

from cayu.deadlines import ExecutionDeadline
from tests.qualification.repository_maintenance_worker_case import wait_for_publication


@pytest.mark.parametrize(
    "outcome", ["barrier", "failure", "return", "expired", "cancel", "worker-cancel"]
)
def test_publication_wait_preserves_owner_and_original_deadline(outcome):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        failure = ConnectionError("retained worker failure")

        async def run():
            if outcome == "failure":
                raise failure
            if outcome == "return":
                return 1
            if outcome == "barrier":
                entered.set()
            await release.wait()
            return 1

        worker = asyncio.create_task(run())
        deadline = ExecutionDeadline.after(0 if outcome == "expired" else 10)
        original = deadline.model_dump()
        observer = asyncio.create_task(wait_for_publication(entered, worker, deadline=deadline))
        try:
            if outcome == "failure":
                with pytest.raises(ConnectionError) as caught:
                    await observer
                assert caught.value is failure
            elif outcome == "return":
                with pytest.raises(AssertionError, match="exited before"):
                    await observer
            elif outcome == "expired":
                with pytest.raises(AssertionError, match="original deadline"):
                    await observer
                assert not worker.done() and worker.cancelling() == 0
            elif outcome == "cancel":
                await asyncio.sleep(0)
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert observer.cancelled() and observer.cancelling() == 1
                assert not worker.done() and worker.cancelling() == 0
            elif outcome == "worker-cancel":
                await asyncio.sleep(0)
                worker.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert worker.cancelled() and worker.cancelling() == 1
                assert observer.cancelled() and observer.cancelling() == 0
            else:
                await observer
                assert not worker.done() and worker.cancelling() == 0
            assert deadline.model_dump() == original
        finally:
            release.set()
            await asyncio.gather(worker, observer, return_exceptions=True)
        assert not (asyncio.all_tasks() - {asyncio.current_task()})

    asyncio.run(scenario())
