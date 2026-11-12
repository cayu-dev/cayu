"""Timed heartbeat reconciliation must not shift foreground injection points."""

import asyncio

import pytest
from tests.fault_injection.harness import (
    Injector,
    ManualClock,
    SimulatedCrash,
    task_store,
    unnumbered_task_heartbeats,
)

from cayu.runtime._durable_worker_loop import DurableWorkerWaitResult
from cayu.tasks import worker

pytestmark = pytest.mark.fault_injection


@pytest.mark.parametrize("dead", [False, True])
def test_heartbeat_reconciliation_is_unnumbered_but_still_obeys_process_death(dead):
    async def run():
        injector = Injector(armed=True)
        store = task_store(injector, ManualClock())

        async def heartbeat():
            raise RuntimeError("lease changed while renewing")

        async def reconcile(error):
            assert str(error) == "lease changed while renewing"
            # Like the worker's reconciliation, this read runs in a child task.
            await asyncio.create_task(store.load_task("missing"))
            return "reconciled"

        async def tick(seconds, stop):
            return DurableWorkerWaitResult.TIMEOUT

        injector.dead = dead
        with unnumbered_task_heartbeats():
            operation = worker.run_durable_lease_heartbeat(
                heartbeat,
                lease_seconds=300,
                stop=asyncio.Event(),
                stopped_outcome="stopped",
                on_failure=reconcile,
                wait=tick,
            )
            if dead:
                with pytest.raises(SimulatedCrash):
                    await operation
            else:
                assert await operation == "reconciled"
        assert injector.trace == []
        # The context must not leak into foreground work after the timer exits.
        injector.dead = False
        await store.load_task("missing")
        assert [b.site for b in injector.trace] == ["store.load_task:enter"]

    asyncio.run(run())
