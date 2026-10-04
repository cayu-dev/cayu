"""WarmSparePool on its own, with an in-memory backend."""

from __future__ import annotations

import asyncio
import os
import warnings

import pytest

from cayu.environments import WarmSpareBackend, WarmSparePool
from cayu.environments.admission import ExecutionRequirements
from cayu.environments.warm_spares import WarmSpareRequirementsUnsatisfied


class _Spare:
    def __init__(self, name: str) -> None:
        self.name = name
        self.alive = True
        self.discarded = False


class MemoryBackend:
    def __init__(self) -> None:
        self.spares: dict[str, _Spare] = {}
        self.claim_error: Exception | None = None
        self.claims = 0
        self.discarded: list[str] = []
        self.block_alive: asyncio.Event | None = None
        self.block_claim: asyncio.Event | None = None
        self.block_discard: asyncio.Event | None = None
        self.alive_error: Exception | None = None

    async def create_spare(self, name: str) -> _Spare:
        spare = _Spare(name)
        self.spares[name] = spare
        return spare

    async def spare_is_alive(self, spare: _Spare) -> bool:
        if self.block_alive is not None:
            await self.block_alive.wait()
        if self.alive_error is not None:
            raise self.alive_error
        return spare.alive

    async def claim_spare(self, spare: _Spare, name: str, requirements) -> _Spare:
        self.claims += 1
        if self.block_claim is not None:
            await self.block_claim.wait()
        if self.claim_error is not None:
            raise self.claim_error
        del self.spares[spare.name]
        spare.name = name
        return spare

    async def discard_spare(self, spare: _Spare) -> None:
        if self.block_discard is not None:
            await self.block_discard.wait()
        spare.discarded = True
        self.discarded.append(spare.name)
        self.spares.pop(spare.name, None)

    async def list_spare_names(self, prefix: str) -> list[str]:
        return [name for name in self.spares if name.startswith(prefix)]

    async def remove_spare_named(self, name: str) -> None:
        self.spares.pop(name, None)


def test_a_custom_backend_composes_with_the_pool() -> None:
    backend: WarmSpareBackend[_Spare] = MemoryBackend()
    requirements = ExecutionRequirements.trusted()

    async def run():
        pool = WarmSparePool(backend, size=2, name_prefix="memory-spare-host-")
        pool.schedule_refill()
        await pool.wait_for_refill()
        idle = pool.idle_count
        first = await pool.take("session-a", requirements)
        backend.claim_error = RuntimeError("claim refused")
        refused = await pool.take("session-b", requirements)
        backend.claim_error = None
        await pool.wait_for_refill()
        await pool.close()
        return idle, first, refused, pool

    idle, first, refused, pool = asyncio.run(run())

    assert idle == 2
    assert first is not None and first.name == "session-a"
    assert refused is None  # a refused claim discards that spare; the caller cold-starts
    assert pool.closed and pool.idle_count == 0
    assert backend.spares == {}  # every idle spare was removed at close
    assert not first.discarded  # the claimed resource belongs to its consumer now


def _executables(*names: str) -> ExecutionRequirements:
    return ExecutionRequirements.model_validate(
        {
            **ExecutionRequirements.trusted().model_dump(mode="python", warnings=False),
            "required_executables": names,
        }
    )


async def _filled(backend: MemoryBackend, size: int) -> WarmSparePool:
    pool = WarmSparePool(backend, size=size, name_prefix="memory-spare-host-")
    pool.schedule_refill()
    await pool.wait_for_refill()
    return pool


def test_an_unsatisfiable_claim_costs_one_spare_and_later_claims_go_cold() -> None:
    backend = MemoryBackend()

    async def run():
        pool = await _filled(backend, 3)
        backend.claim_error = WarmSpareRequirementsUnsatisfied("image lacks git")
        first = await pool.take("session-a", _executables("git"))
        claims_after_first = backend.claims
        second = await pool.take("session-b", _executables("git"))
        backend.claim_error = None
        other = await pool.take("session-c", _executables("sh"))
        await pool.close()
        return first, claims_after_first, second, other

    first, claims_after_first, second, other = asyncio.run(run())

    assert first is None and second is None
    assert claims_after_first == 1  # the loop stopped after one failed claim
    assert backend.claims == 2  # the remembered requirements never reached a spare
    assert backend.discarded[0].startswith("memory-spare-host-")
    assert other is not None and other.name == "session-c"


def test_an_ordinary_claim_failure_stops_without_burning_the_pool() -> None:
    backend = MemoryBackend()

    async def run():
        pool = await _filled(backend, 3)
        backend.claim_error = RuntimeError("rename failed")
        refused = await pool.take("session-a", _executables("git"))
        idle = pool.idle_count
        backend.claim_error = None
        retried = await pool.take("session-b", _executables("git"))
        await pool.close()
        return refused, idle, retried

    refused, idle, retried = asyncio.run(run())

    assert refused is None
    assert idle == 2  # one spare discarded, the rest kept
    assert retried is not None  # not remembered as unsatisfiable


def test_a_dead_spare_makes_take_try_the_next_one() -> None:
    backend = MemoryBackend()

    async def run():
        pool = await _filled(backend, 2)
        pool._spares[0].alive = False
        return await pool.take("session-a", _executables("sh"))

    claimed = asyncio.run(run())

    assert claimed is not None and claimed.name == "session-a"
    assert len(backend.discarded) == 1


@pytest.mark.parametrize("stage", ["alive", "claim"])
def test_cancelling_take_never_leaks_the_popped_spare(stage: str) -> None:
    backend = MemoryBackend()

    async def run():
        pool = await _filled(backend, 1)
        blocker = asyncio.Event()
        if stage == "alive":
            backend.block_alive = blocker
        else:
            backend.block_claim = blocker
        task = asyncio.create_task(pool.take("session-a", _executables("sh")))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pool.idle_count

    idle = asyncio.run(run())

    if stage == "alive":
        assert idle == 1 and not backend.discarded  # returned to the pool
    else:
        assert idle == 0 and len(backend.discarded) == 1  # possibly renamed: removed


def test_a_cancellation_during_the_discard_after_a_failed_claim_propagates() -> None:
    backend = MemoryBackend()

    async def run():
        pool = await _filled(backend, 1)
        backend.claim_error = RuntimeError("rename failed")
        backend.block_discard = asyncio.Event()
        task = asyncio.create_task(pool.take("session-a", _executables("sh")))
        await asyncio.sleep(0.01)  # the claim failed; the discard is in flight
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task  # never a None that would send the caller to a cold start
        backend.block_discard.set()
        await asyncio.sleep(0.01)
        return list(backend.discarded)

    assert len(asyncio.run(run())) == 1  # the shielded discard still completed


@pytest.mark.parametrize("interruption", ["cancelled", "raised"])
def test_a_spare_checked_while_the_pool_closes_is_discarded(interruption: str) -> None:
    backend = MemoryBackend()

    async def run():
        pool = await _filled(backend, 1)
        backend.block_alive = asyncio.Event()
        task = asyncio.create_task(pool.take("session-a", _executables("sh")))
        await asyncio.sleep(0.01)
        await pool.close()  # the popped spare is not in the pool for close to see
        if interruption == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            backend.alive_error = RuntimeError("inspect failed")
            backend.block_alive.set()
            assert await task is None
        return pool.idle_count

    assert asyncio.run(run()) == 0
    assert len(backend.discarded) == 1


class ExitAwareBackend(MemoryBackend):
    def __init__(self) -> None:
        super().__init__()
        self.removed_at_exit: list[str] = []

    def remove_spares_at_exit(self, spares) -> None:
        self.removed_at_exit.extend(spare.name for spare in spares)


def test_idle_spares_are_removed_at_interpreter_exit_when_never_closed() -> None:
    backend = ExitAwareBackend()

    async def fill():
        return await _filled(backend, 2)

    pool = asyncio.run(fill())
    pool._exit_hook()  # what atexit runs for a pool nobody drained or closed

    assert sorted(backend.removed_at_exit) == sorted(backend.spares)
    assert pool.idle_count == 0


def test_a_closed_pool_has_nothing_left_for_its_exit_hook() -> None:
    backend = ExitAwareBackend()

    async def fill_and_close():
        pool = await _filled(backend, 1)
        await pool.close()
        return pool

    pool = asyncio.run(fill_and_close())
    pool._exit_hook()

    assert backend.removed_at_exit == []


def test_the_exit_hook_ignores_a_pool_owned_by_another_process() -> None:
    backend = ExitAwareBackend()

    async def fill():
        return await _filled(backend, 1)

    pool = asyncio.run(fill())
    pool._owner_pid = -1  # as if this process were a fork without at-fork support
    pool._exit_hook()

    assert backend.removed_at_exit == []
    assert pool.idle_count == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_never_hands_out_or_removes_the_parent_s_spares() -> None:
    backend = ExitAwareBackend()

    async def fill():
        return await _filled(backend, 2)

    pool = asyncio.run(fill())
    parent_spares = {spare.name for spare in pool._spares}
    read_end, write_end = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # fork with threads
        pid = os.fork()
    if pid == 0:  # child
        os.close(read_end)
        try:
            idle = pool.idle_count
            taken = asyncio.run(pool.take("child-session", _executables("sh")))
            pool._exit_hook()  # may remove spares the child made, never the parent's
            touched = parent_spares & {*backend.removed_at_exit, *backend.discarded}
            report = f"{idle} {taken is None} {len(touched)}"
        except BaseException as error:
            report = f"error {error!r}"
        os.write(write_end, report.encode())
        os._exit(0)
    os.close(write_end)
    with os.fdopen(read_end) as child_report:
        report = child_report.read()
    os.waitpid(pid, 0)

    assert report == "0 True 0"
    # The parent still owns both spares and its exit hook still removes them.
    assert pool.idle_count == 2
    pool._exit_hook()
    assert len(backend.removed_at_exit) == 2


@pytest.mark.parametrize("meanwhile", ["trimmed", "refilled"])
def test_an_interrupted_check_never_puts_a_spare_back_after_trim_or_beyond_size(
    meanwhile: str,
) -> None:
    backend = MemoryBackend()

    async def run():
        pool = await _filled(backend, 1)
        backend.block_alive = asyncio.Event()
        task = asyncio.create_task(pool.take("session-a", _executables("sh")))
        await asyncio.sleep(0.01)
        if meanwhile == "trimmed":
            await pool.release_idle()
        else:
            pool.schedule_refill()  # the popped spare left room; refill fills it
            await pool.wait_for_refill()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pool.idle_count

    idle = asyncio.run(run())

    assert idle == (0 if meanwhile == "trimmed" else 1)  # never above size 1
    assert len(backend.discarded) == 1  # the interrupted spare was removed
