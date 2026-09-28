"""Explicit local host lifetime over the common durable-worker cadence.

Only native host adapters call this private scheduler. It owns observation, not
business decisions or foreign-effect settlement. No task starts at construction.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from cayu.collaboration._host_ownership import _seconds
from cayu.runtime._durable_worker_loop import DurableWorkerStep, run_durable_worker_loop


@dataclass(frozen=True, slots=True)
class HostPassOutcome:
    progressed: int
    coverage_complete: bool

    def __post_init__(self) -> None:
        if type(self.progressed) is not int or not 0 <= self.progressed <= 128:
            raise ValueError("Host pass progress exceeds its finite operation bound.")
        if type(self.coverage_complete) is not bool:
            raise TypeError("Host coverage must be explicit.")


@dataclass(frozen=True, slots=True)
class HostPassObservation:
    outcome: HostPassOutcome | None
    pending: bool
    closing: bool


class HostLifecycle:
    """One retained service pass and one public observer per host.

    The adapter checks stop/deadline before every new dispatch. It supervises
    admitted execution separately; completing a pass is not execution settlement.
    A timed-out or cancelled observer cannot abandon the still-running pass.
    """

    def __init__(
        self,
        step: Callable[[float, asyncio.Event], Awaitable[HostPassOutcome]],
        *,
        observation_timeout_s: float,
        poll_interval_s: float,
    ) -> None:
        _seconds(observation_timeout_s)
        _seconds(poll_interval_s)
        if not callable(step):
            raise TypeError("Host lifecycle requires a native servicing adapter.")
        self._step = step
        self._timeout = observation_timeout_s
        self._poll_interval = poll_interval_s
        self._stop = asyncio.Event()
        self._pass: asyncio.Task[HostPassOutcome] | None = None
        self._observer: asyncio.Task | None = None
        self._observer_done = asyncio.Event()
        self._observer_done.set()
        self._close_observer = False
        self._closing = False

    @property
    def closing(self) -> bool:
        return self._closing

    def inspect(self) -> HostPassObservation:
        # Inspection never consumes an exception or exposes private diagnostics.
        return HostPassObservation(None, self._pass is not None, self._closing)

    def _enter(self) -> None:
        if self._closing:
            raise RuntimeError("Collaboration host is closing.")
        if self._observer is not None:
            raise RuntimeError("Collaboration host servicing is already active.")
        self._observer = asyncio.current_task()
        self._observer_done.clear()

    async def _observe(self) -> HostPassObservation:
        if self._pass is None:
            if self._closing:
                return self.inspect()
            deadline = asyncio.get_running_loop().time() + self._timeout

            async def execute():
                await asyncio.sleep(0)  # Install ownership before eager factories enter adapters.
                result = await self._step(deadline, self._stop)
                if type(result) is not HostPassOutcome:
                    raise TypeError("Native host adapter returned an invalid pass result.")
                return result

            coroutine = execute()
            try:
                self._pass = asyncio.create_task(coroutine, name="cayu-collaboration-host-pass")
            except BaseException:
                coroutine.close()
                raise
        task = self._pass
        await asyncio.wait((task,), timeout=self._timeout)
        if not task.done():
            return self.inspect()
        # Consume only a definite task outcome. Original control signals remain
        # signals; they are not converted into ordinary blocked-work reports.
        self._pass = None
        return HostPassObservation(task.result(), False, self._closing)

    async def service_once(self) -> HostPassObservation:
        self._enter()
        try:
            return await self._observe()
        finally:
            self._observer = None
            self._observer_done.set()

    async def run(self) -> None:
        self._enter()
        try:

            async def step(_now, _handled):
                observed = await self._observe()
                progressed = 0 if observed.outcome is None else observed.outcome.progressed
                return DurableWorkerStep(
                    handled=progressed, activity=bool(progressed), idle=True, stop=self._closing
                )

            await run_durable_worker_loop(
                step, poll_interval_s=self._poll_interval, stop=self._stop
            )
        finally:
            self._observer = None
            self._observer_done.set()

    def request_close(self) -> None:
        self._closing = True
        self._stop.set()

    async def aclose(self, *, timeout_s: float) -> HostPassObservation:
        _seconds(timeout_s)
        if self._observer is asyncio.current_task():
            raise RuntimeError("Host servicing cannot join its own observer.")
        if self._close_observer:
            raise RuntimeError("Collaboration host close observation is already active.")
        self._close_observer = True
        try:
            return await self._close_once(timeout_s)
        finally:
            self._close_observer = False

    async def _close_once(self, timeout_s: float) -> HostPassObservation:
        self.request_close()
        deadline = asyncio.get_running_loop().time() + timeout_s
        if self._observer is not None:
            # Join servicing, not unrelated work subsequently done by its caller.
            try:
                async with asyncio.timeout(timeout_s):
                    await self._observer_done.wait()
            except TimeoutError:
                return self.inspect()
        task = self._pass
        if task is None:
            return self.inspect()
        remaining = max(0, deadline - asyncio.get_running_loop().time())
        await asyncio.wait((task,), timeout=remaining)
        if not task.done():
            return self.inspect()
        if self._pass is task:
            self._pass = None
        return HostPassObservation(task.result(), False, True)
