"""Bounded retained source reads, independent of the current service observer.

Only native read adapters use this owner. Results are observations, never claims
or renewed authority; the receiving operation repeats its normal checks.
"""

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HostReadResult:
    value: Any = None
    error: BaseException | None = None


def _completed_result(task):
    # Cancellation before the coroutine's first entry cannot be caught inside
    # it. Preserve that actual signal as an outcome too, so draining the read
    # neither strands its reservation nor prevents observing sibling failures.
    try:
        return task.result()
    except BaseException as error:
        return HostReadResult(error=error)


class HostReads:
    def __init__(self, *, slots, bytes_limit):
        if type(slots) is not int or not 1 <= slots <= 32:
            raise ValueError("Host discovery slots must be between one and 32.")
        if type(bytes_limit) is not int or not 65536 <= bytes_limit <= 4 * 1024 * 1024:
            raise ValueError("Host discovery bytes exceed their supported bounds.")
        self._slots = slots
        self._bytes_limit = bytes_limit
        self._tasks = {}
        self._closing = False

    @property
    def pending(self):
        return len(self._tasks)

    async def observe(self, key, *, expectation, reserved_bytes, read):
        if type(key) is not str or not key or len(key.encode("utf-8")) > 256:
            raise ValueError("Host read key must be bounded.")
        if type(expectation) is not bytes or len(expectation) > 65536:
            raise ValueError("Host read expectation must be bounded bytes.")
        if type(reserved_bytes) is not int or not 1 <= reserved_bytes <= 131072:
            raise ValueError("Host read reservation exceeds its bound.")
        existing = self._tasks.get(key)
        if existing is not None:
            wanted, size, task = existing
            if wanted != expectation or size != reserved_bytes:
                raise ValueError("Pending host read conflicts with its exact query.")
        else:
            if self._closing:
                return None
            used = sum(entry[1] for entry in self._tasks.values())
            if len(self._tasks) >= self._slots or used + reserved_bytes > self._bytes_limit:
                return None

            async def owned():
                try:
                    await asyncio.sleep(0)
                    return HostReadResult(value=await read())
                except BaseException as error:
                    return HostReadResult(error=error)

            coroutine = owned()
            try:
                task = asyncio.create_task(coroutine, name="cayu-collaboration-host-read")
            except BaseException:
                coroutine.close()
                raise
            self._tasks[key] = (expectation, reserved_bytes, task)
        # Give ready local reads a small observation window; neither its expiry
        # nor caller cancellation cancels, replaces, or detaches the read task.
        await asyncio.wait((task,), timeout=0.001)
        if not task.done():
            return None
        result = _completed_result(task)
        del self._tasks[key]
        if result.error is not None:
            raise result.error
        return result

    def request_close(self):
        self._closing = True

    async def close(self, timeout):
        self.request_close()
        if self._tasks and timeout > 0:
            await asyncio.wait(tuple(entry[2] for entry in self._tasks.values()), timeout=timeout)
        errors = []
        for key, (_expected, _size, task) in tuple(self._tasks.items()):
            if not task.done():
                continue
            result = _completed_result(task)
            del self._tasks[key]
            if result.error is not None:
                errors.append(result.error)
        return tuple(errors)
