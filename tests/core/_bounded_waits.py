"""Bounded waits for tests that drive a caller task through a fake's steps.

A test that waits on an event only a fake sets hangs until the CI job times out
when the caller fails before it reaches the fake, and hides that failure. These
helpers bound each wait and surface the caller's own error instead.
"""

from __future__ import annotations

import asyncio
from typing import Any

# Upper bound for one step of a fake scenario. The fakes answer immediately, so
# reaching this means the scenario is stuck.
FAKE_STEP_TIMEOUT_SECONDS = 10.0


async def wait_for_fake_step(event: asyncio.Event, caller: asyncio.Task[Any]) -> None:
    """Wait for a fake's step, failing fast if the caller ends first."""

    step = asyncio.ensure_future(event.wait())
    try:
        done, _ = await asyncio.wait(
            {step, caller},
            timeout=FAKE_STEP_TIMEOUT_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        step.cancel()
    if step in done:
        return
    if caller in done:
        if not caller.cancelled():
            caller.result()
        raise AssertionError("The caller finished before the fake step ran.")
    raise AssertionError(f"The fake step did not run within {FAKE_STEP_TIMEOUT_SECONDS} seconds.")


async def wait_for_caller_settlement(caller: asyncio.Task[Any]) -> None:
    """Bound the wait for the caller to finish without cancelling it again."""

    await asyncio.wait({caller}, timeout=FAKE_STEP_TIMEOUT_SECONDS)
    if not caller.done():
        raise AssertionError(
            f"The caller did not settle within {FAKE_STEP_TIMEOUT_SECONDS} seconds."
        )
