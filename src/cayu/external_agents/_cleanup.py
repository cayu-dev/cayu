"""Keep cleanup failures distinct from a quiesced harness failure."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress

from cayu.external_agents.base import ExternalAgentError
from cayu.external_agents.environment import join_cleanup


class ExternalCleanupIncomplete(ExternalAgentError):
    def __init__(self, retry: Callable[[], Awaitable[None]]) -> None:
        super().__init__("External guest cleanup is not yet confirmed.")
        self.retry = retry


async def close_external_guest(close: Callable[[], Awaitable[None]]) -> None:
    try:
        await join_cleanup(close())
    except Exception:
        raise ExternalCleanupIncomplete(close) from None


async def settle_external_cleanup(
    failure: ExternalCleanupIncomplete, quiesced: asyncio.Event
) -> None:
    while not quiesced.is_set():
        try:
            await failure.retry()
            return
        except Exception:
            with suppress(TimeoutError):
                await asyncio.wait_for(quiesced.wait(), 1.0)
