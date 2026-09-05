"""Private handoff from completed publication to durable allocation disposal."""

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any

DisposalCheckpoint = Callable[[dict[str, Any]], Awaitable[None]]
finalization_disposal_checkpoint: ContextVar[DisposalCheckpoint | None] = ContextVar(
    "finalization_disposal_checkpoint", default=None
)


async def checkpoint_finalization_disposal(state: dict[str, Any]) -> None:
    checkpoint = finalization_disposal_checkpoint.get()
    if checkpoint is not None:
        await checkpoint(state)
