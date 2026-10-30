"""Observe an owned CLI transport without confusing cleanup with a natural exit."""

from __future__ import annotations

import asyncio
import subprocess
from typing import Any

from cayu.external_agents.base import ExternalAgentError, ExternalAgentEvent

_EXIT_GRACE_SECONDS = 0.25


async def natural_exit(process: Any, *, synchronous: bool) -> bool:
    """Allow a bounded natural exit before the existing cleanup takes ownership."""
    if process is None:
        return False
    if process.returncode is not None:
        return True
    try:
        if synchronous:
            await asyncio.to_thread(process.wait, timeout=_EXIT_GRACE_SECONDS)
        else:
            await asyncio.wait_for(process.wait(), timeout=_EXIT_GRACE_SECONDS)
    except (TimeoutError, subprocess.TimeoutExpired):
        return process.returncode is not None
    return True


def exit_observation(process: Any, exited: bool) -> ExternalAgentEvent:
    code = None if process is None else process.returncode
    return ExternalAgentEvent(
        "process_exit",
        {
            "exit_code": code if type(code) is int else None,
            "disposition": "exited" if exited else "closed_by_cayu",
        },
        "final:process-exit",
    )


def require_successful_exit(event: ExternalAgentEvent, *, enabled: bool) -> None:
    if enabled and event.payload["disposition"] == "exited" and event.payload["exit_code"] != 0:
        raise ExternalAgentError("External CLI exited nonzero after returning its result.")
