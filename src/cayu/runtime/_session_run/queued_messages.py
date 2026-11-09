"""Queued message delivery results for a session run."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal

from cayu.events import Event
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.sessions.records import Session


@dataclass(frozen=True, slots=True)
class _QueuedCompletionResult:
    outcome: Literal["continue", "complete", "stopped"]
    session: Session
    events: list[Event]
    invocation_context: InvocationContext | None
    completion_finalization_marker: dict[str, Any] | None = None
    continuation: AsyncGenerator[Event, None] | None = None
