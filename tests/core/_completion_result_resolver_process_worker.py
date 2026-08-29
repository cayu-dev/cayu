from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from tests.core.test_completion_decision_application import _contract, _result

from cayu import (
    CayuApp,
    CompletionResultResolutionRequest,
    CompletionResultResolver,
    CompletionResultResolverRequest,
    Event,
    EventType,
    SQLiteSessionStore,
    SQLiteTaskStore,
)
from cayu.runtime.sessions import CheckpointTransform, Session

_PROCESS_LOSS_EXIT_CODE = 86


class _CrashBeforeResultEventStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1
    supports_completion_result_event_publication_reservations = True

    async def _publish_completion_result_event_publication(
        self,
        session_id: str,
        *,
        checkpoint_transform: CheckpointTransform,
        events: list[Event],
    ) -> Session:
        if any(event.type is EventType.TASK_COMPLETION_RESULT_RESOLVED for event in events):
            os._exit(_PROCESS_LOSS_EXIT_CODE)
        return await super()._publish_completion_result_event_publication(
            session_id,
            checkpoint_transform=checkpoint_transform,
            events=events,
        )


class _MarkerResolver(CompletionResultResolver):
    def __init__(self, marker: Path) -> None:
        self._marker = marker

    async def resolve(
        self,
        request: CompletionResultResolverRequest,
    ) -> dict[str, object]:
        del request
        if self._marker.exists():
            raise AssertionError("result resolver was invoked more than once")
        self._marker.write_text("resolved\n", encoding="utf-8")
        return _result("1")


async def _run() -> None:
    session_store = _CrashBeforeResultEventStore(Path(sys.argv[1]))
    task_store = SQLiteTaskStore(Path(sys.argv[2]))
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_completion_result_resolver(
        _contract().result_resolver,
        _MarkerResolver(Path(sys.argv[3])),
    )
    await app.resolve_completion_result(
        CompletionResultResolutionRequest(
            task_id="application-task",
            decision_id=sys.argv[4],
            idempotency_key="resolve-result-1",
        )
    )
    raise AssertionError("process-loss boundary was not reached")


if __name__ == "__main__":
    asyncio.run(_run())
