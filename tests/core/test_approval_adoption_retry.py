"""A recoverable continuation failure must not strand an adopted approval."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_tool_approval_profile_adoption import (
    _app,
    _approve,
    _collect,
    _FileBug,
    _pause,
    _use_cayu,
)

from cayu.events import EventType
from cayu.sessions.memory import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore


class _StopOnceBeforeDispatch:
    """Lose one event append before it commits, so no tool was dispatched."""

    invocation_lifecycle_command_version = 1
    stop_before_dispatch = False
    injected = False

    async def append_events(self, session_id, events):
        if self.stop_before_dispatch and any(
            event.type is EventType.TOOL_CALL_STARTED for event in events
        ):
            self.stop_before_dispatch = False
            self.injected = True
            raise ConnectionError("temporary event append failure before dispatch")
        return await super().append_events(session_id, events)


class _MemoryStore(_StopOnceBeforeDispatch, InMemorySessionStore):
    invocation_lifecycle_command_version = 1


class _SQLiteStore(_StopOnceBeforeDispatch, SQLiteSessionStore):
    invocation_lifecycle_command_version = 1


@pytest.mark.parametrize("upgrade", [False, True])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_interrupted_approval_remains_retryable_after_runtime_adoption(
    upgrade, backend, tmp_path, monkeypatch
):
    async def scenario():
        store = _MemoryStore() if backend == "memory" else _SQLiteStore(tmp_path / "s.db")
        _use_cayu(monkeypatch, "0.15.0")
        original = _app(store)
        resumed = None
        try:
            requested = await _pause(original)
            await original.aclose()
            if upgrade:
                _use_cayu(monkeypatch, "0.16.0")
            tool = _FileBug()
            resumed = _app(store, tool=tool)
            resolution = _approve(requested)
            store.stop_before_dispatch = True
            first = await _collect(resumed.resolve_tool_approval(resolution))
            assert store.injected
            assert first[-1].type is EventType.SESSION_INTERRUPTED
            assert tool.calls == []
            if upgrade:
                assert any(
                    event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
                    and event.payload["decision"] == "adopted"
                    for event in first
                )
            # The one-shot storage error is gone. The same accepted approval
            # should now run exactly once under the already-adopted runtime.
            retried = await _collect(resumed.resolve_tool_approval(resolution))
            assert retried[-1].type is EventType.SESSION_COMPLETED
            assert tool.calls == [{"title": "Crash on save"}]
        finally:
            if resumed is not None:
                await resumed.aclose()
            await original.aclose()
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
