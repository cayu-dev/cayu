"""An explicitly adopted approval must use its adopted run-budget authority."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_tool_approval_profile_adoption import (
    _AdoptExplicitRequests,
    _app,
    _approve,
    _collect,
    _durable_store,
    _FileBug,
    _intent,
)
from tests.runtime.runtime_budget_support import fake_budget_limit

from cayu.events import EventType
from cayu.messages import Message
from cayu.sessions.requests import RunRequest


async def _exercise_adopted_budget(scope, backend, tmp_path):
    store = _durable_store(backend, tmp_path, None)
    tool = _FileBug()
    policy = _AdoptExplicitRequests()
    app = _app(store, tool=tool, profile_policy=policy)
    try:
        paused = await _collect(
            app.run(
                RunRequest(
                    agent_name="reporter",
                    session_id="bug-report",
                    messages=[Message.text("user", "Save crashes the editor.")],
                    budget_limits=(fake_budget_limit("1", scope=scope, allow_unpriced=True),),
                )
            )
        )
        requested = next(
            event.payload
            for event in paused
            if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        resolution = _approve(
            requested,
            profile_adoption=_intent("adopt-new-budget"),
            budget_limits=(fake_budget_limit("2", scope=scope, allow_unpriced=True),),
        )
        continued = await _collect(app.resolve_tool_approval(resolution))
        assert any(
            event.type is EventType.SESSION_EXECUTION_PROFILE_DECIDED
            and event.payload["decision"] == "adopted"
            for event in continued
        )
        assert continued[-1].type is EventType.SESSION_COMPLETED, {
            "terminal": continued[-1].type.value,
            "error": continued[-1].payload.get("error"),
            "error_type": continued[-1].payload.get("error_type"),
            "tool_calls": tool.calls,
        }
        assert tool.calls == [{"title": "Crash on save"}]
    finally:
        await app.aclose()
        if backend == "sqlite":
            await store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("scope", ["session", "run"])
def test_approval_can_continue_with_an_explicitly_adopted_budget(scope, backend, tmp_path):
    asyncio.run(_exercise_adopted_budget(scope, backend, tmp_path))
