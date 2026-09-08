"""Exercise review claims through the same store matrix as durable operations."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_human_review import CONTEXT, ReviewPolicy, decision, make_app, pause, resolve
from tests.core.test_session_store_shared_conformance import (
    _close_store,
    _open_store,
    conformance_postgres_dsn,  # noqa: F401 - imported pytest fixture
    session_store_case,  # noqa: F401 - imported pytest fixture
)


@pytest.mark.parametrize("approval", [False, True])
def test_review_store_atomic_content_binding(session_store_case, approval):  # noqa: F811
    async def scenario():
        store = await _open_store(session_store_case)
        try:
            app, provider, tool = make_app(store, ReviewPolicy(), approval=approval)
            await pause(app)
            view = await app.inspect_human_review("review-session", context=CONTEXT)
            assert view.status == "permitted"

            def change(_session, checkpoint):
                key = "pending_tool_approval" if approval else "pending_user_input"
                pending = checkpoint[key]
                pending["arguments"]["unexpected"] = "changed"
                pending["tool_calls"][0]["arguments"]["unexpected"] = "changed"
                if approval:
                    checkpoint["pending_tool_round"]["tool_calls"][0]["arguments"]["unexpected"] = (
                        "changed"
                    )
                return checkpoint

            await store.transform_checkpoint("review-session", change)
            with pytest.raises((ValueError, RuntimeError)):
                await resolve(app, decision(view, approval=approval))
            assert len(provider.requests) == 1
            if approval:
                assert not tool.calls
        finally:
            await _close_store(store)

    asyncio.run(scenario())
