"""A server restart with no undelivered event side effects decodes no stored events.

The startup sweep and the background recovery loop find pending deliveries by
the side-effect table's delivery-state index and load only the events those
rows point to, so the history a store keeps does not slow down a restart.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

import cayu.events as events_module
import cayu.server as server_module
from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelRequest, ModelStreamEvent
from cayu.server import OpenAccess, mount_cayu
from cayu.sessions.event_delivery import PersistedEventSideEffectStatus
from cayu.storage import SQLiteSessionStore
from cayu.storage import _sqlite_records as sqlite_records

if TYPE_CHECKING:
    import pytest


def _respond(request: ModelRequest) -> list[ModelStreamEvent]:
    del request
    return [
        ModelStreamEvent.text_delta("Done."),
        ModelStreamEvent.completed(
            {"finish_reason": "stop", "usage": {"input_tokens": 10, "output_tokens": 2}}
        ),
    ]


def _app(store: SQLiteSessionStore) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider(name="scripted", response_factory=_respond))
    app.register_agent(AgentSpec(name="writer", model="scripted-model"))
    return app


async def _write_history(path: Any, sessions: int) -> None:
    store = SQLiteSessionStore(path)
    try:
        app = _app(store)
        for index in range(sessions):
            async for _event in app.run(
                RunRequest(
                    agent_name="writer",
                    session_id=f"history-{index}",
                    messages=[Message.text("user", "Write a line.")],
                )
            ):
                pass
        await app.aclose()
    finally:
        await store.close()


def test_restart_with_nothing_pending_decodes_no_events(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sessions.db"
    asyncio.run(_write_history(path, sessions=3))

    decodes: list[str] = []
    validations: list[str] = []
    decode = sqlite_records.event_from_row
    validate = events_module._validate_event_candidate

    def counting_decode(row: Any) -> Any:
        decodes.append(row["event_type"])
        return decode(row)

    def counting_validate(event: Any) -> Any:
        validations.append(str(event.type))
        return validate(event)

    async def restart() -> None:
        store = SQLiteSessionStore(path)
        try:
            delivered = await store.list_persisted_event_side_effect_deliveries(
                statuses={PersistedEventSideEffectStatus.DELIVERED}, limit=1000
            )
            assert len(delivered) > 10
            pending = await store.list_persisted_event_side_effect_deliveries(
                claimable_only=True, limit=1
            )
            assert pending == []
            app = _app(store)
            server = FastAPI()
            mount_cayu(server, app, access=OpenAccess(), dashboard=False)
            monkeypatch.setattr(sqlite_records, "event_from_row", counting_decode)
            monkeypatch.setattr(events_module, "_validate_event_candidate", counting_validate)
            async with server.router.lifespan_context(server):
                # One sweep of the background loop, which otherwise waits 30 s.
                await server_module._recover_persisted_event_side_effects_until_idle(app)
        finally:
            await store.close()

    asyncio.run(restart())
    assert decodes == []
    assert validations == []
