"""Replacement recovery ends lost model work with its environment still registered."""

from __future__ import annotations

import asyncio

import pytest
from tests.runtime.test_orphaned_model_work_reconciliation import (
    _application,
    _budget_settled_once,
    _unfinished,
    _worker_died_in,
)
from tests.runtime.test_orphaned_session_reconciliation import (
    _slot,
    _stores,
)
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, SessionStatus


@pytest.mark.parametrize("mode", ["compaction", "auxiliary"])
def test_lost_nested_model_work_is_settled_with_registered_environment(
    tmp_path, database_url, mode
):
    _worker_died_in(tmp_path, database_url, mode)

    async def scenario():
        stores = _stores(database_url, tmp_path / "cayu.db")
        app = _application(stores.session_store, mode, environment=_slot())
        try:
            stage = await _unfinished(stores.session_store, mode)
            await app.reconcile_orphaned_sessions()
            session = await stores.session_store.load("scan")
            assert session.status is SessionStatus.INTERRUPTED
            assert await stores.session_store.load_active_model_completion_stage("scan") is None
            await _budget_settled_once(stores, stage, conservative=True)
            events = await stores.session_store.load_events("scan")
            assert [event.type for event in events].count(EventType.SESSION_INTERRUPTED) == 1
            assert await app.reconcile_orphaned_sessions() == ()
            assert await stores.session_store.load_events("scan") == events
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())
