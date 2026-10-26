"""`cayu serve` logs one storage-retention hint at startup for a grown store."""

import asyncio
import logging
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from tests.core.session_retention_conformance import (
    RetentionClock,
    RetentionHarness,
    create_retention_session,
)

from cayu import CayuApp
from cayu.server import ServerConfig, create_server
from cayu.storage import SQLiteSessionStore

_HINT = "Storage retention is off by default"


@pytest.mark.parametrize("age_days", [0, 31])
def test_server_startup_logs_retention_hint_only_for_a_grown_store(
    tmp_path, caplog, age_days
) -> None:
    clock = RetentionClock()
    store = SQLiteSessionStore(tmp_path / "sessions.db", ownership_clock=clock)
    app = CayuApp(session_store=store, enable_logging=False)
    asyncio.run(create_retention_session(RetentionHarness(store=store, clock=clock), "old"))
    clock.now += timedelta(days=age_days)

    async def wait_for_report() -> None:
        for _ in range(200):
            task = next(
                (
                    task
                    for task in asyncio.all_tasks()
                    if task.get_name() == "cayu-storage-growth-report"
                ),
                None,
            )
            if task is None or task.done():
                return
            await asyncio.sleep(0.01)

    server = create_server(app, config=ServerConfig.local_development())
    with caplog.at_level(logging.WARNING, logger="cayu.server"), TestClient(server) as client:
        client.portal.call(wait_for_report)
    hints = [record.getMessage() for record in caplog.records if _HINT in record.getMessage()]
    assert len(hints) == (1 if age_days else 0)
