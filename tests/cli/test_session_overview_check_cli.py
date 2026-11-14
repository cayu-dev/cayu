"""``cayu session overview`` checks stored summary rows against the event log."""

from __future__ import annotations

import asyncio
import json
import sqlite3

from tests.core.test_cloud_reporting import _app, _start

from cayu.cli import main
from cayu.storage.sqlite import SQLiteSessionStore


def _store(path) -> None:
    async def scenario():
        store = SQLiteSessionStore(path)
        try:
            app = _app(store)
            await _start(app, "asking", "asker")
            await _start(app, "done", "writer")
        finally:
            await store.close()

    asyncio.run(scenario())


def test_recompute_reports_a_matching_summary(tmp_path, capsys) -> None:
    database = tmp_path / "store.db"
    _store(database)
    assert main(["session", "overview", "asking", "--recompute", "--sqlite", str(database)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["matches"] is True
    assert payload["recomputed"]["state"] == "waiting_input"
    assert payload["stored"]["state"] == "waiting_input"
    assert payload["row"]["agent_name"] == "asker"
    assert payload["row"]["source_generation"] > 0

    # The CLI opens the store read-only, so nothing settles; every row is still
    # compared, none reported as changing.
    assert main(["session", "overview", "--all", "--sqlite", str(database)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert (payload["checked"], payload["mismatched"], payload["changing"]) == (2, [], [])


def test_recompute_fails_on_a_stored_summary_that_disagrees(tmp_path, capsys) -> None:
    database = tmp_path / "store.db"
    _store(database)

    async def settle():
        store = SQLiteSessionStore(database)
        try:
            from cayu.server._session_overview import uses_browser
            from cayu.server._session_summaries import SessionSummaries

            await SessionSummaries(store, uses_browser=uses_browser).derive("done")
        finally:
            await store.close()

    asyncio.run(settle())
    # Corrupt the stored document without advancing its generation.
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE cayu_session_summaries SET derived_json = "
            "json_set(derived_json, '$.state', 'failed') WHERE session_id = 'done'"
        )
    assert main(["session", "overview", "done", "--recompute", "--sqlite", str(database)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["current"] is True
    assert (payload["stored"]["state"], payload["recomputed"]["state"]) == ("failed", "completed")
    assert main(["session", "overview", "--all", "--sqlite", str(database)]) == 1
    assert json.loads(capsys.readouterr().out)["mismatched"] == ["done"]
