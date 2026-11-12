"""Home's counts and the Sessions page stay fast on a store of deployment size.

The store holds 8,000 sessions over 60 days, as a busy deployment does, built by
``scripts/benchmark_session_overview_reads.py``. Counts must read only the
sessions that can change them, and both reads must finish well inside the time
a page can wait. The time bounds are loose enough for a busy CI runner; the read
bound is exact.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from scripts import benchmark_session_overview_reads as benchmark

from cayu.sessions.overview import SessionOverviewReader
from cayu.storage import SQLiteSessionStore

SESSIONS = 8_000
# Copies of two real runs keep the build to well under a minute.
TEMPLATES = 2
COUNTS_SECONDS = 10.0
PAGE_SECONDS = 5.0


@pytest.fixture(scope="module")
def deployment_store(tmp_path_factory):
    path = tmp_path_factory.mktemp("overview-scale") / "sessions.db"
    built = benchmark.build_store(path, sessions=SESSIONS, templates=TEMPLATES)
    assert built["sessions"] == SESSIONS + TEMPLATES + benchmark.RUNNING
    return path


def test_counts_read_only_the_window_and_running_sessions(deployment_store, monkeypatch):
    async def scenario():
        store = SQLiteSessionStore(deployment_store)
        rows = 0
        listed, loaded = store.list_sessions, store.load_sessions

        async def counted_list(query=None):
            nonlocal rows
            page = await listed(query)
            rows += len(page.sessions)
            return page

        async def counted_load(session_ids):
            nonlocal rows
            found = await loaded(session_ids)
            rows += len(found)
            return found

        monkeypatch.setattr(store, "list_sessions", counted_list)
        monkeypatch.setattr(store, "load_sessions", counted_load)
        try:
            until = datetime.now(UTC)
            counts = await SessionOverviewReader(store).counts(
                since=until - timedelta(days=1), until=until
            )
        finally:
            await store.close()
        # A day holds 1/60 of the copies plus the templates; three older runs
        # are still running, each started days before the window.
        expected = SESSIONS // benchmark.DAYS + TEMPLATES
        assert abs(counts.total_count - expected) <= 2, counts
        assert counts.running_now == benchmark.RUNNING
        assert counts.input_tokens >= counts.total_count * benchmark._USAGE["input_tokens"]
        # Each window session is listed once; older candidates are loaded once.
        assert rows <= 2 * (counts.total_count + benchmark.RUNNING) + 50, rows

    asyncio.run(scenario())


def test_counts_and_a_sessions_page_finish_in_seconds(deployment_store):
    report = benchmark.measure(deployment_store, samples=3)
    assert report["list_overview"]["rows"] == benchmark.PAGE
    assert report["counts"]["running_now"] == str(benchmark.RUNNING)
    assert report["counts"]["median_seconds"] < COUNTS_SECONDS, report
    assert report["list_overview"]["median_seconds"] < PAGE_SECONDS, report
    assert report["list_overview_first_message"]["median_seconds"] < PAGE_SECONDS, report
