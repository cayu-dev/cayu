from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.core.test_context_views import _manifest_for_store, _replace_manifest

from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.sessions.context_views import (
    ContextViewLimits,
    ContextViewOwnershipRequest,
    ContextViewSelectionRequest,
)
from cayu.storage.sqlite import SQLiteSessionStore


def _factory(backend, tmp_path, request, now):
    if backend == "memory":
        return lambda: InMemorySessionStore(ownership_clock=lambda: now[0])
    if backend == "sqlite":
        return lambda: SQLiteSessionStore(tmp_path / "views.sqlite", ownership_clock=lambda: now[0])
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    dsn = request.getfixturevalue("postgres_dsn")
    return lambda: PostgresSessionStore(
        dsn, schema_mode=SchemaMode.CREATE, clock=lambda: now[0], max_size=4
    )


async def _new_manifest(store):
    manifest = await _manifest_for_store(store, session_id=str(uuid4()))
    return _replace_manifest(manifest, view_id=f"view-{manifest.source_session_id}")


async def _close(store):
    if hasattr(store, "close"):
        await store.close()


def _selection(manifest, key, **limits):
    return ContextViewSelectionRequest(
        source_owner=manifest.source_owner,
        source_session_id=manifest.source_session_id,
        source_session_instance_id=manifest.source_session_instance_id,
        selector="latest",
        projection_schema=manifest.projection_schema,
        extension_set_commitment=manifest.extension_set_commitment,
        limits=ContextViewLimits(**limits),
        selection_key=f"{manifest.view_id}:{key}",
    )


def _ownership(receipt, key, operation="release"):
    return ContextViewOwnershipRequest(
        selection_key=receipt.selection_key,
        view_id=receipt.view.view_id,
        pin_commitment=receipt.pin_commitment,
        expected_state=receipt.state,
        expected_revision=receipt.ownership_revision,
        current_owner=receipt.owner,
        destination_owner=receipt.owner if operation != "release" else None,
        operation=operation,
        operation_key=f"{receipt.view.view_id}:{key}",
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_selection_revalidates_deleted_and_replaced_source(backend, tmp_path, request):
    factory = _factory(backend, tmp_path, request, [datetime.now(UTC)])

    async def run():
        store = factory()
        try:
            manifest = await _new_manifest(store)
            await store.publish_context_view(
                manifest, publication_key=f"{manifest.view_id}:publish"
            )
            intent = _selection(manifest, "original")
            selected = await store.select_context_view(intent)
            with pytest.raises(ValueError, match="retention pin"):
                await store.delete_session(manifest.source_session_id)
            released = await store.transition_context_view_ownership(
                _ownership(selected, "release")
            )
            await store.delete_session(manifest.source_session_id)
            if backend != "memory":
                await _close(store)
                store = factory()
            assert await store.select_context_view(intent) == released
            with pytest.raises(LookupError, match="source session incarnation"):
                await store.select_context_view(_selection(manifest, "late"))
            replacement = await store.create(
                RunRequest(agent_name="agent", session_id=manifest.source_session_id, messages=[]),
                identity=SessionIdentity(provider_name="provider", model="model"),
            )
            assert replacement.instance_id != manifest.source_session_instance_id
            with pytest.raises(LookupError, match="source session incarnation"):
                await store.select_context_view(_selection(manifest, "late"))
            # Refused stale acquisition cannot block the replacement incarnation.
            await store.delete_session(replacement.id)
            assert await store.select_context_view(intent) == released
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("expiry_entrance", ["selection", "transition"])
def test_lifecycle_quota_reserves_terminal_evidence(
    backend, expiry_entrance, tmp_path, request, monkeypatch
):
    import cayu.sessions.context_views as contracts

    monkeypatch.setattr(contracts, "CONTEXT_VIEW_MAX_LIFECYCLE_EVENTS_PER_VIEW", 4)
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    factory = _factory(backend, tmp_path, request, now)

    async def run():
        store = factory()
        try:
            manifest = await _new_manifest(store)
            await store.publish_context_view(
                manifest, publication_key=f"{manifest.view_id}:publish"
            )
            first = await store.select_context_view(_selection(manifest, "first"))
            expiring_request = _selection(manifest, "expiring", max_lifetime_seconds=1)
            expiring = await store.select_context_view(expiring_request)
            adopted = await store.transition_context_view_ownership(
                _ownership(first, "adopt", "adopt")
            )
            transferred = await store.transition_context_view_ownership(
                _ownership(adopted, "transfer", "transfer")
            )
            # Two events + two unsettled pins are exactly the ceiling.
            with pytest.raises(OverflowError, match="lifecycle evidence quota"):
                await store.select_context_view(_selection(manifest, "over-quota"))
            with pytest.raises(OverflowError, match="lifecycle evidence quota"):
                await store.transition_context_view_ownership(
                    _ownership(transferred, "over-quota-transfer", "transfer")
                )
            if backend != "memory":
                await _close(store)
                store = factory()
            now[0] += timedelta(seconds=2)
            if expiry_entrance == "transition":
                with pytest.raises(LookupError, match="expired"):
                    await store.transition_context_view_ownership(_ownership(expiring, "too-late"))
            expired = await store.select_context_view(expiring_request)
            assert expired.state == "expired"
            release = _ownership(transferred, "release")
            settled = await store.transition_context_view_ownership(release)
            assert settled.state == "released"
            assert await store.transition_context_view_ownership(release) == settled
            assert await store.select_context_view(expiring_request) == expired
            events = await store.read_context_view_lifecycle_events(manifest.view_id)
            assert len(events) == 4
            assert sum(event.state == "expired" for event in events) == 1
            # A typed model_copy is not proof of a valid operation: terminal
            # receipts cannot be released again under a fresh operation key.
            forged = release.model_copy(
                update={
                    "expected_state": "released",
                    "expected_revision": settled.ownership_revision,
                    "operation_key": f"{manifest.view_id}:forged-release",
                }
            )
            with pytest.raises(ValueError):
                await store.transition_context_view_ownership(forged)
            assert await store.read_context_view_lifecycle_events(manifest.view_id) == events
            with pytest.raises(OverflowError, match="lifecycle evidence quota"):
                await store.select_context_view(_selection(manifest, "over-quota"))
            await store.delete_session(manifest.source_session_id)
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_selection_bounds_candidates_before_manifest_hydration(
    backend, tmp_path, request, monkeypatch
):
    import cayu.sessions.context_views as contracts

    factory = _factory(backend, tmp_path, request, [datetime.now(UTC)])

    async def run():
        store = factory()
        try:
            manifest = await _new_manifest(store)
            for index in ("0", "9", "Z", "_", "a"):
                await store.publish_context_view(
                    _replace_manifest(manifest, view_id=f"{manifest.view_id}-{index}"),
                    publication_key=f"{manifest.view_id}:publication-{index}",
                )
            calls = []
            original = contracts.validate_context_view_manifest_storage

            def counted(*args, **kwargs):
                calls.append(kwargs["view_id"])
                return original(*args, **kwargs)

            monkeypatch.setattr(contracts, "validate_context_view_manifest_storage", counted)
            with pytest.raises(OverflowError, match="count exceeds"):
                await store.select_context_view(_selection(manifest, "too-many", max_views=4))
            assert calls == []
            selected = await store.select_context_view(
                _selection(manifest, "at-limit", max_views=5)
            )
            assert selected.view.view_id == f"{manifest.view_id}-a"
            assert calls == [f"{manifest.view_id}-a"]
        finally:
            await _close(store)

    asyncio.run(run())


@pytest.mark.parametrize("cancel_retry", [False, True])
def test_postgres_concurrent_identical_ownership_replays_after_row_wait(postgres_dsn, cancel_retry):
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    async def run():
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE, max_size=4)
        try:
            manifest = await _new_manifest(store)
            await store.publish_context_view(
                manifest, publication_key=f"{manifest.view_id}:publish"
            )
            selected = await store.select_context_view(_selection(manifest, "selection"))
            intent = _ownership(selected, "identical", "adopt")
            async with store._connection() as blocker, blocker.cursor() as cursor:
                await cursor.execute(
                    "SELECT selection_key FROM cayu_context_view_selections WHERE selection_key = %s FOR UPDATE",
                    (selected.selection_key,),
                )
                first = asyncio.create_task(store.transition_context_view_ownership(intent))
                second = asyncio.create_task(store.transition_context_view_ownership(intent))

                async def both_waiting():
                    while True:
                        await cursor.execute("SELECT pg_stat_clear_snapshot()")
                        await cursor.execute(
                            "SELECT COUNT(*) FROM pg_stat_activity WHERE datname = current_database() "
                            "AND pid <> pg_backend_pid() AND wait_event_type = 'Lock'"
                        )
                        if (await cursor.fetchone())[0] >= 2:
                            return
                        await asyncio.sleep(0.01)

                await asyncio.wait_for(both_waiting(), 10)
                if cancel_retry:
                    second.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await second
                    assert second.cancelling() == 1 and second.cancelled()
                await blocker.commit()
            if cancel_retry:
                second = asyncio.create_task(store.transition_context_view_ownership(intent))
            a, b = await asyncio.wait_for(asyncio.gather(first, second), 10)
            assert a == b and a.ownership_revision == 2
            assert len(await store.read_context_view_lifecycle_events(manifest.view_id)) == 1
            with pytest.raises(ValueError, match="conflicts with its request"):
                await store.transition_context_view_ownership(
                    intent.model_copy(update={"operation": "release", "destination_owner": None})
                )
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_expired_exact_retry_outside_cleanup_batch_keeps_evidence(
    backend, tmp_path, request, monkeypatch
):
    import cayu.sessions.context_views as contracts

    monkeypatch.setattr(contracts, "CONTEXT_VIEW_EXPIRY_BATCH_SIZE", 2)
    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    factory = _factory(backend, tmp_path, request, now)

    async def run():
        store = factory()
        try:
            manifest = await _new_manifest(store)
            await store.publish_context_view(
                manifest, publication_key=f"{manifest.view_id}:publish"
            )
            for key in ("a", "b", "z"):
                intent = _selection(manifest, key, max_lifetime_seconds=1)
                await store.select_context_view(intent)
            now[0] += timedelta(seconds=2)
            expired = await store.select_context_view(intent)
            assert expired.state == "expired"
            events = await store.read_context_view_lifecycle_events(manifest.view_id)
            assert len(events) == 2
            assert any(event.selection_key == expired.selection_key for event in events)
            if backend != "memory":
                await _close(store)
                store = factory()
            assert await store.select_context_view(intent) == expired
            events = await store.read_context_view_lifecycle_events(manifest.view_id)
            assert len(events) == 3
            assert len({event.selection_key for event in events}) == 3
        finally:
            await _close(store)

    asyncio.run(run())


def test_sqlite_expiry_selection_serializes_independent_deletion(tmp_path):
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    now = [datetime(2026, 1, 1, tzinfo=UTC)]
    path = tmp_path / "race.sqlite"

    async def prepare():
        store = SQLiteSessionStore(path, ownership_clock=lambda: now[0])
        try:
            manifest = await _new_manifest(store)
            await store.publish_context_view(
                manifest, publication_key=f"{manifest.view_id}:publish"
            )
            await store.select_context_view(_selection(manifest, "old", max_lifetime_seconds=1))
            return manifest
        finally:
            await store.close()

    manifest = asyncio.run(prepare())
    now[0] += timedelta(seconds=2)
    captured = Event()
    proceed = Event()

    def select_in_worker():
        async def run():
            store = SQLiteSessionStore(path, ownership_clock=lambda: now[0])

            def trace(query):
                if "SELECT view_id FROM cayu_context_views" in query:
                    captured.set()
                    assert proceed.wait(10)

            store._connection.set_trace_callback(trace)
            try:
                return await store.select_context_view(_selection(manifest, "new"))
            finally:
                await store.close()

        return asyncio.run(run())

    async def compete():
        store = SQLiteSessionStore(path, ownership_clock=lambda: now[0])
        try:
            store._connection.execute("PRAGMA busy_timeout=0")
            with ThreadPoolExecutor(max_workers=1) as workers:
                selected = workers.submit(select_in_worker)
                try:
                    assert captured.wait(10)
                    with pytest.raises(sqlite3.OperationalError, match="locked"):
                        await store.delete_session(manifest.source_session_id)
                finally:
                    proceed.set()
                assert selected.result(timeout=10).state == "selected"
            with pytest.raises(ValueError, match="retention pin"):
                await store.delete_session(manifest.source_session_id)
            events = await store.read_context_view_lifecycle_events(manifest.view_id)
            assert len(events) == 1 and events[0].state == "expired"
        finally:
            await store.close()

    asyncio.run(compete())


def test_postgres_expiry_keeps_source_serialization(postgres_dsn):
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    now = [datetime(2026, 1, 1, tzinfo=UTC)]

    async def run():
        store = PostgresSessionStore(
            postgres_dsn, schema_mode=SchemaMode.CREATE, clock=lambda: now[0], max_size=4
        )
        try:
            manifest = await _new_manifest(store)
            await store.publish_context_view(
                manifest, publication_key=f"{manifest.view_id}:publish"
            )
            old = await store.select_context_view(
                _selection(manifest, "old", max_lifetime_seconds=1)
            )
            now[0] += timedelta(seconds=2)
            original = store._require_context_view_lifecycle_capacity
            reached = asyncio.Event()
            resume = asyncio.Event()
            calls = 0

            async def gated(cur, view_id, *, additional_slots=0):
                nonlocal calls
                calls += 1
                # First check accompanies expiry; second admits the new pin.
                if calls == 2:
                    reached.set()
                    await resume.wait()
                await original(cur, view_id, additional_slots=additional_slots)

            store._require_context_view_lifecycle_capacity = gated
            acquisition = asyncio.create_task(
                store.select_context_view(_selection(manifest, "new"))
            )
            try:
                await asyncio.wait_for(reached.wait(), 10)
                async with store._connection() as conn, conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"context-view-session:{manifest.source_session_id}",),
                    )
                    assert (await cur.fetchone())[0] is False
                deletion = asyncio.create_task(store.delete_session(manifest.source_session_id))
                resume.set()
                selected = await asyncio.wait_for(acquisition, 10)
                with pytest.raises(ValueError, match="retention pin"):
                    await asyncio.wait_for(deletion, 10)
                assert selected.state == "selected"
                events = await store.read_context_view_lifecycle_events(manifest.view_id)
                assert len(events) == 1 and events[0].selection_key == old.selection_key
                assert events[0].state == "expired"
            finally:
                resume.set()
                await asyncio.gather(acquisition, return_exceptions=True)
        finally:
            await store.close()

    asyncio.run(run())
