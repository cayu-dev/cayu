"""Concurrent exact provider-failure retries must accept a peer's durable outcomes."""

import asyncio

import pytest
from tests.core.test_provider_operation_offline_recovery import (
    _OfflineOperationProvider,
    _prepare_explicit_fallback_resolution,
)

from cayu import EventType, SessionStatus
from cayu.providers.operations import ProviderOperationStatus
from cayu.runtime.provider_operations import (
    ProviderOperationResolutionAction,
    load_pending_provider_operation_disposition,
)
from cayu.sessions.memory import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["model_error", "interaction_failed"])
async def test_exact_failure_retry_accepts_outcome_committed_after_its_read(
    tmp_path, monkeypatch, backend, outcome
):
    store = (
        InMemorySessionStore()
        if backend == "memory"
        else SQLiteSessionStore(tmp_path / "concurrent-failure.sqlite")
    )
    provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
    app, fallback_request = await _prepare_explicit_fallback_resolution(
        store, session_id="concurrent-failure", provider=provider
    )
    request = fallback_request.model_copy(update={"action": ProviderOperationResolutionAction.FAIL})
    checked_absent = asyncio.Event()
    release_reader = asyncio.Event()
    delayed_task = None
    original_query = store.query_events
    paused = False

    async def query_then_pause(query):
        nonlocal paused
        records = await original_query(query)
        if (
            asyncio.current_task() is delayed_task
            and not paused
            and query.event_id is not None
            and query.event_id.endswith(":" + outcome)
            and not records
        ):
            paused = True
            checked_absent.set()
            await release_reader.wait()
        return records

    async def resolve():
        return [event async for event in app.resolve_provider_operation(request)]

    monkeypatch.setattr(store, "query_events", query_then_pause)
    try:
        delayed_task = asyncio.create_task(resolve())
        await asyncio.wait_for(checked_absent.wait(), 10)
        peer_events = await asyncio.wait_for(resolve(), 10)
        assert EventType.SESSION_FAILED in {event.type for event in peer_events}
        assert (await store.load(request.session_id)).status is SessionStatus.FAILED
        release_reader.set()
        # Both requests have identical durable authority. The slower request
        # must reconcile the winner, not fail on the now-stale absence read.
        await asyncio.wait_for(delayed_task, 10)
        durable = await store.load_events(request.session_id)
        for event_type in [
            EventType.MODEL_ERROR,
            EventType.INTERACTION_FAILED,
            EventType.SESSION_FAILED,
        ]:
            assert sum(event.type is event_type for event in durable) == 1
        assert (
            await load_pending_provider_operation_disposition(app.session_store, request.session_id)
            is None
        )
        assert provider.adapter.start_calls == 0
    finally:
        release_reader.set()
        if delayed_task is not None:
            await asyncio.gather(delayed_task, return_exceptions=True)
        await app.aclose()
        if backend == "sqlite":
            await store.close()
