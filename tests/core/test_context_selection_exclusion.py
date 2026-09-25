"""Native arbitration characterization; planner acceptance lives in its public suite."""

import asyncio
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from tests.core.test_context_views import _manifest_for_store, _replace_manifest
from tests.core.test_prepared_admission_public import native_stores as native_stores

from cayu.collaboration._contracts import ExactConflict, ExactMatch, ExactNotFound, ExactUnavailable
from cayu.sessions._context_selection_fence import (
    _CONTEXT_SELECTION_AUTHORITY,
    ContextViewSelectionConflict,
    ContextViewSelectionExcluded,
    require_selection_fence_store,
)
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.context_views import ContextViewLimits, ContextViewSelectionRequest

pytestmark = pytest.mark.anyio


async def scenario(store=None):
    store = store or InMemorySessionStore()
    identity = uuid4().hex
    view = _replace_manifest(
        await _manifest_for_store(store, session_id=identity), view_id=identity
    )
    owner = view.source_owner.model_copy(update={"application_scope": identity})
    view = _replace_manifest(
        view,
        source_owner=owner.model_dump(mode="json"),
        participant=view.participant.model_copy(update={"owner": owner}).model_dump(mode="json"),
    )
    await store.publish_context_view(view, publication_key=identity)
    request = ContextViewSelectionRequest(
        source_owner=view.source_owner,
        source_session_id=view.source_session_id,
        source_session_instance_id=view.source_session_instance_id,
        selector="latest",
        projection_schema=view.projection_schema,
        extension_set_commitment=view.extension_set_commitment,
        limits=ContextViewLimits(),
        selection_key=identity,
    )
    return store, request


@pytest.mark.parametrize("first", ["selection", "exclusion"])
async def test_native_selection_exclusion_orders_and_exact_replay(native_stores, first):
    store, request = await scenario(native_stores[1])
    assert isinstance(await store.read_context_view_selection_decision(request), ExactNotFound)
    if first == "selection":
        selected = await store.select_context_view(request)
    result = await store._exclude_context_view_selection(
        request, authority=_CONTEXT_SELECTION_AUTHORITY
    )
    assert result.state == ("selected" if first == "selection" else "excluded")
    assert (
        await store._exclude_context_view_selection(request, authority=_CONTEXT_SELECTION_AUTHORITY)
        == result
    )
    found = await store.read_context_view_selection_decision(request)
    assert isinstance(found, ExactMatch) and found.receipt == result
    if first == "selection":
        assert await store.select_context_view(request) == selected
        # Exclusion never invents release of a positively retained pin.
        assert (
            await store.lookup_context_view_selection(request.selection_key)
        ).state == "selected"
    else:
        with pytest.raises(ContextViewSelectionExcluded):
            await store.select_context_view(request)
        assert await store.lookup_context_view_selection(request.selection_key) is None


@pytest.mark.parametrize("first", ["selection", "exclusion"])
@pytest.mark.parametrize(
    "field",
    [
        "source_owner",
        "source_session_id",
        "source_session_instance_id",
        "selector",
        "minimum_transcript_cursor",
        "projection_schema",
        "extension_set_commitment",
        "limits",
    ],
)
async def test_fixed_key_changes_conflict_without_selection(native_stores, first, field):
    store, request = await scenario(native_stores[1])
    if first == "selection":
        await store.select_context_view(request)
    decision = await store._exclude_context_view_selection(
        request, authority=_CONTEXT_SELECTION_AUTHORITY
    )
    changed = {
        "source_owner": request.source_owner.model_copy(update={"incarnation": "other"}),
        "source_session_id": "other",
        "source_session_instance_id": "other",
        "selector": "exact",
        "minimum_transcript_cursor": 2,
        "projection_schema": "other",
        "extension_set_commitment": "other",
        "limits": request.limits.model_copy(update={"max_pins": 1}),
    }[field]
    update = {field: changed}
    if field == "selector":
        update["exact_view_id"] = "view-1"
    conflicting = ContextViewSelectionRequest.model_validate(request.model_copy(update=update))
    before = await store.lookup_context_view_selection(request.selection_key)
    assert isinstance(await store.read_context_view_selection_decision(conflicting), ExactConflict)
    with pytest.raises(ContextViewSelectionConflict):
        await store._exclude_context_view_selection(
            conflicting, authority=_CONTEXT_SELECTION_AUTHORITY
        )
    with pytest.raises(ValueError):
        await store.select_context_view(conflicting)
    assert before == await store.lookup_context_view_selection(request.selection_key)
    assert (await store.read_context_view_selection_decision(request)).receipt == decision


async def test_raw_cleanup_value_cannot_exclude(native_stores):
    store, request = await scenario(native_stores[1])
    require_selection_fence_store(store)
    with pytest.raises(PermissionError, match="trusted receiving owner"):
        await store._exclude_context_view_selection(request, authority=object())
    assert isinstance(await store.read_context_view_selection_decision(request), ExactNotFound)
    assert (await store.select_context_view(request)).state == "selected"


async def test_divergent_index_is_unavailable_not_exclusion(native_stores):
    store, request = await scenario(native_stores[1])
    original = await store._exclude_context_view_selection(
        request, authority=_CONTEXT_SELECTION_AUTHORITY
    )
    wrong = original.model_copy(
        update={"request": request.model_copy(update={"selection_key": "wrong-index"})}
    )
    backend = native_stores[3][0]
    if backend == "memory":
        store._context_selection_exclusions[request.selection_key] = wrong
    elif backend == "sqlite":
        async with store._context_view_transaction():
            store._connection.execute(
                "UPDATE cayu_context_selection_exclusions SET decision_json = ? WHERE selection_key = ?",
                (wrong.model_dump_json(), request.selection_key),
            )
    else:
        async with store._connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                "UPDATE cayu_context_selection_exclusions SET decision_json = %s WHERE selection_key = %s",
                (wrong.model_dump_json(), request.selection_key),
            )
    assert isinstance(await store.read_context_view_selection_decision(request), ExactUnavailable)
    with pytest.raises(ValueError):
        await store._exclude_context_view_selection(request, authority=_CONTEXT_SELECTION_AUTHORITY)


def test_forwarding_wrapper_cannot_inherit_native_qualification():
    class Unqualified(InMemorySessionStore):
        pass

    class Qualified(InMemorySessionStore):
        context_view_selection_fence_version = 1

    with pytest.raises(NotImplementedError):
        require_selection_fence_store(Unqualified())
    require_selection_fence_store(Qualified())


@pytest.mark.parametrize("kind", ["selection", "exclusion", "reservation"])
async def test_lost_receiving_ack_reconciles_exact_decision(native_stores, tmp_path, kind):
    store, request = await scenario(native_stores[1])

    async def commit_then_lose_ack():
        if kind == "selection":
            await store.select_context_view(request)
        elif kind == "reservation":
            await store._reserve_context_view_selection_control(
                request, authority=_CONTEXT_SELECTION_AUTHORITY
            )
        else:
            await store._exclude_context_view_selection(
                request, authority=_CONTEXT_SELECTION_AUTHORITY
            )
        raise OSError("receiving acknowledgement lost")

    with pytest.raises(OSError, match="acknowledgement lost"):
        await commit_then_lose_ack()
    owner = store
    if native_stores[3][0] != "memory":
        await store.close()
        owner = reopened_store(native_stores, tmp_path)
    try:
        evidence = await owner.read_context_view_selection_decision(request)
        assert isinstance(evidence, ExactMatch)
        expected = {"selection": "selected", "exclusion": "excluded", "reservation": "reserved"}[
            kind
        ]
        assert evidence.receipt.state == expected
        if kind == "reservation":
            assert (
                await owner._reserve_context_view_selection_control(
                    request, authority=_CONTEXT_SELECTION_AUTHORITY
                )
                == evidence.receipt
            )
            # A recovered reservation is not exclusion and has not acquired a
            # view. Only the native cleanup mutation can settle it negatively.
            assert await owner.lookup_context_view_selection(request.selection_key) is None
            excluded = await owner._exclude_context_view_selection(
                request, authority=_CONTEXT_SELECTION_AUTHORITY
            )
            assert excluded.state == "excluded"
            with pytest.raises(ContextViewSelectionExcluded):
                await owner.select_context_view(request)
            return
        assert (
            await owner._exclude_context_view_selection(
                request, authority=_CONTEXT_SELECTION_AUTHORITY
            )
            == evidence.receipt
        )
    finally:
        if owner is not store:
            await owner.close()


async def test_cancellation_of_waiting_observer_is_not_exclusion():
    store, request = await scenario()
    await store._lock.acquire()
    pending = asyncio.create_task(store.select_context_view(request))
    try:
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert pending.cancelled() and pending.cancelling() == 1
    finally:
        store._lock.release()
    assert isinstance(await store.read_context_view_selection_decision(request), ExactNotFound)
    # Only a subsequent actual native exclusion can fence a delayed retry.
    result = await store._exclude_context_view_selection(
        request, authority=_CONTEXT_SELECTION_AUTHORITY
    )
    assert result.state == "excluded"
    with pytest.raises(ContextViewSelectionExcluded):
        await store.select_context_view(request)


async def test_reserved_cleanup_survives_full_optional_and_control_capacity(
    native_stores, monkeypatch
):
    from cayu.sessions import _context_selection_fence as native_fence
    from cayu.sessions import context_views
    from cayu.storage import _context_selection_fence as sql_fence

    # Narrow the actual native hard ceilings to exercise both complete pools.
    monkeypatch.setattr(native_fence, "CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER", 2)
    monkeypatch.setattr(sql_fence, "CONTEXT_SELECTION_MAX_CONTROLS_PER_OWNER", 2)
    monkeypatch.setattr(context_views, "CONTEXT_VIEW_MAX_SELECTIONS_PER_OWNER", 1)
    store, request = await scenario(native_stores[1])
    second = request.model_copy(update={"selection_key": request.selection_key + ":2"})
    third = request.model_copy(update={"selection_key": request.selection_key + ":3"})
    first_reserve = await store._reserve_context_view_selection_control(
        request, authority=_CONTEXT_SELECTION_AUTHORITY
    )
    assert first_reserve.state == "reserved"
    assert (await store.read_context_view_selection_decision(request)).receipt == first_reserve
    assert await store.lookup_context_view_selection(request.selection_key) is None
    assert (
        await store._reserve_context_view_selection_control(
            request, authority=_CONTEXT_SELECTION_AUTHORITY
        )
        == first_reserve
    )
    assert (
        await store._reserve_context_view_selection_control(
            second, authority=_CONTEXT_SELECTION_AUTHORITY
        )
    ).state == "reserved"
    with pytest.raises(OverflowError, match="control quota"):
        await store._reserve_context_view_selection_control(
            third, authority=_CONTEXT_SELECTION_AUTHORITY
        )
    assert isinstance(await store.read_context_view_selection_decision(third), ExactNotFound)
    selected = await store.select_context_view(request)
    with pytest.raises(OverflowError, match="selection quota"):
        await store.select_context_view(second)
    # Full ordinary capacity did not consume the pending operation's control
    # slot; full control capacity cannot refuse its already-reserved settlement.
    excluded = await store._exclude_context_view_selection(
        second, authority=_CONTEXT_SELECTION_AUTHORITY
    )
    assert excluded.state == "excluded"
    assert (
        await store._reserve_context_view_selection_control(
            second, authority=_CONTEXT_SELECTION_AUTHORITY
        )
        == excluded
    )
    with pytest.raises(ContextViewSelectionExcluded):
        await store.select_context_view(second)
    positive = await store._exclude_context_view_selection(
        request, authority=_CONTEXT_SELECTION_AUTHORITY
    )
    assert positive.state == "selected" and positive.view_id == selected.view.view_id


@pytest.mark.parametrize("first", ["selection", "exclusion"])
async def test_competing_native_workers_share_the_same_selection_lock(first):
    store, request = await scenario()
    await store._lock.acquire()
    operations = {
        "selection": lambda: store.select_context_view(request),
        "exclusion": lambda: store._exclude_context_view_selection(
            request, authority=_CONTEXT_SELECTION_AUTHORITY
        ),
    }
    second = "exclusion" if first == "selection" else "selection"
    tasks = []
    try:
        tasks.append(asyncio.create_task(operations[first]()))
        await asyncio.sleep(0)
        tasks.append(asyncio.create_task(operations[second]()))
        await asyncio.sleep(0)
        assert all(not task.done() for task in tasks)
    finally:
        store._lock.release()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    if first == "exclusion":
        assert results[0].state == "excluded"
        assert isinstance(results[1], ContextViewSelectionExcluded)
    else:
        assert results[0].state == results[1].state == "selected"
        assert results[0].view.view_id == results[1].view_id


def reopened_store(native_stores, tmp_path):
    backend, address = native_stores[3]
    if backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        return SQLiteSessionStore(tmp_path / "sessions.sqlite")
    from cayu.storage.postgres import PostgresSessionStore

    return PostgresSessionStore(address)


@pytest.mark.parametrize("native_stores", ["sqlite", "postgres"], indirect=True)
@pytest.mark.parametrize("first", ["selection", "exclusion"])
async def test_persistent_two_connection_race_and_restart(
    native_stores, tmp_path, monkeypatch, first
):
    store, request = await scenario(native_stores[1])
    backend = native_stores[3][0]
    entered, release, competing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    if backend == "sqlite":
        original = store._context_view_transaction

        @asynccontextmanager
        async def hold_transaction():
            async with original():
                entered.set()
                await release.wait()
                yield

        monkeypatch.setattr(store, "_context_view_transaction", hold_transaction)
    else:
        original = store._lock_context_view_admission

        async def hold_admission(*args, **kwargs):
            await original(*args, **kwargs)
            entered.set()
            await release.wait()

        monkeypatch.setattr(store, "_lock_context_view_admission", hold_admission)

    async def run(owner, kind):
        if kind == "selection":
            return await owner.select_context_view(request)
        return await owner._exclude_context_view_selection(
            request, authority=_CONTEXT_SELECTION_AUTHORITY
        )

    loop = asyncio.get_running_loop()

    async def competing_operation():
        owner = reopened_store(native_stores, tmp_path)
        try:
            loop.call_soon_threadsafe(competing.set)
            return await run(owner, "exclusion" if first == "selection" else "selection")
        finally:
            await owner.close()

    # A separate event loop also prevents SQLite's synchronous BEGIN from
    # blocking the owner that must release the first physical transaction.
    pending = asyncio.create_task(run(store, first))
    contender = None
    try:
        await asyncio.wait_for(entered.wait(), 20)
        contender = asyncio.create_task(
            asyncio.to_thread(lambda: asyncio.run(competing_operation()))
        )
        await asyncio.wait_for(competing.wait(), 20)
        assert not pending.done() and not contender.done()
    finally:
        release.set()
    results = await asyncio.gather(pending, contender, return_exceptions=True)
    if first == "selection":
        assert results[0].state == results[1].state == "selected"
        assert results[0].view.view_id == results[1].view_id
    else:
        assert results[0].state == "excluded"
        assert isinstance(results[1], ContextViewSelectionExcluded)

    # Close the creating connection and reconstruct durable exact evidence.
    await store.close()
    restored = reopened_store(native_stores, tmp_path)
    try:
        found = await restored.read_context_view_selection_decision(request)
        assert isinstance(found, ExactMatch)
        assert found.receipt.state == ("selected" if first == "selection" else "excluded")
        assert (
            await restored._exclude_context_view_selection(
                request, authority=_CONTEXT_SELECTION_AUTHORITY
            )
            == found.receipt
        )
    finally:
        await restored.close()
