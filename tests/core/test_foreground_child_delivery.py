"""Real cancellation and lease retention for the foreground wakeup owner."""

import asyncio

import pytest

from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._foreground_child_delivery import ForegroundChildDeliveryOwner
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cancelled_delivery_retains_renewal_until_runtime_cleanup_settles(tmp_path, backend):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "delivery.sqlite")
        )
        owner = ForegroundChildDeliveryOwner(store)
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()
        caught = []
        try:
            session = await store.create(
                RunRequest(
                    session_id="child", agent_name="child", messages=[Message.text("user", "go")]
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            event = Event(type=EventType.SESSION_COMPLETED, session_id=session.id)
            await store.append_event(session.id, event)
            claim = await store.claim_persisted_event_side_effect(
                session_id=session.id, event_id=event.id, lease_seconds=1.5
            )
            assert claim is not None

            async def operation(before_mutation):
                await before_mutation()
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    assert task is not None and task.cancelling() == 1
                    cancelled.set()
                    await release.wait()
                    raise

            async def caller():
                try:
                    await owner.run(claim, operation)
                except asyncio.CancelledError:
                    caught.append("ordinary cancellation")
                    raise

            task = asyncio.create_task(caller())
            try:
                await asyncio.wait_for(entered.wait(), 10)
                task.cancel("stop foreground wait")
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(task), 2)
                assert task.cancelled() and task.cancelling() == 1
                assert caught == ["ordinary cancellation"]
                await asyncio.wait_for(cancelled.wait(), 10)
                assert owner.active(session.id)
                assert not await owner.drain(timeout_s=0.01)
                # Stay blocked beyond the original lease, using real elapsed
                # time; renewal must prevent another worker from claiming it.
                await asyncio.sleep(1.6)
                assert (
                    await store.claim_persisted_event_side_effect(
                        session_id=session.id, event_id=event.id
                    )
                    is None
                )
                current = await store.get_persisted_event_side_effect_delivery(
                    session_id=session.id, event_id=event.id
                )
                assert current is not None and current.lease_expires_at is not None
                assert current.claim_id == claim.claim_id
                assert current.lease_expires_at > claim.lease_expires_at
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                assert await owner.drain(timeout_s=10)
            assert not owner.active(session.id)
            await store.defer_persisted_event_side_effect(claim)
            replacement = await store.claim_persisted_event_side_effect(
                session_id=session.id, event_id=event.id
            )
            assert replacement is not None and replacement.claim_id != claim.claim_id
        finally:
            release.set()
            await owner.drain(timeout_s=10)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


def test_delivery_retains_renewal_failure_and_runtime_cleanup_failure(monkeypatch):
    async def scenario():
        store = InMemorySessionStore()
        owner = ForegroundChildDeliveryOwner(store)
        entered = asyncio.Event()
        renewal_failure = ConnectionError("renewal acknowledgement lost")
        cleanup_failure = RuntimeError("runtime cleanup failed")
        session = await store.create(
            RunRequest(
                session_id="child", agent_name="child", messages=[Message.text("user", "go")]
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        event = Event(type=EventType.SESSION_COMPLETED, session_id=session.id)
        await store.append_event(session.id, event)
        claim = await store.claim_persisted_event_side_effect(
            session_id=session.id, event_id=event.id, lease_seconds=1.5
        )
        assert claim is not None
        renew = store.renew_persisted_event_side_effect

        async def lose_acknowledgement(*args, **kwargs):
            result = await renew(*args, **kwargs)
            if entered.is_set():
                raise renewal_failure
            return result

        monkeypatch.setattr(store, "renew_persisted_event_side_effect", lose_acknowledgement)

        async def operation(before_mutation):
            await before_mutation()
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise cleanup_failure from None

        with pytest.raises(ExceptionGroup) as caught:
            await asyncio.wait_for(owner.run(claim, operation), 10)
        assert caught.value.exceptions == (renewal_failure, cleanup_failure)
        assert await owner.drain(timeout_s=10)
        assert not owner.active(session.id)

    asyncio.run(scenario())
