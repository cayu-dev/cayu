"""Peer operations share queue state while retaining replay and authorization."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import timedelta
from functools import partial
from types import SimpleNamespace

import pytest
from tests.core.test_memory_queued_message_ownership import (
    NOW,
    _assert_import_without_stores,
    _Composition,
    _request,
    _session,
)
from tests.core.test_native_peer_content_ownership import _exercise
from tests.core.test_native_peer_content_ownership import _request as _peer_request

from cayu.sessions import _memory_peer_content as peer

OPERATIONS = (
    "append_peer_content",
    "read_peer_content_attempt",
    "list_pending_peer_content",
    "read_peer_content",
    "record_peer_content_exposure",
    "begin_peer_content_exposure",
    "read_peer_content_exposure",
    "exclude_peer_content",
    "retry_pending_peer_content",
)


def _bindings(source, target):
    return {
        session.id: SimpleNamespace(
            session_instance_id=session.instance_id,
            participant=SimpleNamespace(participant_id=name, incarnation=name + "-v1"),
        )
        for session, name in ((source, "sender"), (target, "consumer"))
    }


class _PeerComposition:
    def __init__(self):
        self.queue = _Composition()
        q = self.queue.state
        source = _session("source")
        q.sessions[source.id] = source
        target = q.sessions["target"]
        self.request = _peer_request([source, target])
        self.state = peer.MemoryPeerContentState(
            append_events=q.append_events,
            checkpoints=q.checkpoints,
            ownership_clock=q.ownership_clock,
            participant_session_bindings=_bindings(source, target),
            peer_content_attempts={},
            peer_content_deleted_targets=set(),
            peer_content_exposures={},
            peer_content_operation_commitments={},
            peer_content_operations={},
            peer_content_receipts={},
            peer_content_requests={},
            pending_session_messages=q.pending_session_messages,
            queued_session_messages_by_idempotency=q.queued_session_messages_by_idempotency,
            session_creation_decisions={},
            ordering=q.ordering,
            session_operation_records=q.session_operation_records,
            sessions=q.sessions,
            transcripts=q.transcripts,
        )
        self.ops = SimpleNamespace(
            **{name: partial(getattr(peer, name), self.locked) for name in OPERATIONS}
        )

    @asynccontextmanager
    async def locked(self):
        async with self.queue.lock:
            yield self.state

    def snapshot(self):
        return deepcopy(tuple(value for value in self.state if not callable(value)))


def test_memory_peer_imports_without_stores():
    _assert_import_without_stores("cayu.sessions._memory_peer_content")


def test_memory_peer_composes_with_retries_replay_and_exposure():
    async def scenario():
        harness = _PeerComposition()
        await _exercise(harness.ops, harness.request, harness.queue.assert_idle)
        assert harness.state.ordering.next_key == 2
        assert len(harness.state.pending_session_messages) == 1

    asyncio.run(scenario())


def test_memory_peer_and_ordinary_admission_share_global_queue_ordering():
    async def scenario():
        harness = _PeerComposition()
        q = harness.queue
        first = await q.ops.enqueue_session_message(_request("first"))
        appended, last = await asyncio.gather(
            harness.ops.append_peer_content(harness.request, qualify_target=lambda session: None),
            q.ops.enqueue_session_message(_request("last")),
        )
        queued_peer = q.state.queued_session_messages_by_idempotency["target"][
            harness.request.operation_key
        ]
        assert appended.status == "appended"
        assert first.message.ordering_key == 1
        assert {queued_peer.ordering_key, last.message.ordering_key} == {2, 3}
        assert q.state.ordering.next_key == 4
        # Exact peer replay cannot allocate another queue key.
        assert (
            await harness.ops.append_peer_content(
                harness.request, qualify_target=lambda session: None
            )
        ).replayed
        assert q.state.ordering.next_key == 4
        batch = await q.ops.deliver_queued_session_messages(
            "target", include_on_idle=True, delivery_id="boundary", eligible_through=1
        )
        assert [message.ordering_key for message in batch.messages] == [1]
        assert batch.eligible_through == 1
        remaining = await q.ops.deliver_queued_session_messages(
            "target", include_on_idle=True, delivery_id="remaining"
        )
        assert {message.ordering_key for message in remaining.messages} == {2, 3}
        assert remaining.eligible_through == 3
        assert len(q.indexes.messages["target"]) == 3

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["qualification", "publication"])
def test_memory_peer_failure_preserves_queue_receipts_and_ordering(failure):
    async def scenario():
        harness = _PeerComposition()
        before = harness.snapshot(), harness.queue.snapshot()

        def qualify(session):
            assert harness.queue.lock.locked()
            assert session.id == "target"
            if failure == "qualification":
                raise RuntimeError("qualification rejected")

        harness.queue.fail_append = failure == "publication"
        with pytest.raises(RuntimeError, match="rejected"):
            await harness.ops.append_peer_content(harness.request, qualify_target=qualify)
        assert (harness.snapshot(), harness.queue.snapshot()) == before
        harness.queue.assert_idle()

    asyncio.run(scenario())


def test_memory_peer_cancelled_lock_wait_preserves_admission_state():
    async def scenario():
        harness = _PeerComposition()
        before = harness.snapshot(), harness.queue.snapshot()
        async with harness.queue.lock:
            task = asyncio.create_task(
                harness.ops.append_peer_content(
                    harness.request, qualify_target=lambda session: None
                )
            )
            await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert (harness.snapshot(), harness.queue.snapshot()) == before
        assert (
            await harness.ops.append_peer_content(
                harness.request, qualify_target=lambda session: None
            )
        ).status == "appended"
        assert harness.state.ordering.next_key == 2

    asyncio.run(scenario())


def test_memory_peer_adapter_resolves_clock_and_indexes_after_lock_wait():
    async def scenario():
        from cayu.sessions.memory import InMemorySessionStore
        from cayu.sessions.records import SessionIdentity
        from cayu.sessions.requests import RunRequest

        store = InMemorySessionStore(ownership_clock=lambda: NOW)
        sessions = [
            await store.create(
                RunRequest(session_id=sid, agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            for sid in ("source", "target")
        ]
        store._participant_session_bindings = _bindings(*sessions)
        request = _peer_request(sessions)
        async with store._lock:
            task = asyncio.create_task(store.append_peer_content(request))
            await asyncio.sleep(0)
            assert not task.done()
            old = store._peer_content_receipts
            store._peer_content_receipts = {}
            store._ownership_clock = lambda: NOW + timedelta(days=36500)
        result = await task
        assert result.status == "excluded" and result.reason == "delivery_deadline_expired"
        assert old == {}
        assert store._peer_content_receipts[request.append_key.model_dump_json()] == result
        assert store._queued_session_messages_by_idempotency == {}
        assert store._session_message_ordering.next_key == 1

    asyncio.run(scenario())
