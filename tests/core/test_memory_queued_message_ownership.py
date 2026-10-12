"""Queue operations compose with caller-owned state and atomic publication."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.core.test_native_session_message_ownership import _exercise

import cayu
from cayu.sessions import _memory_queued_messages as queue
from cayu.sessions import _memory_transcripts as transcripts
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation
from cayu.sessions.messaging import EnqueueSessionMessageRequest, SessionMessageQuery
from cayu.sessions.records import EventRecord, Session, SessionStatus

NOW = datetime(2026, 1, 1, tzinfo=UTC)
OPERATIONS = (
    "inspect_session_messages",
    "apply_session_message_action",
    "snapshot_session_message_source",
    "enqueue_session_message",
    "deliver_queued_session_messages",
    "repair_queued_interaction_profile_handoff",
)


def _session(sid):
    return Session(
        id=sid,
        agent_name="assistant",
        provider_name="fake",
        model="fake",
        status=SessionStatus.RUNNING,
        invocation=SessionInvocation(
            origin=InvocationOrigin(trust="unattributed"),
            root_invocation_id="11111111-1111-4111-8111-111111111111",
            root_session_id=sid,
            source="sdk_run",
        ),
    )


class _Composition:
    """An independent lock, records and event sink; no session store is used."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.fail_append = False
        self.now = NOW
        self.indexes = transcripts.MemoryTranscriptIndexes({}, {}, {}, {}, {}, {})
        self.state = queue.MemoryQueuedMessageState(
            append_events=self.append,
            checkpoints={"target": {"queue_test": True}},
            event_records_by_id={},
            ordering=queue.MemoryMessageOrdering(),
            ownership_clock=self.clock,
            pending_session_messages={},
            queued_session_messages_by_idempotency={},
            reject_new_work_after_steering=self.reject_steering,
            session_closure_progress={},
            session_message_acceptance_event_ids={},
            session_message_delivery_records={},
            session_message_terminal_receipts={},
            session_operation_records={},
            sessions={"target": _session("target")},
            store_checkpoint=self.checkpoint,
            transcript_indexes=self.transcript_indexes,
            transcript_interaction_ids=self.indexes.interaction_ids,
            transcripts=self.indexes.messages,
        )
        self.ops = SimpleNamespace(
            **{name: partial(getattr(queue, name), self.locked) for name in OPERATIONS}
        )

    @asynccontextmanager
    async def locked(self):
        async with self.lock:
            yield self.state

    def assert_idle(self):
        assert not self.lock.locked()

    def clock(self):
        assert self.lock.locked()
        return self.now

    def append(self, session, events):
        assert self.lock.locked()
        if self.fail_append:
            raise RuntimeError("event publication rejected")
        prepared = [
            EventRecord(sequence=len(self.state.event_records_by_id) + i + 1, event=event)
            for i, event in enumerate(events)
        ]
        for row in prepared:
            self.state.event_records_by_id[(session.id, row.event.id)] = row
        return session.model_copy(update={"last_activity_at": self.now})

    def checkpoint(self, sid, checkpoint):
        assert self.lock.locked()
        self.state.checkpoints[sid] = checkpoint

    def transcript_indexes(self):
        assert self.lock.locked()
        return self.indexes

    def reject_steering(self, session, *, allow_completed_interaction=False):
        assert self.lock.locked()
        raise AssertionError("This fixture has no interaction handoff")

    def snapshot(self):
        return deepcopy((*(value for value in self.state if not callable(value)), self.indexes))


def _request(key="message"):
    return EnqueueSessionMessageRequest(
        session_id="target", idempotency_key=key, content=key, delivery_mode="next_turn"
    )


def _assert_import_without_stores(module):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys
class RejectStores(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.sessions.base", "cayu.sessions.store", "cayu.sessions.memory",
                        "cayu.storage.sqlite", "cayu.storage.postgres"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectStores())
owner = importlib.import_module(sys.argv[1])
assert owner.__file__
""",
            module,
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_memory_queue_imports_without_stores():
    _assert_import_without_stores("cayu.sessions._memory_queued_messages")


def test_memory_queue_composes_with_replay_inspection_and_transcript_publication():
    async def scenario():
        harness = _Composition()
        await _exercise(harness.ops, "target", harness.assert_idle)
        assert [message.content[0].text for message in harness.indexes.messages["target"]] == [
            "deliver this message"
        ]
        assert harness.indexes.search_documents["target"]
        assert harness.indexes.interaction_ids["target"] == [None]
        assert harness.state.ordering.next_key == 3

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["enqueue", "deliver", "withdraw"])
def test_memory_queue_publication_failure_preserves_records_and_ordering(operation):
    async def scenario():
        from cayu.sessions.messaging import SessionMessageActionRequest

        harness = _Composition()
        if operation != "enqueue":
            await harness.ops.enqueue_session_message(_request())
        page = await harness.ops.inspect_session_messages(SessionMessageQuery(session_id="target"))
        before = harness.snapshot()
        harness.fail_append = True
        with pytest.raises(RuntimeError, match="event publication rejected"):
            if operation == "enqueue":
                await harness.ops.enqueue_session_message(_request())
            elif operation == "deliver":
                await harness.ops.deliver_queued_session_messages(
                    "target", include_on_idle=True, delivery_id="delivery"
                )
            else:
                record = page.records[0]
                await harness.ops.apply_session_message_action(
                    SessionMessageActionRequest(
                        session_id="target",
                        session_instance_id=page.session_instance_id,
                        queue_id=record.queue_id,
                        expected_revision=record.revision,
                        idempotency_key="withdraw",
                        action="withdraw",
                    )
                )
        assert harness.snapshot() == before
        harness.assert_idle()

    asyncio.run(scenario())


def test_memory_queue_cancelled_lock_wait_does_not_publish_or_consume_ordering():
    async def scenario():
        harness = _Composition()
        before = harness.snapshot()
        async with harness.lock:
            task = asyncio.create_task(harness.ops.enqueue_session_message(_request()))
            await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert harness.snapshot() == before
        assert (await harness.ops.enqueue_session_message(_request())).message.ordering_key == 1

    asyncio.run(scenario())


def test_memory_queue_adapter_resolves_replaced_state_after_lock_wait():
    async def scenario():
        from cayu.sessions.memory import InMemorySessionStore
        from cayu.sessions.records import SessionIdentity
        from cayu.sessions.requests import RunRequest

        store = InMemorySessionStore()
        await store.create(
            RunRequest(session_id="target", agent_name="assistant", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        async with store._lock:
            task = asyncio.create_task(store.enqueue_session_message(_request()))
            await asyncio.sleep(0)
            assert not task.done()
            # Model deletion/replacement of maps while a caller is waiting.
            old = store._sessions
            store._sessions = {"target": _session("target")}
            store._ownership_clock = lambda: NOW
        result = await task
        assert old["target"].status == SessionStatus.PENDING
        assert result.message.accepted_at == NOW
        assert store._sessions["target"].last_activity_at == NOW
        assert result.message.ordering_key == 1

    asyncio.run(scenario())
