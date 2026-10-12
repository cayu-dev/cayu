"""Memory creation composes independently and retains its lock boundaries."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from traceback import TracebackException
from types import SimpleNamespace

import pytest

import cayu
from cayu.messages import Message
from cayu.sessions import _memory_session_creation as creation
from cayu.sessions import _memory_transcripts as transcripts
from cayu.sessions.forks import fork_session_invocation
from cayu.sessions.records import Session, SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest

IDENTITY = SessionIdentity(provider_name="provider", model="model")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class _Checkpoint:
    checkpoint: dict


class _Composition:
    """Caller-owned records, locks and publication sinks; no concrete store."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.participant_lock = asyncio.Lock()
        self.now = NOW
        self.published = []
        self.aliases = []
        self.indexes = transcripts.MemoryTranscriptIndexes({}, {}, {}, {}, {}, {})
        bindings = {field.name: {} for field in fields(creation.MemorySessionCreationState)}
        bindings.update(
            apply_checkpoint_store=self.apply_checkpoint,
            apply_event_append=self.unexpected,
            append_events=self.unexpected,
            index_session_parent=self.assert_locked,
            ownership_clock=self.clock,
            prepare_checkpoint_store=self.prepare_checkpoint,
            prepare_event_append=self.unexpected,
            refresh_child_lifecycle_candidate=self.assert_locked,
            register_private_authority_alias=self.register_alias,
            require_available_closure_identity=self.assert_locked,
            require_external_creation=self.assert_locked,
            store_checkpoint=self.store_checkpoint,
            supports_model_failover_stage_protocol=lambda: True,
            transcript_indexes=self.transcript_indexes,
            transcripts=self.indexes.messages,
            transcript_interaction_ids=self.indexes.interaction_ids,
            latest_transcript_indexes_by_role=self.indexes.latest_indexes_by_role,
            transcript_search_documents=self.indexes.search_documents,
        )
        self.state = creation.MemorySessionCreationState(**bindings)
        self.ops = SimpleNamespace(
            create=partial(creation.create, self.locked),
            create_fork=partial(creation.create_fork, self.locked),
            create_participant_owned_session=partial(
                creation.create_participant_owned_session, self.participants
            ),
            lookup_participant_session_creation=partial(
                creation.lookup_participant_session_creation, self.participants
            ),
            load_participant_session_binding=partial(
                creation.load_participant_session_binding, self.read_participants
            ),
            load_participant_session_creation_receipt=partial(
                creation.load_participant_session_creation_receipt, self.read_participants
            ),
            scan_participant_session_bindings=partial(
                creation.scan_participant_session_bindings, self.read_participants
            ),
        )

    @asynccontextmanager
    async def locked(self):
        async with self.lock:
            yield self.state

    def participant_records(self):
        return {
            field.name: getattr(self.state, field.name)
            for field in fields(creation.MemoryParticipantState)
        }

    @asynccontextmanager
    async def read_participants(self):
        async with self.lock:
            yield creation.MemoryParticipantState(**self.participant_records())

    @asynccontextmanager
    async def participants(self):
        async with self.participant_lock:
            yield creation.MemoryParticipantCreationState(
                **self.participant_records(),
                create=self.ops.create,
                read_creation_decision=self.unexpected,
                created_receipt=lambda sid: self.state.participant_session_receipts.get(sid),
            )

    def assert_locked(self, *args):
        assert self.lock.locked()

    def unexpected(self, *args, **kwargs):
        raise AssertionError("Unexpected publication capability")

    def clock(self):
        self.assert_locked()
        return self.now

    def transcript_indexes(self):
        self.assert_locked()
        return self.indexes

    def register_alias(self, value, *, field_name, scope_session_id=None):
        self.assert_locked()
        self.aliases.append((value, field_name, scope_session_id))

    def prepare_checkpoint(self, session_id, checkpoint, *, additional_event_records=()):
        self.assert_locked()
        assert not additional_event_records
        return _Checkpoint(deepcopy(checkpoint))

    def apply_checkpoint(self, session_id, prepared):
        self.assert_locked()
        assert session_id in self.state.sessions
        self.published.append(session_id)
        self.state.checkpoints[session_id] = prepared.checkpoint

    def store_checkpoint(self, session_id, checkpoint):
        self.apply_checkpoint(session_id, self.prepare_checkpoint(session_id, checkpoint))

    def snapshot(self):
        assert not self.lock.locked() and not self.participant_lock.locked()
        return deepcopy(
            (
                {
                    f.name: getattr(self.state, f.name)
                    for f in fields(self.state)
                    if not callable(getattr(self.state, f.name))
                },
                self.indexes,
                self.aliases,
                self.published,
            )
        )


def _fork(source, **updates):
    return Session(
        id="child",
        agent_name=source.agent_name,
        provider_name=source.provider_name,
        model=source.model,
        parent_session_id=source.id,
        causal_budget_id=source.causal_budget_id,
        invocation=fork_session_invocation(source),
        status=source.status,
    ).model_copy(update=updates)


def _fork_kwargs(source):
    return dict(
        source_session_id=source.id,
        fork=_fork(source),
        source_statuses={SessionStatus.PENDING},
        transcript_cursor=None,
        checkpoint_transform=None,
        system_prompt_replacement=None,
        expected_source_run_epoch=source.run_epoch,
        transcript_validator=None,
    )


def test_memory_creation_owner_imports_without_store_adapters():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import importlib.abc
import sys
class RejectAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"cayu.sessions.memory", "cayu.sessions.store", "cayu.storage.sqlite", "cayu.storage.postgres", "psycopg", "psycopg_pool"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectAdapters())
owner = importlib.import_module("cayu.sessions._memory_session_creation")
assert callable(owner.create) and callable(owner.create_fork)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _format_state_failure(state):
    try:
        raise ValueError("rejected operation")
    except ValueError as error:
        return "".join(TracebackException.from_exception(error, capture_locals=True).format())


@pytest.mark.parametrize("context", ["locked", "read_participants", "participants"])
def test_memory_creation_state_diagnostics_do_not_expand_stored_records(context):
    secret = "creation-state-diagnostic-secret-canary"

    async def run():
        harness = _Composition()
        session = await harness.ops.create(
            RunRequest(
                agent_name="agent", session_id="private", messages=[], labels={"private": secret}
            ),
            identity=IDENTITY,
            checkpoint_transform=lambda session, prior: {"private": secret},
        )
        harness.indexes.messages[session.id] = [Message.text("user", secret)]

        async with getattr(harness, context)() as state:
            diagnostic = _format_state_failure(state)
            assert type(state).__name__ in diagnostic
            assert secret not in diagnostic
            assert secret not in repr(state)
            assert secret not in str(state)

        # Formatting borrowed state must leave the caller's records intact.
        assert harness.state.sessions[session.id].labels == {"private": secret}
        assert harness.state.checkpoints[session.id] == {"private": secret}
        assert harness.indexes.messages[session.id] == [Message.text("user", secret)]

    asyncio.run(run())


def test_memory_creation_decision_reader_diagnostics_do_not_expand_callback():
    @dataclass
    class Reader:
        credential: str

        async def __call__(self, target):
            raise AssertionError("Diagnostics must not call the reader")

    secret = "creation-reader-diagnostic-secret-canary"
    callback = Reader(credential=secret)
    reader = creation.CreationDecisionReader(callback)
    diagnostic = _format_state_failure(reader)
    assert type(reader).__name__ in diagnostic
    assert secret not in diagnostic
    assert secret not in repr(reader)
    assert secret not in str(reader)
    assert reader.read_session_creation_decision is callback


def test_memory_creation_and_fork_compose_without_a_store():
    async def run():
        harness = _Composition()
        source = await harness.ops.create(
            RunRequest(agent_name="agent", session_id="source", messages=[]),
            identity=IDENTITY,
            checkpoint_transform=lambda session, prior: {"original": [1]},
        )
        assert source.created_at == NOW
        harness.indexes.messages[source.id] = [Message.text("user", "historical input")]
        harness.indexes.interaction_ids[source.id] = ["historical-interaction"]
        kwargs = _fork_kwargs(source)
        kwargs["checkpoint_transform"] = lambda session, prior: {"copied": prior["original"]}
        child = await harness.ops.create_fork(**kwargs)
        assert child.instance_id != source.instance_id
        assert harness.state.checkpoints[child.id] == {"copied": [1]}
        assert harness.indexes.interaction_ids[child.id] == ["historical-interaction"]
        assert harness.indexes.latest_indexes_by_role[child.id] == {"user": 0}
        assert harness.indexes.search_documents[child.id]
        assert ("historical-interaction", "interaction_id", child.id) in harness.aliases
        child.labels["caller"] = "detached"
        assert not harness.state.sessions[child.id].labels
        harness.state.checkpoints[child.id]["copied"].append(2)
        assert harness.state.checkpoints[source.id] == {"original": [1]}
        harness.snapshot()

    asyncio.run(run())


@pytest.mark.parametrize(
    "phase", ["operation_initializer", "checkpoint_transform", "participant_binding_factory"]
)
@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_memory_creation_rejected_preparation_publishes_nothing(phase, error):
    async def run():
        harness = _Composition()
        before = harness.snapshot()

        def fail(*args):
            harness.assert_locked()
            raise error("rejected preparation")

        with pytest.raises(error):
            await harness.ops.create(
                RunRequest(agent_name="agent", session_id="rejected", messages=[]),
                identity=IDENTITY,
                **{phase: fail},
            )
        assert harness.snapshot() == before

    asyncio.run(run())


def test_memory_fork_rejected_transcript_publishes_nothing():
    async def run():
        harness = _Composition()
        source = await harness.ops.create(
            RunRequest(agent_name="agent", session_id="source", messages=[]), identity=IDENTITY
        )
        before = harness.snapshot()
        kwargs = _fork_kwargs(source)
        kwargs["transcript_validator"] = lambda snapshot: False
        with pytest.raises(ValueError, match="Fork transcript failed atomic copy validation"):
            await harness.ops.create_fork(**kwargs)
        assert harness.snapshot() == before

    asyncio.run(run())


def test_memory_participant_creation_composes_and_replays_without_a_store():
    from tests.core.test_context_views import (
        _test_memory_participant_creation_is_atomic_and_replayable,
    )

    async def run():
        harness = _Composition()
        request, receipt, sid = await _test_memory_participant_creation_is_atomic_and_replayable(
            harness.ops, cleanup=False
        )
        assert await harness.ops.load_participant_session_creation_receipt(sid) == receipt
        assert (await harness.ops.lookup_participant_session_creation(request))[1] == receipt
        refs = await harness.ops.scan_participant_session_bindings(receipt.binding.participant)
        assert len(refs) == 1 and refs[0].session_id == sid
        harness.snapshot()

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["create", "fork", "participant"])
def test_memory_creation_cancelled_lock_wait_leaves_state_unchanged(operation):
    from cayu.sessions.context_views import ParticipantSessionCreationRequest

    async def run():
        harness = _Composition()
        source = await harness.ops.create(
            RunRequest(agent_name="agent", session_id="source", messages=[]), identity=IDENTITY
        )
        before = harness.snapshot()
        lock = harness.participant_lock if operation == "participant" else harness.lock
        async with lock:
            if operation == "create":
                call = harness.ops.create(
                    RunRequest(agent_name="agent", messages=[]), identity=IDENTITY
                )
            elif operation == "fork":
                call = harness.ops.create_fork(**_fork_kwargs(source))
            else:
                request = ParticipantSessionCreationRequest(
                    request=RunRequest(agent_name="agent", messages=[]), creation_key="waiting"
                )
                call = harness.ops.create_participant_owned_session(
                    request,
                    resolved_request=request.request,
                    identity=IDENTITY,
                    binding_factory=harness.unexpected,
                )
            task = asyncio.create_task(call)
            await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert harness.snapshot() == before

    asyncio.run(run())


def test_memory_creation_resolves_current_state_after_store_lock_wait():
    from cayu.sessions.memory import InMemorySessionStore

    async def run():
        store = InMemorySessionStore(ownership_clock=lambda: NOW)
        old_sessions = store._sessions
        async with store._lock:
            task = asyncio.create_task(
                store.create(
                    RunRequest(agent_name="agent", session_id="fresh", messages=[]),
                    identity=IDENTITY,
                )
            )
            await asyncio.sleep(0)
            assert not task.done()
            store._sessions = {}
            store._ownership_clock = lambda: NOW + timedelta(days=1)
        session = await task
        assert not old_sessions
        assert store._sessions[session.id] == session
        assert session.created_at == NOW + timedelta(days=1)
        assert not store._lock.locked()

    asyncio.run(run())


def test_memory_participant_creation_reads_receipt_after_nested_lock_wait():
    from tests.core.test_context_views import (
        _test_memory_participant_creation_is_atomic_and_replayable,
    )

    from cayu.sessions.memory import InMemorySessionStore

    async def run():
        store = InMemorySessionStore()
        async with store._lock:
            task = asyncio.create_task(
                _test_memory_participant_creation_is_atomic_and_replayable(store, cleanup=False)
            )
            await asyncio.sleep(0)
            assert store._participant_creation_lock.locked() and not task.done()
            old_receipts = store._participant_session_receipts
            store._participant_session_receipts = {}
        request, receipt, sid = await task
        assert not old_receipts
        assert store._participant_session_receipts[sid] == receipt
        assert not store._participant_creation_lock.locked()
        assert (await store.lookup_participant_session_creation(request))[1] == receipt

    asyncio.run(run())


def test_memory_participant_nested_lock_cancellation_releases_serialization():
    from tests.core.test_context_views import (
        _test_memory_participant_creation_is_atomic_and_replayable,
    )

    from cayu.sessions.memory import InMemorySessionStore

    async def run():
        store = InMemorySessionStore()
        async with store._lock:
            task = asyncio.create_task(
                _test_memory_participant_creation_is_atomic_and_replayable(store, cleanup=False)
            )
            await asyncio.sleep(0)
            assert store._participant_creation_lock.locked()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not store._participant_creation_lock.locked()
            assert not store._sessions and not store._participant_session_receipts
        await _test_memory_participant_creation_is_atomic_and_replayable(store)

    asyncio.run(run())


def test_memory_participant_deletion_waits_for_creation_receipt(monkeypatch):
    from tests.core.test_context_views import (
        _test_memory_participant_creation_is_atomic_and_replayable,
    )

    from cayu.sessions.memory import InMemorySessionStore

    async def run():
        store = InMemorySessionStore()
        create_participant = store.create_participant_owned_session
        binding_factory = None

        async def capture_binding(*args, **kwargs):
            nonlocal binding_factory
            binding_factory = kwargs["binding_factory"]
            return await create_participant(*args, **kwargs)

        monkeypatch.setattr(store, "create_participant_owned_session", capture_binding)
        request, _, sid = await _test_memory_participant_creation_is_atomic_and_replayable(
            store, cleanup=False
        )
        await store.delete_session(sid)
        entered, release = asyncio.Event(), asyncio.Event()
        original = store.create

        async def create_then_wait(*args, **kwargs):
            session = await original(*args, **kwargs)
            entered.set()
            await release.wait()
            return session

        monkeypatch.setattr(store, "create", create_then_wait)

        assert binding_factory is not None
        creating = asyncio.create_task(
            create_participant(
                request,
                resolved_request=request.request,
                identity=IDENTITY,
                binding_factory=binding_factory,
            )
        )
        deleting = None
        try:
            await asyncio.wait_for(entered.wait(), 5)
            created_id = next(iter(store._sessions))
            deleting = asyncio.create_task(store.delete_session(created_id))
            await asyncio.sleep(0)
            assert not deleting.done()
            release.set()
            session, created_receipt = await asyncio.wait_for(creating, 5)
            assert created_receipt.binding.session_instance_id == session.instance_id
            await asyncio.wait_for(deleting, 5)
            assert await store.load(session.id) is None
        finally:
            release.set()
            for task in (creating, deleting):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(t for t in (creating, deleting) if t is not None), return_exceptions=True
            )

    asyncio.run(run())
