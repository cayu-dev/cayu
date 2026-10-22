"""Memory grant operations compose without a store and retain atomic publication."""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

import cayu
from cayu.events import Event, EventType
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions import _memory_tool_grants as grants
from cayu.sessions._closure_records import ClosureProgressIndex
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.invocation import InvocationOrigin, SessionInvocation
from cayu.sessions.records import EventRecord, Session, SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tools.grants import (
    PreparedTargetedToolGrant,
    TargetedToolGrant,
    TargetedToolUseRequest,
    build_targeted_tool_grant_record,
    persisted_targeted_tool_grant_batch_fingerprint,
    targeted_tool_grant_event,
)

OPERATIONS = (
    "issue_targeted_tool_grants",
    "list_targeted_tool_grants",
    "load_targeted_tool_grant_state",
    "bind_targeted_tool_grant_use",
    "revoke_targeted_tool_grant",
    "reconstruct_targeted_tool_grants",
)
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _codec():
    key = base64.urlsafe_b64encode(bytes([53]) * 32).decode("ascii").rstrip("=")
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})
    )


def _grant(session, codec):
    record = build_targeted_tool_grant_record(
        PreparedTargetedToolGrant(
            request=TargetedToolGrant(request_id="request", tool_id="cayu:remember", max_calls=2),
            tool_name="remember",
            catalogue_revision="sha256:" + "1" * 64,
            descriptor_version="sha256:" + "2" * 64,
            schema_fingerprint="sha256:" + "3" * 64,
        ),
        session_id=session.id,
        interaction_id="interaction",
        generation_id="sha256:" + "4" * 64,
        agent_name=session.agent_name,
        task_id=None,
        environment_name=session.environment_name,
        principal=session.invocation.origin.subject,
        tenant=session.invocation.origin.tenant,
        issued_at=NOW,
        codec=codec,
    )
    started = Event(
        id="started",
        type=EventType.INTERACTION_STARTED,
        timestamp=NOW,
        session_id=session.id,
        interaction_id=record.interaction_id,
        payload={
            "targeted_tool_grant_count": 1,
            "targeted_tool_grant_batch_fingerprint": persisted_targeted_tool_grant_batch_fingerprint(
                (record,)
            ),
        },
    )
    issued = targeted_tool_grant_event(
        record,
        event_type=EventType.TARGETED_TOOL_GRANT_ISSUED,
        timestamp=NOW,
        outcome="issued",
        event_id_suffix="issued",
    )
    return record, started, issued


def _calls(ops, session, record, issued):
    request = TargetedToolUseRequest(
        **{
            key: getattr(record, key)
            for key in (
                "tool_ref",
                "session_id",
                "interaction_id",
                "generation_id",
                "agent_name",
                "task_id",
                "environment_name",
                "principal",
                "tenant",
                "catalogue_revision",
                "descriptor_version",
                "schema_fingerprint",
                "tool_id",
                "tool_name",
            )
        },
        model_step_id="step",
        outer_tool_call_id="call",
        invocation_id="invocation",
        arguments_sha256="sha256:" + "5" * 64,
        expected_run_epoch=session.run_epoch,
    )
    return SimpleNamespace(
        issue=partial(
            ops.issue_targeted_tool_grants,
            session.id,
            expected_run_epoch=session.run_epoch,
            records=(record,),
            events=(issued,),
        ),
        listing=partial(ops.list_targeted_tool_grants, session.id),
        snapshot=partial(ops.load_targeted_tool_grant_state, session.id),
        bind=partial(ops.bind_targeted_tool_grant_use, request, observed_at=NOW),
        revoke=partial(
            ops.revoke_targeted_tool_grant,
            record.tool_ref,
            session_id=session.id,
            expected_run_epoch=session.run_epoch,
            reason="operator-revoked",
            revoked_at=NOW + timedelta(seconds=1),
        ),
        reconstruct=partial(
            ops.reconstruct_targeted_tool_grants,
            session.id,
            expected_run_epoch=session.run_epoch,
            interaction_id=record.interaction_id,
            generation_id=record.generation_id,
            agent_name=record.agent_name,
            task_id=record.task_id,
            environment_name=record.environment_name,
            principal=record.principal,
            tenant=record.tenant,
            catalogue_revision=record.catalogue_revision,
            descriptors_by_id={
                record.tool_id: (
                    record.tool_name,
                    record.descriptor_version,
                    record.schema_fingerprint,
                )
            },
            capability_ceiling_names=frozenset({record.tool_name}),
            observed_at=NOW,
        ),
        request=request,
    )


class _Composition:
    """A caller-owned lock and event sink; no concrete session store is retained."""

    def __init__(self):
        self.lock = asyncio.Lock()
        self.codec_reads = []
        self.fail_prepare = False
        self.published = []
        self.codec = _codec()
        self.session = Session(
            id="grants",
            agent_name="assistant",
            provider_name="fake",
            model="fake",
            status=SessionStatus.RUNNING,
            invocation=SessionInvocation(
                origin=InvocationOrigin(trust="unattributed"),
                root_invocation_id="11111111-1111-4111-8111-111111111111",
                root_session_id="grants",
                source="sdk_run",
            ),
        )
        self.record, started, self.issued = _grant(self.session, self.codec)
        event_record = EventRecord(sequence=1, event=started)
        self.state = grants.MemoryToolGrantState(
            event_records_by_id={(self.session.id, started.id): event_record},
            interaction_event_records={self.session.id: {"interaction": [event_record]}},
            latest_interaction_event_records_by_sequence={self.session.id: [event_record]},
            public_authority_aliases={},
            session_closure_progress=ClosureProgressIndex(),
            sessions={self.session.id: self.session},
            targeted_tool_grant_ids_by_interaction={},
            targeted_tool_grant_ids_by_request={},
            targeted_tool_grant_ids_by_session={},
            targeted_tool_grant_ids_by_tool={},
            targeted_tool_grant_refs={},
            targeted_tool_grants={},
            targeted_tool_use_by_invocation={},
            targeted_tool_use_by_outer_call={},
            targeted_tool_uses={},
            prepare_events=self.prepare,
            apply_events=self.apply,
        )
        self.ops = SimpleNamespace(
            **{
                name: partial(
                    getattr(grants, name),
                    self.locked,
                    **(
                        {}
                        if name == "reconstruct_targeted_tool_grants"
                        else {"get_codec": self.get_codec}
                    ),
                )
                for name in OPERATIONS
            }
        )
        self.calls = _calls(self.ops, self.session, self.record, self.issued)

    @asynccontextmanager
    async def locked(self):
        async with self.lock:
            yield self.state

    def get_codec(self):
        self.codec_reads.append(self.lock.locked())
        return self.codec

    def prepare(self, session, events):
        assert self.lock.locked() and session.id == self.session.id
        if self.fail_prepare:
            raise RuntimeError("event preparation failed")
        records = tuple(
            EventRecord(
                sequence=len(self.state.event_records_by_id) + i + 1,
                event=event.model_copy(deep=True),
            )
            for i, event in enumerate(events)
        )
        assert all(
            (session.id, row.event.id) not in self.state.event_records_by_id for row in records
        )
        return records

    def apply(self, session, prepared):
        assert self.lock.locked()
        # Preparation and grant mutation must precede event publication.
        for row in prepared:
            if row.event.type == EventType.TARGETED_TOOL_GRANT_ISSUED:
                assert self.record.grant_id in self.state.targeted_tool_grants
            if row.event.type == EventType.TARGETED_TOOL_REFERENCE_CONSUMED:
                assert self.state.targeted_tool_grants[self.record.grant_id].used_calls == 1
                assert len(self.state.targeted_tool_use_by_invocation) == 1
                assert len(self.state.targeted_tool_use_by_outer_call) == 1
            if row.event.type == EventType.TARGETED_TOOL_GRANT_REVOKED:
                assert self.state.targeted_tool_grants[self.record.grant_id].revoked_at is not None
            self.state.event_records_by_id[(session.id, row.event.id)] = row
            self.published.append(row.event.id)
        return session.model_copy(update={"last_activity_at": NOW})

    def snapshot(self):
        # Empty use buckets are an existing lazy cache; compare durable records.
        return deepcopy(
            (
                *(
                    value
                    for name, value in zip(self.state._fields, self.state, strict=True)
                    if name not in {"prepare_events", "apply_events", "targeted_tool_uses"}
                ),
                [
                    binding
                    for uses in self.state.targeted_tool_uses.values()
                    for binding in uses.values()
                ],
                self.published,
            )
        )


def test_memory_tool_grants_import_without_store_implementations():
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
owner = importlib.import_module("cayu.sessions._memory_tool_grants")
assert callable(owner.issue_targeted_tool_grants)
assert callable(owner.reconstruct_targeted_tool_grants)
""",
        ],
        env={**os.environ, "PYTHONPATH": str(Path(cayu.__file__).resolve().parent.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_memory_tool_grants_compose_with_replay_and_atomic_publication():
    async def scenario():
        harness = _Composition()
        calls = harness.calls
        assert await calls.listing() == ()
        assert (await calls.snapshot()).records == ()

        async def fail_then_retry(call):
            before = harness.snapshot()
            harness.fail_prepare = True
            with pytest.raises(RuntimeError, match="event preparation failed"):
                await call()
            harness.fail_prepare = False
            assert harness.snapshot() == before and not harness.lock.locked()
            return await call()

        harness.codec_reads.clear()
        assert (await fail_then_retry(calls.issue)).outcomes == ("issued",)
        assert harness.codec_reads == [False, False]
        harness.codec_reads.clear()
        assert (await calls.issue()).outcomes == ("reused",)
        assert harness.codec_reads == [False, True]
        assert (await fail_then_retry(calls.reconstruct)).valid == (harness.record,)
        bound = await fail_then_retry(calls.bind)
        assert bound.disposition == "bound"
        assert (await calls.bind()).disposition == "rejoined"
        published = list(harness.published)
        assert (await calls.bind()).disposition == "rejoined"
        assert harness.published == published
        state = await calls.snapshot()
        assert state.records[0].used_calls == 1 and state.uses == (bound.binding,)
        before = harness.snapshot()
        with pytest.raises(SessionRunFenced):
            await harness.ops.bind_targeted_tool_grant_use(
                calls.request.model_copy(update={"expected_run_epoch": 1}),
                observed_at=NOW,
            )
        assert harness.snapshot() == before
        revoked = await fail_then_retry(calls.revoke)
        assert await calls.revoke() == revoked
        assert (await calls.listing()) == (revoked,)
        assert len(harness.published) == len(set(harness.published))
        assert not harness.lock.locked()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "operation", ["issue", "listing", "snapshot", "bind", "revoke", "reconstruct"]
)
def test_memory_tool_grants_resolve_replaced_maps_after_lock(operation):
    from cayu.sessions.memory import InMemorySessionStore

    async def scenario():
        store = InMemorySessionStore(public_authority_alias_codec=_codec())
        session = await store.create(
            RunRequest(session_id="grants", agent_name="assistant", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        session = await store.transition_status(
            session.id,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        record, started, issued = _grant(session, store.public_authority_alias_codec)
        await store.append_event(session.id, started)
        calls = _calls(store, session, record, issued)
        if operation != "issue":
            await calls.issue()
        old_sessions = store._sessions
        await store._lock.acquire()
        task = asyncio.create_task(getattr(calls, operation)())
        try:
            await asyncio.sleep(0)
            assert not task.done()
            # Session deletion replaces maps. A queued operation must not retain
            # the earlier session/grant dictionaries after it acquires the lock.
            store._sessions = {}
            store._targeted_tool_grants = {}
        finally:
            store._lock.release()
        with pytest.raises(KeyError, match="Session not found"):
            await task
        assert session.id in old_sessions and not store._lock.locked()
        assert store._targeted_tool_grants == {}

    asyncio.run(scenario())


def test_memory_tool_grant_waiter_cancellation_preserves_state():
    async def scenario():
        harness = _Composition()
        before = harness.snapshot()
        await harness.lock.acquire()
        task = asyncio.create_task(harness.calls.issue())
        try:
            await asyncio.sleep(0)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert harness.snapshot() == before
        finally:
            harness.lock.release()
        assert (await harness.calls.issue()).outcomes == ("issued",)
        assert not harness.lock.locked()

    asyncio.run(scenario())


def test_memory_tool_grant_codec_reads_follow_validation_and_reference_lookup():
    async def scenario():
        harness = _Composition()
        with pytest.raises(ValueError):
            await harness.ops.list_targeted_tool_grants(" ")
        with pytest.raises(KeyError):
            await harness.ops.load_targeted_tool_grant_state("missing")
        assert harness.codec_reads == []
        await harness.calls.issue()
        harness.codec_reads.clear()
        await harness.calls.listing()
        await harness.calls.snapshot()
        await harness.calls.bind()
        assert harness.codec_reads == [True, True, True]
        harness.codec_reads.clear()
        rejected = await harness.ops.bind_targeted_tool_grant_use(
            harness.calls.request.model_copy(update={"tool_ref": "unknown"}),
            observed_at=NOW,
        )
        assert rejected.disposition == "rejected" and harness.codec_reads == []

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["issue", "bind", "revoke", "reconstruct"])
def test_memory_tool_grant_preparation_failure_preserves_public_state_and_outbox(
    operation, monkeypatch
):
    from cayu.sessions.memory import InMemorySessionStore

    async def scenario():
        store = InMemorySessionStore(public_authority_alias_codec=_codec())
        session = await store.create(
            RunRequest(session_id="grants", agent_name="assistant", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        session = await store.transition_status(
            session.id,
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        record, started, issued = _grant(session, store.public_authority_alias_codec)
        await store.append_event(session.id, started)
        calls = _calls(store, session, record, issued)
        if operation != "issue":
            await calls.issue()

        async def snapshot():
            return (
                await calls.snapshot(),
                await store.load(session.id),
                await store.load_events(session.id),
                await store.list_persisted_event_side_effect_deliveries(),
                deepcopy(store._public_authority_aliases),
                deepcopy(store._targeted_tool_grant_refs),
                store._next_event_sequence,
            )

        before = await snapshot()
        prepare = store._prepare_event_append_unlocked

        def fail(session, events):
            assert store._lock.locked()
            prepare(session, events)
            raise RuntimeError("event preparation failed")

        with monkeypatch.context() as patch:
            patch.setattr(store, "_prepare_event_append_unlocked", fail)
            with pytest.raises(RuntimeError, match="event preparation failed"):
                await getattr(calls, operation)()
        assert await snapshot() == before and not store._lock.locked()
        await getattr(calls, operation)()
        after = await snapshot()
        assert len(after[2]) == len(before[2]) + 1
        assert len(after[3]) == len(before[3]) + 1
        assert after[-1] == before[-1] + 1

    asyncio.run(scenario())
