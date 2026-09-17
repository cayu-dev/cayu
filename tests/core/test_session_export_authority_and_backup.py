"""Cross-principal authority, qualified backup history, and four-way readback."""

from __future__ import annotations

import asyncio
import io
import json
from contextlib import asynccontextmanager

import pytest
from tests.core.test_session_exports import (
    CONTEXT,
    OWNER,
    AcceptanceReader,
    Policy,
    harness,
    published,
)
from tests.core.test_session_exports import backend as backend

from cayu.collaboration.exports import (
    SessionExportAccessContext,
    SessionExportAuthorization,
    SessionExportConflict,
    SessionExportDenied,
    SessionExportSettlementRequest,
    SessionExportUnavailable,
)
from cayu.events import Event, EventType, copy_event
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionIdentity
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.jsonl_export import export_sessions, import_sessions


class MultiPrincipalPolicy(Policy):
    def __init__(self, issuer=OWNER):
        super().__init__()
        self.issuer = issuer

    @property
    def ref(self):
        return super().ref.model_copy(update={"owner": self.issuer})

    @asynccontextmanager
    async def acquire(self, context, **kwargs):
        if self.denied.intersection(kwargs["actions"]):
            raise SessionExportDenied()
        yield SessionExportAuthorization(
            issuer=self.issuer,
            principal=context.principal,
            policy=self.ref,
            revision=1,
            expires_at_ms=4102444800000,
        )


def settlement(request, mode="retire", key="settle"):
    return SessionExportSettlementRequest(
        request=request,
        operation=request.ref.operation.model_copy(update={"caller_key": key}),
        mode=mode,
    )


@pytest.mark.parametrize("mode", ["release", "retire"])
def test_authorized_reader_and_settler_need_not_be_exporter(backend, mode):
    async def run():
        async with harness(backend) as case:
            app, store, policy, projector = case.app(
                policy=MultiPrincipalPolicy(), readers=(AcceptanceReader(),)
            )
            await case.create(store)
            request = await case.request(app)
            receipt = await app.export_session(request, context=CONTEXT)
            bob = SessionExportAccessContext(principal="bob")
            assert (await app.lookup_session_export(request, context=bob)).receipt == receipt
            assert await app.read_session_export(request, context=bob) == {"count": 5}
            with pytest.raises(SessionExportConflict):
                await app.export_session(request, context=bob)
            policy.denied.add(mode)
            with pytest.raises(SessionExportDenied):
                await app.settle_session_export(settlement(request, mode), context=bob)
            policy.denied.clear()
            result = await app.settle_session_export(settlement(request, mode), context=bob)
            assert result.initiator.principal == "bob"
            assert result.initiator.issuer == OWNER
            assert (await app.lookup_session_export(request, context=bob)).receipt == receipt
            assert receipt.expected.initiator.principal == "alice"
            assert await app.settle_session_export(settlement(request, mode), context=bob) == result
            with pytest.raises(SessionExportConflict):
                await app.settle_session_export(settlement(request, mode), context=CONTEXT)
            assert projector.calls == 1
            # An administrator's settlement releases the source deletion fence.
            await store.delete_session(case.session_id)

    asyncio.run(run())


@pytest.mark.parametrize("changed", ["issuer_name", "issuer_incarnation", "source_owner"])
def test_mutation_replay_binds_complete_identity_and_source(backend, changed):
    async def run():
        async with harness(backend) as case:
            app, store, _, _ = case.app(policy=MultiPrincipalPolicy())
            await case.create(store)
            request = await case.request(app)
            receipt = await app.export_session(request, context=CONTEXT)
            command = settlement(request)
            result = await app.settle_session_export(command, context=CONTEXT)
            owner = (
                OWNER.model_copy(update={"incarnation": "replacement-source"})
                if changed == "source_owner"
                else OWNER
            )
            issuer = (
                OWNER.model_copy(
                    update={"owner_id": "issuer-b"}
                    if changed == "issuer_name"
                    else {"incarnation": "issuer-v2"}
                )
                if changed != "source_owner"
                else OWNER
            )
            reopened, reopened_store, _, projector = case.app(
                policy=MultiPrincipalPolicy(issuer), owner=owner
            )
            for call, argument in (
                (reopened.export_session, request),
                (reopened.settle_session_export, command),
            ):
                with pytest.raises(SessionExportConflict):
                    await call(argument, context=CONTEXT)
            assert projector.calls == 0
            assert len(await published(reopened_store, case.session_id)) == 1
            if changed != "source_owner":
                # Current access is independent of historical mutation identity.
                assert (
                    await reopened.lookup_session_export(request, context=CONTEXT)
                ).receipt == receipt
                with pytest.raises(SessionExportUnavailable):
                    await reopened.read_session_export(request, context=CONTEXT)
            assert await app.settle_session_export(command, context=CONTEXT) == result

    asyncio.run(run())


def test_lookup_has_four_outcomes_but_denial_and_cancellation_remain_signals(backend):
    base = {
        "memory": InMemorySessionStore,
        "sqlite": SQLiteSessionStore,
        "postgres": PostgresSessionStore,
    }[backend[0]]

    class ReadbackStore(base):
        session_export_version = 1
        readback_hook = None

        async def load_session_operation(self, *args, **kwargs):
            if self.readback_hook is not None:
                await self.readback_hook()
            return await super().load_session_operation(*args, **kwargs)

    async def run():
        async with harness(backend) as case:
            app, store, policy, _ = case.app(store_type=ReadbackStore)
            await case.create(store)
            request = await case.request(app)
            assert (await app.lookup_session_export(request, context=CONTEXT)).status == "not_found"
            await app.export_session(request, context=CONTEXT)
            assert (await app.lookup_session_export(request, context=CONTEXT)).status == "match"
            assert (
                await app.lookup_session_export(
                    request.model_copy(update={"source_indices": (1,)}), context=CONTEXT
                )
            ).status == "conflict"

            async def unavailable():
                raise OSError("readback unavailable")

            store.readback_hook = unavailable
            assert (
                await app.lookup_session_export(request, context=CONTEXT)
            ).status == "unavailable"
            policy.denied.add("readback")
            with pytest.raises(SessionExportDenied):
                await app.lookup_session_export(request, context=CONTEXT)
            policy.denied.clear()
            entered, release = asyncio.Event(), asyncio.Event()

            async def blocked():
                entered.set()
                await release.wait()

            store.readback_hook = blocked
            task = asyncio.create_task(app.lookup_session_export(request, context=CONTEXT))
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
            release.set()
            await app.drain_session_exports()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["release", "retire"])
def test_trusted_jsonl_restores_exact_settled_history_not_export_authority(backend, mode):
    async def run():
        async with harness(backend) as case:
            app, store, _, _ = case.app(readers=(AcceptanceReader(),))
            await case.create(store)
            request = await case.request(app)
            await app.export_session(request, context=CONTEXT)
            stream = io.StringIO()
            await export_sessions(store, stream=stream)
            lines = [
                line
                for line in stream.getvalue().splitlines()
                if json.loads(line)["session"]["id"] == case.session_id
            ]
            with pytest.raises(SessionExportConflict):
                list(import_sessions(lines))
            await app.settle_session_export(settlement(request, mode), context=CONTEXT)
            stream = io.StringIO()
            await export_sessions(store, stream=stream)
            lines = [
                line
                for line in stream.getvalue().splitlines()
                if json.loads(line)["session"]["id"] == case.session_id
            ]
            [imported] = list(import_sessions(lines))
            assert "session_exports" not in (imported.checkpoint or {})
            # Restore to each native backend, using a new session after removing
            # the settled source. Reconstructed history confers no receipt authority.
            await store.delete_session(case.session_id)
            await store.create(
                RunRequest(session_id=case.session_id, agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await store.checkpoint(case.session_id, imported.checkpoint or {})
            export_event = next(
                event
                for event in imported.events
                if event.type == EventType.SESSION_EXPORT_PUBLISHED
            )
            for forged in (
                Event.model_validate(export_event.model_dump(mode="json")),
                copy_event(export_event).model_copy(update={"id": "forged-event"}),
                copy_event(export_event).model_copy(
                    update={"payload": {**export_event.payload, "output_commitment": "f" * 64}}
                ),
            ):
                with pytest.raises((ValueError, SessionExportConflict)):
                    await store.append_event(case.session_id, forged)
            await store.append_events(case.session_id, imported.events)
            restored = await store.load_events(case.session_id)
            assert [event.model_dump(mode="json") for event in restored] == [
                event.model_dump(mode="json") for event in imported.events
            ]
            # The same qualified import also restores to another store.
            destination = InMemorySessionStore()
            await destination.create(
                RunRequest(session_id=case.session_id, agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await destination.append_events(case.session_id, imported.events)
            assert len(await destination.load_events(case.session_id)) == len(imported.events)

    asyncio.run(run())
