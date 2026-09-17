"""Public application export flows across native session-store backends."""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from copy import deepcopy
from uuid import uuid4

import pytest
from tests.core.test_targeted_tool_grants import _codec

from cayu.applications import CayuApp
from cayu.collaboration._contracts import ExactMatch, ObjectRef, OperationRef, OwnerRef
from cayu.collaboration.exports import (
    ExportLimits,
    SessionExportAcceptance,
    SessionExportAcceptanceReader,
    SessionExportAccessContext,
    SessionExportAuthorization,
    SessionExportCapacityExceeded,
    SessionExportConflict,
    SessionExportDenied,
    SessionExportPolicy,
    SessionExportProjector,
    SessionExportRef,
    SessionExportRegistration,
    SessionExportRequest,
    SessionExportSettlementRequest,
    SessionExportUnavailable,
)
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.base import (
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
    SessionOperationPublication,
    SessionStatus,
)
from cayu.storage import PostgresSessionStore, SQLiteSessionStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor

OWNER = OwnerRef(application_scope="export-tests", owner_id="source", incarnation="v1")
AUDIENCE = OwnerRef(application_scope="export-tests", owner_id="receiver", incarnation="v1")
CONTEXT = SessionExportAccessContext(principal="alice")
SECRET = "export-secret-canary-74893"


def _ref(kind):
    return ObjectRef(owner=OWNER, kind=kind, object_id=kind, incarnation="v1", revision=1)


class Policy(SessionExportPolicy):
    def __init__(self):
        self.lock = asyncio.Lock()
        self.denied = set()
        self.calls = []

    @property
    def ref(self):
        return _ref("policy")

    @asynccontextmanager
    async def acquire(self, context, *, session_id, session_instance_id, actions, audience=None):
        async with self.lock:
            self.calls.append((session_id, session_instance_id, actions, audience))
            if (
                context.principal != CONTEXT.principal
                or not actions
                or self.denied.intersection(actions)
            ):
                raise SessionExportDenied()
            yield SessionExportAuthorization(
                issuer=OWNER,
                principal=context.principal,
                policy=self.ref,
                revision=1,
                expires_at_ms=4102444800000,
            )


class Projector(SessionExportProjector):
    def __init__(self):
        self.calls = 0
        self.entered = threading.Event()
        self.release = None
        self.barrier = None
        self.secret = False

    @property
    def ref(self):
        return _ref("projector")

    @staticmethod
    def expected(source):
        return {
            "count": sum(
                len(part.text)
                for row in source
                for part in row.message.content
                if part.type == "text"
            )
        }

    def project(self, source):
        self.calls += 1
        self.entered.set()
        if self.release is not None and not self.release.wait(10):
            raise AssertionError("Projector release barrier timed out")
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        return {"secret": SECRET} if self.secret else self.expected(source)

    def validate(self, source, output, audience):
        return audience == AUDIENCE and (self.secret or output == self.expected(source))


class AcceptanceReader(SessionExportAcceptanceReader):
    def __init__(self):
        self.calls = []
        self.conflict = False

    @property
    def owner(self):
        return AUDIENCE

    async def lookup(self, receipt):
        self.calls.append(receipt)
        accepted = (
            receipt.model_copy(update={"event_id": "other-event"}) if self.conflict else receipt
        )
        return ExactMatch[SessionExportAcceptance](
            receipt=SessionExportAcceptance(
                export_receipt=accepted,
                receiving_owner=AUDIENCE,
                receipt_id="acceptance",
            )
        )


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def backend(request, tmp_path):
    kind = request.param
    dsn = request.getfixturevalue("postgres_dsn") if kind == "postgres" else None
    memory = (
        InMemorySessionStore(public_authority_alias_codec=_codec()) if kind == "memory" else None
    )

    def factory(store_type=None):
        if kind == "memory":
            return (
                memory if store_type is None else store_type(public_authority_alias_codec=_codec())
            )
        if kind == "sqlite":
            return (store_type or SQLiteSessionStore)(
                tmp_path / "exports.sqlite", public_authority_alias_codec=_codec()
            )
        return (store_type or PostgresSessionStore)(
            dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=_codec()
        )

    return kind, factory


class Harness:
    def __init__(self, factory):
        self.factory = factory
        self.stores = []
        self.apps = []
        self.projectors = []
        self.session_id = f"exports-{uuid4().hex}"

    def app(
        self,
        *,
        limits=None,
        redactor=None,
        projectors=None,
        readers=(),
        store_type=None,
        policy=None,
        owner=OWNER,
    ):
        store = self.factory(store_type)
        if store not in self.stores:
            self.stores.append(store)
        policy = Policy() if policy is None else policy
        projector = Projector()
        selected = (projector,) if projectors is None else projectors
        self.projectors.extend(selected)
        registration = SessionExportRegistration(
            owner=owner,
            policy=policy,
            projectors=selected,
            limits=limits
            or ExportLimits(max_exports=16, max_pending=8, max_retained_bytes=1024 * 1024),
            readers=readers,
        )
        app = CayuApp(
            session_store=store,
            session_exports=registration,
            enable_logging=False,
            secret_redactor=redactor,
        )
        self.apps.append(app)
        return app, store, policy, projector

    async def create(self, store):
        await store.create(
            RunRequest(session_id=self.session_id, agent_name="assistant", messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await store.append_transcript_messages(self.session_id, [Message.text("user", "hello")])
        await store.update_status(self.session_id, SessionStatus.COMPLETED)
        await store.append_event(
            self.session_id, Event(type=EventType.SESSION_COMPLETED, session_id=self.session_id)
        )

    async def request(self, app, key="export"):
        namespace = await app.initialize_session_exports(self.session_id, context=CONTEXT)
        return SessionExportRequest(
            ref=SessionExportRef(
                session_id=self.session_id,
                session_instance_id=namespace.session_instance_id,
                operation=OperationRef(
                    application_scope=OWNER.application_scope,
                    namespace_incarnation=namespace.namespace_incarnation,
                    generation=namespace.generation,
                    caller_key=key,
                ),
            ),
            source_indices=(0,),
            audience=AUDIENCE,
            projector=_ref("projector"),
            policy=_ref("policy"),
        )

    async def close(self):
        for projector in self.projectors:
            if projector.release is not None:
                projector.release.set()
        for app in self.apps:
            await app.drain_session_exports()
        for store in self.stores:
            if hasattr(store, "close"):
                await store.close()


@asynccontextmanager
async def harness(backend):
    case = Harness(backend[1])
    try:
        yield case
    finally:
        await case.close()


async def published(store, session_id):
    return [
        event
        for event in await store.load_events(session_id)
        if event.type == EventType.SESSION_EXPORT_PUBLISHED
    ]


def test_public_export_namespace_replay_and_exposure(backend):
    async def run():
        async with harness(backend) as case:
            app, store, policy, projector = case.app()
            await case.create(store)
            req = await case.request(app)
            first = await app.initialize_session_exports(case.session_id, context=CONTEXT)
            assert first == await app.initialize_session_exports(case.session_id, context=CONTEXT)
            assert (await app.lookup_session_export(req, context=CONTEXT)).status == "not_found"
            receipt = await app.export_session(req, context=CONTEXT)
            assert receipt == await app.export_session(req, context=CONTEXT)
            lookup = await app.lookup_session_export(req, context=CONTEXT)
            assert lookup.status == "match" and lookup.receipt == receipt
            assert await app.read_session_export(req, context=CONTEXT) == {"count": 5}
            assert projector.calls == 1
            events = await published(store, case.session_id)
            assert [event.id for event in events] == [receipt.event_id]
            assert all("count" not in event.payload for event in events)
            assert ("readback", "source", "export") in [call[2] for call in policy.calls]
            assert ("readback", "expose") in [call[2] for call in policy.calls]
            assert all(
                call[:2] == (case.session_id, req.ref.session_instance_id) for call in policy.calls
            )

    asyncio.run(run())


@pytest.mark.parametrize("denied", ["source", "export"])
def test_export_requires_each_independent_permission_before_projection(backend, denied):
    async def run():
        async with harness(backend) as case:
            app, store, policy, projector = case.app()
            await case.create(store)
            req = await case.request(app)
            policy.denied.add(denied)
            with pytest.raises(SessionExportDenied):
                await app.export_session(req, context=CONTEXT)
            assert projector.calls == 0
            assert not await published(store, case.session_id)
            assert (await app.lookup_session_export(req, context=CONTEXT)).status == "not_found"
            policy.denied.clear()
            await app.export_session(req, context=CONTEXT)
            assert projector.calls == 1

    asyncio.run(run())


def test_lost_commit_acknowledgement_reconciles_without_projection_or_event_repeat(backend):
    base = {
        "memory": InMemorySessionStore,
        "sqlite": SQLiteSessionStore,
        "postgres": PostgresSessionStore,
    }[backend[0]]

    class LostAcknowledgementStore(base):
        # Delegate mutation to the native owner; lose only the post-commit ack.
        session_export_version = 1
        dispatched = 0

        async def publish_session_operation_guarded_with_store_time(self, *args, **kwargs):
            result = await super().publish_session_operation_guarded_with_store_time(
                *args, **kwargs
            )
            if any(event.type == EventType.SESSION_EXPORT_PUBLISHED for event in kwargs["events"]):
                self.dispatched += 1
                raise OSError("Simulated lost publication acknowledgement")
            return result

    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app(store_type=LostAcknowledgementStore)
            await case.create(store)
            req = await case.request(app)
            receipt = await app.export_session(req, context=CONTEXT)
            assert store.dispatched == 1
            assert await app.export_session(req, context=CONTEXT) == receipt
            assert (await app.lookup_session_export(req, context=CONTEXT)).receipt == receipt
            assert await app.read_session_export(req, context=CONTEXT) == {"count": 5}
            assert projector.calls == 1
            assert len(await published(store, case.session_id)) == 1

    asyncio.run(run())


def test_reopen_replays_without_registered_old_projector(backend):
    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app()
            await case.create(store)
            req = await case.request(app)
            receipt = await app.export_session(req, context=CONTEXT)
            await app.drain_session_exports()
            if backend[0] != "memory":
                await store.close()
                case.stores.remove(store)
            reopened, peer, _, _ = case.app(projectors=())
            assert await reopened.initialize_session_exports(case.session_id, context=CONTEXT)
            assert await reopened.export_session(req, context=CONTEXT) == receipt
            assert await reopened.read_session_export(req, context=CONTEXT) == {"count": 5}
            assert projector.calls == 1
            assert len(await published(peer, case.session_id)) == 1

    asyncio.run(run())


@pytest.mark.parametrize("field", ["source_indices", "audience", "projector", "policy"])
def test_same_key_changed_request_conflicts_without_projection(backend, field):
    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app()
            await case.create(store)
            req = await case.request(app)
            receipt = await app.export_session(req, context=CONTEXT)
            value = getattr(req, field)
            changed = req.model_copy(
                update={
                    field: ()
                    if field == "source_indices"
                    else value.model_copy(update={"incarnation": "other"})
                }
            )
            assert (await app.lookup_session_export(changed, context=CONTEXT)).status == "conflict"
            with pytest.raises(SessionExportConflict):
                await app.export_session(changed, context=CONTEXT)
            assert await app.export_session(req, context=CONTEXT) == receipt
            assert projector.calls == 1
            assert len(await published(store, case.session_id)) == 1

    asyncio.run(run())


def test_readback_does_not_imply_payload_exposure(backend):
    async def run():
        async with harness(backend) as case:
            app, store, policy, _ = case.app()
            await case.create(store)
            req = await case.request(app)
            receipt = await app.export_session(req, context=CONTEXT)
            policy.denied.add("expose")
            assert (await app.lookup_session_export(req, context=CONTEXT)).receipt == receipt
            with pytest.raises(SessionExportDenied):
                await app.read_session_export(req, context=CONTEXT)

    asyncio.run(run())


@pytest.mark.parametrize("same_key", [True, False])
def test_independent_apps_converge_namespace_and_compare_and_swap(backend, same_key):
    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app()
            peer, _, _, other = case.app()
            await case.create(store)
            namespaces = await asyncio.gather(
                app.initialize_session_exports(case.session_id, context=CONTEXT),
                peer.initialize_session_exports(case.session_id, context=CONTEXT),
            )
            assert namespaces[0] == namespaces[1]
            first = await case.request(app)
            second = await case.request(peer, "export" if same_key else "other")
            projector.barrier = other.barrier = threading.Barrier(2)
            receipts = await asyncio.wait_for(
                asyncio.gather(
                    app.export_session(first, context=CONTEXT),
                    peer.export_session(second, context=CONTEXT),
                ),
                timeout=15,
            )
            if same_key:
                assert receipts[0] == receipts[1]
            assert len(await published(store, case.session_id)) == (1 if same_key else 2)
            assert (await app.lookup_session_export(first, context=CONTEXT)).receipt == receipts[0]
            assert (await peer.lookup_session_export(second, context=CONTEXT)).receipt == receipts[
                1
            ]

    asyncio.run(run())


def test_pending_retention_blocks_delete_until_exact_retirement(backend):
    async def run():
        async with harness(backend) as case:
            app, store, _, _ = case.app()
            await case.create(store)
            req = await case.request(app)
            receipt = await app.export_session(req, context=CONTEXT)
            with pytest.raises(SessionExportConflict):
                await store.delete_session(case.session_id)
            assert await store.load(case.session_id) is not None
            settlement = SessionExportSettlementRequest(
                request=req,
                operation=req.ref.operation.model_copy(update={"caller_key": "retire"}),
                mode="retire",
            )
            retired = await app.settle_session_export(settlement, context=CONTEXT)
            assert retired == await app.settle_session_export(settlement, context=CONTEXT)
            assert (await app.lookup_session_export(req, context=CONTEXT)).receipt == receipt
            with pytest.raises(SessionExportUnavailable):
                await app.read_session_export(req, context=CONTEXT)
            events = await store.load_events(case.session_id)
            assert sum(event.type == EventType.SESSION_EXPORT_RETIRED for event in events) == 1
            await store.delete_session(case.session_id)
            assert await store.load(case.session_id) is None

    asyncio.run(run())


def test_pending_capacity_is_not_released_by_duplicate_replay(backend):
    async def run():
        async with harness(backend) as case:
            app, store, _, _ = case.app(
                limits=ExportLimits(max_exports=4, max_pending=1, max_retained_bytes=262144)
            )
            await case.create(store)
            req = await case.request(app)
            receipt = await app.export_session(req, context=CONTEXT)
            assert await app.export_session(req, context=CONTEXT) == receipt
            second = await case.request(app, "second")
            with pytest.raises(SessionExportCapacityExceeded):
                await app.export_session(second, context=CONTEXT)
            assert (await app.lookup_session_export(second, context=CONTEXT)).status == "not_found"
            assert len(await published(store, case.session_id)) == 1

    asyncio.run(run())


@pytest.mark.parametrize("conflict_first", [False, True])
def test_release_requires_exact_receiving_acceptance_and_replays_without_lookup(
    backend, conflict_first
):
    async def run():
        async with harness(backend) as case:
            reader = AcceptanceReader()
            app, store, _, projector = case.app(readers=(reader,))
            await case.create(store)
            req = await case.request(app)
            receipt = await app.export_session(req, context=CONTEXT)
            settlement = SessionExportSettlementRequest(
                request=req,
                operation=req.ref.operation.model_copy(update={"caller_key": "release"}),
                mode="release",
            )
            if conflict_first:
                reader.conflict = True
                with pytest.raises(SessionExportConflict):
                    await app.settle_session_export(settlement, context=CONTEXT)
                assert not any(
                    event.type == EventType.SESSION_EXPORT_RELEASED
                    for event in await store.load_events(case.session_id)
                )
                with pytest.raises(SessionExportConflict):
                    await store.delete_session(case.session_id)
                reader.conflict = False
            released = await app.settle_session_export(settlement, context=CONTEXT)
            assert released.acceptance.export_receipt == receipt
            assert released.acceptance.receiving_owner == AUDIENCE
            assert released == await app.settle_session_export(settlement, context=CONTEXT)
            assert len(reader.calls) == (2 if conflict_first else 1)
            assert projector.calls == 1
            assert (
                sum(
                    event.type == EventType.SESSION_EXPORT_RELEASED
                    for event in await store.load_events(case.session_id)
                )
                == 1
            )
            conflicting = settlement.model_copy(update={"mode": "retire"})
            with pytest.raises(SessionExportConflict):
                await app.settle_session_export(conflicting, context=CONTEXT)
            await store.delete_session(case.session_id)
            assert await store.load(case.session_id) is None

    asyncio.run(run())


def test_cancellation_after_thread_dispatch_retains_owner_and_exact_retry(backend):
    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app()
            await case.create(store)
            req = await case.request(app)
            projector.release = threading.Event()
            task = asyncio.create_task(app.export_session(req, context=CONTEXT))
            retry = None
            try:
                assert await asyncio.to_thread(projector.entered.wait, 5)
                assert task.cancelling() == 0
                task.cancel("cancel-export")
                assert task.cancelling() == 1
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and task.cancelling() == 1
                retry = asyncio.create_task(app.export_session(req, context=CONTEXT))
                await asyncio.sleep(0.05)
                assert not retry.done()
                assert projector.calls == 1
                assert not await published(store, case.session_id)
                projector.release.set()
                receipt = await asyncio.wait_for(retry, 10)
                assert await app.export_session(req, context=CONTEXT) == receipt
                assert projector.calls == 1
                assert len(await published(store, case.session_id)) == 1
            finally:
                projector.release.set()
                await asyncio.gather(
                    task, *([] if retry is None else [retry]), return_exceptions=True
                )

    asyncio.run(run())


def test_secret_output_rejected_before_publication(backend, caplog, capsys):
    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app(redactor=SecretRedactor(SECRET))
            projector.secret = True
            await case.create(store)
            req = await case.request(app)
            with pytest.raises(Exception) as error:
                await app.export_session(req, context=CONTEXT)
            assert SECRET not in str(error.value) + repr(error.value)
            assert projector.calls == 1
            assert not await published(store, case.session_id)
            assert (await app.lookup_session_export(req, context=CONTEXT)).status == "not_found"
            assert SECRET not in str(await store.load_checkpoint(case.session_id))
            projector.secret = False
            await app.export_session(req, context=CONTEXT)
            assert await app.read_session_export(req, context=CONTEXT) == {"count": 5}
            assert len(await published(store, case.session_id)) == 1
        assert SECRET not in caplog.text
        captured = capsys.readouterr()
        assert SECRET not in captured.out + captured.err

    asyncio.run(run())


@pytest.mark.parametrize("change", ["session_instance", "selected_source"])
def test_source_change_after_projection_dispatch_rejects_stale_publication(backend, change):
    if change == "selected_source" and backend[0] != "sqlite":
        pytest.skip("Only SQLiteSessionStore exposes public transcript compaction")

    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app()
            await case.create(store)
            req = await case.request(app)
            projector.release = threading.Event()
            task = asyncio.create_task(app.export_session(req, context=CONTEXT))
            try:
                assert await asyncio.to_thread(projector.entered.wait, 5)
                if change == "session_instance":
                    await store.delete_session(case.session_id)
                    await case.create(store)
                    assert (
                        await store.load(case.session_id)
                    ).instance_id != req.ref.session_instance_id
                else:
                    assert await store.compact_transcript(case.session_id, keep_last=0) == 1
                    assert not (
                        await store.load_transcript_window(case.session_id, start_index=0, limit=1)
                    ).records
                projector.release.set()
                with pytest.raises((SessionExportConflict, SessionExportUnavailable)):
                    await asyncio.wait_for(task, 10)
                assert not await published(store, case.session_id)
                if change == "selected_source":
                    assert (
                        await app.lookup_session_export(req, context=CONTEXT)
                    ).status == "not_found"
            finally:
                projector.release.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_raw_checkpoint_and_operation_record_cannot_mint_export_authority(backend):
    async def run():
        async with harness(backend) as case:
            app, store, _, _ = case.app()
            await case.create(store)
            await case.request(app)
            before = await store.load_checkpoint(case.session_id)
            with pytest.raises(SessionExportConflict):
                await store.checkpoint(case.session_id, {"session_exports": {"forged": True}})
            assert await store.load_checkpoint(case.session_id) == before

            def forge(_session, checkpoint, _record):
                return SessionOperationPublication(
                    checkpoint=checkpoint or {},
                    operation_records={"session-export:forged": {"forged": True}},
                )

            with pytest.raises(SessionExportConflict):
                await store.publish_session_operation(
                    case.session_id,
                    idempotency_key="session-export:forged",
                    operation_transform=forge,
                    events=[],
                )
            with pytest.raises(SessionExportConflict):
                await store.load_session_operation(case.session_id, "session-export:namespace")
            assert await store.load_checkpoint(case.session_id) == before
            assert not await published(store, case.session_id)

    asyncio.run(run())


@pytest.mark.parametrize(
    "field",
    [
        "issuer",
        "principal",
        "authorization_issuer",
        "authorization_principal",
        "max_exports",
        "max_pending",
        "max_retained_bytes",
        "payload",
        "state",
        "settlement",
        "invocation",
        "interaction",
        "reserved_bytes",
    ],
)
def test_public_replay_rejects_corrupt_exact_receipt_without_reexecution(backend, field):
    base = {
        "memory": InMemorySessionStore,
        "sqlite": SQLiteSessionStore,
        "postgres": PostgresSessionStore,
    }[backend[0]]

    class CorruptReadStore(base):
        # Native ownership/read resolution is unchanged. Only the detached
        # returned export record is corrupted, after the real owner read.
        session_export_version = 1
        corrupt = False
        corrupted_reads = 0
        terminal = None

        async def load_session_operation(self, *args, **kwargs):
            raw = await super().load_session_operation(*args, **kwargs)
            if not self.corrupt or not isinstance(raw, dict) or "receipt" not in raw:
                return raw
            detached = deepcopy(raw)
            self.corrupted_reads += 1
            expected = detached["receipt"]["expected"]
            if field == "issuer":
                expected["initiator"]["issuer"]["incarnation"] = "corrupt"
            elif field == "principal":
                expected["initiator"]["principal"] = "mallory"
            elif field == "authorization_issuer":
                expected["intent"]["authorization"]["issuer"]["incarnation"] = "corrupt"
            elif field == "authorization_principal":
                expected["intent"]["authorization"]["principal"] = "mallory"
            elif field.startswith("max_"):
                expected["intent"]["limits"][field] -= 1
            elif field == "payload":
                detached["payload_json"] = '{"count":6}'
            elif field == "state":
                detached["state"] = "retired"
            elif field == "settlement":
                detached["settlement"] = deepcopy(self.terminal)
            elif field == "invocation":
                expected["initiator"]["invocation_id"] = "forged-invocation"
            elif field == "interaction":
                expected["initiator"]["interaction_id"] = "forged-interaction"
            else:
                detached["reserved_bytes"] -= 1
            return detached

    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app(store_type=CorruptReadStore)
            await case.create(store)
            req = await case.request(app)
            receipt = await app.export_session(req, context=CONTEXT)
            baseline = await store.load_events(case.session_id)
            store.terminal = {
                "request": SessionExportSettlementRequest(
                    request=req,
                    operation=req.ref.operation.model_copy(update={"caller_key": "retire"}),
                    mode="retire",
                ).model_dump(mode="json"),
                "initiator": receipt.expected.initiator.model_dump(mode="json"),
                "acceptance": None,
                "event_id": "forged-retirement",
            }
            store.corrupt = True
            for entrance in (app.export_session, app.read_session_export):
                with pytest.raises((SessionExportConflict, SessionExportUnavailable)):
                    await entrance(req, context=CONTEXT)
            try:
                lookup = await app.lookup_session_export(req, context=CONTEXT)
            except (SessionExportConflict, SessionExportUnavailable):
                pass
            else:
                assert lookup.status in {"conflict", "unavailable"}
            assert store.corrupted_reads == 3
            assert projector.calls == 1
            assert await store.load_events(case.session_id) == baseline
            store.corrupt = False
            assert await app.export_session(req, context=CONTEXT) == receipt
            assert await app.read_session_export(req, context=CONTEXT) == {"count": 5}
            assert projector.calls == 1
            assert await store.load_events(case.session_id) == baseline

    asyncio.run(run())


def test_pending_same_key_changed_intent_is_typed_conflict(backend):
    async def run():
        async with harness(backend) as case:
            app, store, _, projector = case.app()
            await case.create(store)
            req = await case.request(app)
            projector.release = threading.Event()
            original = asyncio.create_task(app.export_session(req, context=CONTEXT))
            try:
                assert await asyncio.to_thread(projector.entered.wait, 5)
                changed = req.model_copy(update={"source_indices": ()})
                with pytest.raises(SessionExportConflict):
                    await asyncio.wait_for(app.export_session(changed, context=CONTEXT), 2)
                assert not original.done()
                assert projector.calls == 1
                assert not await published(store, case.session_id)
                projector.release.set()
                receipt = await asyncio.wait_for(original, 10)
                assert await app.export_session(req, context=CONTEXT) == receipt
                assert projector.calls == 1
                assert [event.id for event in await published(store, case.session_id)] == [
                    receipt.event_id
                ]
            finally:
                projector.release.set()
                await asyncio.gather(original, return_exceptions=True)

    asyncio.run(run())
