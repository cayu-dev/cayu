"""Public export smoke coverage and ownership-attestation regressions."""

import asyncio
from contextlib import asynccontextmanager, nullcontext
from copy import deepcopy
from uuid import uuid4

import pytest
from tests.core.test_browser_control_publisher import publication_fixture
from tests.core.test_session_export_contracts import limits, object_ref, owner
from tests.core.test_session_export_snapshot import _store

from cayu.applications import CayuApp
from cayu.collaboration._contracts import OperationRef
from cayu.collaboration._session_export_store import ROOT_KEY, read_scope
from cayu.collaboration.exports import (
    SessionExportAccessContext,
    SessionExportAuthorization,
    SessionExportConflict,
    SessionExportPolicy,
    SessionExportProjector,
    SessionExportRef,
    SessionExportRegistration,
    SessionExportRequest,
)
from cayu.messages import Message
from cayu.runtime._checkpoint_store import (
    _RuntimeCheckpointSessionStore,
    runtime_checkpoint_session_store,
)
from cayu.sessions.base import (
    _SESSION_EXPORT_OWNER_METHODS,
    InMemorySessionStore,
    RunRequest,
    SessionIdentity,
)
from cayu.storage.sqlite import SQLiteSessionStore


class _Policy(SessionExportPolicy):
    ref = object_ref("policy")

    @asynccontextmanager
    async def acquire(self, context, **kwargs):
        yield SessionExportAuthorization(
            issuer=owner(),
            principal=context.principal,
            policy=self.ref,
            revision=1,
            expires_at_ms=9_999_999_999_999,
        )


class _Projector(SessionExportProjector):
    ref = object_ref("projector")

    def __init__(self):
        self.calls = 0

    def project(self, source):
        self.calls += 1
        assert tuple(row.index for row in source) == (0,)
        return {"approved": True}

    def validate(self, source, output, audience):
        return output == {"approved": True}


@pytest.mark.parametrize("publication_kind", ["plain", "guarded", "store_time"])
@pytest.mark.parametrize("export_read", [False, True])
def test_memory_operation_view_changes_only_export_root(tmp_path, publication_kind, export_read):
    class ObservedCheckpoint(Exception):
        pass

    async def scenario():
        raw = InMemorySessionStore()
        async with publication_fixture("memory", tmp_path, raw_store=raw) as (wrapped, command):
            with command.scope():
                await wrapped.publish_session_operation_guarded_with_store_time(
                    "session",
                    idempotency_key=command.storage_key,
                    operation_transform=command.transform,
                    commit_guard=lambda: None,
                    commit_time_guard=command.validate_commit_time,
                    events=[],
                )
            app = CayuApp(
                session_store=raw,
                enable_logging=False,
                session_exports=SessionExportRegistration(
                    owner=owner(),
                    policy=_Policy(),
                    projectors=(),
                    limits=limits(),
                ),
            )
            try:
                await app.initialize_session_exports(
                    "session", context=SessionExportAccessContext(principal="principal")
                )
                retained = await raw.load_checkpoint("session")
                assert "browser_controls" in retained
                assert "active_invocation_execution_profile" in retained
                assert ROOT_KEY in retained
                expected = deepcopy(retained)
                if not export_read:
                    expected.pop(ROOT_KEY)

                def observe(_session, checkpoint, _record, *clock):
                    assert checkpoint == expected
                    checkpoint["browser_controls"].clear()
                    checkpoint.clear()
                    raise ObservedCheckpoint()

                kwargs = dict(idempotency_key="view-test", operation_transform=observe, events=[])
                if publication_kind == "plain":
                    publish = raw.publish_session_operation
                elif publication_kind == "guarded":
                    publish = raw.publish_session_operation_guarded
                    kwargs["commit_guard"] = lambda: None
                else:
                    publish = raw.publish_session_operation_guarded_with_store_time
                    kwargs["commit_guard"] = lambda: None
                    kwargs["commit_time_guard"] = lambda now: None
                scope = read_scope("session") if export_read else nullcontext()
                with scope, pytest.raises(ObservedCheckpoint):
                    await publish("session", **kwargs)
                assert await raw.load_checkpoint("session") == retained
            finally:
                await app.drain_session_exports()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_public_export_publication_replay_and_pending_erasure(backend, tmp_path, request):
    async def scenario():
        store = _store(backend, tmp_path, request)
        projector = _Projector()
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            session_exports=SessionExportRegistration(
                owner=owner(),
                policy=_Policy(),
                projectors=(projector,),
                limits=limits(),
            ),
        )
        try:
            assert store._supports_session_export_protocol()
            assert runtime_checkpoint_session_store(store)._supports_session_export_protocol()
            session = await store.create(
                RunRequest(session_id=uuid4().hex, agent_name="assistant", messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake"),
            )
            await store.append_transcript_messages(session.id, [Message.text("user", "private")])
            context = SessionExportAccessContext(principal="principal")
            namespace = await app.initialize_session_exports(session.id, context=context)
            command = SessionExportRequest(
                ref=SessionExportRef(
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    operation=OperationRef(
                        application_scope=namespace.owner.application_scope,
                        namespace_incarnation=namespace.namespace_incarnation,
                        generation=namespace.generation,
                        caller_key="export",
                    ),
                ),
                source_indices=(0,),
                audience=owner(),
                projector=projector.ref,
                policy=_Policy.ref,
            )
            receipt = await app.export_session(command, context=context)
            assert await app.export_session(command, context=context) == receipt
            assert projector.calls == 1
            assert await app.read_session_export(command, context=context) == {"approved": True}
            with pytest.raises(SessionExportConflict):
                await store.delete_session(session.id)
            assert await store.load(session.id) is not None
        finally:
            await app.drain_session_exports()
            close = getattr(store, "close", None)
            if close is not None:
                await close()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", _SESSION_EXPORT_OWNER_METHODS)
def test_export_capability_requires_override_reattestation(method):
    async def override(*args, **kwargs):
        raise AssertionError("Capability discovery must not invoke an override.")

    store_type = type("UnqualifiedStore", (InMemorySessionStore,), {method: override})
    store = store_type()
    assert not store._supports_session_export_protocol()
    assert runtime_checkpoint_session_store(store).session_export_version == 0
    attested_type = type("AttestedStore", (store_type,), {"session_export_version": 1})
    assert attested_type()._supports_session_export_protocol()


def test_export_capability_rejects_instance_override_and_boolean_version():
    store = InMemorySessionStore()
    store.delete_session = lambda *args, **kwargs: None
    assert not store._supports_session_export_protocol()
    boolean_type = type("BooleanStore", (InMemorySessionStore,), {"session_export_version": True})
    assert not boolean_type()._supports_session_export_protocol()


def test_native_sqlite_connection_resource_is_not_a_method_override(tmp_path):
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "capability.sqlite")
        try:
            assert "_connection" in vars(store)
            assert store._supports_session_export_protocol()
            assert runtime_checkpoint_session_store(store).session_export_version == 1
            # Real method shadowing still fails closed, including noncallables.
            store._run_write = None
            assert not store._supports_session_export_protocol()
            assert runtime_checkpoint_session_store(store).session_export_version == 0
            del store._run_write
            assert store._supports_session_export_protocol()
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("method", _SESSION_EXPORT_OWNER_METHODS)
def test_export_wrapper_capability_requires_override_reattestation(method):
    async def override(*args, **kwargs):
        raise AssertionError("Capability discovery must not invoke an override.")

    wrapper_type = type("UnqualifiedWrapper", (_RuntimeCheckpointSessionStore,), {method: override})
    wrapper = wrapper_type(InMemorySessionStore())
    assert not wrapper._supports_session_export_protocol()
    assert wrapper.session_export_version == 0
