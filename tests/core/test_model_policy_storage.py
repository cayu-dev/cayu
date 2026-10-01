import asyncio
import json
import threading
from pathlib import Path
from uuid import uuid4

import pytest

from cayu.runtime._policy_storage import (
    InMemoryModelPolicyStore,
    ModelPolicyStore,
    PolicyStorageCommand,
    PolicyStorageView,
    binding_bytes,
    state_bytes,
)
from cayu.runtime._policy_wire import PolicyContractError

pytestmark = pytest.mark.anyio
SCOPE = json.loads((Path(__file__).parents[1] / "fixtures/model_policy/contract.json").read_text())[
    "effective"
]["scope"]
BINDING = binding_bytes(
    {"scope": SCOPE, "agent_name": "assistant", "provider_name": "cayu_gateway"}
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_binding(monkeypatch):
    scope = {**SCOPE, "instance_id": uuid4().hex}
    monkeypatch.setitem(
        globals(),
        "BINDING",
        binding_bytes({"scope": scope, "agent_name": "assistant", "provider_name": "cayu_gateway"}),
    )


@pytest.fixture(params=["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def factory(request, tmp_path):
    if request.param == "memory":
        store = InMemoryModelPolicyStore()
        return lambda: store
    if request.param == "sqlite":
        from cayu.storage.model_policy_sqlite import SQLiteModelPolicyStore

        return lambda: SQLiteModelPolicyStore(tmp_path / "policy.sqlite")
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.model_policy_postgres import PostgresModelPolicyStore

    dsn = request.getfixturevalue("postgres_dsn")
    return lambda: PostgresModelPolicyStore(dsn, schema_mode=SchemaMode.CREATE)


def command(action, owner="one", **kw):
    return PolicyStorageCommand(action, BINDING, owner, **kw)


async def test_atomic_policy_state_lease_cas_and_reopen(factory):
    store = factory()
    try:
        initial = await store.execute(command("claim"))
        assert initial.state == b"{}" and initial.revision == 0
        assert 0 < initial.lease_remaining_seconds <= 60
        with pytest.raises(PolicyContractError):
            await store.execute(command("claim", "other"))
        body = state_bytes({"default": "model-a", "pending": ["report-1"]})
        updated = await store.execute(command("write", expected=initial, state=body))
        assert updated.revision == 1 and updated.state == body
        with pytest.raises(PolicyContractError):
            await store.execute(command("write", expected=initial, state=b"{}"))
        await store.execute(command("release"))
    finally:
        await store.close()
    reopened = factory()
    try:
        recovered = await reopened.execute(command("claim", "other"))
        assert recovered.revision == 1 and recovered.state == body
        with pytest.raises(PolicyContractError):
            await reopened.execute(command("write", expected=updated, state=b"{}"))
        await reopened.execute(command("release", "other"))
    finally:
        await reopened.close()


async def test_one_of_two_competing_owners_wins(factory):
    store = factory()
    try:
        results = await asyncio.gather(
            store.execute(command("claim")),
            store.execute(command("claim", "other")),
            return_exceptions=True,
        )
        assert sum(isinstance(result, PolicyStorageView) for result in results) == 1
        assert sum(isinstance(result, PolicyContractError) for result in results) == 1
    finally:
        await store.close()


async def test_same_owner_replay_does_not_extend_lease(factory):
    store = factory()
    try:
        initial = await store.execute(command("claim"))
        replay = await store.execute(command("claim"))
        assert replay.revision == initial.revision and replay.state == initial.state
        assert 0 < replay.lease_remaining_seconds <= initial.lease_remaining_seconds
        renewed = await store.execute(command("renew"))
        assert renewed.lease_remaining_seconds == 60
        await store.execute(command("release"))
        with pytest.raises(PolicyContractError):
            await store.execute(command("read"))
    finally:
        await store.close()


async def test_release_replay_never_releases_replacement_owner(factory):
    store = factory()
    try:
        await store.execute(command("claim"))
        await store.execute(command("release"))
        await store.execute(command("release"))
        await store.execute(command("claim", "replacement"))
        await store.execute(command("release"))
        assert (await store.execute(command("read", "replacement"))).lease_remaining_seconds > 0
        with pytest.raises(PolicyContractError):
            await store.execute(command("claim", "competitor"))
    finally:
        await store.close()


@pytest.mark.parametrize(
    "backend", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
)
async def test_application_store_assembly_owns_policy_store(
    backend, request, tmp_path, monkeypatch
):
    from cayu import open_application_stores

    monkeypatch.delenv("CAYU_REQUIRE_POSTGRES", raising=False)
    dsn = None
    if backend == "postgres":
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.model_policy_postgres import PostgresModelPolicyStore

        dsn = request.getfixturevalue("postgres_dsn")
        initialized = PostgresModelPolicyStore(dsn, schema_mode=SchemaMode.CREATE)
        await initialized.execute(command("release"))
        await initialized.close()
        dsn = request.getfixturevalue("postgres_url")
        monkeypatch.setenv("CAYU_DATABASE_URL", dsn)
    else:
        monkeypatch.delenv("CAYU_DATABASE_URL", raising=False)
    stores = open_application_stores(
        dsn, sqlite_path=tmp_path / "assembled.sqlite", model_policy=True
    )
    try:
        policy = stores.model_policy_store
        assert policy is not None
        initial = await policy.execute(command("claim"))
        assert initial.state == b"{}"
        if backend == "postgres":
            assert policy._pool is stores.session_store._pool
        await policy.execute(command("release"))
    finally:
        await stores.close()


async def test_one_scope_cannot_create_a_second_sequence_by_changing_local_mapping(factory):
    store = factory()
    try:
        await store.execute(command("claim"))
        await store.execute(command("release"))
        for field, changed in (("agent_name", "other-agent"), ("provider_name", "other-provider")):
            binding = json.loads(BINDING)
            binding[field] = changed
            with pytest.raises(PolicyContractError):
                await store.execute(PolicyStorageCommand("claim", binding_bytes(binding), "other"))
    finally:
        await store.close()


async def test_memory_expiry_fences_stale_owner(monkeypatch):
    now = 1000.0
    monkeypatch.setattr("cayu.runtime._policy_storage.time.time", lambda: now)
    store = InMemoryModelPolicyStore()
    initial = await store.execute(command("claim"))
    now += 60
    with pytest.raises(PolicyContractError):
        await store.execute(command("renew"))
    await store.execute(command("claim", "replacement"))
    with pytest.raises(PolicyContractError):
        await store.execute(command("write", expected=initial, state=b"{}"))


async def test_sqlite_cancel_keeps_connection_owned_until_publication_settles(tmp_path):
    from cayu.storage.model_policy_sqlite import SQLiteModelPolicyStore

    store = SQLiteModelPolicyStore(tmp_path / "policy.sqlite")
    initial = await store.execute(command("claim"))
    reached = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def pause_write(sql):
        if sql.startswith("INSERT INTO cayu_model_policy_state"):
            loop.call_soon_threadsafe(reached.set)
            assert release.wait(10)

    store._connection.set_trace_callback(pause_write)
    writer = asyncio.create_task(
        store.execute(command("write", expected=initial, state=state_bytes({"pending": [1]})))
    )
    reader = None
    try:
        await asyncio.wait_for(reached.wait(), 5)
        writer.cancel()
        assert writer.cancelling() == 1
        reader = asyncio.create_task(store.execute(command("read")))
        await asyncio.sleep(0)
        assert not writer.done() and not reader.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await writer
        assert writer.cancelled() and writer.cancelling() == 1
        recovered = await reader
        assert recovered.revision == 1
        assert recovered.state == state_bytes({"pending": [1]})
    finally:
        release.set()
        await asyncio.gather(writer, *([] if reader is None else [reader]), return_exceptions=True)
        store._connection.set_trace_callback(None)
        await store.close()


async def test_owner_fences_acknowledgement_loss_and_recovers_complete_installation(factory):
    from cayu.runtime._policy_installation import PolicyInstallationOwner
    from cayu.runtime._policy_wire import canonical, decode

    vectors = json.loads(
        (Path(__file__).parents[1] / "fixtures/model_policy/contract.json").read_text()
    )
    binding = decode(BINDING)
    wire = canonical({**vectors["report"], "scope": binding["scope"], "operation_id": "policy-1"})
    store = factory()

    class LostAcknowledgement(ModelPolicyStore):
        fail = True

        async def execute(self, command):
            result = await store.execute(command)
            if command.action == "write" and self.fail:
                self.fail = False
                raise OSError("simulated acknowledgement loss")
            return result

    owner = PolicyInstallationOwner(
        LostAcknowledgement(), binding=binding, incarnation=("incarnation-1", 1)
    )
    try:
        await owner.start()
        with pytest.raises(OSError, match="acknowledgement loss"):
            await owner.install(wire)
        with pytest.raises(PolicyContractError):
            owner.current()
        await owner.reconcile()
        assert owner.current() == wire and owner.pending() == (wire,)
        await owner.close()
    finally:
        await store.close()
    reopened = factory()
    replacement = PolicyInstallationOwner(
        reopened, binding=binding, incarnation=("incarnation-1", 1)
    )
    try:
        await replacement.start()
        assert replacement.current() == wire and replacement.pending() == (wire,)
        await replacement.close()
    finally:
        await reopened.close()


async def test_owner_cancellation_after_commit_preserves_pending_and_cancellation(factory):
    from cayu.runtime._policy_installation import PolicyInstallationOwner
    from cayu.runtime._policy_wire import canonical, decode

    vectors = json.loads(
        (Path(__file__).parents[1] / "fixtures/model_policy/contract.json").read_text()
    )
    binding = decode(BINDING)
    wire = canonical({**vectors["report"], "scope": binding["scope"], "operation_id": "policy-1"})
    store = factory()
    committed = asyncio.Event()

    class DelayedAcknowledgement(ModelPolicyStore):
        async def execute(self, command):
            result = await store.execute(command)
            if command.action == "write":
                committed.set()
                await asyncio.Event().wait()
            return result

    owner = PolicyInstallationOwner(
        DelayedAcknowledgement(), binding=binding, incarnation=("incarnation-1", 1)
    )
    task = None
    try:
        await owner.start()
        task = asyncio.create_task(owner.install(wire))
        await asyncio.wait_for(committed.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
        with pytest.raises(PolicyContractError):
            owner.current()
        await owner.reconcile()
        assert owner.current() == wire and owner.pending() == (wire,)
        await owner.close()
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await store.close()
