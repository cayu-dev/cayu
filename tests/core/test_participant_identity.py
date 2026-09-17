"""Real SDK identity conformance across the three collaboration owners."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import warnings
from contextlib import asynccontextmanager
from dataclasses import replace
from uuid import uuid4

import pytest

from cayu import CayuApp
from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExactMatch,
)
from cayu.collaboration.access import (
    CollaborationAccessContext,
    CollaborationAccessDenied,
    CollaborationAccessGrant,
    CollaborationAccessPolicy,
    CollaborationRegistration,
)
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.participants import (
    CollaborationBootstrap,
    CollaborationCapacityExceeded,
    CollaborationLimits,
    CollaborationNotInitialized,
    CollaborationUnavailable,
    ParticipantAliasChange,
    ParticipantConfiguration,
    ParticipantConfigurationRef,
    ParticipantConfigure,
    ParticipantCreate,
    ParticipantReceipt,
    ParticipantRef,
)
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio
CONTEXT = CollaborationAccessContext(principal="operator")


@pytest.mark.parametrize("entrance", ["inspect", "discover", "events", "alias"])
@pytest.mark.parametrize("close_fails", [False, True])
@pytest.mark.parametrize("read_fails", [False, True])
async def test_sqlite_read_cancellation_preserves_safe_rollback_failure(
    tmp_path, entrance, close_fails, read_fails, caplog, capsys
):
    store = SQLiteCollaborationStore(tmp_path / "read-cancellation.sqlite")
    application = app(store, registration(), secret_redactor=SecretRedactor("private-canary"))
    initialized = await application.initialize_collaboration()
    _, receipt = await create(application, initialized, alias="reviewer")
    connection = store._connection
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()

    class Connection:
        def __getattr__(self, name):
            return getattr(connection, name)

        def execute(self, sql, *args):
            result = connection.execute(sql, *args)
            if read_fails and sql.startswith("SELECT"):
                raise ConnectionError("read private-canary")
            return result

        def rollback(self):
            raise ExceptionGroup("private-canary", [OSError("rollback private-canary")])

        def close(self):
            connection.close()
            if close_fails:
                raise RuntimeError("close private-canary")

    def trace(sql):
        if sql.startswith("SELECT"):
            loop.call_soon_threadsafe(entered.set)
            release.wait(10)

    connection.set_trace_callback(trace)
    store._connection = Connection()
    operations = {
        "inspect": lambda: application.inspect_participant(
            receipt.participants[0].reference, context=CONTEXT
        ),
        "discover": lambda: application.discover_participants(context=CONTEXT),
        "events": lambda: application.list_participant_events(context=CONTEXT),
        "alias": lambda: application.resolve_participant_alias("reviewer", context=CONTEXT),
    }
    caller = asyncio.create_task(operations[entrance]())
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            await asyncio.wait_for(entered.wait(), 2)
            caller.cancel("private-canary")
            caller.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await caller
        assert caller.cancelled() and caller.cancelling() == 2
        error = caught.value
        assert error.__context__ is None
        assert isinstance(error.__cause__, ExceptionGroup)
        failures = error.__cause__.exceptions
        if read_fails:
            assert type(failures[0]) is ConnectionError
            assert "read" in str(failures[0])
        nested = failures[int(read_fails)]
        assert isinstance(nested, ExceptionGroup)
        assert type(nested.exceptions[0]) is OSError
        assert len(failures) == 1 + int(read_fails) + int(close_fails)
        if close_fails:
            assert type(failures[-1]) is RuntimeError
        pending = [error]
        graph = []
        while pending:
            value = pending.pop()
            graph.append(value)
            assert value.__context__ is None
            if value.__cause__ is not None:
                pending.append(value.__cause__)
            if isinstance(value, BaseExceptionGroup):
                pending.extend(value.exceptions)
        assert sum(isinstance(value, asyncio.CancelledError) for value in graph) == 1
        assert sum(type(value) is ConnectionError for value in graph) == int(read_fails)
        output = capsys.readouterr()
        assert "private-canary" not in (
            repr(graph)
            + caplog.text
            + output.out
            + output.err
            + repr([str(w.message) for w in recorded])
        )
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        store._connection = connection
        await store.close()


async def test_public_cancelled_group_preserves_shared_causal_failure_once(stores, monkeypatch):
    store = stores()
    application = app(store, registration(), secret_redactor=SecretRedactor("private-canary"))
    initialized = await application.initialize_collaboration()
    _, receipt = await create(application, initialized)
    entered = asyncio.Event()

    async def failed(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            read = ConnectionError("read private-canary")
            cancellation.__cause__ = ExceptionGroup("private-canary", [read])
            failure = BaseExceptionGroup(
                "private-canary",
                [
                    cancellation,
                    ExceptionGroup("cleanup", [read, OSError("rollback private-canary")]),
                ],
            )
        raise failure

    monkeypatch.setattr(store, "inspect", failed)
    caller = asyncio.create_task(
        application.inspect_participant(receipt.participants[0].reference, context=CONTEXT)
    )
    await asyncio.wait_for(entered.wait(), 2)
    caller.cancel("private-canary")
    caller.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await caller
    assert caller.cancelled() and caller.cancelling() == 2
    assert caught.value.__context__ is None
    evidence = caught.value.__cause__
    assert isinstance(evidence, ExceptionGroup)
    read_group, cleanup_group = evidence.exceptions
    assert isinstance(read_group, ExceptionGroup)
    assert isinstance(cleanup_group, ExceptionGroup)
    assert len(read_group.exceptions) == len(cleanup_group.exceptions) == 1
    assert type(read_group.exceptions[0]) is ConnectionError
    assert type(cleanup_group.exceptions[0]) is OSError
    for value in (
        evidence,
        read_group,
        cleanup_group,
        *read_group.exceptions,
        *cleanup_group.exceptions,
    ):
        assert value.__context__ is None and value.__cause__ is None
        assert "private-canary" not in str(value)


@pytest.mark.parametrize("case", ["child", "historical", "fatal"])
async def test_public_read_group_distinguishes_cancellation_and_fatal_signals(
    stores, monkeypatch, case
):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, receipt = await create(application, initialized)
    entered = asyncio.Event()

    class FatalSignal(BaseException):
        pass

    fatal_group = None

    async def failed(*args, **kwargs):
        nonlocal fatal_group
        if case == "child":
            raise BaseExceptionGroup(
                "dependency", [asyncio.CancelledError(), RuntimeError("cleanup")]
            )
        if case == "historical":
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as old:
                failure = RuntimeError("later independent failure")
                failure.__cause__ = old
            raise ExceptionGroup("later", [failure])
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            fatal_group = BaseExceptionGroup("fatal", [cancellation, FatalSignal()])
        raise fatal_group

    monkeypatch.setattr(store, "inspect", failed)
    caller = asyncio.create_task(
        application.inspect_participant(receipt.participants[0].reference, context=CONTEXT)
    )
    if case == "fatal":
        await asyncio.wait_for(entered.wait(), 2)
        caller.cancel()
        with pytest.raises(BaseExceptionGroup) as caught:
            await caller
        assert caught.value is fatal_group
        assert caller.cancelling() == 1
    else:
        with pytest.raises(CollaborationUnavailable):
            await caller
        assert caller.cancelling() == (1 if case == "historical" else 0)
    assert not caller.cancelled()


@pytest.mark.parametrize("restricted", [False, True])
async def test_large_valid_participant_and_event_pages_traverse_completely(stores, restricted):
    from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES
    from cayu.collaboration._preparation import contract_bytes

    policy = Policy()
    config = ParticipantConfiguration(
        definition=ParticipantConfigurationRef(name="d" * 512, version=1),
        routing=ParticipantConfigurationRef(name="r" * 512, version=1),
        admission=ParticipantConfigurationRef(name="a" * 512, version=1),
    )
    reg = registration(scope=uuid4().hex + "s" * 480, policy=policy)
    reg = replace(
        reg,
        bootstrap=reg.bootstrap.model_copy(update={"owner_name": "o" * 512}),
        configurations=(config,),
    )
    application = app(stores(), reg)
    initialized = await application.initialize_collaboration()
    refs, events = [], []
    for index in range(32):
        receipt = await application.create_participant(
            ParticipantCreate(
                operation=initialized.operation(f"large-{index}"), configuration=config
            ),
            context=CONTEXT,
        )
        refs.append(receipt.participants[0].reference)
        events.append(receipt.event)
    previous = None
    for index in range(32):
        target = refs[index % 2]
        receipt = await application.change_participant_alias(
            ParticipantAliasChange(
                operation=initialized.operation(f"alias-{index}"),
                alias="reviewer",
                expected_alias_revision=index,
                expected_target=previous,
                target=target,
            ),
            context=CONTEXT,
        )
        previous = target
        events.append(receipt.event)
    if restricted:
        policy.allowed = tuple(refs)
    for event_page in (False, True):
        observed, counts, cursor = [], [], None
        for _ in range(100):
            page = (
                await application.list_participant_events(context=CONTEXT, cursor=cursor)
                if event_page
                else await application.discover_participants(context=CONTEXT, cursor=cursor)
            )
            assert len(contract_bytes(page, redactor=SecretRedactor())) <= MAX_ENVELOPE_BYTES
            records = page.events if event_page else page.participants
            observed.extend(
                record.sequence if event_page else record.reference.participant_id
                for record in records
            )
            if page.next_cursor is not None:
                assert records
                counts.append(len(records))
            cursor = page.next_cursor
            if cursor is None:
                break
        else:
            pytest.fail("Pagination did not make progress")
        expected = (
            ([1] if not restricted else []) + [event.sequence for event in events]
            if event_page
            else sorted(ref.participant_id for ref in refs)
        )
        assert observed == expected
        assert any(count < 32 for count in counts)
    if restricted:
        # A valid selection can leave too little space for even one record
        # plus the continuation. Refuse explicitly rather than looping empty.
        while len(policy.allowed) < 64:
            candidate = (
                *policy.allowed,
                ParticipantRef(
                    owner=initialized.owner,
                    participant_id=uuid4().hex,
                    incarnation=uuid4().hex,
                ),
            )
            try:
                grant = CollaborationAccessGrant(
                    application_scope=reg.bootstrap.application_scope, participants=candidate
                )
                contract_bytes(grant, redactor=SecretRedactor())
            except ValueError:
                break
            policy.allowed = candidate
        with pytest.raises(CollaborationUnavailable, match="page capacity"):
            await application.discover_participants(context=CONTEXT, limit=1)


@pytest.mark.parametrize("phase", ["read", "mutation", "close"])
@pytest.mark.parametrize("signal", ["timeout", "cancel"])
async def test_sqlite_close_retains_bounded_shutdown(tmp_path, phase, signal):
    store = SQLiteCollaborationStore(tmp_path / "shutdown.sqlite")
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    connection = store._connection
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    closes = 0

    def block():
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10), "test failed to release physical SQLite work"

    class Connection:
        def __getattr__(self, name):
            return getattr(connection, name)

        def close(self):
            nonlocal closes
            closes += 1
            if phase == "close":
                block()
            connection.close()

    store._connection = Connection()
    if phase in ("read", "mutation"):
        marker = "SELECT" if phase == "read" else "INSERT INTO cayu_collaboration_participants"
        connection.set_trace_callback(lambda sql: block() if sql.startswith(marker) else None)
        reader = asyncio.create_task(
            application.discover_participants(context=CONTEXT)
            if phase == "read"
            else create(application, initialized)
        )
    else:
        reader = None
    observer = None
    retry = None
    try:
        if reader is not None:
            await asyncio.wait_for(entered.wait(), 2)
        store._owners.observation_timeout = 0.03 if signal == "timeout" else 5
        observer = asyncio.create_task(store.close())
        if phase == "close":
            await asyncio.wait_for(entered.wait(), 2)
        if signal == "cancel":
            await asyncio.sleep(0)
            observer.cancel()
            observer.cancel()
        done, _ = await asyncio.wait((observer,), timeout=0.5)
        assert observer in done, "shutdown observation exceeded its bound"
        if signal == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled() and observer.cancelling() == 2
        else:
            with pytest.raises(CollaborationUnavailable):
                await observer
        store._owners.observation_timeout = 0.03
        retry = asyncio.create_task(store.close())
        done, _ = await asyncio.wait((retry,), timeout=0.5)
        assert retry in done, "retry observation exceeded its bound"
        with pytest.raises(CollaborationUnavailable):
            await retry
        assert closes == (1 if phase == "close" else 0)
        with pytest.raises(CollaborationUnavailable):
            await application.discover_participants(context=CONTEXT)
    finally:
        release.set()
        if reader is not None:
            await asyncio.wait_for(reader, 5)
        if observer is not None:
            await asyncio.gather(observer, return_exceptions=True)
        if retry is not None:
            await asyncio.gather(retry, return_exceptions=True)
        store._owners.observation_timeout = 5
        await store.close()
    await store.close()
    assert closes == 1
    if phase == "mutation":
        reopened = SQLiteCollaborationStore(tmp_path / "shutdown.sqlite")
        try:
            recovered = app(reopened, reg)
            await recovered.initialize_collaboration()
            assert (await create(recovered, initialized))[1].event.sequence == 2
            assert len((await recovered.list_participant_events(context=CONTEXT)).events) == 2
        finally:
            await reopened.close()


async def test_sqlite_failed_close_can_be_retried(tmp_path):
    store = SQLiteCollaborationStore(tmp_path / "close-retry.sqlite")
    connection = store._connection
    closes = 0

    class Connection:
        def close(self):
            nonlocal closes
            closes += 1
            if closes == 1:
                raise OSError("close failed")
            connection.close()

    store._connection = Connection()
    try:
        with pytest.raises(OSError, match="close failed"):
            await store.close()
        await asyncio.gather(store.close(), store.close())
        await store.close()
        assert closes == 2
    finally:
        connection.close()


@pytest.mark.parametrize("signal", ["cancel", "timeout"])
async def test_sqlite_write_contention_keeps_observation_responsive(tmp_path, signal):
    path = tmp_path / "contended.sqlite"
    store = SQLiteCollaborationStore(path)
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    locker = sqlite3.connect(path)
    locker.execute("BEGIN IMMEDIATE")
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()

    def trace(statement):
        if statement == "BEGIN IMMEDIATE":
            loop.call_soon_threadsafe(entered.set)

    store._connection.set_trace_callback(trace)
    store._owners.observation_timeout = 0.1
    request = ParticipantCreate(
        operation=initialized.operation("contended"), configuration=configuration()
    )
    started = loop.time()
    caller = asyncio.create_task(application.create_participant(request, context=CONTEXT))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        # The SQLite busy handler is now waiting on a real second connection.
        await asyncio.sleep(0)
        assert loop.time() - started < 1
        if signal == "cancel":
            caller.cancel()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(caller, 1)
            assert caller.cancelled() and caller.cancelling() == 2
        else:
            with pytest.raises(CollaborationUnavailable):
                await asyncio.wait_for(caller, 1)
        assert store._owners.pending
        with pytest.raises(CollaborationUnavailable):
            await asyncio.wait_for(application.create_participant(request, context=CONTEXT), 1)
        assert len(store._owners.pending) == 2
    finally:
        locker.rollback()
        locker.close()
        await asyncio.gather(caller, return_exceptions=True)
        await asyncio.wait_for(asyncio.gather(*store._owners.pending), 5)
        store._connection.set_trace_callback(None)
        store._owners.observation_timeout = 10
        await store.close()

    reopened = SQLiteCollaborationStore(path)
    try:
        recovered = app(reopened, reg)
        await recovered.initialize_collaboration()
        receipt = await recovered.create_participant(request, context=CONTEXT)
        assert receipt.event.sequence == 2
        assert len((await recovered.list_participant_events(context=CONTEXT)).events) == 2
    finally:
        await reopened.close()


async def test_retry_readback_is_bounded_while_mutation_holds_transaction(stores, monkeypatch):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    entered, release = asyncio.Event(), asyncio.Event()
    original = store._transaction
    writes = 0

    @asynccontextmanager
    async def held(scope, *, write):
        nonlocal writes
        async with original(scope, write=write) as tx:
            yield tx
            if write:
                writes += 1
                entered.set()
                await release.wait()

    monkeypatch.setattr(store, "_transaction", held)
    request = ParticipantCreate(
        operation=initialized.operation("locked-retry"), configuration=configuration()
    )
    caller = asyncio.create_task(application.create_participant(request, context=CONTEXT))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert caller.cancelled() and caller.cancelling() == 1
        store._owners.observation_timeout = 0.03
        for _ in range(2):
            with pytest.raises(CollaborationUnavailable):
                await asyncio.wait_for(application.create_participant(request, context=CONTEXT), 2)
            assert len(store._owners.pending) == 2  # One mutation and one shared readback.
        assert writes == 1
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        await asyncio.wait_for(asyncio.gather(*store._owners.pending), 5)
        store._owners.observation_timeout = 10
    receipt = await application.create_participant(request, context=CONTEXT)
    assert receipt.event.sequence == 2
    assert writes == 1
    assert len((await application.list_participant_events(context=CONTEXT)).events) == 2


@pytest.mark.parametrize("entrance", ["_initialize", "_apply", "_lookup"])
async def test_settled_failure_survives_caller_cancellation(
    stores, monkeypatch, entrance, caplog, capsys
):
    store = stores()
    application = app(store, registration(), secret_redactor=SecretRedactor("private-value"))
    initialized = None
    if entrance != "_initialize":
        initialized = await application.initialize_collaboration()

    async def failed(*args, **kwargs):
        observer.cancel("private-value")
        observer.cancel()
        raise ExceptionGroup(
            "private-value",
            [
                ConnectionError("commit private-value"),
                ExceptionGroup("cleanup", [OSError("rollback private-value")]),
            ],
        )

    monkeypatch.setattr(store, entrance, failed)
    observer = asyncio.create_task(
        application.initialize_collaboration()
        if initialized is None
        else create(application, initialized)
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(asyncio.CancelledError) as caught:
            await observer
    assert observer.cancelled() and observer.cancelling() == 2
    error = caught.value
    assert error.__context__ is None
    assert isinstance(error.__cause__, ExceptionGroup)
    leaves = error.__cause__.exceptions
    assert len(leaves) == 2 and type(leaves[0]) is ConnectionError
    assert isinstance(leaves[1], ExceptionGroup)
    assert len(leaves[1].exceptions) == 1 and type(leaves[1].exceptions[0]) is OSError
    graph = [error, error.__cause__, leaves[0], leaves[1], leaves[1].exceptions[0]]
    assert all(item.__context__ is None for item in graph)
    assert all(item.__cause__ is None for item in graph[1:])
    output = capsys.readouterr()
    assert "private-value" not in (
        repr(graph)
        + caplog.text
        + output.out
        + output.err
        + repr([str(w.message) for w in recorded])
    )
    assert not store._owners.pending


async def test_successful_commit_collision_preserves_cancellation_and_replay(stores, monkeypatch):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    original = store._apply
    committed = []

    async def commit_then_cancel(*args, **kwargs):
        result = await original(*args, **kwargs)
        committed.append(result)
        observer.cancel()
        return result

    monkeypatch.setattr(store, "_apply", commit_then_cancel)
    request = ParticipantCreate(
        operation=initialized.operation("commit-cancellation"), configuration=configuration()
    )
    observer = asyncio.create_task(application.create_participant(request, context=CONTEXT))
    with pytest.raises(asyncio.CancelledError) as caught:
        await observer
    assert observer.cancelled() and observer.cancelling() == 1
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert await application.create_participant(request, context=CONTEXT) == committed[0]
    assert len(committed) == 1


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Policy(CollaborationAccessPolicy):
    def __init__(self):
        self.allowed = None
        self.denied = set()

    def authorize(self, context, *, application_scope, action):
        if context.principal != "operator" or action in self.denied:
            raise CollaborationAccessDenied("denied")
        return CollaborationAccessGrant(
            application_scope=application_scope, participants=self.allowed
        )


def configuration(version=1):
    return ParticipantConfiguration(
        definition=ParticipantConfigurationRef(name="reviewer", version=version),
        routing=ParticipantConfigurationRef(name="route", version=1),
        admission=ParticipantConfigurationRef(name="admission", version=1),
    )


def registration(*, scope=None, policy=None, limits=None):
    return CollaborationRegistration(
        bootstrap=CollaborationBootstrap(
            application_scope=scope or uuid4().hex,
            provisioning_scope="application",
            owner_name="participants",
            limits=limits
            or CollaborationLimits(
                participants=64,
                aliases=64,
                operations=128,
                events=256,
                retained_bytes=4 * 1024 * 1024,
                control_operations=8,
                control_events=8,
                control_bytes=65536,
                namespaces=4,
                generations=8,
                obligations=64,
            ),
        ),
        access_policy=policy or Policy(),
        configurations=(configuration(), configuration(2)),
    )


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def stores(request, tmp_path):
    opened = []
    backend = request.param
    address = str(tmp_path / "participants.sqlite") if backend == "sqlite" else None
    if backend == "postgres":
        address = request.getfixturevalue("postgres_dsn")

    def factory():
        if backend == "memory":
            if opened:
                return opened[0]
            value = InMemoryCollaborationStore()
        elif backend == "sqlite":
            value = SQLiteCollaborationStore(address)
        else:
            value = PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
        opened.append(value)
        return value

    yield factory
    for value in opened:
        await value.close()


def app(store, reg, **kwargs):
    return CayuApp(collaboration_store=store, collaboration=reg, enable_logging=False, **kwargs)


async def create(application, initialized, key="create", **kwargs):
    request = ParticipantCreate(
        operation=initialized.operation(key), configuration=configuration(), **kwargs
    )
    return request, await application.create_participant(request, context=CONTEXT)


async def test_explicit_initialization_and_no_hidden_execution(stores):
    store = stores()
    reg = registration()
    application = app(store, reg)
    with pytest.raises(CollaborationNotInitialized):
        await application.discover_participants(context=CONTEXT)
    assert application._participant_coordinator._initialized is None
    results = await asyncio.gather(
        *(app(stores(), reg).initialize_collaboration() for _ in range(4))
    )
    assert all(value == results[0] for value in results)
    initialized = await application.initialize_collaboration()
    assert initialized == results[0]
    first, second = await asyncio.gather(
        create(application, initialized, "one"), create(application, initialized, "two")
    )
    assert first[1].participants[0].reference != second[1].participants[0].reference
    assert first[1].participants[0].configuration == second[1].participants[0].configuration
    assert not application._agents and not application._providers
    assert application.task_store is None
    assert not store._owners.pending


async def test_exact_replay_and_configuration_history(stores):
    reg = registration()
    application = app(stores(), reg)
    initialized = await application.initialize_collaboration()
    request, receipt = await create(application, initialized, alias="reviewer")
    ref = receipt.participants[0].reference
    updated = await application.configure_participant(
        ParticipantConfigure(
            operation=initialized.operation("configure"),
            participant=ref,
            expected_configuration_revision=1,
            configuration=configuration(2),
        ),
        context=CONTEXT,
    )
    assert updated.participants[0].configuration_revision == 2
    reopened = app(stores(), replace(reg, configurations=()))
    assert await reopened.initialize_collaboration() == initialized
    assert await reopened.create_participant(request, context=CONTEXT) == receipt
    assert (
        await reopened.inspect_participant(ref, context=CONTEXT)
    ).participant == updated.participants[0]
    found = await reopened.lookup_participant_operation(receipt.expected, context=CONTEXT)
    assert isinstance(found, ExactMatch) and found.receipt == receipt
    with pytest.raises(CollaborationUnavailable):
        await create(reopened, initialized, "unregistered")
    assert len((await reopened.list_participant_events(context=CONTEXT)).events) == 3


@pytest.mark.parametrize("matching_first", [False, True])
async def test_concurrent_readback_expectations_do_not_elect_intent(
    stores, monkeypatch, matching_first
):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    request, receipt = await create(application, initialized)
    matching = receipt.expected
    conflicting = matching.model_copy(
        update={"initiator": matching.initiator.model_copy(update={"principal": "different"})}
    )
    first, second = (matching, conflicting) if matching_first else (conflicting, matching)
    entered, release = asyncio.Event(), asyncio.Event()
    original = store._lookup
    calls = []

    async def blocked(initialization, expected, *, redactor):
        calls.append(expected)
        if expected == first:
            entered.set()
            await release.wait()
        return await original(initialization, expected, redactor=redactor)

    monkeypatch.setattr(store, "_lookup", blocked)
    pending = asyncio.create_task(application.lookup_participant_operation(first, context=CONTEXT))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        other = await asyncio.wait_for(
            application.lookup_participant_operation(second, context=CONTEXT), 5
        )
        assert other.status == ("conflict" if matching_first else "match")
        if not matching_first:
            assert other.receipt == receipt
            assert await application.create_participant(request, context=CONTEXT) == receipt
        assert calls.count(first) == 1
        assert not pending.done()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(pending, return_exceptions=True), 5)
    result = pending.result()
    assert result.status == ("match" if matching_first else "conflict")
    assert len((await application.list_participant_events(context=CONTEXT)).events) == 2


@pytest.mark.parametrize("change", ["alias", "configuration", "principal", "kind"])
async def test_fixed_key_conflict_before_second_effect(stores, change):
    application = app(stores(), registration())
    initialized = await application.initialize_collaboration()
    request, receipt = await create(application, initialized)
    if change == "alias":
        changed = request.model_copy(update={"alias": "another"})
        call = application.create_participant(changed, context=CONTEXT)
    elif change == "configuration":
        call = application.create_participant(
            request.model_copy(update={"configuration": configuration(2)}), context=CONTEXT
        )
    elif change == "principal":
        changed = receipt.expected.model_copy(
            update={
                "initiator": receipt.expected.initiator.model_copy(
                    update={"principal": "different"}
                )
            }
        )
        result = await application.lookup_participant_operation(changed, context=CONTEXT)
        assert result.status == "conflict"
        return
    else:
        call = application.configure_participant(
            ParticipantConfigure(
                operation=request.operation,
                participant=receipt.participants[0].reference,
                expected_configuration_revision=1,
                configuration=configuration(),
            ),
            context=CONTEXT,
        )
    with pytest.raises(CollaborationConflict):
        await call
    assert len((await application.discover_participants(context=CONTEXT)).participants) == 1
    assert len((await application.list_participant_events(context=CONTEXT)).events) == 2


async def test_alias_cas_and_retained_reference(stores):
    application = app(stores(), registration())
    initialized = await application.initialize_collaboration()
    _, first = await create(application, initialized, "first", alias="reviewer")
    _, second = await create(application, initialized, "second")
    old, new = first.participants[0].reference, second.participants[0].reference
    change = ParticipantAliasChange(
        operation=initialized.operation("rebind"),
        alias="reviewer",
        expected_alias_revision=1,
        expected_target=old,
        target=new,
    )
    receipt = await application.change_participant_alias(change, context=CONTEXT)
    assert receipt.alias.target == new
    assert (
        await application.inspect_participant(old, context=CONTEXT)
    ).participant.reference == old
    assert (await application.resolve_participant_alias("reviewer", context=CONTEXT)).target == new
    stale = change.model_copy(update={"operation": initialized.operation("stale")})
    with pytest.raises(CollaborationConflict):
        await application.change_participant_alias(stale, context=CONTEXT)
    assert await application.change_participant_alias(change, context=CONTEXT) == receipt


async def test_access_and_pages_are_scope_bound(stores):
    policy = Policy()
    application = app(stores(), registration(policy=policy))
    initialized = await application.initialize_collaboration()
    _, first = await create(application, initialized, "first", alias="first")
    _, second = await create(
        application, initialized, "second", alias="second", expected_alias_revision=1
    )
    old_page = await application.discover_participants(context=CONTEXT, limit=1)
    policy.allowed = (first.participants[0].reference,)
    page = await application.discover_participants(context=CONTEXT)
    assert tuple(p.reference for p in page.participants) == policy.allowed
    assert await application.resolve_participant_alias("second", context=CONTEXT) is None
    with pytest.raises(CollaborationAccessDenied):
        await application.inspect_participant(second.participants[0].reference, context=CONTEXT)
    with pytest.raises(CollaborationAccessDenied):
        await application.discover_participants(context=CONTEXT, cursor=old_page.next_cursor)
    with pytest.raises(CollaborationAccessDenied):
        await application.discover_participants(context={"principal": "operator"})
    events = (await application.list_participant_events(context=CONTEXT)).events
    assert len(events) == 1 and events[0].participants == policy.allowed
    policy.allowed = None
    policy.denied.add("create")
    with pytest.raises(CollaborationAccessDenied):
        await create(application, initialized, "denied")


async def test_conflicting_bootstrap_and_unknown_namespace(stores):
    reg = registration()
    application = app(stores(), reg)
    initialized = await application.initialize_collaboration()
    conflicting = replace(
        reg, bootstrap=reg.bootstrap.model_copy(update={"owner_name": "different"})
    )
    with pytest.raises(CollaborationConflict):
        await app(stores(), conflicting).initialize_collaboration()
    for field, value in (
        ("namespace_incarnation", "unknown"),
        ("generation", 2),
        ("application_scope", "other"),
    ):
        request = ParticipantCreate(
            operation=initialized.operation("invalid").model_copy(update={field: value}),
            configuration=configuration(),
        )
        with pytest.raises((CollaborationConflict, CollaborationContractError)):
            await application.create_participant(request, context=CONTEXT)
    assert not (await application.discover_participants(context=CONTEXT)).participants


async def test_capacity_does_not_evict_exact_receipts(stores):
    reg = registration()
    reg = replace(
        reg,
        bootstrap=reg.bootstrap.model_copy(
            update={"limits": reg.bootstrap.limits.model_copy(update={"participants": 1})}
        ),
    )
    application = app(stores(), reg)
    initialized = await application.initialize_collaboration()
    request, receipt = await create(application, initialized)
    with pytest.raises(CollaborationCapacityExceeded):
        await create(application, initialized, "second")
    assert await application.create_participant(request, context=CONTEXT) == receipt
    assert len((await application.list_participant_events(context=CONTEXT)).events) == 2


async def test_lost_ack_reconciles_without_new_mutation(stores, monkeypatch):
    store = stores()
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    original = store._apply

    async def lost(*args, **kwargs):
        await original(*args, **kwargs)
        raise ConnectionError("lost acknowledgement")

    monkeypatch.setattr(store, "_apply", lost)
    request = ParticipantCreate(
        operation=initialized.operation("lost"), configuration=configuration()
    )
    with pytest.raises(CollaborationUnavailable):
        await application.create_participant(request, context=CONTEXT)
    reopened = app(stores(), reg)
    await reopened.initialize_collaboration()
    receipt = await reopened.create_participant(request, context=CONTEXT)
    assert len((await reopened.discover_participants(context=CONTEXT)).participants) == 1
    assert len((await reopened.list_participant_events(context=CONTEXT)).events) == 2
    assert receipt.event.sequence == 2


async def test_real_cancellation_retains_dispatched_owner(stores, monkeypatch):
    store = stores()
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    entered, release = asyncio.Event(), asyncio.Event()
    original = store._apply

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "_apply", blocked)
    request = ParticipantCreate(
        operation=initialized.operation("cancel"), configuration=configuration()
    )
    owner = asyncio.create_task(application.create_participant(request, context=CONTEXT))
    await asyncio.wait_for(entered.wait(), 5)
    owner.cancel("secret-cancellation-canary")
    owner.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await owner
    assert owner.cancelled() and owner.cancelling() == 2
    assert "secret-cancellation-canary" not in str(caught.value)
    assert caught.value.__context__ is None
    assert len(store._owners.pending) == 1
    retry = asyncio.create_task(application.create_participant(request, context=CONTEXT))
    await asyncio.sleep(0)
    release.set()
    result = await asyncio.wait_for(retry, 5)
    assert result.event.sequence == 2
    assert len((await application.list_participant_events(context=CONTEXT)).events) == 2


@pytest.mark.parametrize("field", ["alias", "configuration", "unknown"])
async def test_secret_rejection_has_no_diagnostic_side_channel(stores, field, caplog, capsys):
    secret = "participant-sensitive-canary"
    application = app(stores(), registration(), secret_redactor=SecretRedactor(secret))
    initialized = await application.initialize_collaboration()
    request = ParticipantCreate(
        operation=initialized.operation("unsafe"), configuration=configuration()
    )
    if field == "alias":
        object.__setattr__(request, "alias", secret)
    elif field == "configuration":
        object.__setattr__(request.configuration.definition, "name", secret)
    else:

        class Hostile:
            def __repr__(self):
                return secret

        object.__setattr__(request, "alias", Hostile())
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(CollaborationContractError) as caught:
            await application.create_participant(request, context=CONTEXT)
    output = capsys.readouterr()
    assert (
        secret not in str(caught.value) + repr(caught.value) + caplog.text + output.out + output.err
    )
    assert caught.value.__context__ is None
    assert all(secret not in str(w.message) for w in recorded)
    assert not (await application.discover_participants(context=CONTEXT)).participants


async def test_timeout_retains_single_flight_and_rejects_conflicting_intent(stores, monkeypatch):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    entered, release = asyncio.Event(), asyncio.Event()
    original = store._apply
    calls = 0

    async def blocked(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(store, "_apply", blocked)
    store._owners.observation_timeout = 0.25
    request = ParticipantCreate(
        operation=initialized.operation("timeout"), configuration=configuration()
    )
    try:
        with pytest.raises(CollaborationUnavailable):
            await application.create_participant(request, context=CONTEXT)
        assert entered.is_set() and len(store._owners.pending) == 1
        with pytest.raises(CollaborationUnavailable):
            await application.create_participant(request, context=CONTEXT)
        with pytest.raises(CollaborationConflict):
            await application.create_participant(
                request.model_copy(update={"alias": "other"}), context=CONTEXT
            )
        assert calls == 1
    finally:
        release.set()
        await asyncio.gather(*store._owners.pending)
        store._owners.observation_timeout = 10
    assert (await application.create_participant(request, context=CONTEXT)).event.sequence == 2


async def test_child_cancellation_is_not_caller_cancellation(stores, monkeypatch):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError("dependency-private-canary")

    monkeypatch.setattr(store, "_apply", cancelled)
    caller = asyncio.create_task(create(application, initialized))
    with pytest.raises(CollaborationUnavailable) as caught:
        await caller
    assert not caller.cancelled() and caller.cancelling() == 0
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert "dependency-private-canary" not in str(caught.value)


@pytest.mark.parametrize(
    "boundary", ["participants", "configurations", "aliases", "operations", "events", "anchors"]
)
async def test_transaction_failure_has_no_partial_identity_or_event(stores, monkeypatch, boundary):
    from contextlib import asynccontextmanager

    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    original = store._transaction

    @asynccontextmanager
    async def broken(scope, *, write):
        async with original(scope, write=write) as tx:
            put = tx.put

            async def fail(table, key, value, *, insert):
                await put(table, key, value, insert=insert)
                if table == boundary:
                    raise OSError("injected transaction failure")

            if write:
                tx.put = fail
            yield tx

    monkeypatch.setattr(store, "_transaction", broken)
    with pytest.raises(CollaborationUnavailable):
        await create(application, initialized, alias="reviewer")
    monkeypatch.setattr(store, "_transaction", original)
    assert not (await application.discover_participants(context=CONTEXT)).participants
    assert await application.resolve_participant_alias("reviewer", context=CONTEXT) is None
    assert len((await application.list_participant_events(context=CONTEXT)).events) == 1
    assert (await create(application, initialized, alias="reviewer"))[1].event.sequence == 2


async def test_receipt_rejects_corrupted_result_from_store(stores, monkeypatch):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    request, receipt = await create(application, initialized)
    corrupted = receipt.model_copy(
        update={
            "participants": (
                receipt.participants[0].model_copy(update={"configuration_revision": 2}),
            )
        }
    )

    async def lookup(*args, **kwargs):
        return ExactMatch[ParticipantReceipt](receipt=receipt).model_copy(
            update={"receipt": corrupted}
        )

    monkeypatch.setattr(store, "lookup", lookup)
    with pytest.raises(CollaborationContractError):
        await application.create_participant(request, context=CONTEXT)


async def test_canonical_payload_accounting_matches_retained_rows(stores):
    from cayu.collaboration._participant_state import ParticipantPermitState
    from cayu.collaboration._preparation import contract_bytes, prepare_contract
    from cayu.collaboration.base import _ANCHOR_BYTES, _Anchor
    from cayu.collaboration.lifecycle import NamespaceSnapshot
    from cayu.collaboration.participants import (
        ParticipantConfigurationEvidence,
        ParticipantEvent,
        ParticipantLifecycleEvidence,
        ParticipantReceipt,
        ParticipantSnapshot,
    )

    store = stores()
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    _, first = await create(application, initialized, alias="reviewer")
    ref = first.participants[0].reference
    second = await application.configure_participant(
        ParticipantConfigure(
            operation=initialized.operation("configure"),
            participant=ref,
            expected_configuration_revision=1,
            configuration=configuration(2),
        ),
        context=CONTEXT,
    )
    third = await application.change_participant_alias(
        ParticipantAliasChange(
            operation=initialized.operation("remove"),
            alias="reviewer",
            expected_alias_revision=1,
            expected_target=ref,
            target=None,
        ),
        context=CONTEXT,
    )
    redactor = SecretRedactor()
    total = _ANCHOR_BYTES
    async with store._transaction(reg.bootstrap.application_scope, write=False) as tx:
        anchor = prepare_contract(_Anchor, await tx.get("anchors", ()), redactor=redactor)
        values = [("participants", (ref.participant_id,), ParticipantSnapshot)]
        values += [
            ("participant_permits", (ref.participant_id,), ParticipantPermitState),
            ("lifecycle_history", (ref.participant_id, 1), ParticipantLifecycleEvidence),
        ]
        values += [("namespaces", (initialized.namespace_incarnation, 1), NamespaceSnapshot)]
        values += [
            ("configurations", (ref.participant_id, version), ParticipantConfigurationEvidence)
            for version in (1, 2)
        ]
        values += [("events", (sequence,), ParticipantEvent) for sequence in range(1, 5)]
        values += [
            (
                "operations",
                (initialized.namespace_incarnation, 1, receipt.expected.operation.caller_key),
                ParticipantReceipt,
            )
            for receipt in (first, second, third)
        ]
        for table, key, schema in values:
            value = prepare_contract(schema, await tx.get(table, key), redactor=redactor)
            total += len(contract_bytes(value, redactor=redactor))
    assert anchor.retained_bytes == total


def test_participant_exports_are_discoverable():
    import cayu
    import cayu.collaboration as collaboration

    for name in collaboration.__all__:
        assert name in cayu.__all__
        assert getattr(cayu, name) is getattr(collaboration, name)


async def test_shared_exact_mutation_conformance(stores):
    from tests.core.collaboration_conformance import ReceiverState, assert_exact_mutation_replay

    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    request = ParticipantCreate(
        operation=initialized.operation("conformance"), configuration=configuration()
    )
    command = application._participant_coordinator._expected(request, CONTEXT)
    changed = command.model_copy(
        update={
            "intent": command.intent.model_copy(
                update={"request": request.model_copy(update={"configuration": configuration(2)})}
            )
        }
    )

    class Fixture:
        async def apply(self, expected, *, authority):
            assert authority is CONTEXT
            return await application.create_participant(expected.intent.request, context=authority)

        async def lookup(self, expected, *, authority):
            assert authority is CONTEXT
            return await application.lookup_participant_operation(expected, context=authority)

        async def inspect(self):
            participants = await application.discover_participants(context=CONTEXT)
            events = await application.list_participant_events(context=CONTEXT)
            return ReceiverState(
                prepared=0,
                effects=len(participants.participants),
                events=len(events.events),
                receipts=len(events.events) - 1,
                pending=len(store._owners.pending),
                cleanup_pending=0,
            )

    await assert_exact_mutation_replay(Fixture(), command, changed, authority=CONTEXT)


async def test_independent_workers_elect_one_alias_update(stores):
    reg = registration()
    first, second = app(stores(), reg), app(stores(), reg)
    initialized = await first.initialize_collaboration()
    assert await second.initialize_collaboration() == initialized
    _, left = await create(first, initialized, "left", alias="reviewer")
    _, right = await create(second, initialized, "right")
    common = dict(
        alias="reviewer",
        expected_alias_revision=1,
        expected_target=left.participants[0].reference,
        target=right.participants[0].reference,
    )
    results = await asyncio.gather(
        first.change_participant_alias(
            ParticipantAliasChange(operation=initialized.operation("change-one"), **common),
            context=CONTEXT,
        ),
        second.change_participant_alias(
            ParticipantAliasChange(operation=initialized.operation("change-two"), **common),
            context=CONTEXT,
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(result, CollaborationConflict) for result in results) == 1
    current = await first.resolve_participant_alias("reviewer", context=CONTEXT)
    assert current is not None and current.revision == 2 and current.target == common["target"]
    assert (
        await first.inspect_participant(left.participants[0].reference, context=CONTEXT)
    ).participant == left.participants[0]
    assert len((await second.list_participant_events(context=CONTEXT)).events) == 4


async def test_readback_survives_mutation_policy_removal(stores):
    policy = Policy()
    reg = registration(policy=policy)
    application = app(stores(), reg)
    initialized = await application.initialize_collaboration()
    request, receipt = await create(application, initialized)
    policy.denied.add("create")
    reopened = app(stores(), replace(reg, configurations=()))
    await reopened.initialize_collaboration()
    assert await reopened.create_participant(request, context=CONTEXT) == receipt
    with pytest.raises(CollaborationAccessDenied):
        await create(reopened, initialized, "another")
    policy.denied.add("readback")
    with pytest.raises(CollaborationAccessDenied):
        await reopened.create_participant(request, context=CONTEXT)


async def test_unsupported_capability_refuses_before_mutation_but_allows_exact_read(
    stores, monkeypatch
):
    from cayu.collaboration._capabilities import (
        CapabilityDescriptor,
        CollaborationCapabilityUnavailable,
    )
    from cayu.collaboration.base import IDENTITY_FAMILY

    store = stores()
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    request, receipt = await create(application, initialized)
    monkeypatch.setattr(
        store,
        "capabilities",
        lambda owner: CapabilityDescriptor(owner=owner, mutations=(), readbacks=(IDENTITY_FAMILY,)),
    )
    assert await application.create_participant(request, context=CONTEXT) == receipt
    with pytest.raises(CollaborationCapabilityUnavailable):
        await create(application, initialized, "unsupported")
    assert len((await application.discover_participants(context=CONTEXT)).participants) == 1


@pytest.mark.parametrize("field", ["operations", "events", "retained_bytes"])
async def test_capacity_exact_boundary_and_duplicate_accounting(stores, field):
    from cayu.collaboration._preparation import prepare_contract
    from cayu.collaboration.base import _Anchor

    store = stores()
    reg = registration()
    if field == "retained_bytes":
        reg = replace(
            reg,
            bootstrap=reg.bootstrap.model_copy(
                update={
                    "limits": reg.bootstrap.limits.model_copy(
                        update={"retained_bytes": 999999, "control_bytes": 200000}
                    )
                }
            ),
        )
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    await create(application, initialized)
    async with store._transaction(reg.bootstrap.application_scope, write=False) as tx:
        anchor = prepare_contract(_Anchor, await tx.get("anchors", ()), redactor=SecretRedactor())
    limit = {
        "operations": anchor.operation_count + reg.bootstrap.limits.control_operations,
        "events": anchor.event_count + reg.bootstrap.limits.control_events,
        "retained_bytes": anchor.retained_bytes + reg.bootstrap.limits.control_bytes,
    }[field]
    # Scope/UUID lengths and limit digit counts stay equal between scopes.
    if field == "retained_bytes":
        assert len(str(limit)) == len(str(reg.bootstrap.limits.retained_bytes))
    bounded = registration(limits=reg.bootstrap.limits.model_copy(update={field: limit}))
    bounded_app = app(store, bounded)
    bound_init = await bounded_app.initialize_collaboration()
    request, receipt = await create(bounded_app, bound_init)
    assert await bounded_app.create_participant(request, context=CONTEXT) == receipt
    with pytest.raises(CollaborationCapacityExceeded):
        await create(bounded_app, bound_init, "overflow")
    assert len((await bounded_app.list_participant_events(context=CONTEXT)).events) == 2
    if field == "retained_bytes":
        below = registration(limits=reg.bootstrap.limits.model_copy(update={field: limit - 1}))
        below_app = app(store, below)
        below_init = await below_app.initialize_collaboration()
        with pytest.raises(CollaborationCapacityExceeded):
            await create(below_app, below_init)
        assert len((await below_app.list_participant_events(context=CONTEXT)).events) == 1


@pytest.mark.parametrize("control", ["create", "identity", "committed", "active", "match"])
async def test_known_secret_collision_preserves_fixed_controls(stores, control):
    application = app(stores(), registration(), secret_redactor=SecretRedactor(control))
    initialized = await application.initialize_collaboration()
    # The caller's operation key is untrusted text, not the fixed create control.
    request, receipt = await create(application, initialized, "operation")
    assert await application.create_participant(request, context=CONTEXT) == receipt
    with pytest.raises(CollaborationContractError):
        await create(application, initialized, "unsafe", alias=control)


async def test_pending_caller_cancellation_does_not_dispatch(stores):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()

    async def caller():
        task = asyncio.current_task()
        assert task is not None
        task.cancel("caller-private-canary")
        await create(application, initialized)

    task = asyncio.create_task(caller())
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert task.cancelled() and task.cancelling() == 1
    assert caught.value.__context__ is None
    assert "caller-private-canary" not in str(caught.value)
    assert not store._owners.pending
    assert not (await application.discover_participants(context=CONTEXT)).participants


async def test_store_failure_diagnostic_is_detached(stores, monkeypatch, caplog, capsys):
    store = stores()
    application = app(
        store, registration(), secret_redactor=SecretRedactor("driver-private-canary")
    )
    initialized = await application.initialize_collaboration()

    async def failed(*args, **kwargs):
        raise ConnectionError("driver-private-canary")

    monkeypatch.setattr(store, "_apply", failed)
    with (
        warnings.catch_warnings(record=True) as recorded,
        pytest.raises(CollaborationUnavailable) as caught,
    ):
        await create(application, initialized)
    output = capsys.readouterr()
    assert caught.value.__context__ is None
    assert isinstance(caught.value.__cause__, ConnectionError)
    assert caught.value.__cause__.__context__ is None
    assert (
        "driver-private-canary"
        not in str(caught.value)
        + repr(caught.value)
        + str(caught.value.__cause__)
        + caplog.text
        + output.out
        + output.err
    )
    assert not recorded


@pytest.mark.parametrize("events", [False, True])
async def test_wrapper_cannot_bypass_page_access(stores, monkeypatch, events):
    policy = Policy()
    store = stores()
    application = app(store, registration(policy=policy))
    initialized = await application.initialize_collaboration()
    _, receipt = await create(application, initialized)
    policy.allowed = ()

    async def unfiltered(*args, **kwargs):
        return (1, (receipt.event,)) if events else receipt.participants

    monkeypatch.setattr(store, "scan_events" if events else "scan", unfiltered)
    with pytest.raises(CollaborationAccessDenied):
        if events:
            await application.list_participant_events(context=CONTEXT)
        else:
            await application.discover_participants(context=CONTEXT)


async def test_persistent_schema_requires_exact_primary_authority(stores):
    from cayu.storage._collaboration_schema import (
        validate_postgres_collaboration_schema,
        validate_sqlite_collaboration_schema,
    )
    from cayu.storage.migrations import SchemaError

    store = stores()
    await app(store, registration()).initialize_collaboration()
    if isinstance(store, InMemoryCollaborationStore):
        return  # There is no relational schema in this backend.
    if isinstance(store, SQLiteCollaborationStore):
        connection = store._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute("DROP TABLE cayu_collaboration_participants")
            connection.execute(
                "CREATE TABLE cayu_collaboration_participants(scope TEXT NOT NULL, participant_id TEXT NOT NULL, document TEXT NOT NULL)"
            )
            with pytest.raises(SchemaError):
                validate_sqlite_collaboration_schema(connection)
        finally:
            connection.rollback()
        validate_sqlite_collaboration_schema(connection)
    else:
        async with store._connection() as connection:
            with pytest.raises(SchemaError):
                async with connection.transaction():
                    await connection.execute(
                        "ALTER TABLE cayu_collaboration_participants DROP CONSTRAINT cayu_collaboration_participants_pkey"
                    )
                    async with connection.cursor() as cursor:
                        await validate_postgres_collaboration_schema(cursor)
            async with connection.cursor() as cursor:
                await validate_postgres_collaboration_schema(cursor)


async def test_store_group_preserves_ordered_sanitized_failures(stores, monkeypatch):
    store = stores()
    application = app(store, registration(), secret_redactor=SecretRedactor("private-value"))
    initialized = await application.initialize_collaboration()

    async def failed(*args, **kwargs):
        raise ExceptionGroup(
            "private-value",
            [
                ConnectionError("commit private-value"),
                ExceptionGroup(
                    "cleanup", [OSError("rollback private-value"), TimeoutError("release")]
                ),
            ],
        )

    monkeypatch.setattr(store, "_apply", failed)
    with pytest.raises(CollaborationUnavailable) as caught:
        await create(application, initialized)
    cause = caught.value.__cause__
    assert isinstance(cause, ExceptionGroup)
    assert type(cause.exceptions[0]) is ConnectionError
    nested = cause.exceptions[1]
    assert isinstance(nested, ExceptionGroup)
    assert tuple(type(leaf) for leaf in nested.exceptions) == (OSError, TimeoutError)
    assert "private-value" not in repr(cause)
    assert caught.value.__context__ is None and cause.__context__ is None


async def test_missing_retained_evidence_is_typed_unavailable_not_absence(stores, monkeypatch):
    from contextlib import asynccontextmanager

    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    request, receipt = await create(application, initialized)
    original = store._transaction

    @asynccontextmanager
    async def missing(scope, *, write):
        async with original(scope, write=write) as tx:
            get = tx.get

            async def hide_event(table, key):
                return None if table == "events" else await get(table, key)

            tx.get = hide_event
            yield tx

    monkeypatch.setattr(store, "_transaction", missing)
    result = await application.lookup_participant_operation(receipt.expected, context=CONTEXT)
    assert result.status == "unavailable"
    with pytest.raises(CollaborationUnavailable):
        await application.create_participant(request, context=CONTEXT)
    monkeypatch.setattr(store, "_transaction", original)
    assert await application.create_participant(request, context=CONTEXT) == receipt
