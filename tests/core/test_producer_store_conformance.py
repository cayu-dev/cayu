"""Direct native producer cleanup-receipt retirement conformance on every store.

Receipts are contract-valid native cleanup receipts seeded into each backend's
own receipt index, then reclaimed through the store method the runtime calls.
Cleanup publication and observation need a complete producer registration and
are qualified by the producer scenario tests instead.
"""

import json
from bisect import insort
from hashlib import sha256

import pytest

from cayu.collaboration._contracts import ObjectRef, OperationRef, OwnerRef
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration.lifecycle import NamespaceRef
from cayu.runtime._producer_retirement import (
    ProducerCleanupRetirement,
    _accepted_source_retirement,
    retirement_key,
)
from cayu.sessions._producer_checkpoint import OPERATION_PREFIX
from cayu.sessions._producer_cleanup_contract import NativeProducerCleanupReceipt
from cayu.sessions.memory import InMemorySessionStore
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor

pytestmark = [pytest.mark.anyio, pytest.mark.store_semantics]
REDACTOR = SecretRedactor()
OWNER = OwnerRef(application_scope="app", owner_id="producer-owner", incarnation="one")
RECEIVER = ObjectRef(
    owner=OWNER, kind="receiver", object_id="receiver", incarnation="one", revision=1
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def sessions(request, tmp_path):
    backend = request.param
    if backend == "memory":
        store = InMemorySessionStore()
    elif backend == "sqlite":
        from cayu.storage.sqlite import SQLiteSessionStore

        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    else:
        from cayu.storage.postgres import PostgresSessionStore

        store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    store.backend = backend
    try:
        if backend == "postgres":
            await store._ensure_ready()
            async with store._connection() as connection:
                # postgres_dsn is module-scoped; each case starts without receipts.
                await connection.execute(
                    "TRUNCATE cayu_producer_cleanup_receipts, cayu_producer_cleanup_retirements"
                )
                await connection.commit()
        yield store
    finally:
        if hasattr(store, "close"):
            await store.close()


def retirement(generation, namespace="namespace-a", receiver=RECEIVER):
    return ProducerCleanupRetirement(
        receiver=receiver,
        namespace=NamespaceRef(owner=OWNER, namespace_incarnation=namespace, generation=generation),
    )


def receipt(caller_key, generation, namespace="namespace-a"):
    registration = OperationRef(
        application_scope=OWNER.application_scope,
        namespace_incarnation=namespace,
        generation=generation,
        caller_key=caller_key,
    )
    return NativeProducerCleanupReceipt(
        mode="exclusion",
        registration=registration,
        receiver=RECEIVER,
        registration_commitment="sha256:" + "a" * 64,
        source_cleanup_commitment="sha256:" + "b" * 64,
        session_id=f"session-{caller_key}",
        session_instance_id="instance",
        interaction_id=None,
        run_epoch=None,
        native_release_commitment="sha256:" + "c" * 64,
    )


def operation_key(value):
    return (
        OPERATION_PREFIX + sha256(contract_bytes(value.registration, redactor=REDACTOR)).hexdigest()
    )


def row(value, *, key=None, generation=None):
    namespace = retirement_key(
        retirement(value.registration.generation, value.registration.namespace_incarnation)
    )
    return (
        operation_key(value) if key is None else key,
        namespace,
        value.registration.generation if generation is None else generation,
        value.model_dump(mode="json"),
    )


async def seed(store, rows):
    if store.backend == "memory":
        async with store._lock:
            for key, namespace, generation, document in rows:
                store._producer_cleanup_receipts[key] = document
                insort(store._producer_cleanup_index.setdefault(namespace, []), (generation, key))
        return
    if store.backend == "sqlite":

        def insert(connection):
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                "INSERT INTO cayu_producer_cleanup_receipts "
                "(operation_key, namespace_key, generation, receipt_json) VALUES (?, ?, ?, ?)",
                [(key, ns, gen, json.dumps(doc)) for key, ns, gen, doc in rows],
            )
            connection.commit()

        await store._run_write(insert)
        return
    from cayu.storage._postgres_support import _dumps

    async with store._connection() as connection, connection.cursor() as cursor:
        await cursor.executemany(
            "INSERT INTO cayu_producer_cleanup_receipts "
            "(operation_key, namespace_key, generation, receipt_json) VALUES (%s, %s, %s, %s)",
            [(key, ns, gen, _dumps(doc)) for key, ns, gen, doc in rows],
        )
        await connection.commit()


async def retained(store):
    """Backend-neutral view: sorted (namespace, generation, key) rows and fences."""
    if store.backend == "memory":
        async with store._lock:
            index = sorted(
                (namespace, generation, key)
                for namespace, entries in store._producer_cleanup_index.items()
                for generation, key in entries
            )
            assert {key for _, _, key in index} == set(store._producer_cleanup_receipts)
            return index, dict(store._producer_cleanup_retirements)
    receipts_sql = (
        "SELECT namespace_key, generation, operation_key FROM cayu_producer_cleanup_receipts"
    )
    fences_sql = "SELECT namespace_key, through_generation FROM cayu_producer_cleanup_retirements"
    if store.backend == "sqlite":

        def query(connection):
            return (
                connection.execute(receipts_sql).fetchall(),
                connection.execute(fences_sql).fetchall(),
            )

        receipts, fences = await store._run_read(query)
    else:
        async with store._connection() as connection, connection.cursor() as cursor:
            await cursor.execute(receipts_sql)
            receipts = await cursor.fetchall()
            await cursor.execute(fences_sql)
            fences = await cursor.fetchall()
            await connection.rollback()
    return sorted(tuple(item) for item in receipts), dict(fences)


async def retire(store, value, limit):
    return await store._retire_native_producer_cleanup(
        value, authority=_accepted_source_retirement(value), limit=limit
    )


def keys(*values):
    return sorted(
        (row(value)[1], value.registration.generation, operation_key(value)) for value in values
    )


async def test_retirement_reclaims_in_generation_then_key_order(sessions):
    namespace_a = retirement_key(retirement(1))
    first = [receipt(f"first-{index}", 1) for index in range(3)]
    second = [receipt(f"second-{index}", 2) for index in range(2)]
    later = [receipt("later", 3)]
    other = [receipt("other", 1, namespace="namespace-b")]
    everything = [*later, *second, *other, *first]
    await seed(sessions, [row(value) for value in everything])
    assert await retained(sessions) == (keys(*everything), {})
    # A fresh namespace has nothing to reclaim but still installs its fence.
    empty = await retire(sessions, retirement(5, namespace="namespace-c"), 32)
    assert (empty.removed, empty.remaining) == (0, False)
    fence_c = retirement_key(retirement(1, namespace="namespace-c"))
    assert (await retained(sessions))[1] == {fence_c: 5}

    ordered = sorted(first, key=operation_key) + sorted(second, key=operation_key)
    removed = []
    for limit, expected in ((2, (2, True)), (2, (2, True)), (1, (1, False))):
        result = await retire(sessions, retirement(2), limit)
        assert result.retirement == retirement(2)
        assert (result.removed, result.remaining) == expected
        removed.extend(ordered[len(removed) : len(removed) + result.removed])
        assert await retained(sessions) == (
            keys(*[value for value in everything if value not in removed]),
            {fence_c: 5, namespace_a: 2},
        )
    assert removed == ordered
    replay = await retire(sessions, retirement(2), 32)
    assert (replay.removed, replay.remaining) == (0, False)
    assert await retained(sessions) == (keys(*later, *other), {fence_c: 5, namespace_a: 2})


async def test_retirement_batch_boundary_and_monotonic_fence(sessions):
    namespace = retirement_key(retirement(1))
    values = [receipt(f"bounded-{index}", 1 + index % 2) for index in range(4)]
    await seed(sessions, [row(value) for value in values])
    exact = await retire(sessions, retirement(4), 4)
    assert (exact.removed, exact.remaining) == (4, False)
    assert await retained(sessions) == ([], {namespace: 4})
    # A lower retirement neither lowers the fence nor reports work.
    lower = await retire(sessions, retirement(2), 1)
    assert (lower.removed, lower.remaining) == (0, False)
    assert await retained(sessions) == ([], {namespace: 4})
    # Generation is part of the retirement, not the namespace fence key.
    higher = retirement(6, receiver=RECEIVER.model_copy(update={"revision": 2}))
    assert retirement_key(higher) == namespace
    assert (await retire(sessions, higher, 32)).removed == 0
    assert await retained(sessions) == ([], {namespace: 6})


async def test_retirement_with_full_batch_and_one_extra_reports_remaining(sessions):
    values = [receipt(f"extra-{index:02d}", 1) for index in range(33)]
    await seed(sessions, [row(value) for value in values])
    first = await retire(sessions, retirement(1), 32)
    assert (first.removed, first.remaining) == (32, True)
    assert (await retained(sessions))[0] == keys(max(values, key=operation_key))
    last = await retire(sessions, retirement(1), 32)
    assert (last.removed, last.remaining) == (1, False)
    assert (await retained(sessions))[0] == []


@pytest.mark.parametrize("corruption", ["operation_key", "indexed_generation", "foreign_owner"])
async def test_invalid_receipt_rejects_whole_batch_without_fence(sessions, corruption):
    valid = [receipt(f"valid-{index}", 2) for index in range(2)]
    corrupt = receipt("corrupt", 1)
    if corruption == "operation_key":
        bad = row(corrupt, key=OPERATION_PREFIX + "0" * 64)
    elif corruption == "indexed_generation":
        bad = row(receipt("corrupt", 2), generation=1)
    else:
        foreign = corrupt.model_copy(
            update={
                "receiver": RECEIVER.model_copy(
                    update={"owner": OWNER.model_copy(update={"owner_id": "other"})}
                )
            }
        )
        bad = row(foreign)
    # The corrupt row sorts first, so a valid prefix is never removed.
    await seed(sessions, [bad, *(row(value) for value in valid)])
    before = await retained(sessions)
    with pytest.raises(Exception, match="conflict") as caught:
        await retire(sessions, retirement(2), 32)
    assert not isinstance(caught.value, PermissionError)
    assert await retained(sessions) == before
    assert before[1] == {}


async def test_retirement_requires_source_authority_and_bounded_batch(sessions):
    value = receipt("guarded", 1)
    await seed(sessions, [row(value)])
    before = await retained(sessions)
    target = retirement(1)
    for authority in (None, _accepted_source_retirement(retirement(2))):
        with pytest.raises(PermissionError):
            await sessions._retire_native_producer_cleanup(target, authority=authority, limit=1)
    for limit in (0, 33, True, 1.0):
        with pytest.raises(ValueError, match="batch"):
            await retire(sessions, target, limit)
    assert await retained(sessions) == before
