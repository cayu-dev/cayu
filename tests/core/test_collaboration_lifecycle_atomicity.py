"""Native transaction and normalized-index qualification for lifecycle maintenance."""

from contextlib import asynccontextmanager

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_collaboration_namespace import rotate
from tests.core.test_collaboration_permits import REDACTOR, permit
from tests.core.test_participant_identity import CONTEXT, app, create, registration

from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.storage._collaboration_schema import (
    validate_postgres_collaboration_schema,
    validate_sqlite_collaboration_schema,
)
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaError

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


@pytest.mark.parametrize("boundary", ["operations", "events", "namespaces", "anchors"])
async def test_pruning_late_failure_rolls_back_all_evidence(stores, monkeypatch, boundary):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    request, created = await create(application, initialized)
    _, rotated = await rotate(store, initialized)
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    events = await application.list_participant_events(context=CONTEXT)
    prune = NamespacePrune(
        operation=rotated.successor.reference.operation("prune"),
        namespace=rotated.namespace.reference,
        expected_retention_revision=before.retention_revision,
    )
    original = store._transaction
    injected = False

    @asynccontextmanager
    async def transaction(scope, *, write):
        nonlocal injected
        async with original(scope, write=write) as tx:
            if write and not injected:
                method = "put" if boundary == "anchors" else "delete"
                operation = getattr(tx, method)

                async def fail_after(table, *args, **kwargs):
                    nonlocal injected
                    result = await operation(table, *args, **kwargs)
                    if table == boundary and not injected:
                        injected = True
                        raise ConnectionError("publication failed after retained evidence changed")
                    return result

                setattr(tx, method, fail_after)
            yield tx

    monkeypatch.setattr(store, "_transaction", transaction)
    with pytest.raises(CollaborationUnavailable):
        await application.prune_collaboration_namespace(prune, context=CONTEXT)
    assert injected
    assert await application.inspect_collaboration_namespace(context=CONTEXT) == before
    assert await application.list_participant_events(context=CONTEXT) == events
    assert await application.create_participant(request, context=CONTEXT) == created
    result = await application.prune_collaboration_namespace(prune, context=CONTEXT)
    assert result.complete
    assert await application.prune_collaboration_namespace(prune, context=CONTEXT) == result


@pytest.mark.parametrize(
    "index",
    [
        "cayu_collaboration_permit_position_idx",
        "cayu_collaboration_permit_pending_idx",
        "cayu_collaboration_history_operation_idx",
    ],
)
async def test_lifecycle_schema_requires_bounded_lookup_indexes(stores, index):
    store = stores()
    await app(store, registration()).initialize_collaboration()
    if isinstance(store, InMemoryCollaborationStore):
        return  # No relational schema; memory transaction tests cover its index semantics.
    if isinstance(store, SQLiteCollaborationStore):
        connection = store._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(f"DROP INDEX {index}")
            with pytest.raises(SchemaError):
                validate_sqlite_collaboration_schema(connection, lifecycle=True)
        finally:
            connection.rollback()
        validate_sqlite_collaboration_schema(connection, lifecycle=True)
    else:
        async with store._connection() as connection:
            with pytest.raises(SchemaError):
                async with connection.transaction():
                    await connection.execute(f"DROP INDEX {index}")
                    async with connection.cursor() as cursor:
                        await validate_postgres_collaboration_schema(cursor, lifecycle=True)
            async with connection.cursor() as cursor:
                await validate_postgres_collaboration_schema(cursor, lifecycle=True)


@pytest.mark.parametrize(
    "table,field",
    [
        ("participant_permits", "outstanding"),
        ("participant_permits", "issued_frontier"),
        ("anchors", "operation_count"),
        ("anchors", "reserved_bytes"),
        ("anchors", "reserved_events"),
    ],
)
async def test_incomplete_durable_accounting_cannot_prove_empty_responsibility(
    stores, monkeypatch, table, field
):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    await store._register_permit(initialized, permit(initialized, ref), redactor=REDACTOR)
    before = await application.inspect_participant(ref, context=CONTEXT)
    original = store._transaction

    @asynccontextmanager
    async def incomplete(scope, *, write):
        async with original(scope, write=write) as tx:
            get = tx.get

            async def missing(family, key):
                value = await get(family, key)
                if family == table:
                    value.pop(field)
                return value

            tx.get = missing
            yield tx

    monkeypatch.setattr(store, "_transaction", incomplete)
    with pytest.raises(CollaborationContractError):
        await application.inspect_participant(ref, context=CONTEXT)
    monkeypatch.setattr(store, "_transaction", original)
    assert await application.inspect_participant(ref, context=CONTEXT) == before
