"""Public wait discovery retains exact intent independently of the host process."""

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_collaboration_waits import wait_for
from tests.core.test_participant_identity import CONTEXT

from cayu import WaitDiscoveryCursor
from cayu.collaboration._contracts import ExactConflict, ExactMatch, ExactNotFound
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


@pytest.mark.parametrize("limit", [1, 32])
async def test_wait_discovery_has_canonical_order_after_reopen(stores, limit):
    from cayu.storage.collaboration_postgres import PostgresCollaborationStore

    application, resolver, values, first, second = await scenario(stores, larger_inventory=True)
    store = application._participant_coordinator._store
    scope = values[1].owner.application_scope
    postgres = isinstance(store, PostgresCollaborationStore)
    if postgres:
        async with store._transaction(scope, write=True) as tx:
            available = await tx._rows(
                await tx._execute("SELECT to_regcollation(?)", ('"en-x-icu"',))
            )
            if available == [(None,)]:
                pytest.skip("Non-C wait ordering qualification requires the en-x-icu collation.")
            await tx._execute(
                "ALTER TABLE cayu_collaboration_wait_discovery "
                'ALTER COLUMN caller_key TYPE TEXT COLLATE "en-x-icu"'
            )
    try:
        expected = {first.operation.caller_key: first, second.operation.caller_key: second}
        for key in ("a", "B", "Z", "é", "z", "Å"):
            wait = first.model_copy(update={"operation": values[1].operation(key)})
            await application.register_collaboration_wait(wait, context=resolver.context)
            expected[key] = wait
        if postgres:
            async with store._transaction(scope, write=False) as tx:
                rows = await tx._rows(
                    await tx._execute(
                        "SELECT caller_key FROM cayu_collaboration_wait_discovery "
                        "WHERE scope=? ORDER BY caller_key",
                        (scope,),
                    )
                )
                # Prove this fixture actually exposes the locale mismatch.
                assert [row[0] for row in rows] != sorted(expected)
        cursor = None
        seen = []
        while True:
            # Reconstruct the application between pages; only the public cursor
            # is carried across the independent persistent-store connections.
            other = identity_tests.app(
                stores(),
                application._participant_coordinator._registration,
                collaboration_requests=application._request_coordinator._registration,
            )
            await other.initialize_collaboration()
            page = await other.list_collaboration_waits(context=CONTEXT, cursor=cursor, limit=limit)
            for item in page.items:
                key = item.recovery.operation.caller_key
                assert key not in seen
                seen.append(key)
                recovered = await other.recover_collaboration_wait(
                    item.recovery, context=resolver.context
                )
                assert isinstance(recovered, ExactMatch) and recovered.receipt == expected[key]
            cursor = page.next_cursor
            if cursor is None:
                break
            assert len(seen) <= len(expected)
        assert seen == sorted(expected)
    finally:
        if postgres:
            async with store._transaction(scope, write=True) as tx:
                await tx._execute(
                    "ALTER TABLE cayu_collaboration_wait_discovery "
                    'ALTER COLUMN caller_key TYPE TEXT COLLATE "default"'
                )


async def scenario(stores, *, larger_inventory=False):
    base = identity_tests.registration()
    reg = identity_tests.registration(
        limits=base.bootstrap.limits.model_copy(
            update={"events": 4096, "retained_bytes": 16 * 1024 * 1024}
            if larger_inventory
            else {"events": 1024}
        )
    )
    application, resolver, values = await public_setup(stores(), reg=reg)
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    first = wait_for(receipt, values[1])
    second = first.model_copy(update={"operation": values[1].operation("wait-two")})
    await application.register_collaboration_wait(first, context=resolver.context)
    await application.register_collaboration_wait(second, context=resolver.context)
    return application, resolver, values, first, second


@pytest.mark.parametrize("stores", ["postgres"], indirect=True)
@pytest.mark.parametrize("invalid", ["missing", "locale"])
async def test_wait_discovery_reopen_rejects_invalid_ordering_index(stores, invalid):
    from cayu.storage._collaboration_wait_schema import POSTGRES_COLLABORATION_WAIT_DDL
    from cayu.storage.migrations import SchemaError

    application, _, values, _, _ = await scenario(stores)
    store = application._participant_coordinator._store
    scope = values[1].owner.application_scope
    name = "idx_cayu_collaboration_wait_discovery_order"
    try:
        async with store._transaction(scope, write=True) as tx:
            await tx._execute(f"DROP INDEX {name}")
            if invalid == "locale":
                await tx._execute(
                    f"CREATE INDEX {name} ON cayu_collaboration_wait_discovery "
                    "(scope, namespace, generation, caller_key)"
                )
        with pytest.raises(SchemaError, match="ordering index"):
            await stores().ensure_schema()
    finally:
        async with store._transaction(scope, write=True) as tx:
            await tx._execute(f"DROP INDEX IF EXISTS {name}")
            for statement in POSTGRES_COLLABORATION_WAIT_DDL:
                if statement.startswith(f"CREATE INDEX IF NOT EXISTS {name} "):
                    await tx._execute(statement)
        await stores().ensure_schema()


async def test_public_wait_discovery_reconstructs_exact_intent_after_reopen(stores):
    application, resolver, _values, first, second = await scenario(stores)
    page = await application.list_collaboration_waits(context=CONTEXT, limit=1)
    assert len(page.items) == 1 and page.next_cursor is not None
    assert page.items[0].recovery.operation == first.operation
    assert page.items[0].state == "pending"
    other = identity_tests.app(
        stores(),
        application._participant_coordinator._registration,
        collaboration_requests=application._request_coordinator._registration,
    )
    await other.initialize_collaboration()
    recovered = await other.recover_collaboration_wait(
        page.items[0].recovery, context=resolver.context
    )
    assert isinstance(recovered, ExactMatch) and recovered.receipt == first
    tail = await other.list_collaboration_waits(context=CONTEXT, cursor=page.next_cursor, limit=1)
    assert tail.items[0].recovery.operation == second.operation
    end = await other.list_collaboration_waits(context=CONTEXT, cursor=tail.next_cursor, limit=1)
    assert end.items == () and end.next_cursor is None
    assert await application.inspect_collaboration_wait(first, context=resolver.context) is not None


async def test_public_wait_recovery_conflict_and_current_access(stores):
    application, resolver, values, first, _ = await scenario(stores)
    page = await application.list_collaboration_waits(context=CONTEXT)
    token = page.items[0].recovery
    conflict = token.model_copy(update={"registration_digest": "0" * 64})
    assert isinstance(
        await application.recover_collaboration_wait(conflict, context=resolver.context),
        ExactConflict,
    )
    missing = token.model_copy(update={"operation": values[1].operation("missing-wait")})
    assert isinstance(
        await application.recover_collaboration_wait(missing, context=resolver.context),
        ExactNotFound,
    )
    future = token.model_copy(
        update={
            "operation": token.operation.model_copy(
                update={"generation": token.operation.generation + 1}
            )
        }
    )
    assert isinstance(
        await application.recover_collaboration_wait(future, context=resolver.context),
        ExactConflict,
    )
    with pytest.raises(CollaborationAccessDenied):
        await application.list_collaboration_waits(
            context=CollaborationAccessContext(principal="foreign")
        )
    with pytest.raises(CollaborationAccessDenied):
        await application.recover_collaboration_wait(
            token, context=resolver.context.model_copy(update={"principal": "foreign"})
        )
    assert await application.inspect_collaboration_wait(first, context=resolver.context) is not None


async def test_wait_index_tracks_terminal_state_without_erasing_recovery(stores):
    application, resolver, _, first, _ = await scenario(stores)
    before = await application.list_collaboration_waits(context=CONTEXT, limit=1)
    await application.cancel_collaboration_wait(first, context=resolver.context)
    after = await application.list_collaboration_waits(context=CONTEXT, limit=1)
    assert before.items[0].recovery == after.items[0].recovery
    assert after.items[0].state == "cancelled"
    assert after.items[0].revision > before.items[0].revision
    recovered = await application.recover_collaboration_wait(
        after.items[0].recovery, context=resolver.context
    )
    assert isinstance(recovered, ExactMatch) and recovered.receipt == first


async def test_wait_discovery_rejects_cross_namespace_cursor_and_unbounded_page(stores):
    application, _, _, first, _ = await scenario(stores)
    with pytest.raises(ValueError):
        await application.list_collaboration_waits(context=CONTEXT, limit=True)
    with pytest.raises(ValueError):
        await application.list_collaboration_waits(context=CONTEXT, limit=33)
    cursor = WaitDiscoveryCursor(
        operation=first.operation.model_copy(update={"namespace_incarnation": "foreign"})
    )
    with pytest.raises(ValueError, match="another owner"):
        await application.list_collaboration_waits(context=CONTEXT, cursor=cursor)


async def test_wait_discovery_projection_rebuilds_from_retained_source(stores):
    from cayu.collaboration.memory import InMemoryCollaborationStore
    from cayu.storage._collaboration_wait_schema import _ddl
    from cayu.storage.collaboration_postgres import PostgresCollaborationStore

    application, resolver, values, first, _ = await scenario(stores)
    await application.cancel_collaboration_wait(first, context=resolver.context)
    before = await application.list_collaboration_waits(context=CONTEXT)
    store = application._participant_coordinator._store
    if isinstance(store, InMemoryCollaborationStore):
        # Memory projects canonical records directly; persistent owners backfill
        # the same records with the actual forward migration statements.
        assert {item.state for item in before.items} == {"pending", "cancelled"}
        return
    async with store._transaction(values[1].owner.application_scope, write=True) as tx:
        await tx._execute("DROP TABLE cayu_collaboration_wait_discovery")
        for statement in _ddl(isinstance(store, PostgresCollaborationStore)):
            await tx._execute(statement)
    other = identity_tests.app(
        stores(),
        application._participant_coordinator._registration,
        collaboration_requests=application._request_coordinator._registration,
    )
    await other.initialize_collaboration()
    assert await other.list_collaboration_waits(context=CONTEXT) == before
