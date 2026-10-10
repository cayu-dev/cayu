"""PostgreSQL grant operations own complete native transactions."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace

from tests.core.test_native_tool_grant_ownership import (
    TABLES,
    assert_owner_imports,
    codec,
    compose,
    exercise,
    seed,
)


def test_postgres_tool_grant_owner_imports_without_adapters_or_driver():
    assert_owner_imports("postgres")


def test_postgres_tool_grant_owner_composes_with_direct_connection(postgres_dsn):
    import psycopg
    from psycopg import sql

    from cayu.storage import _postgres_tool_grants as owner
    from cayu.storage import postgres as adapter
    from cayu.storage.migrations import SchemaMode

    async def run():
        alias_codec = codec()
        store = adapter.PostgresSessionStore(
            postgres_dsn, schema_mode=SchemaMode.CREATE, public_authority_alias_codec=alias_codec
        )
        try:
            session, record, issued = await seed(store, alias_codec)
        finally:
            await store.close()
        active = False
        ready = False
        fault = SimpleNamespace(enabled=False, reading=False)

        async def ensure_ready():
            nonlocal ready
            assert not active
            ready = True

        @asynccontextmanager
        async def connect():
            nonlocal active, ready
            assert ready and not active
            ready, active = False, True
            try:
                async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                    yield connection
            finally:
                active = False

        async def closure_owners(cur, targets):
            assert active and tuple(targets) == (session.id,)
            assert cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
            return ()

        async def register_event_authorities(cur, session_id, events):
            # Seeding registered this interaction. All subsequent events belong
            # to that scope and introduce no new interaction authority.
            assert active and session_id == session.id
            assert all(event.interaction_id == record.interaction_id for event in events)
            assert cur.connection.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS

        # Compose only the existing transaction-local event writers. No session
        # adapter instance is kept alive or supplied to the grant owner.
        writer = SimpleNamespace(
            _session_store_now=adapter.PostgresSessionStore._session_store_now,
            _publish_budget_reservation_identities=adapter.PostgresSessionStore._publish_budget_reservation_identities,
            _register_event_public_authorities=register_event_authorities,
        )
        for name in (
            "_record_invocation_terminal_event_receipts",
            "_insert_event_rows_with_cursor",
            "_append_events_with_cursor",
            "_append_event_once_with_cursor",
        ):
            setattr(writer, name, partial(getattr(adapter.PostgresSessionStore, name), writer))

        async def append_events(cur, session_id, events, *, expected_run_epoch):
            assert active
            await writer._append_events_with_cursor(
                cur, session_id, events, expected_run_epoch=expected_run_epoch
            )
            if fault.enabled:
                raise RuntimeError("audit write failed")

        async def append_once(cur, event, *, expected_run_epoch):
            assert active
            result = await writer._append_event_once_with_cursor(
                cur, event, expected_run_epoch=expected_run_epoch
            )
            if fault.enabled:
                raise RuntimeError("audit write failed")
            return result

        def get_codec():
            assert active is fault.reading
            return alias_codec

        ops = compose(
            owner,
            connect,
            dict(
                ensure_ready=ensure_ready,
                get_codec=get_codec,
                decode_grant=adapter._targeted_tool_grant_from_postgres_row,
                decode_use=adapter._targeted_tool_use_from_postgres_row,
                validate_use_counts=adapter._validate_targeted_tool_use_counts,
                register_alias=partial(
                    adapter.PostgresSessionStore._register_public_authority_alias_row, None
                ),
                append_events=append_events,
                append_event_once=append_once,
                closure_owners=closure_owners,
            ),
        )

        async def snapshot():
            assert not active
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
                values = []
                for table in TABLES:
                    cur = await connection.execute(
                        sql.SQL("SELECT * FROM {} ORDER BY 1").format(sql.Identifier(table))
                    )
                    values.append(tuple(await cur.fetchall()))
                cur = await connection.execute(
                    "SELECT event_seq, last_activity_at FROM cayu_sessions"
                )
                return (*values, tuple(await cur.fetchone()))

        await exercise(ops, session, record, issued, snapshot, fault)

    asyncio.run(run())
