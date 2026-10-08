"""Database-clock policy ownership and atomic publication on PostgreSQL."""

import time

from psycopg.rows import tuple_row

from cayu.runtime._policy_storage import (
    POLICY_STORAGE_REVISION,
    ModelPolicyStore,
    PolicyStorageCommand,
    PolicyStorageView,
    transition,
)
from cayu.runtime._policy_wire import require
from cayu.storage._postgres_base import _PostgresStoreBase


class PostgresModelPolicyStore(_PostgresStoreBase, ModelPolicyStore):
    _min_required_revision = POLICY_STORAGE_REVISION

    async def execute(self, command: PolicyStorageCommand) -> PolicyStorageView:
        command.validate()
        await self._ensure_ready()
        async with (
            self._connection() as connection,
            connection.cursor(row_factory=tuple_row) as cur,
        ):
            await cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (command.key,)
            )
            await cur.execute(
                "SELECT binding, owner, expires, revision, state "
                "FROM cayu_model_policy_state WHERE binding_id=%s FOR UPDATE",
                (command.key,),
            )
            row = await cur.fetchone()
            await cur.execute("SELECT extract(epoch FROM clock_timestamp())::double precision")
            now = (await cur.fetchone())[0]
            if row is not None:
                row = (bytes(row[0]), row[1], row[2], row[3], bytes(row[4]))
            updated, view = transition(row, command, now)
            await cur.execute(
                "INSERT INTO cayu_model_policy_state VALUES (%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT(binding_id) DO UPDATE SET owner=excluded.owner, "
                "expires=excluded.expires, revision=excluded.revision, state=excluded.state",
                (command.key, *updated),
            )
            require(command.deadline_ns is None or time.monotonic_ns() < command.deadline_ns)
            await connection.commit()
        return view
