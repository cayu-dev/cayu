"""Commit a collaboration wait, then lose the owner before child registration."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_collaboration_waits import wait_for

from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode


async def main() -> None:
    backend, address, output = sys.argv[1:]
    store = (
        SQLiteCollaborationStore(address)
        if backend == "sqlite"
        else PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
    )
    application, resolver, values = await public_setup(store)
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1])
    Path(output).write_text(
        json.dumps(
            {
                "scope": values[1].binding.application_scope,
                "expected": source.expected.model_dump(mode="json"),
                "wait": wait.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )

    original = store._transaction

    @asynccontextmanager
    async def lose_ack(scope, *, write):
        async with original(scope, write=write) as tx:
            yield tx
        if write:
            os._exit(19)

    store._transaction = lose_ack
    await application.register_collaboration_wait(wait, context=resolver.context)
    raise AssertionError("wait registration acknowledgement must be lost")


if __name__ == "__main__":
    asyncio.run(main())
