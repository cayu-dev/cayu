"""Crash after committed request control and before caller acknowledgement."""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from tests.core.test_collaboration_request_foundation import public_setup

from cayu.collaboration.requests import RequestControl
from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode


async def main():
    backend, output = sys.argv[1:]
    address = os.environ["CAYU_REQUEST_TEST_STORE"]
    store = (
        SQLiteCollaborationStore(address)
        if backend == "sqlite"
        else PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
    )
    application, resolver, values = await public_setup(store)
    accepted = await application.accept_collaboration_request(values[4], context=resolver.context)
    Path(output).write_text(
        json.dumps(
            {
                "binding": values[1].binding.model_dump(mode="json"),
                "expected": accepted.expected.model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    original = store._transaction

    @asynccontextmanager
    async def transaction(scope, *, write):
        async with original(scope, write=write) as tx:
            yield tx
        if write:
            os._exit(19)

    store._transaction = transaction
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("control"),
            expected=accepted.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    raise AssertionError("Control must lose its acknowledgement.")


if __name__ == "__main__":
    asyncio.run(main())
