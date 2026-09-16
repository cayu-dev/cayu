"""Fresh-process SDK fixture for identity commit/acknowledgement boundaries."""

from __future__ import annotations

import asyncio
import sys

from tests.core.test_participant_identity import CONTEXT, app, create, registration

from cayu.storage.collaboration_postgres import PostgresCollaborationStore
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.storage.migrations import SchemaMode


async def main() -> None:
    backend, address, scope, phase = sys.argv[1:]
    store = (
        SQLiteCollaborationStore(address)
        if backend == "sqlite"
        else PostgresCollaborationStore(address, schema_mode=SchemaMode.CREATE)
    )
    application = app(store, registration(scope=scope))
    if phase in ("bootstrap", "mutation"):
        method = "_initialize" if phase == "bootstrap" else "_apply"
        original = getattr(store, method)

        async def committed(*args, **kwargs):
            result = await original(*args, **kwargs)
            print(result.model_dump_json(), flush=True)
            await asyncio.Event().wait()
            return result

        setattr(store, method, committed)
    initialized = await application.initialize_collaboration()
    _, receipt = await create(application, initialized, alias="reviewer")
    assert len((await application.discover_participants(context=CONTEXT)).participants) == 1
    assert len((await application.list_participant_events(context=CONTEXT)).events) == 2
    print(receipt.model_dump_json(), flush=True)
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
