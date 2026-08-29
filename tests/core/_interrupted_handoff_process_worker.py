"""Real-process helper for interrupted-task handoff acknowledgement loss."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from cayu import SQLiteTaskStore, TaskInterruptedHandoffRequest


async def _commit_and_exit(
    backend: str,
    location: str,
    request_json: str,
) -> None:
    if backend == "sqlite":
        store = SQLiteTaskStore(Path(location))
    elif backend == "postgres":
        from cayu import PostgresTaskStore
        from cayu.storage.migrations import SchemaMode

        dsn = os.environ["CAYU_TEST_INTERRUPTED_HANDOFF_POSTGRES_DSN"]
        store = PostgresTaskStore(
            dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
    else:
        raise ValueError("Unsupported interrupted-handoff process-test backend.")
    request = TaskInterruptedHandoffRequest.model_validate_json(request_json)
    await store.release_interrupted_task_worker(request)
    os._exit(23)


if __name__ == "__main__":
    asyncio.run(_commit_and_exit(sys.argv[1], sys.argv[2], sys.argv[3]))
