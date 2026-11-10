"""A continuation candidate claimed concurrently does not end the PostgreSQL scan.

The scan selects candidate IDs without locks to take their graph locks first,
then locks the rows. A candidate another worker claims in between drops out of
the locked rows; the page is exhausted only when the unlocked selection was.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_postgres_continuation_lease_clock import _ClockedCursor, _Connection

from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery
from cayu.tasks.handoff import interrupted_task_handoff_request

_SCAN_LIMIT = 2
_TAKEN_TASK_ID = "taken-concurrently"


def test_a_concurrently_claimed_candidate_does_not_exhaust_the_page():
    async def run():
        memory = InMemoryTaskStore()
        await memory.create_task(TaskCreate(task_id="continuation", type="review"))
        original = await memory.claim_task("original")
        assert original is not None and original.worker_id is not None
        attached = await memory.attach_task(
            original.id,
            session_id="session",
            session_invocation=await task_backed_session_invocation(memory, original.id, "session"),
            worker_id=original.worker_id,
            lease_expires_at=original.lease_expires_at,
        )
        receipt = await memory.release_interrupted_task_worker(
            interrupted_task_handoff_request(attached, session_run_epoch=1)
        )
        cursor = _ClockedCursor(receipt)
        # A full selection; one candidate is claimed before the rows are locked.
        cursor.candidate_ids = [original.id, _TAKEN_TASK_ID]
        connection = _Connection(cursor)
        store = object.__new__(PostgresTaskStore)

        @asynccontextmanager
        async def owned_connection(_owner):
            yield connection

        store._owned_store_connection = owned_connection
        page = await store._claim_interrupted_task_continuation_unowned(
            "replacement",
            # The remaining candidate is filtered, so nothing is claimed.
            query=TaskQuery(type="other"),
            handoff_id="replacement-handoff",
            task_id=None,
            expected=None,
            expected_sha256=None,
            lease_seconds=1,
            after=None,
            scan_limit=_SCAN_LIMIT,
            connection_owner=None,
        )
        assert (page.task, page.scanned_candidates, page.filtered_candidates) == (None, 1, 1)
        assert not page.exhausted
        assert page.next_after == (receipt.task.created_at, receipt.task.id)

    asyncio.run(run())
