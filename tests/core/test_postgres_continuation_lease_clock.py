"""Continuation leases must start after, not before, candidate-lock waits."""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from tests.core.task_invocation_fixtures import task_backed_session_invocation

from cayu.storage import _postgres_support as pg_support
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery
from cayu.tasks.handoff import interrupted_task_handoff_request


class _ClockedCursor:
    """Small driver double with PostgreSQL's two distinct timestamp semantics.

    Acquiring the candidate row advances wall time by two seconds, without
    advancing transaction_timestamp(). No real database or sleep is needed.
    All task/receipt rows still pass through the production PostgreSQL codecs.
    """

    def __init__(self, receipt):
        self.receipt = receipt
        self.transaction_now = datetime(2026, 9, 1, tzinfo=UTC)
        self.wall_now = self.transaction_now
        self.rows = []
        self.claimed = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def execute(self, statement, parameters=()):
        statement = " ".join(str(statement).split())
        self.rows = []
        if statement.startswith("SELECT pg_advisory_xact_lock"):
            return
        if statement == "SELECT clock_timestamp()":
            self.rows = [(self.wall_now,)]
            return
        if statement == "SELECT transaction_timestamp()":
            self.rows = [(self.transaction_now,)]
            return
        if statement.startswith("SELECT task_id, worker_id, expected_sha256"):
            return  # No earlier claim with this handoff identity.
        if statement.startswith("SELECT 1 FROM cayu_tasks AS task"):
            return  # The new handoff identity is not in use.
        if statement.startswith("SELECT 1 FROM cayu_work_attempt_admissions"):
            return
        if statement.startswith("SELECT request_sha256, request_json, task_json"):
            self.rows = [
                (
                    self.receipt.request_sha256,
                    self.receipt.request.model_dump(mode="json"),
                    self.receipt.task.model_dump(mode="json"),
                    self.receipt.committed_at,
                )
            ]
            return
        if statement.startswith("INSERT INTO cayu_task_interrupted_continuation_claims"):
            return
        if statement.startswith("SELECT ") and "FOR UPDATE OF task LIMIT" in statement:
            self.wall_now += timedelta(seconds=2)
            self.rows = [pg_support.task_insert_values(self.receipt.task)]
            return
        if "UPDATE cayu_tasks AS task" in statement:
            # Bind placeholders before evaluating the timestamp expression, so
            # this also accepts a fix using a sampled, parameter-bound DB time.
            values = {}
            for index, parameter in enumerate(parameters):
                marker = f"bound_{index}"
                statement = statement.replace("%s", marker, 1)
                values[marker] = parameter
            match = re.search(
                r"lease_expires_at = (transaction_timestamp\(\)|clock_timestamp\(\)|bound_\d+)"
                r"\s*(?:::\w+)?\s*\+\s*\(?(bound_\d+) \* INTERVAL '1 second'",
                statement,
            )
            assert match is not None, statement
            basis, duration = match.groups()
            if basis == "transaction_timestamp()":
                lease_start = self.transaction_now
            elif basis == "clock_timestamp()":
                lease_start = self.wall_now
            else:
                lease_start = values[basis]
            worker = re.search(r"SET worker_id = (bound_\d+)", statement).group(1)
            handoff = re.search(r"interrupted_handoff_id = (bound_\d+)", statement).group(1)
            self.claimed = self.receipt.task.model_copy(
                update={
                    "worker_id": values[worker],
                    "interrupted_handoff_id": values[handoff],
                    "lease_expires_at": lease_start + timedelta(seconds=values[duration]),
                    "updated_at": lease_start,
                }
            )
            self.rows = [pg_support.task_insert_values(self.claimed)]
            return
        raise AssertionError(f"Unexpected statement: {statement}")

    async def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    async def fetchall(self):
        result, self.rows = self.rows, []
        return result


class _Connection:
    def __init__(self, cursor):
        self.cursor_value = cursor
        self.committed = False

    def cursor(self):
        return self.cursor_value

    async def commit(self):
        self.committed = True


def test_continuation_claim_returns_a_live_lease_after_candidate_lock_wait():
    async def run():
        # Obtain authentic task and handoff evidence using the same setup as
        # the store conformance tests; only the PostgreSQL driver is doubled.
        memory = InMemoryTaskStore()
        await memory.create_task(TaskCreate(task_id="continuation", type="review"))
        original = await memory.claim_task("original")
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
        connection = _Connection(cursor)
        store = object.__new__(PostgresTaskStore)

        @asynccontextmanager
        async def owned_connection(_owner):
            yield connection

        store._owned_store_connection = owned_connection
        page = await store._claim_interrupted_task_continuation_unowned(
            "replacement",
            query=TaskQuery(type="review"),
            handoff_id="replacement-handoff",
            task_id=original.id,
            expected=None,
            expected_sha256=None,
            lease_seconds=1,
            after=None,
            scan_limit=1,
            connection_owner=None,
        )
        assert connection.committed
        assert page.task is not None
        assert page.task.lease_expires_at > cursor.wall_now, (
            "A successful continuation claim published an already-expired lease: "
            f"expires={page.task.lease_expires_at}, database_now={cursor.wall_now}"
        )

    asyncio.run(run())
