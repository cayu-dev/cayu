"""Time the Developer dashboard's task reads on a production-sized task store.

Run with ``uv run python scripts/benchmark_task_index_reads.py``. It builds a SQLite
task store of ``--tasks`` tasks (150,000 by default) shaped like review.cayu.ai's:
most tasks finished, a third of them Cayu's own, a handful of types and agents,
graph prerequisites on some, spread over 60 days. ``--postgres DSN`` builds it in
an empty PostgreSQL database instead, and ``--store PATH`` reads an existing SQLite
task store (migrated in place, so pass a copy).

Each read runs first on a freshly opened store and server (``cold``) and then
``--samples`` more times. The reads are the Tasks page with counts and its
filters, Home's four lists, a task page with its relations, and dependents.

Template tasks are created through the store and copied with SQL with their IDs,
statuses, agents and times rewritten. The copy is for measurement only; it is not
a supported way to write a store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from cayu import CayuApp
from cayu.server import ServerConfig, create_server
from cayu.storage.migrations import SchemaMode
from cayu.tasks import TaskCreate
from cayu.tasks.records import Task, TaskStatus

TASKS = 150_000
DAYS = 60
_TYPES = (
    "review.pass",
    "review.recheck",
    "review.summary",
    "build.verify",
    "docs.check",
    "cayu.eval.trial",
    "cayu.session.continue",
)
_AGENTS = (None, "reviewer", "rechecker", "summarizer", "builder", "docs")
_STATUSES = (
    (TaskStatus.COMPLETED, 80),
    (TaskStatus.FAILED, 8),
    (TaskStatus.CANCELLED, 5),
    (TaskStatus.PENDING, 3),
    (TaskStatus.RUNNING, 1),
    (TaskStatus.CLAIMED, 1),
    (TaskStatus.BLOCKED, 1),
    (TaskStatus.NEEDS_ATTENTION, 1),
)
_GRAPH = "benchmark-graph"


def _tasks(templates: list[Task], count: int) -> list[Task]:
    rng = random.Random(3944)
    now = datetime.now(UTC)
    statuses = [status for status, weight in _STATUSES for _ in range(weight)]
    tasks: list[Task] = []
    for index in range(count):
        template = templates[index % len(templates)]
        updated = now - timedelta(seconds=rng.uniform(0, DAYS * 86_400))
        status = rng.choice(statuses)
        identity = f"bench-task-{index:07d}"
        prerequisites: tuple[str, ...] = ()
        # One task in twenty waits on one to three earlier tasks.
        if index > 10 and index % 20 == 0:
            prerequisites = tuple(
                sorted({f"bench-task-{rng.randrange(index):07d}" for _ in range(rng.randint(1, 3))})
            )
        tasks.append(
            template.model_copy(
                update={
                    "id": identity,
                    "title": f"{template.title} #{index}",
                    "status": status,
                    "assigned_agent_name": _AGENTS[index % len(_AGENTS)],
                    "created_at": updated - timedelta(minutes=5),
                    "updated_at": updated,
                    "graph_id": _GRAPH if prerequisites else None,
                    "prerequisite_task_ids": prerequisites,
                    "parent_task_id": None if index % 50 else f"bench-task-{index // 2:07d}",
                }
            )
        )
    return tasks


async def _templates(store) -> list[Task]:
    templates = []
    for index, task_type in enumerate(_TYPES):
        templates.append(
            await store.create_task(
                TaskCreate(
                    type=task_type,
                    title=f"Template {task_type}",
                    description="Review a pull request and report what needs fixing.",
                    input={"repository": "cayu-tech/example", "pull_request": index},
                    metadata={"source": "benchmark"},
                )
            )
        )
    return templates


async def build_sqlite(path: Path, count: int) -> None:
    from cayu.storage import _sqlite_records as records
    from cayu.storage.tasks_sqlite import SQLiteTaskStore

    store = SQLiteTaskStore(path)
    try:
        tasks = _tasks(await _templates(store), count)
        columns = [
            str(row[1]) for row in store._connection.execute("PRAGMA table_info(cayu_tasks)")
        ]
        insert = (
            "INSERT INTO cayu_tasks (id, type, title, description, status, session_id, "
            "session_instance_id, parent_task_id, assigned_agent_name, available_at, worker_id, "
            "lease_expires_at, interrupted_handoff_id, status_reason, status_payload_json, "
            "input_json, result_json, error_json, metadata_json, created_at, updated_at, "
            "started_at, completed_at, invocation_json, retry_series_json, work_contract_json, "
            "schedule_json, restartable, restart_count, concurrency_key, max_restarts, "
            "restart_history_json, claim_generation) VALUES (" + ", ".join(["?"] * 33) + ")"
        )
        assert "graph_id" in columns
        with store._connection:
            for start in range(0, len(tasks), 5_000):
                batch = tasks[start : start + 5_000]
                store._connection.executemany(
                    insert, [records.task_to_row_values(task) for task in batch]
                )
                store._connection.executemany(
                    "UPDATE cayu_tasks SET graph_id = ?, prerequisite_task_ids_json = ? "
                    "WHERE id = ?",
                    [
                        (task.graph_id, json.dumps(task.prerequisite_task_ids), task.id)
                        for task in batch
                        if task.prerequisite_task_ids
                    ],
                )
    finally:
        await store.close()


async def build_postgres(dsn: str, count: int) -> None:
    from cayu.storage import _postgres_support as pg
    from cayu.storage.tasks_postgres import PostgresTaskStore

    store = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
    try:
        tasks = _tasks(await _templates(store), count)
        marks = ", ".join(["%s"] * len(pg.TASK_COLUMNS.split(", ")))
        async with store._ready_connection() as connection, connection.cursor() as cursor:
            for start in range(0, len(tasks), 5_000):
                await cursor.executemany(
                    f"INSERT INTO cayu_tasks ({pg.TASK_COLUMNS}) VALUES ({marks})",  # type: ignore[arg-type]
                    [pg.task_insert_values(task) for task in tasks[start : start + 5_000]],
                )
            await cursor.execute("ANALYZE cayu_tasks")
    finally:
        await store.close()


def _open_store(target: str):
    if target.startswith("postgresql://") or target.startswith("host="):
        from cayu.storage.tasks_postgres import PostgresTaskStore

        return PostgresTaskStore(target, schema_mode=SchemaMode.MIGRATE)
    from cayu.storage.tasks_sqlite import SQLiteTaskStore

    return SQLiteTaskStore(Path(target), schema_mode=SchemaMode.MIGRATE)


@asynccontextmanager
async def _server(target: str) -> AsyncIterator[httpx.AsyncClient]:
    store = _open_store(target)
    try:
        # A server opens its store at startup; the first request should not pay for it.
        ensure_schema = getattr(store, "ensure_schema", None)
        if ensure_schema is not None:
            await ensure_schema()
        app = CayuApp(task_store=store, enable_logging=False)
        config = ServerConfig.local_development()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_server(app, config=config)),
            base_url="http://bench",
            timeout=600,
        ) as client:
            yield client
    finally:
        await store.close()


def _reads(sample: dict[str, Any]) -> list[tuple[str, str, Callable[[], Any]]]:
    index = {"include": "counts", "include_internal": "false", "limit": 50}
    task_id, prerequisites, parent = sample["task"], sample["prerequisites"], sample["parent"]
    home = [
        ("Home recent", "/api/tasks", lambda: {"limit": 8, "order_by": "updated_at_desc"}),
        *[
            (
                f"Home {status}",
                "/api/tasks",
                lambda status=status: {"status": status, "limit": 8, "order_by": "updated_at_desc"},
            )
            for status in ("blocked", "needs_attention", "failed")
        ],
    ]
    return [
        ("Tasks page", "/api/tasks", lambda: index),
        ("Tasks page, Cayu's too", "/api/tasks", lambda: {**index, "include_internal": "true"}),
        (
            "Tasks page, Done",
            "/api/tasks",
            lambda: {**index, "statuses": ["completed", "cancelled"]},
        ),
        ("Tasks page, Failed", "/api/tasks", lambda: {**index, "statuses": ["failed"]}),
        ("Tasks page, type", "/api/tasks", lambda: {**index, "type": "review.recheck"}),
        ("Tasks page, agent", "/api/tasks", lambda: {**index, "assigned_agent_name": "reviewer"}),
        (
            "Tasks page, type and agent and Active",
            "/api/tasks",
            lambda: {
                **index,
                "type": "review.pass",
                "assigned_agent_name": "reviewer",
                "statuses": ["claimed", "running"],
            },
        ),
        ("Tasks page, search", "/api/tasks", lambda: {**index, "q": "#12345"}),
        ("Tasks page, broad search", "/api/tasks", lambda: {**index, "q": "recheck"}),
        *home,
        ("Task", f"/api/tasks/{task_id}", dict),
        ("Task relations", "/api/tasks", lambda: {"ids": [*prerequisites, parent]}),
        ("Task dependents", f"/api/tasks/{prerequisites[0]}/dependents", dict),
        (
            "Task children",
            "/api/tasks",
            lambda: {
                "parent_task_id": parent,
                "include_internal": "true",
                "include": "counts",
                "limit": 50,
            },
        ),
    ]


async def _sample(target: str) -> dict[str, Any]:
    store = _open_store(target)
    try:
        from cayu.tasks.queries import TaskQuery

        tasks = await store.list_tasks(TaskQuery(limit=1000))
        task = next((t for t in tasks if t.prerequisite_task_ids), tasks[0])
        parent = next((t.parent_task_id for t in tasks if t.parent_task_id), task.id)
        return {
            "task": task.id,
            "prerequisites": list(task.prerequisite_task_ids) or [task.id],
            "parent": parent,
        }
    finally:
        await store.close()


async def measure(target: str, samples: int) -> list[dict[str, Any]]:
    sample = await _sample(target)
    results = []
    for name, path, params in _reads(sample):
        timings = []
        async with _server(target) as client:
            for _ in range(samples + 1):
                started = time.perf_counter()
                response = await client.get(path, params=params())
                timings.append(time.perf_counter() - started)
                response.raise_for_status()
        warm = timings[1:] or timings
        results.append(
            {
                "read": name,
                "cold_ms": round(timings[0] * 1000, 1),
                "p50_ms": round(statistics.median(warm) * 1000, 1),
                "p95_ms": round(sorted(warm)[max(0, int(len(warm) * 0.95) - 1)] * 1000, 1),
            }
        )
    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Time the dashboard's task reads.")
    parser.add_argument("--tasks", type=int, default=TASKS)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--store", type=Path, help="Read this SQLite task store (migrated).")
    parser.add_argument("--keep", type=Path, help="Build the SQLite store here and keep it.")
    parser.add_argument("--postgres", help="Build the store in this empty PostgreSQL database.")
    parser.add_argument(
        "--no-build", action="store_true", help="With --postgres, read the store already built."
    )
    args = parser.parse_args()
    if args.postgres and args.no_build:
        target = args.postgres
    elif args.postgres:
        started = time.perf_counter()
        await build_postgres(args.postgres, args.tasks)
        print(f"built {args.tasks} tasks in {time.perf_counter() - started:.1f} s")
        target = args.postgres
    elif args.store:
        target = str(args.store)
    else:
        path = args.keep or Path(tempfile.mkdtemp()) / "tasks.db"
        started = time.perf_counter()
        await build_sqlite(path, args.tasks)
        print(f"built {args.tasks} tasks in {time.perf_counter() - started:.1f} s")
        target = str(path)
    for row in await measure(target, args.samples):
        print(json.dumps(row))


if __name__ == "__main__":
    asyncio.run(main())
