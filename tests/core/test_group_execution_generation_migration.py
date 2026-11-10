"""Revision 146 copies each member's claim generation into its retained group executions.

Older binaries recorded a group execution by worker name and start time only.
The migration runs with every worker stopped, so each member's current claim
generation names the claim of its retained execution.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from tests.core.test_task_group_quiescence import _postgres_address
from tests.core.test_task_group_quiescence import store as store
from tests.core.test_task_groups import anyio_backend as anyio_backend

from cayu import CayuApp
from cayu.storage import migrations as m
from cayu.storage.migrations import SchemaMode
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.storage.tasks_postgres import PostgresTaskStore
from cayu.tasks import (
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
)
from cayu.tasks.groups import TaskGroupSnapshot
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery

pytestmark = pytest.mark.anyio

_PRIOR_REVISION = 145
# A generation distinct from the claim's, so the copy is observable.
_MIGRATED_GENERATION = 7
_DELETED_TASK_ID = "deleted"
# A deleted member's settled execution records the initial generation.
_DELETED_GENERATION = 0


async def _started_member(store) -> int:
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(
        TaskGroupCreate(
            group_id="legacy",
            graph=TaskGraphCreate(
                graph_id="legacy-graph",
                nodes=(
                    TaskGraphNode(task=TaskCreate(task_id="member", type="member")),
                    TaskGraphNode(task=TaskCreate(task_id="finalize", type="finalize")),
                ),
            ),
            member_task_ids=("member",),
            policy=TaskGroupPolicy(kind="all"),
            quiescence=TaskGroupQuiescencePolicy(timeout_seconds=60),
            finalizer_task_id="finalize",
        )
    )
    claimed = await store.claim_task("owner", TaskQuery(type="member"))
    assert claimed is not None
    await store.mark_claimed_task_execution_started(
        claimed.id, "owner", claim_generation=claimed.claim_generation
    )
    snapshot = await store.load_task_group("legacy")
    assert snapshot.quiescence.executions[0].claim_generation == claimed.claim_generation
    return claimed.claim_generation


def _legacy(snapshot_json: str) -> str:
    snapshot = json.loads(snapshot_json)
    executions = snapshot["quiescence"]["executions"]
    for execution in executions:
        del execution["claim_generation"]
    # A settled execution whose task row was deleted, as a session closure may do
    # once the group is quiescent; ordered before "member".
    executions.insert(0, {**executions[0], "task_id": _DELETED_TASK_ID})
    executions[0]["settled_at"] = executions[0]["started_at"]
    return json.dumps(snapshot)


def _generations(snapshot_json: str) -> list[int]:
    snapshot = TaskGroupSnapshot.model_validate_json(snapshot_json)
    return [execution.claim_generation for execution in snapshot.quiescence.executions]


async def test_migration_copies_claim_generation_into_retained_executions(store, tmp_path):
    if isinstance(store, InMemoryTaskStore):
        pytest.skip("The memory store keeps no durable group evidence.")
    claimed_generation = await _started_member(store)
    assert claimed_generation != _MIGRATED_GENERATION
    if isinstance(store, SQLiteTaskStore):
        await store.close()
        database = tmp_path / "barriers.sqlite"
        connection = sqlite3.connect(database)
        try:
            (snapshot_json,) = connection.execute(
                "SELECT snapshot_json FROM cayu_task_groups WHERE group_id = 'legacy'"
            ).fetchone()
            legacy = _legacy(snapshot_json)
            connection.execute(
                "UPDATE cayu_task_groups SET snapshot_json = ? WHERE group_id = 'legacy'",
                (legacy,),
            )
            connection.execute(
                "INSERT INTO cayu_task_group_resolutions "
                "(group_id, idempotency_key, request_sha256, snapshot_json) VALUES (?, ?, ?, ?)",
                ("legacy", "resolved", "0" * 64, legacy),
            )
            connection.execute(
                "UPDATE cayu_tasks SET claim_generation = ? WHERE id = 'member'",
                (_MIGRATED_GENERATION,),
            )
            connection.execute(
                "DELETE FROM cayu_schema_migrations WHERE revision > ?", (_PRIOR_REVISION,)
            )
            connection.execute(f"PRAGMA user_version = {_PRIOR_REVISION}")
            connection.commit()
        finally:
            connection.close()
        migrated = SQLiteTaskStore(database, schema_mode=SchemaMode.MIGRATE)
        try:
            snapshot = await migrated.load_task_group("legacy")
        finally:
            await migrated.close()
        connection = sqlite3.connect(database)
        try:
            (resolution_json,) = connection.execute(
                "SELECT snapshot_json FROM cayu_task_group_resolutions"
            ).fetchone()
        finally:
            connection.close()
    else:
        import psycopg

        assert isinstance(store, PostgresTaskStore)
        address = _postgres_address(store)
        async with await psycopg.AsyncConnection.connect(address) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT snapshot_json::text FROM cayu_task_groups WHERE group_id = 'legacy'"
                )
                row = await cursor.fetchone()
                assert row is not None
                (snapshot_json,) = row
                legacy = _legacy(snapshot_json)
                await cursor.execute(
                    "UPDATE cayu_task_groups SET snapshot_json = %s::jsonb "
                    "WHERE group_id = 'legacy'",
                    (legacy,),
                )
                await cursor.execute(
                    "INSERT INTO cayu_task_group_resolutions "
                    "(group_id, idempotency_key, request_sha256, snapshot_json) "
                    "VALUES (%s, %s, %s, %s::jsonb)",
                    ("legacy", "resolved", "0" * 64, legacy),
                )
                await cursor.execute(
                    "UPDATE cayu_tasks SET claim_generation = %s WHERE id = 'member'",
                    (_MIGRATED_GENERATION,),
                )
                await cursor.execute(
                    "DELETE FROM cayu_schema_migrations WHERE revision > %s", (_PRIOR_REVISION,)
                )
            await connection.commit()
        migrated = PostgresTaskStore(address, schema_mode=SchemaMode.MIGRATE)
        try:
            snapshot = await migrated.load_task_group("legacy")
        finally:
            await migrated.close()
        async with (
            await psycopg.AsyncConnection.connect(address) as connection,
            connection.cursor() as cursor,
        ):
            await cursor.execute("SELECT snapshot_json::text FROM cayu_task_group_resolutions")
            row = await cursor.fetchone()
            assert row is not None
            (resolution_json,) = row
    assert snapshot is not None
    expected = [_DELETED_GENERATION, _MIGRATED_GENERATION]
    assert [item.claim_generation for item in snapshot.quiescence.executions] == expected
    assert _generations(resolution_json) == expected


def test_group_execution_generations_fence_old_binaries() -> None:
    current = m.SchemaState(revision=146, compatible_from=146)
    with pytest.raises(m.SchemaTooNew):
        m.validate(current, app_latest=_PRIOR_REVISION, app_min_supported=_PRIOR_REVISION)
    with pytest.raises(m.SchemaTooOld):
        m.validate(m.SchemaState(revision=_PRIOR_REVISION, compatible_from=_PRIOR_REVISION))
    prior = m.SchemaState(revision=_PRIOR_REVISION, compatible_from=_PRIOR_REVISION)
    assert m.validate_migration_input(prior)[0].revision == 146
