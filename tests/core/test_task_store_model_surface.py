"""Native task stores refuse operator-only methods during model tool execution."""

from __future__ import annotations

import asyncio
from inspect import iscoroutinefunction

import pytest
from tests.core.test_session_access import rule

from cayu._resource_access_binding import ResourceExecutionBinding
from cayu._resource_store_surface import _TASK_OPERATIONS
from cayu.resource_access import (
    ResourceAccessPolicy,
    encode_scope,
    execution_access,
    model_data_access,
)
from cayu.sessions._access_rules import SessionAccessScope
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.queries import TaskQuery
from cayu.tasks.terminalization import TaskTerminalizationRequest, TaskTerminalKind

_OPERATOR_CALLS = {
    "claim_task": lambda store: store.claim_task("model-worker"),
    "heartbeat": lambda store: store.heartbeat("guarded", "model-worker", claim_generation=1),
    "release_task": lambda store: store.release_task("guarded", "model-worker", claim_generation=1),
    "mark_claimed_task_execution_started": lambda store: store.mark_claimed_task_execution_started(
        "guarded", "model-worker", claim_generation=1
    ),
    "request_claimed_task_cancellation": lambda store: store.request_claimed_task_cancellation(
        "guarded", "model-worker", claim_generation=1
    ),
    "complete_task": lambda store: store.complete_task(
        "guarded", {"done": True}, worker_id="model-worker", claim_generation=1
    ),
    "fail_task": lambda store: store.fail_task(
        "guarded", {"failed": True}, worker_id="model-worker", claim_generation=1
    ),
    "terminalize_task": lambda store: store.terminalize_task(
        TaskTerminalizationRequest(
            task_id="guarded",
            worker_id="model-worker",
            claim_generation=1,
            kind=TaskTerminalKind.FAILED,
            error={"reason": "model"},
            idempotency_key="model-terminalization",
        )
    ),
    "restart_task": lambda store: store.restart_task("guarded", "model-worker", claim_generation=1),
    "reclaim_expired": lambda store: store.reclaim_expired(),
}


def _store(backend, tmp_path, request):
    if backend == "memory":
        return InMemoryTaskStore()
    if backend == "sqlite":
        return SQLiteTaskStore(tmp_path / "tasks.sqlite")
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.tasks_postgres import PostgresTaskStore

    return PostgresTaskStore(
        request.getfixturevalue("postgres_dsn"),
        min_size=1,
        max_size=2,
        schema_mode=SchemaMode.CREATE,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_operator_task_methods_fail_closed_during_model_tool_execution(backend, tmp_path, request):
    store = _store(backend, tmp_path, request)
    for name in dir(type(store)):
        operation = getattr(type(store), name)
        if (
            not name.startswith("_")
            and name not in _TASK_OPERATIONS
            and iscoroutinefunction(operation)
        ):
            # Every public coroutine outside the model surface carries the guard.
            assert hasattr(operation, "__wrapped__"), name

    scope = SessionAccessScope(
        read=(rule(organization="acme"),), execute=(rule(organization="acme"),)
    )

    class Policy(ResourceAccessPolicy):
        authority = "task-store-surface-test"

        async def resolve(self, subject):
            return scope

    policy = Policy()
    binding = ResourceExecutionBinding(
        authority=policy.authority, subject="model", admitted_json=encode_scope(scope)
    )

    async def run() -> None:
        try:
            await store.create_task(TaskCreate(task_id="guarded", type="guarded"))
            async with (
                execution_access(binding, policy, {"organization": "acme"}),
                model_data_access(),
            ):
                for name, call in _OPERATOR_CALLS.items():
                    with pytest.raises(
                        NotImplementedError, match=f"{name} requires the trusted operator"
                    ):
                        await call(store)
                # The model surface still answers, bounded by the execution's access.
                assert await store.list_tasks(TaskQuery()) == []
            claimed = await store.claim_task("operator-worker")
            assert claimed is not None and claimed.id == "guarded"
        finally:
            if backend != "memory":
                await store.close()

    asyncio.run(run())
