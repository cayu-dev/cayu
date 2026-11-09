"""A dead owner's group execution is settled when recovery completes its member.

A worker acknowledges a member's execution after its callback returns, and a
live terminalization never does it on the worker's behalf. When attached-task
recovery completes the task of an expired, generation-fenced owner, the
session's durable completion proves the work finished, so the recovery settles
that exact claim's execution in the same transaction. A recovered failure
proves nothing about the callback and leaves the execution owed.
"""

from __future__ import annotations

import asyncio

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_task_group_quiescence import store as store
from tests.core.test_task_groups import anyio_backend as anyio_backend

from cayu import CayuApp
from cayu.tasks import (
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupPolicy,
    TaskGroupQuiescencePolicy,
    TaskTerminalizationRequest,
    TaskTerminalKind,
)
from cayu.tasks.groups import TaskGroupExecutionObligation
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskClaimLost, TaskStatus

pytestmark = pytest.mark.anyio

_LEASE_SECONDS = 1
_PAST_LEASE_S = 1.05


async def _started_member(store, *, lease_seconds: int = 60) -> tuple[CayuApp, Task]:
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(
        TaskGroupCreate(
            group_id="members",
            graph=TaskGraphCreate(
                graph_id="members-graph",
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
    claimed = await store.claim_task("owner", TaskQuery(type="member"), lease_seconds=lease_seconds)
    assert claimed is not None
    started = await store.mark_claimed_task_execution_started(
        claimed.id, "owner", claim_generation=claimed.claim_generation
    )
    assert started.started_at is not None
    return app, started


async def _execution(app: CayuApp) -> TaskGroupExecutionObligation:
    snapshot = await app.load_task_group("members")
    assert snapshot is not None
    (execution,) = snapshot.quiescence.executions
    return execution


def _terminalization(task: Task, kind: TaskTerminalKind, key: str) -> TaskTerminalizationRequest:
    assert task.worker_id is not None
    return TaskTerminalizationRequest(
        task_id=task.id,
        worker_id=task.worker_id,
        claim_generation=task.claim_generation,
        handoff_id=task.interrupted_handoff_id,
        kind=kind,
        result={"done": True} if kind is TaskTerminalKind.COMPLETED else None,
        error=None if kind is TaskTerminalKind.COMPLETED else {"code": "failed"},
        idempotency_key=key,
    )


@pytest.mark.parametrize("kind", [TaskTerminalKind.COMPLETED, TaskTerminalKind.FAILED])
async def test_a_live_terminalization_waits_for_its_callback_to_return(store, kind):
    app, started = await _started_member(store)

    await store.terminalize_task(_terminalization(started, kind, "live"))

    # Only the worker's own acknowledgement after its callback returns settles it.
    assert (await _execution(app)).settled_at is None
    await store._settle_task_group_execution(started)
    assert (await _execution(app)).settled_at is not None


async def test_completing_under_the_worker_lease_waits_for_its_callback_to_return(store):
    app, started = await _started_member(store)

    await store.complete_task(
        "member", {"done": True}, worker_id="owner", claim_generation=started.claim_generation
    )

    assert (await _execution(app)).settled_at is None


async def _expired_attached_member(store) -> tuple[CayuApp, Task, Task]:
    app, started = await _started_member(store, lease_seconds=_LEASE_SECONDS)
    attached = await store.attach_task(
        started.id,
        session_id="member-session",
        session_invocation=await task_backed_session_invocation(
            store, started.id, "member-session"
        ),
        worker_id="owner",
        claim_generation=started.claim_generation,
    )
    assert attached.session_id is not None and attached.session_instance_id is not None
    await asyncio.sleep(_PAST_LEASE_S)
    return app, started, attached


async def test_recovery_under_another_claim_generation_settles_nothing(store):
    app, _started, attached = await _expired_attached_member(store)
    assert attached.session_id is not None and attached.session_instance_id is not None
    stale = _terminalization(attached, TaskTerminalKind.COMPLETED, "stale").model_copy(
        update={"claim_generation": attached.claim_generation + 1}
    )

    with pytest.raises(TaskClaimLost):
        await store.recover_attached_task_completion(
            stale,
            session_id=attached.session_id,
            session_instance_id=attached.session_instance_id,
        )

    assert (await store.load_task(attached.id)).status is TaskStatus.RUNNING
    assert (await _execution(app)).settled_at is None


async def test_recovering_a_dead_owners_completed_member_settles_its_execution(store):
    app, started, attached = await _expired_attached_member(store)
    assert attached.session_id is not None and attached.session_instance_id is not None

    recovered = await store.recover_attached_task_completion(
        _terminalization(attached, TaskTerminalKind.COMPLETED, "recover"),
        session_id=attached.session_id,
        session_instance_id=attached.session_instance_id,
    )

    assert recovered.status is TaskStatus.COMPLETED
    execution = await _execution(app)
    assert (execution.worker_id, execution.started_at) == ("owner", started.started_at)
    assert execution.settled_at is not None


async def test_recovering_a_dead_owners_failed_member_leaves_its_execution_owed(store):
    app, _started, attached = await _expired_attached_member(store)
    assert attached.session_id is not None and attached.session_instance_id is not None

    recovered = await store.recover_attached_task_failure(
        _terminalization(attached, TaskTerminalKind.FAILED, "recover"),
        session_id=attached.session_id,
        session_instance_id=attached.session_instance_id,
    )

    # A recovered failure is no proof that the owner's callback stopped.
    assert recovered.status is TaskStatus.FAILED
    assert (await _execution(app)).settled_at is None
