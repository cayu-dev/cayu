"""Permanent pre-admission rejection is positive nondispatch evidence for a group."""

from __future__ import annotations

import asyncio

from tests.core.test_dispatch import (
    FakeProvider,
    ProfileTool,
    _batch,
    _configured_app,
    _dispatch_request,
)
from tests.core.test_task_execution_settlement_owner import create_group

from cayu.messages import Message
from cayu.sessions.invocation import TaskExecutionSource
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest
from cayu.tasks.creation import TaskCreate, task_create_with_runtime_invocation
from cayu.tasks.dispatch import (
    DispatchStatus,
    TaskStoreDispatcher,
    _queued_dispatch_persisted_envelope,
    _queued_dispatch_task_id,
    _QueuedDispatchAuthorityRejected,
)
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import TaskStatus


async def _scenario(*, authority_gone: bool = False) -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    dispatcher = TaskStoreDispatcher(tasks)
    producer = _configured_app(
        session_store=sessions,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=FakeProvider([_batch("initial")]),
        tools=[ProfileTool("original_tool")],
    )
    worker_provider = FakeProvider([])
    worker = _configured_app(
        session_store=sessions,
        task_store=tasks,
        dispatcher=dispatcher,
        provider=worker_provider,
        tools=[ProfileTool("replacement_tool")],
    )
    async for _ in producer.run(
        RunRequest(
            agent_name="assistant",
            session_id="group-rejection",
            messages=[Message.text("user", "initial")],
        )
    ):
        pass
    request = _dispatch_request("group-rejection", "rejected-member")
    task_id = _queued_dispatch_task_id(request)
    envelope = await producer._prepare_queued_dispatch(request, queue_task_id=task_id)
    member = task_create_with_runtime_invocation(
        TaskCreate(
            task_id=task_id,
            type=dispatcher.task_type,
            input={"dispatch": _queued_dispatch_persisted_envelope(envelope)},
        ),
        source=TaskExecutionSource.TASK_DISPATCH,
        session_invocation=await producer.session_invocation_for_dispatch(request.session_id),
    )
    await create_group(producer, member)
    if authority_gone:
        # After admission is refused, the dispatch's authority can no longer be
        # read back. No admission can exist under it, which is the proof.
        read_settlement_state = worker._queued_dispatch_settlement_state
        reads = 0

        async def authority_rejected(envelope):
            nonlocal reads
            reads += 1
            if reads == 1:
                return await read_settlement_state(envelope)
            raise _QueuedDispatchAuthorityRejected("Queued dispatch target no longer exists.")

        worker._queued_dispatch_settlement_state = authority_rejected

    result = await dispatcher.process_next(worker, worker_id="changed-worker")
    assert result is not None and result.status is DispatchStatus.FAILED
    assert worker_provider.requests == []
    rejected = await tasks.load_task(task_id)
    assert rejected is not None and rejected.status is TaskStatus.FAILED
    assert rejected.error["error_type"] == "ExecutionProfileMismatchError"
    await tasks.complete_task("winner", {})
    await producer.reconcile_task_group("owned")
    finalizer = await tasks.load_task("finalizer")
    assert finalizer is not None
    assert finalizer.status is TaskStatus.PENDING


def test_profile_rejection_releases_group_execution_and_finalizer() -> None:
    asyncio.run(_scenario())


def test_rejection_whose_authority_is_gone_still_settles_its_execution() -> None:
    asyncio.run(_scenario(authority_gone=True))
