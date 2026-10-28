"""Event cancellation preserves durable continuation and settlement authority."""

import asyncio
from contextlib import suppress

import pytest
from tests.core.test_continuation_deadlines import _app, _run, _settle, _store
from tests.core.test_continuation_deadlines import skip_ahead as skip_ahead
from tests.core.test_deadline_tool_settlement import (
    test_deadline_failure_retains_exact_settlement_authority as check_settlement,
)

from cayu import Message
from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded, task_event_deadline_scope
from cayu.sessions import ResumeRequest
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus, run_task_worker
from cayu.tasks.worker import complete_managed_task


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("mode", ["scope", "worker"])
def test_event_scope_durably_bounds_unbounded_resume(backend, mode, tmp_path, request, skip_ahead):
    from cayu.deadlines import task_event_deadline_from_metadata

    async def scenario():
        store = _store(backend, tmp_path, request)
        tasks = InMemoryTaskStore()
        app = _app(store, task_store=tasks)
        try:
            await _run(app, "unbounded", "plain")
            before = await store.load("unbounded")

            async def resume():
                return [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="unbounded", messages=[Message.text("user", "again")]
                        )
                    )
                ]

            if mode == "worker":
                task = await tasks.create_task(TaskCreate(type="reply"))
                budget = await tasks.load_task_retry_key(task.invocation.event_retry_key)
                event_deadline = ExecutionDeadline(
                    expires_at=budget.elapsed_deadline, source="task_event"
                )

                async def handle(runtime, claimed, worker_id):
                    await resume()
                    await complete_managed_task(tasks, claimed, worker_id, {"replied": True})

                assert (
                    await run_task_worker(app, tasks, handle, worker_id="worker", max_tasks=1) == 1
                )
                assert (await tasks.load_task(task.id)).status is TaskStatus.COMPLETED
            else:
                event_deadline = ExecutionDeadline.after(60, source="task_event")
                async with task_event_deadline_scope(event_deadline):
                    await resume()
            after = await store.load("unbounded")
            assert after.run_epoch > before.run_epoch
            assert after.execution_deadline.expires_at == event_deadline.expires_at
            assert (
                task_event_deadline_from_metadata(after.metadata).expires_at
                == event_deadline.expires_at
            )
            # No ambient event survives here. A later application still enforces
            # the bound stored by this continuation until a new event is admitted.
            skip_ahead(1801)
            restored = _app(store)
            try:
                with pytest.raises(ExecutionDeadlineExceeded):
                    [
                        event
                        async for event in restored.resume(
                            ResumeRequest(
                                session_id="unbounded",
                                messages=[Message.text("user", "late")],
                            )
                        )
                    ]
            finally:
                await restored.aclose()
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("gate", ["approval", "user_input"])
def test_event_scope_is_retained_when_settling_an_existing_human_wait(
    backend, gate, tmp_path, request
):
    from cayu.deadlines import task_event_deadline_from_metadata

    async def scenario():
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            await _run(app, "waiting", gate)
            assert (await store.load("waiting")).execution_deadline.expires_at is None
            bound = ExecutionDeadline.after(60, source="task_event")
            async with task_event_deadline_scope(bound):
                await _settle(app, "waiting", gate)
            stored = await store.load("waiting")
            assert task_event_deadline_from_metadata(stored.metadata).expires_at == bound.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_stale_resume_does_not_install_an_event_deadline(backend, tmp_path, request):
    async def scenario():
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            await _run(app, "unbounded", "plain")
            before = await store.load("unbounded")
            async with task_event_deadline_scope(ExecutionDeadline.after(60)):
                with pytest.raises(ValueError):
                    [
                        event
                        async for event in app.resume(
                            ResumeRequest(
                                session_id="unbounded",
                                messages=[Message.text("user", "stale")],
                                expected_run_epoch=before.run_epoch - 1,
                            )
                        )
                    ]
            after = await store.load("unbounded")
            assert after.run_epoch == before.run_epoch
            assert after.execution_deadline.expires_at is None
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_event_timer_retains_tool_effect_settlement_authority(backend, tmp_path, monkeypatch):
    check_settlement(
        backend,
        "worker",
        "ordinary",
        False,
        tmp_path,
        "run",
        monkeypatch,
        event_owned=True,
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("next_event", ["bounded", "unlimited", "explicit"])
@pytest.mark.parametrize("created_in_event", [False, True])
def test_completed_session_can_continue_under_a_new_event(
    backend, next_event, created_in_event, tmp_path, request, skip_ahead
):
    async def scenario():
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            first = ExecutionDeadline.after(60, source="task_event")
            if created_in_event:
                async with task_event_deadline_scope(first, event_id="first"):
                    await _run(app, "chat", "plain")
            else:
                await _run(app, "chat", "plain")

            async def resume(**kwargs):
                return [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="chat", messages=[Message.text("user", "next")], **kwargs
                        )
                    )
                ]

            async with task_event_deadline_scope(first, event_id="first"):
                await resume()
            skip_ahead(120)
            # The same durable event cannot reset its elapsed budget.
            async with task_event_deadline_scope(ExecutionDeadline.after(600), event_id="first"):
                with pytest.raises(ExecutionDeadlineExceeded):
                    await resume()
            if next_event == "explicit":
                await resume(execution_deadline=ExecutionDeadline.after(600))
            else:
                new = ExecutionDeadline.after(600 if next_event == "bounded" else None)
                async with task_event_deadline_scope(new, event_id="second"):
                    await resume()
                assert (await store.load("chat")).execution_deadline.expires_at == new.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("gate", ["approval", "user_input"])
def test_new_event_cannot_reset_an_unfinished_human_wait(
    backend, gate, tmp_path, request, skip_ahead
):
    async def scenario():
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            async with task_event_deadline_scope(ExecutionDeadline.after(60), event_id="first"):
                await _run(app, "waiting", gate)
            skip_ahead(120)
            async with task_event_deadline_scope(ExecutionDeadline.after(600), event_id="second"):
                with pytest.raises(ExecutionDeadlineExceeded):
                    await _settle(app, "waiting", gate)
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("created_in_event", [False, True])
def test_rebinding_an_event_preserves_the_sessions_own_deadline(
    backend, created_in_event, tmp_path, request, skip_ahead
):
    async def scenario():
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            original = ExecutionDeadline.after(300)
            if created_in_event:
                async with task_event_deadline_scope(ExecutionDeadline.after(60), event_id="first"):
                    await _run(app, "chat", "plain", execution_deadline=original)
            else:
                await _run(app, "chat", "plain", execution_deadline=original)

            async def resume():
                return [
                    event
                    async for event in app.resume(
                        ResumeRequest(session_id="chat", messages=[Message.text("user", "next")])
                    )
                ]

            async with task_event_deadline_scope(ExecutionDeadline.after(60), event_id="first"):
                await resume()
            skip_ahead(120)
            async with task_event_deadline_scope(ExecutionDeadline.after(600), event_id="second"):
                await resume()
            assert (await store.load("chat")).execution_deadline.expires_at == original.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_explicit_continuation_bound_survives_the_next_event(
    backend, tmp_path, request, skip_ahead
):
    async def scenario():
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            await _run(app, "chat", "plain")
            original = ExecutionDeadline.after(300)

            async def resume(**kwargs):
                return [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="chat", messages=[Message.text("user", "next")], **kwargs
                        )
                    )
                ]

            async with task_event_deadline_scope(ExecutionDeadline.after(60), event_id="first"):
                await resume(execution_deadline=original)
            skip_ahead(120)
            async with task_event_deadline_scope(ExecutionDeadline.after(600), event_id="second"):
                await resume()
            assert (await store.load("chat")).execution_deadline.expires_at == original.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("terminal", ["failed", "interrupted"])
@pytest.mark.parametrize("continuation", ["event", "explicit"])
def test_settled_unsuccessful_turn_can_accept_new_work(
    backend, terminal, continuation, tmp_path, request, skip_ahead, monkeypatch
):
    from tests.core import test_continuation_deadlines as helpers

    from cayu.evals.testing import ScriptedModelProvider

    original = helpers._respond
    original_stream = ScriptedModelProvider.stream
    failing = True
    entered = asyncio.Event()

    async def stream(self, model_request):
        if failing and terminal == "interrupted":
            entered.set()
            await asyncio.Event().wait()
        async for event in original_stream(self, model_request):
            yield event

    monkeypatch.setattr(ScriptedModelProvider, "stream", stream)

    def respond(model_request):
        if failing:
            if terminal == "interrupted":
                raise asyncio.CancelledError()
            raise RuntimeError("synthetic model failure")
        return original(model_request)

    monkeypatch.setattr(helpers, "_respond", respond)

    async def scenario():
        nonlocal failing
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            async with task_event_deadline_scope(ExecutionDeadline.after(60), event_id="first"):
                if terminal == "interrupted":
                    running = asyncio.create_task(_run(app, "chat", "plain"))
                    await asyncio.wait_for(entered.wait(), 5)
                    running.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await running
                else:
                    with suppress(RuntimeError):
                        await _run(app, "chat", "plain")
            assert (await store.load("chat")).status.value == terminal
            failing = False
            skip_ahead(120)

            async def resume(**kwargs):
                return [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="chat", messages=[Message.text("user", "new work")], **kwargs
                        )
                    )
                ]

            if continuation == "explicit":
                await resume(execution_deadline=ExecutionDeadline.after(600))
            else:
                async with task_event_deadline_scope(
                    ExecutionDeadline.after(600), event_id="second"
                ):
                    await resume()
            assert (await store.load("chat")).status.value == "completed"
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("previous_event", [False, True])
def test_explicit_unbounded_continuation_still_retains_event_bound(
    backend, previous_event, tmp_path, request
):
    from contextlib import nullcontext

    from cayu.deadlines import task_event_deadline_from_metadata

    async def scenario():
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            async with (
                task_event_deadline_scope(ExecutionDeadline.after(60), event_id="first")
                if previous_event
                else nullcontext()
            ):
                await _run(app, "chat", "plain")
            bound = ExecutionDeadline.after(600)
            async with task_event_deadline_scope(bound, event_id="second"):
                [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="chat",
                            messages=[Message.text("user", "next")],
                            execution_deadline=ExecutionDeadline(),
                        )
                    )
                ]
            stored = await store.load("chat")
            assert stored.execution_deadline.expires_at == bound.expires_at
            assert task_event_deadline_from_metadata(stored.metadata).expires_at == bound.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_shorter_handler_default_survives_a_new_event(backend, tmp_path, request):
    from cayu.deadlines import default_run_deadline

    async def scenario():
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            original = ExecutionDeadline.after(120, source="task_worker")
            async with task_event_deadline_scope(ExecutionDeadline.after(1800), event_id="first"):
                with default_run_deadline(original):
                    await _run(app, "chat", "plain")
            async with task_event_deadline_scope(ExecutionDeadline(), event_id="second"):
                [
                    event
                    async for event in app.resume(
                        ResumeRequest(session_id="chat", messages=[Message.text("user", "next")])
                    )
                ]
            assert (await store.load("chat")).execution_deadline.expires_at == original.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("explicit_seconds", [600, None])
def test_explicit_run_deadline_replaces_handler_default_across_events(
    backend, explicit_seconds, tmp_path, request, skip_ahead
):
    from cayu.deadlines import default_run_deadline

    async def scenario():
        store = _store(backend, tmp_path, request)
        app = _app(store)
        try:
            requested = ExecutionDeadline.after(explicit_seconds)
            async with task_event_deadline_scope(ExecutionDeadline.after(1800), event_id="first"):
                with default_run_deadline(ExecutionDeadline.after(30, source="task_worker")):
                    await _run(app, "chat", "plain", execution_deadline=requested)
            skip_ahead(60)
            async with task_event_deadline_scope(ExecutionDeadline(), event_id="second"):
                [
                    event
                    async for event in app.resume(
                        ResumeRequest(session_id="chat", messages=[Message.text("user", "next")])
                    )
                ]
            assert (await store.load("chat")).execution_deadline.expires_at == requested.expires_at
        finally:
            await app.aclose()

    asyncio.run(scenario())
