"""Expired manual settlement can acknowledge its exact durable result again."""

import asyncio
from datetime import datetime, timedelta

import pytest
from tests.core._event_projection_support import private_events_for_public_events
from tests.core.test_user_input import (
    _CountingTool,
    _crashed_user_input_resume_events,
    _ScriptedProvider,
)

from cayu import AgentSpec, CayuApp, Event, EventType, Message, RunRequest
from cayu.approvals import ToolApprovalRecoveryOutcome, UserInputRecoveryRequest, UserInputResponse
from cayu.deadlines import ExecutionDeadline
from cayu.observability.hooks import RuntimeHook
from cayu.sessions import InterruptSessionRequest
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict
from cayu.sessions.memory import InMemorySessionStore
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus
from cayu.tools import UserInputTool


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.store_semantics
@pytest.mark.parametrize("attachment", ["none", "local", "failed"])
@pytest.mark.parametrize("stop_after", ["exhausted", "tool", "interrupted", "hook", "cancel-hook"])
def test_expired_manual_user_input_settlement_replays_exact_result(
    backend, tmp_path, monkeypatch, stop_after, attachment
):
    import cayu.deadlines as deadline_module

    expires_in_hook = stop_after.startswith("late-")
    stop_after = stop_after.removeprefix("late-")

    async def scenario():
        advance = timedelta()

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.now(tz) + advance

        monkeypatch.setattr(deadline_module, "datetime", Clock)
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "replay.db")
        )
        tool = _CountingTool()
        provider = _ScriptedProvider(
            [
                ("call_1", "count", {}),
                ("call_2", "ask_user", {"question": "q"}),
            ],
            final_text="must not run",
        )
        tasks = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
        )
        task = None if attachment == "none" else await tasks.create_task(TaskCreate(type="job"))
        hooks, interruption_hooks = [], []
        hook_entered, release_hook = asyncio.Event(), asyncio.Event()
        second_entered, release_second = asyncio.Event(), asyncio.Event()
        second_hooks = []
        if stop_after not in {"hook", "cancel-hook"}:
            release_second.set()
        if stop_after != "interrupted":
            release_hook.set()

        class Hook(RuntimeHook):
            async def after_tool_call(self, context):
                nonlocal advance
                if context.tool_event.payload.get("manual_recovery"):
                    if expires_in_hook:
                        assert not context.session.execution_deadline.expired
                        advance = timedelta(seconds=601)
                    hooks.append(context.tool_event.id)

            async def after_session_interrupted(self, context):
                if context.terminal_event.payload.get("deadline_settlement_only"):
                    hook_entered.set()
                    await release_hook.wait()
                    interruption_hooks.append(context.terminal_event.id)

        class SecondHook(RuntimeHook):
            async def after_tool_call(self, context):
                if context.tool_event.payload.get("manual_recovery"):
                    second_entered.set()
                    await release_second.wait()
                    second_hooks.append(context.tool_event.id)

        app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[UserInputTool(), tool],
            runtime_hooks=[Hook(), SecondHook()],
        )
        try:
            deadline = ExecutionDeadline.after(600)
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="expired-manual-retry",
                        messages=[Message.text("user", "go")],
                        execution_deadline=deadline,
                        task_id=None if task is None else task.id,
                    )
                )
            ]
            awaiting = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            input_id = awaiting.payload["input_id"]
            await store.append_events(
                "expired-manual-retry",
                _crashed_user_input_resume_events(
                    await private_events_for_public_events(store, paused),
                    session_id="expired-manual-retry",
                    tool_call_id="call_1",
                ),
            )
            stuck = [
                event
                async for event in app.resolve_user_input(
                    UserInputResponse(
                        session_id="expired-manual-retry",
                        input_id=input_id,
                        answer="a",
                    )
                )
            ]
            assert any(event.payload.get("manual_recovery_required") for event in stuck)
            if not expires_in_hook:
                advance = timedelta(seconds=601)
            if attachment == "failed":
                session = await store.load("expired-manual-retry")
                await store.append_events(
                    session.id,
                    [
                        Event(
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session.id,
                            agent_name="assistant",
                            payload={
                                "failure_evidence": {
                                    "classification": "deadline",
                                    "session_id": session.id,
                                    "run_epoch": session.run_epoch,
                                },
                                "execution_deadline": deadline.inspection(),
                            },
                        )
                    ],
                )
                assert await app._session_tasks.fail_task_for_execution_deadline(
                    task_id=task.id,
                    task_worker_id=None,
                    task_handoff_id=None,
                    session=session,
                    registered_agent=app._get_registered_agent("assistant"),
                    registered_environment=None,
                )
                assert (await tasks.load_task(task.id)).status is TaskStatus.FAILED
            request = UserInputRecoveryRequest(
                session_id="expired-manual-retry",
                input_id=input_id,
                answer="a",
                tool_call_id="call_1",
                outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                message="externally verified",
            )
            reached, release = asyncio.Event(), asyncio.Event()
            original_fail = tasks.fail_task

            async def blocked_fail(*args, **kwargs):
                reached.set()
                await release.wait()
                return await original_fail(*args, **kwargs)

            if stop_after == "interrupted" and attachment == "local":
                monkeypatch.setattr(tasks, "fail_task", blocked_fail)
            recovered = []
            stream = app.recover_user_input(request)

            async def consume():
                async for event in stream:
                    recovered.append(event)
                    if (
                        (stop_after == "tool" and event.type is EventType.TOOL_CALL_COMPLETED)
                        or (
                            stop_after == "interrupted"
                            and event.type is EventType.SESSION_INTERRUPTED
                        )
                        or (
                            stop_after == "hook"
                            and event.type is EventType.HOOK_COMPLETED
                            and event.payload.get("phase") == "after_tool_call"
                        )
                    ):
                        break

            if stop_after == "marker":
                original_publish = store.publish_session_operation
                marker_failed = False

                async def fail_marker_once(*args, **kwargs):
                    nonlocal marker_failed
                    key = kwargs.get("idempotency_key", "")
                    if not marker_failed and (
                        key.startswith("tool-effect-deadline-settlement:")
                        or (
                            key.startswith("manual-input-deadline:") and key.endswith(":settlement")
                        )
                    ):
                        marker_failed = True
                        raise OSError("injected final settlement marker failure")
                    return await original_publish(*args, **kwargs)

                monkeypatch.setattr(store, "publish_session_operation", fail_marker_once)

            reader = None
            if stop_after == "cancel-hook":
                reader = asyncio.create_task(consume())
                await asyncio.wait_for(second_entered.wait(), 5)
                reader.cancel()
                await asyncio.sleep(0)
                reader.cancel()
            elif stop_after == "marker":
                try:
                    await consume()
                except OSError as error:
                    assert "injected final settlement marker failure" in str(error)
                assert marker_failed
            else:
                await consume()
            terminal = next(
                event for event in recovered if event.type is EventType.TOOL_CALL_COMPLETED
            )
            private_terminal = (await private_events_for_public_events(store, [terminal]))[0]
            if stop_after in {"hook", "cancel-hook"}:
                closing = reader or asyncio.create_task(stream.aclose())
                try:
                    await asyncio.wait_for(second_entered.wait(), 5)
                    assert hooks == [private_terminal.id] and not second_hooks
                    assert not closing.done()
                    assert not await store.load_session_operation(
                        request.session_id,
                        f"manual-input-deadline:{private_terminal.id}:tool-hooks",
                    )
                finally:
                    release_second.set()
                    if reader is not None:
                        with pytest.raises(asyncio.CancelledError):
                            await asyncio.wait_for(reader, 5)
                        await stream.aclose()
                    else:
                        await asyncio.wait_for(closing, 5)
            elif stop_after == "interrupted":
                closing = asyncio.create_task(stream.aclose())
                try:
                    if attachment == "local":
                        await asyncio.wait_for(reached.wait(), 5)
                        assert not closing.done()
                        assert (await tasks.load_task(task.id)).status is TaskStatus.RUNNING
                        release.set()
                    await asyncio.wait_for(hook_entered.wait(), 5)
                    assert not closing.done()
                    assert not await store.load_session_operation(
                        request.session_id,
                        f"manual-input-deadline:{private_terminal.id}:settlement",
                    )
                finally:
                    release.set()
                    release_hook.set()
                    await asyncio.wait_for(closing, 5)
            else:
                await stream.aclose()
            if stop_after == "tool":
                assert not hooks
            if stop_after in {"exhausted", "interrupted"}:
                assert any(e.type is EventType.SESSION_INTERRUPTED for e in recovered)
            terminal = next(
                event for event in recovered if event.type is EventType.TOOL_CALL_COMPLETED
            )
            before = await store.load_events(request.session_id)
            replayed = [event async for event in app.recover_user_input(request)]
            assert [event.id for event in replayed] == [terminal.id]
            assert hooks == [private_terminal.id]
            assert len(interruption_hooks) == len(second_hooks) == 1
            if stop_after in {"exhausted", "interrupted"}:
                assert await store.load_events(request.session_id) == before
            if task is not None:
                failed = await tasks.load_task(task.id)
                assert failed.status is TaskStatus.FAILED
                assert failed.error["error"] == "ExecutionDeadlineExceeded"
                persisted = await store.load_events(request.session_id)
                assert sum(e.type is EventType.TASK_FAILED for e in persisted) == 1
            if stop_after == "tool":
                checkpoint = await store.load_checkpoint(request.session_id)
                assert checkpoint["user_input_resolution_intent"]["execution_state"] == "executing"
                with pytest.raises(RuntimeError, match="Executing user-input resolution"):
                    _ = [
                        event
                        async for event in app.interrupt_session(
                            InterruptSessionRequest(session_id=request.session_id)
                        )
                    ]
                checkpoint = await store.load_checkpoint(request.session_id)
                assert checkpoint["pending_user_input"]
                assert checkpoint["user_input_resolution_intent"]["execution_state"] == "executing"
            before = await store.load_events(request.session_id)
            before_session = await store.load(request.session_id)
            assert [e.id async for e in app.recover_user_input(request)] == [terminal.id]
            assert await store.load_events(request.session_id) == before
            assert await store.load(request.session_id) == before_session
            assert hooks == [private_terminal.id] and len(interruption_hooks) == 1
            for change in (
                {"answer": "different"},
                {"message": "different"},
                {"outcome": ToolApprovalRecoveryOutcome.FAILED},
            ):
                with pytest.raises(SessionRuntimePublicationConflict, match="different resolution"):
                    _ = [
                        event
                        async for event in app.recover_user_input(request.model_copy(update=change))
                    ]
            assert len(provider.requests) == 1 and tool.calls == 0
        finally:
            if backend == "sqlite":
                await store.close()
                await tasks.close()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("stop_after", ["late-hook", "late-cancel-hook"])
def test_manual_recovery_deadline_expires_inside_hook(backend, stop_after, tmp_path, monkeypatch):
    test_expired_manual_user_input_settlement_replays_exact_result(
        backend, tmp_path, monkeypatch, stop_after, "none"
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("attachment", ["none", "local", "failed"])
def test_manual_recovery_retries_failed_final_marker(backend, attachment, tmp_path, monkeypatch):
    test_expired_manual_user_input_settlement_replays_exact_result(
        backend, tmp_path, monkeypatch, "marker", attachment
    )
