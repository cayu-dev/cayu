"""Completed expired receipt recovery is a read-only replay, including without a task."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.core.test_tool_effect_task_reconciliation import (
    _TaskSessionStore,
    _TaskSQLiteSessionStore,
)
from tests.core.test_tool_round_execution_identities import _SequencedProvider, _tool_call_response

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.approvals.review import (
    HumanReviewContext,
    HumanReviewDenied,
    HumanReviewDisclosure,
    HumanReviewField,
    HumanReviewPolicy,
)
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.deadlines import ExecutionDeadline
from cayu.observability.hooks import RuntimeHook
from cayu.providers import ModelStreamEvent
from cayu.runtime._deadline_effect_settlement import deadline_effect_settlement_completed
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime.tool_effects import (
    ToolEffectConflict,
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.storage import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus
from cayu.tools import Tool, ToolEffect, ToolSpec, UserInputTool
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("gate", ["ordinary", "approval", "input"])
@pytest.mark.parametrize("attachment", ["none", "local", "worker"])
@pytest.mark.parametrize("stop_after", ["exhausted", "tool", "interrupted", "hook", "cancel-hook"])
def test_expired_effect_settlement_replays_without_repeating_hooks(
    backend, attachment, stop_after, gate, tmp_path, monkeypatch
):
    import cayu.deadlines as deadline_module

    revoke_review = stop_after == "review-revoked"
    if revoke_review:
        stop_after = "exhausted"
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
            _TaskSessionStore()
            if backend == "memory"
            else _TaskSQLiteSessionStore(tmp_path / "sessions.db")
        )
        tasks = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
        )
        calls, lookups, hooks, interruption_hooks = [], [], [], []
        hook_entered, release_hook = asyncio.Event(), asyncio.Event()
        second_entered, release_second = asyncio.Event(), asyncio.Event()
        second_hooks = []
        if stop_after not in {"hook", "cancel-hook"}:
            release_second.set()
        if stop_after != "interrupted":
            release_hook.set()

        class Operation(Tool):
            spec = ToolSpec(name="record", effect=ToolEffect.EXTERNAL)

            async def run(self, context, arguments):
                calls.append(context.idempotency_key)
                raise ConnectionError("lost acknowledgement")

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                lookups.append(context.idempotency_key)
                return ToolEffectReconciliationResult(
                    outcome="completed",
                    observation="sent",
                    receipt=ToolEffectReceipt(
                        receipt_id="verified",
                        receipt_schema="deployment",
                        receipt_schema_version=1,
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        idempotency_key=context.idempotency_key,
                        outcome="completed",
                        message="verified",
                        source="reconciler",
                        observed_at=datetime.now(UTC),
                    ),
                )

        class Hook(RuntimeHook):
            async def after_tool_call(self, context):
                nonlocal advance
                if (
                    context.tool_event.type is EventType.TOOL_CALL_COMPLETED
                    and context.tool_event.tool_name == "record"
                ):
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
                if (
                    context.tool_event.type is EventType.TOOL_CALL_COMPLETED
                    and context.tool_event.tool_name == "record"
                ):
                    second_entered.set()
                    await release_second.wait()
                    second_hooks.append(context.tool_event.id)

        response = _tool_call_response(1)
        if gate == "input":
            response.insert(
                0,
                ModelStreamEvent.tool_call(
                    id="ask", name="ask_user", arguments={"question": "Proceed?"}
                ),
            )
        provider = _SequencedProvider([response])
        review_context = HumanReviewContext(recipient="operator", purpose="effect-recovery")

        class ReviewPolicy(HumanReviewPolicy):
            version = "deadline-replay-test-v1"
            binding_key = b"deadline-review-test-binding-key-32-bytes"
            can_decide = True

            def authorize(self, context, *, session_id, session_metadata, action):
                return context == review_context and (action == "inspect" or self.can_decide)

            def project(self, context, source):
                return HumanReviewDisclosure(
                    status="permitted",
                    fields=(HumanReviewField(label="Effect", text="Record the reviewed effect."),),
                    sensitive_content="application_attested",
                )

        review_policy = ReviewPolicy() if revoke_review else None
        app = CayuApp(
            session_store=store,
            task_store=tasks,
            enable_logging=False,
            human_review_policy=review_policy,
        )

        async def review_reference():
            if review_policy is None:
                return None
            return (
                await app.inspect_human_review("expired-effect", context=review_context)
            ).reference

        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="agent", model="test"),
            tools=[Operation(), UserInputTool()] if gate == "input" else [Operation()],
            tool_policy=AlwaysRequireApprovalToolPolicy() if gate == "approval" else None,
            runtime_hooks=[Hook(), SecondHook()],
            tool_effect_reconcilers={
                "record": ToolEffectReconciliationRegistration(
                    reconciler=Reconciler(),
                    spec=_spec(supports_lookup=True),
                )
            },
        )
        task = None if attachment == "none" else await tasks.create_task(TaskCreate(type="job"))
        worker = "worker" if attachment == "worker" else None
        if worker is not None:
            task = await tasks.claim_task(worker, lease_seconds=600)
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="expired-effect",
                        messages=[Message.text("user", "act")],
                        execution_deadline=ExecutionDeadline.after(600),
                        task_id=None if task is None else task.id,
                        task_worker_id=worker,
                        task_lease_expires_at=None if task is None else task.lease_expires_at,
                    )
                )
            ]
            input_response = None
            if gate == "approval":
                pending = next(
                    event
                    for event in events
                    if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                )
                _ = [
                    event
                    async for event in app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id="expired-effect",
                            task_worker_id=worker,
                            approval_id=pending.payload["approval_id"],
                            tool_round_id=pending.payload["tool_round_id"],
                            tool_call_id=pending.payload["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                            review_reference=await review_reference(),
                        )
                    )
                ]
            elif gate == "input":
                pending = next(
                    event for event in events if event.type is EventType.SESSION_AWAITING_USER_INPUT
                )
                input_response = UserInputResponse(
                    session_id="expired-effect",
                    task_worker_id=worker,
                    input_id=pending.payload["input_id"],
                    answer="yes",
                    review_reference=await review_reference(),
                )
                _ = [event async for event in app.resolve_user_input(input_response)]
            session = await store.load("expired-effect")
            record = ToolEffectRecord.model_validate(
                await store.load_session_operation(session.id, store.effect_keys[0])
            )
            action = ToolEffectReconciliationRequest(
                **{
                    name: getattr(record.intent, name)
                    for name in (
                        "session_id",
                        "session_instance_id",
                        "tool_round_id",
                        "tool_call_id",
                        "tool_name",
                        "idempotency_key",
                    )
                },
                expected_run_epoch=session.run_epoch,
                expected_revision=record.revision,
                lookup=True,
                task_worker_id=worker,
                user_input_response=input_response,
                review_reference=await review_reference(),
            )

            async def settlement_completed():
                current = await store.load(session.id)
                selected = ToolEffectRecord.model_validate(
                    await store.load_session_operation(session.id, store.effect_keys[0])
                )
                return await deadline_effect_settlement_completed(store, current, selected)

            if not expires_in_hook:
                advance = timedelta(seconds=601)
            reached, release = asyncio.Event(), asyncio.Event()
            settle_method = "terminalize_task" if worker is not None else "fail_task"
            fail_task = getattr(tasks, settle_method)

            async def blocked_fail(*args, **kwargs):
                reached.set()
                await release.wait()
                return await fail_task(*args, **kwargs)

            if stop_after == "interrupted" and task is not None:
                monkeypatch.setattr(tasks, settle_method, blocked_fail)
            events = []
            stream = app.reconcile_tool_effect(action)

            async def consume():
                async for event in stream:
                    events.append(event)
                    if (
                        (
                            stop_after == "tool"
                            and event.type is EventType.TOOL_CALL_COMPLETED
                            and event.tool_name == "record"
                        )
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
            if stop_after in {"hook", "cancel-hook"}:
                closing = reader or asyncio.create_task(stream.aclose())
                try:
                    await asyncio.wait_for(second_entered.wait(), 5)
                    assert len(hooks) == 1 and not second_hooks
                    assert not closing.done() and not await settlement_completed()
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
                    if task is not None:
                        await asyncio.wait_for(reached.wait(), 5)
                        assert not await settlement_completed()
                        assert not closing.done()
                        assert (await tasks.load_task(task.id)).status is TaskStatus.RUNNING
                        release.set()
                    await asyncio.wait_for(hook_entered.wait(), 5)
                    assert not await settlement_completed()
                    assert not closing.done()
                    assert not interruption_hooks
                finally:
                    release.set()
                    release_hook.set()
                    await asyncio.wait_for(closing, 5)
            else:
                await stream.aclose()
            terminal = next(
                event
                for event in events
                if event.type is EventType.TOOL_CALL_COMPLETED and event.tool_name == "record"
            )
            if stop_after == "tool":
                assert hooks == []
            if stop_after in {"tool", "hook", "cancel-hook", "marker"}:
                retry = [event async for event in app.reconcile_tool_effect(action)]
                assert any(event.type is EventType.SESSION_INTERRUPTED for event in retry)
            assert len(hooks) == len(interruption_hooks) == len(second_hooks) == 1
            before_events = await store.load_events(session.id)
            before_session = await store.load(session.id)
            if revoke_review:
                assert await settlement_completed()
                review_policy.can_decide = False
                with pytest.raises(HumanReviewDenied):
                    _ = [event async for event in app.reconcile_tool_effect(action)]
                assert await store.load_events(session.id) == before_events
                assert await store.load(session.id) == before_session
                assert len(hooks) == len(second_hooks) == len(interruption_hooks) == 1
                assert len(calls) == len(lookups) == 1
                return
            for _ in range(2):
                retry = [event async for event in app.reconcile_tool_effect(action)]
                assert [event.id for event in retry] == [terminal.id]
            assert await store.load_events(session.id) == before_events
            assert await store.load(session.id) == before_session
            assert (
                len(hooks)
                == len(interruption_hooks)
                == len(calls)
                == len(lookups)
                == len(provider.requests)
                == 1
            )
            if task is not None:
                assert (await tasks.load_task(task.id)).status is TaskStatus.FAILED
            with pytest.raises(ToolEffectConflict):
                _ = [
                    event
                    async for event in app.reconcile_tool_effect(
                        action.model_copy(
                            update={"expected_revision": action.expected_revision + 1}
                        )
                    )
                ]
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("gate", ["ordinary", "approval", "input"])
@pytest.mark.parametrize("stop_after", ["late-hook", "late-cancel-hook"])
def test_effect_recovery_deadline_expires_inside_hook(
    backend, gate, stop_after, tmp_path, monkeypatch
):
    test_expired_effect_settlement_replays_without_repeating_hooks(
        backend, "none", stop_after, gate, tmp_path, monkeypatch
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("gate", ["ordinary", "approval", "input"])
@pytest.mark.parametrize("attachment", ["none", "local", "worker"])
def test_effect_recovery_retries_failed_final_marker(
    backend, gate, attachment, tmp_path, monkeypatch
):
    test_expired_effect_settlement_replays_without_repeating_hooks(
        backend, attachment, "marker", gate, tmp_path, monkeypatch
    )


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("gate", ["approval", "input"])
def test_deadline_receipt_replay_requires_current_review_authority(
    backend, gate, tmp_path, monkeypatch
):
    test_expired_effect_settlement_replays_without_repeating_hooks(
        backend, "none", "review-revoked", gate, tmp_path, monkeypatch
    )
