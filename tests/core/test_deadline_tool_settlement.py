"""An expired task may settle recorded work but cannot execute another step."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_run_limit_finish import spec as structured_spec
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.core.test_tool_effect_task_reconciliation import (
    _TaskSessionStore,
    _TaskSQLiteSessionStore,
)
from tests.core.test_tool_round_execution_identities import _SequencedProvider, _tool_call_response

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.approvals.tools import (
    ToolApprovalDecision,
    ToolApprovalRecoveryOutcome,
    ToolApprovalRequest,
)
from cayu.approvals.user_input import UserInputResponse
from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded, execution_deadline_scope
from cayu.providers import ModelStreamEvent
from cayu.runtime._tool_effect_state import ToolEffectReconciliationRequired, ToolEffectRecord
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.sessions import ResumeRequest, SessionStatus
from cayu.storage import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskClaimLost, TaskCreate, TaskStatus
from cayu.tools import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.tools.user_input import UserInputTool


@pytest.mark.store_semantics
@pytest.mark.parametrize("expiry", ["run", "before_recovery", "during_recovery", "continuation"])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("worker", [None, "worker"])
@pytest.mark.parametrize(
    "gate,manual_attempt",
    [
        ("ordinary", False),
        ("ordinary", True),
        ("approval", False),
        ("input", False),
    ],
)
def test_deadline_failure_retains_exact_settlement_authority(
    backend, worker, gate, manual_attempt, tmp_path, expiry, monkeypatch, structured=False
):
    import cayu.deadlines as deadline_module

    async def scenario():
        advance = timedelta()

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.now(tz) + advance

        monkeypatch.setattr(deadline_module, "datetime", Clock)

        async def expire_active_timer():
            nonlocal advance
            # Trigger the real owned timer at the phase under test. Setup and
            # store latency must not expire a different phase first.
            advance = timedelta(seconds=601)
            timers = [
                timer
                for timer, selected in deadline_module._TIMERS.get()
                if selected.expires_at == boundary.expires_at and timer.when() is not None
            ]
            assert timers
            timers[-1].reschedule(asyncio.get_running_loop().time())
            await asyncio.Event().wait()

        store = (
            _TaskSessionStore()
            if backend == "memory"
            else _TaskSQLiteSessionStore(tmp_path / "sessions.db")
        )
        tasks = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
        )
        calls, lookups = [], []

        class Operation(Tool):
            spec = ToolSpec(name="record", effect=ToolEffect.EXTERNAL)

            async def run(self, ctx, args):
                calls.append(ctx.idempotency_key)
                if expiry != "run":
                    raise ConnectionError("external acknowledgement lost")
                await expire_active_timer()
                return ToolResult(content="unreachable")

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                nonlocal advance
                lookups.append(context.idempotency_key)
                if expiry == "during_recovery":
                    advance = timedelta(seconds=601)
                return ToolEffectReconciliationResult(
                    outcome="completed",
                    observation="sent",
                    receipt=ToolEffectReceipt(
                        receipt_id="verified-outcome",
                        receipt_schema="deployment",
                        receipt_schema_version=1,
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        idempotency_key=context.idempotency_key,
                        outcome="completed",
                        message="verified result",
                        source="reconciler",
                        observed_at=datetime.now(UTC),
                    ),
                )

        response = _tool_call_response(1)
        if gate == "input":
            response.insert(
                0,
                ModelStreamEvent.tool_call(
                    id="ask",
                    name="ask_user",
                    arguments={"question": "Proceed?"},
                ),
            )

        class Provider(_SequencedProvider):
            async def stream(self, request):
                if expiry == "continuation" and self.requests:
                    self.requests.append(request)
                    await expire_active_timer()
                async for event in super().stream(request):
                    yield event

        provider = Provider([response])
        app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="agent", model="test"),
            tools=[Operation(), UserInputTool()] if gate == "input" else [Operation()],
            tool_policy=AlwaysRequireApprovalToolPolicy() if gate == "approval" else None,
            tool_effect_reconcilers={
                "record": ToolEffectReconciliationRegistration(
                    reconciler=Reconciler(),
                    spec=_spec(supports_lookup=True),
                )
            },
        )
        try:
            task = await tasks.create_task(TaskCreate(type="job"))
            if worker is not None:
                task = await tasks.claim_task(worker, lease_seconds=30)
            events = []
            input_response = None
            boundary = ExecutionDeadline.after(600)
            try:
                async for event in app.run(
                    RunRequest(
                        session_id="deadline-settlement",
                        structured_output=structured_spec("tool") if structured else None,
                        agent_name="agent",
                        task_id=task.id,
                        task_worker_id=worker,
                        task_lease_expires_at=task.lease_expires_at,
                        execution_deadline=boundary,
                        messages=[Message.text("user", "perform the operation")],
                    )
                ):
                    events.append(event)
                if gate != "ordinary":
                    if gate == "approval":
                        pending = next(
                            e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
                        )
                        continuation = app.resolve_tool_approval(
                            ToolApprovalRequest(
                                session_id="deadline-settlement",
                                task_worker_id=worker,
                                approval_id=pending.payload["approval_id"],
                                tool_round_id=pending.payload["tool_round_id"],
                                tool_call_id=pending.payload["tool_call_id"],
                                decision=ToolApprovalDecision.APPROVE,
                            )
                        )
                    else:
                        pending = next(
                            e for e in events if e.type is EventType.SESSION_AWAITING_USER_INPUT
                        )
                        input_response = UserInputResponse(
                            session_id="deadline-settlement",
                            task_worker_id=worker,
                            input_id=pending.payload["input_id"],
                            answer="yes",
                        )
                        continuation = app.resolve_user_input(input_response)
                    # The current Runtime permits the host to own this boundary
                    # while a paused round executes its continuation tools.
                    async with execution_deadline_scope(boundary):
                        async for event in continuation:
                            events.append(event)
            except ExecutionDeadlineExceeded:
                pass  # Existing run iterators report typed expiry after cleanup.
            assert len(calls) == len(provider.requests) == 1
            failed = await tasks.load_task(task.id)
            if expiry == "run":
                assert failed.status is TaskStatus.FAILED
                assert failed.error["error"] == "ExecutionDeadlineExceeded"
            else:
                assert failed.status is TaskStatus.RUNNING
                if expiry == "before_recovery":
                    advance = timedelta(seconds=601)
            session = await store.load("deadline-settlement")
            assert session.status is SessionStatus.INTERRUPTED
            started = next(
                event
                for event in events
                if event.type is EventType.TOOL_CALL_STARTED and event.tool_name == "record"
            )
            if manual_attempt:
                manual = ToolRoundRecoveryRequest(
                    session_id=session.id,
                    task_worker_id=worker,
                    round_id=started.payload["tool_round_id"],
                    tool_call_id=started.payload["tool_call_id"],
                    outcome=ToolApprovalRecoveryOutcome.COMPLETED,
                    message="unverified result",
                )
                with pytest.raises(ToolEffectReconciliationRequired):
                    _ = [event async for event in app.recover_tool_round(manual)]
                session = await store.load(session.id)
                assert not lookups and len(calls) == len(provider.requests) == 1
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
                task_worker_id=worker,
                expected_run_epoch=session.run_epoch,
                expected_revision=record.revision,
                lookup=True,
                user_input_response=input_response,
            )
            entrance = app.reconcile_tool_effect
            if worker is not None:
                with pytest.raises(TaskClaimLost):
                    _ = [
                        event
                        async for event in entrance(
                            action.model_copy(
                                update={
                                    "task_worker_id": "other-worker",
                                    "user_input_response": None
                                    if input_response is None
                                    else input_response.model_copy(
                                        update={"task_worker_id": "other-worker"}
                                    ),
                                }
                            )
                        )
                    ]
                assert not lookups
            settled = [event async for event in entrance(action)]
            assert any(event.type is EventType.TOOL_CALL_COMPLETED for event in settled)
            assert settled[-1].type is EventType.SESSION_INTERRUPTED
            if expiry == "continuation":
                assert settled[-1].payload["limit"] == "execution_deadline"
            else:
                assert settled[-1].payload["deadline_settlement_only"] is True
            assert settled[-1].payload["structured_output_pending"] is structured
            current_task = await tasks.load_task(task.id)
            assert current_task.status is TaskStatus.FAILED
            assert current_task.error["error"] == "ExecutionDeadlineExceeded"
            if expiry == "run":
                assert current_task == failed
            assert (await store.load(session.id)).status is SessionStatus.INTERRUPTED
            expected_requests = 2 if expiry == "continuation" else 1
            assert len(calls) == 1 and len(provider.requests) == expected_requests
            assert lookups == calls
            with pytest.raises((TaskClaimLost, ExecutionDeadlineExceeded)):
                _ = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id=session.id,
                            task_worker_id=worker,
                            messages=[Message.text("user", "continue")],
                        )
                    )
                ]
            assert len(provider.requests) == expected_requests
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_expired_structured_task_receipt_retains_pending_output_status(tmp_path, monkeypatch):
    test_deadline_failure_retains_exact_settlement_authority(
        "memory",
        None,
        "ordinary",
        False,
        tmp_path,
        "before_recovery",
        monkeypatch,
        structured=True,
    )
