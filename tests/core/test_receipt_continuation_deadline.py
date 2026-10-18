"""A receipt may settle after expiry; its sibling tools keep the original timer."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.core.test_tool_effect_task_reconciliation import (
    _TaskSessionStore,
    _TaskSQLiteSessionStore,
)
from tests.core.test_tool_round_execution_identities import _SequencedProvider

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.deadlines import ExecutionDeadline, current_execution_deadline
from cayu.providers import ModelStreamEvent
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.storage import SQLiteTaskStore
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus
from cayu.tools import Tool, ToolEffect, ToolResult, ToolSpec, UserInputTool


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("expiry_phase", ["sibling", "phase_switch"])
def test_receipt_recovery_times_siblings_before_another_external_effect(
    backend, attached, expiry_phase, tmp_path, monkeypatch
):
    import cayu.deadlines as deadline_module

    async def scenario():
        store = (
            _TaskSessionStore()
            if backend == "memory"
            else _TaskSQLiteSessionStore(tmp_path / "sessions.db")
        )
        tasks = (
            InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(tmp_path / "tasks.db")
        )
        calls = []
        boundary = ExecutionDeadline.after(600)
        advance = timedelta()
        receipt_settled = False
        query_events = store.query_events

        async def query_with_expiry(query):
            nonlocal advance
            if (
                expiry_phase == "phase_switch"
                and receipt_settled
                and query.limit == 1
                and query.event_types
                == (
                    EventType.SESSION_STARTED,
                    EventType.SESSION_RESUMED,
                    EventType.SESSION_COMPLETED,
                    EventType.SESSION_FAILED,
                    EventType.SESSION_INTERRUPTED,
                )
            ):
                calls.append("phase_switch_expired")
                advance = timedelta(seconds=601)
            return await query_events(query)

        monkeypatch.setattr(store, "query_events", query_with_expiry)

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.now(tz) + advance

        monkeypatch.setattr(deadline_module, "datetime", Clock)

        class External(Tool):
            spec = ToolSpec(name="record", effect=ToolEffect.EXTERNAL, parallel_safe=False)

            async def run(self, context, arguments):
                calls.append("record")
                raise ConnectionError("external acknowledgement lost")

        class Slow(Tool):
            spec = ToolSpec(name="slow", effect=ToolEffect.NONE, parallel_safe=False)

            async def run(self, context, arguments):
                nonlocal advance
                calls.append("slow")
                assert current_execution_deadline().expires_at == boundary.expires_at
                # Expire only after receipt recovery has admitted this sibling.
                # Store/setup latency must not consume the boundary under test.
                advance = timedelta(seconds=601)
                timers = [
                    timer
                    for timer, selected in deadline_module._TIMERS.get()
                    if selected.expires_at == boundary.expires_at and timer.when() is not None
                ]
                assert timers
                timers[-1].reschedule(asyncio.get_running_loop().time())
                try:
                    await asyncio.Event().wait()
                finally:
                    calls.append("slow_stopped")
                return ToolResult(content="unreachable")

        class Later(Tool):
            spec = ToolSpec(name="later", effect=ToolEffect.EXTERNAL, parallel_safe=False)

            async def run(self, context, arguments):
                calls.append("later")
                return ToolResult(content="must never execute")

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                nonlocal receipt_settled
                assert calls == ["record"]
                receipt_settled = True
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
                        message="Verified external result",
                        source="reconciler",
                        observed_at=datetime.now(UTC),
                    ),
                )

        provider = _SequencedProvider(
            [
                [
                    ModelStreamEvent.tool_call(id="one", name="record", arguments={}),
                    ModelStreamEvent.tool_call(
                        id="ask", name="ask_user", arguments={"question": "Proceed?"}
                    ),
                    ModelStreamEvent.tool_call(id="slow", name="slow", arguments={}),
                    ModelStreamEvent.tool_call(id="later", name="later", arguments={}),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ]
        )
        app = CayuApp(
            session_store=store, task_store=tasks if attached else None, enable_logging=False
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="agent", model="test"),
            tools=[External(), UserInputTool(), Slow(), Later()],
            tool_effect_reconcilers={
                "record": ToolEffectReconciliationRegistration(
                    reconciler=Reconciler(), spec=_spec(supports_lookup=True)
                )
            },
        )
        try:
            task = await tasks.create_task(TaskCreate(type="job")) if attached else None
            paused = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="receipt-siblings",
                        agent_name="agent",
                        task_id=None if task is None else task.id,
                        messages=[Message.text("user", "perform")],
                        execution_deadline=boundary,
                    )
                )
            ]
            pending = next(
                event for event in paused if event.type is EventType.SESSION_AWAITING_USER_INPUT
            )
            response = UserInputResponse(
                session_id="receipt-siblings", input_id=pending.payload["input_id"], answer="yes"
            )
            interrupted = [event async for event in app.resolve_user_input(response)]
            assert interrupted[-1].type is EventType.SESSION_INTERRUPTED
            assert calls == ["record"]
            session = await store.load("receipt-siblings")
            effect = ToolEffectRecord.model_validate(
                await store.load_session_operation(session.id, store.effect_keys[0])
            )
            recovered = [
                event
                async for event in app.reconcile_tool_effect(
                    ToolEffectReconciliationRequest(
                        **{
                            name: getattr(effect.intent, name)
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
                        expected_revision=effect.revision,
                        lookup=True,
                        user_input_response=response,
                    )
                )
            ]
            assert calls == (
                ["record", "slow", "slow_stopped"]
                if expiry_phase == "sibling"
                else ["record", "phase_switch_expired"]
            )
            assert recovered[-1].type is EventType.SESSION_INTERRUPTED
            assert recovered[-1].payload["limit"] == "execution_deadline"
            assert len(provider.requests) == 1
            if task is not None:
                assert (await tasks.load_task(task.id)).status is TaskStatus.FAILED
        finally:
            if backend == "sqlite":
                await store.close()
                await tasks.close()

    asyncio.run(scenario())
