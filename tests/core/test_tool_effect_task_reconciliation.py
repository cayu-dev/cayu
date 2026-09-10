from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.core.test_tool_effect_runtime_dispatch import _ObservingSQLiteStore, _ObservingStore
from tests.core.test_tool_round_execution_identities import _SequencedProvider, _tool_call_response

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    InMemoryTaskStore,
    Message,
    ResumeRequest,
    RunRequest,
    SQLiteTaskStore,
    TaskClaimLost,
    TaskCreate,
    TaskQuery,
    Tool,
    ToolEffect,
    ToolEffectConflict,
    ToolSpec,
    interrupted_task_handoff_request,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime.task_worker import _recover_expired_interrupted_task_handoffs
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)


class _TaskSessionStore(_ObservingStore):
    invocation_lifecycle_command_version = 1
    terminal_interaction_publication_version = 1


class _TaskSQLiteSessionStore(_ObservingSQLiteStore):
    invocation_lifecycle_command_version = 1
    terminal_interaction_publication_version = 1


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("expire_during_lookup", [False, True], ids=["settled", "lease-expired"])
def test_elected_worker_reconciles_real_external_call_without_replay(
    backend, tmp_path, expire_during_lookup
):
    async def scenario():
        task_time = [datetime.now(UTC)]
        store = (
            _TaskSessionStore()
            if backend == "memory"
            else _TaskSQLiteSessionStore(str(tmp_path / "sessions.db"))
        )
        task_store = (
            InMemoryTaskStore(ownership_clock=lambda: task_time[0])
            if backend == "memory"
            else SQLiteTaskStore(str(tmp_path / "tasks.db"), ownership_clock=lambda: task_time[0])
        )
        calls, lookups = [], []
        lookup_entered, release_lookup = asyncio.Event(), asyncio.Event()

        class External(Tool):
            spec = ToolSpec(
                name="record",
                effect=ToolEffect.EXTERNAL,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:task-effect", behavior_version="1", implementation_version="1"
                ),
            )

            async def run(self, ctx, args):
                calls.append(ctx.idempotency_key)
                raise ConnectionError("external acknowledgement lost")

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                assert receipt is None
                lookups.append(context.idempotency_key)
                assert lookups == calls
                if expire_during_lookup:
                    lookup_entered.set()
                    await release_lookup.wait()
                return ToolEffectReconciliationResult(
                    outcome="completed",
                    observation="sent",
                    receipt=ToolEffectReceipt(
                        receipt_id="task-receipt",
                        receipt_schema="deployment",
                        receipt_schema_version=1,
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        idempotency_key=context.idempotency_key,
                        outcome="completed",
                        message="verified external outcome",
                        source="reconciler",
                        observed_at=datetime(2026, 9, 9, tzinfo=UTC),
                    ),
                )

        class Provider(_SequencedProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:task-effect-provider",
                    behavior_version="1",
                    implementation_version="1",
                )

        provider = Provider(
            [
                _tool_call_response(7),
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )

        def build_app():
            app = CayuApp(session_store=store, task_store=task_store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="agent", model="test"),
                tools=[External()],
                tool_effect_reconcilers={
                    "record": ToolEffectReconciliationRegistration(
                        reconciler=Reconciler(), spec=_spec(supports_lookup=True)
                    )
                },
            )
            return app

        async def reopen():
            nonlocal store, task_store
            if backend == "sqlite":
                await task_store.close()
                await store.close()
                store = _TaskSQLiteSessionStore(str(tmp_path / "sessions.db"))
                task_store = SQLiteTaskStore(
                    str(tmp_path / "tasks.db"), ownership_clock=lambda: task_time[0]
                )
            return build_app()

        try:
            app = build_app()
            await task_store.create_task(TaskCreate(task_id="effect-task", type="job"))
            claimed = await task_store.claim_task("original-worker", TaskQuery(type="job"))
            assert claimed is not None
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="task-effect-session",
                        agent_name="agent",
                        task_id=claimed.id,
                        task_worker_id="original-worker",
                        task_lease_expires_at=claimed.lease_expires_at,
                        messages=[Message.text("user", "perform the operation")],
                    )
                )
            ]
            assert initial[-1].type.value == "session.interrupted"
            session = await store.load("task-effect-session")
            attached = await task_store.load_task(claimed.id)
            assert session is not None and attached is not None
            effect_key = store.effect_keys[0]
            before = ToolEffectRecord.model_validate(
                await store.load_session_operation(session.id, effect_key)
            )
            assert before.state == "outcome_unknown"
            await task_store.release_interrupted_task_worker(
                interrupted_task_handoff_request(attached, session_run_epoch=session.run_epoch)
            )
            elected = (
                await task_store.claim_interrupted_task_continuation(
                    "elected-worker", TaskQuery(type="job"), handoff_id=str(uuid4())
                )
            ).task
            assert elected is not None
            request = ToolEffectReconciliationRequest(
                **{
                    name: getattr(before.intent, name)
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
                expected_revision=before.revision,
                lookup=True,
                task_worker_id="elected-worker",
                task_handoff_id=elected.interrupted_handoff_id,
            )
            app = await reopen()
            events_before = await store.load_events(session.id)
            for changes in (
                {"task_worker_id": "original-worker"},
                {"task_handoff_id": str(uuid4())},
            ):
                with pytest.raises(TaskClaimLost):
                    _ = [
                        event
                        async for event in app.reconcile_tool_effect(
                            request.model_copy(update=changes)
                        )
                    ]
                assert lookups == []
                assert await store.load(session.id) == session
                assert await store.load_events(session.id) == events_before
                assert await task_store.load_task(claimed.id) == elected
                assert (
                    ToolEffectRecord.model_validate(
                        await store.load_session_operation(session.id, effect_key)
                    )
                    == before
                )
            if expire_during_lookup:
                recovered = []

                async def collect():
                    async for event in app.reconcile_tool_effect(request):
                        recovered.append(event)

                owner = asyncio.create_task(collect())
                try:
                    await asyncio.wait_for(lookup_entered.wait(), timeout=15)
                    assert elected.lease_expires_at is not None
                    task_time[0] = elected.lease_expires_at + timedelta(seconds=1)
                    release_lookup.set()
                    outcome = await asyncio.gather(owner, return_exceptions=True)
                    assert len(provider.requests) == 1
                    assert len(calls) == len(lookups) == 1
                    assert isinstance(outcome[0], TaskClaimLost), outcome
                    assert not any(event.type.value == "session.completed" for event in recovered)
                    retained = ToolEffectRecord.model_validate(
                        await store.load_session_operation(session.id, effect_key)
                    )
                    assert retained.state == "reconciled_completed"
                    assert retained.intent == before.intent
                    assert retained.terminal is not None
                    assert retained.terminal.receipt.receipt_id == "task-receipt"
                    events_at_expiry = await store.load_events(session.id)
                    with pytest.raises(TaskClaimLost):
                        _ = [event async for event in app.reconcile_tool_effect(request)]
                    assert await store.load_events(session.id) == events_at_expiry
                    assert len(provider.requests) == 1
                    assert len(calls) == len(lookups) == 1
                finally:
                    release_lookup.set()
                    if not owner.done():
                        owner.cancel()
                    await asyncio.gather(owner, return_exceptions=True)
                app = await reopen()
                handoffs = await _recover_expired_interrupted_task_handoffs(
                    app, task_store, after=None, limit=10, stop=None
                )
                assert handoffs.recovered == 1
                replacement = (
                    await task_store.claim_interrupted_task_continuation(
                        "replacement-worker", TaskQuery(type="job"), handoff_id=str(uuid4())
                    )
                ).task
                assert replacement is not None
                continued = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id=session.id,
                            messages=[Message.text("user", "continue from the verified result")],
                            task_worker_id="replacement-worker",
                            task_handoff_id=replacement.interrupted_handoff_id,
                        )
                    )
                ]
                assert continued[-1].type.value == "session.completed"
                assert len(provider.requests) == 2
                assert len(calls) == len(lookups) == 1
                assert (
                    ToolEffectRecord.model_validate(
                        await store.load_session_operation(session.id, effect_key)
                    ).terminal
                    == retained.terminal
                )
                assert (
                    len(
                        [
                            event
                            for event in await store.load_events(session.id)
                            if event.type.value in {"tool.call.completed", "tool.call.failed"}
                        ]
                    )
                    == 1
                )
                return
            recovered = [event async for event in app.reconcile_tool_effect(request)]
            assert recovered[-1].type.value == "session.completed"
            selected = ToolEffectRecord.model_validate(
                await store.load_session_operation(session.id, effect_key)
            )
            assert selected.state == "reconciled_completed"
            assert selected.intent == before.intent
            assert len(calls) == len(lookups) == 1
            assert len(provider.requests) == 2
            terminals = [
                event
                for event in await store.load_events(session.id)
                if event.type.value in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminals) == 1
            events_after = await store.load_events(session.id)
            session_after = await store.load(session.id)
            task_after = await task_store.load_task(claimed.id)
            app = await reopen()
            for changes in (
                {"task_worker_id": "original-worker"},
                {"task_handoff_id": str(uuid4())},
                {"max_steps": 1},
            ):
                with pytest.raises(ToolEffectConflict):
                    _ = [
                        event
                        async for event in app.reconcile_tool_effect(
                            request.model_copy(update=changes)
                        )
                    ]
            replay = [event async for event in app.reconcile_tool_effect(request)]
            assert replay
            assert await store.load_events(session.id) == events_after
            assert await store.load(session.id) == session_after
            assert await task_store.load_task(claimed.id) == task_after
            assert len(calls) == len(lookups) == 1
            assert len(provider.requests) == 2
        finally:
            if backend == "sqlite":
                await task_store.close()
                await store.close()

    asyncio.run(scenario())
