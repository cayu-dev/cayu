from __future__ import annotations

import asyncio
import sys
from decimal import Decimal

import pytest
from tests.core._auxiliary_inference_crash import SESSION_ID, build_app

from cayu.events import EventType
from cayu.runtime import (
    IncompleteSessionRecoveryRequest,
    RecoveryBlockerCode,
    RecoveryDecision,
    RecoveryExecutionRequest,
    RecoveryItemExecutionStatus,
    RecoveryPlanAction,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
)


def test_public_parent_cancellation_leaves_recoverable_auxiliary_accounting():
    from tests.core._auxiliary_inference_crash import BOUNDS

    from cayu import AgentSpec, CayuApp, Message, RunRequest, ScriptedModelProvider
    from cayu.providers import ModelRequest, ModelStreamEvent
    from cayu.tools.base import Tool, ToolSpec
    from cayu.tools.inference import AuxiliaryInferencePolicy

    async def run():
        started = asyncio.Event()

        class Provider(ScriptedModelProvider):
            async def stream(self, request):
                if self.requests:
                    self._consume_batch(request)
                    started.set()
                    await asyncio.Event().wait()
                    return
                async for event in super().stream(request):
                    yield event

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="Wait in auxiliary inference",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=BOUNDS, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                await ctx.inference.invoke(
                    ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                    purpose="tool.summary",
                    limits=BOUNDS,
                )
                raise AssertionError("The parent should cancel this call")

        app = CayuApp(enable_logging=False)
        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", arguments={}, id="parent"),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({})],
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])

        async def consume():
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="parent-cancel-auxiliary",
                    messages=[Message.text("user", "go")],
                )
            ):
                pass

        parent = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), timeout=10)
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
        assert parent.cancelled() and parent.cancelling() == 1
        assert len(provider.requests) == 2
        active = await app.session_store.load_active_model_completion_stage(
            "parent-cancel-auxiliary"
        )
        assert active is not None and active.stage.state == "completed"
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="parent-cancel-auxiliary",
                inactive_for_seconds=None,
            )
        )
        assert len(provider.requests) == 2
        assert (
            await app.session_store.load_active_model_completion_stage("parent-cancel-auxiliary")
            is None
        )
        events = await app.session_store.load_events("parent-cancel-auxiliary")
        auxiliary = [
            event for event in events if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
        ]
        assert len(auxiliary) == 1
        assert auxiliary[0].payload["auxiliary_outcome"] == "cancelled"

    asyncio.run(run())


def test_unattested_store_rejects_public_auxiliary_request_before_dispatch():
    from tests.core._auxiliary_inference_crash import BOUNDS

    from cayu import AgentSpec, CayuApp, Message, RunRequest, ScriptedModelProvider
    from cayu.providers import ModelRequest, ModelStreamEvent
    from cayu.sessions.base import InMemorySessionStore
    from cayu.tools.base import Tool, ToolResult, ToolSpec
    from cayu.tools.inference import AuxiliaryInferencePolicy

    class UnsupportedStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        model_completion_recovery_fence_version = 0

    rejected = []

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            description="Try unsupported inference",
            input_schema={"type": "object"},
            auxiliary_inference=AuxiliaryInferencePolicy(limits=BOUNDS, purposes=("tool.summary",)),
        )

        async def run(self, ctx, args):
            with pytest.raises(NotImplementedError, match="atomic model recovery"):
                await ctx.inference.invoke(
                    ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                    purpose="tool.summary",
                    limits=BOUNDS,
                )
            rejected.append(True)
            return ToolResult(content="unsupported")

    async def run():
        store = UnsupportedStore()
        app = CayuApp(session_store=store, enable_logging=False)
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", arguments={}, id="parent"),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})],
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="unsupported-auxiliary",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert rejected == [True]
        assert len(provider.requests) == 2
        assert not any(event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED for event in events)
        assert await store.load_active_model_completion_stage("unsupported-auxiliary") is None

    asyncio.run(run())


def test_auxiliary_recovery_requires_concrete_atomic_hook_attestation():
    from cayu.sessions.base import InMemorySessionStore

    class Unattested(InMemorySessionStore):
        async def _complete_model_completion_stage_atomic(self, prepared):
            raise AssertionError("Unsupported hook must not receive recovery writes")

    class Attested(Unattested):
        model_completion_recovery_fence_version = 1

    assert InMemorySessionStore()._supports_model_completion_recovery_fence_protocol()
    assert not Unattested()._supports_model_completion_recovery_fence_protocol()
    assert Attested()._supports_model_completion_recovery_fence_protocol()

    async def run():
        with pytest.raises(NotImplementedError, match="atomic model recovery"):
            await Unattested().complete_recovered_model_completion_stage(
                "session",
                stage_id="stage",
                publication=None,
                recovery_fence=None,
            )

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["dispatch", "terminal"])
@pytest.mark.parametrize("through_plan", [False, True])
def test_public_recovery_settles_auxiliary_after_actual_process_loss(
    sqlite_resources, phase, through_plan
):
    async def scenario():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "tests.core._auxiliary_inference_crash",
            str(sqlite_resources.root),
            phase,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        assert process.returncode == (73 if phase == "dispatch" else 74), stderr.decode()
        assert stdout.count(b"PROVIDER_DISPATCH") == 2
        app, provider, tool, store, ledger = build_app(sqlite_resources.root, crash=False)
        sqlite_resources.own(ledger)
        sqlite_resources.own(store)
        try:
            before = await store.load_active_model_completion_stage(SESSION_ID)
            assert before is not None and before.stage.state == (
                "in_flight" if phase == "dispatch" else "completed"
            )
            assert before.stage.purpose == "auxiliary-inference"
            assert len(before.stage.reservation_ids) == 1
            reservation_id = before.stage.reservation_ids[0]
            reservation = await ledger.load_reservation(reservation_id)
            assert reservation is not None and reservation.status == "active"
            assert reservation.dispatch_id == before.stage.intent["model_attempt_id"]
            checkpoint = await store.load_checkpoint(SESSION_ID)
            from cayu.runtime._tool_round_recovery import pending_tool_round_from_checkpoint
            from cayu.runtime.execution_profiles import (
                active_invocation_execution_profile_from_checkpoint,
            )

            profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            pending = pending_tool_round_from_checkpoint(checkpoint)
            assert profile is not None and pending is not None
            assert pending.execution_profile_fingerprint == profile.profile.fingerprint
            if through_plan:
                prior_events = await store.load_events(SESSION_ID)
                plan = await app.plan_recovery(
                    RecoveryPlanRequest(
                        selection=RecoveryPlanSelection(
                            session_ids=(SESSION_ID,),
                            inactive_for_seconds=0,
                        )
                    )
                )
                assert await store.load_checkpoint(SESSION_ID) == checkpoint
                assert await store.load_events(SESSION_ID) == prior_events
                item = plan.items[0]
                assert item.active_model_stage.reservation_count == 1
                assert RecoveryPlanAction.AUTOMATIC_REPAIR not in item.allowed_actions
                assert RecoveryBlockerCode.TOOL_EFFECT_OUTCOME_UNKNOWN in {
                    blocker.code for blocker in item.blockers
                }
                assert RecoveryBlockerCode.MODEL_EFFECT_OUTCOME_UNKNOWN not in {
                    blocker.code for blocker in item.blockers
                }
                assert RecoveryPlanAction.MODEL_MARK_FAILED not in item.allowed_actions
                request = RecoveryExecutionRequest(
                    plan=plan,
                    execution_id="leave-auxiliary",
                    decisions=(
                        RecoveryDecision(
                            item_id=item.item_id, action=RecoveryPlanAction.LEAVE_INTACT
                        ),
                    ),
                )
                result = await app.execute_recovery(request)
                assert result.items[0].status is RecoveryItemExecutionStatus.LEFT_INTACT, result
                replay = await app.execute_recovery(request)
                assert replay.items[0].status is RecoveryItemExecutionStatus.LEFT_INTACT
                assert await store.load_checkpoint(SESSION_ID) == checkpoint
                assert (
                    await store.load_model_completion_stage(SESSION_ID, before.stage.stage_id)
                    == before.stage
                )
            recovered = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=SESSION_ID,
                    inactive_for_seconds=None,
                )
            )
            assert provider.requests == []
            assert tool.calls == 0
            assert await store.load_active_model_completion_stage(SESSION_ID) is None
            terminal = await store.load_model_completion_stage(SESSION_ID, before.stage.stage_id)
            assert terminal is not None and terminal.state == "completed"
            settled = await ledger.load_reservation(reservation_id)
            assert settled is not None and settled.status == "reconciled"
            assert reservation.reserved_amount == Decimal("0.002")
            assert settled.actual_amount == (
                Decimal("0.002") if phase == "dispatch" else Decimal("0.000005")
            )
            durable = await store.load_events(SESSION_ID)
            auxiliary = [
                event
                for event in durable
                if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ]
            assert len(auxiliary) == 1
            assert auxiliary[0].payload["auxiliary_outcome"] == (
                "outcome_unknown" if phase == "dispatch" else "completed"
            )
            assert auxiliary[0].payload["usage_status"] == (
                "missing" if phase == "dispatch" else "observed"
            )
            from cayu.runtime.evidence import (
                RuntimeEvidenceOperation,
                RuntimeEvidenceRequest,
                runtime_evidence,
            )

            report = await runtime_evidence(
                app,
                RuntimeEvidenceRequest(
                    root_session_id=SESSION_ID, max_sessions=10, max_events=1000
                ),
            )
            projected = [
                attempt
                for attempt in report.sessions[0].attempts
                if attempt.operation is RuntimeEvidenceOperation.AUXILIARY_INFERENCE
            ]
            assert len(projected) == 1
            assert projected[0].status.value == auxiliary[0].payload["auxiliary_outcome"]
            assert projected[0].attempt_ordinal == 1
            assert projected[0].auxiliary_inference.purpose == "tool.summary"
            assert report.lineage_totals.model_step_count == 1
            assert report.lineage_totals.attempt_count == 2
            assert (
                auxiliary[0].payload["model_attempt_id"] == before.stage.intent["model_attempt_id"]
            )
            assert recovered is None or (
                len(
                    [
                        event
                        for event in recovered.events
                        if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
                    ]
                )
                == 1
            )
            assert len([event for event in durable if event.type is EventType.MODEL_COMPLETED]) == 1
            usage = await app.get_session_usage(SESSION_ID)
            assert usage.model_steps == 1
            assert usage.unmeasured_model_attempts == (1 if phase == "dispatch" else 0)
            assert usage.usage.total_tokens == (2 if phase == "dispatch" else 7)
            settlements = [
                event
                for event in durable
                if event.type is EventType.BUDGET_RECONCILED
                and event.payload.get("reservation_id") == reservation_id
            ]
            assert len(settlements) == 1
            await store.close()
            await ledger.close()
            app, provider, tool, store, ledger = build_app(sqlite_resources.root, crash=False)
            sqlite_resources.own(ledger)
            sqlite_resources.own(store)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=SESSION_ID,
                    inactive_for_seconds=None,
                )
            )
            assert provider.requests == [] and tool.calls == 0
            assert (
                len(
                    [
                        event
                        for event in await store.load_events(SESSION_ID)
                        if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
                    ]
                )
                == 1
            )
            assert [
                event.id
                for event in await store.load_events(SESSION_ID)
                if event.type is EventType.BUDGET_RECONCILED
                and event.payload.get("reservation_id") == reservation_id
            ] == [settlements[0].id]
        finally:
            await store.close()
            await ledger.close()

    async def run():
        async with sqlite_resources:
            await scenario()

    asyncio.run(run())
