from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
from tests.core._workload_secret_support import RequireApprovalPolicy
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.core.test_tool_effect_runtime_dispatch import _ObservingSQLiteStore, _ObservingStore
from tests.core.test_tool_round_execution_identities import _SequencedProvider, _tool_call_response

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    Message,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.core.tools import DurableToolRecoveryEvidence
from cayu.providers import ModelStreamEvent
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime.approvals import ToolApprovalDecision, ToolApprovalRequest
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.sessions import IncompleteSessionRecoveryRequest
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("approval_gate", [False, True], ids=["ordinary", "approval"])
def test_competing_recovery_preserves_approval_and_audits_late_reconciler(
    backend, approval_gate, tmp_path
):
    async def scenario():
        now = datetime.now(UTC)
        codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="test",
                keys={"test": SecretStr("A" * 43)},
            )
        )
        store = (
            _ObservingStore(ownership_clock=lambda: now, public_authority_alias_codec=codec)
            if backend == "memory"
            else _ObservingSQLiteStore(
                str(tmp_path / "reconciler-race.db"),
                ownership_clock=lambda: now,
                public_authority_alias_codec=codec,
            )
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        callback_returned = asyncio.Event()
        mutations = []
        lookups = []
        native_reads = []
        native_ready = False

        class External(Tool):
            spec = ToolSpec(
                name="record",
                effect=ToolEffect.EXTERNAL,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:reconciler-race-tool",
                    behavior_version="1",
                    implementation_version="1",
                ),
            )

            async def run(self, ctx, args):
                mutations.append(ctx.idempotency_key)
                raise ConnectionError("external acknowledgement lost")

            async def reconcile_durable_tool_call(self, **kwargs):
                native_reads.append(kwargs["idempotency_key"])
                if not native_ready:
                    return None
                assert kwargs["idempotency_key"] == mutations[0]
                return DurableToolRecoveryEvidence(
                    disposition="confirmed",
                    result=ToolResult(content="native confirmed result"),
                )

        class Provider(_SequencedProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:reconciler-race-provider",
                    behavior_version="1",
                    implementation_version="1",
                )

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                lookups.append(context.idempotency_key)
                if len(lookups) == 1:
                    entered.set()
                    await release.wait()
                    callback_returned.set()
                return ToolEffectReconciliationResult(
                    outcome="completed",
                    observation="sent",
                    receipt=ToolEffectReceipt(
                        receipt_id="late-validator-receipt",
                        receipt_schema="deployment",
                        receipt_schema_version=1,
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        idempotency_key=context.idempotency_key,
                        outcome="completed",
                        message="different late validator result",
                        source="reconciler",
                        observed_at=now,
                    ),
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
            app = CayuApp(session_store=store, clock=lambda: now, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="agent", model="test"),
                tools=[External()],
                tool_policy=RequireApprovalPolicy() if approval_gate else None,
                tool_effect_reconcilers={
                    "record": ToolEffectReconciliationRegistration(
                        reconciler=Reconciler(),
                        spec=_spec(supports_lookup=True, timeout_seconds=60),
                    ),
                },
            )
            return app

        app = build_app()
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="late-validator",
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert initial[-1].type.value == "session.interrupted"
        if approval_gate:
            approval = next(
                event for event in initial if event.type.value == "tool.call.approval_requested"
            )
            initial = [
                event
                async for event in app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id="late-validator",
                        approval_id=approval.payload["approval"]["approval_id"],
                        tool_round_id=approval.payload["tool_round_id"],
                        tool_call_id=approval.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]
            assert initial[-1].type.value == "session.interrupted"
        unknown = ToolEffectRecord.model_validate(
            await store.load_session_operation("late-validator", store.effect_keys[0])
        )
        target = await app.inspect_tool_effect(
            "late-validator",
            tool_round_id=unknown.intent.tool_round_id,
            tool_call_id=unknown.intent.tool_call_id,
        )
        request = ToolEffectReconciliationRequest(**target.model_dump(), lookup=True)

        async def reconcile():
            return [event async for event in app.reconcile_tool_effect(request)]

        pending = asyncio.create_task(reconcile())
        try:
            await asyncio.wait_for(entered.wait(), 10)
            admitted = ToolEffectRecord.model_validate(
                await store.load_session_operation("late-validator", store.effect_keys[0])
            )
            assert admitted.reconciliation_attempt is not None
            assert admitted.revision == unknown.revision + 1
            assert admitted.reconciliation_attempt.source_revision == unknown.revision
            native_ready = True
            now += timedelta(seconds=600)
            replacement = build_app()
            await asyncio.wait_for(
                replacement.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="late-validator", inactive_for_seconds=0
                    )
                ),
                15,
            )
            selected = await store.load_session_operation("late-validator", store.effect_keys[0])
            if approval_gate:
                # Automatic round recovery cannot bypass the pending approval
                # owner, even when a tool offers native positive evidence.
                assert ToolEffectRecord.model_validate(selected).state == "outcome_unknown"
                assert native_reads == []
                target = await replacement.inspect_tool_effect(
                    "late-validator",
                    tool_round_id=unknown.intent.tool_round_id,
                    tool_call_id=unknown.intent.tool_call_id,
                )
                resumed = [
                    event
                    async for event in replacement.reconcile_tool_effect(
                        ToolEffectReconciliationRequest(**target.model_dump(), lookup=True)
                    )
                ]
                selected = await store.load_session_operation(
                    "late-validator", store.effect_keys[0]
                )
                assert ToolEffectRecord.model_validate(selected).state == "reconciled_completed"
            else:
                assert ToolEffectRecord.model_validate(selected).state == "completed"
            assert ToolEffectRecord.model_validate(selected).reconciliation_attempt is None
            audits_before_release = [
                event
                for event in await store.load_events("late-validator")
                if event.type.value == "tool.effect.reconciliation.conflict"
            ]
            assert len(audits_before_release) == 1
            assert audits_before_release[0].payload["kind"] == "reconciliation_superseded"
            assert audits_before_release[0].payload["request_digest"] == (
                admitted.reconciliation_attempt.request_digest
            )
            assert not callback_returned.is_set()
            owner_finished_before_release = pending.done()
            if not approval_gate:
                resumed = [
                    event
                    async for event in replacement.resume(
                        ResumeRequest(
                            session_id="late-validator", messages=[Message.text("user", "continue")]
                        )
                    )
                ]
            assert resumed[-1].type.value == "session.completed"
            release.set()
            loser = (await asyncio.wait_for(asyncio.gather(pending, return_exceptions=True), 15))[0]
            await asyncio.wait_for(callback_returned.wait(), 5)
            assert isinstance(loser, BaseException)
            assert (
                await store.load_session_operation("late-validator", store.effect_keys[0])
                == selected
            )
            assert len(mutations) == 1
            assert len(lookups) == (2 if approval_gate else 1)
            assert len(provider.requests) == 2
            events = await store.load_events("late-validator")
            assert (
                sum(
                    event.type.value in {"tool.call.completed", "tool.call.failed"}
                    for event in events
                )
                == 1
            )
            assert [
                event
                for event in events
                if event.type.value == "tool.effect.reconciliation.conflict"
            ] == audits_before_release
            if not approval_gate:
                assert owner_finished_before_release
        finally:
            release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.wait_for(asyncio.gather(pending, return_exceptions=True), 20)
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
