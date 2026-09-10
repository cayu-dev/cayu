from __future__ import annotations

import asyncio
import traceback
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
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
from cayu._exception_groups import iter_exception_tree
from cayu.providers import ModelStreamEvent
from cayu.runtime._tool_effect_state import ToolEffectRecord
from cayu.runtime._tool_round_executor import _ToolRoundPublicationCoordinator
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.sessions import IncompleteSessionRecoveryRequest
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("cancel_during_audit", [0, 1, 2])
@pytest.mark.parametrize("late_boundary", ["tool_return", "terminal_stage"])
def test_public_recovery_fences_and_audits_late_original_completion(
    backend, cancel_during_audit, late_boundary, tmp_path, monkeypatch
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
                str(tmp_path / "late.db"),
                ownership_clock=lambda: now,
                public_authority_alias_codec=codec,
            )
        )
        entered = asyncio.Event()
        stage_entered = asyncio.Event()
        release = asyncio.Event()
        audit_entered = asyncio.Event()
        audit_release = asyncio.Event()
        cancellation_received = asyncio.Event()
        if cancel_during_audit:
            append_audit = store.append_tool_effect_conflict

            async def hold_audit(request):
                audit_entered.set()
                await audit_release.wait()
                return await append_audit(request)

            monkeypatch.setattr(store, "append_tool_effect_conflict", hold_audit)
        mutations = []
        lookups = []
        if late_boundary == "terminal_stage":
            stage_terminal = _ToolRoundPublicationCoordinator.stage_terminal

            async def hold_terminal_stage(coordinator, **kwargs):
                stage_entered.set()
                await release.wait()
                return await stage_terminal(coordinator, **kwargs)

            monkeypatch.setattr(
                _ToolRoundPublicationCoordinator, "stage_terminal", hold_terminal_stage
            )

        class External(Tool):
            spec = ToolSpec(
                name="record",
                effect=ToolEffect.EXTERNAL,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:late-effect", behavior_version="1", implementation_version="1"
                ),
            )

            async def run(self, ctx, args):
                mutations.append(ctx.idempotency_key)
                entered.set()
                if late_boundary == "terminal_stage":
                    return ToolResult(content="verified external outcome")
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        # The old worker has already committed its external effect.
                        # It remains alive until the test releases the real result.
                        continue
                return ToolResult(content="verified external outcome")

        class Provider(_SequencedProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:late-provider", behavior_version="1", implementation_version="1"
                )

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                lookups.append(context.idempotency_key)
                assert lookups == mutations
                return ToolEffectReconciliationResult(
                    outcome="completed",
                    observation="sent",
                    receipt=ToolEffectReceipt(
                        receipt_id="late-receipt",
                        receipt_schema="deployment",
                        receipt_schema_version=1,
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        idempotency_key=context.idempotency_key,
                        outcome="completed",
                        message="verified external outcome",
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
                tool_effect_reconcilers={
                    "record": ToolEffectReconciliationRegistration(
                        reconciler=Reconciler(),
                        spec=_spec(supports_lookup=True),
                    )
                },
            )
            return app

        first = build_app()

        async def run_original():
            try:
                return [
                    event
                    async for event in first.run(
                        RunRequest(
                            session_id="late-effect",
                            agent_name="agent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]
            except asyncio.CancelledError:
                cancellation_received.set()
                raise

        original = asyncio.create_task(run_original())
        try:
            await asyncio.wait_for(entered.wait(), 10)
            if late_boundary == "terminal_stage":
                await asyncio.wait_for(stage_entered.wait(), 10)
            executing = ToolEffectRecord.model_validate(
                await store.load_session_operation("late-effect", store.effect_keys[0])
            )
            assert executing.state == "executing"
            now += timedelta(seconds=600)
            recovery_app = build_app()
            recovered = await asyncio.wait_for(
                recovery_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="late-effect", inactive_for_seconds=0
                    )
                ),
                15,
            )
            assert recovered.status.value == "interrupted"
            assert [action.value for action in recovered.actions] == ["pending_tool_effect"]
            unknown = ToolEffectRecord.model_validate(
                await store.load_session_operation("late-effect", store.effect_keys[0])
            )
            assert unknown.state == "outcome_unknown"
            assert not original.done()
            target = await recovery_app.inspect_tool_effect(
                "late-effect",
                tool_round_id=unknown.intent.tool_round_id,
                tool_call_id=unknown.intent.tool_call_id,
            )
            request = ToolEffectReconciliationRequest(**target.model_dump(), lookup=True)
            receipt_events = [event async for event in recovery_app.reconcile_tool_effect(request)]
            if receipt_events[-1].type.value != "session.completed":
                pytest.fail(repr(receipt_events[-1].payload))
            selected = await store.load_session_operation("late-effect", store.effect_keys[0])
            release.set()
            if cancel_during_audit:
                await asyncio.wait_for(audit_entered.wait(), 10)
                for _ in range(cancel_during_audit):
                    original.cancel()
                    assert original.cancelling() == 1
                    await asyncio.sleep(0)
                    # Owned settlement temporarily consumes each request while
                    # the audit remains in flight and restores them on return.
                    assert original.cancelling() == 0
                audit_release.set()
            old_outcome = (
                await asyncio.wait_for(asyncio.gather(original, return_exceptions=True), 15)
            )[0]
            failures = []
            pending = [old_outcome]
            seen = set()
            while pending:
                error = pending.pop()
                if not isinstance(error, BaseException) or id(error) in seen:
                    continue
                seen.add(id(error))
                failures.append(
                    (
                        type(error).__name__,
                        [
                            (frame.name, frame.lineno)
                            for frame in traceback.extract_tb(error.__traceback__)[-8:]
                        ],
                    )
                )
                pending.extend(iter_exception_tree(error))
                pending.extend((error.__cause__, error.__context__))
            assert (
                await store.load_session_operation("late-effect", store.effect_keys[0]) == selected
            )
            assert (await store.load("late-effect")).status.value == "completed"
            assert len(mutations) == len(lookups) == 1
            assert len(provider.requests) == 2
            events = await store.load_events("late-effect")
            assert (
                len(
                    [
                        event
                        for event in events
                        if event.type.value in {"tool.call.completed", "tool.call.failed"}
                    ]
                )
                == 1
            )
            if not any(
                event.type.value == "tool.effect.reconciliation.conflict" for event in events
            ):
                pytest.fail("Missing loser audit; error ownership trace: " + repr(failures))
            if cancel_during_audit:
                assert isinstance(old_outcome, asyncio.CancelledError), repr(failures)
                assert original.cancelled()
                assert original.cancelling() == cancel_during_audit
                assert cancellation_received.is_set()
        finally:
            release.set()
            audit_release.set()
            if not original.done():
                original.cancel()
            await asyncio.wait_for(asyncio.gather(original, return_exceptions=True), 20)
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_own_terminal_acknowledgement_loss_is_not_a_losing_dispatch(backend, tmp_path, monkeypatch):
    async def scenario():
        store = (
            _ObservingStore()
            if backend == "memory"
            else _ObservingSQLiteStore(str(tmp_path / "own.db"))
        )
        calls = []

        class External(Tool):
            spec = ToolSpec(name="record", effect=ToolEffect.EXTERNAL)

            async def run(self, ctx, args):
                calls.append(ctx.idempotency_key)
                return ToolResult(content="recorded")

        provider = _SequencedProvider(
            [
                _tool_call_response(7),
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="agent", model="test"), tools=[External()])
        stage_terminal = _ToolRoundPublicationCoordinator.stage_terminal
        injected = False

        async def lose_acknowledgement(coordinator, **kwargs):
            nonlocal injected
            result = await stage_terminal(coordinator, **kwargs)
            if not injected:
                injected = True
                raise OSError("own terminal acknowledgement lost")
            return result

        monkeypatch.setattr(
            _ToolRoundPublicationCoordinator, "stage_terminal", lose_acknowledgement
        )
        try:
            initial = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="own-effect",
                        agent_name="agent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert injected
            assert initial[-1].type.value == "session.failed"
            selected = ToolEffectRecord.model_validate(
                await store.load_session_operation("own-effect", store.effect_keys[0])
            )
            assert selected.state == "completed"
            assert not any(
                event.type.value == "tool.effect.reconciliation.conflict"
                for event in await store.load_events("own-effect")
            )
            resumed = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="own-effect",
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
            assert resumed[-1].type.value == "session.completed"
            assert len(calls) == 1
            assert len(provider.requests) == 2
            events = await store.load_events("own-effect")
            assert not any(
                event.type.value == "tool.effect.reconciliation.conflict" for event in events
            )
            assert (
                sum(
                    event.type.value in {"tool.call.completed", "tool.call.failed"}
                    for event in events
                )
                == 1
            )
        finally:
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
