from __future__ import annotations

import asyncio
import warnings
from contextlib import AsyncExitStack
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.core.test_user_input import (
    _FailOnceAppendStore,
    _FailOnceSQLiteAppendStore,
    _ScriptedProvider,
)

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    Message,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolEffect,
    ToolEffectConflict,
    ToolResult,
    ToolSpec,
)
from cayu.core import EventType, ToolResultPart
from cayu.core.thinking import ThinkingConfig
from cayu.runtime import SessionRuntimePublicationConflict, UserInputResponse
from cayu.runtime.hooks import AfterToolCallDecision, RuntimeHook
from cayu.runtime.human_review import HumanReviewContext, HumanReviewReference
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.user_input import UserInputTool


class _CloseAcknowledgementLost(RuntimeError):
    pass


class _CloseReadbackFailed(RuntimeError):
    pass


class _CloseAcknowledgementFaults:
    close_fault = None
    committed_close_id = None
    fail_close_readback = False
    close_readback_failures = 0

    async def publish_runtime_publication(self, session_id, **kwargs):
        request = kwargs["request"]
        if (
            self.armed
            and request.kind == "user-input-close"
            and self.close_fault in {"lost_ack", "lost_ack_readback"}
        ):
            self.armed = False
            result = await super().publish_runtime_publication(session_id, **kwargs)
            assert result.receipt.publication_id == request.publication_id
            self.committed_close_id = request.publication_id
            self.fail_close_readback = self.close_fault == "lost_ack_readback"
            raise _CloseAcknowledgementLost("simulated close acknowledgement loss")
        return await super().publish_runtime_publication(session_id, **kwargs)

    async def load_runtime_publication_receipt(self, session_id, publication_id):
        if self.fail_close_readback and publication_id == self.committed_close_id:
            self.fail_close_readback = False
            self.close_readback_failures += 1
            raise _CloseReadbackFailed("simulated close readback failure")
        return await super().load_runtime_publication_receipt(session_id, publication_id)


class _MemoryEffectStore(_CloseAcknowledgementFaults, _FailOnceAppendStore):
    invocation_lifecycle_command_version = 1


class _SQLiteEffectStore(_CloseAcknowledgementFaults, _FailOnceSQLiteAppendStore):
    invocation_lifecycle_command_version = 1


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "outcome,lookup_control,close_fault",
    [
        (value, None, None)
        for value in ("completed", "failed", "not_found", "unsupported", "conflict")
    ]
    + [
        pytest.param("completed", "cancel", None, id="cancel-lookup"),
        pytest.param("completed", "deadline", None, id="deadline-lookup"),
        pytest.param("completed", "abandon_start", None, id="abandon-start"),
        pytest.param("completed", "abandon_validated", None, id="abandon-validated"),
        pytest.param("completed", "abandon_validated_recovery", None, id="validated-recovery"),
        pytest.param("completed", None, "precommit", id="close-failure"),
        pytest.param("completed", None, "lost_ack", id="close-lost-ack"),
        pytest.param("completed", None, "lost_ack_readback", id="close-lost-ack-readback"),
    ],
)
def test_user_input_sibling_receipt_recovers_without_repeating_external_effect(
    backend, outcome, lookup_control, close_fault, tmp_path, capsys, caplog
):
    async def scenario(store, reopen_store):
        calls = []
        lookups = []
        hook_results = []
        entered = asyncio.Event()
        release = asyncio.Event()
        settled = asyncio.Event()

        class External(Tool):
            spec = ToolSpec(
                name="deploy",
                effect=ToolEffect.EXTERNAL,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:user-input-deploy", behavior_version="1", implementation_version="1"
                ),
            )

            async def run(self, ctx, args):
                calls.append(ctx.idempotency_key)
                raise RuntimeError("external acknowledgement lost")

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                lookups.append(context.idempotency_key)
                if lookup_control in {"cancel", "deadline"} and len(lookups) == 1:
                    entered.set()
                    try:
                        await release.wait()
                    finally:
                        settled.set()
                if outcome in {"not_found", "unsupported", "conflict"}:
                    return ToolEffectReconciliationResult(
                        outcome=outcome, observation="outcome_unknown"
                    )
                return ToolEffectReconciliationResult(
                    outcome=outcome,
                    observation="sent",
                    receipt=ToolEffectReceipt(
                        receipt_id="deployment-receipt",
                        receipt_schema="deployment",
                        receipt_schema_version=1,
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        idempotency_key=context.idempotency_key,
                        outcome=outcome,
                        message="deployment confirmed",
                        source="reconciler",
                        observed_at=datetime(2026, 9, 9, tzinfo=UTC),
                    ),
                )

        class Provider(_ScriptedProvider):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:user-input-provider",
                    behavior_version="1",
                    implementation_version="1",
                )

        provider = Provider(
            [("deploy-call", "deploy", {}), ("input-call", "ask_user", {"question": "Go?"})]
        )

        class ObserveReceipt(RuntimeHook):
            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:user-input-receipt-hook",
                    behavior_version="1",
                    implementation_version="1",
                )

            async def after_tool_call(self, context):
                if context.tool_event.payload.get("effect_reconciled") is True:
                    hook_results.append(context.result.content)
                    assert any(
                        event.id == context.tool_event.id
                        for event in await store.load_events("input-effect")
                    )
                    return AfterToolCallDecision(
                        action="modify",
                        modified_result=ToolResult(content="must not replace receipt"),
                    )

        def build_app():
            application = CayuApp(session_store=store, enable_logging=False)
            application.register_provider(provider, default=True)
            application.register_agent(
                AgentSpec(name="agent", model="test"),
                tools=[External(), UserInputTool()],
                runtime_hooks=[ObserveReceipt()],
                tool_effect_reconcilers={
                    "deploy": ToolEffectReconciliationRegistration(
                        reconciler=Reconciler(),
                        spec=_spec(
                            supports_lookup=True,
                            timeout_seconds=0.1 if lookup_control == "deadline" else 30.0,
                        ),
                    )
                },
            )
            return application

        async def rebuild_app():
            nonlocal store
            if isinstance(store, SQLiteSessionStore):
                store = await reopen_store()
            return build_app()

        app = build_app()
        paused = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="input-effect",
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        awaiting = next(e for e in paused if e.type == EventType.SESSION_AWAITING_USER_INPUT)
        response = UserInputResponse(
            session_id="input-effect", input_id=awaiting.payload["input_id"], answer="yes"
        )
        interrupted = [event async for event in app.resolve_user_input(response)]
        assert len(calls) == 1
        durable_events = await store.load_events("input-effect")
        assert any(e.type.value == "tool.effect.outcome_unknown" for e in durable_events), [
            (e.type.value, e.payload) for e in durable_events
        ]
        started = next(e for e in interrupted if e.type == EventType.TOOL_CALL_STARTED)
        # Drop runtime owners and reopen SQLite, including its read connection.
        # The provider fixture keeps its request history solely to count model calls.
        app = await rebuild_app()
        target = await app.inspect_tool_effect(
            "input-effect",
            tool_round_id=started.payload["tool_round_id"],
            tool_call_id=started.payload["tool_call_id"],
        )
        request = ToolEffectReconciliationRequest(
            **target.model_dump(), lookup=True, user_input_response=response
        )
        before = await store.load_events("input-effect")
        if outcome == "completed" and not lookup_control and not close_fault:
            # Revalidate post-construction mutation through the public entrance,
            # before alias resolution, claiming, or receipt lookup can mutate state.
            class Hostile:
                def __repr__(self):
                    return "private-review-canary"

                def __str__(self):
                    return "private-review-canary"

            reference = HumanReviewReference(
                context=HumanReviewContext(recipient="operator", purpose="recovery"),
                policy_version="1",
                content_tag="a" * 64,
            )
            before_session = (await store.load("input-effect")).model_dump(mode="json")
            for change in (
                {
                    "review_reference": reference.model_copy(
                        update={
                            "context": reference.context.model_copy(update={"recipient": Hostile()})
                        }
                    )
                },
                {
                    "thinking": ThinkingConfig().model_copy(
                        update={"effort": "private-review-canary"}
                    )
                },
                {"thinking": ThinkingConfig().model_copy(update={"effort": Hostile()})},
            ):
                unsafe = request.model_copy(
                    update={"user_input_response": response.model_copy(update=change)}
                )
                capsys.readouterr()
                caplog.clear()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with pytest.raises((ValueError, TypeError)) as raised:
                        _ = [event async for event in app.reconcile_tool_effect(unsafe)]
                assert "private-review-canary" not in str(raised.value)
                assert caught == []
                assert capsys.readouterr() == ("", "")
                assert not caplog.records
                assert lookups == []
                assert len(calls) == len(provider.requests) == 1
                assert await store.load_events("input-effect") == before
                assert (await store.load("input-effect")).model_dump(mode="json") == before_session
        wrong = request.model_copy(
            update={"user_input_response": response.model_copy(update={"answer": "different"})}
        )
        with pytest.raises(SessionRuntimePublicationConflict, match="different resolution"):
            _ = [event async for event in app.reconcile_tool_effect(wrong)]
        assert lookups == []
        assert await store.load_events("input-effect") == before
        expected_lookups = 1
        if lookup_control in {"abandon_start", "abandon_validated", "abandon_validated_recovery"}:
            boundary = (
                EventType.TOOL_EFFECT_RECONCILIATION_STARTED
                if lookup_control == "abandon_start"
                else EventType.TOOL_EFFECT_RECEIPT_VALIDATED
            )
            stream = app.reconcile_tool_effect(request)
            try:
                async for event in stream:
                    if event.type == boundary:
                        break
                else:
                    pytest.fail("Receipt stream did not reach its abandonment boundary.")
            finally:
                await stream.aclose()
            assert len(calls) == len(provider.requests) == 1
            assert len(lookups) == int(lookup_control != "abandon_start")
            durable = await store.load_events("input-effect")
            assert sum(e.payload.get("effect_reconciled") is True for e in durable) == int(
                lookup_control != "abandon_start"
            )
            app = await rebuild_app()
            if lookup_control == "abandon_validated_recovery":
                from cayu import (
                    IncompleteSessionRecoveryAction,
                    IncompleteSessionRecoveryRequest,
                    RecoveryBlockerCode,
                    RecoveryPlanAction,
                    RecoveryPlanRequest,
                    RecoveryPlanSelection,
                )

                plan = await app.plan_recovery(
                    RecoveryPlanRequest(
                        selection=RecoveryPlanSelection(session_ids=("input-effect",))
                    )
                )
                assert plan.items[0].allowed_actions == (RecoveryPlanAction.LEAVE_INTACT,)
                assert RecoveryBlockerCode.USER_INPUT_REQUIRED in {
                    blocker.code for blocker in plan.items[0].blockers
                }
                recovered = await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="input-effect")
                )
                assert IncompleteSessionRecoveryAction.PENDING_USER_INPUT in recovered.actions
                assert len(calls) == len(lookups) == len(provider.requests) == 1
            if lookup_control == "abandon_start":
                target = await app.inspect_tool_effect(
                    "input-effect",
                    tool_round_id=started.payload["tool_round_id"],
                    tool_call_id=started.payload["tool_call_id"],
                )
                request = ToolEffectReconciliationRequest(
                    **target.model_dump(), lookup=True, user_input_response=response
                )
        if lookup_control in {"cancel", "deadline"}:

            async def drain():
                return [event async for event in app.reconcile_tool_effect(request)]

            task = asyncio.create_task(drain())
            try:
                await asyncio.wait_for(entered.wait(), timeout=15)
                if lookup_control == "cancel":
                    task.cancel("cancel receipt lookup")
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=15)
                    assert task.cancelled()
                    assert task.cancelling() == 1
                else:
                    timed_out = await asyncio.wait_for(task, timeout=15)
                    assert timed_out[-1].type == EventType.SESSION_INTERRUPTED
                    assert timed_out[-1].payload["error"] == "Effect reconciliation wait expired."
                    assert not task.cancelled()
                    assert task.cancelling() == 0
                assert not settled.is_set()
                assert len(calls) == 1
                assert len(provider.requests) == 1
                assert not any(
                    e.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                    and e.tool_name == "deploy"
                    for e in await store.load_events("input-effect")
                )
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await asyncio.wait_for(settled.wait(), timeout=15)
            app = await rebuild_app()
            target = await app.inspect_tool_effect(
                "input-effect",
                tool_round_id=started.payload["tool_round_id"],
                tool_call_id=started.payload["tool_call_id"],
            )
            request = ToolEffectReconciliationRequest(
                **target.model_dump(), lookup=True, user_input_response=response
            )
            expected_lookups = 2
        if outcome == "conflict":
            with pytest.raises(ToolEffectConflict):
                _ = [event async for event in app.reconcile_tool_effect(request)]
            before_replay = await store.load_events("input-effect")
            assert any(e.type.value == "tool.effect.reconciliation.conflict" for e in before_replay)
            app = await rebuild_app()
            with pytest.raises(ToolEffectConflict):
                _ = [event async for event in app.reconcile_tool_effect(request)]
            assert await store.load_events("input-effect") == before_replay
            assert len(calls) == len(lookups) == 1
            assert len(provider.requests) == 1
            return
        if close_fault:
            store.armed = True
        recovered = [event async for event in app.reconcile_tool_effect(request)]
        if close_fault == "precommit":
            assert store.armed is False
            assert recovered[-1].type == EventType.SESSION_INTERRUPTED
            assert recovered[-1].payload["error"] == "simulated append failure"
            assert len(calls) == len(lookups) == 1
            assert len(provider.requests) == 1
            app = await rebuild_app()
            recovered = [event async for event in app.reconcile_tool_effect(request)]
        elif close_fault in {"lost_ack", "lost_ack_readback"}:
            assert store.armed is False
            assert store.committed_close_id is not None
            assert store.close_readback_failures == int(close_fault == "lost_ack_readback")
            closure = await store.load_runtime_publication_receipt(
                "input-effect", store.committed_close_id
            )
            assert closure is not None
            assert closure.kind == "user-input-close"
            assert len(calls) == len(lookups) == 1
            if close_fault == "lost_ack_readback":
                assert recovered[-1].type == EventType.SESSION_INTERRUPTED
                assert recovered[-1].payload["error"] == "simulated close readback failure"
                evidence = recovered[-1].payload["failure_evidence"]
                assert evidence["classification"] == "failure"
                for error_type in ("_CloseAcknowledgementLost", "_CloseReadbackFailed"):
                    assert evidence["exception_types"].count(error_type) == 1
                assert len(provider.requests) == 1
                app = await rebuild_app()
                before_replay = await store.load_events("input-effect")
                replayed = [event async for event in app.reconcile_tool_effect(request)]
                assert len(replayed) == 1
                assert replayed[0].type == EventType.TOOL_CALL_COMPLETED
                assert await store.load_events("input-effect") == before_replay
                # Closure consumed the result already; receipt replay is read-only.
                # The existing resume owner continues the remaining model loop.
                recovered = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id="input-effect",
                            messages=[Message.text("user", "Continue the reconciled deployment.")],
                        )
                    )
                ]
        assert lookups == calls * expected_lookups
        assert len(calls) == 1
        if outcome in {"not_found", "unsupported"}:
            assert any(e.type.value == "tool.effect.reconciliation.observed" for e in recovered)
            assert not any(e.type == EventType.SESSION_COMPLETED for e in recovered)
            assert len(provider.requests) == 1
            before_replay = await store.load_events("input-effect")
            app = await rebuild_app()
            replayed = [event async for event in app.reconcile_tool_effect(request)]
            assert len(replayed) == 1
            assert await store.load_events("input-effect") == before_replay
            assert len(calls) == len(lookups) == 1
            return
        assert any(e.type == EventType.SESSION_COMPLETED for e in recovered)
        assert hook_results == ["deployment confirmed"] * (2 if close_fault == "precommit" else 1)
        durable = await store.load_events("input-effect")
        terminals = [e for e in durable if e.payload.get("effect_reconciled") is True]
        assert len(terminals) == 1
        assert terminals[0].payload["result"]["content"] == "deployment confirmed"
        transcript = await store.load_transcript("input-effect")
        tool_results = [
            part
            for message in transcript
            for part in message.content
            if isinstance(part, ToolResultPart) and part.tool_name == "deploy"
        ]
        assert len(tool_results) == 1
        assert tool_results[0].tool_call_id == terminals[0].payload["tool_call_id"]
        assert tool_results[0].content == "deployment confirmed"
        assert len(provider.requests) == 2
        before_replay = await store.load_events("input-effect")
        app = await rebuild_app()
        replayed = [event async for event in app.reconcile_tool_effect(request)]
        assert len(replayed) == 1
        assert replayed[0].type == (
            EventType.TOOL_CALL_COMPLETED if outcome == "completed" else EventType.TOOL_CALL_FAILED
        )
        assert await store.load_events("input-effect") == before_replay
        assert await store.load_transcript("input-effect") == transcript
        assert len(provider.requests) == 2
        assert len(calls) == 1
        assert len(lookups) == expected_lookups
        assert hook_results == ["deployment confirmed"] * (2 if close_fault == "precommit" else 1)

    async def run():
        codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr("A" * 43)})
        )
        async with AsyncExitStack() as cleanup:

            def make_store():
                store = (
                    _MemoryEffectStore(public_authority_alias_codec=codec)
                    if backend == "memory"
                    else _SQLiteEffectStore(
                        tmp_path / "input.sqlite", public_authority_alias_codec=codec
                    )
                )
                store.close_fault = close_fault
                if isinstance(store, SQLiteSessionStore):
                    cleanup.push_async_callback(store.close)
                return store

            async def reopen_store():
                await cleanup.aclose()
                return make_store()

            await scenario(make_store(), reopen_store)

    asyncio.run(run())
