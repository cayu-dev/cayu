from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from tests.core.test_tool_round_execution_identities import _SequencedProvider, _tool_call_response

from cayu import (
    AgentSpec,
    CayuApp,
    IncompleteSessionRecoveryRequest,
    Message,
    ResumeRequest,
    RunRequest,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.providers import ModelStreamEvent
from cayu.runtime import InMemorySessionStore
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime._tool_effect_state import (
    ToolEffectConflict,
    ToolEffectReconciliationRequired,
    ToolEffectRecord,
    ToolEffectStateOwner,
)
from cayu.runtime._tool_round_recovery import pending_tool_round_from_checkpoint
from cayu.runtime.approvals import ToolApprovalRecoveryOutcome
from cayu.runtime.config import CayuConfig, ToolExecutionConfig
from cayu.runtime.event_sinks import EventSink, InMemoryEventSink
from cayu.runtime.interactions import INTERACTION_LIFECYCLE_EVENT_TYPES
from cayu.runtime.tool_rounds import ToolRoundRecoveryRequest
from cayu.storage.sqlite import SQLiteSessionStore


class _ObservingStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(
        self, *, fail_effect_write=False, public_authority_alias_codec=None, ownership_clock=None
    ):
        super().__init__(
            public_authority_alias_codec=public_authority_alias_codec,
            ownership_clock=ownership_clock,
        )
        self.effect_keys = []
        self.fail_effect_write = fail_effect_write

    async def publish_session_operation(self, session_id, **kwargs):
        key = kwargs["idempotency_key"]
        if key.startswith("tool-effect:v1:") and self.fail_effect_write:
            raise RuntimeError("effect publication unavailable")
        result = await super().publish_session_operation(session_id, **kwargs)
        if key.startswith("tool-effect:v1:"):
            self.effect_keys.append(key)
        return result


class _ObservingSQLiteStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path, *, public_authority_alias_codec=None, ownership_clock=None):
        super().__init__(
            path,
            public_authority_alias_codec=public_authority_alias_codec,
            ownership_clock=ownership_clock,
        )
        self.effect_keys = []

    async def publish_session_operation(self, session_id, **kwargs):
        result = await super().publish_session_operation(session_id, **kwargs)
        key = kwargs["idempotency_key"]
        if key.startswith("tool-effect:v1:"):
            self.effect_keys.append(key)
        return result


def test_manual_recovery_cannot_replace_selected_external_terminal():
    from tests.core.test_runtime import _crashed_tool_round_app

    session_id = "selected-effect-manual-conflict"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id)
    round_id = checkpoint["pending_tool_round"]["tool_round_id"]

    async def scenario():
        session = await store.load(session_id)
        owner = ToolEffectStateOwner(runtime_checkpoint_session_store(store))
        selected = await owner.resolve_call(session, tool_round_id=round_id, tool_call_id="call_1")
        assert selected is not None and selected.state == "completed"
        before = await store.load_events(session_id)
        with pytest.raises(ToolEffectConflict, match="cannot replace durable"):
            _ = [
                event
                async for event in app.recover_tool_round(
                    ToolRoundRecoveryRequest(
                        session_id=session_id,
                        round_id=round_id,
                        tool_call_id="call_1",
                        outcome=ToolApprovalRecoveryOutcome.FAILED,
                        message="a contradictory operator result",
                    )
                )
            ]
        assert await owner.load(selected.intent) == selected
        assert [
            event.id
            for event in await store.load_events(session_id)
            if event.type.value in {"tool.call.completed", "tool.call.failed"}
        ] == [
            event.id
            for event in before
            if event.type.value in {"tool.call.completed", "tool.call.failed"}
        ]
        events = [
            event
            async for event in app.resume(
                ResumeRequest(session_id=session_id, messages=[Message.text("user", "continue")])
            )
        ]
        assert events[-1].type.value == "session.completed"
        assert tool.calls == [{}]
        terminals = [
            event
            for event in await store.load_events(session_id)
            if event.type.value in {"tool.call.completed", "tool.call.failed"}
        ]
        assert len(terminals) == 1
        assert terminals[0].id == selected.terminal.event_id
        assert terminals[0].payload["result"]["content"] == "recorded"

    asyncio.run(scenario())


@pytest.mark.parametrize("failed", [False, True])
def test_unprotected_manual_recovery_uses_shared_continuation(monkeypatch, failed):
    from tests.core.test_runtime import SideEffectTool, _crashed_tool_round_app

    # This fixture only counts calls. Explicitly use the unprotected effect
    # contract to verify that receipt guards do not alter existing consumers.
    monkeypatch.setattr(
        SideEffectTool,
        "spec",
        SideEffectTool.spec.model_copy(update={"effect": ToolEffect.IDEMPOTENT}),
    )
    session_id = "unprotected-effect-manual-continuation"
    app, store, tool, checkpoint = _crashed_tool_round_app(session_id)

    async def scenario():
        events = [
            event
            async for event in app.recover_tool_round(
                ToolRoundRecoveryRequest(
                    session_id=session_id,
                    round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                    tool_call_id="call_1",
                    outcome=(
                        ToolApprovalRecoveryOutcome.FAILED
                        if failed
                        else ToolApprovalRecoveryOutcome.COMPLETED
                    ),
                    message="verified manual outcome",
                )
            )
        ]
        assert events[-1].type.value == "session.completed"
        assert tool.calls == [{}]
        transcript = await store.load_transcript(session_id)
        result = transcript[2].content[0]
        assert result.content == "verified manual outcome"
        assert result.is_error is failed

    asyncio.run(scenario())


@pytest.mark.parametrize("effect", [ToolEffect.EXTERNAL, ToolEffect.NONE, ToolEffect.IDEMPOTENT])
def test_real_tool_entry_observes_durable_consumed_exact_intent(effect):
    async def scenario():
        store = _ObservingStore()
        observed = []

        class InspectingTool(Tool):
            spec = ToolSpec(
                name="record",
                effect=effect,
                input_schema={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            )

            async def run(self, ctx, args):
                if effect is ToolEffect.EXTERNAL:
                    assert len(store.effect_keys) == 2
                    assert store.effect_keys[0] == store.effect_keys[1]
                    record = ToolEffectRecord.model_validate(
                        await store.load_session_operation(ctx.session_id, store.effect_keys[0])
                    )
                    assert record.state == "executing"
                    assert record.revision == 1
                    assert record.dispatch_id
                    assert record.intent.idempotency_key == ctx.idempotency_key
                    assert (
                        record.intent.arguments_digest
                        == sha256(canonical_durable_json_bytes(args, "arguments")).hexdigest()
                    )
                    assert record.intent.tool_name == "record"
                    assert record.intent.execution_profile_fingerprint
                    observed.append(record)
                else:
                    assert store.effect_keys == []
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
        app.register_agent(AgentSpec(name="agent", model="test"), tools=[InspectingTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="effect-runtime",
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert any(event.type.value == "session.completed" for event in events), [
            (event.type, event.payload) for event in events if event.type.value == "session.failed"
        ]
        assert len(provider.requests) == 2
        assert len(observed) == (1 if effect is ToolEffect.EXTERNAL else 0)
        if effect is ToolEffect.EXTERNAL:
            settled = ToolEffectRecord.model_validate(
                await store.load_session_operation("effect-runtime", store.effect_keys[0])
            )
            assert settled.state == "completed"
            assert settled.intent == observed[0].intent
            assert settled.dispatch_id == observed[0].dispatch_id
            terminals = [event for event in events if event.type.value == "tool.call.completed"]
            assert len(terminals) == 1
            durable_terminals = [
                event
                for event in await store.load_events("effect-runtime")
                if event.type.value == "tool.call.completed"
            ]
            assert len(durable_terminals) == 1
            assert settled.terminal.event_id == durable_terminals[0].id

    asyncio.run(scenario())


@pytest.mark.parametrize("wrapper_depth", [0, 1, 2])
def test_tool_authored_admission_error_cannot_prove_non_dispatch(wrapper_depth):
    from cayu.environments.admission import (
        ExecutionAdmissionError,
        ExecutionRequirements,
        evaluate_execution_admission,
    )
    from cayu.runtime._tool_execution import ToolDispatchAdmissionRefusal

    async def scenario():
        store = _ObservingStore()
        calls = []
        refusal = ExecutionAdmissionError(
            evaluate_execution_admission(
                candidate="unverified",
                requirements=ExecutionRequirements.trusted(cleanup="confirmed"),
                evidence=None,
            )
        )
        for _ in range(wrapper_depth):
            refusal = ToolDispatchAdmissionRefusal(refusal, owner=object())

        class MutatingTool(Tool):
            spec = ToolSpec(name="record", effect=ToolEffect.EXTERNAL)

            async def run(self, ctx, args):
                calls.append(args)
                raise refusal

        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(_SequencedProvider([_tool_call_response(7)]), default=True)
        app.register_agent(AgentSpec(name="agent", model="test"), tools=[MutatingTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="tool-authored-admission-refusal",
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert len(calls) == 1
        record = ToolEffectRecord.model_validate(
            await store.load_session_operation(
                "tool-authored-admission-refusal", store.effect_keys[0]
            )
        )
        assert record.state == "outcome_unknown"
        assert record.terminal is None
        assert not any(event.type.value == "tool.call.failed" for event in events)

    asyncio.run(scenario())


def test_failed_effect_preparation_never_enters_real_tool():
    async def scenario():
        store = _ObservingStore(fail_effect_write=True)
        calls = []

        class Protected(Tool):
            spec = ToolSpec(name="record", effect=ToolEffect.EXTERNAL)

            async def run(self, ctx, args):
                calls.append(args)
                return ToolResult(content="should not run")

        app = CayuApp(session_store=store, enable_logging=False)
        provider = _SequencedProvider([_tool_call_response(7)])
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="agent", model="test"), tools=[Protected()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="effect-preparation-failure",
                    agent_name="agent",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert calls == []
        assert store.effect_keys == []
        assert any(event.type.value == "session.failed" for event in events)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "signal",
    [
        "failure",
        "cancel",
        "deadline",
        "invalid_return",
        "readback_failure",
        "close_intent_conflict",
        "steering",
    ],
)
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("sink_fails", [False, True])
def test_runtime_unknown_external_effect_preserves_round_and_prevents_resume_dispatch(
    signal, backend, sink_fails, tmp_path, monkeypatch
):
    now = datetime.now(UTC)

    async def scenario(store):
        nonlocal now
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []
        readback_failure = OSError("uncertainty readback unavailable")
        query_events = store.query_events
        fail_readback = signal == "readback_failure"

        async def query_with_failed_uncertainty_readback(query):
            nonlocal fail_readback
            if fail_readback and query.event_type == "tool.effect.outcome_unknown":
                fail_readback = False
                raise readback_failure
            return await query_events(query)

        monkeypatch.setattr(store, "query_events", query_with_failed_uncertainty_readback)

        class FailingUncertaintySink(EventSink):
            async def emit(self, event):
                if event.type.value == "tool.effect.outcome_unknown":
                    raise OSError("uncertainty sink unavailable")

        live_sink = InMemoryEventSink()

        class Ambiguous(Tool):
            spec = ToolSpec(name="record", effect=ToolEffect.EXTERNAL)

            async def run(self, ctx, args):
                calls.append(args)
                entered.set()
                if signal == "steering":
                    await release.wait()
                if signal in {"failure", "readback_failure", "steering"}:
                    raise RuntimeError("external acknowledgement lost")
                if signal == "invalid_return":
                    return "invalid tool result"
                await asyncio.Event().wait()
                raise AssertionError("cancelled tool unexpectedly completed")

        app = CayuApp(
            session_store=store,
            event_sinks=[FailingUncertaintySink() if sink_fails else live_sink],
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(
                    tool_timeout_seconds=0.03 if signal == "deadline" else None
                )
            ),
        )
        provider = _SequencedProvider([_tool_call_response(7)])
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="agent", model="test"), tools=[Ambiguous()])

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="effect-unknown",
                        agent_name="agent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]

        task = asyncio.create_task(run())
        await asyncio.wait_for(entered.wait(), 3)
        if signal == "steering":
            from cayu import StopAfterCurrentToolRoundRequest
            from cayu.runtime.execution_profiles import (
                active_invocation_execution_profile_from_checkpoint,
            )

            # These observing stores delegate publication unchanged. Explicitly
            # opt this test double into the complete upstream steering contract.
            monkeypatch.setattr(type(store), "session_steering_version", 1)
            session = await store.load("effect-unknown")
            profile = active_invocation_execution_profile_from_checkpoint(
                await store.load_checkpoint(session.id)
            )
            assert profile is not None
            try:
                await app.stop_after_current_tool_round(
                    StopAfterCurrentToolRoundRequest(
                        session_id=session.id,
                        session_instance_id=session.instance_id,
                        interaction_id=profile.interaction_id,
                        expected_run_epoch=session.run_epoch,
                        idempotency_key="stop-during-external-effect",
                    )
                )
                assert not task.done() and task.cancelling() == 0
            finally:
                release.set()
        if signal in {"cancel", "close_intent_conflict"}:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            assert task.cancelling() == 1
        elif signal == "readback_failure":
            with pytest.raises(ExceptionGroup) as caught:
                await task
            assert len(caught.value.exceptions) == 2
            assert type(caught.value.exceptions[0]) is ToolEffectReconciliationRequired
            assert caught.value.exceptions[1] is readback_failure
            assert any(
                event.type.value == "session.interrupted"
                for event in await store.load_events("effect-unknown")
            )
        else:
            events = await task
            assert any(event.type.value == "session.interrupted" for event in events), [
                (event.type, event.payload)
                for event in events
                if event.type.value == "session.failed"
            ]
            assert any(event.type.value == "tool.effect.outcome_unknown" for event in events)
        record = ToolEffectRecord.model_validate(
            await store.load_session_operation("effect-unknown", store.effect_keys[0])
        )
        assert record.state == "outcome_unknown"
        uncertainty_events = [
            event
            for event in await store.load_events("effect-unknown")
            if event.type.value == "tool.effect.outcome_unknown"
        ]
        assert len(uncertainty_events) == 1
        uncertainty = uncertainty_events[0]
        assert uncertainty.payload["schema_version"] == 1
        assert uncertainty.payload["state"] == "outcome_unknown"
        assert uncertainty.payload["record_revision"] == record.revision
        assert uncertainty.payload["dispatch_id"] == record.dispatch_id
        assert uncertainty.payload["tool_call_id"] == record.intent.tool_call_id
        assert "arguments" not in uncertainty.payload
        assert "idempotency_key" not in uncertainty.payload
        # Atomic operation publication already creates the existing store-owned
        # delivery intent. A reconstructed app must recover it without either
        # registering or invoking the protected tool again.
        delivery = await store.get_persisted_event_side_effect_delivery(
            session_id=record.intent.session_id, event_id=uncertainty.id
        )
        assert delivery is not None
        assert delivery.status.value == (
            "pending" if signal == "readback_failure" else ("failed" if sink_fails else "delivered")
        )
        if not sink_fails and signal != "readback_failure":
            assert (
                len(
                    [
                        event
                        for event in live_sink.events
                        if event.type.value == "tool.effect.outcome_unknown"
                    ]
                )
                == 1
            )
        sink = InMemoryEventSink()
        recovery_app = CayuApp(session_store=store, event_sinks=[sink], enable_logging=False)
        if sink_fails and signal != "readback_failure":
            # Failed delivery follows the existing store-clock backoff, not an
            # immediate retry. Exercise both sides of that admission boundary.
            assert await recovery_app.recover_persisted_event_side_effects() == []
            assert delivery.next_attempt_at is not None
            now = delivery.next_attempt_at
        recovered = await recovery_app.recover_persisted_event_side_effects()
        recovered_unknown = [
            event for event in recovered if event.type.value == "tool.effect.outcome_unknown"
        ]
        needs_recovery = sink_fails or signal == "readback_failure"
        assert len(recovered_unknown) == (1 if needs_recovery else 0)
        if needs_recovery:
            assert recovered_unknown[0].payload["record_revision"] == record.revision
            assert recovered_unknown[0].payload["dispatch_id"] == record.dispatch_id
        assert [
            event for event in sink.events if event.type.value == "tool.effect.outcome_unknown"
        ] == recovered_unknown
        delivery = await store.get_persisted_event_side_effect_delivery(
            session_id=record.intent.session_id, event_id=uncertainty.id
        )
        assert delivery is not None
        assert delivery.status.value == "delivered"
        assert await recovery_app.recover_persisted_event_side_effects() == []
        assert len(calls) == 1
        assert len(provider.requests) == 1
        checkpoint_store = runtime_checkpoint_session_store(store)
        pending = pending_tool_round_from_checkpoint(
            await checkpoint_store.load_checkpoint("effect-unknown")
        )
        assert pending is not None
        assert not pending.staged_terminals
        lifecycle = [
            event
            for event in await store.load_events("effect-unknown")
            if event.type in INTERACTION_LIFECYCLE_EVENT_TYPES
        ]
        # The existing interaction owner recognizes the retained round as a
        # recovery pause. Receipt recovery must reuse this interaction, not
        # reopen a terminal interaction or manufacture a new one.
        assert lifecycle[-1].type.value == "interaction.paused"
        assert lifecycle[-1].interaction_id == record.intent.interaction_id
        assert lifecycle[-1].payload["pending_action_kind"] == "tool_recovery"
        assert not any(
            event.type.value in {"tool.call.completed", "tool.call.failed"}
            for event in await store.load_events("effect-unknown")
        )
        if signal == "close_intent_conflict":
            from cayu.runtime._recovery_coordinator import _approval_interrupt_close_intent_matches
            from cayu.runtime.sessions import SessionStatus

            def inject_contradictory_close_intent(_session, checkpoint):
                # Deliberately contradictory persisted evidence: real dispatch
                # and cancellation happened above, but a stale close intent
                # claims this exact round was still paused at an approval gate.
                call = checkpoint["pending_tool_round"]["tool_calls"][0]
                call["policy_evidence"] = "authoritative"
                call["policy_decision"] = "require_approval"
                checkpoint["pending_session_interrupt"] = {
                    "interruption_type": "operator_requested",
                    "interruption_request_id": "contradictory-close",
                    "reason": "stale approval close",
                    "approval_close_intent": {
                        "approval_id": "stale-approval",
                        "tool_call_id": record.intent.tool_call_id,
                        "tool_round_id": pending.tool_round_id,
                        "model_step_id": pending.model_step_id,
                        "model_attempt_id": pending.model_attempt_id,
                    },
                }
                return checkpoint

            await checkpoint_store.transition_status_and_checkpoint(
                "effect-unknown",
                from_statuses={SessionStatus.INTERRUPTED},
                to_status=SessionStatus.INTERRUPTING,
                checkpoint_transform=inject_contradictory_close_intent,
            )
            contradictory = await checkpoint_store.load_checkpoint("effect-unknown")
            contradictory_round = pending_tool_round_from_checkpoint(contradictory)
            assert _approval_interrupt_close_intent_matches(
                contradictory, pending_round=contradictory_round
            )
            recovered = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="effect-unknown")
            )
            assert recovered.status is SessionStatus.INTERRUPTED
            assert any(action.value == "pending_tool_effect" for action in recovered.actions)
            assert not any(
                event.type.value in {"tool.call.completed", "tool.call.failed"}
                for event in await store.load_events("effect-unknown")
            )
        with suppress(ToolEffectReconciliationRequired):
            _ = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id="effect-unknown",
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
        assert len(calls) == 1
        assert len(provider.requests) == 1
        assert [
            event.id
            for event in await store.load_events("effect-unknown")
            if event.type.value == "tool.effect.outcome_unknown"
        ] == [uncertainty.id]
        assert (
            pending_tool_round_from_checkpoint(
                await checkpoint_store.load_checkpoint("effect-unknown")
            )
            is not None
        )
        assert (
            ToolEffectRecord.model_validate(
                await store.load_session_operation("effect-unknown", store.effect_keys[0])
            ).state
            == "outcome_unknown"
        )

    async def run_backend():
        store = (
            _ObservingStore(ownership_clock=lambda: now)
            if backend == "memory"
            else _ObservingSQLiteStore(
                str(tmp_path / "runtime-effects.db"), ownership_clock=lambda: now
            )
        )
        try:
            await scenario(store)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run_backend())
