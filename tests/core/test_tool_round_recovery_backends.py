"""Public recovery preserves recorded results and native reconciliation evidence."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core._execution_profile_fixtures import (
    create_admitted_session,
    profiled_session_identity,
)
from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider
from tests.core.test_tool_round_publication_failure_matrix import store_factory as store_factory

from cayu import AgentSpec, CayuApp, Event, EventType, Message
from cayu.runtime import _runtime_records as records
from cayu.runtime import _tool_execution as execution
from cayu.runtime import _tool_round_recovery as recovery
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_units import new_model_step_identity
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions.base import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    RunRequest,
    SessionStatus,
)
from cayu.tools.base import DurableToolRecoveryEvidence, Tool, ToolEffect, ToolResult, ToolSpec
from cayu.tools.exposure import ToolCapabilityCeiling
from cayu.tools.policy import ToolPolicyDecision, ToolPolicyResult


class RecoveryStore:
    """Use the native backend without changing its transaction boundaries."""


class NeverRunTool(Tool):
    def __init__(self, name, *, effect=ToolEffect.IDEMPOTENT):
        super().__init__(
            ToolSpec(
                name=name,
                description="Recover recorded work without executing it again.",
                input_schema={"type": "object"},
                effect=effect,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name=f"tests:round-recovery:{name}",
                    behavior_version="1",
                    implementation_version="1",
                ),
            )
        )
        self.calls = 0

    async def run(self, ctx, args):
        self.calls += 1
        raise AssertionError("Recovery must not execute a tool.")


class NativeRecoveryTool(NeverRunTool):
    def __init__(self, disposition, *, effect=ToolEffect.IDEMPOTENT):
        super().__init__("native", effect=effect)
        self.disposition = disposition
        self.reconciliations = []

    async def reconcile_durable_tool_call(self, **kwargs):
        self.reconciliations.append(kwargs)
        return DurableToolRecoveryEvidence(
            self.disposition,
            ToolResult(
                content=f"native {self.disposition}",
                structured={"observation": self.disposition},
                is_error=self.disposition != "confirmed",
            ),
        )


async def seed_round(store, app, provider, tools, *, started):
    session_id = f"pending-round-recovery-{uuid4()}"
    interaction_id = f"interaction-{uuid4()}"
    identity = profiled_session_identity(
        provider_name=provider.name,
        model="fake-model",
        agent_name="assistant",
        tools=tools,
        causal_budget_id=session_id,
        app=app,
    )
    assert identity.execution_profile is not None
    await create_admitted_session(
        store,
        request=RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "recover this round")],
            tool_capability_ceiling=ToolCapabilityCeiling(
                tool_names=tuple(tool.spec.name for tool in tools)
            ),
        ),
        provider_name=provider.name,
        model="fake-model",
        execution_profile=identity.execution_profile,
        interaction_id=interaction_id,
    )
    calls = [
        records.ToolCallRequest(id=f"call-{tool.spec.name}", name=tool.spec.name, arguments={})
        for tool in tools
    ]
    checkpoint, pending = recovery.checkpoint_with_pending_tool_round(
        await store.load_checkpoint(session_id),
        agent_name="assistant",
        environment_name=None,
        task_id=None,
        tool_calls=calls,
        policy_outcomes=[
            records.ToolCallPolicyOutcome(
                call=call,
                result=ToolPolicyResult(decision=ToolPolicyDecision.ALLOW),
                evidence=records.ToolPolicyEvidence.AUTHORITATIVE,
            )
            for call in calls
        ],
        policy_state="planned",
        policy_context_version=1,
        structured_output=None,
        tool_round_identity=new_model_step_identity().new_attempt().new_tool_round(),
    )
    await store.checkpoint(session_id, checkpoint)
    events = []
    for call in calls:
        payload = {
            **pending_rounds.pending_tool_round_identity(pending).payload(),
            "tool_call_id": call.id,
            "idempotency_key": execution.tool_idempotency_key(
                session_id=session_id, tool_round_id=pending.tool_round_id, tool_call_id=call.id
            ),
            "arguments": {},
        }
        if call.name == "known" or started:
            events.append(
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                    agent_name="assistant",
                    tool_name=call.name,
                    payload=payload,
                )
            )
        if call.name == "known":
            events.append(
                Event(
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id=session_id,
                    interaction_id=interaction_id,
                    agent_name="assistant",
                    tool_name=call.name,
                    payload={
                        **payload,
                        "result": ToolResult(content="already recorded").model_dump(),
                    },
                )
            )
    await store.append_events(session_id, events)
    return session_id, pending


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("disposition", ["confirmed", "not_started", "unresolved"])
def test_public_recovery_retains_native_evidence_and_recorded_siblings(
    store_factory, started, disposition
):
    async def scenario():
        async with store_factory(RecoveryStore) as store:
            provider = _TwoCallProvider([])
            native = NativeRecoveryTool(disposition)
            tools = [NeverRunTool("known"), native, NeverRunTool("unknown")]
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=tools)
            session_id, pending = await seed_round(store, app, provider, tools, started=started)
            result = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
            )
            assert result.status is SessionStatus.INTERRUPTED
            assert [tool.calls for tool in tools] == [0, 0, 0]
            assert provider.requests == []
            assert len(native.reconciliations) == 1
            evidence = native.reconciliations[0]
            assert evidence["started"] is started
            assert evidence["tool_call_id"] == "call-native"
            assert evidence["arguments"] == {}
            transcript = await store.load_transcript(session_id)
            [message] = [message for message in transcript if message.role == "tool"]
            assert [part.tool_call_id for part in message.content] == [
                "call-known",
                "call-native",
                "call-unknown",
            ]
            assert message.content[0].content == "already recorded"
            assert message.content[1].content == f"native {disposition}"
            assert message.content[1].is_error is (disposition != "confirmed")
            events = await store.load_events(session_id)
            terminals = [
                event
                for event in events
                if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            ]
            assert len(terminals) == 3
            unknown = next(
                event for event in terminals if event.payload["tool_call_id"] == "call-unknown"
            )
            assert unknown.payload["result"]["structured"]["started"] is started
            assert unknown.payload["result"]["structured"]["outcome_unknown"] is started
            assert (
                await store.load_runtime_publication_receipt(
                    session_id, f"tool-round:{pending.tool_round_id}"
                )
                is not None
            )
            assert (
                pending_round_reader.pending_tool_round_from_checkpoint(
                    await store.load_checkpoint(session_id)
                )
                is None
            )
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
            )
            assert await store.load_events(session_id) == events
            assert len(native.reconciliations) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("started", [False, True])
def test_public_recovery_keeps_external_call_without_journal_pending(store_factory, started):
    async def scenario():
        async with store_factory(RecoveryStore) as store:
            provider = _TwoCallProvider([])
            native = NativeRecoveryTool("confirmed", effect=ToolEffect.EXTERNAL)
            tools = [NeverRunTool("known"), native, NeverRunTool("unknown")]
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=tools)
            session_id, pending = await seed_round(store, app, provider, tools, started=started)

            result = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
            )

            assert result.status is SessionStatus.INTERRUPTED
            assert IncompleteSessionRecoveryAction.PENDING_TOOL_EFFECT in result.actions
            assert [tool.calls for tool in tools] == [0, 0, 0]
            assert native.reconciliations == []
            assert provider.requests == []
            retained = pending_round_reader.pending_tool_round_from_checkpoint(
                await store.load_checkpoint(session_id)
            )
            assert retained is not None
            assert retained.tool_round_id == pending.tool_round_id
            assert not any(
                message.role == "tool" for message in await store.load_transcript(session_id)
            )
            assert (
                await store.load_runtime_publication_receipt(
                    session_id, f"tool-round:{pending.tool_round_id}"
                )
                is None
            )
            terminals = [
                event
                for event in await store.load_events(session_id)
                if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            ]
            assert [event.payload["tool_call_id"] for event in terminals] == ["call-known"]

    asyncio.run(scenario())
