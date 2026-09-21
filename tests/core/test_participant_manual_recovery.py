from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr
from tests.core.test_context_view_admission import _close
from tests.core.test_participant_continuation_boundaries import activate, collect
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_tool_effect_reconciliation_registration import _spec

from cayu import AlwaysRequireApprovalToolPolicy, ToolApprovalDecision, ToolApprovalRequest
from cayu.agents import AgentSpec
from cayu.approvals.tools import ToolApprovalRecoveryRequest
from cayu.approvals.user_input import UserInputRecoveryRequest, UserInputResponse
from cayu.collaboration.lifecycle import ParticipantLifecycleChange
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.sessions.base import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.tools.rounds import ToolRoundRecoveryRequest
from cayu.tools.user_input import UserInputTool


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("round_kind", ["ordinary", "approval", "input"])
@pytest.mark.parametrize("receipt", [False, True])
def test_participant_manual_recovery_requires_explicit_current_authority(
    backend, round_kind, receipt, tmp_path, request, monkeypatch
):
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr("A" * 43)})
    )
    if backend == "memory":

        def factory():
            return InMemorySessionStore(public_authority_alias_codec=codec)
    elif backend == "sqlite":

        def factory():
            return SQLiteSessionStore(
                tmp_path / "recovery.sqlite", public_authority_alias_codec=codec
            )
    else:
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        dsn = request.getfixturevalue("postgres_dsn")

        def factory():
            return PostgresSessionStore(
                dsn, schema_mode=SchemaMode.CREATE, max_size=4, public_authority_alias_codec=codec
            )

    async def run():
        store = factory()
        collaboration = InMemoryCollaborationStore()
        calls = []
        lookups = []
        entered = asyncio.Event()

        class RecordedTool(Tool):
            spec = ToolSpec(
                name="record",
                description="Record one invocation.",
                input_schema={"type": "object", "properties": {}},
                effect=ToolEffect.EXTERNAL if receipt else ToolEffect.NONE,
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="tests:participant-recovery-tool",
                    behavior_version="1",
                    implementation_version="1",
                ),
            )

            async def run(self, ctx, args):
                calls.append(ctx.idempotency_key)
                if receipt:
                    raise RuntimeError("external acknowledgement lost")
                if round_kind == "input":
                    entered.set()
                    await asyncio.Event().wait()
                return ToolResult(content="recorded")

        class Reconciler:
            async def reconcile(self, *, context, receipt):
                lookups.append(context.idempotency_key)
                return ToolEffectReconciliationResult(
                    outcome="completed",
                    observation="sent",
                    receipt=ToolEffectReceipt(
                        receipt_id="verified-result",
                        receipt_schema="deployment",
                        receipt_schema_version=1,
                        tool_call_id=context.tool_call_id,
                        tool_name=context.tool_name,
                        idempotency_key=context.idempotency_key,
                        outcome="completed",
                        message="recorded",
                        source="reconciler",
                        observed_at=datetime(2026, 9, 9, tzinfo=UTC),
                    ),
                )

        first = [ModelStreamEvent.tool_call(id="record-1", name="record", arguments={})]
        if round_kind == "input":
            first.append(
                ModelStreamEvent.tool_call(
                    id="input-1", name="ask_user", arguments={"question": "Proceed?"}
                )
            )
        provider = ScriptedModelProvider(
            [
                [*first, ModelStreamEvent.completed({"finish_reason": "tool_calls"})],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        reg = registration()

        def configured_app():
            value = app(
                collaboration,
                reg,
                session_store=store,
            )
            value.register_provider(provider, default=True)
            value.register_agent(
                AgentSpec(name="reviewer", model="model"),
                tools=[RecordedTool(), UserInputTool()],
                tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"])
                if round_kind == "approval"
                else None,
                tool_effect_reconcilers={
                    "record": ToolEffectReconciliationRegistration(
                        spec=_spec(supports_lookup=True), reconciler=Reconciler()
                    )
                }
                if receipt
                else None,
            )
            return value

        async def interrupt(stream):
            if receipt or round_kind != "input":
                return await collect(stream)
            events = []

            async def consume():
                async for event in stream:
                    events.append(event)

            task = asyncio.create_task(consume())
            await asyncio.wait_for(entered.wait(), 20)
            task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            return events

        # Ordinary/approval recovery: execution succeeds but its terminal write fails.
        # Input recovery: cancel real in-flight work before it stages a terminal result.
        append = store.append_events
        failed = False

        async def fail_terminal(session_id, events):
            nonlocal failed
            if (
                not receipt
                and round_kind != "input"
                and not failed
                and any(
                    event.type == EventType.TOOL_CALL_COMPLETED
                    and event.payload.get("tool_call_id") == "record-1"
                    for event in events
                )
            ):
                failed = True
                raise RuntimeError("terminal write unavailable")
            await append(session_id, events)

        monkeypatch.setattr(store, "append_events", fail_terminal)
        value = configured_app()
        try:
            initialized = await value.initialize_collaboration()
            _, created = await create(value, initialized)
            participant = created.participants[0].reference
            session, execution = await activate(value, participant)
            events = await (interrupt if round_kind == "ordinary" else collect)(
                value.execute_participant_session(
                    execution, participant=participant, context=CONTEXT
                )
            )
            response = None
            approval_id = None
            if round_kind == "approval":
                approval = next(
                    e for e in events if e.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
                )
                approval_id = approval.payload["approval_id"]
                events = await interrupt(
                    value.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=session.id,
                            approval_id=approval_id,
                            tool_round_id=approval.payload["tool_round_id"],
                            tool_call_id=approval.payload["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                        ),
                        context=CONTEXT,
                    )
                )
            elif round_kind == "input":
                awaiting = next(
                    e for e in events if e.type == EventType.SESSION_AWAITING_USER_INPUT
                )
                response = UserInputResponse(
                    session_id=session.id, input_id=awaiting.payload["input_id"], answer="yes"
                )
                events = await interrupt(value.resolve_user_input(response, context=CONTEXT))
            assert len(calls) == 1
            if receipt:
                assert events[-1].type == EventType.SESSION_INTERRUPTED
            started = next(e for e in events if e.type == EventType.TOOL_CALL_STARTED)
            if backend != "memory":
                await store.close()
                store = factory()
            value = configured_app()
            await value.initialize_collaboration()
            if receipt:
                target = await value.inspect_tool_effect(
                    session.id,
                    tool_round_id=started.payload["tool_round_id"],
                    tool_call_id=started.payload["tool_call_id"],
                )
                intent = ToolEffectReconciliationRequest(
                    **target.model_dump(), lookup=True, user_input_response=response
                )
                method = value.reconcile_tool_effect
            else:
                fields = dict(
                    session_id=session.id,
                    tool_call_id=started.payload["tool_call_id"],
                    outcome="completed",
                    message="recorded",
                )
                if round_kind == "approval":
                    intent = ToolApprovalRecoveryRequest(
                        **fields,
                        approval_id=approval_id,
                        tool_round_id=started.payload["tool_round_id"],
                    )
                    method = value.recover_tool_approval
                elif round_kind == "input":
                    intent = UserInputRecoveryRequest(
                        **fields, input_id=response.input_id, answer=response.answer
                    )
                    method = value.recover_user_input
                else:
                    intent = ToolRoundRecoveryRequest(
                        **fields, round_id=started.payload["tool_round_id"]
                    )
                    method = value.recover_tool_round
            before = (
                await store.load(session.id),
                await store.load_checkpoint(session.id),
                await store.load_events(session.id),
            )
            with pytest.raises(PermissionError, match="administration"):
                await collect(method(intent))
            await value.change_participant_lifecycle(
                ParticipantLifecycleChange(
                    operation=initialized.operation("disable"),
                    participant=participant,
                    expected_lifecycle_revision=1,
                    state="disabled",
                ),
                context=CONTEXT,
            )
            with pytest.raises(PermissionError, match="active participants"):
                await collect(method(intent, context=CONTEXT))
            assert before == (
                await store.load(session.id),
                await store.load_checkpoint(session.id),
                await store.load_events(session.id),
            )
            assert not lookups
            assert len(calls) == len(provider.requests) == 1
            await value.change_participant_lifecycle(
                ParticipantLifecycleChange(
                    operation=initialized.operation("reactivate"),
                    participant=participant,
                    expected_lifecycle_revision=2,
                    state="active",
                ),
                context=CONTEXT,
            )
            recovered = await collect(method(intent, context=CONTEXT))
            if recovered[-1].type != EventType.SESSION_COMPLETED:
                pytest.fail(str(recovered[-1].payload))
            assert len(calls) == 1
            assert len(provider.requests) == 2
            durable = await store.load_events(session.id)
            assert (
                sum(
                    e.type == EventType.TOOL_CALL_COMPLETED
                    and e.payload.get("tool_call_id") == "record-1"
                    for e in durable
                )
                == 1
            )
            if receipt:
                assert len(lookups) == 1
                if backend != "memory":
                    await store.close()
                    store = factory()
                    value = configured_app()
                    await value.initialize_collaboration()
                    method = value.reconcile_tool_effect
                await collect(method(intent, context=CONTEXT))
                assert len(lookups) == len(calls) == 1
                assert await store.load_events(session.id) == durable
                with pytest.raises(PermissionError, match="administration"):
                    await collect(method(intent))
            await value.change_participant_lifecycle(
                ParticipantLifecycleChange(
                    operation=initialized.operation("retire"),
                    participant=participant,
                    expected_lifecycle_revision=3,
                    state="retired",
                ),
                context=CONTEXT,
            )
            with pytest.raises(PermissionError, match="active participants"):
                await collect(method(intent, context=CONTEXT))
            assert await store.load_events(session.id) == durable
            assert len(calls) == 1
            assert len(provider.requests) == 2
        finally:
            await _close(store)
            await collaboration.close()

    asyncio.run(run())
